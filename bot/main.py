import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from config import LOG_LEVEL, PAPER_TRADE, DB_PATH
from db import init_db
from edge import run_edge_scan
from risk import run_all_checks
from execution import get_clob_client, execute_signal
from sizing import get_bankroll

# Ensure logs directory exists before FileHandler is created
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log"),
    ],
)
logger = logging.getLogger("main")

# Silence httpx's raw "HTTP Request: GET ..." lines — we log descriptive
# plain-English messages at each call site instead.
logging.getLogger("httpx").setLevel(logging.WARNING)


def discovery_run() -> list[dict]:
    """
    Run every 4 hours: discover all active highest-temperature events.
    Returns the event list so trading_run can reuse it without a second API call.
    """
    logger.info("=== DISCOVERY RUN START ===")
    from polymarket import search_temp_high_events
    from config import MIN_LIQUIDITY_USD
    events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)
    logger.info(f"Discovery: found {len(events)} highest-temperature events")
    logger.info("=== DISCOVERY RUN END ===")
    return events


def trading_run(events: list[dict] | None = None):
    """
    Run every 30 minutes: analyze all events, execute signals that pass risk checks.

    Accepts a pre-fetched events list to avoid a redundant Gamma API call.
    Both the full event analysis and the filtered signals list are returned
    from run_edge_scan(); signals are passed through risk.py → execution.py.
    """
    logger.info("=== TRADING RUN START ===")

    # Record bias data for any resolved contracts whose ERA5 actuals are now
    # available (target_date <= today - 5 days).  Runs cheaply — most calls
    # return 0 new records after the first week of operation.
    try:
        from bias import record_resolved_contracts, log_bias_summary
        new_records = record_resolved_contracts()
        if new_records > 0:
            log_bias_summary()
    except Exception as e:
        logger.warning(f"Bias recorder failed (non-fatal): {e}")

    bankroll = get_bankroll()
    logger.info(f"Bankroll: ${bankroll:.2f} | Paper trade: {PAPER_TRADE}")

    # Step 1: Analyze all events, generate signals
    all_events, signals = run_edge_scan(bankroll=bankroll, events=events)
    logger.info(f"Analyzed {len(all_events)} events → {len(signals)} raw signals")

    # Step 2: Initialize CLOB client (None in paper mode)
    client = get_clob_client()

    # Step 3: Risk filter and execute
    executed = 0
    skipped  = 0

    for signal in signals:
        passed, failures = run_all_checks(signal, bankroll)

        if not passed:
            logger.info(
                f"Signal skipped [{signal.get('city')} {signal.get('date')} "
                f"{signal.get('question', '')[:30]}]: {'; '.join(failures)}"
            )
            skipped += 1
            continue

        result = execute_signal(signal, client=client)

        if result["status"] in ("placed", "paper"):
            executed += 1
            logger.info(
                f"Executed: {signal['recommended_side']} ${signal['kelly_size']:.2f} "
                f"on {signal['contract_id'][:12]} "
                f"[{signal.get('city')} {signal.get('question', '')[:20]}] "
                f"| status={result['status']}"
            )
        else:
            logger.error(
                f"Execution failed for {signal.get('contract_id', '')[:12]}: {result}"
            )

    logger.info(
        f"Trading run complete: {executed} executed, {skipped} skipped by risk rules"
    )
    logger.info("=== TRADING RUN END ===")


def main():
    logger.info("Highest-temperature arbitrage bot starting up")
    logger.info(f"Paper trade mode: {PAPER_TRADE}")
    logger.info(f"Database: {DB_PATH}")

    init_db()

    # Run once immediately on startup
    events = discovery_run()
    trading_run(events=events)

    scheduler = BlockingScheduler()

    # Market analysis every 30 minutes (reuses cached discovery results)
    scheduler.add_job(
        trading_run,
        trigger=IntervalTrigger(minutes=30),
        id="trading_run",
        name="Trading scan",
        misfire_grace_time=300,
    )

    # Fresh market discovery every 4 hours (new events may appear)
    def _discovery_then_trade():
        fresh_events = discovery_run()
        trading_run(events=fresh_events)

    scheduler.add_job(
        _discovery_then_trade,
        trigger=IntervalTrigger(hours=4),
        id="discovery_run",
        name="Market discovery + trade",
        misfire_grace_time=600,
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        raise


if __name__ == "__main__":
    main()
