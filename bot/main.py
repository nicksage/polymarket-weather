"""
main.py — Entry point and scheduler for the Polymarket weather arbitrage bot.

Three scheduled loops:

  Discovery loop  — every 4 hours at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
                    Queries Polymarket Gamma API for active temperature markets.

  Trading loop    — every hour at :10  (e.g. 00:10, 01:10, 02:10 …)
                    Analyzes events, generates signals, executes trades.
                    Staggered 10 minutes after the top of the hour so that
                    any discovery run at :00 has time to complete first.

  Monitor loop    — every hour at :30  (e.g. 00:30, 01:30, 02:30 …)
                    Cancels unfilled pending orders, detects resolved markets,
                    records realized P&L, and updates unrealized P&L for all
                    open positions.
"""

import logging
import os
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime

from config import LOG_LEVEL, PAPER_TRADE, DB_PATH
from db import init_db
from edge import run_edge_scan
from risk import run_all_checks
from execution import get_clob_client, execute_signal
from sizing import get_bankroll
from monitor import run_monitor_loop
from loops import forecast_pull_run, live_observation_run, retention_run

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

# Silence httpx's raw HTTP request lines — descriptive messages are logged
# at each call site instead.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Cache of the most recently discovered events, shared between discovery and
# trading runs to avoid a redundant Gamma API call within the same hour.
_cached_events: list[dict] | None = None


# ---------------------------------------------------------------------------
# Discovery loop — every 4 hours at :00
# ---------------------------------------------------------------------------

def discovery_run() -> list[dict]:
    """
    Discover all active highest-temperature events from Polymarket.
    Results are cached for reuse by the next trading run.
    """
    global _cached_events
    logger.info("=== DISCOVERY RUN START ===")
    from polymarket import search_temp_high_events
    from config import MIN_LIQUIDITY_USD
    events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)
    _cached_events = events
    logger.info(f"Discovery: found {len(events)} highest-temperature events")
    logger.info("=== DISCOVERY RUN END ===")
    return events


# ---------------------------------------------------------------------------
# Trading loop — every hour at :10
# ---------------------------------------------------------------------------

def trading_run():
    """
    Analyze all discovered events, generate signals, and execute trades that
    pass all risk checks.  Uses the cached event list from the most recent
    discovery run; fetches fresh data if the cache is empty.
    """
    logger.info("=== TRADING RUN START ===")

    # Bias correction data is refreshed by the bias_updater daily job, not
    # here.  (Previously this ran an ERA5-reanalysis-based recorder; that
    # path has been dropped in favour of the VC + Open-Meteo Previous Runs
    # pipeline.  See bias_updater.run_bias_update.)

    bankroll = get_bankroll()
    logger.info(f"Bankroll: ${bankroll:.2f} | Paper trade: {PAPER_TRADE}")

    # Use cached events if available, otherwise fetch fresh
    events = _cached_events
    if not events:
        logger.info("No cached events — running inline discovery")
        events = discovery_run()

    # Analyze events and generate signals
    all_events, signals = run_edge_scan(bankroll=bankroll, events=events)
    logger.info(f"Analyzed {len(all_events)} events -> {len(signals)} raw signals")

    # Sort by EV descending so that when MAX_BIN_BUYS blocks subsequent signals
    # for the same event, the highest-EV signal is always the one that executes.
    signals.sort(key=lambda s: float(s.get("ev") or 0), reverse=True)

    client   = get_clob_client()
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
                f"| status={result['status']} pos_id={result.get('position_id')}"
            )
        else:
            logger.error(
                f"Execution failed for {signal.get('contract_id', '')[:12]}: {result}"
            )

    logger.info(
        f"Trading run complete: {executed} executed, {skipped} skipped by risk rules"
    )
    logger.info("=== TRADING RUN END ===")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("Highest-temperature arbitrage bot starting up")
    logger.info(f"Paper trade mode: {PAPER_TRADE}")
    logger.info(f"Database: {DB_PATH}")

    init_db()

    # Run all loops immediately on startup so the bot is fully current before
    # the scheduler takes over.  Monitor runs last so any positions from the
    # initial trading run are immediately evaluated.
    #
    # Bias update runs first — it ensures forecast_errors is fresh before the
    # trading scan reads per-model bias corrections.
    try:
        from bias_updater import run_bias_update
        run_bias_update()
    except Exception as e:
        logger.warning(f"Startup bias update failed (non-fatal): {e}")

    events = discovery_run()
    _cached_events_ref = events  # seed the module-level cache
    globals()["_cached_events"] = events
    trading_run()
    run_monitor_loop()

    scheduler = BlockingScheduler(timezone="UTC")

    # Discovery: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    scheduler.add_job(
        discovery_run,
        trigger=CronTrigger(hour="0,4,8,12,16,20", minute=0, timezone="UTC"),
        id="discovery_run",
        name="Market discovery",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Trading: every hour at :10
    scheduler.add_job(
        trading_run,
        trigger=CronTrigger(minute=10, timezone="UTC"),
        id="trading_run",
        name="Trading scan",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Monitor: every hour at :30
    scheduler.add_job(
        run_monitor_loop,
        trigger=CronTrigger(minute=30, timezone="UTC"),
        id="monitor_run",
        name="Position monitor",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Bias update: daily at 05:00 UTC.  Fetches the last 14 days of VC
    # observations + Open-Meteo Previous Runs ECMWF/GFS forecasts for every
    # city and rebuilds forecast_errors.  Replaces the previous
    # ERA5-reanalysis-based bias recorder.
    def _scheduled_bias_update() -> None:
        try:
            from bias_updater import run_bias_update
            run_bias_update()
        except Exception as e:
            logger.exception(f"Scheduled bias update failed (non-fatal): {e}")

    scheduler.add_job(
        _scheduled_bias_update,
        trigger=CronTrigger(hour=5, minute=0, timezone="UTC"),
        id="bias_update",
        name="Bias data refresh",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # --- Phase 2: time-versioned forecast + observation loops ---

    def _forecast_pull_job() -> None:
        try:
            forecast_pull_run(events=_cached_events)
        except Exception as e:
            logger.exception(f"Forecast pull run failed: {e}")

    def _live_observation_job() -> None:
        try:
            live_observation_run(events=_cached_events)
        except Exception as e:
            logger.exception(f"Live observation run failed: {e}")

    def _retention_job() -> None:
        try:
            retention_run(older_than_days=90)
        except Exception as e:
            logger.exception(f"Retention run failed: {e}")

    # Forecast pull: every 2 hours at :05 (5 min after discovery at :00)
    scheduler.add_job(
        _forecast_pull_job,
        trigger=CronTrigger(hour="*/2", minute=5, timezone="UTC"),
        id="forecast_pull",
        name="Forecast pull (ECMWF + GFS)",
        misfire_grace_time=600,
        coalesce=True,
    )

    # Live observations: every 20 minutes at :00 / :20 / :40
    scheduler.add_job(
        _live_observation_job,
        trigger=CronTrigger(minute="0,20,40", timezone="UTC"),
        id="live_observation",
        name="Visual Crossing live observation",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Retention: daily at 04:30 UTC (before bias at 05:00)
    scheduler.add_job(
        _retention_job,
        trigger=CronTrigger(hour=4, minute=30, timezone="UTC"),
        id="retention",
        name="Hourly/obs retention purge",
        misfire_grace_time=3600,
        coalesce=True,
    )

    logger.info(
        "Scheduler started | "
        "Discovery: 00/04/08/12/16/20:00 UTC | "
        "Forecast pull: */2h at :05 | "
        "Trading: :10 every hour | "
        "Live obs: :00/:20/:40 | "
        "Monitor: :30 every hour | "
        "Retention: 04:30 UTC daily | "
        "Bias update: 05:00 UTC daily"
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
