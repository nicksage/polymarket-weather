import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from config import LOG_LEVEL, PAPER_TRADE, DB_PATH
from db import init_db
from core_weather.edge import run_edge_scan
from core_weather.edgerisk import run_all_checks
from core_weather.edgeexecution import get_clob_client, execute_signal
from core_weather.edgesizing import get_bankroll

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[        
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log"),
    ],
)
logger = logging.getLogger("main")

def discovery_run():
    """
    Run every 4 hours: scan for new weather markets and add to watchlist.
    """
    logger.info("=== DISCOVERY RUN START ===")
    from bot.polymarket import search_weather_markets
    markets = search_weather_markets()
    logger.info(f"Discovery: found {len(markets)} active weather markets")
    logger.info("=== DISCOVERY RUN END ===")

def trading_run():
    """
    Run every 30 minutes: analyze all markets, execute signals that pass risk checks.
    This is the main loop of the bot.
    """
    logger.info("=== TRADING RUN START ===")

    bankroll = get_bankroll()
    logger.info(f"Current bankroll: ${bankroll:.2f} | Paper trade: {PAPER_TRADE}")

    # Step 1: Generate signals
    signals = run_edge_scan(bankroll=bankroll)
    logger.info(f"Generated {len(signals)} signals")

    # Step 2: Initialize CLOB client (None in paper mode)
    client = get_clob_client()

    # Step 3: Filter by risk rules and execute
    executed = 0
    skipped = 0

    for signal in signals:
        passed, failures = run_all_checks(signal, bankroll)

        if not passed:
            logger.info(
                f"Signal {signal['contract_id'][:12]} skipped: {'; '.join(failures)}"
            )
            skipped += 1
            continue

        result = execute_signal(signal, client=client)

        if result["status"] in ("placed", "paper"):
            executed += 1
            logger.info(
                f"Executed: {signal['recommended_side']} ${signal['kelly_size']:.2f} "
                f"on {signal['contract_id'][:12]} | status={result['status']}"
            )
        else:
            logger.error(f"Execution failed for {signal['contract_id'][:12]}: {result}")

    logger.info(f"Trading run complete: {executed} executed, {skipped} skipped by risk rules")
    logger.info("=== TRADING RUN END ===")

def main():
    """Start the bot and scheduling loop."""
    logger.info("Weather arbitrage bot starting up")
    logger.info(f"Paper trade mode: {PAPER_TRADE}")
    logger.info(f"Database: {DB_PATH}")

    # Initialize database
    init_db()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Run once immediately on startup
    discovery_run()
    trading_run()

    # Schedule recurring runs
    scheduler = BlockingScheduler()

    # Market analysis every 30 minutes
    scheduler.add_job(
        trading_run,
        trigger=IntervalTrigger(minutes=30),
        id="trading_run",
        name="Trading scan",
        misfire_grace_time=300,  # Allow up to 5 min late start
    )

   # Market discovery every 4 hours
    scheduler.add_job(
        discovery_run,
        trigger=IntervalTrigger(hours=4),
        id="discovery_run",
        name="Market discovery",
        misfire_grace_time=600,  # Allow up to 10 min late start
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