"""
predictor_runner.py — Standalone runner for ONLY the intraday bin
predictor (scheduled_predictor.py).  No main.py, no top_k_hedged,
no WebSocket — just APScheduler ticking the predictor every N minutes
per PREDICTOR_SCAN_MIN in .env.

Use this when you want the intraday predictor to be your ONLY active
strategy.  Run as a systemd service via deploy/setup_systemd.sh (which
points ExecStart at this file when ACTIVE_STRATEGY=intraday_only).

Logs land in journalctl with the same identifier as the unit.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

# Reconfigure stdout/stderr for UTF-8 — match scripts/close_all_positions.py
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Load .env via python-dotenv (NOT systemd's EnvironmentFile=, which has
# inline-comment parsing bugs).  Must run before importing config.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
)
log = logging.getLogger("predictor_runner")


def main() -> int:
    log.info("=" * 60)
    log.info("  Intraday predictor runner starting")
    log.info(f"  mode = {os.getenv('PREDICTOR_MODE', 'paper')}")
    log.info(f"  scan interval = {os.getenv('PREDICTOR_SCAN_MIN', '5')} min")
    log.info(f"  min_edge = {os.getenv('PREDICTOR_MIN_EDGE', '0.10')}")
    log.info(f"  max_bins_per_event = {os.getenv('PREDICTOR_MAX_BINS_PER_EVENT', '1')}")
    log.info(f"  flat_stake_usd = ${os.getenv('PREDICTOR_FLAT_STAKE_USD', '5')}")
    log.info("=" * 60)

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        log.error("apscheduler not installed.  "
                  "Run: pip install apscheduler in your venv.")
        return 1

    from scheduled_predictor import register_predictor_jobs, run_intraday_scan

    scheduler = BlockingScheduler(timezone="UTC")
    register_predictor_jobs(scheduler)

    # Graceful shutdown on SIGTERM (systemd stop)
    def _shutdown(signum, frame):
        log.info(f"Received signal {signum} — shutting down scheduler")
        scheduler.shutdown(wait=False)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # Run one scan immediately on startup so we don't have to wait for the
    # first cron tick (which could be up to PREDICTOR_SCAN_MIN minutes away).
    log.info("Running initial scan...")
    try:
        run_intraday_scan()
    except Exception as e:
        log.exception(f"initial scan failed (non-fatal): {e}")

    log.info("Starting scheduler — Ctrl-C or SIGTERM to stop")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())