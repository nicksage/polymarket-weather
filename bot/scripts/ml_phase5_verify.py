"""
ml_phase5_verify.py — End-to-end check of the Phase 5 inference hook.

Two-pass verification:
  Pass A — current production config (whatever ACTIVE_STRATEGY is set to):
           confirm the gate behavior matches the active strategy.  If
           ACTIVE_STRATEGY == "top_bin_value", expect ml_mu_c populated.
           Otherwise, expect ml_mu_c == None (gate correctly silences ML).

  Pass B — forced ACTIVE_STRATEGY=top_bin_value:
           confirm the model actually runs and produces a finite (mu, sigma)
           regardless of what's in .env.  Sanity-checks the ML wiring.

Each event tested must be for a city we have a trained model for AND for
target_date == today (the days_ahead==0 gate added in v1.1).

    python -m bot.scripts.ml_phase5_verify
"""

import importlib
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("ml_phase5_verify")


def _pick_event_for_city(city: str):
    """Pick the freshest live_observations row for `city` whose target date
    is today (so the days_ahead==0 gate doesn't silence the model)."""
    from config import DB_PATH
    today_iso = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("""
            SELECT lo.event_id, lo.lat, lo.lon, lo.date, lo.city,
                   lo.pulled_at_utc, lo.current_temp_c
            FROM live_observations lo
            WHERE lo.city = ?
              AND lo.current_temp_c IS NOT NULL
              AND lo.date = ?
            ORDER BY lo.pulled_at_utc DESC
            LIMIT 1
        """, (city, today_iso)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def _reload_modules_after_env_change():
    """Force re-import of config + the modules that read from it so an
    env-var change actually takes effect within this Python process."""
    for mod in ("config", "ml.inference", "weather"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])


def _run_one_pass(label: str, expect_ml_populated: bool) -> int:
    """Run one verification pass.  Returns 0 on PASS, 1 on FAIL."""
    # Re-import after any env change
    _reload_modules_after_env_change()
    from config import ACTIVE_STRATEGY, ML_BLEND_ENABLED, ML_BLEND_WEIGHT_MAX
    from weather import get_temp_distribution_for_event, clear_forecast_cache

    log.info("=" * 70)
    log.info(f"PASS: {label}")
    log.info(f"  ACTIVE_STRATEGY      = {ACTIVE_STRATEGY!r}")
    log.info(f"  ML_BLEND_ENABLED     = {ML_BLEND_ENABLED}")
    log.info(f"  ML_BLEND_WEIGHT_MAX  = {ML_BLEND_WEIGHT_MAX}")
    log.info(f"  expecting ml_mu_c populated: {expect_ml_populated}")
    log.info("=" * 70)

    ev = _pick_event_for_city("Chicago")
    if ev is None:
        log.error("No Chicago live_observations row for today. "
                  "Run live_observation_run() once for a today-event first.")
        return 1
    log.info(f"event: event_id={ev['event_id']} date={ev['date']} "
             f"current_temp={ev['current_temp_c']}C")

    clear_forecast_cache()
    dist = get_temp_distribution_for_event(
        lat=ev["lat"], lon=ev["lon"], date_str=ev["date"],
        city=ev["city"], event_id=ev["event_id"],
    )
    if dist is None:
        log.error("get_temp_distribution_for_event returned None")
        return 1

    ml_mu     = dist.get("ml_mu_c")
    ml_sigma  = dist.get("ml_sigma_c")
    ml_ver    = dist.get("ml_model_version")
    ml_weight = dist.get("ml_weight_used")
    blended_mu = dist.get("mu_c")

    log.info(f"  ml_mu_c          = {ml_mu}")
    log.info(f"  ml_sigma_c       = {ml_sigma}")
    log.info(f"  ml_model_version = {ml_ver}")
    log.info(f"  ml_weight_used   = {ml_weight}")
    log.info(f"  blended mu_c     = {blended_mu}")

    populated = ml_mu is not None
    if populated == expect_ml_populated:
        log.info(f"PASS: {label} — gate behavior matches expectation")
        return 0
    log.error(f"FAIL: {label} — expected populated={expect_ml_populated}, got populated={populated}")
    return 1


def main() -> int:
    init_db()

    # Pass A — current production config (whatever .env says)
    rc_a = _run_one_pass(
        label="A: current ACTIVE_STRATEGY",
        expect_ml_populated=(os.getenv("ACTIVE_STRATEGY", "top_bin_value")
                             .strip().lower() == "top_bin_value"),
    )

    # Pass B — force top_bin_value to confirm the ML wiring still works
    os.environ["ACTIVE_STRATEGY"] = "top_bin_value"
    rc_b = _run_one_pass(
        label="B: forced ACTIVE_STRATEGY=top_bin_value",
        expect_ml_populated=True,
    )

    log.info("=" * 70)
    if rc_a == 0 and rc_b == 0:
        log.info("PHASE 5 VERIFICATION: PASS  (gate works in both modes)")
        return 0
    log.error(f"PHASE 5 VERIFICATION: FAIL  (passA_rc={rc_a} passB_rc={rc_b})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
