"""
bias_updater.py — Incremental refresh of the bias-correction data pipeline.

Runs on a daily schedule (and once at bot startup).  For every city in
CITY_COORDS it:

  1. Fetches the last N days of daily-max observations from Visual Crossing
     → upsert into `historical_observed_daily`.
  2. Fetches the last N days of ECMWF + GFS historical forecasts (day-3
     lead) from Open-Meteo Previous Runs
     → upsert into `historical_forecasts_previous_runs`.
  3. Rebuilds that city's rows in `forecast_errors` by joining the two
     caches — idempotent via delete-then-insert.

Because the underlying cache tables retain the full multi-year history
populated by the one-shot backfill, the rebuild step still produces the
complete forecast_errors history for each city even though the incremental
fetch only touched the most recent N days.

This replaces the old ERA5-reanalysis-based path (bias.record_resolved_contracts).
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from db import (
    upsert_historical_observed_daily,
    upsert_historical_forecasts_previous_runs,
    rebuild_forecast_errors_from_cache,
    _loc_keys,
)
from polymarket import CITY_COORDS
from visualcrossing import fetch_daily_history
from openmeteo_previous_runs import fetch_historical_forecasts

logger = logging.getLogger(__name__)

# Window of days to re-fetch each run.  14 gives plenty of redundancy if a
# run is skipped and also covers the ~5-day ERA5-style publication lag that
# historical forecasts sometimes have for the most recent dates.
DEFAULT_LOOKBACK_DAYS = 14

# Canonical Open-Meteo model ids recorded by the backfill.
MODELS = ("ecmwf_ifs025", "gfs_global")

# Fixed lead time for bias comparison.  Should match the trading horizon
# (MAX_TRADE_DAYS).  Lead=1 means "how accurate was the model's forecast
# issued 1 day before the target date" — matching D+1 trades.
LEAD_DAYS = 1


def _dedupe_cities() -> list[tuple[str, float, float]]:
    """Same dedup logic as the backfill script: collapse duplicate coords."""
    by_key: dict[tuple[float, float], list[str]] = {}
    for name, (lat, lon) in CITY_COORDS.items():
        by_key.setdefault(_loc_keys(lat, lon), []).append(name)
    out: list[tuple[str, float, float]] = []
    for (lat_k, lon_k), names in by_key.items():
        canonical = sorted(names, key=lambda s: (len(s), s))[0]
        lat, lon = CITY_COORDS[names[0]]
        out.append((canonical, lat, lon))
    return sorted(out, key=lambda r: r[0])


def run_bias_update(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    sleep_between: float = 0.1,
) -> dict:
    """
    One pass over every unique city.  Fetches recent VC + OM data, upserts,
    rebuilds forecast_errors.  Safe to re-run — all writes are idempotent.

    Returns a summary dict:
        {
          "cities":       int,    # processed
          "failed":       int,    # cities that errored (per-city, non-fatal)
          "obs_rows":     int,    # total VC rows upserted
          "fcst_rows":    int,    # total OM rows upserted
          "err_rows":     int,    # total forecast_errors rows rebuilt
          "elapsed_s":    float,
        }
    """
    t0     = time.time()
    today  = date.today()
    start  = today - timedelta(days=lookback_days)
    end    = today - timedelta(days=1)   # yesterday is last fully-resolved day
    cities = _dedupe_cities()

    logger.info(
        f"[BIAS UPDATE] starting — {len(cities)} cities, window {start}..{end} "
        f"({lookback_days}d lookback, lead={LEAD_DAYS}d)"
    )

    totals = {"obs_rows": 0, "fcst_rows": 0, "err_rows": 0, "failed": 0}

    for idx, (city, lat, lon) in enumerate(cities, start=1):
        try:
            # VC observations
            obs = fetch_daily_history(lat, lon, start, end, observed_only=True)
            totals["obs_rows"] += upsert_historical_observed_daily(lat, lon, city, obs)

            # Open-Meteo Previous Runs (both models in one call)
            fcst = fetch_historical_forecasts(
                lat, lon, models=list(MODELS), lead_days=LEAD_DAYS,
                past_days=lookback_days,
            )
            totals["fcst_rows"] += upsert_historical_forecasts_previous_runs(
                lat, lon, city, fcst
            )

            # Rebuild forecast_errors from the full cache (not just the window)
            totals["err_rows"] += rebuild_forecast_errors_from_cache(
                lat, lon, city, lead_days=LEAD_DAYS,
            )

            logger.debug(f"[BIAS UPDATE] [{idx}/{len(cities)}] {city} ok")

        except Exception as e:
            totals["failed"] += 1
            logger.warning(f"[BIAS UPDATE] [{idx}/{len(cities)}] {city} failed: {e}")

        if sleep_between > 0:
            time.sleep(sleep_between)

    # Rebuild per-city forecast accuracy metrics after errors are updated
    try:
        from db import rebuild_city_forecast_accuracy
        n_accuracy = rebuild_city_forecast_accuracy(window_days=30)
        logger.info(f"[BIAS UPDATE] city_forecast_accuracy rebuilt for {n_accuracy} cities")
    except Exception as e:
        n_accuracy = 0
        logger.warning(f"[BIAS UPDATE] city_forecast_accuracy rebuild failed: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"[BIAS UPDATE] complete — cities={len(cities)} failed={totals['failed']} "
        f"obs={totals['obs_rows']} fcst={totals['fcst_rows']} err_rebuilt={totals['err_rows']} "
        f"accuracy_cities={n_accuracy} elapsed={elapsed:.1f}s"
    )

    return {
        "cities":    len(cities),
        "failed":    totals["failed"],
        "obs_rows":  totals["obs_rows"],
        "fcst_rows": totals["fcst_rows"],
        "err_rows":  totals["err_rows"],
        "accuracy_cities": n_accuracy,
        "elapsed_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    # Allow running manually: `python -m bot.bias_updater`
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run_bias_update()
