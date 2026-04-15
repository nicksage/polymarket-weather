"""
backfill_bias_data.py — One-shot (and re-runnable) backfill of historical
weather data used for per-(city, month, model) bias correction.

For every city in polymarket.CITY_COORDS (deduplicated by coordinate), this
script:

  1. Fetches N days of daily-max observations from Visual Crossing
     → stored in `historical_observed_daily`.
  2. Fetches N days of ECMWF + GFS historical forecasts (at a fixed lead
     time, default day-3) from Open-Meteo Previous Runs
     → stored in `historical_forecasts_previous_runs`.
  3. Joins the two caches per-city and writes one row per (date, model) into
     `forecast_errors`, overwriting any previous backfill rows for that
     (location, lead) key (idempotent).

Usage
-----
    # Stop the bot first so no concurrent writes happen.
    cd bot
    python scripts/backfill_bias_data.py                        # full default backfill
    python scripts/backfill_bias_data.py --days 365 --lead 3    # shorter window
    python scripts/backfill_bias_data.py --cities amsterdam,chicago
    python scripts/backfill_bias_data.py --models ecmwf_ifs025  # ECMWF only
    python scripts/backfill_bias_data.py --dry-run              # plan only

Safe to re-run any time; results for unchanged dates are idempotent.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta

from dotenv import load_dotenv

# Load .env before any module that reads env vars.
load_dotenv()

# Make sibling modules importable regardless of launch directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT  = os.path.dirname(_HERE)
sys.path.insert(0, _BOT)

from db import (                                                       # noqa: E402
    init_db,
    upsert_historical_observed_daily,
    upsert_historical_forecasts_previous_runs,
    rebuild_forecast_errors_from_cache,
    _loc_keys,
)
from polymarket import CITY_COORDS                                     # noqa: E402
from visualcrossing import fetch_daily_history                         # noqa: E402
from openmeteo_previous_runs import fetch_historical_forecasts         # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
)
# The httpx and tenacity loggers are noisy at INFO — bump them to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("backfill")


# ---------------------------------------------------------------------------
# De-duplicate CITY_COORDS (e.g. "new york" and "nyc" share coords)
# ---------------------------------------------------------------------------
def dedupe_cities() -> list[tuple[str, float, float]]:
    """
    Return [(canonical_name, lat, lon), ...] with duplicate coordinates
    collapsed.  Canonical name = the shortest/earliest-alphabetical name
    for that coordinate tuple.
    """
    by_key: dict[tuple[float, float], list[str]] = {}
    for name, (lat, lon) in CITY_COORDS.items():
        by_key.setdefault(_loc_keys(lat, lon), []).append(name)

    out: list[tuple[str, float, float]] = []
    for (lat_k, lon_k), names in sorted(by_key.items()):
        canonical = sorted(names, key=lambda s: (len(s), s))[0]
        # Use the first matching city's exact lat/lon from CITY_COORDS
        lat, lon = CITY_COORDS[names[0]]
        out.append((canonical, lat, lon))
    return sorted(out, key=lambda r: r[0])


# ---------------------------------------------------------------------------
# Per-city backfill
# ---------------------------------------------------------------------------
def backfill_city(
    city: str, lat: float, lon: float,
    days: int, lead_days: int, models: list[str],
    fetch_obs: bool, fetch_fcst: bool,
) -> dict:
    """
    Returns a dict with counts and any error messages.
    """
    result = {
        "city":       city,
        "obs_rows":   0,
        "fcst_rows":  0,
        "err_rows":   0,
        "errors":     [],
    }

    today = date.today()
    start = today - timedelta(days=days)
    end   = today - timedelta(days=1)   # yesterday is the last fully-resolved day

    # --- Visual Crossing observations ---
    if fetch_obs:
        try:
            obs = fetch_daily_history(lat, lon, start, end, observed_only=True)
            n = upsert_historical_observed_daily(lat, lon, city, obs)
            result["obs_rows"] = n
        except Exception as e:
            result["errors"].append(f"VC fetch failed: {e}")
            logger.exception(f"[{city}] VC fetch failed")

    # --- Open-Meteo Previous Runs historical forecasts ---
    if fetch_fcst:
        try:
            fcst = fetch_historical_forecasts(
                lat, lon, models=models, lead_days=lead_days, past_days=days,
            )
            n = upsert_historical_forecasts_previous_runs(lat, lon, city, fcst)
            result["fcst_rows"] = n
        except Exception as e:
            result["errors"].append(f"OM fetch failed: {e}")
            logger.exception(f"[{city}] OM fetch failed")

    # --- Rebuild forecast_errors from the two caches ---
    try:
        n = rebuild_forecast_errors_from_cache(lat, lon, city, lead_days=lead_days)
        result["err_rows"] = n
    except Exception as e:
        result["errors"].append(f"rebuild failed: {e}")
        logger.exception(f"[{city}] rebuild failed")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=900,
                        help="Days of history to fetch (default 900 ≈ 2.5 years)")
    parser.add_argument("--lead", type=int, default=3,
                        help="Forecast lead time in days (1..7, default 3)")
    parser.add_argument("--cities", type=str, default="",
                        help="Comma-separated subset of city names to backfill "
                             "(default: all in CITY_COORDS)")
    parser.add_argument("--models", type=str, default="ecmwf_ifs025,gfs_global",
                        help="Comma-separated Open-Meteo model ids")
    parser.add_argument("--skip-obs", action="store_true",
                        help="Skip VC observation fetch (use existing cache)")
    parser.add_argument("--skip-fcst", action="store_true",
                        help="Skip Open-Meteo forecast fetch (use existing cache)")
    parser.add_argument("--sleep", type=float, default=0.2,
                        help="Seconds to sleep between cities (default 0.2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without fetching")
    args = parser.parse_args()

    init_db()

    all_cities = dedupe_cities()
    if args.cities:
        want = {c.strip().lower() for c in args.cities.split(",") if c.strip()}
        all_cities = [c for c in all_cities if c[0].lower() in want]

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    logger.info("=" * 72)
    logger.info(f"Backfill plan: {len(all_cities)} cities, "
                f"{args.days} days, lead={args.lead}d, models={models}")
    logger.info(f"Fetch VC obs:   {not args.skip_obs}")
    logger.info(f"Fetch OM fcst:  {not args.skip_fcst}")
    logger.info("=" * 72)

    if args.dry_run:
        for city, lat, lon in all_cities:
            logger.info(f"  would fetch: {city:20s} ({lat:+.3f}, {lon:+.3f})")
        return 0

    t0 = time.time()
    totals = {"obs_rows": 0, "fcst_rows": 0, "err_rows": 0, "failed_cities": 0}
    per_city_results: list[dict] = []

    for idx, (city, lat, lon) in enumerate(all_cities, start=1):
        logger.info(f"[{idx:2d}/{len(all_cities)}] {city} ({lat:+.3f}, {lon:+.3f})")
        res = backfill_city(
            city, lat, lon,
            days=args.days, lead_days=args.lead, models=models,
            fetch_obs=not args.skip_obs, fetch_fcst=not args.skip_fcst,
        )
        per_city_results.append(res)
        totals["obs_rows"]  += res["obs_rows"]
        totals["fcst_rows"] += res["fcst_rows"]
        totals["err_rows"]  += res["err_rows"]
        if res["errors"]:
            totals["failed_cities"] += 1

        logger.info(
            f"          obs={res['obs_rows']:4d} "
            f"fcst={res['fcst_rows']:4d} "
            f"err_pairs={res['err_rows']:4d} "
            + (f"ERRORS={res['errors']}" if res["errors"] else "")
        )
        time.sleep(args.sleep)

    elapsed = time.time() - t0
    logger.info("=" * 72)
    logger.info(f"Backfill complete in {elapsed:.1f}s")
    logger.info(f"  cities processed: {len(all_cities)}")
    logger.info(f"  cities with errors: {totals['failed_cities']}")
    logger.info(f"  VC observation rows upserted:   {totals['obs_rows']:,}")
    logger.info(f"  OM forecast rows upserted:      {totals['fcst_rows']:,}")
    logger.info(f"  forecast_errors rows rebuilt:   {totals['err_rows']:,}")
    logger.info("=" * 72)

    if totals["failed_cities"] > 0:
        logger.warning("Cities with errors:")
        for r in per_city_results:
            if r["errors"]:
                logger.warning(f"  {r['city']:20s} → {r['errors']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
