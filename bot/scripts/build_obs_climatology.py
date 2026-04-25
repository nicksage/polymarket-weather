"""
build_obs_climatology.py — Aggregate per-(city, doy, hour) climatological
means and stds from VC historical observations.

Reads from the VC disk cache (bot/data/vc_cache).  If a cache miss occurs
during this run, it fetches from VC and writes to cache as a side effect.

Two outputs:
  * obs_climatology_hourly  (city, doy, hour) — hourly variable means/stds
  * obs_climatology_daily   (city, doy)       — daily T_max climatology + percentiles

Used by the v2.0 feature builder (anomaly z-scores) and by the evaluation
script (climatology baseline).

    python -m bot.scripts.build_obs_climatology
    python -m bot.scripts.build_obs_climatology --city Chicago
    python -m bot.scripts.build_obs_climatology --years 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import numpy as np

from visualcrossing import fetch_history_with_hours, vc_cache_stats
from db import (
    init_db, get_ml_backfill_cities, bump_vc_usage,
    upsert_obs_climatology_hourly_bulk, upsert_obs_climatology_daily_bulk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("obs_climatology")


def iter_months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield (y, m)
        m += 1
        if m > 12:
            m = 1; y += 1


def _hour_of_day(dt_str: str | None) -> int | None:
    """VC hourly `datetime` is just 'HH:MM:SS' — parse hour."""
    if not dt_str or len(dt_str) < 2:
        return None
    try:
        return int(dt_str[:2])
    except ValueError:
        return None


def _doy_of(date_str: str) -> int:
    return date.fromisoformat(date_str).timetuple().tm_yday


def build_for_city(
    city: str, lat: float, lon: float, start_date: date, end_date: date,
) -> dict:
    """Walk every cached month for the city, accumulate per-(doy, hour) and
    per-doy lists, then write rows."""
    log.info(f"--- {city} ({lat:.3f},{lon:.3f}) {start_date}..{end_date} ---")

    # (doy, hour) -> dict of variable -> list[float]
    hourly_acc: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # doy -> dict of daily-aggregate -> list[float]
    daily_acc: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    n_months = 0
    n_months_failed = 0
    total_cost = 0
    cache_hits = 0

    for (y, m) in iter_months(start_date, end_date):
        m_first = date(y, m, 1)
        import calendar
        m_last = date(y, m, calendar.monthrange(y, m)[1])
        try:
            resp = fetch_history_with_hours(lat, lon, m_first, m_last, use_cache=True)
        except Exception as e:
            n_months_failed += 1
            log.warning(f"{city} {y}-{m:02d}: fetch failed — {e}")
            continue
        n_months += 1
        if resp.get("from_cache"):
            cache_hits += 1
        cost = int(resp.get("query_cost") or 0)
        total_cost += cost
        bump_vc_usage(cost)

        for d in resp.get("days") or []:
            d_str = d.get("date")
            if not d_str:
                continue
            try:
                doy = _doy_of(d_str)
            except (TypeError, ValueError):
                continue

            # Daily aggregates
            for var, key in (("tmax", "tempmax_c"), ("tmin", "tempmin_c"),
                             ("tmean", "temp_c")):
                v = d.get(key)
                if v is not None:
                    daily_acc[doy][var].append(float(v))

            # Hourly variables
            for h in d.get("hours") or []:
                hr = _hour_of_day(h.get("datetime"))
                if hr is None:
                    continue
                bucket = hourly_acc[(doy, hr)]
                for hourly_key, store_key in (
                    ("temp_c",        "temp"),
                    ("dew_c",         "dew"),
                    ("pressure_hpa",  "pressure"),
                    ("cloudcover",    "cloudcover"),
                    ("windspeed_kph", "windspeed"),
                    ("solarradiation_wm2", "solarradiation"),
                ):
                    v = h.get(hourly_key)
                    if v is not None:
                        bucket[store_key].append(float(v))

    log.info(
        f"  fetched {n_months} months ({cache_hits} cache hits, "
        f"{n_months_failed} failed) cost={total_cost}"
    )

    # ---- Build hourly rows ----
    # Threshold: we WRITE rows even with thin samples so the table is dense;
    # downstream features.py checks n_samples and falls back to NaN when
    # the climatology is too thin to trust (e.g. n_samples < 3).
    fetched_at = datetime.now(timezone.utc).isoformat()
    hourly_rows: list[dict] = []
    for (doy, hr), buckets in hourly_acc.items():
        n = max((len(v) for v in buckets.values()), default=0)
        if n < 1:
            continue
        row = {
            "city":           city,
            "doy":            doy,
            "hour":           hr,
            "n_samples":      n,
            "fetched_at_utc": fetched_at,
        }
        for var in ("temp", "dew", "pressure", "cloudcover", "windspeed", "solarradiation"):
            vals = buckets.get(var) or []
            if len(vals) >= 3:
                row[f"{var}_mu"]    = float(np.mean(vals))
                row[f"{var}_sigma"] = float(np.std(vals))
        hourly_rows.append(row)

    # ---- Build daily rows ----
    daily_rows: list[dict] = []
    for doy, buckets in daily_acc.items():
        tmax_vals = buckets.get("tmax") or []
        if len(tmax_vals) < 3:
            continue
        arr = np.array(tmax_vals, dtype=np.float64)
        row = {
            "city":           city,
            "doy":            doy,
            "n_samples":      len(arr),
            "tmax_mu":        float(np.mean(arr)),
            "tmax_sigma":     float(np.std(arr)),
            "tmax_p10":       float(np.percentile(arr, 10)),
            "tmax_p25":       float(np.percentile(arr, 25)),
            "tmax_p50":       float(np.percentile(arr, 50)),
            "tmax_p75":       float(np.percentile(arr, 75)),
            "tmax_p90":       float(np.percentile(arr, 90)),
            "fetched_at_utc": fetched_at,
        }
        tmin_vals = buckets.get("tmin") or []
        if tmin_vals:
            row["tmin_mu"]    = float(np.mean(tmin_vals))
            row["tmin_sigma"] = float(np.std(tmin_vals))
        tmean_vals = buckets.get("tmean") or []
        if tmean_vals:
            row["tmean_mu"]   = float(np.mean(tmean_vals))
        daily_rows.append(row)

    n_h = upsert_obs_climatology_hourly_bulk(hourly_rows)
    n_d = upsert_obs_climatology_daily_bulk(daily_rows)
    log.info(f"  wrote {n_h} hourly rows, {n_d} daily rows")

    return {
        "city": city,
        "months_fetched": n_months,
        "cache_hits": cache_hits,
        "months_failed": n_months_failed,
        "query_cost": total_cost,
        "hourly_rows": n_h,
        "daily_rows": n_d,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--city", action="append", default=None,
                    help="city name (repeatable); default: all in temp_events")
    ap.add_argument("--years", type=int, default=10,
                    help="lookback years (default 10)")
    ap.add_argument("--end-offset", type=int, default=30)
    args = ap.parse_args()

    if not os.getenv("VISUAL_CROSSING_API_KEY"):
        log.error("VISUAL_CROSSING_API_KEY not set")
        return 1

    init_db()
    log.info(f"VC cache stats: {vc_cache_stats()}")

    # Resolve target cities
    all_rows = get_ml_backfill_cities()
    if args.city:
        wanted = {c.strip().lower() for c in args.city}
        cities = [r for r in all_rows if r["city"].lower() in wanted]
    else:
        cities = all_rows

    today = date.today()
    end_date = today - timedelta(days=args.end_offset)
    start_date = date(end_date.year - args.years, 1, 1)
    log.info(f"Building obs_climatology for {len(cities)} cities, "
             f"window {start_date}..{end_date}")

    totals = {"cities": 0, "hourly_rows": 0, "daily_rows": 0,
              "query_cost": 0, "cache_hits": 0, "months_fetched": 0}
    for r in cities:
        try:
            stat = build_for_city(
                r["city"], r["lat"], r["lon"], start_date, end_date,
            )
            totals["cities"] += 1
            totals["hourly_rows"] += stat["hourly_rows"]
            totals["daily_rows"]  += stat["daily_rows"]
            totals["query_cost"]  += stat["query_cost"]
            totals["cache_hits"]  += stat["cache_hits"]
            totals["months_fetched"] += stat["months_fetched"]
        except Exception as e:
            log.exception(f"{r['city']}: failed — {e}")

    log.info("=" * 70)
    log.info(
        f"OBS CLIMATOLOGY COMPLETE  cities={totals['cities']}  "
        f"hourly_rows={totals['hourly_rows']}  daily_rows={totals['daily_rows']}  "
        f"months_fetched={totals['months_fetched']}  cache_hits={totals['cache_hits']}  "
        f"query_cost={totals['query_cost']}"
    )
    log.info(f"Final VC cache stats: {vc_cache_stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
