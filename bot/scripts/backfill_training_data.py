"""
backfill_training_data.py — Populate ml_training_rows from VC history.

For every city in temp_events, walks the date range
[today - (lookback_years + 1y warmup), today - 30d] one month at a time,
calling fetch_history_with_hours() and assembling (feature vector, T_max)
training rows at two decision moments per day (10:00 and 12:00 local).

Design
------
* One VC call per (city, month) — returns ~720 hourly + ~30 daily records.
* Keeps previous month's hours cached so day-1 of each month can still
  compute 24h-lookback features.
* Keeps a rolling daily t_max cache so lookups for D-1, D-7, D-365 hit
  memory rather than re-calling VC.
* Pre-fetches 12 months of warmup so the earliest target date has all
  year-over-year features populated.
* Resumable: skips rows already present in ml_training_rows.
* Writes VC queryCost to the same daily counter the live loop uses.

Usage
-----
    python -m bot.scripts.backfill_training_data
    python -m bot.scripts.backfill_training_data --city Chicago --years 5
    python -m bot.scripts.backfill_training_data --dry-run

Flags
-----
    --city       one or more cities (repeatable); default: all in temp_events
    --years      lookback years (default 10)
    --end-offset days to back off from today (default 30)
    --dry-run    assemble rows but don't insert
"""

import argparse
import calendar
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import numpy as np
from zoneinfo import ZoneInfo

from visualcrossing import fetch_history_with_hours
from db import (
    init_db,
    get_ml_backfill_cities,
    ml_training_row_exists,
    insert_ml_training_row,
    count_ml_training_rows,
    bump_vc_usage,
    load_all_obs_climatology_hourly,
    load_all_obs_climatology_daily,
)
from ml.features import build_feature_vector
from ml.schema import FEATURE_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("ml_backfill")

DECISION_HOURS_LOCAL = (10, 12, 14)   # v2.0: added 14:00 for late-day inference
LOOKBACK_MATCH_HOURS = 0.5   # an obs within ±30 min of target counts


# ---------------------------------------------------------------------------
# City + tz resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CityTarget:
    city: str
    lat: float
    lon: float
    tz: ZoneInfo

    @property
    def lat_key(self) -> float:
        return round(self.lat, 2)

    @property
    def lon_key(self) -> float:
        return round(self.lon, 2)


def _resolve_tz(lat: float, lon: float) -> ZoneInfo:
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    name = tf.timezone_at(lat=lat, lng=lon)
    if not name:
        raise RuntimeError(f"could not resolve timezone for ({lat},{lon})")
    return ZoneInfo(name)


def resolve_cities(cities_filter: list[str] | None) -> list[CityTarget]:
    rows = get_ml_backfill_cities()
    if cities_filter:
        wanted = {c.strip().lower() for c in cities_filter}
        rows = [r for r in rows if r["city"].lower() in wanted]
    out: list[CityTarget] = []
    for r in rows:
        try:
            tz = _resolve_tz(r["lat"], r["lon"])
        except Exception as e:
            log.warning(f"skipping {r['city']}: tz resolution failed ({e})")
            continue
        out.append(CityTarget(city=r["city"], lat=r["lat"], lon=r["lon"], tz=tz))
    return out


# ---------------------------------------------------------------------------
# Month iteration helpers
# ---------------------------------------------------------------------------

def iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    """Yield (year, month) pairs from start through end inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield (y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


def month_range(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    return first, date(year, month, last_day)


# ---------------------------------------------------------------------------
# Hour-match helper
# ---------------------------------------------------------------------------

def _find_hour_row(hours: list[dict], target_utc_epoch: float) -> dict | None:
    """Nearest hour row within ±30 min of the target UTC epoch, else None."""
    tol = LOOKBACK_MATCH_HOURS * 3600
    best = None
    best_delta = float("inf")
    for h in hours:
        epoch = h.get("datetime_epoch")
        if epoch is None:
            continue
        delta = abs(float(epoch) - target_utc_epoch)
        if delta <= tol and delta < best_delta:
            best = h
            best_delta = delta
    return best


def _slice_recent_obs(
    all_hours: list[dict], decision_dt_utc: datetime, lookback_hours: float = 25.0
) -> list[dict]:
    """Return hours within [decision - lookback, decision].  Caller-friendly;
    build_feature_vector applies its own ±30min window on top."""
    lower = (decision_dt_utc - timedelta(hours=lookback_hours)).timestamp()
    upper = decision_dt_utc.timestamp()
    out = []
    for h in all_hours:
        epoch = h.get("datetime_epoch")
        if epoch is None:
            continue
        if lower <= float(epoch) <= upper:
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Core assembly — one training row
# ---------------------------------------------------------------------------

def assemble_training_row(
    target: CityTarget,
    target_date: date,
    decision_hour_local: int,
    t_max_label: float,
    hours_this_day: list[dict],
    hours_prev_day: list[dict],
    daily_cache: dict[tuple[str, str], float],
    hourly_clim: dict[tuple[int, int], dict] | None = None,
    daily_clim:  dict[int, dict] | None = None,
) -> tuple[np.ndarray, str] | None:
    """Build the v2.0 feature vector + fetched_at_utc stamp for one row, or
    None if the hour row at decision time is missing (can't anchor the
    current_obs).  NaN propagates through for any individual missing field."""
    # Decision moment (tz-aware in the city's local timezone)
    decision_dt_local = datetime(
        target_date.year, target_date.month, target_date.day,
        decision_hour_local, 0, tzinfo=target.tz,
    )
    decision_dt_utc = decision_dt_local.astimezone(timezone.utc)

    # Current obs — must exist for the row to be useful
    curr_row = _find_hour_row(hours_this_day, decision_dt_utc.timestamp())
    if curr_row is None:
        return None

    current_obs = {
        "temp_c":             curr_row.get("temp_c"),
        "humidity":           curr_row.get("humidity"),
        "dew_c":              curr_row.get("dew_c"),
        "pressure_hpa":       curr_row.get("pressure_hpa"),
        "cloudcover":         curr_row.get("cloudcover"),
        "visibility_km":      curr_row.get("visibility_km"),
        "windspeed_kph":      curr_row.get("windspeed_kph"),
        "winddir_deg":        curr_row.get("winddir_deg"),
        "solarradiation_wm2": curr_row.get("solarradiation_wm2"),
        "snowdepth_cm":       curr_row.get("snowdepth_cm"),
    }

    # Recent obs — union of this day + prior day, filtered by time window
    combined = hours_prev_day + hours_this_day
    recent_obs = _slice_recent_obs(combined, decision_dt_utc, lookback_hours=25.0)

    # Historical T_max lookups (D-1, D-7, D-365)
    def _lookup_days_ago(n: int) -> float | None:
        d = (target_date - timedelta(days=n)).isoformat()
        return daily_cache.get((target.city, d))

    historical_tmax = {
        "yesterday_c":      _lookup_days_ago(1),
        "seven_days_ago_c": _lookup_days_ago(7),
        "last_year_c":      _lookup_days_ago(365),
    }

    # v2.0 — climatology lookups
    doy = target_date.timetuple().tm_yday
    daily_clim_row = (daily_clim or {}).get(doy)
    climatology = {"mu_today_c": (daily_clim_row or {}).get("tmax_mu")}

    # tz_offset_hours for the decision moment (handles DST automatically)
    tz_offset_hours = decision_dt_local.utcoffset().total_seconds() / 3600.0 \
        if decision_dt_local.utcoffset() is not None else 0.0

    ctx = {
        # v1.0
        "target_date":       target_date,
        "decision_dt_utc":   decision_dt_utc,
        "decision_dt_local": decision_dt_local,
        "current_obs":       current_obs,
        "recent_obs":        recent_obs,
        "climatology":       climatology,
        "historical_tmax":   historical_tmax,
        # v2.0
        "city_name":          target.city,
        "lat":                target.lat,
        "lon":                target.lon,
        "tz_offset_hours":    tz_offset_hours,
        "hourly_climatology": hourly_clim or {},
    }
    vec = build_feature_vector(ctx)
    return vec, datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-city driver
# ---------------------------------------------------------------------------

def backfill_city(
    target: CityTarget,
    start_date: date,
    end_date: date,
    warmup_years: int = 1,
    dry_run: bool = False,
) -> dict:
    """Fill ml_training_rows for one city over [start_date, end_date].
    Pre-fetches `warmup_years` before start_date to populate year-over-year
    lookups.  Returns a counts dict."""
    warmup_start = date(start_date.year - warmup_years, start_date.month, 1)
    log.info(
        f"--- {target.city} ({target.lat:.3f},{target.lon:.3f}) tz={target.tz.key} "
        f"window={start_date}..{end_date} warmup_from={warmup_start}"
    )

    counts = {
        "months_fetched":  0,
        "months_failed":   0,
        "days_processed":  0,
        "rows_inserted":   0,
        "rows_skipped":    0,
        "rows_no_current": 0,
        "rows_no_label":   0,
        "query_cost":      0,
        "cache_hits":      0,
    }

    # v2.0 — pre-load city's climatology tables once (one DB read per city
    # vs N reads per row).
    hourly_clim = load_all_obs_climatology_hourly(target.city)
    daily_clim  = load_all_obs_climatology_daily(target.city)
    log.info(
        f"  loaded climatology: {len(hourly_clim)} hourly rows, "
        f"{len(daily_clim)} daily rows"
    )

    daily_cache: dict[tuple[str, str], float] = {}
    prev_month_hours: list[dict] = []

    for (y, m) in iter_months(warmup_start, end_date):
        month_first, month_last = month_range(y, m)
        try:
            resp = fetch_history_with_hours(
                target.lat, target.lon, month_first, month_last
            )
        except Exception as e:
            counts["months_failed"] += 1
            log.error(f"{target.city} {y}-{m:02d}: fetch failed — {e}")
            prev_month_hours = []
            continue

        counts["months_fetched"] += 1
        if resp.get("from_cache"):
            counts["cache_hits"] += 1
        cost = int(resp.get("query_cost") or 0)
        counts["query_cost"] += cost
        bump_vc_usage(cost)

        # Flatten the month's hours into a single list (prev-day context)
        curr_month_hours: list[dict] = []
        for d in resp["days"]:
            curr_month_hours.extend(d["hours"])
            if d.get("tempmax_c") is not None:
                daily_cache[(target.city, d["date"])] = float(d["tempmax_c"])

        # Don't emit training rows for warmup period
        if (y, m) < (start_date.year, start_date.month) or date(y, m, 1) < start_date.replace(day=1):
            prev_month_hours = curr_month_hours
            continue

        # Iterate days within this month
        for d in resp["days"]:
            try:
                td = date.fromisoformat(d["date"])
            except (TypeError, ValueError):
                continue
            if td < start_date or td > end_date:
                continue

            counts["days_processed"] += 1
            tmax_label = d.get("tempmax_c")
            if tmax_label is None:
                counts["rows_no_label"] += 1
                continue

            # VC groups hours under their containing day already — every
            # entry in d["hours"] belongs to d["date"] in local time.  The
            # hour `datetime` field is just "HH:MM:SS" so we trust the
            # grouping rather than reparsing.
            hours_this_day = d["hours"]
            # Prior-day hours = previous month (if any) + earlier days of
            # this month.  Both feed build_feature_vector, which filters by
            # absolute UTC epoch anyway.
            hours_prev_day = prev_month_hours + [
                h for od in resp["days"] if od["date"] < d["date"] for h in od["hours"]
            ]

            for decision_hour in DECISION_HOURS_LOCAL:
                if ml_training_row_exists(
                    target.city, d["date"], decision_hour, FEATURE_VERSION
                ):
                    counts["rows_skipped"] += 1
                    continue

                row = assemble_training_row(
                    target, td, decision_hour, float(tmax_label),
                    hours_this_day, hours_prev_day, daily_cache,
                    hourly_clim=hourly_clim,
                    daily_clim=daily_clim,
                )
                if row is None:
                    counts["rows_no_current"] += 1
                    continue
                vec, fetched_at = row

                if dry_run:
                    continue

                # NaN → null in JSON
                feats_as_list = [
                    (None if (v != v) else float(v))   # NaN test without math
                    for v in vec.tolist()
                ]
                insert_ml_training_row(
                    city                = target.city,
                    lat_key             = target.lat_key,
                    lon_key             = target.lon_key,
                    target_date         = d["date"],
                    decision_hour_local = decision_hour,
                    feature_version     = FEATURE_VERSION,
                    features_json       = json.dumps(feats_as_list),
                    t_max_c             = float(tmax_label),
                    fetched_at_utc      = fetched_at,
                )
                counts["rows_inserted"] += 1

        prev_month_hours = curr_month_hours
        log.info(
            f"  {target.city} {y}-{m:02d}: inserted={counts['rows_inserted']} "
            f"skipped={counts['rows_skipped']} "
            f"no_current={counts['rows_no_current']} "
            f"no_label={counts['rows_no_label']} "
            f"cost={counts['query_cost']}"
        )

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--city", action="append", default=None,
                    help="city name; repeatable.  default: all in temp_events")
    ap.add_argument("--years", type=int, default=10, help="lookback years (default 10)")
    ap.add_argument("--end-offset", type=int, default=30,
                    help="days to back off from today (default 30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble rows but do not insert")
    args = ap.parse_args()

    if not os.getenv("VISUAL_CROSSING_API_KEY"):
        log.error("VISUAL_CROSSING_API_KEY not set")
        return 1

    init_db()
    cities = resolve_cities(args.city)
    if not cities:
        log.error("no cities to backfill (temp_events may be empty; "
                  "pass --city explicitly)")
        return 1

    today = date.today()
    end_date = today - timedelta(days=args.end_offset)
    start_date = date(end_date.year - args.years, end_date.month, end_date.day)

    log.info(f"Backfill plan: {len(cities)} cities, window {start_date}..{end_date}, "
             f"decision hours {DECISION_HOURS_LOCAL} local, dry_run={args.dry_run}")

    totals = {"rows_inserted": 0, "query_cost": 0, "cities_done": 0, "cities_failed": 0}

    for t in cities:
        try:
            r = backfill_city(t, start_date, end_date, dry_run=args.dry_run)
            totals["rows_inserted"] += r["rows_inserted"]
            totals["query_cost"]    += r["query_cost"]
            totals["cities_done"]   += 1
            log.info(
                f"  [{t.city}] done  "
                f"inserted={r['rows_inserted']} cost={r['query_cost']} "
                f"no_current={r['rows_no_current']} failed_months={r['months_failed']}"
            )
        except Exception as e:
            totals["cities_failed"] += 1
            log.exception(f"{t.city}: backfill failed ({e})")

    total_rows = count_ml_training_rows(feature_version=FEATURE_VERSION)
    log.info("=" * 70)
    log.info(
        f"BACKFILL COMPLETE  cities_done={totals['cities_done']} "
        f"cities_failed={totals['cities_failed']} "
        f"rows_inserted_this_run={totals['rows_inserted']} "
        f"total_rows_version_{FEATURE_VERSION}={total_rows} "
        f"total_query_cost={totals['query_cost']}"
    )
    return 0 if totals["cities_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
