"""
backfill_backtest_data.py — One-time (re-runnable) fetch of hourly
observations for every resolved event in the backtest window.

Source: Open-Meteo Archive API (ERA5 reanalysis).  Free, no key required.

For each distinct (city, lat, lon, target_date) in temp_events whose target
date is in the past AND has a matching actual tmax in
historical_observed_daily, fetch 24 hourly temperatures and upsert into
backtest.db::historical_hourly_observations.

Usage:
    python bot/scripts/backfill_backtest_data.py              # default 90d
    python bot/scripts/backfill_backtest_data.py --days 30    # narrower
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import date, timedelta
from typing import Iterable

import httpx

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from backtest.backtest_db import (
    init_backtest_db, upsert_historical_hourly_obs, get_historical_hourly_obs,
)
from config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("backfill")
logging.getLogger("httpx").setLevel(logging.WARNING)


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _iter_target_events(window_days: int) -> Iterable[dict]:
    """Yield one dict per distinct (city, lat, lon, target_date) whose date
    is in the past N days AND has a matching actual tmax."""
    cutoff_start = (date.today() - timedelta(days=window_days)).isoformat()
    cutoff_end   = date.today().isoformat()   # exclusive — today hasn't resolved
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT DISTINCT te.city, te.lat, te.lon, te.date
        FROM temp_events te
        JOIN historical_observed_daily h
          ON ROUND(te.lat, 2) = h.lat_key
         AND ROUND(te.lon, 2) = h.lon_key
         AND te.date = h.date
        WHERE te.date >= ? AND te.date < ?
          AND h.tempmax_c IS NOT NULL
          AND te.lat IS NOT NULL AND te.lon IS NOT NULL
        ORDER BY te.date, te.city
    """, (cutoff_start, cutoff_end)).fetchall()
    for r in rows:
        yield dict(r)


def _fetch_archive_hourly(
    lat: float, lon: float, target_date: str,
) -> list[dict]:
    """Return 24 hourly rows for a past date from Open-Meteo Archive API."""
    r = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     "temperature_2m",
            "start_date": target_date,
            "end_date":   target_date,
            "timezone":   "UTC",
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json().get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    rows: list[dict] = []
    for t, v in zip(times, temps):
        if v is None:
            continue
        # Archive returns naive ISO (no tz suffix) when timezone=UTC.
        # Normalize to explicit UTC marker for consistency.
        ts = t if t.endswith("Z") else f"{t}:00Z" if len(t) == 13 else f"{t}Z"
        # Many endpoints return "2026-04-14T00:00"; we want full ISO.
        if len(t) == 16 and "T" in t:
            ts = f"{t}:00Z"
        else:
            ts = t if t.endswith("Z") else f"{t}Z"
        rows.append({
            "date":        target_date,
            "hour_ts_utc": ts,
            "temp_c":      float(v),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90,
                        help="Lookback window in days (default: 90)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip (city,date) pairs already present (default)")
    parser.add_argument("--sleep-ms", type=int, default=150,
                        help="Delay between API calls in ms (default: 150)")
    args = parser.parse_args()

    init_backtest_db()

    targets = list(_iter_target_events(args.days))
    log.info(f"Found {len(targets)} distinct (city, date) targets in last {args.days} days")

    n_skip = 0
    n_fetched = 0
    n_rows = 0
    n_err = 0

    for t in targets:
        city, lat, lon, tgt = t["city"], t["lat"], t["lon"], t["date"]
        existing = get_historical_hourly_obs(lat, lon, tgt)
        if args.skip_existing and existing and len(existing) >= 20:
            n_skip += 1
            continue
        try:
            rows = _fetch_archive_hourly(lat, lon, tgt)
        except Exception as e:
            log.warning(f"  {city:<16} {tgt}  FAIL: {type(e).__name__}: {str(e)[:80]}")
            n_err += 1
            continue
        if not rows:
            log.warning(f"  {city:<16} {tgt}  no data returned")
            n_err += 1
            continue
        upsert_historical_hourly_obs(lat, lon, city, rows)
        n_fetched += 1
        n_rows    += len(rows)
        if n_fetched % 10 == 0:
            log.info(f"  progress: fetched={n_fetched} skipped={n_skip} errors={n_err}")
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    log.info(
        f"Done: fetched={n_fetched} rows_written={n_rows} "
        f"skipped_existing={n_skip} errors={n_err}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
