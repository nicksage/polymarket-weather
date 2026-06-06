"""
station_obs_pull.py — Pull hourly METAR observations from Polymarket's
exact settlement stations into a local SQLite cache.

Source: Iowa State Mesonet ASOS-AWOS archive
   https://mesonet.agron.iastate.edu/request/download.phtml
Free, no API key.  Same underlying NOAA METAR feed that Wunderground
re-publishes on its history pages.

Station list is hardcoded from the Wunderground / weather.gov links the
operator confirmed are used by Polymarket settlement.  Update
CITY_STATIONS if Polymarket changes its sources.

Usage:
    cd bot
    python -m scripts.station_obs_pull              # default 2 years
    python -m scripts.station_obs_pull --days 365
    python -m scripts.station_obs_pull --start 2024-01-01 --end 2026-06-04
    python -m scripts.station_obs_pull --city Tokyo NYC Dallas
    python -m scripts.station_obs_pull --db data/station_obs.db
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta

import httpx

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS as _STATIONS_FULL  # type: ignore

# Backwards-compat: this script previously hardcoded a 3-tuple per city.
# Re-shape the shared 5-tuple (icao, net, tz, lat, lon) into the 3-tuple
# this file's existing fetch/pull code expects.
CITY_STATIONS: dict[str, tuple[str, str, str]] = {
    city: (icao, net, tz)
    for city, (icao, net, tz, _lat, _lon) in _STATIONS_FULL.items()
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("station_obs")
logging.getLogger("httpx").setLevel(logging.WARNING)

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_DB_PATH = os.path.join(_BOT_DIR, "data", "station_obs.db")


# CITY_STATIONS is now imported from station_meta.py (single source of truth).
# It's re-shaped above to keep this script's existing 3-tuple expectations.


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def _init_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            -- Local-time hourly observations.  Each row is one observation
            -- closest to the top of a local hour at the settlement station.
            CREATE TABLE IF NOT EXISTS station_obs (
                city          TEXT NOT NULL,
                station       TEXT NOT NULL,
                ts_local      TEXT NOT NULL,   -- 'YYYY-MM-DD HH:00' local
                date_local    TEXT NOT NULL,   -- 'YYYY-MM-DD' local
                hour_local    INTEGER NOT NULL,
                temp_c        REAL,
                PRIMARY KEY (city, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_so_city_date
                ON station_obs(city, date_local);
            CREATE INDEX IF NOT EXISTS idx_so_station_date
                ON station_obs(station, date_local);

            -- Pre-computed daily max in LOCAL date — matches Polymarket's
            -- "high on date X" market settlement semantics.
            CREATE TABLE IF NOT EXISTS station_daily_max (
                city            TEXT NOT NULL,
                station         TEXT NOT NULL,
                date_local      TEXT NOT NULL,
                tmax_c          REAL,
                tmax_hour_local INTEGER,
                n_obs           INTEGER,
                PRIMARY KEY (city, date_local)
            );

            -- Drop the old UTC-keyed columns if a previous version of the
            -- script created them.  SQLite is permissive about extra cols.
        """)
        # Migrate-on-load: if a prior run used the old ts_utc/date_utc
        # columns, drop those tables so the new schema takes over cleanly.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(station_obs)").fetchall()]
        if "ts_utc" in cols:
            conn.executescript("""
                DROP TABLE station_obs;
                DROP TABLE IF EXISTS station_daily_max;
                CREATE TABLE station_obs (
                    city TEXT NOT NULL, station TEXT NOT NULL,
                    ts_local TEXT NOT NULL, date_local TEXT NOT NULL,
                    hour_local INTEGER NOT NULL, temp_c REAL,
                    PRIMARY KEY (city, ts_local)
                );
                CREATE TABLE station_daily_max (
                    city TEXT NOT NULL, station TEXT NOT NULL,
                    date_local TEXT NOT NULL, tmax_c REAL,
                    tmax_hour_local INTEGER, n_obs INTEGER,
                    PRIMARY KEY (city, date_local)
                );
                CREATE INDEX idx_so_city_date ON station_obs(city, date_local);
                CREATE INDEX idx_so_station_date ON station_obs(station, date_local);
            """)


# ---------------------------------------------------------------------------
# Mesonet pull
# ---------------------------------------------------------------------------

def _fetch_csv(station: str, network: str, timezone: str,
                start: str, end: str, retries: int = 3) -> str:
    """Pull hourly tmpc (°C) from Iowa State Mesonet as CSV, timestamped
    in the station's LOCAL timezone."""
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end,   "%Y-%m-%d")
    params = {
        "station":     station,
        "network":     network,
        "data":        "tmpc",
        "year1":       d0.year,  "month1": d0.month, "day1": d0.day,
        "year2":       d1.year,  "month2": d1.month, "day2": d1.day,
        "tz":          timezone,
        "format":      "onlycomma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": [3, 4],   # routine + special METAR
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.get(ASOS_URL, params=params, timeout=120)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(3 ** attempt)
    raise RuntimeError(f"Mesonet fetch failed for {station}: {last_err}")


def _parse_and_upsert(conn, city: str, station: str, csv_text: str) -> int:
    """Parse Mesonet CSV (station,valid,tmpc).  For each local hour, keep
    the MAXIMUM temperature across all METARs and SPECIs that hour — this
    is what Polymarket settles on (the actual peak), not the top-of-hour
    routine.  Picking by minute-of-hour silently discarded sub-hour spikes
    and caused our day_max to disagree with the market.
    """
    lines = csv_text.splitlines()
    if not lines or not lines[0].lower().startswith("station"):
        return 0
    # Per-hour MAX across all observations (routine METARs + SPECIs)
    by_hour: dict[str, float] = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        _, valid, tmpc = parts[0], parts[1], parts[2]
        if not valid or tmpc in ("M", "", "T"):
            continue
        try:
            dt = datetime.strptime(valid.strip(), "%Y-%m-%d %H:%M")
            t  = float(tmpc)
        except ValueError:
            continue
        hour_key = dt.strftime("%Y-%m-%d %H:00")
        existing = by_hour.get(hour_key)
        if existing is None or t > existing:
            by_hour[hour_key] = t

    if not by_hour:
        return 0
    rows = [(city, station, ts, ts[:10], int(ts[11:13]), temp)
            for ts, temp in by_hour.items()]
    conn.executemany(
        """
        INSERT OR REPLACE INTO station_obs
            (city, station, ts_local, date_local, hour_local, temp_c)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _refresh_daily_max(conn, city: str, station: str,
                        start: str, end: str) -> int:
    """Rebuild station_daily_max for this city's LOCAL date range."""
    rows = conn.execute(
        """
        SELECT date_local,
               MAX(temp_c) AS tmax,
               COUNT(*)    AS n_obs
        FROM station_obs
        WHERE city = ? AND date_local BETWEEN ? AND ?
          AND temp_c IS NOT NULL
        GROUP BY date_local
        """,
        (city, start, end),
    ).fetchall()
    inserted = 0
    for date_local, tmax, n_obs in rows:
        hour_row = conn.execute(
            """
            SELECT hour_local FROM station_obs
            WHERE city = ? AND date_local = ? AND temp_c = ?
            ORDER BY hour_local LIMIT 1
            """,
            (city, date_local, tmax),
        ).fetchone()
        tmax_hour = int(hour_row[0]) if hour_row else None
        conn.execute(
            """
            INSERT OR REPLACE INTO station_daily_max
                (city, station, date_local, tmax_c, tmax_hour_local, n_obs)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (city, station, date_local, tmax, tmax_hour, n_obs),
        )
        inserted += 1
    conn.commit()
    return inserted


def _city_complete(conn, city: str, start: str, end: str,
                    min_obs_per_day: int = 18) -> bool:
    """True iff most days in range have at least min_obs_per_day rows."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end,   "%Y-%m-%d").date()
    n_expected = (d1 - d0).days + 1
    row = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT date_local
            FROM station_obs
            WHERE city = ? AND date_local BETWEEN ? AND ?
              AND temp_c IS NOT NULL
            GROUP BY date_local
            HAVING COUNT(*) >= ?
        )
        """,
        (city, start, end, min_obs_per_day),
    ).fetchone()
    have = int(row[0]) if row else 0
    # Tolerate up to 10 missing days (Mesonet has small gaps)
    return have >= n_expected - 10


def pull_all(db_path: str, cities: dict[str, tuple[str, str, str]],
              start: str, end: str, force: bool = False) -> dict:
    _init_db(db_path)
    summary = {"fetched": 0, "cached": 0, "failed": []}
    with sqlite3.connect(db_path) as conn:
        for i, (city, (station, network, tz)) in enumerate(sorted(cities.items()), 1):
            if not force and _city_complete(conn, city, start, end):
                log.info(f"[{i:>2}/{len(cities)}] {city:<16} {station} cached, skipping")
                summary["cached"] += 1
                continue
            log.info(f"[{i:>2}/{len(cities)}] {city:<16} {station} ({network}, {tz}) "
                     f"fetching {start} → {end} …")
            try:
                csv_text = _fetch_csv(station, network, tz, start, end)
                n = _parse_and_upsert(conn, city, station, csv_text)
                if n == 0:
                    log.warning(f"           empty response — network/station may be wrong")
                    summary["failed"].append((city, station, "empty_response"))
                    continue
                d = _refresh_daily_max(conn, city, station, start, end)
                log.info(f"           wrote {n:,} hourly rows, {d:,} daily-max rows")
                summary["fetched"] += 1
            except Exception as e:
                log.warning(f"           FAILED: {e}")
                summary["failed"].append((city, station, str(e)))
            time.sleep(0.5)   # gentle on Mesonet
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=730,
                   help="Look back N days from today (default: 730 = ~2 years)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all)")
    p.add_argument("--db", default=DEFAULT_DB_PATH,
                   help=f"Output SQLite path (default: {DEFAULT_DB_PATH})")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if data already cached")
    p.add_argument("--list-stations", action="store_true",
                   help="Print the CITY_STATIONS map and exit")
    args = p.parse_args()

    if args.list_stations:
        print(f"{'CITY':<16} {'ICAO':<6} {'NETWORK':<14} {'TIMEZONE'}")
        for city, (station, net, tz) in sorted(CITY_STATIONS.items()):
            print(f"{city:<16} {station:<6} {net:<14} {tz}")
        return 0

    cities = dict(CITY_STATIONS)
    if args.city:
        wanted = {c.strip().lower() for c in args.city}
        cities = {c: v for c, v in cities.items() if c.lower() in wanted}
        if not cities:
            print(f"No cities matched: {args.city}")
            return 1

    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today())
    if args.start:
        start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_d = end_d - timedelta(days=args.days)
    start = start_d.isoformat()
    end   = end_d.isoformat()

    log.info(f"Pulling {len(cities)} stations from Iowa State Mesonet, "
             f"{start} → {end} ({(end_d - start_d).days + 1} days)")
    summary = pull_all(args.db, cities, start, end, force=args.force)

    print()
    print(f"Cached  (already had data): {summary['cached']}")
    print(f"Fetched (new data):         {summary['fetched']}")
    print(f"Failed:                     {len(summary['failed'])}")
    if summary["failed"]:
        for city, station, err in summary["failed"]:
            print(f"  {city:<16} {station:<6} {err[:70]}")
    print(f"\nDB: {args.db}")
    print("Note: Hong Kong is intentionally excluded "
          "(HKO uses a manual climat station with no ASOS equivalent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())