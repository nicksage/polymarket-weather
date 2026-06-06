"""
neighbor_obs_pull.py — Pull historical hourly observations for the
ASOS neighbors of each US Polymarket settlement station, into a local
SQLite cache (data/neighbor_obs.db).

Builds the foundation for lead-lag analysis: with each neighbor's hourly
history alongside `station_obs.db` (which has the settlement stations
themselves), we can compute time-shifted correlations to identify which
neighbors lead the settlement station, and under what wind conditions.

Schema:
    neighbor_meta       — (sid, name, network, polymarket_city, lat, lon,
                            distance_mi, bearing_deg, direction)
    neighbor_obs        — (sid, ts_local, date_local, hour_local, temp_c)
    neighbor_daily_max  — (sid, date_local, tmax_c, tmax_hour_local, n_obs)

Uses the SAME parse-time max-per-hour fix as station_obs_pull, so per-hour
values reflect the actual peak across all METARs/SPECIs that hour.

Usage:
    cd bot
    python -m scripts.neighbor_obs_pull                     # 60d, top 5/city, ASOS-only
    python -m scripts.neighbor_obs_pull --days 90 --neighbors-per-city 8
    python -m scripts.neighbor_obs_pull --include-awos      # add smaller airports
    python -m scripts.neighbor_obs_pull --city Dallas       # one city only
    python -m scripts.neighbor_obs_pull --force             # re-fetch even if cached
"""

from __future__ import annotations

import argparse
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

from station_meta import CITY_STATIONS  # type: ignore
from scripts.find_nearby_stations import (  # type: ignore
    US_CITY_STATES, find_nearby,
)

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_DB = os.path.join(_BOT_DIR, "data", "neighbor_obs.db")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("neighbor_obs")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def _init_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS neighbor_meta (
                sid                TEXT NOT NULL,
                name               TEXT,
                network            TEXT,
                polymarket_city    TEXT NOT NULL,
                polymarket_station TEXT,
                lat                REAL,
                lon                REAL,
                elev_m             REAL,
                distance_mi        REAL,
                bearing_deg        REAL,
                direction          TEXT,
                PRIMARY KEY (sid, polymarket_city)
            );

            CREATE TABLE IF NOT EXISTS neighbor_obs (
                sid          TEXT NOT NULL,
                ts_local     TEXT NOT NULL,
                date_local   TEXT NOT NULL,
                hour_local   INTEGER NOT NULL,
                temp_c       REAL,
                PRIMARY KEY (sid, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_nobs_sid_date
                ON neighbor_obs(sid, date_local);

            CREATE TABLE IF NOT EXISTS neighbor_daily_max (
                sid             TEXT NOT NULL,
                date_local      TEXT NOT NULL,
                tmax_c          REAL,
                tmax_hour_local INTEGER,
                n_obs           INTEGER,
                PRIMARY KEY (sid, date_local)
            );
        """)


def upsert_meta(conn, rows: list[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO neighbor_meta
            (sid, name, network, polymarket_city, polymarket_station,
             lat, lon, elev_m, distance_mi, bearing_deg, direction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(r["sid"], r["name"], r["network"], r["polymarket_city"],
          r["polymarket_station"], r["lat"], r["lon"], r["elev_m"],
          r["distance_mi"], r["bearing_deg"], r["direction"])
         for r in rows],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Mesonet fetch
# ---------------------------------------------------------------------------

def fetch_csv(sid: str, network: str, tz: str, start: str, end: str,
               retries: int = 3) -> str:
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end,   "%Y-%m-%d")
    params = {
        "station":     sid,
        "network":     network,
        "data":        "tmpc",
        "year1":       d0.year, "month1": d0.month, "day1": d0.day,
        "year2":       d1.year, "month2": d1.month, "day2": d1.day,
        "tz":          tz,
        "format":      "onlycomma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": [3, 4],
    }
    last = None
    for attempt in range(retries):
        try:
            r = httpx.get(ASOS_URL, params=params, timeout=120)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(3 ** attempt)
    raise RuntimeError(f"Mesonet fetch failed for {sid}: {last}")


def parse_and_upsert(conn, sid: str, csv_text: str) -> int:
    """Same max-per-hour logic as the settlement-station puller — captures
    the peak across all METARs/SPECIs that hour."""
    lines = csv_text.splitlines()
    if not lines or not lines[0].lower().startswith("station"):
        return 0
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
    rows = [(sid, ts, ts[:10], int(ts[11:13]), temp)
            for ts, temp in by_hour.items()]
    conn.executemany(
        """
        INSERT OR REPLACE INTO neighbor_obs
            (sid, ts_local, date_local, hour_local, temp_c)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def refresh_daily_max(conn, sid: str, start: str, end: str) -> int:
    rows = conn.execute(
        """
        SELECT date_local, MAX(temp_c) AS tmax, COUNT(*) AS n
        FROM neighbor_obs
        WHERE sid = ? AND date_local BETWEEN ? AND ?
          AND temp_c IS NOT NULL
        GROUP BY date_local
        """,
        (sid, start, end),
    ).fetchall()
    n_inserted = 0
    for date_local, tmax, n in rows:
        hour_row = conn.execute(
            """
            SELECT hour_local FROM neighbor_obs
            WHERE sid = ? AND date_local = ? AND temp_c = ?
            ORDER BY hour_local LIMIT 1
            """,
            (sid, date_local, tmax),
        ).fetchone()
        tmax_hour = int(hour_row[0]) if hour_row else None
        conn.execute(
            """
            INSERT OR REPLACE INTO neighbor_daily_max
                (sid, date_local, tmax_c, tmax_hour_local, n_obs)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sid, date_local, tmax, tmax_hour, n),
        )
        n_inserted += 1
    conn.commit()
    return n_inserted


def is_cached(conn, sid: str, start: str, end: str,
               min_obs_per_day: int = 18) -> bool:
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end,   "%Y-%m-%d").date()
    n_expected = (d1 - d0).days + 1
    row = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT date_local FROM neighbor_obs
            WHERE sid = ? AND date_local BETWEEN ? AND ?
              AND temp_c IS NOT NULL
            GROUP BY date_local HAVING COUNT(*) >= ?
        )
        """,
        (sid, start, end, min_obs_per_day),
    ).fetchone()
    have = int(row[0]) if row else 0
    # Require at least 1 day AND >= 90% of expected days (tolerates small
    # Mesonet gaps without falsely claiming empty data is "complete").
    if have == 0:
        return False
    return have >= max(1, int(n_expected * 0.9))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_neighbor_list(cities: list[str], neighbors_per_city: int,
                          include_awos: bool, include_rwis: bool,
                          radius_mi: float) -> list[dict]:
    """For each city, find nearby stations and return up to N candidates
    after filtering to the requested network types."""
    network_cache: dict = {}
    all_rows: list[dict] = []
    for city in cities:
        rows = find_nearby(city, radius_mi=radius_mi, extended=True,
                            cache=network_cache)
        # Filter networks: always ASOS, optionally AWOS, RWIS.  Skip the
        # settlement station itself (it's in station_obs.db already).
        pm_icao = CITY_STATIONS[city][0]
        keep = []
        for r in rows:
            net = r["network"]
            if net.endswith("_ASOS"):
                pass
            elif net.endswith("_AWOS") and include_awos:
                pass
            elif net.endswith("_RWIS") and include_rwis:
                pass
            else:
                continue
            # Don't duplicate the settlement station
            if r["sid"] == pm_icao or r["sid"] == pm_icao.lstrip("K"):
                continue
            keep.append(r)
        # Sort by distance, take the top N
        keep.sort(key=lambda r: r["distance_mi"])
        all_rows.extend(keep[:neighbors_per_city])
    return all_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=60,
                   help="History window in days (default: 60)")
    p.add_argument("--start", help="Start YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",   help="End YYYY-MM-DD (default: today)")
    p.add_argument("--neighbors-per-city", type=int, default=5,
                   help="Max ASOS neighbors per city to pull (default: 5)")
    p.add_argument("--radius-mi", type=float, default=25,
                   help="Search radius when finding neighbors (default: 25)")
    p.add_argument("--include-awos", action="store_true",
                   help="Also pull AWOS (smaller airports)")
    p.add_argument("--include-rwis", action="store_true",
                   help="Also pull RWIS (road-weather sensors — noisy for air temp)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific US cities (default: all 11)")
    p.add_argument("--db", default=DEFAULT_DB,
                   help=f"Output SQLite (default: {DEFAULT_DB})")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even if data is cached for this window")
    args = p.parse_args()

    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today())
    start_d = (datetime.strptime(args.start, "%Y-%m-%d").date()
               if args.start else end_d - timedelta(days=args.days))
    start, end = start_d.isoformat(), end_d.isoformat()

    cities = (args.city if args.city else list(US_CITY_STATES.keys()))
    cities = [c for c in cities if c in US_CITY_STATES]
    if not cities:
        log.error(f"No matching US cities. Available: {sorted(US_CITY_STATES)}")
        return 1

    log.info(f"Window: {start} → {end}  ({(end_d - start_d).days + 1} days)")
    log.info(f"Cities: {len(cities)}  | top {args.neighbors_per_city} ASOS"
             f"{' +AWOS' if args.include_awos else ''}"
             f"{' +RWIS' if args.include_rwis else ''} neighbors each")

    log.info("Discovering neighbors via Mesonet network metadata ...")
    neighbors = build_neighbor_list(
        cities, args.neighbors_per_city,
        args.include_awos, args.include_rwis, args.radius_mi,
    )
    log.info(f"Selected {len(neighbors)} unique (city, neighbor) pairs")

    # Deduplicate by sid for fetching (same neighbor might serve multiple cities,
    # e.g., a station between Dallas and Fort Worth).  But we still want one
    # neighbor_meta row per (sid, city) so the joins make sense.
    _init_db(args.db)
    with sqlite3.connect(args.db) as conn:
        upsert_meta(conn, neighbors)
        # Attach tz from CITY_STATIONS for each (we'll fetch one row per
        # unique sid, then write into neighbor_obs which is sid-keyed)
        unique_sids: dict[str, dict] = {}
        for r in neighbors:
            if r["sid"] not in unique_sids:
                tz = CITY_STATIONS[r["polymarket_city"]][2]
                unique_sids[r["sid"]] = {
                    "sid": r["sid"], "network": r["network"], "tz": tz,
                    "city_hint": r["polymarket_city"],
                }

        fetched = cached = failed = 0
        n_total = len(unique_sids)
        for i, (sid, m) in enumerate(unique_sids.items(), 1):
            if not args.force and is_cached(conn, sid, start, end):
                log.info(f"[{i:>3}/{n_total}] {sid:<7} cached, skipping")
                cached += 1
                continue
            log.info(f"[{i:>3}/{n_total}] {sid:<7} ({m['network']:<12}, "
                     f"{m['tz']}) fetching ...")
            try:
                csv_text = fetch_csv(sid, m["network"], m["tz"], start, end)
                n_rows = parse_and_upsert(conn, sid, csv_text)
                if n_rows == 0:
                    log.warning(f"           empty response (station inactive?)")
                    failed += 1
                    continue
                n_days = refresh_daily_max(conn, sid, start, end)
                log.info(f"           wrote {n_rows:,} hourly rows, "
                         f"{n_days} daily-max rows")
                fetched += 1
            except Exception as e:
                log.warning(f"           FAILED: {e}")
                failed += 1
            time.sleep(0.3)   # be polite to Mesonet

    print()
    print("=" * 78)
    print(f"  Window:   {start} → {end}")
    print(f"  Unique neighbor stations: {n_total}")
    print(f"    fetched (new data):  {fetched}")
    print(f"    cached (skipped):    {cached}")
    print(f"    failed:              {failed}")
    print(f"  (city, neighbor) pairs in neighbor_meta: {len(neighbors)}")
    print(f"  DB: {args.db}")
    print("=" * 78)
    print()
    print("Next: lead-lag analysis (Phase B). With both DBs populated, we can")
    print("compute per-(settlement, neighbor) pair: peak-time deltas, hourly")
    print("cross-correlations, and wind-stratified leader maps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())