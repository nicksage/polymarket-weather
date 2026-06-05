"""
temp_drop_backtest.py — Test the hypothesis:
    "When the temperature drops sharply after the typical daily peak hours,
     the day's existing high usually holds and is not exceeded."

Pulls hourly temperature from Open-Meteo Archive (ERA5 reanalysis) for
every city the bot has traded, in LOCAL time per city.  Scans each day
for drop events (temp[h-W] - temp[h] >= threshold, with h >= after_hour),
records pre-drop high vs day's eventual high, reports hold rate.

The raw hourly data is cached in a separate SQLite (default
data/weather_archive.db) so re-running with different thresholds is
instant — no re-pull from the API.

Usage (defaults: last 2 years, ≥2°C drop in 2h ending after 12:00 local):
    cd bot
    python -m scripts.temp_drop_backtest
    python -m scripts.temp_drop_backtest --days 365
    python -m scripts.temp_drop_backtest --start 2024-01-01 --end 2026-05-07
    python -m scripts.temp_drop_backtest --threshold 3.0 --window-hours 3 --after-hour 13
    python -m scripts.temp_drop_backtest --city Wuhan Tokyo "Los Angeles"
    python -m scripts.temp_drop_backtest --csv out/events.csv
    python -m scripts.temp_drop_backtest --skip-fetch    # use only cached data
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
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

from db import _get_conn as _trading_conn  # only for city list

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_DB_PATH = os.path.join(_BOT_DIR, "data", "weather_archive.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("temp_drop")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Weather archive cache
# ---------------------------------------------------------------------------

def _init_archive_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS hourly_temps (
                city          TEXT NOT NULL,
                lat           REAL,
                lon           REAL,
                ts_local      TEXT NOT NULL,   -- 'YYYY-MM-DD HH:00' city-local
                date_local    TEXT NOT NULL,   -- 'YYYY-MM-DD' city-local
                hour_local    INTEGER NOT NULL,
                temp_c        REAL,
                PRIMARY KEY (city, ts_local)
            );
            CREATE INDEX IF NOT EXISTS idx_ht_city_date
                ON hourly_temps(city, date_local);
        """)


def _fetch_archive(lat: float, lon: float, start: str, end: str,
                    timezone: str = "auto", retries: int = 3) -> dict:
    """Single Open-Meteo Archive call.  timezone='auto' uses lat/lon to
    return local-time-stamped hourly rows (no per-city tz lookup needed)."""
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.get(
                ARCHIVE_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "hourly":     "temperature_2m",
                    "start_date": start,
                    "end_date":   end,
                    "timezone":   timezone,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo fetch failed after {retries} attempts: {last_err}")


def _city_already_complete(conn, city: str, start: str, end: str) -> bool:
    """True iff we already have ~24 hourly rows for every date in range."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end,   "%Y-%m-%d").date()
    expected_days = (d1 - d0).days + 1
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT date_local) AS n
        FROM hourly_temps
        WHERE city = ? AND date_local BETWEEN ? AND ?
        """,
        (city, start, end),
    ).fetchone()
    have_days = int(row[0] or 0)
    # Open-Meteo archive has a ~5-day lag so the tail of `end` may legitimately
    # be empty; tolerate up to 10 missing days at the tail.
    return have_days >= expected_days - 10


def _upsert_hourly(conn, city: str, lat: float, lon: float, payload: dict) -> int:
    h = payload.get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    rows = []
    for t, v in zip(times, temps):
        if v is None:
            continue
        # t looks like '2026-05-08T14:00' in local time (timezone=auto).
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
        ts_local   = dt.strftime("%Y-%m-%d %H:00")
        date_local = dt.strftime("%Y-%m-%d")
        rows.append((city, lat, lon, ts_local, date_local, dt.hour, float(v)))
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO hourly_temps
            (city, lat, lon, ts_local, date_local, hour_local, temp_c)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_city_range(db_path: str, cities: list[dict], start: str, end: str,
                      force: bool = False) -> None:
    _init_archive_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for i, c in enumerate(cities, 1):
            city = c["city"]
            if not force and _city_already_complete(conn, city, start, end):
                log.info(f"[{i}/{len(cities)}] {city:<16} cached, skipping fetch")
                continue
            log.info(f"[{i}/{len(cities)}] {city:<16} fetching {start} → {end} …")
            try:
                payload = _fetch_archive(c["lat"], c["lon"], start, end)
                n = _upsert_hourly(conn, city, c["lat"], c["lon"], payload)
                log.info(f"           wrote {n} hourly rows")
            except Exception as e:
                log.warning(f"           FAILED: {e}")
            time.sleep(0.3)   # gentle rate-limit


# ---------------------------------------------------------------------------
# City list
# ---------------------------------------------------------------------------

def load_city_list(filter_cities: list[str] | None) -> list[dict]:
    """Distinct cities the bot has traded, with lat/lon.  Dedups + drops
    rows missing coords.  Optional whitelist filter."""
    with _trading_conn() as conn:
        rows = conn.execute(
            """
            SELECT city, MAX(lat) AS lat, MAX(lon) AS lon
            FROM positions
            WHERE city IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL
            GROUP BY city
            ORDER BY city
            """
        ).fetchall()
    cities = [dict(r) for r in rows]
    if filter_cities:
        wanted = {c.strip().lower() for c in filter_cities}
        cities = [c for c in cities if c["city"].lower() in wanted]
    return cities


# ---------------------------------------------------------------------------
# Drop detection
# ---------------------------------------------------------------------------

def _detect_drops_for_day(hours: list[tuple[int, float]],
                           threshold: float, window: int,
                           after_hour: int) -> list[dict]:
    """For one day's hourly data, return drop events.  hours = sorted list
    of (hour_local, temp_c) tuples, 0..23 (some may be missing).
    A drop ends at hour h (h >= after_hour, h >= window) when:
        temp_at[h - window] - temp_at[h] >= threshold
    """
    by_hour = {h: t for h, t in hours}
    day_high = max(t for _, t in hours)
    events = []
    for h in range(max(after_hour, window), 24):
        if h not in by_hour or (h - window) not in by_hour:
            continue
        t_start = by_hour[h - window]
        t_end   = by_hour[h]
        drop    = t_start - t_end
        if drop < threshold:
            continue
        # "Pre-drop high" = running max from start-of-day up to and
        # including the drop window's start hour.
        pre_drop_high = max(
            by_hour[hh] for hh in range(0, h - window + 1) if hh in by_hour
        )
        held_exact = abs(day_high - pre_drop_high) < 1e-6
        held_05    = (day_high - pre_drop_high) <= 0.5
        overshoot  = max(0.0, day_high - pre_drop_high)
        events.append({
            "drop_end_hour":   h,
            "drop_start_hour": h - window,
            "drop_start_temp": round(t_start, 2),
            "drop_end_temp":   round(t_end, 2),
            "drop_magnitude":  round(drop, 2),
            "pre_drop_high":   round(pre_drop_high, 2),
            "day_high":        round(day_high, 2),
            "overshoot":       round(overshoot, 2),
            "held_exact":      held_exact,
            "held_within_0.5": held_05,
        })
    return events


def detect_all_drops(db_path: str, cities: list[dict], start: str, end: str,
                      threshold: float, window: int, after_hour: int,
                      first_per_day_only: bool) -> list[dict]:
    """Scan every (city, date) in range; return all qualifying drop events."""
    events_all: list[dict] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for c in cities:
            rows = conn.execute(
                """
                SELECT date_local, hour_local, temp_c
                FROM hourly_temps
                WHERE city = ? AND date_local BETWEEN ? AND ?
                  AND temp_c IS NOT NULL
                ORDER BY date_local, hour_local
                """,
                (c["city"], start, end),
            ).fetchall()
            by_date: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for r in rows:
                by_date[r["date_local"]].append((r["hour_local"], r["temp_c"]))

            for d, hours in by_date.items():
                if len(hours) < 20:   # need most of a day
                    continue
                day_events = _detect_drops_for_day(
                    hours, threshold, window, after_hour,
                )
                if first_per_day_only and day_events:
                    day_events = day_events[:1]
                for e in day_events:
                    e["city"] = c["city"]
                    e["date"] = d
                    events_all.append(e)
    return events_all


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "  —  "


def summarize(events: list[dict], cities: list[dict], start: str, end: str,
              threshold: float, window: int, after_hour: int) -> None:
    n = len(events)
    print()
    print("=" * 86)
    print("  TEMPERATURE-DROP HOLD-RATE BACKTEST")
    print("=" * 86)
    print(f"  Cities:        {len(cities)}")
    print(f"  Date range:    {start} → {end}")
    print(f"  Drop rule:     temp[h-{window}h] − temp[h] ≥ {threshold:.1f}°C, "
          f"h ≥ {after_hour}:00 local")
    print(f"  Drop events:   {n:,}")
    if n == 0:
        print("  Nothing to summarize.")
        return

    held_exact = sum(1 for e in events if e["held_exact"])
    held_05    = sum(1 for e in events if e["held_within_0.5"])
    broken     = n - held_exact

    print()
    print("  HYPOTHESIS — did the pre-drop high hold as the day's high?")
    print(f"    Held EXACTLY (day_high == pre_drop_high):  "
          f"{held_exact:>6,d} / {n:,}  ({_pct(held_exact, n)})")
    print(f"    Held within 0.5°C:                          "
          f"{held_05:>6,d} / {n:,}  ({_pct(held_05, n)})")
    print(f"    BROKEN (day exceeded pre-drop high):        "
          f"{broken:>6,d} / {n:,}  ({_pct(broken, n)})")
    if broken:
        overs = [e["overshoot"] for e in events if not e["held_exact"]]
        avg_over = sum(overs) / len(overs)
        max_over = max(overs)
        print(f"      when broken: avg overshoot {avg_over:.2f}°C, "
              f"max {max_over:.2f}°C")

    # By magnitude
    print()
    print("  BY DROP MAGNITUDE:")
    print(f"    {'bucket':<14} {'events':>7}  {'held':>6}  {'within 0.5':>10}")
    buckets = [
        (f"{threshold:.1f}–3.0°C",  lambda e: e["drop_magnitude"] < 3.0),
        ("3.0–4.0°C",   lambda e: 3.0 <= e["drop_magnitude"] < 4.0),
        ("4.0–6.0°C",   lambda e: 4.0 <= e["drop_magnitude"] < 6.0),
        ("6.0°C+",      lambda e: e["drop_magnitude"] >= 6.0),
    ]
    for label, pred in buckets:
        es = [e for e in events if pred(e)]
        if not es:
            continue
        h_exact = sum(1 for e in es if e["held_exact"])
        h_05    = sum(1 for e in es if e["held_within_0.5"])
        print(f"    {label:<14} {len(es):>7,d}  "
              f"{_pct(h_exact, len(es)):>6}  {_pct(h_05, len(es)):>10}")

    # By hour-of-day of drop end
    print()
    print("  BY DROP-END HOUR (local time):")
    print(f"    {'hour':<5} {'events':>7}  {'held':>6}  {'within 0.5':>10}")
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_hour[e["drop_end_hour"]].append(e)
    for h in sorted(by_hour):
        es = by_hour[h]
        h_exact = sum(1 for e in es if e["held_exact"])
        h_05    = sum(1 for e in es if e["held_within_0.5"])
        print(f"    {h:>2}:00 {len(es):>7,d}  "
              f"{_pct(h_exact, len(es)):>6}  {_pct(h_05, len(es)):>10}")

    # By city — top + bottom 5 hold rates (require n>=10)
    print()
    print("  BY CITY (n≥10 events, sorted by hold-exact rate):")
    by_city: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_city[e["city"]].append(e)
    city_stats = []
    for city, es in by_city.items():
        if len(es) < 10:
            continue
        h_exact = sum(1 for e in es if e["held_exact"])
        city_stats.append((city, len(es), h_exact, h_exact / len(es)))
    city_stats.sort(key=lambda x: -x[3])
    if city_stats:
        print(f"    {'city':<16} {'events':>7}  {'held':>6}")
        if len(city_stats) <= 10:
            for city, n_e, h_e, _ in city_stats:
                print(f"    {city:<16} {n_e:>7,d}  {_pct(h_e, n_e):>6}")
        else:
            for city, n_e, h_e, _ in city_stats[:5]:
                print(f"    {city:<16} {n_e:>7,d}  {_pct(h_e, n_e):>6}  ← top")
            print("    …")
            for city, n_e, h_e, _ in city_stats[-5:]:
                print(f"    {city:<16} {n_e:>7,d}  {_pct(h_e, n_e):>6}  ← bottom")

    # Per-day frequency
    n_city_days = sum(1 for c in cities) * (
        (datetime.strptime(end, "%Y-%m-%d") -
         datetime.strptime(start, "%Y-%m-%d")).days + 1
    )
    distinct_city_days = len({(e["city"], e["date"]) for e in events})
    print()
    print(f"  FREQUENCY:")
    print(f"    City-days in window:                {n_city_days:>7,d}")
    print(f"    City-days with ≥1 qualifying drop:  {distinct_city_days:>7,d}  "
          f"({_pct(distinct_city_days, n_city_days)})")
    print()


def write_csv(events: list[dict], path: str) -> None:
    if not events:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["city", "date", "drop_start_hour", "drop_end_hour",
              "drop_start_temp", "drop_end_temp", "drop_magnitude",
              "pre_drop_high", "day_high", "overshoot",
              "held_exact", "held_within_0.5"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow({k: e[k] for k in fields})
    log.info(f"Wrote {len(events):,} events to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=730,
                   help="Look back N days from today (default: 730 = ~2 years)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="Minimum drop in °C (default: 2.0)")
    p.add_argument("--window-hours", type=int, default=2,
                   help="Window length in hours (default: 2)")
    p.add_argument("--after-hour", type=int, default=12,
                   help="Drop must END at or after this LOCAL hour (default: 12)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all traded)")
    p.add_argument("--db", default=DEFAULT_DB_PATH,
                   help="Weather archive SQLite path "
                        f"(default: {DEFAULT_DB_PATH})")
    p.add_argument("--csv", help="Write per-event CSV to this path")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Use only cached data; don't hit the API")
    p.add_argument("--force-refetch", action="store_true",
                   help="Re-pull all dates even if already cached")
    p.add_argument("--first-per-day", action="store_true",
                   help="Only count the FIRST qualifying drop per city-day "
                        "(default: count every qualifying drop)")
    args = p.parse_args()

    # Resolve date range
    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today())
    if args.start:
        start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_d = end_d - timedelta(days=args.days)
    start = start_d.isoformat()
    end   = end_d.isoformat()

    cities = load_city_list(args.city)
    if not cities:
        print("No cities found — has the bot traded any with lat/lon?")
        return 1
    log.info(f"Cities: {len(cities)}  |  Window: {start} → {end}")

    if not args.skip_fetch:
        fetch_city_range(args.db, cities, start, end, force=args.force_refetch)
    else:
        _init_archive_db(args.db)
        log.info("--skip-fetch set; using only cached hourly data")

    events = detect_all_drops(
        args.db, cities, start, end,
        threshold=args.threshold, window=args.window_hours,
        after_hour=args.after_hour,
        first_per_day_only=args.first_per_day,
    )

    summarize(events, cities, start, end,
              args.threshold, args.window_hours, args.after_hour)

    if args.csv:
        write_csv(events, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())