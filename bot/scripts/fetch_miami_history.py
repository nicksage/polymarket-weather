#!/usr/bin/env python3
"""
fetch_miami_history.py — print a merged terminal table of the temperature
observations and Polymarket bin prices the bot has ALREADY SAVED to its
local SQLite DB (bot/data/signals.db).

Sources (no network calls):
  - raw_metar_log              : NWS METAR readings (every ~5 min)
  - paper_predictor_signals    : bin market_prob (every PREDICTOR_SCAN_MIN min)

Defaults to today, 11:00–16:00 Eastern, Miami 94-95F bin.  Override:

    python scripts/fetch_miami_history.py \
        --date 2026-06-16 --start-local 11:00 --end-local 16:00 \
        --bin 94-95 --city Miami --station KMIA --tz America/New_York

Output: one row per minute that had either a METAR observation or a
predictor scan in the window.  No files written.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
DEFAULT_DB = os.path.join(_BOT_DIR, "data", "signals.db")


def fetch_metars(conn: sqlite3.Connection, icao: str, event_date: str,
                    start_utc: datetime, end_utc: datetime) -> list[dict]:
    """raw_metar_log rows for (icao, event_date) within the UTC window.
    The bot writes one row per METAR cycle (every ~5 min)."""
    try:
        rows = conn.execute(
            """SELECT cycle_timestamp_utc, temp_c, temp_precision, raw_message
               FROM raw_metar_log
               WHERE icao = ? AND event_date = ?
                 AND cycle_timestamp_utc >= ?
                 AND cycle_timestamp_utc <= ?
                 AND temp_c IS NOT NULL
               ORDER BY cycle_timestamp_utc ASC""",
            (icao, event_date,
              start_utc.isoformat(), end_utc.isoformat()),
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"  ERROR reading raw_metar_log: {e}")
        return []
    out = []
    for ts, t_c, prec, raw in rows:
        out.append({
            "timestamp_utc": ts,
            "temp_c":        float(t_c),
            "temp_f":        float(t_c) * 9.0/5.0 + 32.0,
            "precision":     prec or "whole",
            "raw":           raw or "",
        })
    return out


def fetch_prices(conn: sqlite3.Connection, city: str, event_date: str,
                    bin_label: str,
                    start_utc: datetime, end_utc: datetime) -> list[dict]:
    """paper_predictor_signals rows for (city, event_date, bin) within
    the UTC window.  The bot writes one row per bin per scan."""
    # Bin matching: bot's bin_label is like '94F-95F' or '94-95F' depending
    # on city — use LIKE to be lenient.
    bin_pat = f"%{bin_label}%"
    try:
        rows = conn.execute(
            """SELECT scanned_at_utc, market_prob, our_prob, edge, action,
                       bin_label
               FROM paper_predictor_signals
               WHERE city = ? AND event_date = ?
                 AND bin_label LIKE ?
                 AND scanned_at_utc >= ?
                 AND scanned_at_utc <= ?
                 AND market_prob IS NOT NULL
               ORDER BY scanned_at_utc ASC""",
            (city, event_date, bin_pat,
              start_utc.isoformat(), end_utc.isoformat()),
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"  ERROR reading paper_predictor_signals: {e}")
        return []
    out = []
    for ts, mp, op, edge, action, lbl in rows:
        out.append({
            "scanned_at_utc": ts,
            "market_prob":    float(mp),
            "our_prob":       float(op) if op is not None else None,
            "edge":           float(edge) if edge is not None else None,
            "action":         action or "",
            "bin_label":      lbl or "",
        })
    return out


def main(argv: list[str] | None = None) -> int:
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date",        default=today_iso, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--start-local", default="11:00")
    ap.add_argument("--end-local",   default="16:00")
    ap.add_argument("--tz",          default="America/New_York")
    ap.add_argument("--city",        default="Miami")
    ap.add_argument("--station",     default="KMIA")
    ap.add_argument("--bin",         dest="bin_label", default="94-95",
                       help="bin label substring (LIKE match)")
    ap.add_argument("--db",          default=DEFAULT_DB,
                       help=f"path to signals.db (default: {DEFAULT_DB})")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"FATAL: DB not found at {args.db}", file=sys.stderr)
        return 1

    tz = ZoneInfo(args.tz)
    start_local = datetime.strptime(f"{args.date} {args.start_local}",
                                       "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    end_local   = datetime.strptime(f"{args.date} {args.end_local}",
                                       "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc   = end_local.astimezone(timezone.utc)

    print(f"db:      {args.db}")
    print(f"window:  {start_local.isoformat()}  ->  {end_local.isoformat()}  "
          f"({args.tz})")
    print(f"         {start_utc.isoformat()}  ->  {end_utc.isoformat()}  (UTC)")
    print(f"city:    {args.city}   station: {args.station}   bin: {args.bin_label}")
    print()

    with sqlite3.connect(args.db) as conn:
        metars = fetch_metars(conn, args.station, args.date, start_utc, end_utc)
        prices = fetch_prices(conn, args.city, args.date, args.bin_label,
                                start_utc, end_utc)

    print(f"loaded:  {len(metars)} METAR rows, {len(prices)} predictor-scan rows")
    print()

    # Merge by minute so the two streams align visually.  Different
    # cadences (METAR ~5 min, scans every PREDICTOR_SCAN_MIN min) mean
    # most minutes have only one side; that's fine.
    by_minute: dict[datetime, dict] = {}
    for m in metars:
        ts = datetime.fromisoformat(m["timestamp_utc"].replace("Z", "+00:00"))
        key = ts.replace(second=0, microsecond=0)
        by_minute.setdefault(key, {})["metar"] = m
    for p in prices:
        ts = datetime.fromisoformat(p["scanned_at_utc"].replace("Z", "+00:00"))
        key = ts.replace(second=0, microsecond=0)
        by_minute.setdefault(key, {})["price"] = p

    if not by_minute:
        print("(no rows — check that the bot has been scanning this city today, "
              "and that raw_metar_log has KMIA observations for this date)")
        return 0

    width = 112
    print("=" * width)
    print(f"{'UTC time':<22} {'local':<10} {'temp °F':>9} {'temp °C':>9} "
          f"{'prec':>6} {'mkt p':>8} {'our p':>8} {'edge':>8} {'action':<10}")
    print("-" * width)
    for key in sorted(by_minute):
        row = by_minute[key]
        m = row.get("metar")
        p = row.get("price")
        tf = f"{m['temp_f']:>7.2f}" if m else "      --"
        tc = f"{m['temp_c']:>7.2f}" if m else "      --"
        prec = m["precision"] if m else ""
        mp = f"{p['market_prob']:>6.4f}" if p else "      --"
        op = (f"{p['our_prob']:>6.4f}" if p and p["our_prob"] is not None
              else "      --")
        edge = (f"{p['edge']:>+6.3f}" if p and p["edge"] is not None
                else "      --")
        act = (p["action"] if p else "")[:10]
        local = key.astimezone(tz).strftime("%H:%M:%S")
        print(f"{key.strftime('%Y-%m-%d %H:%M:%S')}  {local:<10} {tf:>9} {tc:>9} "
              f"{prec:>6} {mp:>8} {op:>8} {edge:>8} {act:<10}")
    print("=" * width)
    return 0


if __name__ == "__main__":
    sys.exit(main())