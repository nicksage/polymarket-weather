#!/usr/bin/env python3
"""
fetch_miami_history.py — pull Miami temperature observations (KMIA METAR
via NWS) and Polymarket 94-95F YES price history over a local-time
window.  Standalone script, only stdlib + requests if available.

Defaults to today, 11:00–16:00 Eastern, 94-95F bin.  Override via flags:

    python scripts/fetch_miami_history.py \
        --date 2026-06-16 --start-local 11:00 --end-local 16:00 \
        --bin 94-95 --city Miami --station KMIA --tz America/New_York

Output:
  - prints both series to stdout, aligned to nearest minute
  - writes two CSVs alongside this script (temps + prices)

The bin lookup hits Polymarket Gamma + CLOB.  No auth needed; both
endpoints are public-read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
NWS_API   = "https://api.weather.gov"


# ---------------------------------------------------------------------------
# HTTP helper — stdlib only so this runs anywhere
# ---------------------------------------------------------------------------

def http_get(url: str, params: dict | None = None,
                accept: str = "application/json") -> dict | list:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"Accept": accept,
                  "User-Agent": "polymarket-weather/fetch_miami_history (research)"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Polymarket — find the daily-high event and pull the bin's token ID
# ---------------------------------------------------------------------------

def find_yes_token_id(city: str, date_iso: str, bin_label: str) -> str:
    """Search Gamma for the daily-high event matching (city, date) and
    return the yes_token_id for the bin whose title contains `bin_label`."""
    # Gamma slug pattern: "highest-temperature-in-<city>-on-<month>-<day>-<year>"
    month_name = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%B").lower()
    day_num    = datetime.strptime(date_iso, "%Y-%m-%d").day
    year_num   = datetime.strptime(date_iso, "%Y-%m-%d").year
    city_slug  = city.lower().replace(" ", "-")
    slug_hint  = f"highest-temperature-in-{city_slug}-on-{month_name}-{day_num}-{year_num}"

    # Search Gamma by slug substring.  Gamma's `slug` filter is exact;
    # fall back to a tag/keyword search if exact miss.
    events = http_get(f"{GAMMA_API}/events", params={"slug": slug_hint})
    if not events:
        # Loosen — search by title text
        events = http_get(f"{GAMMA_API}/events",
                            params={"limit": 50, "active": "true",
                                     "tag_slug": "weather"})
        events = [e for e in events
                   if city.lower() in (e.get("title", "") + " " + e.get("slug", "")).lower()
                   and date_iso in json.dumps(e)]
    if not events:
        raise RuntimeError(f"No Gamma event found for {city} {date_iso}")
    event = events[0]
    markets = event.get("markets") or []

    # Pick the market (sub-question) whose groupItemTitle contains the bin label
    target = None
    for m in markets:
        gtitle = (m.get("groupItemTitle") or m.get("question") or "")
        if bin_label in gtitle:
            target = m
            break
    if not target:
        # Print what's available for debugging
        avail = [m.get("groupItemTitle") or m.get("question") for m in markets]
        raise RuntimeError(
            f"No bin matching {bin_label!r} on event {event.get('slug')}. "
            f"Available: {avail}"
        )

    # Token IDs sit in `clobTokenIds` as a JSON-encoded list [YES_id, NO_id]
    raw = target.get("clobTokenIds") or "[]"
    if isinstance(raw, str):
        token_ids = json.loads(raw)
    else:
        token_ids = list(raw)
    if not token_ids:
        raise RuntimeError(f"No CLOB token IDs on market {target.get('id')}")
    return str(token_ids[0])    # YES side


def fetch_prices_history(token_id: str, start_ts: int, end_ts: int,
                            fidelity_min: int = 1) -> list[dict]:
    """CLOB prices-history endpoint.  Returns list of {t: unix_seconds, p: price}."""
    data = http_get(
        f"{CLOB_API}/prices-history",
        params={"market": token_id,
                 "startTs": start_ts, "endTs": end_ts,
                 "fidelity": fidelity_min},
    )
    history = data.get("history") if isinstance(data, dict) else data
    return history or []


# ---------------------------------------------------------------------------
# NWS — METAR observations for the station within the window
# ---------------------------------------------------------------------------

def fetch_metar_observations(station: str,
                                  start_iso: str, end_iso: str) -> list[dict]:
    """NWS station observations endpoint.  Returns GeoJSON features."""
    data = http_get(
        f"{NWS_API}/stations/{station}/observations",
        params={"start": start_iso, "end": end_iso, "limit": 500},
    )
    return data.get("features") or []


def parse_metar_feature(feat: dict) -> dict | None:
    props = feat.get("properties") or {}
    ts = props.get("timestamp")
    temp = (props.get("temperature") or {}).get("value")
    if ts is None or temp is None:
        return None
    raw = props.get("rawMessage") or ""
    # T-group precision detection: "T00" remarks group encodes tenths.
    is_t_group = " T0" in raw or " T1" in raw
    return {
        "timestamp_utc": ts,
        "temp_c":        float(temp),
        "temp_f":        float(temp) * 9.0/5.0 + 32.0,
        "precision":     "tenths" if is_t_group else "whole",
        "raw":           raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date",        default=today_iso, help="YYYY-MM-DD (default today UTC)")
    ap.add_argument("--start-local", default="11:00",   help="HH:MM in local TZ")
    ap.add_argument("--end-local",   default="16:00",   help="HH:MM in local TZ")
    ap.add_argument("--tz",          default="America/New_York")
    ap.add_argument("--city",        default="Miami")
    ap.add_argument("--station",     default="KMIA")
    ap.add_argument("--bin",         dest="bin_label", default="94-95",
                       help="bin label substring, e.g. '94-95'")
    ap.add_argument("--fidelity-min", type=int, default=1,
                       help="CLOB price-history granularity in minutes")
    ap.add_argument("--out-dir",     default=os.path.dirname(os.path.abspath(__file__)),
                       help="CSVs go here (default = same dir as this script)")
    args = ap.parse_args(argv)

    tz = ZoneInfo(args.tz)
    start_local = datetime.strptime(f"{args.date} {args.start_local}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    end_local   = datetime.strptime(f"{args.date} {args.end_local}",   "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc   = end_local.astimezone(timezone.utc)

    print(f"window:  {start_local.isoformat()}  ->  {end_local.isoformat()}  "
          f"({args.tz})")
    print(f"         {start_utc.isoformat()}  ->  {end_utc.isoformat()}  (UTC)")
    print(f"city:    {args.city}   station: {args.station}   bin: {args.bin_label}")
    print()

    # ----- Polymarket -----
    print("[1/2] Polymarket: finding event + token id …")
    try:
        token_id = find_yes_token_id(args.city, args.date, args.bin_label)
        print(f"      yes_token_id: {token_id[:16]}…{token_id[-8:]}")
        prices = fetch_prices_history(token_id,
                                          int(start_utc.timestamp()),
                                          int(end_utc.timestamp()),
                                          fidelity_min=args.fidelity_min)
        print(f"      {len(prices)} price points returned")
    except Exception as e:
        print(f"      ERROR: {e}")
        prices = []

    # ----- NWS METAR -----
    print("[2/2] NWS: pulling KMIA observations …")
    try:
        feats = fetch_metar_observations(
            args.station,
            start_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
            end_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        metars = [m for m in (parse_metar_feature(f) for f in feats) if m]
        metars.sort(key=lambda r: r["timestamp_utc"])
        print(f"      {len(metars)} observations parsed")
    except Exception as e:
        print(f"      ERROR: {e}")
        metars = []

    # ----- Print combined timeline -----
    print()
    print("=" * 96)
    print(f"{'UTC time':<22} {'local':<12} {'temp °F':>9} {'temp °C':>9} "
          f"{'precision':>10} {'YES price':>11}")
    print("-" * 96)

    # Merge: every minute in the window, look up nearest METAR (within ±15min)
    # and nearest price (within ±2*fidelity min).  Print only minutes that
    # had at least one of the two.
    by_minute: dict[datetime, dict] = {}
    for m in metars:
        ts = datetime.fromisoformat(m["timestamp_utc"].replace("Z", "+00:00"))
        key = ts.replace(second=0, microsecond=0)
        by_minute.setdefault(key, {})["metar"] = m
    for p in prices:
        ts = datetime.fromtimestamp(int(p.get("t", 0)), tz=timezone.utc)
        key = ts.replace(second=0, microsecond=0)
        by_minute.setdefault(key, {})["price"] = float(p.get("p", 0))

    for key in sorted(by_minute):
        row = by_minute[key]
        m = row.get("metar")
        p = row.get("price")
        tf = f"{m['temp_f']:>7.2f}" if m else "      --"
        tc = f"{m['temp_c']:>7.2f}" if m else "      --"
        prec = m["precision"] if m else ""
        px = f"{p:>9.4f}" if p is not None else "       --"
        local = key.astimezone(tz).strftime("%H:%M:%S")
        print(f"{key.strftime('%Y-%m-%d %H:%M:%S')}  {local:<12} {tf:>9} {tc:>9} "
              f"{prec:>10} {px:>11}")

    # ----- CSV writes -----
    if metars:
        path = os.path.join(args.out_dir,
                              f"miami_{args.date}_metar.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["timestamp_utc", "temp_c",
                                                   "temp_f", "precision", "raw"])
            w.writeheader()
            w.writerows(metars)
        print(f"\nwrote {len(metars)} rows -> {path}")
    if prices:
        path = os.path.join(args.out_dir,
                              f"miami_{args.date}_prices_{args.bin_label}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp_utc", "yes_price"])
            for p in prices:
                ts = datetime.fromtimestamp(int(p.get("t", 0)), tz=timezone.utc)
                w.writerow([ts.isoformat(), float(p.get("p", 0))])
        print(f"wrote {len(prices)} rows -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())