#!/usr/bin/env python3
"""
twc_forecast_probe.py — Phase A of TWC Goal 2 (probabilistic forecast).

Standalone CLI.  For each US (city, event_date), call TWC's
probabilistic hourly forecast endpoint, derive the daily-max
distribution from the prototype ensemble, and print per-bin
probabilities alongside our model and the market.

Pure measurement.  No DB writes.  No live-trading impact.

USAGE
    # All US cities for today
    python bot/scripts/twc_forecast_probe.py

    # Single city
    python bot/scripts/twc_forecast_probe.py --city Miami

    # Specific event date (e.g., tomorrow)
    python bot/scripts/twc_forecast_probe.py --event-date 2026-06-20

WHY PROTOTYPES (NOT discretePdfs OR probabilities)
The TWC docs warn explicitly:
    "Calibration is not available for min(), max(), or sum()
     aggregations of hourly temperatures or other parameters."

The hourly PDFs and per-hour range probabilities are BMA-calibrated,
but combining them across hours to get a daily-max distribution
violates that calibration (the hours aren't independent).

Prototypes preserve temporal correlation — each is a coherent
forecast realization.  Taking max(prototype) per day gives a sample
from the true daily-max distribution, and counting samples per bin
gives a calibration-preserving probability estimate.

REQUIRES
    TWC_API_KEY in .env (or shell env)
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone, date as date_t
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_BOT_DIR), ".env"), override=True)
except ImportError:
    pass

from config import DB_PATH                # type: ignore
from station_meta import CITY_STATIONS    # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("twc_forecast_probe")
# Silence httpx URL-with-apiKey logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


TWC_API_BASE = os.getenv("TWC_API_BASE", "https://api.weather.com")
TWC_API_KEY  = os.getenv("TWC_API_KEY", "")
TWC_PROBABILISTIC_PATH = os.getenv(
    "TWC_PROBABILISTIC_PATH", "/v3/wx/forecast/probabilistic")
TWC_LANGUAGE  = os.getenv("TWC_LANGUAGE", "en-US")
TWC_TIMEOUT_S = float(os.getenv("TWC_TIMEOUT_S", "30"))
N_PROTOTYPES_DEFAULT = int(os.getenv("TWC_N_PROTOTYPES", "50"))
HOURS_DEFAULT        = int(os.getenv("TWC_FORECAST_HOURS", "72"))


# ============================================================
# TWC API
# ============================================================

def _units_for(settlement_unit: str) -> str:
    """TWC units code: 'e' = English (°F), 'm' = Metric (°C)."""
    return "e" if (settlement_unit or "").lower() == "fahrenheit" else "m"


def fetch_probabilistic(icao: str, settlement_unit: str,
                            n_prototypes: int = N_PROTOTYPES_DEFAULT,
                            hours: int = HOURS_DEFAULT) -> dict:
    """Call /v3/wx/forecast/probabilistic for one station and return
    the forecasts1Hour dict (with fcstValid + prototypes).

    We only request `prototypes` — that's what we need for the daily-max
    derivation.  Skipping percentiles / probabilities / discretePdfs
    keeps the response size down."""
    if not TWC_API_KEY:
        raise RuntimeError("TWC_API_KEY env var not set")
    params = {
        "icaoCode":   icao,
        "units":      _units_for(settlement_unit),
        "language":   TWC_LANGUAGE,
        "format":     "json",
        "hours":      hours,
        "prototypes": f"temperature:{n_prototypes}",
        "apiKey":     TWC_API_KEY,
    }
    url = f"{TWC_API_BASE}{TWC_PROBABILISTIC_PATH}"
    resp = httpx.get(url, params=params, timeout=TWC_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"TWC HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("forecasts1Hour", {})


# ============================================================
# Daily-max derivation (prototype ensemble → bin probabilities)
# ============================================================

def derive_daily_max_samples(
    forecasts_1hour: dict, event_date: str, tz_str: str,
) -> tuple[list[float], int]:
    """From forecasts1Hour, return (sample_maxes, n_hours_covered):
      - sample_maxes: one daily-max sample per prototype
      - n_hours_covered: how many hourly slots fell on event_date local
    Filters forecast hours to those whose station-local date matches
    event_date, then takes max per prototype across those hours."""
    fcst_valid = forecasts_1hour.get("fcstValid", [])
    protos = forecasts_1hour.get("prototypes", [])
    if not fcst_valid or not protos:
        return [], 0

    temp_proto = next(
        (p for p in protos if p.get("parameter") == "temperature"), None)
    if not temp_proto:
        return [], 0
    forecasts = temp_proto.get("forecast", [])   # 2D: prototypes × hours
    if not forecasts:
        return [], 0

    tz = ZoneInfo(tz_str)
    target_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    hour_indices: list[int] = []
    for i, ts in enumerate(fcst_valid):
        try:
            dt_local = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tz)
        except (TypeError, ValueError, OSError):
            continue
        if dt_local.date() == target_date:
            hour_indices.append(i)
    if not hour_indices:
        return [], 0

    sample_maxes: list[float] = []
    for proto in forecasts:
        vals = [proto[i] for i in hour_indices if i < len(proto)]
        if vals:
            sample_maxes.append(max(vals))
    return sample_maxes, len(hour_indices)


def _round_half_up(x: float) -> int:
    """Half-up rounding to match Polymarket settlement convention."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def bin_probabilities(sample_maxes: list[float],
                          bins: list[dict]) -> dict[str, float]:
    """For each Polymarket bin, return the fraction of samples whose
    half-up-rounded value falls inside [range_low, range_high].
    Open-ended bins (None on one edge) supported."""
    if not sample_maxes:
        return {}
    n = len(sample_maxes)
    out: dict[str, float] = {}
    for b in bins:
        lo, hi = b.get("range_low"), b.get("range_high")
        cnt = 0
        for v in sample_maxes:
            r = _round_half_up(float(v))
            lo_ok = (lo is None) or (r >= lo)
            hi_ok = (hi is None) or (r <= hi)
            if lo_ok and hi_ok:
                cnt += 1
        out[b["label"]] = cnt / n
    return out


# ============================================================
# Bin lookup (from paper_predictor_signals)
# ============================================================

def fetch_event_bins(conn: sqlite3.Connection,
                          city: str, event_date: str) -> list[dict]:
    """Latest-scan bin set for this (city, event_date), with our_prob
    and market_prob attached so we can print all three side-by-side."""
    try:
        rows = conn.execute(
            """SELECT s.bin_label, s.bin_range_low, s.bin_range_high,
                       s.unit, s.our_prob, s.market_prob
               FROM paper_predictor_signals s
               WHERE s.city = ? AND s.event_date = ?
                 AND s.bin_range_low IS NOT NULL
                 AND s.scanned_at_utc = (
                     SELECT MAX(scanned_at_utc) FROM paper_predictor_signals
                     WHERE city = ? AND event_date = ?
                 )
               ORDER BY s.bin_range_low ASC""",
            (city, event_date, city, event_date)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"label": r[0], "range_low": r[1], "range_high": r[2],
         "unit": r[3], "our_prob": r[4], "market_prob": r[5]}
        for r in rows
    ]


# ============================================================
# Per-city probe
# ============================================================

def probe_city(conn: sqlite3.Connection,
                 city: str, event_date: str) -> dict:
    """Run the probe for one (city, event_date) and print results.
    Returns a small summary dict for the final overview table."""
    meta = CITY_STATIONS.get(city)
    if not meta:
        print(f"\n{city}: no ICAO mapping in station_meta — skipping")
        return {"city": city, "status": "no_icao"}
    icao, _net, tz_str = meta[0], meta[1], meta[2]

    bins = fetch_event_bins(conn, city, event_date)
    if not bins:
        print(f"\n{city} {event_date} ({icao}): no signals in DB — skipping "
              f"(event_date may be outside the scan window)")
        return {"city": city, "status": "no_bins"}

    unit = (bins[0].get("unit") or "fahrenheit").lower()
    unit_sym = "°F" if unit == "fahrenheit" else "°C"

    try:
        fh = fetch_probabilistic(icao, unit)
    except Exception as e:
        print(f"\n{city} {event_date} ({icao}): TWC fetch failed: {e}")
        return {"city": city, "status": "fetch_error", "err": str(e)}

    samples, n_hours = derive_daily_max_samples(fh, event_date, tz_str)
    if not samples:
        print(f"\n{city} {event_date} ({icao}): no prototypes covering "
              f"event_date (n_hours_in_window={n_hours}).  "
              f"Event may be outside the 72h forecast horizon.")
        return {"city": city, "status": "out_of_window"}

    twc_probs = bin_probabilities(samples, bins)

    # Distribution stats from the 50 samples
    sm = sorted(samples)
    def _p(q): return sm[min(len(sm)-1, max(0, int(q * len(sm))))]
    p10, p50, p90 = _p(0.10), _p(0.50), _p(0.90)
    mean = sum(samples) / len(samples)

    # Identify the top-probability bin per source for the verdict line
    def _top(d: dict) -> Optional[str]:
        if not d:
            return None
        return max(d, key=d.get)
    our_top   = _top({b["label"]: (b.get("our_prob") or 0)    for b in bins})
    mkt_top   = _top({b["label"]: (b.get("market_prob") or 0) for b in bins})
    twc_top   = _top(twc_probs)

    print(f"\n{city}  {event_date}  ({icao}, {tz_str})")
    print(f"  TWC: {len(samples)} prototypes × {n_hours} hours of event_date")
    print(f"  TWC daily-max dist: mean={mean:.1f}{unit_sym}  "
          f"P10/P50/P90 = {p10:.1f} / {p50:.1f} / {p90:.1f}{unit_sym}")
    print(f"  top-P bin:  Our={our_top}   Mkt={mkt_top}   TWC={twc_top}")
    print(f"  {'bin':<14} {'Our P':>7} {'Mkt P':>7} {'TWC P':>7} "
          f"{'TWC-Mkt':>9} {'TWC-Our':>9}")
    print(f"  " + "-" * 64)
    for b in bins:
        lbl = b["label"]
        our_p = float(b.get("our_prob")    or 0.0)
        mkt_p = float(b.get("market_prob") or 0.0)
        twc_p = twc_probs.get(lbl, 0.0)
        d_mkt = twc_p - mkt_p
        d_our = twc_p - our_p
        # Mark top-P rows with a *
        marker = ""
        if lbl == twc_top: marker += "*"
        print(f"  {lbl:<14} {our_p*100:>6.1f}% {mkt_p*100:>6.1f}% "
              f"{twc_p*100:>6.1f}% {d_mkt*100:>+7.1f}pp {d_our*100:>+7.1f}pp"
              f" {marker}")
    return {
        "city": city, "status": "ok",
        "twc_top": twc_top, "mkt_top": mkt_top, "our_top": our_top,
        "twc_mean": mean,
    }


# ============================================================
# Main
# ============================================================

def main(argv: Optional[list] = None) -> int:
    today_iso = date_t.today().isoformat()
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=DB_PATH,
                       help="path to signals.db (default: config.DB_PATH)")
    ap.add_argument("--event-date", default=today_iso,
                       help=f"YYYY-MM-DD (default: today = {today_iso})")
    ap.add_argument("--city", default=None,
                       help="single-city filter (default: all US cities)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"FATAL: DB not found at {args.db}", file=sys.stderr)
        return 1
    if not TWC_API_KEY:
        print("FATAL: TWC_API_KEY env var not set.  Add it to .env or export "
              "it, then re-run.", file=sys.stderr)
        return 1

    if args.city:
        cities = [args.city]
    else:
        # All US cities (K-prefixed ICAOs) with mappings
        cities = sorted(
            c for c, m in CITY_STATIONS.items()
            if m and isinstance(m[0], str) and m[0].startswith("K")
        )

    print(f"=== TWC probabilistic forecast probe ===")
    print(f"event_date: {args.event_date}")
    print(f"cities ({len(cities)}): {', '.join(cities)}")
    print(f"horizon: {HOURS_DEFAULT}h forecast, {N_PROTOTYPES_DEFAULT} prototypes per station")

    summaries: list[dict] = []
    with sqlite3.connect(args.db, timeout=30.0) as conn:
        for city in cities:
            s = probe_city(conn, city, args.event_date)
            summaries.append(s)

    # Final overview
    print()
    print("=" * 72)
    print("OVERVIEW")
    print("=" * 72)
    print(f"{'city':<14} {'status':<14} {'TWC top-P':<12} "
          f"{'Mkt top-P':<12} {'agree?':<8}")
    print("-" * 72)
    for s in summaries:
        agree = ""
        if s.get("status") == "ok":
            if s.get("twc_top") == s.get("mkt_top"):
                agree = "yes"
            else:
                agree = "NO"
        print(f"{s['city']:<14} {s.get('status',''):<14} "
              f"{(s.get('twc_top') or '--'):<12} "
              f"{(s.get('mkt_top') or '--'):<12} "
              f"{agree:<8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())