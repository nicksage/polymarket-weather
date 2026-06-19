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
# Synthesize Polymarket-style bins from TWC's forecast distribution.
# Used when no DB bins exist for the (city, event_date) — e.g.,
# future dates where the bot hasn't scanned yet, or events that aren't
# resolved markets.  Lets us still print TWC's view of the day.
# ============================================================

def synthesize_bins_from_samples(
    sample_maxes: list[float], unit: str = "fahrenheit",
) -> list[dict]:
    """Generate Polymarket-style 2°F bins spanning the prototype range.

    For US °F: bins are (lo, lo+1) with lo even, covering
    [floor(P05)-aligned_even .. ceil(P95)-aligned_even+2].  Matches
    Polymarket's '90-91°F' style — each bin covers a 2°F settlement
    window via half-up rounding.

    For non-US: returns 1°C bins."""
    if not sample_maxes:
        return []
    sm = sorted(sample_maxes)
    n = len(sm)
    p05 = sm[max(0, int(0.05 * n))]
    p95 = sm[min(n - 1, int(0.95 * n))]
    # Widen by 2 units on each side so the tails of the distribution
    # are visible, not clipped to the edge bins.
    lo_int = int(p05) - 2
    hi_int = int(p95) + 2

    bins: list[dict] = []
    if (unit or "").lower() == "fahrenheit":
        # Align lo to an even integer (Polymarket US bins are 90-91, 92-93, ...)
        if lo_int % 2 != 0:
            lo_int -= 1
        if hi_int % 2 == 0:
            hi_int += 1
        for lo in range(lo_int, hi_int + 1, 2):
            hi = lo + 1
            bins.append({
                "label":      f"{lo}-{hi}°F",
                "range_low":  float(lo),
                "range_high": float(hi),
                "unit":       "fahrenheit",
                "our_prob":   None,
                "market_prob": None,
            })
    else:
        for c in range(lo_int, hi_int + 1):
            bins.append({
                "label":      f"{c}°C",
                "range_low":  float(c),
                "range_high": float(c),
                "unit":       "celsius",
                "our_prob":   None,
                "market_prob": None,
            })
    return bins


# ============================================================
# Per-city probe
# ============================================================

def probe_city(conn: sqlite3.Connection,
                 city: str, event_date: str,
                 n_prototypes: int = N_PROTOTYPES_DEFAULT) -> dict:
    """Run the probe for one (city, event_date) and print results.
    Returns a small summary dict for the final overview table.

    Two display modes:
      * 'compare' — DB bins exist for this event_date.  Print
                     Our P / Mkt P / TWC P side by side per bin.
      * 'twc_only' — no DB bins.  Synthesize 2°F bins from TWC's
                      forecast range and show TWC P only.  Used for
                      future dates the bot hasn't scanned yet.
    """
    meta = CITY_STATIONS.get(city)
    if not meta:
        print(f"\n{city}: no ICAO mapping in station_meta — skipping")
        return {"city": city, "status": "no_icao"}
    icao, _net, tz_str = meta[0], meta[1], meta[2]

    # Always fetch TWC first so we have a forecast regardless of DB state
    db_bins = fetch_event_bins(conn, city, event_date)
    unit = (db_bins[0].get("unit") if db_bins else "fahrenheit").lower()
    unit_sym = "°F" if unit == "fahrenheit" else "°C"

    try:
        fh = fetch_probabilistic(icao, unit, n_prototypes=n_prototypes)
    except Exception as e:
        print(f"\n{city} {event_date} ({icao}): TWC fetch failed: {e}")
        return {"city": city, "status": "fetch_error", "err": str(e)}

    samples, n_hours = derive_daily_max_samples(fh, event_date, tz_str)
    if not samples:
        print(f"\n{city} {event_date} ({icao}): no prototypes covering "
              f"event_date (n_hours_in_window={n_hours}).  "
              f"Event may be outside the 72h forecast horizon.")
        return {"city": city, "status": "out_of_window"}

    sm = sorted(samples)
    def _p(q): return sm[min(len(sm)-1, max(0, int(q * len(sm))))]
    p10, p50, p90 = _p(0.10), _p(0.50), _p(0.90)
    mean = sum(samples) / len(samples)

    def _top(d: dict) -> Optional[str]:
        if not d: return None
        return max(d, key=d.get)

    # --- Decide display mode ---
    if db_bins:
        # Compare mode: use real Polymarket bins from DB
        bins = db_bins
        mode = "compare"
    else:
        # TWC-only mode: synthesize bins from forecast range
        bins = synthesize_bins_from_samples(samples, unit=unit)
        mode = "twc_only"

    twc_probs = bin_probabilities(samples, bins)
    twc_top   = _top(twc_probs)

    # --- Header (common to both modes) ---
    print(f"\n{city}  {event_date}  ({icao}, {tz_str})  [{mode}]")
    print(f"  TWC: {len(samples)} prototypes × {n_hours} hours of event_date")
    print(f"  TWC daily-max dist: mean={mean:.1f}{unit_sym}  "
          f"P10/P50/P90 = {p10:.1f} / {p50:.1f} / {p90:.1f}{unit_sym}")

    # --- Per-bin table ---
    if mode == "compare":
        our_top = _top({b["label"]: (b.get("our_prob") or 0)    for b in bins})
        mkt_top = _top({b["label"]: (b.get("market_prob") or 0) for b in bins})
        print(f"  top-P bin:  Our={our_top}   Mkt={mkt_top}   TWC={twc_top}")
        print(f"  {'bin':<14} {'Our P':>7} {'Mkt P':>7} {'TWC P':>7} "
              f"{'TWC-Mkt':>9} {'TWC-Our':>9}")
        print(f"  " + "-" * 64)
        for b in bins:
            lbl = b["label"]
            our_p = float(b.get("our_prob")    or 0.0)
            mkt_p = float(b.get("market_prob") or 0.0)
            twc_p = twc_probs.get(lbl, 0.0)
            marker = "*" if lbl == twc_top else ""
            print(f"  {lbl:<14} {our_p*100:>6.1f}% {mkt_p*100:>6.1f}% "
                  f"{twc_p*100:>6.1f}% "
                  f"{(twc_p-mkt_p)*100:>+7.1f}pp "
                  f"{(twc_p-our_p)*100:>+7.1f}pp {marker}")
        return {"city": city, "status": "ok", "mode": mode,
                "twc_top": twc_top, "mkt_top": mkt_top, "our_top": our_top,
                "twc_mean": mean}
    else:
        # TWC-only: just show synthesized bins + TWC P
        print(f"  top-P bin:  TWC={twc_top}  (no DB bins to compare)")
        print(f"  {'bin (synth)':<14} {'TWC P':>7}")
        print(f"  " + "-" * 26)
        for b in bins:
            lbl = b["label"]
            twc_p = twc_probs.get(lbl, 0.0)
            marker = "*" if lbl == twc_top else ""
            # Skip near-zero bins to keep output compact
            if twc_p < 0.005:
                continue
            print(f"  {lbl:<14} {twc_p*100:>6.1f}% {marker}")
        return {"city": city, "status": "ok", "mode": mode,
                "twc_top": twc_top, "mkt_top": None, "our_top": None,
                "twc_mean": mean}


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
    ap.add_argument("--n-prototypes", type=int, default=N_PROTOTYPES_DEFAULT,
                       help=f"how many prototypes TWC returns per station "
                            f"(default: {N_PROTOTYPES_DEFAULT}).  More = "
                            f"tighter Monte Carlo error on per-bin probabilities "
                            f"(±13pp at N=50, ±6pp at N=200) at the "
                            f"cost of larger responses.")
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
    print(f"horizon: {HOURS_DEFAULT}h forecast, {args.n_prototypes} prototypes per station")

    summaries: list[dict] = []
    with sqlite3.connect(args.db, timeout=30.0) as conn:
        for city in cities:
            s = probe_city(conn, city, args.event_date,
                              n_prototypes=args.n_prototypes)
            summaries.append(s)

    # Final overview
    print()
    print("=" * 72)
    print("OVERVIEW")
    print("=" * 72)
    print(f"{'city':<14} {'mode':<10} {'TWC top-P':<12} "
          f"{'Mkt top-P':<12} {'agree?':<8}")
    print("-" * 72)
    for s in summaries:
        mkt_top = s.get("mkt_top")
        twc_top = s.get("twc_top")
        if s.get("status") != "ok":
            agree = ""
        elif mkt_top is None:
            agree = "n/a"   # twc_only mode — nothing to compare
        elif twc_top == mkt_top:
            agree = "yes"
        else:
            agree = "NO"
        mode = s.get("mode") or s.get("status", "")
        print(f"{s['city']:<14} {mode:<10} "
              f"{(twc_top or '--'):<12} "
              f"{(mkt_top or '--'):<12} "
              f"{agree:<8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())