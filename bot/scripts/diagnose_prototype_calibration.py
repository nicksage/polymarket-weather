"""
diagnose_prototype_calibration.py — Phase 2 of the cold-bias plan.

QUESTION
========
TWC's docs say their per-hour temperature distributions are BMA-calibrated.
The docs do NOT explicitly say which pool the `prototypes` are drawn from.
If prototypes come from the CALIBRATED distribution, our prototype-derived
daily-max bin probabilities have correct tails.  If they come from a
DIFFERENT (uncalibrated) pool, our settlement-edge bin probabilities are
biased — silently.

METHOD
======
For each city, request prototypes AND fine-grained percentiles in ONE
call.  For each forecast hour, compare:
  - empirical percentile from the 100 prototype values at that hour
  - TWC's calibrated percentile at that hour

If they agree across all hours and percentile points within ~0.5°F (the
sampling-noise floor for N=100), prototypes are calibrated.  If we see
systematic bias (mean gap consistently non-zero, especially in tails),
they're not, and copula-MC becomes mandatory.

OUTPUT
======
Per-city: max gap, mean gap, sign-of-bias indicator, verdict
Overall verdict at the end: "calibrated" / "uncalibrated" / "mixed"

USAGE
=====
    cd bot
    python scripts/diagnose_prototype_calibration.py
    python scripts/diagnose_prototype_calibration.py --scope all
    python scripts/diagnose_prototype_calibration.py --city Miami
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from typing import Optional

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

from station_meta import CITY_STATIONS    # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("calib_diag")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


TWC_API_BASE = os.getenv("TWC_API_BASE", "https://api.weather.com")
TWC_API_KEY  = os.getenv("TWC_API_KEY", "")
TWC_PROBABILISTIC_PATH = os.getenv(
    "TWC_PROBABILISTIC_PATH", "/v3/wx/forecast/probabilistic")
TWC_LANGUAGE   = os.getenv("TWC_LANGUAGE", "en-US")
TWC_TIMEOUT_S  = float(os.getenv("TWC_TIMEOUT_S", "30"))

# Percentile points we request from TWC.  Same set is used for the
# empirical comparison from prototypes.  Wider spread in the tails
# (P5/P10/P90/P95) and tighter in the body (P25/P50/P75).
PERCENTILE_POINTS = [5, 10, 25, 50, 75, 90, 95]

# Agreement thresholds (in the requested settlement unit, °F or °C).
# Calibration is the question; sampling noise is the floor.
#   N=100 samples gives ~0.5°F SE on tail percentiles for typical daily
#   temperature distributions, so anything within ±1.0°F is plausible
#   noise.  Body percentiles have tighter SE (~0.3°F), so we expect
#   tighter agreement there.
GAP_OK_BODY    = 0.5    # P25/P50/P75
GAP_OK_TAILS   = 1.0    # P5/P10/P90/P95
SYSTEMATIC_BIAS_THRESHOLD = 0.3   # mean gap above this = real bias, not noise


# ============================================================
# TWC API
# ============================================================

def _units_for(settlement_unit: str) -> str:
    return "e" if (settlement_unit or "").lower() == "fahrenheit" else "m"


def is_domestic_icao(icao: str) -> bool:
    return bool(icao) and icao.upper().startswith("K")


def default_settlement_unit_for_icao(icao: str) -> str:
    return "fahrenheit" if is_domestic_icao(icao) else "celsius"


def filter_cities_by_scope(scope: str) -> list[str]:
    s = (scope or "all").lower()
    out = []
    for city, meta in CITY_STATIONS.items():
        if not meta or not isinstance(meta[0], str):
            continue
        icao = meta[0]
        dom = is_domestic_icao(icao)
        if s == "domestic" and dom:        out.append(city)
        elif s == "international" and not dom: out.append(city)
        elif s == "all":                    out.append(city)
    return sorted(out)


def fetch_probabilistic_with_percentiles(
    icao: str, settlement_unit: str, hours: int = 72,
    n_prototypes: int = 100,
) -> dict:
    """Single call returning both prototypes and calibrated percentiles."""
    if not TWC_API_KEY:
        raise RuntimeError("TWC_API_KEY not set")
    pp = ":".join(str(p) for p in PERCENTILE_POINTS)
    params = {
        "icaoCode":    icao,
        "units":       _units_for(settlement_unit),
        "language":    TWC_LANGUAGE,
        "format":      "json",
        "hours":       hours,
        "prototypes":  f"temperature:{n_prototypes}",
        "percentiles": f"temperature:{pp}",
        "apiKey":      TWC_API_KEY,
    }
    url = f"{TWC_API_BASE}{TWC_PROBABILISTIC_PATH}"
    resp = httpx.get(url, params=params, timeout=TWC_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(f"TWC HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("forecasts1Hour", {})


# ============================================================
# Empirical percentile from prototype samples
# ============================================================

def empirical_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (p in [0, 100]) from a list of
    samples.  Linear interpolation between adjacent ranks — the
    closest-match to TWC's continuous percentile semantics."""
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return sorted_v[0]
    # linear interpolation between order statistics
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


# ============================================================
# Per-city diagnostic
# ============================================================

def diagnose_city(city: str) -> dict:
    """Fetch, compute per-(hour, percentile) gaps, summarize."""
    meta = CITY_STATIONS.get(city)
    if not meta:
        return {"city": city, "status": "no_icao"}
    icao = meta[0]
    unit = default_settlement_unit_for_icao(icao)
    unit_sym = "°F" if unit == "fahrenheit" else "°C"

    try:
        fh = fetch_probabilistic_with_percentiles(icao, unit)
    except Exception as e:
        return {"city": city, "icao": icao, "status": "fetch_error",
                "error": str(e)}

    # Find the temperature blocks for both prototypes and percentiles
    protos = next((p for p in fh.get("prototypes", [])
                    if p.get("parameter") == "temperature"), None)
    percentiles = next((p for p in fh.get("percentiles", [])
                          if p.get("parameter") == "temperature"), None)
    if not protos or not percentiles:
        return {"city": city, "icao": icao, "status": "missing_section",
                "have_prototypes": bool(protos),
                "have_percentiles": bool(percentiles)}

    proto_forecasts = protos.get("forecast", [])     # [N prototypes][H hours]
    pct_points      = percentiles.get("percentilePoints", [])
    pct_values      = percentiles.get("percentileValues", [])    # [H hours][len(pct_points)]

    if not proto_forecasts or not pct_values:
        return {"city": city, "icao": icao, "status": "empty_blocks"}

    n_prototypes = len(proto_forecasts)
    n_hours      = len(pct_values)
    if n_hours == 0:
        return {"city": city, "icao": icao, "status": "no_hours"}

    # Build column-per-hour view of prototypes: [hour h][N prototype values]
    proto_by_hour = []
    for h in range(n_hours):
        col = [proto_forecasts[i][h] for i in range(n_prototypes)
                 if h < len(proto_forecasts[i])]
        proto_by_hour.append(col)

    # Compare per (hour, percentile_point)
    pairs = []
    for h in range(n_hours):
        twc_row = pct_values[h]
        if len(twc_row) != len(pct_points):
            continue
        for pi, p in enumerate(pct_points):
            twc_val = twc_row[pi]
            emp_val = empirical_percentile(proto_by_hour[h], float(p))
            gap = emp_val - twc_val
            tail = (p <= 10 or p >= 90)
            threshold = GAP_OK_TAILS if tail else GAP_OK_BODY
            pairs.append({
                "hour": h, "percentile": float(p),
                "twc": twc_val, "empirical": emp_val,
                "gap": gap, "abs_gap": abs(gap),
                "is_tail": tail,
                "within_threshold": abs(gap) <= threshold,
            })

    if not pairs:
        return {"city": city, "icao": icao, "status": "no_pairs"}

    abs_gaps     = [p["abs_gap"] for p in pairs]
    signed_gaps  = [p["gap"]      for p in pairs]
    tail_gaps    = [p["gap"]      for p in pairs if p["is_tail"]]
    body_gaps    = [p["gap"]      for p in pairs if not p["is_tail"]]
    n_within     = sum(1 for p in pairs if p["within_threshold"])

    max_pair = max(pairs, key=lambda p: p["abs_gap"])

    return {
        "city": city, "icao": icao, "unit": unit, "unit_sym": unit_sym,
        "status": "ok",
        "n_hours": n_hours,
        "n_pairs": len(pairs),
        "n_within_threshold": n_within,
        "pct_within": n_within / len(pairs),
        "max_abs_gap":     max(abs_gaps),
        "mean_abs_gap":    statistics.mean(abs_gaps),
        "mean_signed_gap": statistics.mean(signed_gaps),
        "mean_tail_gap":   statistics.mean(tail_gaps) if tail_gaps else None,
        "mean_body_gap":   statistics.mean(body_gaps) if body_gaps else None,
        "worst_pair": max_pair,
    }


# ============================================================
# Reporting
# ============================================================

def _city_verdict(r: dict) -> str:
    """Per-city classification:
      ✓ calibrated   — all gaps within threshold AND mean bias near zero
      ⚠ noisy        — most within threshold but mean bias > threshold
      ✗ uncalibrated — material fraction outside threshold OR strong bias"""
    pct = r["pct_within"]
    mean_signed = abs(r["mean_signed_gap"])
    if pct >= 0.95 and mean_signed < SYSTEMATIC_BIAS_THRESHOLD:
        return "✓ calibrated"
    elif pct >= 0.80 or mean_signed < SYSTEMATIC_BIAS_THRESHOLD * 2:
        return "⚠ noisy"
    else:
        return "✗ uncalibrated"


def print_per_city_table(results: list[dict]) -> None:
    print()
    print("=" * 96)
    print(f"PER-CITY CALIBRATION (target: all gaps within ±{GAP_OK_BODY}°F body / "
          f"±{GAP_OK_TAILS}°F tails)")
    print("=" * 96)
    print(f"{'city':<14} {'unit':>4} {'pairs':>6} {'within%':>8} "
          f"{'max gap':>9} {'mean |gap|':>11} {'mean signed':>12} "
          f"{'verdict':<18}")
    print("-" * 96)
    for r in results:
        if r["status"] != "ok":
            print(f"{r['city']:<14} {'?':>4} {'--':>6} {'--':>8} "
                  f"{'--':>9} {'--':>11} {'--':>12} "
                  f"  {r['status']}")
            continue
        sym = r["unit_sym"]
        print(f"{r['city']:<14} {sym:>4} {r['n_pairs']:>6} "
              f"{r['pct_within']*100:>7.1f}% "
              f"{r['max_abs_gap']:>+8.2f} "
              f"{r['mean_abs_gap']:>+10.2f} "
              f"{r['mean_signed_gap']:>+11.2f} "
              f"  {_city_verdict(r):<18}")


def print_worst_pairs(results: list[dict]) -> None:
    print()
    print("=" * 96)
    print("WORST DISAGREEMENT PER CITY (single highest |gap| pair)")
    print("=" * 96)
    print(f"{'city':<14} {'hour':>5} {'P':>5} {'TWC':>8} "
          f"{'empirical':>10} {'gap':>8}")
    print("-" * 70)
    for r in results:
        if r["status"] != "ok":
            continue
        w = r["worst_pair"]
        sym = r["unit_sym"]
        print(f"{r['city']:<14} {w['hour']:>5} {int(w['percentile']):>4}% "
              f"{w['twc']:>+7.1f}{sym} "
              f"{w['empirical']:>+8.1f}{sym} "
              f"{w['gap']:>+7.2f}{sym}")


def print_overall_verdict(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("\nNo cities returned usable data — can't issue a verdict.")
        return
    n_calib  = sum(1 for r in ok if _city_verdict(r).startswith("✓"))
    n_noisy  = sum(1 for r in ok if _city_verdict(r).startswith("⚠"))
    n_uncal  = sum(1 for r in ok if _city_verdict(r).startswith("✗"))

    # Aggregate bias direction
    all_signed = [r["mean_signed_gap"] for r in ok]
    overall_mean_signed = statistics.mean(all_signed) if all_signed else 0
    all_tail = [r["mean_tail_gap"] for r in ok if r.get("mean_tail_gap") is not None]
    overall_tail_signed = statistics.mean(all_tail) if all_tail else 0

    print()
    print("=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)
    print(f"cities tested: {len(ok)}")
    print(f"  ✓ calibrated   : {n_calib}")
    print(f"  ⚠ noisy        : {n_noisy}")
    print(f"  ✗ uncalibrated : {n_uncal}")
    print(f"aggregate signed mean gap (all percentiles): {overall_mean_signed:+.3f}")
    print(f"aggregate signed mean gap (tails only)     : {overall_tail_signed:+.3f}")
    print()
    if n_uncal == 0 and abs(overall_mean_signed) < SYSTEMATIC_BIAS_THRESHOLD:
        print("=> PROTOTYPES ARE CALIBRATED.")
        print("   Our prototype-derived daily-max bin probabilities have")
        print("   correct tails.  Copula MC remains a NICE-TO-HAVE for")
        print("   noise reduction (N=100 → smoother output), not a")
        print("   correctness fix.")
    elif n_uncal == 0:
        print("=> MOSTLY CALIBRATED but with a small systematic bias.")
        print(f"   Direction: prototypes run {('hot' if overall_mean_signed > 0 else 'cold')} "
              f"by {abs(overall_mean_signed):.2f} on average.")
        print("   Copula MC would correct this — but the bias is small")
        print("   enough that pragmatic decision: tolerate and move on, OR")
        print("   add a constant bias-correction term if you want to be tight.")
    else:
        print("=> PROTOTYPES ARE NOT CALIBRATED.")
        print("   Material disagreement with TWC's calibrated marginals.")
        print("   Daily-max bin probabilities at settlement edges are biased.")
        print("   COPULA MC IS MANDATORY — use TWC's calibrated marginals")
        print("   as the source of truth, take the rank correlation from")
        print("   prototypes, draw correlated paths, take max.  Phase 3.")


# ============================================================
# Main
# ============================================================

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", choices=["domestic", "international", "all"],
                       default="domestic",
                       help="which stations to test (default: domestic; "
                            "use 'all' for the full 47-city set)")
    ap.add_argument("--city", default=None,
                       help="single-city override")
    args = ap.parse_args(argv)

    if not TWC_API_KEY:
        print("FATAL: TWC_API_KEY not set in env.", file=sys.stderr)
        return 1

    if args.city:
        cities = [args.city]
    else:
        cities = filter_cities_by_scope(args.scope)

    print(f"=== Prototype calibration diagnostic ===")
    print(f"scope: {args.scope}; cities ({len(cities)}): {', '.join(cities)}")
    print(f"per call: hours=72, prototypes=100, "
          f"percentiles={','.join(str(p) for p in PERCENTILE_POINTS)}")
    print(f"agreement thresholds: ±{GAP_OK_BODY} body / ±{GAP_OK_TAILS} tails "
          f"(in settlement units)")

    results: list[dict] = []
    for city in cities:
        log.info(f"  → {city}")
        r = diagnose_city(city)
        results.append(r)

    print_per_city_table(results)
    print_worst_pairs(results)
    print_overall_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())