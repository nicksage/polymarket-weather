"""
forecast_rmse_calibration.py — Compute per-city forecast σ for the
intraday predictor by comparing Open-Meteo historical forecasts to the
observed daily max.

For each (city, past_date) in the window:
  observed_max  = max(temp_c) in station_obs.db for that city/date
  forecast_max  = max of Open-Meteo's hourly forecast for that date at
                  the settlement station's lat/lon, pulled from the
                  historical-forecast-api (model's predicted curve, not
                  reanalysis truth)
  residual      = forecast_max - observed_max

Aggregates to per-city stats:
  bias  = mean(residual)        — systematic over-/under-forecast
  mae   = mean(|residual|)
  rmse  = sqrt(mean(residual²)) — what the predictor uses as σ
  n     = number of days

Output:
  data/forecast_calibration.json — keyed by city, used by intraday_predictor

Why per-city?  Forecast skill varies a lot:
  * Coastal LA: high uncertainty (marine layer breaks unpredictably)
  * Continental Chicago: tighter (passing weather systems are well-modeled)
  * Tropical Miami: moderate (convection is hard, but bounded range)

Using one global σ=2.0 under- or over-states confidence depending on city.

Usage:
    cd bot
    python -m scripts.forecast_rmse_calibration                 # last 60d, all US
    python -m scripts.forecast_rmse_calibration --days 90
    python -m scripts.forecast_rmse_calibration --city Dallas   # one city
    python -m scripts.forecast_rmse_calibration --all-cities    # include non-US
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from station_meta import CITY_STATIONS  # type: ignore
try:
    from scripts.find_nearby_stations import US_CITY_STATES  # type: ignore
except Exception:
    US_CITY_STATES = {}

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("calibration")
logging.getLogger("httpx").setLevel(logging.WARNING)

STATION_DB  = os.path.join(_BOT_DIR, "data", "station_obs.db")
OUTPUT_JSON = os.path.join(_BOT_DIR, "data", "forecast_calibration.json")

# Open-Meteo's historical-forecast endpoint — returns what was PREDICTED
# for each past date (not reanalysis truth, which is at archive-api).
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Sanity bounds — clamp σ to a reasonable range so a single bad calibration
# can't completely break the predictor.  Wide enough to allow honest tuning.
MIN_SIGMA_C = 0.8
MAX_SIGMA_C = 4.0
FALLBACK_SIGMA_C = 2.0


def fetch_historical_forecast(lat: float, lon: float, tz: str,
                                start: str, end: str) -> dict[str, float]:
    """Return {date_local (YYYY-MM-DD): forecast_daily_max_c} for each
    date in [start, end] at the given station coords, in local tz."""
    try:
        r = httpx.get(
            HISTORICAL_FORECAST_URL,
            params={
                "latitude":   lat,
                "longitude":  lon,
                "hourly":     "temperature_2m",
                "timezone":   tz,
                "start_date": start,
                "end_date":   end,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"  Open-Meteo historical-forecast failed for ({lat:.2f},{lon:.2f}): {e}")
        return {}

    h = data.get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    by_date: dict[str, float] = {}
    for t, v in zip(times, temps):
        if v is None:
            continue
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
            d_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
        v = float(v)
        if d_str not in by_date or v > by_date[d_str]:
            by_date[d_str] = v
    return by_date


def load_observed_daily_max(db: str, city: str, start: str, end: str
                              ) -> dict[str, float]:
    """{date_local: observed_max_c} for the given city from station_obs.db."""
    out: dict[str, float] = {}
    if not os.path.exists(db):
        return out
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT date_local, tmax_c FROM station_daily_max "
            "WHERE city = ? AND date_local BETWEEN ? AND ? AND tmax_c IS NOT NULL",
            (city, start, end),
        ):
            out[r["date_local"]] = float(r["tmax_c"])
    return out


def compute_stats(residuals: list[float]) -> dict:
    if not residuals:
        return {"n": 0, "bias": None, "mae": None, "rmse": None}
    n = len(residuals)
    bias = sum(residuals) / n
    mae  = sum(abs(r) for r in residuals) / n
    rmse = math.sqrt(sum(r*r for r in residuals) / n)
    return {
        "n":    n,
        "bias": round(bias, 3),
        "mae":  round(mae, 3),
        "rmse": round(rmse, 3),
    }


def calibrate_city(city: str, start: str, end: str) -> dict | None:
    s = CITY_STATIONS.get(city)
    if not s:
        return None
    icao, _net, tz, lat, lon = s

    log.info(f"  {city:<14} {icao}  fetching historical forecast …")
    forecast = fetch_historical_forecast(lat, lon, tz, start, end)
    if not forecast:
        return {"city": city, **compute_stats([]), "reason": "no_forecast"}

    observed = load_observed_daily_max(STATION_DB, city, start, end)
    if not observed:
        return {"city": city, **compute_stats([]), "reason": "no_observed"}

    # Common dates
    common = sorted(set(forecast) & set(observed))
    residuals = [forecast[d] - observed[d] for d in common]
    stats = compute_stats(residuals)

    # Clamp sigma to safe bounds for use as predictor σ
    raw_rmse = stats["rmse"]
    if raw_rmse is None:
        sigma = FALLBACK_SIGMA_C
    else:
        sigma = max(MIN_SIGMA_C, min(MAX_SIGMA_C, raw_rmse))

    # Largest single-day errors (sanity check — outliers worth seeing)
    worst = sorted(
        ((d, forecast[d], observed[d], forecast[d] - observed[d]) for d in common),
        key=lambda x: -abs(x[3]),
    )[:3]

    # === W2 Phase C — centered residuals for empirical-CDF construction ===
    # The per-day residuals (forecast - observed), mean-subtracted so the
    # shape captures asymmetry / fat tails WITHOUT re-encoding the mean
    # bias (which is already handled by station_bias_calibration upstream
    # of the predictor).  Sorted ascending for fast CDF interpolation.
    bias_val = stats.get("bias")
    if residuals and bias_val is not None:
        centered = sorted(round(r - bias_val, 3) for r in residuals)
    else:
        centered = []

    return {
        "city":     city,
        "station":  icao,
        "tz":       tz,
        **stats,
        "sigma":    round(sigma, 3),
        "centered_residuals": centered,
        "worst_days": [
            {"date": d, "forecast": round(f, 2), "observed": round(o, 2),
             "residual": round(r, 2)} for d, f, o, r in worst
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=60,
                   help="Look back N days from today (default: 60)")
    p.add_argument("--start", help="Start YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",   help="End YYYY-MM-DD (default: today)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all US)")
    p.add_argument("--all-cities", action="store_true",
                   help="Calibrate for ALL settlement cities, not just US")
    p.add_argument("--output", default=OUTPUT_JSON,
                   help=f"Output JSON (default: {OUTPUT_JSON})")
    args = p.parse_args()

    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today())
    start_d = (datetime.strptime(args.start, "%Y-%m-%d").date()
               if args.start else end_d - timedelta(days=args.days))
    # End at yesterday — today's data may be incomplete
    end_d = min(end_d, date.today() - timedelta(days=1))
    start, end = start_d.isoformat(), end_d.isoformat()

    if args.city:
        cities = [c for c in args.city if c in CITY_STATIONS]
    elif args.all_cities:
        cities = list(CITY_STATIONS.keys())
    else:
        # Default to US only (matches intraday_predictor scope)
        cities = list(US_CITY_STATES.keys()) if US_CITY_STATES else [
            c for c, m in CITY_STATIONS.items() if m[0].startswith("K")
        ]
        cities = [c for c in cities if c in CITY_STATIONS]

    log.info(f"Window: {start} → {end}  ({(end_d - start_d).days + 1} days)")
    log.info(f"Calibrating {len(cities)} cities")

    results: list[dict] = []
    for i, city in enumerate(cities, 1):
        r = calibrate_city(city, start, end)
        if r is not None:
            results.append(r)
        time.sleep(0.3)   # gentle on Open-Meteo

    # Sort by σ ascending (best-forecast cities first)
    results.sort(key=lambda r: r.get("rmse") or 99)

    print()
    print("=" * 92)
    print(f"  FORECAST CALIBRATION  ({start} → {end})")
    print("=" * 92)
    print(f"  {'city':<16} {'station':<7} {'n':>4} {'bias':>7} {'mae':>6} "
          f"{'rmse':>6} {'σ used':>7}  worst day")
    print("  " + "-" * 88)
    for r in results:
        if r.get("n", 0) == 0:
            print(f"  {r['city']:<16} {r.get('station',''):<7}    0 — — — —  "
                   f"{r.get('reason','')}")
            continue
        worst = r["worst_days"][0] if r["worst_days"] else None
        worst_str = (f"{worst['date']}: forecast {worst['forecast']:.1f} "
                       f"vs obs {worst['observed']:.1f}  ({worst['residual']:+.1f})"
                       if worst else "")
        b = r["bias"]; m = r["mae"]; rm = r["rmse"]; sig = r["sigma"]
        bsign = "+" if b >= 0 else ""
        print(f"  {r['city']:<16} {r['station']:<7} {r['n']:>4d} "
              f"{bsign}{b:>+6.2f}° {m:>5.2f}° {rm:>5.2f}° {sig:>6.2f}°  "
              f"{worst_str}")

    # Write JSON for predictor consumption
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_data = {
        "generated_at":     datetime.now().isoformat(),
        "window":           {"start": start, "end": end},
        "fallback_sigma_c": FALLBACK_SIGMA_C,
        "sigma_min_c":      MIN_SIGMA_C,
        "sigma_max_c":      MAX_SIGMA_C,
        "by_city": {r["city"]: {k: r[k] for k in r
                                  if k not in ("city", "worst_days")}
                      for r in results if r.get("n", 0) > 0},
    }
    # Diagnostic: how many cities have enough samples for the W2 Phase C
    # empirical CDF.  Below the threshold the predictor falls back to
    # the gaussian path even when PREDICTOR_CDF_IMPL=empirical.
    EMPIRICAL_MIN = 30
    enough = sum(1 for c in out_data["by_city"].values()
                  if len(c.get("centered_residuals") or []) >= EMPIRICAL_MIN)
    print(f"  Empirical CDF eligible: {enough}/{len(out_data['by_city'])} cities "
           f"have >= {EMPIRICAL_MIN} centered residuals")
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=2)
    print()
    print(f"  Wrote {len(out_data['by_city'])} city calibrations to {args.output}")
    print(f"  intraday_predictor.py will auto-load this on next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())