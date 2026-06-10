"""
ensemble_forecast.py — Build a multi-station forecast ensemble for each
US settlement city.

Why: a single NWS gridpoint forecast for the settlement station can be
noisy, locally biased, or missing a model revision.  By fetching forecasts
from N nearby ASOS stations (within ~30mi) and computing ensemble stats,
we get:

  * mean / median forecast peak    — a better mu point estimate
  * std deviation across stations  — empirical forecast uncertainty
  * divergence of settlement vs ensemble — flag suspicious forecasts

These are consumed by estimate_day_high_dist via the new `ensemble_stats`
parameter (Phase 1 = sigma inflation, Phase 2 = mu blending toward ensemble).

Data:
  - Neighbor stations come from data/nearby_stations_us.csv (built by
    scripts.find_nearby_stations).  We pick the N nearest ASOS stations
    per settlement city, since RWIS/COOP networks don't have reliable
    NWS gridpoint forecasts.
  - Forecast fetcher: fetch_nws_today_forecast() — already implemented,
    cached, and falls back to Open-Meteo if NWS unreachable.
"""

from __future__ import annotations

import csv
import logging
import os
import statistics
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

log = logging.getLogger("ensemble_forecast")

NEARBY_CSV = os.path.join(_BOT_DIR, "data", "nearby_stations_us.csv")
ENSEMBLE_CACHE_SEC = int(os.getenv("PREDICTOR_ENSEMBLE_CACHE_SEC",
                                     str(int(os.getenv("PREDICTOR_WEATHER_CACHE_SEC", "300")))))
ENSEMBLE_N_NEIGHBORS = int(os.getenv("PREDICTOR_ENSEMBLE_N_NEIGHBORS", "5"))
ENSEMBLE_MAX_RADIUS_MI = float(os.getenv("PREDICTOR_ENSEMBLE_MAX_RADIUS_MI", "40"))

# In-memory cache for neighbor station list per city (loaded once)
_NEIGHBOR_CACHE: dict[str, list[dict]] | None = None
# Cache for ensemble result per city (TTL = ENSEMBLE_CACHE_SEC)
_ENSEMBLE_RESULT_CACHE: dict[str, tuple[float, dict]] = {}


def _load_neighbor_stations() -> dict[str, list[dict]]:
    """Read nearby_stations_us.csv and group by settlement city.
    Filters to ASOS-network stations only (those have reliable NWS
    gridpoint forecasts; RWIS/COOP don't).
    """
    global _NEIGHBOR_CACHE
    if _NEIGHBOR_CACHE is not None:
        return _NEIGHBOR_CACHE
    out: dict[str, list[dict]] = {}
    if not os.path.exists(NEARBY_CSV):
        log.warning(f"{NEARBY_CSV} not found — ensemble will only use settlement station")
        _NEIGHBOR_CACHE = {}
        return _NEIGHBOR_CACHE
    with open(NEARBY_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            net = (row.get("network") or "").upper()
            # ASOS = airport stations with full NWS forecasts.
            # AWOS works too (smaller airports).  Skip RWIS, COOP, etc.
            if "ASOS" not in net and "AWOS" not in net:
                continue
            try:
                dist = float(row.get("distance_mi") or "0")
                lat = float(row.get("lat") or "0")
                lon = float(row.get("lon") or "0")
            except ValueError:
                continue
            if dist > ENSEMBLE_MAX_RADIUS_MI:
                continue
            city = row.get("polymarket_city")
            if not city:
                continue
            out.setdefault(city, []).append({
                "sid":          row.get("sid"),
                "name":         row.get("name"),
                "lat":          lat,
                "lon":          lon,
                "distance_mi":  dist,
                "bearing_deg":  float(row.get("bearing_deg") or "0"),
                "direction":    row.get("direction"),
            })
    # Sort each city's neighbors by distance ascending
    for city in out:
        out[city].sort(key=lambda r: r["distance_mi"])
    _NEIGHBOR_CACHE = out
    return _NEIGHBOR_CACHE


def get_neighbor_stations(city: str, n: int = ENSEMBLE_N_NEIGHBORS,
                            include_settlement: bool = False) -> list[dict]:
    """Return the N nearest ASOS/AWOS neighbor stations for a city.

    include_settlement=False (default) excludes the settlement station
    itself (distance_mi == 0) since the caller already has its forecast.
    Setting True is useful for tests / standalone use.
    """
    nbrs = _load_neighbor_stations().get(city, [])
    if not include_settlement:
        nbrs = [s for s in nbrs if s["distance_mi"] > 0.5]
    return nbrs[:n]


def compute_ensemble_stats(city: str, settlement_forecast: dict,
                             tz_str: str,
                             n: int = ENSEMBLE_N_NEIGHBORS) -> dict:
    """For a given city, fetch forecasts from N nearest neighbor ASOS
    stations + use the provided settlement_forecast, then return ensemble
    statistics.

    Args:
        city: Polymarket city name (matches CITY_STATIONS keys)
        settlement_forecast: dict from fetch_nws_today_forecast for the
            settlement station, with `forecast_high` and `forecast_peak_hour`
        tz_str: timezone (e.g. "America/Chicago")
        n: number of neighbor stations to fetch (default 5)

    Returns:
        {
          n_stations_used:    int (settlement + successful neighbors),
          settlement_high:    float (°C — settlement station's forecast),
          ensemble_mean:      float (°C — mean across all stations),
          ensemble_median:    float (°C — median),
          ensemble_std:       float (°C — std deviation),
          ensemble_min:       float,
          ensemble_max:       float,
          divergence_c:       float (settlement_high - ensemble_median),
          settlement_is_outlier: bool (|divergence| > 1.5°C),
          stations:           list of {sid, distance_mi, forecast_high}
        }
    Returns empty dict if settlement_forecast is missing or no
    neighbor data could be fetched.
    """
    if not settlement_forecast or "forecast_high" not in settlement_forecast:
        return {}

    # Cache hit check
    import time as _time
    now_epoch = _time.time()
    cached = _ENSEMBLE_RESULT_CACHE.get(city)
    if cached and (now_epoch - cached[0]) < ENSEMBLE_CACHE_SEC:
        return cached[1]

    # Import inside function to avoid circular import at module load
    from scripts.intraday_predictor import fetch_nws_today_forecast

    settlement_high = settlement_forecast["forecast_high"]
    highs = [settlement_high]
    station_diag = [{"sid": "SETTLEMENT", "distance_mi": 0.0,
                      "forecast_high": settlement_high}]

    for nb in get_neighbor_stations(city, n=n, include_settlement=False):
        nb_fcst = fetch_nws_today_forecast(nb["lat"], nb["lon"], tz_str)
        if not nb_fcst or "forecast_high" not in nb_fcst:
            continue
        highs.append(nb_fcst["forecast_high"])
        station_diag.append({
            "sid":           nb["sid"],
            "distance_mi":   round(nb["distance_mi"], 1),
            "forecast_high": round(nb_fcst["forecast_high"], 2),
        })

    if len(highs) < 2:
        # Only settlement station — no useful ensemble
        result = {
            "n_stations_used":       1,
            "settlement_high":       settlement_high,
            "ensemble_mean":         settlement_high,
            "ensemble_median":       settlement_high,
            "ensemble_std":          0.0,
            "ensemble_min":          settlement_high,
            "ensemble_max":          settlement_high,
            "divergence_c":          0.0,
            "settlement_is_outlier": False,
            "stations":              station_diag,
        }
    else:
        mean_h   = statistics.fmean(highs)
        median_h = statistics.median(highs)
        std_h    = statistics.stdev(highs) if len(highs) > 1 else 0.0
        divergence = settlement_high - median_h
        result = {
            "n_stations_used":       len(highs),
            "settlement_high":       settlement_high,
            "ensemble_mean":         mean_h,
            "ensemble_median":       median_h,
            "ensemble_std":          std_h,
            "ensemble_min":          min(highs),
            "ensemble_max":          max(highs),
            "divergence_c":          divergence,
            "settlement_is_outlier": abs(divergence) > 1.5,
            "stations":              station_diag,
        }
    _ENSEMBLE_RESULT_CACHE[city] = (now_epoch, result)
    return result


# ---------------------------------------------------------------------------
# Standalone CLI for ad-hoc inspection
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--city", default="Dallas", help="city to inspect")
    p.add_argument("--n", type=int, default=5, help="neighbor stations")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        from station_meta import CITY_STATIONS
    except ImportError as e:
        print(f"cannot import station_meta: {e}"); return 1

    st = CITY_STATIONS.get(args.city)
    if not st:
        print(f"unknown city {args.city}; known: {list(CITY_STATIONS.keys())}")
        return 1
    icao, _net, tz_str, lat, lon = st

    from scripts.intraday_predictor import fetch_nws_today_forecast
    print(f"\n=== Settlement forecast: {args.city} ({icao}) ===")
    settlement = fetch_nws_today_forecast(lat, lon, tz_str)
    if not settlement:
        print("  (NWS returned no forecast)"); return 1
    print(f"  forecast_high      = {settlement['forecast_high']:.1f}°C "
          f"({settlement['forecast_high']*9/5+32:.1f}°F) "
          f"@ {settlement['forecast_peak_hour']}:00")

    print(f"\n=== Neighbor stations within {ENSEMBLE_MAX_RADIUS_MI}mi ===")
    nbrs = get_neighbor_stations(args.city, n=args.n)
    for nb in nbrs:
        print(f"  {nb['sid']:<8}  {nb['distance_mi']:>5.1f}mi  "
              f"{nb['direction']:<3}  {nb['name']}")

    print(f"\n=== Ensemble stats ===")
    stats = compute_ensemble_stats(args.city, settlement, tz_str, n=args.n)
    for k, v in stats.items():
        if k == "stations":
            print(f"  {k}:")
            for s in v:
                print(f"    {s}")
        elif isinstance(v, float):
            print(f"  {k:<22} = {v:.3f}")
        else:
            print(f"  {k:<22} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())