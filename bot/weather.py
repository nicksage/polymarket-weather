"""
weather.py — Multi-source weather probability engine

Each public function returns:
    {"source": str, "probability": float, "confidence": float}
or None on failure.

`get_ensemble_probability()` aggregates all sources into a single weighted
probability and is the only function called by edge.py.

Caching: forecasts are cached by (lat, lon, date, variable) for the duration
of one scan cycle. Call clear_forecast_cache() at the start of each scan.
"""
import os
import re
import logging
import statistics
from datetime import date, datetime
from tenacity import retry, stop_after_attempt, wait_exponential

import httpx

from config import BRIER_WEIGHTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level forecast cache
# Key: (round(lat,3), round(lon,3), date_str, variable)
# Cleared at the start of each scan cycle by clear_forecast_cache()
# ---------------------------------------------------------------------------
_forecast_cache: dict[tuple, dict | None] = {}


def clear_forecast_cache() -> None:
    """Call once at the start of each edge scan to reset the per-cycle cache."""
    _forecast_cache.clear()
    logger.debug("Forecast cache cleared")


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _to_celsius(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit == "fahrenheit":
        return (value - 32) * 5 / 9
    return float(value)  # already celsius or dimensionless


def _to_mm(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit == "inches":
        return value * 25.4
    return float(value)  # already mm


# ---------------------------------------------------------------------------
# NWS US-only guard
# ---------------------------------------------------------------------------
_US_LAT = (15.0, 72.0)
_US_LON = (-180.0, -60.0)


def _is_us_coordinate(lat: float, lon: float) -> bool:
    """NWS covers CONUS, Alaska, Hawaii, Puerto Rico, Guam — nothing outside."""
    return _US_LAT[0] <= lat <= _US_LAT[1] and _US_LON[0] <= lon <= _US_LON[1]


# ---------------------------------------------------------------------------
# NOAA CDO — historical base rate
# ---------------------------------------------------------------------------
NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _noaa_get(path: str, params: dict, headers: dict) -> dict:
    resp = httpx.get(f"{NOAA_BASE}{path}", params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _find_nearest_noaa_station(lat: float, lon: float) -> str | None:
    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GHCND",
        "extent": f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}",
        "limit": 1,
        "sortfield": "name",
    }
    try:
        results = _noaa_get("/stations", params, headers).get("results", [])
        return results[0]["id"] if results else None
    except Exception as e:
        logger.debug(f"NOAA station lookup failed: {e}")
        return None


def _get_noaa_historical(station_id: str, date_str: str, datatype: str) -> float | None:
    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": datatype,
        "startdate": date_str,
        "enddate": date_str,
        "units": "standard",
        "limit": 1,
    }
    try:
        results = _noaa_get("/data", params, headers).get("results", [])
        return results[0]["value"] if results else None
    except Exception:
        return None


def _get_historical_values_for_day(
    station_id: str, datatype: str, month_day: str, years: int = 10
) -> list[float]:
    current_year = date.today().year
    values = []
    for year in range(current_year - years, current_year):
        val = _get_noaa_historical(station_id, f"{year}-{month_day}", datatype)
        if val is not None:
            values.append(float(val))
    return values


def get_noaa_probability(
    lat: float, lon: float, date_str: str, variable: str,
    threshold: float | None = None, unit: str | None = None,
) -> dict | None:
    """
    Historical base rate from NOAA GHCND daily summaries (10-year window).

    NOAA stores values in tenths of a unit:
      PRCP/SNOW → tenths of mm   (e.g. 254 = 25.4 mm = 1 inch)
      TMAX/TMIN → tenths of °C   (e.g. 290 = 29.0°C)

    Returns P(observed_value >= threshold) over the historical sample.
    """
    if not NOAA_TOKEN:
        return None

    datatype_map = {"rain": "PRCP", "snow": "SNOW", "temp_high": "TMAX", "temp_low": "TMIN"}
    datatype = datatype_map.get(variable)
    if not datatype:
        return None

    station_id = _find_nearest_noaa_station(lat, lon)
    if not station_id:
        return None

    target_date = date.fromisoformat(date_str)
    month_day = f"{target_date.month:02d}-{target_date.day:02d}"

    historical = _get_historical_values_for_day(station_id, datatype, month_day, years=10)
    if not historical:
        return None

    if variable in ("rain", "snow"):
        # NOAA PRCP/SNOW in tenths of mm
        threshold_mm = _to_mm(threshold, unit) if threshold is not None else 0.1
        threshold_tenths = threshold_mm * 10
        prob = sum(1 for v in historical if v >= threshold_tenths) / len(historical)
    else:
        # NOAA TMAX/TMIN in tenths of °C
        threshold_c = _to_celsius(threshold, unit)
        if threshold_c is None:
            return None
        threshold_tenths = threshold_c * 10
        if variable == "temp_high":
            prob = sum(1 for v in historical if v >= threshold_tenths) / len(historical)
        else:
            prob = sum(1 for v in historical if v <= threshold_tenths) / len(historical)

    logger.debug(f"NOAA [{variable}] lat={lat:.2f} lon={lon:.2f} threshold={threshold}{unit} "
                 f"→ prob={prob:.3f} (n={len(historical)})")
    return {"source": "noaa", "probability": round(prob, 4), "confidence": 0.5}


# ---------------------------------------------------------------------------
# Open-Meteo — ensemble + deterministic fallback
# ---------------------------------------------------------------------------
OPENMETEO_BASE     = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Map our variable names to Open-Meteo daily parameter names
_OM_PARAM_MAP = {
    "rain":      "precipitation_sum",
    "snow":      "snowfall_sum",
    "temp_high": "temperature_2m_max",
    "temp_low":  "temperature_2m_min",
}


def _extract_member_values(daily: dict, param: str) -> list[float]:
    """
    Extract per-member forecast values from Open-Meteo ensemble response.

    The ensemble API returns keys like:
        "temperature_2m_max_member01": [21.3]
        "temperature_2m_max_member02": [19.8]
        ...
    This collects index-0 of each member array for a single-day request.
    """
    prefix = f"{param}_member"
    values = []
    for key, val_list in daily.items():
        if key.startswith(prefix) and isinstance(val_list, list) and val_list:
            v = val_list[0]
            if v is not None:
                values.append(float(v))
    return values


def get_openmeteo_probability(
    lat: float, lon: float, date_str: str, variable: str,
    threshold: float | None = None, unit: str | None = None,
) -> dict | None:
    """
    Probability from Open-Meteo ECMWF ensemble (39 members).

    Computes P(member_value >= threshold) across all ensemble members.
    Falls back to deterministic forecast if ensemble returns no member data.
    """
    param = _OM_PARAM_MAP.get(variable)
    if not param:
        return None

    try:
        resp = httpx.get(
            OPENMETEO_ENSEMBLE,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": param,
                "start_date": date_str,
                "end_date": date_str,
                "models": "ecmwf_ifs04",
            },
            timeout=12,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
    except httpx.HTTPError:
        return _get_openmeteo_deterministic(lat, lon, date_str, variable, threshold, unit)

    member_values = _extract_member_values(daily, param)

    if not member_values:
        # Ensemble returned nulls (too far in future or model outage) — use deterministic
        return _get_openmeteo_deterministic(lat, lon, date_str, variable, threshold, unit)

    n = len(member_values)

    if variable in ("rain", "snow"):
        threshold_mm = _to_mm(threshold, unit) if threshold is not None else 0.1
        prob = sum(1 for v in member_values if v >= threshold_mm) / n
    elif variable == "temp_high":
        threshold_c = _to_celsius(threshold, unit)
        if threshold_c is None:
            return None
        prob = sum(1 for v in member_values if v >= threshold_c) / n
    elif variable == "temp_low":
        threshold_c = _to_celsius(threshold, unit)
        if threshold_c is None:
            return None
        prob = sum(1 for v in member_values if v <= threshold_c) / n
    else:
        return None

    logger.debug(f"Open-Meteo ensemble [{variable}] threshold={threshold}{unit} "
                 f"→ prob={prob:.3f} ({n} members, mean={sum(member_values)/n:.1f})")
    return {"source": "openmeteo", "probability": round(prob, 4), "confidence": 0.8}


def _get_openmeteo_deterministic(
    lat: float, lon: float, date_str: str, variable: str,
    threshold: float | None = None, unit: str | None = None,
) -> dict | None:
    """
    Fallback when ensemble members are all null (far-future or outage).

    For precipitation: uses precipitation_probability_max (already a probability).
    For temperature:   uses the point-estimate max/min + a ±3°C uncertainty model
                       to produce a soft probability.
    """
    try:
        resp = httpx.get(
            OPENMETEO_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_probability_max,precipitation_sum,"
                         "snowfall_sum,temperature_2m_max,temperature_2m_min",
                "start_date": date_str,
                "end_date": date_str,
            },
            timeout=10,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
    except httpx.HTTPError as e:
        logger.warning(f"Open-Meteo deterministic failed: {e}")
        return None

    if variable == "rain":
        pp = daily.get("precipitation_probability_max", [None])[0]
        if pp is None:
            return None
        # If a threshold > 0.1mm was specified, downscale the PoP proportionally
        if threshold is not None:
            threshold_mm = _to_mm(threshold, unit) or 0.1
            precip_sum = daily.get("precipitation_sum", [0])[0] or 0
            # Rough adjustment: if expected total < threshold, PoP is optimistic
            if precip_sum > 0 and precip_sum < threshold_mm:
                pp = pp * (precip_sum / threshold_mm)
        return {"source": "openmeteo_det", "probability": round(min(pp / 100, 1.0), 4), "confidence": 0.65}

    elif variable == "snow":
        snowfall = daily.get("snowfall_sum", [None])[0]
        if snowfall is None:
            return None
        threshold_mm = _to_mm(threshold, unit) if threshold is not None else 0.1
        # Convert cm snowfall to mm (Open-Meteo returns cm)
        snowfall_mm = snowfall * 10
        if snowfall_mm >= threshold_mm:
            return {"source": "openmeteo_det", "probability": 0.85, "confidence": 0.6}
        elif snowfall_mm > 0:
            return {"source": "openmeteo_det", "probability": 0.3, "confidence": 0.5}
        else:
            return {"source": "openmeteo_det", "probability": 0.05, "confidence": 0.6}

    elif variable in ("temp_high", "temp_low"):
        key = "temperature_2m_max" if variable == "temp_high" else "temperature_2m_min"
        temp_c = daily.get(key, [None])[0]
        threshold_c = _to_celsius(threshold, unit)
        if temp_c is None or threshold_c is None:
            return None
        # Soft probability: linear model with ±3°C typical forecast error
        # delta > 0 → forecast is above threshold → high probability
        delta = (temp_c - threshold_c) if variable == "temp_high" else (threshold_c - temp_c)
        prob = min(max(0.5 + delta * 0.12, 0.03), 0.97)
        logger.debug(f"Open-Meteo det [{variable}] forecast={temp_c:.1f}°C "
                     f"threshold={threshold_c:.1f}°C delta={delta:.1f} → prob={prob:.3f}")
        return {"source": "openmeteo_det", "probability": round(prob, 4), "confidence": 0.50}

    return None


# ---------------------------------------------------------------------------
# NWS — US-only, authoritative precipitation probability + temperature
# ---------------------------------------------------------------------------
NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {"User-Agent": "WeatherArbBot/1.0 (weather-arb@example.com)"}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=20))
def get_nws_probability(
    lat: float, lon: float, date_str: str, variable: str,
    threshold: float | None = None, unit: str | None = None,
) -> dict | None:
    """
    NWS probability of precipitation (US only).

    Returns None immediately for non-US coordinates — no API call made.
    For temperature, converts point estimate to soft probability using threshold.
    """
    if not _is_us_coordinate(lat, lon):
        return None

    try:
        points_resp = httpx.get(
            f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
            headers=NWS_HEADERS,
            timeout=10,
        )
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecast"]

        forecast_resp = httpx.get(forecast_url, headers=NWS_HEADERS, timeout=10)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json()["properties"]["periods"]
    except Exception:
        return None

    target_date = date.fromisoformat(date_str)
    matching = [
        p for p in periods
        if datetime.fromisoformat(
            p["startTime"].replace("Z", "+00:00")
        ).date() == target_date
    ]
    if not matching:
        return None

    if variable in ("rain", "snow"):
        probs = []
        for p in matching:
            pop = p.get("probabilityOfPrecipitation", {})
            if pop and pop.get("value") is not None:
                probs.append(pop["value"] / 100.0)
            else:
                m = re.search(r"(\d+)\s*percent", p.get("detailedForecast", ""), re.I)
                if m:
                    probs.append(int(m.group(1)) / 100.0)
        if not probs:
            return None
        prob = max(probs)
        if variable == "snow":
            snow_mentioned = any("snow" in p.get("shortForecast", "").lower() for p in matching)
            if not snow_mentioned:
                prob *= 0.3
        return {"source": "nws", "probability": round(prob, 4), "confidence": 0.85}

    elif variable in ("temp_high", "temp_low"):
        day_periods = [p for p in matching if p.get("isDaytime", variable == "temp_high")]
        if not day_periods:
            return None
        temp_f = day_periods[0].get("temperature")
        if temp_f is None:
            return None
        temp_c = (temp_f - 32) * 5 / 9
        threshold_c = _to_celsius(threshold, unit)
        if threshold_c is None:
            return None
        delta = (temp_c - threshold_c) if variable == "temp_high" else (threshold_c - temp_c)
        prob = min(max(0.5 + delta * 0.12, 0.03), 0.97)
        return {"source": "nws", "probability": round(prob, 4), "confidence": 0.75}

    return None


# ---------------------------------------------------------------------------
# Tomorrow.io — hyperlocal, global coverage
# ---------------------------------------------------------------------------
TOMORROWIO_BASE = "https://api.tomorrow.io/v4/weather/forecast"
TOMORROWIO_KEY  = os.getenv("TOMORROWIO_API_KEY")

_tomorrowio_exhausted = False  # set True on 429, reset each scan cycle


def reset_tomorrowio_limit() -> None:
    global _tomorrowio_exhausted
    _tomorrowio_exhausted = False


def get_tomorrowio_probability(
    lat: float, lon: float, date_str: str, variable: str,
    threshold: float | None = None, unit: str | None = None,
) -> dict | None:
    """
    Tomorrow.io hyperlocal forecast. Tries until rate-limited (429), then skips
    gracefully for the remainder of the scan cycle.
    """
    global _tomorrowio_exhausted
    if not TOMORROWIO_KEY or _tomorrowio_exhausted:
        return None

    try:
        resp = httpx.get(
            TOMORROWIO_BASE,
            params={
                "location": f"{lat},{lon}",
                "apikey": TOMORROWIO_KEY,
                "fields": "precipitationProbability,precipitationType,"
                          "temperatureMax,temperatureMin,snowAccumulation",
                "timesteps": "1d",
                "startTime": f"{date_str}T00:00:00Z",
                "endTime":   f"{date_str}T23:59:59Z",
                "units": "metric",
            },
            timeout=10,
        )
        if resp.status_code == 429:
            logger.info("Tomorrow.io rate limit reached — skipping for remainder of scan")
            _tomorrowio_exhausted = True
            return None
        resp.raise_for_status()
        data = resp.json()

    except httpx.HTTPError as e:
        logger.debug(f"Tomorrow.io request failed: {e}")
        return None

    daily = data.get("timelines", {}).get("daily", [])
    if not daily:
        return None
    values = daily[0].get("values", {})

    if variable == "rain":
        prob = values.get("precipitationProbability")
        if prob is None:
            return None
        prob = prob / 100.0
        if values.get("precipitationType") == 2:  # snow, not rain
            prob *= 0.1
        return {"source": "tomorrowio", "probability": round(prob, 4), "confidence": 0.75}

    elif variable == "snow":
        snow_mm = values.get("snowAccumulation", 0) or 0
        prob_raw = (values.get("precipitationProbability") or 0) / 100.0
        threshold_mm = _to_mm(threshold, unit) if threshold is not None else 0.1
        if snow_mm >= threshold_mm:
            return {"source": "tomorrowio", "probability": round(prob_raw, 4), "confidence": 0.75}
        else:
            return {"source": "tomorrowio", "probability": 0.0, "confidence": 0.75}

    elif variable in ("temp_high", "temp_low"):
        key = "temperatureMax" if variable == "temp_high" else "temperatureMin"
        temp_c = values.get(key)
        threshold_c = _to_celsius(threshold, unit)
        if temp_c is None or threshold_c is None:
            return None
        delta = (temp_c - threshold_c) if variable == "temp_high" else (threshold_c - temp_c)
        prob = min(max(0.5 + delta * 0.12, 0.03), 0.97)
        return {"source": "tomorrowio", "probability": round(prob, 4), "confidence": 0.70}

    return None


# ---------------------------------------------------------------------------
# Ensemble aggregation — the only function called by edge.py
# ---------------------------------------------------------------------------

def get_ensemble_probability(
    lat: float, lon: float, date_str: str, variable: str,
    threshold: float | None = None, unit: str | None = None,
) -> dict | None:
    """
    Fetch probability from all sources, apply Brier-score weights, return ensemble.

    Results are cached by (lat, lon, date, variable) for the scan cycle.
    Call clear_forecast_cache() at the start of each scan to reset.

    Returns:
        {
            "probability":  float,          # weighted ensemble estimate
            "sources":      list[dict],     # per-source results
            "disagreement": float,          # std dev across sources (high = uncertain)
            "n_sources":    int,
        }
    or None if no sources returned usable data.
    """
    cache_key = (round(lat, 3), round(lon, 3), date_str, variable)
    if cache_key in _forecast_cache:
        cached = _forecast_cache[cache_key]
        logger.debug(f"Cache hit: {cache_key}")
        return cached

    fetchers = [
        get_noaa_probability,
        get_openmeteo_probability,
        get_nws_probability,
        get_tomorrowio_probability,
    ]

    results = []
    for fetcher in fetchers:
        try:
            r = fetcher(lat, lon, date_str, variable, threshold=threshold, unit=unit)
            if r and r.get("probability") is not None:
                results.append(r)
        except Exception as e:
            logger.debug(f"{fetcher.__name__} failed: {e}")

    if not results:
        _forecast_cache[cache_key] = None
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        w = BRIER_WEIGHTS.get(r["source"], 0.2)
        weighted_sum += r["probability"] * w
        total_weight += w

    if total_weight == 0:
        _forecast_cache[cache_key] = None
        return None

    ensemble_p = max(0.0, min(1.0, weighted_sum / total_weight))
    probs = [r["probability"] for r in results]
    disagreement = statistics.stdev(probs) if len(probs) > 1 else 0.0

    source_summary = ", ".join(
        f"{r['source']}={r['probability']:.3f}" for r in results
    )
    logger.info(
        f"Ensemble [{variable}] ({lat:.2f},{lon:.2f}) {date_str} "
        f"threshold={threshold}{unit} → p={ensemble_p:.3f} "
        f"disagree={disagreement:.3f} | {source_summary}"
    )

    result = {
        "probability":  round(ensemble_p, 4),
        "sources":      results,
        "disagreement": round(disagreement, 4),
        "n_sources":    len(results),
    }
    _forecast_cache[cache_key] = result
    return result
