import httpx
from datetime import date

'''
Best endpoints:

OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
'''

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_openmeteo_probability(lat: float, lon: float, date_str: str, variable: str) -> dict | None:
    """
    Use Open-Meteo ensemble models to derive probability estimates.

    The ensemble API returns 51 member forecasts. The fraction of members
    predicting the event gives a direct probability estimate.

    Returns: {"source": "openmeteo", "probability": float, "confidence": 0.8}
    """
    # Map variable to Open-Meteo parameter names
    param_map = {
        "rain": "precipitation",
        "snow": "snowfall",
        "temp_high": "temperature_2m_max",
        "temp_low": "temperature_2m_min",
    }
    param = param_map.get(variable)
    if not param:
        return None

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": param,
        "start_date": date_str,
        "end_date": date_str,
        "models": "ecmwf_ifs04",  # ECMWF ensemble — best calibration
    }

    try:
        resp = httpx.get(OPENMETEO_ENSEMBLE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        # Fall back to deterministic forecast if ensemble fails
        return _get_openmeteo_deterministic(lat, lon, date_str, variable)

    daily = data.get("daily", {})
    # Ensemble returns a list of values across members
    values = daily.get(param, [])
    if not values:
        return None
    
# Flatten: Open-Meteo ensemble returns nested list [members][days]
    # For single-day request, extract the value for our date
    flat_values = [v for v in values if v is not None]
    if not flat_values:
        return None

    # Probability = fraction of ensemble members predicting event
    if variable in ("rain", "snow"):
        threshold = 2.54  # 0.1 inches in mm for rain; adjust for snow
        prob = sum(1 for v in flat_values if v >= threshold) / len(flat_values)
    else:
        # For temperature, caller should pass threshold; default to median split
        median = sorted(flat_values)[len(flat_values) // 2]
        prob = sum(1 for v in flat_values if v >= median) / len(flat_values)

    return {"source": "openmeteo", "probability": round(prob, 4), "confidence": 0.8}


def _get_openmeteo_deterministic(lat: float, lon: float, date_str: str, variable: str) -> dict | None:
    """Fallback: deterministic Open-Meteo forecast with precipitation probability."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
        "start_date": date_str,
        "end_date": date_str,
    }
    resp = httpx.get(OPENMETEO_BASE, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    if variable == "rain":
        # precipitation_probability_max is 0-100
        pp = daily.get("precipitation_probability_max", [None])[0]
        if pp is None:
            return None
        return {"source": "openmeteo_det", "probability": round(pp / 100, 4), "confidence": 0.65}
    elif variable == "snow":
        snowfall = daily.get("snowfall_sum", [None])[0]
        if snowfall is None:
            return None
        # Return 1.0 if any snowfall predicted, 0.0 otherwise (simplified)
        return {"source": "openmeteo_det", "probability": 1.0 if snowfall > 0 else 0.0, "confidence": 0.6}
    return None