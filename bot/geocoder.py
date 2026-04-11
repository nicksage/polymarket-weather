"""
geocoder.py — City name → (lat, lon) via Open-Meteo Geocoding API

Free, no API key required, numeric results, full international coverage.
Results are cached permanently for the process lifetime (cities don't move).
"""
import httpx
import logging

logger = logging.getLogger(__name__)

GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1/search"

# Process-lifetime cache: city_name.lower() → (lat, lon) or None
_cache: dict[str, tuple[float, float] | None] = {}


def get_coordinates(city_name: str) -> tuple[float, float] | None:
    """
    Resolve a city name to (lat, lon).

    1. Checks the in-memory cache (populated on first call per city).
    2. On cache miss, calls the Open-Meteo Geocoding API.
    3. Returns None and logs a warning if the city cannot be resolved.
    """
    key = city_name.lower().strip()
    if key in _cache:
        return _cache[key]

    try:
        resp = httpx.get(
            GEOCODE_BASE,
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            logger.warning(f"Geocoding: no result for '{city_name}'")
            _cache[key] = None
            return None

        r = results[0]
        coords = (float(r["latitude"]), float(r["longitude"]))
        logger.info(
            f"Geocoding API call for '{city_name}' was successful — "
            f"resolved to ({coords[0]:.4f}, {coords[1]:.4f}), {r.get('country', '')}"
        )
        _cache[key] = coords
        return coords

    except Exception as e:
        logger.error(f"Geocoding failed for '{city_name}': {e}")
        _cache[key] = None
        return None


def extract_city_from_title(title: str) -> str | None:
    """
    Extract the city name from a Polymarket event title using multiple regex patterns.

    Tries patterns in order of specificity:
      1. "temperature in <City>"          — most common pattern
      2. "weather in <City>"
      3. "rainfall in <City>" / "rain in <City>"
      4. "snowfall in <City>" / "snow in <City>"
      5. "flood in <City>" / "hurricane in <City>"
      6. "<City> temperature"             — reversed order
      7. "in <City> on"                   — generic fallback
      8. "in <City>?"                     — end-of-sentence fallback

    Returns the extracted city string (title-cased) or None.
    """
    import re

    patterns = [
        # Specific weather + "in <City>"
        r"(?:highest|lowest|high|low|max|min)?\s*temperature\s+in\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+on\s|\s*\?|$)",
        r"weather\s+in\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+on\s|\s*\?|$)",
        r"(?:rainfall|rain)\s+in\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+on\s|\s*\?|$)",
        r"(?:snowfall|snow)\s+in\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+on\s|\s*\?|$)",
        r"(?:flood|hurricane|frost|precipitation)\s+in\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+on\s|\s*\?|$)",
        # "<City> temperature" (reversed)
        r"([A-Z][a-zA-Z\s\-]+?)\s+temperature",
        # Generic "in <City> on <date>" or "in <City>?"
        r"\bin\s+([A-Z][a-zA-Z\s\-]{2,25}?)\s+on\s",
        r"\bin\s+([A-Z][a-zA-Z\s\-]{2,25}?)\s*\?",
    ]

    for pattern in patterns:
        m = re.search(pattern, title)
        if m:
            city = m.group(1).strip().rstrip("?").strip()
            # Reject obvious non-city matches (single chars, common words)
            if len(city) >= 2 and city.lower() not in {"the", "a", "an", "any", "all"}:
                return city

    return None
