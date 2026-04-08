import httpx
import re
import time
import logging
from typing import Generator
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

WEATHER_KEYWORDS = ["rain", "snow", "temperature", "weather", "precipitation", "inches", "degrees", "storm", "hurricane", "flood", "frost"]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gamma_get(path: str, params: dict) -> dict:
    """Rate-limited GET to Gamma API."""
    time.sleep(1)  # Respect rate limits: max 1 req/sec
    resp = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def search_weather_markets(min_liquidity: float = 500.0) -> list[dict]:
    """
    Search Polymarket Gamma API for active weather contracts.

        Returns a list of normalized contract dicts.
    Filters to contracts with active=True and liquidity >= min_liquidity.
    """
    all_markets = []

    for keyword in WEATHER_KEYWORDS:
        try:
            data = _gamma_get("/markets", {
                "active": "true",
                "closed": "false",
                "q": keyword,
                "limit": 100,
                "offset": 0,
            })
            markets = data if isinstance(data, list) else data.get("markets", [])
            all_markets.extend(markets)
        except Exception as e:
            logger.error(f"Failed to search keyword '{keyword}': {e}")
            continue

    # Deduplicate by market ID
    seen = set()
    unique = []
    for m in all_markets:
        mid = m.get("id") or m.get("conditionId")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(m)

    # Normalize and filter
    result = []
    for m in unique:
        normalized = _normalize_market(m)
        if normalized and normalized["liquidity_usd"] >= min_liquidity:
            result.append(normalized)

    logger.info(f"Found {len(result)} weather markets above ${min_liquidity} liquidity")
    return result

def _normalize_market(raw: dict) -> dict | None:
    """Convert Gamma API market dict to our standard schema."""
    try:
        # Extract tokens (YES/NO)
        tokens = raw.get("tokens", [])
        yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
        no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), None)

        if not yes_token or not no_token:
            return None
        
        yes_price = float(yes_token.get("price", 0))
        no_price = float(no_token.get("price", 0))

        return {
            "contract_id": raw.get("conditionId") or raw.get("id"),
            "question": raw.get("question", ""),
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_token_id": yes_token.get("token_id"),
            "no_token_id": no_token.get("token_id"),
            "liquidity_usd": float(raw.get("liquidity", 0)),
            "volume_usd": float(raw.get("volume", 0)),
            "resolution_date": raw.get("endDate") or raw.get("resolutionDate", ""),
            "resolution_source": raw.get("resolutionSource", ""),
            "description": raw.get("description", ""),
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Failed to normalize market: {e}")
        return None
    
def get_contract_price(contract_id: str) -> dict | None:
    """Get current YES/NO prices for a specific contract."""
    try:
        data = _gamma_get(f"/markets/{contract_id}", {})
        return _normalize_market(data)
    except Exception as e:
        logger.error(f"Failed to get price for {contract_id}: {e}")
        return None

def parse_contract_metadata(contract: dict) -> dict | None:
    """
    Extract meteorological parameters from a contract question string.

    Maps the English question to structured parameters:
    - lat/lon of the location
    - date of the event
    - variable type (rain, snow, temp_high, temp_low)
    - threshold value and unit

    Returns None if the question can't be parsed

    Example:
    "Will it snow more than 2 inches in NYC on March 20, 2026?"
    -> {"lat": 40.71, "lon": -74.01, "date": "2026-03-20",
    "variable": "snow", "threshold": 2.0, "unit": "inches"}
    """
    question = contract.get("question", "").lower()

    # Location extraction using a simple city->coord lookup
    # In production, replace with a geocoding API call
    CITY_COORDS = {
        "new york": (40.7128, -74.0060),
        "nyc": (40.7128, -74.0060),
        "chicago": (41.8781, -87.6298),
        "los angeles": (34.0522, -118.2437),
        "miami": (25.7617, -80.1918),
        "seattle": (47.6062, -122.3321),
        "boston": (42.3601, -71.0589),
        "dallas": (32.7767, -96.7970),
        "denver": (39.7392, -104.9903),
        "atlanta": (33.7490, -84.3880),
        "houston": (29.7604, -95.3698),
        "phoenix": (33.4484, -112.0740),
    }

    lat, lon = None, None
    for city, coords in CITY_COORDS.items():
        if city in question:
            lat, lon = coords
            break

    if lat is None:
        return None

    if variable is None:
        return None

    # Variable extraction
    variable = None
    if any(w in question for w in ["snow", "snowfall", "blizzard"]):
        variable = "snow"
    elif any(w in question for w in ["rain", "precipitation", "precip", "flood"]):
        variable = "rain"
    elif "high" in question or "maximum" in question or "max temp" in question:
        variable = "temp_high"
    elif "low" in question or "minimum" in question or "min temp" in question:
        variable = "temp_low"

    if variable is None:
        return None
    # Threshold extraction
    threshold = None
    unit = None

    # Match patterns like "2 inches", "1.5 inch", "85 degrees", "90°F"
    inch_match = re.search(r"(\d+(?:\.\d+)?)\s*inch(?:es)?", question)
    deg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°f?)", question)
    
    if inch_match:
        threshold = float(inch_match.group(1))
        unit = "inches"
    elif deg_match:
        threshold = float(deg_match.group(1))
        unit = "fahrenheit"

    # Date extraction — look for month/day patterns
    date_str = _extract_date_from_question(question, contract.get("resolution_date", ""))

    return {
        "lat": lat,
        "lon": lon,
        "date": date_str,
        "variable": variable,
        "threshold": threshold,
    "unit": unit,
    }

def _extract_date_from_question(question: str, resolution_date: str) -> str | None:
    """Extract ISO date from question text or fall back to resolution_date."""
    from datetime import datetime, date
    import calendar

    # Try to parse from resolution_date first (most reliable)
    if resolution_date:
        try:
            # Handle ISO format
            dt = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except ValueError:
            pass

    # Parse month/day from question text
    MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    MONTHS.update({m[:3].lower(): i for i, m in enumerate(calendar.month_name) if m})

    for month_str, month_num in MONTHS.items():
        if month_str in question:
            day_match = re.search(rf"{month_str}\s+(\d{{1,2}})", question)
            if day_match:
                day = int(day_match.group(1))
                year_match = re.search(r"\b(202\d)\b", question)
                year = int(year_match.group(1)) if year_match else date.today().year
                try:
                    return date(year, month_num, day).isoformat()
                except ValueError:
                    pass
    return None