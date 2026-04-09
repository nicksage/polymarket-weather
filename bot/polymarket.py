import httpx
import re
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from geocoder import get_coordinates, extract_city_from_title

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# City name → (lat, lon). Used both for market discovery filtering and metadata parsing.
CITY_COORDS: dict[str, tuple[float, float]] = {
    # North America
    "new york":    (40.7128,  -74.0060),
    "nyc":         (40.7128,  -74.0060),
    "chicago":     (41.8781,  -87.6298),
    "los angeles": (34.0522, -118.2437),
    "miami":       (25.7617,  -80.1918),
    "seattle":     (47.6062, -122.3321),
    "boston":      (42.3601,  -71.0589),
    "dallas":      (32.7767,  -96.7970),
    "denver":      (39.7392, -104.9903),
    "atlanta":     (33.7490,  -84.3880),
    "houston":     (29.7604,  -95.3698),
    "phoenix":     (33.4484, -112.0740),
    # Europe
    "london":      (51.5074,   -0.1278),
    "paris":       (48.8566,    2.3522),
    "amsterdam":   (52.3676,    4.9041),
    "berlin":      (52.5200,   13.4050),
    "madrid":      (40.4168,   -3.7038),
    "rome":        (41.9028,   12.4964),
    "moscow":      (55.7558,   37.6173),
    "istanbul":    (41.0082,   28.9784),
    # Asia
    "shanghai":    (31.2304,  121.4737),
    "beijing":     (39.9042,  116.4074),
    "tokyo":       (35.6762,  139.6503),
    "seoul":       (37.5665,  126.9780),
    "hong kong":   (22.3193,  114.1694),
    "singapore":   ( 1.3521,  103.8198),
    "mumbai":      (19.0760,   72.8777),
    "dubai":       (25.2048,   55.2708),
    "bangkok":     (13.7563,  100.5018),
    # Oceania
    "sydney":      (-33.8688, 151.2093),
    "melbourne":   (-37.8136, 144.9631),
    "wellington":  (-41.2866, 174.7756),
    "auckland":    (-36.8485, 174.7633),
    # South America
    "sao paulo":   (-23.5505,  -46.6333),
    "buenos aires":(-34.6037,  -58.3816),
}

# Weather variable words used to recognise weather events from their title.
WEATHER_WORDS = {
    "temperature", "rainfall", "precipitation",
    "snowfall", "hurricane", "flood", "rain", "snow", "frost",
}


def _is_weather_event(title: str) -> bool:
    """Return True if the event title describes a localised weather market."""
    t = title.lower()
    return (
        any(city in t for city in CITY_COORDS)
        and any(word in t for word in WEATHER_WORDS)
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gamma_get(path: str, params: dict) -> dict:
    """Rate-limited GET to Gamma API."""
    time.sleep(1)  # Respect rate limits: max 1 req/sec
    resp = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def search_weather_markets(min_liquidity: float = 500.0) -> list[dict]:
    """
    Browse ALL active Polymarket events, filter for weather ones, return their markets.

    The Gamma API's `q` search parameter is non-functional (returns the same
    popular markets regardless of query). We therefore paginate through all events
    and apply our own title-based weather filter client-side.

    Each event contains:
      - title   → "Highest temperature in Shanghai on April 9?"  (has city + variable)
      - markets → individual binary sub-markets ("29°C or higher", etc.)

    We attach the event title as `group_title` on every market so
    parse_contract_metadata() can find the city and weather variable.
    """
    all_events: list[dict] = []
    page_size = 100
    max_pages = 60          # Safety cap: 6 000 events maximum
    offset = 0

    for _ in range(max_pages):
        try:
            data = _gamma_get("/events", {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": offset,
            })
            page = data if isinstance(data, list) else data.get("events", [])
            if not page:
                break
            all_events.extend(page)
            logger.debug(f"Fetched {len(page)} events at offset={offset} (total={len(all_events)})")
            if len(page) < page_size:
                break          # Reached the last page
            offset += page_size
        except Exception as e:
            logger.error(f"Failed to fetch events at offset={offset}: {e}")
            break

    # Apply client-side weather filter based on event title
    weather_events = [e for e in all_events if _is_weather_event(e.get("title", ""))]
    logger.info(
        f"Scanned {len(all_events)} events — {len(weather_events)} are weather events"
    )

    # Flatten each weather event's markets, injecting the event title as context
    result: list[dict] = []
    seen_markets: set[str] = set()

    for event in weather_events:
        event_title = event.get("title", "")
        for raw_market in event.get("markets", []):
            mid = raw_market.get("conditionId") or raw_market.get("id")
            if not mid or mid in seen_markets:
                continue
            seen_markets.add(mid)
            raw_market["_event_title"] = event_title
            normalized = _normalize_market(raw_market)
            if normalized and normalized["liquidity_usd"] >= min_liquidity:
                result.append(normalized)

    logger.info(f"Found {len(result)} weather markets above ${min_liquidity} liquidity")
    return result


def _normalize_market(raw: dict) -> dict | None:
    """Convert Gamma API market dict to our standard schema.

    The Gamma API returns outcomes, outcomePrices, and clobTokenIds as
    JSON-encoded strings (not native arrays) — they must be parsed before use.
    The liquidity field is also a string; liquidityNum is the numeric version.
    """
    import json as _json
    try:
        # Gamma API encodes these as JSON strings, not native arrays
        def _parse(field):
            val = raw.get(field, "[]")
            if isinstance(val, str):
                return _json.loads(val)
            return val or []

        outcomes   = _parse("outcomes")       # e.g. ["Yes", "No"]
        prices     = _parse("outcomePrices")  # e.g. ["0.43", "0.57"]
        token_ids  = _parse("clobTokenIds")   # e.g. ["8501...", "2527..."]

        if len(outcomes) < 2 or len(prices) < 2:
            return None

        yes_idx = next((i for i, o in enumerate(outcomes) if str(o).upper() == "YES"), None)
        no_idx  = next((i for i, o in enumerate(outcomes) if str(o).upper() == "NO"),  None)

        if yes_idx is None or no_idx is None:
            return None

        yes_price = float(prices[yes_idx])
        no_price  = float(prices[no_idx])
        yes_token_id = token_ids[yes_idx] if yes_idx < len(token_ids) else None
        no_token_id  = token_ids[no_idx]  if no_idx  < len(token_ids) else None

        # liquidityNum is already numeric; fall back to string "liquidity" field
        liquidity = float(raw.get("liquidityNum") or raw.get("liquidity") or 0)
        volume    = float(raw.get("volumeNum")    or raw.get("volume")    or 0)

        # group_title carries the parent event title injected by search_weather_markets()
        # (e.g. "Highest temperature in Shanghai on April 9?"). This is the primary
        # source of city + variable context — the binary market question is too short.
        group_title = (
            raw.get("_event_title")          # injected from /events response
            or raw.get("groupItemTitle")     # fallback for direct /markets calls
            or raw.get("groupTitle")
            or ""
        )

        return {
            "contract_id": raw.get("conditionId") or raw.get("id"),
            "question": raw.get("question", ""),
            "group_title": group_title,
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "liquidity_usd": liquidity,
            "volume_usd": volume,
            "resolution_date": raw.get("endDate") or raw.get("resolutionDate", ""),
            "resolution_source": raw.get("resolutionSource", ""),
            "description": raw.get("description", ""),
        }
    except (KeyError, ValueError, TypeError, _json.JSONDecodeError) as e:
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
    # Combine all available text — the city/variable often lives in group_title,
    # not the short binary question (e.g. question="29°C or higher",
    # group_title="Highest temperature in Shanghai on April 9?")
    question    = contract.get("question", "")
    group_title = contract.get("group_title", "")
    # Use ONLY question + group_title for city/variable detection.
    # The description is excluded: it causes false positives (e.g. a legal market
    # whose description mentions "Los Angeles" matching our city lookup).
    search_text = f"{question} {group_title}".lower()

    logger.debug(f"Parsing metadata | q='{question}' | group='{group_title[:60]}'")

    # Step 1: check static CITY_COORDS (fast, no network call)
    lat, lon = None, None
    for city, coords in CITY_COORDS.items():
        if city in search_text:
            lat, lon = coords
            break

    # Step 2: if not in static dict, try geocoding via city extracted from title
    if lat is None:
        city_name = extract_city_from_title(group_title or question)
        if city_name:
            coords = get_coordinates(city_name)
            if coords:
                lat, lon = coords
                logger.debug(f"Geocoded '{city_name}' → ({lat}, {lon})")

    if lat is None:
        return None

    # Variable extraction — all matches use search_text (question + group_title only).
    # "high"/"low" are too common in non-weather text, so temperature requires
    # an explicit "temperature" or "temp" word alongside them.
    variable = None
    if any(w in search_text for w in ["snow", "snowfall", "blizzard"]):
        variable = "snow"
    elif any(w in search_text for w in ["rain", "precipitation", "precip"]):
        variable = "rain"
    elif any(w in search_text for w in ["temperature", "temp"]) and (
        "high" in search_text or "maximum" in search_text or "max" in search_text
    ):
        variable = "temp_high"
    elif any(w in search_text for w in ["temperature", "temp"]) and (
        "low" in search_text or "minimum" in search_text or "min" in search_text
    ):
        variable = "temp_low"
    # Celsius/Fahrenheit threshold in the question itself is a strong temperature signal
    elif re.search(r"\d+\s*(?:°c|°f|celsius|fahrenheit)", search_text):
        variable = "temp_high"  # Default — threshold in binary question implies temperature

    if variable is None:
        return None

    # Threshold extraction — search_text covers both question and group_title
    threshold = None
    unit = None

    # Match patterns like "2 inches", "1.5 inch", "85 degrees", "90°F", "29°C"
    inch_match    = re.search(r"(\d+(?:\.\d+)?)\s*inch(?:es)?", search_text)
    celsius_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°c|celsius)", search_text)
    fahr_match    = re.search(r"(\d+(?:\.\d+)?)\s*(?:°f|fahrenheit|degrees?)", search_text)

    if inch_match:
        threshold = float(inch_match.group(1))
        unit = "inches"
    elif celsius_match:
        threshold = float(celsius_match.group(1))
        unit = "celsius"
    elif fahr_match:
        threshold = float(fahr_match.group(1))
        unit = "fahrenheit"

    # Date extraction — use search_text, fall back to resolution_date
    date_str = _extract_date_from_question(search_text, contract.get("resolution_date", ""))

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