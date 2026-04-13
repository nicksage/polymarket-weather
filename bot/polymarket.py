"""
polymarket.py — Polymarket contract discovery for highest-temperature markets.

Scans the Gamma API for all active events whose title matches the pattern
"Highest temperature in <City> on <Date>", groups their binary sub-markets
(temperature range outcomes) into event-level dicts, and parses each
sub-market's temperature range from its question string.

Public API used by edge.py:
    search_temp_high_events(min_liquidity) -> list[event_dict]

Each event_dict contains:
    event_id, event_title, city, date, lat, lon, display_unit,
    outcomes: [{ contract_id, question, range_low, range_high, unit,
                 yes_price, no_price, yes_token_id, no_token_id,
                 liquidity_usd, volume_usd }]
"""

import httpx
import re
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from geocoder import get_coordinates, extract_city_from_title

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Static city → (lat, lon) table.
# Covered cities are checked first (O(1) lookup) before falling back to the
# geocoding API.  All coordinate pairs are (latitude, longitude).
# US cities are tagged with their common Fahrenheit convention.
# ---------------------------------------------------------------------------
CITY_COORDS: dict[str, tuple[float, float]] = {
    # North America — Fahrenheit markets on Polymarket
    "new york":     (40.7128,  -74.0060),
    "nyc":          (40.7128,  -74.0060),
    "chicago":      (41.8781,  -87.6298),
    "los angeles":  (34.0522, -118.2437),
    "miami":        (25.7617,  -80.1918),
    "seattle":      (47.6062, -122.3321),
    "boston":       (42.3601,  -71.0589),
    "dallas":       (32.7767,  -96.7970),
    "denver":       (39.7392, -104.9903),
    "atlanta":      (33.7490,  -84.3880),
    "houston":      (29.7604,  -95.3698),
    "phoenix":      (33.4484, -112.0740),
    "las vegas":    (36.1699, -115.1398),
    "minneapolis":  (44.9778,  -93.2650),
    "detroit":      (42.3314,  -83.0458),
    "toronto":      (43.6532,  -79.3832),
    "montreal":     (45.5017,  -73.5673),
    "vancouver":    (49.2827, -123.1207),
    "mexico city":  (19.4326,  -99.1332),
    # Europe — Celsius markets
    "london":       (51.5074,   -0.1278),
    "paris":        (48.8566,    2.3522),
    "amsterdam":    (52.3676,    4.9041),
    "berlin":       (52.5200,   13.4050),
    "madrid":       (40.4168,   -3.7038),
    "rome":         (41.9028,   12.4964),
    "moscow":       (55.7558,   37.6173),
    "istanbul":     (41.0082,   28.9784),
    "vienna":       (48.2082,   16.3738),
    "zurich":       (47.3769,    8.5417),
    "stockholm":    (59.3293,   18.0686),
    "oslo":         (59.9139,   10.7522),
    "copenhagen":   (55.6761,   12.5683),
    "helsinki":     (60.1699,   24.9384),
    "warsaw":       (52.2297,   21.0122),
    "prague":       (50.0755,   14.4378),
    "budapest":     (47.4979,   19.0402),
    "bucharest":    (44.4268,   26.1025),
    "athens":       (37.9838,   23.7275),
    "lisbon":       (38.7169,   -9.1399),
    "brussels":     (50.8503,    4.3517),
    # Asia — Celsius markets
    "shanghai":     (31.2304,  121.4737),
    "beijing":      (39.9042,  116.4074),
    "tokyo":        (35.6762,  139.6503),
    "seoul":        (37.5665,  126.9780),
    "hong kong":    (22.3193,  114.1694),
    "singapore":    ( 1.3521,  103.8198),
    "mumbai":       (19.0760,   72.8777),
    "dubai":        (25.2048,   55.2708),
    "bangkok":      (13.7563,  100.5018),
    "shenzhen":     (22.5431,  114.0579),
    "taipei":       (25.0330,  121.5654),
    "manila":       (14.5995,  120.9842),
    "jakarta":      (-6.2088,  106.8456),
    "kuala lumpur": ( 3.1390,  101.6869),
    "delhi":        (28.7041,   77.1025),
    "karachi":      (24.8607,   67.0011),
    "dhaka":        (23.8103,   90.4125),
    "kolkata":      (22.5726,   88.3639),
    # Middle East / Africa
    "cairo":        (30.0444,   31.2357),
    "riyadh":       (24.7136,   46.6753),
    "tel aviv":     (32.0853,   34.7818),
    "nairobi":      (-1.2921,   36.8219),
    "johannesburg": (-26.2041,  28.0473),
    "lagos":        ( 6.5244,    3.3792),
    "jeddah":       (21.4858,   39.1925),
    "cape town":    (-33.9249,  18.4241),
    # Additional cities seen on Polymarket temperature markets
    "panama city":  ( 8.9936,  -79.5197),
    "ankara":       (39.9334,   32.8597),
    "lucknow":      (26.8467,   80.9462),
    "munich":       (48.1351,   11.5820),
    "milan":        (45.4642,    9.1900),
    "chongqing":    (29.4316,  106.9123),
    "wuhan":        (30.5928,  114.3055),
    "chengdu":      (30.5728,  104.0668),
    "san francisco":(37.7749, -122.4194),
    "austin":       (30.2672,  -97.7431),
    "busan":        (35.1796,  129.0756),
    "guadalajara":  (20.6597,  -103.3496),
    "monterrey":    (25.6866,   -100.3161),
    "casablanca":   (33.5731,   -7.5898),
    "accra":        ( 5.6037,   -0.1870),
    "dar es salaam":(-6.7924,   39.2083),
    "colombo":      ( 6.9271,   79.8612),
    "ho chi minh":  (10.8231,  106.6297),
    "hanoi":        (21.0285,  105.8542),
    "yangon":       (16.8661,   96.1951),
    "kathmandu":    (27.7172,   85.3240),
    # Oceania
    "sydney":       (-33.8688,  151.2093),
    "melbourne":    (-37.8136,  144.9631),
    "wellington":   (-41.2866,  174.7756),
    "auckland":     (-36.8485,  174.7633),
    "brisbane":     (-27.4698,  153.0251),
    # South America
    "sao paulo":    (-23.5505,  -46.6333),
    "buenos aires": (-34.6037,  -58.3816),
    "lima":         (-12.0464,  -77.0428),
    "bogota":       ( 4.7110,   -74.0721),
    "santiago":     (-33.4489,  -70.6693),
}

# Cities that use Fahrenheit on Polymarket (US + Canada border cities).
# Everything else defaults to Celsius.
FAHRENHEIT_CITIES: set[str] = {
    "new york", "nyc", "chicago", "los angeles", "miami", "seattle",
    "boston", "dallas", "denver", "atlanta", "houston", "phoenix",
    "las vegas", "minneapolis", "detroit",
    "san francisco", "austin",
    # Canadian cities — Polymarket sometimes lists these with °F
    "toronto", "montreal", "vancouver",
}

# Phrases whose presence in a lowercase event title identifies a highest-temp market.
_HIGHEST_TEMP_PHRASES = [
    "highest temperature in",
    "highest temp in",
    "max temperature in",
    "maximum temperature in",
]


def _is_highest_temp_event(title: str) -> bool:
    """Return True iff the event title describes a highest-temperature market."""
    t = title.lower()
    return any(phrase in t for phrase in _HIGHEST_TEMP_PHRASES)


# ---------------------------------------------------------------------------
# Gamma API helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gamma_get(path: str, params: dict) -> dict:
    """Rate-limited GET to Gamma API (max 1 req/s)."""
    time.sleep(1)
    resp = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    page_count = len(data) if isinstance(data, list) else len(data.get("events", []))
    logger.debug(
        f"Polymarket Gamma API call for '{path}' "
        f"(offset={params.get('offset', 0)}) was successful — {page_count} results"
    )
    return data


def _normalize_sub_market(raw: dict, event_title: str) -> dict | None:
    """
    Convert a raw Gamma API market dict into a structured outcome dict.

    Each sub-market is a binary YES/NO contract whose question is a
    temperature range string like "17°C", "16°C or below", "25°C or higher".
    """
    import json as _json
    try:
        def _parse(field):
            val = raw.get(field, "[]")
            if isinstance(val, str):
                return _json.loads(val)
            return val or []

        outcomes   = _parse("outcomes")
        prices     = _parse("outcomePrices")
        token_ids  = _parse("clobTokenIds")

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

        liquidity = float(raw.get("liquidityNum") or raw.get("liquidity") or 0)
        volume    = float(raw.get("volumeNum")    or raw.get("volume")    or 0)

        question = raw.get("question", "")

        # groupItemTitle (e.g. "27°C", "19°C or higher") is the clean display
        # label — much easier to parse than the full verbose question string.
        # Try it first; fall back to the full question if needed.
        group_item = raw.get("groupItemTitle", "").strip()
        parse_source = group_item if group_item else question
        low, high, unit = parse_temperature_range(parse_source)

        # Second attempt: full question string
        if low is None and high is None and group_item:
            low, high, unit = parse_temperature_range(question)

        # NOTE: We do NOT reject markets where parsing returns (None, None).
        # All sub-markets belonging to a temperature event are included so the
        # dashboard shows every available contract.  The edge calculation in
        # edge.py will skip probability computation for unparseable ranges but
        # will still display the market price and contract metadata.

        return {
            "contract_id":      raw.get("conditionId") or raw.get("id"),
            "gamma_market_id":  str(raw.get("id", "")),   # numeric Gamma ID for /markets/{id} lookups
            "question":         question,
            "range_low":        low,
            "range_high":       high,
            "unit":             unit,
            "yes_price":        yes_price,
            "no_price":         no_price,
            "yes_token_id":     yes_token_id,
            "no_token_id":      no_token_id,
            "liquidity_usd":    liquidity,
            "volume_usd":       volume,
            "resolution_date":  raw.get("endDate") or raw.get("resolutionDate", ""),
        }
    except (KeyError, ValueError, TypeError, _json.JSONDecodeError) as e:
        logger.debug(f"Failed to normalize sub-market: {e}")
        return None


# ---------------------------------------------------------------------------
# Temperature range parser
# ---------------------------------------------------------------------------

def parse_temperature_range(question: str) -> tuple[float | None, float | None, str]:
    """
    Extract (range_low, range_high, unit) from a temperature sub-market question.

    Handles patterns:
        "17°C"                → (17.0, 17.0, "celsius")
        "16°C or 17°C"        → (16.0, 17.0, "celsius")
        "16-17°C"             → (16.0, 17.0, "celsius")
        "16°C or below"       → (None, 16.0, "celsius")
        "25°C or higher"      → (25.0, None, "celsius")
        "25°C or above"       → (25.0, None, "celsius")
        "90°F or above"       → (90.0, None, "fahrenheit")
        "54-55°F"             → (54.0, 55.0, "fahrenheit")
        "below 16°C"          → (None, 16.0, "celsius")
        "at least 25°C"       → (25.0, None, "celsius")

    Returns (None, None, "celsius") when no temperature value is found.
    """
    q = question.lower().strip()

    # Detect unit
    if "°f" in q or "fahrenheit" in q:
        unit = "fahrenheit"
        sym  = r"(?:°f|f|fahrenheit)"
    else:
        unit = "celsius"
        sym  = r"(?:°c|c|celsius)?"

    # Pattern: single number followed by unit, then "or below / or lower"
    m = re.search(
        rf"(-?\d+(?:\.\d+)?)\s*{sym}\s+(?:or\s+)?(?:below|lower|under|less)", q
    )
    if m:
        return (None, float(m.group(1)), unit)

    # Pattern: "below / under / less than" followed by number
    m = re.search(
        rf"(?:below|under|less\s+than)\s+(-?\d+(?:\.\d+)?)\s*{sym}", q
    )
    if m:
        return (None, float(m.group(1)), unit)

    # Pattern: single number followed by "or higher / or above / or more / at least"
    m = re.search(
        rf"(-?\d+(?:\.\d+)?)\s*{sym}\s+(?:or\s+)?(?:higher|above|more|over|plus)", q
    )
    if m:
        return (float(m.group(1)), None, unit)

    # Pattern: "at least / above / over" followed by number
    m = re.search(
        rf"(?:at\s+least|above|over|higher\s+than)\s+(-?\d+(?:\.\d+)?)\s*{sym}", q
    )
    if m:
        return (float(m.group(1)), None, unit)

    # Pattern: two numbers connected by "or", "to", "-", "–"
    m = re.search(
        rf"(-?\d+(?:\.\d+)?)\s*{sym}?\s*(?:or|to|–|-)\s*(-?\d+(?:\.\d+)?)\s*{sym}", q
    )
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a, b), max(a, b), unit)

    # Pattern: single temperature value (exact match)
    m = re.search(rf"(-?\d+(?:\.\d+)?)\s*{sym}", q)
    if m:
        v = float(m.group(1))
        return (v, v, unit)

    return (None, None, unit)


# ---------------------------------------------------------------------------
# City / date extraction from event title
# ---------------------------------------------------------------------------

def _extract_city_and_date(event: dict) -> tuple[str | None, str | None, float | None, float | None, str]:
    """
    Extract (city_name, date_str, lat, lon, display_unit) from an event dict.

    Uses the event title as the primary source, falls back to the geocoding
    API for cities not in CITY_COORDS.
    """
    title      = event.get("title", "")
    resolution = event.get("endDate") or event.get("resolutionDate", "")

    search = title.lower()

    # Step 1: static lookup
    city_name = None
    lat = lon = None
    for city, coords in CITY_COORDS.items():
        if city in search:
            city_name = city
            lat, lon  = coords
            break

    # Step 2: geocoding fallback
    if lat is None:
        city_name = extract_city_from_title(title)
        if city_name:
            coords = get_coordinates(city_name)
            if coords:
                lat, lon = coords

    if lat is None:
        return (None, None, None, None, "celsius")

    # Display unit: Fahrenheit for known US cities
    display_unit = "fahrenheit" if (city_name or "").lower() in FAHRENHEIT_CITIES else "celsius"

    # Date extraction
    date_str = _extract_date_from_title(title, resolution)

    return (city_name, date_str, lat, lon, display_unit)


def _extract_date_from_title(title: str, resolution_date: str) -> str | None:
    """Extract ISO date string from event title or resolution date."""
    from datetime import datetime, date
    import calendar

    # Prefer resolution_date (most reliable)
    if resolution_date:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(resolution_date[:len(fmt)], fmt)
                return dt.date().isoformat()
            except ValueError:
                pass
        try:
            dt = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except ValueError:
            pass

    # Parse month/day from title text
    MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    MONTHS.update({m[:3].lower(): i for i, m in enumerate(calendar.month_name) if m})

    t = title.lower()
    for month_str, month_num in MONTHS.items():
        if month_str in t:
            day_match = re.search(rf"{month_str}\s+(\d{{1,2}})", t)
            if day_match:
                day = int(day_match.group(1))
                year_match = re.search(r"\b(202\d)\b", t)
                year = int(year_match.group(1)) if year_match else date.today().year
                try:
                    return date(year, month_num, day).isoformat()
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def _fetch_events_by_tag_slug(tag_slug: str) -> list[dict]:
    """
    Fetch ALL active events with a specific Gamma API tag_slug, paginating
    until exhausted.  Returns a flat list of raw event dicts.

    The Gamma API sorts by internal popularity when no tag_slug is provided,
    meaning temperature events may not appear within the first 6 000 results.
    Querying by tag_slug=temperature returns ONLY the ~176 temperature events
    in 2 pages, regardless of their global sort position.
    """
    events: list[dict] = []
    page_size = 100
    offset    = 0
    max_pages = 20       # 20 × 100 = 2 000 events max per tag; well above actual count

    for _ in range(max_pages):
        try:
            data = _gamma_get("/events", {
                "active":   "true",
                "closed":   "false",
                "limit":    page_size,
                "offset":   offset,
                "tag_slug": tag_slug,
            })
            page = data if isinstance(data, list) else data.get("events", [])
            if not page:
                break
            events.extend(page)
            logger.debug(
                f"tag_slug={tag_slug}: fetched {len(page)} events at offset={offset} "
                f"(total={len(events)})"
            )
            if len(page) < page_size:
                break
            offset += page_size
        except Exception as e:
            logger.error(f"Failed to fetch events (tag_slug={tag_slug}, offset={offset}): {e}")
            break

    return events


def search_temp_high_events(min_liquidity: float = 500.0) -> list[dict]:
    """
    Fetch all active highest-temperature Polymarket events and return a list
    of event-level dicts — one per city+date — each containing all temperature-
    range sub-markets as `outcomes`.

    Discovery strategy:
        Primary:  tag_slug=temperature  (Polymarket's "Daily Temperature" tag)
                  This is the correct and efficient approach — it returns all
                  ~176 temperature events in 2 API pages.
        Fallback: if the tag returns 0 results (API change), fall back to a
                  full event scan filtered by title keywords.
    """
    # Primary: fetch via the Polymarket temperature tag slug
    all_events = _fetch_events_by_tag_slug("temperature")

    # Fallback: full scan (slow — only used if tag approach fails)
    if not all_events:
        logger.warning(
            "tag_slug=temperature returned 0 events — falling back to full event scan. "
            "This is slow (~60 API pages). Check if the tag slug has changed."
        )
        page_size = 100
        offset    = 0
        for _ in range(60):
            try:
                data = _gamma_get("/events", {
                    "active": "true",
                    "closed": "false",
                    "limit":  page_size,
                    "offset": offset,
                })
                page = data if isinstance(data, list) else data.get("events", [])
                if not page:
                    break
                all_events.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size
            except Exception as e:
                logger.error(f"Full scan failed at offset={offset}: {e}")
                break

    # Filter: keep only highest-temperature events (title check)
    temp_events = [e for e in all_events if _is_highest_temp_event(e.get("title", ""))]
    logger.info(
        f"Fetched {len(all_events)} temperature-tagged events — "
        f"{len(temp_events)} match highest-temperature title filter"
    )

    result: list[dict] = []
    seen_events: set[str] = set()

    for raw_event in temp_events:
        event_id    = str(raw_event.get("id") or raw_event.get("slug", ""))
        event_title = raw_event.get("title", "")

        # Deduplicate
        if event_id in seen_events:
            continue
        seen_events.add(event_id)

        city_name, date_str, lat, lon, display_unit = _extract_city_and_date(raw_event)
        if lat is None or date_str is None:
            logger.debug(f"Could not parse city/date from: '{event_title}'")
            continue

        # Collect all valid sub-markets for this event.
        # We do NOT filter individual bins by liquidity here — temperature bins
        # are mutually exclusive and exhaustive, so dropping any bin corrupts the
        # probability normalization across the full set.  Liquidity is checked at
        # execution time by risk.py.  The min_liquidity guard is applied at the
        # EVENT level below (events with zero total liquidity are skipped).
        outcomes = []
        for raw_market in raw_event.get("markets", []):
            outcome = _normalize_sub_market(raw_market, event_title)
            if outcome is None:
                continue
            outcomes.append(outcome)

        if not outcomes:
            logger.debug(f"No valid outcomes for event: '{event_title}'")
            continue

        # Event-level liquidity guard: skip events where the total volume
        # across ALL bins is below the minimum.  This filters out dead/inactive
        # events without silently dropping individual bins within a live event.
        total_event_liquidity = sum(o["liquidity_usd"] for o in outcomes)
        if total_event_liquidity < min_liquidity:
            logger.debug(
                f"Skipping event (total liquidity ${total_event_liquidity:.0f} "
                f"< ${min_liquidity:.0f}): '{event_title}'"
            )
            continue

        # Sort outcomes by range_low (ascending), putting open-ended lower bounds first
        def _sort_key(o):
            lo = o["range_low"]
            hi = o["range_high"]
            if lo is None and hi is not None:
                return (hi - 1000,)
            if lo is not None:
                return (lo,)
            return (0,)

        outcomes.sort(key=_sort_key)

        result.append({
            "event_id":     event_id,
            "event_title":  event_title,
            "city":         city_name.title() if city_name else "",
            "city_key":     (city_name or "").lower(),
            "date":         date_str,
            "lat":          lat,
            "lon":          lon,
            "display_unit": display_unit,
            "outcomes":     outcomes,
        })

    logger.info(f"Found {len(result)} highest-temperature events with valid outcomes")
    return result


# ---------------------------------------------------------------------------
# CLOB price refresh (used during execution)
# ---------------------------------------------------------------------------

def get_contract_price(contract_id: str) -> dict | None:
    """Fetch the latest YES/NO prices for a specific sub-market contract."""
    try:
        result = _gamma_get("/markets", {"conditionId": contract_id})
        data = result[0] if isinstance(result, list) and result else result
        if not data:
            return None
        return _normalize_sub_market(data, "")
    except Exception as e:
        logger.error(f"Failed to get price for {contract_id}: {e}")
        return None


def get_market_status(contract_id: str, gamma_market_id: str = None) -> dict | None:
    """
    Fetch resolution status and current prices for a single sub-market.

    Uses the numeric Gamma market ID (e.g. "1886470") for a direct path
    lookup: GET /markets/{gamma_market_id}.  Falls back to scanning the
    parent event if gamma_market_id is not provided.

    Returns a dict with:
        closed       bool   — True when market has resolved
        yes_price    float  — current YES token price (1.0 or 0.0 if resolved)
        no_price     float  — current NO token price
        winner       str|None — "YES", "NO", or None if still open
    Returns None on API error.
    """
    import json as _json
    try:
        if gamma_market_id:
            data = _gamma_get(f"/markets/{gamma_market_id}", {})
        else:
            # Fallback: scan first page for a matching conditionId
            result = _gamma_get("/markets", {"limit": 100})
            markets = result if isinstance(result, list) else []
            data = next((m for m in markets if m.get("conditionId") == contract_id), None)

        if not data:
            return None

        closed = bool(data.get("closed", False))
        active = bool(data.get("active", True))

        def _parse(field):
            val = data.get(field, "[]")
            if isinstance(val, str):
                return _json.loads(val)
            return val or []

        outcomes = _parse("outcomes")
        prices   = _parse("outcomePrices")

        yes_idx = next((i for i, o in enumerate(outcomes) if str(o).upper() == "YES"), None)
        no_idx  = next((i for i, o in enumerate(outcomes) if str(o).upper() == "NO"),  None)

        yes_price = float(prices[yes_idx]) if yes_idx is not None and yes_idx < len(prices) else None
        no_price  = float(prices[no_idx])  if no_idx  is not None and no_idx  < len(prices) else None

        # Determine winner: after resolution, the winning token trades at 1.0
        winner = None
        if closed:
            if yes_price is not None and yes_price >= 0.99:
                winner = "YES"
            elif no_price is not None and no_price >= 0.99:
                winner = "NO"

        return {
            "closed":    closed,
            "active":    active,
            "yes_price": yes_price,
            "no_price":  no_price,
            "winner":    winner,
        }
    except Exception as e:
        logger.error(f"Failed to get market status for {contract_id}: {e}")
        return None


def get_data_api_positions(wallet_address: str) -> list[dict]:
    """
    Fetch all on-chain positions for a wallet from the Polymarket Data API.
    This is the blockchain source of truth for live (non-paper) positions.

    Returns a list of position dicts with keys:
        token_id, title, outcome, size, avg_price,
        initial_value, current_value, cash_pnl, percent_pnl
    """
    if not wallet_address:
        return []
    try:
        resp = httpx.get(
            "https://data-api.polymarket.com/positions",
            params={"address": wallet_address, "sizeThreshold": "0"},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            return []
        result = []
        for p in raw:
            asset = p.get("asset", {})
            result.append({
                "token_id":      asset.get("token_id") or p.get("asset_id", ""),
                "title":         p.get("title", ""),
                "outcome":       asset.get("outcome", ""),
                "size":          float(p.get("size", 0)),
                "avg_price":     float(p.get("avgPrice", 0)),
                "initial_value": float(p.get("initialValue", 0)),
                "current_value": float(p.get("currentValue", 0)),
                "cash_pnl":      float(p.get("cashPnl", 0)),
                "percent_pnl":   float(p.get("percentPnl", 0)),
            })
        logger.info(f"Data API: fetched {len(result)} on-chain positions for {wallet_address[:10]}...")
        return result
    except Exception as e:
        logger.error(f"Data API positions fetch failed: {e}")
        return []
