"""Shared city metadata: IANA timezone per Polymarket city + a helper to render
a UTC collection timestamp in that city's local time.

Used by both collectors so the city list / timezone map has a single source.
The city keys must match the strings Polymarket uses (events.city), e.g. "NYC",
"Sao Paulo", "Hong Kong".
"""
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

# Polymarket city string -> IANA timezone. DST is handled by zoneinfo.
CITY_TZ = {
    "Amsterdam":     "Europe/Amsterdam",
    "Ankara":        "Europe/Istanbul",
    "Atlanta":       "America/New_York",
    "Austin":        "America/Chicago",
    "Beijing":       "Asia/Shanghai",
    "Buenos Aires":  "America/Argentina/Buenos_Aires",
    "Busan":         "Asia/Seoul",
    "Cape Town":     "Africa/Johannesburg",
    "Chengdu":       "Asia/Shanghai",
    "Chicago":       "America/Chicago",
    "Chongqing":     "Asia/Shanghai",
    "Dallas":        "America/Chicago",
    "Denver":        "America/Denver",
    "Guangzhou":     "Asia/Shanghai",
    "Helsinki":      "Europe/Helsinki",
    "Hong Kong":     "Asia/Hong_Kong",
    "Houston":       "America/Chicago",
    "Istanbul":      "Europe/Istanbul",
    "Jeddah":        "Asia/Riyadh",
    "Jinan":         "Asia/Shanghai",
    "Karachi":       "Asia/Karachi",
    "Kuala Lumpur":  "Asia/Kuala_Lumpur",
    "London":        "Europe/London",
    "Los Angeles":   "America/Los_Angeles",
    "Lucknow":       "Asia/Kolkata",
    "Madrid":        "Europe/Madrid",
    "Manila":        "Asia/Manila",
    "Mexico City":   "America/Mexico_City",
    "Miami":         "America/New_York",
    "Milan":         "Europe/Rome",
    "Moscow":        "Europe/Moscow",
    "Munich":        "Europe/Berlin",
    "NYC":           "America/New_York",
    "Panama City":   "America/Panama",
    "Paris":         "Europe/Paris",
    "Qingdao":       "Asia/Shanghai",
    "San Francisco": "America/Los_Angeles",
    "Sao Paulo":     "America/Sao_Paulo",
    "Seattle":       "America/Los_Angeles",
    "Seoul":         "Asia/Seoul",
    "Shanghai":      "Asia/Shanghai",
    "Shenzhen":      "Asia/Shanghai",
    "Singapore":     "Asia/Singapore",
    "Taipei":        "Asia/Taipei",
    "Tel Aviv":      "Asia/Jerusalem",
    "Tokyo":         "Asia/Tokyo",
    "Toronto":       "America/Toronto",
    "Warsaw":        "Europe/Warsaw",
    "Wellington":    "Pacific/Auckland",
    "Wuhan":         "Asia/Shanghai",
    "Zhengzhou":     "Asia/Shanghai",
}


@lru_cache(maxsize=None)
def _zone(name):
    return ZoneInfo(name)


def local_iso(utc_value, city):
    """Render a UTC collection timestamp in `city`'s local time (ISO 8601 with
    offset), or None if the city is unknown or the value can't be parsed.

    `utc_value` may be a timezone-aware datetime or an ISO-8601 string (e.g.
    the collectors' `fetched_at` / `recorded_at`).
    """
    tzname = CITY_TZ.get(city)
    if not tzname:
        return None
    if isinstance(utc_value, str):
        try:
            dt = datetime.fromisoformat(utc_value)
        except ValueError:
            return None
    else:
        dt = utc_value
    if dt is None:
        return None
    try:
        return dt.astimezone(_zone(tzname)).isoformat()
    except Exception:
        return None
