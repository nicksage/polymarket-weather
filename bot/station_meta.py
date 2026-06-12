"""
station_meta.py — Single source of truth for Polymarket settlement stations.

Each entry: city -> (icao_code, mesonet_network, iana_timezone, lat, lon)

The lat/lon is the SETTLEMENT STATION's coordinates (e.g., KLAX 33.94, -118.39),
NOT the city center.  This is the critical fix to the LA +4.5°C bias: forecast
at the station, not downtown.

Networks are Iowa State's identifiers (used by station_obs_pull.py).
Timezones are IANA strings (used to interpret hourly forecasts and obs in
the same local frame as the market resolution date).

Skipped: Hong Kong — settles on HKO (manual climat station), no ICAO
airport equivalent in any free API.

Sources:
  - ICAO codes: Polymarket market resolution links (Wunderground / weather.gov)
  - Networks:   Iowa State Mesonet station registry
  - Lat/lon:    Iowa State Mesonet station metadata GeoJSON
"""

from __future__ import annotations

# city -> (icao, mesonet_network, iana_tz, lat, lon)
CITY_STATIONS: dict[str, tuple[str, str, str, float, float]] = {
    "Ankara":        ("LTAC", "TR__ASOS",    "Europe/Istanbul",                40.1281,   32.9951),
    "Atlanta":       ("KATL", "US_ASOS",     "America/New_York",               33.6301,  -84.4418),
    "Austin":        ("KAUS", "US_ASOS",     "America/Chicago",                30.1830,  -97.6799),
    "Beijing":       ("ZBAA", "CN__ASOS",    "Asia/Shanghai",                  40.0741,  116.5870),
    "Buenos Aires":  ("SAEZ", "AR__ASOS",    "America/Argentina/Buenos_Aires", -34.8222, -58.5358),
    "Busan":         ("RKPK", "KR__ASOS",    "Asia/Seoul",                     35.1795,  128.9382),
    "Cape Town":     ("FACT", "ZA__ASOS",    "Africa/Johannesburg",            -33.9667,  18.6000),
    "Chengdu":       ("ZUUU", "CN__ASOS",    "Asia/Shanghai",                  30.6667,  104.0167),
    "Chicago":       ("KORD", "US_ASOS",     "America/Chicago",                41.9602,  -87.9316),
    "Chongqing":     ("ZUCK", "CN__ASOS",    "Asia/Shanghai",                  29.5200,  106.4800),
    "Dallas":        ("KDAL", "US_ASOS",     "America/Chicago",                32.8471,  -96.8518),
    "Denver":        ("KBKF", "US_ASOS",     "America/Denver",                 39.7017, -104.7517),
    "Guangzhou":     ("ZGGG", "CN__ASOS",    "Asia/Shanghai",                  23.3964,  113.3008),
    "Helsinki":      ("EFHK", "FI__ASOS",    "Europe/Helsinki",                60.3172,   24.9633),
    "Houston":       ("KHOU", "US_ASOS",     "America/Chicago",                29.6375,  -95.2824),
    "Istanbul":      ("LTFM", "TR__ASOS",    "Europe/Istanbul",                41.2629,   28.7413),
    "Jeddah":        ("OEJN", "SA__ASOS",    "Asia/Riyadh",                    21.6598,   39.1222),
    "Karachi":       ("OPKC", "PK__ASOS",    "Asia/Karachi",                   24.8456,   67.1614),
    "Kuala Lumpur":  ("WMKK", "MY__ASOS",    "Asia/Kuala_Lumpur",               2.7167,  101.7000),
    "London":        ("EGLC", "GB__ASOS",    "Europe/London",                  51.5053,    0.0553),
    "Los Angeles":   ("KLAX", "US_ASOS",     "America/Los_Angeles",            33.9382, -118.3865),
    "Lucknow":       ("VILK", "IN__ASOS",    "Asia/Kolkata",                   26.7606,   80.8893),
    "Madrid":        ("LEMD", "ES__ASOS",    "Europe/Madrid",                  40.4667,   -3.5556),
    "Manila":        ("RPLL", "PH__ASOS",    "Asia/Manila",                    14.5069,  121.0042),
    "Mexico City":   ("MMMX", "MX__ASOS",    "America/Mexico_City",            19.4363,  -99.0721),
    "Miami":         ("KMIA", "US_ASOS",     "America/New_York",               25.7880,  -80.3169),
    "Milan":         ("LIMC", "IT__ASOS",    "Europe/Rome",                    45.6300,    8.7231),
    "Moscow":        ("UUWW", "RU__ASOS",    "Europe/Moscow",                  55.5915,   37.2615),
    "Munich":        ("EDDM", "DE__ASOS",    "Europe/Berlin",                  48.3583,   11.8092),
    "NYC":           ("KLGA", "US_ASOS",     "America/New_York",               40.7794,  -73.8803),
    "Panama City":   ("MPMG", "PA__ASOS",    "America/Panama",                  8.9833,  -79.5167),
    "Paris":         ("LFPB", "FR__ASOS",    "Europe/Paris",                   48.9672,    2.4272),
    "Qingdao":       ("ZSQD", "CN__ASOS",    "Asia/Shanghai",                  36.0667,  120.3333),
    "San Francisco": ("KSFO", "US_ASOS",     "America/Los_Angeles",            37.6190, -122.3749),
    "Sao Paulo":     ("SBGR", "BR__ASOS",    "America/Sao_Paulo",              -23.4321, -46.4695),
    "Seattle":       ("KSEA", "US_ASOS",     "America/Los_Angeles",            47.4447, -122.3144),
    "Seoul":         ("RKSI", "KR__ASOS",    "Asia/Seoul",                     37.4667,  126.4500),
    "Shanghai":      ("ZSPD", "CN__ASOS",    "Asia/Shanghai",                  31.1167,  121.7667),
    "Shenzhen":      ("ZGSZ", "CN__ASOS",    "Asia/Shanghai",                  22.5500,  114.1000),
    "Singapore":     ("WSSS", "SG__ASOS",    "Asia/Singapore",                  1.3667,  103.9833),
    "Taipei":        ("RCSS", "TW__ASOS",    "Asia/Taipei",                    25.0694,  121.5517),
    "Tel Aviv":      ("LLBG", "IL__ASOS",    "Asia/Jerusalem",                 32.0114,   34.8867),
    "Tokyo":         ("RJTT", "JP__ASOS",    "Asia/Tokyo",                     35.5533,  139.7811),
    "Toronto":       ("CYYZ", "CA_ON_ASOS",  "America/Toronto",                43.6772,  -79.6306),
    "Warsaw":        ("EPWA", "PL__ASOS",    "Europe/Warsaw",                  52.1628,   20.9611),
    "Wellington":    ("NZWN", "NZ__ASOS",    "Pacific/Auckland",              -41.3272,  174.8056),
    "Wuhan":         ("ZHHH", "CN__ASOS",    "Asia/Shanghai",                  30.6200,  114.1300),
}


def get_station(city: str) -> tuple[str, str, str, float, float] | None:
    """Return (icao, network, tz, lat, lon) for a city, or None if not mapped."""
    return CITY_STATIONS.get(city)


# ---------------------------------------------------------------------------
# Same-day model assignment per city  (HRRR ceiling, Phase 0 prereq).
# ---------------------------------------------------------------------------
#
# Polymarket-weather expansion: which rapid-update CAM (convection-allowing
# model) produces a same-day ceiling for each city?
#
# Values:
#   "hrrr"       — High-Resolution Rapid Refresh, 3 km CONUS hourly
#   "icon_d2"    — DWD ICON-D2, ~2 km Central Europe hourly
#   None         — No fresh CAM available; same-day ceiling falls back to
#                  observation-based logic (rising-floor + remaining-solar
#                  heuristic), or is skipped entirely
#
# Edge-of-domain calls (London/Madrid/Helsinki) marked None pending
# explicit ICON-D2 domain verification — see Open Question §10.1 in
# docs/hrrr_ceiling_spec.md.
SAME_DAY_MODEL_BY_CITY: dict[str, str | None] = {
    # CONUS — full HRRR coverage
    "Atlanta":       "hrrr",
    "Austin":        "hrrr",
    "Chicago":       "hrrr",
    "Dallas":        "hrrr",
    "Denver":        "hrrr",
    "Houston":       "hrrr",
    "Los Angeles":   "hrrr",
    "Miami":         "hrrr",
    "NYC":           "hrrr",
    "San Francisco": "hrrr",
    "Seattle":       "hrrr",

    # Central Europe — ICON-D2 covers core of domain
    "Munich":        "icon_d2",
    "Milan":         "icon_d2",
    "Paris":         "icon_d2",
    "Warsaw":        "icon_d2",

    # Edge-of-domain — pending verification
    "London":        None,
    "Madrid":        None,
    "Helsinki":      None,
    "Moscow":        None,

    # No CAM available — observation-based logic only
    "Ankara":        None,
    "Beijing":       None,
    "Buenos Aires":  None,
    "Busan":         None,
    "Cape Town":     None,
    "Chengdu":       None,
    "Chongqing":     None,
    "Guangzhou":     None,
    "Istanbul":      None,
    "Jeddah":        None,
    "Karachi":       None,
    "Kuala Lumpur":  None,
    "Lucknow":       None,
    "Manila":        None,
    "Mexico City":   None,
    "Panama City":   None,
    "Qingdao":       None,
    "Sao Paulo":     None,
    "Seoul":         None,
    "Shanghai":      None,
    "Shenzhen":      None,
    "Singapore":     None,
    "Taipei":        None,
    "Tel Aviv":      None,
    "Tokyo":         None,
    "Toronto":       None,
    "Wellington":    None,
    "Wuhan":         None,
}


def get_same_day_model(city: str) -> str | None:
    """Return the rapid-update model identifier for this city, or None
    if no fresh CAM is available.  Used by the HRRR-ceiling dispatch
    in predict_bins / estimate_day_high_dist."""
    return SAME_DAY_MODEL_BY_CITY.get(city)


def get_station_latlon(city: str) -> tuple[float, float] | None:
    """Just the station coordinates — for hitting forecast APIs."""
    s = CITY_STATIONS.get(city)
    return (s[3], s[4]) if s else None


def get_station_tz(city: str) -> str | None:
    s = CITY_STATIONS.get(city)
    return s[2] if s else None


def get_all_cities() -> list[str]:
    return sorted(CITY_STATIONS.keys())