"""
weather_api.py — Poll weather sources every ~30 min for the cities that have
active Polymarket highest-temperature markets, and persist to db/main.db.

Two data streams:

  NWS (api.weather.gov, US cities only, no key) -> the coarse ensemble tables
      weather_forecasts (daily high/low) + weather_observations (current).

  TWC (api.weather.com, enterprise key) -> full-fidelity per-station tables,
      keyed by the ICAO airport code Polymarket actually resolves against:
        - twc_current        current observations by ICAO (48 fields)
        - twc_hourly         enterprise hourly forecast by ICAO (2day, 42 fields)
        - twc_fifteenminute  15-minute forecast, next ~7h by ICAO (17 fields)
        - twc_probabilistic  probabilistic hourly forecast by ICAO (48h; all 10
                             params x pdf/percentiles/probabilities/prototypes)
      EVERY forecast period and EVERY field is stored on every poll, not just
      the current/most-recent value.

Discovery: each cycle queries the Polymarket Gamma API for active
highest-temperature events, parses the city from the title and the ICAO code
from the market's Wunderground resolutionSource (e.g. .../jinan/ZSJN -> ZSJN).
City -> (lat, lon, is_us) for NWS comes from the static gazetteer below.

Runs as systemd service weather-collector.service. Handles SIGTERM/SIGINT
gracefully so `systemctl stop` produces clean RUN SUMMARY entries in the log.
Run a single cycle for testing with:  python collectors/weather_api.py --once
"""
import argparse
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone, date
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config.env_loader import DB_PATH, LOG_DIR, TWC_API_KEY
from config.cities import local_iso

LOG_PATH = os.path.join(LOG_DIR, "weather_collector.log")
ACTIVITY_LOG_PATH = os.path.join(LOG_DIR, "activity.log")

WEATHER_INTERVAL = 1800     # 30 minutes
HEALTH_LOG_INTERVAL = 300   # 5 minutes
MAX_LEAD_DAYS = 7           # store NWS daily forecasts out to +7 days
PER_CALL_PAUSE = 0.4        # polite gap between HTTP calls (seconds)

NWS_USER_AGENT = "polymarket-weather/1.0 (nickrable@gmail.com)"

GAMMA_BASE = "https://gamma-api.polymarket.com"
TWC_BASE = "https://api.weather.com/v3"
TWC_UNITS = "m"             # metric: degC, km/h, hPa, mm, km
TWC_HOURLY_DURATION = "2day"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("weather_api")

# ------------------------------------------------------------------
# City gazetteer: exact Polymarket city string -> (lat, lon, is_us).
# Only used for NWS (api.weather.gov needs lat/lon and covers the US only).
# TWC needs no coords — it is keyed by the market's ICAO code.
# ------------------------------------------------------------------
CITIES = {
    "Amsterdam":     (52.3728,   4.8936, False),
    "Ankara":        (39.9334,  32.8597, False),
    "Atlanta":       (33.7490, -84.3880, True),
    "Austin":        (30.2672, -97.7431, True),
    "Beijing":       (39.9042, 116.4074, False),
    "Buenos Aires":  (-34.6037, -58.3816, False),
    "Busan":         (35.1796, 129.0756, False),
    "Cape Town":     (-33.9249, 18.4241, False),
    "Chengdu":       (30.5728, 104.0668, False),
    "Chicago":       (41.8781, -87.6298, True),
    "Chongqing":     (29.4316, 106.9123, False),
    "Dallas":        (32.7767, -96.7970, True),
    "Denver":        (39.7392, -104.9903, True),
    "Guangzhou":     (23.1291, 113.2644, False),
    "Helsinki":      (60.1699,  24.9384, False),
    "Hong Kong":     (22.3193, 114.1694, False),
    "Houston":       (29.7604, -95.3698, True),
    "Istanbul":      (41.0082,  28.9784, False),
    "Jeddah":        (21.4858,  39.1925, False),
    "Jinan":         (36.6512, 117.1201, False),
    "Karachi":       (24.8607,  67.0011, False),
    "Kuala Lumpur":  (3.1390,  101.6869, False),
    "London":        (51.5074,  -0.1278, False),
    "Los Angeles":   (34.0522, -118.2437, True),
    "Lucknow":       (26.8467,  80.9462, False),
    "Madrid":        (40.4168,  -3.7038, False),
    "Manila":        (14.5995, 120.9842, False),
    "Mexico City":   (19.4326, -99.1332, False),
    "Miami":         (25.7617, -80.1918, True),
    "Milan":         (45.4642,   9.1900, False),
    "Moscow":        (55.7558,  37.6173, False),
    "Munich":        (48.1351,  11.5820, False),
    "NYC":           (40.7128, -74.0060, True),
    "Panama City":   (8.9824,  -79.5199, False),
    "Paris":         (48.8566,   2.3522, False),
    "Qingdao":       (36.0671, 120.3826, False),
    "San Francisco": (37.7749, -122.4194, True),
    "Sao Paulo":     (-23.5505, -46.6333, False),
    "Seattle":       (47.6062, -122.3321, True),
    "Seoul":         (37.5665, 126.9780, False),
    "Shanghai":      (31.2304, 121.4737, False),
    "Shenzhen":      (22.5431, 114.0579, False),
    "Singapore":     (1.3521,  103.8198, False),
    "Taipei":        (25.0330, 121.5654, False),
    "Tel Aviv":      (32.0853,  34.7818, False),
    "Tokyo":         (35.6762, 139.6503, False),
    "Toronto":       (43.6532, -79.3832, False),
    "Warsaw":        (52.2297,  21.0122, False),
    "Wellington":    (-41.2865, 174.7762, False),
    "Wuhan":         (30.5928, 114.3055, False),
    "Zhengzhou":     (34.7466, 113.6254, False),
}

# Nearest-major-airport ICAO for cities whose Polymarket market resolves against
# a national observatory (not a Wunderground airport), so no ICAO is parseable.
# These are APPROXIMATIONS — the TWC station is not the exact resolution source,
# so treat forecasts/obs for these ICAOs as "near the city" rather than "the
# resolving station". Better than no TWC coverage at all.
CITY_ICAO_OVERRIDE = {
    "Hong Kong": "VHHH",   # resolves via Hong Kong Observatory
    "Istanbul":  "LTFM",   # Istanbul Airport
    "Moscow":    "UUEE",   # Sheremetyevo
    "Tel Aviv":  "LLBG",   # Ben Gurion
}

# Streams tracked in health/summary output.
STREAMS = ["nws", "twc_current", "twc_hourly", "twc_15min", "twc_prob"]

_session = {
    "started_at": None, "startup_ok": False, "cycles": 0, "cycle_errors": 0,
    "forecasts_written": 0, "observations_written": 0,
    "twc_current_rows": 0, "twc_hourly_rows": 0, "twc_15min_rows": 0,
    "twc_prob_rows": 0,
    "src_ok": {s: 0 for s in STREAMS}, "src_err": {s: 0 for s in STREAMS},
    "last_cycle_at": None, "shutdown_signal": None,
}
_stop = threading.Event()

# Cache of NWS /points metadata (grid + station never change per coordinate).
_nws_points_cache = {}


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------
def c_to_f(c):
    return round(c * 9 / 5 + 32, 1) if c is not None else None


def f_to_c(f):
    return round((f - 32) * 5 / 9, 1) if f is not None else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _append_activity_row(status: str, **fields):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts = [ts, "weather", f"{status:<5}"]
    parts.extend(f"{k}={v}" for k, v in fields.items())
    try:
        with open(ACTIVITY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(" | ".join(parts) + "\n")
    except Exception as e:
        logger.warning(f"activity.log write failed: {e}")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def _install_signal_handlers():
    def _shutdown(signum, _frame):
        signame = signal.Signals(signum).name
        logger.info(f"Received {signame} - shutting down")
        _session["shutdown_signal"] = signame
        _stop.set()
    for sig_name in ("SIGTERM", "SIGINT"):
        try:
            signal.signal(getattr(signal, sig_name), _shutdown)
        except (ValueError, OSError, AttributeError):
            pass


def _log_health():
    started = _session["started_at"]
    uptime = _fmt_duration((datetime.now(timezone.utc) - started).total_seconds()) if started else "?"
    last = _session["last_cycle_at"]
    since = _fmt_duration((datetime.now(timezone.utc) - last).total_seconds()) if last else "never"
    src = " ".join(f"{s}={_session['src_ok'][s]}/{_session['src_err'][s]}" for s in STREAMS)
    logger.info(
        f"HEALTH | uptime={uptime} | cycles={_session['cycles']} "
        f"(errors={_session['cycle_errors']}, last {since} ago) | "
        f"nws_fc={_session['forecasts_written']} nws_obs={_session['observations_written']} "
        f"twc_rows(cur/hr/15m/prob)={_session['twc_current_rows']}/{_session['twc_hourly_rows']}/"
        f"{_session['twc_15min_rows']}/{_session['twc_prob_rows']} | "
        f"ok/err {src}"
    )


def _get(url, params=None, headers=None, timeout=20, tries=1):
    """GET returning parsed JSON, or None on any error. Logs 429 distinctly.

    Retries up to `tries` times on transient 503s (TWC returns these on large
    responses) with a short backoff.
    """
    for attempt in range(1, tries + 1):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                logger.warning(f"429 rate-limited: {url}")
                return None
            if r.status_code == 503 and attempt < tries:
                _stop.wait(timeout=2.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < tries:
                _stop.wait(timeout=2.0)
                continue
            logger.warning(f"GET failed {url}: {e}")
            return None
        finally:
            _stop.wait(timeout=PER_CALL_PAUSE)
    return None


# ------------------------------------------------------------------
# Discovery: active markets -> city + ICAO (from Gamma API)
# ------------------------------------------------------------------
_ICAO_RE = re.compile(r"/([A-Z]{4})(?:[/\s\).?]|$)")
_CITY_RE = re.compile(r"temperature in (.+?) on", re.IGNORECASE)


def _fetch_gamma_events():
    all_events, offset = [], 0
    while not _stop.is_set():
        data = _get(f"{GAMMA_BASE}/events", params={
            "tag_slug": "highest-temperature",
            "active": "true", "closed": "false",
            "limit": 100, "offset": offset,
        })
        events = data if isinstance(data, list) else []
        if not events:
            break
        all_events.extend(events)
        if len(events) < 100:
            break
        offset += 100
    return all_events


def _parse_icao(event):
    """Extract the 4-letter ICAO from a market's Wunderground resolution URL."""
    src = event.get("resolutionSource") or ""
    if not src:
        for mkt in event.get("markets", []) or []:
            m = re.search(r"https?://\S*wunderground\.com/\S+", mkt.get("description") or "")
            if m:
                src = m.group(0)
                break
    m = _ICAO_RE.search(src)
    return m.group(1) if m else None


# Sentinels for open-ended market bins ("X or below/above"), in Celsius.
PROB_TEMP_SENT_LO = -100.0
PROB_TEMP_SENT_HI = 100.0


def _parse_range(question: str):
    """Parse a Polymarket temp-market question into whole-degree (low, high).

    Mirrors collectors/polymarket_prices.py. Either bound may be None
    (open-ended). A single value like "be 16" returns (16, 16).
    """
    q = question
    m = re.search(r"between\s+(-?\d+\.?\d*)\s*°?\s*[CF]?\s+and\s+(-?\d+\.?\d*)", q, re.IGNORECASE)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r"(-?\d+\.?\d*)\s*°?\s*[CF]?\s*[-–—]\s*(-?\d+\.?\d*)\s*°?\s*[CF]?", q, re.IGNORECASE)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r"(-?\d+\.?\d*)\s*°?\s*[CF]?\s+or\s+(?:above|higher|more|greater|over)", q, re.IGNORECASE)
    if m: return float(m.group(1)), None
    m = re.search(r"at least\s+(-?\d+\.?\d*)", q, re.IGNORECASE)
    if m: return float(m.group(1)), None
    m = re.search(r"(-?\d+\.?\d*)\s*°?\s*[CF]?\s+or\s+(?:below|lower|less|under)", q, re.IGNORECASE)
    if m: return None, float(m.group(1))
    m = re.search(r"at most\s+(-?\d+\.?\d*)", q, re.IGNORECASE)
    if m: return None, float(m.group(1))
    m = re.search(r"be\s+(?:exactly\s+)?(-?\d+\.?\d*)\s*°?\s*[CF]?(?:\s+on|\?|$)", q, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return v, v
    return None, None


def _market_temp_bins_celsius(markets):
    """Continuous Celsius (lb, ub) ranges for a market's temperature bins.

    Polymarket resolves to whole degrees, so a "16°C" bin means the high rounds
    to 16 -> [15.5, 16.5); a "64-65°F" bin -> [63.5, 65.5)°F. We expand each
    whole-degree label by ±0.5, convert Fahrenheit markets to Celsius (TWC is
    requested in metric), and use sentinels for open-ended bins. Returns a set
    of (lb, ub) tuples so P(high in bin) matches each contract's resolution.
    """
    bins = set()
    for mkt in markets or []:
        q = mkt.get("question", "") or ""
        lo, hi = _parse_range(q)
        if lo is None and hi is None:
            continue
        is_f = ("fahrenheit" in q.lower()) or ("°f" in q.lower()) or ("ºf" in q.lower())
        lb = lo - 0.5 if lo is not None else None
        ub = hi + 0.5 if hi is not None else None
        if is_f:
            lb = f_to_c(lb) if lb is not None else None
            ub = f_to_c(ub) if ub is not None else None
        lb = round(lb, 1) if lb is not None else PROB_TEMP_SENT_LO
        ub = round(ub, 1) if ub is not None else PROB_TEMP_SENT_HI
        if lb < ub:
            bins.add((lb, ub))
    return bins


def discover_targets():
    """Return list of {city, icao, lat, lon, is_us, temp_bins} for active markets.

    temp_bins is the union (across all of the city's active events) of the
    continuous Celsius ranges corresponding to that city's Polymarket
    temperature contracts — used to request the `probabilities` product on the
    exact bins in play.
    """
    events = _fetch_gamma_events()
    seen = {}
    for e in events:
        m = _CITY_RE.search(e.get("title", "") or "")
        if not m:
            continue
        city = m.group(1).strip()
        icao = _parse_icao(e) or CITY_ICAO_OVERRIDE.get(city)
        bins = _market_temp_bins_celsius(e.get("markets", []))
        if city not in seen:
            lat, lon, is_us = CITIES.get(city, (None, None, False))
            seen[city] = {"city": city, "icao": icao, "lat": lat, "lon": lon,
                          "is_us": is_us, "temp_bins": set(bins)}
        else:
            if icao and not seen[city]["icao"]:
                seen[city]["icao"] = icao
            seen[city]["temp_bins"] |= bins
    for t in seen.values():
        t["temp_bins"] = sorted(t["temp_bins"])
    targets = sorted(seen.values(), key=lambda t: t["city"])
    no_icao = [t["city"] for t in targets if not t["icao"]]
    no_coord = [t["city"] for t in targets if t["is_us"] and t["lat"] is None]
    if no_icao:
        logger.warning(f"No ICAO parsed for {len(no_icao)} city(ies): {', '.join(no_icao)}")
    if no_coord:
        logger.warning(f"US city without gazetteer coords: {', '.join(no_coord)}")
    return targets


# ------------------------------------------------------------------
# Source: NWS (api.weather.gov) — US only -> coarse ensemble tables
# ------------------------------------------------------------------
def _nws_point(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _nws_points_cache:
        return _nws_points_cache[key]
    data = _get(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"},
    )
    props = (data or {}).get("properties") or {}
    meta = {
        "forecast": props.get("forecast"),
        "stations": props.get("observationStations"),
    }
    if meta["forecast"]:
        _nws_points_cache[key] = meta
    return meta


def fetch_nws(city, lat, lon, today):
    forecasts, obs = [], None
    meta = _nws_point(lat, lon)
    # --- daily forecast (12h periods, temps in F) ---
    if meta.get("forecast"):
        data = _get(meta["forecast"], headers={"User-Agent": NWS_USER_AGENT,
                                               "Accept": "application/geo+json"})
        periods = ((data or {}).get("properties") or {}).get("periods") or []
        by_date = {}
        for p in periods:
            d = (p.get("startTime") or "")[:10]
            if not d:
                continue
            temp_f = _num(p.get("temperature")) if p.get("temperatureUnit") == "F" else None
            if temp_f is None:
                temp_f = c_to_f(_num(p.get("temperature")))
            pop = ((p.get("probabilityOfPrecipitation") or {}).get("value"))
            slot = by_date.setdefault(d, {"high_f": None, "low_f": None, "pop": None})
            if p.get("isDaytime"):
                slot["high_f"] = temp_f
            else:
                slot["low_f"] = temp_f
            if pop is not None:
                slot["pop"] = max(slot["pop"] or 0, pop)
        for d, slot in by_date.items():
            high_c = f_to_c(slot["high_f"])
            low_c = f_to_c(slot["low_f"])
            forecasts.append({
                "target_date": d,
                "high_c": high_c, "low_c": low_c,
                "high_f": slot["high_f"], "low_f": slot["low_f"],
                "precip_prob": slot["pop"], "humidity": None, "wind_kph": None,
            })
    # --- current observation ---
    if meta.get("stations"):
        sdata = _get(meta["stations"], headers={"User-Agent": NWS_USER_AGENT,
                                               "Accept": "application/geo+json"})
        feats = (sdata or {}).get("features") or []
        if feats:
            sid = (feats[0].get("properties") or {}).get("stationIdentifier")
            if sid:
                odata = _get(
                    f"https://api.weather.gov/stations/{sid}/observations/latest",
                    headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"},
                )
                op = (odata or {}).get("properties") or {}
                temp_c = _num((op.get("temperature") or {}).get("value"))
                humidity = _num((op.get("relativeHumidity") or {}).get("value"))
                wind = _num((op.get("windSpeed") or {}).get("value"))  # km/h per wmoUnit
                obs = {
                    "temp_c": temp_c, "temp_f": c_to_f(temp_c),
                    "humidity": humidity, "wind_kph": wind,
                    "conditions": op.get("textDescription"),
                    "observed_at": op.get("timestamp"),
                }
    return forecasts, obs


def _write_nws(fc_rows, obs_rows):
    conn = sqlite3.connect(DB_PATH)
    try:
        if fc_rows:
            conn.executemany(
                """INSERT INTO weather_forecasts
                   (city, source, target_date, lead_days, high_c, low_c, high_f, low_f,
                    precip_prob, humidity, wind_kph, fetched_at, fetched_at_local)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", fc_rows)
        if obs_rows:
            conn.executemany(
                """INSERT INTO weather_observations
                   (city, source, temp_c, temp_f, humidity, wind_kph, conditions,
                    observed_at, fetched_at, fetched_at_local)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""", obs_rows)
        conn.commit()
    finally:
        conn.close()
    _session["forecasts_written"] += len(fc_rows)
    _session["observations_written"] += len(obs_rows)


def collect_nws(targets, fetched_at, today):
    fc_rows, obs_rows = [], []
    for t in targets:
        if _stop.is_set():
            break
        if not (t["is_us"] and t["lat"] is not None):
            continue
        city = t["city"]
        f_local = local_iso(fetched_at, city)
        try:
            forecasts, obs = fetch_nws(city, t["lat"], t["lon"], today)
        except Exception as e:
            logger.warning(f"nws fetch failed for {city}: {e}")
            _session["src_err"]["nws"] += 1
            continue
        got = False
        for f in forecasts:
            d = f.get("target_date")
            if not d:
                continue
            try:
                lead = date.fromisoformat(d).toordinal() - today.toordinal()
            except ValueError:
                lead = None
            if lead is not None and (lead < 0 or lead > MAX_LEAD_DAYS):
                continue
            fc_rows.append((
                city, "nws", d, lead,
                f.get("high_c"), f.get("low_c"), f.get("high_f"), f.get("low_f"),
                f.get("precip_prob"), f.get("humidity"), f.get("wind_kph"),
                fetched_at, f_local,
            ))
            got = True
        if obs:
            obs_rows.append((
                city, "nws", obs.get("temp_c"), obs.get("temp_f"),
                obs.get("humidity"), obs.get("wind_kph"), obs.get("conditions"),
                obs.get("observed_at"), fetched_at, f_local,
            ))
            got = True
        _session["src_ok"]["nws" ] += 1 if got else 0
        _session["src_err"]["nws"] += 0 if got else 1
    _write_nws(fc_rows, obs_rows)
    return len(fc_rows), len(obs_rows)


# ------------------------------------------------------------------
# Source: TWC (api.weather.com) — full-fidelity capture by ICAO
# Field maps: (json_key, db_column). Every documented/observed field is kept.
# ------------------------------------------------------------------
TWC_CURRENT_FIELDS = [
    ("validTimeLocal", "valid_time_local"),
    ("validTimeUtc", "valid_time_utc"),
    ("expirationTimeUtc", "expiration_time_utc"),
    ("dayOfWeek", "day_of_week"),
    ("dayOrNight", "day_or_night"),
    ("temperature", "temperature"),
    ("temperatureFeelsLike", "temperature_feels_like"),
    ("temperatureDewPoint", "temperature_dew_point"),
    ("temperatureHeatIndex", "temperature_heat_index"),
    ("temperatureWindChill", "temperature_wind_chill"),
    ("temperatureWetBulbGlobe", "temperature_wet_bulb_globe"),
    ("temperatureMax24Hour", "temperature_max_24hour"),
    ("temperatureMin24Hour", "temperature_min_24hour"),
    ("temperatureMaxSince7Am", "temperature_max_since_7am"),
    ("temperatureChange24Hour", "temperature_change_24hour"),
    ("relativeHumidity", "relative_humidity"),
    ("precip1Hour", "precip_1hour"),
    ("precip6Hour", "precip_6hour"),
    ("precip24Hour", "precip_24hour"),
    ("snow1Hour", "snow_1hour"),
    ("snow6Hour", "snow_6hour"),
    ("snow24Hour", "snow_24hour"),
    ("windSpeed", "wind_speed"),
    ("windDirection", "wind_direction"),
    ("windDirectionCardinal", "wind_direction_cardinal"),
    ("windGust", "wind_gust"),
    ("pressureAltimeter", "pressure_altimeter"),
    ("pressureMeanSeaLevel", "pressure_mean_sea_level"),
    ("pressureChange", "pressure_change"),
    ("pressureTendencyCode", "pressure_tendency_code"),
    ("pressureTendencyTrend", "pressure_tendency_trend"),
    ("cloudCover", "cloud_cover"),
    ("cloudCoverPhrase", "cloud_cover_phrase"),
    ("cloudCeiling", "cloud_ceiling"),
    ("visibility", "visibility"),
    ("uvIndex", "uv_index"),
    ("uvDescription", "uv_description"),
    ("iconCode", "icon_code"),
    ("iconCodeExtend", "icon_code_extend"),
    ("wxPhraseLong", "wx_phrase_long"),
    ("wxPhraseMedium", "wx_phrase_medium"),
    ("wxPhraseShort", "wx_phrase_short"),
    ("obsQualifierCode", "obs_qualifier_code"),
    ("obsQualifierSeverity", "obs_qualifier_severity"),
    ("sunriseTimeLocal", "sunrise_time_local"),
    ("sunriseTimeUtc", "sunrise_time_utc"),
    ("sunsetTimeLocal", "sunset_time_local"),
    ("sunsetTimeUtc", "sunset_time_utc"),
]

TWC_HOURLY_FIELDS = [
    ("validTimeLocal", "valid_time_local"),
    ("validTimeUtc", "valid_time_utc"),
    ("expirationTimeUtc", "expiration_time_utc"),
    ("dayOfWeek", "day_of_week"),
    ("dayOrNight", "day_or_night"),
    ("temperature", "temperature"),
    ("temperatureDewPoint", "temperature_dew_point"),
    ("temperatureFeelsLike", "temperature_feels_like"),
    ("temperatureHeatIndex", "temperature_heat_index"),
    ("temperatureWindChill", "temperature_wind_chill"),
    ("temperatureWetBulbGlobe", "temperature_wet_bulb_globe"),
    ("relativeHumidity", "relative_humidity"),
    ("precipChance", "precip_chance"),
    ("precipType", "precip_type"),
    ("qpf", "qpf"),
    ("qpfRain", "qpf_rain"),
    ("qpfSnow", "qpf_snow"),
    ("qpfIce", "qpf_ice"),
    ("conditionalProbabilityRain", "cond_prob_rain"),
    ("conditionalProbabilitySnow", "cond_prob_snow"),
    ("conditionalProbabilitySleet", "cond_prob_sleet"),
    ("conditionalProbabilityFreezingRain", "cond_prob_freezing_rain"),
    ("conditionalProbabilityThunder", "cond_prob_thunder"),
    ("windSpeed", "wind_speed"),
    ("windDirection", "wind_direction"),
    ("windDirectionCardinal", "wind_direction_cardinal"),
    ("windGust", "wind_gust"),
    ("pressureAltimeter", "pressure_altimeter"),
    ("pressureMeanSeaLevel", "pressure_mean_sea_level"),
    ("cloudCover", "cloud_cover"),
    ("ceiling", "ceiling"),
    ("scatteredCloudBaseHeight", "scattered_cloud_base_height"),
    ("visibility", "visibility"),
    ("uvIndex", "uv_index"),
    ("uvDescription", "uv_description"),
    ("iconCode", "icon_code"),
    ("iconCodeExtend", "icon_code_extend"),
    ("wxPhraseLong", "wx_phrase_long"),
    ("wxPhraseShort", "wx_phrase_short"),
    ("wxString", "wx_string"),
    ("wxSeverity", "wx_severity"),
    ("qualifierSet", "qualifier_set"),
]

TWC_FIFTEEN_FIELDS = [
    ("validTimeLocal", "valid_time_local"),
    ("dayOfWeek", "day_of_week"),
    ("temperature", "temperature"),
    ("temperatureFeelsLike", "temperature_feels_like"),
    ("relativeHumidity", "relative_humidity"),
    ("precipChance", "precip_chance"),
    ("precipRate", "precip_rate"),
    ("precipType", "precip_type"),
    ("snowRate", "snow_rate"),
    ("windSpeed", "wind_speed"),
    ("windDirection", "wind_direction"),
    ("windDirectionCardinal", "wind_direction_cardinal"),
    ("iconCode", "icon_code"),
    ("iconCodeExtend", "icon_code_extend"),
    ("wxPhraseLong", "wx_phrase_long"),
    ("wxPhraseShort", "wx_phrase_short"),
    ("wxSeverity", "wx_severity"),
]


# --- Probabilistic hourly forecast config ---------------------------------
# All 10 supported parameters, all 4 products, 48h horizon. The endpoint
# returns nothing unless products are explicitly requested per parameter, and
# a single all-products request is too large (503), so each product is fetched
# in its own call. Multiple parameters are separated by ';' in the spec string.
PROB_PARAMS = [
    "temperature", "temperatureDewPoint", "relativeHumidity",
    "qpf", "qpfSnow", "windSpeed", "windGust", "windDirection",
    "ceiling", "visibility",
]
PROB_HOURS = 48             # forecast horizon; 48h covers today + tomorrow's markets
PROB_RESOLUTION = "medium"   # discretePdfs / percentiles bin resolution
PROB_PROTOTYPE_N = 100        # ensemble member traces per parameter (max 100)

# Generic [lb, ub] bands per parameter for the `probabilities` product
# (metric units). The PDF also lets any range be computed offline; these are
# convenience bands. Polymarket's own bins vary per market, so we use fixed ones.
PROB_BANDS = {
    "temperature":         [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40), (40, 50)],
    "temperatureDewPoint": [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40)],
    "relativeHumidity":    [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)],
    "qpf":                 [(0, 1), (1, 5), (5, 10), (10, 25), (25, 50)],
    "qpfSnow":             [(0, 1), (1, 5), (5, 10), (10, 25)],
    "windSpeed":           [(0, 10), (10, 20), (20, 40), (40, 60), (60, 100)],
    "windGust":            [(0, 20), (20, 40), (40, 60), (60, 100)],
    "windDirection":       [(0, 90), (90, 180), (180, 270), (270, 360)],
    "ceiling":             [(0, 500), (500, 1000), (1000, 3000), (3000, 10000)],
    "visibility":          [(0, 1), (1, 5), (5, 10), (10, 20)],
}

# Static products (same spec every call). (short name, query-param, response key, spec)
PROB_PRODUCTS = [
    ("pdf", "discretePdfs", "discretePdfs",
     ";".join(f"{p}:{PROB_RESOLUTION}" for p in PROB_PARAMS)),
    ("percentiles", "percentiles", "percentiles",
     ";".join(f"{p}:{PROB_RESOLUTION}" for p in PROB_PARAMS)),
    ("prototypes", "prototypes", "prototypes",
     ";".join(f"{p}:{PROB_PROTOTYPE_N}" for p in PROB_PARAMS)),
]


def _fmt_band(lb, ub):
    """Format a band edge without trailing .0 (keeps the spec string compact)."""
    def f(x):
        return str(int(x)) if float(x).is_integer() else str(x)
    return f"{f(lb)},{f(ub)}"


def _prob_probabilities_spec(temp_bins):
    """Build the `probabilities` spec: temperature uses the live Polymarket
    market bins for this city (Celsius); the other params use generic bands."""
    temp_pairs = temp_bins if temp_bins else PROB_BANDS["temperature"]
    parts = ["temperature:" + ":".join(_fmt_band(lb, ub) for lb, ub in temp_pairs)]
    for p in PROB_PARAMS:
        if p == "temperature":
            continue
        parts.append(p + ":" + ":".join(_fmt_band(lb, ub) for lb, ub in PROB_BANDS[p]))
    return ";".join(parts)

PROB_META_FIELDS = [
    ("initTime", "init_time"), ("procTime", "proc_time"),
    ("latitude", "latitude"), ("longitude", "longitude"),
    ("elevation", "elevation"), ("landuse", "landuse"),
    ("spatialApp", "spatial_app"), ("version", "version"),
    ("expires", "expires"), ("requestId", "request_id"),
]


def _twc_val(v):
    return json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v


def _twc_get(path, icao):
    return _get(f"{TWC_BASE}{path}", params={
        "icaoCode": icao, "units": TWC_UNITS,
        "language": "en-US", "format": "json", "apiKey": TWC_API_KEY,
    })


def _insert_series(conn, table, static, field_map, data):
    """Insert one row per forecast period from a dict of parallel arrays."""
    length = 0
    for jk, _ in field_map:
        v = data.get(jk)
        if isinstance(v, list):
            length = max(length, len(v))
    if not length:
        return 0
    cols = list(static.keys()) + [c for _, c in field_map]
    placeholders = ",".join("?" * len(cols))
    base = list(static.values())
    rows = []
    for i in range(length):
        vals = list(base)
        for jk, _ in field_map:
            arr = data.get(jk)
            vals.append(_twc_val(arr[i]) if isinstance(arr, list) and i < len(arr) else None)
        rows.append(vals)
    conn.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", rows)
    return len(rows)


def _insert_one(conn, table, static, field_map, data):
    """Insert a single row from a dict of scalar fields."""
    cols = list(static.keys()) + [c for _, c in field_map]
    placeholders = ",".join("?" * len(cols))
    vals = list(static.values()) + [_twc_val(data.get(jk)) for jk, _ in field_map]
    conn.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", vals)
    return 1


def _store_prob(conn, city, icao, product_short, resp_key, resp, fetched_at, fetched_at_local):
    """Store one row per (product, parameter) from a probabilistic response.

    `data` keeps that parameter's full payload as JSON — lossless. probabilities
    returns several entries per parameter (one per band); they are grouped into
    a JSON list. pdf/percentiles/prototypes return a single entry per parameter.
    """
    md = resp.get("metadata") or {}
    fc = resp.get("forecasts1Hour") or {}
    fcst_valid = json.dumps(fc.get("fcstValid") or [], separators=(",", ":"))
    by_param = {}
    for e in fc.get(resp_key) or []:
        p = e.get("parameter")
        if p is None:
            continue
        by_param.setdefault(p, []).append({k: v for k, v in e.items() if k != "parameter"})
    if not by_param:
        return 0
    meta_vals = [md.get(jk) for jk, _ in PROB_META_FIELDS]
    cols = (["city", "icao", "units", "hours", "product", "parameter"]
            + [c for _, c in PROB_META_FIELDS]
            + ["fcst_valid", "data", "fetched_at", "fetched_at_local"])
    placeholders = ",".join("?" * len(cols))
    rows = []
    for p, items in by_param.items():
        payload = items[0] if len(items) == 1 else items
        rows.append([city, icao, TWC_UNITS, PROB_HOURS, product_short, p]
                    + meta_vals
                    + [fcst_valid, json.dumps(payload, separators=(",", ":")),
                       fetched_at, fetched_at_local])
    conn.executemany(
        f"INSERT INTO twc_probabilistic ({','.join(cols)}) VALUES ({placeholders})", rows)
    return len(rows)


def collect_prob(conn, city, icao, temp_bins, fetched_at, fetched_at_local):
    """Fetch all 4 probabilistic products (each a separate call) for one ICAO.

    The `probabilities` product requests temperature on this city's live
    Polymarket market bins; the other three products use static specs.
    """
    products = list(PROB_PRODUCTS) + [
        ("probabilities", "probabilities", "probabilities",
         _prob_probabilities_spec(temp_bins)),
    ]
    added = 0
    for product_short, api_param, resp_key, spec in products:
        if _stop.is_set():
            break
        resp = _get(f"{TWC_BASE}/wx/forecast/probabilistic", params={
            "icaoCode": icao, "units": TWC_UNITS, "format": "json",
            "apiKey": TWC_API_KEY, "hours": str(PROB_HOURS), api_param: spec,
        }, timeout=60, tries=3)
        if isinstance(resp, dict) and resp.get("forecasts1Hour"):
            n = _store_prob(conn, city, icao, product_short, resp_key, resp,
                            fetched_at, fetched_at_local)
            added += n
            _session["src_ok"]["twc_prob"] += 1 if n else 0
            _session["src_err"]["twc_prob"] += 0 if n else 1
        else:
            _session["src_err"]["twc_prob"] += 1
    return added


def collect_twc(targets, fetched_at):
    if not TWC_API_KEY:
        logger.warning("TWC_API_KEY not set — skipping TWC")
        return 0, 0, 0, 0
    n_cur = n_hr = n_15 = n_prob = 0
    conn = sqlite3.connect(DB_PATH)
    try:
        for t in targets:
            if _stop.is_set():
                break
            icao, city = t["icao"], t["city"]
            if not icao:
                continue
            f_local = local_iso(fetched_at, city)
            base = {"city": city, "icao": icao, "units": TWC_UNITS}

            cur = _twc_get("/wx/observations/current", icao)
            if isinstance(cur, dict) and cur:
                n_cur += _insert_one(conn, "twc_current",
                                     {**base, "fetched_at": fetched_at,
                                      "fetched_at_local": f_local},
                                     TWC_CURRENT_FIELDS, cur)
                _session["src_ok"]["twc_current"] += 1
            else:
                _session["src_err"]["twc_current"] += 1
            if _stop.is_set():
                break

            hourly = _twc_get(f"/wx/forecast/hourly/{TWC_HOURLY_DURATION}/enterprise", icao)
            if isinstance(hourly, dict) and hourly:
                added = _insert_series(conn, "twc_hourly",
                                       {**base, "duration": TWC_HOURLY_DURATION,
                                        "fetched_at": fetched_at,
                                        "fetched_at_local": f_local},
                                       TWC_HOURLY_FIELDS, hourly)
                n_hr += added
                _session["src_ok"]["twc_hourly"] += 1 if added else 0
                _session["src_err"]["twc_hourly"] += 0 if added else 1
            else:
                _session["src_err"]["twc_hourly"] += 1
            if _stop.is_set():
                break

            fifteen = _twc_get("/wx/forecast/fifteenminute", icao)
            if isinstance(fifteen, dict) and fifteen:
                added = _insert_series(conn, "twc_fifteenminute",
                                       {**base, "fetched_at": fetched_at,
                                        "fetched_at_local": f_local},
                                       TWC_FIFTEEN_FIELDS, fifteen)
                n_15 += added
                _session["src_ok"]["twc_15min"] += 1 if added else 0
                _session["src_err"]["twc_15min"] += 0 if added else 1
            else:
                _session["src_err"]["twc_15min"] += 1
            if _stop.is_set():
                break

            n_prob += collect_prob(conn, city, icao, t.get("temp_bins"),
                                   fetched_at, f_local)
            # Commit per city so the single SQLite write lock is released ~51x
            # per cycle instead of held for the whole ~5-min cycle — otherwise
            # the price collector's writes fail with "database is locked".
            conn.commit()
        conn.commit()
    finally:
        conn.close()
    _session["twc_current_rows"] += n_cur
    _session["twc_hourly_rows"] += n_hr
    _session["twc_15min_rows"] += n_15
    _session["twc_prob_rows"] += n_prob
    return n_cur, n_hr, n_15, n_prob


# ------------------------------------------------------------------
# One collection cycle
# ------------------------------------------------------------------
def run_cycle() -> tuple:
    fetched_at = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date()
    targets = discover_targets()

    n_fc, n_obs = collect_nws(targets, fetched_at, today)
    n_cur, n_hr, n_15, n_prob = collect_twc(targets, fetched_at)

    _session["last_cycle_at"] = datetime.now(timezone.utc)
    logger.info(
        f"Cycle: {len(targets)} markets | NWS {n_fc} fc / {n_obs} obs | "
        f"TWC current={n_cur} hourly={n_hr} 15min={n_15} prob={n_prob}"
    )
    return targets, n_fc, n_obs, n_cur, n_hr, n_15, n_prob


def _log_run_summary(exit_reason: str):
    started = _session["started_at"]
    uptime = _fmt_duration((datetime.now(timezone.utc) - started).total_seconds()) if started else "?"
    success = _session["startup_ok"] and (
        exit_reason in ("keyboard_interrupt", "loop_exited", "once")
        or exit_reason.startswith("signal:")
    )
    logger.info("-" * 72)
    logger.info("  RUN SUMMARY")
    logger.info("-" * 72)
    logger.info(f"  success:        {success}")
    logger.info(f"  exit reason:    {exit_reason}")
    logger.info(f"  cycles:         {_session['cycles']}")
    logger.info(f"  cycle errors:   {_session['cycle_errors']}")
    logger.info(f"  nws forecasts:  {_session['forecasts_written']}")
    logger.info(f"  nws obs:        {_session['observations_written']}")
    logger.info(f"  twc current:    {_session['twc_current_rows']} rows")
    logger.info(f"  twc hourly:     {_session['twc_hourly_rows']} rows")
    logger.info(f"  twc 15-minute:  {_session['twc_15min_rows']} rows")
    logger.info(f"  twc prob:       {_session['twc_prob_rows']} rows")
    for s in STREAMS:
        logger.info(f"  {s:<14} ok={_session['src_ok'][s]} err={_session['src_err'][s]}")
    logger.info(f"  uptime:         {uptime}")
    _append_activity_row(
        "OK" if success else "FAIL",
        uptime=uptime, reason=exit_reason,
        nws_fc=_session["forecasts_written"], nws_obs=_session["observations_written"],
        twc_cur=_session["twc_current_rows"], twc_hr=_session["twc_hourly_rows"],
        twc_15=_session["twc_15min_rows"], twc_prob=_session["twc_prob_rows"],
        cycles=_session["cycles"], cycle_errors=_session["cycle_errors"],
    )


def _next_aligned(now_ts):
    """Epoch of the next wall-clock boundary strictly after now_ts.

    WEATHER_INTERVAL (1800s) evenly divides an hour and the Unix epoch is
    aligned to :00:00 UTC, so multiples of it land exactly on :00 and :30 of
    every hour (globally, since minute boundaries are timezone-independent).
    """
    return (int(now_ts) // WEATHER_INTERVAL + 1) * WEATHER_INTERVAL


def main():
    parser = argparse.ArgumentParser(description="Weather API collector")
    parser.add_argument("--once", action="store_true",
                        help="Run a single collection cycle and exit (for testing)")
    args = parser.parse_args()

    _session["started_at"] = datetime.now(timezone.utc)
    _append_activity_row("START", pid=os.getpid(), mode="once" if args.once else "loop")
    logger.info("=" * 72)
    logger.info("  WEATHER API COLLECTOR - RUN START")
    logger.info("=" * 72)
    logger.info(f"  db path:  {DB_PATH}")
    logger.info(f"  log dir:  {LOG_DIR}")
    logger.info(f"  streams:  nws (coarse) + twc current/hourly({TWC_HOURLY_DURATION})/15min/"
                f"prob({PROB_HOURS}h,{len(PROB_PARAMS)}params) (ICAO)")
    logger.info(f"  twc key:  {'y' if TWC_API_KEY else 'n'}   (nws=no-key)")
    if not args.once:
        logger.info("  cycle aligned to wall clock :00 and :30 (first run at next boundary)")
    _install_signal_handlers()

    exit_reason = "unknown"
    if args.once:
        try:
            run_cycle()
            _session["cycles"] += 1
            _session["startup_ok"] = True
        except Exception as e:
            logger.exception(f"Startup FAILED: {e}")
            _log_run_summary("startup_failed")
            raise
        _log_run_summary("once")
        return

    # Loop mode: no immediate startup cycle — the first run fires at the next
    # aligned :00/:30 boundary, so every stored row lands on the cadence.
    _session["startup_ok"] = True
    now = time.time()
    next_cycle = _next_aligned(now)
    next_health = now + HEALTH_LOG_INTERVAL

    try:
        while not _stop.is_set():
            now = time.time()
            if now >= next_cycle:
                try:
                    run_cycle()
                    _session["cycles"] += 1
                except Exception as e:
                    logger.warning(f"Cycle error: {e}")
                    _session["cycle_errors"] += 1
                next_cycle = _next_aligned(time.time())
            if now >= next_health:
                _log_health()
                next_health = time.time() + HEALTH_LOG_INTERVAL
            sleep_s = max(0.5, min(30.0, min(next_cycle, next_health) - time.time()))
            _stop.wait(timeout=sleep_s)
        exit_reason = f"signal:{_session['shutdown_signal']}" if _session["shutdown_signal"] else "loop_exited"
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        exit_reason = f"error: {e}"
    finally:
        _stop.set()
        _log_run_summary(exit_reason)


if __name__ == "__main__":
    main()
