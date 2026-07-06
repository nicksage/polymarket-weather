"""
weather_api.py — Poll 4 weather sources every ~30 min for the cities that
have active Polymarket highest-temperature markets, and write daily-high/low
forecasts + current observations to db/main.db.

Sources:
    nws  — api.weather.gov  (US cities only, no key, User-Agent required)
    twc  — api.weather.com   (global, TWC_API_KEY)

(Tomorrow.io and Visual Crossing were removed 2026-07-06: their free-tier
quotas can't sustain 30-min polling of ~49 cities. TWC coverage will be
expanded instead.)

City list is derived dynamically each cycle from the `events` table (only
cities with markets on/after today). Cities are mapped to coordinates via the
static gazetteer below; unknown cities are logged and skipped until added.

Runs as systemd service weather-collector.service. Handles SIGTERM/SIGINT
gracefully so `systemctl stop` produces clean RUN SUMMARY entries in the log.
Run a single cycle for testing with:  python collectors/weather_api.py --once
"""
import argparse
import logging
import os
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

LOG_PATH = os.path.join(LOG_DIR, "weather_collector.log")
ACTIVITY_LOG_PATH = os.path.join(LOG_DIR, "activity.log")

WEATHER_INTERVAL = 1800     # 30 minutes — forecasts + observations each cycle
HEALTH_LOG_INTERVAL = 300   # 5 minutes
MAX_LEAD_DAYS = 7           # store daily forecasts out to +7 days
PER_CALL_PAUSE = 0.4        # polite gap between HTTP calls (seconds)

NWS_USER_AGENT = "polymarket-weather/1.0 (nickrable@gmail.com)"

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
# City gazetteer: exact events.city string -> (lat, lon, is_us)
# is_us gates the NWS source (api.weather.gov covers the US only).
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

SOURCES = ["nws", "twc"]

_session = {
    "started_at": None, "startup_ok": False, "cycles": 0, "cycle_errors": 0,
    "forecasts_written": 0, "observations_written": 0,
    "src_ok": {s: 0 for s in SOURCES}, "src_err": {s: 0 for s in SOURCES},
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
    src = " ".join(f"{s}={_session['src_ok'][s]}/{_session['src_err'][s]}" for s in SOURCES)
    logger.info(
        f"HEALTH | uptime={uptime} | cycles={_session['cycles']} "
        f"(errors={_session['cycle_errors']}, last {since} ago) | "
        f"forecasts={_session['forecasts_written']} obs={_session['observations_written']} | "
        f"src ok/err {src}"
    )


def _get(url, params=None, headers=None, timeout=20):
    """GET returning parsed JSON, or None on any error. Logs 429 distinctly."""
    try:
        r = httpx.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 429:
            logger.warning(f"429 rate-limited: {url}")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"GET failed {url}: {e}")
        return None
    finally:
        _stop.wait(timeout=PER_CALL_PAUSE)


# ------------------------------------------------------------------
# City discovery from the events table
# ------------------------------------------------------------------
def active_cities():
    """Distinct events.city with a market on/after today, mapped to coords.

    Returns list of (city, lat, lon, is_us). Unknown cities are logged once
    per cycle and skipped.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT DISTINCT city FROM events WHERE date >= date('now') AND city IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    out, unknown = [], []
    for (city,) in rows:
        meta = CITIES.get(city)
        if meta is None:
            unknown.append(city)
            continue
        lat, lon, is_us = meta
        out.append((city, lat, lon, is_us))
    if unknown:
        logger.warning(f"No gazetteer entry for {len(unknown)} city(ies): {', '.join(sorted(unknown))}")
    return sorted(out)


# ------------------------------------------------------------------
# Source: NWS (api.weather.gov) — US only
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
                # some deployments report C; fall back to raw value
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


# ------------------------------------------------------------------
# Source: The Weather Company (api.weather.com) — global
# ------------------------------------------------------------------
def fetch_twc(city, lat, lon, today):
    if not TWC_API_KEY:
        return [], None
    forecasts, obs = [], None
    geocode = f"{lat},{lon}"
    # 5-day daily forecast (metric: C, km/h)
    data = _get("https://api.weather.com/v3/wx/forecast/daily/5day",
                params={"geocode": geocode, "format": "json", "units": "m",
                        "language": "en-US", "apiKey": TWC_API_KEY})
    if data:
        valid = data.get("validTimeLocal") or []
        highs = data.get("calendarDayTemperatureMax") or []
        lows = data.get("calendarDayTemperatureMin") or []
        dayparts = (data.get("daypart") or [{}])[0] or {}
        precip = dayparts.get("precipChance") or []
        for i, vt in enumerate(valid):
            d = (vt or "")[:10]
            if not d:
                continue
            high_c = _num(highs[i]) if i < len(highs) else None
            low_c = _num(lows[i]) if i < len(lows) else None
            # daypart arrays are 2x length (day/night); take the day slot for this date
            pop = _num(precip[i * 2]) if precip and i * 2 < len(precip) else None
            forecasts.append({
                "target_date": d,
                "high_c": high_c, "low_c": low_c,
                "high_f": c_to_f(high_c), "low_f": c_to_f(low_c),
                "precip_prob": pop, "humidity": None, "wind_kph": None,
            })
    # current observation
    cur = _get("https://api.weather.com/v3/wx/observations/current",
               params={"geocode": geocode, "format": "json", "units": "m",
                       "language": "en-US", "apiKey": TWC_API_KEY})
    if cur:
        temp_c = _num(cur.get("temperature"))
        obs = {
            "temp_c": temp_c, "temp_f": c_to_f(temp_c),
            "humidity": _num(cur.get("relativeHumidity")),
            "wind_kph": _num(cur.get("windSpeed")),
            "conditions": cur.get("wxPhraseLong"),
            "observed_at": cur.get("validTimeLocal"),
        }
    return forecasts, obs


FETCHERS = {
    "nws": fetch_nws,
    "twc": fetch_twc,
}


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------
def _write(fc_rows, obs_rows):
    conn = sqlite3.connect(DB_PATH)
    try:
        if fc_rows:
            conn.executemany(
                """INSERT INTO weather_forecasts
                   (city, source, target_date, lead_days, high_c, low_c, high_f, low_f,
                    precip_prob, humidity, wind_kph, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", fc_rows)
        if obs_rows:
            conn.executemany(
                """INSERT INTO weather_observations
                   (city, source, temp_c, temp_f, humidity, wind_kph, conditions,
                    observed_at, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""", obs_rows)
        conn.commit()
    finally:
        conn.close()
    _session["forecasts_written"] += len(fc_rows)
    _session["observations_written"] += len(obs_rows)


# ------------------------------------------------------------------
# One collection cycle
# ------------------------------------------------------------------
def run_cycle() -> tuple:
    fetched_at = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date()
    cities = active_cities()
    fc_rows, obs_rows = [], []

    for city, lat, lon, is_us in cities:
        if _stop.is_set():
            break
        for source in SOURCES:
            if _stop.is_set():
                break
            if source == "nws" and not is_us:
                continue  # api.weather.gov is US-only
            try:
                forecasts, obs = FETCHERS[source](city, lat, lon, today)
            except Exception as e:
                logger.warning(f"{source} fetch failed for {city}: {e}")
                _session["src_err"][source] += 1
                continue
            got_data = False
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
                    city, source, d, lead,
                    f.get("high_c"), f.get("low_c"), f.get("high_f"), f.get("low_f"),
                    f.get("precip_prob"), f.get("humidity"), f.get("wind_kph"), fetched_at,
                ))
                got_data = True
            if obs:
                obs_rows.append((
                    city, source, obs.get("temp_c"), obs.get("temp_f"),
                    obs.get("humidity"), obs.get("wind_kph"), obs.get("conditions"),
                    obs.get("observed_at"), fetched_at,
                ))
                got_data = True
            if got_data:
                _session["src_ok"][source] += 1
            else:
                _session["src_err"][source] += 1

    _write(fc_rows, obs_rows)
    _session["last_cycle_at"] = datetime.now(timezone.utc)
    logger.info(
        f"Cycle: {len(cities)} cities -> {len(fc_rows)} forecast rows, "
        f"{len(obs_rows)} observation rows"
    )
    return len(fc_rows), len(obs_rows)


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
    logger.info(f"  success:       {success}")
    logger.info(f"  exit reason:   {exit_reason}")
    logger.info(f"  cycles:        {_session['cycles']}")
    logger.info(f"  cycle errors:  {_session['cycle_errors']}")
    logger.info(f"  forecasts:     {_session['forecasts_written']}")
    logger.info(f"  observations:  {_session['observations_written']}")
    for s in SOURCES:
        logger.info(f"  {s:<14} ok={_session['src_ok'][s]} err={_session['src_err'][s]}")
    logger.info(f"  uptime:        {uptime}")
    _append_activity_row(
        "OK" if success else "FAIL",
        uptime=uptime, reason=exit_reason,
        forecasts=_session["forecasts_written"],
        observations=_session["observations_written"],
        cycles=_session["cycles"], cycle_errors=_session["cycle_errors"],
    )


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
    logger.info(f"  sources:  {', '.join(SOURCES)}")
    logger.info(f"  keys:     twc={'y' if TWC_API_KEY else 'n'} (nws=no-key)")
    if not args.once:
        logger.info(f"  cycle every {WEATHER_INTERVAL}s")
    _install_signal_handlers()

    exit_reason = "unknown"
    try:
        run_cycle()
        _session["cycles"] += 1
        _session["startup_ok"] = True
    except Exception as e:
        logger.exception(f"Startup FAILED: {e}")
        _log_run_summary("startup_failed")
        raise

    if args.once:
        _log_run_summary("once")
        return

    now = time.time()
    next_cycle = now + WEATHER_INTERVAL
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
                next_cycle = time.time() + WEATHER_INTERVAL
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
