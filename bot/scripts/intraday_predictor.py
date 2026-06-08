"""
intraday_predictor.py — Real-time bin-probability predictor for active
US Polymarket "highest temperature" markets, with a visual dashboard.

Architecture (per city, per moment):

  PRIOR    = Open-Meteo daily-max forecast at the settlement station
             coords, treated as N(forecast_high, forecast_sigma²).
             forecast_sigma narrows as we approach/pass the forecast peak.

  EVIDENCE = (1) live observations from the settlement station via the
             NWS API (the same NOAA METAR feed that Wunderground / Poly-
             market read from for US airports).
             (2) Today's neighbor observations + their wind alignment vs
             the city's prevailing wind, from the cached neighbor_obs.db.

  POSTERIOR = truncated normal: P(day_high in bin) constrained to
              day_high >= observed_max_so_far.  Neighbor "upwind has
              peaked + is cooling" boosts confidence in current bin.

  EDGE      = our_p[bin] − polymarket_yes_price[bin].  Buy when > margin.

Produces a single self-contained HTML dashboard with:
  - One card per city: current temp / wind / forecast high / observed high
  - Sortable table of every (city, bin) decision with our P vs market P,
    edge, and recommendation
  - Filters by city, date, recommendation, min edge

Usage:
    cd bot
    python -m scripts.intraday_predictor                       # all US cities, today
    python -m scripts.intraday_predictor --city Dallas         # one city
    python -m scripts.intraday_predictor --min-edge 0.10
    python -m scripts.intraday_predictor --html data/intraday.html
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS  # type: ignore
from polymarket  import search_temp_high_events  # type: ignore
try:
    from scripts.find_nearby_stations import US_CITY_STATES  # type: ignore
except Exception:
    US_CITY_STATES = {}

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("intraday")
logging.getLogger("httpx").setLevel(logging.WARNING)

NEIGHBOR_DB    = os.path.join(_BOT_DIR, "data", "neighbor_obs.db")
DEFAULT_OUT    = os.path.join(_BOT_DIR, "data", "intraday_dashboard.html")
NWS_BASE       = "https://api.weather.gov"
NWS_USER_AGENT = "polymarket-weather-bot/1.0 (intraday-predictor)"
OPENMETEO_URL  = "https://api.open-meteo.com/v1/forecast"

# Forecast sigma — fallback used when a city isn't in the per-city
# calibration file (data/forecast_calibration.json).  Overridable via
# the PREDICTOR_DEFAULT_SIGMA_C env var in .env.
DEFAULT_FORECAST_SIGMA_C = float(os.getenv("PREDICTOR_DEFAULT_SIGMA_C", "2.0"))

# Per-city σ calibration — loaded from data/forecast_calibration.json
# (produced by scripts.forecast_rmse_calibration).  Falls back to the
# default above for any city not present in the file or if the file
# doesn't exist yet.
CALIBRATION_PATH = os.path.join(_BOT_DIR, "data", "forecast_calibration.json")
_CALIBRATION: dict = {}


def _load_calibration() -> None:
    """Loaded once at script start.  Re-run scripts.forecast_rmse_calibration
    to refresh the file."""
    global _CALIBRATION
    if not os.path.exists(CALIBRATION_PATH):
        log.info(f"No forecast calibration file at {CALIBRATION_PATH} — "
                  f"using DEFAULT_FORECAST_SIGMA_C={DEFAULT_FORECAST_SIGMA_C} "
                  "for all cities.  Run scripts.forecast_rmse_calibration "
                  "to generate per-city σ values.")
        return
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as fh:
            _CALIBRATION = json.load(fh)
        n = len(_CALIBRATION.get("by_city", {}))
        log.info(f"Loaded per-city σ calibration for {n} cities "
                  f"(generated {_CALIBRATION.get('generated_at', '?')[:19]})")
    except Exception as e:
        log.warning(f"Failed to load {CALIBRATION_PATH}: {e}.  Using default σ.")
        _CALIBRATION = {}


def get_city_sigma(city: str) -> float:
    """Return the σ to use for this city's prior, with safe fallback."""
    entry = (_CALIBRATION.get("by_city") or {}).get(city)
    if not entry:
        return DEFAULT_FORECAST_SIGMA_C
    s = entry.get("sigma")
    if s is None or s <= 0:
        return DEFAULT_FORECAST_SIGMA_C
    return float(s)


# ---------------------------------------------------------------------------
# Statistics helpers — truncated normal CDF without scipy
# ---------------------------------------------------------------------------

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def truncated_normal_prob(lo: float | None, hi: float | None,
                           mu: float, sigma: float,
                           truncate_at: float) -> float:
    """Probability that X falls in [lo, hi], given X ~ N(mu, sigma²) and
    X >= truncate_at.  lo/hi can be None for open-ended intervals."""
    # Effective lower bound
    eff_lo = max(lo, truncate_at) if lo is not None else truncate_at
    if hi is not None and hi <= truncate_at:
        return 0.0   # bin entirely below truncation
    if hi is not None and eff_lo >= hi:
        return 0.0

    Z = 1.0 - normal_cdf(truncate_at, mu, sigma)
    if Z <= 1e-12:
        # Almost no mass above truncate — everything goes to "current or higher"
        # In practice this means we're confident the day high is at observed
        if hi is None:
            return 1.0   # "or higher" catches the residual
        return 1.0 if (lo is not None and lo <= truncate_at < hi) else 0.0

    cdf_lo = normal_cdf(eff_lo, mu, sigma)
    cdf_hi = normal_cdf(hi, mu, sigma) if hi is not None else 1.0
    return max(0.0, (cdf_hi - cdf_lo) / Z)


# ---------------------------------------------------------------------------
# NWS API — live observations
# ---------------------------------------------------------------------------

def fetch_nws_today_obs(icao: str, tz_str: str) -> list[dict]:
    """Return list of {hour_local, temp_c, wind_dir_deg, timestamp_utc} for
    today's observations from the given US airport station.  Uses the NWS
    public API — same NOAA METAR feed Wunderground reads."""
    tz = ZoneInfo(tz_str)
    now_local = datetime.now(tz)
    today_local = now_local.date()
    # Start at local midnight, convert to UTC for the API
    start_local = datetime.combine(today_local, datetime.min.time(), tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    url = f"{NWS_BASE}/stations/{icao}/observations"
    try:
        r = httpx.get(url, params={"start": start_utc}, headers=headers, timeout=30)
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as e:
        log.warning(f"NWS fetch failed for {icao}: {e}")
        return []

    # Group by local hour, keeping the MAX temp per hour (matches our
    # max-per-hour fix in station_obs_pull)
    by_hour: dict[int, dict] = {}
    for f in features:
        props = f.get("properties") or {}
        ts_str = props.get("timestamp")
        temp_obj = props.get("temperature") or {}
        wind_obj = props.get("windDirection") or {}
        t_c = temp_obj.get("value")
        wd  = wind_obj.get("value")
        if not ts_str or t_c is None:
            continue
        try:
            ts_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        ts_local = ts_utc.astimezone(tz)
        if ts_local.date() != today_local:
            continue
        h = ts_local.hour
        existing = by_hour.get(h)
        if existing is None or t_c > existing["temp_c"]:
            by_hour[h] = {
                "hour_local":    h,
                "temp_c":        float(t_c),
                "wind_dir_deg":  float(wd) if wd is not None else None,
                "timestamp_utc": ts_str,
            }
    return sorted(by_hour.values(), key=lambda r: r["hour_local"])


# ---------------------------------------------------------------------------
# NWS — forecast prior (hourly grid forecast for the settlement station coords)
# ---------------------------------------------------------------------------

# Cache the points → grid resolution since it rarely changes per (lat,lon)
_NWS_POINTS_CACHE: dict[tuple[float, float], str] = {}


def _sunset_local_hour(d: date, lat: float, lon: float, tz_str: str) -> int:
    """NOAA solar-position sunset calc.  No external API; accurate to ~5 min.
    Used because NWS's forecast endpoint doesn't include sunset and we don't
    want to call a second service just for that."""
    doy = d.timetuple().tm_yday
    gamma = 2 * math.pi * (doy - 1) / 365
    # Solar declination (radians)
    decl = (0.006918 - 0.399912*math.cos(gamma) + 0.070257*math.sin(gamma)
            - 0.006758*math.cos(2*gamma) + 0.000907*math.sin(2*gamma)
            - 0.002697*math.cos(3*gamma) + 0.00148*math.sin(3*gamma))
    # Equation of time (minutes)
    eot = 229.18 * (
        0.000075 + 0.001868*math.cos(gamma) - 0.032077*math.sin(gamma)
        - 0.014615*math.cos(2*gamma) - 0.040849*math.sin(2*gamma)
    )
    # Hour angle for sunset (zenith = 90.833° accounts for atmospheric refraction)
    lat_r = math.radians(lat)
    try:
        cos_ha = ((math.cos(math.radians(90.833))
                    - math.sin(lat_r) * math.sin(decl))
                   / (math.cos(lat_r) * math.cos(decl)))
    except ZeroDivisionError:
        return 19
    if cos_ha > 1:  return 0     # polar night
    if cos_ha < -1: return 23    # polar day
    ha_deg = math.degrees(math.acos(cos_ha))
    # Sunset in minutes since UTC midnight
    sunset_utc_min = 720 + 4 * (-lon) + ha_deg * 4 - eot
    sunset_utc = (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
                   + timedelta(minutes=sunset_utc_min))
    return sunset_utc.astimezone(ZoneInfo(tz_str)).hour


def fetch_nws_today_forecast(lat: float, lon: float, tz_str: str) -> dict:
    """Hourly forecast from the official NWS API for the given coords.

    This is the SAME data source Wunderground uses for US settlement
    stations, and matches what Polymarket's resolver reads.  Returns the
    same shape as fetch_openmeteo_today() so the rest of the pipeline
    doesn't change:

        {hourly: [(hour_local, temp_c), ...],
         forecast_high: float (C),
         forecast_peak_hour: int (local),
         sunset_hour: int (local)}

    NWS gives forecast temps in Fahrenheit by default; we convert to C
    since the rest of the model is metric.
    """
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    cache_key = (round(lat, 4), round(lon, 4))

    # Step 1 — points endpoint (gives us the gridpoint forecast URL)
    forecast_url = _NWS_POINTS_CACHE.get(cache_key)
    if not forecast_url:
        try:
            r = httpx.get(f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
                           headers=headers, timeout=30)
            r.raise_for_status()
            forecast_url = r.json()["properties"]["forecastHourly"]
            _NWS_POINTS_CACHE[cache_key] = forecast_url
        except Exception as e:
            log.warning(f"NWS points lookup failed at ({lat:.3f},{lon:.3f}): {e}")
            return {}

    # Step 2 — hourly forecast
    try:
        r = httpx.get(forecast_url, headers=headers, timeout=30)
        r.raise_for_status()
        periods = (r.json().get("properties") or {}).get("periods") or []
    except Exception as e:
        log.warning(f"NWS hourly forecast failed at ({lat:.3f},{lon:.3f}): {e}")
        return {}

    # Convert + filter to today (in city local time)
    tz = ZoneInfo(tz_str)
    today_local = datetime.now(tz).date()
    hourly: list[tuple[int, float]] = []
    for p in periods:
        ts_str = p.get("startTime")
        temp = p.get("temperature")
        unit = (p.get("temperatureUnit") or "F").upper()
        if not ts_str or temp is None:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        ts_local = ts.astimezone(tz)
        if ts_local.date() != today_local:
            continue
        temp_c = (float(temp) - 32) * 5/9 if unit == "F" else float(temp)
        hourly.append((ts_local.hour, temp_c))

    if not hourly:
        return {}

    forecast_peak_hour, forecast_high = max(hourly, key=lambda x: x[1])
    sunset_hour = _sunset_local_hour(today_local, lat, lon, tz_str)

    return {
        "hourly":             hourly,
        "forecast_high":      forecast_high,
        "forecast_peak_hour": forecast_peak_hour,
        "sunset_hour":        sunset_hour,
    }


# ---------------------------------------------------------------------------
# Open-Meteo — DEPRECATED forecast source.  Kept only as a fallback if NWS
# is unreachable.  The main pipeline now calls fetch_nws_today_forecast()
# above, which matches Polymarket's settlement source (NWS METAR feed
# routed through Wunderground).
# ---------------------------------------------------------------------------

def fetch_openmeteo_today(lat: float, lon: float, tz_str: str) -> dict:
    """Returns {hourly: [(hour_local, temp_c), ...], sunset_hour: int,
                  forecast_high: float, forecast_peak_hour: int}"""
    try:
        r = httpx.get(
            OPENMETEO_URL,
            params={
                "latitude":  lat,
                "longitude": lon,
                "hourly":    "temperature_2m",
                "daily":     "sunset",
                "timezone":  tz_str,
                "forecast_days": 1,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"Open-Meteo fetch failed at ({lat:.3f},{lon:.3f}): {e}")
        return {}

    h = data.get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    hourly: list[tuple[int, float]] = []
    for t, v in zip(times, temps):
        if v is None:
            continue
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
            hourly.append((dt.hour, float(v)))
        except ValueError:
            continue
    if not hourly:
        return {}

    forecast_peak_hour, forecast_high = max(hourly, key=lambda x: x[1])

    sunset_hour = 19   # safe default
    daily = data.get("daily", {}) or {}
    sset = (daily.get("sunset") or [None])[0]
    if sset:
        try:
            sunset_hour = datetime.strptime(sset, "%Y-%m-%dT%H:%M").hour
        except ValueError:
            pass

    return {
        "hourly":             hourly,
        "forecast_high":      forecast_high,
        "forecast_peak_hour": forecast_peak_hour,
        "sunset_hour":        sunset_hour,
    }


# ---------------------------------------------------------------------------
# Neighbor signal — adjust posterior based on upwind neighbor cooling
# ---------------------------------------------------------------------------

CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
def deg_to_cardinal(deg: float) -> str:
    return CARDINALS[int((deg + 22.5) / 45) % 8]


def vector_mean_dir(degrees: list[float]) -> float | None:
    if not degrees:
        return None
    u = sum(math.sin(math.radians(d)) for d in degrees) / len(degrees)
    v = sum(math.cos(math.radians(d)) for d in degrees) / len(degrees)
    if u == 0 and v == 0:
        return None
    return (math.degrees(math.atan2(u, v)) + 360) % 360


def compute_neighbor_signal(city: str, today_str: str,
                              afternoon_wind_octant: str | None) -> dict:
    """For US cities only — checks neighbor_obs.db for today's data from
    wind-aligned upwind neighbors.  Returns:
       { upwind_neighbors: [{sid, dir, peak_hr, peak_temp, current_temp,
                              cooling_from_peak_c}],
         strong_cooling_signal: bool,   # ≥1 upwind neighbor cooled ≥1°C
         signal_strength: float          # 0..1 score
       }"""
    if not os.path.exists(NEIGHBOR_DB) or not afternoon_wind_octant:
        return {"upwind_neighbors": [], "strong_cooling_signal": False,
                "signal_strength": 0.0}

    upwind: list[dict] = []
    with sqlite3.connect(NEIGHBOR_DB) as conn:
        conn.row_factory = sqlite3.Row
        # Neighbors in the direction the wind comes from = upwind
        nbrs = conn.execute(
            "SELECT sid, name, direction, distance_mi FROM neighbor_meta "
            "WHERE polymarket_city = ? AND direction = ?",
            (city, afternoon_wind_octant),
        ).fetchall()
        for n in nbrs:
            rows = conn.execute(
                "SELECT hour_local, temp_c FROM neighbor_obs "
                "WHERE sid = ? AND date_local = ? AND temp_c IS NOT NULL "
                "ORDER BY hour_local",
                (n["sid"], today_str),
            ).fetchall()
            if len(rows) < 4:
                continue
            peak_row = max(rows, key=lambda r: r["temp_c"])
            current = rows[-1]
            cooling = peak_row["temp_c"] - current["temp_c"]
            upwind.append({
                "sid":         n["sid"],
                "name":        n["name"],
                "direction":   n["direction"],
                "distance_mi": n["distance_mi"],
                "peak_hour":   peak_row["hour_local"],
                "peak_temp":   round(peak_row["temp_c"], 1),
                "current_hour": current["hour_local"],
                "current_temp": round(current["temp_c"], 1),
                "cooling_c":   round(cooling, 1),
                "cooling_since_peak_h": current["hour_local"] - peak_row["hour_local"],
            })

    strong = any(u["cooling_c"] >= 1.0 and u["cooling_since_peak_h"] >= 1
                  for u in upwind)
    # Signal strength: average cooling across upwind neighbors, capped at 1.0
    strength = 0.0
    if upwind:
        avg_cool = sum(max(0, u["cooling_c"]) for u in upwind) / len(upwind)
        strength = min(1.0, avg_cool / 3.0)   # 3°C cool = full signal
    return {
        "upwind_neighbors":      upwind,
        "strong_cooling_signal": strong,
        "signal_strength":       round(strength, 3),
    }


# ---------------------------------------------------------------------------
# Predictor — combines all sources into bin probabilities
# ---------------------------------------------------------------------------

def estimate_day_high_dist(forecast_high: float, forecast_peak_hour: int,
                            observed_max: float, observed_peak_hour: int,
                            current_hour: int, sunset_hour: int,
                            neighbor_signal: dict,
                            base_sigma_c: float | None = None) -> tuple[float, float]:
    """Returns (mu, sigma) of the day-high distribution after all adjustments.

    base_sigma_c: per-city σ from calibration (RMSE of Open-Meteo vs observed
    over the last ~60 days).  Falls back to DEFAULT_FORECAST_SIGMA_C if None.
    """
    mu = forecast_high
    sigma = base_sigma_c if base_sigma_c is not None else DEFAULT_FORECAST_SIGMA_C
    sigma_ceiling = sigma   # for later clamps that reference the prior

    # FIX 1a: Observed already exceeded forecast — pull mu up to at least
    # observed (+0.3°C buffer for further climb).
    if observed_max > -50 and observed_max > forecast_high:
        excess = observed_max - forecast_high
        mu = observed_max + 0.3
        sigma = min(sigma_ceiling, max(1.0, sigma - 0.3 + excess * 0.2))

    # Narrow uncertainty as we pass the forecast peak hour
    if current_hour >= forecast_peak_hour:
        hours_past = current_hour - forecast_peak_hour
        narrow = max(0.30, 1.0 - hours_past * 0.12)
        sigma *= narrow

        # FIX 1b: Blend mu toward observed_max in BOTH directions
        if hours_past >= 1 and observed_max > -50:
            blend = min(0.8, hours_past * 0.15)
            mu = (1 - blend) * mu + blend * observed_max

    # NEW: Time-since-OBSERVED-peak narrowing.  Once we've actually seen
    # the peak and a few hours have passed without it being exceeded, the
    # day-high is essentially locked at observed_max.  Empirically (from
    # the temp-drop backtest): hold rate is 84% at 1h past peak, 97% at
    # 2h, 99%+ at 3h+.  We narrow sigma aggressively to reflect this.
    if observed_max > -50 and observed_peak_hour is not None and observed_peak_hour >= 0:
        hours_since_obs_peak = current_hour - observed_peak_hour
        if hours_since_obs_peak >= 1:
            # Geometric narrowing: 0.7^h
            #   1h → 0.70x, 2h → 0.49x, 3h → 0.34x, 4h+ → 0.24x (floor)
            narrow = max(0.20, 0.7 ** hours_since_obs_peak)
            sigma *= narrow
            # And pull mu toward observed (since the longer it's been, the
            # more likely observed IS the day high)
            blend = min(0.9, hours_since_obs_peak * 0.25)
            mu = (1 - blend) * mu + blend * observed_max

    # After sunset, the day high is essentially locked at observed_max
    if current_hour >= sunset_hour:
        mu = observed_max if observed_max > -50 else mu
        sigma = max(0.3, sigma * 0.2)

    # Upwind cooling neighbor signal — narrows sigma and pulls mu toward observed
    if neighbor_signal.get("strong_cooling_signal"):
        strength = neighbor_signal.get("signal_strength", 0.0)
        narrow = 1.0 - 0.3 * strength
        sigma *= narrow
        if observed_max > -50:
            blend = 0.2 * strength
            mu = (1 - blend) * mu + blend * observed_max

    sigma = max(0.3, sigma)
    return mu, sigma


def bin_temp_range(bin_dict: dict) -> tuple[float | None, float | None]:
    """Translate a Polymarket bin (range_low, range_high) into the actual
    real-line temperature range (in °C) it represents on settlement.

    Polymarket uses integer-labeled bins with half-up rounding.  E.g.:
      single bin '28°C' = actual in [27.5, 28.5) — width 1°C
      2°F bin '88-89°F' = actual in [87.5, 89.5)°F — width 2°F
      '23°C or below'   = actual ≤ 23.5°C — open downward
      '33°C or higher'  = actual ≥ 32.5°C — open upward

    Output is in CELSIUS regardless of input unit, since our forecast and
    observation streams are all Celsius.
    """
    lo = bin_dict.get("range_low")
    hi = bin_dict.get("range_high")
    unit = (bin_dict.get("unit") or "celsius").lower()
    is_f = unit == "fahrenheit"

    def to_c(v):
        return (v - 32) * 5 / 9 if is_f else v

    # Half-step in the bin's NATIVE unit, then converted.  For F bins this
    # is 0.5°F which is ~0.28°C — applied to the labelled boundary BEFORE
    # converting to Celsius keeps the half-step semantics correct.
    if lo is None and hi is not None:
        # "X°? or below" → actual <= X + 0.5 (native units)
        return (None, to_c(hi + 0.5))
    if lo is not None and hi is None:
        # "X°? or higher" → actual >= X - 0.5 (native units)
        return (to_c(lo - 0.5), None)
    if lo is not None and hi is not None:
        # FIX 2: Whether it's a single-integer bin ('28°C', lo==hi) or a
        # multi-step bin ('88-89°F', lo!=hi), the actual-range endpoints
        # are lo-0.5 and hi+0.5 in the bin's native unit.  Previously a
        # multi-step bin was treated as (lo, hi) without the half-step
        # padding, which under-counted probability at the boundaries.
        return (to_c(lo - 0.5), to_c(hi + 0.5))
    return (None, None)


def predict_bins(event: dict, settlement_obs: list[dict],
                  forecast: dict, neighbor_signal: dict,
                  current_hour: int, city: str | None = None) -> dict:
    """Per-event bin probability + edge + recommendation."""
    bins = event.get("outcomes") or []
    if not bins or not forecast:
        return {"bins": [], "skipped": "no_data"}

    # Observed max in Celsius
    if settlement_obs:
        observed_max_c = max(r["temp_c"] for r in settlement_obs)
        observed_peak_hour = max(r["hour_local"] for r in settlement_obs
                                   if r["temp_c"] == observed_max_c)
    else:
        # No live data — fall back to "the day hasn't started" assumption.
        # Use forecast morning low.
        observed_max_c = -100   # nothing observed yet → no truncation
        observed_peak_hour = -1

    base_sigma = get_city_sigma(city) if city else DEFAULT_FORECAST_SIGMA_C
    mu, sigma = estimate_day_high_dist(
        forecast_high       = forecast["forecast_high"],
        forecast_peak_hour  = forecast["forecast_peak_hour"],
        observed_max        = observed_max_c if observed_max_c > -100 else forecast["forecast_high"],
        observed_peak_hour  = observed_peak_hour if observed_peak_hour >= 0 else None,
        current_hour        = current_hour,
        sunset_hour         = forecast["sunset_hour"],
        neighbor_signal     = neighbor_signal,
        base_sigma_c        = base_sigma,
    )

    truncate_at = observed_max_c if observed_max_c > -100 else -100.0

    bin_results = []
    for b in bins:
        c_lo, c_hi = bin_temp_range(b)
        p = truncated_normal_prob(c_lo, c_hi, mu, sigma, truncate_at)
        market_p = float(b.get("yes_price") or 0)
        edge = p - market_p
        bin_results.append({
            "contract_id":   b.get("contract_id"),
            "yes_token_id":  b.get("yes_token_id"),
            "range_low":     b.get("range_low"),
            "range_high":    b.get("range_high"),
            "unit":          b.get("unit"),
            "c_lo":          round(c_lo, 2) if c_lo is not None else None,
            "c_hi":          round(c_hi, 2) if c_hi is not None else None,
            "label":         _bin_label(b.get("range_low"), b.get("range_high"), b.get("unit")),
            "our_prob":      round(p, 4),
            "market_prob":   round(market_p, 4),
            "edge":          round(edge, 4),
            "liquidity_usd": round(float(b.get("liquidity_usd") or 0), 0),
        })
    return {
        "bins":               bin_results,
        "mu":                 round(mu, 2),
        "sigma":              round(sigma, 2),
        "observed_max_c":     round(observed_max_c, 2) if observed_max_c > -100 else None,
        "observed_peak_hour": observed_peak_hour if observed_peak_hour >= 0 else None,
        "forecast_high":      round(forecast["forecast_high"], 2),
        "forecast_peak_hour": forecast["forecast_peak_hour"],
        "sunset_hour":        forecast["sunset_hour"],
    }


def _bin_label(lo, hi, unit) -> str:
    suffix = "F" if (unit or "celsius").lower() == "fahrenheit" else "C"
    if lo is None and hi is not None: return f"≤{int(hi)}°{suffix}"
    if lo is not None and hi is None: return f"≥{int(lo)}°{suffix}"
    if lo is not None and hi is not None:
        if int(lo) == int(hi): return f"{int(lo)}°{suffix}"
        return f"{int(lo)}–{int(hi)}°{suffix}"
    return "?"


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #0f172a; color: #e2e8f0; font-size: 13px; }
a { color: #818cf8; }
header { background: linear-gradient(90deg, #1e293b, #0f172a);
         padding: 16px 24px; border-bottom: 1px solid #334155;
         display: flex; align-items: center; justify-content: space-between; }
header h1 { margin: 0; font-size: 18px; font-weight: 600; color: white; }
header .meta { font-family: monospace; font-size: 11px; color: #94a3b8; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 8px; padding: 12px 24px; background: #0f172a;
        border-bottom: 1px solid #1e293b; }
.kpi { background: #1e293b; padding: 10px 14px; border-radius: 6px;
       border-left: 3px solid #4338ca; }
.kpi .label { font-size: 9px; color: #94a3b8; text-transform: uppercase;
              letter-spacing: 0.5px; font-weight: 600; }
.kpi .val { font-size: 22px; font-weight: 700; margin-top: 2px; font-family: monospace;
            color: white; }
.kpi.pos { border-left-color: #22c55e; }
.kpi.pos .val { color: #4ade80; }
.kpi.neg { border-left-color: #ef4444; }
.kpi.neg .val { color: #f87171; }
.city-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
             gap: 10px; padding: 12px 24px; }
.city-card { background: #1e293b; border-radius: 8px; padding: 12px 14px;
             border-top: 3px solid #4338ca; cursor: pointer;
             transition: transform 0.15s; }
.city-card:hover { transform: translateY(-2px); background: #273449; }
.city-card.has-buy { border-top-color: #22c55e; }
.city-card.no-data { opacity: 0.6; border-top-color: #475569; }
.city-card .city-name { font-weight: 700; font-size: 14px; color: white;
                         display: flex; justify-content: space-between; align-items: center; }
.city-card .station { font-size: 10px; color: #94a3b8; font-family: monospace; }
.city-card .row { display: flex; justify-content: space-between; margin: 4px 0;
                  font-size: 12px; }
.city-card .row .lbl { color: #94a3b8; }
.city-card .row .v { font-family: monospace; color: #e2e8f0; }
.city-card .row .v.hot { color: #fbbf24; }
.city-card .buys { display: inline-block; padding: 2px 8px; border-radius: 10px;
                    background: #22c55e; color: white; font-size: 10px;
                    font-weight: 700; }
.filters { background: #1e293b; padding: 10px 24px; display: flex;
           gap: 18px; flex-wrap: wrap; align-items: center; font-size: 12px;
           border-bottom: 1px solid #334155; position: sticky; top: 0; z-index: 10; }
.filters label { font-weight: 600; color: #cbd5e1; margin-right: 4px; }
.filters select, .filters input { padding: 4px 8px; font-size: 12px;
           background: #0f172a; color: white; border: 1px solid #475569;
           border-radius: 4px; }
.filters .count { color: #94a3b8; font-family: monospace; margin-left: auto; }
table { width: calc(100% - 48px); margin: 12px 24px; background: #1e293b;
        border-collapse: collapse; border-radius: 6px; overflow: hidden;
        font-size: 12px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 10px;
     text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer;
     user-select: none; }
th:hover { background: #1e293b; color: white; }
th.sorted-asc::after { content: ' ▲'; color: #818cf8; }
th.sorted-desc::after { content: ' ▼'; color: #818cf8; }
tr.buy { background: rgba(34, 197, 94, 0.12); }
tr.buy:hover { background: rgba(34, 197, 94, 0.20); }
tr.skip { color: #64748b; }
tr.avoid { background: rgba(239, 68, 68, 0.08); color: #cbd5e1; }
td.num { font-family: monospace; text-align: right; }
td.bin-label { font-weight: 600; color: white; }
td .pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 10px; font-weight: 700; }
.pill.buy { background: #22c55e; color: white; }
.pill.skip { background: #475569; color: #cbd5e1; }
.pill.avoid { background: #ef4444; color: white; }
.bar { display: inline-block; width: 100px; height: 8px; background: #334155;
       border-radius: 4px; overflow: hidden; vertical-align: middle; }
.bar .fill { height: 100%; background: #818cf8; }
.bar .fill.market { background: #fbbf24; }
.bar .fill.edge-pos { background: #22c55e; }
.bar .fill.edge-neg { background: #ef4444; }
.edge-cell { font-family: monospace; font-weight: 700; }
.edge-cell.pos { color: #4ade80; }
.edge-cell.neg { color: #f87171; }
.section-title { padding: 18px 24px 8px; font-size: 11px; font-weight: 700;
                  color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.empty { text-align: center; color: #64748b; padding: 40px 0; font-size: 14px; }
"""


def render_dashboard(predictions: list[dict], cities_summary: list[dict],
                      kpis: dict, generated_at: str, out_path: str) -> None:
    bins_data = []
    for p in predictions:
        for b in p.get("bins", []):
            edge = b["edge"]
            rec = "BUY" if (edge >= 0.10 and b["market_prob"] < 0.95 and b["liquidity_usd"] >= 300) \
                  else ("AVOID" if edge <= -0.20 else "SKIP")
            bins_data.append({
                "city":         p["city"],
                "date":         p["date"],
                "station":      p["station"],
                "bin":          b["label"],
                "our_prob":     b["our_prob"],
                "market_prob":  b["market_prob"],
                "edge":         edge,
                "liquidity":    b["liquidity_usd"],
                "recommend":    rec,
                "contract_id":  b["contract_id"],
            })

    bins_json   = json.dumps(bins_data, default=str, separators=(",", ":"))
    cities_json = json.dumps(cities_summary, default=str, separators=(",", ":"))

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Polymarket Weather — Intraday Bin Predictor</title>
<style>{DASHBOARD_CSS}</style></head><body>

<header>
  <div>
    <h1>Polymarket Weather — Intraday Bin Predictor</h1>
    <div class='meta' style='color:#94a3b8;font-size:11px;margin-top:2px'>
      Settlement obs: NWS API · Forecast prior: Open-Meteo · Neighbor signal: cached neighbor_obs.db
    </div>
  </div>
  <div class='meta'>generated {generated_at}</div>
</header>

<div class='kpis'>
  <div class='kpi'><div class='label'>US cities</div><div class='val'>{kpis['n_cities']}</div></div>
  <div class='kpi'><div class='label'>Active events</div><div class='val'>{kpis['n_events']}</div></div>
  <div class='kpi'><div class='label'>Total bins</div><div class='val'>{kpis['n_bins']}</div></div>
  <div class='kpi pos'><div class='label'>BUY signals</div><div class='val'>{kpis['n_buys']}</div></div>
  <div class='kpi'><div class='label'>Avg buy edge</div>
    <div class='val'>{kpis['avg_buy_edge']:+.0%}</div></div>
  <div class='kpi pos'><div class='label'>Total potential edge $/$1</div>
    <div class='val'>${kpis['total_buy_edge']:+.2f}</div></div>
</div>

<div class='section-title'>City snapshots</div>
<div class='city-grid' id='city-grid'></div>

<div class='section-title'>All bins · filter and sort</div>
<div class='filters'>
  <div><label>City</label>
    <select id='f-city'><option value=''>All</option></select></div>
  <div><label>Date</label>
    <select id='f-date'><option value=''>All</option></select></div>
  <div><label>Recommend</label>
    <select id='f-rec'>
      <option value=''>All</option>
      <option value='BUY'>BUY only</option>
      <option value='SKIP'>SKIP only</option>
      <option value='AVOID'>AVOID only</option>
    </select></div>
  <div><label>Min edge</label>
    <input id='f-edge' type='number' step='0.05' min='-1' max='1' value='-1' style='width:70px'></div>
  <div><label>Min liquidity $</label>
    <input id='f-liq' type='number' step='100' min='0' value='0' style='width:80px'></div>
  <div class='count' id='count'>—</div>
</div>

<table id='bins'>
  <thead><tr>
    <th data-key='date'>Date</th>
    <th data-key='city'>City</th>
    <th data-key='bin'>Bin</th>
    <th data-key='our_prob'>Our P</th>
    <th data-key='market_prob'>Market P</th>
    <th data-key='edge'>Edge</th>
    <th data-key='liquidity'>Liquidity</th>
    <th data-key='recommend'>Recommend</th>
  </tr></thead>
  <tbody id='tbody'></tbody>
</table>

<script>
const BINS = {bins_json};
const CITIES = {cities_json};
const $ = id => document.getElementById(id);

// City grid render
function renderCities() {{
  const html = CITIES.map(c => {{
    const cls = c.n_buys > 0 ? 'has-buy' : (c.has_data ? '' : 'no-data');
    const hot = c.observed_max_c !== null && c.forecast_high !== null
                && c.observed_max_c >= c.forecast_high - 0.5;
    return `<div class='city-card ${{cls}}'>
      <div class='city-name'>
        <span>${{c.city}}</span>
        ${{c.n_buys > 0 ? `<span class='buys'>${{c.n_buys}} BUY</span>` : ''}}
      </div>
      <div class='station'>${{c.station}} · ${{c.tz_label || ''}}</div>
      <div class='row'><span class='lbl'>Now</span>
        <span class='v'>${{c.current_temp_c !== null ? c.current_temp_c.toFixed(1)+'°C' : '—'}}
          ${{c.current_wind_dir !== null ? '· wind ' + c.current_wind_dir : ''}}</span></div>
      <div class='row'><span class='lbl'>Observed high</span>
        <span class='v ${{hot ? 'hot' : ''}}'>${{c.observed_max_c !== null ? c.observed_max_c.toFixed(1)+'°C @ '+c.observed_peak_hour+':00' : '—'}}</span></div>
      <div class='row'><span class='lbl'>Forecast high</span>
        <span class='v'>${{c.forecast_high !== null ? c.forecast_high.toFixed(1)+'°C @ '+c.forecast_peak_hour+':00' : '—'}}</span></div>
      <div class='row'><span class='lbl'>Sunset</span>
        <span class='v'>${{c.sunset_hour !== null ? c.sunset_hour+':00' : '—'}}</span></div>
      ${{c.upwind_summary ? `<div class='row'><span class='lbl'>Upwind</span>
        <span class='v'>${{c.upwind_summary}}</span></div>` : ''}}
    </div>`;
  }}).join('');
  $('city-grid').innerHTML = html || `<div class='empty'>No city data</div>`;
}}

// Filter dropdowns
const cities = [...new Set(BINS.map(b => b.city))].sort();
for (const c of cities) {{
  const o = document.createElement('option'); o.value = o.textContent = c;
  $('f-city').appendChild(o);
}}
const dates = [...new Set(BINS.map(b => b.date))].sort();
for (const d of dates) {{
  const o = document.createElement('option'); o.value = o.textContent = d;
  $('f-date').appendChild(o);
}}

let SORT_KEY = 'edge', SORT_DIR = -1;

function row(b) {{
  const rec = b.recommend;
  const cls = rec.toLowerCase();
  const edgeCls = b.edge >= 0 ? 'pos' : 'neg';
  const edgeSign = b.edge >= 0 ? '+' : '';
  const ourBar = Math.round(Math.max(0, Math.min(1, b.our_prob)) * 100);
  const mktBar = Math.round(Math.max(0, Math.min(1, b.market_prob)) * 100);
  return `<tr class='${{cls}}'>
    <td>${{b.date}}</td>
    <td><b>${{b.city}}</b><br><span style='color:#64748b;font-size:10px'>${{b.station}}</span></td>
    <td class='bin-label'>${{b.bin}}</td>
    <td class='num'>${{(b.our_prob*100).toFixed(1)}}%
      <div class='bar'><div class='fill' style='width:${{ourBar}}%'></div></div>
    </td>
    <td class='num'>${{(b.market_prob*100).toFixed(1)}}%
      <div class='bar'><div class='fill market' style='width:${{mktBar}}%'></div></div>
    </td>
    <td class='num edge-cell ${{edgeCls}}'>${{edgeSign}}${{(b.edge*100).toFixed(1)}}%</td>
    <td class='num'>$${{Math.round(b.liquidity).toLocaleString()}}</td>
    <td><span class='pill ${{cls}}'>${{rec}}</span></td>
  </tr>`;
}}

function render() {{
  const c = $('f-city').value, d = $('f-date').value, r = $('f-rec').value;
  const me = parseFloat($('f-edge').value);
  const ml = parseFloat($('f-liq').value);
  let rows = BINS.filter(b =>
    (!c || b.city === c)
    && (!d || b.date === d)
    && (!r || b.recommend === r)
    && (isNaN(me) || b.edge >= me)
    && (isNaN(ml) || b.liquidity >= ml)
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (typeof av === 'number') return SORT_DIR * (av - bv);
    return SORT_DIR * String(av).localeCompare(String(bv));
  }});
  $('count').textContent = rows.length + ' / ' + BINS.length;
  $('tbody').innerHTML = rows.length ? rows.map(row).join('')
    : `<tr><td colspan='8' class='empty'>No bins match filters</td></tr>`;
  document.querySelectorAll('th').forEach(th => {{
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? 'sorted-asc' : 'sorted-desc');
  }});
}}

document.querySelectorAll('th').forEach(th => {{
  th.addEventListener('click', () => {{
    if (!th.dataset.key) return;
    if (SORT_KEY === th.dataset.key) SORT_DIR = -SORT_DIR;
    else {{ SORT_KEY = th.dataset.key; SORT_DIR = 1; }}
    render();
  }});
}});
['f-city','f-date','f-rec','f-edge','f-liq'].forEach(id =>
  $(id).addEventListener('input', render));

renderCities();
render();
</script>
</body></html>"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info(f"Wrote {os.path.getsize(out_path)/1024:.0f} KB dashboard to {out_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all US)")
    p.add_argument("--min-edge", type=float, default=0.10,
                   help="Minimum edge for BUY recommendation (default: 0.10)")
    p.add_argument("--html", default=DEFAULT_OUT,
                   help=f"Output HTML path (default: {DEFAULT_OUT})")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON to stdout instead of HTML")
    args = p.parse_args()

    # Load per-city σ calibration (silently uses defaults if file missing)
    _load_calibration()

    us_cities = list(US_CITY_STATES.keys()) if US_CITY_STATES else [
        c for c, m in CITY_STATIONS.items() if m[0].startswith("K")
    ]
    cities = args.city or us_cities
    cities = [c for c in cities if c in CITY_STATIONS]
    log.info(f"Predicting for {len(cities)} US cities")

    log.info("Discovering active Polymarket events ...")
    all_events = search_temp_high_events(min_liquidity=100)
    events_by_city: dict[str, list[dict]] = defaultdict(list)
    for e in all_events:
        if e.get("city") in cities:
            events_by_city[e["city"]].append(e)

    predictions: list[dict] = []
    cities_summary: list[dict] = []
    for city in cities:
        s = CITY_STATIONS[city]
        icao, _net, tz_str, lat, lon = s
        tz = ZoneInfo(tz_str)
        now_local = datetime.now(tz)
        today_str = now_local.date().isoformat()

        log.info(f"  {city:<14} {icao}  fetching NWS obs + forecast ...")

        nws_obs   = fetch_nws_today_obs(icao, tz_str)
        # NWS forecast first (matches Polymarket settlement source).  Fall
        # back to Open-Meteo only if NWS is unreachable.
        forecast  = fetch_nws_today_forecast(lat, lon, tz_str)
        if not forecast:
            log.warning(f"  {city:<14} NWS forecast empty — falling back to Open-Meteo")
            forecast = fetch_openmeteo_today(lat, lon, tz_str)

        # Afternoon mean wind direction for neighbor-signal lookup
        afternoon_winds = [r["wind_dir_deg"] for r in nws_obs
                            if r["hour_local"] in range(11, 19)
                            and r["wind_dir_deg"] is not None]
        wind_mean = vector_mean_dir(afternoon_winds)
        wind_octant = deg_to_cardinal(wind_mean) if wind_mean is not None else None

        neighbor_signal = compute_neighbor_signal(city, today_str, wind_octant)

        # Summary for the city card
        observed_max = max((r["temp_c"] for r in nws_obs), default=None)
        observed_peak_hour = None
        if observed_max is not None:
            observed_peak_hour = max(r["hour_local"] for r in nws_obs
                                      if r["temp_c"] == observed_max)
        current_temp = nws_obs[-1]["temp_c"] if nws_obs else None
        current_wd = nws_obs[-1]["wind_dir_deg"] if nws_obs else None
        upwind_summary = ""
        if neighbor_signal["upwind_neighbors"]:
            top = neighbor_signal["upwind_neighbors"][:2]
            upwind_summary = ", ".join(
                f"{u['sid']} @{u['peak_hour']}:00 ({u['peak_temp']:.0f}°→{u['current_temp']:.0f}°)"
                for u in top
            )

        # Per-event predictions
        n_buys_this_city = 0
        for ev in sorted(events_by_city.get(city, []), key=lambda e: e.get("date") or ""):
            pred = predict_bins(ev, nws_obs, forecast, neighbor_signal,
                                 now_local.hour, city=city)
            if not pred.get("bins"):
                continue
            for b in pred["bins"]:
                if (b["edge"] >= args.min_edge
                    and b["market_prob"] < 0.95
                    and b["liquidity_usd"] >= 300):
                    n_buys_this_city += 1
            predictions.append({
                "city":              city,
                "station":           icao,
                "date":              ev.get("date"),
                "event_title":       ev.get("event_title"),
                "bins":              pred["bins"],
                "mu":                pred["mu"],
                "sigma":             pred["sigma"],
                "observed_max_c":    pred["observed_max_c"],
                "observed_peak_hour": pred["observed_peak_hour"],
                "forecast_high":     pred["forecast_high"],
                "forecast_peak_hour": pred["forecast_peak_hour"],
                "sunset_hour":       pred["sunset_hour"],
            })

        cities_summary.append({
            "city":               city,
            "station":            icao,
            "tz_label":           tz_str.split("/")[-1].replace("_", " "),
            "current_temp_c":     round(current_temp, 1) if current_temp is not None else None,
            "current_wind_dir":   deg_to_cardinal(current_wd) if current_wd is not None else None,
            "observed_max_c":     round(observed_max, 1) if observed_max is not None else None,
            "observed_peak_hour": observed_peak_hour,
            "forecast_high":      round(forecast.get("forecast_high"), 1) if forecast else None,
            "forecast_peak_hour": forecast.get("forecast_peak_hour") if forecast else None,
            "sunset_hour":        forecast.get("sunset_hour") if forecast else None,
            "wind_octant_after":  wind_octant,
            "upwind_summary":     upwind_summary,
            "has_data":           bool(nws_obs) and bool(forecast),
            "n_buys":             n_buys_this_city,
        })

    # KPI rollup
    all_bins = [b for p in predictions for b in p["bins"]]
    buys = [b for b in all_bins
             if b["edge"] >= args.min_edge
             and b["market_prob"] < 0.95
             and b["liquidity_usd"] >= 300]
    kpis = {
        "n_cities":       len(cities),
        "n_events":       len(predictions),
        "n_bins":         len(all_bins),
        "n_buys":         len(buys),
        "avg_buy_edge":   (sum(b["edge"] for b in buys) / len(buys)) if buys else 0.0,
        "total_buy_edge": sum(b["edge"] for b in buys),
    }

    if args.json:
        print(json.dumps({"kpis": kpis, "cities": cities_summary,
                            "predictions": predictions}, default=str, indent=2))
        return 0

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    render_dashboard(predictions, cities_summary, kpis, generated_at, args.html)

    print()
    print("=" * 78)
    print(f"  KPIs: {kpis['n_cities']} cities · {kpis['n_events']} events · "
          f"{kpis['n_bins']} bins · {kpis['n_buys']} BUY signals")
    if buys:
        print(f"  Avg buy edge: {kpis['avg_buy_edge']:+.1%} · "
              f"Total edge $/$1: ${kpis['total_buy_edge']:+.2f}")
    print(f"  Dashboard: {args.html}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())