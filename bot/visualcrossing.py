"""
visualcrossing.py — Visual Crossing Timeline Weather API client.

Used for two integrations:

  1. Historical bias correction
     `fetch_daily_history(lat, lon, start, end)` pulls multi-year daily max
     temperatures (source="obs" only — real station observations) so they can
     be paired against historical model forecasts from Open-Meteo Previous
     Runs to compute per-(city, month, model) biases.

  2. Intraday observed-vs-forecast tracking
     `fetch_intraday(lat, lon)` returns a single combined snapshot of today:
         - Current observed temperature at the nearest station
         - Observed hourly temperatures for the elapsed hours of today
         - Forecast hourly temperatures for the remaining hours of today
         - VC's best day-high estimate (blended obs+fcst)
     The per-hour `source` field ("obs" or "fcst") lets us compute
     observed_max_so_far and forecast_remaining_max client-side.

Design notes
------------
* Single `VISUAL_CROSSING_API_KEY` in .env is the only auth.
* Rate limits are generous on the paid tier (no daily budget cap); we still
  retry transient 429/500 with exponential backoff via tenacity.
* queryCost is logged for every call — expected cost per call:
      fetch_daily_history  → 1 record per day returned
      fetch_intraday       → 1 record per today-with-hours query
* All temperatures are returned in Celsius (unitGroup=metric).
"""

import calendar
import json
import os
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Disk cache for raw VC monthly responses.
#
# Once a (lat, lon, year, month) range is fetched once, we persist the full
# JSON response to bot/data/vc_cache/{lat}_{lon}/{YYYY}-{MM}.json.  Future
# fetches with the same range read from disk — costing zero queryCost.
#
# Only month-aligned (lat, lon, start, end) calls are cached; ad-hoc ranges
# fall through to the network as before.  Cache is opt-in via the new
# `use_cache=True` kwarg on fetch_history_with_hours().
# ---------------------------------------------------------------------------
_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
_VC_CACHE_DIR = Path(_BOT_DIR) / "data" / "vc_cache"


def _is_month_aligned(start_date: date, end_date: date) -> bool:
    """True iff (start, end) covers exactly one full calendar month."""
    if start_date.day != 1:
        return False
    last_day = calendar.monthrange(start_date.year, start_date.month)[1]
    return end_date == date(start_date.year, start_date.month, last_day)


def _cache_path_for_month(lat: float, lon: float, start_date: date) -> Path:
    """Cache file for a (lat, lon, year, month).  Caller must verify
    month-aligned first via _is_month_aligned()."""
    loc_dir = _VC_CACHE_DIR / f"{round(lat, 3)}_{round(lon, 3)}"
    return loc_dir / f"{start_date.year}-{start_date.month:02d}.json"


def _read_vc_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"VC cache read failed for {path}: {e}; will re-fetch")
        return None


def _write_vc_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as e:
        logger.warning(f"VC cache write failed for {path}: {e}")


def vc_cache_stats() -> dict:
    """Quick report on the on-disk cache: per-location file count and total size."""
    if not _VC_CACHE_DIR.exists():
        return {"locations": 0, "files": 0, "size_mb": 0.0}
    n_files = 0
    n_bytes = 0
    n_locs = 0
    for loc_dir in _VC_CACHE_DIR.iterdir():
        if not loc_dir.is_dir():
            continue
        n_locs += 1
        for f in loc_dir.iterdir():
            if f.suffix == ".json":
                n_files += 1
                n_bytes += f.stat().st_size
    return {"locations": n_locs, "files": n_files, "size_mb": round(n_bytes / 1e6, 2)}

logger = logging.getLogger(__name__)

VC_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "VISUAL_CROSSING_API_KEY is not set in the environment. "
            "Add it to .env before calling Visual Crossing functions."
        )
    return key


# ---------------------------------------------------------------------------
# Shared HTTP helper (retry on 429/5xx, surface 4xx immediately)
# ---------------------------------------------------------------------------

class _RetryableVCError(Exception):
    """Transient error worth retrying (rate limit or server side)."""


@retry(
    retry=retry_if_exception_type(_RetryableVCError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    reraise=True,
)
def _get_vc(path_suffix: str, params: dict[str, Any], timeout: float = 60.0) -> dict:
    """
    GET a Timeline endpoint with retry-on-transient and queryCost logging.
    `path_suffix` is everything after VC_BASE (e.g., "/{lat},{lon}/today").
    """
    params = {**params, "key": _api_key(), "contentType": "json"}
    try:
        r = httpx.get(f"{VC_BASE}{path_suffix}", params=params, timeout=timeout)
    except httpx.HTTPError as e:
        raise _RetryableVCError(f"Network error: {e}") from e

    if r.status_code in (429, 500, 502, 503, 504):
        raise _RetryableVCError(f"Transient {r.status_code}: {r.text[:200]}")

    if r.status_code >= 400:
        # 400/401/403/404 are config/auth errors — fail fast, do not retry
        logger.error(
            f"Visual Crossing returned {r.status_code} for {path_suffix} "
            f"— body: {r.text[:300]}"
        )
        r.raise_for_status()

    j = r.json()
    cost = j.get("queryCost")
    if cost is not None:
        logger.debug(f"VC call {path_suffix} succeeded (queryCost={cost})")
    return j


# ---------------------------------------------------------------------------
# Integration 1 — Historical daily observations
# ---------------------------------------------------------------------------

def fetch_daily_history(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    observed_only: bool = True,
) -> list[dict]:
    """
    Pull daily max/min/mean temperatures for a lat/lon over a date range.

    Parameters
    ----------
    lat, lon
        Coordinates.  VC resolves these to the nearest contributing weather
        stations automatically.
    start_date, end_date
        Inclusive date range.  Typical call spans years.
    observed_only
        When True (default), filters to rows where VC marks the data as
        station observations (`source == "obs"`).  Set False only if you
        want to include blended/statistical historical rows.

    Returns
    -------
    list of dicts, one per day, each with keys:
        date (str, YYYY-MM-DD)
        tempmax_c (float | None)
        tempmin_c (float | None)
        temp_c    (float | None, daily mean)
        source    (str)                    — "obs", "comb", "stats", etc.
        stations  (list[str])              — station IDs that contributed
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    path = f"/{lat},{lon}/{start_date.isoformat()}/{end_date.isoformat()}"
    j = _get_vc(path, {
        "unitGroup": "metric",
        "include":   "days",
        "elements":  "datetime,tempmax,tempmin,temp,source,stations",
    })

    out: list[dict] = []
    for d in (j.get("days") or []):
        src = d.get("source")
        if observed_only and src != "obs":
            continue
        out.append({
            "date":       d.get("datetime"),
            "tempmax_c":  d.get("tempmax"),
            "tempmin_c":  d.get("tempmin"),
            "temp_c":     d.get("temp"),
            "source":     src,
            "stations":   list(d.get("stations") or []),
        })
    logger.info(
        f"VC daily history ({lat:.3f},{lon:.3f}) {start_date}..{end_date}: "
        f"{len(out)} observed-day rows (total days returned: {len(j.get('days') or [])})"
    )
    return out


# ---------------------------------------------------------------------------
# Integration 1b — Historical daily + hourly, rich feature set (ML backfill)
# ---------------------------------------------------------------------------

def fetch_history_with_hours(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    use_cache: bool = True,
) -> dict:
    """
    Pull VC historical data over [start_date, end_date] with both day-level
    summaries AND hour-level observations.  Used by the ML training data
    backfill — returns the full rich feature set (temp, humidity, dew,
    pressure, cloud, wind, solar, snow, precip, etc.) at hourly granularity.

    When use_cache=True (default) and the call is month-aligned, raw VC
    responses are persisted to bot/data/vc_cache/{lat}_{lon}/{YYYY}-{MM}.json
    on first fetch and read from disk on subsequent calls — costing zero
    queryCost.  Pass use_cache=False to force a network fetch (e.g. to
    refresh stale data).  Non-month-aligned calls always go to the network.

    Returns a dict:
        {
          "resolved_address": str,
          "timezone":         str,      -- IANA tz of the location
          "latitude":         float,
          "longitude":        float,
          "query_cost":       int,
          "days": [
              {
                "date":       "YYYY-MM-DD",   -- local date
                "tempmax_c":  float | None,
                "tempmin_c":  float | None,
                "temp_c":     float | None,
                "source":     str,
                "hours": [
                   { datetime, datetime_epoch, temp_c, feelslike_c,
                     humidity, dew_c, pressure_hpa, cloudcover,
                     visibility_km, windspeed_kph, windgust_kph,
                     winddir_deg, precip_mm, precip_prob, preciptype,
                     snow_cm, snowdepth_cm, solarradiation_wm2,
                     solarenergy_mj, uvindex, conditions, source,
                     observed_at_utc            -- derived from datetime_epoch
                   }, ...
                ]
              }, ...
          ]
        }

    Cost: ~24 queryCost per day returned (one per hour) plus 1 per day.
    For a full year that's ≈ 9,125 records per call — well within VC's
    per-call response budget.  Caller should still chunk by month/year for
    safety.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} < start_date {start_date}")

    # ---- Cache lookup ----
    cache_path = None
    if use_cache and _is_month_aligned(start_date, end_date):
        cache_path = _cache_path_for_month(lat, lon, start_date)
        cached = _read_vc_cache(cache_path)
        if cached is not None:
            # Force query_cost to 0 on cache hits so daily-usage counters
            # don't double-count the original fetch.  Original cost is
            # preserved under "query_cost_original" for diagnostics.
            cached_result = dict(cached)
            cached_result["query_cost_original"] = cached_result.get("query_cost")
            cached_result["query_cost"] = 0
            cached_result["from_cache"] = True
            logger.debug(
                f"VC cache HIT  ({lat:.3f},{lon:.3f}) {start_date}..{end_date} "
                f"[orig_cost={cached_result.get('query_cost_original')}]"
            )
            return cached_result

    path = f"/{lat},{lon}/{start_date.isoformat()}/{end_date.isoformat()}"
    j = _get_vc(path, {
        "unitGroup": "metric",
        "include":   "days,hours,obs",
        "elements":  (
            "datetime,datetimeEpoch,temp,tempmax,tempmin,feelslike,"
            "humidity,dew,pressure,cloudcover,visibility,"
            "windspeed,windgust,winddir,"
            "precip,precipprob,preciptype,snow,snowdepth,"
            "solarradiation,solarenergy,uvindex,"
            "conditions,source,stations"
        ),
    })

    out_days: list[dict] = []
    for d in (j.get("days") or []):
        hrs: list[dict] = []
        for h in (d.get("hours") or []):
            epoch = h.get("datetimeEpoch")
            observed_at_utc = None
            if epoch is not None:
                try:
                    observed_at_utc = datetime.fromtimestamp(
                        int(epoch), tz=timezone.utc
                    ).isoformat()
                except (ValueError, OSError, TypeError):
                    observed_at_utc = None
            hrs.append({
                "datetime":             h.get("datetime"),
                "datetime_epoch":       epoch,
                "observed_at_utc":      observed_at_utc,
                "temp_c":               h.get("temp"),
                "feelslike_c":          h.get("feelslike"),
                "humidity":             h.get("humidity"),
                "dew_c":                h.get("dew"),
                "pressure_hpa":         h.get("pressure"),
                "cloudcover":           h.get("cloudcover"),
                "visibility_km":        h.get("visibility"),
                "windspeed_kph":        h.get("windspeed"),
                "windgust_kph":         h.get("windgust"),
                "winddir_deg":          h.get("winddir"),
                "precip_mm":            h.get("precip"),
                "precip_prob":          h.get("precipprob"),
                "preciptype":           h.get("preciptype"),
                "snow_cm":              h.get("snow"),
                "snowdepth_cm":         h.get("snowdepth"),
                "solarradiation_wm2":   h.get("solarradiation"),
                "solarenergy_mj":       h.get("solarenergy"),
                "uvindex":              h.get("uvindex"),
                "conditions":           h.get("conditions"),
                "source":               h.get("source"),
            })
        out_days.append({
            "date":      d.get("datetime"),
            "tempmax_c": d.get("tempmax"),
            "tempmin_c": d.get("tempmin"),
            "temp_c":    d.get("temp"),
            "source":    d.get("source"),
            "hours":     hrs,
        })

    result = {
        "resolved_address": j.get("resolvedAddress"),
        "timezone":         j.get("timezone"),
        "latitude":         j.get("latitude"),
        "longitude":        j.get("longitude"),
        "query_cost":       j.get("queryCost"),
        "days":             out_days,
        "from_cache":       False,
    }

    # ---- Cache write (only if month-aligned) ----
    if cache_path is not None:
        _write_vc_cache(cache_path, result)
        logger.debug(
            f"VC cache MISS — wrote {cache_path.name} "
            f"({len(out_days)} days, queryCost={result['query_cost']})"
        )

    logger.info(
        f"VC history_with_hours ({lat:.3f},{lon:.3f}) {start_date}..{end_date}: "
        f"{len(out_days)} days (queryCost={result['query_cost']})"
    )
    return result


# ---------------------------------------------------------------------------
# Integration 2 — Intraday observed + forecast snapshot
# ---------------------------------------------------------------------------

def fetch_future_day(lat: float, lon: float, target_date: date) -> dict:
    """Pull VC forecast for a single FUTURE date (D+1, D+2...).  Used by the
    Phase 2c diagnostic layer only — does NOT feed production μ/σ.

    Returns a dict shaped like fetch_intraday() so downstream code can use
    the same diagnostic builder.  Cost: 24 queryCost per call.
    """
    path = f"/{lat},{lon}/{target_date.isoformat()}"
    j = _get_vc(path, {
        "unitGroup": "metric",
        "include":   "hours,days",
        "elements":  (
            "datetime,datetimeEpoch,temp,tempmax,tempmin,feelslike,"
            "humidity,dew,pressure,cloudcover,visibility,"
            "windspeed,windgust,winddir,"
            "precip,precipprob,preciptype,snow,snowdepth,"
            "solarradiation,solarenergy,uvindex,"
            "conditions,source,stations"
        ),
    })
    days = j.get("days") or []
    if not days:
        raise RuntimeError(f"VC returned no days for ({lat},{lon})/{target_date}")
    day = days[0]
    hours = day.get("hours") or []

    def _norm(h: dict) -> dict:
        return {
            "datetime":             h.get("datetime"),
            "datetime_epoch":       h.get("datetimeEpoch"),
            "temp_c":               h.get("temp"),
            "feelslike_c":          h.get("feelslike"),
            "humidity":             h.get("humidity"),
            "dew_c":                h.get("dew"),
            "pressure_hpa":         h.get("pressure"),
            "cloudcover":           h.get("cloudcover"),
            "visibility_km":        h.get("visibility"),
            "windspeed_kph":        h.get("windspeed"),
            "windgust_kph":         h.get("windgust"),
            "winddir_deg":          h.get("winddir"),
            "precip_mm":            h.get("precip"),
            "precip_prob":          h.get("precipprob"),
            "preciptype":           h.get("preciptype"),
            "snow_cm":              h.get("snow"),
            "snowdepth_cm":         h.get("snowdepth"),
            "solarradiation_wm2":   h.get("solarradiation"),
            "solarenergy_mj":       h.get("solarenergy"),
            "uvindex":              h.get("uvindex"),
            "conditions":           h.get("conditions"),
            "source":               h.get("source"),
            "stations":             list(h.get("stations") or []),
        }

    forecast = [_norm(h) for h in hours if h.get("source") == "fcst"]
    forecast_max = max((h["temp_c"] for h in forecast if h["temp_c"] is not None),
                       default=None)

    result = {
        "as_of":                    datetime.utcnow().isoformat() + "Z",
        "resolved_address":         j.get("resolvedAddress"),
        "timezone":                 j.get("timezone"),
        "current_temp_c":           None,
        "day_tempmax_estimate_c":   day.get("tempmax"),
        "day_vc_source":            day.get("source"),
        "observed_hours":           [],
        "forecast_hours":           forecast,
        "observed_max_so_far_c":    None,
        "forecast_remaining_max_c": forecast_max,
        "projected_day_max_c":      day.get("tempmax"),
        "stations":                 j.get("stations") or {},
        "query_cost":               j.get("queryCost"),
        "raw_day":                  day,
    }
    logger.info(
        f"VC future_day ({lat:.3f},{lon:.3f}) {target_date}: "
        f"day_max={result['day_tempmax_estimate_c']}°C fcst_hours={len(forecast)} "
        f"queryCost={result['query_cost']}"
    )
    return result


def fetch_intraday(lat: float, lon: float) -> dict:
    """
    Get a single snapshot of today at this location.  One API call returns:
        - VC's current-conditions reading
        - All 24 hours of today, each tagged source="obs" (past) or "fcst"
        - The blended daily max estimate

    Returns
    -------
    dict with keys:
        as_of                   (str)   — timestamp of the fetch (UTC iso)
        resolved_address        (str)
        timezone                (str)
        current_temp_c          (float | None)
        current_time            (str | None)        — VC's currentConditions.datetime
        current_stations        (list[str])
        day_tempmax_estimate_c  (float | None)      — VC's blended day high
        observed_hours          (list[dict])        — hours with source=="obs"
        forecast_hours          (list[dict])        — hours with source=="fcst"
        observed_max_so_far_c   (float | None)
        forecast_remaining_max_c(float | None)
        projected_day_max_c     (float | None)      — max of observed & forecast remaining
        stations                (dict)              — top-level station metadata
        query_cost              (int | None)
        raw_day                 (dict)              — raw day record for debugging

    Each element of observed_hours / forecast_hours is:
        {datetime, datetime_epoch, temp_c, source, stations}
    """
    path = f"/{lat},{lon}/today"
    j = _get_vc(path, {
        "unitGroup": "metric",
        "include":   "hours,current,days",
        "elements":  (
            "datetime,datetimeEpoch,temp,tempmax,tempmin,feelslike,"
            "humidity,dew,pressure,cloudcover,visibility,"
            "windspeed,windgust,winddir,"
            "precip,precipprob,preciptype,snow,snowdepth,"
            "solarradiation,solarenergy,uvindex,"
            "conditions,source,stations"
        ),
    })

    days = j.get("days") or []
    if not days:
        raise RuntimeError(f"VC returned no days for ({lat},{lon})/today")
    day = days[0]
    hours = day.get("hours") or []

    def _norm_hour(h: dict) -> dict:
        return {
            "datetime":             h.get("datetime"),
            "datetime_epoch":       h.get("datetimeEpoch"),
            "temp_c":               h.get("temp"),
            "feelslike_c":          h.get("feelslike"),
            "humidity":             h.get("humidity"),
            "dew_c":                h.get("dew"),
            "pressure_hpa":         h.get("pressure"),
            "cloudcover":           h.get("cloudcover"),
            "visibility_km":        h.get("visibility"),
            "windspeed_kph":        h.get("windspeed"),
            "windgust_kph":         h.get("windgust"),
            "winddir_deg":          h.get("winddir"),
            "precip_mm":            h.get("precip"),
            "precip_prob":          h.get("precipprob"),
            "preciptype":           h.get("preciptype"),
            "snow_cm":              h.get("snow"),
            "snowdepth_cm":         h.get("snowdepth"),
            "solarradiation_wm2":   h.get("solarradiation"),
            "solarenergy_mj":       h.get("solarenergy"),
            "uvindex":              h.get("uvindex"),
            "conditions":           h.get("conditions"),
            "source":               h.get("source"),
            "stations":             list(h.get("stations") or []),
        }

    observed  = [_norm_hour(h) for h in hours if h.get("source") == "obs"]
    forecast  = [_norm_hour(h) for h in hours if h.get("source") == "fcst"]

    observed_max  = max((h["temp_c"] for h in observed if h["temp_c"] is not None), default=None)
    forecast_max  = max((h["temp_c"] for h in forecast if h["temp_c"] is not None), default=None)
    projected = None
    if observed_max is not None and forecast_max is not None:
        projected = max(observed_max, forecast_max)
    elif observed_max is not None:
        projected = observed_max
    elif forecast_max is not None:
        projected = forecast_max

    current = j.get("currentConditions") or {}
    result = {
        "as_of":                    datetime.utcnow().isoformat() + "Z",
        "resolved_address":         j.get("resolvedAddress"),
        "timezone":                 j.get("timezone"),
        "current_temp_c":           current.get("temp"),
        "current_feelslike_c":      current.get("feelslike"),
        "current_humidity":         current.get("humidity"),
        "current_dew_c":            current.get("dew"),
        "current_pressure_hpa":     current.get("pressure"),
        "current_cloudcover":       current.get("cloudcover"),
        "current_visibility_km":    current.get("visibility"),
        "current_windspeed_kph":    current.get("windspeed"),
        "current_windgust_kph":     current.get("windgust"),
        "current_winddir_deg":      current.get("winddir"),
        "current_precip_mm":        current.get("precip"),
        "current_preciptype":       current.get("preciptype"),
        "current_snow_cm":          current.get("snow"),
        "current_snowdepth_cm":     current.get("snowdepth"),
        "current_solarradiation_wm2": current.get("solarradiation"),
        "current_solarenergy_mj":   current.get("solarenergy"),
        "current_uvindex":          current.get("uvindex"),
        "current_conditions":       current.get("conditions"),
        "current_vc_source":        current.get("source"),
        "current_time":             current.get("datetime"),
        "current_stations":         list(current.get("stations") or []),
        "day_tempmax_estimate_c":   day.get("tempmax"),
        "day_vc_source":            day.get("source"),
        "observed_hours":           observed,
        "forecast_hours":           forecast,
        "observed_max_so_far_c":    observed_max,
        "forecast_remaining_max_c": forecast_max,
        "projected_day_max_c":      projected,
        "stations":                 j.get("stations") or {},
        "query_cost":               j.get("queryCost"),
        "raw_day":                  day,
    }
    logger.debug(
        f"VC intraday ({lat:.3f},{lon:.3f}): "
        f"obs_hours={len(observed)} fcst_hours={len(forecast)} "
        f"obs_max={observed_max}°C fcst_max={forecast_max}°C "
        f"projected={projected}°C queryCost={result['query_cost']}"
    )
    return result
