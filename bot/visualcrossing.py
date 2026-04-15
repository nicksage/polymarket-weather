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

import os
import logging
from datetime import date, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
        logger.info(f"VC call {path_suffix} succeeded (queryCost={cost})")
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
# Integration 2 — Intraday observed + forecast snapshot
# ---------------------------------------------------------------------------

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
        "elements":  "datetime,datetimeEpoch,temp,tempmax,tempmin,source,stations",
    })

    days = j.get("days") or []
    if not days:
        raise RuntimeError(f"VC returned no days for ({lat},{lon})/today")
    day = days[0]
    hours = day.get("hours") or []

    def _norm_hour(h: dict) -> dict:
        return {
            "datetime":       h.get("datetime"),
            "datetime_epoch": h.get("datetimeEpoch"),
            "temp_c":         h.get("temp"),
            "source":         h.get("source"),
            "stations":       list(h.get("stations") or []),
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
        "current_time":             current.get("datetime"),
        "current_stations":         list(current.get("stations") or []),
        "day_tempmax_estimate_c":   day.get("tempmax"),
        "observed_hours":           observed,
        "forecast_hours":           forecast,
        "observed_max_so_far_c":    observed_max,
        "forecast_remaining_max_c": forecast_max,
        "projected_day_max_c":      projected,
        "stations":                 j.get("stations") or {},
        "query_cost":               j.get("queryCost"),
        "raw_day":                  day,
    }
    logger.info(
        f"VC intraday ({lat:.3f},{lon:.3f}): "
        f"obs_hours={len(observed)} fcst_hours={len(forecast)} "
        f"obs_max={observed_max}°C fcst_max={forecast_max}°C "
        f"projected={projected}°C queryCost={result['query_cost']}"
    )
    return result
