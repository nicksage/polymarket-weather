"""
openmeteo_previous_runs.py — Client for Open-Meteo's Previous Runs API.

The Previous Runs API (https://previous-runs-api.open-meteo.com) lets us ask
"what did model X predict for this location, as of N days before the target
date?"  It exposes lead-time-controlled forecasts via suffix variables:

    temperature_2m                    — latest run (day-0, near analysis)
    temperature_2m_previous_day1      — forecast issued 24h before the target
    temperature_2m_previous_day3      — forecast issued 72h before the target
    ... up to _previous_day7

This is the correct API for bias correction because it preserves lead-time
semantics — i.e. the forecast that was actually seen at the time a trade
would have been placed.

Two quirks of the API that this module hides:

  1. Only HOURLY variables support the _previous_dayN suffix — there is no
     temperature_2m_max_previous_dayN.  We fetch hourly temps and compute
     daily max client-side.
  2. When multiple models are requested in one call, response fields are
     auto-suffixed with the model id, e.g.
         temperature_2m_previous_day3_ecmwf_ifs025
         temperature_2m_previous_day3_gfs_global
     We normalise these into a (date, model) tuple keyspace.

All daily grouping is done in the LOCATION'S LOCAL TIMEZONE (timezone=auto)
because Polymarket contracts resolve on the local calendar day.
"""

import logging
from collections import defaultdict
from datetime import date
from typing import Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

OM_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Canonical model ids supported by this module.  Add more as needed — the
# Previous Runs API supports many (icon, gem, arpege, jma, etc.).
SUPPORTED_MODELS = ("ecmwf_ifs025", "gfs_global")


# ---------------------------------------------------------------------------
# Retry primitive — network / 429 / 5xx are transient
# ---------------------------------------------------------------------------

class _RetryableOMError(Exception):
    """Transient error worth retrying."""


@retry(
    retry=retry_if_exception_type(_RetryableOMError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    reraise=True,
)
def _get_om(params: dict, timeout: float = 90.0) -> dict:
    try:
        r = httpx.get(OM_PREVIOUS_RUNS, params=params, timeout=timeout)
    except httpx.HTTPError as e:
        raise _RetryableOMError(f"Network error: {e}") from e

    if r.status_code in (429, 500, 502, 503, 504):
        raise _RetryableOMError(f"Transient {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        logger.error(
            f"Open-Meteo Previous Runs returned {r.status_code} — body: {r.text[:300]}"
        )
        r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Field-name resolution for single-model vs multi-model responses
# ---------------------------------------------------------------------------

def _field_for(hourly_keys: list[str], base: str, model: str) -> str | None:
    """
    Find the hourly response key that matches `base` (e.g. "temperature_2m_
    previous_day3") for the requested `model`.

    Open-Meteo behaviour:
      • Single-model request: key is exactly `base` (no suffix).
      • Multi-model request:  key is `base_<model>`  (suffix appended).
    """
    suffixed = f"{base}_{model}"
    if suffixed in hourly_keys:
        return suffixed
    if base in hourly_keys:
        # Single-model call — only valid if exactly one model was requested
        return base
    return None


# ---------------------------------------------------------------------------
# Public: fetch historical forecasts for one or more models at a fixed lead
# ---------------------------------------------------------------------------

def fetch_historical_forecasts(
    lat: float,
    lon: float,
    models: Iterable[str] = SUPPORTED_MODELS,
    lead_days: int = 3,
    past_days: int = 90,
) -> list[dict]:
    """
    Fetch hourly historical forecasts for `models` at a fixed lead time,
    roll them up into daily max temperatures, and return one row per
    (date, model) pair.

    Parameters
    ----------
    lat, lon
        Location.
    models
        Iterable of Open-Meteo Previous Runs model ids (see SUPPORTED_MODELS).
    lead_days
        Which `_previous_dayN` suffix to request (1..7).  Canonical choice
        for Polymarket bias correction is 3 — balances accuracy vs.
        typical contract horizon.
    past_days
        How far back to fetch.  Open-Meteo's own caps are ~1000 for ECMWF
        and ~1500 for GFS (day-3 lead).  Pass a large value for the initial
        backfill; rows with no data are simply omitted.

    Returns
    -------
    list of dicts, one per (date, model) pair, each:
        {
          "date":              str,     # YYYY-MM-DD, in LOCAL tz
          "model":             str,
          "lead_days":         int,
          "forecast_tempmax_c": float,
          "n_hours":           int,     # hours used to compute the max (≤24)
        }
    Rows with fewer than 12 hours of data (incomplete day at range edges)
    are skipped.
    """
    models = list(models)
    if not models:
        raise ValueError("At least one model must be specified")
    if not (1 <= lead_days <= 7):
        raise ValueError(f"lead_days must be 1..7, got {lead_days}")

    base_var = f"temperature_2m_previous_day{lead_days}"

    params = {
        "latitude":      lat,
        "longitude":     lon,
        "models":        ",".join(models),
        "hourly":        base_var,
        "past_days":     int(past_days),
        "forecast_days": 0,
        "timezone":      "auto",   # group by local calendar day
    }
    logger.info(
        f"OM Previous Runs call ({lat:.3f},{lon:.3f}) models={models} "
        f"lead={lead_days}d past_days={past_days}"
    )
    j = _get_om(params)

    hourly = j.get("hourly") or {}
    times  = hourly.get("time") or []
    if not times:
        logger.warning(f"OM Previous Runs returned no data for ({lat},{lon})")
        return []

    # For each model, find its response key, then group by local date -> max.
    out: list[dict] = []
    for model in models:
        # When only one model is requested the key is unsuffixed; otherwise suffixed.
        key = _field_for(list(hourly.keys()), base_var, model) if len(models) > 1 else (
            base_var if base_var in hourly else None
        )
        if not key:
            logger.warning(f"OM response missing expected field for {model} (base={base_var})")
            continue

        values = hourly[key]
        bucket: dict[str, list[float]] = defaultdict(list)
        for t, v in zip(times, values):
            if v is None:
                continue
            d = t[:10]   # YYYY-MM-DD in local tz because timezone=auto
            bucket[d].append(float(v))

        kept = 0
        skipped = 0
        for d, vs in sorted(bucket.items()):
            if len(vs) < 12:
                skipped += 1
                continue
            out.append({
                "date":               d,
                "model":              model,
                "lead_days":          lead_days,
                "forecast_tempmax_c": max(vs),
                "n_hours":            len(vs),
            })
            kept += 1
        logger.info(
            f"OM {model} day{lead_days} ({lat:.3f},{lon:.3f}): "
            f"{kept} complete days, {skipped} partial-days skipped"
        )

    return out
