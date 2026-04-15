"""
live_adjustment.py — Live intraday adjustment layer (Phase 2b, v1).

Sits on top of the existing ECMWF + GFS + climatology blend without
modifying it.  Reads the Phase-2 DB tables (live_observations,
forecast_hourly, forecast_runs) to compute:

  * mu_adjustment_c  — signed additive correction to the blended μ
  * adjusted_mu_c    — blended μ + mu_adjustment_c
  * adjusted_sigma_c — blended σ (unchanged in v1)
  * live_adjustment_score — |mu_adjustment_c| / blended_sigma (units of σ)
  * observed_max_so_far_c — for downstream floor truncation
  * components       — typed JSON of every intermediate value for audit

v1 scope (deliberately minimal):
  - smoothed intraday residual (current_temp - expected_temp_now)
  - time-of-day weight (triangular, peak at local 13:00)
  - recency weight (ramp up when last forecast >2 h old)
  - hard cap: min(LIVE_ADJ_MAX_ABS_C, LIVE_ADJ_MAX_SIGMA * σ)
  - cold-start fallback when <LIVE_ADJ_MIN_OBS observations available

Deferred to later phases: σ shrinkage, forecast-disagreement weighting,
cloud/wind deviation, multi-station consensus.

Pure function — no API calls, no mutation.  Safe to call on every edge
scan.  Returns a dict; caller persists `adjusted_*` + `components` JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import math

from config import (
    LIVE_ADJ_ALPHA,
    LIVE_ADJ_MAX_ABS_C,
    LIVE_ADJ_MAX_SIGMA,
    LIVE_ADJ_MIN_OBS,
    LIVE_ADJ_SMOOTH_WINDOW_MIN,
    LIVE_ADJ_TOD_PEAK_HOUR,
    LIVE_ADJ_TOD_RAMP_HOURS,
    LIVE_ADJ_SIGMA_ABS_FLOOR_C,
    LIVE_ADJ_SIGMA_REL_FLOOR,
    LIVE_ADJ_SIGMA_RESIDUAL_CAP,
    LIVE_ADJ_SIGMA_RESIDUAL_DIV,
    LIVE_ADJ_SIGMA_TIME_MIN,
)
from db import (
    get_latest_forecast_hourly,
    get_latest_forecast_run,
    get_recent_observations,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _local_hour(utc_dt: datetime, timezone_str: str | None) -> float | None:
    """Return the local-time hour (0..24 float) for a UTC datetime.
    Falls back to UTC hour when tz parsing fails."""
    if utc_dt is None:
        return None
    try:
        if timezone_str:
            from zoneinfo import ZoneInfo
            local = utc_dt.astimezone(ZoneInfo(timezone_str))
            return local.hour + local.minute / 60.0
    except Exception:
        pass
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    u = utc_dt.astimezone(timezone.utc)
    return u.hour + u.minute / 60.0


def _triangular_weight(hour: float, peak: float, ramp: float) -> float:
    """Triangle centered at `peak` with half-width `ramp`.
    Returns 1.0 at peak, 0.0 at peak±ramp, linear between, clipped to [0,1]."""
    if hour is None or ramp <= 0:
        return 0.0
    dist = abs(hour - peak)
    if dist >= ramp:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist / ramp))


def _interp_expected_temp(hourly_rows: list[dict], at_utc: datetime) -> float | None:
    """Return the temp_c from the forecast_hourly row closest to at_utc
    (within 90 minutes).  None if no row is close enough."""
    if not hourly_rows or at_utc is None:
        return None
    best_row = None
    best_gap: float | None = None
    for r in hourly_rows:
        ts = _parse_iso(r.get("hour_ts_utc"))
        if ts is None:
            continue
        gap = abs((ts - at_utc).total_seconds())
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_row = r
    if best_row is None or best_gap is None or best_gap > 90 * 60:
        return None
    return best_row.get("temp_c")


def _smooth_residuals(
    obs_rows: list[dict],
    hourly_rows: list[dict],
    now_utc: datetime,
    window_min: int,
) -> tuple[float | None, float | None, int, bool]:
    """Compute the residual for every obs in the last `window_min` minutes
    where both current_temp and expected_temp_now are available.

    Returns (raw_residual_now, smoothed_residual, n_used, station_changed).
      raw_residual_now   — residual on the most recent obs (no smoothing)
      smoothed_residual  — mean of all residuals in window
      n_used             — number of residuals used
      station_changed    — True if the station list differs across window
    """
    if not obs_rows:
        return (None, None, 0, False)

    window_start = now_utc - timedelta(minutes=window_min)

    residuals: list[float] = []
    last_stations: str | None = None
    station_changed = False
    raw_residual_now: float | None = None

    # Iterate chronologically (obs_rows is already ASC from get_recent_observations)
    for r in obs_rows:
        pulled = _parse_iso(r.get("pulled_at_utc"))
        if pulled is None or pulled < window_start:
            continue
        curr = r.get("current_temp_c")
        if curr is None:
            continue
        obs_at = _parse_iso(r.get("observed_at_utc")) or pulled
        expected = _interp_expected_temp(hourly_rows, obs_at)
        if expected is None:
            continue
        residual = float(curr) - float(expected)
        residuals.append(residual)
        raw_residual_now = residual  # last one wins (chronological)

        stations = r.get("stations")
        if last_stations is not None and stations != last_stations:
            station_changed = True
        last_stations = stations

    if not residuals:
        return (None, None, 0, station_changed)
    smoothed = sum(residuals) / len(residuals)
    return (raw_residual_now, smoothed, len(residuals), station_changed)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _compute_shrunk_sigma(
    blended_sigma_c: float,
    remaining_daylight_frac: float | None,
    residual_c: float | None,
    time_enabled: bool,
    residual_enabled: bool,
) -> tuple[float, dict]:
    """Dynamic σ shrinkage (Phase 2b-v2).

    Model:
        time_factor     = max(sqrt(remaining_daylight_frac), TIME_MIN)  if enabled else 1.0
        residual_factor = 1 + min(|residual|/DIV, CAP)                   if enabled else 1.0
        shrunk          = blended_sigma * time_factor * residual_factor
        shrunk          = max(shrunk, ABS_FLOOR_C)                       # hard absolute floor
        shrunk          = max(shrunk, REL_FLOOR * blended_sigma)         # hard relative floor

    Both floors apply independently to prevent collapse.  Returns (shrunk_sigma,
    components_dict) with every intermediate value for auditability.
    """
    # Time factor
    if time_enabled and remaining_daylight_frac is not None:
        time_factor = max(math.sqrt(max(remaining_daylight_frac, 0.0)),
                          LIVE_ADJ_SIGMA_TIME_MIN)
    else:
        time_factor = 1.0

    # Residual factor — widens σ on surprising days
    if residual_enabled and residual_c is not None:
        residual_factor = 1.0 + min(abs(residual_c) / LIVE_ADJ_SIGMA_RESIDUAL_DIV,
                                    LIVE_ADJ_SIGMA_RESIDUAL_CAP)
    else:
        residual_factor = 1.0

    pre_floor = blended_sigma_c * time_factor * residual_factor

    abs_floor = LIVE_ADJ_SIGMA_ABS_FLOOR_C
    rel_floor = LIVE_ADJ_SIGMA_REL_FLOOR * blended_sigma_c
    shrunk = max(pre_floor, abs_floor, rel_floor)

    abs_floor_hit = pre_floor < abs_floor
    rel_floor_hit = pre_floor < rel_floor

    return shrunk, {
        "sigma_time_factor":       round(time_factor, 4),
        "sigma_residual_factor":   round(residual_factor, 4),
        "sigma_pre_floor_c":       round(pre_floor, 4),
        "sigma_abs_floor_hit":     abs_floor_hit,
        "sigma_rel_floor_hit":     rel_floor_hit,
        "sigma_remaining_frac":    (None if remaining_daylight_frac is None
                                    else round(remaining_daylight_frac, 4)),
        "sigma_residual_c":        (None if residual_c is None
                                    else round(residual_c, 3)),
    }


def _cold_start_result(
    reason: str, blended_mu_c: float, blended_sigma_c: float,
    extra: dict | None = None,
) -> dict:
    comps = {
        "raw_residual_c":      None,
        "smoothed_residual_c": None,
        "tod_weight":          None,
        "recency_weight":      None,
        "alpha":               LIVE_ADJ_ALPHA,
        "hours_since_forecast": None,
        "n_obs_used":          0,
        "station_changed":     False,
        "cap_abs_c":           LIVE_ADJ_MAX_ABS_C,
        "cap_sigma_c":         LIVE_ADJ_MAX_SIGMA * blended_sigma_c,
        "capped_by":           None,
        "cold_start":          True,
        "reason":              reason,
    }
    if extra:
        comps.update(extra)
    return {
        "adjusted_mu_c":          blended_mu_c,
        "adjusted_sigma_c":       blended_sigma_c,
        "mu_adjustment_c":        0.0,
        "observed_max_so_far_c":  (extra or {}).get("observed_max_so_far_c"),
        "live_adjustment_score":  0.0,
        "components":             comps,
    }


def _compute_from_data(
    blended_mu_c: float,
    blended_sigma_c: float,
    obs_rows: list[dict],
    hourly_rows: list[dict],
    hours_since_forecast: float | None,
    timezone_str: str | None,
    now_utc: datetime,
    residual_enabled: bool = True,
    sigma_mode: str = "none",
    remaining_daylight_frac: float | None = None,
) -> dict:
    """Pure-data implementation of the live adjustment.  Reused by both
    the production wrapper (compute_live_adjustment) and the backtest
    replay (which passes rows from backtest tables instead of live DB).

    residual_enabled=False — backtest-only floor-focused mode.  The residual
    (μ adjustment) is hard-zeroed but observed_max_so_far_c is still returned
    so the caller can apply the floor at the probability layer.  Never use
    this in production paths.

    sigma_mode — dynamic σ shrinkage (Phase 2b-v2):
        "none"                 — σ = blended_sigma_c (current production default)
        "time_only"            — shrink σ by time factor only
        "time+residual"        — production mode; residual = smoothed intraday residual
        "time+surprise_proxy"  — backtest mode; residual = current_temp - blended_mu_c
                                  (used when no true forecast path is available)
    """
    # Observed max so far (pure obs rollup).  Uses observed_max_so_far_c if
    # present on the obs row (live path); falls back to current_temp_c max
    # (backtest path where obs rows are bare hourly readings).
    observed_max: float | None = None
    for r in obs_rows:
        v = r.get("observed_max_so_far_c")
        if v is None:
            v = r.get("current_temp_c")
        if v is not None:
            observed_max = v if observed_max is None else max(observed_max, v)

    extra = {"observed_max_so_far_c": observed_max}

    # ---- σ shrinkage (v2) resolution ----
    # Figure out which residual (if any) feeds the residual_factor.
    def _resolve_sigma_residual() -> float | None:
        mode = sigma_mode
        if mode in ("none", "time_only"):
            return None
        if mode == "time+surprise_proxy":
            # Day-level surprise used by the Part-1 backtest where we don't
            # have a true historical forecast path.  = latest_obs - blended_mu
            if observed_max is None:
                return None
            # Use the most recent observation's current_temp_c (not the max)
            # so the residual reflects the day's direction, not its peak.
            latest = obs_rows[-1] if obs_rows else None
            if not latest:
                return None
            cur = latest.get("current_temp_c")
            if cur is None:
                return None
            return float(cur) - float(blended_mu_c)
        # "time+residual" — production path (set later after smoothed_residual
        # is computed).  For floor-only mode without residual, we fall back to
        # the surprise-proxy so callers can still exercise the residual term
        # in isolation when they explicitly ask for it.
        if mode == "time+residual" and residual_enabled is False:
            latest = obs_rows[-1] if obs_rows else None
            if latest and latest.get("current_temp_c") is not None:
                return float(latest["current_temp_c"]) - float(blended_mu_c)
        return None

    def _shrink(sigma: float, resid: float | None) -> tuple[float, dict]:
        if sigma_mode == "none":
            return sigma, {"sigma_mode": "none"}
        time_on = sigma_mode in ("time_only", "time+residual", "time+surprise_proxy")
        resid_on = sigma_mode in ("time+residual", "time+surprise_proxy")
        s, comps = _compute_shrunk_sigma(
            blended_sigma_c=sigma,
            remaining_daylight_frac=remaining_daylight_frac,
            residual_c=resid,
            time_enabled=time_on,
            residual_enabled=resid_on,
        )
        comps["sigma_mode"] = sigma_mode
        return s, comps

    # Floor-only backtest mode: skip the residual math entirely, but preserve
    # the observed_max_so_far_c so the caller can apply the floor truncation.
    if not residual_enabled:
        sigma_resid = _resolve_sigma_residual()
        shrunk_sigma, sigma_comps = _shrink(blended_sigma_c, sigma_resid)
        comps = {
            "raw_residual_c":      None,
            "smoothed_residual_c": None,
            "tod_weight":          None,
            "recency_weight":      None,
            "alpha":               LIVE_ADJ_ALPHA,
            "hours_since_forecast": hours_since_forecast,
            "n_obs_used":          0,
            "station_changed":     False,
            "cap_abs_c":           LIVE_ADJ_MAX_ABS_C,
            "cap_sigma_c":         LIVE_ADJ_MAX_SIGMA * blended_sigma_c,
            "capped_by":           None,
            "cold_start":          False,
            "reason":              "floor_only_backtest",
            "observed_max_so_far_c": observed_max,
        }
        comps.update(sigma_comps)
        return {
            "adjusted_mu_c":          blended_mu_c,
            "adjusted_sigma_c":       round(shrunk_sigma, 3),
            "mu_adjustment_c":        0.0,
            "observed_max_so_far_c":  observed_max,
            "live_adjustment_score":  0.0,
            "components":             comps,
        }

    # ---- Full residual + caps path (production) ----
    if not hourly_rows:
        return _cold_start_result("no_forecast_hourly", blended_mu_c, blended_sigma_c, extra)
    if len(obs_rows) < LIVE_ADJ_MIN_OBS:
        return _cold_start_result("insufficient_obs", blended_mu_c, blended_sigma_c, extra)

    raw_res, smoothed_res, n_used, station_changed = _smooth_residuals(
        obs_rows, hourly_rows, now_utc, LIVE_ADJ_SMOOTH_WINDOW_MIN,
    )
    if smoothed_res is None or n_used < LIVE_ADJ_MIN_OBS:
        return _cold_start_result("no_matched_residuals", blended_mu_c, blended_sigma_c, extra)

    local_hour = _local_hour(now_utc, timezone_str)
    tod_w = _triangular_weight(
        local_hour or 0.0,
        peak=float(LIVE_ADJ_TOD_PEAK_HOUR),
        ramp=float(LIVE_ADJ_TOD_RAMP_HOURS),
    )
    if hours_since_forecast is None:
        recency_w = 1.0
    else:
        recency_w = 1.0 + min(max(hours_since_forecast, 0.0) / 2.0, 1.0) * 0.3

    raw_adjustment = smoothed_res * tod_w * recency_w * LIVE_ADJ_ALPHA

    cap_abs = LIVE_ADJ_MAX_ABS_C
    cap_sigma = LIVE_ADJ_MAX_SIGMA * max(blended_sigma_c, 1e-6)
    cap = min(cap_abs, cap_sigma)
    if raw_adjustment > cap:
        mu_adjustment = cap
        capped_by = "abs" if cap_abs <= cap_sigma else "sigma"
    elif raw_adjustment < -cap:
        mu_adjustment = -cap
        capped_by = "abs" if cap_abs <= cap_sigma else "sigma"
    else:
        mu_adjustment = raw_adjustment
        capped_by = None

    adjusted_mu = blended_mu_c + mu_adjustment

    # σ shrinkage — uses the SMOOTHED intraday residual for 'time+residual' mode.
    # Falls back to surprise-proxy if explicitly set.
    if sigma_mode == "time+residual":
        shrunk_sigma, sigma_comps = _shrink(blended_sigma_c, smoothed_res)
    elif sigma_mode == "time+surprise_proxy":
        proxy_resid = None
        if observed_max is not None:
            latest = obs_rows[-1] if obs_rows else None
            if latest and latest.get("current_temp_c") is not None:
                proxy_resid = float(latest["current_temp_c"]) - float(blended_mu_c)
        shrunk_sigma, sigma_comps = _shrink(blended_sigma_c, proxy_resid)
    elif sigma_mode == "time_only":
        shrunk_sigma, sigma_comps = _shrink(blended_sigma_c, None)
    else:
        shrunk_sigma, sigma_comps = blended_sigma_c, {"sigma_mode": "none"}

    adjusted_sigma = shrunk_sigma
    score = abs(mu_adjustment) / max(blended_sigma_c, 1e-6)

    components = {
        "raw_residual_c":       None if raw_res is None else round(raw_res, 3),
        "smoothed_residual_c":  round(smoothed_res, 3),
        "tod_weight":           round(tod_w, 3),
        "recency_weight":       round(recency_w, 3),
        "alpha":                LIVE_ADJ_ALPHA,
        "hours_since_forecast": (None if hours_since_forecast is None
                                 else round(hours_since_forecast, 2)),
        "local_hour":           (None if local_hour is None
                                 else round(local_hour, 2)),
        "n_obs_used":           n_used,
        "station_changed":      bool(station_changed),
        "cap_abs_c":            cap_abs,
        "cap_sigma_c":          round(cap_sigma, 3),
        "capped_by":            capped_by,
        "cold_start":           False,
        "reason":               "ok",
        "observed_max_so_far_c": observed_max,
    }
    components.update(sigma_comps)

    return {
        "adjusted_mu_c":         round(adjusted_mu, 3),
        "adjusted_sigma_c":      round(adjusted_sigma, 3),
        "mu_adjustment_c":       round(mu_adjustment, 3),
        "observed_max_so_far_c": observed_max,
        "live_adjustment_score": round(score, 4),
        "components":            components,
    }


def compute_live_adjustment(
    event_id: str,
    blended_mu_c: float,
    blended_sigma_c: float,
    target_date: str,
    timezone_str: str | None = None,
    now_utc: datetime | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    """Production entry point — fetches live data from the DB and calls
    _compute_from_data with residual_enabled=True.  Sigma mode is gated by
    USE_LIVE_ADJUSTMENT_SIGMA_SHRINK (default False)."""
    from config import (
        USE_LIVE_ADJUSTMENT_SIGMA_SHRINK,
        LIVE_ADJ_SIGMA_TIME_ENABLED,
        LIVE_ADJ_SIGMA_RESIDUAL_ENABLED,
    )

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    since = (now_utc - timedelta(minutes=max(LIVE_ADJ_SMOOTH_WINDOW_MIN * 3, 180))).isoformat()
    obs_rows = get_recent_observations(event_id, since) or []

    latest_ecmwf = get_latest_forecast_run(event_id, "ecmwf")
    hourly_rows: list[dict] = []
    hours_since_forecast: float | None = None
    if latest_ecmwf:
        hourly_rows = get_latest_forecast_hourly(latest_ecmwf["id"]) or []
        pulled = _parse_iso(latest_ecmwf.get("pulled_at"))
        if pulled is not None:
            hours_since_forecast = (now_utc - pulled).total_seconds() / 3600.0

    # Sigma shrinkage mode resolution (production)
    if USE_LIVE_ADJUSTMENT_SIGMA_SHRINK:
        if LIVE_ADJ_SIGMA_TIME_ENABLED and LIVE_ADJ_SIGMA_RESIDUAL_ENABLED:
            sigma_mode = "time+residual"
        elif LIVE_ADJ_SIGMA_TIME_ENABLED:
            sigma_mode = "time_only"
        else:
            sigma_mode = "none"
    else:
        sigma_mode = "none"

    # Solar fraction — best-effort; None in polar regions or on failure.
    remaining_frac: float | None = None
    if sigma_mode != "none" and lat is not None and lon is not None:
        try:
            from solar import remaining_daylight_fraction
            remaining_frac = remaining_daylight_fraction(lat, lon, now_utc)
        except Exception:
            remaining_frac = None

    return _compute_from_data(
        blended_mu_c=blended_mu_c,
        blended_sigma_c=blended_sigma_c,
        obs_rows=obs_rows,
        hourly_rows=hourly_rows,
        hours_since_forecast=hours_since_forecast,
        timezone_str=timezone_str,
        now_utc=now_utc,
        residual_enabled=True,
        sigma_mode=sigma_mode,
        remaining_daylight_frac=remaining_frac,
    )


def components_to_json(components: dict) -> str:
    """Serialize the components dict for storage.  Kept as a helper so the
    format stays consistent across writers."""
    return json.dumps(components, separators=(",", ":"), default=str)
