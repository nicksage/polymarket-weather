"""
features.py — build_feature_vector(ctx): the single function used by BOTH
the training pipeline (historical backfill rows) and the live inference
hook.  Must stay aligned with schema.FEATURE_NAMES — add features only at
the end and bump FEATURE_VERSION when changing layout.

Missing inputs yield NaN in the corresponding slot; downstream model
(HistGradientBoostingRegressor) handles NaN natively.

Context dict contract (v2.0)
----------------------------
Required (same as v1.0):
  target_date         datetime.date       the day we're predicting T_max for
  decision_dt_utc     datetime (tz-aware) prediction moment in UTC
  decision_dt_local   datetime            prediction moment in local timezone

Recommended (NaN slots if absent):
  current_obs         dict | None         current VC observation at decision moment
                                          keys: temp_c, humidity, dew_c, pressure_hpa,
                                          cloudcover, visibility_km, windspeed_kph,
                                          winddir_deg, solarradiation_wm2, snowdepth_cm
  recent_obs          list[dict] | None   recent observations for trajectory features
                                          each: {observed_at_utc (ISO str),
                                                 temp_c, pressure_hpa, cloudcover}
  climatology         dict | None         {"mu_today_c": float | None}
  historical_tmax     dict | None         {"yesterday_c", "seven_days_ago_c", "last_year_c"}

NEW for v2.0 (NaN/zero slots if absent):
  city_name           str | None          looked up in city_static_features.json
  lat, lon            float | None        for solar geometry (clear-sky GHI)
  tz_offset_hours     float | None        local tz offset, for clear-sky integration window
  hourly_climatology  dict | None         (doy, hour) -> {temp_mu, dew_mu, ..., n_samples}
                                          pre-loaded once per city via
                                          db.load_all_obs_climatology_hourly(city)
"""

import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from ml.schema import FEATURE_NAMES, N_FEATURES
from ml.city_features import get_city_features, koppen_one_hot
from ml.solar import (
    clear_sky_ghi_wm2,
    clear_sky_ghi_remaining_kwh,
    noon_solar_elevation_deg,
)

_NAN = float("nan")

# Match windows for "nearest prior observation at lookback" — an obs is
# counted only if its timestamp is within this many seconds of the
# requested point.  30 min gives enough slack for a 20-min-cadence pull.
_LOOKBACK_MATCH_SECONDS = 1800

# Climatology rows with fewer than this many samples are too thin to use
# for anomaly z-scores; we fall back to NaN instead.
_MIN_CLIM_SAMPLES = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cyc(value: float | None, period: float) -> tuple[float, float]:
    """Encode a cyclical value as (sin, cos) over `period`.  None → (nan, nan)."""
    if value is None:
        return _NAN, _NAN
    theta = 2.0 * math.pi * float(value) / period
    return math.sin(theta), math.cos(theta)


def _parse_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None


def _obs_at_lookback(
    recent_obs: list[dict], decision_dt_utc: datetime, hours_back: float, key: str,
) -> float | None:
    """Value of `key` from the obs closest to (decision - hours_back), within ±30 min."""
    target_epoch = (decision_dt_utc - timedelta(hours=hours_back)).timestamp()
    best: float | None = None
    best_delta = float("inf")
    for o in recent_obs:
        o_dt = _parse_utc(o.get("observed_at_utc"))
        if o_dt is None:
            continue
        delta = abs(o_dt.timestamp() - target_epoch)
        if delta > _LOOKBACK_MATCH_SECONDS or delta >= best_delta:
            continue
        v = o.get(key)
        if v is None:
            continue
        best = float(v)
        best_delta = delta
    return best


def _mean_over_window(
    recent_obs: list[dict], decision_dt_utc: datetime, hours_back: float, key: str,
) -> float | None:
    lower = (decision_dt_utc - timedelta(hours=hours_back)).timestamp()
    upper = decision_dt_utc.timestamp()
    vals: list[float] = []
    for o in recent_obs:
        o_dt = _parse_utc(o.get("observed_at_utc"))
        if o_dt is None:
            continue
        o_epoch = o_dt.timestamp()
        if not (lower <= o_epoch <= upper):
            continue
        v = o.get(key)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _earliest_obs_today(
    recent_obs: list[dict], target_date: date, before_hour_utc: int,
) -> tuple[datetime, float] | None:
    """Find the earliest obs whose UTC date matches target_date AND whose
    UTC hour is < before_hour_utc, with non-null temp_c.  Returns
    (datetime, temp_c) or None."""
    best_dt = None
    best_temp = None
    for o in recent_obs:
        o_dt = _parse_utc(o.get("observed_at_utc"))
        if o_dt is None or o_dt.hour >= before_hour_utc:
            continue
        if o_dt.date() != target_date:
            continue
        t = o.get("temp_c")
        if t is None:
            continue
        if best_dt is None or o_dt < best_dt:
            best_dt = o_dt
            best_temp = float(t)
    if best_dt is None:
        return None
    return (best_dt, best_temp)


def _days_since_solstice(d: date) -> int:
    y = d.year
    candidates = [c for c in (date(y - 1, 12, 21), date(y, 6, 21), date(y, 12, 21)) if c <= d]
    return (d - max(candidates)).days


def _diff(a: float | None, b: float | None) -> float | None:
    return (a - b) if (a is not None and b is not None) else None


def _z_anomaly(
    value: float | None, mu: float | None, sigma: float | None, n_samples: int | None,
) -> float | None:
    """Standardized anomaly with thinness guard."""
    if value is None or mu is None or sigma is None:
        return None
    if n_samples is not None and n_samples < _MIN_CLIM_SAMPLES:
        return None
    if sigma < 0.01:           # avoid divide-by-near-zero
        return None
    return (float(value) - float(mu)) / float(sigma)


def _hours_since(dt_a: datetime | None, dt_b: datetime | None) -> float | None:
    if dt_a is None or dt_b is None:
        return None
    return (dt_b.timestamp() - dt_a.timestamp()) / 3600.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_vector(ctx: dict[str, Any]) -> np.ndarray:
    """Build the v2.0 feature vector (46 elements).  See module docstring
    for the ctx contract.  Missing inputs yield NaN in the corresponding slots."""
    target_date: date = ctx["target_date"]
    decision_dt_utc: datetime = ctx["decision_dt_utc"]
    decision_dt_local: datetime = ctx["decision_dt_local"]
    curr: dict = ctx.get("current_obs") or {}
    recent: list[dict] = ctx.get("recent_obs") or []
    clim: dict = ctx.get("climatology") or {}
    hist: dict = ctx.get("historical_tmax") or {}

    # v2.0 additions to ctx (graceful when missing)
    city_name = ctx.get("city_name")
    lat = ctx.get("lat")
    lon = ctx.get("lon")
    tz_offset_hours = ctx.get("tz_offset_hours")
    hourly_clim: dict = ctx.get("hourly_climatology") or {}

    doy = target_date.timetuple().tm_yday
    decision_hour_local = decision_dt_local.hour

    # =======================================================================
    # === v1.0 features ====================================================
    # =======================================================================

    doy_sin, doy_cos = _cyc(doy, 365.25)
    hour = decision_dt_local.hour + decision_dt_local.minute / 60.0
    hour_sin, hour_cos = _cyc(hour, 24.0)

    temp_now  = curr.get("temp_c")
    humidity  = curr.get("humidity")
    dew       = curr.get("dew_c")
    pressure  = curr.get("pressure_hpa")
    cc        = curr.get("cloudcover")
    vis       = curr.get("visibility_km")
    wspd      = curr.get("windspeed_kph")
    wdir      = curr.get("winddir_deg")
    solar_now = curr.get("solarradiation_wm2")
    snowdepth = curr.get("snowdepth_cm")
    winddir_sin, winddir_cos = _cyc(wdir, 360.0)

    temp_3h  = _obs_at_lookback(recent, decision_dt_utc, 3,  "temp_c")
    temp_6h  = _obs_at_lookback(recent, decision_dt_utc, 6,  "temp_c")
    temp_24h = _obs_at_lookback(recent, decision_dt_utc, 24, "temp_c")
    pres_3h  = _obs_at_lookback(recent, decision_dt_utc, 3,  "pressure_hpa")
    pres_24h = _obs_at_lookback(recent, decision_dt_utc, 24, "pressure_hpa")

    temp_change_3h  = _diff(temp_now, temp_3h)
    temp_change_6h  = _diff(temp_now, temp_6h)
    temp_change_24h = _diff(temp_now, temp_24h)
    pres_change_3h  = _diff(pressure, pres_3h)
    pres_change_24h = _diff(pressure, pres_24h)
    cc_mean_6h      = _mean_over_window(recent, decision_dt_utc, 6, "cloudcover")

    clim_mu = clim.get("mu_today_c")
    temp_minus_clim = _diff(temp_now, clim_mu)
    days_since_sol = _days_since_solstice(target_date)

    tmax_yesterday = hist.get("yesterday_c")
    tmax_7d_ago    = hist.get("seven_days_ago_c")
    tmax_last_year = hist.get("last_year_c")

    # =======================================================================
    # === v2.0 additions ===================================================
    # =======================================================================

    # --- Morning heating rate ---
    # Earliest temp obs from earlier today (UTC), before noon UTC, vs now.
    morning_heating_rate: float | None = None
    earliest = _earliest_obs_today(recent, decision_dt_utc.date(), before_hour_utc=12)
    if earliest is not None and temp_now is not None:
        early_dt, early_temp = earliest
        hours_elapsed = _hours_since(early_dt, decision_dt_utc)
        if hours_elapsed is not None and hours_elapsed >= 1.0:
            morning_heating_rate = (temp_now - early_temp) / hours_elapsed

    # --- Dew point depression + Tdd-rule proxy ---
    dew_depression = _diff(temp_now, dew)
    tmax_dewpoint_proxy: float | None = None
    if temp_now is not None and dew_depression is not None:
        tmax_dewpoint_proxy = temp_now + 0.6 * dew_depression

    # --- Solar / GHI features (need lat, lon, tz_offset) ---
    clear_sky_ghi_remaining_val: float | None = None
    insolation_efficiency: float | None = None
    noon_elev = None
    if lat is not None and lon is not None:
        # Remaining GHI (kWh/m²) from now to 16:00 local
        if tz_offset_hours is not None:
            try:
                clear_sky_ghi_remaining_val = clear_sky_ghi_remaining_kwh(
                    float(lat), float(lon), decision_dt_utc,
                    end_local_hour=16,
                    tz_offset_hours=float(tz_offset_hours),
                    step_minutes=30,
                )
            except (ValueError, OverflowError):
                pass
        # Insolation efficiency: observed_solar_now / theoretical clear_sky_now
        try:
            csky_now_w = clear_sky_ghi_wm2(float(lat), float(lon), decision_dt_utc)
            if csky_now_w > 30.0 and solar_now is not None:
                insolation_efficiency = max(0.0, min(1.5, float(solar_now) / csky_now_w))
        except (ValueError, OverflowError):
            pass
        # Noon solar elevation for the warming-potential composite
        try:
            noon_elev = noon_solar_elevation_deg(float(lat), float(lon), decision_dt_utc)
        except (ValueError, OverflowError):
            pass

    # --- Effective warming potential ---
    # (1 - cloud_frac) * dew_depression * cos(noon_zenith)
    effective_warming_potential: float | None = None
    if cc is not None and dew_depression is not None and noon_elev is not None:
        cloud_frac = max(0.0, min(1.0, float(cc) / 100.0))
        cos_noon_zenith = math.cos(math.radians(max(0.0, 90.0 - noon_elev)))
        effective_warming_potential = (1.0 - cloud_frac) * dew_depression * cos_noon_zenith

    # --- Climatology anomalies ---
    clim_lookup = hourly_clim.get((doy, decision_hour_local))
    pressure_anomaly = _z_anomaly(
        pressure,
        (clim_lookup or {}).get("pressure_mu"),
        (clim_lookup or {}).get("pressure_sigma"),
        (clim_lookup or {}).get("n_samples"),
    )
    dew_anomaly = _z_anomaly(
        dew,
        (clim_lookup or {}).get("dew_mu"),
        (clim_lookup or {}).get("dew_sigma"),
        (clim_lookup or {}).get("n_samples"),
    )
    cloudcover_anomaly = _z_anomaly(
        cc,
        (clim_lookup or {}).get("cloudcover_mu"),
        (clim_lookup or {}).get("cloudcover_sigma"),
        (clim_lookup or {}).get("n_samples"),
    )
    windspeed_anomaly = _z_anomaly(
        wspd,
        (clim_lookup or {}).get("windspeed_mu"),
        (clim_lookup or {}).get("windspeed_sigma"),
        (clim_lookup or {}).get("n_samples"),
    )

    # --- Inversion suspect flag ---
    # Mid-morning + clear + calm + slow warming = surface inversion that may persist.
    inversion_suspect: float = 0.0
    hours_since_decision_dawn = (decision_dt_local.hour - 6)   # rough "hours since sunrise"
    if (hours_since_decision_dawn > 2
            and temp_change_3h is not None and temp_change_3h < 0.5
            and cc is not None and cc < 30
            and wspd is not None and wspd < 5):
        inversion_suspect = 1.0

    # --- Static city features ---
    city = get_city_features(city_name)
    city_elevation_m = float(city.get("elevation_m") or 0.0)
    city_lat_val     = float(lat) if lat is not None else 0.0
    city_hemisphere  = float(city.get("hemisphere") or 0)
    city_coastal     = float(city.get("coastal") or 0)
    k_a, k_b, k_c, k_d = koppen_one_hot(city.get("koppen_main"))

    # =======================================================================
    # === Pack into the canonical FEATURE_NAMES order ======================
    # =======================================================================

    named: dict[str, float | None] = {
        # v1.0
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "temp_now_c": temp_now,
        "humidity": humidity,
        "dew_c": dew,
        "pressure_hpa": pressure,
        "cloudcover": cc,
        "visibility_km": vis,
        "windspeed_kph": wspd,
        "winddir_sin": winddir_sin,
        "winddir_cos": winddir_cos,
        "solarradiation_wm2": solar_now,
        "snowdepth_cm": snowdepth,
        "temp_change_3h": temp_change_3h,
        "temp_change_6h": temp_change_6h,
        "temp_change_24h": temp_change_24h,
        "pressure_change_3h": pres_change_3h,
        "pressure_change_24h": pres_change_24h,
        "cloudcover_mean_6h": cc_mean_6h,
        "climatology_mu_today": clim_mu,
        "temp_now_minus_clim": temp_minus_clim,
        "days_since_solstice": float(days_since_sol),
        "tmax_yesterday": tmax_yesterday,
        "tmax_7d_ago": tmax_7d_ago,
        "tmax_same_date_last_year": tmax_last_year,
        # v2.0
        "morning_heating_rate": morning_heating_rate,
        "tmax_dewpoint_proxy": tmax_dewpoint_proxy,
        "dew_point_depression": dew_depression,
        "clear_sky_ghi_remaining": clear_sky_ghi_remaining_val,
        "insolation_efficiency": insolation_efficiency,
        "effective_warming_potential": effective_warming_potential,
        "pressure_anomaly": pressure_anomaly,
        "dew_anomaly": dew_anomaly,
        "cloudcover_anomaly": cloudcover_anomaly,
        "windspeed_anomaly": windspeed_anomaly,
        "inversion_suspect": inversion_suspect,
        "city_elevation_m": city_elevation_m,
        "city_lat": city_lat_val,
        "city_hemisphere": city_hemisphere,
        "city_coastal": city_coastal,
        "city_koppen_A": k_a,
        "city_koppen_B": k_b,
        "city_koppen_C": k_c,
        "city_koppen_D": k_d,
    }

    vec = np.full(N_FEATURES, _NAN, dtype=np.float64)
    for i, name in enumerate(FEATURE_NAMES):
        v = named.get(name)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        vec[i] = fv
    return vec


def feature_vector_as_dict(vec: np.ndarray) -> dict[str, float]:
    if len(vec) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {len(vec)}")
    return dict(zip(FEATURE_NAMES, vec.tolist()))
