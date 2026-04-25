"""
inference.py — Live inference hook for the ML T_max distribution model.

Loaded once per process (module-level cache); returns (μ, σ) for an event's
target date given current live-observation context.  Wired into weather.py
as a fourth ensemble source alongside ECMWF / GFS / climatology.

Model precedence:
  1. PooledQuantileDistributionModel  (v2.0+, single file: pooled_v2.0.joblib)
  2. Per-city TempDistributionModel    (v1.0 legacy, one file per city)

The pooled model is preferred when present.  Both produce the same
{mu_c, sigma_c, model_version} result shape so downstream code is
unchanged.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import DB_PATH, ML_MODELS_DIR
from ml.features import build_feature_vector
from ml.schema import FEATURE_VERSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# Pooled model: None means "tried, not present"; missing key means "not yet checked"
_POOLED_MODEL: Any = "__unset__"

# city -> per-city legacy model | None.  None marks "checked, not present."
_LEGACY_MODEL_CACHE: dict[str, Any] = {}

# city -> {(doy, hour) -> climatology row}, cached per process for cheap lookups
_HOURLY_CLIM_CACHE: dict[str, dict[tuple[int, int], dict]] = {}

# (lat, lon) -> ZoneInfo
_TZ_CACHE: dict[tuple[float, float], Any] = {}


def clear_model_cache() -> None:
    """Test hook — drop every loaded model + climatology + tz cache."""
    global _POOLED_MODEL
    _POOLED_MODEL = "__unset__"
    _LEGACY_MODEL_CACHE.clear()
    _HOURLY_CLIM_CACHE.clear()
    _TZ_CACHE.clear()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_pooled_model():
    """Lazy-load the single pooled model.  Returns the model instance or
    None if not present at the expected path."""
    global _POOLED_MODEL
    if _POOLED_MODEL != "__unset__":
        return _POOLED_MODEL
    path = Path(ML_MODELS_DIR) / f"pooled_{FEATURE_VERSION}.joblib"
    if not path.exists():
        logger.debug(f"ml.inference: no pooled model at {path}")
        _POOLED_MODEL = None
        return None
    try:
        # Lazy import — only pay the joblib load cost when actually used
        from ml.pooled_distribution_model import PooledQuantileDistributionModel
        model = PooledQuantileDistributionModel.load(path)
        logger.info(f"ml.inference: loaded pooled model ({model.version}) "
                    f"with {len(model.city_to_idx)} cities")
        _POOLED_MODEL = model
        return model
    except Exception as e:
        logger.warning(f"ml.inference: pooled model load failed: {e}")
        _POOLED_MODEL = None
        return None


def _load_legacy_model_for_city(city: str):
    if city in _LEGACY_MODEL_CACHE:
        return _LEGACY_MODEL_CACHE[city]
    path = Path(ML_MODELS_DIR) / f"{city}_v1.0.joblib"
    if not path.exists():
        _LEGACY_MODEL_CACHE[city] = None
        return None
    try:
        from ml.distribution_model import TempDistributionModel
        model = TempDistributionModel.load(path)
        logger.info(f"ml.inference: loaded legacy model for {city} ({model.version})")
        _LEGACY_MODEL_CACHE[city] = model
        return model
    except Exception as e:
        logger.warning(f"ml.inference: legacy model load for {city} failed: {e}")
        _LEGACY_MODEL_CACHE[city] = None
        return None


# ---------------------------------------------------------------------------
# Live-state → feature-builder ctx
# ---------------------------------------------------------------------------

def _current_obs_from_live_row(row: dict | None) -> dict:
    if row is None:
        return {}
    return {
        "temp_c":             row.get("current_temp_c"),
        "humidity":           row.get("humidity"),
        "dew_c":              row.get("dew_c"),
        "pressure_hpa":       row.get("pressure_hpa"),
        "cloudcover":         row.get("cloudcover"),
        "visibility_km":      row.get("visibility_km"),
        "windspeed_kph":      row.get("windspeed_kph"),
        "winddir_deg":        row.get("winddir_deg"),
        "solarradiation_wm2": row.get("solarradiation_wm2"),
        "snowdepth_cm":       row.get("snowdepth_cm"),
    }


def _recent_obs_from_live_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        out.append({
            "observed_at_utc": r.get("pulled_at_utc"),
            "temp_c":          r.get("current_temp_c"),
            "pressure_hpa":    r.get("pressure_hpa"),
            "cloudcover":      r.get("cloudcover"),
        })
    return out


def _lookup_historical_tmax(city: str, target_date: date) -> dict:
    """Best-effort D-1, D-7, D-365 T_max lookups."""
    out = {"yesterday_c": None, "seven_days_ago_c": None, "last_year_c": None}
    offsets = {"yesterday_c": 1, "seven_days_ago_c": 7, "last_year_c": 365}
    conn = sqlite3.connect(DB_PATH)
    try:
        for key, n in offsets.items():
            d = (target_date - timedelta(days=n)).isoformat()
            row = conn.execute(
                "SELECT t_max_c FROM ml_training_rows "
                "WHERE city = ? AND target_date = ? LIMIT 1",
                (city, d),
            ).fetchone()
            if row and row[0] is not None:
                out[key] = float(row[0])
                continue
            row = conn.execute(
                "SELECT tempmax_c FROM historical_observed_daily "
                "WHERE city = ? AND date = ? LIMIT 1",
                (city, d),
            ).fetchone()
            if row and row[0] is not None:
                out[key] = float(row[0])
    finally:
        conn.close()
    return out


def _lookup_climatology_today_c(city: str, target_date: date) -> float | None:
    """T_max climatology for (city, doy) from obs_climatology_daily."""
    doy = target_date.timetuple().tm_yday
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT tmax_mu FROM obs_climatology_daily WHERE city = ? AND doy = ?",
            (city, doy),
        ).fetchone()
    finally:
        conn.close()
    return float(row[0]) if (row and row[0] is not None) else None


def _hourly_clim_for_city(city: str) -> dict[tuple[int, int], dict]:
    """Lazy load + cache the (doy, hour) -> climatology row map for a city.
    Avoids 8784 DB lookups per scan when reading a single (doy, hour)."""
    cached = _HOURLY_CLIM_CACHE.get(city)
    if cached is not None:
        return cached
    from db import load_all_obs_climatology_hourly
    out = load_all_obs_climatology_hourly(city)
    _HOURLY_CLIM_CACHE[city] = out
    return out


def _resolve_tz(lat: float, lon: float):
    from zoneinfo import ZoneInfo
    key = (round(lat, 3), round(lon, 3))
    if key in _TZ_CACHE:
        return _TZ_CACHE[key]
    try:
        from timezonefinder import TimezoneFinder
        name = TimezoneFinder().timezone_at(lat=lat, lng=lon)
        tz = ZoneInfo(name) if name else timezone.utc
    except Exception:
        tz = timezone.utc
    _TZ_CACHE[key] = tz
    return tz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ML_INFERENCE_HOUR_MIN = 9.0    # local hour — earliest the model is trusted
ML_INFERENCE_HOUR_MAX = 17.0   # local hour — latest the model is trusted
TRAINED_FOLDS = (10, 12, 14)   # decision moments the v2.0 model was trained on


def _closest_fold(local_hour: float) -> int:
    """Return the trained decision-moment fold (10, 12, or 14) nearest to
    `local_hour`.  Used as a diagnostic in shadow logs — the model itself
    is called at the actual current hour via cyclical features."""
    return min(TRAINED_FOLDS, key=lambda f: abs(f - local_hour))


def get_ml_distribution(
    city: str | None,
    lat: float,
    lon: float,
    target_date: date,
    event_id: str | None,
    decision_dt_utc: datetime | None = None,
) -> dict | None:
    """Return an ML (μ, σ) prediction for the event's target date, or None
    if no prediction can be produced.  Never raises.

    Gates (return None silently before doing any work):
      * target_date must equal today — model trained on same-day morning
        context; D+1+ is out of distribution.
      * Local hour must be within [ML_INFERENCE_HOUR_MIN, ML_INFERENCE_HOUR_MAX]
        — keeps the cyclical hour features within ~3h of the trained folds
        (10, 12, 14 local).  Outside that window the model extrapolates.

    Note: the model is called at the actual current local hour (cyclical
    features handle interpolation between trained folds).  The closest
    trained fold is logged as `closest_fold` purely as a diagnostic.

    Returns:
        {mu_c, sigma_c, model_version, n_recent_obs, has_current_obs,
         model_kind ('pooled'|'legacy'), decision_hour_local, closest_fold}
    """
    # Gate 1 — same-day only
    days_ahead = (target_date - date.today()).days
    if days_ahead != 0:
        return None

    if not city or not event_id:
        return None

    # Pick the best available model
    pooled = _load_pooled_model()
    legacy = None
    model_kind: str
    if pooled is not None:
        model_kind = "pooled"
    else:
        legacy = _load_legacy_model_for_city(city)
        if legacy is None:
            return None
        model_kind = "legacy"

    from db import get_latest_observation, get_recent_observations

    if decision_dt_utc is None:
        decision_dt_utc = datetime.now(timezone.utc)
    tz = _resolve_tz(lat, lon)
    decision_dt_local_aware = decision_dt_utc.astimezone(tz)
    decision_dt_local = decision_dt_local_aware.replace(tzinfo=None)
    tz_offset_hours = decision_dt_local_aware.utcoffset().total_seconds() / 3600.0 \
        if decision_dt_local_aware.utcoffset() is not None else 0.0

    # Gate 2 — local-hour window guard
    local_hour = decision_dt_local.hour + decision_dt_local.minute / 60.0
    if not (ML_INFERENCE_HOUR_MIN <= local_hour <= ML_INFERENCE_HOUR_MAX):
        logger.debug(
            f"ml.inference: {city} local_hour={local_hour:.1f} outside "
            f"[{ML_INFERENCE_HOUR_MIN}, {ML_INFERENCE_HOUR_MAX}]; skip"
        )
        return None
    closest_fold = _closest_fold(local_hour)

    latest = get_latest_observation(event_id)
    if latest is None:
        logger.debug(f"ml.inference: {city} {event_id} — no live obs yet; skip")
        return None

    current_obs = _current_obs_from_live_row(latest)

    lookback_since = (decision_dt_utc - timedelta(hours=25)).isoformat()
    recent_rows = get_recent_observations(event_id, lookback_since)
    recent_obs = _recent_obs_from_live_rows(recent_rows)

    historical_tmax = _lookup_historical_tmax(city, target_date)
    clim_today = _lookup_climatology_today_c(city, target_date)
    hourly_clim = _hourly_clim_for_city(city)

    ctx = {
        # v1.0 fields
        "target_date":       target_date,
        "decision_dt_utc":   decision_dt_utc,
        "decision_dt_local": decision_dt_local,
        "current_obs":       current_obs,
        "recent_obs":        recent_obs,
        "climatology":       {"mu_today_c": clim_today},
        "historical_tmax":   historical_tmax,
        # v2.0 fields
        "city_name":          city,
        "lat":                lat,
        "lon":                lon,
        "tz_offset_hours":    tz_offset_hours,
        "hourly_climatology": hourly_clim,
    }

    try:
        vec = build_feature_vector(ctx)
        if model_kind == "pooled":
            mu, sigma = pooled.predict(vec, city)
            version = pooled.version or FEATURE_VERSION
        else:
            mu, sigma = legacy.predict(vec, target_date)
            version = legacy.version or "v1.0"
    except Exception as e:
        logger.warning(f"ml.inference: predict failed for {city} ({model_kind}): {e}")
        return None

    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        logger.warning(f"ml.inference: non-finite output for {city}: mu={mu} sigma={sigma}")
        return None

    return {
        "mu_c":                float(mu),
        "sigma_c":             float(sigma),
        "model_version":       version,
        "n_recent_obs":        len(recent_obs),
        "has_current_obs":     True,
        "model_kind":          model_kind,
        "decision_hour_local": int(decision_dt_local.hour),
        "closest_fold":        closest_fold,
    }


def get_ml_bin_probabilities(
    city: str | None,
    lat: float,
    lon: float,
    target_date: date,
    event_id: str | None,
    bins: list[tuple[float | None, float | None]],
    decision_dt_utc: datetime | None = None,
) -> dict | None:
    """Empirical-CDF bin probabilities — pooled model only.

    Returns a dict:
        {
            "probabilities":       list[float],   # one per bin, normalized to sum to 1
            "model_version":       str,
            "decision_hour_local": int,
            "closest_fold":        int,
            "model_kind":          'pooled',
        }
    Or None if the pooled model isn't loaded or any gate fails.

    Each bin is (range_low_c, range_high_c).  Use None for open-ended
    bounds (e.g., (None, 0.0) is "anything below 0°C").
    """
    pooled = _load_pooled_model()
    if pooled is None:
        return None
    # Reuse the dist call to apply gates + build ctx + feature vector
    dist = get_ml_distribution(city, lat, lon, target_date, event_id, decision_dt_utc)
    if dist is None or dist.get("model_kind") != "pooled":
        return None
    # Rebuild the feature vector so we can call bin_probability.  This
    # duplicates work but keeps get_ml_distribution's contract clean.
    # Cheap (~ms).
    from db import get_latest_observation, get_recent_observations
    if decision_dt_utc is None:
        decision_dt_utc = datetime.now(timezone.utc)
    tz = _resolve_tz(lat, lon)
    decision_dt_local_aware = decision_dt_utc.astimezone(tz)
    tz_offset_hours = decision_dt_local_aware.utcoffset().total_seconds() / 3600.0 \
        if decision_dt_local_aware.utcoffset() is not None else 0.0
    ctx = {
        "target_date":        target_date,
        "decision_dt_utc":    decision_dt_utc,
        "decision_dt_local":  decision_dt_local_aware.replace(tzinfo=None),
        "current_obs":        _current_obs_from_live_row(get_latest_observation(event_id)),
        "recent_obs":         _recent_obs_from_live_rows(
            get_recent_observations(event_id,
                (decision_dt_utc - timedelta(hours=25)).isoformat())
        ),
        "climatology":        {"mu_today_c": _lookup_climatology_today_c(city, target_date)},
        "historical_tmax":    _lookup_historical_tmax(city, target_date),
        "city_name":          city,
        "lat":                lat,
        "lon":                lon,
        "tz_offset_hours":    tz_offset_hours,
        "hourly_climatology": _hourly_clim_for_city(city),
    }
    vec = build_feature_vector(ctx)

    raw = [pooled.bin_probability(vec, city, lo, hi) for (lo, hi) in bins]
    total = sum(raw)
    probs = [p / total for p in raw] if total > 0 else raw
    return {
        "probabilities":       probs,
        "model_version":       dist.get("model_version"),
        "decision_hour_local": dist.get("decision_hour_local"),
        "closest_fold":        dist.get("closest_fold"),
        "model_kind":          "pooled",
    }
