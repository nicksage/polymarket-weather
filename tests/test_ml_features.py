"""Smoke tests for bot.ml.features.build_feature_vector."""

import math
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pytest

# Make bot/ importable (no conftest.py in repo, so tests set up their own path).
_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from ml.features import build_feature_vector, feature_vector_as_dict
from ml.schema import FEATURE_NAMES, N_FEATURES, FEATURE_VERSION


def _fresh_ctx():
    """A fully-populated context dict for a summer afternoon in Chicago."""
    return {
        "target_date":       date(2025, 7, 15),
        "decision_dt_utc":   datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc),  # 10am local CDT
        "decision_dt_local": datetime(2025, 7, 15, 10, 0),
        "current_obs": {
            "temp_c":             22.5,
            "humidity":           65.0,
            "dew_c":              15.7,
            "pressure_hpa":       1013.0,
            "cloudcover":         40.0,
            "visibility_km":      16.0,
            "windspeed_kph":      10.0,
            "winddir_deg":        180.0,
            "solarradiation_wm2": 500.0,
            "snowdepth_cm":       0.0,
        },
        "recent_obs": [
            # 3h prior
            {"observed_at_utc": "2025-07-15T12:00:00+00:00",
             "temp_c": 18.0, "pressure_hpa": 1015.0, "cloudcover": 60.0},
            # 6h prior
            {"observed_at_utc": "2025-07-15T09:00:00+00:00",
             "temp_c": 15.5, "pressure_hpa": 1016.0, "cloudcover": 70.0},
            # 24h prior
            {"observed_at_utc": "2025-07-14T15:00:00+00:00",
             "temp_c": 21.0, "pressure_hpa": 1014.0, "cloudcover": 30.0},
        ],
        "climatology":     {"mu_today_c": 24.0},
        "historical_tmax": {
            "yesterday_c":      23.0,
            "seven_days_ago_c": 20.0,
            "last_year_c":      25.0,
        },
    }


def test_feature_vector_shape_and_dtype():
    vec = build_feature_vector(_fresh_ctx())
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (N_FEATURES,)
    assert vec.dtype == np.float64


def test_feature_vector_deterministic():
    v1 = build_feature_vector(_fresh_ctx())
    v2 = build_feature_vector(_fresh_ctx())
    np.testing.assert_array_equal(v1, v2)


def test_feature_vector_named_values():
    vec = build_feature_vector(_fresh_ctx())
    d = feature_vector_as_dict(vec)

    # Raw passthroughs
    assert d["temp_now_c"]          == 22.5
    assert d["humidity"]            == 65.0
    assert d["dew_c"]               == 15.7
    assert d["pressure_hpa"]        == 1013.0
    assert d["cloudcover"]          == 40.0
    assert d["visibility_km"]       == 16.0
    assert d["windspeed_kph"]       == 10.0
    assert d["solarradiation_wm2"] == 500.0
    assert d["climatology_mu_today"] == 24.0

    # Computed deltas
    assert d["temp_change_3h"]       == pytest.approx(22.5 - 18.0)
    assert d["temp_change_6h"]       == pytest.approx(22.5 - 15.5)
    assert d["temp_change_24h"]      == pytest.approx(22.5 - 21.0)
    assert d["pressure_change_3h"]   == pytest.approx(1013.0 - 1015.0)
    assert d["pressure_change_24h"]  == pytest.approx(1013.0 - 1014.0)
    assert d["temp_now_minus_clim"] == pytest.approx(22.5 - 24.0)

    # Cloud cover mean over last 6h includes the 3h and 6h points (not 24h)
    assert d["cloudcover_mean_6h"]   == pytest.approx((60.0 + 70.0) / 2)

    # Year-over-year
    assert d["tmax_yesterday"]           == 23.0
    assert d["tmax_7d_ago"]              == 20.0
    assert d["tmax_same_date_last_year"] == 25.0

    # Cyclical encodings: sin² + cos² == 1
    assert d["doy_sin"]**2 + d["doy_cos"]**2   == pytest.approx(1.0)
    assert d["hour_sin"]**2 + d["hour_cos"]**2 == pytest.approx(1.0)
    assert d["winddir_sin"]**2 + d["winddir_cos"]**2 == pytest.approx(1.0)

    # Hour 10 AM => sin(2π·10/24) ≈ sin(2.618) ≈ 0.5
    assert d["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 10 / 24))

    # Solstice: July 15 is 24 days after June 21
    assert d["days_since_solstice"] == 24.0


def test_missing_inputs_yield_nan():
    ctx = _fresh_ctx()
    ctx["current_obs"]     = {}
    ctx["recent_obs"]      = []
    ctx["climatology"]     = None
    ctx["historical_tmax"] = None

    d = feature_vector_as_dict(build_feature_vector(ctx))

    # Temporal features are always computable from the date itself
    assert not math.isnan(d["doy_sin"])
    assert not math.isnan(d["doy_cos"])
    assert not math.isnan(d["hour_sin"])
    assert not math.isnan(d["days_since_solstice"])

    # Everything observation-derived should be NaN
    for k in (
        "temp_now_c", "humidity", "dew_c", "pressure_hpa", "cloudcover",
        "visibility_km", "windspeed_kph", "winddir_sin", "winddir_cos",
        "solarradiation_wm2", "snowdepth_cm",
        "temp_change_3h", "temp_change_6h", "temp_change_24h",
        "pressure_change_3h", "pressure_change_24h", "cloudcover_mean_6h",
        "climatology_mu_today", "temp_now_minus_clim",
        "tmax_yesterday", "tmax_7d_ago", "tmax_same_date_last_year",
    ):
        assert math.isnan(d[k]), f"{k} should be NaN when input missing"


def test_partial_recent_obs_yields_partial_trajectory():
    """Only the 3h lookback is present — 6h and 24h deltas should go NaN."""
    ctx = _fresh_ctx()
    ctx["recent_obs"] = [ctx["recent_obs"][0]]   # keep only the 3h-ago row
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert not math.isnan(d["temp_change_3h"])
    assert math.isnan(d["temp_change_6h"])
    assert math.isnan(d["temp_change_24h"])


def test_lookback_window_ignores_stale_obs():
    """Obs more than 30 min from the lookback target do not match."""
    ctx = _fresh_ctx()
    # Move the "3h ago" obs 2h off target — should no longer match
    ctx["recent_obs"] = [
        {"observed_at_utc": "2025-07-15T10:00:00+00:00",   # 5h ago, not 3h
         "temp_c": 18.0, "pressure_hpa": 1015.0, "cloudcover": 60.0},
    ]
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert math.isnan(d["temp_change_3h"])
    assert math.isnan(d["temp_change_6h"])    # also not within ±30min of 6h


def test_feature_names_unique_and_sized():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert N_FEATURES == len(FEATURE_NAMES)
    assert N_FEATURES == 46
    assert FEATURE_VERSION == "v2.0"


def test_winddir_cyclical_encoding():
    """North (0°) and South (180°) should have different (sin, cos) encodings."""
    ctx_n = _fresh_ctx()
    ctx_n["current_obs"] = {**ctx_n["current_obs"], "winddir_deg": 0.0}
    d_n = feature_vector_as_dict(build_feature_vector(ctx_n))

    ctx_s = _fresh_ctx()
    ctx_s["current_obs"] = {**ctx_s["current_obs"], "winddir_deg": 180.0}
    d_s = feature_vector_as_dict(build_feature_vector(ctx_s))

    assert d_n["winddir_sin"] == pytest.approx(0.0, abs=1e-9)
    assert d_n["winddir_cos"] == pytest.approx(1.0)
    assert d_s["winddir_sin"] == pytest.approx(0.0, abs=1e-9)
    assert d_s["winddir_cos"] == pytest.approx(-1.0)


# =============================================================================
# === v2.0 feature tests =====================================================
# =============================================================================

def _fresh_ctx_v2():
    """Same as _fresh_ctx() but with v2.0 ctx fields populated.  Adds an
    early-morning observation so morning_heating_rate is computable."""
    ctx = _fresh_ctx()
    # Add an early-morning obs (06:00 UTC = 01:00 local CDT)
    ctx["recent_obs"].append({
        "observed_at_utc": "2025-07-15T07:00:00+00:00",
        "temp_c": 14.5, "pressure_hpa": 1016.5, "cloudcover": 65.0,
    })
    ctx["city_name"]       = "Chicago"
    ctx["lat"]             = 41.85
    ctx["lon"]             = -87.65
    ctx["tz_offset_hours"] = -5.0   # CDT
    # Hourly climatology lookup for (doy=196, hour=10)  [July 15]
    ctx["hourly_climatology"] = {
        (196, 10): {
            "n_samples":         12,
            "temp_mu":           20.0, "temp_sigma":           2.5,
            "dew_mu":            14.0, "dew_sigma":            2.0,
            "pressure_mu":       1014.0, "pressure_sigma":     5.0,
            "cloudcover_mu":     50.0, "cloudcover_sigma":    25.0,
            "windspeed_mu":      9.0,  "windspeed_sigma":      3.0,
            "solarradiation_mu": 450.0, "solarradiation_sigma": 100.0,
        },
    }
    return ctx


def test_v2_dew_point_depression_and_proxy():
    d = feature_vector_as_dict(build_feature_vector(_fresh_ctx_v2()))
    # temp_now=22.5, dew=15.7 → dew_depression=6.8
    assert d["dew_point_depression"] == pytest.approx(22.5 - 15.7)
    # tmax_dewpoint_proxy = temp_now + 0.6 * dew_depression
    assert d["tmax_dewpoint_proxy"] == pytest.approx(22.5 + 0.6 * (22.5 - 15.7))


def test_v2_morning_heating_rate():
    d = feature_vector_as_dict(build_feature_vector(_fresh_ctx_v2()))
    # Earliest obs today: 07:00 UTC, temp_c=14.5
    # Decision: 15:00 UTC, temp_c=22.5 → 8h elapsed
    # rate = (22.5 - 14.5) / 8 = 1.0 °C/hr
    assert d["morning_heating_rate"] == pytest.approx(1.0)


def test_v2_anomalies_when_climatology_present():
    d = feature_vector_as_dict(build_feature_vector(_fresh_ctx_v2()))
    # pressure_anomaly = (1013 - 1014) / 5 = -0.2
    assert d["pressure_anomaly"] == pytest.approx((1013.0 - 1014.0) / 5.0)
    # dew_anomaly = (15.7 - 14.0) / 2.0 = 0.85
    assert d["dew_anomaly"] == pytest.approx((15.7 - 14.0) / 2.0)
    # cloudcover_anomaly = (40 - 50) / 25 = -0.4
    assert d["cloudcover_anomaly"] == pytest.approx((40.0 - 50.0) / 25.0)


def test_v2_anomalies_nan_when_climatology_missing():
    ctx = _fresh_ctx_v2()
    ctx["hourly_climatology"] = {}        # no lookup available
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert math.isnan(d["pressure_anomaly"])
    assert math.isnan(d["dew_anomaly"])
    assert math.isnan(d["cloudcover_anomaly"])
    assert math.isnan(d["windspeed_anomaly"])


def test_v2_anomalies_nan_when_climatology_too_thin():
    ctx = _fresh_ctx_v2()
    # Same lookup but with only 2 samples (below _MIN_CLIM_SAMPLES=3)
    ctx["hourly_climatology"] = {
        (196, 10): {**ctx["hourly_climatology"][(196, 10)], "n_samples": 2}
    }
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert math.isnan(d["pressure_anomaly"])


def test_v2_static_city_features():
    d = feature_vector_as_dict(build_feature_vector(_fresh_ctx_v2()))
    # Chicago: elevation 181m, Cfa? actually Dfa, hemisphere 1, coastal 0
    assert d["city_elevation_m"] == pytest.approx(181.0)
    assert d["city_hemisphere"] == 1.0
    assert d["city_coastal"] == 0.0
    # Köppen: D (continental humid)
    assert d["city_koppen_D"] == 1.0
    assert d["city_koppen_A"] == 0.0
    assert d["city_koppen_B"] == 0.0
    assert d["city_koppen_C"] == 0.0


def test_v2_static_city_features_unknown_city():
    """Unknown city should fall back to zeros, never raise."""
    ctx = _fresh_ctx_v2()
    ctx["city_name"] = "Atlantis"
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert d["city_elevation_m"] == 0.0
    assert d["city_koppen_A"] == 0.0
    assert d["city_koppen_D"] == 0.0


def test_v2_solar_features_present():
    """Clear-sky GHI features should be populated when lat/lon/tz given."""
    d = feature_vector_as_dict(build_feature_vector(_fresh_ctx_v2()))
    # July 15 Chicago at 10am: plenty of remaining sun until 16:00
    assert not math.isnan(d["clear_sky_ghi_remaining"])
    assert d["clear_sky_ghi_remaining"] > 0
    # Insolation efficiency = observed / theoretical; theoretical at 10am
    # CDT in summer is large (~700 W/m²); observed=500 → ratio ~0.5-0.8
    assert not math.isnan(d["insolation_efficiency"])
    assert 0.3 < d["insolation_efficiency"] < 1.5
    # Effective warming potential should be positive on a partly-cloudy day
    assert not math.isnan(d["effective_warming_potential"])
    assert d["effective_warming_potential"] > 0


def test_v2_solar_features_nan_when_no_geo():
    """No lat/lon/tz → solar features go NaN gracefully."""
    ctx = _fresh_ctx_v2()
    ctx["lat"] = None; ctx["lon"] = None; ctx["tz_offset_hours"] = None
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert math.isnan(d["clear_sky_ghi_remaining"])
    assert math.isnan(d["insolation_efficiency"])
    assert math.isnan(d["effective_warming_potential"])


def test_v2_inversion_suspect_flag():
    """Mid-morning + clear + calm + slow warming → flag fires."""
    ctx = _fresh_ctx_v2()
    # Adjust for the inversion conditions
    ctx["current_obs"] = {**ctx["current_obs"],
                          "cloudcover": 10.0, "windspeed_kph": 2.0}
    # Slow warming: 3h-prior temp very close to now
    ctx["recent_obs"] = [
        {"observed_at_utc": "2025-07-15T12:00:00+00:00",
         "temp_c": 22.2, "pressure_hpa": 1015.0, "cloudcover": 10.0},
    ]
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert d["inversion_suspect"] == 1.0


def test_v2_inversion_suspect_off_when_warming():
    ctx = _fresh_ctx_v2()
    ctx["current_obs"] = {**ctx["current_obs"],
                          "cloudcover": 10.0, "windspeed_kph": 2.0}
    # Strong warming over 3h
    ctx["recent_obs"] = [
        {"observed_at_utc": "2025-07-15T12:00:00+00:00",
         "temp_c": 16.0, "pressure_hpa": 1015.0, "cloudcover": 10.0},
    ]
    d = feature_vector_as_dict(build_feature_vector(ctx))
    assert d["inversion_suspect"] == 0.0
