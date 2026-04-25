"""
schema.py — canonical feature ordering for the ML distribution model.

Single source of truth for which features exist and the order they appear
in the feature vector.  Anything that serializes or deserializes feature
vectors (training rows, model files, inference payloads) MUST use
FEATURE_NAMES in this exact order.

Versioning rule:
  * v1.0  — 27 features, observation-only baseline (10:00, 12:00 decisions)
  * v2.0  — 46 features, adds derived weather features, climatology
            anomalies, inversion flag, and static city features.  Also
            adds a 14:00 decision moment in the backfill (handled by the
            backfill script, not the feature builder).

Bumping FEATURE_VERSION invalidates all trained models — they were fit on
the old layout.  Appending features at the END of FEATURE_NAMES is the
migration-friendly way to extend; reordering or inserting in the middle is
a hard break.
"""

FEATURE_VERSION = "v2.0"

FEATURE_NAMES: list[str] = [
    # ====================================================================
    # === v1.0 features (27) — preserved in original order ===============
    # ====================================================================

    # --- Temporal (4) ---
    "doy_sin",                    # sin(2π × day-of-year / 365.25)
    "doy_cos",                    # cos(2π × day-of-year / 365.25)
    "hour_sin",                   # sin(2π × local hour / 24)
    "hour_cos",                   # cos(2π × local hour / 24)

    # --- Current observations at decision moment (11) ---
    "temp_now_c",
    "humidity",
    "dew_c",
    "pressure_hpa",
    "cloudcover",
    "visibility_km",
    "windspeed_kph",
    "winddir_sin",
    "winddir_cos",
    "solarradiation_wm2",
    "snowdepth_cm",

    # --- Recent trajectory (6) ---
    "temp_change_3h",
    "temp_change_6h",
    "temp_change_24h",
    "pressure_change_3h",
    "pressure_change_24h",
    "cloudcover_mean_6h",

    # --- Climatology-relative (3) ---
    "climatology_mu_today",       # T_max climatology for this (city, doy)
    "temp_now_minus_clim",
    "days_since_solstice",

    # --- Year-over-year T_max lags (3) ---
    "tmax_yesterday",
    "tmax_7d_ago",
    "tmax_same_date_last_year",

    # ====================================================================
    # === v2.0 additions (19) ============================================
    # ====================================================================

    # --- Derived weather features (6) ---
    "morning_heating_rate",       # (temp_now - temp_at_sunrise) / hours_since_sunrise
    "tmax_dewpoint_proxy",        # temp_now + 0.6*(temp_now - dew_now)  [NWS Tdd rule]
    "dew_point_depression",       # temp_now - dew_now
    "clear_sky_ghi_remaining",    # integrated clear-sky GHI from decision to 16:00 local (kWh/m²)
    "insolation_efficiency",      # observed_solar_now / theoretical_clear_sky_now (0..1)
    "effective_warming_potential", # (1-cloud) * dew_depression * cos(noon_zenith)

    # --- Climatology anomaly z-scores (4) ---
    "pressure_anomaly",           # (pressure_now - mu) / sigma  for (city, doy, hour)
    "dew_anomaly",
    "cloudcover_anomaly",
    "windspeed_anomaly",

    # --- Boundary-layer flags (1) ---
    "inversion_suspect",          # 1 if hours_since_sunrise>2 AND temp_change_3h<0.5
                                  #         AND cloudcover<30 AND windspeed<5, else 0

    # --- Static city features (8) ---
    # Constant per-city; no value for per-city models but essential for the
    # pooled multi-city model in Phase 9.  Per-city models will simply not
    # split on them.
    "city_elevation_m",
    "city_lat",
    "city_hemisphere",            # 1 if N, -1 if S
    "city_coastal",               # 1 if within ~20km of ocean/major sea, else 0
    "city_koppen_A",              # one-hot Köppen main letter
    "city_koppen_B",
    "city_koppen_C",
    "city_koppen_D",
]

N_FEATURES: int = len(FEATURE_NAMES)
