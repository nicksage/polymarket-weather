import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone

# Single source of truth — always resolves to bot/data/signals.db (absolute).
from config import DB_PATH


def _set_pragmas(conn: sqlite3.Connection) -> None:
    """Apply concurrency/durability pragmas to a fresh connection.

    journal_mode=WAL — sticks per database file once set; allows
    concurrent readers + one writer instead of the default rollback
    journal where readers block writers and vice versa.  Critical for
    a multi-process setup (bot writer + dashboard reader + scheduler
    threads all touching the same DB).

    busy_timeout=30000 — if the DB is locked at write time, wait up to
    30 seconds instead of immediately erroring.  WAL mode makes locked
    states rare (only on schema changes / VACUUM), but the timeout is
    cheap insurance.

    synchronous=NORMAL — safe under WAL (atomic at COMMIT boundary),
    measurably faster than FULL.  The default FULL is overkill for
    our durability needs (we're not running a bank).
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        # Pragmas should never fail, but if they do, fall through with
        # whatever defaults sqlite picked — better than hard-failing
        # every connection.
        pass


@contextmanager
def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    _set_pragmas(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            -- Legacy signals table (retained for backward compatibility)
            CREATE TABLE IF NOT EXISTS signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                contract_id      TEXT    NOT NULL,
                question         TEXT,
                market_p         REAL,
                model_p          REAL,
                ev               REAL,
                recommended_side TEXT,
                kelly_size       REAL,
                executed         INTEGER DEFAULT 0,
                outcome          TEXT,
                pnl              REAL
            );

            -- Positions table (used by execution.py for live/paper trades)
            CREATE TABLE IF NOT EXISTS positions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id  TEXT,
                side         TEXT,
                size_usdc    REAL,
                entry_price  REAL,
                entry_time   TEXT,
                status       TEXT    DEFAULT 'open',
                exit_price   REAL,
                exit_time    TEXT,
                pnl          REAL
            );

            -- One row per highest-temperature event (city + date) per scan.
            -- Stores the forecast distribution parameters derived from the
            -- ensemble blend so the dashboard can display them without re-fetching.
            CREATE TABLE IF NOT EXISTS temp_events (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_timestamp      TEXT    NOT NULL,
                event_id            TEXT    NOT NULL,
                event_title         TEXT,
                city                TEXT,
                date                TEXT,
                lat                 REAL,
                lon                 REAL,
                forecast_mu_c       REAL,
                forecast_sigma_c    REAL,
                clim_mu_c           REAL,
                clim_sigma_c        REAL,
                forecast_mu_display REAL,    -- in display unit (C or F)
                display_unit        TEXT,    -- 'celsius' or 'fahrenheit'
                days_ahead          INTEGER,
                market_overround    REAL,
                model_probs_sum     REAL,
                normalization_warning INTEGER DEFAULT 0,
                n_outcomes          INTEGER,
                n_sources           INTEGER
            );

            -- One row per temperature-range outcome within an event.
            -- Linked to temp_events via event_row_id.
            CREATE TABLE IF NOT EXISTS temp_outcomes (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                event_row_id     INTEGER REFERENCES temp_events(id),
                scan_timestamp   TEXT    NOT NULL,
                contract_id      TEXT    NOT NULL,
                question         TEXT,
                range_low        REAL,
                range_high       REAL,
                unit             TEXT,
                market_price     REAL,
                model_prob       REAL,
                raw_model_prob   REAL,
                ev               REAL,
                edge             REAL,
                recommended_side TEXT,
                kelly_size       REAL,
                is_signal        INTEGER DEFAULT 0,
                liquidity_usd    REAL,
                volume_usd       REAL,
                yes_token_id     TEXT,
                no_token_id      TEXT,
                executed         INTEGER DEFAULT 0,
                outcome          TEXT,
                pnl              REAL
            );

            CREATE INDEX IF NOT EXISTS idx_temp_events_scan ON temp_events(scan_timestamp);
            CREATE INDEX IF NOT EXISTS idx_temp_outcomes_event ON temp_outcomes(event_row_id);
            CREATE INDEX IF NOT EXISTS idx_temp_outcomes_signal ON temp_outcomes(is_signal, scan_timestamp);
        """)

        # Add columns that may not exist in DBs created before this patch.
        # SQLite does not support IF NOT EXISTS on ALTER TABLE, so we use
        # try/except for each column independently.
        for col_def in [
            "ALTER TABLE temp_outcomes ADD COLUMN yes_price REAL",
            "ALTER TABLE temp_outcomes ADD COLUMN no_price  REAL",
            # positions table — expanded for full trade lifecycle tracking
            "ALTER TABLE positions ADD COLUMN order_id        TEXT",
            "ALTER TABLE positions ADD COLUMN is_paper        INTEGER DEFAULT 1",
            "ALTER TABLE positions ADD COLUMN question        TEXT",
            "ALTER TABLE positions ADD COLUMN city            TEXT",
            "ALTER TABLE positions ADD COLUMN date            TEXT",
            "ALTER TABLE positions ADD COLUMN event_id        TEXT",
            "ALTER TABLE positions ADD COLUMN model_prob      REAL",
            "ALTER TABLE positions ADD COLUMN market_prob     REAL",
            "ALTER TABLE positions ADD COLUMN ev              REAL",
            "ALTER TABLE positions ADD COLUMN edge            REAL",
            "ALTER TABLE positions ADD COLUMN fill_status     TEXT DEFAULT 'filled'",
            "ALTER TABLE positions ADD COLUMN shares          REAL",
            "ALTER TABLE positions ADD COLUMN unrealized_pnl  REAL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN current_price   REAL",
            "ALTER TABLE positions ADD COLUMN scan_timestamp   TEXT",
            "ALTER TABLE positions ADD COLUMN cancelled_reason TEXT",
            "ALTER TABLE positions ADD COLUMN gamma_market_id  TEXT",
            "ALTER TABLE positions ADD COLUMN range_low        REAL",
            "ALTER TABLE positions ADD COLUMN range_high       REAL",
            "ALTER TABLE positions ADD COLUMN unit             TEXT",
            "ALTER TABLE positions ADD COLUMN local_time       TEXT",
            "ALTER TABLE positions ADD COLUMN yes_token_id     TEXT",
            "ALTER TABLE positions ADD COLUMN no_token_id      TEXT",
            "ALTER TABLE positions ADD COLUMN lat              REAL",
            "ALTER TABLE positions ADD COLUMN lon              REAL",
            "ALTER TABLE positions ADD COLUMN forecast_sigma_c REAL",
            # forecast_errors table — model column added for per-(city, month, model) bias.
            # Existing rows default to 'ecmwf' since that's the only model previously tracked.
            "ALTER TABLE forecast_errors ADD COLUMN model TEXT DEFAULT 'ecmwf'",
            "ALTER TABLE forecast_errors ADD COLUMN lead_days_bucket INTEGER",
            # --- temp_events: current-state cache columns (Phase 2 VC upgrade) ---
            "ALTER TABLE temp_events ADD COLUMN timezone TEXT",
            "ALTER TABLE temp_events ADD COLUMN latest_forecast_ts TEXT",
            "ALTER TABLE temp_events ADD COLUMN latest_observation_ts TEXT",
            "ALTER TABLE temp_events ADD COLUMN current_temp_c REAL",
            "ALTER TABLE temp_events ADD COLUMN observed_max_so_far_c REAL",
            "ALTER TABLE temp_events ADD COLUMN expected_temp_now_c REAL",
            "ALTER TABLE temp_events ADD COLUMN actual_minus_expected_c REAL",
            "ALTER TABLE temp_events ADD COLUMN forecast_delta_mu_c REAL",
            "ALTER TABLE temp_events ADD COLUMN forecast_delta_sigma_c REAL",
            "ALTER TABLE temp_events ADD COLUMN forecast_agreement_c REAL",
            # --- temp_outcomes: decision-context columns ---
            "ALTER TABLE temp_outcomes ADD COLUMN edge_after_fees REAL",
            "ALTER TABLE temp_outcomes ADD COLUMN edge_after_live_adjustment REAL",
            "ALTER TABLE temp_outcomes ADD COLUMN bin_rank_within_event INTEGER",
            "ALTER TABLE temp_outcomes ADD COLUMN exit_priority_score REAL",
            "ALTER TABLE temp_outcomes ADD COLUMN stale_signal INTEGER DEFAULT 0",
            # --- positions: entry/exit attribution columns ---
            "ALTER TABLE positions ADD COLUMN entry_snapshot_id INTEGER",
            "ALTER TABLE positions ADD COLUMN exit_snapshot_id INTEGER",
            "ALTER TABLE positions ADD COLUMN entry_forecast_delta_mu_c REAL",
            "ALTER TABLE positions ADD COLUMN entry_actual_vs_expected_c REAL",
            "ALTER TABLE positions ADD COLUMN max_favorable_excursion REAL",
            "ALTER TABLE positions ADD COLUMN max_adverse_excursion REAL",
            "ALTER TABLE positions ADD COLUMN exit_reason TEXT",
            # Phase 3: liquidity-aware sizing
            "ALTER TABLE positions ADD COLUMN target_size_usdc REAL",
            "ALTER TABLE positions ADD COLUMN stop_loss_price REAL",
            "ALTER TABLE positions ADD COLUMN peak_price REAL",
            "ALTER TABLE positions ADD COLUMN strategy TEXT",
            "ALTER TABLE positions ADD COLUMN pre_entry_volatility REAL",
            "ALTER TABLE positions ADD COLUMN pre_entry_trend REAL",
            "ALTER TABLE positions ADD COLUMN pre_entry_momentum REAL",
            # --- Phase 2b: live adjustment layer ---
            "ALTER TABLE temp_events ADD COLUMN adjusted_mu_c REAL",
            "ALTER TABLE temp_events ADD COLUMN adjusted_sigma_c REAL",
            "ALTER TABLE temp_events ADD COLUMN live_adjustment_score REAL",
            "ALTER TABLE temp_events ADD COLUMN live_adjustment_components TEXT",
            "ALTER TABLE temp_outcomes ADD COLUMN model_prob_blended REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN adjusted_mu_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN adjusted_sigma_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN live_adjustment_score REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN live_adjustment_components TEXT",
            "ALTER TABLE decision_snapshots ADD COLUMN obs_floor_applied INTEGER DEFAULT 0",
            # --- Phase 2c: VC forecast diagnostic layer ---
            "ALTER TABLE decision_snapshots ADD COLUMN vc_projected_day_max_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN vc_vs_blended_mu_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN vc_vs_adjusted_mu_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN vc_hourly_path_rmse_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN vc_bins_apart INTEGER",
            "ALTER TABLE decision_snapshots ADD COLUMN flag_vc_disagreement_large INTEGER DEFAULT 0",
            "ALTER TABLE decision_snapshots ADD COLUMN flag_vc_warns_hotter INTEGER DEFAULT 0",
            "ALTER TABLE decision_snapshots ADD COLUMN flag_vc_warns_colder INTEGER DEFAULT 0",
            # --- Phase ML-v1: expanded VC observation features for training ---
            "ALTER TABLE live_observations ADD COLUMN feelslike_c REAL",
            "ALTER TABLE live_observations ADD COLUMN dew_c REAL",
            "ALTER TABLE live_observations ADD COLUMN pressure_hpa REAL",
            "ALTER TABLE live_observations ADD COLUMN visibility_km REAL",
            "ALTER TABLE live_observations ADD COLUMN windgust_kph REAL",
            "ALTER TABLE live_observations ADD COLUMN winddir_deg REAL",
            "ALTER TABLE live_observations ADD COLUMN preciptype TEXT",
            "ALTER TABLE live_observations ADD COLUMN snow_cm REAL",
            "ALTER TABLE live_observations ADD COLUMN snowdepth_cm REAL",
            "ALTER TABLE live_observations ADD COLUMN solarradiation_wm2 REAL",
            "ALTER TABLE live_observations ADD COLUMN solarenergy_mj REAL",
            "ALTER TABLE live_observations ADD COLUMN uvindex REAL",
            # --- Phase ML-v1: ML distribution model shadow-log columns ---
            "ALTER TABLE decision_snapshots ADD COLUMN ml_mu_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN ml_sigma_c REAL",
            "ALTER TABLE decision_snapshots ADD COLUMN ml_model_version TEXT",
            "ALTER TABLE decision_snapshots ADD COLUMN ml_weight_used REAL",
            # --- Dashboard ML bin-prob layer (D=0 events): per-outcome
            #     empirical-CDF probability from the pooled v2.0 model. ---
            "ALTER TABLE temp_outcomes ADD COLUMN ml_bin_prob REAL",
            "ALTER TABLE temp_outcomes ADD COLUMN ml_decision_hour INTEGER",
            "ALTER TABLE temp_outcomes ADD COLUMN ml_model_version TEXT",
            # --- Live exit ladder (Live Phase 3): tracks the in-flight sell
            # order for a position whose exit has been triggered but not
            # yet filled.  See bot/exit_ladder.py + bot/execution.execute_exit. ---
            "ALTER TABLE positions ADD COLUMN exit_order_id TEXT",
            "ALTER TABLE positions ADD COLUMN exit_intended_price REAL",
            "ALTER TABLE positions ADD COLUMN actual_exit_price REAL",
            "ALTER TABLE positions ADD COLUMN exit_retry_count INTEGER DEFAULT 0",
            # --- Live top-up (Live Phase 6): tracks an in-flight CLOB buy
            # that's adding to an existing position.  Filled by monitor's
            # reconciliation; merged into the parent via update_position_topup. ---
            "ALTER TABLE positions ADD COLUMN pending_topup_order_id TEXT",
            "ALTER TABLE positions ADD COLUMN pending_topup_amount_usdc REAL",
            "ALTER TABLE positions ADD COLUMN pending_topup_intended_price REAL",
            # --- Fee accounting (Live Phase 7): captured from CLOB fill
            # responses.  entry_fees ACCUMULATES across initial buy + every
            # top-up.  pnl_net = pnl - entry_fees - exit_fees, computed when
            # the exit fill is reconciled. ---
            "ALTER TABLE positions ADD COLUMN entry_fees REAL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN exit_fees REAL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN pnl_net REAL",
            # --- User-channel WS lifecycle tracking (Live Phase 9):
            # Polymarket trades progress MATCHED → MINED → CONFIRMED.
            # We only mark fill_status='filled' on CONFIRMED, since MATCHED
            # can still revert during the mining phase.  These columns track
            # the last-seen lifecycle stage per side. ---
            "ALTER TABLE positions ADD COLUMN trade_status TEXT",
            "ALTER TABLE positions ADD COLUMN exit_trade_status TEXT",
            "ALTER TABLE positions ADD COLUMN last_trade_event_id TEXT",
            "ALTER TABLE positions ADD COLUMN last_exit_trade_event_id TEXT",
            # Cumulative USDC realised across all exit-fill chunks.  Used by
            # add_position_exit_fill to compute weighted-average exit price
            # and final pnl when a multi-chunk exit completes.  Default 0.
            "ALTER TABLE positions ADD COLUMN exit_proceeds_usdc REAL DEFAULT 0",
        ]:
            try:
                conn.execute(col_def)
            except Exception:
                pass   # column already exists

        conn.executescript("""

            -- Persistent ERA5 reanalysis cache.
            -- One row per (location, date). Data older than ~3 months is immutable
            -- so rows are written once and never updated.  The bot checks this table
            -- before calling the Open-Meteo Archive API, eliminating redundant fetches
            -- on every run after the first.
            CREATE TABLE IF NOT EXISTS era5_daily (
                lat_key     REAL NOT NULL,   -- round(lat, 2)
                lon_key     REAL NOT NULL,   -- round(lon, 2)
                date        TEXT NOT NULL,   -- YYYY-MM-DD (UTC)
                tmax_c      REAL NOT NULL,   -- daily maximum 2m temperature (°C)
                fetched_at  TEXT NOT NULL,   -- ISO timestamp of DB insert
                PRIMARY KEY (lat_key, lon_key, date)
            );
            CREATE INDEX IF NOT EXISTS idx_era5_loc_doy
                ON era5_daily(lat_key, lon_key, date);

            -- Forecast error log for ECMWF bias correction.
            -- After each contract resolves, we compare what ECMWF predicted
            -- (forecast_mu_c) to what ERA5 says actually happened (actual_tmax_c).
            -- The mean error over recent observations is subtracted from future
            -- forecasts for the same location + calendar period.
            CREATE TABLE IF NOT EXISTS forecast_errors (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                lat_key             REAL    NOT NULL,
                lon_key             REAL    NOT NULL,
                city                TEXT,
                calendar_month_day  TEXT    NOT NULL,  -- MM-DD
                target_date         TEXT    NOT NULL,  -- YYYY-MM-DD
                days_ahead          INTEGER,
                forecast_mu_c       REAL,
                actual_tmax_c       REAL,
                error_c             REAL,              -- actual - forecast (+ve = forecast was too cold)
                recorded_at         TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_bias_loc_cmd
                ON forecast_errors(lat_key, lon_key, calendar_month_day);

            -- Cache of Visual Crossing daily-max observations, keyed by
            -- coordinate + date.  Populated by the bias backfill script and
            -- the monitor loop.  Source is always station-observed ("obs").
            CREATE TABLE IF NOT EXISTS historical_observed_daily (
                lat_key     REAL NOT NULL,        -- round(lat, 2)
                lon_key     REAL NOT NULL,        -- round(lon, 2)
                city        TEXT,
                date        TEXT NOT NULL,        -- YYYY-MM-DD local
                tempmax_c   REAL,
                tempmin_c   REAL,
                temp_c      REAL,
                stations    TEXT,                  -- JSON list of station IDs
                fetched_at  TEXT NOT NULL,
                PRIMARY KEY (lat_key, lon_key, date)
            );
            CREATE INDEX IF NOT EXISTS idx_hod_city_date
                ON historical_observed_daily(city, date);

            -- Cache of Open-Meteo Previous Runs historical forecasts.
            -- One row per (location, date, model, lead_days) — lets us store
            -- multiple lead times and multiple models for the same day.
            CREATE TABLE IF NOT EXISTS historical_forecasts_previous_runs (
                lat_key             REAL NOT NULL,
                lon_key             REAL NOT NULL,
                city                TEXT,
                date                TEXT NOT NULL,
                model               TEXT NOT NULL,
                lead_days           INTEGER NOT NULL,
                forecast_tempmax_c  REAL,
                n_hours             INTEGER,
                fetched_at          TEXT NOT NULL,
                PRIMARY KEY (lat_key, lon_key, date, model, lead_days)
            );
            CREATE INDEX IF NOT EXISTS idx_hfpr_city_model_date
                ON historical_forecasts_previous_runs(city, model, date);

            -- =============================================================
            -- Phase 2: Time-versioned forecast + observation tables
            -- =============================================================

            -- One row per (event, source, pull).  Raw upstream pulls only —
            -- derived/blended values live in decision_snapshots.
            CREATE TABLE IF NOT EXISTS forecast_runs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id          TEXT    NOT NULL,
                city              TEXT,
                date              TEXT    NOT NULL,       -- target contract date
                lat               REAL,
                lon               REAL,
                source            TEXT    NOT NULL,       -- 'ecmwf' | 'gfs' (future: 'vc')
                pulled_at         TEXT    NOT NULL,       -- ISO UTC
                model_run_ts      TEXT,
                forecast_mu_c     REAL,
                forecast_sigma_c  REAL,
                forecast_high_c   REAL,
                days_ahead        INTEGER,
                raw_json          TEXT,
                UNIQUE(event_id, source, pulled_at)
            );
            CREATE INDEX IF NOT EXISTS idx_fr_event_source_time
                ON forecast_runs(event_id, source, pulled_at DESC);
            CREATE INDEX IF NOT EXISTS idx_fr_city_date_source
                ON forecast_runs(city, date, source);
            CREATE INDEX IF NOT EXISTS idx_fr_date_time
                ON forecast_runs(date, pulled_at DESC);

            -- Hourly detail for a forecast run (72h dense + optional
            -- is_target_day=1 daily-summary row for events >3 days out).
            CREATE TABLE IF NOT EXISTS forecast_hourly (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            INTEGER NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
                hour_ts_utc       TEXT    NOT NULL,
                hour_ts_local     TEXT,
                is_target_day     INTEGER DEFAULT 0,
                temp_c            REAL,
                humidity          REAL,
                cloudcover        REAL,
                windspeed_kph     REAL,
                precip_mm         REAL,
                precip_prob       REAL,
                conditions        TEXT,
                UNIQUE(run_id, hour_ts_utc)
            );
            CREATE INDEX IF NOT EXISTS idx_fh_run_time
                ON forecast_hourly(run_id, hour_ts_utc);

            -- Observations only (Visual Crossing 20-minute loop).
            -- Derived values (forecast_remaining_max, projected_day_max) are
            -- NOT stored here; they live in decision_snapshots.
            CREATE TABLE IF NOT EXISTS live_observations (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id              TEXT    NOT NULL,
                city                  TEXT,
                date                  TEXT    NOT NULL,
                lat                   REAL,
                lon                   REAL,
                pulled_at_utc         TEXT    NOT NULL,
                observed_at_utc       TEXT,
                observed_at_local     TEXT,
                vc_source             TEXT,           -- 'obs'|'fcst'|'comb'|'stats'
                current_temp_c        REAL,
                humidity              REAL,
                cloudcover            REAL,
                windspeed_kph         REAL,
                precip_mm             REAL,
                conditions            TEXT,
                observed_max_so_far_c REAL,
                stations              TEXT,           -- JSON list
                query_cost            INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_lo_event_time
                ON live_observations(event_id, pulled_at_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_lo_city_date_time
                ON live_observations(city, date, pulled_at_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_lo_date_time
                ON live_observations(date, pulled_at_utc DESC);

            -- One row per outcome (bin) per trading run.  Groups via
            -- event_snapshot_group_id (UUID set once per cycle).  Logical
            -- FKs latest_*_run_id / latest_obs_id — not enforced by SQLite.
            CREATE TABLE IF NOT EXISTS decision_snapshots (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                event_snapshot_group_id   TEXT    NOT NULL,
                event_id                  TEXT    NOT NULL,
                contract_id               TEXT    NOT NULL,
                city                      TEXT,
                date                      TEXT,
                evaluated_at_utc          TEXT    NOT NULL,
                -- Forecast context
                latest_ecmwf_run_id       INTEGER,
                latest_gfs_run_id         INTEGER,
                blended_mu_c              REAL,
                blended_sigma_c           REAL,
                -- Observation context
                latest_obs_id             INTEGER,
                current_temp_c            REAL,
                temp_change_1h_c          REAL,
                temp_change_3h_c          REAL,
                observed_max_so_far_c     REAL,
                forecast_remaining_max_c  REAL,
                projected_day_max_c       REAL,
                -- Comparison
                expected_temp_now_c       REAL,
                actual_minus_expected_c   REAL,
                forecast_delta_mu_c       REAL,
                forecast_delta_sigma_c    REAL,
                forecast_agreement_c      REAL,
                -- Market
                market_price              REAL,
                model_prob                REAL,
                raw_model_prob            REAL,
                edge                      REAL,
                ev                        REAL,
                recommended_side          TEXT,
                kelly_size                REAL,
                liquidity_usd             REAL,
                -- Output
                action                    TEXT,
                reason                    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ds_event_time
                ON decision_snapshots(event_id, evaluated_at_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_ds_contract_time
                ON decision_snapshots(contract_id, evaluated_at_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_ds_group
                ON decision_snapshots(event_snapshot_group_id);
            CREATE INDEX IF NOT EXISTS idx_ds_date_time
                ON decision_snapshots(date, evaluated_at_utc DESC);

            -- Phase 2c — VC forecast diagnostic layer (shadow-only).
            -- Captures what VC is forecasting alongside our model at each
            -- evaluation point so disagreement can be analyzed retrospectively.
            -- NEVER feeds the production μ/σ blend.
            CREATE TABLE IF NOT EXISTS vc_forecast_diagnostics (
                id                         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id                   TEXT NOT NULL,
                city                       TEXT,
                target_date                TEXT NOT NULL,
                pulled_at_utc              TEXT NOT NULL,
                kind                       TEXT NOT NULL,         -- 'same_day' | 'future_day'
                -- VC forecast-side output
                vc_projected_day_max_c     REAL,
                vc_forecast_remaining_max_c REAL,
                vc_day_vc_source           TEXT,
                -- Context at pull time
                blended_mu_c               REAL,
                blended_sigma_c            REAL,
                adjusted_mu_c              REAL,
                adjusted_sigma_c           REAL,
                current_temp_c             REAL,
                observed_max_so_far_c      REAL,
                -- Disagreement metrics (precomputed on write)
                vc_vs_blended_mu_c         REAL,
                vc_vs_adjusted_mu_c        REAL,
                vc_vs_observed_max_c       REAL,
                abs_vc_vs_blended          REAL,
                abs_vc_vs_adjusted         REAL,
                vc_bins_apart              INTEGER,
                -- Hourly path (same-day only)
                vc_hourly_path_rmse_c      REAL,
                vc_hourly_path_n           INTEGER,
                -- Flags (informational; never trade-gating in v1)
                flag_vc_disagreement_large INTEGER DEFAULT 0,
                flag_vc_warns_hotter       INTEGER DEFAULT 0,
                flag_vc_warns_colder       INTEGER DEFAULT 0,
                -- Raw VC hourly path preserved for re-analysis
                vc_hourly_forecast_json    TEXT,
                UNIQUE(event_id, pulled_at_utc)
            );
            CREATE INDEX IF NOT EXISTS idx_vcdiag_event_time
                ON vc_forecast_diagnostics(event_id, pulled_at_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_vcdiag_date_time
                ON vc_forecast_diagnostics(target_date, pulled_at_utc DESC);

            -- Phase ML-v1 — per-city ML distribution model registry.
            -- One row per (city, version) — records training inputs, evaluation
            -- scores, and on-disk model path.  Used by inference.py to look up
            -- the active model version for a city.  `activate_at_utc` gates
            -- when a trained model is allowed to contribute to the blend.
            CREATE TABLE IF NOT EXISTS ml_model_registry (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                version              TEXT    NOT NULL,
                city                 TEXT    NOT NULL,
                trained_at_utc       TEXT    NOT NULL,
                training_window_start TEXT,
                training_window_end   TEXT,
                feature_count        INTEGER,
                n_training_rows      INTEGER,
                point_rmse_c         REAL,
                residual_sigma_c     REAL,
                brier_score          REAL,
                model_path           TEXT    NOT NULL,
                activate_at_utc      TEXT,
                notes                TEXT,
                UNIQUE(city, version)
            );
            CREATE INDEX IF NOT EXISTS idx_mlreg_city_trained
                ON ml_model_registry(city, trained_at_utc DESC);

            -- Phase ML-v1 — per-city training rows assembled by the backfill.
            -- One row per (city, target_date, decision_hour_local, feature_version).
            -- features_json is a JSON-encoded list of floats in FEATURE_NAMES
            -- order (see bot/ml/schema.py); NaN is stored as null.
            -- Resumable: backfill script skips rows already present.
            CREATE TABLE IF NOT EXISTS ml_training_rows (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                city                  TEXT    NOT NULL,
                lat_key               REAL    NOT NULL,
                lon_key               REAL    NOT NULL,
                target_date           TEXT    NOT NULL,
                decision_hour_local   INTEGER NOT NULL,
                feature_version       TEXT    NOT NULL,
                features_json         TEXT    NOT NULL,
                t_max_c               REAL    NOT NULL,
                fetched_at_utc        TEXT    NOT NULL,
                UNIQUE(city, target_date, decision_hour_local, feature_version)
            );
            CREATE INDEX IF NOT EXISTS idx_mltr_city_date
                ON ml_training_rows(city, target_date);
            CREATE INDEX IF NOT EXISTS idx_mltr_city_version
                ON ml_training_rows(city, feature_version);

            -- Phase 8 — per-(city, doy, hour) climatological mean/std for
            -- observation variables.  Used by the v2.0 feature builder to
            -- compute *_anomaly z-scores (e.g., pressure_anomaly).  Built
            -- once from the VC cache via bot/scripts/build_obs_climatology.
            CREATE TABLE IF NOT EXISTS obs_climatology_hourly (
                city                 TEXT    NOT NULL,
                doy                  INTEGER NOT NULL,        -- 1..366
                hour                 INTEGER NOT NULL,        -- 0..23 local
                n_samples            INTEGER NOT NULL,
                temp_mu              REAL,
                temp_sigma           REAL,
                dew_mu               REAL,
                dew_sigma            REAL,
                pressure_mu          REAL,
                pressure_sigma       REAL,
                cloudcover_mu        REAL,
                cloudcover_sigma     REAL,
                windspeed_mu         REAL,
                windspeed_sigma      REAL,
                solarradiation_mu    REAL,
                solarradiation_sigma REAL,
                fetched_at_utc       TEXT    NOT NULL,
                PRIMARY KEY (city, doy, hour)
            );
            CREATE INDEX IF NOT EXISTS idx_oclim_h_city_doy
                ON obs_climatology_hourly(city, doy);

            -- Phase 8 — per-(city, doy) daily T_max/T_min climatology +
            -- selected percentiles.  Used as a calibrated baseline in the
            -- evaluation script and as a feature in v2.0 (climatology_mu_today).
            CREATE TABLE IF NOT EXISTS obs_climatology_daily (
                city                 TEXT    NOT NULL,
                doy                  INTEGER NOT NULL,
                n_samples            INTEGER NOT NULL,
                tmax_mu              REAL,
                tmax_sigma           REAL,
                tmax_p10             REAL,
                tmax_p25             REAL,
                tmax_p50             REAL,
                tmax_p75             REAL,
                tmax_p90             REAL,
                tmin_mu              REAL,
                tmin_sigma           REAL,
                tmean_mu             REAL,
                fetched_at_utc       TEXT    NOT NULL,
                PRIMARY KEY (city, doy)
            );
            CREATE INDEX IF NOT EXISTS idx_oclim_d_city
                ON obs_climatology_daily(city);
            CREATE INDEX IF NOT EXISTS idx_vcdiag_large
                ON vc_forecast_diagnostics(flag_vc_disagreement_large, target_date);

            -- Per-city forecast accuracy metrics (rebuilt daily by bias updater).
            -- Rolling 30-day window.  Used for city ranking and per-city
            -- confidence adjustments in trading strategies.
            CREATE TABLE IF NOT EXISTS city_forecast_accuracy (
                city                TEXT PRIMARY KEY,
                lat                 REAL,
                lon                 REAL,
                window_days         INTEGER,
                n_days              INTEGER,
                -- Core metrics
                mae_c               REAL,
                rmse_c              REAL,
                bias_c              REAL,
                -- Stability
                error_std_c         REAL,
                max_error_c         REAL,
                -- Trading usefulness
                pct_within_1c       REAL,        -- fraction of days |error| <= 1.0
                pct_within_2c       REAL,        -- fraction of days |error| <= 2.0
                -- Direction
                pct_underpredicted  REAL,        -- fraction of days actual > forecast
                pct_overpredicted   REAL,        -- fraction of days actual < forecast
                -- Time horizon (MAE by lead)
                mae_d0_c            REAL,
                mae_d1_c            REAL,
                mae_d2_c            REAL,
                n_d0                INTEGER,
                n_d1                INTEGER,
                n_d2                INTEGER,
                -- Average forecast uncertainty (sigma) across recent events
                avg_uncertainty_c   REAL,
                -- Composite score (0-100, higher = more accurate/tradeable)
                accuracy_score      REAL,
                -- Metadata
                updated_at          TEXT
            );

            -- VC cost accounting (cheap surface so we don't scan obs table).
            CREATE TABLE IF NOT EXISTS vc_usage_daily (
                date              TEXT PRIMARY KEY,       -- YYYY-MM-DD UTC
                total_query_cost  INTEGER DEFAULT 0,
                n_calls           INTEGER DEFAULT 0,
                updated_at        TEXT
            );

            -- Price history: periodic snapshots of ALL bin prices for backtesting.
            -- Populated by the trading scan for all bins, not just traded ones.
            CREATE TABLE IF NOT EXISTS bin_price_history (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id          TEXT    NOT NULL,
                contract_id       TEXT    NOT NULL,
                city              TEXT,
                date              TEXT,
                yes_price         REAL,
                no_price          REAL,
                volume_usd        REAL,
                liquidity_usd     REAL,
                recorded_at       TEXT    NOT NULL
            );

            -- Event resolution: records which bin won each event.
            -- Populated by the monitor when it detects a resolved market.
            CREATE TABLE IF NOT EXISTS event_resolutions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id          TEXT    UNIQUE NOT NULL,
                city              TEXT,
                date              TEXT,
                winning_contract_id TEXT,
                winning_range_low REAL,
                winning_range_high REAL,
                winning_yes_price REAL,
                resolved_at       TEXT,
                recorded_at       TEXT    NOT NULL
            );

            -- Activity log: every critical bot action (orders, fills,
            -- cancellations, risk events, WS connect/disconnect).  Same
            -- content as logs/activity.log but indexed for fast dashboard
            -- queries — the operator can scroll the recent feed without
            -- shelling into the droplet to tail the log file.
            -- Per-position order ledger (added 2026-04-30, Phase B).
            -- Tracks EVERY individual CLOB order placed for a position
            -- (initial entry, top-ups, exits) — so we can compute
            --    committed_usdc = sum(intended_usdc) for non-cancelled orders
            -- which is what the top-up gap calc needs to avoid double-
            -- committing capital.  The `positions` table aggregates
            -- (size_usdc, shares, entry_price) are derived from this
            -- ledger by summing across role IN ('entry','topup').
            CREATE TABLE IF NOT EXISTS position_orders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id     INTEGER NOT NULL,
                order_id        TEXT    NOT NULL UNIQUE,    -- CLOB order hash
                role            TEXT    NOT NULL,           -- 'entry' | 'topup' | 'exit'
                intended_usdc   REAL    NOT NULL,           -- size we asked for
                intended_shares REAL    NOT NULL,
                limit_price     REAL    NOT NULL,
                -- Order lifecycle.  status is OUR view; trade_status is
                -- the on-chain WS lifecycle when we have it.
                status          TEXT    NOT NULL,           -- 'pending'|'live'|'partial'|'filled'|'cancelled'|'failed'
                trade_status    TEXT,                       -- 'matched'|'mined'|'confirmed'|'failed'
                filled_shares   REAL    DEFAULT 0,
                filled_usdc     REAL    DEFAULT 0,
                fill_price      REAL,                       -- weighted avg
                fee_usdc        REAL    DEFAULT 0,
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                closed_at       TEXT,
                cancelled_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pos_orders_pid
                ON position_orders(position_id);
            CREATE INDEX IF NOT EXISTS idx_pos_orders_status
                ON position_orders(status);
            CREATE INDEX IF NOT EXISTS idx_pos_orders_orderid
                ON position_orders(order_id);

            -- Activity log: every critical bot action (orders, fills,
            -- cancellations, risk events, WS connect/disconnect).  Same
            -- content as logs/activity.log but indexed for fast dashboard
            -- queries — the operator can scroll the recent feed without
            -- shelling into the droplet to tail the log file.
            CREATE TABLE IF NOT EXISTS activity_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                level       TEXT    NOT NULL,    -- INFO | WARN | ERROR
                category    TEXT    NOT NULL,    -- BUY | SELL | FILL | etc
                message     TEXT    NOT NULL,
                position_id INTEGER,             -- nullable — not every event has one
                metadata    TEXT                 -- JSON blob, optional
            );
            CREATE INDEX IF NOT EXISTS idx_activity_log_ts
                ON activity_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_activity_log_cat
                ON activity_log(category, timestamp DESC);

            -- Per-monitor-cycle health snapshot.  Read by the dashboard
            -- to surface bot health (WS connectivity, wallet balance vs
            -- bankroll cap, on-chain reconciliation drift) without
            -- requiring the dashboard to tail logs or import bot modules.
            -- One row written at the end of each monitor cycle.
            CREATE TABLE IF NOT EXISTS monitor_health (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at              TEXT    NOT NULL,
                ws_running               INTEGER,        -- 0/1; NULL in paper mode
                wallet_balance_usdc      REAL,
                effective_bankroll_usdc  REAL,
                drift_orphan_db          INTEGER DEFAULT 0,
                drift_share_drift        INTEGER DEFAULT 0,
                drift_orphan_chain       INTEGER DEFAULT 0,
                buys_filled              INTEGER DEFAULT 0,
                sells_filled             INTEGER DEFAULT 0,
                topups_filled            INTEGER DEFAULT 0,
                positions_open           INTEGER DEFAULT 0,
                positions_pending        INTEGER DEFAULT 0,
                positions_exiting        INTEGER DEFAULT 0,
                summary_text             TEXT
            );

            -- Per-trade-event dedup table.  Polymarket emits ONE trade
            -- event per match, and a single limit order frequently matches
            -- against multiple resting asks (yielding multiple events for
            -- the same order_id, each with a unique event_id).  We dedup
            -- on event_id so each unique fill applies exactly once, even
            -- across the WS+REST safety-net dual paths and Polymarket's
            -- at-least-once redelivery (matched -> mined -> confirmed
            -- redeliveries of the SAME trade carry the same event_id and
            -- are caught here).
            --
            -- Replaces the broken per-position trade_status gate that
            -- previously dropped chunks 2..N of any chunked fill.
            CREATE TABLE IF NOT EXISTS processed_trade_events (
                event_id     TEXT    PRIMARY KEY,
                processed_at TEXT    NOT NULL
            );
        """)

        # --- Phase 2 indexes on existing table (date-windowed scans) ---
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_temp_events_date "
                         "ON temp_events(date)")
        except Exception:
            pass
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bin_price_history_event "
                         "ON bin_price_history(event_id, contract_id, recorded_at)")
        except Exception:
            pass
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_resolutions_event "
                         "ON event_resolutions(event_id)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bin price history (for backtesting any strategy)
# ---------------------------------------------------------------------------

def insert_bin_price_snapshot(
    event_id: str, contract_id: str, city: str, date: str,
    yes_price: float, no_price: float,
    volume_usd: float, liquidity_usd: float,
    recorded_at: str,
) -> int:
    sql = """
        INSERT INTO bin_price_history
            (event_id, contract_id, city, date, yes_price, no_price,
             volume_usd, liquidity_usd, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            event_id, contract_id, city, date,
            yes_price, no_price, volume_usd, liquidity_usd, recorded_at,
        ))
        return cur.lastrowid


def insert_bin_price_snapshots_bulk(rows: list[dict], recorded_at: str) -> int:
    """Insert price snapshots for all bins in one batch."""
    sql = """
        INSERT INTO bin_price_history
            (event_id, contract_id, city, date, yes_price, no_price,
             volume_usd, liquidity_usd, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        conn.executemany(sql, [
            (r["event_id"], r["contract_id"], r.get("city"), r.get("date"),
             r.get("yes_price"), r.get("no_price"),
             r.get("volume_usd"), r.get("liquidity_usd"), recorded_at)
            for r in rows
        ])
        return len(rows)


def insert_event_resolution(
    event_id: str, city: str, date: str,
    winning_contract_id: str, winning_range_low: float,
    winning_range_high: float, winning_yes_price: float,
    resolved_at: str, recorded_at: str,
) -> int:
    sql = """
        INSERT OR IGNORE INTO event_resolutions
            (event_id, city, date, winning_contract_id,
             winning_range_low, winning_range_high, winning_yes_price,
             resolved_at, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            event_id, city, date, winning_contract_id,
            winning_range_low, winning_range_high, winning_yes_price,
            resolved_at, recorded_at,
        ))
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Legacy signal helpers (retained for execution.py)
# ---------------------------------------------------------------------------

def insert_signal(
    timestamp: str,
    contract_id: str,
    question: str = None,
    market_p: float = None,
    model_p: float = None,
    ev: float = None,
    recommended_side: str = None,
    kelly_size: float = None,
) -> int:
    sql = """
        INSERT INTO signals
            (timestamp, contract_id, question, market_p, model_p, ev,
             recommended_side, kelly_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(
            sql,
            (timestamp, contract_id, question, market_p, model_p, ev,
             recommended_side, kelly_size),
        )
        return cur.lastrowid


def insert_position(
    contract_id: str,
    side: str,
    size_usdc: float,
    entry_price: float,
    entry_time: str,
    order_id: str = None,
    is_paper: int = 1,
    question: str = None,
    city: str = None,
    date: str = None,
    event_id: str = None,
    model_prob: float = None,
    market_prob: float = None,
    ev: float = None,
    edge: float = None,
    fill_status: str = "filled",
    shares: float = None,
    scan_timestamp: str = None,
    gamma_market_id: str = None,
    range_low: float = None,
    range_high: float = None,
    unit: str = None,
    yes_token_id: str = None,
    no_token_id: str = None,
    lat: float = None,
    lon: float = None,
    forecast_sigma_c: float = None,
    entry_snapshot_id: int = None,
    target_size_usdc: float = None,
    stop_loss_price: float = None,
    strategy: str = None,
    pre_entry_volatility: float = None,
    pre_entry_trend: float = None,
    pre_entry_momentum: float = None,
) -> int:
    sql = """
        INSERT INTO positions (
            contract_id, side, size_usdc, entry_price, entry_time,
            order_id, is_paper, question, city, date, event_id,
            model_prob, market_prob, ev, edge,
            fill_status, shares, scan_timestamp,
            unrealized_pnl, current_price, gamma_market_id,
            range_low, range_high, unit,
            yes_token_id, no_token_id, lat, lon,
            forecast_sigma_c, entry_snapshot_id, target_size_usdc,
            stop_loss_price, peak_price, strategy,
            pre_entry_volatility, pre_entry_trend, pre_entry_momentum
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            contract_id, side, size_usdc, entry_price, entry_time,
            order_id, is_paper, question, city, date, event_id,
            model_prob, market_prob, ev, edge,
            fill_status, shares, scan_timestamp,
            entry_price,  # current_price initialised to entry_price
            gamma_market_id,
            range_low, range_high, unit,
            yes_token_id, no_token_id, lat, lon,
            forecast_sigma_c,
            entry_snapshot_id,
            target_size_usdc,
            stop_loss_price,
            entry_price,  # peak_price initialised to entry_price
            strategy,
            pre_entry_volatility,
            pre_entry_trend,
            pre_entry_momentum,
        ))
        return cur.lastrowid


def get_open_positions() -> list[dict]:
    """Return positions that are economically still open — meaning capital
    is still committed and on-chain.  Includes:
      * status='open'    — fully active positions
      * status='exiting' — sell order placed but not yet filled (still
                            on-chain, still consumes exposure budget)

    Callers that ONLY want fully-active positions (e.g. to avoid
    re-firing exit logic on positions already exiting) should filter
    further by `status == 'open'` themselves.
    """
    sql = (
        "SELECT * FROM positions "
        "WHERE status IN ('open', 'exiting') "
        "ORDER BY entry_time ASC"
    )
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_pending_positions() -> list[dict]:
    """Return live orders placed but not yet confirmed as filled."""
    sql = """
        SELECT * FROM positions
        WHERE status = 'open' AND fill_status = 'pending'
        ORDER BY entry_time ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_recent_cancelled_count(contract_id: str, within_hours: int = 6) -> int:
    """Count BUY orders cancelled for this contract in the last N hours.

    Used by the trading loop to cap buy retries on contracts that never
    fill — without the cap, a contract whose limit is fundamentally too
    low (book bids never reach our ask) would generate infinite cancel/
    re-issue cycles every scan.

    Counts:
      * status='closed' AND fill_status='cancelled'  (the cancel pass)
    Excludes:
      * filled orders (entered the book and either matched or are
        being held)
      * positions cancelled for other reasons (e.g., manual closure)

    NOTE: this counts the per-CONTRACT side (e.g., "Chicago 60-65F YES").
    A different bin in the same event is a different contract, so this
    doesn't accidentally cap a wholly different bet.
    """
    # SQLite stores entry_time as TEXT in ISO-8601 (sometimes with 'T'
    # separator + microseconds + tz offset, sometimes without).  Wrap in
    # datetime() on both sides so comparison is done on parsed timestamps,
    # not on raw strings (which would mis-order 'T' vs space-separated).
    sql = """
        SELECT COUNT(*) FROM positions
        WHERE contract_id = ?
          AND status = 'closed'
          AND fill_status = 'cancelled'
          AND datetime(entry_time) >= datetime('now', ?)
    """
    with _get_conn() as conn:
        row = conn.execute(
            sql, (contract_id, f"-{int(within_hours)} hours")
        ).fetchone()
        return int(row[0]) if row else 0


def get_open_positions_for_event(city: str, date: str) -> list[dict]:
    """Return all open (non-cancelled) positions for a given city+date event."""
    sql = """
        SELECT * FROM positions
        WHERE city = ? AND date = ? AND status = 'open'
        ORDER BY entry_time ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (city, date)).fetchall()]


def count_open_bins_for_event(city: str, date: str, side: str | None = None) -> int:
    """Count distinct open positions for a city+date event, optionally filtered by side."""
    if side:
        sql = """
            SELECT COUNT(*) FROM positions
            WHERE city = ? AND date = ? AND status = 'open' AND side = ?
        """
        with _get_conn() as conn:
            row = conn.execute(sql, (city, date, side)).fetchone()
            return row[0] if row else 0
    sql = """
        SELECT COUNT(*) FROM positions
        WHERE city = ? AND date = ? AND status = 'open'
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (city, date)).fetchone()
        return row[0] if row else 0


def update_position_outcome(
    position_id: int,
    exit_price: float,
    exit_time: str,
    pnl: float,
    status: str = "closed",
    exit_reason: str | None = None,
    exit_snapshot_id: int | None = None,
) -> None:
    sql = """
        UPDATE positions
        SET exit_price = ?, exit_time = ?, pnl = ?, status = ?,
            unrealized_pnl = 0,
            exit_reason = COALESCE(?, exit_reason),
            exit_snapshot_id = COALESCE(?, exit_snapshot_id)
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (exit_price, exit_time, pnl, status,
                           exit_reason, exit_snapshot_id, position_id))


def update_position_exit_pending(
    position_id: int,
    exit_order_id: str,
    exit_intended_price: float,
    exit_retry_count: int,
    exit_reason: str | None = None,
) -> None:
    """Mark a position as actively exiting via a CLOB sell order.

    Called by execute_exit() in live mode to:
      * record the live order id so the monitor can poll/cancel it
      * stamp the intended_exit_price on the FIRST attempt (preserved across
        retries via COALESCE — the original trigger price doesn't change)
      * bump the retry counter so the next monitor cycle knows which
        ladder rung to use
      * set status='exiting' so the rest of the bot knows the position is
        in the middle of being closed (don't double-fire exits, don't
        treat as still-open for sizing decisions).
    """
    sql = """
        UPDATE positions
        SET exit_order_id = ?,
            exit_intended_price = COALESCE(exit_intended_price, ?),
            exit_retry_count = ?,
            status = 'exiting',
            exit_reason = COALESCE(?, exit_reason)
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (
            exit_order_id, exit_intended_price, exit_retry_count,
            exit_reason, position_id,
        ))


def update_position_exit_filled(
    position_id: int,
    actual_exit_price: float,
    exit_time: str,
    pnl: float,
    exit_snapshot_id: int | None = None,
) -> None:
    """Confirm an exit order fill: actual_exit_price captures the real
    fill (vs intended), and the position transitions from 'exiting' to
    'closed'.  Called by the fill-reconciliation step in monitor.py."""
    sql = """
        UPDATE positions
        SET status = 'closed',
            actual_exit_price = ?,
            exit_price = ?,
            exit_time = ?,
            pnl = ?,
            unrealized_pnl = 0,
            exit_snapshot_id = COALESCE(?, exit_snapshot_id)
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (
            actual_exit_price, actual_exit_price, exit_time, pnl,
            exit_snapshot_id, position_id,
        ))


# ---------------------------------------------------------------------------
# Live Phase 9 — Trade lifecycle tracking (user-channel WS)
# ---------------------------------------------------------------------------

# Lifecycle stages, monotonic.  Higher rank = closer to terminal.
# Used by update_position_trade_status to enforce "never downgrade":
# at-least-once delivery means a CONFIRMED event might be followed by a
# delayed MATCHED for the same trade — we must ignore the regression.
_TRADE_STATUS_RANK = {
    None:        0,
    "":          0,
    "matched":   1,
    "mined":     2,
    "retrying":  2,   # parallel branch — same priority as mined
    "confirmed": 3,
    "failed":    3,
}


def _trade_rank(status: str | None) -> int:
    return _TRADE_STATUS_RANK.get((status or "").lower(), 0)


def update_position_trade_status(
    position_id: int,
    new_status: str,
    *,
    side: str = "entry",
    last_event_id: str | None = None,
) -> bool:
    """Advance the trade lifecycle stage on a position.  Idempotent and
    monotonic — a regression (e.g. CONFIRMED → MATCHED from a delayed
    duplicate) is ignored.  Returns True if the row was updated.

    side='entry' updates trade_status; side='exit' updates exit_trade_status.

    The corresponding last_*_trade_event_id column is updated to the most
    recent event we acted on, useful for debug + tests.
    """
    if side not in ("entry", "exit"):
        raise ValueError(f"side must be 'entry' or 'exit', got {side!r}")
    new_status_norm = (new_status or "").lower()
    new_rank = _trade_rank(new_status_norm)
    if new_rank == 0:
        return False

    col_status = "trade_status" if side == "entry" else "exit_trade_status"
    col_event  = "last_trade_event_id" if side == "entry" else "last_exit_trade_event_id"

    with _get_conn() as conn:
        row = conn.execute(
            f"SELECT {col_status} FROM positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            return False
        cur_rank = _trade_rank(row[0])
        if new_rank <= cur_rank:
            return False  # never regress; same-rank duplicates are no-ops too

        conn.execute(
            f"UPDATE positions SET {col_status} = ?, {col_event} = ? WHERE id = ?",
            (new_status_norm, last_event_id, position_id),
        )
        return True


def get_position_by_order_id(order_id: str) -> dict | None:
    """Find the position whose entry order_id, exit_order_id, or
    pending_topup_order_id matches.  Returns the first match or None.

    Used by the user-channel WS handler to route an order/trade event
    back to the position it belongs to.  We check all three columns
    because the same wallet places buys, sells, and top-ups on the
    same channel.
    """
    if not order_id:
        return None
    sql = """
        SELECT * FROM positions
        WHERE order_id = ?
           OR exit_order_id = ?
           OR pending_topup_order_id = ?
        ORDER BY id DESC
        LIMIT 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (order_id, order_id, order_id)).fetchone()
        return dict(row) if row else None


def classify_position_role(position: dict, order_id: str) -> str:
    """Return 'entry', 'exit', or 'topup' based on which order_id column
    matched.  Used by the WS dispatcher to pick the right write path."""
    if not order_id:
        return "entry"
    if position.get("exit_order_id") == order_id:
        return "exit"
    if position.get("pending_topup_order_id") == order_id:
        return "topup"
    return "entry"


# ---------------------------------------------------------------------------
# Monitor health snapshots (for dashboard health strip)
# ---------------------------------------------------------------------------

def insert_monitor_health(snapshot: dict) -> None:
    """Append a row to monitor_health.  Caller passes a dict matching the
    column names in the table (extras are ignored; missing fields default
    to NULL/0).  Called once per monitor cycle."""
    cols = [
        "recorded_at", "ws_running",
        "wallet_balance_usdc", "effective_bankroll_usdc",
        "drift_orphan_db", "drift_share_drift", "drift_orphan_chain",
        "buys_filled", "sells_filled", "topups_filled",
        "positions_open", "positions_pending", "positions_exiting",
        "summary_text",
    ]
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO monitor_health ({','.join(cols)}) VALUES ({placeholders})"
    values = tuple(snapshot.get(c) for c in cols)
    with _get_conn() as conn:
        conn.execute(sql, values)


def get_latest_monitor_health() -> dict | None:
    """Most recent monitor_health snapshot, or None if the table is empty
    (e.g. fresh install before the first monitor cycle has run)."""
    sql = "SELECT * FROM monitor_health ORDER BY id DESC LIMIT 1"
    with _get_conn() as conn:
        row = conn.execute(sql).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Position-orders ledger (Phase B, 2026-04-30)
# ---------------------------------------------------------------------------

# Statuses we consider "still committing capital" — i.e., they could yet
# fill on the book, so they count toward `committed_usdc` for top-up
# gap calculation.  'partial' means the order matched some shares and
# the rest is still resting (the screenshot bug — we now correctly
# count this as committed).
_ORDER_STATUS_COMMITTING = ("pending", "live", "matched", "partial")
# Terminal statuses — no further fills possible.
_ORDER_STATUS_TERMINAL   = ("filled", "cancelled", "failed")


def insert_position_order(
    *,
    position_id:     int,
    order_id:        str,
    role:            str,                # 'entry' | 'topup' | 'exit'
    intended_usdc:   float,
    intended_shares: float,
    limit_price:     float,
    status:          str = "pending",
    trade_status:    str | None = None,
) -> int:
    """Record a newly-placed CLOB order in the position_orders ledger.

    Called from execute_signal (entry), execute_topup (topup), and
    execute_exit (exit) immediately after the CLOB POST succeeds.
    The lifecycle (status, trade_status, filled_*) is updated later
    by fill_handler events and monitor reconciliation.
    """
    if role not in ("entry", "topup", "exit"):
        raise ValueError(f"role must be 'entry'|'topup'|'exit', got {role!r}")
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO position_orders
            (position_id, order_id, role, intended_usdc, intended_shares,
             limit_price, status, trade_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            position_id, order_id, role, intended_usdc, intended_shares,
            limit_price, status, trade_status, now, now,
        ))
        return cur.lastrowid


def update_position_order_status(
    order_id: str,
    *,
    status:           str | None = None,
    trade_status:     str | None = None,
    filled_shares:    float | None = None,
    filled_usdc:      float | None = None,
    fill_price:       float | None = None,
    fee_usdc:         float | None = None,
    cancelled_reason: str | None = None,
    closed:           bool = False,
) -> bool:
    """Update the lifecycle/fill data for a position_orders row.

    Lookup is by `order_id` (CLOB order hash) since that's the stable
    identifier surfaced from the CLOB.  Any None field is left unchanged.
    `closed=True` stamps closed_at to now (call this on terminal status).
    Returns True if a row was updated.
    """
    sets = []
    args: list = []
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if trade_status is not None:
        sets.append("trade_status = ?")
        args.append(trade_status)
    if filled_shares is not None:
        sets.append("filled_shares = ?")
        args.append(filled_shares)
    if filled_usdc is not None:
        sets.append("filled_usdc = ?")
        args.append(filled_usdc)
    if fill_price is not None:
        sets.append("fill_price = ?")
        args.append(fill_price)
    if fee_usdc is not None:
        sets.append("fee_usdc = ?")
        args.append(fee_usdc)
    if cancelled_reason is not None:
        sets.append("cancelled_reason = ?")
        args.append(cancelled_reason)
    now = datetime.now(timezone.utc).isoformat()
    sets.append("updated_at = ?")
    args.append(now)
    if closed:
        sets.append("closed_at = ?")
        args.append(now)
    args.append(order_id)
    sql = f"UPDATE position_orders SET {', '.join(sets)} WHERE order_id = ?"
    with _get_conn() as conn:
        cur = conn.execute(sql, args)
        return cur.rowcount > 0


def get_position_orders(position_id: int, *, role: str | None = None) -> list[dict]:
    """All orders for a position, oldest first.  Optional role filter
    ('entry'|'topup'|'exit') for queries like 'sum filled across all
    entries+topups'."""
    sql = "SELECT * FROM position_orders WHERE position_id = ?"
    args: list = [position_id]
    if role:
        sql += " AND role = ?"
        args.append(role)
    sql += " ORDER BY id ASC"
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, tuple(args)).fetchall()]


def get_committed_usdc(position_id: int) -> float:
    """Sum the dollar exposure that COULD STILL fill — i.e., orders
    not in a terminal cancelled/failed state.  This is what the top-up
    gap calc uses: `target - committed_usdc` is the true remaining gap,
    counting both filled-on-chain AND still-resting-on-book portions.

    Replaces the buggy `target - filled_only` calc that placed top-ups
    on top of resting orders, double-committing capital (the user-
    reported screenshot bug from 2026-04-30).

    Edge case: 'filled' status with filled_usdc significantly less than
    intended_usdc means the order matched a partial fill but the rest
    is still resting.  The proper status would be 'partial' but legacy
    fill_handler may have stamped 'filled'.  We defensively count those
    as still-committing (use intended_usdc) so we don't underestimate.
    """
    sql = """
        SELECT COALESCE(SUM(
            CASE
                -- Fully filled (within 1¢ rounding) → count actual cost
                WHEN status = 'filled' AND filled_usdc >= intended_usdc * 0.99
                    THEN filled_usdc
                -- "Filled" but really partial + still resting → count intended
                WHEN status = 'filled' AND filled_usdc < intended_usdc * 0.99
                    THEN intended_usdc
                -- Still actively committing → count intended (rest can fill)
                WHEN status IN ('pending','live','matched','partial')
                    THEN intended_usdc
                -- Cancelled with a partial fill: the filled shares are
                -- real on-chain capital; the rest was freed by the cancel.
                -- Count only the filled portion (the partial-cancelled
                -- semantic — "we kept what we got, gave up the rest").
                WHEN status = 'cancelled' AND filled_usdc > 0
                    THEN filled_usdc
                -- Cancelled with no fill OR failed on chain → no capital
                ELSE 0
            END
        ), 0)
        FROM position_orders
        WHERE position_id = ?
          AND role IN ('entry', 'topup')
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (position_id,)).fetchone()
        return float(row[0] if row else 0.0)


def get_filled_usdc(position_id: int) -> float:
    """Sum of `filled_usdc` for entry+topup orders.  This is the real
    on-chain cost so far (excludes resting orders).  Used for accurate
    P&L math + the dashboard 'Filled' column."""
    sql = """
        SELECT COALESCE(SUM(filled_usdc), 0)
        FROM position_orders
        WHERE position_id = ? AND role IN ('entry', 'topup')
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (position_id,)).fetchone()
        return float(row[0] if row else 0.0)


def get_overcommitted_positions() -> list[dict]:
    """Open live positions whose ledger-derived committed_usdc exceeds
    target_size_usdc.  Used by the auto-cancel sweep to identify which
    positions need a resting order trimmed.

    Returns rows with: position_id, target_size_usdc, committed_usdc, excess.
    """
    # Mirrors the get_committed_usdc semantics so we can sort + filter
    # in one query.  Tolerance: $1 to avoid sub-cent rounding triggers.
    sql = """
        SELECT p.id AS position_id, p.target_size_usdc, p.city, p.date,
               COALESCE(SUM(
                   CASE
                       WHEN po.status = 'filled' AND po.filled_usdc >= po.intended_usdc * 0.99
                           THEN po.filled_usdc
                       WHEN po.status = 'filled' AND po.filled_usdc < po.intended_usdc * 0.99
                           THEN po.intended_usdc
                       WHEN po.status IN ('pending','live','matched','partial')
                           THEN po.intended_usdc
                       WHEN po.status = 'cancelled' AND po.filled_usdc > 0
                           THEN po.filled_usdc
                       ELSE 0
                   END
               ), 0) AS committed_usdc
        FROM positions p
        JOIN position_orders po ON po.position_id = p.id
        WHERE p.is_paper = 0
          AND p.status = 'open'
          AND po.role IN ('entry', 'topup')
          AND p.target_size_usdc IS NOT NULL
          AND p.target_size_usdc > 0
        GROUP BY p.id, p.target_size_usdc, p.city, p.date
        HAVING committed_usdc > p.target_size_usdc + 1.0
    """
    with _get_conn() as conn:
        rows = []
        for r in conn.execute(sql).fetchall():
            d = dict(r)
            d["excess"] = round(d["committed_usdc"] - d["target_size_usdc"], 4)
            rows.append(d)
        return rows


def get_cancellable_orders_for_position(position_id: int) -> list[dict]:
    """Orders that are currently committing capital AND have a non-zero
    resting portion (intended - filled).  Returned oldest-first so the
    auto-cancel sweep cancels the longest-stuck order first.

    Excludes already-terminal statuses (filled/cancelled/failed).  An
    order with status='partial' and filled_usdc < intended_usdc qualifies —
    cancelling it will free the unfilled portion while the filled shares
    stay on chain.
    """
    sql = """
        SELECT *,
               (intended_usdc - COALESCE(filled_usdc, 0)) AS resting_usdc
        FROM position_orders
        WHERE position_id = ?
          AND role IN ('entry', 'topup')
          AND status IN ('pending', 'live', 'matched', 'partial')
          AND (intended_usdc - COALESCE(filled_usdc, 0)) > 0.01
        ORDER BY created_at ASC, id ASC   -- oldest by wall-clock first
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (position_id,)).fetchall()]


def get_position_order_by_id(order_id: str) -> dict | None:
    """Look up a position_orders row by its CLOB order id."""
    if not order_id:
        return None
    sql = "SELECT * FROM position_orders WHERE order_id = ? LIMIT 1"
    with _get_conn() as conn:
        row = conn.execute(sql, (order_id,)).fetchone()
        return dict(row) if row else None


def backfill_position_orders() -> dict:
    """One-shot migration: walk every existing position and synthesize
    position_orders rows from what we can infer from the legacy fields.

    Idempotent — skips positions that already have ledger rows.  Safe
    to run on every startup as a no-op when there's nothing to backfill.

    For each existing position:
      * If `order_id` is set → create one 'entry' row.  Status is derived
        from fill_status: filled→'filled', cancelled→'cancelled',
        pending→'pending'.  filled_usdc/shares come from the position
        row's current values.
      * If `pending_topup_order_id` is set → create one 'topup' row in
        'pending' state.  Reconciliation will close it.
      * If `exit_order_id` is set → create one 'exit' row in
        'pending'/'filled'/'cancelled' based on position.status.

    Returns counts {entries, topups, exits, skipped} for logging.
    """
    counts = {"entries": 0, "topups": 0, "exits": 0, "skipped": 0}
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        positions = [dict(r) for r in conn.execute("""
            SELECT id, order_id, pending_topup_order_id, exit_order_id,
                   size_usdc, target_size_usdc, shares, entry_price,
                   exit_intended_price, fill_status, status, trade_status,
                   exit_trade_status, cancelled_reason, entry_time,
                   pending_topup_amount_usdc, pending_topup_intended_price,
                   actual_exit_price, exit_fees, entry_fees
            FROM positions
        """).fetchall()]

        existing_ids = {r[0] for r in conn.execute(
            "SELECT order_id FROM position_orders"
        ).fetchall()}

        for p in positions:
            pid = p["id"]
            target = float(p.get("target_size_usdc") or p.get("size_usdc") or 0)
            entry_price = float(p.get("entry_price") or 0)

            # ---- entry ----
            entry_oid = p.get("order_id")
            if entry_oid and entry_oid not in existing_ids:
                fs = (p.get("fill_status") or "").lower()
                ts = (p.get("trade_status") or "")
                cancelled = (fs == "cancelled")
                filled = (fs == "filled")
                # `status` maps fill_status → ledger status
                if cancelled:
                    o_status = "cancelled"
                elif filled:
                    o_status = "filled"
                else:
                    o_status = "pending"
                shares = float(p.get("shares") or 0)
                filled_usdc = float(p.get("size_usdc") or 0) if filled else 0.0
                conn.execute("""
                    INSERT INTO position_orders
                        (position_id, order_id, role, intended_usdc,
                         intended_shares, limit_price, status, trade_status,
                         filled_shares, filled_usdc, fill_price, fee_usdc,
                         created_at, updated_at, closed_at, cancelled_reason)
                    VALUES (?, ?, 'entry', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid, entry_oid,
                    target if target > 0 else filled_usdc,
                    (target / entry_price) if (target > 0 and entry_price > 0)
                        else shares,
                    entry_price,
                    o_status, ts or None,
                    shares if filled else 0,
                    filled_usdc,
                    entry_price if filled and shares > 0 else None,
                    float(p.get("entry_fees") or 0),
                    p.get("entry_time") or now, now,
                    now if o_status in ("filled", "cancelled") else None,
                    p.get("cancelled_reason"),
                ))
                counts["entries"] += 1

            # ---- in-flight top-up ----
            tup_oid = p.get("pending_topup_order_id")
            if tup_oid and tup_oid not in existing_ids:
                amt = float(p.get("pending_topup_amount_usdc") or 0)
                pri = float(p.get("pending_topup_intended_price") or entry_price or 0)
                shares_intended = (amt / pri) if (amt > 0 and pri > 0) else 0
                conn.execute("""
                    INSERT INTO position_orders
                        (position_id, order_id, role, intended_usdc,
                         intended_shares, limit_price, status,
                         created_at, updated_at)
                    VALUES (?, ?, 'topup', ?, ?, ?, 'pending', ?, ?)
                """, (pid, tup_oid, amt, shares_intended, pri, now, now))
                counts["topups"] += 1

            # ---- exit (in-flight or completed) ----
            exit_oid = p.get("exit_order_id")
            if exit_oid and exit_oid not in existing_ids:
                exit_pri = float(p.get("exit_intended_price")
                                 or p.get("actual_exit_price") or 0)
                shares = float(p.get("shares") or 0)
                pos_status = (p.get("status") or "").lower()
                actual_exit = p.get("actual_exit_price")
                if pos_status == "closed" and actual_exit is not None:
                    o_status = "filled"
                    filled_shares_x = shares
                    filled_usdc_x = shares * float(actual_exit)
                elif pos_status == "exiting":
                    o_status = "pending"
                    filled_shares_x = 0
                    filled_usdc_x = 0
                else:
                    o_status = "cancelled"
                    filled_shares_x = 0
                    filled_usdc_x = 0
                conn.execute("""
                    INSERT INTO position_orders
                        (position_id, order_id, role, intended_usdc,
                         intended_shares, limit_price, status, trade_status,
                         filled_shares, filled_usdc, fill_price, fee_usdc,
                         created_at, updated_at, closed_at)
                    VALUES (?, ?, 'exit', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid, exit_oid,
                    shares * exit_pri if exit_pri > 0 else 0,
                    shares, exit_pri,
                    o_status, p.get("exit_trade_status"),
                    filled_shares_x, filled_usdc_x,
                    float(actual_exit) if (o_status == "filled" and actual_exit is not None) else None,
                    float(p.get("exit_fees") or 0),
                    now, now,
                    now if o_status in ("filled", "cancelled") else None,
                ))
                counts["exits"] += 1

    return counts


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------

def insert_activity_log(
    *,
    timestamp:   str,
    level:       str,
    category:    str,
    message:     str,
    position_id: int | None = None,
    metadata:    str | None = None,
) -> None:
    """Append one critical-event row.  Called by activity.log_activity().
    Never raises — the caller guards with a try/except so a DB write
    failure can't mask the underlying action."""
    sql = """
        INSERT INTO activity_log
            (timestamp, level, category, message, position_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        conn.execute(sql, (timestamp, level, category, message, position_id, metadata))


def get_recent_activity(
    limit:       int = 200,
    *,
    categories:  list[str] | None = None,
    levels:      list[str] | None = None,
    since_iso:   str | None = None,
) -> list[dict]:
    """Return recent activity_log rows, newest first, for the dashboard
    Activity tab.  All filter args are optional."""
    where = []
    args: list = []
    if categories:
        ph = ",".join(["?"] * len(categories))
        where.append(f"category IN ({ph})")
        args.extend(categories)
    if levels:
        ph = ",".join(["?"] * len(levels))
        where.append(f"level IN ({ph})")
        args.extend(levels)
    if since_iso:
        where.append("timestamp >= ?")
        args.append(since_iso)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT * FROM activity_log
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """
    args.append(int(limit))
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, tuple(args)).fetchall()]


def get_activity_categories() -> list[str]:
    """Distinct categories present in the table — drives the dashboard
    category filter dropdown."""
    sql = "SELECT DISTINCT category FROM activity_log ORDER BY category"
    with _get_conn() as conn:
        return [r[0] for r in conn.execute(sql).fetchall()]


def add_position_entry_fee(position_id: int, fee_usdc: float) -> None:
    """Accumulate an entry-side fee on a position.

    Called when a buy fills (initial or top-up).  Uses COALESCE so the
    first call from a row that pre-dates the column (NULL) initializes
    correctly.  Subsequent calls add to the running total.
    """
    if fee_usdc <= 0:
        return
    sql = """
        UPDATE positions
        SET entry_fees = COALESCE(entry_fees, 0) + ?
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (float(fee_usdc), position_id))


def set_position_exit_fee_and_net_pnl(
    position_id: int, exit_fee_usdc: float,
) -> None:
    """Record the exit-side fee and recompute pnl_net = pnl - all fees.

    Called when the exit order fills (during reconciliation).  pnl_net is
    derived in SQL from the freshly-set exit_fees + accumulated entry_fees +
    the gross pnl that update_position_exit_filled already wrote.
    """
    sql = """
        UPDATE positions
        SET exit_fees = ?,
            pnl_net = COALESCE(pnl, 0)
                      - COALESCE(entry_fees, 0)
                      - COALESCE(?, 0)
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (float(exit_fee_usdc), float(exit_fee_usdc), position_id))


def update_position_topup_pending(
    position_id: int,
    order_id: str,
    amount_usdc: float,
    intended_price: float,
) -> None:
    """Stamp the pending top-up fields on a position when execute_topup
    posts a live CLOB buy.  Cleared on fill (via update_position_topup
    which now also clears these) or on cancel (via clear_position_topup_pending).

    Only ONE pending top-up at a time per position — _run_topups checks
    pending_topup_order_id IS NULL before issuing.
    """
    sql = """
        UPDATE positions
        SET pending_topup_order_id       = ?,
            pending_topup_amount_usdc    = ?,
            pending_topup_intended_price = ?
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (order_id, amount_usdc, intended_price, position_id))


def clear_position_topup_pending(position_id: int) -> None:
    """Clear the in-flight top-up fields without merging — used when the
    top-up order is cancelled (externally or because it sat unfilled long
    enough to be killed by the monitor)."""
    sql = """
        UPDATE positions
        SET pending_topup_order_id       = NULL,
            pending_topup_amount_usdc    = NULL,
            pending_topup_intended_price = NULL
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (position_id,))


def get_positions_with_pending_topup() -> list[dict]:
    """Positions whose pending_topup_order_id is set — the in-flight
    top-up buys awaiting fill.  Used by monitor reconciliation."""
    sql = """
        SELECT * FROM positions
        WHERE pending_topup_order_id IS NOT NULL
          AND status IN ('open', 'exiting')
        ORDER BY id ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_exiting_positions() -> list[dict]:
    """Positions whose status='exiting' — the in-flight sell ladder.
    Returned sorted by oldest exit_order_id first (so monitor processes
    the longest-pending exits before the newer ones)."""
    sql = """
        SELECT * FROM positions
        WHERE status = 'exiting'
        ORDER BY id ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def update_position_topup(
    position_id: int,
    added_usdc: float,
    added_shares: float,
    new_avg_price: float,
) -> None:
    """Add to an existing position's size (liquidity-aware top-up).
    Updates size_usdc, shares, recalculates entry_price as weighted avg,
    and clears any pending_topup_* fields (the merge replaces the in-flight
    state).

    SINGLE-SHOT semantics — clears pending_topup_* atomically.  Use only
    when you know the topup is fully complete (paper mode, or a synthesized
    one-shot apply).  For per-chunk WS fills use add_position_topup_fill
    instead, since clearing pending_topup_order_id mid-stream prevents
    chunks 2..N from being routed back to this position.
    """
    sql = """
        UPDATE positions
        SET size_usdc = size_usdc + ?,
            shares = shares + ?,
            entry_price = ?,
            current_price = ?,
            pending_topup_order_id       = NULL,
            pending_topup_amount_usdc    = NULL,
            pending_topup_intended_price = NULL
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (added_usdc, added_shares, new_avg_price,
                           new_avg_price, position_id))


def add_position_topup_fill(
    position_id: int,
    added_usdc: float,
    added_shares: float,
) -> None:
    """Apply ONE chunk of a topup-order fill, additively, WITHOUT clearing
    the pending_topup_* fields.

    Mirrors update_position_topup's accumulation math but leaves the
    in-flight markers alone so subsequent chunks of the same topup order
    can still be routed back to this position via
    get_position_by_order_id (which matches on pending_topup_order_id).

    Caller is responsible for invoking clear_position_topup_pending(pid)
    when the position_orders ledger row signals the order is fully filled.

    Recomputes entry_price as weighted average using the cumulative
    size_usdc / shares post-update — same cost-basis semantics as the
    single-shot helper, just split across multiple calls.
    """
    a_usdc   = float(added_usdc)
    a_shares = float(added_shares)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT shares, size_usdc FROM positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            return
        prev_shares = float(row["shares"]    or 0)
        prev_usdc   = float(row["size_usdc"] or 0)
        new_shares  = prev_shares + a_shares
        new_usdc    = prev_usdc   + a_usdc
        new_avg     = (new_usdc / new_shares) if new_shares > 0 else 0.0
        conn.execute("""
            UPDATE positions
            SET size_usdc     = ?,
                shares        = ?,
                entry_price   = ?,
                current_price = ?
            WHERE id = ?
        """, (new_usdc, new_shares, new_avg, new_avg, position_id))


# Tolerance for "is the position fully exited?".  Polymarket fills are
# at most 4 decimals of share precision, so a residual ≤ 0.001 is rounding.
_EXIT_COMPLETE_SHARE_TOLERANCE = 0.001


def add_position_exit_fill(
    position_id: int,
    sold_shares: float,
    fill_price: float,
    fee_usdc: float,
) -> dict:
    """Apply ONE chunk of an exit-order fill, decrementing shares and
    accumulating exit proceeds.  Returns a status dict the caller uses
    to decide whether to log a closed-position activity entry.

    Two modes, gated on whether the cumulative sell would zero out shares:

      partial (shares > tolerance after decrement):
          shares             := shares - sold_shares
          exit_proceeds_usdc := exit_proceeds_usdc + sold_shares × fill_price
          exit_fees          := exit_fees + fee_usdc
          status stays 'exiting' (or 'open' if it was open)

      complete (shares ≤ tolerance after decrement):
          shares = 0
          exit_proceeds_usdc final
          exit_fees final
          status = 'closed'
          actual_exit_price = exit_proceeds_usdc / total_shares_sold (weighted avg)
          pnl = exit_proceeds_usdc - size_usdc - entry_fees - exit_fees

    Mirrors add_position_entry_fill / add_position_topup_fill — same
    accumulation pattern, opposite sign (shares decrement instead of
    increment).

    Returns dict:
      {"is_complete": bool, "shares_after": float, "pnl": float | None,
       "actual_exit_price": float | None}
    """
    from datetime import datetime, timezone as _tz
    a_shares = float(sold_shares)
    a_price  = float(fill_price)
    a_fee    = float(fee_usdc or 0)
    a_proceeds = round(a_shares * a_price, 6)

    with _get_conn() as conn:
        row = conn.execute("""
            SELECT shares, size_usdc, entry_price, entry_fees, exit_fees,
                   exit_proceeds_usdc, status
            FROM positions WHERE id = ?
        """, (position_id,)).fetchone()
        if row is None:
            return {"is_complete": False, "shares_after": 0.0,
                    "pnl": None, "actual_exit_price": None}

        prev_shares    = float(row["shares"]             or 0)
        size_usdc      = float(row["size_usdc"]          or 0)
        entry_fees     = float(row["entry_fees"]         or 0)
        prev_exit_fees = float(row["exit_fees"]          or 0)
        prev_proceeds  = float(row["exit_proceeds_usdc"] or 0)

        new_shares    = max(0.0, prev_shares - a_shares)
        new_proceeds  = prev_proceeds + a_proceeds
        new_exit_fees = prev_exit_fees + a_fee

        is_complete = new_shares <= _EXIT_COMPLETE_SHARE_TOLERANCE

        if is_complete:
            # Total shares sold across all chunks (entry-side shares minus
            # whatever's left, which is ~0).  Used to compute weighted-avg
            # exit price.  size_usdc / entry_price gives the original entry
            # share count if shares column had been mutated by prior partials.
            entry_share_count = (
                size_usdc / float(row["entry_price"])
                if (row["entry_price"] or 0) > 0 else prev_shares + a_shares
            )
            avg_exit_price = (
                new_proceeds / entry_share_count
                if entry_share_count > 0 else a_price
            )
            # Schema convention (preserved from legacy code):
            #   pnl      = GROSS realised pnl (proceeds - cost), pre-fees
            #   pnl_net  = NET realised pnl (gross - entry_fees - exit_fees)
            # Dashboard reads both; downstream consumers (closed-positions
            # list, daily P&L) historically use pnl_net for the bottom line
            # and pnl for the trading-headline number.
            gross_pnl = round(new_proceeds - size_usdc, 4)
            net_pnl   = round(gross_pnl - entry_fees - new_exit_fees, 4)
            now_iso = datetime.now(_tz.utc).astimezone().isoformat()
            conn.execute("""
                UPDATE positions
                SET shares             = 0,
                    status             = 'closed',
                    exit_proceeds_usdc = ?,
                    exit_fees          = ?,
                    actual_exit_price  = ?,
                    exit_price         = ?,
                    exit_time          = COALESCE(exit_time, ?),
                    pnl                = ?,
                    pnl_net            = ?,
                    unrealized_pnl     = 0
                WHERE id = ?
            """, (
                new_proceeds, new_exit_fees,
                avg_exit_price, avg_exit_price,
                now_iso, gross_pnl, net_pnl, position_id,
            ))
            return {
                "is_complete":       True,
                "shares_after":      0.0,
                "gross_pnl":         gross_pnl,
                "net_pnl":           net_pnl,
                "actual_exit_price": avg_exit_price,
            }
        else:
            # Partial — accumulate, don't close.  Position may still be
            # 'exiting' (ladder mid-flight) or 'open' (rare, but if a
            # manual sell was applied via this path).
            conn.execute("""
                UPDATE positions
                SET shares             = ?,
                    exit_proceeds_usdc = ?,
                    exit_fees          = ?
                WHERE id = ?
            """, (new_shares, new_proceeds, new_exit_fees, position_id))
            return {
                "is_complete":       False,
                "shares_after":      new_shares,
                "pnl":               None,
                "actual_exit_price": None,
            }


def get_underfilled_positions() -> list[dict]:
    """Return open positions where size_usdc < target_size_usdc."""
    sql = """
        SELECT * FROM positions
        WHERE status = 'open' AND fill_status = 'filled'
          AND target_size_usdc IS NOT NULL
          AND size_usdc < target_size_usdc
        ORDER BY entry_time ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def update_position_excursions(
    position_id: int,
    max_favorable: float | None,
    max_adverse: float | None,
) -> None:
    """Track peak-to-trough during position lifetime for attribution."""
    sql = """
        UPDATE positions SET
            max_favorable_excursion = CASE
                WHEN max_favorable_excursion IS NULL THEN ?
                WHEN ? > max_favorable_excursion THEN ?
                ELSE max_favorable_excursion END,
            max_adverse_excursion = CASE
                WHEN max_adverse_excursion IS NULL THEN ?
                WHEN ? < max_adverse_excursion THEN ?
                ELSE max_adverse_excursion END
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (max_favorable, max_favorable, max_favorable,
                           max_adverse, max_adverse, max_adverse,
                           position_id))


def update_position_fill(
    position_id: int,
    fill_status: str,
    shares: float,
    entry_price: float,
) -> None:
    """Update a pending position once fill is confirmed via CLOB API.

    REPLACE semantics — sets shares to the given value.  Suitable when
    the caller already has the cumulative (final) fill totals.  For
    per-chunk additive accumulation (which is what the WS fill path now
    needs since each match emits its own trade event), use
    add_position_entry_fill() instead.

    Also recomputes size_usdc = shares × entry_price so it reflects the
    ACTUAL filled cost, not the originally-intended one.  Critical for
    partial fills: a $10 buy that only matched 7 shares at $0.29 has a
    real cost of $2.03, not $10 — exposure caps and P&L math both depend
    on this.  Without the recompute, MAX_TOTAL_EXPOSURE_PCT would treat
    the position as 5x larger than it actually is on chain.
    """
    actual_size_usdc = round(float(shares) * float(entry_price), 4)
    sql = """
        UPDATE positions
        SET fill_status = ?, shares = ?, entry_price = ?,
            current_price = ?, size_usdc = ?
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (fill_status, shares, entry_price,
                           entry_price, actual_size_usdc, position_id))


def add_position_entry_fill(
    position_id: int,
    added_shares: float,
    fill_price: float,
) -> None:
    """Apply ONE chunk of an entry-order fill, accumulating across chunks.

    Replaces update_position_fill on the WS fill path so that entry
    orders that match against multiple resting asks (yielding N trade
    events for the same order_id) accumulate correctly.

    Two modes, gated on the position's current fill_status:

      fill_status == 'pending'  (no chunks have landed yet):
          REPLACE shares/size_usdc/entry_price with this chunk's values
          and flip fill_status -> 'filled'.  The seeded shares/size_usdc
          from insert_position are intended estimates, not actuals — so
          we discard them in favour of the on-chain truth.

      fill_status == 'filled'   (a previous chunk already landed):
          shares      := shares    + added_shares
          size_usdc   := size_usdc + added_shares × fill_price
          entry_price := size_usdc / shares   (weighted average cost basis)

    The weighted-average recompute mirrors the topup path and keeps
    cost-basis honest when chunks land at different prices (e.g. a
    sweep walks through the book paying $0.34, $0.35, $0.36).

    No-op for cancelled / closed / non-existent positions.
    """
    a_shares = float(added_shares)
    a_price  = float(fill_price)
    a_usdc   = round(a_shares * a_price, 6)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT shares, size_usdc, fill_status "
            "FROM positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            return
        cur_status = (row["fill_status"] or "").lower()
        if cur_status not in ("pending", "filled"):
            # cancelled / closed / unknown — don't touch.
            return
        if cur_status == "pending":
            # First chunk lands.  Discard the intended-shares seed
            # (set by insert_position) and adopt this chunk's truth.
            new_shares = a_shares
            new_usdc   = a_usdc
            new_avg    = a_price
        else:
            prev_shares = float(row["shares"]    or 0)
            prev_usdc   = float(row["size_usdc"] or 0)
            new_shares  = prev_shares + a_shares
            new_usdc    = prev_usdc   + a_usdc
            new_avg     = (new_usdc / new_shares) if new_shares > 0 else a_price
        conn.execute("""
            UPDATE positions
            SET fill_status   = 'filled',
                shares        = ?,
                size_usdc     = ?,
                entry_price   = ?,
                current_price = ?
            WHERE id = ?
        """, (new_shares, new_usdc, new_avg, new_avg, position_id))


def mark_event_processed(event_id: str) -> bool:
    """Atomic 'have we already processed this trade event?' check.

    INSERT OR IGNORE into processed_trade_events; returns True if the
    row was inserted (first time seeing this event_id) and False if it
    was already there (duplicate).  Cheap (~one indexed PRIMARY KEY
    write or noop), called once per incoming WS+REST trade event.

    Empty/None event_id returns True without recording — callers should
    fall back to whatever pre-existing dedup the lifecycle column gives
    them.  In practice Polymarket events always carry an `id`.
    """
    if not event_id:
        return True
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO processed_trade_events "
            "(event_id, processed_at) VALUES (?, ?)",
            (str(event_id), now),
        )
        return cur.rowcount > 0


def cancel_position(
    position_id: int,
    cancelled_reason: str,
    exit_time: str,
) -> None:
    """Mark a position as cancelled (unfilled order, expired, etc.)."""
    sql = """
        UPDATE positions
        SET status = 'closed', fill_status = 'cancelled',
            cancelled_reason = ?, exit_time = ?, pnl = 0
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (cancelled_reason, exit_time, position_id))


def update_position_market_price(
    position_id: int,
    current_price: float,
    unrealized_pnl: float,
    local_time: str = None,
) -> None:
    """Refresh current market price, unrealized P&L, and local time for an open position."""
    sql = """
        UPDATE positions
        SET current_price = ?, unrealized_pnl = ?, local_time = COALESCE(?, local_time)
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (current_price, unrealized_pnl, local_time, position_id))


def backfill_gamma_market_ids(outcomes: list[dict]) -> int:
    """
    For any open position with NULL gamma_market_id, attempt to fill it by
    matching contract_id against a list of outcome dicts (from the discovery
    pipeline).  Each outcome dict must have 'contract_id' and 'gamma_market_id'.
    Returns the number of rows updated.
    """
    if not outcomes:
        return 0
    # Build a lookup: conditionId → numeric gamma_market_id
    lookup = {
        o["contract_id"]: o["gamma_market_id"]
        for o in outcomes
        if o.get("contract_id") and o.get("gamma_market_id")
    }
    if not lookup:
        return 0
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, contract_id FROM positions WHERE gamma_market_id IS NULL AND status = 'open'"
        ).fetchall()
        updated = 0
        for row in rows:
            gid = lookup.get(row[1])
            if gid:
                conn.execute(
                    "UPDATE positions SET gamma_market_id = ? WHERE id = ?",
                    (str(gid), row[0]),
                )
                updated += 1
    return updated


def backfill_position_coords() -> int:
    """
    For any position with NULL lat/lon, attempt to fill from the CITY_COORDS
    lookup table in polymarket.py.  Returns the number of rows updated.

    This is a one-time migration helper; it is safe to call repeatedly.
    """
    try:
        from polymarket import CITY_COORDS
    except ImportError:
        return 0

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, city FROM positions WHERE lat IS NULL OR lon IS NULL"
        ).fetchall()
        updated = 0
        for row in rows:
            pos_id = row[0]
            city   = (row[1] or "").lower().strip()
            coords = CITY_COORDS.get(city)
            if coords:
                conn.execute(
                    "UPDATE positions SET lat = ?, lon = ? WHERE id = ?",
                    (coords[0], coords[1], pos_id),
                )
                updated += 1
    return updated


def backfill_position_sigma() -> int:
    """
    For any position with NULL forecast_sigma_c, attempt to fill it by joining
    positions.contract_id + positions.scan_timestamp against temp_outcomes and
    temp_events (which store forecast_sigma_c per scan).

    Falls back to matching on contract_id alone (any scan) when the
    scan_timestamp is missing or the exact scan row has been purged.

    Returns the number of rows updated.  Safe to call repeatedly.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, contract_id, scan_timestamp FROM positions "
            "WHERE forecast_sigma_c IS NULL"
        ).fetchall()
        if not rows:
            return 0

        updated = 0
        for pos_id, contract_id, scan_ts in rows:
            sigma = None

            # Preferred: exact scan_timestamp match
            if scan_ts:
                row = conn.execute(
                    """
                    SELECT e.forecast_sigma_c
                    FROM temp_outcomes o
                    JOIN temp_events e ON o.event_row_id = e.id
                    WHERE o.contract_id = ? AND o.scan_timestamp = ?
                    LIMIT 1
                    """,
                    (contract_id, scan_ts),
                ).fetchone()
                if row:
                    sigma = row[0]

            # Fallback: any scan for this contract
            if sigma is None:
                row = conn.execute(
                    """
                    SELECT e.forecast_sigma_c
                    FROM temp_outcomes o
                    JOIN temp_events e ON o.event_row_id = e.id
                    WHERE o.contract_id = ?
                    ORDER BY o.scan_timestamp DESC
                    LIMIT 1
                    """,
                    (contract_id,),
                ).fetchone()
                if row:
                    sigma = row[0]

            if sigma is not None:
                conn.execute(
                    "UPDATE positions SET forecast_sigma_c = ? WHERE id = ?",
                    (sigma, pos_id),
                )
                updated += 1

    return updated


def get_daily_pnl(for_date: str = None) -> float:
    """
    Return today's P&L: realized (closed positions) + unrealized (open filled positions).
    Used by check_daily_drawdown so that a deeply underwater open position
    still counts against the drawdown limit, not just closed trades.
    """
    if for_date is None:
        for_date = date.today().isoformat()
    with _get_conn() as conn:
        realized = conn.execute(
            """
            SELECT COALESCE(SUM(pnl), 0.0) FROM positions
            WHERE status = 'closed' AND fill_status != 'cancelled'
              AND DATE(exit_time) = ?
            """,
            (for_date,),
        ).fetchone()[0]
        unrealized = conn.execute(
            """
            SELECT COALESCE(SUM(unrealized_pnl), 0.0) FROM positions
            WHERE status = 'open' AND fill_status = 'filled'
            """,
        ).fetchone()[0]
    return realized + unrealized


# ---------------------------------------------------------------------------
# Temperature event helpers
# ---------------------------------------------------------------------------

def insert_temp_event(event: dict, scan_timestamp: str) -> int:
    """Insert a top-level temperature event record. Returns the new row ID."""
    sql = """
        INSERT INTO temp_events (
            scan_timestamp, event_id, event_title, city, date, lat, lon,
            forecast_mu_c, forecast_sigma_c, clim_mu_c, clim_sigma_c,
            forecast_mu_display, display_unit, days_ahead,
            market_overround, model_probs_sum, normalization_warning,
            n_outcomes, n_sources,
            adjusted_mu_c, adjusted_sigma_c,
            live_adjustment_score, live_adjustment_components
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            scan_timestamp,
            event.get("event_id"),
            event.get("event_title"),
            event.get("city"),
            event.get("date"),
            event.get("lat"),
            event.get("lon"),
            event.get("forecast_mu_c"),
            event.get("forecast_sigma_c"),
            event.get("clim_mu_c"),
            event.get("clim_sigma_c"),
            event.get("forecast_mu_display"),
            event.get("display_unit"),
            event.get("days_ahead"),
            event.get("market_overround"),
            event.get("model_probs_sum"),
            1 if event.get("normalization_warning") else 0,
            event.get("n_outcomes", 0),
            event.get("n_sources", 0),
            event.get("adjusted_mu_c"),
            event.get("adjusted_sigma_c"),
            event.get("live_adjustment_score"),
            event.get("live_adjustment_components"),
        ))
        return cur.lastrowid


def insert_temp_outcome(outcome: dict, event_row_id: int, scan_timestamp: str) -> int:
    """Insert a single temperature-range outcome linked to its parent event."""
    sql = """
        INSERT INTO temp_outcomes (
            event_row_id, scan_timestamp, contract_id, question,
            range_low, range_high, unit,
            market_price, yes_price, no_price,
            model_prob, raw_model_prob, ev, edge,
            recommended_side, kelly_size, is_signal,
            liquidity_usd, volume_usd, yes_token_id, no_token_id,
            model_prob_blended
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    yes_price = outcome.get("yes_price", outcome.get("market_price"))
    no_price  = outcome.get("no_price",  1.0 - float(yes_price) if yes_price is not None else None)
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            event_row_id,
            scan_timestamp,
            outcome.get("contract_id"),
            outcome.get("question"),
            outcome.get("range_low"),
            outcome.get("range_high"),
            outcome.get("unit"),
            outcome.get("market_price"),
            yes_price,
            no_price,
            outcome.get("model_prob"),
            outcome.get("raw_model_prob"),
            outcome.get("ev"),
            outcome.get("edge"),
            outcome.get("recommended_side"),
            outcome.get("kelly_size"),
            1 if outcome.get("is_signal") else 0,
            outcome.get("liquidity_usd"),
            outcome.get("volume_usd"),
            outcome.get("yes_token_id"),
            outcome.get("no_token_id"),
            outcome.get("model_prob_blended"),
        ))
        return cur.lastrowid


def purge_scan_data() -> dict:
    """
    Delete all rows from temp_events and temp_outcomes.

    Use this after a code fix that affects stored probabilities or forecast
    values, so the next bot run starts clean.

    What is deleted:
        temp_events   — event-level scan records (city, date, forecast μ/σ,
                        clim μ/σ, overround, days_ahead, etc.)
        temp_outcomes — per-bin records (model_prob, market_price, yes_price,
                        no_price, edge, EV, is_signal, etc.)

    What is NOT deleted:
        era5_daily      — historical climate cache (correct data, expensive to rebuild)
        forecast_errors — ECMWF bias tracking (accumulates over weeks)
        signals         — legacy signal log
        positions       — trade records

    After calling this, run the bot once to repopulate both tables with
    corrected probability data.

    Returns:
        {"events_deleted": int, "outcomes_deleted": int}
    """
    with _get_conn() as conn:
        outcomes_deleted = conn.execute("DELETE FROM temp_outcomes").rowcount
        events_deleted   = conn.execute("DELETE FROM temp_events").rowcount
    return {"events_deleted": events_deleted, "outcomes_deleted": outcomes_deleted}


def get_latest_scan_timestamp() -> str | None:
    """Return the most recent scan_timestamp stored in temp_events."""
    sql = "SELECT MAX(scan_timestamp) FROM temp_events"
    with _get_conn() as conn:
        row = conn.execute(sql).fetchone()
        return row[0]


def get_latest_temp_events(limit: int = 300) -> list[dict]:
    """Return all event rows from the most recent scan."""
    ts = get_latest_scan_timestamp()
    if not ts:
        return []
    sql = """
        SELECT * FROM temp_events
        WHERE scan_timestamp = ?
        ORDER BY city, date
        LIMIT ?
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, (ts, limit)).fetchall()
        return [dict(r) for r in rows]


def get_latest_temp_outcomes(signals_only: bool = False, limit: int = 2000) -> list[dict]:
    """Return all outcome rows from the most recent scan, optionally filtered to signals."""
    ts = get_latest_scan_timestamp()
    if not ts:
        return []
    sql = """
        SELECT o.*, e.city, e.date, e.forecast_mu_c, e.forecast_sigma_c,
               e.clim_mu_c, e.display_unit, e.event_title, e.days_ahead,
               e.normalization_warning
        FROM temp_outcomes o
        JOIN temp_events e ON o.event_row_id = e.id
        WHERE o.scan_timestamp = ?
        {}
        ORDER BY o.ev DESC
        LIMIT ?
    """.format("AND o.is_signal = 1" if signals_only else "")
    with _get_conn() as conn:
        rows = conn.execute(sql, (ts, limit)).fetchall()
        return [dict(r) for r in rows]


def mark_outcome_executed(contract_id: str, scan_timestamp: str) -> None:
    sql = """
        UPDATE temp_outcomes SET executed = 1
        WHERE contract_id = ? AND scan_timestamp = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (contract_id, scan_timestamp))


# ---------------------------------------------------------------------------
# ERA5 persistent cache helpers
# ---------------------------------------------------------------------------

def _loc_keys(lat: float, lon: float) -> tuple[float, float]:
    """Canonical location key — rounded to 2 dp to group nearby API calls."""
    return (round(lat, 2), round(lon, 2))


def get_era5_dates_present(lat: float, lon: float, dates: list[str]) -> set[str]:
    """
    Return the subset of `dates` already stored in era5_daily for this location.
    Used to compute the minimal fetch list before calling the Archive API.
    """
    if not dates:
        return set()
    lat_key, lon_key = _loc_keys(lat, lon)
    placeholders = ",".join("?" * len(dates))
    sql = f"""
        SELECT date FROM era5_daily
        WHERE lat_key = ? AND lon_key = ? AND date IN ({placeholders})
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, [lat_key, lon_key] + list(dates)).fetchall()
        return {r[0] for r in rows}


def get_era5_values(lat: float, lon: float, dates: list[str]) -> dict[str, float]:
    """
    Return {date: tmax_c} for all requested dates present in era5_daily.
    """
    if not dates:
        return {}
    lat_key, lon_key = _loc_keys(lat, lon)
    placeholders = ",".join("?" * len(dates))
    sql = f"""
        SELECT date, tmax_c FROM era5_daily
        WHERE lat_key = ? AND lon_key = ? AND date IN ({placeholders})
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, [lat_key, lon_key] + list(dates)).fetchall()
        return {r[0]: r[1] for r in rows}


def insert_era5_rows(lat: float, lon: float, date_tmax: dict[str, float]) -> int:
    """
    Bulk-insert ERA5 daily max temperature rows. Uses INSERT OR IGNORE so
    re-fetching the same dates is safe (no duplicates, no errors).
    Returns the number of rows newly inserted.
    """
    if not date_tmax:
        return 0
    lat_key, lon_key = _loc_keys(lat, lon)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT OR IGNORE INTO era5_daily (lat_key, lon_key, date, tmax_c, fetched_at)
        VALUES (?, ?, ?, ?, ?)
    """
    rows = [(lat_key, lon_key, d, v, now) for d, v in date_tmax.items()]
    with _get_conn() as conn:
        conn.executemany(sql, rows)
        # SQLite doesn't expose per-executemany row count cleanly; approximate
        return len(rows)


def count_era5_rows(lat: float, lon: float) -> int:
    """Return total ERA5 rows stored for a location (diagnostic / dashboard use)."""
    lat_key, lon_key = _loc_keys(lat, lon)
    sql = "SELECT COUNT(*) FROM era5_daily WHERE lat_key = ? AND lon_key = ?"
    with _get_conn() as conn:
        return conn.execute(sql, (lat_key, lon_key)).fetchone()[0]


# ---------------------------------------------------------------------------
# Forecast error / bias correction helpers
# ---------------------------------------------------------------------------

def insert_forecast_error(
    lat: float, lon: float, city: str,
    target_date: str, days_ahead: int,
    forecast_mu_c: float, actual_tmax_c: float,
    model: str = "ecmwf_ifs025",
) -> None:
    """
    Record one forecast-vs-actual comparison.
    Called by bias.py after a contract resolves and ERA5 data is available.

    `model` defaults to 'ecmwf_ifs025' to match the backfilled rows produced
    by scripts/backfill_bias_data.py.  Pass a different model id (e.g.
    'gfs_global') if recording GFS errors separately.
    """
    lat_key, lon_key = _loc_keys(lat, lon)
    from datetime import datetime, timezone
    month_day = target_date[5:]   # MM-DD
    error_c   = actual_tmax_c - forecast_mu_c
    now       = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO forecast_errors
            (lat_key, lon_key, city, calendar_month_day, target_date,
             days_ahead, forecast_mu_c, actual_tmax_c, error_c, recorded_at,
             model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        conn.execute(sql, (
            lat_key, lon_key, city, month_day, target_date,
            days_ahead, forecast_mu_c, actual_tmax_c, error_c, now,
            model,
        ))


def get_bias_correction(
    lat: float, lon: float,
    calendar_month_day: str,
    window_days: int = 30,
    min_observations: int = 10,
    model: str = "ecmwf",
) -> tuple[float, int]:
    """
    Compute the mean forecast error for a location + model over a rolling
    calendar window centred on `calendar_month_day` (MM-DD format).

    We use a ±window_days calendar window (ignoring year) to capture
    seasonal bias rather than one calendar date in isolation.

    Returns (bias_correction_c, n_observations).
        bias_correction_c > 0 means the model historically runs too cold → add to forecast
        bias_correction_c < 0 means the model historically runs too warm → subtract
    Returns (0.0, 0) if fewer than min_observations are available.

    `model` filters forecast_errors by the model column.  Legacy rows inserted
    before the column existed default to 'ecmwf'.
    """
    lat_key, lon_key = _loc_keys(lat, lon)

    # Build a list of MM-DD strings within ±window_days of the target
    from datetime import date as _date, timedelta
    try:
        month = int(calendar_month_day[:2])
        day   = int(calendar_month_day[3:])
        anchor = _date(2000, month, day)   # leap-year-safe reference year
    except ValueError:
        return (0.0, 0)

    window_dates = set()
    for delta in range(-window_days, window_days + 1):
        d = anchor + timedelta(days=delta)
        window_dates.add(f"{d.month:02d}-{d.day:02d}")

    placeholders = ",".join("?" * len(window_dates))
    sql = f"""
        SELECT error_c FROM forecast_errors
        WHERE lat_key = ? AND lon_key = ?
          AND COALESCE(model, 'ecmwf') = ?
          AND calendar_month_day IN ({placeholders})
        ORDER BY recorded_at DESC
        LIMIT 2000
    """
    with _get_conn() as conn:
        rows = conn.execute(
            sql, [lat_key, lon_key, model] + list(window_dates)
        ).fetchall()

    errors = [r[0] for r in rows if r[0] is not None]
    n = len(errors)
    if n < min_observations:
        return (0.0, n)
    return (sum(errors) / n, n)


# ---------------------------------------------------------------------------
# Upsert helpers for the VC + OM backfill caches
# ---------------------------------------------------------------------------

def upsert_historical_observed_daily(
    lat: float, lon: float, city: str | None, rows: list[dict]
) -> int:
    """
    Insert/replace Visual Crossing daily observation rows.

    `rows` items must have: date, tempmax_c, tempmin_c, temp_c, stations (list).
    Returns number of rows written.
    """
    if not rows:
        return 0
    import json
    lat_key, lon_key = _loc_keys(lat, lon)
    now = datetime.utcnow().isoformat() + "Z"
    sql = """
        INSERT INTO historical_observed_daily
            (lat_key, lon_key, city, date, tempmax_c, tempmin_c, temp_c,
             stations, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lat_key, lon_key, date) DO UPDATE SET
            city       = excluded.city,
            tempmax_c  = excluded.tempmax_c,
            tempmin_c  = excluded.tempmin_c,
            temp_c     = excluded.temp_c,
            stations   = excluded.stations,
            fetched_at = excluded.fetched_at
    """
    with _get_conn() as conn:
        for r in rows:
            conn.execute(sql, (
                lat_key, lon_key, city, r["date"],
                r.get("tempmax_c"), r.get("tempmin_c"), r.get("temp_c"),
                json.dumps(r.get("stations") or []),
                now,
            ))
    return len(rows)


def upsert_historical_forecasts_previous_runs(
    lat: float, lon: float, city: str | None, rows: list[dict]
) -> int:
    """
    Insert/replace Open-Meteo Previous Runs forecast rows.

    `rows` items must have: date, model, lead_days, forecast_tempmax_c, n_hours.
    Returns number of rows written.
    """
    if not rows:
        return 0
    lat_key, lon_key = _loc_keys(lat, lon)
    now = datetime.utcnow().isoformat() + "Z"
    sql = """
        INSERT INTO historical_forecasts_previous_runs
            (lat_key, lon_key, city, date, model, lead_days,
             forecast_tempmax_c, n_hours, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lat_key, lon_key, date, model, lead_days) DO UPDATE SET
            city               = excluded.city,
            forecast_tempmax_c = excluded.forecast_tempmax_c,
            n_hours            = excluded.n_hours,
            fetched_at         = excluded.fetched_at
    """
    with _get_conn() as conn:
        for r in rows:
            conn.execute(sql, (
                lat_key, lon_key, city, r["date"], r["model"], r["lead_days"],
                r.get("forecast_tempmax_c"), r.get("n_hours"),
                now,
            ))
    return len(rows)


def rebuild_forecast_errors_from_cache(
    lat: float, lon: float,
    city: str | None = None,
    lead_days: int = 3,
) -> int:
    """
    Join historical_observed_daily + historical_forecasts_previous_runs for
    one location, compute error = actual - forecast, and write rows into
    forecast_errors (one per (date, model) pair).

    Idempotent: deletes any existing backfilled forecast_errors rows for the
    location before re-inserting, so re-running this after an updated cache
    won't create duplicates.

    A "backfilled" row is distinguished from a live-recorded row by having
    lead_days_bucket matching the lead_days argument AND being written by
    this function.  We key the delete on (lat_key, lon_key, lead_days_bucket).

    Returns the number of forecast_errors rows written.
    """
    lat_key, lon_key = _loc_keys(lat, lon)
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        # Remove any previously-backfilled rows for this (location, lead) so
        # we can re-insert cleanly from the updated cache.
        conn.execute(
            "DELETE FROM forecast_errors "
            "WHERE lat_key = ? AND lon_key = ? AND lead_days_bucket = ?",
            (lat_key, lon_key, lead_days),
        )

        # Join the two caches and insert a forecast_errors row per match.
        cursor = conn.execute("""
            SELECT f.date, f.model, f.lead_days, f.forecast_tempmax_c,
                   o.tempmax_c
            FROM historical_forecasts_previous_runs f
            JOIN historical_observed_daily o
              ON f.lat_key = o.lat_key
             AND f.lon_key = o.lon_key
             AND f.date    = o.date
            WHERE f.lat_key = ? AND f.lon_key = ?
              AND f.lead_days = ?
              AND f.forecast_tempmax_c IS NOT NULL
              AND o.tempmax_c IS NOT NULL
        """, (lat_key, lon_key, lead_days))

        inserted = 0
        for date_str, model, lead, fcst, actual in cursor.fetchall():
            error_c = actual - fcst
            cmd     = date_str[5:]   # MM-DD
            conn.execute("""
                INSERT INTO forecast_errors
                    (lat_key, lon_key, city, calendar_month_day, target_date,
                     days_ahead, forecast_mu_c, actual_tmax_c, error_c,
                     recorded_at, model, lead_days_bucket)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lat_key, lon_key, city, cmd, date_str,
                lead, fcst, actual, error_c, now, model, lead,
            ))
            inserted += 1

    return inserted


# ---------------------------------------------------------------------------
# Phase 2 — forecast_runs / forecast_hourly helpers
# ---------------------------------------------------------------------------

def insert_forecast_run(
    event_id: str,
    city: str | None,
    date: str,
    lat: float | None,
    lon: float | None,
    source: str,
    pulled_at: str,
    forecast_mu_c: float | None,
    forecast_sigma_c: float | None,
    forecast_high_c: float | None = None,
    days_ahead: int | None = None,
    model_run_ts: str | None = None,
    raw_json: str | None = None,
) -> int:
    """Insert one forecast_runs row. Returns the new run_id.

    UNIQUE(event_id, source, pulled_at) — re-insertions with the same timestamp
    are ignored and the existing row id is returned.
    """
    sql = """
        INSERT OR IGNORE INTO forecast_runs
            (event_id, city, date, lat, lon, source, pulled_at,
             model_run_ts, forecast_mu_c, forecast_sigma_c,
             forecast_high_c, days_ahead, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            event_id, city, date, lat, lon, source, pulled_at,
            model_run_ts, forecast_mu_c, forecast_sigma_c,
            forecast_high_c, days_ahead, raw_json,
        ))
        if cur.lastrowid:
            return cur.lastrowid
        # Row already existed (IGNORE hit) — look it up
        row = conn.execute(
            "SELECT id FROM forecast_runs "
            "WHERE event_id = ? AND source = ? AND pulled_at = ?",
            (event_id, source, pulled_at),
        ).fetchone()
        return int(row[0]) if row else 0


def insert_forecast_hourly_bulk(run_id: int, hours: list[dict]) -> int:
    """Bulk-insert hourly rows for a forecast_runs row.

    Each `hours` dict may include: hour_ts_utc, hour_ts_local, is_target_day,
    temp_c, humidity, cloudcover, windspeed_kph, precip_mm, precip_prob,
    conditions.  hour_ts_utc is required.
    """
    if not hours:
        return 0
    sql = """
        INSERT OR IGNORE INTO forecast_hourly
            (run_id, hour_ts_utc, hour_ts_local, is_target_day,
             temp_c, humidity, cloudcover, windspeed_kph,
             precip_mm, precip_prob, conditions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [(
        run_id,
        h.get("hour_ts_utc"),
        h.get("hour_ts_local"),
        1 if h.get("is_target_day") else 0,
        h.get("temp_c"),
        h.get("humidity"),
        h.get("cloudcover"),
        h.get("windspeed_kph"),
        h.get("precip_mm"),
        h.get("precip_prob"),
        h.get("conditions"),
    ) for h in hours]
    with _get_conn() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def get_latest_forecast_run(event_id: str, source: str) -> dict | None:
    sql = """
        SELECT * FROM forecast_runs
        WHERE event_id = ? AND source = ?
        ORDER BY pulled_at DESC
        LIMIT 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (event_id, source)).fetchone()
        return dict(row) if row else None


def get_previous_forecast_run(event_id: str, source: str) -> dict | None:
    """Return the second-most-recent forecast_runs row (prior pull)."""
    sql = """
        SELECT * FROM forecast_runs
        WHERE event_id = ? AND source = ?
        ORDER BY pulled_at DESC
        LIMIT 1 OFFSET 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (event_id, source)).fetchone()
        return dict(row) if row else None


def get_latest_forecast_distribution(
    lat: float, lon: float, date_str: str, source: str,
    max_age_hours: float = 3.0,
) -> dict | None:
    """Read the most recent forecast_runs row for a (location, date, source).

    Returns {"mu_c", "sigma_c", "source", "n", "pulled_at"} matching the
    format of _get_ecmwf_ensemble_distribution / _get_gfs_ensemble_distribution,
    or None if no data exists within the age window.

    Used by the trading scan to avoid redundant API calls when the forecast
    pull loop already wrote fresh data 5 minutes ago.
    """
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    sql = """
        SELECT forecast_mu_c, forecast_sigma_c, source, pulled_at
        FROM forecast_runs
        WHERE ABS(lat - ?) < 0.05 AND ABS(lon - ?) < 0.05
          AND date = ? AND source = ?
          AND pulled_at >= ?
          AND forecast_mu_c IS NOT NULL
        ORDER BY pulled_at DESC
        LIMIT 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (lat, lon, date_str, source, cutoff)).fetchone()
        if not row:
            return None
        return {
            "mu_c":      float(row[0]),
            "sigma_c":   float(row[1]),
            "source":    row[2],
            "n":         0,
            "pulled_at": row[3],
        }


def get_latest_forecast_hourly(run_id: int) -> list[dict]:
    sql = """
        SELECT * FROM forecast_hourly
        WHERE run_id = ?
        ORDER BY hour_ts_utc ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (run_id,)).fetchall()]


# ---------------------------------------------------------------------------
# Phase 2 — live_observations helpers
# ---------------------------------------------------------------------------

def insert_live_observation(
    event_id: str,
    city: str | None,
    date: str,
    lat: float | None,
    lon: float | None,
    pulled_at_utc: str,
    observed_at_utc: str | None,
    observed_at_local: str | None,
    vc_source: str | None,
    current_temp_c: float | None,
    humidity: float | None = None,
    cloudcover: float | None = None,
    windspeed_kph: float | None = None,
    precip_mm: float | None = None,
    conditions: str | None = None,
    observed_max_so_far_c: float | None = None,
    stations: str | None = None,
    query_cost: int | None = None,
    # Phase ML-v1 — expanded VC observation features
    feelslike_c: float | None = None,
    dew_c: float | None = None,
    pressure_hpa: float | None = None,
    visibility_km: float | None = None,
    windgust_kph: float | None = None,
    winddir_deg: float | None = None,
    preciptype: str | None = None,
    snow_cm: float | None = None,
    snowdepth_cm: float | None = None,
    solarradiation_wm2: float | None = None,
    solarenergy_mj: float | None = None,
    uvindex: float | None = None,
) -> int:
    sql = """
        INSERT INTO live_observations
            (event_id, city, date, lat, lon, pulled_at_utc,
             observed_at_utc, observed_at_local, vc_source,
             current_temp_c, humidity, cloudcover, windspeed_kph,
             precip_mm, conditions, observed_max_so_far_c,
             stations, query_cost,
             feelslike_c, dew_c, pressure_hpa, visibility_km,
             windgust_kph, winddir_deg, preciptype,
             snow_cm, snowdepth_cm,
             solarradiation_wm2, solarenergy_mj, uvindex)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            event_id, city, date, lat, lon, pulled_at_utc,
            observed_at_utc, observed_at_local, vc_source,
            current_temp_c, humidity, cloudcover, windspeed_kph,
            precip_mm, conditions, observed_max_so_far_c,
            stations, query_cost,
            feelslike_c, dew_c, pressure_hpa, visibility_km,
            windgust_kph, winddir_deg, preciptype,
            snow_cm, snowdepth_cm,
            solarradiation_wm2, solarenergy_mj, uvindex,
        ))
        return cur.lastrowid


def get_recent_observations(event_id: str, since_utc: str) -> list[dict]:
    sql = """
        SELECT * FROM live_observations
        WHERE event_id = ? AND pulled_at_utc >= ?
        ORDER BY pulled_at_utc ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (event_id, since_utc)).fetchall()]


def get_latest_observation(event_id: str) -> dict | None:
    sql = """
        SELECT * FROM live_observations
        WHERE event_id = ?
        ORDER BY pulled_at_utc DESC
        LIMIT 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (event_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Phase 2 — decision_snapshots helpers
# ---------------------------------------------------------------------------

def insert_decision_snapshot(snapshot: dict) -> int:
    """Insert one decision_snapshots row.  All fields are optional except
    event_snapshot_group_id, event_id, contract_id, evaluated_at_utc."""
    cols = [
        "event_snapshot_group_id", "event_id", "contract_id", "city", "date",
        "evaluated_at_utc",
        "latest_ecmwf_run_id", "latest_gfs_run_id",
        "blended_mu_c", "blended_sigma_c",
        "latest_obs_id", "current_temp_c",
        "temp_change_1h_c", "temp_change_3h_c",
        "observed_max_so_far_c", "forecast_remaining_max_c", "projected_day_max_c",
        "expected_temp_now_c", "actual_minus_expected_c",
        "forecast_delta_mu_c", "forecast_delta_sigma_c", "forecast_agreement_c",
        "market_price", "model_prob", "raw_model_prob",
        "edge", "ev", "recommended_side", "kelly_size", "liquidity_usd",
        "action", "reason",
        "adjusted_mu_c", "adjusted_sigma_c",
        "live_adjustment_score", "live_adjustment_components",
        "obs_floor_applied",
        "vc_projected_day_max_c", "vc_vs_blended_mu_c", "vc_vs_adjusted_mu_c",
        "vc_hourly_path_rmse_c", "vc_bins_apart",
        "flag_vc_disagreement_large", "flag_vc_warns_hotter", "flag_vc_warns_colder",
        # Phase ML-v1 — ML distribution model shadow-log
        "ml_mu_c", "ml_sigma_c", "ml_model_version", "ml_weight_used",
    ]
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT INTO decision_snapshots ({','.join(cols)}) VALUES ({placeholders})"
    vals = [snapshot.get(c) for c in cols]
    with _get_conn() as conn:
        cur = conn.execute(sql, vals)
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Phase 2 — temp_events current-state updater
# ---------------------------------------------------------------------------

def get_snapshot_by_id(snapshot_id: int) -> dict | None:
    """Fetch a single decision_snapshots row by primary key."""
    sql = "SELECT * FROM decision_snapshots WHERE id = ?"
    with _get_conn() as conn:
        row = conn.execute(sql, (snapshot_id,)).fetchone()
        return dict(row) if row else None


def get_latest_snapshot_for_contract(contract_id: str) -> dict | None:
    """Return the most recent decision_snapshots row for a contract."""
    sql = ("SELECT * FROM decision_snapshots WHERE contract_id = ? "
           "ORDER BY evaluated_at_utc DESC LIMIT 1")
    with _get_conn() as conn:
        row = conn.execute(sql, (contract_id,)).fetchone()
        return dict(row) if row else None


def get_latest_snapshot_id_for_contract(contract_id: str) -> int | None:
    """Return the most recent decision_snapshots.id for a contract.
    Used by execution.py to populate positions.entry_snapshot_id."""
    sql = """
        SELECT id FROM decision_snapshots
        WHERE contract_id = ?
        ORDER BY evaluated_at_utc DESC
        LIMIT 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (contract_id,)).fetchone()
        return int(row[0]) if row else None


_EVENT_STATE_COLS = {
    "timezone", "latest_forecast_ts", "latest_observation_ts",
    "current_temp_c", "observed_max_so_far_c",
    "expected_temp_now_c", "actual_minus_expected_c",
    "forecast_delta_mu_c", "forecast_delta_sigma_c", "forecast_agreement_c",
}


def update_event_current_state(event_id: str, **fields) -> int:
    """Update one or more current-state columns on the most recent temp_events
    row for an event_id.  Silently drops unknown keys.  Returns rows affected."""
    clean = {k: v for k, v in fields.items() if k in _EVENT_STATE_COLS}
    if not clean:
        return 0
    assignments = ", ".join(f"{k} = ?" for k in clean)
    sql = f"""
        UPDATE temp_events
        SET {assignments}
        WHERE id = (
            SELECT id FROM temp_events
            WHERE event_id = ?
            ORDER BY scan_timestamp DESC
            LIMIT 1
        )
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, list(clean.values()) + [event_id])
        return cur.rowcount


# ---------------------------------------------------------------------------
# Phase 2 — retention helpers
# ---------------------------------------------------------------------------

def purge_forecast_hourly(older_than_days: int = 90) -> int:
    """Delete forecast_hourly rows whose parent forecast_runs.pulled_at is
    older than N days.  Returns rows deleted."""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    sql = """
        DELETE FROM forecast_hourly
        WHERE run_id IN (
            SELECT id FROM forecast_runs WHERE pulled_at < ?
        )
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (cutoff,))
        return cur.rowcount


def purge_live_observations(older_than_days: int = 90) -> int:
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    sql = "DELETE FROM live_observations WHERE pulled_at_utc < ?"
    with _get_conn() as conn:
        cur = conn.execute(sql, (cutoff,))
        return cur.rowcount


# ---------------------------------------------------------------------------
# Phase 2 — VC usage cost accounting
# ---------------------------------------------------------------------------

def insert_vc_forecast_diagnostic(row: dict) -> int:
    """Insert one vc_forecast_diagnostics row.  UNIQUE on (event_id,
    pulled_at_utc) — re-insertions with the same timestamp are ignored."""
    cols = [
        "event_id", "city", "target_date", "pulled_at_utc", "kind",
        "vc_projected_day_max_c", "vc_forecast_remaining_max_c", "vc_day_vc_source",
        "blended_mu_c", "blended_sigma_c", "adjusted_mu_c", "adjusted_sigma_c",
        "current_temp_c", "observed_max_so_far_c",
        "vc_vs_blended_mu_c", "vc_vs_adjusted_mu_c", "vc_vs_observed_max_c",
        "abs_vc_vs_blended", "abs_vc_vs_adjusted", "vc_bins_apart",
        "vc_hourly_path_rmse_c", "vc_hourly_path_n",
        "flag_vc_disagreement_large", "flag_vc_warns_hotter", "flag_vc_warns_colder",
        "vc_hourly_forecast_json",
    ]
    placeholders = ",".join("?" * len(cols))
    sql = (f"INSERT OR IGNORE INTO vc_forecast_diagnostics "
           f"({','.join(cols)}) VALUES ({placeholders})")
    vals = [row.get(c) for c in cols]
    with _get_conn() as conn:
        cur = conn.execute(sql, vals)
        return cur.lastrowid or 0


def get_latest_vc_forecast_diagnostic(
    event_id: str, since_utc: str | None = None,
) -> dict | None:
    """Return the most recent vc_forecast_diagnostics row for an event.
    Optionally restricted to rows newer than `since_utc` (ISO string)."""
    if since_utc:
        sql = ("SELECT * FROM vc_forecast_diagnostics "
               "WHERE event_id = ? AND pulled_at_utc >= ? "
               "ORDER BY pulled_at_utc DESC LIMIT 1")
        params = (event_id, since_utc)
    else:
        sql = ("SELECT * FROM vc_forecast_diagnostics "
               "WHERE event_id = ? "
               "ORDER BY pulled_at_utc DESC LIMIT 1")
        params = (event_id,)
    with _get_conn() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def purge_vc_forecast_diagnostics(older_than_days: int = 90) -> int:
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    sql = "DELETE FROM vc_forecast_diagnostics WHERE pulled_at_utc < ?"
    with _get_conn() as conn:
        cur = conn.execute(sql, (cutoff,))
        return cur.rowcount


def purge_temp_scan_data(older_than_days: int = 30) -> dict:
    """Purge old scan data from temp_outcomes and temp_events.

    Deletes outcomes first (FK dependency), then events.  Only removes
    rows whose scan_timestamp is older than the cutoff.  Positions,
    decision_snapshots, and other tables are unaffected.

    Returns {"events_deleted": int, "outcomes_deleted": int}.
    """
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    with _get_conn() as conn:
        outcomes = conn.execute(
            "DELETE FROM temp_outcomes WHERE scan_timestamp < ?", (cutoff,)
        ).rowcount
        events = conn.execute(
            "DELETE FROM temp_events WHERE scan_timestamp < ?", (cutoff,)
        ).rowcount
    return {"events_deleted": events, "outcomes_deleted": outcomes}


def rebuild_city_forecast_accuracy(window_days: int = 30) -> int:
    """Recompute per-city forecast accuracy metrics from forecast_errors
    + historical_observed_daily over a rolling window.  Called daily by
    the bias updater.  Returns the number of cities updated."""
    import math
    from datetime import date, timedelta, timezone

    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        # Get distinct cities with enough error data
        cities = conn.execute("""
            SELECT DISTINCT city, lat_key, lon_key
            FROM forecast_errors
            WHERE city IS NOT NULL AND target_date >= ?
            GROUP BY city
            HAVING COUNT(*) >= 5
        """, (cutoff,)).fetchall()

        updated = 0
        # Collect all scores for normalization pass
        all_metrics: list[dict] = []

        for city_row in cities:
            city = city_row[0]
            lat = city_row[1]
            lon = city_row[2]

            # All errors for this city in the window
            rows = conn.execute("""
                SELECT error_c, days_ahead
                FROM forecast_errors
                WHERE city = ? AND target_date >= ? AND error_c IS NOT NULL
            """, (city, cutoff)).fetchall()

            if len(rows) < 5:
                continue

            errors = [float(r[0]) for r in rows]
            abs_errors = [abs(e) for e in errors]
            n = len(errors)

            # Core
            mae = sum(abs_errors) / n
            rmse = math.sqrt(sum(e * e for e in errors) / n)
            bias = sum(errors) / n

            # Stability
            mean_err = sum(errors) / n
            error_std = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / n) if n > 1 else 0.0
            max_error = max(abs_errors)

            # Trading usefulness
            within_1 = sum(1 for e in abs_errors if e <= 1.0) / n
            within_2 = sum(1 for e in abs_errors if e <= 2.0) / n

            # Direction (error_c = actual - forecast; positive = underpredicted)
            underpredicted = sum(1 for e in errors if e > 0) / n
            overpredicted = sum(1 for e in errors if e < 0) / n

            # By lead time (days_ahead may be None for some rows)
            def _mae_by_lead(target_lead):
                lead_errors = [abs(float(r[0])) for r in rows
                               if r[1] is not None and int(r[1]) == target_lead]
                if not lead_errors:
                    return (None, 0)
                return (sum(lead_errors) / len(lead_errors), len(lead_errors))

            mae_d0, n_d0 = _mae_by_lead(0)
            mae_d1, n_d1 = _mae_by_lead(1)
            mae_d2, n_d2 = _mae_by_lead(2)

            # Average forecast uncertainty (sigma) from recent temp_events
            avg_unc = None
            try:
                unc_rows = conn.execute(
                    "SELECT forecast_sigma_c FROM temp_events "
                    "WHERE LOWER(city) = LOWER(?) AND forecast_sigma_c IS NOT NULL "
                    "AND scan_timestamp >= ? "
                    "GROUP BY date ORDER BY scan_timestamp DESC",
                    (city, cutoff),
                ).fetchall()
                if unc_rows:
                    avg_unc = sum(float(r[0]) for r in unc_rows) / len(unc_rows)
            except Exception:
                pass

            all_metrics.append({
                "city": city, "lat": lat, "lon": lon,
                "n": n, "mae": mae, "rmse": rmse, "bias": bias,
                "error_std": error_std, "max_error": max_error,
                "within_1": within_1, "within_2": within_2,
                "underpredicted": underpredicted, "overpredicted": overpredicted,
                "mae_d0": mae_d0, "mae_d1": mae_d1, "mae_d2": mae_d2,
                "n_d0": n_d0, "n_d1": n_d1, "n_d2": n_d2,
                "avg_uncertainty": avg_unc,
            })

        if not all_metrics:
            return 0

        # Normalization pass: scale MAE, std, max_error to [0,1] across cities
        all_mae = [m["mae"] for m in all_metrics]
        all_std = [m["error_std"] for m in all_metrics]
        all_max = [m["max_error"] for m in all_metrics]

        def _normalize(val, vals):
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return 0.5
            return (val - lo) / (hi - lo)

        for m in all_metrics:
            norm_mae = _normalize(m["mae"], all_mae)
            norm_std = _normalize(m["error_std"], all_std)
            norm_max = _normalize(m["max_error"], all_max)

            score = (
                m["within_1"] * 35
                + (1 - norm_std) * 25
                + (1 - norm_mae) * 20
                + m["within_2"] * 10
                + (1 - norm_max) * 10
            )

            conn.execute("""
                INSERT INTO city_forecast_accuracy (
                    city, lat, lon, window_days, n_days,
                    mae_c, rmse_c, bias_c,
                    error_std_c, max_error_c,
                    pct_within_1c, pct_within_2c,
                    pct_underpredicted, pct_overpredicted,
                    mae_d0_c, mae_d1_c, mae_d2_c,
                    n_d0, n_d1, n_d2,
                    avg_uncertainty_c,
                    accuracy_score, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(city) DO UPDATE SET
                    lat=excluded.lat, lon=excluded.lon,
                    window_days=excluded.window_days, n_days=excluded.n_days,
                    mae_c=excluded.mae_c, rmse_c=excluded.rmse_c, bias_c=excluded.bias_c,
                    error_std_c=excluded.error_std_c, max_error_c=excluded.max_error_c,
                    pct_within_1c=excluded.pct_within_1c, pct_within_2c=excluded.pct_within_2c,
                    pct_underpredicted=excluded.pct_underpredicted,
                    pct_overpredicted=excluded.pct_overpredicted,
                    mae_d0_c=excluded.mae_d0_c, mae_d1_c=excluded.mae_d1_c,
                    mae_d2_c=excluded.mae_d2_c,
                    n_d0=excluded.n_d0, n_d1=excluded.n_d1, n_d2=excluded.n_d2,
                    avg_uncertainty_c=excluded.avg_uncertainty_c,
                    accuracy_score=excluded.accuracy_score, updated_at=excluded.updated_at
            """, (
                m["city"], m["lat"], m["lon"], window_days, m["n"],
                round(m["mae"], 4), round(m["rmse"], 4), round(m["bias"], 4),
                round(m["error_std"], 4), round(m["max_error"], 4),
                round(m["within_1"], 4), round(m["within_2"], 4),
                round(m["underpredicted"], 4), round(m["overpredicted"], 4),
                round(m["mae_d0"], 4) if m["mae_d0"] is not None else None,
                round(m["mae_d1"], 4) if m["mae_d1"] is not None else None,
                round(m["mae_d2"], 4) if m["mae_d2"] is not None else None,
                m["n_d0"], m["n_d1"], m["n_d2"],
                round(m["avg_uncertainty"], 4) if m["avg_uncertainty"] is not None else None,
                round(score, 2), now_iso,
            ))
            updated += 1

    return updated


def bump_vc_usage(query_cost: int | None) -> None:
    """Increment the daily VC cost counter by `query_cost` (default 0) and
    count one call.  Idempotent per (date, call) — caller is responsible for
    calling once per VC request."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    cost = int(query_cost or 0)
    sql = """
        INSERT INTO vc_usage_daily (date, total_query_cost, n_calls, updated_at)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(date) DO UPDATE SET
            total_query_cost = total_query_cost + excluded.total_query_cost,
            n_calls          = n_calls + 1,
            updated_at       = excluded.updated_at
    """
    with _get_conn() as conn:
        conn.execute(sql, (day, cost, now.isoformat()))


# ---------------------------------------------------------------------------
# Phase ML-v1 — training row helpers
# ---------------------------------------------------------------------------

def get_ml_backfill_cities() -> list[dict]:
    """Return unique (city, lat, lon) tuples seen in temp_events, for use
    by the ML training-data backfill script.  One row per city — if a city
    has been seen at multiple coordinates we pick the most recent.
    """
    sql = """
        SELECT city, lat, lon
        FROM temp_events
        WHERE city IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL
        GROUP BY city
        HAVING MAX(scan_timestamp)
        ORDER BY city
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def ml_training_row_exists(
    city: str, target_date: str, decision_hour_local: int, feature_version: str
) -> bool:
    sql = """
        SELECT 1 FROM ml_training_rows
        WHERE city = ? AND target_date = ?
          AND decision_hour_local = ? AND feature_version = ?
        LIMIT 1
    """
    with _get_conn() as conn:
        return conn.execute(
            sql, (city, target_date, decision_hour_local, feature_version)
        ).fetchone() is not None


def insert_ml_training_row(
    city: str,
    lat_key: float,
    lon_key: float,
    target_date: str,
    decision_hour_local: int,
    feature_version: str,
    features_json: str,
    t_max_c: float,
    fetched_at_utc: str,
) -> int:
    sql = """
        INSERT OR IGNORE INTO ml_training_rows
            (city, lat_key, lon_key, target_date, decision_hour_local,
             feature_version, features_json, t_max_c, fetched_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            city, lat_key, lon_key, target_date, decision_hour_local,
            feature_version, features_json, t_max_c, fetched_at_utc,
        ))
        return cur.lastrowid or 0


def load_ml_training_rows(city: str, feature_version: str) -> list[dict]:
    """Return all training rows for a (city, feature_version), ordered by
    target_date then decision_hour_local.  Each row keeps features_json as
    the raw JSON string — parsing is the caller's responsibility."""
    sql = """
        SELECT city, lat_key, lon_key, target_date, decision_hour_local,
               feature_version, features_json, t_max_c, fetched_at_utc
        FROM ml_training_rows
        WHERE city = ? AND feature_version = ?
        ORDER BY target_date ASC, decision_hour_local ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (city, feature_version)).fetchall()]


def register_ml_model(
    version: str,
    city: str,
    trained_at_utc: str,
    training_window_start: str | None,
    training_window_end: str | None,
    feature_count: int,
    n_training_rows: int,
    point_rmse_c: float | None,
    residual_sigma_c: float | None,
    brier_score: float | None,
    model_path: str,
    activate_at_utc: str | None = None,
    notes: str | None = None,
) -> int:
    sql = """
        INSERT OR REPLACE INTO ml_model_registry
            (version, city, trained_at_utc, training_window_start,
             training_window_end, feature_count, n_training_rows,
             point_rmse_c, residual_sigma_c, brier_score,
             model_path, activate_at_utc, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            version, city, trained_at_utc, training_window_start,
            training_window_end, feature_count, n_training_rows,
            point_rmse_c, residual_sigma_c, brier_score,
            model_path, activate_at_utc, notes,
        ))
        return cur.lastrowid or 0


def get_active_ml_model(city: str) -> dict | None:
    """Return the most recently trained ml_model_registry row for a city."""
    sql = """
        SELECT * FROM ml_model_registry
        WHERE city = ?
        ORDER BY trained_at_utc DESC
        LIMIT 1
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (city,)).fetchone()
        return dict(row) if row else None


def count_ml_training_rows(
    city: str | None = None, feature_version: str | None = None
) -> int:
    where = []
    args: list = []
    if city is not None:
        where.append("city = ?")
        args.append(city)
    if feature_version is not None:
        where.append("feature_version = ?")
        args.append(feature_version)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT COUNT(*) FROM ml_training_rows{clause}"
    with _get_conn() as conn:
        return int(conn.execute(sql, args).fetchone()[0])


# ---------------------------------------------------------------------------
# Phase 8 — obs_climatology helpers
# ---------------------------------------------------------------------------

_OBS_CLIM_HOURLY_COLS = [
    "city", "doy", "hour", "n_samples",
    "temp_mu", "temp_sigma",
    "dew_mu", "dew_sigma",
    "pressure_mu", "pressure_sigma",
    "cloudcover_mu", "cloudcover_sigma",
    "windspeed_mu", "windspeed_sigma",
    "solarradiation_mu", "solarradiation_sigma",
    "fetched_at_utc",
]

_OBS_CLIM_DAILY_COLS = [
    "city", "doy", "n_samples",
    "tmax_mu", "tmax_sigma",
    "tmax_p10", "tmax_p25", "tmax_p50", "tmax_p75", "tmax_p90",
    "tmin_mu", "tmin_sigma", "tmean_mu",
    "fetched_at_utc",
]


def upsert_obs_climatology_hourly_bulk(rows: list[dict]) -> int:
    """Replace-on-conflict bulk write of (city, doy, hour) climatology rows."""
    if not rows:
        return 0
    placeholders = ",".join("?" * len(_OBS_CLIM_HOURLY_COLS))
    sql = (f"INSERT OR REPLACE INTO obs_climatology_hourly "
           f"({','.join(_OBS_CLIM_HOURLY_COLS)}) VALUES ({placeholders})")
    with _get_conn() as conn:
        conn.executemany(sql, [
            tuple(r.get(c) for c in _OBS_CLIM_HOURLY_COLS) for r in rows
        ])
    return len(rows)


def upsert_obs_climatology_daily_bulk(rows: list[dict]) -> int:
    if not rows:
        return 0
    placeholders = ",".join("?" * len(_OBS_CLIM_DAILY_COLS))
    sql = (f"INSERT OR REPLACE INTO obs_climatology_daily "
           f"({','.join(_OBS_CLIM_DAILY_COLS)}) VALUES ({placeholders})")
    with _get_conn() as conn:
        conn.executemany(sql, [
            tuple(r.get(c) for c in _OBS_CLIM_DAILY_COLS) for r in rows
        ])
    return len(rows)


def update_outcomes_ml_bin_probs_bulk(updates: list[dict]) -> int:
    """Update ml_bin_prob, ml_decision_hour, ml_model_version on existing
    temp_outcomes rows.  Each update dict needs:
        scan_timestamp, contract_id, ml_bin_prob, ml_decision_hour, ml_model_version

    Matches on (scan_timestamp, contract_id) — the natural per-scan key.
    Returns the number of rows touched (sum across updates)."""
    if not updates:
        return 0
    sql = """
        UPDATE temp_outcomes
        SET ml_bin_prob       = ?,
            ml_decision_hour  = ?,
            ml_model_version  = ?
        WHERE scan_timestamp = ? AND contract_id = ?
    """
    n = 0
    with _get_conn() as conn:
        for u in updates:
            cur = conn.execute(sql, (
                u.get("ml_bin_prob"),
                u.get("ml_decision_hour"),
                u.get("ml_model_version"),
                u.get("scan_timestamp"),
                u.get("contract_id"),
            ))
            n += cur.rowcount
    return n


def get_obs_climatology_hourly(city: str, doy: int, hour: int) -> dict | None:
    sql = ("SELECT * FROM obs_climatology_hourly "
           "WHERE city = ? AND doy = ? AND hour = ? LIMIT 1")
    with _get_conn() as conn:
        row = conn.execute(sql, (city, doy, hour)).fetchone()
        return dict(row) if row else None


def get_obs_climatology_daily(city: str, doy: int) -> dict | None:
    sql = "SELECT * FROM obs_climatology_daily WHERE city = ? AND doy = ? LIMIT 1"
    with _get_conn() as conn:
        row = conn.execute(sql, (city, doy)).fetchone()
        return dict(row) if row else None


def load_all_obs_climatology_hourly(city: str) -> dict[tuple[int, int], dict]:
    """Pre-load an entire city's hourly climatology into a (doy, hour) -> row
    dict.  Backfill calls this once per city to avoid 366*24 = 8784 SQL
    round-trips per city."""
    sql = "SELECT * FROM obs_climatology_hourly WHERE city = ?"
    with _get_conn() as conn:
        return {(r["doy"], r["hour"]): dict(r)
                for r in conn.execute(sql, (city,)).fetchall()}


def load_all_obs_climatology_daily(city: str) -> dict[int, dict]:
    sql = "SELECT * FROM obs_climatology_daily WHERE city = ?"
    with _get_conn() as conn:
        return {r["doy"]: dict(r) for r in conn.execute(sql, (city,)).fetchall()}


import calendar  # kept for any downstream code that imports it from here
