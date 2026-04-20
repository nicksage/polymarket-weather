"""
backtest_db.py — Schema and helpers for the backtest-only database.

Kept in a SEPARATE file (bot/data/backtest.db) so that backfill rebuilds
and experimentation cannot touch the live trading DB.  Cross-DB reads
to signals.db use ATTACH DATABASE from the backtest scripts.

Tables
------
  historical_hourly_observations — Archive API (ERA5) hourly obs per city/day
  empirical_sigma_by_lead        — σ per (model, lead_days_bucket) derived
                                   from live forecast_errors data
  backtest_runs                  — one row per replay invocation
  backtest_results               — per replay-point × bin
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from config import BACKTEST_DB_PATH


@contextmanager
def _get_conn():
    os.makedirs(os.path.dirname(BACKTEST_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(BACKTEST_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_backtest_db() -> None:
    """Create all backtest tables + indexes.  Safe to call repeatedly."""
    with _get_conn() as conn:
        conn.executescript("""
            -- Hourly observations backfilled from Open-Meteo Archive API.
            -- One row per (lat, lon, hour_ts_utc).  Reanalysis (ERA5) values.
            CREATE TABLE IF NOT EXISTS historical_hourly_observations (
                lat_key        REAL NOT NULL,
                lon_key        REAL NOT NULL,
                city           TEXT,
                date           TEXT NOT NULL,         -- YYYY-MM-DD UTC
                hour_ts_utc    TEXT NOT NULL,
                temp_c         REAL,
                fetched_at     TEXT NOT NULL,
                PRIMARY KEY (lat_key, lon_key, hour_ts_utc)
            );
            CREATE INDEX IF NOT EXISTS idx_hho_city_date
                ON historical_hourly_observations(city, date);

            -- Empirical σ by (model, lead_days_bucket) from forecast_errors.
            -- Refreshed by empirical_sigma_builder.py.
            CREATE TABLE IF NOT EXISTS empirical_sigma_by_lead (
                model                TEXT NOT NULL,
                lead_days_bucket     INTEGER NOT NULL,
                n_observations       INTEGER,
                rmse_c               REAL,
                mean_abs_error_c     REAL,
                fetched_at           TEXT NOT NULL,
                PRIMARY KEY (model, lead_days_bucket)
            );

            -- One row per backtest invocation.  params_json captures the
            -- active LIVE_ADJ_* flags at run time so older results remain
            -- interpretable after tuning changes.
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at     TEXT NOT NULL,
                finished_at    TEXT,
                kind           TEXT,               -- 'floor_only' | 'full'
                params_json    TEXT,
                window_days    INTEGER,
                n_events       INTEGER,
                n_replay_pts   INTEGER,
                summary_json   TEXT
            );

            -- Variant column added in v2; default 'floor_only' preserves
            -- meaning of existing v1 rows.
            CREATE TABLE IF NOT EXISTS backtest_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              INTEGER NOT NULL REFERENCES backtest_runs(id),
                event_id            TEXT,
                city                TEXT,
                target_date         TEXT,
                replay_at_utc       TEXT,
                local_hour          REAL,
                contract_id         TEXT,
                bin_low_c           REAL,
                bin_high_c          REAL,
                blended_mu_c        REAL,
                blended_sigma_c     REAL,
                adjusted_mu_c       REAL,
                adjusted_sigma_c    REAL,
                mu_adjustment_c     REAL,
                obs_floor_c         REAL,
                p_blended           REAL,
                p_blended_floor     REAL,       -- blended μ/σ + floor
                p_adjusted          REAL,       -- active config
                actual_tmax_c       REAL,
                bin_contains_actual INTEGER,
                floor_applied       INTEGER,
                cold_start          INTEGER,
                score               REAL,
                components_json     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_bt_run_event
                ON backtest_results(run_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_bt_run_hour
                ON backtest_results(run_id, local_hour);
        """)

        # v2: add variant column if missing
        try:
            conn.execute(
                "ALTER TABLE backtest_results ADD COLUMN variant TEXT DEFAULT 'floor_only'"
            )
        except Exception:
            pass
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bt_run_variant "
                "ON backtest_results(run_id, variant)"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Hourly observation helpers
# ---------------------------------------------------------------------------

def _loc_keys(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, 2), round(lon, 2))


def upsert_historical_hourly_obs(
    lat: float, lon: float, city: str | None, rows: list[dict],
) -> int:
    """Bulk upsert Archive hourly temp rows.

    Each `rows` dict must have: date, hour_ts_utc, temp_c.
    """
    if not rows:
        return 0
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    lat_k, lon_k = _loc_keys(lat, lon)
    sql = """
        INSERT INTO historical_hourly_observations
            (lat_key, lon_key, city, date, hour_ts_utc, temp_c, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lat_key, lon_key, hour_ts_utc) DO UPDATE SET
            city       = excluded.city,
            date       = excluded.date,
            temp_c     = excluded.temp_c,
            fetched_at = excluded.fetched_at
    """
    with _get_conn() as conn:
        for r in rows:
            conn.execute(sql, (
                lat_k, lon_k, city, r["date"], r["hour_ts_utc"],
                r.get("temp_c"), now,
            ))
    return len(rows)


def get_historical_hourly_obs(
    lat: float, lon: float, date_str: str,
) -> list[dict]:
    lat_k, lon_k = _loc_keys(lat, lon)
    sql = """
        SELECT * FROM historical_hourly_observations
        WHERE lat_key = ? AND lon_key = ? AND date = ?
        ORDER BY hour_ts_utc ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (lat_k, lon_k, date_str)).fetchall()]


# ---------------------------------------------------------------------------
# Empirical sigma helpers
# ---------------------------------------------------------------------------

def upsert_empirical_sigma(
    model: str, lead_days_bucket: int, n_obs: int,
    rmse_c: float, mae_c: float,
) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO empirical_sigma_by_lead
            (model, lead_days_bucket, n_observations, rmse_c, mean_abs_error_c, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(model, lead_days_bucket) DO UPDATE SET
            n_observations   = excluded.n_observations,
            rmse_c           = excluded.rmse_c,
            mean_abs_error_c = excluded.mean_abs_error_c,
            fetched_at       = excluded.fetched_at
    """
    with _get_conn() as conn:
        conn.execute(sql, (model, lead_days_bucket, n_obs, rmse_c, mae_c, now))


def get_empirical_sigma(model: str, lead_days_bucket: int) -> float | None:
    """Return the σ (RMSE) for a given model + lead bucket.  Falls back
    to the nearest bucket if the exact one is missing."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT rmse_c FROM empirical_sigma_by_lead "
            "WHERE model = ? AND lead_days_bucket = ?",
            (model, lead_days_bucket),
        ).fetchone()
        if row:
            return float(row[0])
        # Nearest fallback
        row = conn.execute(
            "SELECT rmse_c, ABS(lead_days_bucket - ?) AS dist "
            "FROM empirical_sigma_by_lead WHERE model = ? "
            "ORDER BY dist ASC LIMIT 1",
            (lead_days_bucket, model),
        ).fetchone()
        return float(row[0]) if row else None


# ---------------------------------------------------------------------------
# Backtest run/result helpers
# ---------------------------------------------------------------------------

def insert_backtest_run(
    started_at: str, kind: str, params_json: str, window_days: int,
) -> int:
    sql = """
        INSERT INTO backtest_runs (started_at, kind, params_json, window_days)
        VALUES (?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (started_at, kind, params_json, window_days))
        return cur.lastrowid


def finalize_backtest_run(
    run_id: int, finished_at: str, n_events: int, n_replay_pts: int,
    summary_json: str,
) -> None:
    sql = """
        UPDATE backtest_runs
        SET finished_at = ?, n_events = ?, n_replay_pts = ?, summary_json = ?
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (finished_at, n_events, n_replay_pts, summary_json, run_id))


def insert_backtest_result(row: dict) -> None:
    cols = [
        "run_id", "variant", "event_id", "city", "target_date",
        "replay_at_utc", "local_hour",
        "contract_id", "bin_low_c", "bin_high_c",
        "blended_mu_c", "blended_sigma_c", "adjusted_mu_c", "adjusted_sigma_c",
        "mu_adjustment_c", "obs_floor_c",
        "p_blended", "p_blended_floor", "p_adjusted",
        "actual_tmax_c", "bin_contains_actual",
        "floor_applied", "cold_start", "score", "components_json",
    ]
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT INTO backtest_results ({','.join(cols)}) VALUES ({placeholders})"
    vals = [row.get(c) for c in cols]
    with _get_conn() as conn:
        conn.execute(sql, vals)


def insert_backtest_results_bulk(rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = [
        "run_id", "variant", "event_id", "city", "target_date",
        "replay_at_utc", "local_hour",
        "contract_id", "bin_low_c", "bin_high_c",
        "blended_mu_c", "blended_sigma_c", "adjusted_mu_c", "adjusted_sigma_c",
        "mu_adjustment_c", "obs_floor_c",
        "p_blended", "p_blended_floor", "p_adjusted",
        "actual_tmax_c", "bin_contains_actual",
        "floor_applied", "cold_start", "score", "components_json",
    ]
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT INTO backtest_results ({','.join(cols)}) VALUES ({placeholders})"
    with _get_conn() as conn:
        conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    return len(rows)
