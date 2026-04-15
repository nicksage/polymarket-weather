"""
schema_check.py — Quick sanity report on the Phase-2 DB schema.

Prints per-table row counts and key column presence for the time-versioned
forecast / observation / snapshot tables.  Safe to run any time.

    python bot/scripts/schema_check.py
"""

import os
import sqlite3
import sys

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from config import DB_PATH


PHASE2_TABLES = [
    "forecast_runs",
    "forecast_hourly",
    "live_observations",
    "decision_snapshots",
    "vc_usage_daily",
]

EXISTING_TABLES = [
    "temp_events",
    "temp_outcomes",
    "positions",
    "signals",
    "forecast_errors",
    "era5_daily",
]

REQUIRED_EVENT_COLS = [
    "timezone", "latest_forecast_ts", "latest_observation_ts",
    "current_temp_c", "observed_max_so_far_c", "expected_temp_now_c",
    "actual_minus_expected_c", "forecast_delta_mu_c",
    "forecast_delta_sigma_c", "forecast_agreement_c",
    # Phase 2b: live adjustment
    "adjusted_mu_c", "adjusted_sigma_c",
    "live_adjustment_score", "live_adjustment_components",
]

REQUIRED_OUTCOME_COLS = [
    # Phase 2b: blended baseline alongside the active model_prob
    "model_prob_blended",
]

REQUIRED_POSITION_COLS = [
    "entry_snapshot_id", "exit_snapshot_id",
    "entry_forecast_delta_mu_c", "entry_actual_vs_expected_c",
    "max_favorable_excursion", "max_adverse_excursion", "exit_reason",
]

REQUIRED_SNAPSHOT_COLS = [
    "blended_mu_c", "blended_sigma_c",
    "adjusted_mu_c", "adjusted_sigma_c",
    "live_adjustment_score", "live_adjustment_components",
    "obs_floor_applied",
]


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print(f"DB: {DB_PATH}\n")

    print("--- Phase 2 tables ---")
    for tbl in PHASE2_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")]
            print(f"  {tbl:<22} rows={n:<8}  cols={len(cols)}")
        except sqlite3.OperationalError as e:
            print(f"  {tbl:<22} MISSING  ({e})")

    print("\n--- Existing tables ---")
    for tbl in EXISTING_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:<22} rows={n}")
        except sqlite3.OperationalError:
            print(f"  {tbl:<22} MISSING")

    print("\n--- temp_events required columns ---")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(temp_events)")}
    for c in REQUIRED_EVENT_COLS:
        mark = "OK " if c in cols else "MISS"
        print(f"  [{mark}] {c}")

    print("\n--- temp_outcomes required columns ---")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(temp_outcomes)")}
    for c in REQUIRED_OUTCOME_COLS:
        mark = "OK " if c in cols else "MISS"
        print(f"  [{mark}] {c}")

    print("\n--- positions required columns ---")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(positions)")}
    for c in REQUIRED_POSITION_COLS:
        mark = "OK " if c in cols else "MISS"
        print(f"  [{mark}] {c}")

    print("\n--- decision_snapshots required columns ---")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decision_snapshots)")}
    for c in REQUIRED_SNAPSHOT_COLS:
        mark = "OK " if c in cols else "MISS"
        print(f"  [{mark}] {c}")

    print("\n--- indexes ---")
    rows = conn.execute(
        "SELECT name, tbl_name FROM sqlite_master "
        "WHERE type='index' AND name LIKE 'idx_%' ORDER BY tbl_name, name"
    ).fetchall()
    for r in rows:
        print(f"  {r['tbl_name']:<22} {r['name']}")

    # VC cost this UTC day
    try:
        row = conn.execute(
            "SELECT * FROM vc_usage_daily ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row:
            print(f"\nVC usage latest: {dict(row)}")
    except sqlite3.OperationalError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
