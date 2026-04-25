"""
ml_phase1_verify.py — End-to-end check of the Phase 1 schema + VC expansion.

What it does:
  1. Calls fetch_intraday() for one city.
  2. Prints every NEW field the expanded `elements=` should now return,
     so you can confirm Visual Crossing is actually sending them.
  3. Initializes the DB (applies the Phase 1 migrations).
  4. Inserts one synthetic live_observations row using all the new kwargs.
  5. Reads the row back and prints the new columns to confirm round-trip.

Safe to run repeatedly.  Writes one extra row to live_observations per run
under the synthetic event_id "PHASE1_VERIFY" — easy to filter or delete.

    python -m bot.scripts.ml_phase1_verify
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from visualcrossing import fetch_intraday
from db import init_db, insert_live_observation
from config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("ml_phase1_verify")

CITY, LAT, LON = "Chicago", 41.85, -87.65

NEW_CURRENT_FIELDS = [
    "current_feelslike_c",
    "current_dew_c",
    "current_pressure_hpa",
    "current_visibility_km",
    "current_windgust_kph",
    "current_winddir_deg",
    "current_preciptype",
    "current_snow_cm",
    "current_snowdepth_cm",
    "current_solarradiation_wm2",
    "current_solarenergy_mj",
    "current_uvindex",
]

NEW_DB_COLUMNS = [
    "feelslike_c", "dew_c", "pressure_hpa", "visibility_km",
    "windgust_kph", "winddir_deg", "preciptype",
    "snow_cm", "snowdepth_cm",
    "solarradiation_wm2", "solarenergy_mj", "uvindex",
]


def step_1_call_vc() -> dict:
    log.info("=" * 70)
    log.info(f"STEP 1 — fetch_intraday({CITY}, {LAT}, {LON})")
    log.info("=" * 70)
    j = fetch_intraday(LAT, LON)
    log.info(
        f"queryCost={j.get('query_cost')} "
        f"obs_hours={len(j.get('observed_hours') or [])} "
        f"fcst_hours={len(j.get('forecast_hours') or [])}"
    )
    return j


def step_2_check_new_fields(j: dict) -> int:
    log.info("=" * 70)
    log.info("STEP 2 — checking that the expanded VC response contains new fields")
    log.info("=" * 70)
    n_present = 0
    n_missing = 0
    for k in NEW_CURRENT_FIELDS:
        if k in j:
            v = j[k]
            n_present += 1
            tag = "OK" if v is not None else "OK(null)"
            log.info(f"  [{tag}] {k} = {v!r}")
        else:
            n_missing += 1
            log.error(f"  [MISSING] {k} not present in fetch_intraday() result")
    log.info(f"summary: {n_present}/{len(NEW_CURRENT_FIELDS)} new keys present, "
             f"{n_missing} missing")
    return n_missing


def step_3_init_db() -> None:
    log.info("=" * 70)
    log.info(f"STEP 3 — init_db()  ({DB_PATH})")
    log.info("=" * 70)
    init_db()
    log.info("init_db() returned OK (migrations applied if missing)")


def step_4_insert_row(j: dict) -> int:
    log.info("=" * 70)
    log.info("STEP 4 — insert_live_observation() with new fields")
    log.info("=" * 70)
    pulled_at = datetime.now(timezone.utc).isoformat()

    preciptype_raw = j.get("current_preciptype")
    preciptype_str = (
        json.dumps(preciptype_raw)
        if isinstance(preciptype_raw, (list, tuple))
        else preciptype_raw
    )

    obs_id = insert_live_observation(
        event_id              = "PHASE1_VERIFY",
        city                  = CITY,
        date                  = datetime.utcnow().date().isoformat(),
        lat                   = LAT,
        lon                   = LON,
        pulled_at_utc         = pulled_at,
        observed_at_utc       = j.get("current_time"),
        observed_at_local     = j.get("current_time"),
        vc_source             = j.get("current_vc_source"),
        current_temp_c        = j.get("current_temp_c"),
        humidity              = j.get("current_humidity"),
        cloudcover            = j.get("current_cloudcover"),
        windspeed_kph         = j.get("current_windspeed_kph"),
        precip_mm             = j.get("current_precip_mm"),
        conditions            = j.get("current_conditions"),
        observed_max_so_far_c = j.get("observed_max_so_far_c"),
        stations              = json.dumps(j.get("current_stations") or []),
        query_cost            = j.get("query_cost"),
        feelslike_c           = j.get("current_feelslike_c"),
        dew_c                 = j.get("current_dew_c"),
        pressure_hpa          = j.get("current_pressure_hpa"),
        visibility_km         = j.get("current_visibility_km"),
        windgust_kph          = j.get("current_windgust_kph"),
        winddir_deg           = j.get("current_winddir_deg"),
        preciptype            = preciptype_str,
        snow_cm               = j.get("current_snow_cm"),
        snowdepth_cm          = j.get("current_snowdepth_cm"),
        solarradiation_wm2    = j.get("current_solarradiation_wm2"),
        solarenergy_mj        = j.get("current_solarenergy_mj"),
        uvindex               = j.get("current_uvindex"),
    )
    log.info(f"inserted live_observations.id = {obs_id}")
    return obs_id


def step_5_read_back(obs_id: int) -> int:
    log.info("=" * 70)
    log.info(f"STEP 5 — read live_observations.id = {obs_id} and verify columns")
    log.info("=" * 70)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM live_observations WHERE id = ?", (obs_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        log.error(f"row id={obs_id} not found")
        return 1

    d = dict(row)
    n_present_in_schema = 0
    n_populated         = 0
    for col in NEW_DB_COLUMNS:
        if col in d:
            n_present_in_schema += 1
            v = d[col]
            if v is not None:
                n_populated += 1
                tag = "OK"
            else:
                tag = "OK(null)"
            log.info(f"  [{tag}] {col} = {v!r}")
        else:
            log.error(f"  [MISSING-COLUMN] {col} not present in live_observations table")

    log.info(
        f"summary: {n_present_in_schema}/{len(NEW_DB_COLUMNS)} new columns exist; "
        f"{n_populated} populated with non-null values"
    )
    return 0 if n_present_in_schema == len(NEW_DB_COLUMNS) else 2


def step_6_check_decision_snapshots_columns() -> int:
    log.info("=" * 70)
    log.info("STEP 6 — verify decision_snapshots ML columns exist")
    log.info("=" * 70)
    expected = ["ml_mu_c", "ml_sigma_c", "ml_model_version", "ml_weight_used"]
    conn = sqlite3.connect(DB_PATH)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(decision_snapshots)")}
    finally:
        conn.close()
    missing = [c for c in expected if c not in cols]
    if missing:
        log.error(f"missing columns on decision_snapshots: {missing}")
        return 1
    log.info(f"all {len(expected)} ML columns present on decision_snapshots: {expected}")
    return 0


def step_7_check_ml_registry_table() -> int:
    log.info("=" * 70)
    log.info("STEP 7 — verify ml_model_registry table exists")
    log.info("=" * 70)
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ml_model_registry'"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        log.error("ml_model_registry table not present")
        return 1
    log.info("ml_model_registry table present")
    return 0


def main() -> int:
    if not os.getenv("VISUAL_CROSSING_API_KEY"):
        log.error("VISUAL_CROSSING_API_KEY is not set in environment")
        return 1

    j = step_1_call_vc()
    n_missing = step_2_check_new_fields(j)
    step_3_init_db()
    obs_id = step_4_insert_row(j)
    rb = step_5_read_back(obs_id)
    rc6 = step_6_check_decision_snapshots_columns()
    rc7 = step_7_check_ml_registry_table()

    log.info("=" * 70)
    if n_missing == 0 and rb == 0 and rc6 == 0 and rc7 == 0:
        log.info("PHASE 1 VERIFICATION: PASS")
        return 0
    log.error(
        f"PHASE 1 VERIFICATION: FAIL "
        f"(missing_vc={n_missing} read_back_rc={rb} ds_cols_rc={rc6} registry_rc={rc7})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
