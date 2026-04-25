"""
train.py — Train per-city TempDistributionModel(s) from ml_training_rows.

Usage
-----
    python -m bot.ml.train                      # all cities with ≥ MIN rows
    python -m bot.ml.train --city Chicago
    python -m bot.ml.train --min-rows 400 --folds 5

For each city:
  1. Load all rows from ml_training_rows at the current FEATURE_VERSION.
  2. Parse features_json → np.ndarray[float], null → NaN.
  3. Fit TempDistributionModel with TimeSeriesSplit(n_folds) on unique dates
     (10:00 and 12:00 rows of the same date stay in the same fold — no
     intra-day label leakage).
  4. Save model to {ML_MODELS_DIR}/{city}_{FEATURE_VERSION}.joblib.
  5. Register in ml_model_registry with training metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import numpy as np

from config import ML_MODELS_DIR
from db import (
    init_db,
    load_ml_training_rows,
    count_ml_training_rows,
    register_ml_model,
    get_ml_backfill_cities,
)
from ml.distribution_model import TempDistributionModel
from ml.schema import FEATURE_VERSION, N_FEATURES, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("ml_train")


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------

def load_city_dataset(
    city: str, feature_version: str
) -> tuple[np.ndarray, np.ndarray, list[date]] | None:
    """Load training rows for a city, parse features, return (X, y, dates).
    Returns None if the city has no rows."""
    rows = load_ml_training_rows(city, feature_version)
    if not rows:
        return None

    X = np.full((len(rows), N_FEATURES), np.nan, dtype=np.float64)
    y = np.empty(len(rows), dtype=np.float64)
    dates: list[date] = []
    for i, r in enumerate(rows):
        feats = json.loads(r["features_json"])
        if len(feats) != N_FEATURES:
            raise ValueError(
                f"row {r['id'] if 'id' in r else i} has {len(feats)} features, "
                f"expected {N_FEATURES}"
            )
        for j, v in enumerate(feats):
            if v is not None:
                X[i, j] = float(v)
        y[i] = float(r["t_max_c"])
        dates.append(date.fromisoformat(r["target_date"]))
    return X, y, dates


# ---------------------------------------------------------------------------
# Per-city training
# ---------------------------------------------------------------------------

def train_one_city(city: str, n_folds: int, models_dir: Path) -> dict:
    log.info(f"--- training {city} ---")
    ds = load_city_dataset(city, FEATURE_VERSION)
    if ds is None:
        log.warning(f"[{city}] no training rows")
        return {"city": city, "status": "no_rows"}
    X, y, dates = ds

    # Quick feature-nullness report
    null_rate = np.isnan(X).mean(axis=0)
    dense_features = int((null_rate < 0.5).sum())
    log.info(
        f"[{city}] loaded: rows={X.shape[0]} unique_dates={len(set(dates))} "
        f"features_dense(<50% null)={dense_features}/{N_FEATURES}"
    )

    model = TempDistributionModel(city=city)
    stats = model.fit(X, y, dates, n_folds=n_folds)

    path = models_dir / f"{city}_{FEATURE_VERSION}.joblib"
    model.save(path)

    register_ml_model(
        version               = model.version or FEATURE_VERSION,
        city                  = city,
        trained_at_utc        = model.trained_at_utc or "",
        training_window_start = stats.window_start,
        training_window_end   = stats.window_end,
        feature_count         = N_FEATURES,
        n_training_rows       = stats.n_rows,
        point_rmse_c          = stats.point_rmse_c,
        residual_sigma_c      = stats.residual_sigma_c,
        brier_score           = None,   # computed in Phase 6 vs. live outcomes
        model_path            = str(path),
        notes                 = f"feature_version={FEATURE_VERSION} n_folds={n_folds}",
    )
    return {
        "city":   city,
        "status": "ok",
        "rmse":   stats.point_rmse_c,
        "sigma":  stats.residual_sigma_c,
        "rows":   stats.n_rows,
        "path":   str(path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--city", action="append", default=None,
                    help="city to train (repeatable); default: all with enough rows")
    ap.add_argument("--min-rows", type=int, default=400,
                    help="skip cities with fewer than this many training rows (default 400)")
    ap.add_argument("--folds", type=int, default=5,
                    help="TimeSeriesSplit folds (default 5)")
    args = ap.parse_args()

    init_db()
    models_dir = Path(ML_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.city:
        city_names = args.city
    else:
        city_names = [c["city"] for c in get_ml_backfill_cities()]

    results: list[dict] = []
    for city in city_names:
        n = count_ml_training_rows(city=city, feature_version=FEATURE_VERSION)
        if n < args.min_rows:
            log.info(f"[{city}] skip: {n} rows < min-rows={args.min_rows}")
            results.append({"city": city, "status": "insufficient_rows", "rows": n})
            continue
        try:
            r = train_one_city(city, n_folds=args.folds, models_dir=models_dir)
        except Exception as e:
            log.exception(f"[{city}] training failed: {e}")
            r = {"city": city, "status": "error", "error": str(e)}
        results.append(r)

    log.info("=" * 70)
    log.info("TRAINING SUMMARY")
    log.info("=" * 70)
    ok   = [r for r in results if r["status"] == "ok"]
    skip = [r for r in results if r["status"] != "ok"]
    for r in ok:
        log.info(
            f"  [{r['city']:20s}] OK   "
            f"rows={r['rows']:5d} RMSE={r['rmse']:.3f}C "
            f"sigma={r['sigma']:.3f}C  {r['path']}"
        )
    for r in skip:
        extra = r.get("rows", "?")
        log.info(f"  [{r['city']:20s}] SKIP status={r['status']} rows={extra}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
