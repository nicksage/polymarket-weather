"""
train_pooled.py — Train ONE PooledQuantileDistributionModel across all cities.

Loads every v2.0 row from ml_training_rows, trains 7 quantile HGBRs on the
pooled dataset with city as a native categorical, and persists a single
joblib file.  Replaces 50 per-city models with one.

Usage
-----
    python -m bot.ml.train_pooled
    python -m bot.ml.train_pooled --folds 5 --min-rows-per-city 1000
    python -m bot.ml.train_pooled --cities Chicago,Tokyo,Mexico_City   # subset

Compared to bot.ml.train (per-city):
  * one model file (~50× smaller deployment surface)
  * city categorical lets a single model learn city-specific quirks
  * static city features (elevation, Köppen, hemisphere, coastal) let it
    generalize across analogous cities
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import numpy as np

from config import ML_MODELS_DIR
from db import (
    init_db, load_ml_training_rows, count_ml_training_rows,
    register_ml_model, get_ml_backfill_cities,
)
from ml.pooled_distribution_model import (
    PooledQuantileDistributionModel, DEFAULT_QUANTILES,
)
from ml.schema import FEATURE_VERSION, N_FEATURES, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("ml_train_pooled")


# ---------------------------------------------------------------------------
# Loading the pooled dataset
# ---------------------------------------------------------------------------

def load_pooled_dataset(
    feature_version: str, city_filter: list[str] | None,
    min_rows_per_city: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[date], list[str], dict[str, int]]:
    """Load every (city) row at the given feature_version.  Returns
    (X, y, dates, cities, per_city_counts)."""
    all_cities = [c["city"] for c in get_ml_backfill_cities()]
    if city_filter:
        wanted = {c.strip().lower() for c in city_filter}
        all_cities = [c for c in all_cities if c.lower() in wanted]

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    date_blocks: list[list[date]] = []
    city_blocks: list[list[str]] = []
    per_city_counts: dict[str, int] = {}

    for city in all_cities:
        n = count_ml_training_rows(city=city, feature_version=feature_version)
        if n < min_rows_per_city:
            log.info(f"[{city}] skip: {n} rows < min={min_rows_per_city}")
            continue
        rows = load_ml_training_rows(city, feature_version)
        Xc = np.full((len(rows), N_FEATURES), np.nan, dtype=np.float64)
        yc = np.empty(len(rows), dtype=np.float64)
        dc: list[date] = []
        for i, r in enumerate(rows):
            feats = json.loads(r["features_json"])
            if len(feats) != N_FEATURES:
                raise ValueError(
                    f"row from {city} target_date={r['target_date']} has "
                    f"{len(feats)} features, expected {N_FEATURES} "
                    f"(feature_version drift?)"
                )
            for j, v in enumerate(feats):
                if v is not None:
                    Xc[i, j] = float(v)
            yc[i] = float(r["t_max_c"])
            dc.append(date.fromisoformat(r["target_date"]))
        X_blocks.append(Xc)
        y_blocks.append(yc)
        date_blocks.append(dc)
        city_blocks.append([city] * len(rows))
        per_city_counts[city] = len(rows)
        log.info(f"[{city}] loaded {len(rows)} rows")

    if not X_blocks:
        raise RuntimeError("no cities had enough rows to train")

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    dates = [d for block in date_blocks for d in block]
    cities = [c for block in city_blocks for c in block]
    return X, y, dates, cities, per_city_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-rows-per-city", type=int, default=500)
    ap.add_argument("--cities", type=str, default=None,
                    help="comma-separated city subset (default: all eligible)")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=400)
    args = ap.parse_args()

    init_db()
    models_dir = Path(ML_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    city_filter = None
    if args.cities:
        city_filter = [c.strip().replace("_", " ") for c in args.cities.split(",") if c.strip()]

    log.info(f"Loading pooled dataset feature_version={FEATURE_VERSION}")
    X, y, dates, cities, per_city_counts = load_pooled_dataset(
        FEATURE_VERSION, city_filter, args.min_rows_per_city,
    )
    log.info(
        f"Pooled dataset: rows={X.shape[0]} features={X.shape[1]} "
        f"cities={len(per_city_counts)} unique_dates={len(set(dates))}"
    )

    # Quick null-rate report
    null_rate = np.isnan(X).mean(axis=0)
    n_dense = int((null_rate < 0.5).sum())
    log.info(f"  dense features (<50% null): {n_dense}/{N_FEATURES}")

    model = PooledQuantileDistributionModel(
        learning_rate=args.learning_rate, max_iter=args.max_iter,
    )
    stats = model.fit(X, y, dates, cities, n_folds=args.folds)

    path = models_dir / f"pooled_{FEATURE_VERSION}.joblib"
    model.save(path)

    register_ml_model(
        version               = model.version or FEATURE_VERSION,
        city                  = "__pooled__",
        trained_at_utc        = model.trained_at_utc or "",
        training_window_start = stats.window_start,
        training_window_end   = stats.window_end,
        feature_count         = N_FEATURES,
        n_training_rows       = stats.n_rows,
        point_rmse_c          = stats.point_rmse_c,
        residual_sigma_c      = None,    # quantile model — sigma is derived
        brier_score           = None,    # computed later in evaluate.py
        model_path            = str(path),
        notes=(
            f"pooled fv={FEATURE_VERSION} cities={stats.n_unique_cities} "
            f"q50_rmse={stats.point_rmse_c:.3f} cov80={stats.coverage_80:.3f} "
            f"cov50={stats.coverage_50:.3f}"
        ),
    )

    # Per-city RMSE summary (sorted worst → best)
    log.info("=" * 70)
    log.info("PER-CITY RMSE (q50 OOF)")
    log.info("=" * 70)
    by_rmse = sorted(stats.per_city_rmse.items(), key=lambda x: -x[1])
    for c, rmse in by_rmse:
        log.info(f"  [{c:20s}] RMSE={rmse:.3f}C  rows={per_city_counts.get(c, 0)}")

    log.info("=" * 70)
    log.info(
        f"POOLED MODEL TRAINED  rows={stats.n_rows}  cities={stats.n_unique_cities}  "
        f"q50_RMSE={stats.point_rmse_c:.3f}C  coverage_80={stats.coverage_80*100:.1f}%  "
        f"-> {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
