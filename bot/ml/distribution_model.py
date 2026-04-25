"""
distribution_model.py — TempDistributionModel.

A per-city learner that predicts a Gaussian distribution N(μ, σ) over the
day's maximum temperature from an observation-driven feature vector.

Mean model
----------
HistGradientBoostingRegressor — sklearn's native tabular learner.  Handles
NaN inputs (common in our feature set: snowdepth and climatology are
frequently null) without requiring imputation.

Spread estimation
-----------------
σ is learned from out-of-fold residuals during cross-validation, bucketed
by target-date month.  Months with fewer than MIN_BUCKET_SAMPLES OOF
observations fall back to the city-wide residual std.  This captures the
seasonal variation in forecast difficulty without overfitting.

Temporal integrity
------------------
Cross-validation uses TimeSeriesSplit across UNIQUE target dates — both
the 10:00 and 12:00 rows for a single date land in the same fold, so
label information from one cannot leak into the other's training set.

Persistence
-----------
joblib pickle containing the sklearn estimator, spread dict, feature-name
contract, training stats, and version stamp.  Loading verifies the pickle's
FEATURE_VERSION matches the current schema — refuses to load stale models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from ml.schema import FEATURE_NAMES, N_FEATURES, FEATURE_VERSION

logger = logging.getLogger(__name__)

# Minimum OOF observations per month before its dedicated σ is trusted.
# Below this threshold the month falls back to the city-wide residual std.
MIN_BUCKET_SAMPLES = 30


# ---------------------------------------------------------------------------
# Training stats container
# ---------------------------------------------------------------------------

@dataclass
class TrainingStats:
    n_rows:           int
    n_unique_dates:   int
    n_folds:          int
    feature_version:  str
    point_rmse_c:     float
    residual_sigma_c: float
    mean_bias_c:      float                                     # mean(y - ypred)
    spread_per_bucket: dict[int, float] = field(default_factory=dict)
    window_start:     str | None = None
    window_end:       str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class TempDistributionModel:
    """Per-city temperature distribution regressor.  Fit once; use
    `predict(X, target_date)` at inference time to obtain (μ, σ)."""

    def __init__(
        self,
        city: str,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 400,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.city = city
        self._hparams = {
            "learning_rate":    learning_rate,
            "max_iter":         max_iter,
            "max_leaf_nodes":   max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "random_state":     random_state,
        }
        self.mean_model: HistGradientBoostingRegressor | None = None
        self.spread_per_bucket: dict[int, float] = {}
        self.fallback_sigma: float | None = None
        self.feature_version: str = FEATURE_VERSION
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.trained_at_utc: str | None = None
        self.version: str | None = None
        self.stats: TrainingStats | None = None

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_dates: list[date],
        n_folds: int = 5,
    ) -> TrainingStats:
        """Fit the mean model with TimeSeriesSplit grouped by date, collect
        OOF residuals, compute per-month σ.  Refit mean model on all data."""
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"expected {N_FEATURES} features, got {X.shape[1]} "
                f"(feature_version={FEATURE_VERSION})"
            )
        if X.shape[0] != y.shape[0] or X.shape[0] != len(target_dates):
            raise ValueError("X, y, target_dates must have matching lengths")
        if X.shape[0] < 100:
            raise ValueError(f"need ≥100 training rows, got {X.shape[0]}")

        # Unique dates, chronologically ordered
        unique_dates = sorted(set(target_dates))
        n_unique = len(unique_dates)
        if n_unique < n_folds + 1:
            raise ValueError(
                f"need at least n_folds+1={n_folds+1} unique dates, got {n_unique}"
            )
        date_to_mask: dict[date, np.ndarray] = {}
        target_dates_np = np.array(target_dates)
        for d in unique_dates:
            date_to_mask[d] = (target_dates_np == d)

        # --- Out-of-fold predictions ---
        oof_pred = np.full(len(y), np.nan, dtype=np.float64)
        tss = TimeSeriesSplit(n_splits=n_folds)

        for fold_idx, (train_date_idx, val_date_idx) in enumerate(
            tss.split(unique_dates)
        ):
            train_dates = {unique_dates[i] for i in train_date_idx}
            val_dates   = {unique_dates[i] for i in val_date_idx}

            # Build row-level masks
            train_mask = np.zeros(len(y), dtype=bool)
            val_mask   = np.zeros(len(y), dtype=bool)
            for d in train_dates:
                train_mask |= date_to_mask[d]
            for d in val_dates:
                val_mask   |= date_to_mask[d]

            fold_model = HistGradientBoostingRegressor(**self._hparams)
            fold_model.fit(X[train_mask], y[train_mask])
            oof_pred[val_mask] = fold_model.predict(X[val_mask])

            logger.debug(
                f"fold {fold_idx+1}/{n_folds}: "
                f"n_train={train_mask.sum()} n_val={val_mask.sum()}"
            )

        # --- OOF metrics ---
        valid = ~np.isnan(oof_pred)
        if valid.sum() == 0:
            raise RuntimeError("no OOF predictions produced (all folds failed?)")
        resid = y[valid] - oof_pred[valid]
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mean_bias = float(np.mean(resid))
        fallback_sigma = float(np.std(resid))

        # --- Per-month spread ---
        months = np.array([d.month for d in target_dates])[valid]
        spread_per_bucket: dict[int, float] = {}
        for m in range(1, 13):
            m_mask = months == m
            if m_mask.sum() >= MIN_BUCKET_SAMPLES:
                spread_per_bucket[m] = float(np.std(resid[m_mask]))

        # --- Final fit on all data ---
        self.mean_model = HistGradientBoostingRegressor(**self._hparams)
        self.mean_model.fit(X, y)

        self.spread_per_bucket = spread_per_bucket
        self.fallback_sigma    = fallback_sigma
        self.trained_at_utc    = datetime.now(timezone.utc).isoformat()
        self.version           = f"{FEATURE_VERSION}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

        self.stats = TrainingStats(
            n_rows           = int(X.shape[0]),
            n_unique_dates   = n_unique,
            n_folds          = n_folds,
            feature_version  = FEATURE_VERSION,
            point_rmse_c     = rmse,
            residual_sigma_c = fallback_sigma,
            mean_bias_c      = mean_bias,
            spread_per_bucket = {str(k): v for k, v in spread_per_bucket.items()},
            window_start     = unique_dates[0].isoformat(),
            window_end       = unique_dates[-1].isoformat(),
        )
        logger.info(
            f"[{self.city}] trained: n={X.shape[0]} RMSE={rmse:.3f}°C "
            f"σ_fallback={fallback_sigma:.3f}°C bias={mean_bias:+.3f}°C "
            f"per-month σ: {{{', '.join(f'{m}:{s:.2f}' for m,s in sorted(spread_per_bucket.items()))}}}"
        )
        return self.stats

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        target_date: date,
    ) -> tuple[float, float]:
        """Predict (μ, σ) for a single feature vector at target_date.
        σ is drawn from the month-specific bucket with fallback to the
        city-wide residual std."""
        if self.mean_model is None:
            raise RuntimeError("model is not fitted; call fit() or load()")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"expected {N_FEATURES} features, got {X.shape[1]}"
            )
        mu = float(self.mean_model.predict(X)[0])
        sigma = self.spread_per_bucket.get(target_date.month, self.fallback_sigma)
        if sigma is None:
            raise RuntimeError("no σ available — model was not properly fit")
        return mu, float(sigma)

    def predict_batch(
        self,
        X: np.ndarray,
        target_dates: list[date],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized multi-row predict.  Returns (mu_arr, sigma_arr)."""
        if self.mean_model is None:
            raise RuntimeError("model is not fitted")
        if X.shape[1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {X.shape[1]}")
        mu_arr = self.mean_model.predict(X).astype(np.float64)
        sigma_arr = np.array([
            self.spread_per_bucket.get(d.month, self.fallback_sigma)
            for d in target_dates
        ], dtype=np.float64)
        return mu_arr, sigma_arr

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if self.mean_model is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind":              "TempDistributionModel",
            "format_version":    1,
            "feature_version":   self.feature_version,
            "feature_names":     self.feature_names,
            "city":              self.city,
            "version":           self.version,
            "trained_at_utc":    self.trained_at_utc,
            "hparams":           self._hparams,
            "mean_model":        self.mean_model,
            "spread_per_bucket": self.spread_per_bucket,
            "fallback_sigma":    self.fallback_sigma,
            "stats":             self.stats.as_dict() if self.stats else None,
        }
        joblib.dump(payload, path)
        logger.info(f"saved {path} ({self.city} {self.version})")
        return path

    @classmethod
    def load(cls, path: str | Path) -> TempDistributionModel:
        path = Path(path)
        payload = joblib.load(path)
        if payload.get("kind") != "TempDistributionModel":
            raise ValueError(f"{path} is not a TempDistributionModel file")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError(
                f"{path} was trained with feature_version={payload['feature_version']}, "
                f"but the current schema is {FEATURE_VERSION}.  Retrain."
            )
        model = cls(city=payload["city"], **payload.get("hparams", {}))
        model.mean_model        = payload["mean_model"]
        model.spread_per_bucket = payload["spread_per_bucket"]
        model.fallback_sigma    = payload["fallback_sigma"]
        model.feature_version   = payload["feature_version"]
        model.feature_names     = payload["feature_names"]
        model.version           = payload.get("version")
        model.trained_at_utc    = payload.get("trained_at_utc")
        stats_d = payload.get("stats")
        if stats_d is not None:
            # Restore int keys in spread_per_bucket if they survived as strs
            spb = stats_d.get("spread_per_bucket") or {}
            stats_d["spread_per_bucket"] = {str(k): v for k, v in spb.items()}
            model.stats = TrainingStats(**stats_d)
        return model
