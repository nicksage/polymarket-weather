"""
pooled_distribution_model.py — PooledQuantileDistributionModel.

A SINGLE model trained on all 50 cities pooled, that predicts a non-parametric
distribution over the day's maximum temperature via quantile regression.

Why pooled?
-----------
Per-city models are limited to ~7K training rows each.  Pooling combines
them into ~365K rows so the model can learn universal physics (dew → temp,
solar → temp) once and reuse it across cities.  Each city is identified
by:
  * a categorical `city_idx` (HGBR native categorical handling) for
    learning city-specific splits
  * the static city features already in the v2.0 schema (elevation, Köppen,
    coastal flag, hemisphere) for learning to generalize across analogous
    cities (e.g., Mexico City benefits from Denver's data because both are
    high-altitude temperate)

Why quantile regression?
------------------------
The previous TempDistributionModel produces (μ, σ) and assumes the conditional
distribution of T_max is Gaussian.  T_max is in fact skewed (heat-wave clusters
in the upper tail; cold-air pools clip the lower tail).  Quantile regression
makes no parametric assumption — we learn 7 quantiles directly and integrate
empirically to get bin probabilities.

Public API
----------
fit(X, y, dates, cities)         — pool training across cities
predict_quantiles(X, city)       — dict{q -> value} per row
predict(X, city)                 — (mu, sigma) for backwards compat
bin_probability(low, high, ..)   — empirical CDF integration over a temp range
save(path) / load(path)          — joblib persistence with version + city map
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

# Quantiles we train on.  Symmetric set for empirical CDF construction.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# Approximate Gaussian-equivalent sigma factor: q90 - q10 ≈ 2.563 * σ
# (the 10th and 90th percentiles of a standard normal are at ±1.282σ)
_GAUSSIAN_80PCT_FACTOR = 2.563


# ---------------------------------------------------------------------------
# Stats container
# ---------------------------------------------------------------------------

@dataclass
class PooledStats:
    n_rows:           int
    n_unique_dates:   int
    n_unique_cities:  int
    n_folds:          int
    feature_version:  str
    quantiles:        list[float]
    point_rmse_c:     float                       # OOF RMSE on q50 across all rows
    mean_bias_c:      float
    coverage_80:      float                       # frac of actuals within q10..q90
    coverage_50:      float                       # frac within q25..q75
    per_city_rmse:    dict[str, float] = field(default_factory=dict)
    window_start:     str | None = None
    window_end:       str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Empirical CDF helpers
# ---------------------------------------------------------------------------

def _cdf_from_quantiles(qs: np.ndarray, vs: np.ndarray, x: float) -> float:
    """Piecewise-linear interpolation of the CDF at point `x` given samples
    (qs[i], vs[i]) where qs are quantile levels in [0,1] and vs are the
    predicted T_max values.  Linear extrapolation in tails, clamped to [0, 1].

    qs and vs must be sorted by qs (ascending).
    """
    if x <= vs[0]:
        # Lower tail — linearly extrapolate using slope of (q5, q10)
        if vs[1] > vs[0]:
            slope = (qs[1] - qs[0]) / (vs[1] - vs[0])
        else:
            slope = 0.0
        return max(0.0, qs[0] + slope * (x - vs[0]))
    if x >= vs[-1]:
        # Upper tail
        if vs[-1] > vs[-2]:
            slope = (qs[-1] - qs[-2]) / (vs[-1] - vs[-2])
        else:
            slope = 0.0
        return min(1.0, qs[-1] + slope * (x - vs[-1]))
    # Find bracketing pair via numpy searchsorted
    i = int(np.searchsorted(vs, x, side="right") - 1)
    i = max(0, min(i, len(vs) - 2))
    if vs[i + 1] > vs[i]:
        frac = (x - vs[i]) / (vs[i + 1] - vs[i])
    else:
        frac = 0.0
    return qs[i] + frac * (qs[i + 1] - qs[i])


def _ensure_monotonic(values: np.ndarray) -> np.ndarray:
    """Sort + force strictly non-decreasing.  Quantile regressors can
    occasionally cross (q90 < q75) for out-of-sample inputs; we patch by
    enforcing non-decreasing in place.  Tiny epsilon between equal values
    keeps the interpolator well-behaved."""
    out = np.sort(values).astype(np.float64)
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1] + 1e-6
    return out


# ---------------------------------------------------------------------------
# Pooled quantile model
# ---------------------------------------------------------------------------

class PooledQuantileDistributionModel:
    """Multi-city, multi-quantile gradient boosting regressor for T_max."""

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 400,
        max_leaf_nodes: int = 63,           # larger leaf budget for pooled
        min_samples_leaf: int = 30,
        random_state: int = 42,
    ):
        self.quantiles = tuple(sorted(quantiles))
        self._hparams = {
            "learning_rate":    learning_rate,
            "max_iter":         max_iter,
            "max_leaf_nodes":   max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "random_state":     random_state,
        }
        self.quantile_models: dict[float, HistGradientBoostingRegressor] = {}
        self.city_to_idx: dict[str, int] = {}
        self.idx_to_city: dict[int, str] = {}
        self.feature_version: str = FEATURE_VERSION
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.trained_at_utc: str | None = None
        self.version: str | None = None
        self.stats: PooledStats | None = None

    # -------------------------------------------------------------------
    # City categorical encoding
    # -------------------------------------------------------------------

    def _build_city_map(self, cities: list[str]) -> None:
        unique = sorted(set(cities))
        self.city_to_idx = {c: i for i, c in enumerate(unique)}
        self.idx_to_city = {i: c for c, i in self.city_to_idx.items()}

    def _append_city_col(self, X: np.ndarray, cities: list[str] | str) -> np.ndarray:
        """Append the city categorical as the last column.  Unknown cities
        get NaN (HGBR routes those to the missing-branch automatically)."""
        if isinstance(cities, str):
            cities = [cities] * X.shape[0]
        idx_col = np.array([
            self.city_to_idx.get(c, np.nan) for c in cities
        ], dtype=np.float64).reshape(-1, 1)
        return np.hstack([X, idx_col])

    @property
    def _city_col_index(self) -> int:
        return N_FEATURES   # appended after the schema's 46 features

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: list[date],
        cities: list[str],
        n_folds: int = 5,
    ) -> PooledStats:
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"expected {N_FEATURES} feature columns (excluding city), got {X.shape[1]}"
            )
        if len(y) != X.shape[0] or len(dates) != X.shape[0] or len(cities) != X.shape[0]:
            raise ValueError("X, y, dates, cities must have matching lengths")
        if X.shape[0] < 1000:
            raise ValueError(f"need >= 1000 pooled rows; got {X.shape[0]}")

        self._build_city_map(cities)
        X_full = self._append_city_col(X, cities)
        cat_idx = [self._city_col_index]

        unique_dates = sorted(set(dates))
        n_unique = len(unique_dates)
        if n_unique < n_folds + 1:
            raise ValueError(
                f"need n_folds+1={n_folds+1} unique dates, got {n_unique}"
            )

        # Build per-row mask for each unique date once (reused across folds + quantiles)
        target_dates_np = np.array(dates)
        date_to_mask = {d: (target_dates_np == d) for d in unique_dates}

        # OOF predictions per quantile (fold by date)
        oof_per_q: dict[float, np.ndarray] = {
            q: np.full(len(y), np.nan, dtype=np.float64) for q in self.quantiles
        }
        tss = TimeSeriesSplit(n_splits=n_folds)

        for fold_idx, (train_date_idx, val_date_idx) in enumerate(
            tss.split(unique_dates)
        ):
            train_dates = {unique_dates[i] for i in train_date_idx}
            val_dates   = {unique_dates[i] for i in val_date_idx}
            train_mask = np.zeros(len(y), dtype=bool)
            val_mask   = np.zeros(len(y), dtype=bool)
            for d in train_dates:
                train_mask |= date_to_mask[d]
            for d in val_dates:
                val_mask |= date_to_mask[d]

            for q in self.quantiles:
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q,
                    categorical_features=cat_idx, **self._hparams,
                )
                m.fit(X_full[train_mask], y[train_mask])
                oof_per_q[q][val_mask] = m.predict(X_full[val_mask])

            logger.info(
                f"fold {fold_idx+1}/{n_folds}: "
                f"n_train={int(train_mask.sum())} n_val={int(val_mask.sum())}"
            )

        # OOF metrics — use q50 for point RMSE
        valid = ~np.isnan(oof_per_q[0.50])
        if valid.sum() == 0:
            raise RuntimeError("no OOF predictions produced")
        resid = y[valid] - oof_per_q[0.50][valid]
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        bias = float(np.mean(resid))

        # Coverage: fraction of actuals within q10..q90 (target 0.80) and q25..q75 (0.50)
        q10 = oof_per_q[0.10][valid]; q90 = oof_per_q[0.90][valid]
        q25 = oof_per_q[0.25][valid]; q75 = oof_per_q[0.75][valid]
        cov_80 = float(np.mean((y[valid] >= q10) & (y[valid] <= q90)))
        cov_50 = float(np.mean((y[valid] >= q25) & (y[valid] <= q75)))

        # Per-city OOF RMSE
        per_city_rmse: dict[str, float] = {}
        cities_np = np.array(cities)
        for c in self.idx_to_city.values():
            mask = (cities_np == c) & valid
            if mask.sum() >= 30:
                r = y[mask] - oof_per_q[0.50][mask]
                per_city_rmse[c] = float(np.sqrt(np.mean(r ** 2)))

        # Final fit on all data — this is the deployed model
        self.quantile_models.clear()
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q,
                categorical_features=cat_idx, **self._hparams,
            )
            m.fit(X_full, y)
            self.quantile_models[q] = m

        self.trained_at_utc = datetime.now(timezone.utc).isoformat()
        self.version = (
            f"{FEATURE_VERSION}-pooled-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        self.stats = PooledStats(
            n_rows=int(X.shape[0]),
            n_unique_dates=n_unique,
            n_unique_cities=len(self.city_to_idx),
            n_folds=n_folds,
            feature_version=FEATURE_VERSION,
            quantiles=list(self.quantiles),
            point_rmse_c=rmse,
            mean_bias_c=bias,
            coverage_80=cov_80,
            coverage_50=cov_50,
            per_city_rmse=per_city_rmse,
            window_start=unique_dates[0].isoformat(),
            window_end=unique_dates[-1].isoformat(),
        )
        logger.info(
            f"[pooled] trained: n={X.shape[0]} cities={len(self.city_to_idx)} "
            f"q50_RMSE={rmse:.3f}C bias={bias:+.3f}C "
            f"coverage_80={cov_80*100:.1f}% (target 80%) "
            f"coverage_50={cov_50*100:.1f}% (target 50%)"
        )
        return self.stats

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def predict_quantiles(
        self, X: np.ndarray, city: str | list[str],
    ) -> np.ndarray:
        """Predict quantile values.  Returns array shaped (n_rows, n_quantiles)
        in the order of self.quantiles.  Always non-decreasing across the
        quantile axis (rare crossings are repaired)."""
        if not self.quantile_models:
            raise RuntimeError("model is not fitted")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {X.shape[1]}")
        X_full = self._append_city_col(X, city)
        out = np.empty((X.shape[0], len(self.quantiles)), dtype=np.float64)
        for j, q in enumerate(self.quantiles):
            out[:, j] = self.quantile_models[q].predict(X_full)
        # Repair non-monotone rows
        for i in range(out.shape[0]):
            out[i, :] = _ensure_monotonic(out[i, :])
        return out

    def predict(self, X: np.ndarray, city: str) -> tuple[float, float]:
        """Backwards-compat (μ, σ) predict for a single row.  μ = q50,
        σ derived from the q90 - q10 spread (Gaussian-equivalent factor)."""
        qmat = self.predict_quantiles(X, city)
        row = qmat[0]
        med_idx = self.quantiles.index(0.50)
        q10_idx = self.quantiles.index(0.10) if 0.10 in self.quantiles else 0
        q90_idx = self.quantiles.index(0.90) if 0.90 in self.quantiles else len(self.quantiles) - 1
        mu = float(row[med_idx])
        spread = float(row[q90_idx] - row[q10_idx])
        sigma = max(spread / _GAUSSIAN_80PCT_FACTOR, 0.1)
        return mu, sigma

    def bin_probability(
        self,
        X: np.ndarray,
        city: str,
        range_low_c: float | None,
        range_high_c: float | None,
    ) -> float:
        """P(range_low <= T_max < range_high) under the empirical CDF derived
        from the quantile predictions.  None bounds map to ±∞."""
        qmat = self.predict_quantiles(X, city)
        row = qmat[0]
        qs = np.array(self.quantiles, dtype=np.float64)
        lo_cdf = 0.0 if range_low_c  is None else _cdf_from_quantiles(qs, row, float(range_low_c))
        hi_cdf = 1.0 if range_high_c is None else _cdf_from_quantiles(qs, row, float(range_high_c))
        return max(0.0, min(1.0, hi_cdf - lo_cdf))

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if not self.quantile_models:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind":             "PooledQuantileDistributionModel",
            "format_version":   1,
            "feature_version":  self.feature_version,
            "feature_names":    self.feature_names,
            "quantiles":        list(self.quantiles),
            "city_to_idx":      self.city_to_idx,
            "version":          self.version,
            "trained_at_utc":   self.trained_at_utc,
            "hparams":          self._hparams,
            "quantile_models":  self.quantile_models,
            "stats":            self.stats.as_dict() if self.stats else None,
        }
        joblib.dump(payload, path)
        logger.info(f"saved {path} ({self.version})")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PooledQuantileDistributionModel:
        path = Path(path)
        payload = joblib.load(path)
        if payload.get("kind") != "PooledQuantileDistributionModel":
            raise ValueError(f"{path} is not a PooledQuantileDistributionModel file")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError(
                f"{path} was trained with feature_version={payload['feature_version']}, "
                f"current schema is {FEATURE_VERSION}.  Retrain."
            )
        m = cls(quantiles=tuple(payload["quantiles"]), **payload.get("hparams", {}))
        m.quantile_models = payload["quantile_models"]
        m.city_to_idx     = payload["city_to_idx"]
        m.idx_to_city     = {i: c for c, i in m.city_to_idx.items()}
        m.feature_version = payload["feature_version"]
        m.feature_names   = payload["feature_names"]
        m.version         = payload.get("version")
        m.trained_at_utc  = payload.get("trained_at_utc")
        stats_d = payload.get("stats")
        if stats_d is not None:
            m.stats = PooledStats(**stats_d)
        return m
