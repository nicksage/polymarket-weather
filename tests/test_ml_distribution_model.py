"""Smoke tests for bot.ml.distribution_model.TempDistributionModel."""

import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from ml.distribution_model import TempDistributionModel
from ml.schema import N_FEATURES, FEATURE_VERSION


def _synthetic_dataset(n_days: int = 400, rows_per_day: int = 2, seed: int = 0):
    """Build a synthetic (X, y, dates) dataset where y is a noisy linear
    combo of a few features.  n_days must be large enough for TimeSeriesSplit."""
    rng = np.random.default_rng(seed)
    # One row per (date, decision_hour).  Two rows per date share the label.
    dates: list[date] = []
    start = date(2020, 1, 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        for _ in range(rows_per_day):
            dates.append(d)
    n = len(dates)

    X = rng.normal(0, 1, size=(n, N_FEATURES))
    # Randomly sprinkle NaN (~5%) to mimic real feature-nullness
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan

    # Label = linear combo of features 4 (temp_now_c) and 7 (pressure_hpa) with noise
    # Fill NaN with column mean for label construction
    X_imputed = np.where(np.isnan(X), np.nanmean(X, axis=0, keepdims=True), X)
    y = 20.0 + 3.0 * X_imputed[:, 4] + 0.5 * X_imputed[:, 7] + rng.normal(0, 1.0, size=n)
    # Two rows per date should share the same label (temporal consistency)
    for i in range(0, n, rows_per_day):
        avg = float(np.mean(y[i:i + rows_per_day]))
        y[i:i + rows_per_day] = avg
    return X, y, dates


def test_fit_predict_roundtrip():
    X, y, dates = _synthetic_dataset(n_days=400)
    m = TempDistributionModel(city="Testville")
    stats = m.fit(X, y, dates, n_folds=5)
    assert stats.n_rows == X.shape[0]
    assert stats.point_rmse_c > 0
    assert stats.residual_sigma_c > 0
    # Mean bias should be near zero for a well-specified linear target
    assert abs(stats.mean_bias_c) < 0.5

    # Single-row predict
    mu, sigma = m.predict(X[0], dates[0])
    assert np.isfinite(mu)
    assert sigma > 0

    # Batch predict
    mu_arr, sigma_arr = m.predict_batch(X[:10], dates[:10])
    assert mu_arr.shape == (10,)
    assert sigma_arr.shape == (10,)
    assert np.all(sigma_arr > 0)


def test_time_series_split_no_intra_date_leakage():
    """10am and 12pm rows for the same date share a label; CV must keep
    them in the same fold to prevent trivial lookup."""
    X, y, dates = _synthetic_dataset(n_days=400, rows_per_day=2)
    m = TempDistributionModel(city="Testville")
    stats = m.fit(X, y, dates, n_folds=5)
    # If there were intra-date leakage, RMSE would be near 0 (the model would
    # just memorize shared labels).  Synthetic noise std is 1.0, so proper CV
    # should produce RMSE ≈ 1.0, not << 1.0.
    assert stats.point_rmse_c > 0.5, (
        f"RMSE {stats.point_rmse_c:.3f} too small — suspect intra-date leakage"
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    X, y, dates = _synthetic_dataset(n_days=400)
    m1 = TempDistributionModel(city="Testville")
    m1.fit(X, y, dates, n_folds=5)

    path = tmp_path / "model.joblib"
    m1.save(path)
    assert path.exists()

    m2 = TempDistributionModel.load(path)
    assert m2.city == "Testville"
    assert m2.feature_version == FEATURE_VERSION

    # Predictions should match exactly
    for i in (0, 50, 100):
        mu1, sig1 = m1.predict(X[i], dates[i])
        mu2, sig2 = m2.predict(X[i], dates[i])
        assert mu1 == pytest.approx(mu2)
        assert sig1 == pytest.approx(sig2)


def test_fit_rejects_wrong_feature_count():
    X, y, dates = _synthetic_dataset(n_days=400)
    X_wrong = X[:, :-1]   # drop one column
    m = TempDistributionModel(city="Testville")
    with pytest.raises(ValueError, match="expected"):
        m.fit(X_wrong, y, dates)


def test_fit_rejects_too_few_rows():
    X, y, dates = _synthetic_dataset(n_days=20)   # 40 rows < 100 threshold
    m = TempDistributionModel(city="Testville")
    with pytest.raises(ValueError, match="need"):
        m.fit(X, y, dates)


def test_predict_before_fit_raises():
    m = TempDistributionModel(city="Testville")
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict(np.zeros(N_FEATURES), date(2025, 7, 15))
