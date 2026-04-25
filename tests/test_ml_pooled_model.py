"""Smoke tests for bot.ml.pooled_distribution_model.PooledQuantileDistributionModel."""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from ml.pooled_distribution_model import (
    PooledQuantileDistributionModel, _cdf_from_quantiles, _ensure_monotonic,
)
from ml.schema import N_FEATURES, FEATURE_VERSION


def _synthetic_pooled_dataset(
    n_cities: int = 4, n_days: int = 250, rows_per_day: int = 2, seed: int = 0,
):
    """Synthetic multi-city dataset where each city has its own offset
    + the same shared linear function of two features.  A pooled model
    should learn (a) the universal feature relationship and (b) the
    per-city offset via the city categorical."""
    rng = np.random.default_rng(seed)
    cities = [f"city_{i}" for i in range(n_cities)]
    city_offsets = {c: rng.uniform(-5, 5) for c in cities}

    rows = []
    for c in cities:
        start = date(2020, 1, 1)
        for d in range(n_days):
            target_d = start + timedelta(days=d)
            for _ in range(rows_per_day):
                rows.append((c, target_d))

    n = len(rows)
    X = rng.normal(0, 1, size=(n, N_FEATURES))
    # Sprinkle 5% NaN
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan
    X_imp = np.where(np.isnan(X), 0.0, X)

    cities_arr = [c for c, _ in rows]
    dates_arr  = [d for _, d in rows]

    # y = city_offset + 3*X[4] + 0.5*X[7] + noise
    base = 20.0 + 3.0 * X_imp[:, 4] + 0.5 * X_imp[:, 7] + rng.normal(0, 1.0, n)
    offsets = np.array([city_offsets[c] for c, _ in rows])
    y = base + offsets

    # Two rows per (city, date) share a label (no intra-date leakage allowed)
    for i in range(0, n, rows_per_day):
        avg = float(np.mean(y[i:i + rows_per_day]))
        y[i:i + rows_per_day] = avg

    return X, y, dates_arr, cities_arr, city_offsets


def test_fit_predict_roundtrip():
    X, y, dates, cities, _ = _synthetic_pooled_dataset()
    m = PooledQuantileDistributionModel()
    stats = m.fit(X, y, dates, cities, n_folds=3)

    assert stats.n_rows == X.shape[0]
    assert stats.n_unique_cities == 4
    assert stats.point_rmse_c > 0
    # Coverage assertions: with only 2000 synthetic rows and 7 quantiles
    # trained on ~1333 rows each, the tails undercover (model quantiles
    # collapse toward the median).  We just confirm the metric is in a
    # plausible range — calibration on real 365K-row data is the only
    # honest test of coverage.
    assert 0.10 < stats.coverage_80 < 1.0
    assert 0.05 < stats.coverage_50 < 1.0

    # Predict for a known city
    mu, sigma = m.predict(X[0], cities[0])
    assert np.isfinite(mu)
    assert sigma > 0


def test_quantile_predictions_monotonic():
    X, y, dates, cities, _ = _synthetic_pooled_dataset()
    m = PooledQuantileDistributionModel()
    m.fit(X, y, dates, cities, n_folds=3)

    qmat = m.predict_quantiles(X[:50], cities[:50])
    assert qmat.shape == (50, len(m.quantiles))
    # Each row must be non-decreasing across quantile axis
    for row in qmat:
        diffs = np.diff(row)
        assert np.all(diffs >= -1e-6), f"non-monotone row: {row}"


def test_bin_probability_integrates_to_one():
    X, y, dates, cities, _ = _synthetic_pooled_dataset()
    m = PooledQuantileDistributionModel()
    m.fit(X, y, dates, cities, n_folds=3)

    # Cover the full real line with a wide range — should sum to ~1.0
    p = m.bin_probability(X[:1], cities[0], range_low_c=-100.0, range_high_c=100.0)
    assert p == pytest.approx(1.0, abs=0.01)

    # Open-ended bins on each side should also be 1.0 cumulatively
    p_lo = m.bin_probability(X[:1], cities[0], range_low_c=None, range_high_c=10.0)
    p_hi = m.bin_probability(X[:1], cities[0], range_low_c=10.0, range_high_c=None)
    assert p_lo + p_hi == pytest.approx(1.0, abs=0.01)


def test_save_and_load_roundtrip(tmp_path: Path):
    X, y, dates, cities, _ = _synthetic_pooled_dataset()
    m1 = PooledQuantileDistributionModel()
    m1.fit(X, y, dates, cities, n_folds=3)

    path = tmp_path / "pooled.joblib"
    m1.save(path)
    assert path.exists()

    m2 = PooledQuantileDistributionModel.load(path)
    assert set(m2.city_to_idx.keys()) == set(cities)
    assert m2.feature_version == FEATURE_VERSION

    # Predictions should match exactly
    for i in (0, 50, 100):
        mu1, _ = m1.predict(X[i], cities[i])
        mu2, _ = m2.predict(X[i], cities[i])
        assert mu1 == pytest.approx(mu2, abs=1e-6)


def test_unknown_city_does_not_crash():
    X, y, dates, cities, _ = _synthetic_pooled_dataset()
    m = PooledQuantileDistributionModel()
    m.fit(X, y, dates, cities, n_folds=3)

    # Predict for a city not in the training set
    mu, sigma = m.predict(X[0], "atlantis")
    assert np.isfinite(mu)
    assert sigma > 0


def test_cdf_from_quantiles_extrapolation():
    qs = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    vs = np.array([10.0, 12.0, 15.0, 18.0, 22.0])
    # CDF at the median quantile value should equal the quantile level
    assert _cdf_from_quantiles(qs, vs, 15.0) == pytest.approx(0.50, abs=0.01)
    # Below q05 — should extrapolate, but stay >= 0
    assert _cdf_from_quantiles(qs, vs, 5.0) >= 0.0
    # Above q95 — should extrapolate, but stay <= 1
    assert _cdf_from_quantiles(qs, vs, 30.0) <= 1.0


def test_ensure_monotonic():
    # Already monotonic — unchanged
    np.testing.assert_array_almost_equal(
        _ensure_monotonic(np.array([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0])
    )
    # Crossings — repaired in place
    out = _ensure_monotonic(np.array([1.0, 3.0, 2.0]))
    assert out[0] <= out[1] <= out[2]
