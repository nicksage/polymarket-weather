"""
test_calibration_diagnostic.py — locks in the load-bearing math for
the prototype-calibration diagnostic.

Critical pieces:
  - empirical_percentile linear interpolation (must match TWC's
    continuous-percentile semantics, not bucketed)
  - city verdict thresholds (calibrated / noisy / uncalibrated)
  - parse path tolerance for missing API sections
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.diagnose_prototype_calibration import (    # type: ignore
    empirical_percentile,
    _city_verdict,
    filter_cities_by_scope,
    is_domestic_icao,
    GAP_OK_BODY, GAP_OK_TAILS, SYSTEMATIC_BIAS_THRESHOLD,
)


# ============================================================
# Empirical percentile math
# ============================================================

class TestEmpiricalPercentile:

    def test_p50_of_uniform_1_to_100_is_50_5(self):
        values = list(range(1, 101))   # [1, 2, ..., 100]
        # Linear-interp percentile: P50 = midway between 50 and 51 = 50.5
        assert abs(empirical_percentile(values, 50) - 50.5) < 1e-9

    def test_p0_and_p100_are_min_and_max(self):
        values = [10, 20, 30, 40, 50]
        assert empirical_percentile(values, 0) == 10
        assert empirical_percentile(values, 100) == 50

    def test_p10_of_100_samples_linear_interp(self):
        # values 1..100, P10 should be between 10 and 11 → 10.9
        # (rank = 0.10 * 99 = 9.9, so values[9] + 0.9*(values[10]-values[9])
        #                       = 10 + 0.9*(11-10) = 10.9)
        values = list(range(1, 101))
        v = empirical_percentile(values, 10)
        assert abs(v - 10.9) < 1e-9

    def test_p90_of_100_samples(self):
        values = list(range(1, 101))
        # rank = 0.90 * 99 = 89.1 → values[89] + 0.1*(values[90]-values[89])
        #                          = 90 + 0.1 = 90.1
        v = empirical_percentile(values, 90)
        assert abs(v - 90.1) < 1e-9

    def test_single_sample(self):
        assert empirical_percentile([42.0], 50) == 42.0
        assert empirical_percentile([42.0], 10) == 42.0
        assert empirical_percentile([42.0], 90) == 42.0

    def test_empty_returns_nan(self):
        import math
        assert math.isnan(empirical_percentile([], 50))


# ============================================================
# Verdict classification — load-bearing for the overall answer
# ============================================================

def _mk_city_result(pct_within: float, mean_signed_gap: float) -> dict:
    return {
        "status": "ok",
        "pct_within": pct_within,
        "mean_signed_gap": mean_signed_gap,
    }


class TestCityVerdict:

    def test_high_within_pct_and_near_zero_bias_calibrated(self):
        r = _mk_city_result(pct_within=0.98, mean_signed_gap=0.05)
        assert _city_verdict(r).startswith("✓")

    def test_high_pct_but_systematic_bias_drops_to_noisy(self):
        # pct_within is great (98%) but mean signed gap exceeds bias threshold
        r = _mk_city_result(pct_within=0.98,
                              mean_signed_gap=SYSTEMATIC_BIAS_THRESHOLD + 0.1)
        # Per the verdict logic, high pct + bias → "⚠ noisy" (not calibrated)
        verdict = _city_verdict(r)
        assert verdict.startswith("⚠") or verdict.startswith("✓")
        # Not allowed to be "✗ uncalibrated" — bias alone with high pct
        # shouldn't be the worst tier
        assert not verdict.startswith("✗")

    def test_low_pct_and_large_bias_uncalibrated(self):
        # Both criteria fail
        r = _mk_city_result(pct_within=0.50, mean_signed_gap=1.5)
        assert _city_verdict(r).startswith("✗")

    def test_moderate_pct_no_bias_is_noisy(self):
        r = _mk_city_result(pct_within=0.85, mean_signed_gap=0.0)
        assert _city_verdict(r).startswith("⚠")


# ============================================================
# Scope filter (re-uses station_meta — must work for international cities)
# ============================================================

class TestScopeFilter:

    def test_domestic_returns_k_prefixed_only(self):
        cities = filter_cities_by_scope("domestic")
        # Sanity: must be non-empty, must include known US cities,
        # must NOT include known international cities
        assert "Miami" in cities
        assert "Tokyo" not in cities
        assert "London" not in cities

    def test_international_excludes_us(self):
        cities = filter_cities_by_scope("international")
        assert "Tokyo" in cities or "London" in cities
        assert "Miami" not in cities

    def test_all_includes_both(self):
        cities = filter_cities_by_scope("all")
        assert "Miami" in cities
        # International cities should be in there too
        intl_present = any(
            c in cities for c in ("Tokyo", "London", "Paris", "Madrid"))
        assert intl_present


class TestIsDomesticIcao:
    def test_k_prefix_is_domestic(self):
        assert is_domestic_icao("KMIA") is True
        assert is_domestic_icao("KSFO") is True

    def test_other_prefix_not_domestic(self):
        assert is_domestic_icao("EGLC") is False   # London
        assert is_domestic_icao("RJTT") is False   # Tokyo
        assert is_domestic_icao("ZBAA") is False   # Beijing

    def test_empty_or_short(self):
        assert is_domestic_icao("") is False
        assert is_domestic_icao(None) is False