"""
test_predictor.py — Locks in correct behavior of the prediction pipeline.

Run:
    cd bot
    python -m pytest tests/test_predictor.py -v
        # or, without pytest:
    python tests/test_predictor.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.intraday_predictor import (
    estimate_day_high_dist,
    truncated_normal_prob,
    bin_temp_range,
    normal_cdf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(actual, expected, tol=0.5, msg=""):
    """Approximate float equality with helpful failure message."""
    assert abs(actual - expected) <= tol, (
        f"{msg}\n  expected ≈ {expected:.3f} (tol={tol})\n  actual   = {actual:.3f}"
    )


def assert_range(actual, lo, hi, msg=""):
    assert lo <= actual <= hi, (
        f"{msg}\n  expected in [{lo:.3f}, {hi:.3f}]\n  actual = {actual:.3f}"
    )


# ---------------------------------------------------------------------------
# estimate_day_high_dist — pre-peak (morning) scenarios
# ---------------------------------------------------------------------------

def test_morning_observation_does_not_collapse_mu():
    """REGRESSION: Denver 2026-06-10 bug.  At 1pm with morning obs at 26°C
    and forecast peak at 30.6°C @ 4pm, the model used to pull mu down to
    27°C and collapse sigma to 0.33°C, then bought '≤83°F' at 100% our_p.
    The fix: gate observed-peak narrowing on day_has_likely_peaked."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.6,
        forecast_peak_hour = 16,
        observed_max       = 26.0,    # morning obs, well below forecast
        observed_peak_hour = 10,      # 10am
        current_hour       = 13,      # 1pm
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    # mu should stay essentially at forecast (no narrowing/blending)
    assert_close(mu, 30.6, tol=0.5, msg="mu should remain near forecast pre-peak")
    # sigma should NOT collapse (no narrowing branch should fire)
    assert_range(sigma, 1.5, 2.5, msg="sigma should stay near base pre-peak")


def test_morning_with_no_observations():
    """No obs at all (early morning, station hasn't published yet)."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.0,
        forecast_peak_hour = 16,
        observed_max       = 30.0,    # predict_bins passes forecast as obs when none
        observed_peak_hour = None,
        current_hour       = 6,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    assert_close(mu, 30.0, tol=0.1, msg="mu stays at forecast with no obs")
    assert_close(sigma, 2.0, tol=0.1, msg="sigma stays at base with no obs")


def test_obs_just_under_forecast_with_day_still_warming():
    """Obs at 29°C @ 2pm, forecast 30.5°C peak @ 4pm.  Still warming.
    Should NOT narrow yet (day_has_likely_peaked = False)."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.5,
        forecast_peak_hour = 16,
        observed_max       = 29.0,
        observed_peak_hour = 14,
        current_hour       = 14,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    assert_close(mu, 30.5, tol=0.5, msg="mu stays near forecast pre-peak")
    assert_range(sigma, 1.5, 2.5, msg="sigma stays at base pre-peak")


# ---------------------------------------------------------------------------
# estimate_day_high_dist — peak / post-peak scenarios
# ---------------------------------------------------------------------------

def test_obs_reaches_forecast_triggers_narrowing():
    """Obs hits 29.5°C, forecast 30.0°C.  Within 1°C → 'day has peaked'.
    Narrowing branch fires."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.0,
        forecast_peak_hour = 16,
        observed_max       = 29.5,
        observed_peak_hour = 12,
        current_hour       = 13,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    # mu blends slightly toward observed
    assert_range(mu, 29.5, 30.0, msg="mu blends between obs and forecast")
    # sigma narrows from base (1h since obs peak, ~0.7x narrowing)
    assert_range(sigma, 1.0, 2.0, msg="sigma narrows post-obs-peak")


def test_post_forecast_peak_narrowing():
    """1h past forecast peak hour with obs matching forecast."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.0,
        forecast_peak_hour = 16,
        observed_max       = 30.2,
        observed_peak_hour = 16,
        current_hour       = 17,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    # Obs > forecast, mu pulled up
    assert_range(mu, 30.0, 30.7, msg="mu pulled to obs+epsilon when obs > forecast")
    # Multiple narrowing branches fire — sigma reduces significantly
    assert_range(sigma, 0.5, 1.8, msg="sigma narrows post-peak")


def test_late_afternoon_well_past_peak():
    """3h past peak, obs is the day high.  Aggressive narrowing."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.0,
        forecast_peak_hour = 16,
        observed_max       = 30.5,
        observed_peak_hour = 16,
        current_hour       = 19,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    # mu locked near observed
    assert_close(mu, 30.5, tol=0.5, msg="mu locks at observed late post-peak")
    # sigma very narrow (multiple branches multiplied)
    assert_range(sigma, 0.3, 1.2, msg="sigma very narrow late post-peak")


def test_after_sunset_locks_to_observed():
    """Past sunset → mu locked, sigma floored."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 30.0,
        forecast_peak_hour = 16,
        observed_max       = 30.5,
        observed_peak_hour = 16,
        current_hour       = 21,       # past sunset (20)
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    assert_close(mu, 30.5, tol=0.1, msg="mu locked at observed after sunset")
    assert_close(sigma, 0.3, tol=0.05, msg="sigma at floor after sunset")


# ---------------------------------------------------------------------------
# estimate_day_high_dist — observation exceeds forecast
# ---------------------------------------------------------------------------

def test_observed_exceeds_forecast_pulls_mu_up():
    """Hotter than forecast.  FIX 1a should pull mu up."""
    mu, sigma = estimate_day_high_dist(
        forecast_high      = 28.0,
        forecast_peak_hour = 16,
        observed_max       = 31.0,    # 3°C hotter than forecast
        observed_peak_hour = 15,
        current_hour       = 15,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    # mu should be observed + 0.3 (= 31.3) or higher
    assert_range(mu, 31.0, 32.5, msg="mu pulled up when obs > forecast")


# ---------------------------------------------------------------------------
# Neighbor signal interaction
# ---------------------------------------------------------------------------

def test_strong_cooling_neighbor_narrows_sigma():
    """Upwind cooling signal narrows sigma."""
    base_no_neighbor = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=30.0,
        observed_peak_hour=16, current_hour=17, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
    )
    with_neighbor = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=30.0,
        observed_peak_hour=16, current_hour=17, sunset_hour=20,
        neighbor_signal={"strong_cooling_signal": True, "signal_strength": 0.8},
        base_sigma_c=2.0,
    )
    assert with_neighbor[1] < base_no_neighbor[1], (
        "Cooling neighbor signal should narrow sigma further"
    )


# ---------------------------------------------------------------------------
# truncated_normal_prob — math sanity
# ---------------------------------------------------------------------------

def test_truncated_normal_centered_bin():
    """Bin centered on mu should get most of the probability mass."""
    p = truncated_normal_prob(lo=29.5, hi=30.5, mu=30.0, sigma=1.0, truncate_at=-100)
    # ~38% of a N(30, 1) lies in [29.5, 30.5]
    assert_range(p, 0.30, 0.45, msg="centered bin gets ~38% mass")


def test_truncated_normal_open_high():
    """'or higher' bin captures upper tail."""
    p = truncated_normal_prob(lo=32.0, hi=None, mu=30.0, sigma=1.0, truncate_at=-100)
    # P(X >= 32 | X ~ N(30, 1)) = P(Z >= 2) ≈ 0.025
    assert_range(p, 0.01, 0.05, msg="'or higher' captures tail correctly")


def test_truncated_normal_open_low():
    """'or below' bin captures lower tail."""
    p = truncated_normal_prob(lo=None, hi=28.0, mu=30.0, sigma=1.0, truncate_at=-100)
    # P(X <= 28 | X ~ N(30, 1)) = P(Z <= -2) ≈ 0.025
    assert_range(p, 0.01, 0.05, msg="'or below' captures tail correctly")


def test_truncated_normal_with_truncation():
    """Truncation at observed_max removes lower mass and renormalizes."""
    # With truncate_at=29 (we've observed 29°C, day-high can't be below it)
    p = truncated_normal_prob(lo=None, hi=28.0, mu=30.0, sigma=1.0, truncate_at=29.0)
    # Bin is entirely below truncation → probability 0
    assert_close(p, 0.0, tol=0.001, msg="bin below truncate_at gets 0 mass")


def test_truncated_normal_bin_split_by_truncation():
    """Bin [28, 31] partially below truncate_at=29.
    Effective bin becomes [29, 31], renormalized."""
    p = truncated_normal_prob(lo=28.0, hi=31.0, mu=30.0, sigma=1.0, truncate_at=29.0)
    # Bin straddles truncation — should get significant mass
    assert_range(p, 0.6, 1.0, msg="straddle bin gets most of upper mass")


# ---------------------------------------------------------------------------
# bin_temp_range — Polymarket bin translation
# ---------------------------------------------------------------------------

def test_bin_temp_range_celsius_single():
    """'28°C' bin = actual in [27.5, 28.5)°C."""
    lo, hi = bin_temp_range({"range_low": 28, "range_high": 28, "unit": "celsius"})
    assert_close(lo, 27.5, tol=0.01)
    assert_close(hi, 28.5, tol=0.01)


def test_bin_temp_range_fahrenheit_double():
    """'88-89°F' bin = actual in [87.5, 89.5)°F = [30.83, 31.94)°C."""
    lo, hi = bin_temp_range({"range_low": 88, "range_high": 89, "unit": "fahrenheit"})
    assert_close(lo, (87.5 - 32) * 5/9, tol=0.05)
    assert_close(hi, (89.5 - 32) * 5/9, tol=0.05)


def test_bin_temp_range_or_below_fahrenheit():
    """'83°F or below' = actual <= 83.5°F (range_low=None, range_high=83)."""
    lo, hi = bin_temp_range({"range_low": None, "range_high": 83, "unit": "fahrenheit"})
    assert lo is None
    assert_close(hi, (83.5 - 32) * 5/9, tol=0.05, msg="'or below' upper bound = label+0.5°F")


def test_bin_temp_range_or_higher_fahrenheit():
    """'92°F or higher' = actual >= 91.5°F (range_low=92, range_high=None)."""
    lo, hi = bin_temp_range({"range_low": 92, "range_high": None, "unit": "fahrenheit"})
    assert_close(lo, (91.5 - 32) * 5/9, tol=0.05, msg="'or higher' lower bound = label-0.5°F")
    assert hi is None


# ---------------------------------------------------------------------------
# Ensemble forecast integration
# ---------------------------------------------------------------------------

def test_ensemble_disabled_when_not_provided():
    """No ensemble_stats arg → behaves identically to legacy single-station."""
    a = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=30.0,
        observed_peak_hour=14, current_hour=15, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
    )
    b = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=30.0,
        observed_peak_hour=14, current_hour=15, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0, ensemble_stats=None,
    )
    assert a == b, "passing ensemble_stats=None should be a no-op"


def test_ensemble_agreement_does_not_inflate_sigma():
    """When neighbors agree closely (std <= 0.5°C), sigma should be unchanged."""
    no_ens = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
    )
    with_agreement = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
        ensemble_stats={
            "n_stations_used": 5, "ensemble_median": 30.0, "ensemble_std": 0.3,
            "divergence_c": 0.0, "settlement_is_outlier": False,
        },
    )
    assert_close(no_ens[1], with_agreement[1], tol=0.05,
                 msg="agreement → no sigma inflation")


def test_ensemble_disagreement_inflates_sigma():
    """High ensemble std → sigma inflates."""
    base = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
    )
    inflated = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
        ensemble_stats={
            "n_stations_used": 5, "ensemble_median": 30.0,
            "ensemble_std": 2.0,    # high disagreement
            "divergence_c": 0.0, "settlement_is_outlier": False,
        },
    )
    assert inflated[1] > base[1], (
        f"high ensemble std should inflate sigma: base={base[1]:.2f} vs "
        f"inflated={inflated[1]:.2f}"
    )


def test_ensemble_outlier_settlement_blends_mu():
    """If settlement diverges from ensemble median, mu blends toward median."""
    # Settlement says 30°C but neighbors median 33°C (3°C divergence)
    no_ens = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
    )
    with_ens = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
        ensemble_stats={
            "n_stations_used": 5,
            "ensemble_median": 33.0,
            "ensemble_std": 0.8,
            "divergence_c": -3.0,             # settlement is 3°C colder than median
            "settlement_is_outlier": True,
        },
    )
    assert with_ens[0] > no_ens[0], (
        f"outlier-settlement should pull mu up: no_ens={no_ens[0]:.2f} vs "
        f"with_ens={with_ens[0]:.2f}"
    )
    # but should be capped — should NOT exceed median
    assert with_ens[0] <= 33.0, "mu should not exceed ensemble median"


def test_ensemble_with_few_stations_ignored():
    """If only 1-2 stations succeeded, don't apply ensemble adjustments."""
    no_ens = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
    )
    sparse = estimate_day_high_dist(
        forecast_high=30.0, forecast_peak_hour=16, observed_max=-100,
        observed_peak_hour=None, current_hour=10, sunset_hour=20,
        neighbor_signal={}, base_sigma_c=2.0,
        ensemble_stats={
            "n_stations_used": 2,    # too few — should be ignored
            "ensemble_median": 33.0, "ensemble_std": 2.0,
            "divergence_c": -3.0, "settlement_is_outlier": True,
        },
    )
    assert no_ens == sparse, "ensemble with <3 stations should be no-op"


# ---------------------------------------------------------------------------
# Test runner (use either pytest or python directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    n_passed = n_failed = 0
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            n_failed += 1
            failures.append((t.__name__, str(e)))
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            n_failed += 1
    print()
    print(f"{'=' * 60}")
    print(f"  {n_passed} passed, {n_failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if n_failed == 0 else 1)