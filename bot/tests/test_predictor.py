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
    detect_cooling,
    find_bin_containing_temp,
    predict_bins,
    _cooling_via_obs_trajectory,
    _cooling_via_forecast_tracking,
    _cooling_via_derivative,
    STRONG_COOLING_THRESHOLD,
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
# Cooling detection helpers
# ---------------------------------------------------------------------------

def _mkobs(hours_temps):
    return [{"hour_local": h, "temp_c": t, "wind_dir_deg": None,
              "timestamp_utc": ""} for h, t in hours_temps]


def test_cooling_obs_trajectory_pre3pm_needs_2():
    """Before 15:00 local: N=2 cooling obs required."""
    obs = _mkobs([(10, 26), (11, 28), (12, 30), (13, 31), (14, 32),
                   (15, 33), (16, 32)])   # only 1 cooling obs
    # Wait, current_hour=14 is BEFORE 3pm but peak is hour 15 which is after current
    # Let me use a scenario where current is before 3pm AND we have a peak
    obs = _mkobs([(8, 24), (9, 26), (10, 28), (11, 30), (12, 31), (13, 30)])
    # observed_max=31 at hour 12, current=13 (before 3pm), 1 cooling obs (hour 13)
    conf, _ = _cooling_via_obs_trajectory(obs, observed_max=31, observed_peak_hour=12,
                                            current_hour=13)
    assert conf == 0.0, f"pre-3pm with only 1 cooling obs should be 0, got {conf}"


def test_cooling_obs_trajectory_pre3pm_with_2_obs():
    """Before 15:00 local: 2 cooling obs satisfies N=2."""
    obs = _mkobs([(8, 24), (9, 26), (10, 28), (11, 30), (12, 31), (13, 30), (14, 29)])
    conf, _ = _cooling_via_obs_trajectory(obs, observed_max=31, observed_peak_hour=12,
                                            current_hour=14)
    assert conf >= 0.5, f"pre-3pm with 2 cooling obs should be >= 0.5, got {conf}"


def test_cooling_obs_trajectory_post3pm_needs_1():
    """After 15:00 local: just 1 cooling obs is enough."""
    obs = _mkobs([(13, 31), (14, 32), (15, 33), (16, 33), (17, 32)])
    # observed_max=33 at hour 16, current=17, 1 cooling obs at hour 17
    conf, _ = _cooling_via_obs_trajectory(obs, observed_max=33, observed_peak_hour=16,
                                            current_hour=17)
    assert conf >= 0.5, f"post-3pm with 1 cooling obs should be >= 0.5, got {conf}"


def test_cooling_obs_no_obs_after_peak():
    """Peak is the latest obs — no cooling signal yet."""
    obs = _mkobs([(13, 31), (14, 32), (15, 33), (16, 33)])
    conf, _ = _cooling_via_obs_trajectory(obs, observed_max=33, observed_peak_hour=16,
                                            current_hour=17)
    assert conf == 0.0


def test_cooling_via_forecast_tracking_fires_when_cooling_forecast():
    """Forecast shows clear cooling AND obs tracking → high confidence."""
    obs = _mkobs([(13, 31), (14, 32), (15, 33), (16, 33)])
    fcst = [(13, 31), (14, 32), (15, 33), (16, 33),
            (17, 32), (18, 31), (19, 30)]   # cooling ahead
    conf, _ = _cooling_via_forecast_tracking(obs, fcst, current_hour=16)
    assert conf >= 0.5, f"matched obs + cooling forecast should be >= 0.5, got {conf}"


def test_cooling_via_forecast_tracking_zero_when_no_forecast_cooling():
    """Forecast shows warming continues → no cooling signal."""
    obs = _mkobs([(13, 31), (14, 32), (15, 33)])
    fcst = [(13, 31), (14, 32), (15, 33), (16, 34), (17, 35), (18, 35)]  # warming
    conf, _ = _cooling_via_forecast_tracking(obs, fcst, current_hour=15)
    assert conf == 0.0


def test_cooling_via_derivative_negative_slope():
    """Linear fit with negative slope → cooling signal."""
    obs = _mkobs([(14, 33), (15, 33), (16, 32.5), (17, 32), (18, 31)])
    conf, _ = _cooling_via_derivative(obs, current_hour=18)
    assert conf > 0, f"negative slope should give >0 cooling signal, got {conf}"


def test_cooling_via_derivative_positive_slope():
    """Linear fit with positive slope → 0 cooling signal."""
    obs = _mkobs([(10, 26), (11, 28), (12, 30), (13, 31)])
    conf, _ = _cooling_via_derivative(obs, current_hour=13)
    assert conf == 0.0


def test_detect_cooling_combined_houston():
    """The actual Houston scenario from 2026-06-10: bot bet on 92-93°F
    when it should have been 90-91°F.  After cooling detection fires,
    the combined confidence should be >= STRONG_COOLING_THRESHOLD (0.7)
    so that bin-lock engages."""
    # 5pm scan after 4pm peak.  17:00 obs published, shows day plateauing.
    obs = _mkobs([(10, 28), (11, 29), (12, 30), (13, 31), (14, 32),
                   (15, 33), (16, 33), (17, 32.5)])
    fcst = [(10, 28), (11, 29), (12, 30), (13, 31), (14, 32),
            (15, 32.5), (16, 32.8), (17, 32.8), (18, 32), (19, 30.5)]
    conf, reason = detect_cooling(obs, fcst, observed_max=33.0,
                                    observed_peak_hour=16, current_hour=17)
    assert conf >= STRONG_COOLING_THRESHOLD, (
        f"Houston-style scenario should trigger bin-lock "
        f"(conf={conf:.2f}, need ≥ {STRONG_COOLING_THRESHOLD}, reason={reason})"
    )


def test_detect_cooling_still_warming():
    """Pre-peak, day still warming → cooling confidence ≈ 0."""
    obs = _mkobs([(8, 22), (9, 25), (10, 28), (11, 30), (12, 31)])
    fcst = [(8, 22), (9, 25), (10, 28), (11, 30), (12, 31),
            (13, 32), (14, 32.5), (15, 33), (16, 33)]
    conf, _ = detect_cooling(obs, fcst, observed_max=31.0,
                              observed_peak_hour=12, current_hour=12)
    assert conf < 0.3, f"warming day should have low cooling confidence, got {conf}"


# ---------------------------------------------------------------------------
# find_bin_containing_temp
# ---------------------------------------------------------------------------

def test_find_bin_containing_temp_normal():
    bins = [
        {"range_low": 86, "range_high": 87, "unit": "fahrenheit"},
        {"range_low": 88, "range_high": 89, "unit": "fahrenheit"},
        {"range_low": 90, "range_high": 91, "unit": "fahrenheit"},
        {"range_low": 92, "range_high": 93, "unit": "fahrenheit"},
    ]
    # 91.4°F → 90-91°F bin (since 91.4 < 91.5 = bin's upper bound in half-step)
    found = find_bin_containing_temp(33.0, bins)   # 33.0°C = 91.4°F
    assert found is not None and found["range_low"] == 90, (
        f"33.0°C should land in 90-91°F bin, got {found}"
    )


def test_find_bin_containing_temp_above_all():
    bins = [
        {"range_low": 86, "range_high": 87, "unit": "fahrenheit"},
        {"range_low": 88, "range_high": 89, "unit": "fahrenheit"},
    ]
    # 100°F is way above — should return None (or an open-ended bin if present)
    found = find_bin_containing_temp(37.8, bins)   # 100°F
    assert found is None


# ---------------------------------------------------------------------------
# End-to-end: Houston scenario via predict_bins
# ---------------------------------------------------------------------------

def test_houston_scenario_bin_locks_correctly():
    """Regression test for the 2026-06-10 Houston bug.
    Day peaked at 33.0°C (91.4°F) at 4pm with one cooling obs at 5pm.
    Cooling should be strongly detected → bin-lock → 90-91°F gets >50% probability."""
    # Build Polymarket-style bins (full Houston bin set)
    bins = [{"range_low": lo, "range_high": hi, "unit": "fahrenheit",
              "contract_id": f"c{lo}", "yes_token_id": f"t{lo}",
              "yes_price": mkt, "liquidity_usd": 500}
             for lo, hi, mkt in [
                 (None, 79, 0.001), (80, 81, 0.001), (82, 83, 0.001),
                 (84, 85, 0.001), (86, 87, 0.001), (88, 89, 0.001),
                 (90, 91, 0.89), (92, 93, 0.095), (94, 95, 0.008),
                 (96, 97, 0.001), (98, None, 0.001),
             ]]
    event = {"outcomes": bins, "settlement_station": "KHOU"}
    # Observations: morning warming, 4pm peak, 5pm slight cooling
    obs = _mkobs([(10, 28), (11, 29), (12, 30), (13, 31), (14, 32),
                   (15, 33), (16, 33), (17, 32.5)])
    forecast = {
        "forecast_high":      32.8,
        "forecast_peak_hour": 17,
        "sunset_hour":        20,
        "hourly": [(10, 28), (11, 29), (12, 30), (13, 31), (14, 32),
                    (15, 32.5), (16, 32.8), (17, 32.8), (18, 32), (19, 30.5)],
    }
    pred = predict_bins(event, obs, forecast, {}, current_hour=17, city="Houston")

    # Find the 90-91°F bin in the result
    bin_90 = next(b for b in pred["bins"] if b["range_low"] == 90)
    bin_92 = next(b for b in pred["bins"] if b["range_low"] == 92)

    print(f"\n  Houston test result:")
    print(f"    cooling_confidence: {pred['cooling_confidence']:.3f}")
    print(f"    bin_locked: {pred['bin_locked']}")
    print(f"    mu: {pred['mu']}, sigma: {pred['sigma']}")
    print(f"    90-91°F our_p: {bin_90['our_prob']:.3f}  (was wrong as 0.04)")
    print(f"    92-93°F our_p: {bin_92['our_prob']:.3f}  (was wrong as 0.74)")

    assert pred["bin_locked"], (
        f"Houston scenario should bin-lock (cooling_conf={pred['cooling_confidence']})"
    )
    assert bin_90["our_prob"] > bin_92["our_prob"], (
        f"90-91°F ({bin_90['our_prob']:.3f}) should be top-P, not "
        f"92-93°F ({bin_92['our_prob']:.3f})"
    )
    assert bin_90["our_prob"] > 0.7, (
        f"With bin-lock, 90-91°F should be very confident "
        f"(got {bin_90['our_prob']:.3f})"
    )


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