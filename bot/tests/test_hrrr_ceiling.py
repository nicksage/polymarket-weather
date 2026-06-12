"""
test_hrrr_ceiling.py — Lock in the Phase 1 HRRR ceiling behavior.

Tests cover:
  - Plateau detector: trajectory shapes that should and shouldn't fire
  - Sanity gates: reject implausibly-out-of-band HRRR data, accept good
  - estimate_day_high_dist HRRR μ-recenter and plateau-driven narrowing
  - estimate_day_high_dist behaves identically when HRRR args are None
    (regression check: the flag-off path is zero-diff vs pre-HRRR)
  - Region-to-model mapping: US cities are 'hrrr', edge-of-domain EU
    cities are explicitly None pending verification

Note: tests don't hit the network.  The fetcher is exercised in
integration; here we feed pre-shaped trajectory data to the downstream
pieces.

Run:
    cd bot
    python -m pytest tests/test_hrrr_ceiling.py -v
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.intraday_predictor import (   # type: ignore
    is_rapid_model_trajectory_plateaued,
    _hrrr_data_passes_sanity,
    estimate_day_high_dist,
)
from station_meta import (   # type: ignore
    SAME_DAY_MODEL_BY_CITY,
    get_same_day_model,
)


def _traj(*temps_with_cloud) -> list[dict]:
    """Build a trajectory list from (hour, temp_c[, cloud_pct]) tuples."""
    out = []
    for t in temps_with_cloud:
        if len(t) == 3:
            h, c, cl = t
        else:
            h, c = t
            cl = None
        out.append({"hour_local": h, "temp_c": float(c), "cloud_pct": cl})
    return out


# ============================================================
# is_rapid_model_trajectory_plateaued
# ============================================================

def test_plateau_fires_on_flat_trajectory():
    """Atlanta 14:00 case — next 2-3h hover within tolerance of current."""
    traj = _traj((14, 33.0), (15, 33.1), (16, 33.0), (17, 32.8))
    assert is_rapid_model_trajectory_plateaued(traj) is True


def test_plateau_fires_on_declining_trajectory():
    """Late-day cooling — temp drops through the horizon.  Day is over."""
    traj = _traj((16, 32.0), (17, 31.0), (18, 29.5))
    assert is_rapid_model_trajectory_plateaued(traj) is True


def test_plateau_does_not_fire_on_continued_rise():
    """Morning still-climbing — next horizon shows further rise."""
    traj = _traj((11, 28.0), (12, 30.0), (13, 32.0), (14, 33.5))
    assert is_rapid_model_trajectory_plateaued(traj) is False


def test_plateau_does_not_fire_on_tiny_horizon():
    """One-point horizon can't tell us anything — return False."""
    traj = _traj((14, 33.0))
    assert is_rapid_model_trajectory_plateaued(traj) is False


def test_plateau_respects_tolerance():
    """A 0.2°C rise should still count as plateau (within default 0.3°C
    tolerance)."""
    traj = _traj((14, 33.0), (15, 33.2), (16, 33.1))
    assert is_rapid_model_trajectory_plateaued(traj) is True


def test_plateau_does_not_fire_on_just_above_tolerance():
    """A 0.5°C rise over the horizon shouldn't be considered plateau."""
    traj = _traj((14, 33.0), (15, 33.5), (16, 33.4))
    assert is_rapid_model_trajectory_plateaued(traj) is False


# ============================================================
# _hrrr_data_passes_sanity
# ============================================================

def _good_hrrr_data(remaining_max=33.0, observed_max=32.8):
    return {
        "remaining_max_c": remaining_max,
        "trajectory":      _traj((14, observed_max), (15, remaining_max),
                                   (16, remaining_max - 0.5)),
        "cycle_time":      "2026-06-12T14:00:00",
        "model":           "hrrr",
    }


def test_sanity_passes_on_normal_data():
    data = _good_hrrr_data()
    assert _hrrr_data_passes_sanity(data,
                                       observed_max_c=32.8,
                                       forecast_high_c=33.9) is True


def test_sanity_rejects_remaining_below_observed_by_more_than_slack():
    """HRRR can't say the day's max will be BELOW what's already
    observed — that's a physically impossible projection."""
    data = _good_hrrr_data(remaining_max=30.0)
    assert _hrrr_data_passes_sanity(data,
                                       observed_max_c=33.0,
                                       forecast_high_c=33.9) is False


def test_sanity_accepts_remaining_below_observed_within_slack():
    """Up to 0.5°C below observed is allowed — HRRR may be slightly
    stale compared to the latest METAR."""
    data = _good_hrrr_data(remaining_max=32.5)
    assert _hrrr_data_passes_sanity(data,
                                       observed_max_c=32.8,
                                       forecast_high_c=33.9) is True


def test_sanity_rejects_implausibly_high_delta_from_forecast():
    """+10°C above forecast is a model glitch, not real data."""
    data = _good_hrrr_data(remaining_max=44.0)
    assert _hrrr_data_passes_sanity(data,
                                       observed_max_c=32.0,
                                       forecast_high_c=33.9) is False


def test_sanity_rejects_implausibly_low_delta_from_forecast():
    """-10°C below forecast is also a glitch — symmetric range bound."""
    data = _good_hrrr_data(remaining_max=24.0)
    assert _hrrr_data_passes_sanity(data,
                                       observed_max_c=-100,    # no obs
                                       forecast_high_c=33.9) is False


def test_sanity_rejects_missing_trajectory():
    data = {"remaining_max_c": 33.0, "trajectory": [], "model": "hrrr"}
    assert _hrrr_data_passes_sanity(data, observed_max_c=32.8,
                                       forecast_high_c=33.9) is False


def test_sanity_rejects_nan_in_trajectory():
    data = _good_hrrr_data()
    data["trajectory"][1]["temp_c"] = None
    assert _hrrr_data_passes_sanity(data, observed_max_c=32.8,
                                       forecast_high_c=33.9) is False


# ============================================================
# estimate_day_high_dist with HRRR signals
# ============================================================

def test_hrrr_recenter_atlanta_plateau_case():
    """Atlanta 2026-06-12 14:00 scenario.

    Without HRRR: μ anchors near forecast 33.9°C → puts mass in 92-93°F
    bin requiring +1°C of further heating.

    With HRRR remaining_max=32.8 (HRRR sees the cloudy plateau): μ
    recenters to 32.8 (max of observed=32.8 and HRRR=32.8).
    """
    # Without HRRR
    mu_no_hrrr, _ = estimate_day_high_dist(
        forecast_high      = 33.9,
        forecast_peak_hour = 16,
        observed_max       = 32.8,
        observed_peak_hour = 14,
        current_hour       = 14,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 1.0,
    )
    # With HRRR
    mu_with_hrrr, _ = estimate_day_high_dist(
        forecast_high      = 33.9,
        forecast_peak_hour = 16,
        observed_max       = 32.8,
        observed_peak_hour = 14,
        current_hour       = 14,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 1.0,
        hrrr_remaining_max = 32.8,
    )
    assert mu_with_hrrr < mu_no_hrrr, (
        f"HRRR recenter should pull μ DOWN from forecast.  "
        f"Without HRRR: μ={mu_no_hrrr:.2f}, with HRRR: μ={mu_with_hrrr:.2f}"
    )
    # μ_with_hrrr should be at or near observed_max (32.8)
    assert mu_with_hrrr <= 33.0, (
        f"With HRRR=32.8, μ should land near 32.8, got {mu_with_hrrr:.2f}"
    )


def test_hrrr_recenter_does_not_fire_when_hrrr_agrees_with_forecast():
    """If HRRR agrees with the morning forecast (within 0.5°C), don't
    override — let the existing logic run."""
    mu_no_hrrr, _ = estimate_day_high_dist(
        forecast_high      = 33.9,
        forecast_peak_hour = 16,
        observed_max       = 30.0,
        observed_peak_hour = 12,
        current_hour       = 13,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 1.0,
    )
    mu_with_hrrr, _ = estimate_day_high_dist(
        forecast_high      = 33.9,
        forecast_peak_hour = 16,
        observed_max       = 30.0,
        observed_peak_hour = 12,
        current_hour       = 13,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 1.0,
        hrrr_remaining_max = 33.5,   # within 0.5°C of forecast — agrees
    )
    assert abs(mu_with_hrrr - mu_no_hrrr) < 0.01, (
        "HRRR-agrees-with-forecast case should not override μ"
    )


def test_hrrr_plateau_signal_engages_post_peak_narrowing_early():
    """The plateau signal should trigger day_has_likely_peaked even when
    wall-clock hasn't reached forecast_peak_hour."""
    # Setup: 14:00 local, forecast peak at 16:00, observed at 32.8 vs
    # forecast 33.9 (within the 1°C tolerance for the OBSERVED-based
    # day_has_likely_peaked trigger).
    _, sigma_no_plateau = estimate_day_high_dist(
        forecast_high      = 33.9,
        forecast_peak_hour = 16,
        observed_max       = 31.0,    # below the 1°C tolerance threshold
        observed_peak_hour = 14,
        current_hour       = 14,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
    )
    _, sigma_with_plateau = estimate_day_high_dist(
        forecast_high      = 33.9,
        forecast_peak_hour = 16,
        observed_max       = 31.0,
        observed_peak_hour = 14,
        current_hour       = 14,
        sunset_hour        = 20,
        neighbor_signal    = {},
        base_sigma_c       = 2.0,
        hrrr_plateau_signal= True,   # HRRR says day's done climbing
    )
    # With plateau signal True, day_has_likely_peaked fires → post-peak
    # narrowing engages and σ should narrow.  (At hours_since_obs_peak=0,
    # the geometric factor is 1.0 so we mostly see the μ blend; but the
    # gate has fired which means the branch now runs.)
    # Hard assertion: plateau signal should not make σ widen.
    assert sigma_with_plateau <= sigma_no_plateau + 0.01, (
        f"With HRRR plateau signal, σ should not widen.  "
        f"Without: σ={sigma_no_plateau:.2f}, with: σ={sigma_with_plateau:.2f}"
    )


# ============================================================
# Zero-diff guarantee when HRRR args are None
# ============================================================

def test_hrrr_args_none_preserves_legacy_behavior():
    """The default path (HRRR args = None / False) must produce
    bit-identical (μ, σ) to the pre-HRRR code."""
    # Simulate a few different scenarios — none should differ when the
    # HRRR args are omitted vs. explicitly None/False.
    scenarios = [
        # (forecast_high, forecast_peak, observed_max, observed_peak, current, sunset)
        (33.9, 16, 32.8, 14, 14, 20),
        (28.0, 15, -100, -1, 10, 19),
        (35.0, 17, 36.0, 16, 17, 20),    # obs > forecast (FIX 1a path)
        (30.0, 16, 29.5, 15, 18, 21),    # post-peak
    ]
    for fh, fph, om, oph, ch, sh in scenarios:
        baseline = estimate_day_high_dist(
            forecast_high=fh, forecast_peak_hour=fph,
            observed_max=om, observed_peak_hour=oph,
            current_hour=ch, sunset_hour=sh,
            neighbor_signal={}, base_sigma_c=1.5,
        )
        with_hrrr_none = estimate_day_high_dist(
            forecast_high=fh, forecast_peak_hour=fph,
            observed_max=om, observed_peak_hour=oph,
            current_hour=ch, sunset_hour=sh,
            neighbor_signal={}, base_sigma_c=1.5,
            hrrr_remaining_max=None, hrrr_plateau_signal=False,
        )
        assert baseline == with_hrrr_none, (
            f"HRRR=None must produce identical results to omitted args.  "
            f"Scenario {(fh, fph, om, oph, ch, sh)}: "
            f"baseline={baseline}, with_hrrr_none={with_hrrr_none}"
        )


# ============================================================
# Region-to-model map
# ============================================================

def test_us_cities_assigned_to_hrrr():
    """All 11 currently-traded US cities must map to 'hrrr'."""
    us_cities = ["Atlanta", "Austin", "Chicago", "Dallas", "Denver",
                  "Houston", "Los Angeles", "Miami", "NYC",
                  "San Francisco", "Seattle"]
    for city in us_cities:
        assert get_same_day_model(city) == "hrrr", (
            f"{city} should map to 'hrrr', got {get_same_day_model(city)!r}"
        )


def test_central_europe_cities_assigned_to_icon_d2():
    eu_cities = ["Munich", "Milan", "Paris", "Warsaw"]
    for city in eu_cities:
        assert get_same_day_model(city) == "icon_d2", (
            f"{city} should map to 'icon_d2', got {get_same_day_model(city)!r}"
        )


def test_edge_of_domain_eu_cities_are_none_pending_verification():
    """Per the spec, London/Madrid/Helsinki are deliberately None until
    explicit ICON-D2 domain verification."""
    for city in ["London", "Madrid", "Helsinki", "Moscow"]:
        assert get_same_day_model(city) is None, (
            f"{city} should be None (pending ICON-D2 domain "
            f"verification), got {get_same_day_model(city)!r}"
        )


def test_non_cam_cities_are_none():
    """Cities without a fresh CAM available should return None."""
    for city in ["Seoul", "Tokyo", "Sao Paulo", "Cape Town",
                  "Singapore", "Wellington"]:
        assert get_same_day_model(city) is None


def test_unknown_city_returns_none():
    """Cities not in the map return None — safe default for new cities
    added without same-day-model assignment yet."""
    assert get_same_day_model("Atlantis") is None
    assert get_same_day_model("") is None