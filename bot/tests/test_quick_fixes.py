"""
test_quick_fixes.py — Lock in Quick Fixes A, B, C (2026-06-12).

These three fixes ship LIVE with default ON (unlike HRRR which ships
dormant). The tests cover:

  A. PREDICTOR_USE_CLIMATOLOGICAL_SIGMA defaults to ON. σ prior of
     1.75°C is the documented climatological band, NOT a tuned floor.

  B. PREDICTOR_IMMEDIATE_POST_PEAK_NARROW defaults to ON. Closes the
     2-hour lag where the day plateaued but the bot hadn't recognized
     it. When disabled, falls back to legacy hours_since >= 1 trigger.

  C. PREDICTOR_USE_PLAUSIBILITY_CEILING defaults to ON. Computes a
     crude physical-plausibility upper truncation from NWS skyCover +
     required-heating-rate. Fires only on clearly-implausible afternoon
     plateaus; cold-start skipped; combined with HRRR via min().

Run:
    cd bot
    python -m pytest tests/test_quick_fixes.py -v
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.intraday_predictor import (   # type: ignore
    PREDICTOR_USE_CLIMATOLOGICAL_SIGMA,
    PREDICTOR_CLIMATOLOGICAL_SIGMA_C,
    PREDICTOR_IMMEDIATE_POST_PEAK_NARROW,
    PREDICTOR_USE_PLAUSIBILITY_CEILING,
    PREDICTOR_CEILING_BUFFER_C,
    PREDICTOR_IMPLAUSIBLE_RATE_C_PER_H,
    PREDICTOR_AFTERNOON_HOUR_THRESHOLD,
    PREDICTOR_CLOUD_THRESHOLD_PCT,
    plausible_remaining_rise_c,
    avg_remaining_cloud_pct,
    compute_plausibility_ceiling_c,
)


# ============================================================
# Fix A — Default ON
# ============================================================

def test_fix_a_default_on():
    """PREDICTOR_USE_CLIMATOLOGICAL_SIGMA must default to ON per the
    Quick-Fix-A spec. If a future commit flips this default, the spec
    is being violated — fail loudly."""
    assert PREDICTOR_USE_CLIMATOLOGICAL_SIGMA is True, (
        "Quick Fix A says σ prior ships default ON.  If this assertion "
        "fails, either the env var changed (`PREDICTOR_USE_CLIMATOLOGICAL_SIGMA=0`) "
        "or the source default was edited.  Revisit the Quick-Fix-A spec "
        "before flipping this."
    )


def test_fix_a_prior_value_in_documented_band():
    """The prior must be in the published 1.5–2.0°C NWS day-ahead summer
    MaxT MAE band. If this is outside, someone hand-tuned without
    updating the citation."""
    assert 1.5 <= PREDICTOR_CLIMATOLOGICAL_SIGMA_C <= 2.0, (
        f"σ prior ({PREDICTOR_CLIMATOLOGICAL_SIGMA_C}°C) is outside "
        f"the documented day-ahead NWS MaxT MAE band of 1.5–2.0°C.  "
        f"If this change is intentional, update the citation in the "
        f"constant's comment block."
    )


# ============================================================
# Fix B — Default ON
# ============================================================

def test_fix_b_default_on():
    """PREDICTOR_IMMEDIATE_POST_PEAK_NARROW must default to ON."""
    assert PREDICTOR_IMMEDIATE_POST_PEAK_NARROW is True, (
        "Quick Fix B says immediate post-peak narrowing ships default ON."
    )


# ============================================================
# Fix C — Default ON
# ============================================================

def test_fix_c_default_on():
    """PREDICTOR_USE_PLAUSIBILITY_CEILING must default to ON."""
    assert PREDICTOR_USE_PLAUSIBILITY_CEILING is True, (
        "Quick Fix C says plausibility ceiling ships default ON."
    )


def test_fix_c_buffer_at_loose_end():
    """Buffer should start LOOSE (1.0°C) per the spec — asymmetric loss
    reasoning. Tighten only when logs prove it's costing edge without
    causing misses."""
    assert PREDICTOR_CEILING_BUFFER_C >= 1.0, (
        f"Fix C buffer should start at 1.0°C (loose) per the spec.  "
        f"Got {PREDICTOR_CEILING_BUFFER_C}°C."
    )


# ============================================================
# plausible_remaining_rise_c
# ============================================================

def test_remaining_rise_zero_when_no_hours_left():
    """At sunset, no further rise possible."""
    assert plausible_remaining_rise_c(0.0, 50.0) == 0.0
    assert plausible_remaining_rise_c(-1.0, 50.0) == 0.0


def test_remaining_rise_clear_sky_yields_max_rate():
    """0% clouds → max heating rate (~2°C/hr) × remaining hours."""
    # 3 clear hours = up to 6°C rise, but capped at 5°C
    rise = plausible_remaining_rise_c(3.0, 0.0)
    assert rise == 5.0, f"3 clear hours should cap at 5°C, got {rise}"


def test_remaining_rise_cloudy_reduces_rate():
    """100% clouds → minimum rate (0.5°C/hr by formula, floor 0.3)."""
    # 4 cloudy hours × 0.5°C/hr = 2°C
    rise = plausible_remaining_rise_c(4.0, 100.0)
    assert 1.5 < rise < 3.0, (
        f"4 cloudy hours should yield ~2°C, got {rise:.2f}"
    )


def test_remaining_rise_scales_with_cloud_cover():
    """More clouds → less rise."""
    clear = plausible_remaining_rise_c(2.0, 0.0)
    partial = plausible_remaining_rise_c(2.0, 50.0)
    cloudy = plausible_remaining_rise_c(2.0, 100.0)
    assert clear > partial > cloudy


def test_remaining_rise_total_caps_at_5c():
    """Even many clear hours shouldn't permit >5°C rise — physical limit."""
    rise = plausible_remaining_rise_c(10.0, 0.0)
    assert rise == 5.0


# ============================================================
# avg_remaining_cloud_pct
# ============================================================

def test_avg_cloud_handles_empty():
    assert avg_remaining_cloud_pct({}, 14, 20) is None


def test_avg_cloud_handles_past_sunset():
    assert avg_remaining_cloud_pct({14: 50}, 21, 20) is None


def test_avg_cloud_only_future_hours():
    """We average ONLY remaining-to-sunset hours."""
    clouds = {12: 20.0, 13: 40.0, 14: 60.0, 15: 80.0, 16: 100.0}
    # At 14:00, sunset at 20: avg over hours 14, 15, 16 = (60+80+100)/3 ≈ 80
    avg = avg_remaining_cloud_pct(clouds, 14, 20)
    assert avg is not None
    assert abs(avg - 80.0) < 0.01


# ============================================================
# compute_plausibility_ceiling_c — triggers
# ============================================================

def _atlanta_scenario(**overrides):
    """Atlanta 2026-06-12 14:00 case as a reusable kwargs dict.
    Defaults are tuned so the ceiling SHOULD fire."""
    base = dict(
        current_temp_c     = 33.0,    # plateau temp
        forecast_high_c    = 33.9,    # forecast 0.9°C above current
        forecast_peak_hour = 16,
        current_hour       = 14,      # afternoon
        sunset_hour        = 20,
        cloud_pct          = 80.0,    # cloudy
    )
    base.update(overrides)
    return base


def test_ceiling_fires_when_all_triggers_hit():
    """Atlanta-style case: hot forecast vs. observed plateau under
    afternoon clouds. With required rate forced above threshold via
    a synthetic spread, the ceiling should fire."""
    # Force required_rate above 2.0: forecast 36, current 30, 2 hrs to peak
    # → required_rate = (36-30)/2 = 3.0 > 2.0 ✓
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=30.0, forecast_high_c=36.0, forecast_peak_hour=16,
        current_hour=14))
    assert result is not None, "Ceiling should fire on Atlanta-class case"
    assert "required_rate" in result["fired_reason"]
    assert "cloud" in result["fired_reason"]
    assert result["ceiling_c"] > 30.0   # ceiling is above current temp
    assert result["ceiling_c"] < 36.0   # but below the implausible forecast


def test_ceiling_does_not_fire_when_morning():
    """Before AFTERNOON_HOUR_THRESHOLD, no fire — there's still plenty
    of day left to heat."""
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=24.0, current_hour=10))
    assert result is None


def test_ceiling_does_not_fire_when_clear_sky():
    """Even with high required rate, clear sky → don't constrain."""
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=30.0, forecast_high_c=36.0, cloud_pct=10.0))
    assert result is None


def test_ceiling_does_not_fire_when_rate_is_plausible():
    """If required_rate <= threshold, the forecast is achievable — no fire."""
    # 1°C over 2 hours = 0.5°C/hr, well below 2.0 threshold
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=33.0, forecast_high_c=34.0, forecast_peak_hour=16,
        current_hour=14))
    assert result is None


def test_ceiling_does_not_fire_when_forecast_peak_passed():
    """If we're past forecast peak hour, the day has already done its
    climbing — heuristic doesn't apply."""
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        forecast_peak_hour=14, current_hour=15))
    assert result is None


def test_ceiling_does_not_fire_when_cloud_data_missing():
    """No cloud data → can't fire (we have no signal to anchor on)."""
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=30.0, forecast_high_c=36.0, cloud_pct=None))
    assert result is None


def test_ceiling_does_not_fire_when_obs_invalid():
    """Sentinel-low observed_max (no obs yet) → don't fire."""
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=-100, forecast_high_c=36.0))
    assert result is None


def test_ceiling_value_uses_buffer():
    """The returned ceiling must include CEILING_BUFFER_C — leaving
    the conservative pad for HRRR's known short-range error."""
    result = compute_plausibility_ceiling_c(**_atlanta_scenario(
        current_temp_c=30.0, forecast_high_c=36.0,
        ceiling_buffer_c=2.0))
    assert result is not None
    # ceiling = current + plausible_rise + 2.0 buffer
    # ceiling - current - 2.0 = plausible_rise → positive
    extra = result["ceiling_c"] - 30.0 - 2.0
    assert extra > 0, (
        f"Ceiling should include the +2.0 buffer.  "
        f"current=30, buffer=2.0, ceiling={result['ceiling_c']}"
    )


# ============================================================
# Spec invariants — sanity defaults
# ============================================================

def test_spec_default_thresholds_match_spec():
    """The constants exposed via env vars must match the spec defaults.
    If the source defaults drifted, the spec doc and the code disagree."""
    assert PREDICTOR_IMPLAUSIBLE_RATE_C_PER_H == 2.0
    assert PREDICTOR_AFTERNOON_HOUR_THRESHOLD == 13
    assert PREDICTOR_CLOUD_THRESHOLD_PCT == 50.0
    assert PREDICTOR_CEILING_BUFFER_C == 1.0