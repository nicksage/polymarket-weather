"""
test_sigma_floor_and_cache_fix.py — Pin the 2026-06-15 calibration fixes.

  Fix 1: PREDICTOR_SIGMA_FLOOR_C = 1.3°C
    Q2 realism check showed σ under-calibrated: avg 0.96 vs avg |error|
    1.25.  The 0.3°C legacy floor essentially never bound, letting σ
    collapse to ~0.30°C after observations land — which makes off-by-
    one-bin errors a 100% loss (the bin we bought gets 0% mass when
    actual lands in an adjacent bin).  Floor at 1.3°C distributes mass
    over 2-3 adjacent bins.

    Carve-out: post-sunset retains the legacy 0.3°C floor because the
    day's high is locked at observed_max — we KNOW the answer, no
    point widening σ artificially.

  Fix 2: Weather cache freshness check includes today_str_local
    Without this, the in-memory _WEATHER_CACHE serves yesterday's
    nws_obs for up to WEATHER_CACHE_SEC (default 300s) AFTER the local
    day rolls over.  MAX(observed_max_c) picks up the leaked yesterday
    peak under today's event_date, contaminating any analysis built
    on that column.  Confirmed in production: Denver 06-13 had 2 scans
    at 00:00 / 00:02 MDT writing 33.2°C while every other scan
    correctly showed 26.3°C.

Run:
    cd bot
    python -m pytest tests/test_sigma_floor_and_cache_fix.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)


# ============================================================
# Fix 1 — σ floor at 1.3°C
# ============================================================

def test_sigma_floor_default_is_1_3():
    """Default floor must be 1.3°C (the value Q2's avg|error|/avg σ
    ratio of 1.30 argued for).  If a future commit drops this back to
    0.3°C, σ will silently collapse again and off-by-one-bin losses
    return."""
    from scripts.intraday_predictor import PREDICTOR_SIGMA_FLOOR_C
    if os.environ.get("PREDICTOR_SIGMA_FLOOR_C"):
        pytest.skip("env override set; skipping default check")
    assert PREDICTOR_SIGMA_FLOOR_C == 1.3, (
        f"PREDICTOR_SIGMA_FLOOR_C should default to 1.3.  Got "
        f"{PREDICTOR_SIGMA_FLOOR_C}.  Review docstring before changing."
    )


def test_sigma_floor_binds_when_narrowing_branches_collapse(monkeypatch):
    """estimate_day_high_dist's σ output is never lower than the floor
    PRE-sunset, even when every narrowing branch fires."""
    from scripts import intraday_predictor as ip

    # Force the floor so this test is robust to future env tuning
    monkeypatch.setattr(ip, "PREDICTOR_SIGMA_FLOOR_C", 1.3)

    # Set up a scenario that maximally narrows σ: past forecast peak,
    # observed peak hours ago, strong cooling signal, high cooling
    # confidence.  Everything stacks downward.
    mu, sigma = ip.estimate_day_high_dist(
        forecast_high      = 33.0,
        forecast_peak_hour = 14,
        observed_max       = 33.0,
        observed_peak_hour = 14,
        current_hour       = 18,    # past peak, but pre-sunset
        sunset_hour        = 20,
        neighbor_signal    = {"strong_cooling_signal": True,
                               "signal_strength": 1.0},
        base_sigma_c       = 0.5,   # tight base to compound the narrowing
        cooling_confidence = 0.9,
    )
    assert sigma >= 1.3, (
        f"σ should be floored at 1.3°C pre-sunset; got {sigma:.3f}.  "
        f"Narrowing branches must not collapse it below the realism floor."
    )


def test_sigma_floor_does_not_apply_post_sunset(monkeypatch):
    """Post-sunset we KNOW the day's high (locked at observed_max).
    σ should retain the 0.3°C legacy floor (pure measurement noise) —
    artificially widening to 1.3°C would mass-distribute over bins
    we know didn't win."""
    from scripts import intraday_predictor as ip
    monkeypatch.setattr(ip, "PREDICTOR_SIGMA_FLOOR_C", 1.3)

    mu, sigma = ip.estimate_day_high_dist(
        forecast_high      = 33.0,
        forecast_peak_hour = 14,
        observed_max       = 33.0,
        observed_peak_hour = 14,
        current_hour       = 21,    # POST-sunset
        sunset_hour        = 20,
        neighbor_signal    = {"strong_cooling_signal": False,
                               "signal_strength": 0.0},
        base_sigma_c       = 1.5,
        cooling_confidence = 0.0,
    )
    # Sunset block sets σ = max(0.3, σ × 0.2) — 1.5 × 0.2 = 0.30.
    # Post-sunset carve-out keeps the legacy 0.3 floor, so we expect
    # σ to land below the new 1.3 floor without it overriding.
    assert sigma < 1.3, (
        f"Post-sunset σ ({sigma:.3f}) should NOT be floored at 1.3 — "
        f"the day is locked and σ should reflect that certainty."
    )
    assert sigma >= 0.3, (
        f"Post-sunset σ should still respect the legacy 0.3 measurement"
        f"-noise floor.  Got {sigma:.3f}."
    )


def test_sigma_floor_env_override_works(monkeypatch):
    """The constant is env-tunable.  Setting PREDICTOR_SIGMA_FLOOR_C
    binds at module reload."""
    import importlib
    monkeypatch.setenv("PREDICTOR_SIGMA_FLOOR_C", "1.8")
    from scripts import intraday_predictor as ip
    importlib.reload(ip)
    assert ip.PREDICTOR_SIGMA_FLOOR_C == 1.8


# ============================================================
# Fix 2 — Weather cache day-boundary invalidation
# ============================================================

def test_cache_hit_path_checks_today_str_local():
    """The freshness check must include today_str_local match.  Source
    string regression check — if a future refactor drops this, the
    day-boundary contamination bug returns silently."""
    import inspect
    from scheduled_predictor import run_intraday_scan
    src = inspect.getsource(run_intraday_scan)
    # The exact comparison we want to see in the cache-hit gate
    assert "today_str_local" in src, (
        "run_intraday_scan must reference today_str_local in the cache "
        "check.  Without it, the cache serves yesterday's nws_obs at "
        "local-midnight."
    )
    assert "cache_day_matches" in src or "cached_day == today_str_local" in src, (
        "Cache hit must guard on local-day match.  Re-introducing the "
        "day-key-less check brings back the Denver 06-13 phantom 91.8°F "
        "case."
    )


def test_cache_write_includes_today_str_local():
    """The cached dict written at fetch time must include today_str_local
    so subsequent hits can detect rollover."""
    import inspect
    from scheduled_predictor import run_intraday_scan
    src = inspect.getsource(run_intraday_scan)
    # The cache-write block stores today_str_local alongside the data
    assert '"today_str_local": today_str_local' in src, (
        'Cache write must stamp today_str_local: today_str_local for the '
        "freshness check to work.  Without it, the check always says "
        "(None == 'YYYY-MM-DD'), which fails-open — every cache entry "
        "looks like wrong-day and forces a refetch, defeating the cache."
    )