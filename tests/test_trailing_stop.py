"""
test_trailing_stop.py — Trailing-stop unit tests (single source of truth).

Covers the 6 cases from the engineering spec plus 3 we added during the
codebase review:
  7. Restart-safety in the realtime path (peak cache seeds from DB)
  8. Single-tier table behaves like legacy single-pct trail did
     (back-compat sanity for `TRAIL_TIERS=0.00:1.00:0.10`)
  9. Tier boundary at exactly 0.30 falls into the upper tier (half-open)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest

# Make bot/ importable
_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from trailing_stop import (
    evaluate_trailing_stop, lookup_trail_pct, update_peak,
)


# Default tier table from the spec
TIERS = [
    (0.00, 0.30, 0.15),
    (0.30, 0.50, 0.10),
    (0.50, 0.70, 0.07),
    (0.70, 0.90, 0.05),
    (0.90, 1.00, 0.03),
]
ACTIVATION = 0.03


@dataclass
class Position:
    """Minimal position-like object for tests.  Real position rows have
    many more fields but the trailing logic only needs these three."""
    entry: float
    peak: float


def _eval(pos: Position, current: float):
    """Caller-side wrapper that does the unconditional peak update before
    calling the pure evaluator — matching what the real callers do."""
    pos.peak = update_peak(pos.peak, current, pos.entry)
    return evaluate_trailing_stop(
        entry_price=pos.entry,
        peak_price=pos.peak,
        current_price=current,
        activation_gain=ACTIVATION,
        tiers=TIERS,
    )


# ===========================================================================
# Spec test 1 — Peak update happens even when disarmed
# ===========================================================================

def test_peak_update_when_disarmed():
    pos = Position(entry=0.30, peak=0.30)
    result = _eval(pos, current=0.31)
    assert pos.peak == pytest.approx(0.31)   # peak updated
    # 0.31 - 0.30 = 0.01 < ACTIVATION (0.03) → no exit yet
    assert result is None


# ===========================================================================
# Spec test 2 — Activation gate prevents same-bar exit
# ===========================================================================

def test_activation_gate_prevents_same_bar_exit():
    pos = Position(entry=0.30, peak=0.30)
    # Price drops below entry — should NOT fire because peak hasn't risen
    result = _eval(pos, current=0.295)
    assert result is None


# ===========================================================================
# Spec test 3 — Tier 2 (0.50–0.70 → 7%) fires correctly
# ===========================================================================

def test_tier_2_fires_correctly():
    pos = Position(entry=0.30, peak=0.60)
    # stop = 0.60 * (1 - 0.07) = 0.558
    result = _eval(pos, current=0.555)
    assert result is not None
    stop_level, reason = result
    assert stop_level == pytest.approx(0.558)
    assert reason == "TRAILING_STOP"

    # Just above the stop — should NOT fire
    pos2 = Position(entry=0.30, peak=0.60)
    result2 = _eval(pos2, current=0.560)
    assert result2 is None


# ===========================================================================
# Spec test 4 — Top-tier inclusivity at exactly 1.0
# ===========================================================================

def test_top_tier_inclusive_at_one():
    pos = Position(entry=0.30, peak=1.00)
    # stop = 1.00 * (1 - 0.03) = 0.97
    result = _eval(pos, current=0.96)
    assert result is not None
    stop_level, reason = result
    assert stop_level == pytest.approx(0.97)
    assert reason == "TRAILING_STOP"


# ===========================================================================
# Spec test 5 — Tier transition: peak crosses tier boundary on a tick
# ===========================================================================

def test_tier_transition_in_one_tick():
    pos = Position(entry=0.30, peak=0.45)   # tier 1: 10%

    # First tick brings peak up to 0.55 → moves into tier 2 (7%)
    # New stop = 0.55 * (1 - 0.07) = 0.5115
    # Current price IS 0.55, which is above stop → no exit yet
    result = _eval(pos, current=0.55)
    assert result is None
    assert pos.peak == pytest.approx(0.55)

    # Next tick: price drops to 0.50.  Peak unchanged; stop unchanged.
    # 0.50 <= 0.5115 → fires
    result2 = _eval(pos, current=0.50)
    assert result2 is not None
    stop_level, _ = result2
    assert stop_level == pytest.approx(0.5115)


# ===========================================================================
# Spec test 6 — Restart safety: position loaded with peak=0.65, fires correctly
# ===========================================================================

def test_restart_safety_pure_evaluator():
    # Simulates: bot restarted, position re-loaded from DB with peak=0.65
    # (tier 2 → 7%).  stop = 0.65 * 0.93 = 0.6045
    pos = Position(entry=0.30, peak=0.65)
    result = _eval(pos, current=0.60)
    assert result is not None
    stop_level, _ = result
    assert stop_level == pytest.approx(0.6045)


# ===========================================================================
# Our test 7 — Restart safety in the realtime path (DB seed)
# ===========================================================================

def test_realtime_peak_seeds_from_db_on_first_sighting():
    """The realtime path's _update_peak must seed from DB peak_price on the
    first tick after a process restart.  Otherwise a position with peak=0.85
    in the DB but a current tick at 0.70 would lose the historical high."""
    from realtime_exits import _update_peak, clear_exited_cache

    clear_exited_cache()   # ensure empty in-memory cache (simulates restart)

    pid = 999
    entry = 0.30
    db_peak = 0.85

    # First tick: price below the DB peak — peak should NOT drop to price,
    # it should seed from db_peak.
    peak_after = _update_peak(pid, price=0.70, entry_price=entry, db_peak=db_peak)
    assert peak_after == pytest.approx(0.85)

    # Subsequent tick at 0.90 — peak rises further
    peak_after = _update_peak(pid, price=0.90, entry_price=entry, db_peak=db_peak)
    assert peak_after == pytest.approx(0.90)

    # Tick at 0.80 — peak stays at 0.90
    peak_after = _update_peak(pid, price=0.80, entry_price=entry, db_peak=db_peak)
    assert peak_after == pytest.approx(0.90)


def test_realtime_peak_seeds_from_entry_when_db_peak_null():
    """Defensive: if DB peak is NULL (legacy row before column existed),
    seed from entry_price rather than crashing."""
    from realtime_exits import _update_peak, clear_exited_cache
    clear_exited_cache()

    pid = 1000
    peak_after = _update_peak(pid, price=0.45, entry_price=0.40, db_peak=None)
    # update_peak() floors at entry_price; 0.45 > 0.40 so peak = 0.45
    assert peak_after == pytest.approx(0.45)


# ===========================================================================
# Our test 8 — Single-tier table reproduces legacy single-pct behavior
# ===========================================================================

def test_single_tier_table_arms_and_fires_correctly(monkeypatch):
    """A single-row TRAIL_TIERS table reproduces a flat single-percentage
    trail.  Pins the equivalence so a refactor of the tiered evaluator
    can't silently break the single-tier migration story.

    NB: TRAIL_ACTIVATION_GAIN is ADDITIVE (peak >= entry + gain), not
    multiplicative.  Operators migrating from MPV_TRAIL_ACTIVATION (which
    was multiplicative: peak >= entry * (1+pct)) need to translate.  For
    typical entry prices ~0.20-0.30, MPV_TRAIL_ACTIVATION=0.40 is
    roughly equivalent to TRAIL_ACTIVATION_GAIN=0.10."""
    import config as cfg
    monkeypatch.setattr(cfg, "TRAIL_TIERS", [(0.0, 1.0, 0.10)])
    monkeypatch.setattr(cfg, "TRAIL_ACTIVATION_GAIN", 0.10)
    # Force re-import so realtime_exits picks up the patched values
    import importlib, realtime_exits
    importlib.reload(realtime_exits)
    from realtime_exits import _evaluate_trail

    entry      = 0.30
    activation = 0.10   # additive — peak must reach entry + 0.10 = 0.40
    trail_pct  = 0.10   # 10% trail across the whole [0, 1] range

    # Peak high enough to arm the trail
    peak = entry + activation + 0.05         # 0.45
    expected_stop = peak * (1 - trail_pct)   # 0.405

    # Price below the stop → fire
    result = _evaluate_trail(entry, peak, price=expected_stop - 0.01)
    assert result is not None
    stop, reason = result
    assert stop == pytest.approx(expected_stop)
    assert reason == "TRAILING_STOP"

    # Price above the stop → no fire
    assert _evaluate_trail(entry, peak, price=expected_stop + 0.01) is None

    # Peak below activation gate → no fire even if price collapses
    pre_activation_peak = entry + activation - 0.02
    assert _evaluate_trail(entry, pre_activation_peak, price=entry * 0.5) is None


# ===========================================================================
# Our test 9 — Tier boundary at exactly 0.30 falls into upper tier
# ===========================================================================

def test_tier_boundary_at_lo_goes_to_upper_tier():
    """Half-open intervals [lo, hi).  A peak of exactly 0.30 must match
    the (0.30, 0.50, 0.10) tier, not (0.00, 0.30, 0.15)."""
    pct = lookup_trail_pct(0.30, TIERS)
    assert pct == pytest.approx(0.10)

    # Same for the next boundary
    pct = lookup_trail_pct(0.50, TIERS)
    assert pct == pytest.approx(0.07)

    # Just below boundary stays in lower tier
    pct = lookup_trail_pct(0.299999, TIERS)
    assert pct == pytest.approx(0.15)


def test_top_tier_lookup_at_exactly_one():
    """Last tier's hi is inclusive — peak == 1.00 must resolve."""
    pct = lookup_trail_pct(1.00, TIERS)
    assert pct == pytest.approx(0.03)


def test_lookup_returns_none_for_out_of_range():
    """Defensive: peak above all tiers (shouldn't happen for [0,1] prices,
    but the function shouldn't crash)."""
    assert lookup_trail_pct(1.01, TIERS) is None
    assert lookup_trail_pct(-0.01, TIERS) is None
