"""
test_exit_ladder.py — Pure-function tests for the sell-side ladder.
"""

from __future__ import annotations

import os
import sys

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from exit_ladder import (
    ladder_price, cross_spread_price, is_ladder_exhausted,
    SELL_LADDER_MULTIPLIERS, MAX_LADDER_RETRIES,
)


# ===========================================================================
# Ladder pricing — each rung
# ===========================================================================

def test_rung_0_is_99pct_of_intended():
    assert ladder_price(0.50, 0) == pytest.approx(0.495)
    assert ladder_price(0.80, 0) == pytest.approx(0.792)


def test_rung_1_is_98pct():
    assert ladder_price(0.50, 1) == pytest.approx(0.49)


def test_rung_2_is_97pct():
    assert ladder_price(0.50, 2) == pytest.approx(0.485)


def test_rung_3_is_96pct():
    assert ladder_price(0.50, 3) == pytest.approx(0.48)


def test_rung_4_returns_none_exhausted():
    assert ladder_price(0.50, 4) is None
    assert ladder_price(0.50, 99) is None


def test_negative_retry_raises():
    with pytest.raises(ValueError):
        ladder_price(0.50, -1)


def test_pricing_is_rounded_to_4dp():
    """Polymarket quantum is 0.001; we round to 4 dp defensively."""
    # 0.123456 * 0.99 = 0.12222144 → rounded to 0.1222
    p = ladder_price(0.123456, 0)
    assert p == pytest.approx(0.1222)


def test_max_ladder_retries_matches_multipliers():
    assert MAX_LADDER_RETRIES == len(SELL_LADDER_MULTIPLIERS) == 4


# ===========================================================================
# Ladder exhaustion
# ===========================================================================

def test_is_ladder_exhausted():
    assert not is_ladder_exhausted(0)
    assert not is_ladder_exhausted(1)
    assert not is_ladder_exhausted(2)
    assert not is_ladder_exhausted(3)
    assert is_ladder_exhausted(4)
    assert is_ladder_exhausted(5)
    assert is_ladder_exhausted(100)


# ===========================================================================
# Cross-spread pricing
# ===========================================================================

def test_cross_spread_subtracts_one_tick():
    assert cross_spread_price(0.52) == pytest.approx(0.519)
    assert cross_spread_price(0.95) == pytest.approx(0.949)


def test_cross_spread_floors_at_one_tick():
    """Posting at 0.001 is the minimum; never go non-positive."""
    assert cross_spread_price(0.001) == pytest.approx(0.001)
    assert cross_spread_price(0.0005) == pytest.approx(0.001)


def test_cross_spread_zero_or_negative_raises():
    with pytest.raises(ValueError):
        cross_spread_price(0.0)
    with pytest.raises(ValueError):
        cross_spread_price(-0.10)


# ===========================================================================
# End-to-end ladder progression for a hypothetical exit at 0.55
# ===========================================================================

def test_ladder_progression_for_realistic_exit():
    intended = 0.55
    expected = [0.5445, 0.5390, 0.5335, 0.5280]
    for rung, expected_price in enumerate(expected):
        actual = ladder_price(intended, rung)
        assert actual == pytest.approx(expected_price, abs=1e-4), (
            f"rung {rung}: expected {expected_price}, got {actual}"
        )
    # Beyond rung 3 → None (caller crosses spread)
    assert ladder_price(intended, 4) is None


def test_max_giveup_is_4pct():
    """Floor of the ladder is 96% of intended — 4% give-up."""
    intended = 1.00
    last_rung = ladder_price(intended, MAX_LADDER_RETRIES - 1)
    assert last_rung == pytest.approx(0.96)
    # Confirm the give-up is exactly 4%
    assert (intended - last_rung) / intended == pytest.approx(0.04)


# ===========================================================================
# Bid-anchored re-anchoring (the patient-rung-vs-stale-trigger fix)
#
# Semantics: ladder_price returns min(patient × multiplier, bid + 1 tick).
# Patient is the price ceiling we'd accept; bid+tick is the most-fillable
# price.  We chase whichever is LOWER (more aggressive sell, more likely
# to fill).  Marketable case (patient < bid) fills at bid via Polymarket's
# best-execution semantics.
# ===========================================================================

def test_bid_far_above_patient_returns_patient():
    """When bid+tick is far above patient, patient is the cheaper sell
    and would fill immediately at the bid (marketable cross).  Polymarket
    fills at best execution — we capture the bid price."""
    # Trigger 0.50, rung 1 → patient = 0.49.
    # Bid 0.495 → bid+tick 0.496.  min(0.49, 0.496) = 0.49.
    # Posting 0.49 means our sell is below the bid — marketable.
    assert ladder_price(0.50, 1, current_best_bid=0.495) == pytest.approx(0.49)


def test_bid_below_patient_chases_down_to_bid_plus_tick():
    """When bid has dropped well below patient, post bid+tick instead.
    The patient price would sit unfillable far above the bid; the bid+tick
    price makes us the lowest ask, fillable on a small upward tick."""
    # Trigger 0.40, rung 3 → patient = 0.384.
    # Bid 0.20 → bid+tick 0.201.  min(0.384, 0.201) = 0.201.
    # Posting 0.201 chases the bid; small upward tick lifts us.
    assert ladder_price(0.40, 3, current_best_bid=0.20) == pytest.approx(0.201)


def test_bid_at_patient_picks_patient():
    """When patient and bid+tick are equal, either choice is fine; min
    returns patient (the equal value).  Marginal edge case, included
    for confidence."""
    # Trigger 0.50, rung 0 → patient = 0.495.
    # Bid 0.494 → bid+tick = 0.495.  min(0.495, 0.495) = 0.495.
    assert ladder_price(0.50, 0, current_best_bid=0.494) == pytest.approx(0.495)


def test_bid_just_below_patient_chases():
    """When bid+tick is even slightly below patient, bid+tick wins."""
    # Trigger 0.50, rung 0 → patient = 0.495.
    # Bid 0.490 → bid+tick = 0.491.  min(0.495, 0.491) = 0.491.
    assert ladder_price(0.50, 0, current_best_bid=0.490) == pytest.approx(0.491)


def test_no_bid_provided_preserves_legacy_behavior():
    """When current_best_bid is None (book unreachable), behavior matches
    the pre-fix code: pure intended × multiplier.  Safe degradation."""
    assert ladder_price(0.50, 0, current_best_bid=None) == pytest.approx(0.495)
    assert ladder_price(0.50, 1, current_best_bid=None) == pytest.approx(0.49)
    # Same as omitting the kwarg entirely
    assert ladder_price(0.50, 2) == pytest.approx(0.485)


def test_zero_or_negative_bid_treated_as_missing():
    """Bid of 0 or negative is the same as None — defensive: don't post
    at 0.001 just because the API returned junk."""
    assert ladder_price(0.50, 0, current_best_bid=0.0) == pytest.approx(0.495)
    assert ladder_price(0.50, 0, current_best_bid=-0.01) == pytest.approx(0.495)


def test_bid_anchor_returns_none_when_exhausted():
    """The bid-anchor logic doesn't fire past rung 3 — caller is expected
    to use cross_spread_price directly there."""
    assert ladder_price(0.50, 4, current_best_bid=0.30) is None


def test_realistic_chunked_advance_with_dropping_bid():
    """End-to-end: trigger fires at 0.40, market drops across cycles.
    Each rung re-anchors closer to (or below) the falling bid, fixing
    the user-reported bleed bug where rungs sit unfillable above the bid.
    """
    intended = 0.40
    # Bid sequence as price drifts down across cycles
    bid_sequence = [0.395, 0.380, 0.330, 0.220]  # rungs 0..3
    expected = [
        # rung 0: patient 0.396, bid+tick 0.396 → tied; min picks 0.396
        min(0.40 * 0.99, 0.395 + 0.001),
        # rung 1: patient 0.392, bid+tick 0.381 → bid+tick wins
        min(0.40 * 0.98, 0.380 + 0.001),
        # rung 2: patient 0.388, bid+tick 0.331 → bid+tick wins
        min(0.40 * 0.97, 0.330 + 0.001),
        # rung 3: patient 0.384, bid+tick 0.221 → bid+tick wins
        min(0.40 * 0.96, 0.220 + 0.001),
    ]
    for rung, (bid, expected_price) in enumerate(zip(bid_sequence, expected)):
        actual = ladder_price(intended, rung, current_best_bid=bid)
        assert actual == pytest.approx(round(expected_price, 4)), (
            f"rung {rung} bid={bid}: expected {expected_price}, got {actual}"
        )


def test_realistic_collapsed_bid_chases_all_the_way_down():
    """When the bid has collapsed FAR below trigger, every rung re-anchors
    to bid+tick.  In production the bleed circuit-breaker (monitor.py)
    would force cross-spread before getting here — this test confirms the
    pricing layer would also produce a fillable price even if it wasn't.
    """
    intended = 0.40
    bid = 0.10  # bid collapsed to 25% of trigger
    expected_price = 0.101  # bid+tick — all rungs converge here
    for rung in range(4):
        actual = ladder_price(intended, rung, current_best_bid=bid)
        assert actual == pytest.approx(expected_price), (
            f"rung {rung}: with collapsed bid {bid}, expected bid+tick "
            f"{expected_price}, got {actual}"
        )
