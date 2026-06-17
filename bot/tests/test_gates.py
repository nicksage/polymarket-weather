"""
test_gates.py — Lock in evaluate_gates behavior.

Specifically W4 (market-anchored risk cap), but also regression coverage
for the existing market-skeptic, low-edge, and exposure gates so they
don't silently change shape when future workstreams touch them.

Run:
    cd bot
    python -m pytest tests/test_gates.py -v
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scheduled_predictor import (  # type: ignore
    evaluate_gates,
    MARKET_DISAGREEMENT_LIQ_THRESHOLD,
    MARKET_DISAGREEMENT_PP_THRESHOLD,
    MIN_EDGE,
    MIN_EDGE_LOW_MKT,
    HIGH_MKT_THRESHOLD,
)


def _ok_gate_call(**overrides):
    """Baseline kwargs: a sensible BUY candidate (cheap market, decent
    edge, liquid, not yet acted, mid-afternoon hour).  Tests override
    one knob at a time to isolate which gate trips."""
    base = dict(
        current_hour=15,
        edge=0.10,         # +10pp edge (passes MIN_EDGE_LOW_MKT)
        market_p=0.30,     # cheap (< HIGH_MKT_THRESHOLD)
        liquidity=500,     # below W4 threshold
        deployed_today=0,
        trades_today=0,
        already_acted=False,
    )
    base.update(overrides)
    return base


def test_baseline_passes():
    ok, reason = evaluate_gates(**_ok_gate_call())
    assert ok, f"baseline candidate should pass gates, got reason={reason!r}"


# ---------------------------------------------------------------------------
# W4 — market-anchored risk cap
# ---------------------------------------------------------------------------

def test_w4_fires_on_high_liq_extreme_edge():
    """Liquid market + huge edge → veto.  This is the case the
    guardrail exists to catch: market_p above the skeptic floor (so
    market_too_skeptical doesn't fire) but a huge edge that's
    suspicious given the market's liquidity."""
    ok, reason = evaluate_gates(**_ok_gate_call(
        liquidity=25_000,                              # well above LIQ threshold
        edge=MARKET_DISAGREEMENT_PP_THRESHOLD + 0.01,  # just over PP threshold
        market_p=0.20,                                 # above skeptic floor
    ))
    assert not ok, "liquid + extreme-edge should veto"
    assert reason.startswith("liquid_market_strong_disagreement"), (
        f"expected liquid_market_strong_disagreement reason, got: {reason!r}"
    )


def test_w4_quiet_on_low_liq_extreme_edge():
    """Low-liquidity market with extreme edge does NOT trip W4 — the
    thin book is precisely where genuine edge can survive uncorrected."""
    ok, reason = evaluate_gates(**_ok_gate_call(
        liquidity=MARKET_DISAGREEMENT_LIQ_THRESHOLD - 1,
        edge=0.60,                                # huge edge
        market_p=0.20,                            # above skeptic floor
    ))
    assert ok, (f"low-liq extreme edge should NOT trip W4, "
                 f"got veto: {reason!r}")


def test_w4_quiet_on_high_liq_moderate_edge():
    """Liquid market with normal edge passes — W4 only fires on the
    extreme-disagreement case, not on every liquid book."""
    ok, reason = evaluate_gates(**_ok_gate_call(
        liquidity=50_000,                             # liquid
        edge=MARKET_DISAGREEMENT_PP_THRESHOLD - 0.05, # just under PP threshold
        market_p=0.30,
    ))
    assert ok, (f"high-liq moderate-edge should pass W4, "
                 f"got veto: {reason!r}")


def test_w4_fires_after_low_edge_check():
    """Sanity: W4 sits AFTER the low-edge gate, so a tiny-edge candidate
    is rejected as low_edge not as liquid_market_strong_disagreement.
    Keeps gate reasons informative (low-edge is the actual cause)."""
    ok, reason = evaluate_gates(**_ok_gate_call(
        liquidity=25_000,
        edge=0.02,        # under MIN_EDGE_LOW_MKT
        market_p=0.30,
    ))
    assert not ok
    assert reason.startswith("low_edge"), (
        f"expected low_edge to fire before W4, got: {reason!r}"
    )


# ---------------------------------------------------------------------------
# Regression coverage so future edits don't accidentally reshape these
# ---------------------------------------------------------------------------

def test_market_too_skeptical_blocks_garbage_signals(monkeypatch):
    # As of 2026-06-17, the loss-stopper gate sits ABOVE
    # market_too_skeptical and catches the same garbage with a more
    # specific reason.  Disable it here so this test isolates
    # market_too_skeptical's behavior.  When LOSS_STOPPER_ENABLED is
    # eventually removed (per its checked removal trigger), this
    # monkeypatch becomes a no-op and the test still passes.
    import scheduled_predictor as sp
    monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", False)
    ok, reason = evaluate_gates(**_ok_gate_call(market_p=0.05, edge=0.85))
    assert not ok
    assert "market_too_skeptical" in reason


def test_loss_stopper_fires_before_market_too_skeptical():
    """Lock in the new gate order: loss-stopper takes precedence over
    market_too_skeptical when both would match, because its reason is
    more diagnostic for the cold-bias failure mode."""
    ok, reason = evaluate_gates(**_ok_gate_call(market_p=0.05, edge=0.85))
    assert not ok
    assert "loss_stopper_high_disagreement" in reason


def test_too_early_blocks_before_min_hour():
    ok, reason = evaluate_gates(**_ok_gate_call(current_hour=8))
    assert not ok
    assert "too_early" in reason


def test_trades_cap_blocks_after_limit():
    ok, reason = evaluate_gates(**_ok_gate_call(trades_today=10_000))
    assert not ok
    assert "trades_cap" in reason