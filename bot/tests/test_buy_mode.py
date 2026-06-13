"""
test_buy_mode.py — Lock in PREDICTOR_BUY_MODE dispatch (2026-06-12).

Two strategies for deciding whether the top-P bin is buyable:
  "edge"        — default; existing behavior (requires edge >= MIN_EDGE)
  "probability" — buy when our_p >= PREDICTOR_MIN_PROB_TO_BUY,
                  regardless of edge

The probability mode is the new code path.  These tests pin:
  - edge mode (default) preserves legacy gate semantics
  - probability mode flips ONLY the edge gate; every other gate
    (market sanity floor, W4, priced_in, thin_book, dedup,
     trade caps) still applies
  - threshold gate fires when our_p < PREDICTOR_MIN_PROB_TO_BUY
  - threshold gate passes when our_p >= PREDICTOR_MIN_PROB_TO_BUY
  - unknown mode strings fall back to edge mode

Run:
    cd bot
    python -m pytest tests/test_buy_mode.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import scheduled_predictor as sp  # type: ignore


@pytest.fixture
def edge_mode(monkeypatch):
    monkeypatch.setattr(sp, "PREDICTOR_BUY_MODE", "edge")


@pytest.fixture
def prob_mode(monkeypatch):
    monkeypatch.setattr(sp, "PREDICTOR_BUY_MODE", "probability")
    monkeypatch.setattr(sp, "PREDICTOR_MIN_PROB_TO_BUY", 0.50)


# ============================================================
# Defaults — read env-driven values at import time
# ============================================================

def test_default_buy_mode_is_edge():
    """Without env override, BUY_MODE defaults to 'edge' — legacy
    behavior preserved unless operator opts in."""
    # If pytest is being run with PREDICTOR_BUY_MODE set in the shell,
    # skip — this asserts the source default, not the active value.
    if os.environ.get("PREDICTOR_BUY_MODE"):
        pytest.skip("PREDICTOR_BUY_MODE set in env; can't verify source default")
    assert sp.PREDICTOR_BUY_MODE == "edge"


def test_default_min_prob_to_buy_is_half():
    """0.50 is the documented spec default."""
    if os.environ.get("PREDICTOR_MIN_PROB_TO_BUY"):
        pytest.skip("PREDICTOR_MIN_PROB_TO_BUY set in env")
    assert sp.PREDICTOR_MIN_PROB_TO_BUY == 0.50


def test_env_var_is_lowercased():
    """The .lower() call on the env read means UPPERCASE/MixedCase
    values are normalized. Verified by reading the source —
    `os.getenv(...).lower()` is right there."""
    import inspect
    src = inspect.getsource(sp)
    assert 'os.getenv("PREDICTOR_BUY_MODE", "edge").lower()' in src, (
        "PREDICTOR_BUY_MODE env read should call .lower() so "
        "'PROBABILITY' / 'Probability' all work."
    )


# ============================================================
# Edge mode — legacy behavior preserved
# ============================================================

def test_edge_mode_passes_high_edge(edge_mode):
    """edge=0.20 over a 0.50 market is a clear buy in edge mode."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.20, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.70,
    )
    assert ok, f"high-edge buy should pass; got reason={reason!r}"


def test_edge_mode_rejects_low_edge(edge_mode):
    """In edge mode, edge=0.03 below MIN_EDGE_LOW_MKT=0.05 is rejected
    even when our_p is high."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.03, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.53,
    )
    assert not ok
    assert "low_edge" in reason


def test_edge_mode_low_prob_irrelevant_to_gate(edge_mode):
    """In edge mode, our_p<0.50 with strong edge still buys — the
    probability gate must NOT fire in edge mode."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.20, market_p=0.20,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.40,    # below 0.50
    )
    assert ok, f"edge mode should ignore our_p; got reason={reason!r}"


# ============================================================
# Probability mode — new behavior
# ============================================================

def test_probability_mode_passes_high_prob_zero_edge(prob_mode):
    """In probability mode, edge=0 but our_p=0.80 must pass.
    This is the headline difference from edge mode."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.80,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.80,
    )
    assert ok, f"probability mode should pass on our_p alone; "\
        f"got reason={reason!r}"


def test_probability_mode_rejects_low_prob(prob_mode):
    """In probability mode, our_p=0.40 fails the gate even when
    edge is huge. This is the headline 'no edge requirement, but
    high confidence required' behavior."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.20, market_p=0.20,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.40,
    )
    assert not ok
    assert "low_prob" in reason, f"expected low_prob, got {reason!r}"


def test_probability_mode_at_threshold_passes(prob_mode):
    """our_p exactly at PREDICTOR_MIN_PROB_TO_BUY passes (>= comparison)."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.50,
    )
    assert ok, f"our_p at threshold should pass; got reason={reason!r}"


def test_probability_mode_custom_threshold(monkeypatch):
    """Operator-tuned threshold: 0.70."""
    monkeypatch.setattr(sp, "PREDICTOR_BUY_MODE", "probability")
    monkeypatch.setattr(sp, "PREDICTOR_MIN_PROB_TO_BUY", 0.70)
    # our_p=0.65 < 0.70 → reject
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.65,
    )
    assert not ok
    assert "low_prob" in reason
    # our_p=0.75 >= 0.70 → pass
    ok2, _ = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.75,
    )
    assert ok2


# ============================================================
# Cross-mode invariants — other gates still apply
# ============================================================

def test_probability_mode_market_sanity_floor_still_applies(prob_mode):
    """MIN_MARKET_PROB=0.15 sanity floor protects probability mode too —
    even if our_p is high, market_p<0.15 means model is almost certainly
    broken."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.05,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.90,
    )
    assert not ok
    assert "market_too_skeptical" in reason


def test_probability_mode_w4_still_applies(prob_mode):
    """W4 — liquid-market strong-disagreement veto — protects both modes."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.50, market_p=0.20,
        liquidity=20000.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.70,
    )
    assert not ok
    assert "liquid_market_strong_disagreement" in reason


def test_probability_mode_priced_in_still_applies(prob_mode):
    """MAX_MARKET_PRICE=0.95 priced-in veto fires in both modes."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.96,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.99,
    )
    assert not ok
    assert "priced_in" in reason


def test_probability_mode_thin_book_still_applies(prob_mode):
    """MIN_LIQUIDITY_USD veto fires in both modes."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.50,
        liquidity=50.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.80,
    )
    assert not ok
    assert "thin_book" in reason


def test_probability_mode_dedup_still_applies(prob_mode):
    """already_acted=True vetoes both modes."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=True, our_p=0.80,
    )
    assert not ok
    assert "dedup" in reason


def test_probability_mode_trade_cap_still_applies(prob_mode):
    """Daily trade cap fires in both modes."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.00, market_p=0.50,
        liquidity=500.0, deployed_today=0.0,
        trades_today=sp.MAX_TRADES_PER_DAY,
        already_acted=False, our_p=0.80,
    )
    assert not ok
    assert "trades_cap" in reason


def test_probability_mode_time_window_still_applies(prob_mode):
    """current_hour < MIN_TRIGGER_HOUR rejected even in probability mode."""
    ok, reason = sp.evaluate_gates(
        current_hour=5, edge=0.00, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.80,
    )
    assert not ok
    assert "too_early" in reason


# ============================================================
# Fallback / safety
# ============================================================

def test_our_p_derived_when_omitted(edge_mode):
    """For backwards compatibility, omitting our_p derives it from
    edge + market_p (edge mode behavior unchanged)."""
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.20, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False,
        # our_p omitted
    )
    assert ok, f"omitting our_p should not break edge mode; "\
        f"got reason={reason!r}"


def test_unknown_mode_falls_back_to_edge(monkeypatch):
    """An unknown PREDICTOR_BUY_MODE string is treated as 'edge'
    rather than crashing. The else branch in the dispatch handles
    anything that isn't literally 'probability'."""
    monkeypatch.setattr(sp, "PREDICTOR_BUY_MODE", "garbage")
    ok, reason = sp.evaluate_gates(
        current_hour=15, edge=0.02, market_p=0.50,
        liquidity=500.0, deployed_today=0.0, trades_today=0,
        already_acted=False, our_p=0.52,
    )
    assert not ok
    assert "low_edge" in reason, (
        f"unknown mode should fall back to edge gate; got {reason!r}"
    )