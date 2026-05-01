"""
test_ask_depth_sizing.py — Regression tests for the orderbook ask-depth
sizing cap (replaces the deprecated MAX_LIQUIDITY_TAKE_PCT rule).

Bug story (2026-04-30)
-----------------------
The old `MAX_LIQUIDITY_TAKE_PCT × Gamma's liquidity_usd` rule used a
wrong basis (bid + ask combined).  It over-restricted on books with
deep bids and thin asks, and under-restricted on the inverse.  The
new rule:

  acceptable_ask_depth = sum(p × size for asks at p ≤ sweep_limit)
  max_take = acceptable_ask_depth × MAX_TAKE_PCT_OF_ASK_DEPTH
  if max_take < MIN_FILLABLE_USDC: skip
  final_size = min(intended_size, max_take)

These tests pin all four behaviors (cap doesn't bind, cap binds,
book-too-thin skip, intent-already-fits-deep-book passthrough).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import execution


def _book(asks, bids=None):
    return {
        "asks":      [{"price": str(p), "size": str(s)} for p, s in asks],
        "bids":      [{"price": str(p), "size": str(s)} for p, s in (bids or [])],
        "tick_size": "0.01", "neg_risk": True,
    }


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _basic_signal(**overrides):
    sig = {
        "contract_id":      "0xabc",
        "recommended_side": "YES",
        "kelly_size":       10.0,
        "yes_token_id":     "tok_yes",
        "no_token_id":      "tok_no",
        "yes_price":        0.30,
        "market_p":         0.30,
        "city":             "Test",
        "date":             "2026-05-01",
        "scan_timestamp":   "2026-04-30T22:00:00+00:00",
    }
    sig.update(overrides)
    return sig


# ===========================================================================
# Cap doesn't bind on a deep book
# ===========================================================================

def test_intended_size_passes_through_on_deep_book(temp_db, monkeypatch):
    """Deep ask side at acceptable prices → cap doesn't bind, full $10
    intent flows through to the order."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    # 100 shares @ 0.29 = $29 acceptable depth.  66% of $29 = $19.14, well > $10
    client.get_order_book.return_value = _book(
        asks=[(0.29, 100)],
        bids=[(0.28, 50)],
    )
    captured = {}
    class _SpyOrderArgs:
        def __init__(self, **kw):
            captured.update(kw)
            self.__dict__.update(kw)
    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xok", "status": "live",
        "takingAmount": "", "makingAmount": "",
    }

    result = execution.execute_signal(_basic_signal(), client=client)
    assert result["status"] == "placed"
    # Order size = $10 / 0.30 = 33.33 shares.  Cap didn't shrink it.
    assert captured["size"] == pytest.approx(10.0 / captured["price"], rel=1e-3)


# ===========================================================================
# Cap binds on a thinner book
# ===========================================================================

def test_intended_size_capped_when_ask_depth_thin(temp_db, monkeypatch):
    """$10 intent on $5 of acceptable asks.  At 66% cap, max take = $3.30.
    Order size shrinks accordingly; intended size preserved as
    target_size_usdc for top-up to fill the gap later."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    # 17 shares × 0.29 ≈ $4.93 of acceptable asks.  Cap = $4.93 × 0.66 ≈ $3.25
    client.get_order_book.return_value = _book(
        asks=[(0.29, 17), (0.40, 100)],   # 0.40 above sweep_limit (0.30)
        bids=[(0.28, 1)],
    )
    captured = {}
    class _SpyOrderArgs:
        def __init__(self, **kw):
            captured.update(kw)
            self.__dict__.update(kw)
    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xcapped", "status": "live",
        "takingAmount": "", "makingAmount": "",
    }

    result = execution.execute_signal(_basic_signal(kelly_size=10.0), client=client)
    assert result["status"] == "placed"
    # Order size = $3.25 / limit_price ≈ 11 shares (much less than $10's 33)
    final_usdc = captured["size"] * captured["price"]
    assert final_usdc < 4.0, f"expected cap to bind, got ${final_usdc:.2f}"
    assert final_usdc > 2.5, f"expected ~$3.25, got ${final_usdc:.2f}"


# ===========================================================================
# Skip when book is too thin
# ===========================================================================

def test_skip_when_max_take_below_floor(temp_db, monkeypatch):
    """Acceptable ask depth × cap < MIN_FILLABLE_USDC → skip the order
    entirely (don't place a tiny-fraction order; wait for liquidity)."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    # 5 shares × 0.29 = $1.45 acceptable depth.  Cap = $0.96 < $2 floor.
    client.get_order_book.return_value = _book(
        asks=[(0.29, 5), (0.40, 100)],
        bids=[(0.28, 1)],
    )

    result = execution.execute_signal(_basic_signal(), client=client)
    assert result["status"] == "skip"
    assert result["reason"] == "book_too_thin"
    # CLOB was never asked to place an order
    client.create_and_post_order.assert_not_called()


# ===========================================================================
# Sizing flag disables the cap entirely
# ===========================================================================

def test_disabled_flag_bypasses_cap(temp_db, monkeypatch):
    """LIQUIDITY_AWARE_SIZING=False → no cap at all, regardless of book.
    Useful for testing or for users who explicitly opt out."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "LIQUIDITY_AWARE_SIZING", False)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    # Tiny book — would normally be skipped, but flag is off
    client.get_order_book.return_value = _book(
        asks=[(0.29, 1)], bids=[(0.28, 1)]
    )
    captured = {}
    class _SpyOrderArgs:
        def __init__(self, **kw): captured.update(kw)
    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xok", "status": "live",
        "takingAmount": "", "makingAmount": "",
    }

    result = execution.execute_signal(_basic_signal(), client=client)
    assert result["status"] == "placed", "should place when cap is disabled"


# ===========================================================================
# Cap basis: ask-side ONLY, not bid + ask combined
# ===========================================================================

def test_bid_depth_does_not_inflate_cap(temp_db, monkeypatch):
    """The point of moving from `total liquidity` to `ask-side depth`:
    deep bids should NOT make the cap more permissive when buying.
    If asks are thin, the cap binds regardless of how deep bids are."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    # Massive bids ($300), tiny asks ($4.5).  Old rule: cap on $304 total
    # = ~$120 max — way over our $10 intent.  New rule: cap on $4.5 asks
    # × 0.66 = $2.97 — much smaller.
    client.get_order_book.return_value = _book(
        asks=[(0.29, 15), (0.40, 100)],     # $4.35 acceptable
        bids=[(0.28, 1000)],                 # $280 deep bid
    )
    captured = {}
    class _SpyOrderArgs:
        def __init__(self, **kw):
            captured.update(kw)
    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xok", "status": "live",
        "takingAmount": "", "makingAmount": "",
    }

    execution.execute_signal(_basic_signal(kelly_size=10.0), client=client)
    final_usdc = captured["size"] * captured["price"]
    # Should be capped to ~$2.87 ($4.35 × 0.66), NOT free-running to $10
    assert final_usdc < 3.5, (
        f"deep bids must not inflate the buy-side cap, got ${final_usdc:.2f}"
    )


# ===========================================================================
# Sweep limit + cap interaction
# ===========================================================================

def test_only_asks_at_acceptable_prices_count(temp_db, monkeypatch):
    """Asks at prices > sweep_limit (best_ask + walk_cents, capped at
    MPV_MAX_PRICE) don't count toward acceptable depth — they're outside
    our willingness-to-pay window."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    # touch=0.29, walk_limit=0.30 (best_ask + 1¢).
    # Asks at 0.29 + 0.30 ≤ limit (counted).  Asks at 0.50 > limit (excluded).
    client.get_order_book.return_value = _book(
        asks=[(0.29, 5), (0.30, 5), (0.50, 1000)],   # acceptable: 5×0.29 + 5×0.30 = $2.95
        bids=[(0.28, 1)],
    )

    # MIN_FILLABLE_USDC = $2.0; acceptable cap = $2.95 × 0.66 = $1.95 < floor → skip
    result = execution.execute_signal(_basic_signal(kelly_size=10.0), client=client)
    assert result["status"] == "skip", (
        "asks at $0.50 must NOT inflate acceptable depth (they're past the walk limit)"
    )
    assert result["reason"] == "book_too_thin"
