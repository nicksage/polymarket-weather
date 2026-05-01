"""
test_polymarket_v2_drift.py — Regression tests for the py_clob_client v1→v2
migration and downstream Polymarket API drift.

Why this file exists
--------------------
On 2026-04-29 we hit two critical live-trading bugs caused by API drift:
  1. v1 of py_clob_client signs orders against deprecated Polymarket
     exchange contracts → every live order returned `order_version_mismatch`
  2. v2's `get_order_book` returns bids ASCENDING (worst→best); v1's
     OrderBookSummary may have been descending.  Reading `bids[0]` for
     "best bid" silently returned the WORST bid → the cross-spread sell
     would execute at the bottom of the book.

These tests pin both fixes so a future refactor or a v1 reinstall can't
silently regress them.
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

import execution
import polymarket
import risk


# ===========================================================================
# _get_best_bid — must return MAX bid regardless of sort order
# ===========================================================================

def test_get_best_bid_returns_max_with_v2_dict_ascending():
    """v2 returns the orderbook as a dict with bids ASCENDING (worst→best).
    Bug 2026-04-29: code did `bids[0]` and returned the worst price.
    Fix: take max() explicitly.  This pins the fix."""
    client = MagicMock()
    client.get_order_book.return_value = {
        "asset_id": "tok",
        "bids": [
            {"price": "0.01", "size": "2000000"},
            {"price": "0.02", "size": "100"},
            {"price": "0.45", "size": "500"},
            {"price": "0.50", "size": "200"},
            {"price": "0.53", "size": "50"},   # ← actual best bid
        ],
        "asks": [],
        "tick_size": "0.01",
        "neg_risk": True,
    }
    best = execution._get_best_bid(client, "tok")
    assert best == pytest.approx(0.53), (
        f"expected best bid 0.53 (max), got {best} — "
        f"likely regressed to bids[0] which is the WORST bid"
    )


def test_get_best_bid_returns_max_with_descending_order():
    """v1's OrderBookSummary may have been descending (best→worst).
    The fix uses max() so either ordering produces the right answer."""
    client = MagicMock()
    client.get_order_book.return_value = {
        "bids": [
            {"price": "0.53", "size": "50"},
            {"price": "0.45", "size": "500"},
            {"price": "0.01", "size": "2000000"},
        ],
    }
    assert execution._get_best_bid(client, "tok") == pytest.approx(0.53)


def test_get_best_bid_handles_v1_orderbook_summary_object():
    """Back-compat: if a v1-style object is returned with .bids of
    objects with .price attributes, the helper still works."""
    class _V1Bid:
        def __init__(self, price): self.price = price

    class _V1OrderBook:
        bids = [_V1Bid("0.42"), _V1Bid("0.50")]

    client = MagicMock()
    client.get_order_book.return_value = _V1OrderBook()
    assert execution._get_best_bid(client, "tok") == pytest.approx(0.50)


def test_get_best_bid_empty_book_returns_none():
    client = MagicMock()
    client.get_order_book.return_value = {"bids": [], "asks": []}
    assert execution._get_best_bid(client, "tok") is None


def test_get_best_bid_garbage_prices_skipped():
    """Non-numeric prices are dropped, not crashed on."""
    client = MagicMock()
    client.get_order_book.return_value = {
        "bids": [
            {"price": "not-a-number", "size": "1"},
            {"price": "0.30", "size": "1"},
            {"price": None, "size": "1"},
        ],
    }
    assert execution._get_best_bid(client, "tok") == pytest.approx(0.30)


def test_get_best_bid_api_failure_returns_none():
    """A network/API failure returns None (caller falls back), doesn't raise."""
    client = MagicMock()
    client.get_order_book.side_effect = RuntimeError("network down")
    assert execution._get_best_bid(client, "tok") is None


# ===========================================================================
# Gamma response: new fields surface in normalized output
# ===========================================================================

def _make_market(**overrides) -> dict:
    """Synthetic Gamma market dict with realistic field shapes."""
    import json
    base = {
        "id": "1234567",
        "conditionId": "0xabc",
        "question": "Will the high temp on Apr 9 be 72°F?",
        "groupItemTitle": "72°F",
        "outcomes": json.dumps(["YES", "NO"]),
        "outcomePrices": json.dumps(["0.42", "0.58"]),
        "clobTokenIds": json.dumps(["tok_yes", "tok_no"]),
        "liquidityNum": 5000.0,
        "volumeNum": 12000.0,
        "endDate": "2026-04-10T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_normalize_surfaces_accepting_orders():
    raw = _make_market(acceptingOrders=False)
    out = polymarket._normalize_sub_market(raw, "")
    assert out["accepting_orders"] is False


def test_normalize_defaults_accepting_orders_true_when_absent():
    """Old cached responses without the field default to True (permissive
    — never block trading on a missing flag)."""
    raw = _make_market()  # no acceptingOrders field
    out = polymarket._normalize_sub_market(raw, "")
    assert out["accepting_orders"] is True


def test_normalize_surfaces_enable_order_book():
    raw = _make_market(enableOrderBook=False)
    out = polymarket._normalize_sub_market(raw, "")
    assert out["enable_order_book"] is False


def test_normalize_surfaces_neg_risk_fields():
    raw = _make_market(
        negRisk=True,
        negRiskMarketID="0xdeadbeef",
        orderPriceMinTickSize=0.01,
    )
    out = polymarket._normalize_sub_market(raw, "")
    assert out["neg_risk"] is True
    assert out["neg_risk_market_id"] == "0xdeadbeef"
    assert out["tick_size"] == pytest.approx(0.01)


def test_normalize_surfaces_fee_schedule():
    raw = _make_market(
        feeSchedule={"makerBaseRateBps": 0, "takerBaseRateBps": 100},
    )
    out = polymarket._normalize_sub_market(raw, "")
    assert out["fee_schedule"] == {"makerBaseRateBps": 0, "takerBaseRateBps": 100}


def test_normalize_fee_schedule_defaults_to_empty():
    raw = _make_market()  # no feeSchedule field
    out = polymarket._normalize_sub_market(raw, "")
    assert out["fee_schedule"] == {}


def test_normalize_tick_size_garbage_returns_none():
    """Bad tick_size in payload shouldn't crash — store None."""
    raw = _make_market(orderPriceMinTickSize="not-a-number")
    out = polymarket._normalize_sub_market(raw, "")
    assert out["tick_size"] is None


# ===========================================================================
# check_market_accepting_orders pre-trade gate
# ===========================================================================

def test_check_accepting_orders_passes_default():
    """Missing field defaults True (permissive)."""
    sig = {"city": "Chicago"}
    assert risk.check_market_accepting_orders(sig).passed is True


def test_check_accepting_orders_blocks_when_false():
    sig = {"accepting_orders": False, "city": "Chicago"}
    result = risk.check_market_accepting_orders(sig)
    assert result.passed is False
    assert "acceptingOrders=False" in result.reason


def test_check_enable_order_book_blocks_when_false():
    sig = {"enable_order_book": False, "city": "Chicago"}
    result = risk.check_market_accepting_orders(sig)
    assert result.passed is False
    assert "enableOrderBook=False" in result.reason


def test_check_accepting_orders_passes_when_both_true():
    sig = {"accepting_orders": True, "enable_order_book": True}
    assert risk.check_market_accepting_orders(sig).passed is True


def test_check_accepting_orders_wired_into_pre_checks():
    """Pin the integration: a paused market must fail run_pre_checks."""
    sig = {
        "recommended_side": "YES",
        "kelly_size":       1.0,
        "liquidity_usd":    10000,
        "accepting_orders": False,   # ← paused
        "enable_order_book": True,
        "city":             "Chicago",
        "date":             "2026-04-09",
        "event_id":         "ev1",
        "lat":              41.88,
        "lon":              -87.63,
    }
    passed, failures = risk.run_pre_checks(sig, bankroll=200.0)
    assert passed is False
    assert any("acceptingOrders=False" in f for f in failures), (
        f"expected acceptingOrders failure in {failures!r}"
    )
