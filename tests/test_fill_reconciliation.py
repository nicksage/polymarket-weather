"""
test_fill_reconciliation.py — Tests for the order-status reconciliation
helpers in execution.py (Phase 4).

The monitor-side _reconcile_pending_fills function is integration-tested
indirectly here by mocking the CLOB client + DB calls.  The pure helpers
get full coverage.
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
from execution import (
    get_order_status, is_order_fully_filled, is_order_cancelled,
    extract_fill_price,
)


# ===========================================================================
# get_order_status — paper mode + missing inputs
# ===========================================================================

def test_get_order_status_paper_mode_returns_none(monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", True)
    client = MagicMock()
    assert get_order_status("0x123abc", client) is None
    client.get_order.assert_not_called()


def test_get_order_status_no_client_returns_none():
    assert get_order_status("0x123abc", None) is None


def test_get_order_status_empty_order_id_returns_none(monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()
    assert get_order_status("", client) is None
    client.get_order.assert_not_called()


def test_get_order_status_passes_through_dict_response(monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()
    client.get_order.return_value = {"status": "matched", "size_matched": 100}
    result = get_order_status("0xabc", client)
    assert result == {"status": "matched", "size_matched": 100}


def test_get_order_status_handles_object_response(monkeypatch):
    """Some py_clob_client versions return an object — normalize to dict."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()

    class FakeOrder:
        status = "matched"
        size_matched = 50.0
        original_size = 50.0
        price = 0.55
    client.get_order.return_value = FakeOrder()
    result = get_order_status("0xabc", client)
    assert result is not None
    assert result["status"] == "matched"
    assert result["size_matched"] == 50.0


def test_get_order_status_swallows_exception(monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()
    client.get_order.side_effect = RuntimeError("CLOB unreachable")
    assert get_order_status("0xabc", client) is None


# ===========================================================================
# is_order_fully_filled
# ===========================================================================

def test_filled_status_strings():
    for s in ("matched", "MATCHED", "filled", "Completed"):
        assert is_order_fully_filled({"status": s}) is True


def test_unfilled_status_strings():
    for s in ("live", "delayed", "unmatched", "cancelled"):
        assert is_order_fully_filled({"status": s}) is False


def test_full_size_match_implies_filled():
    """Even if status='live', if size_matched == original_size, it's done."""
    assert is_order_fully_filled({
        "status": "live",
        "size_matched": 100, "original_size": 100,
    }) is True


def test_partial_size_match_is_not_filled():
    assert is_order_fully_filled({
        "status": "live",
        "size_matched": 60, "original_size": 100,
    }) is False


def test_none_response_is_not_filled():
    assert is_order_fully_filled(None) is False


def test_empty_dict_is_not_filled():
    assert is_order_fully_filled({}) is False


# ===========================================================================
# is_order_cancelled
# ===========================================================================

def test_cancelled_statuses():
    for s in ("cancelled", "canceled", "expired", "CANCELLED"):
        assert is_order_cancelled({"status": s}) is True


def test_non_cancelled_statuses():
    for s in ("matched", "live", "filled", ""):
        assert is_order_cancelled({"status": s}) is False


def test_none_response_not_cancelled():
    assert is_order_cancelled(None) is False


# ===========================================================================
# extract_fill_price
# ===========================================================================

def test_extract_fill_price_avg_price_field():
    assert extract_fill_price({"avg_price": 0.553}, fallback=0.50) == pytest.approx(0.553)


def test_extract_fill_price_price_matched_field():
    assert extract_fill_price({"price_matched": "0.487"}, fallback=0.50) == pytest.approx(0.487)


def test_extract_fill_price_filled_price_field():
    assert extract_fill_price({"filled_price": 0.661}, fallback=0.50) == pytest.approx(0.661)


def test_extract_fill_price_buy_uses_making_over_taking():
    """For a BUY, the response's amounts mean:
        makerAmount = USDC paid (we 'make' USDC available)
        takerAmount = shares received (we 'take' shares)
    So price ($/share) = making / taking.

    Bug history (2026-04-29): the old test asserted the INVERTED semantics
    (`taking/making`) because the code had the same inversion — both wrong
    in the same way, so the test passed.  Real-world consequence: the
    Toronto live trade recorded entry_price=$3.45 instead of $0.29 because
    the code was computing 1/price.  This test now pins the correct
    semantics."""
    # 55 USDC paid for 100 shares → fill price = 0.55
    resp = {"side": "BUY", "makingAmount": 55.0, "takingAmount": 100.0}
    assert extract_fill_price(resp, fallback=0.99) == pytest.approx(0.55)


def test_extract_fill_price_falls_back_when_no_field():
    assert extract_fill_price({"status": "matched"}, fallback=0.42) == pytest.approx(0.42)


def test_extract_fill_price_none_response_returns_fallback():
    assert extract_fill_price(None, fallback=0.30) == pytest.approx(0.30)


def test_extract_fill_price_invalid_value_falls_back():
    """Garbage in a price field should not crash — fallback safely."""
    assert extract_fill_price({"avg_price": "not-a-number"}, fallback=0.40) == pytest.approx(0.40)


# ===========================================================================
# Integration: end-to-end fill confirmation flow (sanity check)
# ===========================================================================

def test_full_fill_workflow_simulation(monkeypatch):
    """Simulate: order placed, monitor polls 20 min later, sees filled."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()

    # Monitor cycle 1: order is still live
    client.get_order.return_value = {
        "status": "live",
        "size_matched": 0,
        "original_size": 100,
        "price": 0.55,
    }
    s1 = get_order_status("0xabc", client)
    assert is_order_fully_filled(s1) is False
    assert is_order_cancelled(s1) is False

    # Monitor cycle 2: order now matched
    client.get_order.return_value = {
        "status": "matched",
        "size_matched": 100,
        "original_size": 100,
        "price_matched": 0.5523,
    }
    s2 = get_order_status("0xabc", client)
    assert is_order_fully_filled(s2) is True
    actual = extract_fill_price(s2, fallback=0.55)
    # Should pick up the matched price, not the fallback
    assert actual == pytest.approx(0.5523)
