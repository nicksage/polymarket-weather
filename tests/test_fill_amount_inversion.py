"""
test_fill_amount_inversion.py — Regression tests for the maker/taker
amount inversion bug + partial fill handling.

The bug story (2026-04-29)
---------------------------
A live BUY at limit $0.30 partially filled 7 shares at $0.29, but our
DB recorded entry_price=$3.45 and shares=2.03 — both INVERTED.  Then
the cancel pass nuked the position entirely, losing the 7 real on-chain
shares from our books.

Three things were wrong:
  1. execute_signal computed `fill_price = takingAmount / makingAmount`
     which is shares/USDC = 1/price for a BUY (the actual price is
     `makingAmount / takingAmount`).  Same field also used for shares,
     storing USDC instead.
  2. extract_fill_price had the same inversion AND ignored the order's
     explicit `price` field which is the most reliable signal.
  3. _reconcile_pending_fills only handled fully-filled OR fully-cancelled
     orders — partial fills (status='canceled' with size_matched > 0)
     fell through and the matched shares disappeared.

These tests pin all three fixes.
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
import monitor


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


# ===========================================================================
# Bug 1: execute_signal fill_price + fill_shares inversion
# ===========================================================================

def test_execute_signal_records_correct_price_and_shares_on_partial_match(
    temp_db, monkeypatch
):
    """Reproduces the Toronto bug: $10 buy at limit $0.30, only 7 shares
    matched at $0.29.  Polymarket returns:
        makerAmount = 2.03  (USDC paid)
        takerAmount = 7     (shares received)
    Position should record entry_price=0.29, shares=7, size_usdc=$2.03 —
    NOT entry_price=3.45 and shares=2.03 (the inverted bug)."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    client = MagicMock()
    # Simulate what Polymarket's CLOB returns for a partial match
    client.create_and_post_order.return_value = {
        "success":      True,
        "orderID":      "0xtoronto",
        "status":       "matched",
        "makingAmount": "2.03",      # USDC paid for the partial
        "takingAmount": "7",          # shares received
    }

    sig = {
        "contract_id":      "0xabc",
        "recommended_side": "YES",
        "kelly_size":       10.0,
        "yes_token_id":     "tok_yes",
        "no_token_id":      "tok_no",
        "yes_price":        0.30,
        "market_p":         0.30,
        "city":             "Toronto",
        "date":             "2026-05-01",
        "scan_timestamp":   "2026-04-29T22:00:00+00:00",
    }
    result = execution.execute_signal(sig, client=client)
    assert result["status"] == "placed"
    pid = result["position_id"]

    import sqlite3 as _s
    conn = _s.connect(temp_db); conn.row_factory = _s.Row
    pos = dict(conn.execute(
        "SELECT entry_price, shares, size_usdc FROM positions WHERE id=?", (pid,)
    ).fetchone())
    conn.close()

    # Critical assertions — these would FAIL under the old inverted code
    assert pos["entry_price"] == pytest.approx(0.29), (
        f"entry_price should be 0.29 (USDC/share = making/taking), "
        f"got {pos['entry_price']} — likely regressed to taking/making (= 1/price)"
    )
    assert pos["shares"] == pytest.approx(7.0), (
        f"shares should be 7 (= takerAmount), got {pos['shares']} — "
        f"likely regressed to using makerAmount (= USDC) as shares"
    )


def test_execute_signal_fallback_when_response_amounts_empty(temp_db, monkeypatch):
    """When Polymarket returns 'live' (resting on book), takingAmount and
    makingAmount come back empty.  Code must fall back to limit_price + intended shares."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success":      True,
        "orderID":      "0xresting",
        "status":       "live",
        "makingAmount": "",   # empty for resting orders
        "takingAmount": "",
    }
    sig = {
        "contract_id": "0xabc", "recommended_side": "YES",
        "kelly_size": 10.0, "yes_token_id": "tok_yes", "no_token_id": "tok_no",
        "yes_price": 0.45, "market_p": 0.45,
        "city": "Wuhan", "date": "2026-05-01",
        "scan_timestamp": "2026-04-29T22:00:00+00:00",
    }
    result = execution.execute_signal(sig, client=client)
    pid = result["position_id"]
    import sqlite3 as _s
    conn = _s.connect(temp_db); conn.row_factory = _s.Row
    pos = dict(conn.execute(
        "SELECT entry_price, shares FROM positions WHERE id=?", (pid,)
    ).fetchone())
    conn.close()
    # actual_entry = limit_price = round(min(0.45 * 1.005, 0.99), 4) = 0.4523
    assert pos["entry_price"] == pytest.approx(0.4523, abs=1e-3)
    # Shares = size_usdc / limit_price (the price the order will fill at).
    # 10 / 0.4523 ≈ 22.11.  (Previously the code used entry_price for this
    # calc, which gave a slightly wrong 22.22 — fixed alongside the
    # ask-depth cap refactor on 2026-04-30.)
    assert pos["shares"] == pytest.approx(22.11, abs=0.1)


# ===========================================================================
# Bug 2: extract_fill_price inversion + ignoring explicit price field
# ===========================================================================

def test_extract_fill_price_prefers_explicit_price_field():
    """v2 get_order returns `price` as the actual matched price.  Use it
    directly — most reliable, no inversion concerns."""
    status = {
        "id": "0x", "side": "BUY", "price": "0.29",
        "size_matched": "7", "original_size": "35",
        # Ratios that would give a DIFFERENT answer if used:
        "makingAmount": "2.03", "takingAmount": "7",
    }
    # All of these would compute different values; explicit price wins
    assert execution.extract_fill_price(status, fallback=0.30) == pytest.approx(0.29)


def test_extract_fill_price_buy_ratio_correct_when_no_explicit_price():
    """Without explicit price, BUY uses making/taking (USDC/shares)."""
    status = {
        "side": "BUY",
        "makingAmount": "2.03",  # USDC paid
        "takingAmount": "7",     # shares received
        # no `price` field
    }
    # price = making/taking = 2.03 / 7 ≈ 0.29
    assert execution.extract_fill_price(status, fallback=0.99) == pytest.approx(0.29, abs=0.001)


def test_extract_fill_price_sell_ratio_inverted_correctly():
    """For SELL, the maker/taker semantics flip: maker=shares offered,
    taker=USDC received → price = taker/maker."""
    status = {
        "side": "SELL",
        "makingAmount": "10",    # shares sold
        "takingAmount": "5.5",   # USDC received
    }
    # price = taking/making = 5.5/10 = 0.55
    assert execution.extract_fill_price(status, fallback=0.99) == pytest.approx(0.55)


def test_extract_fill_price_unknown_side_defaults_to_buy_semantics():
    """Defensive: if `side` is missing, treat as BUY rather than inverting."""
    status = {"makingAmount": "2.03", "takingAmount": "7"}  # no side field
    # Default to BUY semantics (making/taking)
    assert execution.extract_fill_price(status, fallback=0.99) == pytest.approx(0.29, abs=0.001)


def test_extract_fill_price_no_data_returns_fallback():
    assert execution.extract_fill_price(None, fallback=0.42) == 0.42
    assert execution.extract_fill_price({}, fallback=0.42) == 0.42


def test_extract_fill_price_zero_price_skipped():
    """A zero/empty price field shouldn't be treated as a real value."""
    status = {"price": "0", "side": "BUY",
              "makingAmount": "2.03", "takingAmount": "7"}
    # Should fall through to ratio = 0.29, not return 0
    assert execution.extract_fill_price(status, fallback=0.99) == pytest.approx(0.29, abs=0.001)


# ===========================================================================
# Bug 3: update_position_fill recomputes size_usdc
# ===========================================================================

def test_update_position_fill_updates_size_usdc(temp_db):
    """The actual cost of a partial fill differs from the originally
    intended size.  size_usdc must reflect what was actually paid, not
    the intent — otherwise exposure caps over-count."""
    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-01-01T00:00:00",
        shares=33.33, yes_token_id="tok_yes",
        is_paper=0, fill_status="pending",
    )
    # Partial fill: only 7 shares filled at 0.29
    db.update_position_fill(
        position_id=pid, fill_status="filled", shares=7.0, entry_price=0.29
    )
    import sqlite3 as _s
    conn = _s.connect(temp_db); conn.row_factory = _s.Row
    pos = dict(conn.execute(
        "SELECT shares, entry_price, size_usdc FROM positions WHERE id=?", (pid,)
    ).fetchone())
    conn.close()
    assert pos["shares"] == pytest.approx(7.0)
    assert pos["entry_price"] == pytest.approx(0.29)
    # size_usdc should be the ACTUAL cost (7 × 0.29 = 2.03), not 10
    assert pos["size_usdc"] == pytest.approx(2.03, abs=0.01)


# ===========================================================================
# Bug 4: _reconcile_pending_fills handles partial fills on cancelled orders
# ===========================================================================

def test_reconcile_records_partial_fill_on_cancelled_buy(temp_db, monkeypatch):
    """The Toronto scenario end-to-end: BUY placed, partially filled,
    cancelled by monitor sweep (or operator).  The matched shares MUST
    land in the DB — not silently lost."""
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)
    # Seed a pending live BUY
    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-01-01T00:00:00",
        shares=33.33, yes_token_id="tok_yes",
        order_id="0xtoronto",
        is_paper=0, fill_status="pending",
    )

    # Mock CLOB to return CANCELED order with 7 shares matched at 0.29
    client = MagicMock()
    monkeypatch.setattr(execution, "get_order_status",
        lambda order_id, c: {
            "id": order_id,
            "status": "CANCELED",      # cancelled, but...
            "side": "BUY",
            "original_size": "33.33",
            "size_matched": "7",        # ...with a partial fill!
            "price": "0.29",
        }
    )

    buys, _, _ = monitor._reconcile_pending_fills(client)
    assert buys == 1, "should have recorded the partial fill"

    import sqlite3 as _s
    conn = _s.connect(temp_db); conn.row_factory = _s.Row
    pos = dict(conn.execute(
        "SELECT fill_status, shares, entry_price, size_usdc FROM positions WHERE id=?",
        (pid,)
    ).fetchone())
    conn.close()
    assert pos["fill_status"] == "filled", (
        f"partial fill should mark position 'filled' (with the actual matched "
        f"shares), got fill_status={pos['fill_status']!r}.  Bug: cancelled-with-"
        f"partial-fill silently lost the on-chain shares from the DB."
    )
    assert pos["shares"] == pytest.approx(7.0)
    assert pos["entry_price"] == pytest.approx(0.29)
    assert pos["size_usdc"] == pytest.approx(2.03, abs=0.01)


def test_reconcile_does_not_record_zero_match_cancellation(temp_db, monkeypatch):
    """Inverse: a TRULY cancelled order (size_matched=0) must not be
    accidentally recorded as a fill."""
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)
    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-01-01T00:00:00",
        shares=33.33, yes_token_id="tok_yes",
        order_id="0xnoFill",
        is_paper=0, fill_status="pending",
    )
    client = MagicMock()
    monkeypatch.setattr(execution, "get_order_status",
        lambda order_id, c: {
            "id": order_id, "status": "CANCELED",
            "side": "BUY", "original_size": "33.33",
            "size_matched": "0", "price": "0.30",
        }
    )

    buys, _, _ = monitor._reconcile_pending_fills(client)
    assert buys == 0, "no fill should be recorded for size_matched=0"


def test_reconcile_records_full_fill_normally(temp_db, monkeypatch):
    """Sanity check: the happy path (fully matched, not cancelled) still works."""
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)
    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-01-01T00:00:00",
        shares=33.33, yes_token_id="tok_yes",
        order_id="0xfull",
        is_paper=0, fill_status="pending",
    )
    client = MagicMock()
    monkeypatch.setattr(execution, "get_order_status",
        lambda order_id, c: {
            "id": order_id, "status": "MATCHED",
            "side": "BUY", "original_size": "33.33",
            "size_matched": "33.33", "price": "0.30",
        }
    )

    buys, _, _ = monitor._reconcile_pending_fills(client)
    assert buys == 1
    import sqlite3 as _s
    conn = _s.connect(temp_db); conn.row_factory = _s.Row
    pos = dict(conn.execute(
        "SELECT fill_status, shares FROM positions WHERE id=?", (pid,)
    ).fetchone())
    conn.close()
    assert pos["fill_status"] == "filled"
    assert pos["shares"] == pytest.approx(33.33, abs=0.01)
