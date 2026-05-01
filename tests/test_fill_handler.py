"""
test_fill_handler.py — Tests for the user-channel WS fill handler (Phase 9).

Covers:
  * Lifecycle helpers (update_position_trade_status, get_position_by_order_id,
    classify_position_role)
  * apply_trade_event for entry / exit / topup roles
  * Lifecycle progression: matched → mined → confirmed
  * Idempotency: same CONFIRMED event applied twice = one fill
  * Regression: confirmed-then-matched is ignored (no downgrade)
  * FAILED handling for entry / topup / exit roles
  * extract_our_order_id (taker, maker, fallback)
  * apply_order_event (PLACEMENT, CANCELLATION)
  * user_ws._dispatch routes correctly
  * user_ws._build_subscribe_payload shape
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import fill_handler


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _seed(
    *,
    order_id: str = "0xord_entry",
    contract_id: str = "0xabc",
    yes_token_id: str = "tok_yes",
    side: str = "YES",
    size_usdc: float = 50.0,
    entry_price: float = 0.50,
    shares: float = 100.0,
    fill_status: str = "pending",
) -> int:
    return db.insert_position(
        contract_id  = contract_id,
        side         = side,
        size_usdc    = size_usdc,
        entry_price  = entry_price,
        entry_time   = "2026-01-01T00:00:00",
        order_id     = order_id,
        shares       = shares,
        yes_token_id = yes_token_id,
        is_paper     = 0,
        fill_status  = fill_status,
    )


# ===========================================================================
# Lifecycle helper: update_position_trade_status
# ===========================================================================

def test_update_trade_status_advances(temp_db):
    pid = _seed()
    assert db.update_position_trade_status(pid, "matched", side="entry") is True
    assert db.update_position_trade_status(pid, "mined",   side="entry") is True
    assert db.update_position_trade_status(pid, "confirmed", side="entry") is True

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT trade_status FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == "confirmed"


def test_update_trade_status_refuses_regression(temp_db):
    """A delayed MATCHED event after CONFIRMED must NOT regress."""
    pid = _seed()
    db.update_position_trade_status(pid, "confirmed", side="entry")
    advanced = db.update_position_trade_status(pid, "matched", side="entry")
    assert advanced is False

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT trade_status FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == "confirmed"


def test_update_trade_status_idempotent_at_same_rank(temp_db):
    """Same status applied twice — second call is no-op (returns False)."""
    pid = _seed()
    db.update_position_trade_status(pid, "confirmed", side="entry")
    assert db.update_position_trade_status(pid, "confirmed", side="entry") is False


def test_update_trade_status_exit_side_independent(temp_db):
    """entry and exit sides are tracked independently."""
    pid = _seed()
    db.update_position_trade_status(pid, "confirmed", side="entry")
    db.update_position_trade_status(pid, "matched",   side="exit")

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT trade_status, exit_trade_status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == "confirmed"
    assert row[1] == "matched"


def test_update_trade_status_unknown_status_ignored(temp_db):
    pid = _seed()
    assert db.update_position_trade_status(pid, "garbage", side="entry") is False


# ===========================================================================
# Lifecycle helper: get_position_by_order_id + classify_position_role
# ===========================================================================

def test_get_position_by_entry_order_id(temp_db):
    pid = _seed(order_id="0xord_entry")
    row = db.get_position_by_order_id("0xord_entry")
    assert row is not None and row["id"] == pid
    assert db.classify_position_role(row, "0xord_entry") == "entry"


def test_get_position_by_exit_order_id(temp_db):
    pid = _seed(order_id="0xord_entry")
    db.update_position_exit_pending(
        position_id=pid, exit_order_id="0xord_exit",
        exit_intended_price=0.55, exit_retry_count=0,
    )
    row = db.get_position_by_order_id("0xord_exit")
    assert row is not None and row["id"] == pid
    assert db.classify_position_role(row, "0xord_exit") == "exit"


def test_get_position_by_topup_order_id(temp_db):
    pid = _seed(order_id="0xord_entry")
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord_topup",
        amount_usdc=20.0, intended_price=0.51,
    )
    row = db.get_position_by_order_id("0xord_topup")
    assert row is not None and row["id"] == pid
    assert db.classify_position_role(row, "0xord_topup") == "topup"


def test_get_position_by_unknown_order_id_returns_none(temp_db):
    _seed()
    assert db.get_position_by_order_id("0xnonexistent") is None


# ===========================================================================
# extract_our_order_id
# ===========================================================================

def test_extract_order_id_taker_with_wallet():
    event = {
        "taker_order_id": "0xtaker",
        "owner": "0xMyWallet",
        "maker_orders": [{"order_id": "0xmaker", "owner": "0xother"}],
    }
    assert fill_handler.extract_our_order_id(event, "0xmywallet") == "0xtaker"


def test_extract_order_id_maker_with_wallet():
    """If we're the maker, return the matching maker_orders[].order_id."""
    event = {
        "taker_order_id": "0xtaker_other",
        "owner": "0xother",
        "maker_orders": [
            {"order_id": "0xother_maker", "owner": "0xother2"},
            {"order_id": "0xmy_maker",    "owner": "0xMyWallet"},
        ],
    }
    assert fill_handler.extract_our_order_id(event, "0xmywallet") == "0xmy_maker"


def test_extract_order_id_no_wallet_falls_back_to_taker():
    event = {
        "taker_order_id": "0xtaker",
        "maker_orders": [{"order_id": "0xmaker", "owner": "0xother"}],
    }
    assert fill_handler.extract_our_order_id(event, None) == "0xtaker"


def test_extract_order_id_empty_event_returns_none():
    assert fill_handler.extract_our_order_id({}, "0xany") is None


# ===========================================================================
# apply_trade_event — entry side
# ===========================================================================

def test_apply_trade_confirmed_entry_writes_fill(temp_db):
    pid = _seed(order_id="0xord_entry", shares=100.0, entry_price=0.50,
                fill_status="pending")
    event = {
        "id": "trade1", "status": "confirmed",
        "taker_order_id": "0xord_entry",
        "size": 100.0, "price": 0.501,
    }
    result = fill_handler.apply_trade_event(event)
    assert result["action"] == "filled"
    assert result["role"] == "entry"

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT fill_status, entry_price, shares, trade_status FROM positions WHERE id=?",
        (pid,)).fetchone()
    conn.close()
    assert row[0] == "filled"
    assert row[1] == pytest.approx(0.501)
    assert row[2] == pytest.approx(100.0)
    assert row[3] == "confirmed"


def test_apply_trade_matched_does_not_fill(temp_db):
    """MATCHED is NOT terminal — must not flip fill_status."""
    pid = _seed(fill_status="pending")
    event = {"id": "t1", "status": "matched", "taker_order_id": "0xord_entry",
             "size": 100.0, "price": 0.50}
    result = fill_handler.apply_trade_event(event)
    assert result["action"] == "lifecycle_advanced"
    assert result["to"] == "matched"

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT fill_status, trade_status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == "pending"
    assert row[1] == "matched"


def test_apply_trade_lifecycle_progression(temp_db):
    pid = _seed(fill_status="pending")
    for status in ("matched", "mined", "confirmed"):
        fill_handler.apply_trade_event({
            "id": f"t_{status}", "status": status,
            "taker_order_id": "0xord_entry", "size": 100.0, "price": 0.50,
        })
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT fill_status, trade_status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == "filled"
    assert row[1] == "confirmed"


def test_apply_trade_confirmed_idempotent(temp_db):
    """Same trade event (same event_id) applied twice must be a no-op.
    Dedup is now via processed_trade_events table on event_id, so the
    second call returns 'ignored_duplicate_event'.
    """
    pid = _seed(fill_status="pending")
    e = {"id": "t1", "status": "confirmed", "taker_order_id": "0xord_entry",
         "size": 100.0, "price": 0.50}
    r1 = fill_handler.apply_trade_event(e)
    r2 = fill_handler.apply_trade_event(e)
    assert r1["action"] == "filled"
    assert r2["action"] == "ignored_duplicate_event"


def test_apply_trade_confirmed_then_matched_ignored(temp_db):
    """Late MATCHED arriving after CONFIRMED is dropped — but with a
    distinct event_id, so it passes the dedup and is rejected by the
    lifecycle regression check instead."""
    pid = _seed(fill_status="pending")
    fill_handler.apply_trade_event({
        "id": "t1", "status": "confirmed", "taker_order_id": "0xord_entry",
        "size": 100.0, "price": 0.50,
    })
    r = fill_handler.apply_trade_event({
        "id": "t2", "status": "matched", "taker_order_id": "0xord_entry",
        "size": 100.0, "price": 0.50,
    })
    assert r["action"] == "lifecycle_regression"


# ===========================================================================
# Multi-chunk fill accumulation (the fix for the share-drift bug)
#
# A single Polymarket limit order frequently matches against multiple
# resting asks, producing N trade events with N distinct event_ids but
# the SAME order_id.  Each chunk must apply its own fill.  Pre-fix,
# the per-position trade_status gate dropped chunks 2..N silently.
# ===========================================================================

def test_apply_trade_entry_chunked_fill_accumulates(temp_db):
    """Three CONFIRMED events for the same entry order with distinct
    event_ids accumulate shares into the position row (not just chunk 1).
    """
    pid = _seed(order_id="0xord_entry", fill_status="pending")
    chunks = [
        {"id": "t1", "size": 5.0,  "price": 0.34},
        {"id": "t2", "size": 7.5,  "price": 0.35},
        {"id": "t3", "size": 12.5, "price": 0.36},
    ]
    for c in chunks:
        result = fill_handler.apply_trade_event({
            "status":         "confirmed",
            "taker_order_id": "0xord_entry",
            **c,
        })
        assert result["action"] == "filled", c

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, size_usdc, entry_price, fill_status "
        "FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    expected_shares = 5.0 + 7.5 + 12.5
    expected_usdc = 5.0 * 0.34 + 7.5 * 0.35 + 12.5 * 0.36
    assert row[0] == pytest.approx(expected_shares)
    assert row[1] == pytest.approx(expected_usdc)
    # Weighted-average cost basis
    assert row[2] == pytest.approx(expected_usdc / expected_shares)
    assert row[3] == "filled"


def test_apply_trade_entry_chunked_dedup_per_event_id(temp_db):
    """Re-emitting any single chunk's event must NOT double-credit it
    even when other chunks are interleaved (the WS+REST both-paths
    redelivery scenario)."""
    pid = _seed(order_id="0xord_entry", fill_status="pending")
    base = {"taker_order_id": "0xord_entry", "status": "confirmed",
            "size": 10.0, "price": 0.30}

    fill_handler.apply_trade_event({**base, "id": "tA"})
    # Polymarket re-emits 'tA' (Polymarket at-least-once) and we also see
    # it via the REST safety net — both must be no-ops.
    r2 = fill_handler.apply_trade_event({**base, "id": "tA"})
    r3 = fill_handler.apply_trade_event({**base, "id": "tA"})
    fill_handler.apply_trade_event({**base, "id": "tB", "size": 4.0})

    assert r2["action"] == "ignored_duplicate_event"
    assert r3["action"] == "ignored_duplicate_event"

    conn = sqlite3.connect(temp_db)
    shares = conn.execute(
        "SELECT shares FROM positions WHERE id=?", (pid,)
    ).fetchone()[0]
    conn.close()
    # Only the unique chunks count: 10 (tA) + 4 (tB) = 14
    assert shares == pytest.approx(14.0)


def test_apply_trade_topup_chunked_fill_accumulates(temp_db):
    """Same mechanism applies to a topup that fills in chunks: the
    position's existing shares grow by the sum of all chunks, and the
    pending_topup_* fields stay set until the LAST chunk lands so
    intermediate chunks can still be routed back to the position.
    """
    # Start with a filled entry of 20 shares.
    pid = _seed(order_id="0xord_entry", fill_status="filled",
                shares=20.0, entry_price=0.30, size_usdc=6.0)
    # Stamp a pending topup so the order_id classifier returns 'topup'.
    db.update_position_topup_pending(
        position_id    = pid,
        order_id       = "0xord_topup",
        amount_usdc    = 5.0,
        intended_price = 0.32,
    )
    # Production also inserts a ledger row when the topup is placed
    # (execution.execute_topup, post-CLOB POST).  Mirror that — the
    # ledger's intended_shares is what tells the fill handler when the
    # LAST chunk has landed (and therefore when to clear pending_topup_*).
    db.insert_position_order(
        position_id     = pid,
        order_id        = "0xord_topup",
        role            = "topup",
        intended_usdc   = 5.0,
        intended_shares = 15.0,            # 4 + 6 + 5 sums to 15
        limit_price     = 0.32,
        status          = "pending",
    )

    chunks = [
        {"id": "tu1", "size": 4.0, "price": 0.31},
        {"id": "tu2", "size": 6.0, "price": 0.32},
        {"id": "tu3", "size": 5.0, "price": 0.33},
    ]
    pending_after_each = []
    for c in chunks:
        result = fill_handler.apply_trade_event({
            "status":         "confirmed",
            "taker_order_id": "0xord_topup",
            **c,
        })
        assert result["action"] == "filled"
        assert result["role"] == "topup"
        with sqlite3.connect(temp_db) as _c:
            pending_after_each.append(_c.execute(
                "SELECT pending_topup_order_id FROM positions WHERE id=?",
                (pid,),
            ).fetchone()[0])

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, pending_topup_order_id FROM positions WHERE id=?",
        (pid,),
    ).fetchone()
    conn.close()
    # 20 (entry) + 4 + 6 + 5 = 35 shares accumulated
    assert row[0] == pytest.approx(35.0)
    # Pending stays set after chunks 1 + 2 (so they route correctly), and
    # only clears after chunk 3 brings cumulative filled >= intended.
    assert pending_after_each[0] == "0xord_topup"   # after chunk 1
    assert pending_after_each[1] == "0xord_topup"   # after chunk 2
    assert pending_after_each[2] is None             # after chunk 3 (full)
    assert row[1] is None


def test_mark_event_processed_dedup_directly(temp_db):
    """Direct test of the dedup helper: first call returns True, second
    returns False.  Empty event_ids return True without persisting."""
    assert db.mark_event_processed("evt_abc") is True
    assert db.mark_event_processed("evt_abc") is False
    assert db.mark_event_processed("evt_xyz") is True
    # Empty / None: no persistence, callers can't dedup what they don't
    # have an id for.
    assert db.mark_event_processed("") is True
    assert db.mark_event_processed(None) is True


def test_add_position_entry_fill_first_chunk_replaces(temp_db):
    """First chunk lands while fill_status='pending' — replaces the
    intended-shares seed from insert_position with the actual fill."""
    pid = _seed(shares=100.0, size_usdc=50.0, entry_price=0.50,
                fill_status="pending")
    db.add_position_entry_fill(pid, added_shares=30.0, fill_price=0.40)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, size_usdc, entry_price, fill_status "
        "FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(30.0)        # replaced, not 100+30
    assert row[1] == pytest.approx(30.0 * 0.40)  # 12.00
    assert row[2] == pytest.approx(0.40)
    assert row[3] == "filled"


def test_add_position_entry_fill_subsequent_chunk_accumulates(temp_db):
    """Once fill_status='filled', further chunks add to the running
    totals and recompute weighted-average entry price."""
    pid = _seed(shares=0.0, size_usdc=0.0, fill_status="pending")
    db.add_position_entry_fill(pid, added_shares=10.0, fill_price=0.30)
    db.add_position_entry_fill(pid, added_shares=20.0, fill_price=0.40)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, size_usdc, entry_price FROM positions WHERE id=?",
        (pid,),
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(30.0)
    expected_usdc = 10.0 * 0.30 + 20.0 * 0.40
    assert row[1] == pytest.approx(expected_usdc)
    assert row[2] == pytest.approx(expected_usdc / 30.0)  # weighted avg


def test_add_position_entry_fill_skips_cancelled(temp_db):
    """A fill landing on a cancelled position is a no-op (not an error)."""
    pid = _seed(shares=10.0, size_usdc=5.0, fill_status="cancelled")
    db.add_position_entry_fill(pid, added_shares=10.0, fill_price=0.50)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, fill_status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(10.0)         # unchanged
    assert row[1] == "cancelled"                  # unchanged


# ===========================================================================
# Multi-chunk exit accumulation (Layer 3 fix)
# ===========================================================================

def _seed_exiting(*, entry_price=0.50, shares=100.0, size_usdc=50.0,
                  entry_fees=0.0):
    """Seed a filled position transitioned to 'exiting' with an exit_order_id."""
    pid = _seed(
        order_id="0xord_entry", entry_price=entry_price,
        shares=shares, size_usdc=size_usdc, fill_status="filled",
    )
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute(
        "UPDATE positions SET fill_status='filled', entry_fees=? WHERE id=?",
        (entry_fees, pid),
    )
    conn.commit(); conn.close()
    db.update_position_exit_pending(
        position_id=pid, exit_order_id="0xord_exit",
        exit_intended_price=0.60, exit_retry_count=0,
    )
    return pid


def test_add_position_exit_fill_single_chunk_closes(temp_db):
    """Single chunk that consumes all shares closes the position with
    correct pnl + weighted-avg exit price."""
    pid = _seed_exiting(entry_price=0.50, shares=100.0, size_usdc=50.0)
    res = db.add_position_exit_fill(pid, sold_shares=100.0,
                                    fill_price=0.58, fee_usdc=0.0)
    assert res["is_complete"] is True
    assert res["gross_pnl"] == pytest.approx(8.0)  # 58 - 50, no fees
    assert res["net_pnl"]   == pytest.approx(8.0)
    assert res["actual_exit_price"] == pytest.approx(0.58)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status, exit_proceeds_usdc, pnl FROM positions WHERE id=?",
        (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.0)
    assert row[1] == "closed"
    assert row[2] == pytest.approx(58.0)


def test_add_position_exit_fill_two_chunks_close_only_on_second(temp_db):
    """Two chunks: first leaves position 'exiting' with reduced shares;
    second consumes the rest and closes with weighted-avg pnl."""
    pid = _seed_exiting(entry_price=0.50, shares=100.0, size_usdc=50.0)

    # Chunk 1: sell 60 shares at $0.55 → $33 proceeds, 40 shares left
    res1 = db.add_position_exit_fill(pid, sold_shares=60.0,
                                     fill_price=0.55, fee_usdc=0.0)
    assert res1["is_complete"] is False
    assert res1["shares_after"] == pytest.approx(40.0)
    assert res1["pnl"] is None

    # Position should still be 'exiting' (or whatever it was), shares=40
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status, exit_proceeds_usdc FROM positions WHERE id=?",
        (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(40.0)
    assert row[1] == "exiting"
    assert row[2] == pytest.approx(33.0)

    # Chunk 2: sell remaining 40 at $0.50 → +$20 proceeds, total $53
    res2 = db.add_position_exit_fill(pid, sold_shares=40.0,
                                     fill_price=0.50, fee_usdc=0.0)
    assert res2["is_complete"] is True
    # Total proceeds = $53, total cost = $50, no fees → gross = net = $3
    assert res2["gross_pnl"] == pytest.approx(3.0)
    assert res2["net_pnl"]   == pytest.approx(3.0)
    # Weighted avg exit price: (60*0.55 + 40*0.50) / 100 = 53 / 100 = $0.53
    assert res2["actual_exit_price"] == pytest.approx(0.53)


def test_add_position_exit_fill_includes_fees_in_pnl(temp_db):
    """Net pnl subtracts both entry_fees (already paid) and accumulated
    exit_fees (per-chunk)."""
    pid = _seed_exiting(entry_price=0.50, shares=100.0, size_usdc=50.0,
                       entry_fees=0.10)
    db.add_position_exit_fill(pid, sold_shares=50.0, fill_price=0.55,
                              fee_usdc=0.05)
    res = db.add_position_exit_fill(pid, sold_shares=50.0, fill_price=0.55,
                                    fee_usdc=0.05)
    # Proceeds = 100 * 0.55 = $55; cost = $50; entry_fees = $0.10;
    # exit_fees = $0.05 + $0.05 = $0.10
    # gross = $5; net = gross - 0.10 - 0.10 = $4.80
    assert res["gross_pnl"] == pytest.approx(5.00)
    assert res["net_pnl"]   == pytest.approx(4.80)


def test_add_position_exit_fill_partial_keeps_status(temp_db):
    """Partial sell that doesn't fully close MUST leave status='exiting'
    (so the ladder advancer keeps managing it)."""
    pid = _seed_exiting(shares=100.0)
    db.add_position_exit_fill(pid, sold_shares=20.0, fill_price=0.50,
                              fee_usdc=0.0)
    conn = sqlite3.connect(temp_db)
    status = conn.execute(
        "SELECT status FROM positions WHERE id=?", (pid,)
    ).fetchone()[0]
    conn.close()
    assert status == "exiting"


def test_add_position_exit_fill_tolerance_closes_near_zero(temp_db):
    """Residual shares ≤ 0.001 (rounding noise) trips the close."""
    pid = _seed_exiting(shares=100.0, size_usdc=50.0)
    db.add_position_exit_fill(pid, sold_shares=99.9999, fill_price=0.50,
                              fee_usdc=0.0)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    # Residual = 0.0001 < 0.001 tolerance → closed and shares zeroed
    assert row[0] == pytest.approx(0.0)
    assert row[1] == "closed"


def test_add_position_exit_fill_no_op_for_missing_position(temp_db):
    """Calling on a non-existent pid returns the no-op result and
    doesn't raise."""
    res = db.add_position_exit_fill(99999, sold_shares=10.0,
                                    fill_price=0.50, fee_usdc=0.0)
    assert res["is_complete"] is False
    assert res["shares_after"] == 0.0


def test_apply_trade_exit_chunked_via_fill_handler(temp_db):
    """End-to-end: 3 chunks of an exit order arriving via apply_trade_event
    accumulate correctly and only close on the last."""
    pid = _seed_exiting(entry_price=0.50, shares=100.0, size_usdc=50.0)
    base = {"taker_order_id": "0xord_exit", "status": "confirmed"}
    chunks = [
        {"id": "ec1", "size": 30.0, "price": 0.55},
        {"id": "ec2", "size": 30.0, "price": 0.54},
        {"id": "ec3", "size": 40.0, "price": 0.53},
    ]
    results = []
    for c in chunks:
        results.append(fill_handler.apply_trade_event({**base, **c}))

    # Only the last chunk should be exit_complete
    assert results[0]["exit_complete"] is False
    assert results[1]["exit_complete"] is False
    assert results[2]["exit_complete"] is True

    # Final state
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status, actual_exit_price, pnl, exit_proceeds_usdc "
        "FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    # Total proceeds: 30*0.55 + 30*0.54 + 40*0.53 = 16.5 + 16.2 + 21.2 = $53.9
    # pnl = 53.9 - 50 - 0 = $3.9; weighted avg = 53.9 / 100 = $0.539
    assert row[0] == pytest.approx(0.0)
    assert row[1] == "closed"
    assert row[2] == pytest.approx(0.539)
    assert row[3] == pytest.approx(3.9)
    assert row[4] == pytest.approx(53.9)


def test_apply_trade_exit_partial_then_cancelled_leaves_open(temp_db):
    """The user-reported scenario: exit fills partially, then the
    rest gets cancelled.  Position should reflect the partial sell
    (shares decremented) but stay 'exiting' for the ladder advancer."""
    pid = _seed_exiting(entry_price=0.35, shares=28.57, size_usdc=10.0)
    base = {"taker_order_id": "0xord_exit", "status": "confirmed"}
    # Two partial fills (the 26.15 / 28.57 case from the user's log)
    fill_handler.apply_trade_event({**base, "id": "p1", "size": 13.0, "price": 0.23})
    fill_handler.apply_trade_event({**base, "id": "p2", "size": 13.15, "price": 0.23})
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status, exit_proceeds_usdc FROM positions WHERE id=?",
        (pid,)
    ).fetchone()
    conn.close()
    # 28.57 - 26.15 = 2.42 shares remain → not closed
    assert row[0] == pytest.approx(2.42, abs=0.01)
    assert row[1] == "exiting"
    # Proceeds = 26.15 * 0.23 = $6.0145
    assert row[2] == pytest.approx(26.15 * 0.23, abs=0.01)


# ===========================================================================
# Layer 2: graceful "not enough balance" handler
# ===========================================================================

def test_handle_exit_balance_mismatch_chain_zero_closes(temp_db, monkeypatch):
    """When CLOB rejects with not-enough-balance and the Data API confirms
    the chain holds 0 shares, the position is closed via the recovery path."""
    pid = _seed_exiting(entry_price=0.35, shares=28.57, size_usdc=10.0)
    # Mock the Data API to return zero holdings for the position's token
    monkeypatch.setattr(
        "polymarket.get_data_api_positions",
        lambda *_a, **_k: [],   # no holdings
    )
    # Need a WALLET_ADDRESS for the helper to call out
    monkeypatch.setattr("config.WALLET_ADDRESS", "0xwallet")
    import execution
    res = execution._handle_exit_balance_mismatch(
        pid              = pid,
        contract_id      = "0xabc",
        token_id         = "tok_yes",
        intended_shares  = 28.57,
        limit_price      = 0.23,
        exit_reason      = "test_recovery",
    )
    assert res is not None
    assert res["status"] == "closed_via_balance_recovery"
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status, pnl FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.0)
    assert row[1] == "closed"
    # Proxy pnl ≈ (0.23 - 0.35) × 28.57 = -3.43
    assert row[2] == pytest.approx(-3.43, abs=0.05)


def test_handle_exit_balance_mismatch_chain_partial_resyncs(temp_db, monkeypatch):
    """When chain has SOME shares but fewer than we tried to sell, just
    sync the DB and let the next ladder cycle retry."""
    pid = _seed_exiting(entry_price=0.35, shares=28.57, size_usdc=10.0)
    monkeypatch.setattr(
        "polymarket.get_data_api_positions",
        lambda *_a, **_k: [{"asset": "tok_yes", "size": "10.5"}],
    )
    monkeypatch.setattr("config.WALLET_ADDRESS", "0xwallet")
    import execution
    res = execution._handle_exit_balance_mismatch(
        pid              = pid,
        contract_id      = "0xabc",
        token_id         = "tok_yes",
        intended_shares  = 28.57,
        limit_price      = 0.23,
        exit_reason      = "test_partial_resync",
    )
    assert res["status"] == "shares_resynced"
    assert res["new_shares"] == pytest.approx(10.5)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT shares, status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(10.5)
    # NOT closed — let next ladder cycle handle it
    assert row[1] == "exiting"


def test_handle_exit_balance_mismatch_recovery_failure_returns_none(temp_db, monkeypatch):
    """If the recovery itself fails (e.g., Data API down), return None
    so the caller falls through to the generic error path."""
    def boom(*a, **k):
        raise RuntimeError("data API down")
    monkeypatch.setattr("polymarket.get_data_api_positions", boom)
    monkeypatch.setattr("config.WALLET_ADDRESS", "0xwallet")
    import execution
    pid = _seed_exiting()
    res = execution._handle_exit_balance_mismatch(
        pid              = pid,
        contract_id      = "0xabc",
        token_id         = "tok_yes",
        intended_shares  = 28.57,
        limit_price      = 0.23,
        exit_reason      = "test_recovery_fail",
    )
    assert res is None


def test_apply_trade_no_position_match(temp_db):
    """Trade event for an order we don't track is silently ignored."""
    _seed()
    event = {"id": "t1", "status": "confirmed", "taker_order_id": "0xforeign",
             "size": 5.0, "price": 0.5}
    r = fill_handler.apply_trade_event(event)
    assert r["action"] == "ignored_no_position"


def test_apply_trade_unknown_status_ignored(temp_db):
    _seed()
    r = fill_handler.apply_trade_event({
        "id": "t1", "status": "WEIRDO", "taker_order_id": "0xord_entry",
    })
    assert r["action"] == "ignored_unknown_status"


def test_apply_trade_no_order_id_ignored(temp_db):
    _seed()
    r = fill_handler.apply_trade_event({
        "id": "t1", "status": "confirmed", "size": 100.0, "price": 0.5,
    })
    assert r["action"] == "ignored_no_order_id"


# ===========================================================================
# apply_trade_event — exit side
# ===========================================================================

def test_apply_trade_confirmed_exit_writes_fill_and_pnl(temp_db):
    pid = _seed(order_id="0xord_entry", entry_price=0.50, shares=100.0)
    # Mark as filled + transition to exiting
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE positions SET fill_status='filled' WHERE id=?", (pid,))
    conn.commit(); conn.close()
    db.update_position_exit_pending(
        position_id=pid, exit_order_id="0xord_exit",
        exit_intended_price=0.60, exit_retry_count=0,
    )

    event = {
        "id": "exit_trade", "status": "confirmed",
        "taker_order_id": "0xord_exit",
        "size": 100.0, "price": 0.58,  # actual fill below intended
    }
    r = fill_handler.apply_trade_event(event)
    assert r["action"] == "filled"
    assert r["role"] == "exit"
    assert r["exit_complete"] is True
    # New result schema (Layer 3 fix, 2026-04-30): per-chunk decrement
    # accumulates exit_proceeds_usdc.  Schema preserved from legacy:
    # pnl = gross (proceeds - cost), pnl_net = gross - fees.
    assert r["gross_pnl"] == pytest.approx(8.0)       # $58 - $50, no fees in this test
    assert r["net_pnl"]   == pytest.approx(8.0)       # same: zero fees
    assert r["avg_exit_price"] == pytest.approx(0.58)

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT status, actual_exit_price, pnl, pnl_net, exit_trade_status, shares "
        "FROM positions WHERE id=?",
        (pid,)).fetchone()
    conn.close()
    assert row[0] == "closed"
    assert row[1] == pytest.approx(0.58)
    assert row[2] == pytest.approx(8.0)               # pnl = gross
    assert row[3] == pytest.approx(8.0)               # pnl_net = net (same here)
    assert row[4] == "confirmed"
    assert row[5] == pytest.approx(0.0)               # shares decremented to 0


# ===========================================================================
# apply_trade_event — topup
# ===========================================================================

def test_apply_trade_confirmed_topup_merges(temp_db):
    pid = _seed(size_usdc=100.0, entry_price=0.50, shares=200.0)
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord_topup",
        amount_usdc=50.0, intended_price=0.51,
    )

    r = fill_handler.apply_trade_event({
        "id": "topup_trade", "status": "confirmed",
        "taker_order_id": "0xord_topup",
        "size": 100.0, "price": 0.50,  # actual fill at 0.50
    })
    assert r["action"] == "filled"
    assert r["role"] == "topup"

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT size_usdc, shares, entry_price FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(150.0)  # 100 + 50
    assert row[1] == pytest.approx(300.0)  # 200 + 100
    assert row[2] == pytest.approx(0.50)


# ===========================================================================
# apply_trade_event — FAILED handling
# ===========================================================================

def test_apply_trade_failed_entry_cancels_position(temp_db):
    pid = _seed(fill_status="pending")
    r = fill_handler.apply_trade_event({
        "id": "t1", "status": "failed", "taker_order_id": "0xord_entry",
        "size": 100.0, "price": 0.50,
    })
    assert r["action"] == "failed"
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT status, fill_status, cancelled_reason FROM positions WHERE id=?",
        (pid,)).fetchone()
    conn.close()
    assert row[0] == "closed"
    assert row[1] == "cancelled"
    assert row[2] == "trade_failed_onchain"


def test_apply_trade_failed_topup_clears_pending(temp_db):
    pid = _seed()
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord_topup",
        amount_usdc=50.0, intended_price=0.51,
    )
    r = fill_handler.apply_trade_event({
        "id": "t1", "status": "failed", "taker_order_id": "0xord_topup",
        "size": 100.0, "price": 0.51,
    })
    assert r["action"] == "failed"
    assert r["role"] == "topup"
    assert len(db.get_positions_with_pending_topup()) == 0


# ===========================================================================
# apply_order_event — CANCELLATION
# ===========================================================================

def test_apply_order_cancellation_topup_clears_pending(temp_db):
    pid = _seed()
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord_topup",
        amount_usdc=50.0, intended_price=0.51,
    )
    r = fill_handler.apply_order_event({
        "id": "0xord_topup", "type": "CANCELLATION",
    })
    assert r["action"] == "cancelled"
    assert r["role"] == "topup"
    assert len(db.get_positions_with_pending_topup()) == 0


def test_apply_order_cancellation_pending_entry_cancels(temp_db):
    pid = _seed(fill_status="pending")
    r = fill_handler.apply_order_event({
        "id": "0xord_entry", "type": "CANCELLATION",
    })
    assert r["action"] == "cancelled"
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT status, fill_status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[1] == "cancelled"


def test_apply_order_cancellation_filled_entry_noop(temp_db):
    """Cancellation event arriving AFTER the fill must not undo the fill."""
    pid = _seed(fill_status="filled")
    r = fill_handler.apply_order_event({
        "id": "0xord_entry", "type": "CANCELLATION",
    })
    assert r["action"] == "cancelled_noop"
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT fill_status FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == "filled"


def test_apply_order_placement_acks(temp_db):
    _seed()
    r = fill_handler.apply_order_event({
        "id": "0xord_entry", "type": "PLACEMENT",
    })
    assert r["action"] == "placement_ack"


def test_apply_order_unknown_order_ignored(temp_db):
    _seed()
    r = fill_handler.apply_order_event({"id": "0xforeign", "type": "PLACEMENT"})
    assert r["action"] == "ignored_no_position"


# ===========================================================================
# user_ws — dispatch + subscribe payload
# ===========================================================================

def test_user_ws_dispatch_routes_trade(temp_db, monkeypatch):
    """_dispatch with event_type='trade' must call apply_trade_event."""
    import user_ws

    captured = []

    def fake_apply_trade(event, my_wallet=None):
        captured.append(("trade", event, my_wallet))
        return {"action": "filled", "position_id": 1}

    def fake_apply_order(event):
        captured.append(("order", event))
        return {"action": "placement_ack"}

    monkeypatch.setattr("fill_handler.apply_trade_event", fake_apply_trade)
    monkeypatch.setattr("fill_handler.apply_order_event", fake_apply_order)
    monkeypatch.setattr(user_ws, "_my_wallet", "0xmywallet")

    user_ws._dispatch({"event_type": "trade", "id": "t1", "status": "confirmed"})
    assert captured[0][0] == "trade"
    assert captured[0][2] == "0xmywallet"


def test_user_ws_dispatch_routes_order(temp_db, monkeypatch):
    import user_ws

    captured = []
    monkeypatch.setattr("fill_handler.apply_order_event",
                        lambda e: captured.append(e) or {"action": "placement_ack"})
    monkeypatch.setattr("fill_handler.apply_trade_event",
                        lambda e, my_wallet=None: pytest.fail("should not be called"))

    user_ws._dispatch({"event_type": "order", "id": "0xo1", "type": "PLACEMENT"})
    assert len(captured) == 1


def test_user_ws_dispatch_unknown_event_silently_ignored(temp_db, monkeypatch):
    import user_ws

    monkeypatch.setattr("fill_handler.apply_trade_event",
                        lambda e, my_wallet=None: pytest.fail("should not call"))
    monkeypatch.setattr("fill_handler.apply_order_event",
                        lambda e: pytest.fail("should not call"))
    user_ws._dispatch({"event_type": "weather_alert"})  # made-up


def test_user_ws_subscribe_payload_shape(temp_db):
    """The first frame must include auth + markets + type='user'."""
    import user_ws

    user_ws._auth = ("api-key-1", "secret-1", "passphrase-1")
    payload_str = asyncio.run(
        user_ws._build_subscribe_payload({"0xmkt_a", "0xmkt_b"})
    )
    payload = json.loads(payload_str)
    assert payload["type"] == "user"
    assert payload["auth"] == {
        "apiKey":     "api-key-1",
        "secret":     "secret-1",
        "passphrase": "passphrase-1",
    }
    assert sorted(payload["markets"]) == ["0xmkt_a", "0xmkt_b"]
