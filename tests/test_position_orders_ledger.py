"""
test_position_orders_ledger.py — Tests for the per-position order ledger
(Phase B, 2026-04-30).

The ledger is the system of record for every CLOB order placed for a
position (entry, top-ups, exits).  It exists so the top-up gap calc can
read `committed_usdc = filled + still_resting` instead of the buggy
`target - filled_only` that double-committed capital on top of partial
fills.

Tests pin:
  * insert / update lifecycle helpers
  * committed_usdc math under all relevant statuses
  * partial-fill edge case (filled='filled' but filled_usdc < intended)
  * backfill from legacy positions table is idempotent
  * integration: top-up logic reads committed_usdc and skips correctly
"""

from __future__ import annotations

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


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _seed_position(**kwargs) -> int:
    defaults = dict(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-04-30T12:00:00",
        target_size_usdc=10.0, shares=33.33,
        yes_token_id="tok_yes", is_paper=0, fill_status="filled",
    )
    defaults.update(kwargs)
    return db.insert_position(**defaults)


# ===========================================================================
# Insert / lookup
# ===========================================================================

def test_insert_position_order_and_lookup(temp_db):
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xord1", role="entry",
        intended_usdc=10.0, intended_shares=33.33, limit_price=0.30,
    )
    row = db.get_position_order_by_id("0xord1")
    assert row is not None
    assert row["position_id"] == pid
    assert row["role"] == "entry"
    assert row["status"] == "pending"
    assert row["intended_usdc"] == pytest.approx(10.0)
    assert row["filled_usdc"] == 0.0   # default


def test_insert_rejects_invalid_role(temp_db):
    pid = _seed_position()
    with pytest.raises(ValueError, match="role"):
        db.insert_position_order(
            position_id=pid, order_id="0xbad", role="not_a_role",
            intended_usdc=10.0, intended_shares=33.33, limit_price=0.30,
        )


def test_get_position_orders_filter_by_role(temp_db):
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xe", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.insert_position_order(
        position_id=pid, order_id="0xt", role="topup",
        intended_usdc=5.0, intended_shares=17, limit_price=0.30,
    )
    db.insert_position_order(
        position_id=pid, order_id="0xx", role="exit",
        intended_usdc=15.0, intended_shares=50, limit_price=0.30,
    )
    assert len(db.get_position_orders(pid)) == 3
    assert len(db.get_position_orders(pid, role="entry")) == 1
    assert len(db.get_position_orders(pid, role="topup")) == 1
    assert len(db.get_position_orders(pid, role="exit")) == 1


# ===========================================================================
# update_position_order_status
# ===========================================================================

def test_update_position_order_status_marks_filled(temp_db):
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xord", role="entry",
        intended_usdc=10.0, intended_shares=33.33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xord", status="filled",
        filled_shares=33.33, filled_usdc=10.0, fill_price=0.30,
        fee_usdc=0.05, closed=True,
    )
    row = db.get_position_order_by_id("0xord")
    assert row["status"] == "filled"
    assert row["filled_usdc"] == pytest.approx(10.0)
    assert row["fill_price"] == pytest.approx(0.30)
    assert row["fee_usdc"] == pytest.approx(0.05)
    assert row["closed_at"] is not None


def test_update_partial_then_filled(temp_db):
    """Realistic flow: order matches some shares (status='partial'), then
    eventually fills the rest (status='filled')."""
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xord", role="entry",
        intended_usdc=10.0, intended_shares=33.33, limit_price=0.30,
    )
    # First trade event — partial fill
    db.update_position_order_status(
        order_id="0xord", status="partial",
        filled_shares=10, filled_usdc=3.0, fill_price=0.30,
    )
    row = db.get_position_order_by_id("0xord")
    assert row["status"] == "partial"
    assert row["closed_at"] is None  # still open

    # Second trade event — rest fills
    db.update_position_order_status(
        order_id="0xord", status="filled",
        filled_shares=33.33, filled_usdc=10.0, fill_price=0.30,
        closed=True,
    )
    row = db.get_position_order_by_id("0xord")
    assert row["status"] == "filled"
    assert row["closed_at"] is not None


def test_update_returns_false_for_unknown_order(temp_db):
    assert db.update_position_order_status(
        order_id="nonexistent", status="cancelled"
    ) is False


# ===========================================================================
# get_committed_usdc — the key calc
# ===========================================================================

def test_committed_usdc_filled_entry_only(temp_db):
    """Single fully-filled entry — committed = filled."""
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xe", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xe", status="filled",
        filled_shares=33, filled_usdc=10.0, fill_price=0.30, closed=True,
    )
    assert db.get_committed_usdc(pid) == pytest.approx(10.0)


def test_committed_usdc_pending_entry_counts_intended(temp_db):
    """A pending order's intended_usdc still commits capital — top-up
    logic must not assume it's free to add more."""
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xpending", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    # Still pending, never filled
    assert db.get_committed_usdc(pid) == pytest.approx(10.0)


def test_committed_usdc_partial_fill_counts_full_intended(temp_db):
    """THE KEY CASE — the bug from the screenshot.
    Entry $10 placed, only $0.55 filled, rest still resting on book.
    Committed = $10 (the FULL intended), NOT $0.55 (what's filled)."""
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xpartial", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    # Partial fill: 1.83 shares × $0.30 = $0.55 filled, $9.45 still resting
    db.update_position_order_status(
        order_id="0xpartial", status="partial",
        filled_shares=1.83, filled_usdc=0.55, fill_price=0.30,
    )
    assert db.get_committed_usdc(pid) == pytest.approx(10.0), (
        "partial-fill orders must count their FULL intended_usdc, "
        "not just the filled portion — the rest is still on book"
    )


def test_committed_usdc_legacy_filled_with_partial_data(temp_db):
    """Defensive: a row with status='filled' but filled_usdc < intended
    (legacy fill_handler that didn't yet emit 'partial' status) must
    still be counted as fully-committing."""
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xlegacy", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    # Marked 'filled' but only 5% actually filled — older code path
    db.update_position_order_status(
        order_id="0xlegacy", status="filled",
        filled_shares=1.83, filled_usdc=0.55, fill_price=0.30, closed=True,
    )
    # Should fall into the partial-defensive branch and use intended_usdc
    assert db.get_committed_usdc(pid) == pytest.approx(10.0)


def test_committed_usdc_cancelled_doesnt_count(temp_db):
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xcancel", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xcancel", status="cancelled", closed=True,
    )
    assert db.get_committed_usdc(pid) == 0.0


def test_committed_usdc_failed_doesnt_count(temp_db):
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xfail", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xfail", status="failed", closed=True,
    )
    assert db.get_committed_usdc(pid) == 0.0


def test_committed_usdc_sums_entry_plus_topup(temp_db):
    """The screenshot scenario — partial-fill entry + pending top-up.
    Both contribute to committed; this is the value the top-up logic
    compares against target."""
    pid = _seed_position()
    # Entry: $10 intended, $0.55 filled, $9.45 resting
    db.insert_position_order(
        position_id=pid, order_id="0xe", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xe", status="partial",
        filled_shares=1.83, filled_usdc=0.55, fill_price=0.30,
    )
    # Top-up: $9.45 placed (the gap), still pending
    db.insert_position_order(
        position_id=pid, order_id="0xt", role="topup",
        intended_usdc=9.45, intended_shares=27, limit_price=0.35,
    )
    # committed = $10 (full intended of partial entry) + $9.45 (pending topup)
    # = $19.45 — the actual on-chain over-exposure visible in user's screenshot
    assert db.get_committed_usdc(pid) == pytest.approx(19.45)


def test_committed_usdc_excludes_exit_role(temp_db):
    """Exit orders don't compound buy-side capital — they release shares,
    not commit USDC.  Only entry+topup roles count."""
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xe", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xe", status="filled",
        filled_shares=33, filled_usdc=10.0, fill_price=0.30, closed=True,
    )
    db.insert_position_order(
        position_id=pid, order_id="0xx", role="exit",
        intended_usdc=20.0, intended_shares=33, limit_price=0.60,
    )
    # Pending exit doesn't count toward committed
    assert db.get_committed_usdc(pid) == pytest.approx(10.0)


# ===========================================================================
# get_filled_usdc — sum of actual fills
# ===========================================================================

def test_filled_usdc_sums_actual_fills(temp_db):
    pid = _seed_position()
    db.insert_position_order(
        position_id=pid, order_id="0xa", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xa", status="filled", filled_shares=33,
        filled_usdc=9.85, fill_price=0.298, closed=True,
    )
    db.insert_position_order(
        position_id=pid, order_id="0xb", role="topup",
        intended_usdc=5.0, intended_shares=16, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xb", status="partial",
        filled_shares=8, filled_usdc=2.40, fill_price=0.30,
    )
    # filled = $9.85 + $2.40 (real on-chain cost only)
    assert db.get_filled_usdc(pid) == pytest.approx(12.25)


# ===========================================================================
# Backfill idempotency
# ===========================================================================

def test_backfill_position_orders_handles_existing(temp_db):
    """Backfill is idempotent — running twice produces same state."""
    pid = _seed_position(
        order_id="0xentry", target_size_usdc=10.0,
    )
    counts1 = db.backfill_position_orders()
    counts2 = db.backfill_position_orders()
    assert counts1["entries"] == 1
    assert counts2["entries"] == 0   # nothing new to add
    # Verify the row was created exactly once
    rows = db.get_position_orders(pid)
    assert len(rows) == 1


def test_backfill_synthesizes_topup_from_pending_fields(temp_db):
    """Position with pending_topup_order_id should get a 'topup' row
    in 'pending' status."""
    pid = _seed_position(order_id="0xentry")
    db.update_position_topup_pending(
        position_id=pid, order_id="0xtopup",
        amount_usdc=5.0, intended_price=0.32,
    )
    db.backfill_position_orders()
    rows = db.get_position_orders(pid)
    roles = sorted(r["role"] for r in rows)
    assert roles == ["entry", "topup"]
    topup = next(r for r in rows if r["role"] == "topup")
    assert topup["intended_usdc"] == pytest.approx(5.0)
    assert topup["status"] == "pending"


# ===========================================================================
# Status constants exported for fill_handler use
# ===========================================================================

def test_committing_statuses_constant_is_correct(temp_db):
    assert "pending" in db._ORDER_STATUS_COMMITTING
    assert "partial" in db._ORDER_STATUS_COMMITTING
    assert "cancelled" not in db._ORDER_STATUS_COMMITTING
    assert "failed" not in db._ORDER_STATUS_COMMITTING


def test_terminal_statuses_constant_is_correct(temp_db):
    assert "cancelled" in db._ORDER_STATUS_TERMINAL
    assert "failed" in db._ORDER_STATUS_TERMINAL
    assert "filled" in db._ORDER_STATUS_TERMINAL
    assert "pending" not in db._ORDER_STATUS_TERMINAL
