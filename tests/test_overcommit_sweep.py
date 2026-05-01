"""
test_overcommit_sweep.py — Tests for the over-commitment auto-cancel
sweep (Phase B completion, 2026-04-30).

Phase B's get_committed_usdc + top-up gap calc prevents NEW
double-commits.  The sweep tested here cleans up EXISTING
over-commitments (e.g. a partial-fill entry + a top-up both resting on
the book from before Phase B was deployed).

Critical design point: the sweep ONLY fires when committed > target.
A normal partial-fill resting on the book at-target is left alone
(let it keep filling).  See test_no_sweep_for_at_target_partial below.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import monitor


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _seed_position(target=10.0) -> int:
    return db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-04-30T12:00:00",
        target_size_usdc=target, shares=33,
        yes_token_id="tok_yes", is_paper=0, fill_status="filled",
    )


def _add_order(
    pid: int, order_id: str, role: str, intended: float,
    *, status: str = "pending", filled: float = 0.0,
    age_minutes: float = 30.0,
) -> None:
    """Insert a position_orders row + backdate it by `age_minutes`."""
    db.insert_position_order(
        position_id=pid, order_id=order_id, role=role,
        intended_usdc=intended, intended_shares=intended/0.30,
        limit_price=0.30, status=status,
    )
    if filled > 0 or status != "pending":
        db.update_position_order_status(
            order_id=order_id, status=status,
            filled_usdc=filled, filled_shares=filled/0.30 if filled > 0 else 0,
            fill_price=0.30 if filled > 0 else None,
        )
    # Backdate created_at so age check passes
    import sqlite3
    backdated = (datetime.now(timezone.utc)
                 - timedelta(minutes=age_minutes)).isoformat()
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE position_orders SET created_at=? WHERE order_id=?",
                 (backdated, order_id))
    conn.commit(); conn.close()


# ===========================================================================
# get_overcommitted_positions identifies the right positions
# ===========================================================================

def test_overcommitted_positions_excludes_at_target(temp_db):
    """Position with committed = target (no excess) should NOT show up."""
    pid = _seed_position(target=10.0)
    _add_order(pid, "0xord", "entry", intended=10.0, status="pending")
    over = db.get_overcommitted_positions()
    assert over == []


def test_overcommitted_positions_includes_excess(temp_db):
    """Position with entry partial + topup pending → over by topup amount."""
    pid = _seed_position(target=10.0)
    # Entry partial: $0.55 filled, $9.45 still resting → contributes $10
    _add_order(pid, "0xentry", "entry", intended=10.0,
               status="partial", filled=0.55)
    # Topup: $9.45 pending → contributes $9.45
    _add_order(pid, "0xtopup", "topup", intended=9.45, status="pending")

    over = db.get_overcommitted_positions()
    assert len(over) == 1
    assert over[0]["position_id"] == pid
    assert over[0]["committed_usdc"] == pytest.approx(19.45)
    assert over[0]["excess"] == pytest.approx(9.45)


def test_overcommitted_ignores_paper_positions(temp_db):
    """Paper positions don't have CLOB orders to cancel — skip."""
    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0,
        entry_price=0.30, entry_time="2026-04-30T12:00:00",
        target_size_usdc=10.0, shares=33, yes_token_id="tok",
        is_paper=1, fill_status="filled",
    )
    _add_order(pid, "0xe", "entry", 10.0, status="partial", filled=0.55)
    _add_order(pid, "0xt", "topup", 9.45, status="pending")
    over = db.get_overcommitted_positions()
    assert over == []


# ===========================================================================
# get_cancellable_orders returns oldest-first, only resting orders
# ===========================================================================

def test_cancellable_orders_excludes_terminal_statuses(temp_db):
    pid = _seed_position()
    # Filled order — terminal, no resting capital
    _add_order(pid, "0xfilled", "entry", 10.0, status="filled", filled=10.0)
    # Cancelled order — terminal
    _add_order(pid, "0xcancel", "topup", 5.0, status="cancelled")
    # Pending — has resting capital
    _add_order(pid, "0xpending", "topup", 3.0, status="pending")

    cans = db.get_cancellable_orders_for_position(pid)
    order_ids = [c["order_id"] for c in cans]
    assert "0xpending" in order_ids
    assert "0xfilled" not in order_ids
    assert "0xcancel" not in order_ids


def test_cancellable_orders_excludes_zero_resting(temp_db):
    """A 'partial' order with filled == intended (within tolerance) has
    zero resting and shouldn't be cancellable."""
    pid = _seed_position()
    _add_order(pid, "0xfull", "entry", 10.0, status="partial", filled=10.0)
    cans = db.get_cancellable_orders_for_position(pid)
    assert len(cans) == 0


def test_cancellable_orders_oldest_first(temp_db):
    """Oldest order gets cancelled first."""
    pid = _seed_position()
    _add_order(pid, "0xnew", "topup", 5.0, status="pending", age_minutes=5)
    _add_order(pid, "0xold", "entry", 10.0, status="pending", age_minutes=60)
    cans = db.get_cancellable_orders_for_position(pid)
    assert cans[0]["order_id"] == "0xold"


# ===========================================================================
# Sweep behavior — the actual auto-cancel
# ===========================================================================

def test_sweep_cancels_overcommitted(temp_db, monkeypatch):
    """The screenshot scenario: entry $10 partial-filled $0.55, topup $9.45
    pending → committed $19.45 / target $10.  Sweep cancels the oldest
    resting order (the entry's resting portion)."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_position(target=10.0)
    _add_order(pid, "0xentry", "entry", 10.0, status="partial",
               filled=0.55, age_minutes=30)
    _add_order(pid, "0xtopup", "topup", 9.45, status="pending",
               age_minutes=20)

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": ["0xentry"]}

    cancelled = monitor._cancel_overcommitted_orders(client)
    assert cancelled == 1
    # Oldest order (entry) gets cancelled
    client.cancel_orders.assert_called_once_with(["0xentry"])
    # Ledger reflects the cancellation
    row = db.get_position_order_by_id("0xentry")
    assert row["status"] == "cancelled"


def test_sweep_does_not_fire_at_target_partial(temp_db, monkeypatch):
    """The user's confirmation case: a partial-fill at-target should NOT
    be cancelled.  Just $10 entry partial-filled $5, no topup, committed
    still $10 (full intended of partial).  Bot should leave it alone."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_position(target=10.0)
    _add_order(pid, "0xentry", "entry", 10.0, status="partial",
               filled=5.0, age_minutes=30)

    client = MagicMock()
    cancelled = monitor._cancel_overcommitted_orders(client)
    assert cancelled == 0
    client.cancel_orders.assert_not_called()


def test_sweep_skips_too_young_orders(temp_db, monkeypatch):
    """Even when over-committed, a freshly-placed order shouldn't be
    cancelled — give it a chance to fill first.  Same age cutoff as
    the regular cancel pass."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_position(target=10.0)
    _add_order(pid, "0xentry", "entry", 10.0, status="partial",
               filled=0.55, age_minutes=2)   # fresh
    _add_order(pid, "0xtopup", "topup", 9.45, status="pending",
               age_minutes=1)                # also fresh

    client = MagicMock()
    cancelled = monitor._cancel_overcommitted_orders(client)
    assert cancelled == 0
    client.cancel_orders.assert_not_called()


def test_sweep_paper_mode_noop(temp_db, monkeypatch):
    monkeypatch.setattr(monitor, "PAPER_TRADE", True)
    pid = _seed_position(target=10.0)
    _add_order(pid, "0xentry", "entry", 10.0, status="partial", filled=0.55)
    _add_order(pid, "0xtopup", "topup", 9.45, status="pending")
    client = MagicMock()
    assert monitor._cancel_overcommitted_orders(client) == 0


def test_sweep_cancels_one_per_position_per_cycle(temp_db, monkeypatch):
    """If a position has TWO over-committing orders, only cancel one this
    cycle.  Avoids cascading cancels before the CLOB has had time to
    register the first one."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_position(target=10.0)
    _add_order(pid, "0xord1", "entry", 10.0, status="pending", age_minutes=30)
    _add_order(pid, "0xord2", "topup", 8.0, status="pending", age_minutes=25)
    _add_order(pid, "0xord3", "topup", 5.0, status="pending", age_minutes=20)
    # committed = $10 + $8 + $5 = $23, excess = $13

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": []}
    cancelled = monitor._cancel_overcommitted_orders(client)
    # Only the OLDEST order gets cancelled this cycle (not all 3)
    assert cancelled == 1
    assert client.cancel_orders.call_count == 1
    assert client.cancel_orders.call_args[0][0] == ["0xord1"]


def test_sweep_leaves_normal_positions_alone(temp_db, monkeypatch):
    """Multiple positions; only the over-committed one gets touched."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid_ok = _seed_position(target=10.0)
    _add_order(pid_ok, "0xok", "entry", 10.0,
               status="filled", filled=10.0, age_minutes=30)

    pid_over = _seed_position(target=10.0)
    _add_order(pid_over, "0xover_e", "entry", 10.0,
               status="partial", filled=0.55, age_minutes=30)
    _add_order(pid_over, "0xover_t", "topup", 9.45,
               status="pending", age_minutes=25)

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": []}
    monitor._cancel_overcommitted_orders(client)
    # The OK position's filled order is untouched
    client.cancel_orders.assert_called_once()
    args = client.cancel_orders.call_args[0][0]
    assert args == ["0xover_e"]
    # OK position's order remains filled
    assert db.get_position_order_by_id("0xok")["status"] == "filled"


# ===========================================================================
# Edge case: cancelled-with-partial-fill semantics
# ===========================================================================

def test_committed_usdc_after_partial_cancelled(temp_db):
    """When a partial-filled order gets cancelled, the filled portion
    stays counted (real on-chain shares) but the rest is freed."""
    pid = _seed_position(target=10.0)
    db.insert_position_order(
        position_id=pid, order_id="0xord", role="entry",
        intended_usdc=10.0, intended_shares=33, limit_price=0.30,
    )
    db.update_position_order_status(
        order_id="0xord", status="cancelled",
        filled_shares=1.83, filled_usdc=0.55, fill_price=0.30,
        cancelled_reason="overcommit_sweep", closed=True,
    )
    # Was 'partial' → counted full $10; now cancelled with $0.55 filled
    # → counts only the $0.55 (real on-chain capital)
    assert db.get_committed_usdc(pid) == pytest.approx(0.55)
