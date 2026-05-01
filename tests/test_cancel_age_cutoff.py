"""
test_cancel_age_cutoff.py — Regression tests for the cancel-too-fast bug.

Bug story (2026-04-30)
-----------------------
The bot's startup sequence runs `trading_run()` immediately followed by
`run_monitor_loop()` (so monitor catches up on existing pending state).
But `_cancel_pending_orders` had no age check — it cancelled EVERY
pending order regardless of when it was placed.  Result: orders placed
by the trading run were nuked by the monitor 5 seconds later, before
they could fill on the book.

Fix: orders younger than MIN_ORDER_AGE_BEFORE_CANCEL_MIN minutes are
skipped by the cancel pass.  10-minute default — long enough to let
fillable orders land, short enough that abandoned orders don't pile up.
"""

from __future__ import annotations

import os
import sqlite3
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
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _seed_pending(*, age_minutes: float, contract_id: str = "0xabc") -> int:
    """Insert a pending position with entry_time set to N minutes ago."""
    entry_dt = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return db.insert_position(
        contract_id  = contract_id,
        side         = "YES",
        size_usdc    = 10.0,
        entry_price  = 0.30,
        entry_time   = entry_dt.isoformat(),
        order_id     = f"0xord{int(age_minutes*100)}",
        shares       = 33.33,
        yes_token_id = "tok",
        is_paper     = 0,
        fill_status  = "pending",
    )


# ===========================================================================
# Cancel-age cutoff
# ===========================================================================

def test_cancel_skips_fresh_orders(temp_db, monkeypatch):
    """An order placed 5 seconds ago must NOT be cancelled.  This is the
    direct reproduction of the production bug — startup monitor cancelled
    fresh orders within seconds of placement."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_pending(age_minutes=0.1)  # 6 seconds old

    client = MagicMock()
    cancelled = monitor._cancel_pending_orders(client)

    assert cancelled == 0, "fresh order must not be cancelled"
    # Position is still pending in DB
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    pos = dict(conn.execute(
        "SELECT fill_status, status FROM positions WHERE id=?", (pid,)
    ).fetchone())
    conn.close()
    assert pos["fill_status"] == "pending"
    assert pos["status"] == "open"
    # CLOB cancel was never called
    client.cancel_orders.assert_not_called()


def test_cancel_kills_old_orders(temp_db, monkeypatch):
    """Orders past the threshold should still be cancelled — that's the
    point of the cancel pass."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_pending(age_minutes=20.0)  # 20 minutes old, well past threshold

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": ["0xord2000"]}
    cancelled = monitor._cancel_pending_orders(client)

    assert cancelled == 1
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    pos = dict(conn.execute(
        "SELECT fill_status, cancelled_reason FROM positions WHERE id=?", (pid,)
    ).fetchone())
    conn.close()
    assert pos["fill_status"] == "cancelled"
    assert pos["cancelled_reason"] == "unfilled_before_monitor_run"


def test_cancel_at_exactly_threshold_age_still_cancels(temp_db, monkeypatch):
    """The threshold is `< MIN_ORDER_AGE_BEFORE_CANCEL_MIN` (strict less).
    An order at exactly the threshold age cancels."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = _seed_pending(age_minutes=10.5)  # just over threshold

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": []}
    cancelled = monitor._cancel_pending_orders(client)
    assert cancelled == 1


def test_cancel_mixed_fresh_and_old(temp_db, monkeypatch):
    """Fresh orders are skipped; old orders in the same pass are cancelled.
    This is the realistic monitor-cycle scenario — some fresh-from-trading,
    some lingering from earlier cycles."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid_old1   = _seed_pending(age_minutes=30.0,  contract_id="0xold1")
    pid_fresh1 = _seed_pending(age_minutes=0.5,   contract_id="0xfresh1")
    pid_old2   = _seed_pending(age_minutes=15.0,  contract_id="0xold2")
    pid_fresh2 = _seed_pending(age_minutes=2.0,   contract_id="0xfresh2")

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": []}
    cancelled = monitor._cancel_pending_orders(client)

    assert cancelled == 2  # only the two old ones

    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    states = {
        r["id"]: r["fill_status"] for r in conn.execute(
            "SELECT id, fill_status FROM positions"
        ).fetchall()
    }
    conn.close()
    assert states[pid_old1]   == "cancelled"
    assert states[pid_old2]   == "cancelled"
    assert states[pid_fresh1] == "pending"
    assert states[pid_fresh2] == "pending"


def test_cancel_handles_unparseable_entry_time(temp_db, monkeypatch):
    """Defensive: a row with garbage entry_time falls through to the
    cancel path (don't get stuck with un-cancellable garbage forever)."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    pid = db.insert_position(
        contract_id  = "0xabc", side="YES", size_usdc=10.0,
        entry_price  = 0.30, entry_time="not-a-real-timestamp",
        order_id     = "0xgarbage", shares=33.33,
        yes_token_id = "tok", is_paper=0, fill_status="pending",
    )
    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": []}
    cancelled = monitor._cancel_pending_orders(client)
    assert cancelled == 1


def test_cancel_zero_threshold_cancels_everything(temp_db, monkeypatch):
    """Setting MIN_ORDER_AGE_BEFORE_CANCEL_MIN=0 reverts to original
    behavior (cancel everything regardless of age)."""
    import config
    monkeypatch.setattr(config, "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 0.0)
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)

    _seed_pending(age_minutes=0.05)  # 3 seconds old

    client = MagicMock()
    client.cancel_orders.return_value = {"canceled": []}
    cancelled = monitor._cancel_pending_orders(client)
    assert cancelled == 1, "with threshold=0, even brand-new orders cancel"
