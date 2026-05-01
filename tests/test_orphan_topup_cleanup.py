"""
test_orphan_topup_cleanup.py — Tests for the safety-net poll that detects
externally-cancelled topup orders and clears stale pending_topup pointers.

Polymarket may cancel resting orders without a corresponding WS event
(account risk checks, WS auth disconnect, manual UI cancel).  Without
this safety net, the DB carries a dead pointer forever and _run_topups
silently skips the position.  See monitor.detect_externally_cancelled_topups.
"""

from __future__ import annotations

import os
import sys
import sqlite3
from unittest.mock import MagicMock

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import monitor
import config


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


@pytest.fixture
def live_mode(monkeypatch):
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)
    yield


def _seed_position_with_pending_topup(
    *,
    topup_order_id: str = "0xpending_topup_order",
    topup_amount: float = 3.36,
    is_paper: int = 0,
) -> int:
    pid = db.insert_position(
        contract_id="0xabc", side="YES",
        size_usdc=6.64, entry_price=0.32,
        entry_time="2026-05-01T02:00:00",
        order_id="0xord_entry", target_size_usdc=10.0, shares=20.75,
        yes_token_id="tok_yes", is_paper=is_paper, fill_status="filled",
    )
    # Stamp the pending-topup pointer
    db.update_position_topup_pending(
        position_id    = pid,
        order_id       = topup_order_id,
        amount_usdc    = topup_amount,
        intended_price = 0.32,
    )
    # Production also inserts a ledger row at topup placement time
    db.insert_position_order(
        position_id     = pid,
        order_id        = topup_order_id,
        role            = "topup",
        intended_usdc   = topup_amount,
        intended_shares = topup_amount / 0.32,
        limit_price     = 0.32,
        status          = "pending",
    )
    return pid


# ===========================================================================
# Detection: dead orders get cleaned up
# ===========================================================================

def test_clob_returns_none_clears_pointer(temp_db, live_mode, monkeypatch):
    """When CLOB get_order returns None (order doesn't exist), clear the
    pointer and mark the ledger cancelled."""
    pid = _seed_position_with_pending_topup(topup_order_id="0xdead_order_1")

    def fake_get_order_status(order_id, client):
        return None  # CLOB says order doesn't exist

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)

    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 1

    # Pointer cleared
    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT pending_topup_order_id FROM positions WHERE id=?", (pid,)
        ).fetchone()
        assert row[0] is None

    # Ledger marked cancelled
    ledger = db.get_position_order_by_id("0xdead_order_1")
    assert ledger["status"] == "cancelled"
    assert ledger["cancelled_reason"] == "cancelled_externally"


def test_clob_returns_canceled_status_clears_pointer(temp_db, live_mode, monkeypatch):
    """When CLOB get_order returns a dict with status='CANCELED', clean up.
    This is the case Polymarket actually emits — explicit dead status."""
    pid = _seed_position_with_pending_topup(topup_order_id="0xexplicit_canceled")

    def fake_get_order_status(order_id, client):
        return {"id": order_id, "status": "CANCELED", "size_matched": "0"}

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)

    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 1

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT pending_topup_order_id FROM positions WHERE id=?", (pid,)
        ).fetchone()
        assert row[0] is None


def test_clob_returns_cancelled_lowercase_also_cleared(temp_db, live_mode, monkeypatch):
    """Defensive: handle case-variant 'CANCELLED' (British spelling) the
    same as 'CANCELED' — both observed in Polymarket responses."""
    pid = _seed_position_with_pending_topup(topup_order_id="0xbritish_spelling")

    def fake_get_order_status(order_id, client):
        return {"id": order_id, "status": "cancelled"}  # lowercase

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)

    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 1


# ===========================================================================
# Non-detection: live orders left alone
# ===========================================================================

def test_clob_returns_live_leaves_pointer_alone(temp_db, live_mode, monkeypatch):
    """When CLOB says LIVE, the pointer must be preserved — no cleanup."""
    pid = _seed_position_with_pending_topup(topup_order_id="0xstill_live")

    def fake_get_order_status(order_id, client):
        return {"id": order_id, "status": "LIVE", "size_matched": "0",
                "original_size": "10.49"}

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)

    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 0

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT pending_topup_order_id FROM positions WHERE id=?", (pid,)
        ).fetchone()
        assert row[0] == "0xstill_live"


def test_clob_returns_matched_leaves_pointer_alone(temp_db, live_mode, monkeypatch):
    """MATCHED status (engine-matched but not on chain yet) is still a live
    state — leave alone, the WS will eventually report the fill."""
    pid = _seed_position_with_pending_topup(topup_order_id="0xmatched_pending")

    def fake_get_order_status(order_id, client):
        return {"id": order_id, "status": "MATCHED"}

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)
    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 0


# ===========================================================================
# Defensive cases
# ===========================================================================

def test_paper_position_skipped(temp_db, live_mode, monkeypatch):
    """Paper positions never have CLOB orders to clean up — skip entirely."""
    pid = _seed_position_with_pending_topup(
        topup_order_id="0xpaper_topup", is_paper=1
    )

    fake = MagicMock()  # would crash if it got called

    def fake_get_order_status(order_id, client):
        raise AssertionError("CLOB should not be queried for paper positions")

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)
    n = monitor.detect_externally_cancelled_topups(client=fake)
    assert n == 0


def test_paper_mode_global_skipped(temp_db, monkeypatch):
    """When the bot is in PAPER_TRADE mode entirely, the function is a no-op."""
    monkeypatch.setattr(monitor, "PAPER_TRADE", True)
    _seed_position_with_pending_topup(topup_order_id="0xanything")
    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 0


def test_no_client_returns_zero(temp_db, live_mode):
    """No CLOB client (None) → no-op, returns 0."""
    _seed_position_with_pending_topup(topup_order_id="0xanything")
    n = monitor.detect_externally_cancelled_topups(client=None)
    assert n == 0


def test_no_pending_topups_returns_zero(temp_db, live_mode):
    """When no positions have a pending pointer, the function returns 0
    without making any CLOB calls."""
    db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=10.0, entry_price=0.30,
        entry_time="2026-05-01T02:00:00", target_size_usdc=10.0, shares=33.3,
        is_paper=0, fill_status="filled",
    )
    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    assert n == 0


def test_clob_exception_does_not_crash(temp_db, live_mode, monkeypatch):
    """If get_order_status raises (transient API error), skip that position
    and continue.  Don't blow up the whole sweep."""
    pid_a = _seed_position_with_pending_topup(topup_order_id="0xerrors")
    # Add a second position whose CLOB call will succeed
    db.insert_position(
        contract_id="0xdef", side="YES", size_usdc=6.0, entry_price=0.30,
        entry_time="2026-05-01T02:00:00", target_size_usdc=10.0, shares=20.0,
        is_paper=0, fill_status="filled", order_id="0xord_b",
    )
    pid_b_rows = []
    with sqlite3.connect(temp_db) as conn:
        pid_b = conn.execute(
            "SELECT id FROM positions WHERE contract_id='0xdef'"
        ).fetchone()[0]
    db.update_position_topup_pending(
        position_id=pid_b, order_id="0xworks",
        amount_usdc=4.0, intended_price=0.30,
    )
    db.insert_position_order(
        position_id=pid_b, order_id="0xworks", role="topup",
        intended_usdc=4.0, intended_shares=13.3,
        limit_price=0.30, status="pending",
    )

    def fake_get_order_status(order_id, client):
        if order_id == "0xerrors":
            raise RuntimeError("transient CLOB error")
        return None  # other order: dead

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)
    n = monitor.detect_externally_cancelled_topups(client=MagicMock())
    # Only the second position cleaned up — error on first should be swallowed
    assert n == 1


# ===========================================================================
# Audit trail
# ===========================================================================

def test_repair_logged_to_activity_log(temp_db, live_mode, monkeypatch):
    """Every cleanup should leave an activity_log entry with category=REPAIR
    and repair_kind=externally_cancelled_topup so the audit trail is
    reconstructible."""
    pid = _seed_position_with_pending_topup(topup_order_id="0xaudited")

    def fake_get_order_status(order_id, client):
        return None

    monkeypatch.setattr("execution.get_order_status", fake_get_order_status)
    monitor.detect_externally_cancelled_topups(client=MagicMock())

    # Read back the activity_log row
    rows = db.get_recent_activity(limit=10, categories=["REPAIR"])
    assert len(rows) >= 1
    repair = rows[0]
    assert repair["position_id"] == pid
    assert "0xaudited" in repair["message"]
    import json
    meta = json.loads(repair["metadata"])
    assert meta["repair_kind"] == "externally_cancelled_topup"
    assert meta["stale_order_id"] == "0xaudited"


# ===========================================================================
# Fast wrapper smoke tests (the */5 scheduler entry-point)
# ===========================================================================

def test_run_orphan_topup_cleanup_fast_paper_mode(monkeypatch):
    monkeypatch.setattr(monitor, "PAPER_TRADE", True)
    assert monitor.run_orphan_topup_cleanup_fast() == 0


def test_run_orphan_topup_cleanup_fast_no_client(temp_db, live_mode, monkeypatch):
    monkeypatch.setattr("execution.get_clob_client", lambda: None)
    assert monitor.run_orphan_topup_cleanup_fast() == 0


def test_run_orphan_topup_cleanup_fast_swallows_exceptions(temp_db, live_mode, monkeypatch):
    monkeypatch.setattr("execution.get_clob_client", lambda: MagicMock())

    def boom(client):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(monitor, "detect_externally_cancelled_topups", boom)
    assert monitor.run_orphan_topup_cleanup_fast() == 0
