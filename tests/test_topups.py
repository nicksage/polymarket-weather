"""
test_topups.py — Tests for live top-up orders (Phase 6).

Covers:
  * execute_topup paper mode (immediate merge, no client call)
  * execute_topup live mode (places order + stamps pending fields)
  * execute_topup skips when a top-up is already pending on the parent
  * DB helpers: update_position_topup_pending, clear_position_topup_pending,
    get_positions_with_pending_topup
  * update_position_topup auto-clears pending fields when called
"""

from __future__ import annotations

import os
import sqlite3
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


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _seed_position(
    db_path: str,
    *,
    contract_id: str = "0xabc",
    size_usdc: float = 100.0,
    target_size_usdc: float = 200.0,
    entry_price: float = 0.50,
    shares: float = 200.0,
    yes_token_id: str = "0xyestoken",
) -> int:
    """Insert a position via the public helper to be schema-agnostic."""
    return db.insert_position(
        contract_id      = contract_id,
        side             = "YES",
        size_usdc        = size_usdc,
        entry_price      = entry_price,
        entry_time       = "2026-01-01T00:00:00",
        target_size_usdc = target_size_usdc,
        shares           = shares,
        yes_token_id     = yes_token_id,
        no_token_id      = None,
        is_paper         = 0,
        fill_status      = "filled",
    )


# ===========================================================================
# DB helpers
# ===========================================================================

def test_pending_topup_set_and_clear_roundtrip(temp_db):
    pid = _seed_position(temp_db)
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord1",
        amount_usdc=50.0, intended_price=0.502,
    )
    rows = db.get_positions_with_pending_topup()
    assert len(rows) == 1
    assert rows[0]["pending_topup_order_id"] == "0xord1"
    assert rows[0]["pending_topup_amount_usdc"] == pytest.approx(50.0)
    assert rows[0]["pending_topup_intended_price"] == pytest.approx(0.502)

    db.clear_position_topup_pending(pid)
    assert len(db.get_positions_with_pending_topup()) == 0


def test_get_pending_topups_excludes_closed_positions(temp_db):
    pid = _seed_position(temp_db)
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord1",
        amount_usdc=50.0, intended_price=0.5,
    )
    # Close the parent position
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE positions SET status='closed' WHERE id=?", (pid,))
    conn.commit(); conn.close()

    # Should NOT return — closed parent positions shouldn't show as pending top-ups
    assert len(db.get_positions_with_pending_topup()) == 0


def test_update_position_topup_clears_pending_fields(temp_db):
    """Merge call (called on fill) must wipe pending_* so nothing else
    tries to reconcile / cancel the now-merged top-up."""
    pid = _seed_position(temp_db)
    db.update_position_topup_pending(
        position_id=pid, order_id="0xord1",
        amount_usdc=50.0, intended_price=0.5,
    )
    db.update_position_topup(
        position_id=pid, added_usdc=50.0,
        added_shares=100.0, new_avg_price=0.50,
    )
    pending = db.get_positions_with_pending_topup()
    assert len(pending) == 0


# ===========================================================================
# execute_topup — paper mode
# ===========================================================================

def test_execute_topup_paper_merges_immediately(temp_db, monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", True)
    pid = _seed_position(temp_db, size_usdc=100.0, shares=200.0, entry_price=0.50)
    pos = {"id": pid, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.50, "yes_token_id": "0xyestoken"}

    result = execution.execute_topup(pos, add_amount_usdc=50.0, client=None)

    assert result["status"] == "paper"
    # Confirm DB merged the top-up
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT size_usdc, shares FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(150.0)   # 100 + 50
    assert row[1] == pytest.approx(300.0)   # 200 + 100


def test_execute_topup_paper_does_not_call_client(temp_db, monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", True)
    pid = _seed_position(temp_db)
    pos = {"id": pid, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.50, "yes_token_id": "0xyestoken"}
    client = MagicMock()
    execution.execute_topup(pos, add_amount_usdc=50.0, client=client)
    client.create_and_post_order.assert_not_called()


# ===========================================================================
# execute_topup — live mode
# ===========================================================================

def test_execute_topup_live_places_order_and_stamps_pending(temp_db, monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    pid = _seed_position(temp_db, size_usdc=100.0, entry_price=0.50)
    pos = {"id": pid, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.50, "yes_token_id": "0xyestoken",
           "pending_topup_order_id": None}
    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success": True,
        "orderID": "0xnewtopup",
        "status": "live",
    }

    result = execution.execute_topup(pos, add_amount_usdc=50.0, client=client)

    assert result["status"] == "placed"
    assert result["order_id"] == "0xnewtopup"
    client.create_and_post_order.assert_called_once()

    # Confirm DB has the pending fields stamped (NOT merged into size yet)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT size_usdc, pending_topup_order_id, "
        "pending_topup_amount_usdc, pending_topup_intended_price "
        "FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(100.0)        # NOT merged yet
    assert row[1] == "0xnewtopup"
    assert row[2] == pytest.approx(50.0)
    # limit price = entry_price * 1.005 = 0.5025 (within rounding)
    assert row[3] == pytest.approx(0.5025, abs=1e-4)


def test_execute_topup_live_skips_when_pending(temp_db, monkeypatch):
    """Don't double-issue when there's already an in-flight top-up."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    pid = _seed_position(temp_db)
    db.update_position_topup_pending(
        position_id=pid, order_id="0xexisting",
        amount_usdc=30.0, intended_price=0.5,
    )
    pos = {"id": pid, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.50, "yes_token_id": "0xyestoken",
           "pending_topup_order_id": "0xexisting"}
    client = MagicMock()

    result = execution.execute_topup(pos, add_amount_usdc=50.0, client=client)
    assert result["status"] == "skip"
    assert result["reason"] == "topup_already_pending"
    client.create_and_post_order.assert_not_called()


def test_execute_topup_live_handles_clob_failure(temp_db, monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    pid = _seed_position(temp_db)
    pos = {"id": pid, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.50, "yes_token_id": "0xyestoken",
           "pending_topup_order_id": None}
    client = MagicMock()
    client.create_and_post_order.return_value = {"success": False, "error": "rejected"}

    result = execution.execute_topup(pos, add_amount_usdc=50.0, client=client)
    assert result["status"] == "failed"

    # Ensure no pending fields were stamped after a failure
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT pending_topup_order_id FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] is None


def test_execute_topup_live_missing_token_id_fails(temp_db, monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    pid = _seed_position(temp_db)
    pos = {"id": pid, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.50, "yes_token_id": None,
           "pending_topup_order_id": None}
    client = MagicMock()

    result = execution.execute_topup(pos, add_amount_usdc=50.0, client=client)
    assert result["status"] == "failed"
    assert "token_id" in result.get("reason", "")
    client.create_and_post_order.assert_not_called()


# ===========================================================================
# Edge cases
# ===========================================================================

def test_execute_topup_zero_entry_price_returns_failed(temp_db, monkeypatch):
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    pos = {"id": 1, "contract_id": "0xabc", "side": "YES",
           "entry_price": 0.0, "yes_token_id": "0xyestoken",
           "pending_topup_order_id": None}
    result = execution.execute_topup(pos, add_amount_usdc=50.0, client=MagicMock())
    assert result["status"] == "failed"
