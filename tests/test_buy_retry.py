"""
test_buy_retry.py — Tests for the buy-retry cap (Phase 5).

The retry "logic" itself is implicit (cancel → next-scan signal → fresh buy
flows naturally through the existing code).  What we test here is the
SOFT CAP that prevents thrash on contracts whose book never reaches our
limit:

  * get_recent_cancelled_count()  — counts cancelled BUYS only, within window
  * execute_signal()              — gates with status='skip' when cap is hit
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Spin up a fresh sqlite DB for each test, isolated from production."""
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    # Patch the config too in case anything reads it directly
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _insert_position(
    db_path: str, *, contract_id: str, status: str,
    fill_status: str, entry_time: str,
) -> int:
    """Lightweight helper — only sets the columns this test cares about."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "INSERT INTO positions (contract_id, side, size_usdc, entry_price, "
        "entry_time, status, fill_status) "
        "VALUES (?, 'YES', 100.0, 0.50, ?, ?, ?)",
        (contract_id, entry_time, status, fill_status),
    )
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid


# ===========================================================================
# get_recent_cancelled_count — coverage of the SQL filters
# ===========================================================================

def test_no_history_returns_zero(temp_db):
    assert db.get_recent_cancelled_count("0xnewcontract") == 0


def test_one_recent_cancellation_counts(temp_db):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_position(
        temp_db, contract_id="0xabc",
        status="closed", fill_status="cancelled",
        entry_time=now.isoformat().replace("+00:00", ""),
    )
    assert db.get_recent_cancelled_count("0xabc") == 1


def test_filled_position_does_not_count(temp_db):
    """A filled buy is not a cancellation — should not decrement the budget."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_position(
        temp_db, contract_id="0xabc",
        status="open", fill_status="filled",
        entry_time=now.isoformat().replace("+00:00", ""),
    )
    assert db.get_recent_cancelled_count("0xabc") == 0


def test_closed_for_other_reason_does_not_count(temp_db):
    """status='closed' WITHOUT fill_status='cancelled' (e.g., normal exit)
    should not consume retry budget."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_position(
        temp_db, contract_id="0xabc",
        status="closed", fill_status="filled",
        entry_time=now.isoformat().replace("+00:00", ""),
    )
    assert db.get_recent_cancelled_count("0xabc") == 0


def test_old_cancellations_excluded_by_window(temp_db):
    """Cancellations older than the window don't count."""
    old = (datetime.now(timezone.utc) - timedelta(hours=12)).replace(microsecond=0)
    _insert_position(
        temp_db, contract_id="0xabc",
        status="closed", fill_status="cancelled",
        entry_time=old.isoformat().replace("+00:00", ""),
    )
    # Default window 6h → old (12h ago) excluded
    assert db.get_recent_cancelled_count("0xabc", within_hours=6) == 0
    # Wider window 24h → old included
    assert db.get_recent_cancelled_count("0xabc", within_hours=24) == 1


def test_different_contracts_dont_pollute_each_other(temp_db):
    """A cancelled buy on contract A should not count toward contract B."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    _insert_position(
        temp_db, contract_id="0xaaa",
        status="closed", fill_status="cancelled",
        entry_time=now.isoformat().replace("+00:00", ""),
    )
    assert db.get_recent_cancelled_count("0xaaa") == 1
    assert db.get_recent_cancelled_count("0xbbb") == 0


def test_multiple_recent_cancellations_sum(temp_db):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for _ in range(5):
        _insert_position(
            temp_db, contract_id="0xabc",
            status="closed", fill_status="cancelled",
            entry_time=now.isoformat().replace("+00:00", ""),
        )
    assert db.get_recent_cancelled_count("0xabc") == 5


# ===========================================================================
# execute_signal — retry cap behavior
# ===========================================================================

@pytest.fixture
def mock_clob_client():
    """Mock CLOB client whose create_and_post_order returns a successful fill."""
    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success": True,
        "orderID": "0xorder123",
        "status": "matched",
        "takingAmount": "55.0",
        "makingAmount": "100.0",
    }
    return client


def _signal(contract_id: str = "0xabc"):
    return {
        "contract_id":      contract_id,
        "recommended_side": "YES",
        "kelly_size":       100.0,
        "yes_token_id":     "0xtoken",
        "no_token_id":      "0xnotoken",
        "market_p":         0.50,
        "yes_price":        0.50,
        "scan_timestamp":   "2026-01-01T00:00:00Z",
        "city":             "Chicago",
        "date":             "2026-01-01",
    }


def test_cap_blocks_after_3_cancellations(temp_db, mock_clob_client, monkeypatch):
    """Three cancelled buys in last 6h → next attempt returns status='skip'."""
    import execution
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for _ in range(3):
        _insert_position(
            temp_db, contract_id="0xabc",
            status="closed", fill_status="cancelled",
            entry_time=now.isoformat().replace("+00:00", ""),
        )
    result = execution.execute_signal(_signal("0xabc"), client=mock_clob_client)
    assert result["status"] == "skip"
    assert result["reason"] == "retry_exhausted"
    assert result["cancelled_count"] == 3
    # No order was placed
    mock_clob_client.create_and_post_order.assert_not_called()


def test_below_cap_proceeds(temp_db, mock_clob_client, monkeypatch):
    """Two prior cancellations → still allowed to retry (cap is 3)."""
    import execution
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    monkeypatch.setattr(execution, "EXIT_HARD_STOP_PCT", -0.30)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for _ in range(2):
        _insert_position(
            temp_db, contract_id="0xabc",
            status="closed", fill_status="cancelled",
            entry_time=now.isoformat().replace("+00:00", ""),
        )
    result = execution.execute_signal(_signal("0xabc"), client=mock_clob_client)
    # Should NOT be a skip — order should have been placed
    assert result["status"] != "skip"
    mock_clob_client.create_and_post_order.assert_called_once()


def test_paper_mode_ignores_cap(temp_db, mock_clob_client, monkeypatch):
    """Paper mode never has cancellations from CLOB, so the cap is irrelevant."""
    import execution
    monkeypatch.setattr(execution, "PAPER_TRADE", True)
    monkeypatch.setattr(execution, "EXIT_HARD_STOP_PCT", -0.30)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    # Even with 100 fake cancellations, paper mode bypasses the count
    for _ in range(100):
        _insert_position(
            temp_db, contract_id="0xabc",
            status="closed", fill_status="cancelled",
            entry_time=now.isoformat().replace("+00:00", ""),
        )
    result = execution.execute_signal(_signal("0xabc"), client=None)
    assert result["status"] == "paper"
