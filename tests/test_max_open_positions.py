"""
test_max_open_positions.py — Regression tests for the MAX_OPEN_POSITIONS
hard count cap.

Why this test exists
--------------------
Added 2026-04-29 as a guard while easing into live trading.  The cap
limits the bot to N simultaneously-open positions (per mode).  Without
the test, "I removed that check during a refactor" could silently
remove the cap and let the bot pile on every signal it sees.

Counts include pending buys and exiting positions (anything that
consumes or could consume capital), but exclude top-ups (they merge
into existing positions).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import risk


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
    contract_id: str = "0xabc",
    is_paper: int = 0,
    fill_status: str = "filled",
    status_override: str | None = None,
) -> int:
    pid = db.insert_position(
        contract_id  = contract_id,
        side         = "YES",
        size_usdc    = 50.0,
        entry_price  = 0.50,
        entry_time   = "2026-01-01T00:00:00",
        shares       = 100.0,
        yes_token_id = "tok_yes",
        is_paper     = is_paper,
        fill_status  = fill_status,
    )
    if status_override:
        import sqlite3 as _s
        conn = _s.connect(db.DB_PATH)
        conn.execute("UPDATE positions SET status=? WHERE id=?",
                     (status_override, pid))
        conn.commit(); conn.close()
    return pid


# ===========================================================================
# Happy paths
# ===========================================================================

def test_check_passes_when_below_cap(temp_db, monkeypatch):
    """3 open positions, cap of 5 → trade allowed."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 5)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(3):
        _seed(contract_id=f"0x{i}", is_paper=0)
    result = risk.check_open_position_count()
    assert result.passed is True


def test_check_blocks_at_exact_cap(temp_db, monkeypatch):
    """5 open positions, cap of 5 → 6th trade blocked.

    The check is `len(relevant) >= MAX_OPEN_POSITIONS`, so reaching the
    cap (not exceeding it) blocks the next entry."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 5)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(5):
        _seed(contract_id=f"0x{i}", is_paper=0)
    result = risk.check_open_position_count()
    assert result.passed is False
    assert "MAX_OPEN_POSITIONS cap reached" in result.reason
    assert "5 of 5" in result.reason


def test_check_blocks_above_cap(temp_db, monkeypatch):
    """7 open positions, cap of 5 → blocked, count reflected in reason."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 5)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(7):
        _seed(contract_id=f"0x{i}", is_paper=0)
    result = risk.check_open_position_count()
    assert result.passed is False
    assert "7 of 5" in result.reason


def test_zero_cap_is_unlimited(temp_db, monkeypatch):
    """MAX_OPEN_POSITIONS=0 means unlimited — never blocks."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 0)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(50):
        _seed(contract_id=f"0x{i}", is_paper=0)
    result = risk.check_open_position_count()
    assert result.passed is True


# ===========================================================================
# What counts (and what doesn't)
# ===========================================================================

def test_pending_buys_count(temp_db, monkeypatch):
    """Pending buys (fill_status='pending') consume the budget — they
    must count against the cap to prevent piling on entries before
    the first ones confirm."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 3)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(3):
        _seed(contract_id=f"0x{i}", is_paper=0, fill_status="pending")
    result = risk.check_open_position_count()
    assert result.passed is False


def test_exiting_positions_count(temp_db, monkeypatch):
    """status='exiting' positions are still on chain — must count."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 3)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(3):
        _seed(contract_id=f"0x{i}", is_paper=0, status_override="exiting")
    result = risk.check_open_position_count()
    assert result.passed is False


def test_cancelled_positions_dont_count(temp_db, monkeypatch):
    """Cancelled rows (fill_status='cancelled') don't represent any
    capital exposure — they should NOT count."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 3)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(5):
        _seed(contract_id=f"0x{i}", is_paper=0, fill_status="cancelled",
              status_override="closed")
    # Only 2 currently-active positions; under the cap
    _seed(contract_id="0xa", is_paper=0)
    _seed(contract_id="0xb", is_paper=0)
    result = risk.check_open_position_count()
    assert result.passed is True


def test_paper_and_live_counted_separately(temp_db, monkeypatch):
    """In live mode, paper positions don't consume the live cap (and
    vice versa) — matches the rest of the portfolio guards."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 3)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    # 5 paper positions
    for i in range(5):
        _seed(contract_id=f"0xp{i}", is_paper=1)
    # 0 live → live cap not exhausted
    result = risk.check_open_position_count()
    assert result.passed is True


def test_paper_mode_only_counts_paper(temp_db, monkeypatch):
    """Inverse of the above — in paper mode, live positions don't count."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 3)
    monkeypatch.setattr(risk, "PAPER_TRADE", True)
    for i in range(5):
        _seed(contract_id=f"0xl{i}", is_paper=0)
    result = risk.check_open_position_count()
    assert result.passed is True


# ===========================================================================
# Integration with run_portfolio_checks
# ===========================================================================

def test_run_portfolio_checks_includes_count_cap(temp_db, monkeypatch):
    """The cap must be wired into run_portfolio_checks — otherwise it's
    inert.  This pins the integration so a refactor that drops the
    check from the list will fail this test."""
    monkeypatch.setattr(risk, "MAX_OPEN_POSITIONS", 2)
    monkeypatch.setattr(risk, "PAPER_TRADE", False)
    for i in range(2):
        _seed(contract_id=f"0x{i}", is_paper=0)

    signal = {
        "kelly_size":       1.0,
        "city":             "Chicago",
        "date":             "2026-04-09",
        "event_id":         "ev_test",
        "recommended_side": "YES",
        "contract_id":      "0xnew",
    }
    passed, failures = risk.run_portfolio_checks(signal, bankroll=200.0)
    assert passed is False
    assert any("MAX_OPEN_POSITIONS" in f for f in failures), (
        f"expected MAX_OPEN_POSITIONS failure in {failures!r}"
    )
