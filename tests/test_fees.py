"""
test_fees.py — Tests for fee/slippage accounting (Phase 7).

Covers:
  * extract_fee_amount: direct fields, derived from feeRateBps, fallbacks, edge cases
  * extract_fee_rate_bps: pulls bps from response or returns None
  * add_position_entry_fee: COALESCE accumulation across multiple calls
  * set_position_exit_fee_and_net_pnl: pnl_net = pnl - entry_fees - exit_fees
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
    size_usdc: float = 100.0,
    entry_price: float = 0.50,
    shares: float = 200.0,
) -> int:
    return db.insert_position(
        contract_id  = "0xabc",
        side         = "YES",
        size_usdc    = size_usdc,
        entry_price  = entry_price,
        entry_time   = "2026-01-01T00:00:00",
        shares       = shares,
        yes_token_id = "0xyestoken",
        is_paper     = 0,
        fill_status  = "filled",
    )


# ===========================================================================
# extract_fee_amount
# ===========================================================================

def test_extract_fee_amount_none_response():
    assert execution.extract_fee_amount(None) == 0.0
    assert execution.extract_fee_amount({}) == 0.0


def test_extract_fee_amount_direct_fee_field():
    assert execution.extract_fee_amount({"fee": "0.55"}) == pytest.approx(0.55)
    assert execution.extract_fee_amount({"fee": 0.42}) == pytest.approx(0.42)


def test_extract_fee_amount_alt_naming():
    assert execution.extract_fee_amount({"feesAccrued": "1.25"}) == pytest.approx(1.25)
    assert execution.extract_fee_amount({"feesPaid": "0.10"}) == pytest.approx(0.10)


def test_extract_fee_amount_direct_takes_precedence_over_rate():
    """If both 'fee' and 'feeRateBps' are present, the direct field wins."""
    resp = {"fee": "0.99", "feeRateBps": 200, "takingAmount": "100"}
    assert execution.extract_fee_amount(resp) == pytest.approx(0.99)


def test_extract_fee_amount_derived_from_bps_with_taking_amount():
    """200 bps = 2% of takingAmount."""
    resp = {"feeRateBps": 200, "takingAmount": "27.5"}
    assert execution.extract_fee_amount(resp) == pytest.approx(0.55)


def test_extract_fee_amount_derived_from_bps_with_caller_fill():
    """When response has rate but no takingAmount, fall back to caller's fill_amount."""
    resp = {"feeRateBps": 100}  # 1%
    assert execution.extract_fee_amount(resp, fill_amount_usdc=50.0) == pytest.approx(0.50)


def test_extract_fee_amount_taking_overrides_caller_fill():
    """Response's takingAmount is more accurate than caller's pre-fill estimate."""
    resp = {"feeRateBps": 200, "takingAmount": "100"}
    # caller passes 50 — should still use response's 100
    assert execution.extract_fee_amount(resp, fill_amount_usdc=50.0) == pytest.approx(2.0)


def test_extract_fee_amount_zero_bps_returns_zero():
    resp = {"feeRateBps": 0, "takingAmount": "100"}
    assert execution.extract_fee_amount(resp) == 0.0


def test_extract_fee_amount_bps_without_amount_returns_zero():
    """No takingAmount and no fill_amount_usdc → can't derive."""
    assert execution.extract_fee_amount({"feeRateBps": 200}) == 0.0


def test_extract_fee_amount_handles_garbage_values():
    """Should not raise on non-numeric data."""
    assert execution.extract_fee_amount({"fee": "not-a-number"}) == 0.0
    assert execution.extract_fee_amount({"feeRateBps": "abc"}) == 0.0


def test_extract_fee_amount_snake_case_rate():
    """fee_rate_bps (snake) should also work."""
    resp = {"fee_rate_bps": 200, "taking_amount": "10"}
    assert execution.extract_fee_amount(resp) == pytest.approx(0.20)


# ===========================================================================
# extract_fee_rate_bps
# ===========================================================================

def test_extract_fee_rate_bps_camel():
    assert execution.extract_fee_rate_bps({"feeRateBps": 200}) == 200


def test_extract_fee_rate_bps_snake():
    assert execution.extract_fee_rate_bps({"fee_rate_bps": 100}) == 100


def test_extract_fee_rate_bps_none_or_missing():
    assert execution.extract_fee_rate_bps(None) is None
    assert execution.extract_fee_rate_bps({}) is None


def test_extract_fee_rate_bps_garbage_returns_none():
    assert execution.extract_fee_rate_bps({"feeRateBps": "abc"}) is None


# ===========================================================================
# add_position_entry_fee
# ===========================================================================

def test_add_position_entry_fee_initial(temp_db):
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.55)
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT entry_fees FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.55)


def test_add_position_entry_fee_accumulates(temp_db):
    """Multiple calls (initial buy + top-up) should add up."""
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.50)
    db.add_position_entry_fee(pid, 0.25)
    db.add_position_entry_fee(pid, 0.10)
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT entry_fees FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.85)


def test_add_position_entry_fee_zero_is_noop(temp_db):
    """Zero/negative fees should not write anything (avoid NULL→0 churn)."""
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.0)
    db.add_position_entry_fee(pid, -1.0)
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT entry_fees FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    # Either NULL or 0 — anything but a positive accumulation is fine
    assert (row[0] is None) or (row[0] == 0.0)


def test_add_position_entry_fee_handles_null_baseline(temp_db):
    """COALESCE must let the first call work on a row that pre-dates the column."""
    pid = _seed_position(temp_db)
    # Force the column to NULL to simulate a legacy row
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE positions SET entry_fees = NULL WHERE id=?", (pid,))
    conn.commit(); conn.close()

    db.add_position_entry_fee(pid, 0.30)
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT entry_fees FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.30)


# ===========================================================================
# set_position_exit_fee_and_net_pnl
# ===========================================================================

def test_set_position_exit_fee_computes_net_pnl(temp_db):
    """net = gross - entry_fees - exit_fees."""
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.50)

    # Manually set gross pnl (normally written by update_position_exit_filled)
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE positions SET pnl = ? WHERE id=?", (10.0, pid))
    conn.commit(); conn.close()

    db.set_position_exit_fee_and_net_pnl(pid, 0.40)

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT pnl, entry_fees, exit_fees, pnl_net FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(10.0)
    assert row[1] == pytest.approx(0.50)
    assert row[2] == pytest.approx(0.40)
    assert row[3] == pytest.approx(10.0 - 0.50 - 0.40)


def test_set_position_exit_fee_with_no_entry_fees(temp_db):
    """Works even when no entry fees were captured (NULL → treated as 0)."""
    pid = _seed_position(temp_db)
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "UPDATE positions SET pnl = ?, entry_fees = NULL WHERE id=?", (5.0, pid)
    )
    conn.commit(); conn.close()

    db.set_position_exit_fee_and_net_pnl(pid, 0.25)

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT pnl_net FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(4.75)


def test_set_position_exit_fee_with_negative_pnl(temp_db):
    """Net P&L on a losing trade is even more negative after fees."""
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.30)
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE positions SET pnl = ? WHERE id=?", (-3.0, pid))
    conn.commit(); conn.close()

    db.set_position_exit_fee_and_net_pnl(pid, 0.20)

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT pnl_net FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(-3.50)


def test_set_position_exit_fee_with_null_pnl(temp_db):
    """If pnl is NULL (shouldn't happen in practice, but defensive), pnl_net
    should still be computed as -fees rather than crash."""
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.10)
    # pnl is NULL by default on insert
    db.set_position_exit_fee_and_net_pnl(pid, 0.15)
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT pnl_net FROM positions WHERE id=?", (pid,)).fetchone()
    conn.close()
    assert row[0] == pytest.approx(-0.25)


def test_set_position_exit_fee_zero_fee(temp_db):
    """A zero exit fee should still set the net properly."""
    pid = _seed_position(temp_db)
    db.add_position_entry_fee(pid, 0.20)
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE positions SET pnl = ? WHERE id=?", (2.0, pid))
    conn.commit(); conn.close()

    db.set_position_exit_fee_and_net_pnl(pid, 0.0)
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT exit_fees, pnl_net FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.0)
    assert row[1] == pytest.approx(1.80)
