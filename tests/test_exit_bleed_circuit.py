"""
test_exit_bleed_circuit.py — Tests for the bleed circuit-breaker in
monitor._advance_exit_ladders and the fast-cycle wrapper
monitor.run_exit_ladder_fast (added 2026-04-30 to fix the unfillable-
patient-rung bleed bug).

The bleed circuit forces an immediate cross-spread when the live best
bid has fallen more than EXIT_BLEED_CROSS_PCT (default 15%) below the
original trigger price.  This stops the "rung 3 limit at $0.40 sitting
unfilled while bid is $0.20" pattern that bleeds the position for hours.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
    """Disable PAPER_TRADE so _advance_exit_ladders runs the live path."""
    monkeypatch.setattr(monitor, "PAPER_TRADE", False)
    yield


def _seed_exiting_position(
    *,
    pid_seed: int = None,
    intended_exit_price: float = 0.40,
    retry_count: int = 1,
    yes_token_id: str = "tok_yes_abc",
) -> int:
    """Insert a live position already in 'exiting' status with an exit
    order id stamped on it, simulating a stop-loss that already fired."""
    pid = db.insert_position(
        contract_id="0xabc", side="YES",
        size_usdc=10.0, entry_price=0.50,
        entry_time="2026-04-30T12:00:00",
        order_id="0xord_entry",
        target_size_usdc=10.0, shares=20.0,
        yes_token_id=yes_token_id, is_paper=0, fill_status="filled",
    )
    # Manually transition into 'exiting' state with the relevant fields.
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("""
        UPDATE positions
        SET status              = 'exiting',
            exit_intended_price = ?,
            exit_order_id       = 'existing_exit_order',
            exit_retry_count    = ?
        WHERE id = ?
    """, (intended_exit_price, retry_count, pid))
    conn.commit(); conn.close()
    return pid


# ===========================================================================
# Bleed circuit-breaker triggers correctly
# ===========================================================================

def test_bleed_circuit_fires_when_bid_below_threshold(temp_db, live_mode, monkeypatch):
    """Bid 0.20 vs trigger 0.40 = 50% below.  EXIT_BLEED_CROSS_PCT is
    0.15 (15%).  Bleed circuit must fire → cross-spread, not rung advance."""
    monkeypatch.setattr(config, "EXIT_BLEED_CROSS_PCT", 0.15)
    pid = _seed_exiting_position(intended_exit_price=0.40, retry_count=1)

    captured = {}

    def fake_get_best_bid(client, token_id):
        return 0.20

    def fake_cancel_exit_order(pos, client):
        return True

    def fake_execute_exit(**kwargs):
        captured["retry_count"]  = kwargs.get("retry_count")
        captured["cross_spread"] = kwargs.get("cross_spread", False)
        captured["exit_reason"]  = kwargs.get("exit_reason", "")
        return {"status": "exit_pending", "position_id": kwargs["position"]["id"]}

    monkeypatch.setattr("execution._get_best_bid", fake_get_best_bid)
    monkeypatch.setattr("execution.cancel_exit_order", fake_cancel_exit_order)
    monkeypatch.setattr("execution.execute_exit", fake_execute_exit)

    advanced = monitor._advance_exit_ladders(client=MagicMock())
    assert advanced == 1
    assert captured["cross_spread"] is True
    assert "bleed_cross" in captured["exit_reason"]


def test_bleed_circuit_does_not_fire_when_bid_above_threshold(temp_db, live_mode, monkeypatch):
    """Bid 0.36 vs trigger 0.40 = 10% below.  Below the 15% threshold,
    so normal rung advance, not cross-spread."""
    monkeypatch.setattr(config, "EXIT_BLEED_CROSS_PCT", 0.15)
    pid = _seed_exiting_position(intended_exit_price=0.40, retry_count=1)

    captured = {}

    def fake_get_best_bid(client, token_id):
        return 0.36   # only 10% below trigger

    def fake_cancel_exit_order(pos, client):
        return True

    def fake_execute_exit(**kwargs):
        captured["retry_count"]  = kwargs.get("retry_count")
        captured["cross_spread"] = kwargs.get("cross_spread", False)
        return {"status": "exit_pending", "position_id": kwargs["position"]["id"]}

    monkeypatch.setattr("execution._get_best_bid", fake_get_best_bid)
    monkeypatch.setattr("execution.cancel_exit_order", fake_cancel_exit_order)
    monkeypatch.setattr("execution.execute_exit", fake_execute_exit)

    advanced = monitor._advance_exit_ladders(client=MagicMock())
    assert advanced == 1
    assert captured["cross_spread"] is False
    assert captured["retry_count"] == 2  # normal rung advance: 1 → 2


def test_bleed_circuit_skipped_when_bid_unavailable(temp_db, live_mode, monkeypatch):
    """If _get_best_bid returns None (book unreachable), skip the bleed
    check rather than blindly forcing cross-spread.  Defensive fallback."""
    monkeypatch.setattr(config, "EXIT_BLEED_CROSS_PCT", 0.15)
    pid = _seed_exiting_position(intended_exit_price=0.40, retry_count=1)

    captured = {}

    def fake_get_best_bid(client, token_id):
        return None   # API failure / empty book

    def fake_cancel_exit_order(pos, client):
        return True

    def fake_execute_exit(**kwargs):
        captured["cross_spread"] = kwargs.get("cross_spread", False)
        return {"status": "exit_pending", "position_id": kwargs["position"]["id"]}

    monkeypatch.setattr("execution._get_best_bid", fake_get_best_bid)
    monkeypatch.setattr("execution.cancel_exit_order", fake_cancel_exit_order)
    monkeypatch.setattr("execution.execute_exit", fake_execute_exit)

    advanced = monitor._advance_exit_ladders(client=MagicMock())
    # Bleed didn't fire → normal rung advance behavior
    assert advanced == 1
    assert captured["cross_spread"] is False


def test_bleed_circuit_disabled_when_pct_is_one(temp_db, live_mode, monkeypatch):
    """EXIT_BLEED_CROSS_PCT = 1.0 disables the bleed circuit entirely.
    Even a 99% drop wouldn't trigger force-cross."""
    monkeypatch.setattr(config, "EXIT_BLEED_CROSS_PCT", 1.0)
    pid = _seed_exiting_position(intended_exit_price=0.40, retry_count=1)

    captured = {}

    def fake_cancel_exit_order(pos, client):
        return True

    def fake_execute_exit(**kwargs):
        captured["cross_spread"] = kwargs.get("cross_spread", False)
        return {"status": "exit_pending", "position_id": kwargs["position"]["id"]}

    monkeypatch.setattr("execution.cancel_exit_order", fake_cancel_exit_order)
    monkeypatch.setattr("execution.execute_exit", fake_execute_exit)

    # Note: with EXIT_BLEED_CROSS_PCT = 1.0, the bid lookup is skipped
    # entirely (see monitor.py); confirm by NOT patching _get_best_bid.
    advanced = monitor._advance_exit_ladders(client=MagicMock())
    assert advanced == 1
    assert captured["cross_spread"] is False


def test_bleed_circuit_at_threshold_exactly_does_not_fire(temp_db, live_mode, monkeypatch):
    """bid = trigger × (1 - threshold) is the boundary.  Strict less-than
    means the boundary case does NOT force cross-spread — leaves room for
    the normal rung path."""
    monkeypatch.setattr(config, "EXIT_BLEED_CROSS_PCT", 0.15)
    pid = _seed_exiting_position(intended_exit_price=0.40, retry_count=1)

    captured = {}

    def fake_get_best_bid(client, token_id):
        return 0.40 * (1 - 0.15)   # exactly at threshold = 0.34

    def fake_cancel_exit_order(pos, client):
        return True

    def fake_execute_exit(**kwargs):
        captured["cross_spread"] = kwargs.get("cross_spread", False)
        return {"status": "exit_pending", "position_id": kwargs["position"]["id"]}

    monkeypatch.setattr("execution._get_best_bid", fake_get_best_bid)
    monkeypatch.setattr("execution.cancel_exit_order", fake_cancel_exit_order)
    monkeypatch.setattr("execution.execute_exit", fake_execute_exit)

    advanced = monitor._advance_exit_ladders(client=MagicMock())
    assert captured["cross_spread"] is False  # exactly-at threshold = normal advance


# ===========================================================================
# Fast-cycle wrapper
# ===========================================================================

def test_run_exit_ladder_fast_paper_mode_noop(monkeypatch):
    """In paper mode, the fast wrapper is a no-op."""
    monkeypatch.setattr(monitor, "PAPER_TRADE", True)
    assert monitor.run_exit_ladder_fast() == 0


def test_run_exit_ladder_fast_no_client_noop(temp_db, live_mode, monkeypatch):
    """If the CLOB client can't be obtained (e.g., creds missing), wrapper
    returns 0 without raising."""
    monkeypatch.setattr("execution.get_clob_client", lambda: None)
    assert monitor.run_exit_ladder_fast() == 0


def test_run_exit_ladder_fast_calls_advance(temp_db, live_mode, monkeypatch):
    """Happy path: wrapper forwards to _advance_exit_ladders with the
    fetched client."""
    fake_client = MagicMock()
    monkeypatch.setattr("execution.get_clob_client", lambda: fake_client)

    captured = {}

    def fake_advance(client):
        captured["client_passed"] = client
        return 7   # arbitrary count

    monkeypatch.setattr(monitor, "_advance_exit_ladders", fake_advance)
    assert monitor.run_exit_ladder_fast() == 7
    assert captured["client_passed"] is fake_client


def test_run_exit_ladder_fast_swallows_exceptions(temp_db, live_mode, monkeypatch):
    """If _advance_exit_ladders raises, the wrapper logs and returns 0
    rather than letting the scheduler job crash."""
    monkeypatch.setattr("execution.get_clob_client", lambda: MagicMock())

    def raises(client):
        raise RuntimeError("boom")

    monkeypatch.setattr(monitor, "_advance_exit_ladders", raises)
    assert monitor.run_exit_ladder_fast() == 0


# ===========================================================================
# Concurrency lock — second concurrent invocation no-ops
# ===========================================================================

def test_advance_exit_ladders_lock_prevents_race(temp_db, live_mode, monkeypatch):
    """If the lock is already held (e.g., by the hourly monitor at :40
    when the */5 fast cycle also fires), a second concurrent call short-
    circuits to 0 instead of double-cancelling the same orders."""
    pid = _seed_exiting_position()

    # Pre-acquire the lock to simulate another thread already inside.
    monitor._advance_exit_ladders_lock.acquire()
    try:
        result = monitor._advance_exit_ladders(client=MagicMock())
        assert result == 0
    finally:
        monitor._advance_exit_ladders_lock.release()
