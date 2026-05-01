"""
test_stale_topup_refresh.py — Tests for the */5 min stale-topup
re-pricing routine (Lightweight Option B from the topup-WS analysis).

When a pending topup's limit price drifts more than
TOPUP_REPRICE_THRESHOLD_CENTS below the live best_ask, the routine
cancels the stale order and re-issues at the fresh price.  Direction-
aware: only fires on UPWARD ask drift (asks moved away from us).
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


def _seed_with_pending_topup(
    *,
    entry_price: float = 0.30,
    pending_limit: float = 0.30,
    pending_amount: float = 5.0,
    target: float = 10.0,
    size_usdc: float = 5.0,
    shares: float = 16.66,
    topup_oid: str = "0xpending_topup",
    is_paper: int = 0,
) -> int:
    pid = db.insert_position(
        contract_id="0xmarket", side="YES",
        size_usdc=size_usdc, entry_price=entry_price,
        entry_time="2026-05-01T00:00:00",
        order_id="0xord_entry", target_size_usdc=target, shares=shares,
        yes_token_id="tok_yes", is_paper=is_paper, fill_status="filled",
    )
    db.update_position_topup_pending(
        position_id    = pid,
        order_id       = topup_oid,
        amount_usdc    = pending_amount,
        intended_price = pending_limit,
    )
    db.insert_position_order(
        position_id     = pid,
        order_id        = topup_oid,
        role            = "topup",
        intended_usdc   = pending_amount,
        intended_shares = pending_amount / pending_limit,
        limit_price     = pending_limit,
        status          = "pending",
    )
    return pid


# ===========================================================================
# Threshold + direction logic
# ===========================================================================

def test_no_drift_does_not_refresh(temp_db, live_mode, monkeypatch):
    """When the live ask matches our limit (drift = 0¢), no action."""
    pid = _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)

    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.30, "best_bid": 0.29, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.30, 100)],
                          "bids_sorted_desc": [(0.29, 100)]},
    )
    cancel_calls = []
    monkeypatch.setattr("execution.cancel_order", lambda *a, **k: cancel_calls.append(a) or True)

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0
    assert cancel_calls == []


def test_drift_below_threshold_no_action(temp_db, live_mode, monkeypatch):
    """Ask drifted up 1¢ but threshold is 1.5¢ — no action."""
    pid = _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)

    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.31, "best_bid": 0.30, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.31, 100)],
                          "bids_sorted_desc": [(0.30, 100)]},
    )
    cancel_calls = []
    monkeypatch.setattr("execution.cancel_order", lambda *a, **k: cancel_calls.append(a) or True)

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0
    assert cancel_calls == []


def test_drift_above_threshold_triggers_refresh(temp_db, live_mode, monkeypatch):
    """Ask moved 2¢ above our limit; threshold is 1.5¢ — must refresh."""
    pid = _seed_with_pending_topup(pending_limit=0.30, target=10.0,
                                    size_usdc=5.0)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)

    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.32, "best_bid": 0.31, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.32, 100)],
                          "bids_sorted_desc": [(0.31, 100)]},
    )
    monkeypatch.setattr(
        "execution.cancel_order",
        lambda oid, client: True,
    )
    placed = {}
    def fake_execute_topup(pos, amount, client=None):
        placed["pid"]    = pos["id"]
        placed["amount"] = amount
        return {"status": "placed", "limit_price": 0.33,
                "order_id": "0xfresh_oid", "add_usdc": amount}
    monkeypatch.setattr("execution.execute_topup", fake_execute_topup)

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 1
    assert placed["pid"] == pid
    # Gap = target ($10) - committed (cancelled topup no longer counts) ≈ $5
    assert placed["amount"] == pytest.approx(5.0, abs=0.5)


def test_downward_drift_does_not_refresh(temp_db, live_mode, monkeypatch):
    """Ask moved DOWN below our limit — Polymarket would fill us at the
    cheaper price by default; no need to cancel/refresh."""
    pid = _seed_with_pending_topup(pending_limit=0.32)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)

    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.28, "best_bid": 0.27, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.28, 100)],
                          "bids_sorted_desc": [(0.27, 100)]},
    )
    cancel_calls = []
    monkeypatch.setattr("execution.cancel_order", lambda *a, **k: cancel_calls.append(a) or True)

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0
    assert cancel_calls == []


def test_drift_exactly_at_threshold_no_action(temp_db, live_mode, monkeypatch):
    """Drift = threshold exactly (boundary case).  Strict greater-than:
    1.5¢ exactly does NOT trigger."""
    pid = _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)

    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.315, "best_bid": 0.30, "spread_cents": 1.5,
                          "asks_sorted_asc": [(0.315, 100)],
                          "bids_sorted_desc": [(0.30, 100)]},
    )
    cancel_calls = []
    monkeypatch.setattr("execution.cancel_order", lambda *a, **k: cancel_calls.append(a) or True)

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0
    assert cancel_calls == []


# ===========================================================================
# Defensive cases
# ===========================================================================

def test_threshold_zero_disables_routine(temp_db, live_mode, monkeypatch):
    pid = _seed_with_pending_topup(pending_limit=0.20)  # massive drift
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 0.0)
    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.50, "best_bid": 0.49, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.50, 100)],
                          "bids_sorted_desc": [(0.49, 100)]},
    )
    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0


def test_no_book_data_skips(temp_db, live_mode, monkeypatch):
    """When orderbook fetch fails (transient API), leave alone for next cycle."""
    _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)
    monkeypatch.setattr("execution.get_orderbook_snapshot",
                        lambda *a, **k: None)
    cancel_calls = []
    monkeypatch.setattr("execution.cancel_order",
                        lambda *a, **k: cancel_calls.append(a) or True)

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0
    assert cancel_calls == []


def test_no_best_ask_in_book_skips(temp_db, live_mode, monkeypatch):
    """Empty ask side — can't compute drift, so leave alone."""
    _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)
    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": None, "best_bid": 0.29, "spread_cents": None,
                          "asks_sorted_asc": [],
                          "bids_sorted_desc": [(0.29, 100)]},
    )
    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0


def test_paper_position_skipped(temp_db, live_mode, monkeypatch):
    """Paper positions never have CLOB orders to refresh."""
    _seed_with_pending_topup(pending_limit=0.30, is_paper=1)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)
    # If get_orderbook_snapshot is called, the test fails — paper rows
    # should be filtered out by the SQL WHERE clause before any CLOB call.
    def boom(*a, **k):
        raise AssertionError("paper position should not trigger book fetch")
    monkeypatch.setattr("execution.get_orderbook_snapshot", boom)
    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0


def test_paper_mode_global_skipped(temp_db, monkeypatch):
    monkeypatch.setattr(monitor, "PAPER_TRADE", True)
    _seed_with_pending_topup()
    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0


def test_no_client_returns_zero(temp_db, live_mode):
    _seed_with_pending_topup()
    n = monitor.refresh_stale_topups(client=None)
    assert n == 0


def test_no_pending_topups_returns_zero(temp_db, live_mode):
    db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=5.0, entry_price=0.30,
        entry_time="2026-05-01T00:00:00", target_size_usdc=10.0, shares=16.66,
        is_paper=0, fill_status="filled",
    )
    # No pending topup pointer
    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0


# ===========================================================================
# Cancel-failure handling
# ===========================================================================

def test_cancel_failure_leaves_stale_in_place(temp_db, live_mode, monkeypatch):
    """If cancel_order returns False (CLOB rejected the cancel — usually
    because the order already filled or doesn't exist), don't try to
    place a fresh one.  Pointer stays set; next /5 min cycle will retry
    or the orphan-cleanup will detect it."""
    pid = _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)
    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.40, "best_bid": 0.39, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.40, 100)],
                          "bids_sorted_desc": [(0.39, 100)]},
    )
    monkeypatch.setattr("execution.cancel_order", lambda *a, **k: False)
    place_calls = []
    monkeypatch.setattr("execution.execute_topup",
                        lambda *a, **k: place_calls.append(a) or {"status": "placed"})

    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0
    assert place_calls == []   # never attempted re-issue


def test_reissue_skipped_book_too_thin_no_count(temp_db, live_mode, monkeypatch):
    """If the cancel succeeds but execute_topup returns 'skip' (book too
    thin to re-place at any usable size), refreshed count stays 0 but
    pointer is correctly cleared (cancel did its job)."""
    pid = _seed_with_pending_topup(pending_limit=0.30)
    monkeypatch.setattr(config, "TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)
    monkeypatch.setattr(
        "execution.get_orderbook_snapshot",
        lambda *a, **k: {"best_ask": 0.40, "best_bid": 0.39, "spread_cents": 1.0,
                          "asks_sorted_asc": [(0.40, 1)],
                          "bids_sorted_desc": [(0.39, 100)]},
    )
    # Mock cancel_order to simulate the patched behavior: clear pending pointer
    def fake_cancel(oid, client):
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute(
                "UPDATE positions SET pending_topup_order_id=NULL, "
                "pending_topup_amount_usdc=NULL, pending_topup_intended_price=NULL "
                "WHERE pending_topup_order_id=?", (oid,),
            )
        return True
    monkeypatch.setattr("execution.cancel_order", fake_cancel)
    monkeypatch.setattr(
        "execution.execute_topup",
        lambda *a, **k: {"status": "skip", "reason": "book_too_thin"},
    )
    n = monitor.refresh_stale_topups(client=MagicMock())
    assert n == 0   # not counted as 'refreshed' — no new order placed


# ===========================================================================
# Concurrency lock
# ===========================================================================

def test_concurrency_lock_prevents_double_run(temp_db, live_mode):
    """Pre-acquire the lock to simulate another invocation in flight; the
    second call short-circuits to 0."""
    monitor._refresh_stale_topups_lock.acquire()
    try:
        n = monitor.refresh_stale_topups(client=MagicMock())
        assert n == 0
    finally:
        monitor._refresh_stale_topups_lock.release()


# ===========================================================================
# Fast-cycle wrapper
# ===========================================================================

def test_run_stale_topup_refresh_fast_paper_mode(monkeypatch):
    monkeypatch.setattr(monitor, "PAPER_TRADE", True)
    assert monitor.run_stale_topup_refresh_fast() == 0


def test_run_stale_topup_refresh_fast_no_client(temp_db, live_mode, monkeypatch):
    monkeypatch.setattr("execution.get_clob_client", lambda: None)
    assert monitor.run_stale_topup_refresh_fast() == 0


def test_run_stale_topup_refresh_fast_swallows_exceptions(temp_db, live_mode, monkeypatch):
    monkeypatch.setattr("execution.get_clob_client", lambda: MagicMock())
    monkeypatch.setattr(monitor, "refresh_stale_topups",
                        lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    assert monitor.run_stale_topup_refresh_fast() == 0
