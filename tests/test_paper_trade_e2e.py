"""
test_paper_trade_e2e.py — End-to-end trade lifecycle integration tests.

These exercise the *integration* between execution, fill_handler, monitor
settlement, fee accounting, and activity logging — the kind of bug
unit tests miss because each module passes individually but the wiring
between them is broken.

Two scenarios:

  1. PAPER trade lifecycle:
       signal → execute_signal (paper) → DB row exists with fill_status='filled'
                                       → activity_log has BUY entry
       → market resolves YES → monitor._settle_resolved_positions runs
                            → DB row status='closed', pnl > 0
                            → activity_log has CLOSE entry

  2. LIVE trade lifecycle:
       signal → execute_signal (live, mock CLOB) → DB row pending, trade_status='matched'
       → WS sends 'confirmed' trade event → fill_handler.apply_trade_event
                                          → fill_status='filled', activity 'FILL'
       → execute_exit triggered → exit order placed, status='exiting'
       → WS sends 'confirmed' exit trade event → exit_filled, position closed
                                                with realized P&L net of fees

These would catch:
  * "I changed insert_position's signature; nothing else compiles in tests"
  * "I removed activity logging from execute_signal by accident"
  * "Monitor settle stopped writing pnl_net after a refactor"
  * "WS handler stopped finding positions after a column rename"
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
import fill_handler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _make_signal(
    *,
    contract_id: str = "0xabc123",
    side: str = "YES",
    yes_price: float = 0.45,
    yes_token_id: str = "tok_yes_42",
    no_token_id: str = "tok_no_42",
    kelly_size: float = 50.0,
    city: str = "Chicago",
    date: str = "2026-04-09",
    gamma_market_id: str = "1886470",
) -> dict:
    """Build a realistic signal dict — same shape strategy.generate_signals
    returns and execute_signal expects."""
    return {
        "contract_id":      contract_id,
        "recommended_side": side,
        "kelly_size":       kelly_size,
        "yes_token_id":     yes_token_id,
        "no_token_id":      no_token_id,
        "yes_price":        yes_price,
        "market_p":         yes_price,
        "city":             city,
        "date":             date,
        "gamma_market_id":  gamma_market_id,
        "question":         "Will Chicago's high temperature on April 9 be 72°F?",
        "model_prob":       0.62,
        "market_prob":      yes_price,
        "ev":               0.085,
        "edge":             0.17,
        "scan_timestamp":   "2026-04-08T20:00:00+00:00",
    }


# ===========================================================================
# Scenario 1: Paper trade lifecycle
# ===========================================================================

def test_paper_trade_full_lifecycle(temp_db, monkeypatch):
    """signal → paper buy → DB filled → market resolves → settle closes it.

    Verifies the path that 99% of operator-observable behavior runs through
    in dev: paper trades land immediately, and when the market resolves
    the monitor closes them with correct P&L."""
    monkeypatch.setattr(execution, "PAPER_TRADE", True)
    # mark_outcome_executed reads from temp_outcomes — it's a no-op when the
    # row doesn't exist; safe to skip seeding here.

    # --- Step 1: place the paper buy ---
    sig = _make_signal(yes_price=0.45, kelly_size=50.0)
    result = execution.execute_signal(sig, client=None)

    assert result["status"] == "paper"
    pid = result["position_id"]
    assert pid is not None
    assert result["entry_price"] == pytest.approx(0.45)
    # 50 USDC / 0.45 ≈ 111.1 shares
    assert result["shares"] == pytest.approx(50 / 0.45, rel=1e-3)

    # Position should be in DB, fill_status='filled' (paper mode confirms instantly)
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    assert row["fill_status"] == "filled"
    assert row["status"] == "open"
    assert row["is_paper"] == 1
    assert row["side"] == "YES"
    assert row["city"] == "Chicago"

    # Activity log should have one BUY entry tied to this position
    acts = db.get_recent_activity(categories=["BUY"], limit=10)
    assert len(acts) == 1
    assert acts[0]["position_id"] == pid
    assert "paper" in acts[0]["message"].lower()

    # --- Step 2: market resolves YES → monitor settle closes the position ---
    import monitor
    # Stub get_market_status to return closed+YES so settle closes the position
    monkeypatch.setattr(monitor, "get_market_status", lambda *a, **kw: {
        "closed":    True,
        "active":    False,
        "yes_price": 1.0,
        "no_price":  0.0,
        "winner":    "YES",
    })

    closed_count = monitor._settle_resolved_positions(data_api_index={})
    assert closed_count == 1

    # Verify the row closed correctly
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    final = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    assert final["status"] == "closed"
    assert final["exit_price"] == 1.0
    # P&L = (1.0 - 0.45) * shares = 0.55 * (50/0.45) ≈ 61.11
    assert final["pnl"] == pytest.approx((1.0 - 0.45) * (50 / 0.45), rel=1e-3)

    # CLOSE event in activity log
    closes = db.get_recent_activity(categories=["CLOSE"], limit=10)
    assert len(closes) == 1
    assert closes[0]["position_id"] == pid
    assert "WON" in closes[0]["message"]


def test_paper_trade_lifecycle_lost(temp_db, monkeypatch):
    """Same flow but YES loses → P&L is -size_usdc, CLOSE logged at WARN."""
    monkeypatch.setattr(execution, "PAPER_TRADE", True)

    sig = _make_signal(yes_price=0.45, kelly_size=50.0)
    result = execution.execute_signal(sig, client=None)
    pid = result["position_id"]

    import monitor
    monkeypatch.setattr(monitor, "get_market_status", lambda *a, **kw: {
        "closed": True, "active": False,
        "yes_price": 0.0, "no_price": 1.0, "winner": "NO",
    })

    monitor._settle_resolved_positions(data_api_index={})

    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    final = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    # P&L = (0 - 0.45) * shares = -0.45 * (50/0.45) = -50
    assert final["pnl"] == pytest.approx(-50.0, rel=1e-3)

    closes = db.get_recent_activity(categories=["CLOSE"], limit=10)
    assert closes[0]["level"] == "WARN"  # losing trade logs at WARN
    assert "LOST" in closes[0]["message"]


# ===========================================================================
# Scenario 2: Live trade lifecycle (mock CLOB client + WS event)
# ===========================================================================

def test_live_trade_full_lifecycle_via_ws(temp_db, monkeypatch):
    """live BUY → MATCHED engine response → CONFIRMED via WS → exit triggered
    → exit CONFIRMED via WS → position closed with net P&L net of fees.

    This is the path that actually runs in production.  Catches end-to-end
    breaks like: "I refactored fill_handler to take a kwarg instead of
    positional and execute_signal still passes positional"."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    # --- Step 1: live BUY placement (CLOB returns 'matched' immediately) ---
    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success":      True,
        "orderID":      "0xord_buy_1",
        "status":       "matched",
        "takingAmount": "50.0",
        "makingAmount": "111.1",  # ≈ 50 / 0.45
    }

    sig = _make_signal(yes_price=0.45, kelly_size=50.0)
    result = execution.execute_signal(sig, client=client)

    assert result["status"] == "placed"
    pid = result["position_id"]

    # Phase 9 correctness: MATCHED engine response is NOT a confirmed fill.
    # Position lands as 'pending', trade_status='matched' — awaits CONFIRMED.
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    assert row["fill_status"] == "pending"
    assert row["trade_status"] == "matched"
    assert row["order_id"] == "0xord_buy_1"

    # --- Step 2: WS sends a CONFIRMED trade event for the entry ---
    confirmed_event = {
        "id":              "trade_buy_1",
        "status":          "confirmed",
        "taker_order_id":  "0xord_buy_1",
        "size":            111.1,
        "price":           0.451,        # actual fill slightly above limit
        "feeRateBps":      200,           # 2% fee
    }
    handler_result = fill_handler.apply_trade_event(confirmed_event)
    assert handler_result["action"] == "filled"
    assert handler_result["role"] == "entry"

    # Verify position now filled, lifecycle advanced, fee captured
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    assert row["fill_status"] == "filled"
    assert row["trade_status"] == "confirmed"
    assert row["entry_price"] == pytest.approx(0.451)
    assert row["shares"] == pytest.approx(111.1)
    # Fee = 111.1 * 0.451 * 0.02 ≈ 1.00 USDC
    assert row["entry_fees"] == pytest.approx(111.1 * 0.451 * 0.02, rel=1e-2)

    # --- Step 3: trigger an exit (e.g. profit target hit) ---
    # Reload the position dict freshly so execute_exit sees current state
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    pos_dict = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()

    client.create_and_post_order.reset_mock()
    client.create_and_post_order.return_value = {
        "success":  True,
        "orderID":  "0xord_exit_1",
        "status":   "matched",
    }
    # Mock the orderbook lookup execute_exit may consult for cross_spread
    client.get_order_book = MagicMock(return_value=MagicMock(bids=[]))

    exit_result = execution.execute_exit(
        position             = pos_dict,
        intended_exit_price  = 0.60,
        exit_reason          = "profit_target",
        client               = client,
        retry_count          = 0,
    )
    assert exit_result["status"] == "exit_pending"
    assert exit_result["order_id"] == "0xord_exit_1"

    # Position transitioned to 'exiting'
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    assert row["status"] == "exiting"
    assert row["exit_order_id"] == "0xord_exit_1"
    assert row["exit_intended_price"] == pytest.approx(0.60)

    # --- Step 4: WS sends CONFIRMED for the exit ---
    exit_confirm = {
        "id":              "trade_exit_1",
        "status":          "confirmed",
        "taker_order_id":  "0xord_exit_1",
        "size":            111.1,
        "price":           0.585,       # filled below intended (slippage)
        "feeRateBps":      200,
    }
    h = fill_handler.apply_trade_event(exit_confirm)
    assert h["action"] == "filled"
    assert h["role"] == "exit"

    # Verify final state: closed, pnl/pnl_net populated, fees accumulated
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    final = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()

    assert final["status"] == "closed"
    assert final["exit_trade_status"] == "confirmed"
    assert final["actual_exit_price"] == pytest.approx(0.585)

    # Gross P&L = (0.585 - 0.451) * 111.1 ≈ 14.89
    expected_gross = (0.585 - 0.451) * 111.1
    assert final["pnl"] == pytest.approx(expected_gross, rel=1e-2)

    # Net P&L = gross - entry_fees - exit_fee
    # entry_fee ≈ 1.002, exit_fee = 111.1 * 0.585 * 0.02 ≈ 1.30
    expected_entry_fee = 111.1 * 0.451 * 0.02
    expected_exit_fee  = 111.1 * 0.585 * 0.02
    expected_net = expected_gross - expected_entry_fee - expected_exit_fee
    assert final["pnl_net"] == pytest.approx(expected_net, rel=1e-2)
    assert final["entry_fees"] == pytest.approx(expected_entry_fee, rel=1e-2)
    assert final["exit_fees"]  == pytest.approx(expected_exit_fee, rel=1e-2)

    # --- Step 5: activity log should have the full sequence ---
    all_acts = db.get_recent_activity(limit=20)
    cats = [a["category"] for a in all_acts]
    # Should contain BUY (placement), FILL (entry confirmed), SELL (exit
    # placed), FILL (exit confirmed) — order is newest first
    assert "BUY" in cats
    assert "SELL" in cats
    assert cats.count("FILL") == 2  # one entry + one exit


def test_live_trade_lifecycle_failed_on_chain(temp_db, monkeypatch):
    """live BUY → MATCHED → trade FAILED on chain → position released.

    Catches: "we silently leave failed trades stuck in 'pending'"."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success":      True,
        "orderID":      "0xord_fail",
        "status":       "matched",
        "takingAmount": "50.0",
        "makingAmount": "111.1",
    }
    sig = _make_signal()
    result = execution.execute_signal(sig, client=client)
    pid = result["position_id"]

    # WS reports the trade FAILED on chain
    failed_event = {
        "id":              "trade_failed",
        "status":          "failed",
        "taker_order_id":  "0xord_fail",
    }
    h = fill_handler.apply_trade_event(failed_event)
    assert h["action"] == "failed"

    # Position should be cancelled, capital "released" (status=closed, fill_status=cancelled)
    conn = sqlite3.connect(temp_db); conn.row_factory = sqlite3.Row
    final = dict(conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    conn.close()
    assert final["status"] == "closed"
    assert final["fill_status"] == "cancelled"
    assert final["cancelled_reason"] == "trade_failed_onchain"

    # Activity log captures the FAIL at ERROR level
    fails = db.get_recent_activity(categories=["FAIL"], limit=5)
    assert len(fails) == 1
    assert fails[0]["level"] == "ERROR"
    assert fails[0]["position_id"] == pid


# ===========================================================================
# Regression: OrderArgs.side must always be 'BUY' or 'SELL', not 'YES'/'NO'
# ===========================================================================
# Bug 2026-04-29: execute_signal and execute_topup were passing the
# POSITION side (YES/NO) into OrderArgs(side=...), but py_clob_client
# expects the ORDER side (BUY/SELL).  Every live order failed with
# "ValueError: order_args.side must be 'BUY' or 'SELL'".  These tests
# pin the fix so a refactor that re-conflates them will fail loudly.

def test_execute_signal_passes_BUY_to_order_args(temp_db, monkeypatch):
    """OrderArgs.side must be Side.BUY for entry orders, regardless of
    whether the position is YES or NO.  The position side is encoded by
    which token_id we send.  Migrated 2026-04-29 to py_clob_client_v2,
    which uses the Side enum (Side.BUY=0, Side.SELL=1) instead of strings."""
    from py_clob_client_v2 import Side
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    captured: list[dict] = []

    class _SpyOrderArgs:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)

    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xok", "status": "matched",
        "takingAmount": "50.0", "makingAmount": "111.1",
    }

    # Test with both YES and NO positions — both should send Side.BUY
    for side in ("YES", "NO"):
        captured.clear()
        sig = _make_signal(side=side)
        execution.execute_signal(sig, client=client)
        assert len(captured) == 1
        assert captured[0]["side"] == Side.BUY, (
            f"position side {side} → OrderArgs.side should be Side.BUY, "
            f"got {captured[0]['side']!r}"
        )
        # Position side encoded by token_id
        if side == "YES":
            assert captured[0]["token_id"] == sig["yes_token_id"]
        else:
            assert captured[0]["token_id"] == sig["no_token_id"]


def test_execute_topup_passes_BUY_to_order_args(temp_db, monkeypatch):
    """Same regression for top-ups — they're always BUYs of the same
    token type the parent position holds."""
    from py_clob_client_v2 import Side
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    captured: list[dict] = []

    class _SpyOrderArgs:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)

    # Seed an existing live position so execute_topup has a parent
    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=100.0,
        entry_price=0.50, entry_time="2026-01-01T00:00:00",
        shares=200.0, yes_token_id="tok_yes_42",
        is_paper=0, fill_status="filled",
    )

    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xtopup", "status": "matched",
    }

    pos = {
        "id": pid, "contract_id": "0xabc", "side": "YES",
        "entry_price": 0.50, "yes_token_id": "tok_yes_42",
        "size_usdc": 100.0, "shares": 200.0,
        "pending_topup_order_id": None,
        "gamma_market_id": "1886470",
    }
    execution.execute_topup(pos, add_amount_usdc=20.0, client=client)

    assert len(captured) == 1
    assert captured[0]["side"] == Side.BUY, (
        f"top-up OrderArgs.side should be Side.BUY, got {captured[0]['side']!r}"
    )


def test_execute_exit_passes_SELL_to_order_args(temp_db, monkeypatch):
    """Inverse: exits must always be Side.SELL."""
    from py_clob_client_v2 import Side
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    captured: list[dict] = []

    class _SpyOrderArgs:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)

    pid = db.insert_position(
        contract_id="0xabc", side="YES", size_usdc=100.0,
        entry_price=0.50, entry_time="2026-01-01T00:00:00",
        shares=200.0, yes_token_id="tok_yes_42",
        is_paper=0, fill_status="filled",
    )

    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xexit", "status": "matched",
    }
    client.get_order_book = MagicMock(return_value=MagicMock(bids=[]))

    pos = {
        "id": pid, "contract_id": "0xabc", "side": "YES",
        "entry_price": 0.50, "yes_token_id": "tok_yes_42",
        "shares": 200.0, "exit_order_id": None,
    }
    execution.execute_exit(
        position=pos, intended_exit_price=0.60,
        exit_reason="profit_target", client=client, retry_count=0,
    )

    assert len(captured) == 1
    assert captured[0]["side"] == Side.SELL, (
        f"exit OrderArgs.side should be Side.SELL, got {captured[0]['side']!r}"
    )


# ===========================================================================
# Scenario 3: Idempotency end-to-end (REST + WS race)
# ===========================================================================

def test_rest_and_ws_both_apply_fill_only_once(temp_db, monkeypatch):
    """The REST poller's safety-net path constructs a synthetic
    confirmed trade event and feeds it through fill_handler — same path
    the WS uses.  If both deliver the fill (race during reconnect),
    fill_handler's monotonic gate ensures only ONE fill is applied.

    Catches: "I broke idempotency; double-fees are now possible"."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    client = MagicMock()
    client.create_and_post_order.return_value = {
        "success":      True,
        "orderID":      "0xord_race",
        "status":       "matched",
        "takingAmount": "50.0",
        "makingAmount": "111.1",
    }
    sig = _make_signal()
    result = execution.execute_signal(sig, client=client)
    pid = result["position_id"]

    # Both paths see the fill — apply twice
    event = {
        "id":              "trade_race",
        "status":          "confirmed",
        "taker_order_id":  "0xord_race",
        "size":            111.1,
        "price":           0.45,
        "feeRateBps":      200,
    }
    r1 = fill_handler.apply_trade_event(event)
    r2 = fill_handler.apply_trade_event(event)

    assert r1["action"] == "filled"
    # Second attempt deduped via processed_trade_events on event_id (the
    # WS+REST race protection mechanism — see fill_handler.py).
    assert r2["action"] == "ignored_duplicate_event"

    # Critical: entry_fees should have been written ONCE, not twice
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT entry_fees FROM positions WHERE id=?", (pid,)
    ).fetchone()
    conn.close()
    expected_fee = 111.1 * 0.45 * 0.02
    assert row[0] == pytest.approx(expected_fee, rel=1e-2)
    # If idempotency broke, this would be 2x expected_fee
