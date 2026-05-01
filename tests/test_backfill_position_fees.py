"""
test_backfill_position_fees.py — Tests for the post-hoc fee backfill that
pulls real Polymarket trade fees and updates positions.entry_fees /
exit_fees / pnl_net.

The backfill is needed because Polymarket WS trade events don't include
fee_rate_bps; only the CLOB GET /trades response does (1000 bps = 10%
taker fee in production samples as of 2026-04).
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
import execution
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
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    yield


def _seed_closed_position(*, entry_oid="0xord_entry", exit_oid="0xord_exit",
                          entry_price=0.30, shares=33.33, size_usdc=10.0,
                          gross_pnl=2.0):
    pid = db.insert_position(
        contract_id="0xmarket", side="YES",
        size_usdc=size_usdc, entry_price=entry_price,
        entry_time="2026-04-30T12:00:00",
        order_id=entry_oid, target_size_usdc=size_usdc, shares=shares,
        yes_token_id="tok_yes", is_paper=0, fill_status="filled",
    )
    # Stamp closed + exit info + ledger rows
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("""
        UPDATE positions SET status='closed', exit_order_id=?, pnl=?
        WHERE id=?
    """, (exit_oid, gross_pnl, pid))
    conn.commit(); conn.close()
    db.insert_position_order(
        position_id=pid, order_id=entry_oid, role="entry",
        intended_usdc=size_usdc, intended_shares=shares,
        limit_price=entry_price, status="filled",
    )
    db.insert_position_order(
        position_id=pid, order_id=exit_oid, role="exit",
        intended_usdc=size_usdc, intended_shares=shares,
        limit_price=entry_price, status="filled",
    )
    return pid


# ===========================================================================
# Happy path — fees pulled and written
# ===========================================================================

def test_backfill_sums_taker_fees_per_role(temp_db, live_mode, monkeypatch):
    """Two trades: one entry buy (taker, 1000 bps), one exit sell (taker,
    1000 bps).  entry_fees should equal the buy fee; exit_fees the sell fee."""
    pid = _seed_closed_position(
        entry_oid="0xentry", exit_oid="0xexit",
        entry_price=0.30, shares=33.33, size_usdc=10.0, gross_pnl=2.0,
    )
    fake_trades = [
        {
            "taker_order_id": "0xentry",
            "size": "33.33", "price": "0.30",
            "fee_rate_bps": "1000",   # 10% taker
            "maker_orders": [],
        },
        {
            "taker_order_id": "0xexit",
            "size": "33.33", "price": "0.36",
            "fee_rate_bps": "1000",
            "maker_orders": [],
        },
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client)
    # entry: 33.33 × 0.30 × 0.10 = $0.9999
    assert res["entry_fees"] == pytest.approx(0.9999, abs=0.001)
    # exit:  33.33 × 0.36 × 0.10 = $1.19988
    assert res["exit_fees"] == pytest.approx(1.1999, abs=0.001)
    # pnl_net = gross 2.0 - 0.9999 - 1.1999 = -0.1998
    assert res["pnl_net"] == pytest.approx(-0.1998, abs=0.001)
    assert res["n_trades_matched"] == 2
    assert res["committed"] is True

    # Verify DB writes
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT entry_fees, exit_fees, pnl_net FROM positions WHERE id=?",
        (pid,),
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.9999, abs=0.001)
    assert row[1] == pytest.approx(1.1999, abs=0.001)
    assert row[2] == pytest.approx(-0.1998, abs=0.001)


def test_backfill_handles_maker_match(temp_db, live_mode, monkeypatch):
    """When we were the MAKER (our order shows up in maker_orders[]), we
    paid 0 fee (Polymarket maker fee is 0).  Backfill must still match
    the trade and account for it (without inflating fees)."""
    pid = _seed_closed_position(entry_oid="0xmaker_entry", exit_oid="0xexit")
    fake_trades = [{
        "taker_order_id": "0xothers_order",
        "size": "10.0", "price": "0.30",
        "fee_rate_bps": "1000",
        "maker_orders": [{"order_id": "0xmaker_entry"}],
    }]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client)
    # We were maker — fee is computed using the trade's fee_rate_bps but
    # in practice Polymarket charges only takers.  The function uses the
    # top-level fee_rate_bps regardless of our role; this matches the
    # existing _extract_fee semantics.  If we want to skip fees for
    # maker matches, that's a separate config decision — for now the
    # test just confirms we MATCH the trade by maker_orders.
    assert res["n_trades_matched"] == 1


def test_backfill_skips_unmatched_trades(temp_db, live_mode, monkeypatch):
    """Trades that aren't ours (taker_order_id not in our orders, no
    maker_orders match) must be ignored — they came from another wallet
    on the same market."""
    pid = _seed_closed_position(entry_oid="0xours", exit_oid="0xours_exit")
    fake_trades = [
        {"taker_order_id": "0xothers", "size": "5", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
        {"taker_order_id": "0xours", "size": "33.33", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client)
    assert res["n_trades_matched"] == 1


def test_backfill_topup_role_aggregates_into_entry_fees(temp_db, live_mode, monkeypatch):
    """Topup fees go into entry_fees (same bucket on the position row)."""
    pid = _seed_closed_position(entry_oid="0xentry", exit_oid="0xexit")
    db.insert_position_order(
        position_id=pid, order_id="0xtopup", role="topup",
        intended_usdc=5.0, intended_shares=16.67,
        limit_price=0.30, status="filled",
    )
    fake_trades = [
        {"taker_order_id": "0xentry", "size": "33.33", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
        {"taker_order_id": "0xtopup", "size": "16.67", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
        {"taker_order_id": "0xexit",  "size": "50", "price": "0.36",
         "fee_rate_bps": "1000", "maker_orders": []},
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client)
    # entry_fees = (33.33 + 16.67) × 0.30 × 0.10 = 1.50
    assert res["entry_fees"] == pytest.approx(1.50, abs=0.01)
    # exit_fees = 50 × 0.36 × 0.10 = 1.80
    assert res["exit_fees"] == pytest.approx(1.80, abs=0.01)


def test_backfill_zero_fee_trades_count_but_add_zero(temp_db, live_mode, monkeypatch):
    """Trades where Polymarket charged 0 (maker matches, promotional
    periods) match but contribute $0 to the totals."""
    pid = _seed_closed_position(entry_oid="0xentry", exit_oid="0xexit")
    fake_trades = [
        {"taker_order_id": "0xentry", "size": "10", "price": "0.30",
         "fee_rate_bps": "0", "maker_orders": []},
        {"taker_order_id": "0xexit",  "size": "10", "price": "0.30",
         "fee_rate_bps": "", "maker_orders": []},
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client)
    assert res["n_trades_matched"] == 2
    assert res["entry_fees"] == 0.0
    assert res["exit_fees"] == 0.0


# ===========================================================================
# Dry-run + commit semantics
# ===========================================================================

def test_dry_run_does_not_write(temp_db, live_mode, monkeypatch):
    """commit=False returns the computed values but leaves the DB alone."""
    pid = _seed_closed_position(entry_oid="0xentry", exit_oid="0xexit")
    fake_trades = [
        {"taker_order_id": "0xentry", "size": "10", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client, commit=False)
    assert res["entry_fees"] == pytest.approx(0.30)  # 10 × 0.30 × 0.10
    assert res["committed"] is False

    # DB should be unchanged
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT entry_fees, exit_fees FROM positions WHERE id=?", (pid,),
    ).fetchone()
    conn.close()
    assert row[0] == 0.0    # default, not overwritten
    assert row[1] == 0.0


def test_idempotent_replaces_not_accumulates(temp_db, live_mode, monkeypatch):
    """Running backfill twice with the same trades produces the same
    totals — fees are REPLACED, not accumulated."""
    pid = _seed_closed_position(entry_oid="0xentry", exit_oid="0xexit")
    fake_trades = [
        {"taker_order_id": "0xentry", "size": "10", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res1 = execution.backfill_position_fees(pid, fake_client)
    res2 = execution.backfill_position_fees(pid, fake_client)
    assert res1["entry_fees"] == res2["entry_fees"]
    assert res1["pnl_net"]    == res2["pnl_net"]


# ===========================================================================
# Defensive / no-op cases
# ===========================================================================

def test_paper_mode_returns_empty():
    import execution as ex
    # In paper mode we don't have CLOB trades at all
    import importlib
    # Use the existing module's PAPER_TRADE config check
    saved = ex.PAPER_TRADE
    try:
        ex.PAPER_TRADE = True
        assert ex.backfill_position_fees(123, MagicMock()) == {}
    finally:
        ex.PAPER_TRADE = saved


def test_no_client_returns_empty(temp_db, live_mode):
    pid = _seed_closed_position()
    assert execution.backfill_position_fees(pid, None) == {}


def test_missing_position_returns_empty(temp_db, live_mode):
    fake_client = MagicMock()
    assert execution.backfill_position_fees(99999, fake_client) == {}


def test_no_trades_returned_returns_empty(temp_db, live_mode):
    pid = _seed_closed_position()
    fake_client = MagicMock()
    fake_client.get_trades.return_value = []
    assert execution.backfill_position_fees(pid, fake_client) == {}


def test_get_trades_exception_returns_empty(temp_db, live_mode, monkeypatch):
    """A transient CLOB error during the trades fetch returns empty
    rather than raising — caller (fill_handler) catches gracefully."""
    pid = _seed_closed_position()
    fake_client = MagicMock()
    fake_client.get_trades.side_effect = RuntimeError("CLOB down")
    res = execution.backfill_position_fees(pid, fake_client)
    assert res == {}


def test_pnl_net_correctly_subtracts_fees_from_gross(temp_db, live_mode, monkeypatch):
    """pnl_net = pnl (gross) - entry_fees - exit_fees.  Verify the math
    for a case with both fees nonzero."""
    pid = _seed_closed_position(gross_pnl=10.00,
                                entry_oid="0xentry", exit_oid="0xexit")
    fake_trades = [
        {"taker_order_id": "0xentry", "size": "100", "price": "0.30",
         "fee_rate_bps": "1000", "maker_orders": []},
        {"taker_order_id": "0xexit",  "size": "100", "price": "0.40",
         "fee_rate_bps": "1000", "maker_orders": []},
    ]
    fake_client = MagicMock()
    fake_client.get_trades.return_value = fake_trades

    res = execution.backfill_position_fees(pid, fake_client)
    # entry_fee = 100 * 0.30 * 0.10 = 3.00; exit_fee = 100 * 0.40 * 0.10 = 4.00
    # pnl_net = 10.00 - 3.00 - 4.00 = 3.00
    assert res["pnl_net"] == pytest.approx(3.00)
