"""
test_pending_order_race.py — Lock in the fix for the pending-order race
condition that produced NYC double position AND Houston $74.99 overshoot
on 2026-06-12.

Two distinct failure modes, same root cause: actual_deployed (from the
Polymarket positions API) doesn't include orders that have been placed
but not yet filled.  Without counting pending orders as "committed,"
the bot:

  - NYC case: bought 92-93°F at 17:37, then 94-95°F at 18:32 on the
    same event (MAX_BINS_PER_EVENT=1) because the API hadn't seen the
    92-93°F fill yet.  Two positions on a one-bin-per-event market.

  - Houston case: kept placing $10 topup orders for the same contract
    across multiple scans because actual_deployed reported $0 while
    multiple orders sat on the book unfilled.  When they all eventually
    filled, total deployed reached $74.99 — 7.5x the $10 target.

The fix exposes two helpers (`pending_contracts_today`,
`pending_stake_for_contract_today`) that return the set of contracts
with placed-but-not-resolved orders today and the sum of their stakes.

Run:
    cd bot
    python -m pytest tests/test_pending_order_race.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scheduled_predictor import (   # type: ignore
    _SCHEMA_SQL,
    pending_contracts_today,
    pending_stake_for_contract_today,
)


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fresh_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    for col, ddl in [("market_closed", "INTEGER DEFAULT 0"),
                       ("data_quality_flag", "TEXT"),
                       ("cooling_confidence", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE paper_predictor_signals "
                          f"ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError:
            pass
    return conn


def _insert_order(conn, contract_id, status="placed", stake_usd=10.0,
                    placed_at_utc=None):
    placed_at = placed_at_utc or (datetime.now(timezone.utc)
                                     .isoformat()[:19] + "+00:00")
    conn.execute(
        """INSERT INTO live_predictor_orders
            (placed_at_utc, city, contract_id, status, stake_usd, order_id)
            VALUES (?, ?, ?, ?, ?, ?)""",
        (placed_at, "TestCity", contract_id, status, stake_usd,
         f"orderid_{contract_id[:8]}_{stake_usd}"),
    )
    conn.commit()


# ============================================================
# pending_contracts_today
# ============================================================

def test_no_orders_returns_empty_set():
    conn = _fresh_db()
    assert pending_contracts_today(conn) == set()


def test_single_placed_order_appears():
    conn = _fresh_db()
    _insert_order(conn, "0xABC", status="placed", stake_usd=10.0)
    assert pending_contracts_today(conn) == {"0xABC"}


def test_filled_orders_do_not_appear():
    """Once reconciliation promotes an order from 'placed' to 'filled',
    it stops counting as pending — the position now shows in API."""
    conn = _fresh_db()
    _insert_order(conn, "0xABC", status="placed", stake_usd=10.0)
    _insert_order(conn, "0xDEF", status="filled", stake_usd=10.0)
    _insert_order(conn, "0xGHI", status="cancelled", stake_usd=10.0)
    _insert_order(conn, "0xJKL", status="stale", stake_usd=10.0)
    _insert_order(conn, "0xMNO", status="error", stake_usd=10.0)
    assert pending_contracts_today(conn) == {"0xABC"}


def test_multiple_pending_contracts():
    """Each distinct contract_id appears once regardless of how many
    pending orders exist on it."""
    conn = _fresh_db()
    _insert_order(conn, "0xABC", status="placed", stake_usd=10.0)
    _insert_order(conn, "0xABC", status="placed", stake_usd=5.0)
    _insert_order(conn, "0xDEF", status="placed", stake_usd=10.0)
    assert pending_contracts_today(conn) == {"0xABC", "0xDEF"}


def test_yesterdays_pending_orders_do_not_count():
    """Pending orders from yesterday don't count toward today's cap.
    Yesterday's resting orders should have been resolved (filled or
    cancelled / staled by reconciliation) before today's window."""
    conn = _fresh_db()
    yesterday = "2026-06-11T18:00:00+00:00"
    _insert_order(conn, "0xABC", status="placed", stake_usd=10.0,
                    placed_at_utc=yesterday)
    assert pending_contracts_today(conn) == set()


# ============================================================
# pending_stake_for_contract_today
# ============================================================

def test_pending_stake_sums_across_multiple_orders():
    """The Houston case: scan A places $10, scan B places $10 (because
    API still showed actual_deployed=$0), scan C places $10.  All
    three sit on the book → pending_stake = $30."""
    conn = _fresh_db()
    _insert_order(conn, "0xHOUSTON", status="placed", stake_usd=10.0)
    _insert_order(conn, "0xHOUSTON", status="placed", stake_usd=10.0)
    _insert_order(conn, "0xHOUSTON", status="placed", stake_usd=10.0)
    assert pending_stake_for_contract_today(conn, "0xHOUSTON") == 30.0


def test_pending_stake_ignores_filled():
    conn = _fresh_db()
    _insert_order(conn, "0xHOUSTON", status="filled", stake_usd=10.0)
    _insert_order(conn, "0xHOUSTON", status="placed", stake_usd=10.0)
    # Only the pending one counts — filled is already in actual_deployed
    assert pending_stake_for_contract_today(conn, "0xHOUSTON") == 10.0


def test_pending_stake_isolated_per_contract():
    conn = _fresh_db()
    _insert_order(conn, "0xHOUSTON", status="placed", stake_usd=10.0)
    _insert_order(conn, "0xDALLAS", status="placed", stake_usd=15.0)
    assert pending_stake_for_contract_today(conn, "0xHOUSTON") == 10.0
    assert pending_stake_for_contract_today(conn, "0xDALLAS") == 15.0
    assert pending_stake_for_contract_today(conn, "0xUNKNOWN") == 0.0


# ============================================================
# The race scenarios, as integration-level assertions
# ============================================================

def test_houston_topup_race_committed_math():
    """Replay the Houston scenario at the committed-deployed math level.

    State after 3 scans where the bot placed orders without seeing any
    fill back in the API:
       actual_deployed (from API) = $0
       pending orders = 3 × $10 = $30
       committed = actual + pending = $30
       target = $10

    remaining_to_target = max(0, 10 - 30) = 0.  Bot does NOT place a
    fourth order.  This is the math the fix enforces.
    """
    conn = _fresh_db()
    contract = "0xHOUSTON_BIN_92_93"
    _insert_order(conn, contract, status="placed", stake_usd=10.0)
    _insert_order(conn, contract, status="placed", stake_usd=10.0)
    _insert_order(conn, contract, status="placed", stake_usd=10.0)

    actual_deployed = 0.0     # API hasn't seen any fills yet
    pending_stake = pending_stake_for_contract_today(conn, contract)
    committed_deployed = actual_deployed + pending_stake
    target_stake = 10.0

    remaining_to_target = max(0.0, target_stake - committed_deployed)
    assert remaining_to_target == 0.0, (
        f"Pending orders should consume all target headroom.  "
        f"Pending = ${pending_stake}, target = ${target_stake}, "
        f"remaining = ${remaining_to_target}.  Without this fix, the "
        f"bot would keep placing $10 orders and Houston would blow "
        f"past target as fills eventually accumulate."
    )


def test_nyc_fresh_bin_race_committed_contracts():
    """Replay the NYC scenario at the committed-contracts set level.

    State after the bot's 17:37 buy:
       held_contracts (filled positions from API) = {} (not filled yet)
       pending_contracts_today = {92-93°F contract}
       committed = held ∪ pending = {92-93°F}

    The cap check now sees buys_already = 1, event_at_cap = True.
    fresh_bins excludes the 94-95°F bin from new-buy candidacy
    because... wait, 94-95°F is a different contract and isn't in
    committed.  But event_at_cap blocks fresh_candidates entirely:

       if event_at_cap: fresh_candidates = []

    So even though 94-95°F is "available" by contract_id, the event's
    one slot is filled by 92-93°F's pending order.
    """
    conn = _fresh_db()
    bin_92_93 = "0xNYC_BIN_92_93"
    bin_94_95 = "0xNYC_BIN_94_95"

    # NYC 17:37 buy creates a pending order on 92-93°F
    _insert_order(conn, bin_92_93, status="placed", stake_usd=10.0)
    _insert_order(conn, bin_92_93, status="placed", stake_usd=5.0)

    held_contracts: set[str] = set()   # API hasn't seen any fills yet
    pending = pending_contracts_today(conn)
    committed = held_contracts | pending

    # 92-93°F is committed (via pending) even though not filled yet
    assert bin_92_93 in committed
    assert bin_94_95 not in committed

    # If we now consider candidates for the same event:
    MAX_BINS_PER_EVENT = 1
    market_open = {bin_92_93: True, bin_94_95: True}
    already_bought = {c for c in committed if market_open.get(c, True)}
    buys_already = len(already_bought)
    event_at_cap = buys_already >= MAX_BINS_PER_EVENT
    assert event_at_cap, (
        "With committed = {92-93°F pending}, the event must already "
        "be AT CAP.  Without this fix, buys_already = 0 (held was "
        "empty) and the bot would happily buy 94-95°F as a fresh bin "
        "while 92-93°F was still resting on the book."
    )


def test_houston_cap_uses_committed_not_just_actual():
    """The per-contract cap should fire on committed (actual + pending)
    not just actual.  Otherwise even the cap doesn't help with the
    Houston race because actual_deployed reports $0."""
    conn = _fresh_db()
    contract = "0xHOUSTON_BIN_92_93"
    # Three $10 orders pending, none filled yet
    _insert_order(conn, contract, status="placed", stake_usd=10.0)
    _insert_order(conn, contract, status="placed", stake_usd=10.0)
    _insert_order(conn, contract, status="placed", stake_usd=10.0)

    actual_deployed = 0.0
    pending_stake = pending_stake_for_contract_today(conn, contract)
    committed_deployed = actual_deployed + pending_stake
    MAX_PER_CONTRACT_USD = 15.0

    contract_ceiling_remaining = max(
        0.0, MAX_PER_CONTRACT_USD - committed_deployed)
    # 15 - 30 = -15 → clamped at 0.  No cap headroom.
    assert contract_ceiling_remaining == 0.0, (
        f"Cap headroom should be 0 (committed=${committed_deployed} >= "
        f"cap=${MAX_PER_CONTRACT_USD}).  Without this fix, the cap "
        f"would compute 15-0=15 and let the bot keep stacking orders."
    )