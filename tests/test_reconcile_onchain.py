"""
test_reconcile_onchain.py — Tests for on-chain reconciliation (Phase 8).

Covers:
  * Empty data_api_index → silent no-op (paper / fetch failed)
  * Perfect match → no drift logged
  * orphan_db: DB has it but chain doesn't
  * share_drift: both have it, sizes diverge beyond tolerance
  * Within-tolerance delta → no drift logged
  * orphan_chain: chain has tokens DB doesn't
  * Skips paper positions
  * Skips positions without a token_id
  * Tiny on-chain dust positions don't count as orphan_chain
"""

from __future__ import annotations

import os
import sys

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import monitor


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
    yes_token_id: str | None = "tok_yes_1",
    side: str = "YES",
    shares: float = 100.0,
    is_paper: int = 0,
    fill_status: str = "filled",
) -> int:
    return db.insert_position(
        contract_id  = contract_id,
        side         = side,
        size_usdc    = shares * 0.50,
        entry_price  = 0.50,
        entry_time   = "2026-01-01T00:00:00",
        shares       = shares,
        yes_token_id = yes_token_id,
        is_paper     = is_paper,
        fill_status  = fill_status,
    )


# ===========================================================================
# Empty index — silent no-op
# ===========================================================================

def test_empty_index_returns_zero_drift(temp_db, caplog):
    _seed()
    result = monitor._reconcile_onchain({})
    assert result == {"orphan_db": 0, "share_drift": 0, "orphan_chain": 0}
    # Should not have logged anything (no DRIFT messages)
    assert not any("DRIFT" in r.message for r in caplog.records)


# ===========================================================================
# Perfect match
# ===========================================================================

def test_perfect_match_no_drift(temp_db, caplog):
    _seed(yes_token_id="tok_a", shares=100.0)
    index = {"tok_a": {"size": 100.0, "title": "Will it rain"}}
    result = monitor._reconcile_onchain(index)
    assert result == {"orphan_db": 0, "share_drift": 0, "orphan_chain": 0}
    assert not any("DRIFT" in r.message for r in caplog.records)


def test_within_tolerance_no_drift(temp_db, caplog):
    """0.3-share delta is below the 0.5-share floating-point tolerance."""
    _seed(yes_token_id="tok_a", shares=100.0)
    index = {"tok_a": {"size": 100.3, "title": "X"}}
    result = monitor._reconcile_onchain(index)
    assert result["share_drift"] == 0


# ===========================================================================
# orphan_db
# ===========================================================================

def test_orphan_db_logged(temp_db, caplog):
    """DB shows the position open, chain doesn't have it."""
    import logging
    caplog.set_level(logging.WARNING)
    _seed(yes_token_id="tok_missing", shares=100.0)
    # Index has *some* position so we don't bail on empty index — just not ours
    index = {"tok_other": {"size": 50.0, "title": "Other"}}
    result = monitor._reconcile_onchain(index)
    assert result["orphan_db"] == 1
    assert any("orphan_db" in r.message for r in caplog.records)


# ===========================================================================
# share_drift
# ===========================================================================

def test_share_drift_logged(temp_db, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    _seed(yes_token_id="tok_a", shares=100.0)
    index = {"tok_a": {"size": 80.0, "title": "X"}}
    result = monitor._reconcile_onchain(index)
    assert result["share_drift"] == 1
    msgs = [r.message for r in caplog.records]
    assert any("share_drift" in m and "delta=20" in m for m in msgs)


def test_share_drift_other_direction(temp_db):
    """Chain having MORE shares than DB also counts as drift."""
    _seed(yes_token_id="tok_a", shares=100.0)
    index = {"tok_a": {"size": 150.0, "title": "X"}}
    result = monitor._reconcile_onchain(index)
    assert result["share_drift"] == 1


# ===========================================================================
# orphan_chain
# ===========================================================================

def test_orphan_chain_logged(temp_db, caplog):
    """Chain has a token that no DB row references."""
    import logging
    caplog.set_level(logging.INFO)
    _seed(yes_token_id="tok_a", shares=100.0)
    index = {
        "tok_a":      {"size": 100.0, "title": "X"},
        "tok_extra":  {"size": 50.0,  "title": "Manual trade"},
    }
    result = monitor._reconcile_onchain(index)
    assert result["orphan_chain"] == 1
    assert any("orphan_chain" in r.message for r in caplog.records)


def test_orphan_chain_dust_ignored(temp_db):
    """Tiny chain-side positions (rounding dust) shouldn't count as orphans."""
    _seed(yes_token_id="tok_a", shares=100.0)
    index = {
        "tok_a":    {"size": 100.0, "title": "X"},
        "tok_dust": {"size": 0.1,   "title": "Dust"},
    }
    result = monitor._reconcile_onchain(index)
    assert result["orphan_chain"] == 0


# ===========================================================================
# Filtering: skip paper / missing-token positions
# ===========================================================================

def test_paper_positions_skipped(temp_db):
    """Paper positions don't have on-chain footprint — never reconcile."""
    _seed(yes_token_id="tok_paper", shares=100.0, is_paper=1)
    # Index empty for that token — would otherwise be orphan_db
    index = {"tok_other": {"size": 50.0, "title": "X"}}
    result = monitor._reconcile_onchain(index)
    assert result["orphan_db"] == 0
    # And tok_other doesn't get flagged as orphan_chain because the
    # filter uses live (non-paper) DB tokens — but the chain entry IS
    # an orphan since it's not in any non-paper DB row.
    assert result["orphan_chain"] == 1


def test_missing_token_id_skipped(temp_db):
    """Legacy positions without yes/no_token_id can't be reconciled — skip."""
    _seed(yes_token_id=None, shares=100.0)
    index = {"tok_a": {"size": 50.0, "title": "X"}}
    result = monitor._reconcile_onchain(index)
    # No drift attributed to the legacy row, but tok_a is orphan_chain
    assert result["orphan_db"] == 0
    assert result["orphan_chain"] == 1


def test_unfilled_positions_skipped(temp_db):
    """Pending (unfilled) buys don't have a confirmed on-chain footprint."""
    _seed(yes_token_id="tok_pending", shares=100.0, fill_status="pending")
    index = {"tok_other": {"size": 50.0, "title": "X"}}
    result = monitor._reconcile_onchain(index)
    assert result["orphan_db"] == 0


# ===========================================================================
# Mixed scenario
# ===========================================================================

def test_mixed_scenario_counts(temp_db):
    """Multiple drift types in one cycle — each counted independently."""
    _seed(contract_id="0x1", yes_token_id="tok_match",   shares=100.0)
    _seed(contract_id="0x2", yes_token_id="tok_drift",   shares=100.0)
    _seed(contract_id="0x3", yes_token_id="tok_orphan",  shares=100.0)

    index = {
        "tok_match":  {"size": 100.0, "title": "OK"},
        "tok_drift":  {"size": 60.0,  "title": "Drifty"},
        # tok_orphan absent → orphan_db
        "tok_extra":  {"size": 25.0,  "title": "Manual"},  # orphan_chain
    }
    result = monitor._reconcile_onchain(index)
    assert result == {"orphan_db": 1, "share_drift": 1, "orphan_chain": 1}
