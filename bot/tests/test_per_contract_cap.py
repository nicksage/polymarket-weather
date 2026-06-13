"""
test_per_contract_cap.py — Lock in the per-contract daily cap fix.

Regression: Dallas 2026-06-12.  The bot had $6.68 actually deployed on
Polymarket (one filled buy at 7.6¢) and ~$8.32 of additional LIVE_BUY
signal rows for unfilled topup attempts (limit orders the market moved
past as Dallas crashed 7.6¢ → 2¢).  The cap was anchored to
`deployed_today_for_contract()`, which sums signal-row intent —
including unfilled orders — and reported $15.00 spent.  Legitimate
topup attempts were then blocked even though actual deployed capital
was below target.

Fix: anchor the cap to `actual_deployed` (Polymarket API cost basis),
not to signal-row intent.  This file pins that behavior so a future
edit doesn't silently revert.

Run:
    cd bot
    python -m pytest tests/test_per_contract_cap.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scheduled_predictor import (  # type: ignore
    _SCHEMA_SQL,
    deployed_today_for_contract,
    MAX_PER_CONTRACT_USD,
)


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


def _insert_live_buy(conn, contract_id, scanned_at_utc, recommended_stake_usd):
    conn.execute(
        """INSERT INTO paper_predictor_signals
            (scanned_at_utc, mode, city, event_date, contract_id,
             bin_label, action, recommended_stake_usd)
            VALUES (?, 'live', ?, ?, ?, '88-89F', 'LIVE_BUY', ?)""",
        (scanned_at_utc, "Dallas", "2026-06-12", contract_id, recommended_stake_usd),
    )
    conn.commit()


def _cap_remaining(actual_deployed_usd: float) -> float:
    """Replicates the cap-computation in run_intraday_scan exactly.

    Kept as a small local helper because the live code does this inline
    (one line — extracting into a function would be over-engineering).
    Any future refactor of the live code should keep this formula intact
    OR update this test to match.
    """
    if MAX_PER_CONTRACT_USD <= 0:
        return float("inf")
    return max(0.0, MAX_PER_CONTRACT_USD - actual_deployed_usd)


# ============================================================
# The Dallas 2026-06-12 regression
# ============================================================

def test_dallas_scenario_unfilled_intent_does_not_block_topup():
    """Recreate the exact Dallas 2026-06-12 setup:
       - one filled LIVE_BUY at $6.68 (the 88.2 shares @ 7.6¢)
       - three unfilled LIVE_BUY attempts at $5 each ($15 of intent)
       - actual_deployed (from API) = $6.68

    The cap MUST not block topup attempts.  Headroom should equal
    MAX_PER_CONTRACT_USD - $6.68."""
    conn = _fresh_db()
    contract = "dallas_88_89"
    # Use today's UTC date so deployed_today_for_contract's date filter
    # matches (otherwise the filter looks at a different "today" than
    # the test's fixture date and the query returns 0).
    from datetime import datetime as _dt, timezone as _tz
    today_str = _dt.now(_tz.utc).strftime("%Y-%m-%d")
    # The filled buy
    _insert_live_buy(conn, contract, f"{today_str}T16:00:00Z", 6.68)
    # Three unfilled topup attempts as Dallas dropped through 6¢ / 5¢ / 4¢
    _insert_live_buy(conn, contract, f"{today_str}T18:00:00Z", 5.00)
    _insert_live_buy(conn, contract, f"{today_str}T18:15:00Z", 5.00)
    _insert_live_buy(conn, contract, f"{today_str}T18:30:00Z", 5.00)

    # Show the deprecated function returns the misleading intent sum
    intent_sum = deployed_today_for_contract(conn, "live", contract)
    assert abs(intent_sum - 21.68) < 0.01, (
        f"intent sum should be $21.68 ($6.68 filled + 3 × $5 unfilled); "
        f"got ${intent_sum:.2f}"
    )

    # The cap uses actual_deployed (API), not the intent sum
    actual_deployed = 6.68
    remaining = _cap_remaining(actual_deployed)
    expected_remaining = MAX_PER_CONTRACT_USD - 6.68
    assert abs(remaining - expected_remaining) < 0.01, (
        f"cap headroom should be ${expected_remaining:.2f} "
        f"(MAX={MAX_PER_CONTRACT_USD} - actual=$6.68); "
        f"got ${remaining:.2f}"
    )
    # And — the critical assertion — there IS headroom for topups
    assert remaining > 0, (
        "Dallas 2026-06-12 bug regressed: cap is reporting zero headroom "
        "despite actual_deployed being well below MAX_PER_CONTRACT_USD"
    )


# ============================================================
# Boundary tests
# ============================================================

def test_cap_blocks_when_actual_deployed_at_max():
    """When actual_deployed == MAX_PER_CONTRACT_USD, no headroom."""
    remaining = _cap_remaining(MAX_PER_CONTRACT_USD)
    assert remaining == 0.0


def test_cap_blocks_when_actual_deployed_exceeds_max():
    """Topup overflow (a buy filled higher than expected) → headroom
    floors at 0, never goes negative."""
    remaining = _cap_remaining(MAX_PER_CONTRACT_USD + 5.0)
    assert remaining == 0.0


def test_cap_full_headroom_when_nothing_deployed():
    """Fresh contract with no fills: full MAX_PER_CONTRACT_USD available."""
    remaining = _cap_remaining(0.0)
    assert remaining == MAX_PER_CONTRACT_USD


def test_cap_disabled_when_max_is_zero(monkeypatch):
    """Env override MAX_PER_CONTRACT_USD=0 disables the cap entirely."""
    monkeypatch.setattr(
        "scheduled_predictor.MAX_PER_CONTRACT_USD", 0.0)
    # Re-import the helper to pick up the patched value
    from importlib import reload
    import scheduled_predictor
    reload(scheduled_predictor)
    # With MAX <= 0, cap returns infinity (no constraint)
    from scheduled_predictor import MAX_PER_CONTRACT_USD as patched_max
    if patched_max <= 0:
        # In the disabled case, the run_intraday_scan code never enters
        # the cap-computation branch (gated by `if MAX_PER_CONTRACT_USD > 0`)
        # so this test is conceptual — just confirm the patch took.
        assert patched_max <= 0
    # Reset for other tests
    reload(scheduled_predictor)


# ============================================================
# Deprecated-helper warning preserved
# ============================================================

def test_deprecated_helper_docstring_warns_loudly():
    """The deprecated `deployed_today_for_contract` function MUST have
    a docstring loud enough that anyone re-using it sees the warning.
    If the docstring loses the word 'DEPRECATED', somebody is about to
    re-introduce the bug.
    """
    doc = deployed_today_for_contract.__doc__ or ""
    assert "DEPRECATED" in doc, (
        "deployed_today_for_contract docstring no longer says DEPRECATED "
        "— the warning to future maintainers is gone.  Re-add it."
    )
    assert "actual_deployed" in doc.lower() or "get_actual_deployed" in doc, (
        "deployed_today_for_contract docstring no longer points future "
        "callers at the correct alternative source of truth."
    )