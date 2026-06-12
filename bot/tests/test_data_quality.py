"""
test_data_quality.py — Tests for the data-quality contract pieces:
  - is_cold_start_day detection
  - compute_data_quality_size_factor scalar computation
  - composable flag resolution (gaussian,cold_start_suspect → MIN)

Companion: docs/data_quality_contract.md
"""

from __future__ import annotations

import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scheduled_predictor import (   # type: ignore
    _SCHEMA_SQL,
    is_cold_start_day,
    compute_data_quality_size_factor,
    COLD_START_PEAK_HOUR_LOCAL,
    DATA_QUALITY_SIZE_COLD_START_SUSPECT,
    DATA_QUALITY_SIZE_GAUSSIAN,
    DATA_QUALITY_SIZE_GAUSSIAN_DEFAULT_SIGMA,
    DATA_QUALITY_SIZE_BLOCK,
)


def _fresh_db():
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


def _insert(conn, city, event_date, scanned_at_utc):
    conn.execute(
        """INSERT INTO paper_predictor_signals
            (scanned_at_utc, mode, city, event_date, contract_id,
             bin_label, action)
            VALUES (?, 'live', ?, ?, 'c1', '70-71F', 'SKIP')""",
        (scanned_at_utc, city, event_date),
    )
    conn.commit()


# ============================================================
# is_cold_start_day
# ============================================================

def test_cold_start_first_scan_post_peak():
    """First scan of the day is at 18:00 PDT — past the 14:00 cold-start
    cutoff.  Should flag cold-start."""
    conn = _fresh_db()
    # 2026-06-12T01:00:00Z = 2026-06-11T18:00 PDT (yesterday in local)
    # Need a scan that's TODAY in local.  Use 2026-06-12T22:00:00Z =
    # 2026-06-12T15:00 PDT — 15:00 local is past the 14:00 cutoff.
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T22:00:00+00:00")
    result = is_cold_start_day(
        conn, "San Francisco", "2026-06-12",
        "America/Los_Angeles",
        "2026-06-12T23:00:00+00:00",
    )
    assert result is True, (
        f"first scan at 15:00 PDT (after {COLD_START_PEAK_HOUR_LOCAL}:00 cutoff) "
        "should flag cold-start"
    )


def test_cold_start_first_scan_pre_peak():
    """First scan at 08:00 PDT (well before the 14:00 cutoff).  Should
    NOT flag cold-start — the day's morning view of the forecast peak
    was captured."""
    conn = _fresh_db()
    # 2026-06-12T15:00:00Z = 08:00 PDT
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T15:00:00+00:00")
    result = is_cold_start_day(
        conn, "San Francisco", "2026-06-12",
        "America/Los_Angeles",
        "2026-06-12T23:00:00+00:00",
    )
    assert result is False, (
        "first scan at 08:00 PDT should NOT flag cold-start"
    )


def test_cold_start_no_prior_scans_uses_candidate():
    """No prior scans for this (city, event_date) — the helper falls
    back to using the candidate scan time as the 'first scan'.  Useful
    for the very first scan of the day flagging itself."""
    conn = _fresh_db()
    # No inserts; candidate is at 17:00 PDT (post-cutoff)
    result = is_cold_start_day(
        conn, "San Francisco", "2026-06-12",
        "America/Los_Angeles",
        "2026-06-13T00:00:00+00:00",   # 17:00 PDT
    )
    assert result is True


def test_cold_start_isolated_by_city():
    """Denver's morning scan must not let SF off the hook."""
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12", "2026-06-12T13:00:00+00:00")
    # SF candidate at 17:00 PDT, no SF scans yet
    result = is_cold_start_day(
        conn, "San Francisco", "2026-06-12",
        "America/Los_Angeles",
        "2026-06-13T00:00:00+00:00",
    )
    assert result is True, "SF should still flag — Denver's prior is irrelevant"


def test_cold_start_isolated_by_event_date():
    """Yesterday's morning scan must not affect today's cold-start
    detection."""
    conn = _fresh_db()
    _insert(conn, "San Francisco", "2026-06-11", "2026-06-11T15:00:00+00:00")
    # Today is 06-12; first scan candidate at 17:00 PDT
    result = is_cold_start_day(
        conn, "San Francisco", "2026-06-12",
        "America/Los_Angeles",
        "2026-06-13T00:00:00+00:00",
    )
    assert result is True


# ============================================================
# compute_data_quality_size_factor
# ============================================================

def test_size_factor_none_flag_returns_one():
    """Default behavior: no flag = no haircut."""
    assert compute_data_quality_size_factor(None) == 1.00


def test_size_factor_empty_flag_returns_one():
    assert compute_data_quality_size_factor("") == 1.00


def test_size_factor_gaussian_is_neutral_today():
    """Per the contract: relative tiers all at 1.00 because no PRIMARY
    tier exists yet.  When a PRIMARY tier ships, set
    DATA_QUALITY_SIZE_GAUSSIAN below 1.0."""
    assert compute_data_quality_size_factor("gaussian") == DATA_QUALITY_SIZE_GAUSSIAN
    assert DATA_QUALITY_SIZE_GAUSSIAN == 1.00, (
        "Gaussian haircut should be 1.00 today — see contract section "
        "'Two different kinds of haircut'"
    )


def test_size_factor_cold_start_haircut():
    """Cold-start is absolute trustability — fires at 0.30 regardless
    of the relative tier."""
    assert compute_data_quality_size_factor("cold_start_suspect") == (
        DATA_QUALITY_SIZE_COLD_START_SUSPECT
    )
    assert DATA_QUALITY_SIZE_COLD_START_SUSPECT == 0.30


def test_size_factor_composable_takes_min():
    """gaussian,cold_start_suspect → min(1.00, 0.30) = 0.30."""
    factor = compute_data_quality_size_factor("gaussian,cold_start_suspect")
    assert factor == 0.30


def test_size_factor_composable_three_flags():
    """gaussian_default_sigma,cold_start_suspect,gaussian → 0.30."""
    factor = compute_data_quality_size_factor(
        "gaussian_default_sigma,cold_start_suspect,gaussian")
    assert factor == 0.30


def test_size_factor_block_dominates():
    """BLOCK is 0.00 — overrides everything else.  Defense in depth
    for catastrophic-failure cases."""
    factor = compute_data_quality_size_factor("gaussian,block")
    assert factor == DATA_QUALITY_SIZE_BLOCK
    assert factor == 0.00


def test_size_factor_strips_reason_suffix():
    """Flag values can carry a ':reason' suffix (e.g.
    'primary_fallback:stale_11h').  The base flag is what looks up to
    the multiplier — the reason is informational for audits."""
    factor = compute_data_quality_size_factor("gaussian:cdf_used_no_residuals")
    assert factor == DATA_QUALITY_SIZE_GAUSSIAN


def test_size_factor_unknown_flag_neutral():
    """A flag we don't know about (e.g. a new tier added by a future
    workstream that hasn't updated the table) returns 1.00 — neutral.
    This is fail-open by design: an unknown flag shouldn't accidentally
    haircut sizing.  The contract says known absolute-trustability
    flags fire their haircut; unknown flags are observational only."""
    factor = compute_data_quality_size_factor("future_unknown_tier")
    assert factor == 1.00


def test_size_factor_known_and_unknown_mixed():
    """Mix of known + unknown: known still drives the result."""
    factor = compute_data_quality_size_factor(
        "future_unknown_tier,cold_start_suspect")
    assert factor == DATA_QUALITY_SIZE_COLD_START_SUSPECT