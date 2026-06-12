"""
test_forecast_recovery.py — Lock in the fix for the NWS hourly
evening-scan bug.

Bug: NWS /forecastHourly only returns periods from "now" forward.  By
late afternoon, today's "remaining" periods describe only the evening
cooling curve.  A 9pm scan therefore sees a forecast_high of ~17°C (the
last few hours of today before midnight) instead of the actual day's
high that was visible in the morning (e.g., ~28°C for SF in June).

Fix (recover_persisted_day_forecast): if a prior scan for the same
(city, event_date) recorded a higher forecast_high_c, use that.

Verification reference: SF 2026-06-11 in production data.
  - 07:04 UTC (00:04 PDT, fresh fetch): 28.33°C  ← real day high
  - 28.33 stable through 23:06 UTC      (16:06 PDT)
  - 23:08 UTC onward: ratchets down — 27.78, 26.67, 25.56, 24.44,
    20.56, 19.44, 18.33, 17.22  ← buggy
  - 17.22°C at 23:24 UTC = 16:24 PDT, exactly tracking the cooling curve

Run:
    cd bot
    python -m pytest tests/test_forecast_recovery.py -v
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
    recover_persisted_day_forecast,
    _SCHEMA_SQL,
)


def _fresh_db() -> sqlite3.Connection:
    """In-memory DB with the production schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    return conn


def _insert_signal(conn, city: str, event_date: str,
                    forecast_high_c: float | None,
                    forecast_peak_hour: int | None = 15,
                    scanned_at_utc: str = "2026-06-11T07:04:00Z") -> None:
    conn.execute(
        """INSERT INTO paper_predictor_signals
            (scanned_at_utc, mode, city, event_date,
             contract_id, bin_label, forecast_high_c, forecast_peak_hour, action)
            VALUES (?, 'live', ?, ?, 'c1', '70-71F', ?, ?, 'SKIP')""",
        (scanned_at_utc, city, event_date, forecast_high_c, forecast_peak_hour),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Recovery logic
# ---------------------------------------------------------------------------

def test_recovery_prefers_higher_persisted_value():
    """The exact SF 2026-06-11 case: morning scan stored 28.33°C; evening
    scan tries to write 17.22°C; recovery returns 28.33."""
    conn = _fresh_db()
    _insert_signal(conn, "San Francisco", "2026-06-11",
                    forecast_high_c=28.33, forecast_peak_hour=15)
    high, peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=17.22, candidate_peak_hour=23,
    )
    assert abs(high - 28.33) < 1e-9, f"expected 28.33, got {high}"
    assert peak == 15, f"expected peak_hour=15 (morning), got {peak}"


def test_recovery_keeps_higher_candidate():
    """Open-Meteo fallback (no hourly-endpoint bug) returns a higher
    value than what's persisted — prefer the fresh candidate."""
    conn = _fresh_db()
    _insert_signal(conn, "San Francisco", "2026-06-11",
                    forecast_high_c=17.22)
    high, peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=28.33, candidate_peak_hour=15,
    )
    assert high == 28.33, "should keep higher candidate over stale persisted"
    assert peak == 15


def test_recovery_no_data_returns_candidate():
    """Cold start: no prior signal rows for this (city, event_date).
    Helper must not pretend it found data — return the candidate
    unchanged (still buggy on cold-start days, but no worse)."""
    conn = _fresh_db()
    high, peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=17.22, candidate_peak_hour=23,
    )
    assert high == 17.22, "no prior data → candidate unchanged"
    assert peak == 23


def test_recovery_isolated_by_city():
    """A morning scan for Denver must not bleed into SF's recovery."""
    conn = _fresh_db()
    _insert_signal(conn, "Denver", "2026-06-11", forecast_high_c=35.0)
    high, _peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=17.22, candidate_peak_hour=23,
    )
    assert high == 17.22, f"Denver data must not affect SF, got {high}"


def test_recovery_isolated_by_event_date():
    """A morning scan for yesterday must not bleed into today's recovery."""
    conn = _fresh_db()
    _insert_signal(conn, "San Francisco", "2026-06-10",
                    forecast_high_c=35.0,
                    scanned_at_utc="2026-06-10T07:04:00Z")
    high, _peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=17.22, candidate_peak_hour=23,
    )
    assert high == 17.22, f"yesterday's high must not recover today, got {high}"


def test_recovery_picks_max_across_multiple_prior_scans():
    """Multiple scans throughout the day — pick the highest persisted
    value, not the most recent.  The afternoon's 27.78°C and the
    morning's 28.33°C are both persisted; helper picks 28.33."""
    conn = _fresh_db()
    _insert_signal(conn, "San Francisco", "2026-06-11",
                    forecast_high_c=28.33, forecast_peak_hour=15,
                    scanned_at_utc="2026-06-11T07:04:00Z")
    _insert_signal(conn, "San Francisco", "2026-06-11",
                    forecast_high_c=27.78, forecast_peak_hour=16,
                    scanned_at_utc="2026-06-11T23:08:00Z")
    _insert_signal(conn, "San Francisco", "2026-06-11",
                    forecast_high_c=20.56, forecast_peak_hour=20,
                    scanned_at_utc="2026-06-12T03:36:00Z")
    high, peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=17.22, candidate_peak_hour=23,
    )
    assert abs(high - 28.33) < 1e-9, f"should pick MAX persisted, got {high}"
    assert peak == 15, "peak hour should match the row with the max high"


def test_recovery_null_persisted_high_gracefully_handled():
    """Edge case: a row exists but its forecast_high_c is NULL (rare —
    a row from a different code path that didn't populate it).  Helper
    should fall through to the candidate, not crash."""
    conn = _fresh_db()
    _insert_signal(conn, "San Francisco", "2026-06-11",
                    forecast_high_c=None, forecast_peak_hour=None)
    high, peak = recover_persisted_day_forecast(
        conn, "San Francisco", "2026-06-11",
        candidate_high_c=22.0, candidate_peak_hour=14,
    )
    assert high == 22.0
    assert peak == 14