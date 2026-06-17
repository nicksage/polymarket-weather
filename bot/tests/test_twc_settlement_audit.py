"""
test_twc_settlement_audit.py — lock in the load-bearing pieces of the
TWC settlement-truth audit script:

  * half-up rounding matches Polymarket convention (93.5 -> 94, NOT 93)
  * bin assignment: a TWC reading falls inside the winning bin iff
    its rounded settlement value is between winning_low and
    winning_high INCLUSIVE on both ends (US 2°F bins span two rounded
    values)
  * unit conversion: °C TWC response -> °F settlement
  * the audit script never makes HTTP calls under --dry-run
  * the schema bootstrap creates the table if missing

Run:
    cd bot && python -m pytest tests/test_twc_settlement_audit.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.twc_settlement_audit import (   # type: ignore
    _round_half_up_int,
    _ensure_settlement_unit_value,
    assign_bin_in_settlement_unit,
    ensure_audit_table,
    twc_daily_summary_max,
    twc_sitebased_daily_max,
)


# ============================================================
# Half-up rounding — the rounding-bug regression test
# ============================================================

class TestHalfUpRounding:

    def test_half_rounds_up_not_to_even(self):
        # Banker's rounding (round half to even) would send 92.5 -> 92.
        # Polymarket uses half-up, so 92.5 -> 93.
        assert _round_half_up_int(92.5) == 93
        assert _round_half_up_int(93.5) == 94
        assert _round_half_up_int(94.5) == 95

    def test_below_half_rounds_down(self):
        assert _round_half_up_int(92.49) == 92
        assert _round_half_up_int(93.499999) == 93

    def test_exact_integer(self):
        assert _round_half_up_int(94.0) == 94
        assert _round_half_up_int(0.0) == 0


# ============================================================
# Bin assignment — the comparator that decides MATCH vs miss
# ============================================================

class TestBinAssignment:
    """US Polymarket bins are 2°F wide.  '94-95°F' bin = winning_low=94,
    winning_high=95.  A TWC reading whose rounded settlement value is
    94 OR 95 matches this bin; 93 or 96 do not."""

    LO, HI = 94.0, 95.0

    def test_rounded_value_inside_bin_matches(self):
        for raw_f in (94.0, 94.49, 94.5, 95.0, 95.49):
            _, _, matched = assign_bin_in_settlement_unit(raw_f, self.LO, self.HI)
            assert matched is True, f"{raw_f}°F should be inside 94-95 bin"

    def test_rounded_value_below_bin_misses(self):
        for raw_f in (93.49, 93.0, 90.0):
            _, _, matched = assign_bin_in_settlement_unit(raw_f, self.LO, self.HI)
            assert matched is False, f"{raw_f}°F should be outside 94-95 bin"

    def test_rounded_value_above_bin_misses(self):
        for raw_f in (95.5, 96.0, 100.0):
            _, _, matched = assign_bin_in_settlement_unit(raw_f, self.LO, self.HI)
            assert matched is False, f"{raw_f}°F should be outside 94-95 bin"

    def test_assignment_records_rounded_value(self):
        bin_lo, bin_hi, _ = assign_bin_in_settlement_unit(94.7, self.LO, self.HI)
        assert bin_lo == 95.0   # 94.7 rounds up to 95
        assert bin_hi == 95.0


# ============================================================
# Unit conversion — TWC may return °C even when we asked for °F
# ============================================================

class TestUnitConversion:

    def test_f_to_f_passthrough(self):
        assert _ensure_settlement_unit_value(93.5, "F", "fahrenheit") == 93.5

    def test_c_to_f_conversion(self):
        # 34°C = 93.2°F
        v = _ensure_settlement_unit_value(34.0, "C", "fahrenheit")
        assert abs(v - 93.2) < 1e-6

    def test_c_to_c_passthrough(self):
        assert _ensure_settlement_unit_value(28.0, "C", "celsius") == 28.0

    def test_f_to_c_conversion(self):
        v = _ensure_settlement_unit_value(95.0, "F", "celsius")
        assert abs(v - 35.0) < 1e-6

    def test_none_returns_none(self):
        assert _ensure_settlement_unit_value(None, "F", "fahrenheit") is None


# ============================================================
# Schema bootstrap — script must work standalone
# ============================================================

def test_ensure_audit_table_creates_schema():
    conn = sqlite3.connect(":memory:")
    ensure_audit_table(conn)
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(twc_settlement_audit)").fetchall()]
    # Load-bearing columns the reporter and audit_one read by name.
    for required in ("event_id", "polymarket_winning_low",
                       "polymarket_winning_high",
                       "twc_dailysummary_max", "dailysummary_match",
                       "twc_sitebased_max", "sitebased_match",
                       "captured_at_utc"):
        assert required in cols, f"missing column {required}"


def test_ensure_audit_table_is_idempotent():
    """Running ensure twice in a row should never raise."""
    conn = sqlite3.connect(":memory:")
    ensure_audit_table(conn)
    ensure_audit_table(conn)


# ============================================================
# Dry-run safety — must not make HTTP calls
# ============================================================

class TestDryRunSafety:
    """The script's most dangerous failure mode would be burning
    through TWC trial credits accidentally.  The dry-run path must
    NEVER touch httpx.get."""

    def test_daily_summary_dry_run_skips_http(self, monkeypatch):
        called = {"n": 0}
        def boom(*a, **k):
            called["n"] += 1
            raise RuntimeError("httpx.get must not be called under dry_run")
        monkeypatch.setattr("httpx.get", boom)
        # Even with no API key, dry_run should succeed without HTTP
        monkeypatch.setenv("TWC_API_KEY", "")
        v, unit, window, notes = twc_daily_summary_max(
            "KMIA", "2026-06-15", "fahrenheit", dry_run=True)
        assert v is None
        assert notes == "dry_run"
        assert called["n"] == 0

    def test_sitebased_dry_run_skips_http(self, monkeypatch):
        called = {"n": 0}
        def boom(*a, **k):
            called["n"] += 1
            raise RuntimeError("httpx.get must not be called under dry_run")
        monkeypatch.setattr("httpx.get", boom)
        monkeypatch.setenv("TWC_API_KEY", "")
        v, unit, n, notes = twc_sitebased_daily_max(
            "KMIA", "2026-06-15", "fahrenheit", dry_run=True)
        assert v is None
        assert notes == "dry_run"
        assert called["n"] == 0


# ============================================================
# Integration probe — end-to-end audit_one with mocked TWC
# ============================================================

def test_audit_one_writes_row_and_records_match(monkeypatch):
    """Full happy path: mock both TWC calls to return readings that
    fall inside the 94-95 winning bin, confirm the row is written
    with match=1 for both candidates."""
    from scripts import twc_settlement_audit as twc_mod

    # Mock both TWC endpoints
    monkeypatch.setattr(twc_mod, "twc_daily_summary_max",
        lambda icao, date_iso, su, *, dry_run: (94.7, "F", "7am-7am-local", ""))
    monkeypatch.setattr(twc_mod, "twc_sitebased_daily_max",
        lambda icao, date_iso, su, *, dry_run: (95.0, "F", 24, ""))

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_audit_table(conn)

    ev = {
        "event_id":       "test-event-001",
        "city":           "Miami",
        "event_date":     "2026-06-15",
        "icao":           "KMIA",
        "settlement_unit": "fahrenheit",
        "winning_low":    94.0,
        "winning_high":   95.0,
        "winning_label":  "94-95°F",
    }
    row = twc_mod.audit_one(conn, ev, dry_run=False, rate_limit_ms=0)
    assert row["dailysummary_match"] == 1
    assert row["sitebased_match"] == 1
    # Persisted row should be readable back
    db_row = dict(conn.execute(
        "SELECT * FROM twc_settlement_audit WHERE event_id = ?",
        (ev["event_id"],)
    ).fetchone())
    assert db_row["dailysummary_match"] == 1
    assert db_row["sitebased_match"] == 1


def test_audit_one_records_miss_on_wrong_bin(monkeypatch):
    """Negative case: TWC returns a reading that lands OUTSIDE the
    winning bin → match=0 (not NULL; NULL means API failure)."""
    from scripts import twc_settlement_audit as twc_mod

    # Both candidates return 90°F → way below the 94-95 bin
    monkeypatch.setattr(twc_mod, "twc_daily_summary_max",
        lambda icao, date_iso, su, *, dry_run: (90.0, "F", "7am-7am-local", ""))
    monkeypatch.setattr(twc_mod, "twc_sitebased_daily_max",
        lambda icao, date_iso, su, *, dry_run: (90.2, "F", 24, ""))

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_audit_table(conn)
    ev = {
        "event_id":       "test-event-002",
        "city":           "Miami",
        "event_date":     "2026-06-15",
        "icao":           "KMIA",
        "settlement_unit": "fahrenheit",
        "winning_low":    94.0,
        "winning_high":   95.0,
        "winning_label":  "94-95°F",
    }
    row = twc_mod.audit_one(conn, ev, dry_run=False, rate_limit_ms=0)
    assert row["dailysummary_match"] == 0
    assert row["sitebased_match"] == 0


def test_audit_one_records_none_on_api_failure(monkeypatch):
    """When TWC returns None (e.g., HTTP 500, parse error), the match
    column is None — distinguishable from 0 (genuine miss)."""
    from scripts import twc_settlement_audit as twc_mod
    monkeypatch.setattr(twc_mod, "twc_daily_summary_max",
        lambda icao, date_iso, su, *, dry_run: (None, "", "", "http_error: 500"))
    monkeypatch.setattr(twc_mod, "twc_sitebased_daily_max",
        lambda icao, date_iso, su, *, dry_run: (None, "", 0, "no_observations"))
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_audit_table(conn)
    ev = {
        "event_id": "test-event-003", "city": "Miami",
        "event_date": "2026-06-15", "icao": "KMIA",
        "settlement_unit": "fahrenheit",
        "winning_low": 94.0, "winning_high": 95.0,
        "winning_label": "94-95°F",
    }
    row = twc_mod.audit_one(conn, ev, dry_run=False, rate_limit_ms=0)
    assert row["dailysummary_match"] is None
    assert row["sitebased_match"] is None
    assert "http_error" in row["dailysummary_notes"]