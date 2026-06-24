"""
test_calendar_day_floor.py — Phase 4 (critic issue #4 fix).

Lock-in tests for the calendar-day floor convention used in the
TWC fusion path.  The settlement window is station-local calendar
day (00:00 → 23:59 local), NOT the 7am-anchored window TWC's
`temperatureMaxSince7Am` exposes.

Critical behaviors:
  - METAR cycles from the local-day before 07:00 are captured
    (the whole reason we don't trust TWC's since-7am alone)
  - Cycles from yesterday's UTC date that fall on TODAY's local
    date are still counted (and vice versa)
  - max(metar_floor, twc_since7am) — neither is dropped silently
  - Unit conversion (raw_metar_log is °C; settlement may be °F)
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.twc_forecast_probe import (   # type: ignore
    _c_to_unit,
    _query_metar_calendar_day_max_c,
    compute_calendar_day_floor,
)


# ============================================================
# Fixtures
# ============================================================

def _mk_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE raw_metar_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icao TEXT NOT NULL,
            event_date TEXT NOT NULL,
            cycle_timestamp_utc TEXT NOT NULL,
            raw_message TEXT,
            temp_c REAL,
            dewpoint_c REAL,
            wind_dir_deg REAL,
            wind_speed_mps REAL,
            present_weather TEXT,
            temp_precision TEXT,
            persisted_at_utc TEXT NOT NULL,
            UNIQUE(icao, cycle_timestamp_utc)
        );
    """)
    return conn


def _insert(conn, icao, event_date, ts_utc_iso, temp_c):
    conn.execute(
        """INSERT INTO raw_metar_log
             (icao, event_date, cycle_timestamp_utc, temp_c,
              temp_precision, persisted_at_utc)
           VALUES (?,?,?,?,?,?)""",
        (icao, event_date, ts_utc_iso, temp_c, "whole",
         "2026-06-24T00:00:00+00:00"),
    )
    conn.commit()


# ============================================================
# _c_to_unit
# ============================================================

class TestCelsiusConversion:
    def test_fahrenheit_round_trip(self):
        # 0°C = 32°F, 100°C = 212°F
        assert abs(_c_to_unit(0, "fahrenheit") - 32.0) < 1e-9
        assert abs(_c_to_unit(100, "fahrenheit") - 212.0) < 1e-9

    def test_celsius_is_identity(self):
        assert _c_to_unit(25.7, "celsius") == 25.7

    def test_default_to_celsius(self):
        assert _c_to_unit(25.0, "") == 25.0


# ============================================================
# _query_metar_calendar_day_max_c
# ============================================================

class TestMetarCalendarDayMax:

    def test_empty_table_returns_none(self):
        conn = _mk_conn()
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v is None

    def test_picks_highest_temp_in_calendar_day(self):
        conn = _mk_conn()
        # All on 2026-06-24 local (EDT = UTC-4)
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T05:00:00+00:00", 24.0)  # 01:00 EDT
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T15:00:00+00:00", 30.0)  # 11:00 EDT
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T19:00:00+00:00", 32.5)  # 15:00 EDT
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v == 32.5

    def test_captures_overnight_pre_7am_max(self):
        """The whole point — an overnight max BEFORE 7am must be captured."""
        conn = _mk_conn()
        # Overnight cold-front max at 03:00 local
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T07:00:00+00:00", 31.5)  # 03:00 EDT
        # Lower temperatures during the day
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T17:00:00+00:00", 27.0)  # 13:00 EDT
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v == 31.5

    def test_filters_out_other_local_day(self):
        """A cycle whose UTC-date matches event_date but whose LOCAL
        date is the prior or next day must be excluded."""
        conn = _mk_conn()
        # Cycle stored as event_date=2026-06-24 with UTC ts 03:00 UTC.
        # In EDT (UTC-4), that converts to 2026-06-23 23:00 local —
        # prior local day.  Must be dropped.
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T03:00:00+00:00", 40.0)  # 23:00 prior day local
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T15:00:00+00:00", 28.0)  # 11:00 today local
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        # The 40.0 cycle is from yesterday local, must be dropped
        assert v == 28.0

    def test_filters_out_other_icao(self):
        conn = _mk_conn()
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T17:00:00+00:00", 30.0)
        _insert(conn, "KSFO", "2026-06-24",
                "2026-06-24T17:00:00+00:00", 35.0)
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v == 30.0

    def test_bad_timestamp_skipped(self):
        conn = _mk_conn()
        _insert(conn, "KMIA", "2026-06-24",
                "not-a-timestamp", 99.0)
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T17:00:00+00:00", 28.0)
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v == 28.0

    def test_null_temp_skipped(self):
        conn = _mk_conn()
        conn.execute(
            """INSERT INTO raw_metar_log
                 (icao, event_date, cycle_timestamp_utc, temp_c,
                  temp_precision, persisted_at_utc)
               VALUES (?,?,?,?,?,?)""",
            ("KMIA", "2026-06-24", "2026-06-24T17:00:00+00:00",
             None, "whole", "2026-06-24T00:00:00+00:00"),
        )
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T18:00:00+00:00", 27.0)
        conn.commit()
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v == 27.0

    def test_table_missing_returns_none(self):
        """Tolerate the OperationalError path for fresh DBs without
        the table — must not raise."""
        conn = sqlite3.connect(":memory:")
        v = _query_metar_calendar_day_max_c(
            conn, "KMIA", "2026-06-24", "America/New_York")
        assert v is None


# ============================================================
# compute_calendar_day_floor — combined helper
# ============================================================

class TestComputeCalendarDayFloor:

    def test_no_metar_no_twc_returns_none(self):
        conn = _mk_conn()
        floor, note = compute_calendar_day_floor(
            conn, "KMIA", "2026-06-24", "America/New_York",
            twc_max_since_7am=None, settlement_unit="fahrenheit")
        assert floor is None
        assert "no observed-max" in note.lower()

    def test_twc_only_when_no_metar(self):
        conn = _mk_conn()
        floor, note = compute_calendar_day_floor(
            conn, "KMIA", "2026-06-24", "America/New_York",
            twc_max_since_7am=88.0, settlement_unit="fahrenheit")
        assert floor == 88.0
        assert "twc7am=88.0°F" in note
        assert "calendar-day" in note

    def test_metar_only_when_no_twc_converts_to_fahrenheit(self):
        conn = _mk_conn()
        # 30°C = 86°F
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T18:00:00+00:00", 30.0)
        floor, note = compute_calendar_day_floor(
            conn, "KMIA", "2026-06-24", "America/New_York",
            twc_max_since_7am=None, settlement_unit="fahrenheit")
        assert abs(floor - 86.0) < 1e-6
        assert "metar=86.0°F" in note

    def test_metar_exceeds_twc_wins(self):
        """The 'overnight max before 7am' case: METAR floor must beat
        TWC's since-7am floor."""
        conn = _mk_conn()
        # 03:00 EDT cycle = 35°C = 95°F (overnight max)
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T07:00:00+00:00", 35.0)
        floor, note = compute_calendar_day_floor(
            conn, "KMIA", "2026-06-24", "America/New_York",
            twc_max_since_7am=88.0, settlement_unit="fahrenheit")
        assert abs(floor - 95.0) < 1e-6
        # Note should show both candidates
        assert "metar=95.0°F" in note
        assert "twc7am=88.0°F" in note

    def test_twc_exceeds_metar_wins(self):
        """The 'TWC CC fresher than latest METAR' case."""
        conn = _mk_conn()
        _insert(conn, "KMIA", "2026-06-24",
                "2026-06-24T18:00:00+00:00", 30.0)  # 86°F
        floor, note = compute_calendar_day_floor(
            conn, "KMIA", "2026-06-24", "America/New_York",
            twc_max_since_7am=89.5, settlement_unit="fahrenheit")
        assert abs(floor - 89.5) < 1e-6
        assert "metar=86.0°F" in note
        assert "twc7am=89.5°F" in note

    def test_celsius_unit_no_conversion(self):
        conn = _mk_conn()
        _insert(conn, "EGLC", "2026-06-24",
                "2026-06-24T13:00:00+00:00", 24.0)  # 13:00 UTC = 14:00 BST
        floor, note = compute_calendar_day_floor(
            conn, "EGLC", "2026-06-24", "Europe/London",
            twc_max_since_7am=23.0, settlement_unit="celsius")
        assert floor == 24.0
        assert "°C" in note
        assert "°F" not in note