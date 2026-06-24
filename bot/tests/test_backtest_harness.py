"""
test_backtest_harness.py — locks in the backtest infrastructure.

Critical correctness pieces:
  - Brier scoring (multi-class) — 0 for perfect, 2 for fully wrong
  - log_loss returns inf for zero-probability winners
  - top_bin_correct tie-break is strict (ties don't count)
  - Bin assignment from integer
  - Three rounding-convention helpers in backtest_rounding
  - End-to-end on a fixture: discovery + hydration + scoring works
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.backtest_harness import (    # type: ignore
    Bin, Event, HourlyTemp, Method,
    brier_score, log_loss, top_bin_correct,
    assign_bin_for_integer,
    discover_resolved_events, hydrate_event,
    score_method_against_events,
)
from scripts.backtest_rounding import (   # type: ignore
    _truncate, _round_half_up, _round_half_even,
    _convert_to_settlement_unit,
    TruncationMethod, HalfUpMethod, HalfEvenMethod,
)


# ============================================================
# Scoring math
# ============================================================

class TestBrierScore:

    BINS = ["A", "B", "C"]

    def test_perfect_prediction(self):
        # All mass on winner → 0
        assert brier_score({"A": 1.0, "B": 0.0, "C": 0.0}, "A", self.BINS) == 0.0

    def test_completely_wrong(self):
        # All mass on a non-winner → 2.0 (the max for multi-class Brier)
        assert brier_score({"A": 1.0, "B": 0.0, "C": 0.0}, "B", self.BINS) == 2.0

    def test_uniform_three_bins_winner_a(self):
        # 1/3 each → (1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2 = 4/9 + 1/9 + 1/9 = 6/9
        v = brier_score({"A": 1/3, "B": 1/3, "C": 1/3}, "A", self.BINS)
        assert abs(v - 6/9) < 1e-9

    def test_missing_bin_treated_as_zero(self):
        # Method only returns 'A' → other bins implicit 0
        v = brier_score({"A": 1.0}, "A", self.BINS)
        assert v == 0.0


class TestLogLoss:

    def test_perfect_prediction_finite(self):
        # log_loss(p=1.0 for winner) = 0
        assert log_loss({"A": 1.0}, "A") == 0.0

    def test_zero_prob_for_winner_is_inf(self):
        # Catastrophic prediction
        assert log_loss({"A": 0.0, "B": 1.0}, "A") == float("inf")
        assert log_loss({}, "A") == float("inf")

    def test_uniform_three_bins(self):
        import math
        v = log_loss({"A": 1/3, "B": 1/3, "C": 1/3}, "A")
        assert abs(v - math.log(3)) < 1e-9


class TestTopBinCorrect:

    def test_unique_max_on_winner(self):
        assert top_bin_correct({"A": 0.6, "B": 0.3, "C": 0.1}, "A") is True

    def test_unique_max_not_on_winner(self):
        assert top_bin_correct({"A": 0.6, "B": 0.3, "C": 0.1}, "B") is False

    def test_tie_at_top_not_counted_as_correct(self):
        # Strict argmax: a tie is not a win, no matter which is the "winner"
        # — we don't want methods to score correct on truly ambiguous cases
        assert top_bin_correct({"A": 0.5, "B": 0.5}, "A") is False

    def test_empty_prediction(self):
        assert top_bin_correct({}, "A") is False


# ============================================================
# Bin assignment
# ============================================================

class TestBinAssignment:

    BINS = [
        Bin(label="86-87°F", range_low=86.0, range_high=87.0, unit="fahrenheit"),
        Bin(label="88-89°F", range_low=88.0, range_high=89.0, unit="fahrenheit"),
        Bin(label="90-91°F", range_low=90.0, range_high=91.0, unit="fahrenheit"),
    ]

    def test_value_inside_bin(self):
        assert assign_bin_for_integer(86, self.BINS).label == "86-87°F"
        assert assign_bin_for_integer(87, self.BINS).label == "86-87°F"
        assert assign_bin_for_integer(88, self.BINS).label == "88-89°F"

    def test_value_below_all_bins(self):
        assert assign_bin_for_integer(80, self.BINS) is None

    def test_value_above_all_bins(self):
        assert assign_bin_for_integer(95, self.BINS) is None


# ============================================================
# Rounding helpers — must lock in exact behavior so the
# convention test gives the right answer
# ============================================================

class TestRoundingHelpers:

    def test_truncate(self):
        assert _truncate(85.0) == 85
        assert _truncate(85.99) == 85
        assert _truncate(85.5) == 85
        assert _truncate(85.01) == 85

    def test_half_up(self):
        # 85.5 is the boundary case: half-up rounds UP
        assert _round_half_up(85.5) == 86
        assert _round_half_up(85.49) == 85
        assert _round_half_up(85.49999) == 85
        assert _round_half_up(86.0) == 86

    def test_half_even_bankers(self):
        # Python's built-in round() is banker's rounding
        assert _round_half_even(85.5) == 86   # rounds to even
        assert _round_half_even(84.5) == 84   # rounds to even
        assert _round_half_even(85.49) == 85
        assert _round_half_even(85.51) == 86

    def test_conventions_disagree_on_halves(self):
        """The whole point of the test: confirm these three CAN disagree."""
        assert _truncate(85.5) != _round_half_up(85.5)
        assert _truncate(84.5) != _round_half_up(84.5)
        # Half-even agrees with half-up on .5 → 86 (since 86 is even)
        # but disagrees on 84.5 → 84 (even) vs half-up 85
        assert _round_half_even(84.5) != _round_half_up(84.5)


class TestUnitConversion:

    def test_us_celsius_to_fahrenheit(self):
        # 30°C = 86°F
        v = _convert_to_settlement_unit(30.0, "fahrenheit")
        assert abs(v - 86.0) < 1e-9

    def test_intl_celsius_passthrough(self):
        v = _convert_to_settlement_unit(30.0, "celsius")
        assert v == 30.0


# ============================================================
# Method protocol — end-to-end on a fixture
# ============================================================

class _ConstantMethod(Method):
    """Test fixture: always returns the same prediction."""
    def __init__(self, prediction: dict):
        self.name = "constant"
        self._prediction = prediction

    def predict(self, event):
        return dict(self._prediction)


def _make_test_event(winning_label: str = "88-89°F",
                       actual_max_c: float = 31.0) -> Event:
    """Lightweight Event fixture for scoring tests."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("America/New_York")
    bins = [
        Bin("86-87°F", 86.0, 87.0, "fahrenheit"),
        Bin("88-89°F", 88.0, 89.0, "fahrenheit"),
        Bin("90-91°F", 90.0, 91.0, "fahrenheit"),
    ]
    winner = next(b for b in bins if b.label == winning_label)
    return Event(
        event_id="test-evt-1",
        city="Miami", event_date="2026-06-20",
        icao="KMIA", tz_str="America/New_York",
        settlement_unit="fahrenheit",
        bins=bins, winning_bin=winner,
        hourly_temps=[HourlyTemp(
            timestamp_utc="2026-06-20T18:00:00+00:00",
            timestamp_local=_dt(2026, 6, 20, 14, 0, tzinfo=tz),
            temp_c=actual_max_c, temp_precision="tenths",
        )],
    )


def test_score_method_picks_up_perfect_prediction():
    event = _make_test_event(winning_label="88-89°F")
    method = _ConstantMethod({"88-89°F": 1.0})
    res = score_method_against_events(method, [event])
    assert res["n_events"] == 1
    assert res["mean_brier"] == 0.0
    assert res["top_correct_rate"] == 1.0
    assert res["mean_log_loss_finite"] == 0.0


def test_score_method_picks_up_completely_wrong():
    event = _make_test_event(winning_label="88-89°F")
    method = _ConstantMethod({"90-91°F": 1.0})
    res = score_method_against_events(method, [event])
    assert res["mean_brier"] == 2.0
    assert res["top_correct_rate"] == 0.0
    assert res["n_log_loss_inf"] == 1   # zero prob on winner


# ============================================================
# Rounding-method end-to-end
# ============================================================

class TestRoundingMethodsEndToEnd:
    """Lock in exactly what each rounding method predicts on edge cases.
    These are the cases that DECIDE the Phase 1 verdict."""

    def test_halfway_case_85_5f_truncation_vs_halfup_differ(self):
        # actual_max_c → settlement temp 85.5°F
        # truncate -> 85 -> 84-85 bin
        # half-up -> 86 -> 86-87 bin
        bins = [
            Bin("84-85°F", 84.0, 85.0, "fahrenheit"),
            Bin("86-87°F", 86.0, 87.0, "fahrenheit"),
        ]
        # 85.5°F = 29.722°C exactly
        event = Event(
            event_id="halfway", city="TestCity",
            event_date="2026-06-20", icao="KTST",
            tz_str="America/New_York", settlement_unit="fahrenheit",
            bins=bins, winning_bin=bins[0],   # arbitrary; just need predict()
            hourly_temps=[HourlyTemp(
                timestamp_utc="2026-06-20T18:00:00+00:00",
                timestamp_local=None, temp_c=(85.5 - 32) * 5/9,
                temp_precision="tenths")],
        )
        # Truncation predicts 84-85 bin
        assert TruncationMethod.predict(event) == {"84-85°F": 1.0}
        # Half-up predicts 86-87 bin
        assert HalfUpMethod.predict(event) == {"86-87°F": 1.0}

    def test_no_matching_bin_returns_empty(self):
        # Actual max way outside the available bin range
        bins = [Bin("90-91°F", 90.0, 91.0, "fahrenheit")]
        event = Event(
            event_id="oor", city="TestCity",
            event_date="2026-06-20", icao="KTST",
            tz_str="America/New_York", settlement_unit="fahrenheit",
            bins=bins, winning_bin=bins[0],
            hourly_temps=[HourlyTemp(
                timestamp_utc="2026-06-20T18:00:00+00:00",
                timestamp_local=None, temp_c=10.0, temp_precision="tenths")],
        )
        # 50°F falls below the only bin → empty prediction
        assert TruncationMethod.predict(event) == {}
        assert HalfUpMethod.predict(event) == {}


# ============================================================
# Database integration — discovery + hydration on a tiny in-memory DB
# ============================================================

def _setup_test_db():
    """In-memory DB with the bare schema the backtest harness reads."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE paper_predictor_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at_utc TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'live',
            city TEXT NOT NULL,
            event_date TEXT,
            event_id TEXT,
            contract_id TEXT,
            bin_label TEXT,
            bin_range_low REAL,
            bin_range_high REAL,
            unit TEXT,
            market_prob REAL
        );
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


def test_discover_resolved_events_finds_winning_bin():
    conn = _setup_test_db()
    # One event with two bins; the 88-89 bin has market_prob 0.99 → winner
    base = ("2026-06-20T15:00:00+00:00", "live", "Miami",
            "2026-06-20", "evt-1")
    conn.execute(
        "INSERT INTO paper_predictor_signals "
        "(scanned_at_utc, mode, city, event_date, event_id, "
        " contract_id, bin_label, bin_range_low, bin_range_high, unit, market_prob) "
        "VALUES (?,?,?,?,?, 'cid-1', '86-87°F', 86, 87, 'fahrenheit', 0.05)",
        base)
    conn.execute(
        "INSERT INTO paper_predictor_signals "
        "(scanned_at_utc, mode, city, event_date, event_id, "
        " contract_id, bin_label, bin_range_low, bin_range_high, unit, market_prob) "
        "VALUES (?,?,?,?,?, 'cid-2', '88-89°F', 88, 89, 'fahrenheit', 0.99)",
        base)
    conn.commit()
    events = discover_resolved_events(conn, days_back=30)
    assert len(events) == 1
    assert events[0]["winning_bin"].label == "88-89°F"
    assert len(events[0]["bins"]) == 2