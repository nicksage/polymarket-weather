"""
test_boundary_watcher.py — Lock in the boundary-crossing latency
strategy's pure-logic primitives.

The mission-critical tests:

  1. Settlement-unit rounding: half-up convention matches Polymarket
  2. Trigger A (SPECI body) ONLY fires when the reading's rounding
     window fits ENTIRELY inside the bin — boundary jitter must NOT
     fire (the entire reason we redesigned the strategy)
  3. Trigger B (T-group) fires on confirmed, contradicts on below-bin
  4. Arming respects all four gates (forecast margin, market_p ceiling,
     heating time, post-peak disarm)
  5. Defaults are safe: BOUNDARY_STRATEGY_ENABLED defaults OFF,
     BOUNDARY_DRY_RUN defaults ON

Run:
    cd bot
    python -m pytest tests/test_boundary_watcher.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from boundary_watcher import (   # type: ignore
    BOUNDARY_STRATEGY_ENABLED,
    BOUNDARY_DRY_RUN,
    BOUNDARY_TARGET_STAKE_USD,
    BOUNDARY_MAX_ENTRY_PRICE,
    BOUNDARY_ARM_MAX_MARKET_PRICE,
    bin_settlement_range,
    c_to_f,
    f_to_c,
    reading_in_settlement_unit,
    settlement_unit_round,
    is_supported_bin,
    compute_arming_state,
    evaluate_trigger,
    in_hard_poll_window,
)


# ============================================================
# Safe defaults
# ============================================================

def test_strategy_defaults_off():
    """BOUNDARY_STRATEGY_ENABLED defaults OFF — explicit opt-in only.
    Bypassing real safety gates on real money requires the operator
    to consciously enable.  If a future commit flips this default, it
    silently turns on a strategy class the user hasn't approved."""
    if os.environ.get("BOUNDARY_STRATEGY_ENABLED"):
        pytest.skip("env override set")
    assert BOUNDARY_STRATEGY_ENABLED is False


def test_dry_run_defaults_on():
    """BOUNDARY_DRY_RUN defaults ON.  Flipping to 0 is a deliberate
    operator action gated by ≥30 logged events + favorable lookahead
    distribution + manual sign-off (per module docstring)."""
    if os.environ.get("BOUNDARY_DRY_RUN"):
        pytest.skip("env override set")
    assert BOUNDARY_DRY_RUN is True


def test_max_entry_price_is_tight():
    """The reviewer's specific guidance: BOUNDARY_MAX_ENTRY_PRICE
    must be tight (0.15-0.20).  If it drifts above 0.30 the strategy
    is chasing repriced moves not capturing latency.  See module
    docstring's Miami example for why."""
    if os.environ.get("BOUNDARY_MAX_ENTRY_PRICE"):
        pytest.skip("env override set")
    assert BOUNDARY_MAX_ENTRY_PRICE <= 0.30, (
        f"BOUNDARY_MAX_ENTRY_PRICE={BOUNDARY_MAX_ENTRY_PRICE} is too "
        f"high — at this price the market has largely repriced; you'd "
        f"be chasing not catching the latency edge."
    )


# ============================================================
# Settlement-unit rounding
# ============================================================

def test_c_to_f_correct():
    assert abs(c_to_f(0) - 32.0) < 1e-9
    assert abs(c_to_f(100) - 212.0) < 1e-9
    assert abs(c_to_f(32.2) - 89.96) < 1e-4   # T03220239 case


def test_f_to_c_correct():
    assert abs(f_to_c(32.0) - 0.0) < 1e-9
    assert abs(f_to_c(212.0) - 100.0) < 1e-9


def test_reading_in_settlement_unit_us():
    """For a Fahrenheit-settled bin, °C reading converts to °F."""
    assert abs(reading_in_settlement_unit(32.2, "fahrenheit") - 89.96) < 1e-4


def test_reading_in_settlement_unit_eu():
    """For a Celsius-settled bin, °C reading passes through unchanged."""
    assert abs(reading_in_settlement_unit(32.2, "celsius") - 32.2) < 1e-9


def test_settlement_unit_round_half_up():
    """Half-up rounding: 93.5°F → 94, NOT 93.  Matches Polymarket bin
    convention."""
    # 93.5°F should round UP to 94 (half-up)
    # First we need a Celsius reading that converts to 93.5°F:
    # f_to_c(93.5) = (93.5 - 32) * 5/9 ≈ 34.1666°C
    c = f_to_c(93.5)
    assert settlement_unit_round(c, "fahrenheit") == 94


def test_settlement_unit_round_just_below():
    """93.4°F rounds DOWN to 93."""
    c = f_to_c(93.4)
    assert settlement_unit_round(c, "fahrenheit") == 93


# ============================================================
# Bin geometry — what we support and don't
# ============================================================

def test_supports_2f_us_bins():
    """The canonical case: '94-95°F' stored as lo=94 hi=95."""
    assert is_supported_bin(94, 95, "fahrenheit") is True


def test_rejects_celsius_bins():
    """v1 ships US-only.  EU °C bins skip silently."""
    assert is_supported_bin(28, 28, "celsius") is False


def test_rejects_1f_single_bins():
    """A '94°F' bin (lo=94, hi=94) is single-degree and strong-margin
    can never cleanly fire on it — skip in v1."""
    assert is_supported_bin(94, 94, "fahrenheit") is False


def test_rejects_open_ended_bins():
    """'≥100°F' (lo=100, hi=None) has no upper boundary to test
    rounding against."""
    assert is_supported_bin(100, None, "fahrenheit") is False
    assert is_supported_bin(None, 80, "fahrenheit") is False


def test_bin_settlement_range_94_95():
    """The 94-95°F bin should map to [93.5, 95.5)°F per half-up
    rounding semantics."""
    lo, hi = bin_settlement_range(94, 95, "fahrenheit")
    assert abs(lo - 93.5) < 1e-9
    assert abs(hi - 95.5) < 1e-9


# ============================================================
# Trigger A — SPECI body classification.  This is the WHOLE GAME.
# Boundary jitter MUST NOT fire as "strong" — that's the entire
# reason we redesigned the strategy.
# ============================================================

class TestTriggerASpeci:
    """All these use the 94-95°F bin (settlement range [93.5, 95.5)°F).
    Whole-°C body precision = ±0.5°C ≈ ±0.9°F rounding window."""

    BIN_LO = 94
    BIN_HI = 95
    UNIT = "fahrenheit"

    def _eval(self, temp_c):
        return evaluate_trigger(
            reading_c=temp_c, is_t_group=False,
            bin_lo=self.BIN_LO, bin_hi=self.BIN_HI,
            settlement_unit=self.UNIT,
        )

    def test_jitter_at_lower_boundary_does_not_fire_strong(self):
        """34°C body = 93.2°F.  True temp window [33.5, 34.5)°C ≈
        [92.3, 94.1)°F.  This SPANS the 93.5°F boundary — could be 93
        OR 94 in settlement.  This is the JITTER ZONE and must NOT
        classify as 'strong' (the entire reason for the redesign)."""
        tr = self._eval(34.0)
        assert tr.classification != "strong", (
            f"34°C body reading at boundary classified '{tr.classification}' "
            f"with would_fire={tr.would_fire} — this is the jitter case "
            f"the strategy was redesigned to exclude."
        )
        assert tr.would_fire is False

    def test_jitter_just_above_boundary_does_not_fire_strong(self):
        """35°C body = 95.0°F.  True temp window [34.5, 35.5)°C ≈
        [94.1, 95.9)°F.  Spans the 95.5°F upper boundary of the bin —
        could settle 94 OR 96.  Still jitter, still not strong."""
        tr = self._eval(35.0)
        assert tr.classification != "strong"
        assert tr.would_fire is False

    def test_body_outside_bin_no_signal(self):
        """30°C body = 86°F.  Window [85.1, 86.9)°F, way below bin.
        No signal."""
        tr = self._eval(30.0)
        assert tr.classification == "no_signal"
        assert tr.would_fire is False

    def test_body_above_bin_no_signal(self):
        """40°C body = 104°F.  Window [103.1, 104.9)°F, way above the
        94-95 bin (which tops at 95.5).  No signal — temp has skipped
        past the target bin."""
        tr = self._eval(40.0)
        assert tr.classification == "no_signal"
        assert tr.would_fire is False


# ============================================================
# Trigger B — T-group classification.  This is what actually
# fires the strategy.
# ============================================================

class TestTriggerBTGroup:
    BIN_LO = 94
    BIN_HI = 95
    UNIT = "fahrenheit"

    def _eval(self, temp_c):
        return evaluate_trigger(
            reading_c=temp_c, is_t_group=True,
            bin_lo=self.BIN_LO, bin_hi=self.BIN_HI,
            settlement_unit=self.UNIT,
        )

    def test_tenths_in_bin_confirmed(self):
        """34.4°C T-group = 93.92°F.  Inside [93.5, 95.5).  Confirmed."""
        tr = self._eval(34.4)
        assert tr.classification == "confirmed"
        assert tr.would_fire is True
        assert tr.size_usd == BOUNDARY_TARGET_STAKE_USD

    def test_tenths_just_above_boundary_confirmed(self):
        """f_to_c(93.6) ≈ 34.22°C.  93.6°F is just above 93.5 boundary.
        With tenths precision this is unambiguous — confirmed."""
        c = f_to_c(93.6)
        tr = self._eval(c)
        assert tr.classification == "confirmed"
        assert tr.would_fire is True

    def test_tenths_just_below_boundary_contradicted(self):
        """f_to_c(93.4) ≈ 34.11°C.  93.4°F is below 93.5 boundary.
        Tenths precision makes this unambiguous — contradicted."""
        c = f_to_c(93.4)
        tr = self._eval(c)
        assert tr.classification == "contradicted"
        assert tr.would_fire is False

    def test_tenths_above_bin_upper_no_signal(self):
        """f_to_c(96.0) ≈ 35.56°C.  Above bin's 95.5 upper edge.
        Temp skipped past — no signal for THIS bin (would fire on a
        higher bin if one were armed, but that's a different event)."""
        c = f_to_c(96.0)
        tr = self._eval(c)
        assert tr.classification == "no_signal"
        assert tr.would_fire is False


# ============================================================
# Phase 1 — Arming
# ============================================================

class TestArming:
    UNIT = "fahrenheit"
    BIN_LO = 94
    BIN_HI = 95

    def _arm(self, *,
                forecast_high_c=33.0,    # ~91.4°F, 2.1°F below 93.5 boundary
                forecast_peak_hour=15,
                current_local_hour=12,
                market_p=0.02,
                observed_max_c=20.0,
                ):
        return compute_arming_state(
            forecast_high_c=forecast_high_c,
            forecast_peak_hour=forecast_peak_hour,
            current_local_hour=current_local_hour,
            settlement_unit=self.UNIT,
            candidate_bin_lo=self.BIN_LO,
            candidate_bin_hi=self.BIN_HI,
            candidate_market_p=market_p,
            observed_max_c=observed_max_c,
        )

    def test_arms_when_all_conditions_met(self):
        """Forecast 33°C (91.4°F) is 2.1°F below boundary 93.5°F, which
        exceeds default 0.5°C (~0.9°F) margin. Should NOT arm."""
        st = self._arm(forecast_high_c=33.0)
        assert st.armed is False

    def test_arms_when_forecast_close_to_boundary(self):
        """Forecast 34°C (93.2°F) is 0.3°F below 93.5°F boundary.
        Within the 0.9°F default margin → arm."""
        st = self._arm(forecast_high_c=34.0)
        assert st.armed is True, f"Should arm; got reason={st.reason!r}"
        assert st.target_bin_lo == 94
        assert abs(st.boundary_value_settlement - 93.5) < 0.01

    def test_does_not_arm_when_forecast_above_boundary(self):
        """If forecast is already above boundary, the latency-arb
        thesis doesn't apply — the market should already have priced
        it.  Don't arm."""
        st = self._arm(forecast_high_c=35.0)   # 95°F, above 93.5 boundary
        assert st.armed is False
        assert "above_boundary" in st.reason

    def test_does_not_arm_when_market_p_already_high(self):
        """If next-bin market_p is already above
        BOUNDARY_ARM_MAX_MARKET_PRICE, the market has already priced
        the move — no latency edge left."""
        st = self._arm(forecast_high_c=34.0,
                          market_p=BOUNDARY_ARM_MAX_MARKET_PRICE + 0.01)
        assert st.armed is False
        assert "market_p" in st.reason

    def test_does_not_arm_late_in_day(self):
        """If less than BOUNDARY_ARM_MIN_HEATING_MIN minutes to peak,
        too late to bother."""
        st = self._arm(forecast_high_c=34.0,
                          forecast_peak_hour=12, current_local_hour=12)
        assert st.armed is False
        assert "too_late" in st.reason

    def test_disarms_long_past_peak_no_crossing(self):
        """Way past peak, observed_max never approached boundary →
        disarm.  Two disarm branches can apply here (too_late_in_day
        and past_peak); whichever short-circuits first wins. Both are
        correct outcomes — what matters is armed is False."""
        st = self._arm(
            forecast_high_c=34.0,
            forecast_peak_hour=14, current_local_hour=20,
            observed_max_c=28.0,   # ~82°F, far below 93.5°F boundary
        )
        assert st.armed is False


# ============================================================
# Hard-poll window
# ============================================================

def test_hard_poll_window_active_at_55():
    from datetime import datetime, timezone
    t = datetime(2026, 6, 15, 17, 55, 0, tzinfo=timezone.utc)
    assert in_hard_poll_window(t) is True


def test_hard_poll_window_inactive_at_30():
    from datetime import datetime, timezone
    t = datetime(2026, 6, 15, 17, 30, 0, tzinfo=timezone.utc)
    assert in_hard_poll_window(t) is False


def test_hard_poll_window_active_at_boundary():
    from datetime import datetime, timezone
    t52 = datetime(2026, 6, 15, 17, 52, 0, tzinfo=timezone.utc)
    t59 = datetime(2026, 6, 15, 17, 59, 0, tzinfo=timezone.utc)
    assert in_hard_poll_window(t52) is True
    assert in_hard_poll_window(t59) is True


# ============================================================
# Phase 3 / 4 / 5 — _execute_boundary_fire
#
# These tests build a fake in-memory boundary_trigger_log + mock
# execute_signal to validate dry-run, daily caps, and the retry loop
# without touching the live CLOB.
# ============================================================

import sqlite3 as _sqlite3
import boundary_watcher as bw


def _make_conn_with_log():
    """Fresh in-memory DB with boundary_trigger_log so the daily-cap
    helpers don't blow up on OperationalError fallback."""
    conn = _sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE boundary_trigger_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at_utc TEXT NOT NULL,
            would_fire_size_usd REAL,
            actually_fired INTEGER
        )
    """)
    return conn


def _candidate():
    return {
        "contract_id":    "0xCONTRACT",
        "yes_token_id":   "tok-yes",
        "bin_label":      "94-95F",
        "bin_range_low":  94.0,
        "bin_range_high": 95.0,
        "unit":           "fahrenheit",
        "market_prob":    0.05,
        "liquidity_usd":  500.0,
        "boundary":       93.5,
    }


class TestExecuteBoundaryFire:
    """The function being tested ships ONE hard kill switch
    (BOUNDARY_DRY_RUN) plus two soft caps (daily trade count, daily
    budget).  Each must work in isolation."""

    def test_dry_run_returns_zero_and_skips_execute(self, monkeypatch):
        """Even with caps clear and a happy mock execute_signal, the
        dry-run gate MUST return actually_fired=0 with no order placed.
        This is the operator's load-bearing safety switch."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", True)
        call_count = {"n": 0}

        def fake_execute_signal(*a, **k):
            call_count["n"] += 1
            return {"status": "placed", "order_id": "x",
                     "entry_price": 0.05, "shares": 100}

        # We expect execute_signal NOT to be imported at all under dry-run.
        # Still patch it as a safety net so a leak would show up loud.
        monkeypatch.setattr("execution.execute_signal", fake_execute_signal,
                             raising=False)

        conn = _make_conn_with_log()
        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=_candidate(), signal_origin="boundary_confirmed",
        )
        assert result["actually_fired"] == 0
        assert result["exec_notes"] == "dry_run_safety_gate"
        assert call_count["n"] == 0, "execute_signal must not be called in dry-run"

    def test_daily_trade_cap_blocks(self, monkeypatch):
        """When today's actually_fired count reaches the cap, the
        function returns with actually_fired=0 and a daily_trade_cap
        note BEFORE calling execute_signal."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", False)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_TRADES_PER_DAY", 2)

        conn = _make_conn_with_log()
        # Seed today with 2 already-fired rows to hit the cap.
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for _ in range(2):
            conn.execute(
                "INSERT INTO boundary_trigger_log (evaluated_at_utc, "
                "would_fire_size_usd, actually_fired) VALUES (?, ?, ?)",
                (today + "T12:00:00+00:00", 20.0, 1),
            )
        conn.commit()

        called = {"n": 0}
        monkeypatch.setattr(
            "execution.execute_signal",
            lambda *a, **k: (called.__setitem__("n", called["n"]+1) or
                              {"status": "placed", "shares": 0, "entry_price": 0.05}),
            raising=False,
        )

        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=_candidate(), signal_origin="boundary_confirmed",
        )
        assert result["actually_fired"] == 0
        assert "daily_trade_cap" in result["exec_notes"]
        assert called["n"] == 0

    def test_daily_budget_cap_blocks(self, monkeypatch):
        """Same as trade cap, but on accumulated $ spent."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", False)
        monkeypatch.setattr(bw, "BOUNDARY_DAILY_BUDGET_USD", 50.0)
        monkeypatch.setattr(bw, "BOUNDARY_TARGET_STAKE_USD", 20.0)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_TRADES_PER_DAY", 100)

        conn = _make_conn_with_log()
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # $40 already spent — next $20 would exceed $50 budget.
        conn.execute(
            "INSERT INTO boundary_trigger_log (evaluated_at_utc, "
            "would_fire_size_usd, actually_fired) VALUES (?, ?, ?)",
            (today + "T12:00:00+00:00", 40.0, 1),
        )
        conn.commit()

        called = {"n": 0}
        monkeypatch.setattr(
            "execution.execute_signal",
            lambda *a, **k: (called.__setitem__("n", called["n"]+1) or
                              {"status": "placed", "shares": 0, "entry_price": 0.05}),
            raising=False,
        )

        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=_candidate(), signal_origin="boundary_confirmed",
        )
        assert result["actually_fired"] == 0
        assert "daily_budget_cap" in result["exec_notes"]
        assert called["n"] == 0

    def test_live_path_calls_execute_with_signal_origin(self, monkeypatch):
        """When dry-run is OFF and caps are clear, execute_signal is
        called with strategy='boundary_watcher' and the supplied
        signal_origin propagated.  This is what attributes the
        position to boundary in downstream analytics."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", False)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_ON_PARTIAL", False)
        monkeypatch.setattr(bw, "BOUNDARY_TARGET_STAKE_USD", 20.0)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_TRADES_PER_DAY", 10)
        monkeypatch.setattr(bw, "BOUNDARY_DAILY_BUDGET_USD", 100.0)

        captured = {}

        def fake_execute_signal(signal, client=None):
            captured["signal"] = signal
            return {"status": "placed", "order_id": "ord-1",
                     "entry_price": 0.06, "shares": 333.33,
                     "position_id": 42, "fill_status": "pending"}

        monkeypatch.setattr("execution.execute_signal", fake_execute_signal,
                             raising=False)
        monkeypatch.setattr("execution.get_clob_client",
                             lambda: object(), raising=False)

        conn = _make_conn_with_log()
        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=_candidate(), signal_origin="boundary_confirmed",
        )
        assert result["actually_fired"] == 1
        assert result["actual_order_id"] == "ord-1"
        sig = captured["signal"]
        assert sig["strategy"] == "boundary_watcher"
        assert sig["signal_origin"] == "boundary_confirmed"
        assert sig["recommended_side"] == "YES"
        assert sig["contract_id"] == "0xCONTRACT"
        assert sig["max_price_cap"] == bw.BOUNDARY_MAX_PRICE_CAP

    def test_retry_walks_price_up(self, monkeypatch):
        """With retry-on-partial ON and execute_signal returning small
        partial fills, the function should walk market_p up by
        WALK_CENTS each iteration, stopping at RETRY_MAX_PRICE."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", False)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_ON_PARTIAL", True)
        monkeypatch.setattr(bw, "BOUNDARY_TARGET_STAKE_USD", 20.0)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_TRADES_PER_DAY", 100)
        monkeypatch.setattr(bw, "BOUNDARY_DAILY_BUDGET_USD", 1000.0)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_DELAY_SEC", 0)  # speed up
        monkeypatch.setattr(bw, "BOUNDARY_WALK_CENTS", 10)       # 10c steps
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_MAX_PRICE", 0.30)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_RETRIES", 10)

        prices_seen = []

        def fake_execute_signal(signal, client=None):
            # Each call fills only $1 worth of shares (forcing retry).
            prices_seen.append(signal["market_p"])
            return {"status": "placed", "order_id": f"ord-{len(prices_seen)}",
                     "entry_price": signal["market_p"], "shares": 1.0}

        monkeypatch.setattr("execution.execute_signal", fake_execute_signal,
                             raising=False)
        monkeypatch.setattr("execution.get_clob_client",
                             lambda: object(), raising=False)

        conn = _make_conn_with_log()
        cand = _candidate()
        cand["market_prob"] = 0.05  # start cheap

        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=cand, signal_origin="boundary_confirmed",
        )

        # We expect monotonically increasing prices, capped at 0.30
        assert prices_seen[0] == 0.05
        # Each step is +0.10
        for i in range(1, len(prices_seen)):
            assert prices_seen[i] > prices_seen[i-1]
            assert prices_seen[i] - prices_seen[i-1] == pytest.approx(0.10, abs=1e-9)
        # Last price must not exceed cap (else loop should have stopped)
        assert max(prices_seen) <= 0.30 + 1e-9
        # And we should have made at least 2 attempts (proves walk happened)
        assert len(prices_seen) >= 2
        assert result["actually_fired"] == 1
        assert result["retries_used"] >= 1

    def test_retry_stops_when_filled_to_target(self, monkeypatch):
        """If the first call fills enough shares to cover the target
        stake, no retry should happen — even with RETRY_ON_PARTIAL on."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", False)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_ON_PARTIAL", True)
        monkeypatch.setattr(bw, "BOUNDARY_TARGET_STAKE_USD", 20.0)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_TRADES_PER_DAY", 100)
        monkeypatch.setattr(bw, "BOUNDARY_DAILY_BUDGET_USD", 1000.0)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_DELAY_SEC", 0)

        call_count = {"n": 0}

        def fake_execute_signal(signal, client=None):
            call_count["n"] += 1
            # Fully fill: shares * price = $20
            return {"status": "placed", "order_id": "ord-1",
                     "entry_price": 0.05,
                     "shares": 400.0}  # 400 * 0.05 = $20

        monkeypatch.setattr("execution.execute_signal", fake_execute_signal,
                             raising=False)
        monkeypatch.setattr("execution.get_clob_client",
                             lambda: object(), raising=False)

        conn = _make_conn_with_log()
        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=_candidate(), signal_origin="boundary_confirmed",
        )

        assert call_count["n"] == 1, "should stop after first fill covers target"
        assert result["retries_used"] == 0
        assert result["actually_fired"] == 1
        assert "filled_to_target" in result["exec_notes"]

    def test_retry_stops_on_hard_failure(self, monkeypatch):
        """A status='error' or 'failed' from execute_signal should NOT
        be retried — it's not a partial fill, it's a real problem."""
        monkeypatch.setattr(bw, "BOUNDARY_DRY_RUN", False)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_ON_PARTIAL", True)
        monkeypatch.setattr(bw, "BOUNDARY_TARGET_STAKE_USD", 20.0)
        monkeypatch.setattr(bw, "BOUNDARY_MAX_TRADES_PER_DAY", 100)
        monkeypatch.setattr(bw, "BOUNDARY_DAILY_BUDGET_USD", 1000.0)
        monkeypatch.setattr(bw, "BOUNDARY_RETRY_DELAY_SEC", 0)

        call_count = {"n": 0}
        monkeypatch.setattr(
            "execution.execute_signal",
            lambda *a, **k: (call_count.__setitem__("n", call_count["n"]+1) or
                              {"status": "error", "reason": "boom"}),
            raising=False,
        )
        monkeypatch.setattr("execution.get_clob_client",
                             lambda: object(), raising=False)

        conn = _make_conn_with_log()
        result = bw._execute_boundary_fire(
            conn=conn, city="Miami", event_date="2026-06-16",
            candidate=_candidate(), signal_origin="boundary_confirmed",
        )
        assert call_count["n"] == 1, "should not retry on error"
        assert result["actually_fired"] == 0


# ============================================================
# Integration: signal_origin must flow through to insert_position
# ============================================================

def test_insert_position_accepts_signal_origin():
    """Regression test for the db.py signature change.  If a future
    refactor drops signal_origin from insert_position, the boundary
    watcher loses its attribution and analytics will miscount."""
    import inspect
    import db
    sig = inspect.signature(db.insert_position)
    assert "signal_origin" in sig.parameters, (
        "insert_position must accept signal_origin kwarg — boundary "
        "watcher depends on this for analytics attribution."
    )


# ============================================================
# Market-consensus override (2026-06-16): SF case where forecast
# is 80°F but market consensus is 70-71°F at 96%.  Forecast-anchored
# logic would pick 82-83 (unreachable today); override should pick
# 72-73 (just above consensus).
# ============================================================

def _make_signals_conn():
    """In-memory DB with paper_predictor_signals seeded so
    find_candidate_bin can run against it."""
    conn = _sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE paper_predictor_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at_utc  TEXT NOT NULL,
            city            TEXT NOT NULL,
            event_date      TEXT,
            contract_id     TEXT,
            yes_token_id    TEXT,
            bin_label       TEXT,
            bin_range_low   REAL,
            bin_range_high  REAL,
            unit            TEXT,
            market_prob     REAL,
            liquidity_usd   REAL
        )
    """)
    return conn


def _seed_sf_bins(conn):
    """Seed San Francisco 2026-06-16 with the bin set from the real
    Polymarket market: 65 -> 84+ in 2°F increments, with 70-71 at 96%."""
    bins = [
        # (label, lo, hi, market_prob)
        ("65F-or-below", 65, 65, 0.005),   # 1°F bin — gets filtered by is_supported_bin
        ("66-67F", 66, 67, 0.005),
        ("68-69F", 68, 69, 0.005),
        ("70-71F", 70, 71, 0.96),          # the consensus bin
        ("72-73F", 72, 73, 0.02),          # the candidate in override mode
        ("74-75F", 74, 75, 0.05),
        ("76-77F", 76, 77, 0.01),
        ("78-79F", 78, 79, 0.005),
        ("80-81F", 80, 81, 0.005),         # forecast bin (forecast=80°F)
        ("82-83F", 82, 83, 0.005),         # forecast-anchored candidate
        ("84F-or-above", 84, 84, 0.005),   # 1°F bin — filtered
    ]
    for label, lo, hi, mp in bins:
        conn.execute(
            "INSERT INTO paper_predictor_signals "
            "(scanned_at_utc, city, event_date, contract_id, yes_token_id, "
            " bin_label, bin_range_low, bin_range_high, unit, "
            " market_prob, liquidity_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2026-06-16T12:00:00+00:00", "San Francisco", "2026-06-16",
             f"0xCONT{lo}", f"tok-{lo}", label, lo, hi, "fahrenheit",
             mp, 500.0),
        )
    conn.commit()


class TestConsensusOverride:
    """SF case: forecast=80°F (≈26.7°C); consensus=70-71°F at 96%."""

    SF_FORECAST_C = (80 - 32) * 5/9   # ≈ 26.667

    def test_override_disabled_default_picks_forecast_bin(self, monkeypatch):
        """With BOUNDARY_CONSENSUS_OVERRIDE_ENABLED=0 (default), the
        SF case picks 82-83°F (the bin just above the 80°F forecast).
        This is the pre-override behavior — must NOT change for callers
        who don't opt in."""
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_ENABLED", False)
        conn = _make_signals_conn()
        _seed_sf_bins(conn)
        from boundary_watcher import find_candidate_bin
        cand = find_candidate_bin(conn, "San Francisco", "2026-06-16",
                                       self.SF_FORECAST_C)
        assert cand is not None
        assert cand["bin_label"] == "82-83F"
        assert cand["anchor_mode"] == "forecast"

    def test_override_enabled_picks_above_consensus(self, monkeypatch):
        """With override ON, the SF case picks 72-73°F (just above
        the 70-71 consensus bin) — NOT the forecast-anchored 82-83."""
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_ENABLED", True)
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_THRESHOLD", 0.70)
        conn = _make_signals_conn()
        _seed_sf_bins(conn)
        from boundary_watcher import find_candidate_bin
        cand = find_candidate_bin(conn, "San Francisco", "2026-06-16",
                                       self.SF_FORECAST_C)
        assert cand is not None
        assert cand["bin_label"] == "72-73F"
        assert cand["anchor_mode"] == "market_consensus"
        # Override metadata surfaced so the dashboard can show it
        assert cand["consensus_bin_label"] == "70-71F"
        assert abs(cand["consensus_market_prob"] - 0.96) < 1e-6

    def test_override_skipped_when_max_market_p_below_threshold(self, monkeypatch):
        """If no bin reaches the consensus threshold (e.g., dispersed
        market), override does NOT trigger — falls through to forecast."""
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_ENABLED", True)
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_THRESHOLD", 0.99)
        # Above threshold is 0.99 but max bin is 0.96 → no override
        conn = _make_signals_conn()
        _seed_sf_bins(conn)
        from boundary_watcher import find_candidate_bin
        cand = find_candidate_bin(conn, "San Francisco", "2026-06-16",
                                       self.SF_FORECAST_C)
        assert cand is not None
        assert cand["anchor_mode"] == "forecast"
        assert cand["bin_label"] == "82-83F"

    def test_override_skipped_when_consensus_agrees_with_forecast(self, monkeypatch):
        """If consensus bin is the forecast bin (or higher), override
        does NOT apply — the original forecast logic already does the
        right thing in that case.  Forecast 70°F + consensus 70-71°F
        → no override (they agree)."""
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_ENABLED", True)
        monkeypatch.setattr(bw, "BOUNDARY_CONSENSUS_OVERRIDE_THRESHOLD", 0.70)
        conn = _make_signals_conn()
        _seed_sf_bins(conn)
        forecast_70c = (70 - 32) * 5/9   # ≈ 21.1°C
        from boundary_watcher import find_candidate_bin
        cand = find_candidate_bin(conn, "San Francisco", "2026-06-16",
                                       forecast_70c)
        assert cand is not None
        # Consensus upper edge is 71.5°F; forecast is 70°F → consensus
        # upper is ABOVE forecast → "agrees", no override → forecast
        # picks the bin just above 70°F boundary, which is 72-73.
        assert cand["anchor_mode"] == "forecast"

    def test_arming_consensus_mode_uses_observed_max_gate(self, monkeypatch):
        """In market_consensus mode, the arming gate is observed_max
        within OBS_MARGIN of the boundary — NOT the forecast-margin
        gate.  An SF-style event with forecast 80°F (way above the new
        72-73 boundary) but observed_max 70°F should ARM."""
        monkeypatch.setattr(bw, "BOUNDARY_ARM_OBS_MARGIN_F", 3.0)
        monkeypatch.setattr(bw, "BOUNDARY_ARM_MAX_MARKET_PRICE", 0.05)
        from boundary_watcher import compute_arming_state
        st = compute_arming_state(
            forecast_high_c     = 26.67,         # 80°F — way above boundary
            forecast_peak_hour  = 15,
            current_local_hour  = 12,
            settlement_unit     = "fahrenheit",
            candidate_bin_lo    = 72.0,
            candidate_bin_hi    = 73.0,
            candidate_market_p  = 0.02,
            observed_max_c      = (70 - 32) * 5/9,   # 70°F, 1.5°F below 71.5 boundary
            anchor_mode         = "market_consensus",
        )
        assert st.armed is True, f"should arm; got reason={st.reason!r}"

    def test_arming_consensus_mode_rejects_observed_too_far(self, monkeypatch):
        """Same setup but observed_max 60°F — too far below the
        71.5°F boundary (margin 11.5°F > 3°F threshold).  Reject."""
        monkeypatch.setattr(bw, "BOUNDARY_ARM_OBS_MARGIN_F", 3.0)
        monkeypatch.setattr(bw, "BOUNDARY_ARM_MAX_MARKET_PRICE", 0.05)
        from boundary_watcher import compute_arming_state
        st = compute_arming_state(
            forecast_high_c     = 26.67,
            forecast_peak_hour  = 15,
            current_local_hour  = 12,
            settlement_unit     = "fahrenheit",
            candidate_bin_lo    = 72.0,
            candidate_bin_hi    = 73.0,
            candidate_market_p  = 0.02,
            observed_max_c      = (60 - 32) * 5/9,   # 60°F
            anchor_mode         = "market_consensus",
        )
        assert st.armed is False
        assert "observed_too_far_below_boundary" in st.reason

    def test_arming_consensus_mode_rejects_observed_at_boundary(self, monkeypatch):
        """If observed_max is already AT or ABOVE the boundary, the
        market already saw it — no latency edge left.  Reject."""
        monkeypatch.setattr(bw, "BOUNDARY_ARM_OBS_MARGIN_F", 3.0)
        from boundary_watcher import compute_arming_state
        st = compute_arming_state(
            forecast_high_c     = 26.67,
            forecast_peak_hour  = 15,
            current_local_hour  = 12,
            settlement_unit     = "fahrenheit",
            candidate_bin_lo    = 72.0,
            candidate_bin_hi    = 73.0,
            candidate_market_p  = 0.02,
            observed_max_c      = (72 - 32) * 5/9,   # 72°F, already past 71.5
            anchor_mode         = "market_consensus",
        )
        assert st.armed is False
        assert "observed_already_at_or_above_boundary" in st.reason

    def test_arming_default_mode_unchanged(self, monkeypatch):
        """Regression: arming with anchor_mode='forecast' (default)
        MUST behave identically to pre-override code.  No accidental
        cross-contamination from the new branch."""
        from boundary_watcher import compute_arming_state
        # Forecast 93.2°F = 34°C, boundary 93.5°F (94-95 bin) → margin
        # 0.3°F, within 0.9°F threshold → should arm.
        st = compute_arming_state(
            forecast_high_c     = 34.0,
            forecast_peak_hour  = 15,
            current_local_hour  = 12,
            settlement_unit     = "fahrenheit",
            candidate_bin_lo    = 94.0,
            candidate_bin_hi    = 95.0,
            candidate_market_p  = 0.02,
            observed_max_c      = 30.0,
            # anchor_mode omitted → defaults to "forecast"
        )
        assert st.armed is True


def test_execute_signal_threads_signal_origin():
    """The execute_signal function must read signal.get('signal_origin')
    into position_kwargs.  Without this the position row gets a NULL
    signal_origin even though the boundary watcher passed one."""
    import execution
    src = open(execution.__file__, encoding="utf-8").read()
    assert 'signal.get("signal_origin")' in src or \
           "signal.get('signal_origin')" in src, (
        "execute_signal must thread signal['signal_origin'] into "
        "position_kwargs — see boundary_watcher integration."
    )