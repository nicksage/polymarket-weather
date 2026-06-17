"""
test_cold_bias_day1.py — Day 1 of the cold-bias work (2026-06-17):

  * Stage 4a — bin-lock σ now respects PREDICTOR_SIGMA_FLOOR_C
  * Stage 3  — loss-stopper gate fires correctly
  * Stage 3  — removal-trigger checker has the right semantics
                 (1) per-city bias condition
                 (2) high-disagreement bucket condition
                 (3) gate-already-disabled short-circuit
                 (4) the gate-doing-its-job edge case
                     (no bucket trades — should ONLY clear when
                     bias has also cleared, not on its own)

Run:
    cd bot
    python -m pytest tests/test_cold_bias_day1.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)


# ============================================================
# Stage 4a — bin-lock σ floor
# ============================================================

class TestBinLockSigmaFloor:
    """The bin-lock branch in scripts/intraday_predictor.py used to set
    σ to (bin_width/6), bypassing PREDICTOR_SIGMA_FLOOR_C entirely.  The
    fix raises σ to max(floor, bin_width/6).  This test reads the source
    so we can lock the fix in place without standing up the full
    predict_bins pipeline (which needs forecasts/observations/HRRR/etc)."""

    def test_bin_lock_sigma_uses_floor_constant(self):
        from pathlib import Path
        src = (Path(_BOT_DIR) / "scripts" / "intraday_predictor.py").read_text(
            encoding="utf-8")
        # The fix must reference the floor constant inside the bin-lock
        # body.  If a future refactor reverts to the literal 0.15, this
        # test fails loudly.
        assert "max(PREDICTOR_SIGMA_FLOOR_C" in src, (
            "bin-lock σ must use PREDICTOR_SIGMA_FLOOR_C, not a literal — "
            "otherwise the dashboard's ±0.30°C high-confidence bets come back."
        )
        # And the old broken literal must be gone from THIS branch.
        # (The literal 0.15 may still appear elsewhere; what matters is
        # that the bin-lock line uses the constant.)
        bin_lock_section = src.split("BIN-LOCK:")[1][:600] \
            if "BIN-LOCK:" in src else ""
        assert "max(0.15" not in bin_lock_section, (
            "bin-lock branch still has the bypassing 0.15 literal."
        )


# ============================================================
# Stage 3 — loss-stopper gate
# ============================================================

class TestLossStopperGate:
    """The gate sits inside evaluate_gates and rejects bins where
    (our_p >= 0.35 AND market_p <= 0.15).  We invoke evaluate_gates
    directly with tuned-up monkeypatch defaults so we don't depend on
    .env state on the test runner."""

    def _gate(self, *, our_p, market_p, edge=0.10, liquidity=2000,
                already_acted=False, trades_today=0, deployed_today=0,
                current_hour=14):
        import scheduled_predictor as sp
        return sp.evaluate_gates(
            current_hour=current_hour,
            our_p=our_p, market_p=market_p, edge=edge, liquidity=liquidity,
            already_acted=already_acted, trades_today=trades_today,
            deployed_today=deployed_today,
        )

    def test_high_disagreement_vetoed(self, monkeypatch):
        """Houston 6/15 case: model 43%, market 13.2% → veto."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", True)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)
        ok, reason = self._gate(our_p=0.43, market_p=0.132)
        assert ok is False
        assert "loss_stopper_high_disagreement" in reason

    def test_below_model_threshold_not_vetoed(self, monkeypatch):
        """Model_p below 0.35 → gate doesn't fire even with low mkt_p."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", True)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)
        ok, reason = self._gate(our_p=0.30, market_p=0.10)
        # Other gates may still reject; what matters is that loss_stopper
        # ISN'T the reason.
        assert "loss_stopper" not in reason

    def test_above_market_threshold_not_vetoed(self, monkeypatch):
        """Market_p above 0.15 → gate doesn't fire (the bot may have
        a legitimate edge there)."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", True)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)
        ok, reason = self._gate(our_p=0.50, market_p=0.25)
        assert "loss_stopper" not in reason

    def test_gate_disabled_via_env(self, monkeypatch):
        """PREDICTOR_LOSS_STOPPER_ENABLED=0 → gate never fires."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", False)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)
        ok, reason = self._gate(our_p=0.43, market_p=0.132)
        assert "loss_stopper" not in reason


# ============================================================
# Stage 3 — removal-trigger checker
# ============================================================

def _make_checker_conn():
    """Minimal in-memory schema for the checker.  Mirrors the three
    tables it reads — paper_predictor_signals, positions,
    resolution_observations — only the columns the SQL touches."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE paper_predictor_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at_utc TEXT, action TEXT, city TEXT, event_date TEXT,
            contract_id TEXT, our_prob REAL, market_prob REAL, mu_c REAL
        )
    """)
    conn.execute("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT, date TEXT, city TEXT, is_paper INTEGER,
            status TEXT, pnl_net REAL
        )
    """)
    conn.execute("""
        CREATE TABLE resolution_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT, event_date TEXT,
            wunderground_high_c REAL, metar_peak_t_group_c REAL,
            bot_observed_max_c REAL
        )
    """)
    return conn


def _seed_trade(conn, *, idx, city, days_ago, our_p, mkt_p, mu_c,
                  actual_c, pnl):
    """Add one closed trade with matching signal + resolution rows.
    days_ago lets us spread trades across the lookback window."""
    cid = f"0xC{idx}"
    event_date = f"DATE('now', '-{days_ago} days')"
    # We use SQL date() for portability of the test seeds — the checker's
    # WHERE clauses use the same date() function.
    conn.execute(f"""
        INSERT INTO paper_predictor_signals
            (scanned_at_utc, action, city, event_date, contract_id,
             our_prob, market_prob, mu_c)
        VALUES ((SELECT date('now', '-{days_ago} days') || 'T12:00:00+00:00'),
                'LIVE_BUY', ?, (SELECT date('now', '-{days_ago} days')),
                ?, ?, ?, ?)
    """, (city, cid, our_p, mkt_p, mu_c))
    conn.execute(f"""
        INSERT INTO positions
            (contract_id, date, city, is_paper, status, pnl_net)
        VALUES (?, (SELECT date('now', '-{days_ago} days')), ?, 0, 'closed', ?)
    """, (cid, city, pnl))
    conn.execute(f"""
        INSERT INTO resolution_observations
            (city, event_date, wunderground_high_c)
        VALUES (?, (SELECT date('now', '-{days_ago} days')), ?)
    """, (city, actual_c))


class TestRemovalTriggerChecker:

    def test_gate_already_disabled_short_circuits(self, monkeypatch):
        """If the operator has flipped the gate off, the checker says so
        and skips both queries (a fresh DB shouldn't crash anything)."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", False)
        conn = _make_checker_conn()
        result = sp.check_loss_stopper_removal_condition(conn, lookback_days=30)
        assert result["should_remove"] is False
        assert any("gate already disabled" in r for r in result["reasons"])

    def test_bias_in_tolerance_and_no_bucket_trades_clears(self, monkeypatch):
        """The edge case the critic flagged: when the gate has been
        doing its job (no high-disagreement trades), we should ONLY
        recommend removal IF the per-city bias has also cleared.
        Seed: 30 closed trades, actual ≈ model → cond1 passes.  No
        high-disagreement bucket trades exist (the gate prevented them).
        Both conditions clear → recommend removal."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", True)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)

        conn = _make_checker_conn()
        # 30 trades for Houston, bias = 0.5°C (within ±1°C tolerance).
        # Model_p / mkt_p both at 0.30 — outside the high-disagreement
        # bucket, so condition 2 sees n_dis=0.
        for i in range(30):
            _seed_trade(conn, idx=i, city="Houston", days_ago=(i % 25) + 1,
                          our_p=0.30, mkt_p=0.30,
                          mu_c=20.0, actual_c=20.5, pnl=1.0)

        result = sp.check_loss_stopper_removal_condition(
            conn, lookback_days=30, min_n=30)
        assert result["should_remove"] is True, \
            f"expected clear; got reasons={result['reasons']}"
        assert result["disagreement_bucket_n"] == 0

    def test_bias_out_of_tolerance_blocks_clearance(self, monkeypatch):
        """If even one city has bias > tolerance, condition 1 fails and
        the checker should NOT recommend removal, even if the bucket
        condition is fine."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", True)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)

        conn = _make_checker_conn()
        for i in range(30):
            # Big positive bias: model too cold by 3°C
            _seed_trade(conn, idx=i, city="Houston", days_ago=(i % 25) + 1,
                          our_p=0.30, mkt_p=0.30,
                          mu_c=20.0, actual_c=23.0, pnl=1.0)

        result = sp.check_loss_stopper_removal_condition(
            conn, lookback_days=30, min_n=30, bias_tol_c=1.0)
        assert result["should_remove"] is False
        assert any("cond1" in r for r in result["reasons"])

    def test_negative_pnl_bucket_blocks_clearance(self, monkeypatch):
        """If the high-disagreement bucket has net-negative PnL over the
        window, the loss-stopper is still earning its keep — don't
        recommend removal even if the bias has cleared."""
        import scheduled_predictor as sp
        monkeypatch.setattr(sp, "LOSS_STOPPER_ENABLED", True)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MODEL_P_MIN", 0.35)
        monkeypatch.setattr(sp, "LOSS_STOPPER_MKT_P_MAX",   0.15)

        conn = _make_checker_conn()
        # 30 well-centered trades for cond1
        for i in range(30):
            _seed_trade(conn, idx=i, city="Houston", days_ago=(i % 25) + 1,
                          our_p=0.30, mkt_p=0.30,
                          mu_c=20.0, actual_c=20.5, pnl=1.0)
        # 30 high-disagreement trades with net loss
        for i in range(30, 60):
            _seed_trade(conn, idx=i, city="Houston", days_ago=(i % 25) + 1,
                          our_p=0.45, mkt_p=0.10,
                          mu_c=20.0, actual_c=20.5, pnl=-2.0)

        result = sp.check_loss_stopper_removal_condition(
            conn, lookback_days=30, min_n=30)
        assert result["should_remove"] is False
        assert result["disagreement_bucket_n"] >= 30
        assert (result["disagreement_bucket_pnl"] or 0) < 0
        assert any("cond2" in r for r in result["reasons"])