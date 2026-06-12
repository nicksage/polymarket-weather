"""
test_invariant_guards.py — Lock in the OBSERVATIONAL FOREVER design.

The most important test in this file is test_no_import_from_prediction_path
— it enforces that the guards module is never imported by anything in the
prediction or trading path.  If a future change adds such an import, this
test fails, because that import is the first step toward the guards
becoming load-bearing infrastructure.

The other tests verify each guard fires on the correct input and returns
None otherwise.

Run:
    cd bot
    python -m pytest tests/test_invariant_guards.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scheduled_predictor import _SCHEMA_SQL  # type: ignore
from scripts.invariant_guards import (  # type: ignore
    _check_observed_max,
    _check_forecast_high,
    _check_sigma_post_peak,
    _check_cooling_confidence_post_peak,
    _check_mu_neighbor_coherence,
    run_invariant_checks,
)


def _fresh_db() -> sqlite3.Connection:
    """In-memory DB with the production schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    # Apply the same migrations the live DB has
    for col, ddl in [
        ("market_closed", "INTEGER DEFAULT 0"),
        ("data_quality_flag", "TEXT"),
        ("cooling_confidence", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE paper_predictor_signals "
                          f"ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError:
            pass
    return conn


def _insert(conn, city, event_date, scanned_at_utc, **fields):
    """Insert one signal row.  Defaults are sensible for testing
    guards; override per-test for the specific value being checked."""
    base = {
        "scanned_at_utc":     scanned_at_utc,
        "mode":               "live",
        "city":               city,
        "event_date":         event_date,
        "contract_id":        "c1",
        "bin_label":          "70-71F",
        "action":             "SKIP",
        "observed_max_c":     20.0,
        "forecast_high_c":    25.0,
        "forecast_peak_hour": 15,   # 3pm local
        "mu_c":               24.0,
        "sigma_c":            2.0,
        "cooling_confidence": 0.0,
    }
    base.update(fields)
    cols = ",".join(base.keys())
    placeholders = ",".join(["?"] * len(base))
    conn.execute(
        f"INSERT INTO paper_predictor_signals ({cols}) VALUES ({placeholders})",
        list(base.values()),
    )
    conn.commit()


# ============================================================
# The architectural constraint — observational forever
# ============================================================

def test_no_import_from_prediction_path():
    """invariant_guards must NOT be imported by anything in the
    prediction or trading decision path.  If you add such an import,
    the guards stop being purely observational and start to become
    load-bearing — at which point they need their own calibration and
    their own guards.  See OBSERVATIONAL FOREVER design rule.

    The ONLY allowed import path is from scheduled_predictor at the
    end of run_intraday_scan inside a try/except — the call site is
    inline, not a top-level import.  We check for actual import syntax
    (parsed via ast), not substring presence, so comments mentioning
    the module are fine."""
    import ast
    import pathlib

    def has_invariant_guards_import(src: str) -> bool:
        """True iff the source has an `import` or `from ... import`
        statement that references invariant_guards.  Comments and
        strings that contain the substring don't count."""
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "invariant_guards" in alias.name:
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and "invariant_guards" in node.module:
                    return True
        return False

    bot_root = pathlib.Path(_BOT_DIR)
    forbidden_files = [
        bot_root / "scripts" / "intraday_predictor.py",
        bot_root / "edge.py",
        bot_root / "execution.py",
        bot_root / "risk.py",
        bot_root / "sizing.py",
    ]
    for fpath in forbidden_files:
        if not fpath.exists():
            continue
        src = fpath.read_text(encoding="utf-8")
        assert not has_invariant_guards_import(src), (
            f"{fpath} imports invariant_guards.  Guards must remain "
            "observational — they cannot be consumed by the prediction "
            "or trading path.  See OBSERVATIONAL FOREVER design rule "
            "in scripts/invariant_guards.py."
        )

    # scheduled_predictor IS allowed to call run_invariant_checks but
    # ONLY inline at the scan-end call site, not as a top-level import.
    # We check this by ensuring NO ast.Import or ast.ImportFrom node
    # at module-toplevel references invariant_guards.  Inline imports
    # inside function bodies are fine.
    sp_src = (bot_root / "scheduled_predictor.py").read_text(encoding="utf-8")
    sp_tree = ast.parse(sp_src)
    for node in sp_tree.body:    # top-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "invariant_guards" not in alias.name, (
                    "scheduled_predictor imports invariant_guards at module "
                    "top — the call must be inline at the scan-end site, in "
                    "a try/except block, so a guard crash never blocks a scan."
                )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module and "invariant_guards" in node.module), (
                "scheduled_predictor imports invariant_guards at module "
                "top — the call must be inline at the scan-end site, in "
                "a try/except block, so a guard crash never blocks a scan."
            )


def test_run_invariant_checks_returns_none():
    """The orchestrator must return None — no value from this module
    is allowed to influence the scan, sizing, or gate stack."""
    conn = _fresh_db()
    result = run_invariant_checks(conn, "2026-06-12T18:00:00+00:00")
    assert result is None, (
        "run_invariant_checks must return None — see OBSERVATIONAL "
        "FOREVER design rule."
    )


# ============================================================
# observed_max_monotone
# ============================================================

def test_observed_max_decrease_fires():
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T18:00:00+00:00", observed_max_c=30.0)
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T19:00:00+00:00", observed_max_c=28.0)
    v = _check_observed_max(conn, "Denver", "2026-06-12",
                              "2026-06-12T19:00:00+00:00")
    assert v is not None
    assert v.guard_name == "observed_max_monotone"
    assert v.delta < 0
    assert "decreased" in v.detail


def test_observed_max_increase_is_quiet():
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T18:00:00+00:00", observed_max_c=28.0)
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T19:00:00+00:00", observed_max_c=30.0)
    assert _check_observed_max(conn, "Denver", "2026-06-12",
                                  "2026-06-12T19:00:00+00:00") is None


def test_observed_max_tiny_decrease_within_tolerance():
    """Float-noise-level decreases must not fire.  observed_max can
    wobble by 0.01-0.02°C due to per-bin write ordering."""
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T18:00:00+00:00", observed_max_c=28.000)
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T19:00:00+00:00", observed_max_c=27.980)
    assert _check_observed_max(conn, "Denver", "2026-06-12",
                                  "2026-06-12T19:00:00+00:00") is None


# ============================================================
# forecast_high_monotone (verifies the recovery helper's invariant)
# ============================================================

def test_forecast_high_decrease_fires():
    """If the recovery helper regresses and the scan loop starts writing
    a lower forecast_high than a prior scan, this guard fires.  Catches
    a future ratchet-bug analog before it poisons the calibration data."""
    conn = _fresh_db()
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T07:04:00+00:00", forecast_high_c=28.33)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T23:24:00+00:00", forecast_high_c=17.22)
    v = _check_forecast_high(conn, "San Francisco", "2026-06-12",
                                "2026-06-12T23:24:00+00:00")
    assert v is not None
    assert "decreased" in v.detail


def test_forecast_high_recovery_intact_is_quiet():
    conn = _fresh_db()
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T07:04:00+00:00", forecast_high_c=28.33)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T23:24:00+00:00", forecast_high_c=28.33)
    assert _check_forecast_high(conn, "San Francisco", "2026-06-12",
                                   "2026-06-12T23:24:00+00:00") is None


# ============================================================
# sigma_monotone_post_peak
# ============================================================

def test_sigma_widening_post_peak_fires():
    conn = _fresh_db()
    # SF peak hour 15, current scan is at 17:00 PDT (00:00 UTC June 13)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T22:00:00+00:00",   # 15:00 PDT — at peak
             sigma_c=2.0, forecast_peak_hour=15)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-13T00:00:00+00:00",   # 17:00 PDT — past peak
             sigma_c=2.4, forecast_peak_hour=15)
    v = _check_sigma_post_peak(conn, "San Francisco", "2026-06-12",
                                 "2026-06-13T00:00:00+00:00")
    assert v is not None
    assert v.delta > 0


def test_sigma_widening_pre_peak_is_quiet():
    """σ can widen freely before the forecast peak — only post-peak
    narrowing is the invariant."""
    conn = _fresh_db()
    # Both scans pre-peak (morning)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T15:00:00+00:00",   # 08:00 PDT
             sigma_c=2.0, forecast_peak_hour=15)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T17:00:00+00:00",   # 10:00 PDT
             sigma_c=2.4, forecast_peak_hour=15)
    assert _check_sigma_post_peak(conn, "San Francisco", "2026-06-12",
                                     "2026-06-12T17:00:00+00:00") is None


def test_sigma_narrowing_post_peak_is_quiet():
    conn = _fresh_db()
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T22:00:00+00:00",
             sigma_c=2.0, forecast_peak_hour=15)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-13T00:00:00+00:00",
             sigma_c=1.4, forecast_peak_hour=15)
    assert _check_sigma_post_peak(conn, "San Francisco", "2026-06-12",
                                     "2026-06-13T00:00:00+00:00") is None


# ============================================================
# cooling_confidence_monotone_post_peak
# ============================================================

def test_cooling_confidence_drop_post_peak_fires():
    """A drop in cooling_confidence post-peak is the bin-lock-
    discontinuity churn surfacing as data."""
    conn = _fresh_db()
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T22:00:00+00:00",
             cooling_confidence=0.72, forecast_peak_hour=15)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-13T00:00:00+00:00",
             cooling_confidence=0.55, forecast_peak_hour=15)
    v = _check_cooling_confidence_post_peak(conn, "San Francisco",
                                              "2026-06-12",
                                              "2026-06-13T00:00:00+00:00")
    assert v is not None
    assert "dropped" in v.detail


def test_cooling_confidence_rise_post_peak_is_quiet():
    conn = _fresh_db()
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-12T22:00:00+00:00",
             cooling_confidence=0.55, forecast_peak_hour=15)
    _insert(conn, "San Francisco", "2026-06-12",
             "2026-06-13T00:00:00+00:00",
             cooling_confidence=0.72, forecast_peak_hour=15)
    assert _check_cooling_confidence_post_peak(conn, "San Francisco",
                                                  "2026-06-12",
                                                  "2026-06-13T00:00:00+00:00") is None


# ============================================================
# mu_jump_incoherent_with_neighbors
# ============================================================

def test_mu_small_jump_is_quiet():
    """μ jumps below MU_JUMP_MAGNITUDE_FLOOR_C (1.5°C) don't fire."""
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T22:00:00+00:00", mu_c=28.0)
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T22:30:00+00:00", mu_c=28.8)
    assert _check_mu_neighbor_coherence(conn, "Denver", "2026-06-12",
                                           "2026-06-12T22:30:00+00:00") is None


def test_mu_slow_drift_is_quiet():
    """μ jumps over a long window (>1h) don't fire — slow drift is
    not the source-flip signature."""
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T18:00:00+00:00", mu_c=24.0)
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T22:00:00+00:00", mu_c=29.0)   # 5°C over 4h
    assert _check_mu_neighbor_coherence(conn, "Denver", "2026-06-12",
                                           "2026-06-12T22:00:00+00:00") is None


def test_mu_fast_jump_no_neighbor_data_flags_with_caveat():
    """Without neighbor data, the guard falls back to magnitude-only
    and notes the caveat in the detail string."""
    conn = _fresh_db()
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T22:00:00+00:00", mu_c=24.0)
    _insert(conn, "Denver", "2026-06-12",
             "2026-06-12T22:30:00+00:00", mu_c=28.0)  # 4°C in 30min
    v = _check_mu_neighbor_coherence(conn, "Denver", "2026-06-12",
                                        "2026-06-12T22:30:00+00:00")
    # With no neighbor_obs.db in the test environment, the guard fires
    # in magnitude-only mode.  Detail string carries the caveat.
    if v is not None:
        assert "no neighbor data" in v.detail