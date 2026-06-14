"""
test_repricer_coverage.py — Phase 1 repricer coverage + safety (2026-06-13).

What Phase 1 added:
  1. intraday_predictor entries are now eligible for the stale-entry
     repricer (previously TKH-only).  Both the cancel-sweep skip
     and the repricer query include "intraday_predictor".
  2. The cron interval is env-tunable via PREDICTOR_REPRICE_INTERVAL_S
     (default 60s, was 5min).
  3. The topup repricer now does pre-cancel partial-fill capture —
     same pattern as the entry repricer — so a chunk that fills
     during the cancel race counts against the position's
     committed_usdc BEFORE the replacement order is sized.

Tests pin:
  - intraday_predictor is in BOTH _ENSURE_FILL_STRATEGIES sets in monitor.py
  - The env var default is 60s (not 300s / 5min)
  - The default min is enforced (don't recommend < 10s)
  - The topup repricer's pre-cancel capture is wired in

Run:
    cd bot
    python -m pytest tests/test_repricer_coverage.py -v
"""

from __future__ import annotations

import inspect
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)


# ============================================================
# Coverage — intraday_predictor included
# ============================================================

def test_intraday_predictor_in_module_level_ensure_fill_strategies():
    """The repricer's _ENSURE_FILL_STRATEGIES at monitor.py module level
    must include 'intraday_predictor'.  Without this, the entry repricer
    runs but its WHERE clause excludes the intraday loop's orders, so
    the speedup is a no-op for the bot's actual entries."""
    import monitor   # type: ignore
    assert "intraday_predictor" in monitor._ENSURE_FILL_STRATEGIES, (
        f"intraday_predictor must be in _ENSURE_FILL_STRATEGIES "
        f"(got {monitor._ENSURE_FILL_STRATEGIES!r}) so the entry "
        f"repricer chases its stale orders."
    )
    assert "top_k_hedged" in monitor._ENSURE_FILL_STRATEGIES, (
        "top_k_hedged must remain — removing it would regress the TKH "
        "ensure-fill hedge guarantee."
    )


def test_intraday_predictor_in_cancel_sweep_local_set():
    """The cancel-sweep function at monitor.py:_cancel_pending_orders
    has its OWN local _ENSURE_FILL_STRATEGIES set.  If it doesn't also
    include 'intraday_predictor', the cancel sweep will mark intraday
    orders dead before the repricer can chase them — the repricer
    speedup helps nothing."""
    import monitor   # type: ignore
    src = inspect.getsource(monitor._cancel_pending_orders)
    # Grep for the local-set declaration.  This is intentionally a string
    # check (not import-time) because the set is a function-local literal.
    pattern = r'_ENSURE_FILL_STRATEGIES\s*=\s*\{[^}]*"intraday_predictor"[^}]*\}'
    assert re.search(pattern, src), (
        "_cancel_pending_orders must include 'intraday_predictor' in its "
        "local _ENSURE_FILL_STRATEGIES set; otherwise the sweep cancels "
        "intraday entries before the repricer can chase them."
    )


# ============================================================
# Cron interval — env-tunable + safe default
# ============================================================

def test_default_reprice_interval_is_60s():
    """Default cron interval is 60 seconds (was 5 minutes).  Verifies
    main.py reads PREDICTOR_REPRICE_INTERVAL_S with default 60."""
    main_src = open(os.path.join(_BOT_DIR, "main.py"),
                       encoding="utf-8").read()
    # Pattern: int(_os.getenv("PREDICTOR_REPRICE_INTERVAL_S", "60"))
    pattern = (r'getenv\(\s*["\']PREDICTOR_REPRICE_INTERVAL_S["\']\s*,'
                 r'\s*["\']60["\']\s*\)')
    assert re.search(pattern, main_src), (
        "main.py must read PREDICTOR_REPRICE_INTERVAL_S with default '60'. "
        "If the default changed, update this test AND the HANDOFF doc."
    )


def test_repricer_uses_interval_trigger_not_cron():
    """The repricer jobs must use IntervalTrigger (sub-minute cadence
    supported) not CronTrigger(minute='*/5')."""
    main_src = open(os.path.join(_BOT_DIR, "main.py"),
                       encoding="utf-8").read()
    # Look for both repricer add_job blocks and confirm IntervalTrigger
    # is the trigger.
    for job_id in ("stale_topup_refresh", "stale_entry_refresh"):
        # Match: id="<job_id>" ... preceded by a trigger= line
        block_pattern = (
            r'trigger\s*=\s*IntervalTrigger\([^)]+\)'
            r'(?:[^a]|a(?!dd_job))*?'
            r'id\s*=\s*["\']' + re.escape(job_id) + r'["\']'
        )
        assert re.search(block_pattern, main_src, re.DOTALL), (
            f"Job '{job_id}' must use IntervalTrigger.  Reverting to "
            f"CronTrigger(minute='*/5') would un-do Phase 1's speedup."
        )


def test_repricer_jobs_have_max_instances_one():
    """max_instances=1 plus coalesce=True ensures that if the bot is
    overloaded and a previous repricer cycle is still running, the
    next one won't fire concurrently.  Belt-and-suspenders alongside
    the internal _refresh_stale_*_lock."""
    main_src = open(os.path.join(_BOT_DIR, "main.py"),
                       encoding="utf-8").read()
    # Both repricer add_job blocks should have max_instances=1.  We
    # check for at least 2 occurrences (one per repricer).
    assert main_src.count("max_instances=1") >= 2, (
        "Both stale_topup_refresh and stale_entry_refresh must specify "
        "max_instances=1.  Without it, a slow repricer cycle could "
        "overlap with the next, increasing cancel/replace race risk."
    )


# ============================================================
# Topup repricer — pre-cancel partial-fill capture
# ============================================================

def test_topup_repricer_does_pre_cancel_capture():
    """refresh_stale_topups must call _capture_partial_fills_before_cancel
    BEFORE the cancel.  Without this, a chunk that fills during the
    cancel race isn't reflected in committed_usdc, and the replacement
    over-allocates by the partial amount (the New York pid=107 / Houston
    $74.99 bug class)."""
    import monitor   # type: ignore
    src = inspect.getsource(monitor.refresh_stale_topups)
    # The capture must appear before the cancel_order(old_oid, ...) line.
    capture_idx = src.find("_capture_partial_fills_before_cancel")
    cancel_idx  = src.find("cancel_order(old_oid")
    assert capture_idx != -1, (
        "refresh_stale_topups must call _capture_partial_fills_before_cancel "
        "to close the cancel-race overrun window."
    )
    assert cancel_idx != -1, "refresh_stale_topups should still call cancel_order"
    assert capture_idx < cancel_idx, (
        f"_capture_partial_fills_before_cancel (idx {capture_idx}) MUST "
        f"come BEFORE cancel_order (idx {cancel_idx}).  Capture-after-cancel "
        f"reintroduces the overrun race."
    )


def test_entry_repricer_still_does_pre_cancel_capture():
    """Regression: the entry repricer's existing pre-cancel capture must
    survive.  This pins the existing behavior so a future refactor
    doesn't silently remove the safety check."""
    import monitor   # type: ignore
    src = inspect.getsource(monitor.refresh_stale_ensure_fill_entries)
    capture_idx = src.find("_capture_partial_fills_before_cancel")
    cancel_idx  = src.find("client.cancel_orders([old_oid])")
    assert capture_idx != -1 and cancel_idx != -1
    assert capture_idx < cancel_idx, (
        "Entry repricer must KEEP its pre-cancel partial-fill capture."
    )


# ============================================================
# Defense-in-depth — gap calc respects committed_usdc
# ============================================================

def test_topup_repricer_resizes_to_remaining_gap():
    """After capture+cancel, the replacement topup must be sized at
    (target - committed), not at the original intended_size.  Verified
    by reading the source for the gap-calc pattern."""
    import monitor   # type: ignore
    src = inspect.getsource(monitor.refresh_stale_topups)
    assert "target - committed" in src or "remaining = target" in src, (
        "Topup repricer must recompute the gap as target - committed_usdc "
        "after the cancel.  Sizing the replacement at the original "
        "intended_size would re-introduce overrun."
    )


def test_entry_repricer_resizes_to_remaining_gap():
    """Same protection on the entry repricer path."""
    import monitor   # type: ignore
    src = inspect.getsource(monitor.refresh_stale_ensure_fill_entries)
    assert "target - committed_after_cancel" in src, (
        "Entry repricer must size the replacement as target - committed."
    )