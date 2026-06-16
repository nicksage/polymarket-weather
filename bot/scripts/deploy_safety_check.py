#!/usr/bin/env python3
"""
deploy_safety_check.py — Smoke test that the load-bearing names in
scheduled_predictor.py are present.  Designed to be cheap, fast, and
run from a pre-commit hook so silent regressions can't leave the
developer's machine.

=== Why this exists ===

On 2026-06-12 a commit silently removed three production features
from bot/scheduled_predictor.py at the same time it added unrelated
new files:
  - W4 risk-cap constants and gate (MARKET_DISAGREEMENT_*,
    liquid_market_strong_disagreement)
  - The NWS-evening-scan recovery helper (recover_persisted_day_forecast
    and the log line "forecast_high recovered for")
  - MAX_MARKET_PRICE constant — left a latent NameError in evaluate_gates
    that would crash any signal with market_p >= 0.95

All three were running in production on the prior commit and were not
intended to be removed.  The deleted code wasn't caught by any test
because the regression masked a test-collection failure rather than a
test assertion failure.  Manual restore took ~15 minutes.

This script encodes those load-bearing names as a smoke test.  It is
NOT a substitute for unit tests — it's a CHEAP precondition check that
runs in <50ms before every commit.  Tests are the second line of
defense; this check is the first.

=== How to use ===

Manual:
    python bot/scripts/deploy_safety_check.py

Pre-commit hook (recommended — install once per fresh clone):
    python bot/scripts/install_hooks.py

After install, every `git commit` runs this check automatically and
blocks the commit if any required name is missing.

=== Adding / removing required names ===

To require a new name:    append to REQUIRED_NAMES below.
To deprecate a name:      delete from REQUIRED_NAMES in the SAME commit
                           that removes the source feature.  Otherwise
                           the next commit will fail until the list
                           catches up — which is the point.

Exit codes:
    0 — all required names present
    1 — one or more missing; commit/deploy is blocked

To bypass for one commit (EMERGENCY ONLY, e.g. mid-rebase hotfix):
    git commit --no-verify
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = REPO_ROOT / "bot" / "scheduled_predictor.py"

# Substrings that must be present in scheduled_predictor.py for the
# bot to be safe to deploy.  Grouped by feature for legibility, but
# the check is just substring presence.
REQUIRED_NAMES: list[str] = [
    # --- W4: market-anchored risk cap ---
    "MARKET_DISAGREEMENT_LIQ_THRESHOLD",
    "MARKET_DISAGREEMENT_PP_THRESHOLD",
    "liquid_market_strong_disagreement",   # gate reason emitted by W4

    # --- Other load-bearing gate constants ---
    # If any of these go missing, evaluate_gates raises NameError on
    # whatever signal first hits the reference.
    "MIN_MARKET_PROB",
    "MAX_MARKET_PRICE",
    "MIN_EDGE",
    "MIN_LIQUIDITY_USD",
    "MAX_DAILY_EXPOSURE",
    "MAX_TRADES_PER_DAY",

    # --- Forecast-recovery (NWS hourly evening-scan bug fix) ---
    "def recover_persisted_day_forecast",   # the helper itself
    "forecast_high recovered for",          # the log line — confirms wiring

    # --- Core entry points ---
    "def evaluate_gates",
    "def run_intraday_scan",
    "def ensure_schema",

    # --- Three-signal position model (W1 companion work) ---
    "market_closed",                         # column written per bin
    "data_quality_flag",                     # column written per bin

    # --- Invariant guards integration ---
    "run_invariant_checks",                  # called at scan end

    # --- Data-quality contract: cold-start + sizing scalar ---
    "COLD_START_PEAK_HOUR_LOCAL",            # constant for cold-start cutoff
    "def is_cold_start_day",                 # detector helper
    "cold_start_suspect",                    # flag value appended in scan loop
    "def compute_data_quality_size_factor",  # MIN-over-flags sizing helper
    "DATA_QUALITY_SIZE_COLD_START_SUSPECT",  # absolute-trustability haircut
    "DATA_QUALITY_SIZE_BLOCK",               # don't-trade tier
    "event_size_factor",                     # the actual sizing-application wiring
    "data_quality_blocked",                  # SKIP reason when size_factor=0

    # --- raw_metar_log temp_precision column (T-group support) ---
    "temp_precision",                        # column in raw_metar_log

    # --- Pending-order race fix (NYC double position, Houston $74.99) ---
    "def pending_contracts_today",           # set of contracts with placed orders
    "def pending_stake_for_contract_today",  # $ sum of pending stakes per contract
    "committed_contracts",                   # held + pending — for cap check
    "committed_deployed",                    # actual + pending — for target math

    # --- HRRR ceiling Phase 0 + Phase 1 (docs/hrrr_ceiling_spec.md) ---
    "resolution_observations",               # Phase 0a audit table
    "cold_start_suspect",                    # already required, but pulls
                                              # into the HRRR dispatch path too

    # --- Quick Fixes A/B/C — default ON, ship-now, not data-gated ---
    "plausibility_ceiling_applied",          # data_quality_flag value

    # --- Buy-mode dispatch (PREDICTOR_BUY_MODE) ---
    # Switching this to "probability" must not raise NameError on the
    # threshold constant or the gate-reason string.
    "PREDICTOR_BUY_MODE",
    "PREDICTOR_MIN_PROB_TO_BUY",
    "low_prob",                              # probability-mode gate reason

    # --- Per-mode execution price ceiling (2026-06-13) ---
    # Probability mode must thread its own max_price_cap to execute_signal,
    # otherwise expensive bins silently hit MPV_MAX_PRICE=0.32 and never fill.
    "PREDICTOR_PROBABILITY_MAX_PRICE",
    "max_price_cap",                         # key in sig_for_exec dict

    # --- Boundary-crossing latency strategy (2026-06-16) ---
    # Schema additions live in scheduled_predictor's _SCHEMA_SQL.  If any
    # of these go missing, boundary_watcher.write_trigger_log silently
    # no-ops and we lose the entire learning signal.
    "boundary_trigger_log",                  # table name in _SCHEMA_SQL
    "signal_origin",                         # column on positions table
]

# Names that must be present in bot/main.py to confirm the boundary
# watcher is wired into the scheduler.  Without these the module loads
# but no job fires, and the strategy silently does nothing.
MAIN_REQUIRED_NAMES: list[str] = [
    "boundary_watcher",                      # import line
    "run_boundary_watcher_tick",             # function called from jobs
    "BOUNDARY_STRATEGY_ENABLED",             # gate check
    "_boundary_normal_job",                  # APScheduler job
    "_boundary_hard_poll_job",               # APScheduler job
]

# Names that must be present in bot/boundary_watcher.py itself.  The
# whole module is new; if a refactor accidentally deletes a critical
# helper the import succeeds (Python is forgiving) but the trigger
# pipeline breaks at runtime in a hard-to-debug way.
BOUNDARY_REQUIRED_NAMES: list[str] = [
    "BOUNDARY_STRATEGY_ENABLED",
    "BOUNDARY_DRY_RUN",
    "BOUNDARY_MAX_ENTRY_PRICE",
    "def compute_arming_state",
    "def evaluate_trigger",
    "def run_boundary_watcher_tick",
    "def write_trigger_log",
    "def in_hard_poll_window",
    "def is_supported_bin",
    "def settlement_unit_round",
]

MAIN_FILE = REPO_ROOT / "bot" / "main.py"
BOUNDARY_FILE = REPO_ROOT / "bot" / "boundary_watcher.py"


def _check_file(path: Path, required: list[str]) -> list[tuple[str, str]]:
    """Return list of (file_label, missing_name) for any required name
    not present in the file at `path`.  If the file itself is missing,
    every required name counts as missing."""
    label = path.name
    if not path.exists():
        return [(label, f"<file not found: {path}>")]
    src = path.read_text(encoding="utf-8")
    return [(label, n) for n in required if n not in src]


def main() -> int:
    checks = [
        (TARGET_FILE, REQUIRED_NAMES),
        (MAIN_FILE, MAIN_REQUIRED_NAMES),
        (BOUNDARY_FILE, BOUNDARY_REQUIRED_NAMES),
    ]
    total_required = sum(len(r) for _, r in checks)
    missing: list[tuple[str, str]] = []
    for path, required in checks:
        missing.extend(_check_file(path, required))

    if not missing:
        print(f"deploy_safety_check: PASS "
              f"({total_required} required names present across "
              f"{len(checks)} files)")
        return 0

    print("=" * 70, file=sys.stderr)
    print("DEPLOY SAFETY VIOLATION", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"{len(missing)} required name(s) missing:", file=sys.stderr)
    for label, name in missing:
        print(f"  - [{label}] {name!r}", file=sys.stderr)
    print("", file=sys.stderr)
    print("This commit would remove load-bearing functionality.  If the", file=sys.stderr)
    print("removal is intentional, edit the relevant *_REQUIRED_NAMES list in", file=sys.stderr)
    print(f"  {Path(__file__).relative_to(REPO_ROOT)}", file=sys.stderr)
    print("in the SAME commit as the source file change.", file=sys.stderr)
    print("", file=sys.stderr)
    print("To bypass for one commit (emergency only):", file=sys.stderr)
    print("  git commit --no-verify", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())