"""
reset_for_new_strategy.py — Wipe trading-state tables so the bot starts
clean under a new strategy.  Preserves all observability, history, and
discovery data.

What it CLEARS by default (--apply required to commit):
    * positions                (live only; --include-paper to also clear paper)
    * position_orders          (per-CLOB-order ledger)
    * processed_trade_events   (trade-event dedup state)

What it PRESERVES (always):
    * activity_log             (audit trail of every bot action)
    * monitor_health           (per-cycle health snapshots)
    * temp_events / temp_outcomes  (discovery + bin-price scan history)
    * event_resolutions        (past winners, used by analyze_stop_losses)
    * All weather-pipeline tables (forecast_runs, live_observations, etc.)
    * All ML / climatology tables
    * signals                  (legacy, empty)

The reset is logged to activity_log (category='RESET') so the cleared
state is auditable: timestamp, table, rows-deleted, mode (live vs paper).

Usage:
    cd bot
    python -m scripts.reset_for_new_strategy                  # dry run
    python -m scripts.reset_for_new_strategy --apply          # commit
    python -m scripts.reset_for_new_strategy --apply --include-paper  # also paper
    python -m scripts.reset_for_new_strategy --apply --vacuum  # reclaim disk space
"""

from __future__ import annotations

import argparse
import os
import sys
import sqlite3
from datetime import datetime, timezone

# UTF-8 stdout/stderr so logger arrows etc. don't crash on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn, DB_PATH  # type: ignore


# Tables whose row counts we report + clear (in order, due to logical
# dependencies: position_orders references positions via position_id).
TABLES_TO_CLEAR = ["position_orders", "processed_trade_events", "positions"]


def _row_counts(conn) -> dict[str, int]:
    """Snapshot row counts for the trading-state tables."""
    out: dict[str, int] = {}
    for t in TABLES_TO_CLEAR:
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.DatabaseError:
            out[t] = -1   # corrupt or missing
    return out


def _open_position_summary(conn, include_paper: bool) -> dict:
    """Stats on what we're about to delete: open vs closed, live vs paper."""
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"
    rows = conn.execute(f"""
        SELECT
            SUM(CASE WHEN status IN ('open', 'exiting') THEN 1 ELSE 0 END) AS n_open,
            SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END)             AS n_closed,
            SUM(CASE WHEN COALESCE(is_paper, 0) = 0 THEN 1 ELSE 0 END)     AS n_live,
            SUM(CASE WHEN COALESCE(is_paper, 0) = 1 THEN 1 ELSE 0 END)     AS n_paper,
            COUNT(*)                                                       AS n_total
        FROM positions
        WHERE 1=1 {paper_clause}
    """).fetchone()
    return {
        "n_open":   row_or_zero(rows, "n_open"),
        "n_closed": row_or_zero(rows, "n_closed"),
        "n_live":   row_or_zero(rows, "n_live"),
        "n_paper":  row_or_zero(rows, "n_paper"),
        "n_total":  row_or_zero(rows, "n_total"),
    }


def row_or_zero(row, key):
    if row is None:
        return 0
    try:
        return int(row[key] or 0)
    except (KeyError, TypeError, ValueError):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Commit the deletes.  Without this, runs in dry-run mode.")
    ap.add_argument("--include-paper", action="store_true",
                    help="Also delete paper positions (default: live only).")
    ap.add_argument("--vacuum", action="store_true",
                    help="Run VACUUM after deletes to reclaim disk space.  "
                         "Slower but cleaner.")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return 1

    print()
    print(f"Reset script — DB: {DB_PATH}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Include paper: {args.include_paper}")
    print()

    # ---- Snapshot before ----
    with _get_conn() as conn:
        counts_before = _row_counts(conn)
        pos_summary = _open_position_summary(conn, args.include_paper)

    print("Before:")
    for t, n in counts_before.items():
        print(f"  {t:<28} {n:>10,} rows")
    print()
    print(f"Positions in scope ({'live + paper' if args.include_paper else 'live only'}):")
    print(f"  open / exiting:         {pos_summary['n_open']:>10,}")
    print(f"  closed:                 {pos_summary['n_closed']:>10,}")
    print(f"  total in scope:         {pos_summary['n_total']:>10,}")
    if not args.include_paper:
        print(f"  paper (preserved):      {pos_summary['n_paper']:>10,}")
    print()

    if not args.apply:
        print("Dry run — no DB writes performed.")
        print(f"Re-run with --apply to clear {sum(counts_before.values())} rows "
              f"across {len([t for t in TABLES_TO_CLEAR if counts_before[t] > 0])} table(s).")
        if args.include_paper:
            print("(--include-paper) — paper positions will ALSO be cleared.")
        return 0

    # ---- Audit log BEFORE deletes (so we capture the pre-state) ----
    try:
        from activity import log_activity
        log_activity(
            "RESET", level="WARN",
            message=(
                f"DB reset for new strategy: clearing {pos_summary['n_total']} "
                f"position(s) + {counts_before['position_orders']} ledger row(s) "
                f"+ {counts_before['processed_trade_events']} dedup row(s)"
            ),
            include_paper=args.include_paper,
            **{f"before_{k}": v for k, v in counts_before.items()},
            **{f"positions_{k}": v for k, v in pos_summary.items()},
        )
    except Exception as e:
        print(f"WARNING: could not write audit-log entry before reset: {e}")

    # ---- Delete ----
    deleted: dict[str, int] = {}
    paper_clause = "" if args.include_paper else "AND COALESCE(is_paper, 0) = 0"

    with _get_conn() as conn:
        # Delete position_orders for positions we'll delete
        cur = conn.execute(f"""
            DELETE FROM position_orders
            WHERE position_id IN (
                SELECT id FROM positions WHERE 1=1 {paper_clause}
            )
        """)
        deleted["position_orders"] = cur.rowcount

        # Clear the trade-event dedup table entirely (it's session-state,
        # never useful across a strategy reset).
        cur = conn.execute("DELETE FROM processed_trade_events")
        deleted["processed_trade_events"] = cur.rowcount

        # Delete positions
        cur = conn.execute(f"DELETE FROM positions WHERE 1=1 {paper_clause}")
        deleted["positions"] = cur.rowcount

    # ---- Snapshot after ----
    with _get_conn() as conn:
        counts_after = _row_counts(conn)

    print("Deleted:")
    for t, n in deleted.items():
        print(f"  {t:<28} {n:>10,} rows")
    print()
    print("After:")
    for t, n in counts_after.items():
        print(f"  {t:<28} {n:>10,} rows")
    print()

    # ---- VACUUM (optional, slow on big DBs) ----
    if args.vacuum:
        print("Running VACUUM to reclaim disk space (may take a few seconds)...")
        with sqlite3.connect(DB_PATH) as conn:
            # VACUUM cannot run inside a transaction nor on a WAL connection
            # that holds locks; use a fresh raw connection.
            conn.execute("VACUUM")
        print("VACUUM complete.")
        print()

    # ---- Audit log AFTER ----
    try:
        from activity import log_activity
        log_activity(
            "RESET", level="INFO",
            message=(
                f"DB reset complete: deleted {deleted['positions']} positions, "
                f"{deleted['position_orders']} ledger rows, "
                f"{deleted['processed_trade_events']} dedup rows. "
                f"Bot is ready for a clean start."
            ),
            **{f"deleted_{k}": v for k, v in deleted.items()},
            **{f"after_{k}": v for k, v in counts_after.items()},
            vacuum_run=args.vacuum,
        )
    except Exception as e:
        print(f"WARNING: could not write audit-log entry after reset: {e}")

    print(f"Done.  See activity_log category='RESET' for the audit trail.")
    print(f"You can now restart the bot with the new strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
