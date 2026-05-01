"""
backfill_position_fees.py — Pull real Polymarket trade fees post-hoc and
update positions.entry_fees / exit_fees / pnl_net for any closed positions.

Why this exists:
    Polymarket's WebSocket trade events do NOT include fee_rate_bps, so
    the bot's fill handler can only record fees as 0.  Real taker fees
    (1000 bps = 10% as observed in production samples) are present in
    the CLOB's GET /trades response, which we fetch + sum here.

The bot now auto-runs this for each position when its exit completes
(see fill_handler.py).  This script handles existing closed positions
that were closed BEFORE the auto-backfill landed.

Usage:
    cd bot
    python -m scripts.backfill_position_fees                # all closed live positions
    python -m scripts.backfill_position_fees --pid 222      # single position
    python -m scripts.backfill_position_fees --apply        # required to commit;
                                                              dry-run by default
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore
from execution import backfill_position_fees, get_clob_client  # type: ignore


def _list_target_pids(only_pid: int | None) -> list[dict]:
    """Return the list of position rows to process — all live closed
    positions (default), or a single pid if --pid was passed."""
    sql = """
        SELECT id, city, date, side, pnl, pnl_net, entry_fees, exit_fees,
               status, fill_status, is_paper
        FROM positions
        WHERE COALESCE(is_paper, 0) = 0
          AND status = 'closed'
    """
    args: tuple = ()
    if only_pid is not None:
        sql += " AND id = ?"
        args = (only_pid,)
    sql += " ORDER BY id ASC"
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pid", type=int, default=None,
        help="Backfill a single position id (default: all closed live positions)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Commit changes.  Without this flag, runs in dry-run mode "
             "(prints the plan without writing).",
    )
    args = ap.parse_args()

    targets = _list_target_pids(args.pid)
    if not targets:
        print("No closed live positions to process.")
        return 0

    client = get_clob_client()
    if client is None:
        print("ERROR: no CLOB client available (paper mode or missing creds)")
        return 1

    mode = "APPLY" if args.apply else "DRY RUN"
    print()
    print(f"Fee backfill — {len(targets)} closed live position(s) — mode: {mode}")
    print()
    print(f"{'pid':>4}  {'city':<14}  {'date':<11}  {'gross':>8}  "
          f"{'old_fees':>9}  {'new_fees':>9}  {'old_net':>8}  {'new_net':>8}  trades")
    print("-" * 100)

    summary = {"updated": 0, "skipped_no_trades": 0, "no_change": 0, "errors": 0}

    for pos in targets:
        pid = pos["id"]
        old_total_fees = float((pos.get("entry_fees") or 0) + (pos.get("exit_fees") or 0))
        old_net = pos.get("pnl_net")
        gross   = float(pos.get("pnl") or 0)
        try:
            # commit=True only when --apply.  Dry-run computes + reports
            # without writing.
            result = backfill_position_fees(pid, client, commit=args.apply)
        except Exception as e:
            print(f"  pid={pid:<4} {pos.get('city',''):<14} ERROR: {e}")
            summary["errors"] += 1
            continue

        if not result:
            print(f"  pid={pid:<4} {pos.get('city',''):<14} (no trades returned / no token)")
            summary["skipped_no_trades"] += 1
            continue

        new_fees = result["entry_fees"] + result["exit_fees"]
        new_net  = result["pnl_net"]
        n_match  = result["n_trades_matched"]

        old_net_str = f"${old_net:+.2f}" if old_net is not None else "  --  "
        delta_str = ""
        if abs(new_fees - old_total_fees) > 0.0001:
            delta_str = f"  Δ=+${new_fees - old_total_fees:.4f}"
            summary["updated"] += 1
        else:
            summary["no_change"] += 1

        print(f"  {pid:>4}  {str(pos.get('city',''))[:14]:<14}  "
              f"{str(pos.get('date',''))[:11]:<11}  "
              f"${gross:>+7.2f}  ${old_total_fees:>+8.4f}  ${new_fees:>+8.4f}  "
              f"{old_net_str:>8}  ${new_net:>+7.2f}  {n_match:>2}{delta_str}")

    print()
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if not args.apply:
        print()
        print("Dry run — no DB writes performed.  Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
