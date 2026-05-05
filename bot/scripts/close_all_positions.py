"""
close_all_positions.py — Force-close every open live position on Polymarket
by placing marketable (cross-spread) SELL orders.

Defaults to DRY-RUN.  Pass --apply to actually submit.  Strongly recommended
to dry-run first and inspect the proposed actions.

Behavior per position:
    1. Fetch live best_bid for the position's token
    2. Cancel any existing exit-ladder order (best-effort)
    3. Submit a marketable SELL via execution.execute_exit(cross_spread=True)
    4. Log to activity_log (category='CLOSE_ALL')
    5. Sleep 0.5s between positions to avoid CLOB rate limits

What this script CANNOT do:
    * Close orphan-chain holdings (positions you own on chain but the bot
      doesn't track).  Those need a manual sell via the Polymarket UI.
      List them with --list-orphans for visibility.

Usage:
    cd bot
    python -m scripts.close_all_positions               # dry run
    python -m scripts.close_all_positions --apply       # commit
    python -m scripts.close_all_positions --list-orphans
    python -m scripts.close_all_positions --apply --include-exiting  # also close 'exiting'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Reconfigure stdout/stderr to UTF-8 so log messages containing arrow
# characters (e.g. "result=…") don't crash on Windows cp1252 consoles.
# Same fix as bot/main.py — needed because standalone scripts don't run
# through main's startup path.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore
from execution import (  # type: ignore
    get_clob_client, execute_exit, cancel_exit_order, _get_best_bid,
)


def _list_orphan_chain():
    """Read on-chain holdings via Polymarket Data API and report any
    that don't have a matching open DB position."""
    from polymarket import get_data_api_positions
    from config import WALLET_ADDRESS
    if not WALLET_ADDRESS:
        print("WALLET_ADDRESS not configured — cannot list orphans.")
        return

    chain_pos = get_data_api_positions(WALLET_ADDRESS) or []

    # Build set of bot-tracked token_ids (positions that are open OR exiting)
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT yes_token_id, no_token_id
            FROM positions
            WHERE COALESCE(is_paper, 0) = 0
              AND status IN ('open', 'exiting')
        """).fetchall()
    tracked: set[str] = set()
    for r in rows:
        for t in (r["yes_token_id"], r["no_token_id"]):
            if t:
                tracked.add(str(t))

    orphans = [p for p in chain_pos if str(p.get("token_id", "")) not in tracked]
    print()
    print(f"=== Chain orphans ({len(orphans)} positions) ===")
    print(f"These are on-chain holdings the bot doesn't track. Sell manually if desired.")
    print()
    if not orphans:
        print("  (none)")
        return
    print(f"{'token_id':<22}  {'size':>10}  {'avg':>6}  {'title':<60}")
    print("-" * 110)
    for p in orphans:
        tok = str(p.get("token_id", ""))[:18] + "..."
        size = float(p.get("size", 0) or 0)
        avg  = float(p.get("avg_price", 0) or 0)
        title = (p.get("title", "") or "")[:60]
        print(f"  {tok:<22}  {size:>10.4f}  {avg:>6.4f}  {title}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually place sell orders.  Default is dry-run.")
    ap.add_argument("--include-exiting", action="store_true",
                    help="Also force-close positions already in 'exiting' "
                         "state (cancels their existing ladder + cross-spreads).")
    ap.add_argument("--list-orphans", action="store_true",
                    help="Also list on-chain holdings the bot doesn't track")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to sleep between sells (rate-limit guard, default 0.5)")
    args = ap.parse_args()

    client = get_clob_client()
    if client is None:
        print("ERROR: no CLOB client available (paper mode or missing creds)")
        return 1

    # Pick which positions are in scope
    statuses = ["open"]
    if args.include_exiting:
        statuses.append("exiting")
    placeholders = ",".join("?" for _ in statuses)

    with _get_conn() as conn:
        positions = [dict(r) for r in conn.execute(f"""
            SELECT id, contract_id, side, yes_token_id, no_token_id,
                   shares, entry_price, size_usdc, status, fill_status,
                   exit_order_id, city, date, range_low, range_high
            FROM positions
            WHERE COALESCE(is_paper, 0) = 0
              AND status IN ({placeholders})
              AND COALESCE(shares, 0) > 0
            ORDER BY id
        """, statuses).fetchall()]

    mode = "APPLY" if args.apply else "DRY RUN"
    print()
    print(f"Close-all positions — mode: {mode}")
    print(f"Found {len(positions)} live position(s) in scope (status: {', '.join(statuses)})")
    print()

    if not positions:
        print("Nothing to do.")
        if args.list_orphans:
            _list_orphan_chain()
        return 0

    # Show plan
    print(f"{'pid':>4}  {'city':<14}  {'date':<11}  {'side':<3}  "
          f"{'shares':>7}  {'entry':>6}  {'best_bid':>8}  {'status':<8}")
    print("-" * 90)

    plan = []
    for pos in positions:
        side = pos.get("side", "YES")
        token_id = (pos.get("yes_token_id") if side == "YES"
                    else pos.get("no_token_id"))
        if not token_id:
            print(f"  pid={pos['id']:<4} {pos.get('city','?'):<14} "
                  f"NO TOKEN_ID — skipping")
            continue
        bid = _get_best_bid(client, token_id) if args.apply else None
        bid_str = f"{bid:.4f}" if bid else "?(dry)"
        print(f"  {pos['id']:>4}  {str(pos.get('city',''))[:14]:<14}  "
              f"{str(pos.get('date',''))[:11]:<11}  "
              f"{str(side)[:3]:<3}  "
              f"{float(pos['shares']):>7.2f}  "
              f"{float(pos.get('entry_price') or 0):>6.4f}  "
              f"{bid_str:>8}  {pos['status']:<8}")
        plan.append(pos)

    if not args.apply:
        print()
        print(f"Dry run — no sells placed.  {len(plan)} would be closed.")
        print("Re-run with --apply to commit.")
        if args.list_orphans:
            _list_orphan_chain()
        return 0

    # Execute the closes
    print()
    print(f"Executing {len(plan)} cross-spread sell(s) with {args.sleep}s spacing...")
    print()

    results = {"closed": 0, "error": 0, "self_heal": 0, "skipped": 0}

    for pos in plan:
        pid = pos["id"]
        city = pos.get("city", "")

        # Cancel any existing exit-ladder order first (best-effort).  If it
        # fails (425, already filled, etc.) we still try the new sell.
        if pos.get("exit_order_id"):
            try:
                cancelled_ok = cancel_exit_order(pos, client)
                if not cancelled_ok:
                    print(f"  pid={pid:<4} {city:<14}  warning: existing exit "
                          f"order {pos['exit_order_id'][:12]} cancel failed — proceeding anyway")
            except Exception as e:
                print(f"  pid={pid:<4} {city:<14}  cancel error: {e}")

        # Determine intended_exit_price for the log + retry_count semantics.
        # Use entry_price as the trigger reference (what we'd "intended" to
        # exit at if a stop fired now); cross_spread overrides actual price
        # to bid-1tick anyway.
        intended_exit_price = float(pos.get("entry_price") or 0.50)

        try:
            result = execute_exit(
                position             = pos,
                intended_exit_price  = intended_exit_price,
                exit_reason          = f"manual_close_all_script",
                client               = client,
                retry_count          = 4,        # past ladder, force cross-spread path
                cross_spread         = True,
            )
        except Exception as e:
            print(f"  pid={pid:<4} {city:<14}  EXCEPTION: {e}")
            results["error"] += 1
            continue

        status = result.get("status")
        if status == "exit_pending":
            print(f"  pid={pid:<4} {city:<14}  SELL placed @ "
                  f"{result.get('limit_price', '?')} (will fill via WS)")
            results["closed"] += 1
        elif status == "closed_via_balance_recovery":
            print(f"  pid={pid:<4} {city:<14}  closed via self-heal "
                  f"(chain had 0 shares, approx pnl=${result.get('approx_pnl', 0):+.2f})")
            results["self_heal"] += 1
        elif status == "shares_resynced":
            print(f"  pid={pid:<4} {city:<14}  shares re-synced to chain "
                  f"({result.get('new_shares')}); next monitor cycle will retry")
            results["skipped"] += 1
        else:
            print(f"  pid={pid:<4} {city:<14}  unexpected status={status}: "
                  f"{result.get('reason', '')}")
            results["error"] += 1

        # Audit log
        try:
            from activity import log_activity
            log_activity(
                "CLOSE_ALL", level="WARN", position_id=pid,
                message=(
                    f"manual close-all: pid={pid} {pos.get('city')} "
                    f"{pos.get('date')} → result={status}"
                ),
                source="close_all_script",
                shares=float(pos["shares"]),
                exit_status=status,
            )
        except Exception:
            pass

        # Brief pause to avoid CLOB rate-limit
        time.sleep(args.sleep)

    print()
    print("=== Summary ===")
    print(f"  cross-spread sells placed:        {results['closed']}")
    print(f"  closed via balance-mismatch heal: {results['self_heal']}")
    print(f"  shares re-synced (retry needed):  {results['skipped']}")
    print(f"  errors:                           {results['error']}")
    print()
    print("Closed positions will transition to status='closed' in the DB once")
    print("their trade events confirm via the WebSocket fill handler.  Watch")
    print("the bot terminal for [FILL] entries over the next ~30 seconds.")

    if args.list_orphans:
        _list_orphan_chain()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
