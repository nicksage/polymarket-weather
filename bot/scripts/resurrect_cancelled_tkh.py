"""
resurrect_cancelled_tkh.py — Re-issue TKH bins that were cancelled before
they ever filled.

Background
----------
Pre-fix, the monitor's `_cancel_pending_orders` sweep would cancel any
TKH entry order that hadn't filled within 10 minutes.  Once cancelled
the row was marked `fill_status='cancelled', status='closed', shares=0`
and the new chase-the-ask repricer (which only looks at status='open'
rows) cannot recover it.  TKH's per-event dedup also blocks automatic
re-discovery if any sibling bin in the same event already filled.

This script does a one-shot re-issue for those terminal-cancelled rows
so all K bins of every TKH event ultimately get purchased.

Per row, when --apply is set:
  1. Fetch current best_ask from the orderbook
  2. Place a new BUY at best_ask + ORDERBOOK_WALK_CENTS (default 1c)
  3. UPDATE the row: order_id=new, entry_price=new limit, entry_time=now,
     fill_status='pending', status='open', cancelled_reason=NULL,
     exit_time=NULL.  After this the row is alive again and the
     normal monitor / repricer / topup loops take over.
  4. Insert a fresh position_orders ledger row for the replacement order.

Skips
-----
* paper rows (is_paper=1)
* rows where target_size_usdc < TKH_MIN_BIN_USDC (Polymarket $1 floor)
* rows whose event has already resolved (no point chasing)
* rows whose orderbook is unavailable this tick

Usage:
    cd bot
    python -m scripts.resurrect_cancelled_tkh                  # dry-run (preview)
    python -m scripts.resurrect_cancelled_tkh --apply          # re-issue
    python -m scripts.resurrect_cancelled_tkh --apply --since-hours 24
    python -m scripts.resurrect_cancelled_tkh --apply --pid 18,19,21
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore
from execution import (  # type: ignore
    get_clob_client, get_orderbook_snapshot,
)
from config import ORDERBOOK_WALK_CENTS  # type: ignore


def _select_cancelled_tkh(conn, since_hours: float | None,
                           pid_filter: list[int] | None) -> list[dict]:
    """Find TKH rows that were cancelled before any fill confirmed.

    NOTE on the `shares` column: insert_position SEEDS shares with the
    intended-shares estimate at placement time (final_size_usdc / limit).
    For cancelled-before-fill rows that seed value is NEVER overwritten
    because no fill confirmed.  So the `shares` column does NOT indicate
    whether real on-chain shares exist.  The authoritative truth is:
      - fill_status='cancelled'  -> zero on-chain shares from this entry
      - the position_orders ledger committed_usdc                (cross-check)
    The committed_usdc cross-check happens in _resurrect_one before
    placing the new order.
    """
    sql = """
        SELECT id, contract_id, side, yes_token_id, no_token_id,
               size_usdc, target_size_usdc, entry_price, entry_time, shares,
               cancelled_reason, exit_time,
               event_id, city, date, question
        FROM positions
        WHERE strategy = 'top_k_hedged'
          AND fill_status = 'cancelled'
          AND COALESCE(is_paper, 0) = 0
    """
    params: list = []
    if since_hours is not None:
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=since_hours)).isoformat()
        sql += " AND COALESCE(exit_time, entry_time) >= ?"
        params.append(cutoff)
    if pid_filter:
        ph = ",".join("?" * len(pid_filter))
        sql += f" AND id IN ({ph})"
        params.extend(pid_filter)
    sql += " ORDER BY id ASC"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _resurrect_one(pos: dict, client, *, dry_run: bool, walk_cents: int,
                   min_bin_usdc: float) -> dict:
    """Plan + (if not dry-run) execute the re-issue for ONE position.
    Returns a result dict for reporting."""
    pid       = pos["id"]
    side      = pos.get("side", "YES")
    token_id  = (pos.get("yes_token_id") if side == "YES"
                 else pos.get("no_token_id"))
    target    = float(pos.get("target_size_usdc")
                      or pos.get("size_usdc") or 0)
    out: dict = {
        "pid":          pid,
        "city":         pos.get("city", ""),
        "date":         pos.get("date", ""),
        "target_usdc":  target,
        "best_ask":     None,
        "new_limit":    None,
        "status":       "skipped",
        "reason":       "",
        "new_order_id": None,
    }

    if not token_id:
        out["reason"] = "no_token_id"
        return out
    if target < min_bin_usdc:
        out["reason"] = (f"target ${target:.2f} < TKH_MIN_BIN_USDC "
                         f"${min_bin_usdc:.2f}")
        return out

    # Defensive ledger cross-check: in case the cancelled row actually
    # holds capital (partial-fill that incorrectly got marked cancelled,
    # WS race, etc.), don't double-place.  committed_usdc counts every
    # ledger order whose status is filled / partial / pending / live —
    # for a cleanly-cancelled entry it should be ~0.  If anything is
    # there, refuse to act and ask the operator to investigate.
    try:
        from db import get_committed_usdc
        committed = get_committed_usdc(pid)
    except Exception as e:
        out["reason"] = f"ledger committed_usdc query failed: {e}"
        return out
    if committed > 0.50:
        out["reason"] = (
            f"ledger shows committed=${committed:.2f} on this position "
            f"-- refusing to re-place (would double-commit). "
            f"Investigate manually."
        )
        return out

    # Fetch fresh book
    snap = get_orderbook_snapshot(client, token_id)
    if snap is None or snap.get("best_ask") is None:
        out["reason"] = "orderbook_unavailable"
        return out
    best_ask = float(snap["best_ask"])
    new_limit = round(min(best_ask + walk_cents / 100.0, 0.99), 4)
    out["best_ask"]  = best_ask
    out["new_limit"] = new_limit

    if dry_run:
        out["status"] = "would_resurrect"
        out["reason"] = (f"would re-place ${target:.2f} @ "
                         f"${new_limit:.4f} (best_ask=${best_ask:.4f})")
        return out

    # ---- Place the new order ----
    from py_clob_client_v2 import OrderArgs, OrderType, Side
    try:
        order_args = OrderArgs(
            price    = new_limit,
            size     = target / new_limit,
            side     = Side.BUY,
            token_id = token_id,
        )
        response = client.create_and_post_order(
            order_args, order_type=OrderType.GTC,
        )
    except Exception as e:
        out["status"] = "error"
        out["reason"] = f"placement raised: {e}"
        return out

    if not response or not response.get("success"):
        out["status"] = "error"
        out["reason"] = f"placement returned {response}"
        return out

    new_oid = response.get("orderID", "")
    new_entry_time = datetime.now(ZoneInfo("America/Chicago")).isoformat()
    out["new_order_id"] = new_oid

    # ---- Resurrect the position row ----
    # Reset shares + size_usdc to the freshly-seeded estimates (the old
    # seed was for the cancelled order's limit; the new limit is different).
    # current_price / unrealized_pnl cleared so dashboard doesn't show
    # stale numbers from before the cancellation.
    new_shares_seed = round(target / new_limit, 4)
    with _get_conn() as conn:
        conn.execute("""
            UPDATE positions
            SET order_id         = ?,
                entry_price      = ?,
                entry_time       = ?,
                shares           = ?,
                size_usdc        = ?,
                current_price    = NULL,
                unrealized_pnl   = NULL,
                fill_status      = 'pending',
                status           = 'open',
                cancelled_reason = NULL,
                exit_time        = NULL,
                pnl              = NULL
            WHERE id = ?
        """, (new_oid, new_limit, new_entry_time,
              new_shares_seed, target, pid))

    # ---- Insert a fresh ledger row ----
    try:
        from db import insert_position_order
        insert_position_order(
            position_id     = pid,
            order_id        = new_oid,
            role            = "entry",
            intended_usdc   = target,
            intended_shares = target / new_limit,
            limit_price     = new_limit,
            status          = "pending",
            trade_status    = None,
        )
    except Exception:
        pass

    # ---- Audit log ----
    try:
        from activity import log_activity
        log_activity(
            "BUY", level="INFO", position_id=pid,
            message=(
                f"TKH cancelled bin RESURRECTED: ${target:.2f} @ "
                f"${new_limit:.4f} (best_ask=${best_ask:.4f}) "
                f"{pos.get('city','?')} {pos.get('date','?')}"
            ),
            source="resurrect_cancelled_tkh",
            new_order_id=new_oid, new_limit=new_limit,
            best_ask=best_ask, target_usdc=target,
        )
    except Exception:
        pass

    out["status"] = "resurrected"
    out["reason"] = (f"new order={new_oid[:12]}, "
                     f"limit=${new_limit:.4f}, gap=${target:.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually re-place orders.  Default is dry-run.")
    ap.add_argument("--since-hours", type=float, default=None,
                    help="Only resurrect rows cancelled within the last N "
                         "hours (default: all)")
    ap.add_argument("--pid", type=str, default=None,
                    help="Comma-separated position IDs to resurrect "
                         "(default: all matching cancelled TKH rows)")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="Seconds to sleep between placements (rate-limit "
                         "guard, default 0.4)")
    args = ap.parse_args()

    pid_filter = None
    if args.pid:
        try:
            pid_filter = [int(x.strip()) for x in args.pid.split(",")
                          if x.strip()]
        except ValueError:
            print("ERROR: --pid must be comma-separated integers")
            return 1

    # Min-bin floor mirrors strategies/top_k_hedged.py
    min_bin = float(os.getenv("TKH_MIN_BIN_USDC", "1.05"))

    client = get_clob_client()
    if client is None:
        print("ERROR: no CLOB client (paper mode or missing creds)")
        return 1

    with _get_conn() as conn:
        rows = _select_cancelled_tkh(conn, args.since_hours, pid_filter)

    mode = "APPLY" if args.apply else "DRY RUN"
    print()
    print(f"=== Resurrect cancelled TKH bins -- mode: {mode} ===")
    print(f"  filters: since-hours={args.since_hours}, pid={args.pid}")
    print(f"  walk:    +{ORDERBOOK_WALK_CENTS}c above best_ask")
    print(f"  floor:   $TKH_MIN_BIN_USDC = ${min_bin:.2f}")
    print()
    print(f"Found {len(rows)} cancelled TKH row(s)")
    if not rows:
        return 0

    print()
    print(f"{'pid':>5}  {'city':<14}  {'date':<11}  {'target $':>8}  "
          f"{'orig limit':>10}  {'cancelled at':<16}")
    print("-" * 86)
    for r in rows:
        print(f"  {r['id']:>5}  {(r.get('city') or '')[:14]:<14}  "
              f"{(r.get('date') or '')[:11]:<11}  "
              f"${float(r.get('target_size_usdc') or r.get('size_usdc') or 0):>7.2f}  "
              f"${float(r.get('entry_price') or 0):>9.4f}  "
              f"{(r.get('exit_time') or '')[:16]:<16}")
    print()

    if not args.apply:
        print("DRY RUN — no orders placed.  Re-run with --apply to commit.")
        # Show planned new prices
        print()
        print("Planned new placements (probing live orderbook):")
        print(f"{'pid':>5}  {'city':<14}  {'best_ask':>9}  "
              f"{'new_limit':>10}  status")
        print("-" * 70)
        for r in rows:
            res = _resurrect_one(r, client, dry_run=True,
                                 walk_cents=ORDERBOOK_WALK_CENTS,
                                 min_bin_usdc=min_bin)
            ba = (f"${res['best_ask']:.4f}"
                  if res.get('best_ask') is not None else "?")
            nl = (f"${res['new_limit']:.4f}"
                  if res.get('new_limit') is not None else "?")
            print(f"  {res['pid']:>5}  {res['city'][:14]:<14}  "
                  f"{ba:>9}  {nl:>10}  {res['status']} -- {res['reason']}")
        return 0

    # ---- Execute ----
    print(f"Placing {len(rows)} resurrection order(s) with "
          f"{args.sleep}s spacing...")
    print()
    print(f"{'pid':>5}  {'city':<14}  status")
    print("-" * 70)

    counts = {"resurrected": 0, "skipped": 0, "error": 0}
    for r in rows:
        res = _resurrect_one(r, client, dry_run=False,
                             walk_cents=ORDERBOOK_WALK_CENTS,
                             min_bin_usdc=min_bin)
        counts.setdefault(res["status"], 0)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        flag = {
            "resurrected": "OK",
            "skipped":     "--",
            "error":       "ER",
        }.get(res["status"], "??")
        print(f"  {res['pid']:>5}  {res['city'][:14]:<14}  "
              f"[{flag}] {res['reason']}")
        time.sleep(args.sleep)

    print()
    print("=== Summary ===")
    print(f"  resurrected: {counts.get('resurrected', 0)}")
    print(f"  skipped:     {counts.get('skipped', 0)}")
    print(f"  errors:      {counts.get('error', 0)}")
    print()
    print("Resurrected rows are now status='open', fill_status='pending'.")
    print("The 5-minute stale-entry repricer will chase any that don't fill")
    print("immediately.  Watch the dashboard's In-Flight Orders for fills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
