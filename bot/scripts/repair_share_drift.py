"""
repair_share_drift.py — One-shot repair: align positions.shares with the
on-chain holdings reported by the most recent share_drift event.

Run order:
    1. python -m scripts.diagnose_share_drift           (read-only)
    2. python -m scripts.repair_share_drift             (dry run, default)
    3. python -m scripts.repair_share_drift --apply     (commit)

Gates — a position is repaired only when ALL of these hold:
    * is_paper       == 0          (live, never paper)
    * fill_status    == 'filled'   (skip pending / cancelled)
    * status         == 'open'     (skip 'exiting' AND 'closed' — both
                                    indicate the chain holding has
                                    changed since the drift event was
                                    logged, so chain_size is stale)
    * latest activity_log share_drift event exists
    * chain_size > 0
    * (chain_size - db_shares) >= TOLERANCE  (only repair upward, never
      shrink; protects against a stale drift event where shares have
      since been corrected by other means)

For every repair, we:
    * UPDATE positions SET shares = chain_size WHERE id = ?
    * Append an activity_log row at level=INFO category='REPAIR' with
      metadata {old_shares, new_shares, chain_size, drift_event_ts,
      repair_kind} so the change is auditable and recoverable.

Idempotent — re-running after a successful apply is a no-op (the delta
gate fails because db_shares now equals chain_size).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Allow running as `python -m scripts.repair_share_drift` from bot/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore
from activity import log_activity  # type: ignore


TOLERANCE = 0.5  # mirrors monitor.py:_RECONCILE_SHARE_TOLERANCE


def _latest_drift_per_position() -> dict[int, dict]:
    """Newest share_drift event per position, parsed from activity_log
    metadata.  Same shape as the diagnostic script."""
    out: dict[int, dict] = {}
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT position_id, timestamp, message, metadata
            FROM activity_log
            WHERE category = 'DRIFT' AND position_id IS NOT NULL
            ORDER BY timestamp DESC
        """).fetchall()
    for r in rows:
        pid = r["position_id"]
        if pid in out:
            continue
        try:
            meta = json.loads(r["metadata"] or "{}")
        except Exception:
            continue
        if meta.get("drift_kind") != "share_drift":
            continue
        out[pid] = {
            "timestamp":  r["timestamp"],
            "chain_size": float(meta.get("chain_size") or 0),
            "db_shares":  float(meta.get("db_shares") or 0),
            "delta":      float(meta.get("delta") or 0),
            "token_id":   meta.get("token_id"),
        }
    return out


def _position_state(position_ids: list[int]) -> dict[int, dict]:
    if not position_ids:
        return {}
    placeholders = ",".join("?" for _ in position_ids)
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT id, city, date, side, shares, fill_status, status, is_paper
            FROM positions
            WHERE id IN ({placeholders})
        """, position_ids).fetchall()
    return {r["id"]: dict(r) for r in rows}


def _gate(pos: dict, drift: dict) -> tuple[bool, str]:
    """Return (eligible, reason).  Reason is human-readable when not eligible."""
    if int(pos.get("is_paper") or 0) != 0:
        return False, "paper"
    fs = (pos.get("fill_status") or "").lower()
    if fs != "filled":
        return False, f"fill_status={fs!r}"
    st = (pos.get("status") or "").lower()
    if st != "open":
        # 'exiting' = ladder in flight, chain holding actively changing.
        # 'closed' = position resolved/sold, drift event's chain_size is stale.
        return False, f"status={st!r} (only 'open' is safe — chain_size may be stale)"
    chain = float(drift.get("chain_size") or 0)
    if chain <= 0:
        return False, f"chain_size={chain:.2f}"
    db_now = float(pos.get("shares") or 0)
    delta = chain - db_now
    if delta < TOLERANCE:
        return False, f"delta={delta:+.2f} below tolerance ({TOLERANCE}) — already in sync"
    return True, "ok"


def _recompute_cost_basis_from_chain(
    pid: int, client,
) -> tuple[float | None, float | None]:
    """Recompute (entry_price_avg, size_usdc_total) from on-chain trades.

    Walks the position's order_ids (entry + topup roles only -- never
    exit, which would offset cost basis) and sums Σ(size × price) and
    Σ(size) across ALL matched trades for those orders.  Returns
    (None, None) on any failure -- caller falls back to leaving the
    cost-basis columns untouched.

    Why this is needed: pre-fix, repair_share_drift only synced the
    `shares` column.  After repair, a position's `shares` matched chain
    truth but `entry_price` and `size_usdc` were stale (still seeded
    from the original placement).  Result: shares × entry_price ≠
    size_usdc, breaking P&L calc and dashboard "Traded $" display.
    """
    if client is None:
        return None, None
    try:
        from py_clob_client_v2.clob_types import TradeParams
    except Exception:
        return None, None

    # Pull the position's market + token + ledger orders.
    with _get_conn() as conn:
        prow = conn.execute(
            "SELECT contract_id, side, yes_token_id, no_token_id "
            "FROM positions WHERE id = ?", (pid,),
        ).fetchone()
        if prow is None:
            return None, None
        ledger = [dict(r) for r in conn.execute(
            "SELECT order_id, role FROM position_orders "
            "WHERE position_id = ? AND role IN ('entry', 'topup')",
            (pid,),
        ).fetchall()]
    market   = prow["contract_id"]
    side     = prow["side"] or "YES"
    token_id = prow["yes_token_id"] if side == "YES" else prow["no_token_id"]
    if not market or not token_id:
        return None, None

    # If the ledger has no entry/topup orders, fall back to the legacy
    # `order_id` column on the row itself.
    our_order_ids: set[str] = {
        r["order_id"] for r in ledger if r.get("order_id")
    }
    if not our_order_ids:
        with _get_conn() as conn:
            legacy = conn.execute(
                "SELECT order_id FROM positions WHERE id = ?", (pid,),
            ).fetchone()
        if legacy and legacy["order_id"]:
            our_order_ids.add(legacy["order_id"])
    if not our_order_ids:
        return None, None

    try:
        trades = client.get_trades(
            params=TradeParams(market=market, asset_id=token_id),
            only_first_page=False,
        ) or []
    except Exception:
        return None, None

    total_shares = 0.0
    total_usdc   = 0.0
    for t in trades:
        # Match: taker_order_id is one of ours OR any maker_orders[].order_id is.
        taker_oid = t.get("taker_order_id") or ""
        is_match = (taker_oid in our_order_ids)
        if not is_match:
            for mo in (t.get("maker_orders") or []):
                if mo.get("order_id") in our_order_ids:
                    is_match = True
                    break
        if not is_match:
            continue
        try:
            sz    = float(t.get("size") or 0)
            price = float(t.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if sz <= 0 or price <= 0:
            continue
        total_shares += sz
        total_usdc   += sz * price

    if total_shares <= 0:
        return None, None
    avg_price = total_usdc / total_shares
    return round(avg_price, 6), round(total_usdc, 4)


def _apply_repair(pid: int, old_shares: float, new_shares: float,
                  drift: dict, client=None,
                  cost_basis_only: bool = False) -> dict:
    """Apply share-count repair AND (when client available) recompute
    cost basis from on-chain trades.  Returns a result dict.

    When cost_basis_only=True, leaves the `shares` column untouched
    (used by --rebuild-basis mode, where shares are already in sync
    and the drift event's chain_size may be stale).
    """
    avg_price, size_usdc = _recompute_cost_basis_from_chain(pid, client)

    with _get_conn() as conn:
        if cost_basis_only:
            # Only fix cost basis; trust that shares are correct.
            if avg_price is not None and size_usdc is not None:
                conn.execute(
                    "UPDATE positions "
                    "SET entry_price = ?, size_usdc = ? "
                    "WHERE id = ?",
                    (avg_price, size_usdc, pid),
                )
            # If we couldn't fetch trades, do nothing -- shares are
            # already in sync, no harm done by leaving cost basis stale.
        elif avg_price is not None and size_usdc is not None:
            # Full repair: shares + cost basis
            conn.execute(
                "UPDATE positions "
                "SET shares = ?, entry_price = ?, size_usdc = ? "
                "WHERE id = ?",
                (new_shares, avg_price, size_usdc, pid),
            )
        else:
            # Fallback: shares only (matches legacy behaviour)
            conn.execute(
                "UPDATE positions SET shares = ? WHERE id = ?",
                (new_shares, pid),
            )

    metadata = {
        "repair_kind":     ("cost_basis_only" if cost_basis_only
                            else "share_drift_chain_match"),
        "old_shares":      old_shares,
        "new_shares":      old_shares if cost_basis_only else new_shares,
        "chain_size":      float(drift.get("chain_size") or 0),
        "drift_event_ts":  drift.get("timestamp"),
        "token_id":        drift.get("token_id"),
    }
    if cost_basis_only:
        if avg_price is not None and size_usdc is not None:
            msg = (
                f"cost basis recomputed from chain trades: "
                f"entry_price -> ${avg_price:.4f}, "
                f"size_usdc -> ${size_usdc:.2f}  "
                f"(shares unchanged at {old_shares:.4f})"
            )
            metadata["new_entry_price"] = avg_price
            metadata["new_size_usdc"]   = size_usdc
        else:
            msg = (
                f"cost-basis recompute SKIPPED -- "
                f"no matching trades found on chain (shares unchanged at "
                f"{old_shares:.4f})"
            )
    elif avg_price is not None and size_usdc is not None:
        msg = (
            f"shares {old_shares:.4f} -> {new_shares:.4f}, "
            f"entry_price -> ${avg_price:.4f}, "
            f"size_usdc -> ${size_usdc:.2f} (chain reconcile)"
        )
        metadata["new_entry_price"] = avg_price
        metadata["new_size_usdc"]   = size_usdc
    else:
        msg = (
            f"shares {old_shares:.4f} -> {new_shares:.4f} "
            f"(chain reconcile; cost basis NOT recomputed -- "
            f"client unavailable or no trades found)"
        )

    log_activity("REPAIR", msg, level="INFO", position_id=pid, **metadata)
    return {
        "pid":         pid,
        "new_shares":  old_shares if cost_basis_only else new_shares,
        "new_avg":     avg_price,
        "new_usdc":    size_usdc,
        "cost_basis_only": cost_basis_only,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write the repairs.  Without this flag the script "
             "prints the plan and exits (dry run).",
    )
    ap.add_argument(
        "--rebuild-basis", action="store_true",
        help="Recompute entry_price + size_usdc from on-chain trade history "
             "for every position with a share_drift event, even if shares "
             "are already in sync.  Use this AFTER the legacy shares-only "
             "repair to fix the stale cost-basis columns.",
    )
    args = ap.parse_args()

    drifts = _latest_drift_per_position()
    if not drifts:
        print("No share_drift events in activity_log.  Nothing to do.")
        return 0

    pids = sorted(drifts.keys())
    pos_rows = _position_state(pids)

    plan: list[tuple[dict, dict]] = []
    skipped: list[tuple[int, str]] = []

    for pid in pids:
        pos = pos_rows.get(pid)
        if not pos:
            skipped.append((pid, "position not found in DB"))
            continue
        ok, reason = _gate(pos, drifts[pid])
        if ok:
            plan.append((pos, drifts[pid]))
            continue
        # --rebuild-basis bypasses the share-delta gate: even rows that
        # are already in sync on `shares` get their cost basis recomputed
        # from chain trades.  Still respects the safety gates (no paper,
        # no closed/exiting, must be filled) -- those remain enforced.
        if args.rebuild_basis and "below tolerance" in reason:
            fs = (pos.get("fill_status") or "").lower()
            st = (pos.get("status") or "").lower()
            if (int(pos.get("is_paper") or 0) == 0
                    and fs == "filled"
                    and st == "open"):
                plan.append((pos, drifts[pid]))
                continue
        skipped.append((pid, reason))

    mode = "APPLY" if args.apply else "DRY RUN"
    print()
    print(f"Repair plan (mode: {mode})")
    print(f"Tolerance: ±{TOLERANCE} shares")
    print()

    if plan:
        print(f"  {'pid':>4}  {'city':<14}  {'date':<10}  {'side':<3}  "
              f"{'old':>8}  {'new':>8}  {'delta':>8}")
        for pos, drift in plan:
            old = float(pos.get("shares") or 0)
            new = float(drift["chain_size"])
            print(f"  {pos['id']:>4}  {str(pos.get('city',''))[:14]:<14}  "
                  f"{str(pos.get('date',''))[:10]:<10}  "
                  f"{str(pos.get('side',''))[:3]:<3}  "
                  f"{old:>8.2f}  {new:>8.2f}  {new - old:>+8.2f}")
    else:
        print("  (no positions eligible for repair)")

    if skipped:
        print()
        print("Skipped:")
        for pid, reason in skipped:
            print(f"  pid={pid:<4}  reason: {reason}")

    if args.apply and plan:
        # Acquire CLOB client up-front so cost-basis recompute can fire
        # for every row.  If unavailable (paper / missing creds), the
        # repair falls back to shares-only (legacy behaviour) per row.
        client = None
        try:
            from execution import get_clob_client
            client = get_clob_client()
        except Exception as e:
            print(f"  WARN: could not acquire CLOB client ({e}); "
                  f"cost basis will NOT be recomputed")

        print()
        print(f"Applying {len(plan)} repair(s)"
              + (" (cost basis WILL be recomputed)" if client else
                 " (cost basis recompute SKIPPED -- no client)")
              + "...")
        n_basis_recomputed = 0
        for pos, drift in plan:
            old = float(pos.get("shares") or 0)
            new = float(drift["chain_size"])
            # In --rebuild-basis mode, ALWAYS cost-basis-only.  We never
            # touch `shares` -- the drift event's chain_size may be stale
            # (more fills may have landed since), and the operator has
            # explicitly opted into "cost basis fix only" by passing the
            # flag.  Without this, a -11 share delta would silently
            # SHRINK the position to a long-stale chain snapshot.
            cost_basis_only = bool(args.rebuild_basis)
            res = _apply_repair(
                pos["id"], old, new, drift, client=client,
                cost_basis_only=cost_basis_only,
            )
            basis_msg = ""
            if res.get("new_avg") is not None:
                n_basis_recomputed += 1
                basis_msg = (f"entry=${res['new_avg']:.4f}, "
                             f"size=${res['new_usdc']:.2f}")
            if cost_basis_only:
                print(f"  pid={pos['id']:<4}  [basis-only] {basis_msg}  "
                      f"({pos.get('city','?')})")
            else:
                share_msg = f"shares: {old:.4f} -> {new:.4f}"
                if basis_msg:
                    print(f"  pid={pos['id']:<4}  {share_msg}, {basis_msg}  "
                          f"({pos.get('city','?')})")
                else:
                    print(f"  pid={pos['id']:<4}  {share_msg}  "
                          f"({pos.get('city','?')})")
        print()
        print(f"Done.  {len(plan)} row(s) updated"
              + (f" ({n_basis_recomputed} with full cost-basis recompute)"
                 if n_basis_recomputed else "")
              + ".  See activity_log category='REPAIR' for audit trail.")
    elif plan:
        print()
        print("Dry run — no changes written.  Re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
