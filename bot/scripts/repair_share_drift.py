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


def _apply_repair(pid: int, old_shares: float, new_shares: float, drift: dict) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE positions SET shares = ? WHERE id = ?",
            (new_shares, pid),
        )
    log_activity(
        "REPAIR",
        f"shares {old_shares:.4f} -> {new_shares:.4f} (chain reconcile)",
        level="INFO",
        position_id=pid,
        repair_kind="share_drift_chain_match",
        old_shares=old_shares,
        new_shares=new_shares,
        chain_size=float(drift.get("chain_size") or 0),
        drift_event_ts=drift.get("timestamp"),
        token_id=drift.get("token_id"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write the repairs.  Without this flag the script "
             "prints the plan and exits (dry run).",
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
        if not ok:
            skipped.append((pid, reason))
            continue
        plan.append((pos, drifts[pid]))

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
        print()
        print(f"Applying {len(plan)} repair(s)...")
        for pos, drift in plan:
            old = float(pos.get("shares") or 0)
            new = float(drift["chain_size"])
            _apply_repair(pos["id"], old, new, drift)
            print(f"  pid={pos['id']:<4}  shares: {old:.4f} -> {new:.4f}  "
                  f"({pos.get('city','?')})")
        print()
        print(f"Done.  {len(plan)} row(s) updated.  See activity_log "
              f"category='REPAIR' for audit trail.")
    elif plan:
        print()
        print("Dry run — no changes written.  Re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
