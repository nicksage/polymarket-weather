"""
diagnose_share_drift.py — Inspect each share-drifted position and classify
whether `positions.shares` can be safely repaired from the position_orders
ledger, or whether the drift requires a chain-direct fix (riskier).

Read-only.  No mutation.  Run with:
    cd bot
    python -m scripts.diagnose_share_drift

For each position that has a recent `share_drift` activity_log event:
  * db_shares       — what positions.shares currently says (the broken value)
  * chain_size      — what Polymarket's Data API said at last drift detection
  * ledger_filled   — sum of position_orders.filled_shares for this position
  * Classification  — one of:
      LEDGER_MATCHES_CHAIN  ledger ≈ chain → repair: positions.shares := ledger
      LEDGER_MATCHES_DB     ledger ≈ db    → ledger was backfilled from the
                                              broken DB; chain-direct repair
                                              would be needed (manual review)
      LEDGER_BETWEEN        ledger between db and chain — partial recovery only
      NO_LEDGER             no position_orders rows — can't analyze
      OTHER                 ledger disagrees with both — manual review

The classification answers the gating question before we even consider a
write: "do we have a second source of truth that agrees with chain?"
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

# Allow running as `python -m scripts.diagnose_share_drift` from bot/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


# Tolerance: same as monitor's drift threshold (monitor.py:781).
TOLERANCE = 0.5


def _latest_drift_per_position() -> dict[int, dict]:
    """Walk activity_log newest-first, return the most recent share_drift
    metadata per position_id."""
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
        meta = {}
        try:
            meta = json.loads(r["metadata"] or "{}")
        except Exception:
            pass
        if meta.get("drift_kind") != "share_drift":
            continue
        out[pid] = {
            "timestamp":  r["timestamp"],
            "message":    r["message"],
            "chain_size": float(meta.get("chain_size") or 0),
            "db_shares":  float(meta.get("db_shares") or 0),
            "delta":      float(meta.get("delta") or 0),
            "token_id":   meta.get("token_id"),
        }
    return out


def _ledger_sums(position_ids: list[int]) -> dict[int, dict]:
    """For each pos, sum filled_shares across all position_orders rows
    plus a per-role breakdown."""
    if not position_ids:
        return {}
    placeholders = ",".join("?" for _ in position_ids)
    out: dict[int, dict] = defaultdict(
        lambda: {"total": 0.0, "by_role": defaultdict(float), "rows": []}
    )
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT position_id, role, status, trade_status,
                   intended_usdc, intended_shares,
                   filled_shares, filled_usdc, fill_price,
                   order_id, created_at
            FROM position_orders
            WHERE position_id IN ({placeholders})
            ORDER BY position_id, created_at
        """, position_ids).fetchall()
    for r in rows:
        pid = r["position_id"]
        fs = float(r["filled_shares"] or 0)
        out[pid]["total"] += fs
        out[pid]["by_role"][r["role"]] += fs
        out[pid]["rows"].append(dict(r))
    return out


def _current_position_rows(position_ids: list[int]) -> dict[int, dict]:
    if not position_ids:
        return {}
    placeholders = ",".join("?" for _ in position_ids)
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT id, city, date, side, shares,
                   size_usdc, target_size_usdc,
                   fill_status, status, entry_price
            FROM positions
            WHERE id IN ({placeholders})
        """, position_ids).fetchall()
    return {r["id"]: dict(r) for r in rows}


def classify(db_shares: float, chain_size: float, ledger: float | None) -> str:
    if ledger is None or ledger == 0:
        return "NO_LEDGER" if (ledger is None) else "LEDGER_MATCHES_DB"
    if abs(ledger - chain_size) <= TOLERANCE:
        return "LEDGER_MATCHES_CHAIN"
    if abs(ledger - db_shares) <= TOLERANCE:
        return "LEDGER_MATCHES_DB"
    if db_shares - TOLERANCE <= ledger <= chain_size + TOLERANCE:
        return "LEDGER_BETWEEN"
    return "OTHER"


def main() -> int:
    drifts = _latest_drift_per_position()
    if not drifts:
        print("No share_drift events in activity_log.  Nothing to diagnose.")
        return 0

    pids = sorted(drifts.keys())
    pos_rows = _current_position_rows(pids)
    ledger = _ledger_sums(pids)

    rows: list[dict] = []
    for pid in pids:
        d = drifts[pid]
        p = pos_rows.get(pid, {})
        l = ledger.get(pid)
        ledger_total = float(l["total"]) if l else None
        # Use the *current* db_shares from positions, not the snapshot from
        # the drift event — drift may have been logged earlier and the bot
        # may have processed more fills since.
        db_now = float(p.get("shares") or d["db_shares"])
        cls = classify(db_now, d["chain_size"], ledger_total)
        rows.append({
            "pid":         pid,
            "city":        p.get("city", "?"),
            "date":        p.get("date", "?"),
            "side":        p.get("side", "?"),
            "status":      p.get("status", "?"),
            "fill_status": p.get("fill_status", "?"),
            "db_now":      db_now,
            "drift_chain": d["chain_size"],
            "drift_db":    d["db_shares"],
            "ledger":      ledger_total,
            "by_role":     dict(l["by_role"]) if l else {},
            "n_orders":    len(l["rows"]) if l else 0,
            "delta_now":   d["chain_size"] - db_now,
            "class":       cls,
            "ts":          d["timestamp"],
        })

    # ---- Summary table ----
    print()
    print(f"Share-drift diagnostic — {len(rows)} drifted position(s)")
    print(f"Tolerance: ±{TOLERANCE} shares")
    print()
    hdr = (
        f"{'pid':>4}  {'city':<14} {'date':<10} {'side':<3}  "
        f"{'db_now':>8}  {'chain':>8}  {'ledger':>8}  "
        f"{'orders':>6}  {'class':<22}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ledger_str = f"{r['ledger']:.2f}" if r["ledger"] is not None else "  (none)"
        print(
            f"{r['pid']:>4}  {str(r['city'])[:14]:<14} {str(r['date'])[:10]:<10} "
            f"{str(r['side'])[:3]:<3}  "
            f"{r['db_now']:>8.2f}  {r['drift_chain']:>8.2f}  {ledger_str:>8}  "
            f"{r['n_orders']:>6}  {r['class']:<22}"
        )

    # ---- Per-class summary ----
    print()
    print("Counts by classification:")
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["class"]] += 1
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<22} {n}")

    # ---- Per-position role breakdown for the interesting cases ----
    interesting = [r for r in rows if r["class"] != "LEDGER_MATCHES_CHAIN"]
    if interesting:
        print()
        print("Role breakdown for non-trivially-recoverable positions:")
        for r in interesting:
            print(
                f"  pid={r['pid']} {r['city']:<14} {r['side']}  "
                f"orders={r['n_orders']}  by_role={r['by_role']}  "
                f"db_now={r['db_now']:.2f}  chain={r['drift_chain']:.2f}  "
                f"ledger={r['ledger']}"
            )

    # ---- Repair plan summary ----
    print()
    print("Repair plan implication:")
    safe = counts.get("LEDGER_MATCHES_CHAIN", 0)
    backfilled = counts.get("LEDGER_MATCHES_DB", 0)
    other = sum(n for c, n in counts.items()
                if c not in ("LEDGER_MATCHES_CHAIN", "LEDGER_MATCHES_DB"))
    print(f"  {safe} position(s) can be repaired safely (ledger == chain).")
    print(f"  {backfilled} position(s) only have backfilled ledger rows that")
    print(f"     mirror the broken db_shares — chain-direct repair only.")
    print(f"  {other} position(s) need manual review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
