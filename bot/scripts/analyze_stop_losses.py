"""
analyze_stop_losses.py — Post-trade analysis of stop-loss outcomes.

For every closed live position whose event has resolved (the winning bin
is detectable in temp_outcomes via yes_price >= 0.99), this script
answers:

    1. Did our bin actually win?
    2. If we exited via stop/trail, was that exit "premature"
       (the bin would have won if held)?
    3. How deep did the price dip before recovering (informs stop-loss
       tuning — a stop tighter than the median dip clips winners; a
       stop looser than the median loss leaks money on losers)?
    4. Counterfactual: what would we have made if we'd held to resolution?

Run:
    cd bot
    python -m scripts.analyze_stop_losses                     # all closed live
    python -m scripts.analyze_stop_losses --csv out.csv       # also dump CSV
    python -m scripts.analyze_stop_losses --reason HARD_STOP  # filter exit type

Limitations:
    * Only positions whose EVENT has resolved are analyzed (the winning
      bin needs to have hit yes_price >= 0.99 in some scan).  As more
      events resolve, the analyzable set grows.
    * Dip depth uses the MIN(yes_price) across all scans between entry
      and exit/resolution — limited by scan frequency (~every 15 min).
      Sub-cycle dips between scans aren't visible.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


# Exit reasons that count as "stop-style" — i.e., the bot got out
# proactively and we want to know if that was the right call.  Anything
# else (settled, manual, profit_target) we leave out of the
# "premature?" classification.
STOP_REASONS = ("HARD_STOP", "RT_HARD_STOP", "TRAILING_STOP", "DYING")


def _classify_reason(reason: str) -> str:
    """Map a verbose exit_reason string into a coarse bucket."""
    if not reason:
        return "unknown"
    r = reason.upper()
    for k in STOP_REASONS:
        if k in r:
            return k
    if "PROFIT" in r or "TAKE_PROFIT" in r:
        return "PROFIT"
    if "BALANCE_MISMATCH" in r or "BLEED" in r:
        return "RECOVERY"
    if "WEAKENED" in r:
        return "WEAKENED"
    return "OTHER"


def _winning_bin_for_event(conn, event_id: str) -> dict | None:
    """Find the bin of `event_id` whose yes_price hit >= 0.99 in any scan
    (i.e., the resolution winner).  Returns the contract_id + range, or
    None if no resolution detected yet."""
    row = conn.execute("""
        SELECT o.contract_id, o.range_low, o.range_high, o.unit,
               MAX(o.yes_price) AS final_yes,
               MAX(o.scan_timestamp) AS resolution_ts
        FROM temp_outcomes o
        JOIN temp_events e ON o.event_row_id = e.id
        WHERE e.event_id = ?
        GROUP BY o.contract_id
        HAVING final_yes >= 0.99
        ORDER BY resolution_ts DESC
        LIMIT 1
    """, (event_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def _bin_label(range_low, range_high, unit) -> str:
    if range_low is None and range_high is None:
        return "?"
    suf = "F" if (unit or "celsius") == "fahrenheit" else "C"
    if range_low == range_high:
        return f"{int(range_low)}{suf}"
    if range_low is None:
        return f"<={int(range_high)}{suf}"
    if range_high is None:
        return f">={int(range_low)}{suf}"
    return f"{int(range_low)}-{int(range_high)}{suf}"


def _dip_stats(conn, contract_id: str, t_start: str, t_end: str) -> dict:
    """Min/max yes_price for `contract_id` between two timestamps.
    `t_end` is inclusive — usually entry_time → exit_time, OR
    entry_time → resolution_time for the counterfactual hold case."""
    row = conn.execute("""
        SELECT MIN(yes_price) AS min_yes,
               MAX(yes_price) AS max_yes,
               COUNT(*) AS n_scans
        FROM temp_outcomes
        WHERE contract_id = ?
          AND scan_timestamp BETWEEN ? AND ?
          AND yes_price IS NOT NULL
    """, (contract_id, t_start, t_end)).fetchone()
    if row is None:
        return {"min_yes": None, "max_yes": None, "n_scans": 0}
    return dict(row)


def _resolution_ts_for_event(conn, event_id: str) -> Optional[str]:
    """Find the earliest scan timestamp where any bin of this event hit
    yes_price >= 0.99 — proxy for "when the event resolved."""
    row = conn.execute("""
        SELECT MIN(o.scan_timestamp) AS resolved_at
        FROM temp_outcomes o
        JOIN temp_events e ON o.event_row_id = e.id
        WHERE e.event_id = ?
          AND o.yes_price >= 0.99
    """, (event_id,)).fetchone()
    if row is None:
        return None
    return row["resolved_at"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reason", help="Filter to exit reasons containing this substring (case-insensitive)")
    ap.add_argument("--csv", help="Write per-trade rows to this CSV file")
    ap.add_argument("--include-paper", action="store_true",
                    help="Include paper-trade positions (default: live only)")
    args = ap.parse_args()

    with _get_conn() as conn:
        # Pull every closed position with the data we need
        sql = """
            SELECT id, contract_id, event_id, city, date, side,
                   range_low, range_high, unit,
                   shares, entry_price, actual_exit_price,
                   pnl, pnl_net, entry_fees, exit_fees,
                   exit_reason, exit_time, entry_time,
                   size_usdc, target_size_usdc, is_paper
            FROM positions
            WHERE status = 'closed'
        """
        if not args.include_paper:
            sql += " AND COALESCE(is_paper, 0) = 0"
        sql += " ORDER BY entry_time"
        positions = [dict(r) for r in conn.execute(sql).fetchall()]

        analyzed: list[dict] = []
        pending: list[dict] = []

        for p in positions:
            evid = p.get("event_id") or ""
            if not evid:
                continue   # legacy row, no event_id

            winner = _winning_bin_for_event(conn, evid)
            if winner is None:
                pending.append(p)
                continue

            our_bin_won = (winner["contract_id"] == p["contract_id"])
            entry_t = p.get("entry_time") or ""
            exit_t  = p.get("exit_time")  or ""
            resolution_t = _resolution_ts_for_event(conn, evid) or exit_t

            # Dip stats during HOLDING window
            hold_dip = _dip_stats(conn, p["contract_id"], entry_t, exit_t)
            # Dip stats from entry → resolution (would-have-held window)
            full_dip = _dip_stats(conn, p["contract_id"], entry_t, resolution_t)

            # Counterfactual: hold to resolution
            shares = float(p.get("shares") or 0)
            entry_price = float(p.get("entry_price") or 0)
            entry_fee = float(p.get("entry_fees") or 0)
            cost = shares * entry_price + entry_fee
            if our_bin_won:
                counterfactual_payout = shares * 1.0
            else:
                counterfactual_payout = 0.0
            counterfactual_pnl = round(counterfactual_payout - cost, 4)

            actual_pnl_net = p.get("pnl_net")
            if actual_pnl_net is None:
                actual_pnl_net = float(p.get("pnl") or 0) - entry_fee - float(p.get("exit_fees") or 0)

            cls = _classify_reason(p.get("exit_reason") or "")
            row = {
                "pid":          p["id"],
                "city":         p.get("city", ""),
                "date":         p.get("date", ""),
                "side":         p.get("side", ""),
                "our_bin":      _bin_label(p.get("range_low"), p.get("range_high"), p.get("unit")),
                "winning_bin":  _bin_label(winner.get("range_low"), winner.get("range_high"), winner.get("unit")),
                "won":          our_bin_won,
                "exit_class":   cls,
                "exit_reason":  (p.get("exit_reason") or "")[:40],
                "entry_$":      round(entry_price, 4),
                "exit_$":       round(float(p.get("actual_exit_price") or 0), 4),
                "shares":       round(shares, 2),
                "actual_pnl":   round(actual_pnl_net, 4),
                "counter_pnl":  counterfactual_pnl,
                "missed":       round(counterfactual_pnl - actual_pnl_net, 4) if our_bin_won else 0,
                "min_during_hold":   round(hold_dip["min_yes"], 4) if hold_dip["min_yes"] is not None else None,
                "min_full_window":   round(full_dip["min_yes"], 4) if full_dip["min_yes"] is not None else None,
                "dip_pct_below_entry":  round((entry_price - (full_dip["min_yes"] or 0)) / entry_price * 100, 1) if entry_price > 0 and full_dip["min_yes"] is not None else None,
            }
            if args.reason and args.reason.lower() not in cls.lower():
                continue
            analyzed.append(row)

    # ---- Per-trade table ----
    if analyzed:
        print()
        print(f"Analyzed {len(analyzed)} resolved closed position(s)"
              + (f", filtered to '{args.reason}'" if args.reason else ""))
        print()
        hdr = (
            f"{'pid':>4}  {'city':<14}  {'date':<11}  "
            f"{'our':<5}  {'win':<5}  {'won':<3}  {'exit':<14}  "
            f"{'entry':>6}  {'exit':>5}  {'actual':>8}  {'cfact':>8}  {'missed':>8}  "
            f"{'minHld':>7}  {'minWin':>7}  {'dip%':>5}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in analyzed:
            won_marker = "✓" if r["won"] else "✗"
            print(f"{r['pid']:>4}  {str(r['city'])[:14]:<14}  {str(r['date'])[:11]:<11}  "
                  f"{str(r['our_bin'])[:5]:<5}  {str(r['winning_bin'])[:5]:<5}  {won_marker:<3}  "
                  f"{r['exit_class'][:14]:<14}  "
                  f"{r['entry_$']:>6.4f}  {r['exit_$']:>5.2f}  ${r['actual_pnl']:>+7.2f}  "
                  f"${r['counter_pnl']:>+7.2f}  ${r['missed']:>+7.2f}  "
                  f"{r['min_during_hold'] or '-':>7}  {r['min_full_window'] or '-':>7}  "
                  f"{r['dip_pct_below_entry'] or '-':>5}")

    # ---- Summary ----
    print()
    print("=== Summary ===")
    print(f"  resolved positions analyzed:   {len(analyzed)}")
    print(f"  positions with pending events: {len(pending)}")
    if not analyzed:
        return 0

    n_won = sum(1 for r in analyzed if r["won"])
    print(f"  bin-won rate:                  {n_won}/{len(analyzed)}  ({n_won/len(analyzed)*100:.1f}%)")

    # Per-exit-class breakdown
    print()
    print("By exit class:")
    by_cls: dict = {}
    for r in analyzed:
        cls = r["exit_class"]
        agg = by_cls.setdefault(cls, {
            "n": 0, "won": 0, "actual_total": 0.0,
            "counter_total": 0.0, "missed_total": 0.0,
        })
        agg["n"] += 1
        if r["won"]:
            agg["won"] += 1
            agg["missed_total"] += r["missed"]
        agg["actual_total"]  += r["actual_pnl"]
        agg["counter_total"] += r["counter_pnl"]

    print(f"  {'class':<12} {'count':>6} {'win-yet-stopped':>16} {'sum actual':>11} {'sum counter':>12} {'forfeit':>10}")
    for cls in sorted(by_cls.keys()):
        a = by_cls[cls]
        prem_pct = a["won"] / a["n"] * 100 if a["n"] else 0
        print(f"  {cls:<12} {a['n']:>6} {a['won']}/{a['n']} ({prem_pct:>4.0f}%)   "
              f"${a['actual_total']:>+8.2f}   ${a['counter_total']:>+9.2f}   "
              f"${a['missed_total']:>+8.2f}")

    # Stop-style specific: dip depth distribution for winners that we stopped out of
    stop_winners = [r for r in analyzed
                    if r["exit_class"] in STOP_REASONS and r["won"]]
    if stop_winners:
        print()
        print(f"Stop-out winners ({len(stop_winners)} positions where stop fired but bin actually won):")
        print(f"  These trades 'cut' a winner.  How tight was the stop relative to the dip?")
        dips = sorted(r["dip_pct_below_entry"] for r in stop_winners
                      if r["dip_pct_below_entry"] is not None)
        if dips:
            print(f"  Min dip from entry to RESOLUTION: {min(dips):.1f}%   "
                  f"Median: {dips[len(dips)//2]:.1f}%   Max: {max(dips):.1f}%")
            print(f"  → If your stop is set tighter than {min(dips):.1f}%, you'd cut the easiest winner.")

    # Stop-style: how much did stops save us on actual losers?
    stop_losers = [r for r in analyzed
                   if r["exit_class"] in STOP_REASONS and not r["won"]]
    if stop_losers:
        print()
        print(f"Stop-out losers ({len(stop_losers)} positions where stop fired and bin lost — stop did its job):")
        # Compare actual loss vs counterfactual loss (held to $0)
        avg_saved = sum((r["counter_pnl"] - r["actual_pnl"]) for r in stop_losers) / len(stop_losers)
        print(f"  Average dollars saved by stopping out vs holding to $0 resolution: ${-avg_saved:.2f}")

    if pending:
        print()
        print(f"({len(pending)} closed positions are PENDING event resolution — re-run later)")

    if args.csv and analyzed:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=analyzed[0].keys())
            w.writeheader()
            w.writerows(analyzed)
        print()
        print(f"CSV written: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
