"""
rank_win_rate.py — Did the model rank the bins correctly within each event?

The TKH strategy ranks bins by model probability (descending) and allocates
60/25/15 of the $20 budget to ranks 0/1/2.  That split is profitable IFF
rank-0 actually wins ~55% of resolved events.  This script measures the
real distribution by looking at which bin (by target_size_usdc) won every
event the bot resolved, and reports:

  * basket hit rate     = events where bot held the winning bin / total
  * conditional rank-N  = given the winner was held, did rank-0/1/2 win?
  * P&L by winner-rank  = how much each rank-of-winner pays on average

Interpretation:
  * If rank-0 wins ~55% → ranking is well-calibrated, TKH math holds
  * If rank-0 wins <50% AND rank-1/2 win more → MODEL'S CONFIDENCE
    ORDERING IS WRONG.  The right basket is being chosen but the wrong
    bin within the basket is being sized largest.  Fix by improving the
    model's probability calibration, OR flatten the 60/25/15 split toward
    33/33/33 to reduce the cost of mis-ranking.

Limitations:
  * Only events where the bot HELD the winner can be rank-attributed.
    Events the bot missed entirely are tallied as "basket miss" — we
    can't tell what rank the actual winner would have had in the model's
    eyes without persisting the full per-event ranking.
  * Winner detection uses the same heuristic as pnl_by_day.py:
    exit_price >= 0.97 or 'TAKE_PROFIT' in exit_reason.

Usage:
    cd bot
    python -m scripts.rank_win_rate                       # all-time
    python -m scripts.rank_win_rate --range 2026-04-25 2026-05-07
    python -m scripts.rank_win_rate --since-days 14
    python -m scripts.rank_win_rate --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date as _date, datetime, timedelta

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore

# TKH design assumption: rank-0 should win this fraction of events for the
# 60/25/15 split to be profitable in expectation.  Sourced from
# strategies/top_k_hedged.py — adjust if the split changes.
DESIGN_RANK_WINRATE = {0: 0.55, 1: 0.25, 2: 0.15}


def _is_winning(exit_price, exit_reason, side) -> bool:
    if exit_price is None:
        return False
    side_u = (side or "YES").upper()
    if side_u == "YES" and exit_price >= 0.97:
        return True
    if side_u == "NO" and exit_price <= 0.03:
        return True
    if exit_reason and "TAKE_PROFIT" in exit_reason:
        return True
    return False


def _fetch_positions(conn, dates: list[str] | None, include_paper: bool) -> list[dict]:
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"
    if dates:
        placeholders = ",".join("?" * len(dates))
        sql = f"""
            SELECT id, contract_id, event_id, city, date, side, status,
                   shares, size_usdc, entry_price,
                   exit_price, pnl, exit_reason,
                   target_size_usdc, model_prob, market_prob, edge,
                   range_low, range_high, unit
            FROM positions
            WHERE strategy = 'top_k_hedged'
              AND date IN ({placeholders})
              {paper_clause}
        """
        rows = conn.execute(sql, tuple(dates)).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT id, contract_id, event_id, city, date, side, status,
                   shares, size_usdc, entry_price,
                   exit_price, pnl, exit_reason,
                   target_size_usdc, model_prob, market_prob, edge,
                   range_low, range_high, unit
            FROM positions
            WHERE strategy = 'top_k_hedged'
              {paper_clause}
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _group_by_event(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group positions into events keyed by (city, date, event_id)."""
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["city"] or "?", r["date"] or "?",
               r["event_id"] or f"_{r['city']}_{r['date']}")
        out[key].append(r)
    return out


def _bin_label(rl, rh, unit) -> str:
    suffix = "F" if (unit or "celsius").lower() == "fahrenheit" else "C"
    if rl is None and rh is None:
        return "?"
    if rl is None:
        return f"<{int(rh)}{suffix}"
    if rh is None:
        return f">{int(rl)}{suffix}"
    if int(rl) == int(rh):
        return f"{int(rl)}{suffix}"
    return f"{int(rl)}-{int(rh)}{suffix}"


def _rank_event(positions: list[dict]) -> dict:
    """For one event's positions, sort by target_size_usdc DESC (rank order
    used at entry time) and find the winner's rank.  Returns a dict
    describing the event's outcome + per-rank P&L."""
    # Resolved-only check: if any position has status='open', the event
    # is still pending — skip rank attribution.
    if any(p["status"] == "open" for p in positions):
        return {"resolved": False, "any_winner": False}

    # Sort by target_size_usdc desc; tie-break by model_prob desc.  Ties
    # at the same target (e.g., from fill_missing_bins) share a rank but
    # we still want a deterministic order for the winner-lookup.
    ranked = sorted(
        positions,
        key=lambda p: (
            -float(p["target_size_usdc"] or 0),
            -float(p["model_prob"] or 0),
            int(p["id"]),
        ),
    )

    winner_idx = None
    for i, p in enumerate(ranked):
        if _is_winning(p["exit_price"], p["exit_reason"], p["side"]):
            winner_idx = i
            break

    realized_total = sum(float(p["pnl"] or 0) for p in positions)
    cost_total     = sum(float(p["size_usdc"] or 0) for p in positions)

    bins = []
    for i, p in enumerate(ranked):
        bins.append({
            "rank":       i,
            "label":      _bin_label(p["range_low"], p["range_high"], p["unit"]),
            "target":     float(p["target_size_usdc"] or 0),
            "size_usdc":  float(p["size_usdc"] or 0),
            "model_prob": float(p["model_prob"] or 0),
            "market_prob": float(p["market_prob"] or 0),
            "entry_price": float(p["entry_price"] or 0),
            "exit_price":  float(p["exit_price"] or 0) if p["exit_price"] else None,
            "pnl":        float(p["pnl"] or 0),
            "is_winner":  i == winner_idx,
        })

    return {
        "resolved":      True,
        "any_winner":    winner_idx is not None,
        "winner_rank":   winner_idx,
        "n_bins":        len(positions),
        "cost_total":    round(cost_total, 4),
        "realized_total": round(realized_total, 4),
        "bins":          bins,
    }


def compute(dates: list[str] | None, include_paper: bool = False) -> dict:
    with _get_conn() as conn:
        rows = _fetch_positions(conn, dates, include_paper)
    events = _group_by_event(rows)

    by_rank_count: dict[int, int] = defaultdict(int)   # winner_rank -> n events
    by_rank_pnl:   dict[int, float] = defaultdict(float)
    by_rank_cost:  dict[int, float] = defaultdict(float)

    miss_count = 0
    miss_pnl   = 0.0
    miss_cost  = 0.0

    pending_count = 0
    resolved_count = 0
    event_details: list[dict] = []

    for key, positions in events.items():
        out = _rank_event(positions)
        if not out["resolved"]:
            pending_count += 1
            continue

        resolved_count += 1
        ed = {
            "city":     key[0],
            "date":     key[1],
            "event_id": key[2],
            **out,
        }
        event_details.append(ed)

        if out["any_winner"]:
            r = out["winner_rank"]
            by_rank_count[r] += 1
            by_rank_pnl[r]   += out["realized_total"]
            by_rank_cost[r]  += out["cost_total"]
        else:
            miss_count += 1
            miss_pnl   += out["realized_total"]
            miss_cost  += out["cost_total"]

    held_count = sum(by_rank_count.values())
    basket_hit_rate = held_count / resolved_count if resolved_count else 0.0

    rank_breakdown = []
    for r in sorted(by_rank_count.keys()):
        n = by_rank_count[r]
        rank_breakdown.append({
            "rank":           r,
            "events_won":     n,
            "share_of_held":  round(n / held_count, 4) if held_count else 0.0,
            "share_of_all":   round(n / resolved_count, 4) if resolved_count else 0.0,
            "design_target":  DESIGN_RANK_WINRATE.get(r),
            "cost_total":     round(by_rank_cost[r], 4),
            "pnl_total":      round(by_rank_pnl[r], 4),
            "avg_pnl":        round(by_rank_pnl[r] / n, 4) if n else 0.0,
            "roi":            round(by_rank_pnl[r] / by_rank_cost[r], 4) if by_rank_cost[r] else 0.0,
        })

    return {
        "scope_dates":       dates,
        "events_total":      resolved_count + pending_count,
        "events_resolved":   resolved_count,
        "events_pending":    pending_count,
        "basket_hits":       held_count,
        "basket_misses":     miss_count,
        "basket_hit_rate":   round(basket_hit_rate, 4),
        "rank_breakdown":    rank_breakdown,
        "miss_cost":         round(miss_cost, 4),
        "miss_pnl":          round(miss_pnl, 4),
        "events":            event_details,
    }


def _fmt_usd(x: float) -> str:
    sign = "-" if x < -0.005 else " "
    return f"{sign}${abs(x):>9,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{100*x:>5.1f}%"


def _print_report(result: dict, verbose: bool) -> None:
    print()
    print("=" * 86)
    scope = "all dates"
    if result["scope_dates"]:
        scope = f"{result['scope_dates'][0]} → {result['scope_dates'][-1]}"
    print(f"  TKH RANK-WIN-RATE ANALYSIS  ({scope})")
    print("=" * 86)
    print(f"  Events seen:        {result['events_total']:>4d}")
    print(f"    resolved:         {result['events_resolved']:>4d}")
    print(f"    still pending:    {result['events_pending']:>4d}")
    print()
    print(f"  BASKET HIT RATE: {result['basket_hits']}/{result['events_resolved']} "
          f"= {_fmt_pct(result['basket_hit_rate'])} of resolved events")
    print(f"    └ basket misses (winner not held): {result['basket_misses']}  "
          f"cost {_fmt_usd(result['miss_cost'])}  P&L {_fmt_usd(result['miss_pnl'])}")
    print()

    if not result["rank_breakdown"]:
        print("  No resolved events with identified winners — nothing to rank.")
        return

    print("  WHEN WINNER WAS HELD — distribution by rank:")
    print()
    print(f"  {'rank':>4}  {'events':>6}  {'%held':>6}  {'%all':>6}  "
          f"{'design':>6}  {'cost':>10}  {'pnl':>10}  {'avg':>9}  {'ROI':>6}")
    print("  " + "-" * 82)
    held_n = result["basket_hits"]
    for r in result["rank_breakdown"]:
        des = f"{100*r['design_target']:.0f}%" if r["design_target"] is not None else "  —  "
        flag = ""
        if r["design_target"] is not None:
            actual = r["share_of_held"]
            diff = actual - r["design_target"]
            if abs(diff) >= 0.10:
                flag = "  ← " + ("LOW " if diff < 0 else "HIGH") + f" by {abs(100*diff):.0f}pt"
        print(
            f"  {r['rank']:>4d}  {r['events_won']:>6d}  {_fmt_pct(r['share_of_held'])}  "
            f"{_fmt_pct(r['share_of_all'])}  {des:>6s}  "
            f"{_fmt_usd(r['cost_total'])}  {_fmt_usd(r['pnl_total'])}  "
            f"{_fmt_usd(r['avg_pnl'])}  {_fmt_pct(r['roi'])}{flag}"
        )
    print()

    # Diagnosis line
    rank0 = next((r for r in result["rank_breakdown"] if r["rank"] == 0), None)
    if rank0 and result["events_resolved"] >= 10:
        actual_r0 = rank0["share_of_held"]
        if actual_r0 < 0.40:
            print("  DIAGNOSIS:  Rank-0 win share is FAR below design (55%).  The model's")
            print("              confidence ordering is broken — the bin labelled most-likely")
            print("              is rarely the actual winner.  Either improve probability")
            print("              calibration, or flatten the 60/25/15 split toward equal weights.")
        elif actual_r0 < 0.50:
            print("  DIAGNOSIS:  Rank-0 win share is below design.  Marginally miscalibrated;")
            print("              consider 50/30/20 split as a hedge against ranking error.")
        else:
            print("  DIAGNOSIS:  Rank-0 win share is on/above design.  Ranking looks OK;")
            print("              losses are coming from basket-miss rate, not rank ordering.")
    print()

    print("  P&L SOURCE BREAKDOWN (where the money actually went):")
    total_pnl = sum(r["pnl_total"] for r in result["rank_breakdown"]) + result["miss_pnl"]
    total_cost = sum(r["cost_total"] for r in result["rank_breakdown"]) + result["miss_cost"]
    for r in result["rank_breakdown"]:
        share = r["pnl_total"] / total_pnl if total_pnl else 0.0
        print(f"    rank {r['rank']} winners: {_fmt_usd(r['pnl_total']):>10s}  "
              f"({_fmt_pct(share)} of total)")
    miss_share = result["miss_pnl"] / total_pnl if total_pnl else 0.0
    print(f"    basket misses:  {_fmt_usd(result['miss_pnl']):>10s}  "
          f"({_fmt_pct(miss_share)} of total)")
    print("    " + "-" * 50)
    print(f"    GRAND TOTAL:    {_fmt_usd(total_pnl):>10s}  on {_fmt_usd(total_cost)} cost")

    if not verbose:
        print()
        print("  Run with --verbose for per-event rank tables.")
        return

    print()
    print("-" * 86)
    print("  PER-EVENT DETAIL")
    print("-" * 86)
    for ed in sorted(result["events"], key=lambda e: (e["date"], e["city"])):
        wr = ed.get("winner_rank")
        wlabel = "MISS" if wr is None else f"rank-{wr} won"
        print(f"\n  {ed['date']}  {ed['city']:<14s}  {wlabel:<14s}  "
              f"P&L {_fmt_usd(ed['realized_total'])}")
        print(f"    {'rk':>2} {'bin':>10} {'target':>7} {'mprob':>6} {'mkt':>6} "
              f"{'entry':>6} {'exit':>6} {'pnl':>9}")
        for b in ed["bins"]:
            star = "★" if b["is_winner"] else " "
            ex = f"{b['exit_price']:.3f}" if b["exit_price"] is not None else "  -  "
            print(f"  {star} {b['rank']:>2d} {b['label']:>10s} "
                  f"{b['target']:>6.2f}  {b['model_prob']:>5.3f}  {b['market_prob']:>5.3f}  "
                  f"{b['entry_price']:>5.3f}  {ex:>6}  {_fmt_usd(b['pnl'])}")


def _expand_dates(args) -> list[str] | None:
    if args.range:
        d0 = datetime.strptime(args.range[0], "%Y-%m-%d").date()
        d1 = datetime.strptime(args.range[1], "%Y-%m-%d").date()
        if d1 < d0:
            d0, d1 = d1, d0
        out, cur = [], d0
        while cur <= d1:
            out.append(cur.isoformat())
            cur += timedelta(days=1)
        return out
    if args.since_days:
        d1 = _date.today()
        d0 = d1 - timedelta(days=args.since_days)
        out, cur = [], d0
        while cur <= d1:
            out.append(cur.isoformat())
            cur += timedelta(days=1)
        return out
    if args.date:
        return [args.date]
    return None  # all-time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--date", help="Single event date YYYY-MM-DD")
    p.add_argument("--range", nargs=2, metavar=("START", "END"),
                   help="Inclusive date range, both YYYY-MM-DD")
    p.add_argument("--since-days", type=int,
                   help="Look back N days from today")
    p.add_argument("--include-paper", action="store_true",
                   help="Include paper-trade positions (default: live only)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-event rank tables")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON")
    args = p.parse_args()

    dates = _expand_dates(args)
    result = compute(dates, include_paper=args.include_paper)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())