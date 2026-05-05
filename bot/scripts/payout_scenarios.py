"""
payout_scenarios.py — Compute max / min / per-rank payout scenarios for
all currently-held TKH bins.

For every event the bot has alive positions on, computes:
  - cost_basis  = sum(chain initial_value across all bins for the event)
  - if_rank_N_wins = (bin_N share_count * $1.00) - cost_basis
                    (i.e., that bin redeems at $1, all other bins go to $0)
  - if_best_wins  = max(if_rank_N_wins) across all bins in the event
  - if_none_wins  = -cost_basis (every bin loses to a non-held outcome)

Aggregates across all events for each scenario:
  - MAX PAYOUT (rank 0 wins everywhere)
  - MAX PAYOUT (best bin per event wins -- best case)
  - PER-RANK PAYOUT (rank 1 / rank 2 / etc. wins everywhere)
  - MAX LOSS (no held bin wins anywhere)

Data sources (chain truth via Polymarket Data API):
  - /positions  -- current size, avg_price, initial_value per token
  - bot DB     -- map token_id -> event_id (for grouping bins by event)

Usage:
    cd bot
    python -m scripts.payout_scenarios
    python -m scripts.payout_scenarios --by-event       # per-event detail
    python -m scripts.payout_scenarios --date 2026-05-06   # one resolution date
    python -m scripts.payout_scenarios --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


def _bin_label(rl, rh, unit="celsius") -> str:
    suffix = "F" if (unit or "celsius").lower() == "fahrenheit" else "C"
    if rl is not None and rh is not None:
        if int(rl) == int(rh):
            return f"{int(rl)}{suffix}"
        return f"{int(rl)}-{int(rh)}{suffix}"
    return "?"


def _fetch_db_positions(strategy: str, date_filter: str | None,
                         include_paper: bool) -> list[dict]:
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"
    where = [f"strategy = '{strategy}'", paper_clause.lstrip("AND ")]
    args: list = []
    if date_filter:
        where.append("date = ?")
        args.append(date_filter)
    where_sql = " AND ".join(w for w in where if w)
    sql = f"""
        SELECT id, event_id, city, date, range_low, range_high, unit,
               side, contract_id, yes_token_id, no_token_id,
               target_size_usdc, size_usdc, shares, entry_price,
               status, fill_status
        FROM positions
        WHERE {where_sql}
          AND status IN ('open', 'exiting')
          AND COALESCE(fill_status, '') != 'cancelled'
        ORDER BY event_id, target_size_usdc DESC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def _fetch_chain_index(wallet: str) -> dict[str, dict]:
    if not wallet:
        return {}
    from polymarket import get_data_api_positions
    return {p["token_id"]: p for p in (get_data_api_positions(wallet) or [])
            if p.get("token_id")}


def _build_event_groups(positions: list[dict],
                         chain_index: dict[str, dict]) -> dict[str, dict]:
    """event_id -> {city, date, bins[]} where each bin has shares + cost
    pulled from chain truth (falling back to DB shares × DB entry_price
    when the chain doesn't have the position).
    """
    out: dict[str, dict] = {}
    for p in positions:
        ev = p.get("event_id") or f"_orphan_{p['id']}"
        side = p.get("side", "YES")
        tok  = p.get("yes_token_id") if side == "YES" else p.get("no_token_id")
        chain = chain_index.get(tok or "")

        if chain:
            shares = float(chain.get("size", 0))
            cost   = float(chain.get("initial_value")
                           or shares * float(chain.get("avg_price", 0)))
            avg    = float(chain.get("avg_price", 0))
            source = "chain"
        else:
            # Fallback to DB (probably not actively held; useful when
            # chain hasn't refreshed but DB shows recent placement)
            shares = float(p.get("shares", 0) or 0)
            cost   = float(p.get("size_usdc", 0) or 0)
            avg    = float(p.get("entry_price", 0) or 0)
            source = "db_fallback"

        if shares <= 0 and cost <= 0:
            continue   # truly empty position; skip

        if ev not in out:
            out[ev] = {
                "city":  p.get("city", ""),
                "date":  p.get("date", ""),
                "bins":  [],
            }
        out[ev]["bins"].append({
            "pid":          p["id"],
            "bin":          _bin_label(p.get("range_low"),
                                        p.get("range_high"),
                                        p.get("unit")),
            "target_usdc":  float(p.get("target_size_usdc", 0) or 0),
            "shares":       shares,
            "cost":         cost,
            "avg_price":    avg,
            "source":       source,
        })

    # Sort each event's bins by target_usdc desc (so [0] = rank 0 etc)
    for ev in out.values():
        ev["bins"].sort(key=lambda b: b["target_usdc"], reverse=True)
    return out


def _scenarios_for_event(ev: dict) -> dict:
    """For one event compute payout under each scenario."""
    bins = ev["bins"]
    cost = sum(b["cost"] for b in bins)
    if not bins:
        return {"cost": 0.0, "if_none_wins": 0.0,
                "if_best_wins": 0.0, "by_rank": {}}
    # Per-rank: if bin at rank R wins, payout = R.shares * $1.00 - cost
    by_rank = {}
    for r, b in enumerate(bins):
        payout = b["shares"] * 1.0
        net    = payout - cost
        by_rank[r] = {
            "bin":      b["bin"],
            "shares":   b["shares"],
            "payout":   payout,
            "net":      net,
        }
    best_rank = max(by_rank, key=lambda r: by_rank[r]["net"])
    return {
        "cost":         cost,
        "if_none_wins": -cost,
        "if_best_wins": by_rank[best_rank]["net"],
        "best_rank":    best_rank,
        "by_rank":      by_rank,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="top_k_hedged",
                    help="Strategy to filter by (default: top_k_hedged)")
    ap.add_argument("--date", default=None,
                    help="Filter to one resolution date (default: all)")
    ap.add_argument("--include-paper", action="store_true",
                    help="Include paper positions")
    ap.add_argument("--by-event", action="store_true",
                    help="Show per-event breakdown")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON for piping")
    ap.add_argument("--win-weights", type=str, default=None,
                    help="Apply custom per-rank win probabilities to "
                         "compute expected ROI.  Format: 'p0:p1:p2[:p3:...]'.  "
                         "Each p_i is the probability that rank i wins.  "
                         "Should sum to <= 1.0 (remainder = no-bin-wins).  "
                         "Example: --win-weights 0.467:0.333:0.231 "
                         "(applies the 5/4 historical distribution).")
    args = ap.parse_args()

    from config import WALLET_ADDRESS
    if not WALLET_ADDRESS:
        print("ERROR: WALLET_ADDRESS not configured")
        return 1

    print(f"Loading {args.strategy} positions"
          + (f" for date={args.date}" if args.date else "") + "...")
    db_pos = _fetch_db_positions(args.strategy, args.date, args.include_paper)
    if not db_pos:
        print(f"No alive {args.strategy} positions in DB.")
        return 0
    print(f"  {len(db_pos)} alive positions")

    print(f"Fetching chain holdings from Polymarket Data API...")
    chain = _fetch_chain_index(WALLET_ADDRESS)
    print(f"  {len(chain)} tokens currently held on chain")

    events = _build_event_groups(db_pos, chain)
    print(f"  grouped into {len(events)} event(s)")
    print()

    # Per-event scenarios
    per_event = {ev_id: _scenarios_for_event(ev)
                 for ev_id, ev in events.items()}

    # Aggregates
    total_cost = sum(s["cost"] for s in per_event.values())

    sum_if_none      = sum(s["if_none_wins"] for s in per_event.values())
    sum_if_best      = sum(s["if_best_wins"] for s in per_event.values())

    # Per-rank: only sum events where that rank exists
    n_per_rank: dict[int, int] = defaultdict(int)
    sum_per_rank: dict[int, float] = defaultdict(float)
    for ev_id, sc in per_event.items():
        for r, info in sc["by_rank"].items():
            n_per_rank[r] += 1
            sum_per_rank[r] += info["net"]

    if args.json:
        print(json.dumps({
            "events":     {ev_id: {"city": events[ev_id]["city"],
                                    "date": events[ev_id]["date"],
                                    **per_event[ev_id]}
                            for ev_id in per_event},
            "totals": {
                "cost":         total_cost,
                "if_none_wins": sum_if_none,
                "if_best_wins": sum_if_best,
                "per_rank":     {r: {"n_events": n_per_rank[r],
                                      "sum_net": sum_per_rank[r]}
                                  for r in sorted(sum_per_rank)},
            },
        }, indent=2, default=str))
        return 0

    # ---- Per-event detail ----
    if args.by_event:
        print(f"--- Per-event payout scenarios ---")
        # Determine max number of ranks to show
        max_k = max((len(s["by_rank"]) for s in per_event.values()),
                    default=0)
        rank_headers = "  ".join(f"if_R{r}_wins"
                                  for r in range(max_k))
        print(f"{'event':<8} {'city':<14} {'date':<11} {'#':>2} "
              f"{'cost':>8}  {'if_none':>9}  {rank_headers}  {'best':>9}")
        for ev_id in sorted(per_event):
            sc = per_event[ev_id]
            ev = events[ev_id]
            row = (f"{ev_id:<8} {ev['city'][:14]:<14} {ev['date']:<11} "
                   f"{len(sc['by_rank']):>2} "
                   f"${sc['cost']:>7.2f}  ${sc['if_none_wins']:>+8.2f}  ")
            for r in range(max_k):
                info = sc["by_rank"].get(r)
                if info is None:
                    row += f"{'--':>10}  "
                else:
                    row += f"${info['net']:>+9.2f}  "
            row += f"${sc['if_best_wins']:>+8.2f}"
            print(row)
        print()

    # ---- Aggregate scenarios ----
    print(f"=== TOTAL PAYOUT SCENARIOS ===")
    print(f"  Events held:       {len(per_event)}")
    print(f"  Total cost basis:  ${total_cost:>8.2f}")
    print()
    print(f"  Scenario                                     Net P&L      ROI")
    print(f"  --------                                     -------      ---")
    print(f"  MAX LOSS (no held bin wins anywhere):     "
          f"${sum_if_none:>+9.2f}  {(sum_if_none/total_cost*100) if total_cost else 0:>+6.2f}%")
    for r in sorted(sum_per_rank):
        n = n_per_rank[r]
        net = sum_per_rank[r]
        roi = (net / total_cost * 100) if total_cost else 0
        # Cost of events that have this rank (for ROI denominator)
        cost_with_rank = sum(
            sc["cost"] for sc in per_event.values()
            if r in sc["by_rank"]
        )
        roi_per_rank = (net / cost_with_rank * 100) if cost_with_rank else 0
        print(f"  Rank {r} wins everywhere ({n} events):     "
              f"${net:>+9.2f}  {roi_per_rank:>+6.2f}% (of ${cost_with_rank:.0f})")
    print(f"  MAX PAYOUT (best bin per event wins):     "
          f"${sum_if_best:>+9.2f}  {(sum_if_best/total_cost*100) if total_cost else 0:>+6.2f}%")
    print()

    # ---- Custom win-weight expected-value simulator ----
    if args.win_weights:
        try:
            weights = [float(x) for x in args.win_weights.split(":")]
        except Exception as e:
            print(f"  ERROR: --win-weights parse failed: {e}")
            return 0
        sum_weights = sum(weights)
        no_win_weight = max(0.0, 1.0 - sum_weights)
        print()
        print(f"=== EXPECTED PAYOUT under custom win-weights ===")
        weights_str = " / ".join(f"{w*100:.1f}%" for w in weights)
        print(f"  Probability rank wins: {weights_str}")
        print(f"  Probability no held bin wins: {no_win_weight*100:.1f}%")
        print()

        # For each event, expected_net = sum_r(weight[r] * rank_r_payout)
        #                              + no_win_weight * if_none_wins
        ev_expected: list[tuple[str, str, float, float, float]] = []
        for ev_id, sc in per_event.items():
            ev_meta = events[ev_id]
            cost = sc["cost"]
            ev_exp = 0.0
            for r, w in enumerate(weights):
                if r in sc["by_rank"]:
                    ev_exp += w * sc["by_rank"][r]["net"]
                else:
                    # Rank doesn't exist for this event (fewer bins than r+1)
                    # Treat as if-none-wins for this slice of probability
                    ev_exp += w * sc["if_none_wins"]
            ev_exp += no_win_weight * sc["if_none_wins"]
            ev_expected.append((ev_id, ev_meta["city"], cost, ev_exp,
                                 (ev_exp/cost*100) if cost else 0))

        if args.by_event:
            print(f"  Per-event expected payout (under win-weights):")
            print(f"  {'event':<8} {'city':<14} {'cost':>8} {'exp_net':>10} {'exp_ROI':>9}")
            for ev_id, city, cost, exp, roi in sorted(ev_expected,
                                                        key=lambda x: x[3]):
                print(f"  {ev_id:<8} {city[:14]:<14} ${cost:>7.2f} "
                      f"${exp:>+9.2f}  {roi:>+7.2f}%")
            print()

        total_exp = sum(e[3] for e in ev_expected)
        total_cost_w = sum(e[2] for e in ev_expected)
        print(f"  EXPECTED TOTAL P&L:    ${total_exp:>+9.2f}")
        print(f"  Cost basis:            ${total_cost_w:>9.2f}")
        if total_cost_w:
            print(f"  EXPECTED ROI:          {total_exp/total_cost_w*100:>+9.2f}%")
        # Show rough confidence bounds (very approximate)
        # Standard deviation of binary outcomes per event:
        # var = sum_r weight[r] * (payout[r] - mean)^2 + no_win * (-cost - mean)^2
        # Sum across independent events.  Approximate.
        import math
        total_var = 0.0
        for ev_id, sc in per_event.items():
            mean_ev = 0.0
            for r, w in enumerate(weights):
                payoff = sc["by_rank"].get(r, {}).get("net", sc["if_none_wins"])
                mean_ev += w * payoff
            mean_ev += no_win_weight * sc["if_none_wins"]
            v = 0.0
            for r, w in enumerate(weights):
                payoff = sc["by_rank"].get(r, {}).get("net", sc["if_none_wins"])
                v += w * (payoff - mean_ev) ** 2
            v += no_win_weight * (sc["if_none_wins"] - mean_ev) ** 2
            total_var += v
        std = math.sqrt(total_var)
        print(f"  Approx 1-stddev range: ${total_exp - std:+.2f} to ${total_exp + std:+.2f}")
        print(f"  (+/- 1 stddev assumes independent events, which is roughly true)")
        print()

    # ---- Reality check ----
    # If TKH model is roughly calibrated (rank 0 wins ~70% of the time),
    # expected ROI = (0.70 × rank-0_avg_payout) + (0.20 × rank-1) + (0.10 × rank-2) + (...)
    if all(r in sum_per_rank for r in (0, 1, 2)):
        c0 = sum(sc["cost"] for sc in per_event.values() if 0 in sc["by_rank"])
        c1 = sum(sc["cost"] for sc in per_event.values() if 1 in sc["by_rank"])
        c2 = sum(sc["cost"] for sc in per_event.values() if 2 in sc["by_rank"])
        if c0 > 0 and c1 > 0 and c2 > 0:
            ev0 = sum_per_rank[0] / c0
            ev1 = sum_per_rank[1] / c1
            ev2 = sum_per_rank[2] / c2
            # Simple weighted EV using your TKH split as the win-rate proxy
            try:
                from strategies.top_k_hedged import TKH_SPLIT
                w0, w1, w2 = TKH_SPLIT[0], TKH_SPLIT[1], TKH_SPLIT[2]
                expected_roi = w0 * ev0 + w1 * ev1 + w2 * ev2
                print(f"  Expected blended ROI (using TKH_SPLIT "
                      f"{w0:.0%}/{w1:.0%}/{w2:.0%} as win-rate proxies):"
                      f"  {expected_roi*100:>+6.2f}%")
                print(f"    rank-0 ROI per event {ev0*100:+.1f}%, "
                      f"rank-1 {ev1*100:+.1f}%, rank-2 {ev2*100:+.1f}%")
            except Exception:
                pass
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
