"""
analyze_basket_wins.py — TKH basket performance analysis: which RANK
bin actually wins?

For each completed TKH event (basket of K bins sized per
TKH_PERCENTAGE_SPLIT, e.g. 70/20/10 = $14/$4/$2 of a $20 bet), this
script determines:

    * Did the rank-0 (largest, highest-prob) bin win?
    * Did the rank-1 (middle) bin win?
    * Did the rank-2 (smallest, lowest-prob) bin win?
    * Did NO bin in the basket win? (out-of-range / model-missed event)

Then aggregates across all events to show:
    - Hit rate by rank (count, %)
    - Total $ deployed by rank
    - Total $ recouped by rank (sells + redemptions)
    - Net P&L by rank

Lets you see whether the strategy's allocation matches reality:
  - If rank-0 wins 70% of the time, the 70% allocation is well-calibrated.
  - If rank-2 wins 30% of the time, the 10% slice is under-allocated.
  - If "no bin wins" >50%, the basket-edge gate or model needs work.

Data sources (everything from Polymarket, not the bot DB):
    * Polymarket Data API /activity -- BUY/SELL/REDEEM transactions per token
    * Bot DB only used to identify which positions were TKH and group them
      by event/rank

Usage:
    cd bot
    python -m scripts.analyze_basket_wins                       # all completed events
    python -m scripts.analyze_basket_wins --date 2026-05-04
    python -m scripts.analyze_basket_wins --since-days 7
    python -m scripts.analyze_basket_wins --by-event            # per-event detail
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import httpx
from collections import defaultdict
from datetime import datetime, timedelta

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


# ---------------------------------------------------------------------------
# Polymarket activity fetch (paginated)
# ---------------------------------------------------------------------------

def _fetch_all_activity(wallet: str, max_pages: int = 50) -> list[dict]:
    out: list[dict] = []
    offset = 0
    PAGE = 100
    for _ in range(max_pages):
        try:
            r = httpx.get(
                "https://data-api.polymarket.com/activity",
                params={"user": wallet, "limit": PAGE, "offset": offset},
                timeout=20,
            )
            r.raise_for_status()
            page = r.json()
        except Exception as e:
            print(f"WARN: activity fetch at offset={offset}: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return out


def _activity_buckets(activity: list[dict]) -> tuple[dict[str, dict],
                                                        dict[str, dict]]:
    """Bucket activity TWO ways:
       (a) by_token: TRADE buys/sells aggregated per asset (token_id)
       (b) by_condition: REDEEM events aggregated per conditionId
           (Polymarket REDEEMs have empty `asset` -- they're recorded
           at the market/condition level, so we have to match by
           conditionId == position.contract_id later)
    """
    by_token: dict[str, dict] = defaultdict(lambda: {
        "buy_usdc": 0.0, "buy_size": 0.0,
        "sell_usdc": 0.0, "sell_size": 0.0,
        "n_buys": 0, "n_sells": 0,
    })
    by_condition: dict[str, dict] = defaultdict(lambda: {
        "redeem_usdc": 0.0, "redeem_size": 0.0,
        "n_redeems": 0, "title": "",
    })
    for it in activity:
        ttype = it.get("type", "")
        size  = float(it.get("size", 0) or 0)
        usdc  = float(it.get("usdcSize", 0) or 0)
        if ttype == "TRADE":
            token = str(it.get("asset") or "")
            if not token:
                continue
            b = by_token[token]
            side = it.get("side", "")
            if side == "BUY":
                b["buy_size"]  += size; b["buy_usdc"]  += usdc; b["n_buys"]  += 1
            elif side == "SELL":
                b["sell_size"] += size; b["sell_usdc"] += usdc; b["n_sells"] += 1
        elif ttype == "REDEEM":
            cid = str(it.get("conditionId") or "")
            if not cid:
                continue
            r = by_condition[cid]
            r["redeem_size"] += size
            r["redeem_usdc"] += usdc
            r["n_redeems"]   += 1
            if not r["title"]:
                r["title"] = it.get("title", "")
    return dict(by_token), dict(by_condition)


# ---------------------------------------------------------------------------
# Bot DB: list TKH events
# ---------------------------------------------------------------------------

def _list_tkh_events(date_filter: str | None,
                     since_days: int | None,
                     include_paper: bool) -> dict[str, list[dict]]:
    """event_id -> list of bot positions (one per bin) for TKH events."""
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"
    where = ["strategy = 'top_k_hedged'", paper_clause.lstrip("AND ")]
    args: list = []
    if date_filter:
        where.append("date = ?")
        args.append(date_filter)
    elif since_days:
        cutoff = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
        where.append("date >= ?")
        args.append(cutoff)
    where_sql = " AND ".join(w for w in where if w)
    sql = f"""
        SELECT id, event_id, city, date, range_low, range_high, unit,
               side, contract_id, yes_token_id, no_token_id,
               gamma_market_id,
               target_size_usdc, size_usdc, shares, entry_price,
               status, fill_status, pnl
        FROM positions
        WHERE {where_sql}
        ORDER BY event_id, target_size_usdc DESC
    """
    out: dict[str, list[dict]] = defaultdict(list)
    with _get_conn() as conn:
        for r in conn.execute(sql, args).fetchall():
            d = dict(r)
            ev = d.get("event_id") or ""
            if ev:
                out[ev].append(d)
    return dict(out)


def _fetch_chain_positions_index(wallet: str) -> dict[str, dict]:
    """token_id -> chain position dict (for unrealized current_value)."""
    if not wallet:
        return {}
    from polymarket import get_data_api_positions
    return {p["token_id"]: p for p in (get_data_api_positions(wallet) or [])
            if p.get("token_id")}


# ---------------------------------------------------------------------------
# Per-event analysis
# ---------------------------------------------------------------------------

def _bin_label(pos: dict) -> str:
    rl = pos.get("range_low"); rh = pos.get("range_high")
    unit = (pos.get("unit") or "celsius").lower()
    suffix = "F" if unit == "fahrenheit" else "C"
    if rl is not None and rh is not None:
        return f"{int(rl)}{suffix}" if int(rl) == int(rh) else f"{int(rl)}-{int(rh)}{suffix}"
    return "?"


def _is_event_complete(positions: list[dict],
                        condition_activity: dict[str, dict],
                        gamma_status_cache: dict[str, dict]) -> bool:
    """An event is 'complete' iff its underlying market(s) have actually
    resolved on chain.  Two signals:
      (a) any of its bin contracts has a REDEEM event (winner paid out)
      (b) Gamma reports the contract as `closed=True`
    """
    for p in positions:
        cid = p.get("contract_id") or ""
        if cid and condition_activity.get(cid, {}).get("n_redeems", 0) > 0:
            return True
        if cid in gamma_status_cache:
            st = gamma_status_cache[cid]
            if st and st.get("closed"):
                return True
    return False


def _classify_event(positions: list[dict],
                     token_activity: dict[str, dict],
                     condition_activity: dict[str, dict],
                     gamma_status_cache: dict[str, dict],
                     chain_index: dict[str, dict]) -> dict:
    """For one event, figure out:
       - rank of each bin (by target_size_usdc desc)
       - winning bin (REDEEM event match OR Gamma winner field)
       - per-bin financials from token activity + chain unrealized
    """
    # Sort by target_size_usdc desc -> rank 0 = biggest, 1 = next, etc.
    ranked = sorted(positions,
                    key=lambda p: float(p.get("target_size_usdc") or 0),
                    reverse=True)

    bins_out = []
    winning_rank = None
    winning_bin = None
    for rank, p in enumerate(ranked):
        side = p.get("side", "YES")
        tok  = p.get("yes_token_id") if side == "YES" else p.get("no_token_id")
        cid  = p.get("contract_id") or ""
        tact = token_activity.get(tok or "", {})
        cact = condition_activity.get(cid, {})
        chain = chain_index.get(tok or "")

        # Win signals (any one is sufficient):
        #   (a) a REDEEM event fired for this bin's contract
        #   (b) Gamma reports the market closed AND winner == our side
        won = False
        if cact.get("n_redeems", 0) > 0 and cact.get("redeem_usdc", 0) > 0:
            won = True
        elif cid in gamma_status_cache:
            st = gamma_status_cache[cid]
            if st and st.get("closed"):
                w = (st.get("winner") or "").upper()
                if w == (side or "YES").upper():
                    won = True

        if won and winning_rank is None:
            winning_rank = rank
            winning_bin  = _bin_label(p)

        # Unrealized = current chain value of remaining shares.
        # Captures HEDGE_RESOLVED partial sells (those add to sell_usdc),
        # leaving the remainder in unrealized until resolution.
        chain_value = float(chain.get("current_value", 0)) if chain else 0.0

        bins_out.append({
            "pid":              p["id"],
            "rank":             rank,
            "bin":              _bin_label(p),
            "target_usdc":      float(p.get("target_size_usdc") or 0),
            "entry_price":      float(p.get("entry_price") or 0),
            "buy_usdc":         tact.get("buy_usdc", 0),
            "buy_size":         tact.get("buy_size", 0),
            "sell_usdc":        tact.get("sell_usdc", 0),
            "redeem_usdc":      cact.get("redeem_usdc", 0),
            "chain_value":      chain_value,
            "won":              won,
            "n_buys":           tact.get("n_buys", 0),
            "n_sells":          tact.get("n_sells", 0),
            "n_redeems":        cact.get("n_redeems", 0),
        })
    return {
        "event_id":     (positions[0].get("event_id") or "") if positions else "",
        "city":         (positions[0].get("city") or "")     if positions else "",
        "date":         (positions[0].get("date") or "")     if positions else "",
        "n_bins":       len(positions),
        "winning_rank": winning_rank,
        "winning_bin":  winning_bin,
        "bins":         bins_out,
    }


# ---------------------------------------------------------------------------
# Aggregation across events
# ---------------------------------------------------------------------------

def _aggregate(event_summaries: list[dict], strict_k: int | None = None) -> dict:
    """Per-rank totals across all events.  Includes both realized
    (sells + redemptions) and unrealized (current chain value of held)
    so the breakdown is meaningful for in-flight events too."""
    by_rank: dict[int, dict] = defaultdict(lambda: {
        "n_events":   0,
        "n_resolved": 0,    # events where winner is known
        "n_wins":     0,
        "buys":       0.0,  # total cost basis (USDC actually paid)
        "deployed":   0.0,  # target_size_usdc (intended bet)
        "sells":      0.0,
        "redeems":    0.0,
        "recouped":   0.0,  # sells + redeems
        "current":    0.0,  # current chain value of still-held shares
        "realized":   0.0,  # recouped - cost_basis_disposed (approximated)
        "unrealized": 0.0,  # current - cost_basis_held (approximated)
        "net":        0.0,
    })
    n_total            = len(event_summaries)
    n_resolved_events  = 0
    n_no_win_resolved  = 0
    grand: dict = {"buys": 0.0, "sells": 0.0, "redeems": 0.0,
                   "current": 0.0, "deployed": 0.0}

    for ev in event_summaries:
        # An event is "resolved" if at least one bin has a winner attribution.
        resolved = any(b["won"] for b in ev["bins"]) or (
            ev["winning_rank"] is not None
        )
        if resolved:
            n_resolved_events += 1
            if ev["winning_rank"] is None:
                n_no_win_resolved += 1
        for b in ev["bins"]:
            r = b["rank"]
            # strict_k: ignore over-allocated bins (rank >= K).  Lets
            # us answer "what would ROI be if we strictly capped at K?"
            if strict_k is not None and r >= strict_k:
                continue
            br = by_rank[r]
            br["n_events"] += 1
            if resolved:
                br["n_resolved"] += 1
            if b["won"]:
                br["n_wins"] += 1
            br["buys"]     += b["buy_usdc"]
            br["deployed"] += b["target_usdc"]
            br["sells"]    += b["sell_usdc"]
            br["redeems"]  += b["redeem_usdc"]
            br["current"]  += b["chain_value"]
            br["recouped"] = br["sells"] + br["redeems"]
            # Grand totals respect strict_k filter -- only count bins
            # the operator would have actually placed under strict K cap.
            grand["buys"]     += b["buy_usdc"]
            grand["sells"]    += b["sell_usdc"]
            grand["redeems"]  += b["redeem_usdc"]
            grand["current"]  += b["chain_value"]
            grand["deployed"] += b["target_usdc"]
    for r in by_rank:
        br = by_rank[r]
        # Net = (sells + redeems + current value of remaining shares) - cost basis
        br["net"] = br["sells"] + br["redeems"] + br["current"] - br["buys"]
    grand_net = (grand["sells"] + grand["redeems"]
                 + grand["current"] - grand["buys"])
    return {
        "n_events":          n_total,
        "n_resolved":        n_resolved_events,
        "n_no_win_resolved": n_no_win_resolved,
        "by_rank":           dict(by_rank),
        "grand":             grand,
        "grand_net":         grand_net,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=str, default=None,
                    help="Single resolution date (e.g. 2026-05-04)")
    ap.add_argument("--since-days", type=int, default=None,
                    help="All events from N days ago to today")
    ap.add_argument("--by-event", action="store_true",
                    help="Show per-event detail rows")
    ap.add_argument("--include-paper", action="store_true",
                    help="Include paper positions")
    ap.add_argument("--only-resolved", action="store_true",
                    help="Filter to events whose markets have resolved on chain. "
                         "Default behaviour includes in-flight events so you "
                         "can see live performance.")
    ap.add_argument("--strict-k", type=int, default=None,
                    help="Show what ROI would be if we strictly capped at K "
                         "bins per event (ignore over-allocated extras at "
                         "rank K+).  Pass the K value, e.g. --strict-k 3.")
    ap.add_argument("--basket-edge", action="store_true",
                    help="Show entry-time basket-edge analysis: distribution "
                         "of (1 - sum_top_K_entry_prices) by win/loss, plus "
                         "threshold recommendations for TKH_BASKET_EDGE_MIN.")
    ap.add_argument("--simulate-hold", action="store_true",
                    help="Counterfactual: simulate ROI assuming we'd held "
                         "every bin to midnight (no HEDGE_RESOLVED sells, "
                         "no early exits).  Winning bin pays buy_size * $1.00; "
                         "losing bins go to $0.  Compares simulated vs actual.")
    ap.add_argument("--json", action="store_true",
                    help="Emit raw JSON for piping")
    args = ap.parse_args()

    if not args.date and not args.since_days:
        # Default: last 7 days
        args.since_days = 7

    from config import WALLET_ADDRESS
    if not WALLET_ADDRESS:
        print("ERROR: WALLET_ADDRESS not configured")
        return 1

    print(f"Loading TKH events from DB...")
    events = _list_tkh_events(args.date, args.since_days, args.include_paper)
    if not events:
        print(f"No TKH events found.")
        return 0
    print(f"  {len(events)} TKH events with positions")

    print(f"Fetching activity history from Polymarket...")
    activity = _fetch_all_activity(WALLET_ADDRESS)
    print(f"  {len(activity)} activity items")
    token_activity, condition_activity = _activity_buckets(activity)
    print(f"  {len(token_activity)} tokens with TRADE activity, "
          f"{len(condition_activity)} markets with REDEEM events")

    print(f"Fetching current chain holdings (for unrealized calc)...")
    chain_index = _fetch_chain_positions_index(WALLET_ADDRESS)
    print(f"  {len(chain_index)} currently-held tokens")

    # Pre-fetch Gamma market status for every contract we care about.
    # Pass gamma_market_id (the numeric ID) when available -- the
    # condition_id fallback now uses the conditionId query param after
    # the polymarket.py fix, so it works either way.
    print(f"Fetching Gamma market status for resolved-event detection...")
    from polymarket import get_market_status
    gamma_cache: dict[str, dict] = {}
    pairs: dict[str, str] = {}     # contract_id -> gamma_market_id
    for plist in events.values():
        for p in plist:
            cid = p.get("contract_id") or ""
            gid = p.get("gamma_market_id") or ""
            if cid and cid not in pairs:
                pairs[cid] = gid
    for cid, gid in pairs.items():
        try:
            st = get_market_status(cid, gamma_market_id=(gid or None))
            if st is not None:
                gamma_cache[cid] = st
        except Exception:
            pass
    n_resolved_contracts = sum(1 for st in gamma_cache.values() if st.get("closed"))
    print(f"  {len(gamma_cache)}/{len(pairs)} contracts queried, "
          f"{n_resolved_contracts} reported closed")

    # Classify each event
    summaries = []
    n_skipped_incomplete = 0
    for ev_id in sorted(events.keys()):
        positions = events[ev_id]
        if args.only_resolved and not _is_event_complete(
                positions, condition_activity, gamma_cache):
            n_skipped_incomplete += 1
            continue
        summaries.append(_classify_event(
            positions, token_activity, condition_activity, gamma_cache,
            chain_index,
        ))

    if args.json:
        print(json.dumps({
            "events":      summaries,
            "aggregate":   _aggregate(summaries),
            "skipped_incomplete": n_skipped_incomplete,
        }, indent=2, default=str))
        return 0

    agg = _aggregate(summaries, strict_k=args.strict_k)
    n_events_total    = agg["n_events"]
    n_resolved        = agg["n_resolved"]
    n_no_win_resolved = agg["n_no_win_resolved"]

    print()
    if args.strict_k is not None:
        print(f"=== TKH basket performance (STRICT K={args.strict_k}) ===")
        print(f"  Aggregates exclude bins at rank >= {args.strict_k}, simulating")
        print(f"  what the strategy would have produced with strict K-cap.")
    else:
        print(f"=== TKH basket performance analysis ===")
    print(f"  events shown:      {len(summaries)}")
    print(f"  events resolved:   {n_resolved}")
    print(f"  events in-flight:  {n_events_total - n_resolved}")
    if n_skipped_incomplete:
        print(f"  events skipped:    {n_skipped_incomplete} (--only-resolved)")
    print()

    # Per-event detail
    if args.by_event and summaries:
        print(f"--- Per-event detail ---")
        print(f"{'event':<8} {'city':<14} {'date':<11} {'#':>3} "
              f"{'state':<9} {'won by':<12} "
              f"{'cost':>9} {'sold':>9} {'redeem':>9} {'value':>9} "
              f"{'NET':>10}")
        for ev in summaries:
            cost = sum(b["buy_usdc"] for b in ev["bins"])
            sold = sum(b["sell_usdc"] for b in ev["bins"])
            redm = sum(b["redeem_usdc"] for b in ev["bins"])
            curr = sum(b["chain_value"] for b in ev["bins"])
            net  = sold + redm + curr - cost
            if ev["winning_rank"] is not None:
                state = "RESOLVED"
                won_str = f"rank {ev['winning_rank']} ({ev['winning_bin']})"
            elif any(b["redeem_usdc"] for b in ev["bins"]):
                state = "RESOLVED"; won_str = "(losing)"
            else:
                state = "in-flight"; won_str = "(pending)"
            print(f"{ev['event_id']:<8} {ev['city'][:14]:<14} "
                  f"{ev['date']:<11} {ev['n_bins']:>3} "
                  f"{state:<9} {won_str:<12} "
                  f"${cost:>8.2f} ${sold:>8.2f} ${redm:>8.2f} "
                  f"${curr:>8.2f} ${net:>+9.2f}")
        print()

    # Per-rank breakdown
    print(f"--- Performance by rank (across all {n_events_total} events) ---")
    print(f"{'rank':<6} {'split':<10} {'events':>7} {'cost':>9} "
          f"{'sold':>9} {'redeem':>9} {'value':>9} "
          f"{'NET':>10}  {'ROI':>7}")
    for r in sorted(agg["by_rank"].keys()):
        d = agg["by_rank"][r]
        try:
            from strategies.top_k_hedged import TKH_SPLIT
            pct_label = (f"{TKH_SPLIT[r]*100:.0f}%"
                         if r < len(TKH_SPLIT) else "extra")
        except Exception:
            pct_label = "?"
        roi = (d["net"] / d["buys"] * 100) if d["buys"] else 0
        print(f"  {r}    {pct_label:<8}  {d['n_events']:>7} "
              f"${d['buys']:>8.2f} ${d['sells']:>8.2f} "
              f"${d['redeems']:>8.2f} ${d['current']:>8.2f} "
              f"${d['net']:>+9.2f}  {roi:>+6.1f}%")
    print()

    # Win-rate breakdown — only meaningful for resolved events
    if n_resolved > 0:
        print(f"--- Win rate by rank (across {n_resolved} resolved events) ---")
        print(f"{'rank':<6} {'split':<10} {'n_resolved':>11} "
              f"{'n_wins':>7} {'win rate':>10}")
        for r in sorted(agg["by_rank"].keys()):
            d = agg["by_rank"][r]
            try:
                from strategies.top_k_hedged import TKH_SPLIT
                pct_label = (f"{TKH_SPLIT[r]*100:.0f}%"
                             if r < len(TKH_SPLIT) else "extra")
            except Exception:
                pct_label = "?"
            win_rate = ((d["n_wins"] / d["n_resolved"] * 100)
                        if d["n_resolved"] else 0)
            print(f"  {r}    {pct_label:<8}  {d['n_resolved']:>11} "
                  f"{d['n_wins']:>7}  {win_rate:>8.1f}%")
        print(f"  Events where NO bin won the basket:  "
              f"{n_no_win_resolved} / {n_resolved} "
              f"({n_no_win_resolved/n_resolved*100:.1f}%)")
        print()
    else:
        print(f"--- Win rate by rank ---")
        print(f"  No events have resolved on chain yet.  This view will")
        print(f"  populate as your weather markets close at midnight (local")
        print(f"  time per city).  Run again tomorrow morning.")
        print()

    # Grand summary
    g = agg["grand"]
    print(f"--- Grand totals ---")
    print(f"  Deployed (intended):   ${g['deployed']:>9.2f}")
    print(f"  Cost basis (actual):   ${g['buys']:>9.2f}")
    print(f"  Sold:                  ${g['sells']:>9.2f}")
    print(f"  Redeemed:              ${g['redeems']:>9.2f}")
    print(f"  Current chain value:   ${g['current']:>9.2f}")
    print(f"  NET P&L:               ${agg['grand_net']:>+9.2f}")
    if g['buys']:
        print(f"  Strategy ROI:          {agg['grand_net']/g['buys']*100:+.2f}%")

    # ---- Basket-edge analysis ----
    if args.basket_edge:
        try:
            from strategies.top_k_hedged import TKH_TOP_K
            K = TKH_TOP_K
        except Exception:
            K = 3
        if args.strict_k is not None:
            K = args.strict_k

        # Per-event: sum of top-K entry prices, basket_edge = 1 - that sum
        ev_edges: list[dict] = []
        for ev in summaries:
            top_k_bins = sorted(ev["bins"], key=lambda b: b["target_usdc"],
                                reverse=True)[:K]
            entry_sum = sum(b["entry_price"] for b in top_k_bins
                            if b["entry_price"] > 0)
            if entry_sum <= 0:
                continue
            edge = 1.0 - entry_sum
            won_top_k = any(b["won"] for b in top_k_bins)
            ev_cost = sum(b["buy_usdc"] for b in ev["bins"])
            ev_pnl  = (sum(b["sell_usdc"]   for b in ev["bins"])
                       + sum(b["redeem_usdc"] for b in ev["bins"])
                       + sum(b["chain_value"] for b in ev["bins"])
                       - ev_cost)
            ev_edges.append({
                "event_id":  ev["event_id"],
                "city":      ev["city"],
                "edge":      edge,
                "entry_sum": entry_sum,
                "won":       won_top_k,
                "resolved":  any(b["won"] for b in ev["bins"]) or ev["winning_rank"] is not None,
                "cost":      ev_cost,
                "pnl":       ev_pnl,
            })

        if not ev_edges:
            print()
            print(f"--- Basket-edge analysis ---")
            print(f"  No events with valid entry prices to analyze.")
            return 0

        print()
        print(f"=== Basket-edge analysis (top-{K} entry prices) ===")
        print(f"  Edge = 1 - sum(top-{K} entry prices).  Higher edge means")
        print(f"  the basket bought 'cheap' relative to total prob mass.")
        print(f"  Currently TKH_BASKET_EDGE_MIN gates entry by this number.")
        print()

        # Bucket by win/loss (resolved events only) + by edge bands
        won_edges    = [e for e in ev_edges if e["resolved"] and e["won"]]
        lost_edges   = [e for e in ev_edges if e["resolved"] and not e["won"]]
        pending      = [e for e in ev_edges if not e["resolved"]]

        def _stats(rows):
            edges = sorted(e["edge"] for e in rows)
            n = len(edges)
            if n == 0: return (0, 0, 0, 0, 0, 0)
            return (n, min(edges), edges[n//4], edges[n//2],
                    edges[(3*n)//4], max(edges))

        print(f"  {'Group':<22} {'n':>4}  {'min':>7} "
              f"{'p25':>7} {'median':>8} {'p75':>7} {'max':>7}")
        for label, rows in (("All events", ev_edges),
                            ("Resolved + WON top-K", won_edges),
                            ("Resolved + LOST top-K", lost_edges),
                            ("In-flight (pending)", pending)):
            n, mn, p25, p50, p75, mx = _stats(rows)
            if n:
                print(f"  {label:<22} {n:>4}  {mn:>7.4f} "
                      f"{p25:>7.4f} {p50:>8.4f} {p75:>7.4f} {mx:>7.4f}")

        # Threshold simulator: for each candidate min-edge, compute what
        # would have happened if we had ONLY entered events with edge >= T.
        print()
        print(f"--- TKH_BASKET_EDGE_MIN threshold simulator ---")
        print(f"  For each threshold T, would have only entered events where")
        print(f"  basket_edge >= T.  Shows how many events would have been")
        print(f"  filtered out, plus the simulated NET P&L on the remaining.")
        print()
        print(f"  {'threshold':>10}  {'n_kept':>7} {'n_filtered':>11} "
              f"{'cost':>9} {'sim NET':>10} {'sim ROI':>9}")
        thresholds = [0.00, 0.05, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
        for t in thresholds:
            kept = [e for e in ev_edges if e["edge"] >= t]
            filtered = len(ev_edges) - len(kept)
            sim_cost = sum(e["cost"] for e in kept)
            sim_pnl  = sum(e["pnl"]  for e in kept)
            sim_roi  = (sim_pnl / sim_cost * 100) if sim_cost else 0
            marker = " <-- current" if abs(t - 0.09) < 0.001 else ""
            print(f"  {t:>10.2f}  {len(kept):>7} {filtered:>11} "
                  f"${sim_cost:>8.2f} ${sim_pnl:>+9.2f} "
                  f"{sim_roi:>+8.2f}%{marker}")

    # ---- Hold-to-maturity counterfactual ----
    if args.simulate_hold:
        # Per-event: what would NET have been if we'd held every bin to
        # midnight (no HEDGE_RESOLVED, no stop-loss, no early sells)?
        #   - Winning bin payout = buy_size shares * $1.00
        #   - Losing bins payout = 0
        #   - sim_net = winner_payout - sum(all bin buy_usdc)
        #
        # Only meaningful for events that have actually resolved (we
        # need to know which bin won).  Skips in-flight events.

        sim_rows = []
        for ev in summaries:
            # Only simulate resolved events (need known winner)
            resolved = (ev["winning_rank"] is not None) or any(
                b["won"] for b in ev["bins"]
            )
            if not resolved:
                continue
            cost = sum(b["buy_usdc"] for b in ev["bins"])
            # Find the winning bin (any bin marked won=True)
            winner = next((b for b in ev["bins"] if b["won"]), None)
            if winner is None:
                # Resolved but no top-K winner (basket missed completely)
                payout = 0.0
            else:
                # If we'd HELD all winner shares to maturity:
                # payout = total_buy_size * $1.00
                payout = float(winner["buy_size"])
            sim_net = payout - cost
            actual_net = (sum(b["sell_usdc"]   for b in ev["bins"])
                          + sum(b["redeem_usdc"] for b in ev["bins"])
                          + sum(b["chain_value"] for b in ev["bins"])
                          - cost)
            sim_rows.append({
                "event_id":   ev["event_id"],
                "city":       ev["city"],
                "winner":     (f"rank {ev['winning_rank']} ({ev['winning_bin']})"
                               if ev["winning_rank"] is not None
                               else "(off-rank or none)"),
                "cost":       cost,
                "winner_shares": (winner["buy_size"] if winner else 0),
                "actual_net": actual_net,
                "sim_net":    sim_net,
                "delta":      sim_net - actual_net,
            })

        print()
        print(f"=== Hold-to-maturity counterfactual ===")
        print(f"  What if we had NEVER triggered HEDGE_RESOLVED or any other")
        print(f"  early sell?  Simulates: hold every bin to midnight, collect")
        print(f"  $1.00/share for the winning bin, $0 for losers.  Compared")
        print(f"  to ACTUAL P&L (which includes all early sells).")
        print()

        if not sim_rows:
            print(f"  No resolved events to simulate yet.  Re-run after")
            print(f"  more events resolve at midnight.")
        else:
            print(f"  {'event':<8} {'city':<14} {'won by':<22} "
                  f"{'cost':>9} {'winner_sh':>10} "
                  f"{'actual':>9} {'sim_hold':>10} {'delta':>10}")
            for r in sim_rows:
                print(f"  {r['event_id']:<8} {r['city'][:14]:<14} "
                      f"{r['winner']:<22} "
                      f"${r['cost']:>8.2f} {r['winner_shares']:>10.2f} "
                      f"${r['actual_net']:>+8.2f} "
                      f"${r['sim_net']:>+9.2f} ${r['delta']:>+9.2f}")
            print()
            tot_cost   = sum(r["cost"]       for r in sim_rows)
            tot_actual = sum(r["actual_net"] for r in sim_rows)
            tot_sim    = sum(r["sim_net"]    for r in sim_rows)
            tot_delta  = sum(r["delta"]      for r in sim_rows)
            actual_roi = (tot_actual / tot_cost * 100) if tot_cost else 0
            sim_roi    = (tot_sim    / tot_cost * 100) if tot_cost else 0
            print(f"  --- Aggregate over {len(sim_rows)} resolved events ---")
            print(f"  Cost basis:        ${tot_cost:>9.2f}")
            print(f"  Actual NET:        ${tot_actual:>+9.2f}  "
                  f"({actual_roi:+.2f}% ROI)")
            print(f"  Sim hold-to-mat:   ${tot_sim:>+9.2f}  "
                  f"({sim_roi:+.2f}% ROI)")
            print(f"  Delta (sim-actual):${tot_delta:>+9.2f}  "
                  f"<-- value HEDGE_RESOLVED+stops cost us")
            print()
            n_better = sum(1 for r in sim_rows if r["delta"] > 0.50)
            n_worse  = sum(1 for r in sim_rows if r["delta"] < -0.50)
            n_neutral = len(sim_rows) - n_better - n_worse
            print(f"  Hold would have been better in:  {n_better}/{len(sim_rows)} events")
            print(f"  Hold would have been worse in:   {n_worse}/{len(sim_rows)} events")
            print(f"  ~Neutral (within $0.50):         {n_neutral}/{len(sim_rows)} events")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
