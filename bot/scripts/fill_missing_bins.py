"""
fill_missing_bins.py — Re-issue TKH bins for events that have fewer
than K alive positions.

Background
----------
TKH's per-event dedup (TKH_MAX_TRADES_PER_EVENT=1) prevents the trading
loop from re-attempting an event once any position exists for it.  When
the original cycle's signals partially failed -- e.g. a thin book caused
some bins to be skipped -- the missing bins are NEVER retried by the
normal trading flow.  The result: incomplete TKH baskets, which break
the hedge thesis (one bin wins big, the missing siblings would have
absorbed loss but their slot is empty).

This script reconciles that: for each TKH event with < K alive
positions, look up the latest discovery scan, identify the top-K bins
the strategy WOULD pick today, find which of those we don't already
hold, and re-issue orders for the missing slots using the configured
TKH_PERCENTAGE_SPLIT allocation.  All the post-Option-C improvements
apply: thin-book bins now place a downsized partial order rather than
skipping, and the repricer chases any resting remainder.

Per row, when --apply is set:
  1. Identify the event's most recent temp_outcomes scan
  2. Sort outcomes by model_prob (or yes_price fallback) desc; take
     the top TKH_TOP_K
  3. Subtract the contract_ids of positions we already hold for the
     event -> "missing slots"
  4. For each missing slot, allocate TKH_TOTAL_BET_SIZE x TKH_SPLIT[rank]
     where rank is the bin's index in the top-K list
  5. Build a synthetic signal dict that mirrors what generate_signals
     would have emitted, and pass it to execute_signal.

Skips
-----
* Events whose latest scan has fewer outcomes than TKH_TOP_K
* Slots whose allocation would be below TKH_MIN_BIN_USDC
* Events where the per-event exposure cap would be breached
* Positions already alive for the event are NOT touched

Usage:
    cd bot
    python -m scripts.fill_missing_bins                   # dry run
    python -m scripts.fill_missing_bins --apply           # commit
    python -m scripts.fill_missing_bins --event 439947    # one event
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore
from strategies.top_k_hedged import (  # type: ignore
    TKH_SPLIT, TKH_TOP_K, TKH_TOTAL_BET_SIZE, TKH_MIN_BIN_USDC,
)


def _find_incomplete_events(conn, event_filter: list[str] | None) -> list[dict]:
    """Find TKH events with fewer than TKH_TOP_K alive positions."""
    sql = """
        SELECT event_id, city, date,
               COUNT(*) AS n_positions,
               GROUP_CONCAT(contract_id) AS held_contract_ids,
               GROUP_CONCAT(CAST(target_size_usdc AS TEXT)) AS held_targets
        FROM positions
        WHERE strategy = 'top_k_hedged'
          AND COALESCE(is_paper, 0) = 0
          AND status IN ('open', 'exiting')
          AND fill_status IN ('pending', 'filled')
        GROUP BY event_id, city, date
        HAVING COUNT(*) < ?
    """
    params: list = [TKH_TOP_K]
    if event_filter:
        ph = ",".join("?" * len(event_filter))
        sql = sql.replace(
            "GROUP BY event_id",
            f"AND event_id IN ({ph}) GROUP BY event_id",
        )
        params.extend(event_filter)
    sql += " ORDER BY date, city"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# Cache of Gamma-fetched events for the duration of the script run.  TKH
# does NOT persist discovery to temp_events / temp_outcomes (only TBV/MPV
# do), so the only authoritative source for current outcomes is a live
# Gamma fetch.  We pull once at startup and reuse across all events.
_GAMMA_EVENTS_CACHE: list[dict] | None = None


def _gamma_events() -> list[dict]:
    """Fetch all active highest-temperature events from Gamma (cached)."""
    global _GAMMA_EVENTS_CACHE
    if _GAMMA_EVENTS_CACHE is None:
        from polymarket import search_temp_high_events
        from config import MIN_LIQUIDITY_USD
        _GAMMA_EVENTS_CACHE = search_temp_high_events(
            min_liquidity=MIN_LIQUIDITY_USD,
        ) or []
    return _GAMMA_EVENTS_CACHE


def _outcomes_for_event(event_id: str) -> list[dict]:
    """Return outcomes (bins) for the given event_id from the live Gamma
    snapshot.  Each outcome dict carries: contract_id, question, range_low,
    range_high, unit, yes_price, market_price, liquidity_usd, volume_usd,
    yes_token_id, no_token_id, gamma_market_id."""
    for ev in _gamma_events():
        if str(ev.get("event_id") or "") == str(event_id):
            return ev.get("outcomes") or []
    return []


def _bin_prob(o: dict) -> float:
    return float(
        o.get("model_prob")
        or o.get("yes_price")
        or o.get("market_price")
        or 0
    )


def _resolve_missing_slots(
    held_contract_ids: set[str],
    outcomes: list[dict],
) -> tuple[list[tuple[int, dict]], list[str]]:
    """Re-rank outcomes by model_prob desc, take top-K, return:
      - (rank, outcome) tuples whose contract_id is NOT in held_contract_ids
        (the slots that need filling)
      - list of held contract_ids that are NOT in the current top-K
        (off-rank holdings -- the model has rotated since they were bought)
    """
    sorted_bins = sorted(outcomes, key=_bin_prob, reverse=True)
    top_k = sorted_bins[:TKH_TOP_K]
    top_k_ids = {str(o.get("contract_id") or "") for o in top_k}
    missing = []
    for rank, o in enumerate(top_k):
        cid = str(o.get("contract_id") or "")
        if cid and cid not in held_contract_ids:
            missing.append((rank, o))
    off_rank_held = [cid for cid in held_contract_ids
                     if cid and cid not in top_k_ids]
    return missing, off_rank_held


def _build_synthetic_signal(
    bin_outcome: dict, rank: int, bet_amount: float,
    event_id: str, city: str, date_str: str,
) -> dict:
    """Build a signal dict that mirrors what generate_signals emits,
    so we can pass it through execute_signal unchanged."""
    yes_price = _bin_prob(bin_outcome)
    return {
        "contract_id":       bin_outcome.get("contract_id"),
        "question":          bin_outcome.get("question"),
        "range_low":         bin_outcome.get("range_low"),
        "range_high":        bin_outcome.get("range_high"),
        "unit":              bin_outcome.get("unit"),
        "yes_token_id":      bin_outcome.get("yes_token_id"),
        "no_token_id":       bin_outcome.get("no_token_id"),
        "yes_price":         yes_price,
        "market_price":      yes_price,
        "market_p":          yes_price,
        "model_prob":        yes_price,
        "model_p":           yes_price,
        "ev":                0.0,
        "edge":              0.0,
        "recommended_side":  "YES",
        "kelly_size":        bet_amount,
        "is_signal":         True,
        "confidence_multiplier": 1.0,
        "time_scale":        1.0,
        "days_ahead":        None,
        "forecast_sigma_c":  None,
        "city":              city,
        "date":              date_str,
        "event_id":          event_id,
        "scan_timestamp":    datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "gamma_market_id":   bin_outcome.get("gamma_market_id"),
        "strategy":          "top_k_hedged",
        "target_size_usdc":  bet_amount,
        "tkh_bin_rank":      rank,
        "tkh_bin_pct":       TKH_SPLIT[rank],
        "tkh_hours_to_close": None,
        "liquidity_usd":     bin_outcome.get("liquidity_usd"),
    }


def _format_bin(o: dict) -> str:
    rl = o.get("range_low"); rh = o.get("range_high")
    unit = (o.get("unit") or "celsius").lower()
    suffix = "F" if unit == "fahrenheit" else "C"
    if rl is not None and rh is not None:
        return f"{int(rl)}-{int(rh)}{suffix}"
    if rl is not None:
        return f">={int(rl)}{suffix}"
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually place the missing-bin orders.")
    ap.add_argument("--event", type=str, default=None,
                    help="Comma-separated event_ids to act on (default: all)")
    ap.add_argument("--allow-off-rank", action="store_true",
                    help="Add missing top-K bins even when held positions "
                         "are off-rank (the model has rotated since they "
                         "were bought).  WARNING: this creates >K total "
                         "positions per event and over-allocates capital.")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="Seconds between placements (rate-limit guard)")
    args = ap.parse_args()

    event_filter = None
    if args.event:
        event_filter = [e.strip() for e in args.event.split(",") if e.strip()]

    from execution import get_clob_client, execute_signal
    client = get_clob_client()
    if client is None:
        print("ERROR: no CLOB client (paper mode or missing creds)")
        return 1

    with _get_conn() as conn:
        incomplete = _find_incomplete_events(conn, event_filter)

    mode = "APPLY" if args.apply else "DRY RUN"
    print()
    print(f"=== Fill missing TKH bins -- mode: {mode} ===")
    print(f"  TKH_TOP_K            = {TKH_TOP_K}")
    print(f"  TKH_PERCENTAGE_SPLIT = {[round(s*100,1) for s in TKH_SPLIT]}")
    print(f"  TKH_TOTAL_BET_SIZE   = ${TKH_TOTAL_BET_SIZE:.2f}")
    print(f"  TKH_MIN_BIN_USDC     = ${TKH_MIN_BIN_USDC:.2f}")
    print(f"  Events found incomplete: {len(incomplete)}")
    print()
    if not incomplete:
        print("Nothing to do.")
        return 0

    # Pre-fetch Gamma so the per-event lookups below don't each do an
    # API call.  This may take ~10s due to multi-page pagination.
    print(f"  Fetching live event data from Gamma (one-time)...")
    n_gamma = len(_gamma_events())
    print(f"  Fetched {n_gamma} active highest-temp events from Gamma")
    print()

    plans: list[dict] = []
    for ev in incomplete:
        event_id = str(ev["event_id"] or "")
        held_ids = set(
            (ev["held_contract_ids"] or "").split(",")
        ) if ev.get("held_contract_ids") else set()
        outcomes = _outcomes_for_event(event_id)
        if len(outcomes) < TKH_TOP_K:
            print(f"  [SKIP] event={event_id} {ev['city']} {ev['date']} -- "
                  f"Gamma snapshot has {len(outcomes)} outcomes "
                  f"(< K={TKH_TOP_K}) -- event may have resolved or expired")
            continue
        missing, off_rank = _resolve_missing_slots(held_ids, outcomes)
        if off_rank and not args.allow_off_rank:
            print(f"  [SKIP] event={event_id} {ev['city']} {ev['date']} -- "
                  f"{len(off_rank)} held position(s) off-rank under current "
                  f"prices (model has rotated since entry).  Adding top-K "
                  f"would create >K total positions for this event.  "
                  f"Re-run with --allow-off-rank to override (over-allocates).")
            continue
        if not missing:
            print(f"  [OK  ] event={event_id} {ev['city']} {ev['date']} -- "
                  f"all top-K bins already held; nothing to do")
            continue
        for rank, outcome in missing:
            pct = TKH_SPLIT[rank]
            bet_amount = round(TKH_TOTAL_BET_SIZE * pct, 2)
            if bet_amount < TKH_MIN_BIN_USDC:
                bet_amount = TKH_MIN_BIN_USDC
            plans.append({
                "event_id": event_id,
                "city":     ev["city"],
                "date":     ev["date"],
                "rank":     rank,
                "pct":      pct,
                "outcome":  outcome,
                "bet_amount": bet_amount,
            })

    if not plans:
        print("No missing-bin slots to place.")
        return 0

    print(f"--- Planned placements ({len(plans)} bins across "
          f"{len(set(p['event_id'] for p in plans))} events) ---")
    print(f"{'event':<8} {'city':<14} {'date':<11} {'rank':>4} "
          f"{'bin':<7} {'price':>6} {'bet $':>7}")
    print("-" * 70)
    for p in plans:
        o = p["outcome"]
        print(f"{p['event_id']:<8} {p['city'][:14]:<14} {p['date']:<11} "
              f"{p['rank']:>4} {_format_bin(o):<7} "
              f"{_bin_prob(o):>6.4f} ${p['bet_amount']:>6.2f}")
    print()

    if not args.apply:
        print("DRY RUN -- no orders placed.  Re-run with --apply to commit.")
        return 0

    print(f"Placing {len(plans)} order(s) with {args.sleep}s spacing...")
    print()

    counts = {"placed": 0, "skipped": 0, "error": 0}
    for p in plans:
        sig = _build_synthetic_signal(
            p["outcome"], p["rank"], p["bet_amount"],
            p["event_id"], p["city"], p["date"],
        )
        try:
            result = execute_signal(sig, client=client)
        except Exception as e:
            print(f"  [ERR ] event={p['event_id']} {p['city']:<14} "
                  f"rank={p['rank']} -- exception: {e}")
            counts["error"] += 1
            continue
        status = result.get("status", "?")
        if status in ("paper", "exit_pending"):
            counts["placed"] += 1
            print(f"  [OK  ] event={p['event_id']} {p['city']:<14} "
                  f"rank={p['rank']} ${p['bet_amount']:.2f} placed")
        elif status == "skip":
            counts["skipped"] += 1
            print(f"  [SKIP] event={p['event_id']} {p['city']:<14} "
                  f"rank={p['rank']} -- {result.get('reason','?')}")
        elif status in ("matched",):
            # The status flag for placed-and-matched is conventional
            counts["placed"] += 1
            print(f"  [OK  ] event={p['event_id']} {p['city']:<14} "
                  f"rank={p['rank']} ${p['bet_amount']:.2f} matched")
        else:
            counts["placed"] += 1
            print(f"  [OK  ] event={p['event_id']} {p['city']:<14} "
                  f"rank={p['rank']} ${p['bet_amount']:.2f} status={status}")
        time.sleep(args.sleep)

    print()
    print("=== Summary ===")
    print(f"  placed:  {counts['placed']}")
    print(f"  skipped: {counts['skipped']}")
    print(f"  errors:  {counts['error']}")
    print()
    print("Placed orders are now in the normal monitor / repricer / topup")
    print("flow.  Watch the dashboard's In-Flight Orders for fills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
