"""
verify_pnl.py — Authoritative P&L cross-check from Polymarket activity history.

Reconstructs P&L from the wallet's complete transaction history via the
Polymarket Data API `/activity` endpoint -- the same data source the
Polymarket UI's "History" tab uses.  This is the gold standard: every
BUY, SELL, and REDEEM event with exact USDC amounts and timestamps.

For each event (city + resolution date) we report:
  - All BUY transactions: total shares acquired, total USDC spent
  - All SELL transactions: total shares sold, total USDC received
  - All REDEEM transactions: redemption value (winning shares paying out)
  - REALIZED P&L = sells + redemptions - buys
  - UNREALIZED P&L = current_value of still-held shares - their cost basis
  - NET P&L = realized + unrealized

Per-bin breakdown shows each token (each temperature range) separately.
Per-event totals show the basket-level outcome (winning bin + losing bins).

Compares to the bot DB's recorded P&L so any divergence is visible.

Why this is more accurate than the previous approach:
  - The /positions endpoint only shows CURRENTLY HELD positions.
    Bins that have been sold or lost-at-resolution are no longer there,
    so their realized P&L is invisible to that endpoint.
  - The /activity endpoint includes EVERY transaction in wallet history,
    so closed positions (sold + redeemed) are correctly reconstructed.

Usage:
    cd bot
    python -m scripts.verify_pnl                          # today's date
    python -m scripts.verify_pnl --date 2026-05-04
    python -m scripts.verify_pnl --date 2026-05-04 --by-event
    python -m scripts.verify_pnl --date 2026-05-04 --city Busan
    python -m scripts.verify_pnl --date 2026-05-04 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import httpx
from datetime import datetime, timedelta, timezone
from collections import defaultdict

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


# ---------------------------------------------------------------------------
# Polymarket Data API fetchers
# ---------------------------------------------------------------------------

def _fetch_all_activity(wallet: str, max_pages: int = 50) -> list[dict]:
    """Pull the wallet's full activity history (paginated).  Each item is
    a TRADE (BUY/SELL) or REDEEM event."""
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
            print(f"WARN: activity fetch failed at offset={offset}: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return out


def _fetch_chain_positions(wallet: str) -> dict[str, dict]:
    """token_id -> {size, avg_price, cost_basis, current_value, ...}"""
    if not wallet:
        return {}
    from polymarket import get_data_api_positions
    positions = get_data_api_positions(wallet) or []
    return {p["token_id"]: p for p in positions if p.get("token_id")}


# ---------------------------------------------------------------------------
# DB fetcher
# ---------------------------------------------------------------------------

def _fetch_db_positions(date_str: str, include_paper: bool,
                         city_filter: str | None) -> list[dict]:
    """All bot positions whose date column == date_str."""
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"
    city_clause  = "AND city LIKE ?"  if city_filter else ""
    args: list = [date_str]
    if city_filter:
        args.append(f"%{city_filter}%")
    sql = f"""
        SELECT
            id, strategy, city, date, side, range_low, range_high, unit,
            contract_id, yes_token_id, no_token_id, event_id,
            entry_price, current_price, shares, size_usdc,
            target_size_usdc, status, fill_status, cancelled_reason,
            pnl, pnl_net, unrealized_pnl, entry_fees, exit_fees,
            entry_time, exit_time
        FROM positions
        WHERE date = ?
        {paper_clause} {city_clause}
        ORDER BY city, range_low
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


# ---------------------------------------------------------------------------
# P&L reconstruction
# ---------------------------------------------------------------------------

def _per_token_ledger(activity: list[dict]) -> dict[str, dict]:
    """Bucket every activity item by asset (token_id) and aggregate
    cash flows.  Returns:
        token_id -> {
          buys:     [(ts, size, usdc, price), ...],
          sells:    [(ts, size, usdc, price), ...],
          redeems:  [(ts, size, usdc), ...],
          buy_size, buy_usdc, sell_size, sell_usdc, redeem_usdc,
          conditionId, title,
        }
    """
    out: dict[str, dict] = defaultdict(lambda: {
        "buys": [], "sells": [], "redeems": [],
        "buy_size": 0.0, "buy_usdc": 0.0,
        "sell_size": 0.0, "sell_usdc": 0.0,
        "redeem_size": 0.0, "redeem_usdc": 0.0,
        "conditionId": "", "title": "",
    })
    for it in activity:
        token = str(it.get("asset") or "")
        if not token:
            continue
        bucket = out[token]
        bucket["conditionId"] = it.get("conditionId", "")
        bucket["title"]       = it.get("title", "")
        ts    = int(it.get("timestamp", 0))
        size  = float(it.get("size", 0) or 0)
        usdc  = float(it.get("usdcSize", 0) or 0)
        price = float(it.get("price", 0) or 0)
        ttype = it.get("type", "")
        side  = it.get("side", "")
        if ttype == "TRADE" and side == "BUY":
            bucket["buys"].append((ts, size, usdc, price))
            bucket["buy_size"] += size
            bucket["buy_usdc"] += usdc
        elif ttype == "TRADE" and side == "SELL":
            bucket["sells"].append((ts, size, usdc, price))
            bucket["sell_size"] += size
            bucket["sell_usdc"] += usdc
        elif ttype == "REDEEM":
            bucket["redeems"].append((ts, size, usdc))
            bucket["redeem_size"] += size
            bucket["redeem_usdc"] += usdc
    return dict(out)


def _compute_token_pnl(token_id: str, ledger: dict,
                        chain: dict | None) -> dict:
    """For one token, compute realized + unrealized P&L from its
    activity ledger and current chain holding."""
    buy_size    = float(ledger["buy_size"])
    buy_usdc    = float(ledger["buy_usdc"])
    sell_size   = float(ledger["sell_size"])
    sell_usdc   = float(ledger["sell_usdc"])
    redeem_size = float(ledger["redeem_size"])
    redeem_usdc = float(ledger["redeem_usdc"])

    # Net shares activity says we should hold:
    activity_net_size = buy_size - sell_size - redeem_size

    # Current chain holding (may still hold shares that haven't been
    # sold or redeemed yet, e.g., losing-bin shares post-resolution that
    # are auto-redeemed at $0 -- wallet still shows them but value=0).
    chain_size  = float(chain.get("size", 0))         if chain else 0.0
    chain_value = float(chain.get("current_value", 0)) if chain else 0.0
    chain_avg   = float(chain.get("avg_price", 0))    if chain else 0.0

    # Cost basis attributable to currently-held shares.  Use chain's
    # avg_price if available (most accurate); otherwise approximate
    # from the activity buy weighted-avg.
    if chain_size > 0:
        cost_basis_held = chain_size * chain_avg
        # Cost basis attributable to shares that have left the wallet
        # (sold or redeemed) = total_buy_usdc - cost_basis_held
        # But that's the SAME shares the sells & redeems consumed.
        cost_basis_disposed = buy_usdc - cost_basis_held
    else:
        cost_basis_held = 0.0
        cost_basis_disposed = buy_usdc

    # Realized P&L (chain truth from transaction history):
    realized = sell_usdc + redeem_usdc - cost_basis_disposed

    # Unrealized P&L (current value minus cost basis of held shares):
    unrealized = chain_value - cost_basis_held

    return {
        "token_id":          token_id,
        "title":             ledger["title"],
        "conditionId":       ledger["conditionId"],
        "buy_size":          buy_size,
        "buy_usdc":          buy_usdc,
        "sell_size":         sell_size,
        "sell_usdc":         sell_usdc,
        "redeem_size":       redeem_size,
        "redeem_usdc":       redeem_usdc,
        "chain_size":        chain_size,
        "chain_avg":         chain_avg,
        "chain_value":       chain_value,
        "cost_basis_held":   cost_basis_held,
        "cost_basis_disposed": cost_basis_disposed,
        "realized":          realized,
        "unrealized":        unrealized,
        "net":               realized + unrealized,
        "activity_net_size": activity_net_size,
    }


# ---------------------------------------------------------------------------
# Bin label + DB-position helpers
# ---------------------------------------------------------------------------

def _bin_label(pos: dict) -> str:
    rl = pos.get("range_low"); rh = pos.get("range_high")
    unit = (pos.get("unit") or "celsius").lower()
    suffix = "F" if unit == "fahrenheit" else "C"
    if rl is not None and rh is not None:
        return f"{int(rl)}{suffix}" if int(rl) == int(rh) else f"{int(rl)}-{int(rh)}{suffix}"
    return "?"


def _db_pnl_for_position(pos: dict) -> float:
    """The bot DB's own number for this position -- realised if closed,
    unrealised if open.  Used purely for comparison."""
    pnl_realized = pos.get("pnl")
    pnl_unreal   = pos.get("unrealized_pnl")
    if pnl_realized is not None:
        return float(pnl_realized)
    if pnl_unreal is not None:
        return float(pnl_unreal)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=str,
                    default=datetime.now().strftime("%Y-%m-%d"),
                    help="Resolution date to verify (default: today)")
    ap.add_argument("--by-event", action="store_true",
                    help="Show event-level summary only (skip per-bin detail)")
    ap.add_argument("--include-paper", action="store_true",
                    help="Include paper positions (default: live only)")
    ap.add_argument("--city", type=str, default=None,
                    help="Filter to one city (substring match)")
    ap.add_argument("--json", action="store_true",
                    help="Emit raw JSON for piping")
    args = ap.parse_args()

    from config import WALLET_ADDRESS
    if not WALLET_ADDRESS:
        print("ERROR: WALLET_ADDRESS not configured in .env")
        return 1

    # ---- Fetch sources ------------------------------------------------
    print(f"Fetching DB positions for date={args.date}"
          + (f", city~{args.city!r}" if args.city else "") + "...")
    db_rows = _fetch_db_positions(args.date, args.include_paper, args.city)
    if not db_rows:
        print(f"No positions in DB for date={args.date}")
        return 0
    print(f"  {len(db_rows)} DB positions")

    print(f"Fetching activity history from Polymarket Data API...")
    activity = _fetch_all_activity(WALLET_ADDRESS)
    print(f"  {len(activity)} activity items")

    print(f"Fetching current chain positions...")
    chain_index = _fetch_chain_positions(WALLET_ADDRESS)
    print(f"  {len(chain_index)} currently-held tokens")

    # ---- Build per-token ledger from activity --------------------------
    full_ledger = _per_token_ledger(activity)

    # ---- Filter to tokens that match our DB positions for this date ----
    db_tokens: dict[str, dict] = {}      # token -> db_pos
    for pos in db_rows:
        side = pos.get("side", "YES")
        tok = pos.get("yes_token_id") if side == "YES" else pos.get("no_token_id")
        if tok:
            db_tokens[str(tok)] = pos

    # Compute per-token P&L for every token we track
    enriched: list[dict] = []
    for tok, db_pos in db_tokens.items():
        ledger = full_ledger.get(tok)
        if ledger is None:
            # No activity for this token (never filled or activity not yet
            # indexed by Polymarket).  Show as zero-flow.
            ledger = {
                "buys": [], "sells": [], "redeems": [],
                "buy_size": 0.0, "buy_usdc": 0.0,
                "sell_size": 0.0, "sell_usdc": 0.0,
                "redeem_size": 0.0, "redeem_usdc": 0.0,
                "conditionId": db_pos.get("contract_id", ""),
                "title": db_pos.get("question", ""),
            }
        chain = chain_index.get(tok)
        metrics = _compute_token_pnl(tok, ledger, chain)
        metrics["pid"]      = db_pos["id"]
        metrics["city"]     = db_pos.get("city", "")
        metrics["bin"]      = _bin_label(db_pos)
        metrics["side"]     = db_pos.get("side", "YES")
        metrics["event_id"] = db_pos.get("event_id") or ""
        metrics["status"]   = db_pos.get("status", "")
        metrics["fill_status"] = db_pos.get("fill_status", "")
        metrics["db_pnl"]   = _db_pnl_for_position(db_pos)
        enriched.append(metrics)

    if args.json:
        print(json.dumps(enriched, indent=2, default=str))
        return 0

    # ---- Group by event ------------------------------------------------
    events: dict[str, list[dict]] = defaultdict(list)
    for m in enriched:
        ev = m.get("event_id") or f"_orphan_{m['pid']}"
        events[ev].append(m)

    # ---- Render -------------------------------------------------------
    print()
    print(f"=== Activity-based P&L verification (date={args.date}) ===")
    print(f"  wallet: {WALLET_ADDRESS[:10]}...{WALLET_ADDRESS[-6:]}")
    print(f"  source: Polymarket Data API /activity (gold standard)")
    print()

    grand = {"buy": 0.0, "sell": 0.0, "redeem": 0.0,
             "realized": 0.0, "unrealized": 0.0,
             "net": 0.0, "db": 0.0}

    for event_id in sorted(events.keys()):
        bins = sorted(events[event_id], key=lambda b: b["bin"])
        first = bins[0]
        city = first["city"]

        ev_buy_usdc    = sum(b["buy_usdc"]    for b in bins)
        ev_sell_usdc   = sum(b["sell_usdc"]   for b in bins)
        ev_redeem_usdc = sum(b["redeem_usdc"] for b in bins)
        ev_realized    = sum(b["realized"]    for b in bins)
        ev_unrealized  = sum(b["unrealized"]  for b in bins)
        ev_net         = ev_realized + ev_unrealized
        ev_db_pnl      = sum(b["db_pnl"]      for b in bins)

        grand["buy"]        += ev_buy_usdc
        grand["sell"]       += ev_sell_usdc
        grand["redeem"]     += ev_redeem_usdc
        grand["realized"]   += ev_realized
        grand["unrealized"] += ev_unrealized
        grand["net"]        += ev_net
        grand["db"]         += ev_db_pnl

        print(f"--- {city:<14} event={event_id:<8} "
              f"({len(bins)} bin{'s' if len(bins)!=1 else ''}) ---")

        if not args.by_event:
            print(f"  {'pid':>4} {'bin':<7} "
                  f"{'buys':>16} {'sells':>16} {'redeem':>10} "
                  f"{'real':>9} {'unreal':>9} {'NET':>9}  "
                  f"{'DB':>9}  {'diff':>8}")
            for b in bins:
                buys_s  = (f"{b['buy_size']:>5.2f}sh ${b['buy_usdc']:>5.2f}"
                           if b['buy_size'] else "       --       ")
                sells_s = (f"{b['sell_size']:>5.2f}sh ${b['sell_usdc']:>5.2f}"
                           if b['sell_size'] else "       --       ")
                redeem_s = (f"${b['redeem_usdc']:>7.2f}"
                            if b['redeem_usdc'] else "      --")
                diff = b["net"] - b["db_pnl"]
                print(f"  {b['pid']:>4} {b['bin']:<7} "
                      f"{buys_s:>16} {sells_s:>16} {redeem_s:>10} "
                      f"${b['realized']:>+8.2f} "
                      f"${b['unrealized']:>+8.2f} "
                      f"${b['net']:>+8.2f}  "
                      f"${b['db_pnl']:>+8.2f}  "
                      f"${diff:>+7.2f}")

        diff_event = ev_net - ev_db_pnl
        print(f"  {'EVENT TOTAL':>14}  "
              f"buys=${ev_buy_usdc:.2f}  "
              f"sells=${ev_sell_usdc:.2f}  "
              f"redeem=${ev_redeem_usdc:.2f}  "
              f"REAL=${ev_realized:+.2f}  "
              f"UNREAL=${ev_unrealized:+.2f}  "
              f"NET=${ev_net:+.2f}  "
              f"DB=${ev_db_pnl:+.2f}  "
              f"diff=${diff_event:+.2f}")
        print()

    # ---- Grand summary ------------------------------------------------
    diff_grand = grand["net"] - grand["db"]
    print("=" * 78)
    print(f"GRAND TOTAL for date={args.date}")
    print(f"  Total bought:          ${grand['buy']:>9.2f}")
    print(f"  Total sold:            ${grand['sell']:>9.2f}")
    print(f"  Total redeemed:        ${grand['redeem']:>9.2f}")
    print(f"  Realized P&L:          ${grand['realized']:>+9.2f}  "
          f"(sells + redemptions - cost basis of disposed shares)")
    print(f"  Unrealized P&L:        ${grand['unrealized']:>+9.2f}  "
          f"(current value - cost basis of currently-held shares)")
    print(f"  CHAIN NET P&L:         ${grand['net']:>+9.2f}  "
          f"<-- authoritative")
    print(f"  DB-reported P&L:       ${grand['db']:>+9.2f}")
    print(f"  DIFFERENCE (chain-DB): ${diff_grand:>+9.2f}")
    if abs(diff_grand) > 0.50:
        print()
        print(f"  WARNING: chain vs DB diverges by more than $0.50.")
        print(f"  The chain figure is authoritative -- it's reconstructed")
        print(f"  from every transaction in your wallet history.  Likely")
        print(f"  causes for DB drift:")
        print(f"    - DB cost-basis stale (run repair_share_drift --rebuild-basis)")
        print(f"    - Closed-via-self-heal positions used approximate PnL")
        print(f"    - Auto-close orphan_db marked positions at total loss")
        print(f"      when the chain actually had partial proceeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
