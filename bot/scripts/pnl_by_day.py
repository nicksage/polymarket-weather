"""
pnl_by_day.py — Per-event P&L breakdown for events resolving on a given date.

Groups closed positions by their event `date` (the day the underlying weather
market resolves, NOT the day the order was placed) and reconstructs
per-event payouts: which bin won, what the bot paid, what it got back, and
where each bin's P&L came from.

Heuristic for "winning bin" (event_resolutions table is empty in production):
  * exit_price >= 0.97  → likely a YES-side bin that hit TP near $1
  * exit_reason contains 'TAKE_PROFIT' → explicit TP exit
  * for NO-side bins, mirror the heuristic in NO-price terms (rare for TKH)
A position can be marked WINNER even when realized < cost — that just means
the bot bought in late at a high price.  At most one winner per event.

Still-open positions on the date are shown separately with current MTM so
you can see what's still in flight.

Usage:
    cd bot
    python -m scripts.pnl_by_day                       # today
    python -m scripts.pnl_by_day --date 2026-05-04     # specific date
    python -m scripts.pnl_by_day --range 2026-05-04 2026-05-07
    python -m scripts.pnl_by_day --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date as _date, datetime, timedelta, timezone

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


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


def _is_winning(exit_price: float | None, exit_reason: str | None,
                side: str) -> bool:
    """Heuristic: did this bin resolve YES?  See module docstring."""
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


def _fetch_for_dates(conn, dates: list[str], include_paper: bool) -> dict:
    """Return {date: {(city, event_id): {closed: [...], open: [...]}}}."""
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"
    placeholders = ",".join("?" * len(dates))

    rows = conn.execute(
        f"""
        SELECT id, contract_id, event_id, city, date, side, status,
               shares, size_usdc, entry_price, entry_time,
               exit_price, exit_time, pnl, exit_reason,
               current_price, unrealized_pnl,
               range_low, range_high, unit, strategy
        FROM positions
        WHERE date IN ({placeholders})
          {paper_clause}
          AND shares IS NOT NULL
          AND entry_price IS NOT NULL
        """,
        tuple(dates),
    ).fetchall()

    by_date: dict[str, dict[tuple, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"closed": [], "open": []})
    )
    for r in rows:
        d = dict(r)
        key = (d["city"] or "?", d["event_id"] or f"_{d['city']}_{d['date']}")
        bucket = "open" if d["status"] == "open" else "closed"
        by_date[d["date"]][key][bucket].append(d)
    return by_date


def _build_event_summary(city: str, event_id: str,
                          closed: list[dict], open_: list[dict]) -> dict:
    bins: list[dict] = []
    cost_total    = 0.0
    payout_total  = 0.0
    realized_total = 0.0
    open_cost_total = 0.0
    open_mtm_total  = 0.0
    open_upnl_total = 0.0
    winner_label = None

    for p in closed:
        # IMPORTANT: shares is zeroed out on exit for many closed positions
        # (execution.py clears it when the SELL fill confirms).  Always use
        # size_usdc for cost basis; derive entry-time share count from
        # size_usdc/entry_price for payout math.
        entry    = float(p["entry_price"])
        size_in  = float(p["size_usdc"]) if p["size_usdc"] is not None else 0.0
        shares_at_entry = (size_in / entry) if entry > 0 else 0.0
        exit_pr  = float(p["exit_price"]) if p["exit_price"] is not None else None
        cost     = size_in
        payout   = shares_at_entry * exit_pr if exit_pr is not None else 0.0
        realized = float(p["pnl"]) if p["pnl"] is not None else (payout - cost)
        is_win  = _is_winning(exit_pr, p["exit_reason"], p["side"])
        if is_win and winner_label is None:
            winner_label = _bin_label(p["range_low"], p["range_high"], p["unit"])

        cost_total     += cost
        payout_total   += payout
        realized_total += realized

        bins.append({
            "pid":          p["id"],
            "status":       "closed",
            "side":         p["side"],
            "label":        _bin_label(p["range_low"], p["range_high"], p["unit"]),
            "shares":       round(shares_at_entry, 4),
            "entry_price":  round(entry, 4),
            "exit_price":   round(exit_pr, 4) if exit_pr is not None else None,
            "cost":         round(cost, 4),
            "payout":       round(payout, 4),
            "realized_pnl": round(realized, 4),
            "is_winner":    is_win,
            "exit_reason":  (p["exit_reason"] or "")[:48],
        })

    for p in open_:
        entry  = float(p["entry_price"])
        # Open positions normally have shares populated, but be defensive
        # in case the monitor hasn't reconciled yet — derive from size_usdc.
        shares = float(p["shares"]) if p["shares"] else (
            float(p["size_usdc"]) / entry if (p["size_usdc"] and entry > 0) else 0.0
        )
        cur    = p["current_price"]
        cost   = shares * entry
        if cur is not None:
            cur_side = float(cur)
            if (p["side"] or "YES").upper() == "NO":
                cur_side = 1.0 - cur_side
            mtm = shares * cur_side
        else:
            cur_side = entry
            mtm      = cost
        upnl = mtm - cost

        open_cost_total += cost
        open_mtm_total  += mtm
        open_upnl_total += upnl

        bins.append({
            "pid":           p["id"],
            "status":        "open",
            "side":          p["side"],
            "label":         _bin_label(p["range_low"], p["range_high"], p["unit"]),
            "shares":        round(shares, 4),
            "entry_price":   round(entry, 4),
            "current_price": round(cur_side, 4),
            "cost":          round(cost, 4),
            "current_value": round(mtm, 4),
            "unrealized_pnl": round(upnl, 4),
            "is_winner":     False,
        })

    bins.sort(key=lambda b: (-int(b.get("is_winner", False)), -b.get("payout", 0) or 0))

    return {
        "city":               city,
        "event_id":           event_id,
        "winner_bin":         winner_label,
        "bins_total":         len(bins),
        "bins_closed":        len(closed),
        "bins_open":          len(open_),
        "closed_cost":        round(cost_total, 4),
        "closed_payout":      round(payout_total, 4),
        "closed_realized":    round(realized_total, 4),
        "open_cost":          round(open_cost_total, 4),
        "open_mtm":           round(open_mtm_total, 4),
        "open_unrealized":    round(open_upnl_total, 4),
        "bins":               bins,
    }


def _build_day_summary(d: str, events: dict) -> dict:
    rows = [_build_event_summary(c, eid, v["closed"], v["open"])
            for (c, eid), v in events.items()]
    rows.sort(key=lambda e: (e["city"], e["event_id"]))

    closed_cost     = sum(e["closed_cost"]     for e in rows)
    closed_payout   = sum(e["closed_payout"]   for e in rows)
    closed_realized = sum(e["closed_realized"] for e in rows)
    open_cost       = sum(e["open_cost"]       for e in rows)
    open_mtm        = sum(e["open_mtm"]        for e in rows)
    open_upnl       = sum(e["open_unrealized"] for e in rows)

    n_events_resolved = sum(1 for e in rows if e["bins_closed"] > 0)
    n_winners_found   = sum(1 for e in rows if e["winner_bin"] is not None)

    return {
        "date":              d,
        "events_total":      len(rows),
        "events_resolved":   n_events_resolved,
        "events_with_winner": n_winners_found,
        "closed_cost":       round(closed_cost, 4),
        "closed_payout":     round(closed_payout, 4),
        "closed_realized":   round(closed_realized, 4),
        "open_cost":         round(open_cost, 4),
        "open_mtm":          round(open_mtm, 4),
        "open_unrealized":   round(open_upnl, 4),
        "events":            rows,
    }


def compute(dates: list[str], include_paper: bool = False) -> dict:
    with _get_conn() as conn:
        by_date = _fetch_for_dates(conn, dates, include_paper)
    days = [_build_day_summary(d, by_date.get(d, {})) for d in dates]
    return {
        "dates":           dates,
        "days":            days,
        "totals": {
            "closed_cost":     round(sum(d["closed_cost"]     for d in days), 4),
            "closed_payout":   round(sum(d["closed_payout"]   for d in days), 4),
            "closed_realized": round(sum(d["closed_realized"] for d in days), 4),
            "open_cost":       round(sum(d["open_cost"]       for d in days), 4),
            "open_mtm":        round(sum(d["open_mtm"]        for d in days), 4),
            "open_unrealized": round(sum(d["open_unrealized"] for d in days), 4),
        },
    }


def _fmt_usd(x: float) -> str:
    sign = "-" if x < -0.005 else " "
    return f"{sign}${abs(x):>9,.2f}"


def _print_day(day: dict, verbose: bool) -> None:
    print()
    print("=" * 88)
    print(f"  {day['date']}  —  {day['events_total']} event(s), "
          f"{day['events_resolved']} resolved, "
          f"{day['events_with_winner']} winner identified")
    print("=" * 88)
    print(f"  Closed bins:  cost {_fmt_usd(day['closed_cost'])}   "
          f"payout {_fmt_usd(day['closed_payout'])}   "
          f"realized {_fmt_usd(day['closed_realized'])}")
    if day["open_cost"] > 0:
        print(f"  Open bins:    cost {_fmt_usd(day['open_cost'])}   "
              f"MTM    {_fmt_usd(day['open_mtm'])}   "
              f"unreal.  {_fmt_usd(day['open_unrealized'])}")
    print(f"  Day total P&L (closed+unrealized): "
          f"{_fmt_usd(day['closed_realized'] + day['open_unrealized'])}")

    for e in day["events"]:
        win = f"WINNER={e['winner_bin']}" if e["winner_bin"] else "winner=?"
        print()
        print(f"  ── {e['city']:14s}  {win:18s}  "
              f"({e['bins_total']} bins: {e['bins_closed']}c/{e['bins_open']}o)")
        print(f"     closed: cost {_fmt_usd(e['closed_cost'])}  "
              f"payout {_fmt_usd(e['closed_payout'])}  "
              f"P&L {_fmt_usd(e['closed_realized'])}", end="")
        if e["bins_open"]:
            print(f"   |  open: cost {_fmt_usd(e['open_cost'])}  "
                  f"MTM {_fmt_usd(e['open_mtm'])}  "
                  f"uPnL {_fmt_usd(e['open_unrealized'])}")
        else:
            print()

        if not verbose:
            continue

        for b in e["bins"]:
            mark = "★" if b["is_winner"] else " "
            if b["status"] == "closed":
                line = (f"     {mark} pid={b['pid']:>5d} {b['side']:>3} "
                        f"{b['label']:>10s} sh={b['shares']:>7.2f} "
                        f"@{b['entry_price']:>5.3f}→{b['exit_price']:>5.3f}  "
                        f"cost {_fmt_usd(b['cost'])} "
                        f"payout {_fmt_usd(b['payout'])} "
                        f"P&L {_fmt_usd(b['realized_pnl'])}")
                if b["exit_reason"]:
                    line += f"  [{b['exit_reason']}]"
            else:
                line = (f"       pid={b['pid']:>5d} {b['side']:>3} "
                        f"{b['label']:>10s} sh={b['shares']:>7.2f} "
                        f"@{b['entry_price']:>5.3f}→{b['current_price']:>5.3f}  "
                        f"cost {_fmt_usd(b['cost'])} "
                        f"MTM {_fmt_usd(b['current_value'])} "
                        f"uPnL {_fmt_usd(b['unrealized_pnl'])}  [open]")
            print(line)


def _print_report(result: dict, verbose: bool) -> None:
    for day in result["days"]:
        _print_day(day, verbose)

    if len(result["days"]) > 1:
        t = result["totals"]
        total_pnl = t["closed_realized"] + t["open_unrealized"]
        print()
        print("=" * 88)
        print(f"  RANGE TOTALS  ({result['dates'][0]} → {result['dates'][-1]})")
        print("=" * 88)
        print(f"  Closed:  cost {_fmt_usd(t['closed_cost'])}   "
              f"payout {_fmt_usd(t['closed_payout'])}   "
              f"realized {_fmt_usd(t['closed_realized'])}")
        if t["open_cost"] > 0:
            print(f"  Open:    cost {_fmt_usd(t['open_cost'])}   "
                  f"MTM    {_fmt_usd(t['open_mtm'])}   "
                  f"unreal.  {_fmt_usd(t['open_unrealized'])}")
        print(f"  TOTAL P&L (realized + unrealized): {_fmt_usd(total_pnl)}")


def _expand_dates(args) -> list[str]:
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
    if args.date:
        return [args.date]
    return [_date.today().isoformat()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--date",
                   help="Event date YYYY-MM-DD (default: today)")
    p.add_argument("--range", nargs=2, metavar=("START", "END"),
                   help="Inclusive date range, both YYYY-MM-DD")
    p.add_argument("--include-paper", action="store_true",
                   help="Include paper-trade positions (default: live only)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-bin rows under each event")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of formatted text")
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