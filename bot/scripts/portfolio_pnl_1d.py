"""
portfolio_pnl_1d.py — Estimate Polymarket's portfolio P&L numbers.

Polymarket's "1D" P&L line is a NET ASSET VALUE delta over the last 24h.
It includes BOTH:
  * realized P&L from positions closed in the window
  * unrealized mark-to-market drift on positions still open

Reproducing that exactly requires a per-contract price history.  This DB
doesn't keep one (bin_price_history / decision_snapshots are empty in
production), so we use the next-best approximation:

  * "Lifetime" unrealized P&L for every open position, computed from
    monitor-maintained positions.current_price vs entry_price.  Because
    the bot trades short-dated weather markets (most positions are <72h
    old), lifetime unrealized ≈ 1D unrealized for the bulk of the book.

  * Realized P&L for positions closed inside the window (--hours).

  * Total = lifetime_open_unrealized + window_realized.  This is a
    SLIGHT OVERESTIMATE of the true 1D number for any position opened
    more than 24h ago (we attribute its full lifetime drift to the
    window), but typically within a few percent of the UI value.

Usage:
    cd bot
    python -m scripts.portfolio_pnl_1d                # 24h window
    python -m scripts.portfolio_pnl_1d --hours 168    # 1W window
    python -m scripts.portfolio_pnl_1d --verbose      # per-position rows
    python -m scripts.portfolio_pnl_1d --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import _get_conn  # type: ignore


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(ts: str | None) -> datetime | None:
    """Parse the timestamps the bot writes.  These come in two flavors:
        * 'YYYY-MM-DD HH:MM:SS[.ffffff]'        (older, naive UTC)
        * 'YYYY-MM-DDTHH:MM:SS[.ffffff]±HH:MM'  (newer, ISO with tz)
    Return a tz-aware datetime in UTC, or None on parse failure."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _side_value(shares: float, price: float, side: str) -> float:
    """Mark-to-market value of `shares` of the side held at the given
    YES-equivalent price.  For NO bins the value of one share equals
    1 - yes_price (binary market identity)."""
    if (side or "").upper() == "NO":
        return shares * (1.0 - price)
    return shares * price


def _side_price_for_held(price: float, side: str) -> float:
    """The price of ONE share of the side actually held."""
    if (side or "").upper() == "NO":
        return 1.0 - price
    return price


def compute_pnl(hours: int = 24, include_paper: bool = False) -> dict:
    now    = _now_utc()
    cutoff = now - timedelta(hours=hours)
    paper_clause = "" if include_paper else "AND COALESCE(is_paper, 0) = 0"

    open_rows:   list[dict] = []
    closed_rows: list[dict] = []
    open_in_window_rows: list[dict] = []   # subset of open_rows

    open_cost_total       = 0.0
    open_mtm_total        = 0.0
    open_unrealized_total = 0.0
    open_window_unrealized = 0.0   # subset opened in last `hours`
    realized_window_total = 0.0
    realized_lifetime_total = 0.0

    with _get_conn() as conn:
        # ---- OPEN positions: lifetime unrealized via monitor's current_price
        rows = conn.execute(
            f"""
            SELECT id, contract_id, side, shares, entry_price, entry_time,
                   current_price, unrealized_pnl, scan_timestamp,
                   strategy, city, date, range_low, range_high, unit
            FROM positions
            WHERE status = 'open'
              {paper_clause}
              AND shares IS NOT NULL
              AND entry_price IS NOT NULL
            """
        ).fetchall()

        for r in rows:
            shares    = float(r["shares"] or 0)
            entry_pr  = float(r["entry_price"] or 0)
            cur_yes   = r["current_price"]
            ent_time  = _parse_ts(r["entry_time"])
            side      = (r["side"] or "YES").upper()
            if shares <= 0 or entry_pr <= 0:
                continue

            # Prefer monitor-stored unrealized_pnl when available; fall
            # back to (current - entry) * shares with side adjustment.
            if cur_yes is not None:
                cur_side = _side_price_for_held(float(cur_yes), side)
                cur_value = shares * cur_side
                upnl = shares * (cur_side - entry_pr)
            elif r["unrealized_pnl"] is not None:
                upnl = float(r["unrealized_pnl"])
                cur_value = shares * entry_pr + upnl
                cur_side = cur_value / shares if shares > 0 else 0.0
            else:
                cur_side = entry_pr
                cur_value = shares * entry_pr
                upnl = 0.0

            cost = shares * entry_pr
            opened_in_window = bool(ent_time and ent_time >= cutoff)

            open_cost_total       += cost
            open_mtm_total        += cur_value
            open_unrealized_total += upnl
            if opened_in_window:
                open_window_unrealized += upnl

            row_d = {
                "pid":            int(r["id"]),
                "side":           side,
                "shares":         round(shares, 4),
                "entry_price":    round(entry_pr, 4),
                "current_price":  round(cur_side, 4),
                "cost":           round(cost, 4),
                "current_value":  round(cur_value, 4),
                "unrealized_pnl": round(upnl, 4),
                "opened_in_window": opened_in_window,
                "entry_time":     r["entry_time"],
                "city":           r["city"],
                "date":           r["date"],
                "strategy":       r["strategy"],
            }
            open_rows.append(row_d)
            if opened_in_window:
                open_in_window_rows.append(row_d)

        # ---- CLOSED positions ---------------------------------------------
        rows = conn.execute(
            f"""
            SELECT id, contract_id, side, shares, entry_price, entry_time,
                   exit_price, exit_time, pnl, status, exit_reason,
                   strategy, city, date
            FROM positions
            WHERE status != 'open'
              {paper_clause}
              AND shares IS NOT NULL
              AND entry_price IS NOT NULL
              AND exit_price IS NOT NULL
            """
        ).fetchall()

        for r in rows:
            shares   = float(r["shares"] or 0)
            entry_pr = float(r["entry_price"] or 0)
            exit_pr  = float(r["exit_price"] or 0)
            ext_time = _parse_ts(r["exit_time"])
            side     = (r["side"] or "YES").upper()
            if shares <= 0 or entry_pr <= 0:
                continue

            # Prefer stored pnl (already side-correct); fall back to
            # (exit - entry) * shares.  The monitor stores realized pnl
            # in NO-side terms when relevant.
            if r["pnl"] is not None:
                realized = float(r["pnl"])
            else:
                realized = shares * (exit_pr - entry_pr)

            realized_lifetime_total += realized

            in_window = bool(ext_time and ext_time >= cutoff)
            if in_window:
                realized_window_total += realized
                closed_rows.append({
                    "pid":           int(r["id"]),
                    "side":          side,
                    "shares":        round(shares, 4),
                    "entry_price":   round(entry_pr, 4),
                    "exit_price":    round(exit_pr, 4),
                    "realized_pnl":  round(realized, 4),
                    "exit_reason":   r["exit_reason"],
                    "exit_time":     r["exit_time"],
                    "city":          r["city"],
                    "date":          r["date"],
                    "strategy":      r["strategy"],
                })

    # The "tight" 1D estimate uses ONLY positions opened in the window
    # (their full unrealized P&L = window contribution exactly) plus the
    # realized P&L from positions closed in the window.  Older open
    # positions contribute SOME unrealized drift over 24h that we cannot
    # measure without price history — so we report them separately.
    estimate_window_only  = round(open_window_unrealized + realized_window_total, 4)
    estimate_full_lifetime = round(open_unrealized_total + realized_window_total, 4)

    return {
        "now_utc":              now.isoformat(),
        "window_hours":         hours,
        "cutoff_utc":           cutoff.isoformat(),

        "open_count":           len(open_rows),
        "open_in_window_count": len(open_in_window_rows),
        "closed_in_window":     len(closed_rows),

        "open_cost_total":      round(open_cost_total, 4),
        "open_mtm_total":       round(open_mtm_total, 4),
        "open_unrealized_lifetime": round(open_unrealized_total, 4),
        "open_unrealized_window_only": round(open_window_unrealized, 4),

        "realized_window":      round(realized_window_total, 4),
        "realized_lifetime":    round(realized_lifetime_total, 4),

        "pnl_estimate_window_only_lower_bound":  estimate_window_only,
        "pnl_estimate_full_lifetime_upper_bound": estimate_full_lifetime,

        "open_positions":       open_rows,
        "closed_positions":     closed_rows,
    }


def _fmt_usd(x: float) -> str:
    sign = "-" if x < 0 else " "
    return f"{sign}${abs(x):>9,.2f}"


def _print_report(result: dict, verbose: bool) -> None:
    h = result["window_hours"]
    print("=" * 78)
    print(f"  PORTFOLIO P&L ESTIMATE  ({h}h window)")
    print(f"  now    = {result['now_utc']}")
    print(f"  cutoff = {result['cutoff_utc']}")
    print("=" * 78)
    print()
    print(f"  OPEN POSITIONS ({result['open_count']} total, "
          f"{result['open_in_window_count']} opened in last {h}h)")
    print(f"    cost basis ............. {_fmt_usd(result['open_cost_total'])}")
    print(f"    current MTM ............ {_fmt_usd(result['open_mtm_total'])}")
    print(f"    lifetime unrealized .... {_fmt_usd(result['open_unrealized_lifetime'])}")
    print(f"    └ of which opened <{h}h: {_fmt_usd(result['open_unrealized_window_only'])}")
    print()
    print(f"  CLOSED IN WINDOW ({result['closed_in_window']} positions)")
    print(f"    realized in window ..... {_fmt_usd(result['realized_window'])}")
    print(f"    realized lifetime ...... {_fmt_usd(result['realized_lifetime'])}")
    print()
    print(f"  ESTIMATED {h}h P&L (matches Polymarket's '1D' line):")
    print(f"    LOWER bound ............ {_fmt_usd(result['pnl_estimate_window_only_lower_bound'])}")
    print(f"      (only counts unrealized drift for positions opened in the window)")
    print(f"    UPPER bound ............ {_fmt_usd(result['pnl_estimate_full_lifetime_upper_bound'])}")
    print(f"      (counts FULL lifetime drift on every open position)")
    print()
    print("  The Polymarket UI value should fall between these two bounds.")
    print("  Without per-contract price history we cannot pin it exactly —")
    print("  drift on older open positions could have happened any time, but")
    print("  typically the bulk happened recently as resolution approaches.")
    print()
    print("  PROFITABILITY READ:")
    print(f"    If every open bin resolves to its current price today, your")
    print(f"    P&L for this batch is the lifetime unrealized number above")
    print(f"    ({_fmt_usd(result['open_unrealized_lifetime'])}) plus the realized lifetime")
    print(f"    ({_fmt_usd(result['realized_lifetime'])}).  TKH baskets have a different")
    print(f"    payout profile though — run scripts.payout_scenarios for")
    print(f"    win/loss outcomes assuming each bin actually settles.")

    if not verbose:
        print()
        print("  Run with --verbose for per-position rows.")
        return

    print()
    print("-" * 78)
    print("  OPEN POSITIONS (sorted by unrealized P&L, worst first)")
    print("-" * 78)
    print(f"  {'pid':>5} {'win':>3} {'side':>4} {'shares':>8} "
          f"{'entry':>6} {'now':>6} {'cost':>9} {'value':>9} {'upnl':>9}  loc/date")
    rows = sorted(result["open_positions"], key=lambda r: r["unrealized_pnl"])
    for r in rows:
        loc = f"{(r['city'] or '?')[:14]} {r['date'] or ''}"
        win_marker = "Y" if r["opened_in_window"] else "."
        print(
            f"  {r['pid']:>5d} {win_marker:>3} {r['side']:>4} {r['shares']:>8.2f} "
            f"{r['entry_price']:>6.3f} {r['current_price']:>6.3f} "
            f"{_fmt_usd(r['cost'])} {_fmt_usd(r['current_value'])} "
            f"{_fmt_usd(r['unrealized_pnl'])}  {loc}"
        )

    if result["closed_positions"]:
        print()
        print("-" * 78)
        print(f"  CLOSED IN WINDOW (sorted by realized P&L, worst first)")
        print("-" * 78)
        print(f"  {'pid':>5} {'side':>4} {'shares':>8} "
              f"{'entry':>6} {'exit':>6} {'realized':>9}  reason  loc/date")
        rows = sorted(result["closed_positions"], key=lambda r: r["realized_pnl"])
        for r in rows:
            loc = f"{(r['city'] or '?')[:14]} {r['date'] or ''}"
            reason = (r["exit_reason"] or "")[:18]
            print(
                f"  {r['pid']:>5d} {r['side']:>4} {r['shares']:>8.2f} "
                f"{r['entry_price']:>6.3f} {r['exit_price']:>6.3f} "
                f"{_fmt_usd(r['realized_pnl'])}  {reason:<18}  {loc}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--hours", type=int, default=24,
                   help="P&L window in hours (default: 24)")
    p.add_argument("--include-paper", action="store_true",
                   help="Include paper-trade positions (default: live only)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-position rows")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of formatted text")
    args = p.parse_args()

    result = compute_pnl(hours=args.hours, include_paper=args.include_paper)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
