"""
strategy1_backtest.py — Replay Strategy 1 (post-peak confirmation) against
historical Polymarket bin-price snapshots + station observations.

For every resolved (city, date) where we have BOTH price history (from
bin_price_history) AND hourly station temps (from station_obs.db), walks
through the day hour-by-hour and asks:

  At what hour would the strategy have fired BUY?
  What YES price would we have paid for the bin?
  Did that bin actually win the day?
  → entry P&L = (1.0 if won else 0.0) - entry_price

Strategy simulation differs from live in one place: we don't have a
historical forecast cache, so "predicted peak hour" is approximated by
the OBSERVED peak hour as of the trigger moment.  The strategy fires
once observations have been stable past their own peak for
--hours-after-peak.  This is slightly OPTIMISTIC vs live (where forecast
peak time may diverge from observed peak) — interpret as upper bound.

Safety gates from live strategy are preserved:
  * skip if observed_max == day_max but more obs still to come (peak
    actually still rising — caught here by "stable past peak" check)
  * skip if target bin priced < 0.05 (market_disagrees)
  * skip if target bin priced >= --threshold (already priced in, no edge)

Prerequisites:
  1. Bot's main DB must have populated `bin_price_history` (price_ws.py
     writes these every 2 min when running).
  2. `data/station_obs.db` must exist — run:
       python -m scripts.station_obs_pull --days 60
     before running this backtest.
  3. Bin metadata source — looked up in this order:
       (a) `temp_outcomes` table (best — has range_low/high for ALL bins)
       (b) `positions` table   (fallback — only bins the bot traded)
     City-days where neither has the bin boundaries are skipped (counted).

Usage:
    cd bot
    python -m scripts.strategy1_backtest                    # last 60 days
    python -m scripts.strategy1_backtest --days 30
    python -m scripts.strategy1_backtest --start 2026-05-01 --end 2026-06-01
    python -m scripts.strategy1_backtest --threshold 0.85
    python -m scripts.strategy1_backtest --hours-after-peak 2
    python -m scripts.strategy1_backtest --city Madrid Tokyo Wuhan
    python -m scripts.strategy1_backtest --csv data/strategy1_backtest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS  # type: ignore
from config       import DB_PATH        # type: ignore

STATION_DB = os.path.join(_BOT_DIR, "data", "station_obs.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Bin metadata resolution
# ---------------------------------------------------------------------------

def load_bin_metadata(conn) -> dict[str, dict]:
    """Return {contract_id: {range_low, range_high, unit, city, date}}.
    Tries temp_outcomes first (covers all bins the bot ever saw), falls
    back to positions (only traded bins) for anything missing."""
    out: dict[str, dict] = {}

    # temp_outcomes is the richest source.  Schema differs slightly across
    # bot versions; guard with try/except.
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT
                o.contract_id, o.range_low, o.range_high, o.unit,
                e.city, e.date
            FROM temp_outcomes o
            JOIN temp_events  e ON o.event_row_id = e.id
            WHERE o.contract_id IS NOT NULL
              AND o.range_low IS NOT NULL OR o.range_high IS NOT NULL
            """
        ).fetchall()
        for r in rows:
            cid = r[0]
            if cid and cid not in out:
                out[cid] = {
                    "range_low":  r[1], "range_high": r[2],
                    "unit":       r[3] or "celsius",
                    "city":       r[4], "date": r[5],
                }
    except sqlite3.OperationalError as e:
        log.debug(f"temp_outcomes lookup failed: {e}")

    # Fill gaps from positions table
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT contract_id, range_low, range_high, unit, city, date
            FROM positions
            WHERE contract_id IS NOT NULL
              AND (range_low IS NOT NULL OR range_high IS NOT NULL)
            """
        ).fetchall()
        for r in rows:
            cid = r[0]
            if cid and cid not in out:
                out[cid] = {
                    "range_low":  r[1], "range_high": r[2],
                    "unit":       r[3] or "celsius",
                    "city":       r[4], "date": r[5],
                }
    except sqlite3.OperationalError as e:
        log.debug(f"positions lookup failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Bin matching (mirrors live strategy)
# ---------------------------------------------------------------------------

def _bin_contains(low, high, unit: str, temp_c: float) -> bool:
    if unit and unit.lower() == "fahrenheit":
        t = temp_c * 9 / 5 + 32
    else:
        t = temp_c
    if low is None and high is not None: return t <= high
    if low is not None and high is None: return t >= low
    if low is not None and high is not None:
        if low == high:
            return abs(t - low) < 0.5
        return low <= t <= high
    return False


def _bin_label(low, high, unit: str) -> str:
    suffix = "F" if (unit or "celsius").lower() == "fahrenheit" else "C"
    if low is None and high is not None: return f"≤{int(high)}°{suffix}"
    if low is not None and high is None: return f"≥{int(low)}°{suffix}"
    if low is not None and high is not None:
        if int(low) == int(high): return f"{int(low)}°{suffix}"
        return f"{int(low)}–{int(high)}°{suffix}"
    return "?"


# ---------------------------------------------------------------------------
# Simulation per (city, date)
# ---------------------------------------------------------------------------

def simulate_day(city: str, date_str: str,
                  hourly_temps: list[tuple[int, float]],
                  bins_meta: list[dict],
                  bin_price_history: dict[str, list[tuple[str, float]]],
                  threshold: float, hours_after_peak: int) -> dict | None:
    """Returns trade dict if strategy would have fired, else None.

    hourly_temps:    [(hour_local, temp_c), ...]  for this day
    bins_meta:       [{contract_id, range_low, range_high, unit}, ...]
                     for this city-date (sourced from temp_outcomes/positions)
    bin_price_history: {contract_id: [(recorded_at_iso, yes_price), ...]}
                     for this city-date (sourced from bin_price_history table)
    """
    if len(hourly_temps) < 18:
        return None

    day_max = max(t for _, t in hourly_temps)
    actual_winning = next(
        (b for b in bins_meta
         if _bin_contains(b["range_low"], b["range_high"],
                          b.get("unit", "celsius"), day_max)),
        None,
    )
    actual_winning_label = (_bin_label(actual_winning["range_low"],
                                         actual_winning["range_high"],
                                         actual_winning.get("unit", "celsius"))
                             if actual_winning else "?")

    by_hour = {h: t for h, t in hourly_temps}

    # Walk forward through the day from 13:00 (one hour after our 12:00
    # after_hour floor) and look for the first trigger.
    for trigger_h in range(13, 24):
        # Observations available up to and including trigger_h
        obs_so_far = [(h, t) for (h, t) in hourly_temps if h <= trigger_h]
        if len(obs_so_far) < 6:
            continue

        observed_max_so_far = max(t for _, t in obs_so_far)
        observed_peak_hour  = max(h for h, t in obs_so_far
                                   if abs(t - observed_max_so_far) < 1e-6)

        # Stability gate: peak must be hours_after_peak ago or older,
        # i.e. we've seen at least N hours of temps without exceeding it.
        if trigger_h < observed_peak_hour + hours_after_peak:
            continue

        # Match the observed max to a bin
        target = next(
            (b for b in bins_meta
             if _bin_contains(b["range_low"], b["range_high"],
                              b.get("unit", "celsius"), observed_max_so_far)),
            None,
        )
        if target is None:
            continue
        cid = target["contract_id"]
        price_series = bin_price_history.get(cid, [])
        if not price_series:
            continue

        # Find the price snapshot nearest to trigger_h local time.
        # Snapshots are stored with UTC recorded_at; we approximate by
        # using just the hour-of-day match — for backtest purposes this
        # is fine since snapshots are every ~2 min.  More robust: filter
        # by date_local + hour_local in UTC equivalent, but the
        # approximation is good enough here.
        target_hour = trigger_h
        best_snapshot = None
        best_diff = 1e9
        for ts, yp in price_series:
            try:
                snap_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            diff = abs(snap_dt.hour - target_hour)
            if diff < best_diff:
                best_diff = diff
                best_snapshot = (ts, yp)
        if best_snapshot is None:
            continue
        _, entry_price = best_snapshot

        # Safety gates
        if entry_price < 0.05:
            return {
                "city": city, "date": date_str,
                "fired_at_hour": trigger_h,
                "observed_max":  round(observed_max_so_far, 2),
                "target_bin":    _bin_label(target["range_low"],
                                              target["range_high"],
                                              target.get("unit", "celsius")),
                "entry_price":   round(entry_price, 4),
                "day_max":       round(day_max, 2),
                "winning_bin":   actual_winning_label,
                "won":           False,
                "pnl":           None,
                "action":        "SKIP_MARKET_DISAGREES",
            }
        if entry_price >= threshold:
            return {
                "city": city, "date": date_str,
                "fired_at_hour": trigger_h,
                "observed_max":  round(observed_max_so_far, 2),
                "target_bin":    _bin_label(target["range_low"],
                                              target["range_high"],
                                              target.get("unit", "celsius")),
                "entry_price":   round(entry_price, 4),
                "day_max":       round(day_max, 2),
                "winning_bin":   actual_winning_label,
                "won":           actual_winning is not None and target["contract_id"] == actual_winning["contract_id"],
                "pnl":           None,
                "action":        "SKIP_PRICED_IN",
            }

        won = actual_winning is not None and target["contract_id"] == actual_winning["contract_id"]
        return {
            "city": city, "date": date_str,
            "fired_at_hour": trigger_h,
            "observed_max":  round(observed_max_so_far, 2),
            "target_bin":    _bin_label(target["range_low"],
                                          target["range_high"],
                                          target.get("unit", "celsius")),
            "entry_price":   round(entry_price, 4),
            "day_max":       round(day_max, 2),
            "winning_bin":   actual_winning_label,
            "won":           won,
            "pnl":           round((1.0 if won else 0.0) - entry_price, 4),
            "action":        "BUY_YES",
        }

    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_station_temps(station_db: str, cities: list[str], start: str, end: str
                        ) -> dict[tuple[str, str], list[tuple[int, float]]]:
    out: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    placeholders = ",".join("?" * len(cities))
    with sqlite3.connect(station_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT city, date_local, hour_local, temp_c
            FROM station_obs
            WHERE city IN ({placeholders})
              AND date_local BETWEEN ? AND ?
              AND temp_c IS NOT NULL
            ORDER BY city, date_local, hour_local
            """,
            (*cities, start, end),
        ).fetchall()
    for r in rows:
        out[(r["city"], r["date_local"])].append((r["hour_local"], r["temp_c"]))
    return out


def load_price_history(conn, start: str, end: str
                        ) -> dict[tuple[str, str, str], list[tuple[str, float]]]:
    """Returns {(city, date, contract_id): [(recorded_at, yes_price), ...]}"""
    out: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
    rows = conn.execute(
        """
        SELECT city, date, contract_id, recorded_at, yes_price
        FROM bin_price_history
        WHERE date BETWEEN ? AND ?
          AND yes_price IS NOT NULL
        ORDER BY recorded_at
        """,
        (start, end),
    ).fetchall()
    for r in rows:
        out[(r[0], r[1], r[2])].append((r[3], float(r[4])))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=60,
                   help="Look back N days from today (default: 60)")
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end",   help="End date YYYY-MM-DD (default: yesterday)")
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Max yes_price for a BUY (default: 0.90)")
    p.add_argument("--hours-after-peak", type=int, default=1,
                   help="Wait N hours after observed peak before firing (default: 1)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all mapped)")
    p.add_argument("--station-db", default=STATION_DB,
                   help=f"Station obs DB (default: {STATION_DB})")
    p.add_argument("--db", default=DB_PATH,
                   help=f"Main bot DB with bin_price_history (default: {DB_PATH})")
    p.add_argument("--csv", help="Write per-trade rows to this CSV path")
    args = p.parse_args()

    # Date range — exclude today since markets haven't settled
    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today() - timedelta(days=1))
    if args.start:
        start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_d = end_d - timedelta(days=args.days)
    start, end = start_d.isoformat(), end_d.isoformat()

    cities = list(args.city) if args.city else list(CITY_STATIONS.keys())
    log.info(f"Backtest window: {start} → {end}  ({(end_d - start_d).days + 1} days)")
    log.info(f"Cities: {len(cities)}  | station_db={args.station_db}  | bot_db={args.db}")

    # 1. Station temps
    if not os.path.exists(args.station_db):
        log.error(f"Station DB not found: {args.station_db}")
        log.error(f"Run: python -m scripts.station_obs_pull --days {args.days}")
        return 1
    station_temps = load_station_temps(args.station_db, cities, start, end)
    log.info(f"Loaded station temps for {len(station_temps):,} city-days")

    # 2. Bin metadata + price history from bot DB
    with sqlite3.connect(args.db) as conn:
        bin_meta = load_bin_metadata(conn)
        log.info(f"Loaded {len(bin_meta):,} bin metadata records")
        price_history = load_price_history(conn, start, end)
        log.info(f"Loaded price-history for {len(price_history):,} bin-days")

    # Pre-group bins by (city, date) for fast lookup
    bins_by_event: dict[tuple[str, str], list[dict]] = defaultdict(list)
    prices_by_event: dict[tuple[str, str], dict[str, list]] = defaultdict(dict)
    for cid, meta in bin_meta.items():
        if meta.get("city") and meta.get("date"):
            bins_by_event[(meta["city"], meta["date"])].append({
                "contract_id": cid, **meta,
            })
    for (city, date_str, cid), series in price_history.items():
        prices_by_event[(city, date_str)][cid] = series

    # 3. Run simulation
    trades: list[dict] = []
    no_bin_meta = 0
    no_prices   = 0
    no_obs      = 0

    for (city, date_str), temps in station_temps.items():
        if city not in cities:
            continue
        bins  = bins_by_event.get((city, date_str), [])
        if not bins:
            no_bin_meta += 1
            continue
        prices = prices_by_event.get((city, date_str), {})
        if not prices:
            no_prices += 1
            continue
        # Filter bins to those with price history
        bins_with_prices = [b for b in bins if b["contract_id"] in prices]
        if not bins_with_prices:
            no_prices += 1
            continue

        result = simulate_day(city, date_str, temps, bins_with_prices,
                              prices, args.threshold, args.hours_after_peak)
        if result:
            trades.append(result)

    # 4. Summary
    bought = [t for t in trades if t["action"] == "BUY_YES"]
    won    = [t for t in bought if t["won"]]
    lost   = [t for t in bought if not t["won"]]
    skipped_disagree = [t for t in trades if t["action"] == "SKIP_MARKET_DISAGREES"]
    skipped_priced   = [t for t in trades if t["action"] == "SKIP_PRICED_IN"]
    skipped_priced_won = [t for t in skipped_priced if t["won"]]

    print()
    print("=" * 86)
    print(f"  STRATEGY 1 BACKTEST  ({start} → {end})")
    print("=" * 86)
    print(f"  Threshold:           yes_price < {args.threshold}")
    print(f"  Wait after peak:     {args.hours_after_peak}h")
    print()
    print(f"  COVERAGE")
    print(f"    city-days w/ station temps: {len(station_temps):>5,d}")
    print(f"    skipped: no bin metadata:    {no_bin_meta:>5,d}")
    print(f"    skipped: no price history:   {no_prices:>5,d}")
    print()
    print(f"  TRIGGERED: {len(trades):,d} city-days reached a strategy decision")
    print(f"    BUY_YES:                    {len(bought):>5,d}")
    print(f"    SKIP (market_disagrees):    {len(skipped_disagree):>5,d}")
    print(f"    SKIP (priced_in ≥ {args.threshold}):    {len(skipped_priced):>5,d}")
    print()

    if bought:
        avg_entry = sum(t["entry_price"] for t in bought) / len(bought)
        win_rate  = len(won) / len(bought)
        total_pnl = sum(t["pnl"] for t in bought)
        avg_pnl   = total_pnl / len(bought)
        roi       = total_pnl / sum(t["entry_price"] for t in bought)
        print(f"  BUY_YES TRADES — RESULTS")
        print(f"    n trades:                   {len(bought):>5,d}")
        print(f"    win rate:                   {100*win_rate:>5.1f}%  ({len(won)}W / {len(lost)}L)")
        print(f"    avg entry price:            ${avg_entry:.3f}")
        print(f"    avg P&L per $1 staked:      ${avg_pnl:+.3f}")
        print(f"    total P&L per $1 per trade: ${total_pnl:+.2f}")
        print(f"    ROI on cost basis:          {100*roi:+.1f}%")
        print()
        print(f"  ENTRY PRICE DISTRIBUTION")
        for lo, hi in [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60),
                       (0.60, 0.80), (0.80, 0.90)]:
            bucket = [t for t in bought if lo <= t["entry_price"] < hi]
            if bucket:
                wr = sum(1 for t in bucket if t["won"]) / len(bucket)
                ap = sum(t["pnl"] for t in bucket) / len(bucket)
                print(f"    {lo:.2f}-{hi:.2f}:  n={len(bucket):>4d}  "
                      f"win {100*wr:>5.1f}%  avg P&L ${ap:+.3f}")

    if skipped_priced:
        print()
        sp_win = sum(1 for t in skipped_priced if t["won"]) / len(skipped_priced)
        print(f"  COMPARISON: events we SKIPPED because already priced in")
        print(f"    n: {len(skipped_priced)}  win rate (if we had bought): {100*sp_win:.1f}%")
        print(f"    avg yes_price at trigger: "
              f"${sum(t['entry_price'] for t in skipped_priced)/len(skipped_priced):.3f}")
        print(f"    (this is the {args.threshold}+ priced-in group; high win rate = "
              f"justifiably so)")

    if args.csv and trades:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        fields = ["city", "date", "action", "fired_at_hour", "observed_max",
                  "target_bin", "entry_price", "day_max", "winning_bin",
                  "won", "pnl"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({k: t.get(k) for k in fields})
        print()
        print(f"  Wrote {len(trades)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())