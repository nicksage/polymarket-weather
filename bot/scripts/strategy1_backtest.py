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
import json
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
# HTML dashboard
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #f3f4f6; color: #111827; font-size: 13px; }
header { background: #111827; color: #f3f4f6; padding: 12px 20px;
         display: flex; align-items: center; justify-content: space-between; }
header h1 { margin: 0; font-size: 17px; }
header .meta { font-family: monospace; font-size: 11px; color: #9ca3af; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 10px; padding: 14px 18px; background: white;
        border-bottom: 1px solid #e5e7eb; }
.kpi { background: #f9fafb; padding: 10px 14px; border-radius: 6px;
       border-left: 4px solid #4338ca; }
.kpi .label { font-size: 10px; color: #6b7280; text-transform: uppercase;
              letter-spacing: 0.3px; }
.kpi .val { font-size: 22px; font-weight: 600; margin-top: 2px; font-family: monospace; }
.kpi.pos { border-left-color: #16a34a; }
.kpi.neg { border-left-color: #dc2626; }
.kpi.pos .val { color: #166534; }
.kpi.neg .val { color: #991b1b; }
.charts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;
          padding: 12px 18px; }
.chart-card { background: white; padding: 12px 14px; border-radius: 6px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.chart-card h3 { margin: 0 0 8px; font-size: 12px; color: #4b5563;
                 text-transform: uppercase; letter-spacing: 0.4px; }
.filters { background: white; padding: 10px 18px; border-top: 1px solid #e5e7eb;
           border-bottom: 1px solid #e5e7eb; display: flex; gap: 18px;
           flex-wrap: wrap; align-items: center; font-size: 12px; }
.filters label { font-weight: 600; color: #374151; margin-right: 4px; }
.filters select, .filters input { padding: 3px 6px; font-size: 12px; }
.filters .count { color: #6b7280; font-family: monospace; margin-left: auto; }
table { width: calc(100% - 36px); margin: 0 18px 18px; background: white;
        border-collapse: collapse; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-radius: 6px; overflow: hidden; font-size: 12px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #f3f4f6; }
th { background: #f9fafb; color: #6b7280; font-weight: 600; font-size: 10px;
     text-transform: uppercase; letter-spacing: 0.3px; cursor: pointer;
     user-select: none; }
th:hover { background: #eef2ff; }
th.sorted-asc::after { content: " ▲"; color: #4338ca; }
th.sorted-desc::after { content: " ▼"; color: #4338ca; }
tr.WIN { background: #f0fdf4; }
tr.LOSS { background: #fef2f2; }
tr.SKIP_MARKET_DISAGREES { background: #fefce8; }
tr.SKIP_PRICED_IN { background: #f3f4f6; color: #6b7280; }
td.num { font-family: monospace; text-align: right; }
td .pill { display: inline-block; padding: 1px 6px; border-radius: 10px;
           font-size: 10px; font-weight: 600; }
.pill.WIN { background: #dcfce7; color: #166534; }
.pill.LOSS { background: #fee2e2; color: #991b1b; }
.pill.SKIP_MARKET_DISAGREES { background: #fef3c7; color: #854d0e; }
.pill.SKIP_PRICED_IN { background: #e5e7eb; color: #374151; }
"""


def _bar_chart_svg(buckets: list[tuple[str, int, float | None]],
                    title: str, width: int = 480, height: int = 200,
                    overlay_pct: bool = False) -> str:
    """buckets = [(label, count, optional_pct), ...].  If overlay_pct, draw
    a small percentage label above each bar."""
    if not buckets:
        return "<svg viewBox='0 0 480 200'></svg>"
    pad_l, pad_r, pad_t, pad_b = 36, 8, 14, 30
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b
    max_n = max(c for _, c, _ in buckets) or 1
    bar_w = iw / len(buckets) * 0.78
    gap   = iw / len(buckets) * 0.22
    s = [f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"]
    # y grid
    for i in range(4):
        v = max_n * (i / 3)
        y = pad_t + (1 - i / 3) * ih
        s.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' "
                 f"stroke='#e5e7eb' stroke-width='0.5'/>")
        s.append(f"<text x='{pad_l-3}' y='{y+3:.1f}' font-size='9' text-anchor='end' "
                 f"fill='#9ca3af' font-family='monospace'>{int(v)}</text>")
    for i, (label, n, pct) in enumerate(buckets):
        x = pad_l + i * (bar_w + gap) + gap/2
        bar_h = (n / max_n) * ih if max_n else 0
        y = pad_t + ih - bar_h
        s.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
                 f"height='{bar_h:.1f}' fill='#4338ca' rx='2'/>")
        if overlay_pct and pct is not None:
            s.append(f"<text x='{x+bar_w/2:.1f}' y='{y-3:.1f}' font-size='9' "
                     f"text-anchor='middle' fill='#16a34a' font-weight='600' "
                     f"font-family='monospace'>{100*pct:.0f}%</text>")
        s.append(f"<text x='{x+bar_w/2:.1f}' y='{height-pad_b+12}' font-size='9' "
                 f"text-anchor='middle' fill='#6b7280' font-family='monospace'>"
                 f"{label}</text>")
    s.append("</svg>")
    return "".join(s)


def _line_chart_svg(points: list[tuple[str, float]],
                     title: str, width: int = 480, height: int = 200) -> str:
    """Cumulative-line chart: [(x_label, y_value), ...]"""
    if not points:
        return "<svg viewBox='0 0 480 200'></svg>"
    pad_l, pad_r, pad_t, pad_b = 42, 10, 14, 28
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b
    ys = [y for _, y in points]
    y_lo, y_hi = min(0, min(ys)), max(0, max(ys))
    if y_hi == y_lo:
        y_hi = y_lo + 1
    def yp(v):
        return pad_t + (1 - (v - y_lo) / (y_hi - y_lo)) * ih
    def xp(i):
        return pad_l + (i / max(1, len(points)-1)) * iw
    s = [f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"]
    # zero line if range crosses it
    if y_lo < 0 < y_hi:
        s.append(f"<line x1='{pad_l}' y1='{yp(0):.1f}' x2='{width-pad_r}' y2='{yp(0):.1f}' "
                 f"stroke='#9ca3af' stroke-width='0.8' stroke-dasharray='3,3'/>")
    # y grid + labels
    for i in range(4):
        v = y_lo + (y_hi - y_lo) * (i / 3)
        y = yp(v)
        s.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' "
                 f"stroke='#e5e7eb' stroke-width='0.5'/>")
        s.append(f"<text x='{pad_l-3}' y='{y+3:.1f}' font-size='9' text-anchor='end' "
                 f"fill='#9ca3af' font-family='monospace'>${v:+.1f}</text>")
    # line
    pts = " ".join(f"{xp(i):.1f},{yp(v):.1f}" for i, (_, v) in enumerate(points))
    color = "#16a34a" if ys[-1] >= 0 else "#dc2626"
    s.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.6'/>")
    # x labels: show start/middle/end
    for idx in [0, len(points)//2, len(points)-1]:
        if 0 <= idx < len(points):
            s.append(f"<text x='{xp(idx):.1f}' y='{height-pad_b+12}' font-size='9' "
                     f"text-anchor='middle' fill='#6b7280' font-family='monospace'>"
                     f"{points[idx][0]}</text>")
    s.append("</svg>")
    return "".join(s)


def render_dashboard(trades: list[dict], coverage: dict, args) -> str:
    """Build a single self-contained HTML dashboard."""
    bought = [t for t in trades if t["action"] == "BUY_YES"]
    won    = [t for t in bought if t["won"]]
    lost   = [t for t in bought if not t["won"]]
    skipped_disagree = [t for t in trades if t["action"] == "SKIP_MARKET_DISAGREES"]
    skipped_priced   = [t for t in trades if t["action"] == "SKIP_PRICED_IN"]

    if bought:
        avg_entry = sum(t["entry_price"] for t in bought) / len(bought)
        win_rate  = len(won) / len(bought)
        total_pnl = sum(t["pnl"] for t in bought)
        roi       = total_pnl / sum(t["entry_price"] for t in bought) if bought else 0
    else:
        avg_entry = win_rate = total_pnl = roi = 0

    # Entry price histogram (buckets of 0.10) — with win-rate overlay
    buckets = []
    edges = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        bucket = [t for t in bought if lo <= t["entry_price"] < hi]
        wr = (sum(1 for t in bucket if t["won"]) / len(bucket)) if bucket else None
        buckets.append((f"{lo:.1f}", len(bucket), wr))
    price_hist_svg = _bar_chart_svg(buckets, "entry price", overlay_pct=True)

    # Win rate by trigger hour
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for t in bought:
        by_hour[t["fired_at_hour"]].append(t)
    hour_buckets = []
    for h in sorted(by_hour.keys()):
        b = by_hour[h]
        wr = sum(1 for t in b if t["won"]) / len(b)
        hour_buckets.append((f"{h}", len(b), wr))
    hour_chart_svg = _bar_chart_svg(hour_buckets, "win rate by hour",
                                      overlay_pct=True)

    # Win rate by city — top 10 by count
    by_city: dict[str, list[dict]] = defaultdict(list)
    for t in bought:
        by_city[t["city"]].append(t)
    top_cities = sorted(by_city.items(), key=lambda x: -len(x[1]))[:10]
    city_buckets = []
    for city, b in top_cities:
        wr = sum(1 for t in b if t["won"]) / len(b)
        city_buckets.append((city[:10], len(b), wr))
    city_chart_svg = _bar_chart_svg(city_buckets, "win rate by city",
                                      overlay_pct=True)

    # Cumulative P&L over trades (ordered by date)
    sorted_bought = sorted(bought, key=lambda t: t["date"])
    cum_pnl = []
    running = 0.0
    for t in sorted_bought:
        running += t["pnl"]
        cum_pnl.append((t["date"][5:], round(running, 2)))   # MM-DD label
    pnl_chart_svg = _line_chart_svg(cum_pnl, "cumulative P&L")

    # Trades JSON for table
    trades_json = json.dumps(trades, default=str, separators=(",", ":"))

    pnl_class = "pos" if total_pnl >= 0 else "neg"
    roi_class = "pos" if roi >= 0 else "neg"

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Strategy 1 backtest dashboard</title>
<style>{DASHBOARD_CSS}</style></head><body>
<header>
  <div>
    <h1>Strategy 1 — Post-peak confirmation backtest</h1>
    <div class='meta'>window {args.start_actual} → {args.end_actual} &nbsp;·&nbsp;
        threshold &lt; {args.threshold} &nbsp;·&nbsp;
        wait {args.hours_after_peak}h after peak</div>
  </div>
  <div class='meta'>
    coverage: {coverage['station_days']:,} city-days · {coverage['no_bin_meta']:,} no bins
    · {coverage['no_prices']:,} no prices
  </div>
</header>

<div class='kpis'>
  <div class='kpi'><div class='label'>Trades fired</div>
    <div class='val'>{len(bought):,}</div></div>
  <div class='kpi pos'><div class='label'>Wins</div>
    <div class='val'>{len(won):,}</div></div>
  <div class='kpi neg'><div class='label'>Losses</div>
    <div class='val'>{len(lost):,}</div></div>
  <div class='kpi'><div class='label'>Win rate</div>
    <div class='val'>{100*win_rate:.1f}%</div></div>
  <div class='kpi'><div class='label'>Avg entry</div>
    <div class='val'>${avg_entry:.3f}</div></div>
  <div class='kpi {pnl_class}'><div class='label'>Total P&amp;L per $1</div>
    <div class='val'>${total_pnl:+.2f}</div></div>
  <div class='kpi {roi_class}'><div class='label'>ROI</div>
    <div class='val'>{100*roi:+.1f}%</div></div>
  <div class='kpi'><div class='label'>Skipped (mkt disagree)</div>
    <div class='val'>{len(skipped_disagree):,}</div></div>
  <div class='kpi'><div class='label'>Skipped (priced in)</div>
    <div class='val'>{len(skipped_priced):,}</div></div>
</div>

<div class='charts'>
  <div class='chart-card'><h3>Entry price distribution (green % = win rate)</h3>
    {price_hist_svg}</div>
  <div class='chart-card'><h3>Trades + win rate by trigger hour (local)</h3>
    {hour_chart_svg}</div>
  <div class='chart-card'><h3>Top 10 cities by trade count + win rate</h3>
    {city_chart_svg}</div>
  <div class='chart-card'><h3>Cumulative P&amp;L (per $1 staked)</h3>
    {pnl_chart_svg}</div>
</div>

<div class='filters'>
  <div><label>Status</label>
    <select id='f-status'>
      <option value=''>All</option>
      <option value='WIN'>Wins only</option>
      <option value='LOSS'>Losses only</option>
      <option value='BUY_YES'>Buys only (WIN + LOSS)</option>
      <option value='SKIP_MARKET_DISAGREES'>Skipped: market disagrees</option>
      <option value='SKIP_PRICED_IN'>Skipped: priced in</option>
    </select></div>
  <div><label>City</label>
    <select id='f-city'><option value=''>All</option></select></div>
  <div><label>Min entry</label>
    <input id='f-min' type='number' step='0.05' min='0' max='1' style='width:60px'></div>
  <div><label>Max entry</label>
    <input id='f-max' type='number' step='0.05' min='0' max='1' style='width:60px'></div>
  <div class='count' id='count'>—</div>
</div>

<table id='trades'>
  <thead><tr>
    <th data-key='date'>Date</th>
    <th data-key='city'>City</th>
    <th data-key='fired_at_hour'>Hr</th>
    <th data-key='observed_max'>Obs max</th>
    <th data-key='target_bin'>Target bin</th>
    <th data-key='entry_price'>Entry</th>
    <th data-key='day_max'>Day high</th>
    <th data-key='winning_bin'>Winning bin</th>
    <th data-key='_status'>Status</th>
    <th data-key='pnl'>P&amp;L</th>
  </tr></thead>
  <tbody id='tbody'></tbody>
</table>

<script>
const TRADES = {trades_json};
const $ = id => document.getElementById(id);

// Populate city dropdown
const cities = [...new Set(TRADES.map(t => t.city))].sort();
const sel = $('f-city');
for (const c of cities) {{
  const o = document.createElement('option'); o.value = o.textContent = c; sel.appendChild(o);
}}

function statusLabel(t) {{
  if (t.action === 'BUY_YES') return t.won ? 'WIN' : 'LOSS';
  return t.action;
}}

let SORT_KEY = 'date', SORT_DIR = -1;

function row(t) {{
  const st = statusLabel(t);
  const fmt = (v, d=3) => v == null ? '' : (typeof v === 'number' ? v.toFixed(d) : v);
  const pnl = t.pnl == null ? '' : (t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(3);
  return `<tr class='${{st}}'>
    <td>${{t.date}}</td>
    <td>${{t.city}}</td>
    <td class='num'>${{t.fired_at_hour}}</td>
    <td class='num'>${{fmt(t.observed_max, 1)}}°</td>
    <td>${{t.target_bin}}</td>
    <td class='num'>${{fmt(t.entry_price, 3)}}</td>
    <td class='num'>${{fmt(t.day_max, 1)}}°</td>
    <td>${{t.winning_bin}}</td>
    <td><span class='pill ${{st}}'>${{st}}</span></td>
    <td class='num'>${{pnl}}</td>
  </tr>`;
}}

function render() {{
  const status = $('f-status').value;
  const city = $('f-city').value;
  const lo = parseFloat($('f-min').value);
  const hi = parseFloat($('f-max').value);
  let rows = TRADES.filter(t => {{
    if (status === 'WIN' && (t.action !== 'BUY_YES' || !t.won)) return false;
    if (status === 'LOSS' && (t.action !== 'BUY_YES' || t.won)) return false;
    if (status === 'BUY_YES' && t.action !== 'BUY_YES') return false;
    if (['SKIP_MARKET_DISAGREES','SKIP_PRICED_IN'].includes(status) && t.action !== status) return false;
    if (city && t.city !== city) return false;
    if (!isNaN(lo) && t.entry_price < lo) return false;
    if (!isNaN(hi) && t.entry_price > hi) return false;
    return true;
  }});
  rows.sort((a, b) => {{
    const k = SORT_KEY;
    let av = k === '_status' ? statusLabel(a) : a[k];
    let bv = k === '_status' ? statusLabel(b) : b[k];
    if (av == null) av = '';
    if (bv == null) bv = '';
    if (typeof av === 'number') return SORT_DIR * (av - bv);
    return SORT_DIR * String(av).localeCompare(String(bv));
  }});
  $('count').textContent = rows.length + ' / ' + TRADES.length;
  $('tbody').innerHTML = rows.map(row).join('');
  document.querySelectorAll('th').forEach(th => {{
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? 'sorted-asc' : 'sorted-desc');
  }});
}}

document.querySelectorAll('th').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.key;
    if (SORT_KEY === k) SORT_DIR = -SORT_DIR;
    else {{ SORT_KEY = k; SORT_DIR = 1; }}
    render();
  }});
}});
['f-status','f-city','f-min','f-max'].forEach(id =>
  $(id).addEventListener('input', render));
render();
</script>
</body></html>"""


def serve_html(path: str, port: int) -> None:
    """Serve the directory containing `path` over HTTP on `port` forever.
    Prints SSH-tunnel command for Windows access."""
    import http.server, socketserver
    serve_dir = os.path.dirname(os.path.abspath(path)) or "."
    fname     = os.path.basename(path)
    os.chdir(serve_dir)
    handler = http.server.SimpleHTTPRequestHandler

    print()
    print("=" * 78)
    print(f"  Serving {path} on port {port}")
    print(f"  Local URL:  http://localhost:{port}/{fname}")
    print()
    print(f"  Windows access via SSH tunnel (run this on your Windows machine):")
    print(f"     ssh -L {port}:localhost:{port} <user>@<vps>")
    print(f"  Then open in your browser:")
    print(f"     http://localhost:{port}/{fname}")
    print()
    print(f"  Or, if the VPS port {port} is exposed publicly:")
    print(f"     http://<vps-ip>:{port}/{fname}")
    print()
    print("  Press Ctrl-C to stop.")
    print("=" * 78)

    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


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
    p.add_argument("--html", default=os.path.join(_BOT_DIR, "data",
                                                    "strategy1_dashboard.html"),
                   help="Write self-contained dashboard HTML to this path "
                        "(default: data/strategy1_dashboard.html)")
    p.add_argument("--no-html", action="store_true",
                   help="Skip dashboard HTML generation")
    p.add_argument("--serve", type=int, metavar="PORT",
                   help="After writing, start an HTTP server on PORT so "
                        "you can access the dashboard from a browser "
                        "(use SSH tunnel: ssh -L PORT:localhost:PORT user@vps)")
    args = p.parse_args()

    # Date range — exclude today since markets haven't settled
    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today() - timedelta(days=1))
    if args.start:
        start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_d = end_d - timedelta(days=args.days)
    start, end = start_d.isoformat(), end_d.isoformat()
    args.start_actual = start
    args.end_actual   = end

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

    # Dashboard HTML
    if not args.no_html and trades:
        coverage = {
            "station_days": len(station_temps),
            "no_bin_meta":  no_bin_meta,
            "no_prices":    no_prices,
        }
        html = render_dashboard(trades, coverage, args)
        os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(html)
        size_kb = os.path.getsize(args.html) / 1024
        print(f"  Wrote {size_kb:,.0f} KB dashboard to {args.html}")

    if args.serve:
        if args.no_html or not trades:
            print("  Nothing to serve (--no-html set or no trades).")
            return 0
        serve_html(args.html, args.serve)

    return 0


if __name__ == "__main__":
    sys.exit(main())