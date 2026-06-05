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

# Default path to the user's standalone price collector DB on the VPS.
# Override with --price-db.  The collector schema is:
#   events           (event_id, city, date, event_title, n_bins, ...)
#   bins             (id, event_id, contract_id, question,
#                     range_low, range_high, unit, yes_token_id, no_token_id)
#   price_snapshots  (id, event_id, contract_id, yes_price,
#                     volume_usd, liquidity_usd, recorded_at)
#   resolutions      (event_id, city, date, winning_contract_id,
#                     winning_range_low, winning_range_high, resolved_at)
DEFAULT_COLLECTOR_DB = os.path.expanduser(
    "~/apps/weather-data/backtest-collector/data/prices.db"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Collector-DB loaders
# (schema confirmed from VPS: ~/apps/weather-data/backtest-collector/data/prices.db)
# ---------------------------------------------------------------------------

def _open_ro(path: str) -> sqlite3.Connection:
    """Read-only, lock-tolerant connection to a DB the collector may be
    actively writing to.  Returns a row-factory connection."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"prices.db not found at {path}")
    conn = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
        timeout=60,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def load_resolved_events(conn, start: str, end: str,
                          cities: list[str]) -> list[dict]:
    """Return list of {event_id, city, date} for events whose `date` is
    in [start, end] AND that have settled (resolved=1)."""
    placeholders = ",".join("?" * len(cities))
    rows = conn.execute(
        f"""
        SELECT event_id, city, date, event_title, n_bins
        FROM events
        WHERE date BETWEEN ? AND ?
          AND resolved = 1
          AND city IN ({placeholders})
        ORDER BY date, city
        """,
        (start, end, *cities),
    ).fetchall()
    return [dict(r) for r in rows]


def load_bins_for_event(conn, event_id: str) -> list[dict]:
    """All bins belonging to an event."""
    rows = conn.execute(
        """
        SELECT contract_id, range_low, range_high, unit
        FROM bins WHERE event_id = ?
        """,
        (event_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def load_snapshots_for_event(conn, event_id: str
                              ) -> dict[str, list[tuple[str, float]]]:
    """Returns {contract_id: [(recorded_at_iso_utc, yes_price), ...]}
    sorted by time.  Uses idx_price_snap_event for instant lookup."""
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    rows = conn.execute(
        """
        SELECT contract_id, recorded_at, yes_price
        FROM price_snapshots
        WHERE event_id = ? AND yes_price IS NOT NULL
        ORDER BY recorded_at
        """,
        (event_id,),
    ).fetchall()
    for r in rows:
        out[r["contract_id"]].append((r["recorded_at"], r["yes_price"]))
    return out


def load_resolution(conn, event_id: str) -> dict | None:
    """The authoritative settlement record for an event, or None if none."""
    r = conn.execute(
        """
        SELECT winning_contract_id, winning_range_low, winning_range_high,
               resolved_at
        FROM resolutions WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    return dict(r) if r else None


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

def simulate_day(city: str, date_str: str, tz: str,
                  hourly_temps: list[tuple[int, float]],
                  bins_meta: list[dict],
                  bin_price_history: dict[str, list[tuple[str, float]]],
                  resolution: dict | None,
                  threshold: float, hours_after_peak: int) -> dict | None:
    """Returns trade dict if strategy would have fired, else None.

    hourly_temps:      [(hour_local, temp_c), ...]
    bins_meta:         [{contract_id, range_low, range_high, unit}, ...]
    bin_price_history: {contract_id: [(recorded_at_iso_utc, yes_price), ...]}
    resolution:        authoritative settlement record (or None) — when
                       present, we use it to determine win/loss instead
                       of inferring the winning bin from the day's max
    tz:                station IANA timezone (e.g. 'Europe/Madrid')
    """
    if len(hourly_temps) < 18:
        return None

    from zoneinfo import ZoneInfo
    station_tz = ZoneInfo(tz)

    day_max = max(t for _, t in hourly_temps)

    # Sanity check — should never happen since obs_so_far ⊆ hourly_temps.
    # If it does, the station_obs data is corrupted for this city-date.
    obs_check = max((t for _, t in hourly_temps), default=None)
    if obs_check is not None and obs_check != day_max:
        pass  # impossible; placeholder

    # Determine the market's unit for this event (all bins of one event
    # share the same unit — F for US markets, C for everyone else).
    market_unit = (bins_meta[0].get("unit") or "celsius") if bins_meta else "celsius"

    # Authoritative winner from resolutions table, if available; otherwise
    # infer from day_max + bin boundaries.
    if resolution:
        winning_contract_id = resolution["winning_contract_id"]
        # Look up the winning bin in our bins_meta to get its unit.  The
        # resolutions table doesn't store unit, but bins_meta does.
        winning_bin_obj = next(
            (b for b in bins_meta if b["contract_id"] == winning_contract_id),
            None,
        )
        winning_unit = ((winning_bin_obj or {}).get("unit")
                         or market_unit)
        winning_label = _bin_label(resolution["winning_range_low"],
                                     resolution["winning_range_high"],
                                     winning_unit)
    else:
        inferred = next(
            (b for b in bins_meta
             if _bin_contains(b["range_low"], b["range_high"],
                              b.get("unit", "celsius"), day_max)),
            None,
        )
        winning_contract_id = inferred["contract_id"] if inferred else None
        winning_label = (_bin_label(inferred["range_low"],
                                      inferred["range_high"],
                                      inferred.get("unit", "celsius"))
                         if inferred else "?")

    # Pre-parse snapshot timestamps into LOCAL hours for trigger matching.
    # Each snapshot's recorded_at is ISO UTC; convert to local datetime once.
    snapshots_by_local_hour: dict[str, dict[int, tuple[str, float]]] = {}
    for cid, series in bin_price_history.items():
        per_hour: dict[int, tuple[str, float]] = {}
        for ts, yp in series:
            try:
                snap_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            snap_local = snap_utc.astimezone(station_tz)
            # Only snapshots on the SAME LOCAL DATE as the market
            if snap_local.strftime("%Y-%m-%d") != date_str:
                continue
            # Keep the snapshot nearest the END of each local hour
            # (i.e. closest to the next top-of-hour boundary)
            h = snap_local.hour
            existing = per_hour.get(h)
            if existing is None:
                per_hour[h] = (ts, yp)
            else:
                # Prefer the one closer to hour:55
                existing_min = datetime.fromisoformat(
                    existing[0].replace("Z", "+00:00")
                ).astimezone(station_tz).minute
                if abs(snap_local.minute - 55) < abs(existing_min - 55):
                    per_hour[h] = (ts, yp)
        snapshots_by_local_hour[cid] = per_hour

    # Walk forward through the day from 13:00 local; look for first trigger.
    for trigger_h in range(13, 24):
        obs_so_far = [(h, t) for (h, t) in hourly_temps if h <= trigger_h]
        if len(obs_so_far) < 6:
            continue

        observed_max_so_far = max(t for _, t in obs_so_far)
        observed_peak_hour  = max(h for h, t in obs_so_far
                                   if abs(t - observed_max_so_far) < 1e-6)

        # Stability gate: peak must be hours_after_peak ago or older
        if trigger_h < observed_peak_hour + hours_after_peak:
            continue

        # Match observed max to a bin
        target = next(
            (b for b in bins_meta
             if _bin_contains(b["range_low"], b["range_high"],
                              b.get("unit", "celsius"), observed_max_so_far)),
            None,
        )
        if target is None:
            continue
        cid = target["contract_id"]
        per_hour = snapshots_by_local_hour.get(cid, {})

        # Find the snapshot for this trigger hour, or the nearest prior one
        snapshot = None
        for h in range(trigger_h, -1, -1):
            if h in per_hour:
                snapshot = per_hour[h]; break
        if snapshot is None:
            continue
        _, entry_price = snapshot

        target_label = _bin_label(target["range_low"], target["range_high"],
                                    target.get("unit", "celsius"))
        won = winning_contract_id is not None and \
              target["contract_id"] == winning_contract_id

        # Display values in market unit (set up once for all return paths)
        if market_unit.lower() == "fahrenheit":
            _obs_disp = observed_max_so_far * 9 / 5 + 32
            _day_disp = day_max            * 9 / 5 + 32
            _unit     = "°F"
        else:
            _obs_disp = observed_max_so_far
            _day_disp = day_max
            _unit     = "°C"

        # Safety gates
        if entry_price < 0.05:
            return {
                "city": city, "date": date_str,
                "fired_at_hour": trigger_h,
                "observed_max":  round(_obs_disp, 2),
                "observed_max_c": round(observed_max_so_far, 2),
                "target_bin":    target_label,
                "entry_price":   round(entry_price, 4),
                "day_max":       round(_day_disp, 2),
                "day_max_c":     round(day_max, 2),
                "winning_bin":   winning_label,
                "unit":          _unit,
                "won":           False,
                "pnl":           None,
                "action":        "SKIP_MARKET_DISAGREES",
            }
        if entry_price >= threshold:
            return {
                "city": city, "date": date_str,
                "fired_at_hour": trigger_h,
                "observed_max":  round(_obs_disp, 2),
                "observed_max_c": round(observed_max_so_far, 2),
                "target_bin":    target_label,
                "entry_price":   round(entry_price, 4),
                "day_max":       round(_day_disp, 2),
                "day_max_c":     round(day_max, 2),
                "winning_bin":   winning_label,
                "unit":          _unit,
                "won":           won,
                "pnl":           None,
                "action":        "SKIP_PRICED_IN",
            }

        return {
            "city": city, "date": date_str,
            "fired_at_hour": trigger_h,
            "observed_max":  round(_obs_disp, 2),
            "observed_max_c": round(observed_max_so_far, 2),
            "target_bin":    target_label,
            "entry_price":   round(entry_price, 4),
            "day_max":       round(_day_disp, 2),
            "day_max_c":     round(day_max, 2),
            "winning_bin":   winning_label,
            "unit":          _unit,
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
.sizing { background: #1f2937; color: #f3f4f6; padding: 10px 18px;
          display: flex; gap: 22px; flex-wrap: wrap; align-items: center;
          font-size: 12px; border-bottom: 1px solid #374151; }
.sizing label { font-weight: 600; color: #d1d5db; margin-right: 4px; }
.sizing .mode { display: inline-flex; border: 1px solid #4b5563; border-radius: 4px;
                overflow: hidden; }
.sizing .mode button { background: #374151; color: #d1d5db; border: none;
                       padding: 3px 10px; font-size: 11px; cursor: pointer; }
.sizing .mode button.active { background: #4338ca; color: white; }
.sizing input[type=number] { width: 70px; padding: 3px 6px; background: #374151;
                              border: 1px solid #4b5563; color: white;
                              border-radius: 3px; }
.sizing .result { color: #fbbf24; font-family: monospace; font-weight: 600; }
.sizing .result.neg { color: #f87171; }
.sizing .hint { color: #9ca3af; font-size: 11px; }
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

<div class='sizing'>
  <div><label>Stake mode</label>
    <span class='mode'>
      <button id='m-fixed' class='active'>Fixed $/trade</button>
      <button id='m-kelly'>Fractional Kelly</button>
      <button id='m-compound'>Compound (start bankroll)</button>
    </span>
  </div>
  <div id='input-fixed'><label>$ per trade</label>
    <input id='stake-fixed' type='number' value='20' min='1' step='5'></div>
  <div id='input-kelly' style='display:none'><label>Bankroll $</label>
    <input id='stake-bank' type='number' value='1000' min='10' step='100'>
    <label style='margin-left:8px'>Kelly fraction</label>
    <input id='kelly-frac' type='number' value='0.25' min='0.05' max='1' step='0.05'>
    <span class='hint'>(0.25 = quarter-Kelly, safer; 1.0 = full Kelly, aggressive)</span></div>
  <div id='input-compound' style='display:none'><label>Start bankroll $</label>
    <input id='stake-start' type='number' value='1000' min='10' step='100'>
    <label style='margin-left:8px'>% of bankroll/trade</label>
    <input id='stake-pct' type='number' value='5' min='0.5' max='50' step='0.5'></div>
  <div class='result' id='sizing-result'>—</div>
  <div class='hint'>(applies to the currently-filtered BUY_YES trades)</div>
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
  const u = t.unit || '°';
  return `<tr class='${{st}}'>
    <td>${{t.date}}</td>
    <td>${{t.city}}</td>
    <td class='num'>${{t.fired_at_hour}}</td>
    <td class='num'>${{fmt(t.observed_max, 1)}}${{u}}</td>
    <td>${{t.target_bin}}</td>
    <td class='num'>${{fmt(t.entry_price, 3)}}</td>
    <td class='num'>${{fmt(t.day_max, 1)}}${{u}}</td>
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

// ===== Sizing controls =====
let SIZING_MODE = 'fixed';   // 'fixed' | 'kelly' | 'compound'

function setSizingMode(m) {{
  SIZING_MODE = m;
  ['fixed','kelly','compound'].forEach(x => {{
    $('m-'+x).classList.toggle('active', x === m);
    $('input-'+x).style.display = (x === m) ? '' : 'none';
  }});
  updateSizing();
}}
['fixed','kelly','compound'].forEach(m =>
  $('m-'+m).addEventListener('click', () => setSizingMode(m)));

// Compute sizing on the currently-filtered BUY_YES trades only
function getBuyTrades() {{
  const status = $('f-status').value;
  const city = $('f-city').value;
  const lo = parseFloat($('f-min').value);
  const hi = parseFloat($('f-max').value);
  return TRADES.filter(t => {{
    if (t.action !== 'BUY_YES') return false;
    if (city && t.city !== city) return false;
    if (!isNaN(lo) && t.entry_price < lo) return false;
    if (!isNaN(hi) && t.entry_price > hi) return false;
    // Honor "Wins/Losses only" so user can see "what if I only had the wins"
    if (status === 'WIN'  && !t.won) return false;
    if (status === 'LOSS' &&  t.won) return false;
    return true;
  }});
}}

function updateSizing() {{
  const trades = getBuyTrades();
  const n = trades.length;
  if (n === 0) {{ $('sizing-result').textContent = 'no trades match filters'; return; }}

  let out = '';
  if (SIZING_MODE === 'fixed') {{
    const stake = parseFloat($('stake-fixed').value) || 0;
    const totalCost = n * stake;
    const totalPnl  = trades.reduce((s, t) => s + (t.pnl * stake), 0);
    const wins      = trades.filter(t => t.won).length;
    out = `${{n}} trades × $${{stake.toFixed(0)}} = `
        + `$${{totalCost.toLocaleString()}} deployed · `
        + `P&L $${{totalPnl >= 0 ? '+' : ''}}${{totalPnl.toFixed(2)}} · `
        + `ROI ${{(100*totalPnl/totalCost).toFixed(1)}}% · `
        + `${{wins}}W/${{n-wins}}L`;
    if (totalPnl < 0) $('sizing-result').classList.add('neg');
    else $('sizing-result').classList.remove('neg');
  }} else if (SIZING_MODE === 'kelly') {{
    const bank    = parseFloat($('stake-bank').value) || 0;
    const frac    = parseFloat($('kelly-frac').value) || 0;
    // Kelly per trade requires p(win); we use the BUCKET'S empirical win rate
    // (so trades near same entry price share the same p estimate).
    const buckets = [[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.01]];
    const bWins = {{}}, bN = {{}};
    for (const t of trades) {{
      for (const [lo,hi] of buckets) {{
        if (t.entry_price >= lo && t.entry_price < hi) {{
          const k = lo+'-'+hi;
          bN[k] = (bN[k]||0) + 1;
          if (t.won) bWins[k] = (bWins[k]||0) + 1;
          break;
        }}
      }}
    }}
    let totalCost = 0, totalPnl = 0, wins = 0;
    for (const t of trades) {{
      for (const [lo,hi] of buckets) {{
        if (t.entry_price >= lo && t.entry_price < hi) {{
          const k = lo+'-'+hi;
          const p = (bWins[k]||0) / Math.max(1, bN[k]);
          const B = (1/t.entry_price) - 1;   // payout odds
          const fStar = (p*B - (1-p)) / B;   // full-Kelly fraction
          const stake = Math.max(0, bank * frac * fStar);
          if (stake <= 0) break;
          totalCost += stake;
          totalPnl  += t.pnl * stake;
          if (t.won) wins++;
          break;
        }}
      }}
    }}
    out = `bankroll $${{bank.toLocaleString()}} × ${{frac.toFixed(2)}}-Kelly · `
        + `$${{totalCost.toFixed(0)}} deployed across ${{n}} trades · `
        + `P&L $${{totalPnl >= 0 ? '+' : ''}}${{totalPnl.toFixed(2)}} · `
        + `${{wins}}W/${{n-wins}}L`;
    if (totalPnl < 0) $('sizing-result').classList.add('neg');
    else $('sizing-result').classList.remove('neg');
  }} else {{
    // Compound: % of CURRENT bankroll on each trade, chronologically
    const start = parseFloat($('stake-start').value) || 0;
    const pct   = parseFloat($('stake-pct').value)/100 || 0;
    let bank = start;
    const sorted = trades.slice().sort((a,b) => a.date.localeCompare(b.date));
    let wins = 0;
    for (const t of sorted) {{
      const stake = bank * pct;
      bank += t.pnl * stake;
      if (t.won) wins++;
    }}
    const ret = (bank/start - 1) * 100;
    out = `start $${{start.toLocaleString()}} → end $${{bank.toFixed(2)}} `
        + `(${{ret >= 0 ? '+' : ''}}${{ret.toFixed(1)}}%) · `
        + `${{n}} trades @ ${{(pct*100).toFixed(1)}}% bankroll/trade · `
        + `${{wins}}W/${{n-wins}}L`;
    if (bank < start) $('sizing-result').classList.add('neg');
    else $('sizing-result').classList.remove('neg');
  }}
  $('sizing-result').textContent = out;
}}

['stake-fixed','stake-bank','kelly-frac','stake-start','stake-pct'].forEach(id =>
  $(id).addEventListener('input', updateSizing));

// Re-run sizing whenever filters change too (wrap existing handlers)
['f-status','f-city','f-min','f-max'].forEach(id =>
  $(id).addEventListener('input', updateSizing));

render();
updateSizing();
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

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True   # avoid "Address already in use" on quick reruns
    try:
        httpd = ReusableTCPServer(("0.0.0.0", port), handler)
    except OSError as e:
        print(f"\n  Could not bind port {port}: {e}")
        print(f"  Another process is using it.  Find it with:")
        print(f"      ss -tlnp | grep ':{port}'")
        print(f"  Or just pick a different port: --serve {port + 1}")
        return
    try:
        with httpd:
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
    p.add_argument("--price-db", default=DEFAULT_COLLECTOR_DB,
                   help=f"Collector DB with price_snapshots + bins + events + "
                        f"resolutions (default: {DEFAULT_COLLECTOR_DB})")
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
    log.info(f"Cities: {len(cities)}  | station_db={args.station_db}")
    log.info(f"Price collector DB: {args.price_db}")

    # 1. Station temps
    if not os.path.exists(args.station_db):
        log.error(f"Station DB not found: {args.station_db}")
        log.error(f"Run: python -m scripts.station_obs_pull --days {args.days}")
        return 1
    station_temps = load_station_temps(args.station_db, cities, start, end)
    log.info(f"Loaded station temps for {len(station_temps):,} city-days")

    if not os.path.exists(args.price_db):
        log.error(f"Price collector DB not found: {args.price_db}")
        log.error(f"Pass --price-db /path/to/prices.db")
        return 1

    # 2. Walk events from collector DB; one event at a time keeps memory flat
    price_conn = _open_ro(args.price_db)
    events = load_resolved_events(price_conn, start, end, cities)
    log.info(f"Found {len(events):,} resolved events in window")

    trades: list[dict] = []
    no_bin_meta = 0
    no_prices   = 0
    no_obs      = 0
    n_processed = 0

    for ev in events:
        city, date_str, event_id = ev["city"], ev["date"], ev["event_id"]
        s = CITY_STATIONS.get(city)
        if not s:
            continue   # silently skip cities outside our station map
        tz = s[2]

        temps = station_temps.get((city, date_str))
        if not temps or len(temps) < 18:
            no_obs += 1
            continue

        bins = load_bins_for_event(price_conn, event_id)
        if not bins:
            no_bin_meta += 1
            continue

        snapshots = load_snapshots_for_event(price_conn, event_id)
        if not snapshots:
            no_prices += 1
            continue

        resolution = load_resolution(price_conn, event_id)

        result = simulate_day(city, date_str, tz, temps, bins, snapshots,
                              resolution, args.threshold, args.hours_after_peak)
        if result:
            trades.append(result)
        n_processed += 1
        if n_processed % 100 == 0:
            log.info(f"  processed {n_processed}/{len(events)} events ...")
    price_conn.close()
    log.info(f"Simulation complete: {n_processed}/{len(events)} events processed, "
             f"{len(trades)} reached a decision")

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
        fields = ["city", "date", "action", "fired_at_hour",
                  "observed_max", "observed_max_c", "unit",
                  "target_bin", "entry_price",
                  "day_max", "day_max_c",
                  "winning_bin", "won", "pnl"]
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