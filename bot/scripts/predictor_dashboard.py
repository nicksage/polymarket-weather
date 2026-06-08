"""
predictor_dashboard.py — Live dashboard for the intraday predictor.

Reads from paper_predictor_signals + live_predictor_orders and renders a
self-contained HTML page with:

  * Top mode toggle (paper / live / both)
  * KPI strip — counts respect the mode toggle
  * Per-city panels — one for every US city, showing today's event with
    EVERY bin's our_p, market_p, edge, action, entry price, and P&L
  * Today's signals table — filterable, sortable, grouped by city

Configured entirely via .env (no CLI flags needed for normal operation):
  DASHBOARD_PORT       (default 8082)
  DASHBOARD_WATCH_SEC  (default 30, auto-regenerates HTML)
  DASHBOARD_DAYS       (default 1, lookback for the table)

Service install:  sudo bash deploy/setup_dashboard_systemd.sh
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import socketserver
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BOT_DIR)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

# Force python-dotenv to load .env BEFORE config.py imports.  This
# matters when run via systemd, whose EnvironmentFile= parser does NOT
# strip inline comments — so a line like "MAX_OPEN_POSITIONS=0  # cap"
# arrives as the literal string '0  # cap' and int() chokes.  Loading
# with override=True via python-dotenv (which DOES strip comments)
# rewrites the polluted vars before config.py touches them.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)
except ImportError:
    pass

from config import DB_PATH  # type: ignore
try:
    from station_meta import CITY_STATIONS  # type: ignore
except Exception:
    CITY_STATIONS = {}
try:
    from scripts.find_nearby_stations import US_CITY_STATES  # type: ignore
except Exception:
    US_CITY_STATES = {}

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("predictor_dash")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_tables(db: str) -> None:
    """Create predictor tables if missing — lets the dashboard run before
    the bot's first scan has populated them."""
    try:
        from scheduled_predictor import ensure_schema  # type: ignore
        ensure_schema()
    except Exception as e:
        log.debug(f"ensure_schema import failed: {e}")


def load_signals(db: str, since_utc: datetime) -> list[dict]:
    if not os.path.exists(db):
        return []
    _ensure_tables(db)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, scanned_at_utc, mode, city, settlement_station,
                       event_date, event_id, contract_id, bin_label,
                       our_prob, market_prob, edge, liquidity_usd,
                       action, gate_blocked_by,
                       recommended_stake_usd, recommended_limit_price,
                       current_hour_local, observed_max_c, observed_peak_hour,
                       forecast_high_c, forecast_peak_hour,
                       mu_c, sigma_c, wind_octant
                FROM paper_predictor_signals
                WHERE scanned_at_utc >= ?
                ORDER BY scanned_at_utc DESC
                """,
                (since_utc.isoformat(),),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning(f"paper_predictor_signals not readable: {e}")
            return []
    return [dict(r) for r in rows]


def load_live_orders(db: str, since_utc: datetime) -> list[dict]:
    if not os.path.exists(db):
        return []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM live_predictor_orders WHERE placed_at_utc >= ? "
                "ORDER BY placed_at_utc DESC", (since_utc.isoformat(),)
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Static city list (US only — what the predictor trades)
# ---------------------------------------------------------------------------

def _us_cities() -> list[dict]:
    us = (list(US_CITY_STATES.keys()) if US_CITY_STATES
           else [c for c, m in CITY_STATIONS.items() if m[0].startswith("K")])
    out = []
    for city in sorted(us):
        s = CITY_STATIONS.get(city)
        if not s:
            continue
        icao, _net, tz, _lat, _lon = s
        tz_label = tz.split("/")[-1].replace("_", " ")
        out.append({"city": city, "station": icao, "tz_label": tz_label})
    return out


# ---------------------------------------------------------------------------
# HTML / CSS / JS
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #0f172a; color: #e2e8f0; font-size: 13px; }

header { background: linear-gradient(90deg, #1e293b, #0f172a);
         padding: 14px 24px; border-bottom: 1px solid #334155;
         display: flex; align-items: center; justify-content: space-between; }
header h1 { margin: 0; font-size: 18px; font-weight: 600; color: white; }
header .meta { font-family: monospace; font-size: 11px; color: #94a3b8; text-align: right; }

/* Mode toggle in the header */
.mode-toggle { display: inline-flex; background: #0f172a; border-radius: 6px;
               overflow: hidden; border: 1px solid #334155; }
.mode-toggle button { background: transparent; border: none;
                       color: #94a3b8; padding: 6px 14px; cursor: pointer;
                       font-size: 12px; font-weight: 600; }
.mode-toggle button:hover { background: #273449; color: white; }
.mode-toggle button.active.paper { background: #1e3a8a; color: #dbeafe; }
.mode-toggle button.active.live  { background: #7f1d1d; color: #fef2f2; }
.mode-toggle button.active.both  { background: #4338ca; color: white; }
.mode-toggle button.active.view  { background: #475569; color: white; }

/* KPI strip */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 8px; padding: 12px 24px; background: #0f172a; }
.kpi { background: #1e293b; padding: 10px 14px; border-radius: 6px;
       border-left: 3px solid #4338ca; }
.kpi .label { font-size: 9px; color: #94a3b8; text-transform: uppercase;
              letter-spacing: 0.5px; font-weight: 600; }
.kpi .val { font-size: 22px; font-weight: 700; margin-top: 4px;
            font-family: monospace; color: white; }
.kpi .sub { font-size: 10px; color: #94a3b8; margin-top: 2px; font-family: monospace; }
.kpi.buy { border-left-color: #22c55e; }
.kpi.buy .val { color: #4ade80; }
.kpi.live { border-left-color: #f59e0b; }
.kpi.live .val { color: #fbbf24; }
.kpi.pos { border-left-color: #22c55e; }
.kpi.pos .val { color: #4ade80; }
.kpi.neg { border-left-color: #ef4444; }
.kpi.neg .val { color: #f87171; }

.section-title { padding: 18px 24px 6px; font-size: 11px; font-weight: 700;
                  color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

/* === Per-city panels === */
.city-grid { display: grid; gap: 14px; padding: 6px 24px 18px;
             grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); }
.city-panel { background: #1e293b; border-radius: 8px; overflow: hidden;
               border: 1px solid #334155; display: flex; flex-direction: column; }
.city-panel.has-position { border-color: #22c55e; }
.city-panel.has-live { border-color: #f59e0b; }
.city-panel.no-data { opacity: 0.55; }

.city-header { display: flex; justify-content: space-between; align-items: center;
                background: #0f172a; padding: 10px 14px; border-bottom: 1px solid #334155; }
.city-header .name { font-size: 16px; font-weight: 700; color: white; }
.city-header .station { font-family: monospace; color: #94a3b8; font-size: 11px;
                         margin-top: 2px; }
.city-header .now { text-align: right; font-family: monospace; font-size: 11px;
                     color: #94a3b8; }
.city-header .now .temp { color: #fbbf24; font-size: 16px; font-weight: 700;
                           font-family: monospace; }

.city-stats { padding: 8px 14px; background: #1a2438; font-size: 11px;
               color: #94a3b8; display: grid; grid-template-columns: 1fr 1fr 1fr;
               gap: 8px; border-bottom: 1px solid #334155; font-family: monospace; }
.city-stats .v { color: #e2e8f0; font-weight: 600; }
.city-stats .v.hot { color: #fbbf24; }

table.bins { width: 100%; border-collapse: collapse; font-size: 12px; }
table.bins th, table.bins td { padding: 6px 12px; text-align: right;
                                 border-bottom: 1px solid #273449; }
table.bins th { background: #0f172a; color: #64748b; font-weight: 600;
                 font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px; }
table.bins th:first-child, table.bins td:first-child { text-align: left; }
table.bins td.bin-label { font-weight: 700; color: white; font-family: monospace; }
table.bins td.bin-label.top { color: #fbbf24; }
table.bins td.action { text-align: center; }
table.bins td.action .pill { display: inline-block; padding: 1px 6px;
                              border-radius: 10px; font-size: 9px; font-weight: 700; }
.pill.PAPER_BUY { background: #22c55e; color: white; }
.pill.LIVE_BUY  { background: #f59e0b; color: white; }
.pill.idle      { background: #334155; color: #94a3b8; }
.pnl.pos { color: #4ade80; font-weight: 700; }
.pnl.neg { color: #f87171; font-weight: 700; }

table.bins tr.bought { background: rgba(34, 197, 94, 0.08); }
table.bins tr.bought.live { background: rgba(245, 158, 11, 0.10); }
table.bins tr.top-p { box-shadow: inset 3px 0 0 #fbbf24; }

.city-empty { padding: 24px; text-align: center; color: #64748b; font-size: 12px;
               font-style: italic; }

/* Event sub-section header (only shown when a city has >1 event) */
.event-header { display: flex; justify-content: space-between; align-items: center;
                background: #273449; padding: 6px 14px; font-size: 11px;
                color: #cbd5e1; border-top: 1px solid #334155; }
.event-header .event-date { font-weight: 700; color: white; font-family: monospace; }
.event-header .event-stats { color: #94a3b8; font-family: monospace; }

/* === Signals table === */
.filters { background: #1e293b; padding: 10px 24px; display: flex;
           gap: 16px; flex-wrap: wrap; align-items: center; font-size: 12px;
           border-bottom: 1px solid #334155; position: sticky; top: 0; z-index: 10; }
.filters label { font-weight: 600; color: #cbd5e1; margin-right: 4px; }
.filters select, .filters input { padding: 4px 8px; font-size: 12px;
           background: #0f172a; color: white; border: 1px solid #475569;
           border-radius: 4px; }
.filters .count { color: #94a3b8; font-family: monospace; margin-left: auto; }

table.signals { width: calc(100% - 48px); margin: 12px 24px; background: #1e293b;
                 border-collapse: collapse; border-radius: 6px; overflow: hidden;
                 font-size: 12px; }
table.signals th, table.signals td { padding: 7px 10px; text-align: left;
                                       border-bottom: 1px solid #273449; }
table.signals th { background: #0f172a; color: #94a3b8; font-weight: 600;
                    font-size: 10px; text-transform: uppercase;
                    letter-spacing: 0.5px; cursor: pointer; user-select: none; }
table.signals th:hover { background: #1e293b; color: white; }
table.signals th.sorted-asc::after { content: ' ▲'; color: #818cf8; }
table.signals th.sorted-desc::after { content: ' ▼'; color: #818cf8; }
table.signals td.num { font-family: monospace; text-align: right; }
table.signals td.tstamp { font-family: monospace; font-size: 11px; color: #94a3b8; }

/* City-break: thicker top border when city changes */
table.signals tr.city-break td { border-top: 3px solid #475569; }

table.signals tr.LIVE_BUY  { background: rgba(245, 158, 11, 0.18);
                              border-left: 4px solid #f59e0b; }
table.signals tr.PAPER_BUY { background: rgba(34, 197, 94, 0.10); }
table.signals tr.AVOID     { background: rgba(239, 68, 68, 0.06); color: #94a3b8; }
table.signals tr.SKIP      { color: #64748b; }
table.signals tr:hover     { background: #273449 !important; }

table.signals td .pill {
  display: inline-block; padding: 2px 7px; border-radius: 10px;
  font-size: 10px; font-weight: 700;
}
table.signals .pill.LIVE_BUY  { background: #f59e0b; color: white; }
table.signals .pill.PAPER_BUY { background: #22c55e; color: white; }
table.signals .pill.SKIP      { background: #475569; color: #cbd5e1; }
table.signals .pill.AVOID     { background: #ef4444; color: white; }

.edge { font-family: monospace; font-weight: 700; }
.edge.pos { color: #4ade80; }
.edge.neg { color: #f87171; }

.empty { text-align: center; color: #64748b; padding: 32px 0; font-size: 13px; }

/* Live orders table */
table.live-orders { width: calc(100% - 48px); margin: 8px 24px 16px;
                     background: #1e293b; border-collapse: collapse;
                     border-radius: 6px; overflow: hidden; font-size: 12px; }
table.live-orders th, table.live-orders td { padding: 7px 12px;
                     border-bottom: 1px solid #273449; }
table.live-orders th { background: #0f172a; color: #94a3b8; font-weight: 600;
                       font-size: 10px; text-transform: uppercase; }
.pill.placed { background: #22c55e; color: white; }
.pill.filled { background: #16a34a; color: white; }
.pill.failed { background: #ef4444; color: white; }
.pill.error  { background: #dc2626; color: white; }
.pill.skip   { background: #475569; color: #cbd5e1; }
"""


def build_dashboard(signals: list[dict], live_orders: list[dict],
                     generated_at_utc: str,
                     auto_refresh_sec: int | None = None) -> str:
    """Build the full HTML page.  All data filtering / aggregation happens
    in JS so the mode toggle is instant and reuses one data blob."""
    cities_meta = _us_cities()
    sig_json    = json.dumps(signals, default=str, separators=(",", ":"))
    live_json   = json.dumps(live_orders, default=str, separators=(",", ":"))
    cities_json = json.dumps(cities_meta, default=str, separators=(",", ":"))

    # No meta refresh — the JS does a silent fetch + re-render every
    # auto_refresh_sec, preserving sort order, filter state, and scroll
    # position.  No blank flash on update.
    refresh_badge = (f'<span style="color:#94a3b8;font-size:10px">auto-refresh: {auto_refresh_sec}s</span>'
                      if auto_refresh_sec else "")
    refresh_sec_js = int(auto_refresh_sec or 0)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Predictor Dashboard</title>
<style>{DASHBOARD_CSS}</style></head><body>

<header>
  <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
    <h1>Predictor Dashboard</h1>
    <div class="mode-toggle">
      <button id="mode-paper" class="active paper">Paper</button>
      <button id="mode-live">Live</button>
      <button id="mode-both">Both</button>
    </div>
    <div class="mode-toggle view-toggle">
      <button id="view-signals" class="active view">Signals</button>
      <button id="view-trades" class="view">Trades</button>
    </div>
  </div>
  <div class="meta">generated {generated_at_utc} {refresh_badge}<br>
    <span id="hdr-counts">—</span>
  </div>
</header>

<div class="kpis" id="kpis"></div>

<div class="section-title" id="panels-title">Per-city panels</div>
<div class="filters">
  <div><label>Date</label><select id="f-date"></select></div>
  <div><label>City</label><select id="f-city"><option value="">All</option></select></div>
  <div><label>Action</label><select id="f-action">
    <option value="">All</option>
    <option>PAPER_BUY</option><option>LIVE_BUY</option>
    <option>SKIP</option><option>AVOID</option>
  </select></div>
  <div><label>Min edge</label>
    <input id="f-edge" type="number" step="0.05" value="-1" style="width:70px"></div>
  <div><label>Buys only</label>
    <input id="f-buys" type="checkbox"></div>
  <div><label>Latest scan only</label>
    <input id="f-latest" type="checkbox" checked></div>
  <div class="count" id="count">—</div>
</div>
<div class="city-grid" id="city-grid"></div>

<div class="section-title" id="signals-title">All signals</div>

<table class="signals" id="sig-table">
  <thead><tr>
    <th data-key="scanned_at_utc">Scanned (UTC)</th>
    <th data-key="city">City</th>
    <th data-key="bin_label">Bin</th>
    <th data-key="our_prob">Our P</th>
    <th data-key="market_prob">Mkt P</th>
    <th data-key="edge">Edge</th>
    <th data-key="liquidity_usd">Liquidity</th>
    <th data-key="recommended_stake_usd">Stake</th>
    <th data-key="action">Action</th>
    <th data-key="gate_blocked_by">Reason</th>
  </tr></thead>
  <tbody id="sig-tbody"></tbody>
</table>

<table class="signals" id="trades-table" style="display:none">
  <thead><tr>
    <th data-key="scanned_at_utc">Placed (UTC)</th>
    <th data-key="city">City</th>
    <th data-key="bin_label">Bin</th>
    <th data-key="action">Mode</th>
    <th data-key="entry_price">Entry</th>
    <th data-key="current_price">Now</th>
    <th data-key="stake">Stake</th>
    <th data-key="shares">Shares</th>
    <th data-key="current_value">Value</th>
    <th data-key="pnl_usd">P&L $</th>
    <th data-key="pnl_pct">P&L %</th>
    <th data-key="live_status">Order status</th>
    <th data-key="live_order_id">Order ID</th>
  </tr></thead>
  <tbody id="trades-tbody"></tbody>
</table>

<script>
// Mutable so the silent-refresh fetch can swap them in place
let SIGNALS = {sig_json};
let LIVE_ORDERS = {live_json};
const CITIES = {cities_json};
const REFRESH_SEC = {refresh_sec_js};
const $ = id => document.getElementById(id);

// Active mode filter for the entire dashboard: 'paper' | 'live' | 'both'
let MODE_FILTER = "paper";
// Right-side bottom table: 'signals' (all evaluations) or 'trades' (just BUYs
// with entry, current, P&L, live status).
let VIEW_MODE = "signals";

// The "date" filter is the master — KPIs, panels, signals table all
// respect it.  Defaults to today's UTC date but can be changed via the
// dropdown to inspect historical data.
let SELECTED_DATE = "";

// Recomputed on every refresh
let DATE_SIGNALS = [];      // signals matching SELECTED_DATE (scan side)
let DATE_LIVE_ORDERS = [];  // live orders placed on SELECTED_DATE
let LATEST_SCAN_TS = "";    // most-recent scan within SELECTED_DATE

function todayUtcStr() {{
  return new Date().toISOString().slice(0, 10);
}}

function recomputeDerived() {{
  if (!SELECTED_DATE) SELECTED_DATE = todayUtcStr();
  // CHANGED: filter by event_date (the resolution day of the Polymarket
  // market), not by scanned_at_utc.  This is semantically what the user
  // means by "show me data for this day" — events resolving that day,
  // regardless of when our bot scanned them.  Also fixes the list-view
  // duplicate-rows issue: when two events for different dates were both
  // scanned today, the list previously showed bins from both.
  DATE_SIGNALS = SIGNALS.filter(s => s.event_date === SELECTED_DATE);
  // Live orders also have event_date, so filter the same way.  Fall
  // back to placed_at_utc match if event_date is missing on the row.
  DATE_LIVE_ORDERS = LIVE_ORDERS.filter(o =>
    (o.event_date && o.event_date === SELECTED_DATE)
    || (!o.event_date && o.placed_at_utc
         && o.placed_at_utc.slice(0, 10) === SELECTED_DATE));
  LATEST_SCAN_TS = DATE_SIGNALS.length
    ? DATE_SIGNALS.reduce((m, s) => s.scanned_at_utc > m ? s.scanned_at_utc : m,
                          DATE_SIGNALS[0].scanned_at_utc)
    : "";
}}
recomputeDerived();

// ===== Mode filter helpers =====
function isBuy(s) {{ return s.action === "PAPER_BUY" || s.action === "LIVE_BUY"; }}
function matchesMode(s) {{
  if (MODE_FILTER === "both") return true;
  if (MODE_FILTER === "live") return s.mode === "live";
  return s.mode === "paper";
}}
function matchesBuyMode(s) {{
  // For BUYs specifically: filter the buy action by current mode toggle
  if (MODE_FILTER === "both") return isBuy(s);
  if (MODE_FILTER === "live") return s.action === "LIVE_BUY";
  return s.action === "PAPER_BUY";
}}

// ===== Per-city aggregation =====
// Within a city, signals can belong to MULTIPLE events (Polymarket sometimes
// has more than one "highest temp" market per city per day).  We group bins
// by event_id and render each event as its own sub-table within the panel.
function buildCitySnapshot(cityMeta) {{
  const city = cityMeta.city;
  // Only signals where event_date matches the selected date — keeps
  // panels strictly scoped to the day the user picked (no future-event
  // bleed from older scans that evaluated tomorrow's markets).
  const todayInCity = DATE_SIGNALS.filter(s =>
    s.city === city && s.event_date === SELECTED_DATE);
  if (todayInCity.length === 0) {{
    return {{...cityMeta, hasData: false}};
  }}
  // Latest scan for this city (all events update together each scan)
  const latestTs = todayInCity.reduce((m, s) =>
    s.scanned_at_utc > m ? s.scanned_at_utc : m, todayInCity[0].scanned_at_utc);
  const latestRows = todayInCity.filter(s => s.scanned_at_utc === latestTs);

  // Map BUYs by contract_id for fast lookup (current mode only)
  const buyRows = todayInCity.filter(matchesBuyMode);
  const buyByContract = {{}};
  for (const b of buyRows) {{
    if (!buyByContract[b.contract_id] ||
        b.scanned_at_utc < buyByContract[b.contract_id].scanned_at_utc) {{
      buyByContract[b.contract_id] = b;
    }}
  }}

  // Group latest scan's rows by event_id
  const eventGroups = {{}};
  for (const r of latestRows) {{
    const eid = r.event_id || 'unknown';
    if (!eventGroups[eid]) {{
      eventGroups[eid] = {{
        event_id:   r.event_id,
        event_date: r.event_date,
        rows:       []
      }};
    }}
    eventGroups[eid].rows.push(r);
  }}

  // For each event group, sort bins by our_p desc and build display rows
  const events = Object.values(eventGroups).map(eg => {{
    eg.rows.sort((a, b) => b.our_prob - a.our_prob);
    const bins = eg.rows.map((r, i) => {{
      const buy = buyByContract[r.contract_id];
      let entry_price = null, stake_usd = null, pnl_usd = null;
      if (buy) {{
        entry_price = buy.market_prob;
        stake_usd   = buy.recommended_stake_usd || 0;
        if (entry_price > 0) {{
          pnl_usd = stake_usd * ((r.market_prob / entry_price) - 1);
        }}
      }}
      return {{
        bin:          r.bin_label,
        our_prob:     r.our_prob,
        market_prob:  r.market_prob,
        edge:         r.edge,
        action:       buy ? buy.action : null,
        entry_price:  entry_price,
        stake_usd:    stake_usd,
        pnl_usd:      pnl_usd,
        is_top_p:     i === 0,
      }};
    }});
    return {{
      event_id:      eg.event_id,
      event_date:    eg.event_date,
      bins:          bins,
      nBuys:         bins.filter(b => b.action).length,
      nLiveBuys:     bins.filter(b => b.action === "LIVE_BUY").length,
      totalDeployed: bins.reduce((s, b) => s + (b.stake_usd || 0), 0),
      totalPnl:      bins.reduce((s, b) => s + (b.pnl_usd || 0), 0),
    }};
  }});

  // Sort events: today first, then by date
  events.sort((a, b) => (a.event_date || '').localeCompare(b.event_date || ''));

  // City-level rollups across all events
  const anyRow = latestRows[0];
  return {{
    ...cityMeta,
    hasData:          true,
    events:           events,
    eventCount:       events.length,
    observedMax:      anyRow.observed_max_c,
    observedPeakHour: anyRow.observed_peak_hour,
    forecastHigh:     anyRow.forecast_high_c,
    forecastPeakHour: anyRow.forecast_peak_hour,
    windOctant:       anyRow.wind_octant,
    nBuys:            events.reduce((s, e) => s + e.nBuys, 0),
    nLiveBuys:        events.reduce((s, e) => s + e.nLiveBuys, 0),
    totalDeployed:    events.reduce((s, e) => s + e.totalDeployed, 0),
    totalPnl:         events.reduce((s, e) => s + e.totalPnl, 0),
  }};
}}

// ===== KPI rendering =====
function renderKPIs() {{
  const buys = DATE_SIGNALS.filter(matchesBuyMode);
  const nBuys = buys.length;
  const deployed = buys.reduce((s, b) => s + (b.recommended_stake_usd || 0), 0);

  // Compute total P&L across all buys in current mode
  let totalPnl = 0;
  for (const city of CITIES) {{
    const snap = buildCitySnapshot(city);
    totalPnl += snap.totalPnl || 0;
  }}

  const avgEdge = nBuys ? buys.reduce((s, b) => s + b.edge, 0) / nBuys : 0;
  const nSkips  = DATE_SIGNALS.filter(s => s.action === "SKIP" && matchesMode(s)).length;

  const pnlClass = totalPnl >= 0 ? "pos" : "neg";
  const pnlSign  = totalPnl >= 0 ? "+" : "";
  const buyClass = MODE_FILTER === "live" ? "live" : "buy";
  const buyLabel = MODE_FILTER === "live" ? "LIVE BUYs"
                : MODE_FILTER === "both" ? "Total BUYs"
                : "PAPER BUYs";

  $("kpis").innerHTML = `
    <div class="kpi ${{buyClass}}"><div class="label">${{buyLabel}}</div>
      <div class="val">${{nBuys}}</div>
      <div class="sub">$${{deployed.toFixed(2)}} deployed</div></div>
    <div class="kpi ${{pnlClass}}"><div class="label">Total P&L</div>
      <div class="val">${{pnlSign}}$${{totalPnl.toFixed(2)}}</div>
      <div class="sub">${{deployed > 0 ? (pnlSign + (100*totalPnl/deployed).toFixed(1) + '%') : '—'}} ROI</div></div>
    <div class="kpi"><div class="label">Avg edge on BUYs</div>
      <div class="val">${{nBuys ? (avgEdge >= 0 ? '+' : '') + (avgEdge*100).toFixed(1) + '%' : '—'}}</div></div>
    <div class="kpi"><div class="label">Skipped today</div>
      <div class="val">${{nSkips.toLocaleString()}}</div></div>
    <div class="kpi"><div class="label">Scanned today</div>
      <div class="val">${{DATE_SIGNALS.filter(matchesMode).length.toLocaleString()}}</div>
      <div class="sub">${{DATE_SIGNALS.length ? 'last: ' + LATEST_SCAN_TS.slice(11,16) + 'Z' : ''}}</div></div>
  `;
  $("hdr-counts").textContent =
    `${{nBuys}} BUYs · $${{deployed.toFixed(0)}} dep · ${{pnlSign}}$${{totalPnl.toFixed(2)}} P&L`;
}}

// ===== City panel rendering =====
function cToF(c) {{ return c * 9/5 + 32; }}

function renderBinRow(b) {{
  const edgeClass = b.edge >= 0 ? "edge pos" : "edge neg";
  const edgeStr = (b.edge >= 0 ? "+" : "") + (b.edge*100).toFixed(1) + "%";
  const action = b.action
    ? `<span class="pill ${{b.action}}">${{b.action === 'LIVE_BUY' ? 'LIVE' : 'PAPER'}}</span>`
    : '<span class="pill idle">—</span>';
  const entry = b.entry_price !== null ? '$' + b.entry_price.toFixed(3) : '';
  const pnl = b.pnl_usd !== null
    ? `<span class="pnl ${{b.pnl_usd >= 0 ? 'pos' : 'neg'}}">${{b.pnl_usd >= 0 ? '+' : ''}}$${{b.pnl_usd.toFixed(2)}}</span>`
    : '';
  let rowCls = '';
  if (b.action === 'LIVE_BUY') rowCls = 'bought live';
  else if (b.action === 'PAPER_BUY') rowCls = 'bought';
  if (b.is_top_p) rowCls += ' top-p';
  return `<tr class="${{rowCls}}">
    <td class="bin-label ${{b.is_top_p ? 'top' : ''}}">${{b.is_top_p ? '★ ' : ''}}${{b.bin}}</td>
    <td class="num">${{(b.our_prob*100).toFixed(1)}}%</td>
    <td class="num">${{(b.market_prob*100).toFixed(1)}}%</td>
    <td class="num ${{edgeClass}}">${{edgeStr}}</td>
    <td class="action">${{action}}</td>
    <td class="num">${{entry}}</td>
    <td class="num">${{pnl}}</td>
  </tr>`;
}}

function renderEventSection(ev, showHeader) {{
  const binsHtml = ev.bins.map(renderBinRow).join('');
  const header = showHeader
    ? `<div class="event-header">
         <span class="event-date">Event ${{ev.event_date}}</span>
         <span class="event-stats">${{ev.bins.length}} bins
           ${{ev.nBuys ? '· ' + ev.nBuys + ' buy' + (ev.nBuys === 1 ? '' : 's') : ''}}
           ${{ev.totalDeployed > 0 ? '· $' + ev.totalDeployed.toFixed(2) + ' deployed' : ''}}
           ${{ev.totalPnl !== 0 ? '· P&L ' + (ev.totalPnl >= 0 ? '+' : '') + '$' + ev.totalPnl.toFixed(2) : ''}}
         </span>
       </div>`
    : '';
  return `${{header}}<table class="bins">
    <thead><tr>
      <th>Bin</th><th>Our P</th><th>Mkt P</th><th>Edge</th>
      <th>Action</th><th>Entry</th><th>P&L</th>
    </tr></thead>
    <tbody>${{binsHtml}}</tbody>
  </table>`;
}}

function renderCityPanel(snap) {{
  if (!snap.hasData) {{
    return `<div class="city-panel no-data">
      <div class="city-header">
        <div><div class="name">${{snap.city}}</div>
          <div class="station">${{snap.station}} · ${{snap.tz_label}}</div></div>
        <div class="now"><span style="color:#64748b">—</span></div>
      </div>
      <div class="city-empty">No scan data for today yet</div>
    </div>`;
  }}

  let cls = "city-panel";
  if (snap.nLiveBuys > 0) cls += " has-live";
  else if (snap.nBuys > 0) cls += " has-position";

  const obsClass = (snap.observedMax !== null && snap.forecastHigh !== null
                     && snap.observedMax >= snap.forecastHigh - 0.5) ? "hot" : "";

  // Header temperature: show both °C and °F
  let tempDisplay = '—';
  if (snap.observedMax !== null && snap.observedMax !== undefined) {{
    tempDisplay = `${{snap.observedMax.toFixed(1)}}°C
      <span style="color:#94a3b8;font-size:11px">/ ${{cToF(snap.observedMax).toFixed(1)}}°F</span>`;
  }}

  // Forecast high — also show both units
  let fcstDisplay = '—';
  if (snap.forecastHigh !== null && snap.forecastHigh !== undefined) {{
    fcstDisplay = `${{snap.forecastHigh.toFixed(1)}}°C /
      ${{cToF(snap.forecastHigh).toFixed(1)}}°F @ ${{snap.forecastPeakHour}}:00`;
  }}

  // Render each event as its own sub-table.  Hide event headers if only one
  // event (which is the common case — looks cleaner without them).
  const showEventHeaders = snap.events.length > 1;
  const eventSections = snap.events.map(ev =>
    renderEventSection(ev, showEventHeaders)).join('');

  return `<div class="${{cls}}">
    <div class="city-header">
      <div>
        <div class="name">${{snap.city}}</div>
        <div class="station">${{snap.station}} · ${{snap.tz_label}}${{snap.windOctant ? ' · wind ' + snap.windOctant : ''}}</div>
      </div>
      <div class="now">
        <div class="temp ${{obsClass}}">${{tempDisplay}}</div>
        <div>observed @ ${{snap.observedPeakHour}}:00 local</div>
      </div>
    </div>
    <div class="city-stats">
      <div>Forecast high: <span class="v">${{fcstDisplay}}</span></div>
      <div>Events: <span class="v">${{snap.eventCount}}</span></div>
      <div>Position(s): <span class="v">${{snap.nBuys || 0}}</span>
        ${{snap.totalDeployed > 0 ? '· $' + snap.totalDeployed.toFixed(2) + ' deployed' : ''}}
        ${{snap.totalPnl !== 0 ? '· P&L <span class="' + (snap.totalPnl >= 0 ? 'pnl pos' : 'pnl neg') + '">' + (snap.totalPnl >= 0 ? '+' : '') + '$' + snap.totalPnl.toFixed(2) + '</span>' : ''}}
      </div>
    </div>
    ${{eventSections}}
  </div>`;
}}

function renderCityPanels() {{
  // City filter narrows which panels are shown.  Date filter (SELECTED_DATE)
  // is applied inside buildCitySnapshot.
  const cityFilter = $("f-city").value;
  const cities = cityFilter ? CITIES.filter(c => c.city === cityFilter) : CITIES;
  const html = cities.map(c => renderCityPanel(buildCitySnapshot(c))).join('');
  $("city-grid").innerHTML = html ||
    '<div class="empty">No matching cities — try changing the date or city filter</div>';
}}

// (renderLiveOrders removed — live order info now appears in the Trades
//  view's Status / Order ID / Error columns.)

// ===== Signals table =====
let SORT_KEY = "scanned_at_utc", SORT_DIR = -1;

function sigRow(s, addCityBreak) {{
  const eClass = s.edge >= 0 ? "pos" : "neg";
  const eStr = (s.edge >= 0 ? "+" : "") + (s.edge*100).toFixed(1) + "%";
  const stake = s.recommended_stake_usd ? "$" + s.recommended_stake_usd.toFixed(2) : "—";
  const gate = s.gate_blocked_by || "";
  let cls = s.action;
  if (addCityBreak) cls += " city-break";
  return `<tr class="${{cls}}">
    <td class="tstamp">${{s.scanned_at_utc ? s.scanned_at_utc.slice(0,16).replace('T',' ') : ''}}</td>
    <td><b>${{s.city}}</b><br><span style="color:#64748b;font-size:10px">${{s.settlement_station || ''}}</span></td>
    <td><b>${{s.bin_label || ''}}</b></td>
    <td class="num">${{(s.our_prob*100).toFixed(1)}}%</td>
    <td class="num">${{(s.market_prob*100).toFixed(1)}}%</td>
    <td class="num edge ${{eClass}}">${{eStr}}</td>
    <td class="num">$${{Math.round(s.liquidity_usd||0).toLocaleString()}}</td>
    <td class="num">${{stake}}</td>
    <td><span class="pill ${{s.action}}">${{s.action}}</span></td>
    <td style="color:#94a3b8;font-size:11px">${{gate}}</td>
  </tr>`;
}}

function renderSigTable() {{
  const city = $("f-city").value;
  const act  = $("f-action").value;
  const me   = parseFloat($("f-edge").value);
  const buys = $("f-buys").checked;
  const latest = $("f-latest").checked;
  let rows = DATE_SIGNALS.filter(s =>
       matchesMode(s)
    && (!latest || s.scanned_at_utc === LATEST_SCAN_TS)
    && (!city || s.city === city)
    && (!act  || s.action === act)
    && (isNaN(me) || s.edge >= me)
    && (!buys || isBuy(s))
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (typeof av === "number") return SORT_DIR * (av - bv);
    return SORT_DIR * String(av || '').localeCompare(String(bv || ''));
  }});
  rows = rows.slice(0, 500);
  $("count").textContent = rows.length + " / " + DATE_SIGNALS.length;

  // Add city-break visual divider when city changes between rows
  let html = '';
  let lastCity = null;
  for (const r of rows) {{
    const breakNow = lastCity !== null && lastCity !== r.city;
    html += sigRow(r, breakNow);
    lastCity = r.city;
  }}
  $("sig-tbody").innerHTML = html ||
    '<tr><td colspan="10" class="empty">No signals match filters</td></tr>';
  document.querySelectorAll("th").forEach(th => {{
    th.classList.remove("sorted-asc","sorted-desc");
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? "sorted-asc" : "sorted-desc");
  }});
}}

// ===== Date + city dropdown population (refreshed on every render) =====
function refreshDateDropdown() {{
  const dates = [...new Set(SIGNALS.map(s => s.event_date).filter(Boolean))]
    .sort().reverse();
  const sel = $("f-date");
  const today = todayUtcStr();
  // Preserve user's current selection if it's still in the new data
  const currentVal = SELECTED_DATE;
  const fallback = dates.includes(today) ? today : (dates[0] || today);
  sel.innerHTML = dates.map(d =>
    `<option value="${{d}}">${{d}}${{d === today ? ' (today)' : ''}}</option>`
  ).join('') || `<option value="${{today}}">${{today}} (today, no data yet)</option>`;
  const chosen = dates.includes(currentVal) ? currentVal : fallback;
  sel.value = chosen;
  SELECTED_DATE = chosen;
}}

function refreshCityDropdown() {{
  // Cities available on the selected date (preserve current selection)
  const citiesInData = [...new Set(DATE_SIGNALS.map(s => s.city).filter(Boolean))].sort();
  const sel = $("f-city");
  const currentVal = sel.value;
  sel.innerHTML = '<option value="">All</option>'
    + citiesInData.map(c => `<option>${{c}}</option>`).join('');
  if (citiesInData.includes(currentVal)) sel.value = currentVal;
}}

function updateSectionTitles() {{
  const dateLabel = SELECTED_DATE === todayUtcStr() ? '(today)' : `(${{SELECTED_DATE}})`;
  $("panels-title").textContent = `Per-city panels ${{dateLabel}}`;
  $("signals-title").textContent =
    (VIEW_MODE === 'trades' ? 'Trades ' : 'All signals ') + dateLabel;
}}

// Sort handlers
document.querySelectorAll("th").forEach(th => {{
  th.addEventListener("click", () => {{
    if (!th.dataset.key) return;
    if (SORT_KEY === th.dataset.key) SORT_DIR = -SORT_DIR;
    else {{ SORT_KEY = th.dataset.key; SORT_DIR = 1; }}
    renderSigTable();
  }});
}});
// ===== Trades view (paper + live BUYs with full lifecycle info) =====
let TRADES_SORT_KEY = "scanned_at_utc", TRADES_SORT_DIR = -1;

function buildTradesData() {{
  // All BUY rows in the current date range
  const buys = DATE_SIGNALS.filter(s =>
    s.action === "PAPER_BUY" || s.action === "LIVE_BUY");
  // Latest known market price per contract (any scan in this date)
  const latestByContract = {{}};
  for (const s of DATE_SIGNALS) {{
    if (!latestByContract[s.contract_id]
        || s.scanned_at_utc > latestByContract[s.contract_id].scanned_at_utc) {{
      latestByContract[s.contract_id] = s;
    }}
  }}
  // Live order info per contract (most-recent placed order for that bin)
  const liveByContract = {{}};
  for (const o of DATE_LIVE_ORDERS) {{
    const k = o.contract_id;
    if (!liveByContract[k]
        || (o.placed_at_utc || "") > (liveByContract[k].placed_at_utc || "")) {{
      liveByContract[k] = o;
    }}
  }}
  return buys.map(b => {{
    const latest = latestByContract[b.contract_id] || b;
    const live   = b.action === "LIVE_BUY" ? liveByContract[b.contract_id] : null;
    const entry  = b.market_prob;
    const cur    = latest.market_prob;
    const stake  = b.recommended_stake_usd || 0;
    const shares = entry > 0 ? stake / entry : 0;
    const value  = shares * cur;
    const pnl    = value - stake;
    const pct    = entry > 0 ? (cur / entry - 1) : 0;
    return {{
      scanned_at_utc: b.scanned_at_utc,
      city:           b.city,
      station:        b.settlement_station,
      bin_label:      b.bin_label,
      action:         b.action,
      entry_price:    entry,
      current_price:  cur,
      stake:          stake,
      shares:         shares,
      current_value:  value,
      pnl_usd:        pnl,
      pnl_pct:        pct,
      live_status:    live ? (live.status || "") : "",
      live_order_id:  live ? (live.order_id || "") : "",
      live_error:     live ? (live.error || "") : "",
    }};
  }});
}}

function tradeRow(t, addCityBreak) {{
  const modeCls = t.action;
  const modeLabel = t.action === "LIVE_BUY" ? "LIVE" : "PAPER";
  const pnlCls = t.pnl_usd >= 0 ? "pos" : "neg";
  const pnlSign = t.pnl_usd >= 0 ? "+" : "";
  let cls = modeCls + (addCityBreak ? " city-break" : "");
  return `<tr class="${{cls}}">
    <td class="tstamp">${{(t.scanned_at_utc || "").slice(0,16).replace("T"," ")}}</td>
    <td><b>${{t.city}}</b><br><span style="color:#64748b;font-size:10px">${{t.station || ""}}</span></td>
    <td><b>${{t.bin_label || ""}}</b></td>
    <td><span class="pill ${{modeCls}}">${{modeLabel}}</span></td>
    <td class="num">$${{t.entry_price.toFixed(3)}}</td>
    <td class="num">$${{t.current_price.toFixed(3)}}</td>
    <td class="num">$${{t.stake.toFixed(2)}}</td>
    <td class="num">${{t.shares.toFixed(2)}}</td>
    <td class="num">$${{t.current_value.toFixed(2)}}</td>
    <td class="num edge ${{pnlCls}}">${{pnlSign}}$${{t.pnl_usd.toFixed(2)}}</td>
    <td class="num edge ${{pnlCls}}">${{pnlSign}}${{(t.pnl_pct*100).toFixed(1)}}%</td>
    <td><span class="pill ${{(t.live_status || "").toLowerCase()}}">${{(t.live_status || "—").toUpperCase()}}</span></td>
    <td class="tstamp">${{(t.live_order_id || "").slice(0,14)}}</td>
  </tr>`;
}}

function renderTradesTable() {{
  const cityFilter = $("f-city").value;
  let trades = buildTradesData().filter(t =>
    matchesBuyMode({{action: t.action, mode: t.action === "LIVE_BUY" ? "live" : "paper"}})
    && (!cityFilter || t.city === cityFilter)
  );
  trades.sort((a, b) => {{
    let av = a[TRADES_SORT_KEY], bv = b[TRADES_SORT_KEY];
    if (typeof av === "number") return TRADES_SORT_DIR * (av - bv);
    return TRADES_SORT_DIR * String(av || "").localeCompare(String(bv || ""));
  }});
  trades = trades.slice(0, 500);
  $("count").textContent = trades.length + " trades";
  let html = "";
  let lastCity = null;
  for (const t of trades) {{
    const breakNow = lastCity !== null && lastCity !== t.city;
    html += tradeRow(t, breakNow);
    lastCity = t.city;
  }}
  $("trades-tbody").innerHTML = html ||
    '<tr><td colspan="13" class="empty">No trades match filters</td></tr>';
  // Update sort indicator on the trades table headers
  document.querySelectorAll("#trades-table th").forEach(th => {{
    th.classList.remove("sorted-asc","sorted-desc");
    if (th.dataset.key === TRADES_SORT_KEY)
      th.classList.add(TRADES_SORT_DIR > 0 ? "sorted-asc" : "sorted-desc");
  }});
}}

// Wire trades-table column sort handlers (same pattern as signals table)
document.querySelectorAll("#trades-table th").forEach(th => {{
  th.addEventListener("click", () => {{
    if (!th.dataset.key) return;
    if (TRADES_SORT_KEY === th.dataset.key) TRADES_SORT_DIR = -TRADES_SORT_DIR;
    else {{ TRADES_SORT_KEY = th.dataset.key; TRADES_SORT_DIR = 1; }}
    renderTradesTable();
  }});
}});

// ===== View toggle (Signals vs Trades) =====
function setView(v) {{
  VIEW_MODE = v;
  ["signals", "trades"].forEach(x => {{
    const btn = $("view-" + x);
    btn.classList.toggle("active", x === v);
  }});
  // Show the right table, hide the other
  $("sig-table").style.display    = (v === "signals" ? "" : "none");
  $("trades-table").style.display = (v === "trades"  ? "" : "none");
  renderAll();
}}
$("view-signals").addEventListener("click", () => setView("signals"));
$("view-trades").addEventListener("click",  () => setView("trades"));

// Date dropdown drives the whole dashboard
$("f-date").addEventListener("change", () => {{
  SELECTED_DATE = $("f-date").value;
  recomputeDerived();
  renderAll();
}});
// City filter affects panels too, so it triggers renderAll (not just table)
$("f-city").addEventListener("change", () => renderAll());
// These only affect the signals table
["f-action","f-edge","f-buys","f-latest"].forEach(id =>
  $(id).addEventListener("input", renderSigTable));

// Mode toggle handlers
function setMode(m) {{
  MODE_FILTER = m;
  ["paper","live","both"].forEach(x => {{
    const btn = $(`mode-${{x}}`);
    btn.classList.toggle("active", x === m);
    btn.className = btn.className.replace(/\\b(paper|live|both)\\b/g, '').trim();
    if (x === m) btn.classList.add("active", x);
  }});
  renderAll();
}}
$("mode-paper").addEventListener("click", () => setMode("paper"));
$("mode-live").addEventListener("click",  () => setMode("live"));
$("mode-both").addEventListener("click",  () => setMode("both"));

function renderAll() {{
  refreshDateDropdown();   // SELECTED_DATE may shift if today rolled over
  recomputeDerived();       // re-filter DATE_SIGNALS for new SELECTED_DATE
  refreshCityDropdown();    // city options depend on SELECTED_DATE
  updateSectionTitles();
  renderKPIs();
  renderCityPanels();
  if (VIEW_MODE === 'trades') renderTradesTable();
  else                          renderSigTable();
}}
renderAll();

// ===== Silent refresh (no blank page on update) =====
// Re-fetch the same URL every REFRESH_SEC, extract the SIGNALS/LIVE_ORDERS
// blobs from the new HTML, swap them in place, and re-render.  Preserves:
//   * mode toggle state
//   * sort order
//   * all filter inputs
//   * scroll position
//   * dropdown open/closed state
function extractJsonBlob(text, varName) {{
  // Match: let VAR = [...]; or let VAR = {{...}};
  // Use a tolerant pattern: capture everything between the assignment and
  // the next ";\\n" at top-level.
  const re = new RegExp('let\\\\s+' + varName + '\\\\s*=\\\\s*(\\\\[[\\\\s\\\\S]*?\\\\]);', 'm');
  const m = text.match(re);
  if (!m) return null;
  try {{ return JSON.parse(m[1]); }} catch (e) {{ return null; }}
}}

async function silentRefresh() {{
  try {{
    const resp = await fetch(window.location.href, {{cache: "no-store"}});
    if (!resp.ok) return;
    const text = await resp.text();
    const newSignals    = extractJsonBlob(text, "SIGNALS");
    const newLiveOrders = extractJsonBlob(text, "LIVE_ORDERS");
    if (newSignals === null && newLiveOrders === null) return;
    if (newSignals !== null)    SIGNALS = newSignals;
    if (newLiveOrders !== null) LIVE_ORDERS = newLiveOrders;
    recomputeDerived();
    renderAll();
    // Update generated-at timestamp in header if present
    const tsMatch = text.match(/generated ([0-9-]+ [0-9:]+ UTC)/);
    if (tsMatch) {{
      document.querySelectorAll("header .meta").forEach(el => {{
        const html = el.innerHTML.replace(/generated [0-9-]+ [0-9:]+ UTC/, "generated " + tsMatch[1]);
        if (html !== el.innerHTML) el.innerHTML = html;
      }});
    }}
  }} catch (e) {{
    console.error("silent refresh failed:", e);
  }}
}}

if (REFRESH_SEC > 0) {{
  setInterval(silentRefresh, REFRESH_SEC * 1000);
}}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# HTTP server + watch loop
# ---------------------------------------------------------------------------

def serve(path: str, port: int, regenerate_fn=None, watch_sec: int | None = None) -> None:
    serve_dir = os.path.dirname(os.path.abspath(path)) or "."
    fname = os.path.basename(path)
    os.chdir(serve_dir)

    class Reusable(socketserver.TCPServer):
        allow_reuse_address = True

    print()
    print("=" * 72)
    print(f"  Serving {path} on 0.0.0.0:{port}")
    print(f"  Tailscale URL:  http://<vps-tailscale-ip>:{port}/{fname}")
    if regenerate_fn and watch_sec:
        print(f"  Watch:          regenerating HTML every {watch_sec}s")
    print("=" * 72)

    if regenerate_fn and watch_sec:
        import threading, time as _time
        def _watcher():
            while True:
                _time.sleep(watch_sec)
                try:
                    regenerate_fn()
                except Exception as e:
                    log.warning(f"regenerate failed: {e}")
        threading.Thread(target=_watcher, daemon=True).start()

    with Reusable(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_port  = int(os.getenv("DASHBOARD_PORT", "8082"))
    default_watch = int(os.getenv("DASHBOARD_WATCH_SEC", "30"))
    default_days  = int(os.getenv("DASHBOARD_DAYS", "1"))

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=default_days)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--html", default=os.path.join(_BOT_DIR, "data",
                                                    "predictor_dashboard.html"))
    p.add_argument("--serve", type=int, default=default_port, nargs="?")
    p.add_argument("--no-serve", action="store_true")
    p.add_argument("--watch", type=int, default=default_watch)
    args = p.parse_args()
    if args.no_serve:
        args.serve = None

    def regenerate() -> str:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        signals = load_signals(args.db, since)
        live_orders = load_live_orders(args.db, since)
        log.info(f"regenerate: {len(signals)} signals + "
                  f"{len(live_orders)} live orders since {since.isoformat()}")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        html = build_dashboard(signals, live_orders, generated_at,
                                auto_refresh_sec=args.watch or None)
        os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(html)
        log.info(f"wrote {os.path.getsize(args.html)/1024:.0f} KB to {args.html}")
        return args.html

    regenerate()

    if args.serve:
        serve(args.html, args.serve,
              regenerate_fn=regenerate if args.watch else None,
              watch_sec=args.watch or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())