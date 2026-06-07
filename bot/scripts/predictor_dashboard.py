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
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

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

    refresh_tag = (f'<meta http-equiv="refresh" content="{auto_refresh_sec}">'
                    if auto_refresh_sec else "")
    refresh_badge = (f'<span style="color:#94a3b8;font-size:10px">auto-refresh: {auto_refresh_sec}s</span>'
                      if auto_refresh_sec else "")

    return f"""<!doctype html><html><head><meta charset="utf-8">
{refresh_tag}
<title>Predictor Dashboard</title>
<style>{DASHBOARD_CSS}</style></head><body>

<header>
  <div style="display:flex;align-items:center;gap:18px">
    <h1>Predictor Dashboard</h1>
    <div class="mode-toggle">
      <button id="mode-paper" class="active paper">Paper</button>
      <button id="mode-live">Live</button>
      <button id="mode-both">Both</button>
    </div>
  </div>
  <div class="meta">generated {generated_at_utc} {refresh_badge}<br>
    <span id="hdr-counts">—</span>
  </div>
</header>

<div class="kpis" id="kpis"></div>

<div class="section-title">Per-city panels (today)</div>
<div class="city-grid" id="city-grid"></div>

<div class="section-title">Live order log (today)</div>
<div id="live-orders-section"></div>

<div class="section-title">All signals (today)</div>
<div class="filters">
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

<script>
const SIGNALS = {sig_json};
const LIVE_ORDERS = {live_json};
const CITIES = {cities_json};
const $ = id => document.getElementById(id);

// Active mode filter for the entire dashboard: 'paper' | 'live' | 'both'
let MODE_FILTER = "paper";

// UTC today's date string — used to filter signals to today only
const today = new Date().toISOString().slice(0, 10);
const TODAY_SIGNALS = SIGNALS.filter(s =>
  s.scanned_at_utc && s.scanned_at_utc.slice(0, 10) === today);
const TODAY_LIVE_ORDERS = LIVE_ORDERS.filter(o =>
  o.placed_at_utc && o.placed_at_utc.slice(0, 10) === today);

// Most recent scan timestamp (for the "latest only" filter)
const LATEST_SCAN_TS = TODAY_SIGNALS.length
  ? TODAY_SIGNALS.reduce((m, s) => s.scanned_at_utc > m ? s.scanned_at_utc : m,
                          TODAY_SIGNALS[0].scanned_at_utc)
  : "";

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
// For each city, build a "today snapshot": city meta + bins from the latest
// scan + any BUYs (in the current mode) for those bins, with computed P&L.
function buildCitySnapshot(cityMeta) {{
  const city = cityMeta.city;
  const todayInCity = TODAY_SIGNALS.filter(s => s.city === city);
  if (todayInCity.length === 0) {{
    return {{...cityMeta, hasData: false}};
  }}
  // Latest scan for this city
  const latestTs = todayInCity.reduce((m, s) =>
    s.scanned_at_utc > m ? s.scanned_at_utc : m, todayInCity[0].scanned_at_utc);
  const latestRows = todayInCity.filter(s => s.scanned_at_utc === latestTs);

  // Sort latest rows: top-our_p bin first, then descending
  latestRows.sort((a, b) => b.our_prob - a.our_prob);

  // Find BUY rows (in current mode filter) for these bins.  An event can
  // have at most MAX_BINS_PER_EVENT buys today; we look up the earliest
  // BUY row per (contract_id) for entry price.
  const buyRows = todayInCity.filter(matchesBuyMode);
  const buyByContract = {{}};
  for (const b of buyRows) {{
    if (!buyByContract[b.contract_id] ||
        b.scanned_at_utc < buyByContract[b.contract_id].scanned_at_utc) {{
      buyByContract[b.contract_id] = b;
    }}
  }}

  // Build bin rows with P&L if bought
  const bins = latestRows.map((r, i) => {{
    const buy = buyByContract[r.contract_id];
    let entry_price = null, stake_usd = null, pnl_usd = null, pnl_pct = null;
    if (buy) {{
      entry_price = buy.market_prob;
      stake_usd   = buy.recommended_stake_usd || 0;
      // Paper P&L (also used for live as a proxy when actual fill not tracked):
      //   shares = stake / entry, P&L = shares * (current - entry)
      //         = stake * (current/entry - 1)
      if (entry_price > 0) {{
        pnl_pct = (r.market_prob / entry_price) - 1;
        pnl_usd = stake_usd * pnl_pct;
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
      pnl_pct:      pnl_pct,
      is_top_p:     i === 0,    // highest our_p
    }};
  }});

  // City-level "current" stats from the latest scan (any row will do)
  const anyRow = latestRows[0];
  return {{
    ...cityMeta,
    hasData:          true,
    eventDate:        anyRow.event_date,
    observedMax:      anyRow.observed_max_c,
    observedPeakHour: anyRow.observed_peak_hour,
    forecastHigh:     anyRow.forecast_high_c,
    forecastPeakHour: anyRow.forecast_peak_hour,
    windOctant:       anyRow.wind_octant,
    bins:             bins,
    nBuys:            bins.filter(b => b.action).length,
    nLiveBuys:        bins.filter(b => b.action === "LIVE_BUY").length,
    totalDeployed:    bins.reduce((s, b) => s + (b.stake_usd || 0), 0),
    totalPnl:         bins.reduce((s, b) => s + (b.pnl_usd || 0), 0),
    latestScanLocalHr: anyRow.current_hour_local,
  }};
}}

// ===== KPI rendering =====
function renderKPIs() {{
  const buys = TODAY_SIGNALS.filter(matchesBuyMode);
  const nBuys = buys.length;
  const deployed = buys.reduce((s, b) => s + (b.recommended_stake_usd || 0), 0);

  // Compute total P&L across all buys in current mode
  let totalPnl = 0;
  for (const city of CITIES) {{
    const snap = buildCitySnapshot(city);
    totalPnl += snap.totalPnl || 0;
  }}

  const avgEdge = nBuys ? buys.reduce((s, b) => s + b.edge, 0) / nBuys : 0;
  const nSkips  = TODAY_SIGNALS.filter(s => s.action === "SKIP" && matchesMode(s)).length;

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
      <div class="val">${{TODAY_SIGNALS.filter(matchesMode).length.toLocaleString()}}</div>
      <div class="sub">${{TODAY_SIGNALS.length ? 'last: ' + LATEST_SCAN_TS.slice(11,16) + 'Z' : ''}}</div></div>
  `;
  $("hdr-counts").textContent =
    `${{nBuys}} BUYs · $${{deployed.toFixed(0)}} dep · ${{pnlSign}}$${{totalPnl.toFixed(2)}} P&L`;
}}

// ===== City panel rendering =====
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

  const obsT = snap.observedMax !== null && snap.observedMax !== undefined
    ? snap.observedMax.toFixed(1) + '°C @ ' + snap.observedPeakHour + ':00' : '—';
  const fcstT = snap.forecastHigh !== null && snap.forecastHigh !== undefined
    ? snap.forecastHigh.toFixed(1) + '°C @ ' + snap.forecastPeakHour + ':00' : '—';
  const obsClass = (snap.observedMax !== null && snap.forecastHigh !== null
                     && snap.observedMax >= snap.forecastHigh - 0.5) ? "hot" : "";

  const binsRows = snap.bins.map(b => {{
    const ourBar = Math.round(Math.max(0, Math.min(1, b.our_prob)) * 100);
    const mktBar = Math.round(Math.max(0, Math.min(1, b.market_prob)) * 100);
    const edgeClass = b.edge >= 0 ? "edge pos" : "edge neg";
    const edgeStr = (b.edge >= 0 ? "+" : "") + (b.edge*100).toFixed(1) + "%";
    const action = b.action
      ? `<span class="pill ${{b.action}}">${{b.action === 'LIVE_BUY' ? 'LIVE' : 'PAPER'}}</span>`
      : '<span class="pill idle">—</span>';
    const entry = b.entry_price !== null
      ? '$' + b.entry_price.toFixed(3) : '';
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
  }}).join('');

  return `<div class="${{cls}}">
    <div class="city-header">
      <div>
        <div class="name">${{snap.city}}</div>
        <div class="station">${{snap.station}} · ${{snap.tz_label}}${{snap.windOctant ? ' · wind ' + snap.windOctant : ''}}</div>
      </div>
      <div class="now">
        <div class="temp ${{obsClass}}">${{snap.observedMax !== null ? snap.observedMax.toFixed(1) + '°C' : '—'}}</div>
        <div>observed @ ${{snap.observedPeakHour}}:00 local</div>
      </div>
    </div>
    <div class="city-stats">
      <div>Forecast high: <span class="v">${{fcstT}}</span></div>
      <div>Event date: <span class="v">${{snap.eventDate}}</span></div>
      <div>Position(s): <span class="v">${{snap.nBuys || 0}}</span>
        ${{snap.totalDeployed > 0 ? '· $' + snap.totalDeployed.toFixed(2) + ' deployed' : ''}}
      </div>
    </div>
    <table class="bins">
      <thead><tr>
        <th>Bin</th><th>Our P</th><th>Mkt P</th><th>Edge</th>
        <th>Action</th><th>Entry</th><th>P&L</th>
      </tr></thead>
      <tbody>${{binsRows}}</tbody>
    </table>
  </div>`;
}}

function renderCityPanels() {{
  const html = CITIES.map(c => renderCityPanel(buildCitySnapshot(c))).join('');
  $("city-grid").innerHTML = html || '<div class="empty">No cities configured</div>';
}}

// ===== Live orders table =====
function renderLiveOrders() {{
  const visible = TODAY_LIVE_ORDERS.filter(o => MODE_FILTER !== "paper");
  if (visible.length === 0) {{
    $("live-orders-section").innerHTML =
      '<div class="empty" style="padding:6px 24px 16px">No live orders today</div>';
    return;
  }}
  const rows = visible.slice(0, 50).map(o => {{
    const status = (o.status || "?").toLowerCase();
    const stake = '$' + (o.stake_usd || 0).toFixed(2);
    const lim = o.limit_price ? o.limit_price.toFixed(4) : "—";
    const oid = (o.order_id || "").slice(0, 18);
    const err = (o.error || "").slice(0, 60);
    return `<tr>
      <td class="tstamp">${{(o.placed_at_utc || "").slice(0,19).replace("T"," ")}}</td>
      <td><b>${{o.city || ""}}</b></td>
      <td>${{o.bin_label || ""}}</td>
      <td class="num">${{stake}}</td>
      <td class="num">${{lim}}</td>
      <td><span class="pill ${{status}}">${{status.toUpperCase()}}</span></td>
      <td class="tstamp">${{oid}}</td>
      <td style="color:#f87171;font-size:11px">${{err}}</td>
    </tr>`;
  }}).join("");
  $("live-orders-section").innerHTML = `<table class="live-orders">
    <thead><tr><th>Placed</th><th>City</th><th>Bin</th><th>Stake</th>
      <th>Limit</th><th>Status</th><th>Order ID</th><th>Error</th></tr></thead>
    <tbody>${{rows}}</tbody></table>`;
}}

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
  let rows = TODAY_SIGNALS.filter(s =>
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
  $("count").textContent = rows.length + " / " + TODAY_SIGNALS.length;

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

// Populate city dropdown
const citiesInData = [...new Set(TODAY_SIGNALS.map(s => s.city).filter(Boolean))].sort();
for (const c of citiesInData) {{
  const o = document.createElement("option"); o.value = o.textContent = c;
  $("f-city").appendChild(o);
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
["f-city","f-action","f-edge","f-buys","f-latest"].forEach(id =>
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
  renderKPIs();
  renderCityPanels();
  renderLiveOrders();
  renderSigTable();
}}
renderAll();
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