"""
predictor_dashboard.py — Visual dashboard for the intraday predictor.

Reads from:
  paper_predictor_signals  (every scan's per-bin decisions + context)
  live_predictor_orders    (every order placed in live mode)

Renders a dark-mode HTML page with:
  * Header KPIs:  signals today, BUY count, deployed $, avg edge
  * Mode banner: PAPER vs LIVE indicator + current config snapshot
  * Per-city summary cards: BUY counts + last-scan timestamp
  * Live orders table (if any) — status / fill / errors
  * Filterable signals table — sortable by every column
  * 24-hour timeline chart — scans + BUY decisions over time

Usage:
    cd bot
    python -m scripts.predictor_dashboard
    python -m scripts.predictor_dashboard --days 3
    python -m scripts.predictor_dashboard --html data/predictor.html
    python -m scripts.predictor_dashboard --serve 8082
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

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("predictor_dash")


DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #0f172a; color: #e2e8f0; font-size: 13px; }

header { background: linear-gradient(90deg, #1e293b, #0f172a);
         padding: 14px 24px; border-bottom: 1px solid #334155;
         display: flex; align-items: center; justify-content: space-between; }
header h1 { margin: 0; font-size: 18px; font-weight: 600; color: white; }
header .meta { font-family: monospace; font-size: 11px; color: #94a3b8; text-align: right; }

.mode-banner { padding: 10px 24px; font-size: 13px; font-weight: 600;
               text-align: center; border-bottom: 1px solid #334155; }
.mode-banner.paper { background: #1e3a8a; color: #dbeafe; }
.mode-banner.live { background: #7f1d1d; color: #fef2f2; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% {opacity:1} 50% {opacity:0.85} }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px; padding: 14px 24px; background: #0f172a; }
.kpi { background: #1e293b; padding: 12px 16px; border-radius: 6px;
       border-left: 3px solid #4338ca; }
.kpi .label { font-size: 9px; color: #94a3b8; text-transform: uppercase;
              letter-spacing: 0.5px; font-weight: 600; }
.kpi .val { font-size: 24px; font-weight: 700; margin-top: 4px;
            font-family: monospace; color: white; }
.kpi .sub { font-size: 10px; color: #94a3b8; margin-top: 2px; font-family: monospace; }
.kpi.buy { border-left-color: #22c55e; }
.kpi.buy .val { color: #4ade80; }
.kpi.skip { border-left-color: #6b7280; }
.kpi.avoid { border-left-color: #ef4444; }
.kpi.avoid .val { color: #f87171; }

.city-grid { display: grid;
             grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: 8px; padding: 4px 24px 18px; }
.city-card { background: #1e293b; border-radius: 6px; padding: 10px 12px;
             border-left: 3px solid #475569; font-size: 12px; }
.city-card.has-buy { border-left-color: #22c55e; background: #14361f; }
.city-card.has-live { border-left-color: #f59e0b; background: #3a2c14; }
.city-card .name { font-weight: 700; color: white; font-size: 13px; }
.city-card .stats { color: #94a3b8; font-family: monospace; font-size: 11px;
                     margin-top: 4px; }
.city-card .stats .b { color: #4ade80; font-weight: 700; }
.city-card .stats .l { color: #fbbf24; font-weight: 700; }

.section-title { padding: 16px 24px 6px; font-size: 11px; font-weight: 700;
                  color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

.filters { background: #1e293b; padding: 10px 24px; display: flex;
           gap: 16px; flex-wrap: wrap; align-items: center; font-size: 12px;
           border-bottom: 1px solid #334155; position: sticky; top: 0; z-index: 10; }
.filters label { font-weight: 600; color: #cbd5e1; margin-right: 4px; }
.filters select, .filters input { padding: 4px 8px; font-size: 12px;
           background: #0f172a; color: white; border: 1px solid #475569;
           border-radius: 4px; }
.filters .count { color: #94a3b8; font-family: monospace; margin-left: auto; }

table { width: calc(100% - 48px); margin: 12px 24px; background: #1e293b;
        border-collapse: collapse; border-radius: 6px; overflow: hidden;
        font-size: 12px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 10px;
     text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer;
     user-select: none; }
th:hover { background: #1e293b; color: white; }
th.sorted-asc::after { content: ' ▲'; color: #818cf8; }
th.sorted-desc::after { content: ' ▼'; color: #818cf8; }
td.num { font-family: monospace; text-align: right; }
td.tstamp { font-family: monospace; font-size: 11px; color: #94a3b8; }

tr.LIVE_BUY { background: rgba(245,158,11,0.18); border-left: 3px solid #f59e0b; }
tr.PAPER_BUY { background: rgba(34,197,94,0.10); }
tr.AVOID { background: rgba(239,68,68,0.06); color: #94a3b8; }
tr.SKIP { color: #64748b; }
tr:hover { background: #273449 !important; }

td .pill { display: inline-block; padding: 2px 7px; border-radius: 10px;
           font-size: 10px; font-weight: 700; }
.pill.LIVE_BUY { background: #f59e0b; color: white; }
.pill.PAPER_BUY { background: #22c55e; color: white; }
.pill.SKIP { background: #475569; color: #cbd5e1; }
.pill.AVOID { background: #ef4444; color: white; }
.pill.placed { background: #22c55e; color: white; }
.pill.filled { background: #16a34a; color: white; }
.pill.failed { background: #ef4444; color: white; }
.pill.error { background: #dc2626; color: white; }
.pill.skip { background: #475569; color: #cbd5e1; }

.edge { font-family: monospace; font-weight: 700; }
.edge.pos { color: #4ade80; }
.edge.neg { color: #f87171; }

.empty { text-align: center; color: #64748b; padding: 32px 0; font-size: 13px; }
.timeline { padding: 0 24px 18px; }
.timeline svg { background: #1e293b; border-radius: 6px; }
"""


def _ensure_tables(db: str) -> None:
    """Create predictor tables if missing.  Idempotent — safe to call every
    time.  Lets the dashboard run before the scheduler's first scan has
    populated the schema."""
    try:
        from scheduled_predictor import ensure_schema  # type: ignore
        ensure_schema()
    except Exception as e:
        log.debug(f"ensure_schema import failed (non-fatal): {e}")


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
            log.warning(f"paper_predictor_signals not readable: {e} — "
                         "treating as empty.  Bot probably hasn't run a scan yet.")
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


def build_dashboard(signals: list[dict], live_orders: list[dict],
                     days: int, generated_at_utc: str) -> str:
    # Aggregate stats
    by_action: dict[str, int] = defaultdict(int)
    by_city_today: dict[str, dict] = defaultdict(lambda: {
        "n_total": 0, "n_buy": 0, "n_live_buy": 0, "last_scan": None,
        "deployed": 0.0,
    })
    deployed_today = 0.0
    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for s in signals:
        by_action[s["action"]] += 1
        d = s["scanned_at_utc"][:10] if s["scanned_at_utc"] else ""
        if d == today_date:
            c = by_city_today[s["city"]]
            c["n_total"] += 1
            if s["action"] in ("PAPER_BUY", "LIVE_BUY"):
                c["n_buy"] += 1
                c["deployed"] += s.get("recommended_stake_usd") or 0
                deployed_today += s.get("recommended_stake_usd") or 0
            if s["action"] == "LIVE_BUY":
                c["n_live_buy"] += 1
            if c["last_scan"] is None or s["scanned_at_utc"] > c["last_scan"]:
                c["last_scan"] = s["scanned_at_utc"]

    buys = [s for s in signals if s["action"] in ("PAPER_BUY", "LIVE_BUY")]
    avg_edge = sum(s["edge"] for s in buys) / len(buys) if buys else 0.0
    mode_is_live = any(s["mode"] == "live" for s in signals[:50])
    current_mode = "live" if mode_is_live else "paper"

    # KPI ints
    n_signals = len(signals)
    n_paper_buy = by_action.get("PAPER_BUY", 0)
    n_live_buy = by_action.get("LIVE_BUY", 0)
    n_skip = by_action.get("SKIP", 0)
    n_avoid = by_action.get("AVOID", 0)

    # 24h timeline (signal counts per hour, BUYs highlighted)
    by_hour_total: dict[str, int] = defaultdict(int)
    by_hour_buy:   dict[str, int] = defaultdict(int)
    for s in signals[-2000:]:   # last 2000 rows for speed
        ts = s["scanned_at_utc"]
        if not ts:
            continue
        hr_key = ts[:13]   # YYYY-MM-DDTHH
        by_hour_total[hr_key] += 1
        if s["action"] in ("PAPER_BUY", "LIVE_BUY"):
            by_hour_buy[hr_key] += 1

    hours = sorted(by_hour_total.keys())[-48:]   # last 48h
    timeline_data = [{"h": h, "n": by_hour_total[h], "buy": by_hour_buy[h]}
                      for h in hours]
    timeline_svg = render_timeline_svg(timeline_data)

    # City cards
    city_cards_html = "".join(
        render_city_card(c, s) for c, s in sorted(by_city_today.items())
    ) or '<div class="empty">No scans yet today</div>'

    # Live orders table
    live_rows = "".join(render_live_order_row(o) for o in live_orders[:50])

    # Filter inputs need to know available cities/dates/actions
    cities_in_data = sorted({s["city"] for s in signals if s["city"]})
    dates_in_data  = sorted({s["scanned_at_utc"][:10] for s in signals
                              if s["scanned_at_utc"]}, reverse=True)
    actions_in_data = ["LIVE_BUY", "PAPER_BUY", "SKIP", "AVOID"]

    sig_json = json.dumps(signals, default=str, separators=(",", ":"))

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Predictor Signals Dashboard</title>
<style>{DASHBOARD_CSS}</style></head><body>

<header>
  <div>
    <h1>Intraday Predictor — Signals Dashboard</h1>
    <div style="font-size:11px;color:#94a3b8;margin-top:3px">
      Last {days}d · paper_predictor_signals + live_predictor_orders
    </div>
  </div>
  <div class="meta">generated {generated_at_utc}<br>
    {n_signals:,} signals · {n_live_buy + n_paper_buy:,} BUYs
  </div>
</header>

<div class="mode-banner {current_mode}">
  CURRENT MODE: <b>{current_mode.upper()}</b>
  {' &nbsp;·&nbsp; LIVE orders are real CLOB submissions' if current_mode == 'live' else ' &nbsp;·&nbsp; no real orders being placed'}
</div>

<div class="kpis">
  <div class="kpi"><div class="label">Signals (window)</div>
    <div class="val">{n_signals:,}</div></div>
  <div class="kpi buy"><div class="label">LIVE BUYs</div>
    <div class="val">{n_live_buy}</div>
    <div class="sub">${deployed_today:.2f} deployed today</div></div>
  <div class="kpi buy"><div class="label">PAPER BUYs</div>
    <div class="val">{n_paper_buy}</div></div>
  <div class="kpi skip"><div class="label">SKIPs (gated)</div>
    <div class="val">{n_skip:,}</div></div>
  <div class="kpi avoid"><div class="label">AVOID (neg edge)</div>
    <div class="val">{n_avoid}</div></div>
  <div class="kpi"><div class="label">Avg edge on BUYs</div>
    <div class="val">{avg_edge*100:+.1f}%</div></div>
</div>

<div class="section-title">Per-city snapshot (today only)</div>
<div class="city-grid">{city_cards_html}</div>

<div class="section-title">Last 48 hours — scans (gray) and BUYs (green)</div>
<div class="timeline">{timeline_svg}</div>

{"<div class='section-title'>Live order log</div>" if live_orders else ""}
{f"<table><thead><tr><th>placed at</th><th>city</th><th>bin</th><th>stake</th><th>limit</th><th>status</th><th>order id</th><th>error</th></tr></thead><tbody>{live_rows}</tbody></table>" if live_orders else ""}

<div class="section-title">All signals — filter and sort</div>

<div class="filters">
  <div><label>City</label><select id="f-city"><option value="">All</option>
    {"".join(f'<option>{c}</option>' for c in cities_in_data)}</select></div>
  <div><label>Date</label><select id="f-date"><option value="">All</option>
    {"".join(f'<option>{d}</option>' for d in dates_in_data)}</select></div>
  <div><label>Action</label><select id="f-action"><option value="">All</option>
    {"".join(f'<option>{a}</option>' for a in actions_in_data)}</select></div>
  <div><label>Min edge</label>
    <input id="f-edge" type="number" step="0.05" value="-1" style="width:70px"></div>
  <div><label>Buys only</label>
    <input id="f-buys" type="checkbox"></div>
  <div class="count" id="count">—</div>
</div>

<table id="sig-table">
  <thead><tr>
    <th data-key="scanned_at_utc">Scanned (UTC)</th>
    <th data-key="city">City</th>
    <th data-key="event_date">Event date</th>
    <th data-key="bin_label">Bin</th>
    <th data-key="our_prob">Our P</th>
    <th data-key="market_prob">Mkt P</th>
    <th data-key="edge">Edge</th>
    <th data-key="liquidity_usd">Liquidity</th>
    <th data-key="recommended_stake_usd">Stake</th>
    <th data-key="action">Action</th>
    <th data-key="gate_blocked_by">Gate / Reason</th>
  </tr></thead>
  <tbody id="sig-tbody"></tbody>
</table>

<script>
const SIGNALS = {sig_json};
const $ = id => document.getElementById(id);
let SORT_KEY = "scanned_at_utc", SORT_DIR = -1;

function row(s) {{
  const eClass = s.edge >= 0 ? "pos" : "neg";
  const eStr = (s.edge >= 0 ? "+" : "") + (s.edge*100).toFixed(1) + "%";
  const stake = s.recommended_stake_usd ? "$" + s.recommended_stake_usd.toFixed(2) : "—";
  const gate = s.gate_blocked_by || "";
  return `<tr class="${{s.action}}">
    <td class="tstamp">${{s.scanned_at_utc ? s.scanned_at_utc.slice(0,16).replace('T',' ') : ''}}</td>
    <td><b>${{s.city}}</b><br><span style="color:#64748b;font-size:10px">${{s.settlement_station || ''}}</span></td>
    <td class="tstamp">${{s.event_date || ''}}</td>
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

function render() {{
  const city = $("f-city").value;
  const date = $("f-date").value;
  const act  = $("f-action").value;
  const me   = parseFloat($("f-edge").value);
  const buys = $("f-buys").checked;
  let rows = SIGNALS.filter(s =>
    (!city || s.city === city)
    && (!date || (s.scanned_at_utc||'').startsWith(date))
    && (!act  || s.action === act)
    && (isNaN(me) || s.edge >= me)
    && (!buys || s.action === "PAPER_BUY" || s.action === "LIVE_BUY")
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (typeof av === "number") return SORT_DIR * (av - bv);
    return SORT_DIR * String(av || '').localeCompare(String(bv || ''));
  }});
  rows = rows.slice(0, 500);
  $("count").textContent = rows.length + " / " + SIGNALS.length;
  $("sig-tbody").innerHTML = rows.length ? rows.map(row).join("")
    : '<tr><td colspan="11" class="empty">No signals match filters</td></tr>';
  document.querySelectorAll("th").forEach(th => {{
    th.classList.remove("sorted-asc","sorted-desc");
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? "sorted-asc" : "sorted-desc");
  }});
}}

document.querySelectorAll("th").forEach(th => {{
  th.addEventListener("click", () => {{
    if (!th.dataset.key) return;
    if (SORT_KEY === th.dataset.key) SORT_DIR = -SORT_DIR;
    else {{ SORT_KEY = th.dataset.key; SORT_DIR = 1; }}
    render();
  }});
}});
["f-city","f-date","f-action","f-edge","f-buys"].forEach(id =>
  $(id).addEventListener("input", render));
render();
</script>
</body></html>"""


def render_timeline_svg(data: list[dict], w: int = 1200, h: int = 130) -> str:
    if not data:
        return "<div class='empty'>No timeline data yet</div>"
    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 22
    iw = w - pad_l - pad_r
    ih = h - pad_t - pad_b
    n = len(data)
    max_n = max((d["n"] for d in data), default=1) or 1
    bw = iw / n
    s = [f"<svg viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg' width='100%' height='{h}'>"]
    for i in range(4):
        v = max_n * i / 3
        y = pad_t + (1 - i/3) * ih
        s.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{w-pad_r}' y2='{y:.1f}' "
                  f"stroke='#334155' stroke-width='0.5'/>")
        s.append(f"<text x='{pad_l-4}' y='{y+3:.1f}' font-size='9' "
                  f"text-anchor='end' fill='#64748b' font-family='monospace'>{int(v)}</text>")
    for i, d in enumerate(data):
        x = pad_l + i * bw
        h_total = (d["n"] / max_n) * ih
        h_buy   = (d["buy"] / max_n) * ih if d["buy"] else 0
        # gray total bar
        s.append(f"<rect x='{x+1:.1f}' y='{pad_t + ih - h_total:.1f}' "
                  f"width='{max(1, bw-2):.1f}' height='{h_total:.1f}' "
                  f"fill='#475569'/>")
        # green BUY overlay
        if h_buy > 0:
            s.append(f"<rect x='{x+1:.1f}' y='{pad_t + ih - h_buy:.1f}' "
                      f"width='{max(1, bw-2):.1f}' height='{h_buy:.1f}' "
                      f"fill='#22c55e'/>")
    # x labels: every 6 hours
    for i in range(0, n, max(1, n // 8)):
        x = pad_l + i * bw + bw/2
        lbl = data[i]["h"][11:13] + ":00"
        s.append(f"<text x='{x:.1f}' y='{h-pad_b+14}' font-size='9' "
                  f"text-anchor='middle' fill='#64748b' font-family='monospace'>{lbl}</text>")
    s.append("</svg>")
    return "".join(s)


def render_city_card(city: str, stats: dict) -> str:
    cls = "city-card"
    if stats["n_live_buy"] > 0:
        cls += " has-live"
    elif stats["n_buy"] > 0:
        cls += " has-buy"
    last = (stats["last_scan"] or "")[11:16] if stats["last_scan"] else "—"
    return (
        f'<div class="{cls}">'
        f'<div class="name">{city}</div>'
        f'<div class="stats">'
        f'scans <b>{stats["n_total"]}</b> · '
        f'<span class="b">{stats["n_buy"]}B</span>'
        + (f' · <span class="l">{stats["n_live_buy"]}L</span>' if stats["n_live_buy"] else '')
        + f'<br>${stats["deployed"]:.2f} dep · last {last}UTC'
        f'</div></div>'
    )


def render_live_order_row(o: dict) -> str:
    status = (o.get("status") or "?").lower()
    stake = f'${(o.get("stake_usd") or 0):.2f}'
    lim = f'{(o.get("limit_price") or 0):.4f}' if o.get("limit_price") else "—"
    oid = (o.get("order_id") or "")[:18]
    err = (o.get("error") or "")[:60]
    return (
        f'<tr><td class="tstamp">{(o.get("placed_at_utc") or "")[:19].replace("T"," ")}</td>'
        f'<td><b>{o.get("city","")}</b></td>'
        f'<td>{o.get("bin_label","")}</td>'
        f'<td class="num">{stake}</td>'
        f'<td class="num">{lim}</td>'
        f'<td><span class="pill {status}">{status.upper()}</span></td>'
        f'<td class="tstamp">{oid}</td>'
        f'<td style="color:#f87171;font-size:11px">{err}</td></tr>'
    )


def serve(path: str, port: int) -> None:
    serve_dir = os.path.dirname(os.path.abspath(path)) or "."
    fname = os.path.basename(path)
    os.chdir(serve_dir)

    class Reusable(socketserver.TCPServer):
        allow_reuse_address = True

    print()
    print("=" * 72)
    print(f"  Serving {path} on port {port}")
    print(f"  SSH tunnel:  ssh -L {port}:localhost:{port} <user>@<vps>")
    print(f"  Browser:     http://localhost:{port}/{fname}")
    print("  Ctrl-C to stop.")
    print("=" * 72)

    with Reusable(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=3,
                   help="Lookback window in days (default: 3)")
    p.add_argument("--db", default=DB_PATH,
                   help=f"DB path (default: {DB_PATH})")
    p.add_argument("--html", default=os.path.join(_BOT_DIR, "data",
                                                    "predictor_dashboard.html"),
                   help="Output HTML path")
    p.add_argument("--serve", type=int, metavar="PORT",
                   help="After writing, start an HTTP server on PORT")
    args = p.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    signals = load_signals(args.db, since)
    live_orders = load_live_orders(args.db, since)
    log.info(f"loaded {len(signals)} signals + {len(live_orders)} live orders "
             f"since {since.isoformat()}")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_dashboard(signals, live_orders, args.days, generated_at)
    os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
    with open(args.html, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info(f"wrote {os.path.getsize(args.html)/1024:.0f} KB dashboard to {args.html}")

    if args.serve:
        serve(args.html, args.serve)
    return 0


if __name__ == "__main__":
    sys.exit(main())