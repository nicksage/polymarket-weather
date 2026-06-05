"""
temp_drop_charts.py — Build an interactive HTML page of 24-hour
temperature curves for every "drop" event the bot's cities have seen.

The page is fully self-contained (no server, no internet, no install):
embed the hourly temperature data + a small JS engine that detects drop
events and renders SVG charts in the browser.  All variables — threshold,
window length, end-hour cutoff, magnitude filter, hour filter, held/broken,
sort, limit — are live controls.  Re-running the CLI is only needed to
change the date range or city pre-filter.

Layout:
  ┌─────────────────────────────────────────────────────────────┐
  │  TOP CONTROLS:  threshold | window | after-hour | hold/break │
  │                 magnitude range | hour range | sort | limit  │
  ├──────────┬──────────────────────────────────────────────────┤
  │  CITIES  │  GRID of event cards (filtered by left sidebar)  │
  │  ☑ Tokyo │  ┌──────────┐ ┌──────────┐ ┌──────────┐         │
  │  ☑ NYC   │  │ chart    │ │ chart    │ │ chart    │         │
  │  …       │  └──────────┘ └──────────┘ └──────────┘         │
  └──────────┴──────────────────────────────────────────────────┘

Usage:
    cd bot
    python -m scripts.temp_drop_charts                # full 2 yr × all cities
    python -m scripts.temp_drop_charts --city Lagos Tokyo
    python -m scripts.temp_drop_charts --start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.temp_drop_backtest import DEFAULT_DB_PATH, load_city_list


def _load_hourly(db_path: str, cities: list[str],
                  start: str, end: str) -> dict[str, dict[str, list[float | None]]]:
    """Return {city: {date: [t0..t23] with None for missing hours}}."""
    out: dict[str, dict[str, list[float | None]]] = defaultdict(dict)
    placeholders = ",".join("?" * len(cities))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT city, date_local, hour_local, temp_c
            FROM hourly_temps
            WHERE city IN ({placeholders})
              AND date_local BETWEEN ? AND ?
              AND temp_c IS NOT NULL
            ORDER BY city, date_local, hour_local
            """,
            (*cities, start, end),
        ).fetchall()
    by_pair: dict[tuple[str, str], list[float | None]] = {}
    for r in rows:
        key = (r["city"], r["date_local"])
        if key not in by_pair:
            by_pair[key] = [None] * 24
        h = int(r["hour_local"])
        if 0 <= h < 24:
            by_pair[key][h] = round(float(r["temp_c"]), 2)
    for (city, d), temps in by_pair.items():
        # Only keep days with most of the data
        if sum(1 for t in temps if t is not None) >= 20:
            out[city][d] = temps
    return out


# ---------------------------------------------------------------------------
# HTML / JS
# ---------------------------------------------------------------------------

PAGE_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
  margin: 0; background: #f3f4f6; color: #111827;
}
header {
  background: #111827; color: #f3f4f6; padding: 10px 18px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 10;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
header h1 { margin: 0; font-size: 16px; font-weight: 600; }
header .stats { font-size: 12px; color: #9ca3af; font-family: monospace; }
.controls {
  background: white; padding: 10px 18px; border-bottom: 1px solid #e5e7eb;
  display: flex; flex-wrap: wrap; gap: 14px 22px; font-size: 12px;
  position: sticky; top: 39px; z-index: 9;
}
.ctrl { display: flex; flex-direction: column; gap: 2px; min-width: 130px; }
.ctrl label { font-weight: 600; color: #374151; font-size: 11px; }
.ctrl .row { display: flex; align-items: center; gap: 4px; }
.ctrl input[type=range] { width: 110px; }
.ctrl .val { font-family: monospace; min-width: 36px; color: #1f2937;
             background: #eef2ff; padding: 1px 5px; border-radius: 3px; }
.ctrl input[type=number] { width: 56px; padding: 2px 4px; }
.ctrl select { padding: 2px 4px; }
.layout { display: flex; min-height: calc(100vh - 95px); }
aside {
  width: 200px; background: white; border-right: 1px solid #e5e7eb;
  padding: 12px 10px; overflow-y: auto; max-height: calc(100vh - 95px);
  position: sticky; top: 95px; align-self: flex-start;
}
aside h3 { font-size: 11px; text-transform: uppercase; color: #6b7280;
           margin: 0 0 6px; letter-spacing: 0.5px; }
.city-actions { display: flex; gap: 6px; margin-bottom: 8px; }
.city-actions button { font-size: 10px; padding: 2px 6px; cursor: pointer;
                       background: #eef2ff; border: 1px solid #c7d2fe;
                       border-radius: 3px; color: #4338ca; }
.city-list { display: flex; flex-direction: column; gap: 1px; }
.city-row { display: flex; align-items: center; gap: 5px;
            font-size: 12px; padding: 3px 5px; cursor: pointer;
            border-radius: 3px; user-select: none; }
.city-row:hover { background: #f3f4f6; }
.city-row .count { margin-left: auto; color: #6b7280; font-family: monospace;
                   font-size: 10px; }
.city-row.solo { background: #fef3c7; }
main { flex: 1; padding: 14px; overflow-x: hidden; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
        gap: 12px; }
.card {
  background: white; padding: 8px 10px 4px; border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #16a34a;
}
.card.broken { border-left-color: #dc2626; }
.title { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
.sub { font-size: 10px; color: #6b7280; margin-bottom: 4px; font-family: monospace; }
.pill { display: inline-block; padding: 1px 6px; border-radius: 10px;
        font-size: 10px; margin-left: 6px; vertical-align: middle; }
.pill.held { background: #dcfce7; color: #166534; }
.pill.broken { background: #fee2e2; color: #991b1b; }
.empty { text-align: center; color: #9ca3af; padding: 60px 0; font-size: 14px; }

/* view toggle */
.view-toggle { display: inline-flex; border: 1px solid #c7d2fe; border-radius: 5px;
               overflow: hidden; background: white; }
.view-toggle button { background: white; border: none; padding: 4px 12px;
                      font-size: 12px; cursor: pointer; color: #4338ca;
                      font-weight: 600; }
.view-toggle button.active { background: #4338ca; color: white; }
.view-toggle button:not(.active):hover { background: #eef2ff; }

/* list view */
.day-group { background: white; border-radius: 6px; margin-bottom: 10px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; }
.day-header { background: #1f2937; color: #f3f4f6; padding: 6px 12px;
              font-size: 12px; font-weight: 600; display: flex;
              justify-content: space-between; align-items: center; }
.day-header .stat { font-family: monospace; font-size: 11px; color: #9ca3af; }
.day-header .stat .h { color: #86efac; }
.day-header .stat .b { color: #fca5a5; }
table.events { width: 100%; border-collapse: collapse; font-size: 12px; }
table.events th, table.events td {
  padding: 4px 10px; text-align: left; border-bottom: 1px solid #f3f4f6;
}
table.events th { background: #f9fafb; color: #6b7280; font-weight: 600;
                  font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px; }
table.events tr.broken { background: #fef2f2; }
table.events tr:hover { background: #f3f4f6; }
table.events tr.broken:hover { background: #fee2e2; }
table.events td.num { font-family: monospace; text-align: right; }
table.events td.city { font-weight: 600; }
table.events td.pill-cell { text-align: center; }
table.events td .mini-pill {
  display: inline-block; padding: 1px 6px; border-radius: 10px;
  font-size: 10px; font-weight: 600;
}
table.events td .mini-pill.held { background: #dcfce7; color: #166534; }
table.events td .mini-pill.broken { background: #fee2e2; color: #991b1b; }
"""


PAGE_JS = r"""
// ===== state =====
const $ = id => document.getElementById(id);
let SELECTED_CITIES = new Set(Object.keys(DATA));   // start: all on
let SOLO_CITY = null;                               // when one row is "soloed"
let VIEW = "chart";                                 // "chart" | "list"

const CTRL = {
  threshold:    $("threshold"),
  window:       $("windowH"),
  afterHour:    $("afterHour"),
  magMin:       $("magMin"),
  magMax:       $("magMax"),
  hourMin:      $("hourMin"),
  hourMax:      $("hourMax"),
  status:       $("status"),
  sort:         $("sort"),
  limit:        $("limit"),
};
const VAL = {  // value display elements
  threshold: $("thresholdVal"), window: $("windowVal"),
  afterHour: $("afterHourVal"), magMin: $("magMinVal"),
  magMax: $("magMaxVal"), hourMin: $("hourMinVal"),
  hourMax: $("hourMaxVal"), limit: $("limitVal"),
};

// ===== drop detection (mirrors temp_drop_backtest.py) =====
function detectDropsForDay(temps, threshold, windowH, afterHour) {
  const dayHigh = Math.max(...temps.filter(t => t != null));
  const events = [];
  for (let h = Math.max(afterHour, windowH); h < 24; h++) {
    const tStart = temps[h - windowH];
    const tEnd   = temps[h];
    if (tStart == null || tEnd == null) continue;
    const drop = tStart - tEnd;
    if (drop < threshold) continue;
    let preDropHigh = -Infinity;
    for (let hh = 0; hh <= h - windowH; hh++) {
      if (temps[hh] != null && temps[hh] > preDropHigh) preDropHigh = temps[hh];
    }
    const heldExact = Math.abs(dayHigh - preDropHigh) < 1e-6;
    const overshoot = Math.max(0, dayHigh - preDropHigh);
    events.push({
      dropStartHour: h - windowH,
      dropEndHour: h,
      dropStartTemp: tStart,
      dropEndTemp: tEnd,
      dropMagnitude: drop,
      preDropHigh: preDropHigh,
      dayHigh: dayHigh,
      overshoot: overshoot,
      heldExact: heldExact,
    });
  }
  return events;
}

// ===== compute all events under current params (for all cities, unfiltered) =====
function computeAllEvents() {
  const threshold = parseFloat(CTRL.threshold.value);
  const windowH   = parseInt(CTRL.window.value);
  const afterHour = parseInt(CTRL.afterHour.value);
  const all = [];
  for (const city of Object.keys(DATA)) {
    for (const dateStr of Object.keys(DATA[city])) {
      const temps = DATA[city][dateStr];
      const drops = detectDropsForDay(temps, threshold, windowH, afterHour);
      for (const e of drops) {
        e.city = city; e.date = dateStr;
        all.push(e);
      }
    }
  }
  return all;
}

// ===== filter + sort + slice =====
function applyFilters(events) {
  const magMin   = parseFloat(CTRL.magMin.value);
  const magMax   = parseFloat(CTRL.magMax.value);
  const hMin     = parseInt(CTRL.hourMin.value);
  const hMax     = parseInt(CTRL.hourMax.value);
  const status   = CTRL.status.value;
  const cities   = SOLO_CITY ? new Set([SOLO_CITY]) : SELECTED_CITIES;

  let filtered = events.filter(e =>
    cities.has(e.city)
    && e.dropMagnitude >= magMin && e.dropMagnitude <= magMax
    && e.dropEndHour >= hMin && e.dropEndHour <= hMax
    && (status === "all"
        || (status === "held"   && e.heldExact)
        || (status === "broken" && !e.heldExact))
  );

  const sort = CTRL.sort.value;
  if (sort === "magnitude") filtered.sort((a,b) => b.dropMagnitude - a.dropMagnitude);
  else if (sort === "overshoot") filtered.sort((a,b) => b.overshoot - a.overshoot);
  else if (sort === "hour") filtered.sort((a,b) => a.dropEndHour - b.dropEndHour);
  else if (sort === "city") filtered.sort((a,b) =>
      a.city.localeCompare(b.city) || a.date.localeCompare(b.date));
  else if (sort === "date") filtered.sort((a,b) => b.date.localeCompare(a.date));
  else {  // random
    let seed = 42;
    const rng = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
    filtered = filtered.map(e => [rng(), e]).sort((a,b) => a[0]-b[0]).map(x => x[1]);
  }

  const limit = parseInt(CTRL.limit.value);
  return { all: filtered, shown: filtered.slice(0, limit) };
}

// ===== SVG rendering =====
function renderSVG(event, temps, w=540, h=200) {
  const padL=36, padR=10, padT=10, padB=20;
  const innerW = w - padL - padR, innerH = h - padT - padB;
  const valid = temps.filter(t => t != null);
  let tLo = Math.min(...valid), tHi = Math.max(...valid);
  const pad = Math.max(0.5, (tHi - tLo) * 0.10);
  tLo -= pad; tHi += pad;
  const x = h => padL + (h / 23) * innerW;
  const y = t => padT + (1 - (t - tLo) / (tHi - tLo)) * innerH;

  let s = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border-radius:4px;width:100%;height:auto">`;

  // drop window band
  s += `<rect x="${x(event.dropStartHour).toFixed(1)}" y="${padT}" `
    +  `width="${(x(event.dropEndHour)-x(event.dropStartHour)).toFixed(1)}" `
    +  `height="${innerH}" fill="#cfe2ff" opacity="0.6"/>`;

  // y grid + labels
  for (let i = 0; i < 4; i++) {
    const tv = tLo + (tHi - tLo) * (i / 3);
    const yy = y(tv);
    s += `<line x1="${padL}" y1="${yy.toFixed(1)}" x2="${w-padR}" y2="${yy.toFixed(1)}" stroke="#e5e7eb" stroke-width="0.5"/>`;
    s += `<text x="${padL-3}" y="${(yy+3).toFixed(1)}" font-size="9" text-anchor="end" fill="#6b7280" font-family="monospace">${tv.toFixed(1)}</text>`;
  }
  for (let h2 = 0; h2 < 24; h2 += 3) {
    s += `<text x="${x(h2).toFixed(1)}" y="${h-padB+12}" font-size="9" text-anchor="middle" fill="#6b7280" font-family="monospace">${String(h2).padStart(2,'0')}</text>`;
  }

  // pre-drop high (orange dashed)
  const yPdh = y(event.preDropHigh);
  s += `<line x1="${padL}" y1="${yPdh.toFixed(1)}" x2="${w-padR}" y2="${yPdh.toFixed(1)}" stroke="#e89923" stroke-width="1" stroke-dasharray="3,3"/>`;
  s += `<text x="${w-padR-2}" y="${(yPdh-3).toFixed(1)}" font-size="9" text-anchor="end" fill="#b06d00" font-family="monospace">pre-drop ${event.preDropHigh.toFixed(1)}</text>`;

  // day high (green/red) — only if differs noticeably
  if (Math.abs(event.preDropHigh - event.dayHigh) > 0.05) {
    const color = event.heldExact ? "#16a34a" : "#dc2626";
    const yDh = y(event.dayHigh);
    s += `<line x1="${padL}" y1="${yDh.toFixed(1)}" x2="${w-padR}" y2="${yDh.toFixed(1)}" stroke="${color}" stroke-width="1"/>`;
    s += `<text x="${w-padR-2}" y="${(yDh+10).toFixed(1)}" font-size="9" text-anchor="end" fill="${color}" font-family="monospace">day ${event.dayHigh.toFixed(1)}</text>`;
  }

  // temp curve
  const pts = [];
  for (let h2 = 0; h2 < 24; h2++) {
    if (temps[h2] != null) pts.push(`${x(h2).toFixed(1)},${y(temps[h2]).toFixed(1)}`);
  }
  if (pts.length) s += `<polyline points="${pts.join(' ')}" fill="none" stroke="#1f2937" stroke-width="1.6"/>`;
  for (let h2 = 0; h2 < 24; h2++) {
    if (temps[h2] != null) s += `<circle cx="${x(h2).toFixed(1)}" cy="${y(temps[h2]).toFixed(1)}" r="1.6" fill="#1f2937"/>`;
  }
  s += `</svg>`;
  return s;
}

function renderCard(event) {
  const temps = DATA[event.city][event.date];
  const held = event.heldExact;
  const cls = held ? "card" : "card broken";
  const pill = held
    ? `<span class="pill held">HELD</span>`
    : `<span class="pill broken">BROKEN +${event.overshoot.toFixed(1)}°C</span>`;
  const sub = `drop ${event.dropStartTemp.toFixed(1)}→${event.dropEndTemp.toFixed(1)}°C `
    + `(${event.dropMagnitude.toFixed(1)}°C in ${parseInt(CTRL.window.value)}h, `
    + `ending ${String(event.dropEndHour).padStart(2,'0')}:00) `
    + `| pre-drop high ${event.preDropHigh.toFixed(1)}°C | day high ${event.dayHigh.toFixed(1)}°C`;
  return `<div class="${cls}">
    <div class="title">${event.city} — ${event.date}${pill}</div>
    <div class="sub">${sub}</div>
    ${renderSVG(event, temps)}
  </div>`;
}

// ===== sidebar =====
function renderSidebar(eventsByCity) {
  const wrap = $("city-list");
  wrap.innerHTML = "";
  const cities = Object.keys(DATA).sort();
  for (const c of cities) {
    const n = eventsByCity[c] || 0;
    const checked = SELECTED_CITIES.has(c) ? "checked" : "";
    const soloCls = (SOLO_CITY === c) ? " solo" : "";
    const row = document.createElement("div");
    row.className = "city-row" + soloCls;
    row.innerHTML = `<input type="checkbox" data-city="${c}" ${checked}>
      <span class="name" data-city="${c}">${c}</span>
      <span class="count">${n}</span>`;
    wrap.appendChild(row);
  }
  // wire checkboxes
  wrap.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", (e) => {
      const c = e.target.dataset.city;
      if (e.target.checked) SELECTED_CITIES.add(c); else SELECTED_CITIES.delete(c);
      SOLO_CITY = null;
      rerender();
    });
  });
  // clicking the city name solos it (single-city focus)
  wrap.querySelectorAll(".name").forEach(span => {
    span.addEventListener("click", (e) => {
      const c = e.target.dataset.city;
      SOLO_CITY = (SOLO_CITY === c) ? null : c;
      rerender();
    });
  });
}

// ===== main re-render =====
let CACHED_ALL = null;        // cache events under current threshold/window/afterHour
let CACHED_KEY = "";
function getAllEvents() {
  const k = `${CTRL.threshold.value}|${CTRL.window.value}|${CTRL.afterHour.value}`;
  if (k !== CACHED_KEY) {
    CACHED_KEY = k;
    CACHED_ALL = computeAllEvents();
  }
  return CACHED_ALL;
}

function rerender() {
  // update value pills
  VAL.threshold.textContent = parseFloat(CTRL.threshold.value).toFixed(1) + "°C";
  VAL.window.textContent    = CTRL.window.value + "h";
  VAL.afterHour.textContent = CTRL.afterHour.value + ":00";
  VAL.magMin.textContent    = parseFloat(CTRL.magMin.value).toFixed(1);
  VAL.magMax.textContent    = parseFloat(CTRL.magMax.value).toFixed(1);
  VAL.hourMin.textContent   = CTRL.hourMin.value + ":00";
  VAL.hourMax.textContent   = CTRL.hourMax.value + ":00";
  VAL.limit.textContent     = CTRL.limit.value;

  const all = getAllEvents();
  const { all: matching, shown } = applyFilters(all);

  // event counts by city (using filters EXCEPT the city filter)
  const cFilters = {
    magMin: parseFloat(CTRL.magMin.value), magMax: parseFloat(CTRL.magMax.value),
    hMin: parseInt(CTRL.hourMin.value), hMax: parseInt(CTRL.hourMax.value),
    status: CTRL.status.value,
  };
  const byCity = {};
  for (const e of all) {
    if (e.dropMagnitude < cFilters.magMin || e.dropMagnitude > cFilters.magMax) continue;
    if (e.dropEndHour < cFilters.hMin || e.dropEndHour > cFilters.hMax) continue;
    if (cFilters.status === "held" && !e.heldExact) continue;
    if (cFilters.status === "broken" && e.heldExact) continue;
    byCity[e.city] = (byCity[e.city] || 0) + 1;
  }
  renderSidebar(byCity);

  // stats
  const held = matching.filter(e => e.heldExact).length;
  const broken = matching.length - held;
  const holdRate = matching.length ? (100*held/matching.length).toFixed(1) : "—";
  $("stats").innerHTML = `${matching.length.toLocaleString()} events `
    + `<span style="color:#86efac">·</span> ${held.toLocaleString()} held `
    + `<span style="color:#fca5a5">·</span> ${broken.toLocaleString()} broken `
    + `<span style="color:#9ca3af">·</span> hold rate ${holdRate}% `
    + `<span style="color:#9ca3af">·</span> showing ${shown.length}`;

  // grid
  const grid = $("grid");
  if (!shown.length) {
    grid.className = "";
    grid.innerHTML = `<div class="empty">No events match the current filters.</div>`;
  } else if (VIEW === "chart") {
    grid.className = "grid";
    grid.innerHTML = shown.map(renderCard).join("");
  } else {
    grid.className = "";
    grid.innerHTML = renderList(shown);
  }
}

// ===== list view: group events by date, render as compact table =====
function renderList(events) {
  // Group by date, descending (most recent first)
  const byDate = {};
  for (const e of events) {
    (byDate[e.date] = byDate[e.date] || []).push(e);
  }
  const dates = Object.keys(byDate).sort().reverse();
  const windowH = parseInt(CTRL.window.value);

  const sections = dates.map(date => {
    // Inside each day: sort by city ASC (stable, easy to scan)
    const dayEvents = byDate[date].slice().sort((a, b) =>
      a.city.localeCompare(b.city) || a.dropEndHour - b.dropEndHour);
    const held = dayEvents.filter(e => e.heldExact).length;
    const broken = dayEvents.length - held;
    const rows = dayEvents.map(e => {
      const cls = e.heldExact ? "" : " class=\"broken\"";
      const pill = e.heldExact
        ? `<span class="mini-pill held">HELD</span>`
        : `<span class="mini-pill broken">+${e.overshoot.toFixed(1)}°</span>`;
      const window = `${String(e.dropStartHour).padStart(2,'0')}:00→`
                   + `${String(e.dropEndHour).padStart(2,'0')}:00`;
      return `<tr${cls}>
        <td class="city">${e.city}</td>
        <td class="num">${window}</td>
        <td class="num">${e.dropStartTemp.toFixed(1)}→${e.dropEndTemp.toFixed(1)}°</td>
        <td class="num">−${e.dropMagnitude.toFixed(1)}°</td>
        <td class="num">${e.preDropHigh.toFixed(1)}°</td>
        <td class="num">${e.dayHigh.toFixed(1)}°</td>
        <td class="pill-cell">${pill}</td>
      </tr>`;
    }).join("");
    return `<div class="day-group">
      <div class="day-header">
        <span>${date}</span>
        <span class="stat">${dayEvents.length} event${dayEvents.length===1?'':'s'} `
        + `<span class="h">${held} held</span> · `
        + `<span class="b">${broken} broken</span></span>
      </div>
      <table class="events">
        <thead><tr>
          <th>City</th>
          <th style="text-align:right">Window (${windowH}h)</th>
          <th style="text-align:right">Drop</th>
          <th style="text-align:right">Δ</th>
          <th style="text-align:right">Pre-drop high</th>
          <th style="text-align:right">Day high</th>
          <th style="text-align:center">Result</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  });
  return sections.join("");
}

// ===== sidebar action buttons =====
$("city-all").addEventListener("click", () => {
  SELECTED_CITIES = new Set(Object.keys(DATA)); SOLO_CITY = null; rerender();
});
$("city-none").addEventListener("click", () => {
  SELECTED_CITIES = new Set(); SOLO_CITY = null; rerender();
});

// ===== wire all controls =====
for (const k of Object.keys(CTRL)) {
  CTRL[k].addEventListener("input", rerender);
  CTRL[k].addEventListener("change", rerender);
}

// ===== view toggle =====
function setView(v) {
  VIEW = v;
  $("view-chart").classList.toggle("active", v === "chart");
  $("view-list").classList.toggle("active",  v === "list");
  rerender();
}
$("view-chart").addEventListener("click", () => setView("chart"));
$("view-list").addEventListener("click",  () => setView("list"));

// ===== boot =====
rerender();
"""


def _render_page(data: dict, meta: dict) -> str:
    data_json = json.dumps(data, separators=(",", ":"))
    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>Temperature drop events</title>")
    parts.append(f"<style>{PAGE_CSS}</style>")
    parts.append("</head><body>")
    parts.append(
        f"<header>"
        f"<div style='display:flex;align-items:center;gap:18px'>"
        f"<h1>Temperature-drop events</h1>"
        f"<div class='view-toggle'>"
        f"<button id='view-chart' class='active'>Charts</button>"
        f"<button id='view-list'>List</button>"
        f"</div></div>"
        f"<div id='stats' class='stats'></div></header>"
    )
    parts.append("<div class='controls'>")
    parts.append(_ctrl_range("threshold", "Drop threshold",
                              0.5, 10.0, 0.5, 2.0, "°C"))
    parts.append(_ctrl_range("windowH",   "Window",
                              1, 6, 1, 2, "h", val_id="windowVal"))
    parts.append(_ctrl_range("afterHour", "Drop end ≥",
                              0, 23, 1, 12, ":00", val_id="afterHourVal"))
    parts.append(_ctrl_range("magMin",    "Magnitude min",
                              0.5, 10.0, 0.5, 2.0, "°C"))
    parts.append(_ctrl_range("magMax",    "Magnitude max",
                              0.5, 20.0, 0.5, 20.0, "°C"))
    parts.append(_ctrl_range("hourMin",   "Hour min",
                              0, 23, 1, 12, ":00", val_id="hourMinVal"))
    parts.append(_ctrl_range("hourMax",   "Hour max",
                              0, 23, 1, 23, ":00", val_id="hourMaxVal"))
    parts.append("""
      <div class='ctrl'><label>Status</label>
        <select id='status'>
          <option value='all'>All</option>
          <option value='held'>Held only</option>
          <option value='broken'>Broken only</option>
        </select></div>
      <div class='ctrl'><label>Sort</label>
        <select id='sort'>
          <option value='random'>Random sample</option>
          <option value='magnitude'>Largest drops</option>
          <option value='overshoot'>Largest overshoots</option>
          <option value='hour'>By hour</option>
          <option value='city'>By city</option>
          <option value='date'>Most recent</option>
        </select></div>
    """)
    parts.append(_ctrl_range("limit",     "Max charts",
                              10, 500, 10, 60, ""))
    parts.append("</div>")  # controls

    parts.append("<div class='layout'>")
    parts.append(
        "<aside>"
        "<h3>Cities</h3>"
        "<div class='city-actions'>"
        "<button id='city-all'>All</button>"
        "<button id='city-none'>None</button>"
        "</div>"
        "<div id='city-list' class='city-list'></div>"
        "<div style='margin-top:14px; font-size:10px; color:#9ca3af; "
        "line-height:1.4'>Click a city <i>name</i> to solo it; "
        "click again to un-solo.</div>"
        "</aside>"
    )
    parts.append("<main><div id='grid' class='grid'></div></main>")
    parts.append("</div>")  # layout

    parts.append(f"<script>const DATA = {data_json};</script>")
    parts.append(f"<script>{PAGE_JS}</script>")
    parts.append("</body></html>")
    return "".join(parts)


def _ctrl_range(id_: str, label: str, lo, hi, step, default, unit,
                 val_id: str | None = None) -> str:
    val_id = val_id or (id_ + "Val")
    return (
        f"<div class='ctrl'><label>{label}</label>"
        f"<div class='row'>"
        f"<input type='range' id='{id_}' min='{lo}' max='{hi}' "
        f"step='{step}' value='{default}'>"
        f"<span class='val' id='{val_id}'></span></div></div>"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=730,
                   help="Look back N days from today (default: 730)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all)")
    p.add_argument("--db", default=DEFAULT_DB_PATH,
                   help=f"Weather archive SQLite (default: {DEFAULT_DB_PATH})")
    p.add_argument("--output", default=os.path.join(_BOT_DIR, "data",
                                                     "temp_drop_charts.html"),
                   help="Output HTML path (default: data/temp_drop_charts.html)")
    args = p.parse_args()

    end_d = (datetime.strptime(args.end, "%Y-%m-%d").date()
             if args.end else date.today())
    if args.start:
        start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_d = end_d - timedelta(days=args.days)
    start = start_d.isoformat()
    end   = end_d.isoformat()

    cities_meta = load_city_list(args.city)
    if not cities_meta:
        print("No cities found.")
        return 1
    city_names = [c["city"] for c in cities_meta]
    print(f"Loading hourly data for {len(city_names)} cities, "
          f"{start} → {end} …")

    data = _load_hourly(args.db, city_names, start, end)
    total_days = sum(len(v) for v in data.values())
    print(f"Loaded {total_days:,} city-days "
          f"({sum(len(v)*24 for v in data.values()):,} hourly points)")

    meta = {"start": start, "end": end}
    html = _render_page(data, meta)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"Wrote {size_kb:,.0f} KB to {args.output}")
    print(f"Open it: file:///{os.path.abspath(args.output).replace(os.sep, '/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())