"""
neighbor_leadlag_by_wind.py — Wind-stratified lead-lag analysis.

For each (settlement, neighbor) pair, the basic lead-lag aggregate is
WIND-BLIND.  But a neighbor that's geographically W of the settlement
station mainly leads when the wind is coming FROM the W.  Stratifying
by wind direction sharpens the signal dramatically.

For each (settlement, neighbor, wind_octant) triple:
  - count days where the settlement's afternoon mean wind fell in that octant
  - compute peak-time lead-lag distribution within those days
  - report median lead, % lead+tie, correlation r

Inputs:
  data/station_obs.db    — settlement (with wind_dir_deg)
  data/neighbor_obs.db   — neighbors (with wind_dir_deg)

Output:
  Console: top wind-conditioned leaders per city
  HTML at data/neighbor_leadlag_wind.html — filterable by city + wind
  Optional CSV with per-triple stats

Usage:
    cd bot
    python -m scripts.neighbor_leadlag_by_wind
    python -m scripts.neighbor_leadlag_by_wind --city Dallas
    python -m scripts.neighbor_leadlag_by_wind --min-days 5
    python -m scripts.neighbor_leadlag_by_wind --csv data/leadlag_wind.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS  # type: ignore

STATION_DB  = os.path.join(_BOT_DIR, "data", "station_obs.db")
NEIGHBOR_DB = os.path.join(_BOT_DIR, "data", "neighbor_obs.db")
DEFAULT_OUT = os.path.join(_BOT_DIR, "data", "neighbor_leadlag_wind.html")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("leadlag_wind")

# Afternoon hours that drive temperature trajectory.  Mean wind over these
# hours = the "regime" that produced today's peak.
AFTERNOON_HOURS = list(range(11, 19))   # 11:00 → 18:59 local


# ---------------------------------------------------------------------------
# Wind helpers
# ---------------------------------------------------------------------------

CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def deg_to_cardinal(deg: float) -> str:
    return CARDINALS[int((deg + 22.5) / 45) % 8]


def vector_mean_dir(degrees: list[float]) -> float | None:
    """Mean of circular (wind direction) data via vector averaging.
    Returns None if no inputs."""
    if not degrees:
        return None
    u = sum(math.sin(math.radians(d)) for d in degrees) / len(degrees)
    v = sum(math.cos(math.radians(d)) for d in degrees) / len(degrees)
    if u == 0 and v == 0:
        return None
    return (math.degrees(math.atan2(u, v)) + 360) % 360


def pearson_r(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx2 = sum((xi - mx) ** 2 for xi in x)
    sy2 = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(sx2 * sy2)
    return sxy / denom if denom > 0 else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_settlement(db: str, city: str) -> dict[str, dict]:
    """Returns {date: {hourly: {h: (temp_c, wind_dir_deg)},
                       daily:  (tmax, tmax_hour),
                       wind_octant: 'SW'|...|None}}"""
    out: dict[str, dict] = defaultdict(lambda: {"hourly": {}, "daily": None,
                                                  "wind_octant": None})
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT date_local, hour_local, temp_c, wind_dir_deg "
            "FROM station_obs WHERE city = ? AND temp_c IS NOT NULL "
            "ORDER BY date_local, hour_local", (city,)
        ):
            out[r["date_local"]]["hourly"][r["hour_local"]] = (
                r["temp_c"], r["wind_dir_deg"]
            )
        for r in conn.execute(
            "SELECT date_local, tmax_c, tmax_hour_local FROM station_daily_max "
            "WHERE city = ?", (city,)
        ):
            out[r["date_local"]]["daily"] = (r["tmax_c"], r["tmax_hour_local"])

    # Compute afternoon mean wind octant per day
    for d, data in out.items():
        winds = [v[1] for h, v in data["hourly"].items()
                  if h in AFTERNOON_HOURS and v[1] is not None]
        m = vector_mean_dir(winds)
        data["wind_octant"] = deg_to_cardinal(m) if m is not None else None
    return out


def load_neighbors(db: str, city: str) -> dict[str, dict]:
    """{sid: {meta: {...}, daily: {date: (tmax, hr)}, hourly: {date: {h: temp_c}}}}"""
    out: dict[str, dict] = {}
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            """
            SELECT sid, name, network, lat, lon, distance_mi,
                   bearing_deg, direction
            FROM neighbor_meta WHERE polymarket_city = ?
            ORDER BY distance_mi
            """, (city,)
        ):
            out[r["sid"]] = {
                "meta":  dict(r),
                "daily":  {},
                "hourly": defaultdict(dict),
            }
        for sid, data in out.items():
            for r in conn.execute(
                "SELECT date_local, tmax_c, tmax_hour_local "
                "FROM neighbor_daily_max WHERE sid = ?", (sid,)
            ):
                data["daily"][r["date_local"]] = (r["tmax_c"], r["tmax_hour_local"])
            for r in conn.execute(
                "SELECT date_local, hour_local, temp_c FROM neighbor_obs "
                "WHERE sid = ? AND temp_c IS NOT NULL "
                "ORDER BY date_local, hour_local", (sid,)
            ):
                data["hourly"][r["date_local"]][r["hour_local"]] = r["temp_c"]
    return out


# ---------------------------------------------------------------------------
# Per (settlement, neighbor, wind_octant) analysis
# ---------------------------------------------------------------------------

def analyze(city: str, neighbor_sid: str, neighbor_meta: dict,
             settlement: dict, neighbor: dict, min_days: int) -> list[dict]:
    """Return one dict per wind octant where at least min_days are available."""
    # For each date both stations have a peak, group by settlement wind octant
    by_octant: dict[str | None, list[dict]] = defaultdict(list)
    for d, s_data in settlement.items():
        if s_data["daily"] is None or s_data["wind_octant"] is None:
            continue
        n_daily = neighbor["daily"].get(d)
        if n_daily is None or n_daily[1] is None:
            continue
        s_hr = s_data["daily"][1]
        n_hr = n_daily[1]
        if s_hr is None:
            continue
        delta = s_hr - n_hr
        by_octant[s_data["wind_octant"]].append({
            "date":           d,
            "delta_hours":    delta,
            "settlement_max": round(s_data["daily"][0], 2),
            "settlement_hr":  s_hr,
            "neighbor_max":   round(n_daily[0], 2),
            "neighbor_hr":    n_hr,
        })

    rows: list[dict] = []
    s = CITY_STATIONS[city]
    for octant, days in by_octant.items():
        if len(days) < min_days:
            continue
        deltas = [d["delta_hours"] for d in days]
        n_led   = sum(1 for x in deltas if x > 0)
        n_tied  = sum(1 for x in deltas if x == 0)
        n_lag   = sum(1 for x in deltas if x < 0)
        rows.append({
            "city":             city,
            "settlement":       s[0],
            "neighbor_sid":     neighbor_sid,
            "neighbor_name":    neighbor_meta.get("name"),
            "neighbor_dir":     neighbor_meta.get("direction"),
            "distance_mi":      neighbor_meta.get("distance_mi"),
            "wind_from":        octant,    # afternoon mean wind octant
            "n_days":           len(days),
            "n_neighbor_led":   n_led,
            "n_tied":           n_tied,
            "n_settl_led":      n_lag,
            "pct_neighbor_led_or_tied": round(100 * (n_led + n_tied) / len(days), 1),
            "median_lead_h":    statistics.median(deltas),
            "mean_lead_h":      round(statistics.mean(deltas), 2),
            "per_day":          sorted(days, key=lambda x: x["date"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_city_report(city: str, rows: list[dict], min_days: int) -> None:
    if not rows:
        print(f"\n  {city}: no (neighbor, wind) triples reached {min_days}-day threshold")
        return
    s = CITY_STATIONS[city]
    print()
    print("=" * 102)
    print(f"  {city.upper()}  —  {s[0]}  —  {len(rows)} (neighbor, wind) triples")
    print("=" * 102)
    print(f"  {'neighbor':<6} {'geo':>3}  {'dist':>5} "
          f"{'wind from':>9}  {'days':>4} {'led':>4} {'tied':>4} "
          f"{'lag':>3} {'%lead+tie':>9} {'med Δh':>7} {'mean Δh':>8}")
    print("  " + "-" * 98)
    # Sort by best leader first
    rows.sort(key=lambda r: (
        -r["pct_neighbor_led_or_tied"],
        -r["median_lead_h"],
    ))
    for r in rows:
        sign = "+" if r["median_lead_h"] >= 0 else ""
        sign_m = "+" if r["mean_lead_h"] >= 0 else ""
        print(f"  {r['neighbor_sid']:<6} {r['neighbor_dir']:>3}  "
              f"{r['distance_mi']:>4.1f}m  "
              f"{r['wind_from']:>9}  {r['n_days']:>4d} "
              f"{r['n_neighbor_led']:>4d} {r['n_tied']:>4d} "
              f"{r['n_settl_led']:>3d} {r['pct_neighbor_led_or_tied']:>8.1f}% "
              f"{sign}{r['median_lead_h']:>5.1f}h "
              f"{sign_m}{r['mean_lead_h']:>6.2f}h")


def render_dashboard(all_rows: list[dict], output_path: str) -> None:
    data_json = json.dumps(all_rows, default=str, separators=(",", ":"))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Wind-stratified lead-lag</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #f3f4f6; color: #111827; font-size: 13px; }}
header {{ background: #111827; color: #f3f4f6; padding: 12px 20px;
         display: flex; justify-content: space-between; align-items: center; }}
header h1 {{ margin: 0; font-size: 17px; }}
header .meta {{ font-family: monospace; font-size: 11px; color: #9ca3af; }}
.legend {{ background: white; padding: 10px 18px; font-size: 12px;
           color: #4b5563; border-bottom: 1px solid #e5e7eb; }}
.legend code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
.filters {{ background: white; padding: 10px 18px; display: flex; gap: 18px;
            flex-wrap: wrap; align-items: center; font-size: 12px;
            border-bottom: 1px solid #e5e7eb; }}
.filters label {{ font-weight: 600; color: #374151; margin-right: 4px; }}
.filters select, .filters input {{ padding: 3px 6px; font-size: 12px; }}
.filters .count {{ color: #6b7280; font-family: monospace; margin-left: auto; }}
table {{ width: calc(100% - 36px); margin: 14px 18px; background: white;
         border-collapse: collapse; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
         border-radius: 6px; overflow: hidden; font-size: 12px; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #f3f4f6; }}
th {{ background: #f9fafb; color: #6b7280; font-weight: 600; font-size: 10px;
      text-transform: uppercase; letter-spacing: 0.3px; cursor: pointer;
      user-select: none; }}
th:hover {{ background: #eef2ff; }}
th.sorted-asc::after {{ content: ' ▲'; color: #4338ca; }}
th.sorted-desc::after {{ content: ' ▼'; color: #4338ca; }}
td.num {{ font-family: monospace; text-align: right; }}
td .pill {{ display: inline-block; padding: 1px 6px; border-radius: 10px;
            font-size: 10px; font-weight: 600; }}
.wind {{ display: inline-block; background: #ddd6fe; color: #4338ca;
         padding: 2px 8px; border-radius: 10px; font-weight: 600; }}
tr.leader {{ background: #f0fdf4; }}
tr.leader td .pill {{ background: #dcfce7; color: #166534; }}
tr.lagger {{ background: #fef2f2; }}
tr.lagger td .pill {{ background: #fee2e2; color: #991b1b; }}
tr.neutral td .pill {{ background: #e5e7eb; color: #374151; }}
tr.aligned {{ box-shadow: inset 4px 0 0 #4338ca; }}
tr.detail {{ background: #fafbff; }}
tr.detail td {{ padding: 4px 14px; font-family: monospace; font-size: 11px;
                color: #4b5563; }}
</style></head><body>

<header>
  <h1>Wind-stratified lead-lag</h1>
  <div class='meta'>{len(all_rows)} (settlement, neighbor, wind) triples</div>
</header>

<div class='legend'>
  <b>How to read:</b>
  Each row is one combination of (Polymarket settlement station, nearby
  neighbor station, afternoon mean wind direction).
  <code>geo dir</code> = neighbor's static bearing from settlement.
  <code>wind from</code> = afternoon mean wind octant at the settlement station.
  <code>med Δh</code> = median (settlement_peak_hour − neighbor_peak_hour);
  positive = neighbor peaks earlier (LEADS).
  Rows with a <span style='color:#4338ca'>blue left border</span> are pairs
  where neighbor is UPWIND (wind blows FROM neighbor's direction → settlement)
  — these are the physically-expected leaders.
</div>

<div class='filters'>
  <div><label>City</label>
    <select id='f-city'><option value=''>All</option></select></div>
  <div><label>Wind from</label>
    <select id='f-wind'>
      <option value=''>All</option>
      <option>N</option><option>NE</option><option>E</option><option>SE</option>
      <option>S</option><option>SW</option><option>W</option><option>NW</option>
    </select></div>
  <div><label>Neighbor dir</label>
    <select id='f-dir'>
      <option value=''>All</option>
      <option>N</option><option>NE</option><option>E</option><option>SE</option>
      <option>S</option><option>SW</option><option>W</option><option>NW</option>
    </select></div>
  <div><label>Min days</label>
    <input id='f-min-days' type='number' value='5' min='1' style='width:60px'></div>
  <div><label>Min %lead+tie</label>
    <input id='f-min-pct' type='number' value='60' min='0' max='100' style='width:60px'></div>
  <div><label><input id='f-aligned' type='checkbox'> Only wind-aligned (upwind)</label></div>
  <div class='count' id='count'>—</div>
</div>

<table id='triples'>
  <thead><tr>
    <th data-key='city'>City</th>
    <th data-key='neighbor_sid'>Neighbor</th>
    <th data-key='neighbor_dir'>Geo Dir</th>
    <th data-key='distance_mi'>Dist (mi)</th>
    <th data-key='wind_from'>Wind From</th>
    <th data-key='n_days'>Days</th>
    <th data-key='pct_neighbor_led_or_tied'>%Lead+Tie</th>
    <th data-key='median_lead_h'>Med Δh</th>
    <th data-key='mean_lead_h'>Mean Δh</th>
  </tr></thead>
  <tbody id='tbody'></tbody>
</table>

<script>
const ROWS = {data_json};
const $ = id => document.getElementById(id);

// Wind is "FROM" an octant.  Upwind = wind originates from the neighbor's
// geographic direction relative to settlement (so air flows neighbor → settlement).
function isAligned(r) {{ return r.wind_from === r.neighbor_dir; }}

function classify(r) {{
  if (r.pct_neighbor_led_or_tied >= 70) return 'leader';
  if (r.pct_neighbor_led_or_tied <= 30) return 'lagger';
  return 'neutral';
}}

const cities = [...new Set(ROWS.map(r => r.city))].sort();
for (const c of cities) {{
  const o = document.createElement('option');
  o.value = o.textContent = c;
  $('f-city').appendChild(o);
}}

let SORT_KEY = 'pct_neighbor_led_or_tied', SORT_DIR = -1;
let EXPANDED = new Set();

function row(r) {{
  const cls = classify(r) + (isAligned(r) ? ' aligned' : '');
  const med = (r.median_lead_h >= 0 ? '+' : '') + r.median_lead_h.toFixed(1);
  const mean = (r.mean_lead_h >= 0 ? '+' : '') + r.mean_lead_h.toFixed(2);
  const k = `${{r.city}}|${{r.neighbor_sid}}|${{r.wind_from}}`;
  return `<tr class='${{cls}}' data-key='${{k}}'>
    <td><b>${{r.city}}</b><br><span style='color:#6b7280;font-size:10px'>vs ${{r.settlement}}</span></td>
    <td>${{r.neighbor_sid}}<br><span style='color:#6b7280;font-size:10px'>${{(r.neighbor_name||'').slice(0,28)}}</span></td>
    <td><span class='pill'>${{r.neighbor_dir}}</span></td>
    <td class='num'>${{r.distance_mi.toFixed(1)}}</td>
    <td><span class='wind'>${{r.wind_from}}</span></td>
    <td class='num'>${{r.n_days}}</td>
    <td class='num'>${{r.pct_neighbor_led_or_tied.toFixed(0)}}%</td>
    <td class='num'>${{med}}h</td>
    <td class='num'>${{mean}}h</td>
  </tr>`;
}}

function detailRow(r) {{
  const days = r.per_day || [];
  const cells = days.map(d => {{
    const c = d.delta_hours > 0 ? '#16a34a' : d.delta_hours < 0 ? '#dc2626' : '#6b7280';
    const sign = d.delta_hours > 0 ? '+' : '';
    return `<span style='color:${{c}};margin-right:8px' title='${{d.date}}: settlement ${{d.settlement_max}}°@${{d.settlement_hr}} vs neighbor ${{d.neighbor_max}}°@${{d.neighbor_hr}}'>`
      + `${{d.date.slice(5)}}:${{sign}}${{d.delta_hours}}h</span>`;
  }}).join('');
  return `<tr class='detail'><td colspan='9'>per-day: ${{cells}}</td></tr>`;
}}

function render() {{
  const city = $('f-city').value;
  const wf = $('f-wind').value;
  const dir = $('f-dir').value;
  const minD = parseInt($('f-min-days').value) || 0;
  const minP = parseFloat($('f-min-pct').value) || 0;
  const onlyAligned = $('f-aligned').checked;
  let rows = ROWS.filter(r =>
    (!city || r.city === city)
    && (!wf || r.wind_from === wf)
    && (!dir || r.neighbor_dir === dir)
    && r.n_days >= minD
    && r.pct_neighbor_led_or_tied >= minP
    && (!onlyAligned || isAligned(r))
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (av == null) av = -Infinity;
    if (bv == null) bv = -Infinity;
    if (typeof av === 'number') return SORT_DIR * (av - bv);
    return SORT_DIR * String(av).localeCompare(String(bv));
  }});
  $('count').textContent = rows.length + ' / ' + ROWS.length;
  const html = [];
  for (const r of rows) {{
    html.push(row(r));
    const k = `${{r.city}}|${{r.neighbor_sid}}|${{r.wind_from}}`;
    if (EXPANDED.has(k)) html.push(detailRow(r));
  }}
  $('tbody').innerHTML = html.join('');
  document.querySelectorAll('th').forEach(th => {{
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? 'sorted-asc' : 'sorted-desc');
  }});
  document.querySelectorAll('tr[data-key]').forEach(tr => {{
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => {{
      const k = tr.dataset.key;
      if (EXPANDED.has(k)) EXPANDED.delete(k);
      else EXPANDED.add(k);
      render();
    }});
  }});
}}

document.querySelectorAll('th').forEach(th => {{
  th.addEventListener('click', () => {{
    if (!th.dataset.key) return;
    if (SORT_KEY === th.dataset.key) SORT_DIR = -SORT_DIR;
    else {{ SORT_KEY = th.dataset.key; SORT_DIR = 1; }}
    render();
  }});
}});
['f-city','f-wind','f-dir','f-min-days','f-min-pct','f-aligned'].forEach(id =>
  $(id).addEventListener('input', render));
render();
</script>
</body></html>"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info(f"Wrote {os.path.getsize(output_path)/1024:.0f} KB dashboard to {output_path}")


def write_csv(rows: list[dict], path: str) -> None:
    fields = ["city", "settlement", "neighbor_sid", "neighbor_name",
              "neighbor_dir", "distance_mi", "wind_from", "n_days",
              "n_neighbor_led", "n_tied", "n_settl_led",
              "pct_neighbor_led_or_tied", "median_lead_h", "mean_lead_h"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    log.info(f"Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all)")
    p.add_argument("--min-days", type=int, default=5,
                   help="Min days per (neighbor, wind octant) triple (default: 5)")
    p.add_argument("--station-db", default=STATION_DB,
                   help=f"Settlement DB (default: {STATION_DB})")
    p.add_argument("--neighbor-db", default=NEIGHBOR_DB,
                   help=f"Neighbor DB (default: {NEIGHBOR_DB})")
    p.add_argument("--csv", help="Per-triple stats CSV")
    p.add_argument("--html", default=DEFAULT_OUT,
                   help=f"Dashboard HTML (default: {DEFAULT_OUT})")
    p.add_argument("--no-html", action="store_true")
    args = p.parse_args()

    with sqlite3.connect(args.neighbor_db) as conn:
        avail = [r[0] for r in conn.execute(
            "SELECT DISTINCT polymarket_city FROM neighbor_meta ORDER BY 1"
        ).fetchall()]
    if not avail:
        log.error("No neighbor data. Run scripts.neighbor_obs_pull first.")
        return 1

    cities = args.city or avail
    cities = [c for c in cities if c in avail]
    if not cities:
        log.error(f"No matching cities. Available: {avail}")
        return 1

    # Sanity: confirm we have wind_dir_deg in the station_obs schema
    with sqlite3.connect(args.station_db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(station_obs)").fetchall()]
    if "wind_dir_deg" not in cols:
        log.error("station_obs is missing wind_dir_deg — re-pull with the updated "
                   "scripts.station_obs_pull (script will ALTER the table on next run).")
        return 1

    log.info(f"Analyzing {len(cities)} cities; min {args.min_days} days/octant")
    all_rows: list[dict] = []
    for city in cities:
        s = load_settlement(args.station_db, city)
        if not s:
            log.warning(f"  {city}: no settlement data")
            continue
        # Sanity check that the settlement actually has wind data
        n_with_wind = sum(1 for d in s.values() if d["wind_octant"] is not None)
        if n_with_wind == 0:
            log.warning(f"  {city}: 0 days have afternoon wind data — re-pull station_obs?")
            continue
        nbrs = load_neighbors(args.neighbor_db, city)
        city_rows: list[dict] = []
        for sid, data in nbrs.items():
            rows = analyze(city, sid, data["meta"], s, data, args.min_days)
            city_rows.extend(rows)
            all_rows.extend(rows)
        print_city_report(city, city_rows, args.min_days)

    print()
    print("=" * 102)
    aligned = [r for r in all_rows if r["wind_from"] == r["neighbor_dir"]]
    print(f"  GRAND TOTAL: {len(all_rows)} triples analyzed")
    print(f"  Of which wind-aligned (upwind): {len(aligned)}")
    leaders = [r for r in aligned if r["pct_neighbor_led_or_tied"] >= 70]
    print(f"  Strong upwind leaders (≥70% lead+tie when aligned): {len(leaders)}")
    print("=" * 102)

    if not args.no_html and all_rows:
        render_dashboard(all_rows, args.html)
    if args.csv and all_rows:
        write_csv(all_rows, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())