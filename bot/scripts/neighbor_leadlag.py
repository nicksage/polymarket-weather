"""
neighbor_leadlag.py — Per (settlement, neighbor) pair, measure whether
the neighbor's temperature curve LEADS the settlement station, and by
how much.

Two analyses per pair:

  1. PEAK-TIME LEAD (simple, most actionable)
     For each day where both stations have data, compute
        delta_hours = settlement_peak_hour − neighbor_peak_hour
     positive  → neighbor peaked earlier (LEADS)
     zero      → tied
     negative  → settlement peaked earlier (settlement leads)
     Aggregate across all days: median lead, % days leading, distribution.

  2. CURVE CROSS-CORRELATION AT LAG (more sensitive)
     For each day, shift the neighbor's full 24h curve by k hours
     (k = -3..+3) and compute Pearson r vs the settlement curve.
     The k that maximizes r is that day's "best lag".  Aggregate.

Inputs:
  - data/station_obs.db   (settlement stations from station_obs_pull.py)
  - data/neighbor_obs.db  (neighbors from neighbor_obs_pull.py)

Output:
  - Console summary: ranked leaderboard per city
  - HTML dashboard at data/neighbor_leadlag.html
  - Optional CSV with the per-pair stats

Usage:
    cd bot
    python -m scripts.neighbor_leadlag                  # all cities, ASOS data
    python -m scripts.neighbor_leadlag --city Dallas
    python -m scripts.neighbor_leadlag --min-days 10    # require >=10 valid days
    python -m scripts.neighbor_leadlag --csv data/leadlag.csv
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
DEFAULT_OUT = os.path.join(_BOT_DIR, "data", "neighbor_leadlag.html")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("leadlag")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_settlement_data(db: str, city: str
                          ) -> tuple[dict[str, tuple[float, int]],
                                       dict[str, dict[int, float]]]:
    """Returns (daily_max_by_date, hourly_by_date) for the settlement station."""
    daily: dict[str, tuple[float, int]] = {}
    hourly: dict[str, dict[int, float]] = defaultdict(dict)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT date_local, tmax_c, tmax_hour_local FROM station_daily_max "
            "WHERE city = ? ORDER BY date_local", (city,)
        ):
            daily[r["date_local"]] = (r["tmax_c"], r["tmax_hour_local"])
        for r in conn.execute(
            "SELECT date_local, hour_local, temp_c FROM station_obs "
            "WHERE city = ? AND temp_c IS NOT NULL "
            "ORDER BY date_local, hour_local", (city,)
        ):
            hourly[r["date_local"]][r["hour_local"]] = r["temp_c"]
    return daily, hourly


def load_neighbor_data(db: str, city: str
                        ) -> dict[str, dict]:
    """Returns {sid: {meta: {...}, daily: {...}, hourly: {...}}}"""
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
                "daily":  {},   # date -> (tmax_c, tmax_hour_local)
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
# Statistics helpers
# ---------------------------------------------------------------------------

def pearson_r(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation coefficient.  Returns None if undefined."""
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


def best_lag(settlement: dict[int, float], neighbor: dict[int, float],
              lags: list[int] = (-3, -2, -1, 0, 1, 2, 3)) -> tuple[int, float] | None:
    """For each lag k (hours), compute Pearson r of settlement_t vs
    neighbor_{t-k}.  Positive k = neighbor LEADS by k hours (its earlier
    value matches settlement's current value).  Returns (best_k, r)."""
    best: tuple[int, float] | None = None
    for k in lags:
        # Pair settlement_h with neighbor_{h-k} where both exist
        pairs = [(settlement[h], neighbor[h - k])
                  for h in settlement
                  if (h - k) in neighbor]
        if len(pairs) < 6:
            continue
        sx = [p[0] for p in pairs]
        sy = [p[1] for p in pairs]
        r = pearson_r(sx, sy)
        if r is None:
            continue
        if best is None or r > best[1]:
            best = (k, r)
    return best


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def analyze_pair(city: str, neighbor_sid: str, neighbor_meta: dict,
                  settlement_daily: dict, settlement_hourly: dict,
                  neighbor_daily: dict, neighbor_hourly: dict,
                  min_days: int) -> dict | None:
    """Compute lead-lag stats for one (settlement, neighbor) pair."""
    # Days where both have a daily max recorded
    common_dates = sorted(set(settlement_daily) & set(neighbor_daily))
    deltas: list[int] = []
    per_day: list[dict] = []
    for d in common_dates:
        s_tmax, s_hr = settlement_daily[d]
        n_tmax, n_hr = neighbor_daily[d]
        if s_hr is None or n_hr is None:
            continue
        delta = s_hr - n_hr   # positive = neighbor led
        deltas.append(delta)
        per_day.append({
            "date":           d,
            "settlement_max": round(s_tmax, 2),
            "settlement_hr":  s_hr,
            "neighbor_max":   round(n_tmax, 2),
            "neighbor_hr":    n_hr,
            "delta_hours":    delta,
        })

    if len(deltas) < min_days:
        return None

    # Lead-time aggregation
    n_total       = len(deltas)
    n_led         = sum(1 for d in deltas if d > 0)
    n_tied        = sum(1 for d in deltas if d == 0)
    n_settl_led   = sum(1 for d in deltas if d < 0)
    median_lead   = statistics.median(deltas)
    mean_lead     = round(statistics.mean(deltas), 2)
    dist          = Counter(deltas)

    # Cross-correlation at best lag, averaged across days where both have
    # near-complete hourly data
    lag_votes: list[int] = []
    lag_rs:    list[float] = []
    for d in common_dates:
        sh = settlement_hourly.get(d, {})
        nh = neighbor_hourly.get(d, {})
        if len(sh) < 18 or len(nh) < 18:
            continue
        result = best_lag(sh, nh)
        if result is None:
            continue
        k, r = result
        lag_votes.append(k)
        lag_rs.append(r)
    if lag_votes:
        median_xcorr_lag = statistics.median(lag_votes)
        mean_r_at_best   = round(statistics.mean(lag_rs), 4)
    else:
        median_xcorr_lag = None
        mean_r_at_best   = None

    return {
        "city":             city,
        "settlement":       CITY_STATIONS[city][0],
        "neighbor_sid":     neighbor_sid,
        "neighbor_name":    neighbor_meta.get("name"),
        "network":          neighbor_meta.get("network"),
        "distance_mi":      neighbor_meta.get("distance_mi"),
        "bearing_deg":      neighbor_meta.get("bearing_deg"),
        "direction":        neighbor_meta.get("direction"),
        "n_days":           n_total,
        "n_neighbor_led":   n_led,
        "n_tied":           n_tied,
        "n_settl_led":      n_settl_led,
        "pct_neighbor_led_or_tied": round(100 * (n_led + n_tied) / n_total, 1),
        "pct_neighbor_strictly_led": round(100 * n_led / n_total, 1),
        "median_lead_h":    median_lead,
        "mean_lead_h":      mean_lead,
        "lead_distribution": dict(sorted(dist.items())),
        "median_xcorr_lag": median_xcorr_lag,
        "mean_r_at_best_lag": mean_r_at_best,
        "per_day":          per_day,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_city_report(city: str, results: list[dict]) -> None:
    if not results:
        print(f"\n  {city}: no neighbor data (run scripts.neighbor_obs_pull)")
        return
    s = CITY_STATIONS[city]
    print()
    print("=" * 96)
    print(f"  {city.upper()}  —  settlement: {s[0]} "
          f"({s[3]:.3f}, {s[4]:.3f})  —  {len(results)} neighbors with ≥1 day")
    print("=" * 96)
    print(f"  {'neighbor':<6} {'dir':>3} {'dist':>5} "
          f"{'days':>5} {'led':>5} {'tied':>5} {'lagged':>6} "
          f"{'%lead+tie':>10} {'med Δh':>7} {'mean Δh':>8} "
          f"{'xcorr lag':>10} {'r':>7}")
    print("  " + "-" * 92)
    # Sort by best leader first (highest %led+tied, then median lead)
    rows = sorted(results, key=lambda r: (
        -r["pct_neighbor_led_or_tied"],
        -r["median_lead_h"],
    ))
    for r in rows:
        xc = (f"{r['median_xcorr_lag']:>+3}h" if r["median_xcorr_lag"] is not None
              else "  -  ")
        rr = (f"{r['mean_r_at_best_lag']:.3f}" if r["mean_r_at_best_lag"] is not None
              else "  -  ")
        sign = "+" if r["median_lead_h"] >= 0 else ""
        sign_m = "+" if r["mean_lead_h"] >= 0 else ""
        print(f"  {r['neighbor_sid']:<6} {r['direction']:>3} "
              f"{r['distance_mi']:>4.1f}m "
              f"{r['n_days']:>5d} {r['n_neighbor_led']:>5d} "
              f"{r['n_tied']:>5d} {r['n_settl_led']:>6d} "
              f"{r['pct_neighbor_led_or_tied']:>9.1f}% "
              f"{sign}{r['median_lead_h']:>5.1f}h "
              f"{sign_m}{r['mean_lead_h']:>6.2f}h "
              f"{xc:>10} {rr:>7}")
    print()
    print("  Reading: 'med Δh' = median (settlement_peak_hour − neighbor_peak_hour).")
    print("           Positive = neighbor peaks first (LEADS).  'xcorr lag' = the")
    print("           hour shift that maximizes correlation between the curves.")


def render_dashboard(all_results: list[dict], output_path: str) -> None:
    """Self-contained HTML with sortable table + per-pair drill-down."""
    data_json = json.dumps(all_results, default=str, separators=(",", ":"))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Neighbor lead-lag analysis</title>
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
tr.leader {{ background: #f0fdf4; }}
tr.leader td .pill {{ background: #dcfce7; color: #166534; }}
tr.lagger {{ background: #fef2f2; }}
tr.lagger td .pill {{ background: #fee2e2; color: #991b1b; }}
tr.neutral td .pill {{ background: #e5e7eb; color: #374151; }}
tr.expanded {{ background: #eef2ff; }}
tr.detail {{ background: #fafbff; }}
tr.detail td {{ padding: 4px 14px; font-family: monospace; font-size: 11px;
                color: #4b5563; }}
.bearing-dot {{ display: inline-block; width: 14px; height: 14px;
                 border-radius: 50%; background: #4338ca; vertical-align: middle;
                 margin-right: 4px; position: relative; }}
.dist-dist {{ font-family: monospace; font-size: 11px; color: #6b7280; }}
</style></head><body>

<header>
  <h1>Neighbor lead-lag analysis</h1>
  <div class='meta'>{len(all_results)} pairs analyzed</div>
</header>

<div class='legend'>
  <b>How to read:</b>
  <code>med Δh</code> = median (settlement_peak_hour − neighbor_peak_hour).
  Positive = neighbor peaks EARLIER (lead).
  <code>%lead+tie</code> = fraction of days the neighbor peaked at or before the settlement station.
  <code>xcorr lag</code> = hour shift maximizing Pearson r between curves;
  +1h means shifting the neighbor 1h LATER matches the settlement curve best (so neighbor leads by 1h).
  <code>r</code> at best lag indicates how closely the curves track once aligned.
</div>

<div class='filters'>
  <div><label>City</label>
    <select id='f-city'><option value=''>All</option></select></div>
  <div><label>Direction</label>
    <select id='f-dir'>
      <option value=''>All</option>
      <option>N</option><option>NE</option><option>E</option><option>SE</option>
      <option>S</option><option>SW</option><option>W</option><option>NW</option>
    </select></div>
  <div><label>Min days</label>
    <input id='f-min-days' type='number' value='5' min='1' style='width:60px'></div>
  <div><label>Min %lead+tie</label>
    <input id='f-min-pct' type='number' value='50' min='0' max='100' style='width:60px'></div>
  <div class='count' id='count'>—</div>
</div>

<table id='pairs'>
  <thead><tr>
    <th data-key='city'>City</th>
    <th data-key='neighbor_sid'>Neighbor</th>
    <th data-key='direction'>Dir</th>
    <th data-key='distance_mi'>Dist (mi)</th>
    <th data-key='n_days'>Days</th>
    <th data-key='pct_neighbor_led_or_tied'>%Lead+Tie</th>
    <th data-key='median_lead_h'>Med Δh</th>
    <th data-key='mean_lead_h'>Mean Δh</th>
    <th data-key='median_xcorr_lag'>xcorr lag</th>
    <th data-key='mean_r_at_best_lag'>r</th>
    <th>Distribution</th>
  </tr></thead>
  <tbody id='tbody'></tbody>
</table>

<script>
const PAIRS = {data_json};
const $ = id => document.getElementById(id);

// Populate city dropdown
const cities = [...new Set(PAIRS.map(p => p.city))].sort();
for (const c of cities) {{
  const o = document.createElement('option');
  o.value = o.textContent = c;
  $('f-city').appendChild(o);
}}

let SORT_KEY = 'pct_neighbor_led_or_tied', SORT_DIR = -1;
let EXPANDED = new Set();

function classify(p) {{
  if (p.pct_neighbor_led_or_tied >= 70) return 'leader';
  if (p.pct_neighbor_led_or_tied <= 30) return 'lagger';
  return 'neutral';
}}

function distSparkline(p) {{
  const dist = p.lead_distribution || {{}};
  const keys = Object.keys(dist).map(Number).sort((a,b) => a-b);
  if (!keys.length) return '';
  return keys.map(k => {{
    const sign = k > 0 ? '+' : (k < 0 ? '−' : '');
    return `<span title='${{k}}h: ${{dist[k]}} days'>${{sign}}${{Math.abs(k)}}h:${{dist[k]}}</span>`;
  }}).join(' ');
}}

function row(p) {{
  const cls = classify(p);
  const xc = p.median_xcorr_lag !== null
    ? (p.median_xcorr_lag >= 0 ? '+' : '') + p.median_xcorr_lag + 'h' : '—';
  const rr = p.mean_r_at_best_lag !== null
    ? p.mean_r_at_best_lag.toFixed(3) : '—';
  const med = (p.median_lead_h >= 0 ? '+' : '') + p.median_lead_h.toFixed(1);
  const mean = (p.mean_lead_h >= 0 ? '+' : '') + p.mean_lead_h.toFixed(2);
  return `<tr class='${{cls}}' data-key='${{p.city}}|${{p.neighbor_sid}}'>
    <td><b>${{p.city}}</b><br><span style='color:#6b7280;font-size:10px'>vs ${{p.settlement}}</span></td>
    <td>${{p.neighbor_sid}}<br><span style='color:#6b7280;font-size:10px'>${{(p.neighbor_name||'').slice(0,28)}}</span></td>
    <td><span class='pill'>${{p.direction}}</span></td>
    <td class='num'>${{p.distance_mi.toFixed(1)}}</td>
    <td class='num'>${{p.n_days}}</td>
    <td class='num'>${{p.pct_neighbor_led_or_tied.toFixed(0)}}%</td>
    <td class='num'>${{med}}h</td>
    <td class='num'>${{mean}}h</td>
    <td class='num'>${{xc}}</td>
    <td class='num'>${{rr}}</td>
    <td class='dist-dist'>${{distSparkline(p)}}</td>
  </tr>`;
}}

function detailRow(p) {{
  const days = p.per_day || [];
  const cells = days.map(d => {{
    const delta = d.delta_hours;
    const c = delta > 0 ? '#16a34a' : delta < 0 ? '#dc2626' : '#6b7280';
    const sign = delta > 0 ? '+' : '';
    return `<span style='color:${{c}};margin-right:8px' title='${{d.date}}: settlement ${{d.settlement_max}}°@${{d.settlement_hr}} vs neighbor ${{d.neighbor_max}}°@${{d.neighbor_hr}}'>`
      + `${{d.date.slice(5)}}:${{sign}}${{delta}}h</span>`;
  }}).join('');
  return `<tr class='detail'><td colspan='11'>per-day: ${{cells}}</td></tr>`;
}}

function render() {{
  const city = $('f-city').value;
  const dir = $('f-dir').value;
  const minD = parseInt($('f-min-days').value) || 0;
  const minP = parseFloat($('f-min-pct').value) || 0;
  let rows = PAIRS.filter(p =>
    (!city || p.city === city)
    && (!dir  || p.direction === dir)
    && p.n_days >= minD
    && p.pct_neighbor_led_or_tied >= minP
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (av == null) av = -Infinity;
    if (bv == null) bv = -Infinity;
    if (typeof av === 'number') return SORT_DIR * (av - bv);
    return SORT_DIR * String(av).localeCompare(String(bv));
  }});
  $('count').textContent = rows.length + ' / ' + PAIRS.length;
  const html = [];
  for (const p of rows) {{
    html.push(row(p));
    const key = p.city + '|' + p.neighbor_sid;
    if (EXPANDED.has(key)) html.push(detailRow(p));
  }}
  $('tbody').innerHTML = html.join('');
  document.querySelectorAll('th').forEach(th => {{
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? 'sorted-asc' : 'sorted-desc');
  }});
  // Wire row clicks to expand detail
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
['f-city','f-dir','f-min-days','f-min-pct'].forEach(id =>
  $(id).addEventListener('input', render));
render();
</script>
</body></html>"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info(f"Wrote {os.path.getsize(output_path)/1024:.0f} KB dashboard to {output_path}")


def write_csv(results: list[dict], path: str) -> None:
    fields = ["city", "settlement", "neighbor_sid", "neighbor_name", "network",
              "direction", "distance_mi", "bearing_deg", "n_days",
              "n_neighbor_led", "n_tied", "n_settl_led",
              "pct_neighbor_led_or_tied", "pct_neighbor_strictly_led",
              "median_lead_h", "mean_lead_h",
              "median_xcorr_lag", "mean_r_at_best_lag"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})
    log.info(f"Wrote {len(results)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all with neighbor data)")
    p.add_argument("--min-days", type=int, default=5,
                   help="Min overlapping days for a pair to be analyzed (default: 5)")
    p.add_argument("--station-db", default=STATION_DB,
                   help=f"Settlement station DB (default: {STATION_DB})")
    p.add_argument("--neighbor-db", default=NEIGHBOR_DB,
                   help=f"Neighbor DB (default: {NEIGHBOR_DB})")
    p.add_argument("--csv", help="Write per-pair stats to CSV")
    p.add_argument("--html", default=DEFAULT_OUT,
                   help=f"Write dashboard HTML (default: {DEFAULT_OUT})")
    p.add_argument("--no-html", action="store_true",
                   help="Skip HTML generation")
    args = p.parse_args()

    # Determine which cities have neighbor data
    with sqlite3.connect(args.neighbor_db) as conn:
        avail_cities = [r[0] for r in conn.execute(
            "SELECT DISTINCT polymarket_city FROM neighbor_meta ORDER BY polymarket_city"
        ).fetchall()]
    if not avail_cities:
        log.error(f"No neighbor data in {args.neighbor_db}. "
                   "Run scripts.neighbor_obs_pull first.")
        return 1

    cities = args.city or avail_cities
    cities = [c for c in cities if c in avail_cities]
    if not cities:
        log.error(f"No matching cities. Available: {avail_cities}")
        return 1

    log.info(f"Analyzing {len(cities)} cities: {cities}")

    all_results: list[dict] = []
    for city in cities:
        s_daily, s_hourly = load_settlement_data(args.station_db, city)
        if not s_daily:
            log.warning(f"  {city}: no settlement data in {args.station_db}")
            continue
        neighbors = load_neighbor_data(args.neighbor_db, city)
        if not neighbors:
            log.warning(f"  {city}: no neighbor data in {args.neighbor_db}")
            continue
        city_results: list[dict] = []
        for sid, data in neighbors.items():
            res = analyze_pair(city, sid, data["meta"],
                                s_daily, s_hourly,
                                data["daily"], data["hourly"],
                                args.min_days)
            if res is not None:
                city_results.append(res)
                all_results.append(res)
        print_city_report(city, city_results)

    if not all_results:
        log.error("No pairs reached --min-days threshold. "
                   "Either widen --min-days or pull more history.")
        return 1

    print()
    print("=" * 96)
    print(f"  GRAND TOTAL: {len(all_results)} (settlement, neighbor) pairs analyzed")
    print(f"  Pairs where neighbor leads >= 70% of days: "
          f"{sum(1 for r in all_results if r['pct_neighbor_led_or_tied'] >= 70)}")
    print("=" * 96)

    if not args.no_html:
        render_dashboard(all_results, args.html)
    if args.csv:
        write_csv(all_results, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())