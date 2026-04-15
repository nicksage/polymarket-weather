"""
bias_diagnostics.py — Inspect the per-(city, month, model) bias table that
the backfill populated into forecast_errors.

Outputs three sections:

  1. Overall coverage summary
     Row counts per model, date range, cities represented.

  2. Per-city, per-model bias table
     Ranked by the larger of |ECMWF bias| and |GFS bias|.  Flags cities where
     the two models disagree dramatically (|ECMWF − GFS| > 1°C) — those are
     the places where per-model correction will help the most.

  3. Seasonal breakdown for the top-N most-biased cities
     Monthly bias (Jan–Dec) for ECMWF and GFS.  Reveals whether the bias is
     constant or season-dependent (e.g. summer heat-island that winter lacks).

Usage
-----
    cd bot
    python scripts/bias_diagnostics.py              # full report
    python scripts/bias_diagnostics.py --top 20     # deeper seasonal dive
    python scripts/bias_diagnostics.py --city chicago  # single-city drill-down
    python scripts/bias_diagnostics.py --csv biases.csv  # also dump CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT  = os.path.dirname(_HERE)
sys.path.insert(0, _BOT)

from db import _get_conn                                               # noqa: E402

MODELS = ("ecmwf_ifs025", "gfs_global")
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fetch_all_errors() -> list[dict]:
    """Pull every backfilled forecast_errors row as a plain list of dicts."""
    with _get_conn() as c:
        rows = c.execute("""
            SELECT city, model, target_date, error_c, forecast_mu_c, actual_tmax_c
            FROM forecast_errors
            WHERE model IS NOT NULL
              AND error_c IS NOT NULL
              AND city IS NOT NULL
        """).fetchall()
    return [dict(r) for r in rows]


def _stats(values: list[float]) -> dict:
    """Mean / std / min / max for a list of floats.  Empty returns zeros."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {
        "n":    n,
        "mean": mean,
        "std":  var ** 0.5,
        "min":  min(values),
        "max":  max(values),
    }


# ---------------------------------------------------------------------------
# Section 1 — Overall coverage summary
# ---------------------------------------------------------------------------
def section_coverage(all_rows: list[dict]) -> None:
    print("=" * 78)
    print("1. COVERAGE SUMMARY")
    print("=" * 78)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)

    all_cities = sorted({r["city"] for r in all_rows})
    all_dates  = sorted({r["target_date"] for r in all_rows})

    print(f"  total rows: {len(all_rows):,}")
    print(f"  cities:     {len(all_cities)}")
    print(f"  date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} unique days)")
    print()
    for m in MODELS:
        rows = by_model.get(m, [])
        s = _stats([r["error_c"] for r in rows])
        print(f"  {m:<15s} n={s['n']:>7,}  "
              f"bias_mean={s['mean']:+.3f}°C  std={s['std']:.3f}°C  "
              f"min={s['min']:+.1f}°C  max={s['max']:+.1f}°C")
    print()


# ---------------------------------------------------------------------------
# Section 2 — Per-city, per-model bias table
# ---------------------------------------------------------------------------
def section_per_city(all_rows: list[dict]) -> list[dict]:
    print("=" * 78)
    print("2. PER-CITY BIAS (full-history mean)")
    print("=" * 78)

    by_cm: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in all_rows:
        by_cm[(r["city"], r["model"])].append(r["error_c"])

    cities = sorted({k[0] for k in by_cm.keys()})
    rows: list[dict] = []
    for city in cities:
        e_ecmwf = by_cm.get((city, "ecmwf_ifs025"), [])
        e_gfs   = by_cm.get((city, "gfs_global"),   [])
        if not e_ecmwf and not e_gfs:
            continue
        s_e = _stats(e_ecmwf)
        s_g = _stats(e_gfs)
        disagreement = abs(s_e["mean"] - s_g["mean"])
        rows.append({
            "city":          city,
            "ecmwf_bias":    s_e["mean"],
            "ecmwf_n":       s_e["n"],
            "ecmwf_std":     s_e["std"],
            "gfs_bias":      s_g["mean"],
            "gfs_n":         s_g["n"],
            "gfs_std":       s_g["std"],
            "disagreement":  disagreement,
            "max_abs_bias":  max(abs(s_e["mean"]), abs(s_g["mean"])),
        })

    # Sort by max |bias| descending so the most-biased cities appear first
    rows.sort(key=lambda r: r["max_abs_bias"], reverse=True)

    print(f"  {'city':<16} {'ECMWF bias':>12} {'ECMWF std':>9} {'GFS bias':>12} "
          f"{'GFS std':>9} {'n (ECMWF/GFS)':>16} {'d_model':>8}")
    print("  " + "-" * 76)
    for r in rows:
        flag = ""
        if r["disagreement"] > 1.0:
            flag = " *"          # models disagree noticeably
        if r["max_abs_bias"] > 2.0:
            flag += "!"          # large absolute bias
        print(
            f"  {r['city']:<16} "
            f"{r['ecmwf_bias']:+11.2f}°C "
            f"{r['ecmwf_std']:8.2f} "
            f"{r['gfs_bias']:+11.2f}°C "
            f"{r['gfs_std']:8.2f} "
            f"{r['ecmwf_n']:>7}/{r['gfs_n']:<7} "
            f"{r['disagreement']:6.2f}°C{flag}"
        )
    print()
    print("  Legend: * = |ECMWF-GFS| > 1C (models disagree - per-model correction matters)")
    print("          ! = |bias| > 2C (large systematic error)")
    print()
    return rows


# ---------------------------------------------------------------------------
# Section 3 — Seasonal breakdown for the top-N most-biased cities
# ---------------------------------------------------------------------------
def section_seasonal(all_rows: list[dict], top_cities: list[str]) -> None:
    print("=" * 78)
    print(f"3. SEASONAL BIAS (monthly means, top {len(top_cities)} cities)")
    print("=" * 78)
    print("  Bias by calendar month (°C) — positive = model too cold")
    print()

    by_cmm: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in all_rows:
        try:
            month = int(r["target_date"][5:7])
        except Exception:
            continue
        by_cmm[(r["city"], r["model"], month)].append(r["error_c"])

    for city in top_cities:
        print(f"  [{city}]")
        for model in MODELS:
            vals = []
            for m in range(1, 13):
                xs = by_cmm.get((city, model, m), [])
                vals.append(sum(xs) / len(xs) if xs else None)
            row = "    " + f"{model:<15}"
            for v in vals:
                row += f"{v:+6.1f}" if v is not None else "   -- "
            print(row)
        print("    " + "month          " + "".join(f"{m:>6}" for m in _MONTHS))
        print()


# ---------------------------------------------------------------------------
# Optional — dump CSV for external analysis
# ---------------------------------------------------------------------------
def write_csv(per_city: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["city", "ecmwf_bias", "ecmwf_std", "ecmwf_n",
                        "gfs_bias", "gfs_std", "gfs_n",
                        "disagreement", "max_abs_bias"],
        )
        w.writeheader()
        for r in per_city:
            w.writerow(r)
    print(f"CSV written: {path}")


# ---------------------------------------------------------------------------
# Single-city drill-down mode
# ---------------------------------------------------------------------------
def city_drilldown(all_rows: list[dict], city: str) -> None:
    city_rows = [r for r in all_rows if r["city"] == city]
    if not city_rows:
        print(f"No data for city '{city}'")
        return
    print("=" * 78)
    print(f"DRILL-DOWN: {city}")
    print("=" * 78)

    # Per-model full history
    for m in MODELS:
        vals = [r["error_c"] for r in city_rows if r["model"] == m]
        s = _stats(vals)
        print(f"  {m}: n={s['n']} bias={s['mean']:+.3f}°C std={s['std']:.3f}°C "
              f"min={s['min']:+.1f}°C max={s['max']:+.1f}°C")
    print()

    # Per-month breakdown
    print("  Month  " + "".join(f"{x:>10}" for x in ["ECMWF bias", "ECMWF n", "GFS bias", "GFS n"]))
    print("  " + "-" * 52)
    for m in range(1, 13):
        vals_e = [r["error_c"] for r in city_rows if r["model"] == "ecmwf_ifs025"
                  and r["target_date"][5:7] == f"{m:02d}"]
        vals_g = [r["error_c"] for r in city_rows if r["model"] == "gfs_global"
                  and r["target_date"][5:7] == f"{m:02d}"]
        eb = sum(vals_e)/len(vals_e) if vals_e else None
        gb = sum(vals_g)/len(vals_g) if vals_g else None
        eb_s = f"{eb:+9.2f}" if eb is not None else "       --"
        gb_s = f"{gb:+9.2f}" if gb is not None else "       --"
        print(f"  {_MONTHS[m-1]:<6} {eb_s:>10} {len(vals_e):>10} {gb_s:>10} {len(vals_g):>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=10,
                   help="How many top-biased cities to show in seasonal section (default 10)")
    p.add_argument("--city", type=str, default="",
                   help="Drill down on a single city instead of the full report")
    p.add_argument("--csv", type=str, default="",
                   help="Also write per-city table to this CSV path")
    args = p.parse_args()

    rows = _fetch_all_errors()
    if not rows:
        print("No backfilled forecast_errors rows found — run backfill_bias_data.py first.")
        return 1

    if args.city:
        city_drilldown(rows, args.city.lower())
        return 0

    section_coverage(rows)
    per_city = section_per_city(rows)
    top_cities = [r["city"] for r in per_city[:args.top]]
    section_seasonal(rows, top_cities)

    if args.csv:
        write_csv(per_city, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
