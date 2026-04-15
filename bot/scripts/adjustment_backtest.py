"""
adjustment_backtest.py — Floor-focused backtest with σ-shrinkage variants
(Phase 2b v2).

For every (city, target_date) with actual tmax + bin structure + hourly obs
we run 20-minute replay points across four variants:

  1. baseline             — blended μ/σ, no floor, no shrinkage
  2. floor_only           — blended μ/σ + floor truncation (v1 result)
  3. v2_time              — blended μ + time-shrunk σ + floor (primary gate)
  4. v2_time_surprise     — blended μ + (time+residual)-shrunk σ + floor (diag)

v2_time is the PRIMARY PRODUCTION-GATING VARIANT.  v2_time_surprise is
exploratory because its "residual" is a day-level surprise proxy (not a
true forecast-path deviation) and will be validated separately via live
shadow-mode accumulation.

Outputs:
  backtest.db → backtest_runs + backtest_results rows tagged with variant
  logs/backtest_{run_id}.md             — human-readable summary
  logs/backtest_{run_id}_reliability.csv — reliability data for plotting

Usage:
    python bot/scripts/adjustment_backtest.py --days 90
    python bot/scripts/adjustment_backtest.py --variant v2_time
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import config
from backtest_db import (
    init_backtest_db, get_historical_hourly_obs, get_empirical_sigma,
    insert_backtest_run, finalize_backtest_run, insert_backtest_results_bulk,
)
from live_adjustment import _compute_from_data
from solar import remaining_daylight_fraction
from weather import get_temp_range_probability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("backtest")

REPLAY_INTERVAL_MIN = 20
REPLAY_START_LOCAL_HOUR = 6
REPLAY_END_LOCAL_HOUR = 20

VARIANTS = ["baseline", "floor_only", "v2_time", "v2_time_surprise"]


def _iter_events(window_days: int) -> Iterable[dict]:
    start = (date.today() - timedelta(days=window_days)).isoformat()
    end = date.today().isoformat()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        WITH latest AS (
            SELECT te.id, te.event_id, te.city, te.date, te.lat, te.lon,
                   te.forecast_mu_c, te.forecast_sigma_c, te.days_ahead,
                   ROW_NUMBER() OVER (PARTITION BY te.event_id ORDER BY te.scan_timestamp DESC) rn
            FROM temp_events te
            WHERE te.date >= ? AND te.date < ?
              AND te.lat IS NOT NULL AND te.lon IS NOT NULL
        )
        SELECT l.*, h.tempmax_c AS actual_tmax_c
        FROM latest l
        JOIN historical_observed_daily h
          ON ROUND(l.lat, 2) = h.lat_key
         AND ROUND(l.lon, 2) = h.lon_key
         AND l.date = h.date
        WHERE l.rn = 1 AND h.tempmax_c IS NOT NULL
        ORDER BY l.date, l.city
    """, (start, end)).fetchall()

    for r in rows:
        bins = conn.execute("""
            SELECT contract_id, range_low, range_high, unit
            FROM temp_outcomes
            WHERE event_row_id = ?
              AND range_low IS NOT NULL AND range_high IS NOT NULL
        """, (r["id"],)).fetchall()
        ev = dict(r)
        ev["bins"] = [dict(b) for b in bins]
        yield ev


def _to_c(v, unit: str) -> float | None:
    if v is None:
        return None
    return (v - 32) * 5.0 / 9.0 if unit == "fahrenheit" else float(v)


def _bin_contains(tmax_c: float, lo_c: float | None, hi_c: float | None) -> int:
    if lo_c is None and hi_c is None:
        return 1
    if lo_c is not None and tmax_c < lo_c:
        return 0
    if hi_c is not None and tmax_c >= hi_c:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def _variant_config(variant: str) -> dict:
    """Map variant name to (sigma_mode, apply_floor, apply_flatten)."""
    if variant == "baseline":
        return {"sigma_mode": "none",                "apply_floor": False, "apply_flatten": False}
    if variant == "floor_only":
        return {"sigma_mode": "none",                "apply_floor": True,  "apply_flatten": False}
    if variant == "v2_time":
        return {"sigma_mode": "time_only",           "apply_floor": True,  "apply_flatten": True}
    if variant == "v2_time_surprise":
        return {"sigma_mode": "time+surprise_proxy", "apply_floor": True,  "apply_flatten": True}
    raise ValueError(f"unknown variant: {variant}")


def _flatten_and_renormalize(probs: list[float], exponent: float) -> list[float]:
    """Apply p' = p ** exponent to each bin then renormalize so probs sum to 1.
    Preserves per-bin ranking.  Exponent == 1.0 is a no-op."""
    if exponent == 1.0 or not probs:
        return probs
    raised = [max(p, 0.0) ** exponent for p in probs]
    total = sum(raised)
    if total <= 0:
        return probs
    return [p / total for p in raised]


def replay_event(event: dict, run_id: int, variants: list[str]) -> tuple[int, int]:
    """Run every variant on this event.  Returns (n_replay_points, n_rows_written).
    n_replay_points is counted once per replay time (not per variant)."""
    hourly = get_historical_hourly_obs(event["lat"], event["lon"], event["date"])
    if not hourly:
        return (0, 0)
    obs_sorted = sorted(hourly, key=lambda r: r["hour_ts_utc"])

    blended_mu    = event.get("forecast_mu_c")
    blended_sigma = event.get("forecast_sigma_c")
    if blended_sigma is None or blended_sigma <= 0:
        e_sigma = get_empirical_sigma("ecmwf", 3) or 2.0
        g_sigma = get_empirical_sigma("gfs",   3) or 2.2
        blended_sigma = (e_sigma + g_sigma) / 2.0
    if blended_mu is None:
        return (0, 0)

    target_date = date.fromisoformat(event["date"])
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    step = timedelta(minutes=REPLAY_INTERVAL_MIN)

    actual_tmax = float(event["actual_tmax_c"])
    lat = float(event["lat"])
    lon = float(event["lon"])

    results: list[dict] = []
    n_pts = 0
    t = day_start + timedelta(hours=REPLAY_START_LOCAL_HOUR)
    end_t = day_start + timedelta(hours=REPLAY_END_LOCAL_HOUR)

    while t <= end_t:
        obs_before: list[dict] = []
        for o in obs_sorted:
            try:
                ots = datetime.fromisoformat(o["hour_ts_utc"].replace("Z", "+00:00"))
            except Exception:
                continue
            if ots <= t:
                obs_before.append({
                    "current_temp_c": o["temp_c"],
                    "pulled_at_utc":  o["hour_ts_utc"],
                })

        frac = remaining_daylight_fraction(lat, lon, t)

        # Precompute bin Celsius bounds (same for all variants)
        bin_ctx: list[dict] = []
        for b in event["bins"]:
            unit = b.get("unit", "celsius")
            lo_c = _to_c(b["range_low"],  unit)
            hi_c = _to_c(b["range_high"], unit)
            if lo_c is not None and hi_c is not None and lo_c == hi_c:
                hi_c = lo_c + (1.0 if unit == "celsius" else 5.0/9.0)
            bin_ctx.append({
                "contract_id": b["contract_id"],
                "unit": unit, "lo_raw": b["range_low"], "hi_raw": b["range_high"],
                "lo_c": lo_c, "hi_c": hi_c,
                "actual_in_bin": _bin_contains(actual_tmax, lo_c, hi_c),
            })

        for variant in variants:
            cfg = _variant_config(variant)
            adj = _compute_from_data(
                blended_mu_c            = blended_mu,
                blended_sigma_c         = blended_sigma,
                obs_rows                = obs_before,
                hourly_rows             = [],
                hours_since_forecast    = None,
                timezone_str            = None,
                now_utc                 = t,
                residual_enabled        = False,       # μ adjustment always off in backtest
                sigma_mode              = cfg["sigma_mode"],
                remaining_daylight_frac = frac,
            )

            obs_floor = adj["observed_max_so_far_c"]
            apply_floor = cfg["apply_floor"] and obs_floor is not None
            floor_applied = 1 if apply_floor else 0

            active_sigma = adj["adjusted_sigma_c"]

            # Phase 1 — compute per-bin probabilities for this variant
            p_blended_list: list[float] = []
            p_blended_floor_list: list[float] = []
            p_adjusted_list: list[float] = []
            for bc in bin_ctx:
                p_blended_list.append(get_temp_range_probability(
                    blended_mu, blended_sigma,
                    bc["lo_raw"], bc["hi_raw"], bc["unit"],
                ))
                p_blended_floor_list.append(get_temp_range_probability(
                    blended_mu, blended_sigma,
                    bc["lo_raw"], bc["hi_raw"], bc["unit"],
                    obs_floor_c=obs_floor if apply_floor else None,
                ))
                p_adjusted_list.append(get_temp_range_probability(
                    blended_mu, active_sigma,
                    bc["lo_raw"], bc["hi_raw"], bc["unit"],
                    obs_floor_c=obs_floor if apply_floor else None,
                ))

            # Phase 2 — tail flattening for v2 variants.  Apply p**E directly
            # to the raw per-bin probabilities (which are the truncated-
            # distribution conditional probabilities) and renormalize once.
            # Intentionally NOT renormalizing before flattening — that would
            # artificially inflate concentrated bins before the flattening
            # could temper them, producing the opposite of the intended
            # effect.
            if cfg["apply_flatten"]:
                p_adjusted_list = _flatten_and_renormalize(
                    p_adjusted_list, config.LIVE_ADJ_TAIL_FLATTEN_EXPONENT,
                )

            # Phase 3 — write rows
            for bc, p_b, p_bf, p_a in zip(bin_ctx, p_blended_list,
                                          p_blended_floor_list, p_adjusted_list):
                results.append({
                    "run_id":              run_id,
                    "variant":             variant,
                    "event_id":            event.get("event_id"),
                    "city":                event.get("city"),
                    "target_date":         event.get("date"),
                    "replay_at_utc":       t.isoformat(),
                    "local_hour":          float(t.hour) + t.minute / 60.0,
                    "contract_id":         bc["contract_id"],
                    "bin_low_c":           bc["lo_c"],
                    "bin_high_c":          bc["hi_c"],
                    "blended_mu_c":        blended_mu,
                    "blended_sigma_c":     blended_sigma,
                    "adjusted_mu_c":       adj["adjusted_mu_c"],
                    "adjusted_sigma_c":    adj["adjusted_sigma_c"],
                    "mu_adjustment_c":     adj["mu_adjustment_c"],
                    "obs_floor_c":         obs_floor,
                    "p_blended":           p_b,
                    "p_blended_floor":     p_bf,
                    "p_adjusted":          p_a,
                    "actual_tmax_c":       actual_tmax,
                    "bin_contains_actual": bc["actual_in_bin"],
                    "floor_applied":       floor_applied,
                    "cold_start":          0,
                    "score":               adj["live_adjustment_score"],
                    "components_json":     json.dumps(adj["components"], default=str),
                })
        n_pts += 1
        t += step

    if results:
        insert_backtest_results_bulk(results)
    return (n_pts, len(results))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _brier(rows: list[dict], p_key: str) -> float:
    vals = [(r[p_key] - r["bin_contains_actual"]) ** 2
            for r in rows if r[p_key] is not None]
    return (sum(vals) / len(vals)) if vals else float("nan")


def _reliability(rows: list[dict], p_key: str, n_buckets: int = 20) -> list[dict]:
    buckets = [{"lo": i / n_buckets, "hi": (i + 1) / n_buckets,
                "pred_sum": 0.0, "actual_sum": 0, "n": 0}
               for i in range(n_buckets)]
    for r in rows:
        p = r[p_key]
        if p is None:
            continue
        idx = min(int(p * n_buckets), n_buckets - 1)
        b = buckets[idx]
        b["pred_sum"]   += p
        b["actual_sum"] += r["bin_contains_actual"]
        b["n"]          += 1
    for b in buckets:
        b["pred_mean"]      = (b["pred_sum"] / b["n"]) if b["n"] > 0 else None
        b["empirical_freq"] = (b["actual_sum"] / b["n"]) if b["n"] > 0 else None
    return buckets


def _high_confidence_gap(rows: list[dict], p_key: str, threshold: float = 0.7) -> dict:
    """Average calibration gap + overconfidence volume + tail max for rows
    with predicted probability >= threshold."""
    hi = [r for r in rows if r[p_key] is not None and r[p_key] >= threshold]
    if not hi:
        return {"n": 0, "n_wrong": 0, "pred_mean": None, "empirical_freq": None,
                "gap": None, "max_p": None}
    pred_mean = sum(r[p_key] for r in hi) / len(hi)
    emp_freq  = sum(r["bin_contains_actual"] for r in hi) / len(hi)
    n_wrong   = sum(1 for r in hi if r["bin_contains_actual"] == 0)
    return {
        "n":              len(hi),
        "n_wrong":        n_wrong,
        "pred_mean":      pred_mean,
        "empirical_freq": emp_freq,
        "gap":            pred_mean - emp_freq,
        "max_p":          max(r[p_key] for r in hi),
    }


def _max_bucket_above_threshold(
    rows: list[dict], p_key: str, threshold: float = 0.7, n_buckets: int = 20,
) -> int:
    """Return the largest single-bucket count among reliability buckets whose
    predicted-mean is >= threshold.  Used for tail safety criterion 3B."""
    buckets = _reliability(rows, p_key, n_buckets=n_buckets)
    big = [b["n"] for b in buckets
           if b["pred_mean"] is not None and b["pred_mean"] >= threshold]
    return max(big) if big else 0


def _bucket_hour(h: float) -> str:
    if h < 10:  return "06-10"
    if h < 13:  return "10-13"
    if h < 17:  return "13-17"
    return "17-20"


def _summarize_variant(rows: list[dict], p_key: str = "p_adjusted") -> dict:
    by_hour: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_hour[_bucket_hour(r["local_hour"])].append(r)

    return {
        "n_rows":         len(rows),
        "brier":          _brier(rows, p_key),
        "brier_by_hour":  {k: _brier(v, p_key) for k, v in by_hour.items()},
        "high_conf_gap":  _high_confidence_gap(rows, p_key),
    }


def _floor_hit_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    # Unique (event, replay_at) combos
    seen: set[tuple] = set()
    hit = 0
    for r in rows:
        k = (r["event_id"], r["replay_at_utc"])
        if k in seen:
            continue
        seen.add(k)
        if r["floor_applied"]:
            hit += 1
    return hit / len(seen) if seen else 0.0


def _sigma_floor_hit_rate(rows: list[dict]) -> dict:
    """Fraction of rows where the σ hard floor (abs or rel) kicked in."""
    if not rows:
        return {"abs": 0.0, "rel": 0.0}
    abs_n = rel_n = 0
    n = 0
    for r in rows:
        try:
            c = json.loads(r.get("components_json") or "{}")
        except Exception:
            continue
        n += 1
        if c.get("sigma_abs_floor_hit"):
            abs_n += 1
        if c.get("sigma_rel_floor_hit"):
            rel_n += 1
    if n == 0:
        return {"abs": 0.0, "rel": 0.0}
    return {"abs": abs_n / n, "rel": rel_n / n}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    run_id: int,
    rows_by_variant: dict[str, list[dict]],
    events_n: int, window_days: int,
    variants: list[str],
) -> str:
    logs_dir = os.path.join(_BOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    report_path = os.path.join(logs_dir, f"backtest_{run_id}.md")
    csv_path    = os.path.join(logs_dir, f"backtest_{run_id}_reliability.csv")

    # Summarize each variant using its own p_adjusted column
    summaries = {v: _summarize_variant(rows_by_variant[v]) for v in variants}

    # Reference baseline for delta calculations
    base = summaries.get("baseline")
    baseline_brier = base["brier"] if base else None

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Live-Adjustment Backtest Report (v2)\n\n")
        f.write(f"Run ID: {run_id}  \n")
        f.write(f"Window: last {window_days} days  \n")
        f.write(f"Events: {events_n}  \n")
        f.write(f"Variants: {', '.join(variants)}  \n\n")

        f.write("## Variant Comparison (lower Brier = better)\n\n")
        f.write("| Variant | Brier (all) | vs baseline | Brier 17-20 | vs baseline | Floor-hit % | σ-floor abs% | σ-floor rel% |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for v in variants:
            s = summaries[v]
            b = s["brier"]
            late = s["brier_by_hour"].get("17-20", float("nan"))
            d_all  = (b    - baseline_brier) if baseline_brier is not None else 0.0
            d_late = (late - summaries["baseline"]["brier_by_hour"].get("17-20", 0)) \
                     if "baseline" in summaries else 0.0
            fhr  = _floor_hit_rate(rows_by_variant[v]) * 100
            sfh  = _sigma_floor_hit_rate(rows_by_variant[v])
            f.write(f"| {v:<18} | {b:.5f} | {d_all:+.5f} | {late:.5f} | {d_late:+.5f} | "
                    f"{fhr:.1f}% | {sfh['abs']*100:.1f}% | {sfh['rel']*100:.1f}% |\n")
        f.write("\n_Delta < 0 means the variant improves Brier vs the unadjusted baseline._\n\n")

        f.write("## High-Confidence Calibration Gap (predicted >= 0.7)\n\n")
        f.write("Primary overconfidence metric.  Positive gap = overconfident.  "
                "v2 success means the gap shrinks vs floor_only.\n\n")
        f.write("| Variant | N | Predicted mean | Empirical freq | Gap (pred - emp) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for v in variants:
            g = summaries[v]["high_conf_gap"]
            pm = f"{g['pred_mean']:.3f}" if g["pred_mean"] is not None else "—"
            ef = f"{g['empirical_freq']:.3f}" if g["empirical_freq"] is not None else "—"
            gp = f"{g['gap']:+.3f}" if g["gap"] is not None else "—"
            f.write(f"| {v:<18} | {g['n']} | {pm} | {ef} | **{gp}** |\n")
        f.write("\n")

        f.write("## Brier by Local-Hour Bucket\n\n")
        f.write("| Variant | 06-10 | 10-13 | 13-17 | 17-20 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for v in variants:
            bh = summaries[v]["brier_by_hour"]
            f.write(f"| {v:<18} | {bh.get('06-10', float('nan')):.5f} | "
                    f"{bh.get('10-13', float('nan')):.5f} | "
                    f"{bh.get('13-17', float('nan')):.5f} | "
                    f"{bh.get('17-20', float('nan')):.5f} |\n")
        f.write("\n")

        # Production gating — v2_time is the primary gating variant.
        # 3A = overconfidence volume reduction vs floor_only (>=80%).
        # 3B = tail safety: max predicted p <= 0.75  OR  no bucket with
        #      pred>=0.7 has N>=50.
        # Decision rule (from user): all of Brier (all) >=10% better than
        # baseline, Brier 17-20 better, 3A, and 3B must pass.
        f.write("## Production Gating (primary variant: v2_time)\n\n")
        have = ("v2_time" in summaries and "baseline" in summaries
                and "floor_only" in summaries)
        if have:
            v_rows  = rows_by_variant["v2_time"]
            fo_rows = rows_by_variant["floor_only"]
            v  = summaries["v2_time"]
            b  = summaries["baseline"]

            # Criterion 1 — Brier (all) improvement >= 10% vs baseline
            brier_pct_improve = ((b["brier"] - v["brier"]) / b["brier"] * 100.0
                                 if b["brier"] else 0.0)
            c1 = brier_pct_improve >= 10.0

            # Criterion 2 — Brier 17-20 better than baseline (strict)
            b_late = b["brier_by_hour"].get("17-20", float("inf"))
            v_late = v["brier_by_hour"].get("17-20", float("inf"))
            c2 = v_late < b_late

            # Criterion 3A — overconfidence volume reduction >= 80% vs floor_only
            wrong_fo = summaries["floor_only"]["high_conf_gap"]["n_wrong"]
            wrong_v2 = v["high_conf_gap"]["n_wrong"]
            if wrong_fo > 0:
                vol_reduction_pct = (1.0 - wrong_v2 / wrong_fo) * 100.0
            else:
                vol_reduction_pct = 100.0 if wrong_v2 == 0 else 0.0
            c3a = vol_reduction_pct >= 80.0

            # Criterion 3B — tail safety
            max_p = v["high_conf_gap"]["max_p"]
            max_bucket_big = _max_bucket_above_threshold(v_rows, "p_adjusted", 0.7)
            tail_max_ok   = (max_p is None) or (max_p <= 0.75)
            tail_nbkt_ok  = max_bucket_big < 50
            c3b = tail_max_ok or tail_nbkt_ok

            all_pass = c1 and c2 and c3a and c3b

            f.write("| Criterion | Detail | Result |\n|---|---|---|\n")
            f.write(f"| (1) Brier (all) >=10% better than baseline | "
                    f"{brier_pct_improve:+.2f}% | {'PASS' if c1 else 'FAIL'} |\n")
            f.write(f"| (2) Brier 17-20 v2_time < baseline | "
                    f"{v_late:.5f} vs {b_late:.5f} | {'PASS' if c2 else 'FAIL'} |\n")
            f.write(f"| (3A) Overconfidence-wrongs >=80% lower vs floor_only | "
                    f"wrong: {wrong_fo} -> {wrong_v2} ({vol_reduction_pct:+.1f}%) | "
                    f"{'PASS' if c3a else 'FAIL'} |\n")
            f.write(f"| (3B) Tail safety (max_p<=0.75 OR no pred>=0.7 bucket with N>=50) | "
                    f"max_p={max_p if max_p else 'n/a'}, "
                    f"max_bucket_above_0.7={max_bucket_big} | "
                    f"{'PASS' if c3b else 'FAIL'} |\n")
            verdict = ("PASS - safe to enable LIVE_ADJ_OBS_FLOOR + SIGMA_SHRINK "
                       "in production" if all_pass
                       else "FAIL - keep both disabled")
            f.write(f"\n**Overall: {verdict}**\n\n")
        else:
            f.write("_skipped - missing baseline, floor_only, or v2_time_\n\n")

        # Reliability CSV — include all variants
        with open(csv_path, "w", encoding="utf-8") as fc:
            fc.write("variant,bucket_lo,bucket_hi,pred_mean,empirical_freq,n\n")
            for v in variants:
                reliability = _reliability(rows_by_variant[v], "p_adjusted")
                for b in reliability:
                    pm = f"{b['pred_mean']:.5f}" if b['pred_mean'] is not None else ""
                    ef = f"{b['empirical_freq']:.5f}" if b['empirical_freq'] is not None else ""
                    fc.write(f"{v},{b['lo']:.3f},{b['hi']:.3f},{pm},{ef},{b['n']}\n")

        f.write("## Reliability — v2_time (buckets with N > 10)\n\n")
        f.write("Full detail per variant in the companion CSV.\n\n")
        f.write("| Predicted | Empirical | N |\n|---:|---:|---:|\n")
        for b in _reliability(rows_by_variant.get("v2_time", []), "p_adjusted"):
            if b["n"] > 10:
                f.write(f"| {b['pred_mean']:.3f} | {b['empirical_freq']:.3f} | {b['n']} |\n")

    return report_path


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--variant", type=str, default=None,
                        help="Restrict to one variant (debugging).  Default: run all 4.")
    args = parser.parse_args()

    init_backtest_db()

    variants = [args.variant] if args.variant else VARIANTS
    if args.variant and args.variant not in VARIANTS:
        log.error(f"Unknown variant: {args.variant}.  Choose from: {VARIANTS}")
        return 1

    started = datetime.now(timezone.utc).isoformat()
    params = {
        "variants": variants,
        "USE_LIVE_ADJUSTMENT_SIGMA_SHRINK": config.USE_LIVE_ADJUSTMENT_SIGMA_SHRINK,
        "LIVE_ADJ_SIGMA_TIME_MIN":          config.LIVE_ADJ_SIGMA_TIME_MIN,
        "LIVE_ADJ_SIGMA_RESIDUAL_DIV":      config.LIVE_ADJ_SIGMA_RESIDUAL_DIV,
        "LIVE_ADJ_SIGMA_RESIDUAL_CAP":      config.LIVE_ADJ_SIGMA_RESIDUAL_CAP,
        "LIVE_ADJ_SIGMA_ABS_FLOOR_C":       config.LIVE_ADJ_SIGMA_ABS_FLOOR_C,
        "LIVE_ADJ_SIGMA_REL_FLOOR":         config.LIVE_ADJ_SIGMA_REL_FLOOR,
    }
    run_id = insert_backtest_run(started, "v2_multi_variant",
                                 json.dumps(params), args.days)
    log.info(f"Backtest run_id={run_id} window={args.days}d variants={variants}")

    events = list(_iter_events(args.days))
    log.info(f"Events to replay: {len(events)}")

    n_replay_pts = 0
    n_rows = 0
    replayed = 0
    for ev in events:
        pts, rows = replay_event(ev, run_id, variants)
        n_replay_pts += pts
        n_rows       += rows
        if rows:
            replayed += 1
        if replayed % 20 == 0 and replayed > 0:
            log.info(f"  {replayed}/{len(events)} events, {n_replay_pts} pts, {n_rows} rows")

    log.info(f"Replay done: {replayed}/{len(events)} events, "
             f"{n_replay_pts} points, {n_rows} rows")

    # Pull results grouped by variant
    conn = sqlite3.connect(config.BACKTEST_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows_by_variant: dict[str, list[dict]] = {}
    for v in variants:
        rs = conn.execute(
            "SELECT * FROM backtest_results WHERE run_id = ? AND variant = ?",
            (run_id, v),
        ).fetchall()
        rows_by_variant[v] = [dict(r) for r in rs]
        log.info(f"  variant {v}: {len(rows_by_variant[v])} rows")

    report_path = write_report(run_id, rows_by_variant, replayed, args.days, variants)

    summary = {v: {
        "brier":         _brier(rows_by_variant[v], "p_adjusted"),
        "high_conf_gap": _high_confidence_gap(rows_by_variant[v], "p_adjusted"),
    } for v in variants}

    finalize_backtest_run(
        run_id=run_id,
        finished_at=datetime.now(timezone.utc).isoformat(),
        n_events=replayed,
        n_replay_pts=n_replay_pts,
        summary_json=json.dumps(summary, default=str),
    )
    log.info(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
