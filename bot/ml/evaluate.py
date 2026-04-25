"""
evaluate.py — Phase 6 v1 evaluation of TempDistributionModel.

Uses the existing ml_training_rows + the trained model's CV pattern to
answer "is this ML model worth turning on?" — without waiting for live
shadow logging.

What it computes (from out-of-fold predictions on the training set):
  * Headline RMSE / mean bias / fallback σ
  * Per-month RMSE breakdown (which seasons does the model fail on?)
  * Per-decision-hour RMSE (10:00 vs 12:00 — does later help?)
  * Calibration: do |residuals| respect the predicted σ?
  * Naive baseline RMSEs (persistence, climatology, year-over-year) so
    we can see whether ML actually adds value over trivial heuristics

Writes one markdown report per city to bot/ml/diagnostics/{city}_{date}_report.md.

Usage
-----
    python -m bot.ml.evaluate                     # all eligible cities
    python -m bot.ml.evaluate --city Chicago      # one city
    python -m bot.ml.evaluate --folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from db import init_db, load_ml_training_rows, count_ml_training_rows, get_ml_backfill_cities
from ml.schema import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("ml_evaluate")

DIAG_DIR = Path(_BOT_DIR) / "ml" / "diagnostics"

# Must match TempDistributionModel defaults so OOF metrics here match
# what training reports.
_HPARAMS = dict(
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42,
)
MIN_BUCKET_SAMPLES = 30


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_dataset(city: str) -> tuple[np.ndarray, np.ndarray, list[date], list[int]] | None:
    rows = load_ml_training_rows(city, FEATURE_VERSION)
    if not rows:
        return None
    n = len(rows)
    X = np.full((n, N_FEATURES), np.nan, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    dates: list[date] = []
    decision_hours: list[int] = []
    for i, r in enumerate(rows):
        feats = json.loads(r["features_json"])
        for j, v in enumerate(feats):
            if v is not None:
                X[i, j] = float(v)
        y[i] = float(r["t_max_c"])
        dates.append(date.fromisoformat(r["target_date"]))
        decision_hours.append(int(r["decision_hour_local"]))
    return X, y, dates, decision_hours


# ---------------------------------------------------------------------------
# OOF predictions (date-grouped TimeSeriesSplit, matches model.fit())
# ---------------------------------------------------------------------------

def compute_oof_predictions(
    X: np.ndarray, y: np.ndarray, dates: list[date], n_folds: int = 5
) -> np.ndarray:
    """Out-of-fold predictions via the same date-grouped CV used in
    TempDistributionModel.fit() — kept here so evaluate doesn't depend on
    internal model state.  Two rows for the same date land in the same fold."""
    unique_dates = sorted(set(dates))
    if len(unique_dates) < n_folds + 1:
        raise ValueError(
            f"need at least n_folds+1={n_folds+1} unique dates, got {len(unique_dates)}"
        )
    target_dates_np = np.array(dates)
    date_to_mask = {d: (target_dates_np == d) for d in unique_dates}

    oof = np.full(len(y), np.nan, dtype=np.float64)
    tss = TimeSeriesSplit(n_splits=n_folds)
    for fold_idx, (train_date_idx, val_date_idx) in enumerate(
        tss.split(unique_dates)
    ):
        train_dates = {unique_dates[i] for i in train_date_idx}
        val_dates   = {unique_dates[i] for i in val_date_idx}
        train_mask = np.zeros(len(y), dtype=bool)
        val_mask   = np.zeros(len(y), dtype=bool)
        for d in train_dates:
            train_mask |= date_to_mask[d]
        for d in val_dates:
            val_mask   |= date_to_mask[d]
        m = HistGradientBoostingRegressor(**_HPARAMS)
        m.fit(X[train_mask], y[train_mask])
        oof[val_mask] = m.predict(X[val_mask])
    return oof


# ---------------------------------------------------------------------------
# Per-city evaluation
# ---------------------------------------------------------------------------

def evaluate_city(city: str, n_folds: int = 5) -> dict:
    log.info(f"--- evaluating {city} ---")
    ds = _load_dataset(city)
    if ds is None:
        return {"city": city, "status": "no_rows"}
    X, y, dates, decision_hours = ds
    if len(set(dates)) < n_folds + 1:
        return {"city": city, "status": "insufficient_dates",
                "unique_dates": len(set(dates))}

    # ----- OOF predictions -----
    try:
        oof = compute_oof_predictions(X, y, dates, n_folds=n_folds)
    except Exception as e:
        return {"city": city, "status": "oof_failed", "error": str(e)}
    valid = ~np.isnan(oof)
    if valid.sum() == 0:
        return {"city": city, "status": "no_oof"}

    residuals = y[valid] - oof[valid]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))

    months = np.array([d.month for d in dates])
    hours  = np.array(decision_hours)
    target_dates_arr = np.array(dates)

    # ----- Per-month -----
    per_month: dict[int, dict] = {}
    for m in range(1, 13):
        mask = (months == m) & valid
        if mask.sum() >= 10:
            r = y[mask] - oof[mask]
            per_month[m] = {
                "n":     int(mask.sum()),
                "rmse":  float(np.sqrt(np.mean(r ** 2))),
                "sigma": float(np.std(r)),
                "bias":  float(np.mean(r)),
            }

    # ----- Per-decision-hour -----
    per_hour: dict[int, dict] = {}
    for h in (10, 12):
        mask = (hours == h) & valid
        if mask.sum() >= 10:
            r = y[mask] - oof[mask]
            per_hour[h] = {
                "n":    int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "bias": float(np.mean(r)),
            }

    # ----- σ buckets (per-month, fallback city-wide) -----
    spread_per_bucket: dict[int, float] = {}
    for m in range(1, 13):
        mask = months[valid] == m
        if mask.sum() >= MIN_BUCKET_SAMPLES:
            spread_per_bucket[m] = float(np.std(residuals[mask]))
    fallback_sigma = float(np.std(residuals))

    # ----- Calibration: do |residuals| respect the predicted σ? -----
    sigmas_used = np.array([
        spread_per_bucket.get(d.month, fallback_sigma)
        for d in target_dates_arr[valid]
    ])
    z = np.abs(residuals) / sigmas_used
    calibration = {
        "frac_within_1sigma": float(np.mean(z <= 1.0)),
        "frac_within_2sigma": float(np.mean(z <= 2.0)),
        "frac_within_3sigma": float(np.mean(z <= 3.0)),
    }

    # ----- Naive baselines -----
    feat_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    baselines: dict[str, float | None] = {}
    for label, fname in [
        ("persistence_yesterday",     "tmax_yesterday"),
        ("yoy_same_date_last_year",   "tmax_same_date_last_year"),
        ("lag_7d_ago",                "tmax_7d_ago"),
    ]:
        col_idx = feat_idx[fname]
        baseline = X[:, col_idx]
        present = ~np.isnan(baseline) & valid
        if present.sum() < 10:
            baselines[label] = None
            continue
        diff = y[present] - baseline[present]
        baselines[label] = float(np.sqrt(np.mean(diff ** 2)))

    # Climatology baseline — leave-one-(date)-out mean T_max for each
    # day-of-year.  Excluding all rows that share the prediction's target
    # date (not just the row itself) prevents the trivial case where the
    # 10:00 and 12:00 rows for the same date "predict" each other through
    # the doy mean.  With <2 years of data most doys have only one
    # contributing date, so the LOO baseline is mostly N/A — that's the
    # honest report; climatology becomes useful only with multi-year data.
    doy_arr = np.array([d.timetuple().tm_yday for d in dates])
    date_arr_iso = np.array([d.isoformat() for d in dates])
    clim_pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        if not valid[i]:
            continue
        same_doy_other_date = (doy_arr == doy_arr[i]) & (date_arr_iso != date_arr_iso[i])
        if same_doy_other_date.sum() >= 1:
            clim_pred[i] = float(np.mean(y[same_doy_other_date]))
    clim_valid = ~np.isnan(clim_pred) & valid
    if clim_valid.sum() >= 10:
        clim_rmse = float(np.sqrt(np.mean((y[clim_valid] - clim_pred[clim_valid]) ** 2)))
        baselines["climatology_doy_mean"] = clim_rmse
    else:
        baselines["climatology_doy_mean"] = None

    return {
        "city":              city,
        "status":            "ok",
        "n_rows":            int(X.shape[0]),
        "n_valid_oof":       int(valid.sum()),
        "n_unique_dates":    len(set(dates)),
        "rmse_c":            rmse,
        "mean_bias_c":       bias,
        "fallback_sigma_c":  fallback_sigma,
        "per_month":         per_month,
        "per_hour":          per_hour,
        "calibration":       calibration,
        "baselines":         baselines,
        "spread_per_bucket": spread_per_bucket,
        "window_start":      sorted(set(dates))[0].isoformat(),
        "window_end":        sorted(set(dates))[-1].isoformat(),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(result: dict, out_path: Path) -> None:
    if result["status"] != "ok":
        return

    md: list[str] = []
    md.append(f"# ML Distribution Model — Evaluation Report")
    md.append("")
    md.append(f"**City:** {result['city']}")
    md.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    md.append(f"**Feature version:** `{FEATURE_VERSION}`")
    md.append(f"**Training window:** {result['window_start']} → {result['window_end']}  ")
    md.append(f"**Training rows:** {result['n_rows']}  "
              f"(unique dates: {result['n_unique_dates']}, OOF predictions: {result['n_valid_oof']})")
    md.append("")

    md.append(f"## Headline metrics (out-of-fold)")
    md.append("")
    md.append(f"| Metric | Value |")
    md.append(f"|---|---|")
    md.append(f"| Point RMSE | **{result['rmse_c']:.3f} °C** |")
    md.append(f"| Mean bias | {result['mean_bias_c']:+.3f} °C |")
    md.append(f"| Fallback σ | {result['fallback_sigma_c']:.3f} °C |")
    md.append("")

    md.append(f"## Comparison vs naive baselines")
    md.append("")
    md.append(f"All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.")
    md.append("")
    md.append(f"| Method | RMSE | Δ vs ML | Verdict |")
    md.append(f"|---|---|---|---|")
    md.append(f"| **ML model (this)** | **{result['rmse_c']:.3f}** | — | — |")
    for name, val in result["baselines"].items():
        if val is None:
            md.append(f"| {name} | (n/a) | — | — |")
        else:
            delta = val - result["rmse_c"]
            verdict = "ML wins" if delta > 0 else "ML loses"
            md.append(f"| {name} | {val:.3f} | {delta:+.3f} | {verdict} |")
    md.append("")
    md.append(f"_(positive Δ means ML beats that baseline)_")
    md.append("")

    md.append(f"## Calibration — do residuals respect the predicted σ?")
    md.append("")
    md.append(f"σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.")
    md.append("")
    cal = result["calibration"]
    md.append(f"| Window | Empirical | Expected | Verdict |")
    md.append(f"|---|---|---|---|")
    for label, key, expected in [
        ("±1σ", "frac_within_1sigma", 0.683),
        ("±2σ", "frac_within_2sigma", 0.954),
        ("±3σ", "frac_within_3sigma", 0.997),
    ]:
        emp = cal[key]
        diff = emp - expected
        v = "well-calibrated" if abs(diff) < 0.05 else (
            "over-confident (bands too tight)" if emp < expected
            else "under-confident (bands too wide)"
        )
        md.append(f"| {label} | {emp*100:.1f}% | {expected*100:.1f}% | {v} |")
    md.append("")

    md.append(f"## Per-month performance")
    md.append("")
    md.append(f"Months with fewer than 10 OOF samples are omitted.")
    md.append("")
    md.append(f"| Month | n | RMSE °C | σ °C | Bias °C |")
    md.append(f"|---|---|---|---|---|")
    for m in sorted(result["per_month"]):
        s = result["per_month"][m]
        md.append(f"| {m:02d} | {s['n']} | {s['rmse']:.2f} | {s['sigma']:.2f} | {s['bias']:+.2f} |")
    md.append("")

    md.append(f"## Per-decision-hour performance")
    md.append("")
    md.append(f"Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.")
    md.append("")
    md.append(f"| Hour | n | RMSE °C | Bias °C |")
    md.append(f"|---|---|---|---|")
    for h in sorted(result["per_hour"]):
        s = result["per_hour"][h]
        md.append(f"| {h:02d}:00 | {s['n']} | {s['rmse']:.3f} | {s['bias']:+.3f} |")
    md.append("")

    md.append(f"## σ per month (used at inference time)")
    md.append("")
    md.append(f"| Month | σ °C |")
    md.append(f"|---|---|")
    for m in sorted(result["spread_per_bucket"]):
        md.append(f"| {m:02d} | {result['spread_per_bucket'][m]:.2f} |")
    md.append(f"| _fallback_ | _{result['fallback_sigma_c']:.2f}_ |")
    md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--city", action="append", default=None,
                    help="city to evaluate (repeatable); default: all with enough rows")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=400)
    args = ap.parse_args()

    init_db()

    if args.city:
        cities = args.city
    else:
        cities = []
        for c in get_ml_backfill_cities():
            n = count_ml_training_rows(city=c["city"], feature_version=FEATURE_VERSION)
            if n >= args.min_rows:
                cities.append(c["city"])
        if not cities:
            log.warning(f"no cities have ≥{args.min_rows} rows in ml_training_rows")
            return 1

    today = date.today().isoformat()
    summary: list[dict] = []
    for city in cities:
        try:
            r = evaluate_city(city, n_folds=args.folds)
        except Exception as e:
            log.exception(f"[{city}] eval failed: {e}")
            r = {"city": city, "status": "error", "error": str(e)}
        if r["status"] != "ok":
            log.info(f"[{city}] skip: {r['status']}")
            summary.append(r)
            continue
        out_path = DIAG_DIR / f"{city}_{today}_report.md"
        write_report(r, out_path)
        log.info(
            f"[{city}] RMSE={r['rmse_c']:.3f}C "
            f"bias={r['mean_bias_c']:+.3f} "
            f"sigma_fb={r['fallback_sigma_c']:.3f}  -> {out_path}"
        )
        summary.append(r)

    log.info("=" * 70)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 70)
    for r in summary:
        if r["status"] != "ok":
            log.info(f"  [{r['city']:20s}] SKIP {r['status']}")
            continue
        ml = r["rmse_c"]
        bl = r["baselines"]
        beats = sum(1 for v in bl.values() if v is not None and v > ml)
        total = sum(1 for v in bl.values() if v is not None)
        log.info(
            f"  [{r['city']:20s}] OK   ML_RMSE={ml:.3f}C "
            f"beats {beats}/{total} baselines"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
