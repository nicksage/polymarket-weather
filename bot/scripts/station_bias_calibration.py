"""
station_bias_calibration.py — Compute per-station systematic forecast bias
from historical data, persisted to data/station_bias.json.

For each settlement station:
  - Pull historical NWS forecasts (saved by intraday scans in
    paper_predictor_signals.forecast_high_c)
  - Pull historical actual highs (from station_obs.db, the day-end max)
  - Compute mean forecast error (forecast - actual)
    - Positive = forecast runs HOT (overpredicts)
    - Negative = forecast runs COLD (underpredicts)
  - Save mean offset + std error per station

The intraday predictor will read this file and apply per-station bias
correction:  corrected_forecast = forecast - station_bias

Run periodically (e.g., weekly cron) to keep bias estimates fresh as
seasons change.

Usage:
    cd bot
    python -m scripts.station_bias_calibration --days 30
    python -m scripts.station_bias_calibration --days 14 --min-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import statistics
import sys
from datetime import date, datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

# Load .env via python-dotenv before importing config
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_BOT_DIR), ".env"), override=True)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("bias_cal")

BIAS_OUT = os.path.join(_BOT_DIR, "data", "station_bias.json")


def collect_forecast_history(db_path: str, days: int) -> dict[str, list[dict]]:
    """For each station, return list of {date, forecast_high_c} from the
    LAST scan of each day in the lookback window."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""
            SELECT settlement_station AS station,
                   event_date,
                   forecast_high_c,
                   scanned_at_utc
            FROM paper_predictor_signals
            WHERE event_date >= ?
              AND forecast_high_c IS NOT NULL
              AND settlement_station IS NOT NULL
            ORDER BY station, event_date, scanned_at_utc DESC
        """, (cutoff,)).fetchall()

    # Pick the LAST scan per (station, date) — that's our final forecast
    # estimate for that day before observations took over.
    by_station_date: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["station"], r["event_date"])
        if key not in by_station_date:
            by_station_date[key] = {
                "date": r["event_date"],
                "forecast_high_c": r["forecast_high_c"],
            }
    by_station: dict[str, list[dict]] = {}
    for (station, _date), payload in by_station_date.items():
        by_station.setdefault(station, []).append(payload)
    return by_station


def collect_observed_history(db_path: str, days: int) -> dict[str, dict[str, float]]:
    """For each station, return {date: observed_max_c} for days in lookback."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        # Take the maximum observed temp per (station, date) across all scans
        try:
            rows = c.execute("""
                SELECT settlement_station AS station,
                       event_date,
                       MAX(observed_max_c) AS observed_max
                FROM paper_predictor_signals
                WHERE event_date >= ?
                  AND observed_max_c IS NOT NULL
                  AND observed_max_c > -100
                  AND settlement_station IS NOT NULL
                GROUP BY settlement_station, event_date
            """, (cutoff,)).fetchall()
        except sqlite3.OperationalError:
            return {}

    by_station: dict[str, dict[str, float]] = {}
    for r in rows:
        by_station.setdefault(r["station"], {})[r["event_date"]] = r["observed_max"]
    return by_station


def compute_bias(forecasts: list[dict], observed: dict[str, float],
                  min_samples: int) -> dict | None:
    """Mean(forecast - actual) across matched days.  Positive = forecast hot."""
    pairs = []
    for f in forecasts:
        if f["date"] in observed:
            pairs.append((f["forecast_high_c"], observed[f["date"]]))
    if len(pairs) < min_samples:
        return None
    errors = [fc - obs for fc, obs in pairs]
    return {
        "n_samples":    len(pairs),
        "mean_bias_c":  round(statistics.fmean(errors), 3),
        "median_bias_c": round(statistics.median(errors), 3),
        "stdev_c":      round(statistics.stdev(errors) if len(errors) > 1 else 0, 3),
        "first_date":   min(f["date"] for f in forecasts if f["date"] in observed),
        "last_date":    max(f["date"] for f in forecasts if f["date"] in observed),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db",          default=None, help="signals DB path")
    p.add_argument("--days",        type=int, default=30,
                    help="lookback window in days (default 30)")
    p.add_argument("--min-samples", type=int, default=5,
                    help="minimum days of paired data to compute bias (default 5)")
    p.add_argument("--out",         default=BIAS_OUT, help="output JSON path")
    args = p.parse_args()

    from config import DB_PATH
    db_path = args.db or DB_PATH

    log.info(f"reading paper_predictor_signals from {db_path}")
    log.info(f"lookback: {args.days} days, min_samples: {args.min_samples}")

    forecast_hist = collect_forecast_history(db_path, args.days)
    observed_hist = collect_observed_history(db_path, args.days)
    log.info(f"found forecast data for {len(forecast_hist)} stations, "
              f"observed data for {len(observed_hist)} stations")

    biases = {}
    for station, fcsts in sorted(forecast_hist.items()):
        obs = observed_hist.get(station, {})
        b = compute_bias(fcsts, obs, args.min_samples)
        if b is None:
            log.info(f"  {station:6s}: insufficient data ({len(fcsts)} forecasts, "
                      f"{len(obs)} observed) — skipping")
            continue
        biases[station] = b
        sign = "HOT " if b["mean_bias_c"] > 0 else "COOL"
        log.info(f"  {station:6s}: {sign} mean_bias={b['mean_bias_c']:+.2f}°C "
                  f"({b['mean_bias_c']*9/5:+.2f}°F)  ±{b['stdev_c']:.2f}°C  "
                  f"n={b['n_samples']}")

    payload = {
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "lookback_days": args.days,
        "min_samples":   args.min_samples,
        "by_station":    biases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info(f"wrote {len(biases)} station biases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())