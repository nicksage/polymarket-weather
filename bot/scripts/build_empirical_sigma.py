"""
build_empirical_sigma.py — Derive σ per (model, lead_days_bucket) from
forecast_errors and write to empirical_sigma_by_lead (backtest DB).

σ is defined as the RMSE of forecast errors:
    rmse = sqrt( mean( (actual - forecast)^2 ) )

This gives us the forecast's realized 1-sigma uncertainty — the empirical
replacement for the live ensemble spread we can't reconstruct for past dates.

Model name normalization:
    forecast_errors.model           backtest canonical name
    ─────────────────────────────   ───────────────────────
    ecmwf / ecmwf_ifs025 / ecmwf_ifs04   → 'ecmwf'
    gfs_global / gfs025                  → 'gfs'

Usage:
    python bot/scripts/build_empirical_sigma.py
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import sys

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from backtest.backtest_db import init_backtest_db, upsert_empirical_sigma
from config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("empirical_sigma")


def _normalize_model(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.lower()
    if "ecmwf" in r:
        return "ecmwf"
    if "gfs" in r:
        return "gfs"
    return r


def main() -> int:
    init_backtest_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Group rows by (canonical_model, lead_days_bucket) and compute RMSE
    rows = conn.execute("""
        SELECT model, lead_days_bucket, error_c
        FROM forecast_errors
        WHERE error_c IS NOT NULL
    """).fetchall()

    groups: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        canonical = _normalize_model(r["model"])
        if canonical is None:
            continue
        lead = r["lead_days_bucket"]
        if lead is None:
            lead = -1   # sentinel for rows lacking a lead bucket
        key = (canonical, int(lead))
        groups.setdefault(key, []).append(float(r["error_c"]))

    if not groups:
        log.error("No usable rows in forecast_errors")
        return 1

    log.info(f"Computing σ for {len(groups)} (model, lead) groups...")
    for (model, lead), errors in sorted(groups.items()):
        n = len(errors)
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e * e for e in errors) / n)
        upsert_empirical_sigma(
            model=model, lead_days_bucket=lead,
            n_obs=n, rmse_c=rmse, mae_c=mae,
        )
        log.info(
            f"  model={model:<6} lead={lead:<3} n={n:<6} "
            f"rmse(σ)={rmse:.3f}°C  mae={mae:.3f}°C"
        )

    # Verify
    from backtest.backtest_db import _get_conn
    with _get_conn() as c:
        total = c.execute("SELECT COUNT(*) FROM empirical_sigma_by_lead").fetchone()[0]
    log.info(f"empirical_sigma_by_lead rows: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
