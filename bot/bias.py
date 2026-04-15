"""
bias.py — Bias diagnostic helpers.

Historical notes
----------------
This module used to run an ERA5-reanalysis-based recorder that compared
resolved-contract ECMWF forecasts against Open-Meteo ERA5 actuals and
inserted rows into `forecast_errors`.  That path has been replaced by the
VC + Open-Meteo Previous Runs pipeline in bias_updater.run_bias_update,
which runs daily and covers both ECMWF and GFS for every city.

The only thing retained here is `log_bias_summary`, a read-only diagnostic
that prints the current aggregate biases per location.  It's safe to keep
and is called occasionally from the CLI / ad-hoc scripts.
"""

import logging

from db import _get_conn

logger = logging.getLogger(__name__)


def log_bias_summary() -> None:
    """
    Log the current bias correction values for all city/location pairs
    that have enough observations.  Useful for monitoring model drift.
    """
    sql = """
        SELECT city, lat_key, lon_key,
               COALESCE(model, 'ecmwf') AS model,
               COUNT(*)                 AS n,
               AVG(error_c)             AS mean_error,
               MIN(target_date)         AS earliest,
               MAX(target_date)         AS latest
        FROM forecast_errors
        GROUP BY city, lat_key, lon_key, COALESCE(model, 'ecmwf')
        HAVING COUNT(*) >= 10
        ORDER BY ABS(AVG(error_c)) DESC
        LIMIT 40
    """
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        logger.info("Bias summary: no locations with 10+ observations yet")
        return

    logger.info("=== Bias summary (actual - forecast), top cities by |mean| ===")
    for r in rows:
        logger.info(
            f"  {r['city'] or '?':20s} {r['model']:14s} n={r['n']:4d} | "
            f"mean_error={r['mean_error']:+.2f}C | "
            f"{r['earliest']} to {r['latest']}"
        )
