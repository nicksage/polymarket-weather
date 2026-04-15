"""
bias.py — ECMWF forecast bias correction

After each Polymarket temperature contract resolves, we know two things:
    1. What ECMWF predicted (forecast_mu_c, stored in temp_events)
    2. What actually happened (actual_tmax_c, fetched from ERA5 once available)

This module:
    record_resolved_contracts() — run periodically to check if any past contracts
        have resolved and ERA5 data is now available; writes rows to forecast_errors

    apply_bias_corrections() — called during the trading run to log current bias
        stats (actual correction is applied inside weather.py via get_bias_correction)

ERA5 data becomes available ~5 days after the event date, so we check for
resolvable contracts with target_date <= today - 5 days.
"""

import logging
from datetime import date, timedelta, datetime, timezone

import httpx

from db import (
    _get_conn, _loc_keys,
    insert_forecast_error, get_bias_correction,
    get_era5_values, insert_era5_rows,
)

logger = logging.getLogger(__name__)

ERA5_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_single_era5_tmax(lat: float, lon: float, date_str: str) -> float | None:
    """Fetch a single day's ERA5 tmax from Open-Meteo Archive API."""
    try:
        resp = httpx.get(
            ERA5_ARCHIVE,
            params={
                "latitude":   lat,
                "longitude":  lon,
                "start_date": date_str,
                "end_date":   date_str,
                "daily":      "temperature_2m_max",
                "timezone":   "UTC",
            },
            timeout=15,
        )
        resp.raise_for_status()
        temps = resp.json().get("daily", {}).get("temperature_2m_max", [])
        if temps and temps[0] is not None:
            actual = float(temps[0])
            logger.info(
                f"ERA5 Archive API call for ({lat:.2f}, {lon:.2f}) on {date_str} "
                f"was successful — actual tmax: {actual:.1f}°C"
            )
            return actual
        logger.info(
            f"ERA5 Archive API call for ({lat:.2f}, {lon:.2f}) on {date_str} "
            f"was successful — no data available for this date"
        )
        return None
    except Exception as e:
        logger.debug(f"ERA5 single fetch failed ({lat},{lon} {date_str}): {e}")
        return None


def _get_unrecorded_resolved_events() -> list[dict]:
    """
    Find temp_events rows where:
      - target date is at least 5 days in the past (ERA5 now available)
      - no forecast_error row yet exists for this (lat, lon, date)
    Returns list of dicts with: city, date, lat, lon, forecast_mu_c
    """
    cutoff = (date.today() - timedelta(days=5)).isoformat()
    sql = """
        SELECT DISTINCT
            e.city,
            e.date,
            e.lat,
            e.lon,
            e.forecast_mu_c
        FROM temp_events e
        WHERE e.date <= ?
          AND e.forecast_mu_c IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM forecast_errors fe
              WHERE fe.lat_key = ROUND(e.lat, 2)
                AND fe.lon_key = ROUND(e.lon, 2)
                AND fe.target_date = e.date
          )
        ORDER BY e.date DESC
        LIMIT 100
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, (cutoff,)).fetchall()
        return [dict(r) for r in rows]


def record_resolved_contracts() -> int:
    """
    Check for resolved contracts whose ERA5 actual temperature is now available,
    compare against the stored ECMWF forecast, and write forecast_errors rows.

    Should be called once per trading run (or on a separate daily schedule).
    Returns the number of new error records written.
    """
    candidates = _get_unrecorded_resolved_events()
    if not candidates:
        logger.debug("Bias recorder: no unrecorded resolved events found")
        return 0

    logger.info(f"Bias recorder: checking {len(candidates)} resolved events for ERA5 actuals")
    written = 0

    for ev in candidates:
        city         = ev["city"] or ""
        date_str     = ev["date"]
        lat          = ev["lat"]
        lon          = ev["lon"]
        forecast_mu  = ev["forecast_mu_c"]

        # Check DB cache first
        cached = get_era5_values(lat, lon, [date_str])
        if date_str in cached:
            actual_tmax = cached[date_str]
        else:
            # Fetch from API and cache
            actual_tmax = _fetch_single_era5_tmax(lat, lon, date_str)
            if actual_tmax is not None:
                insert_era5_rows(lat, lon, {date_str: actual_tmax})

        if actual_tmax is None:
            logger.debug(f"ERA5 not yet available for {city} {date_str}")
            continue

        # Compute days_ahead from the scan that generated this forecast
        # (approximate — use 0 as a safe default if unavailable)
        days_ahead = 0
        try:
            days_ahead_sql = """
                SELECT days_ahead FROM temp_events
                WHERE ROUND(lat,2) = ? AND ROUND(lon,2) = ? AND date = ?
                ORDER BY scan_timestamp ASC LIMIT 1
            """
            lat_key, lon_key = _loc_keys(lat, lon)
            with _get_conn() as conn:
                row = conn.execute(days_ahead_sql, (lat_key, lon_key, date_str)).fetchone()
                if row and row[0] is not None:
                    days_ahead = int(row[0])
        except Exception:
            pass

        insert_forecast_error(
            lat=lat, lon=lon, city=city,
            target_date=date_str, days_ahead=days_ahead,
            forecast_mu_c=forecast_mu, actual_tmax_c=actual_tmax,
        )
        error_c = actual_tmax - forecast_mu
        logger.info(
            f"Bias recorded: {city} {date_str} | "
            f"ECMWF={forecast_mu:.1f}°C ERA5={actual_tmax:.1f}°C "
            f"error={error_c:+.2f}°C"
        )
        written += 1

    logger.info(f"Bias recorder: wrote {written} new error records")
    return written


def log_bias_summary() -> None:
    """
    Log the current bias correction values for all city/location pairs
    that have enough observations.  Useful for monitoring model drift.
    """
    sql = """
        SELECT city, lat_key, lon_key,
               COUNT(*) as n,
               AVG(error_c) as mean_error,
               MIN(target_date) as earliest,
               MAX(target_date) as latest
        FROM forecast_errors
        GROUP BY city, lat_key, lon_key
        HAVING COUNT(*) >= 3
        ORDER BY ABS(AVG(error_c)) DESC
        LIMIT 30
    """
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        logger.info("Bias summary: no locations with 3+ observations yet")
        return

    logger.info("=== ECMWF Bias Summary (actual - forecast) ===")
    for r in rows:
        logger.info(
            f"  {r['city'] or '?':20s} n={r['n']:3d} | "
            f"mean_error={r['mean_error']:+.2f}°C | "
            f"{r['earliest']} to {r['latest']}"
        )
