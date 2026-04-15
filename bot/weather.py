"""
weather.py — Temperature distribution modeling engine

Estimates a probability distribution N(μ, σ) for the daily maximum temperature
at a given location and date, then computes P(temp in [low, high]) for each
Polymarket outcome range using the normal CDF.

Public API used by edge.py:
    get_temp_distribution_for_event(lat, lon, date_str) -> dist_dict | None
    get_temp_range_probability(mu_c, sigma_c, range_low, range_high, unit) -> float

Sources (highest to lowest weight in the final blend):
    1. ECMWF IFS ensemble  — 51 members, 15-day horizon  (primary)
    2. GFS ensemble        — 31 members, 35-day horizon  (secondary)
    3. ERA5 climatology    — 10-year historical baseline via Open-Meteo Archive API
    4. NWS / NOAA NBM      — US-only, NWS hourly forecast + MDL NBM text quantiles
    5. Tomorrow.io         — point estimate cross-check only

Bayesian blending schedule (forecast weight α):
    Days 0–7:  α = 1.0  (pure forecast)
    Days 8–15: α decreases linearly to 0.40
    Days >15:  event skipped (MAX_FORECAST_DAYS guard in config)

Caching: distributions are cached by (round(lat,3), round(lon,3), date_str)
for the duration of one scan cycle.  Call clear_forecast_cache() between scans.
"""

import os
import re
import logging
import statistics
from datetime import date, datetime, timedelta
from math import sqrt
from tenacity import retry, stop_after_attempt, wait_exponential

import httpx
from scipy.stats import norm as scipy_norm

import numpy as np
from scipy.stats import gaussian_kde

from config import (
    ECMWF_WEIGHT, GFS_WEIGHT,
    BLEND_START_DAYS, MAX_FORECAST_DAYS,
    CLIM_LOOKBACK_YEARS, CLIM_WINDOW_DAYS,
    MIN_FORECAST_SIGMA_C, MAX_FORECAST_SIGMA_C,
    USE_KDE_CLIM, MIN_BIAS_OBSERVATIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scan-cycle cache
# ---------------------------------------------------------------------------
_dist_cache: dict[tuple, dict | None] = {}


def clear_forecast_cache() -> None:
    _dist_cache.clear()
    logger.debug("Forecast distribution cache cleared")


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9

def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32

def _to_celsius(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit == "fahrenheit":
        return _f_to_c(value)
    return float(value)


# ---------------------------------------------------------------------------
# NWS US-only guard
# ---------------------------------------------------------------------------
_US_LAT = (15.0, 72.0)
_US_LON = (-180.0, -60.0)

def _is_us_coordinate(lat: float, lon: float) -> bool:
    return _US_LAT[0] <= lat <= _US_LAT[1] and _US_LON[0] <= lon <= _US_LON[1]


# ---------------------------------------------------------------------------
# Gaussian parameter helpers
# ---------------------------------------------------------------------------

def _fit_normal(values: list[float]) -> tuple[float, float] | None:
    """Fit a normal distribution to a list of values. Returns (mu, sigma)."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], 2.0)
    mu    = sum(values) / len(values)
    sigma = statistics.stdev(values)
    return (mu, max(sigma, MIN_FORECAST_SIGMA_C))


def _blend_gaussians(
    mu1: float, sigma1: float, w1: float,
    mu2: float, sigma2: float, w2: float,
) -> tuple[float, float]:
    """
    Mixture-of-Gaussians blend.  Returns the mixture mean and a conservative
    sigma that accounts for both within-component uncertainty and the spread
    between component means.
    """
    total = w1 + w2
    if total == 0:
        return (mu1, sigma1)
    a1, a2 = w1 / total, w2 / total
    mu_mix = a1 * mu1 + a2 * mu2
    # Var(mix) = a1*(sigma1² + (mu1-mu_mix)²) + a2*(sigma2² + (mu2-mu_mix)²)
    var_mix = (
        a1 * (sigma1 ** 2 + (mu1 - mu_mix) ** 2)
        + a2 * (sigma2 ** 2 + (mu2 - mu_mix) ** 2)
    )
    return (mu_mix, max(sqrt(var_mix), MIN_FORECAST_SIGMA_C))


def _forecast_alpha(days_ahead: int) -> float:
    """
    Forecast weight (0–1) for the Bayesian blend with climatology.
    1.0 for near-term, decreasing linearly past BLEND_START_DAYS.
    """
    if days_ahead <= BLEND_START_DAYS:
        return 1.0
    excess = days_ahead - BLEND_START_DAYS
    return max(0.40, 1.0 - excess * 0.08)


# ---------------------------------------------------------------------------
# Source 1: ECMWF IFS ensemble (51 members) via Open-Meteo Ensemble API
# ---------------------------------------------------------------------------
OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _get_ecmwf_ensemble_distribution(
    lat: float, lon: float, date_str: str
) -> dict | None:
    """
    Fetch 51-member ECMWF IFS04 daily max temperature ensemble, fit N(μ,σ).
    Returns {"mu_c": float, "sigma_c": float, "source": "ecmwf", "n": int} or None.
    """
    try:
        resp = httpx.get(
            OPENMETEO_ENSEMBLE,
            params={
                "latitude":   lat,
                "longitude":  lon,
                "daily":      "temperature_2m_max",
                "start_date": date_str,
                "end_date":   date_str,
                "models":     "ecmwf_ifs04",
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(
            f"ECMWF ensemble API call for ({lat:.2f}, {lon:.2f}) on {date_str} was successful"
        )
        daily = resp.json().get("daily", {})
    except httpx.HTTPError as e:
        logger.debug(f"ECMWF ensemble request failed: {e}")
        return None

    # Collect all member values for index 0 (single target day)
    prefix = "temperature_2m_max_member"
    values = []
    for key, arr in daily.items():
        if key.startswith(prefix) and isinstance(arr, list) and arr and arr[0] is not None:
            values.append(float(arr[0]))

    if not values:
        logger.debug("ECMWF ensemble returned no member values (too far out?)")
        return None

    fit = _fit_normal(values)
    if fit is None:
        return None
    mu, sigma = fit
    logger.debug(f"ECMWF ensemble: n={len(values)} mu={mu:.2f}°C sd={sigma:.2f}°C")
    return {"mu_c": mu, "sigma_c": sigma, "source": "ecmwf", "n": len(values)}


# ---------------------------------------------------------------------------
# Source 2: GFS ensemble (31 members) via Open-Meteo Ensemble API
# ---------------------------------------------------------------------------

def _get_gfs_ensemble_distribution(
    lat: float, lon: float, date_str: str
) -> dict | None:
    """
    Fetch 31-member GFS025 daily max temperature ensemble, fit N(μ,σ).
    GFS extends to 35 days which helps cross-validate ECMWF at 8–15 days.
    """
    try:
        resp = httpx.get(
            OPENMETEO_ENSEMBLE,
            params={
                "latitude":   lat,
                "longitude":  lon,
                "daily":      "temperature_2m_max",
                "start_date": date_str,
                "end_date":   date_str,
                "models":     "gfs025",
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(
            f"GFS ensemble API call for ({lat:.2f}, {lon:.2f}) on {date_str} was successful"
        )
        daily = resp.json().get("daily", {})
    except httpx.HTTPError as e:
        logger.debug(f"GFS ensemble request failed: {e}")
        return None

    prefix = "temperature_2m_max_member"
    values = []
    for key, arr in daily.items():
        if key.startswith(prefix) and isinstance(arr, list) and arr and arr[0] is not None:
            values.append(float(arr[0]))

    if not values:
        return None

    fit = _fit_normal(values)
    if fit is None:
        return None
    mu, sigma = fit
    logger.debug(f"GFS ensemble: n={len(values)} mu={mu:.2f}°C sd={sigma:.2f}°C")
    return {"mu_c": mu, "sigma_c": sigma, "source": "gfs", "n": len(values)}


# ---------------------------------------------------------------------------
# Source 3: ERA5 climatological baseline — DB-cached, trend-corrected
# ---------------------------------------------------------------------------
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _build_era5_date_list(target_date_str: str) -> list[str]:
    """
    Build the full list of historical dates needed for the climatological prior:
    CLIM_LOOKBACK_YEARS years × ±CLIM_WINDOW_DAYS centred on the target
    calendar date.  Clamps to the ERA5 availability cutoff (~5 days ago).
    """
    target   = date.fromisoformat(target_date_str)
    cutoff   = (date.today() - timedelta(days=5)).isoformat()
    cur_year = date.today().year
    needed: list[str] = []

    for yr in range(cur_year - CLIM_LOOKBACK_YEARS, cur_year):
        try:
            anchor = date(yr, target.month, target.day)
        except ValueError:
            anchor = date(yr, target.month, min(target.day, 28))

        for delta in range(-CLIM_WINDOW_DAYS, CLIM_WINDOW_DAYS + 1):
            d = (anchor + timedelta(days=delta)).isoformat()
            if d <= cutoff:
                needed.append(d)

    return needed


def _fetch_era5_from_api(lat: float, lon: float, dates_needed: list[str]) -> dict[str, float]:
    """
    Fetch missing ERA5 dates from the Open-Meteo Archive API.

    Batches consecutive dates into year-sized requests to minimise API calls
    (one call per year-block rather than one call per date).

    Returns {date_str: tmax_c} for all successfully fetched dates.
    """
    if not dates_needed:
        return {}

    # Group into contiguous year-blocks so we can batch by year
    from itertools import groupby
    dates_needed_sorted = sorted(dates_needed)
    year_groups: dict[int, list[str]] = {}
    for d in dates_needed_sorted:
        yr = int(d[:4])
        year_groups.setdefault(yr, []).append(d)

    result: dict[str, float] = {}
    cutoff = (date.today() - timedelta(days=5)).isoformat()

    for yr, yr_dates in year_groups.items():
        start = min(yr_dates)
        end   = min(max(yr_dates), cutoff)
        if start > end:
            continue
        try:
            resp = httpx.get(
                OPENMETEO_ARCHIVE,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "start_date": start,
                    "end_date":   end,
                    "daily":      "temperature_2m_max",
                    "timezone":   "UTC",
                },
                timeout=20,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            date_list = daily.get("time", [])
            temp_list = daily.get("temperature_2m_max", [])
            yr_set    = set(yr_dates)
            year_count = 0
            for d, v in zip(date_list, temp_list):
                if v is not None and d in yr_set:
                    result[d] = float(v)
                    year_count += 1
            logger.info(
                f"ERA5 Archive API call for ({lat:.2f}, {lon:.2f}) year {yr} "
                f"was successful — {year_count} temperature values retrieved"
            )
        except Exception as e:
            logger.debug(f"ERA5 fetch failed for year {yr}: {e}")

    return result


def _apply_trend_correction(
    values: list[float], years: list[int], target_year: int
) -> tuple[list[float], float]:
    """
    Remove a linear warming trend from historical values and project the
    trend to the target year.

    Returns (detrended_values, trend_adjustment_c) where trend_adjustment_c
    is the correction to add back to bring the mean to the target year level.
    """
    if len(values) < 5:
        return values, 0.0
    arr_y = np.array(years, dtype=float)
    arr_v = np.array(values, dtype=float)
    slope, intercept = np.polyfit(arr_y, arr_v, 1)
    mean_year = arr_y.mean()
    # Detrended residuals: subtract only the trend component relative to the
    # mean year, preserving the original temperature scale.
    # Correct formula: detrended = v - slope*(y - mean_year)
    # Bug was: detrended = v - slope*y  which subtracts ~slope*2020 ≈ ±40°C
    detrended = arr_v - slope * (arr_y - mean_year)
    # Project trend from mean year to target year
    trend_adj = slope * (target_year - mean_year)
    logger.debug(
        f"ERA5 trend correction: slope={slope*10:.3f}°C/decade "
        f"trend_adj={trend_adj:+.2f}°C to year {target_year}"
    )
    return detrended.tolist(), trend_adj


def _get_era5_clim_distribution(
    lat: float, lon: float, target_date_str: str
) -> dict | None:
    """
    Build a climatological prior from ERA5 reanalysis data.

    Strategy:
      1. Compute the full list of historical dates needed
      2. Query era5_daily DB for already-stored dates (cache hit)
      3. Fetch only missing dates from Open-Meteo Archive API
      4. Write newly fetched rows to the DB
      5. Apply linear warming-trend correction to the raw values
      6. Fit either a normal distribution or KDE (controlled by USE_KDE_CLIM)

    Returns {"mu_c", "sigma_c", "source": "era5", "n", "kde": optional} or None.
    """
    from db import get_era5_dates_present, get_era5_values, insert_era5_rows

    needed_dates = _build_era5_date_list(target_date_str)
    if not needed_dates:
        return None

    # Step 1: DB lookup
    present = get_era5_dates_present(lat, lon, needed_dates)
    missing = [d for d in needed_dates if d not in present]

    fetched_new = 0
    if missing:
        logger.debug(
            f"ERA5 cache: {len(present)} hits, {len(missing)} misses "
            f"for ({lat:.2f},{lon:.2f}) — fetching from API"
        )
        new_data = _fetch_era5_from_api(lat, lon, missing)
        if new_data:
            insert_era5_rows(lat, lon, new_data)
            fetched_new = len(new_data)
    else:
        logger.debug(
            f"ERA5 cache: full hit ({len(present)} dates) for ({lat:.2f},{lon:.2f})"
        )

    # Step 2: Load all values from DB
    all_data = get_era5_values(lat, lon, needed_dates)
    if len(all_data) < 5:
        logger.debug(f"ERA5 clim: insufficient data ({len(all_data)} values)")
        return None

    # Build parallel (value, year) lists for trend correction
    values = []
    years  = []
    for d, v in sorted(all_data.items()):
        values.append(v)
        years.append(int(d[:4]))

    target_year = date.today().year
    detrended, trend_adj = _apply_trend_correction(values, years, target_year)

    # Trend-adjusted mean: mean of detrended values + forward-projected trend
    trend_adj_mean = (sum(detrended) / len(detrended)) + trend_adj

    if USE_KDE_CLIM and len(detrended) >= 10:
        # KDE path — store kernel for use in get_temp_range_probability_kde()
        kde = gaussian_kde(detrended, bw_method="scott")
        # Compute effective mu/sigma from KDE for logging and normal-path fallback
        mu_c    = float(trend_adj_mean)
        sigma_c = float(statistics.stdev(detrended))
        sigma_c = max(sigma_c, MIN_FORECAST_SIGMA_C)
        logger.debug(
            f"ERA5 clim (KDE): n={len(detrended)} mu={mu_c:.2f}°C sd={sigma_c:.2f}°C "
            f"trend_adj={trend_adj:+.2f}°C new_fetched={fetched_new}"
        )
        return {
            "mu_c": mu_c, "sigma_c": sigma_c,
            "source": "era5", "n": len(detrended),
            "kde": kde, "trend_adj": trend_adj,
        }
    else:
        # Normal distribution path
        sigma_c = statistics.stdev(detrended) if len(detrended) > 1 else 3.0
        sigma_c = max(sigma_c, MIN_FORECAST_SIGMA_C)
        mu_c    = float(trend_adj_mean)
        logger.debug(
            f"ERA5 clim (normal): n={len(detrended)} mu={mu_c:.2f}°C sd={sigma_c:.2f}°C "
            f"trend_adj={trend_adj:+.2f}°C new_fetched={fetched_new}"
        )
        return {
            "mu_c": mu_c, "sigma_c": sigma_c,
            "source": "era5", "n": len(detrended),
            "kde": None, "trend_adj": trend_adj,
        }


# ---------------------------------------------------------------------------
# Source 4: NWS / NOAA NBM  (US-only)
# ---------------------------------------------------------------------------
NWS_BASE    = "https://api.weather.gov"
NWS_MDL_BASE = "https://www.weather.gov/mdl"
NWS_HEADERS = {"User-Agent": "WeatherArbBot/1.0 (weather-arb@example.com)"}


def _get_nws_observation_station(lat: float, lon: float) -> str | None:
    """Return the nearest NWS/ASOS observation station ICAO identifier."""
    try:
        points = httpx.get(
            f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
            headers=NWS_HEADERS, timeout=10
        )
        points.raise_for_status()
        logger.info(f"NWS Points API call for ({lat:.4f}, {lon:.4f}) was successful")
        obs_url = points.json().get("properties", {}).get("observationStations")
        if not obs_url:
            return None
        stations = httpx.get(obs_url, headers=NWS_HEADERS, timeout=10)
        stations.raise_for_status()
        features = stations.json().get("features", [])
        if features:
            station_id = features[0]["properties"].get("stationIdentifier", "unknown")
            logger.info(
                f"NWS Observation Stations API call was successful — "
                f"nearest station: {station_id}"
            )
            return station_id
        return None
    except Exception as e:
        logger.debug(f"NWS station lookup failed: {e}")
        return None


def _get_noaa_nbm_distribution(lat: float, lon: float, date_str: str) -> dict | None:
    """
    Fetch NOAA NBM MaxT quantile forecast via the MDL NBM text product.

    The MDL endpoint provides Q10/Q25/Q50/Q75/Q90 temperature quantiles for
    the nearest ASOS/AWOS station.  From five quantiles we recover μ and σ
    under the assumption of approximate normality:
        μ ≈ Q50
        σ ≈ (Q75 - Q25) / (2 * 0.6745)   [IQR ≈ 1.3490 * σ]

    Returns {"mu_c", "sigma_c", "source": "noaa_nbm", "quantiles_f"} or None.
    Only called for US coordinates.
    """
    if not _is_us_coordinate(lat, lon):
        return None

    icao = _get_nws_observation_station(lat, lon)
    if not icao:
        return None

    raw_text = None
    for cyc in ("12", "06", "00", "18"):
        try:
            resp = httpx.get(
                f"{NWS_MDL_BASE}/nbm_text",
                params={"ele": "MaxT", "cyc": cyc, "sta": icao,
                        "type": "txt", "israw": "yes"},
                timeout=12,
            )
            if resp.status_code == 200 and resp.text.strip():
                raw_text = resp.text
                logger.info(
                    f"NOAA NBM text product API call for station {icao} "
                    f"(cycle {cyc}) on {date_str} was successful"
                )
                break
        except Exception:
            continue

    if not raw_text:
        logger.debug(f"NOAA NBM text product unavailable for {icao}")
        return None

    # Parse the target date's quantile row.
    # The text product contains one column per valid time; we look for the
    # row containing five temperature values that are plausible (°F range).
    target = datetime.fromisoformat(date_str)
    lines  = raw_text.splitlines()

    # Identify the date header line(s) and find the column index for our date.
    # MDL products typically have dates in the header as "TUE APR 10" etc.
    col_idx = None
    for line in lines:
        months = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
        if any(m in line.lower() for m in months):
            # Look for the target month + day in this header line
            target_str = target.strftime("%b %d").upper().replace(" 0", "  ")
            if target_str in line.upper():
                parts = line.split()
                for i, p in enumerate(parts):
                    if target.strftime("%d").lstrip("0") in p or target_str in " ".join(parts[max(0,i-1):i+2]).upper():
                        col_idx = i
                        break
                break

    # Extract a row with 5 numeric values in a plausible Fahrenheit range
    quantile_row = None
    for line in lines:
        nums = re.findall(r"-?\d+", line)
        if len(nums) == 5:
            try:
                vals = [int(n) for n in nums]
                if all(-60 <= v <= 150 for v in vals) and vals == sorted(vals):
                    quantile_row = vals
                    break
            except ValueError:
                pass

    if not quantile_row or len(quantile_row) < 5:
        logger.debug(f"NBM: could not parse quantile row for {icao}")
        return None

    q10, q25, q50, q75, q90 = quantile_row
    mu_f    = float(q50)
    # IQR method: sigma ≈ IQR / 1.349
    sigma_f = max((q75 - q25) / 1.349, 1.0)

    mu_c    = _f_to_c(mu_f)
    sigma_c = sigma_f * 5 / 9

    logger.debug(
        f"NOAA NBM [{icao}]: Q10={q10} Q25={q25} Q50={q50} Q75={q75} Q90={q90}°F "
        f"-> mu={mu_c:.2f}°C sd={sigma_c:.2f}°C"
    )
    return {
        "mu_c":       round(mu_c, 2),
        "sigma_c":    round(sigma_c, 2),
        "source":     "noaa_nbm",
        "quantiles_f": {"q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90},
    }


def _get_nws_hourly_distribution(
    lat: float, lon: float, date_str: str
) -> dict | None:
    """
    Fallback NWS source for US cities: derive temperature distribution from
    the NWS hourly gridpoint forecast.

    Fetches all hourly temperatures for the target day, extracts the expected
    maximum, and estimates σ from the daytime temperature spread.
    """
    if not _is_us_coordinate(lat, lon):
        return None

    try:
        pts = httpx.get(
            f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
            headers=NWS_HEADERS, timeout=10,
        )
        pts.raise_for_status()
        logger.info(f"NWS Points API call for ({lat:.4f}, {lon:.4f}) was successful")
        props    = pts.json()["properties"]
        hourly_url = props.get("forecastHourly")
        if not hourly_url:
            return None

        hr = httpx.get(hourly_url, headers=NWS_HEADERS, timeout=12)
        hr.raise_for_status()
        logger.info(
            f"NWS hourly forecast API call for ({lat:.4f}, {lon:.4f}) "
            f"on {date_str} was successful"
        )
        periods = hr.json()["properties"]["periods"]
    except Exception as e:
        logger.debug(f"NWS hourly fetch failed: {e}")
        return None

    target = date.fromisoformat(date_str)
    day_temps_f = []
    for p in periods:
        try:
            start = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
            if start.date() != target:
                continue
            t = p.get("temperature")
            if t is not None:
                day_temps_f.append(float(t))
        except (ValueError, KeyError):
            continue

    if not day_temps_f:
        return None

    # Expected max temperature for the day
    mu_f    = max(day_temps_f)
    # σ from diurnal spread (daytime variance)
    spread  = max(day_temps_f) - min(day_temps_f)
    sigma_f = max(spread / 4.0, 2.0)   # heuristic: spread ≈ 4σ for a bell-shaped day

    mu_c    = _f_to_c(mu_f)
    sigma_c = sigma_f * 5 / 9

    logger.debug(
        f"NWS hourly: {len(day_temps_f)} hourly temps, "
        f"max={mu_f:.1f}°F -> mu={mu_c:.2f}°C sd={sigma_c:.2f}°C"
    )
    return {"mu_c": mu_c, "sigma_c": sigma_c, "source": "nws_hourly"}


# ---------------------------------------------------------------------------
# Source 5: Tomorrow.io — point estimate cross-check
# ---------------------------------------------------------------------------
TOMORROWIO_BASE = "https://api.tomorrow.io/v4/weather/forecast"
TOMORROWIO_KEY  = os.getenv("TOMORROWIO_API_KEY")
_tomorrowio_exhausted = False


def reset_tomorrowio_limit() -> None:
    global _tomorrowio_exhausted
    _tomorrowio_exhausted = False


def _get_tomorrowio_point_estimate(
    lat: float, lon: float, date_str: str
) -> float | None:
    """
    Returns the Tomorrow.io forecast temperatureMax (°C) for the target day,
    or None if unavailable / rate-limited.  Used only as a sanity check.
    """
    global _tomorrowio_exhausted
    if not TOMORROWIO_KEY or _tomorrowio_exhausted:
        return None

    try:
        resp = httpx.get(
            TOMORROWIO_BASE,
            params={
                "location": f"{lat},{lon}",
                "apikey":   TOMORROWIO_KEY,
                "fields":   "temperatureMax",
                "timesteps": "1d",
                "startTime": f"{date_str}T00:00:00Z",
                "endTime":   f"{date_str}T23:59:59Z",
                "units":    "metric",
            },
            timeout=10,
        )
        if resp.status_code == 429:
            _tomorrowio_exhausted = True
            return None
        resp.raise_for_status()
        daily = resp.json().get("timelines", {}).get("daily", [])
        if not daily:
            return None
        temp_max = daily[0].get("values", {}).get("temperatureMax")
        if temp_max is not None:
            logger.info(
                f"Tomorrow.io API call for ({lat:.2f}, {lon:.2f}) on {date_str} "
                f"was successful — temperatureMax: {temp_max:.1f}°C"
            )
        return temp_max
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main distribution function
# ---------------------------------------------------------------------------

def get_temp_distribution_for_event(
    lat: float, lon: float, date_str: str
) -> dict | None:
    """
    Estimate the probability distribution N(μ_c, σ_c) for the daily maximum
    temperature at (lat, lon) on date_str.

    Algorithm:
        1. Fetch ECMWF ensemble → μ₁, σ₁
        2. Fetch GFS ensemble   → μ₂, σ₂
        3. Blend ECMWF + GFS via mixture-of-Gaussians (65%/35% default)
        4. Fetch ERA5 climatological prior → μ_clim, σ_clim
        5. Bayesian-blend forecast with climatology based on days_ahead
        6. For US: fetch NOAA NBM (or NWS hourly) → optionally fine-tune μ
        7. Apply Tomorrow.io as an outlier sanity check (log warning if >5°C off)

    Returns:
        {
            "mu_c":        float,   # final blended mean (°C)
            "sigma_c":     float,   # final blended std dev (°C)
            "clim_mu_c":   float,   # climatological mean (°C)
            "clim_sigma_c":float,   # climatological std dev (°C)
            "days_ahead":  int,
            "sources":     list[str],
            "n_members":   int,     # total ensemble members used
        }
    or None if no usable forecast data.
    """
    cache_key = (round(lat, 3), round(lon, 3), date_str)
    if cache_key in _dist_cache:
        return _dist_cache[cache_key]

    # Days ahead
    try:
        days_ahead = (date.fromisoformat(date_str) - date.today()).days
    except ValueError:
        _dist_cache[cache_key] = None
        return None

    # Ensemble sources
    ecmwf = _get_ecmwf_ensemble_distribution(lat, lon, date_str)
    gfs   = _get_gfs_ensemble_distribution(lat, lon, date_str)

    if ecmwf is None and gfs is None:
        logger.warning(f"No ensemble data for ({lat:.2f},{lon:.2f}) {date_str}")
        _dist_cache[cache_key] = None
        return None

    # ---- Per-model bias correction (applied BEFORE blending) --------------
    # Each model's systematic error is its own — correct ECMWF with the
    # ECMWF-specific bias history, and GFS with GFS-specific history.  These
    # biases come from the forecast_errors table, populated by
    # scripts/backfill_bias_data.py (historical) and bias.py (ongoing).
    from db import get_bias_correction
    calendar_month_day = date_str[5:]   # MM-DD
    ecmwf_bias_c = 0.0; ecmwf_bias_n = 0
    gfs_bias_c   = 0.0; gfs_bias_n   = 0
    if ecmwf is not None:
        ecmwf_bias_c, ecmwf_bias_n = get_bias_correction(
            lat, lon, calendar_month_day,
            min_observations=MIN_BIAS_OBSERVATIONS,
            model="ecmwf_ifs025",
        )
        if ecmwf_bias_c != 0.0:
            ecmwf = {**ecmwf, "mu_c": ecmwf["mu_c"] + ecmwf_bias_c}
    if gfs is not None:
        gfs_bias_c, gfs_bias_n = get_bias_correction(
            lat, lon, calendar_month_day,
            min_observations=MIN_BIAS_OBSERVATIONS,
            model="gfs_global",
        )
        if gfs_bias_c != 0.0:
            gfs = {**gfs, "mu_c": gfs["mu_c"] + gfs_bias_c}
    if ecmwf_bias_c != 0.0 or gfs_bias_c != 0.0:
        logger.debug(
            f"Per-model bias: ECMWF {ecmwf_bias_c:+.2f}C (n={ecmwf_bias_n}) "
            f"GFS {gfs_bias_c:+.2f}C (n={gfs_bias_n})"
        )

    # Blend ECMWF + GFS
    if ecmwf and gfs:
        mu_fcst, sigma_fcst = _blend_gaussians(
            ecmwf["mu_c"], ecmwf["sigma_c"], ECMWF_WEIGHT,
            gfs["mu_c"],   gfs["sigma_c"],   GFS_WEIGHT,
        )
        n_members = ecmwf["n"] + gfs["n"]
        sources   = ["ecmwf", "gfs"]
    elif ecmwf:
        mu_fcst, sigma_fcst = ecmwf["mu_c"], ecmwf["sigma_c"]
        n_members = ecmwf["n"]
        sources   = ["ecmwf"]
    else:
        mu_fcst, sigma_fcst = gfs["mu_c"], gfs["sigma_c"]  # type: ignore[union-attr]
        n_members = gfs["n"]  # type: ignore[union-attr]
        sources   = ["gfs"]

    # Tag the source list so downstream can see bias was applied
    if ecmwf_bias_c != 0.0 or gfs_bias_c != 0.0:
        sources.append("bias_corrected")

    # Climatological prior
    clim = _get_era5_clim_distribution(lat, lon, date_str)
    clim_mu_c    = clim["mu_c"]    if clim else mu_fcst
    clim_sigma_c = clim["sigma_c"] if clim else max(sigma_fcst, 3.0)
    clim_kde     = clim.get("kde") if clim else None   # KDE object or None
    if clim:
        sources.append("era5")

    # Bayesian blend: forecast ↔ climatology
    alpha = _forecast_alpha(days_ahead)
    if clim and alpha < 1.0:
        mu_c, sigma_c = _blend_gaussians(
            mu_fcst,    sigma_fcst,    alpha,
            clim_mu_c,  clim_sigma_c,  1.0 - alpha,
        )
    else:
        mu_c, sigma_c = mu_fcst, sigma_fcst

    # US supplemental: NOAA NBM or NWS hourly
    if _is_us_coordinate(lat, lon):
        nbm = _get_noaa_nbm_distribution(lat, lon, date_str)
        if nbm:
            # Incorporate NBM with moderate weight (0.30) as an additional source
            mu_c, sigma_c = _blend_gaussians(
                mu_c,        sigma_c,        0.70,
                nbm["mu_c"], nbm["sigma_c"], 0.30,
            )
            sources.append("noaa_nbm")
        else:
            nws_hr = _get_nws_hourly_distribution(lat, lon, date_str)
            if nws_hr:
                mu_c, sigma_c = _blend_gaussians(
                    mu_c,              sigma_c,              0.80,
                    nws_hr["mu_c"],    nws_hr["sigma_c"],    0.20,
                )
                sources.append("nws_hourly")

    # (Bias correction happens per-model BEFORE blending — see above.)

    # Tomorrow.io outlier sanity check
    tio_temp = _get_tomorrowio_point_estimate(lat, lon, date_str)
    if tio_temp is not None:
        deviation = abs(tio_temp - mu_c)
        if deviation > 5.0:
            logger.warning(
                f"Tomorrow.io sanity check: forecast mu={mu_c:.1f}°C but "
                f"Tomorrow.io={tio_temp:.1f}°C (delta={deviation:.1f}°C)"
            )

    # Final sanity gate
    if sigma_c < MIN_FORECAST_SIGMA_C:
        sigma_c = MIN_FORECAST_SIGMA_C
    if sigma_c > MAX_FORECAST_SIGMA_C:
        logger.warning(
            f"Forecast sigma={sigma_c:.2f}°C exceeds MAX ({MAX_FORECAST_SIGMA_C}) — skipping event"
        )
        _dist_cache[cache_key] = None
        return None

    result = {
        "mu_c":        round(mu_c, 3),
        "sigma_c":     round(sigma_c, 3),
        "clim_mu_c":   round(clim_mu_c, 3),
        "clim_sigma_c":round(clim_sigma_c, 3),
        "days_ahead":  days_ahead,
        "sources":     sources,
        "n_members":   n_members,
        "alpha":       round(alpha, 3),
        # Per-model bias corrections applied to the raw model means before blending.
        # `bias_c` is the ECMWF-weighted contribution to the blended mu shift,
        # retained under its original key for backwards compatibility with
        # downstream consumers (dashboard, bias logger).
        "bias_c":       round(
            (ecmwf_bias_c if ecmwf else 0.0) * (ECMWF_WEIGHT if ecmwf and gfs else 1.0)
            + (gfs_bias_c if gfs else 0.0) * (GFS_WEIGHT   if ecmwf and gfs else (1.0 if gfs else 0.0)),
            3,
        ),
        "n_bias_obs":   max(ecmwf_bias_n, gfs_bias_n),
        "ecmwf_bias_c": round(ecmwf_bias_c, 3),
        "ecmwf_bias_n": ecmwf_bias_n,
        "gfs_bias_c":   round(gfs_bias_c, 3),
        "gfs_bias_n":   gfs_bias_n,
        "clim_kde":    clim_kde,    # KDE object (not serialisable — stays in memory only)
    }

    logger.info(
        f"Distribution ({lat:.2f},{lon:.2f}) {date_str} "
        f"mu={mu_c:.2f}°C sd={sigma_c:.2f}°C alpha={alpha:.2f} "
        f"sources={sources}"
    )

    _dist_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Probability computation
# ---------------------------------------------------------------------------

def get_temp_range_probability(
    mu_c: float,
    sigma_c: float,
    range_low: float | None,
    range_high: float | None,
    unit: str,
    kde=None,
) -> float:
    """
    Compute P(range_low ≤ T_max < range_high).

    When a KDE object is supplied (USE_KDE_CLIM=True and sufficient history),
    uses the KDE to integrate the climatological distribution directly.
    Otherwise falls back to the blended normal distribution N(μ_c, σ_c).

    All bounds are converted to Celsius before evaluation.
    Open-ended bounds (None) map to ±∞.

    Exact-bin interpretation: Polymarket bins are labelled in whole degrees
    (e.g. "15°C"), meaning the recorded temperature equals that whole-degree
    value.  We model this as the continuous interval [N, N+1) — i.e., [15.0,
    16.0) for "15°C" and [46.0, 47.0) for "46°F".  When range_low == range_high
    (the parser's representation of an exact bin), we extend range_high by 1.0
    in the original unit before converting to Celsius.

    Args:
        mu_c, sigma_c: blended distribution parameters (Celsius)
        range_low, range_high: outcome bounds in `unit` (or None for open end)
        unit: "celsius" or "fahrenheit"
        kde: scipy gaussian_kde object from climatological prior, or None

    Returns a probability in (0, 1).
    """
    # Exact-bin fix: "15°C" is stored as (15.0, 15.0).
    # Extend to [15.0, 16.0) before converting to Celsius so the bin has
    # nonzero width.  This applies to both °C and °F bins.
    if range_low is not None and range_high is not None and range_low == range_high:
        range_high = range_low + 1.0

    lo_c = _to_celsius(range_low,  unit)
    hi_c = _to_celsius(range_high, unit)

    if lo_c is None and hi_c is None:
        return 1.0

    # KDE path — numerical integration over the kernel density
    if kde is not None:
        _LO = lo_c if lo_c is not None else (mu_c - 10 * sigma_c)
        _HI = hi_c if hi_c is not None else (mu_c + 10 * sigma_c)
        try:
            prob = float(kde.integrate_box_1d(_LO, _HI))
            return max(0.0, min(1.0, prob))
        except Exception:
            pass  # fall through to normal CDF on any numerical error

    # Normal distribution path
    if lo_c is None:
        return float(scipy_norm.cdf(hi_c, loc=mu_c, scale=sigma_c))
    if hi_c is None:
        return float(1.0 - scipy_norm.cdf(lo_c, loc=mu_c, scale=sigma_c))
    return float(
        scipy_norm.cdf(hi_c, loc=mu_c, scale=sigma_c)
        - scipy_norm.cdf(lo_c, loc=mu_c, scale=sigma_c)
    )
