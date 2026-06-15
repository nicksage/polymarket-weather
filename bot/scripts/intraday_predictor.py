"""
intraday_predictor.py — Real-time bin-probability predictor for active
US Polymarket "highest temperature" markets, with a visual dashboard.

Architecture (per city, per moment):

  PRIOR    = Open-Meteo daily-max forecast at the settlement station
             coords, treated as N(forecast_high, forecast_sigma²).
             forecast_sigma narrows as we approach/pass the forecast peak.

  EVIDENCE = (1) live observations from the settlement station via the
             NWS API (the same NOAA METAR feed that Wunderground / Poly-
             market read from for US airports).
             (2) Today's neighbor observations + their wind alignment vs
             the city's prevailing wind, from the cached neighbor_obs.db.

  POSTERIOR = truncated normal: P(day_high in bin) constrained to
              day_high >= observed_max_so_far.  Neighbor "upwind has
              peaked + is cooling" boosts confidence in current bin.

  EDGE      = our_p[bin] − polymarket_yes_price[bin].  Buy when > margin.

Produces a single self-contained HTML dashboard with:
  - One card per city: current temp / wind / forecast high / observed high
  - Sortable table of every (city, bin) decision with our P vs market P,
    edge, and recommendation
  - Filters by city, date, recommendation, min edge

Usage:
    cd bot
    python -m scripts.intraday_predictor                       # all US cities, today
    python -m scripts.intraday_predictor --city Dallas         # one city
    python -m scripts.intraday_predictor --min-edge 0.10
    python -m scripts.intraday_predictor --html data/intraday.html
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
from zoneinfo import ZoneInfo

import httpx

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS  # type: ignore
from polymarket  import search_temp_high_events  # type: ignore
try:
    from scripts.find_nearby_stations import US_CITY_STATES  # type: ignore
except Exception:
    US_CITY_STATES = {}

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("intraday")
logging.getLogger("httpx").setLevel(logging.WARNING)

NEIGHBOR_DB    = os.path.join(_BOT_DIR, "data", "neighbor_obs.db")
DEFAULT_OUT    = os.path.join(_BOT_DIR, "data", "intraday_dashboard.html")
NWS_BASE       = "https://api.weather.gov"
NWS_USER_AGENT = "polymarket-weather-bot/1.0 (intraday-predictor)"
OPENMETEO_URL  = "https://api.open-meteo.com/v1/forecast"

# Forecast sigma — fallback used when a city isn't in the per-city
# calibration file (data/forecast_calibration.json).  Overridable via
# the PREDICTOR_DEFAULT_SIGMA_C env var in .env.
DEFAULT_FORECAST_SIGMA_C = float(os.getenv("PREDICTOR_DEFAULT_SIGMA_C", "2.0"))

# === Climatological σ prior — Quick Fix A (default ON 2026-06-12) ===
#
# The per-city σ calibration in forecast_calibration.json was generated
# against Open-Meteo's `historical-forecast-api`, which we suspect serves
# short-lead-time (near-nowcast) forecasts rather than true day-ahead.
# This produces RMSEs in the 0.5–1.1°C range across our 11 US cities,
# vs. published day-ahead NWS MaxT MAE of ~1.5–2.0°C in summer.
#
# Using those numbers as σ makes the predictor systematically over-
# confident on upper bins (Atlanta/NYC plateau cases, June 12 2026).
#
# This flag DEFAULTS ON.  It overrides the contaminated per-city values
# with a uniform climatological prior:
#   1.75°C — midpoint of documented day-ahead summer NWS MaxT MAE band
#            (Glahn & Lowry 1972; NCEP NBM validation papers).
#
# This is a defensible PRIOR with a citation, NOT a tuned floor.  It
# does not need backtest validation to ship because widening σ can
# only REDUCE confidence — it cannot make the bot bet more aggressively.
# Asymmetric risk: lower trade volume / smaller sizes; no new losses.
#
# Known limitation accepted: marine cities (SF, LA) where genuine NWS
# skill is tight will be over-widened by this uniform prior, costing
# some edge.  Do NOT tune per-city to fit recent losses — that's the
# contaminated-calibration trap with the sign flipped.
#
# Will be replaced by clean NWS-native per-city residuals after ~30
# days of accumulation (post-recovery-helper commit).  Until then,
# this is the safe-by-construction quarantine of bad calibration.
#
# Disable with `PREDICTOR_USE_CLIMATOLOGICAL_SIGMA=0`.
PREDICTOR_USE_CLIMATOLOGICAL_SIGMA = bool(int(
    os.getenv("PREDICTOR_USE_CLIMATOLOGICAL_SIGMA", "1")))
PREDICTOR_CLIMATOLOGICAL_SIGMA_C = float(
    os.getenv("PREDICTOR_CLIMATOLOGICAL_SIGMA_C", "1.75"))

# === σ floor (2026-06-15, after Q2 realism check) ===
#
# Q2 diagnostic confirmed σ is materially under-calibrated:
#   avg σ_c     = 0.96
#   avg |error| = 1.25  (ratio 1.30)
#   pct within 1σ = 60.0% (expected ~68%)
#   pct within 2σ = 75.6% (expected ~95%)
#
# The 2σ shortfall is the louder flag — fat tails the Gaussian can't
# represent.  W2 Phase C empirical-residual CDF is the durable fix;
# this floor is the safe-by-construction patch in the meantime.
#
# Was 0.3°C (essentially "almost never bind").  Raised to 1.3°C — the
# value Q2's ratio argues for as a sigma "should be" lower bound.  This
# stops σ from collapsing to ~0.30°C after observations land, which is
# the "100% on one bin → off-by-one-bin guaranteed loss" failure mode.
#
# Cannot make the bot more aggressive — strictly widens.  After 24h of
# live data, re-run Q2.  If pct_within_1σ is still well below 68% even
# with the floor binding, that's evidence the BASE σ (not just the
# collapsed values) needs widening — argument for the empirical CDF.
#
# Override with `PREDICTOR_SIGMA_FLOOR_C` if you want it tighter/wider.
PREDICTOR_SIGMA_FLOOR_C = float(
    os.getenv("PREDICTOR_SIGMA_FLOOR_C", "1.3"))

# === Immediate post-peak narrowing — Quick Fix B (default ON 2026-06-12) ===
#
# When True (default), the time-since-observed-peak narrowing in
# estimate_day_high_dist fires at hours_since_obs_peak >= 0 instead of
# >= 1.  Closes the ~2-hour window where the day has plateaued but the
# bot hasn't recognized it yet (Atlanta/NYC pattern).
#
# Safe by construction: this can only narrow σ post-peak — never widen.
# Cannot make the bot bet more aggressively.
#
# At hours_since = 0 the geometric factor is 0.7^0 = 1.0 so σ is
# unchanged in the instant of the peak itself; the benefit is at
# h=0.5+ (fractional hours-since-peak from sub-hour scan timing).
#
# Will be replaced by the HRRR plateau signal when HRRR activates (the
# HRRR signal is a better, physically-grounded trigger for the same
# code path).  B stays as the non-CAM-city fallback indefinitely.
#
# Disable with `PREDICTOR_IMMEDIATE_POST_PEAK_NARROW=0`.
PREDICTOR_IMMEDIATE_POST_PEAK_NARROW = bool(int(
    os.getenv("PREDICTOR_IMMEDIATE_POST_PEAK_NARROW", "1")))

# === Plausibility ceiling — Quick Fix C (default ON 2026-06-12) ===
#
# Crude prototype of the HRRR/W3 physical ceiling using NWS cloud-cover
# data we already fetch but don't currently use.  When fired, populates
# `truncate_at_hi` in the probability_in_bin integrator — the same slot
# W2 Phase A reserved for W3.
#
# Triggers when ALL true:
#   - required_rate > IMPLAUSIBLE_RATE_C_PER_H (heating rate needed
#     to reach forecast peak from current state)
#   - current_hour >= AFTERNOON_HOUR_THRESHOLD
#   - cloud_cover_pct > CLOUD_THRESHOLD
#
# When fired, sets:
#   ceiling = observed_max + plausible_remaining_rise(remaining_hours,
#                                                       cloud_pct)
#   truncate_at_hi = ceiling + CEILING_BUFFER_C
#
# Cold-start skip enforced: when cold_start_suspect is True, do not
# apply (the rise model is meaningless if peak already happened).
#
# Logs every fire so the HRRR/W3 work has data on whether the
# heuristic pointed the right direction, and any clipped winner is
# traceable back to a specific cap event.
#
# Combined with HRRR ceiling (when HRRR active) via min() — most
# conservative wins, so the two signals can coexist safely.
#
# Disable with `PREDICTOR_USE_PLAUSIBILITY_CEILING=0`.
PREDICTOR_USE_PLAUSIBILITY_CEILING = bool(int(
    os.getenv("PREDICTOR_USE_PLAUSIBILITY_CEILING", "1")))
PREDICTOR_CEILING_BUFFER_C = float(
    os.getenv("PREDICTOR_CEILING_BUFFER_C", "1.0"))
PREDICTOR_IMPLAUSIBLE_RATE_C_PER_H = float(
    os.getenv("PREDICTOR_IMPLAUSIBLE_RATE_C_PER_H", "2.0"))
PREDICTOR_AFTERNOON_HOUR_THRESHOLD = int(
    os.getenv("PREDICTOR_AFTERNOON_HOUR_THRESHOLD", "13"))
PREDICTOR_CLOUD_THRESHOLD_PCT = float(
    os.getenv("PREDICTOR_CLOUD_THRESHOLD_PCT", "50"))

# === HRRR ceiling — Phase 1 of the HRRR plan (docs/hrrr_ceiling_spec.md) ===
#
# When PREDICTOR_USE_HRRR_CEILING=1, predict_bins fetches the latest HRRR
# (CONUS) or ICON-D2 (Central Europe) run via Open-Meteo for the
# settlement station, then:
#
#   (a) Recenters μ on max(observed_max, hrrr_remaining_max) when HRRR
#       is more pessimistic than the morning forecast — the "skepticism
#       mechanism" the spec calls out as fixing forecast-as-gospel.
#
#   (b) Caps the distribution from above at hrrr_remaining_max + buffer
#       (default 1.0°C) — the physical upper truncation that fixes
#       the Atlanta/NYC upper-tail overconfidence pattern.
#
#   (c) Triggers post-peak σ narrowing when HRRR's next 2-3h trajectory
#       is flat/falling — the "plateau signal" replacement for the
#       wall-clock hours_since_observed_peak lag.  Wall-clock branch
#       stays as fallback for cold-start days and non-CAM cities.
#
# Skipped automatically when:
#   - city has no same_day_model assigned (non-US, non-CE)
#   - cold_start_suspect flag is set on the event (HRRR's "remaining
#     hours" view can't recover a peak that already happened)
#   - HRRR fetch fails or returns implausible values (sanity gates)
#
# Activation has TWO gates: Phase 2 backtest improvement + Phase 0b
# confirming the T-group fix actually closed the observed_max-vs-
# settlement gap.  Both must pass before flipping this on.
PREDICTOR_USE_HRRR_CEILING = bool(int(
    os.getenv("PREDICTOR_USE_HRRR_CEILING", "0")))
PREDICTOR_HRRR_CEILING_BUFFER_C = float(
    os.getenv("PREDICTOR_HRRR_CEILING_BUFFER_C", "1.0"))
PREDICTOR_HRRR_MAX_STALENESS_H = float(
    os.getenv("PREDICTOR_HRRR_MAX_STALENESS_H", "3.0"))

# Per-city σ calibration — loaded from data/forecast_calibration.json
# (produced by scripts.forecast_rmse_calibration).  Falls back to the
# default above for any city not present in the file or if the file
# doesn't exist yet.
CALIBRATION_PATH = os.path.join(_BOT_DIR, "data", "forecast_calibration.json")
_CALIBRATION: dict = {}

# Per-station systematic forecast bias.  Produced by
# scripts.station_bias_calibration from historical forecast-vs-observed
# comparisons.  Applied as: corrected_mu = forecast - bias.  Positive
# bias = station's forecast runs HOT (we subtract); negative = cold.
STATION_BIAS_PATH = os.path.join(_BOT_DIR, "data", "station_bias.json")
_STATION_BIAS: dict = {}


def _load_calibration() -> None:
    """Loaded once at script start.  Re-run scripts.forecast_rmse_calibration
    to refresh the file."""
    global _CALIBRATION
    if not os.path.exists(CALIBRATION_PATH):
        log.info(f"No forecast calibration file at {CALIBRATION_PATH} — "
                  f"using DEFAULT_FORECAST_SIGMA_C={DEFAULT_FORECAST_SIGMA_C} "
                  "for all cities.  Run scripts.forecast_rmse_calibration "
                  "to generate per-city σ values.")
        return
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as fh:
            _CALIBRATION = json.load(fh)
        n = len(_CALIBRATION.get("by_city", {}))
        log.info(f"Loaded per-city σ calibration for {n} cities "
                  f"(generated {_CALIBRATION.get('generated_at', '?')[:19]})")
    except Exception as e:
        log.warning(f"Failed to load {CALIBRATION_PATH}: {e}.  Using default σ.")
        _CALIBRATION = {}


def get_city_sigma(city: str) -> float:
    """Return the σ to use for this city's prior, with safe fallback.

    When PREDICTOR_USE_CLIMATOLOGICAL_SIGMA=1, override the per-city
    calibration with a uniform climatological prior (default 1.75°C).
    The per-city numbers in forecast_calibration.json are derived from
    Open-Meteo's historical-forecast endpoint, which produces tighter-
    than-real RMSEs and makes the predictor overconfident on upper
    bins.  See the comment block above the constants for the full
    rationale.  Same epistemic status as quarantining contaminated
    data — we use a documented prior until clean data exists.
    """
    if PREDICTOR_USE_CLIMATOLOGICAL_SIGMA:
        return PREDICTOR_CLIMATOLOGICAL_SIGMA_C
    entry = (_CALIBRATION.get("by_city") or {}).get(city)
    if not entry:
        return DEFAULT_FORECAST_SIGMA_C
    s = entry.get("sigma")
    if s is None or s <= 0:
        return DEFAULT_FORECAST_SIGMA_C
    return float(s)


# W2 Phase C — minimum residual count below which the empirical CDF is
# too noisy to trust.  Below this, the gaussian path is used even when
# PREDICTOR_CDF_IMPL=empirical.  Tuned conservatively — a 60-day window
# typically yields ~50 usable days per city after gaps.
EMPIRICAL_MIN_SAMPLES = int(os.getenv("PREDICTOR_EMPIRICAL_MIN_SAMPLES", "30"))


def get_city_centered_residuals(city: str) -> list[float] | None:
    """Return the sorted list of centered (mean-zero) forecast residuals
    for this city, or None if not enough samples for the empirical CDF.
    Centered residuals carry the SHAPE of forecast error — asymmetry, fat
    tails — without re-encoding the per-station mean bias (that's handled
    by station_bias upstream).  W2 Phase C consumer."""
    entry = (_CALIBRATION.get("by_city") or {}).get(city)
    if not entry:
        return None
    r = entry.get("centered_residuals")
    if not isinstance(r, list) or len(r) < EMPIRICAL_MIN_SAMPLES:
        return None
    return [float(x) for x in r]


def _load_station_bias() -> None:
    """Loaded once at script start.  Refresh via station_bias_calibration."""
    global _STATION_BIAS
    if not os.path.exists(STATION_BIAS_PATH):
        log.info(f"No station bias file at {STATION_BIAS_PATH} — "
                  "forecasts used as-is (no per-station bias correction). "
                  "Run scripts.station_bias_calibration to generate.")
        return
    try:
        with open(STATION_BIAS_PATH, encoding="utf-8") as fh:
            _STATION_BIAS = json.load(fh)
        n = len(_STATION_BIAS.get("by_station", {}))
        log.info(f"Loaded per-station bias for {n} stations "
                  f"(generated {_STATION_BIAS.get('generated_at', '?')[:19]})")
    except Exception as e:
        log.warning(f"Failed to load {STATION_BIAS_PATH}: {e}.  No bias correction.")
        _STATION_BIAS = {}


def get_station_bias(station_icao: str) -> float:
    """Return the per-station forecast bias (°C).  0.0 if no data.
    Subtract this from the forecast: corrected = forecast - bias."""
    entry = (_STATION_BIAS.get("by_station") or {}).get(station_icao)
    if not entry:
        return 0.0
    return float(entry.get("mean_bias_c") or 0.0)


# ---------------------------------------------------------------------------
# Statistics helpers — truncated normal CDF without scipy
# ---------------------------------------------------------------------------

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def truncated_normal_prob(lo: float | None, hi: float | None,
                           mu: float, sigma: float,
                           truncate_at: float) -> float:
    """Probability that X falls in [lo, hi], given X ~ N(mu, sigma²) and
    X >= truncate_at.  lo/hi can be None for open-ended intervals.

    Legacy interface — preserved for backtest scripts that import it
    directly.  Internally the live predictor now uses probability_in_bin
    (CDF-agnostic) so future distributions (NBM percentiles, empirical
    residuals) can drop in without touching this code path.
    """
    # Effective lower bound
    eff_lo = max(lo, truncate_at) if lo is not None else truncate_at
    if hi is not None and hi <= truncate_at:
        return 0.0   # bin entirely below truncation
    if hi is not None and eff_lo >= hi:
        return 0.0

    Z = 1.0 - normal_cdf(truncate_at, mu, sigma)
    if Z <= 1e-12:
        # Almost no mass above truncate — everything goes to "current or higher"
        # In practice this means we're confident the day high is at observed
        if hi is None:
            return 1.0   # "or higher" catches the residual
        return 1.0 if (lo is not None and lo <= truncate_at < hi) else 0.0

    cdf_lo = normal_cdf(eff_lo, mu, sigma)
    cdf_hi = normal_cdf(hi, mu, sigma) if hi is not None else 1.0
    return max(0.0, (cdf_hi - cdf_lo) / Z)


# ---------------------------------------------------------------------------
# W2 Phase A — CDF-agnostic probability integrator
# ---------------------------------------------------------------------------
#
# A "day-high CDF" is any callable that maps a temperature (°C) to
# P(day_high <= temp).  Concrete implementations:
#
#   make_gaussian_cdf(mu, sigma)       — current behavior (W2 Phase A; in place)
#   make_empirical_residual_cdf(...)   — W2 Phase C, fed from per-city
#                                         historical (forecast - actual)
#                                         residuals.  Asymmetric, fat tails.
#   make_nbm_percentile_cdf(...)       — W2 Phase B, fed from NBM/ensemble
#                                         percentile points if/when ingestion
#                                         lands.
#
# probability_in_bin integrates any of these over a bin's [lo, hi] with
# OPTIONAL two-sided truncation:
#   truncate_at_lo  — day_high >= this (current: observed_max_so_far)
#   truncate_at_hi  — day_high <= this (W3: physical ceiling estimate)
#
# When called with only truncate_at_lo, output matches truncated_normal_prob
# numerically to <1e-10 for any gaussian CDF.  Pure refactor — verified by
# tests added in test_predictor.py.

from typing import Callable

DayHighCDF = Callable[[float], float]


def make_gaussian_cdf(mu: float, sigma: float) -> DayHighCDF:
    """Drop-in CDF for the current Gaussian-day-high model.  W2 Phase A
    default; future workstreams add make_empirical_residual_cdf /
    make_nbm_percentile_cdf and dispatch via PREDICTOR_CDF_IMPL."""
    safe_sigma = max(sigma, 1e-6)
    return lambda t: normal_cdf(t, mu, safe_sigma)


def make_empirical_residual_cdf(center_temp_c: float,
                                  centered_residuals: list[float],
                                  scale: float = 1.0) -> DayHighCDF:
    """W2 Phase C — empirical day-high CDF built from per-city historical
    forecast residuals.  Captures asymmetry and fat tails that a
    symmetric Gaussian erases.

    The distribution is constructed as:
       implied_day_high_i = center_temp_c - (centered_residual_i * scale)

    center_temp_c: bias-corrected forecast high (output of upstream mu
       adjustments — observations, ensemble blending, cooling).
    centered_residuals: sorted, mean-zero per-city residual list from
       data/forecast_calibration.json.  Mean is ~0 by construction; the
       station bias is already applied upstream so we don't double-count.
    scale: multiplicative width factor.  Default 1.0 = use the empirical
       distribution as-is.  Pass `sigma / base_sigma_c` to inherit the
       wall-clock contraction that estimate_day_high_dist applies — keeps
       the empirical SHAPE while still narrowing late-day.

    Returns a CDF callable that linearly interpolates between sorted
    sample points.  O(log n) per query.
    """
    if not centered_residuals:
        # Degenerate point distribution — falls back to "all mass at center".
        return lambda t: 0.0 if t < center_temp_c else 1.0

    safe_scale = max(scale, 1e-6)
    samples = sorted(center_temp_c - r * safe_scale
                      for r in centered_residuals)
    n = len(samples)

    def _cdf(t: float) -> float:
        if t <= samples[0]:
            return 0.0
        if t >= samples[-1]:
            return 1.0
        # Binary search: largest index where samples[idx] <= t
        lo, hi = 0, n - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if samples[mid] <= t:
                lo = mid
            else:
                hi = mid
        x0, x1 = samples[lo], samples[hi]
        if x1 == x0:
            return (lo + 1) / n
        frac = (t - x0) / (x1 - x0)
        return (lo + frac) / n
    return _cdf


def probability_in_bin(bin_lo_c: float | None, bin_hi_c: float | None,
                         cdf: DayHighCDF,
                         truncate_at_lo: float | None = None,
                         truncate_at_hi: float | None = None) -> float:
    """P(day_high in [bin_lo_c, bin_hi_c]) given two-sided truncation.

    - bin_lo_c=None means the bin extends to -infinity ("X or below").
    - bin_hi_c=None means the bin extends to +infinity ("X or higher").
    - truncate_at_lo, if set, asserts day_high >= truncate_at_lo
      (e.g., the temperature we've already observed today).
    - truncate_at_hi, if set, asserts day_high <= truncate_at_hi
      (e.g., the W3 physical-ceiling estimate from remaining solar /
      sky state).  Currently always None — set to None preserves the
      pre-W3 one-sided-truncation behavior exactly.

    Numerical guarantee: with cdf=make_gaussian_cdf(mu, sigma) and
    truncate_at_hi=None, this returns the SAME value as
    truncated_normal_prob(bin_lo_c, bin_hi_c, mu, sigma, truncate_at_lo)
    to within floating-point noise.
    """
    # Clip bin to effective truncation range
    eff_lo = bin_lo_c
    if truncate_at_lo is not None:
        eff_lo = (max(eff_lo, truncate_at_lo) if eff_lo is not None
                   else truncate_at_lo)
    eff_hi = bin_hi_c
    if truncate_at_hi is not None:
        eff_hi = (min(eff_hi, truncate_at_hi) if eff_hi is not None
                   else truncate_at_hi)

    # Bin entirely outside truncation window → zero
    if (truncate_at_lo is not None and bin_hi_c is not None
        and bin_hi_c <= truncate_at_lo):
        return 0.0
    if (truncate_at_hi is not None and bin_lo_c is not None
        and bin_lo_c >= truncate_at_hi):
        return 0.0
    # Bin clipped down to empty range
    if (eff_lo is not None and eff_hi is not None
        and eff_lo >= eff_hi):
        return 0.0

    # Normalization Z = P(day_high in [truncate_at_lo, truncate_at_hi])
    Z_lo = cdf(truncate_at_lo) if truncate_at_lo is not None else 0.0
    Z_hi = cdf(truncate_at_hi) if truncate_at_hi is not None else 1.0
    Z = Z_hi - Z_lo

    if Z <= 1e-12:
        # Degenerate: distribution gives essentially zero mass to the
        # truncated range.  Matches truncated_normal_prob's fallback:
        # the "or higher" bin catches the residual; otherwise lo<=ref<hi.
        if bin_hi_c is None:
            return 1.0
        ref = truncate_at_lo if truncate_at_lo is not None else (Z_lo + Z_hi) / 2
        if bin_lo_c is not None and bin_lo_c <= ref < bin_hi_c:
            return 1.0
        return 0.0

    p_lo = cdf(eff_lo) if eff_lo is not None else 0.0
    p_hi = cdf(eff_hi) if eff_hi is not None else 1.0
    return max(0.0, (p_hi - p_lo) / Z)


# Dispatch knob.  W2 Phase A ships with only "gaussian" wired.  Phase C
# will register "empirical"; Phase B will register "nbm" (if pursued).
PREDICTOR_CDF_IMPL = os.getenv("PREDICTOR_CDF_IMPL", "gaussian").lower()


# ---------------------------------------------------------------------------
# NWS API — live observations
# ---------------------------------------------------------------------------
#
# METAR precision handling — see the Atlanta 2026-06-12 finding for context.
#
# NWS API returns observation temperatures at two different precisions:
#   - Synoptic METARs (typically :52 past the hour): tenths precision,
#     parsed from the T-group in the REMARKS section (e.g. "T03280206"
#     means +32.8°C / +20.6°C).
#   - 5-minute MADIS / SPECI cycles: whole-°C precision only, the body
#     value rounded half-up.
#
# Trusting both equally and taking the max means rounded-up body values
# systematically overshoot truth by up to 0.5°C, which manifests as the
# bot's observed_max sitting ~0.2-0.5°C above what Wunderground / DSM
# settle to.  On Atlanta 2026-06-12, T-groups consistently showed 32.8°C
# while body-only readings showed "33" → bot recorded 33.0°C.

import re as _re

# T-group regex: T <signT> <T*3> <signD> <D*3>, anywhere in REMARKS.
#   T03280206 → +32.8°C / +20.6°C
#   T13280206 → -32.8°C / +20.6°C
#   T03281089 → +32.8°C / -8.9°C
_METAR_T_GROUP_RE = _re.compile(r'\bT([01])(\d{3})([01])(\d{3})(?:\s|$)')


def parse_metar_t_group(raw_message: str | None
                          ) -> tuple[float | None, float | None]:
    """Parse the T-group from a METAR raw message.  Returns
    (temp_c, dewpoint_c) at tenths precision, or (None, None) if no
    T-group present.  Format and semantics per FMH-1 / WMO 306.

    See the precision-handling comment block above for why this matters."""
    if not raw_message:
        return None, None
    m = _METAR_T_GROUP_RE.search(raw_message)
    if not m:
        return None, None
    s_t, t_digits, s_d, d_digits = m.groups()
    temp_c = float(t_digits) / 10.0
    if s_t == '1':
        temp_c = -temp_c
    dewpoint_c = float(d_digits) / 10.0
    if s_d == '1':
        dewpoint_c = -dewpoint_c
    return temp_c, dewpoint_c


# Conservative bound applied to body-only readings.
# METAR body temps are rounded half-up to whole °C; a body of "33" means
# the precise temp was in [32.5, 33.5).  For observed_max purposes
# (truncation floor of the day-high distribution), we use the lower
# bound — we only claim what we can confirm.  This prevents the
# "body rounded up to 33 while precise was 32.8" pattern from pushing
# observed_max 0.2°C above truth.
METAR_BODY_CONSERVATIVE_OFFSET_C = -0.5


def precise_temp_from_cycle(api_temp_c: float | None,
                              raw_message: str | None,
                              ) -> tuple[float | None, str]:
    """Return (precise_temp_c, precision_label) for one METAR cycle.

    precision_label values:
       'tenths'  — T-group found in raw_message, precise to ±0.05°C
       'whole'   — body-only reading, conservative -0.5°C lower bound applied
       'missing' — no temperature data

    The conservative bound for 'whole' is asymmetric: it makes
    observed_max a truthful lower bound rather than a likely-overshooting
    estimate.  Downstream uses (cooling detection trajectories, etc.)
    see the same value, which preserves the shape of the temperature
    curve since the offset is applied uniformly to body-only readings.
    """
    t_group_t_c, _ = parse_metar_t_group(raw_message)
    if t_group_t_c is not None:
        return t_group_t_c, 'tenths'
    if api_temp_c is None:
        return None, 'missing'
    return float(api_temp_c) + METAR_BODY_CONSERVATIVE_OFFSET_C, 'whole'


def fetch_nws_today_obs(icao: str, tz_str: str) -> list[dict]:
    """Return list of {hour_local, temp_c, wind_dir_deg, timestamp_utc} for
    today's observations from the given US airport station.  Uses the NWS
    public API — same NOAA METAR feed Wunderground reads.

    Thin wrapper around fetch_nws_obs_with_raw — keeps the existing
    signature stable so backtest scripts and tests don't break, while
    the scan loop opts into the richer two-value return.
    """
    hourly, _raw = fetch_nws_obs_with_raw(icao, tz_str)
    return hourly


def fetch_nws_obs_with_raw(icao: str, tz_str: str
                              ) -> tuple[list[dict], list[dict]]:
    """Same as fetch_nws_today_obs, plus raw cycles.  Returns:
       (hourly_max, raw_cycles)
       hourly_max:  what the predictor consumes (one row per local hour, max temp)
       raw_cycles:  every observation returned by NWS, preserving the rawMessage
                    METAR text.  Used by the scan loop to populate
                    `raw_metar_log` so future audits can diagnose
                    settle_divergence cases from the raw obs, not just
                    the parsed max.

    Single HTTP call shared between the two outputs.
    """
    tz = ZoneInfo(tz_str)
    now_local = datetime.now(tz)
    today_local = now_local.date()
    start_local = datetime.combine(today_local, datetime.min.time(), tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    url = f"{NWS_BASE}/stations/{icao}/observations"
    try:
        r = httpx.get(url, params={"start": start_utc}, headers=headers, timeout=30)
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as e:
        log.warning(f"NWS fetch failed for {icao}: {e}")
        return [], []

    by_hour: dict[int, dict] = {}
    raw_cycles: list[dict] = []
    for f in features:
        props = f.get("properties") or {}
        ts_str = props.get("timestamp")
        temp_obj = props.get("temperature") or {}
        wind_obj = props.get("windDirection") or {}
        ws_obj   = props.get("windSpeed")   or {}
        dew_obj  = props.get("dewpoint")    or {}
        api_t_c   = temp_obj.get("value")
        wd        = wind_obj.get("value")
        ws        = ws_obj.get("value")
        api_dew_c = dew_obj.get("value")
        present_weather = props.get("presentWeather") or []
        raw_msg = props.get("rawMessage")

        if not ts_str:
            continue
        try:
            ts_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        ts_local = ts_utc.astimezone(tz)
        if ts_local.date() != today_local:
            continue

        # Apply precision-aware temperature reading:
        #   - T-group present in raw_message → tenths precision
        #   - Body-only reading → conservative -0.5°C lower bound
        # See METAR precision handling block above.
        precise_t_c, precision = precise_temp_from_cycle(api_t_c, raw_msg)
        # Same logic for dewpoint when T-group provides it.
        _, t_group_dew = parse_metar_t_group(raw_msg)
        precise_dew_c = t_group_dew if t_group_dew is not None else (
            float(api_dew_c) if api_dew_c is not None else None)

        # Persist EVERY cycle to raw, even those without a temperature
        # reading — a missing-temp METAR around peak hour is itself a
        # diagnostic clue.  Hourly-max only takes valid temp rows.
        raw_cycles.append({
            "icao":                icao,
            "event_date":          today_local.isoformat(),
            "cycle_timestamp_utc": ts_str,
            "raw_message":         raw_msg,
            "temp_c":              precise_t_c,
            "temp_precision":      precision,
            "dewpoint_c":          precise_dew_c,
            "wind_dir_deg":        float(wd) if wd is not None else None,
            "wind_speed_mps":      float(ws) if ws is not None else None,
            "present_weather":     ",".join(p.get("rawString", "")
                                              for p in present_weather)
                                    if present_weather else None,
        })

        if precise_t_c is None:
            continue
        h = ts_local.hour
        existing = by_hour.get(h)
        if existing is None or precise_t_c > existing["temp_c"]:
            by_hour[h] = {
                "hour_local":    h,
                "temp_c":        precise_t_c,
                "wind_dir_deg":  float(wd) if wd is not None else None,
                "timestamp_utc": ts_str,
            }
    hourly = sorted(by_hour.values(), key=lambda r: r["hour_local"])
    return hourly, raw_cycles


# ---------------------------------------------------------------------------
# NWS — forecast prior (hourly grid forecast for the settlement station coords)
# ---------------------------------------------------------------------------

# Cache the points → grid resolution since it rarely changes per (lat,lon)
_NWS_POINTS_CACHE: dict[tuple[float, float], str] = {}


def _sunset_local_hour(d: date, lat: float, lon: float, tz_str: str) -> int:
    """NOAA solar-position sunset calc.  No external API; accurate to ~5 min.
    Used because NWS's forecast endpoint doesn't include sunset and we don't
    want to call a second service just for that."""
    doy = d.timetuple().tm_yday
    gamma = 2 * math.pi * (doy - 1) / 365
    # Solar declination (radians)
    decl = (0.006918 - 0.399912*math.cos(gamma) + 0.070257*math.sin(gamma)
            - 0.006758*math.cos(2*gamma) + 0.000907*math.sin(2*gamma)
            - 0.002697*math.cos(3*gamma) + 0.00148*math.sin(3*gamma))
    # Equation of time (minutes)
    eot = 229.18 * (
        0.000075 + 0.001868*math.cos(gamma) - 0.032077*math.sin(gamma)
        - 0.014615*math.cos(2*gamma) - 0.040849*math.sin(2*gamma)
    )
    # Hour angle for sunset (zenith = 90.833° accounts for atmospheric refraction)
    lat_r = math.radians(lat)
    try:
        cos_ha = ((math.cos(math.radians(90.833))
                    - math.sin(lat_r) * math.sin(decl))
                   / (math.cos(lat_r) * math.cos(decl)))
    except ZeroDivisionError:
        return 19
    if cos_ha > 1:  return 0     # polar night
    if cos_ha < -1: return 23    # polar day
    ha_deg = math.degrees(math.acos(cos_ha))
    # Sunset in minutes since UTC midnight
    sunset_utc_min = 720 + 4 * (-lon) + ha_deg * 4 - eot
    sunset_utc = (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
                   + timedelta(minutes=sunset_utc_min))
    return sunset_utc.astimezone(ZoneInfo(tz_str)).hour


def fetch_nws_today_forecast(lat: float, lon: float, tz_str: str) -> dict:
    """Hourly forecast from the official NWS API for the given coords.

    This is the SAME data source Wunderground uses for US settlement
    stations, and matches what Polymarket's resolver reads.  Returns the
    same shape as fetch_openmeteo_today() so the rest of the pipeline
    doesn't change:

        {hourly: [(hour_local, temp_c), ...],
         forecast_high: float (C),
         forecast_peak_hour: int (local),
         sunset_hour: int (local)}

    NWS gives forecast temps in Fahrenheit by default; we convert to C
    since the rest of the model is metric.
    """
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    cache_key = (round(lat, 4), round(lon, 4))

    # Step 1 — points endpoint (gives us the gridpoint forecast URL)
    forecast_url = _NWS_POINTS_CACHE.get(cache_key)
    if not forecast_url:
        try:
            r = httpx.get(f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
                           headers=headers, timeout=30)
            r.raise_for_status()
            forecast_url = r.json()["properties"]["forecastHourly"]
            _NWS_POINTS_CACHE[cache_key] = forecast_url
        except Exception as e:
            log.warning(f"NWS points lookup failed at ({lat:.3f},{lon:.3f}): {e}")
            return {}

    # Step 2 — hourly forecast
    try:
        r = httpx.get(forecast_url, headers=headers, timeout=30)
        r.raise_for_status()
        periods = (r.json().get("properties") or {}).get("periods") or []
    except Exception as e:
        log.warning(f"NWS hourly forecast failed at ({lat:.3f},{lon:.3f}): {e}")
        return {}

    # Convert + filter to today (in city local time).  Also extract
    # cloud cover per hour for the Quick Fix C plausibility ceiling —
    # NWS exposes `properties.skyCover.value` (percentage 0-100) on
    # each period.
    tz = ZoneInfo(tz_str)
    today_local = datetime.now(tz).date()
    hourly: list[tuple[int, float]] = []
    hourly_clouds: dict[int, float] = {}
    for p in periods:
        ts_str = p.get("startTime")
        temp = p.get("temperature")
        unit = (p.get("temperatureUnit") or "F").upper()
        if not ts_str or temp is None:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        ts_local = ts.astimezone(tz)
        if ts_local.date() != today_local:
            continue
        temp_c = (float(temp) - 32) * 5/9 if unit == "F" else float(temp)
        hourly.append((ts_local.hour, temp_c))
        sky = p.get("skyCover")
        if isinstance(sky, dict):
            sky_val = sky.get("value")
            if sky_val is not None:
                try:
                    hourly_clouds[ts_local.hour] = float(sky_val)
                except (TypeError, ValueError):
                    pass

    if not hourly:
        return {}

    forecast_peak_hour, forecast_high = max(hourly, key=lambda x: x[1])
    sunset_hour = _sunset_local_hour(today_local, lat, lon, tz_str)

    return {
        "hourly":             hourly,
        "hourly_clouds":      hourly_clouds,
        "forecast_high":      forecast_high,
        "forecast_peak_hour": forecast_peak_hour,
        "sunset_hour":        sunset_hour,
    }


# ---------------------------------------------------------------------------
# Open-Meteo — DEPRECATED forecast source.  Kept only as a fallback if NWS
# is unreachable.  The main pipeline now calls fetch_nws_today_forecast()
# above, which matches Polymarket's settlement source (NWS METAR feed
# routed through Wunderground).
# ---------------------------------------------------------------------------

def fetch_openmeteo_today(lat: float, lon: float, tz_str: str) -> dict:
    """Returns {hourly: [(hour_local, temp_c), ...], sunset_hour: int,
                  forecast_high: float, forecast_peak_hour: int}"""
    try:
        r = httpx.get(
            OPENMETEO_URL,
            params={
                "latitude":  lat,
                "longitude": lon,
                "hourly":    "temperature_2m",
                "daily":     "sunset",
                "timezone":  tz_str,
                "forecast_days": 1,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"Open-Meteo fetch failed at ({lat:.3f},{lon:.3f}): {e}")
        return {}

    h = data.get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    hourly: list[tuple[int, float]] = []
    for t, v in zip(times, temps):
        if v is None:
            continue
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
            hourly.append((dt.hour, float(v)))
        except ValueError:
            continue
    if not hourly:
        return {}

    forecast_peak_hour, forecast_high = max(hourly, key=lambda x: x[1])

    sunset_hour = 19   # safe default
    daily = data.get("daily", {}) or {}
    sset = (daily.get("sunset") or [None])[0]
    if sset:
        try:
            sunset_hour = datetime.strptime(sset, "%Y-%m-%dT%H:%M").hour
        except ValueError:
            pass

    return {
        "hourly":             hourly,
        "forecast_high":      forecast_high,
        "forecast_peak_hour": forecast_peak_hour,
        "sunset_hour":        sunset_hour,
    }


# ---------------------------------------------------------------------------
# HRRR / ICON-D2 rapid-update fetch — Phase 1 of the HRRR ceiling plan
# ---------------------------------------------------------------------------
#
# Pulls a rapid-update CAM (HRRR for CONUS, ICON-D2 for Central Europe)
# via Open-Meteo's `models=` parameter.  Returns the "remaining-day max"
# and the trajectory needed for both the upper-truncation ceiling AND
# the plateau-signal post-peak narrowing.  Both are baseline (not
# optional) per the spec discussion.
#
# Open-Meteo exposes both models without GRIB2 plumbing.  If Tier 1's
# accuracy is insufficient at hard terrain/coastal stations (Phase 3
# in the spec), this gets replaced by Herbie-based direct GRIB2 access
# for controlled point extraction.
#
# Behavior under failure: returns None.  Callers must handle this and
# fall back to the existing (HRRR-unaware) distribution path.  Cold-
# start days (cold_start_suspect flag set) should skip the fetch
# entirely — see the dispatch in scheduled_predictor.

OPENMETEO_RAPID_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_rapid_model_remaining_max(
    lat: float, lon: float, tz_str: str, model: str
) -> dict | None:
    """Fetch rapid-update CAM hourly forecast and extract:
       {
         "remaining_max_c": float — max 2m temperature from "now" through
                                     end of station-local day
         "trajectory":      list of dicts {hour_local, temp_c, cloud_pct}
                              for the same window — used by the plateau
                              signal (HRRR says day is done climbing)
         "cycle_time":      ISO timestamp of latest model run available
                              (for staleness check; Open-Meteo doesn't
                              always expose this — best-effort)
         "model":           the model identifier requested
       }
    Returns None if:
       - the fetch fails (network, 4xx, etc.)
       - no usable hourly data in the response
       - all temperatures in the response window are missing

    Caller is responsible for staleness/sanity gates and feature-flag.
    """
    # Logical name (used in data_quality_flag, station_meta, tests)
    # → Open-Meteo API model name.  Open-Meteo restructured the model
    # identifiers some time after the spec was written: HRRR is exposed
    # as `gfs_hrrr` (NOAA branch), ICON-D2 as `dwd_icon_d2` (DWD branch).
    # Verified live 2026-06-13 against the production VPS — the bare
    # `models=hrrr` value returns HTTP 400 with no body; the prefixed
    # form returns full hourly data.  Keep the logical names stable so
    # the `*_ceiling_applied` data-quality flag values don't churn.
    _API_MODEL_NAME = {
        "hrrr":      "gfs_hrrr",
        "ncep_hrrr": "gfs_hrrr",
        "icon_d2":   "dwd_icon_d2",
    }
    api_model = _API_MODEL_NAME.get(model)
    if api_model is None:
        log.warning(f"fetch_rapid_model: unsupported model {model!r}; "
                     "returning None")
        return None
    try:
        r = httpx.get(
            OPENMETEO_RAPID_URL,
            params={
                "latitude":      lat,
                "longitude":     lon,
                "hourly":        "temperature_2m,cloud_cover",
                "models":        api_model,
                "timezone":      tz_str,
                "forecast_days": 1,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"Open-Meteo rapid-model fetch failed "
                    f"({model}) at ({lat:.3f},{lon:.3f}): {e}")
        return None

    h = data.get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    clouds = h.get("cloud_cover") or []
    if not times or not temps:
        return None

    tz = ZoneInfo(tz_str)
    now_local = datetime.now(tz)
    today_local = now_local.date()
    # "Remaining day" = hourly entries from current hour through end
    # of today in local time.
    trajectory: list[dict] = []
    for i, t in enumerate(times):
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
        # Open-Meteo returns times in the requested timezone already
        if dt.date() != today_local:
            continue
        if dt.hour < now_local.hour:
            continue
        if i >= len(temps):
            continue
        v = temps[i]
        if v is None:
            continue
        cloud_pct = (clouds[i] if i < len(clouds) and clouds[i] is not None
                       else None)
        trajectory.append({
            "hour_local": dt.hour,
            "temp_c":     float(v),
            "cloud_pct":  float(cloud_pct) if cloud_pct is not None else None,
        })

    if not trajectory:
        return None

    remaining_max_c = max(pt["temp_c"] for pt in trajectory)
    # Open-Meteo doesn't reliably expose the source model's cycle time
    # in this endpoint.  Best-effort: use the response's earliest
    # trajectory time as a proxy for "data is current as of at least
    # this hour."
    earliest_local_iso = (now_local.replace(hour=trajectory[0]["hour_local"],
                                              minute=0, second=0,
                                              microsecond=0)
                              .isoformat())
    return {
        "remaining_max_c": round(remaining_max_c, 2),
        "trajectory":      trajectory,
        "cycle_time":      earliest_local_iso,
        "model":           model,
    }


def _hrrr_data_passes_sanity(hrrr_data: dict,
                               observed_max_c: float,
                               forecast_high_c: float) -> bool:
    """Sanity gates for HRRR data before we let it influence pricing.

    Rejects HRRR data when:
      - remaining_max is implausibly far from forecast or observed
        (> 8°C delta — likely a model glitch or wrong-point extraction)
      - trajectory contains NaN-ish values

    Range-plausibility check is asymmetric: we accept HRRR being much
    LOWER than forecast (that's the whole point — skepticism mechanism)
    but reject HRRR being much HIGHER, because either it's spurious or
    indicates the morning forecast was the conservative one.  In the
    latter case we'd rather not over-anchor.

    Staleness check is best-effort — Open-Meteo doesn't reliably
    expose the source cycle time, so we use the response data's
    earliest-hour value as a proxy.
    """
    remaining_max = hrrr_data.get("remaining_max_c")
    if remaining_max is None:
        return False
    # Asymmetric range plausibility
    if observed_max_c > -50:
        # If we already have observations, HRRR can't be below observed
        # (the day can't end colder than what's already happened).
        # Allow 0.5°C slack for HRRR being slightly stale vs. our latest
        # METAR cycle.
        if remaining_max < observed_max_c - 0.5:
            return False
    # Reject obviously-broken cycles where HRRR is >8°C away from
    # forecast in either direction.
    if abs(remaining_max - forecast_high_c) > 8.0:
        return False
    trajectory = hrrr_data.get("trajectory") or []
    if not trajectory:
        return False
    for pt in trajectory:
        if pt.get("temp_c") is None:
            return False
    return True


# ---------------------------------------------------------------------------
# Plausibility ceiling — Quick Fix C
# ---------------------------------------------------------------------------
#
# Crude prototype of the HRRR/W3 physical ceiling using NWS skyCover
# already in our fetch path.  Fires only on clearly-implausible
# heating regimes (cloudy afternoons where the bot would otherwise
# bet the day climbs >2°C/hour).
#
# Will be superseded by the HRRR Tier 1 ceiling when HRRR activates,
# and by W3's empirical model when it fits.  Stays as the non-CAM
# city fallback indefinitely.


def plausible_remaining_rise_c(hours_remaining: float,
                                  cloud_pct: float) -> float:
    """Return a conservative upper bound on how much further the temp
    can plausibly rise given remaining hours of solar and cloud cover.

    Heuristic (NOT a physics model):
       max_rate = 2.0°C/hr under clear sky
       max_rate scales linearly down with cloud cover, floor 0.3°C/hr
       total rise capped at hours_remaining × max_rate
       hard ceiling of 5°C total rise regardless

    Returns 0.0 when hours_remaining <= 0 or inputs are invalid.

    This is intentionally a crude heuristic — its job is to flag
    obviously-implausible hot-bin probabilities, not to model the
    boundary layer.  W3 / HRRR replace it with proper models.
    """
    if hours_remaining <= 0:
        return 0.0
    cloud_pct = max(0.0, min(100.0, cloud_pct))
    # 0% clouds → 2.0°C/hr; 100% clouds → 0.5°C/hr; linear between
    max_rate_c_per_h = 2.0 - (cloud_pct / 100.0) * 1.5
    max_rate_c_per_h = max(0.3, max_rate_c_per_h)
    rise = min(hours_remaining * max_rate_c_per_h, 5.0)
    return rise


def avg_remaining_cloud_pct(hourly_clouds: dict[int, float],
                              current_hour: int,
                              sunset_hour: int) -> float | None:
    """Average cloud cover over the hours from now until sunset.
    Returns None if no cloud data available for the remaining window
    — caller treats None as "can't fire the ceiling, no data."
    """
    if not hourly_clouds or sunset_hour <= current_hour:
        return None
    values = [v for h, v in hourly_clouds.items()
                 if current_hour <= h < sunset_hour]
    if not values:
        return None
    return sum(values) / len(values)


def compute_plausibility_ceiling_c(
    current_temp_c: float,
    forecast_high_c: float,
    forecast_peak_hour: int,
    current_hour: int,
    sunset_hour: int,
    cloud_pct: float | None,
    ceiling_buffer_c: float = 1.0,
    implausible_rate_c_per_h: float = 2.0,
    afternoon_hour_threshold: int = 13,
    cloud_threshold_pct: float = 50.0,
) -> dict | None:
    """Compute the plausibility ceiling (Quick Fix C).

    Returns a dict with the ceiling value AND diagnostics for logging:
       {
         "ceiling_c":         float — upper bound for truncate_at_hi
         "fired_reason":      str   — why this fire happened
         "required_rate_c_per_h": float
         "remaining_hours":   int
         "cloud_pct":         float
         "plausible_rise_c":  float
       }
    Returns None when the ceiling should NOT fire (any of the three
    triggers misses, OR cloud data unavailable, OR forecast peak
    already passed, OR current_temp invalid).
    """
    if current_temp_c is None or current_temp_c <= -50:
        return None
    if cloud_pct is None:
        return None
    if current_hour < afternoon_hour_threshold:
        return None
    if cloud_pct < cloud_threshold_pct:
        return None

    # Hours until forecast peak — if peak has passed, no implausibility
    # to check (the day's high has either happened or is happening now).
    hours_to_peak = forecast_peak_hour - current_hour
    if hours_to_peak <= 0:
        return None

    required_rate = (forecast_high_c - current_temp_c) / hours_to_peak
    if required_rate <= implausible_rate_c_per_h:
        return None

    # All triggers fired — compute the ceiling.
    hours_remaining = max(0, sunset_hour - current_hour)
    plausible_rise = plausible_remaining_rise_c(hours_remaining, cloud_pct)
    ceiling_c = current_temp_c + plausible_rise + ceiling_buffer_c

    return {
        "ceiling_c":             round(ceiling_c, 2),
        "fired_reason":          (f"required_rate={required_rate:.2f}C/hr "
                                    f">{implausible_rate_c_per_h:.1f} "
                                    f"+ cloud={cloud_pct:.0f}% "
                                    f">{cloud_threshold_pct:.0f}%"),
        "required_rate_c_per_h": round(required_rate, 2),
        "remaining_hours":       hours_remaining,
        "cloud_pct":             round(cloud_pct, 1),
        "plausible_rise_c":      round(plausible_rise, 2),
    }


def is_rapid_model_trajectory_plateaued(
    trajectory: list[dict],
    plateau_tolerance_c: float = 0.3,
    minimum_horizon_h: int = 2,
) -> bool:
    """Return True if the rapid-update model's trajectory says the day
    has already topped out — i.e. the next `minimum_horizon_h` hours
    show NO rise above the current value by more than `plateau_tolerance_c`.

    This is the HRRR-driven replacement for the wall-clock
    `hours_since_observed_peak` narrowing trigger.  When True, the
    caller should accelerate the post-peak narrowing (start the σ
    contraction immediately and pull μ aggressively toward observed_max)
    regardless of clock time.

    Returns False if the trajectory is too short or shows continued rise.
    """
    if not trajectory or len(trajectory) < minimum_horizon_h:
        return False
    # Compare each subsequent reading to the current (first) hour's value.
    current_c = trajectory[0]["temp_c"]
    horizon = trajectory[:max(minimum_horizon_h, len(trajectory))]
    max_in_horizon = max(pt["temp_c"] for pt in horizon)
    return (max_in_horizon - current_c) <= plateau_tolerance_c


# ---------------------------------------------------------------------------
# Neighbor signal — adjust posterior based on upwind neighbor cooling
# ---------------------------------------------------------------------------

CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
def deg_to_cardinal(deg: float) -> str:
    return CARDINALS[int((deg + 22.5) / 45) % 8]


def vector_mean_dir(degrees: list[float]) -> float | None:
    if not degrees:
        return None
    u = sum(math.sin(math.radians(d)) for d in degrees) / len(degrees)
    v = sum(math.cos(math.radians(d)) for d in degrees) / len(degrees)
    if u == 0 and v == 0:
        return None
    return (math.degrees(math.atan2(u, v)) + 360) % 360


def compute_neighbor_signal(city: str, today_str: str,
                              afternoon_wind_octant: str | None) -> dict:
    """For US cities only — checks neighbor_obs.db for today's data from
    wind-aligned upwind neighbors.  Returns:
       { upwind_neighbors: [{sid, dir, peak_hr, peak_temp, current_temp,
                              cooling_from_peak_c}],
         strong_cooling_signal: bool,   # ≥1 upwind neighbor cooled ≥1°C
         signal_strength: float          # 0..1 score
       }"""
    if not os.path.exists(NEIGHBOR_DB) or not afternoon_wind_octant:
        return {"upwind_neighbors": [], "strong_cooling_signal": False,
                "signal_strength": 0.0}

    upwind: list[dict] = []
    with sqlite3.connect(NEIGHBOR_DB) as conn:
        conn.row_factory = sqlite3.Row
        # Neighbors in the direction the wind comes from = upwind
        nbrs = conn.execute(
            "SELECT sid, name, direction, distance_mi FROM neighbor_meta "
            "WHERE polymarket_city = ? AND direction = ?",
            (city, afternoon_wind_octant),
        ).fetchall()
        for n in nbrs:
            rows = conn.execute(
                "SELECT hour_local, temp_c FROM neighbor_obs "
                "WHERE sid = ? AND date_local = ? AND temp_c IS NOT NULL "
                "ORDER BY hour_local",
                (n["sid"], today_str),
            ).fetchall()
            if len(rows) < 4:
                continue
            peak_row = max(rows, key=lambda r: r["temp_c"])
            current = rows[-1]
            cooling = peak_row["temp_c"] - current["temp_c"]
            upwind.append({
                "sid":         n["sid"],
                "name":        n["name"],
                "direction":   n["direction"],
                "distance_mi": n["distance_mi"],
                "peak_hour":   peak_row["hour_local"],
                "peak_temp":   round(peak_row["temp_c"], 1),
                "current_hour": current["hour_local"],
                "current_temp": round(current["temp_c"], 1),
                "cooling_c":   round(cooling, 1),
                "cooling_since_peak_h": current["hour_local"] - peak_row["hour_local"],
            })

    strong = any(u["cooling_c"] >= 1.0 and u["cooling_since_peak_h"] >= 1
                  for u in upwind)
    # Signal strength: average cooling across upwind neighbors, capped at 1.0
    strength = 0.0
    if upwind:
        avg_cool = sum(max(0, u["cooling_c"]) for u in upwind) / len(upwind)
        strength = min(1.0, avg_cool / 3.0)   # 3°C cool = full signal
    return {
        "upwind_neighbors":      upwind,
        "strong_cooling_signal": strong,
        "signal_strength":       round(strength, 3),
    }


# ---------------------------------------------------------------------------
# Predictor — combines all sources into bin probabilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cooling-phase detection — combines 3 independent signals to determine
# whether today's daily high has already been observed.
# ---------------------------------------------------------------------------

# Confidence threshold above which we engage "bin-lock" (mu locked to
# the bin containing observed_max, sigma very narrow).
STRONG_COOLING_THRESHOLD = float(os.getenv("PREDICTOR_STRONG_COOLING_THRESHOLD", "0.7"))


def _cooling_via_obs_trajectory(obs_list: list[dict],
                                  observed_max: float,
                                  observed_peak_hour: int | None,
                                  current_hour: int) -> tuple[float, str]:
    """How many obs AFTER the observed peak have shown cooling?
    N-rule: need 2 cooling obs before 3pm local, 1 after.
    Returns (confidence ∈ [0,1], reasoning).
    """
    if observed_peak_hour is None or observed_peak_hour < 0:
        return 0.0, "no_obs_peak"
    obs_after_peak = [r for r in obs_list
                       if r["hour_local"] > observed_peak_hour]
    if not obs_after_peak:
        return 0.0, "no_obs_after_peak"
    # Cooling threshold: at least 0.3°C below peak (filters out noise)
    cooling_obs = [r for r in obs_after_peak
                    if r["temp_c"] < observed_max - 0.3]
    n_required = 2 if current_hour < 15 else 1
    if len(cooling_obs) < n_required:
        return 0.0, (f"need_{n_required}_cooling_obs_have_{len(cooling_obs)}")
    # Cap at 1.0; extra cooling obs don't add more confidence
    confidence = min(1.0, len(cooling_obs) / max(n_required, 1))
    return confidence, f"{len(cooling_obs)}_cooling_obs_after_peak"


def _cooling_via_forecast_tracking(obs_list: list[dict],
                                     forecast_hourly: list[tuple[int, float]],
                                     current_hour: int) -> tuple[float, str]:
    """Compare today's obs to NWS forecast trajectory.
    If obs are tracking the forecast AND forecast predicts cooling
    in the next 3hrs, we're in a high-confidence cooling phase.
    Returns (confidence ∈ [0,1], reasoning).
    """
    if not forecast_hourly or not obs_list:
        return 0.0, "no_forecast_or_obs"
    obs_by_hour = {r["hour_local"]: r["temp_c"] for r in obs_list}
    fcst_by_hour = dict(forecast_hourly)
    matched = [(h, obs_by_hour[h], fcst_by_hour[h])
                for h in obs_by_hour
                if h in fcst_by_hour and h <= current_hour]
    if len(matched) < 3:
        return 0.0, "insufficient_overlap"
    residuals = [obs - fcst for _, obs, fcst in matched]
    residual_std = (
        math.sqrt(sum((r - sum(residuals)/len(residuals))**2 for r in residuals)
                   / (len(residuals) - 1))
        if len(residuals) > 1 else 0.0
    )
    # Tracking quality: 1.0 means perfect, drops to 0 as residual_std → 2°C
    tracking_quality = max(0.0, 1.0 - residual_std / 2.0)
    # Look ahead 3 hours in forecast
    future = sorted([(h, t) for h, t in forecast_hourly
                      if h > current_hour])[:3]
    if not future:
        return 0.0, "no_forecast_ahead"
    current_temp = obs_by_hour.get(current_hour) or matched[-1][1]
    cooling_rate = current_temp - future[-1][1]    # positive = cooling
    if cooling_rate < 0.5:                          # < 0.5°C drop in 3hr
        return 0.0, "forecast_not_cooling"
    confidence = tracking_quality * min(1.0, cooling_rate / 2.0)
    return confidence, (f"fcst_cooling_{cooling_rate:+.1f}C_in_3h "
                        f"tracking_std_{residual_std:.2f}C")


def _cooling_via_derivative(obs_list: list[dict],
                              current_hour: int) -> tuple[float, str]:
    """Fit quadratic to last 5 hourly obs, check derivative at current hour.
    Catches cooling when forecast is wrong but obs clearly show it.
    Returns (confidence ∈ [0,1], reasoning).
    """
    recent = sorted(obs_list, key=lambda r: r["hour_local"])[-5:]
    if len(recent) < 4:
        return 0.0, "insufficient_obs_for_fit"
    hours = [r["hour_local"] for r in recent]
    temps = [r["temp_c"] for r in recent]
    # Simple linear regression — robust enough for 4-5 obs.  Slope = derivative.
    n = len(hours)
    mean_h = sum(hours) / n
    mean_t = sum(temps) / n
    num = sum((h - mean_h) * (t - mean_t) for h, t in zip(hours, temps))
    den = sum((h - mean_h) ** 2 for h in hours)
    if den < 1e-9:
        return 0.0, "no_hour_variance"
    slope = num / den    # °C per hour
    if slope >= -0.2:    # not cooling fast enough to confirm
        return 0.0, f"slope_{slope:+.2f}C/hr_not_cooling"
    # Map slope to confidence: -0.2 → 0, -1.0 → 1.0
    confidence = min(1.0, abs(slope + 0.2) / 0.8)
    return confidence, f"slope_{slope:+.2f}C/hr"


def detect_cooling(obs_list: list[dict],
                    forecast_hourly: list[tuple[int, float]],
                    observed_max: float,
                    observed_peak_hour: int | None,
                    current_hour: int) -> tuple[float, str]:
    """Combined cooling-phase confidence ∈ [0,1].

    Weighted average of three independent signals:
      - obs_trajectory (weight 0.5): cooling obs after peak, N-rule gated
      - forecast_tracking (weight 0.3): NWS forecast predicts cooling AND
        today's obs are tracking the forecast closely
      - derivative (weight 0.2): linear fit to recent obs has negative slope

    Confidence ≥ STRONG_COOLING_THRESHOLD (default 0.7) → engage bin-lock
    in predict_bins.
    """
    obs_c, obs_r = _cooling_via_obs_trajectory(
        obs_list, observed_max, observed_peak_hour, current_hour)
    fcst_c, fcst_r = _cooling_via_forecast_tracking(
        obs_list, forecast_hourly, current_hour)
    deriv_c, deriv_r = _cooling_via_derivative(obs_list, current_hour)
    weights = {"obs": 0.5, "fcst": 0.3, "deriv": 0.2}
    confidence = (weights["obs"]   * obs_c
                + weights["fcst"]  * fcst_c
                + weights["deriv"] * deriv_c)
    reason = f"obs={obs_c:.2f}({obs_r}) fcst={fcst_c:.2f}({fcst_r}) deriv={deriv_c:.2f}({deriv_r})"
    return confidence, reason


def find_bin_containing_temp(temp_c: float, bins: list[dict]) -> dict | None:
    """Return the bin (from a list of Polymarket bin dicts) whose Celsius
    range contains temp_c.  Used by bin-lock when cooling is confirmed."""
    for b in bins:
        lo, hi = bin_temp_range(b)
        if lo is not None and temp_c < lo:
            continue
        if hi is not None and temp_c >= hi:
            continue
        return b
    return None


def estimate_day_high_dist(forecast_high: float, forecast_peak_hour: int,
                            observed_max: float, observed_peak_hour: int,
                            current_hour: int, sunset_hour: int,
                            neighbor_signal: dict,
                            base_sigma_c: float | None = None,
                            ensemble_stats: dict | None = None,
                            cooling_confidence: float = 0.0,
                            hrrr_remaining_max: float | None = None,
                            hrrr_plateau_signal: bool = False,
                            ) -> tuple[float, float]:
    """Returns (mu, sigma) of the day-high distribution after all adjustments.

    base_sigma_c: per-city σ from calibration (RMSE of Open-Meteo vs observed
    over the last ~60 days).  Falls back to DEFAULT_FORECAST_SIGMA_C if None.

    ensemble_stats: optional dict from compute_ensemble_stats() containing
    multi-station forecast consensus.  When provided:
      * Inflates sigma if neighbor forecasts disagree (high std)
      * Blends mu toward ensemble median if settlement is an outlier
    None = ensemble disabled (legacy single-station behavior).

    cooling_confidence: ∈ [0,1] from detect_cooling().  When non-zero,
    reduces the FIX 1a "+0.3°C buffer" proportionally — at cooling=1.0
    the buffer disappears and mu locks exactly at observed_max.  Also
    narrows sigma proportionally on top of all other narrowing branches.
    Bin-lock (mu = bin_center) is applied separately in predict_bins
    when cooling_confidence >= STRONG_COOLING_THRESHOLD.

    hrrr_remaining_max: optional fresh CAM (HRRR / ICON-D2) projection
    of remaining-day max temperature.  When provided AND lower than the
    morning forecast by >0.5°C, μ recenters from forecast_high toward
    max(observed_max, hrrr_remaining_max).  This is the skepticism
    mechanism: HRRR has assimilated today's actual conditions; if it
    disagrees with the morning forecast, prefer it.  Spec §3.2.

    hrrr_plateau_signal: True when HRRR's next 2-3h trajectory shows
    no further rise (the day is done climbing).  When True, the
    day_has_likely_peaked gate fires regardless of wall-clock — closes
    the lag in the existing wall-clock branch.  Wall-clock stays as
    fallback for cold-start days and non-CAM cities where HRRR isn't
    available.  Spec §3.4.
    """
    mu = forecast_high
    sigma = base_sigma_c if base_sigma_c is not None else DEFAULT_FORECAST_SIGMA_C
    sigma_ceiling = sigma   # for later clamps that reference the prior

    # === HRRR μ RECENTER (skepticism mechanism) ===
    # When a fresh CAM run disagrees with the morning forecast by
    # > 0.5°C in the cooler direction, prefer the CAM.  It has
    # assimilated today's actual clouds / boundary-layer conditions
    # that the morning forecast was blind to.  This is the Atlanta /
    # NYC plateau case: morning forecast said 34°C, HRRR says
    # remaining-day max is 32.5°C, observed_max is 32.8°C.  μ should
    # be ~32.8 (HRRR-aligned), not 34 (forecast-anchored).
    #
    # We don't replace the existing observed_max-based pull (FIX 1a
    # below); HRRR sets the starting μ, then observed_max can still
    # override upward if obs exceeds HRRR.
    if (hrrr_remaining_max is not None
        and hrrr_remaining_max < forecast_high - 0.5):
        # HRRR is materially more pessimistic — recenter on it.
        # max(observed_max, hrrr_remaining_max) keeps the floor
        # honest: μ never lands below what's already been observed.
        obs_for_floor = observed_max if observed_max > -50 else -100.0
        mu = max(obs_for_floor, hrrr_remaining_max)

    # ENSEMBLE PHASE 2 — mu blending toward neighbor consensus.  Done
    # BEFORE all the time-of-day adjustments so the obs-based logic
    # works on the corrected mu.
    if ensemble_stats and ensemble_stats.get("n_stations_used", 1) >= 3:
        divergence = ensemble_stats.get("divergence_c", 0.0)
        if abs(divergence) > 1.5:
            # Settlement station is an outlier — partial pull toward
            # ensemble median.  Capped at 50% blend so we never let
            # neighbors override settlement entirely.
            median = ensemble_stats["ensemble_median"]
            blend = min(0.5, (abs(divergence) - 1.5) / 5.0 + 0.2)
            mu = (1 - blend) * mu + blend * median

    # FIX 1a: Observed already exceeded forecast.  Pull mu up to at
    # least observed, plus a buffer that:
    #   - tapers as we approach/pass forecast_peak_hour (Proposal B)
    #   - shrinks proportionally with cooling_confidence (Proposal A)
    #
    # The legacy +0.3°C buffer was time-blind and caused the Houston bug:
    # at 5pm with the day cooling, +0.3°C inflated mu just above a bin
    # boundary, spilling probability into the wrong bin.
    if observed_max > -50 and observed_max > forecast_high:
        excess = observed_max - forecast_high
        # Tapered base buffer
        hours_to_peak = forecast_peak_hour - current_hour
        if hours_to_peak <= 0:
            # Past forecast peak: small residual floor at 0.1°C, taper
            # linearly with each hour past peak.
            base_buffer = max(0.1, 0.3 + hours_to_peak * 0.1)
        else:
            base_buffer = 0.3
        # Cooling-confidence override: at high cooling confidence, the
        # buffer disappears entirely (mu = observed_max).
        effective_buffer = base_buffer * (1.0 - cooling_confidence)
        mu = observed_max + effective_buffer
        sigma = min(sigma_ceiling, max(1.0, sigma - 0.3 + excess * 0.2))

    # Narrow uncertainty as we pass the forecast peak hour
    if current_hour >= forecast_peak_hour:
        hours_past = current_hour - forecast_peak_hour
        narrow = max(0.30, 1.0 - hours_past * 0.12)
        sigma *= narrow

        # FIX 1b: Blend mu toward observed_max in BOTH directions
        if hours_past >= 1 and observed_max > -50:
            blend = min(0.8, hours_past * 0.15)
            mu = (1 - blend) * mu + blend * observed_max

    # Time-since-OBSERVED-peak narrowing.  Once we've actually seen the
    # peak and a few hours have passed without it being exceeded, the
    # day-high is essentially locked at observed_max.  Empirically (from
    # the temp-drop backtest): hold rate is 84% at 1h past peak, 97% at
    # 2h, 99%+ at 3h+.  We narrow sigma aggressively to reflect this.
    #
    # BUG FIX (2026-06-10): this branch used to fire ANY time observations
    # existed, even before the day's forecasted peak.  Result: a 10am
    # morning reading was treated as "the peak", aggressively pulling
    # mu toward 80°F when forecast was 87°F at 4pm.  Now gated on
    # "day has plausibly peaked": either we're past forecast_peak_hour,
    # or observed_max has actually reached near forecast_high.  If the
    # day's still warming, the morning observation is a LOWER BOUND
    # (handled by truncation in predict_bins), not a "peak."
    day_has_likely_peaked = (
        current_hour >= forecast_peak_hour            # past forecasted peak
        or (observed_max > -50
            and observed_max >= forecast_high - 1.0)  # obs reached forecast
        or hrrr_plateau_signal                        # HRRR: trajectory flat
    )
    if (observed_max > -50
        and observed_peak_hour is not None
        and observed_peak_hour >= 0
        and day_has_likely_peaked):
        hours_since_obs_peak = current_hour - observed_peak_hour
        # FIX B (Quick-Fix-B 2026-06-12): trigger at hours_since >= 0 by
        # default (PREDICTOR_IMMEDIATE_POST_PEAK_NARROW=1).  When disabled,
        # falls back to >= 1 (legacy behavior).  See constant docstring.
        b_threshold = 0 if PREDICTOR_IMMEDIATE_POST_PEAK_NARROW else 1
        if hours_since_obs_peak >= b_threshold:
            # Geometric narrowing: 0.7^h
            #   0h → 1.00x (no narrowing yet), 1h → 0.70x, 2h → 0.49x,
            #   3h → 0.34x, 4h+ → 0.24x (floor)
            narrow = max(0.20, 0.7 ** hours_since_obs_peak)
            sigma *= narrow
            # And pull mu toward observed (since the longer it's been, the
            # more likely observed IS the day high)
            blend = min(0.9, hours_since_obs_peak * 0.25)
            mu = (1 - blend) * mu + blend * observed_max

    # After sunset, the day high is essentially locked at observed_max
    if current_hour >= sunset_hour:
        mu = observed_max if observed_max > -50 else mu
        sigma = max(0.3, sigma * 0.2)

    # Upwind cooling neighbor signal — narrows sigma and pulls mu toward observed
    if neighbor_signal.get("strong_cooling_signal"):
        strength = neighbor_signal.get("signal_strength", 0.0)
        narrow = 1.0 - 0.3 * strength
        sigma *= narrow
        if observed_max > -50:
            blend = 0.2 * strength
            mu = (1 - blend) * mu + blend * observed_max

    # ENSEMBLE PHASE 1 — sigma inflation based on neighbor disagreement.
    # Done AFTER all narrowing branches: if 5 nearby stations disagree
    # by ±2°C, our base_sigma_c (which was calibrated for one station's
    # forecast error) is too tight.  Multiplicative so we don't inflate
    # away the post-peak narrowing — we just acknowledge the residual
    # regional uncertainty.
    #   ensemble_std < 0.5°C: forecasts agree → no inflation
    #   ensemble_std = 1.0°C: +30%
    #   ensemble_std = 2.0°C: +100%
    #   capped at 3x the base
    if ensemble_stats and ensemble_stats.get("n_stations_used", 1) >= 3:
        std = ensemble_stats.get("ensemble_std", 0.0)
        if std > 0.5:
            inflation = max(1.0, 1.0 + (std - 0.5) * 0.6)
            sigma = min(sigma * inflation, sigma_ceiling * 3.0)

    # COOLING confidence sigma narrowing.  When cooling is confidently
    # detected, the day's remaining uncertainty collapses — the high is
    # essentially the observed peak.  Narrows sigma proportionally on
    # top of any other narrowing.  Capped at 50% reduction so we don't
    # double-narrow with the post-peak branches.
    if cooling_confidence > 0:
        extra_narrow = 1.0 - cooling_confidence * 0.5
        sigma *= extra_narrow

    # σ floor — see PREDICTOR_SIGMA_FLOOR_C constant docs.  Stops the
    # cascade of narrowing branches from collapsing σ below what Q2's
    # realism check shows is the calibrated lower bound.
    #
    # Carve-out: post-sunset the day is locked at observed_max, so σ
    # SHOULD stay collapsed (we know the day's high already).  Apply
    # the wider floor pre-sunset only; post-sunset retains the legacy
    # 0.3°C floor representing pure station-level measurement noise.
    if current_hour < sunset_hour:
        sigma = max(PREDICTOR_SIGMA_FLOOR_C, sigma)
    else:
        sigma = max(0.3, sigma)
    return mu, sigma


def bin_temp_range(bin_dict: dict) -> tuple[float | None, float | None]:
    """Translate a Polymarket bin (range_low, range_high) into the actual
    real-line temperature range (in °C) it represents on settlement.

    Polymarket uses integer-labeled bins with half-up rounding.  E.g.:
      single bin '28°C' = actual in [27.5, 28.5) — width 1°C
      2°F bin '88-89°F' = actual in [87.5, 89.5)°F — width 2°F
      '23°C or below'   = actual ≤ 23.5°C — open downward
      '33°C or higher'  = actual ≥ 32.5°C — open upward

    Output is in CELSIUS regardless of input unit, since our forecast and
    observation streams are all Celsius.
    """
    lo = bin_dict.get("range_low")
    hi = bin_dict.get("range_high")
    unit = (bin_dict.get("unit") or "celsius").lower()
    is_f = unit == "fahrenheit"

    def to_c(v):
        return (v - 32) * 5 / 9 if is_f else v

    # Half-step in the bin's NATIVE unit, then converted.  For F bins this
    # is 0.5°F which is ~0.28°C — applied to the labelled boundary BEFORE
    # converting to Celsius keeps the half-step semantics correct.
    if lo is None and hi is not None:
        # "X°? or below" → actual <= X + 0.5 (native units)
        return (None, to_c(hi + 0.5))
    if lo is not None and hi is None:
        # "X°? or higher" → actual >= X - 0.5 (native units)
        return (to_c(lo - 0.5), None)
    if lo is not None and hi is not None:
        # FIX 2: Whether it's a single-integer bin ('28°C', lo==hi) or a
        # multi-step bin ('88-89°F', lo!=hi), the actual-range endpoints
        # are lo-0.5 and hi+0.5 in the bin's native unit.  Previously a
        # multi-step bin was treated as (lo, hi) without the half-step
        # padding, which under-counted probability at the boundaries.
        return (to_c(lo - 0.5), to_c(hi + 0.5))
    return (None, None)


def predict_bins(event: dict, settlement_obs: list[dict],
                  forecast: dict, neighbor_signal: dict,
                  current_hour: int, city: str | None = None,
                  ensemble_stats: dict | None = None,
                  cold_start_suspect: bool = False,
                  lat: float | None = None,
                  lon: float | None = None,
                  tz_str: str | None = None) -> dict:
    """Per-event bin probability + edge + recommendation.

    ensemble_stats: optional output from compute_ensemble_stats() — when
    provided, the day-high mu/sigma estimation uses multi-station consensus
    for outlier detection (mu) and uncertainty calibration (sigma).

    cold_start_suspect: when True, the HRRR ceiling dispatch is skipped
    entirely (HRRR's "remaining hours" view can't recover a peak that
    already happened before the bot's first scan).  Falls back to the
    floor-only distribution path.  Computed by the scan loop's cold-
    start detector in scheduled_predictor.

    lat, lon, tz_str: settlement station coordinates and IANA timezone,
    used to fetch the rapid-update CAM run.  Required for HRRR/ICON-D2
    fetch; when None the HRRR ceiling dispatch is skipped.
    """
    bins = event.get("outcomes") or []
    if not bins or not forecast:
        return {"bins": [], "skipped": "no_data"}

    # Observed max in Celsius
    if settlement_obs:
        observed_max_c = max(r["temp_c"] for r in settlement_obs)
        observed_peak_hour = max(r["hour_local"] for r in settlement_obs
                                   if r["temp_c"] == observed_max_c)
    else:
        # No live data — fall back to "the day hasn't started" assumption.
        # Use forecast morning low.
        observed_max_c = -100   # nothing observed yet → no truncation
        observed_peak_hour = -1

    base_sigma = get_city_sigma(city) if city else DEFAULT_FORECAST_SIGMA_C

    # Apply per-station forecast bias correction.  If a station's NWS
    # forecast historically runs 1.5°C HOT (overpredicts), we subtract
    # 1.5°C from the input forecast.  Calibrated by
    # scripts.station_bias_calibration from saved historical data.
    settlement_station = event.get("settlement_station") or ""
    bias_c = get_station_bias(settlement_station) if settlement_station else 0.0
    bias_corrected_forecast = forecast["forecast_high"] - bias_c

    # COOLING DETECTION — combine 3 independent signals to decide if the
    # day's high has already been observed and we're cooling.  Drives
    # both mu/sigma adjustments (inside estimate_day_high_dist) and the
    # bin-lock override below.
    cooling_confidence, cooling_reason = detect_cooling(
        obs_list           = settlement_obs,
        forecast_hourly    = forecast.get("hourly") or [],
        observed_max       = observed_max_c if observed_max_c > -100 else bias_corrected_forecast,
        observed_peak_hour = observed_peak_hour if observed_peak_hour >= 0 else None,
        current_hour       = current_hour,
    )

    # === HRRR ceiling dispatch (Phase 1, behind flag) ===
    # When PREDICTOR_USE_HRRR_CEILING=1 and the city has a same_day_model
    # assigned and the event isn't flagged cold-start, fetch the fresh
    # CAM run.  Returns None on any failure → falls through to the
    # legacy (HRRR-unaware) path.
    hrrr_remaining_max: float | None = None
    hrrr_plateau_signal = False
    hrrr_used = False
    if (PREDICTOR_USE_HRRR_CEILING
        and not cold_start_suspect
        and city is not None
        and lat is not None and lon is not None and tz_str is not None):
        try:
            from station_meta import get_same_day_model  # type: ignore
            model = get_same_day_model(city)
        except Exception:
            model = None
        if model is not None:
            try:
                hrrr_data = fetch_rapid_model_remaining_max(
                    lat, lon, tz_str, model)
            except Exception as e:
                log.warning(f"HRRR fetch raised for {city}: {e}")
                hrrr_data = None
            if hrrr_data and _hrrr_data_passes_sanity(hrrr_data,
                                                        observed_max_c,
                                                        bias_corrected_forecast):
                hrrr_remaining_max = hrrr_data["remaining_max_c"]
                hrrr_plateau_signal = is_rapid_model_trajectory_plateaued(
                    hrrr_data["trajectory"])
                hrrr_used = True
                log.debug(f"HRRR {model} for {city}: "
                          f"remaining_max={hrrr_remaining_max:.2f}°C, "
                          f"plateau={hrrr_plateau_signal}")

    mu, sigma = estimate_day_high_dist(
        forecast_high       = bias_corrected_forecast,
        forecast_peak_hour  = forecast["forecast_peak_hour"],
        observed_max        = observed_max_c if observed_max_c > -100 else forecast["forecast_high"],
        observed_peak_hour  = observed_peak_hour if observed_peak_hour >= 0 else None,
        current_hour        = current_hour,
        sunset_hour         = forecast["sunset_hour"],
        neighbor_signal     = neighbor_signal,
        base_sigma_c        = base_sigma,
        ensemble_stats      = ensemble_stats,
        cooling_confidence  = cooling_confidence,
        hrrr_remaining_max  = hrrr_remaining_max,
        hrrr_plateau_signal = hrrr_plateau_signal,
    )

    truncate_at = observed_max_c if observed_max_c > -100 else -100.0

    # BIN-LOCK: when cooling is STRONGLY confirmed, the day's high is
    # essentially the observed peak.  Override the truncated-normal
    # mu/sigma with the bin geometry containing observed_max.  This
    # bypasses the boundary-effect bug where mu sits right at a bin
    # edge and probability spills into the next bin up.
    bin_locked = False
    if (cooling_confidence >= STRONG_COOLING_THRESHOLD
        and observed_max_c > -100):
        obs_bin = find_bin_containing_temp(observed_max_c, bins)
        if obs_bin is not None:
            bin_lo_c, bin_hi_c = bin_temp_range(obs_bin)
            # For open-ended bins (≤X or ≥X), can't compute a clean
            # center — fall back to non-bin-lock behavior.
            if bin_lo_c is not None and bin_hi_c is not None:
                mu = (bin_lo_c + bin_hi_c) / 2          # bin center
                sigma = max(0.15, (bin_hi_c - bin_lo_c) / 6)  # 3σ each side
                # Relax truncation by 0.1°C to admit measurement uncertainty
                # in the observation (NWS reports in 0.1°C precision).
                truncate_at = bin_lo_c - 0.1
                bin_locked = True
                log.debug(
                    f"BIN-LOCK ({cooling_reason}): obs_bin={obs_bin.get('range_low')}"
                    f"-{obs_bin.get('range_high')}{obs_bin.get('unit')} "
                    f"mu={mu:.2f}C sigma={sigma:.2f}C trunc={truncate_at:.2f}C"
                )

    # Construct the day-high CDF once per event.  Dispatch lets future
    # workstreams plug in empirical / NBM CDFs without changing the bin
    # loop.  Phase A: gaussian.  Phase C: empirical residual.
    cdf_choice = PREDICTOR_CDF_IMPL
    cdf_used   = "gaussian"   # what we actually fell back to (for diagnostics)
    if cdf_choice == "empirical":
        residuals = get_city_centered_residuals(city) if city else None
        if residuals:
            # Scale the empirical width by sigma/base_sigma so the
            # late-day wall-clock narrowing from estimate_day_high_dist
            # carries through.  base_sigma was computed earlier in this
            # function (see above) — using the same value as the
            # gaussian path's starting sigma.
            scale = (sigma / base_sigma) if base_sigma > 0 else 1.0
            day_high_cdf = make_empirical_residual_cdf(mu, residuals, scale=scale)
            cdf_used = "empirical"
        else:
            # Insufficient samples or no calibration entry — fall back
            # rather than ship a bad distribution.  W3's data-quality
            # flag (paper_predictor_signals.data_quality_flag) will
            # eventually carry this fact to the dashboard.
            day_high_cdf = make_gaussian_cdf(mu, sigma)
    elif cdf_choice == "gaussian":
        day_high_cdf = make_gaussian_cdf(mu, sigma)
    else:
        log.warning(f"unknown PREDICTOR_CDF_IMPL={cdf_choice!r}, "
                     f"falling back to gaussian")
        day_high_cdf = make_gaussian_cdf(mu, sigma)

    # Upper truncation from the HRRR ceiling (Phase 1 of the HRRR plan).
    # When HRRR is active and passed sanity, cap the distribution at
    # `hrrr_remaining_max + ceiling_buffer`.  Buffer is intentionally
    # generous (default 1.0°C, asymmetric-loss reasoning per W3 spec):
    # clipping a bin that wins is a total loss; leaving slightly too
    # much upper tail is a mild misize.
    #
    # When HRRR isn't active, truncate_at_hi stays None and the
    # distribution behaves exactly as it did pre-HRRR.
    truncate_at_hi = None
    if hrrr_used and hrrr_remaining_max is not None:
        truncate_at_hi = hrrr_remaining_max + PREDICTOR_HRRR_CEILING_BUFFER_C

    # Quick Fix C — plausibility ceiling.  Crude prototype of the
    # HRRR/W3 physical ceiling using NWS skyCover already in our
    # fetch path.  Fires only on clearly-implausible heating regimes
    # (cloudy afternoons where bot would otherwise bet >2°C/hour rise).
    # Cold-start skip enforced: peak may already have happened.
    # Combined with HRRR via min() — most conservative ceiling wins.
    plausibility_info = None
    if (PREDICTOR_USE_PLAUSIBILITY_CEILING
        and not cold_start_suspect
        and observed_max_c > -50):
        # Current temp = the latest reading from our settlement obs.
        # We use observed_max as a conservative proxy (truth's at least
        # this hot) — for the plausibility check what matters is the
        # delta to forecast_high.
        cloud_avg = avg_remaining_cloud_pct(
            forecast.get("hourly_clouds") or {},
            current_hour,
            forecast["sunset_hour"])
        plausibility_info = compute_plausibility_ceiling_c(
            current_temp_c=observed_max_c,
            forecast_high_c=bias_corrected_forecast,
            forecast_peak_hour=forecast["forecast_peak_hour"],
            current_hour=current_hour,
            sunset_hour=forecast["sunset_hour"],
            cloud_pct=cloud_avg,
            ceiling_buffer_c=PREDICTOR_CEILING_BUFFER_C,
            implausible_rate_c_per_h=PREDICTOR_IMPLAUSIBLE_RATE_C_PER_H,
            afternoon_hour_threshold=PREDICTOR_AFTERNOON_HOUR_THRESHOLD,
            cloud_threshold_pct=PREDICTOR_CLOUD_THRESHOLD_PCT,
        )
        if plausibility_info is not None:
            plausibility_ceiling = plausibility_info["ceiling_c"]
            # Log every fire so the HRRR/W3 work has data on whether
            # the heuristic pointed the right direction, and any
            # clipped winner is traceable back to a specific cap.
            log.info(
                f"plausibility ceiling fired for {city or '?'}: "
                f"ceiling={plausibility_ceiling:.2f}°C "
                f"({plausibility_info['fired_reason']}, "
                f"plausible_rise={plausibility_info['plausible_rise_c']:.2f}°C, "
                f"observed={observed_max_c:.2f}°C, "
                f"forecast={bias_corrected_forecast:.2f}°C)"
            )
            # Combine with HRRR ceiling via min — both signals can
            # coexist; most conservative wins.
            if truncate_at_hi is None:
                truncate_at_hi = plausibility_ceiling
            else:
                truncate_at_hi = min(truncate_at_hi, plausibility_ceiling)

    bin_results = []
    for b in bins:
        c_lo, c_hi = bin_temp_range(b)
        # truncate_at_lo: observed_max (the rising floor).
        # truncate_at_hi: HRRR ceiling when active, else None (legacy).
        p = probability_in_bin(
            c_lo, c_hi, day_high_cdf,
            truncate_at_lo=truncate_at,
            truncate_at_hi=truncate_at_hi,
        )
        market_p = float(b.get("yes_price") or 0)
        edge = p - market_p
        bin_results.append({
            "contract_id":   b.get("contract_id"),
            "yes_token_id":  b.get("yes_token_id"),
            "range_low":     b.get("range_low"),
            "range_high":    b.get("range_high"),
            "unit":          b.get("unit"),
            "c_lo":          round(c_lo, 2) if c_lo is not None else None,
            "c_hi":          round(c_hi, 2) if c_hi is not None else None,
            "label":         _bin_label(b.get("range_low"), b.get("range_high"), b.get("unit")),
            "our_prob":      round(p, 4),
            "market_prob":   round(market_p, 4),
            "edge":          round(edge, 4),
            "liquidity_usd": round(float(b.get("liquidity_usd") or 0), 0),
            # Per-bin market resolution flag (True = market has settled).
            # Lets the scan loop and dashboard answer "is this market still
            # tradeable?" without inferring it from position value.
            "closed":        bool(b.get("closed", False)),
        })
    return {
        "bins":               bin_results,
        "mu":                 round(mu, 2),
        "sigma":              round(sigma, 2),
        "cdf_used":           cdf_used,           # "gaussian" | "empirical"
        "hrrr_used":          hrrr_used,
        "hrrr_remaining_max": hrrr_remaining_max,
        "hrrr_plateau_signal": hrrr_plateau_signal,
        "plausibility_ceiling_fired": plausibility_info is not None,
        "plausibility_info":  plausibility_info,
        "truncate_at_hi":     truncate_at_hi,
        "cooling_confidence": round(cooling_confidence, 3),
        "cooling_reason":     cooling_reason,
        "bin_locked":         bin_locked,
        "observed_max_c":     round(observed_max_c, 2) if observed_max_c > -100 else None,
        "observed_peak_hour": observed_peak_hour if observed_peak_hour >= 0 else None,
        "forecast_high":      round(forecast["forecast_high"], 2),
        "forecast_peak_hour": forecast["forecast_peak_hour"],
        "sunset_hour":        forecast["sunset_hour"],
    }


def _bin_label(lo, hi, unit) -> str:
    suffix = "F" if (unit or "celsius").lower() == "fahrenheit" else "C"
    if lo is None and hi is not None: return f"≤{int(hi)}°{suffix}"
    if lo is not None and hi is None: return f"≥{int(lo)}°{suffix}"
    if lo is not None and hi is not None:
        if int(lo) == int(hi): return f"{int(lo)}°{suffix}"
        return f"{int(lo)}–{int(hi)}°{suffix}"
    return "?"


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #0f172a; color: #e2e8f0; font-size: 13px; }
a { color: #818cf8; }
header { background: linear-gradient(90deg, #1e293b, #0f172a);
         padding: 16px 24px; border-bottom: 1px solid #334155;
         display: flex; align-items: center; justify-content: space-between; }
header h1 { margin: 0; font-size: 18px; font-weight: 600; color: white; }
header .meta { font-family: monospace; font-size: 11px; color: #94a3b8; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 8px; padding: 12px 24px; background: #0f172a;
        border-bottom: 1px solid #1e293b; }
.kpi { background: #1e293b; padding: 10px 14px; border-radius: 6px;
       border-left: 3px solid #4338ca; }
.kpi .label { font-size: 9px; color: #94a3b8; text-transform: uppercase;
              letter-spacing: 0.5px; font-weight: 600; }
.kpi .val { font-size: 22px; font-weight: 700; margin-top: 2px; font-family: monospace;
            color: white; }
.kpi.pos { border-left-color: #22c55e; }
.kpi.pos .val { color: #4ade80; }
.kpi.neg { border-left-color: #ef4444; }
.kpi.neg .val { color: #f87171; }
.city-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
             gap: 10px; padding: 12px 24px; }
.city-card { background: #1e293b; border-radius: 8px; padding: 12px 14px;
             border-top: 3px solid #4338ca; cursor: pointer;
             transition: transform 0.15s; }
.city-card:hover { transform: translateY(-2px); background: #273449; }
.city-card.has-buy { border-top-color: #22c55e; }
.city-card.no-data { opacity: 0.6; border-top-color: #475569; }
.city-card .city-name { font-weight: 700; font-size: 14px; color: white;
                         display: flex; justify-content: space-between; align-items: center; }
.city-card .station { font-size: 10px; color: #94a3b8; font-family: monospace; }
.city-card .row { display: flex; justify-content: space-between; margin: 4px 0;
                  font-size: 12px; }
.city-card .row .lbl { color: #94a3b8; }
.city-card .row .v { font-family: monospace; color: #e2e8f0; }
.city-card .row .v.hot { color: #fbbf24; }
.city-card .buys { display: inline-block; padding: 2px 8px; border-radius: 10px;
                    background: #22c55e; color: white; font-size: 10px;
                    font-weight: 700; }
.filters { background: #1e293b; padding: 10px 24px; display: flex;
           gap: 18px; flex-wrap: wrap; align-items: center; font-size: 12px;
           border-bottom: 1px solid #334155; position: sticky; top: 0; z-index: 10; }
.filters label { font-weight: 600; color: #cbd5e1; margin-right: 4px; }
.filters select, .filters input { padding: 4px 8px; font-size: 12px;
           background: #0f172a; color: white; border: 1px solid #475569;
           border-radius: 4px; }
.filters .count { color: #94a3b8; font-family: monospace; margin-left: auto; }
table { width: calc(100% - 48px); margin: 12px 24px; background: #1e293b;
        border-collapse: collapse; border-radius: 6px; overflow: hidden;
        font-size: 12px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 10px;
     text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer;
     user-select: none; }
th:hover { background: #1e293b; color: white; }
th.sorted-asc::after { content: ' ▲'; color: #818cf8; }
th.sorted-desc::after { content: ' ▼'; color: #818cf8; }
tr.buy { background: rgba(34, 197, 94, 0.12); }
tr.buy:hover { background: rgba(34, 197, 94, 0.20); }
tr.skip { color: #64748b; }
tr.avoid { background: rgba(239, 68, 68, 0.08); color: #cbd5e1; }
td.num { font-family: monospace; text-align: right; }
td.bin-label { font-weight: 600; color: white; }
td .pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 10px; font-weight: 700; }
.pill.buy { background: #22c55e; color: white; }
.pill.skip { background: #475569; color: #cbd5e1; }
.pill.avoid { background: #ef4444; color: white; }
.bar { display: inline-block; width: 100px; height: 8px; background: #334155;
       border-radius: 4px; overflow: hidden; vertical-align: middle; }
.bar .fill { height: 100%; background: #818cf8; }
.bar .fill.market { background: #fbbf24; }
.bar .fill.edge-pos { background: #22c55e; }
.bar .fill.edge-neg { background: #ef4444; }
.edge-cell { font-family: monospace; font-weight: 700; }
.edge-cell.pos { color: #4ade80; }
.edge-cell.neg { color: #f87171; }
.section-title { padding: 18px 24px 8px; font-size: 11px; font-weight: 700;
                  color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.empty { text-align: center; color: #64748b; padding: 40px 0; font-size: 14px; }
"""


def render_dashboard(predictions: list[dict], cities_summary: list[dict],
                      kpis: dict, generated_at: str, out_path: str) -> None:
    bins_data = []
    for p in predictions:
        for b in p.get("bins", []):
            edge = b["edge"]
            rec = "BUY" if (edge >= 0.10 and b["market_prob"] < 0.95 and b["liquidity_usd"] >= 300) \
                  else ("AVOID" if edge <= -0.20 else "SKIP")
            bins_data.append({
                "city":         p["city"],
                "date":         p["date"],
                "station":      p["station"],
                "bin":          b["label"],
                "our_prob":     b["our_prob"],
                "market_prob":  b["market_prob"],
                "edge":         edge,
                "liquidity":    b["liquidity_usd"],
                "recommend":    rec,
                "contract_id":  b["contract_id"],
            })

    bins_json   = json.dumps(bins_data, default=str, separators=(",", ":"))
    cities_json = json.dumps(cities_summary, default=str, separators=(",", ":"))

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Polymarket Weather — Intraday Bin Predictor</title>
<style>{DASHBOARD_CSS}</style></head><body>

<header>
  <div>
    <h1>Polymarket Weather — Intraday Bin Predictor</h1>
    <div class='meta' style='color:#94a3b8;font-size:11px;margin-top:2px'>
      Settlement obs: NWS API · Forecast prior: Open-Meteo · Neighbor signal: cached neighbor_obs.db
    </div>
  </div>
  <div class='meta'>generated {generated_at}</div>
</header>

<div class='kpis'>
  <div class='kpi'><div class='label'>US cities</div><div class='val'>{kpis['n_cities']}</div></div>
  <div class='kpi'><div class='label'>Active events</div><div class='val'>{kpis['n_events']}</div></div>
  <div class='kpi'><div class='label'>Total bins</div><div class='val'>{kpis['n_bins']}</div></div>
  <div class='kpi pos'><div class='label'>BUY signals</div><div class='val'>{kpis['n_buys']}</div></div>
  <div class='kpi'><div class='label'>Avg buy edge</div>
    <div class='val'>{kpis['avg_buy_edge']:+.0%}</div></div>
  <div class='kpi pos'><div class='label'>Total potential edge $/$1</div>
    <div class='val'>${kpis['total_buy_edge']:+.2f}</div></div>
</div>

<div class='section-title'>City snapshots</div>
<div class='city-grid' id='city-grid'></div>

<div class='section-title'>All bins · filter and sort</div>
<div class='filters'>
  <div><label>City</label>
    <select id='f-city'><option value=''>All</option></select></div>
  <div><label>Date</label>
    <select id='f-date'><option value=''>All</option></select></div>
  <div><label>Recommend</label>
    <select id='f-rec'>
      <option value=''>All</option>
      <option value='BUY'>BUY only</option>
      <option value='SKIP'>SKIP only</option>
      <option value='AVOID'>AVOID only</option>
    </select></div>
  <div><label>Min edge</label>
    <input id='f-edge' type='number' step='0.05' min='-1' max='1' value='-1' style='width:70px'></div>
  <div><label>Min liquidity $</label>
    <input id='f-liq' type='number' step='100' min='0' value='0' style='width:80px'></div>
  <div class='count' id='count'>—</div>
</div>

<table id='bins'>
  <thead><tr>
    <th data-key='date'>Date</th>
    <th data-key='city'>City</th>
    <th data-key='bin'>Bin</th>
    <th data-key='our_prob'>Our P</th>
    <th data-key='market_prob'>Market P</th>
    <th data-key='edge'>Edge</th>
    <th data-key='liquidity'>Liquidity</th>
    <th data-key='recommend'>Recommend</th>
  </tr></thead>
  <tbody id='tbody'></tbody>
</table>

<script>
const BINS = {bins_json};
const CITIES = {cities_json};
const $ = id => document.getElementById(id);

// City grid render
function renderCities() {{
  const html = CITIES.map(c => {{
    const cls = c.n_buys > 0 ? 'has-buy' : (c.has_data ? '' : 'no-data');
    const hot = c.observed_max_c !== null && c.forecast_high !== null
                && c.observed_max_c >= c.forecast_high - 0.5;
    return `<div class='city-card ${{cls}}'>
      <div class='city-name'>
        <span>${{c.city}}</span>
        ${{c.n_buys > 0 ? `<span class='buys'>${{c.n_buys}} BUY</span>` : ''}}
      </div>
      <div class='station'>${{c.station}} · ${{c.tz_label || ''}}</div>
      <div class='row'><span class='lbl'>Now</span>
        <span class='v'>${{c.current_temp_c !== null ? c.current_temp_c.toFixed(1)+'°C' : '—'}}
          ${{c.current_wind_dir !== null ? '· wind ' + c.current_wind_dir : ''}}</span></div>
      <div class='row'><span class='lbl'>Observed high</span>
        <span class='v ${{hot ? 'hot' : ''}}'>${{c.observed_max_c !== null ? c.observed_max_c.toFixed(1)+'°C @ '+c.observed_peak_hour+':00' : '—'}}</span></div>
      <div class='row'><span class='lbl'>Forecast high</span>
        <span class='v'>${{c.forecast_high !== null ? c.forecast_high.toFixed(1)+'°C @ '+c.forecast_peak_hour+':00' : '—'}}</span></div>
      <div class='row'><span class='lbl'>Sunset</span>
        <span class='v'>${{c.sunset_hour !== null ? c.sunset_hour+':00' : '—'}}</span></div>
      ${{c.upwind_summary ? `<div class='row'><span class='lbl'>Upwind</span>
        <span class='v'>${{c.upwind_summary}}</span></div>` : ''}}
    </div>`;
  }}).join('');
  $('city-grid').innerHTML = html || `<div class='empty'>No city data</div>`;
}}

// Filter dropdowns
const cities = [...new Set(BINS.map(b => b.city))].sort();
for (const c of cities) {{
  const o = document.createElement('option'); o.value = o.textContent = c;
  $('f-city').appendChild(o);
}}
const dates = [...new Set(BINS.map(b => b.date))].sort();
for (const d of dates) {{
  const o = document.createElement('option'); o.value = o.textContent = d;
  $('f-date').appendChild(o);
}}

let SORT_KEY = 'edge', SORT_DIR = -1;

function row(b) {{
  const rec = b.recommend;
  const cls = rec.toLowerCase();
  const edgeCls = b.edge >= 0 ? 'pos' : 'neg';
  const edgeSign = b.edge >= 0 ? '+' : '';
  const ourBar = Math.round(Math.max(0, Math.min(1, b.our_prob)) * 100);
  const mktBar = Math.round(Math.max(0, Math.min(1, b.market_prob)) * 100);
  return `<tr class='${{cls}}'>
    <td>${{b.date}}</td>
    <td><b>${{b.city}}</b><br><span style='color:#64748b;font-size:10px'>${{b.station}}</span></td>
    <td class='bin-label'>${{b.bin}}</td>
    <td class='num'>${{(b.our_prob*100).toFixed(1)}}%
      <div class='bar'><div class='fill' style='width:${{ourBar}}%'></div></div>
    </td>
    <td class='num'>${{(b.market_prob*100).toFixed(1)}}%
      <div class='bar'><div class='fill market' style='width:${{mktBar}}%'></div></div>
    </td>
    <td class='num edge-cell ${{edgeCls}}'>${{edgeSign}}${{(b.edge*100).toFixed(1)}}%</td>
    <td class='num'>$${{Math.round(b.liquidity).toLocaleString()}}</td>
    <td><span class='pill ${{cls}}'>${{rec}}</span></td>
  </tr>`;
}}

function render() {{
  const c = $('f-city').value, d = $('f-date').value, r = $('f-rec').value;
  const me = parseFloat($('f-edge').value);
  const ml = parseFloat($('f-liq').value);
  let rows = BINS.filter(b =>
    (!c || b.city === c)
    && (!d || b.date === d)
    && (!r || b.recommend === r)
    && (isNaN(me) || b.edge >= me)
    && (isNaN(ml) || b.liquidity >= ml)
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (typeof av === 'number') return SORT_DIR * (av - bv);
    return SORT_DIR * String(av).localeCompare(String(bv));
  }});
  $('count').textContent = rows.length + ' / ' + BINS.length;
  $('tbody').innerHTML = rows.length ? rows.map(row).join('')
    : `<tr><td colspan='8' class='empty'>No bins match filters</td></tr>`;
  document.querySelectorAll('th').forEach(th => {{
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? 'sorted-asc' : 'sorted-desc');
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
['f-city','f-date','f-rec','f-edge','f-liq'].forEach(id =>
  $(id).addEventListener('input', render));

renderCities();
render();
</script>
</body></html>"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info(f"Wrote {os.path.getsize(out_path)/1024:.0f} KB dashboard to {out_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all US)")
    p.add_argument("--min-edge", type=float, default=0.10,
                   help="Minimum edge for BUY recommendation (default: 0.10)")
    p.add_argument("--html", default=DEFAULT_OUT,
                   help=f"Output HTML path (default: {DEFAULT_OUT})")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON to stdout instead of HTML")
    args = p.parse_args()

    # Load per-city σ calibration (silently uses defaults if file missing)
    _load_calibration()

    us_cities = list(US_CITY_STATES.keys()) if US_CITY_STATES else [
        c for c, m in CITY_STATIONS.items() if m[0].startswith("K")
    ]
    cities = args.city or us_cities
    cities = [c for c in cities if c in CITY_STATIONS]
    log.info(f"Predicting for {len(cities)} US cities")

    log.info("Discovering active Polymarket events ...")
    all_events = search_temp_high_events(min_liquidity=100)
    events_by_city: dict[str, list[dict]] = defaultdict(list)
    for e in all_events:
        if e.get("city") in cities:
            events_by_city[e["city"]].append(e)

    predictions: list[dict] = []
    cities_summary: list[dict] = []
    for city in cities:
        s = CITY_STATIONS[city]
        icao, _net, tz_str, lat, lon = s
        tz = ZoneInfo(tz_str)
        now_local = datetime.now(tz)
        today_str = now_local.date().isoformat()

        log.info(f"  {city:<14} {icao}  fetching NWS obs + forecast ...")

        nws_obs   = fetch_nws_today_obs(icao, tz_str)
        # NWS forecast first (matches Polymarket settlement source).  Fall
        # back to Open-Meteo only if NWS is unreachable.
        forecast  = fetch_nws_today_forecast(lat, lon, tz_str)
        if not forecast:
            log.warning(f"  {city:<14} NWS forecast empty — falling back to Open-Meteo")
            forecast = fetch_openmeteo_today(lat, lon, tz_str)

        # Afternoon mean wind direction for neighbor-signal lookup
        afternoon_winds = [r["wind_dir_deg"] for r in nws_obs
                            if r["hour_local"] in range(11, 19)
                            and r["wind_dir_deg"] is not None]
        wind_mean = vector_mean_dir(afternoon_winds)
        wind_octant = deg_to_cardinal(wind_mean) if wind_mean is not None else None

        neighbor_signal = compute_neighbor_signal(city, today_str, wind_octant)

        # Summary for the city card
        observed_max = max((r["temp_c"] for r in nws_obs), default=None)
        observed_peak_hour = None
        if observed_max is not None:
            observed_peak_hour = max(r["hour_local"] for r in nws_obs
                                      if r["temp_c"] == observed_max)
        current_temp = nws_obs[-1]["temp_c"] if nws_obs else None
        current_wd = nws_obs[-1]["wind_dir_deg"] if nws_obs else None
        upwind_summary = ""
        if neighbor_signal["upwind_neighbors"]:
            top = neighbor_signal["upwind_neighbors"][:2]
            upwind_summary = ", ".join(
                f"{u['sid']} @{u['peak_hour']}:00 ({u['peak_temp']:.0f}°→{u['current_temp']:.0f}°)"
                for u in top
            )

        # Per-event predictions
        n_buys_this_city = 0
        for ev in sorted(events_by_city.get(city, []), key=lambda e: e.get("date") or ""):
            pred = predict_bins(ev, nws_obs, forecast, neighbor_signal,
                                 now_local.hour, city=city)
            if not pred.get("bins"):
                continue
            for b in pred["bins"]:
                if (b["edge"] >= args.min_edge
                    and b["market_prob"] < 0.95
                    and b["liquidity_usd"] >= 300):
                    n_buys_this_city += 1
            predictions.append({
                "city":              city,
                "station":           icao,
                "date":              ev.get("date"),
                "event_title":       ev.get("event_title"),
                "bins":              pred["bins"],
                "mu":                pred["mu"],
                "sigma":             pred["sigma"],
                "observed_max_c":    pred["observed_max_c"],
                "observed_peak_hour": pred["observed_peak_hour"],
                "forecast_high":     pred["forecast_high"],
                "forecast_peak_hour": pred["forecast_peak_hour"],
                "sunset_hour":       pred["sunset_hour"],
            })

        cities_summary.append({
            "city":               city,
            "station":            icao,
            "tz_label":           tz_str.split("/")[-1].replace("_", " "),
            "current_temp_c":     round(current_temp, 1) if current_temp is not None else None,
            "current_wind_dir":   deg_to_cardinal(current_wd) if current_wd is not None else None,
            "observed_max_c":     round(observed_max, 1) if observed_max is not None else None,
            "observed_peak_hour": observed_peak_hour,
            "forecast_high":      round(forecast.get("forecast_high"), 1) if forecast else None,
            "forecast_peak_hour": forecast.get("forecast_peak_hour") if forecast else None,
            "sunset_hour":        forecast.get("sunset_hour") if forecast else None,
            "wind_octant_after":  wind_octant,
            "upwind_summary":     upwind_summary,
            "has_data":           bool(nws_obs) and bool(forecast),
            "n_buys":             n_buys_this_city,
        })

    # KPI rollup
    all_bins = [b for p in predictions for b in p["bins"]]
    buys = [b for b in all_bins
             if b["edge"] >= args.min_edge
             and b["market_prob"] < 0.95
             and b["liquidity_usd"] >= 300]
    kpis = {
        "n_cities":       len(cities),
        "n_events":       len(predictions),
        "n_bins":         len(all_bins),
        "n_buys":         len(buys),
        "avg_buy_edge":   (sum(b["edge"] for b in buys) / len(buys)) if buys else 0.0,
        "total_buy_edge": sum(b["edge"] for b in buys),
    }

    if args.json:
        print(json.dumps({"kpis": kpis, "cities": cities_summary,
                            "predictions": predictions}, default=str, indent=2))
        return 0

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    render_dashboard(predictions, cities_summary, kpis, generated_at, args.html)

    print()
    print("=" * 78)
    print(f"  KPIs: {kpis['n_cities']} cities · {kpis['n_events']} events · "
          f"{kpis['n_bins']} bins · {kpis['n_buys']} BUY signals")
    if buys:
        print(f"  Avg buy edge: {kpis['avg_buy_edge']:+.1%} · "
              f"Total edge $/$1: ${kpis['total_buy_edge']:+.2f}")
    print(f"  Dashboard: {args.html}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())