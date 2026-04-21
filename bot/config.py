# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def _bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")

# Polymarket
POLYMARKET_PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")

# Weather APIs
NOAA_API_TOKEN: str = os.getenv("NOAA_API_TOKEN", "")
TOMORROWIO_API_KEY: str = os.getenv("TOMORROWIO_API_KEY", "")

# Trading
INITIAL_BANKROLL: float = _float("BANKROLL_USDC", 1000.0)
PAPER_TRADE: bool = _bool("PAPER_TRADE", True)
KELLY_FRACTION: float = _float("KELLY_FRACTION", 0.25)
EDGE_THRESHOLD: float = _float("EDGE_THRESHOLD", 0.07)

# Risk Limits
MAX_POSITION_PCT: float = _float("MAX_POSITION_PCT", 0.02)
MAX_TOTAL_EXPOSURE_PCT: float = _float("MAX_TOTAL_EXPOSURE_PCT", 0.15)

# Event-level and city-day concentration caps (Phase 3).
# MAX_EVENT_EXPOSURE_PCT caps total capital across all bins in one event.
# MAX_CITY_DAY_EXPOSURE caps total capital across all events for one (city, date).
MAX_EVENT_EXPOSURE_PCT: float = _float("MAX_EVENT_EXPOSURE_PCT", 0.03)
MAX_CITY_DAY_EXPOSURE: float = _float("MAX_CITY_DAY_EXPOSURE", 0.05)
MIN_LIQUIDITY_USD: float = _float("MIN_LIQUIDITY_USD", 500.0)
MIN_HOURS_TO_EXPIRY: float = _float("MIN_HOURS_TO_EXPIRY", 6.0)
MAX_DAILY_DRAWDOWN_PCT: float = _float("MAX_DAILY_DRAWDOWN_PCT", 0.10)

# System
# Always resolve DB_PATH to bot/data/signals.db regardless of where Python
# is launched from (repo root, bot/, or elsewhere).  This prevents stale
# duplicate DBs from appearing at ./data/signals.db when the bot is run
# from the project root and the dashboard is run from bot/.
_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.join(_BOT_DIR, "data", "signals.db")

# Separate DB for backtest data — kept out of the live trading DB so
# backfills/rebuilds cannot accidentally touch production state.
BACKTEST_DB_PATH: str = os.getenv(
    "BACKTEST_DB_PATH", os.path.join(_BOT_DIR, "data", "backtest.db")
)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Forecast horizon
# ECMWF ensemble is reliable to ~15 days; beyond that member spread is too
# wide to produce a calibrated probability distribution.
# ---------------------------------------------------------------------------
MAX_FORECAST_DAYS: int = int(os.getenv("MAX_FORECAST_DAYS", "15"))

# ---------------------------------------------------------------------------
# Temperature distribution modeling
# ---------------------------------------------------------------------------

# Ensemble model weights when blending ECMWF + GFS distributions.
ECMWF_WEIGHT: float = _float("ECMWF_WEIGHT", 0.65)
GFS_WEIGHT: float = _float("GFS_WEIGHT", 0.35)

# Bayesian blending: pure forecast used for days 0–BLEND_START_DAYS,
# then climatological prior is linearly mixed in at 10% per additional day.
BLEND_START_DAYS: int = int(os.getenv("BLEND_START_DAYS", "7"))

# Climatological prior: how many years of ERA5 history to fetch,
# and the ±day window around the target calendar date.
CLIM_LOOKBACK_YEARS: int = int(os.getenv("CLIM_LOOKBACK_YEARS", "10"))
CLIM_WINDOW_DAYS: int = int(os.getenv("CLIM_WINDOW_DAYS", "14"))

# Sanity gates on the blended forecast σ (°C).
# Below MIN → suspect forecast (API returned a single point, no spread).
# Above MAX → skip this event (ensemble completely failed).
MIN_FORECAST_SIGMA_C: float = _float("MIN_FORECAST_SIGMA_C", 0.5)
MAX_FORECAST_SIGMA_C: float = _float("MAX_FORECAST_SIGMA_C", 8.0)

# ---------------------------------------------------------------------------
# Probability normalization warning thresholds
# The sum of raw model probabilities across all bins in an event should be
# close to 1.0.  If the raw sum falls outside [LOW, HIGH] before normalization,
# a warning flag is attached to the event — signal sizing is halved.
# ---------------------------------------------------------------------------
NORM_WARNING_LOW: float = _float("NORM_WARNING_LOW", 0.70)
NORM_WARNING_HIGH: float = _float("NORM_WARNING_HIGH", 1.30)

# ---------------------------------------------------------------------------
# ERA5 persistent cache + bias correction
# ---------------------------------------------------------------------------

# USE_KDE_CLIM: when True, the climatological prior is fit using Kernel Density
# Estimation (KDE) instead of a normal distribution.  KDE captures non-normal
# temperature distributions (skew, fat tails) more accurately but requires at
# least 10 historical data points.  Defaults to False (normal distribution)
# until sufficient history is accumulated.
USE_KDE_CLIM: bool = _bool("USE_KDE_CLIM", False)

# MIN_BIAS_OBSERVATIONS: minimum number of (forecast, actual) pairs required
# before bias correction is applied.  Below this threshold the correction is
# 0.0 (no adjustment) — avoids noisy corrections from sparse data.
MIN_BIAS_OBSERVATIONS: int = int(os.getenv("MIN_BIAS_OBSERVATIONS", "10"))

# ---------------------------------------------------------------------------
# Geography filter
# ---------------------------------------------------------------------------

# US bounding box used by both risk.py (trade guard) and dashboard.py (filter).
# Covers contiguous US + Alaska + Hawaii with generous margins.
US_LAT: tuple[float, float] = (15.0, 72.0)
US_LON: tuple[float, float] = (-180.0, -60.0)

# When True, the risk layer blocks execution on any signal whose lat/lon falls
# outside the US bounding box.  Dashboard filter is independent and always
# available regardless of this setting.
TRADE_US_ONLY: bool = _bool("TRADE_US_ONLY", False)

# ---------------------------------------------------------------------------
# Position controls
# ---------------------------------------------------------------------------

# MAX_BIN_BUYS: maximum number of temperature-range bins the bot may hold
# simultaneously for the same city+date event.  Default 1 means the bot
# takes at most one position per event.  Set to 0 to disable the check.
MAX_BIN_BUYS: int = int(os.getenv("MAX_BIN_BUYS", "1"))

# MAX_TRADE_DAYS: how many calendar days ahead the bot may enter trades.
# 0 = only contracts resolving today (local time).
# 1 = today + tomorrow.
# N = any contract resolving within the next N days (inclusive).
# Default 7 — avoids very near-term contracts where the edge check plus
# MIN_HOURS_TO_EXPIRY already blocks same-day trades anyway.
MAX_TRADE_DAYS: int = int(os.getenv("MAX_TRADE_DAYS", "7"))

# ALLOWED_SIDES: restricts which contract sides the bot may trade.
# "yes"  → only BUY YES contracts
# "no"   → only BUY NO contracts
# "both" → no restriction (default)
ALLOWED_SIDES: str = os.getenv("ALLOWED_SIDES", "both").lower().strip()

# ---------------------------------------------------------------------------
# Wallet / Data API
# ---------------------------------------------------------------------------

# Polymarket proxy wallet address — used to query on-chain position data from
# data-api.polymarket.com.  Required for live trading enrichment; ignored in
# paper mode.
WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")

# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------
# Which trading strategy to use.  Each strategy has its own signal generation,
# ranking, and exit logic.  Available: "edge_disagreement", "top_bin_value".
ACTIVE_STRATEGY: str = os.getenv("ACTIVE_STRATEGY", "edge_disagreement")

# ---------------------------------------------------------------------------
# Live adjustment layer (Phase 2b)
# Additive adjustment on top of the ECMWF + GFS + climatology blend, driven
# by Visual Crossing live observations.  Shadow mode by default — both
# blended and adjusted values are computed and logged, but trading continues
# to use the blended values until USE_LIVE_ADJUSTMENT is flipped True.
# ---------------------------------------------------------------------------

USE_LIVE_ADJUSTMENT: bool = _bool("USE_LIVE_ADJUSTMENT", False)

# Global damping on the smoothed intraday residual.
LIVE_ADJ_ALPHA: float = _float("LIVE_ADJ_ALPHA", 0.5)

# Local hour where the time-of-day weight reaches 1.0.  Residuals at this
# hour are assumed most predictive of the daily max.  Triangular taper to
# 0 across LIVE_ADJ_TOD_RAMP_HOURS on either side.
LIVE_ADJ_TOD_PEAK_HOUR: int = int(os.getenv("LIVE_ADJ_TOD_PEAK_HOUR", "13"))
LIVE_ADJ_TOD_RAMP_HOURS: int = int(os.getenv("LIVE_ADJ_TOD_RAMP_HOURS", "5"))

# Minimum number of observations required before trusting the residual.
# Below this threshold, adjusted = blended (cold-start fallback).
LIVE_ADJ_MIN_OBS: int = int(os.getenv("LIVE_ADJ_MIN_OBS", "2"))

# Safety caps: actual adjustment is clipped to min(ABS_C, SIGMA * blended_sigma).
LIVE_ADJ_MAX_ABS_C: float = _float("LIVE_ADJ_MAX_ABS_C", 3.0)
LIVE_ADJ_MAX_SIGMA: float = _float("LIVE_ADJ_MAX_SIGMA", 2.0)

# When True, the observed-max-so-far is applied as a lower-bound truncation
# on the bin probability distribution (mass below observed_max is collapsed
# and remaining mass is renormalized).
LIVE_ADJ_OBS_FLOOR: bool = _bool("LIVE_ADJ_OBS_FLOOR", True)

# Rolling window for residual smoothing (minutes).
LIVE_ADJ_SMOOTH_WINDOW_MIN: int = int(os.getenv("LIVE_ADJ_SMOOTH_WINDOW_MIN", "60"))

# ---------------------------------------------------------------------------
# Live adjustment v2 — dynamic σ shrinkage (Phase 2b-v2)
# Time-of-day based σ shrinkage with an optional residual-widening term that
# prevents overconfidence on days that behave unexpectedly.  Disabled by
# default in production; enable only after the floor+shrinkage combination
# passes the backtest decision criteria (Brier, late-day Brier, reliability
# improvement in the high-probability range).
# ---------------------------------------------------------------------------

USE_LIVE_ADJUSTMENT_SIGMA_SHRINK: bool = _bool("LIVE_ADJ_SIGMA_SHRINK", False)

# Enable/disable each factor independently.  Keeping them separable lets the
# backtest compare time_only vs time+residual cleanly.
LIVE_ADJ_SIGMA_TIME_ENABLED: bool = _bool("LIVE_ADJ_SIGMA_TIME_ENABLED", True)
LIVE_ADJ_SIGMA_RESIDUAL_ENABLED: bool = _bool("LIVE_ADJ_SIGMA_RESIDUAL_ENABLED", True)

# Time factor: shrunk_factor_time = max(sqrt(remaining_daylight_frac), TIME_MIN)
LIVE_ADJ_SIGMA_TIME_MIN: float = _float("LIVE_ADJ_SIGMA_TIME_MIN", 0.35)

# Residual factor: 1 + min(|residual| / DIV, CAP) — widens σ on surprising days
LIVE_ADJ_SIGMA_RESIDUAL_DIV: float = _float("LIVE_ADJ_SIGMA_RESIDUAL_DIV", 3.0)
LIVE_ADJ_SIGMA_RESIDUAL_CAP: float = _float("LIVE_ADJ_SIGMA_RESIDUAL_CAP", 0.5)

# Hard floors — protect against runaway collapse even when both factors push σ low.
LIVE_ADJ_SIGMA_ABS_FLOOR_C: float = _float("LIVE_ADJ_SIGMA_ABS_FLOOR_C", 1.0)
LIVE_ADJ_SIGMA_REL_FLOOR: float = _float("LIVE_ADJ_SIGMA_REL_FLOOR", 0.35)

# ---------------------------------------------------------------------------
# VC forecast diagnostic layer (Phase 2c)
# Shadow / measurement layer — captures what VC is forecasting alongside our
# model so we can measure whether VC disagreement contains useful information.
# Purely diagnostic: does NOT feed the μ/σ blend or alter trade decisions.
# ---------------------------------------------------------------------------

# Disagreement thresholds (degrees Celsius).
VC_DIAG_DISAGREE_LARGE_C: float = _float("VC_DIAG_DISAGREE_LARGE_C", 2.5)
VC_DIAG_WARN_DIRECTIONAL_C: float = _float("VC_DIAG_WARN_DIRECTIONAL_C", 1.5)

# Future-day horizon.  0 = disabled, 1 = pull diagnostics for D+1 events,
# 2 = D+1 and D+2, etc.  Adds one VC call per future-day event per 2-hour pull.
VC_DIAG_FUTURE_DAY_HORIZON: int = int(os.getenv("VC_DIAG_FUTURE_DAY_HORIZON", "1"))

# Hour window used for vc_hourly_path_rmse_c (same-day only).  Default 6
# hours gives a cleaner near-term signal than the full remaining day.
VC_DIAG_RMSE_WINDOW_HOURS: int = int(os.getenv("VC_DIAG_RMSE_WINDOW_HOURS", "6"))

# Approximate bin width (°C) used to translate disagreements into a rough
# "bins apart" metric.  Most Polymarket bins are 1°F = ~0.556°C; set via env
# if the market bin size changes.
VC_DIAG_BIN_WIDTH_C: float = _float("VC_DIAG_BIN_WIDTH_C", 0.556)


# ---------------------------------------------------------------------------
# Exit engine (Phase 3)
# Active position management: classify open positions as INVALIDATED /
# DYING / WEAKENED / THRIVING / HEALTHY and execute accordingly.
# ---------------------------------------------------------------------------

EXIT_EVAL_ENABLED: bool = _bool("EXIT_EVAL_ENABLED", True)

# INVALIDATED — observed max exceeds bin's upper bound + buffer
EXIT_DEAD_BUFFER_C: float = _float("EXIT_DEAD_BUFFER_C", 0.5)

# DYING — current model_prob collapsed vs entry
EXIT_DYING_PROB_THRESHOLD: float = _float("EXIT_DYING_PROB_THRESHOLD", 0.05)
EXIT_DYING_ENTRY_MIN: float = _float("EXIT_DYING_ENTRY_MIN", 0.15)

# WEAKENED — cumulative μ shift since entry exceeds N × entry_sigma
EXIT_WEAKENED_SIGMA_SHIFT: float = _float("EXIT_WEAKENED_SIGMA_SHIFT", 1.0)
EXIT_EDGE_REVERSED_ENABLED: bool = _bool("EXIT_EDGE_REVERSED_ENABLED", True)

# THRIVING — EV-based hold vs sell comparison
EXIT_EV_CAPTURE_RATIO: float = _float("EXIT_EV_CAPTURE_RATIO", 0.95)

# Hard stop-loss (circuit breaker, configurable)
EXIT_HARD_STOP_ENABLED: bool = _bool("EXIT_HARD_STOP_ENABLED", True)
EXIT_HARD_STOP_PCT: float = _float("EXIT_HARD_STOP_PCT", -0.70)

# ---------------------------------------------------------------------------
# Confidence-weighted Kelly (Phase 3)
# ---------------------------------------------------------------------------

CONFIDENCE_AGREEMENT_THRESHOLD: float = _float("CONFIDENCE_AGREEMENT_THRESHOLD", 2.0)

# ---------------------------------------------------------------------------
# Bracket entry (Phase 3)
# ---------------------------------------------------------------------------

BRACKET_ENABLED: bool = _bool("BRACKET_ENABLED", True)
BRACKET_MAX_BINS: int = int(os.getenv("BRACKET_MAX_BINS", "2"))
BRACKET_MIN_ADJACENT_EDGE: float = _float("BRACKET_MIN_ADJACENT_EDGE", 0.04)

# ---------------------------------------------------------------------------
# Liquidity-aware sizing (Phase 3)
# When enabled, position size is capped at a fraction of displayed liquidity.
# Underfilled positions are topped up on subsequent trading cycles as more
# liquidity becomes available.
# ---------------------------------------------------------------------------

LIQUIDITY_AWARE_SIZING: bool = _bool("LIQUIDITY_AWARE_SIZING", True)
MAX_LIQUIDITY_TAKE_PCT: float = _float("MAX_LIQUIDITY_TAKE_PCT", 0.40)


# Tail-probability flattening — applied AFTER per-event bin normalization:
#     p' = p ** LIVE_ADJ_TAIL_FLATTEN_EXPONENT
# Then re-normalized across bins so probabilities still sum to 1.  Preserves
# bin ranking while trimming overconfident spikes in the upper tail.  Only
# applied when USE_LIVE_ADJUSTMENT_SIGMA_SHRINK is True (bundled with v2).
# Exponent of 1.0 disables flattening.
# ---------------------------------------------------------------------------
# Forecast DB read (eliminates redundant API calls during trading scan)
# When enabled, the trading scan reads ECMWF/GFS mu/sigma from forecast_runs
# (populated by the forecast pull at :05) instead of re-calling the API.
# Falls back to API if no fresh data exists in the DB.
# ---------------------------------------------------------------------------

FORECAST_DB_READ_ENABLED: bool = _bool("FORECAST_DB_READ_ENABLED", True)
FORECAST_DB_MAX_AGE_HOURS: float = _float("FORECAST_DB_MAX_AGE_HOURS", 3.0)


LIVE_ADJ_TAIL_FLATTEN_EXPONENT: float = _float("LIVE_ADJ_TAIL_FLATTEN_EXPONENT", 0.95)
