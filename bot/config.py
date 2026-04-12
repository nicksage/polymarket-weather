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
MAX_TOTAL_EXPOSURE_PCT: float = _float("MAX_TOTAL_EXPOSURE_PCT", 0.20)
MIN_LIQUIDITY_USD: float = _float("MIN_LIQUIDITY_USD", 500.0)
MIN_HOURS_TO_EXPIRY: float = _float("MIN_HOURS_TO_EXPIRY", 6.0)
MAX_DAILY_DRAWDOWN_PCT: float = _float("MAX_DAILY_DRAWDOWN_PCT", 0.10)

# System
DB_PATH: str = os.getenv("DB_PATH", "data/signals.db")
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
