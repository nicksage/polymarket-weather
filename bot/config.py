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
MAX_SOURCE_DISAGREEMENT: float = _float("MAX_SOURCE_DISAGREEMENT", 0.15)
MAX_DAILY_DRAWDOWN_PCT: float = _float("MAX_DAILY_DRAWDOWN_PCT", 0.10)

# System
DB_PATH: str = os.getenv("DB_PATH", "data/signals.db")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Forecast horizon: skip contracts resolving more than this many days out.
# ECMWF ensemble is reliable to ~10 days. Beyond that, member spread is too
# wide to produce a calibrated probability — model_p would be unreliable.
MAX_FORECAST_DAYS: int = int(os.getenv("MAX_FORECAST_DAYS", "10"))

# Brier Score Weights for ensemble
BRIER_WEIGHTS: dict[str, float] = {
    "nws": _float("BRIER_WEIGHT_NWS", 0.40),
    "openmeteo": _float("BRIER_WEIGHT_OPENMETEO", 0.35),
    "openmeteo_det": _float("BRIER_WEIGHT_OPENMETEO", 0.25),
    "noaa": _float("BRIER_WEIGHT_NOAA", 0.15),
    "tomorrowio": _float("BRIER_WEIGHT_TOMORROWIO", 0.10),
}