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

# Risk Limits
MAX_POSITION_PCT: float = _float("MAX_POSITION_PCT", 0.02)
MAX_TOTAL_EXPOSURE_PCT: float = _float("MAX_TOTAL_EXPOSURE_PCT", 0.15)

# Event-level and city-day concentration caps (Phase 3).
# MAX_EVENT_EXPOSURE_PCT caps total capital across all bins in one event.
# MAX_CITY_DAY_EXPOSURE caps total capital across all events for one (city, date).
MAX_EVENT_EXPOSURE_PCT: float = _float("MAX_EVENT_EXPOSURE_PCT", 0.03)
MAX_CITY_DAY_EXPOSURE: float = _float("MAX_CITY_DAY_EXPOSURE", 0.05)

# Hard cap on count of simultaneously-open positions (per mode).
# 0 = unlimited.  Counts anything that's still consuming or could consume
# capital: status='open' (filled or pending buy) and status='exiting'
# (sell ladder in progress).  Top-ups don't count — they merge into an
# existing position rather than create a new one.  Useful guard while
# easing into live trading: cap exposure by COUNT, not just by dollars.
MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "0"))

# Minimum age (minutes) a pending order must reach before the monitor's
# cancel pass is allowed to kill it.  Without this, fresh orders placed
# seconds before a monitor cycle (especially the inline cycle that runs
# right after `trading_run` on startup) get nuked before they have a
# chance to fill on the book.  10 minutes is a sensible default — long
# enough that most fillable orders land, short enough that abandoned
# orders don't pile up across cycles.
MIN_ORDER_AGE_BEFORE_CANCEL_MIN: float = _float(
    "MIN_ORDER_AGE_BEFORE_CANCEL_MIN", 10.0
)

# --- Orderbook-aware execution + ranking (added 2026-04-30) ---
# Walk-up tolerance (cents) above the touch when placing a BUY.  We compute
# `limit = min(best_ask + walk, MPV_MAX_PRICE)` and submit GTC.  Polymarket
# matches the order against asks ≤ limit (cheapest first) and rests the
# remainder on the book at our limit.  1¢ default keeps slippage tight.
ORDERBOOK_WALK_CENTS: int = int(os.getenv("ORDERBOOK_WALK_CENTS", "1"))

# Wide-spread skip filter.  Bins where (best_ask - best_bid) exceeds this
# many cents at signal time are dropped before ranking — the spread
# implies illiquid book + slippage risk that's not worth the entry.
# Set to 0 (or very high) to disable the filter.
MAX_SPREAD_CENTS_FOR_ENTRY: int = int(os.getenv("MAX_SPREAD_CENTS_FOR_ENTRY", "6"))

# Cap on how many top candidates get an orderbook fetch during ranking.
# We fetch books for the top N by static `liquidity_usd`, then rank those
# by spread + sweepable depth.  Keeps API call count predictable
# (Polymarket's CLOB rate-limits aggressively from datacenter IPs).
RANK_TOP_N_FOR_ORDERBOOK: int = int(os.getenv("RANK_TOP_N_FOR_ORDERBOOK", "15"))

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

# Custom SUMMARY log level — sits between INFO (20) and WARNING (30).
# Usage: logger.log(SUMMARY_LEVEL, "message")
SUMMARY_LEVEL: int = 25

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

MAX_YES_BINS: int = int(os.getenv("MAX_YES_BINS", "2"))
MAX_NO_BINS: int = int(os.getenv("MAX_NO_BINS", "0"))

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

# Polymarket proxy wallet address — the contract address that actually holds
# your USDC and positions.  Critical: this is NOT the same as the EOA derived
# from POLYMARKET_PRIVATE_KEY.  The private key signs requests; the proxy
# holds the funds.  Without this set, the CLOB client queries the signer's
# EOA balance (always ~$0 for a wallet-connect setup) and can't sign orders
# correctly.  Required for live trading.
WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")

# Polymarket signature type — tells the CLOB client which proxy contract
# scheme owns WALLET_ADDRESS:
#   0 = EOA              (no proxy — signer == funder; rare)
#   1 = POLY_PROXY       (Polymarket-app-issued wallets — DEFAULT)
#   2 = POLY_GNOSIS_SAFE (only if you imported your own Gnosis Safe)
#
# Default 1 covers the common case: any wallet created by signing into
# polymarket.com (whether via email or MetaMask connect) — Polymarket
# wraps the signer EOA in their own proxy contract.  Only set 2 if you
# manually attached a separately-deployed Gnosis Safe.  If orders fail
# with `order_version_mismatch`, the signature_type doesn't match the
# on-chain proxy type — verify by inspecting the bytecode at WALLET_ADDRESS.
WALLET_SIGNATURE_TYPE: int = int(os.getenv("WALLET_SIGNATURE_TYPE", "1"))

# ---------------------------------------------------------------------------
# Wallet balance sync (live trading guard)
# ---------------------------------------------------------------------------
# When enabled, the bot queries the actual on-chain USDC balance via the CLOB
# client at the start of each trading cycle and caps the effective bankroll at
# min(BANKROLL_USDC, wallet_balance - reserve).  Silent scale-down (logs INFO
# when the cap binds, never aborts).  Disabled in paper mode automatically.
#
# Use case: you set BANKROLL_USDC=5000 but later withdraw $4000 from the
# wallet — without this check, the bot tries to size against the stale cap
# and hits CLOB-level rejections.  With this check, sizing scales down to
# the actual balance gracefully.
# ---------------------------------------------------------------------------

WALLET_BALANCE_CHECK_ENABLED: bool = _bool("WALLET_BALANCE_CHECK_ENABLED", True)

# Reserve held back from the effective bankroll — covers gas, rounding,
# and a small safety margin so we never try to spend the very last dollar.
WALLET_BALANCE_RESERVE_USDC: float = _float("WALLET_BALANCE_RESERVE_USDC", 10.0)

# How long to cache a balance reading before re-querying.  Polymarket's
# Data API is fast but we don't need to hammer it on every call — once per
# cycle is plenty.
WALLET_BALANCE_REFRESH_MIN: float = _float("WALLET_BALANCE_REFRESH_MIN", 30.0)

# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------
# Which trading strategy to use.  Each strategy has its own signal generation,
# ranking, and exit logic.  Available: "edge_disagreement", "top_bin_value".
ACTIVE_STRATEGY: str = os.getenv("ACTIVE_STRATEGY", "top_bin_value")

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

# DYING — market price dropped below threshold
EXIT_DYING_PROB_THRESHOLD: float = _float("EXIT_DYING_PROB_THRESHOLD", 0.05)

# WEAKENED — cumulative μ shift since entry exceeds N × entry_sigma
EXIT_WEAKENED_SIGMA_SHIFT: float = _float("EXIT_WEAKENED_SIGMA_SHIFT", 1.0)

# Hard stop-loss (circuit breaker, configurable)
EXIT_HARD_STOP_ENABLED: bool = _bool("EXIT_HARD_STOP_ENABLED", True)
EXIT_HARD_STOP_PCT: float = _float("EXIT_HARD_STOP_PCT", -0.70)

# Bleed circuit-breaker for the exit ladder.  When an in-flight stop-out's
# current best_bid drops more than this fraction below the original trigger
# price, force an immediate cross-spread instead of climbing through the
# patient ladder rungs.  Stops the "rung 3 limit at $0.40 sitting unfilled
# while bid is $0.20" failure mode that bleeds the position for hours.
# Default 0.15 = 15% (consistent with practitioner literature on stop-out
# urgency in thin markets).  Set to 1.0 to disable.
EXIT_BLEED_CROSS_PCT: float = _float("EXIT_BLEED_CROSS_PCT", 0.15)

# Stale-topup re-pricing threshold (Lightweight Option B).
# When a pending topup's limit price is more than this many cents BELOW the
# live best_ask (i.e., asks moved upward and our bid is now stale), the
# */5 min stale-topup refresher cancels + re-issues at the fresh price.
# Direction-aware: only fires on upward drift.  Asks moving DOWN don't
# need action (Polymarket matches us at the cheaper price by default).
# Default 1.5¢ = balance between catching meaningful drifts and avoiding
# constant re-pricing churn (which would leak info to market makers).
# Set to 0 to disable the refresher entirely.
TOPUP_REPRICE_THRESHOLD_CENTS: float = _float("TOPUP_REPRICE_THRESHOLD_CENTS", 1.5)

# ---------------------------------------------------------------------------
# Confidence-weighted Kelly (Phase 3)
# ---------------------------------------------------------------------------

CONFIDENCE_AGREEMENT_THRESHOLD: float = _float("CONFIDENCE_AGREEMENT_THRESHOLD", 2.0)

# ---------------------------------------------------------------------------
# Liquidity-aware sizing (Phase 3)
# When enabled, position size is capped at a fraction of displayed liquidity.
# Underfilled positions are topped up on subsequent trading cycles as more
# liquidity becomes available.
# ---------------------------------------------------------------------------

LIQUIDITY_AWARE_SIZING: bool = _bool("LIQUIDITY_AWARE_SIZING", True)

# DEPRECATED 2026-04-30 — replaced by MAX_TAKE_PCT_OF_ASK_DEPTH below.
# The old "% of total displayed liquidity" rule used Gamma's `liquidityNum`
# field, which is bid + ask combined.  That over-restricts when bids are
# deep + asks thin (we get capped despite enough sellers) and under-restricts
# the inverse.  Kept as a config var only so legacy .env files don't fail
# config import; nothing in the live code reads it anymore.
MAX_LIQUIDITY_TAKE_PCT: float = _float("MAX_LIQUIDITY_TAKE_PCT", 0.40)

# Cap on how much of the ask-side depth at acceptable prices we'll take
# in one order.  Acceptable prices = asks ≤ sweep_limit (best_ask + walk
# ¢, capped at MPV_MAX_PRICE).  This is the right basis: we only care
# about asks when buying, and we only care about asks at prices we'd
# actually pay.  Default 0.66 leaves ⅓ of the relevant depth for other
# market participants — comfortably above the "dominant taker" floor
# (most quant rules say 33-50%) but permissive enough that retail $10
# orders rarely hit the cap unless the book is genuinely thin.
MAX_TAKE_PCT_OF_ASK_DEPTH: float = _float("MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)

# Floor (USDC) below which we don't bother placing an order at all.
# Acceptable-ask-depth × MAX_TAKE_PCT below this means the book is too
# thin to fill a meaningful fraction of our intended size — better to
# wait a cycle and re-evaluate than to leave a tiny order on the book.
MIN_FILLABLE_USDC: float = _float("MIN_FILLABLE_USDC", 2.0)

# Ensure-fill strategies (TKH and any future strategy whose thesis
# requires owning every bin in a basket) use a LOWER min-fillable floor
# so a thin book doesn't cause a bin to be skipped entirely.  TKH's
# per-event dedup means a skipped bin is never retried -- breaking the
# hedge.  The repricer + topup loop chase any remaining gap, so the
# correct behaviour is "place what the book allows, even if small".
# Default $1.05 = Polymarket's $1 strict minimum + 5c buffer for the
# share-rounding that occurs inside py_clob_client.
ENSURE_FILL_MIN_FILLABLE_USDC: float = _float(
    "ENSURE_FILL_MIN_FILLABLE_USDC", 1.05
)
ENSURE_FILL_STRATEGIES: set[str] = {"top_k_hedged"}

# Per-rank execution aggressiveness for TKH bins.  Format: colon-
# separated walk_cents (relative to best_ask) for ranks 0, 1, 2, ...
# Positive = above best_ask (aggressive, marketable); 0 = at the ask
# (marketable but no premium); negative = below best_ask (passive,
# rests between bid and ask, fills only when seller hits us).
#
# Default "1:0:-3" = rank 0 aggressive (current behaviour, +1c above
# ask for fast fills on the largest bin), rank 1 at-the-ask (still
# marketable but doesn't pay the spread premium), rank 2+ passive
# (-3c below ask) so smaller thin-book bins try to catch sellers
# at a better price first; if not filled within ~5 min the stale-
# entry repricer escalates to aggressive automatically.
#
# Set to "1:1:1" to revert to the legacy "+1c for everything" behaviour.
def _parse_rank_walks(raw: str | None) -> list[int]:
    if not raw:
        return [1, 0, -3]
    try:
        out = [int(x.strip()) for x in raw.split(":") if x.strip()]
        return out if out else [1, 0, -3]
    except (ValueError, AttributeError):
        return [1, 0, -3]
TKH_RANK_WALK_CENTS: list[int] = _parse_rank_walks(
    os.getenv("TKH_RANK_WALK_CENTS")
)

# Drift auto-healing.  When True, the monitor's on-chain reconciliation
# does not just LOG share_drift -- it AUTOMATICALLY syncs the position
# row to chain truth (shares + entry_price + size_usdc).  Without this,
# every drift event accumulates and requires a manual run of
# repair_share_drift to clean up.  With it on, drift never persists
# beyond one monitor cycle; the chain becomes the authoritative source.
# orphan_db (DB has shares chain doesn't) is still WARN-only -- closing
# a position based on a possibly-transient API result is too risky to
# automate.  Set to False to revert to legacy log-only behaviour.
DRIFT_AUTO_HEAL: bool = _bool("DRIFT_AUTO_HEAL", True)

# Orphan_db auto-close.  An orphan_db row means the DB believes we own
# shares but the chain says zero.  Closing such a position based on a
# single Data API result is risky (a transient API miss could close real
# positions), so we require multi-cycle confirmation: only close when a
# row has been orphan_db for >= ORPHAN_DB_AUTO_CLOSE_CYCLES consecutive
# monitor cycles.  Counter is tracked in positions.orphan_db_cycles and
# RESET to 0 whenever the chain shows shares again.
#
# When the threshold is reached, the position is marked status='closed',
# shares=0, pnl = -size_usdc (assume total loss -- the shares aren't on
# chain, they were either never received or sold without our knowledge).
# Audited under activity_log category='REPAIR'.
ORPHAN_DB_AUTO_CLOSE: bool = _bool("ORPHAN_DB_AUTO_CLOSE", True)
ORPHAN_DB_AUTO_CLOSE_CYCLES: int = int(
    os.getenv("ORPHAN_DB_AUTO_CLOSE_CYCLES", "2")
)


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


# ---------------------------------------------------------------------------
# ML distribution model (Framing C — per-city observation-driven regressor)
# Shadow mode by default: predictions are computed and persisted on every
# scan, but the blended μ/σ is unchanged until ML_BLEND_ENABLED flips True.
# ---------------------------------------------------------------------------

# Master switch.  When False the inference hook still runs and logs
# (ml_mu_c, ml_sigma_c) to decision_snapshots, but ML_BLEND_WEIGHT_MAX is
# forced to 0 in the blender — no behavior change.
ML_BLEND_ENABLED: bool = _bool("ML_BLEND_ENABLED", False)

# Hard ceiling on the ML source's weight in the ensemble blender.  Start low,
# ramp up as shadow-mode metrics accumulate.  Set to 0.0 to fully disable
# blending even when ML_BLEND_ENABLED=True (useful for pure shadow runs).
ML_BLEND_WEIGHT_MAX: float = _float("ML_BLEND_WEIGHT_MAX", 0.0)

# Per-city models must beat this Brier score on the training hold-out before
# their weight is allowed to rise above 0 at inference time.  Prevents an
# undertrained model from degrading the blend.
ML_MIN_BRIER_TO_ACTIVATE: float = _float("ML_MIN_BRIER_TO_ACTIVATE", 0.20)

# Where per-city trained models are persisted (joblib pickles).
ML_MODELS_DIR: str = os.getenv(
    "ML_MODELS_DIR", os.path.join(_BOT_DIR, "ml", "models")
)


# ---------------------------------------------------------------------------
# Trailing stop-loss (single source of truth)
# ---------------------------------------------------------------------------
# A trailing stop exits a position when price falls TRAIL_TIERS[…].pct below
# its peak.  The tier-table format handles BOTH cases:
#
#   * Single trail across the whole probability range:
#       TRAIL_TIERS=0.00:1.00:0.10                # 10% trail everywhere
#
#   * Tiered trail that adapts as price climbs:
#       TRAIL_TIERS=0.00:0.30:0.15,0.30:0.50:0.10,0.50:0.70:0.07,0.70:0.90:0.05,0.90:1.00:0.03
#       (looser at low peaks → tighter as resolution converges to YES)
#
# Format: comma-separated triples of "lo:hi:pct".  Tiers MUST be in
# ascending order, MUST partition [0, 1] without overlaps, and pct must be
# in [0, 1].  Tier matching uses half-open intervals [lo, hi); the last
# tier's hi is treated inclusively so peak == 1.0 still resolves.
#
# TRAIL_ACTIVATION_GAIN: peak must rise at least this far above entry
# before the trail arms — prevents stops on entry-bar noise.
# ---------------------------------------------------------------------------

TRAIL_ACTIVATION_GAIN: float = _float("TRAIL_ACTIVATION_GAIN", 0.03)

# Default tier table per the engineering spec:
#   peak in [0.00, 0.30) -> 15% trail
#   peak in [0.30, 0.50) -> 10% trail
#   peak in [0.50, 0.70) ->  7% trail
#   peak in [0.70, 0.90) ->  5% trail
#   peak in [0.90, 1.00] ->  3% trail   (last tier inclusive at top)
_DEFAULT_TIERS = "0.00:0.30:0.15,0.30:0.50:0.10,0.50:0.70:0.07,0.70:0.90:0.05,0.90:1.00:0.03"


def _parse_tiers(env_value: str) -> list[tuple[float, float, float]]:
    """Parse 'lo:hi:pct,lo:hi:pct,...' into a validated tier list.

    Validates:
      * each triple has 3 numeric fields
      * 0 <= lo < hi <= 1, 0 <= pct <= 1
      * tiers are in ascending order with no overlaps

    Raises ValueError on any malformed input — _load_tiers() converts that
    to a logged warning + fallback to defaults so a typo in .env doesn't
    crash the whole bot at import time.
    """
    tiers: list[tuple[float, float, float]] = []
    for triple in env_value.split(","):
        triple = triple.strip()
        if not triple:
            continue
        parts = triple.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"bad tier triple {triple!r}, expected 'lo:hi:pct'"
            )
        try:
            lo, hi, pct = (float(p) for p in parts)
        except ValueError as e:
            raise ValueError(f"bad tier numbers in {triple!r}: {e}") from e
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"bad tier bounds in {triple!r}: need 0<=lo<hi<=1, got lo={lo} hi={hi}"
            )
        if not (0.0 <= pct <= 1.0):
            raise ValueError(
                f"bad trail pct in {triple!r}: need 0<=pct<=1, got pct={pct}"
            )
        tiers.append((lo, hi, pct))
    if not tiers:
        raise ValueError("no tiers parsed (empty TRAIL_TIERS)")
    for i in range(1, len(tiers)):
        if tiers[i][0] < tiers[i - 1][1]:
            raise ValueError(
                f"tiers overlap: {tiers[i - 1]} and {tiers[i]}"
            )
    return tiers


def _load_tiers() -> list[tuple[float, float, float]]:
    """Parse TRAIL_TIERS from env; fall back to spec defaults on any error.
    Prints to stderr because this runs at import time before logging is set up."""
    raw = os.getenv("TRAIL_TIERS", _DEFAULT_TIERS)
    try:
        return _parse_tiers(raw)
    except ValueError as e:
        import sys
        print(
            f"WARN: TRAIL_TIERS invalid ({e}); falling back to default tiers",
            file=sys.stderr,
        )
        return _parse_tiers(_DEFAULT_TIERS)


TRAIL_TIERS: list[tuple[float, float, float]] = _load_tiers()
