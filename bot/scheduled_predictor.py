"""
scheduled_predictor.py — APScheduler entry point for the intraday bin
predictor.

Runs every N minutes (default 15) during US trading hours.  For each US
city in its active window, fetches live observations + forecast + neighbor
signals, computes per-bin probabilities, applies the gate stack, sizes
positions with fractional Kelly, and writes the result to
paper_predictor_signals.

Two modes (controlled by PREDICTOR_MODE env var):
  paper  — default; writes intended trades to paper_predictor_signals.
           No orders placed, no real money at risk.
  live   — places real CLOB orders for any signal that passes all gates.
           NOT WIRED YET — paper mode validated first.

Gate stack (all must pass for PAPER_BUY):
  1. min_trigger_hour_local      ≥ 13 (no false-peak triggers)
  2. max_trigger_hour_local      ≤ 22 (markets illiquid after)
  3. min_edge                    ≥ 0.10 (10pp edge over market)
  4. market_yes_price            < 0.95 (room to squeeze)
  5. liquidity_usd               ≥ $300
  6. dedup                       no PAPER_BUY for same (event,bin) today
  7. exposure_cap                today's deployed ≤ MAX_DAILY_EXPOSURE
  8. trades_per_day              today's count   < MAX_TRADES_PER_DAY

Wire into main.py with ONE line:
    from scheduled_predictor import register_predictor_jobs
    register_predictor_jobs(scheduler)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Force python-dotenv to load .env with comment-stripping BEFORE
# importing config.  Some systemd versions don't strip inline comments
# in EnvironmentFile, polluting values like "0  # comment" → int() fails.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)
except ImportError:
    pass

from station_meta import CITY_STATIONS  # type: ignore
from polymarket  import search_temp_high_events  # type: ignore
from config      import DB_PATH  # type: ignore

# Reuse the intraday predictor's plumbing instead of duplicating it
from scripts.intraday_predictor import (  # type: ignore
    fetch_nws_today_obs,
    fetch_nws_obs_with_raw,        # returns (hourly_max, raw_cycles) in one call
    fetch_nws_today_forecast,
    fetch_openmeteo_today,        # kept as fallback if NWS unreachable
    compute_neighbor_signal,
    predict_bins,
    vector_mean_dir,
    deg_to_cardinal,
    _load_calibration,
)
try:
    from scripts.find_nearby_stations import US_CITY_STATES  # type: ignore
except Exception:
    US_CITY_STATES = {}

log = logging.getLogger("scheduled_predictor")


# ---------------------------------------------------------------------------
# Tunable params — env-var overridable
# ---------------------------------------------------------------------------

PREDICTOR_MODE       = os.getenv("PREDICTOR_MODE", "paper").strip().lower()
# Flat position size — used in BOTH paper and live unless USE_KELLY=1.
# Live mode placing real orders should start with a small flat size so
# bugs/miscalibrations cost dollars, not hundreds.
FLAT_STAKE_USD       = float(os.getenv("PREDICTOR_FLAT_STAKE_USD", "5"))
USE_KELLY            = os.getenv("PREDICTOR_USE_KELLY", "0").strip() in ("1", "true", "yes")
PAPER_BANKROLL_USD   = float(os.getenv("PAPER_BANKROLL_USD",     "1000"))
MIN_EDGE             = float(os.getenv("PREDICTOR_MIN_EDGE",     "0.10"))
# Tiered edge: when market is cheap (mkt_p < HIGH_MKT_THRESHOLD), accept a
# smaller edge.  Rationale: the MIN_EDGE=0.10 gate is meant to filter out
# bins that are already expensively priced — if mkt_p is low (say 0.4), a
# small edge of 0.05 represents +12.5% expected return per dollar, which is
# still a worthwhile bet.  At mkt_p=0.90 the same 0.05 edge is only +5.5%
# expected return AND you lose your full stake on the (likely) wrong side,
# so the stricter MIN_EDGE protects you there.
MIN_EDGE_LOW_MKT     = float(os.getenv("PREDICTOR_MIN_EDGE_LOW_MKT", "0.05"))
HIGH_MKT_THRESHOLD   = float(os.getenv("PREDICTOR_HIGH_MKT_THRESHOLD", "0.75"))
# Market-probability floor: refuse to buy any bin whose market_p is below
# this threshold.  Catches catastrophic model miscalibrations — e.g., the
# 2026-06-10 Denver bug where the model computed our_p=1.0 for "≤83°F"
# while the market priced it at mkt_p=0.014.  When market is THAT skeptical,
# the model is almost certainly wrong (consensus aggregates a lot of info).
# Default 0.15 = blocks any bin the market deems <15% likely.
MIN_MARKET_PROB = float(os.getenv("PREDICTOR_MIN_MARKET_PROB", "0.15"))

# === Buy mode (2026-06-12) ===
# Two strategies for deciding whether the top-P bin is buyable:
#
#   "edge"        — default; existing behavior.  Requires edge (our_p -
#                    market_p) to exceed MIN_EDGE (or MIN_EDGE_LOW_MKT
#                    when market_p is in the cheap tier).  This is the
#                    classic edge-trading strategy: buy when we think
#                    we know more than the market.
#
#   "probability" — buy whenever our_p >= MIN_PROB_TO_BUY, regardless
#                    of edge or market price.  The bet is on our model
#                    being right, not on outperforming the market.
#                    Useful when you have high model confidence and
#                    don't care whether the market agrees.
#
# Other gates (MIN_MARKET_PROB sanity floor, W4 liquidity cap,
# priced_in, thin_book, dedup, trade/exposure caps) apply in BOTH
# modes.  Only the edge gate switches.
#
# To use probability mode:
#   PREDICTOR_BUY_MODE=probability
#   PREDICTOR_MIN_PROB_TO_BUY=0.50    # adjust to taste
PREDICTOR_BUY_MODE = os.getenv("PREDICTOR_BUY_MODE", "edge").lower()
PREDICTOR_MIN_PROB_TO_BUY = float(os.getenv("PREDICTOR_MIN_PROB_TO_BUY", "0.50"))

# W4 — market-anchored risk cap.  Blowup-preventer for the case where a
# liquid market disagrees with our model by a huge margin.  Reasoning:
# our_p = 95% and mkt_p = 5% on a $25k-liquid market is almost certainly
# either (a) a stale model price (we missed an NWS update / NBM cycle)
# or (b) a bug.  Either way, we shouldn't bet the farm.  This veto
# operates ENTIRELY at the gate layer — it never touches the edge
# calc, so our_p stays clean and the diagnostic signal stays measurable.
#
# Disable by setting MARKET_DISAGREEMENT_LIQ_THRESHOLD=0 (any value of
# liquidity will compare false to that gate) or
# MARKET_DISAGREEMENT_PP_THRESHOLD=1.1 (no real edge ever exceeds 1.0).
MARKET_DISAGREEMENT_LIQ_THRESHOLD = float(
    os.getenv("PREDICTOR_MARKET_DISAGREEMENT_LIQ_THRESHOLD", "10000"))
MARKET_DISAGREEMENT_PP_THRESHOLD  = float(
    os.getenv("PREDICTOR_MARKET_DISAGREEMENT_PP_THRESHOLD", "0.40"))

# Cold-start detection (data-quality contract).  If the first scan of a
# city's day is at or after this local hour, NWS /forecastHourly may
# have returned only evening cooling periods on that very first fetch —
# the recovery helper has no higher prior value to draw on, so
# forecast_high_c for that city today may be the evening cooling curve,
# not the actual day's high.  Per the contract, these rows get
# `cold_start_suspect` appended to their data_quality_flag, which the
# sizing multiplier picks up to apply a haircut (default 0.30).
COLD_START_PEAK_HOUR_LOCAL = int(
    os.getenv("PREDICTOR_COLD_START_PEAK_HOUR_LOCAL", "14"))

# Sizing scalar multipliers per data-quality flag — see
# docs/data_quality_contract.md.  Relative tiers all at 1.00 (no
# informational haircut today because no PRIMARY tier exists yet),
# absolute-trustability tiers at 0.30 (uncalibrated cities, cold-start
# suspects), block tier at 0.00.
DATA_QUALITY_SIZE_PRIMARY                = float(
    os.getenv("PREDICTOR_DQ_SIZE_PRIMARY", "1.00"))
DATA_QUALITY_SIZE_EMPIRICAL              = float(
    os.getenv("PREDICTOR_DQ_SIZE_EMPIRICAL", "1.00"))
DATA_QUALITY_SIZE_GAUSSIAN               = float(
    os.getenv("PREDICTOR_DQ_SIZE_GAUSSIAN", "1.00"))
DATA_QUALITY_SIZE_GAUSSIAN_DEFAULT_SIGMA = float(
    os.getenv("PREDICTOR_DQ_SIZE_GAUSSIAN_DEFAULT_SIGMA", "0.30"))
DATA_QUALITY_SIZE_COLD_START_SUSPECT     = float(
    os.getenv("PREDICTOR_DQ_SIZE_COLD_START_SUSPECT", "0.30"))
DATA_QUALITY_SIZE_BLOCK                  = float(
    os.getenv("PREDICTOR_DQ_SIZE_BLOCK", "0.00"))

MIN_LIQUIDITY_USD    = float(os.getenv("PREDICTOR_MIN_LIQUIDITY", "300"))
MIN_TRIGGER_HOUR     = int  (os.getenv("PREDICTOR_MIN_HOUR",     "13"))
MAX_TRIGGER_HOUR     = int  (os.getenv("PREDICTOR_MAX_HOUR",     "22"))
MAX_DAILY_EXPOSURE   = float(os.getenv("PREDICTOR_MAX_DAILY_EXP", "200"))
MAX_TRADES_PER_DAY   = int  (os.getenv("PREDICTOR_MAX_TRADES",   "25"))
# Upper price ceiling: bins priced >= this are "priced in" — the wrong-
# side stake-loss risk dominates any remaining edge.  Default 0.95.
MAX_MARKET_PRICE     = float(os.getenv("PREDICTOR_MAX_MKT_PRICE", "0.95"))
MAX_BINS_PER_EVENT   = int  (os.getenv("PREDICTOR_MAX_BINS_PER_EVENT", "1"))
KELLY_FRACTION       = float(os.getenv("PREDICTOR_KELLY_FRAC",   "0.25"))
MAX_PCT_PER_TRADE    = float(os.getenv("PREDICTOR_MAX_PCT",      "0.05"))
MIN_STAKE_USD        = float(os.getenv("PREDICTOR_MIN_STAKE",    "2.00"))
SCAN_INTERVAL_MIN    = int  (os.getenv("PREDICTOR_SCAN_MIN",     "15"))
# Topup logic — when actual filled position is below target stake, top up.
# TOPUP_TOLERANCE_PCT: position is "at target" when filled >= target * (1 - tol).
#   Default 0.05 = stop topping up when within 5% of target.
# TOPUP_MIN_USD: don't bother attempting topup orders smaller than this.
TOPUP_TOLERANCE_PCT  = float(os.getenv("PREDICTOR_TOPUP_TOLERANCE_PCT", "0.05"))
TOPUP_MIN_USD        = float(os.getenv("PREDICTOR_TOPUP_MIN_USD",       "1.50"))
# Per-contract daily $ ceiling.  Belt-and-suspenders against runaway
# topups when target_stake re-computes higher each scan against an
# updating cost basis.  No single contract gets more than this many $
# of fresh capital today, regardless of what target_stake says.
# Disable with 0.
MAX_PER_CONTRACT_USD = float(os.getenv("PREDICTOR_MAX_PER_CONTRACT_USD", "15.00"))
# Weather cache TTL — NWS forecast + observation calls cached for this
# many seconds.  Polymarket prices are always refreshed every scan; only
# weather data is cached.  Default: 300s (5 min).  Combined with
# PREDICTOR_SCAN_MIN=2, this gives Polymarket polling every 2 min and
# weather refresh every 5 min, matching real-world data-change rates.
WEATHER_CACHE_SEC    = int  (os.getenv("PREDICTOR_WEATHER_CACHE_SEC", "300"))

# In-memory weather cache per city: {city: (fetched_at_epoch, {obs, forecast, ...})}
_WEATHER_CACHE: dict[str, tuple[float, dict]] = {}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_predictor_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at_utc      TEXT    NOT NULL,
    mode                TEXT    NOT NULL,        -- 'paper' | 'live'
    city                TEXT    NOT NULL,
    settlement_station  TEXT,
    event_date          TEXT,
    event_id            TEXT,
    contract_id         TEXT,
    yes_token_id        TEXT,
    bin_label           TEXT,
    bin_range_low       REAL,
    bin_range_high      REAL,
    unit                TEXT,
    our_prob            REAL,
    market_prob         REAL,
    edge                REAL,
    liquidity_usd       REAL,
    -- decision
    action              TEXT,                    -- PAPER_BUY | SKIP | AVOID
    gate_blocked_by     TEXT,                    -- which gate failed (if any)
    recommended_stake_usd  REAL,
    recommended_limit_price REAL,
    -- context snapshot
    current_hour_local  INTEGER,
    observed_max_c      REAL,
    observed_peak_hour  INTEGER,
    forecast_high_c     REAL,
    forecast_peak_hour  INTEGER,
    mu_c                REAL,
    sigma_c             REAL,
    wind_octant         TEXT,
    upwind_signal_strength REAL,
    -- Gamma's per-bin resolution state at scan time.  1 = market has
    -- settled (sub-market closed); 0 = still tradeable.  Used by the
    -- dashboard to distinguish RESOLVED held positions from LIVE ones
    -- without inferring market state from position $-value.
    market_closed       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pps_city_date
    ON paper_predictor_signals(city, event_date);
CREATE INDEX IF NOT EXISTS idx_pps_action_scanned
    ON paper_predictor_signals(action, scanned_at_utc);

-- LIVE order log — populated when PREDICTOR_MODE=live and execute_signal
-- returns a placed order.  Links back to paper_predictor_signals via
-- signal_id so we can compare intended vs actual fill.
CREATE TABLE IF NOT EXISTS live_predictor_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER,          -- FK → paper_predictor_signals.id
    placed_at_utc   TEXT NOT NULL,
    city            TEXT,
    event_date      TEXT,
    contract_id     TEXT,
    bin_label       TEXT,
    side            TEXT,
    stake_usd       REAL,
    limit_price     REAL,
    order_id        TEXT,
    position_id     INTEGER,
    status          TEXT,             -- placed | filled | failed | skip | error
    response        TEXT,             -- raw execute_signal response (JSON str)
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_lpo_placed
    ON live_predictor_orders(placed_at_utc);

-- Raw METAR persistence.  Captures every NWS observation cycle as it
-- comes in (rawMessage + parsed fields), one row per (icao, cycle).
-- Cheap forward-path evidence: when a settle_divergence case surfaces
-- weeks later, the original METAR strings around peak hour are
-- recoverable here instead of having to reconstruct from Iowa State.
-- Idempotent insert via UNIQUE — same cycle pulled multiple times in
-- a scan window is a no-op.
CREATE TABLE IF NOT EXISTS raw_metar_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    icao                 TEXT NOT NULL,
    event_date           TEXT NOT NULL,
    cycle_timestamp_utc  TEXT NOT NULL,
    raw_message          TEXT,
    temp_c               REAL,
    dewpoint_c           REAL,
    wind_dir_deg         REAL,
    wind_speed_mps       REAL,
    present_weather      TEXT,
    -- 'tenths' | 'whole' | 'missing' — see precision-handling block in
    -- intraday_predictor.py for the semantics.  Surfaces whether temp_c
    -- came from the METAR T-group (precise to ±0.05°C) or the body
    -- value (whole-°C body, conservative -0.5°C lower bound applied).
    temp_precision       TEXT,
    persisted_at_utc     TEXT NOT NULL,
    UNIQUE(icao, cycle_timestamp_utc)
);
CREATE INDEX IF NOT EXISTS idx_rml_icao_date
    ON raw_metar_log(icao, event_date);

-- Resolution observations — Phase 0a of the HRRR ceiling plan.
-- One row per resolved Polymarket market, decomposing the
-- bot-vs-settlement gap into its component sources:
--
--   bot_observed_max_c    : what the bot recorded for the day
--   metar_peak_body_c     : whole-°C body value at peak hour (no T-group)
--   metar_peak_t_group_c  : T-group tenths-precision value at peak synoptic
--   wunderground_high_c   : Wunderground's daily high display (settlement
--                            reference for US markets)
--   winning_range_low/high: from resolutions — the bin that won
--
-- This lets Phase 0b decompose any bot-vs-settlement gap:
--   (body → t_group) gap = body-rounding error (which T-group fix
--                            this turn should have closed)
--   (t_group → wunderground) gap = DSM aggregation / source mismatch
--                                    (separate fix if non-zero)
--   (wunderground → winning_bin) gap = should be zero if Wunderground
--                                        is the settlement source
--
-- Captured by bot/scripts/capture_resolution_truth.py, run nightly.
-- WRITE-ONLY from the scan loop's perspective; never read by trading
-- decisions.
CREATE TABLE IF NOT EXISTS resolution_observations (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id               TEXT NOT NULL,
    city                   TEXT,
    event_date             TEXT,
    icao                   TEXT,
    bot_observed_max_c     REAL,
    metar_peak_body_c      REAL,
    metar_peak_t_group_c   REAL,
    metar_peak_cycle_utc   TEXT,
    wunderground_high_c    REAL,
    winning_range_low      REAL,
    winning_range_high     REAL,
    winning_bin_label      TEXT,
    captured_at_utc        TEXT NOT NULL,
    capture_notes          TEXT,
    UNIQUE(event_id)
);
CREATE INDEX IF NOT EXISTS idx_ro_city_date
    ON resolution_observations(city, event_date);

-- Invariant guard violations.  Permanent observational record of every
-- within-day monotonicity / coherence violation surfaced by the guards
-- in scripts/invariant_guards.py.  WRITE-ONLY from this file's perspective:
-- the scan loop and the prediction path NEVER read this table.  See the
-- OBSERVATIONAL FOREVER design rule in invariant_guards.py.
--
-- Idempotent via UNIQUE — re-running guards on the same scan is a no-op.
CREATE TABLE IF NOT EXISTS guard_violations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at_utc TEXT NOT NULL,
    scan_at_utc     TEXT NOT NULL,
    guard_name      TEXT NOT NULL,
    city            TEXT NOT NULL,
    event_date      TEXT,
    prev_value      REAL,
    curr_value      REAL,
    delta           REAL,
    detail          TEXT,
    UNIQUE(scan_at_utc, guard_name, city)
);
CREATE INDEX IF NOT EXISTS idx_gv_detected
    ON guard_violations(detected_at_utc);
CREATE INDEX IF NOT EXISTS idx_gv_city_guard
    ON guard_violations(city, guard_name);
"""


def ensure_schema() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA_SQL)
        # === Live-running migrations ===
        # Schemas above use CREATE TABLE IF NOT EXISTS, so columns added
        # after the table first existed need explicit ALTERs.  Each block
        # is idempotent — the OperationalError is swallowed if the column
        # already exists.
        def _add_column(table: str, col: str, type_sql: str) -> None:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {type_sql}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # market_closed: per-bin Gamma resolution state at scan time.
        # Lets the dashboard show a RESOLVED badge for held tokens whose
        # underlying market has settled (instead of inferring "is this
        # market open?" from a $-value dust filter on positions).
        _add_column("paper_predictor_signals", "market_closed", "INTEGER DEFAULT 0")
        # data_quality_flag: reserved for W2 distribution rewrite.  Will
        # carry one of {primary, empirical_fallback, gaussian_fallback}
        # indicating which probability path produced our_prob for this
        # row.  Defaults to NULL until W2 ships.  Created now so audits
        # and dashboards can query it without a downstream migration.
        _add_column("paper_predictor_signals", "data_quality_flag", "TEXT")
        # cooling_confidence: from estimate_day_high_dist's detect_cooling
        # output.  Persisted so the invariant_guards module can check the
        # post-peak monotonicity invariant (cooling_confidence should be
        # non-decreasing once forecast_peak_hour has passed; oscillation
        # is the bin-lock-discontinuity churn surfacing as data).
        _add_column("paper_predictor_signals", "cooling_confidence", "REAL")
        # temp_precision: per METAR cycle, records whether temp_c was
        # derived from the T-group (tenths precision, ±0.05°C) or the
        # body value (whole-°C, conservatively bound to lower edge).
        # See the precision-handling block in intraday_predictor.py.
        # Surfaced for diagnostic queries: SELECT temp_precision, COUNT(*)
        # FROM raw_metar_log GROUP BY temp_precision tells you what
        # fraction of obs are precision-quality at any given station.
        _add_column("raw_metar_log", "temp_precision", "TEXT")
        conn.commit()


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def kelly_stake(edge: float, market_price: float, bankroll: float) -> float:
    """Quarter-Kelly stake (capped at MAX_PCT_PER_TRADE * bankroll).
    Only used when PREDICTOR_USE_KELLY=1.  Default is flat sizing."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    p = max(0.0, min(1.0, market_price + edge))
    b = (1.0 / market_price) - 1.0
    if b <= 0:
        return 0.0
    f_star = (p * b - (1 - p)) / b
    if f_star <= 0:
        return 0.0
    frac = min(KELLY_FRACTION * f_star, MAX_PCT_PER_TRADE)
    return round(frac * bankroll, 2)


def compute_stake(edge: float, market_price: float, bankroll: float) -> float:
    """Choose stake based on PREDICTOR_USE_KELLY.  Default: flat $X.
    Flat is safer for the initial live rollout — Kelly amplifies any
    miscalibration error in our edge estimate, and we haven't yet
    validated the predictor against months of resolved outcomes.
    """
    if USE_KELLY:
        return kelly_stake(edge, market_price, bankroll)
    return round(FLAT_STAKE_USD, 2)


def marketable_limit(market_price: float) -> float:
    """At-the-ask limit with 1¢ buffer to cross the spread."""
    return round(min(0.99, market_price + 0.01), 4)


# ---------------------------------------------------------------------------
# Gates (in the order they're checked — earlier = cheaper)
# ---------------------------------------------------------------------------

def evaluate_gates(*, current_hour: int, edge: float, market_p: float,
                    liquidity: float, deployed_today: float,
                    trades_today: int, already_acted: bool,
                    our_p: float | None = None) -> tuple[bool, str]:
    """Returns (pass, reason).  reason is empty when pass=True.

    our_p is required when PREDICTOR_BUY_MODE="probability"; in edge
    mode (default) it can be omitted and is derived as market_p + edge.
    """
    # Derive our_p from edge + market_p when not provided.  Used by
    # tests that don't yet pass our_p explicitly.
    if our_p is None:
        our_p = market_p + edge

    if current_hour < MIN_TRIGGER_HOUR:
        return False, f"too_early (hour={current_hour} < {MIN_TRIGGER_HOUR})"
    if current_hour > MAX_TRIGGER_HOUR:
        return False, f"too_late (hour={current_hour} > {MAX_TRIGGER_HOUR})"
    # Market sanity floor — bin must have at least MIN_MARKET_PROB market
    # confidence.  Catches catastrophic model errors: if market thinks a
    # bin has <15% chance, our model claiming 100% is almost certainly
    # a bug, not edge.  Applies in BOTH edge and probability modes —
    # the model-vs-market-sanity check is independent of which strategy
    # the operator chose for the buy decision.
    if market_p < MIN_MARKET_PROB:
        return False, (f"market_too_skeptical (mkt={market_p:.3f} < "
                       f"{MIN_MARKET_PROB:.2f})")
    # Buy-mode dispatch.  Edge mode (default) requires meaningful edge
    # over the market; probability mode requires high model confidence
    # regardless of edge.  Other gates (W4, priced_in, thin_book,
    # dedup, trade caps) apply in both modes.
    if PREDICTOR_BUY_MODE == "probability":
        if our_p < PREDICTOR_MIN_PROB_TO_BUY:
            return False, (f"low_prob (our_p={our_p:.3f} < "
                           f"{PREDICTOR_MIN_PROB_TO_BUY:.2f})")
    else:
        # Tiered edge gate: stricter when market_p is "expensive"
        # (>= 0.75), looser when cheap (<0.75).  MIN_EDGE=0.10 was
        # designed to filter out high-priced bins where we'd lose the
        # full stake on the wrong side; for cheaper bins the same edge
        # is more attractive on expected-value terms.
        required_edge = MIN_EDGE if market_p >= HIGH_MKT_THRESHOLD else MIN_EDGE_LOW_MKT
        if edge < required_edge:
            return False, (f"low_edge ({edge:+.3f} < {required_edge:.2f}, "
                           f"mkt_p={market_p:.2f})")
    # W4 — market-anchored risk cap.  On a LIQUID market, an extreme
    # disagreement (>40pp by default) between our model and the market
    # is more likely a stale-model / bug case than genuine edge.  Veto
    # without affecting the edge calc — diagnostic signal stays clean.
    # Sits AFTER the low_edge gate so it doesn't fire on normal
    # small-edge buys; only the suspicious "huge edge on liquid book"
    # case trips it.
    if (liquidity >= MARKET_DISAGREEMENT_LIQ_THRESHOLD
        and edge >= MARKET_DISAGREEMENT_PP_THRESHOLD):
        return False, (f"liquid_market_strong_disagreement "
                        f"(edge={edge:+.2f} >= {MARKET_DISAGREEMENT_PP_THRESHOLD:.2f}, "
                        f"liq=${liquidity:.0f} >= "
                        f"${MARKET_DISAGREEMENT_LIQ_THRESHOLD:.0f})")
    if market_p >= MAX_MARKET_PRICE:
        return False, f"priced_in (mkt={market_p:.3f} ≥ {MAX_MARKET_PRICE:.2f})"
    if liquidity < MIN_LIQUIDITY_USD:
        return False, f"thin_book (liq=${liquidity:.0f} < ${MIN_LIQUIDITY_USD:.0f})"
    if already_acted:
        return False, "dedup_today"
    if trades_today >= MAX_TRADES_PER_DAY:
        return False, f"trades_cap ({trades_today} >= {MAX_TRADES_PER_DAY})"
    if deployed_today >= MAX_DAILY_EXPOSURE:
        return False, f"exposure_cap (${deployed_today:.0f} >= ${MAX_DAILY_EXPOSURE:.0f})"
    return True, ""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _today_utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _action_for_mode(mode: str) -> str:
    """Map mode → the action string written for a successful BUY."""
    return "LIVE_BUY" if mode == "live" else "PAPER_BUY"


def already_acted_today(conn, event_id: str, contract_id: str,
                          mode: str) -> bool:
    """True if THIS exact (event,bin) was already bought IN THIS MODE
    for this event.  We don't filter by scan_date — each Polymarket
    event has a unique event_id and a single resolution day, so a
    "previously bought" check should look at the event's lifetime.
    Previously this used substr(scanned_at_utc) = today_utc, which
    reset the counter at UTC midnight and caused duplicate buys on
    events that hadn't yet resolved in the city's local timezone."""
    row = conn.execute(
        """
        SELECT 1 FROM paper_predictor_signals
        WHERE event_id = ? AND contract_id = ? AND action = ?
        LIMIT 1
        """,
        (event_id, contract_id, _action_for_mode(mode)),
    ).fetchone()
    return row is not None


def event_has_buy_today(conn, event_id: str, mode: str) -> bool:
    """True if ANY bin of this event was bought today in this mode.
    Convenience wrapper around event_buys_today_count()."""
    return event_buys_today_count(conn, event_id, mode) > 0


def event_buys_today_count(conn, event_id: str, mode: str) -> int:
    """Number of DISTINCT bins bought for this event in this mode.
    Used by MAX_BINS_PER_EVENT cap.

    Counts DISTINCT contract_ids so that topup orders (multiple BUY
    rows for the same contract_id) don't inflate the count.  Each
    distinct (event, contract) pair = one bin bought, regardless of
    how many orders it took to fill.
    """
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT contract_id) FROM paper_predictor_signals
        WHERE event_id = ? AND action = ?
        """,
        (event_id, _action_for_mode(mode)),
    ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Polymarket positions API — used for actual-deployed lookup + topup logic
# ---------------------------------------------------------------------------

POLYMARKET_DATA_API = "https://data-api.polymarket.com"


def _get_proxy_address() -> str | None:
    """Find the wallet/proxy address that holds our Polymarket positions."""
    for var in ("POLYMARKET_FUNDER_ADDRESS", "POLYMARKET_PROXY_ADDRESS",
                 "POLY_PROXY", "BROWSER_ADDRESS", "PROXY_ADDRESS",
                 "WALLET_ADDRESS"):
        v = os.getenv(var)
        if v and isinstance(v, str) and v.lower().startswith("0x") and len(v) == 42:
            return v
    return None


def fetch_polymarket_positions_by_token() -> dict[str, dict] | None:
    """Returns {token_id: {size, avg_price, deployed_usdc, cur_price, cash_pnl}}.

    NO VALUE FILTER.  Returns EVERY token with size > 0 that the wallet
    holds, including resolved-to-zero positions.  This is the raw
    "what does our wallet hold?" signal used to build the three derived
    signals downstream:

      HELD_TOKENS:         set of token_ids in this dict
                           → "have I bought this contract?"  (dedup, cap)
      DEPLOYED_BY_TOKEN:   size * avg_price per token
                           → "how much cost basis is deployed?"  (topup)
      MARKET_OPEN per bin: comes from Gamma's `closed` flag at scan time,
                           NOT from this dict
                           → "is this position still tradeable?" (display
                              + abort fresh-buy gate)

    Return value semantics (important for callers):
      * dict (possibly EMPTY {}) — API call succeeded.  Empty = wallet
        holds zero tokens.  Callers should TRUST this.
      * None — API call FAILED (network, missing creds).  Callers should
        fall back to DB-derived sums conservatively.
    """
    addr = _get_proxy_address()
    if not addr:
        log.debug("no proxy address in env — skipping live position fetch")
        return None
    try:
        import urllib.request, urllib.parse
        params = {"user": addr, "sizeThreshold": "0.01",
                   "limit": "500", "sortBy": "CURRENT"}
        url = f"{POLYMARKET_DATA_API}/positions?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url,
                                       headers={"User-Agent": "polymarket-weather/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        out: dict[str, dict] = {}
        for p in data:
            token = p.get("asset") or p.get("token_id") or p.get("tokenId")
            if not token:
                continue
            size = float(p.get("size") or 0)
            avg  = float(p.get("avgPrice") or p.get("avg_price") or 0)
            cur  = float(p.get("curPrice") or p.get("cur_price") or 0)
            if size <= 0 or avg <= 0:
                continue
            out[str(token)] = {
                "size":          size,
                "avg_price":     avg,
                "deployed_usdc": size * avg,
                "cur_price":     cur,
                "cash_pnl":      float(p.get("cashPnl") or p.get("cash_pnl") or 0),
            }
        log.info(f"Polymarket positions: {len(out)} held tokens fetched "
                  f"(no value filter; market_open is decided per-bin from Gamma)")
        return out
    except Exception as e:
        log.warning(f"Polymarket positions API failed: {e} — topup falls back to DB")
        return None


def get_actual_deployed_usd(conn, event_id: str, contract_id: str,
                              yes_token_id: str | None, mode: str,
                              live_positions: dict[str, dict] | None) -> float:
    """Returns actual $ deployed for THIS contract in this mode.

    For LIVE: trusts Polymarket API as the source of truth.
      * API has the position → actual size × avgPrice
      * API does NOT have the position → 0  (position is closed, was
        cancelled, never filled, etc.  Slot is open for re-buy/topup.)
      * API unreachable (live_positions=None) → fall back to DB sum
        (defensive, won't fire topups but won't lose data either)

    For PAPER: sums recommended_stake_usd across all BUY rows for this
    contract (paper always fills full, so DB sum = total deployed).
    """
    if mode == "live":
        if live_positions is None:
            # API unreachable — defensive fallback to DB sum so we don't
            # try to re-buy when we genuinely can't tell what we hold.
            pass
        else:
            pos = live_positions.get(str(yes_token_id)) if yes_token_id else None
            return pos["deployed_usdc"] if pos else 0.0
    # Paper mode (or live with API unreachable) — sum from DB
    action = _action_for_mode(mode)
    row = conn.execute(
        """SELECT COALESCE(SUM(recommended_stake_usd), 0)
           FROM paper_predictor_signals
           WHERE event_id = ? AND contract_id = ? AND action = ?""",
        (event_id, contract_id, action),
    ).fetchone()
    return float(row[0]) if row else 0.0


def deployed_today_usd(conn, mode: str) -> float:
    """Sum of stakes for today, IN THIS MODE only."""
    today = _today_utc_date_str()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(recommended_stake_usd), 0) FROM paper_predictor_signals
        WHERE action = ? AND substr(scanned_at_utc, 1, 10) = ?
        """,
        (_action_for_mode(mode), today),
    ).fetchone()
    return float(row[0]) if row else 0.0


def deployed_today_for_contract(conn, mode: str, contract_id: str) -> float:
    """DEPRECATED — do NOT use for per-contract cap decisions or any
    other "how much have we deployed?" question.

    This function sums `recommended_stake_usd` across today's LIVE_BUY
    rows for the given contract.  That count includes orders that
    NEVER FILLED — limit orders that the market moved past, cancelled
    attempts, etc.  Using it for the per-contract daily cap caused a
    bug on Dallas 2026-06-12: $6.68 actually deployed on Polymarket,
    $15 of signal-row intent in the DB, cap blocked legitimate topups.

    The right source of truth for "how much $ is in this contract" is
    `get_actual_deployed_usd()`, which reads cost basis from the
    Polymarket positions API.

    Kept callable for any historical reporting that genuinely wants the
    "sum of attempted-stake intent" — but flag that intent at the call
    site, don't pretend it's deployed capital.
    """
    today = _today_utc_date_str()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(recommended_stake_usd), 0) FROM paper_predictor_signals
        WHERE action = ? AND substr(scanned_at_utc, 1, 10) = ?
              AND contract_id = ?
        """,
        (_action_for_mode(mode), today, contract_id),
    ).fetchone()
    return float(row[0]) if row else 0.0


def trades_today(conn, mode: str) -> int:
    """Count of today's buys IN THIS MODE only."""
    today = _today_utc_date_str()
    row = conn.execute(
        """
        SELECT COUNT(*) FROM paper_predictor_signals
        WHERE action = ? AND substr(scanned_at_utc, 1, 10) = ?
        """,
        (_action_for_mode(mode), today),
    ).fetchone()
    return int(row[0]) if row else 0


def pending_contracts_today(conn) -> set[str]:
    """Contracts with at least one order placed today that hasn't yet
    transitioned to filled / cancelled / stale / error.

    These count as "committed" for cap purposes — a contract with a
    pending order should NOT permit a new bin entry for the same event
    (the NYC 2026-06-12 race condition).  The reconciliation sweep runs
    at the start of every scan, so by the time this query fires, any
    orders that actually filled have already been promoted out of
    'placed' status.

    Returns empty set in paper mode or when the table is empty —
    pending-order tracking is a live-mode concept.
    """
    today = _today_utc_date_str()
    try:
        rows = conn.execute(
            """SELECT DISTINCT contract_id FROM live_predictor_orders
               WHERE substr(placed_at_utc, 1, 10) = ?
                 AND status = 'placed'
                 AND contract_id IS NOT NULL""",
            (today,),
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r[0] if not hasattr(r, "keys") else r["contract_id"] for r in rows}


def pending_stake_for_contract_today(conn, contract_id: str) -> float:
    """Sum of stakes for orders on `contract_id` placed today that
    haven't yet filled / cancelled / staled.

    These count as "committed" toward the target stake — the Houston
    2026-06-12 race condition.  Without this, the per-bin loop sees
    actual_deployed (filled-only) below target and keeps placing more
    orders, then they all eventually fill and total deployed blows past
    the target.

    Returns 0.0 in paper mode or when the table is unavailable.
    """
    today = _today_utc_date_str()
    try:
        row = conn.execute(
            """SELECT COALESCE(SUM(stake_usd), 0)
               FROM live_predictor_orders
               WHERE substr(placed_at_utc, 1, 10) = ?
                 AND status = 'placed'
                 AND contract_id = ?""",
            (today, contract_id),
        ).fetchone()
    except sqlite3.OperationalError:
        return 0.0
    return float(row[0]) if row else 0.0


def write_signal(conn, row: dict) -> int:
    """Insert and return the new row's id."""
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    cur = conn.execute(
        f"INSERT INTO paper_predictor_signals ({','.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return int(cur.lastrowid)


def write_live_order(conn, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(
        f"INSERT INTO live_predictor_orders ({','.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()


def is_cold_start_day(conn, city: str, event_date: str, tz_str: str,
                        candidate_scan_at_utc: str) -> bool:
    """True if the first scan of (city, event_date) is at or after
    COLD_START_PEAK_HOUR_LOCAL in local time.

    A cold-start day's first NWS fetch may have caught only the evening
    cooling curve.  Subsequent scans inherit the buggy forecast_high
    via the recovery helper (since there's no higher prior to recover
    from), so cold_start_suspect is a per-DAY label not a per-scan one.

    The candidate_scan_at_utc parameter is used when no prior scans
    exist for this (city, event_date) — we treat THIS scan as the
    candidate "first scan" so a single late-day cold-start scan still
    gets correctly flagged.
    """
    from zoneinfo import ZoneInfo
    row = conn.execute(
        """SELECT MIN(scanned_at_utc) FROM paper_predictor_signals
           WHERE city = ? AND event_date = ?""",
        (city, event_date),
    ).fetchone()
    first_scan_at_utc = row[0] if row else None
    if not first_scan_at_utc:
        first_scan_at_utc = candidate_scan_at_utc
    try:
        tz = ZoneInfo(tz_str)
        first_t = datetime.fromisoformat(
            first_scan_at_utc.replace("Z", "+00:00"))
        first_hour_local = first_t.astimezone(tz).hour
    except (ValueError, KeyError, AttributeError):
        return False
    return first_hour_local >= COLD_START_PEAK_HOUR_LOCAL


# Lookup table for sizing-scalar computation.  Stored as module-level so
# tests can override (e.g. monkeypatching DATA_QUALITY_SIZE_GAUSSIAN to
# verify the haircut path).  Recomputed at call time, not memoized, so
# env-var overrides take effect immediately.
def _dq_size_by_flag() -> dict[str, float]:
    return {
        "primary":                  DATA_QUALITY_SIZE_PRIMARY,
        "empirical":                DATA_QUALITY_SIZE_EMPIRICAL,
        "gaussian":                 DATA_QUALITY_SIZE_GAUSSIAN,
        "gaussian_default_sigma":   DATA_QUALITY_SIZE_GAUSSIAN_DEFAULT_SIGMA,
        "cold_start_suspect":       DATA_QUALITY_SIZE_COLD_START_SUSPECT,
        "block":                    DATA_QUALITY_SIZE_BLOCK,
    }


def compute_data_quality_size_factor(flag_str: str | None) -> float:
    """Return the most conservative size multiplier across all flag
    components in flag_str.

    flag_str is a comma-separated list of flag values, each optionally
    suffixed with ":reason" (e.g. "primary_fallback:stale_11h").  We
    split on commas, strip the optional reason suffix, look up each
    base flag, and return the minimum factor — composable haircuts.

    Unknown flags don't contribute (treated as 1.00 — neutral).  Empty
    or None flag → 1.00.

    Examples:
       compute_data_quality_size_factor(None)               → 1.00
       compute_data_quality_size_factor("gaussian")         → 1.00 (today)
       compute_data_quality_size_factor("cold_start_suspect") → 0.30
       compute_data_quality_size_factor("gaussian,cold_start_suspect") → 0.30
       compute_data_quality_size_factor("primary_fallback:stale_11h,empirical") → 1.00
    """
    if not flag_str:
        return 1.00
    table = _dq_size_by_flag()
    factors: list[float] = []
    for component in flag_str.split(","):
        base = component.strip().split(":")[0]   # drop ":reason" suffix
        if base in table:
            factors.append(table[base])
    return min(factors) if factors else 1.00


def recover_persisted_day_forecast(conn, city: str, event_date: str,
                                      candidate_high_c: float,
                                      candidate_peak_hour: int | None,
                                      ) -> tuple[float, int | None]:
    """Defend against the NWS `/forecastHourly` evening-scan bug.

    NWS hourly only returns periods from "now" forward.  By late
    afternoon, "today's" remaining periods describe the evening cooling
    curve, not the day's actual high.  A late-day fetch therefore stores
    artificially-low forecast_high_c values — the column ratchets DOWN
    through the afternoon (verified against SF 2026-06-11: 28.33°C from
    07:04 UTC stable through 23:06 UTC, then declining to 17.22°C by
    23:24 UTC = 16:24 PDT, tracking the cooling curve exactly).

    Fix: if a PRIOR scan for this (city, event_date) recorded a higher
    forecast_high_c, use that.  Works as long as at least one earlier
    scan ran while the hourly endpoint still contained the day's peak
    periods.

    Cold-start days where no morning scan ran cannot be recovered —
    those rows must be filtered at calibration time (indicator: a
    settled day where bot's observed_max_c ended up higher than its
    own forecast_high_c by > 2°C).

    Does NOT punish the Open-Meteo fallback path: Open-Meteo's
    forecast_days=1 returns the full local day from midnight, so its
    candidate_high_c is already correct.  Helper takes max(persisted,
    candidate) so an accurate Open-Meteo candidate beats any stale
    persisted value.
    """
    row = conn.execute(
        """SELECT forecast_high_c, forecast_peak_hour
           FROM paper_predictor_signals
           WHERE city = ? AND event_date = ? AND forecast_high_c IS NOT NULL
           ORDER BY forecast_high_c DESC LIMIT 1""",
        (city, event_date),
    ).fetchone()
    if not row:
        return candidate_high_c, candidate_peak_hour
    prev_high = row[0] if not hasattr(row, "keys") else row["forecast_high_c"]
    prev_peak = row[1] if not hasattr(row, "keys") else row["forecast_peak_hour"]
    if prev_high is not None and prev_high > candidate_high_c:
        recovered_peak = int(prev_peak) if prev_peak is not None else candidate_peak_hour
        return float(prev_high), recovered_peak
    return candidate_high_c, candidate_peak_hour


def persist_raw_metar_cycles(conn, raw_cycles: list[dict]) -> int:
    """Insert raw METAR cycles into raw_metar_log.  Idempotent — the
    UNIQUE(icao, cycle_timestamp_utc) constraint means re-running this
    on the same cycles is a no-op.  Returns count of NEW rows inserted.

    Cheap forward-path evidence: when W1 needs to investigate a
    settle_divergence tuple, the raw METAR around peak hour for that
    (icao, date) is here.  Persisting now even though W0 hasn't run
    yet — evidence accumulates from this commit on, retroactive
    recovery is via Iowa State for the audit window only.
    """
    if not raw_cycles:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for c in raw_cycles:
        cur = conn.execute(
            """INSERT OR IGNORE INTO raw_metar_log
                 (icao, event_date, cycle_timestamp_utc, raw_message,
                  temp_c, dewpoint_c, wind_dir_deg, wind_speed_mps,
                  present_weather, temp_precision, persisted_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (c.get("icao"), c.get("event_date"), c.get("cycle_timestamp_utc"),
             c.get("raw_message"),
             c.get("temp_c"), c.get("dewpoint_c"),
             c.get("wind_dir_deg"), c.get("wind_speed_mps"),
             c.get("present_weather"), c.get("temp_precision"), now_iso),
        )
        inserted += cur.rowcount or 0
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# LIVE execution path — only invoked when PREDICTOR_MODE=live
# ---------------------------------------------------------------------------

_LIVE_CLOB_CLIENT = None   # lazily initialized; reused across bins in a scan

def _get_live_client():
    """Cache the CLOB client at module level to avoid re-auth every bin."""
    global _LIVE_CLOB_CLIENT
    if _LIVE_CLOB_CLIENT is not None:
        return _LIVE_CLOB_CLIENT
    try:
        from execution import get_clob_client
        _LIVE_CLOB_CLIENT = get_clob_client()
        if _LIVE_CLOB_CLIENT is None:
            log.warning("get_clob_client() returned None — live mode unavailable")
        return _LIVE_CLOB_CLIENT
    except Exception as e:
        log.error(f"failed to initialize CLOB client: {e}")
        return None


def execute_live(conn, signal_id: int, city: str, ev: dict, b: dict,
                   stake: float, limit_px: float) -> dict:
    """Build a signal dict in execute_signal's format and place the order.
    Records every attempt in live_predictor_orders regardless of outcome."""
    import json as _json
    from datetime import datetime as _dt, timezone as _tz

    placed_at = _dt.now(_tz.utc).isoformat()
    base_row = {
        "signal_id":     signal_id,
        "placed_at_utc": placed_at,
        "city":          city,
        "event_date":    ev.get("date"),
        "contract_id":   b.get("contract_id"),
        "bin_label":     b.get("label"),
        "side":          "YES",
        "stake_usd":     stake,
        "limit_price":   limit_px,
    }

    client = _get_live_client()
    if client is None:
        row = {**base_row, "order_id": None, "position_id": None,
                "status": "error", "response": None,
                "error": "clob_client_unavailable"}
        write_live_order(conn, row)
        return row

    try:
        from execution import execute_signal
    except Exception as e:
        row = {**base_row, "order_id": None, "position_id": None,
                "status": "error", "response": None,
                "error": f"execute_signal import failed: {e}"}
        write_live_order(conn, row)
        return row

    sig_for_exec = {
        "contract_id":      b["contract_id"],
        "yes_token_id":     b.get("yes_token_id"),
        "no_token_id":      None,   # YES-only strategy
        "recommended_side": "YES",
        "kelly_size":       stake,             # execute_signal uses this as $ stake
        "market_p":         b["market_prob"],
        "yes_price":        b["market_prob"],
        "city":             city,
        "date":             ev.get("date"),
        "event_id":         ev.get("event_id"),
        "gamma_market_id":  ev.get("gamma_market_id"),
        "model_prob":       b["our_prob"],
        "edge":             b["edge"],
        "strategy":         "intraday_predictor",
        "question":         ev.get("event_title") or "",
        "range_low":        b.get("range_low"),
        "range_high":       b.get("range_high"),
        "unit":             b.get("unit"),
    }
    try:
        result = execute_signal(sig_for_exec, client=client)
    except Exception as e:
        log.exception(f"execute_signal raised for {city} {b['label']}: {e}")
        row = {**base_row, "order_id": None, "position_id": None,
                "status": "error", "response": None, "error": str(e)}
        write_live_order(conn, row)
        return row

    status   = (result or {}).get("status", "unknown")
    order_id = (result or {}).get("order_id")
    position_id = (result or {}).get("position_id")
    row = {
        **base_row,
        "order_id":    order_id,
        "position_id": position_id,
        "status":      status,
        "response":    _json.dumps(result, default=str)[:1000],
        "error":       (result or {}).get("reason"),
    }
    write_live_order(conn, row)
    log.info(f"  LIVE order: {city} {b['label']} ${stake:.2f} @ {limit_px:.4f} "
             f"→ status={status} order_id={order_id}")
    return row


# ---------------------------------------------------------------------------
# Order reconciliation — ask the CLOB about resting orders and
# update their status to filled/cancelled.  Runs at the start of each
# scan (before placing new orders) so the topup math uses an accurate
# view of what's actually filled.
# ---------------------------------------------------------------------------

# How far back to look for orders to reconcile.  Older orders that are
# still 'placed' get marked as 'stale' so they stop blocking topups.
RECONCILE_LOOKBACK_HOURS = int(os.getenv("RECONCILE_LOOKBACK_HOURS", "12"))

def reconcile_pending_orders(conn: sqlite3.Connection) -> dict:
    """Sweep `live_predictor_orders` rows where status='placed' and ask
    the CLOB for definitive resolution.  Updates each row to one of:
       - filled    : order fully matched (size_matched >= original_size)
       - cancelled : order was cancelled/expired
       - stale     : couldn't determine after RECONCILE_LOOKBACK_HOURS
       - placed    : still resting (unchanged)
    Returns a summary dict for the scan log.
    """
    from datetime import datetime as _dt, timezone as _tz, timedelta
    summary = {"checked": 0, "filled": 0, "cancelled": 0,
               "stale": 0, "still_placed": 0, "errors": 0}

    cutoff = (_dt.now(_tz.utc) - timedelta(hours=RECONCILE_LOOKBACK_HOURS))\
              .isoformat()
    rows = conn.execute(
        """SELECT id, order_id, placed_at_utc, city, bin_label
           FROM live_predictor_orders
           WHERE status = 'placed'
             AND order_id IS NOT NULL AND order_id != ''
             AND placed_at_utc >= ?""",
        (cutoff,),
    ).fetchall()
    if not rows:
        return summary

    client = _get_live_client()
    if client is None:
        log.debug("reconcile: no CLOB client, skipping")
        return summary

    try:
        from execution import (get_order_status, is_order_fully_filled,
                                 is_order_cancelled)
    except Exception as e:
        log.warning(f"reconcile: execution import failed: {e}")
        return summary

    for r in rows:
        summary["checked"] += 1
        try:
            resp = get_order_status(r["order_id"], client)
            if resp is None:
                # Couldn't reach CLOB or got null — leave as placed.
                # Stale-bump only fires below for the AGE cutoff branch.
                summary["still_placed"] += 1
                continue
            if is_order_fully_filled(resp):
                conn.execute(
                    "UPDATE live_predictor_orders SET status = 'filled' WHERE id = ?",
                    (r["id"],))
                summary["filled"] += 1
            elif is_order_cancelled(resp):
                conn.execute(
                    "UPDATE live_predictor_orders SET status = 'cancelled' WHERE id = ?",
                    (r["id"],))
                summary["cancelled"] += 1
            else:
                summary["still_placed"] += 1
        except Exception as e:
            log.warning(f"reconcile: order {r['order_id'][:12]} failed: {e}")
            summary["errors"] += 1

    # Mark genuinely-old "still placed" orders as stale.  A 12-hour-old
    # 'placed' order with no fill is effectively dead — keeping it in the
    # 'placed' state would distort future reconciliation runs.
    stale_cutoff = (_dt.now(_tz.utc) - timedelta(hours=RECONCILE_LOOKBACK_HOURS))\
                    .isoformat()
    stale_cur = conn.execute(
        """UPDATE live_predictor_orders SET status = 'stale'
           WHERE status = 'placed' AND placed_at_utc < ?""",
        (stale_cutoff,),
    )
    summary["stale"] = stale_cur.rowcount or 0
    conn.commit()

    log.info(f"reconcile: checked={summary['checked']} "
             f"filled={summary['filled']} cancelled={summary['cancelled']} "
             f"stale={summary['stale']} still_placed={summary['still_placed']}")
    return summary


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run_intraday_scan(*, dry_run: bool = False) -> dict:
    """Single scan across all US cities.  Returns summary dict.  Safe to
    call standalone for testing.

    dry_run: if True, computes the scan but does NOT write to the DB.
             Useful for ad-hoc runs without polluting the paper log.
    """
    scan_start = datetime.now(timezone.utc)
    sizing_desc = (f"Kelly ¼ · bankroll ${PAPER_BANKROLL_USD:.0f}"
                    if USE_KELLY else f"flat ${FLAT_STAKE_USD:.2f}/trade")
    log.info(f"intraday_scan starting (mode={PREDICTOR_MODE}, "
             f"sizing={sizing_desc}, min_edge={MIN_EDGE}, "
             f"hour_window={MIN_TRIGGER_HOUR}-{MAX_TRIGGER_HOUR})")

    ensure_schema()
    _load_calibration()
    try:
        from scripts.intraday_predictor import _load_station_bias
        _load_station_bias()
    except Exception as e:
        log.warning(f"station bias load failed: {e}")

    us_cities = list(US_CITY_STATES.keys()) if US_CITY_STATES else [
        c for c, m in CITY_STATIONS.items() if m[0].startswith("K")
    ]
    us_cities = [c for c in us_cities if c in CITY_STATIONS]

    try:
        all_events = search_temp_high_events(min_liquidity=100)
    except Exception as e:
        log.error(f"event discovery failed: {e}")
        return {"error": str(e), "scanned_at_utc": scan_start.isoformat()}

    # Fetch live positions ONCE per scan — used for topup decisions and
    # accurate dedup.  Returns None for paper mode (no API call) AND
    # for live mode if the API call fails — both signal "fall back to
    # DB-derived sums".  Returns dict (possibly empty) only when API
    # call succeeded.
    live_positions = (
        fetch_polymarket_positions_by_token() if PREDICTOR_MODE == "live" else None
    )

    # City-name normalization: Polymarket's parser title-cases city tokens
    # ("nyc" → "Nyc"), but acronyms in our CITY_STATIONS stay uppercase
    # ("NYC").  Build a case-insensitive lookup from us_cities so we can
    # canonicalize whatever Polymarket gives us.
    us_cities_ci = {c.lower(): c for c in us_cities}

    events_by_city: dict[str, list[dict]] = {}
    for e in all_events:
        raw_city = e.get("city") or ""
        canonical = us_cities_ci.get(raw_city.lower())
        if canonical is not None:
            # Rewrite the event in place so downstream code (write_signal,
            # dashboard) sees the canonical name and rows aren't fragmented
            # between "Nyc" and "NYC".
            e["city"] = canonical
            events_by_city.setdefault(canonical, []).append(e)

    paper_buys = 0
    skips_by_reason: dict[str, int] = {}
    n_bins_evaluated = 0

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    # Use Row factory so dict-style column access (r["contract_id"]) works
    # in the new candidate-selection queries that ship today.
    conn.row_factory = sqlite3.Row
    n_events_skipped_not_today = 0
    n_events_skipped_already_bought = 0
    try:
        # Reconcile any orders that were 'placed' but not yet resolved.
        # Runs BEFORE we re-evaluate topup math so a freshly-filled order
        # is reflected in actual_deployed via the position API, not via
        # ghost-resting orders that distort cost basis.
        if PREDICTOR_MODE == "live" and not dry_run:
            try:
                reconcile_pending_orders(conn)
            except Exception as e:
                log.warning(f"order reconciliation failed (non-fatal): {e}")

        deployed = deployed_today_usd(conn, PREDICTOR_MODE)
        n_trades = trades_today(conn, PREDICTOR_MODE)

        # Track how many bins were bought for each event IN THIS SCAN.
        # Combined with event_buys_today_count() (DB-level, prior scans),
        # this gates the MAX_BINS_PER_EVENT cap end-to-end.
        events_bought_this_scan: dict[str, int] = {}

        for city in us_cities:
            events = events_by_city.get(city, [])
            if not events:
                continue
            s = CITY_STATIONS[city]
            icao, _net, tz_str, lat, lon = s
            tz = ZoneInfo(tz_str)
            now_local = datetime.now(tz)
            today_str_local = now_local.date().isoformat()

            # Filter to TODAY's events only (using the city's LOCAL date,
            # so Tokyo's "today" is its own calendar day, not UTC's).
            todays_events = [e for e in events
                              if e.get("date") == today_str_local]
            n_events_skipped_not_today += len(events) - len(todays_events)
            if not todays_events:
                continue

            # Weather cache check — refetch only if older than WEATHER_CACHE_SEC.
            # Polymarket data was already pulled fresh at scan start (above);
            # weather lags on a 5-min model cycle so caching it saves API
            # calls without hurting decision quality.
            import time as _time
            now_epoch = _time.time()
            cached = _WEATHER_CACHE.get(city)
            cache_age = (now_epoch - cached[0]) if cached else None
            if cached and cache_age < WEATHER_CACHE_SEC:
                w = cached[1]
                nws_obs        = w["nws_obs"]
                forecast       = w["forecast"]
                wind_octant    = w["wind_octant"]
                nbr_signal     = w["nbr_signal"]
                ensemble_stats = w.get("ensemble_stats")
                log.debug(f"weather cache HIT for {city} (age={cache_age:.0f}s)")
            else:
                # Pull hourly_max AND raw METAR cycles in a single HTTP call.
                # nws_obs feeds the predictor unchanged; raw_cycles get
                # persisted to raw_metar_log for future W1 audits — see
                # persist_raw_metar_cycles docstring.
                nws_obs, raw_cycles = fetch_nws_obs_with_raw(icao, tz_str)
                try:
                    n_new = persist_raw_metar_cycles(conn, raw_cycles)
                    if n_new:
                        log.debug(f"raw_metar_log: {icao} +{n_new} cycles")
                except Exception as e:
                    # Persistence failure must never block the scan.
                    log.warning(f"raw_metar persistence failed for {icao}: {e}")
                # PRIMARY: NWS forecast.  Fallback to Open-Meteo only if NWS down.
                forecast = fetch_nws_today_forecast(lat, lon, tz_str)
                if not forecast:
                    log.warning(f"NWS forecast empty for {city} — falling back to Open-Meteo")
                    forecast = fetch_openmeteo_today(lat, lon, tz_str)
                if not forecast:
                    continue
                # Defend forecast_high against the NWS hourly evening-scan
                # bug.  See recover_persisted_day_forecast docstring.  Done
                # BEFORE the cache write so every subsequent cache hit
                # within this day inherits the recovered value.
                try:
                    recovered_high, recovered_peak = recover_persisted_day_forecast(
                        conn, city, today_str_local,
                        candidate_high_c   = float(forecast["forecast_high"]),
                        candidate_peak_hour= forecast.get("forecast_peak_hour"),
                    )
                    if recovered_high > forecast["forecast_high"] + 0.01:
                        log.info(
                            f"forecast_high recovered for {city}: "
                            f"{forecast['forecast_high']:.2f}°C → {recovered_high:.2f}°C "
                            f"(NWS hourly evening-scan bug)"
                        )
                    forecast["forecast_high"] = recovered_high
                    if recovered_peak is not None:
                        forecast["forecast_peak_hour"] = recovered_peak
                except Exception as e:
                    log.warning(f"forecast_high recovery failed for {city}: {e}")
                afternoon_winds = [r["wind_dir_deg"] for r in nws_obs
                                    if r["hour_local"] in range(11, 19)
                                    and r["wind_dir_deg"] is not None]
                wind_mean = vector_mean_dir(afternoon_winds)
                wind_octant = deg_to_cardinal(wind_mean) if wind_mean is not None else None
                nbr_signal = compute_neighbor_signal(city, today_str_local, wind_octant)
                # Ensemble forecast from neighbor ASOS stations.  Multi-
                # station consensus catches local NWS gridpoint biases +
                # gives us empirical uncertainty for sigma inflation.
                try:
                    from scripts.ensemble_forecast import compute_ensemble_stats
                    ensemble_stats = compute_ensemble_stats(city, forecast, tz_str)
                    if ensemble_stats.get("settlement_is_outlier"):
                        log.info(
                            f"  {city}: settlement forecast is OUTLIER "
                            f"(divergence={ensemble_stats['divergence_c']:+.2f}°C vs "
                            f"{ensemble_stats['n_stations_used']}-station ensemble "
                            f"median={ensemble_stats['ensemble_median']:.2f}°C, "
                            f"std={ensemble_stats['ensemble_std']:.2f}°C)"
                        )
                except Exception as e:
                    log.warning(f"ensemble fetch failed for {city}: {e}")
                    ensemble_stats = None
                _WEATHER_CACHE[city] = (now_epoch, {
                    "nws_obs":        nws_obs,
                    "forecast":       forecast,
                    "wind_octant":    wind_octant,
                    "nbr_signal":     nbr_signal,
                    "ensemble_stats": ensemble_stats,
                })
                log.debug(f"weather cache REFRESH for {city}")

            for ev in todays_events:
                event_id = ev.get("event_id") or ""

                # IMPORTANT: we always re-evaluate every event (even if it's
                # already at the buy cap) so the dashboard sees fresh
                # market_prob values every scan.

                # === THREE-SIGNAL POSITION MODEL ===
                # Position tracking is decomposed into three independent
                # signals so no single check has to do triple duty:
                #
                #   1. HELD set       — contracts whose token is currently
                #                       in the wallet (LIVE_BUY rows whose
                #                       token shows up in live_positions).
                #                       This answers "did a fill happen?" —
                #                       used for dedup + cap-counting.
                #                       Resolved-to-zero positions ARE in
                #                       this set (you own losing shares); we
                #                       gate them out via MARKET_OPEN below.
                #
                #   2. DEPLOYED $/contract — from live_positions: size × avg.
                #                       Cost basis.  Used for topup sizing.
                #
                #   3. MARKET_OPEN per bin — from Gamma's `closed` flag
                #                       (propagated through predict_bins).
                #                       True = market is still tradeable.
                #                       Used to (a) block fresh buys on
                #                       resolved markets, (b) exclude
                #                       resolved positions from the cap
                #                       (closed market = settled, no
                #                       longer occupying a slot), and
                #                       (c) drive dashboard pills.
                #
                # PAPER mode: HELD = DB BUY rows (paper "fills" are
                # simulated and persist).  No API.  MARKET_OPEN still
                # applies — paper buys against a closed market are
                # nonsense.
                _action_str = _action_for_mode(PREDICTOR_MODE)
                db_buys = {r["contract_id"]: r["yes_token_id"]
                            for r in conn.execute(
                    """SELECT DISTINCT contract_id, yes_token_id
                       FROM paper_predictor_signals
                       WHERE event_id = ? AND action = ?""",
                    (event_id, _action_str)).fetchall()}

                # SIGNAL 1: HELD set — contracts whose token actually
                # appears in the wallet (live) or whose buy row exists
                # (paper, since paper fills are simulated).
                if PREDICTOR_MODE == "live" and live_positions is not None:
                    held_contracts = {
                        c for c, tok in db_buys.items()
                        if tok and str(tok) in live_positions
                    }
                else:
                    # Paper mode (or live with API unreachable) — DB-derived.
                    # API-unreachable fallback stays conservative to avoid
                    # spam-rebuys when we can't verify.
                    held_contracts = set(db_buys.keys())

                # Compute cold-start suspect BEFORE predict_bins so we
                # can pass it as a kwarg.  The HRRR ceiling dispatch
                # inside predict_bins skips the rapid-model fetch on
                # cold-start days (HRRR's "remaining hours" view can't
                # recover a peak that already happened before the
                # bot's first scan today).
                cold_start_suspect = False
                try:
                    cold_start_suspect = is_cold_start_day(
                        conn, city, today_str_local, tz_str,
                        scan_start.isoformat())
                except Exception as e:
                    log.warning(f"cold-start detection failed for {city}: {e}")

                # Run the bin predictor — needed before MARKET_OPEN/cap
                # checks because we need each bin's `closed` flag.
                # Pass lat/lon/tz_str so the HRRR ceiling dispatch can
                # fetch the rapid-update CAM run.  When flag is off
                # (PREDICTOR_USE_HRRR_CEILING=0), these are unused.
                pred = predict_bins(ev, nws_obs, forecast, nbr_signal,
                                     now_local.hour, city=city,
                                     ensemble_stats=ensemble_stats,
                                     cold_start_suspect=cold_start_suspect,
                                     lat=lat, lon=lon, tz_str=tz_str)
                if not pred.get("bins"):
                    continue

                # Compose the data-quality flag for every signal row we
                # write this event.  Starts with the CDF path that produced
                # our_prob (e.g. "gaussian"), appends "cold_start_suspect"
                # when set, and "hrrr_ceiling_applied" / "icon_d2_ceiling_applied"
                # when the rapid-model ceiling fired (Spec §5).
                _dq_components: list[str] = []
                if pred.get("cdf_used"):
                    _dq_components.append(pred["cdf_used"])
                if cold_start_suspect:
                    _dq_components.append("cold_start_suspect")
                if pred.get("hrrr_used"):
                    # Distinguish HRRR vs ICON-D2 in the flag so audits
                    # can segment by source.
                    try:
                        from station_meta import get_same_day_model  # type: ignore
                        model = get_same_day_model(city)
                    except Exception:
                        model = None
                    if model == "icon_d2":
                        _dq_components.append("icon_d2_ceiling_applied")
                    else:
                        _dq_components.append("hrrr_ceiling_applied")
                if pred.get("plausibility_ceiling_fired"):
                    _dq_components.append("plausibility_ceiling_applied")
                event_data_quality_flag = (
                    ",".join(_dq_components) if _dq_components else None)
                event_size_factor = compute_data_quality_size_factor(
                    event_data_quality_flag)

                # SIGNAL 3: MARKET_OPEN per bin — Gamma's `closed` flag.
                # Build a contract→open map so we can answer both
                # "is this bin tradeable as a fresh buy?" and
                # "does this held position still count toward the cap?"
                market_open: dict[str, bool] = {
                    b["contract_id"]: not bool(b.get("closed"))
                    for b in pred["bins"]
                }

                # FIX 2 (2026-06-12): committed = held + pending.
                # A contract with a placed-but-not-yet-filled order
                # should count toward the cap.  Without this, the bot
                # would buy a DIFFERENT bin on the same event while
                # the first bin's order was still resting on the book
                # (NYC race: bought 92-93°F at 17:37, then 94-95°F at
                # 18:32 because API hadn't seen 92-93°F fill yet, ended
                # up holding both bins on a MAX_BINS_PER_EVENT=1 event).
                # Pending orders are paper-mode no-op (empty set).
                #
                # FIX 3 (2026-06-13): scope _pending_today to THIS
                # event's bins.  pending_contracts_today() returns the
                # cross-event union; without this intersection, a
                # single pending order on one city (e.g. Seattle)
                # would propagate into every other event's
                # committed_contracts and fire event_at_cap_today
                # everywhere — observed live with one Seattle order
                # capping Atlanta/Austin/Chicago/Dallas/Denver.  The
                # cause was that market_open.get(c, True) defaulted
                # to True for contracts not in the current event's
                # bin map, so cross-event pending tokens slipped
                # through the open-market filter.  Intersecting at
                # the source keeps the NYC same-event guarantee while
                # eliminating the cross-event pollution.
                if PREDICTOR_MODE == "live":
                    _this_event_contracts = set(market_open.keys())
                    _pending_today = (pending_contracts_today(conn)
                                       & _this_event_contracts)
                else:
                    _pending_today = set()
                committed_contracts = held_contracts | _pending_today

                # Cap is occupied by COMMITTED contracts (held OR with a
                # pending order) whose underlying market is still open.
                # Resolved-to-zero positions are held but the market is
                # closed → no slot occupied.  (Topups against closed
                # markets are also rejected below.)
                already_bought_contracts = {
                    c for c in committed_contracts
                    if market_open.get(c, True)
                }
                buys_already = len(already_bought_contracts)
                event_at_cap = buys_already >= MAX_BINS_PER_EVENT
                if event_at_cap:
                    n_events_skipped_already_bought += 1

                # Rank bins by OUR PROBABILITY descending.  Two kinds of
                # candidates:
                #   1. FRESH-BUY candidates: top P-rank bins NOT already
                #      committed AND market still open.  Cap-bound.
                #   2. TOPUP candidates: bins ALREADY HELD (filled) and
                #      market still open.  Not cap-bound (no new bin
                #      added).  NOTE: topup target is only bins in
                #      `held_contracts` — bins with merely pending
                #      orders don't get more topup attempts piled on
                #      until their first order resolves.
                bins_by_p = sorted(pred["bins"], key=lambda x: -x["our_prob"])

                fresh_bins = [b for b in bins_by_p
                               if b["contract_id"] not in committed_contracts
                               and market_open.get(b["contract_id"], True)]
                topup_bins = [b for b in bins_by_p
                               if b["contract_id"] in already_bought_contracts
                               and b["contract_id"] in held_contracts]

                if event_at_cap:
                    fresh_candidates = []     # no new bins allowed at cap
                    non_candidate_reason = "event_at_cap_today"
                else:
                    slots_remaining = MAX_BINS_PER_EVENT - buys_already
                    fresh_candidates = fresh_bins[:slots_remaining]
                    non_candidate_reason = "not_top_p_in_event"

                top_candidates = fresh_candidates + topup_bins
                # Bins not even considered for a fresh buy AND not already
                # bought → these are the SKIP rows for visibility.
                non_candidates = [b for b in bins_by_p if b not in top_candidates]
                # Track which candidates are topups so we don't double-count
                # them against the per-event cap.
                _topup_contract_ids = {b["contract_id"] for b in topup_bins}

                # Track whether the event was aborted (top-P bin failed
                # gates).  Any remaining top-candidates that haven't been
                # processed yet will inherit this reason.
                event_aborted = False

                # Process top candidates in P-rank order
                for rank_idx, b in enumerate(top_candidates):
                    n_bins_evaluated += 1
                    edge       = b["edge"]
                    market_p   = b["market_prob"]
                    our_p      = b["our_prob"]
                    liquidity  = b["liquidity_usd"]
                    contract_id  = b["contract_id"]
                    yes_token_id = b["yes_token_id"]
                    is_topup     = contract_id in _topup_contract_ids

                    # Compute target stake + remaining-to-target for topup support.
                    target_stake = compute_stake(edge, market_p, PAPER_BANKROLL_USD)
                    actual_deployed = get_actual_deployed_usd(
                        conn, event_id, contract_id, yes_token_id,
                        PREDICTOR_MODE, live_positions,
                    )

                    # FIX 1 (2026-06-12): committed = actual + pending.
                    # Orders placed today but not yet filled don't show
                    # up in actual_deployed (which reads cost basis from
                    # Polymarket's positions API).  Without counting
                    # them as committed, the bot keeps placing more
                    # topup orders thinking it's under target — and when
                    # they all eventually fill, total deployed blows
                    # past the target.  This was the Houston 2026-06-12
                    # bug: $74.99 actual deployed despite $10 target,
                    # produced by repeated pending-order stacking across
                    # multiple scans before any order resolved.
                    if PREDICTOR_MODE == "live":
                        pending_stake = pending_stake_for_contract_today(
                            conn, contract_id)
                    else:
                        pending_stake = 0.0
                    committed_deployed = actual_deployed + pending_stake
                    remaining_to_target = max(
                        0.0, target_stake - committed_deployed)

                    # Per-contract daily $ ceiling.  Anchored to COMMITTED
                    # cost basis on Polymarket (actual_deployed + any
                    # pending order stakes), not to the sum of signal-row
                    # intents.  Earlier versions summed
                    # `recommended_stake_usd` across all today's LIVE_BUY
                    # rows, which counted unfilled-and-cancelled orders
                    # against the cap — a volatile market producing
                    # unfilled topup attempts would block legitimate
                    # target completion.
                    #
                    # Using committed (actual + pending) here means we
                    # cap based on capital we've ACTUALLY put on the line
                    # — filled positions + resting orders.  Sold shares
                    # reduce actual_deployed → free cap headroom (also
                    # the right behavior).  Cancelled orders move out of
                    # 'placed' → free cap headroom (via reconciliation
                    # at next scan).
                    contract_ceiling_remaining = None
                    if MAX_PER_CONTRACT_USD > 0:
                        contract_ceiling_remaining = max(
                            0.0, MAX_PER_CONTRACT_USD - committed_deployed,
                        )
                        remaining_to_target = min(
                            remaining_to_target, contract_ceiling_remaining,
                        )

                    # "At target" if within TOPUP_TOLERANCE_PCT OR remaining
                    # is below TOPUP_MIN_USD (not worth a tiny order).
                    at_target_threshold = max(
                        TOPUP_MIN_USD,
                        target_stake * TOPUP_TOLERANCE_PCT,
                    )
                    at_target = remaining_to_target < at_target_threshold
                    # If the ceiling is what's cutting us off (not the
                    # cost-basis-vs-target math), give the SKIP row a more
                    # informative reason.
                    contract_ceiling_hit = (
                        contract_ceiling_remaining is not None
                        and contract_ceiling_remaining < at_target_threshold
                        and remaining_to_target == contract_ceiling_remaining
                    )

                    if event_aborted:
                        ok = False
                        reason = "event_aborted (higher-P bin failed gates)"
                    elif at_target:
                        # Include pending stake in the reason so the
                        # dashboard distinguishes "no headroom because
                        # filled" from "no headroom because orders are
                        # still resting on the book."
                        _pending_str = (f" + ${pending_stake:.2f} pending"
                                          if pending_stake > 0.01 else "")
                        if contract_ceiling_hit:
                            ok = False
                            reason = (f"per_contract_daily_cap "
                                      f"(committed=${committed_deployed:.2f}"
                                      f"{_pending_str} "
                                      f"/ cap=${MAX_PER_CONTRACT_USD:.2f})")
                        else:
                            ok = False
                            reason = (f"at_target "
                                      f"(committed=${committed_deployed:.2f}"
                                      f"{_pending_str} "
                                      f"/ target=${target_stake:.2f})")
                    else:
                        # Note: already_acted is now False because we handle
                        # the "fully filled" case via at_target above.  Topups
                        # to a partially-filled bin should pass the dedup gate.
                        ok, reason = evaluate_gates(
                            current_hour    = now_local.hour,
                            edge            = edge,
                            market_p        = market_p,
                            liquidity       = liquidity,
                            deployed_today  = deployed,
                            trades_today    = n_trades,
                            already_acted   = False,
                            our_p           = our_p,
                        )
                        # Per-scan cap — only counts FRESH buys (topups don't
                        # add new bins to the event)
                        if (ok and not is_topup
                            and events_bought_this_scan.get(event_id, 0)
                                 >= MAX_BINS_PER_EVENT - buys_already):
                            ok = False
                            reason = "event_cap_reached_this_scan"
                        # Strict abort: if rank-0 fresh-buy candidate fails
                        # gates, abort the event.  Topup failures don't abort
                        # (the position was decided in an earlier scan; failing
                        # now just means we stop topping up).
                        if not ok and rank_idx == 0 and not is_topup:
                            event_aborted = True

                    if ok:
                        # Stake for THIS order = remaining_to_target.
                        # Initial buy: == target.  Topup: == what's left to fill.
                        stake = remaining_to_target
                        # Data-quality sizing scalar — applied AFTER Kelly
                        # but BEFORE the MIN_STAKE_USD floor.  See
                        # docs/data_quality_contract.md.  Today, the
                        # relative tiers (gaussian / empirical) are all at
                        # 1.00; the only haircut that actually fires is
                        # cold_start_suspect at 0.30 (and BLOCK at 0.00).
                        if event_size_factor < 1.0:
                            stake_pre_dq = stake
                            stake = stake * event_size_factor
                            if event_size_factor <= 0.0:
                                ok = False
                                reason = (f"data_quality_blocked "
                                          f"(flag={event_data_quality_flag!r}, "
                                          f"size_factor=0.00)")
                            else:
                                log.info(
                                    f"  data-quality haircut: {city} {b['label']} "
                                    f"${stake_pre_dq:.2f} → ${stake:.2f} "
                                    f"(flag={event_data_quality_flag!r}, "
                                    f"×{event_size_factor:.2f})"
                                )
                        if ok and stake < MIN_STAKE_USD:
                            ok = False
                            reason = f"stake_too_small (${stake:.2f} < ${MIN_STAKE_USD:.2f})"

                    if ok:
                        action = "LIVE_BUY" if PREDICTOR_MODE == "live" else "PAPER_BUY"
                        paper_buys += 1
                        n_trades   += 1
                        deployed  += stake
                        limit_px   = marketable_limit(market_p)
                        gate_blocked_by = None
                        # Only count FRESH buys toward the per-scan cap
                        if not is_topup:
                            events_bought_this_scan[event_id] = (
                                events_bought_this_scan.get(event_id, 0) + 1
                            )
                        log.info(
                            f"  {'TOPUP' if is_topup else 'BUY'}: {city} {b['label']} "
                            f"${stake:.2f} (target=${target_stake:.2f}, "
                            f"already=${actual_deployed:.2f})"
                        )
                    else:
                        action = "AVOID" if edge <= -0.20 else "SKIP"
                        stake = 0.0
                        limit_px = None
                        gate_blocked_by = reason
                        skips_by_reason[reason] = skips_by_reason.get(reason, 0) + 1

                    sig_row = {
                        "scanned_at_utc":         scan_start.isoformat(),
                        "mode":                   PREDICTOR_MODE,
                        "city":                   city,
                        "settlement_station":     icao,
                        "event_date":             ev.get("date"),
                        "event_id":               ev.get("event_id") or "",
                        "contract_id":            contract_id,
                        "yes_token_id":           yes_token_id,
                        "bin_label":              b["label"],
                        "bin_range_low":          b["range_low"],
                        "bin_range_high":         b["range_high"],
                        "unit":                   b["unit"],
                        "our_prob":               our_p,
                        "market_prob":            market_p,
                        "edge":                   edge,
                        "liquidity_usd":          liquidity,
                        "action":                 action,
                        "gate_blocked_by":        gate_blocked_by,
                        "recommended_stake_usd":  stake,
                        "recommended_limit_price": limit_px,
                        "current_hour_local":     now_local.hour,
                        "observed_max_c":         pred["observed_max_c"],
                        "observed_peak_hour":     pred["observed_peak_hour"],
                        "forecast_high_c":        pred["forecast_high"],
                        "forecast_peak_hour":     pred["forecast_peak_hour"],
                        "mu_c":                   pred["mu"],
                        "sigma_c":                pred["sigma"],
                        "wind_octant":            wind_octant,
                        "upwind_signal_strength": nbr_signal.get("signal_strength"),
                        "market_closed":          1 if b.get("closed") else 0,
                        "data_quality_flag":      event_data_quality_flag,
                        "cooling_confidence":     pred.get("cooling_confidence"),
                    }
                    if not dry_run:
                        signal_id = write_signal(conn, sig_row)
                        # LIVE execution path — only when explicitly enabled
                        # AND the gates passed.  Dry-run is honored (still no
                        # orders placed when --dry-run).
                        if (PREDICTOR_MODE == "live"
                            and action == "LIVE_BUY"
                            and not dry_run):
                            execute_live(conn, signal_id, city, ev, b,
                                          stake, limit_px)

                # Write SKIP rows for non-candidate bins.  Two cases:
                #   1. event at cap → ALL bins land here, reason = event_at_cap_today
                #   2. event has slots → only lower-P bins land here,
                #      reason = not_top_p_in_event
                # Either way, this guarantees we write fresh market_prob
                # rows for every bin every scan, keeping the dashboard
                # data live even for events we've already bought.
                for b in non_candidates:
                    n_bins_evaluated += 1
                    # A bin can land in non_candidates because (a) the
                    # event was at cap, (b) it isn't top-P this scan, OR
                    # (c) its underlying market has resolved.  Make case
                    # (c) explicit so the dashboard can show a clear
                    # "market_closed" SKIP reason instead of conflating it
                    # with "not the top bin."
                    reason = ("market_closed" if b.get("closed")
                                else non_candidate_reason)
                    skips_by_reason[reason] = skips_by_reason.get(reason, 0) + 1
                    sig_row = {
                        "scanned_at_utc":         scan_start.isoformat(),
                        "mode":                   PREDICTOR_MODE,
                        "city":                   city,
                        "settlement_station":     icao,
                        "event_date":             ev.get("date"),
                        "event_id":               event_id,
                        "contract_id":            b["contract_id"],
                        "yes_token_id":           b["yes_token_id"],
                        "bin_label":              b["label"],
                        "bin_range_low":          b["range_low"],
                        "bin_range_high":         b["range_high"],
                        "unit":                   b["unit"],
                        "our_prob":               b["our_prob"],
                        "market_prob":            b["market_prob"],
                        "edge":                   b["edge"],
                        "liquidity_usd":          b["liquidity_usd"],
                        "action":                 "SKIP",
                        "gate_blocked_by":        reason,
                        "recommended_stake_usd":  0.0,
                        "recommended_limit_price": None,
                        "current_hour_local":     now_local.hour,
                        "observed_max_c":         pred["observed_max_c"],
                        "observed_peak_hour":     pred["observed_peak_hour"],
                        "forecast_high_c":        pred["forecast_high"],
                        "forecast_peak_hour":     pred["forecast_peak_hour"],
                        "mu_c":                   pred["mu"],
                        "sigma_c":                pred["sigma"],
                        "wind_octant":            wind_octant,
                        "upwind_signal_strength": nbr_signal.get("signal_strength"),
                        "market_closed":          1 if b.get("closed") else 0,
                        "data_quality_flag":      event_data_quality_flag,
                        "cooling_confidence":     pred.get("cooling_confidence"),
                    }
                    if not dry_run:
                        write_signal(conn, sig_row)
    finally:
        # Invariant guards.  PURE OBSERVATIONAL — see the "observational
        # forever" design rule at the top of scripts/invariant_guards.py.
        # Failures here are logged but never block the scan, and the
        # guards' outputs never re-enter the prediction or trading path.
        if not dry_run:
            try:
                from scripts.invariant_guards import run_invariant_checks
                run_invariant_checks(conn, scan_start.isoformat())
            except Exception as e:
                log.warning(f"invariant_guards failed (non-fatal): {e}")
        conn.close()

    action_label = "live_buys" if PREDICTOR_MODE == "live" else "paper_buys"
    summary = {
        "scanned_at_utc":    scan_start.isoformat(),
        "mode":              PREDICTOR_MODE,
        "n_cities":          len(us_cities),
        "n_events":          sum(len(v) for v in events_by_city.values()),
        "events_skipped_not_today":         n_events_skipped_not_today,
        "events_skipped_already_bought":    n_events_skipped_already_bought,
        "n_bins_evaluated":  n_bins_evaluated,
        f"n_{action_label}":                paper_buys,
        f"deployed_today_{PREDICTOR_MODE}": round(deployed, 2),
        f"trades_today_{PREDICTOR_MODE}":   n_trades,
        "skips_by_reason":   skips_by_reason,
        "elapsed_sec":       (datetime.now(timezone.utc) - scan_start).total_seconds(),
    }
    buy_label = "LIVE_BUY" if PREDICTOR_MODE == "live" else "PAPER_BUY"
    log.info(f"intraday_scan done: {paper_buys} {buy_label} · "
             f"{n_bins_evaluated} bins · "
             f"{n_events_skipped_not_today} non-today events skipped · "
             f"{n_events_skipped_already_bought} events already bought · "
             f"deployed today ${deployed:.0f} ({PREDICTOR_MODE}) · "
             f"{summary['elapsed_sec']:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# APScheduler integration
# ---------------------------------------------------------------------------

def register_predictor_jobs(scheduler) -> None:
    """Register the intraday scan as an APScheduler cron job.

    Wire in main.py with:
        from scheduled_predictor import register_predictor_jobs
        register_predictor_jobs(scheduler)

    Runs every PREDICTOR_SCAN_MIN minutes (default 15).  The gates handle
    per-city time-of-day filtering, so the job runs always; cities outside
    their trading window contribute zero PAPER_BUYs.
    """
    from apscheduler.triggers.cron import CronTrigger

    def _job() -> None:
        try:
            run_intraday_scan()
        except Exception as e:
            log.exception(f"intraday_scan failed: {e}")

    scheduler.add_job(
        _job,
        trigger=CronTrigger(minute=f"*/{SCAN_INTERVAL_MIN}", timezone="UTC"),
        id="intraday_predictor_scan",
        name=f"Intraday predictor scan (every {SCAN_INTERVAL_MIN} min, "
              f"mode={PREDICTOR_MODE})",
        misfire_grace_time=120,
        coalesce=True,
        max_instances=1,
    )
    log.info(f"registered intraday_predictor_scan job: every "
             f"{SCAN_INTERVAL_MIN}m, mode={PREDICTOR_MODE}, "
             f"min_edge={MIN_EDGE}, bankroll=${PAPER_BANKROLL_USD:.0f}")


# ---------------------------------------------------------------------------
# CLI: one-shot scan for ad-hoc testing
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse, json
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)-7s | %(message)s")
    p = argparse.ArgumentParser(description="One-shot intraday predictor scan")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute the scan but do NOT write to the DB")
    p.add_argument("--json", action="store_true",
                   help="Emit summary as JSON instead of human-readable")
    args = p.parse_args()
    result = run_intraday_scan(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print()
        print("=" * 72)
        for k, v in result.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())