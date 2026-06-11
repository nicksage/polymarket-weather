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
MIN_LIQUIDITY_USD    = float(os.getenv("PREDICTOR_MIN_LIQUIDITY", "300"))
MIN_TRIGGER_HOUR     = int  (os.getenv("PREDICTOR_MIN_HOUR",     "13"))
MAX_TRIGGER_HOUR     = int  (os.getenv("PREDICTOR_MAX_HOUR",     "22"))
MAX_MARKET_PRICE     = float(os.getenv("PREDICTOR_MAX_MKT_PRICE", "0.95"))
MAX_DAILY_EXPOSURE   = float(os.getenv("PREDICTOR_MAX_DAILY_EXP", "200"))
MAX_TRADES_PER_DAY   = int  (os.getenv("PREDICTOR_MAX_TRADES",   "25"))
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
    upwind_signal_strength REAL
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
"""


def ensure_schema() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA_SQL)


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
                    trades_today: int, already_acted: bool) -> tuple[bool, str]:
    """Returns (pass, reason).  reason is empty when pass=True."""
    if current_hour < MIN_TRIGGER_HOUR:
        return False, f"too_early (hour={current_hour} < {MIN_TRIGGER_HOUR})"
    if current_hour > MAX_TRIGGER_HOUR:
        return False, f"too_late (hour={current_hour} > {MAX_TRIGGER_HOUR})"
    # Market sanity floor — bin must have at least MIN_MARKET_PROB market
    # confidence.  Catches catastrophic model errors: if market thinks a
    # bin has <15% chance, our model claiming 100% is almost certainly
    # a bug, not edge.  Runs BEFORE edge calc so we don't burn cycles on
    # garbage signals.
    if market_p < MIN_MARKET_PROB:
        return False, (f"market_too_skeptical (mkt={market_p:.3f} < "
                       f"{MIN_MARKET_PROB:.2f})")
    # Tiered edge gate: stricter when market_p is "expensive" (>= 0.75),
    # looser when cheap (<0.75).  The original MIN_EDGE=0.10 was designed
    # to filter out high-priced bins where we'd lose the full stake on the
    # wrong side; for cheaper bins the same edge is more attractive on
    # expected-value terms.
    required_edge = MIN_EDGE if market_p >= HIGH_MKT_THRESHOLD else MIN_EDGE_LOW_MKT
    if edge < required_edge:
        return False, (f"low_edge ({edge:+.3f} < {required_edge:.2f}, "
                       f"mkt_p={market_p:.2f})")
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

    Used in the scan loop to determine actual deployed capital per
    contract — driving topup decisions and dedup checks (live mode).
    Return value semantics (important for callers):
      * dict (possibly EMPTY {}) — API call succeeded.  Empty = we have
        zero open positions.  Callers should TRUST this (no positions =
        no positions, slots are open for re-buy).
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
        n_dust = 0
        for p in data:
            token = p.get("asset") or p.get("token_id") or p.get("tokenId")
            if not token:
                continue
            size = float(p.get("size") or 0)
            avg  = float(p.get("avgPrice") or p.get("avg_price") or 0)
            cur  = float(p.get("curPrice") or p.get("cur_price") or 0)
            if size <= 0 or avg <= 0:
                continue
            # POSITION-VALIDITY FILTER.  Polymarket's data API returns
            # positions for any token the wallet holds — including:
            #   - dust positions with sub-cent current value
            #   - resolved-to-zero positions (market closed, cur≈0)
            # Both should be treated as "we no longer hold this for
            # trading purposes" so they don't block fresh buys via the
            # at_target gate.
            #
            # A position is VALID (tradeable) only when all true:
            #   1. cur > 0.005   — market is still tradeable (>0.5¢)
            #   2. size * cur >= 0.50  — current value is meaningful
            current_value = size * cur
            if cur < 0.005 or current_value < 0.50:
                n_dust += 1
                continue
            out[str(token)] = {
                "size":          size,
                "avg_price":     avg,
                "deployed_usdc": size * avg,
                "cur_price":     cur,
                "cash_pnl":      float(p.get("cashPnl") or p.get("cash_pnl") or 0),
            }
        log.info(f"Polymarket positions: {len(out)} live positions fetched"
                  f"{f' (+ {n_dust} dust/resolved filtered)' if n_dust else ''}")
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
                nws_obs = fetch_nws_today_obs(icao, tz_str)
                # PRIMARY: NWS forecast.  Fallback to Open-Meteo only if NWS down.
                forecast = fetch_nws_today_forecast(lat, lon, tz_str)
                if not forecast:
                    log.warning(f"NWS forecast empty for {city} — falling back to Open-Meteo")
                    forecast = fetch_openmeteo_today(lat, lon, tz_str)
                if not forecast:
                    continue
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
                # market_prob values every scan.  Previously we short-
                # circuited cap-reached events to save the NWS call, but
                # that froze the dashboard's market data and P&L.  The
                # cap is now enforced as a per-bin gate below (every bin
                # gets a SKIP row with reason "event_at_cap"), so prices
                # stay live.
                buys_already = (event_buys_today_count(conn, event_id, PREDICTOR_MODE)
                                  if event_id else 0)
                event_at_cap = buys_already >= MAX_BINS_PER_EVENT
                if event_at_cap:
                    n_events_skipped_already_bought += 1   # for the summary

                pred = predict_bins(ev, nws_obs, forecast, nbr_signal,
                                     now_local.hour, city=city,
                                     ensemble_stats=ensemble_stats)
                if not pred.get("bins"):
                    continue

                # Identify contracts that count as "currently bought" for
                # this event.  Used to (a) allow topups to existing
                # positions without re-tripping the cap, and (b) prevent
                # double-buying the same bin.
                #
                # LIVE mode: source of truth is the Polymarket API.  A DB
                # LIVE_BUY row that no longer has an API position (closed
                # manually, order failed, never filled) does NOT count as
                # "bought" — the slot is free for re-buy.
                #
                # PAPER mode: DB is the truth (paper "fills" are simulated
                # and persist as long as the row exists).
                _action_str = _action_for_mode(PREDICTOR_MODE)
                db_buys = {r["contract_id"]: r["yes_token_id"]
                            for r in conn.execute(
                    """SELECT DISTINCT contract_id, yes_token_id
                       FROM paper_predictor_signals
                       WHERE event_id = ? AND action = ?""",
                    (event_id, _action_str)).fetchall()}
                if PREDICTOR_MODE == "live" and live_positions is not None:
                    # Only count contracts whose token is currently in the
                    # Polymarket API positions list.
                    already_bought_contracts = {
                        c for c, tok in db_buys.items()
                        if tok and str(tok) in live_positions
                    }
                else:
                    # Paper mode (or live with API unreachable) — all DB
                    # buys count.  API-unreachable fallback is conservative
                    # so we don't spam re-buys when we can't verify.
                    already_bought_contracts = set(db_buys.keys())

                # Rank bins by OUR PROBABILITY descending.  Two kinds of
                # candidates:
                #   1. FRESH-BUY candidates: top P-rank bins NOT already
                #      bought.  Limited to slots_remaining (cap-bound).
                #   2. TOPUP candidates: bins ALREADY bought that may still
                #      have headroom below the target stake.  Not cap-bound
                #      because topping up an existing bin doesn't add a new
                #      bin to the event.
                bins_by_p = sorted(pred["bins"], key=lambda x: -x["our_prob"])

                fresh_bins = [b for b in bins_by_p
                               if b["contract_id"] not in already_bought_contracts]
                topup_bins = [b for b in bins_by_p
                               if b["contract_id"] in already_bought_contracts]

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
                    remaining_to_target = max(0.0, target_stake - actual_deployed)
                    # "At target" if within TOPUP_TOLERANCE_PCT OR remaining
                    # is below TOPUP_MIN_USD (not worth a tiny order).
                    at_target_threshold = max(
                        TOPUP_MIN_USD,
                        target_stake * TOPUP_TOLERANCE_PCT,
                    )
                    at_target = remaining_to_target < at_target_threshold

                    if event_aborted:
                        ok = False
                        reason = "event_aborted (higher-P bin failed gates)"
                    elif at_target:
                        ok = False
                        reason = (f"at_target (deployed=${actual_deployed:.2f} "
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
                        if stake < MIN_STAKE_USD:
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
                    reason = non_candidate_reason
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
                    }
                    if not dry_run:
                        write_signal(conn, sig_row)
    finally:
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