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
    fetch_openmeteo_today,
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
    if edge < MIN_EDGE:
        return False, f"low_edge ({edge:+.3f} < {MIN_EDGE:.2f})"
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
    """True if THIS exact (event,bin) was already bought today IN THIS MODE.
    Paper and live are independent — a paper-bought bin can still be
    live-bought (they're different 'pools')."""
    today = _today_utc_date_str()
    row = conn.execute(
        """
        SELECT 1 FROM paper_predictor_signals
        WHERE event_id = ? AND contract_id = ? AND action = ?
          AND substr(scanned_at_utc, 1, 10) = ?
        LIMIT 1
        """,
        (event_id, contract_id, _action_for_mode(mode), today),
    ).fetchone()
    return row is not None


def event_has_buy_today(conn, event_id: str, mode: str) -> bool:
    """True if ANY bin of this event was bought today in this mode.
    Convenience wrapper around event_buys_today_count()."""
    return event_buys_today_count(conn, event_id, mode) > 0


def event_buys_today_count(conn, event_id: str, mode: str) -> int:
    """Number of bins bought for this event today in this mode.  Used by
    the MAX_BINS_PER_EVENT cap — when the count equals the cap, the
    event is full and no more bins can be bought today."""
    today = _today_utc_date_str()
    row = conn.execute(
        """
        SELECT COUNT(*) FROM paper_predictor_signals
        WHERE event_id = ? AND action = ?
          AND substr(scanned_at_utc, 1, 10) = ?
        """,
        (event_id, _action_for_mode(mode), today),
    ).fetchone()
    return int(row[0]) if row else 0


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

    us_cities = list(US_CITY_STATES.keys()) if US_CITY_STATES else [
        c for c, m in CITY_STATIONS.items() if m[0].startswith("K")
    ]
    us_cities = [c for c in us_cities if c in CITY_STATIONS]

    try:
        all_events = search_temp_high_events(min_liquidity=100)
    except Exception as e:
        log.error(f"event discovery failed: {e}")
        return {"error": str(e), "scanned_at_utc": scan_start.isoformat()}

    events_by_city: dict[str, list[dict]] = {}
    for e in all_events:
        c = e.get("city")
        if c in us_cities:
            events_by_city.setdefault(c, []).append(e)

    paper_buys = 0
    skips_by_reason: dict[str, int] = {}
    n_bins_evaluated = 0

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
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

            nws_obs   = fetch_nws_today_obs(icao, tz_str)
            forecast  = fetch_openmeteo_today(lat, lon, tz_str)
            if not forecast:
                continue

            afternoon_winds = [r["wind_dir_deg"] for r in nws_obs
                                if r["hour_local"] in range(11, 19)
                                and r["wind_dir_deg"] is not None]
            wind_mean = vector_mean_dir(afternoon_winds)
            wind_octant = deg_to_cardinal(wind_mean) if wind_mean is not None else None
            nbr_signal = compute_neighbor_signal(city, today_str_local, wind_octant)

            for ev in todays_events:
                event_id = ev.get("event_id") or ""

                # Skip if this event has already hit its MAX_BINS_PER_EVENT
                # cap today in this mode.  Avoids the NWS/Open-Meteo cost.
                buys_already = (event_buys_today_count(conn, event_id, PREDICTOR_MODE)
                                  if event_id else 0)
                if buys_already >= MAX_BINS_PER_EVENT:
                    n_events_skipped_already_bought += 1
                    continue

                pred = predict_bins(ev, nws_obs, forecast, nbr_signal,
                                     now_local.hour, city=city)
                if not pred.get("bins"):
                    continue

                # NEW: rank bins by OUR PROBABILITY descending.  The top
                # MAX_BINS_PER_EVENT are eligible candidates.  Lower-P bins
                # are explicitly excluded — the user's rule is "never buy a
                # bin if a higher-P bin exists in the same event."  Within
                # the top-N, we evaluate in P-rank order: if the #1 bin
                # fails gates, the whole event is aborted (we never fall
                # through to less-confident bins).
                bins_by_p = sorted(pred["bins"], key=lambda x: -x["our_prob"])
                slots_remaining = MAX_BINS_PER_EVENT - buys_already
                top_candidates  = bins_by_p[:slots_remaining]
                non_candidates  = bins_by_p[slots_remaining:]

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

                    if event_aborted:
                        ok = False
                        reason = "event_aborted (higher-P bin failed gates)"
                    else:
                        acted = already_acted_today(conn, event_id, contract_id,
                                                      PREDICTOR_MODE)
                        ok, reason = evaluate_gates(
                            current_hour    = now_local.hour,
                            edge            = edge,
                            market_p        = market_p,
                            liquidity       = liquidity,
                            deployed_today  = deployed,
                            trades_today    = n_trades,
                            already_acted   = acted,
                        )
                        # NEW: enforce per-scan cap (covers race within scan)
                        if ok and events_bought_this_scan.get(event_id, 0) >= MAX_BINS_PER_EVENT - buys_already:
                            ok = False
                            reason = "event_cap_reached_this_scan"
                        # NEW: if top-P (rank 0) bin fails gates, abort
                        # the rest of this event per the strict rule.
                        if not ok and rank_idx == 0:
                            event_aborted = True

                    if ok:
                        stake = compute_stake(edge, market_p, PAPER_BANKROLL_USD)
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
                        events_bought_this_scan[event_id] = (
                            events_bought_this_scan.get(event_id, 0) + 1
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

                # Write SKIP rows for non-candidate bins (those with lower
                # our_p than the top MAX_BINS_PER_EVENT).  We still want
                # them in the dashboard for visibility — operators want to
                # see what the model thought of every bin, not just the
                # ones we considered buying.
                for b in non_candidates:
                    n_bins_evaluated += 1
                    reason = "not_top_p_in_event"
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