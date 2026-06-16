"""
boundary_watcher.py — Latency-arbitrage strategy on bin-boundary crossings.

THESIS
======
Polymarket re-prices reactively.  When a temperature reading crosses a
bin boundary, the market re-prices within seconds to a few minutes —
but the bot, polling NWS observations more aggressively than the
market re-prices, can sometimes see the move BEFORE Polymarket reflects
it.  The edge isn't predictive; it's latency-based execution against
a market that hasn't caught up yet.

THE LOAD-BEARING SIGNAL: THE T-GROUP AT :53
============================================
Two reading sources from NWS observations:
  - SPECI body (whole-degC precision):  every 5 min
  - T-group remarks (tenths-degC):       at :53 only (hourly METAR)

For a 2°F US bin (covers true temps in [bin_lo - 0.5, bin_hi + 0.5)°F),
the SPECI body's ±0.5°C ≈ ±0.9°F rounding window is too coarse to fit
cleanly inside the bin — a body reading at the boundary is ambiguous
("temp is in [33.5, 34.5)°C → bin could be 93°F OR 94°F").  Only the
T-group's tenths precision unambiguously resolves which side of the
boundary the true temperature is on.

So:
  - Trigger B (T-group at :53) = the strategy.  ~95% of fires.
  - Trigger A (SPECI body) = logged bonus that fires only when the
                              body reading's rounding window fits
                              entirely INSIDE the bin's settlement
                              range.  Rare for 2°F bins by design.
                              Don't loosen — that re-admits the
                              boundary jitter we excluded.

WHAT IT BYPASSES, WHAT IT RESPECTS
===================================
A confirmed boundary signal BYPASSES the predictor's:
  - at_target_today (per-event budget cap)
  - market_too_skeptical and liquid_market_strong_disagreement (W4)

It RESPECTS (its own dedicated):
  - BOUNDARY_DAILY_BUDGET_USD
  - BOUNDARY_MAX_TRADES_PER_DAY
  - priced_in (>= 0.95)
  - thin_book (liquidity floor)
  - dedup_today (no duplicate fire on same trigger)

DRY-RUN IS A SAFETY GATE
========================
PHASE 6 (this file's logging behavior) ships first and runs until:
  1. >=30 would_fire rows logged
  2. Acceptable contradiction rate (provisional T-group disagreement)
  3. Favorable market_prob_60s_later distribution
  4. Manual operator sign-off (no auto-flip)

Until all four clear, BOUNDARY_DRY_RUN=1 (default) means
the watcher logs every trigger evaluation but NEVER places an order.

INITIAL SCOPE
=============
US 2°F bins only.  International (°C bins) and 1°F bins skip silently —
the T-group reliability and the strong-margin math are different and
need separate validation.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("boundary_watcher")


# ===========================================================================
# Configuration — all env-tunable
# ===========================================================================

# Master switches.  Both default OFF; explicit opt-in required.
BOUNDARY_STRATEGY_ENABLED = bool(int(
    os.getenv("BOUNDARY_STRATEGY_ENABLED", "0")))
# Dry-run = log fire decisions, don't execute.  Default 1 (safe).
# Flipping to 0 requires the four-gate sign-off above.
BOUNDARY_DRY_RUN = bool(int(os.getenv("BOUNDARY_DRY_RUN", "1")))

# --- Phase 1: arming ---------------------------------------------------
# Forecast must be within this many °C BELOW the next bin's boundary
# for the watcher to arm.
BOUNDARY_ARM_FORECAST_MARGIN_C = float(
    os.getenv("BOUNDARY_ARM_FORECAST_MARGIN_C", "0.5"))
# Next-bin's market price must be at most this much to be "underpriced"
# enough that the latency edge could exist.
BOUNDARY_ARM_MAX_MARKET_PRICE = float(
    os.getenv("BOUNDARY_ARM_MAX_MARKET_PRICE", "0.05"))
# Minimum minutes of heating remaining (current_local_hour < peak_hour
# requires at least this gap).  No point arming late in the day.
BOUNDARY_ARM_MIN_HEATING_MIN = int(
    os.getenv("BOUNDARY_ARM_MIN_HEATING_MIN", "30"))
# How long after forecast peak to keep watching before disarming.
BOUNDARY_DISARM_GRACE_MIN = int(
    os.getenv("BOUNDARY_DISARM_GRACE_MIN", "120"))
# Disarm if observed_max stays this far BELOW boundary post-peak.
BOUNDARY_DISARM_OBS_MARGIN_F = float(
    os.getenv("BOUNDARY_DISARM_OBS_MARGIN_F", "1.0"))

# --- Phase 2: triggers -------------------------------------------------
# Trigger A (SPECI body) classification thresholds.  See module docstring
# for why these matter and why "strong" is rare on 2°F bins by design.
BOUNDARY_PROVISIONAL_MARGIN_F = float(
    os.getenv("BOUNDARY_PROVISIONAL_MARGIN_F", "1.0"))
BOUNDARY_STRONG_MARGIN_F = float(
    os.getenv("BOUNDARY_STRONG_MARGIN_F", "2.0"))

# --- Phase 3: polling cadence -----------------------------------------
BOUNDARY_NORMAL_POLL_SEC = int(os.getenv("BOUNDARY_NORMAL_POLL_SEC", "120"))
BOUNDARY_HARD_POLL_SEC = int(os.getenv("BOUNDARY_HARD_POLL_SEC", "20"))
BOUNDARY_HARD_POLL_START_MIN = int(
    os.getenv("BOUNDARY_HARD_POLL_START_MIN", "52"))
BOUNDARY_HARD_POLL_END_MIN = int(
    os.getenv("BOUNDARY_HARD_POLL_END_MIN", "59"))
BOUNDARY_BYPASS_CACHE_IN_HARD_POLL = bool(int(
    os.getenv("BOUNDARY_BYPASS_CACHE_IN_HARD_POLL", "1")))

# --- Phase 4: gate carve-outs (only matter in live mode) --------------
BOUNDARY_BYPASS_AT_TARGET = bool(int(
    os.getenv("BOUNDARY_BYPASS_AT_TARGET", "1")))
BOUNDARY_BYPASS_DISAGREEMENT_VETO = bool(int(
    os.getenv("BOUNDARY_BYPASS_DISAGREEMENT_VETO", "1")))
BOUNDARY_RESPECT_PRICED_IN = bool(int(
    os.getenv("BOUNDARY_RESPECT_PRICED_IN", "1")))
BOUNDARY_RESPECT_THIN_BOOK = bool(int(
    os.getenv("BOUNDARY_RESPECT_THIN_BOOK", "1")))

# --- Phase 5: sizing + execution + dedicated caps ---------------------
BOUNDARY_TARGET_STAKE_USD = float(
    os.getenv("BOUNDARY_TARGET_STAKE_USD", "20"))
BOUNDARY_DAILY_BUDGET_USD = float(
    os.getenv("BOUNDARY_DAILY_BUDGET_USD", "100"))
BOUNDARY_MAX_TRADES_PER_DAY = int(
    os.getenv("BOUNDARY_MAX_TRADES_PER_DAY", "10"))
# Stand down if the market has already mostly repriced.  0.15-0.20 is
# the right zone — the Miami case showed prices going 0.02 → 0.4775
# → 0.98; by 0.48 the latency race is already lost.  Tune via
# dry-run's would-fire-price distribution.
BOUNDARY_MAX_ENTRY_PRICE = float(
    os.getenv("BOUNDARY_MAX_ENTRY_PRICE", "0.20"))
BOUNDARY_RETRY_ON_PARTIAL = bool(int(
    os.getenv("BOUNDARY_RETRY_ON_PARTIAL", "1")))
BOUNDARY_MAX_RETRIES = int(os.getenv("BOUNDARY_MAX_RETRIES", "10"))
BOUNDARY_RETRY_DELAY_SEC = int(os.getenv("BOUNDARY_RETRY_DELAY_SEC", "2"))
BOUNDARY_RETRY_MAX_PRICE = float(
    os.getenv("BOUNDARY_RETRY_MAX_PRICE", "0.50"))
BOUNDARY_WALK_CENTS = int(os.getenv("BOUNDARY_WALK_CENTS", "20"))
BOUNDARY_MAX_PRICE_CAP = float(os.getenv("BOUNDARY_MAX_PRICE_CAP", "0.95"))

# --- Phase 6: logging + lookahead --------------------------------------
BOUNDARY_REPRICE_LOOKAHEAD_SEC = int(
    os.getenv("BOUNDARY_REPRICE_LOOKAHEAD_SEC", "60"))
BOUNDARY_REPRICE_LOOKAHEAD_LONG_SEC = int(
    os.getenv("BOUNDARY_REPRICE_LOOKAHEAD_LONG_SEC", "300"))


# ===========================================================================
# Pure unit-handling helpers (testable in isolation)
# ===========================================================================

def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def reading_in_settlement_unit(temp_c: float, settlement_unit: str) -> float:
    """Convert a Celsius reading to whatever unit the bin settles in.
    Settlement unit comes from the bin row's `unit` column."""
    if str(settlement_unit).lower() == "fahrenheit":
        return c_to_f(temp_c)
    return temp_c   # already Celsius


def bin_settlement_range(bin_lo: float, bin_hi: float,
                            settlement_unit: str) -> tuple[float, float]:
    """Bin's actual settlement range in settlement units, half-up rounding.

    A bin labelled "94-95°F" (lo=94, hi=95) captures true temps in
    [93.5, 95.5)°F.  Half-step is applied in the bin's native unit
    (matches existing bin_temp_range() in intraday_predictor.py).
    """
    # The bin's lo/hi are stored in settlement_unit, so the half-step
    # is in that unit.  Return (lower_inclusive, upper_exclusive).
    return (float(bin_lo) - 0.5, float(bin_hi) + 0.5)


def settlement_unit_round(reading_c: float, settlement_unit: str) -> int:
    """Convert a Celsius reading to settlement unit and apply half-up
    rounding to an integer.  This is what determines which bin a
    given reading 'settles' in.

    Half-up: 0.5 rounds up (93.5 -> 94, not 93).  Matches Polymarket
    half-up bin convention.
    """
    v = reading_in_settlement_unit(reading_c, settlement_unit)
    return int(v + 0.5) if v >= 0 else -int(-v + 0.5)


# ===========================================================================
# Bin geometry — initial scope check
# ===========================================================================

def is_supported_bin(bin_lo: float, bin_hi: float, unit: str) -> bool:
    """The boundary strategy ships for US 2°F bins ONLY in v1.
    1°F bins, °C bins, and open-ended bins (>=/<=) all return False —
    the trigger math and T-group availability differ per class and
    we're building/proving on one class at a time.
    """
    if bin_lo is None or bin_hi is None:
        return False    # open-ended bins (≤X or ≥X) not supported v1
    if str(unit).lower() != "fahrenheit":
        return False    # °C international bins not supported v1
    width = float(bin_hi) - float(bin_lo)
    return abs(width - 1.0) < 0.01   # "94-95" stored as lo=94 hi=95 → width=1
    # (the displayed label says "94-95°F" but the half-step makes the
    # actual range [93.5, 95.5) which is 2°F wide; lo/hi being adjacent
    # integers is the 2°F-bin signature)


# ===========================================================================
# Phase 1 — Arming
# ===========================================================================

@dataclass
class ArmingState:
    armed:   bool
    reason:  str           # human-readable why-armed-or-not
    target_bin_lo:  Optional[float] = None
    target_bin_hi:  Optional[float] = None
    boundary_value_settlement: Optional[float] = None


def compute_arming_state(
    *,
    forecast_high_c:     float,
    forecast_peak_hour:  int,
    current_local_hour:  int,
    settlement_unit:     str,
    candidate_bin_lo:    float,
    candidate_bin_hi:    float,
    candidate_market_p:  float,
    observed_max_c:      float,
) -> ArmingState:
    """Decide whether to arm a watcher for (city, event, candidate_bin).

    Caller is responsible for selecting the candidate bin as "the next
    bin above the forecast" (the bin whose lower edge is the smallest
    value > forecast in settlement units).  We don't pick the bin
    here — we just validate it.
    """
    # Bin must be one we support (US 2°F)
    if not is_supported_bin(candidate_bin_lo, candidate_bin_hi, settlement_unit):
        return ArmingState(False, "unsupported_bin_geometry")

    # Boundary = bin's lower edge in settlement unit
    bin_range = bin_settlement_range(candidate_bin_lo, candidate_bin_hi,
                                         settlement_unit)
    boundary  = bin_range[0]
    fc_in_settlement = reading_in_settlement_unit(forecast_high_c,
                                                     settlement_unit)
    # Forecast must be BELOW the boundary (above-boundary forecast =
    # no latency-arb shape; the market has already priced the bin in).
    if fc_in_settlement >= boundary:
        return ArmingState(False, "forecast_at_or_above_boundary",
                              candidate_bin_lo, candidate_bin_hi, boundary)
    # And forecast must be WITHIN margin of the boundary
    fc_margin_settlement = boundary - fc_in_settlement
    margin_unit = (BOUNDARY_ARM_FORECAST_MARGIN_C
                   if str(settlement_unit).lower() == "celsius"
                   else BOUNDARY_ARM_FORECAST_MARGIN_C * 9.0 / 5.0)
    if fc_margin_settlement > margin_unit:
        return ArmingState(False,
            f"forecast_too_far_below_boundary "
            f"({fc_margin_settlement:.2f} > {margin_unit:.2f})",
            candidate_bin_lo, candidate_bin_hi, boundary)

    # Next-bin must be underpriced
    if candidate_market_p is None:
        return ArmingState(False, "no_market_p", candidate_bin_lo,
                              candidate_bin_hi, boundary)
    if candidate_market_p > BOUNDARY_ARM_MAX_MARKET_PRICE:
        return ArmingState(False,
            f"market_p_already_too_high "
            f"({candidate_market_p:.4f} > {BOUNDARY_ARM_MAX_MARKET_PRICE})",
            candidate_bin_lo, candidate_bin_hi, boundary)

    # Heating time must remain
    minutes_to_peak = (forecast_peak_hour - current_local_hour) * 60
    if minutes_to_peak < BOUNDARY_ARM_MIN_HEATING_MIN:
        return ArmingState(False,
            f"too_late_in_day "
            f"({minutes_to_peak}min < {BOUNDARY_ARM_MIN_HEATING_MIN}min to peak)",
            candidate_bin_lo, candidate_bin_hi, boundary)

    # Disarm post-peak if observed_max never approached boundary
    minutes_past_peak = (current_local_hour - forecast_peak_hour) * 60
    if minutes_past_peak > BOUNDARY_DISARM_GRACE_MIN:
        obs_settlement = reading_in_settlement_unit(observed_max_c,
                                                       settlement_unit) \
                         if observed_max_c is not None and observed_max_c > -50 \
                         else None
        if (obs_settlement is None
            or (boundary - obs_settlement) > BOUNDARY_DISARM_OBS_MARGIN_F):
            return ArmingState(False,
                f"past_peak_no_crossing_in_sight "
                f"({minutes_past_peak}min past peak)",
                candidate_bin_lo, candidate_bin_hi, boundary)

    return ArmingState(True, "armed", candidate_bin_lo, candidate_bin_hi,
                          boundary)


# ===========================================================================
# Phase 2 — Trigger evaluation
# ===========================================================================

@dataclass
class TriggerEval:
    """Classification of a single METAR cycle against an armed bin."""
    classification:    str          # no_signal | at_boundary | strong | confirmed
    margin_from_boundary: float     # signed, in settlement unit
    reading_settlement: float
    would_fire:        bool
    size_usd:          float
    notes:             str


def evaluate_trigger(
    *,
    reading_c:           float,
    is_t_group:          bool,         # True = tenths precision available
    bin_lo:              float,
    bin_hi:              float,
    settlement_unit:     str,
) -> TriggerEval:
    """Classify a METAR cycle's temperature against an armed bin.

    Two paths:

      A. SPECI body (is_t_group=False, whole-°C precision):
         - The reading's rounding window (±0.5 in settlement unit) must
           fit ENTIRELY inside the bin's [lo-0.5, hi+0.5) range for the
           signal to be "strong" (no rounding ambiguity).  For 2°F bins
           this is rare by design — see module docstring.
         - At-boundary readings (window touches bin edge) are LOGGED but
           never fire — we deliberately skip them because that's the
           jitter zone the strategy was redesigned to exclude.

      B. T-group (is_t_group=True, tenths precision):
         - Tenths precision (±0.05°C ≈ ±0.09°F) fits cleanly inside a
           bin's settlement range.
         - Confirmed if rounded value lands inside bin; contradicted
           otherwise.  Contradicted is no_signal here (we never opened
           a provisional position to exit; the contradiction matters
           only for measurement, not action).
    """
    bin_range = bin_settlement_range(bin_lo, bin_hi, settlement_unit)
    boundary_lo = bin_range[0]
    boundary_hi = bin_range[1]
    reading_settlement = reading_in_settlement_unit(reading_c, settlement_unit)
    margin_from_lo = reading_settlement - boundary_lo

    if is_t_group:
        # Tenths precision — direct settlement test.
        if reading_settlement >= boundary_lo and reading_settlement < boundary_hi:
            return TriggerEval(
                classification="confirmed",
                margin_from_boundary=margin_from_lo,
                reading_settlement=reading_settlement,
                would_fire=True,
                size_usd=BOUNDARY_TARGET_STAKE_USD,
                notes="t_group reading lands inside bin",
            )
        if reading_settlement < boundary_lo:
            return TriggerEval(
                classification="contradicted",
                margin_from_boundary=margin_from_lo,
                reading_settlement=reading_settlement,
                would_fire=False,
                size_usd=0.0,
                notes="t_group reading below boundary — would have been wrong",
            )
        # Above bin's upper edge (skipped past the target bin)
        return TriggerEval(
            classification="no_signal",
            margin_from_boundary=margin_from_lo,
            reading_settlement=reading_settlement,
            would_fire=False,
            size_usd=0.0,
            notes="t_group reading above bin's upper edge — past target bin",
        )

    # SPECI body — whole-°C precision.
    # The reading's rounding window in settlement units: convert one
    # half-degree-C into settlement units to get the half-window width.
    half_window_settlement = (0.5 if str(settlement_unit).lower() == "celsius"
                                  else 0.5 * 9.0 / 5.0)   # ≈ 0.9°F
    window_lo = reading_settlement - half_window_settlement
    window_hi = reading_settlement + half_window_settlement
    # Strong = the entire ±half_window fits INSIDE the bin's settlement
    # range.  That is the only configuration where rounding can't be
    # hiding a reversal.
    fits_inside_bin = (window_lo >= boundary_lo and window_hi <= boundary_hi)
    if fits_inside_bin:
        return TriggerEval(
            classification="strong",
            margin_from_boundary=margin_from_lo,
            reading_settlement=reading_settlement,
            would_fire=True,
            size_usd=BOUNDARY_TARGET_STAKE_USD,
            notes="speci body rounding window fits inside bin",
        )
    # At-boundary or off-target — log but don't fire.
    if margin_from_lo > -half_window_settlement and reading_settlement < boundary_hi:
        return TriggerEval(
            classification="at_boundary",
            margin_from_boundary=margin_from_lo,
            reading_settlement=reading_settlement,
            would_fire=False,
            size_usd=0.0,
            notes="speci body at boundary — jitter zone, hold for t_group",
        )
    return TriggerEval(
        classification="no_signal",
        margin_from_boundary=margin_from_lo,
        reading_settlement=reading_settlement,
        would_fire=False,
        size_usd=0.0,
        notes="speci body outside bin",
    )


# ===========================================================================
# Phase 3 — Polling cadence helper
# ===========================================================================

def in_hard_poll_window(now_utc: datetime) -> bool:
    """True if we're in the :52-:59 minute-of-hour window where the
    T-group confirmation can arrive.  This is THE window the entire
    strategy optimizes for; one chance per hour to fire on confirmation."""
    m = now_utc.minute
    return (BOUNDARY_HARD_POLL_START_MIN <= m <= BOUNDARY_HARD_POLL_END_MIN)


# ===========================================================================
# Phase 6 — Logging
# ===========================================================================

def write_trigger_log(conn: sqlite3.Connection, row: dict) -> int:
    """Insert one row into boundary_trigger_log.  Always called
    regardless of fire/no-fire — that's how the dry-run measures
    the contradiction rate and lookahead distribution.
    """
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    cur = conn.execute(
        f"INSERT INTO boundary_trigger_log ({','.join(cols)}) "
        f"VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def populate_lookahead(conn: sqlite3.Connection,
                          fetch_market_p_now,    # callable: (contract_id) -> float|None
                          ) -> int:
    """Periodic pass over boundary_trigger_log rows whose
    evaluated_at_utc is >= 60s (or >=300s) in the past and whose
    market_prob_60s_later / _300s_later are NULL.  Fetches the current
    market_p for the row's contract and stamps it.

    Run this as its own light APScheduler job (every ~30s) — the
    market_prob_60s_later and _300s_later columns are the single most
    important calibration metric and need to be populated even when
    the bot is idle (no new triggers firing).
    """
    now = datetime.now(timezone.utc)
    cutoff_60 = (now - timedelta(seconds=BOUNDARY_REPRICE_LOOKAHEAD_SEC)).isoformat()
    cutoff_300 = (now - timedelta(seconds=BOUNDARY_REPRICE_LOOKAHEAD_LONG_SEC)).isoformat()

    updated = 0

    # 60s lookahead
    rows = conn.execute(
        "SELECT id, contract_id FROM boundary_trigger_log "
        "WHERE evaluated_at_utc <= ? AND market_prob_60s_later IS NULL "
        "  AND contract_id IS NOT NULL "
        "ORDER BY evaluated_at_utc DESC LIMIT 50",
        (cutoff_60,),
    ).fetchall()
    for rid, cid in rows:
        try:
            p = fetch_market_p_now(cid)
        except Exception as e:
            log.warning(f"lookahead fetch failed for {cid[:12]}: {e}")
            continue
        if p is not None:
            conn.execute(
                "UPDATE boundary_trigger_log SET market_prob_60s_later = ? "
                "WHERE id = ?", (float(p), rid))
            updated += 1
    # 300s lookahead — same pattern
    rows = conn.execute(
        "SELECT id, contract_id FROM boundary_trigger_log "
        "WHERE evaluated_at_utc <= ? AND market_prob_300s_later IS NULL "
        "  AND contract_id IS NOT NULL "
        "ORDER BY evaluated_at_utc DESC LIMIT 50",
        (cutoff_300,),
    ).fetchall()
    for rid, cid in rows:
        try:
            p = fetch_market_p_now(cid)
        except Exception as e:
            log.warning(f"lookahead fetch failed for {cid[:12]}: {e}")
            continue
        if p is not None:
            conn.execute(
                "UPDATE boundary_trigger_log SET market_prob_300s_later = ? "
                "WHERE id = ?", (float(p), rid))
            updated += 1
    if updated:
        conn.commit()
    return updated


# ===========================================================================
# Daily-cap accounting (Phase 5 — independent of predictor's caps)
# ===========================================================================

def boundary_trades_today(conn: sqlite3.Connection) -> int:
    """Count today's boundary fires (actually_fired=1) — used to enforce
    BOUNDARY_MAX_TRADES_PER_DAY."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM boundary_trigger_log "
            "WHERE substr(evaluated_at_utc, 1, 10) = ? "
            "  AND actually_fired = 1",
            (today,),
        ).fetchone()[0]
        return int(n or 0)
    except sqlite3.OperationalError:
        return 0


def boundary_budget_spent_today(conn: sqlite3.Connection) -> float:
    """Sum of would_fire_size_usd for actually-fired rows today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        s = conn.execute(
            "SELECT COALESCE(SUM(would_fire_size_usd), 0) "
            "FROM boundary_trigger_log "
            "WHERE substr(evaluated_at_utc, 1, 10) = ? "
            "  AND actually_fired = 1",
            (today,),
        ).fetchone()[0]
        return float(s or 0)
    except sqlite3.OperationalError:
        return 0.0


# ===========================================================================
# Phase 3 + 4 + 5: Live execution
#
# _execute_boundary_fire wraps execute_signal with the strategy's own
# safety gates.  Phase 4 carve-outs (bypass per-event cap, bypass W4
# disagreement veto, allow 2nd bin on same event) are achieved by NOT
# going through the predictor's gate stack — boundary calls execute_signal
# directly, so none of predictor's gate logic applies.  The gates that
# DO apply are:
#   - BOUNDARY_DRY_RUN: hard kill switch, default ON.  Always returns
#     actually_fired=0 with notes="dry_run_safety_gate" when set.
#   - BOUNDARY_MAX_TRADES_PER_DAY / BOUNDARY_DAILY_BUDGET_USD: dedicated
#     daily caps independent of predictor's MAX_TRADES_PER_DAY.
#   - BOUNDARY_MAX_ENTRY_PRICE: already enforced in evaluate_one_event
#     before this is called (sets would_fire=False if violated).
#   - execute_signal's own thin-book skip + buy-retry cap: kept as-is.
#
# Phase 5 — retry-on-partial — places ONE order per call to execute_signal,
# then if filled $ < target_$ and remaining capacity > $1, walks the price
# up by BOUNDARY_WALK_CENTS and places another order (DOES NOT cancel the
# prior resting order — older limits stay on book and may still match
# cheaper depth as it appears).  Capped at BOUNDARY_MAX_RETRIES and
# BOUNDARY_RETRY_MAX_PRICE.  Each placed order writes its own positions
# row; the dashboard rolls these up by (city, event_date, contract_id).
# ===========================================================================

def _execute_boundary_fire(
    *, conn: sqlite3.Connection,
    city: str,
    event_date: str,
    candidate: dict,
    signal_origin: str,
) -> dict:
    """Place a boundary order (or a sequence of price-walked orders for
    Phase 5 retry-on-partial).  Returns a dict with execution outcome
    keys to be merged into the trigger_log row:
        actually_fired, actual_order_id, fill_price, shares_filled_total,
        retries_used, exec_notes
    """
    import time

    # ---- Phase 3 gate: dry-run safety switch -----------------------
    # Always honored regardless of every other condition.  This is the
    # operator's hard kill switch until the four-gate sign-off clears.
    if BOUNDARY_DRY_RUN:
        return {
            "actually_fired":      0,
            "actual_order_id":     None,
            "exec_notes":          "dry_run_safety_gate",
        }

    # ---- Phase 5 gate: daily caps ----------------------------------
    trades_today = boundary_trades_today(conn)
    if trades_today >= BOUNDARY_MAX_TRADES_PER_DAY:
        return {
            "actually_fired":      0,
            "actual_order_id":     None,
            "exec_notes":          (f"daily_trade_cap "
                                      f"({trades_today}/{BOUNDARY_MAX_TRADES_PER_DAY})"),
        }
    spent_today = boundary_budget_spent_today(conn)
    if spent_today + BOUNDARY_TARGET_STAKE_USD > BOUNDARY_DAILY_BUDGET_USD:
        return {
            "actually_fired":      0,
            "actual_order_id":     None,
            "exec_notes":          (f"daily_budget_cap "
                                      f"(${spent_today:.2f}+${BOUNDARY_TARGET_STAKE_USD:.2f}"
                                      f" > ${BOUNDARY_DAILY_BUDGET_USD:.2f})"),
        }

    # ---- Build the execute_signal payload --------------------------
    try:
        from execution import execute_signal, get_clob_client
    except Exception as e:
        return {
            "actually_fired":      0,
            "actual_order_id":     None,
            "exec_notes":          f"execute_signal_import_failed: {e}",
        }
    client = get_clob_client()
    if client is None:
        return {
            "actually_fired":      0,
            "actual_order_id":     None,
            "exec_notes":          "clob_client_unavailable",
        }

    market_p_now = float(candidate.get("market_prob") or 0.05)
    base_sig = {
        "contract_id":      candidate["contract_id"],
        "yes_token_id":     candidate.get("yes_token_id"),
        "no_token_id":      None,
        "recommended_side": "YES",
        "market_p":         market_p_now,
        "yes_price":        market_p_now,
        "city":             city,
        "date":             event_date,
        "event_id":         None,
        "gamma_market_id":  None,
        "model_prob":       None,
        "edge":             None,
        "strategy":         "boundary_watcher",
        "question":         f"{city} {event_date} {candidate['bin_label']}",
        "range_low":        candidate["bin_range_low"],
        "range_high":       candidate["bin_range_high"],
        "unit":             candidate["unit"],
        "max_price_cap":    BOUNDARY_MAX_PRICE_CAP,
        "signal_origin":    signal_origin,
        "forecast_sigma_c": None,
    }

    # ---- Phase 5 retry-on-partial loop -----------------------------
    target_usdc = BOUNDARY_TARGET_STAKE_USD
    cumulative_filled_usdc = 0.0
    walked_price = market_p_now
    last_order_id: Optional[str] = None
    last_fill_price: Optional[float] = None
    retries_used = 0
    notes_parts: list = []

    while True:
        remaining = target_usdc - cumulative_filled_usdc
        if remaining < 1.0:
            notes_parts.append(f"filled_to_target (${cumulative_filled_usdc:.2f}/${target_usdc:.2f})")
            break

        sig = dict(base_sig)
        sig["market_p"]  = walked_price
        sig["yes_price"] = walked_price
        sig["kelly_size"] = remaining

        try:
            result = execute_signal(sig, client=client)
        except Exception as e:
            log.exception(f"boundary execute_signal raised: {e}")
            notes_parts.append(f"execute_raised:{e}")
            break

        status = (result or {}).get("status", "unknown")
        order_id = (result or {}).get("order_id")
        if order_id:
            last_order_id = order_id

        # Accept any path that opened a position row, even a paper one
        # (covers PAPER_TRADE deployments where boundary lives alongside
        # paper predictor).  For "placed" responses we count the
        # SUBMITTED (final_size_usdc) — fills land asynchronously via
        # the monitor; treating the submission as committed capital is
        # the correct cap-side accounting.
        if status in ("placed", "paper", "filled"):
            # execute_signal returns entry_price (actual or limit) and shares
            fill_px = float(result.get("entry_price") or walked_price)
            shares = float(result.get("shares") or 0)
            this_filled_usdc = shares * fill_px
            cumulative_filled_usdc += this_filled_usdc
            last_fill_price = fill_px
            notes_parts.append(
                f"order#{retries_used}={status}@{fill_px:.4f} "
                f"${this_filled_usdc:.2f}"
            )
        elif status == "skip":
            reason = (result or {}).get("reason", "?")
            notes_parts.append(f"order#{retries_used}=skip:{reason}")
            # Don't retry on book-too-thin or retry-cap reasons —
            # they won't resolve in 2 sec.
            break
        else:
            notes_parts.append(f"order#{retries_used}={status}")
            # Hard failure / unmatched — don't retry.
            break

        if not BOUNDARY_RETRY_ON_PARTIAL:
            break

        # Has the most recent fill brought us to target?  If so, stop
        # BEFORE incrementing retries_used — retries_used is the count
        # of EXTRA orders beyond the first, not the number of loop
        # iterations.
        if target_usdc - cumulative_filled_usdc < 1.0:
            notes_parts.append(
                f"filled_to_target (${cumulative_filled_usdc:.2f}/${target_usdc:.2f})"
            )
            break

        if retries_used >= BOUNDARY_MAX_RETRIES:
            notes_parts.append(f"max_retries_hit ({BOUNDARY_MAX_RETRIES})")
            break

        # Walk the price up for the next attempt.
        new_walked = walked_price + (BOUNDARY_WALK_CENTS / 100.0)
        if new_walked > BOUNDARY_RETRY_MAX_PRICE:
            notes_parts.append(
                f"retry_max_price_hit (next={new_walked:.2f} > "
                f"{BOUNDARY_RETRY_MAX_PRICE:.2f})"
            )
            break
        walked_price = new_walked
        retries_used += 1
        time.sleep(max(0, BOUNDARY_RETRY_DELAY_SEC))

    actually_fired = 1 if cumulative_filled_usdc > 0 else 0
    return {
        "actually_fired":      actually_fired,
        "actual_order_id":     last_order_id,
        "fill_price":          last_fill_price,
        "shares_filled_total": cumulative_filled_usdc,
        "retries_used":        retries_used,
        "exec_notes":          "; ".join(notes_parts) or "no_attempt",
    }


# ===========================================================================
# Conflict check — does the predictor already hold a bin on this event?
# ===========================================================================

def find_candidate_bin(conn: sqlite3.Connection, city: str,
                          event_date: str, forecast_high_c: float,
                          ) -> Optional[dict]:
    """Find the single 'next bin above forecast' for this event.

    Returns a dict with bin metadata + latest market_prob, or None if no
    eligible bin exists.  Uses the latest scan's bin set so the prices
    are fresh; ties broken by lowest range_low (the closest bin above
    the forecast).
    """
    forecast_f = c_to_f(forecast_high_c)
    try:
        rows = conn.execute(
            """
            SELECT s.contract_id, s.yes_token_id, s.bin_label,
                   s.bin_range_low, s.bin_range_high, s.unit,
                   s.market_prob, s.liquidity_usd
            FROM paper_predictor_signals s
            WHERE s.city = ? AND s.event_date = ?
              AND s.bin_range_low IS NOT NULL
              AND s.bin_range_high IS NOT NULL
              AND s.scanned_at_utc = (
                SELECT MAX(scanned_at_utc) FROM paper_predictor_signals
                WHERE city = ? AND event_date = ?
              )
            """,
            (city, event_date, city, event_date),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    # Filter: bins whose lower-edge settlement value > forecast.  Pick the
    # smallest such — the closest unfilled bin above the forecast.
    candidates = []
    for r in rows:
        cid, tok, lbl, lo, hi, unit, mkt, liq = r
        if not is_supported_bin(lo, hi, unit):
            continue
        boundary = bin_settlement_range(lo, hi, unit)[0]
        fc_settlement = (forecast_f if str(unit).lower() == "fahrenheit"
                         else float(forecast_high_c))
        if boundary > fc_settlement:
            candidates.append({
                "contract_id":  cid,
                "yes_token_id": tok,
                "bin_label":    lbl,
                "bin_range_low": float(lo),
                "bin_range_high": float(hi),
                "unit":         unit,
                "market_prob":  float(mkt) if mkt is not None else None,
                "liquidity_usd": float(liq) if liq is not None else None,
                "boundary":     boundary,
            })
    if not candidates:
        return None
    # Smallest boundary = closest above forecast = the one in play first
    candidates.sort(key=lambda c: c["boundary"])
    return candidates[0]


def latest_metar_cycle(conn: sqlite3.Connection, icao: str,
                          event_date: str) -> Optional[dict]:
    """Most recent METAR cycle persisted in raw_metar_log for this
    (icao, event_date).  Returns the cycle dict including temp_c,
    temp_precision, raw_message, cycle_timestamp_utc.
    """
    try:
        row = conn.execute(
            """SELECT cycle_timestamp_utc, temp_c, temp_precision, raw_message
               FROM raw_metar_log
               WHERE icao = ? AND event_date = ? AND temp_c IS NOT NULL
               ORDER BY cycle_timestamp_utc DESC LIMIT 1""",
            (icao, event_date),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return {
        "cycle_timestamp_utc": row[0],
        "temp_c":              float(row[1]),
        "temp_precision":      row[2] or "whole",
        "raw_message":         row[3] or "",
    }


def evaluate_one_event(
    *, conn: sqlite3.Connection,
    city:             str,
    event_date:       str,
    icao:             str,
    tz_str:           str,
    forecast_high_c:  float,
    forecast_peak_hour: int,
    current_local_hour: int,
    observed_max_c:   Optional[float],
) -> Optional[int]:
    """Per-tick evaluation for one (city, event_date).  Returns the
    trigger-log row id if a row was written, else None.

    NEVER places an order, EVEN in live mode.  This function only
    decides 'would fire vs would not' and logs.  A separate execution
    step (not in v1) would consume the would_fire flag and place the
    order — that step ships only after the dry-run safety gate clears.
    """
    if not BOUNDARY_STRATEGY_ENABLED:
        return None
    # Find the candidate bin
    candidate = find_candidate_bin(conn, city, event_date, forecast_high_c)
    if not candidate:
        return None
    # Check arming
    arm = compute_arming_state(
        forecast_high_c     = forecast_high_c,
        forecast_peak_hour  = forecast_peak_hour,
        current_local_hour  = current_local_hour,
        settlement_unit     = candidate["unit"],
        candidate_bin_lo    = candidate["bin_range_low"],
        candidate_bin_hi    = candidate["bin_range_high"],
        candidate_market_p  = candidate["market_prob"] or 0,
        observed_max_c      = observed_max_c if observed_max_c is not None else -100,
    )
    # Get the latest METAR cycle
    cycle = latest_metar_cycle(conn, icao, event_date)
    if not cycle:
        # No METAR to evaluate against — log "no_signal" with arm_state
        # so the dry-run shows the watcher considered this event.
        row = {
            "evaluated_at_utc":           datetime.now(timezone.utc).isoformat(),
            "city":                       city,
            "event_date":                 event_date,
            "contract_id":                candidate["contract_id"],
            "bin_label":                  candidate["bin_label"],
            "bin_range_low":              candidate["bin_range_low"],
            "bin_range_high":             candidate["bin_range_high"],
            "settlement_unit":            candidate["unit"],
            "trigger_classification":     "no_signal",
            "arm_state":                  "armed" if arm.armed else "never_armed",
            "would_fire":                 0,
            "would_fire_size_usd":        0.0,
            "would_fire_limit_price":     0.0,
            "actually_fired":             0,
            "conflict_with_predictor":    1 if predictor_holds_bin_on_event(conn, city, event_date) else 0,
            "market_prob_at_eval":        candidate["market_prob"],
            "forecast_high_c":            forecast_high_c,
            "forecast_peak_hour":         forecast_peak_hour,
            "observed_max_c":             observed_max_c,
            "notes":                      "no_metar_cycle_available",
        }
        return write_trigger_log(conn, row)
    # Run the trigger evaluator
    is_t_group = (cycle["temp_precision"] == "tenths")
    cycle_kind = "t_group" if is_t_group else "speci"
    tr = evaluate_trigger(
        reading_c       = cycle["temp_c"],
        is_t_group      = is_t_group,
        bin_lo          = candidate["bin_range_low"],
        bin_hi          = candidate["bin_range_high"],
        settlement_unit = candidate["unit"],
    )
    # Arming controls would_fire: a trigger that classifies as
    # confirmed/strong only fires if the event was armed.  An unarmed
    # event still logs the classification — useful for measuring how
    # often the strategy would have caught something but the arming
    # gate held it back.
    would_fire = bool(tr.would_fire and arm.armed)
    # Entry-price gate: market may have already repriced past our cap.
    if would_fire and candidate["market_prob"] is not None:
        if float(candidate["market_prob"]) > BOUNDARY_MAX_ENTRY_PRICE:
            would_fire = False
            arm_note_suffix = (f" | suppressed: market_p {candidate['market_prob']:.3f}"
                                 f" > BOUNDARY_MAX_ENTRY_PRICE {BOUNDARY_MAX_ENTRY_PRICE}")
        else:
            arm_note_suffix = ""
    else:
        arm_note_suffix = ""

    signal_origin = None
    if would_fire:
        signal_origin = ("boundary_confirmed" if tr.classification == "confirmed"
                          else "boundary_strong" if tr.classification == "strong"
                          else None)

    # Phase 3 + 5: execute (no-op under DRY_RUN).  We always populate
    # actually_fired/actual_order_id from this — under dry-run it returns
    # zeros; under live it returns the real outcome.  Skipped entirely
    # when would_fire=False (avoids logging "dry_run_safety_gate" notes
    # on every no-signal row).
    if would_fire:
        exec_result = _execute_boundary_fire(
            conn          = conn,
            city          = city,
            event_date    = event_date,
            candidate     = candidate,
            signal_origin = signal_origin or "boundary_unknown",
        )
    else:
        exec_result = {
            "actually_fired":  0,
            "actual_order_id": None,
            "exec_notes":      "",
        }

    row = {
        "evaluated_at_utc":           datetime.now(timezone.utc).isoformat(),
        "city":                       city,
        "event_date":                 event_date,
        "contract_id":                candidate["contract_id"],
        "bin_label":                  candidate["bin_label"],
        "bin_range_low":              candidate["bin_range_low"],
        "bin_range_high":             candidate["bin_range_high"],
        "settlement_unit":            candidate["unit"],
        "cycle_timestamp_utc":        cycle["cycle_timestamp_utc"],
        "cycle_kind":                 cycle_kind,
        "reading_native_c":           cycle["temp_c"],
        "reading_settlement":         tr.reading_settlement,
        "boundary_value_settlement":  arm.boundary_value_settlement,
        "margin_from_boundary":       tr.margin_from_boundary,
        "trigger_classification":     tr.classification,
        "arm_state":                  "armed" if arm.armed else "never_armed",
        "would_fire":                 1 if would_fire else 0,
        "would_fire_size_usd":        tr.size_usd if would_fire else 0.0,
        "would_fire_limit_price":     min((candidate["market_prob"] or 0) + 0.05,
                                            BOUNDARY_MAX_PRICE_CAP),
        "actually_fired":             int(exec_result.get("actually_fired", 0)),
        "actual_order_id":            exec_result.get("actual_order_id"),
        "conflict_with_predictor":    1 if predictor_holds_bin_on_event(conn, city, event_date) else 0,
        "market_prob_at_eval":        candidate["market_prob"],
        "forecast_high_c":            forecast_high_c,
        "forecast_peak_hour":         forecast_peak_hour,
        "observed_max_c":             observed_max_c,
        "signal_origin":              signal_origin,
        "notes":                      (f"arm={arm.reason}; trigger={tr.notes}"
                                         f"{arm_note_suffix}"
                                         + (f"; exec={exec_result.get('exec_notes', '')}"
                                              if would_fire else "")),
    }
    return write_trigger_log(conn, row)


def run_boundary_watcher_tick(conn: sqlite3.Connection) -> int:
    """Top-level entry called by the APScheduler job.  Iterates all
    today's events for cities the bot trades, calls evaluate_one_event
    for each, returns count of trigger rows written.

    Wrapped by caller in try/except — a bug here MUST NOT crash the
    predictor.  See main.py's _boundary_watcher_job wrapper.
    """
    if not BOUNDARY_STRATEGY_ENABLED:
        return 0
    # Discover today's events.  Use the same source the predictor uses —
    # any (city, event_date) that has a latest-scan signal row.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        events = conn.execute(
            """SELECT DISTINCT s.city, s.event_date,
                                s.settlement_station,
                                s.forecast_high_c,
                                s.forecast_peak_hour,
                                s.observed_max_c,
                                s.current_hour_local
               FROM paper_predictor_signals s
               WHERE s.event_date = ?
                 AND s.scanned_at_utc = (
                   SELECT MAX(scanned_at_utc) FROM paper_predictor_signals
                   WHERE city = s.city AND event_date = ?
                 )""",
            (today, today),
        ).fetchall()
    except sqlite3.OperationalError as e:
        log.warning(f"boundary watcher: event discovery failed: {e}")
        return 0

    n_written = 0
    for ev in events:
        city, event_date, icao, fc_high_c, fc_peak_hour, obs_max_c, cur_hour = ev
        if fc_high_c is None or fc_peak_hour is None:
            continue
        try:
            row_id = evaluate_one_event(
                conn               = conn,
                city               = city,
                event_date         = event_date,
                icao               = icao or "",
                tz_str             = "",   # not needed at this layer
                forecast_high_c    = float(fc_high_c),
                forecast_peak_hour = int(fc_peak_hour),
                current_local_hour = int(cur_hour) if cur_hour is not None else 0,
                observed_max_c     = float(obs_max_c) if obs_max_c is not None else None,
            )
            if row_id:
                n_written += 1
        except Exception as e:
            log.warning(f"boundary watcher: {city} {event_date} eval raised: {e}")
            continue
    return n_written


def predictor_holds_bin_on_event(conn: sqlite3.Connection,
                                     city: str, event_date: str) -> bool:
    """True if there's at least one OPEN or CLOSED predictor position
    on this event.  Used to populate the conflict_with_predictor flag
    on every trigger log row.

    In live mode (post dry-run sign-off), this DOES NOT block firing —
    the whole point of Option B is to allow boundary to add a second
    bin alongside predictor's stale one.  But we record the flag so
    the dry-run lookahead lets us measure how often the conflict case
    is the profitable case.
    """
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM positions "
            "WHERE city = ? AND date = ? "
            "  AND COALESCE(is_paper, 0) = 0 "
            "  AND (signal_origin IS NULL "
            "       OR signal_origin NOT LIKE 'boundary%') "
            "  AND status IN ('open', 'closed')",
            (city, event_date),
        ).fetchone()[0]
        return int(n or 0) > 0
    except sqlite3.OperationalError:
        return False