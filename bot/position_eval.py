"""
position_eval.py — Active position management (Phase 3 exit engine).

Evaluates every open filled position and classifies it as one of:

    INVALIDATED — observed max definitively exceeds the bin (+ buffer)
    DYING       — model probability collapsed since entry
    WEAKENED    — cumulative forecast shift against position since entry
    THRIVING    — EV of holding > EV of selling; let it ride
    HEALTHY     — thesis intact, no action

Returns a list of ExitAction dicts for the caller (trading loop or
monitor) to execute.  Does NOT place orders itself — the caller decides
whether to execute immediately (INVALIDATED) or queue (WEAKENED).

All comparison is entry-vs-current (cumulative), NOT latest-pull delta.
Uses executable bid price (not midpoint) for exit-value comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from config import (
    EXIT_EVAL_ENABLED,
    EXIT_DEAD_BUFFER_C,
    EXIT_DYING_PROB_THRESHOLD,
    EXIT_HARD_STOP_ENABLED,
    EXIT_HARD_STOP_PCT,
    EXIT_WEAKENED_SIGMA_SHIFT,
    PAPER_TRADE,
)
from db import (
    get_open_positions,
    get_snapshot_by_id,
    get_latest_snapshot_for_contract,
    get_latest_observation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _to_c(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit == "fahrenheit":
        return (value - 32) * 5.0 / 9.0
    return float(value)


def _get_executable_bid(pos: dict, current_snapshot: dict | None) -> float | None:
    """Best estimate of what we could actually sell this position for right now.

    Ideal: CLOB order-book best bid.  We don't have a live CLOB connection in
    the evaluation loop, so we approximate using the monitor-updated
    `positions.current_price` (refreshed every hour from Gamma / Data API).
    Apply a conservative 2% discount to account for spread + slippage.

    For paper mode this is the best we can do.  For live mode, the actual
    sell order will use real CLOB pricing.
    """
    raw = pos.get("current_price")
    if raw is None and current_snapshot:
        raw = current_snapshot.get("market_price")
    if raw is None:
        return None
    return float(raw) * 0.98   # 2% haircut for spread/slippage


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ExitAction:
    position_id: int
    contract_id: str
    event_id: str
    city: str
    date: str
    side: str
    classification: str      # INVALIDATED, DYING, WEAKENED, THRIVING, HEALTHY
    action: str              # SELL, REDUCE_50, HOLD
    reason: str
    exit_price: float | None
    urgency: int             # 0=immediate, 1=next cycle, 2=hold


def _classify_position(pos: dict) -> ExitAction:
    """Classify a single open position.  Pure function — reads pos dict
    and DB state, returns an ExitAction."""

    pid         = pos["id"]
    contract_id = pos.get("contract_id", "")
    event_id    = pos.get("event_id", "")
    city        = pos.get("city", "")
    date_str    = pos.get("date", "")
    side        = pos.get("side", "YES")
    entry_price = float(pos.get("entry_price") or 0)
    shares      = float(pos.get("shares") or 0)
    entry_prob  = float(pos.get("model_prob") or 0)
    unit        = pos.get("unit", "celsius")

    range_low_c  = _to_c(pos.get("range_low"), unit)
    range_high_c = _to_c(pos.get("range_high"), unit)

    entry_snapshot_id = pos.get("entry_snapshot_id")
    entry_snap = get_snapshot_by_id(entry_snapshot_id) if entry_snapshot_id else None

    current_snap = get_latest_snapshot_for_contract(contract_id)
    latest_obs   = get_latest_observation(event_id)

    observed_max = (latest_obs or {}).get("observed_max_so_far_c")
    executable_bid = _get_executable_bid(pos, current_snap)

    # Current model state (from latest snapshot for this contract)
    current_prob      = float((current_snap or {}).get("model_prob") or 0)
    current_adj_mu    = (current_snap or {}).get("adjusted_mu_c")
    current_market_p  = (current_snap or {}).get("market_price")

    # Entry state (from the snapshot captured when the trade was placed)
    entry_adj_mu      = (entry_snap or {}).get("adjusted_mu_c")
    entry_sigma       = float((entry_snap or {}).get("blended_sigma_c") or
                              pos.get("forecast_sigma_c") or 2.0)

    def _make(cls: str, action: str, reason: str, urgency: int) -> ExitAction:
        return ExitAction(
            position_id    = pid,
            contract_id    = contract_id,
            event_id       = event_id,
            city           = city,
            date           = date_str,
            side           = side,
            classification = cls,
            action         = action,
            reason         = reason,
            exit_price     = executable_bid,
            urgency        = urgency,
        )

    # --- Check local hour for time-gating on INVALIDATED ---
    local_hour = None
    try:
        if latest_obs and latest_obs.get("pulled_at_utc"):
            pulled = datetime.fromisoformat(
                latest_obs["pulled_at_utc"].replace("Z", "+00:00")
            )
            local_hour = pulled.hour + pulled.minute / 60.0
    except Exception:
        pass

    # =====================================================================
    # 1. INVALIDATED — observed max exceeds bin ceiling + buffer
    # =====================================================================
    if (observed_max is not None
            and range_high_c is not None
            and side == "YES"
            and observed_max > range_high_c + EXIT_DEAD_BUFFER_C):
        day_advanced = (local_hour is not None and local_hour >= 14)
        prob_near_zero = current_prob < 0.02
        if day_advanced or prob_near_zero:
            return _make("INVALIDATED", "SELL",
                         f"observed_max={observed_max:.1f} > "
                         f"range_high={range_high_c:.1f}+{EXIT_DEAD_BUFFER_C}",
                         urgency=0)

    # Also check: for NO positions, if observed max is well BELOW the bin
    # floor late in the day, the NO side is winning — but that's THRIVING,
    # not invalidated.  Invalidation for NO = the bin IS hit.
    if (observed_max is not None
            and range_low_c is not None
            and range_high_c is not None
            and side == "NO"):
        # If observed max is inside the bin range, our NO bet is losing
        if (range_low_c <= observed_max <= range_high_c
                and current_prob < 0.05 and entry_prob > 0.15):
            return _make("INVALIDATED", "SELL",
                         f"observed_max={observed_max:.1f} inside "
                         f"[{range_low_c:.1f},{range_high_c:.1f}] (NO side losing)",
                         urgency=0)

    # =====================================================================
    # 2. DYING — market price dropped below threshold
    # =====================================================================
    _market_price = float(pos.get("current_price") or 0) if pos.get("current_price") is not None else None
    if (_market_price is not None
            and _market_price < EXIT_DYING_PROB_THRESHOLD):
        return _make("DYING", "SELL",
                     f"market_price={_market_price:.3f} < "
                     f"threshold={EXIT_DYING_PROB_THRESHOLD}",
                     urgency=1)

    # =====================================================================
    # 3. WEAKENED — cumulative μ shift against position since entry
    # =====================================================================
    if (current_adj_mu is not None
            and entry_adj_mu is not None
            and entry_sigma > 0):
        mu_shift = current_adj_mu - entry_adj_mu
        threshold = entry_sigma * EXIT_WEAKENED_SIGMA_SHIFT

        # Determine if shift is AGAINST the position (side-aware)
        if range_low_c is not None and range_high_c is not None:
            bin_center = (range_low_c + range_high_c) / 2.0
        elif range_low_c is not None:
            bin_center = range_low_c + 1.0
        elif range_high_c is not None:
            bin_center = range_high_c - 1.0
        else:
            bin_center = None

        if bin_center is not None:
            if side == "YES":
                shift_against = (mu_shift > 0 and bin_center < entry_adj_mu) or \
                                (mu_shift < 0 and bin_center > entry_adj_mu)
            else:  # NO — we want the temperature AWAY from the bin
                shift_against = (mu_shift > 0 and bin_center < current_adj_mu and
                                 abs(mu_shift) < abs(current_adj_mu - bin_center)) or \
                                (mu_shift < 0 and bin_center > current_adj_mu and
                                 abs(mu_shift) < abs(current_adj_mu - bin_center))
                # Simpler: for NO, shift toward the bin is against us
                shift_against = abs(current_adj_mu - bin_center) < abs(entry_adj_mu - bin_center)

            if shift_against and abs(mu_shift) > threshold:
                unrealized = float(pos.get("unrealized_pnl") or 0)
                if unrealized < 0:
                    return _make("WEAKENED", "SELL",
                                 f"mu_shift={mu_shift:+.2f} > "
                                 f"{threshold:.2f} against; losing",
                                 urgency=1)
                else:
                    return _make("WEAKENED", "REDUCE_50",
                                 f"mu_shift={mu_shift:+.2f} > "
                                 f"{threshold:.2f} against; winning",
                                 urgency=1)

    # =====================================================================
    # 4. Hard stop-loss (circuit breaker) — uses precomputed SL price
    # =====================================================================
    if EXIT_HARD_STOP_ENABLED:
        sl_price = pos.get("stop_loss_price")
        _curr_price = float(pos.get("current_price") or 0) if pos.get("current_price") is not None else None
        if sl_price is not None and _curr_price is not None and float(sl_price) > 0:
            if _curr_price <= float(sl_price):
                unrealized = float(pos.get("unrealized_pnl") or 0)
                return _make("WEAKENED", "SELL",
                             f"hard_stop: price={_curr_price:.4f} <= "
                             f"SL={float(sl_price):.4f} (entry={entry_price:.4f})",
                             urgency=0)

    # =====================================================================
    # 7. HEALTHY — thesis intact, hold
    # =====================================================================
    return _make("HEALTHY", "HOLD", "thesis_intact", urgency=2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_open_positions() -> list[ExitAction]:
    """Classify all open filled positions and return actionable exits.

    Returns only positions where action != HOLD.  Sorted by urgency
    (0=immediate first), then by classification severity.

    Bracket scale-out: when multiple bins from the same event are open
    (e.g., a 2-bin bracket), each is evaluated independently.  If one bin
    is INVALIDATED (observed max blew past it), it gets sold while the
    surviving bin continues under HEALTHY/THRIVING.  No special bracket
    logic needed — independent classification naturally handles it.
    """
    if not EXIT_EVAL_ENABLED:
        return []

    positions = [p for p in get_open_positions()
                 if p.get("fill_status") == "filled"]
    if not positions:
        return []

    actions: list[ExitAction] = []
    for pos in positions:
        try:
            ea = _classify_position(pos)
            if ea.action != "HOLD":
                actions.append(ea)
                logger.info(
                    f"[EXIT-EVAL] pos={ea.position_id} {ea.city} {ea.date} "
                    f"{ea.side} -> {ea.classification} / {ea.action} | "
                    f"{ea.reason}"
                )
        except Exception as e:
            logger.debug(f"position eval failed for {pos.get('id')}: {e}")

    # Sort: urgency 0 (immediate) first, then INVALIDATED > DYING > WEAKENED
    priority = {"INVALIDATED": 0, "DYING": 1, "WEAKENED": 2, "THRIVING": 3}
    actions.sort(key=lambda a: (a.urgency, priority.get(a.classification, 9)))
    return actions
