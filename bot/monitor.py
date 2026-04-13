"""
monitor.py — Position monitor loop.

Runs every hour at :30 (staggered from the trading run at :10).
Three sequential responsibilities per run:

  1. Cancel stale pending orders
     Live orders that were placed but not filled before this monitor run
     are cancelled via the CLOB API and logged with a cancellation reason.

  2. Detect resolved markets and close positions
     For every open filled position, check the Gamma API to see if the
     market has closed.  Compute realized P&L and mark the position closed.
     For live positions, cross-check against the Polymarket Data API
     (on-chain source of truth) to verify and enrich fill details.

  3. Update unrealized P&L for open positions
     For paper trades: fetch current YES/NO price from Gamma API.
     For live trades:  use the Polymarket Data API (blockchain authoritative).
     Write updated current_price and unrealized_pnl to DB.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from timezonefinder import TimezoneFinder as _TZF
    _tf = _TZF()
except Exception:
    _tf = None

from config import PAPER_TRADE, WALLET_ADDRESS, MIN_LIQUIDITY_USD
from db import (
    get_open_positions,
    get_pending_positions,
    update_position_fill,
    update_position_outcome,
    update_position_market_price,
    cancel_position,
    backfill_gamma_market_ids,
    backfill_position_coords,
)
from polymarket import get_market_status, get_data_api_positions, search_temp_high_events
from execution import cancel_order, get_clob_client

logger = logging.getLogger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_time_str(lat: float | None, lon: float | None) -> str | None:
    """
    Return the current local time at the given coordinates as 'HH:MM MM-DD-YYYY'.
    Returns None if coordinates are missing or timezone cannot be resolved.
    """
    if lat is None or lon is None or _tf is None:
        return None
    try:
        tz_name = _tf.timezone_at(lat=lat, lng=lon)
        if not tz_name:
            return None
        tz  = ZoneInfo(tz_name)
        now = datetime.now(tz=tz)
        return now.strftime("%H:%M %m-%d-%Y")
    except (ZoneInfoNotFoundError, Exception):
        return None


def _build_data_api_index(wallet_address: str) -> dict[str, dict]:
    """
    Fetch on-chain positions from the Polymarket Data API and index them
    by token_id for O(1) lookup.  Returns {} if wallet address is not set
    or the fetch fails.
    """
    if not wallet_address or PAPER_TRADE:
        return {}
    positions = get_data_api_positions(wallet_address)
    return {p["token_id"]: p for p in positions if p.get("token_id")}


# ---------------------------------------------------------------------------
# Step 1 — Cancel stale pending orders
# ---------------------------------------------------------------------------

def _cancel_pending_orders(client) -> int:
    """
    Cancel any live order that was not filled before this monitor run.
    Paper trades are never pending, so this step is a no-op in paper mode.
    Returns the count of orders cancelled.
    """
    pending = get_pending_positions()
    if not pending:
        return 0

    cancelled = 0
    now = _now()

    for pos in pending:
        pos_id   = pos["id"]
        order_id = pos.get("order_id")
        city     = pos.get("city", "")
        date     = pos.get("date", "")
        contract = pos.get("contract_id", "")[:12]

        if order_id and client:
            success = cancel_order(order_id, client)
            if success:
                logger.info(
                    f"[MONITOR] Cancelled unfilled order: {contract} "
                    f"[{city} {date}] order_id={order_id[:12]} "
                    f"— not filled within monitor window"
                )
            else:
                logger.warning(
                    f"[MONITOR] Failed to cancel order {order_id[:12]} via CLOB — "
                    f"marking cancelled in DB anyway"
                )

        cancel_position(
            position_id      = pos_id,
            cancelled_reason = "unfilled_before_monitor_run",
            exit_time        = now,
        )
        cancelled += 1
        logger.info(
            f"[MONITOR] Position {pos_id} logged as cancelled "
            f"[{city} {date} {contract}]"
        )

    return cancelled


# ---------------------------------------------------------------------------
# Step 2 — Detect resolved markets, close positions, record P&L
# ---------------------------------------------------------------------------

def _settle_resolved_positions(data_api_index: dict) -> int:
    """
    Check every open filled position against the Gamma API.  If the market
    has resolved, compute P&L and mark the position closed.

    For live positions, cross-check with the Data API:
      - If the on-chain position still shows size > 0, the payout hasn't been
        redeemed yet — we still record the resolution P&L from Gamma outcome.
      - If the on-chain position shows cashPnl, use that as the authoritative
        realized P&L figure.

    Returns count of positions closed.
    """
    open_positions = [p for p in get_open_positions() if p.get("fill_status") == "filled"]
    if not open_positions:
        return 0

    closed_count = 0
    now = _now()

    for pos in open_positions:
        pos_id      = pos["id"]
        contract_id = pos.get("contract_id", "")
        side        = pos.get("side", "YES")
        entry_price = float(pos.get("entry_price") or 0)
        shares      = float(pos.get("shares") or 0)
        is_paper    = bool(pos.get("is_paper", 1))
        city        = pos.get("city", "")
        date        = pos.get("date", "")

        gamma_market_id = pos.get("gamma_market_id")
        status = get_market_status(contract_id, gamma_market_id=gamma_market_id)
        if status is None:
            logger.debug(f"[MONITOR] Could not fetch status for {contract_id[:12]}")
            continue

        if not status["closed"]:
            # Market still active — no action needed here
            continue

        winner = status.get("winner")
        if winner is None:
            # Market closed but winner not determinable from price yet
            logger.warning(
                f"[MONITOR] Market {contract_id[:12]} is closed but winner unclear "
                f"(yes={status.get('yes_price')} no={status.get('no_price')}) — skipping"
            )
            continue

        # Determine whether our side won
        our_side_won = (side == winner)
        exit_price   = 1.0 if our_side_won else 0.0

        # Default P&L from our own math
        pnl = round((exit_price - entry_price) * shares, 4)

        # For live positions: prefer on-chain cashPnl from Data API if available
        if not is_paper and data_api_index:
            token_key = pos.get("yes_token_id") if side == "YES" else pos.get("no_token_id")
            on_chain  = data_api_index.get(token_key)
            if on_chain and on_chain.get("cash_pnl") is not None:
                pnl = round(float(on_chain["cash_pnl"]), 4)
                logger.debug(
                    f"[MONITOR] Using Data API cashPnl={pnl} for pos {pos_id} "
                    f"(own calc was {round((exit_price - entry_price) * shares, 4)})"
                )

        update_position_outcome(pos_id, exit_price, now, pnl, "closed")
        closed_count += 1

        result_label = "WON" if our_side_won else "LOST"
        logger.info(
            f"[MONITOR] Position {pos_id} CLOSED — {result_label} | "
            f"{side} {contract_id[:12]} [{city} {date}] "
            f"entry={entry_price:.4f} exit={exit_price:.2f} "
            f"shares={shares:.2f} pnl=${pnl:+.4f}"
        )

    return closed_count


# ---------------------------------------------------------------------------
# Step 3 — Update unrealized P&L for still-open positions
# ---------------------------------------------------------------------------

def _update_unrealized_pnl(data_api_index: dict) -> int:
    """
    For every open filled position, refresh current_price, unrealized_pnl,
    and local_time (current local time at the contract's city).

    Live positions: Data API current_value/size gives current price per share;
                    Gamma API used as fallback.
    Paper positions: Gamma API outcomePrices is the only source.

    market_prob is intentionally NOT updated here — it is captured at entry
    time and kept static so the dashboard shows what the market was pricing
    when the trade was made.

    Returns count of positions updated.
    """
    open_positions = [p for p in get_open_positions() if p.get("fill_status") == "filled"]
    if not open_positions:
        return 0

    updated = 0

    for pos in open_positions:
        pos_id      = pos["id"]
        contract_id = pos.get("contract_id", "")
        side        = pos.get("side", "YES")
        entry_price = float(pos.get("entry_price") or 0)
        shares      = float(pos.get("shares") or 0)
        is_paper    = bool(pos.get("is_paper", 1))
        lat         = pos.get("lat")
        lon         = pos.get("lon")

        current_price = None

        # --- Live: try Data API first ---
        if not is_paper and data_api_index:
            token_key = pos.get("yes_token_id") if side == "YES" else pos.get("no_token_id")
            on_chain  = data_api_index.get(token_key)
            if on_chain:
                size = float(on_chain.get("size") or 0)
                cv   = float(on_chain.get("current_value") or 0)
                if size > 0:
                    current_price = round(cv / size, 4)

        # --- Fallback: Gamma API (also primary for paper trades) ---
        if current_price is None:
            gamma_market_id = pos.get("gamma_market_id")
            status = get_market_status(contract_id, gamma_market_id=gamma_market_id)
            if status and not status.get("closed"):
                current_price = status.get("yes_price") if side == "YES" else status.get("no_price")

        if current_price is None:
            continue

        unrealized_pnl = round((current_price - entry_price) * shares, 4)
        local_time     = _local_time_str(lat, lon)

        update_position_market_price(pos_id, current_price, unrealized_pnl, local_time)
        updated += 1

    return updated


# ---------------------------------------------------------------------------
# Main monitor entry point
# ---------------------------------------------------------------------------

def run_monitor_loop() -> dict:
    """
    Execute the full position monitor cycle.  Called by the scheduler at :30.

    Returns a summary dict for logging.
    """
    logger.info("=== MONITOR RUN START ===")

    # Step 0a — Backfill lat/lon for any positions missing coordinates.
    # Uses the static CITY_COORDS table; safe to call every run (no-op if all filled).
    coords_filled = backfill_position_coords()
    if coords_filled:
        logger.info(f"[MONITOR] Backfilled lat/lon for {coords_filled} position(s)")

    # Step 0b — Backfill gamma_market_id for any open positions missing it.
    # Runs a fresh discovery scan so we get the current numeric Gamma IDs;
    # without these, get_market_status can only do an unreliable page-scan fallback.
    open_missing_gid = [
        p for p in get_open_positions()
        if not p.get("gamma_market_id")
    ]
    if open_missing_gid:
        logger.info(
            f"[MONITOR] {len(open_missing_gid)} open position(s) missing gamma_market_id "
            "— running discovery backfill"
        )
        try:
            events   = search_temp_high_events(min_liquidity=0)
            outcomes = [o for ev in events for o in ev.get("outcomes", [])]
            filled   = backfill_gamma_market_ids(outcomes)
            logger.info(f"[MONITOR] gamma_market_id backfilled for {filled} position(s)")
        except Exception as _e:
            logger.warning(f"[MONITOR] gamma_market_id backfill failed (non-fatal): {_e}")

    # Initialise CLOB client (None in paper mode — only needed for cancellations)
    client = None if PAPER_TRADE else get_clob_client()

    # Fetch on-chain positions once; reused across all three steps
    data_api_index = _build_data_api_index(WALLET_ADDRESS)
    if data_api_index:
        logger.info(f"Data API: {len(data_api_index)} on-chain positions loaded")
    elif not PAPER_TRADE and WALLET_ADDRESS:
        logger.warning("Data API index empty — P&L enrichment will use Gamma API only")

    cancelled = _cancel_pending_orders(client)
    closed    = _settle_resolved_positions(data_api_index)
    updated   = _update_unrealized_pnl(data_api_index)

    summary = {
        "cancelled": cancelled,
        "closed":    closed,
        "updated":   updated,
    }
    logger.info(
        f"[MONITOR] cancelled={cancelled} resolved={closed} pnl_updated={updated}"
    )
    logger.info("=== MONITOR RUN END ===")
    return summary
