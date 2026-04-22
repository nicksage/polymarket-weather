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

from config import PAPER_TRADE, WALLET_ADDRESS, MIN_LIQUIDITY_USD, SUMMARY_LEVEL
from db import (
    get_open_positions,
    get_pending_positions,
    update_position_fill,
    update_position_outcome,
    update_position_market_price,
    update_position_excursions,
    cancel_position,
    backfill_gamma_market_ids,
    backfill_position_coords,
    backfill_position_sigma,
)
from polymarket import get_market_status, get_data_api_positions, search_temp_high_events
from execution import cancel_order, get_clob_client

logger = logging.getLogger(__name__)


def _phase_end():
    logger.log(SUMMARY_LEVEL, "")


DATA_API_BASE = "https://data-api.polymarket.com"


def _scan_event_resolutions() -> int:
    """Check past events for resolved winners using latest price data.
    Records winners in event_resolutions for backtesting any strategy."""
    from db import insert_event_resolution
    import sqlite3 as _sq
    from config import DB_PATH

    try:
        conn = _sq.connect(DB_PATH)
        conn.row_factory = _sq.Row

        # Find past events with a bin at 90%+ (likely resolved)
        # that don't yet have a resolution record
        past_events = conn.execute("""
            SELECT DISTINCT ds.event_id, e.city, e.date
            FROM decision_snapshots ds
            JOIN temp_events e ON ds.event_id = e.event_id
            WHERE e.date < date('now')
              AND ds.market_price >= 0.90
              AND ds.event_id NOT IN (SELECT event_id FROM event_resolutions)
            GROUP BY ds.event_id
            ORDER BY e.date DESC
            LIMIT 50
        """).fetchall()

        found = 0
        for ev in past_events:
            eid, city, date_str = ev["event_id"], ev["city"], ev["date"]

            # Check if any bin hit 90%+ in the latest snapshot (=winner)
            winner = conn.execute("""
                SELECT ds.contract_id, MAX(ds.market_price) as peak
                FROM decision_snapshots ds
                WHERE ds.event_id = ? AND ds.market_price >= 0.90
                GROUP BY ds.contract_id
                ORDER BY peak DESC
                LIMIT 1
            """, (eid,)).fetchone()

            if not winner:
                continue

            # Get range info for the winning bin
            range_info = conn.execute("""
                SELECT range_low, range_high FROM temp_outcomes
                WHERE contract_id = ?
                ORDER BY scan_timestamp DESC LIMIT 1
            """, (winner["contract_id"],)).fetchone()

            rl = float(range_info["range_low"]) if range_info and range_info["range_low"] else None
            rh = float(range_info["range_high"]) if range_info and range_info["range_high"] else None

            now = _now()
            insert_event_resolution(
                event_id=eid, city=city, date=date_str,
                winning_contract_id=winner["contract_id"],
                winning_range_low=rl, winning_range_high=rh,
                winning_yes_price=float(winner["peak"]),
                resolved_at=date_str + "T23:59:59", recorded_at=now,
            )
            found += 1

        conn.close()
        if found:
            logger.info(f"[MONITOR] Recorded {found} event resolution(s) for backtesting")
        return found
    except Exception as e:
        logger.debug(f"Event resolution scan failed: {e}")
        return 0


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

def _settle_resolved_positions(data_api_index: dict,
                               status_cache: dict | None = None) -> int:
    """
    Check every open filled position against the Gamma API.  If the market
    has resolved, compute P&L and mark the position closed.

    status_cache: shared dict of {contract_id: status_dict} to avoid
    duplicate API calls when _update_unrealized_pnl runs right after.

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
        if status_cache is not None and contract_id in status_cache:
            status = status_cache[contract_id]
        else:
            status = get_market_status(contract_id, gamma_market_id=gamma_market_id)
            if status_cache is not None and status is not None:
                status_cache[contract_id] = status
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

        # Record which bin won this event (for backtesting)
        if our_side_won and side == "YES":
            try:
                from db import insert_event_resolution
                insert_event_resolution(
                    event_id=pos.get("event_id", ""),
                    city=city, date=date,
                    winning_contract_id=contract_id,
                    winning_range_low=pos.get("range_low"),
                    winning_range_high=pos.get("range_high"),
                    winning_yes_price=status.get("yes_price"),
                    resolved_at=now, recorded_at=now,
                )
            except Exception:
                pass

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

def _update_unrealized_pnl(data_api_index: dict,
                           status_cache: dict | None = None) -> int:
    """
    For every open filled position, refresh current_price, unrealized_pnl,
    and local_time (current local time at the contract's city).

    status_cache: shared dict of {contract_id: status_dict} populated by
    _settle_resolved_positions — avoids duplicate Gamma API calls.

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

        # --- Try WebSocket live price cache first (sub-second freshness) ---
        try:
            from price_ws import get_live_price, get_price_age
            token_key = pos.get("yes_token_id") if side == "YES" else pos.get("no_token_id")
            if token_key:
                ws_price = get_live_price(token_key)
                ws_age = get_price_age(token_key)
                if ws_price is not None and ws_age is not None and ws_age < 300:
                    current_price = round(ws_price, 4)
        except Exception:
            pass

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
            if status_cache is not None and contract_id in status_cache:
                status = status_cache[contract_id]
            else:
                gamma_market_id = pos.get("gamma_market_id")
                status = get_market_status(contract_id, gamma_market_id=gamma_market_id)
                if status_cache is not None and status is not None:
                    status_cache[contract_id] = status
            if status and not status.get("closed"):
                current_price = status.get("yes_price") if side == "YES" else status.get("no_price")

        if current_price is None:
            continue

        unrealized_pnl = round((current_price - entry_price) * shares, 4)
        local_time     = _local_time_str(lat, lon)

        update_position_market_price(pos_id, current_price, unrealized_pnl, local_time)
        update_position_excursions(pos_id, unrealized_pnl, unrealized_pnl)
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
    logger.log(SUMMARY_LEVEL, "=== MONITOR RUN ===")

    # Step 0a — Backfill lat/lon for any positions missing coordinates.
    # Uses the static CITY_COORDS table; safe to call every run (no-op if all filled).
    coords_filled = backfill_position_coords()
    if coords_filled:
        logger.info(f"[MONITOR] Backfilled lat/lon for {coords_filled} position(s)")

    # Step 0b-σ — Backfill forecast_sigma_c for positions that predate the column.
    sigma_filled = backfill_position_sigma()
    if sigma_filled:
        logger.info(f"[MONITOR] Backfilled forecast_sigma_c for {sigma_filled} position(s)")

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

    # Shared cache so _settle and _update_pnl don't duplicate Gamma API calls
    _status_cache: dict[str, dict] = {}

    cancelled = _cancel_pending_orders(client)
    closed    = _settle_resolved_positions(data_api_index, _status_cache)

    n_open = len([p for p in get_open_positions() if p.get("fill_status") == "filled"])
    if n_open > 0:
        logger.log(SUMMARY_LEVEL,
            f"Updating P&L for {n_open} positions "
            f"({len(_status_cache)} cached from settle phase)..."
        )
    updated   = _update_unrealized_pnl(data_api_index, _status_cache)

    # Scan for resolved events we didn't trade (for backtesting data)
    resolutions_found = _scan_event_resolutions()

    summary = {
        "cancelled": cancelled,
        "closed":    closed,
        "updated":   updated,
        "resolutions": resolutions_found,
    }
    logger.log(SUMMARY_LEVEL,
        f"Monitor complete: {cancelled} cancelled | {closed} resolved | {updated} P&L updated"
    )
    _phase_end()
    return summary
