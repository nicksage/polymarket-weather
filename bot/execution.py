"""
execution.py — Trade execution and position logging.

Handles both live (CLOB) and paper trading paths.  Every attempted trade —
whether placed on-chain or simulated — is written to the positions table so
that risk checks, P&L tracking, and the monitor loop all have a consistent
source of truth.

Live order status values returned by Polymarket CLOB:
    "matched"   — fully or partially filled immediately → log as fill_status='filled'
    "live"      — resting on the book (GTC) → log as fill_status='pending'
    "unmatched" — marketable order, no fill after delay → do not log, treat as failed
    "delayed"   — sports-market delay; treated same as 'live' here
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON

from config import PAPER_TRADE
from db import insert_position, mark_outcome_executed

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = POLYGON  # 137


def get_clob_client() -> ClobClient | None:
    """
    Initialize and return an authenticated CLOB client.
    Returns None in paper trading mode.
    """
    if PAPER_TRADE:
        logger.info("PAPER_TRADE=True — CLOB client not initialized")
        return None

    import os
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        raise ValueError("POLYMARKET_PRIVATE_KEY not set in environment")

    client = ClobClient(host=CLOB_HOST, key=private_key, chain_id=CHAIN_ID)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def execute_signal(signal: dict, client: ClobClient | None = None) -> dict:
    """
    Execute a trade signal on Polymarket, or simulate it in paper trading mode.

    For paper trades  → position is logged immediately with fill_status='filled'
                        using the current market price as entry price.
    For live trades   → order is placed via CLOB; position is logged with
                        fill_status='filled' (if matched immediately) or
                        fill_status='pending' (if resting on book).
                        The monitor loop later confirms pending fills or cancels
                        orders that remain unfilled.

    Returns a result dict with: status, order_id, position_id, and details.
    """
    contract_id = signal["contract_id"]
    side        = signal["recommended_side"]
    size_usdc   = signal["kelly_size"]
    # Store entry_time in local US Central (CST/CDT auto-handled by ZoneInfo).
    # The ISO string carries an explicit offset (e.g. -05:00) so the instant
    # remains unambiguous.
    entry_time  = datetime.now(ZoneInfo("America/Chicago")).isoformat()

    # Determine token ID and price based on side
    if side == "YES":
        token_id    = signal.get("yes_token_id")
        entry_price = float(signal.get("market_p") or signal.get("yes_price") or signal.get("market_price", 0))
    else:
        token_id    = signal.get("no_token_id")
        entry_price = float(1.0 - (signal.get("market_p") or signal.get("yes_price") or signal.get("market_price", 0)))

    if not token_id:
        logger.error(f"No token_id for {contract_id} side={side}")
        return {"status": "error", "reason": "missing_token_id"}

    # Shares = dollars risked / price per share
    shares = round(size_usdc / entry_price, 4) if entry_price > 0 else 0

    # Common position metadata drawn from the signal
    position_kwargs = dict(
        contract_id     = contract_id,
        side            = side,
        size_usdc       = size_usdc,
        entry_price     = entry_price,
        entry_time      = entry_time,
        question        = signal.get("question"),
        city            = signal.get("city"),
        date            = signal.get("date"),
        event_id        = signal.get("event_id"),
        model_prob      = signal.get("model_p") or signal.get("model_prob"),
        market_prob     = signal.get("market_p") or signal.get("market_price"),
        ev              = signal.get("ev"),
        edge            = signal.get("edge"),
        shares          = shares,
        scan_timestamp  = signal.get("scan_timestamp"),
        gamma_market_id = signal.get("gamma_market_id"),
        range_low       = signal.get("range_low"),
        range_high      = signal.get("range_high"),
        unit            = signal.get("unit"),
        yes_token_id    = signal.get("yes_token_id"),
        no_token_id     = signal.get("no_token_id"),
        lat              = signal.get("lat"),
        lon              = signal.get("lon"),
        forecast_sigma_c = signal.get("forecast_sigma_c"),
    )

    # -------------------------------------------------------------------------
    # Paper trading path
    # -------------------------------------------------------------------------
    if PAPER_TRADE or client is None:
        position_id = insert_position(
            **position_kwargs,
            order_id    = None,
            is_paper    = 1,
            fill_status = "filled",
        )
        mark_outcome_executed(contract_id, signal.get("scan_timestamp", ""))
        logger.info(
            f"[PAPER] {side} ${size_usdc:.2f} | {contract_id[:12]} "
            f"@ {entry_price:.4f} | {signal.get('city')} {signal.get('date')} "
            f"| pos_id={position_id}"
        )
        return {
            "status":      "paper",
            "order_id":    None,
            "position_id": position_id,
            "entry_price": entry_price,
            "shares":      shares,
        }

    # -------------------------------------------------------------------------
    # Live trading path
    # -------------------------------------------------------------------------
    # Apply a small buffer above mid to improve fill probability
    limit_price = round(min(entry_price * 1.005, 0.99), 4)

    try:
        order_args = OrderArgs(
            price    = limit_price,
            size     = size_usdc / limit_price,
            side     = side,
            token_id = token_id,
        )
        response = client.create_and_post_order(order_args)

        if not response or not response.get("success"):
            logger.error(f"Order placement failed: {response}")
            return {"status": "failed", "response": str(response)}

        order_id      = response.get("orderID", "")
        order_status  = response.get("status", "")

        # Derive fill price from CLOB response amounts when available
        taking = float(response.get("takingAmount", 0) or 0)
        making = float(response.get("makingAmount", 0) or 0)
        if taking > 0 and making > 0:
            fill_price = taking / making
            fill_shares = making
        else:
            fill_price  = limit_price
            fill_shares = shares

        if order_status in ("matched",):
            # Immediate fill — log as confirmed position
            fill_status = "filled"
            actual_entry = fill_price
            actual_shares = fill_shares
            logger.info(
                f"[LIVE FILLED] {side} ${size_usdc:.2f} | {contract_id[:12]} "
                f"@ {actual_entry:.4f} | order_id={order_id[:12]}"
            )
        elif order_status in ("live", "delayed"):
            # Resting on book — log as pending; monitor loop will confirm or cancel
            fill_status   = "pending"
            actual_entry  = limit_price
            actual_shares = shares
            logger.info(
                f"[LIVE PENDING] {side} ${size_usdc:.2f} | {contract_id[:12]} "
                f"@ {limit_price:.4f} | order_id={order_id[:12]} — awaiting fill"
            )
        else:
            # unmatched or unexpected — do not log a position
            logger.warning(
                f"Order returned status='{order_status}' for {contract_id[:12]} — not logging position"
            )
            return {"status": "unmatched", "order_id": order_id, "order_status": order_status}

        position_id = insert_position(
            **position_kwargs,
            order_id    = order_id,
            is_paper    = 0,
            fill_status = fill_status,
            entry_price = actual_entry,
            shares      = actual_shares,
        )
        mark_outcome_executed(contract_id, signal.get("scan_timestamp", ""))

        return {
            "status":       "placed",
            "order_id":     order_id,
            "order_status": order_status,
            "position_id":  position_id,
            "fill_status":  fill_status,
            "entry_price":  actual_entry,
            "shares":       actual_shares,
        }

    except Exception as e:
        logger.exception(f"Execution error for {contract_id}: {e}")
        return {"status": "error", "reason": str(e)}


def cancel_order(order_id: str, client: ClobClient) -> bool:
    """Cancel an open order by ID."""
    if PAPER_TRADE:
        logger.info(f"[PAPER] Would cancel order {order_id}")
        return True
    try:
        resp = client.cancel(order_id)
        return bool(resp)
    except Exception as e:
        logger.error(f"Failed to cancel {order_id}: {e}")
        return False


def get_open_orders(client: ClobClient) -> list[dict]:
    """Get all open orders from CLOB (live mode only)."""
    if PAPER_TRADE:
        return []
    try:
        return client.get_orders() or []
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []
