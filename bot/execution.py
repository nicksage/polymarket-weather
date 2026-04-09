import os
import logging
from datetime import datetime
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON
from config import PAPER_TRADE
from db import insert_signal  # reuse for logging paper trades

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = POLYGON  # 137

def get_clob_client() -> ClobClient | None:
    """
    Initialize and return an authenticated CLOB client.

    Requires POLYMARKET_PRIVATE_KEY in environment.
    Returns None in paper trading mode.
    """
    if PAPER_TRADE:
        logger.info("PAPER_TRADE=True — CLOB client not initialized")
        return None
    
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        raise ValueError("POLYMARKET_PRIVATE_KEY not set in environment")

    client = ClobClient(
        host=CLOB_HOST,
        key=private_key,
        chain_id=CHAIN_ID,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def execute_signal(signal: dict, client: ClobClient | None = None) -> dict:
    """
    Execute a trade signal on Polymarket, or log it in paper trading mode.

    Args:
        signal: Signal dict from edge.py (must have contract_id, recommended_side,
                kelly_size, market_p, yes_token_id, no_token_id)
        client: Authenticated ClobClient (None in paper mode)

        Returns:
            Execution result dict with status, order_id (or "paper"), and details
    """
    contract_id = signal["contract_id"]
    side = signal["recommended_side"]
    size_usdc = signal["kelly_size"]

    # Determine token ID and price based on side
    if side == "YES":
        token_id = signal.get("yes_token_id")
        entry_price = signal["market_p"]
    else:
        token_id = signal.get("no_token_id")
        entry_price = 1 - signal["market_p"]
    if not token_id:
        logger.error(f"No token_id for {contract_id} side={side}")
        return {"status": "error", "reason": "missing_token_id"}
  
    # Apply a small buffer to improve fill probability
    # Buy slightly above mid to get fills (0.5% buffer)
    limit_price = round(min(entry_price * 1.005, 0.99), 4)

    result = {
        "contract_id": contract_id,
        "side": side,
        "size_usdc": size_usdc,
        "limit_price": limit_price,
        "entry_time": datetime.utcnow().isoformat(),
        "paper_trade": PAPER_TRADE,
    }

    if PAPER_TRADE or client is None:
        # Paper trading mode: log the signal without executing
        logger.info(
            f"[PAPER] Would execute: {side} ${size_usdc:.2f} on {contract_id[:12]} "
            f"@ {limit_price:.4f}"
        )
        result.update({"status": "paper", "order_id": None})
        return result

    # Live execution
    try:
        order_args = OrderArgs(
            price=limit_price,
            size=size_usdc / limit_price,  # Convert USDC to shares
            side=side,
            token_id=token_id,
        )

        response = client.create_and_post_order(order_args)

        if response and response.get("success"):
            order_id = response.get("orderID")
            logger.info(
                f"Order placed: {side} ${size_usdc:.2f} on {contract_id[:12]} "
                f"@ {limit_price:.4f} | order_id={order_id}"
            )
            result.update({"status": "placed", "order_id": order_id})
        else:
            logger.error(f"Order failed: {response}")
            result.update({"status": "failed", "response": str(response)})

    except Exception as e:
        logger.exception(f"Execution error for {contract_id}: {e}")
        result.update({"status": "error", "reason": str(e)})
    
    return result


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
    """Get all open orders for monitoring."""
    if PAPER_TRADE:
        return []
    try:
        return client.get_orders() or []
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []