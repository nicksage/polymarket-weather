"""
price_ws.py — Real-time price streaming via Polymarket WebSocket.

Maintains a persistent WebSocket connection to the Polymarket CLOB market
channel, receiving live price updates for all open position token IDs.
Provides:

  1. A live price cache: token_id -> latest price (thread-safe)
  2. Real-time stop-loss monitoring: checks every price update against
     open positions and queues exit actions when thresholds are breached
  3. Dynamic subscription management: add/remove tokens as positions
     open and close without reconnecting

The WebSocket runs in a background thread managed by start_price_stream()
and stop_price_stream().  The main bot scheduler and monitor loop can
read prices from get_live_price() at any time.

Connection: wss://ws-subscriptions-clob.polymarket.com/ws/market
Authentication: none required (public channel)
Heartbeat: PING every 10 seconds
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL = 10  # seconds
RECONNECT_DELAY = 5  # seconds between reconnection attempts
MAX_RECONNECT_DELAY = 60

# ---------------------------------------------------------------------------
# Thread-safe price cache
# ---------------------------------------------------------------------------

_price_lock = threading.Lock()
_live_prices: dict[str, float] = {}      # token_id -> latest price
_price_timestamps: dict[str, float] = {} # token_id -> unix timestamp of update

# Subscribed token IDs (managed by add/remove functions)
_subscribed_tokens: set[str] = set()
_sub_lock = threading.Lock()

# Callback for stop-loss checks (set by wire_stop_loss_callback)
_stop_loss_callback: Callable[[str, float], None] | None = None

# Background thread handle
_ws_thread: threading.Thread | None = None
_ws_stop_event = threading.Event()


def get_live_price(token_id: str) -> float | None:
    """Get the latest price for a token.  Returns None if no price received yet."""
    with _price_lock:
        return _live_prices.get(token_id)


def get_live_prices() -> dict[str, float]:
    """Get a snapshot of all live prices."""
    with _price_lock:
        return dict(_live_prices)


def get_price_age(token_id: str) -> float | None:
    """Seconds since the last price update for a token.  None if never received."""
    with _price_lock:
        ts = _price_timestamps.get(token_id)
        if ts is None:
            return None
        return time.time() - ts


def add_tokens(token_ids: list[str]) -> None:
    """Add token IDs to the subscription set.  The WebSocket loop will
    pick these up on its next subscription refresh cycle."""
    with _sub_lock:
        _subscribed_tokens.update(tid for tid in token_ids if tid)


def remove_tokens(token_ids: list[str]) -> None:
    """Remove token IDs from the subscription set."""
    with _sub_lock:
        _subscribed_tokens.difference_update(token_ids)
    with _price_lock:
        for tid in token_ids:
            _live_prices.pop(tid, None)
            _price_timestamps.pop(tid, None)


def wire_stop_loss_callback(callback: Callable[[str, float], None]) -> None:
    """Register a callback invoked on every price update.

    callback(token_id: str, price: float) — the caller should check
    whether this price breaches any stop-loss and act accordingly.
    """
    global _stop_loss_callback
    _stop_loss_callback = callback


def _update_price(token_id: str, price: float) -> None:
    """Internal: update the cache and fire the stop-loss callback."""
    with _price_lock:
        _live_prices[token_id] = price
        _price_timestamps[token_id] = time.time()

    if _stop_loss_callback:
        try:
            _stop_loss_callback(token_id, price)
        except Exception as e:
            logger.debug(f"Stop-loss callback error for {token_id[:12]}: {e}")


# ---------------------------------------------------------------------------
# WebSocket connection loop (runs in asyncio inside a background thread)
# ---------------------------------------------------------------------------

async def _ws_loop() -> None:
    """Persistent WebSocket connection with auto-reconnect."""
    import websockets

    reconnect_delay = RECONNECT_DELAY
    _last_subscribed: set[str] = set()

    _last_token_refresh = 0

    while not _ws_stop_event.is_set():
        # Periodically refresh token list from DB (every 60s)
        if time.time() - _last_token_refresh > 60:
            try:
                load_open_position_tokens()
                load_all_event_tokens()
                _last_token_refresh = time.time()
            except Exception:
                pass

        try:
            with _sub_lock:
                current_tokens = set(_subscribed_tokens)

            if not current_tokens:
                logger.debug("No tokens to subscribe — waiting 10s")
                await asyncio.sleep(10)
                continue

            logger.info(
                f"[WS] Connecting to {WS_URL} with {len(current_tokens)} token(s)..."
            )

            async with websockets.connect(
                WS_URL,
                ping_interval=None,  # we handle pings manually
                close_timeout=5,
            ) as ws:
                # Subscribe
                sub_msg = json.dumps({
                    "assets_ids": list(current_tokens),
                    "type": "market",
                    "custom_feature_enabled": True,
                })
                await ws.send(sub_msg)
                _last_subscribed = set(current_tokens)
                reconnect_delay = RECONNECT_DELAY

                logger.info(f"[WS] Connected and subscribed to {len(current_tokens)} token(s)")

                last_ping = time.time()

                while not _ws_stop_event.is_set():
                    # Send PING every 10 seconds
                    if time.time() - last_ping >= PING_INTERVAL:
                        try:
                            await ws.send("PING")
                            last_ping = time.time()
                        except Exception:
                            break

                    # Check for subscription changes
                    with _sub_lock:
                        current_tokens = set(_subscribed_tokens)
                    new_tokens = current_tokens - _last_subscribed
                    removed_tokens = _last_subscribed - current_tokens

                    if new_tokens:
                        try:
                            await ws.send(json.dumps({
                                "assets_ids": list(new_tokens),
                                "operation": "subscribe",
                                "custom_feature_enabled": True,
                            }))
                            _last_subscribed.update(new_tokens)
                            logger.info(f"[WS] Subscribed to {len(new_tokens)} new token(s)")
                        except Exception:
                            break

                    if removed_tokens:
                        try:
                            await ws.send(json.dumps({
                                "assets_ids": list(removed_tokens),
                                "operation": "unsubscribe",
                            }))
                            _last_subscribed -= removed_tokens
                            logger.debug(f"[WS] Unsubscribed {len(removed_tokens)} token(s)")
                        except Exception:
                            break

                    # Receive messages with timeout
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

                    if raw == "PONG":
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    # API may send a single dict or a list of dicts
                    if isinstance(msg, list):
                        for m in msg:
                            if isinstance(m, dict):
                                _process_message(m)
                    elif isinstance(msg, dict):
                        _process_message(msg)

        except Exception as e:
            if _ws_stop_event.is_set():
                break
            logger.warning(f"[WS] Connection error: {e} — reconnecting in {reconnect_delay}s")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)

    logger.info("[WS] WebSocket loop stopped")


def _process_message(msg: dict) -> None:
    """Extract price from various message types."""
    event_type = msg.get("event_type")

    if event_type == "last_trade_price":
        # {"event_type": "last_trade_price", "asset_id": "...", "price": "0.45", ...}
        token_id = msg.get("asset_id")
        price_str = msg.get("price")
        if token_id and price_str:
            try:
                _update_price(token_id, float(price_str))
            except (ValueError, TypeError):
                pass

    elif event_type == "book":
        # Full orderbook snapshot — extract best bid as current price
        token_id = msg.get("asset_id")
        bids = msg.get("bids") or []
        if token_id and bids:
            try:
                best_bid = max(bids, key=lambda b: float(b.get("price", 0)))
                _update_price(token_id, float(best_bid["price"]))
            except (ValueError, TypeError, KeyError):
                pass

    elif event_type == "price_change":
        token_id = msg.get("asset_id")
        price_str = msg.get("price")
        if token_id and price_str:
            try:
                _update_price(token_id, float(price_str))
            except (ValueError, TypeError):
                pass


# ---------------------------------------------------------------------------
# Background thread management
# ---------------------------------------------------------------------------

def _run_async_loop() -> None:
    """Entry point for the background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_loop())
    finally:
        loop.close()


def start_price_stream() -> None:
    """Start the WebSocket price stream in a background daemon thread."""
    global _ws_thread
    if _ws_thread is not None and _ws_thread.is_alive():
        logger.debug("[WS] Price stream already running")
        return

    _ws_stop_event.clear()
    _ws_thread = threading.Thread(
        target=_run_async_loop,
        name="price-ws",
        daemon=True,
    )
    _ws_thread.start()
    logger.info("[WS] Price stream background thread started")


def stop_price_stream() -> None:
    """Signal the WebSocket loop to stop."""
    _ws_stop_event.set()
    logger.info("[WS] Price stream stop requested")


def load_open_position_tokens() -> None:
    """Load all token IDs from open positions and subscribe to them."""
    try:
        from db import get_open_positions
        positions = get_open_positions()
        token_ids = []
        for p in positions:
            if p.get("fill_status") != "filled":
                continue
            side = p.get("side", "YES")
            if side == "YES" and p.get("yes_token_id"):
                token_ids.append(p["yes_token_id"])
            elif side == "NO" and p.get("no_token_id"):
                token_ids.append(p["no_token_id"])
        if token_ids:
            add_tokens(token_ids)
            logger.info(f"[WS] Loaded {len(token_ids)} token IDs from {len(positions)} open positions")
        else:
            logger.info(f"[WS] No token IDs found in {len(positions)} open positions")
    except Exception as e:
        logger.error(f"[WS] Failed to load position tokens: {e}", exc_info=True)


# Token -> (event_id, contract_id, city, date) mapping for snapshot writes
_token_metadata: dict[str, dict] = {}
_token_meta_lock = threading.Lock()


def load_all_event_tokens() -> None:
    """Subscribe to ALL active event bin tokens for market-wide price capture."""
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT o.yes_token_id, o.no_token_id, o.contract_id,
                   e.event_id, e.city, e.date
            FROM temp_outcomes o
            JOIN temp_events e ON o.event_row_id = e.id
            INNER JOIN (
                SELECT event_id, MAX(scan_timestamp) AS max_ts
                FROM temp_events GROUP BY event_id
            ) latest ON e.event_id = latest.event_id
                    AND e.scan_timestamp = latest.max_ts
            WHERE e.date >= date('now')
              AND o.yes_token_id IS NOT NULL
        """).fetchall()
        conn.close()

        token_ids = []
        with _token_meta_lock:
            for r in rows:
                yes_tid = r["yes_token_id"]
                if yes_tid:
                    token_ids.append(yes_tid)
                    _token_metadata[yes_tid] = {
                        "event_id": r["event_id"],
                        "contract_id": r["contract_id"],
                        "city": r["city"],
                        "date": r["date"],
                    }
        if token_ids:
            add_tokens(token_ids)
            logger.info(f"[WS] Subscribed to {len(token_ids)} tokens across all active events")
    except Exception as e:
        logger.debug(f"[WS] Failed to load all event tokens: {e}")


def write_price_snapshots() -> int:
    """Write current live prices to bin_price_history for backtesting.
    Called periodically (every 5-10 min) by the scheduler."""
    from db import insert_bin_price_snapshots_bulk
    from datetime import datetime, timezone

    with _price_lock:
        prices = dict(_live_prices)

    with _token_meta_lock:
        meta = dict(_token_metadata)

    if not prices:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for token_id, price in prices.items():
        m = meta.get(token_id)
        if not m:
            continue
        rows.append({
            "event_id": m["event_id"],
            "contract_id": m["contract_id"],
            "city": m.get("city"),
            "date": m.get("date"),
            "yes_price": price,
            "no_price": round(1.0 - price, 4) if price is not None else None,
            "volume_usd": None,
            "liquidity_usd": None,
        })

    if not rows:
        return 0

    try:
        inserted = insert_bin_price_snapshots_bulk(rows, now)
        logger.debug(f"[WS] Wrote {inserted} price snapshots to bin_price_history")
        return inserted
    except Exception as e:
        logger.debug(f"[WS] Price snapshot write failed: {e}")
        return 0
