"""
user_ws.py — Real-time order/trade event streaming via Polymarket's
authenticated user-channel WebSocket.

Maintains a persistent connection to wss://ws-subscriptions-clob.polymarket.com/ws/user,
subscribed to every condition_id (Gamma market id) we have an open or
in-flight order on.  Events are routed through `fill_handler.apply_*`,
which is the SAME write path used by monitor.py's REST fallback — so the
DB sees identical writes regardless of source and the two paths can't
diverge.

Why a separate module from price_ws.py
--------------------------------------
* Different endpoint (/ws/user vs /ws/market)
* Different auth model (user requires apiKey/secret/passphrase, market is public)
* Different filter shape (`markets` vs `assets_ids`)
* Different event semantics (fills are state-changing, prices are not)

Lifecycle handling
------------------
Polymarket trades progress MATCHED → MINED → CONFIRMED.  We act on
CONFIRMED.  The handler enforces monotonic transitions, so duplicate or
late events become no-ops at the DB layer — this module just dispatches.

Reconnect
---------
There is no replay buffer.  After every successful (re)connect we trigger
a single REST reconciliation pass to backfill any events missed during
downtime.  This makes the WS-vs-REST relationship one of "WS is primary,
REST closes any gap on reconnect + as a periodic backstop."
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
PING_INTERVAL = 10
RECONNECT_DELAY = 5
MAX_RECONNECT_DELAY = 60
SUBSCRIPTION_REFRESH_SEC = 60
RECV_TIMEOUT_SEC = 15


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_ws_thread: threading.Thread | None = None
_ws_stop_event = threading.Event()

# Auth tuple (apiKey, secret, passphrase) — set by start_user_stream()
_auth: tuple[str, str, str] | None = None

# Wallet address (lowercased) for maker/taker disambiguation
_my_wallet: str | None = None

# Subscribed condition_ids — set by `load_open_market_ids` from DB
_subscribed_markets: set[str] = set()
_sub_lock = threading.Lock()

# Backfill callback — called after every successful (re)connect to
# pull anything missed during downtime through REST.
_backfill_callback: callable | None = None


def add_markets(market_ids: list[str]) -> None:
    """Add condition_ids to the subscription set.  The WS loop picks
    these up on the next subscription refresh cycle (or immediately on
    reconnect)."""
    with _sub_lock:
        _subscribed_markets.update(m for m in market_ids if m)


def remove_markets(market_ids: list[str]) -> None:
    with _sub_lock:
        _subscribed_markets.difference_update(market_ids)


def load_open_market_ids() -> None:
    """Load all condition_ids (gamma_market_id) for live (non-paper)
    positions in any non-terminal state, and add them to the subscription
    set.  Called periodically and on reconnect.

    Note: condition_id IS gamma_market_id in our schema — both refer to
    the same on-chain market identifier.
    """
    try:
        from db import get_open_positions, get_pending_positions, get_positions_with_pending_topup
        market_ids: set[str] = set()
        for fn in (get_open_positions, get_pending_positions, get_positions_with_pending_topup):
            for p in fn():
                if bool(p.get("is_paper", 1)):
                    continue
                gid = p.get("gamma_market_id")
                if gid:
                    market_ids.add(gid)
        if market_ids:
            add_markets(list(market_ids))
            logger.debug(f"[USER_WS] Loaded {len(market_ids)} market id(s) from DB")
    except Exception as e:
        logger.debug(f"[USER_WS] load_open_market_ids failed: {e}")


def wire_backfill_callback(cb: callable) -> None:
    """Register a function to call after every successful (re)connect.
    Typically wired to monitor._reconcile_pending_fills so any fills that
    happened during WS downtime get caught up via REST."""
    global _backfill_callback
    _backfill_callback = cb


# ---------------------------------------------------------------------------
# WS loop (background thread)
# ---------------------------------------------------------------------------

async def _build_subscribe_payload(markets: set[str]) -> str:
    """Build the initial auth+subscribe frame.  `markets` is required by
    the user channel and filters events to those condition_ids."""
    if _auth is None:
        raise RuntimeError("user_ws: _auth not set — call start_user_stream first")
    api_key, secret, passphrase = _auth
    return json.dumps({
        "auth": {
            "apiKey":     api_key,
            "secret":     secret,
            "passphrase": passphrase,
        },
        "markets": sorted(markets),
        "type":    "user",
    })


async def _ws_loop() -> None:
    """Persistent connection with auto-reconnect + REST backfill on each
    successful (re)connect."""
    import websockets

    reconnect_delay = RECONNECT_DELAY
    last_refresh = 0.0

    while not _ws_stop_event.is_set():
        # Refresh subscription set from DB periodically.  Keeps the WS
        # subscribed even after restart without requiring the trader
        # path to also touch this module.
        if time.time() - last_refresh > SUBSCRIPTION_REFRESH_SEC:
            try:
                load_open_market_ids()
            except Exception:
                pass
            last_refresh = time.time()

        with _sub_lock:
            current_markets = set(_subscribed_markets)

        if not current_markets:
            logger.debug("[USER_WS] No markets to subscribe — sleeping 10s")
            await asyncio.sleep(10)
            continue

        try:
            logger.info(
                f"[USER_WS] Connecting to {WS_URL} with {len(current_markets)} market(s)"
            )
            async with websockets.connect(
                WS_URL,
                ping_interval=None,  # we send PING manually
                close_timeout=5,
            ) as ws:
                # Auth + subscribe in the FIRST frame
                sub_msg = await _build_subscribe_payload(current_markets)
                await ws.send(sub_msg)
                try:
                    from activity import log_activity
                    log_activity(
                        "WS",
                        message=(
                            f"user-channel WS connected; subscribed to "
                            f"{len(current_markets)} market(s)"
                        ),
                        markets_count=len(current_markets),
                    )
                except Exception:
                    logger.info(
                        f"[USER_WS] Connected; subscribed to {len(current_markets)} market(s)"
                    )
                reconnect_delay = RECONNECT_DELAY

                # Trigger REST backfill — anything that filled during the
                # downtime gap won't replay over the WS, so we need a sweep.
                if _backfill_callback is not None:
                    try:
                        _backfill_callback()
                    except Exception as e:
                        logger.warning(f"[USER_WS] backfill_callback failed: {e}")

                last_ping = time.time()

                while not _ws_stop_event.is_set():
                    # Heartbeat
                    if time.time() - last_ping >= PING_INTERVAL:
                        try:
                            await ws.send("PING")
                            last_ping = time.time()
                        except Exception:
                            break

                    # Add new market subscriptions mid-stream (best effort —
                    # community reports unsubscribe is unreliable, so we
                    # only ever add).
                    with _sub_lock:
                        wanted = set(_subscribed_markets)
                    new_markets = wanted - current_markets
                    if new_markets:
                        try:
                            await ws.send(json.dumps({
                                "operation": "subscribe",
                                "markets":   sorted(new_markets),
                                "type":      "user",
                            }))
                            current_markets |= new_markets
                            logger.info(
                                f"[USER_WS] Added {len(new_markets)} new market subscription(s)"
                            )
                        except Exception:
                            break

                    # Receive
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SEC)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

                    if raw == "PONG":
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug(f"[USER_WS] Non-JSON message dropped: {raw[:80]!r}")
                        continue

                    if isinstance(msg, list):
                        for m in msg:
                            if isinstance(m, dict):
                                _dispatch(m)
                    elif isinstance(msg, dict):
                        _dispatch(msg)

        except Exception as e:
            if _ws_stop_event.is_set():
                break
            try:
                from activity import log_activity
                log_activity(
                    "WS", level="WARN",
                    message=(
                        f"user-channel WS disconnected: {e} "
                        f"— reconnecting in {reconnect_delay}s"
                    ),
                    error=str(e), reconnect_delay=reconnect_delay,
                )
            except Exception:
                logger.warning(
                    f"[USER_WS] Connection error: {e} — reconnecting in {reconnect_delay}s"
                )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)

    logger.info("[USER_WS] Loop stopped")


def _dispatch(event: dict) -> None:
    """Route an event to the right fill_handler entry point."""
    from fill_handler import apply_trade_event, apply_order_event

    et = (event.get("event_type") or event.get("eventType") or "").lower()
    try:
        if et == "trade":
            result = apply_trade_event(event, my_wallet=_my_wallet)
            action = result.get("action")
            if action in ("filled", "failed"):
                logger.info(f"[USER_WS] trade dispatched → {result}")
            elif action and action.startswith("ignored_"):
                logger.debug(f"[USER_WS] trade {action}: {result}")
        elif et == "order":
            result = apply_order_event(event)
            action = result.get("action")
            if action in ("cancelled",):
                logger.info(f"[USER_WS] order dispatched → {result}")
            else:
                logger.debug(f"[USER_WS] order {action}: {result}")
        else:
            logger.debug(f"[USER_WS] Unknown event_type={et!r} — ignored")
    except Exception as e:
        logger.exception(f"[USER_WS] dispatch crashed on event: {e}")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _run_async_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_loop())
    finally:
        loop.close()


def start_user_stream(client: Any, wallet_address: str | None = None) -> None:
    """Start the authenticated user-channel WS in a background daemon thread.

    `client` must be an authenticated py_clob_client ClobClient — we read
    its `creds.api_key/api_secret/api_passphrase` to build the auth frame.

    `wallet_address` (optional) is used by the handler to disambiguate
    maker vs taker order ids on trade events.  If omitted we fall back
    to taker_order_id only.
    """
    global _ws_thread, _auth, _my_wallet

    if _ws_thread is not None and _ws_thread.is_alive():
        logger.debug("[USER_WS] Already running")
        return

    creds = getattr(client, "creds", None)
    if creds is None:
        raise RuntimeError("user_ws: client has no .creds — call set_api_creds first")
    _auth = (
        getattr(creds, "api_key", "") or "",
        getattr(creds, "api_secret", "") or "",
        getattr(creds, "api_passphrase", "") or "",
    )
    if not all(_auth):
        raise RuntimeError("user_ws: incomplete API creds on client")

    _my_wallet = (wallet_address or "").lower() or None

    # Pre-load subscriptions so the first connect attempt has work to do
    load_open_market_ids()

    _ws_stop_event.clear()
    _ws_thread = threading.Thread(
        target = _run_async_loop,
        name   = "user-ws",
        daemon = True,
    )
    _ws_thread.start()
    logger.info("[USER_WS] Background thread started")


def stop_user_stream() -> None:
    _ws_stop_event.set()
    logger.info("[USER_WS] Stop requested")


def is_running() -> bool:
    return _ws_thread is not None and _ws_thread.is_alive()
