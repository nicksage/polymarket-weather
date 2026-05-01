"""
heartbeat.py — Polymarket maker-side keep-alive thread.

Polymarket auto-cancels every one of your open orders if it doesn't
receive a heartbeat within ~15 seconds (10s nominal + 5s buffer).  See:
    https://docs.polymarket.com/trading/orders/cancel
    https://docs.polymarket.com/trading/orders/create

Without this module, the bot's orders silently disappear during:
  * restarts (each one ≥15s of downtime → all open orders cancelled)
  * brief WS hiccups
  * any operation that blocks the main thread for >15s

The cancellation is permanent; the order itself is gone (see
docstring of monitor.detect_externally_cancelled_topups for the
recovery flow once the safety-net poll detects the orphan pointer).

Why a dedicated daemon thread (vs. APScheduler vs. user_ws coroutine):
  * APScheduler at interval=5s: works but adds a sub-minute job to a
    scheduler designed for minute-cron-style work; misfire handling
    becomes thorny on small intervals.
  * async inside user_ws: heartbeat dies with user_ws — exactly when
    we'd most want it alive to keep orders safe through the gap.
  * Dedicated daemon thread: fully independent of every other component.
    APScheduler crash, user_ws disconnect, monitor cycle hang — all
    survivable.  Process exit kills the daemon cleanly (no thread join).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Heartbeat period.  Polymarket allows up to 15s between heartbeats
# (10s nominal timeout + 5s buffer per their docs).  5s gives us 3x
# headroom — even one missed call is fine, two in a row is ~10s and
# still safe, only three in a row trips the cancel.
HEARTBEAT_INTERVAL_SEC = 5.0

# Threshold for warn-level logging on consecutive failures.  Polymarket's
# cancel trigger is ~3 missed (15s) — log loudly at 3 because by then
# our orders may already be gone.
WARN_AFTER_CONSEC_FAILURES = 3

# Lock + state — protected so the dashboard's get_heartbeat_state read
# is safe across threads.
_state_lock = threading.Lock()
_state: dict = {
    "running":              False,
    "last_success_ts":      None,    # epoch seconds of last 200 OK
    "last_failure_ts":      None,    # epoch seconds of last failure
    "last_error":           None,    # str(exception) of last failure
    "consecutive_failures": 0,
    "total_sent":           0,
    "total_failed":         0,
}

_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def get_heartbeat_state() -> dict:
    """Snapshot of the heartbeat daemon's health.  Used by the dashboard
    health strip to surface 'orders may be at risk' to the operator."""
    with _state_lock:
        return dict(_state)


def _heartbeat_loop(client) -> None:
    """Send post_heartbeat every HEARTBEAT_INTERVAL_SEC until stop_event.

    Polymarket's post_heartbeat returns a heartbeat_id in the response;
    pass it back next call so the server can detect dropped sequences.
    The first call sends an empty id (per Polymarket SDK convention).
    """
    last_id = ""
    logger.info(
        f"[HEARTBEAT] loop started, interval={HEARTBEAT_INTERVAL_SEC}s"
    )
    while not _stop_event.is_set():
        try:
            resp = client.post_heartbeat(heartbeat_id=last_id)
            if isinstance(resp, dict):
                new_id = resp.get("heartbeat_id") or resp.get("id") or ""
                if new_id:
                    last_id = str(new_id)
            with _state_lock:
                _state["last_success_ts"]      = time.time()
                _state["consecutive_failures"] = 0
                _state["total_sent"]          += 1
                _state["last_error"]           = None
        except Exception as e:
            with _state_lock:
                _state["last_failure_ts"]      = time.time()
                _state["consecutive_failures"] += 1
                _state["total_failed"]        += 1
                _state["last_error"]           = str(e)[:200]
                consec = _state["consecutive_failures"]
            if consec >= WARN_AFTER_CONSEC_FAILURES:
                # Loud warning — at this point Polymarket's auto-cancel
                # has already fired and we'll be losing orders.  The
                # safety-net poll (monitor.detect_externally_cancelled_topups)
                # will clean up stale pointers, then the next trading
                # cycle re-issues — but the caller should investigate.
                logger.warning(
                    f"[HEARTBEAT] {consec} consecutive failures "
                    f"(~{consec * HEARTBEAT_INTERVAL_SEC:.0f}s of silence) — "
                    f"open orders likely auto-cancelled.  "
                    f"Last error: {e}"
                )
            else:
                logger.debug(
                    f"[HEARTBEAT] attempt failed (will retry): {e}"
                )
        # Interruptible sleep — stop_heartbeat() fires the event and we
        # exit within at most one HEARTBEAT_INTERVAL_SEC.
        _stop_event.wait(HEARTBEAT_INTERVAL_SEC)
    logger.info("[HEARTBEAT] loop exited cleanly")
    with _state_lock:
        _state["running"] = False


def start_heartbeat(client) -> None:
    """Spawn the heartbeat daemon thread.  Idempotent — a second call
    while already running is a no-op (logged at DEBUG)."""
    global _thread
    if client is None:
        logger.info("[HEARTBEAT] no client — heartbeat not started")
        return
    with _state_lock:
        if _state["running"]:
            logger.debug("[HEARTBEAT] start requested but already running")
            return
        _state["running"] = True
    _stop_event.clear()
    _thread = threading.Thread(
        target=_heartbeat_loop,
        args=(client,),
        name="polymarket-heartbeat",
        daemon=True,
    )
    _thread.start()


def stop_heartbeat(timeout: float = 6.0) -> None:
    """Signal the daemon to exit.  Blocks until the thread has exited
    (or `timeout` elapses).  Safe to call multiple times.

    Default timeout 6s = one heartbeat interval + 1s buffer, so a clean
    shutdown waits at most one in-flight send to complete.
    """
    _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=timeout)
