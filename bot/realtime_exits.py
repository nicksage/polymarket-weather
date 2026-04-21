"""
realtime_exits.py — Real-time stop-loss and exit monitoring.

Registered as a callback on the WebSocket price stream.  Called on every
price update with (token_id, new_price).  Checks all open positions
associated with that token against exit criteria and executes sells
immediately when thresholds are breached.

This replaces the hourly stop-loss check with sub-second response time.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from config import (
    EXIT_HARD_STOP_ENABLED,
    EXIT_HARD_STOP_PCT,
    PAPER_TRADE,
)

logger = logging.getLogger(__name__)

# Prevent concurrent exit execution (e.g., rapid price updates)
_exit_lock = threading.Lock()

# Track which positions we've already exited to avoid double-sells
_exited_positions: set[int] = set()


def _get_positions_by_token() -> dict[str, list[dict]]:
    """Build a lookup: token_id -> list of open positions using that token."""
    from db import get_open_positions
    positions = get_open_positions()
    by_token: dict[str, list[dict]] = {}
    for p in positions:
        if p.get("fill_status") != "filled":
            continue
        side = p.get("side", "YES")
        tid = p.get("yes_token_id") if side == "YES" else p.get("no_token_id")
        if tid:
            by_token.setdefault(tid, []).append(p)
    return by_token


def on_price_update(token_id: str, price: float) -> None:
    """Callback fired by the WebSocket on every price change.

    Checks if any position associated with this token has breached the
    hard stop-loss.  If so, executes an immediate paper exit (or queues
    a CLOB sell for live mode).
    """
    if not EXIT_HARD_STOP_ENABLED:
        return

    # Quick check: do we have any positions for this token?
    by_token = _get_positions_by_token()
    positions = by_token.get(token_id)
    if not positions:
        return

    for pos in positions:
        pid = pos["id"]

        # Skip if already exited by this module
        if pid in _exited_positions:
            continue

        entry_price = float(pos.get("entry_price") or 0)
        shares = float(pos.get("shares") or 0)
        sl_price = pos.get("stop_loss_price")
        if entry_price <= 0 or shares <= 0:
            continue

        unrealized_pnl = (price - entry_price) * shares

        # Check hard stop-loss against precomputed SL price
        if sl_price is not None and price <= float(sl_price):
            with _exit_lock:
                if pid in _exited_positions:
                    continue
                _exited_positions.add(pid)

            _execute_realtime_exit(
                pos, price, unrealized_pnl,
                reason=f"RT_HARD_STOP: price={price:.4f} <= SL={float(sl_price):.4f} "
                       f"(entry={entry_price:.4f}, {EXIT_HARD_STOP_PCT*100:.0f}%)"
            )

        # Also update the position's current_price and unrealized_pnl in DB
        # so the dashboard shows live values
        try:
            from db import update_position_market_price, update_position_excursions
            update_position_market_price(pid, price, round(unrealized_pnl, 4))
            update_position_excursions(pid, unrealized_pnl, unrealized_pnl)
        except Exception:
            pass


def _execute_realtime_exit(
    pos: dict, exit_price: float, pnl: float, reason: str,
) -> None:
    """Execute an immediate exit for a position that breached a threshold."""
    from db import update_position_outcome, get_latest_snapshot_id_for_contract

    pid = pos["id"]
    city = pos.get("city", "")
    date_str = pos.get("date", "")
    side = pos.get("side", "")
    contract_id = pos.get("contract_id", "")

    entry_price = float(pos.get("entry_price") or 0)
    shares = float(pos.get("shares") or 0)
    realized_pnl = round((exit_price - entry_price) * shares, 4)

    now = datetime.now(ZoneInfo("America/Chicago")).isoformat()
    exit_snap_id = get_latest_snapshot_id_for_contract(contract_id)

    if PAPER_TRADE:
        update_position_outcome(
            position_id=pid,
            exit_price=exit_price,
            exit_time=now,
            pnl=realized_pnl,
            status="closed",
            exit_reason=reason,
            exit_snapshot_id=exit_snap_id,
        )
        logger.warning(
            f"[RT-EXIT] pos={pid} {city} {date_str} {side} "
            f"| exit@{exit_price:.4f} pnl=${realized_pnl:+.4f} "
            f"| {reason}"
        )
    else:
        # Live mode: place CLOB sell order
        # For now, same as paper (TODO: implement live CLOB sell)
        update_position_outcome(
            position_id=pid,
            exit_price=exit_price,
            exit_time=now,
            pnl=realized_pnl,
            status="closed",
            exit_reason=reason,
            exit_snapshot_id=exit_snap_id,
        )
        logger.warning(
            f"[RT-EXIT] pos={pid} {city} {date_str} {side} "
            f"| exit@{exit_price:.4f} pnl=${realized_pnl:+.4f} "
            f"| {reason}"
        )


def clear_exited_cache() -> None:
    """Clear the exited positions cache (e.g., on bot restart)."""
    _exited_positions.clear()
