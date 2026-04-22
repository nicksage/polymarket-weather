"""
realtime_exits.py — Real-time stop-loss and trailing stop monitoring.

Registered as a callback on the WebSocket price stream.  Called on every
price update with (token_id, new_price).  Checks all open positions
associated with that token against exit criteria and executes sells
immediately when thresholds are breached.

Supports two exit modes (strategy-dependent):
  - Hard stop: fixed percentage below entry (top_bin_value)
  - Trailing stop: percentage below peak price (market_price_value)
  - Take profit: sell when price reaches a target level
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from config import (
    EXIT_HARD_STOP_ENABLED,
    EXIT_HARD_STOP_PCT,
    ACTIVE_STRATEGY,
    PAPER_TRADE,
)

logger = logging.getLogger(__name__)

_exit_lock = threading.Lock()
_exited_positions: set[int] = set()

# Peak price tracking: pid -> highest price seen
_peak_prices: dict[int, float] = {}
_peak_lock = threading.Lock()

# MPV trailing stop config (read from env)
_MPV_TRAIL_PCT = float(os.getenv("MPV_TRAIL_PCT", "0.12"))
_MPV_TRAIL_ACTIVATION = float(os.getenv("MPV_TRAIL_ACTIVATION", "0.10"))
_MPV_TAKE_PROFIT = float(os.getenv("MPV_TAKE_PROFIT", "0.90"))
_MPV_HARD_STOP_PCT = float(os.getenv("MPV_HARD_STOP_PCT", "0.30"))


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


def _update_peak(pid: int, price: float, entry_price: float) -> float:
    """Update and return the peak price for a position."""
    with _peak_lock:
        current_peak = _peak_prices.get(pid, entry_price)
        if price > current_peak:
            _peak_prices[pid] = price
            return price
        return current_peak


def on_price_update(token_id: str, price: float) -> None:
    """Callback fired by the WebSocket on every price change.

    Checks trailing stop (MPV) or hard stop (TBV) depending on strategy.
    Also updates peak_price in DB for position tracking.
    """
    by_token = _get_positions_by_token()
    positions = by_token.get(token_id)
    if not positions:
        return

    is_mpv = ACTIVE_STRATEGY == "market_price_value"

    for pos in positions:
        pid = pos["id"]

        if pid in _exited_positions:
            continue

        entry_price = float(pos.get("entry_price") or 0)
        shares = float(pos.get("shares") or 0)
        if entry_price <= 0 or shares <= 0:
            continue

        unrealized_pnl = (price - entry_price) * shares
        peak = _update_peak(pid, price, entry_price)

        # Update current_price, peak_price, and P&L in DB
        try:
            from db import update_position_market_price, update_position_excursions
            update_position_market_price(pid, price, round(unrealized_pnl, 4))
            update_position_excursions(pid, unrealized_pnl, unrealized_pnl)
            _update_peak_in_db(pid, peak)
        except Exception:
            pass

        if is_mpv:
            _check_mpv_exits(pos, pid, price, peak, entry_price, shares, unrealized_pnl)
        else:
            _check_tbv_exits(pos, pid, price, entry_price, shares, unrealized_pnl)


def _bin_label(pos: dict) -> str:
    """Build a short bin label like '12-12C' or '68-70F'."""
    rl = pos.get("range_low")
    rh = pos.get("range_high")
    unit = pos.get("unit", "celsius")
    suffix = "F" if unit == "fahrenheit" else "C"
    if rl is not None and rh is not None:
        return f"{int(rl)}-{int(rh)}{suffix}"
    elif rl is not None:
        return f">={int(rl)}{suffix}"
    elif rh is not None:
        return f"<={int(rh)}{suffix}"
    return "?"


def _check_mpv_exits(
    pos: dict, pid: int, price: float, peak: float,
    entry_price: float, shares: float, unrealized_pnl: float,
) -> None:
    """Market Price Value exits: take profit, trailing stop, hard stop."""
    city = pos.get("city", "")
    date_str = pos.get("date", "")
    side = pos.get("side", "YES")
    bin_str = _bin_label(pos)

    # 1. Take profit
    if price >= _MPV_TAKE_PROFIT:
        with _exit_lock:
            if pid in _exited_positions:
                return
            _exited_positions.add(pid)
        _execute_realtime_exit(
            pos, price, unrealized_pnl,
            reason=f"RT_TAKE_PROFIT: price={price:.4f} >= TP={_MPV_TAKE_PROFIT}"
        )
        logger.warning(
            f"[ Take Profit ]  |  {city}  |  {date_str} {side}  |  {bin_str}  |  "
            f"Entry: ${entry_price:.4f}  |  Exit: ${price:.4f}  |  "
            f"P&L: ${(price - entry_price) * shares:+.2f}"
        )
        return

    # 2. Trailing stop (only after price rises above entry + activation)
    if peak >= entry_price * (1 + _MPV_TRAIL_ACTIVATION):
        trail_level = peak * (1 - _MPV_TRAIL_PCT)
        if price <= trail_level:
            with _exit_lock:
                if pid in _exited_positions:
                    return
                _exited_positions.add(pid)
            _execute_realtime_exit(
                pos, trail_level, unrealized_pnl,
                reason=f"RT_TRAILING_STOP: price={price:.4f} <= trail={trail_level:.4f} "
                       f"(peak={peak:.4f}, {_MPV_TRAIL_PCT:.0%} trail)"
            )
            logger.warning(
                f"[ Trail Stop ]  |  {city}  |  {date_str} {side}  |  {bin_str}  |  "
                f"Entry: ${entry_price:.4f}  |  Exit: ${trail_level:.4f}  |  "
                f"P&L: ${(trail_level - entry_price) * shares:+.2f}"
            )
            return

    # 3. Hard stop (before trail activates)
    hard_stop_level = entry_price * (1 - _MPV_HARD_STOP_PCT)
    if price <= hard_stop_level:
        with _exit_lock:
            if pid in _exited_positions:
                return
            _exited_positions.add(pid)
        _execute_realtime_exit(
            pos, price, unrealized_pnl,
            reason=f"RT_HARD_STOP: price={price:.4f} <= stop={hard_stop_level:.4f}"
        )
        logger.warning(
            f"[ Stop Loss ]  |  {city}  |  {date_str} {side}  |  {bin_str}  |  "
            f"Entry: ${entry_price:.4f}  |  Exit: ${price:.4f}  |  "
            f"P&L: ${(price - entry_price) * shares:+.2f}"
        )


def _check_tbv_exits(
    pos: dict, pid: int, price: float,
    entry_price: float, shares: float, unrealized_pnl: float,
) -> None:
    """Top Bin Value exits: hard stop only (original behavior)."""
    if not EXIT_HARD_STOP_ENABLED:
        return

    sl_price = pos.get("stop_loss_price")
    if sl_price is not None and price <= float(sl_price):
        with _exit_lock:
            if pid in _exited_positions:
                return
            _exited_positions.add(pid)

        city = pos.get("city", "")
        date_str = pos.get("date", "")
        side = pos.get("side", "YES")
        bin_str = _bin_label(pos)

        _execute_realtime_exit(
            pos, price, unrealized_pnl,
            reason=f"HARD_STOP: price={price:.4f} <= SL={float(sl_price):.4f} "
                   f"(entry={entry_price:.4f}, {EXIT_HARD_STOP_PCT*100:.0f}%)"
        )
        logger.warning(
            f"[ Stop Loss ]  |  {city}  |  {date_str} {side}  |  {bin_str}  |  "
            f"Entry: ${entry_price:.4f}  |  Exit: ${price:.4f}  |  "
            f"P&L: ${(price - entry_price) * shares:+.2f}"
        )


def _execute_realtime_exit(
    pos: dict, exit_price: float, pnl: float, reason: str,
) -> None:
    """Execute an immediate exit for a position that breached a threshold."""
    from db import update_position_outcome, get_latest_snapshot_id_for_contract

    pid = pos["id"]
    contract_id = pos.get("contract_id", "")
    entry_price = float(pos.get("entry_price") or 0)
    shares = float(pos.get("shares") or 0)
    realized_pnl = round((exit_price - entry_price) * shares, 4)

    now = datetime.now(ZoneInfo("America/Chicago")).isoformat()
    exit_snap_id = get_latest_snapshot_id_for_contract(contract_id)

    update_position_outcome(
        position_id=pid,
        exit_price=exit_price,
        exit_time=now,
        pnl=realized_pnl,
        status="closed",
        exit_reason=reason,
        exit_snapshot_id=exit_snap_id,
    )


def _update_peak_in_db(pid: int, peak_price: float) -> None:
    """Update peak_price column in positions table."""
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE positions SET peak_price = MAX(COALESCE(peak_price, 0), ?) WHERE id = ?",
            (peak_price, pid)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def clear_exited_cache() -> None:
    """Clear the exited positions cache (e.g., on bot restart)."""
    _exited_positions.clear()
    with _peak_lock:
        _peak_prices.clear()
