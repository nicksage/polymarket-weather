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
    TRAIL_ACTIVATION_GAIN,
    TRAIL_TIERS,
)
from trailing_stop import evaluate_trailing_stop, lookup_trail_pct, update_peak

logger = logging.getLogger(__name__)

_exit_lock = threading.Lock()
_exited_positions: set[int] = set()

# Peak price tracking: pid -> highest price seen.
# This is a perf cache only — the DB column positions.peak_price is the
# source of truth.  On first sighting of a position in a fresh process,
# the cache is seeded from the DB row so a restart never erases the
# locked-in peak (and therefore the trail).  See _update_peak() below.
_peak_prices: dict[int, float] = {}
_peak_lock = threading.Lock()

# MPV non-trail exit config (trail config lives in config.py: TRAIL_TIERS + TRAIL_ACTIVATION_GAIN)
_MPV_TAKE_PROFIT = float(os.getenv("MPV_TAKE_PROFIT", "0.90"))
_MPV_HARD_STOP_PCT = float(os.getenv("MPV_HARD_STOP_PCT", "0.30"))


def _get_positions_by_token() -> dict[str, list[dict]]:
    """Build a lookup: token_id -> list of open, fully-active positions
    (NOT positions already in the exit ladder — those have a pending sell
    order; we'd be double-firing if we re-evaluated stops on them).
    """
    from db import get_open_positions
    positions = get_open_positions()
    by_token: dict[str, list[dict]] = {}
    for p in positions:
        if p.get("fill_status") != "filled":
            continue
        if p.get("status") != "open":     # explicitly exclude 'exiting'
            continue
        side = p.get("side", "YES")
        tid = p.get("yes_token_id") if side == "YES" else p.get("no_token_id")
        if tid:
            by_token.setdefault(tid, []).append(p)
    return by_token


def _update_peak(
    pid: int, price: float, entry_price: float, db_peak: float | None,
) -> float:
    """Update and return the peak price for a position.

    Restart safety: on first sighting of a position in this process, seed
    the in-memory cache from the DB row's peak_price (passed as `db_peak`).
    Without this seed, after a restart we'd lose every peak above
    entry_price until a new tick arrived ABOVE that lost peak — meaning a
    position with peak=0.85 in the DB and a current tick of 0.70 would
    arm the trail from 0.70 instead of 0.85.  That's exactly the failure
    mode the engineering spec calls out.
    """
    with _peak_lock:
        if pid not in _peak_prices:
            # First sighting in this process — seed from DB
            _peak_prices[pid] = update_peak(
                current_peak=(db_peak if db_peak is not None else entry_price),
                new_price=price,
                entry_price=entry_price,
            )
        else:
            _peak_prices[pid] = update_peak(
                current_peak=_peak_prices[pid],
                new_price=price,
                entry_price=entry_price,
            )
        return _peak_prices[pid]


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
        # Seed peak cache from the DB column on first sighting (restart-safe)
        db_peak_raw = pos.get("peak_price")
        db_peak = float(db_peak_raw) if db_peak_raw is not None else None
        peak = _update_peak(pid, price, entry_price, db_peak)

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


def _evaluate_trail(
    entry_price: float, peak: float, price: float,
) -> tuple[float, str] | None:
    """Evaluate the trailing stop using the (single source of truth) tier
    table from config.  Returns (stop_level, "TRAILING_STOP") when the
    trail fires, or None.  Single-tier vs multi-tier is just a matter of
    how many rows are in TRAIL_TIERS — same code path either way.

    Both this module and strategies/market_price_value._classify_position
    call this helper, so the two evaluation sites can't drift apart.
    """
    return evaluate_trailing_stop(
        entry_price=entry_price,
        peak_price=peak,
        current_price=price,
        activation_gain=TRAIL_ACTIVATION_GAIN,
        tiers=TRAIL_TIERS,
    )


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

    # 2. Trailing stop — tier table from config (single source of truth)
    trail_decision = _evaluate_trail(entry_price, peak, price)
    if trail_decision is not None:
        trail_level, _reason_tag = trail_decision
        with _exit_lock:
            if pid in _exited_positions:
                return
            _exited_positions.add(pid)
        tier_pct = lookup_trail_pct(peak, TRAIL_TIERS) or 0.0
        reason = (
            f"RT_TRAILING_STOP: price={price:.4f} <= trail={trail_level:.4f} "
            f"(peak={peak:.4f}, tier={tier_pct:.0%} trail)"
        )
        log_pct = tier_pct
        _execute_realtime_exit(pos, trail_level, unrealized_pnl, reason=reason)
        logger.warning(
            f"[ Trail Stop ]  |  {city}  |  {date_str} {side}  |  {bin_str}  |  "
            f"Entry: ${entry_price:.4f}  |  Peak: ${peak:.4f}  |  "
            f"Exit: ${trail_level:.4f}  |  Trail: {log_pct:.0%}  |  "
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
    """Trigger an exit for a position that breached a real-time threshold.

    Paper mode: closes the position in DB (status='closed') with the
    breach price as exit_price.

    Live mode: places a CLOB sell at the FIRST ladder rung (retry_count=0,
    price = exit_price * 0.99) via execution.execute_exit().  Position
    transitions to status='exiting'; subsequent monitor cycles advance the
    ladder if the order doesn't fill.
    """
    # Lazy import to avoid circular at module-load time
    from execution import execute_exit, get_clob_client

    # Reuse a CLOB client if one is in scope; otherwise create a fresh one.
    # In production this is called from the WebSocket callback thread,
    # which doesn't have a client passed in; we get_clob_client() each call.
    # In paper mode, get_clob_client() returns None and execute_exit handles it.
    client = get_clob_client()

    result = execute_exit(
        position             = pos,
        intended_exit_price  = exit_price,
        exit_reason          = reason,
        client               = client,
        retry_count          = 0,
    )
    # Result dict logged at INFO inside execute_exit; nothing more to do.
    if result.get("status") not in ("paper_closed", "exit_pending"):
        logger.warning(
            f"realtime exit for pos={pos.get('id')} returned unexpected "
            f"status={result.get('status')}: {result}"
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
