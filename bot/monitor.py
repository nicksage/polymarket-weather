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
    Cancel pending orders that are old enough to have had a fair chance
    to fill.  An order placed seconds before a monitor cycle (e.g. the
    inline cycle that runs right after `trading_run` on startup) MUST
    NOT be cancelled — it hasn't had time to fill.

    Skip rule: orders younger than MIN_ORDER_AGE_BEFORE_CANCEL_MIN
    minutes are left alone for future cycles to catch up with.

    Paper trades are never pending, so this step is a no-op in paper mode.
    Returns the count of orders cancelled.
    """
    from config import MIN_ORDER_AGE_BEFORE_CANCEL_MIN
    pending = get_pending_positions()
    if not pending:
        return 0

    cancelled = 0
    now = _now()
    now_dt = datetime.fromisoformat(now)
    skipped_too_young = 0

    for pos in pending:
        pos_id   = pos["id"]
        order_id = pos.get("order_id")
        city     = pos.get("city", "")
        date     = pos.get("date", "")
        contract = pos.get("contract_id", "")[:12]
        entry    = pos.get("entry_time") or ""

        # Age check — skip orders too young to be presumed dead.  Without
        # this, the inline monitor that runs ~5s after trading_run would
        # cancel every brand-new order before it could fill.
        try:
            entry_dt = datetime.fromisoformat(entry)
            age_min  = (now_dt - entry_dt).total_seconds() / 60.0
        except Exception:
            # Can't parse entry_time → fall through (cancel as before, since
            # we can't tell age — but this should only happen for legacy
            # rows; new positions always have a parseable entry_time).
            age_min = float("inf")

        if age_min < MIN_ORDER_AGE_BEFORE_CANCEL_MIN:
            skipped_too_young += 1
            logger.debug(
                f"[MONITOR] Skipping cancel of pos={pos_id} {city} {date} "
                f"{contract} — only {age_min:.1f}m old, threshold "
                f"{MIN_ORDER_AGE_BEFORE_CANCEL_MIN}m"
            )
            continue

        if order_id and client:
            success = cancel_order(order_id, client)
            if not success:
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
        from activity import log_activity
        log_activity(
            "CANCEL", position_id=pos_id,
            message=(
                f"BUY cancelled by monitor sweep: {city} {date} {contract} "
                f"order={(order_id or '')[:12]} — not filled within window"
            ),
            source="monitor_sweep",
            order_id=order_id, contract_id=contract,
        )

    if skipped_too_young > 0:
        logger.info(
            f"[MONITOR] Cancel pass skipped {skipped_too_young} order(s) younger "
            f"than {MIN_ORDER_AGE_BEFORE_CANCEL_MIN}m — they'll be re-evaluated next cycle"
        )
    return cancelled


# ---------------------------------------------------------------------------
# Step 1a — Reconcile pending fills (BEFORE cancel + advance steps)
# ---------------------------------------------------------------------------

def _cancel_overcommitted_orders(client) -> int:
    """Auto-cancel resting orders for positions that are over-committed
    (committed_usdc > target_size_usdc).  Added 2026-04-30 as the
    natural completion of Phase B — Phase B prevents NEW double-commits
    via committed_usdc-aware top-up gap calc, but couldn't clean up
    over-committed state already on the book from before the fix.

    Algorithm (per over-committed position):
      1. List orders still committing capital (pending/live/matched/partial),
         oldest first.
      2. Cancel the oldest order whose resting (intended − filled) portion
         contributes to the excess.  Cancellation marks the ledger row
         'cancelled'; if the order had a partial fill, the filled shares
         stay on chain (the partial-cancelled semantic).
      3. Re-evaluate: if still over-committed by > $1, cancel next oldest
         next cycle (don't burn through multiple cancels in one sweep —
         CLOB takes a moment to register cancellations).

    Skip rule: orders younger than MIN_ORDER_AGE_BEFORE_CANCEL_MIN minutes
    are left alone — same protection as _cancel_pending_orders.  Without
    this, a freshly-placed top-up could be auto-cancelled before it has
    a chance to fill, even if the math says we're over-committed (because
    the entry's resting portion will eventually cancel out of its own
    accord at the cancel-pass age cutoff).

    Paper mode: no-op.
    """
    from config import MIN_ORDER_AGE_BEFORE_CANCEL_MIN
    from db import (
        get_overcommitted_positions, get_cancellable_orders_for_position,
        update_position_order_status,
    )
    from execution import cancel_order

    if PAPER_TRADE or client is None:
        return 0

    over_positions = get_overcommitted_positions()
    if not over_positions:
        return 0

    cancelled_count = 0
    now_dt = datetime.fromisoformat(_now())

    for pos in over_positions:
        pid = pos["position_id"]
        target = float(pos["target_size_usdc"])
        committed = float(pos["committed_usdc"])
        excess = float(pos["excess"])

        candidates = get_cancellable_orders_for_position(pid)
        if not candidates:
            continue

        for candidate in candidates:
            order_id = candidate["order_id"]
            created_at = candidate.get("created_at") or ""
            try:
                created_dt = datetime.fromisoformat(created_at)
                age_min = (now_dt - created_dt).total_seconds() / 60.0
            except Exception:
                age_min = float("inf")

            if age_min < MIN_ORDER_AGE_BEFORE_CANCEL_MIN:
                logger.debug(
                    f"[MONITOR] Over-committed pos={pid} "
                    f"({pos['city']} {pos['date']}) — oldest cancellable "
                    f"order is only {age_min:.1f}m old (< "
                    f"{MIN_ORDER_AGE_BEFORE_CANCEL_MIN}m floor); waiting"
                )
                break  # don't try younger orders either

            # Cancel this order
            ok = cancel_order(order_id, client)
            if not ok:
                # cancel_order logs its own error; mark the ledger row
                # cancelled anyway so subsequent cycles don't re-target it
                update_position_order_status(
                    order_id        = order_id,
                    status          = "cancelled",
                    cancelled_reason = "overcommit_sweep_clob_cancel_failed",
                    closed          = True,
                )
            cancelled_count += 1
            from activity import log_activity
            log_activity(
                "CANCEL", position_id=pid,
                message=(
                    f"over-commitment sweep cancelled order: pos={pid} "
                    f"{pos['city']} {pos['date']} role={candidate['role']} "
                    f"intended=${candidate['intended_usdc']:.2f} "
                    f"filled=${candidate.get('filled_usdc', 0):.2f} "
                    f"resting=${candidate['resting_usdc']:.2f} — "
                    f"committed=${committed:.2f} > target=${target:.2f} "
                    f"(excess=${excess:.2f})"
                ),
                source="overcommit_sweep",
                order_id=order_id,
            )
            # Cancel ONE order per position per cycle.  CLOB takes a
            # moment to register; let next cycle re-evaluate from a
            # clean state (avoids cascading cancels).
            break

    if cancelled_count > 0:
        logger.log(SUMMARY_LEVEL,
            f"[MONITOR] Over-commitment sweep cancelled {cancelled_count} "
            f"resting order(s) across {len(over_positions)} position(s)"
        )
    return cancelled_count


def _reconcile_pending_fills(client) -> tuple[int, int, int]:
    """Safety-net reconciliation for in-flight orders (BUYS, SELLS, TOP-UPS).

    Phase 9 changed this from the PRIMARY fill detector into a backup for
    the user-channel WebSocket.  The WS gives sub-second fill detection;
    this REST sweep catches anything that slipped through during WS
    downtime, dedupe via the handler's monotonic trade_status check.

    For each pending order we:
      1. Poll its CLOB status.
      2. If engine-matched, build a synthetic 'confirmed' trade event from
         the response.
      3. Hand it to fill_handler.apply_trade_event — which is the SAME
         function the WS uses, so no path divergence.
      4. If externally cancelled, hand it to apply_order_event so the
         pending_topup_* fields get cleared (or the position cancelled).

    Returns (buys_filled, sells_filled, topups_filled) for summary log
    compatibility — counts are derived from handler return values.

    Paper mode: no-op (paper orders are filled immediately on placement).
    """
    from db import (
        get_pending_positions, get_exiting_positions,
        get_positions_with_pending_topup,
        clear_position_topup_pending,
    )
    from execution import (
        get_order_status, is_order_fully_filled, is_order_cancelled,
        extract_fill_price, extract_fee_amount, extract_fee_rate_bps,
        cancel_order,
    )
    from fill_handler import apply_trade_event, apply_order_event

    if PAPER_TRADE or client is None:
        return 0, 0, 0

    buys_filled = 0
    sells_filled = 0
    topups_filled = 0

    def _synthesize_trade_event(
        *, order_id: str, status: dict, fallback_price: float,
        fallback_shares: float | None = None,
    ) -> dict:
        """Build a 'confirmed'-status trade event from a REST get_order
        response.  The handler will dedupe via trade_status if the WS
        already wrote this fill."""
        actual_fill = extract_fill_price(status, fallback=fallback_price)
        actual_size = float(
            status.get("size_matched")
            or status.get("size")
            or (fallback_shares if fallback_shares is not None else 0)
        )
        fill_usdc = actual_size * actual_fill
        fee_usdc  = extract_fee_amount(status, fill_amount_usdc=fill_usdc)
        bps       = extract_fee_rate_bps(status)
        synth: dict = {
            "id":              status.get("id") or f"rest:{order_id}",
            "status":          "confirmed",
            "taker_order_id":  order_id,
            "size":            actual_size,
            "price":           actual_fill,
        }
        # Only set fee fields when we actually have data — the handler's
        # own _extract_fee will return 0 when none is present.
        if fee_usdc > 0:
            synth["fee"] = fee_usdc
        if bps is not None:
            synth["fee_rate_bps"] = bps
        return synth

    def _has_partial_fill(status_resp: dict) -> bool:
        """Order has at least some on-chain fill, even if not fully matched
        and even if the order itself has been cancelled.  Critical for
        recording the matched portion of an order that filled some shares
        before the rest got cancelled — without this, partial fills become
        ghost positions on chain that the bot's DB doesn't know about."""
        try:
            return float(status_resp.get("size_matched") or 0) > 0
        except (TypeError, ValueError):
            return False

    # ---- BUY-side reconciliation ----
    for pos in get_pending_positions():
        order_id = pos.get("order_id")
        if not order_id:
            continue
        status = get_order_status(order_id, client)
        if status is None:
            continue
        # Partial-fill check FIRST — even cancelled orders may have a real
        # matched portion that needs to land in the DB (bug 2026-04-29:
        # cancelled-with-partial-fill was treated as full cancel, losing
        # the on-chain shares from our books).
        if _has_partial_fill(status):
            event = _synthesize_trade_event(
                order_id        = order_id,
                status          = status,
                fallback_price  = float(pos.get("entry_price") or 0),
                fallback_shares = pos.get("shares"),
            )
            result = apply_trade_event(event)
            if result.get("action") == "filled":
                buys_filled += 1
            elif result.get("action") == "ignored_regression":
                logger.debug(
                    f"[MONITOR] BUY pos={pos['id']} already CONFIRMED via WS "
                    f"— REST poll skipped (idempotent dedup)"
                )
            # Whether fully filled or just partial, the DB now has the
            # matched portion.  If the order was cancelled with a partial
            # fill, the unfilled remainder is implicitly released by the
            # CLOB itself; we don't need to (and shouldn't) cancel again.
            continue
        # No partial fill — handle as fully cancelled or still-pending
        if is_order_cancelled(status):
            apply_order_event({"id": order_id, "type": "CANCELLATION"})

    # ---- SELL-side reconciliation ----
    for pos in get_exiting_positions():
        order_id = pos.get("exit_order_id")
        if not order_id:
            continue
        status = get_order_status(order_id, client)
        if status is None:
            continue
        # Same partial-fill-first pattern as buys.  A sell that partially
        # matched and then was cancelled (e.g. operator cancel) needs the
        # filled portion recorded as exit shares.
        if _has_partial_fill(status):
            event = _synthesize_trade_event(
                order_id        = order_id,
                status          = status,
                fallback_price  = float(pos.get("exit_intended_price") or 0),
                fallback_shares = pos.get("shares"),
            )
            result = apply_trade_event(event)
            if result.get("action") == "filled":
                sells_filled += 1
            continue
        if is_order_cancelled(status):
            # Cancelled with no fill — leave position in 'exiting' for the
            # ladder advancer to retry next cycle.
            continue

    # ---- TOP-UP reconciliation ----
    for pos in get_positions_with_pending_topup():
        order_id = pos.get("pending_topup_order_id")
        amt_usdc = float(pos.get("pending_topup_amount_usdc") or 0)
        intended = float(pos.get("pending_topup_intended_price") or 0)
        if not order_id or amt_usdc <= 0 or intended <= 0:
            continue
        status = get_order_status(order_id, client)
        if status is None:
            continue
        if _has_partial_fill(status):
            event = _synthesize_trade_event(
                order_id        = order_id,
                status          = status,
                fallback_price  = intended,
                fallback_shares = (amt_usdc / intended) if intended > 0 else None,
            )
            result = apply_trade_event(event)
            if result.get("action") == "filled":
                topups_filled += 1
        elif is_order_cancelled(status):
            clear_position_topup_pending(pos["id"])
            logger.info(
                f"[MONITOR] TOP-UP cancelled (external): pos={pos['id']} "
                f"order={order_id[:12]} — cleared pending fields"
            )
        else:
            # Still live and unfilled — cancel it.  Next strategy scan can
            # re-issue if conditions still warrant.
            cancelled_ok = cancel_order(order_id, client)
            clear_position_topup_pending(pos["id"])
            logger.info(
                f"[MONITOR] TOP-UP unfilled — cancelled order={order_id[:12]} "
                f"(cancel_ok={cancelled_ok}); next scan may retry"
            )

    if buys_filled or sells_filled or topups_filled:
        logger.log(SUMMARY_LEVEL,
            f"[MONITOR] Fill reconciliation (REST safety net): "
            f"{buys_filled} buy(s), {sells_filled} sell(s), "
            f"{topups_filled} top-up(s) confirmed"
        )
    return buys_filled, sells_filled, topups_filled


# ---------------------------------------------------------------------------
# Step 1b — Advance exit ladders (cancel unfilled rung + escalate)
# ---------------------------------------------------------------------------

# Concurrency guard: this function is invoked by both the hourly monitor
# and the */5 fast-exit job.  If both fire near the same time (e.g. at
# minute 40), the second invocation skips rather than racing on cancels.
import threading as _threading
_advance_exit_ladders_lock = _threading.Lock()


def _advance_exit_ladders(client) -> int:
    """Walk every position with status='exiting' and either:
       - confirm a fill (TODO Phase 4 — currently relies on monitor's
         existing P&L-via-Data-API logic to detect fills); for now, this
         step focuses on the unfilled case
       - bleed circuit: if current bid has dropped >EXIT_BLEED_CROSS_PCT
         below the original trigger, cancel + cross-spread immediately
         (skip the patient ladder)
       - cancel the existing sell order and re-issue at the next ladder rung
         (rung price is bid-anchored — see exit_ladder.ladder_price)
       - if all 4 ladder rungs have been tried, cross the spread

    Paper mode: no-op (paper exits close immediately, status='closed').
    Returns the number of positions whose ladder advanced this cycle.

    Idempotent across concurrent invocations via _advance_exit_ladders_lock.
    """
    from db import get_exiting_positions
    from execution import cancel_exit_order, execute_exit, _get_best_bid
    from exit_ladder import is_ladder_exhausted, MAX_LADDER_RETRIES
    from config import EXIT_BLEED_CROSS_PCT

    if PAPER_TRADE or client is None:
        return 0

    if not _advance_exit_ladders_lock.acquire(blocking=False):
        logger.debug(
            "[MONITOR] _advance_exit_ladders already running in another "
            "thread — skipping this invocation"
        )
        return 0
    try:
        exiting = get_exiting_positions()
        if not exiting:
            return 0

        advanced = 0
        for pos in exiting:
            pid           = pos["id"]
            intended      = pos.get("exit_intended_price")
            retry_count   = int(pos.get("exit_retry_count") or 0)
            contract      = pos.get("contract_id", "")[:12]
            city          = pos.get("city", "")
            date_str      = pos.get("date", "")
            prior_reason  = pos.get("exit_reason") or "ladder_advance"
            side          = pos.get("side", "YES")
            token_id      = (pos.get("yes_token_id") if side == "YES"
                             else pos.get("no_token_id"))

            if intended is None:
                logger.warning(
                    f"[MONITOR] pos={pid} status='exiting' but no intended_exit_price; "
                    f"leaving alone for human review"
                )
                continue

            # ---- Bleed circuit-breaker -------------------------------------
            # If the current bid has bled more than EXIT_BLEED_CROSS_PCT below
            # the original trigger, force cross-spread regardless of which
            # rung we're on.  Patient laddering at the trigger price would
            # leave us posting unfillable offers above the bid.
            force_cross = False
            current_bid = None
            if token_id and EXIT_BLEED_CROSS_PCT < 1.0:
                current_bid = _get_best_bid(client, token_id)
                if current_bid is not None and current_bid > 0:
                    bleed_threshold = float(intended) * (1.0 - EXIT_BLEED_CROSS_PCT)
                    if current_bid < bleed_threshold:
                        force_cross = True
                        logger.warning(
                            f"[MONITOR] pos={pid} {city} {date_str} {contract} "
                            f"BLEED CIRCUIT: bid={current_bid:.4f} < "
                            f"{(1-EXIT_BLEED_CROSS_PCT)*100:.0f}% × "
                            f"trigger={float(intended):.4f} ({bleed_threshold:.4f}) "
                            f"— forcing cross-spread"
                        )

            # Cancel the existing sell order (best-effort).  If the order
            # actually filled in the meantime, the cancel will fail or be a
            # no-op; the next monitor cycle's P&L update will detect that the
            # position is closed on-chain via the Data API.
            cancelled_ok = cancel_exit_order(pos, client)
            if not cancelled_ok:
                logger.warning(
                    f"[MONITOR] pos={pid} could not cancel exit order "
                    f"{pos.get('exit_order_id', '')[:12]} — skipping advance "
                    f"(may have already filled)"
                )
                continue

            # Decide next step: bleed-forced cross, ladder rung, or normal cross-spread
            next_retry = retry_count + 1
            if force_cross:
                # Bleed circuit fired — jump to cross-spread regardless of rung
                result = execute_exit(
                    position             = pos,
                    intended_exit_price  = float(intended),
                    exit_reason          = f"{prior_reason}|bleed_cross",
                    client               = client,
                    retry_count          = MAX_LADDER_RETRIES,  # force cross path
                    cross_spread         = True,
                )
            elif is_ladder_exhausted(next_retry):
                # Already on rung 3 → escalate to cross-spread
                logger.info(
                    f"[MONITOR] pos={pid} {city} {date_str} {contract} — "
                    f"ladder exhausted at rung {retry_count}, crossing spread"
                )
                result = execute_exit(
                    position             = pos,
                    intended_exit_price  = float(intended),
                    exit_reason          = f"{prior_reason}|cross_spread",
                    client               = client,
                    retry_count          = next_retry,
                    cross_spread         = True,
                )
            else:
                logger.info(
                    f"[MONITOR] pos={pid} {city} {date_str} {contract} — "
                    f"advancing exit ladder rung {retry_count} → {next_retry}"
                )
                result = execute_exit(
                    position             = pos,
                    intended_exit_price  = float(intended),
                    exit_reason          = f"{prior_reason}|advance_{next_retry}",
                    client               = client,
                    retry_count          = next_retry,
                )

            status = result.get("status")
            if status == "exit_pending":
                advanced += 1
            elif status in ("closed_via_balance_recovery", "shares_resynced"):
                # The balance-mismatch self-heal fired (execution.py).  This
                # isn't a failure — the position was either closed cleanly
                # (chain=0) or its shares column was synced to chain truth so
                # the next cycle retries at the correct size.  Log at INFO.
                logger.info(
                    f"[MONITOR] pos={pid} ladder advance superseded by "
                    f"balance-mismatch self-heal: {status}"
                )
            else:
                logger.warning(
                    f"[MONITOR] pos={pid} ladder advance failed: {result}"
                )

        if advanced:
            logger.log(SUMMARY_LEVEL,
                f"[MONITOR] {advanced} exit ladder(s) advanced "
                f"(of {len(exiting)} exiting position(s))"
            )
        return advanced
    finally:
        _advance_exit_ladders_lock.release()


def run_exit_ladder_fast() -> int:
    """Fast-cycle wrapper for the */5 minute APScheduler job.

    Acquires its own CLOB client (the */5 job runs in a different thread
    than run_monitor_loop) and calls _advance_exit_ladders.  All
    concurrency / no-op safety lives in _advance_exit_ladders itself.
    """
    if PAPER_TRADE:
        return 0
    try:
        from execution import get_clob_client
        client = get_clob_client()
        if client is None:
            return 0
        return _advance_exit_ladders(client)
    except Exception as e:
        logger.exception(f"run_exit_ladder_fast failed (non-fatal): {e}")
        return 0


# ---------------------------------------------------------------------------
# Step 1c — Externally-cancelled topup pointer cleanup (safety net)
#
# Polymarket sometimes cancels resting orders on its own (account-level
# risk checks, WS auth disconnects, manual UI cancels).  When this happens
# the bot should receive a WS CANCELLATION event — but the WS path is
# unreliable enough that we miss them periodically.  The result: the DB
# carries a `pending_topup_order_id` pointing at a dead order forever,
# and `_run_topups` thinks a topup is still in flight (so it never queues
# another).
#
# This function runs every 5 minutes alongside the exit ladder.  For
# every position with a non-null pending_topup pointer, it asks the CLOB
# directly: is this order still live?  If the CLOB says CANCELED (or
# returns None for "doesn't exist"), the function:
#   * marks the position_orders ledger row 'cancelled' with reason
#     'cancelled_externally'
#   * clears pending_topup_order_id / amount / intended_price on parent
#   * appends a REPAIR audit-log entry
#
# Mirrors the manual cleanup the operator had to run for the Ankara/Lagos
# orphan pointers on 2026-04-30 — but does it automatically.
# ---------------------------------------------------------------------------

def detect_externally_cancelled_topups(client) -> int:
    """Sweep positions with non-null pending_topup_order_id, drop pointers
    whose underlying CLOB order is no longer live (CANCELED or vanished).

    Paper-mode safe (no-op).  Returns count of pointers cleaned up.
    """
    if PAPER_TRADE or client is None:
        return 0

    from db import (
        _get_conn, update_position_order_status,
        clear_position_topup_pending,
    )
    from execution import get_order_status
    from activity import log_activity

    with _get_conn() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT id, city, date, pending_topup_order_id,
                   pending_topup_amount_usdc
            FROM positions
            WHERE pending_topup_order_id IS NOT NULL
              AND COALESCE(is_paper, 0) = 0
        """).fetchall()]

    if not rows:
        return 0

    cleaned = 0
    for r in rows:
        pid = r["id"]
        oid = r["pending_topup_order_id"]
        try:
            stat = get_order_status(oid, client)
        except Exception as e:
            logger.debug(
                f"[CLEANUP] CLOB lookup for {oid[:14]} failed (non-fatal): {e}"
            )
            continue

        # Decide whether the order is dead.  CLOB returns:
        #   None                  → order doesn't exist (cancelled long ago,
        #                            or never made it to the book)
        #   {'status': 'CANCELED'} → cancelled by Polymarket / by user / WS-missed
        # Anything else (LIVE, MATCHED, FILLED, etc.) means it's still in
        # play and we leave the pointer alone.
        is_dead = False
        reason = ""
        if stat is None:
            is_dead = True
            reason = "clob_returned_none"
        elif isinstance(stat, dict):
            s = (stat.get("status") or "").upper()
            if s in ("CANCELED", "CANCELLED"):
                is_dead = True
                reason = f"clob_status={s}"

        if not is_dead:
            continue

        # Apply cleanup — both halves wrapped individually so a failure
        # in one doesn't block the other.
        try:
            update_position_order_status(
                order_id         = oid,
                status           = "cancelled",
                cancelled_reason = "cancelled_externally",
                closed           = True,
            )
        except Exception as e:
            logger.debug(
                f"[CLEANUP] ledger mark failed for {oid[:14]} (non-fatal): {e}"
            )
        try:
            clear_position_topup_pending(pid)
        except Exception as e:
            logger.debug(
                f"[CLEANUP] clear-pending failed for pid={pid} (non-fatal): {e}"
            )
        try:
            log_activity(
                "REPAIR",
                f"cleared stale pending_topup_order_id={oid[:14]} "
                f"(CLOB confirmed cancelled, reason={reason})",
                level="INFO",
                position_id=pid,
                repair_kind="externally_cancelled_topup",
                stale_order_id=oid,
                clob_reason=reason,
                stale_amount_usdc=float(r.get("pending_topup_amount_usdc") or 0),
            )
        except Exception:
            pass
        cleaned += 1

    if cleaned > 0:
        logger.log(
            SUMMARY_LEVEL,
            f"[MONITOR] {cleaned} externally-cancelled topup pointer(s) "
            f"cleaned up (CLOB safety-net poll)"
        )
    return cleaned


def run_orphan_topup_cleanup_fast() -> int:
    """Fast-cycle wrapper for the */5 minute APScheduler job.  Same shape
    as run_exit_ladder_fast — acquires the CLOB client and calls the
    cleanup, swallowing any exception so the scheduler job doesn't crash.
    """
    if PAPER_TRADE:
        return 0
    try:
        from execution import get_clob_client
        client = get_clob_client()
        if client is None:
            return 0
        return detect_externally_cancelled_topups(client)
    except Exception as e:
        logger.exception(
            f"run_orphan_topup_cleanup_fast failed (non-fatal): {e}"
        )
        return 0


# ---------------------------------------------------------------------------
# Step 1d — Stale topup re-pricing (Lightweight Option B)
#
# When a topup's limit was set at minute 0 and asks have since moved UP
# beyond TOPUP_REPRICE_THRESHOLD_CENTS, our resting bid is no longer
# competitive — sellers won't cross down to it.  This function cancels +
# re-issues the topup at the fresh best_ask + walk price so the position
# can keep filling.
#
# Direction-aware: only fires when asks moved AWAY from us (upward drift
# > threshold).  Asks moving DOWN doesn't need action — Polymarket's
# matching engine fills us at the cheaper price by default.
#
# Idempotent across concurrent invocations via an internal lock.
# ---------------------------------------------------------------------------

_refresh_stale_topups_lock = _threading.Lock()


def refresh_stale_topups(client) -> int:
    """Cancel + re-issue topup orders whose resting limit has gone stale
    relative to the live best_ask.

    Paper-mode safe (no-op).  Returns count of topups re-priced.
    """
    if PAPER_TRADE or client is None:
        return 0

    from db import _get_conn, get_committed_usdc
    from execution import (
        cancel_order, execute_topup, get_orderbook_snapshot,
    )
    from config import TOPUP_REPRICE_THRESHOLD_CENTS

    if TOPUP_REPRICE_THRESHOLD_CENTS <= 0:
        return 0   # disabled

    if not _refresh_stale_topups_lock.acquire(blocking=False):
        logger.debug(
            "[REPRICE] refresh_stale_topups already running — skipping invocation"
        )
        return 0
    try:
        with _get_conn() as conn:
            rows = [dict(r) for r in conn.execute("""
                SELECT id, contract_id, side, yes_token_id, no_token_id,
                       pending_topup_order_id, pending_topup_intended_price,
                       target_size_usdc, city, date
                FROM positions
                WHERE pending_topup_order_id IS NOT NULL
                  AND COALESCE(is_paper, 0) = 0
            """).fetchall()]

        if not rows:
            return 0

        refreshed = 0
        for pos in rows:
            pid       = pos["id"]
            old_oid   = pos["pending_topup_order_id"]
            old_limit = float(pos.get("pending_topup_intended_price") or 0)
            if old_limit <= 0:
                continue
            side      = pos.get("side", "YES")
            token_id  = (pos.get("yes_token_id") if side == "YES"
                         else pos.get("no_token_id"))
            if not token_id:
                continue

            # Fetch fresh orderbook
            snap = get_orderbook_snapshot(client, token_id)
            if snap is None or snap.get("best_ask") is None:
                continue   # no book data this tick → leave alone, retry next cycle
            best_ask = float(snap["best_ask"])

            # Direction-aware drift: only fire when ask moved UP beyond threshold.
            # Negative drift (ask went down) means our bid is at or above the
            # current ask — Polymarket's match engine fills us at the better
            # price; no action needed.
            #
            # Epsilon (1e-6) absorbs floating-point noise from the price
            # subtraction so a drift that is mathematically equal to the
            # threshold doesn't tip over due to fp imprecision (e.g.
            # (0.315 - 0.300) * 100 = 1.5000000000000013, not 1.5).
            drift_cents = (best_ask - old_limit) * 100
            if drift_cents <= TOPUP_REPRICE_THRESHOLD_CENTS + 1e-6:
                continue

            # Cancel the stale order.  Our cancel_order patches both the
            # ledger row (status='cancelled') AND clears the parent's
            # pending_topup_order_id, so the next execute_topup sees
            # "no pending" and proceeds with the fresh placement.
            ok = cancel_order(old_oid, client)
            if not ok:
                logger.warning(
                    f"[REPRICE] pid={pid} {pos.get('city','?')} cancel of "
                    f"{old_oid[:12]} failed — leaving stale topup in place"
                )
                continue

            # Re-fetch position state after the cancel cleared pending_topup_*.
            # The fresh row tells execute_topup we're not duplicating an
            # in-flight topup.
            with _get_conn() as conn:
                fresh_row = conn.execute(
                    "SELECT * FROM positions WHERE id = ?", (pid,),
                ).fetchone()
            if fresh_row is None:
                continue
            fresh_pos = dict(fresh_row)

            # Recompute the gap from scratch (committed_usdc reflects the
            # cancel — the cancelled topup no longer counts).
            target    = float(fresh_pos.get("target_size_usdc") or 0)
            committed = get_committed_usdc(pid)
            remaining = target - committed
            if remaining < 1.0:
                # Gap closed by something else (drift, manual fill).  Nothing to add.
                logger.debug(
                    f"[REPRICE] pid={pid} gap closed during cancel — no re-issue"
                )
                continue

            result = execute_topup(fresh_pos, remaining, client=client)
            status = result.get("status")
            if status in ("placed", "paper"):
                refreshed += 1
                new_limit = result.get("limit_price")
                logger.warning(
                    f"[REPRICE] pid={pid} {pos.get('city','?')} {pos.get('date','?')} "
                    f"topup re-issued: old_limit=${old_limit:.4f}, "
                    f"new_ask=${best_ask:.4f}, drift={drift_cents:.1f}¢, "
                    f"gap=${remaining:.2f}"
                    + (f", new_limit=${float(new_limit):.4f}" if new_limit else "")
                )
            else:
                # 'skip' (book too thin now) or 'failed' — pointer is already
                # cleared by cancel_order, next /15 min trading_run will retry
                logger.debug(
                    f"[REPRICE] pid={pid} re-issue skipped: "
                    f"status={status} reason={result.get('reason')}"
                )

        if refreshed > 0:
            logger.log(SUMMARY_LEVEL,
                f"[MONITOR] {refreshed} stale topup(s) re-priced "
                f"(drift > {TOPUP_REPRICE_THRESHOLD_CENTS}¢)"
            )
        return refreshed
    finally:
        _refresh_stale_topups_lock.release()


def run_stale_topup_refresh_fast() -> int:
    """Fast-cycle wrapper for the */5 minute APScheduler job."""
    if PAPER_TRADE:
        return 0
    try:
        from execution import get_clob_client
        client = get_clob_client()
        if client is None:
            return 0
        return refresh_stale_topups(client)
    except Exception as e:
        logger.exception(
            f"run_stale_topup_refresh_fast failed (non-fatal): {e}"
        )
        return 0


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
        from activity import log_activity
        log_activity(
            "CLOSE", position_id=pos_id,
            level="INFO" if our_side_won else "WARN",
            message=(
                f"market resolved {result_label}: {side} {city} {date} "
                f"{contract_id[:12]} entry={entry_price:.4f} "
                f"exit={exit_price:.2f} shares={shares:.2f} pnl=${pnl:+.4f}"
            ),
            won=our_side_won, side=side, entry_price=entry_price,
            exit_price=exit_price, shares=shares, pnl=pnl,
            mode="paper" if is_paper else "live",
        )

    return closed_count


# ---------------------------------------------------------------------------
# Step 2b — On-chain reconciliation (log-only)
# ---------------------------------------------------------------------------

# Tolerance for share-count drift between DB and on-chain.  Anything below
# this is treated as floating-point/rounding noise; anything above is logged.
_RECONCILE_SHARE_TOLERANCE = 0.5


def _reconcile_onchain(data_api_index: dict) -> dict:
    """Compare DB live positions against on-chain positions and log drift.

    Three drift classes:
      * orphan_db    — DB has it open, chain has no balance for that token
      * share_drift  — both have it, |db_shares - chain_size| > tolerance
      * orphan_chain — chain has a token that no open DB row references

    Pure log-only.  Does not mutate state — by design, per the user's
    "log only" call.  If the bot's view diverges from on-chain, a human
    decides what to do.

    Skipped when:
      * data_api_index is empty (paper mode, missing wallet, or fetch failed
        — we can't tell drift apart from "couldn't reach the chain")

    Returns {orphan_db, share_drift, orphan_chain} counts for the summary.
    """
    if not data_api_index:
        return {"orphan_db": 0, "share_drift": 0, "orphan_chain": 0}

    # Live positions that should have an on-chain footprint.  Top-up state
    # ('exiting' status) still has shares on chain until the sell fills, so
    # include those too.
    live_positions = [
        p for p in get_open_positions()
        if not bool(p.get("is_paper", 1))
        and (p.get("fill_status") or "") == "filled"
    ]

    db_token_ids: set[str] = set()
    orphan_db = 0
    share_drift = 0

    for pos in live_positions:
        side      = pos.get("side", "YES")
        token_id  = pos.get("yes_token_id") if side == "YES" else pos.get("no_token_id")
        if not token_id:
            # Can't reconcile without a token_id — these are typically very
            # old positions placed before yes/no_token_id columns existed.
            continue
        db_token_ids.add(token_id)
        db_shares = float(pos.get("shares") or 0)
        on_chain  = data_api_index.get(token_id)

        if on_chain is None:
            orphan_db += 1
            from activity import log_activity
            log_activity(
                "DRIFT", level="WARN", position_id=pos["id"],
                message=(
                    f"DRIFT orphan_db: {pos.get('city')} {pos.get('date')} "
                    f"{side} db_shares={db_shares:.4f} but no on-chain "
                    f"balance for token={token_id[:14]}..."
                ),
                drift_kind="orphan_db", token_id=token_id,
                db_shares=db_shares,
            )
            continue

        chain_size = float(on_chain.get("size") or 0)
        delta = abs(db_shares - chain_size)
        if delta > _RECONCILE_SHARE_TOLERANCE:
            share_drift += 1
            from activity import log_activity
            log_activity(
                "DRIFT", level="WARN", position_id=pos["id"],
                message=(
                    f"DRIFT share_drift: {pos.get('city')} {pos.get('date')} "
                    f"{side} db_shares={db_shares:.4f} "
                    f"chain_size={chain_size:.4f} delta={delta:.4f}"
                ),
                drift_kind="share_drift", token_id=token_id,
                db_shares=db_shares, chain_size=chain_size, delta=delta,
            )

    # Tokens on-chain that no open DB row claims.  These are usually
    # positions held from outside the bot (manual trades, prior strategies)
    # — log at INFO since they're informational, not actionable.
    orphan_chain = 0
    for token_id, on_chain in data_api_index.items():
        if token_id in db_token_ids:
            continue
        chain_size = float(on_chain.get("size") or 0)
        if chain_size <= _RECONCILE_SHARE_TOLERANCE:
            continue
        orphan_chain += 1
        logger.info(
            f"[MONITOR] DRIFT orphan_chain: on-chain size={chain_size:.4f} "
            f"for token={token_id[:14]}... title='{on_chain.get('title', '')[:40]}' "
            f"— not tracked in DB (manual trade or pre-bot position?)"
        )

    if orphan_db or share_drift or orphan_chain:
        logger.log(SUMMARY_LEVEL,
            f"[MONITOR] Reconciliation: {orphan_db} orphan_db | "
            f"{share_drift} share_drift | {orphan_chain} orphan_chain"
        )
    return {
        "orphan_db":    orphan_db,
        "share_drift":  share_drift,
        "orphan_chain": orphan_chain,
    }


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

    # Reconcile FIRST so we don't cancel orders that just filled or
    # advance ladders for sells whose original order already filled.
    buys_filled, sells_filled, topups_filled = _reconcile_pending_fills(client)
    cancelled = _cancel_pending_orders(client)
    overcommit_cancelled = _cancel_overcommitted_orders(client)
    advanced  = _advance_exit_ladders(client)
    closed    = _settle_resolved_positions(data_api_index, _status_cache)

    n_open = len([p for p in get_open_positions() if p.get("fill_status") == "filled"])
    if n_open > 0:
        logger.log(SUMMARY_LEVEL,
            f"Updating P&L for {n_open} positions "
            f"({len(_status_cache)} cached from settle phase)..."
        )
    updated   = _update_unrealized_pnl(data_api_index, _status_cache)

    # On-chain reconciliation runs LAST — after settle has had a chance to
    # close anything chain-side that was already resolved, so we don't
    # false-flag positions mid-resolution as orphan_db.
    drift = _reconcile_onchain(data_api_index)

    # Scan for resolved events we didn't trade (for backtesting data)
    resolutions_found = _scan_event_resolutions()

    summary = {
        "buys_filled":          buys_filled,
        "sells_filled":         sells_filled,
        "topups_filled":        topups_filled,
        "cancelled":            cancelled,
        "overcommit_cancelled": overcommit_cancelled,
        "advanced":             advanced,
        "closed":               closed,
        "updated":              updated,
        "resolutions":          resolutions_found,
        "drift":                drift,
    }
    drift_total = drift["orphan_db"] + drift["share_drift"] + drift["orphan_chain"]
    overcommit_part = (
        f" | {overcommit_cancelled} over-commit cancelled"
        if overcommit_cancelled else ""
    )
    summary_text = (
        f"Monitor complete: {buys_filled} buys filled | {sells_filled} sells filled | "
        f"{topups_filled} top-ups filled | {cancelled} cancelled"
        f"{overcommit_part} | "
        f"{advanced} exit ladders advanced | {closed} resolved | {updated} P&L updated"
        f" | {drift_total} drift"
    )
    logger.log(SUMMARY_LEVEL, summary_text)

    # Snapshot for the dashboard health strip.  Best-effort — never raise
    # if a probe fails (e.g. WS module unavailable in some environments).
    try:
        from db import insert_monitor_health, get_pending_positions, get_exiting_positions
        try:
            from user_ws import is_running as _ws_is_running
            ws_running = bool(_ws_is_running()) if not PAPER_TRADE else None
        except Exception:
            ws_running = None
        try:
            from wallet import get_wallet_usdc_balance, get_effective_bankroll
            wallet_balance = get_wallet_usdc_balance(client) if not PAPER_TRADE else None
            effective_bankroll = get_effective_bankroll(client) if not PAPER_TRADE else None
        except Exception:
            wallet_balance, effective_bankroll = None, None
        n_open_filled = len([
            p for p in get_open_positions()
            if (p.get("fill_status") or "") == "filled" and (p.get("status") or "") == "open"
        ])
        n_pending = len(get_pending_positions())
        n_exiting = len(get_exiting_positions())
        insert_monitor_health({
            "recorded_at":              _now(),
            "ws_running":               (1 if ws_running else 0) if ws_running is not None else None,
            "wallet_balance_usdc":      wallet_balance,
            "effective_bankroll_usdc":  effective_bankroll,
            "drift_orphan_db":          drift["orphan_db"],
            "drift_share_drift":        drift["share_drift"],
            "drift_orphan_chain":       drift["orphan_chain"],
            "buys_filled":              buys_filled,
            "sells_filled":             sells_filled,
            "topups_filled":            topups_filled,
            "positions_open":           n_open_filled,
            "positions_pending":        n_pending,
            "positions_exiting":        n_exiting,
            "summary_text":             summary_text,
        })
    except Exception as _e:
        logger.debug(f"[MONITOR] monitor_health snapshot failed (non-fatal): {_e}")

    _phase_end()
    return summary
