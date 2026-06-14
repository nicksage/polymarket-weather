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
    skipped_ensure_fill = 0

    # Strategies whose entry orders must ultimately fill (not abandoned).
    # For these, the stale-entry repricer (run_stale_tkh_entry_refresh_fast)
    # cancels-and-replaces at the new best_ask instead of the cancel sweep
    # marking them dead.
    # IMPORTANT: keep in sync with _ENSURE_FILL_STRATEGIES at module-level
    # (used by refresh_stale_ensure_fill_entries).  If they drift apart,
    # an intraday_predictor entry would either be (a) cancelled by the
    # sweep before the repricer can chase it, or (b) repriced indefinitely
    # past expiry.
    _ENSURE_FILL_STRATEGIES = {"top_k_hedged", "intraday_predictor"}

    for pos in pending:
        pos_id   = pos["id"]
        order_id = pos.get("order_id")
        city     = pos.get("city", "")
        date     = pos.get("date", "")
        contract = pos.get("contract_id", "")[:12]
        entry    = pos.get("entry_time") or ""
        strategy = (pos.get("strategy") or "").strip()

        # Ensure-fill strategies (TKH): NEVER abandon a pending entry.
        # The stale-entry repricer chases the moving best_ask until the
        # order crosses; without this exemption the monitor would mark
        # a TKH bin "cancelled" the moment the market drifted past our
        # limit, and TKH per-event dedup would lock out re-entry.
        if strategy in _ENSURE_FILL_STRATEGIES:
            skipped_ensure_fill += 1
            logger.debug(
                f"[MONITOR] Skipping cancel of pos={pos_id} {city} {date} "
                f"{contract} -- strategy={strategy} is ensure-fill "
                f"(handled by stale-entry repricer instead)"
            )
            continue

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
    if skipped_ensure_fill > 0:
        logger.info(
            f"[MONITOR] Cancel pass skipped {skipped_ensure_fill} ensure-fill order(s) "
            f"(TKH etc.) -- handled by the stale-entry repricer"
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
            elif status == "skip":
                # Intentional skip from a gate inside execute_exit (let-it-decay
                # price floor or dust-shares floor).  Not an error -- the
                # position will be auto-closed when chain catches up.
                logger.info(
                    f"[MONITOR] pos={pid} ladder advance skipped: "
                    f"{result.get('reason','?')} ({result})"
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

            # PHASE 1 hardening (2026-06-13): pre-cancel partial-fill
            # capture, mirroring refresh_stale_ensure_fill_entries.
            # Closes the race where chunks crossed our resting topup
            # between the snapshot read and the cancel arriving at
            # Polymarket.  Without this, the captured fills would not
            # be reflected in committed_usdc when we recompute below,
            # causing the replacement topup to be sized as if those
            # chunks never landed — and the next fill stacks on top.
            # This is the same overrun pattern as the New York pid=107
            # bug, just on the topup path instead of the entry path.
            cid = pos.get("contract_id") or ""
            token_id_for_capture = (pos.get("yes_token_id") if side == "YES"
                                       else pos.get("no_token_id"))
            try:
                _capture_partial_fills_before_cancel(
                    client,
                    position_id = pid,
                    contract_id = cid,
                    token_id    = token_id_for_capture or "",
                    old_oid     = old_oid,
                )
            except Exception as _e:
                logger.warning(
                    f"[REPRICE] pid={pid} pre-cancel partial-fill capture "
                    f"raised: {_e}; proceeding anyway (committed_usdc will "
                    f"reflect whatever WS has delivered)"
                )

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
            # cancel + any partial fills captured above — the cancelled
            # topup no longer counts; any chunks that landed before the
            # cancel ARE counted via add_position_entry_fill).  This is
            # the overrun protection: if 80% of the topup actually filled
            # during the cancel race, `remaining` shrinks accordingly so
            # the replacement only chases the remaining 20%.
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
# Aggressive REST trade-fill polling (Option A — WS replacement).
#
# Polymarket's user-channel WebSocket has been confirmed degraded:
# connections succeed but trade events never arrive.  Until that's
# resolved upstream, the only authoritative path for fills is REST.
# The hourly safety-net poll (`_reconcile_pending_fills`) catches
# eventually, but a 1-hour latency on fills creates:
#   - drift snapshots in every monitor cycle (DB lags chain by up to 60min)
#   - stale dashboard P&L between fills and reconciliation
#   - delayed exit decisions (TAKE_PROFIT etc fire on stale `current_price`)
#
# This polling job runs every 2 minutes and pulls fresh trades from
# Polymarket via `client.get_trades(TradeParams(market, asset_id))` for
# every (market, token) pair we have an open position on.  Each new trade
# is dedup'd via `mark_event_processed(event.id)` and applied via the
# SAME `apply_trade_event` handler the WS uses -- so no path divergence,
# no double-counting if WS ever recovers.
#
# Cost: ~30 unique tokens × 30 polls/hour = 900 API calls/hour, well
# below CLOB's ~6 req/s rate limit (~21,600/hour).
# ---------------------------------------------------------------------------

_poll_trade_fills_lock = _threading.Lock()


def poll_trade_fills_via_get_trades(client) -> dict:
    """Pull fresh trades for every active (market, token) pair and apply
    any new chunks via apply_trade_event.  Returns a dict of counts."""
    if PAPER_TRADE or client is None:
        return {"polled_pairs": 0, "trades_seen": 0, "trades_applied": 0,
                "buys": 0, "sells": 0, "topups": 0, "duplicates": 0,
                "no_position": 0}

    if not _poll_trade_fills_lock.acquire(blocking=False):
        logger.debug(
            "[POLL-FILLS] poll_trade_fills_via_get_trades already running"
            " -- skipping invocation"
        )
        return {"polled_pairs": 0, "trades_seen": 0, "trades_applied": 0,
                "buys": 0, "sells": 0, "topups": 0, "duplicates": 0,
                "no_position": 0, "lock_busy": True}

    counts = {
        "polled_pairs":   0,
        "trades_seen":    0,
        "trades_applied": 0,
        "buys":           0,
        "sells":          0,
        "topups":         0,
        "duplicates":     0,
        "no_position":    0,
        "errors":         0,
    }
    try:
        from db import _get_conn
        from fill_handler import apply_trade_event
        from py_clob_client_v2.clob_types import TradeParams

        # Gather unique (market, token_id) pairs across every position with
        # a live or recently-active footprint.  Includes:
        #   - status='open' rows (entry, topup, exit-in-flight)
        #   - status='exiting' rows (sell ladder mid-flight)
        # Excludes paper rows.
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT contract_id, side, yes_token_id, no_token_id
                FROM positions
                WHERE status IN ('open', 'exiting')
                  AND COALESCE(is_paper, 0) = 0
                  AND contract_id IS NOT NULL
            """).fetchall()

        # Build token list (deduped).  YES and NO tokens for the same
        # market are different tokens -- include whichever side(s) we
        # actually hold positions on.
        pairs: set[tuple[str, str]] = set()
        for r in rows:
            market = r["contract_id"]
            side   = r["side"] or "YES"
            tok    = (r["yes_token_id"] if side == "YES"
                      else r["no_token_id"])
            if market and tok:
                pairs.add((market, tok))

        if not pairs:
            return counts

        # Wallet address used by apply_trade_event to disambiguate
        # maker_orders -- pass it through as the WS path does.
        my_wallet = (WALLET_ADDRESS or "").lower() or None

        for (market, token_id) in pairs:
            counts["polled_pairs"] += 1
            try:
                trades = client.get_trades(
                    params=TradeParams(market=market, asset_id=token_id),
                    only_first_page=False,
                ) or []
            except Exception as e:
                counts["errors"] += 1
                logger.warning(
                    f"[POLL-FILLS] get_trades failed for "
                    f"market={market[:12]} token={token_id[:14]}: {e}"
                )
                continue

            for t in trades:
                counts["trades_seen"] += 1
                # Synthesize the event shape apply_trade_event expects.
                # get_trades returns confirmed trades by definition (they
                # wouldn't appear in /trades otherwise), so status='confirmed'.
                event = dict(t)
                event["status"] = "confirmed"
                # Ensure the dedup id is present; Polymarket's trade dict
                # carries it as 'id'.  If missing (defensive), synthesize
                # from order_id + size to make duplicates collide.
                if not event.get("id"):
                    event["id"] = (
                        f"trade:{event.get('taker_order_id','?')}:"
                        f"{event.get('size','?')}:{event.get('price','?')}"
                    )

                result = apply_trade_event(event, my_wallet=my_wallet)
                action = result.get("action", "")
                if action == "filled":
                    counts["trades_applied"] += 1
                    role = (result.get("role") or "").lower()
                    if   role == "entry": counts["buys"]   += 1
                    elif role == "exit":  counts["sells"]  += 1
                    elif role == "topup": counts["topups"] += 1
                elif action == "ignored_duplicate_event":
                    counts["duplicates"] += 1
                elif action in ("ignored_no_position", "ignored_no_order_id"):
                    counts["no_position"] += 1

        if counts["trades_applied"] > 0:
            logger.log(SUMMARY_LEVEL,
                f"[POLL-FILLS] applied {counts['trades_applied']} new fill(s) "
                f"({counts['buys']} buy, {counts['sells']} sell, "
                f"{counts['topups']} topup) across "
                f"{counts['polled_pairs']} token(s); "
                f"{counts['duplicates']} dedup'd, "
                f"{counts['no_position']} foreign"
            )
        return counts
    finally:
        _poll_trade_fills_lock.release()


def run_trade_fill_poll_fast() -> dict:
    """Fast-cycle wrapper for the */2 minute APScheduler job."""
    if PAPER_TRADE:
        return {}
    try:
        from execution import get_clob_client
        client = get_clob_client()
        if client is None:
            return {}
        return poll_trade_fills_via_get_trades(client)
    except Exception as e:
        logger.exception(
            f"run_trade_fill_poll_fast failed (non-fatal): {e}"
        )
        return {}


# ---------------------------------------------------------------------------
# Stale ENTRY-ORDER repricer for ensure-fill strategies (TKH).
#
# TKH places its hedged-bin entries at best_ask + 1¢ as marketable BUYs.
# When the ask drifts up before our order matches, the order rests below
# the new best_ask and never crosses.  The legacy monitor sweep would
# then mark the position cancelled, abandoning the bin and (worse)
# locking the event out of re-entry forever via TKH per-event dedup.
#
# This refresher is the entry-order analogue of refresh_stale_topups:
# detect upward drift past ENTRY_REPRICE_THRESHOLD_CENTS, cancel the
# stale CLOB order WITHOUT triggering the position-cancel cleanup
# (which would mark the row dead), and re-place at the new best_ask + 1¢.
# The position row is then UPDATED in place with the new order_id /
# entry_price / entry_time, so target_size_usdc, ledger linkage, and
# event dedup all remain consistent.
#
# Idempotent across concurrent invocations via an internal lock.
# ---------------------------------------------------------------------------

_refresh_stale_entries_lock = _threading.Lock()

# Strategies whose entry orders should be chased by the stale-entry
# repricer until they fill (or the per-event/per-contract cap binds).
# Includes:
#   - "top_k_hedged"        — TKH bins MUST own every bin in the basket;
#                              without chasing, a missed bin breaks the
#                              hedge thesis.
#   - "intraday_predictor"  — Phase 1 repricer coverage (2026-06-13).
#                              The intraday loop's probability-mode buys
#                              were sitting at stale limits while ask
#                              prices drifted up.  Chasing them via the
#                              entry repricer (with its built-in
#                              partial-fill capture + DB re-read pattern)
#                              lets us shorten the cron without risking
#                              the Houston $74.99 over-allocation bug —
#                              every cancel/replace cycle re-checks
#                              committed_usdc before sizing the next
#                              order.
_ENSURE_FILL_STRATEGIES = {"top_k_hedged", "intraday_predictor"}


def _capture_partial_fills_before_cancel(
    client, *, position_id: int, contract_id: str,
    token_id: str, old_oid: str,
) -> tuple[float, float, float]:
    """Reconcile any silent partial fills on `old_oid` BEFORE cancelling.

    Race we're closing: between the snapshot get_order_status() reading
    and the actual cancel arriving at Polymarket, additional asks can
    cross our resting order.  Those fills are real on chain but their
    trade-event WS deliveries may be silently lost (the bot's user
    channel was confirmed degraded -- "REST is doing 100% of fills").
    Without this reconciliation, the ledger marks the cancelled row as
    filled=$0 even though chain holds N shares.  When the repricer then
    re-places the FULL gap, the new order's fill stacks on top of the
    silent partial -> over-allocation (the New York pid=107 bug).

    Algorithm:
      1. Fetch all trades for (market=contract_id, asset_id=token_id)
         from the CLOB REST endpoint.
      2. Filter to trades whose taker_order_id OR any maker_orders[].order_id
         matches `old_oid`.
      3. For each matching trade event with an `id` we haven't yet
         processed, dedupe via mark_event_processed and apply via
         add_position_entry_fill.  This mirrors the WS path exactly,
         so a late-arriving WS event for the same trade is a no-op.
      4. Return (total_filled_shares, total_filled_usdc, weighted_avg_price)
         so the caller can record them on the cancelled ledger row.

    Returns (0.0, 0.0, 0.0) on any failure -- safer to fall back to the
    pre-fix behaviour (treat the cancel as fully-unfilled) than to risk
    misreporting partials.  Worst case: we re-introduce the original
    over-allocation, which the repair_share_drift script can clean up.
    """
    if client is None or not old_oid or not token_id or not contract_id:
        return 0.0, 0.0, 0.0
    try:
        from py_clob_client_v2.clob_types import TradeParams
        trades = client.get_trades(
            params=TradeParams(market=contract_id, asset_id=token_id),
            only_first_page=False,
        ) or []
    except Exception as e:
        logger.warning(
            f"[REPRICE-ENTRY] pid={position_id} get_trades for partial-fill "
            f"reconciliation raised: {e}; treating as no-fill"
        )
        return 0.0, 0.0, 0.0

    from db import mark_event_processed, add_position_entry_fill

    total_shares = 0.0
    total_usdc   = 0.0
    n_chunks_applied = 0
    for t in trades:
        # Match: this trade's taker_order_id is ours, OR one of the
        # maker_orders is ours.  Same matching logic as
        # execution.backfill_position_fees.
        taker_oid = t.get("taker_order_id") or ""
        is_match = (taker_oid == old_oid)
        if not is_match:
            for mo in (t.get("maker_orders") or []):
                if mo.get("order_id") == old_oid:
                    is_match = True
                    break
        if not is_match:
            continue

        try:
            sz    = float(t.get("size") or 0)
            price = float(t.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if sz <= 0 or price <= 0:
            continue

        # Dedup at trade-event level using the same table the WS path uses.
        # If this trade's event has already been processed (e.g. WS delivered
        # it before we ran), skip — the position row is already up-to-date
        # for this chunk, so we just need to count it for the ledger total.
        evt_id = t.get("id") or ""
        is_first_time = mark_event_processed(evt_id) if evt_id else True

        if is_first_time:
            # WS hadn't delivered this -- apply it now so the position
            # row reflects the partial fill.
            try:
                add_position_entry_fill(
                    position_id  = position_id,
                    added_shares = sz,
                    fill_price   = price,
                )
                n_chunks_applied += 1
            except Exception as e:
                logger.warning(
                    f"[REPRICE-ENTRY] pid={position_id} add_position_entry_fill "
                    f"failed for trade chunk size={sz} price={price}: {e}"
                )
                # Don't include this chunk in totals if we couldn't apply
                # it — keeps ledger / position row consistent.
                continue

        # Always count toward ledger totals (whether or not WS already
        # applied it to the position row, the ledger row needs to reflect
        # the order's actual filled amount).
        total_shares += sz
        total_usdc   += sz * price

    if total_shares > 0 or n_chunks_applied > 0:
        avg_price = (total_usdc / total_shares) if total_shares > 0 else 0.0
        logger.warning(
            f"[REPRICE-ENTRY] pid={position_id} captured silent partial fills "
            f"on cancelled order {old_oid[:14]}: {total_shares:.4f} shares "
            f"@ avg ${avg_price:.4f} = ${total_usdc:.2f} "
            f"({n_chunks_applied} chunk(s) applied to position row)"
        )
        try:
            from activity import log_activity
            log_activity(
                "FILL", level="WARN", position_id=position_id,
                message=(
                    f"silent partial fill recovered before reprice cancel: "
                    f"{total_shares:.4f} sh @ ${avg_price:.4f} "
                    f"= ${total_usdc:.2f}"
                ),
                source="reprice_partial_fill_capture",
                old_order_id=old_oid,
                filled_shares=total_shares,
                filled_usdc=total_usdc,
                avg_price=avg_price,
                chunks_applied=n_chunks_applied,
            )
        except Exception:
            pass

    avg_price = (total_usdc / total_shares) if total_shares > 0 else 0.0
    return total_shares, total_usdc, avg_price


def refresh_stale_ensure_fill_entries(client) -> int:
    """Cancel + re-issue ensure-fill (TKH) entry orders whose resting
    limit has fallen behind the live best_ask.  Returns count repriced."""
    if PAPER_TRADE or client is None:
        return 0

    from db import _get_conn
    from execution import get_orderbook_snapshot, compute_sweep_limit
    from config import ORDERBOOK_WALK_CENTS

    # Reprice threshold (cents) — re-place when ask drifted up by at least
    # this many cents.  Uses the same env var as topup repricing so the
    # operator only has one knob to tune.
    import os
    THRESHOLD = float(os.getenv("ENTRY_REPRICE_THRESHOLD_CENTS",
                                os.getenv("TOPUP_REPRICE_THRESHOLD_CENTS", "1.5")))
    if THRESHOLD <= 0:
        return 0

    if not _refresh_stale_entries_lock.acquire(blocking=False):
        logger.debug(
            "[REPRICE-ENTRY] refresh_stale_ensure_fill_entries already running -- skipping invocation"
        )
        return 0
    try:
        # Pull all live entries belonging to ensure-fill strategies that
        # could still have resting capacity on their original entry order.
        # Two flavours qualify:
        #   - fill_status='pending' AND shares=0   -> fully unfilled (no chunks landed yet)
        #   - fill_status='filled'  AND shares>0   -> partial fill landed,
        #                                              but original order may
        #                                              still have a resting
        #                                              remainder we need to chase.
        # The CLOB get_order_status check below filters out fully-filled
        # rows (no resting capacity → nothing to reprice).
        strategy_placeholders = ",".join("?" * len(_ENSURE_FILL_STRATEGIES))
        with _get_conn() as conn:
            rows = [dict(r) for r in conn.execute(f"""
                SELECT id, contract_id, side, yes_token_id, no_token_id,
                       order_id, entry_price, entry_time, size_usdc, shares,
                       target_size_usdc, strategy, city, date
                FROM positions
                WHERE strategy IN ({strategy_placeholders})
                  AND status = 'open'
                  AND fill_status IN ('pending', 'filled')
                  AND COALESCE(is_paper, 0) = 0
                  AND order_id IS NOT NULL
            """, tuple(_ENSURE_FILL_STRATEGIES)).fetchall()]

        if not rows:
            return 0

        from py_clob_client_v2 import OrderArgs, OrderType, Side
        from execution import get_order_status
        from db import get_committed_usdc, update_position_order_status, insert_position_order
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI

        repriced = 0
        skipped_drift = 0
        skipped_no_resting = 0
        for pos in rows:
            pid       = pos["id"]
            old_oid   = pos["order_id"]
            old_limit = float(pos.get("entry_price") or 0)
            side      = pos.get("side", "YES")
            token_id  = (pos.get("yes_token_id") if side == "YES"
                         else pos.get("no_token_id"))
            cur_shares  = float(pos.get("shares") or 0)
            target      = float(pos.get("target_size_usdc") or pos.get("size_usdc") or 0)
            if not token_id or old_limit <= 0 or target <= 0:
                continue

            # ---- Determine if the OLD entry order still has resting
            # capacity on the book.  This is the bridge between "no fill"
            # and "partial fill" cases — both need repricing iff the
            # original order is still LIVE/MATCHED with size_matched <
            # original_size.  A fully-filled or already-cancelled order
            # has no resting capacity and is skipped silently.
            order_stat = get_order_status(old_oid, client)
            if order_stat is None:
                # CLOB query failed — safer to skip this cycle than risk
                # double-placement.  Next 5-min tick will retry.
                continue
            stat_str = (order_stat.get("status") or "").upper()
            sz_match = float(order_stat.get("size_matched") or 0)
            sz_orig  = float(order_stat.get("original_size") or order_stat.get("size") or 0)
            has_resting = (
                stat_str in ("LIVE", "MATCHED", "DELAYED", "PARTIAL")
                and sz_orig > 0
                and sz_match < sz_orig - 1e-9
            )
            if not has_resting:
                skipped_no_resting += 1
                continue

            # Fresh book check — skip silently when book is unavailable.
            snap = get_orderbook_snapshot(client, token_id)
            if snap is None or snap.get("best_ask") is None:
                continue
            best_ask = float(snap["best_ask"])

            # Direction-aware drift: only fire when ask moved UP.  If ask
            # came DOWN to or past our limit, Polymarket's matcher fills
            # us at the cheaper price — no action needed.
            drift_cents = (best_ask - old_limit) * 100
            if drift_cents <= THRESHOLD + 1e-6:
                skipped_drift += 1
                continue

            # ---- Step 0: Capture any silent partial fills BEFORE cancel.
            # Closes the race where chunks crossed our resting order
            # between the get_order_status snapshot and the cancel arriving
            # at Polymarket.  Without this the ledger marks the cancel as
            # filled=$0 even though chain holds N shares -> the new order
            # below stacks on top -> over-allocation (the New York pid=107
            # bug).  Helper applies any unprocessed chunks to the position
            # row and returns the totals to record on the ledger.
            cid = pos.get("contract_id") or ""
            partial_shares, partial_usdc, partial_avg = (
                _capture_partial_fills_before_cancel(
                    client,
                    position_id = pid,
                    contract_id = cid,
                    token_id    = token_id,
                    old_oid     = old_oid,
                )
            )

            # ---- Step 1: Cancel old CLOB order WITHOUT touching position row.
            # We deliberately bypass execution.cancel_order() because its
            # built-in entry-order cleanup would mark fill_status='cancelled'
            # / status='closed', killing the bin we're trying to keep alive.
            try:
                resp = client.cancel_orders([old_oid])
                if not resp:
                    logger.warning(
                        f"[REPRICE-ENTRY] pid={pid} cancel returned falsy -- skipping"
                    )
                    continue
            except Exception as e:
                logger.error(f"[REPRICE-ENTRY] pid={pid} cancel raised: {e}")
                continue

            # Mark the OLD ledger row terminal so committed_usdc reflects
            # reality: any partial fill we captured stays counted as
            # filled (NOT cancelled), and only the resting remainder is
            # released.  When committed = filled_chunks + new_pending,
            # the gap calc below correctly subtracts the partial before
            # placing the replacement order.
            try:
                update_position_order_status(
                    order_id         = old_oid,
                    status           = "cancelled",
                    filled_shares    = (partial_shares if partial_shares > 0 else None),
                    filled_usdc      = (round(partial_usdc, 4) if partial_usdc > 0 else None),
                    fill_price       = (partial_avg if partial_avg > 0 else None),
                    cancelled_reason = "repriced_by_stale_entry_refresh",
                    closed           = True,
                )
            except Exception as _e:
                logger.debug(f"[REPRICE-ENTRY] ledger cancel-mark failed: {_e}")

            # ---- Step 2: Compute new sweep limit at current ask + walk.
            # Cap at 0.99 (Polymarket's max BUY price).
            new_limit = round(min(best_ask + ORDERBOOK_WALK_CENTS / 100.0, 0.99), 4)

            # ---- Compute intended size as the REMAINING gap.
            # For a fully-unfilled row (shares==0), committed after the
            # cancel is ~0 → intended_size = full target.  For a partial
            # fill, committed after the cancel is just the filled portion
            # (~size_usdc of the parent) → intended_size = target - filled.
            committed_after_cancel = get_committed_usdc(pid)
            intended_size = round(max(0.0, target - committed_after_cancel), 2)
            if intended_size < 1.0:
                logger.info(
                    f"[REPRICE-ENTRY] pid={pid} {pos.get('city','?')} "
                    f"{pos.get('date','?')} remaining gap "
                    f"${intended_size:.2f} < $1 minimum -- bin already "
                    f"satisfied (target=${target:.2f}, committed="
                    f"${committed_after_cancel:.2f}); skipping re-place"
                )
                continue

            # ---- Step 3: Place the new order.
            try:
                order_args = OrderArgs(
                    price    = new_limit,
                    size     = intended_size / new_limit,
                    side     = Side.BUY,
                    token_id = token_id,
                )
                response = client.create_and_post_order(order_args, order_type=OrderType.GTC)
            except Exception as e:
                logger.error(
                    f"[REPRICE-ENTRY] pid={pid} {pos.get('city','?')} "
                    f"{pos.get('date','?')} placement raised: {e}"
                )
                # Position row still has fill_status='pending' but its
                # order_id now points at a dead order.  Next cycle of THIS
                # refresher will retry (the SELECT only requires order_id
                # not null — a stale id is fine since it'll be cancelled
                # again as a no-op).  Better: clear the order_id so the
                # next trading_run picks the bin up via topup once filled
                # share count is positive (it's not, here, so just retry).
                continue

            if not response or not response.get("success"):
                logger.error(
                    f"[REPRICE-ENTRY] pid={pid} {pos.get('city','?')} "
                    f"{pos.get('date','?')} re-place FAILED -- response={response}"
                )
                continue

            new_oid = response.get("orderID", "")
            new_entry_time = _dt.now(_ZI("America/Chicago")).isoformat()
            # Treat as partial-fill if EITHER cur_shares (the row's pre-capture
            # snapshot) was > 0 OR the partial-fill capture above just added
            # shares to the row.  Without the second clause, a silent partial
            # fill caught by the capture would be erroneously treated as a
            # no-fill and overwrite entry_price/fill_status, blowing away the
            # weighted-avg cost basis we just established.
            is_partial_fill = (cur_shares > 0) or (partial_shares > 0)

            # ---- Step 4: Update the position row in place.
            #
            # Two flavours, branching on whether ANY shares have already
            # confirmed for this position:
            #
            # No-fill (cur_shares == 0):
            #     The original entry never crossed.  Replace order_id,
            #     entry_price, entry_time; fill_status stays 'pending'.
            #
            # Partial-fill (cur_shares > 0):
            #     The first chunk(s) already landed at the original ask.
            #     entry_price is the weighted-avg cost basis of the
            #     filled chunks -- DO NOT overwrite.  fill_status is
            #     already 'filled' and must stay that way.  We only swap
            #     the order_id (so the next fill chunk routes to this
            #     position via apply_order_event) and update entry_time
            #     to mark the latest re-issue.
            with _get_conn() as conn:
                if is_partial_fill:
                    conn.execute(
                        "UPDATE positions SET order_id = ?, entry_time = ? "
                        "WHERE id = ?",
                        (new_oid, new_entry_time, pid),
                    )
                else:
                    conn.execute(
                        "UPDATE positions SET order_id = ?, entry_price = ?, "
                        "entry_time = ?, fill_status = 'pending' "
                        "WHERE id = ?",
                        (new_oid, new_limit, new_entry_time, pid),
                    )

            # ---- Step 5: Insert new ledger row for the replacement order.
            try:
                insert_position_order(
                    position_id     = pid,
                    order_id        = new_oid,
                    role            = "entry",
                    intended_usdc   = intended_size,
                    intended_shares = intended_size / new_limit,
                    limit_price     = new_limit,
                    status          = "pending",
                    trade_status    = None,
                )
            except Exception as _e:
                logger.debug(f"[REPRICE-ENTRY] ledger insert failed: {_e}")

            repriced += 1
            kind = "partial-fill remainder" if is_partial_fill else "no-fill entry"
            logger.warning(
                f"[REPRICE-ENTRY] pid={pid} {pos.get('city','?')} "
                f"{pos.get('date','?')} {kind} re-placed: "
                f"old_limit=${old_limit:.4f}, new_ask=${best_ask:.4f}, "
                f"drift={drift_cents:.1f}c, new_limit=${new_limit:.4f}, "
                f"gap=${intended_size:.2f}, new_order={new_oid[:12]}"
            )
            try:
                from activity import log_activity
                log_activity(
                    "BUY", level="INFO", position_id=pid,
                    message=(
                        f"TKH {kind} repriced: old=${old_limit:.4f} -> "
                        f"new=${new_limit:.4f} (best_ask=${best_ask:.4f}, "
                        f"drift={drift_cents:.1f}c, gap=${intended_size:.2f}) "
                        f"{pos.get('city','?')} {pos.get('date','?')}"
                    ),
                    source="stale_entry_refresh",
                    old_order_id=old_oid, new_order_id=new_oid,
                    old_limit=old_limit, new_limit=new_limit,
                    drift_cents=drift_cents,
                    gap_usdc=intended_size,
                    is_partial_fill=is_partial_fill,
                )
            except Exception:
                pass

        if repriced > 0:
            logger.log(SUMMARY_LEVEL,
                f"[MONITOR] {repriced} stale ensure-fill entry order(s) "
                f"re-priced (drift > {THRESHOLD}c)"
            )
        return repriced
    finally:
        _refresh_stale_entries_lock.release()


def run_stale_entry_refresh_fast() -> int:
    """Fast-cycle wrapper for the */5 minute APScheduler job — repricing
    of TKH (and other ensure-fill) entry orders that have fallen behind
    the live best_ask."""
    if PAPER_TRADE:
        return 0
    try:
        from execution import get_clob_client
        client = get_clob_client()
        if client is None:
            return 0
        return refresh_stale_ensure_fill_entries(client)
    except Exception as e:
        logger.exception(
            f"run_stale_entry_refresh_fast failed (non-fatal): {e}"
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


def _bump_orphan_db_counter(pid: int) -> int:
    """Increment positions.orphan_db_cycles and return the new value.
    Used by the multi-cycle confirmation gate before auto-close fires."""
    from db import _get_conn
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE positions "
                "SET orphan_db_cycles = COALESCE(orphan_db_cycles, 0) + 1 "
                "WHERE id = ?",
                (pid,),
            )
            row = conn.execute(
                "SELECT COALESCE(orphan_db_cycles, 0) AS n "
                "FROM positions WHERE id = ?", (pid,),
            ).fetchone()
        return int(row["n"] if row else 0)
    except Exception as e:
        logger.warning(f"[ORPHAN-DB] counter bump failed pid={pid}: {e}")
        return 0


def _reset_orphan_db_counter(pid: int) -> None:
    """Reset positions.orphan_db_cycles to 0.  Called whenever the chain
    shows shares for this position again -- the orphan condition cleared
    on its own (e.g., late-arriving Data API confirmation)."""
    from db import _get_conn
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE positions "
                "SET orphan_db_cycles = 0 "
                "WHERE id = ? AND COALESCE(orphan_db_cycles, 0) > 0",
                (pid,),
            )
    except Exception:
        pass   # best-effort -- a stale counter just delays auto-close


def _fetch_activity_history(wallet: str, max_pages: int = 30) -> list[dict]:
    """Pull the wallet's full activity history from the Polymarket Data
    API (paginated).  Used by _compute_realized_pnl_from_activity to
    determine the ACTUAL realized P&L for orphan_db positions before
    auto-closing them.

    Returns [] on any error (caller falls back to the legacy
    "assume total loss" path).
    """
    if not wallet:
        return []
    import httpx as _httpx
    out: list[dict] = []
    offset = 0
    PAGE = 100
    for _ in range(max_pages):
        try:
            r = _httpx.get(
                "https://data-api.polymarket.com/activity",
                params={"user": wallet, "limit": PAGE, "offset": offset},
                timeout=20,
            )
            r.raise_for_status()
            page = r.json()
        except Exception as e:
            logger.warning(
                f"[ORPHAN-CLOSE] activity fetch at offset={offset} failed: {e}"
            )
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return out


def _compute_realized_pnl_from_activity(
    pos: dict, activity: list[dict],
) -> tuple[float, str]:
    """For an orphan_db position, walk the wallet's activity history to
    determine the ACTUAL realized P&L.  Returns (pnl, classification).

    Classifications:
      - 'never_filled':            no BUY trades for this token; assume
                                    pnl=$0 (placement never confirmed
                                    on chain, no capital deployed)
      - 'computed_from_activity':  realized = sells + redeems - buys
      - 'no_activity_data':        couldn't fetch activity; legacy
                                    fallback (caller uses -size_usdc)

    The chain says 0 shares for this position.  If we never bought any,
    we never paid anything -- pnl=$0, not pnl=-size_usdc.  If we bought
    but didn't sell/redeem, the realized loss is the cost basis.  If we
    bought AND sold/redeemed, realized = whatever the trades imply.
    """
    if not activity:
        return -float(pos.get("size_usdc", 0) or 0), "no_activity_data"

    side       = pos.get("side", "YES")
    token_id   = (pos.get("yes_token_id") if side == "YES"
                  else pos.get("no_token_id")) or ""
    contract_id = pos.get("contract_id", "") or ""

    if not token_id and not contract_id:
        return -float(pos.get("size_usdc", 0) or 0), "no_token_id"

    sum_buy    = 0.0
    sum_sell   = 0.0
    sum_redeem = 0.0
    for it in activity:
        ttype = it.get("type", "")
        if ttype == "TRADE":
            asset = str(it.get("asset", "") or "")
            if asset != str(token_id):
                continue
            usdc = float(it.get("usdcSize", 0) or 0)
            tside = it.get("side", "")
            if tside == "BUY":
                sum_buy  += usdc
            elif tside == "SELL":
                sum_sell += usdc
        elif ttype == "REDEEM":
            cid = str(it.get("conditionId", "") or "")
            if cid != str(contract_id):
                continue
            sum_redeem += float(it.get("usdcSize", 0) or 0)

    if sum_buy == 0:
        # Position record says we placed an order, but no BUY trade ever
        # confirmed for this token.  No capital was actually deployed.
        return 0.0, "never_filled"

    realized = round(sum_sell + sum_redeem - sum_buy, 4)
    return realized, "computed_from_activity"


def _auto_close_orphan_db(pid: int, db_shares: float, size_usdc: float,
                           city: str, date_str: str, side: str,
                           token_id: str,
                           pos: dict | None = None,
                           activity: list[dict] | None = None) -> bool:
    """Mark an orphan_db position closed.

    Triggered when a position has been orphan_db for >=
    ORPHAN_DB_AUTO_CLOSE_CYCLES consecutive monitor cycles.  By then a
    transient Data API miss has been ruled out; the chain genuinely has
    no balance for this token.

    P&L computation (when `pos` and `activity` are passed):
      * Walks Polymarket /activity to find actual BUY/SELL/REDEEM events
        for this token.
      * realized_pnl = sells + redeems - buys.
      * If no BUY trades ever happened, pnl=$0 (placement never confirmed
        on chain -- no capital was actually deployed, so no real loss).

    Legacy fallback (when activity unavailable):
      * Assumes total loss: pnl = -size_usdc.

    Sets status='closed', fill_status='cancelled', shares=0,
    exit_time=now, pnl=<computed>, cancelled_reason='orphan_db_auto_close'.
    Audit-logged under category='REPAIR'.

    Returns True on success, False on DB error.
    """
    from db import _get_conn
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    # Compute true realized PnL when we have the data; fall back to
    # legacy "assume total loss" otherwise.
    if pos is not None and activity is not None:
        pnl, classification = _compute_realized_pnl_from_activity(pos, activity)
    else:
        pnl = round(-float(size_usdc or 0), 4)
        classification = "legacy_assumed_total_loss"
    try:
        with _get_conn() as conn:
            conn.execute("""
                UPDATE positions
                SET status           = 'closed',
                    fill_status      = 'cancelled',
                    shares           = 0,
                    exit_time        = ?,
                    pnl              = ?,
                    cancelled_reason = 'orphan_db_auto_close'
                WHERE id = ?
            """, (now_iso, pnl, pid))
        try:
            from activity import log_activity
            # Tailor the audit message to the PnL classification so the
            # operator can immediately tell "real loss" from "phantom".
            if classification == "never_filled":
                msg = (
                    f"orphan_db auto-closed: {city} {date_str} {side} "
                    f"db_shares={db_shares:.4f} -- placement never confirmed "
                    f"on chain (no BUY trades).  Marked closed at $0 "
                    f"(no capital actually deployed).  Token={token_id[:14]}..."
                )
            elif classification == "computed_from_activity":
                msg = (
                    f"orphan_db auto-closed: {city} {date_str} {side} "
                    f"db_shares={db_shares:.4f} but chain held 0.  Realized "
                    f"P&L computed from activity: ${pnl:+.2f}.  "
                    f"Token={token_id[:14]}..."
                )
            else:
                msg = (
                    f"orphan_db auto-closed: {city} {date_str} {side} "
                    f"db_shares={db_shares:.4f} but chain held 0 across "
                    f"multiple monitor cycles.  Marked closed at assumed "
                    f"total loss (pnl=${pnl:.2f}).  "
                    f"Token={token_id[:14]}..."
                )
            log_activity(
                "REPAIR", level="WARN", position_id=pid,
                message=msg,
                repair_kind="orphan_db_auto_close",
                pnl_classification=classification,
                db_shares=db_shares, size_usdc=size_usdc, pnl=pnl,
                token_id=token_id,
            )
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning(f"[ORPHAN-DB-CLOSE] write failed pid={pid}: {e}")
        return False


def _heal_share_drift_auto(
    pid: int, contract_id: str, token_id: str,
    chain_size: float, client,
) -> tuple[bool, float | None, float | None]:
    """Sync a position's shares + cost basis to chain truth.

    Replaces what `repair_share_drift.py --apply` did manually.  Walks
    the position's ledger orders (entry + topup), pulls trades for the
    market+token via CLOB get_trades, and recomputes:
      shares       = chain_size                       (Data API truth)
      entry_price  = sum(size*price) / sum(size)      (weighted avg from trades)
      size_usdc    = sum(size*price)                  (actual capital deployed)

    When the trade fetch fails (no client / API error / no matched
    trades), falls back to shares-only repair so we still converge on
    the chain truth even if cost basis stays approximate.

    Returns (ok, new_avg_price, new_size_usdc).  ok=False means we
    couldn't write anything (caller falls back to WARN logging).
    """
    from db import _get_conn
    new_avg = None
    new_usdc = None

    # ---- Try to fetch trades for cost-basis recompute ----
    if client is not None:
        try:
            from py_clob_client_v2.clob_types import TradeParams
            with _get_conn() as conn:
                ledger_orders = [r["order_id"] for r in conn.execute(
                    "SELECT order_id FROM position_orders "
                    "WHERE position_id = ? AND role IN ('entry', 'topup') "
                    "  AND order_id IS NOT NULL",
                    (pid,),
                ).fetchall()]
                if not ledger_orders:
                    legacy = conn.execute(
                        "SELECT order_id FROM positions WHERE id = ?",
                        (pid,),
                    ).fetchone()
                    if legacy and legacy["order_id"]:
                        ledger_orders = [legacy["order_id"]]
            our_oids: set[str] = set(ledger_orders)
            if our_oids and contract_id and token_id:
                trades = client.get_trades(
                    params=TradeParams(market=contract_id, asset_id=token_id),
                    only_first_page=False,
                ) or []
                tot_sh = 0.0
                tot_usd = 0.0
                for t in trades:
                    taker_oid = t.get("taker_order_id") or ""
                    is_match = (taker_oid in our_oids)
                    if not is_match:
                        for mo in (t.get("maker_orders") or []):
                            if mo.get("order_id") in our_oids:
                                is_match = True
                                break
                    if not is_match:
                        continue
                    try:
                        sz    = float(t.get("size") or 0)
                        price = float(t.get("price") or 0)
                    except (TypeError, ValueError):
                        continue
                    if sz <= 0 or price <= 0:
                        continue
                    tot_sh  += sz
                    tot_usd += sz * price
                if tot_sh > 0:
                    new_avg = round(tot_usd / tot_sh, 6)
                    new_usdc = round(tot_usd, 4)
        except Exception as e:
            logger.debug(
                f"[DRIFT-HEAL] cost-basis fetch failed pid={pid}: {e} "
                f"-- falling back to shares-only"
            )

    # ---- Apply the update ----
    try:
        with _get_conn() as conn:
            if new_avg is not None and new_usdc is not None:
                conn.execute(
                    "UPDATE positions "
                    "SET shares = ?, entry_price = ?, size_usdc = ? "
                    "WHERE id = ?",
                    (chain_size, new_avg, new_usdc, pid),
                )
            else:
                conn.execute(
                    "UPDATE positions SET shares = ? WHERE id = ?",
                    (chain_size, pid),
                )
        return True, new_avg, new_usdc
    except Exception as e:
        logger.warning(f"[DRIFT-HEAL] write failed pid={pid}: {e}")
        return False, None, None


def _reconcile_onchain(data_api_index: dict) -> dict:
    """Compare DB live positions against on-chain positions.

    Three drift classes:
      * orphan_db    — DB has it open, chain has no balance for that token
                       Behaviour: log at WARN (auto-close is too risky --
                       a transient Data API hiccup could close real positions).
      * share_drift  — both have it, |db_shares - chain_size| > tolerance
                       Behaviour: when DRIFT_AUTO_HEAL=True (default), sync
                       the position row to chain truth (shares +
                       cost basis from trades).  Otherwise log at WARN.
      * orphan_chain — chain has a token that no open DB row references
                       Behaviour: log at INFO -- usually pre-bot positions
                       or manual trades.

    Skipped when data_api_index is empty (paper mode / missing wallet /
    Data API fetch failed) -- can't distinguish drift from API failure.

    Returns {orphan_db, share_drift, orphan_chain, share_drift_healed,
    share_drift_unhealed} for the cycle summary.
    """
    if not data_api_index:
        return {
            "orphan_db": 0, "share_drift": 0, "orphan_chain": 0,
            "share_drift_healed": 0, "share_drift_unhealed": 0,
            "orphan_db_auto_closed": 0,
        }

    from config import (
        DRIFT_AUTO_HEAL,
        ORPHAN_DB_AUTO_CLOSE,
        ORPHAN_DB_AUTO_CLOSE_CYCLES,
    )
    auto_heal = bool(DRIFT_AUTO_HEAL)
    auto_close_orphans = bool(ORPHAN_DB_AUTO_CLOSE)
    orphan_close_threshold = int(ORPHAN_DB_AUTO_CLOSE_CYCLES)

    # Lazy-fetched once per cycle and reused across multiple orphan-db
    # auto-closes that fire in this same cycle.  None means "not fetched
    # yet"; [] means "tried but got nothing".
    _wallet_activity_cache: list[dict] | None = None

    # Lazy-acquire CLOB client once for cost-basis recomputes.  None if
    # paper mode or creds missing -- helper falls back to shares-only.
    client = None
    if auto_heal:
        try:
            from execution import get_clob_client
            client = get_clob_client()
        except Exception:
            client = None

    # Live positions that should have an on-chain footprint.  Top-up
    # state ('exiting' status) still has shares on chain until the sell
    # fills, so include those too.
    live_positions = [
        p for p in get_open_positions()
        if not bool(p.get("is_paper", 1))
        and (p.get("fill_status") or "") == "filled"
    ]

    db_token_ids: set[str] = set()
    orphan_db = 0
    orphan_db_auto_closed = 0
    share_drift = 0
    share_drift_healed = 0
    share_drift_unhealed = 0

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
            # Orphan_db: DB thinks we own shares, chain says zero.
            #
            # Multi-cycle confirmation gate: increment a counter on the
            # position row.  When the counter reaches the threshold (default
            # 2), auto-close the position at total loss.  Until then, just
            # WARN-log so a single Data API hiccup can't close real
            # positions.
            orphan_db += 1
            from activity import log_activity

            cycles_so_far = (
                _bump_orphan_db_counter(pos["id"])
                if auto_close_orphans else 0
            )

            if (auto_close_orphans
                    and cycles_so_far >= orphan_close_threshold):
                # Lazy-fetch wallet activity once per cycle: the orphan
                # auto-close uses it to compute REAL realized P&L
                # (avoids over-stating losses when an order never
                # actually filled on chain).  See
                # _compute_realized_pnl_from_activity for details.
                if _wallet_activity_cache is None:
                    _wallet_activity_cache = _fetch_activity_history(
                        WALLET_ADDRESS or ""
                    )
                ok = _auto_close_orphan_db(
                    pid        = pos["id"],
                    db_shares  = db_shares,
                    size_usdc  = float(pos.get("size_usdc") or 0),
                    city       = pos.get("city") or "",
                    date_str   = pos.get("date") or "",
                    side       = side,
                    token_id   = token_id,
                    pos        = pos,
                    activity   = _wallet_activity_cache,
                )
                if ok:
                    orphan_db_auto_closed += 1
                    continue
                # Fall through to WARN log if write failed.

            log_activity(
                "DRIFT", level="WARN", position_id=pos["id"],
                message=(
                    f"DRIFT orphan_db: {pos.get('city')} {pos.get('date')} "
                    f"{side} db_shares={db_shares:.4f} but no on-chain "
                    f"balance for token={token_id[:14]}... "
                    f"(cycle {cycles_so_far}/{orphan_close_threshold})"
                    if auto_close_orphans else
                    f"DRIFT orphan_db: {pos.get('city')} {pos.get('date')} "
                    f"{side} db_shares={db_shares:.4f} but no on-chain "
                    f"balance for token={token_id[:14]}..."
                ),
                drift_kind="orphan_db", token_id=token_id,
                db_shares=db_shares, cycles_so_far=cycles_so_far,
            )
            continue

        # Chain shows shares for this token -- reset the orphan counter
        # in case it had been incrementing.
        if auto_close_orphans:
            _reset_orphan_db_counter(pos["id"])

        chain_size = float(on_chain.get("size") or 0)
        delta = abs(db_shares - chain_size)
        if delta <= _RECONCILE_SHARE_TOLERANCE:
            continue

        share_drift += 1
        from activity import log_activity

        if not auto_heal:
            # Legacy behaviour: log only.
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
            continue

        # Auto-heal: sync to chain truth.
        ok, new_avg, new_usdc = _heal_share_drift_auto(
            pos["id"],
            pos.get("contract_id") or "",
            token_id, chain_size, client,
        )
        if ok:
            share_drift_healed += 1
            basis_msg = ""
            if new_avg is not None and new_usdc is not None:
                basis_msg = (f", entry=${new_avg:.4f}, "
                             f"size=${new_usdc:.2f}")
            log_activity(
                "DRIFT", level="INFO", position_id=pos["id"],
                message=(
                    f"DRIFT auto-healed: {pos.get('city')} "
                    f"{pos.get('date')} {side} shares "
                    f"{db_shares:.4f} -> {chain_size:.4f} "
                    f"(delta {delta:.4f}){basis_msg}"
                ),
                drift_kind="share_drift_healed", token_id=token_id,
                db_shares=db_shares, chain_size=chain_size, delta=delta,
                new_entry_price=new_avg, new_size_usdc=new_usdc,
            )
        else:
            share_drift_unhealed += 1
            log_activity(
                "DRIFT", level="WARN", position_id=pos["id"],
                message=(
                    f"DRIFT share_drift (heal failed): {pos.get('city')} "
                    f"{pos.get('date')} {side} db_shares={db_shares:.4f} "
                    f"chain_size={chain_size:.4f} delta={delta:.4f}"
                ),
                drift_kind="share_drift_unhealed", token_id=token_id,
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
        # Build a single-line summary that surfaces auto-action counts when
        # they're nonzero.  Keeps quiet otherwise (no log spam).
        parts = []
        if orphan_db_auto_closed:
            parts.append(
                f"{orphan_db} orphan_db ({orphan_db_auto_closed} auto-closed)"
            )
        else:
            parts.append(f"{orphan_db} orphan_db")
        if share_drift_healed and auto_heal:
            parts.append(
                f"{share_drift} share_drift "
                f"({share_drift_healed} auto-healed, "
                f"{share_drift_unhealed} unhealed)"
            )
        else:
            parts.append(f"{share_drift} share_drift")
        parts.append(f"{orphan_chain} orphan_chain")
        logger.log(SUMMARY_LEVEL,
            f"[MONITOR] Reconciliation: " + " | ".join(parts)
        )
    return {
        "orphan_db":             orphan_db,
        "orphan_db_auto_closed": orphan_db_auto_closed,
        "share_drift":           share_drift,
        "share_drift_healed":    share_drift_healed,
        "share_drift_unhealed":  share_drift_unhealed,
        "orphan_chain":          orphan_chain,
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
