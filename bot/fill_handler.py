"""
fill_handler.py — Single source of truth for applying CLOB order/trade
events to the database.

Used by BOTH:
  * user_ws.py  — real-time event handlers (sub-second latency)
  * monitor.py  — periodic REST reconciliation (safety net)

Both paths flow through the same `apply_trade_event` / `apply_order_event`
functions, so the DB sees identical writes regardless of source.  This lets
the WS be the primary path and the REST poll be a backup, with no
divergence in business logic.

Idempotency model
-----------------
Polymarket does NOT publish a delivery guarantee — events are at-least-once.
The same trade is *intentionally* re-emitted across MATCHED → MINED →
CONFIRMED, and reconnect/duplicate-subscribe quirks can re-deliver true
duplicates.  We dedup via two complementary mechanisms:

  1. `positions.trade_status` is monotonic — `update_position_trade_status`
     refuses to regress (CONFIRMED → MATCHED never wins).  This survives
     restarts without a separate dedup table.

  2. The fill-application step (which writes price/shares/fees) only fires
     on the FIRST transition to CONFIRMED, gated by the prior status read
     before the lifecycle column is advanced.

Trade lifecycle (per Polymarket docs)
-------------------------------------
  matched   → engine matched, tx not on chain — keep position 'pending'
  mined     → tx mined, no finality threshold — still 'pending'
  confirmed → strong probabilistic finality — flip to 'filled' and capture fee
  retrying  → tx reverted/reorged, being resubmitted — keep 'pending'
  failed    → terminal failure — release the position (cancel)
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle parsing
# ---------------------------------------------------------------------------

# Map raw status strings (case-insensitive) to our normalized lifecycle stage.
# Anything not in this map is treated as unknown and skipped — better to
# under-react than misclassify.
_STATUS_NORMALIZE = {
    "matched":   "matched",
    "mined":     "mined",
    "confirmed": "confirmed",
    "retrying":  "retrying",
    "failed":    "failed",
    # REST 'live' / 'delayed' map to matched-or-earlier; treat as 'matched'
    # so we at least record that the order has been acknowledged.
    "live":      "matched",
    "delayed":   "matched",
}


def _normalize_status(raw: str | None) -> str | None:
    if raw is None:
        return None
    return _STATUS_NORMALIZE.get(str(raw).lower())


def _now_iso() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).isoformat()


# ---------------------------------------------------------------------------
# Order-ID extraction (we may be maker or taker)
# ---------------------------------------------------------------------------

def extract_our_order_id(event: dict, my_wallet: str | None) -> str | None:
    """Pull our order's CLOB id out of a `trade` event.

    The trade event contains `taker_order_id` and `maker_orders[]`.  Our
    order is the one whose owner matches our wallet.  When `my_wallet` is
    None we fall back to taker_order_id (most common for the bot since we
    cross the spread on stops).
    """
    if not event:
        return None
    taker_id = event.get("taker_order_id") or event.get("takerOrderID")
    if my_wallet is None:
        return taker_id

    my_wallet_lc = my_wallet.lower()
    # Were we the taker?
    owner = (event.get("trade_owner") or event.get("owner") or "").lower()
    if owner == my_wallet_lc and taker_id:
        return taker_id

    # Were we one of the makers?
    for mo in (event.get("maker_orders") or []):
        if (mo.get("owner") or "").lower() == my_wallet_lc:
            mid = mo.get("order_id") or mo.get("orderID")
            if mid:
                return mid

    # Fall back to taker_id — better to attempt a lookup than ignore.
    return taker_id


# ---------------------------------------------------------------------------
# Fee extraction (mirrors execution.extract_fee_amount but works on WS shape)
# ---------------------------------------------------------------------------

def _extract_fee(event: dict, fill_amount_usdc: float) -> float:
    """Best-effort fee extraction from a trade event.  WS payloads don't
    consistently carry fee data — fall back to feeRateBps × fill amount
    when only the rate is exposed.  Returns 0.0 when nothing usable."""
    for k in ("fee", "feesAccrued", "feesPaid"):
        v = event.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    bps = event.get("fee_rate_bps") or event.get("feeRateBps")
    if bps is not None:
        try:
            rate = float(bps) / 10_000.0
        except (TypeError, ValueError):
            rate = 0.0
        if rate > 0 and fill_amount_usdc > 0:
            return float(fill_amount_usdc) * rate
    return 0.0


# ---------------------------------------------------------------------------
# Trade event — the substantive state change
# ---------------------------------------------------------------------------

def apply_trade_event(event: dict, my_wallet: str | None = None) -> dict:
    """Apply a Polymarket user-channel `trade` event to the DB.

    Returns {action, position_id?, status?, role?, ...} for logging.
    Possible actions:
      * "ignored_unknown_status"    — couldn't parse the status field
      * "ignored_duplicate_event"   — event_id already in processed_trade_events
      * "ignored_no_order_id"       — couldn't determine which order this is for
      * "ignored_no_position"       — order not in our DB (foreign or pre-bot)
      * "lifecycle_regression"      — non-confirmed event arrived after a
                                       later stage (logged, no fill action)
      * "lifecycle_advanced"        — status moved forward but no fill applied
      * "filled"                    — CONFIRMED event applied its fill
                                       (additive — fires per chunk)
      * "failed"                    — terminal failure, position released
    """
    from db import (
        get_position_by_order_id, classify_position_role,
        update_position_trade_status,
        update_position_fill, update_position_exit_filled,
        update_position_topup, cancel_position,
        clear_position_topup_pending,
        add_position_entry_fee, set_position_exit_fee_and_net_pnl,
        get_latest_snapshot_id_for_contract,
        mark_event_processed,
    )

    if not isinstance(event, dict):
        return {"action": "ignored_unknown_status"}

    new_status = _normalize_status(event.get("status"))
    if new_status is None:
        return {"action": "ignored_unknown_status",
                "raw_status": event.get("status")}

    # Per-trade-event dedup.  This catches Polymarket's at-least-once
    # redelivery (matched -> mined -> confirmed redeliveries of the SAME
    # trade carry the same event_id) AND duplicate paths (WS + REST
    # safety net both reporting the same event).  CRITICAL: a single
    # limit order matching against multiple resting asks emits N trade
    # events with N distinct event_ids — those are NOT duplicates and
    # must each apply their own fill (this was the chunked-fill bug).
    event_id = event.get("id")
    if event_id and not mark_event_processed(event_id):
        return {"action": "ignored_duplicate_event", "event_id": str(event_id)}

    order_id = extract_our_order_id(event, my_wallet)
    if not order_id:
        return {"action": "ignored_no_order_id"}

    pos = get_position_by_order_id(order_id)
    if pos is None:
        return {"action": "ignored_no_position", "order_id": order_id}

    role = classify_position_role(pos, order_id)  # 'entry' / 'exit' / 'topup'
    side_param = "exit" if role == "exit" else "entry"

    # Read prior lifecycle stage BEFORE we advance, so we can detect the
    # CONFIRMED transition exactly once.
    prior_status_col = "exit_trade_status" if role == "exit" else "trade_status"
    prior_status = (pos.get(prior_status_col) or "").lower()

    # Try to advance lifecycle.  Returns False on regression
    # (e.g. a 'matched' event arriving after we've already seen the
    # 'confirmed' for chunk 1 of the same order).  Lifecycle stays
    # monotonic — but lifecycle regression NO LONGER blocks fill
    # application.  Fill application is now gated only by:
    #   * per-event-id dedup at the top of this function
    #   * the new_status == 'confirmed' check below
    # That separation is what fixes the chunked-fill bug — multiple
    # CONFIRMED events for the same order_id (different event_ids,
    # different trade matches) now each apply their fill additively.
    advanced = update_position_trade_status(
        position_id    = pos["id"],
        new_status     = new_status,
        side           = side_param,
        last_event_id  = event.get("id"),
    )
    # Mirror the lifecycle stage onto the position_orders ledger row so
    # the dashboard / queries see consistent state across both tables.
    try:
        from db import update_position_order_status
        update_position_order_status(order_id, trade_status=new_status)
    except Exception:
        pass

    # Lifecycle regressions (matched arriving after confirmed for a
    # later-chunk event) don't carry fill information we'd act on
    # anyway — only 'confirmed' events apply fills, and lifecycle
    # tracking is just for observability.  Log and move on.
    if not advanced and new_status != "confirmed":
        return {
            "action":      "lifecycle_regression",
            "position_id": pos["id"],
            "from":        prior_status,
            "to":          new_status,
            "role":        role,
        }

    # Lifecycle advanced (or stayed at 'confirmed' for a new chunk).
    # Two terminal cases warrant action:
    #   confirmed → write the fill (per-event, deduped at the top)
    #   failed    → release the position

    if new_status == "confirmed":
        return _apply_confirmed_fill(
            event       = event,
            position    = pos,
            role        = role,
            order_id    = order_id,
            update_position_fill         = update_position_fill,
            update_position_exit_filled  = update_position_exit_filled,
            update_position_topup        = update_position_topup,
            add_position_entry_fee       = add_position_entry_fee,
            set_position_exit_fee_and_net_pnl = set_position_exit_fee_and_net_pnl,
            get_latest_snapshot_id_for_contract = get_latest_snapshot_id_for_contract,
        )

    if new_status == "failed":
        return _apply_failed(
            event       = event,
            position    = pos,
            role        = role,
            order_id    = order_id,
            cancel_position             = cancel_position,
            clear_position_topup_pending = clear_position_topup_pending,
        )

    return {
        "action":      "lifecycle_advanced",
        "position_id": pos["id"],
        "from":        prior_status,
        "to":          new_status,
        "role":        role,
    }


def _apply_confirmed_fill(
    *,
    event: dict,
    position: dict,
    role: str,
    order_id: str,
    update_position_fill,
    update_position_exit_filled,
    update_position_topup,
    add_position_entry_fee,
    set_position_exit_fee_and_net_pnl,
    get_latest_snapshot_id_for_contract,
) -> dict:
    """Write the actual fill to the DB.  Routes by role."""
    try:
        trade_size  = float(event.get("size") or 0)
        trade_price = float(event.get("price") or 0)
    except (TypeError, ValueError):
        return {"action": "filled_failed_parse", "position_id": position["id"]}
    if trade_size <= 0 or trade_price <= 0:
        return {"action": "filled_failed_zero", "position_id": position["id"]}

    fill_usdc = trade_size * trade_price
    fee_usdc  = _extract_fee(event, fill_usdc)
    pid       = position["id"]
    contract  = position.get("contract_id", "")

    # Update the ledger row for this CLOB order.  A trade event might be a
    # PARTIAL fill (the order matched some shares; rest still on book) or
    # the LAST chunk of a multi-match fill.  We compare cumulative filled
    # vs intended:
    #   filled >= intended × 0.99 (within rounding) → status='filled', closed
    #   else                                          → status='partial'
    #
    # `order_is_complete` is read by the topup branch below so it knows
    # whether this chunk was the last one (clear pending_topup_*) or
    # one of many (leave them set so subsequent chunks route correctly).
    order_is_complete = False
    try:
        from db import (
            get_position_order_by_id, update_position_order_status,
        )
        ledger_row = get_position_order_by_id(order_id)
        if ledger_row is not None:
            # Cumulative across multiple trade events for the same order
            prev_filled_shares = float(ledger_row.get("filled_shares") or 0)
            prev_filled_usdc   = float(ledger_row.get("filled_usdc")   or 0)
            prev_fee           = float(ledger_row.get("fee_usdc")      or 0)
            new_filled_shares  = prev_filled_shares + trade_size
            new_filled_usdc    = prev_filled_usdc   + fill_usdc
            new_fee            = prev_fee           + fee_usdc
            new_avg_price      = (
                new_filled_usdc / new_filled_shares
                if new_filled_shares > 0 else trade_price
            )
            intended = float(ledger_row.get("intended_shares") or 0)
            order_is_complete = (intended > 0
                                 and new_filled_shares >= intended * 0.99)
            new_status = "filled" if order_is_complete else "partial"
            update_position_order_status(
                order_id      = order_id,
                status        = new_status,
                filled_shares = new_filled_shares,
                filled_usdc   = new_filled_usdc,
                fill_price    = new_avg_price,
                fee_usdc      = new_fee,
                closed        = order_is_complete,
            )
    except Exception as _e:
        logger.debug(f"position_orders update on fill failed (non-fatal): {_e}")

    if role == "entry":
        # Additive: each chunk of a multi-match entry order adds its
        # shares to the running total.  First chunk also flips
        # fill_status pending -> filled.  See db.add_position_entry_fill
        # for the cost-basis weighted-average recompute.
        from db import add_position_entry_fill
        add_position_entry_fill(
            position_id  = pid,
            added_shares = trade_size,
            fill_price   = trade_price,
        )
        if fee_usdc > 0:
            add_position_entry_fee(pid, fee_usdc)
        from activity import log_activity
        log_activity(
            "FILL", position_id=pid,
            message=(
                f"entry CONFIRMED on chain: {position.get('city')} "
                f"{position.get('date')} {trade_size:.4f} shares @ "
                f"{trade_price:.4f} fee=${fee_usdc:.4f}"
            ),
            role="entry", shares=trade_size, price=trade_price, fee=fee_usdc,
        )
        return {
            "action": "filled", "position_id": pid, "role": "entry",
            "shares": trade_size, "price": trade_price, "fee": fee_usdc,
        }

    if role == "exit":
        # Per-chunk additive accounting (Layer 3 fix, 2026-04-30).
        # Each CONFIRMED trade event for an exit order:
        #   * decrements positions.shares by trade_size
        #   * accumulates exit_proceeds_usdc and exit_fees
        #   * only flips status='closed' when shares hit ~0 (last chunk)
        # The legacy code (single update_position_exit_filled call) treated
        # every chunk as the FULL position close — wrong on multi-chunk
        # exits, where it (a) overwrote actual_exit_price each chunk
        # (last-chunk-wins instead of weighted avg), (b) computed pnl
        # against the full position size at every chunk's price, and
        # (c) never decremented shares — leaving the bot trying to sell
        # tokens it no longer held (the user-reported pos=222 KL bleed).
        from db import add_position_exit_fill
        result = add_position_exit_fill(
            position_id  = pid,
            sold_shares  = trade_size,
            fill_price   = trade_price,
            fee_usdc     = fee_usdc,
        )
        from activity import log_activity
        if result["is_complete"]:
            # Last chunk landed — record the final, weighted-average exit.
            # Stamp exit_snapshot_id separately since add_position_exit_fill
            # is schema-agnostic (doesn't know about snapshots).
            exit_snap = get_latest_snapshot_id_for_contract(contract)
            if exit_snap is not None:
                try:
                    from db import _get_conn
                    with _get_conn() as _c:
                        _c.execute(
                            "UPDATE positions SET exit_snapshot_id = ? "
                            "WHERE id = ? AND exit_snapshot_id IS NULL",
                            (exit_snap, pid),
                        )
                except Exception:
                    pass
            # Backfill real fees from Polymarket trades API.  WS trade
            # events don't carry fee_rate_bps, so the per-chunk fee
            # accumulator can only sum to 0; this fetches the actual
            # fees post-fact and updates entry_fees / exit_fees / pnl_net.
            # Best-effort: failures here don't block the close.
            try:
                from execution import backfill_position_fees, get_clob_client
                _bf = backfill_position_fees(pid, get_clob_client())
                if _bf and _bf.get("n_trades_matched", 0) > 0:
                    logger.info(
                        f"[FEES] pid={pid} backfilled: "
                        f"entry=${_bf['entry_fees']:.4f}, "
                        f"exit=${_bf['exit_fees']:.4f}, "
                        f"pnl_net=${_bf['pnl_net']:+.4f} "
                        f"({_bf['n_trades_matched']} trades matched)"
                    )
            except Exception as _e:
                logger.debug(
                    f"[FEES] pid={pid} fee backfill failed (non-fatal): {_e}"
                )
            gross_pnl = result["gross_pnl"]
            net_pnl   = result["net_pnl"]
            avg_exit  = result["actual_exit_price"]
            entry_fees = float(position.get("entry_fees") or 0)
            log_activity(
                "FILL", position_id=pid,
                level="INFO" if (net_pnl or 0) >= 0 else "WARN",
                message=(
                    f"exit COMPLETE on chain: {position.get('city')} "
                    f"{position.get('date')} avg_exit=${avg_exit:.4f} "
                    f"gross=${gross_pnl:+.4f} net=${(net_pnl or 0):+.4f} "
                    f"(last_chunk @{trade_price:.4f})"
                ),
                role="exit", chunk_price=trade_price,
                avg_exit_price=avg_exit,
                gross_pnl=gross_pnl, net_pnl=net_pnl,
                entry_fees=entry_fees, exit_fee=fee_usdc,
            )
            return {
                "action": "filled", "position_id": pid, "role": "exit",
                "price": trade_price, "avg_exit_price": avg_exit,
                "gross_pnl": gross_pnl, "net_pnl": net_pnl,
                "fee": fee_usdc, "exit_complete": True,
            }
        else:
            # Partial — log per-chunk progress for traceability without
            # claiming the position is closed.
            log_activity(
                "FILL", position_id=pid, level="INFO",
                message=(
                    f"exit PARTIAL on chain: {position.get('city')} "
                    f"{position.get('date')} chunk={trade_size:.4f} "
                    f"@{trade_price:.4f} shares_left={result['shares_after']:.4f}"
                ),
                role="exit", chunk_size=trade_size, chunk_price=trade_price,
                shares_remaining=result["shares_after"], fee=fee_usdc,
            )
            return {
                "action": "filled", "position_id": pid, "role": "exit",
                "price": trade_price, "fee": fee_usdc,
                "shares_remaining": result["shares_after"],
                "exit_complete": False,
            }

    # role == "topup"
    # Additive: each chunk grows the position's shares + size_usdc and
    # recomputes weighted-average entry_price.  Pending_topup_* fields
    # stay set until the LAST chunk lands (signalled by order_is_complete
    # from the ledger above) so chunks 2..N can still be routed back to
    # this position via get_position_by_order_id matching on
    # pending_topup_order_id.
    from db import add_position_topup_fill, clear_position_topup_pending
    add_position_topup_fill(
        position_id  = pid,
        added_usdc   = round(fill_usdc, 4),
        added_shares = trade_size,
    )
    if order_is_complete:
        clear_position_topup_pending(pid)
    if fee_usdc > 0:
        add_position_entry_fee(pid, fee_usdc)
    # Compute the post-fill weighted average for the log line — this is
    # the running cost basis after THIS chunk landed (existing + chunk).
    new_total_shares = float(position.get("shares") or 0) + trade_size
    new_total_usdc   = float(position.get("size_usdc") or 0) + fill_usdc
    new_avg = (new_total_usdc / new_total_shares) if new_total_shares > 0 else trade_price
    from activity import log_activity
    log_activity(
        "FILL", position_id=pid,
        message=(
            f"topup CONFIRMED on chain: {position.get('city')} "
            f"{position.get('date')} +${fill_usdc:.2f} @{trade_price:.4f} "
            f"new_avg={new_avg:.4f} fee=${fee_usdc:.4f}"
        ),
        role="topup", added_usdc=fill_usdc, price=trade_price,
        new_avg=new_avg, fee=fee_usdc,
    )
    return {
        "action": "filled", "position_id": pid, "role": "topup",
        "added_usdc": fill_usdc, "price": trade_price,
        "new_avg": new_avg, "fee": fee_usdc,
    }


def _apply_failed(
    *,
    event: dict,
    position: dict,
    role: str,
    order_id: str,
    cancel_position,
    clear_position_topup_pending,
) -> dict:
    """Terminal trade failure — release whatever resource the position was
    holding so the bot doesn't think it's still in flight."""
    pid = position["id"]
    # Mark the ledger row as failed (terminal — won't count in committed_usdc).
    try:
        from db import update_position_order_status
        update_position_order_status(
            order_id        = order_id,
            status          = "failed",
            cancelled_reason = "trade_failed_onchain",
            closed          = True,
        )
    except Exception:
        pass
    from activity import log_activity
    if role == "entry":
        cancel_position(
            position_id      = pid,
            cancelled_reason = "trade_failed_onchain",
            exit_time        = _now_iso(),
        )
        log_activity(
            "FAIL", level="ERROR", position_id=pid,
            message=(
                f"entry trade FAILED on chain: {position.get('city')} "
                f"{position.get('date')} — position cancelled, capital released"
            ),
            role="entry",
        )
        return {"action": "failed", "position_id": pid, "role": "entry"}

    if role == "topup":
        clear_position_topup_pending(pid)
        log_activity(
            "FAIL", level="ERROR", position_id=pid,
            message=(
                f"topup trade FAILED on chain: {position.get('city')} "
                f"{position.get('date')} — pending fields cleared, "
                f"parent position untouched"
            ),
            role="topup",
        )
        return {"action": "failed", "position_id": pid, "role": "topup"}

    # role == "exit": leave the position in 'exiting' so the ladder advancer
    # picks it up next cycle.  Don't auto-revert to 'open' — that would
    # require recomputing peak_price/stop_loss which we'd rather do via the
    # explicit ladder flow.
    log_activity(
        "FAIL", level="ERROR", position_id=pid,
        message=(
            f"exit trade FAILED on chain: {position.get('city')} "
            f"{position.get('date')} — leaving in 'exiting' state for ladder advancer"
        ),
        role="exit",
    )
    return {"action": "failed", "position_id": pid, "role": "exit"}


# ---------------------------------------------------------------------------
# Order event — lifecycle metadata + cancellation handling
# ---------------------------------------------------------------------------

def apply_order_event(event: dict) -> dict:
    """Process a Polymarket user-channel `order` event.

    Order events carry the order's resting state (size_matched, etc.) but
    the substantive fill is applied via trade events above.  This handler:
      * Logs PLACEMENT for traceability
      * Clears pending state on CANCELLATION (so monitor.py doesn't think
        we still have an in-flight order)
      * Logs UPDATE (partial-fill notification — we ignore for now since
        trade events carry the same info with price)
    """
    from db import (
        get_position_by_order_id, classify_position_role,
        clear_position_topup_pending, cancel_position,
    )

    if not isinstance(event, dict):
        return {"action": "ignored"}

    order_id = event.get("id") or event.get("order_id")
    if not order_id:
        return {"action": "ignored_no_order_id"}

    pos = get_position_by_order_id(order_id)
    if pos is None:
        return {"action": "ignored_no_position", "order_id": order_id}

    op = (event.get("type") or "").upper()
    role = classify_position_role(pos, order_id)
    pid = pos["id"]

    if op == "PLACEMENT":
        logger.debug(f"[FILL] order PLACEMENT pos={pid} role={role} ack")
        return {"action": "placement_ack", "position_id": pid, "role": role}

    if op == "UPDATE":
        try:
            sm = float(event.get("size_matched") or 0)
            os_ = float(event.get("original_size") or 0)
        except (TypeError, ValueError):
            sm, os_ = 0.0, 0.0
        logger.debug(
            f"[FILL] order UPDATE pos={pid} role={role} "
            f"size_matched={sm}/{os_}"
        )
        return {"action": "update_ack", "position_id": pid, "role": role,
                "size_matched": sm, "original_size": os_}

    if op == "CANCELLATION":
        # If this is a pending top-up, clear the parent's pending fields.
        # If it's an entry that never filled, mark the position cancelled.
        # If it's an exit, leave the row in 'exiting' for the ladder advancer
        # to handle (mirrors the FAILED behavior above).
        # In all cases, mark the ledger row 'cancelled' (terminal status,
        # won't count toward committed_usdc).
        try:
            from db import update_position_order_status
            update_position_order_status(
                order_id        = order_id,
                status          = "cancelled",
                cancelled_reason = "cancelled_via_ws",
                closed          = True,
            )
        except Exception:
            pass
        from activity import log_activity
        if role == "topup":
            clear_position_topup_pending(pid)
            log_activity(
                "CANCEL", position_id=pid,
                message=(
                    f"topup CANCELLED via WS: {pos.get('city')} "
                    f"{pos.get('date')} — pending fields cleared"
                ),
                role="topup", source="ws",
            )
            return {"action": "cancelled", "position_id": pid, "role": "topup"}
        if role == "entry":
            # Only mark the position cancelled if it's still pending — a
            # filled position got cancelled would mean the cancel raced the
            # fill, which we shouldn't treat as cancellation.
            if (pos.get("fill_status") or "") == "pending":
                cancel_position(
                    position_id      = pid,
                    cancelled_reason = "cancelled_via_ws",
                    exit_time        = _now_iso(),
                )
                log_activity(
                    "CANCEL", position_id=pid,
                    message=(
                        f"entry CANCELLED via WS: {pos.get('city')} "
                        f"{pos.get('date')} {pos.get('side')} — never filled"
                    ),
                    role="entry", source="ws",
                )
                return {"action": "cancelled", "position_id": pid, "role": "entry"}
        logger.debug(
            f"[FILL] order CANCELLATION pos={pid} role={role} — no-op "
            f"(non-actionable for this role/status)"
        )
        return {"action": "cancelled_noop", "position_id": pid, "role": role}

    return {"action": "ignored_unknown_op", "op": op, "position_id": pid}
