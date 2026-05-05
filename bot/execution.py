"""
execution.py — Trade execution and position logging.

Handles both live (CLOB) and paper trading paths.  Every attempted trade —
whether placed on-chain or simulated — is written to the positions table so
that risk checks, P&L tracking, and the monitor loop all have a consistent
source of truth.

Live order status values returned by Polymarket CLOB:
    "matched"   — fully or partially filled immediately → log as fill_status='filled'
    "live"      — resting on the book (GTC) → log as fill_status='pending'
    "unmatched" — marketable order, no fill after delay → do not log, treat as failed
    "delayed"   — sports-market delay; treated same as 'live' here
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# py_clob_client_v2 — required for the new Polymarket order schema.
# The v1 package (py_clob_client) signs orders against deprecated exchange
# contract addresses, causing every live order to fail with
# `order_version_mismatch`.  v2 has the current contracts + new schema.
# Migration completed 2026-04-29.
from py_clob_client_v2 import ClobClient, OrderArgs, OrderType, Side
from py_clob_client_v2.constants import POLYGON

from config import PAPER_TRADE, EXIT_HARD_STOP_PCT, ACTIVE_STRATEGY
from db import (
    insert_position, mark_outcome_executed,
    get_latest_snapshot_id_for_contract, get_recent_cancelled_count,
)


# Maximum BUY retries per contract within a 6-hour window.  Hardcoded
# because the right cap is more about "preventing thrash" than something
# users tune frequently.  At 3 retries × 30-min scan = 1.5h of attempts
# before we give up, plenty for normal market liquidity to recover.
_MAX_BUY_RETRIES_PER_6H = 3
_BUY_RETRY_WINDOW_HOURS = 6


# ---------------------------------------------------------------------------
# Geoblock circuit breaker
# ---------------------------------------------------------------------------
# Polymarket geoblocks US (and some other) IPs from order placement.  When
# detected, every subsequent order placement returns 403 with the same
# "Trading restricted in your region" message.  Hammering the API with
# more orders just spams the log — there is nothing we can do until the
# operator switches networks (VPN, droplet in a permitted region, etc.).
#
# Once the breaker trips, subsequent placement attempts short-circuit
# without hitting the API.  The breaker resets only on process restart —
# operator must change network to clear it.

_geoblock_tripped: bool = False


def _reset_geoblock_circuit() -> None:
    """Test hook; not called from production code."""
    global _geoblock_tripped
    _geoblock_tripped = False


def _is_geoblock_error(exc: Exception) -> bool:
    """True iff the exception text indicates Polymarket's geoblock."""
    msg = str(exc).lower()
    return ("geoblock" in msg
            or "trading restricted in your region" in msg
            or ("403" in msg and "available regions" in msg))


def _trip_geoblock_circuit(exc: Exception) -> None:
    """Mark the breaker tripped + log a clear, actionable message ONCE."""
    global _geoblock_tripped
    if _geoblock_tripped:
        return
    _geoblock_tripped = True
    logger.error(
        "════════════════════════════════════════════════════════════════\n"
        "  POLYMARKET GEOBLOCK DETECTED — your IP cannot place orders\n"
        "  (HTTP 403: Trading restricted in your region)\n"
        "\n"
        "  Subsequent order placement attempts in this process will be\n"
        "  short-circuited to avoid log spam.  No code change can bypass\n"
        "  this — Polymarket enforces it server-side per IP.\n"
        "\n"
        "  To resume trading:\n"
        "    1. Run the bot from a non-geoblocked region (e.g., UK/EU/JP\n"
        "       cloud droplet), OR\n"
        "    2. Use a VPN to route requests through a permitted region.\n"
        "    3. Restart the bot to clear the breaker.\n"
        "\n"
        "  See: https://docs.polymarket.com/developers/CLOB/geoblock\n"
        "════════════════════════════════════════════════════════════════"
    )
    try:
        from activity import log_activity
        log_activity(
            "RISK", level="ERROR",
            message=(
                "Polymarket geoblock detected — order placement halted "
                "until process restart from a permitted region"
            ),
            error=str(exc),
        )
    except Exception:
        pass


def _short_circuited_response() -> dict:
    """Return value when the geoblock breaker is tripped — looks like a
    soft failure so the trading loop moves on without crashing."""
    return {
        "status": "skip",
        "reason": "geoblock_circuit_tripped",
    }

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = POLYGON  # 137


def _compute_pre_entry_metrics(contract_id: str) -> tuple[float | None, float | None, float | None]:
    """Compute volatility, trend, and momentum from the last 30 min of price snapshots.

    Returns (volatility, trend, momentum) or (None, None, None) if insufficient data.
    - volatility: std of consecutive price changes
    - trend: last price minus first price in window
    - momentum: fraction of price changes that were positive (0.0-1.0)
    """
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT yes_price FROM bin_price_history
            WHERE contract_id = ? AND yes_price IS NOT NULL
            ORDER BY recorded_at DESC
            LIMIT 15
        """, (contract_id,)).fetchall()
        conn.close()

        if not rows or len(rows) < 5:
            return None, None, None

        prices = [float(r[0]) for r in reversed(rows)]
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        import statistics
        volatility = round(statistics.stdev(changes), 6) if len(changes) >= 2 else None
        trend = round(prices[-1] - prices[0], 4)
        momentum = round(sum(1 for c in changes if c > 0) / len(changes), 4)

        return volatility, trend, momentum
    except Exception:
        return None, None, None


def get_clob_client() -> ClobClient | None:
    """
    Initialize and return an authenticated CLOB client.
    Returns None in paper trading mode.

    Critical: passes funder=WALLET_ADDRESS and signature_type so the
    client targets the user's Polymarket proxy/safe (where USDC actually
    sits), not the signer EOA derived from the private key.  Without
    these, balance queries return $0 and order signing uses the wrong
    owner — every live order would fail.

    Auth strategy:
      1. If CLOB_API_KEY/SECRET/PASSPHRASE are present in env → use them
         directly (no network call).  THIS IS THE PREFERRED PATH for
         production — Polymarket's /auth/api-key endpoint sits behind
         Cloudflare and rate-limits aggressively from datacenter IPs.
      2. Otherwise → fall back to client.create_or_derive_api_key() and
         log instructions for caching the result.  Use the helper script
         at scripts/derive_api_creds.py to bootstrap once when not
         Cloudflare-blocked.
    """
    if PAPER_TRADE:
        logger.info("PAPER_TRADE=True — CLOB client not initialized")
        return None

    import os
    from config import WALLET_ADDRESS, WALLET_SIGNATURE_TYPE

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        raise ValueError("POLYMARKET_PRIVATE_KEY not set in environment")

    if not WALLET_ADDRESS:
        raise ValueError(
            "WALLET_ADDRESS not set in environment.  Required for live "
            "trading — this is the Polymarket proxy/safe address that "
            "holds your USDC (visible in the Polymarket UI under your "
            "profile).  It is NOT the EOA derived from your private key."
        )

    # Cached creds path — preferred.  Avoids hitting /auth/api-key on
    # every startup, which is necessary because Cloudflare rate-limits
    # that endpoint and a blocked startup leaves the bot unable to trade.
    cached_key        = os.getenv("CLOB_API_KEY", "").strip()
    cached_secret     = os.getenv("CLOB_API_SECRET", "").strip()
    cached_passphrase = os.getenv("CLOB_API_PASSPHRASE", "").strip()

    if cached_key and cached_secret and cached_passphrase:
        from py_clob_client_v2 import ApiCreds
        creds = ApiCreds(
            api_key        = cached_key,
            api_secret     = cached_secret,
            api_passphrase = cached_passphrase,
        )
        client = ClobClient(
            host           = CLOB_HOST,
            key            = private_key,
            chain_id       = CHAIN_ID,
            funder         = WALLET_ADDRESS,
            signature_type = WALLET_SIGNATURE_TYPE,
            creds          = creds,
        )
        logger.info("CLOB client initialized with cached API creds from .env")
        return client

    # Fallback: derive on the fly.  Will hit /auth/api-key — fragile due
    # to Cloudflare.  We log a clear instruction so the operator knows
    # how to migrate to the cached path.
    logger.warning(
        "CLOB_API_KEY/SECRET/PASSPHRASE not set in .env — falling back to "
        "/auth/api-key derivation.  This may fail with HTTP 403 (Cloudflare). "
        "Run scripts/derive_api_creds.py once from an unblocked network and "
        "copy the output into your .env to make startup deterministic."
    )
    client = ClobClient(
        host           = CLOB_HOST,
        key            = private_key,
        chain_id       = CHAIN_ID,
        funder         = WALLET_ADDRESS,
        signature_type = WALLET_SIGNATURE_TYPE,
    )
    try:
        client.set_api_creds(client.create_or_derive_api_key())
    except Exception as e:
        raise RuntimeError(
            "Failed to derive CLOB API creds (likely Cloudflare 403).  "
            "Run scripts/derive_api_creds.py to bootstrap, then add the "
            "three CLOB_API_* values to your .env."
        ) from e
    return client


def execute_signal(signal: dict, client: ClobClient | None = None) -> dict:
    """
    Execute a trade signal on Polymarket, or simulate it in paper trading mode.

    For paper trades  → position is logged immediately with fill_status='filled'
                        using the current market price as entry price.
    For live trades   → order is placed via CLOB; position is logged with
                        fill_status='filled' (if matched immediately) or
                        fill_status='pending' (if resting on book).
                        The monitor loop later confirms pending fills or cancels
                        orders that remain unfilled.

    Returns a result dict with: status, order_id, position_id, and details.
    """
    contract_id = signal["contract_id"]
    side        = signal["recommended_side"]
    size_usdc   = signal["kelly_size"]

    # Buy-retry visibility + soft cap.  Counts cancelled BUYS for this
    # contract in the last 6h; if we've already burned 3 retries and the
    # current scan still wants in, the book likely won't fill us at our
    # price — give up rather than thrash forever.
    # Skipped in paper mode (paper buys always fill, never cancel).
    cancelled_count = (
        get_recent_cancelled_count(contract_id, within_hours=_BUY_RETRY_WINDOW_HOURS)
        if not PAPER_TRADE else 0
    )
    if cancelled_count >= _MAX_BUY_RETRIES_PER_6H:
        from activity import log_activity
        log_activity(
            "RISK", level="WARN",
            message=(
                f"BUY skipped (retry cap): {side} ${size_usdc:.2f} "
                f"{signal.get('city')} {signal.get('date')} {contract_id[:12]} — "
                f"{cancelled_count} cancellations in last {_BUY_RETRY_WINDOW_HOURS}h"
            ),
            contract_id=contract_id, side=side, size_usdc=size_usdc,
            cancelled_count=cancelled_count,
        )
        return {"status": "skip", "reason": "retry_exhausted",
                "cancelled_count": cancelled_count}
    if cancelled_count > 0:
        logger.info(
            f"[BUY RETRY {cancelled_count}/{_MAX_BUY_RETRIES_PER_6H}] "
            f"{side} ${size_usdc:.2f} | {contract_id[:12]} "
            f"{signal.get('city')} {signal.get('date')} | "
            f"prior cancellations in last {_BUY_RETRY_WINDOW_HOURS}h"
        )
    # Store entry_time in local US Central (CST/CDT auto-handled by ZoneInfo).
    # The ISO string carries an explicit offset (e.g. -05:00) so the instant
    # remains unambiguous.
    entry_time  = datetime.now(ZoneInfo("America/Chicago")).isoformat()

    # Determine token ID and price based on side
    if side == "YES":
        token_id    = signal.get("yes_token_id")
        entry_price = float(signal.get("market_p") or signal.get("yes_price") or signal.get("market_price", 0))
    else:
        token_id    = signal.get("no_token_id")
        entry_price = float(1.0 - (signal.get("market_p") or signal.get("yes_price") or signal.get("market_price", 0)))

    if not token_id:
        logger.error(f"No token_id for {contract_id} side={side}")
        return {"status": "error", "reason": "missing_token_id"}

    # Shares = dollars risked / price per share
    shares = round(size_usdc / entry_price, 4) if entry_price > 0 else 0

    # Look up the decision snapshot that produced this signal so we can
    # link the position to its exact entry-time context for later exit
    # comparison (thesis-shift detection).
    entry_snapshot_id = get_latest_snapshot_id_for_contract(contract_id)

    # Compute pre-entry price metrics from recent snapshots
    _pre_vol, _pre_trend, _pre_momentum = _compute_pre_entry_metrics(contract_id)

    # Common position metadata drawn from the signal
    position_kwargs = dict(
        contract_id     = contract_id,
        side            = side,
        size_usdc       = size_usdc,
        entry_price     = entry_price,
        entry_time      = entry_time,
        question        = signal.get("question"),
        city            = signal.get("city"),
        date            = signal.get("date"),
        event_id        = signal.get("event_id"),
        model_prob      = signal.get("model_p") or signal.get("model_prob"),
        market_prob     = signal.get("market_p") or signal.get("market_price"),
        ev              = signal.get("ev"),
        edge            = signal.get("edge"),
        shares          = shares,
        scan_timestamp  = signal.get("scan_timestamp"),
        gamma_market_id = signal.get("gamma_market_id"),
        range_low       = signal.get("range_low"),
        range_high      = signal.get("range_high"),
        unit            = signal.get("unit"),
        yes_token_id    = signal.get("yes_token_id"),
        no_token_id     = signal.get("no_token_id"),
        lat              = signal.get("lat"),
        lon              = signal.get("lon"),
        forecast_sigma_c = signal.get("forecast_sigma_c"),
        entry_snapshot_id = entry_snapshot_id,
        target_size_usdc = signal.get("target_size_usdc"),
        stop_loss_price = round(entry_price * (1.0 + EXIT_HARD_STOP_PCT), 4) if entry_price > 0 else None,
        strategy        = ACTIVE_STRATEGY,
        pre_entry_volatility = _pre_vol,
        pre_entry_trend      = _pre_trend,
        pre_entry_momentum   = _pre_momentum,
    )

    # -------------------------------------------------------------------------
    # Paper trading path
    # -------------------------------------------------------------------------
    if PAPER_TRADE or client is None:
        position_id = insert_position(
            **position_kwargs,
            order_id    = None,
            is_paper    = 1,
            fill_status = "filled",
        )
        mark_outcome_executed(contract_id, signal.get("scan_timestamp", ""))
        # Subscribe to real-time price updates for this token
        try:
            from price_ws import add_tokens
            add_tokens([token_id])
        except Exception:
            pass
        from activity import log_activity
        log_activity(
            "BUY", position_id=position_id,
            message=(
                f"paper {side} ${size_usdc:.2f} @ {entry_price:.4f} "
                f"{signal.get('city')} {signal.get('date')} {contract_id[:12]}"
            ),
            mode="paper", side=side, size_usdc=size_usdc,
            entry_price=entry_price, shares=shares,
        )
        return {
            "status":      "paper",
            "order_id":    None,
            "position_id": position_id,
            "entry_price": entry_price,
            "shares":      shares,
        }

    # -------------------------------------------------------------------------
    # Live trading path — orderbook-aware sweep limit
    # -------------------------------------------------------------------------
    # Geoblock short-circuit: if Polymarket has rejected this IP for
    # regional restrictions, skip the API call entirely.  See _trip_geoblock_circuit
    # for the recovery instructions printed once on first detection.
    if _geoblock_tripped:
        return _short_circuited_response()

    # Replaces the old `entry_price * 1.005` heuristic.  Walks the asks
    # up to (best_ask + ORDERBOOK_WALK_CENTS) capped at MPV_MAX_PRICE.
    # Polymarket matches against asks ≤ limit cheapest-first and rests
    # the unfilled portion at our limit on the book.
    from config import ORDERBOOK_WALK_CENTS
    try:
        from strategies.market_price_value import MPV_MAX_PRICE as _MAX_BUY_PRICE
    except Exception:
        _MAX_BUY_PRICE = 0.99   # fallback when MPV strategy isn't loaded
    limit_price, _sweep_diag = compute_sweep_limit(
        client       = client,
        token_id     = token_id,
        intended_price = entry_price,
        max_cap      = _MAX_BUY_PRICE,
        walk_cents   = ORDERBOOK_WALK_CENTS,
    )
    # Cap by ask-side depth at acceptable prices (replaces the old
    # "% of Gamma's total liquidity" rule which used a wrong basis).
    # Uses the SAME orderbook snapshot we just fetched for the sweep
    # limit — no extra API call.
    from config import (
        LIQUIDITY_AWARE_SIZING,
        MAX_TAKE_PCT_OF_ASK_DEPTH,
        MIN_FILLABLE_USDC,
        ENSURE_FILL_MIN_FILLABLE_USDC,
        ENSURE_FILL_STRATEGIES,
    )
    sweepable_usdc = float(_sweep_diag.get("sweepable_usdc", 0) or 0)
    final_size_usdc = size_usdc
    cap_diag = ""

    # Strategy-aware min-fillable floor.  Ensure-fill strategies (TKH)
    # MUST own every bin in the basket; the per-event dedup blocks any
    # retry once an event has been touched.  So a thin-book skip is
    # PERMANENT for those bins -- they break the hedge thesis.  We
    # therefore lower the min-fillable floor to Polymarket's $1 minimum
    # (+ buffer) for ensure-fill strategies and place WHATEVER the book
    # allows.  The repricer + topup loop will chase any remaining gap.
    sig_strategy = (signal.get("strategy") or ACTIVE_STRATEGY or "").strip()
    is_ensure_fill = sig_strategy in ENSURE_FILL_STRATEGIES
    effective_min_fillable = (
        ENSURE_FILL_MIN_FILLABLE_USDC if is_ensure_fill else MIN_FILLABLE_USDC
    )

    if LIQUIDITY_AWARE_SIZING and sweepable_usdc > 0:
        max_take = sweepable_usdc * MAX_TAKE_PCT_OF_ASK_DEPTH
        if max_take < effective_min_fillable:
            from activity import log_activity
            log_activity(
                "RISK", level="WARN",
                message=(
                    f"BUY skipped (book too thin): {side} ${size_usdc:.2f} "
                    f"{signal.get('city')} {signal.get('date')} {contract_id[:12]} "
                    f"— acceptable ask depth ${sweepable_usdc:.2f}, "
                    f"max take ${max_take:.2f} < floor ${effective_min_fillable:.2f}"
                    + (f" (ensure-fill {sig_strategy})" if is_ensure_fill else "")
                ),
                contract_id=contract_id, sweepable_usdc=sweepable_usdc,
                max_take=max_take, min_floor=effective_min_fillable,
                ensure_fill=is_ensure_fill, strategy=sig_strategy,
            )
            return {"status": "skip", "reason": "book_too_thin",
                    "sweepable_usdc": sweepable_usdc, "max_take": max_take}
        if size_usdc > max_take:
            final_size_usdc = round(max_take, 2)
            cap_diag = (
                f" CAPPED: intended=${size_usdc:.2f} → ${final_size_usdc:.2f} "
                f"(ask_depth=${sweepable_usdc:.2f} × "
                f"{MAX_TAKE_PCT_OF_ASK_DEPTH*100:.0f}%)"
                + (" [ensure-fill: repricer/topup will chase remainder]"
                   if is_ensure_fill else "")
            )
    logger.info(
        f"[SWEEP] {contract_id[:12]} side={side} "
        f"intended={entry_price:.4f} limit={limit_price:.4f} "
        f"source={_sweep_diag.get('source')} "
        f"best_ask={_sweep_diag.get('best_ask')} "
        f"sweepable=${sweepable_usdc:.2f}{cap_diag}"
    )

    try:
        # IMPORTANT: OrderArgs.side is the ORDER side (Side.BUY / Side.SELL),
        # not the POSITION side (YES/NO).  The position side is encoded by
        # which token_id we send (yes_token_id vs no_token_id) — see the
        # branch above where token_id was selected.  All entries are BUYs.
        # v2 uses the Side enum, not strings.
        order_args = OrderArgs(
            price    = limit_price,
            size     = final_size_usdc / limit_price,
            side     = Side.BUY,
            token_id = token_id,
        )
        response = client.create_and_post_order(order_args, order_type=OrderType.GTC)

        if not response or not response.get("success"):
            from activity import log_activity
            log_activity(
                "BUY", level="ERROR",
                message=(
                    f"live BUY placement FAILED: {side} ${size_usdc:.2f} "
                    f"{signal.get('city')} {signal.get('date')} {contract_id[:12]} "
                    f"— {response}"
                ),
                contract_id=contract_id, side=side, size_usdc=size_usdc,
            )
            return {"status": "failed", "response": str(response)}

        order_id      = response.get("orderID", "")
        order_status  = response.get("status", "")

        # Derive ACTUAL fill price + shares from the CLOB response amounts.
        # py_clob_client returns the order's matched portion (which may be a
        # PARTIAL fill — the rest stays resting on the book).
        #
        # For a BUY order:
        #   makerAmount = USDC paid (we "make" USDC available)
        #   takerAmount = shares received (we "take" shares)
        #   price ($/share) = makerAmount / takerAmount
        #   shares received = takerAmount
        #
        # Bug history (2026-04-29): previously we had `fill_price = taking/making`
        # which gave shares-per-USDC (i.e., 1/price), inflating entry_price by ~3x.
        # And `fill_shares = making` gave USDC instead of shares.  Both inverted.
        taking = float(response.get("takingAmount", 0) or 0)
        making = float(response.get("makingAmount", 0) or 0)
        if taking > 0 and making > 0:
            fill_price  = making / taking          # USDC per share (correct)
            fill_shares = taking                    # shares received (correct)
        else:
            fill_price  = limit_price
            fill_shares = shares

        # Phase 9 correctness fix: Polymarket's matching engine returns
        # `status="matched"` synchronously when the order matches against
        # the book — but the trade is NOT yet on chain at that point.  It
        # progresses MATCHED → MINED → CONFIRMED, and can revert (FAILED)
        # during mining.  So an engine-match is NOT a confirmed fill.
        #
        # Always log the position as `pending` here; the user-channel WS
        # (or REST safety-net poll) will flip to `filled` only on the
        # CONFIRMED trade event.  trade_status starts as `matched` for
        # immediate matches so we have an audit trail of the lifecycle.
        if order_status in ("matched",):
            fill_status   = "pending"
            actual_entry  = fill_price
            actual_shares = fill_shares
            initial_trade_status = "matched"
            _placement_msg = (
                f"live BUY MATCHED (engine): {side} ${final_size_usdc:.2f} "
                f"@ {actual_entry:.4f} {signal.get('city')} {signal.get('date')} "
                f"{contract_id[:12]} order={order_id[:12]} — awaiting on-chain CONFIRMED"
            )
        elif order_status in ("live", "delayed"):
            fill_status   = "pending"
            actual_entry  = limit_price
            actual_shares = final_size_usdc / limit_price if limit_price > 0 else shares
            initial_trade_status = None
            _placement_msg = (
                f"live BUY resting on book: {side} ${final_size_usdc:.2f} "
                f"@ {limit_price:.4f} {signal.get('city')} {signal.get('date')} "
                f"{contract_id[:12]} order={order_id[:12]}"
            )
        else:
            # unmatched or unexpected — do not log a position
            from activity import log_activity
            log_activity(
                "BUY", level="WARN",
                message=(
                    f"live BUY unexpected status={order_status!r}: {side} "
                    f"${size_usdc:.2f} {contract_id[:12]} — not logging position"
                ),
                contract_id=contract_id, order_status=order_status,
            )
            return {"status": "unmatched", "order_id": order_id, "order_status": order_status}

        # Override the pre-call estimates in position_kwargs with the
        # actual fill values from the CLOB response (may differ slightly
        # for matched orders due to mid-vs-touch slippage).
        live_kwargs = dict(position_kwargs)
        live_kwargs["entry_price"] = actual_entry
        live_kwargs["shares"]      = actual_shares
        # If the ask-depth cap reduced the order, record the SUBMITTED
        # size — not the originally-intended size.  Otherwise exposure
        # caps would over-count the row during the pending window.
        # update_position_fill recomputes from actual fills once they
        # land (which may shrink further on partial fills).
        live_kwargs["size_usdc"]   = final_size_usdc
        position_id = insert_position(
            **live_kwargs,
            order_id    = order_id,
            is_paper    = 0,
            fill_status = fill_status,
        )

        # Record this order in the position_orders ledger (Phase B,
        # 2026-04-30).  Future top-up gap calc reads from here to avoid
        # double-committing capital on top of resting partial fills.
        try:
            from db import insert_position_order
            ledger_intended_shares = (
                final_size_usdc / limit_price if limit_price > 0 else 0
            )
            insert_position_order(
                position_id     = position_id,
                order_id        = order_id,
                role            = "entry",
                intended_usdc   = final_size_usdc,
                intended_shares = ledger_intended_shares,
                limit_price     = limit_price,
                status          = "pending",
                trade_status    = initial_trade_status,
            )
        except Exception as _e:
            logger.debug(f"position_orders insert (entry) failed (non-fatal): {_e}")

        from activity import log_activity
        log_activity(
            "BUY", position_id=position_id,
            message=_placement_msg,
            mode="live", side=side,
            size_usdc=final_size_usdc, intended_size_usdc=size_usdc,
            limit_price=limit_price, actual_entry=actual_entry,
            order_id=order_id, order_status=order_status,
            sweepable_usdc=sweepable_usdc,
        )

        # Stamp initial lifecycle stage so the WS handler's monotonic
        # advancement starts from the right baseline (otherwise a delayed
        # MATCHED event after CONFIRMED would be allowed to advance from None).
        if initial_trade_status:
            try:
                from db import update_position_trade_status
                update_position_trade_status(
                    position_id=position_id, new_status=initial_trade_status,
                    side="entry",
                )
            except Exception as _e:
                logger.debug(f"trade_status stamp failed (non-fatal): {_e}")

        # Subscribe the user-channel WS to this market for fill events
        try:
            from user_ws import add_markets
            gid = signal.get("gamma_market_id") or position_kwargs.get("gamma_market_id")
            if gid:
                add_markets([gid])
        except Exception:
            pass
        mark_outcome_executed(contract_id, signal.get("scan_timestamp", ""))
        # Subscribe to real-time price updates for this token
        try:
            from price_ws import add_tokens
            add_tokens([token_id])
        except Exception:
            pass

        return {
            "status":       "placed",
            "order_id":     order_id,
            "order_status": order_status,
            "position_id":  position_id,
            "fill_status":  fill_status,
            "entry_price":  actual_entry,
            "shares":       actual_shares,
        }

    except Exception as e:
        if _is_geoblock_error(e):
            _trip_geoblock_circuit(e)
            return _short_circuited_response()
        logger.exception(f"Execution error for {contract_id}: {e}")
        return {"status": "error", "reason": str(e)}


def cancel_order(order_id: str, client: ClobClient) -> bool:
    """Cancel an open order by ID.

    v2 renamed `client.cancel(order_id)` → `client.cancel_orders([order_id])`.
    The old method took a single string; the new takes a list — even
    for cancelling one order.

    On success:
      1. Marks the position_orders ledger row as cancelled (terminal
         status, won't count toward committed_usdc anymore).
      2. Synchronously clears the parent position's in-flight pointer
         for whichever role this order was playing (topup / entry / exit).
         Without (2), a cancel from the overcommit sweep leaves
         `positions.pending_topup_order_id` pointing at a dead order,
         and the dashboard's In-Flight Top-ups table shows phantom
         entries until the WS happens to deliver a CANCELLATION event
         (which it often misses, hence this defensive cleanup).
    """
    if PAPER_TRADE:
        logger.info(f"[PAPER] Would cancel order {order_id}")
        return True
    try:
        resp = client.cancel_orders([order_id])
        if not resp:
            return False
        # ---- Step 1: mark the ledger row terminal ---------------------
        try:
            from db import update_position_order_status
            update_position_order_status(
                order_id         = order_id,
                status           = "cancelled",
                cancelled_reason = "cancelled_by_bot",
                closed           = True,
            )
        except Exception as _e:
            logger.debug(f"ledger cancel-mark failed (non-fatal): {_e}")
        # ---- Step 2: clear the parent position's in-flight pointer ----
        # Mirrors apply_order_event CANCELLATION semantics so we don't
        # depend on the WS catching the cancel.  Role routing matches
        # fill_handler.py:apply_order_event.
        try:
            from db import (
                get_position_by_order_id, classify_position_role,
                clear_position_topup_pending, cancel_position,
            )
            from datetime import datetime
            from zoneinfo import ZoneInfo
            pos = get_position_by_order_id(order_id)
            if pos is not None:
                role = classify_position_role(pos, order_id)
                pid  = pos["id"]
                if role == "topup":
                    clear_position_topup_pending(pid)
                elif role == "entry" and (pos.get("fill_status") or "") == "pending":
                    cancel_position(
                        position_id      = pid,
                        cancelled_reason = "cancelled_by_bot",
                        exit_time        = datetime.now(
                            ZoneInfo("America/Chicago")
                        ).isoformat(),
                    )
                # role == 'exit': leave in 'exiting' for the ladder
                # advancer to retry next cycle (matches WS path).
        except Exception as _e:
            logger.debug(f"position-pointer cleanup after cancel failed (non-fatal): {_e}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel {order_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Orderbook helpers (used by execution + ranking)
# ---------------------------------------------------------------------------

def _orderbook_levels(side_entries) -> list[tuple[float, float]]:
    """Normalize bids/asks across response shapes.  Returns [(price, size), ...]
    with garbage entries dropped.  Order is preserved (caller picks min/max)."""
    out: list[tuple[float, float]] = []
    if side_entries is None:
        return out
    for e in side_entries:
        p = getattr(e, "price", None)
        s = getattr(e, "size",  None)
        if p is None and isinstance(e, dict):
            p = e.get("price")
            s = e.get("size")
        try:
            pf = float(p) if p is not None else None
            sf = float(s) if s is not None else None
        except (TypeError, ValueError):
            continue
        if pf is None or sf is None or pf <= 0 or sf <= 0:
            continue
        out.append((pf, sf))
    return out


def get_orderbook_snapshot(client: ClobClient, token_id: str) -> dict | None:
    """Fetch + normalize an orderbook snapshot for `token_id`.

    Returns a dict with:
        best_bid, best_ask, spread_cents (None if either side empty)
        asks_sorted_asc: [(price, size), ...]   (cheapest first)
        bids_sorted_desc: [(price, size), ...]  (best first)
    Or None on any failure (caller should treat as "no book data, fall back").

    This is the single entry point for orderbook reads — keeps the v1/v2
    shape normalization in one place and makes mocking easy in tests.
    """
    try:
        ob = client.get_order_book(token_id)
        if ob is None:
            return None
        raw_bids = getattr(ob, "bids", None) or (ob.get("bids") if isinstance(ob, dict) else None)
        raw_asks = getattr(ob, "asks", None) or (ob.get("asks") if isinstance(ob, dict) else None)
        bids = sorted(_orderbook_levels(raw_bids), key=lambda x: x[0], reverse=True)
        asks = sorted(_orderbook_levels(raw_asks), key=lambda x: x[0])
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        if best_bid is not None and best_ask is not None:
            spread_cents = round((best_ask - best_bid) * 100, 2)
        else:
            spread_cents = None
        return {
            "best_bid":         best_bid,
            "best_ask":         best_ask,
            "spread_cents":     spread_cents,
            "asks_sorted_asc":  asks,
            "bids_sorted_desc": bids,
        }
    except Exception as e:
        logger.warning(f"orderbook fetch failed for {token_id[:16]}...: {e}")
        return None


def compute_sweep_limit(
    client: ClobClient,
    token_id: str,
    *,
    intended_price: float,
    max_cap: float,
    walk_cents: int = 1,
) -> tuple[float, dict]:
    """Compute the GTC limit price that sweeps cheap liquidity then rests.

    Strategy:
      1. Fetch the orderbook.
      2. limit = min(best_ask + walk_cents/100, max_cap).
         Polymarket's matcher will fill against any asks ≤ limit (cheapest
         first) and rest the remainder at our limit on the book.
      3. If the book can't be fetched or has no asks, fall back to the
         current heuristic (intended_price * 1.005) — never refuse to
         place an order just because of a transient orderbook fetch failure.

    Returns (limit_price, diagnostics) where diagnostics has:
        source:             "sweep" | "fallback_no_book" | "fallback_no_asks"
                            | "fallback_capped_at_max"
        best_ask, best_bid, spread_cents:  (or None)
        sweepable_usdc:     dollar value of asks at prices ≤ limit
    """
    fallback = round(min(intended_price * 1.005, 0.99), 4)

    snap = get_orderbook_snapshot(client, token_id)
    if snap is None:
        return fallback, {"source": "fallback_no_book"}

    best_ask = snap["best_ask"]
    if best_ask is None:
        return fallback, {"source": "fallback_no_asks", "best_bid": snap["best_bid"]}

    raw_limit = best_ask + walk_cents / 100.0
    capped    = min(raw_limit, max_cap)
    # Round to Polymarket's tick size (0.01 most markets — we round to
    # 4dp to be safe for fractional-cent ticks).
    limit     = round(capped, 4)

    # Sweepable USDC = sum of (price × size) for asks ≤ limit
    sweepable = sum(
        p * s for (p, s) in snap["asks_sorted_asc"] if p <= limit + 1e-9
    )

    return limit, {
        "source":         ("sweep" if raw_limit <= max_cap else "fallback_capped_at_max"),
        "best_ask":       best_ask,
        "best_bid":       snap["best_bid"],
        "spread_cents":   snap["spread_cents"],
        "sweepable_usdc": round(sweepable, 4),
    }


# ---------------------------------------------------------------------------
# Exit (sell) order placement
# ---------------------------------------------------------------------------

def _get_best_bid(client: ClobClient, token_id: str) -> float | None:
    """Look up the current best bid for `token_id` from the CLOB orderbook.
    Returns None on any failure — caller should treat that as "can't cross
    the spread, fall back to a conservative price."

    py_clob_client_v2 returns a dict like:
        {"bids": [{"price":"0.01","size":"..."}, ...,
                  {"price":"0.53","size":"..."}],   # ASCENDING
         "asks": [...], "tick_size": "...", "neg_risk": ...}
    Bids are sorted ASCENDING (worst first, best last) — confirmed
    against live API on 2026-04-29.  Don't trust the order; take the
    max explicitly.  v1's `OrderBookSummary` may have differed.
    """
    try:
        ob = client.get_order_book(token_id)
        # Normalize across v1 OrderBookSummary (object) vs v2 dict
        bids = getattr(ob, "bids", None)
        if bids is None and isinstance(ob, dict):
            bids = ob.get("bids", [])
        if not bids:
            return None

        def _price_of(entry) -> float | None:
            p = getattr(entry, "price", None)
            if p is None and isinstance(entry, dict):
                p = entry.get("price")
            try:
                return float(p) if p is not None else None
            except (TypeError, ValueError):
                return None

        prices = [p for p in (_price_of(b) for b in bids) if p is not None]
        return max(prices) if prices else None
    except Exception as e:
        logger.warning(f"orderbook fetch failed for {token_id[:16]}...: {e}")
        return None


def execute_exit(
    position: dict,
    intended_exit_price: float,
    exit_reason: str,
    client: ClobClient | None = None,
    *,
    retry_count: int = 0,
    cross_spread: bool = False,
) -> dict:
    """Place a sell order to exit a live position, or simulate it in paper.

    PAPER mode (or client is None):
        Updates DB with status='closed', actual_exit_price=intended,
        and the realized PnL.  Identical to the previous DB-only exit path.

    LIVE mode:
        Picks a sell limit price using bot.exit_ladder:
          retry_count in [0, 3] → intended × {0.99, 0.98, 0.97, 0.96}
          retry_count >= 4 (or cross_spread=True) → best_bid - 1 tick
        Places a CLOB sell, sets status='exiting', stores the order_id and
        intended price.  The monitor loop will later confirm fills or
        escalate to the next ladder rung if unfilled.

    Returns a dict describing the action taken — caller logs / inspects.
    """
    from db import update_position_outcome, update_position_exit_pending
    from exit_ladder import (
        ladder_price, is_ladder_exhausted, cross_spread_price,
    )
    from datetime import datetime
    from zoneinfo import ZoneInfo

    pid           = position["id"]
    contract_id   = position.get("contract_id", "")
    side          = position.get("side", "YES")
    entry_price   = float(position.get("entry_price") or 0)
    shares        = float(position.get("shares") or 0)
    token_id      = (position.get("yes_token_id") if side == "YES"
                     else position.get("no_token_id"))

    now_iso = datetime.now(ZoneInfo("America/Chicago")).isoformat()

    # -----------------------------------------------------------------------
    # Paper mode — same as before
    # -----------------------------------------------------------------------
    if PAPER_TRADE or client is None:
        realized_pnl = round((intended_exit_price - entry_price) * shares, 4)
        exit_snap = get_latest_snapshot_id_for_contract(contract_id)
        update_position_outcome(
            position_id=pid,
            exit_price=intended_exit_price,
            exit_time=now_iso,
            pnl=realized_pnl,
            status="closed",
            exit_reason=exit_reason,
            exit_snapshot_id=exit_snap,
        )
        return {
            "status":      "paper_closed",
            "position_id": pid,
            "exit_price":  intended_exit_price,
            "pnl":         realized_pnl,
        }

    # -----------------------------------------------------------------------
    # Live mode — pick limit price + place CLOB sell
    # -----------------------------------------------------------------------
    if _geoblock_tripped:
        return _short_circuited_response()

    if not token_id:
        logger.error(f"execute_exit: no token_id for pos {pid} side={side}")
        return {"status": "error", "reason": "missing_token_id"}

    # Decide pricing: ladder rung vs cross-spread escalation.
    # Always fetch the live best bid so ladder rungs can re-anchor to
    # the current book (clamp = bid + tick, see exit_ladder.ladder_price).
    # Without this, later rungs post against a stale trigger and become
    # unfillable when the market has moved.
    crossed = False
    bid = _get_best_bid(client, token_id)
    if cross_spread or is_ladder_exhausted(retry_count):
        if bid is None or bid <= 0:
            # Defensive: if we can't get the book, fall back to the most
            # aggressive ladder rung (4% give-up) rather than abandoning
            # the exit entirely.  This trades exit price for liveness.
            limit_price = round(intended_exit_price * SELL_LADDER_MULTIPLIERS_LAST,
                                4)
            logger.warning(
                f"execute_exit pid={pid}: cross-spread requested but no bid "
                f"available; falling back to {SELL_LADDER_MULTIPLIERS_LAST:.0%} "
                f"of intended (={limit_price})"
            )
        else:
            limit_price = cross_spread_price(bid)
            crossed = True
    else:
        rung_price = ladder_price(
            intended_exit_price,
            retry_count,
            current_best_bid = bid,    # bid-anchored clamp; None = legacy
        )
        if rung_price is None:
            # Shouldn't happen given the is_ladder_exhausted check above,
            # but defensive
            return {"status": "error", "reason": f"bad retry_count={retry_count}"}
        limit_price = rung_price

    # ---- "Let it decay" gate ----
    # If the computed limit is below the platform's recoverable
    # threshold, skip the exit attempt entirely and let the position
    # ride to market resolution (where it'll settle at zero anyway).
    # Selling at minimum tick generates fees with negligible recovery
    # and the CLOB rejects sub-minimum limits with "invalid price".
    #
    # Polymarket runs two tick-size regimes per market:
    #   * $0.001 tick -> minimum price $0.001 (most markets)
    #   * $0.01 tick  -> minimum price $0.01  (some markets, common
    #     for thin/decayed bins)
    # We can't tell which a market uses without extra metadata, so the
    # default floor is set to $0.011 -- just above the higher-of-two
    # minimums.  This safely skips both "invalid price (0.0005)" (from
    # $0.001-tick markets) and "invalid price (0.0099), min: 0.01"
    # (from $0.01-tick markets) errors at the cost of giving up on
    # exits between $0.005 and $0.011 -- which were never going to
    # recover meaningful capital anyway.  Tunable via env.
    import os as _os
    _MIN_EXIT_LIMIT = float(_os.getenv("MIN_EXIT_LIMIT_USDC", "0.011"))
    if limit_price < _MIN_EXIT_LIMIT:
        logger.info(
            f"execute_exit pid={pid}: skipping -- computed limit "
            f"${limit_price:.4f} < MIN_EXIT_LIMIT_USDC ${_MIN_EXIT_LIMIT:.4f}.  "
            f"Position is essentially worthless ({shares:.2f} shares); "
            f"letting it decay to market resolution."
        )
        try:
            from activity import log_activity
            log_activity(
                "SELL", level="INFO", position_id=pid,
                message=(
                    f"exit skipped -- price decayed below recoverable "
                    f"threshold (limit=${limit_price:.4f} < "
                    f"${_MIN_EXIT_LIMIT:.4f}).  Letting position decay "
                    f"to market resolution: {pos.get('city')} "
                    f"{pos.get('date')} {contract_id[:12]}"
                ),
                contract_id=contract_id, side=side,
                limit_price=limit_price, min_exit_limit=_MIN_EXIT_LIMIT,
                shares=shares, exit_reason=exit_reason,
            )
        except Exception:
            pass
        return {
            "status": "skip",
            "reason": "below_recoverable_threshold",
            "limit_price": limit_price,
            "min_exit_limit": _MIN_EXIT_LIMIT,
        }

    # ---- "Dust shares" gate ----
    # py_clob_client_v2 internally rounds order size to 4 decimal places
    # when encoding to CLOB.  A residual position with shares < ~0.01
    # rounds to 0 in that encoding, so the maker_amount = 0 and Polymarket
    # rejects with "invalid amounts, maker and taker amount must be
    # higher than 0".  More importantly, even if the encoding accepted
    # tiny sizes, the recoverable USDC (shares * limit_price) would be
    # below sensible minimums.
    #
    # Skip the exit when shares*limit_price < $0.05 (5 cents).  The
    # orphan_db auto-close path will collect these dust positions on
    # its next pass.  Tunable via env.
    _MIN_EXIT_USDC = float(_os.getenv("MIN_EXIT_USDC", "0.05"))
    _potential_proceeds = float(shares) * float(limit_price)
    if _potential_proceeds < _MIN_EXIT_USDC:
        logger.info(
            f"execute_exit pid={pid}: skipping -- only "
            f"{shares:.4f} shares left at limit ${limit_price:.4f} = "
            f"${_potential_proceeds:.4f} potential proceeds < "
            f"MIN_EXIT_USDC ${_MIN_EXIT_USDC:.4f}.  Dust position; "
            f"letting orphan-db auto-close handle it."
        )
        try:
            from activity import log_activity
            log_activity(
                "SELL", level="INFO", position_id=pid,
                message=(
                    f"exit skipped -- dust shares ({shares:.4f}) at "
                    f"${limit_price:.4f} = ${_potential_proceeds:.4f} "
                    f"< ${_MIN_EXIT_USDC:.4f} floor.  Auto-close will "
                    f"collect: {pos.get('city')} {pos.get('date')} "
                    f"{contract_id[:12]}"
                ),
                contract_id=contract_id, side=side,
                shares=shares, limit_price=limit_price,
                potential_proceeds=_potential_proceeds,
                min_exit_usdc=_MIN_EXIT_USDC,
                exit_reason=exit_reason,
            )
        except Exception:
            pass
        return {
            "status": "skip",
            "reason": "dust_shares",
            "shares": shares,
            "limit_price": limit_price,
            "potential_proceeds": _potential_proceeds,
            "min_exit_usdc": _MIN_EXIT_USDC,
        }

    # Place CLOB sell
    try:
        order_args = OrderArgs(
            price    = limit_price,
            size     = shares,             # selling N shares of the token
            side     = Side.SELL,
            token_id = token_id,
        )
        response = client.create_and_post_order(order_args, order_type=OrderType.GTC)
        if not response or not response.get("success"):
            logger.error(f"execute_exit pid={pid}: order failed: {response}")
            from activity import log_activity
            log_activity(
                "SELL", level="ERROR", position_id=pid,
                message=(
                    f"live SELL placement FAILED: pid={pid} {side} "
                    f"{contract_id[:12]} limit={limit_price:.4f} — {response}"
                ),
                contract_id=contract_id, side=side, limit_price=limit_price,
            )
            return {"status": "failed", "response": str(response)}

        order_id     = response.get("orderID", "")
        order_status = response.get("status", "")

        # Mark the position as exiting; monitor loop will confirm fill
        # or escalate at the next cycle.  We stamp intended_exit_price the
        # FIRST time (retry_count==0) and keep it across retries.
        update_position_exit_pending(
            position_id=pid,
            exit_order_id=order_id,
            exit_intended_price=intended_exit_price,
            exit_retry_count=retry_count,
            exit_reason=exit_reason,
        )

        # Record exit in the position_orders ledger (Phase B).
        try:
            from db import insert_position_order
            insert_position_order(
                position_id     = pid,
                order_id        = order_id,
                role            = "exit",
                intended_usdc   = round(shares * limit_price, 4),
                intended_shares = shares,
                limit_price     = limit_price,
                status          = "pending",
                trade_status    = (
                    "matched" if (order_status or "").lower() == "matched"
                    else None
                ),
            )
        except Exception as _e:
            logger.debug(f"position_orders insert (exit) failed (non-fatal): {_e}")

        # Stamp initial exit lifecycle stage so monotonic advancement starts
        # from the right baseline.  Only stamp when the engine matched
        # immediately (status='matched') — for resting orders the WS will
        # write the lifecycle as it progresses.
        if (order_status or "").lower() in ("matched",):
            try:
                from db import update_position_trade_status
                update_position_trade_status(
                    position_id=pid, new_status="matched", side="exit",
                )
            except Exception:
                pass

        # Subscribe the user-channel WS to this market for fill events
        try:
            from user_ws import add_markets
            gid = position.get("gamma_market_id")
            if gid:
                add_markets([gid])
        except Exception:
            pass

        ladder_label = "CROSS_SPREAD" if crossed else f"RUNG_{retry_count}"
        from activity import log_activity
        log_activity(
            "SELL", position_id=pid,
            message=(
                f"live SELL {ladder_label}: pid={pid} {side} {contract_id[:12]} "
                f"limit={limit_price:.4f} intended={intended_exit_price:.4f} "
                f"reason={exit_reason} order={order_id[:12]} status={order_status}"
            ),
            mode="live", side=side, limit_price=limit_price,
            intended_exit_price=intended_exit_price,
            retry_count=retry_count, crossed_spread=crossed,
            exit_reason=exit_reason, order_id=order_id,
        )

        return {
            "status":         "exit_pending",
            "position_id":    pid,
            "order_id":       order_id,
            "limit_price":    limit_price,
            "retry_count":    retry_count,
            "crossed_spread": crossed,
        }

    except Exception as e:
        if _is_geoblock_error(e):
            _trip_geoblock_circuit(e)
            return _short_circuited_response()
        # ---- Graceful "not enough balance / allowance" handling ---------
        # When Polymarket rejects an exit because we're trying to sell more
        # shares than we hold on chain, our positions.shares is stale (the
        # exit-fill code historically didn't decrement on partial fills,
        # see the Layer 3 fix in fill_handler).  Self-heal by syncing
        # positions.shares to chain truth, then deciding what to do:
        #   * chain_shares == 0 → mark position closed (nothing left to sell)
        #   * chain_shares > 0  → update DB and let the next cycle retry
        # Without this handler, the bleed circuit (or any exit attempt)
        # spams ERROR-level rejections every cycle forever.
        msg = str(e)
        if "not enough balance" in msg or "allowance" in msg:
            _result = _handle_exit_balance_mismatch(
                pid              = pid,
                contract_id      = contract_id,
                token_id         = token_id,
                intended_shares  = shares,
                limit_price      = limit_price,
                exit_reason      = exit_reason,
            )
            if _result is not None:
                return _result
            # _result is None → fall through to the generic error below
        logger.exception(f"execute_exit pid={pid}: exception: {e}")
        return {"status": "error", "reason": str(e)}


def _handle_exit_balance_mismatch(
    *,
    pid: int,
    contract_id: str,
    token_id: str,
    intended_shares: float,
    limit_price: float,
    exit_reason: str,
) -> dict | None:
    """Recover from a Polymarket 'not enough balance' rejection on an exit
    by re-syncing positions.shares to on-chain truth.  Returns:

      * dict {status='closed_via_balance_recovery', ...} when the chain
        confirms zero holdings (position is closed)
      * dict {status='shares_resynced', new_shares=...} when the chain
        has fewer shares than we tried to sell but still > 0 (next
        ladder cycle will retry at the correct size)
      * None when the recovery itself failed — caller should fall through
        to the generic exception path

    Logs verbosely (WARN) so the operator can audit the self-heal.
    """
    try:
        from polymarket import get_data_api_positions
        from config import WALLET_ADDRESS
        from db import _get_conn
        from activity import log_activity
        from datetime import datetime
        from zoneinfo import ZoneInfo

        positions = get_data_api_positions(WALLET_ADDRESS) or []
        chain_shares = 0.0
        for p in positions:
            if p.get("asset") == token_id:
                chain_shares = float(p.get("size") or 0)
                break

        # Fetch current DB shares so we have before/after for the audit log
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT shares, entry_price, size_usdc, entry_fees, exit_fees "
                "FROM positions WHERE id = ?",
                (pid,),
            ).fetchone()
            db_shares = float(row["shares"] or 0) if row else 0.0
            entry_price = float(row["entry_price"] or 0) if row else 0.0
            entry_size  = float(row["size_usdc"] or 0) if row else 0.0
            entry_fees  = float(row["entry_fees"] or 0) if row else 0.0
            exit_fees   = float(row["exit_fees"] or 0) if row else 0.0

        logger.warning(
            f"[EXIT BALANCE-MISMATCH] pid={pid} {contract_id[:12]} "
            f"intended_sell={intended_shares:.4f}, db_shares={db_shares:.4f}, "
            f"chain_shares={chain_shares:.4f} — self-healing"
        )

        if chain_shares <= 0.001:
            # Chain says nothing left.  Position is effectively closed.
            # Estimate exit price from the limit (best info we have here);
            # the proper exit-fill accounting would have given us a precise
            # value but didn't run for the partial fills.
            exit_proxy_price = limit_price if limit_price > 0 else 0.0
            # pnl ≈ (proxy - entry) × original_shares - fees.  Imprecise but
            # better than leaving stale 28.57 shares + no pnl.
            pnl = round(
                (exit_proxy_price - entry_price) * db_shares
                - entry_fees - exit_fees, 4
            )
            now_iso = datetime.now(ZoneInfo("America/Chicago")).isoformat()
            with _get_conn() as conn:
                conn.execute("""
                    UPDATE positions
                    SET shares             = 0,
                        status             = 'closed',
                        actual_exit_price  = COALESCE(actual_exit_price, ?),
                        exit_price         = COALESCE(exit_price, ?),
                        exit_time          = COALESCE(exit_time, ?),
                        pnl                = COALESCE(pnl, ?),
                        unrealized_pnl     = 0,
                        exit_reason        = COALESCE(exit_reason, '') || ?
                    WHERE id = ?
                """, (
                    exit_proxy_price, exit_proxy_price, now_iso, pnl,
                    "|balance_mismatch_close", pid,
                ))
            log_activity(
                "REPAIR",
                f"pid={pid} {contract_id[:12]} closed via balance-mismatch "
                f"recovery (chain=0, db was {db_shares:.4f}). "
                f"Approx pnl={pnl:.4f}",
                level="WARN",
                position_id=pid,
                repair_kind="exit_balance_mismatch_close",
                db_shares_before=db_shares,
                chain_shares=chain_shares,
                approx_pnl=pnl,
                exit_proxy_price=exit_proxy_price,
            )
            return {
                "status":     "closed_via_balance_recovery",
                "position_id": pid,
                "chain_shares": chain_shares,
                "approx_pnl":   pnl,
            }

        # Chain has SOME shares but fewer than we tried to sell — sync DB
        # to chain truth.  Don't retry inline (we'd nest call stacks);
        # the next exit-ladder cycle will see the corrected shares and
        # try a smaller sell.
        with _get_conn() as conn:
            conn.execute(
                "UPDATE positions SET shares = ? WHERE id = ?",
                (chain_shares, pid),
            )
        log_activity(
            "REPAIR",
            f"pid={pid} {contract_id[:12]} shares re-synced to chain "
            f"({db_shares:.4f} -> {chain_shares:.4f}); next exit cycle will "
            f"retry at corrected size",
            level="WARN",
            position_id=pid,
            repair_kind="exit_balance_mismatch_resync",
            db_shares_before=db_shares,
            chain_shares=chain_shares,
        )
        return {
            "status":       "shares_resynced",
            "position_id":  pid,
            "old_shares":   db_shares,
            "new_shares":   chain_shares,
        }
    except Exception as recovery_err:
        logger.warning(
            f"[EXIT BALANCE-MISMATCH] pid={pid}: recovery itself failed: "
            f"{recovery_err}.  Falling through to generic error path."
        )
        return None


# ---------------------------------------------------------------------------
# Fee backfill — Polymarket's WS trade events don't include fee_rate_bps,
# so the fill handler always records fees as 0.  This function fetches the
# real trade history from the CLOB (which DOES include fee_rate_bps) and
# updates positions.entry_fees / exit_fees / pnl_net accordingly.
#
# Called automatically after an exit completes (in fill_handler), and can
# also be run manually for any closed position via:
#     python -m scripts.backfill_position_fees [--pid N | --all-closed]
#
# Polymarket fee mechanics (as of 2026-04):
#   * Makers pay 0; takers pay fee_rate_bps (1000 = 10% as observed in
#     production trades — confirmed from live get_trades samples).
#   * Each trade event in get_trades carries `fee_rate_bps`, `size`,
#     `price`, `taker_order_id`, and a `maker_orders` list.  We match
#     trades to our orders via taker_order_id (when we crossed) or the
#     maker_orders[*].order_id (when we rested and were taken).
# ---------------------------------------------------------------------------

def backfill_position_fees(
    pid: int,
    client: ClobClient | None = None,
    *,
    commit: bool = True,
) -> dict:
    """Compute (and optionally write) positions.entry_fees / exit_fees /
    pnl_net for `pid` by summing real fees from Polymarket trade history.

    `commit=True`  (default): writes the computed values to the DB.
    `commit=False`: returns the computed values without writing — useful
                    for dry-run / preview.

    Returns a result dict with the new values + how many trades matched:
        {entry_fees, exit_fees, pnl_net, n_trades_matched, gross_pnl,
         old_entry_fees, old_exit_fees, old_pnl_net, committed}
    Empty dict {} when nothing was done (paper, no client, no trades,
    no token_id, etc.).

    Idempotent — running twice produces the same totals (fees are
    REPLACED, not accumulated).
    """
    if client is None or PAPER_TRADE:
        return {}

    from db import _get_conn
    from py_clob_client_v2.clob_types import TradeParams

    with _get_conn() as conn:
        pos_row = conn.execute("""
            SELECT contract_id, yes_token_id, no_token_id, side, order_id,
                   exit_order_id, pending_topup_order_id, pnl,
                   entry_fees, exit_fees, pnl_net
            FROM positions WHERE id = ?
        """, (pid,)).fetchone()
        if pos_row is None:
            return {}
        pos = dict(pos_row)
        ledger_orders = [dict(r) for r in conn.execute("""
            SELECT order_id, role FROM position_orders
            WHERE position_id = ?
        """, (pid,)).fetchall()]

    side = pos.get("side", "YES")
    token_id = pos.get("yes_token_id") if side == "YES" else pos.get("no_token_id")
    if not token_id:
        logger.debug(f"backfill_position_fees pid={pid}: no token_id, skipping")
        return {}
    market = pos.get("contract_id", "")
    if not market:
        return {}

    # Build order_id -> role map (entry / topup / exit) for matching trades.
    # The position_orders ledger is the source of truth for new positions;
    # legacy positions fall back to the columns directly on the row.
    order_role: dict[str, str] = {}
    for row in ledger_orders:
        oid = row.get("order_id")
        role = row.get("role")
        if oid and role:
            order_role[oid] = role
    for legacy_field, legacy_role in (
        ("order_id", "entry"),
        ("exit_order_id", "exit"),
        ("pending_topup_order_id", "topup"),
    ):
        oid = pos.get(legacy_field)
        if oid and oid not in order_role:
            order_role[oid] = legacy_role
    if not order_role:
        logger.debug(f"backfill_position_fees pid={pid}: no orders to match")
        return {}

    # Fetch trades for this market+asset from Polymarket.  market+asset_id
    # filter narrows to a small number even for very active accounts; we
    # then match by order_id locally.  Pull all pages.
    try:
        trades = client.get_trades(
            params=TradeParams(market=market, asset_id=token_id),
            only_first_page=False,
        )
    except Exception as e:
        logger.warning(
            f"backfill_position_fees pid={pid}: get_trades failed: {e}"
        )
        return {}
    if not trades:
        return {}

    # Sum fees per role.
    entry_fee_total = 0.0
    exit_fee_total = 0.0
    n_matched = 0
    for t in trades:
        # Match: this trade's taker_order_id is one of ours (we crossed),
        # OR one of the maker_orders's order_id is ours (we rested and got
        # taken).  Either way, role tells us entry vs topup vs exit.
        oid = t.get("taker_order_id") or ""
        role = order_role.get(oid)
        if role is None:
            for mo in (t.get("maker_orders") or []):
                m_oid = mo.get("order_id")
                if m_oid in order_role:
                    role = order_role[m_oid]
                    break
        if role is None:
            continue   # not our trade

        # Fee = trade_usdc × (fee_rate_bps / 10000).  Only takers pay; for
        # maker matches the API returns fee_rate_bps='' or '0' in our slot.
        try:
            sz    = float(t.get("size") or 0)
            price = float(t.get("price") or 0)
            bps_raw = t.get("fee_rate_bps")
            bps   = float(bps_raw) if (bps_raw not in (None, "")) else 0.0
        except (TypeError, ValueError):
            continue
        fee_usdc = sz * price * (bps / 10_000.0)
        if role == "exit":
            exit_fee_total += fee_usdc
        else:
            # entry + topup are the same fee bucket on the position row
            entry_fee_total += fee_usdc
        n_matched += 1

    entry_fee_total = round(entry_fee_total, 4)
    exit_fee_total  = round(exit_fee_total, 4)
    gross_pnl = float(pos.get("pnl") or 0)
    new_pnl_net = round(gross_pnl - entry_fee_total - exit_fee_total, 4)

    if commit:
        with _get_conn() as conn:
            conn.execute("""
                UPDATE positions
                SET entry_fees = ?,
                    exit_fees  = ?,
                    pnl_net    = ?
                WHERE id = ?
            """, (entry_fee_total, exit_fee_total, new_pnl_net, pid))

    return {
        "entry_fees":       entry_fee_total,
        "exit_fees":        exit_fee_total,
        "pnl_net":          new_pnl_net,
        "gross_pnl":        gross_pnl,
        "n_trades_matched": n_matched,
        # Pre-update values for audit / dry-run reporting
        "old_entry_fees":   float(pos.get("entry_fees") or 0),
        "old_exit_fees":    float(pos.get("exit_fees") or 0),
        "old_pnl_net":      pos.get("pnl_net"),
        "committed":        commit,
    }


# Last rung of the ladder — used as a fallback when cross-spread can't
# look up a bid.  Imported lazily inside execute_exit to avoid bot/
# import-order issues at module load.
from exit_ladder import SELL_LADDER_MULTIPLIERS  # noqa: E402
SELL_LADDER_MULTIPLIERS_LAST = SELL_LADDER_MULTIPLIERS[-1]


# ---------------------------------------------------------------------------
# Order status / fill reconciliation
# ---------------------------------------------------------------------------

def get_order_status(order_id: str, client: ClobClient | None) -> dict | None:
    """Defensive wrapper around CLOB get_order(order_id).

    Returns the raw response dict (or None on failure / paper mode).
    Caller should treat None as "unknown — try again next cycle."
    """
    if not order_id or PAPER_TRADE or client is None:
        return None
    try:
        # py_clob_client exposes get_order(order_id).  Some versions return
        # an object with attributes; some return a dict.  Normalize to dict.
        resp = client.get_order(order_id)
        if resp is None:
            return None
        if isinstance(resp, dict):
            return resp
        # Fallback for object-style responses
        return {k: getattr(resp, k, None)
                for k in ("status", "size", "size_matched", "original_size",
                          "price", "price_matched", "side")
                if hasattr(resp, k)}
    except Exception as e:
        logger.warning(f"get_order_status({order_id[:12]}...) failed: {e}")
        return None


def is_order_fully_filled(status_resp: dict | None) -> bool:
    """True iff the CLOB order has been completely matched.

    CLOB statuses we treat as "filled":
      * 'matched'   — fully matched on placement
      * 'filled'    — completed after partial fills accumulated to full
      * 'completed' — alternate terminology in some responses

    Defensive across response shapes — if we can't tell, returns False
    (caller defaults to "still pending; retry next cycle").
    """
    if not status_resp:
        return False
    status = (status_resp.get("status") or "").lower()
    if status in ("matched", "filled", "completed"):
        return True
    # Some responses use size_matched + original_size; check that too
    size_matched = float(status_resp.get("size_matched") or 0)
    original = float(
        status_resp.get("original_size")
        or status_resp.get("size")
        or 0
    )
    if original > 0 and size_matched >= original - 1e-9:
        return True
    return False


def is_order_cancelled(status_resp: dict | None) -> bool:
    """True iff the CLOB order was cancelled (by us or externally)."""
    if not status_resp:
        return False
    status = (status_resp.get("status") or "").lower()
    return status in ("cancelled", "canceled", "expired")


def extract_fee_amount(
    response: dict | None, fill_amount_usdc: float = 0.0,
) -> float:
    """Best-effort extraction of the USDC fee paid on an order.

    Tries direct fee fields first, then derives from feeRateBps × fill
    amount as fallback.  Returns 0.0 if nothing usable — better to under-
    report fees than block the workflow on a missing field.

    Common shapes seen in CLOB responses:
        { "fee": "0.5500" }                                   # direct USDC
        { "feesAccrued": "0.55" }                             # alt naming
        { "feeRateBps": 200, "takingAmount": "27.5" }         # rate + amount
        { "feeRateBps": 200 }                                 # rate only — needs caller's fill_amount
    """
    if not response:
        return 0.0

    # Direct fee fields
    for key in ("fee", "feesAccrued", "feesPaid"):
        val = response.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    # Derive from fee rate × fill amount.
    bps = response.get("feeRateBps") or response.get("fee_rate_bps")
    if bps is not None:
        try:
            rate = float(bps) / 10_000.0
        except (TypeError, ValueError):
            rate = 0.0
        if rate > 0:
            # Prefer the response's own takingAmount if present (more accurate
            # than caller's pre-fill estimate, which may be the limit price)
            taking = response.get("takingAmount") or response.get("taking_amount")
            if taking is not None:
                try:
                    return float(taking) * rate
                except (TypeError, ValueError):
                    pass
            if fill_amount_usdc > 0:
                return float(fill_amount_usdc) * rate

    return 0.0


def extract_fee_rate_bps(response: dict | None) -> int | None:
    """Pull the fee rate (basis points) from a CLOB response, or None.
    Logged for diagnostics — different markets charge different rates."""
    if not response:
        return None
    for key in ("feeRateBps", "fee_rate_bps"):
        val = response.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def extract_fill_price(status_resp: dict | None, fallback: float) -> float:
    """Get the actual avg fill price ($/share) from a CLOB get_order response.

    Source preference:
      1. Explicit `price` field on the order (most reliable for v2 — this
         is the actual matched price, not the limit).
      2. Alternate explicit fields (avg_price, price_matched, filled_price).
      3. makerAmount/takerAmount ratio — but this DEPENDS on order side:
           BUY:  price = makerAmount / takerAmount  (USDC paid / shares received)
           SELL: price = takerAmount / makerAmount  (USDC received / shares sold)
      4. The provided `fallback` (typically limit_price).
    """
    if not status_resp:
        return fallback
    # Most reliable: explicit price field
    for k in ("price", "avg_price", "price_matched", "filled_price"):
        v = status_resp.get(k)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    # Side-aware ratio fallback.  Bug history (2026-04-29): always used
    # `taking/making` regardless of side, inverting BUY prices.
    taking = float(status_resp.get("takingAmount") or 0)
    making = float(status_resp.get("makingAmount") or 0)
    if taking > 0 and making > 0:
        side = (status_resp.get("side") or "").upper()
        if side == "SELL":
            return taking / making
        # Default to BUY semantics — safer than inverting on unknown
        return making / taking
    return fallback


def execute_topup(
    position: dict,
    add_amount_usdc: float,
    client: ClobClient | None = None,
) -> dict:
    """Place a live CLOB buy that adds to an existing position.

    Paper mode: caller (main._run_topups) has already merged via
    update_position_topup; this function shouldn't be called.

    Live mode: posts a limit buy at the same `entry_price * 1.005`
    aggressiveness as initial buys, stamps pending_topup_* fields on
    the parent position so the monitor's reconciliation can:
      * confirm fill → call update_position_topup with actual fill data
      * detect cancel → clear the pending fields

    Skips placement if the position already has an in-flight top-up
    (one at a time per parent).

    Returns a result dict with status: 'placed' | 'paper' | 'skip' | 'failed'.
    """
    from db import update_position_topup_pending, update_position_topup

    pid          = position["id"]
    contract_id  = position.get("contract_id", "")
    side         = position.get("side", "YES")
    entry_price  = float(position.get("entry_price") or 0)
    token_id     = (position.get("yes_token_id") if side == "YES"
                    else position.get("no_token_id"))

    # Don't double-issue a top-up while one is already in flight
    if position.get("pending_topup_order_id"):
        return {
            "status": "skip",
            "reason": "topup_already_pending",
            "pending_order": position["pending_topup_order_id"],
        }

    if entry_price <= 0:
        return {"status": "failed", "reason": "no_entry_price"}

    add_shares = round(add_amount_usdc / entry_price, 4)

    # -----------------------------------------------------------------------
    # Paper mode — merge immediately (matches the original behavior)
    # -----------------------------------------------------------------------
    if PAPER_TRADE or client is None:
        update_position_topup(
            position_id   = pid,
            added_usdc    = add_amount_usdc,
            added_shares  = add_shares,
            new_avg_price = entry_price,
        )
        from activity import log_activity
        log_activity(
            "TOPUP", position_id=pid,
            message=(
                f"paper TOPUP pid={pid} {side} +${add_amount_usdc:.2f} "
                f"@{entry_price:.4f} {contract_id[:12]}"
            ),
            mode="paper", side=side, add_usdc=add_amount_usdc,
            add_shares=add_shares, entry_price=entry_price,
        )
        return {
            "status":       "paper",
            "position_id":  pid,
            "add_usdc":     add_amount_usdc,
            "add_shares":   add_shares,
        }

    # -----------------------------------------------------------------------
    # Live mode — post CLOB buy + stamp pending fields
    # -----------------------------------------------------------------------
    if _geoblock_tripped:
        return _short_circuited_response()

    if not token_id:
        return {"status": "failed", "reason": "missing_token_id"}

    # Same orderbook-aware sweep as execute_signal — capture cheap asks first,
    # rest the remainder at our limit.  See compute_sweep_limit's docstring.
    from config import ORDERBOOK_WALK_CENTS
    try:
        from strategies.market_price_value import MPV_MAX_PRICE as _MAX_BUY_PRICE
    except Exception:
        _MAX_BUY_PRICE = 0.99
    limit_price, _topup_diag = compute_sweep_limit(
        client       = client,
        token_id     = token_id,
        intended_price = entry_price,
        max_cap      = _MAX_BUY_PRICE,
        walk_cents   = ORDERBOOK_WALK_CENTS,
    )
    # Same ask-depth cap as execute_signal — uses the fresh orderbook
    # snapshot from compute_sweep_limit above.
    from config import (
        LIQUIDITY_AWARE_SIZING,
        MAX_TAKE_PCT_OF_ASK_DEPTH,
        MIN_FILLABLE_USDC,
        ENSURE_FILL_MIN_FILLABLE_USDC,
        ENSURE_FILL_STRATEGIES,
    )
    sweepable_usdc_t = float(_topup_diag.get("sweepable_usdc", 0) or 0)
    final_add_usdc = add_amount_usdc
    cap_diag_t = ""

    # Strategy-aware floor: ensure-fill positions (TKH) use a lower
    # min-fillable floor so a thin book doesn't permanently halt the
    # top-up loop's progress toward target_size_usdc.  Same rationale
    # as the matching block in execute_signal.
    pos_strategy = (pos.get("strategy") or ACTIVE_STRATEGY or "").strip()
    is_ensure_fill_t = pos_strategy in ENSURE_FILL_STRATEGIES
    effective_min_fillable_t = (
        ENSURE_FILL_MIN_FILLABLE_USDC if is_ensure_fill_t else MIN_FILLABLE_USDC
    )

    if LIQUIDITY_AWARE_SIZING and sweepable_usdc_t > 0:
        max_take_t = sweepable_usdc_t * MAX_TAKE_PCT_OF_ASK_DEPTH
        if max_take_t < effective_min_fillable_t:
            from activity import log_activity
            log_activity(
                "TOPUP", level="WARN", position_id=pid,
                message=(
                    f"top-up skipped (book too thin): pid={pid} "
                    f"+${add_amount_usdc:.2f} {contract_id[:12]} — "
                    f"acceptable ask depth ${sweepable_usdc_t:.2f}, "
                    f"max take ${max_take_t:.2f} < floor ${effective_min_fillable_t:.2f}"
                    + (f" (ensure-fill {pos_strategy})" if is_ensure_fill_t else "")
                ),
                contract_id=contract_id, sweepable_usdc=sweepable_usdc_t,
                max_take=max_take_t, min_floor=effective_min_fillable_t,
                ensure_fill=is_ensure_fill_t, strategy=pos_strategy,
            )
            return {"status": "skip", "reason": "book_too_thin",
                    "sweepable_usdc": sweepable_usdc_t}
        if add_amount_usdc > max_take_t:
            final_add_usdc = round(max_take_t, 2)
            cap_diag_t = (
                f" CAPPED: intended=${add_amount_usdc:.2f} → ${final_add_usdc:.2f} "
                f"(ask_depth=${sweepable_usdc_t:.2f} × "
                f"{MAX_TAKE_PCT_OF_ASK_DEPTH*100:.0f}%)"
                + (" [ensure-fill: will retry next cycle]"
                   if is_ensure_fill_t else "")
            )
    logger.info(
        f"[SWEEP TOPUP] pid={pid} {contract_id[:12]} "
        f"intended={entry_price:.4f} limit={limit_price:.4f} "
        f"source={_topup_diag.get('source')} "
        f"sweepable=${sweepable_usdc_t:.2f}{cap_diag_t}"
    )

    try:
        # IMPORTANT: OrderArgs.side is the ORDER side.  Top-ups are always
        # BUYs — we're adding to an existing position by buying MORE of
        # the same token.  The position side (YES/NO) is encoded by the
        # token_id (yes_token_id vs no_token_id, selected above).
        order_args = OrderArgs(
            price    = limit_price,
            size     = final_add_usdc / limit_price,
            side     = Side.BUY,
            token_id = token_id,
        )
        response = client.create_and_post_order(order_args, order_type=OrderType.GTC)
        if not response or not response.get("success"):
            from activity import log_activity
            log_activity(
                "TOPUP", level="ERROR", position_id=pid,
                message=(
                    f"live TOPUP placement FAILED: pid={pid} {side} "
                    f"+${add_amount_usdc:.2f} @{limit_price:.4f} "
                    f"{contract_id[:12]} — {response}"
                ),
                contract_id=contract_id, side=side,
                add_usdc=add_amount_usdc, limit_price=limit_price,
            )
            return {"status": "failed", "response": str(response)}

        order_id = response.get("orderID", "")
        # CLOB may immediately match; in that case we still stamp the
        # pending fields and let the monitor's reconciliation do the merge
        # (single code path for fills regardless of latency).
        update_position_topup_pending(
            position_id    = pid,
            order_id       = order_id,
            amount_usdc    = final_add_usdc,
            intended_price = limit_price,
        )

        # Record top-up in the position_orders ledger (Phase B).
        try:
            from db import insert_position_order
            ledger_topup_shares = (
                final_add_usdc / limit_price if limit_price > 0 else 0
            )
            insert_position_order(
                position_id     = pid,
                order_id        = order_id,
                role            = "topup",
                intended_usdc   = final_add_usdc,
                intended_shares = ledger_topup_shares,
                limit_price     = limit_price,
                status          = "pending",
            )
        except Exception as _e:
            logger.debug(f"position_orders insert (topup) failed (non-fatal): {_e}")

        # Subscribe the user-channel WS to this market so the top-up's
        # CONFIRMED trade event is dispatched into fill_handler in real time.
        # (Lifecycle for top-ups is tracked on the parent position's
        # entry-side trade_status — we don't have a separate column.)
        try:
            from user_ws import add_markets
            gid = position.get("gamma_market_id")
            if gid:
                add_markets([gid])
        except Exception:
            pass

        from activity import log_activity
        log_activity(
            "TOPUP", position_id=pid,
            message=(
                f"live TOPUP placed: pid={pid} {side} +${final_add_usdc:.2f} "
                f"@{limit_price:.4f} {contract_id[:12]} order={order_id[:12]} "
                f"status={response.get('status', '')}"
            ),
            mode="live", side=side,
            add_usdc=final_add_usdc, intended_add_usdc=add_amount_usdc,
            limit_price=limit_price, order_id=order_id,
            order_status=response.get("status", ""),
            sweepable_usdc=sweepable_usdc_t,
        )
        return {
            "status":       "placed",
            "position_id":  pid,
            "order_id":     order_id,
            "limit_price":  limit_price,
            "add_usdc":     final_add_usdc,
        }

    except Exception as e:
        if _is_geoblock_error(e):
            _trip_geoblock_circuit(e)
            return _short_circuited_response()
        logger.exception(f"execute_topup pid={pid}: exception: {e}")
        return {"status": "error", "reason": str(e)}


def cancel_exit_order(position: dict, client: ClobClient | None) -> bool:
    """Cancel the active exit order for a position (used when escalating
    to the next ladder rung).  Returns True if cancelled or paper mode."""
    if PAPER_TRADE or client is None:
        return True
    order_id = position.get("exit_order_id")
    if not order_id:
        return True   # nothing to cancel
    return cancel_order(order_id, client)


def get_open_orders(client: ClobClient) -> list[dict]:
    """Get all open orders from CLOB (live mode only)."""
    if PAPER_TRADE:
        return []
    try:
        return client.get_orders() or []
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []
