"""
main.py — Entry point and scheduler for the Polymarket weather arbitrage bot.

Three scheduled loops:

  Discovery loop  — every 4 hours at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
                    Queries Polymarket Gamma API for active temperature markets.

  Trading loop    — every hour at :10  (e.g. 00:10, 01:10, 02:10 …)
                    Analyzes events, generates signals, executes trades.
                    Staggered 10 minutes after the top of the hour so that
                    any discovery run at :00 has time to complete first.

  Monitor loop    — every hour at :30  (e.g. 00:30, 01:30, 02:30 …)
                    Cancels unfilled pending orders, detects resolved markets,
                    records realized P&L, and updates unrealized P&L for all
                    open positions.
"""

import logging
import os
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from config import LOG_LEVEL, PAPER_TRADE, DB_PATH, ACTIVE_STRATEGY
from db import init_db
from edge import run_edge_scan
from risk import run_all_checks, run_pre_checks, run_portfolio_checks
from strategies import get_active_strategy
from execution import get_clob_client, execute_signal
from sizing import get_bankroll
from monitor import run_monitor_loop
from loops import (
    forecast_pull_run, live_observation_run, retention_run,
    vc_future_diagnostic_run,
)
from position_eval import evaluate_open_positions

# Custom SUMMARY log level — between INFO (20) and WARNING (30).
# When LOG_LEVEL=SUMMARY, the console shows only condensed phase summaries
# while the log file still captures full INFO detail.
SUMMARY = 25
logging.addLevelName(SUMMARY, "SUMMARY")

_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_BOT_DIR, "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)

_console_level = SUMMARY if LOG_LEVEL.upper() == "SUMMARY" else getattr(logging, LOG_LEVEL, logging.INFO)

# Force UTF-8 on both handlers so non-ASCII glyphs in log messages
# (e.g. "→" in execute_signal CAPPED diagnostics) don't crash the Windows
# console, whose default code page is cp1252.  reconfigure() is best-effort:
# it exists on TextIOWrapper streams (the normal sys.stdout) but not on
# every stream type, so we guard it.
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_console = logging.StreamHandler()
_console.setLevel(_console_level)

_filelog = logging.FileHandler(
    os.path.join(_LOGS_DIR, "bot.log"),
    encoding="utf-8",
)
_filelog.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s")
_console.setFormatter(_fmt)
_filelog.setFormatter(_fmt)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_console, _filelog],
)
logger = logging.getLogger("main")

logging.getLogger("httpx").setLevel(logging.WARNING)


def _phase_end():
    """Log a blank line to visually separate phases in console output."""
    logger.log(SUMMARY, "")

# Cache of the most recently discovered events, shared between discovery and
# trading runs to avoid a redundant Gamma API call within the same hour.
_cached_events: list[dict] | None = None


# ---------------------------------------------------------------------------
# Liquidity top-up helper
# ---------------------------------------------------------------------------

def _run_topups(all_events: list[dict], client) -> int:
    """Top up underfilled positions if more liquidity is now available.

    For each open position where size_usdc < target_size_usdc, check
    the current liquidity for that contract.  If liquidity supports
    adding more.  Sizing/cap is delegated to execute_topup which uses
    of current liquidity and the remaining shortfall), and execute.

    Only tops up if the model still favors the position (the bin is still
    in the model's top bins for the event).  Does NOT top up positions
    where the thesis has weakened.
    """
    from config import PAPER_TRADE
    from db import get_underfilled_positions

    underfilled = get_underfilled_positions()
    if not underfilled:
        return 0

    # Build a lookup of current liquidity by contract_id.
    # Primary source: all_events outcomes from this cycle.
    # Fallback: temp_outcomes DB (has liquidity from discovery scan).
    liquidity_map: dict[str, float] = {}
    model_prob_map: dict[str, float] = {}
    for ev in all_events:
        for o in ev.get("outcomes", []):
            cid = o.get("contract_id")
            if cid:
                liq = float(o.get("liquidity_usd") or 0)
                if liq > 0:
                    liquidity_map[cid] = liq
                mp = float(o.get("model_prob") or o.get("yes_price") or 0)
                if mp > 0:
                    model_prob_map[cid] = mp

    # Fill gaps from DB for contracts not in the event analysis
    if underfilled:
        try:
            import sqlite3 as _sq_liq
            _liq_conn = _sq_liq.connect(DB_PATH)
            _missing_cids = [p.get("contract_id") for p in underfilled
                             if p.get("contract_id") not in liquidity_map]
            if _missing_cids:
                placeholders = ",".join("?" * len(_missing_cids))
                for _r in _liq_conn.execute(
                    f"SELECT contract_id, liquidity_usd, yes_price FROM temp_outcomes "
                    f"WHERE contract_id IN ({placeholders}) AND liquidity_usd > 0 "
                    f"ORDER BY scan_timestamp DESC",
                    _missing_cids,
                ).fetchall():
                    if _r[0] not in liquidity_map:
                        liquidity_map[_r[0]] = float(_r[1])
                    if _r[0] not in model_prob_map and _r[2]:
                        model_prob_map[_r[0]] = float(_r[2])
            _liq_conn.close()
        except Exception:
            pass

    from execution import execute_topup

    from db import get_committed_usdc

    topped_up = 0
    for pos in underfilled:
        pid = pos["id"]
        cid = pos.get("contract_id", "")
        target = float(pos.get("target_size_usdc") or 0)
        # Phase B (2026-04-30): use the position_orders ledger to compute
        # how much capital is currently COMMITTED (filled on chain + still
        # resting on book).  Replaces the old `target - size_usdc` calc
        # which counted only filled, leading to top-ups stacking on top
        # of resting partial-fill orders → double-committed exposure
        # (the user's screenshot bug).
        committed = get_committed_usdc(pid)
        remaining = target - committed
        current_size = float(pos.get("size_usdc") or 0)  # for log line only
        entry_price = float(pos.get("entry_price") or 0)
        city = pos.get("city", "")
        date_str = pos.get("date", "")

        if remaining <= 1.0 or entry_price <= 0:
            if remaining <= 1.0:
                logger.debug(
                    f"[TOPUP] Skip pid={pid} {cid[:12]} — already committed "
                    f"${committed:.2f} of ${target:.2f} target (gap ${remaining:.2f})"
                )
            continue

        # Skip positions that already have an in-flight top-up — one at a
        # time per parent.  The monitor's reconciliation will fill or cancel,
        # then a future scan can issue another if still underfilled.
        if pos.get("pending_topup_order_id"):
            logger.debug(
                f"[TOPUP] Skipping pid={pid} {cid[:12]} — top-up already pending "
                f"(order={pos['pending_topup_order_id'][:12]})"
            )
            continue

        # PHASE A HOTFIX (2026-04-30): skip if the entry order is STILL
        # RESTING ON THE BOOK with unfilled size.  Without this, the
        # top-up logic computes `remaining = target - filled_only` and
        # double-commits capital (the user-reported bug: a $10 entry that
        # only filled $0.55 would get a $9.45 top-up placed on top of
        # $9.45 still resting from the original — total exposure $19.45).
        # Wait until the cancel pass clears the resting portion at the
        # 10-min age cutoff, then top-up evaluates the gap correctly
        # next cycle.  Phase B replaces this with the position_orders
        # ledger which tracks committed (filled + resting) directly.
        entry_order_id = pos.get("order_id")
        if entry_order_id and client is not None:
            try:
                from execution import get_order_status
                _stat = get_order_status(entry_order_id, client)
                if _stat:
                    s = (_stat.get("status") or "").upper()
                    sz_match = float(_stat.get("size_matched") or 0)
                    sz_orig  = float(_stat.get("original_size") or 0)
                    # Resting = order is LIVE/MATCHED with unfilled size remaining
                    if s in ("LIVE", "MATCHED", "DELAYED") and sz_match < sz_orig - 1e-9:
                        logger.info(
                            f"[TOPUP] Skip pid={pid} {cid[:12]} — entry order "
                            f"still resting on book ({sz_match:.2f}/{sz_orig:.2f} "
                            f"shares filled).  Waiting for cancel-pass to free "
                            f"the resting portion before topping up."
                        )
                        continue
            except Exception as _e:
                # If the CLOB query fails, fall through (don't block top-ups
                # on transient errors — the whole point of Phase B is to
                # remove the dependency on a live CLOB query for this).
                logger.debug(f"[TOPUP] Resting-order check failed (non-fatal): {_e}")

        # Check thesis still intact: model_prob should still be meaningful
        current_prob = model_prob_map.get(cid)
        if current_prob is not None and current_prob < 0.05:
            logger.debug(
                f"[TOPUP] Skipping {cid[:12]} — model_prob dropped to "
                f"{current_prob:.3f}, thesis weakened"
            )
            continue

        # Compute the intended top-up size as the FULL remaining gap.
        # execute_topup applies the ask-depth cap with fresh orderbook data
        # (replacing the previous stale-Gamma `liquidity_usd * 0.40` rule);
        # if the book is too thin, execute_topup returns status='skip' and
        # we'll re-evaluate next cycle.  Gamma's `liquidity_usd` is kept
        # only for the [TOPUP PLACED] log line as a soft sanity check.
        liquidity = liquidity_map.get(cid, 0)
        if remaining < 1.0:
            continue
        add_amount = remaining

        # execute_topup handles paper vs live internally.
        # Paper: merges the add into the parent immediately (current behavior).
        # Live:  posts CLOB buy, stamps pending_topup_* fields, monitor
        #        reconciles the fill and merges via update_position_topup.
        result = execute_topup(pos, add_amount, client=client)

        if result.get("status") in ("paper", "placed"):
            topped_up += 1
            actual_add = float(result.get("add_usdc", add_amount))
            logger.log(SUMMARY,
                f"[TOPUP {result['status'].upper()}] pos={pid} {city} {date_str} "
                f"+${actual_add:.2f} (${current_size:.2f} -> "
                f"${current_size + actual_add:.2f} / ${target:.2f} target) "
                f"liquidity=${liquidity:.0f}"
                + (f" | order={result.get('order_id', '')[:12]}"
                   if result.get('order_id') else "")
            )
        elif result.get("status") == "skip":
            # already-pending was handled above; this branch covers any
            # other internal skips from execute_topup.  Quiet log.
            logger.debug(f"[TOPUP] pid={pid} skip: {result.get('reason')}")
        else:
            logger.warning(
                f"[TOPUP FAILED] pid={pid} {cid[:12]}: {result}"
            )

    return topped_up


# ---------------------------------------------------------------------------
# Exit execution helper (Phase 3)
# ---------------------------------------------------------------------------

def _execute_exit_actions(actions, client) -> tuple[int, list[dict]]:
    """Execute queued exit actions from position_eval.

    Paper mode: logs exit + closes position in DB (status='closed').
    Live mode: routes through execution.execute_exit() which places a CLOB
    sell at the appropriate ladder rung (retry_count=0 → 0.99 × intended).
    The position transitions to status='exiting' and the monitor loop
    advances the ladder + confirms fills on subsequent cycles.

    Returns (count_executed, list of exit detail dicts for summary display).
    """
    from db import get_open_positions
    from execution import execute_exit

    executed = 0
    details: list[dict] = []

    # Build a position_id → row lookup so we can pass full context to
    # execute_exit (it needs entry_price, shares, token IDs, etc.)
    open_by_id = {p["id"]: p for p in get_open_positions()
                  if p.get("status") in ("open", "exiting")}

    for ea in actions:
        if ea.action == "HOLD":
            continue

        position = open_by_id.get(ea.position_id)
        if position is None:
            logger.warning(
                f"[EXIT] pos={ea.position_id} not in open positions — skipping"
            )
            continue

        # Skip if already in exit ladder (avoid double-firing within a scan)
        if position.get("status") == "exiting":
            logger.debug(
                f"[EXIT] pos={ea.position_id} already exiting "
                f"(retry_count={position.get('exit_retry_count', 0)}); skipping"
            )
            continue

        exit_price = float(ea.exit_price or 0.0)

        if ea.action == "REDUCE_50":
            # TODO: implement true partial close (sell shares/2).  For now we
            # treat REDUCE as full close to keep behavior unchanged.
            logger.info(
                f"[EXIT] REDUCE_50 treated as full SELL for pos={ea.position_id} "
                f"(partial closes not yet implemented)"
            )

        result = execute_exit(
            position             = position,
            intended_exit_price  = exit_price,
            exit_reason          = f"{ea.classification}:{ea.reason}",
            client               = client,
            retry_count          = 0,
        )

        # In paper mode, execute_exit closes the position immediately and
        # returns realized pnl.  In live mode, it places a CLOB sell and
        # the position transitions to status='exiting' — pnl is realized
        # later when the fill is reconciled.
        executed += 1
        if result.get("status") == "paper_closed":
            details.append({
                "city": ea.city, "date": ea.date, "side": ea.side,
                "classification": ea.classification,
                "pnl": result.get("pnl", 0.0),
                "pos_id": ea.position_id,
            })
            logger.info(
                f"[EXIT PAPER] pos={ea.position_id} {ea.city} {ea.date} "
                f"{ea.side} | {ea.classification} | exit@{exit_price:.4f} "
                f"pnl=${result.get('pnl', 0):+.4f} | {ea.reason}"
            )
        elif result.get("status") == "exit_pending":
            details.append({
                "city": ea.city, "date": ea.date, "side": ea.side,
                "classification": ea.classification,
                "pnl": None,                     # not yet realized
                "pos_id": ea.position_id,
            })
            logger.info(
                f"[EXIT LIVE] pos={ea.position_id} {ea.city} {ea.date} "
                f"{ea.side} | {ea.classification} | "
                f"order={result.get('order_id', '')[:12]} "
                f"limit={result.get('limit_price', 0):.4f} "
                f"intended={exit_price:.4f} | {ea.reason}"
            )
        elif result.get("status") in (
            "skip", "closed_via_balance_recovery", "shares_resynced",
        ):
            # Handled non-failure paths -- log at INFO with a clear label
            # so the operator can distinguish "gate did its job" from
            # "real failure".
            #   skip                          - let-it-decay or dust gate fired
            #   closed_via_balance_recovery   - chain held 0; self-heal closed
            #   shares_resynced               - chain shares re-synced; retry next cycle
            logger.info(
                f"[EXIT SKIPPED] pos={ea.position_id} {ea.city} {ea.date} "
                f"{ea.side} | {ea.classification} | "
                f"status={result.get('status')} "
                f"reason={result.get('reason') or '-'}"
            )
            executed -= 1   # not a real exit, but not an error either
        else:
            # error / failed / unmatched — leave position open and surface
            logger.warning(
                f"[EXIT FAILED] pos={ea.position_id} status={result.get('status')} "
                f"reason={result.get('reason') or result.get('response', '')}"
            )
            executed -= 1   # don't count failures

    if executed:
        logger.info(f"[EXIT] {executed} position(s) exit-triggered this cycle")
    return executed, details


# ---------------------------------------------------------------------------
# Discovery loop — every 4 hours at :00
# ---------------------------------------------------------------------------

def discovery_run() -> list[dict]:
    """
    Discover all active highest-temperature events from Polymarket.
    Results are cached for reuse by the next trading run.
    """
    global _cached_events
    logger.log(SUMMARY, "=== DISCOVERY RUN ===")
    from polymarket import search_temp_high_events
    from config import MIN_LIQUIDITY_USD
    events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)
    _cached_events = events
    logger.log(SUMMARY, f"Discovery: found {len(events)} events")
    _phase_end()
    return events


# ---------------------------------------------------------------------------
# Trading loop — every hour at :10
# ---------------------------------------------------------------------------

def trading_run():
    """
    Analyze all discovered events, generate signals, and execute trades that
    pass all risk checks.  Uses the cached event list from the most recent
    discovery run; fetches fresh data if the cache is empty.
    """
    logger.log(SUMMARY, "=== TRADING RUN START ===")

    # CLOB client is created once at the start of the cycle and reused for:
    #   1. Wallet balance lookup (sizing.get_bankroll → wallet.get_effective_bankroll)
    #   2. Order placement in execute_signal()
    #   3. Top-up orders in _run_topups()
    #   4. Exit orders in _execute_exit_actions()
    # In paper mode get_clob_client() returns None, which all callers handle.
    client = get_clob_client()
    bankroll = get_bankroll(client=client)
    logger.log(SUMMARY, f"Bankroll: ${bankroll:,.2f} | Strategy: {get_active_strategy().name} | Paper: {PAPER_TRADE}")

    # Use cached events if available, otherwise fetch fresh
    events = _cached_events
    if not events:
        logger.info("No cached events — running inline discovery")
        events = discovery_run()

    strategy = get_active_strategy()

    from datetime import timezone as _tz
    _scan_ts = datetime.now(_tz.utc).isoformat()

    all_events, signals = strategy.generate_signals(events, bankroll, _scan_ts)
    logger.info(f"Analyzed {len(all_events)} events -> {len(signals)} raw signals")

    eligible = []
    skipped  = 0
    skip_reasons: dict[str, int] = {}
    for signal in signals:
        passed, failures = run_pre_checks(signal, bankroll)
        if not passed:
            skipped += 1
            for f in failures:
                tag = f.split("=")[0].split(":")[0].split("(")[0].strip()
                skip_reasons[tag] = skip_reasons.get(tag, 0) + 1
            logger.info(
                f"Signal skipped [{signal.get('city')} {signal.get('date')} "
                f"{signal.get('question', '')[:30]}]: {'; '.join(failures)}"
            )
        else:
            eligible.append(signal)

    logger.info(f"Risk pre-filter: {len(eligible)} eligible, {skipped} skipped")

    # Pass the CLOB client so MPV strategy can do orderbook-aware ranking
    # (spread + sweepable depth on the top RANK_TOP_N_FOR_ORDERBOOK candidates).
    # Strategies that don't use the client ignore the kwarg.
    eligible = strategy.rank_signals(eligible, bankroll, client=client)
    # Drop signals filtered out by the spread cap (priority_score=-1000) so
    # they don't show up in the unfunded-reasons summary as "open positions
    # exceeded" — they were spread-rejected, not capacity-rejected.
    pre_drop = len(eligible)
    eligible = [s for s in eligible if s.get("priority_score", 0) > -100]
    spread_dropped = pre_drop - len(eligible)
    if spread_dropped > 0:
        logger.log(SUMMARY,
            f"  > Spread filter dropped {spread_dropped} signal(s) "
            f"(spread > MAX_SPREAD_CENTS_FOR_ENTRY)"
        )

    if eligible:
        logger.info("--- SIGNAL PRIORITY RANKING ---")
        for rank, s in enumerate(eligible, 1):
            pc = s.get("priority_components", {})
            logger.info(
                f"[RANK] #{rank} {s.get('city')} {s.get('date')} "
                f"[{s.get('question', '')[:25]}] {s.get('recommended_side')} "
                f"| score={s.get('priority_score', 0):.4f} "
                f"ev/$={pc.get('ev_per_dollar', 0):.3f} "
                f"conf={pc.get('confidence', 0):.2f} "
                f"time={pc.get('time_efficiency', 0):.4f} "
                f"hrs={pc.get('hours_to_resolve', 0):.0f} "
                f"confirm={pc.get('confirmation', 0):.1f}"
            )
        logger.info("--- END RANKING ---")

    logger.log(SUMMARY,
        f"Signals: {len(signals)} generated | {len(eligible)} eligible | "
        f"{skipped} filtered"
    )
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.log(SUMMARY, f"  > {reason}: {count}")
        # If position-cap skips dominated this cycle, surface the actionable
        # mitigation right next to the count so the operator doesn't have
        # to reason about what to do.  Per-position cap skips are
        # INTENTIONAL (config: don't enter at less than full target), but
        # they look identical to a bug at first glance — explicit hint
        # avoids that confusion.
        cap_skips = sum(c for r, c in skip_reasons.items()
                        if "per-position cap" in r)
        if cap_skips > 0 and cap_skips >= 0.5 * skipped:
            from config import MAX_POSITION_PCT as _mpp
            logger.log(SUMMARY,
                f"  > NOTE: {cap_skips} skip(s) are per-position-cap rejections "
                f"(MAX_POSITION_PCT={_mpp:.2%}). To trade at smaller sizes, "
                f"lower the cap in .env; OR wait for bankroll to recover."
            )
    logger.log(SUMMARY, "")

    # `client` was already initialized at the top of trading_run() and used
    # for the bankroll lookup; reuse it here for execute_signal and below.
    executed = 0
    unfunded = 0
    unfunded_reasons: dict[str, int] = {}
    executed_trades: list[dict] = []
    total_deployed = 0.0

    for signal in eligible:
        passed, failures = run_portfolio_checks(signal, bankroll)

        if not passed:
            unfunded += 1
            for f in failures:
                tag = f.split("=")[0].split(":")[0].split("(")[0].strip()
                unfunded_reasons[tag] = unfunded_reasons.get(tag, 0) + 1
            logger.info(
                f"Signal unfunded [{signal.get('city')} {signal.get('date')} "
                f"{signal.get('question', '')[:30]}]: {'; '.join(failures)}"
            )
            continue

        # Skip if we already hold this contract, or if it was stopped/trailed out
        _sig_cid = signal.get("contract_id")
        try:
            import sqlite3 as _sq_dup
            _dup_conn = _sq_dup.connect(DB_PATH)
            # Block if currently open
            _held = _dup_conn.execute(
                "SELECT COUNT(*) FROM positions WHERE contract_id = ? AND status = 'open'",
                (_sig_cid,)
            ).fetchone()[0]
            # Block if previously exited by a stop/trail (don't re-enter losing trades)
            _stopped = _dup_conn.execute(
                "SELECT COUNT(*) FROM positions WHERE contract_id = ? AND status = 'closed' "
                "AND exit_reason LIKE '%STOP%' OR (contract_id = ? AND status = 'closed' "
                "AND exit_reason LIKE '%TRAIL%') OR (contract_id = ? AND status = 'closed' "
                "AND exit_reason LIKE '%DYING%')",
                (_sig_cid, _sig_cid, _sig_cid)
            ).fetchone()[0]
            _dup_conn.close()
        except Exception:
            _held = 0
            _stopped = 0

        if _held > 0:
            logger.debug(f"Skipping {_sig_cid[:12]} — already held")
            continue
        if _stopped > 0:
            logger.debug(f"Skipping {_sig_cid[:12]} — previously stopped out")
            continue

        # Liquidity sizing was previously capped here using Gamma's stale
        # `liquidity_usd` (bid + ask combined).  As of 2026-04-30 the cap
        # has moved INSIDE execute_signal where it uses the FRESH orderbook
        # snapshot's ask-side depth at acceptable prices — see
        # MAX_TAKE_PCT_OF_ASK_DEPTH in config.  We just record the original
        # intent here so top-ups can fill the gap on later cycles.
        signal["target_size_usdc"] = signal.get("kelly_size", 0)

        result = execute_signal(signal, client=client)

        if result["status"] in ("placed", "paper"):
            executed += 1
            size = signal["kelly_size"]
            total_deployed += size
            executed_trades.append({
                "side": signal["recommended_side"],
                "city": signal.get("city", ""),
                "date": signal.get("date", ""),
                "price": result.get("entry_price", 0),
                "size": size,
                "range_low": signal.get("range_low"),
                "range_high": signal.get("range_high"),
                "unit": signal.get("unit", "celsius"),
                "pos_id": result.get("position_id"),
            })
            logger.info(
                f"Executed: {signal['recommended_side']} ${size:.2f} "
                f"on {signal['contract_id'][:12]} "
                f"[{signal.get('city')} {signal.get('question', '')[:20]}] "
                f"| status={result['status']} pos_id={result.get('position_id')} "
                f"priority=#{executed}"
            )
        elif result["status"] == "skip":
            # Expected skip — execute_signal hit a guard like the buy-retry
            # cap.  Already logged at WARNING by execution.py; nothing more
            # to do here.  Don't count as executed or as failed.
            pass
        else:
            logger.error(
                f"Execution failed for {signal.get('contract_id', '')[:12]}: {result}"
            )

    logger.info(
        f"Trading run complete: {executed} executed, {skipped} pre-filtered, "
        f"{unfunded} unfunded (capital exhausted), "
        f"{len(signals)} raw signals -> {len(eligible)} eligible"
    )

    # --- SUMMARY: Orders placed (NOT filled — fills come later via WS) ---
    # The bot has SUBMITTED these orders to Polymarket.  They may or may
    # not have filled yet:
    #   * Engine-matched orders may already be on chain (status='matched')
    #   * Resting orders sit on the book (status='live') until taken
    #   * The user-channel WS catches the actual fill confirmation and
    #     updates the position to fill_status='filled' with a separate
    #     [FILL] activity log entry — that's the truth-of-fill signal.
    if executed_trades:
        logger.log(SUMMARY, "--- Orders Placed ---")
        logger.log(SUMMARY,
            "  (These were submitted to Polymarket — fill confirmation "
            "appears later in the activity log as [FILL] entries)")
        for t in executed_trades:
            rl = t.get("range_low")
            rh = t.get("range_high")
            unit = t.get("unit", "celsius")
            suffix = "F" if unit == "fahrenheit" else "C"
            if rl is not None and rh is not None:
                bin_str = f"{int(rl)}-{int(rh)}{suffix}"
            elif rl is not None:
                bin_str = f">={int(rl)}{suffix}"
            elif rh is not None:
                bin_str = f"<={int(rh)}{suffix}"
            else:
                bin_str = "?"
            logger.log(SUMMARY,
                f"  [ ORDER PLACED ]  |  {t['city']:<8}  |  {t['date']}  {t['side']:<3}  |  "
                f"{bin_str:<8}  |  Limit: ${t['price']:.2f}  | Size: ${t['size']:,.0f}"
            )
    else:
        logger.log(SUMMARY, "--- No orders placed ---")

    # --- Liquidity-aware top-ups ---
    from config import LIQUIDITY_AWARE_SIZING
    topped_up = 0
    if LIQUIDITY_AWARE_SIZING:
        try:
            topped_up = _run_topups(all_events, client)
            if topped_up:
                logger.log(SUMMARY, f"[TOPUP] {topped_up} position(s) topped up")
        except Exception as e:
            logger.debug(f"Top-up pass failed (non-fatal): {e}")

    # --- Exit engine ---
    exit_count = 0
    exit_details: list[dict] = []
    try:
        exit_actions = strategy.evaluate_positions()
        if exit_actions:
            exit_count, exit_details = _execute_exit_actions(exit_actions, client)
    except Exception as e:
        logger.exception(f"Exit evaluation failed (non-fatal): {e}")

    if exit_details:
        logger.log(SUMMARY, "--- Exits ---")
        for ed in exit_details:
            # Live exits have pnl=None until the sell fills (see
            # _execute_exit_actions); paper exits realise pnl immediately.
            _pnl = ed.get('pnl')
            pnl_str = f"${_pnl:+.2f}" if _pnl is not None else "pending"
            logger.log(SUMMARY,
                f"  SELL {ed['side']:<3}  {ed['city']:<18} {ed['date']}  "
                f"{ed['classification']:<18} pnl={pnl_str}  pos={ed['pos_id']}"
            )

    if unfunded_reasons:
        logger.log(SUMMARY, f"--- Unfunded ({unfunded}) ---")
        for reason, count in sorted(unfunded_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.log(SUMMARY, f"  > {reason}: {count}")

    logger.log(SUMMARY, "")
    parts = []
    # `executed` counts orders SUBMITTED, not on-chain fills.  The fill
    # truth-signal is in the activity log via [FILL] entries that come
    # from the user-channel WS handler (see fill_handler._apply_confirmed_fill).
    parts.append(f"{executed} placed")
    if exit_count:
        parts.append(f"{exit_count} exit orders placed")
    if topped_up:
        parts.append(f"{topped_up} topped up")
    if unfunded:
        parts.append(f"{unfunded} unfunded")
    parts.append(f"${total_deployed:,.0f} submitted this cycle")

    logger.log(SUMMARY, f"--- Summary: {' | '.join(parts)} ---")
    logger.log(SUMMARY, "")
    logger.log(SUMMARY, "=== TRADING RUN END ===")
    _phase_end()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    logger.log(SUMMARY, "Polymarket weather bot starting up")
    logger.log(SUMMARY, f"Paper: {PAPER_TRADE} | DB: {DB_PATH}")

    init_db()

    # One-shot backfill of position_orders ledger (Phase B, 2026-04-30).
    # Idempotent — only inserts rows for orders not yet in the ledger.
    # Existing positions' order_id / pending_topup_order_id / exit_order_id
    # get a synthesized ledger row so committed_usdc queries work from
    # cycle 1.  After every position has flowed through the new code path,
    # this call is a cheap no-op.
    try:
        from db import backfill_position_orders
        _bf = backfill_position_orders()
        if any(_bf.values()):
            logger.log(SUMMARY,
                f"position_orders backfilled: "
                f"{_bf.get('entries', 0)} entries, "
                f"{_bf.get('topups', 0)} topups, "
                f"{_bf.get('exits', 0)} exits"
            )
    except Exception as _e:
        logger.warning(f"position_orders backfill failed (non-fatal): {_e}")

    # ----- Startup health check: validate ensemble API connectivity -----
    # Only needed for weather-based strategies.  Strategies that trade
    # purely on market price (MPV, TKH) skip the weather pipeline entirely
    # for efficiency — no forecast pulls, observations, bias updates, or
    # health checks against weather APIs.
    #
    # Allowlist semantics: a strategy is treated as needing weather data
    # ONLY if it's explicitly listed below.  This is safer than the prior
    # exclusion-list approach (`!= "market_price_value"`) because adding
    # a new market-price strategy doesn't accidentally flip the bot into
    # weather-pulling mode.
    _WEATHER_DATA_STRATEGIES = {"top_bin_value"}
    _needs_weather = ACTIVE_STRATEGY in _WEATHER_DATA_STRATEGIES

    if _needs_weather:
        from weather import _get_ecmwf_ensemble_distribution, _get_gfs_ensemble_distribution
        from db import get_latest_forecast_distribution
        from datetime import timedelta
        _test_lat, _test_lon = 41.85, -87.65   # Chicago

        _health_ok = False
        for _offset in [1, 0]:
            _test_date = (datetime.now().date() + timedelta(days=_offset)).isoformat()
            logger.log(SUMMARY, f"Health check: testing ECMWF + GFS for Chicago {_test_date}...")
            _ecmwf_test = _get_ecmwf_ensemble_distribution(_test_lat, _test_lon, _test_date)
            _gfs_test   = _get_gfs_ensemble_distribution(_test_lat, _test_lon, _test_date)
            if _ecmwf_test and _gfs_test:
                logger.info(f"  ECMWF OK: n={_ecmwf_test['n']} members, mu={_ecmwf_test['mu_c']:.2f}")
                logger.info(f"  GFS OK: n={_gfs_test['n']} members, mu={_gfs_test['mu_c']:.2f}")
                _health_ok = True
                break
            else:
                _e = "OK" if _ecmwf_test else "MISSING"
                _g = "OK" if _gfs_test else "MISSING"
                logger.warning(f"  {_test_date}: ECMWF={_e} GFS={_g} -- trying fallback")

        if not _health_ok:
            _db_ecmwf = get_latest_forecast_distribution(_test_lat, _test_lon, _test_date, "ecmwf", max_age_hours=6.0)
            _db_gfs   = get_latest_forecast_distribution(_test_lat, _test_lon, _test_date, "gfs", max_age_hours=6.0)
            if _db_ecmwf and _db_gfs:
                logger.warning(
                    "API returned no data but DB has recent forecasts -- "
                    "proceeding with cached data. The API may be temporarily down."
                )
                _health_ok = True
            else:
                logger.error(
                    "STARTUP HEALTH CHECK FAILED: Neither API nor DB returned "
                    "ensemble data. Check if the ECMWF/GFS model names have changed "
                    "on the Open-Meteo Ensemble API, or if the API is down. "
                    "Bot will NOT trade until this is resolved."
            )

        if not _health_ok:
            logger.error("Exiting due to failed ensemble health check.")
            sys.exit(1)
        logger.log(SUMMARY, "Ensemble health check passed.")
    else:
        logger.log(SUMMARY,
            f"Strategy: {ACTIVE_STRATEGY} — skipping weather health check "
            f"(strategy doesn't use forecast data)"
        )

    # Start WebSocket price stream immediately so stop-losses are active
    # while the slower startup loops (bias, forecast, trading) run.
    try:
        from price_ws import start_price_stream, load_open_position_tokens, wire_stop_loss_callback
        from realtime_exits import on_price_update
        wire_stop_loss_callback(on_price_update)
        load_open_position_tokens()
        start_price_stream()
        logger.log(SUMMARY, "WebSocket price stream started — stop-losses active")
    except Exception as e:
        logger.warning(f"WebSocket price stream failed to start (non-fatal): {e}")

    # Start authenticated user-channel WS for real-time fill detection.
    # Lives alongside the public price stream above; runs only in live mode.
    # On every reconnect, triggers a REST reconciliation pass to catch up
    # on anything missed during downtime.
    if not PAPER_TRADE:
        try:
            from execution import get_clob_client
            from user_ws import start_user_stream, wire_backfill_callback
            from monitor import _reconcile_pending_fills
            from config import WALLET_ADDRESS

            _user_ws_client = get_clob_client()

            # Polymarket maker-side keep-alive: heartbeat every 5s.  Without
            # this, every restart and every >15s WS hiccup auto-cancels all
            # of our open orders (Polymarket's market-maker safety mechanism).
            # See bot/heartbeat.py for the full rationale.
            try:
                from heartbeat import start_heartbeat
                start_heartbeat(_user_ws_client)
                logger.info("Heartbeat daemon started (5s interval)")
            except Exception as _hb_err:
                logger.warning(
                    f"Heartbeat daemon failed to start (non-fatal): {_hb_err}"
                )

            def _user_ws_backfill() -> None:
                # Run REST sweep through the same fill_handler path the WS
                # uses, catching anything that filled during reconnect downtime.
                try:
                    _reconcile_pending_fills(_user_ws_client)
                except Exception as e:
                    logger.warning(f"[USER_WS] REST backfill failed: {e}")

            wire_backfill_callback(_user_ws_backfill)
            start_user_stream(_user_ws_client, wallet_address=WALLET_ADDRESS)
            logger.log(SUMMARY,
                "User-channel WS started — real-time fill detection active"
            )
        except Exception as e:
            logger.warning(
                f"User-channel WS failed to start (non-fatal — REST poller "
                f"will still reconcile fills every monitor cycle): {e}"
            )

    if _needs_weather:
        try:
            import sqlite3
            _bias_conn = sqlite3.connect(DB_PATH)
            _last_bias = _bias_conn.execute(
                "SELECT MAX(recorded_at) FROM forecast_errors"
            ).fetchone()[0]
            _bias_conn.close()
            _bias_ran_today = (
                _last_bias is not None
                and _last_bias[:10] == datetime.now().strftime("%Y-%m-%d")
            )
        except Exception:
            _bias_ran_today = False

        if _bias_ran_today:
            logger.info("Bias update already ran today — skipping startup refresh")
        else:
            try:
                from bias_correction.bias_updater import run_bias_update
                run_bias_update()
            except Exception as e:
                logger.warning(f"Startup bias update failed (non-fatal): {e}")
    else:
        logger.log(SUMMARY,
            f"Strategy: {ACTIVE_STRATEGY} — skipping weather data loops "
            f"(forecast pulls, live observations, bias updates)"
        )

    events = discovery_run()
    _cached_events_ref = events  # seed the module-level cache
    globals()["_cached_events"] = events

    if _needs_weather:
        import sqlite3 as _sqlite3
        _skip_conn = _sqlite3.connect(DB_PATH)

        try:
            _last_pull = _skip_conn.execute(
                "SELECT MAX(pulled_at) FROM forecast_runs"
            ).fetchone()[0]
            _pull_recent = False
            if _last_pull:
                from datetime import timezone as _tz2
                _pull_age = (datetime.now(_tz2.utc) - datetime.fromisoformat(
                    _last_pull.replace("Z", "+00:00")
                )).total_seconds() / 3600
                _pull_recent = _pull_age < 2.0
        except Exception:
            _pull_recent = False

        if _pull_recent:
            logger.info("Forecast pull ran within last 2 hours -- skipping startup refresh")
        else:
            try:
                logger.info("Startup: running forecast pull before first trade cycle...")
                forecast_pull_run(events=events)
            except Exception as e:
                logger.warning(f"Startup forecast pull failed (non-fatal): {e}")

        try:
            _last_obs = _skip_conn.execute(
                "SELECT MAX(pulled_at_utc) FROM live_observations"
            ).fetchone()[0]
            _obs_recent = False
            if _last_obs:
                from datetime import timezone as _tz3
                _obs_age = (datetime.now(_tz3.utc) - datetime.fromisoformat(
                    _last_obs.replace("Z", "+00:00")
                )).total_seconds() / 60
                _obs_recent = _obs_age < 20.0
        except Exception:
            _obs_recent = False

        _skip_conn.close()

        if _obs_recent:
            logger.info("Live observations ran within last 20 min -- skipping startup refresh")
        else:
            try:
                logger.info("Startup: running live observations before first trade cycle...")
                live_observation_run(events=events)
            except Exception as e:
                logger.warning(f"Startup live observation run failed (non-fatal): {e}")

    try:
        trading_run()
    except Exception as e:
        logger.exception(f"Trading run failed: {e}")

    try:
        run_monitor_loop()
    except Exception as e:
        logger.exception(f"Monitor run failed: {e}")

    scheduler = BlockingScheduler(timezone="UTC")

    # Discovery: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    scheduler.add_job(
        discovery_run,
        trigger=CronTrigger(hour="0,4,8,12,16,20", minute=0, timezone="UTC"),
        id="discovery_run",
        name="Market discovery",
        misfire_grace_time=300,
        coalesce=True,
    )

    if _needs_weather:
        # Weather strategy: hourly at :15 (after forecast pull at :05)
        scheduler.add_job(
            trading_run,
            trigger=CronTrigger(minute=15, timezone="UTC"),
            id="trading_run",
            name="Trading scan",
            misfire_grace_time=300,
            coalesce=True,
        )
    else:
        # Market price strategy: every 15 minutes for top-ups
        scheduler.add_job(
            trading_run,
            trigger=CronTrigger(minute="0,15,30,45", timezone="UTC"),
            id="trading_run",
            name="Trading scan (15-min)",
            misfire_grace_time=300,
            coalesce=True,
        )

    # Monitor: every hour at :30
    scheduler.add_job(
        run_monitor_loop,
        trigger=CronTrigger(minute=40, timezone="UTC"),
        id="monitor_run",
        name="Position monitor",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Fast exit-ladder advancement: every 5 minutes.
    # The hourly monitor also calls _advance_exit_ladders (as a safety
    # net), but the 5-min cadence is what actually drives the ladder
    # forward in normal operation.  This caps max time-to-fill on a
    # stop-out at ~20 minutes (4 rungs × 5 min) instead of 4 hours.
    # Concurrency between the two callers is handled by an internal
    # lock in _advance_exit_ladders — second concurrent invocation
    # short-circuits without racing.
    def _exit_ladder_fast_job() -> None:
        try:
            from monitor import run_exit_ladder_fast
            run_exit_ladder_fast()
        except Exception as e:
            logger.exception(f"Fast exit-ladder job failed (non-fatal): {e}")

    scheduler.add_job(
        _exit_ladder_fast_job,
        trigger=CronTrigger(minute="*/5", timezone="UTC"),
        id="exit_ladder_fast",
        name="Exit ladder fast advance (5-min)",
        misfire_grace_time=120,
        coalesce=True,
    )

    # Orphan topup pointer cleanup: every 5 minutes.
    # Polymarket sometimes cancels resting orders on its own (account risk
    # checks, WS auth disconnects, manual UI cancels).  When the WS misses
    # the resulting CANCELLATION event, the bot's pending_topup_order_id
    # column carries a dead pointer forever and _run_topups thinks a topup
    # is still in flight.  This job polls the CLOB directly for every
    # such pointer and clears the stale ones — see
    # monitor.detect_externally_cancelled_topups.
    def _orphan_topup_cleanup_job() -> None:
        try:
            from monitor import run_orphan_topup_cleanup_fast
            run_orphan_topup_cleanup_fast()
        except Exception as e:
            logger.exception(f"Orphan topup cleanup job failed (non-fatal): {e}")

    scheduler.add_job(
        _orphan_topup_cleanup_job,
        trigger=CronTrigger(minute="*/5", timezone="UTC"),
        id="orphan_topup_cleanup",
        name="Orphan topup pointer cleanup (5-min)",
        misfire_grace_time=120,
        coalesce=True,
    )

    # Stale entry / topup repricer cadence.  PHASE 1 (2026-06-13):
    # repriced from CronTrigger(minute="*/5") to IntervalTrigger so the
    # cadence is sub-minute-tunable via env.  Default 60s — chases asks
    # within one minute of drift instead of within 0-5 minutes.
    #
    # SAFETY: every cancel/replace cycle re-fetches the position's
    # committed_usdc from the DB and sizes the new order at
    # (target - committed).  See refresh_stale_ensure_fill_entries'
    # _capture_partial_fills_before_cancel step for the explicit
    # pre-cancel reconciliation that prevents the Houston $74.99
    # over-allocation bug.  Speeding the cron up does NOT increase
    # overrun risk because:
    #   - cancel_order is synchronous + confirmed
    #   - committed_usdc is re-read AFTER cancel succeeds
    #   - new order size = max(0, target - committed)
    #   - _refresh_stale_*_lock serializes invocations
    #
    # Don't set this below 10s — every cycle hits the CLOB orderbook
    # endpoint per pending order; high-frequency reruns can trip
    # Polymarket's per-API-key rate limit.  60s is the sweet spot for
    # responsiveness vs. API budget.  Set to 0 to disable repricing
    # entirely (falls back to once-per-restart static limits).
    import os as _os
    _REPRICE_INTERVAL_S = int(_os.getenv("PREDICTOR_REPRICE_INTERVAL_S", "60"))

    # Stale TOPUP re-pricing.  When a topup's limit becomes stale relative
    # to the current best_ask (drift > TOPUP_REPRICE_THRESHOLD_CENTS,
    # default 1.5¢), cancel + re-issue at the fresh price.  Fixes the
    # "topup sits at $0.30 forever while asks are now $0.40" failure mode
    # where the position would otherwise never reach target size despite
    # available liquidity.  Direction-aware: only fires on UPWARD ask
    # drift; downward drift is already handled by Polymarket's matching
    # at the better price.
    def _stale_topup_refresh_job() -> None:
        try:
            from monitor import run_stale_topup_refresh_fast
            run_stale_topup_refresh_fast()
        except Exception as e:
            logger.exception(f"Stale topup refresh job failed (non-fatal): {e}")

    if _REPRICE_INTERVAL_S > 0:
        scheduler.add_job(
            _stale_topup_refresh_job,
            trigger=IntervalTrigger(seconds=_REPRICE_INTERVAL_S),
            id="stale_topup_refresh",
            name=f"Stale topup re-pricing ({_REPRICE_INTERVAL_S}s)",
            misfire_grace_time=30,
            coalesce=True,    # if multiple runs queue up, collapse to one
            max_instances=1,  # belt-and-suspenders vs lock
        )

    # Stale ENTRY-order repricer.  Covers ensure-fill strategies (TKH)
    # AND the intraday predictor (added 2026-06-13).  Without this, an
    # entry placed at best_ask + walk_cents would rest below the new ask
    # whenever the market drifted up, sit unfilled, and eventually get
    # nuked by the cancel sweep.  This job chases the moving best_ask
    # until the order crosses (or the position reaches target).
    # _capture_partial_fills_before_cancel in the repricer is the
    # overrun guard — any chunks that filled between snapshot and cancel
    # are applied to the position row BEFORE the replacement is sized.
    def _stale_entry_refresh_job() -> None:
        try:
            from monitor import run_stale_entry_refresh_fast
            run_stale_entry_refresh_fast()
        except Exception as e:
            logger.exception(f"Stale entry refresh job failed (non-fatal): {e}")

    if _REPRICE_INTERVAL_S > 0:
        scheduler.add_job(
            _stale_entry_refresh_job,
            trigger=IntervalTrigger(seconds=_REPRICE_INTERVAL_S),
            id="stale_entry_refresh",
            name=f"Stale entry-order re-pricing ({_REPRICE_INTERVAL_S}s)",
            misfire_grace_time=30,
            coalesce=True,
            max_instances=1,
        )

    # Aggressive REST trade-fill polling -- DISABLED 2026-05-03.
    #
    # This job was doubling share counts because of a cold-start dedup
    # race: when WS is degraded, the hourly REST safety-net synthesizes
    # fake events with id="rest:{order_id}" and applies the aggregate
    # fill to the position row.  Real trade event IDs (the actual trade
    # hashes from Polymarket) never enter the dedup table because WS
    # never delivered them.  When this poller starts pulling real trades
    # via get_trades(), each real trade has a hash that's never been
    # seen, so apply_trade_event treats it as a new fill and ADDS the
    # chunks on top of what the synthetic event already applied.
    # Result: every position doubled (db_shares = 2 x chain_size).
    #
    # The fix needed before re-enabling: a bootstrap pass that walks
    # every active position's order_ids, fetches their trades, and marks
    # each trade event as processed (without applying) IF the position
    # row's shares already reflect that fill.  Only NEW trades after
    # bootstrap should be applied.
    #
    # Until that bootstrap is implemented, leave this job out of the
    # scheduler.  The hourly REST safety-net continues to work as before.
    #
    # def _trade_fill_poll_job() -> None:
    #     try:
    #         from monitor import run_trade_fill_poll_fast
    #         run_trade_fill_poll_fast()
    #     except Exception as e:
    #         logger.exception(f"Trade-fill poll job failed (non-fatal): {e}")
    #
    # scheduler.add_job(
    #     _trade_fill_poll_job,
    #     trigger=CronTrigger(minute="*/2", timezone="UTC"),
    #     id="trade_fill_poll",
    #     name="Aggressive trade-fill polling (2-min)",
    #     misfire_grace_time=60,
    #     coalesce=True,
    # )

    # Bias update: daily at 05:00 UTC.  Fetches the last 14 days of VC
    # observations + Open-Meteo Previous Runs ECMWF/GFS forecasts for every
    # city and rebuilds forecast_errors.  Replaces the previous
    # ERA5-reanalysis-based bias recorder.
    def _scheduled_bias_update() -> None:
        try:
            from bias_correction.bias_updater import run_bias_update
            run_bias_update()
            from weather import clear_city_error_cache
            clear_city_error_cache()
        except Exception as e:
            logger.exception(f"Scheduled bias update failed (non-fatal): {e}")

    if _needs_weather:
        scheduler.add_job(
            _scheduled_bias_update,
            trigger=CronTrigger(hour=5, minute=0, timezone="UTC"),
            id="bias_update",
            name="Bias data refresh",
            misfire_grace_time=3600,
            coalesce=True,
        )

    # --- Phase 2: time-versioned forecast + observation loops ---

    def _forecast_pull_job() -> None:
        try:
            forecast_pull_run(events=_cached_events)
        except Exception as e:
            logger.exception(f"Forecast pull run failed: {e}")

    def _live_observation_job() -> None:
        try:
            live_observation_run(events=_cached_events)
        except Exception as e:
            logger.exception(f"Live observation run failed: {e}")

    def _retention_job() -> None:
        try:
            retention_run(older_than_days=90)
        except Exception as e:
            logger.exception(f"Retention run failed: {e}")

    # Weather-specific loops — only scheduled when strategy needs weather data
    if _needs_weather:
        scheduler.add_job(
            _forecast_pull_job,
            trigger=CronTrigger(hour="*/2", minute=5, timezone="UTC"),
            id="forecast_pull",
            name="Forecast pull (ECMWF + GFS)",
            misfire_grace_time=600,
            coalesce=True,
        )

        scheduler.add_job(
            _live_observation_job,
            trigger=CronTrigger(minute="0,20,40", timezone="UTC"),
            id="live_observation",
            name="Visual Crossing live observation",
            misfire_grace_time=300,
            coalesce=True,
        )

        def _vc_future_diag_job() -> None:
            try:
                vc_future_diagnostic_run(events=_cached_events)
            except Exception as e:
                logger.exception(f"VC future-day diagnostic run failed: {e}")

        scheduler.add_job(
            _vc_future_diag_job,
            trigger=CronTrigger(hour="*/2", minute=7, timezone="UTC"),
            id="vc_future_diag",
            name="VC forecast diagnostics (future day)",
            misfire_grace_time=600,
            coalesce=True,
        )

    # Retention: always runs (cleans up DB regardless of strategy)
    scheduler.add_job(
        _retention_job,
        trigger=CronTrigger(hour=4, minute=30, timezone="UTC"),
        id="retention",
        name="Hourly/obs retention purge",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Price snapshots: every 5 minutes — captures live WebSocket prices
    # for ALL active bins into bin_price_history for backtesting.
    def _price_snapshot_job() -> None:
        try:
            from price_ws import write_price_snapshots
            write_price_snapshots()
        except Exception as e:
            logger.debug(f"Price snapshot failed (non-fatal): {e}")

    scheduler.add_job(
        _price_snapshot_job,
        trigger=CronTrigger(minute="*/2", timezone="UTC"),
        id="price_snapshot",
        name="Bin price snapshot (backtest data)",
        misfire_grace_time=120,
        coalesce=True,
    )

    # Intraday bin predictor — scheduled scan (default: paper mode, every 15 min).
    # Self-contained module; safe to register here.  Controlled by PREDICTOR_MODE
    # env var (defaults to 'paper').  Failures here MUST NOT crash the bot, so
    # wrap in try/except.
    try:
        from scheduled_predictor import register_predictor_jobs
        register_predictor_jobs(scheduler)
    except Exception as e:
        logger.warning(f"intraday_predictor scheduler registration failed (non-fatal): {e}")

    # Load all event tokens on startup so WebSocket captures all prices
    try:
        from price_ws import load_all_event_tokens
        load_all_event_tokens()
    except Exception:
        pass

    if _needs_weather:
        logger.log(SUMMARY,
            "Scheduler started | "
            "Discovery: 00/04/08/12/16/20:00 UTC | "
            "Forecast pull: */2h at :05 | "
            "Trading: :15 every hour | "
            "Live obs: :00/:20/:40 | "
            "Monitor: :40 every hour | "
            "Price snapshots: every 2 min | "
            "Retention: 04:30 UTC daily | "
            "Bias update: 05:00 UTC daily"
        )
    else:
        logger.log(SUMMARY,
            "Scheduler started | "
            "Discovery: 00/04/08/12/16/20:00 UTC | "
            "Trading: every 15 min | "
            "Monitor: :40 every hour | "
            "Price snapshots: every 2 min | "
            "Retention: 04:30 UTC daily"
        )

    try:
        from activity import log_activity
        log_activity(
            "SYSTEM",
            message=(
                f"bot started: strategy={ACTIVE_STRATEGY} "
                f"mode={'paper' if PAPER_TRADE else 'LIVE'}"
            ),
            strategy=ACTIVE_STRATEGY, paper_trade=PAPER_TRADE,
        )
    except Exception:
        pass

    try:
        scheduler.start()
    except KeyboardInterrupt:
        try:
            from activity import log_activity
            log_activity("SYSTEM", message="bot stopped by user (KeyboardInterrupt)")
        except Exception:
            pass
        # Clean shutdown of the heartbeat daemon — on KeyboardInterrupt
        # the process is already exiting, so this is mostly cosmetic
        # (the daemon thread would die with the process anyway), but it
        # gives us a clean log line + cancels the in-flight sleep.
        try:
            from heartbeat import stop_heartbeat
            stop_heartbeat(timeout=2.0)
        except Exception:
            pass
        logger.info("Bot stopped by user")
    except Exception as e:
        try:
            from activity import log_activity
            log_activity(
                "SYSTEM", level="ERROR",
                message=f"bot crashed: {e}",
                error=str(e),
            )
        except Exception:
            pass
        try:
            from heartbeat import stop_heartbeat
            stop_heartbeat(timeout=2.0)
        except Exception:
            pass
        logger.exception(f"Bot crashed: {e}")
        raise


if __name__ == "__main__":
    main()
