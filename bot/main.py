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
from datetime import datetime

from config import LOG_LEVEL, PAPER_TRADE, DB_PATH
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

# Resolve logs directory to bot/logs/ regardless of where Python is launched from
# (same pattern as DB_PATH — prevents duplicate logs/ folders at project root).
_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_BOT_DIR, "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_LOGS_DIR, "bot.log")),
    ],
)
logger = logging.getLogger("main")

# Silence httpx's raw HTTP request lines — descriptive messages are logged
# at each call site instead.
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    adding more, compute the top-up amount (capped at MAX_LIQUIDITY_TAKE_PCT
    of current liquidity and the remaining shortfall), and execute.

    Only tops up if the model still favors the position (the bin is still
    in the model's top bins for the event).  Does NOT top up positions
    where the thesis has weakened.
    """
    from config import MAX_LIQUIDITY_TAKE_PCT, PAPER_TRADE
    from db import get_underfilled_positions, update_position_topup, insert_position

    underfilled = get_underfilled_positions()
    if not underfilled:
        return 0

    # Build a lookup of current liquidity by contract_id from the latest
    # event analysis (which was just computed this cycle).
    liquidity_map: dict[str, float] = {}
    model_prob_map: dict[str, float] = {}
    for ev in all_events:
        for o in ev.get("outcomes", []):
            cid = o.get("contract_id")
            if cid:
                liquidity_map[cid] = float(o.get("liquidity_usd") or 0)
                model_prob_map[cid] = float(o.get("model_prob") or 0)

    topped_up = 0
    for pos in underfilled:
        pid = pos["id"]
        cid = pos.get("contract_id", "")
        current_size = float(pos.get("size_usdc") or 0)
        target = float(pos.get("target_size_usdc") or 0)
        remaining = target - current_size
        entry_price = float(pos.get("entry_price") or 0)
        city = pos.get("city", "")
        date_str = pos.get("date", "")

        if remaining <= 1.0 or entry_price <= 0:
            continue

        # Check thesis still intact: model_prob should still be meaningful
        current_prob = model_prob_map.get(cid)
        if current_prob is not None and current_prob < 0.05:
            logger.debug(
                f"[TOPUP] Skipping {cid[:12]} — model_prob dropped to "
                f"{current_prob:.3f}, thesis weakened"
            )
            continue

        # Check available liquidity
        liquidity = liquidity_map.get(cid, 0)
        max_take = liquidity * MAX_LIQUIDITY_TAKE_PCT
        if max_take < 1.0:
            continue

        add_amount = min(remaining, max_take)
        if add_amount < 1.0:
            continue

        add_shares = round(add_amount / entry_price, 4)

        # Weighted average price (keep entry_price for simplicity in paper
        # mode — in live mode we'd use the new fill price)
        new_avg_price = entry_price

        if PAPER_TRADE:
            update_position_topup(pid, add_amount, add_shares, new_avg_price)
            topped_up += 1
            logger.info(
                f"[TOPUP] pos={pid} {city} {date_str} "
                f"+${add_amount:.2f} (${current_size:.2f} -> "
                f"${current_size + add_amount:.2f} / "
                f"${target:.2f} target) "
                f"liquidity=${liquidity:.0f}"
            )
        else:
            # Live mode: would place a CLOB order for add_amount
            # For now, same as paper (TODO: implement live top-up orders)
            update_position_topup(pid, add_amount, add_shares, new_avg_price)
            topped_up += 1

    return topped_up


# ---------------------------------------------------------------------------
# Exit execution helper (Phase 3)
# ---------------------------------------------------------------------------

def _execute_exit_actions(actions, client) -> int:
    """Execute queued exit actions from position_eval.  Paper mode logs the
    exit; live mode places CLOB sell orders.  Returns count of exits executed."""
    from datetime import timezone
    from zoneinfo import ZoneInfo
    from db import (
        update_position_outcome,
        update_position_excursions,
        get_latest_snapshot_id_for_contract,
    )
    from execution import cancel_order

    executed = 0
    now = datetime.now(ZoneInfo("America/Chicago")).isoformat()

    for ea in actions:
        if ea.action == "HOLD":
            continue

        exit_price = ea.exit_price or 0.0
        # For paper mode we simulate the sell at the executable bid
        shares = 0.0
        entry_price = 0.0
        try:
            from db import _get_conn
            from config import DB_PATH
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT shares, entry_price, unrealized_pnl FROM positions WHERE id = ?",
                (ea.position_id,)
            ).fetchone()
            conn.close()
            if row:
                shares = float(row[0] or 0)
                entry_price = float(row[1] or 0)
        except Exception:
            pass

        if ea.action == "REDUCE_50":
            # For paper: close half the position; for live: sell half shares
            shares = shares * 0.5
            # In paper mode we can't actually split — log as a note and
            # keep the position open with reduced effective size.
            # For v1: treat REDUCE_50 as a full sell with a note.
            # TODO: implement partial position closes when live trading.
            logger.info(
                f"[EXIT] REDUCE_50 treated as SELL for pos={ea.position_id} "
                f"(partial closes not yet implemented)"
            )

        pnl = round((exit_price - entry_price) * shares, 4)
        exit_snap_id = get_latest_snapshot_id_for_contract(ea.contract_id)

        update_position_outcome(
            position_id    = ea.position_id,
            exit_price     = exit_price,
            exit_time      = now,
            pnl            = pnl,
            status         = "closed",
            exit_reason    = f"{ea.classification}:{ea.reason}",
            exit_snapshot_id = exit_snap_id,
        )
        executed += 1
        logger.info(
            f"[EXIT] pos={ea.position_id} {ea.city} {ea.date} {ea.side} "
            f"| {ea.classification} | exit@{exit_price:.4f} pnl=${pnl:+.4f} "
            f"| {ea.reason}"
        )

    if executed:
        logger.info(f"[EXIT] {executed} position(s) exited this cycle")
    return executed


# ---------------------------------------------------------------------------
# Discovery loop — every 4 hours at :00
# ---------------------------------------------------------------------------

def discovery_run() -> list[dict]:
    """
    Discover all active highest-temperature events from Polymarket.
    Results are cached for reuse by the next trading run.
    """
    global _cached_events
    logger.info("=== DISCOVERY RUN START ===")
    from polymarket import search_temp_high_events
    from config import MIN_LIQUIDITY_USD
    events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)
    _cached_events = events
    logger.info(f"Discovery: found {len(events)} highest-temperature events")
    logger.info("=== DISCOVERY RUN END ===")
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
    logger.info("=== TRADING RUN START ===")

    # Bias correction data is refreshed by the bias_updater daily job, not
    # here.  (Previously this ran an ERA5-reanalysis-based recorder; that
    # path has been dropped in favour of the VC + Open-Meteo Previous Runs
    # pipeline.  See bias_updater.run_bias_update.)

    bankroll = get_bankroll()
    logger.info(f"Bankroll: ${bankroll:.2f} | Paper trade: {PAPER_TRADE}")

    # Use cached events if available, otherwise fetch fresh
    events = _cached_events
    if not events:
        logger.info("No cached events — running inline discovery")
        events = discovery_run()

    # Use the active strategy to generate signals
    strategy = get_active_strategy()
    logger.info(f"Active strategy: {strategy.name}")

    from datetime import timezone as _tz
    _scan_ts = datetime.now(_tz.utc).isoformat()

    all_events, signals = strategy.generate_signals(events, bankroll, _scan_ts)
    logger.info(f"Analyzed {len(all_events)} events -> {len(signals)} raw signals")

    # Two-pass signal processing:
    #   Pass 1 — risk-check filter: remove signals that can never execute
    #            (expiry, side, liquidity, drawdown, live-data, etc.)
    #   Pass 2 — rank survivors by composite priority score, then execute
    #            top-to-bottom until portfolio exposure caps bind.
    #
    # This ensures the ranking only scores tradeable opportunities, and
    # capital-dependent checks (exposure caps) are evaluated in priority
    # order so the best signals get funded first.

    eligible = []
    skipped  = 0
    for signal in signals:
        passed, failures = run_pre_checks(signal, bankroll)
        if not passed:
            skipped += 1
            logger.info(
                f"Signal skipped [{signal.get('city')} {signal.get('date')} "
                f"{signal.get('question', '')[:30]}]: {'; '.join(failures)}"
            )
        else:
            eligible.append(signal)

    logger.info(f"Risk pre-filter: {len(eligible)} eligible, {skipped} skipped")

    # Rank only the eligible signals using the active strategy's ranking
    eligible = strategy.rank_signals(eligible, bankroll)

    if eligible:
        logger.info("--- SIGNAL PRIORITY RANKING ---")
        for rank, s in enumerate(eligible, 1):
            pc = s.get("priority_components", {})
            logger.info(
                f"[RANK] #{rank} {s.get('city')} {s.get('date')} "
                f"[{s.get('question', '')[:25]}] {s.get('recommended_side')} "
                f"| score={s.get('priority_score', 0):.4f} "
                f"ev/$={pc.get('ev_per_dollar', 0):.3f} "
                f"thesis={pc.get('thesis_margin', 0):.2f} "
                f"conf={pc.get('confidence', 0):.2f} "
                f"time={pc.get('time_efficiency', 0):.4f} "
                f"hrs={pc.get('hours_to_resolve', 0):.0f} "
                f"confirm={pc.get('confirmation', 0):.1f}"
            )
        logger.info("--- END RANKING ---")

    client   = get_clob_client()
    executed = 0
    unfunded = 0

    for signal in eligible:
        # Only portfolio-level checks remain (exposure caps) — these are
        # capital-dependent and must be evaluated in priority order.
        passed, failures = run_portfolio_checks(signal, bankroll)

        if not passed:
            unfunded += 1
            logger.info(
                f"Signal unfunded [{signal.get('city')} {signal.get('date')} "
                f"{signal.get('question', '')[:30]}]: {'; '.join(failures)}"
            )
            continue

        # Skip if we already hold this exact contract (prevents bracket
        # promotion from double-buying the same bin in the same cycle).
        from db import get_open_positions
        already_held = any(
            p.get("contract_id") == signal.get("contract_id")
            for p in get_open_positions()
            if p.get("status") == "open"
        )
        if already_held:
            logger.debug(
                f"Skipping duplicate contract {signal.get('contract_id', '')[:12]} "
                f"-- already held"
            )
            continue

        # Liquidity-aware sizing: cap the position at a fraction of
        # displayed liquidity.  Store the full Kelly target on the
        # position so underfilled positions can be topped up later.
        from config import LIQUIDITY_AWARE_SIZING, MAX_LIQUIDITY_TAKE_PCT
        original_kelly = signal.get("kelly_size", 0)
        if LIQUIDITY_AWARE_SIZING:
            liquidity = float(signal.get("liquidity_usd") or 0)
            max_take = liquidity * MAX_LIQUIDITY_TAKE_PCT
            if max_take > 0 and original_kelly > max_take:
                logger.info(
                    f"Liquidity cap: {signal.get('city')} {signal.get('date')} "
                    f"${original_kelly:.2f} -> ${max_take:.2f} "
                    f"(liquidity=${liquidity:.0f} x {MAX_LIQUIDITY_TAKE_PCT*100:.0f}%)"
                )
                signal["kelly_size"] = round(max_take, 2)
            signal["target_size_usdc"] = original_kelly

        result = execute_signal(signal, client=client)

        if result["status"] in ("placed", "paper"):
            executed += 1
            logger.info(
                f"Executed: {signal['recommended_side']} ${signal['kelly_size']:.2f} "
                f"on {signal['contract_id'][:12]} "
                f"[{signal.get('city')} {signal.get('question', '')[:20]}] "
                f"| status={result['status']} pos_id={result.get('position_id')} "
                f"priority=#{executed}"
            )
        else:
            logger.error(
                f"Execution failed for {signal.get('contract_id', '')[:12]}: {result}"
            )

    logger.info(
        f"Trading run complete: {executed} executed, {skipped} pre-filtered, "
        f"{unfunded} unfunded (capital exhausted), "
        f"{len(signals)} raw signals -> {len(eligible)} eligible"
    )

    # --- Liquidity-aware top-ups for underfilled positions ---
    from config import LIQUIDITY_AWARE_SIZING
    if LIQUIDITY_AWARE_SIZING:
        try:
            topped_up = _run_topups(all_events, client)
            if topped_up:
                logger.info(f"[TOPUP] {topped_up} position(s) topped up")
        except Exception as e:
            logger.debug(f"Top-up pass failed (non-fatal): {e}")

    # --- Phase 3: Active position management (exit engine) ---
    try:
        exit_actions = strategy.evaluate_positions()
        if exit_actions:
            _execute_exit_actions(exit_actions, client)
    except Exception as e:
        logger.exception(f"Exit evaluation failed (non-fatal): {e}")

    logger.info("=== TRADING RUN END ===")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("Highest-temperature arbitrage bot starting up")
    logger.info(f"Paper trade mode: {PAPER_TRADE}")
    logger.info(f"Database: {DB_PATH}")

    init_db()

    # ----- Startup health check: validate ensemble API connectivity -----
    # Both ECMWF and GFS must return ensemble member data.  If either fails,
    # the bot cannot produce a valid blended distribution and should NOT
    # trade.  This catches model-name changes, API outages, and auth issues
    # before any capital is deployed.
    from weather import _get_ecmwf_ensemble_distribution, _get_gfs_ensemble_distribution
    from db import get_latest_forecast_distribution
    from datetime import timedelta
    _test_lat, _test_lon = 41.85, -87.65   # Chicago

    # Try tomorrow first, fall back to today (API may not have tomorrow's run yet)
    _health_ok = False
    for _offset in [1, 0]:
        _test_date = (datetime.now().date() + timedelta(days=_offset)).isoformat()
        logger.info(f"Startup health check: testing ECMWF + GFS for Chicago {_test_date}...")
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

    # If API is temporarily down, check if DB has recent forecast data
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
    logger.info("Ensemble health check passed.")

    # Run all loops immediately on startup so the bot is fully current before
    # the scheduler takes over.  Monitor runs last so any positions from the
    # initial trading run are immediately evaluated.
    #
    # Bias update runs first — it ensures forecast_errors is fresh before the
    # trading scan reads per-model bias corrections.
    # Skip bias update if it already ran today (avoids ~3 min delay on restart)
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

    events = discovery_run()
    _cached_events_ref = events  # seed the module-level cache
    globals()["_cached_events"] = events

    # Phase 3: run forecast pull + live observations BEFORE the first trading
    # cycle so the live adjustment layer has data (hourly forecast path +
    # VC observations + observed_max_so_far) when computing probabilities.
    # Skip if they already ran recently (avoids redundant API calls on restart).
    import sqlite3 as _sqlite3
    _skip_conn = _sqlite3.connect(DB_PATH)

    # Forecast pull: skip if any forecast_runs row was written in the last 2 hours
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

    # Live observations: skip if any live_observations row was written in the last 20 min
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

    trading_run()
    run_monitor_loop()

    # Start real-time price streaming via WebSocket for stop-loss monitoring
    try:
        from price_ws import start_price_stream, load_open_position_tokens, wire_stop_loss_callback
        from realtime_exits import on_price_update
        wire_stop_loss_callback(on_price_update)
        load_open_position_tokens()
        start_price_stream()
        logger.info("Real-time price stream started for stop-loss monitoring")
    except Exception as e:
        logger.warning(f"Real-time price stream failed to start (non-fatal): {e}")

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

    # Trading: every hour at :15 (offset from forecast pull at :05 to ensure
    # all cities have fresh DB data before the trading scan reads them)
    scheduler.add_job(
        trading_run,
        trigger=CronTrigger(minute=15, timezone="UTC"),
        id="trading_run",
        name="Trading scan",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Monitor: every hour at :30
    scheduler.add_job(
        run_monitor_loop,
        trigger=CronTrigger(minute=30, timezone="UTC"),
        id="monitor_run",
        name="Position monitor",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Bias update: daily at 05:00 UTC.  Fetches the last 14 days of VC
    # observations + Open-Meteo Previous Runs ECMWF/GFS forecasts for every
    # city and rebuilds forecast_errors.  Replaces the previous
    # ERA5-reanalysis-based bias recorder.
    def _scheduled_bias_update() -> None:
        try:
            from bias_correction.bias_updater import run_bias_update
            run_bias_update()
        except Exception as e:
            logger.exception(f"Scheduled bias update failed (non-fatal): {e}")

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

    # Forecast pull: every 2 hours at :05 (5 min after discovery at :00)
    scheduler.add_job(
        _forecast_pull_job,
        trigger=CronTrigger(hour="*/2", minute=5, timezone="UTC"),
        id="forecast_pull",
        name="Forecast pull (ECMWF + GFS)",
        misfire_grace_time=600,
        coalesce=True,
    )

    # Live observations: every 20 minutes at :00 / :20 / :40
    scheduler.add_job(
        _live_observation_job,
        trigger=CronTrigger(minute="0,20,40", timezone="UTC"),
        id="live_observation",
        name="Visual Crossing live observation",
        misfire_grace_time=300,
        coalesce=True,
    )

    # Retention: daily at 04:30 UTC (before bias at 05:00)
    scheduler.add_job(
        _retention_job,
        trigger=CronTrigger(hour=4, minute=30, timezone="UTC"),
        id="retention",
        name="Hourly/obs retention purge",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # VC future-day diagnostic: every 2 hours at :07 (runs just after the
    # forecast pull at :05 so it uses fresh μ context in disagreement metrics).
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

    logger.info(
        "Scheduler started | "
        "Discovery: 00/04/08/12/16/20:00 UTC | "
        "Forecast pull: */2h at :05 | "
        "Trading: :10 every hour | "
        "Live obs: :00/:20/:40 | "
        "Monitor: :30 every hour | "
        "Retention: 04:30 UTC daily | "
        "Bias update: 05:00 UTC daily"
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        raise


if __name__ == "__main__":
    main()
