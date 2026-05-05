"""
top_k_hedged.py — Hedged top-K strategy for Polymarket weather bins.

Thesis
------
For events resolving in a configurable hours-to-close window (default 35-38h),
buy the top K bins by model probability and split a total bet across them in
configurable percentages (default 70/20/10 for K=3).  This is a "hedged" play
because the K bins of a mutually-exclusive event collectively cover most of
the probability mass; one of them is very likely to win, and the asymmetric
% split lets the highest-conviction bin dominate the payout.

Per-event lifecycle
-------------------
  1. Discovery scan picks up an event currently 35-38h from resolution.
  2. We pick the top K bins by model_prob, gated by [MIN_PRICE, MAX_PRICE].
  3. Each gets sized at TKH_TOTAL_BET_SIZE × split[i].
  4. Each bin is treated as a normal position (entry, topup, stop, etc.).
  5. CONVERGENCE EXIT: if any bin in this event hits TKH_CONFIRM_PRICE, we
     immediately exit the OTHER bins in the same event — that bin "won" and
     the others are about to drop to ~zero.
  6. Per-event dedup: once we've placed any position for an event, we never
     re-enter it (controlled by TKH_MAX_TRADES_PER_EVENT, default 1).

     # === Top-K Hedged strategy ===

ACTIVE_STRATEGY=top_k_hedged

TKH_PERCENTAGE_SPLIT=70:20:10            # K (top-bin count) derived from this — must sum to 100
TKH_TOTAL_BET_SIZE=10                    # $ per qualifying event, split per percentages above
TKH_HOURS_TO_CLOSE_MIN=35                # event window lower bound (hours)
TKH_HOURS_TO_CLOSE_MAX=38                # event window upper bound
TKH_TAKE_PROFIT=0.85                     # exit a bin at this price
TKH_HARD_STOP_PCT=0.50                   # exit if price drops 50% from entry
TKH_CONFIRM_PRICE=0.80                   # any sibling bin hitting this → exit the others (HEDGE_RESOLVED)
TKH_MAX_TRADES_PER_EVENT=1               # never re-enter an event we've touched
TKH_MIN_BIN_USDC=1.05                    # floor each bin's $ allocation here (Polymarket $1 min + buffer)
Shared knobs (apply across strategies, no change needed): ALLOWED_SIDES, MAX_YES_BINS, MAX_TOTAL_EXPOSURE_PCT, MAX_EVENT_EXPOSURE_PCT, 
MAX_CITY_DAY_EXPOSURE_PCT, MAX_POSITION_PCT, MAX_SPREAD_CENTS_FOR_ENTRY, TRAIL_TIERS, TRAIL_ACTIVATION_GAIN.

Configuration
-------------
All TKH-prefixed env vars are independent of MPV_/TBV_; you can run any
strategy without touching the others.  Shared knobs (TRAIL_TIERS,
TRAIL_ACTIVATION_GAIN, MAX_YES_BINS, ALLOWED_SIDES, exposure caps) work
across strategies as-is.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from config import (
    ALLOWED_SIDES,
    MAX_YES_BINS,
)
from position_eval import ExitAction
from strategies.base import Strategy
from strategies import register

logger = logging.getLogger(__name__)


# =============================================================================
# Config — TKH-specific tunables
# =============================================================================

# Per-bin allocation ratios.  Format: "70:20:10" (must sum to 100).
# K (number of bins to buy) is derived from the length of this list.
TKH_PERCENTAGE_SPLIT_RAW = os.getenv("TKH_PERCENTAGE_SPLIT", "70:20:10")

# Total $ deployed per qualifying event, split per the percentages above.
# Default: $10 → $7/$2/$1 with the default 70:20:10 split.
TKH_TOTAL_BET_SIZE = float(os.getenv("TKH_TOTAL_BET_SIZE", "10"))

# Hours-to-event-resolution window.  Events outside this window are skipped.
TKH_HOURS_TO_CLOSE_MIN = float(os.getenv("TKH_HOURS_TO_CLOSE_MIN", "35"))
TKH_HOURS_TO_CLOSE_MAX = float(os.getenv("TKH_HOURS_TO_CLOSE_MAX", "38"))

# Basket-edge gate.  An event is only traded when the top K bins'
# combined market prices leave at least this much "room" relative to a
# fully-priced basket:
#
#     basket_edge = 1.0 − sum(top_k_market_prices)
#
# basket_edge > TKH_BASKET_EDGE_MIN  →  trade the event
#
# Intuition:
# A mutually-exclusive event has YES prices that ideally sum to ~1.0.
# When the top K bins cover most of that mass (high sum, small remainder),
# their individual prices are already high → buying them is paying close
# to par for the hedged set, so any payout barely beats cost.
# When the top K bins only cover, say, 0.65 of the mass (basket_edge =
# 0.35), the rest of the probability is scattered across long-tail bins
# and the top K are individually cheap → paying $0.65 of capital for a
# basket that pays $1.00 if any of those K wins.
#
# Default 0.05 = require at least 5 cents of room before entering.
# Set to 0.0 to disable the gate (any event in the time window qualifies).
TKH_BASKET_EDGE_MIN = float(os.getenv("TKH_BASKET_EDGE_MIN", "0.05"))

# Exit thresholds.
TKH_TAKE_PROFIT = float(os.getenv("TKH_TAKE_PROFIT", "0.85"))
TKH_HARD_STOP_PCT = float(os.getenv("TKH_HARD_STOP_PCT", "0.50"))   # looser than MPV's 0.25 — hedged set tolerates more dip
TKH_CONFIRM_PRICE = float(os.getenv("TKH_CONFIRM_PRICE", "0.80"))   # convergence threshold; 1 sibling above this exits the others

# Per-event re-entry cap.  1 = never re-enter an event (matches the
# backtest's max_trades_per_bin=1 semantics).  Set higher only for future
# strategy variants.
TKH_MAX_TRADES_PER_EVENT = int(os.getenv("TKH_MAX_TRADES_PER_EVENT", "1"))

# Polymarket CLOB minimum marketable order size, with a small buffer for
# share-quantization rounding.  Polymarket rejects BUY orders whose
# size_usdc < $1 with "invalid amount for a marketable BUY order, min
# size: $1".  When the share-rounding inside py_clob_client trims the
# requested size to e.g. 0.9994, an exact $1.00 allocation can still
# fall through.  We therefore floor each bin's allocation at $1.05 by
# default — the operator can lower this once they confirm CLOB rounding
# behaves at their split levels.
TKH_MIN_BIN_USDC = float(os.getenv("TKH_MIN_BIN_USDC", "1.05"))


def _parse_split(raw: str) -> list[float]:
    """Parse "70:20:10" into [0.70, 0.20, 0.10].  Validates sum == 100."""
    try:
        parts = [float(p.strip()) for p in raw.split(":") if p.strip()]
    except ValueError as e:
        raise ValueError(
            f"TKH_PERCENTAGE_SPLIT must be colon-separated numbers, got {raw!r}"
        ) from e
    if not parts:
        raise ValueError("TKH_PERCENTAGE_SPLIT cannot be empty")
    total = sum(parts)
    if abs(total - 100.0) > 0.1:
        raise ValueError(
            f"TKH_PERCENTAGE_SPLIT must sum to 100, got {total} from {raw!r}"
        )
    return [p / 100.0 for p in parts]


TKH_SPLIT: list[float] = _parse_split(TKH_PERCENTAGE_SPLIT_RAW)
TKH_TOP_K: int = len(TKH_SPLIT)


# =============================================================================
# Startup validation
# =============================================================================
# The shared `MAX_YES_BINS` risk gate caps how many YES positions can exist
# per (city, date) event.  TKH wants to enter exactly TKH_TOP_K bins per
# event; if MAX_YES_BINS < TKH_TOP_K the K-th bin will be silently rejected
# at the risk pipeline, degrading the strategy from "K-bin hedged" to
# "(K-1)-bin hedged" (or worse).
#
# We emit a loud WARN at module load so the operator sees the misconfig
# even before activating TKH.  We don't raise — the bot can still run
# under MPV/TBV with TKH inactive — but the warning makes the issue
# visible.  The first generate_signals call also re-emits the same warning
# at SUMMARY level (terminal) for visibility during active TKH operation.

def _check_max_yes_bins_alignment() -> None:
    """Compare MAX_YES_BINS to TKH_TOP_K and warn on misalignment."""
    if MAX_YES_BINS < TKH_TOP_K:
        logger.warning(
            f"[TKH STARTUP] MAX_YES_BINS={MAX_YES_BINS} but TKH_PERCENTAGE_SPLIT "
            f"implies K={TKH_TOP_K} bins per event.  The {TKH_TOP_K - MAX_YES_BINS} "
            f"lowest-priority bin(s) per event will be silently REJECTED by the "
            f"shared per-event YES-bin cap, degrading TKH to a "
            f"{MAX_YES_BINS}-bin hedge instead of {TKH_TOP_K}-bin.  "
            f"Set MAX_YES_BINS={TKH_TOP_K} (or higher) in .env to fix."
        )


_check_max_yes_bins_alignment()
# Track whether we've already emitted the runtime SUMMARY-level warning.
# (Module-load WARN above is for non-TKH-active operation; this gate is
# for the per-cycle visible warning during active TKH use.)
_max_yes_bins_warning_emitted = False


# =============================================================================
# Helpers
# =============================================================================

def _hours_to_close(event_or_signal: dict) -> float | None:
    """Return hours until the event's local-time resolution (23:59 local).
    Wraps sizing._hours_to_resolution; returns None on any failure so the
    caller can skip the event safely."""
    try:
        from sizing import _hours_to_resolution
        h = _hours_to_resolution(event_or_signal)
        return float(h) if h is not None else None
    except Exception:
        return None


def _held_contract_ids_for_event(event_id: str) -> set[str]:
    """Return the set of contract_ids we currently HOLD ALIVE positions
    for under TKH for this event.  An "alive" position is one that:
      - is not paper
      - has status 'open' or 'exiting' (not 'closed')
      - has fill_status != 'cancelled'

    Used by generate_signals to:
      (a) decide if the basket is already complete (len >= TKH_TOP_K).
      (b) avoid re-emitting signals for bins we already own (preventing
          duplicate positions while still allowing MISSING bins to be
          added next cycle).

    Empty set when the event has no live positions yet.
    """
    if not event_id:
        return set()
    try:
        from db import _get_conn
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT contract_id FROM positions "
                "WHERE strategy = 'top_k_hedged' "
                "  AND event_id = ? "
                "  AND COALESCE(is_paper, 0) = 0 "
                "  AND status IN ('open', 'exiting') "
                "  AND COALESCE(fill_status, '') != 'cancelled'",
                (event_id,),
            ).fetchall()
        return {r["contract_id"] for r in rows if r["contract_id"]}
    except Exception as e:
        logger.warning(
            f"[TKH] _held_contract_ids_for_event query failed: {e}"
        )
        return set()


def _event_has_been_traded(event_id: str) -> int:    # noqa: U100
    """Backwards-compat shim: returns the number of held bins for
    this event.  Kept for tests in tests/test_top_k_hedged.py and any
    external callers that may import this symbol -- the IDE may flag
    it as unused inside this module, that's expected."""
    return len(_held_contract_ids_for_event(event_id))


# =============================================================================
# Strategy
# =============================================================================

class TopKHedgedStrategy(Strategy):
    name = "top_k_hedged"

    # ------------------------------------------------------------------
    # SIGNAL GENERATION
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        events: list[dict],
        bankroll: float,
        scan_ts: str,
    ) -> tuple[list[dict], list[dict]]:
        """For each event currently in the [MIN, MAX] hours-to-close window
        and not yet traded under this strategy, emit K signals for the
        top K bins by model_prob, sized per TKH_PERCENTAGE_SPLIT.
        """
        if ALLOWED_SIDES == "no":
            # TKH only buys YES bins of the top K; if YES is not allowed, skip.
            return events, []

        all_events_analyzed: list[dict] = []
        signals: list[dict] = []

        # Per-cycle telemetry — surfaces in terminal + activity_log so the
        # operator can see how many events the funnel is producing at each
        # gate (window → dedup → bin-count → basket-edge → emitted).
        funnel = {
            "n_total":         len(events),
            "n_in_window":     0,
            "n_already_traded":0,
            "n_too_few_bins":  0,
            "n_below_edge":    0,
            "n_qualified":     0,
        }

        for event in events:
            # NOTE: TKH operates purely on Polymarket market data and does
            # NOT consume weather forecasts.  We therefore SKIP the
            # `analyze_event_base()` call (which would invoke the
            # NWS/Open-Meteo/NOAA/Tomorrow.io ensemble pipeline).  The
            # event's `outcomes[].yes_price` is already populated by the
            # upstream Polymarket discovery scan, which is all we need.
            # This mirrors the market_price_value strategy's behavior and
            # avoids unnecessary API calls.
            all_events_analyzed.append(event)

            # ---- Per-event filters ----------------------------------------
            event_id = event.get("event_id") or ""

            # Time-window filter: trade on first cycle the event drops INTO
            # the window AND we haven't already engaged with it.
            hours = _hours_to_close(event)
            if hours is None or not (
                TKH_HOURS_TO_CLOSE_MIN <= hours <= TKH_HOURS_TO_CLOSE_MAX
            ):
                continue
            funnel["n_in_window"] += 1

            # ---- Per-event dedup: count of HELD bins, not trade attempts -
            # Self-healing semantics: skip the event ONLY if we already
            # have all K bins filled.  Otherwise allow signals for the
            # missing bins (filtered per-bin below to skip those we
            # already own).  Replaces the old "any position = skip"
            # rule, which permanently locked partially-placed baskets
            # into incomplete state.
            held_ids = _held_contract_ids_for_event(event_id)
            n_held = len(held_ids)
            if n_held >= TKH_TOP_K:
                funnel["n_already_traded"] += 1
                logger.debug(
                    f"[TKH] event {event_id} {event.get('city')} "
                    f"{event.get('date')} skipped — basket already "
                    f"complete ({n_held}/{TKH_TOP_K} bins held)"
                )
                continue

            # ---- Top-K bin selection --------------------------------------
            outcomes = event.get("outcomes") or []
            if not outcomes:
                funnel["n_too_few_bins"] += 1
                continue

            # Sort all bins by model_prob desc.  Use yes_price as fallback
            # when model_prob is NULL (e.g. when running without weather data).
            def _bin_prob(o: dict) -> float:
                return float(
                    o.get("model_prob")
                    or o.get("yes_price")
                    or o.get("market_price")
                    or 0
                )

            sorted_bins = sorted(outcomes, key=_bin_prob, reverse=True)
            top_k = sorted_bins[:TKH_TOP_K]
            if len(top_k) < TKH_TOP_K:
                # Event has fewer bins than K — can't form a full hedged set.
                funnel["n_too_few_bins"] += 1
                continue

            # ---- Off-rank holdings safeguard ----
            # If we already hold bins for this event but they're not in
            # the current top-K (model has rotated since we entered),
            # placing the new top-K would create >K total positions for
            # the event -- breaking the basket semantic and over-
            # deploying capital.  Skip the event with a clear log and
            # let the operator decide via fill_missing_bins.py.
            if n_held > 0:
                top_k_ids = {str(b.get("contract_id") or "") for b in top_k}
                off_rank_held = [c for c in held_ids if c not in top_k_ids]
                if off_rank_held:
                    funnel["n_already_traded"] += 1
                    logger.info(
                        f"[TKH] event {event_id} {event.get('city')} "
                        f"{event.get('date')} skipped -- "
                        f"{len(off_rank_held)} held bin(s) off-rank under "
                        f"current prices (model rotated).  Adding top-K "
                        f"would create >K total positions; manual review "
                        f"via fill_missing_bins.py recommended."
                    )
                    continue

            # ---- Basket-edge gate ----------------------------------------
            # basket_edge = 1.0 − sum(top_k market prices)
            # Skip if edge ≤ TKH_BASKET_EDGE_MIN.  See module docstring on
            # the variable for intuition.
            sum_top_k = sum(_bin_prob(b) for b in top_k)
            basket_edge = 1.0 - sum_top_k
            if basket_edge <= TKH_BASKET_EDGE_MIN:
                funnel["n_below_edge"] += 1
                logger.debug(
                    f"[TKH] event {event_id} {event.get('city')} "
                    f"{event.get('date')} skipped -- basket_edge "
                    f"{basket_edge:.4f} <= TKH_BASKET_EDGE_MIN "
                    f"{TKH_BASKET_EDGE_MIN:.4f} (top {TKH_TOP_K} sum "
                    f"{sum_top_k:.4f})"
                )
                continue
            funnel["n_qualified"] += 1

            # ---- Per-bin sizing ------------------------------------------
            n_emitted_for_event = 0
            n_skipped_already_held = 0
            for rank, bin_outcome in enumerate(top_k):
                # Per-bin self-heal: skip emitting a signal for any bin
                # we already hold.  Lets the basket be completed across
                # multiple cycles without creating duplicate positions.
                bin_cid = str(bin_outcome.get("contract_id") or "")
                if bin_cid and bin_cid in held_ids:
                    n_skipped_already_held += 1
                    continue

                pct = TKH_SPLIT[rank]
                bet_amount = round(TKH_TOTAL_BET_SIZE * pct, 2)
                if bet_amount <= 0:
                    continue
                n_emitted_for_event += 1

                # Floor each bin's allocation at TKH_MIN_BIN_USDC so that
                # share-quantization rounding inside the CLOB client never
                # drops the marketable amount below Polymarket's $1 minimum.
                # The smallest split slot (e.g. 10% of $10 = $1.00) is the
                # one that gets bumped; the larger slots are already well
                # above the floor.  Slight totals overshoot is acceptable.
                if bet_amount < TKH_MIN_BIN_USDC:
                    logger.info(
                        f"[TKH] bin rank={rank} {event.get('city')} "
                        f"{event.get('date')} allocation ${bet_amount:.2f} "
                        f"below floor ${TKH_MIN_BIN_USDC:.2f} -- bumping up "
                        f"to clear Polymarket $1 minimum"
                    )
                    bet_amount = TKH_MIN_BIN_USDC

                yes_price = _bin_prob(bin_outcome)
                shares = round(bet_amount / yes_price, 4) if yes_price > 0 else 0

                signal = self.flatten_signal(
                    {
                        **bin_outcome,
                        "market_price":       yes_price,
                        "model_prob":         yes_price,
                        "ev":                 0.0,
                        "edge":               0.0,
                        "recommended_side":   "YES",
                        "kelly_size":         bet_amount,
                        "is_signal":          True,
                        "confidence_multiplier": 1.0,
                        "time_scale":         1.0,
                        "days_ahead":         None,
                        "forecast_sigma_c":   None,
                    },
                    event,
                    scan_ts,
                )
                signal["gamma_market_id"] = bin_outcome.get("gamma_market_id")
                signal["strategy"] = "top_k_hedged"
                # Per-event sizing target: each bin's slice.  Topups won't
                # over-fund the bin since target_size_usdc is per-bin.
                signal["target_size_usdc"] = bet_amount
                # Tag the bin's intra-event rank for diagnostics.
                signal["tkh_bin_rank"] = rank
                signal["tkh_bin_pct"] = pct
                signal["tkh_hours_to_close"] = hours
                signals.append(signal)

            # Self-heal visibility: when an event is partially-held and
            # we emit fresh signals for the missing bins, surface that
            # so the operator can see auto-completion happening.  Quiet
            # when n_skipped_already_held == 0 (fresh event, normal path).
            if n_skipped_already_held > 0:
                logger.info(
                    f"[TKH] event {event_id} {event.get('city')} "
                    f"{event.get('date')} self-heal: emitted "
                    f"{n_emitted_for_event} new signal(s) for missing bins "
                    f"({n_skipped_already_held} bin(s) already held)"
                )

        # ---- Runtime MAX_YES_BINS misalignment warning ------------------
        # Module-load already emitted a WARN; this surfaces it again at
        # SUMMARY level (visible in the operator's terminal flow) the first
        # time TKH actually runs a cycle.  After the first cycle, we go
        # quiet — repeating every cycle would be noise.
        global _max_yes_bins_warning_emitted
        if not _max_yes_bins_warning_emitted and MAX_YES_BINS < TKH_TOP_K:
            logger.log(25,
                f"[TKH STARTUP-CHECK] MAX_YES_BINS={MAX_YES_BINS} < "
                f"TKH_TOP_K={TKH_TOP_K} — only {MAX_YES_BINS} of {TKH_TOP_K} "
                f"hedged bins per event will pass the per-event YES-bin cap.  "
                f"Raise MAX_YES_BINS to {TKH_TOP_K} in .env for full hedging."
            )
            try:
                from activity import log_activity
                log_activity(
                    "TKH", level="WARN",
                    message=(
                        f"MAX_YES_BINS={MAX_YES_BINS} below TKH_TOP_K={TKH_TOP_K} "
                        f"— hedged set will be truncated to {MAX_YES_BINS} bins per event"
                    ),
                    max_yes_bins=MAX_YES_BINS, tkh_top_k=TKH_TOP_K,
                    misconfig="max_yes_bins_below_tkh_top_k",
                )
            except Exception:
                pass
            _max_yes_bins_warning_emitted = True

        # ---- Per-cycle funnel summary -----------------------------------
        # Always emit (even when nothing qualified) so the operator can
        # see WHY a cycle produced no signals.  SUMMARY level reaches the
        # terminal; activity_log entry persists for the dashboard.
        n_qualifying_events = len(set(s.get("event_id") for s in signals))
        funnel_msg = (
            f"[TKH FUNNEL] {funnel['n_total']} events scanned -> "
            f"{funnel['n_in_window']} in [{TKH_HOURS_TO_CLOSE_MIN:g}-"
            f"{TKH_HOURS_TO_CLOSE_MAX:g}h] window -> "
            f"{funnel['n_qualified']} passed all gates -> "
            f"{len(signals)} signals across {n_qualifying_events} event(s)"
        )
        # Detail line of why each in-window event was filtered (if any)
        skipped_in_window = (
            funnel["n_already_traded"]
            + funnel["n_too_few_bins"]
            + funnel["n_below_edge"]
        )
        skip_breakdown = ""
        if skipped_in_window > 0:
            parts = []
            if funnel["n_already_traded"]:
                parts.append(f"already-traded={funnel['n_already_traded']}")
            if funnel["n_too_few_bins"]:
                parts.append(f"too-few-bins={funnel['n_too_few_bins']}")
            if funnel["n_below_edge"]:
                parts.append(
                    f"below-basket-edge={funnel['n_below_edge']} "
                    f"(threshold={TKH_BASKET_EDGE_MIN:.4f})"
                )
            skip_breakdown = " | in-window skips: " + ", ".join(parts)

        # SUMMARY-level so it appears in the terminal alongside the trading
        # cycle's other top-line counters.
        logger.log(25, funnel_msg + skip_breakdown)   # 25 = SUMMARY level

        # Persist to activity_log so the dashboard's Activity tab can show
        # the funnel history over time.  Best-effort; don't crash on logger errors.
        try:
            from activity import log_activity
            log_activity(
                "TKH",
                f"funnel: {funnel['n_in_window']}/{funnel['n_total']} in window, "
                f"{funnel['n_qualified']} qualified -> {len(signals)} signals",
                level="INFO",
                **funnel,
                n_signals_emitted=len(signals),
                n_qualifying_events=n_qualifying_events,
                window_min_h=TKH_HOURS_TO_CLOSE_MIN,
                window_max_h=TKH_HOURS_TO_CLOSE_MAX,
                basket_edge_min=TKH_BASKET_EDGE_MIN,
            )
        except Exception:
            pass

        return all_events_analyzed, signals

    # ------------------------------------------------------------------
    # RANKING
    # ------------------------------------------------------------------

    def rank_signals(
        self,
        signals: list[dict],
        bankroll: float,
        client=None,
    ) -> list[dict]:
        """Orderbook-aware ranking for TKH.

        Unlike MPV/TBV, TKH does NOT use MAX_SPREAD_CENTS_FOR_ENTRY
        as a HARD GATE.  TKH's basket thesis requires owning every K
        bin in an event; spread-rejecting smaller bins of thin markets
        leaves incomplete baskets which break the hedge structure
        (the event becomes "engaged" via per-event dedup, so the
        rejected bins never get a second chance).
        Wide spreads simply cost a few cents of slippage on entry --
        cheaper than skipping the bin entirely and breaking the basket.

        We still SCORE bins by spread + depth so capital-constrained
        cycles execute the tightest-spread bins first, but a wide-
        spread bin gets a LOW score (not -1000), so it survives the
        spread filter in main.py and gets executed if there's capacity.
        """
        from config import (
            RANK_TOP_N_FOR_ORDERBOOK,
            MAX_SPREAD_CENTS_FOR_ENTRY,
        )

        # Pre-rank by static liquidity proxy (free, no API).
        for s in signals:
            s["_static_liquidity"] = float(s.get("liquidity_usd") or 0)
        signals.sort(key=lambda s: s["_static_liquidity"], reverse=True)

        if client is None:
            # No book data — all signals get default scores; execution
            # goes by static liquidity order.
            for s in signals:
                s.setdefault("priority_score", s["_static_liquidity"])
                s.setdefault("priority_components", {"reason": "no_client"})
            return signals

        from execution import get_orderbook_snapshot
        for s in signals[:RANK_TOP_N_FOR_ORDERBOOK]:
            token_id = s.get("yes_token_id")
            if not token_id:
                s["priority_score"] = -100
                s["priority_components"] = {"reason": "no_token_id"}
                continue
            snap = get_orderbook_snapshot(client, token_id)
            if snap is None:
                s["priority_score"] = -100
                s["priority_components"] = {"reason": "orderbook_fetch_failed"}
                continue
            spread_cents = snap["spread_cents"]
            spread_score = max(0.0, 10.0 - (spread_cents or 99))
            sweepable = float(snap.get("sweepable_usdc") or 0)
            target = float(s.get("kelly_size") or 1)
            depth_score = min(sweepable / target, 1.0) * 10.0 if target > 0 else 0.0
            s["priority_score"] = spread_score + depth_score
            wide_spread = (
                MAX_SPREAD_CENTS_FOR_ENTRY > 0
                and spread_cents is not None
                and spread_cents > MAX_SPREAD_CENTS_FOR_ENTRY
            )
            s["priority_components"] = {
                "spread_cents":     spread_cents,
                "sweepable_usdc":   sweepable,
                "spread_score":     round(spread_score, 2),
                "depth_score":      round(depth_score, 2),
                "wide_spread":      wide_spread,
                "wide_spread_note": (
                    f"spread {spread_cents}c > "
                    f"{MAX_SPREAD_CENTS_FOR_ENTRY}c cap "
                    f"(TKH bypasses spread gate to preserve basket)"
                    if wide_spread else None
                ),
            }
            if wide_spread:
                logger.info(
                    f"[TKH] wide-spread bin {s.get('city')} "
                    f"{s.get('date')} "
                    f"spread={spread_cents}c (cap={MAX_SPREAD_CENTS_FOR_ENTRY}c) "
                    f"-- proceeding anyway to preserve basket"
                )
        # Default score for signals outside top N
        for s in signals[RANK_TOP_N_FOR_ORDERBOOK:]:
            s.setdefault("priority_score", 0.0)
            s.setdefault("priority_components", {"reason": "below_rank_top_n"})
        signals.sort(key=lambda s: s.get("priority_score", 0.0), reverse=True)
        return signals

    # ------------------------------------------------------------------
    # POSITION EVALUATION
    # ------------------------------------------------------------------

    def evaluate_positions(self) -> list[ExitAction]:
        """Classify each open TKH position.  Exit logic in priority order:

          1. TAKE_PROFIT      price ≥ TKH_TAKE_PROFIT
          2. HEDGE_RESOLVED   any sibling bin in this event has hit
                              TKH_CONFIRM_PRICE → exit the others
          3. TRAILING_STOP    shared trail logic (TRAIL_TIERS)
          4. HARD_STOP        price ≤ entry × (1 - TKH_HARD_STOP_PCT)
          5. HEALTHY          HOLD
        """
        from db import get_open_positions

        all_open = get_open_positions()
        positions = [p for p in all_open
                     if (p.get("strategy") or "") == "top_k_hedged"
                     and p.get("status") in ("open", "exiting")]
        if not positions:
            return []

        # Group by event_id so we can do the convergence check.
        by_event: dict[str, list[dict]] = {}
        for p in positions:
            eid = p.get("event_id") or f"{p.get('city')}|{p.get('date')}"
            by_event.setdefault(eid, []).append(p)

        # For each event, find the contract_id of any sibling that hit
        # TKH_CONFIRM_PRICE.  If found, the OTHER bins of that event get
        # an immediate HEDGE_RESOLVED exit.
        confirmed_winner: dict[str, str] = {}
        for eid, group in by_event.items():
            for pos in group:
                price = pos.get("current_price")
                if price is not None and float(price) >= TKH_CONFIRM_PRICE:
                    confirmed_winner[eid] = pos.get("contract_id", "")
                    break

        actions: list[ExitAction] = []
        for pos in positions:
            action = self._classify(pos, confirmed_winner)
            if action and action.action != "HOLD":
                actions.append(action)
                logger.info(
                    f"[TKH-EXIT] pos={action.position_id} "
                    f"{action.city} {action.date} {action.side} -> "
                    f"{action.classification} | {action.reason}"
                )

        # Priority ordering — same shape as MPV's evaluator
        priority = {
            "TAKE_PROFIT":     0,
            "HEDGE_RESOLVED":  0,
            "TRAILING_STOP":   1,
            "HARD_STOP":       1,
        }
        actions.sort(key=lambda a: (a.urgency, priority.get(a.classification, 9)))
        return actions

    def _classify(
        self,
        pos: dict,
        confirmed_winner: dict[str, str],
    ) -> ExitAction | None:
        pid          = pos["id"]
        contract_id  = pos.get("contract_id", "")
        event_id     = pos.get("event_id") or f"{pos.get('city')}|{pos.get('date')}"
        city         = pos.get("city", "")
        date_str     = pos.get("date", "")
        side         = pos.get("side", "YES")
        entry_price  = float(pos.get("entry_price") or 0)
        peak_price   = float(pos.get("peak_price") or entry_price)
        current      = pos.get("current_price")

        def _executable_bid() -> float | None:
            return float(current) * 0.98 if current is not None else None

        def _make(cls: str, action: str, reason: str, urgency: int) -> ExitAction:
            return ExitAction(
                position_id=pid, contract_id=contract_id,
                event_id=event_id, city=city, date=date_str,
                side=side, classification=cls, action=action,
                reason=reason, exit_price=_executable_bid(),
                urgency=urgency,
            )

        if current is None:
            return _make("HEALTHY", "HOLD", "no_price_data", urgency=2)
        price = float(current)

        # ---- 1. TAKE PROFIT ----
        if price >= TKH_TAKE_PROFIT:
            return _make(
                "TAKE_PROFIT", "SELL",
                f"price={price:.4f} >= TP={TKH_TAKE_PROFIT}",
                urgency=0,
            )

        # ---- 2. HEDGE RESOLVED (sibling won) ----
        if event_id in confirmed_winner:
            winner_cid = confirmed_winner[event_id]
            if contract_id != winner_cid:
                return _make(
                    "HEDGE_RESOLVED", "SELL",
                    f"sibling bin {winner_cid[:12]} hit "
                    f"{TKH_CONFIRM_PRICE*100:.0f}% — this bin lost",
                    urgency=0,
                )

        # ---- 3. TRAILING STOP (shared logic) ----
        from realtime_exits import _evaluate_trail as _shared_trail
        trail_decision = _shared_trail(entry_price, peak_price, price)
        if trail_decision is not None:
            trail_level, _tag = trail_decision
            from trailing_stop import lookup_trail_pct
            from config import TRAIL_TIERS as _tiers
            tier_pct = lookup_trail_pct(peak_price, _tiers) or 0.0
            return _make(
                "TRAILING_STOP", "SELL",
                f"price={price:.4f} <= trail={trail_level:.4f} "
                f"(peak={peak_price:.4f}, tier={tier_pct:.0%})",
                urgency=0,
            )

        # ---- 4. HARD STOP ----
        hard_stop_level = entry_price * (1 - TKH_HARD_STOP_PCT)
        if price <= hard_stop_level:
            return _make(
                "HARD_STOP", "SELL",
                f"price={price:.4f} <= hard_stop={hard_stop_level:.4f} "
                f"(entry={entry_price:.4f}, {TKH_HARD_STOP_PCT*100:.0f}%)",
                urgency=0,
            )

        # ---- 5. HEALTHY ----
        return _make("HEALTHY", "HOLD", "thesis_intact", urgency=2)


register("top_k_hedged", TopKHedgedStrategy)
