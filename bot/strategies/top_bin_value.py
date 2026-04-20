"""
top_bin_value.py — Buy the model's highest-conviction bins.

Instead of hunting for edge (model-vs-market disagreement), this strategy
asks: "Which bins does the model think are most likely to win?  Can I buy
them at a reasonable price?"

Entry rules (v1):
  - Consider the model's top N bins by model_prob (N = MAX_BIN_BUYS)
  - Buy YES on bins the model thinks are likely (model_prob >= TBV_MIN_MODEL_PROB)
  - Buy NO on bins the model thinks are very unlikely (implicitly, when
    ALLOWED_SIDES includes "no")
  - No minimum underpricing required to enter — the bot buys the model's
    best bins regardless of market price
  - Underpricing is a ranking factor when capital is limited

Exit rules (v1):
  - INVALIDATED: same as edge_disagreement (observed max blows past bin)
  - TOP_BIN_CONFIRMED: once any bin reaches >= TBV_CONFIRM_PROB (default 90%),
    all other YES positions for that event are exited.  NO positions are kept.
  - DYING: model probability collapsed since entry (same thresholds)
  - HARD_STOP: circuit breaker at -70%
  - No OVERPRICED exit in v1
  - No edge-reversal exit in v1

Ranking:
  model_prob x price_discount x confidence x time_efficiency
  Favors: high-probability bins that the market underprices.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

from strategies.base import Strategy
from strategies import register
from position_eval import ExitAction

from config import (
    ALLOWED_SIDES,
    EXIT_DEAD_BUFFER_C,
    EXIT_DYING_ENTRY_MIN,
    EXIT_DYING_PROB_THRESHOLD,
    EXIT_HARD_STOP_ENABLED,
    EXIT_HARD_STOP_PCT,
    MAX_BIN_BUYS,
)

logger = logging.getLogger(__name__)

# Strategy-specific defaults (overridable via .env)
import os

TBV_MIN_MODEL_PROB = float(os.getenv("TBV_MIN_MODEL_PROB", "0.10"))
TBV_CONFIRM_PROB   = float(os.getenv("TBV_CONFIRM_PROB", "0.90"))
TBV_KELLY_FRACTION = float(os.getenv("TBV_KELLY_FRACTION", "0.25"))
TBV_TOP_N_BINS     = int(os.getenv("TBV_TOP_N_BINS", str(MAX_BIN_BUYS)))


class TopBinValueStrategy(Strategy):
    name = "top_bin_value"

    # ------------------------------------------------------------------
    # ENTRY: generate signals
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        events: list[dict],
        bankroll: float,
        scan_ts: str,
    ) -> tuple[list[dict], list[dict]]:
        from weather import clear_forecast_cache, reset_tomorrowio_limit
        from db import insert_temp_event, insert_temp_outcome, insert_signal
        from edge import _write_decision_snapshots
        import uuid

        logger.info(f"[{self.name}] Starting signal generation...")
        clear_forecast_cache()
        reset_tomorrowio_limit()

        snapshot_group_id = str(uuid.uuid4())
        all_events_analyzed: list[dict] = []
        signals: list[dict] = []
        skipped = 0

        for event in events:
            analysis = self.analyze_event_base(event, bankroll)
            if analysis is None:
                skipped += 1
                continue

            # Persist to DB (same as edge_disagreement — dashboard needs this)
            try:
                event_row_id = insert_temp_event(analysis, scan_ts)
                for outcome in analysis["outcomes"]:
                    insert_temp_outcome(outcome, event_row_id, scan_ts)
            except Exception as e:
                logger.error(f"DB write failed for {analysis.get('city')}: {e}")

            try:
                _write_decision_snapshots(analysis, snapshot_group_id, scan_ts)
            except Exception as e:
                logger.debug(f"Snapshot write failed: {e}")

            all_events_analyzed.append(analysis)

            # --- Strategy-specific signal selection ---
            event_signals = self._select_signals(analysis, bankroll, scan_ts)
            signals.extend(event_signals)

        logger.info(
            f"[{self.name}] Scan complete: {len(all_events_analyzed)} events, "
            f"{len(signals)} signals, {skipped} skipped"
        )
        return all_events_analyzed, signals

    def _select_signals(
        self, analysis: dict, bankroll: float, scan_ts: str,
    ) -> list[dict]:
        """Select the model's top bins as signals."""
        from sizing import calculate_kelly_size, compute_confidence_multiplier, compute_time_scale

        outcomes = analysis.get("outcomes", [])
        if not outcomes:
            return []

        city = analysis.get("city", "")
        date_str = analysis.get("date", "")
        days_ahead = analysis.get("days_ahead", 0)

        # Get timezone for time scaling
        tz_str = None
        try:
            from edge import _lookup_event_timezone
            tz_str = _lookup_event_timezone(analysis.get("event_id", ""))
        except Exception:
            pass

        local_hr = None
        if tz_str:
            try:
                from zoneinfo import ZoneInfo
                now_local = datetime.now(ZoneInfo(tz_str))
                local_hr = float(now_local.hour) + now_local.minute / 60.0
            except Exception:
                pass

        # Build candidates: bins with valid model_prob, sorted by model_prob desc
        candidates = []
        for o in outcomes:
            mp = o.get("model_prob")
            if mp is None:
                continue
            market_price = float(o.get("market_price") or o.get("yes_price") or 0.5)
            candidates.append({
                "outcome": o,
                "model_prob": float(mp),
                "market_price": market_price,
            })

        candidates.sort(key=lambda c: c["model_prob"], reverse=True)

        # Select top N bins
        top_n = min(TBV_TOP_N_BINS, len(candidates))
        selected = candidates[:top_n]

        signals = []
        for c in selected:
            o = c["outcome"]
            mp = c["model_prob"]
            market_price = c["market_price"]

            # Determine side based on model probability
            # High model_prob → YES (we think this bin will win)
            # Low model_prob → NO (we think this bin won't win)
            if mp >= TBV_MIN_MODEL_PROB:
                side = "YES"
                edge = mp - market_price
            else:
                side = "NO"
                edge = (1.0 - mp) - (1.0 - market_price)
                # For NO side: model_no_prob - market_no_prob
                # = (1-mp) - (1-market) = market - mp
                edge = market_price - mp

            # Check ALLOWED_SIDES
            if ALLOWED_SIDES == "yes" and side != "YES":
                continue
            if ALLOWED_SIDES == "no" and side != "NO":
                continue

            # Skip bins with very low model confidence for YES
            if side == "YES" and mp < TBV_MIN_MODEL_PROB:
                continue

            # Compute EV
            ev = self._calculate_ev(mp, market_price, side)

            # Kelly sizing
            if side == "YES":
                if market_price <= 0 or market_price >= 1:
                    continue
                odds = (1.0 - market_price) / market_price
            else:
                no_price = 1.0 - market_price
                if no_price <= 0 or no_price >= 1:
                    continue
                odds = market_price / (1.0 - market_price)

            abs_edge = abs(edge)
            kelly_raw = calculate_kelly_size(
                edge=max(abs_edge, 0.01),  # minimum edge for Kelly formula
                odds=odds,
                bankroll=bankroll,
            )

            # Confidence multiplier
            conf_signal = {
                "range_low": o.get("range_low"),
                "range_high": o.get("range_high"),
                "recommended_side": side,
                "unit": o.get("unit", "celsius"),
                "days_ahead": days_ahead,
                "normalization_warning": analysis.get("normalization_warning"),
            }
            conf_state = {
                "forecast_agreement_c": analysis.get("forecast_agreement_c"),
                "adjusted_mu_c": analysis.get("adjusted_mu_c"),
                "forecast_mu_c": analysis.get("forecast_mu_c"),
                "live_adjustment_score": analysis.get("live_adjustment_score"),
            }
            confidence = compute_confidence_multiplier(conf_signal, conf_state)
            time_scale = compute_time_scale(days_ahead, local_hr)

            kelly_size = round(kelly_raw * confidence * time_scale, 2)
            if kelly_size < 1.0:
                kelly_size = 1.0

            signal = self.flatten_signal(
                {
                    **o,
                    "ev": round(ev, 4),
                    "edge": round(edge, 4),
                    "recommended_side": side,
                    "kelly_size": kelly_size,
                    "is_signal": True,
                    "confidence_multiplier": confidence,
                    "time_scale": time_scale,
                    "days_ahead": days_ahead,
                    "live_adjustment_score": analysis.get("live_adjustment_score"),
                    "adjusted_mu_c": analysis.get("adjusted_mu_c"),
                    "forecast_mu_c": analysis.get("forecast_mu_c"),
                    "forecast_sigma_c": analysis.get("forecast_sigma_c"),
                },
                analysis,
                scan_ts,
            )
            signal["gamma_market_id"] = o.get("gamma_market_id")
            signals.append(signal)

            # Log to legacy signals table
            try:
                insert_signal(
                    timestamp=scan_ts,
                    contract_id=o["contract_id"],
                    question=o.get("question"),
                    market_p=market_price,
                    model_p=mp,
                    ev=ev,
                    recommended_side=side,
                    kelly_size=kelly_size,
                )
            except Exception:
                pass

        if signals:
            logger.info(
                f"[{self.name}] {city} {date_str}: "
                f"{len(signals)} top-bin signals "
                f"(top model_probs: {[round(c['model_prob']*100, 1) for c in selected[:3]]}%)"
            )

        return signals

    # ------------------------------------------------------------------
    # RANKING
    # ------------------------------------------------------------------

    def rank_signals(
        self,
        signals: list[dict],
        bankroll: float,
    ) -> list[dict]:
        """Rank by model conviction x price discount x confidence x time.

        model_prob drives the ranking — we want the bins the model is most
        confident about.  Price discount is a bonus when the market
        underprices a bin, but not required.
        """
        if not signals:
            return signals

        for s in signals:
            model_prob = float(s.get("model_p") or s.get("model_prob") or 0)
            market_price = float(s.get("market_p") or s.get("market_price") or 0.5)
            confidence = float(s.get("confidence_multiplier") or 1.0)
            side = s.get("recommended_side", "YES")

            # Model conviction — directly use model_prob for YES,
            # (1 - model_prob) for NO
            if side == "YES":
                conviction = model_prob
                price_discount = max(model_prob - market_price, 0) / max(market_price, 0.01)
            else:
                conviction = 1.0 - model_prob
                no_market = 1.0 - market_price
                price_discount = max((1.0 - model_prob) - no_market, 0) / max(no_market, 0.01)

            # Price discount bonus: 1.0 when no discount, up to 2.0 when
            # heavily discounted.  Even at 0 discount, the signal survives
            # because conviction drives the score.
            discount_bonus = 1.0 + min(price_discount, 1.0)

            hours = self._hours_to_resolution(s)
            time_efficiency = 1.0 / max(hours, 1.0)

            score = conviction * discount_bonus * confidence * time_efficiency

            s["priority_score"] = round(score, 6)
            s["priority_components"] = {
                "conviction":      round(conviction, 4),
                "discount_bonus":  round(discount_bonus, 3),
                "confidence":      round(confidence, 3),
                "time_efficiency": round(time_efficiency, 5),
                "hours_to_resolve": round(hours, 1),
            }

        signals.sort(key=lambda s: s.get("priority_score", 0), reverse=True)
        return signals

    # ------------------------------------------------------------------
    # EXIT
    # ------------------------------------------------------------------

    def evaluate_positions(self) -> list[ExitAction]:
        """Top-bin-value exit logic:
        - INVALIDATED: observed max blows past bin (shared)
        - TOP_BIN_CONFIRMED: one bin hit >= TBV_CONFIRM_PROB, exit other YES bins
        - DYING: model probability collapsed (shared thresholds)
        - HARD_STOP: circuit breaker
        """
        from db import (
            get_open_positions,
            get_snapshot_by_id,
            get_latest_snapshot_for_contract,
            get_latest_observation,
        )
        from config import EXIT_EVAL_ENABLED

        if not EXIT_EVAL_ENABLED:
            return []

        positions = [p for p in get_open_positions()
                     if p.get("fill_status") == "filled"]
        if not positions:
            return []

        # Group positions by event for TOP_BIN_CONFIRMED logic
        from collections import defaultdict
        by_event: dict[str, list[dict]] = defaultdict(list)
        for pos in positions:
            eid = pos.get("event_id") or f"{pos.get('city')}|{pos.get('date')}"
            by_event[eid].append(pos)

        # Check for confirmed bins (any bin at >= TBV_CONFIRM_PROB)
        confirmed_events: dict[str, str] = {}  # event_id -> confirmed contract_id
        for eid, event_positions in by_event.items():
            for pos in event_positions:
                snap = get_latest_snapshot_for_contract(pos.get("contract_id", ""))
                if snap and snap.get("model_prob") is not None:
                    if float(snap["model_prob"]) >= TBV_CONFIRM_PROB:
                        confirmed_events[eid] = pos.get("contract_id", "")
                        break

        actions: list[ExitAction] = []

        for pos in positions:
            try:
                action = self._classify_position(
                    pos, confirmed_events, by_event,
                )
                if action and action.action != "HOLD":
                    actions.append(action)
                    logger.info(
                        f"[EXIT-EVAL] pos={action.position_id} "
                        f"{action.city} {action.date} "
                        f"{action.side} -> {action.classification} / "
                        f"{action.action} | {action.reason}"
                    )
            except Exception as e:
                logger.debug(f"position eval failed for {pos.get('id')}: {e}")

        priority = {"INVALIDATED": 0, "TOP_BIN_CONFIRMED": 1,
                     "DYING": 2, "WEAKENED": 3, "THRIVING": 4}
        actions.sort(key=lambda a: (a.urgency, priority.get(a.classification, 9)))
        return actions

    def _classify_position(
        self,
        pos: dict,
        confirmed_events: dict[str, str],
        by_event: dict[str, list[dict]],
    ) -> ExitAction | None:
        from db import (
            get_snapshot_by_id,
            get_latest_snapshot_for_contract,
            get_latest_observation,
        )

        pid         = pos["id"]
        contract_id = pos.get("contract_id", "")
        event_id    = pos.get("event_id") or f"{pos.get('city')}|{pos.get('date')}"
        city        = pos.get("city", "")
        date_str    = pos.get("date", "")
        side        = pos.get("side", "YES")
        entry_price = float(pos.get("entry_price") or 0)
        shares      = float(pos.get("shares") or 0)
        entry_prob  = float(pos.get("model_prob") or 0)
        unit        = pos.get("unit", "celsius")

        range_high_c = self._to_c(pos.get("range_high"), unit)
        range_low_c  = self._to_c(pos.get("range_low"), unit)

        current_snap = get_latest_snapshot_for_contract(contract_id)
        latest_obs   = get_latest_observation(event_id)

        observed_max = (latest_obs or {}).get("observed_max_so_far_c")
        current_prob = float((current_snap or {}).get("model_prob") or 0)
        current_price = pos.get("current_price")

        def _executable_bid() -> float | None:
            raw = current_price
            if raw is None and current_snap:
                raw = current_snap.get("market_price")
            return float(raw) * 0.98 if raw is not None else None

        def _make(cls, action, reason, urgency):
            return ExitAction(
                position_id=pid, contract_id=contract_id,
                event_id=event_id, city=city, date=date_str,
                side=side, classification=cls, action=action,
                reason=reason, exit_price=_executable_bid(),
                urgency=urgency,
            )

        # Local hour for time gate
        local_hour = None
        try:
            if latest_obs and latest_obs.get("pulled_at_utc"):
                pulled = datetime.fromisoformat(
                    latest_obs["pulled_at_utc"].replace("Z", "+00:00"))
                local_hour = pulled.hour + pulled.minute / 60.0
        except Exception:
            pass

        # ---- 1. INVALIDATED ----
        if (observed_max is not None
                and range_high_c is not None
                and side == "YES"
                and observed_max > range_high_c + EXIT_DEAD_BUFFER_C):
            day_advanced = local_hour is not None and local_hour >= 14
            prob_near_zero = current_prob < 0.02
            if day_advanced or prob_near_zero:
                return _make("INVALIDATED", "SELL",
                             f"observed_max={observed_max:.1f} > "
                             f"range_high={range_high_c:.1f}+{EXIT_DEAD_BUFFER_C}",
                             urgency=0)

        # For NO: invalidated if observed max is inside the bin range
        if (observed_max is not None
                and range_low_c is not None and range_high_c is not None
                and side == "NO"
                and range_low_c <= observed_max <= range_high_c
                and current_prob < 0.05 and entry_prob > 0.15):
            return _make("INVALIDATED", "SELL",
                         f"observed_max={observed_max:.1f} inside bin "
                         f"(NO side losing)", urgency=0)

        # ---- 2. TOP_BIN_CONFIRMED ----
        # If any bin for this event hit >= TBV_CONFIRM_PROB, exit all
        # other YES positions.  Keep NO positions (they benefit from
        # the confirmed bin not being theirs).
        if event_id in confirmed_events:
            confirmed_cid = confirmed_events[event_id]
            if contract_id != confirmed_cid and side == "YES":
                return _make("TOP_BIN_CONFIRMED", "SELL",
                             f"bin {confirmed_cid[:12]} confirmed at "
                             f">={TBV_CONFIRM_PROB*100:.0f}%",
                             urgency=1)

        # ---- 3. DYING ----
        if (current_prob < EXIT_DYING_PROB_THRESHOLD
                and entry_prob > EXIT_DYING_ENTRY_MIN
                and (entry_prob - current_prob) > 0.10):
            return _make("DYING", "SELL",
                         f"prob_collapse: entry={entry_prob:.3f} "
                         f"current={current_prob:.3f}",
                         urgency=1)

        # ---- 4. HARD STOP ----
        if EXIT_HARD_STOP_ENABLED:
            entry_cost = entry_price * shares
            unrealized = float(pos.get("unrealized_pnl") or 0)
            if entry_cost > 0 and unrealized <= entry_cost * EXIT_HARD_STOP_PCT:
                return _make("WEAKENED", "SELL",
                             f"hard_stop: unrealized={unrealized:.2f} <= "
                             f"{EXIT_HARD_STOP_PCT*100:.0f}% of "
                             f"cost={entry_cost:.2f}",
                             urgency=0)

        # ---- 5. HEALTHY ----
        return _make("HEALTHY", "HOLD", "thesis_intact", urgency=2)


register("top_bin_value", TopBinValueStrategy)
