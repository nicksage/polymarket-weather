"""
top_bin_value.py — Buy the model's highest-conviction bins.

Instead of hunting for edge (model-vs-market disagreement), this strategy
asks: "Which bins does the model think are most likely to win?  Can I buy
them at a reasonable price?"

Entry rules (v1):
  - Consider the model's top N bins by model_prob (N = MAX_YES_BINS)
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
    EXIT_DYING_PROB_THRESHOLD,
    EXIT_HARD_STOP_ENABLED,
    EXIT_HARD_STOP_PCT,
    MAX_YES_BINS,
    MAX_NO_BINS,
)

logger = logging.getLogger(__name__)

# Strategy-specific defaults (overridable via .env)
import os

TBV_MIN_MODEL_PROB  = float(os.getenv("TBV_MIN_MODEL_PROB", "0.10"))
TBV_MIN_MARKET_PROB = float(os.getenv("TBV_MIN_MARKET_PROB", "0.05"))
TBV_CONFIRM_PROB    = float(os.getenv("TBV_CONFIRM_PROB", "0.90"))
TBV_MAX_UNCERTAINTY = float(os.getenv("TBV_MAX_UNCERTAINTY", "999.0"))


def tbv_qualifies_as_signal(
    model_prob: float,
    market_prob: float,
    city: str | None = None,
    forecast_sigma_c: float | None = None,
    _city_error_cache: dict | None = None,
) -> bool:
    """Single source of truth: does this bin qualify as a YES signal
    under the top_bin_value strategy?

    Checks ALL entry criteria:
      - model_prob >= TBV_MIN_MODEL_PROB
      - market_prob >= TBV_MIN_MARKET_PROB
      - forecast_sigma_c <= TBV_MAX_UNCERTAINTY (event-level uncertainty)

    Called by both the strategy (signal generation) and the dashboard
    (display).  Adding a new filter here automatically applies everywhere.
    """
    if model_prob < TBV_MIN_MODEL_PROB:
        return False
    if market_prob < TBV_MIN_MARKET_PROB:
        return False

    # Event-level forecast uncertainty check
    if (forecast_sigma_c is not None
            and TBV_MAX_UNCERTAINTY < 900
            and forecast_sigma_c > TBV_MAX_UNCERTAINTY):
        return False

    return True


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

        logger.debug(f"[{self.name}] Starting signal generation...")
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

            # Record price snapshot for ALL bins (for backtesting any strategy)
            try:
                from db import insert_bin_price_snapshots_bulk
                price_rows = []
                for o in analysis.get("outcomes", []):
                    price_rows.append({
                        "event_id": analysis.get("event_id", ""),
                        "contract_id": o.get("contract_id", ""),
                        "city": analysis.get("city"),
                        "date": analysis.get("date"),
                        "yes_price": o.get("yes_price") or o.get("market_price"),
                        "no_price": o.get("no_price"),
                        "volume_usd": o.get("volume_usd"),
                        "liquidity_usd": o.get("liquidity_usd"),
                    })
                if price_rows:
                    insert_bin_price_snapshots_bulk(price_rows, scan_ts)
            except Exception as e:
                logger.debug(f"Price snapshot write failed: {e}")

            all_events_analyzed.append(analysis)

            # --- Strategy-specific signal selection ---
            event_signals = self._select_signals(analysis, bankroll, scan_ts)
            signals.extend(event_signals)

        logger.debug(
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
        all_candidates = []
        for o in outcomes:
            mp = o.get("model_prob")
            if mp is None:
                continue
            market_price = float(o.get("market_price") or o.get("yes_price") or 0.5)
            all_candidates.append({
                "outcome": o,
                "model_prob": float(mp),
                "market_price": market_price,
            })

        all_candidates.sort(key=lambda c: c["model_prob"], reverse=True)

        # Split into YES candidates (top bins by model_prob) and NO candidates
        # (remaining bins, sorted by NO probability = 1 - model_prob, descending).
        # MAX_YES_BINS caps YES signals per event, MAX_NO_BINS caps NO signals.
        yes_candidates = []
        no_candidates = []
        for c in all_candidates:
            if (tbv_qualifies_as_signal(
                    c["model_prob"], c["market_price"],
                    forecast_sigma_c=analysis.get("forecast_sigma_c"))
                    and len(yes_candidates) < MAX_YES_BINS):
                yes_candidates.append(c)
            else:
                no_candidates.append(c)

        # NO candidates: sort by NO probability (lowest model_prob = highest NO conviction)
        no_candidates.sort(key=lambda c: c["model_prob"])
        no_candidates = no_candidates[:MAX_NO_BINS]

        selected = yes_candidates + no_candidates

        yes_cids = {c["outcome"].get("contract_id") for c in yes_candidates}

        signals = []
        for c in selected:
            o = c["outcome"]
            mp = c["model_prob"]
            market_price = c["market_price"]
            cid = o.get("contract_id")

            # YES for top bins (high model_prob), NO for the rest
            if cid in yes_cids:
                side = "YES"
                edge = mp - market_price
            else:
                side = "NO"
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
            n_yes = sum(1 for s in signals if s.get("recommended_side") == "YES")
            n_no  = sum(1 for s in signals if s.get("recommended_side") == "NO")
            yes_probs = [round(c["model_prob"]*100, 1) for c in yes_candidates[:3]]
            logger.debug(
                f"[{self.name}] {city} {date_str}: "
                f"{n_yes} YES + {n_no} NO signals "
                f"(top YES probs: {yes_probs}%)"
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
        """Rank YES signals first (primary thesis), then NO signals (hedges).

        Within each group, rank by model conviction x price discount x
        confidence x time efficiency.  YES signals always get first claim
        on capital because they represent "which bin will win" — the core
        thesis of this strategy.  NO signals fill remaining slots as hedges.
        """
        if not signals:
            return signals

        # Load city accuracy scores for ranking boost
        city_scores: dict[str, float] = {}
        try:
            import sqlite3
            from config import DB_PATH
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute(
                "SELECT city, accuracy_score FROM city_forecast_accuracy"
            ).fetchall()
            conn.close()
            if rows:
                max_score = max(r[1] for r in rows) or 1.0
                city_scores = {r[0]: r[1] / max_score for r in rows}
        except Exception:
            pass

        for s in signals:
            model_prob = float(s.get("model_p") or s.get("model_prob") or 0)
            market_price = float(s.get("market_p") or s.get("market_price") or 0.5)
            confidence = float(s.get("confidence_multiplier") or 1.0)
            side = s.get("recommended_side", "YES")

            hours = self._hours_to_resolution(s)
            time_efficiency = 1.0 / max(hours, 1.0)

            city = s.get("city", "")
            city_accuracy = 0.5 + 0.5 * city_scores.get(city, 0.5)

            if side == "YES":
                # YES ranking: model probability (conviction that this bin wins)
                # + bonus when market underprices it
                conviction = model_prob
                price_discount = max(model_prob - market_price, 0) / max(market_price, 0.01)
                discount_bonus = 1.0 + min(price_discount, 1.0)
                score = conviction * discount_bonus * confidence * time_efficiency * city_accuracy
            else:
                # NO ranking: prioritize by profit potential and distance from mu.
                #
                # profit_per_dollar: how much we earn per dollar risked if correct.
                #   NO price = 1 - market_price.  Profit = 1 - no_price = market_price.
                #   So profit_per_dollar = market_price / (1 - market_price).
                #   A bin at YES $0.30 (NO cost $0.70) yields $0.30/$0.70 = 43%.
                #   A bin at YES $0.05 (NO cost $0.95) yields $0.05/$0.95 = 5%.
                #   Cheaper YES price = better NO trade.
                #
                # distance_from_mu: how far this bin is from the model's predicted
                #   temperature (in sigma units).  Bins far from mu are safer NO
                #   bets — the temperature is very unlikely to land there.
                no_price = max(1.0 - market_price, 0.01)
                profit_per_dollar = market_price / no_price

                # Distance from mu in sigma units (further = safer NO bet)
                blended_mu = float(s.get("forecast_mu_c") or s.get("adjusted_mu_c") or 0)
                sigma = float(s.get("forecast_sigma_c") or 1.5)
                range_low = s.get("range_low")
                range_high = s.get("range_high")
                unit = s.get("unit", "celsius")
                if range_low is not None and range_high is not None:
                    bin_center = self._to_c((float(range_low) + float(range_high)) / 2.0, unit) or 0
                else:
                    bin_center = blended_mu
                distance_sigma = abs(bin_center - blended_mu) / max(sigma, 0.5)
                # Normalize: 0 sigma = 0.1, 2+ sigma = 1.0
                distance_factor = min(0.1 + distance_sigma * 0.45, 1.0)

                conviction = distance_factor
                discount_bonus = 1.0 + min(profit_per_dollar, 1.0)
                score = conviction * discount_bonus * confidence * time_efficiency * city_accuracy

            s["priority_score"] = round(score, 6)
            s["priority_components"] = {
                "side":            side,
                "conviction":      round(conviction, 4),
                "discount_bonus":  round(discount_bonus, 3),
                "confidence":      round(confidence, 3),
                "time_efficiency": round(time_efficiency, 5),
                "hours_to_resolve": round(hours, 1),
                "city_accuracy":   round(city_accuracy, 3),
            }

        # Split into YES and NO, rank each independently, YES first
        yes_signals = [s for s in signals if s.get("recommended_side") == "YES"]
        no_signals  = [s for s in signals if s.get("recommended_side") == "NO"]
        yes_signals.sort(key=lambda s: s.get("priority_score", 0), reverse=True)
        no_signals.sort(key=lambda s: s.get("priority_score", 0), reverse=True)

        return yes_signals + no_signals

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

        # Check for confirmed bins (any bin's current market price >= TBV_CONFIRM_PROB)
        # Uses the monitor-updated current_price on the position, which reflects
        # the latest market price (YES price for YES positions).
        confirmed_events: dict[str, str] = {}  # event_id -> confirmed contract_id
        for eid, event_positions in by_event.items():
            for pos in event_positions:
                curr_price = pos.get("current_price")
                if curr_price is not None and float(curr_price) >= TBV_CONFIRM_PROB:
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
                             f"market price >={TBV_CONFIRM_PROB*100:.0f}%",
                             urgency=1)

        # ---- 3. DYING — market price dropped below threshold ----
        _market_price = float(current_price) if current_price is not None else None
        if (_market_price is not None
                and _market_price < EXIT_DYING_PROB_THRESHOLD):
            return _make("DYING", "SELL",
                         f"market_price={_market_price:.3f} < "
                         f"threshold={EXIT_DYING_PROB_THRESHOLD}",
                         urgency=1)

        # ---- 4. HARD STOP — uses precomputed SL price ----
        if EXIT_HARD_STOP_ENABLED:
            sl_price = pos.get("stop_loss_price")
            if (sl_price is not None and current_price is not None
                    and float(sl_price) > 0 and float(current_price) <= float(sl_price)):
                return _make("WEAKENED", "SELL",
                             f"hard_stop: price={float(current_price):.4f} <= "
                             f"SL={float(sl_price):.4f} (entry={entry_price:.4f})",
                             urgency=0)

        # ---- 5. HEALTHY ----
        return _make("HEALTHY", "HOLD", "thesis_intact", urgency=2)


register("top_bin_value", TopBinValueStrategy)
