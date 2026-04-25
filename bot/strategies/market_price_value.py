"""
market_price_value.py — Buy bins the market thinks are most likely to win.

Pure market-price strategy: no weather model, no ensemble forecasts, no
API calls beyond Polymarket discovery. Entries are based solely on the
market's YES price falling within a configurable range.

Entry rules:
  - Buy the top N bins by YES price (N = MAX_YES_BINS)
  - Only buy bins where MPV_MIN_PRICE <= yes_price <= MPV_MAX_PRICE
  - Flat dollar sizing per bin (MPV_BET_SIZE)

Exit rules:
  - TRAILING_STOP: adaptive trail from peak price (MPV_TRAIL_PCT)
    Trail activates once price rises MPV_TRAIL_ACTIVATION above entry.
  - TAKE_PROFIT: sell when price reaches MPV_TAKE_PROFIT
  - TOP_BIN_CONFIRMED: when any bin hits >= MPV_CONFIRM_PRICE, exit others
  - DYING: market price drops below EXIT_DYING_PROB_THRESHOLD
  - HARD_STOP: fixed stop at entry * (1 - MPV_HARD_STOP_PCT) as fallback
    before trailing stop activates

Ranking:
  Higher market price first (market's strongest conviction gets priority).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from strategies.base import Strategy
from strategies import register
from position_eval import ExitAction

from config import (
    ALLOWED_SIDES,
    EXIT_DYING_PROB_THRESHOLD,
    MAX_YES_BINS,
    MAX_NO_BINS,
)

logger = logging.getLogger(__name__)

MPV_MIN_PRICE = float(os.getenv("MPV_MIN_PRICE", "0.25"))
MPV_MAX_PRICE = float(os.getenv("MPV_MAX_PRICE", "0.50"))
MPV_BET_SIZE = float(os.getenv("MPV_BET_SIZE", "500"))
MPV_TAKE_PROFIT = float(os.getenv("MPV_TAKE_PROFIT", "0.90"))
MPV_TRAIL_PCT = float(os.getenv("MPV_TRAIL_PCT", "0.12"))
MPV_TRAIL_ACTIVATION = float(os.getenv("MPV_TRAIL_ACTIVATION", "0.10"))
MPV_HARD_STOP_PCT = float(os.getenv("MPV_HARD_STOP_PCT", "0.30"))
MPV_CONFIRM_PRICE = float(os.getenv("MPV_CONFIRM_PRICE", "0.75"))
MPV_TOP_BIN_ONLY = os.getenv("MPV_TOP_BIN_ONLY", "true").lower() in ("true", "1", "yes")


class MarketPriceValueStrategy(Strategy):
    name = "market_price_value"

    # ------------------------------------------------------------------
    # ENTRY: generate signals from market prices only
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        events: list[dict],
        bankroll: float,
        scan_ts: str,
    ) -> tuple[list[dict], list[dict]]:
        from db import (
            insert_temp_event, insert_temp_outcome,
            insert_bin_price_snapshots_bulk,
            update_outcomes_ml_bin_probs_bulk,
        )
        from datetime import date

        all_events: list[dict] = []
        signals: list[dict] = []
        skipped = 0
        ml_persisted_total = 0
        today_local = date.today()

        for event in events:
            city = event.get("city", "")
            date_str = event.get("date")
            outcomes = event.get("outcomes", [])

            if not date_str or not outcomes:
                skipped += 1
                continue

            # Build analysis dict from raw event data (no weather model)
            analysis = {
                **event,
                "forecast_mu_c": None,
                "forecast_sigma_c": None,
                "days_ahead": None,
                "normalization_warning": False,
                "outcomes": outcomes,
            }

            # Persist event + outcomes to DB for dashboard
            try:
                event_row_id = insert_temp_event(analysis, scan_ts)
                for o in outcomes:
                    insert_temp_outcome(o, event_row_id, scan_ts)
            except Exception as e:
                logger.debug(f"DB write failed for {city}: {e}")

            # Record price snapshots for all bins (backtesting data)
            try:
                price_rows = []
                for o in outcomes:
                    price_rows.append({
                        "event_id": event.get("event_id", ""),
                        "contract_id": o.get("contract_id", ""),
                        "city": city,
                        "date": date_str,
                        "yes_price": o.get("yes_price") or o.get("market_price"),
                        "no_price": o.get("no_price"),
                        "volume_usd": o.get("volume_usd"),
                        "liquidity_usd": o.get("liquidity_usd"),
                    })
                if price_rows:
                    insert_bin_price_snapshots_bulk(price_rows, scan_ts)
            except Exception:
                pass

            all_events.append(analysis)

            # ----- ML bin probabilities (D=0 only) ----------------------
            # Compute the pooled model's empirical-CDF probability for each
            # bin and UPDATE the temp_outcomes rows we just inserted.  No
            # behavior change to MPV trade logic — these probs are for
            # dashboard display + future veto/sizing logic.
            try:
                target_d = date.fromisoformat(date_str)
                if (target_d == today_local
                        and event.get("event_id")
                        and event.get("lat") is not None
                        and event.get("lon") is not None):
                    bins_c: list[tuple[float | None, float | None]] = []
                    for o in outcomes:
                        lo = o.get("range_low")
                        hi = o.get("range_high")
                        unit = (o.get("unit") or "celsius").lower()
                        def _to_c(v):
                            if v is None:
                                return None
                            v = float(v)
                            return (v - 32.0) * 5.0 / 9.0 if unit == "fahrenheit" else v
                        bins_c.append((_to_c(lo), _to_c(hi)))

                    from ml.inference import get_ml_bin_probabilities
                    ml = get_ml_bin_probabilities(
                        city=event.get("city"),
                        lat=float(event["lat"]), lon=float(event["lon"]),
                        target_date=target_d,
                        event_id=event.get("event_id"),
                        bins=bins_c,
                    )
                    if ml and ml.get("probabilities"):
                        updates = []
                        for o, p in zip(outcomes, ml["probabilities"]):
                            cid = o.get("contract_id")
                            if not cid:
                                continue
                            updates.append({
                                "scan_timestamp":   scan_ts,
                                "contract_id":      cid,
                                "ml_bin_prob":      float(p),
                                "ml_decision_hour": ml.get("closest_fold"),
                                "ml_model_version": ml.get("model_version"),
                            })
                        n_persisted = update_outcomes_ml_bin_probs_bulk(updates)
                        ml_persisted_total += n_persisted
            except Exception as e:
                logger.debug(f"[{self.name}] ml bin probs failed for {city}: {e}")

            # Select bins by market price
            event_signals = self._select_signals(event, scan_ts)
            signals.extend(event_signals)

        logger.debug(
            f"[{self.name}] Scan complete: {len(all_events)} events, "
            f"{len(signals)} signals, {skipped} skipped, "
            f"ml_bin_probs_persisted={ml_persisted_total}"
        )
        return all_events, signals

    def _select_signals(self, event: dict, scan_ts: str) -> list[dict]:
        outcomes = event.get("outcomes", [])
        if not outcomes:
            return []

        city = event.get("city", "")
        date_str = event.get("date", "")

        # Build candidate list sorted by price descending
        all_bins = []
        for o in outcomes:
            yes_price = float(o.get("yes_price") or o.get("market_price") or 0)
            all_bins.append((o, yes_price))
        all_bins.sort(key=lambda c: c[1], reverse=True)

        if not all_bins:
            return []

        if MPV_TOP_BIN_ONLY:
            # The event's #1 bin must be in our price range to trade this event
            top_bin, top_price = all_bins[0]
            if not (MPV_MIN_PRICE <= top_price <= MPV_MAX_PRICE):
                return []

        # Buy bins in our price range (up to MAX_YES_BINS)
        candidates = [(o, p) for o, p in all_bins if MPV_MIN_PRICE <= p <= MPV_MAX_PRICE]
        selected = candidates[:MAX_YES_BINS]

        if ALLOWED_SIDES == "no":
            return []

        signals = []
        for o, yes_price in selected:
            shares = round(MPV_BET_SIZE / yes_price, 4) if yes_price > 0 else 0

            signal = self.flatten_signal(
                {
                    **o,
                    "market_price": yes_price,
                    "model_prob": yes_price,
                    "ev": 0.0,
                    "edge": 0.0,
                    "recommended_side": "YES",
                    "kelly_size": MPV_BET_SIZE,
                    "is_signal": True,
                    "confidence_multiplier": 1.0,
                    "time_scale": 1.0,
                    "days_ahead": None,
                    "forecast_sigma_c": None,
                },
                event,
                scan_ts,
            )
            signal["gamma_market_id"] = o.get("gamma_market_id")
            signals.append(signal)

        return signals

    # ------------------------------------------------------------------
    # RANKING
    # ------------------------------------------------------------------

    def rank_signals(
        self,
        signals: list[dict],
        bankroll: float,
    ) -> list[dict]:
        """Rank by market price (highest first) then by liquidity."""
        for s in signals:
            market_price = float(s.get("market_p") or s.get("market_price") or 0)
            liquidity = float(s.get("liquidity_usd") or 0)

            score = market_price * 100 + min(liquidity / 1000, 1.0)

            s["priority_score"] = round(score, 4)
            s["priority_components"] = {
                "market_price": round(market_price, 4),
                "liquidity": round(liquidity, 0),
            }

        signals.sort(key=lambda s: s.get("priority_score", 0), reverse=True)
        return signals

    # ------------------------------------------------------------------
    # EXIT
    # ------------------------------------------------------------------

    def evaluate_positions(self) -> list[ExitAction]:
        from db import get_open_positions
        from config import EXIT_EVAL_ENABLED

        if not EXIT_EVAL_ENABLED:
            return []

        positions = [p for p in get_open_positions()
                     if p.get("fill_status") == "filled"]
        if not positions:
            return []

        from collections import defaultdict
        by_event: dict[str, list[dict]] = defaultdict(list)
        for pos in positions:
            eid = pos.get("event_id") or f"{pos.get('city')}|{pos.get('date')}"
            by_event[eid].append(pos)

        # Check for confirmed bins (market price >= MPV_CONFIRM_PRICE)
        confirmed_events: dict[str, str] = {}
        for eid, event_positions in by_event.items():
            for pos in event_positions:
                curr_price = pos.get("current_price")
                if curr_price is not None and float(curr_price) >= MPV_CONFIRM_PRICE:
                    confirmed_events[eid] = pos.get("contract_id", "")
                    break

        actions: list[ExitAction] = []

        for pos in positions:
            action = self._classify_position(pos, confirmed_events)
            if action and action.action != "HOLD":
                actions.append(action)
                logger.info(
                    f"[EXIT-EVAL] pos={action.position_id} "
                    f"{action.city} {action.date} "
                    f"{action.side} -> {action.classification} / "
                    f"{action.action} | {action.reason}"
                )

        priority = {"TAKE_PROFIT": 0, "TOP_BIN_CONFIRMED": 1,
                     "TRAILING_STOP": 2, "DYING": 3, "HARD_STOP": 4}
        actions.sort(key=lambda a: (a.urgency, priority.get(a.classification, 9)))
        return actions

    def _classify_position(
        self,
        pos: dict,
        confirmed_events: dict[str, str],
    ) -> ExitAction | None:
        pid = pos["id"]
        contract_id = pos.get("contract_id", "")
        event_id = pos.get("event_id") or f"{pos.get('city')}|{pos.get('date')}"
        city = pos.get("city", "")
        date_str = pos.get("date", "")
        side = pos.get("side", "YES")
        entry_price = float(pos.get("entry_price") or 0)

        current_price = pos.get("current_price")
        peak_price = float(pos.get("peak_price") or entry_price)

        def _executable_bid() -> float | None:
            raw = current_price
            return float(raw) * 0.98 if raw is not None else None

        def _make(cls, action, reason, urgency):
            return ExitAction(
                position_id=pid, contract_id=contract_id,
                event_id=event_id, city=city, date=date_str,
                side=side, classification=cls, action=action,
                reason=reason, exit_price=_executable_bid(),
                urgency=urgency,
            )

        if current_price is None:
            return _make("HEALTHY", "HOLD", "no_price_data", urgency=2)

        _price = float(current_price)

        # ---- 1. TAKE PROFIT ----
        if _price >= MPV_TAKE_PROFIT:
            return _make("TAKE_PROFIT", "SELL",
                         f"price={_price:.4f} >= TP={MPV_TAKE_PROFIT}",
                         urgency=0)

        # ---- 2. TOP BIN CONFIRMED ----
        if event_id in confirmed_events:
            confirmed_cid = confirmed_events[event_id]
            if contract_id != confirmed_cid and side == "YES":
                return _make("TOP_BIN_CONFIRMED", "SELL",
                             f"bin {confirmed_cid[:12]} confirmed at "
                             f">={MPV_CONFIRM_PRICE*100:.0f}%",
                             urgency=0)

        # ---- 3. TRAILING STOP ----
        if peak_price >= entry_price * (1 + MPV_TRAIL_ACTIVATION):
            trail_level = peak_price * (1 - MPV_TRAIL_PCT)
            if _price <= trail_level:
                return _make("TRAILING_STOP", "SELL",
                             f"price={_price:.4f} <= trail={trail_level:.4f} "
                             f"(peak={peak_price:.4f}, {MPV_TRAIL_PCT:.0%} trail)",
                             urgency=0)

        # ---- 4. DYING ----
        if _price < EXIT_DYING_PROB_THRESHOLD:
            return _make("DYING", "SELL",
                         f"price={_price:.4f} < threshold={EXIT_DYING_PROB_THRESHOLD}",
                         urgency=1)

        # ---- 5. HARD STOP (before trail activates) ----
        hard_stop_level = entry_price * (1 - MPV_HARD_STOP_PCT)
        if _price <= hard_stop_level:
            return _make("HARD_STOP", "SELL",
                         f"price={_price:.4f} <= hard_stop={hard_stop_level:.4f}",
                         urgency=0)

        # ---- 6. HEALTHY ----
        return _make("HEALTHY", "HOLD", "thesis_intact", urgency=2)


register("market_price_value", MarketPriceValueStrategy)
