"""
base.py — Abstract strategy base class + shared analysis utilities.

Every strategy inherits from Strategy and implements three methods:
  generate_signals  — decide which bins to trade
  rank_signals      — prioritize when capital is limited
  evaluate_positions — decide when to exit

The shared method analyze_event_base() handles everything strategy-agnostic:
weather fetching, live adjustment, probability computation, normalization.
It returns an enriched event dict with model_prob per bin but NO signal
decision — that's the strategy's job.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from typing import Any

from position_eval import ExitAction

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(
        self,
        events: list[dict],
        bankroll: float,
        scan_ts: str,
    ) -> tuple[list[dict], list[dict]]:
        """Analyze events and return (all_events_analyzed, signals).

        all_events_analyzed: full event-level data for DB + dashboard.
        signals: flat list of tradeable signal dicts ready for risk
                 checks and execution.
        """

    @abstractmethod
    def rank_signals(
        self,
        signals: list[dict],
        bankroll: float,
        client=None,
    ) -> list[dict]:
        """Score and sort signals by priority (highest first).

        Must attach 'priority_score' and 'priority_components' to each
        signal dict.

        `client` is an optional authenticated CLOB client.  Strategies
        that want orderbook-aware ranking can use it to pull
        spread/depth data; strategies that don't need it ignore the kwarg.
        Passing it via the signature instead of constructor injection
        keeps strategy initialization side-effect-free.
        """

    @abstractmethod
    def evaluate_positions(self) -> list[ExitAction]:
        """Classify all open positions and return non-HOLD exit actions."""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_event_base(event: dict, bankroll: float) -> dict | None:
        """Shared analysis: weather model + live adjustment + probabilities.

        Returns the same enriched event dict that edge.analyze_temp_event
        produces, with model_prob computed per bin.  Does NOT set is_signal,
        kelly_size, or recommended_side — the strategy does that.

        Returns None if the distribution cannot be computed (missing data,
        failed API calls, etc.)
        """
        from edge import analyze_temp_event
        return analyze_temp_event(event, bankroll)

    @staticmethod
    def _calculate_ev(model_prob: float, market_price: float, side: str) -> float:
        if side == "YES":
            return model_prob * (1 - market_price) - (1 - model_prob) * market_price
        else:
            p_no_model = 1 - model_prob
            p_no_market = 1 - market_price
            return p_no_model * (1 - p_no_market) - (1 - p_no_model) * p_no_market

    @staticmethod
    def _to_c(value: float | None, unit: str | None) -> float | None:
        if value is None:
            return None
        if unit == "fahrenheit":
            return (value - 32) * 5.0 / 9.0
        return float(value)

    @staticmethod
    def _hours_to_resolution(signal: dict) -> float:
        from sizing import _hours_to_resolution
        return _hours_to_resolution(signal)

    @staticmethod
    def flatten_signal(outcome: dict, analysis: dict, scan_ts: str) -> dict:
        """Build a flat signal dict from an enriched outcome + event analysis.

        This is the standard signal format expected by risk.py, execution.py,
        and the ranking functions.
        """
        return {
            **outcome,
            "city":           analysis.get("city"),
            "date":           analysis.get("date"),
            "lat":            analysis.get("lat"),
            "lon":            analysis.get("lon"),
            "event_id":       analysis.get("event_id"),
            "event_title":    analysis.get("event_title"),
            "scan_timestamp": scan_ts,
            "contract_id":    outcome.get("contract_id"),
            "market_p":       outcome.get("market_price"),
            "model_p":        outcome.get("model_prob"),
            "metadata": {
                "date":     analysis.get("date"),
                "lat":      analysis.get("lat"),
                "lon":      analysis.get("lon"),
                "variable": "temp_high",
            },
        }
