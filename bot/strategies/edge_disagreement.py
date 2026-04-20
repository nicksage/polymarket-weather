"""
edge_disagreement.py — Original trading strategy (extracted).

Signal rule: buy any bin where |model_prob - market_price| >= EDGE_THRESHOLD.
This is the exact logic that was hardcoded in edge.py + sizing.py +
position_eval.py before the strategy framework was introduced.

Zero behavior change from the pre-framework version.
"""

from __future__ import annotations

import logging

from strategies.base import Strategy
from strategies import register
from position_eval import ExitAction, evaluate_open_positions

logger = logging.getLogger(__name__)


class EdgeDisagreementStrategy(Strategy):
    name = "edge_disagreement"

    def generate_signals(
        self,
        events: list[dict],
        bankroll: float,
        scan_ts: str,
    ) -> tuple[list[dict], list[dict]]:
        """Delegate to the existing run_edge_scan which already implements
        the full edge-disagreement pipeline including bracket promotion."""
        from edge import run_edge_scan
        return run_edge_scan(bankroll=bankroll, events=events)

    def rank_signals(
        self,
        signals: list[dict],
        bankroll: float,
    ) -> list[dict]:
        """Delegate to the existing rank_signals in sizing.py."""
        from sizing import rank_signals
        return rank_signals(signals, bankroll)

    def evaluate_positions(self) -> list[ExitAction]:
        """Delegate to the existing position evaluation logic."""
        return evaluate_open_positions()


register("edge_disagreement", EdgeDisagreementStrategy)
