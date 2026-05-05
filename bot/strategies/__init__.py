"""
strategies — Pluggable trading strategy framework.

Each strategy implements signal generation, ranking, and position evaluation.
The active strategy is selected via ACTIVE_STRATEGY in .env.

Usage:
    from strategies import get_active_strategy
    strategy = get_active_strategy()
    all_events, signals = strategy.generate_signals(events, bankroll, scan_ts)
    ranked = strategy.rank_signals(signals, bankroll)
    exits = strategy.evaluate_positions()
"""

from strategies.base import Strategy

_REGISTRY: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def get_active_strategy() -> Strategy:
    from config import ACTIVE_STRATEGY
    name = ACTIVE_STRATEGY.strip().lower()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {available}. "
            f"Set ACTIVE_STRATEGY in .env to one of these."
        )
    return _REGISTRY[name]()


# Import strategy modules to trigger registration
from strategies import top_bin_value        # noqa: F401, E402
from strategies import market_price_value   # noqa: F401, E402
from strategies import top_k_hedged         # noqa: F401, E402
