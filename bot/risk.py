import logging
from datetime import datetime
from config import (
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_LIQUIDITY_USD,
    MIN_HOURS_TO_EXPIRY,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_FORECAST_SIGMA_C,
)
from db import get_open_positions, get_daily_pnl

logger = logging.getLogger(__name__)


class RiskCheck:
    """Container for a single risk check result."""
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return f"RiskCheck(passed={self.passed}, reason='{self.reason}')"


def check_position_size(size_usdc: float, bankroll: float) -> RiskCheck:
    """Hard cap: no single position > MAX_POSITION_PCT of bankroll (default 2%)."""
    max_size = MAX_POSITION_PCT * bankroll
    if size_usdc > max_size:
        return RiskCheck(False, f"Position ${size_usdc:.2f} exceeds max ${max_size:.2f}")
    return RiskCheck(True)


def check_total_exposure(new_position_size: float, bankroll: float) -> RiskCheck:
    """Total open exposure must not exceed MAX_TOTAL_EXPOSURE_PCT of bankroll (default 20%)."""
    open_positions   = get_open_positions()
    current_exposure = sum(p.get("size_usdc", 0) for p in open_positions)
    new_total        = current_exposure + new_position_size

    if new_total > MAX_TOTAL_EXPOSURE_PCT * bankroll:
        return RiskCheck(
            False,
            f"Total exposure ${new_total:.2f} would exceed "
            f"{MAX_TOTAL_EXPOSURE_PCT*100:.0f}% of bankroll"
        )
    return RiskCheck(True)


def check_time_to_expiry(resolution_date_str: str) -> RiskCheck:
    """
    Do not trade within MIN_HOURS_TO_EXPIRY hours of resolution (default 6h).
    Liquidity dries up and spreads widen dramatically near expiry.
    """
    try:
        resolution_dt = datetime.fromisoformat(resolution_date_str.replace("Z", "+00:00"))
        hours_remaining = (
            resolution_dt - datetime.now(tz=resolution_dt.tzinfo)
        ).total_seconds() / 3600

        if hours_remaining < MIN_HOURS_TO_EXPIRY:
            return RiskCheck(
                False,
                f"Only {hours_remaining:.1f}h to expiry (min {MIN_HOURS_TO_EXPIRY}h)"
            )
        return RiskCheck(True)
    except (ValueError, TypeError):
        return RiskCheck(False, "Could not parse resolution date")


def check_liquidity(liquidity_usd: float) -> RiskCheck:
    """Skip contracts with less than MIN_LIQUIDITY_USD in the order book (default $500)."""
    if liquidity_usd < MIN_LIQUIDITY_USD:
        return RiskCheck(
            False,
            f"Liquidity ${liquidity_usd:.0f} below minimum ${MIN_LIQUIDITY_USD:.0f}"
        )
    return RiskCheck(True)


def check_forecast_sigma(sigma_c: float) -> RiskCheck:
    """
    Skip outcomes from events where the ensemble forecast sigma is implausibly
    large (> MAX_FORECAST_SIGMA_C, default 8°C) — indicates ensemble failure.
    A very wide σ makes the probability estimate unreliable for all bins.
    """
    if sigma_c > MAX_FORECAST_SIGMA_C:
        return RiskCheck(
            False,
            f"Forecast σ={sigma_c:.2f}°C > maximum ({MAX_FORECAST_SIGMA_C}°C) — "
            "ensemble too uncertain"
        )
    return RiskCheck(True)


def check_normalization_warning(signal: dict) -> RiskCheck:
    """
    If the raw model probability sum across all bins was far from 1.0
    (indicating a poorly-constrained distribution), Kelly sizing is already
    halved in edge.py.  Here we block trades where kelly_size is effectively 0.
    """
    if signal.get("kelly_size", 0) <= 0:
        return RiskCheck(False, "Kelly size is zero — no positive edge after adjustments")
    return RiskCheck(True)


def check_daily_drawdown(bankroll: float) -> RiskCheck:
    """
    Halt trading if today's realized P&L loss exceeds MAX_DAILY_DRAWDOWN_PCT
    of bankroll (default 10%).  Prevents runaway losses from model failure.
    """
    daily_pnl = get_daily_pnl()
    if daily_pnl < 0 and abs(daily_pnl) > MAX_DAILY_DRAWDOWN_PCT * bankroll:
        return RiskCheck(
            False,
            f"Daily drawdown ${abs(daily_pnl):.2f} exceeds "
            f"{MAX_DAILY_DRAWDOWN_PCT*100:.0f}% of bankroll "
            f"(${MAX_DAILY_DRAWDOWN_PCT*bankroll:.2f})"
        )
    return RiskCheck(True)


def run_all_checks(signal: dict, bankroll: float) -> tuple[bool, list[str]]:
    """
    Run all risk checks for a trade signal (individual temperature-range outcome).

    Returns (all_passed, list_of_failure_reasons).
    """
    failures: list[str] = []

    checks = [
        check_position_size(signal.get("kelly_size", 0), bankroll),
        check_total_exposure(signal.get("kelly_size", 0), bankroll),
        check_liquidity(signal.get("liquidity_usd", 0)),
        check_daily_drawdown(bankroll),
        check_normalization_warning(signal),
    ]

    # Forecast quality check — sigma is carried in the signal via event context
    # (edge.py attaches forecast_sigma_c if available via the event dict)
    sigma_c = signal.get("forecast_sigma_c")
    if sigma_c is not None:
        checks.append(check_forecast_sigma(sigma_c))

    # Time-to-expiry check uses the event date (resolution at end of that day)
    resolution_date = signal.get("metadata", {}).get("date") or signal.get("date", "")
    if resolution_date:
        # Treat resolution as end-of-day UTC for the event date
        checks.append(check_time_to_expiry(f"{resolution_date}T23:59:59Z"))

    for check in checks:
        if not check:
            failures.append(check.reason)
            logger.info(
                f"Risk FAIL [{signal.get('city', '')} {signal.get('date', '')} "
                f"{signal.get('question', '')[:30]}]: {check.reason}"
            )

    all_passed = len(failures) == 0
    return all_passed, failures
