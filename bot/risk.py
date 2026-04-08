import logging
from datetime import datetime, timedelta

from config import (
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_LIQUIDITY_USD,
    MIN_HOURS_TO_EXPIRY,
    MAX_SOURCE_DISAGREEMENT,
    MAX_DAILY_DRAWDOWN_PCT,
)
from db import get_open_positions, get_daily_pnl

logger = logging.getLogger(__name__)

class RiskCheck:

    """Container for a risk check result."""
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return f"RiskCheck(passed={self.passed}, reason='{self.reason}')"

def check_position_size(size_usdc: float, bankroll: float) -> RiskCheck:
    """
    Hard cap: no single position > MAX_POSITION_PCT of bankroll.
    Default: 2% of bankroll.
    """
    max_size = MAX_POSITION_PCT * bankroll
    if size_usdc > max_size:
        return RiskCheck(False, f"Position ${size_usdc:.2f} exceeds max ${max_size:.2f}")
    return RiskCheck(True)

def check_total_exposure(new_position_size: float, bankroll: float) -> RiskCheck:
    """
    Total open exposure across all weather contracts must not exceed
    MAX_TOTAL_EXPOSURE_PCT of bankroll. Default: 20%.
    """
    open_positions = get_open_positions()
    current_exposure = sum(p.get("size_usdc", 0) for p in open_positions)

    if current_exposure + new_position_size > MAX_TOTAL_EXPOSURE_PCT * bankroll:
        return RiskCheck(
            False,
            f"Total exposure ${current_exposure + new_position_size:.2f} "
            f"would exceed {MAX_TOTAL_EXPOSURE_PCT*100:.0f}% of bankroll"
        )
    return RiskCheck(True)

def check_time_to_expiry(resolution_date_str: str) -> RiskCheck:
    """
    Do not trade within MIN_HOURS_TO_EXPIRY hours of resolution.
    Default: 6 hours. Reason: liquidity dries up, spreads widen dramatically.
    """
    try:
        resolution_dt = datetime.fromisoformat(resolution_date_str.replace("Z", "+00:00"))
        hours_remaining = (resolution_dt - datetime.now(tz=resolution_dt.tzinfo)).total_seconds() / 3600

        if hours_remaining < MIN_HOURS_TO_EXPIRY:

            return RiskCheck(
                False,
                f"Only {hours_remaining:.1f}h to expiry (min {MIN_HOURS_TO_EXPIRY}h)"
            )
        return RiskCheck(True)
    except (ValueError, TypeError):
        # If we can't parse the date, skip the contract
        return RiskCheck(False, "Could not parse resolution date")

def check_liquidity(liquidity_usd: float) -> RiskCheck:
    """
    Skip contracts with less than MIN_LIQUIDITY_USD in the order book.
    Default: $500. Thin books mean large slippage and manipulation risk.
    """
    if liquidity_usd < MIN_LIQUIDITY_USD:
        return RiskCheck(
            False,
            f"Liquidity ${liquidity_usd:.0f} below minimum ${MIN_LIQUIDITY_USD:.0f}"
        )
    return RiskCheck(True)

def check_model_disagreement(disagreement: float) -> RiskCheck:
    """
    Skip if weather model sources disagree by > MAX_SOURCE_DISAGREEMENT.
    Default: 0.15 (15 percentage points).
    High disagreement means the forecast is genuinely uncertain — not edge.
    """
    if disagreement > MAX_SOURCE_DISAGREEMENT:
        return RiskCheck(
            False,
            f"Model disagreement {disagreement:.3f} > {MAX_SOURCE_DISAGREEMENT:.3f}"
        )
    return RiskCheck(True)

def check_daily_drawdown(bankroll: float) -> RiskCheck:
    """
    Stop trading if today's P&L loss exceeds MAX_DAILY_DRAWDOWN_PCT of bankroll.
    Default: 10%. Prevents runaway losses from model failure.
    """
    daily_pnl = get_daily_pnl()

    if daily_pnl < 0 and abs(daily_pnl) > MAX_DAILY_DRAWDOWN_PCT * bankroll:
        return RiskCheck(
            False,
            f"Daily drawdown ${abs(daily_pnl):.2f} exceeds "
            f"{MAX_DAILY_DRAWDOWN_PCT*100:.0f}% of bankroll (${MAX_DAILY_DRAWDOWN_PCT*bankroll:.2f})"
        )
    return RiskCheck(True)

def run_all_checks(signal: dict, bankroll: float) -> tuple[bool, list[str]]:
    """
    Run all risk checks for a trade signal.

    Returns (all_passed, list_of_failures).
    Log failures but do not raise — caller decides whether to skip.
    """
    failures = []

    checks = [
        check_position_size(signal.get("kelly_size", 0), bankroll),
        check_total_exposure(signal.get("kelly_size", 0), bankroll),
        check_liquidity(signal.get("liquidity_usd", 0)),
        check_model_disagreement(signal.get("disagreement", 0)),
        check_daily_drawdown(bankroll),
    ]

    # Time to expiry check needs resolution date
    resolution_date = signal.get("metadata", {}).get("date") or signal.get("resolution_date", "")

    if resolution_date:
        checks.append(check_time_to_expiry(resolution_date))

    for check in checks:
        if not check:
            failures.append(check.reason)
            logger.info(f"Risk check FAILED for {signal.get('contract_id', 'unknown')}: {check.
              reason}")

    all_passed = len(failures) == 0
    if all_passed:
        logger.debug(f"All risk checks passed for {signal.get('contract_id', 'unknown')}")

    return all_passed, failures