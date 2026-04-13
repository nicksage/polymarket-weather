import logging
from datetime import datetime, date as _date, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from timezonefinder import TimezoneFinder as _TZF
    _tf = _TZF()
except Exception:
    _tf = None

from config import (
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_LIQUIDITY_USD,
    MIN_HOURS_TO_EXPIRY,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_FORECAST_SIGMA_C,
    TRADE_US_ONLY,
    US_LAT,
    US_LON,
    MAX_BIN_BUYS,
    ALLOWED_SIDES,
    MAX_TRADE_DAYS,
)
from db import get_open_positions, get_daily_pnl, count_open_bins_for_event

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


def check_time_to_expiry(
    resolution_date_str: str,
    lat: float | None = None,
    lon: float | None = None,
) -> RiskCheck:
    """
    Do not trade within MIN_HOURS_TO_EXPIRY hours of local end-of-day resolution.

    Polymarket temperature contracts resolve at end of the calendar day **in the
    city's local timezone** (e.g. Moscow April 12 resolves at 20:59 UTC, not
    23:59 UTC).  Using UTC midnight would allow trades within the window when
    the market has effectively already closed locally.

    Resolution time = 23:59:59 local time on the event date.
    Falls back to UTC midnight if lat/lon are unavailable or timezonefinder
    cannot determine a timezone for the location.
    """
    try:
        date_str = resolution_date_str[:10]   # keep only YYYY-MM-DD
        year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])

        # Resolve the local timezone from coordinates when possible
        tz = None
        if lat is not None and lon is not None and _tf is not None:
            try:
                tz_name = _tf.timezone_at(lat=lat, lng=lon)
                if tz_name:
                    tz = ZoneInfo(tz_name)
            except (ZoneInfoNotFoundError, Exception):
                pass

        if tz is None:
            tz = ZoneInfo("UTC")

        # End-of-day in local timezone
        resolution_dt = datetime(year, month, day, 23, 59, 59, tzinfo=tz)
        now_utc       = datetime.now(tz=ZoneInfo("UTC"))
        hours_remaining = (resolution_dt.astimezone(ZoneInfo("UTC")) - now_utc).total_seconds() / 3600

        if hours_remaining < MIN_HOURS_TO_EXPIRY:
            return RiskCheck(
                False,
                f"Only {hours_remaining:.1f}h to local expiry (min {MIN_HOURS_TO_EXPIRY}h)"
            )
        return RiskCheck(True)
    except (ValueError, TypeError) as e:
        logger.warning(f"check_time_to_expiry could not parse '{resolution_date_str}': {e}")
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


def check_allowed_side(side: str) -> RiskCheck:
    """
    Block trades on sides not permitted by ALLOWED_SIDES config.
    "yes"  → only YES trades allowed
    "no"   → only NO trades allowed
    "both" → no restriction
    """
    if ALLOWED_SIDES == "both":
        return RiskCheck(True)
    if ALLOWED_SIDES == "yes" and side != "YES":
        return RiskCheck(False, f"ALLOWED_SIDES=yes — blocking {side} trade")
    if ALLOWED_SIDES == "no" and side != "NO":
        return RiskCheck(False, f"ALLOWED_SIDES=no — blocking {side} trade")
    return RiskCheck(True)


def check_bin_limit(city: str, date: str) -> RiskCheck:
    """
    Block trade if the bot already holds MAX_BIN_BUYS open positions
    for the same city+date event.  MAX_BIN_BUYS=0 disables the check.
    """
    if MAX_BIN_BUYS <= 0:
        return RiskCheck(True)
    current = count_open_bins_for_event(city, date)
    if current >= MAX_BIN_BUYS:
        return RiskCheck(
            False,
            f"MAX_BIN_BUYS={MAX_BIN_BUYS} reached for {city} {date} "
            f"({current} open position(s))",
        )
    return RiskCheck(True)


def check_us_only(lat: float | None, lon: float | None) -> RiskCheck:
    """Block non-US trades when TRADE_US_ONLY is enabled."""
    if not TRADE_US_ONLY:
        return RiskCheck(True)
    if lat is None or lon is None:
        return RiskCheck(False, "TRADE_US_ONLY is set but lat/lon unavailable for this signal")
    if not (US_LAT[0] <= lat <= US_LAT[1] and US_LON[0] <= lon <= US_LON[1]):
        return RiskCheck(False, f"TRADE_US_ONLY: ({lat:.2f},{lon:.2f}) is outside US bounds")
    return RiskCheck(True)


def check_trade_days(resolution_date_str: str) -> RiskCheck:
    """
    Block trades on contracts that resolve more than MAX_TRADE_DAYS calendar
    days from today (UTC).  Prevents entering far-future contracts where model
    uncertainty is too high and edge estimates are unreliable.

    MAX_TRADE_DAYS=0 → only today's contracts.
    MAX_TRADE_DAYS=1 → today + tomorrow.
    """
    try:
        event_date = _date.fromisoformat(resolution_date_str[:10])
        today      = _date.today()
        days_away  = (event_date - today).days
        if days_away < 0:
            return RiskCheck(False, f"Contract date {event_date} is in the past")
        if days_away > MAX_TRADE_DAYS:
            return RiskCheck(
                False,
                f"Contract resolves in {days_away}d (MAX_TRADE_DAYS={MAX_TRADE_DAYS})",
            )
        return RiskCheck(True)
    except (ValueError, TypeError):
        return RiskCheck(False, "Could not parse resolution date for trade-day check")


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
        check_allowed_side(signal.get("recommended_side", "")),
        check_bin_limit(signal.get("city", ""), signal.get("date", "")),
        check_position_size(signal.get("kelly_size", 0), bankroll),
        check_total_exposure(signal.get("kelly_size", 0), bankroll),
        check_liquidity(signal.get("liquidity_usd", 0)),
        check_daily_drawdown(bankroll),
        check_normalization_warning(signal),
        check_us_only(signal.get("lat"), signal.get("lon")),
    ]

    # Forecast quality check — sigma is carried in the signal via event context
    # (edge.py attaches forecast_sigma_c if available via the event dict)
    sigma_c = signal.get("forecast_sigma_c")
    if sigma_c is not None:
        checks.append(check_forecast_sigma(sigma_c))

    # Time-to-expiry check — resolution is end-of-day in the city's local timezone
    resolution_date = signal.get("metadata", {}).get("date") or signal.get("date", "")
    if resolution_date:
        checks.append(check_trade_days(resolution_date))
        checks.append(check_time_to_expiry(
            resolution_date,
            lat=signal.get("lat"),
            lon=signal.get("lon"),
        ))

    for check in checks:
        if not check:
            failures.append(check.reason)
            logger.info(
                f"Risk FAIL [{signal.get('city', '')} {signal.get('date', '')} "
                f"{signal.get('question', '')[:30]}]: {check.reason}"
            )

    all_passed = len(failures) == 0
    return all_passed, failures
