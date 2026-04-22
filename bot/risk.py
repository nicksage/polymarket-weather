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
    MAX_EVENT_EXPOSURE_PCT,
    MAX_CITY_DAY_EXPOSURE,
    MIN_LIQUIDITY_USD,
    MIN_HOURS_TO_EXPIRY,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_FORECAST_SIGMA_C,
    TRADE_US_ONLY,
    US_LAT,
    US_LON,
    MAX_YES_BINS,
    MAX_NO_BINS,
    ALLOWED_SIDES,
    MAX_TRADE_DAYS,
    PAPER_TRADE,
)
from db import (
    get_open_positions, get_daily_pnl, count_open_bins_for_event,
    get_latest_observation,
)

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
    """
    Total open exposure must not exceed MAX_TOTAL_EXPOSURE_PCT of bankroll.
    Paper and live positions are tracked separately — in paper mode only
    paper positions count; in live mode only live positions count.  This
    matches the dashboard's per-mode Capital Deployed metrics.
    """
    open_positions = get_open_positions()
    is_paper_int   = 1 if PAPER_TRADE else 0
    relevant = [
        p for p in open_positions
        if p.get("is_paper", 1) == is_paper_int
        and p.get("fill_status") != "cancelled"
    ]
    current_exposure = sum(p.get("size_usdc", 0) for p in relevant)
    new_total        = current_exposure + new_position_size
    cap              = MAX_TOTAL_EXPOSURE_PCT * bankroll

    if new_total > cap:
        mode = "paper" if PAPER_TRADE else "live"
        return RiskCheck(
            False,
            f"Total {mode} exposure ${new_total:,.2f} would exceed "
            f"{MAX_TOTAL_EXPOSURE_PCT*100:.0f}% cap (${cap:,.2f}); "
            f"current={current_exposure:,.2f} over {len(relevant)} position(s)"
        )
    return RiskCheck(True)


def check_event_exposure(
    event_id: str, new_position_size: float, bankroll: float,
) -> RiskCheck:
    """Total capital across all bins in one event must not exceed
    MAX_EVENT_EXPOSURE_PCT of bankroll."""
    if MAX_EVENT_EXPOSURE_PCT >= 1.0:
        return RiskCheck(True)
    open_positions = get_open_positions()
    is_paper_int = 1 if PAPER_TRADE else 0
    current = sum(
        p.get("size_usdc", 0) for p in open_positions
        if p.get("event_id") == event_id
        and p.get("is_paper", 1) == is_paper_int
        and p.get("fill_status") != "cancelled"
    )
    new_total = current + new_position_size
    cap = MAX_EVENT_EXPOSURE_PCT * bankroll
    if new_total > cap:
        return RiskCheck(
            False,
            f"Event exposure ${new_total:,.2f} would exceed "
            f"{MAX_EVENT_EXPOSURE_PCT*100:.0f}% cap (${cap:,.2f}) for {event_id[:16]}"
        )
    return RiskCheck(True)


def check_city_day_exposure(
    city: str, date_str: str, new_position_size: float, bankroll: float,
) -> RiskCheck:
    """Total capital across all events for one (city, date) must not exceed
    MAX_CITY_DAY_EXPOSURE of bankroll."""
    if MAX_CITY_DAY_EXPOSURE >= 1.0:
        return RiskCheck(True)
    open_positions = get_open_positions()
    is_paper_int = 1 if PAPER_TRADE else 0
    current = sum(
        p.get("size_usdc", 0) for p in open_positions
        if p.get("city") == city and p.get("date") == date_str
        and p.get("is_paper", 1) == is_paper_int
        and p.get("fill_status") != "cancelled"
    )
    new_total = current + new_position_size
    cap = MAX_CITY_DAY_EXPOSURE * bankroll
    if new_total > cap:
        return RiskCheck(
            False,
            f"City-day exposure ${new_total:,.2f} would exceed "
            f"{MAX_CITY_DAY_EXPOSURE*100:.0f}% cap (${cap:,.2f}) for {city} {date_str}"
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
            f"Forecast sd={sigma_c:.2f}°C > maximum ({MAX_FORECAST_SIGMA_C}°C) - "
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


def check_event_side_consistency(
    event_id: str, city: str, date_str: str, side: str,
) -> RiskCheck:
    """All positions within a single event must be on the same side.
    If we already hold YES positions for this event, block NO entries
    (and vice versa).  Prevents contradictory positions within the
    same city+date event."""
    open_positions = get_open_positions()
    is_paper_int = 1 if PAPER_TRADE else 0
    existing = [
        p for p in open_positions
        if (p.get("event_id") == event_id or
            (p.get("city") == city and p.get("date") == date_str))
        and p.get("is_paper", 1) == is_paper_int
        and p.get("fill_status") != "cancelled"
    ]
    if not existing:
        return RiskCheck(True)

    existing_side = existing[0].get("side")
    if existing_side and existing_side != side:
        return RiskCheck(
            False,
            f"Side conflict: already hold {existing_side} for {city} {date_str} "
            f"— cannot enter {side} on same event"
        )
    return RiskCheck(True)


def check_bin_limit(city: str, date: str, side: str = "YES") -> RiskCheck:
    """
    Block trade if the bot already holds the max number of open positions
    for this side on the same city+date event.
    MAX_YES_BINS / MAX_NO_BINS = 0 disables the check for that side.
    """
    if side == "YES":
        cap = MAX_YES_BINS
        label = "MAX_YES_BINS"
    else:
        cap = MAX_NO_BINS
        label = "MAX_NO_BINS"

    if cap <= 0:
        return RiskCheck(True)
    current = count_open_bins_for_event(city, date, side=side)
    if current >= cap:
        return RiskCheck(
            False,
            f"{label}={cap} reached for {city} {date} "
            f"({current} open {side} position(s))",
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


def check_same_day_live_data(
    event_id: str, days_ahead: int | None,
) -> RiskCheck:
    """Block same-day trades when the live adjustment layer has no
    observation data yet.  Without observations, the observed-max floor
    and sigma shrinkage are inactive, meaning the model trades on a
    less-informed probability distribution.  Future-day trades (D+1+)
    are unaffected — the live layer contributes little for those anyway."""
    if days_ahead is None or days_ahead >= 1:
        return RiskCheck(True)
    latest_obs = get_latest_observation(event_id or "")
    if latest_obs is None:
        return RiskCheck(
            False,
            f"Same-day trade blocked: no live observations for {event_id[:16]} — "
            f"live adjustment layer has no data yet"
        )
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


def run_pre_checks(signal: dict, bankroll: float) -> tuple[bool, list[str]]:
    """
    Signal-level risk checks that are INDEPENDENT of portfolio state.
    These filter out signals that can never execute regardless of capital
    availability: wrong side, expired, illiquid, no live data, etc.

    Run BEFORE ranking so only tradeable signals get scored.
    """
    failures: list[str] = []

    checks = [
        check_allowed_side(signal.get("recommended_side", "")),
        check_event_side_consistency(
            signal.get("event_id", ""), signal.get("city", ""),
            signal.get("date", ""), signal.get("recommended_side", "")),
        check_position_size(signal.get("kelly_size", 0), bankroll),
        check_liquidity(signal.get("liquidity_usd", 0)),
        check_daily_drawdown(bankroll),
        check_normalization_warning(signal),
        check_us_only(signal.get("lat"), signal.get("lon")),
    ]

    sigma_c = signal.get("forecast_sigma_c")
    if sigma_c is not None:
        checks.append(check_forecast_sigma(sigma_c))

    resolution_date = signal.get("metadata", {}).get("date") or signal.get("date", "")
    if resolution_date:
        checks.append(check_trade_days(resolution_date))
        checks.append(check_time_to_expiry(
            resolution_date,
            lat=signal.get("lat"),
            lon=signal.get("lon"),
        ))

    days_ahead = signal.get("days_ahead")
    if days_ahead is None and resolution_date:
        try:
            from datetime import date as _d
            days_ahead = (_d.fromisoformat(resolution_date[:10]) - _d.today()).days
        except Exception:
            pass
    checks.append(check_same_day_live_data(
        signal.get("event_id", ""), days_ahead,
    ))

    for check in checks:
        if not check:
            failures.append(check.reason)

    return (len(failures) == 0, failures)


def run_portfolio_checks(signal: dict, bankroll: float) -> tuple[bool, list[str]]:
    """
    Portfolio-level risk checks that DEPEND on current capital state.
    These are evaluated in priority order (after ranking) so that the
    highest-scored signals get funded first.

    Checks: total exposure, event exposure, city-day exposure, bin limit.
    """
    failures: list[str] = []

    checks = [
        check_bin_limit(signal.get("city", ""), signal.get("date", ""),
                        signal.get("recommended_side", "YES")),
        check_total_exposure(signal.get("kelly_size", 0), bankroll),
        check_event_exposure(
            signal.get("event_id", ""), signal.get("kelly_size", 0), bankroll),
        check_city_day_exposure(
            signal.get("city", ""), signal.get("date", ""),
            signal.get("kelly_size", 0), bankroll),
    ]

    for check in checks:
        if not check:
            failures.append(check.reason)

    return (len(failures) == 0, failures)


def run_all_checks(signal: dict, bankroll: float) -> tuple[bool, list[str]]:
    """
    Run all risk checks for a trade signal (individual temperature-range outcome).
    Combines pre-checks and portfolio checks.  Retained for backward
    compatibility with any callers that use the combined interface.

    Returns (all_passed, list_of_failure_reasons).
    """
    passed1, failures1 = run_pre_checks(signal, bankroll)
    passed2, failures2 = run_portfolio_checks(signal, bankroll)
    failures = failures1 + failures2

    return (len(failures) == 0, failures)
