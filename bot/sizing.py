import os
import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import KELLY_FRACTION, MAX_POSITION_PCT, CONFIDENCE_AGREEMENT_THRESHOLD

logger = logging.getLogger(__name__)


def compute_confidence_multiplier(signal: dict, event_state: dict | None = None) -> float:
    """Sign-aware confidence multiplier for Kelly fraction.

    Returns a float in [0.1, 1.5] that scales the Kelly fraction based on
    how much the model's ancillary signals support or contradict the trade.

    Inputs come from signal dict (edge.py output) and event_state (from
    temp_events current-state columns).
    """
    confidence = 1.0
    state = event_state or {}

    # --- Forecast agreement: penalize when ECMWF/GFS disagree ---
    agreement = state.get("forecast_agreement_c")
    if agreement is not None and agreement > CONFIDENCE_AGREEMENT_THRESHOLD:
        confidence *= 0.7

    # --- Live adjustment: sign-aware ---
    adj_mu   = state.get("adjusted_mu_c")
    blend_mu = state.get("forecast_mu_c")
    score    = state.get("live_adjustment_score") or 0.0

    if adj_mu is not None and blend_mu is not None and score > 1.0:
        adj_direction = adj_mu - blend_mu  # positive = adjustment hotter

        range_low  = signal.get("range_low")
        range_high = signal.get("range_high")
        side       = signal.get("recommended_side")
        unit       = signal.get("unit", "celsius")

        if range_low is not None and range_high is not None:
            if unit == "fahrenheit":
                bin_center_c = ((range_low + range_high) / 2.0 - 32) * 5.0 / 9.0
                blend_mu_c = blend_mu
            else:
                bin_center_c = (range_low + range_high) / 2.0
                blend_mu_c = blend_mu

            bin_above_mu = bin_center_c > blend_mu_c

            if side == "YES":
                adj_supports = (adj_direction > 0 and bin_above_mu) or \
                               (adj_direction < 0 and not bin_above_mu)
            else:
                adj_supports = (adj_direction > 0 and not bin_above_mu) or \
                               (adj_direction < 0 and bin_above_mu)

            if adj_supports:
                confidence *= 1.1
            else:
                confidence *= 0.7

    # --- Days ahead ---
    days_ahead = signal.get("days_ahead")
    if days_ahead is not None and days_ahead >= 3:
        confidence *= 0.6

    # --- Normalization warning ---
    if signal.get("normalization_warning"):
        confidence *= 0.5

    return max(0.1, min(confidence, 1.5))

'''
#########################################
Section 6: Kelly Criterion Position Sizing


The Kelly Criterion is a formula for optimal bet sizing that maximizes long-term bankroll growth.
For a binary outcome with edge e and decimal odds b:

Kelly fraction = (b*p - q) / b
               = (b*p - (1-p)) / b

Where:
q = 1 - p
p = your probability of winning
b = net odds (payout per dollar risked — for a $0.60 YES share, b = 0.40/0.60 = 0.667)

Rule of thumb: use 1/4 Kelly as default. If your model has been running for 3+ months with a
documented Brier score, consider moving to 1/2 Kelly.

Full Kelly is aggressive. Model uncertainty means your p estimate could be wrong. Using 1/4
Kelly (betting 25% of the theoretical optimal) dramatically reduces variance while preserving
most of the long-term growth.
#########################################
'''


def compute_time_scale(days_ahead: int | None, local_hour: float | None) -> float:
    """Two-regime time-to-expiry scaling.

    Forecast-driven regime (days_ahead >= 1):
        Linear ramp: full size at 24h+, down to 0.25 near expiry.

    Observation-driven regime (days_ahead == 0, same-day):
        Midday (10-16 local) = full size (best data).
        Morning (<10) = 0.5 (sparse obs, forecast-like).
        Late afternoon (16-19) = 0.75 (good data, less runway).
        Evening (19+) = 0 (blocked by MIN_HOURS_TO_EXPIRY anyway).

    Returns a multiplier in [0.0, 1.0].
    """
    if days_ahead is None:
        return 1.0

    if days_ahead >= 1:
        # Forecast-driven: no penalty for >24h; gentle ramp for closer
        if days_ahead >= 2:
            return 1.0
        # days_ahead == 1 → tomorrow; still forecast-driven, slight discount
        return 0.85

    # Same-day: observation-driven regime
    if local_hour is None:
        return 0.5
    if local_hour < 10:
        return 0.5
    if local_hour < 16:
        return 1.0
    if local_hour < 19:
        return 0.75
    return 0.0


def _hours_to_resolution(signal: dict) -> float:
    """Estimate hours until this contract resolves (end of local day)."""
    date_str = signal.get("date") or signal.get("metadata", {}).get("date")
    if not date_str:
        return 48.0   # conservative default
    try:
        target = date.fromisoformat(date_str[:10])
    except ValueError:
        return 48.0

    # Resolution = 23:59 local time on the event date.
    # Try to resolve the local timezone from lat/lon.
    lat, lon = signal.get("lat"), signal.get("lon")
    tz = None
    if lat is not None and lon is not None:
        try:
            from timezonefinder import TimezoneFinder
            tf = TimezoneFinder()
            tz_name = tf.timezone_at(lat=lat, lng=lon)
            if tz_name:
                tz = ZoneInfo(tz_name)
        except Exception:
            pass
    if tz is None:
        tz = ZoneInfo("UTC")

    resolution_dt = datetime(target.year, target.month, target.day,
                             23, 59, 59, tzinfo=tz)
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    hours = (resolution_dt.astimezone(ZoneInfo("UTC")) - now_utc).total_seconds() / 3600.0
    return max(hours, 1.0)


def rank_signals(signals: list[dict], bankroll: float) -> list[dict]:
    """Score and rank all signals by capital-efficiency-weighted priority.

    Composite score:
        priority = ev_per_dollar * confidence * time_efficiency * confirmation_bonus

    Sorts descending.  Attaches 'priority_score' and 'priority_components'
    to each signal dict for logging and snapshot storage.  Does NOT filter —
    returns all signals; the caller funds top-to-bottom until caps bind.
    """
    if not signals:
        return signals

    for s in signals:
        # --- ev_per_dollar: return on capital ---
        ev = float(s.get("ev") or 0)
        entry_price = float(s.get("market_p") or s.get("market_price") or 0.5)
        kelly_size = float(s.get("kelly_size") or 1.0)
        capital_required = max(kelly_size, 1.0)
        ev_per_dollar = abs(ev) / max(entry_price, 0.01)

        # --- confidence: already computed upstream, passed through signal ---
        confidence = float(s.get("confidence_multiplier") or 1.0)

        # --- time_efficiency: EV per hour of capital lockup ---
        hours = _hours_to_resolution(s)
        time_efficiency = 1.0 / max(hours, 1.0)

        # --- confirmation_bonus: same-day with live-adjustment support ---
        days_ahead = s.get("days_ahead")
        if days_ahead is None:
            try:
                d = s.get("date") or s.get("metadata", {}).get("date", "")
                days_ahead = (date.fromisoformat(d[:10]) - date.today()).days
            except Exception:
                days_ahead = None

        confirmation = 1.0
        if days_ahead is not None and days_ahead == 0:
            live_score = float(s.get("live_adjustment_score") or 0)
            if live_score > 0.1:
                # Check directionality: did the adjustment move toward our bin?
                adj_mu = s.get("adjusted_mu_c")
                blend_mu = s.get("forecast_mu_c")
                side = s.get("recommended_side")
                range_low = s.get("range_low")
                range_high = s.get("range_high")
                if (adj_mu is not None and blend_mu is not None
                        and range_low is not None and range_high is not None):
                    unit = s.get("unit", "celsius")
                    if unit == "fahrenheit":
                        bin_center = ((range_low + range_high) / 2.0 - 32) * 5.0 / 9.0
                    else:
                        bin_center = (range_low + range_high) / 2.0
                    adj_toward_bin = (
                        (adj_mu > blend_mu and bin_center > blend_mu) or
                        (adj_mu < blend_mu and bin_center < blend_mu)
                    )
                    if side == "NO":
                        adj_toward_bin = not adj_toward_bin
                    if adj_toward_bin:
                        confirmation = 1.2

        # --- composite score ---
        score = ev_per_dollar * confidence * time_efficiency * confirmation

        s["priority_score"] = round(score, 6)
        s["priority_components"] = {
            "ev_per_dollar":    round(ev_per_dollar, 4),
            "confidence":       round(confidence, 3),
            "time_efficiency":  round(time_efficiency, 5),
            "hours_to_resolve": round(hours, 1),
            "confirmation":     round(confirmation, 2),
        }

    signals.sort(key=lambda s: s.get("priority_score", 0), reverse=True)
    return signals


def calculate_kelly_size(edge: float, odds: float, bankroll: float) -> float:
    """
    Calculate recommended position size using fractional Kelly.

        Args:
        edge: |p_model - p_market| — the probability edge
        odds: decimal odds for the side being bet (payout per dollar risked)
              For YES at price 0.60: odds = (1 - 0.60) / 0.60 = 0.667
        bankroll: total USDC available

        Returns:
        Position size in USDC (capped at MAX_POSITION_PCT of bankroll)

    Example:
        edge=0.10, odds=0.667, bankroll=1000, kelly_fraction=0.25
        raw_kelly = (0.667*0.60 - 0.40) / 0.667 = (0.40 - 0.40) / ...

        Correct calculation using p directly:
        p = p_market + edge (for YES)
    
        kelly = (b*p - (1-p)) / b
    """
    if edge <= 0 or odds <= 0 or bankroll <= 0:
        return 0.0

    # Convert edge + market odds back to our win probability p
    # For YES: p_model = p_market + edge, where p_market = 1/(1+odds)
    p_market = 1.0 / (1.0 + odds)
    p_model = p_market + edge
    p_model = min(max(p_model, 0.0), 1.0)  # Clamp

    b = odds  # Net payout per dollar risked
    q = 1 - p_model

    # Full Kelly formula
    if b == 0:
        return 0.0
    
    full_kelly_fraction = (b * p_model - q) / b
        
    if full_kelly_fraction <= 0:    
        logger.debug(f"Kelly fraction negative ({full_kelly_fraction:.4f}) — no bet")
        return 0.0

    # Apply fractional Kelly (default 0.25)
    fraction = float(os.getenv("KELLY_FRACTION", KELLY_FRACTION))
    fractional_kelly = full_kelly_fraction * fraction

    # Calculate raw position size
    raw_size = fractional_kelly * bankroll

    # Cap at MAX_POSITION_PCT of bankroll (hard risk limit)
    max_size = MAX_POSITION_PCT * bankroll
    position_size = min(raw_size, max_size)

    logger.debug(
        f"Kelly sizing: full_kelly={full_kelly_fraction:.4f} "
        f"fractional={fractional_kelly:.4f} "
        f"raw=${raw_size:.2f} capped=${position_size:.2f}"
    )

    # Round to nearest dollar, minimum $1
    return max(1.0, round(position_size, 2))


def get_bankroll(db_path: str | None = None) -> float:
    """
    Get current available bankroll from config or database.
    In production, this should query your actual USDC balance from
    the Polymarket CLOB API or an on-chain balance check.
    For now, loads from environment variable.
    """
    from config import INITIAL_BANKROLL
    return float(os.getenv("BANKROLL_USDC", INITIAL_BANKROLL))