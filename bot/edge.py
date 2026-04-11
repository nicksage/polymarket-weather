"""
edge.py — Signal generation for highest-temperature Polymarket contracts.

Operates at the EVENT level: each Polymarket "Highest temperature in X on Y"
event contains multiple mutually-exclusive outcome ranges.  For each event we:
    1. Fetch the temperature distribution N(μ, σ) from weather.py
    2. Compute model_prob[i] = P(T_max in range_i) for every outcome bin
    3. Normalize so model probs sum to 1.0
    4. Compare each model_prob[i] to the market's yes_price[i]
    5. Flag outcomes where |edge| > EDGE_THRESHOLD as signals

run_edge_scan() returns BOTH:
    - all_events:  full analysis for every discovered event (dashboard Tab 1)
    - signals:     only outcome-level dicts with is_signal=True (execution + Tab 2)

Signal dict schema (one per outcome):
    contract_id, question, range_low, range_high, unit,
    market_price, model_prob, raw_model_prob, ev, edge,
    recommended_side, kelly_size, is_signal,
    yes_token_id, no_token_id, liquidity_usd, volume_usd,
    city, date, lat, lon, event_id, event_title, scan_timestamp
"""

import logging
from datetime import datetime, date
from config import EDGE_THRESHOLD, MIN_LIQUIDITY_USD, MAX_FORECAST_DAYS, NORM_WARNING_LOW, NORM_WARNING_HIGH
from weather import (
    get_temp_distribution_for_event,
    get_temp_range_probability,
    clear_forecast_cache,
    reset_tomorrowio_limit,
)
from polymarket import search_temp_high_events
from db import (
    insert_temp_event, insert_temp_outcome,
    insert_signal,
)
from sizing import calculate_kelly_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EV calculation
# ---------------------------------------------------------------------------

def _calculate_ev(model_prob: float, market_price: float, side: str) -> float:
    """
    Expected value per dollar wagered on the given side.

    YES: EV = model_prob * (1 - market_price) - (1 - model_prob) * market_price
    NO:  EV = (1 - model_prob) * market_price - model_prob * (1 - market_price)
    """
    if side == "YES":
        return model_prob * (1 - market_price) - (1 - model_prob) * market_price
    else:
        p_no_model  = 1 - model_prob
        p_no_market = 1 - market_price
        return p_no_model * (1 - p_no_market) - (1 - p_no_model) * p_no_market


# ---------------------------------------------------------------------------
# Event-level analysis
# ---------------------------------------------------------------------------

def analyze_temp_event(event: dict, bankroll: float) -> dict | None:
    """
    Full analysis pipeline for a single highest-temperature event.

    Returns an event-level analysis dict containing a list of outcome dicts,
    or None if the forecast distribution could not be obtained.

    The returned dict has the same top-level keys as event, plus:
        forecast_mu_c, forecast_sigma_c, clim_mu_c, clim_sigma_c,
        days_ahead, model_probs_sum, normalization_warning,
        market_overround, n_sources, forecast_mu_display, display_unit
    Each outcome in event["outcomes"] gets additional fields:
        model_prob, raw_model_prob, ev, edge, recommended_side,
        kelly_size, is_signal, market_price (alias for yes_price)
    """
    city       = event.get("city", "")
    date_str   = event.get("date")
    lat        = event.get("lat")
    lon        = event.get("lon")
    display_unit = event.get("display_unit", "celsius")

    if not date_str or lat is None or lon is None:
        return None

    # Guard: skip events outside the reliable forecast window
    try:
        days_ahead = (date.fromisoformat(date_str) - date.today()).days
    except ValueError:
        return None

    if days_ahead < 0:
        logger.debug(f"Skipping past event: {city} {date_str}")
        return None

    if days_ahead > MAX_FORECAST_DAYS:
        logger.info(f"Skipping {city} {date_str}: {days_ahead}d > MAX_FORECAST_DAYS={MAX_FORECAST_DAYS}")
        return None

    # Fetch temperature distribution
    dist = get_temp_distribution_for_event(lat, lon, date_str)
    if dist is None:
        logger.warning(f"No distribution for {city} {date_str}")
        return None

    mu_c     = dist["mu_c"]
    sigma_c  = dist["sigma_c"]
    clim_kde = dist.get("clim_kde")   # KDE object or None

    # Display-unit mu for dashboard
    if display_unit == "fahrenheit":
        mu_display = mu_c * 9 / 5 + 32
    else:
        mu_display = mu_c

    outcomes   = event.get("outcomes", [])
    n_outcomes = len(outcomes)

    # Step 1: compute raw model probabilities for each bin.
    # Outcomes where range_low and range_high are both None (unparseable range)
    # get raw_prob=None and are excluded from normalization but still included
    # in the output so the dashboard can display all discovered contracts.
    raw_probs: list[float | None] = []
    for o in outcomes:
        lo = o.get("range_low")
        hi = o.get("range_high")
        if lo is None and hi is None:
            raw_probs.append(None)  # unparseable — no model probability
        else:
            p = get_temp_range_probability(
                mu_c, sigma_c, lo, hi, o.get("unit", "celsius"),
                kde=clim_kde,
            )
            raw_probs.append(p)

    parseable_probs = [p for p in raw_probs if p is not None]
    raw_sum = sum(parseable_probs)

    # Step 2: normalize parseable probs so they sum to 1.0 (among themselves).
    norm_warning = bool(parseable_probs) and (
        raw_sum < NORM_WARNING_LOW or raw_sum > NORM_WARNING_HIGH
    )
    if norm_warning:
        logger.warning(
            f"Normalization warning for {city} {date_str}: raw_sum={raw_sum:.3f}"
        )

    if raw_sum > 0:
        # Normalize only the parseable probs; keep None for unparseable
        model_probs: list[float | None] = [
            (p / raw_sum) if p is not None else None for p in raw_probs
        ]
    elif parseable_probs:
        # raw_sum is 0 despite having parseable probs — uniform fallback
        n_parseable = len(parseable_probs)
        model_probs = [
            (1.0 / n_parseable) if p is not None else None for p in raw_probs
        ]
    else:
        # No parseable ranges at all — all outcomes get None
        model_probs = [None] * n_outcomes

    # Step 3: market overround
    market_prices  = [o.get("yes_price", 0.5) for o in outcomes]
    market_sum = sum(market_prices)

    # Step 4: build enriched outcome list
    enriched_outcomes = []
    for i, o in enumerate(outcomes):
        market_price = market_prices[i]
        model_prob   = model_probs[i]   # may be None for unparseable ranges
        raw_prob     = raw_probs[i]     # may be None

        # Outcomes without a parseable range: show market data only, no signal
        if model_prob is None:
            enriched_outcomes.append({
                "contract_id":      o.get("contract_id"),
                "question":         o.get("question", ""),
                "range_low":        o.get("range_low"),
                "range_high":       o.get("range_high"),
                "unit":             o.get("unit", "celsius"),
                "market_price":     round(market_price, 4),
                "yes_price":        round(market_price, 4),
                "no_price":         round(o.get("no_price", 1 - market_price), 4),
                "yes_token_id":     o.get("yes_token_id"),
                "no_token_id":      o.get("no_token_id"),
                "liquidity_usd":    o.get("liquidity_usd", 0),
                "volume_usd":       o.get("volume_usd", 0),
                "model_prob":       None,
                "raw_model_prob":   None,
                "ev":               None,
                "edge":             None,
                "recommended_side": None,
                "kelly_size":       0.0,
                "is_signal":        False,
            })
            continue

        model_prob_r = round(model_prob, 6)
        edge         = model_prob_r - market_price
        abs_edge     = abs(edge)

        if edge > 0:
            side = "YES"
        elif edge < 0:
            side = "NO"
        else:
            side = None

        ev = _calculate_ev(model_prob, market_price, side) if side else 0.0

        # A market_price of exactly 0 or 1 means the market considers this bin
        # resolved/impossible — no valid odds exist and the bin is not tradeable.
        market_at_extreme = market_price <= 0.0 or market_price >= 1.0
        is_signal = abs_edge >= EDGE_THRESHOLD and side is not None and not market_at_extreme

        # Kelly sizing — halved when normalization warning is active
        if is_signal:
            # Odds are safe to compute here because market_at_extreme is False
            if side == "YES":
                odds = (1.0 - market_price) / market_price
            else:
                odds = market_price / (1.0 - market_price)
            kelly_raw = calculate_kelly_size(
                edge     = abs_edge,
                odds     = odds,
                bankroll = bankroll,
            )
            kelly_size = round(kelly_raw * (0.5 if norm_warning else 1.0), 2)
        else:
            kelly_size = 0.0

        enriched_outcomes.append({
            # Identity
            "contract_id":      o.get("contract_id"),
            "question":         o.get("question", ""),
            "range_low":        o.get("range_low"),
            "range_high":       o.get("range_high"),
            "unit":             o.get("unit", "celsius"),
            # Prices
            "market_price":     round(market_price, 4),
            "yes_price":        round(market_price, 4),
            "no_price":         round(o.get("no_price", 1 - market_price), 4),
            "yes_token_id":     o.get("yes_token_id"),
            "no_token_id":      o.get("no_token_id"),
            "liquidity_usd":    o.get("liquidity_usd", 0),
            "volume_usd":       o.get("volume_usd", 0),
            # Model
            "model_prob":       model_prob,
            "raw_model_prob":   raw_prob,
            # Signal
            "ev":               round(ev, 4),
            "edge":             round(edge, 4),
            "recommended_side": side,
            "kelly_size":       kelly_size,
            "is_signal":        is_signal,
        })

    # Build event-level analysis dict
    result = {
        **event,
        "forecast_mu_c":       mu_c,
        "forecast_sigma_c":    sigma_c,
        "clim_mu_c":           dist.get("clim_mu_c", mu_c),
        "clim_sigma_c":        dist.get("clim_sigma_c", sigma_c),
        "forecast_mu_display": round(mu_display, 1),
        "display_unit":        display_unit,
        "days_ahead":          days_ahead,
        "model_probs_sum":     round(raw_sum, 4),
        "normalization_warning": norm_warning,
        "market_overround":    round(market_sum, 4),
        "n_outcomes":          n_outcomes,
        "n_sources":           len(dist.get("sources", [])),
        "sources":             dist.get("sources", []),
        "outcomes":            enriched_outcomes,
    }

    n_signals = sum(1 for o in enriched_outcomes if o["is_signal"])
    logger.info(
        f"Event: {city} {date_str} | μ={mu_c:.1f}°C σ={sigma_c:.1f}°C | "
        f"{n_outcomes} outcomes | {n_signals} signals | "
        f"overround={market_sum:.3f}"
    )

    return result


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

def run_edge_scan(
    bankroll: float = 1000.0,
    events: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Discover and analyze all highest-temperature Polymarket events.

    Args:
        bankroll: Current available USDC (used for Kelly sizing).
        events:   Pre-fetched event list from search_temp_high_events().
                  If None, fetches fresh from Polymarket API.

    Returns:
        (all_events_analyzed, signals)
        all_events_analyzed — one dict per event, containing enriched outcomes
                              (used by dashboard Tab 1 and DB write)
        signals             — flat list of individual outcome-level signal dicts
                              (used by risk.py + execution.py)
    """
    logger.info("Starting edge scan...")
    clear_forecast_cache()
    reset_tomorrowio_limit()

    if events is None:
        events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)
    logger.info(f"Analyzing {len(events)} highest-temperature events")

    from datetime import timezone
    scan_ts = datetime.now(timezone.utc).isoformat()

    all_events_analyzed: list[dict] = []
    signals:             list[dict] = []

    skipped_no_dist = 0

    for event in events:
        analysis = analyze_temp_event(event, bankroll=bankroll)

        if analysis is None:
            skipped_no_dist += 1
            continue

        # Persist to DB
        try:
            event_row_id = insert_temp_event(analysis, scan_ts)
            for outcome in analysis["outcomes"]:
                insert_temp_outcome(outcome, event_row_id, scan_ts)
        except Exception as e:
            logger.error(f"DB write failed for {analysis.get('city')} {analysis.get('date')}: {e}")

        all_events_analyzed.append(analysis)

        # Flatten signals with event-level context
        for outcome in analysis["outcomes"]:
            if not outcome.get("is_signal"):
                continue

            signal = {
                # Outcome fields (execution needs these)
                **outcome,
                # Event context
                "city":        analysis.get("city"),
                "date":        analysis.get("date"),
                "lat":         analysis.get("lat"),
                "lon":         analysis.get("lon"),
                "event_id":    analysis.get("event_id"),
                "event_title": analysis.get("event_title"),
                "scan_timestamp": scan_ts,
                # Aliased fields expected by risk.py / execution.py
                "contract_id":      outcome.get("contract_id"),
                "recommended_side": outcome.get("recommended_side"),
                "kelly_size":       outcome.get("kelly_size"),
                "market_p":         outcome.get("market_price"),
                "model_p":          outcome.get("model_prob"),
                "metadata": {
                    "date":     analysis.get("date"),
                    "lat":      analysis.get("lat"),
                    "lon":      analysis.get("lon"),
                    "variable": "temp_high",
                },
            }
            signals.append(signal)

            # Also log to legacy signals table for backward compatibility
            try:
                insert_signal(
                    timestamp        = scan_ts,
                    contract_id      = outcome["contract_id"],
                    question         = outcome.get("question"),
                    market_p         = outcome.get("market_price"),
                    model_p          = outcome.get("model_prob"),
                    ev               = outcome.get("ev"),
                    recommended_side = outcome.get("recommended_side"),
                    kelly_size       = outcome.get("kelly_size"),
                )
            except Exception as e:
                logger.debug(f"Legacy signal insert failed: {e}")

    logger.info(
        f"Edge scan complete: {len(all_events_analyzed)} events analyzed | "
        f"{len(signals)} signals | "
        f"{skipped_no_dist} skipped (no distribution)"
    )

    return all_events_analyzed, signals
