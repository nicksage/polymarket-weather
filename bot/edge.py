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
from config import (
    EDGE_THRESHOLD, MIN_LIQUIDITY_USD, MAX_FORECAST_DAYS,
    NORM_WARNING_LOW, NORM_WARNING_HIGH,
    USE_LIVE_ADJUSTMENT, LIVE_ADJ_OBS_FLOOR,
    USE_LIVE_ADJUSTMENT_SIGMA_SHRINK, LIVE_ADJ_TAIL_FLATTEN_EXPONENT,
)
from live_adjustment import compute_live_adjustment, components_to_json
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
    insert_decision_snapshot,
    get_latest_forecast_run,
    get_previous_forecast_run,
    get_latest_observation,
    get_latest_vc_forecast_diagnostic,
)
from sizing import calculate_kelly_size, compute_confidence_multiplier, compute_time_scale

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timezone lookup (cached per-scan via db read; cheap)
# ---------------------------------------------------------------------------

def _lookup_event_timezone(event_id: str) -> str | None:
    """Return the IANA timezone stored on the most recent temp_events row
    for this event_id, populated by loops.live_observation_run.  None if
    unknown (cold-start before first VC pull)."""
    if not event_id:
        return None
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT timezone FROM temp_events WHERE event_id = ? "
                "AND timezone IS NOT NULL ORDER BY scan_timestamp DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


def _lookup_event_agreement(event_id: str) -> float | None:
    """Return forecast_agreement_c from the most recent temp_events row."""
    if not event_id:
        return None
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT forecast_agreement_c FROM temp_events WHERE event_id = ? "
                "AND forecast_agreement_c IS NOT NULL "
                "ORDER BY scan_timestamp DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            return float(row[0]) if row else None
        finally:
            conn.close()
    except Exception:
        return None


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
# Decision snapshot writer (one row per outcome per trading run)
# ---------------------------------------------------------------------------

def _write_decision_snapshots(
    analysis: dict, group_id: str, evaluated_at_utc: str,
) -> int:
    """Capture the full decision context for every outcome in `analysis`.
    Called once per event per trading run; writes N rows to decision_snapshots
    where N = len(analysis["outcomes"]).  Returns rows written."""
    event_id = analysis.get("event_id") or ""
    if not event_id:
        return 0

    # Pull latest forecast + observation context (shared across all bins)
    latest_ecmwf = get_latest_forecast_run(event_id, "ecmwf")
    latest_gfs   = get_latest_forecast_run(event_id, "gfs")
    prev_ecmwf   = get_previous_forecast_run(event_id, "ecmwf")
    latest_obs   = get_latest_observation(event_id)
    latest_vc_diag = get_latest_vc_forecast_diagnostic(event_id) or {}

    # Forecast delta vs prior pull (ECMWF only; GFS delta is tracked on
    # temp_events.forecast_delta_mu_c as an average in loops.py)
    delta_mu = delta_sigma = None
    if latest_ecmwf and prev_ecmwf:
        if latest_ecmwf.get("forecast_mu_c") is not None and prev_ecmwf.get("forecast_mu_c") is not None:
            delta_mu = latest_ecmwf["forecast_mu_c"] - prev_ecmwf["forecast_mu_c"]
        if latest_ecmwf.get("forecast_sigma_c") is not None and prev_ecmwf.get("forecast_sigma_c") is not None:
            delta_sigma = latest_ecmwf["forecast_sigma_c"] - prev_ecmwf["forecast_sigma_c"]

    agreement = None
    if latest_ecmwf and latest_gfs:
        mu_e = latest_ecmwf.get("forecast_mu_c")
        mu_g = latest_gfs.get("forecast_mu_c")
        if mu_e is not None and mu_g is not None:
            agreement = abs(mu_e - mu_g)

    current_temp        = latest_obs.get("current_temp_c")        if latest_obs else None
    observed_max        = latest_obs.get("observed_max_so_far_c") if latest_obs else None

    # Expected temp now (from latest ECMWF hourly) and derived
    # forecast_remaining_max_c / projected_day_max_c are re-derived in
    # loops.live_observation_run; we surface the values stored on the latest
    # observation row if present, else NULL.
    expected_now = None   # populated by live-obs flow in a future phase
    actual_minus_expected = None
    if current_temp is not None and expected_now is not None:
        actual_minus_expected = current_temp - expected_now

    written = 0
    for rank, outcome in enumerate(analysis.get("outcomes", [])):
        snap = {
            "event_snapshot_group_id": group_id,
            "event_id":                event_id,
            "contract_id":             outcome.get("contract_id"),
            "city":                    analysis.get("city"),
            "date":                    analysis.get("date"),
            "evaluated_at_utc":        evaluated_at_utc,
            "latest_ecmwf_run_id":     latest_ecmwf["id"] if latest_ecmwf else None,
            "latest_gfs_run_id":       latest_gfs["id"]   if latest_gfs   else None,
            "blended_mu_c":            analysis.get("forecast_mu_c"),
            "blended_sigma_c":         analysis.get("forecast_sigma_c"),
            "latest_obs_id":           latest_obs["id"]   if latest_obs   else None,
            "current_temp_c":          current_temp,
            "temp_change_1h_c":        None,
            "temp_change_3h_c":        None,
            "observed_max_so_far_c":   analysis.get("observed_max_so_far_c") or observed_max,
            "forecast_remaining_max_c": None,
            "projected_day_max_c":     None,
            "expected_temp_now_c":     expected_now,
            "actual_minus_expected_c": actual_minus_expected,
            "forecast_delta_mu_c":     delta_mu,
            "forecast_delta_sigma_c":  delta_sigma,
            "forecast_agreement_c":    agreement,
            "market_price":            outcome.get("market_price"),
            "model_prob":              outcome.get("model_prob"),
            "raw_model_prob":          outcome.get("raw_model_prob"),
            "edge":                    outcome.get("edge"),
            "ev":                      outcome.get("ev"),
            "recommended_side":        outcome.get("recommended_side"),
            "kelly_size":              outcome.get("kelly_size"),
            "liquidity_usd":           outcome.get("liquidity_usd"),
            "action":                  "enter" if outcome.get("is_signal") else "skip",
            "reason":                  None,
            # Phase 2b live adjustment (event-level, same across all bins in this run)
            "adjusted_mu_c":             analysis.get("adjusted_mu_c"),
            "adjusted_sigma_c":          analysis.get("adjusted_sigma_c"),
            "live_adjustment_score":     analysis.get("live_adjustment_score"),
            "live_adjustment_components": analysis.get("live_adjustment_components"),
            "obs_floor_applied":         analysis.get("obs_floor_applied", 0),
            # Phase 2c — VC forecast diagnostics at decision time
            "vc_projected_day_max_c":       latest_vc_diag.get("vc_projected_day_max_c"),
            "vc_vs_blended_mu_c":           latest_vc_diag.get("vc_vs_blended_mu_c"),
            "vc_vs_adjusted_mu_c":          latest_vc_diag.get("vc_vs_adjusted_mu_c"),
            "vc_hourly_path_rmse_c":        latest_vc_diag.get("vc_hourly_path_rmse_c"),
            "vc_bins_apart":                latest_vc_diag.get("vc_bins_apart"),
            "flag_vc_disagreement_large":   latest_vc_diag.get("flag_vc_disagreement_large", 0),
            "flag_vc_warns_hotter":         latest_vc_diag.get("flag_vc_warns_hotter", 0),
            "flag_vc_warns_colder":         latest_vc_diag.get("flag_vc_warns_colder", 0),
        }
        try:
            insert_decision_snapshot(snap)
            written += 1
        except Exception as e:
            logger.debug(f"snapshot insert failed: {e}")
    return written


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

    # --- Live adjustment layer (Phase 2b) ---------------------------------
    # Looks up live_observations + latest forecast_hourly for this event.
    # Returns blended-pass-through in cold-start (no VC data yet).
    event_id = event.get("event_id") or ""
    tz_str = _lookup_event_timezone(event_id)
    adj = compute_live_adjustment(
        event_id        = event_id,
        blended_mu_c    = mu_c,
        blended_sigma_c = sigma_c,
        target_date     = date_str,
        timezone_str    = tz_str,
        lat             = lat,
        lon             = lon,
    )
    adjusted_mu    = adj["adjusted_mu_c"]
    adjusted_sigma = adj["adjusted_sigma_c"]
    obs_floor_c    = adj.get("observed_max_so_far_c") if LIVE_ADJ_OBS_FLOOR else None

    # Shadow mode: both pre- and post-adjustment probs are computed.
    # When USE_LIVE_ADJUSTMENT is False, `active_*` = blended; trading logic
    # uses `active_*`.  When True, `active_*` = adjusted.
    if USE_LIVE_ADJUSTMENT:
        active_mu, active_sigma = adjusted_mu, adjusted_sigma
        active_floor            = obs_floor_c
    else:
        active_mu, active_sigma = mu_c, sigma_c
        active_floor            = None

    # Display-unit mu for dashboard (use blended — base model)
    if display_unit == "fahrenheit":
        mu_display = mu_c * 9 / 5 + 32
    else:
        mu_display = mu_c

    outcomes   = event.get("outcomes", [])
    n_outcomes = len(outcomes)

    # Guard: if any single bin is trading at >=95%, the market has
    # effectively resolved this event.  The forecast-based model cannot
    # compete with actual observational data already priced in — every
    # "edge" we see is the model disagreeing with reality, not the market
    # being mispriced.  Skip the entire event.
    for o in outcomes:
        yp = o.get("yes_price", o.get("market_price", 0))
        if yp is not None and float(yp) >= 0.95:
            logger.info(
                f"Skipping {city} {date_str}: bin {o.get('question', '')[:30]} "
                f"trading at {float(yp)*100:.0f}% — market has effectively resolved"
            )
            return None

    # Step 1: compute raw model probabilities for each bin — two parallel
    # passes: blended (no floor, base model) and active (chosen by flag).
    raw_probs:         list[float | None] = []
    raw_probs_blended: list[float | None] = []
    for o in outcomes:
        lo = o.get("range_low")
        hi = o.get("range_high")
        if lo is None and hi is None:
            raw_probs.append(None)
            raw_probs_blended.append(None)
            continue
        unit = o.get("unit", "celsius")
        p_active = get_temp_range_probability(
            active_mu, active_sigma, lo, hi, unit,
            kde=clim_kde, obs_floor_c=active_floor,
        )
        p_blended = get_temp_range_probability(
            mu_c, sigma_c, lo, hi, unit,
            kde=clim_kde,   # no floor on the blended baseline
        )
        raw_probs.append(p_active)
        raw_probs_blended.append(p_blended)

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

    # Parallel normalization for the blended baseline (never uses the floor).
    parseable_blended = [p for p in raw_probs_blended if p is not None]
    blended_sum = sum(parseable_blended)
    if blended_sum > 0:
        model_probs_blended: list[float | None] = [
            (p / blended_sum) if p is not None else None for p in raw_probs_blended
        ]
    elif parseable_blended:
        n_pb = len(parseable_blended)
        model_probs_blended = [
            (1.0 / n_pb) if p is not None else None for p in raw_probs_blended
        ]
    else:
        model_probs_blended = [None] * n_outcomes

    # Tail flattening — applied to the ACTIVE probabilities (model_probs) only,
    # when the v2 σ shrink path is on.  Preserves ranking; trims overconfident
    # tail spikes.  Exponent == 1.0 disables.
    if (USE_LIVE_ADJUSTMENT
            and USE_LIVE_ADJUSTMENT_SIGMA_SHRINK
            and LIVE_ADJ_TAIL_FLATTEN_EXPONENT != 1.0):
        e = LIVE_ADJ_TAIL_FLATTEN_EXPONENT
        raised = [((p ** e) if p is not None else None) for p in model_probs]
        total = sum(p for p in raised if p is not None)
        if total > 0:
            model_probs = [((p / total) if p is not None else None) for p in raised]

    # Step 3: market overround
    market_prices  = [o.get("yes_price", 0.5) for o in outcomes]
    market_sum = sum(market_prices)

    # Step 4: build enriched outcome list
    enriched_outcomes = []
    for i, o in enumerate(outcomes):
        market_price = market_prices[i]
        model_prob   = model_probs[i]   # may be None for unparseable ranges
        raw_prob     = raw_probs[i]     # may be None
        model_prob_blended = model_probs_blended[i]

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
                "model_prob_blended": None,
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

        # Kelly sizing with sign-aware confidence multiplier (Phase 3).
        # Replaces the old binary norm_warning half-Kelly with a smooth
        # multi-factor confidence score.
        if is_signal:
            if side == "YES":
                odds = (1.0 - market_price) / market_price
            else:
                odds = market_price / (1.0 - market_price)
            kelly_raw = calculate_kelly_size(
                edge     = abs_edge,
                odds     = odds,
                bankroll = bankroll,
            )
            conf_signal = {
                "range_low": o.get("range_low"),
                "range_high": o.get("range_high"),
                "recommended_side": side,
                "unit": o.get("unit", "celsius"),
                "days_ahead": days_ahead,
                "normalization_warning": norm_warning,
            }
            conf_state = {
                "forecast_agreement_c": _lookup_event_agreement(event_id),
                "adjusted_mu_c": adjusted_mu,
                "forecast_mu_c": mu_c,
                "live_adjustment_score": adj.get("live_adjustment_score"),
            }
            confidence = compute_confidence_multiplier(conf_signal, conf_state)
            # Two-regime time scaling — observation-driven (same-day)
            # gets full or boosted sizing mid-day; forecast-driven gets a
            # gentle discount as expiry approaches.
            local_hr = None
            if tz_str:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    from zoneinfo import ZoneInfo
                    now_local = _dt.now(ZoneInfo(tz_str))
                    local_hr = float(now_local.hour) + now_local.minute / 60.0
                except Exception:
                    pass
            time_scale = compute_time_scale(days_ahead, local_hr)
            kelly_size = round(kelly_raw * confidence * time_scale, 2)
        else:
            kelly_size = 0.0
            confidence = 1.0
            time_scale = 1.0

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
            "model_prob_blended": model_prob_blended,
            "raw_model_prob":   raw_prob,
            # Signal
            "ev":               round(ev, 4),
            "edge":             round(edge, 4),
            "recommended_side": side,
            "kelly_size":       kelly_size,
            "is_signal":        is_signal,
            # Priority ranking inputs (flow through to signal dict)
            "confidence_multiplier": confidence,
            "time_scale":            time_scale,
            "days_ahead":            days_ahead,
            "live_adjustment_score": adj.get("live_adjustment_score"),
            "adjusted_mu_c":         adjusted_mu,
            "forecast_mu_c":         mu_c,
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
        # Live adjustment (Phase 2b)
        "adjusted_mu_c":               adjusted_mu,
        "adjusted_sigma_c":            adjusted_sigma,
        "live_adjustment_score":       adj.get("live_adjustment_score"),
        "live_adjustment_components":  components_to_json(adj.get("components", {})),
        "mu_adjustment_c":             adj.get("mu_adjustment_c"),
        "observed_max_so_far_c":       adj.get("observed_max_so_far_c"),
        "obs_floor_applied":           1 if (USE_LIVE_ADJUSTMENT and active_floor is not None) else 0,
    }

    # Log the adjustment alongside the existing event summary so shadow-mode
    # behavior is observable without a DB query.
    _comp = adj.get("components", {})
    logger.info(
        f"LiveAdj: {city} {date_str} | blended_mu={mu_c:.2f} adj_mu={adjusted_mu:.2f} "
        f"delta={adj.get('mu_adjustment_c'):+.2f} score={adj.get('live_adjustment_score'):.3f} "
        f"floor={adj.get('observed_max_so_far_c')} "
        f"reason={_comp.get('reason')} n_obs={_comp.get('n_obs_used')} "
        f"mode={'LIVE' if USE_LIVE_ADJUSTMENT else 'SHADOW'}"
    )

    n_signals = sum(1 for o in enriched_outcomes if o["is_signal"])
    logger.info(
        f"Event: {city} {date_str} | mu={mu_c:.1f}°C sd={sigma_c:.1f}°C | "
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
    import uuid
    scan_ts = datetime.now(timezone.utc).isoformat()
    snapshot_group_id = str(uuid.uuid4())   # one per trading-run cycle

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

        # ----- Decision snapshot (one row per outcome, grouped by run) -----
        try:
            _write_decision_snapshots(analysis, snapshot_group_id, scan_ts)
        except Exception as e:
            logger.debug(f"Decision snapshot write failed for {analysis.get('city')}: {e}")

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

    # ----- Ensemble health monitoring (ongoing) -----
    # If a high fraction of events were skipped because one model was missing,
    # log a prominent warning so the operator notices mid-session degradation.
    total_attempted = len(all_events_analyzed) + skipped_no_dist
    if total_attempted > 0 and skipped_no_dist > total_attempted * 0.50:
        logger.error(
            f"ENSEMBLE HEALTH WARNING: {skipped_no_dist}/{total_attempted} events "
            f"({skipped_no_dist/total_attempted*100:.0f}%) had no usable distribution. "
            f"A model source may be down or returning empty data. "
            f"Check ECMWF/GFS connectivity and model names."
        )

    # ----- Phase 3: 2-bin bracket promotion -----
    from config import BRACKET_ENABLED, BRACKET_MAX_BINS, BRACKET_MIN_ADJACENT_EDGE
    if BRACKET_ENABLED:
        signals = _promote_bracket_bins(signals, all_events_analyzed, bankroll,
                                        BRACKET_MIN_ADJACENT_EDGE, BRACKET_MAX_BINS,
                                        scan_ts)

    logger.info(
        f"Edge scan complete: {len(all_events_analyzed)} events analyzed | "
        f"{len(signals)} signals | "
        f"{skipped_no_dist} skipped (no distribution)"
    )

    return all_events_analyzed, signals


def _promote_bracket_bins(
    signals: list[dict],
    all_events: list[dict],
    bankroll: float,
    min_edge: float,
    max_bins: int,
    scan_ts: str,
) -> list[dict]:
    """Identify adjacent bins that qualify for bracket promotion and add them
    as signals.  Bins must be contiguous to an existing signal bin (by
    range_low/range_high adjacency) and have edge >= min_edge."""
    from collections import defaultdict

    # Group existing signals by event_id
    signal_events: dict[str, set[str]] = defaultdict(set)
    for s in signals:
        signal_events[s.get("event_id", "")].add(s.get("contract_id", ""))

    promoted = 0
    new_signals: list[dict] = list(signals)

    for analysis in all_events:
        eid = analysis.get("event_id", "")
        sig_cids = signal_events.get(eid, set())
        if not sig_cids:
            continue

        # Count existing signal bins for this event — respect max_bins
        n_existing = len(sig_cids)
        if n_existing >= max_bins:
            continue

        outcomes = analysis.get("outcomes", [])
        # Build ordered list with edge info
        bins = []
        for o in outcomes:
            if o.get("range_low") is None or o.get("range_high") is None:
                continue
            bins.append(o)
        # Sort by range_low
        bins.sort(key=lambda o: (o.get("range_low") or 0))

        # Find bins adjacent to signal bins that qualify
        for i, b in enumerate(bins):
            if b.get("contract_id") in sig_cids:
                # Check neighbors
                for neighbor_idx in (i - 1, i + 1):
                    if neighbor_idx < 0 or neighbor_idx >= len(bins):
                        continue
                    nb = bins[neighbor_idx]
                    if nb.get("contract_id") in sig_cids:
                        continue
                    nb_cid = nb.get("contract_id")
                    if not nb_cid or nb_cid in sig_cids:
                        continue
                    nb_edge = nb.get("edge")
                    if nb_edge is None:
                        continue
                    nb_side = nb.get("recommended_side")
                    if nb_side is None:
                        continue
                    if abs(nb_edge) < min_edge:
                        continue
                    if n_existing >= max_bins:
                        break

                    # Promote: create signal dict with reduced sizing
                    bracket_signal = {
                        **nb,
                        "city":        analysis.get("city"),
                        "date":        analysis.get("date"),
                        "lat":         analysis.get("lat"),
                        "lon":         analysis.get("lon"),
                        "event_id":    eid,
                        "event_title": analysis.get("event_title"),
                        "scan_timestamp": scan_ts,
                        "contract_id":      nb.get("contract_id"),
                        "recommended_side": nb_side,
                        "kelly_size":       round(nb.get("kelly_size", 0) * 0.75, 2),
                        "market_p":         nb.get("market_price"),
                        "model_p":          nb.get("model_prob"),
                        "is_bracket":       True,
                        "metadata": {
                            "date":     analysis.get("date"),
                            "lat":      analysis.get("lat"),
                            "lon":      analysis.get("lon"),
                            "variable": "temp_high",
                        },
                    }
                    new_signals.append(bracket_signal)
                    sig_cids.add(nb.get("contract_id"))
                    n_existing += 1
                    promoted += 1

    if promoted:
        logger.info(f"Bracket promotion: {promoted} adjacent bins promoted to signals")
    return new_signals
