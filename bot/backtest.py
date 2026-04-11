"""
backtest.py — Replay the temperature distribution model against resolved contracts.

Fetches historical highest-temperature Polymarket contracts, runs the distribution
model to estimate what probability each outcome would have been assigned, and
computes Brier scores and simulated P&L.

Usage (run from bot/):
    python backtest.py

Note: The ERA5-based model uses historical data that is already in the past, so
calling get_temp_distribution_for_event() on a resolved date is valid — it simply
retrieves historical ensemble re-runs and ERA5 reanalysis data for that date.
"""

import httpx
import logging
from datetime import date, timedelta

from weather import get_temp_distribution_for_event, get_temp_range_probability
from polymarket import _normalize_sub_market, _is_highest_temp_event, parse_temperature_range
from polymarket import _extract_city_and_date

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


def fetch_resolved_temp_events(days_back: int = 90) -> list[dict]:
    """
    Fetch resolved highest-temperature events from the last N days via Gamma API.

    The Gamma API's `q` search is unreliable, so we fetch closed events and
    filter client-side for highest-temperature titles.
    """
    start_date = (date.today() - timedelta(days=days_back)).isoformat()
    all_events: list[dict] = []

    # Paginate closed events
    page_size = 100
    offset    = 0
    for _ in range(20):
        try:
            resp = httpx.get(
                f"{GAMMA_BASE}/events",
                params={
                    "closed":       "true",
                    "active":       "false",
                    "limit":        page_size,
                    "offset":       offset,
                    "startDateMin": start_date,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            page = data if isinstance(data, list) else data.get("events", [])
            if not page:
                break
            all_events.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        except Exception as e:
            logger.error(f"Failed to fetch closed events at offset={offset}: {e}")
            break

    # Client-side filter for highest-temperature events
    temp_events = [e for e in all_events if _is_highest_temp_event(e.get("title", ""))]
    logger.info(
        f"Fetched {len(all_events)} closed events — {len(temp_events)} are highest-temperature"
    )
    return temp_events


def _get_winning_outcome(market: dict) -> int | None:
    """
    Determine the resolved outcome (YES=1, NO=0) from a Gamma market dict.
    After resolution, the winning token trades at ≈$1.00.
    """
    import json as _json

    def _parse(field):
        val = market.get(field, "[]")
        if isinstance(val, str):
            try:
                return _json.loads(val)
            except Exception:
                return []
        return val or []

    outcomes = _parse("outcomes")
    prices   = _parse("outcomePrices")

    for i, o in enumerate(outcomes):
        if str(o).upper() == "YES" and i < len(prices):
            try:
                if float(prices[i]) >= 0.99:
                    return 1   # YES won
                elif float(prices[i]) <= 0.01:
                    return 0   # NO won (YES lost)
            except (ValueError, TypeError):
                pass
    return None


def backtest_event(raw_event: dict) -> list[dict] | None:
    """
    Run the distribution model against a single resolved historical event and
    return per-outcome result dicts.

    Each result dict contains:
        question, range_low, range_high, unit,
        model_prob, market_price_at_resolution,
        outcome (0/1), brier_score, had_signal, simulated_pnl
    """
    city_name, date_str, lat, lon, display_unit = _extract_city_and_date(raw_event)
    if lat is None or date_str is None:
        return None

    # Get what the distribution model would have predicted
    dist = get_temp_distribution_for_event(lat, lon, date_str)
    if dist is None:
        return None

    mu_c    = dist["mu_c"]
    sigma_c = dist["sigma_c"]

    markets = raw_event.get("markets", [])
    if not markets:
        return None

    # Compute raw model probs for all sub-markets
    outcome_data = []
    for mkt in markets:
        question = mkt.get("question", "")
        low, high, unit = parse_temperature_range(question)
        if low is None and high is None:
            continue

        resolved_outcome = _get_winning_outcome(mkt)
        if resolved_outcome is None:
            continue

        import json as _j
        def _parse_f(field):
            val = mkt.get(field, "[]")
            if isinstance(val, str):
                try:
                    return _j.loads(val)
                except Exception:
                    return []
            return val or []

        prices = _parse_f("outcomePrices")
        outcomes_list = _parse_f("outcomes")
        yes_idx = next(
            (i for i, o in enumerate(outcomes_list) if str(o).upper() == "YES"), None
        )
        market_price = float(prices[yes_idx]) if yes_idx is not None and yes_idx < len(prices) else 0.5

        raw_prob = get_temp_range_probability(mu_c, sigma_c, low, high, unit)
        outcome_data.append({
            "question":    question,
            "range_low":   low,
            "range_high":  high,
            "unit":        unit,
            "raw_prob":    raw_prob,
            "market_price": market_price,
            "outcome":     resolved_outcome,
        })

    if not outcome_data:
        return None

    # Normalize model probs
    raw_sum = sum(d["raw_prob"] for d in outcome_data)
    if raw_sum <= 0:
        return None

    results = []
    for d in outcome_data:
        model_prob   = d["raw_prob"] / raw_sum
        market_price = d["market_price"]
        outcome      = d["outcome"]

        brier = (model_prob - outcome) ** 2

        # Simulate P&L on a $10 fixed position
        edge = model_prob - market_price
        if abs(edge) > 0.07:
            had_signal = True
            side = "YES" if edge > 0 else "NO"
            if side == "YES":
                pnl = (outcome - market_price) * 10
            else:
                pnl = ((1 - outcome) - (1 - market_price)) * 10
        else:
            had_signal = False
            pnl = 0.0

        results.append({
            "city":            city_name or "",
            "date":            date_str,
            "question":        d["question"],
            "range_low":       d["range_low"],
            "range_high":      d["range_high"],
            "unit":            d["unit"],
            "model_prob":      round(model_prob, 4),
            "market_price":    round(market_price, 4),
            "outcome":         outcome,
            "brier_score":     round(brier, 4),
            "had_signal":      had_signal,
            "simulated_pnl":   round(pnl, 4),
        })

    return results


def run_backtest(days_back: int = 90) -> dict:
    """
    Full backtest run.  Fetches resolved events, scores model, returns summary.
    """
    events = fetch_resolved_temp_events(days_back)
    logger.info(f"Backtesting on {len(events)} resolved events")

    all_results: list[dict] = []
    for event in events:
        outcomes = backtest_event(event)
        if outcomes:
            all_results.extend(outcomes)

    if not all_results:
        return {"error": "No valid backtest results"}

    brier_scores    = [r["brier_score"]   for r in all_results]
    signal_results  = [r for r in all_results if r["had_signal"]]
    wins            = sum(1 for r in signal_results if r["simulated_pnl"] > 0)
    win_rate        = wins / len(signal_results) if signal_results else 0.0
    avg_pnl         = (
        sum(r["simulated_pnl"] for r in signal_results) / len(signal_results)
        if signal_results else 0.0
    )

    summary = {
        "total_outcomes":       len(all_results),
        "signals_generated":    len(signal_results),
        "avg_brier_score":      round(sum(brier_scores) / len(brier_scores), 4),
        "win_rate":             round(win_rate, 4),
        "avg_pnl_per_signal":   round(avg_pnl, 4),
        "total_simulated_pnl":  round(sum(r["simulated_pnl"] for r in all_results), 4),
    }

    logger.info(
        f"Backtest complete: {summary['total_outcomes']} outcomes | "
        f"{summary['signals_generated']} signals | "
        f"Brier={summary['avg_brier_score']:.4f} | "
        f"Win rate={summary['win_rate']:.1%} | "
        f"Avg PnL/signal=${summary['avg_pnl_per_signal']:.2f}"
    )
    return summary


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = run_backtest(days_back=90)
    print(json.dumps(result, indent=2))
