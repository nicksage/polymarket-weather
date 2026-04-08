
'''

Use the Polymarket Gamma API's historical endpoint to pull resolved weather contracts, then
test whether your model would have found edge

'''

# backtest.py
import httpx
import json
import logging
from datetime import date, timedelta
from weather import get_ensemble_probability
from bot.polymarket import _normalize_market, parse_contract_metadata

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


def fetch_resolved_weather_markets(days_back: int = 90) -> list[dict]:
    """
    Fetch resolved weather markets from the last N days.
    Used as the backtest dataset.
    """
    start_date = (date.today() - timedelta(days=days_back)).isoformat()
    all_markets = []

    for keyword in ["rain", "snow", "temperature", "weather"]:
        params = {
            "closed": "true",
            "active": "false",
            "q": keyword,
            "limit": 200,
            "startDateMin": start_date,
        }
        try:
            resp = httpx.get(f"{GAMMA_BASE}/markets", params=params, timeout=15)
            resp.raise_for_status()
            markets = resp.json()
            if isinstance(markets, list):
                all_markets.extend(markets)
        except Exception as e:
            logger.error(f"Failed to fetch resolved markets for '{keyword}': {e}")

    return all_markets


def backtest_contract(contract: dict) -> dict | None:
    """
    Test the model on a single historical contract.

    Returns a result dict with: model_p, market_p, outcome, brier_score, pnl

    """
    metadata = parse_contract_metadata(contract)
    if not metadata or not metadata.get("date"):
        return None

    # Get what the model WOULD have predicted at forecast time
    # (In practice, use the forecast from 24-48h before resolution)
    ensemble = get_ensemble_probability(
        lat=metadata["lat"],
        lon=metadata["lon"],
        date_str=metadata["date"],
        variable=metadata.get("variable", "rain"),
    )

    if not ensemble:
        return None

    p_model = ensemble["probability"]


    # Market price at time of analysis (use opening price as proxy)
    p_market = contract.get("yes_price", 0.5)

    # Actual outcome from resolved contract
    # Polymarket stores winner in the token with price=1.0 after resolution
    tokens = contract.get("tokens", [])
    outcome = None

    for token in tokens:
        if token.get("outcome", "").upper() == "YES" and float(token.get("price", 0)) >= 0.99:
            outcome = 1  # YES won
            break
        elif token.get("outcome", "").upper() == "NO" and float(token.get("price", 0)) >= 0.99:
            outcome = 0  # NO won
            break

    if outcome is None:
        return None

    # Brier score: (p - outcome)^2, lower is better
    brier = (p_model - outcome) ** 2

    # Simulated P&L: would we have bet and won?
    edge = p_model - p_market
    if abs(edge) > 0.07:  # Would have triggered a signal
        side = "YES" if edge > 0 else "NO"
        if side == "YES":
            pnl = (outcome - p_market) * 10  # $10 position
        else:
            pnl = ((1 - outcome) - (1 - p_market)) * 10
    else:
        pnl = 0  # No signal, no trade

    return {
        "contract_id": contract.get("conditionId"),
        "question": contract.get("question", "")[:80],
        "model_p": round(p_model, 4),
        "market_p": round(p_market, 4),
        "outcome": outcome,
        "brier_score": round(brier, 4),
        "pnl": round(pnl, 4),
        "had_signal": abs(edge) > 0.07,
    }

def run_backtest(days_back: int = 90) -> dict:
    """
    Run the full backtest and return summary statistics.
    """
    markets = fetch_resolved_weather_markets(days_back)
    logger.info(f"Backtesting on {len(markets)} resolved contracts")

    results = []
    for market in markets:
        result = backtest_contract(market)
        if result:
            results.append(result)

    if not results:
        return {"error": "No valid backtest results"}

    brier_scores = [r["brier_score"] for r in results]
    signal_results = [r for r in results if r["had_signal"]]

    win_rate = 0.0
    avg_pnl = 0.0

    if signal_results:
        wins = sum(1 for r in signal_results if r["pnl"] > 0)
        win_rate = wins / len(signal_results)
        avg_pnl = sum(r["pnl"] for r in signal_results) / len(signal_results)

    return {
        "total_contracts": len(results),
        "signals_generated": len(signal_results),
        "avg_brier_score": round(sum(brier_scores) / len(brier_scores), 4),
        "win_rate": round(win_rate, 4),
        "avg_pnl_per_signal": round(avg_pnl, 4),
        "total_pnl": round(sum(r["pnl"] for r in results), 4),
    }
