import logging
from datetime import datetime, date, timedelta
from config import EDGE_THRESHOLD, MIN_LIQUIDITY_USD
from weather import get_ensemble_probability
from bot.polymarket import search_weather_markets, parse_contract_metadata
from db import insert_signal
from bot.sizing import calculate_kelly_size

logger = logging.getLogger(__name__)


def calculate_ev(p_model: float, p_market: float, side: str) -> float:
    """
    Calculate expected value per dollar wagered.
    For YES: EV = p_model * (1 - p_market) - (1 - p_model) * p_market
    For NO:  EV = (1 - p_model) * p_model - p_model * (1 - p_market)

    A positive EV means the bet is favorable at current market prices.
    EV of 0.10 means you expect to profit $0.10 for every $1.00 wagered.
    """
    if side == "YES":
        return p_model * (1 - p_market) - (1 - p_model) * p_market
    elif side == "NO":
        # If we think p_model is LOW, we want to buy NO at no_price = 1 - p_market
        p_no_model = 1 - p_model
        p_no_market = 1 - p_market
        return p_no_model * (1 - p_no_market) - (1 - p_no_model) * p_no_market
    else:
        raise ValueError(f"Invalid side: {side}")


def determine_side(p_model: float, p_market: float) -> tuple[str, float]:
    """
    Determine which side to trade and the edge magnitude.
    Returns (side, edge) where edge = |p_model - p_market|
    """
    edge = p_model - p_market
    if edge > 0:
        return "YES", abs(edge)
    else:
        return "NO", abs(edge)


def analyze_contract(contract: dict, bankroll: float = 1000.0) -> dict | None:
    """
    Full analysis pipeline for a single Polymarket weather contract.

    1. Parse meteorological parameters from contract question
    2. Fetch ensemble weather probability
    3. Compare to market implied probability
    4. Calculate EV and Kelly size
    5. Return signal dict if edge exceeds threshold, else None

    Returns a signal dict or None if no edge or parsing failed.
    """
    contract_id = contract.get("contract_id")
    question = contract.get("question", "")

    # Step 1: Parse what the contract is asking
    metadata = parse_contract_metadata(contract)
    if not metadata:
        logger.debug(f"Could not parse metadata for: {question[:80]}")
        return None

    # Check contract is not expiring too soon (< 6 hours)
    try:
        resolution_date = date.fromisoformat(metadata["date"])
        hours_to_expiry = (
            datetime.combine(resolution_date, datetime.max.time()) - datetime.now()
        ).total_seconds() / 3600
        if hours_to_expiry < 6:
            logger.debug(f"Skipping {contract_id}: expires in {hours_to_expiry:.1f}h")
            return None
    except (ValueError, TypeError):
        pass

    # Step 2: Get ensemble weather probability
    ensemble = get_ensemble_probability(
        lat=metadata["lat"],
        lon=metadata["lon"],
        date_str=metadata["date"],
        variable=metadata.get("variable", "rain"),
    )

    if not ensemble or ensemble.get("probability") is None:
        logger.warning(f"No ensemble probability for {contract_id}")
        return None

    p_model = ensemble["probability"]
    disagreement = ensemble.get("disagreement", 0.0)

    # Skip if sources disagree heavily (model uncertainty too high)
    if disagreement > 0.15:
        logger.info(f"Skipping {contract_id}: source disagreement {disagreement:.2f} > 0.15")
        return None

    # Step 3: Market implied probability
    p_market_yes = contract.get("yes_price", 0.5)

    # Step 4: Determine side and edge
    side, edge = determine_side(p_model, p_market_yes)

    if edge < EDGE_THRESHOLD:
        logger.debug(f"No edge on {contract_id}: model={p_model:.3f} market={p_market_yes:.3f}edge={edge:.3f}")
        return None

    # Step 5: Calculate EV
    ev = calculate_ev(p_model, p_market_yes, side)

    # Step 6: Kelly sizing
    kelly_size = calculate_kelly_size(
        edge=edge,
        odds=1.0 / (p_market_yes if side == "YES" else (1 - p_market_yes)),
        bankroll=bankroll,
    )

    signal = {
        "contract_id": contract_id,
        "question": question,
        "market_p": round(p_market_yes, 4),
        "model_p": round(p_model, 4),
        "ev": round(ev, 4),
        "recommended_side": side,
        "edge": round(edge, 4),
        "disagreement": round(disagreement, 4),
        "kelly_size": round(kelly_size, 2),
        "n_sources": ensemble.get("n_sources", 0),
        "sources": ensemble.get("sources", []),
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata,
    }

    logger.info(
        f"SIGNAL: {contract_id[:12]} | side={side} | model={p_model:.3f} "
        f"market={p_market_yes:.3f} | edge={edge:.3f} | EV={ev:.3f} | kelly=${kelly_size:.2f}"
    )

    return signal


def run_edge_scan(bankroll: float = 1000.0) -> list[dict]:
    """
    Main scan loop: find all weather markets, analyze each, save signals.

    Returns list of signals generated this run.
    """
    logger.info("Starting edge scan...")

    contracts = search_weather_markets(min_liquidity=MIN_LIQUIDITY_USD)
    logger.info(f"Analyzing {len(contracts)} weather contracts")

    signals = []
    for contract in contracts:
        signal = analyze_contract(contract, bankroll=bankroll)
        if signal:
            insert_signal(signal)
            signals.append(signal)

    logger.info(f"Edge scan complete: {len(signals)} signals found out of {len(contracts)} contracts")
    return signal
