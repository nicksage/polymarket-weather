import logging
from datetime import datetime, date, timedelta
from config import EDGE_THRESHOLD, MIN_LIQUIDITY_USD, MAX_FORECAST_DAYS
from weather import get_ensemble_probability, clear_forecast_cache, reset_tomorrowio_limit
from polymarket import search_weather_markets, parse_contract_metadata
from db import insert_signal
from sizing import calculate_kelly_size

'''

model_p, market_p, and ev — Plain English
market_p — What the crowd thinks
This is the current YES price on Polymarket, expressed as a probability.

If market_p = 0.43, that means the market collectively believes there is a 43% chance the event happens (e.g., "29°C or higher in Shanghai today").

It's directly readable as a probability because Polymarket is a binary market — a YES share costs $0.43 and pays $1.00 if it resolves YES. The price is the implied probability.

model_p — What the weather models think
This is your ensemble forecast — the weighted blend of NWS, Open-Meteo, NOAA, and Tomorrow.io — expressed as a probability of the same event.

If model_p = 0.72, your weather models collectively say there is a 72% chance the event happens.

ev — Expected profit per dollar wagered
EV (Expected Value) is how much you expect to make per $1 bet, given the gap between your model and the market.

model says 72%, market prices it at 43% → bet YES
EV = 0.72 × (1 - 0.43) - 0.28 × 0.43
   = 0.72 × 0.57  -  0.28 × 0.43
   = 0.410        -  0.120
   = +0.29
An EV of +0.29 means: for every $1 wagered, you expect to profit $0.29 on average over many trades.

Reading a signal as "what are the chances of X happening?"
Yes — this framing is exactly right, and these numbers map directly to it:

Contract:    "Will Shanghai reach 29°C or higher on April 9?"

market_p  =  0.43  →  "The crowd says 43% chance YES"
model_p   =  0.72  →  "Your weather models say 72% chance YES"
edge      =  0.29  →  "You think the market is underpricing this by 29pp"
ev        =  0.29  →  "You expect +$0.29 profit per $1 bet on YES"
signal    =  BUY YES

The entire bet is: "I think X is more likely to happen than the market believes."

When model_p > market_p → bet YES (market underestimates the chance it happens)
When model_p < market_p → bet NO (market overestimates the chance it happens)

The edge threshold (EDGE_THRESHOLD=0.07) means you only act when the disagreement is ≥ 7 percentage points — small gaps aren't worth the transaction cost and model noise.

'''

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

    # Check contract is not expiring too soon (< 6 hours) or too far out (> MAX_FORECAST_DAYS)
    try:
        resolution_date = date.fromisoformat(metadata["date"])
        now = datetime.now()
        hours_to_expiry = (
            datetime.combine(resolution_date, datetime.max.time()) - now
        ).total_seconds() / 3600
        days_to_expiry = hours_to_expiry / 24

        if hours_to_expiry < 6:
            logger.info(f"Skipping {contract_id} '{question[:50]}': expires in {hours_to_expiry:.1f}h")
            return None

        if days_to_expiry > MAX_FORECAST_DAYS:
            logger.info(
                f"Skipping {contract_id} '{question[:50]}': "
                f"{days_to_expiry:.1f} days out > MAX_FORECAST_DAYS={MAX_FORECAST_DAYS}"
            )
            return None
    except (ValueError, TypeError) as e:
        logger.info(f"Could not check expiry for {contract_id}: {e} | date={metadata.get('date')}")

    # Step 2: Get ensemble weather probability
    ensemble = get_ensemble_probability(
        lat=metadata["lat"],
        lon=metadata["lon"],
        date_str=metadata["date"],
        variable=metadata.get("variable", "rain"),
        threshold=metadata.get("threshold"),
        unit=metadata.get("unit"),
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


def run_edge_scan(bankroll: float = 1000.0, contracts: list[dict] | None = None) -> list[dict]:
    """
    Main scan loop: find all weather markets, analyze each, save signals.

    Args:
        contracts: Pre-fetched market list. If None, fetches from Polymarket API.
    Returns list of signals generated this run.
    """
    logger.info("Starting edge scan...")
    clear_forecast_cache()       # Avoid stale cached forecasts across runs
    reset_tomorrowio_limit()     # Allow Tomorrow.io to be retried each scan

    if contracts is None:
        contracts = search_weather_markets(min_liquidity=MIN_LIQUIDITY_USD)
    logger.info(f"Analyzing {len(contracts)} weather contracts")

    signals = []
    drop_no_metadata = 0
    drop_no_edge = 0

    for contract in contracts:
        question = contract.get("question", "")
        group = contract.get("group_title", "")
        metadata = parse_contract_metadata(contract)
        if not metadata:
            drop_no_metadata += 1
            logger.debug(f"No metadata | q='{question[:60]}' group='{group[:60]}'")
            continue

        signal = analyze_contract(contract, bankroll=bankroll)
        if signal is None:
            # analyze_contract logs the specific reason at INFO level now
            drop_no_edge += 1
            continue

        insert_signal(
            timestamp=signal["timestamp"],
            contract_id=signal["contract_id"],
            question=signal.get("question"),
            market_p=signal.get("market_p"),
            model_p=signal.get("model_p"),
            ev=signal.get("ev"),
            recommended_side=signal.get("recommended_side"),
            kelly_size=signal.get("kelly_size"),
        )
        signals.append(signal)

    logger.info(
        f"Edge scan complete: {len(signals)} signals | "
        f"dropped: {drop_no_metadata} no-metadata, "
        f"{drop_no_edge} no-edge/expiry/ensemble "
        f"(out of {len(contracts)} contracts)"
    )
    return signals
