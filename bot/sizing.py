import os
import logging
from config import KELLY_FRACTION, MAX_POSITION_PCT

logger = logging.getLogger(__name__)

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