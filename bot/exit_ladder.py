"""
exit_ladder.py — Sell-side patient ladder pricing.

When the bot wants to exit a position, it doesn't immediately cross the
spread.  It first posts a sell limit "near" the intended exit price, then
escalates progressively if the order doesn't fill within a monitor cycle:

    retry 0:  intended * 0.99       (1% give-up)
    retry 1:  intended * 0.98       (2% give-up)
    retry 2:  intended * 0.97       (3% give-up)
    retry 3:  intended * 0.96       (4% give-up — max patience)
    retry 4+: cross the spread      (post just below best bid — guarantees fill)

Why patient first: posting "at the intended price" preserves expected value
when the book is liquid enough that someone will lift the offer.  The 1%
buffer exists to give the price wiggle room above our resting offer so a
small amount of upward drift triggers a fill.

Why cross-spread eventually: if the book sits below our offer for 80 minutes
(4 retries × 20 min cycle), the market is telling us the spread won't tighten.
A stop-loss must actually stop losses; trading 1-3% slippage for guaranteed
exit is better than holding a deteriorating position indefinitely.
"""

from __future__ import annotations

# Multipliers applied to the intended exit price at each rung.
SELL_LADDER_MULTIPLIERS: tuple[float, ...] = (0.99, 0.98, 0.97, 0.96)

# Number of patient retries before crossing the spread.
MAX_LADDER_RETRIES: int = len(SELL_LADDER_MULTIPLIERS)


def ladder_price(
    intended_price: float,
    retry_count: int,
    current_best_bid: float | None = None,
) -> float | None:
    """Return the limit price for a sell at this rung of the ladder.

    Returns None if retries are exhausted (caller should cross the spread
    via _get_best_bid + post at bid - 1 tick instead of using this function).

    `intended_price` is the price level that triggered the exit decision —
    e.g., the trail level, the take-profit threshold, or the hard-stop level.

    `current_best_bid`, when provided, *re-anchors* each rung relative to
    where the market actually is now.  The patient rung price is computed
    as `intended × multiplier`, but the returned price is clamped to be
    no worse than `best_bid + 1 tick`:

        return min(intended × multiplier, best_bid + 0.001)

    Why min: we want to CHASE the bid DOWN as the market moves away from
    the trigger.  If patient = $0.384 and bid = $0.20 (price collapsed),
    the legacy patient price is unfillable (sits as an ask far above bid).
    Posting at bid+tick = $0.201 makes us the lowest ask, so a small
    upward tick from any buyer will lift us.  We accept a worse fill price
    in exchange for actually filling — exactly the trade-off a stop-loss
    requires when the thesis is broken.

    If patient < bid+tick (price has moved UP since the trigger fired),
    posting at patient becomes marketable — the order fills immediately
    against the resting bid at the bid price (which is HIGHER than our
    patient limit).  We capture the better price.  Polymarket's matching
    engine fills marketable orders at best execution, not at the limit.

    When `current_best_bid` is None (book unreachable), the legacy
    behavior is preserved: pure intended × multiplier.  This is the safe
    degradation path — we'd rather post a stale-but-valid limit than
    skip the rung entirely on a transient API failure.

    The bleed circuit-breaker in monitor._advance_exit_ladders handles
    the catastrophic case (>15% drop): force cross-spread immediately
    rather than walking the ladder at all.
    """
    if retry_count < 0:
        raise ValueError(f"retry_count must be >= 0, got {retry_count}")
    if retry_count >= MAX_LADDER_RETRIES:
        return None
    multiplier = SELL_LADDER_MULTIPLIERS[retry_count]
    patient_price = intended_price * multiplier
    if current_best_bid is not None and current_best_bid > 0:
        # Chase: never post above bid+tick (an unfillable ask far above
        # the touch).  Patient is the *ceiling*; bid+tick is the *floor*
        # at which we'd be most fillable.  min() picks the lower
        # (= more aggressive sell, more likely to fill).
        bid_anchored = current_best_bid + 0.001
        price = min(patient_price, bid_anchored)
    else:
        price = patient_price
    # Polymarket's price quantum is 0.001; round defensively to avoid
    # rejections on overly precise prices.
    return round(price, 4)


def is_ladder_exhausted(retry_count: int) -> bool:
    """True iff the next attempt should cross the spread instead of using
    a ladder rung.  Equivalent to ladder_price(...) is None for that count."""
    return retry_count >= MAX_LADDER_RETRIES


def cross_spread_price(best_bid: float) -> float:
    """Return the marketable-limit sell price given the current best bid.

    Posting at exactly the best bid would queue behind existing bids at
    that level (price-time priority).  Posting slightly below ensures
    we cross the spread and fill against the existing best bid.

    Subtract 1 tick (Polymarket quantum is 0.001).
    """
    if best_bid <= 0:
        raise ValueError(f"best_bid must be > 0, got {best_bid}")
    price = best_bid - 0.001
    # Floor at 0.001 so we never post a non-positive price
    return round(max(price, 0.001), 4)
