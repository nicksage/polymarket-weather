"""
trailing_stop.py — Shared tiered-trailing-stop evaluator.

Single source of truth for trailing-stop decisions, called from BOTH the
real-time WebSocket exit path (realtime_exits.py) and the strategy scan
exit path (market_price_value.py::_classify_position).  Pre-this-module,
the two paths each had their own copy of the trail logic — easy to drift
when one was tuned and the other forgotten.

The evaluator is a PURE FUNCTION: no DB I/O, no side effects, no logging.
Trivial to unit-test.  Callers handle DB reads/writes for peak persistence
and order placement themselves.

Tier semantics
--------------
* Tiers are half-open intervals [lo, hi); the LAST tier's hi is treated
  inclusively so a peak of 1.0 still resolves.
* The trail percentage is always looked up from the PEAK, never the current
  price.  This is what makes the trail monotonically tighten as a position
  matures (peak only goes up, trail % only gets smaller).
* The peak update happens unconditionally (every tick), independent of
  whether the activation gate has been crossed.  Otherwise an early high
  reached before activation is forgotten and the trail later kicks in from
  a lower-than-actual peak.

Activation gate
---------------
The trail does nothing until peak has risen at least `activation_gain`
above entry.  Without this gate, a position entered at 0.30 could be
"in the trail" the very next tick and stop out on a 0.5¢ wiggle.
"""

from __future__ import annotations

from typing import Iterable


# ---------------------------------------------------------------------------
# Tier lookup
# ---------------------------------------------------------------------------

def lookup_trail_pct(
    peak: float,
    tiers: Iterable[tuple[float, float, float]],
) -> float | None:
    """Return the trail percentage for a peak in the given tier table.

    Half-open intervals [lo, hi).  The LAST tier's upper bound is
    inclusive — that handles the peak-equals-1.0 edge case.

    Returns None if the peak falls outside every configured tier (which
    shouldn't happen if tiers partition [0, 1] per the validator, but is
    defensive against a misconfigured tier list).
    """
    tiers = list(tiers)
    last_idx = len(tiers) - 1
    for i, (lo, hi, pct) in enumerate(tiers):
        in_range = (lo <= peak < hi) or (i == last_idx and peak == hi)
        if in_range:
            return pct
    return None


# ---------------------------------------------------------------------------
# Peak update
# ---------------------------------------------------------------------------

def update_peak(current_peak: float, new_price: float, entry_price: float) -> float:
    """Return max(current_peak, new_price, entry_price).

    The entry_price floor matters for restart safety: if a process is
    restarted with no in-memory peak cache and a stale DB row (e.g.,
    peak_price = NULL on a row from before this column existed), we
    must never let "peak" fall below entry — that would arm the trail
    against a phantom historical low.

    This function is intentionally trivial; callers may inline if they
    prefer.  Centralizing it makes the invariant explicit.
    """
    return max(float(current_peak), float(new_price), float(entry_price))


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_trailing_stop(
    entry_price: float,
    peak_price: float,
    current_price: float,
    *,
    activation_gain: float,
    tiers: Iterable[tuple[float, float, float]],
) -> tuple[float, str] | None:
    """Pure trailing-stop evaluator.

    The CALLER is responsible for updating peak_price BEFORE calling this
    (use update_peak() above and persist to DB).  We separate peak update
    from exit evaluation because:
      * peak update happens unconditionally every tick
      * exit evaluation is a downstream decision that may early-return at
        the activation gate

    Returns:
        (stop_level, "TRAILING_STOP") if the trail fires, else None.

    The returned stop_level is the price at which the trail conceptually
    fires — useful for the exit reason string and for backtesting.  In
    LIVE TRADING the actual fill price will likely be worse than this
    because of slippage; capture the real fill from the order response.
    """
    # Activation gate — trail is disarmed until peak has risen far enough
    # above entry.  Critical for entry-bar noise.
    if peak_price < entry_price + activation_gain:
        return None

    # Find which tier the PEAK falls in (not current price).  Trail %
    # therefore depends only on how high we've been, not where we are now.
    trail_pct = lookup_trail_pct(peak_price, tiers)
    if trail_pct is None:
        return None  # peak outside all configured tiers — defensive

    # Stop level is recomputed every tick because peak may have risen this
    # tick AND the tier may have changed.  Both inputs can move on a single
    # call, so caching stop_level on the position would be a bug.
    stop_level = peak_price * (1.0 - trail_pct)

    # Fire if current price has fallen to or below the stop.
    if current_price <= stop_level:
        return (stop_level, "TRAILING_STOP")
    return None
