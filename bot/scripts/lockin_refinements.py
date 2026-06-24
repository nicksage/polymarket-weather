"""
lockin_refinements.py — Phase 5 (critic issue #5).

Three pure helpers that refine the post-peak lock-in trade rule:

  5a. compute_lockin_probability(remaining_day_max_samples,
                                  observed_max_so_far, lo, hi)
      Distribution-driven trigger: instead of "now >= peak_hour + 1h",
      use the model's distribution over remaining-hours max to compute
      P(final_daily_max falls in target_bin).  Trigger when this
      probability exceeds the chosen confidence threshold.

  5b. boundary_distance(observed, lo, hi)
      Safety gate: distance (in settlement units) from observed value
      to the nearest soft bin boundary.  Under half-up rounding the
      bin [lo, hi] occupies the continuous range [lo - 0.5, hi + 0.5).
      A small margin from the boundary protects against late-day
      reading flips.

  5c. compute_net_edge(p_win, ask_price, fee_rate)
      Replaces `yes_price < threshold` with a real expected-value
      calculation against the ASK we'd actually pay, after Polymarket
      taker fees on winnings.

All three are settlement-unit agnostic — caller supplies values in
the same unit as the bin labels (°F for US markets, °C for
international).  No I/O.  No DB.  Pure math.
"""

from __future__ import annotations

from typing import Optional


# ============================================================
# Shared: half-up rounding (matches settlement convention)
# ============================================================

def _round_half_up(x: float) -> int:
    """Half-up rounding — matches Polymarket settlement (Phase 1 backtest
    confirmed 91.0% match vs 69.2% for truncation)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


# ============================================================
# 5a. Distribution-driven lock-in probability
# ============================================================

def compute_lockin_probability(
    remaining_day_max_samples: list[float],
    observed_max_so_far: float,
    target_bin_low: Optional[float],
    target_bin_high: Optional[float],
) -> float:
    """Return P(final_daily_max rounds into the target bin).

    For each forecast realization (prototype), the final daily max is
    max(observed_max_so_far, remaining_day_max_sample).  We round under
    the half-up convention and count what fraction lands in
    [target_bin_low, target_bin_high].  Open-ended bins supported
    (None on either side).

    Caller's responsibility: all values must be in the same unit as
    target_bin_low / target_bin_high (settlement unit).

    Empty samples → 0.0 (no evidence of lock-in)."""
    n = len(remaining_day_max_samples)
    if n == 0:
        return 0.0
    cnt = 0
    for r in remaining_day_max_samples:
        final = max(observed_max_so_far, r)
        rounded = _round_half_up(float(final))
        lo_ok = (target_bin_low is None) or (rounded >= target_bin_low)
        hi_ok = (target_bin_high is None) or (rounded <= target_bin_high)
        if lo_ok and hi_ok:
            cnt += 1
    return cnt / n


# ============================================================
# 5b. Boundary-distance gate
# ============================================================

def boundary_distance(
    observed: float,
    target_bin_low: Optional[float],
    target_bin_high: Optional[float],
) -> float:
    """Distance from `observed` to the nearest SOFT bin boundary.

    Under half-up rounding the bin [lo, hi] (integer labels) covers
    the continuous interval [lo - 0.5, hi + 0.5).  So a 1°F-wide bin
    labelled "88°F" (lo=hi=88) covers [87.5, 88.5); a 2°F-wide US bin
    labelled "88–89°F" (lo=88, hi=89) covers [87.5, 89.5).

    Returns the unsigned distance to the nearest edge.  For open-ended
    bins (None on one side) that side returns +inf.  If `observed`
    is OUTSIDE the bin, the value is still the distance to the nearest
    edge (positive — does not indicate inside/outside)."""
    if target_bin_low is None and target_bin_high is None:
        return float("inf")
    if target_bin_low is not None:
        lower_edge = float(target_bin_low) - 0.5
        dist_lo = abs(observed - lower_edge)
    else:
        dist_lo = float("inf")
    if target_bin_high is not None:
        upper_edge = float(target_bin_high) + 0.5
        dist_hi = abs(upper_edge - observed)
    else:
        dist_hi = float("inf")
    return min(dist_lo, dist_hi)


# ============================================================
# 5c. Net edge against ask + fees
# ============================================================

DEFAULT_FEE_RATE = 0.10   # Polymarket taker fee on winnings (10%)
                          # See bot/execution.py:1511 — observed in production


def compute_net_edge(
    p_win: float, ask_price: float, fee_rate: float = DEFAULT_FEE_RATE,
) -> float:
    """Expected value (in price units) of buying YES at `ask_price`.

    Polymarket charges the taker fee on the PROFIT (1 - ask), not the
    notional.  So:
        payoff_if_win  = 1.0 - fee_rate * (1.0 - ask_price)
        payoff_if_lose = 0.0
        cost           = ask_price
        net_edge       = p_win * payoff_if_win - cost

    Negative result means trading at this ask is unprofitable in
    expectation.  Use a positive `min_net_edge` floor (e.g. 0.03 = 3
    cents per $1 staked) as the trade gate."""
    if ask_price <= 0 or ask_price >= 1:
        return -float("inf")
    payoff = 1.0 - fee_rate * (1.0 - ask_price)
    return p_win * payoff - ask_price


# ============================================================
# Convenience: a unified gate check
# ============================================================

def lockin_gate(
    *,
    remaining_day_max_samples: list[float],
    observed_max_so_far: float,
    ask_price: float,
    target_bin_low: Optional[float],
    target_bin_high: Optional[float],
    min_lockin_prob: float = 0.95,
    min_boundary_distance: float = 0.5,
    min_net_edge: float = 0.03,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> dict:
    """Run all three Phase 5 gates and return a verdict dict.
        {
          "pass": bool,
          "lockin_prob":      float,
          "boundary_dist":    float,
          "net_edge":         float,
          "failed_gates":     list[str],   # which thresholds blocked us
        }
    Helps the caller (post_peak_strategy.evaluate_event or live
    predictor) make a single decision with the rationale attached.
    """
    p_win = compute_lockin_probability(
        remaining_day_max_samples,
        observed_max_so_far,
        target_bin_low, target_bin_high)
    bd = boundary_distance(
        observed_max_so_far, target_bin_low, target_bin_high)
    ne = compute_net_edge(p_win, ask_price, fee_rate=fee_rate)

    failed: list[str] = []
    if p_win < min_lockin_prob:
        failed.append(
            f"lockin_prob {p_win:.3f} < {min_lockin_prob:.2f}")
    if bd < min_boundary_distance:
        failed.append(
            f"boundary_dist {bd:.2f} < {min_boundary_distance:.2f}")
    if ne < min_net_edge:
        failed.append(
            f"net_edge {ne:.3f} < {min_net_edge:.3f}")
    return {
        "pass":           not failed,
        "lockin_prob":    p_win,
        "boundary_dist":  bd,
        "net_edge":       ne,
        "failed_gates":   failed,
    }