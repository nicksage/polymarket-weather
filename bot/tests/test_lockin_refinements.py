"""
test_lockin_refinements.py — Phase 5 (critic issue #5).

Pin the math for the three lock-in refinements:
  5a. compute_lockin_probability — distribution-driven trigger
  5b. boundary_distance         — soft-bin distance under half-up rounding
  5c. compute_net_edge          — EV against ask + fees

Plus the unified `lockin_gate` end-to-end.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.lockin_refinements import (   # type: ignore
    compute_lockin_probability,
    boundary_distance,
    compute_net_edge,
    lockin_gate,
    DEFAULT_FEE_RATE,
    _round_half_up,
)


# ============================================================
# Helper: half-up sanity (already covered elsewhere, smoke test only)
# ============================================================

class TestRoundHalfUp:
    def test_half_rounds_up(self):
        assert _round_half_up(88.5) == 89
        assert _round_half_up(89.5) == 90

    def test_below_half_rounds_down(self):
        assert _round_half_up(88.4) == 88
        assert _round_half_up(88.49) == 88


# ============================================================
# 5a. compute_lockin_probability
# ============================================================

class TestLockinProbability:

    def test_empty_samples_returns_zero(self):
        assert compute_lockin_probability([], 85.0, 88, 89) == 0.0

    def test_all_samples_below_observed_obs_drives_result(self):
        """All remaining-hour maxes are below observed_max_so_far.  Final
        max == observed_max_so_far for every realization."""
        samples = [80.0, 81.0, 82.0]
        # observed=88.4 → rounds to 88 → in [88, 89] bin
        p = compute_lockin_probability(samples, 88.4, 88, 89)
        assert p == 1.0

    def test_some_samples_exceed_bin_high(self):
        """Half the realizations spike past the upper bin edge."""
        # observed=88.4 → rounds to 88
        # samples: 50% below (final=88.4→88, in-bin), 50% spike to 90→out
        samples = [70.0, 71.0, 90.0, 91.0]
        p = compute_lockin_probability(samples, 88.4, 88, 89)
        assert p == 0.5

    def test_open_ended_upper_bin(self):
        """Bin '≥90°F' → low=90, high=None.  All values >=89.5 qualify."""
        samples = [85, 88, 90, 95, 100]
        # observed=89.4 → final = max(89.4, sample) = each sample (4 of 5)
        # but with obs=89.4 → final=89.4 for sample 85,88 → rounds to 89 → out
        # finals: 89.4,89.4,90,95,100 → rounds: 89,89,90,95,100
        # in-bin (>=90): 3 of 5 = 0.6
        p = compute_lockin_probability(samples, 89.4, 90, None)
        assert abs(p - 0.6) < 1e-9

    def test_open_ended_lower_bin(self):
        """Bin '≤30°C' → low=None, high=30.  All rounded values <=30 qualify."""
        samples = [28, 29, 30, 31, 32]
        # observed=27.0 (well below); finals = samples
        # rounded: 28,29,30,31,32 → in-bin <=30: 3 of 5 = 0.6
        p = compute_lockin_probability(samples, 27.0, None, 30)
        assert abs(p - 0.6) < 1e-9

    def test_observed_at_upper_boundary(self):
        """observed=89.5 rounds up to 90 → falls OUT of bin [88,89]
        even when no remaining sample exceeds it."""
        samples = [70, 71, 72]
        # observed=89.5 → rounds half-up to 90 → outside [88,89]
        p = compute_lockin_probability(samples, 89.5, 88, 89)
        assert p == 0.0

    def test_observed_at_lower_boundary(self):
        """observed=87.5 rounds half-up to 88 → in bin [88,89]."""
        samples = [70, 71, 72]
        p = compute_lockin_probability(samples, 87.5, 88, 89)
        assert p == 1.0

    def test_realistic_post_peak_scenario(self):
        """observed=88.2 well inside 88-89 bin; remaining samples mostly
        below.  Expect high lock-in probability."""
        # 95 samples below 88, 5 samples spike to 90+
        samples = [80.0] * 95 + [90.0, 91.0, 92.0, 93.0, 94.0]
        p = compute_lockin_probability(samples, 88.2, 88, 89)
        assert abs(p - 0.95) < 1e-9


# ============================================================
# 5b. boundary_distance
# ============================================================

class TestBoundaryDistance:

    def test_middle_of_2deg_bin(self):
        # Bin [88, 89] covers [87.5, 89.5); observed=88.4
        # dist to 87.5 = 0.9; dist to 89.5 = 1.1; min = 0.9
        d = boundary_distance(88.4, 88, 89)
        assert abs(d - 0.9) < 1e-9

    def test_near_upper_boundary(self):
        # observed=89.4, bin [88,89] → dist to 89.5 = 0.1
        d = boundary_distance(89.4, 88, 89)
        assert abs(d - 0.1) < 1e-9

    def test_near_lower_boundary(self):
        # observed=87.6, bin [88,89] → dist to 87.5 = 0.1
        d = boundary_distance(87.6, 88, 89)
        assert abs(d - 0.1) < 1e-9

    def test_single_temp_bin(self):
        # 1°F (or 1°C) bin: lo=hi=88, covers [87.5, 88.5); observed=88.0
        # dist to 87.5 = 0.5; dist to 88.5 = 0.5; min = 0.5
        d = boundary_distance(88.0, 88, 88)
        assert abs(d - 0.5) < 1e-9

    def test_open_upper_bin(self):
        # Bin ≥90°F: lo=90, hi=None; observed=92.0 → dist to 89.5 = 2.5
        d = boundary_distance(92.0, 90, None)
        assert abs(d - 2.5) < 1e-9

    def test_open_lower_bin(self):
        # Bin ≤30°C: lo=None, hi=30; observed=27.0 → dist to 30.5 = 3.5
        d = boundary_distance(27.0, None, 30)
        assert abs(d - 3.5) < 1e-9

    def test_both_open_returns_inf(self):
        d = boundary_distance(50.0, None, None)
        assert d == float("inf")

    def test_observed_outside_bin_returns_distance_to_nearest_edge(self):
        # observed=86.0, bin [88,89] (covers [87.5,89.5))
        # dist to 87.5 = 1.5; dist to 89.5 = 3.5; min = 1.5
        d = boundary_distance(86.0, 88, 89)
        assert abs(d - 1.5) < 1e-9


# ============================================================
# 5c. compute_net_edge
# ============================================================

class TestNetEdge:

    def test_strong_edge_at_low_ask(self):
        # p=0.98, ask=0.85, fee=10% on profit (0.15)
        # payoff if win = 1 - 0.10*0.15 = 0.985
        # ev = 0.98*0.985 - 0.85 = 0.9653 - 0.85 = 0.1153
        edge = compute_net_edge(0.98, 0.85, fee_rate=0.10)
        assert abs(edge - 0.1153) < 1e-4

    def test_marginal_edge_at_high_ask(self):
        # p=0.98, ask=0.95, fee=10% on profit (0.05)
        # payoff = 1 - 0.10*0.05 = 0.995
        # ev = 0.98*0.995 - 0.95 = 0.9751 - 0.95 = 0.0251
        edge = compute_net_edge(0.98, 0.95, fee_rate=0.10)
        assert abs(edge - 0.0251) < 1e-4

    def test_negative_edge(self):
        # p=0.70, ask=0.80 — clearly bad trade
        edge = compute_net_edge(0.70, 0.80, fee_rate=0.10)
        assert edge < 0

    def test_zero_fee_recovers_simple_ev(self):
        # With zero fee: ev = p*1 - ask
        edge = compute_net_edge(0.90, 0.80, fee_rate=0.0)
        assert abs(edge - 0.10) < 1e-9

    def test_ask_at_or_above_one(self):
        assert compute_net_edge(0.99, 1.0) == -float("inf")
        assert compute_net_edge(0.99, 1.01) == -float("inf")

    def test_ask_at_or_below_zero(self):
        assert compute_net_edge(0.99, 0.0) == -float("inf")
        assert compute_net_edge(0.99, -0.01) == -float("inf")

    def test_default_fee_is_ten_percent(self):
        assert DEFAULT_FEE_RATE == 0.10


# ============================================================
# Unified lockin_gate
# ============================================================

class TestLockinGate:

    def _samples_mostly_low(self):
        """95 realizations all <= 86, 5 spike to 90 — realistic
        post-peak distribution."""
        return [85.0] * 95 + [90.0, 91.0, 92.0, 93.0, 94.0]

    def test_clean_pass_through(self):
        # observed=88.2 well inside bin [88,89]; samples mostly below
        # ask=0.85, p_lockin = 0.95 → ev = 0.95 * 0.985 - 0.85 = 0.086
        out = lockin_gate(
            remaining_day_max_samples=self._samples_mostly_low(),
            observed_max_so_far=88.2,
            ask_price=0.85,
            target_bin_low=88, target_bin_high=89,
            min_lockin_prob=0.95,
            min_boundary_distance=0.5,
            min_net_edge=0.03,
            fee_rate=0.10,
        )
        assert out["pass"] is True
        assert out["failed_gates"] == []
        assert abs(out["lockin_prob"] - 0.95) < 1e-9
        assert out["boundary_dist"] > 0.5
        assert out["net_edge"] > 0.03

    def test_blocked_by_boundary(self):
        # observed=89.4 — only 0.1 from upper edge of [88,89]
        out = lockin_gate(
            remaining_day_max_samples=[80.0] * 100,
            observed_max_so_far=89.4,
            ask_price=0.50,
            target_bin_low=88, target_bin_high=89,
            min_boundary_distance=0.5,
            min_lockin_prob=0.95,
            min_net_edge=0.03,
            fee_rate=0.10,
        )
        assert out["pass"] is False
        assert any("boundary_dist" in g for g in out["failed_gates"])

    def test_blocked_by_lockin_prob(self):
        # 50/50 samples → P(in bin) ~ 0.5, below 0.95 threshold
        samples = [80.0] * 50 + [95.0] * 50
        out = lockin_gate(
            remaining_day_max_samples=samples,
            observed_max_so_far=88.2,
            ask_price=0.50,
            target_bin_low=88, target_bin_high=89,
            min_lockin_prob=0.95,
            min_boundary_distance=0.5,
            min_net_edge=0.03,
        )
        assert out["pass"] is False
        assert any("lockin_prob" in g for g in out["failed_gates"])

    def test_blocked_by_net_edge(self):
        # Strong lock-in, safe boundary, but ask is too high
        # observed=88.2, samples all stay below 89 → p≈1.0
        # ask=0.98 → ev = 1.0 * (1 - 0.10*0.02) - 0.98 = 0.998 - 0.98 = 0.018
        # < min_net_edge=0.03
        out = lockin_gate(
            remaining_day_max_samples=[80.0] * 100,
            observed_max_so_far=88.2,
            ask_price=0.98,
            target_bin_low=88, target_bin_high=89,
            min_lockin_prob=0.95,
            min_boundary_distance=0.5,
            min_net_edge=0.03,
            fee_rate=0.10,
        )
        assert out["pass"] is False
        assert any("net_edge" in g for g in out["failed_gates"])

    def test_multiple_failures_all_listed(self):
        # 50/50 lock-in (fails); near boundary (fails); high ask (fails)
        out = lockin_gate(
            remaining_day_max_samples=[80.0] * 50 + [95.0] * 50,
            observed_max_so_far=89.4,
            ask_price=0.99,
            target_bin_low=88, target_bin_high=89,
            min_lockin_prob=0.95,
            min_boundary_distance=0.5,
            min_net_edge=0.03,
        )
        assert out["pass"] is False
        assert len(out["failed_gates"]) == 3