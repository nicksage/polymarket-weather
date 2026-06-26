"""
test_forecast_confidence.py — Phase: deterministic+confidence add-on
to twc_forecast_probe.py.

Lock the load-bearing math:
  - Option A confidence = P(bin containing det_max under half-up rounding)
  - P50 fallback for when daily/15day endpoint isn't entitled
  - Both response shapes (flat wrapper vs nested-by-product-name)
    handled by the deterministic-fetch parser
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.twc_forecast_probe import (    # type: ignore
    compute_forecast_agreement_confidence,
    deterministic_max_from_p50,
    fetch_deterministic_daily_max,
)


def _bins_us_88_to_94():
    """Standard US 2°F bins from 88-89 through 94-95."""
    return [
        {"label": "88-89°F", "range_low": 88, "range_high": 89},
        {"label": "90-91°F", "range_low": 90, "range_high": 91},
        {"label": "92-93°F", "range_low": 92, "range_high": 93},
        {"label": "94-95°F", "range_low": 94, "range_high": 95},
    ]


def _bins_intl_25_to_30():
    """1°C international bins."""
    return [
        {"label": "25°C", "range_low": 25, "range_high": 25},
        {"label": "26°C", "range_low": 26, "range_high": 26},
        {"label": "27°C", "range_low": 27, "range_high": 27},
        {"label": "28°C", "range_low": 28, "range_high": 28},
        {"label": "29°C", "range_low": 29, "range_high": 29},
        {"label": "30°C", "range_low": 30, "range_high": 30},
    ]


# ============================================================
# compute_forecast_agreement_confidence — Option A
# ============================================================

class TestConfidence:

    def test_det_in_mode_bin_high_confidence(self):
        """Det forecast lands in the highest-prob bin → conf = mode prob."""
        bins = _bins_us_88_to_94()
        probs = {"88-89°F": 0.10, "90-91°F": 0.65,
                 "92-93°F": 0.20, "94-95°F": 0.05}
        out = compute_forecast_agreement_confidence(90.4, probs, bins)
        # 90.4 → rounds to 90 → bin 90-91°F → conf = 0.65
        assert abs(out["confidence"] - 0.65) < 1e-9
        assert out["det_bin_label"] == "90-91°F"
        assert out["mode_bin_label"] == "90-91°F"
        assert out["det_rounded"] == 90

    def test_det_in_off_mode_bin_low_confidence(self):
        """Det lands one bin away from mode → conf = neighbor's prob."""
        bins = _bins_us_88_to_94()
        probs = {"88-89°F": 0.10, "90-91°F": 0.65,
                 "92-93°F": 0.20, "94-95°F": 0.05}
        out = compute_forecast_agreement_confidence(92.4, probs, bins)
        # 92.4 → rounds to 92 → bin 92-93°F → conf = 0.20
        assert abs(out["confidence"] - 0.20) < 1e-9
        assert out["det_bin_label"] == "92-93°F"
        assert out["mode_bin_label"] == "90-91°F"

    def test_half_up_rounding_at_bin_boundary(self):
        """89.5 rounds half-up to 90 → bin 90-91°F (not 88-89°F)."""
        bins = _bins_us_88_to_94()
        probs = {"88-89°F": 0.40, "90-91°F": 0.40,
                 "92-93°F": 0.15, "94-95°F": 0.05}
        out = compute_forecast_agreement_confidence(89.5, probs, bins)
        assert out["det_rounded"] == 90
        assert out["det_bin_label"] == "90-91°F"

    def test_single_temp_intl_bin(self):
        """1°C international bin: det=27.3 → rounds to 27 → 27°C bin."""
        bins = _bins_intl_25_to_30()
        probs = {"25°C": 0.05, "26°C": 0.15, "27°C": 0.50,
                 "28°C": 0.20, "29°C": 0.08, "30°C": 0.02}
        out = compute_forecast_agreement_confidence(27.3, probs, bins)
        assert out["det_rounded"] == 27
        assert out["det_bin_label"] == "27°C"
        assert abs(out["confidence"] - 0.50) < 1e-9

    def test_open_ended_upper_bin(self):
        bins = [
            {"label": "88-89°F", "range_low": 88, "range_high": 89},
            {"label": "≥90°F",   "range_low": 90, "range_high": None},
        ]
        probs = {"88-89°F": 0.40, "≥90°F": 0.60}
        out = compute_forecast_agreement_confidence(95.0, probs, bins)
        assert out["det_bin_label"] == "≥90°F"
        assert abs(out["confidence"] - 0.60) < 1e-9

    def test_open_ended_lower_bin(self):
        bins = [
            {"label": "≤25°C", "range_low": None, "range_high": 25},
            {"label": "26°C",  "range_low": 26,    "range_high": 26},
        ]
        probs = {"≤25°C": 0.70, "26°C": 0.30}
        out = compute_forecast_agreement_confidence(22.0, probs, bins)
        assert out["det_bin_label"] == "≤25°C"
        assert abs(out["confidence"] - 0.70) < 1e-9

    def test_det_outside_all_closed_bins_returns_zero(self):
        """If det rounds outside every bin (no open-ended catch-all),
        confidence is 0 — flags a synthesized-bins coverage problem."""
        bins = _bins_us_88_to_94()
        probs = {"88-89°F": 0.25, "90-91°F": 0.25,
                 "92-93°F": 0.25, "94-95°F": 0.25}
        out = compute_forecast_agreement_confidence(100.0, probs, bins)
        assert out["confidence"] == 0.0
        assert out["det_bin_label"] is None
        # Mode info still computed
        assert out["mode_bin_label"] is not None

    def test_empty_inputs_return_zero(self):
        out = compute_forecast_agreement_confidence(90.0, {}, [])
        assert out["confidence"] == 0.0
        out = compute_forecast_agreement_confidence(
            None, {"x": 1.0}, [{"label": "x", "range_low": 0, "range_high": 1}])
        assert out["confidence"] == 0.0


# ============================================================
# deterministic_max_from_p50 — fallback
# ============================================================

class TestP50Fallback:

    def test_empty_returns_none(self):
        assert deterministic_max_from_p50([]) is None

    def test_p50_is_median(self):
        # 5 samples, median = 3rd (index 2 after sort... but len//2 = 2)
        # values 10,20,30,40,50 → [10,20,30,40,50] → index 2 → 30
        assert deterministic_max_from_p50([30, 10, 50, 20, 40]) == 30

    def test_p50_even_length(self):
        # 4 samples → len//2 = 2 → upper of the two middles
        # values [10,20,30,40] → index 2 → 30
        assert deterministic_max_from_p50([10, 20, 30, 40]) == 30


# ============================================================
# fetch_deterministic_daily_max — response-shape parsing
# ============================================================

def _mock_httpx_response(status_code: int, json_body: dict | None = None,
                          text_body: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("no json")
    resp.text = text_body or json.dumps(json_body or {})
    return resp


class TestFetchDeterministic:

    def test_flat_shape_picks_event_date(self):
        body = {
            "temperatureMax":   [85.0, 91.0, 88.0],
            "validTimeLocal":   ["2026-06-25T07:00:00-0400",
                                  "2026-06-26T07:00:00-0400",
                                  "2026-06-27T07:00:00-0400"],
            "narrative":        ["a", "b", "c"],
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_httpx_response(200, body)):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "ok"
        assert out["today_max"] == 91.0
        assert out["narrative"] == "b"

    def test_nested_shape_picks_event_date(self):
        body = {
            "forecastDaily15Day": {
                "temperatureMax":   [85.0, 91.0],
                "validTimeLocal":   ["2026-06-25T07:00:00-0400",
                                      "2026-06-26T07:00:00-0400"],
            }
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_httpx_response(200, body)):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "ok"
        assert out["today_max"] == 91.0

    def test_event_date_not_in_response(self):
        body = {
            "temperatureMax":  [85.0],
            "validTimeLocal":  ["2026-06-25T07:00:00-0400"],
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_httpx_response(200, body)):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "no_data"
        assert out["today_max"] is None

    def test_temperatureMax_null_post_peak(self):
        """After today's peak, TWC drops the entry (sets to null)."""
        body = {
            "temperatureMax":  [None, 91.0],
            "validTimeLocal":  ["2026-06-26T07:00:00-0400",
                                 "2026-06-27T07:00:00-0400"],
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_httpx_response(200, body)):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "no_data"
        assert out["today_max"] is None
        assert "post-peak" in (out.get("err") or "")

    def test_403_returns_not_entitled(self):
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_httpx_response(403, None,
                                                       "forbidden")):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "not_entitled"
        assert out["today_max"] is None

    def test_no_api_key(self):
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", ""):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "error"
        assert "TWC_API_KEY" in (out["err"] or "")

    def test_network_error_returns_error_status(self):
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    side_effect=RuntimeError("connection refused")):
            out = fetch_deterministic_daily_max(
                "KMIA", "fahrenheit", "2026-06-26", "America/New_York")
        assert out["status"] == "error"
        assert "connection refused" in (out["err"] or "")