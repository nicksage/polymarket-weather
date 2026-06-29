"""
test_15min_nowcast.py — TWC 15-min nowcast intraday-confirmation
add-on to twc_forecast_probe.py.

Lock the load-bearing math:
  - fetch_15min_peak parses both flat and nested response shapes
  - filters slots to event_date in station-local time
  - skips null temp entries (post-peak drop case)
  - compute_intraday_agreement floors forecasts by observed
  - verdict tiers per settlement unit (US 2°F vs intl 1°C)
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
    fetch_15min_peak,
    compute_intraday_agreement,
)


def _mock_resp(status_code: int, body: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    if body is not None:
        resp.json.return_value = body
    else:
        resp.json.side_effect = ValueError("no json")
    resp.text = text or json.dumps(body or {})
    return resp


# ============================================================
# fetch_15min_peak — response-shape parsing + event_date filtering
# ============================================================

class TestFetch15MinPeak:

    def test_flat_shape_picks_peak_on_event_date(self):
        # 5 slots: 2 yesterday (out), 3 today (in)
        body = {
            "temperature": [82, 83, 85, 88, 87],
            "validTimeLocal": [
                "2026-06-28T23:00:00-0400",
                "2026-06-28T23:15:00-0400",
                "2026-06-29T13:00:00-0400",
                "2026-06-29T15:00:00-0400",
                "2026-06-29T15:15:00-0400",
            ],
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_resp(200, body)):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "ok"
        assert out["peak_temp"] == 88
        assert out["peak_local_hour"] == 15
        assert out["n_slots_today"] == 3
        assert abs(out["horizon_hours"] - 0.75) < 1e-9

    def test_nested_shape(self):
        body = {
            "forecastFifteenMinute": {
                "temperature": [88, 90, 87],
                "validTimeLocal": [
                    "2026-06-29T13:00:00-0400",
                    "2026-06-29T15:00:00-0400",
                    "2026-06-29T17:00:00-0400",
                ],
            }
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_resp(200, body)):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "ok"
        assert out["peak_temp"] == 90

    def test_null_temp_slots_skipped(self):
        body = {
            "temperature": [None, 88, None, 91, None],
            "validTimeLocal": [
                "2026-06-29T13:00:00-0400",
                "2026-06-29T14:00:00-0400",
                "2026-06-29T15:00:00-0400",
                "2026-06-29T16:00:00-0400",
                "2026-06-29T17:00:00-0400",
            ],
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_resp(200, body)):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "ok"
        assert out["peak_temp"] == 91
        assert out["n_slots_today"] == 2   # only 2 non-null today

    def test_event_date_outside_horizon(self):
        """Tomorrow's date — all 28 slots are today; nothing matches."""
        body = {
            "temperature": [88, 90],
            "validTimeLocal": [
                "2026-06-29T15:00:00-0400",
                "2026-06-29T17:00:00-0400",
            ],
        }
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_resp(200, body)):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-30", "America/New_York")
        assert out["status"] == "no_data"
        assert out["peak_temp"] is None
        assert "no slots" in (out["err"] or "")

    def test_403_not_entitled(self):
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_resp(403, None, "forbidden")):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "not_entitled"
        assert out["peak_temp"] is None

    def test_no_api_key(self):
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", ""):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "error"
        assert "TWC_API_KEY" in (out["err"] or "")

    def test_network_error(self):
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    side_effect=RuntimeError("connection refused")):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "error"
        assert "connection refused" in (out["err"] or "")

    def test_empty_temperature_array(self):
        body = {"temperature": [], "validTimeLocal": []}
        with patch("scripts.twc_forecast_probe.TWC_API_KEY", "fake"), \
             patch("scripts.twc_forecast_probe.httpx.get",
                    return_value=_mock_resp(200, body)):
            out = fetch_15min_peak(
                "KMIA", "fahrenheit", "2026-06-29", "America/New_York")
        assert out["status"] == "no_data"


# ============================================================
# compute_intraday_agreement — 3-way verdict
# ============================================================

class TestIntradayAgreement:

    def test_tight_agreement_us(self):
        # nowcast=90, prob=89.5, observed=87 → spread = 0.5°F → tight
        out = compute_intraday_agreement(
            nowcast_peak=90.0, probabilistic_p50=89.5,
            observed_max_so_far=87.0, settlement_unit="fahrenheit")
        assert out["spread"] == 0.5
        assert out["verdict"] == "tight"
        assert out["tight_threshold"] == 1.0

    def test_loose_agreement_us(self):
        # nowcast=90, prob=88 → spread = 2.0 → loose (== 2*tight)
        out = compute_intraday_agreement(
            nowcast_peak=90.0, probabilistic_p50=88.0,
            observed_max_so_far=85.0, settlement_unit="fahrenheit")
        assert out["spread"] == 2.0
        assert out["verdict"] == "loose"

    def test_diverged_us(self):
        # nowcast=92, prob=88 → spread = 4.0 → diverged
        out = compute_intraday_agreement(
            nowcast_peak=92.0, probabilistic_p50=88.0,
            observed_max_so_far=85.0, settlement_unit="fahrenheit")
        assert out["spread"] == 4.0
        assert out["verdict"] == "diverged"

    def test_intl_tighter_threshold(self):
        # Celsius: tight = 0.5
        # nowcast=29, prob=28.6 → spread = 0.4 → tight
        out = compute_intraday_agreement(
            nowcast_peak=29.0, probabilistic_p50=28.6,
            observed_max_so_far=27.0, settlement_unit="celsius")
        assert out["tight_threshold"] == 0.5
        assert out["verdict"] == "tight"

    def test_intl_diverged_at_smaller_spread(self):
        # Celsius: diverged when > 1.0
        # nowcast=29, prob=27.5 → spread = 1.5 → diverged
        out = compute_intraday_agreement(
            nowcast_peak=29.0, probabilistic_p50=27.5,
            observed_max_so_far=26.0, settlement_unit="celsius")
        assert out["spread"] == 1.5
        assert out["verdict"] == "diverged"

    def test_observed_floors_both_forecasts(self):
        """When observed already exceeds both forecasts, both get floored
        to observed — spread becomes 0."""
        out = compute_intraday_agreement(
            nowcast_peak=88.0, probabilistic_p50=87.5,
            observed_max_so_far=91.0, settlement_unit="fahrenheit")
        assert out["nowcast_adj"] == 91.0
        assert out["prob_adj"] == 91.0
        assert out["spread"] == 0.0
        assert out["verdict"] == "tight"

    def test_observed_floors_only_lower_forecast(self):
        """Observed beats prob (87) but not nowcast (92).  Spread should
        be |92 - max(87, 87)| = 5.0 not |92 - 87| = 5.0 (same here),
        but the key behaviour: prob gets floored, nowcast doesn't."""
        out = compute_intraday_agreement(
            nowcast_peak=92.0, probabilistic_p50=86.0,
            observed_max_so_far=87.0, settlement_unit="fahrenheit")
        # prob_adj = max(86, 87) = 87; nowcast_adj = max(92, 87) = 92
        assert out["nowcast_adj"] == 92.0
        assert out["prob_adj"] == 87.0
        assert out["spread"] == 5.0
        assert out["verdict"] == "diverged"

    def test_no_observed_no_floor(self):
        out = compute_intraday_agreement(
            nowcast_peak=90.0, probabilistic_p50=89.0,
            observed_max_so_far=None, settlement_unit="fahrenheit")
        assert out["nowcast_adj"] == 90.0
        assert out["prob_adj"] == 89.0
        assert out["spread"] == 1.0
        assert out["verdict"] == "tight"

    def test_no_nowcast_returns_no_nowcast(self):
        out = compute_intraday_agreement(
            nowcast_peak=None, probabilistic_p50=89.0,
            observed_max_so_far=85.0, settlement_unit="fahrenheit")
        assert out["verdict"] == "no_nowcast"
        assert out["spread"] is None
        assert out["nowcast_adj"] is None

    def test_no_prob_p50_returns_no_nowcast(self):
        """If probabilistic P50 is missing too, can't compute agreement."""
        out = compute_intraday_agreement(
            nowcast_peak=90.0, probabilistic_p50=None,
            observed_max_so_far=85.0, settlement_unit="fahrenheit")
        assert out["spread"] is None
        assert out["verdict"] == "no_nowcast"