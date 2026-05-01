"""
test_geoblock_circuit.py — Regression tests for the geoblock circuit breaker.

Polymarket geoblocks US (and some other) IPs from order placement.  When
detected, every subsequent order placement returns 403 with the same
"Trading restricted in your region" message.  Without a circuit breaker,
the bot would spam the log + waste API budget hammering the rejected
endpoint until the operator notices.

Fix (2026-04-30): module-level breaker that:
  1. Detects 403 with geoblock signature in the error text.
  2. Trips on first detection — logs ONCE with recovery instructions.
  3. Short-circuits subsequent placement attempts (no API call).
  4. Stays tripped for the process lifetime — only resets on restart
     (which is when the operator would have moved networks/VPN).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db
import execution


@pytest.fixture(autouse=True)
def reset_breaker():
    """Each test starts with the breaker reset."""
    execution._reset_geoblock_circuit()
    yield
    execution._reset_geoblock_circuit()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


# ===========================================================================
# Geoblock detection — exception classifier
# ===========================================================================

@pytest.mark.parametrize("exc_text", [
    "Trading restricted in your region, please refer to available regions",
    "PolyApiException[status_code=403, error_message={'error': 'Trading restricted in your region'}]",
    "PolyApiException[status_code=403, ...geoblock...]",
    "TRADING RESTRICTED IN YOUR REGION",   # case-insensitive
])
def test_is_geoblock_error_recognizes_signature(exc_text):
    assert execution._is_geoblock_error(Exception(exc_text)) is True


@pytest.mark.parametrize("exc_text", [
    "PolyApiException[status_code=400, error_message={'error': 'order_version_mismatch'}]",
    "PolyApiException[status_code=429, error_message={'error': 'rate limited'}]",
    "Connection refused",
    "Random network error",
])
def test_is_geoblock_error_doesnt_misfire_on_unrelated(exc_text):
    assert execution._is_geoblock_error(Exception(exc_text)) is False


# ===========================================================================
# Tripping the breaker
# ===========================================================================

def test_geoblock_breaker_starts_off():
    assert execution._geoblock_tripped is False


def test_trip_geoblock_circuit_sets_flag():
    execution._trip_geoblock_circuit(Exception("Trading restricted in your region"))
    assert execution._geoblock_tripped is True


def test_trip_geoblock_circuit_logs_only_once(caplog):
    """Repeated trips must NOT spam the ERROR log — once is enough."""
    import logging
    caplog.set_level(logging.ERROR, logger="execution")

    execution._trip_geoblock_circuit(Exception("Trading restricted"))
    first_count = sum("GEOBLOCK" in r.message for r in caplog.records)

    execution._trip_geoblock_circuit(Exception("Trading restricted"))
    execution._trip_geoblock_circuit(Exception("Trading restricted"))
    second_count = sum("GEOBLOCK" in r.message for r in caplog.records)

    assert first_count == 1
    assert second_count == 1, "the breaker should log only on the first trip"


def test_short_circuited_response_shape():
    r = execution._short_circuited_response()
    assert r["status"] == "skip"
    assert r["reason"] == "geoblock_circuit_tripped"


# ===========================================================================
# Integration: execute_signal short-circuits when breaker is tripped
# ===========================================================================

def test_execute_signal_skips_api_call_when_breaker_tripped(temp_db, monkeypatch):
    """Once tripped, execute_signal returns the short-circuit response
    without ever hitting the CLOB client."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    execution._trip_geoblock_circuit(Exception("geoblock"))

    client = MagicMock()
    client.create_and_post_order.side_effect = AssertionError(
        "should not have hit the CLOB after breaker tripped"
    )

    sig = {
        "contract_id": "0xabc", "recommended_side": "YES",
        "kelly_size": 10.0, "yes_token_id": "tok_yes", "no_token_id": "tok_no",
        "yes_price": 0.30, "market_p": 0.30,
        "city": "Test", "date": "2026-05-01",
        "scan_timestamp": "2026-04-30T12:00:00+00:00",
    }
    result = execution.execute_signal(sig, client=client)
    assert result["status"] == "skip"
    assert result["reason"] == "geoblock_circuit_tripped"
    client.create_and_post_order.assert_not_called()


def test_execute_signal_trips_breaker_on_first_geoblock_response(temp_db, monkeypatch):
    """The first time a geoblock 403 surfaces from the CLOB, the breaker
    trips so subsequent signals don't repeat the API call."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)

    class _GeoblockExc(Exception): ...

    client = MagicMock()
    client.create_and_post_order.side_effect = _GeoblockExc(
        "PolyApiException[status_code=403, error_message="
        "{'error': 'Trading restricted in your region'}]"
    )

    sig = {
        "contract_id": "0xabc", "recommended_side": "YES",
        "kelly_size": 10.0, "yes_token_id": "tok_yes", "no_token_id": "tok_no",
        "yes_price": 0.30, "market_p": 0.30,
        "city": "Test", "date": "2026-05-01",
        "scan_timestamp": "2026-04-30T12:00:00+00:00",
    }
    # Patch compute_sweep_limit to avoid an extra mock dance
    monkeypatch.setattr(execution, "compute_sweep_limit",
                        lambda **kw: (0.30, {"source": "test"}))

    assert execution._geoblock_tripped is False
    result = execution.execute_signal(sig, client=client)
    # Breaker tripped + short-circuit response
    assert execution._geoblock_tripped is True
    assert result["status"] == "skip"
    assert result["reason"] == "geoblock_circuit_tripped"


def test_non_geoblock_errors_dont_trip_breaker(temp_db, monkeypatch):
    """A 400 / rate-limit / network error must NOT trip the geoblock breaker
    — that would make the bot give up too easily on transient errors."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    client = MagicMock()
    client.create_and_post_order.side_effect = RuntimeError(
        "PolyApiException[status_code=400, error_message="
        "{'error': 'order_version_mismatch'}]"
    )
    monkeypatch.setattr(execution, "compute_sweep_limit",
                        lambda **kw: (0.30, {"source": "test"}))

    sig = {
        "contract_id": "0xabc", "recommended_side": "YES",
        "kelly_size": 10.0, "yes_token_id": "tok_yes", "no_token_id": "tok_no",
        "yes_price": 0.30, "market_p": 0.30,
        "city": "Test", "date": "2026-05-01",
        "scan_timestamp": "2026-04-30T12:00:00+00:00",
    }
    result = execution.execute_signal(sig, client=client)
    assert execution._geoblock_tripped is False
    # Returns 'error', not 'skip'
    assert result["status"] == "error"
