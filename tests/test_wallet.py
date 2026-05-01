"""
test_wallet.py — Wallet balance sync unit tests.

Covers:
  * Paper mode → returns INITIAL_BANKROLL unchanged
  * Live mode + wallet has more than INITIAL_BANKROLL → cap at INITIAL_BANKROLL
  * Live mode + wallet has less than INITIAL_BANKROLL - reserve → cap at wallet
  * Live mode + balance fetch fails → fall back to INITIAL_BANKROLL (defensive)
  * Cache: second call within TTL doesn't re-query
  * Cache: clear_cache() forces re-query
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

import wallet
import config


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts with a clean wallet cache and known config values."""
    wallet.clear_cache()
    yield
    wallet.clear_cache()


def _mock_client_with_balance(usdc_amount: float):
    """Build a mock CLOB client whose get_balance_allowance returns the given
    USDC balance.  Polymarket returns USDC in micro-units (1e6 = 1 USDC)."""
    client = MagicMock()
    client.get_balance_allowance.return_value = {
        "balance": str(int(usdc_amount * 1_000_000)),
        "allowance": "999999999999",
    }
    return client


def _mock_client_failing():
    """A mock client whose balance call raises."""
    client = MagicMock()
    client.get_balance_allowance.side_effect = RuntimeError("CLOB unreachable")
    return client


# ===========================================================================
# Paper mode + check-disabled paths
# ===========================================================================

def test_paper_mode_returns_bankroll_unchanged(monkeypatch):
    monkeypatch.setattr(wallet, "PAPER_TRADE", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    client = _mock_client_with_balance(50.0)   # would otherwise scale down
    assert wallet.get_effective_bankroll(client) == 1000.0
    # Confirm we didn't even ask the client
    client.get_balance_allowance.assert_not_called()


def test_check_disabled_returns_bankroll_unchanged(monkeypatch):
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", False)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    client = _mock_client_with_balance(50.0)
    assert wallet.get_effective_bankroll(client) == 1000.0
    client.get_balance_allowance.assert_not_called()


def test_no_client_returns_bankroll_unchanged(monkeypatch):
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    assert wallet.get_effective_bankroll(None) == 1000.0


# ===========================================================================
# Live mode — capping logic
# ===========================================================================

def test_wallet_richer_than_config_returns_config(monkeypatch):
    """Wallet has $5000 but INITIAL_BANKROLL=$1000 → cap at $1000."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_RESERVE_USDC", 10.0)

    client = _mock_client_with_balance(5000.0)
    assert wallet.get_effective_bankroll(client) == 1000.0


def test_wallet_poorer_than_config_returns_wallet_minus_reserve(monkeypatch):
    """Wallet has $200, reserve $10, config $1000 → cap at $190."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_RESERVE_USDC", 10.0)

    client = _mock_client_with_balance(200.0)
    assert wallet.get_effective_bankroll(client) == pytest.approx(190.0)


def test_wallet_below_reserve_returns_zero(monkeypatch):
    """Wallet has $5, reserve $10 → effective is 0, not negative."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_RESERVE_USDC", 10.0)

    client = _mock_client_with_balance(5.0)
    assert wallet.get_effective_bankroll(client) == 0.0


def test_wallet_exactly_at_reserve_returns_zero(monkeypatch):
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_RESERVE_USDC", 10.0)

    client = _mock_client_with_balance(10.0)
    assert wallet.get_effective_bankroll(client) == 0.0


# ===========================================================================
# Failure handling
# ===========================================================================

def test_balance_fetch_fails_returns_bankroll(monkeypatch, caplog):
    """If CLOB call raises, fall back to INITIAL_BANKROLL (defensive — better
    to size against stale config than refuse to trade)."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)

    client = _mock_client_failing()
    result = wallet.get_effective_bankroll(client)
    assert result == 1000.0


def test_unexpected_response_shape_returns_none_then_bankroll(monkeypatch):
    """Defensive: malformed CLOB response → None from raw fetch → fall back."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)

    client = MagicMock()
    client.get_balance_allowance.return_value = {"unexpected": "shape"}
    result = wallet.get_effective_bankroll(client)
    assert result == 1000.0


# ===========================================================================
# Caching
# ===========================================================================

def test_cache_avoids_duplicate_query_within_ttl(monkeypatch):
    """Two consecutive calls should hit the API only once."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_REFRESH_MIN", 30.0)

    client = _mock_client_with_balance(500.0)
    wallet.get_effective_bankroll(client)
    wallet.get_effective_bankroll(client)
    wallet.get_effective_bankroll(client)
    # 1 call, not 3
    assert client.get_balance_allowance.call_count == 1


def test_clear_cache_forces_requery(monkeypatch):
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_REFRESH_MIN", 30.0)

    client = _mock_client_with_balance(500.0)
    wallet.get_effective_bankroll(client)
    wallet.clear_cache()
    wallet.get_effective_bankroll(client)
    assert client.get_balance_allowance.call_count == 2


def test_failed_fetch_does_not_poison_cache(monkeypatch):
    """A failed fetch should NOT cache None — next call should retry."""
    monkeypatch.setattr(wallet, "PAPER_TRADE", False)
    monkeypatch.setattr(wallet, "WALLET_BALANCE_CHECK_ENABLED", True)
    monkeypatch.setattr(wallet, "INITIAL_BANKROLL", 1000.0)

    client = _mock_client_failing()
    wallet.get_effective_bankroll(client)   # fails
    wallet.get_effective_bankroll(client)   # should retry, not use stale None
    # 2 attempts because the first failed and was not cached
    assert client.get_balance_allowance.call_count == 2


# ===========================================================================
# Raw balance fetch
# ===========================================================================

def test_get_wallet_usdc_balance_paper_mode_returns_none():
    """No client → None."""
    assert wallet.get_wallet_usdc_balance(None) is None


def test_get_wallet_usdc_balance_parses_micro_units():
    """USDC has 6 decimals; balance string '12500000' should become $12.50."""
    client = MagicMock()
    client.get_balance_allowance.return_value = {"balance": "12500000"}
    assert wallet.get_wallet_usdc_balance(client) == pytest.approx(12.50)
