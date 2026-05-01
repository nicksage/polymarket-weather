"""
test_clob_creds_caching.py — Regression tests for cached-creds startup.

Why this test exists
--------------------
Polymarket's `/auth/api-key` endpoint sits behind Cloudflare and rate-
limits aggressively.  On 2026-04-29 the user hit a startup 403 because
the bot was calling `create_or_derive_api_key()` on every launch.

Fix: when CLOB_API_KEY/SECRET/PASSPHRASE are present in env,
get_clob_client() builds the client with cached creds directly via
ClobClient(creds=...) — no network call.  These tests pin that path
so a refactor that drops the cached path will fail loudly here, not
silently in production.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import execution


@pytest.fixture
def live_mode(monkeypatch):
    """Force live trading + provide a private key + wallet address."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xdeadbeef")
    import config
    monkeypatch.setattr(config, "WALLET_ADDRESS", "0xMyProxy")
    monkeypatch.setattr(config, "WALLET_SIGNATURE_TYPE", 2)


def test_cached_creds_skip_network_call(live_mode, monkeypatch):
    """When all three CLOB_API_* vars are set, ClobClient is built with
    creds=ApiCreds(...) and create_or_derive_api_key is NEVER called."""
    monkeypatch.setenv("CLOB_API_KEY", "test-key")
    monkeypatch.setenv("CLOB_API_SECRET", "test-secret")
    monkeypatch.setenv("CLOB_API_PASSPHRASE", "test-passphrase")

    captured: dict = {}

    class _SpyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def create_or_derive_api_key(self):
            raise AssertionError(
                "create_or_derive_api_key MUST NOT be called when cached "
                "creds are present in env — that's the whole point of caching"
            )
        def set_api_creds(self, creds):
            raise AssertionError(
                "set_api_creds MUST NOT be called when creds were passed "
                "to the constructor — they should already be set"
            )

    monkeypatch.setattr(execution, "ClobClient", _SpyClient)
    client = execution.get_clob_client()

    assert client is not None
    # Verify creds were passed to ClobClient constructor
    assert "creds" in captured, (
        f"creds kwarg should be passed to ClobClient when cached, got {list(captured)!r}"
    )
    creds = captured["creds"]
    assert creds.api_key        == "test-key"
    assert creds.api_secret     == "test-secret"
    assert creds.api_passphrase == "test-passphrase"
    # Other required kwargs still flow through
    assert captured["funder"] == "0xMyProxy"
    assert captured["signature_type"] == 2


def test_missing_creds_falls_back_to_derive(live_mode, monkeypatch):
    """If any CLOB_API_* var is missing, fall back to the derive path
    (and log a warning instructing the user to populate the cache)."""
    # Explicit empty
    monkeypatch.delenv("CLOB_API_KEY", raising=False)
    monkeypatch.delenv("CLOB_API_SECRET", raising=False)
    monkeypatch.delenv("CLOB_API_PASSPHRASE", raising=False)

    derive_called = [False]
    set_called    = [False]

    class _SpyClient:
        def __init__(self, **kwargs): pass
        def create_or_derive_api_key(self):
            derive_called[0] = True
            return MagicMock(api_key="derived", api_secret="x", api_passphrase="y")
        def set_api_creds(self, creds):
            set_called[0] = True

    monkeypatch.setattr(execution, "ClobClient", _SpyClient)
    execution.get_clob_client()

    assert derive_called[0] is True, "should call create_or_derive_api_key when no cache"
    assert set_called[0] is True, "should call set_api_creds with derived creds"


def test_partial_creds_still_derives(live_mode, monkeypatch):
    """If only 2 of 3 cached creds are present, fall back to derive
    (don't try to use partial creds — they'd auth-fail in confusing ways)."""
    monkeypatch.setenv("CLOB_API_KEY", "test-key")
    monkeypatch.setenv("CLOB_API_SECRET", "test-secret")
    monkeypatch.delenv("CLOB_API_PASSPHRASE", raising=False)

    derive_called = [False]

    class _SpyClient:
        def __init__(self, **kwargs):
            assert "creds" not in kwargs, "should NOT pass partial creds to constructor"
        def create_or_derive_api_key(self):
            derive_called[0] = True
            return MagicMock(api_key="d", api_secret="d", api_passphrase="d")
        def set_api_creds(self, creds): pass

    monkeypatch.setattr(execution, "ClobClient", _SpyClient)
    execution.get_clob_client()
    assert derive_called[0] is True


def test_whitespace_in_creds_treated_as_unset(live_mode, monkeypatch):
    """Trailing whitespace or empty strings should fall back to derive
    (catches the case where someone left the env var stub but didn't fill it)."""
    monkeypatch.setenv("CLOB_API_KEY", "   ")
    monkeypatch.setenv("CLOB_API_SECRET", "")
    monkeypatch.setenv("CLOB_API_PASSPHRASE", "value")

    class _SpyClient:
        def __init__(self, **kwargs):
            assert "creds" not in kwargs, (
                "whitespace-only creds should be treated as unset, not passed"
            )
        def create_or_derive_api_key(self):
            return MagicMock(api_key="d", api_secret="d", api_passphrase="d")
        def set_api_creds(self, creds): pass

    monkeypatch.setattr(execution, "ClobClient", _SpyClient)
    execution.get_clob_client()


def test_derive_failure_raises_with_actionable_message(live_mode, monkeypatch):
    """If the derive fallback ALSO fails (Cloudflare 403), the caller
    should get a clear error pointing at the bootstrap script — not a
    cryptic 403 stack trace."""
    monkeypatch.delenv("CLOB_API_KEY", raising=False)

    class _ErrClient:
        def __init__(self, **kwargs): pass
        def create_or_derive_api_key(self):
            raise RuntimeError("403 Forbidden from Cloudflare")
        def set_api_creds(self, creds): pass

    monkeypatch.setattr(execution, "ClobClient", _ErrClient)
    with pytest.raises(RuntimeError, match="derive_api_creds.py"):
        execution.get_clob_client()


def test_paper_mode_returns_none_regardless_of_creds(monkeypatch):
    """Paper mode should bypass everything — no client, no creds check."""
    monkeypatch.setattr(execution, "PAPER_TRADE", True)
    monkeypatch.setenv("CLOB_API_KEY", "would-be-used-but-isnt")
    assert execution.get_clob_client() is None
