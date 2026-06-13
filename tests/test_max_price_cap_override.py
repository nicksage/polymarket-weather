"""
test_max_price_cap_override.py — Pin the caller-override for the
execution price ceiling (2026-06-13).

Regression: probability-mode orders for expensive bins (best_ask >
~0.22) sat unfilled because execute_signal hard-coded its cap to
MPV_MAX_PRICE (0.32) via direct import.  Observed live: Houston
intended=0.94, limit=0.32, source=fallback_capped_at_max,
sweepable=$0.00 — no seller offered at or below 0.32, so the order
just rested at $0.32 indefinitely.

Fix: execute_signal honors `signal["max_price_cap"]` when present.
scheduled_predictor passes PREDICTOR_PROBABILITY_MAX_PRICE (default
0.85) when PREDICTOR_BUY_MODE=probability; passes None (→ MPV default)
otherwise.  This file pins the signal-dict contract.

Run:
    python -m pytest tests/test_max_price_cap_override.py -v
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


def _book(asks, bids=None):
    return {
        "asks":      [{"price": str(p), "size": str(s)} for p, s in asks],
        "bids":      [{"price": str(p), "size": str(s)} for p, s in (bids or [])],
        "tick_size": "0.01", "neg_risk": True,
    }


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


def _expensive_book_signal(max_price_cap=None):
    """Houston-like scenario: best_ask=0.95 on a near-certainty bin.
    With MPV's 0.32 cap, the limit would be capped there and sweep
    nothing.  With a 0.99 caller override, the order can actually
    cross the spread and fill."""
    sig = {
        "contract_id":      "0xhouston",
        "recommended_side": "YES",
        "kelly_size":       10.0,
        "yes_token_id":     "tok_yes_houston",
        "no_token_id":      "tok_no_houston",
        "yes_price":        0.94,
        "market_p":         0.94,
        "city":             "Houston",
        "date":             "2026-06-13",
        "scan_timestamp":   "2026-06-13T20:00:00+00:00",
    }
    if max_price_cap is not None:
        sig["max_price_cap"] = max_price_cap
    return sig


# ============================================================
# The original Houston bug — no override, hits MPV cap, no fill
# ============================================================

def test_expensive_book_caps_at_mpv_default_without_override(
    temp_db, monkeypatch
):
    """Reproduce the live failure mode: signal with NO max_price_cap
    AND expensive book (best_ask=0.95) → limit capped at 0.32 → empty
    sweep.  This is the behavior we WANT in MPV/edge strategies (they
    don't want to overpay) but UNWANTED in probability mode."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.95, 100)], bids=[(0.90, 50)],
    )

    # Probe the sweep limit directly — same code path execute_signal uses.
    # No override → falls back to MPV_MAX_PRICE (0.32 from the strategy
    # module, or 0.99 if the import fails).
    try:
        from strategies.market_price_value import MPV_MAX_PRICE
    except Exception:
        MPV_MAX_PRICE = 0.99   # match execute_signal's fallback
    limit, diag = execution.compute_sweep_limit(
        client, "tok_yes_houston", intended_price=0.94,
        max_cap=MPV_MAX_PRICE, walk_cents=10,
    )
    # When MPV cap binds (0.32 < best_ask=0.95), source must report it
    if MPV_MAX_PRICE <= 0.40:
        assert diag["source"] == "fallback_capped_at_max", (
            "On a 0.95-ask book with a 0.32 cap, the sweep must report "
            "fallback_capped_at_max — this is the exact failure mode "
            "the override fixes."
        )
        assert limit == pytest.approx(MPV_MAX_PRICE)


# ============================================================
# Caller override lets the sweep actually fill
# ============================================================

def test_caller_override_unblocks_expensive_book(temp_db, monkeypatch):
    """When the caller passes max_price_cap=0.95, the same expensive
    book sweeps successfully — the cap doesn't bind."""
    client = MagicMock()
    # Deep book at 0.94 — sweepable.
    client.get_order_book.return_value = _book(
        asks=[(0.94, 100)], bids=[(0.92, 50)],
    )

    limit, diag = execution.compute_sweep_limit(
        client, "tok_yes_houston", intended_price=0.94,
        max_cap=0.95, walk_cents=10,
    )
    # raw = 0.94 + 0.10 = 1.04, but max_cap=0.95 binds.  Still NOT
    # MPV's 0.32 — the order can fill at the ask.
    assert limit == pytest.approx(0.95)
    assert diag["source"] == "fallback_capped_at_max"
    assert diag["sweepable_usdc"] > 0, (
        "With cap raised above best_ask, the sweep must collect some "
        "ask-side depth — otherwise the override didn't help."
    )


def test_caller_override_threaded_through_execute_signal(temp_db, monkeypatch):
    """Integration check: execute_signal must READ signal['max_price_cap']
    and pass it through to compute_sweep_limit.  Without this wiring,
    the constant from scheduled_predictor is silently ignored."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    captured_cap = {}
    orig_compute = execution.compute_sweep_limit
    def _spy_compute_sweep_limit(*, client, token_id, intended_price,
                                    max_cap, walk_cents=1):
        captured_cap["value"] = max_cap
        return orig_compute(client=client, token_id=token_id,
                              intended_price=intended_price,
                              max_cap=max_cap, walk_cents=walk_cents)
    monkeypatch.setattr(execution, "compute_sweep_limit",
                          _spy_compute_sweep_limit)

    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.94, 100)], bids=[(0.92, 50)],
    )
    class _SpyOrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xok", "status": "live",
        "takingAmount": "", "makingAmount": "",
    }

    sig = _expensive_book_signal(max_price_cap=0.95)
    execution.execute_signal(sig, client=client)

    assert captured_cap.get("value") == 0.95, (
        f"execute_signal should pass max_price_cap=0.95 from the signal "
        f"to compute_sweep_limit.  Got {captured_cap.get('value')!r}."
    )


def test_omitted_cap_falls_back_to_mpv_default(temp_db, monkeypatch):
    """Edge mode and legacy callers omit max_price_cap → execute_signal
    falls back to the MPV import.  This preserves the prior behavior for
    every non-probability strategy."""
    monkeypatch.setattr(execution, "PAPER_TRADE", False)
    import config
    monkeypatch.setattr(config, "MAX_TAKE_PCT_OF_ASK_DEPTH", 0.66)
    monkeypatch.setattr(config, "MIN_FILLABLE_USDC", 2.0)
    execution._reset_geoblock_circuit()

    captured_cap = {}
    orig_compute = execution.compute_sweep_limit
    def _spy_compute_sweep_limit(*, client, token_id, intended_price,
                                    max_cap, walk_cents=1):
        captured_cap["value"] = max_cap
        return orig_compute(client=client, token_id=token_id,
                              intended_price=intended_price,
                              max_cap=max_cap, walk_cents=walk_cents)
    monkeypatch.setattr(execution, "compute_sweep_limit",
                          _spy_compute_sweep_limit)

    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.28, 100)], bids=[(0.27, 50)],
    )
    class _SpyOrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    monkeypatch.setattr(execution, "OrderArgs", _SpyOrderArgs)
    client.create_and_post_order.return_value = {
        "success": True, "orderID": "0xok", "status": "live",
        "takingAmount": "", "makingAmount": "",
    }

    sig = _expensive_book_signal(max_price_cap=None)
    # Override the price too — this is a cheap-book scenario now
    sig["yes_price"] = 0.28
    sig["market_p"] = 0.28
    execution.execute_signal(sig, client=client)

    try:
        from strategies.market_price_value import MPV_MAX_PRICE
        expected = MPV_MAX_PRICE
    except Exception:
        expected = 0.99
    assert captured_cap.get("value") == pytest.approx(expected), (
        f"Without override, execute_signal must fall back to MPV_MAX_PRICE "
        f"({expected}).  Got {captured_cap.get('value')!r}."
    )