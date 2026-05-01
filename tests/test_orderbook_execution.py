"""
test_orderbook_execution.py — Tests for the orderbook-aware execution
and ranking work (added 2026-04-30).

Two phases pinned by these tests:
  Phase 1 — execute_signal computes its limit price from the live
            orderbook (best_ask + walk_cents) capped at MPV_MAX_PRICE.
            Polymarket's matcher then sweeps the book cheapest-first.
  Phase 2 — MarketPriceValueStrategy.rank_signals fetches orderbooks
            for the top N candidates, scores by spread + sweepable
            depth, drops anything with spread > MAX_SPREAD_CENTS_FOR_ENTRY.

Together they shift the bot from "always pay touch + 50bps" to
"capture cheap liquidity within a 1¢ window of touch, then rest".
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

import execution


def _book(asks: list[tuple[float, float]],
          bids: list[tuple[float, float]] | None = None) -> dict:
    """Build a v2-shaped orderbook dict for mocking client.get_order_book."""
    return {
        "asks":           [{"price": str(p), "size": str(s)} for p, s in asks],
        "bids":           [{"price": str(p), "size": str(s)} for p, s in (bids or [])],
        "tick_size":      "0.01",
        "neg_risk":       True,
    }


# ===========================================================================
# Phase 1: get_orderbook_snapshot normalization
# ===========================================================================

def test_snapshot_sorts_asks_ascending_bids_descending():
    """v2 returns bids ASCENDING and asks ASCENDING in raw form.  The
    snapshot helper normalizes both: asks asc (cheapest first), bids desc
    (best first), so callers don't have to think about ordering."""
    client = MagicMock()
    client.get_order_book.return_value = _book(
        # Mixed-order asks; helper must sort
        asks=[(0.32, 5), (0.28, 10), (0.30, 7)],
        bids=[(0.27, 50), (0.26, 100)],
    )
    snap = execution.get_orderbook_snapshot(client, "tok")
    assert snap["best_ask"] == pytest.approx(0.28)
    assert snap["best_bid"] == pytest.approx(0.27)
    assert snap["spread_cents"] == pytest.approx(1.0)
    assert [p for p, _ in snap["asks_sorted_asc"]] == [0.28, 0.30, 0.32]
    assert [p for p, _ in snap["bids_sorted_desc"]] == [0.27, 0.26]


def test_snapshot_drops_garbage_levels():
    """Levels with non-numeric or zero values are dropped, not crashed on."""
    client = MagicMock()
    client.get_order_book.return_value = {
        "asks": [
            {"price": "not-a-number", "size": "10"},
            {"price": "0.30", "size": "0"},          # zero size dropped
            {"price": "0.31", "size": "5"},
            {"price": "0.32", "size": None},         # None dropped
        ],
        "bids": [],
    }
    snap = execution.get_orderbook_snapshot(client, "tok")
    assert snap["best_ask"] == pytest.approx(0.31)
    assert len(snap["asks_sorted_asc"]) == 1


def test_snapshot_returns_none_on_api_error():
    client = MagicMock()
    client.get_order_book.side_effect = RuntimeError("network down")
    assert execution.get_orderbook_snapshot(client, "tok") is None


def test_snapshot_handles_missing_side():
    """Empty asks list → best_ask=None and spread=None.  Don't crash."""
    client = MagicMock()
    client.get_order_book.return_value = _book(asks=[], bids=[(0.20, 1)])
    snap = execution.get_orderbook_snapshot(client, "tok")
    assert snap["best_ask"] is None
    assert snap["best_bid"] == pytest.approx(0.20)
    assert snap["spread_cents"] is None


# ===========================================================================
# Phase 1: compute_sweep_limit core formula
# ===========================================================================

def test_sweep_limit_returns_best_ask_plus_walk():
    """Normal book: limit = best_ask + walk_cents/100."""
    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.28, 10), (0.29, 5), (0.30, 3)],
        bids=[(0.27, 50)],
    )
    limit, diag = execution.compute_sweep_limit(
        client, "tok", intended_price=0.30, max_cap=0.99, walk_cents=1,
    )
    assert limit == pytest.approx(0.29)   # 0.28 + 0.01
    assert diag["source"] == "sweep"
    assert diag["best_ask"] == pytest.approx(0.28)


def test_sweep_limit_capped_at_max():
    """Aggressive walk hitting MPV_MAX_PRICE should be capped there."""
    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.295, 5), (0.31, 100)],
        bids=[(0.28, 1)],
    )
    limit, diag = execution.compute_sweep_limit(
        client, "tok", intended_price=0.30, max_cap=0.30, walk_cents=1,
    )
    # raw = 0.305, capped at 0.30
    assert limit == pytest.approx(0.30)
    assert diag["source"] == "fallback_capped_at_max"


def test_sweep_limit_falls_back_when_book_unavailable():
    """API failure → use intended_price * 1.005, never refuse to trade."""
    client = MagicMock()
    client.get_order_book.side_effect = RuntimeError("503")
    limit, diag = execution.compute_sweep_limit(
        client, "tok", intended_price=0.30, max_cap=0.99, walk_cents=1,
    )
    assert limit == pytest.approx(round(0.30 * 1.005, 4))
    assert diag["source"] == "fallback_no_book"


def test_sweep_limit_falls_back_when_no_asks():
    """Empty ask side → fall back (rest at safe price, hope for taker)."""
    client = MagicMock()
    client.get_order_book.return_value = _book(asks=[], bids=[(0.20, 5)])
    limit, diag = execution.compute_sweep_limit(
        client, "tok", intended_price=0.30, max_cap=0.99, walk_cents=1,
    )
    assert limit == pytest.approx(round(0.30 * 1.005, 4))
    assert diag["source"] == "fallback_no_asks"


def test_sweep_limit_diagnostics_include_sweepable_usdc():
    """Diagnostics surface what we'd actually sweep — useful for the
    ranking score in Phase 2."""
    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.28, 10), (0.29, 5), (0.30, 100)],   # 0.28×10=$2.80, 0.29×5=$1.45
        bids=[(0.27, 1)],
    )
    _limit, diag = execution.compute_sweep_limit(
        client, "tok", intended_price=0.30, max_cap=0.99, walk_cents=1,
    )
    # walk window = 0.28 + 0.01 = 0.29; sweepable = 0.28×10 + 0.29×5 = 2.80 + 1.45 = 4.25
    assert diag["sweepable_usdc"] == pytest.approx(4.25)
    assert diag["spread_cents"] == pytest.approx(1.0)


def test_sweep_limit_walk_cents_zero_targets_only_touch():
    """walk_cents=0 means 'only buy at best_ask' (no walk).  Limit equals
    best_ask exactly."""
    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.28, 10), (0.29, 5)],
        bids=[(0.27, 1)],
    )
    limit, _ = execution.compute_sweep_limit(
        client, "tok", intended_price=0.30, max_cap=0.99, walk_cents=0,
    )
    assert limit == pytest.approx(0.28)


# ===========================================================================
# Phase 2: orderbook-aware rank_signals
# ===========================================================================

def test_ranking_falls_back_to_static_liquidity_without_client():
    """When client is None (paper mode, tests), ranking uses static
    liquidity_usd only — no orderbook fetches."""
    from strategies.market_price_value import MarketPriceValueStrategy
    s = MarketPriceValueStrategy()
    signals = [
        {"yes_token_id": "a", "liquidity_usd": 1000, "city": "A"},
        {"yes_token_id": "b", "liquidity_usd": 5000, "city": "B"},
        {"yes_token_id": "c", "liquidity_usd": 200,  "city": "C"},
    ]
    ranked = s.rank_signals(signals, bankroll=200.0, client=None)
    # Highest liquidity first
    assert [r["city"] for r in ranked] == ["B", "A", "C"]


def test_ranking_orderbook_score_prefers_tight_spread(monkeypatch):
    """With equal liquidity, the bin with tighter spread should rank higher."""
    from strategies.market_price_value import MarketPriceValueStrategy
    import config
    monkeypatch.setattr(config, "RANK_TOP_N_FOR_ORDERBOOK", 5)
    monkeypatch.setattr(config, "MAX_SPREAD_CENTS_FOR_ENTRY", 4)
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)

    client = MagicMock()
    def fake_book(token_id):
        if token_id == "tight":
            # 1¢ spread, deep
            return _book(asks=[(0.30, 100)], bids=[(0.29, 100)])
        if token_id == "wide":
            # 3¢ spread, deep
            return _book(asks=[(0.30, 100)], bids=[(0.27, 100)])
        return _book(asks=[], bids=[])
    client.get_order_book.side_effect = fake_book

    s = MarketPriceValueStrategy()
    signals = [
        {"yes_token_id": "tight", "liquidity_usd": 1000, "city": "Tight"},
        {"yes_token_id": "wide",  "liquidity_usd": 1000, "city": "Wide"},
    ]
    ranked = s.rank_signals(signals, bankroll=200.0, client=client)
    assert ranked[0]["city"] == "Tight"
    # Tight (spread=1, score=9) > Wide (spread=3, score=7)
    assert ranked[0]["priority_components"]["spread_score"] == pytest.approx(9.0)
    assert ranked[1]["priority_components"]["spread_score"] == pytest.approx(7.0)


def test_ranking_orderbook_score_prefers_deeper_book(monkeypatch):
    """With equal spread, the bin with more sweepable depth should rank higher."""
    from strategies.market_price_value import MarketPriceValueStrategy
    import config
    monkeypatch.setattr(config, "RANK_TOP_N_FOR_ORDERBOOK", 5)
    monkeypatch.setattr(config, "MAX_SPREAD_CENTS_FOR_ENTRY", 4)
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)

    client = MagicMock()
    def fake_book(token_id):
        if token_id == "deep":
            # Lots of size at touch + 1¢ window
            return _book(
                asks=[(0.30, 100), (0.31, 200)],
                bids=[(0.29, 1)],
            )
        if token_id == "thin":
            # Tiny size in the window
            return _book(
                asks=[(0.30, 1), (0.31, 1)],
                bids=[(0.29, 1)],
            )
        return _book(asks=[], bids=[])
    client.get_order_book.side_effect = fake_book

    s = MarketPriceValueStrategy()
    signals = [
        {"yes_token_id": "thin", "liquidity_usd": 1000, "city": "Thin"},
        {"yes_token_id": "deep", "liquidity_usd": 1000, "city": "Deep"},
    ]
    ranked = s.rank_signals(signals, bankroll=200.0, client=client)
    assert ranked[0]["city"] == "Deep"
    assert ranked[0]["priority_components"]["depth_score"] > ranked[1]["priority_components"]["depth_score"]


def test_ranking_drops_wide_spread_signals(monkeypatch):
    """Spread > MAX_SPREAD_CENTS_FOR_ENTRY → score=-1000 (filterable)."""
    from strategies.market_price_value import MarketPriceValueStrategy
    import config
    monkeypatch.setattr(config, "RANK_TOP_N_FOR_ORDERBOOK", 5)
    monkeypatch.setattr(config, "MAX_SPREAD_CENTS_FOR_ENTRY", 4)
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)

    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.30, 10)],
        bids=[(0.20, 10)],   # 10¢ spread, way over the 4¢ cap
    )

    s = MarketPriceValueStrategy()
    signals = [{"yes_token_id": "wide", "liquidity_usd": 1000, "city": "X"}]
    ranked = s.rank_signals(signals, bankroll=200.0, client=client)
    assert ranked[0]["priority_score"] == -1000
    assert "spread" in ranked[0]["priority_components"]["skip_reason"]


def test_ranking_only_fetches_top_n(monkeypatch):
    """Out of 20 signals, only the top RANK_TOP_N_FOR_ORDERBOOK=3 should
    get an orderbook fetch.  Pins the API-call budget."""
    from strategies.market_price_value import MarketPriceValueStrategy
    import config
    monkeypatch.setattr(config, "RANK_TOP_N_FOR_ORDERBOOK", 3)
    monkeypatch.setattr(config, "MAX_SPREAD_CENTS_FOR_ENTRY", 0)  # disable filter
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)

    client = MagicMock()
    client.get_order_book.return_value = _book(
        asks=[(0.30, 10)], bids=[(0.29, 10)]
    )

    s = MarketPriceValueStrategy()
    signals = [
        {"yes_token_id": f"tok{i}", "liquidity_usd": (20 - i) * 1000, "city": f"C{i}"}
        for i in range(20)
    ]
    s.rank_signals(signals, bankroll=200.0, client=client)
    # Only 3 fetches happened — the highest-liquidity 3
    assert client.get_order_book.call_count == 3


def test_ranking_handles_orderbook_fetch_failure_gracefully(monkeypatch):
    """If the orderbook fetch fails for one signal, that signal gets a
    deprioritized score but doesn't take down the ranking pass."""
    from strategies.market_price_value import MarketPriceValueStrategy
    import config
    monkeypatch.setattr(config, "RANK_TOP_N_FOR_ORDERBOOK", 5)
    monkeypatch.setattr(config, "MAX_SPREAD_CENTS_FOR_ENTRY", 4)
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)

    client = MagicMock()
    def flaky(token_id):
        if token_id == "broken":
            raise RuntimeError("503")
        return _book(asks=[(0.30, 10)], bids=[(0.29, 10)])
    client.get_order_book.side_effect = flaky

    s = MarketPriceValueStrategy()
    signals = [
        {"yes_token_id": "broken", "liquidity_usd": 5000, "city": "Broken"},
        {"yes_token_id": "ok",     "liquidity_usd": 4000, "city": "OK"},
    ]
    ranked = s.rank_signals(signals, bankroll=200.0, client=client)
    # OK ranked first, broken deprioritized
    assert ranked[0]["city"] == "OK"
    assert ranked[1]["priority_score"] == -100


def test_ranking_no_asks_deprioritizes(monkeypatch):
    """A bin with no asks at all (no sellers) gets a low score — can't
    enter even if liquidity_usd looked promising."""
    from strategies.market_price_value import MarketPriceValueStrategy
    import config
    monkeypatch.setattr(config, "RANK_TOP_N_FOR_ORDERBOOK", 5)
    monkeypatch.setattr(config, "MAX_SPREAD_CENTS_FOR_ENTRY", 4)
    monkeypatch.setattr(config, "ORDERBOOK_WALK_CENTS", 1)

    client = MagicMock()
    client.get_order_book.return_value = _book(asks=[], bids=[(0.20, 50)])
    s = MarketPriceValueStrategy()
    signals = [{"yes_token_id": "noasks", "liquidity_usd": 5000, "city": "X"}]
    ranked = s.rank_signals(signals, bankroll=200.0, client=client)
    assert ranked[0]["priority_score"] == -50
