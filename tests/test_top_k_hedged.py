"""
test_top_k_hedged.py — Tests for the top-K hedged strategy.

Covers:
  * _parse_split validation (sum == 100, derives K)
  * _hours_to_close window filter
  * generate_signals: time-window, price-gate, top-K selection, % split
  * Per-event dedup (TKH_MAX_TRADES_PER_EVENT)
  * evaluate_positions: TAKE_PROFIT, HEDGE_RESOLVED, HARD_STOP, HEALTHY
  * Hedge convergence: 1 sibling at TKH_CONFIRM_PRICE → others get HEDGE_RESOLVED
  * Registry: strategy is registered under "top_k_hedged"
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

import db
import config


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


# ===========================================================================
# _parse_split
# ===========================================================================

def test_parse_split_basic():
    from strategies.top_k_hedged import _parse_split
    assert _parse_split("70:20:10") == [0.7, 0.2, 0.1]
    assert _parse_split("60:40")    == [0.6, 0.4]
    assert _parse_split("100")      == [1.0]


def test_parse_split_must_sum_to_100():
    from strategies.top_k_hedged import _parse_split
    with pytest.raises(ValueError, match="must sum to 100"):
        _parse_split("60:30:5")          # 95
    with pytest.raises(ValueError, match="must sum to 100"):
        _parse_split("50:50:50")         # 150


def test_parse_split_rejects_garbage():
    from strategies.top_k_hedged import _parse_split
    with pytest.raises(ValueError):
        _parse_split("abc:def")
    with pytest.raises(ValueError, match="cannot be empty"):
        _parse_split("")


def test_parse_split_tolerates_whitespace():
    from strategies.top_k_hedged import _parse_split
    assert _parse_split("70 : 20 : 10") == [0.7, 0.2, 0.1]


# ===========================================================================
# Strategy registration
# ===========================================================================

def test_strategy_is_registered():
    from strategies import _REGISTRY
    assert "top_k_hedged" in _REGISTRY


def test_strategy_top_k_derived_from_split():
    """K is derived from len(TKH_SPLIT) — no separate K config."""
    from strategies.top_k_hedged import TKH_TOP_K, TKH_SPLIT
    assert TKH_TOP_K == len(TKH_SPLIT)


# ===========================================================================
# generate_signals
# ===========================================================================

def _fake_event(*, event_id="evt1", city="Lagos", date="2099-01-01",
                lat=6.5, lon=3.4, outcomes=None, **extra) -> dict:
    """Minimal event dict shape that matches what discovery returns."""
    return {
        "event_id":     event_id,
        "event_title":  f"Will the highest temperature in {city} be ...",
        "city":         city,
        "date":         date,
        "lat":          lat,
        "lon":          lon,
        "outcomes":     outcomes or [],
        **extra,
    }


def _bin(*, contract_id, model_prob, range_low=20, range_high=20,
         yes_token_id="tok") -> dict:
    return {
        "contract_id":  contract_id,
        "yes_token_id": yes_token_id,
        "no_token_id":  "tok_no",
        "model_prob":   model_prob,
        "yes_price":    model_prob,
        "market_price": model_prob,
        "range_low":    range_low,
        "range_high":   range_high,
        "unit":         "celsius",
        "liquidity_usd": 500.0,
    }


def _patch_window(monkeypatch, hours: float):
    """Force _hours_to_close to return a specific value."""
    import strategies.top_k_hedged as tkh
    monkeypatch.setattr(tkh, "_hours_to_close", lambda *_a, **_kw: hours)


def _patch_analyze_passthrough(monkeypatch):
    """analyze_event_base normally runs the full weather pipeline.  For
    these tests we just pass the event through (its outcomes already have
    model_prob set)."""
    from strategies.base import Strategy
    monkeypatch.setattr(Strategy, "analyze_event_base",
                        staticmethod(lambda event, _bankroll: event))


def test_generate_signals_outside_window_skipped(temp_db, monkeypatch):
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 100)   # well outside 35-38h
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.40),
        _bin(contract_id="c2", model_prob=0.30),
        _bin(contract_id="c3", model_prob=0.20),
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert signals == []


def test_generate_signals_inside_window_emits_top_k(temp_db, monkeypatch):
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)   # inside 35-38
    from strategies.top_k_hedged import (
        TopKHedgedStrategy, TKH_TOP_K, TKH_SPLIT, TKH_TOTAL_BET_SIZE,
    )
    s = TopKHedgedStrategy()
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.40),
        _bin(contract_id="c2", model_prob=0.30),
        _bin(contract_id="c3", model_prob=0.20),
        _bin(contract_id="c4", model_prob=0.10),   # outside top K, should NOT be emitted
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert len(signals) == TKH_TOP_K
    cids = [s["contract_id"] for s in signals]
    assert cids == ["c1", "c2", "c3"]   # top 3 in prob order
    # Per-bin amounts match the split
    for i, s_dict in enumerate(signals):
        expected = round(TKH_TOTAL_BET_SIZE * TKH_SPLIT[i], 2)
        assert s_dict["kelly_size"] == pytest.approx(expected)
        assert s_dict["target_size_usdc"] == pytest.approx(expected)
    # All signals tagged with strategy + rank
    assert all(s["strategy"] == "top_k_hedged" for s in signals)
    assert [s["tkh_bin_rank"] for s in signals] == [0, 1, 2]


def test_generate_signals_basket_edge_passes_when_above_threshold(
    temp_db, monkeypatch
):
    """Top-3 sum 0.65 → basket_edge 0.35 → > 0.05 default → enter all 3."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy, TKH_TOP_K
    s = TopKHedgedStrategy()
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.25),
        _bin(contract_id="c2", model_prob=0.20),
        _bin(contract_id="c3", model_prob=0.20),
        _bin(contract_id="c4", model_prob=0.15),  # not in top K
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert len(signals) == TKH_TOP_K
    assert [s["contract_id"] for s in signals] == ["c1", "c2", "c3"]


def test_generate_signals_basket_edge_skips_when_below_threshold(
    temp_db, monkeypatch
):
    """Top 3 sum 0.96 → basket_edge 0.04 → ≤ 0.05 → skip event entirely.
    Heavily-priced market with no room for the hedged basket to clear cost."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.50),
        _bin(contract_id="c2", model_prob=0.30),
        _bin(contract_id="c3", model_prob=0.16),
        _bin(contract_id="c4", model_prob=0.04),
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert signals == []


def test_generate_signals_basket_edge_just_below_threshold_skips(
    temp_db, monkeypatch
):
    """basket_edge slightly below TKH_BASKET_EDGE_MIN (uses ≤, strict)."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    # Top-3 sum 0.96 → edge 0.04 < 0.05 default → skip
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.32),
        _bin(contract_id="c2", model_prob=0.32),
        _bin(contract_id="c3", model_prob=0.32),
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert signals == []


def test_generate_signals_basket_edge_disabled_at_zero(temp_db, monkeypatch):
    """TKH_BASKET_EDGE_MIN = 0 means: any positive edge is fine.  An event
    with sum 0.99 (edge 0.01) still trades."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    monkeypatch.setattr(
        "strategies.top_k_hedged.TKH_BASKET_EDGE_MIN", 0.0,
    )
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.50),
        _bin(contract_id="c2", model_prob=0.30),
        _bin(contract_id="c3", model_prob=0.19),
        _bin(contract_id="c4", model_prob=0.01),
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert len(signals) == 3


def test_generate_signals_skips_event_with_too_few_bins(temp_db, monkeypatch):
    """Event with fewer outcomes than K → can't form a hedged set → skip."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.30),
        _bin(contract_id="c2", model_prob=0.20),
        # only 2 bins, K=3
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert signals == []


def test_generate_signals_basket_edge_handles_overpriced_book(
    temp_db, monkeypatch
):
    """When sum of top K > 1.0 (overround / overpriced market), basket_edge
    is negative → trivially below threshold → skip."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    # Top 3 sum = 1.20 → basket_edge = -0.20 → skip
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.50),
        _bin(contract_id="c2", model_prob=0.40),
        _bin(contract_id="c3", model_prob=0.30),
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert signals == []


def test_generate_signals_per_event_dedup(temp_db, monkeypatch):
    """Once any position exists for an event under this strategy, the
    next scan for that event emits no signals."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    # Insert a prior position for event_id="evt1" under top_k_hedged
    db.insert_position(
        contract_id="c1", side="YES", size_usdc=7.0, entry_price=0.40,
        entry_time="2099-01-01T00:00:00", target_size_usdc=7.0, shares=17.5,
        is_paper=0, fill_status="filled", strategy="top_k_hedged",
        event_id="evt1", city="Lagos", date="2099-01-01",
    )
    ev = _fake_event(outcomes=[
        _bin(contract_id="c1", model_prob=0.40),
        _bin(contract_id="c2", model_prob=0.30),
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert signals == []


def test_generate_signals_dedup_ignores_other_strategies(temp_db, monkeypatch):
    """If a position exists but under a DIFFERENT strategy, the TKH dedup
    doesn't trigger.  The event is still eligible for TKH."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    db.insert_position(
        contract_id="c1", side="YES", size_usdc=10.0, entry_price=0.40,
        entry_time="2099-01-01T00:00:00", target_size_usdc=10.0, shares=25.0,
        is_paper=0, fill_status="filled", strategy="market_price_value",
        event_id="evt1", city="Lagos", date="2099-01-01",
    )
    ev = _fake_event(outcomes=[
        _bin(contract_id="c2", model_prob=0.40),
        _bin(contract_id="c3", model_prob=0.30),
        _bin(contract_id="c4", model_prob=0.20),   # K=3 + edge 0.10 > 0.05
    ])
    _, signals = s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    assert len(signals) > 0


# ===========================================================================
# evaluate_positions
# ===========================================================================

def _seed_tkh_position(*, pid_marker=None, event_id="evt1", contract_id="c1",
                      entry_price=0.40, peak_price=None, current_price=None,
                      shares=10.0) -> int:
    pid = db.insert_position(
        contract_id=contract_id, side="YES", size_usdc=shares*entry_price,
        entry_price=entry_price, entry_time="2099-01-01T00:00:00",
        target_size_usdc=shares*entry_price, shares=shares,
        is_paper=0, fill_status="filled", strategy="top_k_hedged",
        event_id=event_id, city="Lagos", date="2099-01-01",
    )
    if peak_price is not None or current_price is not None:
        import sqlite3
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute(
                "UPDATE positions SET peak_price=?, current_price=? WHERE id=?",
                (peak_price or current_price, current_price, pid),
            )
    return pid


def test_evaluate_take_profit(temp_db, monkeypatch):
    # Pin TP to a known threshold so this is deterministic regardless of
    # operator .env overrides.
    import strategies.top_k_hedged as tkh
    monkeypatch.setattr(tkh, "TKH_TAKE_PROFIT", 0.85)
    pid = _seed_tkh_position(entry_price=0.40, current_price=0.90)
    s = tkh.TopKHedgedStrategy()
    actions = s.evaluate_positions()
    assert len(actions) == 1
    a = actions[0]
    assert a.position_id == pid
    assert a.classification == "TAKE_PROFIT"
    assert a.action == "SELL"


def test_evaluate_hard_stop(temp_db, monkeypatch):
    # entry $0.40, hard_stop_pct 0.50 → fires at $0.20.  Current $0.18 → fires.
    # Monkeypatch the threshold so the test is deterministic regardless of
    # the operator's .env override (e.g. TKH_HARD_STOP_PCT=0.90 would
    # otherwise put the trigger at $0.04 and skip).
    import strategies.top_k_hedged as tkh
    monkeypatch.setattr(tkh, "TKH_HARD_STOP_PCT", 0.50)
    pid = _seed_tkh_position(entry_price=0.40, current_price=0.18)
    s = tkh.TopKHedgedStrategy()
    actions = s.evaluate_positions()
    assert len(actions) == 1
    assert actions[0].classification == "HARD_STOP"


def test_evaluate_healthy_holds(temp_db, monkeypatch):
    # Pin thresholds so this test is invariant to .env overrides
    import strategies.top_k_hedged as tkh
    monkeypatch.setattr(tkh, "TKH_HARD_STOP_PCT", 0.50)
    monkeypatch.setattr(tkh, "TKH_TAKE_PROFIT", 0.85)
    # Current is between hard stop and take-profit → HEALTHY → no action
    pid = _seed_tkh_position(entry_price=0.40, current_price=0.42)
    s = tkh.TopKHedgedStrategy()
    actions = s.evaluate_positions()
    assert actions == []


def test_evaluate_hedge_resolved_exits_siblings(temp_db):
    """Three bins of one event.  One hits TKH_CONFIRM_PRICE.  The OTHER
    two should get HEDGE_RESOLVED exits.  The winner itself should hit
    TAKE_PROFIT (since 0.85 ≥ 0.85)."""
    p1 = _seed_tkh_position(event_id="hedge_evt", contract_id="c_winner",
                            entry_price=0.50, current_price=0.85)
    p2 = _seed_tkh_position(event_id="hedge_evt", contract_id="c_loser1",
                            entry_price=0.30, current_price=0.20)
    p3 = _seed_tkh_position(event_id="hedge_evt", contract_id="c_loser2",
                            entry_price=0.20, current_price=0.10)

    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    actions = s.evaluate_positions()

    # 3 actions expected: winner takes profit, two losers hedge-resolved
    by_pid = {a.position_id: a for a in actions}
    assert by_pid[p1].classification == "TAKE_PROFIT"
    assert by_pid[p2].classification == "HEDGE_RESOLVED"
    assert by_pid[p3].classification == "HEDGE_RESOLVED"
    # All three are SELL actions
    assert all(a.action == "SELL" for a in actions)


def test_evaluate_only_processes_top_k_hedged_positions(temp_db):
    """A position from a different strategy is NOT classified by TKH's
    evaluator."""
    db.insert_position(
        contract_id="c_other", side="YES", size_usdc=10.0, entry_price=0.40,
        entry_time="2099-01-01T00:00:00", target_size_usdc=10.0, shares=25.0,
        is_paper=0, fill_status="filled", strategy="market_price_value",
        event_id="evt_other", city="Lagos", date="2099-01-01",
    )
    # And a TKH position that should be processed
    pid = _seed_tkh_position(entry_price=0.40, current_price=0.42)

    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    actions = s.evaluate_positions()
    # MPV position skipped; TKH position is HEALTHY → no action
    assert actions == []


def test_evaluate_no_open_positions(temp_db):
    """Empty positions table → no actions, no exception."""
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    assert s.evaluate_positions() == []


# ===========================================================================
# Window edge cases
# ===========================================================================

def _three_bin_event(*, event_id="evt_w") -> dict:
    """Convenience: 3 bins summing to 0.90 (basket_edge = 0.10 > default 0.05)."""
    return _fake_event(event_id=event_id, outcomes=[
        _bin(contract_id="c1", model_prob=0.40),
        _bin(contract_id="c2", model_prob=0.30),
        _bin(contract_id="c3", model_prob=0.20),
    ])


def test_window_lower_boundary_inclusive(temp_db, monkeypatch):
    """hours == TKH_HOURS_TO_CLOSE_MIN → should trade (inclusive)."""
    _patch_analyze_passthrough(monkeypatch)
    from strategies.top_k_hedged import (
        TopKHedgedStrategy, TKH_HOURS_TO_CLOSE_MIN,
    )
    _patch_window(monkeypatch, TKH_HOURS_TO_CLOSE_MIN)
    s = TopKHedgedStrategy()
    _, signals = s.generate_signals(
        [_three_bin_event(event_id="lower")], 200.0, "2099-01-01T00:00:00"
    )
    assert len(signals) >= 1


def test_window_upper_boundary_inclusive(temp_db, monkeypatch):
    _patch_analyze_passthrough(monkeypatch)
    from strategies.top_k_hedged import (
        TopKHedgedStrategy, TKH_HOURS_TO_CLOSE_MAX,
    )
    _patch_window(monkeypatch, TKH_HOURS_TO_CLOSE_MAX)
    s = TopKHedgedStrategy()
    _, signals = s.generate_signals(
        [_three_bin_event(event_id="upper")], 200.0, "2099-01-01T00:00:00"
    )
    assert len(signals) >= 1


def test_window_just_below_lower_skipped(temp_db, monkeypatch):
    _patch_analyze_passthrough(monkeypatch)
    from strategies.top_k_hedged import (
        TopKHedgedStrategy, TKH_HOURS_TO_CLOSE_MIN,
    )
    _patch_window(monkeypatch, TKH_HOURS_TO_CLOSE_MIN - 0.5)
    s = TopKHedgedStrategy()
    _, signals = s.generate_signals(
        [_three_bin_event(event_id="below")], 200.0, "2099-01-01T00:00:00"
    )
    assert signals == []


# ===========================================================================
# Funnel telemetry
# ===========================================================================

def test_funnel_logs_summary_at_summary_level(temp_db, monkeypatch, caplog):
    """Every generate_signals call emits a [TKH FUNNEL] line at SUMMARY (25)
    level so the operator sees the gate breakdown in the terminal."""
    import logging
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    ev = _three_bin_event(event_id="funnel_test")
    with caplog.at_level(logging.DEBUG, logger="strategies.top_k_hedged"):
        s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    funnel_lines = [r for r in caplog.records if "[TKH FUNNEL]" in r.message]
    assert len(funnel_lines) == 1
    assert funnel_lines[0].levelno == 25   # SUMMARY level
    msg = funnel_lines[0].message
    # Verify the breakdown shape
    assert "1 events scanned" in msg
    assert "1 in [35-38h] window" in msg
    assert "1 passed all gates" in msg
    assert "3 signals" in msg


def test_funnel_logs_skip_breakdown_when_events_filtered(
    temp_db, monkeypatch, caplog
):
    """When in-window events are filtered, the funnel line includes a
    breakdown of WHY (already-traded, too-few-bins, below-basket-edge)."""
    import logging
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    # Event with top-3 sum = 0.99 → basket_edge = 0.01 → below 0.05 → skipped
    ev = _fake_event(event_id="below", outcomes=[
        _bin(contract_id="c1", model_prob=0.50),
        _bin(contract_id="c2", model_prob=0.30),
        _bin(contract_id="c3", model_prob=0.19),
    ])
    with caplog.at_level(logging.DEBUG, logger="strategies.top_k_hedged"):
        s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
    funnel_lines = [r for r in caplog.records if "[TKH FUNNEL]" in r.message]
    assert len(funnel_lines) == 1
    msg = funnel_lines[0].message
    assert "below-basket-edge=1" in msg
    assert "0 signals" in msg


def test_funnel_persists_to_activity_log(temp_db, monkeypatch):
    """Verify the funnel data also lands in activity_log (so the
    dashboard can show the funnel history)."""
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    from strategies.top_k_hedged import TopKHedgedStrategy
    s = TopKHedgedStrategy()
    ev = _three_bin_event(event_id="activity_log_test")
    s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")

    rows = db.get_recent_activity(limit=10, categories=["TKH"])
    assert len(rows) >= 1
    row = rows[0]
    assert "funnel" in row["message"]
    import json
    meta = json.loads(row["metadata"])
    assert meta["n_total"] == 1
    assert meta["n_in_window"] == 1
    assert meta["n_qualified"] == 1
    assert meta["n_signals_emitted"] == 3


# ===========================================================================
# Topup behavior — NOT window-gated (integration check)
# ===========================================================================

def test_max_yes_bins_below_tkh_top_k_warns_at_module_load(caplog):
    """If MAX_YES_BINS is configured below TKH_TOP_K, the strategy module
    emits a loud WARN at module-load.  Re-invoke the helper directly and
    assert the warning is captured."""
    import logging
    import strategies.top_k_hedged as tkh
    # Force the misconfig: simulate MAX_YES_BINS=2 with TKH_TOP_K=3
    with caplog.at_level(logging.WARNING, logger="strategies.top_k_hedged"):
        original_max = tkh.MAX_YES_BINS
        tkh.MAX_YES_BINS = 2
        try:
            tkh._check_max_yes_bins_alignment()
        finally:
            tkh.MAX_YES_BINS = original_max
    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "[TKH STARTUP]" in r.message]
    assert len(warns) >= 1
    assert "MAX_YES_BINS=2" in warns[-1].message
    assert "K=3" in warns[-1].message
    assert "Set MAX_YES_BINS=3" in warns[-1].message


def test_max_yes_bins_aligned_no_warning(caplog):
    """When MAX_YES_BINS >= TKH_TOP_K, no startup warning is emitted."""
    import logging
    import strategies.top_k_hedged as tkh
    with caplog.at_level(logging.WARNING, logger="strategies.top_k_hedged"):
        original_max = tkh.MAX_YES_BINS
        tkh.MAX_YES_BINS = 3   # equal to default TKH_TOP_K
        try:
            tkh._check_max_yes_bins_alignment()
        finally:
            tkh.MAX_YES_BINS = original_max
    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "[TKH STARTUP]" in r.message]
    assert warns == []


def test_max_yes_bins_runtime_warning_fires_once_per_session(
    temp_db, monkeypatch, caplog
):
    """During active use, the misalignment warning surfaces at SUMMARY
    level on the FIRST generate_signals call.  Subsequent calls in the
    same session don't re-emit (avoid spam)."""
    import logging
    _patch_analyze_passthrough(monkeypatch)
    _patch_window(monkeypatch, 36)
    import strategies.top_k_hedged as tkh
    monkeypatch.setattr(tkh, "MAX_YES_BINS", 2)
    monkeypatch.setattr(tkh, "_max_yes_bins_warning_emitted", False)

    s = tkh.TopKHedgedStrategy()
    ev = _three_bin_event(event_id="warn_test")
    with caplog.at_level(logging.DEBUG, logger="strategies.top_k_hedged"):
        s.generate_signals([ev], 200.0, "2099-01-01T00:00:00")
        # Second call should NOT re-emit
        ev2 = _three_bin_event(event_id="warn_test_2")
        s.generate_signals([ev2], 200.0, "2099-01-01T00:00:00")

    warns = [r for r in caplog.records if "[TKH STARTUP-CHECK]" in r.message]
    assert len(warns) == 1   # exactly once across both calls
    assert warns[0].levelno == 25   # SUMMARY level


def test_topup_helper_does_not_check_window_or_strategy(temp_db, monkeypatch):
    """`_run_topups` is strategy-agnostic and doesn't consult the
    hours-to-close window.  An underfilled TKH position should be
    eligible for topup REGARDLESS of how far the event is from closing."""
    # Seed an underfilled TKH position whose event is well outside the
    # TKH window (e.g. only 5h to close).  The topup logic doesn't gate
    # on this — it only checks size_usdc < target_size_usdc.
    pid = db.insert_position(
        contract_id="0xclose_to_resolution", side="YES",
        size_usdc=4.0,                 # filled less than target
        entry_price=0.30,
        entry_time="2099-01-01T00:00:00",
        order_id="0xord_close",
        target_size_usdc=10.0,        # gap = $6
        shares=13.33,
        yes_token_id="tok",
        is_paper=0, fill_status="filled", strategy="top_k_hedged",
        event_id="evt_close_to_resolution",
        city="Lagos", date="2099-01-01",
    )
    # Confirm get_underfilled_positions returns it (no window check)
    underfilled = db.get_underfilled_positions()
    assert any(p["id"] == pid for p in underfilled), (
        "Underfilled TKH position should be returned by get_underfilled_positions "
        "regardless of how close the event is to resolving."
    )
