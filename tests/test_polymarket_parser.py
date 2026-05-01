"""
test_polymarket_parser.py — Defends the Polymarket Gamma API parsers
against silent API drift.

The Gamma API has changed shape before — field renames, JSON-string vs
list, optional fields appearing/disappearing.  Our discovery loop
absorbs whatever it gets and normalizes; if a field gets renamed and we
don't notice, we silently get zero signals.  These tests pin the
parser's expected behavior against fixture payloads that mirror real
responses.

Covers:
  * _normalize_sub_market: full happy path, JSON-string vs list fields,
    missing optional fields, malformed payloads, edge cases (single bin,
    "or higher" / "or below" range parsing through the question + groupItemTitle)
  * get_market_status: closed=True with winner, closed=True ambiguous,
    still-open, missing market, bad JSON
  * search_temp_high_events: end-to-end with mocked _fetch_events_by_tag_slug,
    deduplication, event-level liquidity filter, city-not-found drop

Fixtures here are INLINE synthetic dicts that mirror the real Gamma API
shape.  If the API contract changes, these tests will fail loudly —
exactly the protection we're after.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import polymarket


# ---------------------------------------------------------------------------
# Fixture builders — mirror real Gamma API shapes
# ---------------------------------------------------------------------------

def _make_market(
    *,
    gamma_id: str = "1234567",
    condition_id: str = "0xabc123",
    question: str = "Will Chicago's high temperature on April 9 be 72°F?",
    group_item_title: str = "72°F",
    yes_price: float = 0.42,
    no_price: float = 0.58,
    yes_token_id: str = "tok_yes_1",
    no_token_id: str = "tok_no_1",
    liquidity: float = 5000.0,
    volume: float = 12000.0,
    end_date: str = "2026-04-10T00:00:00Z",
    json_strings: bool = True,
) -> dict:
    """Build a single sub-market dict with realistic Gamma API field shapes.

    `json_strings=True` matches the API's quirky habit of returning
    arrays as JSON-encoded strings (e.g. outcomes='["YES","NO"]').
    Set False to test the alternative shape (raw lists).
    """
    outcomes  = ["YES", "NO"]
    prices    = [str(yes_price), str(no_price)]
    token_ids = [yes_token_id, no_token_id]
    return {
        "id":              gamma_id,
        "conditionId":     condition_id,
        "question":        question,
        "groupItemTitle":  group_item_title,
        "outcomes":        json.dumps(outcomes) if json_strings else outcomes,
        "outcomePrices":   json.dumps(prices)   if json_strings else prices,
        "clobTokenIds":    json.dumps(token_ids) if json_strings else token_ids,
        "liquidityNum":    liquidity,
        "volumeNum":       volume,
        "endDate":         end_date,
    }


def _make_event(
    *,
    event_id: str = "ev_chicago_apr9",
    title: str = "Highest temperature in Chicago on April 9, 2026",
    end_date: str = "2026-04-10T00:00:00Z",
    markets: list[dict] | None = None,
) -> dict:
    return {
        "id":      event_id,
        "title":   title,
        "endDate": end_date,
        "markets": markets or [],
    }


# ===========================================================================
# _normalize_sub_market
# ===========================================================================

def test_normalize_happy_path_json_strings():
    raw = _make_market(
        gamma_id="1886470",
        condition_id="0xdeadbeef",
        question="Will the high temp on Apr 9 be 72°F?",
        group_item_title="72°F",
        yes_price=0.42, no_price=0.58,
        yes_token_id="tok_yes_42", no_token_id="tok_no_42",
        liquidity=8500.0, volume=20000.0,
    )
    out = polymarket._normalize_sub_market(raw, "Chicago Apr 9 event")

    assert out is not None
    assert out["contract_id"] == "0xdeadbeef"
    assert out["gamma_market_id"] == "1886470"
    assert out["yes_price"] == pytest.approx(0.42)
    assert out["no_price"] == pytest.approx(0.58)
    assert out["yes_token_id"] == "tok_yes_42"
    assert out["no_token_id"] == "tok_no_42"
    assert out["liquidity_usd"] == pytest.approx(8500.0)
    assert out["volume_usd"] == pytest.approx(20000.0)
    # Range parser should pull 72°F out of "72°F" (groupItemTitle)
    assert out["range_low"] == pytest.approx(72.0)
    assert out["range_high"] == pytest.approx(72.0)
    assert out["unit"] == "fahrenheit"


def test_normalize_happy_path_raw_lists():
    """Some Gamma responses return arrays as raw lists, not JSON strings.
    The parser must handle both shapes."""
    raw = _make_market(json_strings=False)
    out = polymarket._normalize_sub_market(raw, "")
    assert out is not None
    assert out["yes_price"] == pytest.approx(0.42)


def test_normalize_celsius_range_or_below():
    raw = _make_market(group_item_title="16°C or below")
    out = polymarket._normalize_sub_market(raw, "")
    assert out["range_low"] is None
    assert out["range_high"] == pytest.approx(16.0)
    assert out["unit"] == "celsius"


def test_normalize_celsius_range_or_higher():
    raw = _make_market(group_item_title="25°C or higher")
    out = polymarket._normalize_sub_market(raw, "")
    assert out["range_low"] == pytest.approx(25.0)
    assert out["range_high"] is None
    assert out["unit"] == "celsius"


def test_normalize_falls_back_to_question_when_group_unparseable():
    """If groupItemTitle is empty, the parser tries the full question."""
    raw = _make_market(
        group_item_title="",
        question="Will Chicago's high be at least 25°C on April 9?",
    )
    out = polymarket._normalize_sub_market(raw, "")
    assert out["range_low"] == pytest.approx(25.0)


def test_normalize_keeps_unparseable_range_for_dashboard_visibility():
    """Per the comment in _normalize_sub_market: even when range parsing
    fails, the market is RETURNED so the dashboard can show it.  Edge
    calc skips, but the outcome doesn't disappear."""
    raw = _make_market(
        group_item_title="?",
        question="Some unparseable question",
    )
    out = polymarket._normalize_sub_market(raw, "")
    assert out is not None  # NOT None — returned even with no range
    assert out["range_low"] is None
    assert out["range_high"] is None


def test_normalize_rejects_missing_outcomes():
    """A market with fewer than 2 outcomes is malformed — drop it."""
    raw = _make_market()
    raw["outcomes"] = json.dumps(["YES"])  # only one outcome
    raw["outcomePrices"] = json.dumps(["1.0"])
    out = polymarket._normalize_sub_market(raw, "")
    assert out is None


def test_normalize_rejects_when_yes_no_labels_missing():
    """Outcomes must contain both YES and NO labels."""
    raw = _make_market()
    raw["outcomes"] = json.dumps(["TRUE", "FALSE"])
    out = polymarket._normalize_sub_market(raw, "")
    assert out is None


def test_normalize_handles_garbage_payload_gracefully():
    """Total junk shouldn't crash — return None."""
    out = polymarket._normalize_sub_market({"id": "x", "outcomes": "not-json{{"}, "")
    assert out is None


def test_normalize_falls_back_to_id_when_conditionId_absent():
    raw = _make_market()
    del raw["conditionId"]
    out = polymarket._normalize_sub_market(raw, "")
    assert out["contract_id"] == raw["id"]


def test_normalize_falls_back_for_volume_field_alias():
    """Polymarket has used both volumeNum and volume — accept either."""
    raw = _make_market(volume=0)  # volumeNum=0
    raw["volume"] = 9999.0  # legacy alias
    out = polymarket._normalize_sub_market(raw, "")
    assert out["volume_usd"] == pytest.approx(9999.0)


# ===========================================================================
# get_market_status
# ===========================================================================

def _market_status_response(
    *, closed: bool = False, yes_price: float = 0.5, no_price: float = 0.5,
    active: bool = True,
) -> dict:
    return {
        "id":            "999",
        "conditionId":   "0xfoo",
        "closed":        closed,
        "active":        active,
        "outcomes":      json.dumps(["YES", "NO"]),
        "outcomePrices": json.dumps([str(yes_price), str(no_price)]),
    }


def test_get_market_status_open_returns_prices(monkeypatch):
    monkeypatch.setattr(
        polymarket, "_gamma_get",
        lambda path, params: _market_status_response(closed=False, yes_price=0.42, no_price=0.58),
    )
    s = polymarket.get_market_status("0xfoo", gamma_market_id="999")
    assert s is not None
    assert s["closed"] is False
    assert s["winner"] is None
    assert s["yes_price"] == pytest.approx(0.42)


def test_get_market_status_closed_yes_winner(monkeypatch):
    """Closed market with YES at 1.0 → winner=YES."""
    monkeypatch.setattr(
        polymarket, "_gamma_get",
        lambda path, params: _market_status_response(closed=True, yes_price=1.0, no_price=0.0),
    )
    s = polymarket.get_market_status("0xfoo", gamma_market_id="999")
    assert s["closed"] is True
    assert s["winner"] == "YES"


def test_get_market_status_closed_no_winner(monkeypatch):
    monkeypatch.setattr(
        polymarket, "_gamma_get",
        lambda path, params: _market_status_response(closed=True, yes_price=0.0, no_price=1.0),
    )
    s = polymarket.get_market_status("0xfoo", gamma_market_id="999")
    assert s["winner"] == "NO"


def test_get_market_status_closed_ambiguous(monkeypatch):
    """Closed but neither side at 0.99 → winner unclear (None)."""
    monkeypatch.setattr(
        polymarket, "_gamma_get",
        lambda path, params: _market_status_response(closed=True, yes_price=0.5, no_price=0.5),
    )
    s = polymarket.get_market_status("0xfoo", gamma_market_id="999")
    assert s["closed"] is True
    assert s["winner"] is None


def test_get_market_status_api_returns_none(monkeypatch):
    """Empty Gamma response → return None instead of crashing."""
    monkeypatch.setattr(polymarket, "_gamma_get", lambda *a, **k: None)
    assert polymarket.get_market_status("0xfoo", gamma_market_id="999") is None


def test_get_market_status_api_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated 503")
    monkeypatch.setattr(polymarket, "_gamma_get", boom)
    assert polymarket.get_market_status("0xfoo", gamma_market_id="999") is None


# ===========================================================================
# search_temp_high_events — end-to-end with mocked tag fetch
# ===========================================================================

def test_search_temp_high_events_normalizes_full_event(monkeypatch):
    """Feed in a synthetic event with two sub-markets — verify the full
    pipeline produces correctly-shaped output."""
    event = _make_event(
        title="Highest temperature in Chicago on April 9, 2026",
        end_date="2026-04-10T00:00:00Z",
        markets=[
            _make_market(
                gamma_id="1001", condition_id="0xa1",
                group_item_title="72°F", yes_price=0.30, no_price=0.70,
            ),
            _make_market(
                gamma_id="1002", condition_id="0xa2",
                group_item_title="73°F", yes_price=0.40, no_price=0.60,
            ),
        ],
    )
    monkeypatch.setattr(polymarket, "_fetch_events_by_tag_slug", lambda slug: [event])

    out = polymarket.search_temp_high_events(min_liquidity=0)
    assert len(out) == 1
    ev = out[0]
    # Event-level fields — city case mirrors how it's stored in CITY_COORDS
    assert ev["city"].lower() == "chicago"
    # Date comes from endDate (preferred) — title fallback only fires when endDate missing
    assert ev["date"] == "2026-04-10"
    # Sub-markets
    assert len(ev["outcomes"]) == 2
    contract_ids = sorted(o["contract_id"] for o in ev["outcomes"])
    assert contract_ids == ["0xa1", "0xa2"]


def test_search_dedupes_events_by_id(monkeypatch):
    """Same event_id appearing twice — only the first is processed."""
    e1 = _make_event(event_id="dupe", markets=[_make_market(condition_id="0xa")])
    e2 = _make_event(event_id="dupe", markets=[_make_market(condition_id="0xb")])
    monkeypatch.setattr(polymarket, "_fetch_events_by_tag_slug", lambda slug: [e1, e2])
    out = polymarket.search_temp_high_events(min_liquidity=0)
    assert len(out) == 1
    assert out[0]["outcomes"][0]["contract_id"] == "0xa"


def test_search_skips_events_without_recognizable_city(monkeypatch):
    """If we can't extract a city/date, drop the event (would crash later)."""
    bad = _make_event(
        title="Some random event with no city",
        markets=[_make_market()],
    )
    monkeypatch.setattr(polymarket, "_fetch_events_by_tag_slug", lambda slug: [bad])
    out = polymarket.search_temp_high_events(min_liquidity=0)
    assert out == []


def test_search_skips_non_temperature_events(monkeypatch):
    """Even if the tag slug returns something — title filter rejects non-temp."""
    not_temp = _make_event(
        title="Will Trump win the election in November?",
        markets=[_make_market()],
    )
    monkeypatch.setattr(polymarket, "_fetch_events_by_tag_slug", lambda slug: [not_temp])
    assert polymarket.search_temp_high_events(min_liquidity=0) == []


# ===========================================================================
# Temperature range parser direct tests
# ===========================================================================

# ===========================================================================
# get_data_api_positions — regression tests for two API drift bugs
# ===========================================================================
# Pinned 2026-04-29 after the live bot hit a 400 Bad Request:
#   1. Polymarket renamed the query param `address` → `user`
#   2. The response `asset` field changed from {token_id: ...} to a
#      bare string holding the token id directly
# Both broke silently — the broad except in get_data_api_positions
# turned each into a logged warning + empty result, so on-chain P&L
# enrichment quietly stopped working without crashing the bot.

def _make_data_api_position_row(
    *,
    asset_id: str = "68973461397167160251369637004556441740086938961103954922828957470690944606689",
    title: str = "Will Chicago's high temp on April 9 be 72°F?",
    outcome: str = "Yes",
    size: float = 100.0,
    avg_price: float = 0.45,
    current_value: float = 60.0,
    condition_id: str = "0xe31609522ec45e45fdf82fcf238193e4418b4e103bb14f9b1f80931218410473",
    end_date: str = "2026-04-09",
    negative_risk: bool = False,
) -> dict:
    """Synthetic row matching the live 2026-04 Data API shape."""
    return {
        "proxyWallet":   "0x8ed51f724f949d019b93890f6ef81dd17a1c7c3a",
        "asset":         asset_id,
        "conditionId":   condition_id,
        "size":          size,
        "avgPrice":      avg_price,
        "initialValue":  size * avg_price,
        "currentValue":  current_value,
        "cashPnl":       current_value - (size * avg_price),
        "percentPnl":    -10.0,
        "totalBought":   size,
        "realizedPnl":   0.0,
        "percentRealizedPnl": 0.0,
        "curPrice":      0.6,
        "redeemable":    False,
        "mergeable":     False,
        "title":         title,
        "slug":          "test-slug",
        "icon":          "",
        "eventId":       "12345",
        "eventSlug":     "test-event",
        "outcome":       outcome,
        "outcomeIndex":  0 if outcome == "Yes" else 1,
        "oppositeOutcome": "No" if outcome == "Yes" else "Yes",
        "oppositeAsset": "999",
        "endDate":       end_date,
        "negativeRisk":  negative_risk,
    }


def test_data_api_uses_user_param_not_address(monkeypatch):
    """The Polymarket Data API requires `user`, not `address`.  Pinning
    this prevents a regression to the old param name (which silently 400s)."""
    captured_params = {}

    class _FakeResp:
        status_code = 200
        text = "[]"
        def raise_for_status(self): pass
        def json(self): return []

    def fake_get(url, params=None, timeout=None):
        captured_params.update(params or {})
        return _FakeResp()

    monkeypatch.setattr("polymarket.httpx.get", fake_get)
    polymarket.get_data_api_positions("0xdeadbeef")
    assert "user" in captured_params, (
        f"must use `user` param (Polymarket renamed from `address` in 2026-04); "
        f"got {captured_params!r}"
    )
    assert captured_params["user"] == "0xdeadbeef"
    assert "address" not in captured_params, (
        "should not send the obsolete `address` param"
    )


def test_data_api_parses_asset_as_string(monkeypatch):
    """The 2026-04 response shape: `asset` is a top-level string (the
    ERC-1155 token id), not a dict.  Older code did asset.get('token_id')
    which would AttributeError on a string and get swallowed."""
    rows = [_make_data_api_position_row(
        asset_id="11111111111111111111111111111111111111111111111111111111111111111111111111111",
        title="Test Market", outcome="Yes",
        size=42.0, avg_price=0.50, current_value=21.0,
    )]

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return rows

    monkeypatch.setattr("polymarket.httpx.get",
                        lambda url, params=None, timeout=None: _FakeResp())

    out = polymarket.get_data_api_positions("0xdeadbeef")
    assert len(out) == 1
    p = out[0]
    assert p["token_id"] == ("1" * 77)
    assert p["title"] == "Test Market"
    assert p["outcome"] == "Yes"
    assert p["size"] == pytest.approx(42.0)
    assert p["avg_price"] == pytest.approx(0.50)
    assert p["current_value"] == pytest.approx(21.0)


def test_data_api_back_compat_with_legacy_asset_dict(monkeypatch):
    """If Polymarket ever reverts `asset` to a dict shape, the parser
    must still handle it — defensive against another flip-flop."""
    legacy_row = {
        "asset": {"token_id": "9999", "outcome": "No"},
        "size": 10.0, "avgPrice": 0.30,
        "initialValue": 3.0, "currentValue": 4.0,
        "cashPnl": 1.0, "percentPnl": 33.3,
        "title": "Legacy shape",
    }

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return [legacy_row]

    monkeypatch.setattr("polymarket.httpx.get",
                        lambda url, params=None, timeout=None: _FakeResp())

    out = polymarket.get_data_api_positions("0xdeadbeef")
    assert len(out) == 1
    assert out[0]["token_id"] == "9999"
    assert out[0]["outcome"] == "No"


def test_data_api_returns_empty_on_400(monkeypatch):
    """A 400 from Polymarket (e.g. param rename, deprecated endpoint)
    must result in an empty list, not a crash — the rest of the monitor
    cycle has to keep going."""
    import httpx
    class _FakeResp:
        status_code = 400
        text = '{"error":"required query param \'user\' not provided"}'
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "400 Bad Request", request=None, response=None,
            )
        def json(self): return {}

    monkeypatch.setattr("polymarket.httpx.get",
                        lambda url, params=None, timeout=None: _FakeResp())
    out = polymarket.get_data_api_positions("0xdeadbeef")
    assert out == []


def test_data_api_exposes_new_optional_fields(monkeypatch):
    """conditionId, endDate, and negativeRisk are now exposed at the top
    level — surface them in the parsed dict for downstream use."""
    rows = [_make_data_api_position_row(
        condition_id="0xCAFE",
        end_date="2026-12-31",
        negative_risk=True,
    )]

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return rows

    monkeypatch.setattr("polymarket.httpx.get",
                        lambda url, params=None, timeout=None: _FakeResp())

    out = polymarket.get_data_api_positions("0xdeadbeef")
    assert out[0]["condition_id"] == "0xCAFE"
    assert out[0]["end_date"] == "2026-12-31"
    assert out[0]["negative_risk"] is True


def test_data_api_empty_wallet_short_circuits(monkeypatch):
    """No HTTP call should fire when wallet is empty — protects against
    accidentally querying for the empty-string user."""
    called = [False]
    def fake_get(*a, **kw):
        called[0] = True
        raise AssertionError("should not have hit network")
    monkeypatch.setattr("polymarket.httpx.get", fake_get)
    assert polymarket.get_data_api_positions("") == []
    assert called[0] is False


# ===========================================================================
# Temperature range parser direct tests
# ===========================================================================

@pytest.mark.parametrize("text,expected_low,expected_high,expected_unit", [
    ("72°F",            72.0, 72.0, "fahrenheit"),
    ("17°C",            17.0, 17.0, "celsius"),
    ("16°C or below",   None, 16.0, "celsius"),
    ("25°C or higher",  25.0, None, "celsius"),
    ("90°F or above",   90.0, None, "fahrenheit"),
    ("at least 25°C",   25.0, None, "celsius"),
    ("below 16°C",      None, 16.0, "celsius"),
    ("under 90°F",      None, 90.0, "fahrenheit"),
])
def test_parse_temperature_range_patterns(text, expected_low, expected_high, expected_unit):
    low, high, unit = polymarket.parse_temperature_range(text)
    assert low == expected_low
    assert high == expected_high
    assert unit == expected_unit


def test_parse_temperature_range_unparseable_returns_none():
    low, high, unit = polymarket.parse_temperature_range("garbage with no numbers")
    assert low is None
    assert high is None
    # Unit defaults to celsius when no °F marker
    assert unit == "celsius"
