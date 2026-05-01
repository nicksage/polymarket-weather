"""
test_user_ws_loop.py — Integration tests for the user-channel WS asyncio loop.

We patch `websockets.connect` with a controllable fake that lets us:
  * Verify the auth + subscribe payload is sent on connect
  * Feed trade / order events into the loop and verify they reach
    fill_handler.apply_trade_event / apply_order_event
  * Simulate disconnects and verify reconnect behavior
  * Verify the REST-backfill callback fires on every successful (re)connect

Why these tests are valuable
----------------------------
The WS loop is the highest-stakes code path that didn't have integration
coverage before — a reconnect bug means missed fills, a dispatch bug
means trades land in the DB wrong (or not at all).  Mocking is hard
because the loop runs in a background thread with its own event loop
under normal operation; the trick is to drive `_ws_loop()` directly
with `asyncio.run()` and use a stop event to break the loop after the
behavior we're checking.

Real network is not used.  These are integration tests in the sense
that the whole asyncio loop runs end-to-end against a faked transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import user_ws


# ---------------------------------------------------------------------------
# Fake websocket transport
# ---------------------------------------------------------------------------

class _FakeWS:
    """Minimal stand-in for the websockets client connection.

    `inbound` is a list of items to deliver via recv() in order.  Each
    item is either:
      * a string  → returned verbatim from recv()
      * the sentinel object EOF → raises ConnectionClosedError to simulate
        a network drop (triggers the loop's reconnect path)
    Once the queue is exhausted, recv() blocks indefinitely (which the
    test ends by setting the loop's stop_event)."""

    EOF = object()

    def __init__(self, inbound: list):
        self.inbound = list(inbound)
        self.sent: list[str] = []
        self._idx = 0

    async def send(self, msg: str) -> None:
        self.sent.append(msg)

    async def recv(self) -> str:
        if self._idx >= len(self.inbound):
            # Drain — block forever; the test stops the loop externally
            await asyncio.sleep(3600)
        item = self.inbound[self._idx]
        self._idx += 1
        if item is self.EOF:
            # Use the websockets exception type so the loop's except clause
            # treats this as a drop, not a hard failure.
            from websockets.exceptions import ConnectionClosedError
            raise ConnectionClosedError(None, None)
        return item


class _FakeWSContext:
    """async-context-manager wrapper around _FakeWS."""

    def __init__(self, ws: _FakeWS):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *exc):
        return False


def _make_fake_connect(*ws_sequence: _FakeWS):
    """Returns a callable suitable for `monkeypatch.setattr(websockets, 'connect', ...)`.

    Each call returns the next _FakeWS in the sequence.  Once exhausted,
    further calls raise OSError so the loop backs off."""
    seq = list(ws_sequence)
    idx = [0]

    def _fake_connect(*args, **kwargs):
        i = idx[0]
        idx[0] += 1
        if i >= len(seq):
            # Subsequent reconnect attempts after we've delivered all
            # planned WSs — error out so the loop doesn't spin forever
            raise OSError("no more fake WSs available")
        return _FakeWSContext(seq[i])

    return _fake_connect


@pytest.fixture(autouse=True)
def reset_user_ws_state(monkeypatch):
    """Each test starts with fresh module state."""
    user_ws._auth = None
    user_ws._my_wallet = None
    user_ws._subscribed_markets.clear()
    user_ws._backfill_callback = None
    user_ws._ws_stop_event.clear()
    # Don't actually load from DB during tests
    monkeypatch.setattr(user_ws, "load_open_market_ids", lambda: None)
    yield
    user_ws._ws_stop_event.set()


async def _drive_loop(stop_after_sec: float = 1.5) -> None:
    """Run _ws_loop until stop_after_sec has elapsed or it exits naturally."""
    task = asyncio.create_task(user_ws._ws_loop())
    await asyncio.sleep(stop_after_sec)
    user_ws._ws_stop_event.set()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()


# ===========================================================================
# Subscribe / auth
# ===========================================================================

def test_first_frame_is_auth_subscribe(monkeypatch):
    """On connect, the loop must send auth+markets+type as the FIRST frame."""
    user_ws._auth = ("api_k", "api_s", "api_p")
    user_ws._subscribed_markets.update({"0xmkt_a", "0xmkt_b"})

    fake_ws = _FakeWS(inbound=[])  # empty — recv blocks forever
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))

    asyncio.run(_drive_loop(stop_after_sec=0.4))

    # First sent message should be the subscribe payload
    assert len(fake_ws.sent) >= 1
    sub = json.loads(fake_ws.sent[0])
    assert sub["type"] == "user"
    assert sub["auth"] == {"apiKey": "api_k", "secret": "api_s", "passphrase": "api_p"}
    assert sorted(sub["markets"]) == ["0xmkt_a", "0xmkt_b"]


def test_no_subscribe_when_no_markets(monkeypatch):
    """If we have no markets to watch, the loop must NOT connect — opening
    a WS without a market filter would over-subscribe."""
    user_ws._auth = ("k", "s", "p")
    # _subscribed_markets is empty (cleared by fixture)

    connect_calls = [0]

    def _fake_connect(*a, **kw):
        connect_calls[0] += 1
        return _FakeWSContext(_FakeWS(inbound=[]))

    monkeypatch.setattr("websockets.connect", _fake_connect)
    asyncio.run(_drive_loop(stop_after_sec=0.4))
    assert connect_calls[0] == 0


# ===========================================================================
# Event dispatch — trade events
# ===========================================================================

def test_trade_event_routes_to_apply_trade_event(monkeypatch):
    """A 'trade' message arriving over the WS must reach fill_handler."""
    user_ws._auth = ("k", "s", "p")
    user_ws._my_wallet = "0xmywallet"
    user_ws._subscribed_markets.update({"0xmkt"})

    captured: list = []
    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: (
            captured.append(("trade", event, my_wallet))
            or {"action": "filled", "position_id": 1}
        ),
    )
    monkeypatch.setattr(
        "fill_handler.apply_order_event",
        lambda event: pytest.fail("apply_order_event should not be called"),
    )

    payload = json.dumps({
        "event_type":     "trade",
        "id":             "trade123",
        "status":         "confirmed",
        "taker_order_id": "0xord",
        "size":           100, "price": 0.5,
    })
    fake_ws = _FakeWS(inbound=[payload])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))

    asyncio.run(_drive_loop(stop_after_sec=0.5))

    assert len(captured) == 1
    kind, event, wallet = captured[0]
    assert kind == "trade"
    assert event["id"] == "trade123"
    assert wallet == "0xmywallet"


def test_order_event_routes_to_apply_order_event(monkeypatch):
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})

    captured: list = []
    monkeypatch.setattr(
        "fill_handler.apply_order_event",
        lambda event: (captured.append(event) or {"action": "placement_ack"}),
    )
    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: pytest.fail("trade dispatch unexpected"),
    )

    payload = json.dumps({
        "event_type": "order", "id": "0xord", "type": "PLACEMENT",
    })
    fake_ws = _FakeWS(inbound=[payload])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))

    asyncio.run(_drive_loop(stop_after_sec=0.5))

    assert len(captured) == 1
    assert captured[0]["id"] == "0xord"


def test_handles_list_of_events(monkeypatch):
    """Polymarket sometimes batches messages as a JSON array — each
    element must be dispatched individually."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})

    captured: list = []
    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: (captured.append(event) or {"action": "filled"}),
    )
    monkeypatch.setattr(
        "fill_handler.apply_order_event",
        lambda event: {"action": "placement_ack"},
    )

    batch = json.dumps([
        {"event_type": "trade", "id": "t1", "status": "confirmed"},
        {"event_type": "trade", "id": "t2", "status": "confirmed"},
        {"event_type": "trade", "id": "t3", "status": "matched"},
    ])
    fake_ws = _FakeWS(inbound=[batch])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))

    asyncio.run(_drive_loop(stop_after_sec=0.5))
    assert [e["id"] for e in captured] == ["t1", "t2", "t3"]


def test_pong_messages_are_ignored(monkeypatch):
    """The literal string 'PONG' is the keepalive reply — must NOT dispatch."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})

    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: pytest.fail("PONG should not dispatch"),
    )
    monkeypatch.setattr(
        "fill_handler.apply_order_event",
        lambda event: pytest.fail("PONG should not dispatch"),
    )

    fake_ws = _FakeWS(inbound=["PONG", "PONG"])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))
    asyncio.run(_drive_loop(stop_after_sec=0.4))


def test_malformed_json_is_dropped(monkeypatch):
    """Non-JSON garbage must not crash the loop."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})

    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: pytest.fail("garbage should not dispatch"),
    )
    monkeypatch.setattr(
        "fill_handler.apply_order_event",
        lambda event: pytest.fail("garbage should not dispatch"),
    )

    fake_ws = _FakeWS(inbound=["{not valid json", "definitely garbage"])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))
    # Loop should keep running, not raise
    asyncio.run(_drive_loop(stop_after_sec=0.4))


def test_dispatcher_swallows_handler_exceptions(monkeypatch):
    """If apply_trade_event raises, the loop must continue — one bad
    event can't take down the WS subscription for ALL positions."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})

    call_count = [0]

    def boom(event, my_wallet=None):
        call_count[0] += 1
        raise RuntimeError("simulated handler crash")

    monkeypatch.setattr("fill_handler.apply_trade_event", boom)

    payload = json.dumps({"event_type": "trade", "id": "t1", "status": "confirmed"})
    # Send 3 events — all 3 should reach the handler despite each raising
    fake_ws = _FakeWS(inbound=[payload, payload, payload])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))
    asyncio.run(_drive_loop(stop_after_sec=0.5))
    assert call_count[0] == 3


# ===========================================================================
# Reconnect + backfill callback
# ===========================================================================

def test_reconnect_after_disconnect(monkeypatch):
    """After the connection drops (ConnectionClosedError), the loop must
    reconnect and re-send the subscribe payload."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})
    monkeypatch.setattr(user_ws, "RECONNECT_DELAY", 0.05)
    monkeypatch.setattr(user_ws, "MAX_RECONNECT_DELAY", 0.05)

    # First WS sends one event then disconnects; second WS just blocks
    fake_ws_1 = _FakeWS(inbound=[
        json.dumps({"event_type": "trade", "id": "t1", "status": "confirmed"}),
        _FakeWS.EOF,
    ])
    fake_ws_2 = _FakeWS(inbound=[])
    monkeypatch.setattr(
        "websockets.connect", _make_fake_connect(fake_ws_1, fake_ws_2),
    )

    captured: list = []
    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: (
            captured.append(event) or {"action": "filled"}
        ),
    )

    asyncio.run(_drive_loop(stop_after_sec=1.0))

    # First WS got the subscribe + delivered one trade event
    assert len(fake_ws_1.sent) >= 1
    assert json.loads(fake_ws_1.sent[0])["type"] == "user"
    assert len(captured) == 1
    # Second WS also got a fresh subscribe message after reconnect
    assert len(fake_ws_2.sent) >= 1
    assert json.loads(fake_ws_2.sent[0])["type"] == "user"


def test_backfill_callback_fires_on_each_connect(monkeypatch):
    """The REST safety-net callback must run on every successful (re)connect
    so we catch fills that happened during downtime."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})
    monkeypatch.setattr(user_ws, "RECONNECT_DELAY", 0.05)
    monkeypatch.setattr(user_ws, "MAX_RECONNECT_DELAY", 0.05)

    backfill_calls = [0]
    user_ws.wire_backfill_callback(lambda: backfill_calls.__setitem__(0, backfill_calls[0] + 1))

    # Two fake WSs — first disconnects, second blocks
    fake_ws_1 = _FakeWS(inbound=[_FakeWS.EOF])
    fake_ws_2 = _FakeWS(inbound=[])
    monkeypatch.setattr(
        "websockets.connect", _make_fake_connect(fake_ws_1, fake_ws_2),
    )

    asyncio.run(_drive_loop(stop_after_sec=1.0))

    # Callback fires once per successful connect — two connects = two backfills
    assert backfill_calls[0] == 2


def test_backfill_callback_failure_does_not_crash_loop(monkeypatch):
    """If REST backfill raises, the WS must keep running — backfill is
    a 'nice to have' on reconnect, not a blocker."""
    user_ws._auth = ("k", "s", "p")
    user_ws._subscribed_markets.update({"0xmkt"})

    def boom():
        raise RuntimeError("backfill failed")

    user_ws.wire_backfill_callback(boom)

    # Send one trade event after the (failing) backfill
    payload = json.dumps({"event_type": "trade", "id": "t1", "status": "confirmed"})
    fake_ws = _FakeWS(inbound=[payload])
    monkeypatch.setattr("websockets.connect", _make_fake_connect(fake_ws))

    captured: list = []
    monkeypatch.setattr(
        "fill_handler.apply_trade_event",
        lambda event, my_wallet=None: (
            captured.append(event) or {"action": "filled"}
        ),
    )

    asyncio.run(_drive_loop(stop_after_sec=0.5))
    # Even though backfill threw, the trade event should still get dispatched
    assert len(captured) == 1


# ===========================================================================
# Public API
# ===========================================================================

def test_start_user_stream_requires_creds(monkeypatch):
    """Missing creds → must raise rather than silently degrading."""
    bad_client = type("X", (), {"creds": type("C", (), {
        "api_key": "", "api_secret": "", "api_passphrase": "",
    })()})()
    with pytest.raises(RuntimeError, match="incomplete API creds"):
        user_ws.start_user_stream(bad_client)


def test_start_user_stream_requires_creds_object(monkeypatch):
    """Client without .creds at all → also raises."""
    bad_client = object()
    with pytest.raises(RuntimeError, match="no .creds"):
        user_ws.start_user_stream(bad_client)


def test_add_markets_dedupes_empty_strings():
    user_ws.add_markets(["0xa", "", None, "0xb", "0xa"])
    assert user_ws._subscribed_markets == {"0xa", "0xb"}


def test_remove_markets_removes_existing():
    user_ws.add_markets(["0xa", "0xb", "0xc"])
    user_ws.remove_markets(["0xb"])
    assert user_ws._subscribed_markets == {"0xa", "0xc"}
