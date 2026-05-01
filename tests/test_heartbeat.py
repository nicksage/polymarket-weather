"""
test_heartbeat.py — Tests for the Polymarket maker-side keep-alive.

Heartbeat is a daemon thread, so most tests use a very short interval
(0.05s) to keep them fast.  We patch HEARTBEAT_INTERVAL_SEC and verify
the loop's behavior across cycles within ~200ms.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from unittest.mock import MagicMock

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import heartbeat


@pytest.fixture(autouse=True)
def fast_interval(monkeypatch):
    """Run the loop fast so tests are sub-second."""
    monkeypatch.setattr(heartbeat, "HEARTBEAT_INTERVAL_SEC", 0.05)
    yield


@pytest.fixture(autouse=True)
def fresh_state():
    """Reset the module-level state before each test so they don't bleed
    into each other (heartbeat keeps its state in a module dict)."""
    # Make sure no leftover thread is running
    heartbeat.stop_heartbeat(timeout=1.0)
    with heartbeat._state_lock:
        heartbeat._state.update({
            "running":              False,
            "last_success_ts":      None,
            "last_failure_ts":      None,
            "last_error":           None,
            "consecutive_failures": 0,
            "total_sent":           0,
            "total_failed":         0,
        })
    heartbeat._stop_event.clear()
    heartbeat._thread = None
    yield
    heartbeat.stop_heartbeat(timeout=1.0)


def test_start_spawns_thread_and_calls_post_heartbeat():
    client = MagicMock()
    client.post_heartbeat.return_value = {"heartbeat_id": "abc"}

    heartbeat.start_heartbeat(client)

    # Wait for ~3 cycles to confirm repeated calls
    time.sleep(0.20)

    assert client.post_heartbeat.call_count >= 2
    assert heartbeat.get_heartbeat_state()["running"] is True
    assert heartbeat.get_heartbeat_state()["total_sent"] >= 2


def test_passes_returned_heartbeat_id_back_in_next_call():
    """Polymarket returns a new heartbeat_id each call; the SDK should
    feed it back to detect dropped sequences."""
    client = MagicMock()
    # First call returns id 'first', second call returns id 'second'
    responses = [{"heartbeat_id": "first"}, {"heartbeat_id": "second"}, {}]
    client.post_heartbeat.side_effect = responses + [{}] * 100

    heartbeat.start_heartbeat(client)
    time.sleep(0.20)
    heartbeat.stop_heartbeat()

    calls = client.post_heartbeat.call_args_list
    assert len(calls) >= 3
    # First call: empty id
    assert calls[0].kwargs.get("heartbeat_id", "") == ""
    # Second call: id from first response
    assert calls[1].kwargs.get("heartbeat_id") == "first"
    # Third call: id from second response
    assert calls[2].kwargs.get("heartbeat_id") == "second"


def test_fallback_to_id_field_in_response():
    """Some response shapes may use 'id' instead of 'heartbeat_id'.
    The loop should accept either."""
    client = MagicMock()
    client.post_heartbeat.side_effect = [
        {"id": "from_id_field"},
        {},
        {},
    ] + [{}] * 100

    heartbeat.start_heartbeat(client)
    time.sleep(0.15)
    heartbeat.stop_heartbeat()

    calls = client.post_heartbeat.call_args_list
    assert calls[1].kwargs.get("heartbeat_id") == "from_id_field"


def test_swallows_errors_and_continues():
    """A failed heartbeat must not kill the loop — Polymarket APIs have
    transient errors; the loop should retry on the next interval."""
    client = MagicMock()
    # Pattern: error, error, success, success, ...
    client.post_heartbeat.side_effect = [
        RuntimeError("transient 500"),
        ConnectionError("network blip"),
        {"heartbeat_id": "ok1"},
        {"heartbeat_id": "ok2"},
    ] + [{"heartbeat_id": "ok"}] * 100

    heartbeat.start_heartbeat(client)
    time.sleep(0.30)
    heartbeat.stop_heartbeat()

    state = heartbeat.get_heartbeat_state()
    assert state["total_sent"] >= 1   # at least one success after the errors
    assert state["total_failed"] >= 2  # both errors counted


def test_consecutive_failure_count_resets_on_success():
    """After a streak of failures, a single success resets the counter."""
    client = MagicMock()
    client.post_heartbeat.side_effect = [
        RuntimeError("a"),
        RuntimeError("b"),
        {"heartbeat_id": "ok"},     # success resets counter
        RuntimeError("c"),
    ] + [{"heartbeat_id": "ok"}] * 100

    heartbeat.start_heartbeat(client)
    time.sleep(0.40)   # ~8 cycles
    heartbeat.stop_heartbeat()

    state = heartbeat.get_heartbeat_state()
    # After the series above the counter should NOT be ≥4 (the third
    # success in the pattern resets it), and subsequent successes drive it to 0.
    # Specifically, by the time we stop, consecutive_failures should be 0
    # because the rest of the side_effect returns OK.
    assert state["consecutive_failures"] == 0


def test_stop_event_exits_promptly():
    """stop_heartbeat() should make the loop exit within one interval."""
    client = MagicMock()
    client.post_heartbeat.return_value = {"heartbeat_id": "x"}

    heartbeat.start_heartbeat(client)
    time.sleep(0.10)

    t0 = time.time()
    heartbeat.stop_heartbeat(timeout=2.0)
    elapsed = time.time() - t0

    # Should exit within ~one interval (0.05s) + a small buffer
    assert elapsed < 0.3
    assert heartbeat.get_heartbeat_state()["running"] is False


def test_start_is_idempotent():
    """Calling start_heartbeat twice should NOT spawn two threads."""
    client = MagicMock()
    client.post_heartbeat.return_value = {}

    heartbeat.start_heartbeat(client)
    first_thread = heartbeat._thread

    heartbeat.start_heartbeat(client)  # second call
    second_thread = heartbeat._thread

    assert first_thread is second_thread


def test_start_with_none_client_is_noop():
    """If no CLOB client is available (paper mode, no creds), start
    should log + return without spawning a thread."""
    heartbeat.start_heartbeat(client=None)
    assert heartbeat._thread is None
    assert heartbeat.get_heartbeat_state()["running"] is False


def test_get_heartbeat_state_returns_snapshot_not_reference():
    """The returned dict must be a copy — callers mutating it must not
    affect the daemon's internal state."""
    client = MagicMock()
    client.post_heartbeat.return_value = {}

    heartbeat.start_heartbeat(client)
    time.sleep(0.10)
    snap = heartbeat.get_heartbeat_state()
    snap["running"] = "tampered"      # mutate the snapshot
    snap["total_sent"] = -999
    fresh = heartbeat.get_heartbeat_state()
    assert fresh["running"] is True
    assert fresh["total_sent"] >= 1


def test_warn_after_consecutive_failures(caplog):
    """After WARN_AFTER_CONSEC_FAILURES failures in a row, log a warning.
    Earlier failures only log at debug level."""
    import logging
    client = MagicMock()
    client.post_heartbeat.side_effect = RuntimeError("persistent")

    with caplog.at_level(logging.WARNING, logger="heartbeat"):
        heartbeat.start_heartbeat(client)
        time.sleep(0.40)   # ~8 cycles, well past the 3-failure threshold
        heartbeat.stop_heartbeat()

    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "consecutive failures" in r.message]
    assert len(warns) >= 1


def test_thread_is_daemon_so_process_can_exit():
    """The loop runs on a daemon thread so the bot process can exit
    cleanly without an explicit stop."""
    client = MagicMock()
    client.post_heartbeat.return_value = {}

    heartbeat.start_heartbeat(client)
    assert heartbeat._thread.daemon is True
