"""
test_activity.py — Tests for the activity logging infrastructure.

Covers:
  * log_activity round-trips through DB
  * Levels normalize (INFO / WARN / WARNING / ERROR / unknown)
  * position_id and metadata persist correctly
  * get_recent_activity filters: categories, levels, since_iso, limit
  * get_activity_categories distinct list
  * activity.log file gets written to (single integration test)
  * Failure in DB layer doesn't break the call path (best-effort guarantee)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

_BOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"
)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    db.init_db()
    yield str(db_path)


# ===========================================================================
# log_activity → DB round-trip
# ===========================================================================

def test_log_activity_basic(temp_db):
    import activity
    activity.log_activity("BUY", "test buy", position_id=42)
    rows = db.get_recent_activity(limit=10)
    assert len(rows) == 1
    assert rows[0]["category"] == "BUY"
    assert rows[0]["message"] == "test buy"
    assert rows[0]["position_id"] == 42
    assert rows[0]["level"] == "INFO"


def test_log_activity_metadata_persists(temp_db):
    import activity
    activity.log_activity(
        "FILL", "test fill", position_id=7,
        side="YES", price=0.55, shares=100.0,
    )
    rows = db.get_recent_activity(limit=10)
    md = json.loads(rows[0]["metadata"])
    assert md == {"side": "YES", "price": 0.55, "shares": 100.0}


def test_log_activity_no_metadata_stores_null(temp_db):
    import activity
    activity.log_activity("SYSTEM", "bot started")
    rows = db.get_recent_activity(limit=10)
    assert rows[0]["metadata"] is None


def test_log_activity_level_normalization(temp_db):
    import activity
    activity.log_activity("BUY", "info-level", level="INFO")
    activity.log_activity("BUY", "warn", level="WARN")
    activity.log_activity("BUY", "warning", level="WARNING")
    activity.log_activity("BUY", "error", level="ERROR")
    activity.log_activity("BUY", "garbage falls back to INFO", level="garbage")
    rows = db.get_recent_activity(limit=10)
    levels = [r["level"] for r in rows]
    # newest first
    assert levels == ["INFO", "ERROR", "WARN", "WARN", "INFO"]


# ===========================================================================
# get_recent_activity filters
# ===========================================================================

def test_get_recent_activity_category_filter(temp_db):
    import activity
    activity.log_activity("BUY", "b1")
    activity.log_activity("SELL", "s1")
    activity.log_activity("FILL", "f1")
    only_sell = db.get_recent_activity(categories=["SELL"])
    assert len(only_sell) == 1
    assert only_sell[0]["category"] == "SELL"
    sell_or_fill = db.get_recent_activity(categories=["SELL", "FILL"])
    assert len(sell_or_fill) == 2


def test_get_recent_activity_level_filter(temp_db):
    import activity
    activity.log_activity("BUY", "ok", level="INFO")
    activity.log_activity("BUY", "warn", level="WARN")
    activity.log_activity("BUY", "err", level="ERROR")
    err_only = db.get_recent_activity(levels=["ERROR"])
    assert len(err_only) == 1
    assert err_only[0]["message"] == "err"


def test_get_recent_activity_since_filter(temp_db):
    import activity
    activity.log_activity("BUY", "b1")
    activity.log_activity("BUY", "b2")
    # Filter for "now or later" (effectively returns nothing in same second)
    future_iso = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert len(db.get_recent_activity(since_iso=future_iso)) == 0
    # Filter for "1h ago" returns both
    past_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert len(db.get_recent_activity(since_iso=past_iso)) == 2


def test_get_recent_activity_limit_caps(temp_db):
    import activity
    for i in range(5):
        activity.log_activity("BUY", f"b{i}")
    rows = db.get_recent_activity(limit=3)
    assert len(rows) == 3


def test_get_activity_categories_distinct(temp_db):
    import activity
    activity.log_activity("BUY", "a")
    activity.log_activity("BUY", "b")
    activity.log_activity("SELL", "c")
    activity.log_activity("FILL", "d")
    cats = db.get_activity_categories()
    assert cats == ["BUY", "FILL", "SELL"]


# ===========================================================================
# Defensive guarantees
# ===========================================================================

def test_log_activity_swallows_db_errors(temp_db, monkeypatch):
    """If insert_activity_log raises, log_activity must NOT propagate
    the exception — the file + terminal sinks should still fire."""
    import activity

    def boom(**kwargs):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr("db.insert_activity_log", boom)
    # Should not raise
    activity.log_activity("BUY", "still works")


def test_log_activity_writes_to_activity_log_file(temp_db, tmp_path, monkeypatch):
    """Verify the dedicated activity.log file gets a row written.

    We patch the activity module's file-handler init to point at a temp
    location so we don't pollute the real bot/logs/ during tests.
    """
    import logging
    import activity

    # Reset module state from any prior test
    activity._file_handler_attached = False
    for h in list(activity.logger.handlers):
        activity.logger.removeHandler(h)

    log_path = tmp_path / "activity.log"
    handler = logging.FileHandler(str(log_path))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    activity.logger.addHandler(handler)
    activity._file_handler_attached = True

    activity.log_activity("BUY", "round-trip", position_id=99)

    handler.flush()
    contents = log_path.read_text()
    assert "BUY" in contents
    assert "round-trip" in contents
    assert "pos=99" in contents
