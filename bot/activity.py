"""
activity.py — Critical-event activity log.

Single function `log_activity(category, message, ...)` that fans out to:
  1. The terminal (via the standard logging system — same handlers as
     bot.log, so the operator sees every critical action live)
  2. logs/activity.log — a dedicated, low-noise file with timestamps
     for after-the-fact debugging without grepping bot.log
  3. The activity_log SQLite table — for the dashboard "Activity" tab

This is intentionally thin.  Don't add filtering logic, rate limiting, or
buffering here — the volume is low (handful of events per hour) and the
write paths are independent.  If one fails, the others still proceed.

Categories
----------
BUY      entry order placed / skipped (retry cap) / failed at placement
SELL     exit order placed (initial trigger, ladder advance, cross-spread)
TOPUP    top-up order placed
FILL     order CONFIRMED on chain (entry / exit / topup)
CANCEL   order cancelled (operator, monitor sweep, top-up cleanup, etc.)
FAIL     trade FAILED on chain after MATCHED — capital released
CLOSE    position resolved by market close (won / lost)
RISK     bankroll capped, drift detected, retry cap hit, stop-loss fired
WS       user-channel WebSocket connect / disconnect / reconnect
SYSTEM   bot startup / shutdown / config changes

Levels: INFO (normal), WARN (degraded but expected), ERROR (action needed).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

# Public logger name — same hierarchy as everything else in the bot, so
# operators see activity events in their normal terminal alongside bot
# noise.  The dedicated file handler attached below filters this logger
# down to ONLY activity events going into activity.log.
_LOGGER_NAME = "activity"
logger = logging.getLogger(_LOGGER_NAME)
logger.setLevel(logging.INFO)
# Don't double-print: the parent logger ('') already has handlers wired
# to terminal + bot.log.  We add the file handler below for activity.log
# but propagate up so terminal/bot.log also see it.

_file_handler_lock = threading.Lock()
_file_handler_attached = False


def _ensure_file_handler() -> None:
    """Lazily attach the dedicated activity.log file handler — once per
    process.  Path resolution mirrors main.py's _LOGS_DIR setup so we
    land next to bot.log."""
    global _file_handler_attached
    if _file_handler_attached:
        return
    with _file_handler_lock:
        if _file_handler_attached:
            return
        try:
            bot_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(bot_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            handler = logging.FileHandler(
                os.path.join(logs_dir, "activity.log"),
                encoding="utf-8",
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(handler)
            _file_handler_attached = True
        except Exception:
            # Never crash on log-init failure; the DB + terminal paths
            # still work without the file.
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_LEVELS = {"INFO", "WARN", "WARNING", "ERROR"}
_VALID_CATEGORIES = {
    "BUY", "SELL", "TOPUP",
    "FILL", "FAIL", "CANCEL", "CLOSE",
    "RISK", "WS", "SYSTEM", "DRIFT",
}


def log_activity(
    category: str,
    message: str,
    *,
    level: str = "INFO",
    position_id: int | None = None,
    **metadata: Any,
) -> None:
    """Log a critical bot action to terminal + activity.log + activity_log table.

    `category` is one of the values in _VALID_CATEGORIES (above) — used
    to filter the dashboard Activity tab.  Unknown categories are still
    accepted (they just won't show up in the category filter dropdown).

    `message` is the human-readable summary.  Aim for ONE LINE that
    answers "what just happened" without forcing the reader to dig into
    surrounding log context.  Include the position_id, contract, and
    key numbers inline.

    `level` is INFO / WARN / ERROR.  Use WARN for "expected degradation"
    (e.g. WS reconnect, retry cap hit) and ERROR for "operator should
    look" (e.g. order placement failed, on-chain trade FAILED).

    `position_id` is optional — when set, the dashboard can link the
    event to the position row.

    `**metadata` is captured as JSON in the DB column for later inspection.
    Skipped from the terminal/file message to keep them readable.
    """
    _ensure_file_handler()

    level_upper = level.upper()
    if level_upper == "WARNING":
        level_upper = "WARN"
    if level_upper not in _VALID_LEVELS:
        level_upper = "INFO"

    # Compose the human-readable line — same shape across all 3 sinks.
    pid_tag = f" pos={position_id}" if position_id is not None else ""
    line = f"[{category}]{pid_tag} {message}"

    # Sink 1+2: terminal + activity.log (via the activity logger)
    log_method = {
        "INFO":  logger.info,
        "WARN":  logger.warning,
        "ERROR": logger.error,
    }[level_upper]
    log_method(line)

    # Sink 3: activity_log SQLite table.  Best-effort — never raise.
    try:
        from db import insert_activity_log
        insert_activity_log(
            timestamp   = datetime.now(timezone.utc).isoformat(),
            level       = level_upper,
            category    = category,
            message     = message,
            position_id = position_id,
            metadata    = json.dumps(metadata) if metadata else None,
        )
    except Exception:
        # Don't let DB write errors mask the main flow.  The terminal +
        # file paths above already preserve the event.
        pass
