"""
One-shot migration: convert every positions.entry_time value stored as UTC
(`+00:00` suffix) to US Central (`America/Chicago`) while preserving the
actual instant.

Usage:
    # With the bot STOPPED so no concurrent writes occur:
    cd bot
    python scripts/migrate_entry_time_to_cst.py

Safe to re-run — rows already carrying a Chicago offset are skipped.
"""
import os
import sys
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

# Make bot/ importable regardless of where we're launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT  = os.path.dirname(_HERE)
sys.path.insert(0, _BOT)

from config import DB_PATH  # noqa: E402

CHI = ZoneInfo("America/Chicago")


def main() -> None:
    print(f"DB: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("DB not found — nothing to migrate.")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, entry_time FROM positions WHERE entry_time IS NOT NULL AND entry_time != ''"
    ).fetchall()

    converted = 0
    skipped   = 0
    errors    = 0

    for pos_id, ts in rows:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                skipped += 1
                continue
            # Skip if already Chicago
            if dt.utcoffset() == dt.astimezone(CHI).utcoffset() and \
               dt.tzinfo is not None and \
               dt.isoformat() == dt.astimezone(CHI).isoformat():
                skipped += 1
                continue
            new_ts = dt.astimezone(CHI).isoformat()
            if new_ts == ts:
                skipped += 1
                continue
            conn.execute(
                "UPDATE positions SET entry_time = ? WHERE id = ?",
                (new_ts, pos_id),
            )
            converted += 1
        except Exception as e:
            errors += 1
            print(f"  id={pos_id} '{ts}' -> {e}")

    conn.commit()
    conn.close()
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")


if __name__ == "__main__":
    main()
