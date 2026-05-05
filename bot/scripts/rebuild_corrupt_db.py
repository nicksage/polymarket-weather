"""
rebuild_corrupt_db.py — Back up the corrupt signals.db and create a
fresh clean DB.

When PRAGMA integrity_check shows structural corruption (duplicate page
references), SQLite cannot self-repair via VACUUM.  The corrupt-but-
queryable approach is:

    1. Close all connections
    2. Rename signals.db (+ its WAL/SHM sidecars) to a timestamped backup
    3. Run init_db() on a fresh path → recreates full schema empty
    4. Verify integrity on the new DB

After this, the bot starts clean.  Historical activity_log /
monitor_health / temp_outcomes data is preserved in the backup file
and can be selectively recovered later via:
    sqlite3 signals.db.corrupt-YYYYMMDD-HHMMSS ".recover" > recovered.sql

Operational tables (positions, position_orders, processed_trade_events)
were already cleared by reset_for_new_strategy.py before this runs, so
no trading state is lost.

Usage:
    cd bot
    python -m scripts.rebuild_corrupt_db                # dry-run
    python -m scripts.rebuild_corrupt_db --apply        # commit
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH  # type: ignore


def _file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually rename + rebuild.  Without this, dry-run.")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return 1

    db_dir = os.path.dirname(DB_PATH) or "."
    db_name = os.path.basename(DB_PATH)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_main = os.path.join(db_dir, f"{db_name}.corrupt-{timestamp}")
    backup_wal  = os.path.join(db_dir, f"{db_name}-wal.corrupt-{timestamp}")
    backup_shm  = os.path.join(db_dir, f"{db_name}-shm.corrupt-{timestamp}")

    print()
    print(f"Source DB: {DB_PATH}  ({_file_size_mb(DB_PATH):.1f} MB)")
    print(f"Mode:      {'APPLY' if args.apply else 'DRY RUN'}")
    print()
    print("Plan:")
    print(f"  1. Rename  {db_name}            -> {os.path.basename(backup_main)}")
    if os.path.exists(DB_PATH + "-wal"):
        print(f"  2. Rename  {db_name}-wal        -> {os.path.basename(backup_wal)}")
    if os.path.exists(DB_PATH + "-shm"):
        print(f"  3. Rename  {db_name}-shm        -> {os.path.basename(backup_shm)}")
    print(f"  4. init_db() to recreate schema empty")
    print(f"  5. PRAGMA integrity_check on new DB")
    print()
    print("Loss: the historical scan data (temp_events ~55K rows, temp_outcomes ~600K)")
    print("      activity_log, monitor_health, etc. all moved to the backup file.")
    print("      Future bot operation does NOT need any of that.  You can mine the")
    print("      backup later via `sqlite3 <backup> '.recover'` if desired.")
    print()

    if not args.apply:
        print("Dry run — no files moved.  Re-run with --apply to commit.")
        return 0

    # ---- Rename files ----
    print("Renaming corrupt files...")
    moves = [(DB_PATH, backup_main)]
    if os.path.exists(DB_PATH + "-wal"):
        moves.append((DB_PATH + "-wal", backup_wal))
    if os.path.exists(DB_PATH + "-shm"):
        moves.append((DB_PATH + "-shm", backup_shm))
    for src, dst in moves:
        os.rename(src, dst)
        print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")

    # ---- Recreate schema ----
    print()
    print("Re-initializing fresh schema...")
    # Lazy-import db AFTER rename so it doesn't try to connect to the
    # now-renamed corrupt file.  init_db() will create a new file at DB_PATH.
    from db import init_db
    init_db()
    print(f"  Created fresh DB at {DB_PATH}")

    # ---- Verify ----
    print()
    print("Verifying integrity of new DB...")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        if len(rows) == 1 and rows[0][0] == "ok":
            print("  PASSED: integrity_check returned 'ok'")
        else:
            print(f"  WARNING: integrity_check returned {len(rows)} issues:")
            for r in rows[:5]:
                print(f"    {r[0]}")
            return 2

    # ---- Summary of new state ----
    print()
    new_size = _file_size_mb(DB_PATH)
    backup_size = _file_size_mb(backup_main)
    print(f"New DB size: {new_size:.2f} MB (down from corrupt {backup_size:.1f} MB)")
    print()
    print("Tables in fresh DB (all empty):")
    with sqlite3.connect(DB_PATH) as conn:
        tables = sorted(r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall())
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<35} {n:>4} rows")

    print()
    print("Done.  Bot is ready for a clean start.")
    print(f"Corrupt backup retained at: {os.path.basename(backup_main)}")
    print("Delete it manually once you're confident you don't need the history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
