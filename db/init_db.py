"""Create main.db, enable WAL mode, apply schema. Idempotent."""
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DB_PATH", str(REPO_ROOT / "db" / "main.db"))
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"

# Columns added after a table's initial release. CREATE TABLE IF NOT EXISTS
# won't alter an existing table, so add any missing ones here idempotently.
# (table, column, type)
COLUMN_MIGRATIONS = [
    ("events",              "discovered_at_local", "TEXT"),
    ("price_snapshots",     "recorded_at_local",   "TEXT"),
    ("resolutions",         "resolved_at_local",   "TEXT"),
    ("weather_forecasts",   "fetched_at_local",    "TEXT"),
    ("weather_observations", "fetched_at_local",   "TEXT"),
    ("twc_current",         "fetched_at_local",    "TEXT"),
    ("twc_hourly",          "fetched_at_local",    "TEXT"),
    ("twc_fifteenminute",   "fetched_at_local",    "TEXT"),
    ("twc_probabilistic",   "fetched_at_local",    "TEXT"),
]


def _apply_column_migrations(conn) -> int:
    added = 0
    for table, column, coltype in COLUMN_MIGRATIONS:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            continue
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            print(f"  + {table}.{column}")
            added += 1
    return added


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    n = _apply_column_migrations(conn)
    conn.commit()
    conn.close()
    print(f"Initialized DB at {DB_PATH}" + (f" ({n} column(s) migrated)" if n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
