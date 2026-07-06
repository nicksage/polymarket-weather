"""Create main.db, enable WAL mode, apply schema. Idempotent."""
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DB_PATH", str(REPO_ROOT / "db" / "main.db"))
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print(f"Initialized DB at {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
