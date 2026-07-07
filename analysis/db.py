"""Read-only access to the collected data for analysis.

By default opens db/snapshot.db (a local copy pulled from EC2). Override with
the ANALYSIS_DB env var or the `path` arg. Opened read-only so analysis can
never mutate the collection DB.

    from analysis.db import connect, q
    con = connect()
    df = q(con, "SELECT * FROM events LIMIT 5")
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "db" / "snapshot.db"


def connect(path=None, readonly=True):
    p = str(path or os.getenv("ANALYSIS_DB", DEFAULT_DB))
    if readonly:
        return sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    return sqlite3.connect(p)


def q(con, sql, params=()):
    """Run a query and return a pandas DataFrame."""
    return pd.read_sql_query(sql, con, params=params)


def tables(con):
    """List tables with their row counts."""
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return pd.DataFrame(
        [(n, con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]) for n in names],
        columns=["table", "rows"])
