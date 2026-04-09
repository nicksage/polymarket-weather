import os
import sqlite3
from contextlib import contextmanager
from datetime import date

DB_PATH = os.getenv("DB_PATH", "data/signals.db")


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                contract_id      TEXT    NOT NULL,
                question         TEXT,
                market_p         REAL,
                model_p          REAL,
                ev               REAL,
                recommended_side TEXT,
                kelly_size       REAL,
                executed         INTEGER DEFAULT 0,
                outcome          TEXT,
                pnl              REAL
            );

            CREATE TABLE IF NOT EXISTS positions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id  TEXT,
                side         TEXT,
                size_usdc    REAL,
                entry_price  REAL,
                entry_time   TEXT,
                status       TEXT    DEFAULT 'open',
                exit_price   REAL,
                exit_time    TEXT,
                pnl          REAL
            );
        """)


def insert_signal(
    timestamp: str,
    contract_id: str,
    question: str = None,
    market_p: float = None,
    model_p: float = None,
    ev: float = None,
    recommended_side: str = None,
    kelly_size: float = None,
) -> int:
    sql = """
        INSERT INTO signals
            (timestamp, contract_id, question, market_p, model_p, ev,
             recommended_side, kelly_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(
            sql,
            (timestamp, contract_id, question, market_p, model_p, ev,
             recommended_side, kelly_size),
        )
        return cur.lastrowid


def insert_position(
    contract_id: str,
    side: str,
    size_usdc: float,
    entry_price: float,
    entry_time: str,
) -> int:
    sql = """
        INSERT INTO positions
            (contract_id, side, size_usdc, entry_price, entry_time)
        VALUES (?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (contract_id, side, size_usdc, entry_price, entry_time))
        return cur.lastrowid


def get_open_positions() -> list[sqlite3.Row]:
    sql = "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time ASC"
    with _get_conn() as conn:
        return conn.execute(sql).fetchall()


def update_position_outcome(
    position_id: int,
    exit_price: float,
    exit_time: str,
    pnl: float,
    status: str = "closed",
) -> None:
    sql = """
        UPDATE positions
        SET exit_price = ?, exit_time = ?, pnl = ?, status = ?
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (exit_price, exit_time, pnl, status, position_id))


def get_daily_pnl(for_date: str = None) -> float:
    """
    Returns total realized PnL from closed positions for a given date.
    Date format: 'YYYY-MM-DD'. Defaults to today.
    """
    if for_date is None:
        for_date = date.today().isoformat()

    sql = """
        SELECT COALESCE(SUM(pnl), 0.0)
        FROM positions
        WHERE status = 'closed'
          AND DATE(exit_time) = ?
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (for_date,)).fetchone()
        return row[0]
