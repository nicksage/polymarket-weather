import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

# Single source of truth — always resolves to bot/data/signals.db (absolute).
from config import DB_PATH


@contextmanager
def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
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
            -- Legacy signals table (retained for backward compatibility)
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

            -- Positions table (used by execution.py for live/paper trades)
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

            -- One row per highest-temperature event (city + date) per scan.
            -- Stores the forecast distribution parameters derived from the
            -- ensemble blend so the dashboard can display them without re-fetching.
            CREATE TABLE IF NOT EXISTS temp_events (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_timestamp      TEXT    NOT NULL,
                event_id            TEXT    NOT NULL,
                event_title         TEXT,
                city                TEXT,
                date                TEXT,
                lat                 REAL,
                lon                 REAL,
                forecast_mu_c       REAL,
                forecast_sigma_c    REAL,
                clim_mu_c           REAL,
                clim_sigma_c        REAL,
                forecast_mu_display REAL,    -- in display unit (C or F)
                display_unit        TEXT,    -- 'celsius' or 'fahrenheit'
                days_ahead          INTEGER,
                market_overround    REAL,
                model_probs_sum     REAL,
                normalization_warning INTEGER DEFAULT 0,
                n_outcomes          INTEGER,
                n_sources           INTEGER
            );

            -- One row per temperature-range outcome within an event.
            -- Linked to temp_events via event_row_id.
            CREATE TABLE IF NOT EXISTS temp_outcomes (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                event_row_id     INTEGER REFERENCES temp_events(id),
                scan_timestamp   TEXT    NOT NULL,
                contract_id      TEXT    NOT NULL,
                question         TEXT,
                range_low        REAL,
                range_high       REAL,
                unit             TEXT,
                market_price     REAL,
                model_prob       REAL,
                raw_model_prob   REAL,
                ev               REAL,
                edge             REAL,
                recommended_side TEXT,
                kelly_size       REAL,
                is_signal        INTEGER DEFAULT 0,
                liquidity_usd    REAL,
                volume_usd       REAL,
                yes_token_id     TEXT,
                no_token_id      TEXT,
                executed         INTEGER DEFAULT 0,
                outcome          TEXT,
                pnl              REAL
            );

            CREATE INDEX IF NOT EXISTS idx_temp_events_scan ON temp_events(scan_timestamp);
            CREATE INDEX IF NOT EXISTS idx_temp_outcomes_event ON temp_outcomes(event_row_id);
            CREATE INDEX IF NOT EXISTS idx_temp_outcomes_signal ON temp_outcomes(is_signal, scan_timestamp);
        """)

        # Add columns that may not exist in DBs created before this patch.
        # SQLite does not support IF NOT EXISTS on ALTER TABLE, so we use
        # try/except for each column independently.
        for col_def in [
            "ALTER TABLE temp_outcomes ADD COLUMN yes_price REAL",
            "ALTER TABLE temp_outcomes ADD COLUMN no_price  REAL",
            # positions table — expanded for full trade lifecycle tracking
            "ALTER TABLE positions ADD COLUMN order_id        TEXT",
            "ALTER TABLE positions ADD COLUMN is_paper        INTEGER DEFAULT 1",
            "ALTER TABLE positions ADD COLUMN question        TEXT",
            "ALTER TABLE positions ADD COLUMN city            TEXT",
            "ALTER TABLE positions ADD COLUMN date            TEXT",
            "ALTER TABLE positions ADD COLUMN event_id        TEXT",
            "ALTER TABLE positions ADD COLUMN model_prob      REAL",
            "ALTER TABLE positions ADD COLUMN market_prob     REAL",
            "ALTER TABLE positions ADD COLUMN ev              REAL",
            "ALTER TABLE positions ADD COLUMN edge            REAL",
            "ALTER TABLE positions ADD COLUMN fill_status     TEXT DEFAULT 'filled'",
            "ALTER TABLE positions ADD COLUMN shares          REAL",
            "ALTER TABLE positions ADD COLUMN unrealized_pnl  REAL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN current_price   REAL",
            "ALTER TABLE positions ADD COLUMN scan_timestamp   TEXT",
            "ALTER TABLE positions ADD COLUMN cancelled_reason TEXT",
            "ALTER TABLE positions ADD COLUMN gamma_market_id  TEXT",
            "ALTER TABLE positions ADD COLUMN range_low        REAL",
            "ALTER TABLE positions ADD COLUMN range_high       REAL",
            "ALTER TABLE positions ADD COLUMN unit             TEXT",
            "ALTER TABLE positions ADD COLUMN local_time       TEXT",
            "ALTER TABLE positions ADD COLUMN yes_token_id     TEXT",
            "ALTER TABLE positions ADD COLUMN no_token_id      TEXT",
            "ALTER TABLE positions ADD COLUMN lat              REAL",
            "ALTER TABLE positions ADD COLUMN lon              REAL",
            "ALTER TABLE positions ADD COLUMN forecast_sigma_c REAL",
            # forecast_errors table — model column added for per-(city, month, model) bias.
            # Existing rows default to 'ecmwf' since that's the only model previously tracked.
            "ALTER TABLE forecast_errors ADD COLUMN model TEXT DEFAULT 'ecmwf'",
            "ALTER TABLE forecast_errors ADD COLUMN lead_days_bucket INTEGER",
        ]:
            try:
                conn.execute(col_def)
            except Exception:
                pass   # column already exists

        conn.executescript("""

            -- Persistent ERA5 reanalysis cache.
            -- One row per (location, date). Data older than ~3 months is immutable
            -- so rows are written once and never updated.  The bot checks this table
            -- before calling the Open-Meteo Archive API, eliminating redundant fetches
            -- on every run after the first.
            CREATE TABLE IF NOT EXISTS era5_daily (
                lat_key     REAL NOT NULL,   -- round(lat, 2)
                lon_key     REAL NOT NULL,   -- round(lon, 2)
                date        TEXT NOT NULL,   -- YYYY-MM-DD (UTC)
                tmax_c      REAL NOT NULL,   -- daily maximum 2m temperature (°C)
                fetched_at  TEXT NOT NULL,   -- ISO timestamp of DB insert
                PRIMARY KEY (lat_key, lon_key, date)
            );
            CREATE INDEX IF NOT EXISTS idx_era5_loc_doy
                ON era5_daily(lat_key, lon_key, date);

            -- Forecast error log for ECMWF bias correction.
            -- After each contract resolves, we compare what ECMWF predicted
            -- (forecast_mu_c) to what ERA5 says actually happened (actual_tmax_c).
            -- The mean error over recent observations is subtracted from future
            -- forecasts for the same location + calendar period.
            CREATE TABLE IF NOT EXISTS forecast_errors (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                lat_key             REAL    NOT NULL,
                lon_key             REAL    NOT NULL,
                city                TEXT,
                calendar_month_day  TEXT    NOT NULL,  -- MM-DD
                target_date         TEXT    NOT NULL,  -- YYYY-MM-DD
                days_ahead          INTEGER,
                forecast_mu_c       REAL,
                actual_tmax_c       REAL,
                error_c             REAL,              -- actual - forecast (+ve = forecast was too cold)
                recorded_at         TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_bias_loc_cmd
                ON forecast_errors(lat_key, lon_key, calendar_month_day);

            -- Cache of Visual Crossing daily-max observations, keyed by
            -- coordinate + date.  Populated by the bias backfill script and
            -- the monitor loop.  Source is always station-observed ("obs").
            CREATE TABLE IF NOT EXISTS historical_observed_daily (
                lat_key     REAL NOT NULL,        -- round(lat, 2)
                lon_key     REAL NOT NULL,        -- round(lon, 2)
                city        TEXT,
                date        TEXT NOT NULL,        -- YYYY-MM-DD local
                tempmax_c   REAL,
                tempmin_c   REAL,
                temp_c      REAL,
                stations    TEXT,                  -- JSON list of station IDs
                fetched_at  TEXT NOT NULL,
                PRIMARY KEY (lat_key, lon_key, date)
            );
            CREATE INDEX IF NOT EXISTS idx_hod_city_date
                ON historical_observed_daily(city, date);

            -- Cache of Open-Meteo Previous Runs historical forecasts.
            -- One row per (location, date, model, lead_days) — lets us store
            -- multiple lead times and multiple models for the same day.
            CREATE TABLE IF NOT EXISTS historical_forecasts_previous_runs (
                lat_key             REAL NOT NULL,
                lon_key             REAL NOT NULL,
                city                TEXT,
                date                TEXT NOT NULL,
                model               TEXT NOT NULL,
                lead_days           INTEGER NOT NULL,
                forecast_tempmax_c  REAL,
                n_hours             INTEGER,
                fetched_at          TEXT NOT NULL,
                PRIMARY KEY (lat_key, lon_key, date, model, lead_days)
            );
            CREATE INDEX IF NOT EXISTS idx_hfpr_city_model_date
                ON historical_forecasts_previous_runs(city, model, date);
        """)


# ---------------------------------------------------------------------------
# Legacy signal helpers (retained for execution.py)
# ---------------------------------------------------------------------------

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
    order_id: str = None,
    is_paper: int = 1,
    question: str = None,
    city: str = None,
    date: str = None,
    event_id: str = None,
    model_prob: float = None,
    market_prob: float = None,
    ev: float = None,
    edge: float = None,
    fill_status: str = "filled",
    shares: float = None,
    scan_timestamp: str = None,
    gamma_market_id: str = None,
    range_low: float = None,
    range_high: float = None,
    unit: str = None,
    yes_token_id: str = None,
    no_token_id: str = None,
    lat: float = None,
    lon: float = None,
    forecast_sigma_c: float = None,
) -> int:
    sql = """
        INSERT INTO positions (
            contract_id, side, size_usdc, entry_price, entry_time,
            order_id, is_paper, question, city, date, event_id,
            model_prob, market_prob, ev, edge,
            fill_status, shares, scan_timestamp,
            unrealized_pnl, current_price, gamma_market_id,
            range_low, range_high, unit,
            yes_token_id, no_token_id, lat, lon,
            forecast_sigma_c
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            contract_id, side, size_usdc, entry_price, entry_time,
            order_id, is_paper, question, city, date, event_id,
            model_prob, market_prob, ev, edge,
            fill_status, shares, scan_timestamp,
            entry_price,  # current_price initialised to entry_price
            gamma_market_id,
            range_low, range_high, unit,
            yes_token_id, no_token_id, lat, lon,
            forecast_sigma_c,
        ))
        return cur.lastrowid


def get_open_positions() -> list[dict]:
    sql = "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time ASC"
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_pending_positions() -> list[dict]:
    """Return live orders placed but not yet confirmed as filled."""
    sql = """
        SELECT * FROM positions
        WHERE status = 'open' AND fill_status = 'pending'
        ORDER BY entry_time ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def get_open_positions_for_event(city: str, date: str) -> list[dict]:
    """Return all open (non-cancelled) positions for a given city+date event."""
    sql = """
        SELECT * FROM positions
        WHERE city = ? AND date = ? AND status = 'open'
        ORDER BY entry_time ASC
    """
    with _get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, (city, date)).fetchall()]


def count_open_bins_for_event(city: str, date: str) -> int:
    """Count distinct open positions for a city+date event (for MAX_BIN_BUYS check)."""
    sql = """
        SELECT COUNT(*) FROM positions
        WHERE city = ? AND date = ? AND status = 'open'
    """
    with _get_conn() as conn:
        row = conn.execute(sql, (city, date)).fetchone()
        return row[0] if row else 0


def update_position_outcome(
    position_id: int,
    exit_price: float,
    exit_time: str,
    pnl: float,
    status: str = "closed",
) -> None:
    sql = """
        UPDATE positions
        SET exit_price = ?, exit_time = ?, pnl = ?, status = ?,
            unrealized_pnl = 0
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (exit_price, exit_time, pnl, status, position_id))


def update_position_fill(
    position_id: int,
    fill_status: str,
    shares: float,
    entry_price: float,
) -> None:
    """Update a pending position once fill is confirmed via CLOB API."""
    sql = """
        UPDATE positions
        SET fill_status = ?, shares = ?, entry_price = ?, current_price = ?
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (fill_status, shares, entry_price, entry_price, position_id))


def cancel_position(
    position_id: int,
    cancelled_reason: str,
    exit_time: str,
) -> None:
    """Mark a position as cancelled (unfilled order, expired, etc.)."""
    sql = """
        UPDATE positions
        SET status = 'closed', fill_status = 'cancelled',
            cancelled_reason = ?, exit_time = ?, pnl = 0
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (cancelled_reason, exit_time, position_id))


def update_position_market_price(
    position_id: int,
    current_price: float,
    unrealized_pnl: float,
    local_time: str = None,
) -> None:
    """Refresh current market price, unrealized P&L, and local time for an open position."""
    sql = """
        UPDATE positions
        SET current_price = ?, unrealized_pnl = ?, local_time = COALESCE(?, local_time)
        WHERE id = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (current_price, unrealized_pnl, local_time, position_id))


def backfill_gamma_market_ids(outcomes: list[dict]) -> int:
    """
    For any open position with NULL gamma_market_id, attempt to fill it by
    matching contract_id against a list of outcome dicts (from the discovery
    pipeline).  Each outcome dict must have 'contract_id' and 'gamma_market_id'.
    Returns the number of rows updated.
    """
    if not outcomes:
        return 0
    # Build a lookup: conditionId → numeric gamma_market_id
    lookup = {
        o["contract_id"]: o["gamma_market_id"]
        for o in outcomes
        if o.get("contract_id") and o.get("gamma_market_id")
    }
    if not lookup:
        return 0
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, contract_id FROM positions WHERE gamma_market_id IS NULL AND status = 'open'"
        ).fetchall()
        updated = 0
        for row in rows:
            gid = lookup.get(row[1])
            if gid:
                conn.execute(
                    "UPDATE positions SET gamma_market_id = ? WHERE id = ?",
                    (str(gid), row[0]),
                )
                updated += 1
    return updated


def backfill_position_coords() -> int:
    """
    For any position with NULL lat/lon, attempt to fill from the CITY_COORDS
    lookup table in polymarket.py.  Returns the number of rows updated.

    This is a one-time migration helper; it is safe to call repeatedly.
    """
    try:
        from polymarket import CITY_COORDS
    except ImportError:
        return 0

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, city FROM positions WHERE lat IS NULL OR lon IS NULL"
        ).fetchall()
        updated = 0
        for row in rows:
            pos_id = row[0]
            city   = (row[1] or "").lower().strip()
            coords = CITY_COORDS.get(city)
            if coords:
                conn.execute(
                    "UPDATE positions SET lat = ?, lon = ? WHERE id = ?",
                    (coords[0], coords[1], pos_id),
                )
                updated += 1
    return updated


def backfill_position_sigma() -> int:
    """
    For any position with NULL forecast_sigma_c, attempt to fill it by joining
    positions.contract_id + positions.scan_timestamp against temp_outcomes and
    temp_events (which store forecast_sigma_c per scan).

    Falls back to matching on contract_id alone (any scan) when the
    scan_timestamp is missing or the exact scan row has been purged.

    Returns the number of rows updated.  Safe to call repeatedly.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, contract_id, scan_timestamp FROM positions "
            "WHERE forecast_sigma_c IS NULL"
        ).fetchall()
        if not rows:
            return 0

        updated = 0
        for pos_id, contract_id, scan_ts in rows:
            sigma = None

            # Preferred: exact scan_timestamp match
            if scan_ts:
                row = conn.execute(
                    """
                    SELECT e.forecast_sigma_c
                    FROM temp_outcomes o
                    JOIN temp_events e ON o.event_row_id = e.id
                    WHERE o.contract_id = ? AND o.scan_timestamp = ?
                    LIMIT 1
                    """,
                    (contract_id, scan_ts),
                ).fetchone()
                if row:
                    sigma = row[0]

            # Fallback: any scan for this contract
            if sigma is None:
                row = conn.execute(
                    """
                    SELECT e.forecast_sigma_c
                    FROM temp_outcomes o
                    JOIN temp_events e ON o.event_row_id = e.id
                    WHERE o.contract_id = ?
                    ORDER BY o.scan_timestamp DESC
                    LIMIT 1
                    """,
                    (contract_id,),
                ).fetchone()
                if row:
                    sigma = row[0]

            if sigma is not None:
                conn.execute(
                    "UPDATE positions SET forecast_sigma_c = ? WHERE id = ?",
                    (sigma, pos_id),
                )
                updated += 1

    return updated


def get_daily_pnl(for_date: str = None) -> float:
    """
    Return today's P&L: realized (closed positions) + unrealized (open filled positions).
    Used by check_daily_drawdown so that a deeply underwater open position
    still counts against the drawdown limit, not just closed trades.
    """
    if for_date is None:
        for_date = date.today().isoformat()
    with _get_conn() as conn:
        realized = conn.execute(
            """
            SELECT COALESCE(SUM(pnl), 0.0) FROM positions
            WHERE status = 'closed' AND fill_status != 'cancelled'
              AND DATE(exit_time) = ?
            """,
            (for_date,),
        ).fetchone()[0]
        unrealized = conn.execute(
            """
            SELECT COALESCE(SUM(unrealized_pnl), 0.0) FROM positions
            WHERE status = 'open' AND fill_status = 'filled'
            """,
        ).fetchone()[0]
    return realized + unrealized


# ---------------------------------------------------------------------------
# Temperature event helpers
# ---------------------------------------------------------------------------

def insert_temp_event(event: dict, scan_timestamp: str) -> int:
    """Insert a top-level temperature event record. Returns the new row ID."""
    sql = """
        INSERT INTO temp_events (
            scan_timestamp, event_id, event_title, city, date, lat, lon,
            forecast_mu_c, forecast_sigma_c, clim_mu_c, clim_sigma_c,
            forecast_mu_display, display_unit, days_ahead,
            market_overround, model_probs_sum, normalization_warning,
            n_outcomes, n_sources
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            scan_timestamp,
            event.get("event_id"),
            event.get("event_title"),
            event.get("city"),
            event.get("date"),
            event.get("lat"),
            event.get("lon"),
            event.get("forecast_mu_c"),
            event.get("forecast_sigma_c"),
            event.get("clim_mu_c"),
            event.get("clim_sigma_c"),
            event.get("forecast_mu_display"),
            event.get("display_unit"),
            event.get("days_ahead"),
            event.get("market_overround"),
            event.get("model_probs_sum"),
            1 if event.get("normalization_warning") else 0,
            event.get("n_outcomes", 0),
            event.get("n_sources", 0),
        ))
        return cur.lastrowid


def insert_temp_outcome(outcome: dict, event_row_id: int, scan_timestamp: str) -> int:
    """Insert a single temperature-range outcome linked to its parent event."""
    sql = """
        INSERT INTO temp_outcomes (
            event_row_id, scan_timestamp, contract_id, question,
            range_low, range_high, unit,
            market_price, yes_price, no_price,
            model_prob, raw_model_prob, ev, edge,
            recommended_side, kelly_size, is_signal,
            liquidity_usd, volume_usd, yes_token_id, no_token_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    yes_price = outcome.get("yes_price", outcome.get("market_price"))
    no_price  = outcome.get("no_price",  1.0 - float(yes_price) if yes_price is not None else None)
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            event_row_id,
            scan_timestamp,
            outcome.get("contract_id"),
            outcome.get("question"),
            outcome.get("range_low"),
            outcome.get("range_high"),
            outcome.get("unit"),
            outcome.get("market_price"),
            yes_price,
            no_price,
            outcome.get("model_prob"),
            outcome.get("raw_model_prob"),
            outcome.get("ev"),
            outcome.get("edge"),
            outcome.get("recommended_side"),
            outcome.get("kelly_size"),
            1 if outcome.get("is_signal") else 0,
            outcome.get("liquidity_usd"),
            outcome.get("volume_usd"),
            outcome.get("yes_token_id"),
            outcome.get("no_token_id"),
        ))
        return cur.lastrowid


def purge_scan_data() -> dict:
    """
    Delete all rows from temp_events and temp_outcomes.

    Use this after a code fix that affects stored probabilities or forecast
    values, so the next bot run starts clean.

    What is deleted:
        temp_events   — event-level scan records (city, date, forecast μ/σ,
                        clim μ/σ, overround, days_ahead, etc.)
        temp_outcomes — per-bin records (model_prob, market_price, yes_price,
                        no_price, edge, EV, is_signal, etc.)

    What is NOT deleted:
        era5_daily      — historical climate cache (correct data, expensive to rebuild)
        forecast_errors — ECMWF bias tracking (accumulates over weeks)
        signals         — legacy signal log
        positions       — trade records

    After calling this, run the bot once to repopulate both tables with
    corrected probability data.

    Returns:
        {"events_deleted": int, "outcomes_deleted": int}
    """
    with _get_conn() as conn:
        outcomes_deleted = conn.execute("DELETE FROM temp_outcomes").rowcount
        events_deleted   = conn.execute("DELETE FROM temp_events").rowcount
    return {"events_deleted": events_deleted, "outcomes_deleted": outcomes_deleted}


def get_latest_scan_timestamp() -> str | None:
    """Return the most recent scan_timestamp stored in temp_events."""
    sql = "SELECT MAX(scan_timestamp) FROM temp_events"
    with _get_conn() as conn:
        row = conn.execute(sql).fetchone()
        return row[0]


def get_latest_temp_events(limit: int = 300) -> list[dict]:
    """Return all event rows from the most recent scan."""
    ts = get_latest_scan_timestamp()
    if not ts:
        return []
    sql = """
        SELECT * FROM temp_events
        WHERE scan_timestamp = ?
        ORDER BY city, date
        LIMIT ?
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, (ts, limit)).fetchall()
        return [dict(r) for r in rows]


def get_latest_temp_outcomes(signals_only: bool = False, limit: int = 2000) -> list[dict]:
    """Return all outcome rows from the most recent scan, optionally filtered to signals."""
    ts = get_latest_scan_timestamp()
    if not ts:
        return []
    sql = """
        SELECT o.*, e.city, e.date, e.forecast_mu_c, e.forecast_sigma_c,
               e.clim_mu_c, e.display_unit, e.event_title, e.days_ahead,
               e.normalization_warning
        FROM temp_outcomes o
        JOIN temp_events e ON o.event_row_id = e.id
        WHERE o.scan_timestamp = ?
        {}
        ORDER BY o.ev DESC
        LIMIT ?
    """.format("AND o.is_signal = 1" if signals_only else "")
    with _get_conn() as conn:
        rows = conn.execute(sql, (ts, limit)).fetchall()
        return [dict(r) for r in rows]


def mark_outcome_executed(contract_id: str, scan_timestamp: str) -> None:
    sql = """
        UPDATE temp_outcomes SET executed = 1
        WHERE contract_id = ? AND scan_timestamp = ?
    """
    with _get_conn() as conn:
        conn.execute(sql, (contract_id, scan_timestamp))


# ---------------------------------------------------------------------------
# ERA5 persistent cache helpers
# ---------------------------------------------------------------------------

def _loc_keys(lat: float, lon: float) -> tuple[float, float]:
    """Canonical location key — rounded to 2 dp to group nearby API calls."""
    return (round(lat, 2), round(lon, 2))


def get_era5_dates_present(lat: float, lon: float, dates: list[str]) -> set[str]:
    """
    Return the subset of `dates` already stored in era5_daily for this location.
    Used to compute the minimal fetch list before calling the Archive API.
    """
    if not dates:
        return set()
    lat_key, lon_key = _loc_keys(lat, lon)
    placeholders = ",".join("?" * len(dates))
    sql = f"""
        SELECT date FROM era5_daily
        WHERE lat_key = ? AND lon_key = ? AND date IN ({placeholders})
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, [lat_key, lon_key] + list(dates)).fetchall()
        return {r[0] for r in rows}


def get_era5_values(lat: float, lon: float, dates: list[str]) -> dict[str, float]:
    """
    Return {date: tmax_c} for all requested dates present in era5_daily.
    """
    if not dates:
        return {}
    lat_key, lon_key = _loc_keys(lat, lon)
    placeholders = ",".join("?" * len(dates))
    sql = f"""
        SELECT date, tmax_c FROM era5_daily
        WHERE lat_key = ? AND lon_key = ? AND date IN ({placeholders})
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, [lat_key, lon_key] + list(dates)).fetchall()
        return {r[0]: r[1] for r in rows}


def insert_era5_rows(lat: float, lon: float, date_tmax: dict[str, float]) -> int:
    """
    Bulk-insert ERA5 daily max temperature rows. Uses INSERT OR IGNORE so
    re-fetching the same dates is safe (no duplicates, no errors).
    Returns the number of rows newly inserted.
    """
    if not date_tmax:
        return 0
    lat_key, lon_key = _loc_keys(lat, lon)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT OR IGNORE INTO era5_daily (lat_key, lon_key, date, tmax_c, fetched_at)
        VALUES (?, ?, ?, ?, ?)
    """
    rows = [(lat_key, lon_key, d, v, now) for d, v in date_tmax.items()]
    with _get_conn() as conn:
        conn.executemany(sql, rows)
        # SQLite doesn't expose per-executemany row count cleanly; approximate
        return len(rows)


def count_era5_rows(lat: float, lon: float) -> int:
    """Return total ERA5 rows stored for a location (diagnostic / dashboard use)."""
    lat_key, lon_key = _loc_keys(lat, lon)
    sql = "SELECT COUNT(*) FROM era5_daily WHERE lat_key = ? AND lon_key = ?"
    with _get_conn() as conn:
        return conn.execute(sql, (lat_key, lon_key)).fetchone()[0]


# ---------------------------------------------------------------------------
# Forecast error / bias correction helpers
# ---------------------------------------------------------------------------

def insert_forecast_error(
    lat: float, lon: float, city: str,
    target_date: str, days_ahead: int,
    forecast_mu_c: float, actual_tmax_c: float,
    model: str = "ecmwf_ifs025",
) -> None:
    """
    Record one forecast-vs-actual comparison.
    Called by bias.py after a contract resolves and ERA5 data is available.

    `model` defaults to 'ecmwf_ifs025' to match the backfilled rows produced
    by scripts/backfill_bias_data.py.  Pass a different model id (e.g.
    'gfs_global') if recording GFS errors separately.
    """
    lat_key, lon_key = _loc_keys(lat, lon)
    from datetime import datetime, timezone
    month_day = target_date[5:]   # MM-DD
    error_c   = actual_tmax_c - forecast_mu_c
    now       = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO forecast_errors
            (lat_key, lon_key, city, calendar_month_day, target_date,
             days_ahead, forecast_mu_c, actual_tmax_c, error_c, recorded_at,
             model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        conn.execute(sql, (
            lat_key, lon_key, city, month_day, target_date,
            days_ahead, forecast_mu_c, actual_tmax_c, error_c, now,
            model,
        ))


def get_bias_correction(
    lat: float, lon: float,
    calendar_month_day: str,
    window_days: int = 30,
    min_observations: int = 10,
    model: str = "ecmwf",
) -> tuple[float, int]:
    """
    Compute the mean forecast error for a location + model over a rolling
    calendar window centred on `calendar_month_day` (MM-DD format).

    We use a ±window_days calendar window (ignoring year) to capture
    seasonal bias rather than one calendar date in isolation.

    Returns (bias_correction_c, n_observations).
        bias_correction_c > 0 means the model historically runs too cold → add to forecast
        bias_correction_c < 0 means the model historically runs too warm → subtract
    Returns (0.0, 0) if fewer than min_observations are available.

    `model` filters forecast_errors by the model column.  Legacy rows inserted
    before the column existed default to 'ecmwf'.
    """
    lat_key, lon_key = _loc_keys(lat, lon)

    # Build a list of MM-DD strings within ±window_days of the target
    from datetime import date as _date, timedelta
    try:
        month = int(calendar_month_day[:2])
        day   = int(calendar_month_day[3:])
        anchor = _date(2000, month, day)   # leap-year-safe reference year
    except ValueError:
        return (0.0, 0)

    window_dates = set()
    for delta in range(-window_days, window_days + 1):
        d = anchor + timedelta(days=delta)
        window_dates.add(f"{d.month:02d}-{d.day:02d}")

    placeholders = ",".join("?" * len(window_dates))
    sql = f"""
        SELECT error_c FROM forecast_errors
        WHERE lat_key = ? AND lon_key = ?
          AND COALESCE(model, 'ecmwf') = ?
          AND calendar_month_day IN ({placeholders})
        ORDER BY recorded_at DESC
        LIMIT 2000
    """
    with _get_conn() as conn:
        rows = conn.execute(
            sql, [lat_key, lon_key, model] + list(window_dates)
        ).fetchall()

    errors = [r[0] for r in rows if r[0] is not None]
    n = len(errors)
    if n < min_observations:
        return (0.0, n)
    return (sum(errors) / n, n)


# ---------------------------------------------------------------------------
# Upsert helpers for the VC + OM backfill caches
# ---------------------------------------------------------------------------

def upsert_historical_observed_daily(
    lat: float, lon: float, city: str | None, rows: list[dict]
) -> int:
    """
    Insert/replace Visual Crossing daily observation rows.

    `rows` items must have: date, tempmax_c, tempmin_c, temp_c, stations (list).
    Returns number of rows written.
    """
    if not rows:
        return 0
    import json
    lat_key, lon_key = _loc_keys(lat, lon)
    now = datetime.utcnow().isoformat() + "Z"
    sql = """
        INSERT INTO historical_observed_daily
            (lat_key, lon_key, city, date, tempmax_c, tempmin_c, temp_c,
             stations, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lat_key, lon_key, date) DO UPDATE SET
            city       = excluded.city,
            tempmax_c  = excluded.tempmax_c,
            tempmin_c  = excluded.tempmin_c,
            temp_c     = excluded.temp_c,
            stations   = excluded.stations,
            fetched_at = excluded.fetched_at
    """
    with _get_conn() as conn:
        for r in rows:
            conn.execute(sql, (
                lat_key, lon_key, city, r["date"],
                r.get("tempmax_c"), r.get("tempmin_c"), r.get("temp_c"),
                json.dumps(r.get("stations") or []),
                now,
            ))
    return len(rows)


def upsert_historical_forecasts_previous_runs(
    lat: float, lon: float, city: str | None, rows: list[dict]
) -> int:
    """
    Insert/replace Open-Meteo Previous Runs forecast rows.

    `rows` items must have: date, model, lead_days, forecast_tempmax_c, n_hours.
    Returns number of rows written.
    """
    if not rows:
        return 0
    lat_key, lon_key = _loc_keys(lat, lon)
    now = datetime.utcnow().isoformat() + "Z"
    sql = """
        INSERT INTO historical_forecasts_previous_runs
            (lat_key, lon_key, city, date, model, lead_days,
             forecast_tempmax_c, n_hours, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(lat_key, lon_key, date, model, lead_days) DO UPDATE SET
            city               = excluded.city,
            forecast_tempmax_c = excluded.forecast_tempmax_c,
            n_hours            = excluded.n_hours,
            fetched_at         = excluded.fetched_at
    """
    with _get_conn() as conn:
        for r in rows:
            conn.execute(sql, (
                lat_key, lon_key, city, r["date"], r["model"], r["lead_days"],
                r.get("forecast_tempmax_c"), r.get("n_hours"),
                now,
            ))
    return len(rows)


def rebuild_forecast_errors_from_cache(
    lat: float, lon: float,
    city: str | None = None,
    lead_days: int = 3,
) -> int:
    """
    Join historical_observed_daily + historical_forecasts_previous_runs for
    one location, compute error = actual - forecast, and write rows into
    forecast_errors (one per (date, model) pair).

    Idempotent: deletes any existing backfilled forecast_errors rows for the
    location before re-inserting, so re-running this after an updated cache
    won't create duplicates.

    A "backfilled" row is distinguished from a live-recorded row by having
    lead_days_bucket matching the lead_days argument AND being written by
    this function.  We key the delete on (lat_key, lon_key, lead_days_bucket).

    Returns the number of forecast_errors rows written.
    """
    lat_key, lon_key = _loc_keys(lat, lon)
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        # Remove any previously-backfilled rows for this (location, lead) so
        # we can re-insert cleanly from the updated cache.
        conn.execute(
            "DELETE FROM forecast_errors "
            "WHERE lat_key = ? AND lon_key = ? AND lead_days_bucket = ?",
            (lat_key, lon_key, lead_days),
        )

        # Join the two caches and insert a forecast_errors row per match.
        cursor = conn.execute("""
            SELECT f.date, f.model, f.lead_days, f.forecast_tempmax_c,
                   o.tempmax_c
            FROM historical_forecasts_previous_runs f
            JOIN historical_observed_daily o
              ON f.lat_key = o.lat_key
             AND f.lon_key = o.lon_key
             AND f.date    = o.date
            WHERE f.lat_key = ? AND f.lon_key = ?
              AND f.lead_days = ?
              AND f.forecast_tempmax_c IS NOT NULL
              AND o.tempmax_c IS NOT NULL
        """, (lat_key, lon_key, lead_days))

        inserted = 0
        for date_str, model, lead, fcst, actual in cursor.fetchall():
            error_c = actual - fcst
            cmd     = date_str[5:]   # MM-DD
            conn.execute("""
                INSERT INTO forecast_errors
                    (lat_key, lon_key, city, calendar_month_day, target_date,
                     days_ahead, forecast_mu_c, actual_tmax_c, error_c,
                     recorded_at, model, lead_days_bucket)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lat_key, lon_key, city, cmd, date_str,
                lead, fcst, actual, error_c, now, model, lead,
            ))
            inserted += 1

    return inserted


import calendar  # kept for any downstream code that imports it from here
