import os
import sqlite3
from contextlib import contextmanager
from datetime import date

DB_PATH = os.getenv("DB_PATH", "data/signals.db")


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
) -> None:
    """
    Record one ECMWF forecast vs ERA5 actuals comparison.
    Called by bias.py after a contract resolves and ERA5 data is available.
    """
    lat_key, lon_key = _loc_keys(lat, lon)
    from datetime import datetime, timezone
    month_day = target_date[5:]   # MM-DD
    error_c   = actual_tmax_c - forecast_mu_c
    now       = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO forecast_errors
            (lat_key, lon_key, city, calendar_month_day, target_date,
             days_ahead, forecast_mu_c, actual_tmax_c, error_c, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _get_conn() as conn:
        conn.execute(sql, (
            lat_key, lon_key, city, month_day, target_date,
            days_ahead, forecast_mu_c, actual_tmax_c, error_c, now,
        ))


def get_bias_correction(
    lat: float, lon: float,
    calendar_month_day: str,
    window_days: int = 30,
    min_observations: int = 10,
) -> tuple[float, int]:
    """
    Compute the mean ECMWF forecast error for a location over a rolling
    calendar window centred on `calendar_month_day` (MM-DD format).

    We use a ±window_days calendar window (ignoring year) to capture
    seasonal bias rather than one calendar date in isolation.

    Returns (bias_correction_c, n_observations).
        bias_correction_c > 0 means ECMWF historically runs too cold → add to forecast
        bias_correction_c < 0 means ECMWF historically runs too warm → subtract
    Returns (0.0, 0) if fewer than min_observations are available.
    """
    lat_key, lon_key = _loc_keys(lat, lon)

    # Build a list of MM-DD strings within ±window_days of the target
    from datetime import date as _date, timedelta
    import calendar as _cal
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
          AND calendar_month_day IN ({placeholders})
        ORDER BY recorded_at DESC
        LIMIT 60
    """
    with _get_conn() as conn:
        rows = conn.execute(
            sql, [lat_key, lon_key] + list(window_dates)
        ).fetchall()

    errors = [r[0] for r in rows if r[0] is not None]
    n = len(errors)
    if n < min_observations:
        return (0.0, n)
    return (sum(errors) / n, n)


import calendar  # needed by get_bias_correction
