-- ============================================================
-- Consolidated main.db schema for polymarket-weather platform
-- Phase 1: polymarket price data only.
-- Phase 2 will add weather + trading tables.
-- ============================================================

-- Polymarket price data
CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    city           TEXT,
    date           TEXT,
    event_title    TEXT,
    n_bins         INTEGER,
    discovered_at  TEXT,
    resolved       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT NOT NULL,
    contract_id   TEXT NOT NULL UNIQUE,
    question      TEXT,
    range_low     REAL,
    range_high    REAL,
    unit          TEXT,
    yes_token_id  TEXT,
    no_token_id   TEXT
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT NOT NULL,
    contract_id   TEXT NOT NULL,
    yes_price     REAL,
    volume_usd    REAL,
    liquidity_usd REAL,
    recorded_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resolutions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id              TEXT UNIQUE NOT NULL,
    city                  TEXT,
    date                  TEXT,
    winning_contract_id   TEXT,
    winning_range_low     REAL,
    winning_range_high    REAL,
    resolved_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_price_snap_event
    ON price_snapshots(event_id, contract_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_price_snap_time
    ON price_snapshots(recorded_at);
CREATE INDEX IF NOT EXISTS idx_bins_event
    ON bins(event_id);

-- Placeholder for weather sources (schemas to be defined later)
CREATE TABLE IF NOT EXISTS weather_sources (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE NOT NULL,
    notes   TEXT
);
