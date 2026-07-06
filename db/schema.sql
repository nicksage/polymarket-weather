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

-- ============================================================
-- Weather data (Phase 2)
-- Written by collectors/weather_api.py every ~30 min.
-- source is one of: nws | tomorrowio | visualcrossing | twc
-- city matches events.city verbatim (e.g. "NYC", "Sao Paulo").
-- Temps stored in BOTH Celsius and Fahrenheit (canonical = Celsius).
-- ============================================================

-- Daily high/low forecast, one row per (city, source, target_date) per poll.
CREATE TABLE IF NOT EXISTS weather_forecasts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    city         TEXT NOT NULL,
    source       TEXT NOT NULL,
    target_date  TEXT NOT NULL,   -- YYYY-MM-DD, forecast day in the city's local time
    lead_days    INTEGER,         -- target_date minus fetch date, in whole days (0 = today)
    high_c       REAL,
    low_c        REAL,
    high_f       REAL,
    low_f        REAL,
    precip_prob  REAL,            -- daily max probability of precipitation, 0-100
    humidity     REAL,            -- daily avg relative humidity, 0-100 (if available)
    wind_kph     REAL,            -- daily avg/max wind (if available)
    fetched_at   TEXT NOT NULL
);

-- Current observed conditions, one row per (city, source) per poll.
CREATE TABLE IF NOT EXISTS weather_observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    city         TEXT NOT NULL,
    source       TEXT NOT NULL,
    temp_c       REAL,
    temp_f       REAL,
    humidity     REAL,            -- relative humidity, 0-100
    wind_kph     REAL,
    conditions   TEXT,            -- text description, e.g. "Partly Cloudy"
    observed_at  TEXT,            -- API-reported observation time (may differ from fetched_at)
    fetched_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wx_fc_city_date
    ON weather_forecasts(city, target_date, source, fetched_at);
CREATE INDEX IF NOT EXISTS idx_wx_fc_fetched
    ON weather_forecasts(fetched_at);
CREATE INDEX IF NOT EXISTS idx_wx_obs_city
    ON weather_observations(city, source, fetched_at);

-- Reference list of weather sources.
CREATE TABLE IF NOT EXISTS weather_sources (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE NOT NULL,
    notes   TEXT
);
