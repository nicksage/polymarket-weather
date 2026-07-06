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
    discovered_at_local TEXT,   -- discovered_at rendered in the city's local time
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
    recorded_at   TEXT NOT NULL,
    recorded_at_local TEXT   -- recorded_at rendered in the event city's local time
);

CREATE TABLE IF NOT EXISTS resolutions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id              TEXT UNIQUE NOT NULL,
    city                  TEXT,
    date                  TEXT,
    winning_contract_id   TEXT,
    winning_range_low     REAL,
    winning_range_high    REAL,
    resolved_at           TEXT,
    resolved_at_local     TEXT   -- resolved_at rendered in the city's local time
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
-- These coarse ensemble tables are fed by NWS only (api.weather.gov,
-- US cities). TWC data lives in the full-fidelity twc_* tables below.
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
    fetched_at   TEXT NOT NULL,
    fetched_at_local TEXT         -- fetched_at rendered in the city's local time
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
    fetched_at   TEXT NOT NULL,
    fetched_at_local TEXT         -- fetched_at rendered in the city's local time
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

-- ============================================================
-- The Weather Company (TWC) full-fidelity capture.
-- Keyed by ICAO airport code = the exact station Polymarket resolves
-- against (parsed from each market's Wunderground resolutionSource,
-- e.g. .../jinan/ZSJN -> ZSJN). All values are metric (units="m":
-- temps degC, wind km/h, pressure hPa, precip mm, visibility km).
-- Every forecast period and every field is stored on every poll --
-- not just the current/most-recent value.
-- ============================================================

-- Current observations by ICAO. One row per (icao) per poll.
CREATE TABLE IF NOT EXISTS twc_current (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    city                       TEXT,
    icao                       TEXT NOT NULL,
    units                      TEXT,
    valid_time_local           TEXT,
    valid_time_utc             INTEGER,
    expiration_time_utc        INTEGER,
    day_of_week                TEXT,
    day_or_night               TEXT,
    temperature                REAL,
    temperature_feels_like     REAL,
    temperature_dew_point      REAL,
    temperature_heat_index     REAL,
    temperature_wind_chill     REAL,
    temperature_wet_bulb_globe REAL,
    temperature_max_24hour     REAL,
    temperature_min_24hour     REAL,
    temperature_max_since_7am  REAL,
    temperature_change_24hour  REAL,
    relative_humidity          REAL,
    precip_1hour               REAL,
    precip_6hour               REAL,
    precip_24hour              REAL,
    snow_1hour                 REAL,
    snow_6hour                 REAL,
    snow_24hour                REAL,
    wind_speed                 REAL,
    wind_direction             REAL,
    wind_direction_cardinal    TEXT,
    wind_gust                  REAL,
    pressure_altimeter         REAL,
    pressure_mean_sea_level    REAL,
    pressure_change            REAL,
    pressure_tendency_code     INTEGER,
    pressure_tendency_trend    TEXT,
    cloud_cover                REAL,
    cloud_cover_phrase         TEXT,
    cloud_ceiling              REAL,
    visibility                 REAL,
    uv_index                   REAL,
    uv_description             TEXT,
    icon_code                  INTEGER,
    icon_code_extend           INTEGER,
    wx_phrase_long             TEXT,
    wx_phrase_medium           TEXT,
    wx_phrase_short            TEXT,
    obs_qualifier_code         TEXT,
    obs_qualifier_severity     INTEGER,
    sunrise_time_local         TEXT,
    sunrise_time_utc           INTEGER,
    sunset_time_local          TEXT,
    sunset_time_utc            INTEGER,
    fetched_at                 TEXT NOT NULL,
    fetched_at_local           TEXT   -- fetched_at rendered in the city's local time
);

-- Enterprise hourly forecast by ICAO. One row per forecast hour per poll.
CREATE TABLE IF NOT EXISTS twc_hourly (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    city                       TEXT,
    icao                       TEXT NOT NULL,
    units                      TEXT,
    duration                   TEXT,   -- e.g. "2day"
    valid_time_local           TEXT,
    valid_time_utc             INTEGER,
    expiration_time_utc        INTEGER,
    day_of_week                TEXT,
    day_or_night               TEXT,
    temperature                REAL,
    temperature_dew_point      REAL,
    temperature_feels_like     REAL,
    temperature_heat_index     REAL,
    temperature_wind_chill     REAL,
    temperature_wet_bulb_globe REAL,
    relative_humidity          REAL,
    precip_chance              REAL,
    precip_type                TEXT,
    qpf                        REAL,
    qpf_rain                   REAL,
    qpf_snow                   REAL,
    qpf_ice                    REAL,
    cond_prob_rain             REAL,
    cond_prob_snow             REAL,
    cond_prob_sleet            REAL,
    cond_prob_freezing_rain    REAL,
    cond_prob_thunder          REAL,
    wind_speed                 REAL,
    wind_direction             REAL,
    wind_direction_cardinal    TEXT,
    wind_gust                  REAL,
    pressure_altimeter         REAL,
    pressure_mean_sea_level    REAL,
    cloud_cover                REAL,
    ceiling                    REAL,
    scattered_cloud_base_height REAL,
    visibility                 REAL,
    uv_index                   REAL,
    uv_description             TEXT,
    icon_code                  INTEGER,
    icon_code_extend           INTEGER,
    wx_phrase_long             TEXT,
    wx_phrase_short            TEXT,
    wx_string                  TEXT,
    wx_severity                INTEGER,
    qualifier_set              TEXT,   -- JSON array as returned
    fetched_at                 TEXT NOT NULL,
    fetched_at_local           TEXT   -- fetched_at rendered in the city's local time
);

-- 15-minute forecast (next ~7 hours). One row per 15-min period per poll.
CREATE TABLE IF NOT EXISTS twc_fifteenminute (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    city                    TEXT,
    icao                    TEXT NOT NULL,
    units                   TEXT,
    valid_time_local        TEXT,
    day_of_week             TEXT,
    temperature             REAL,
    temperature_feels_like  REAL,
    relative_humidity       REAL,
    precip_chance           REAL,
    precip_rate             REAL,
    precip_type             TEXT,
    snow_rate               REAL,
    wind_speed              REAL,
    wind_direction          REAL,
    wind_direction_cardinal TEXT,
    icon_code               INTEGER,
    icon_code_extend        INTEGER,
    wx_phrase_long          TEXT,
    wx_phrase_short         TEXT,
    wx_severity             INTEGER,
    fetched_at              TEXT NOT NULL,
    fetched_at_local        TEXT   -- fetched_at rendered in the city's local time
);

-- Probabilistic hourly forecast by ICAO. One row per (icao, product, parameter)
-- per poll. `data` holds that parameter's full product payload as JSON, so
-- every returned number is preserved losslessly:
--   product='pdf'           -> {"binEdges":[[...]], "binValues":[[...]]}
--   product='percentiles'   -> {"numPoints":N, "percentilePoints":[...], "percentileValues":[[...]]}
--   product='probabilities' -> [{"lb":..,"ub":..,"probability":[...]}, ...]  (one per requested band)
--   product='prototypes'    -> {"forecast":[[...]]}                          (ensemble member traces)
-- fcst_valid is the JSON array of UNIX hour timestamps the arrays index into.
CREATE TABLE IF NOT EXISTS twc_probabilistic (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    city          TEXT,
    icao          TEXT NOT NULL,
    units         TEXT,
    hours         INTEGER,   -- forecast horizon requested
    product       TEXT NOT NULL,   -- pdf | percentiles | probabilities | prototypes
    parameter     TEXT NOT NULL,   -- temperature, windSpeed, relativeHumidity, ...
    init_time     INTEGER,
    proc_time     INTEGER,
    latitude      REAL,
    longitude     REAL,
    elevation     REAL,
    landuse       REAL,
    spatial_app   INTEGER,
    version       TEXT,
    expires       TEXT,
    request_id    INTEGER,
    fcst_valid    TEXT,      -- JSON array of UNIX times (hourly steps)
    data          TEXT,      -- JSON payload for this (product, parameter)
    fetched_at    TEXT NOT NULL,
    fetched_at_local TEXT     -- fetched_at rendered in the city's local time
);

CREATE INDEX IF NOT EXISTS idx_twc_cur_icao
    ON twc_current(icao, fetched_at);
CREATE INDEX IF NOT EXISTS idx_twc_hr_icao
    ON twc_hourly(icao, valid_time_utc, fetched_at);
CREATE INDEX IF NOT EXISTS idx_twc_15_icao
    ON twc_fifteenminute(icao, valid_time_local, fetched_at);
CREATE INDEX IF NOT EXISTS idx_twc_prob
    ON twc_probabilistic(icao, product, parameter, fetched_at);
