# Polymarket Weather Trading Bot

Automated trading bot for Polymarket binary weather prediction markets. Compares ensemble weather model forecasts against market-implied probabilities to identify and execute trades on daily high temperature contracts.

## Overview

The bot discovers active temperature markets on Polymarket, builds a probability distribution for each city's daily high using ECMWF and GFS ensemble forecasts, and trades bins where the model identifies opportunity. It supports two pluggable trading strategies, active position management with automated exits, and a live intraday adjustment layer powered by Visual Crossing observations.

### Key capabilities

- **Dual-model ensemble forecasting** -- blends 51-member ECMWF and 31-member GFS ensembles with ERA5 climatological priors and per-model bias correction
- **Live intraday adjustment** -- Visual Crossing observations update the forecast distribution every 20 minutes with dynamic sigma shrinkage and observed-max floor truncation
- **Pluggable strategy framework** -- `top_bin_value` strategy buys the model's highest-conviction bins; new strategies can be added via the framework
- **Active exit engine** -- classifies open positions as INVALIDATED / DYING / TOP_BIN_CONFIRMED / WEAKENED / THRIVING and exits accordingly
- **Signal priority ranking** -- when capital is limited, ranks signals by conviction, capital efficiency, confidence, and time efficiency
- **Paper and live trading** -- full paper-trade simulation with identical logic to live CLOB execution
- **Streamlit dashboard** -- real-time monitoring of events, positions, P&L, and forecast accuracy

## Repository Structure

```
polymarket-weather/
    CLAUDE.md                       # Codebase instructions and project context
    requirements.txt                # Python dependencies
    .env                            # Environment variables and configuration (not committed)

    bot/
        main.py                     # Entry point, scheduler, loop orchestration
        config.py                   # Loads all .env variables with defaults
        db.py                       # SQLite schema, migrations, and query helpers
        dashboard.py                # Streamlit monitoring dashboard

        # -- Weather & Forecasting --
        weather.py                  # ECMWF/GFS ensemble blending, ERA5 climatology, NWS/NBM
        live_adjustment.py          # Intraday mu residual + sigma shrinkage from VC observations
        solar.py                    # Sunrise/sunset via astral (for sigma time-of-day scaling)

        # -- Trading Logic --
        edge.py                     # Signal generation, probability computation, bracket promotion
        execution.py                # CLOB order placement (live) and paper-trade logging
        position_eval.py            # Exit engine (INVALIDATED, DYING, WEAKENED, THRIVING)
        sizing.py                   # Kelly criterion sizing, confidence multiplier, signal ranking
        risk.py                     # 13-point risk check gate (exposure, expiry, drawdown, etc.)
        monitor.py                  # Position monitoring, P&L updates, market resolution detection

        # -- External API Clients --
        polymarket.py               # Gamma API market discovery, contract parsing
        visualcrossing.py           # Visual Crossing Timeline API (observations + forecast)
        openmeteo_previous_runs.py  # Open-Meteo Previous Runs (historical forecast retrieval)
        geocoder.py                 # City name to lat/lon via Open-Meteo Geocoding

        # -- Scheduled Loops --
        loops.py                    # Forecast pull, live observation, VC diagnostic, retention loops

        # -- Submodules --
        strategies/                 # Pluggable trading strategies
            __init__.py             # Strategy registry and factory
            base.py                 # Abstract Strategy base class
            top_bin_value.py        # Conviction-based top-bin strategy

        bias_correction/            # Forecast bias correction pipeline
            bias.py                 # Bias computation utilities
            bias_updater.py         # Daily VC + Open-Meteo backfill and error rebuild

        backtest/                   # Backtesting infrastructure (separate DB)
            backtest.py             # Historical replay engine
            backtest_db.py          # Backtest-only database schema and helpers

        scripts/                    # One-off and diagnostic scripts
            adjustment_backtest.py  # Floor-focused backtest with variant comparison
            backfill_backtest_data.py # Archive API hourly observation backfill
            build_empirical_sigma.py  # Derive empirical sigma from forecast errors
            vc_diagnostic_report.py   # VC disagreement analysis report
            vc_diagnostic_smoke.py    # VC diagnostic smoke test
            vc_smoke.py               # VC API connectivity test
            schema_check.py           # Database schema verification
            probe_backtest_apis.py    # API coverage probe for backtest data sources
            bias_diagnostics.py       # Bias correction diagnostic report
            backfill_bias_data.py     # Historical bias data backfill

        data/                       # SQLite databases (not committed)
            signals.db              # Live trading database
            backtest.db             # Backtest-only database

        logs/                       # Log files and reports (not committed)

        _archive/                   # Inactive/backup files
```

## Setup

### Prerequisites

- Python 3.11+
- A Polymarket account with a funded wallet (for live trading)
- API keys: Visual Crossing (required), Tomorrow.io (optional)

### Quick start (recommended)

The setup script installs all dependencies, creates directories, initializes databases, and verifies everything works:

```bash
git clone https://github.com/your-username/polymarket-weather.git
cd polymarket-weather
python setup.py
```

The script will:
1. Verify your Python version (3.11+ required)
2. Install all packages from `requirements.txt`
3. Create `.env` from `.env.example` if it doesn't exist
4. Create `bot/data/` and `bot/logs/` directories
5. Initialize both SQLite databases (`signals.db` and `backtest.db`)
6. Verify all imports succeed
7. Print next steps

After setup completes, edit `.env` and fill in your API keys before starting the bot.

### Manual installation

If you prefer to set up manually:

```bash
git clone https://github.com/your-username/polymarket-weather.git
cd polymarket-weather
pip install -r requirements.txt
cp .env.example .env
```

### Configuration

Edit `.env` and fill in your values:

```bash
# Required
VISUAL_CROSSING_API_KEY=your_key_here

# Required for live trading (not needed for paper mode)
POLYMARKET_PRIVATE_KEY=your_private_key_here
```

Key variables:

| Variable | Description | Default |
|---|---|---|
| `POLYMARKET_PRIVATE_KEY` | Ethereum wallet private key for CLOB orders | -- |
| `VISUAL_CROSSING_API_KEY` | Visual Crossing API key | -- |
| `PAPER_TRADE` | `true` for paper trading, `false` for live | `true` |
| `BANKROLL_USDC` | Starting bankroll in USDC | `1000` |
| `ACTIVE_STRATEGY` | Trading strategy to use | `top_bin_value` |
| `TBV_MIN_MODEL_PROB` | Minimum model probability for top_bin_value YES signals | `0.10` |
| `MAX_BIN_BUYS` | Max total positions (YES + NO) per event | `4` |
| `TBV_TOP_N_BINS` | Max YES positions per event (top_bin_value only) | `3` |

See `.env` for the full list of configuration options with descriptions.

## Usage

### Start the bot

```bash
cd bot
python main.py
```

The bot runs continuously with the following schedule (UTC):

| Time | Loop | Purpose |
|---|---|---|
| Every 4h at :00 | Discovery | Find active Polymarket temperature markets |
| Every 2h at :05 | Forecast pull | Fetch fresh ECMWF + GFS ensemble data |
| Every 2h at :07 | VC diagnostics | Fetch Visual Crossing forecast for D+1 events |
| Every 1h at :10 | Trading | Generate signals, rank, execute, evaluate exits |
| Every 20min | Live observations | Fetch VC intraday data, update current state |
| Every 1h at :30 | Monitor | Update P&L, detect resolved markets, cancel stale orders |
| Daily at 04:30 | Retention | Purge data older than 90 days |
| Daily at 05:00 | Bias update | Refresh forecast error corrections |

### Start the dashboard

```bash
cd bot
streamlit run dashboard.py
```

Opens a browser with four tabs: Contract Data, Paper Trade Data, Live Trade Data, and Forecast Accuracy.

### Run diagnostic scripts

```bash
# Check database schema
python scripts/schema_check.py

# Test Visual Crossing connectivity
python scripts/vc_smoke.py

# Run VC disagreement analysis (after 1-2 weeks of data)
python scripts/vc_diagnostic_report.py --days 14

# Run backtest
python scripts/adjustment_backtest.py --days 90
```

## Trading Strategy

### top_bin_value

Buys the model's highest-probability bins when they meet minimum model and market probability thresholds. Focuses on bins the model genuinely thinks will win. New strategies can be added to the `bot/strategies/` directory by implementing the `Strategy` base class.

## Database

The bot uses two SQLite databases:

**`signals.db`** (live trading) -- contains event scans, positions, decision snapshots, forecast runs, live observations, VC diagnostics, bias corrections, and historical data caches.

**`backtest.db`** (backtesting only) -- contains historical hourly observations, empirical sigma values, and backtest replay results. Kept separate so backtest rebuilds cannot affect live trading state.

## Dependencies

- **httpx** -- HTTP client for all API calls
- **APScheduler** -- Cron-based job scheduling
- **scipy / numpy** -- Normal distribution CDF for bin probability computation
- **astral** -- Sunrise/sunset calculation for sigma time-of-day scaling
- **py-clob-client** -- Polymarket CLOB order placement
- **streamlit / plotly / pandas** -- Dashboard
- **python-dotenv** -- Environment variable loading
- **tenacity** -- Retry logic for transient API failures

## External APIs

| API | Purpose | Auth |
|---|---|---|
| Polymarket Gamma API | Market discovery, contract metadata, resolution detection | None |
| Polymarket CLOB API | Order placement and cancellation (live trading only) | Wallet private key |
| Open-Meteo Ensemble API | ECMWF (51-member) and GFS (31-member) ensemble forecasts | None (free) |
| Open-Meteo Archive API | ERA5 reanalysis for climatological priors | None (free) |
| Open-Meteo Previous Runs API | Historical forecast retrieval for bias correction | None (free) |
| Visual Crossing Timeline API | Station observations, intraday hourly data, forecast diagnostics | API key |
| NWS api.weather.gov | US-only hourly forecasts and NBM quantiles | None (free) |
| Tomorrow.io | Point forecast sanity check | API key (optional) |
