1. High-Level Overview
Goal: Find statistical edge on Polymarket binary weather prediction markets (e.g., "Will it rain in Chicago on April 9?") by comparing the market's implied probability to an ensemble of independent weather model forecasts. When the divergence exceeds a threshold, place a fractional Kelly-sized bet.

Core Logic Loop (every 30 min):

Search Polymarket Gamma API for active weather markets
For each market, fetch weather forecast probability from 4 sources (NWS, Open-Meteo, NOAA, Tomorrow.io)
Blend probabilities into a weighted ensemble (weighted by historical Brier scores)
If |model_p - market_p| > 7% → generate signal
Run risk checks (position size, total exposure, liquidity, drawdown, time-to-expiry)
Execute or paper-log the trade; persist to SQLite
External APIs: Polymarket Gamma API (market discovery), Polymarket CLOB API (order execution), NWS (api.weather.gov), Open-Meteo (ensemble + deterministic), NOAA CDO (ncdc.noaa.gov), Tomorrow.io

2. Project Structure

bot/
├── main.py        — Entry point, scheduler (APScheduler), orchestration loop
├── config.py      — Loads all env vars from .env
├── polymarket.py  — Gamma API market discovery, normalization, metadata parsing
├── weather.py     — NWS / Open-Meteo / NOAA / Tomorrow.io + ensemble aggregation
├── edge.py        — Signal generation: compare model vs market probability
├── risk.py        — Guard checks: position size, exposure, drawdown, liquidity, expiry
├── sizing.py      — Fractional Kelly criterion position sizing
├── execution.py   — CLOB order placement (live or paper)
├── db.py          — SQLite schema + helper functions
├── backtest.py    — Replay model against resolved historical contracts
└── dashboard.py   — (monitoring/reporting)