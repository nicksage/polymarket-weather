# Polymarket Price Collector & Backtester

Standalone price data collector for Polymarket temperature markets. Runs independently from the trading bot. No weather APIs, no trading logic, no API keys needed.

## What It Does

**collector.py** runs 24/7 and:
- Discovers all active temperature events on Polymarket (every 4 hours)
- Subscribes to ALL bin tokens via WebSocket for real-time price updates
- Snapshots prices for every bin every 5 minutes into a local SQLite database
- Detects when events resolve and records the winning bin

**backtest.py** replays the collected price data to simulate any trading strategy:
- Tests different entry price thresholds, number of bins, trailing stops, take profits
- Runs 432 parameter combinations and ranks the top 10 by total P&L
- No look-ahead bias — simulates chronologically through each price path

## Setup

```bash
cd backtest-collector
pip install -r requirements.txt
python collector.py
```

No API keys or `.env` file needed. Polymarket's discovery API and WebSocket are both public.

## Usage

Start the collector (runs forever):
```bash
python collector.py
```

Run a backtest (after 2-3 days of data collection):
```bash
python backtest.py                          # full parameter sweep
python backtest.py --min-price 0.30 --trail 12 --bins 1
python backtest.py --db /path/to/prices.db  # use a specific database
```

## Hosting on a Server

Any Linux VPS ($5/month) works. Create a systemd service:

```bash
sudo nano /etc/systemd/system/polymarket-collector.service
```

```ini
[Unit]
Description=Polymarket Price Collector
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/user/backtest-collector
ExecStart=/home/user/.venv/bin/python collector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable polymarket-collector
sudo systemctl start polymarket-collector
sudo journalctl -u polymarket-collector -f  # view logs
```

## Database

All data is stored in `data/prices.db` (SQLite). Tables:

| Table | Purpose |
|---|---|
| `events` | Discovered temperature events (city, date, title) |
| `bins` | Individual bins per event (contract_id, range, token IDs) |
| `price_snapshots` | Price at every 5-min interval for every bin |
| `resolutions` | Which bin won each event after resolution |

To download data from a server:
```bash
scp user@server:~/backtest-collector/data/prices.db ./data/
python backtest.py --db ./data/prices.db
```

## Data Requirements

- The collector needs to capture a full event lifecycle (discovery → resolution, typically 24-48 hours)
- After 2-3 days: ~50-100 resolved events, enough for meaningful backtesting
- After 1 week: ~200+ resolved events, reliable parameter optimization
- Database grows ~5-10 MB per day
