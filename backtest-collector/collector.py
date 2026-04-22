"""
collector.py — Standalone price data collector for Polymarket temperature markets.

Runs 24/7 on any machine. No weather APIs, no trading logic, no API keys needed.
Captures price snapshots for ALL bins in ALL active temperature events every 5 minutes
via WebSocket, and detects event resolutions when markets settle.

Data is stored in a local SQLite database (data/prices.db) for backtesting.

Usage:
    pip install -r requirements.txt
    python collector.py

To run as a background service on Linux:
    nohup python collector.py > collector.log 2>&1 &

    Or create a systemd service:
    [Unit]
    Description=Polymarket Price Collector
    After=network.target

    [Service]
    Type=simple
    WorkingDir=/path/to/backtest-collector
    ExecStart=/path/to/python collector.py
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone, date
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("collector.log"),
    ],
)
logger = logging.getLogger("collector")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prices.db")
GAMMA_BASE = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SNAPSHOT_INTERVAL = 120  # 2 minutes
DISCOVERY_INTERVAL = 14400  # 4 hours


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            event_id        TEXT PRIMARY KEY,
            city            TEXT,
            date            TEXT,
            event_title     TEXT,
            n_bins          INTEGER,
            discovered_at   TEXT,
            resolved        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS bins (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT NOT NULL,
            contract_id     TEXT NOT NULL UNIQUE,
            question        TEXT,
            range_low       REAL,
            range_high      REAL,
            unit            TEXT,
            yes_token_id    TEXT,
            no_token_id     TEXT
        );

        CREATE TABLE IF NOT EXISTS price_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT NOT NULL,
            contract_id     TEXT NOT NULL,
            yes_price       REAL,
            volume_usd      REAL,
            liquidity_usd   REAL,
            recorded_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS resolutions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT UNIQUE NOT NULL,
            city            TEXT,
            date            TEXT,
            winning_contract_id TEXT,
            winning_range_low   REAL,
            winning_range_high  REAL,
            resolved_at     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_price_snap_event
            ON price_snapshots(event_id, contract_id, recorded_at);
        CREATE INDEX IF NOT EXISTS idx_price_snap_time
            ON price_snapshots(recorded_at);
        CREATE INDEX IF NOT EXISTS idx_bins_event
            ON bins(event_id);
    """)
    conn.close()
    logger.info(f"Database initialized: {DB_PATH}")


# ---------------------------------------------------------------------------
# Polymarket Discovery
# ---------------------------------------------------------------------------

def _gamma_get(path: str, params: dict) -> list | dict | None:
    try:
        r = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Gamma API error: {e}")
        return None


def _parse_range(question: str) -> tuple[float | None, float | None]:
    """Extract temperature range from question text."""
    import re
    m = re.search(r"between\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)", question, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"be\s+(-?\d+\.?\d*)\s*[°]?\s*[CF]?\s+on", question, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return v, v
    m = re.search(r"at least\s+(-?\d+\.?\d*)", question, re.IGNORECASE)
    if m:
        return float(m.group(1)), None
    m = re.search(r"at most\s+(-?\d+\.?\d*)", question, re.IGNORECASE)
    if m:
        return None, float(m.group(1))
    return None, None


def discover_events() -> dict[str, dict]:
    """Find all active highest-temperature events on Polymarket."""
    logger.info("Discovering active temperature events...")

    all_events = []
    offset = 0
    while True:
        data = _gamma_get("/events", {
            "tag_slug": "highest-temperature",
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": offset,
        })
        if not data:
            break
        events = data if isinstance(data, list) else []
        if not events:
            break
        all_events.extend(events)
        if len(events) < 100:
            break
        offset += 100

    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()
    token_map: dict[str, dict] = {}  # token_id -> {event_id, contract_id, ...}
    total_bins = 0

    for event in all_events:
        title = event.get("title", "")
        markets = event.get("markets", [])
        if not markets:
            continue

        # Parse city and date from title
        city = None
        event_date = None
        import re
        m = re.search(r"temperature in (.+?) on (.+?)[\?$]", title, re.IGNORECASE)
        if m:
            city = m.group(1).strip()
            try:
                from dateutil import parser as dateparser
                event_date = dateparser.parse(m.group(2).strip()).strftime("%Y-%m-%d")
            except Exception:
                # Simple fallback
                month_day = m.group(2).strip()
                for fmt in ["%B %d", "%b %d"]:
                    try:
                        d = datetime.strptime(month_day, fmt)
                        event_date = d.replace(year=datetime.now().year).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        pass

        if not city or not event_date:
            continue

        event_id = str(event.get("id", ""))

        # Insert/update event
        conn.execute("""
            INSERT OR IGNORE INTO events (event_id, city, date, event_title, n_bins, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_id, city, event_date, title, len(markets), now))

        for mkt in markets:
            contract_id = mkt.get("conditionId", "")
            question = mkt.get("question", "")
            rl, rh = _parse_range(question)

            tokens = mkt.get("clobTokenIds")
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            yes_tid = tokens[0] if tokens and len(tokens) > 0 else None
            no_tid = tokens[1] if tokens and len(tokens) > 1 else None

            unit = "fahrenheit" if any(x in question.lower() for x in ["°f", "fahrenheit"]) else "celsius"

            conn.execute("""
                INSERT OR IGNORE INTO bins
                    (event_id, contract_id, question, range_low, range_high, unit, yes_token_id, no_token_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_id, contract_id, question, rl, rh, unit, yes_tid, no_tid))

            if yes_tid:
                token_map[yes_tid] = {
                    "event_id": event_id,
                    "contract_id": contract_id,
                    "city": city,
                    "date": event_date,
                }
                total_bins += 1

            # Write initial price snapshot from discovery data
            prices = mkt.get("outcomePrices")
            if isinstance(prices, str):
                prices = json.loads(prices)
            if prices and len(prices) > 0:
                yes_price = float(prices[0])
                conn.execute("""
                    INSERT INTO price_snapshots (event_id, contract_id, yes_price, volume_usd, liquidity_usd, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (event_id, contract_id, yes_price,
                      float(mkt.get("volumeNum") or 0),
                      float(mkt.get("liquidityNum") or 0), now))

    conn.commit()
    conn.close()

    logger.info(f"Discovery complete: {len(all_events)} events, {total_bins} bins, {len(token_map)} tokens")
    return token_map


# ---------------------------------------------------------------------------
# Resolution Detection
# ---------------------------------------------------------------------------

def check_resolutions():
    """Check past events for resolved markets and record winners."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Find unresolved events whose date has passed
    past_events = conn.execute("""
        SELECT event_id, city, date FROM events
        WHERE resolved = 0 AND date < date('now')
    """).fetchall()

    if not past_events:
        conn.close()
        return 0

    found = 0
    for ev in past_events:
        eid = ev["event_id"]

        # Check if any bin hit 95%+ in snapshots
        winner = conn.execute("""
            SELECT ps.contract_id, MAX(ps.yes_price) as peak,
                   b.range_low, b.range_high
            FROM price_snapshots ps
            JOIN bins b ON ps.contract_id = b.contract_id
            WHERE ps.event_id = ? AND ps.yes_price >= 0.95
            GROUP BY ps.contract_id
            ORDER BY peak DESC
            LIMIT 1
        """, (eid,)).fetchone()

        if not winner:
            continue

        conn.execute("""
            INSERT OR IGNORE INTO resolutions
                (event_id, city, date, winning_contract_id,
                 winning_range_low, winning_range_high, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (eid, ev["city"], ev["date"], winner["contract_id"],
              winner["range_low"], winner["range_high"],
              datetime.now(timezone.utc).isoformat()))

        conn.execute("UPDATE events SET resolved = 1 WHERE event_id = ?", (eid,))
        found += 1

    conn.commit()
    conn.close()

    if found:
        logger.info(f"Resolved {found} event(s)")
    return found


# ---------------------------------------------------------------------------
# WebSocket Price Stream
# ---------------------------------------------------------------------------

_live_prices: dict[str, float] = {}
_price_lock = threading.Lock()
_token_metadata: dict[str, dict] = {}
_meta_lock = threading.Lock()
_ws_stop = threading.Event()


def update_tokens(token_map: dict[str, dict]):
    """Update the set of tokens to subscribe to."""
    with _meta_lock:
        _token_metadata.update(token_map)
    logger.info(f"Token map updated: {len(_token_metadata)} tokens tracked")


def write_price_snapshots():
    """Write current live prices to DB."""
    with _price_lock:
        prices = dict(_live_prices)
    with _meta_lock:
        meta = dict(_token_metadata)

    if not prices:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)

    rows = []
    for token_id, price in prices.items():
        m = meta.get(token_id)
        if not m:
            continue
        rows.append((m["event_id"], m["contract_id"], price, None, None, now))

    if rows:
        conn.executemany("""
            INSERT INTO price_snapshots
                (event_id, contract_id, yes_price, volume_usd, liquidity_usd, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

    conn.close()
    logger.info(f"Snapshot: {len(rows)} prices written")
    return len(rows)


async def _ws_loop():
    """WebSocket connection with auto-reconnect."""
    import websockets

    reconnect_delay = 5

    while not _ws_stop.is_set():
        with _meta_lock:
            tokens = list(_token_metadata.keys())

        if not tokens:
            logger.info("No tokens to subscribe — waiting 30s")
            await asyncio.sleep(30)
            continue

        try:
            logger.info(f"Connecting to WebSocket with {len(tokens)} tokens...")
            async with websockets.connect(WS_URL, ping_interval=None, close_timeout=5) as ws:
                # Subscribe in batches (WebSocket may have message size limits)
                batch_size = 500
                for i in range(0, len(tokens), batch_size):
                    batch = tokens[i:i + batch_size]
                    await ws.send(json.dumps({
                        "assets_ids": batch,
                        "type": "market",
                    }))

                logger.info(f"WebSocket connected, subscribed to {len(tokens)} tokens")
                reconnect_delay = 5

                last_ping = time.time()
                last_snapshot = time.time()
                last_discovery = time.time()

                while not _ws_stop.is_set():
                    # Ping
                    if time.time() - last_ping >= 10:
                        try:
                            await ws.send("PING")
                            last_ping = time.time()
                        except Exception:
                            break

                    # Periodic snapshot write
                    if time.time() - last_snapshot >= SNAPSHOT_INTERVAL:
                        try:
                            write_price_snapshots()
                            check_resolutions()
                        except Exception as e:
                            logger.warning(f"Snapshot/resolution error: {e}")
                        last_snapshot = time.time()

                    # Periodic re-discovery
                    if time.time() - last_discovery >= DISCOVERY_INTERVAL:
                        try:
                            new_tokens = discover_events()
                            # Subscribe to any new tokens
                            with _meta_lock:
                                new_only = {k: v for k, v in new_tokens.items()
                                           if k not in _token_metadata}
                                _token_metadata.update(new_tokens)
                            if new_only:
                                await ws.send(json.dumps({
                                    "assets_ids": list(new_only.keys()),
                                    "type": "market",
                                }))
                                logger.info(f"Subscribed to {len(new_only)} new tokens")
                        except Exception as e:
                            logger.warning(f"Re-discovery error: {e}")
                        last_discovery = time.time()

                    # Receive
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

                    if raw == "PONG":
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(msg, list):
                        for m in msg:
                            if isinstance(m, dict):
                                _process_msg(m)
                    elif isinstance(msg, dict):
                        _process_msg(msg)

        except Exception as e:
            if _ws_stop.is_set():
                break
            logger.warning(f"WebSocket error: {e} — reconnecting in {reconnect_delay}s")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)


def _process_msg(msg: dict):
    """Extract price from WebSocket message."""
    event_type = msg.get("event_type")
    token_id = msg.get("asset_id")
    price_str = msg.get("price")

    if token_id and price_str and event_type in ("last_trade_price", "price_change"):
        try:
            with _price_lock:
                _live_prices[token_id] = float(price_str)
        except (ValueError, TypeError):
            pass
    elif event_type == "book" and token_id:
        bids = msg.get("bids") or []
        if bids:
            try:
                best = max(bids, key=lambda b: float(b.get("price", 0)))
                with _price_lock:
                    _live_prices[token_id] = float(best["price"])
            except (ValueError, TypeError, KeyError):
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("  Polymarket Price Collector Starting")
    logger.info("=" * 60)

    init_db()

    # Initial discovery
    token_map = discover_events()
    update_tokens(token_map)

    # Check for any past resolutions we missed
    check_resolutions()

    # Stats
    conn = sqlite3.connect(DB_PATH)
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_bins = conn.execute("SELECT COUNT(*) FROM bins").fetchone()[0]
    n_snaps = conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    n_resolved = conn.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]
    conn.close()

    logger.info(f"DB status: {n_events} events, {n_bins} bins, {n_snaps} snapshots, {n_resolved} resolved")
    logger.info(f"Capturing prices every {SNAPSHOT_INTERVAL}s, re-discovering every {DISCOVERY_INTERVAL}s")
    logger.info("Press Ctrl+C to stop")

    # Run WebSocket loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_ws_loop())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        _ws_stop.set()
        loop.close()


if __name__ == "__main__":
    main()
