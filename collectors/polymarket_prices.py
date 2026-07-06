"""
polymarket_prices.py — Poll Polymarket Gamma REST every 2 min, write
one row per bin per poll to db/main.db.

Runs as systemd service polymarket-collector.service. Handles SIGTERM/SIGINT
gracefully so systemctl stop produces clean RUN SUMMARY entries in the log.
"""
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config.env_loader import DB_PATH, LOG_DIR
from config.cities import local_iso

LOG_PATH = os.path.join(LOG_DIR, "polymarket_collector.log")
ACTIVITY_LOG_PATH = os.path.join(LOG_DIR, "activity.log")
GAMMA_BASE = "https://gamma-api.polymarket.com"
SNAPSHOT_INTERVAL = 120     # 2 minutes
DISCOVERY_INTERVAL = 14400  # 4 hours
HEALTH_LOG_INTERVAL = 300   # 5 minutes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("polymarket_prices")

_session = {
    "started_at": None, "startup_ok": False, "polls": 0, "poll_errors": 0,
    "snapshots_written": 0, "discoveries": 0, "resolutions": 0,
    "last_poll_at": None, "shutdown_signal": None,
}
_stop = threading.Event()


def _append_activity_row(status: str, **fields):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts = [ts, "polymarket", f"{status:<5}"]
    parts.extend(f"{k}={v}" for k, v in fields.items())
    try:
        with open(ACTIVITY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(" | ".join(parts) + "\n")
    except Exception as e:
        logger.warning(f"activity.log write failed: {e}")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def _install_signal_handlers():
    def _shutdown(signum, _frame):
        signame = signal.Signals(signum).name
        logger.info(f"Received {signame} - shutting down")
        _session["shutdown_signal"] = signame
        _stop.set()
    for sig_name in ("SIGTERM", "SIGINT"):
        try:
            signal.signal(getattr(signal, sig_name), _shutdown)
        except (ValueError, OSError, AttributeError):
            pass


def _log_health():
    started = _session["started_at"]
    uptime = _fmt_duration((datetime.now(timezone.utc) - started).total_seconds()) if started else "?"
    last = _session["last_poll_at"]
    since = _fmt_duration((datetime.now(timezone.utc) - last).total_seconds()) if last else "never"
    logger.info(
        f"HEALTH | uptime={uptime} | polls={_session['polls']} "
        f"(errors={_session['poll_errors']}, last {since} ago) | "
        f"snapshots={_session['snapshots_written']} | "
        f"discoveries={_session['discoveries']} resolutions={_session['resolutions']}"
    )


def _gamma_get(path: str, params: dict):
    try:
        r = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Gamma API error: {e}")
        return None


def _fetch_active_temperature_events():
    all_events, offset = [], 0
    while True:
        data = _gamma_get("/events", {
            "tag_slug": "highest-temperature",
            "active": "true", "closed": "false",
            "limit": 100, "offset": offset,
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
    return all_events


def _parse_range(question: str):
    q = question
    m = re.search(r"between\s+(-?\d+\.?\d*)\s*°?\s*[CF]?\s+and\s+(-?\d+\.?\d*)", q, re.IGNORECASE)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r"(-?\d+\.?\d*)\s*°?\s*[CF]?\s*[-–—]\s*(-?\d+\.?\d*)\s*°?\s*[CF]?", q, re.IGNORECASE)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r"(-?\d+\.?\d*)\s*°?\s*[CF]?\s+or\s+(?:above|higher|more|greater|over)", q, re.IGNORECASE)
    if m: return float(m.group(1)), None
    m = re.search(r"at least\s+(-?\d+\.?\d*)", q, re.IGNORECASE)
    if m: return float(m.group(1)), None
    m = re.search(r"(-?\d+\.?\d*)\s*°?\s*[CF]?\s+or\s+(?:below|lower|less|under)", q, re.IGNORECASE)
    if m: return None, float(m.group(1))
    m = re.search(r"at most\s+(-?\d+\.?\d*)", q, re.IGNORECASE)
    if m: return None, float(m.group(1))
    m = re.search(r"be\s+(?:exactly\s+)?(-?\d+\.?\d*)\s*°?\s*[CF]?(?:\s+on|\?|$)", q, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return v, v
    return None, None


def discover_events() -> int:
    logger.info("Discovering active temperature events...")
    all_events = _fetch_active_temperature_events()
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()
    total_bins = 0
    for event in all_events:
        title = event.get("title", "")
        markets = event.get("markets", [])
        if not markets:
            continue
        city, event_date = None, None
        m = re.search(r"temperature in (.+?) on (.+?)[\?$]", title, re.IGNORECASE)
        if m:
            city = m.group(1).strip()
            try:
                from dateutil import parser as dateparser
                event_date = dateparser.parse(m.group(2).strip()).strftime("%Y-%m-%d")
            except Exception:
                for fmt in ["%B %d", "%b %d"]:
                    try:
                        d = datetime.strptime(m.group(2).strip(), fmt)
                        event_date = d.replace(year=datetime.now().year).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        pass
        if not city or not event_date:
            continue
        event_id = str(event.get("id", ""))
        conn.execute(
            """INSERT OR IGNORE INTO events
               (event_id, city, date, event_title, n_bins, discovered_at, discovered_at_local)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event_id, city, event_date, title, len(markets), now, local_iso(now, city)),
        )
        for mkt in markets:
            contract_id = mkt.get("conditionId", "")
            if not contract_id:
                continue
            question = mkt.get("question", "")
            rl, rh = _parse_range(question)
            tokens = mkt.get("clobTokenIds")
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            yes_tid = tokens[0] if tokens and len(tokens) > 0 else None
            no_tid = tokens[1] if tokens and len(tokens) > 1 else None
            unit = "fahrenheit" if any(x in question.lower() for x in ["°f", "fahrenheit"]) else "celsius"
            conn.execute(
                """INSERT OR IGNORE INTO bins
                   (event_id, contract_id, question, range_low, range_high, unit, yes_token_id, no_token_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, contract_id, question, rl, rh, unit, yes_tid, no_tid),
            )
            total_bins += 1
    conn.commit()
    conn.close()
    _session["discoveries"] += 1
    logger.info(f"Discovery complete: {len(all_events)} events, {total_bins} bins")
    return len(all_events)


def poll_all_prices():
    all_events = _fetch_active_temperature_events()
    rows = []
    for event in all_events:
        event_id = str(event.get("id", ""))
        if not event_id:
            continue
        cm = re.search(r"temperature in (.+?) on", event.get("title", "") or "", re.IGNORECASE)
        city = cm.group(1).strip() if cm else None
        for mkt in event.get("markets", []):
            contract_id = mkt.get("conditionId", "")
            if not contract_id:
                continue
            prices = mkt.get("outcomePrices")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except json.JSONDecodeError:
                    continue
            if not prices:
                continue
            try:
                yes_price = float(prices[0])
            except (ValueError, TypeError):
                continue
            volume = float(mkt.get("volumeNum") or 0)
            liquidity = float(mkt.get("liquidityNum") or 0)
            rows.append((event_id, contract_id, yes_price, volume, liquidity, city))
    return rows


def write_price_snapshots(rows):
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    full = [(eid, cid, yp, vol, liq, now, local_iso(now, city))
            for eid, cid, yp, vol, liq, city in rows]
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        """INSERT INTO price_snapshots
           (event_id, contract_id, yes_price, volume_usd, liquidity_usd,
            recorded_at, recorded_at_local)
           VALUES (?, ?, ?, ?, ?, ?, ?)""", full)
    conn.commit()
    conn.close()
    _session["snapshots_written"] += len(full)
    _session["last_poll_at"] = datetime.now(timezone.utc)
    return len(full)


def check_resolutions() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    past = conn.execute(
        """SELECT event_id, city, date FROM events
           WHERE resolved = 0 AND date < date('now')"""
    ).fetchall()
    if not past:
        conn.close()
        return 0
    found = 0
    for ev in past:
        eid = ev["event_id"]
        winner = conn.execute(
            """SELECT ps.contract_id, MAX(ps.yes_price) AS peak,
                      b.range_low, b.range_high
               FROM price_snapshots ps JOIN bins b ON ps.contract_id = b.contract_id
               WHERE ps.event_id = ? AND ps.yes_price >= 0.95
               GROUP BY ps.contract_id ORDER BY peak DESC LIMIT 1""",
            (eid,),
        ).fetchone()
        if not winner:
            continue
        res_now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR IGNORE INTO resolutions
               (event_id, city, date, winning_contract_id, winning_range_low,
                winning_range_high, resolved_at, resolved_at_local)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (eid, ev["city"], ev["date"], winner["contract_id"],
             winner["range_low"], winner["range_high"],
             res_now, local_iso(res_now, ev["city"])),
        )
        conn.execute("UPDATE events SET resolved = 1 WHERE event_id = ?", (eid,))
        found += 1
    conn.commit()
    conn.close()
    if found:
        _session["resolutions"] += found
        logger.info(f"Resolved {found} event(s)")
    return found


def _log_run_summary(exit_reason: str):
    started = _session["started_at"]
    uptime = _fmt_duration((datetime.now(timezone.utc) - started).total_seconds()) if started else "?"
    success = _session["startup_ok"] and (
        exit_reason in ("keyboard_interrupt", "loop_exited")
        or exit_reason.startswith("signal:")
    )
    logger.info("-" * 72)
    logger.info("  RUN SUMMARY")
    logger.info("-" * 72)
    logger.info(f"  success:      {success}")
    logger.info(f"  exit reason:  {exit_reason}")
    logger.info(f"  polls:        {_session['polls']}")
    logger.info(f"  poll errors:  {_session['poll_errors']}")
    logger.info(f"  snapshots:    {_session['snapshots_written']}")
    logger.info(f"  discoveries:  {_session['discoveries']}")
    logger.info(f"  resolutions:  {_session['resolutions']}")
    logger.info(f"  uptime:       {uptime}")
    _append_activity_row(
        "OK" if success else "FAIL",
        uptime=uptime, reason=exit_reason,
        snapshots=_session["snapshots_written"],
        polls=_session["polls"], poll_errors=_session["poll_errors"],
        discoveries=_session["discoveries"], resolutions=_session["resolutions"],
    )


def main():
    _session["started_at"] = datetime.now(timezone.utc)
    _append_activity_row("START", pid=os.getpid())
    logger.info("=" * 72)
    logger.info("  POLYMARKET PRICE COLLECTOR - RUN START")
    logger.info("=" * 72)
    logger.info(f"  db path:  {DB_PATH}")
    logger.info(f"  log dir:  {LOG_DIR}")
    logger.info(f"  poll every {SNAPSHOT_INTERVAL}s, discovery every {DISCOVERY_INTERVAL}s")
    _install_signal_handlers()

    exit_reason = "unknown"
    try:
        discover_events()
        check_resolutions()
        rows = poll_all_prices()
        n = write_price_snapshots(rows)
        _session["polls"] += 1
        logger.info(f"Initial poll: {n} prices written")
        _session["startup_ok"] = True
    except Exception as e:
        logger.exception(f"Startup FAILED: {e}")
        _log_run_summary("startup_failed")
        raise

    now = time.time()
    next_poll = now + SNAPSHOT_INTERVAL
    next_discovery = now + DISCOVERY_INTERVAL
    next_health = now + HEALTH_LOG_INTERVAL

    try:
        while not _stop.is_set():
            now = time.time()
            if now >= next_poll:
                try:
                    rows = poll_all_prices()
                    n = write_price_snapshots(rows)
                    check_resolutions()
                    _session["polls"] += 1
                    logger.info(f"Snapshot: {n} prices written")
                except Exception as e:
                    logger.warning(f"Poll error: {e}")
                    _session["poll_errors"] += 1
                next_poll = time.time() + SNAPSHOT_INTERVAL
            if now >= next_discovery:
                try:
                    discover_events()
                except Exception as e:
                    logger.warning(f"Discovery error: {e}")
                next_discovery = time.time() + DISCOVERY_INTERVAL
            if now >= next_health:
                _log_health()
                next_health = time.time() + HEALTH_LOG_INTERVAL
            sleep_s = max(0.5, min(30.0, min(next_poll, next_discovery, next_health) - time.time()))
            _stop.wait(timeout=sleep_s)
        exit_reason = f"signal:{_session['shutdown_signal']}" if _session["shutdown_signal"] else "loop_exited"
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        exit_reason = f"error: {e}"
    finally:
        _stop.set()
        _log_run_summary(exit_reason)


if __name__ == "__main__":
    main()
