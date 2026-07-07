"""
polymarket_prices.py — Poll Polymarket Gamma REST every 2 min, write
one row per bin per poll to db/main.db.

Runs as systemd service polymarket-collector.service. Handles SIGTERM/SIGINT
gracefully so systemctl stop produces clean RUN SUMMARY entries in the log.
"""
import argparse
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
from config.env_loader import DB_PATH, LOG_DIR, TWC_API_KEY
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


# ------------------------------------------------------------------
# Resolution: authoritative Polymarket outcome + measured high temp
# ------------------------------------------------------------------
_RES_COLS = [
    "event_id", "city", "date", "winning_contract_id", "winning_range_low",
    "winning_range_high", "resolved_at", "resolved_at_local", "outcome_source",
    "actual_high_c", "actual_high_f", "actual_high_source", "actual_high_obs",
]

TWC_BASE = "https://api.weather.com"


def _twc_get(path, params):
    try:
        r = httpx.get(f"{TWC_BASE}{path}",
                      params={**params, "language": "en-US", "format": "json", "apiKey": TWC_API_KEY},
                      timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"TWC API error {path}: {e}")
        return None


def _icao_for(conn, city, event_json):
    """The market's ICAO station — from our TWC data (authoritative, already
    parsed) or, failing that, parsed from the event's Wunderground resolution URL."""
    r = conn.execute(
        "SELECT icao FROM twc_current WHERE city = ? AND icao IS NOT NULL "
        "ORDER BY fetched_at DESC LIMIT 1", (city,)).fetchone()
    if r and r["icao"]:
        return r["icao"]
    src = (event_json or {}).get("resolutionSource") or ""
    if not src:
        for m in (event_json or {}).get("markets", []) or []:
            mm = re.search(r"https?://\S*wunderground\.com/\S+", m.get("description") or "")
            if mm:
                src = mm.group(0)
                break
    m = re.search(r"/([A-Z]{4})(?:[/\s\).?]|$)", src)
    return m.group(1) if m else None


def _daily_summary_high(icao, date, unit):
    """Authoritative measured daily high from TWC's historical daily-summary
    (temperatureMax) for the resolution date, requested in the market's unit
    (F for US markets, C otherwise). Returns the native-unit value or None."""
    if not icao:
        return None
    units = "e" if str(unit or "").lower().startswith("f") else "m"
    data = _twc_get("/v3/wx/conditions/historical/dailysummary/30day",
                    {"icaoCode": icao, "units": units})
    if not data:
        return None
    vtl = data.get("validTimeLocal") or []
    tmax = data.get("temperatureMax") or []
    for i, t in enumerate(vtl):
        if t and t[:10] == date and i < len(tmax) and tmax[i] is not None:
            return float(tmax[i])
    return None


def _gamma_winner(event_json):
    """Winning conditionId from a resolved event, or None if not yet settled.
    A contract is settled when the event is closed and its Yes outcomePrice==1."""
    if not event_json or not event_json.get("closed"):
        return None
    winners = []
    for m in event_json.get("markets", []) or []:
        op = m.get("outcomePrices")
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except json.JSONDecodeError:
                continue
        if not op:
            continue
        try:
            yes = float(op[0])
        except (ValueError, TypeError):
            continue
        if yes > 0.99:
            winners.append(m.get("conditionId"))
    return winners[0] if len(winners) == 1 else None


def _price_winner(conn, event_id):
    """Fallback: the bin whose price peaked >= 0.95 (highest peak wins)."""
    return conn.execute(
        """SELECT ps.contract_id, b.range_low, b.range_high
           FROM price_snapshots ps JOIN bins b ON ps.contract_id = b.contract_id
           WHERE ps.event_id = ? AND ps.yes_price >= 0.95
           GROUP BY ps.contract_id ORDER BY MAX(ps.yes_price) DESC LIMIT 1""",
        (event_id,)).fetchone()


def _bin_range(conn, contract_id):
    r = conn.execute("SELECT range_low, range_high FROM bins WHERE contract_id = ?",
                     (contract_id,)).fetchone()
    return (r["range_low"], r["range_high"]) if r else (None, None)


def _measured_high(conn, city, date):
    """Measured daily high (Celsius) from TWC observations on the resolution
    local date: the max of the sampled temperature and temperature_max_since_7am
    (post-7am observations, which capture the peak even if we didn't sample its
    exact minute). Returns (high_c, high_f, n_obs). (None, None, 0) if no obs."""
    rows = conn.execute(
        """SELECT temperature, temperature_max_since_7am, valid_time_local
           FROM twc_current WHERE city = ? AND substr(valid_time_local,1,10) = ?""",
        (city, date)).fetchall()
    if not rows:
        return (None, None, 0)
    highs = []
    for r in rows:
        if r["temperature"] is not None:
            highs.append(r["temperature"])
        ms, vtl = r["temperature_max_since_7am"], r["valid_time_local"]
        if ms is not None and vtl and len(vtl) >= 13:
            try:
                hour = int(vtl[11:13])
            except ValueError:
                hour = None
            if hour is not None and hour >= 7:
                highs.append(ms)
    if not highs:
        return (None, None, len(rows))
    hc = round(max(highs), 1)
    return (hc, round(hc * 9 / 5 + 32, 1), len(rows))


def _resolve_event(conn, event_id, city, date):
    """Resolve one event via the authoritative Gamma outcome (primary) or price
    convergence (fallback). Returns a record dict, or None if not yet resolvable."""
    ev = _gamma_get(f"/events/{event_id}", {})
    contract_id = _gamma_winner(ev)
    if contract_id:
        source = "gamma"
        rl, rh = _bin_range(conn, contract_id)
    else:
        w = _price_winner(conn, event_id)
        if not w:
            return None
        contract_id, rl, rh, source = w["contract_id"], w["range_low"], w["range_high"], "price_convergence"

    # Measured daily high: authoritative TWC daily-summary temperatureMax
    # (requested in the market's unit), falling back to our sampled observations.
    unit_row = conn.execute("SELECT unit FROM bins WHERE event_id = ? LIMIT 1", (event_id,)).fetchone()
    unit = (unit_row["unit"] if unit_row else "celsius") or "celsius"
    high_native = _daily_summary_high(_icao_for(conn, city, ev), date, unit)
    if high_native is not None:
        if str(unit).lower().startswith("f"):
            hf, hc = round(high_native, 1), round((high_native - 32) * 5 / 9, 1)
        else:
            hc, hf = round(high_native, 1), round(high_native * 9 / 5 + 32, 1)
        high_source, nobs = "twc_daily_summary", None
    else:
        hc, hf, nobs = _measured_high(conn, city, date)
        high_source = "twc_observed" if hc is not None else None

    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": event_id, "city": city, "date": date,
        "winning_contract_id": contract_id, "winning_range_low": rl, "winning_range_high": rh,
        "resolved_at": now, "resolved_at_local": local_iso(now, city),
        "outcome_source": source, "actual_high_c": hc, "actual_high_f": hf,
        "actual_high_source": high_source, "actual_high_obs": nobs,
    }


def _write_resolution(conn, rec, upsert=False):
    cols = ",".join(_RES_COLS)
    ph = ",".join("?" * len(_RES_COLS))
    vals = [rec[c] for c in _RES_COLS]
    if upsert:
        setc = ",".join(f"{c}=excluded.{c}" for c in _RES_COLS if c != "event_id")
        conn.execute(f"INSERT INTO resolutions ({cols}) VALUES ({ph}) "
                     f"ON CONFLICT(event_id) DO UPDATE SET {setc}", vals)
    else:
        conn.execute(f"INSERT OR IGNORE INTO resolutions ({cols}) VALUES ({ph})", vals)


def check_resolutions() -> int:
    """Resolve past events each poll. Also re-checks events still on the
    price-convergence fallback so they upgrade to the authoritative Gamma
    outcome (and refresh the measured high) once Polymarket settles them."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    found = 0
    try:
        events = conn.execute(
            """SELECT e.event_id, e.city, e.date, e.resolved,
                      COALESCE(r.outcome_source, '') AS src
               FROM events e LEFT JOIN resolutions r ON e.event_id = r.event_id
               WHERE e.date < date('now')
                 AND (e.resolved = 0 OR r.outcome_source = 'price_convergence')"""
        ).fetchall()
        for ev in events:
            rec = _resolve_event(conn, ev["event_id"], ev["city"], ev["date"])
            if not rec:
                continue
            if ev["src"] == "gamma" and rec["outcome_source"] != "gamma":
                continue  # never downgrade an authoritative resolution
            _write_resolution(conn, rec, upsert=True)
            conn.execute("UPDATE events SET resolved = 1 WHERE event_id = ?", (ev["event_id"],))
            conn.commit()  # per event: don't hold the write lock across API calls
            if not ev["resolved"] or (ev["src"] == "price_convergence"
                                      and rec["outcome_source"] == "gamma"):
                found += 1
                logger.info(f"Resolved {ev['city']} {ev['date']} via {rec['outcome_source']}")
    finally:
        conn.close()
    if found:
        _session["resolutions"] += found
    return found


def backfill_resolutions() -> int:
    """One-time: re-resolve every past event (resolved or not) with the
    authoritative Gamma outcome + measured high, upserting into resolutions."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    n = 0
    try:
        past = conn.execute(
            "SELECT event_id, city, date FROM events WHERE date < date('now') ORDER BY date, city"
        ).fetchall()
        logger.info(f"backfill_resolutions: {len(past)} past event(s)")
        for ev in past:
            rec = _resolve_event(conn, ev["event_id"], ev["city"], ev["date"])
            if not rec:
                logger.info(f"backfill: {ev['city']} {ev['date']} -> UNRESOLVED (no Gamma/price signal)")
                continue
            _write_resolution(conn, rec, upsert=True)
            conn.execute("UPDATE events SET resolved = 1 WHERE event_id = ?", (ev["event_id"],))
            conn.commit()  # per event: don't hold the write lock across API calls
            n += 1
            logger.info(
                f"backfill: {ev['city']:<12} {ev['date']} -> {rec['outcome_source']:<16} "
                f"bin=[{rec['winning_range_low']},{rec['winning_range_high']}] "
                f"high={rec['actual_high_c']}C/{rec['actual_high_f']}F ({rec['actual_high_source']})")
        conn.commit()
    finally:
        conn.close()
    logger.info(f"backfill_resolutions: wrote {n} resolution(s)")
    return n


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
    parser = argparse.ArgumentParser(description="Polymarket price collector")
    parser.add_argument("--backfill-resolutions", action="store_true",
                        help="Re-resolve all past events via Gamma + measured high, then exit")
    args = parser.parse_args()
    if args.backfill_resolutions:
        n = backfill_resolutions()
        print(f"backfilled {n} resolution(s)")
        return

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
