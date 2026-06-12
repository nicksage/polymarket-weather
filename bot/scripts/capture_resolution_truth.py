"""
capture_resolution_truth.py — Phase 0a of the HRRR ceiling plan.

For each newly-resolved Polymarket market, capture the data needed to
decompose the bot-vs-settlement gap into its component sources:

  1. bot_observed_max_c     — what the bot stored for the day
  2. metar_peak_body_c      — whole-°C body value at peak hour
  3. metar_peak_t_group_c   — T-group tenths-precision at peak synoptic
  4. wunderground_high_c    — Wunderground's daily-high display value
  5. winning_range_low/high — from resolutions (which bin won)

With these five values per resolved market, Phase 0b can answer:

  - Did the T-group fix actually close the body-vs-precise gap?
    (compare metar_peak_body_c to metar_peak_t_group_c — should be
    within rounding boundary; if T-group is consistently 0.2–0.5°C
    below body, the fix is working)

  - Does T-group precision match settlement?
    (compare metar_peak_t_group_c to wunderground_high_c — if close,
    Wunderground uses T-group precision; if Wunderground is lower
    by ~1°C, there's a separate DSM aggregation issue we haven't
    addressed)

  - Does the bot's observed_max land in the correct bin?
    (place bot_observed_max_c in the binning grid, compare to
    winning_range_low/high; mismatch = real settle_divergence)

Without this script's output, every audit relies on the manual
METAR-vs-Wunderground comparison the user did by hand for Atlanta
2026-06-12.  This automates it.

Run nightly via cron:
    0 4 * * *  /path/to/venv/bin/python -m scripts.capture_resolution_truth

Or one-shot for backfill:
    cd bot && python -m scripts.capture_resolution_truth --backfill-days 30
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_BOT_DIR), ".env"), override=True)
except ImportError:
    pass

from config import DB_PATH                                  # type: ignore
from scripts.intraday_predictor import parse_metar_t_group  # type: ignore
from station_meta import CITY_STATIONS                       # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("capture_resolution_truth")

DEFAULT_COLLECTOR_DB = os.path.expanduser(
    "~/apps/weather-data/backtest-collector/data/prices.db")

# Wunderground daily-history page pattern.  HTML scrape — fragile but
# functional and Wunderground does not throttle reasonable nightly use.
# If Wunderground changes its layout, the regex needs updating; the
# rest of the script keeps working with the wunderground_high_c column
# left NULL.
WUNDERGROUND_URL = (
    "https://www.wunderground.com/history/daily/{icao}/date/{ymd}"
)


# ============================================================
# Metar-side capture: peak body value + peak T-group value
# ============================================================

def fetch_peak_metar_values(conn, icao: str, event_date: str
                              ) -> tuple[float | None, float | None,
                                          str | None, str | None]:
    """For (icao, event_date), return:
       (peak_body_c, peak_t_group_c, peak_cycle_utc, notes)

    peak_body_c is the highest body value (whole-°C, NOT conservative-
    bounded) seen across all stored cycles.

    peak_t_group_c is the T-group value (if any) from the synoptic METAR
    near the peak hour.  Selected by finding the synoptic with the
    highest T-group value in the same UTC hour as the peak body reading.
    If no synoptic with a T-group exists in that hour, fall back to
    nearest hour ±2 h.
    """
    # All stored METARs for this (icao, date), ordered by temp descending.
    # Use raw_message presence as a synoptic-ish indicator — only synoptic
    # METARs have rawMessage populated in our store (5-min cycles get null).
    rows = conn.execute(
        """SELECT cycle_timestamp_utc, temp_c, temp_precision, raw_message
           FROM raw_metar_log
           WHERE icao = ? AND event_date = ?
           ORDER BY temp_c DESC, cycle_timestamp_utc DESC""",
        (icao, event_date),
    ).fetchall()
    if not rows:
        return None, None, None, "no_metar_data"

    # Peak body value: highest whole-°C reading.  Note: post-T-group-fix,
    # temp_c may carry the T-group value rather than the body.  We
    # disambiguate with temp_precision: rows where precision='whole' carry
    # the body value + conservative -0.5°C offset (so add 0.5 back to
    # recover the body), rows where precision='tenths' carry the T-group.
    body_candidates: list[tuple[float, str]] = []
    t_group_candidates: list[tuple[float, str, str]] = []
    for r in rows:
        t = r["temp_c"]
        prec = r["temp_precision"]
        raw = r["raw_message"]
        ts = r["cycle_timestamp_utc"]
        if t is None:
            continue
        # Recover body value (undo the conservative -0.5°C if applied)
        if prec == "whole":
            body_c = float(t) + 0.5   # original body value
            body_candidates.append((body_c, ts))
        elif prec == "tenths":
            # T-group rows ARE the precise value
            t_group_candidates.append((float(t), ts, raw or ""))
        else:
            # Legacy rows (pre-T-group-fix) carry the body value directly
            # and we have no precision label.  Treat as body.
            body_candidates.append((float(t), ts))

    # Also pull T-groups directly from raw_message for any synoptic METAR
    # we have stored, in case the temp_precision column wasn't set on
    # older rows.
    for r in rows:
        if r["raw_message"]:
            tg_temp, _ = parse_metar_t_group(r["raw_message"])
            if tg_temp is not None:
                t_group_candidates.append(
                    (tg_temp, r["cycle_timestamp_utc"], r["raw_message"]))

    peak_body = max(body_candidates, default=(None, None))
    peak_body_c = peak_body[0]
    peak_cycle_utc = peak_body[1]

    peak_t_group = max(t_group_candidates, default=(None, None, None))
    peak_t_group_c = peak_t_group[0]

    notes = []
    if peak_body_c is None:
        notes.append("no_body_value")
    if peak_t_group_c is None:
        notes.append("no_t_group_in_remarks")
    if not body_candidates and not t_group_candidates:
        notes.append("no_temperature_data_at_all")
    return peak_body_c, peak_t_group_c, peak_cycle_utc, ",".join(notes) or None


# ============================================================
# Wunderground capture (HTML scrape — best-effort)
# ============================================================

# Wunderground's "High Temp" cell uses different markup over time.
# We match a few patterns conservatively.  If none match, return None
# and the audit can still use the T-group decomposition.
_WG_HIGH_PATTERNS = [
    re.compile(r'High\s*Temp[ature]*\s*</[^>]+>\s*<[^>]+>\s*(\d+(?:\.\d+)?)',
                re.IGNORECASE),
    re.compile(r'"high"\s*:\s*\{[^}]*"value"\s*:\s*(\d+(?:\.\d+)?)',
                re.IGNORECASE),
    re.compile(r'High\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*°', re.IGNORECASE),
]


def fetch_wunderground_daily_high_c(icao: str, event_date: str,
                                       timeout_s: float = 30.0
                                       ) -> tuple[float | None, str | None]:
    """Scrape Wunderground's daily-history page for the daily high.
    Returns (high_c, note).  high_c is None if not parseable; note
    carries the failure mode for diagnostics.

    Wunderground's daily summary is in °F by default for US stations.
    We convert to °C for storage to match the rest of the schema.
    """
    url = WUNDERGROUND_URL.format(icao=icao, ymd=event_date)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                          "polymarket-weather-bot/audit (offline review)"
        }
        r = httpx.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        body = r.text
    except Exception as e:
        return None, f"wunderground_fetch_failed:{type(e).__name__}"

    for pat in _WG_HIGH_PATTERNS:
        m = pat.search(body)
        if m:
            try:
                high_f = float(m.group(1))
                high_c = (high_f - 32) * 5 / 9
                return round(high_c, 2), None
            except ValueError:
                continue
    return None, "wunderground_parse_failed"


# ============================================================
# Main capture
# ============================================================

def capture_one(conn, collector_conn, event_id: str,
                  fetch_wunderground: bool) -> dict:
    """Capture all five reference values for one resolved market.
    Returns a dict suitable for insertion into resolution_observations."""
    res_row = collector_conn.execute(
        """SELECT event_id, city, date,
                  winning_contract_id,
                  winning_range_low, winning_range_high
           FROM resolutions WHERE event_id = ?""",
        (event_id,),
    ).fetchone()
    if not res_row:
        return {"event_id": event_id, "capture_notes": "no_resolution_row"}

    city = res_row["city"]
    event_date = res_row["date"]
    icao = None
    meta = CITY_STATIONS.get(city)
    if meta:
        icao = meta[0]

    # Bot's observed_max — EOD value from paper_predictor_signals
    bot_row = conn.execute(
        """SELECT MAX(observed_max_c) AS bot_max
           FROM paper_predictor_signals
           WHERE city = ? AND event_date = ?
             AND observed_max_c IS NOT NULL""",
        (city, event_date),
    ).fetchone()
    bot_observed_max_c = bot_row["bot_max"] if bot_row else None

    # Metar peak body + T-group
    peak_body, peak_t_group, peak_cycle, metar_notes = (
        fetch_peak_metar_values(conn, icao, event_date)
        if icao else (None, None, None, "no_icao_mapping")
    )

    # Wunderground daily high — optional, fragile
    wunderground_c = None
    wunderground_notes = None
    if fetch_wunderground and icao:
        wunderground_c, wunderground_notes = (
            fetch_wunderground_daily_high_c(icao, event_date)
        )

    notes_parts = [n for n in (metar_notes, wunderground_notes) if n]
    notes = "; ".join(notes_parts) if notes_parts else None

    # Construct a label for the winning bin.  range_low/high are integer
    # °F for US markets; build "92-93°F" style.
    winning_label = None
    wl = res_row["winning_range_low"]
    wh = res_row["winning_range_high"]
    if wl is not None and wh is not None:
        winning_label = f"{int(wl)}-{int(wh)}°F"
    elif wh is not None:
        winning_label = f"≤{int(wh)}°F"
    elif wl is not None:
        winning_label = f"≥{int(wl)}°F"

    return {
        "event_id":            event_id,
        "city":                city,
        "event_date":          event_date,
        "icao":                icao,
        "bot_observed_max_c":  bot_observed_max_c,
        "metar_peak_body_c":   peak_body,
        "metar_peak_t_group_c": peak_t_group,
        "metar_peak_cycle_utc": peak_cycle,
        "wunderground_high_c": wunderground_c,
        "winning_range_low":   wl,
        "winning_range_high":  wh,
        "winning_bin_label":   winning_label,
        "captured_at_utc":     datetime.now(timezone.utc).isoformat(),
        "capture_notes":       notes,
    }


def already_captured(conn, event_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM resolution_observations WHERE event_id = ? LIMIT 1",
        (event_id,),
    ).fetchone()
    return row is not None


def insert_observation(conn, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO resolution_observations "
        f"({','.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()


def run_capture(signals_db: str, collector_db: str,
                  backfill_days: int | None, fetch_wunderground: bool) -> int:
    if not os.path.exists(collector_db):
        log.error(f"collector DB not found: {collector_db}")
        return 1
    if not os.path.exists(signals_db):
        log.error(f"signals DB not found: {signals_db}")
        return 1

    sconn = sqlite3.connect(signals_db, timeout=30.0)
    sconn.row_factory = sqlite3.Row
    cconn = sqlite3.connect(collector_db, timeout=30.0)
    cconn.row_factory = sqlite3.Row

    # Resolved events to consider
    if backfill_days is not None:
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=backfill_days)).isoformat()
        rs_rows = cconn.execute(
            "SELECT event_id FROM resolutions WHERE date >= ? "
            "ORDER BY date DESC",
            (cutoff,),
        ).fetchall()
    else:
        rs_rows = cconn.execute(
            "SELECT event_id FROM resolutions ORDER BY resolved_at DESC LIMIT 50"
        ).fetchall()

    captured = 0
    skipped = 0
    failed = 0
    for r in rs_rows:
        event_id = r["event_id"]
        if already_captured(sconn, event_id):
            skipped += 1
            continue
        try:
            obs = capture_one(sconn, cconn, event_id, fetch_wunderground)
            insert_observation(sconn, obs)
            captured += 1
            if obs.get("capture_notes"):
                log.info(f"  {event_id[:16]}…: captured with notes: "
                         f"{obs['capture_notes']}")
        except Exception as e:
            log.warning(f"  {event_id[:16]}…: capture failed: {e}")
            failed += 1

    log.info(f"capture complete: {captured} new, {skipped} already had, "
              f"{failed} failed")
    sconn.close()
    cconn.close()
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--signals-db", default=None,
                    help="signals DB (default: config.DB_PATH)")
    p.add_argument("--collector-db", default=DEFAULT_COLLECTOR_DB,
                    help=f"resolutions DB (default: {DEFAULT_COLLECTOR_DB})")
    p.add_argument("--backfill-days", type=int, default=None,
                    help="backfill captures for the last N days "
                         "(default: most-recent 50 resolutions)")
    p.add_argument("--no-wunderground", action="store_true",
                    help="skip Wunderground fetching (still capture METAR "
                         "decomposition columns)")
    args = p.parse_args()

    return run_capture(
        signals_db        = args.signals_db or DB_PATH,
        collector_db      = args.collector_db,
        backfill_days     = args.backfill_days,
        fetch_wunderground= not args.no_wunderground,
    )


if __name__ == "__main__":
    sys.exit(main())