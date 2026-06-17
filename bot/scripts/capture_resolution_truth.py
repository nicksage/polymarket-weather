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

DATA SOURCE
-----------
The script discovers resolved events from `paper_predictor_signals`
(default `--source=signals`).  An event is considered resolved if any
bin on its LATEST scan has market_prob >= 0.99 — the same heuristic
the dashboard uses to surface the winner.  This removes the legacy
dependency on the standalone backtest-collector DB; the script reads
only from the bot's own signals.db.

The legacy `--source=collector` path is still available for callers
that still run the backtest-collector — it reads event metadata from
a `resolutions` table in a separate DB.  Use it only if you have
that collector populated.

Run nightly via cron:
    0 4 * * *  /path/to/venv/bin/python -m scripts.capture_resolution_truth

Or one-shot for backfill:
    cd bot && python -m scripts.capture_resolution_truth --backfill-days 30

To refill rows that were written with NULL truth columns (e.g.,
after fixing a broken city-name lookup or restoring a dead source):
    python -m scripts.capture_resolution_truth --backfill-days 30 --force-refill-nulls
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

def capture_one(conn, res_meta: dict, fetch_wunderground: bool) -> dict:
    """Capture all five reference values for one resolved market.
    Returns a dict suitable for insertion into resolution_observations.

    res_meta is the event metadata produced by either the legacy
    collector path or the newer discover_resolved_events_from_signals().
    Required keys: event_id, city, event_date, winning_range_low,
    winning_range_high.  Optional: winning_contract_id.
    """
    event_id   = res_meta["event_id"]
    city       = res_meta["city"]
    event_date = res_meta["event_date"]
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
    wl = res_meta.get("winning_range_low")
    wh = res_meta.get("winning_range_high")
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


def already_captured(conn, event_id: str,
                        *, only_if_complete: bool = False) -> bool:
    """Has this event_id been captured?

    only_if_complete=False (default): row exists at all.
    only_if_complete=True:  row exists AND has non-NULL truth columns.
       Used by --force-refresh / --force-refill-nulls so we re-fetch
       rows that were inserted with NULL truth columns by a prior run
       where the source fetches all returned None (e.g., Wunderground
       JS-rendered breakage, missing METAR rows, format-mismatched
       event_date).
    """
    if only_if_complete:
        row = conn.execute(
            "SELECT 1 FROM resolution_observations WHERE event_id = ? "
            "AND wunderground_high_c IS NOT NULL "
            "AND metar_peak_t_group_c IS NOT NULL "
            "AND bot_observed_max_c IS NOT NULL "
            "LIMIT 1",
            (event_id,),
        ).fetchone()
    else:
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


def discover_resolved_events_from_signals(
    conn, backfill_days: int | None,
) -> list[dict]:
    """Discover events the bot itself observed resolving.

    Reads from paper_predictor_signals: an event is "resolved" if SOME
    bin in its latest scan has market_prob >= 0.99 (the same heuristic
    the dashboard uses to label the winner).  This removes the dead-
    collector dependency the capture script used to have — every
    (city, event_date) the bot scanned is discoverable here.

    Returns list of dicts with the same shape capture_one expects:
        {event_id, city, event_date, winning_range_low,
         winning_range_high, winning_contract_id}
    """
    if backfill_days is not None:
        date_clause = "AND event_date >= date('now', ?)"
        date_arg: tuple = (f"-{int(backfill_days)} days",)
    else:
        date_clause = ""
        date_arg = ()

    # Latest scan per (city, event_date), then the bin(s) on that scan
    # with market_prob >= 0.99.  Most events resolve to one bin; the
    # GROUP BY collapses the rare multi-winner case (shouldn't happen
    # on properly-formed markets but doesn't hurt to be defensive).
    rows = conn.execute(
        f"""
        WITH latest_scan AS (
            SELECT city, event_date, MAX(scanned_at_utc) AS max_ts
            FROM paper_predictor_signals
            WHERE event_date IS NOT NULL
              {date_clause}
            GROUP BY city, event_date
        )
        SELECT s.event_id, s.city, s.event_date,
               s.contract_id  AS winning_contract_id,
               s.bin_range_low  AS winning_range_low,
               s.bin_range_high AS winning_range_high,
               s.unit           AS winning_unit
        FROM paper_predictor_signals s
        JOIN latest_scan ls
          ON ls.city = s.city AND ls.event_date = s.event_date
         AND ls.max_ts = s.scanned_at_utc
        WHERE s.market_prob >= 0.99
          AND s.event_id IS NOT NULL
        ORDER BY s.event_date DESC
        """,
        date_arg,
    ).fetchall()

    return [dict(r) for r in rows]


def run_capture(signals_db: str, collector_db: str,
                  backfill_days: int | None, fetch_wunderground: bool,
                  *, force_refill_nulls: bool = False,
                  source: str = "signals") -> int:
    # Only signals DB is strictly required.  Collector DB is checked
    # later inside the source='collector' branch.
    if not os.path.exists(signals_db):
        log.error(f"signals DB not found: {signals_db}")
        return 1

    sconn = sqlite3.connect(signals_db, timeout=30.0)
    sconn.row_factory = sqlite3.Row

    # Discover resolved events to consider.  Default 'signals' source
    # uses paper_predictor_signals (the bot's own resolution signal).
    # Legacy 'collector' source reads from the standalone backtest-
    # collector DB; kept available for users who still run that
    # collector, but the script no longer requires it.
    res_metas: list[dict] = []
    if source == "signals":
        res_metas = discover_resolved_events_from_signals(
            sconn, backfill_days=backfill_days)
        log.info(f"signals source: discovered {len(res_metas)} resolved events")
    elif source == "collector":
        if not os.path.exists(collector_db):
            log.error(f"collector DB not found: {collector_db}")
            sconn.close()
            return 1
        cconn = sqlite3.connect(collector_db, timeout=30.0)
        cconn.row_factory = sqlite3.Row
        if backfill_days is not None:
            from datetime import date, timedelta
            cutoff = (date.today() - timedelta(days=backfill_days)).isoformat()
            rs_rows = cconn.execute(
                "SELECT event_id, city, date AS event_date, "
                "       winning_contract_id, winning_range_low, "
                "       winning_range_high "
                "FROM resolutions WHERE date >= ? "
                "ORDER BY date DESC",
                (cutoff,),
            ).fetchall()
        else:
            rs_rows = cconn.execute(
                "SELECT event_id, city, date AS event_date, "
                "       winning_contract_id, winning_range_low, "
                "       winning_range_high "
                "FROM resolutions ORDER BY resolved_at DESC LIMIT 50"
            ).fetchall()
        res_metas = [dict(r) for r in rs_rows]
        cconn.close()
        log.info(f"collector source: read {len(res_metas)} resolved events")
    else:
        log.error(f"unknown source: {source!r} (expected 'signals' or 'collector')")
        sconn.close()
        return 1

    captured = 0
    skipped = 0
    failed = 0
    for rm in res_metas:
        event_id = rm.get("event_id")
        if not event_id:
            continue
        if already_captured(sconn, event_id,
                              only_if_complete=force_refill_nulls):
            skipped += 1
            continue
        try:
            obs = capture_one(sconn, rm, fetch_wunderground)
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
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--signals-db", default=None,
                    help="signals DB (default: config.DB_PATH)")
    p.add_argument("--source", choices=["signals", "collector"], default="signals",
                    help="where to discover resolved events.  'signals' (default, "
                         "recommended) reads from paper_predictor_signals — uses "
                         "the bot's own resolution signal (market_prob >= 0.99 on "
                         "latest scan).  'collector' reads from the legacy "
                         "backtest-collector DB (kept for backwards compatibility "
                         "but no longer required).")
    p.add_argument("--collector-db", default=DEFAULT_COLLECTOR_DB,
                    help=f"legacy collector DB path, only used with "
                         f"--source=collector (default: {DEFAULT_COLLECTOR_DB})")
    p.add_argument("--backfill-days", type=int, default=None,
                    help="backfill captures for the last N days "
                         "(default: most-recent 50 resolutions)")
    p.add_argument("--no-wunderground", action="store_true",
                    help="skip Wunderground fetching (still capture METAR "
                         "decomposition columns)")
    p.add_argument("--force-refill-nulls", action="store_true",
                    help="re-fetch any event whose existing row has NULL "
                         "in any of the truth columns (wunderground_high_c, "
                         "metar_peak_t_group_c, bot_observed_max_c).  "
                         "Use to recover from silent-NULL writes after the "
                         "source fetches start working again.")
    args = p.parse_args()

    return run_capture(
        signals_db        = args.signals_db or DB_PATH,
        collector_db      = args.collector_db,
        backfill_days     = args.backfill_days,
        fetch_wunderground= not args.no_wunderground,
        force_refill_nulls= args.force_refill_nulls,
        source            = args.source,
    )


if __name__ == "__main__":
    sys.exit(main())