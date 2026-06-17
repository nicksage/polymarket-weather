#!/usr/bin/env python3
"""
twc_settlement_audit.py — Phase 1 of TWC API integration.

GOAL
----
For every resolved Polymarket event the bot scanned, ask TWC's two
candidate endpoints what the day's max temperature was, round each
result into the settlement unit (Polymarket's half-up convention),
assign it to a bin, and compare to the bin Polymarket actually
settled on.  Decides whether either TWC endpoint can replace the
dead Wunderground scraper as the ground-truth source.

PURE MEASUREMENT — no live-trading code reads `twc_settlement_audit`.

THE TWO CANDIDATES (per the TWC API spec)
-----------------------------------------
  1. Historical Conditions - Daily Summary
        endpoint:  /v3/wx/conditions/historical/dailysummary
        what:      CoD-blended daily max (surface obs + radar + sat + model)
        caveat:    7am-7am-local day window, NOT calendar day
        caveat:    4km grid blend, not raw station — may diverge from
                    what Polymarket settles against

  2. Site-Based Observations (Historical)
        endpoint:  /v3/wx/observations/historical/sitebased
        what:      Raw METAR/SYNOP/BUOY/CMAN station readings
        caveat:    we aggregate to daily max ourselves (calendar day,
                    settlement-station local TZ)
        expected:  closer to Wunderground's display value (which
                    Polymarket tracks)

USAGE
-----
Safety-first defaults so the first run can't burn TWC trial credits:

    # Just show what WOULD be audited; no API calls.
    python bot/scripts/twc_settlement_audit.py --dry-run

    # First real run: cap at 5 events, slow rate.
    python bot/scripts/twc_settlement_audit.py --limit 5

    # Once first run validates field paths, larger batch:
    python bot/scripts/twc_settlement_audit.py --limit 50

    # Full backfill once happy:
    python bot/scripts/twc_settlement_audit.py --backfill-all

    # Read-only report from existing audit rows; no API calls.
    python bot/scripts/twc_settlement_audit.py --report-only

REQUIRES
--------
    TWC_API_KEY  (env var) — TWC v3 developer API key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Optional

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

from config import DB_PATH                # type: ignore
from station_meta import CITY_STATIONS    # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("twc_settlement_audit")

# CRITICAL: httpx's INFO-level logger prints the full request URL on
# every call, which INCLUDES the apiKey query parameter.  Silence it
# so the key cannot leak into shell history / log files / pasted
# terminal output.  Our own _twc_get() prints a redacted URL on demand.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================
# TWC API conventions
# These are best-guess based on TWC v3 docs.  The first real call
# will surface any field-path mismatch via explicit logging — the
# code intentionally does NOT swallow KeyError / response shape
# errors.  Fix the constant + parse paths here when first-run shows
# the actual response shape.
# ============================================================

TWC_API_BASE = os.getenv("TWC_API_BASE", "https://api.weather.com")
TWC_API_KEY  = os.getenv("TWC_API_KEY", "")

# Daily Summary 30-day endpoint per the TWC docs the operator pasted
# 2026-06-17.  Returns ONE response per ICAO covering the most-recent
# 30 days — we batch by station to make ~11 calls cover ~95 events
# instead of 95 individual calls.
TWC_DAILY_SUMMARY_PATH = os.getenv(
    "TWC_DAILY_SUMMARY_PATH",
    "/v3/wx/conditions/historical/dailysummary/30day")
# Site-based historical observations.  As of 2026-06-17 the operator's
# TWC package (Weather Company Data - New Standard Weather APIs) does
# not include this endpoint — 403 "apikey is not authorized for this
# product".  Default OFF; --enable-sitebased re-attempts.
TWC_SITEBASED_PATH = os.getenv(
    "TWC_SITEBASED_PATH",
    "/v3/wx/observations/historical/sitebased")
TWC_SITEBASED_DEFAULT_ENABLED = bool(int(
    os.getenv("TWC_SITEBASED_ENABLED", "0")))

# Default 'language' param is REQUIRED per the docs.  Keeping a
# constant so we set it consistently on every call.
TWC_LANGUAGE = os.getenv("TWC_LANGUAGE", "en-US")

# Per-call HTTP timeout.  TWC historical endpoints can be slow on
# first call after a cold cache.
TWC_TIMEOUT_S = float(os.getenv("TWC_TIMEOUT_S", "30"))


# ============================================================
# Rounding + bin assignment (Polymarket half-up convention)
# ============================================================

def _round_half_up_int(x: float) -> int:
    """Polymarket half-up rounding: 92.5 → 93, 92.49 → 92."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def assign_bin_in_settlement_unit(
    temp_in_settlement_unit: float,
    winning_low: float, winning_high: float,
) -> tuple[float, float, bool]:
    """Round the temp half-up to a whole settlement unit, then return
    (bin_low, bin_high, matches_winner).  For Polymarket US 2°F bins,
    winning_low=94 winning_high=95 means the bin covers settlement
    values where rounded F ∈ {94, 95}.  A reading rounding to 94 OR
    95 matches the winner."""
    rounded = _round_half_up_int(temp_in_settlement_unit)
    matches = (rounded >= winning_low) and (rounded <= winning_high)
    # For the audit row, store the bin we assigned the reading to.
    # Polymarket bins are 2-wide on US markets; we record the rounded
    # value as bin_low and the same+1 as bin_high so the audit table
    # makes sense whether the underlying bin is 1°-wide (intl) or
    # 2°-wide (US).  The match column is what matters for the verdict.
    return float(rounded), float(rounded), matches


def _ensure_settlement_unit_value(
    value: float, returned_unit: str, settlement_unit: str,
) -> Optional[float]:
    """Convert a TWC value to the settlement unit if needed.
    returned_unit: 'F' or 'C' (what TWC sent back).
    settlement_unit: 'fahrenheit' or 'celsius' (what Polymarket settles in).
    """
    if value is None:
        return None
    ru = (returned_unit or "").upper()
    su = (settlement_unit or "").lower()
    want_f = (su == "fahrenheit")
    if want_f and ru == "F":
        return float(value)
    if want_f and ru == "C":
        return float(value) * 9.0 / 5.0 + 32.0
    if not want_f and ru == "C":
        return float(value)
    if not want_f and ru == "F":
        return (float(value) - 32.0) * 5.0 / 9.0
    return float(value)   # unknown unit, assume already correct


# ============================================================
# TWC HTTP calls
# Each returns (parsed_data: dict, raw_response_for_debugging: str).
# On first run, if the response shape doesn't match what we expect,
# the parsing will raise a clear KeyError — fix the path constants
# at the top of the function and re-run.
# ============================================================

def _twc_units_for(settlement_unit: str) -> str:
    """TWC's units param: 'e' = English (°F), 'm' = Metric (°C)."""
    return "e" if (settlement_unit or "").lower() == "fahrenheit" else "m"


def _twc_get(path: str, params: dict, *, dry_run: bool = False) -> dict:
    """Centralized TWC GET.  Always includes the required `language`,
    `format`, `apiKey` params.  Never logs the key.  Surfaces 4xx/5xx
    bodies so first-run failures are easy to diagnose."""
    if not TWC_API_KEY and not dry_run:
        raise RuntimeError(
            "TWC_API_KEY env var is empty — set it in .env and re-run.")
    full = {**params,
            "language": TWC_LANGUAGE,
            "format":   "json",
            "apiKey":   TWC_API_KEY}
    url = f"{TWC_API_BASE}{path}"
    if dry_run:
        return {"_dry_run": True}
    resp = httpx.get(url, params=full, timeout=TWC_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"TWC HTTP {resp.status_code} on {url}: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"TWC returned non-JSON: {resp.text[:500]}") from e


# Per-station cache for the 30-day daily-summary call so we only hit
# the endpoint ONCE per ICAO regardless of how many events we audit.
# Keyed by (icao, units).  Cleared per script invocation.
_DAILY_SUMMARY_CACHE: dict = {}


def _twc_fetch_30day_block(
    icao: str, settlement_unit: str, *, dry_run: bool = False,
) -> tuple[Optional[dict], str]:
    """One call per ICAO: fetch the full 30-day daily-summary block.
    Returns (block_dict, notes) where block_dict has the parsed
    'temperatureMax' / 'validTimeLocal' arrays the date lookup uses.
    Cached so multiple events on the same station share the call."""
    key = (icao, settlement_unit)
    if key in _DAILY_SUMMARY_CACHE:
        return _DAILY_SUMMARY_CACHE[key]
    try:
        data = _twc_get(TWC_DAILY_SUMMARY_PATH, {
            "icaoCode": icao,
            "units":    _twc_units_for(settlement_unit),
        }, dry_run=dry_run)
    except Exception as e:
        _DAILY_SUMMARY_CACHE[key] = (None, f"http_error: {e}")
        return _DAILY_SUMMARY_CACHE[key]
    if dry_run:
        _DAILY_SUMMARY_CACHE[key] = (None, "dry_run")
        return _DAILY_SUMMARY_CACHE[key]
    # Response shape from the TWC docs:
    #   [{"id": "...", "v3-wx-conditions-historical-dailysummary-30day": {
    #       "temperatureMax": [86, ...], "validTimeLocal": [...], ...}}]
    try:
        if isinstance(data, list) and data:
            wrapper = data[0]
        elif isinstance(data, dict):
            wrapper = data
        else:
            _DAILY_SUMMARY_CACHE[key] = (
                None, f"unexpected_response_type: {type(data).__name__}")
            return _DAILY_SUMMARY_CACHE[key]
        # The nested key matches the product name.  Find it by suffix
        # to avoid hard-coding the full string in case TWC versions it.
        block = None
        for k, v in wrapper.items():
            if k == "id":
                continue
            if isinstance(v, dict) and "temperatureMax" in v:
                block = v
                break
        if block is None:
            _DAILY_SUMMARY_CACHE[key] = (
                None,
                f"no_temperatureMax_block (wrapper_keys={list(wrapper.keys())})")
            return _DAILY_SUMMARY_CACHE[key]
        _DAILY_SUMMARY_CACHE[key] = (block, "")
        return _DAILY_SUMMARY_CACHE[key]
    except Exception as e:
        _DAILY_SUMMARY_CACHE[key] = (None, f"parse_error: {e}")
        return _DAILY_SUMMARY_CACHE[key]


def twc_daily_summary_max(
    icao: str, date_iso: str, settlement_unit: str,
    *, dry_run: bool = False,
) -> tuple[Optional[float], str, str, str]:
    """Look up the day's max from the cached 30-day block for this
    (icao, units).  First call per ICAO triggers the actual HTTP fetch;
    subsequent events on the same station read from cache.

    Returns (max_temp_in_returned_unit, returned_unit, day_window, notes).

    Day-window note is always '7am-7am-local' per the TWC docs —
    this is THE critical caveat the audit spec told us to record per
    event, because the 7am-7am window can clip a daily-max that
    occurs near 7am.
    """
    day_window = "7am-7am-local"
    unit_returned = "F" if _twc_units_for(settlement_unit) == "e" else "C"

    block, block_notes = _twc_fetch_30day_block(
        icao, settlement_unit, dry_run=dry_run)
    if block is None:
        return None, unit_returned, day_window, block_notes

    # The 30-day arrays are aligned by index.  Find the day whose
    # validTimeLocal STARTS WITH our event_date (the local-time '07:00'
    # marker shows the start of TWC's 7am-7am window for that calendar day).
    times = block.get("validTimeLocal") or []
    temps = block.get("temperatureMax") or []
    if len(times) != len(temps):
        return None, unit_returned, day_window, (
            f"misaligned_arrays (times={len(times)}, temps={len(temps)})")
    target_prefix = date_iso     # 'YYYY-MM-DD'
    idx = None
    for i, t in enumerate(times):
        if isinstance(t, str) and t.startswith(target_prefix):
            idx = i
            break
    if idx is None:
        return None, unit_returned, day_window, (
            f"date_not_in_30day_block (have {len(times)} days, "
            f"first={times[0] if times else 'none'}, "
            f"last={times[-1] if times else 'none'})")
    v = temps[idx]
    if v is None:
        return None, unit_returned, day_window, "temperatureMax_null_for_date"
    return float(v), unit_returned, day_window, ""


def twc_sitebased_daily_max(
    icao: str, date_iso: str, settlement_unit: str,
    *, dry_run: bool = False,
) -> tuple[Optional[float], str, int, str]:
    """Hit /v3/wx/observations/historical/sitebased for one (icao, date),
    aggregate to daily max ourselves.

    Returns (max_temp_in_returned_unit, returned_unit, n_obs, notes).
    """
    ymd = date_iso.replace("-", "")
    try:
        data = _twc_get(TWC_SITEBASED_PATH, {
            "icaoCode":  icao,
            "startDate": ymd,
            "endDate":   ymd,
            "units":     _twc_units_for(settlement_unit),
        }, dry_run=dry_run)
    except Exception as e:
        return None, "", 0, f"http_error: {e}"

    if dry_run:
        return None, "", 0, "dry_run"

    # PARSE PATH (TWC v3 convention guess):
    #   data["observations"] = [{"temperature": 92.3, "validTimeUtc": ...}, ...]
    #   take max(observations[i]["temperature"])
    try:
        obs = (data.get("observations")
                 or data.get("siteBasedObservations")
                 or [])
        if not obs:
            return None, "", 0, f"no_observations (keys={list(data.keys())})"
        unit_returned = "F" if _twc_units_for(settlement_unit) == "e" else "C"
        temps = []
        for o in obs:
            t = (o.get("temperature")
                 if "temperature" in o else
                 o.get("temp")
                 if "temp" in o else
                 None)
            if t is not None:
                temps.append(float(t))
        if not temps:
            sample = obs[0] if obs else {}
            return None, unit_returned, len(obs), (
                f"no_temperature_field (sample_keys={list(sample.keys())[:8]})")
        return max(temps), unit_returned, len(temps), ""
    except (KeyError, IndexError, TypeError) as e:
        keys = list(data.keys()) if isinstance(data, dict) else "(not dict)"
        return None, "", 0, f"parse_error: {e} (top_keys={keys})"


# ============================================================
# Event discovery + audit-row construction
# ============================================================

_TWC_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS twc_settlement_audit (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id                    TEXT NOT NULL UNIQUE,
    city                        TEXT,
    event_date                  TEXT,
    icao                        TEXT,
    settlement_unit             TEXT,
    polymarket_winning_low      REAL,
    polymarket_winning_high     REAL,
    polymarket_winning_label    TEXT,
    twc_dailysummary_max        REAL,
    twc_dailysummary_unit       TEXT,
    twc_dailysummary_day_window TEXT,
    twc_dailysummary_bin_low    REAL,
    twc_dailysummary_bin_high   REAL,
    dailysummary_match          INTEGER,
    dailysummary_notes          TEXT,
    twc_sitebased_max           REAL,
    twc_sitebased_unit          TEXT,
    twc_sitebased_n_obs         INTEGER,
    twc_sitebased_bin_low       REAL,
    twc_sitebased_bin_high      REAL,
    sitebased_match             INTEGER,
    sitebased_notes             TEXT,
    captured_at_utc             TEXT NOT NULL,
    api_call_notes              TEXT
);
CREATE INDEX IF NOT EXISTS idx_tsa_city_date
    ON twc_settlement_audit(city, event_date);
"""


def ensure_audit_table(conn: sqlite3.Connection) -> None:
    """Create twc_settlement_audit if it doesn't already exist.
    Matches the schema in scheduled_predictor.py — kept in sync here
    so the audit script can be run standalone without first booting
    the bot."""
    conn.executescript(_TWC_AUDIT_SCHEMA)
    conn.commit()


def discover_resolved_events(
    conn: sqlite3.Connection,
    *, backfill_days: Optional[int],
    only_missing: bool = True,
) -> list[dict]:
    """Resolved events the bot scanned, joined with their settled bin.

    Same discovery query as capture_resolution_truth's signals path:
    bin with market_prob >= 0.99 on the latest scan = the winner.
    only_missing=True skips events we already audited.
    """
    if backfill_days is not None:
        date_clause = "AND s.event_date >= date('now', ?)"
        date_arg: tuple = (f"-{int(backfill_days)} days",)
    else:
        date_clause = ""
        date_arg = ()

    rows = conn.execute(
        f"""
        WITH latest_scan AS (
            SELECT city, event_date, MAX(scanned_at_utc) AS max_ts
            FROM paper_predictor_signals
            WHERE event_date IS NOT NULL
              {date_clause.replace('s.event_date', 'event_date')}
            GROUP BY city, event_date
        )
        SELECT s.event_id, s.city, s.event_date,
               s.bin_range_low  AS winning_low,
               s.bin_range_high AS winning_high,
               s.bin_label      AS winning_label,
               s.unit           AS settlement_unit
        FROM paper_predictor_signals s
        JOIN latest_scan ls
          ON ls.city = s.city AND ls.event_date = s.event_date
         AND ls.max_ts = s.scanned_at_utc
        WHERE s.market_prob >= 0.99
          AND s.event_id IS NOT NULL
          {date_clause}
        ORDER BY s.event_date DESC, s.city ASC
        """,
        date_arg + date_arg,
    ).fetchall()
    candidates = [dict(r) for r in rows]

    if only_missing:
        existing = set()
        try:
            for r in conn.execute(
                "SELECT event_id FROM twc_settlement_audit"
            ).fetchall():
                existing.add(r[0])
        except sqlite3.OperationalError:
            pass
        candidates = [c for c in candidates if c["event_id"] not in existing]

    return candidates


def attach_icao(events: list[dict]) -> list[dict]:
    """Resolve city → ICAO via CITY_STATIONS.  Drops events with no
    mapping (international cities not in station_meta) since we can't
    audit them without an ICAO."""
    out = []
    dropped = []
    for ev in events:
        meta = CITY_STATIONS.get(ev["city"])
        if meta:
            ev["icao"] = meta[0]
            out.append(ev)
        else:
            dropped.append(ev["city"])
    if dropped:
        log.info(f"dropped {len(dropped)} events with no ICAO mapping "
                 f"(cities: {sorted(set(dropped))})")
    return out


def audit_one(conn: sqlite3.Connection, ev: dict, *,
                  dry_run: bool, rate_limit_ms: int,
                  enable_sitebased: bool = False) -> dict:
    """Hit both TWC endpoints for one resolved event, compute bin
    matches, write/upsert one row in twc_settlement_audit, return
    the row dict."""
    icao = ev["icao"]
    date_iso = ev["event_date"]
    settlement_unit = (ev.get("settlement_unit") or "fahrenheit").lower()
    win_lo = ev["winning_low"]
    win_hi = ev["winning_high"]

    # --- Daily Summary candidate ---
    # Cached per (icao, units): first event per station triggers the
    # real HTTP fetch (rate-limit applied then); subsequent events on
    # the same station are free cache hits.
    cache_hit_before = (icao, settlement_unit) in _DAILY_SUMMARY_CACHE
    ds_max, ds_unit, ds_window, ds_notes = twc_daily_summary_max(
        icao, date_iso, settlement_unit, dry_run=dry_run)
    if rate_limit_ms and not dry_run and not cache_hit_before:
        time.sleep(rate_limit_ms / 1000.0)
    ds_bin_low = ds_bin_high = None
    ds_match: Optional[int] = None
    if ds_max is not None:
        v_settled = _ensure_settlement_unit_value(ds_max, ds_unit, settlement_unit)
        ds_bin_low, ds_bin_high, ds_matches = assign_bin_in_settlement_unit(
            v_settled, win_lo, win_hi)
        ds_match = 1 if ds_matches else 0

    # --- Site-Based candidate (gated; not entitled by default) ---
    sb_max = None
    sb_unit = ""
    sb_n = 0
    sb_notes = "skipped (use --enable-sitebased once entitlement is confirmed)"
    sb_bin_low = sb_bin_high = None
    sb_match: Optional[int] = None
    if enable_sitebased:
        sb_max, sb_unit, sb_n, sb_notes = twc_sitebased_daily_max(
            icao, date_iso, settlement_unit, dry_run=dry_run)
        if rate_limit_ms and not dry_run:
            time.sleep(rate_limit_ms / 1000.0)
        if sb_max is not None:
            v_settled = _ensure_settlement_unit_value(sb_max, sb_unit, settlement_unit)
            sb_bin_low, sb_bin_high, sb_matches = assign_bin_in_settlement_unit(
                v_settled, win_lo, win_hi)
            sb_match = 1 if sb_matches else 0

    row = {
        "event_id":                 ev["event_id"],
        "city":                     ev["city"],
        "event_date":               date_iso,
        "icao":                     icao,
        "settlement_unit":          settlement_unit,
        "polymarket_winning_low":   win_lo,
        "polymarket_winning_high":  win_hi,
        "polymarket_winning_label": ev.get("winning_label"),
        "twc_dailysummary_max":     ds_max,
        "twc_dailysummary_unit":    ds_unit or None,
        "twc_dailysummary_day_window": ds_window or None,
        "twc_dailysummary_bin_low": ds_bin_low,
        "twc_dailysummary_bin_high": ds_bin_high,
        "dailysummary_match":       ds_match,
        "dailysummary_notes":       ds_notes or None,
        "twc_sitebased_max":        sb_max,
        "twc_sitebased_unit":       sb_unit or None,
        "twc_sitebased_n_obs":      sb_n,
        "twc_sitebased_bin_low":    sb_bin_low,
        "twc_sitebased_bin_high":   sb_bin_high,
        "sitebased_match":          sb_match,
        "sitebased_notes":          sb_notes or None,
        "captured_at_utc":          datetime.now(timezone.utc).isoformat(),
        "api_call_notes":           None,
    }
    if not dry_run:
        cols = list(row.keys())
        ph   = ",".join(["?"] * len(cols))
        conn.execute(
            f"INSERT OR REPLACE INTO twc_settlement_audit "
            f"({','.join(cols)}) VALUES ({ph})",
            [row[c] for c in cols],
        )
        conn.commit()
    return row


# ============================================================
# Reporting
# ============================================================

def print_report(conn: sqlite3.Connection) -> None:
    """Per-candidate match rate, overall + per-city, plus disagreement list."""
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT * FROM twc_settlement_audit
            ORDER BY event_date DESC, city ASC
        """).fetchall()]
    except sqlite3.OperationalError:
        print("(no twc_settlement_audit table yet)")
        return
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM twc_settlement_audit ORDER BY event_date DESC, city ASC"
    ).fetchall()]
    if not rows:
        print("(no audit rows)")
        return

    n = len(rows)
    ds_n = sum(1 for r in rows if r["dailysummary_match"] is not None)
    sb_n = sum(1 for r in rows if r["sitebased_match"] is not None)
    ds_hit = sum(1 for r in rows if r["dailysummary_match"] == 1)
    sb_hit = sum(1 for r in rows if r["sitebased_match"] == 1)

    print()
    print("=" * 80)
    print(f"TWC settlement-truth audit  ({n} events)")
    print("=" * 80)
    def pct(num, den):
        return f"{(100.0 * num / den):.1f}%" if den else "  --"
    print(f"  Daily Summary candidate: {ds_hit}/{ds_n} match  ({pct(ds_hit, ds_n)})")
    print(f"  Site-Based candidate:    {sb_hit}/{sb_n} match  ({pct(sb_hit, sb_n)})")

    # Per-city
    print()
    print(f"{'city':<16} {'N':>4} {'ds_hit':>8} {'ds %':>7}  {'sb_hit':>8} {'sb %':>7}")
    print("-" * 70)
    by_city: dict = {}
    for r in rows:
        by_city.setdefault(r["city"], []).append(r)
    for city in sorted(by_city):
        rs = by_city[city]
        dn = sum(1 for r in rs if r["dailysummary_match"] is not None)
        sn = sum(1 for r in rs if r["sitebased_match"] is not None)
        dh = sum(1 for r in rs if r["dailysummary_match"] == 1)
        sh = sum(1 for r in rs if r["sitebased_match"] == 1)
        print(f"{city:<16} {len(rs):>4} {dh:>8} {pct(dh, dn):>7}  "
              f"{sh:>8} {pct(sh, sn):>7}")

    # Disagreements (cases where one or both candidates missed)
    misses = [r for r in rows
              if r["dailysummary_match"] == 0 or r["sitebased_match"] == 0]
    if misses:
        print()
        print(f"Disagreement cases ({len(misses)}) — eyeball these:")
        print(f"{'date':<12} {'city':<14} "
              f"{'win_lo-hi':>10} {'ds_max':>8} {'ds_bin':>7} "
              f"{'sb_max':>8} {'sb_bin':>7}")
        print("-" * 80)
        for r in misses[:50]:
            w = f"{int(r['polymarket_winning_low'])}-{int(r['polymarket_winning_high'])}"
            ds = (f"{r['twc_dailysummary_max']:.1f}"
                  if r["twc_dailysummary_max"] is not None else "--")
            db = (f"{int(r['twc_dailysummary_bin_low'])}"
                  if r["twc_dailysummary_bin_low"] is not None else "--")
            sb = (f"{r['twc_sitebased_max']:.1f}"
                  if r["twc_sitebased_max"] is not None else "--")
            sbn = (f"{int(r['twc_sitebased_bin_low'])}"
                   if r["twc_sitebased_bin_low"] is not None else "--")
            print(f"{r['event_date']:<12} {r['city']:<14} "
                  f"{w:>10} {ds:>8} {db:>7} {sb:>8} {sbn:>7}")
        if len(misses) > 50:
            print(f"... {len(misses) - 50} more truncated")

    # Failure modes — capture_notes give the parse / HTTP failures
    fails_ds = [r for r in rows if r["dailysummary_notes"]]
    fails_sb = [r for r in rows if r["sitebased_notes"]]
    if fails_ds or fails_sb:
        print()
        print("API failure notes (top 5 of each):")
        for r in fails_ds[:5]:
            print(f"  [DS  ] {r['city']} {r['event_date']}: {r['dailysummary_notes']}")
        for r in fails_sb[:5]:
            print(f"  [SB  ] {r['city']} {r['event_date']}: {r['sitebased_notes']}")


# ============================================================
# Main
# ============================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--backfill-days", type=int, default=30,
                       help="Audit events from the last N days (default: 30). "
                            "Use --backfill-all to audit every resolved event.")
    ap.add_argument("--backfill-all", action="store_true",
                       help="Audit every resolved event (overrides --backfill-days)")
    ap.add_argument("--limit", type=int, default=10,
                       help="Cap events audited THIS RUN (default: 10).  "
                            "First run after schema deploy: keep small to "
                            "burn-test the field paths without wasting "
                            "trial credits.")
    ap.add_argument("--include-existing", action="store_true",
                       help="Re-audit events already in twc_settlement_audit "
                            "(default: skip already-audited).")
    ap.add_argument("--rate-limit-ms", type=int, default=500,
                       help="Sleep between API calls (default: 500ms = 2/sec). "
                            "Only applied on cache MISSES — daily-summary is "
                            "cached per ICAO so re-using a station is free.")
    ap.add_argument("--enable-sitebased", action="store_true",
                       default=TWC_SITEBASED_DEFAULT_ENABLED,
                       help="Also hit the site-based historical observations "
                            "endpoint.  Default OFF — confirm your TWC plan "
                            "includes 'Site-Based Historical Observations' "
                            "before enabling, otherwise every call returns "
                            "403 'apikey is not authorized for this product'.")
    ap.add_argument("--dry-run", action="store_true",
                       help="List events that would be audited; no API calls.")
    ap.add_argument("--report-only", action="store_true",
                       help="Print report from existing audit rows; no API calls.")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"FATAL: signals DB not found at {args.db}", file=sys.stderr)
        return 1

    backfill_days = None if args.backfill_all else args.backfill_days
    with sqlite3.connect(args.db, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        ensure_audit_table(conn)

        if args.report_only:
            print_report(conn)
            return 0

        events = discover_resolved_events(
            conn,
            backfill_days=backfill_days,
            only_missing=not args.include_existing,
        )
        events = attach_icao(events)
        log.info(f"discovered {len(events)} events to audit "
                 f"(backfill_days={backfill_days}, limit={args.limit})")

        if not events:
            print("Nothing to audit.")
            print_report(conn)
            return 0

        events = events[: args.limit]

        if args.dry_run:
            print(f"\nDRY RUN — would audit {len(events)} events:")
            for ev in events[:20]:
                print(f"  {ev['event_date']}  {ev['city']:<14} "
                      f"icao={ev['icao']}  bin={ev.get('winning_label')}")
            if len(events) > 20:
                print(f"  ... {len(events) - 20} more")
            return 0

        if not TWC_API_KEY:
            print("FATAL: TWC_API_KEY env var is empty.  "
                  "Set it in .env and re-run.", file=sys.stderr)
            return 1

        log.info(f"starting audit; rate_limit={args.rate_limit_ms}ms on "
                 f"cache MISSES (daily-summary cached per ICAO); "
                 f"sitebased={'ON' if args.enable_sitebased else 'OFF'}")
        n_ok = n_fail = 0
        for i, ev in enumerate(events, 1):
            try:
                r = audit_one(conn, ev, dry_run=False,
                                  rate_limit_ms=args.rate_limit_ms,
                                  enable_sitebased=args.enable_sitebased)
                ds_v = r["dailysummary_match"]
                sb_v = r["sitebased_match"]
                ds_s = "MATCH" if ds_v == 1 else ("miss" if ds_v == 0 else "fail")
                sb_s = "MATCH" if sb_v == 1 else ("miss" if sb_v == 0 else "fail")
                log.info(f"  [{i:>3}/{len(events)}] {ev['city']:<14} "
                         f"{ev['event_date']}  DS:{ds_s}  SB:{sb_s}")
                n_ok += 1
            except Exception as e:
                log.error(f"  [{i:>3}/{len(events)}] {ev['city']:<14} "
                          f"{ev['event_date']}  raised: {e}")
                n_fail += 1

        log.info(f"audit complete: {n_ok} ok, {n_fail} failed")

        print_report(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())