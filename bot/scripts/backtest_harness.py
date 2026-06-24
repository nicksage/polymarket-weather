"""
backtest_harness.py — Phase 0 backtest infrastructure.

Provides the resolved-event harness + hourly observations as truth, with
a pluggable Method interface so downstream scripts can A/B prediction
methods against ground truth.

Currently used by:
  - scripts/backtest_rounding.py — Phase 1, rounding-convention test
Future:
  - scripts/backtest_methods.py — Phase 3, copula-MC vs current fusion

Architecture:
  1. discover_resolved_events()    — pull (city, event_date, winning bin)
                                     for events whose markets have settled
                                     (any bin has market_prob >= 0.99 on
                                     the latest scan)
  2. load_event_full_bins()        — expand each event to its complete bin
                                     set (winning + losing) so per-bin
                                     probability methods can be scored
  3. load_hourly_temps()           — read raw_metar_log for the (icao,
                                     event_date), filter to calendar-day
                                     hours in station-local time
  4. Method ABC + score_method()   — pluggable: each method returns a
                                     {bin_label: probability} dict; we
                                     score against the winning bin

DATA SOURCES
============
TRUTH:    bot's own raw_metar_log (METAR captures, ~weeks of history)
RESOLVED: paper_predictor_signals + winning bin from latest scan

NO TWC dependency for the rounding test.  Future method-comparison
backtests that need TWC forecasts will use persisted scan-time snapshots
(once Phase B of the TWC integration ships).
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from config import DB_PATH                # type: ignore
from station_meta import CITY_STATIONS    # type: ignore


# ============================================================
# Data shapes
# ============================================================

@dataclass
class Bin:
    label: str
    range_low: float
    range_high: float
    unit: str   # 'fahrenheit' | 'celsius'


@dataclass
class HourlyTemp:
    timestamp_utc: str
    timestamp_local: datetime
    temp_c: float
    temp_precision: str   # 'tenths' | 'whole' | other


@dataclass
class Event:
    event_id: str
    city: str
    event_date: str
    icao: str
    tz_str: str
    settlement_unit: str
    bins: list[Bin]
    winning_bin: Bin

    # Lazy-loaded
    hourly_temps: list[HourlyTemp] = field(default_factory=list)

    @property
    def actual_max_c(self) -> Optional[float]:
        if not self.hourly_temps:
            return None
        return max(h.temp_c for h in self.hourly_temps)

    @property
    def actual_max_in_settlement_unit(self) -> Optional[float]:
        m = self.actual_max_c
        if m is None:
            return None
        if (self.settlement_unit or "").lower() == "fahrenheit":
            return m * 9.0 / 5.0 + 32.0
        return m


# ============================================================
# Event discovery
# ============================================================

def discover_resolved_events(
    conn: sqlite3.Connection,
    *, days_back: Optional[int] = 60,
    city_filter: Optional[str] = None,
) -> list[dict]:
    """Find events that have a clearly-resolved winner.

    Definition of resolved: at least one bin has market_prob >= 0.99 on
    the latest scan for that (city, event_date).  This is the same
    heuristic the dashboard + the TWC capture script use.

    Returns list of dicts with: event_id, city, event_date, settlement_unit,
    winning_bin (Bin), full_bins (list[Bin]).
    """
    # Bindings used ONLY inside the latest_scan CTE.  The outer SELECT
    # joins back via city + event_date + max_ts so no extra filters
    # need binding there.
    where_date = "AND event_date >= date('now', ?)" if days_back is not None else ""
    where_city = "AND city = ?" if city_filter else ""
    cte_args: list = []
    if days_back is not None:
        cte_args.append(f"-{int(days_back)} days")
    if city_filter:
        cte_args.append(city_filter)

    winners_sql = f"""
    WITH latest_scan AS (
        SELECT city, event_date, MAX(scanned_at_utc) AS max_ts
        FROM paper_predictor_signals
        WHERE event_date IS NOT NULL
          {where_date}
          {where_city}
        GROUP BY city, event_date
    )
    SELECT s.event_id, s.city, s.event_date,
           s.bin_range_low AS win_lo, s.bin_range_high AS win_hi,
           s.bin_label AS win_label, s.unit AS settlement_unit
    FROM paper_predictor_signals s
    JOIN latest_scan ls
      ON ls.city = s.city AND ls.event_date = s.event_date
     AND ls.max_ts = s.scanned_at_utc
    WHERE s.market_prob >= 0.99
      AND s.event_id IS NOT NULL
      -- Skip open-ended bins ('>=X' / '<=X' style).  The rounding test
      -- is about half-degree boundaries between closed bins; open-ended
      -- winners are a different question handled separately.
      AND s.bin_range_low IS NOT NULL
      AND s.bin_range_high IS NOT NULL
    ORDER BY s.event_date DESC, s.city ASC
    """
    winner_rows = conn.execute(winners_sql, cte_args).fetchall()

    events: list[dict] = []
    for r in winner_rows:
        event_id = r["event_id"]
        city = r["city"]
        event_date = r["event_date"]
        # Step 2: pull ALL bins for this (city, event_date) from the
        # same latest scan so methods can score per-bin probabilities
        full = conn.execute("""
            WITH ls AS (
                SELECT MAX(scanned_at_utc) AS max_ts
                FROM paper_predictor_signals
                WHERE city = ? AND event_date = ?
            )
            SELECT bin_label, bin_range_low, bin_range_high, unit
            FROM paper_predictor_signals
            WHERE city = ? AND event_date = ?
              AND scanned_at_utc = (SELECT max_ts FROM ls)
              AND bin_range_low IS NOT NULL
              AND bin_range_high IS NOT NULL
            ORDER BY bin_range_low ASC
        """, (city, event_date, city, event_date)).fetchall()

        if not full:
            continue

        bins = [Bin(label=b["bin_label"],
                    range_low=float(b["bin_range_low"]),
                    range_high=float(b["bin_range_high"]),
                    unit=str(b["unit"] or "fahrenheit").lower())
                for b in full]
        winning_bin = Bin(label=r["win_label"],
                          range_low=float(r["win_lo"]),
                          range_high=float(r["win_hi"]),
                          unit=str(r["settlement_unit"] or "fahrenheit").lower())
        events.append({
            "event_id":         event_id,
            "city":             city,
            "event_date":       event_date,
            "settlement_unit":  str(r["settlement_unit"] or "fahrenheit").lower(),
            "bins":             bins,
            "winning_bin":      winning_bin,
        })
    return events


# ============================================================
# Hourly observations (from raw_metar_log)
# ============================================================

def load_hourly_temps(
    conn: sqlite3.Connection, icao: str, event_date: str, tz_str: str,
) -> list[HourlyTemp]:
    """Load raw METAR observations for (icao, event_date) and filter to
    calendar-day hours in the station's local timezone.

    Filters on BOTH raw_metar_log.event_date (the bot's per-station
    labeling) AND on cycle_timestamp_utc-converted-to-local (defensive
    against UTC vs local-date edge cases at midnight)."""
    try:
        rows = conn.execute(
            """SELECT cycle_timestamp_utc, temp_c, temp_precision
               FROM raw_metar_log
               WHERE icao = ? AND event_date = ?
                 AND temp_c IS NOT NULL
               ORDER BY cycle_timestamp_utc ASC""",
            (icao, event_date),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    tz = ZoneInfo(tz_str)
    target_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    out: list[HourlyTemp] = []
    for r in rows:
        ts_str = r["cycle_timestamp_utc"]
        try:
            ts_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        ts_local = ts_utc.astimezone(tz)
        if ts_local.date() != target_date:
            continue
        out.append(HourlyTemp(
            timestamp_utc=ts_str,
            timestamp_local=ts_local,
            temp_c=float(r["temp_c"]),
            temp_precision=str(r["temp_precision"] or "whole"),
        ))
    return out


def hydrate_event(conn: sqlite3.Connection, event_meta: dict) -> Optional[Event]:
    """Build a full Event with hourly_temps loaded.  Returns None if no
    hourly data is available (event is unscoreable)."""
    city = event_meta["city"]
    meta = CITY_STATIONS.get(city)
    if not meta:
        return None
    icao, _net, tz_str = meta[0], meta[1], meta[2]

    hourly = load_hourly_temps(conn, icao, event_meta["event_date"], tz_str)
    if not hourly:
        return None

    return Event(
        event_id=        event_meta["event_id"],
        city=            city,
        event_date=      event_meta["event_date"],
        icao=            icao,
        tz_str=          tz_str,
        settlement_unit= event_meta["settlement_unit"],
        bins=            event_meta["bins"],
        winning_bin=     event_meta["winning_bin"],
        hourly_temps=    hourly,
    )


# ============================================================
# Method interface — pluggable predictors
# ============================================================

class Method:
    """Subclass and implement predict()."""
    name: str = "abstract"

    def predict(self, event: Event) -> dict[str, float]:
        """Return {bin_label: probability}.  Should sum to ~1.0 across
        the event's bin set."""
        raise NotImplementedError


# ============================================================
# Scoring
# ============================================================

def brier_score(predicted: dict[str, float], winning_label: str,
                  all_labels: list[str]) -> float:
    """Multi-class Brier = sum over bins of (predicted_p - actual_indicator)^2
    where actual_indicator = 1 for winning bin, 0 otherwise.

    Range: 0 (perfect — all mass on winner) to 2 (worst — all mass on
    a single wrong bin)."""
    total = 0.0
    for label in all_labels:
        p = predicted.get(label, 0.0)
        actual = 1.0 if label == winning_label else 0.0
        total += (p - actual) ** 2
    return total


def log_loss(predicted: dict[str, float], winning_label: str) -> float:
    """Cross-entropy log loss: -log(p_for_winning_bin).
    Returns float('inf') if the winning bin had zero predicted probability."""
    p = predicted.get(winning_label, 0.0)
    if p <= 0:
        return float("inf")
    return -math.log(p)


def top_bin_correct(predicted: dict[str, float], winning_label: str) -> bool:
    """Was the bin with highest predicted P the winning bin?
    Ties are NOT counted as correct (strict argmax)."""
    if not predicted:
        return False
    top_p = max(predicted.values())
    top_labels = [lbl for lbl, p in predicted.items() if p == top_p]
    return len(top_labels) == 1 and top_labels[0] == winning_label


def score_method_against_events(
    method: Method, events: list[Event],
) -> dict:
    """Run method on every event, compute aggregate scores.

    Returns dict with:
      method, n_events, mean_brier, mean_log_loss (excluding inf),
      n_log_loss_inf (count where p=0 for winner),
      top_correct_rate, per_event (list of detailed records)."""
    records: list[dict] = []
    briers: list[float] = []
    log_losses_finite: list[float] = []
    n_log_loss_inf = 0
    top_correct_count = 0

    for ev in events:
        try:
            pred = method.predict(ev)
        except Exception as e:
            records.append({"event": ev, "error": str(e)})
            continue
        # Normalize: ensure probs are clamped non-negative (defensive)
        pred = {k: max(0.0, float(v)) for k, v in pred.items()}

        all_labels = [b.label for b in ev.bins]
        b = brier_score(pred, ev.winning_bin.label, all_labels)
        ll = log_loss(pred, ev.winning_bin.label)
        tc = top_bin_correct(pred, ev.winning_bin.label)

        briers.append(b)
        if ll == float("inf"):
            n_log_loss_inf += 1
        else:
            log_losses_finite.append(ll)
        if tc:
            top_correct_count += 1
        records.append({
            "event":            ev,
            "predicted":        pred,
            "brier":            b,
            "log_loss":         ll,
            "top_correct":      tc,
            "actual_max_in_unit": ev.actual_max_in_settlement_unit,
        })

    n = len(events)
    return {
        "method":               method.name,
        "n_events":             n,
        "mean_brier":           (sum(briers) / n) if n else None,
        "mean_log_loss_finite": (sum(log_losses_finite) / len(log_losses_finite))
                                  if log_losses_finite else None,
        "n_log_loss_inf":       n_log_loss_inf,
        "top_correct_rate":     (top_correct_count / n) if n else None,
        "per_event":            records,
    }


# ============================================================
# Helpers
# ============================================================

def assign_bin_for_integer(value: int, bins: list[Bin]) -> Optional[Bin]:
    """For a deterministic integer prediction, find the bin whose
    [range_low, range_high] interval contains it.  Returns None if no
    bin matches (rare — usually only if the value is below the lowest
    bin or above the highest)."""
    for b in bins:
        # Open-ended bins (range_low == range_high) are single-degree bins
        # (e.g. '94°F or above' would normally have None edges but we
        # treat the typical case where both edges are real integers).
        if value >= b.range_low and value <= b.range_high:
            return b
    return None


def load_resolved_events(
    db_path: str,
    *, days_back: Optional[int] = 60,
    city_filter: Optional[str] = None,
) -> list[Event]:
    """Top-level convenience: open DB, discover resolved events, hydrate
    them with hourly temps, return only events that successfully hydrate
    (i.e., have raw_metar_log data covering their event_date)."""
    if not os.path.exists(db_path):
        return []
    out: list[Event] = []
    n_skip_no_hourly = 0
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        metas = discover_resolved_events(
            conn, days_back=days_back, city_filter=city_filter)
        for m in metas:
            ev = hydrate_event(conn, m)
            if ev is None:
                n_skip_no_hourly += 1
                continue
            out.append(ev)
    out.sort(key=lambda e: (e.event_date, e.city), reverse=True)
    return out