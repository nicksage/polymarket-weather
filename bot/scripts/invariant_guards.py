"""
invariant_guards.py — Permanent observational checks.

The ratchet bug (NWS hourly evening-scan, root-caused in earlier
session) taught a specific lesson: a one-time code review wouldn't have
caught it; only plotting the time series of forecast_high_c against
scan time did.  The bug had a signature — a value that should be
monotonic-or-stable within a day silently moved the wrong direction
because an assumption about an external feed's shape was wrong.

This module installs PERMANENT INVARIANT GUARDS for every column the
predictor consumes or produces, encoded as expected within-day
monotonicity rules.  Each guard runs after every scan, logs at WARNING
on violation, and writes a row to guard_violations.

=== DESIGN RULE: OBSERVATIONAL FOREVER ===

Guards LOG, COUNT, and SURFACE.  They never GATE or REMEDIATE.

The moment a guard suppresses a trade or corrects a value, it becomes
load-bearing in the trading path — at which point it needs its own
calibration, its own failure modes, and its own guards.  That is
exactly the silent-assumption layer pattern that produced the ratchet
bug in the first place.

Architectural constraints (enforced by test_invariant_guards.py):
  * This module is never imported by anything in the prediction path
    (intraday_predictor.py, scheduled_predictor.run_intraday_scan's
    decision logic — only the guard call at the END of the scan loop)
  * No function in this module returns a value that influences the
    scan, sizing, or gate stack
  * run_invariant_checks() returns None; it produces only side-effects
    (log lines + DB writes)
  * If a guard reveals a problem worth acting on, the action belongs
    in the explicit gate stack or the data-quality contract, NOT here

Guards are a smoke detector, not a sprinkler.

=== Current invariants ===

  observed_max_monotone      observed_max_c non-decreasing within a day
  forecast_high_monotone     forecast_high_c non-decreasing (recovery
                              helper guarantees this; guard ratifies)
  cooling_confidence_monotone_post_peak
                              cooling_confidence non-decreasing once
                              forecast_peak_hour has passed; oscillation
                              here is the bin-lock-discontinuity churn
                              surfacing as data
  sigma_monotone_post_peak   sigma_c non-increasing post-peak.  Known
                              false-positive: ensemble-disagreement
                              inflation can legitimately re-widen σ;
                              guard does not exempt this branch (would
                              require persisting per-row inflation
                              status — deferred).  Operator de-noises
                              via the dashboard's "is-new" flag.
  mu_jump_incoherent_with_neighbors
                              μ jumped >=1.5°C in <1h AND upwind
                              neighbors did NOT move coherently with
                              the jump.  Real weather is spatially
                              coherent; parse errors and source-flips
                              are station-specific.  This is the
                              high-precision source-flip detector.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS  # type: ignore

log = logging.getLogger("invariant_guards")

NEIGHBOR_DB = os.path.join(_BOT_DIR, "data", "neighbor_obs.db")

# Magnitude floors / tolerances
OBS_MAX_FLOAT_TOL_C       = 0.05   # within-noise band; ignore
FORECAST_HIGH_FLOAT_TOL_C = 0.05
COOLING_CONF_TOL          = 0.05   # cooling_confidence is in [0,1]
SIGMA_FLOAT_TOL_C         = 0.10   # σ can wobble on each scan; ignore tiny moves
MU_JUMP_MAGNITUDE_FLOOR_C = 1.5    # do not flag μ moves smaller than this
MU_JUMP_TIME_WINDOW_H     = 1.0    # only check within-hour jumps
MU_NEIGHBOR_COHERENCE_TOL_C = 1.0  # |Δμ - Δnbr| above this = incoherent


@dataclass
class InvariantViolation:
    """A single violation row.  Pure data — no behavior attached."""
    scan_at_utc: str
    guard_name: str
    city: str
    event_date: str | None
    prev_value: float | None
    curr_value: float | None
    delta: float | None
    detail: str

    def to_row(self) -> dict:
        return {
            "detected_at_utc": datetime.now(timezone.utc).isoformat(),
            "scan_at_utc":     self.scan_at_utc,
            "guard_name":      self.guard_name,
            "city":            self.city,
            "event_date":      self.event_date,
            "prev_value":      self.prev_value,
            "curr_value":      self.curr_value,
            "delta":           self.delta,
            "detail":          self.detail,
        }


# ============================================================
# Helpers — read-only DB access
# ============================================================

_PER_EVENT_COLS = (
    "scanned_at_utc, observed_max_c, forecast_high_c, "
    "forecast_peak_hour, mu_c, sigma_c, cooling_confidence"
)


def _prior_scan_row(conn: sqlite3.Connection, city: str, event_date: str,
                      this_scan_at_utc: str) -> sqlite3.Row | None:
    """Most recent scan row for (city, event_date) strictly BEFORE
    this_scan_at_utc.  Returns None if no prior."""
    return conn.execute(
        f"""SELECT {_PER_EVENT_COLS}
           FROM paper_predictor_signals
           WHERE city = ? AND event_date = ?
             AND scanned_at_utc < ?
           ORDER BY scanned_at_utc DESC LIMIT 1""",
        (city, event_date, this_scan_at_utc),
    ).fetchone()


def _this_scan_row(conn: sqlite3.Connection, city: str, event_date: str,
                    scan_at_utc: str) -> sqlite3.Row | None:
    """The just-written row for this scan.  We read one bin's worth —
    they all share the per-event values (mu, sigma, observed_max, etc.)."""
    return conn.execute(
        f"""SELECT {_PER_EVENT_COLS}
           FROM paper_predictor_signals
           WHERE city = ? AND event_date = ? AND scanned_at_utc = ?
           LIMIT 1""",
        (city, event_date, scan_at_utc),
    ).fetchone()


def _neighbor_temp_delta_c(city: str, prev_scan_at_utc: str,
                             curr_scan_at_utc: str) -> float | None:
    """Average current_temp delta across the city's upwind neighbors
    between two scan times.  Returns None if neighbor data isn't
    available for this city/window.  Read-only access to neighbor_obs.db.

    For source-flip detection: if our μ jumped 3°C but the neighbors'
    avg temp moved <0.5°C in the same window, the jump is station-
    specific (parse error / source-flip) and not real weather."""
    if not os.path.exists(NEIGHBOR_DB):
        return None
    meta = CITY_STATIONS.get(city)
    if not meta:
        return None
    try:
        prev_t = datetime.fromisoformat(prev_scan_at_utc.replace("Z", "+00:00"))
        curr_t = datetime.fromisoformat(curr_scan_at_utc.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    # Local hour of each scan in the city's timezone
    from zoneinfo import ZoneInfo
    tz_str = meta[2]
    tz = ZoneInfo(tz_str)
    prev_h = prev_t.astimezone(tz).hour
    curr_h = curr_t.astimezone(tz).hour
    today_str = curr_t.astimezone(tz).date().isoformat()

    try:
        with sqlite3.connect(NEIGHBOR_DB) as ncon:
            ncon.row_factory = sqlite3.Row
            # Average temp of THIS city's neighbors (any direction) at
            # both hours.  Use any neighbor with both hours present.
            # We don't filter by wind direction here — we just want to
            # know if the regional thermal field moved.
            nbrs = ncon.execute(
                "SELECT DISTINCT sid FROM neighbor_meta "
                "WHERE polymarket_city = ?",
                (city,),
            ).fetchall()
            prev_temps, curr_temps = [], []
            for n in nbrs:
                prev_row = ncon.execute(
                    "SELECT temp_c FROM neighbor_obs WHERE sid = ? "
                    "AND date_local = ? AND hour_local = ?",
                    (n["sid"], today_str, prev_h),
                ).fetchone()
                curr_row = ncon.execute(
                    "SELECT temp_c FROM neighbor_obs WHERE sid = ? "
                    "AND date_local = ? AND hour_local = ?",
                    (n["sid"], today_str, curr_h),
                ).fetchone()
                if prev_row and curr_row and prev_row[0] is not None and curr_row[0] is not None:
                    prev_temps.append(float(prev_row[0]))
                    curr_temps.append(float(curr_row[0]))
    except sqlite3.Error:
        return None

    if not prev_temps or not curr_temps:
        return None
    return (sum(curr_temps) / len(curr_temps)) - (sum(prev_temps) / len(prev_temps))


# ============================================================
# Guard registry + decorator
# ============================================================

# Guards are pure functions of (conn, city, event_date, scan_at_utc) that
# return Optional[InvariantViolation].  Registered via @guard decorator
# so adding a new guard is one place, not three.
GuardFn = Callable[[sqlite3.Connection, str, str, str], "InvariantViolation | None"]

_GUARDS: list[tuple[str, GuardFn]] = []


def guard(name: str):
    """Decorator: register a guard function under the given name."""
    def _wrap(fn: GuardFn) -> GuardFn:
        _GUARDS.append((name, fn))
        return fn
    return _wrap


# ============================================================
# Guards
# ============================================================

@guard("observed_max_monotone")
def _check_observed_max(conn, city, event_date, scan_at_utc):
    """observed_max_c is the day's running maximum — it can only rise."""
    this_row = _this_scan_row(conn, city, event_date, scan_at_utc)
    prev_row = _prior_scan_row(conn, city, event_date, scan_at_utc)
    if not this_row or not prev_row:
        return None
    p = prev_row["observed_max_c"]
    c = this_row["observed_max_c"]
    if p is None or c is None:
        return None
    delta = c - p
    if delta < -OBS_MAX_FLOAT_TOL_C:
        return InvariantViolation(
            scan_at_utc=scan_at_utc, guard_name="observed_max_monotone",
            city=city, event_date=event_date,
            prev_value=float(p), curr_value=float(c), delta=delta,
            detail=(f"observed_max_c decreased {p:.2f}°C → {c:.2f}°C "
                    f"(Δ={delta:+.2f}°C); the day's max can only rise"),
        )
    return None


@guard("forecast_high_monotone")
def _check_forecast_high(conn, city, event_date, scan_at_utc):
    """forecast_high_c should be non-decreasing within a day — the
    recovery helper guarantees this; the guard ratifies it as an
    invariant so any regression in the helper's behavior is caught
    immediately."""
    this_row = _this_scan_row(conn, city, event_date, scan_at_utc)
    prev_row = _prior_scan_row(conn, city, event_date, scan_at_utc)
    if not this_row or not prev_row:
        return None
    p = prev_row["forecast_high_c"]
    c = this_row["forecast_high_c"]
    if p is None or c is None:
        return None
    delta = c - p
    if delta < -FORECAST_HIGH_FLOAT_TOL_C:
        return InvariantViolation(
            scan_at_utc=scan_at_utc, guard_name="forecast_high_monotone",
            city=city, event_date=event_date,
            prev_value=float(p), curr_value=float(c), delta=delta,
            detail=(f"forecast_high_c decreased {p:.2f}°C → {c:.2f}°C "
                    f"(Δ={delta:+.2f}°C); recovery helper should have "
                    "preserved the higher prior value"),
        )
    return None


@guard("sigma_monotone_post_peak")
def _check_sigma_post_peak(conn, city, event_date, scan_at_utc):
    """After forecast_peak_hour, σ should be non-increasing.  Known
    false-positive: ensemble-disagreement inflation can legitimately
    re-widen σ.  Operator de-noises via the dashboard's is-new flag —
    a city that fires this guard every day has standing ensemble
    disagreement, not a new bug."""
    this_row = _this_scan_row(conn, city, event_date, scan_at_utc)
    prev_row = _prior_scan_row(conn, city, event_date, scan_at_utc)
    if not this_row or not prev_row:
        return None
    p_sigma = prev_row["sigma_c"]
    c_sigma = this_row["sigma_c"]
    peak_h = this_row["forecast_peak_hour"]
    if p_sigma is None or c_sigma is None or peak_h is None:
        return None
    # Only fire post-peak.  Compute current scan's local hour.
    meta = CITY_STATIONS.get(city)
    if not meta:
        return None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(meta[2])
        curr_t = datetime.fromisoformat(scan_at_utc.replace("Z", "+00:00"))
        local_hour = curr_t.astimezone(tz).hour
    except (ValueError, KeyError):
        return None
    if local_hour < peak_h:
        return None
    delta = c_sigma - p_sigma
    if delta > SIGMA_FLOAT_TOL_C:
        return InvariantViolation(
            scan_at_utc=scan_at_utc, guard_name="sigma_monotone_post_peak",
            city=city, event_date=event_date,
            prev_value=float(p_sigma), curr_value=float(c_sigma), delta=delta,
            detail=(f"σ widened {p_sigma:.2f} → {c_sigma:.2f}°C post-peak "
                    f"(Δ={delta:+.2f}); may be ensemble-disagreement "
                    "inflation, may be regression"),
        )
    return None


@guard("cooling_confidence_monotone_post_peak")
def _check_cooling_confidence_post_peak(conn, city, event_date, scan_at_utc):
    """cooling_confidence is the 3-signal weighted score from
    detect_cooling().  Once forecast_peak_hour has passed, the day is
    past its forecasted heating phase — confidence in cooling should
    rise monotonically as obs accumulate post-peak.

    Oscillation between scans is the bin-lock-discontinuity churn
    surfacing as data: a confidence of 0.69 → 0.71 → 0.68 → 0.72
    crossing the STRONG_COOLING_THRESHOLD repeatedly flips bin-lock on
    and off, which flips our_p discontinuously, which flips Kelly size.
    The guard surfaces the underlying signal noise."""
    this_row = _this_scan_row(conn, city, event_date, scan_at_utc)
    prev_row = _prior_scan_row(conn, city, event_date, scan_at_utc)
    if not this_row or not prev_row:
        return None
    p_cc = prev_row["cooling_confidence"]
    c_cc = this_row["cooling_confidence"]
    peak_h = this_row["forecast_peak_hour"]
    if p_cc is None or c_cc is None or peak_h is None:
        return None
    meta = CITY_STATIONS.get(city)
    if not meta:
        return None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(meta[2])
        curr_t = datetime.fromisoformat(scan_at_utc.replace("Z", "+00:00"))
        local_hour = curr_t.astimezone(tz).hour
    except (ValueError, KeyError):
        return None
    if local_hour < peak_h:
        return None
    delta = c_cc - p_cc
    if delta < -COOLING_CONF_TOL:
        return InvariantViolation(
            scan_at_utc=scan_at_utc,
            guard_name="cooling_confidence_monotone_post_peak",
            city=city, event_date=event_date,
            prev_value=float(p_cc), curr_value=float(c_cc), delta=delta,
            detail=(f"cooling_confidence dropped {p_cc:.2f} → {c_cc:.2f} "
                    f"post-peak (Δ={delta:+.2f}); oscillation here is "
                    "bin-lock-discontinuity churn surfacing as data"),
        )
    return None


@guard("mu_jump_incoherent_with_neighbors")
def _check_mu_neighbor_coherence(conn, city, event_date, scan_at_utc):
    """High-precision source-flip detector: μ jumped >=1.5°C in <1h
    AND upwind neighbors did NOT move coherently with the jump.

    Real weather is spatially coherent — a frontal passage moves
    upwind neighbors with us.  Parse errors and source-flips are
    station-specific.  This is a much better discriminator than any
    magnitude threshold."""
    this_row = _this_scan_row(conn, city, event_date, scan_at_utc)
    prev_row = _prior_scan_row(conn, city, event_date, scan_at_utc)
    if not this_row or not prev_row:
        return None
    p_mu = prev_row["mu_c"]
    c_mu = this_row["mu_c"]
    if p_mu is None or c_mu is None:
        return None

    try:
        prev_t = datetime.fromisoformat(prev_row["scanned_at_utc"].replace("Z", "+00:00"))
        curr_t = datetime.fromisoformat(scan_at_utc.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    dt_hours = (curr_t - prev_t).total_seconds() / 3600.0
    if dt_hours <= 0 or dt_hours > MU_JUMP_TIME_WINDOW_H:
        # Only check fast jumps.  Slow drifts (>1h between scans) are
        # not the source-flip signature.
        return None

    delta_mu = c_mu - p_mu
    if abs(delta_mu) < MU_JUMP_MAGNITUDE_FLOOR_C:
        # Below the magnitude floor — too small to be a bug, regardless
        # of neighbor coherence.
        return None

    nbr_delta = _neighbor_temp_delta_c(city, prev_row["scanned_at_utc"], scan_at_utc)
    if nbr_delta is None:
        # No neighbor data for coherence check.  Fall back to magnitude-
        # only: the jump exists, we just can't verify it spatially.
        # Flag but mark as low-confidence.
        return InvariantViolation(
            scan_at_utc=scan_at_utc,
            guard_name="mu_jump_incoherent_with_neighbors",
            city=city, event_date=event_date,
            prev_value=float(p_mu), curr_value=float(c_mu), delta=delta_mu,
            detail=(f"μ jumped {delta_mu:+.2f}°C in {dt_hours:.2f}h; "
                    "no neighbor data for coherence check"),
        )

    if abs(delta_mu - nbr_delta) >= MU_NEIGHBOR_COHERENCE_TOL_C:
        return InvariantViolation(
            scan_at_utc=scan_at_utc,
            guard_name="mu_jump_incoherent_with_neighbors",
            city=city, event_date=event_date,
            prev_value=float(p_mu), curr_value=float(c_mu), delta=delta_mu,
            detail=(f"μ jumped {delta_mu:+.2f}°C in {dt_hours:.2f}h "
                    f"but upwind neighbors moved {nbr_delta:+.2f}°C "
                    f"(|Δμ - Δnbr|={abs(delta_mu - nbr_delta):.2f}°C, "
                    f"tol={MU_NEIGHBOR_COHERENCE_TOL_C:.2f}°C); "
                    "real weather is spatially coherent, this jump is not"),
        )
    return None


# ============================================================
# Persistence + orchestrator
# ============================================================

def _write_violation(conn: sqlite3.Connection, v: InvariantViolation) -> None:
    """Write one violation row to guard_violations.  Idempotent via
    UNIQUE constraint — repeating a guard run on the same (scan,
    guard, city) tuple is a no-op."""
    row = v.to_row()
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(
        f"INSERT OR IGNORE INTO guard_violations ({','.join(cols)}) "
        f"VALUES ({placeholders})",
        [row[c] for c in cols],
    )


def run_invariant_checks(conn: sqlite3.Connection, scan_at_utc: str) -> None:
    """Run every registered guard against the just-completed scan.
    Pure side-effect: writes to guard_violations + logs at WARNING.

    Returns None by design.  This module's outputs never re-enter the
    prediction or trading path.  See the OBSERVATIONAL FOREVER design
    rule at the top of this file.

    Safe to call with a partially-written scan: guards that can't find
    their inputs silently return None.
    """
    try:
        conn.row_factory = sqlite3.Row
        cities_dates = conn.execute(
            "SELECT DISTINCT city, event_date FROM paper_predictor_signals "
            "WHERE scanned_at_utc = ?",
            (scan_at_utc,),
        ).fetchall()
    except sqlite3.Error as e:
        log.warning(f"invariant_guards: query failed: {e}")
        return

    total_fired = 0
    for cd in cities_dates:
        city, event_date = cd["city"], cd["event_date"]
        if not city or not event_date:
            continue
        for name, fn in _GUARDS:
            try:
                v = fn(conn, city, event_date, scan_at_utc)
            except Exception as e:
                log.warning(
                    f"invariant_guard {name} raised on {city}/{event_date}: {e}"
                )
                continue
            if v is None:
                continue
            try:
                _write_violation(conn, v)
                conn.commit()
            except sqlite3.Error as e:
                log.warning(f"invariant_guard {name} write failed: {e}")
                continue
            log.warning(
                f"INVARIANT VIOLATION [{name}] {city}/{event_date}: {v.detail}"
            )
            total_fired += 1

    if total_fired:
        log.info(f"invariant_guards: {total_fired} violations recorded "
                  f"this scan ({len(cities_dates)} city/date pairs checked)")