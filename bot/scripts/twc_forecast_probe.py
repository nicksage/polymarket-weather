#!/usr/bin/env python3
"""
twc_forecast_probe.py — Phase A of TWC Goal 2 (probabilistic forecast).

Standalone CLI.  For each US (city, event_date), call TWC's
probabilistic hourly forecast endpoint, derive the daily-max
distribution from the prototype ensemble, and print per-bin
probabilities alongside our model and the market.

Pure measurement.  No DB writes.  No live-trading impact.

USAGE
    # All US cities for today
    python bot/scripts/twc_forecast_probe.py

    # Single city
    python bot/scripts/twc_forecast_probe.py --city Miami

    # Specific event date (e.g., tomorrow)
    python bot/scripts/twc_forecast_probe.py --event-date 2026-06-20

WHY PROTOTYPES (NOT discretePdfs OR probabilities)
The TWC docs warn explicitly:
    "Calibration is not available for min(), max(), or sum()
     aggregations of hourly temperatures or other parameters."

The hourly PDFs and per-hour range probabilities are BMA-calibrated,
but combining them across hours to get a daily-max distribution
violates that calibration (the hours aren't independent).

Prototypes preserve temporal correlation — each is a coherent
forecast realization.  Taking max(prototype) per day gives a sample
from the true daily-max distribution, and counting samples per bin
gives a calibration-preserving probability estimate.

REQUIRES
    TWC_API_KEY in .env (or shell env)
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone, date as date_t
from typing import Optional
from zoneinfo import ZoneInfo

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
log = logging.getLogger("twc_forecast_probe")
# Silence httpx URL-with-apiKey logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


TWC_API_BASE = os.getenv("TWC_API_BASE", "https://api.weather.com")
TWC_API_KEY  = os.getenv("TWC_API_KEY", "")
TWC_PROBABILISTIC_PATH = os.getenv(
    "TWC_PROBABILISTIC_PATH", "/v3/wx/forecast/probabilistic")
TWC_CURRENT_CONDITIONS_PATH = os.getenv(
    "TWC_CURRENT_CONDITIONS_PATH", "/v3/wx/observations/current")
# Deterministic daily forecast — TWC's public-facing product (the one
# end-users see in The Weather Channel app).  Independent of the
# probabilistic BMA pipeline, so agreement between the two is a real
# signal (not a circular check).  Falls back to P50 of the probabilistic
# distribution if this endpoint isn't entitled.
TWC_DAILY_FORECAST_PATH = os.getenv(
    "TWC_DAILY_FORECAST_PATH", "/v3/wx/forecast/daily/15day")
# 15-min short-range nowcast — 28 quarter-hour slots = 7h horizon.
# At <7h this is TWC's freshest product (HRRR/RAP-blended with radar);
# when it diverges from the longer-horizon probabilistic forecast, the
# nowcast usually has newer information (afternoon cloud cover, cold
# airmass arrival, etc.).  Used for intraday peak confirmation.
TWC_FIFTEENMIN_PATH = os.getenv(
    "TWC_FIFTEENMIN_PATH", "/v3/wx/forecast/fifteenminute")
TWC_LANGUAGE  = os.getenv("TWC_LANGUAGE", "en-US")
TWC_TIMEOUT_S = float(os.getenv("TWC_TIMEOUT_S", "30"))
N_PROTOTYPES_DEFAULT = int(os.getenv("TWC_N_PROTOTYPES", "100"))
HOURS_DEFAULT        = int(os.getenv("TWC_FORECAST_HOURS", "72"))

# Hard cap enforced at the TWC Akamai edge.  Verified by binary search
# 2026-06-19: N=100 succeeds, N=105 returns the SAME 503 transaction_id
# as N=200, meaning Akamai serves a cached error before the request
# reaches TWC's backend.  Set just above 100 here as a safety stop —
# requesting more is guaranteed to fail.
N_PROTOTYPES_MAX = int(os.getenv("TWC_N_PROTOTYPES_MAX", "100"))


# ============================================================
# TWC API + per-city helpers
# ============================================================

def _units_for(settlement_unit: str) -> str:
    """TWC units code: 'e' = English (°F), 'm' = Metric (°C)."""
    return "e" if (settlement_unit or "").lower() == "fahrenheit" else "m"


def is_domestic_icao(icao: str) -> bool:
    """K-prefixed ICAOs are continental US (Polymarket settles °F)."""
    return bool(icao) and icao.upper().startswith("K")


def default_settlement_unit_for_icao(icao: str) -> str:
    """Best-guess settlement unit when no DB bins exist (twc_only mode).
    Polymarket convention: US markets settle in °F, international in °C.
    Used as the unit for the TWC API call (units=e or m) AND as the
    label for synthesized bins."""
    return "fahrenheit" if is_domestic_icao(icao) else "celsius"


def filter_cities_by_scope(cities_dict: dict, scope: str) -> list[str]:
    """Pick cities to probe based on --scope:
        domestic       -> K-prefixed ICAOs only (continental US)
        international  -> non-K ICAOs
        all            -> everything mapped"""
    s = (scope or "all").lower()
    out = []
    for city, meta in cities_dict.items():
        if not meta or not isinstance(meta[0], str):
            continue
        icao = meta[0]
        dom = is_domestic_icao(icao)
        if s == "domestic" and dom:
            out.append(city)
        elif s == "international" and not dom:
            out.append(city)
        elif s == "all":
            out.append(city)
    return sorted(out)


def fetch_probabilistic(icao: str, settlement_unit: str,
                            n_prototypes: int = N_PROTOTYPES_DEFAULT,
                            hours: int = HOURS_DEFAULT) -> dict:
    """Call /v3/wx/forecast/probabilistic for one station and return
    the forecasts1Hour dict (with fcstValid + prototypes).

    We only request `prototypes` — that's what we need for the daily-max
    derivation.  Skipping percentiles / probabilities / discretePdfs
    keeps the response size down."""
    if not TWC_API_KEY:
        raise RuntimeError("TWC_API_KEY env var not set")
    params = {
        "icaoCode":   icao,
        "units":      _units_for(settlement_unit),
        "language":   TWC_LANGUAGE,
        "format":     "json",
        "hours":      hours,
        "prototypes": f"temperature:{n_prototypes}",
        "apiKey":     TWC_API_KEY,
    }
    url = f"{TWC_API_BASE}{TWC_PROBABILISTIC_PATH}"
    resp = httpx.get(url, params=params, timeout=TWC_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"TWC HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("forecasts1Hour", {})


# ============================================================
# Current conditions fetch — for the observed-max floor (technique #1)
# ============================================================

def fetch_current_conditions(icao: str, settlement_unit: str) -> dict:
    """Call /v3/wx/observations/current.  Returns dict with:
       'temp_now', 'max_since_7am', 'max_24h', 'min_24h',
       'valid_time_local', 'notes'.

    `temperatureMaxSince7Am` is the key field — running daily max
    over the current calendar day (7am-7am-local TWC convention).
    Used as the lower-bound floor for the daily-max distribution.
    """
    if not TWC_API_KEY:
        raise RuntimeError("TWC_API_KEY env var not set")
    params = {
        "icaoCode": icao,
        "units":    _units_for(settlement_unit),
        "language": TWC_LANGUAGE,
        "format":   "json",
        "apiKey":   TWC_API_KEY,
    }
    url = f"{TWC_API_BASE}{TWC_CURRENT_CONDITIONS_PATH}"
    resp = httpx.get(url, params=params, timeout=TWC_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"TWC HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return {
        "temp_now":         data.get("temperature"),
        "max_since_7am":    data.get("temperatureMaxSince7Am"),
        "max_24h":          data.get("temperatureMax24Hour"),
        "min_24h":          data.get("temperatureMin24Hour"),
        "valid_time_local": data.get("validTimeLocal"),
        "notes":            "",
    }


# ============================================================
# Deterministic daily-max forecast — independent product for confidence
# ============================================================

def fetch_deterministic_daily_max(
    icao: str, settlement_unit: str, event_date: str, tz_str: str,
) -> dict:
    """Call /v3/wx/forecast/daily/15day and return TWC's deterministic
    daily-max for `event_date`.  Independent of the probabilistic BMA
    pipeline (the forecast end-users see in TWC's apps), so agreement
    between the two is a meaningful confidence signal.

    Prefers `calendarDayTemperatureMax` (midnight-to-midnight, matches
    Polymarket settlement convention, persists all day).  Falls back to
    `temperatureMax` (7AM-7PM daypart high, goes null after 3PM LAT) if
    the calendar-day field isn't populated.  See docs:
        https://developer.weather.com/docs/openapi/daily-forecast-3-0-0

    Returns dict:
        {
          "status":          "ok" | "not_entitled" | "no_data" | "error",
          "today_max":       float | None  (in settlement_unit)
          "source_field":    "calendarDayTemperatureMax" |
                              "temperatureMax" | None
          "narrative":       str | None
          "valid_time_local": str | None
          "err":             str | None
        }

    On HTTP 401/403 returns status='not_entitled' instead of raising —
    lets the probe gracefully fall back to using the probabilistic P50."""
    if not TWC_API_KEY:
        return {"status": "error", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": "TWC_API_KEY env var not set"}
    params = {
        "icaoCode": icao,
        "units":    _units_for(settlement_unit),
        "language": TWC_LANGUAGE,
        "format":   "json",
        "apiKey":   TWC_API_KEY,
    }
    url = f"{TWC_API_BASE}{TWC_DAILY_FORECAST_PATH}"
    try:
        resp = httpx.get(url, params=params, timeout=TWC_TIMEOUT_S)
    except Exception as e:
        return {"status": "error", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": f"{type(e).__name__}: {e}"}

    if resp.status_code in (401, 403):
        return {"status": "not_entitled", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": f"HTTP {resp.status_code}"}
    if resp.status_code != 200:
        return {"status": "error", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    try:
        data = resp.json()
    except Exception as e:
        return {"status": "error", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": f"json decode: {e}"}

    # The 15-day product has both a flat wrapper shape and a nested-by-
    # product-name shape across TWC variants.  Try both.  Recognize either
    # the calendar-day or the 7am-anchored max field as a valid payload.
    def _looks_like_payload(d: dict) -> bool:
        return bool(d.get("calendarDayTemperatureMax")
                    or d.get("temperatureMax")
                    or d.get("validTimeLocal"))

    payload = data
    if not _looks_like_payload(data):
        for v in data.values():
            if isinstance(v, dict) and _looks_like_payload(v):
                payload = v
                break

    cal_arr   = payload.get("calendarDayTemperatureMax") or []
    max_arr   = payload.get("temperatureMax") or []
    valid_arr = payload.get("validTimeLocal") or []
    narrative_arr = payload.get("narrative") or []

    if not valid_arr or (not cal_arr and not max_arr):
        return {"status": "no_data", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": "no calendarDayTemperatureMax / temperatureMax / "
                       "validTimeLocal arrays in response"}

    # Find the index whose validTimeLocal's local date matches event_date.
    # validTimeLocal entries look like '2026-06-26T07:00:00-0400' — the
    # date prefix already encodes the station-local day.
    target_idx: Optional[int] = None
    for i, vt in enumerate(valid_arr):
        if isinstance(vt, str) and vt[:10] == event_date:
            target_idx = i
            break
    if target_idx is None:
        return {"status": "no_data", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": f"no entry for event_date={event_date} in 15-day forecast"}

    # Prefer calendar-day field: matches Polymarket settlement AND persists
    # all day (vs temperatureMax which goes null after 3PM LAT).
    raw_max: Optional[float] = None
    source_field: Optional[str] = None
    if target_idx < len(cal_arr) and cal_arr[target_idx] is not None:
        raw_max = float(cal_arr[target_idx])
        source_field = "calendarDayTemperatureMax"
    elif target_idx < len(max_arr) and max_arr[target_idx] is not None:
        raw_max = float(max_arr[target_idx])
        source_field = "temperatureMax"

    if raw_max is None:
        return {"status": "no_data", "today_max": None,
                "source_field": None,
                "narrative": None, "valid_time_local": None,
                "err": f"both calendarDayTemperatureMax and temperatureMax "
                       f"are null at index {target_idx} (unusual — most "
                       f"likely a response variant without these fields)"}

    return {
        "status":           "ok",
        "today_max":        raw_max,
        "source_field":     source_field,
        "narrative":        (narrative_arr[target_idx]
                              if target_idx < len(narrative_arr) else None),
        "valid_time_local": valid_arr[target_idx],
        "err":              None,
    }


def deterministic_max_from_p50(samples: list[float]) -> Optional[float]:
    """Fallback when the daily/15day endpoint isn't entitled or has
    dropped today's entry — use the P50 of the prototype-derived daily-max
    distribution as the 'deterministic' point estimate.

    Less independent than the daily forecast (computed from the same BMA
    pipeline), but still useful when the agreement signal is taken with a
    grain of salt."""
    if not samples:
        return None
    s = sorted(samples)
    return float(s[len(s) // 2])


def compute_forecast_agreement_confidence(
    deterministic_max: float, bin_probs: dict[str, float], bins: list[dict],
) -> dict:
    """Option A — confidence = P(bin containing deterministic forecast).

    Interpretation: 'TWC's probabilistic distribution puts X% mass on the
    bin TWC's deterministic forecast points to.'  High = the two TWC
    products agree → high confidence.  Low = internal disagreement.

    Half-up rounding matches Polymarket settlement convention (Phase 1
    backtest: 91.0% match vs 69.2% for truncation).

    Returns:
        {
          "confidence":      float in [0, 1]   — Option A score
          "det_bin_label":   str | None        — bin det_max falls in
          "det_bin_prob":    float | None      — same as confidence;
                                                  exposed for symmetry
          "mode_bin_label":  str | None        — highest-prob bin
          "mode_bin_prob":   float | None
          "det_rounded":     int               — det_max under half-up
        }
    Returns confidence=0.0 if det_max rounds outside every bin (open-ended
    bins should prevent this in practice)."""
    if deterministic_max is None or not bin_probs or not bins:
        return {"confidence": 0.0, "det_bin_label": None,
                "det_bin_prob": None, "mode_bin_label": None,
                "mode_bin_prob": None, "det_rounded": None}

    det_rounded = _round_half_up(float(deterministic_max))

    # Mode bin = highest-probability bin
    mode_lbl = max(bin_probs, key=bin_probs.get)
    mode_p = bin_probs[mode_lbl]

    # Find the bin whose [lo, hi] range contains det_rounded
    det_bin_label: Optional[str] = None
    for b in bins:
        lo, hi = b.get("range_low"), b.get("range_high")
        lo_ok = (lo is None) or (det_rounded >= lo)
        hi_ok = (hi is None) or (det_rounded <= hi)
        if lo_ok and hi_ok:
            det_bin_label = b["label"]
            break

    det_bin_prob = bin_probs.get(det_bin_label, 0.0) if det_bin_label else 0.0
    return {
        "confidence":     det_bin_prob,
        "det_bin_label":  det_bin_label,
        "det_bin_prob":   det_bin_prob,
        "mode_bin_label": mode_lbl,
        "mode_bin_prob":  mode_p,
        "det_rounded":    det_rounded,
    }


# ============================================================
# 15-min nowcast peak — intraday peak confirmation (7h horizon)
# ============================================================

def fetch_15min_peak(
    icao: str, settlement_unit: str, event_date: str, tz_str: str,
) -> dict:
    """Call /v3/wx/forecast/fifteenminute and return the highest forecast
    temperature across the slots whose station-local date == event_date.

    TWC's 15-min product covers 28 slots × 15 min = 7 hours.  At that
    horizon it's the freshest signal we have — HRRR/RAP-blended with
    radar — and consistently beats the longer-horizon hourly forecast
    inside 6h.  Use it to confirm (or override) the longer-horizon
    probabilistic peak estimate.

    Returns dict:
        {
          "status":          "ok" | "not_entitled" | "no_data" | "error"
          "peak_temp":       float | None   (settlement_unit)
          "peak_local_hour": int | None     (0-23, station-local)
          "n_slots_today":   int            (of the 28, how many fell on event_date)
          "horizon_hours":   float          (n_slots_today * 0.25)
          "err":             str | None
        }
    On 401/403 returns status='not_entitled' instead of raising."""
    if not TWC_API_KEY:
        return {"status": "error", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": "TWC_API_KEY env var not set"}
    params = {
        "icaoCode": icao,
        "units":    _units_for(settlement_unit),
        "language": TWC_LANGUAGE,
        "format":   "json",
        "apiKey":   TWC_API_KEY,
    }
    url = f"{TWC_API_BASE}{TWC_FIFTEENMIN_PATH}"
    try:
        resp = httpx.get(url, params=params, timeout=TWC_TIMEOUT_S)
    except Exception as e:
        return {"status": "error", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": f"{type(e).__name__}: {e}"}
    if resp.status_code in (401, 403):
        return {"status": "not_entitled", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": f"HTTP {resp.status_code}"}
    if resp.status_code != 200:
        return {"status": "error", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    try:
        data = resp.json()
    except Exception as e:
        return {"status": "error", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": f"json decode: {e}"}

    # The 15-min product also has both flat and nested-by-product shapes
    # across TWC variants.  Try flat first; fall through to nested.
    payload = data
    if not data.get("temperature") and not data.get("validTimeLocal"):
        for v in data.values():
            if isinstance(v, dict) and (
                v.get("temperature") or v.get("validTimeLocal")):
                payload = v
                break

    temps = payload.get("temperature") or []
    valid = payload.get("validTimeLocal") or []
    if not temps or not valid:
        return {"status": "no_data", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": "no temperature/validTimeLocal in response"}

    # Filter to slots whose local date matches event_date.  validTimeLocal
    # already encodes station-local time so we can substring-match the
    # date prefix without timezone math.
    best_t: Optional[float] = None
    best_hour: Optional[int] = None
    n_today = 0
    for i, vt in enumerate(valid):
        if not isinstance(vt, str) or vt[:10] != event_date:
            continue
        if i >= len(temps) or temps[i] is None:
            continue
        n_today += 1
        t = float(temps[i])
        if best_t is None or t > best_t:
            best_t = t
            try:
                best_hour = int(vt[11:13])
            except (ValueError, IndexError):
                best_hour = None

    if best_t is None:
        return {"status": "no_data", "peak_temp": None,
                "peak_local_hour": None, "n_slots_today": 0,
                "horizon_hours": 0.0,
                "err": f"no slots on event_date={event_date} within 7h horizon"}

    return {
        "status":          "ok",
        "peak_temp":       best_t,
        "peak_local_hour": best_hour,
        "n_slots_today":   n_today,
        "horizon_hours":   n_today * 0.25,
        "err":             None,
    }


def compute_intraday_agreement(
    nowcast_peak: Optional[float],
    probabilistic_p50: Optional[float],
    observed_max_so_far: Optional[float],
    settlement_unit: str,
) -> dict:
    """Three-way agreement between TWC's freshest short-range nowcast,
    the longer-horizon probabilistic P50, and the actual observed max
    so far.

    Both forecasts are floored by observed_max (a forecast can't undercut
    what already happened) before comparison.  Spread = |nowcast - prob|
    in settlement units.

    Returns:
        {
          "nowcast_adj":     float | None
          "prob_adj":        float | None
          "spread":          float | None
          "observed_max":    float | None
          "tight_threshold": float
          "verdict":         "tight" | "loose" | "diverged" | "no_nowcast"
        }

    Verdict tiers (per Polymarket bin width):
        US 2°F bins: tight ≤ 1.0°F, loose ≤ 2.0°F, diverged > 2.0°F
        Intl 1°C bins: tight ≤ 0.5°C, loose ≤ 1.0°C, diverged > 1.0°C"""
    is_fahrenheit = (settlement_unit or "").lower() == "fahrenheit"
    tight = 1.0 if is_fahrenheit else 0.5

    if nowcast_peak is None:
        return {"nowcast_adj": None, "prob_adj": probabilistic_p50,
                "spread": None, "observed_max": observed_max_so_far,
                "tight_threshold": tight, "verdict": "no_nowcast"}

    nowcast_adj = nowcast_peak
    prob_adj = probabilistic_p50
    if observed_max_so_far is not None:
        nowcast_adj = max(nowcast_adj, observed_max_so_far)
        if prob_adj is not None:
            prob_adj = max(prob_adj, observed_max_so_far)

    if prob_adj is None:
        return {"nowcast_adj": nowcast_adj, "prob_adj": None,
                "spread": None, "observed_max": observed_max_so_far,
                "tight_threshold": tight, "verdict": "no_nowcast"}

    spread = abs(nowcast_adj - prob_adj)
    if spread <= tight:
        verdict = "tight"
    elif spread <= 2 * tight:
        verdict = "loose"
    else:
        verdict = "diverged"
    return {
        "nowcast_adj":      nowcast_adj,
        "prob_adj":         prob_adj,
        "spread":           spread,
        "observed_max":     observed_max_so_far,
        "tight_threshold":  tight,
        "verdict":          verdict,
    }


# ============================================================
# Fusion logic — pin the daily-max distribution above the observed floor
# ============================================================

def is_event_today_in_tz(event_date_iso: str, tz_str: str) -> bool:
    """True if event_date matches today's date in the station's
    local timezone.  Determines whether the observed-floor fusion
    applies: only meaningful when the event_date is the current day."""
    try:
        tz = ZoneInfo(tz_str)
        today_local = datetime.now(tz).date()
        target = datetime.strptime(event_date_iso, "%Y-%m-%d").date()
        return target == today_local
    except Exception:
        return False


def apply_observed_floor(samples: list[float], floor: float) -> list[float]:
    """Clip every sample up to the floor — represents the constraint
    that today's actual daily-max can't be less than what we've already
    observed.  Returns a new list, same length, with each sample =
    max(floor, sample)."""
    return [max(floor, s) for s in samples]


def _c_to_unit(temp_c: float, settlement_unit: str) -> float:
    """Convert Celsius to the settlement unit (fahrenheit or celsius)."""
    if (settlement_unit or "").lower() == "fahrenheit":
        return temp_c * 9.0 / 5.0 + 32.0
    return temp_c


def _query_metar_calendar_day_max_c(
    conn: sqlite3.Connection, icao: str, event_date: str, tz_str: str,
) -> Optional[float]:
    """Return the highest temp_c in raw_metar_log for (icao, event_date)
    over the station-local calendar day so far.  None if no rows.

    Filters by BOTH raw_metar_log.event_date AND by
    cycle_timestamp_utc-converted-to-local — defensive against UTC vs
    local-date edge cases at midnight (a UTC-day row may land on the
    prior local day, and vice versa)."""
    try:
        rows = conn.execute(
            """SELECT cycle_timestamp_utc, temp_c
               FROM raw_metar_log
               WHERE icao = ? AND event_date = ? AND temp_c IS NOT NULL""",
            (icao, event_date),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return None
    try:
        tz = ZoneInfo(tz_str)
        target_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    except Exception:
        return None
    best: Optional[float] = None
    for r in rows:
        try:
            ts_utc = datetime.fromisoformat(
                str(r[0]).replace("Z", "+00:00"))
            ts_local = ts_utc.astimezone(tz)
        except (ValueError, TypeError):
            continue
        if ts_local.date() != target_date:
            continue
        t = float(r[1])
        if best is None or t > best:
            best = t
    return best


def compute_calendar_day_floor(
    conn: sqlite3.Connection, icao: str, event_date: str, tz_str: str,
    twc_max_since_7am: Optional[float], settlement_unit: str,
) -> tuple[Optional[float], str]:
    """Compute the observed-max floor over the station-local CALENDAR DAY
    (00:00 → now local), addressing Polymarket's settlement convention
    which is calendar-day — not the 7am-anchored window TWC's
    `temperatureMaxSince7Am` exposes.

    Returns (floor, source_note).  Floor is in `settlement_unit`.

    Strategy:
        - METAR floor: max(raw_metar_log.temp_c) over the local calendar
          day so far (covers the full 00:00→now window, including
          overnight cycles before 07:00).
        - TWC floor: `temperatureMaxSince7Am` (covers 07:00→now, but may
          be fresher than the latest METAR cycle we've persisted).
        - Combined: max(METAR, TWC).  If neither is available, None.

    Rationale: on rare days the daily max occurs overnight (cold front
    passage, marine layer break, etc.).  `temperatureMaxSince7Am` misses
    those entirely; the METAR-day floor catches them.  Conversely,
    TWC's CC endpoint may be minutes fresher than our latest persisted
    METAR cycle, so we don't discard it."""
    metar_max_c = _query_metar_calendar_day_max_c(
        conn, icao, event_date, tz_str)
    metar_floor: Optional[float] = (
        _c_to_unit(metar_max_c, settlement_unit)
        if metar_max_c is not None else None)
    twc_floor: Optional[float] = (
        float(twc_max_since_7am) if twc_max_since_7am is not None else None)

    candidates = [(metar_floor, "metar_calendar_day"),
                    (twc_floor, "twc_since7am")]
    candidates = [(v, src) for (v, src) in candidates if v is not None]
    if not candidates:
        return None, "no observed-max source available"
    best_v, _ = max(candidates, key=lambda x: x[0])

    unit_sym = "°F" if (settlement_unit or "").lower() == "fahrenheit" else "°C"
    parts = []
    if metar_floor is not None:
        parts.append(f"metar={metar_floor:.1f}{unit_sym}")
    if twc_floor is not None:
        parts.append(f"twc7am={twc_floor:.1f}{unit_sym}")
    note = f"floor={best_v:.1f}{unit_sym} (calendar-day; {', '.join(parts)})"
    return best_v, note


# ============================================================
# Daily-max derivation (prototype ensemble → bin probabilities)
# ============================================================

def derive_daily_max_samples(
    forecasts_1hour: dict, event_date: str, tz_str: str,
) -> tuple[list[float], int]:
    """From forecasts1Hour, return (sample_maxes, n_hours_covered):
      - sample_maxes: one daily-max sample per prototype
      - n_hours_covered: how many hourly slots fell on event_date local
    Filters forecast hours to those whose station-local date matches
    event_date, then takes max per prototype across those hours."""
    fcst_valid = forecasts_1hour.get("fcstValid", [])
    protos = forecasts_1hour.get("prototypes", [])
    if not fcst_valid or not protos:
        return [], 0

    temp_proto = next(
        (p for p in protos if p.get("parameter") == "temperature"), None)
    if not temp_proto:
        return [], 0
    forecasts = temp_proto.get("forecast", [])   # 2D: prototypes × hours
    if not forecasts:
        return [], 0

    tz = ZoneInfo(tz_str)
    target_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    hour_indices: list[int] = []
    for i, ts in enumerate(fcst_valid):
        try:
            dt_local = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tz)
        except (TypeError, ValueError, OSError):
            continue
        if dt_local.date() == target_date:
            hour_indices.append(i)
    if not hour_indices:
        return [], 0

    sample_maxes: list[float] = []
    for proto in forecasts:
        vals = [proto[i] for i in hour_indices if i < len(proto)]
        if vals:
            sample_maxes.append(max(vals))
    return sample_maxes, len(hour_indices)


def _round_half_up(x: float) -> int:
    """Half-up rounding to match Polymarket settlement convention."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def bin_probabilities(sample_maxes: list[float],
                          bins: list[dict]) -> dict[str, float]:
    """For each Polymarket bin, return the fraction of samples whose
    half-up-rounded value falls inside [range_low, range_high].
    Open-ended bins (None on one edge) supported."""
    if not sample_maxes:
        return {}
    n = len(sample_maxes)
    out: dict[str, float] = {}
    for b in bins:
        lo, hi = b.get("range_low"), b.get("range_high")
        cnt = 0
        for v in sample_maxes:
            r = _round_half_up(float(v))
            lo_ok = (lo is None) or (r >= lo)
            hi_ok = (hi is None) or (r <= hi)
            if lo_ok and hi_ok:
                cnt += 1
        out[b["label"]] = cnt / n
    return out


# ============================================================
# Bin lookup (from paper_predictor_signals)
# ============================================================

def fetch_event_bins(conn: sqlite3.Connection,
                          city: str, event_date: str) -> list[dict]:
    """Latest-scan bin set for this (city, event_date), with our_prob
    and market_prob attached so we can print all three side-by-side."""
    try:
        rows = conn.execute(
            """SELECT s.bin_label, s.bin_range_low, s.bin_range_high,
                       s.unit, s.our_prob, s.market_prob
               FROM paper_predictor_signals s
               WHERE s.city = ? AND s.event_date = ?
                 AND s.bin_range_low IS NOT NULL
                 AND s.scanned_at_utc = (
                     SELECT MAX(scanned_at_utc) FROM paper_predictor_signals
                     WHERE city = ? AND event_date = ?
                 )
               ORDER BY s.bin_range_low ASC""",
            (city, event_date, city, event_date)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"label": r[0], "range_low": r[1], "range_high": r[2],
         "unit": r[3], "our_prob": r[4], "market_prob": r[5]}
        for r in rows
    ]


# ============================================================
# Synthesize Polymarket-style bins from TWC's forecast distribution.
# Used when no DB bins exist for the (city, event_date) — e.g.,
# future dates where the bot hasn't scanned yet, or events that aren't
# resolved markets.  Lets us still print TWC's view of the day.
# ============================================================

def synthesize_bins_from_samples(
    sample_maxes: list[float], unit: str = "fahrenheit",
) -> list[dict]:
    """Generate Polymarket-style 2°F bins spanning the prototype range.

    For US °F: bins are (lo, lo+1) with lo even, covering
    [floor(P05)-aligned_even .. ceil(P95)-aligned_even+2].  Matches
    Polymarket's '90-91°F' style — each bin covers a 2°F settlement
    window via half-up rounding.

    For non-US: returns 1°C bins."""
    if not sample_maxes:
        return []
    sm = sorted(sample_maxes)
    n = len(sm)
    p05 = sm[max(0, int(0.05 * n))]
    p95 = sm[min(n - 1, int(0.95 * n))]
    # Widen by 2 units on each side so the tails of the distribution
    # are visible, not clipped to the edge bins.
    lo_int = int(p05) - 2
    hi_int = int(p95) + 2

    bins: list[dict] = []
    if (unit or "").lower() == "fahrenheit":
        # Align lo to an even integer (Polymarket US bins are 90-91, 92-93, ...)
        if lo_int % 2 != 0:
            lo_int -= 1
        if hi_int % 2 == 0:
            hi_int += 1
        for lo in range(lo_int, hi_int + 1, 2):
            hi = lo + 1
            bins.append({
                "label":      f"{lo}-{hi}°F",
                "range_low":  float(lo),
                "range_high": float(hi),
                "unit":       "fahrenheit",
                "our_prob":   None,
                "market_prob": None,
            })
    else:
        for c in range(lo_int, hi_int + 1):
            bins.append({
                "label":      f"{c}°C",
                "range_low":  float(c),
                "range_high": float(c),
                "unit":       "celsius",
                "our_prob":   None,
                "market_prob": None,
            })
    return bins


# ============================================================
# Per-city probe
# ============================================================

def probe_city(conn: sqlite3.Connection,
                 city: str, event_date: str,
                 n_prototypes: int = N_PROTOTYPES_DEFAULT,
                 no_fusion: bool = False) -> dict:
    """Run the probe for one (city, event_date) and print results.
    Returns a small summary dict for the final overview table.

    Two display modes:
      * 'compare' — DB bins exist for this event_date.  Print
                     Our P / Mkt P / TWC P side by side per bin.
      * 'twc_only' — no DB bins.  Synthesize 2°F bins from TWC's
                      forecast range and show TWC P only.  Used for
                      future dates the bot hasn't scanned yet.
    """
    meta = CITY_STATIONS.get(city)
    if not meta:
        print(f"\n{city}: no ICAO mapping in station_meta — skipping")
        return {"city": city, "status": "no_icao"}
    icao, _net, tz_str = meta[0], meta[1], meta[2]

    # Always fetch TWC first so we have a forecast regardless of DB state
    db_bins = fetch_event_bins(conn, city, event_date)
    # Pick settlement unit.  When DB bins exist, use what Polymarket says.
    # In twc_only mode, fall back to the per-ICAO heuristic (K=°F, else=°C).
    if db_bins:
        unit = (db_bins[0].get("unit") or "fahrenheit").lower()
    else:
        unit = default_settlement_unit_for_icao(icao)
    unit_sym = "°F" if unit == "fahrenheit" else "°C"

    try:
        fh = fetch_probabilistic(icao, unit, n_prototypes=n_prototypes)
    except Exception as e:
        print(f"\n{city} {event_date} ({icao}): TWC fetch failed: {e}")
        return {"city": city, "status": "fetch_error", "err": str(e)}

    raw_samples, n_hours = derive_daily_max_samples(fh, event_date, tz_str)
    if not raw_samples:
        print(f"\n{city} {event_date} ({icao}): no prototypes covering "
              f"event_date (n_hours_in_window={n_hours}).  "
              f"Event may be outside the 72h forecast horizon.")
        return {"city": city, "status": "out_of_window"}

    # --- Observed-max fusion (technique #1) -----------------------
    # Only applies when event_date == today in station-local time.
    # Future dates: skipped (no observed yet).  Past dates: skipped
    # (Current Conditions doesn't have history; backtest path uses
    # final values differently anyway).
    #
    # Floor uses CALENDAR-DAY window (00:00→now local) — matches
    # Polymarket settlement, not TWC's 7am-anchored field alone.
    # See compute_calendar_day_floor() docstring for rationale.
    floor: Optional[float] = None
    current_obs: Optional[dict] = None
    fusion_note = "skipped (event_date is not today in station-local time)"
    if no_fusion:
        fusion_note = "disabled by --no-fusion"
    elif is_event_today_in_tz(event_date, tz_str):
        twc_since7am: Optional[float] = None
        try:
            current_obs = fetch_current_conditions(icao, unit)
            mx = current_obs.get("max_since_7am")
            if mx is not None:
                twc_since7am = float(mx)
        except Exception as e:
            fusion_note = f"current-conditions fetch failed: {e}"
        floor, fusion_note = compute_calendar_day_floor(
            conn, icao, event_date, tz_str, twc_since7am, unit)

    samples = (apply_observed_floor(raw_samples, floor)
               if floor is not None else raw_samples)

    sm = sorted(samples)
    def _p(q): return sm[min(len(sm)-1, max(0, int(q * len(sm))))]
    p10, p50, p90 = _p(0.10), _p(0.50), _p(0.90)
    mean = sum(samples) / len(samples)

    # Also compute raw (pre-fusion) stats for the BEFORE/AFTER line
    raw_sm = sorted(raw_samples)
    raw_p10 = raw_sm[max(0, int(0.10 * len(raw_sm)))]
    raw_p50 = raw_sm[max(0, int(0.50 * len(raw_sm)))]
    raw_p90 = raw_sm[min(len(raw_sm) - 1, int(0.90 * len(raw_sm)))]
    raw_mean = sum(raw_samples) / len(raw_samples)

    def _top(d: dict) -> Optional[str]:
        if not d: return None
        return max(d, key=d.get)

    # --- Decide display mode ---
    if db_bins:
        # Compare mode: use real Polymarket bins from DB
        bins = db_bins
        mode = "compare"
    else:
        # TWC-only mode: synthesize bins from FUSED forecast range
        # (so bins below the observed floor don't show empty rows)
        bins = synthesize_bins_from_samples(samples, unit=unit)
        mode = "twc_only"

    twc_probs = bin_probabilities(samples, bins)
    twc_top   = _top(twc_probs)

    # --- Deterministic forecast + agreement confidence (Option A) ---
    det = fetch_deterministic_daily_max(icao, unit, event_date, tz_str)
    det_max: Optional[float] = det.get("today_max")
    # Surface which field drove the value so post-3PM-LAT cases (where
    # temperatureMax goes null but calendarDayTemperatureMax persists)
    # are visible in the diagnostic line.
    det_source = (f"daily/15day:{det.get('source_field')}"
                  if det.get("source_field") else "daily/15day")
    if det_max is None:
        fallback = deterministic_max_from_p50(samples)
        if fallback is not None:
            det_max = fallback
            det_source = f"P50 fallback ({det.get('status')})"
    conf_info = compute_forecast_agreement_confidence(
        det_max, twc_probs, bins) if det_max is not None else {
            "confidence": 0.0, "det_bin_label": None, "det_bin_prob": None,
            "mode_bin_label": _top(twc_probs), "mode_bin_prob":
            (twc_probs.get(_top(twc_probs), 0.0) if twc_probs else 0.0),
            "det_rounded": None,
        }

    # --- 15-min nowcast peak + 3-way intraday agreement ---
    # Only meaningful when event_date is today (the 7h horizon can't see
    # tomorrow).  Skip otherwise — the value of nowcast comes from its
    # freshness, not from being a long-horizon prediction.
    nowcast: dict = {"status": "skipped_future_date"}
    intraday: dict = {"verdict": "no_nowcast", "nowcast_adj": None,
                      "prob_adj": None, "spread": None,
                      "tight_threshold": (1.0 if unit == "fahrenheit" else 0.5)}
    if is_event_today_in_tz(event_date, tz_str):
        nowcast = fetch_15min_peak(icao, unit, event_date, tz_str)
        # P50 for the agreement comparison is computed from the FUSED
        # samples (already floored), so it represents the model's CURRENT
        # best median estimate of today's max.
        intraday = compute_intraday_agreement(
            nowcast.get("peak_temp"),
            float(p50),
            floor,
            unit)

    # --- Header (common to both modes) ---
    print(f"\n{city}  {event_date}  ({icao}, {tz_str})  [{mode}]")
    print(f"  TWC: {len(samples)} prototypes × {n_hours} hours of event_date")
    if current_obs is not None:
        tn = current_obs.get("temp_now")
        mx = current_obs.get("max_since_7am")
        vt = current_obs.get("valid_time_local", "")[:19].replace("T", " ")
        print(f"  TWC current obs: {tn}{unit_sym} now, "
              f"max-since-7am = {mx}{unit_sym}  (valid {vt})")
    print(f"  fusion: {fusion_note}")
    # Deterministic line — always emit, even when fetch failed
    if det_max is not None:
        print(f"  TWC deterministic max: {det_max:.1f}{unit_sym}  "
              f"[source: {det_source}]")
        if conf_info.get("det_bin_label"):
            print(f"  confidence: {conf_info['confidence']*100:>5.1f}%  "
                  f"(det rounds to {conf_info['det_rounded']}{unit_sym} → "
                  f"bin {conf_info['det_bin_label']}; "
                  f"mode {conf_info['mode_bin_label']} @ "
                  f"{conf_info['mode_bin_prob']*100:.1f}%)")
    else:
        print(f"  TWC deterministic max: UNAVAILABLE  "
              f"[{det.get('status')}: {det.get('err','')[:80]}]")

    # 15-min nowcast + 3-way agreement
    if nowcast.get("status") == "ok":
        peak_t = nowcast.get("peak_temp")
        peak_h = nowcast.get("peak_local_hour")
        horizon = nowcast.get("horizon_hours", 0.0)
        print(f"  TWC 15-min nowcast peak: {peak_t:.1f}{unit_sym} @ "
              f"{peak_h:02d}:00 local  ({horizon:.1f}h of event_date in horizon)")
        if intraday.get("spread") is not None:
            verdict = intraday["verdict"]
            sym = {"tight": "✓", "loose": "●", "diverged": "⚠"}.get(
                verdict, "·")
            print(f"  intraday agreement: nowcast={intraday['nowcast_adj']:.1f} "
                  f"vs probP50={intraday['prob_adj']:.1f}  "
                  f"spread={intraday['spread']:.2f}{unit_sym}  "
                  f"[{verdict} {sym}]")
    elif nowcast.get("status") == "skipped_future_date":
        # Quiet skip — expected for non-today event dates
        pass
    else:
        print(f"  TWC 15-min nowcast peak: UNAVAILABLE  "
              f"[{nowcast.get('status')}: {nowcast.get('err','')[:80]}]")
    if floor is not None:
        print(f"  TWC daily-max BEFORE fusion: mean={raw_mean:.1f}{unit_sym}  "
              f"P10/P50/P90 = {raw_p10:.1f} / {raw_p50:.1f} / {raw_p90:.1f}{unit_sym}")
        print(f"  TWC daily-max AFTER  fusion: mean={mean:.1f}{unit_sym}  "
              f"P10/P50/P90 = {p10:.1f} / {p50:.1f} / {p90:.1f}{unit_sym}")
    else:
        print(f"  TWC daily-max dist: mean={mean:.1f}{unit_sym}  "
              f"P10/P50/P90 = {p10:.1f} / {p50:.1f} / {p90:.1f}{unit_sym}")

    # --- Per-bin table ---
    if mode == "compare":
        our_top = _top({b["label"]: (b.get("our_prob") or 0)    for b in bins})
        mkt_top = _top({b["label"]: (b.get("market_prob") or 0) for b in bins})
        print(f"  top-P bin:  Our={our_top}   Mkt={mkt_top}   TWC={twc_top}")
        print(f"  {'bin':<14} {'Our P':>7} {'Mkt P':>7} {'TWC P':>7} "
              f"{'TWC-Mkt':>9} {'TWC-Our':>9}")
        print(f"  " + "-" * 64)
        for b in bins:
            lbl = b["label"]
            our_p = float(b.get("our_prob")    or 0.0)
            mkt_p = float(b.get("market_prob") or 0.0)
            twc_p = twc_probs.get(lbl, 0.0)
            marker = "*" if lbl == twc_top else ""
            print(f"  {lbl:<14} {our_p*100:>6.1f}% {mkt_p*100:>6.1f}% "
                  f"{twc_p*100:>6.1f}% "
                  f"{(twc_p-mkt_p)*100:>+7.1f}pp "
                  f"{(twc_p-our_p)*100:>+7.1f}pp {marker}")
        return {"city": city, "status": "ok", "mode": mode,
                "twc_top": twc_top, "mkt_top": mkt_top, "our_top": our_top,
                "twc_mean": mean,
                "det_max": det_max, "det_source": det_source,
                "confidence": conf_info.get("confidence", 0.0),
                "det_bin_label": conf_info.get("det_bin_label"),
                "mode_bin_label": conf_info.get("mode_bin_label"),
                "mode_bin_prob": conf_info.get("mode_bin_prob"),
                "unit_sym": unit_sym,
                "nowcast_peak": nowcast.get("peak_temp"),
                "nowcast_status": nowcast.get("status"),
                "intraday_spread": intraday.get("spread"),
                "intraday_verdict": intraday.get("verdict")}
    else:
        # TWC-only: just show synthesized bins + TWC P
        print(f"  top-P bin:  TWC={twc_top}  (no DB bins to compare)")
        print(f"  {'bin (synth)':<14} {'TWC P':>7}")
        print(f"  " + "-" * 26)
        for b in bins:
            lbl = b["label"]
            twc_p = twc_probs.get(lbl, 0.0)
            marker = "*" if lbl == twc_top else ""
            # Skip near-zero bins to keep output compact
            if twc_p < 0.005:
                continue
            print(f"  {lbl:<14} {twc_p*100:>6.1f}% {marker}")
        return {"city": city, "status": "ok", "mode": mode,
                "twc_top": twc_top, "mkt_top": None, "our_top": None,
                "twc_mean": mean,
                "det_max": det_max, "det_source": det_source,
                "confidence": conf_info.get("confidence", 0.0),
                "det_bin_label": conf_info.get("det_bin_label"),
                "mode_bin_label": conf_info.get("mode_bin_label"),
                "mode_bin_prob": conf_info.get("mode_bin_prob"),
                "unit_sym": unit_sym,
                "nowcast_peak": nowcast.get("peak_temp"),
                "nowcast_status": nowcast.get("status"),
                "intraday_spread": intraday.get("spread"),
                "intraday_verdict": intraday.get("verdict")}


# ============================================================
# Main
# ============================================================

def main(argv: Optional[list] = None) -> int:
    today_iso = date_t.today().isoformat()
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=DB_PATH,
                       help="path to signals.db (default: config.DB_PATH)")
    ap.add_argument("--event-date", default=today_iso,
                       help=f"YYYY-MM-DD (default: today = {today_iso})")
    ap.add_argument("--city", default=None,
                       help="single-city override.  When set, --scope is "
                            "ignored and only this city is probed.")
    ap.add_argument("--scope", choices=["domestic", "international", "all"],
                       default="all",
                       help="which station set to probe (default: all).  "
                            "'domestic' = continental US (K-prefixed ICAOs); "
                            "'international' = everything else; "
                            "'all' = both.")
    ap.add_argument("--no-fusion", action="store_true",
                       help="Skip the observed-max floor fusion entirely.  "
                            "By default the script fuses with temperatureMax"
                            "Since7Am when event_date is today.  Use this to "
                            "see the raw prototype-derived distribution.")
    ap.add_argument("--n-prototypes", type=int, default=N_PROTOTYPES_DEFAULT,
                       help=(f"how many prototypes TWC returns per station "
                             f"(default: {N_PROTOTYPES_DEFAULT}, hard cap: "
                             f"{N_PROTOTYPES_MAX} -- the TWC Akamai-edge "
                             f"limit verified 2026-06-19).  At N=100 the "
                             f"Monte Carlo 95%% CI on a p=30%% bin is ~9pp. "
                             f"Requests above the cap return a cached 503."))
    args = ap.parse_args(argv)
    if args.n_prototypes > N_PROTOTYPES_MAX:
        print(f"FATAL: --n-prototypes={args.n_prototypes} exceeds the TWC "
              f"Akamai-edge cap of {N_PROTOTYPES_MAX}.  Use a value <= "
              f"{N_PROTOTYPES_MAX}.", file=sys.stderr)
        return 1

    if not os.path.exists(args.db):
        print(f"FATAL: DB not found at {args.db}", file=sys.stderr)
        return 1
    if not TWC_API_KEY:
        print("FATAL: TWC_API_KEY env var not set.  Add it to .env or export "
              "it, then re-run.", file=sys.stderr)
        return 1

    if args.city:
        cities = [args.city]
        scope_desc = f"single city: {args.city}"
    else:
        cities = filter_cities_by_scope(CITY_STATIONS, args.scope)
        scope_desc = f"scope={args.scope}"

    print(f"=== TWC probabilistic forecast probe ===")
    print(f"event_date: {args.event_date}")
    print(f"{scope_desc}: {len(cities)} cities ({', '.join(cities)})")
    print(f"horizon: {HOURS_DEFAULT}h forecast, {args.n_prototypes} prototypes per station")

    summaries: list[dict] = []
    with sqlite3.connect(args.db, timeout=30.0) as conn:
        for city in cities:
            s = probe_city(conn, city, args.event_date,
                              n_prototypes=args.n_prototypes,
                              no_fusion=args.no_fusion)
            summaries.append(s)

    # Final overview
    print()
    print("=" * 72)
    print("OVERVIEW")
    print("=" * 72)
    print(f"{'city':<14} {'mode':<10} {'TWC top-P':<12} "
          f"{'Mkt top-P':<12} {'agree?':<8}")
    print("-" * 72)
    for s in summaries:
        mkt_top = s.get("mkt_top")
        twc_top = s.get("twc_top")
        if s.get("status") != "ok":
            agree = ""
        elif mkt_top is None:
            agree = "n/a"   # twc_only mode — nothing to compare
        elif twc_top == mkt_top:
            agree = "yes"
        else:
            agree = "NO"
        mode = s.get("mode") or s.get("status", "")
        print(f"{s['city']:<14} {mode:<10} "
              f"{(twc_top or '--'):<12} "
              f"{(mkt_top or '--'):<12} "
              f"{agree:<8}")

    # --- Confidence ranking (Option A: P(bin containing deterministic)) ---
    rankable = [s for s in summaries if s.get("status") == "ok"
                  and s.get("confidence") is not None
                  and s.get("det_max") is not None]
    rankable.sort(key=lambda s: -float(s.get("confidence") or 0.0))
    print()
    print("=" * 112)
    print("CITIES BY FORECAST CONFIDENCE (most → least)")
    print("  confidence = P(probabilistic bin containing TWC's deterministic forecast)")
    print("  intraday   = |15-min nowcast peak − probabilistic P50|  "
          "(tight ≤ 1°F/0.5°C; diverged > 2°F/1°C)")
    print("=" * 112)
    print(f"{'city':<14} {'det max':>8} {'det bin':<12} "
          f"{'mode bin':<12} {'mode P':>7} {'conf':>7}   "
          f"{'nowcast':>8} {'spread':>7}  {'intraday':<9}  {'src'}")
    print("-" * 112)
    for s in rankable:
        usym = s.get("unit_sym", "")
        det_max = s.get("det_max")
        conf = float(s.get("confidence") or 0.0)
        mode_p = float(s.get("mode_bin_prob") or 0.0)
        marker = "✓" if conf >= 0.50 else (
            "●" if conf >= 0.30 else "⚠")
        det_str = f"{det_max:.1f}{usym}" if det_max is not None else "--"
        src = s.get("det_source", "")
        # cday = calendarDayTemperatureMax (best — matches settlement,
        #        persists all day);  7am = temperatureMax fallback (goes
        #        null after 3PM LAT);  p50 = probabilistic median fallback
        if "calendarDayTemperatureMax" in src:
            src_short = "cday"
        elif "temperatureMax" in src:
            src_short = "7am"
        elif "15day" in src:
            src_short = "ind"
        else:
            src_short = "p50"

        nc_peak = s.get("nowcast_peak")
        nc_str = f"{nc_peak:.1f}{usym}" if nc_peak is not None else "--"
        spr = s.get("intraday_spread")
        spr_str = f"{spr:.2f}{usym}" if spr is not None else "--"
        verdict = s.get("intraday_verdict") or "-"
        v_sym = {"tight": "✓", "loose": "●",
                  "diverged": "⚠", "no_nowcast": "·"}.get(verdict, "·")
        verdict_disp = f"{verdict} {v_sym}"

        print(f"{s['city']:<14} {det_str:>8} "
              f"{(s.get('det_bin_label') or '--'):<12} "
              f"{(s.get('mode_bin_label') or '--'):<12} "
              f"{mode_p*100:>6.1f}% {conf*100:>6.1f}% {marker}  "
              f"{nc_str:>8} {spr_str:>7}  {verdict_disp:<9}  {src_short}")
    if not rankable:
        print("  (no cities with successful TWC fetch)")

    return 0


if __name__ == "__main__":
    sys.exit(main())