"""
probe_backtest_apis.py — Verify API coverage before committing to the full
backtest backfill.

Checks, for 3 representative cities (Chicago, Phoenix, Miami) at leads
[1, 7, 30, 60, 90] days in the past:

  1. Open-Meteo Previous Runs  — ensemble member arrays available?
                                 hourly-forecast-path available?
  2. Visual Crossing Timeline — past-date hourly obs returning source='obs'?

Outputs a compact matrix so you can tell at a glance the backtest window
that will actually work.  Costs ~15 VC queryCost total (5 days x 3 cities).

    python bot/scripts/probe_backtest_apis.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

# Importing config loads .env (side-effect) — ensures VISUAL_CROSSING_API_KEY
# is available to visualcrossing._api_key().
import config  # noqa: F401

import httpx

from visualcrossing import fetch_daily_history, _get_vc   # reuse retry/auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("probe")

logging.getLogger("httpx").setLevel(logging.WARNING)

CITIES = [
    ("Chicago", 41.85,  -87.65),
    ("Phoenix", 33.45, -112.07),
    ("Miami",   25.77,  -80.19),
]

LEAD_DAYS = [1, 7, 30, 60, 90]

OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_PREVIOUS = "https://previous-runs-api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Open-Meteo Previous Runs probes
# ---------------------------------------------------------------------------

def probe_previous_runs_ensemble(lat: float, lon: float, target_date: date) -> dict[str, Any]:
    """
    Check whether ensemble-member daily_max arrays are still available for a
    past target date on the standard ensemble endpoint.
    """
    try:
        r = httpx.get(
            OPENMETEO_ENSEMBLE,
            params={
                "latitude":   lat,
                "longitude":  lon,
                "daily":      "temperature_2m_max",
                "start_date": target_date.isoformat(),
                "end_date":   target_date.isoformat(),
                "models":     "ecmwf_ifs04",
            },
            timeout=20,
        )
        if r.status_code >= 400:
            return {"ok": False, "n_members": 0, "error": f"HTTP {r.status_code}: {r.text[:80]}"}
        daily = r.json().get("daily", {}) or {}
    except Exception as e:
        return {"ok": False, "n_members": 0, "error": f"{type(e).__name__}: {str(e)[:80]}"}

    members = [
        k for k in daily
        if k.startswith("temperature_2m_max_member")
        and isinstance(daily.get(k), list)
        and daily[k]
        and daily[k][0] is not None
    ]
    if not members:
        top_keys = list(daily.keys())[:4]
        return {"ok": False, "n_members": 0,
                "error": f"no members; keys={top_keys}"}
    return {"ok": True, "n_members": len(members), "error": None}


def probe_previous_runs_hourly(lat: float, lon: float, target_date: date) -> dict[str, Any]:
    """
    Check whether the Previous Runs hourly path is available for a past
    target date.
    """
    try:
        r = httpx.get(
            OPENMETEO_PREVIOUS,
            params={
                "latitude":   lat,
                "longitude":  lon,
                "hourly":     "temperature_2m",
                "start_date": target_date.isoformat(),
                "end_date":   target_date.isoformat(),
                "models":     "ecmwf_ifs04",
            },
            timeout=20,
        )
        if r.status_code >= 400:
            return {"ok": False, "n_hours": 0, "total_ts": 0,
                    "error": f"HTTP {r.status_code}: {r.text[:80]}"}
        h = r.json().get("hourly", {}) or {}
    except Exception as e:
        return {"ok": False, "n_hours": 0, "total_ts": 0,
                "error": f"{type(e).__name__}: {str(e)[:80]}"}

    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    n_valid = sum(1 for t in temps if t is not None)
    if n_valid == 0:
        return {"ok": False, "n_hours": 0, "total_ts": len(times),
                "error": f"no temps; keys={list(h.keys())[:4]}"}
    return {"ok": True, "n_hours": n_valid, "total_ts": len(times), "error": None}


def probe_archive_hourly(lat: float, lon: float, target_date: date) -> dict[str, Any]:
    """
    Fallback: Open-Meteo Archive API (ERA5 reanalysis) for historical hourly.
    This is truth, not a forecast — useful as a backup ground-truth source.
    """
    try:
        r = httpx.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "hourly":     "temperature_2m",
                "start_date": target_date.isoformat(),
                "end_date":   target_date.isoformat(),
                "timezone":   "UTC",
            },
            timeout=20,
        )
        if r.status_code >= 400:
            return {"ok": False, "n_hours": 0, "error": f"HTTP {r.status_code}: {r.text[:80]}"}
        h = r.json().get("hourly", {}) or {}
    except Exception as e:
        return {"ok": False, "n_hours": 0, "error": f"{type(e).__name__}: {str(e)[:80]}"}

    temps = h.get("temperature_2m") or []
    n_valid = sum(1 for t in temps if t is not None)
    return {"ok": n_valid > 0, "n_hours": n_valid, "error": None}


# ---------------------------------------------------------------------------
# Visual Crossing probe
# ---------------------------------------------------------------------------

def probe_vc_hourly(lat: float, lon: float, target_date: date) -> dict[str, Any]:
    """
    Confirm VC returns source='obs' hourly data for a past date, and measure
    queryCost.
    """
    path = f"/{lat},{lon}/{target_date.isoformat()}"
    try:
        j = _get_vc(path, {
            "unitGroup": "metric",
            "include":   "hours",
            "elements":  "datetime,temp,source,stations",
        })
    except Exception as e:
        return {"ok": False, "n_obs_hours": 0, "cost": None, "error": str(e)[:120]}

    days = j.get("days") or []
    if not days:
        return {"ok": False, "n_obs_hours": 0, "cost": j.get("queryCost"), "error": "no days"}
    hours = days[0].get("hours") or []
    n_obs  = sum(1 for h in hours if h.get("source") == "obs")
    n_any  = sum(1 for h in hours if h.get("temp") is not None)
    return {
        "ok":           n_obs > 0,
        "n_obs_hours":  n_obs,
        "n_any_hours":  n_any,
        "cost":         j.get("queryCost"),
        "error":        None,
    }


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

def main() -> int:
    today = date.today()
    print()
    print(f"Probe date: {today.isoformat()}")
    print(f"Leads (days back): {LEAD_DAYS}")
    print(f"Cities: {[c[0] for c in CITIES]}")
    print()
    print("Each cell: status for each API at this (city, lead).  X = fail (see errors below table).")
    print("-" * 120)

    hdr = f"{'City':<8} {'Lead':>4}  "
    hdr += f"{'ENS (Ensemble API)':<22} {'HOURLY (PrevRuns)':<22} {'ARCHIVE hr':<14} {'VC HOURLY OBS':<22}"
    print(hdr)
    print("-" * 120)

    errors_log: list[str] = []

    vc_cost_total = 0
    rows_for_summary: list[dict] = []

    for name, lat, lon in CITIES:
        for lead in LEAD_DAYS:
            tgt = today - timedelta(days=lead)

            e = probe_previous_runs_ensemble(lat, lon, tgt)
            h = probe_previous_runs_hourly(lat, lon, tgt)
            a = probe_archive_hourly(lat, lon, tgt)
            v = probe_vc_hourly(lat, lon, tgt)

            if v["cost"]:
                vc_cost_total += int(v["cost"])

            def _cell(res, ok_fmt, width):
                if res["ok"]:
                    return ok_fmt.format(**res)[:width]
                return "X"

            ens_cell = _cell(e, "OK n={n_members}", 22)
            hr_cell  = _cell(h, "OK n={n_hours}/{total_ts}", 22)
            arc_cell = _cell(a, "OK n={n_hours}", 14)
            vc_cell  = _cell(v, "OK n={n_obs_hours} c={cost}", 22)

            print(f"{name:<8} {lead:>4}  {ens_cell:<22} {hr_cell:<22} {arc_cell:<14} {vc_cell:<22}")

            for tag, res in (("ens", e), ("hr", h), ("arc", a), ("vc", v)):
                if not res["ok"] and res.get("error"):
                    errors_log.append(f"  {name:<8} lead={lead:<3} {tag:<3} : {res['error']}")

            rows_for_summary.append({
                "city": name, "lead_days": lead, "date": tgt.isoformat(),
                "ensemble_ok": e["ok"], "ensemble_members": e["n_members"],
                "hourly_ok": h["ok"],   "hourly_count": h["n_hours"],
                "archive_ok": a["ok"],  "archive_count": a["n_hours"],
                "vc_ok": v["ok"],       "vc_obs_hours": v["n_obs_hours"],
                "vc_cost": v["cost"],
            })

    # -----------------------------------------------------------------
    # Summary verdict
    # -----------------------------------------------------------------
    print()
    print("-" * 100)
    print("VERDICT")
    print("-" * 100)

    by_lead: dict[int, dict] = {}
    for r in rows_for_summary:
        d = by_lead.setdefault(r["lead_days"], {"ens": 0, "hr": 0, "arc": 0, "vc": 0, "n": 0})
        d["n"]   += 1
        d["ens"] += int(r["ensemble_ok"])
        d["hr"]  += int(r["hourly_ok"])
        d["arc"] += int(r["archive_ok"])
        d["vc"]  += int(r["vc_ok"])

    print(f"{'Lead':>6}  {'ENS':>6} {'HOURLY':>8} {'ARCHIVE':>8} {'VC':>6}  (ratio of {len(CITIES)} cities)")
    for lead in LEAD_DAYS:
        d = by_lead[lead]
        print(f"{lead:>6}  {d['ens']}/{d['n']:<4} {d['hr']}/{d['n']:<6} {d['arc']}/{d['n']:<6} {d['vc']}/{d['n']:<4}")

    print()
    print(f"VC queryCost total: {vc_cost_total}")

    if errors_log:
        print()
        print("ERROR DETAILS")
        print("-" * 120)
        for line in errors_log:
            print(line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
