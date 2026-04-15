"""
Phase 0 — Verification queries before implementing bias correction + intraday.

Runs five checks:
  1. VC station coverage for each trading city (resolution station present?)
  2. VC `source` field split (observed vs forecast hours today)
  3. VC queryCost sanity check
  4. Open-Meteo Previous Runs ECMWF availability (coverage back to Jan 2024?)
  5. Open-Meteo Previous Runs response shape

Designed to run as a one-off, prints a clean pass/fail summary at the end.
"""
import os
import sys
import json
import httpx
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()
VC_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")
assert VC_KEY, "VISUAL_CROSSING_API_KEY not set in .env"

VC_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
OM_BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"

# City → (lat, lon, expected_icao_station)
CITIES = {
    "Amsterdam":   (52.37,  4.89,  "EHAM"),
    "London":      (51.51, -0.13,  "EGLL"),
    "Tokyo":       (35.68, 139.76, "RJTT"),
    "Seoul":       (37.57, 126.98, "RKSI"),
    "Chicago":     (41.88, -87.63, "KORD"),
    "Miami":       (25.76, -80.19, "KMIA"),
}

results = {"checks": [], "passed": 0, "failed": 0}


def _record(name: str, passed: bool, detail: str = ""):
    results["checks"].append({"name": name, "passed": passed, "detail": detail})
    if passed:
        results["passed"] += 1
    else:
        results["failed"] += 1
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}")
    if detail:
        print(f"         {detail}")


# ---------------------------------------------------------------------------
# Check 1 + 2 + 3 combined — one VC call per city gives all three
# ---------------------------------------------------------------------------
def check_vc_per_city():
    print("\n=== CHECK 1/2/3: VC station coverage, source-field split, queryCost ===")
    for city, (lat, lon, expected_icao) in CITIES.items():
        url = f"{VC_BASE}/{lat},{lon}/today"
        params = {
            "key": VC_KEY,
            "unitGroup": "metric",
            "include": "hours,current,days",
            "elements": "datetime,temp,tempmax,source,stations",
            "contentType": "json",
        }
        try:
            r = httpx.get(url, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            _record(f"VC fetch [{city}]", False, f"HTTP error: {e}")
            continue

        # --- Check 1: station coverage ---
        station_ids = list((j.get("stations") or {}).keys())
        has_station = expected_icao in station_ids
        day = j["days"][0]
        per_day_stations = day.get("stations") or []
        _record(
            f"VC station coverage [{city}]",
            has_station,
            f"expected={expected_icao}, top-level stations={station_ids}, "
            f"day-level stations={per_day_stations}",
        )

        # --- Check 2: source-field split on today's hours ---
        hours = day.get("hours") or []
        obs_hours  = [h for h in hours if h.get("source") == "obs"]
        fcst_hours = [h for h in hours if h.get("source") == "fcst"]
        other      = [h for h in hours if h.get("source") not in ("obs", "fcst")]
        clean_split = len(obs_hours) > 0 and len(fcst_hours) > 0 and len(other) == 0
        _record(
            f"VC obs/fcst hour split [{city}]",
            clean_split,
            f"obs={len(obs_hours)}, fcst={len(fcst_hours)}, other={len(other)} "
            f"(other values: {set(h.get('source') for h in other)})",
        )

        # --- Check 3: queryCost sanity ---
        cost = j.get("queryCost")
        reasonable_cost = cost is not None and 1 <= cost <= 50
        _record(
            f"VC queryCost [{city}]",
            reasonable_cost,
            f"queryCost={cost} for today-with-hours",
        )


# ---------------------------------------------------------------------------
# Check 4 — Open-Meteo Previous Runs ECMWF availability back to Jan 2024
# ---------------------------------------------------------------------------
def check_om_ecmwf_availability():
    print("\n=== CHECK 4: Open-Meteo Previous Runs ECMWF availability ===")
    # Test two dates: one recent, one early-2024
    for label, past_days in [("recent (30d ago)", 30), ("far (22 months ago)", 660)]:
        try:
            r = httpx.get(
                OM_BASE,
                params={
                    "latitude":  52.37,
                    "longitude":  4.89,
                    "models":    "ecmwf_ifs025",
                    "daily":     "temperature_2m_max,temperature_2m_max_previous_day3",
                    "past_days":     past_days,
                    "forecast_days": 0,
                    "timezone":  "UTC",
                },
                timeout=30,
            )
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            _record(f"OM ECMWF {label}", False, f"HTTP error: {e}")
            continue

        daily = j.get("daily") or {}
        times = daily.get("time") or []
        p3    = daily.get("temperature_2m_max_previous_day3") or []
        non_null = [v for v in p3 if v is not None]
        has_data = len(times) > 0 and len(non_null) > 0
        _record(
            f"OM ECMWF {label}",
            has_data,
            f"days returned={len(times)}, non-null previous_day3 values={len(non_null)}, "
            f"first_date={times[0] if times else 'n/a'}, "
            f"sample_value={non_null[0] if non_null else 'n/a'}",
        )


# ---------------------------------------------------------------------------
# Check 5 — Open-Meteo Previous Runs response shape + multi-model
# ---------------------------------------------------------------------------
def check_om_response_shape():
    print("\n=== CHECK 5: OM Previous Runs shape + multi-model ===")
    try:
        r = httpx.get(
            OM_BASE,
            params={
                "latitude":  41.88,
                "longitude": -87.63,
                "models":    "ecmwf_ifs025,gfs_global",
                "daily":     "temperature_2m_max,temperature_2m_max_previous_day1,"
                             "temperature_2m_max_previous_day3,temperature_2m_max_previous_day5",
                "past_days": 14,
                "forecast_days": 0,
                "timezone": "UTC",
            },
            timeout=30,
        )
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        _record("OM response shape", False, f"HTTP error: {e}")
        return

    daily_keys = list((j.get("daily") or {}).keys())
    # When multiple models requested, fields are typically suffixed with _ecmwf_ifs025 / _gfs_global
    has_both_models = (
        any("ecmwf" in k.lower() for k in daily_keys) and
        any("gfs"   in k.lower() for k in daily_keys)
    )
    # If not — maybe it only returns one model's data. Inspect and report.
    if not has_both_models:
        # Log all keys so we see what's actually returned
        _record(
            "OM multi-model response",
            False,
            f"daily keys returned: {daily_keys[:10]}... "
            f"(may need separate calls per model)",
        )
    else:
        _record(
            "OM multi-model response",
            True,
            f"ECMWF+GFS both present. daily keys: {daily_keys[:6]}...",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 0 Verification — Visual Crossing + Open-Meteo Previous Runs")
    print("=" * 70)

    check_vc_per_city()
    check_om_ecmwf_availability()
    check_om_response_shape()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {results['passed']} passed, {results['failed']} failed")
    print("=" * 70)

    if results["failed"] > 0:
        print("\nFailed checks:")
        for c in results["checks"]:
            if not c["passed"]:
                print(f"  - {c['name']}: {c['detail']}")

    sys.exit(0 if results["failed"] == 0 else 1)
