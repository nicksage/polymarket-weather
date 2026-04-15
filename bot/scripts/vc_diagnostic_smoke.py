"""
vc_diagnostic_smoke.py — Dry-run the VC diagnostic paths end to end.

1. Calls fetch_intraday() and fetch_future_day() for 1 city
2. Runs them through _build_vc_diagnostic_row()
3. Prints the resulting diagnostic row WITHOUT writing to DB
4. Estimates daily VC queryCost at current active-event count

Safe to run any time.  Spends ~48 VC queryCost (2 calls).

    python bot/scripts/vc_diagnostic_smoke.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

import config  # noqa: F401  (ensures .env loads)
from loops import _build_vc_diagnostic_row, _get_event_state
from visualcrossing import fetch_future_day, fetch_intraday

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("smoke")
logging.getLogger("httpx").setLevel(logging.WARNING)


TEST_CITY = "Chicago"
LAT, LON = 41.85, -87.65


def _dump(label: str, d: dict) -> None:
    print(f"\n--- {label} ---")
    for k in sorted(d.keys()):
        if k == "vc_hourly_forecast_json":
            v = d[k]
            print(f"  {k}: {'(present, %d chars)' % len(v) if v else '(none)'}")
        else:
            print(f"  {k}: {d[k]}")


def main() -> int:
    print(f"VC diagnostic smoke — {TEST_CITY} ({LAT},{LON})")
    print(f"Thresholds: large={config.VC_DIAG_DISAGREE_LARGE_C}°C  "
          f"warn={config.VC_DIAG_WARN_DIRECTIONAL_C}°C  "
          f"future_horizon={config.VC_DIAG_FUTURE_DAY_HORIZON}  "
          f"bin_width={config.VC_DIAG_BIN_WIDTH_C}°C")

    pulled_at = datetime.now(timezone.utc).isoformat()

    # Same-day path — reuses fetch_intraday
    print("\n[1/2] fetch_intraday()...")
    same = fetch_intraday(LAT, LON)
    print(f"  queryCost={same.get('query_cost')}  "
          f"day_max={same.get('day_tempmax_estimate_c')}°C  "
          f"current={same.get('current_temp_c')}°C")

    same_diag = _build_vc_diagnostic_row(
        event_id      = "smoke-same-day-fake",
        city          = TEST_CITY,
        target_date   = date.today().isoformat(),
        pulled_at_utc = pulled_at,
        kind          = "same_day",
        vc_payload    = same,
        event_state   = None,   # No live event in DB for this smoke city
    )
    if same_diag:
        _dump("same_day diagnostic (NOT written to DB)", same_diag)
    else:
        print("  (no diagnostic produced — VC did not return day_tempmax)")

    # Future-day path
    print(f"\n[2/2] fetch_future_day({date.today() + timedelta(days=1)})...")
    fut = fetch_future_day(LAT, LON, date.today() + timedelta(days=1))
    print(f"  queryCost={fut.get('query_cost')}  "
          f"day_max={fut.get('day_tempmax_estimate_c')}°C  "
          f"fcst_hours={len(fut.get('forecast_hours') or [])}")

    fut_diag = _build_vc_diagnostic_row(
        event_id      = "smoke-future-day-fake",
        city          = TEST_CITY,
        target_date   = (date.today() + timedelta(days=1)).isoformat(),
        pulled_at_utc = pulled_at,
        kind          = "future_day",
        vc_payload    = fut,
        event_state   = None,
    )
    if fut_diag:
        _dump("future_day diagnostic (NOT written to DB)", fut_diag)
    else:
        print("  (no diagnostic produced)")

    # Cost projection
    print("\n--- cost projection ---")
    same_cost = int(same.get("query_cost") or 0)
    fut_cost  = int(fut.get("query_cost") or 0)
    print(f"  Per same-day call: {same_cost} queryCost "
          f"(piggybacks on existing 20-min loop; no new cost)")
    print(f"  Per future-day call: {fut_cost} queryCost")
    print(f"  Future-day cadence: every 2 hours -> 12 pulls/day")
    # Rough estimate based on 20 active future-day events
    est_events = 20
    est_daily = fut_cost * est_events * 12
    print(f"  Estimated daily cost at ~{est_events} future-day events: "
          f"{est_daily:,} queryCost")
    print(f"  (Current config: VC_DIAG_FUTURE_DAY_HORIZON="
          f"{config.VC_DIAG_FUTURE_DAY_HORIZON})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
