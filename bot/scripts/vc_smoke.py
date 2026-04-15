"""
vc_smoke.py — Visual Crossing smoke test.

Confirms the API key, connectivity, and response shape before the
VC integration is wired into the trading loop.  Safe to run repeatedly.

    python -m bot.scripts.vc_smoke
    (or)  python bot/scripts/vc_smoke.py
"""

import logging
import os
import sys
from datetime import date, timedelta

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from visualcrossing import fetch_intraday, fetch_daily_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
log = logging.getLogger("vc_smoke")


CITIES = [
    ("Chicago",  41.85,  -87.65),
    ("Phoenix",  33.45, -112.07),
]


def main() -> int:
    if not os.getenv("VISUAL_CROSSING_API_KEY"):
        log.error("VISUAL_CROSSING_API_KEY is not set")
        return 1

    total_cost = 0

    for name, lat, lon in CITIES:
        log.info(f"--- intraday: {name} ({lat},{lon}) ---")
        try:
            j = fetch_intraday(lat, lon)
        except Exception as e:
            log.error(f"{name} intraday failed: {e}")
            continue
        total_cost += int(j.get("query_cost") or 0)
        log.info(
            f"{name}: current_temp={j.get('current_temp_c')}°C "
            f"vc_source={j.get('current_vc_source') or '(n/a)'} "
            f"obs_hours={len(j.get('observed_hours') or [])} "
            f"fcst_hours={len(j.get('forecast_hours') or [])} "
            f"tz={j.get('timezone')} "
            f"queryCost={j.get('query_cost')}"
        )

    # History: last 3 days for Chicago
    name, lat, lon = CITIES[0]
    start = date.today() - timedelta(days=3)
    end   = date.today() - timedelta(days=1)
    log.info(f"--- daily history: {name} {start}..{end} ---")
    try:
        rows = fetch_daily_history(lat, lon, start, end)
        for r in rows:
            log.info(
                f"  {r['date']} tmax={r['tempmax_c']}°C "
                f"src={r['source']} stations={len(r['stations'])}"
            )
    except Exception as e:
        log.error(f"daily history failed: {e}")

    log.info(f"Smoke test complete. Intraday queryCost total: {total_cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
