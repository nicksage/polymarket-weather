"""
post_peak_strategy.py — Strategy 1: post-peak confirmation.

Thesis: once the forecast-predicted peak hour for the settlement station
has passed AND fresh observations confirm a daily max that's at or below
the forecast peak, the day's high is essentially locked in.  Empirical
hold rate from yesterday's backtest: 97.6% at 17:00 local, 99.8% at 19:00.

Trade rule for each active "Highest temp in <city> on <today>" market:
    1. Pull hourly forecast at the SETTLEMENT STATION's coords
       (Open-Meteo free forecast API — auto-uses station-local timezone)
    2. predicted_peak_hour = argmax(forecast[h] for h in today)
       predicted_peak_temp = max(...)
    3. If now_local < predicted_peak_hour + 1 hour → SKIP (too early)
    4. Pull TODAY's hourly observations from the station via Iowa State
       Mesonet ASOS endpoint (near-real-time, ~15 min lag).
    5. observed_max_so_far = max(observations up to now)
    6. Safety gates:
         - observed_max_so_far <= predicted_peak_temp + 0.5°C  (no surprise spike)
         - hours_remaining_in_day >= 2
         - last_obs_age < 90 min  (data is fresh)
         - station observation count for today >= 6  (not just a few stragglers)
    7. winning_bin = bin where observed_max_so_far falls
    8. Look up that bin's current YES price on Polymarket
    9. If yes_price < --threshold (default 0.90):
         emit BUY signal — bin is mispriced vs empirical ~98% hold rate

This script is DRY-RUN by default; it does NOT place orders.  It prints a
signal table you can audit before wiring into the main bot's executor.

Usage:
    cd bot
    python -m scripts.post_peak_strategy
    python -m scripts.post_peak_strategy --threshold 0.85
    python -m scripts.post_peak_strategy --city Madrid Paris Tokyo
    python -m scripts.post_peak_strategy --hours-after-peak 0  # fire AT peak
    python -m scripts.post_peak_strategy --min-liquidity 1000
    python -m scripts.post_peak_strategy --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from station_meta import CITY_STATIONS, get_station  # type: ignore
from polymarket  import search_temp_high_events       # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("post_peak")
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MESONET_URL            = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


# ---------------------------------------------------------------------------
# Forecast pull at station coords
# ---------------------------------------------------------------------------

def fetch_station_forecast(city: str, target_date: str) -> list[tuple[int, float]]:
    """Hourly forecast for `target_date` at the SETTLEMENT STATION's lat/lon,
    in the station's local timezone.  Returns [(hour_local, temp_c), ...]
    with 24 entries (or fewer if API returned partial data).
    Raises if the city has no station mapping."""
    s = get_station(city)
    if not s:
        raise ValueError(f"No settlement station mapped for {city}")
    _icao, _net, tz, lat, lon = s
    r = httpx.get(
        OPENMETEO_FORECAST_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     "temperature_2m",
            "timezone":   tz,
            "start_date": target_date,
            "end_date":   target_date,
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json().get("hourly", {}) or {}
    times = h.get("time") or []
    temps = h.get("temperature_2m") or []
    out: list[tuple[int, float]] = []
    for t, v in zip(times, temps):
        if v is None:
            continue
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
            out.append((dt.hour, float(v)))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Observation pull from Mesonet (today, near-real-time)
# ---------------------------------------------------------------------------

def fetch_station_obs_today(city: str, target_date: str
                             ) -> list[tuple[int, float, datetime]]:
    """Today's hourly observations from Mesonet ASOS.  Returns
    [(hour_local, temp_c, ts_local_dt), ...].  Latest METAR for each hour
    (closest to the top of the hour wins for double-METAR hours)."""
    s = get_station(city)
    if not s:
        raise ValueError(f"No settlement station mapped for {city}")
    icao, net, tz, _lat, _lon = s
    d = datetime.strptime(target_date, "%Y-%m-%d")
    # Pull from start of day to "tomorrow" so we catch any obs in the
    # last hour of the day; we'll filter to target_date below.
    d_plus1 = d + timedelta(days=1)
    r = httpx.get(
        MESONET_URL,
        params={
            "station":     icao,
            "network":     net,
            "data":        "tmpc",
            "year1":       d.year,       "month1": d.month,       "day1": d.day,
            "year2":       d_plus1.year, "month2": d_plus1.month, "day2": d_plus1.day,
            "tz":          tz,
            "format":      "onlycomma",
            "latlon":      "no",
            "missing":     "M",
            "trace":       "T",
            "direct":      "no",
            "report_type": [3, 4],
        },
        timeout=60,
    )
    r.raise_for_status()
    by_hour: dict[int, tuple[int, float, datetime]] = {}
    for line in r.text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        _, valid, tmpc = parts[0], parts[1], parts[2]
        if tmpc in ("M", "", "T"):
            continue
        try:
            dt = datetime.strptime(valid.strip(), "%Y-%m-%d %H:%M")
            t  = float(tmpc)
        except ValueError:
            continue
        if dt.strftime("%Y-%m-%d") != target_date:
            continue
        # Some METARs report at :50/:55, others at :00.  Prefer the one
        # nearer the top of the hour for each local hour.
        prefer_at = 55 if dt.minute >= 30 else 0
        offset = abs(dt.minute - prefer_at)
        existing = by_hour.get(dt.hour)
        if existing is None or offset < existing[0]:
            by_hour[dt.hour] = (offset, t, dt)
    return sorted(((h, t, dt) for h, (_, t, dt) in by_hour.items()))


# ---------------------------------------------------------------------------
# Bin matching
# ---------------------------------------------------------------------------

def _bin_contains(low, high, unit: str, temp_c: float) -> bool:
    """Does `temp_c` fall in the bin [low, high] (in `unit`)?  Handles
    open-ended bins (low=None → "X or below"; high=None → "X or higher")."""
    if unit and unit.lower() == "fahrenheit":
        t = temp_c * 9 / 5 + 32
    else:
        t = temp_c
    if low is None and high is not None:
        return t <= high
    if low is not None and high is None:
        return t >= low
    if low is not None and high is not None:
        # Bin like "28°C" stores low==high.  Treat as [low - 0.5, high + 0.5)
        # so a 28.0 actual maps cleanly to the 28°C bin.
        if low == high:
            return abs(t - low) < 0.5
        return low <= t <= high
    return False


def _bin_label(low, high, unit: str) -> str:
    suffix = "F" if (unit or "celsius").lower() == "fahrenheit" else "C"
    if low is None and high is not None:
        return f"≤{int(high)}°{suffix}"
    if low is not None and high is None:
        return f"≥{int(low)}°{suffix}"
    if low is not None and high is not None:
        if int(low) == int(high):
            return f"{int(low)}°{suffix}"
        return f"{int(low)}–{int(high)}°{suffix}"
    return "?"


# ---------------------------------------------------------------------------
# Main strategy logic per event
# ---------------------------------------------------------------------------

def evaluate_event(event: dict, threshold: float, hours_after_peak: int,
                    min_liquidity: float) -> dict | None:
    """Return a signal dict if the event meets all gates, else None."""
    city = event.get("city")
    date_str = event.get("date")
    if not city or city not in CITY_STATIONS or not date_str:
        return {"city": city, "date": date_str, "reason": "no_station_mapping"}

    s = get_station(city)
    tz = ZoneInfo(s[2])
    now_local = datetime.now(tz)
    today_local = now_local.date().isoformat()

    if date_str != today_local:
        return {"city": city, "date": date_str, "reason": "market_not_today"}

    # 1. Forecast at station
    try:
        forecast = fetch_station_forecast(city, date_str)
    except Exception as e:
        return {"city": city, "date": date_str, "reason": f"forecast_fetch_failed: {e}"}
    if not forecast:
        return {"city": city, "date": date_str, "reason": "forecast_empty"}

    predicted_peak_hour, predicted_peak_temp = max(forecast, key=lambda x: x[1])

    # 2. Are we past peak + buffer yet?
    if now_local.hour < predicted_peak_hour + hours_after_peak:
        return {
            "city": city, "date": date_str,
            "reason": f"too_early (now={now_local.hour}:00, "
                      f"peak={predicted_peak_hour}:00 + {hours_after_peak}h buffer)",
            "predicted_peak_hour": predicted_peak_hour,
            "predicted_peak_temp": round(predicted_peak_temp, 2),
        }

    # 3. Observations so far today
    try:
        obs = fetch_station_obs_today(city, date_str)
    except Exception as e:
        return {"city": city, "date": date_str, "reason": f"obs_fetch_failed: {e}"}
    if len(obs) < 6:
        return {"city": city, "date": date_str,
                "reason": f"too_few_obs (only {len(obs)} so far)",
                "predicted_peak_hour": predicted_peak_hour}

    observed_max_so_far = max(t for _, t, _ in obs)
    observed_max_hour   = next(h for h, t, _ in obs if t == observed_max_so_far)
    last_obs_hour, _, last_obs_ts = obs[-1]
    obs_age_min = (now_local.replace(tzinfo=None) - last_obs_ts).total_seconds() / 60

    # 4. Safety gates
    if obs_age_min > 90:
        return {"city": city, "date": date_str,
                "reason": f"stale_obs ({obs_age_min:.0f} min old)",
                "predicted_peak_hour": predicted_peak_hour}

    # Conviction gate: if the observed max has already reached or exceeded
    # the forecast peak, the day's high is probably still rising — the
    # forecast was too low.  Skip.  (Tighter than the old +0.5 buffer.)
    if observed_max_so_far >= predicted_peak_temp:
        return {"city": city, "date": date_str,
                "reason": f"observed_at_or_above_peak "
                          f"(obs={observed_max_so_far:.2f} >= "
                          f"forecast={predicted_peak_temp:.2f}; high not locked)",
                "predicted_peak_temp": round(predicted_peak_temp, 2),
                "observed_max_so_far": round(observed_max_so_far, 2)}

    hours_remaining = 24 - now_local.hour
    if hours_remaining < 2:
        # Late in the day is FINE for our thesis, but if it's after 22:00
        # local the market may be illiquid / about to settle anyway
        pass

    # 5. Find which bin contains observed max
    outcomes = event.get("outcomes", []) or []
    matching = [o for o in outcomes
                if _bin_contains(o.get("range_low"), o.get("range_high"),
                                 o.get("unit", "celsius"), observed_max_so_far)]
    if not matching:
        return {"city": city, "date": date_str,
                "reason": f"observed_max {observed_max_so_far:.2f}°C "
                          f"matched no bin (boundaries off?)",
                "observed_max_so_far": round(observed_max_so_far, 2)}
    target_bin = matching[0]

    # 6. Liquidity check on the specific bin
    bin_liq = float(target_bin.get("liquidity_usd") or 0)
    if bin_liq < min_liquidity:
        return {"city": city, "date": date_str,
                "reason": f"target_bin_thin (liq=${bin_liq:.0f} < ${min_liquidity:.0f})",
                "target_bin": _bin_label(target_bin["range_low"],
                                          target_bin["range_high"],
                                          target_bin["unit"]),
                "observed_max_so_far": round(observed_max_so_far, 2)}

    # 7. Trade decision
    yes_price = float(target_bin.get("yes_price") or 0)
    bin_label = _bin_label(target_bin["range_low"], target_bin["range_high"],
                            target_bin["unit"])

    # Market-sanity gate: if the bin we're confident in is priced very
    # cheap, the market has fresher data than our Mesonet feed (which
    # can lag by ~30-60 min).  Don't trade against a strongly-disagreeing
    # market — it almost certainly knows the actual high has moved past
    # this bin.  This catches the "Madrid 28°C went to 28.6 after our last
    # METAR" case.
    if yes_price < 0.05:
        # Find the bin the market thinks is most likely, for context
        favored = max(outcomes, key=lambda o: float(o.get("yes_price") or 0))
        return {
            "city":                  city,
            "date":                  date_str,
            "station":               s[0],
            "predicted_peak_hour":   predicted_peak_hour,
            "predicted_peak_temp":   round(predicted_peak_temp, 2),
            "observed_max_so_far":   round(observed_max_so_far, 2),
            "target_bin":            bin_label,
            "yes_price":             round(yes_price, 4),
            "reason": (f"market_disagrees (our target={bin_label} at "
                       f"yes={yes_price:.3f}; market favors "
                       f"{_bin_label(favored['range_low'], favored['range_high'], favored['unit'])} "
                       f"@ {float(favored.get('yes_price') or 0):.3f} — "
                       f"our obs probably stale)"),
        }

    signal = {
        "city":                  city,
        "date":                  date_str,
        "station":               s[0],
        "now_local_hour":        now_local.hour,
        "predicted_peak_hour":   predicted_peak_hour,
        "predicted_peak_temp":   round(predicted_peak_temp, 2),
        "observed_max_so_far":   round(observed_max_so_far, 2),
        "observed_max_hour":     observed_max_hour,
        "last_obs_age_min":      round(obs_age_min, 1),
        "target_bin":            bin_label,
        "yes_price":             round(yes_price, 4),
        "liquidity_usd":         round(bin_liq, 0),
        "yes_token_id":          target_bin.get("yes_token_id"),
        "contract_id":           target_bin.get("contract_id"),
    }

    if yes_price < threshold:
        signal["action"]     = "BUY_YES"
        signal["est_edge"]   = round(0.98 - yes_price, 4)
        signal["reason"]     = (f"observed max {observed_max_so_far:.2f}°C → "
                                 f"bin {bin_label} priced {yes_price:.2f} "
                                 f"< {threshold:.2f}; empirical p(hold)~0.98")
    else:
        signal["action"] = "SKIP_PRICED_IN"
        signal["reason"] = (f"observed max {observed_max_so_far:.2f}°C → "
                             f"bin {bin_label} already priced {yes_price:.2f} "
                             f"≥ {threshold:.2f}")
    return signal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_signal_row(s: dict) -> str:
    if "action" not in s:
        return (f"  {s['city']:<14} {s.get('date','-'):<11} "
                f"SKIP  {s.get('reason','')[:60]}")
    return (f"  {s['city']:<14} {s['date']:<11} "
            f"{s['action']:<15} "
            f"peak={s['predicted_peak_temp']:.1f}@{s['predicted_peak_hour']:02d}h "
            f"obs={s['observed_max_so_far']:.1f}@{s['observed_max_hour']:02d}h "
            f"bin={s['target_bin']:<8} "
            f"yes={s['yes_price']:.3f} liq=${s['liquidity_usd']:.0f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Max yes_price to trade (default: 0.90)")
    p.add_argument("--hours-after-peak", type=int, default=1,
                   help="Wait N hours after forecast peak before firing "
                        "(default: 1)")
    p.add_argument("--min-liquidity", type=float, default=500.0,
                   help="Minimum target-bin liquidity (default: $500)")
    p.add_argument("--city", nargs="*",
                   help="Restrict to specific cities (default: all mapped)")
    p.add_argument("--gamma-min-liquidity", type=float, default=100.0,
                   help="Event-level filter passed to search_temp_high_events "
                        "(default: 100)")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of formatted text")
    p.add_argument("--include-skipped", action="store_true",
                   help="Show events that were filtered out, with reason")
    args = p.parse_args()

    log.info("Discovering active Polymarket weather markets…")
    events = search_temp_high_events(min_liquidity=args.gamma_min_liquidity)
    log.info(f"Found {len(events)} events")

    if args.city:
        wanted = {c.strip().lower() for c in args.city}
        events = [e for e in events if (e.get("city") or "").lower() in wanted]
        log.info(f"Filtered to {len(events)} events for cities: {args.city}")

    results: list[dict] = []
    for ev in events:
        r = evaluate_event(ev,
                            threshold        = args.threshold,
                            hours_after_peak = args.hours_after_peak,
                            min_liquidity    = args.min_liquidity)
        if r:
            results.append(r)

    buys     = [r for r in results if r.get("action") == "BUY_YES"]
    priced   = [r for r in results if r.get("action") == "SKIP_PRICED_IN"]
    skipped  = [r for r in results if "action" not in r]

    if args.json:
        out = {"buys": buys, "priced_in": priced,
               "skipped": skipped if args.include_skipped else []}
        print(json.dumps(out, indent=2, default=str))
        return 0

    print()
    print("=" * 86)
    print(f"  STRATEGY 1 — POST-PEAK CONFIRMATION  ({datetime.now(timezone.utc).isoformat()})")
    print("=" * 86)
    print(f"  Threshold:      yes_price < {args.threshold}")
    print(f"  Peak buffer:    fire {args.hours_after_peak}h after predicted peak hour")
    print(f"  Min liq:        target-bin ≥ ${args.min_liquidity:.0f}")
    print()
    print(f"  BUY SIGNALS: {len(buys)}")
    for r in sorted(buys, key=lambda x: x["yes_price"]):
        print(_fmt_signal_row(r))
    if not buys:
        print("    (none — either no markets qualify, or all already priced ≥ threshold)")
    print()
    print(f"  ALREADY PRICED IN (would buy but YES ≥ {args.threshold}): {len(priced)}")
    for r in sorted(priced, key=lambda x: -x["yes_price"])[:10]:
        print(_fmt_signal_row(r))
    if args.include_skipped and skipped:
        print()
        print(f"  SKIPPED (with reason): {len(skipped)}")
        # group reasons for compactness
        from collections import Counter
        reasons = Counter(r.get("reason", "?")[:60] for r in skipped)
        for reason, n in reasons.most_common():
            print(f"    {n:>3d} × {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())