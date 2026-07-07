"""Starter comparison: Polymarket implied distribution vs the TWC forecast for a
city's highest-temperature event.

    python -m analysis.compare London

Market side: the latest yes_price per contract IS the market's implied
probability that the day's high lands in that bin.

Model side: TWC's hourly temperature forecast for the resolution day. The
forecast daily HIGH is the max hourly temperature over that local date; the
percentile spread at that peak hour (from the probabilistic `percentiles`
product) gives a rough uncertainty band.

Units: TWC is always collected in Celsius; US-city markets are in Fahrenheit,
so the model temperatures are converted to the market's unit before comparing.

NOTE (modelling caveat): the market is about the daily MAXIMUM, but the
probabilistic products give per-HOUR temperature distributions. Turning the
hourly distributions into a proper daily-max distribution (to get a rigorous
model probability per bin, and thus an edge) is the next modelling step — this
script only lines the two sides up so you can eyeball divergence.
"""
import json
import sys

import pandas as pd

from analysis.db import connect, q


def c_to_f(c):
    return c * 9 / 5 + 32


def bin_label(low, high):
    if pd.isna(low):
        return f"<={high:g}"
    if pd.isna(high):
        return f">={low:g}"
    if low == high:
        return f"{low:g}"
    return f"{low:g}-{high:g}"


def market_distribution(con, event_id):
    df = q(con, """
        SELECT b.range_low AS low, b.range_high AS high, b.unit,
               (SELECT ps.yes_price FROM price_snapshots ps
                WHERE ps.contract_id = b.contract_id
                ORDER BY ps.recorded_at DESC LIMIT 1) AS implied
        FROM bins b WHERE b.event_id = ?
        ORDER BY b.range_low IS NULL DESC, b.range_low
    """, (event_id,))
    df["label"] = [bin_label(l, h) for l, h in zip(df["low"], df["high"])]
    total = df["implied"].sum()
    df["implied_norm"] = df["implied"] / total if total else df["implied"]
    return df


def forecast_high(con, city, date, to_f=False):
    """Forecast daily high (max hourly temp on the local date) from twc_hourly,
    latest fetch, plus the percentile spread at that same peak hour. Returns
    values in Fahrenheit if to_f (TWC data is Celsius)."""
    hourly = q(con, """
        SELECT valid_time_utc, valid_time_local, temperature FROM twc_hourly
        WHERE city = ? AND fetched_at = (SELECT MAX(fetched_at) FROM twc_hourly WHERE city = ?)
          AND substr(valid_time_local, 1, 10) = ?
        ORDER BY temperature DESC
    """, (city, city, date))
    if hourly.empty:
        return None
    peak = hourly.iloc[0]
    peak_utc = int(peak["valid_time_utc"])
    spread = None
    pct = q(con, """
        SELECT data, fcst_valid FROM twc_probabilistic
        WHERE city = ? AND product = 'percentiles' AND parameter = 'temperature'
        ORDER BY fetched_at DESC LIMIT 1
    """, (city,))
    if not pct.empty:
        d = json.loads(pct.iloc[0]["data"])
        fv = json.loads(pct.iloc[0]["fcst_valid"])
        pts, vals = d["percentilePoints"], d["percentileValues"]
        if peak_utc in fv:                       # align the spread to the peak hour
            hv = vals[fv.index(peak_utc)]
            pick = lambda p: hv[pts.index(p)] if p in pts else None
            spread = {"p10": pick(10.0), "p50": pick(50.0), "p90": pick(90.0)}
    conv = c_to_f if to_f else (lambda x: x)
    cv = lambda x: round(conv(x), 1) if x is not None else None
    return {"high": cv(peak["temperature"]), "at": peak["valid_time_local"],
            "spread": {k: cv(v) for k, v in spread.items()} if spread else None}


def compare_event(con, city):
    ev = q(con, """SELECT event_id, city, date FROM events
                   WHERE city = ? AND date >= date('now') ORDER BY date LIMIT 1""", (city,))
    if ev.empty:
        opts = q(con, "SELECT DISTINCT city FROM events WHERE date>=date('now') ORDER BY city")
        print(f"No upcoming event for {city!r}. Try one of:\n  " + ", ".join(opts["city"]))
        return
    event_id, city, date = ev.iloc[0]["event_id"], ev.iloc[0]["city"], ev.iloc[0]["date"]
    print(f"\n{city} - highest temperature on {date}\n" + "=" * 46)

    mkt = market_distribution(con, event_id)
    unit = (mkt.iloc[0]["unit"] or "celsius")[:1].upper() if not mkt.empty else "?"
    to_f = unit == "F"
    print(f"\nMarket implied distribution ({unit}):")
    print(f"  {'bin':>8}  {'implied':>8}  {'norm':>7}")
    for _, r in mkt.iterrows():
        print(f"  {r['label']:>8}  {r['implied']:>8.3f}  {r['implied_norm']:>6.1%}")

    fc = forecast_high(con, city, date, to_f=to_f)
    print("\nTWC forecast:")
    if not fc:
        print("  no hourly forecast covering that local date yet.")
        return
    print(f"  forecast daily HIGH: {fc['high']:g}{unit}  (peak hour {fc['at']})")
    if fc["spread"]:
        s = fc["spread"]
        print(f"  peak-hour spread:    p10={s['p10']:g}  p50={s['p50']:g}  p90={s['p90']:g}")
    likely = mkt.loc[mkt["implied"].idxmax()]
    print(f"\n  market's most-likely bin: {likely['label']}{unit} ({likely['implied_norm']:.0%})")
    print(f"  forecast high {fc['high']:g}{unit} -> compare to the bins above.")
    print("\n  (daily-max distribution modelling still TODO - see module docstring)")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "London"
    compare_event(connect(), city)
