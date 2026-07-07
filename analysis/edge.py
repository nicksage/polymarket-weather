"""Daily-max model + edge.

For each active Polymarket highest-temperature event, derive the distribution of
the day's MAXIMUM temperature from TWC's probabilistic `prototypes` (100 ensemble
member traces), then compute the model probability that the high lands in each
market bin and the edge versus the market's implied probability.

Method (statistically the right use of the ensemble):
  1. Take the latest prototypes(temperature) for the city — 100 hourly traces.
  2. Restrict to the resolution date's hours in the city's local timezone.
  3. Per trace, take the MAX over those hours -> 100 samples of the daily high.
  4. Convert to the market's unit (TWC is Celsius; US markets are Fahrenheit).
  5. model_prob(bin) = fraction of the 100 daily-high samples in the bin's
     continuous range (whole-degree label expanded by +/-0.5, as the market
     resolves to whole degrees).
  6. edge = model_prob - market_prob (latest yes_price = the tradeable price).

Writes/refreshes the `edge` table (one row per contract per active event).

Run against the live DB (default via config.env_loader DB_PATH) or any copy:
    python -m analysis.edge                      # DB_PATH from .env (EC2: main.db)
    DB_PATH=db/snapshot.db python -m analysis.edge
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config.env_loader import DB_PATH
from config.cities import CITY_TZ

EDGE_DDL = """
CREATE TABLE IF NOT EXISTS edge (
    event_id            TEXT,
    city                TEXT,
    date                TEXT,
    contract_id         TEXT,
    unit                TEXT,
    bin_low             REAL,
    bin_high            REAL,
    bin_label           TEXT,
    market_prob         REAL,   -- latest yes_price (tradeable implied prob)
    market_prob_norm    REAL,   -- normalised within the event (overround removed)
    model_prob          REAL,   -- P(daily high in bin) from the ensemble
    edge                REAL,   -- model_prob - market_prob
    dh_p10              REAL,   -- daily-high ensemble percentiles (market unit)
    dh_median           REAL,
    dh_p90              REAL,
    n_samples           INTEGER,
    hours_covered       INTEGER,-- hours of the resolution date present in the forecast
    forecast_fetched_at TEXT,
    price_recorded_at   TEXT,
    computed_at         TEXT
);
"""


def c_to_f(c):
    return c * 9 / 5 + 32


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))
    return sorted_vals[i]


def _bin_label(low, high):
    if low is None:
        return f"<={high:g}"
    if high is None:
        return f">={low:g}"
    if low == high:
        return f"{low:g}"
    return f"{low:g}-{high:g}"


def _daily_high_samples(proto_data, fcst_valid, tz, date):
    """Max over the resolution date's local hours, per ensemble trace (Celsius)."""
    idx = [i for i, t in enumerate(fcst_valid)
           if datetime.fromtimestamp(t, tz).strftime("%Y-%m-%d") == date]
    if not idx:
        return [], 0
    samples = []
    for tr in proto_data.get("forecast", []):
        vals = [tr[i] for i in idx if i < len(tr) and tr[i] is not None]
        if vals:
            samples.append(max(vals))
    return samples, len(idx)


def _model_prob(samples, lo, hi):
    lb = lo - 0.5 if lo is not None else -1e9
    ub = hi + 0.5 if hi is not None else 1e9
    return sum(1 for s in samples if lb < s <= ub) / len(samples)


def build_edge(con):
    now = datetime.now(timezone.utc).isoformat()
    con.execute("PRAGMA busy_timeout=15000")
    con.executescript(EDGE_DDL)
    con.execute("DELETE FROM edge")
    con.row_factory = sqlite3.Row

    events = con.execute(
        "SELECT event_id, city, date FROM events WHERE date >= date('now')"
    ).fetchall()

    proto_cache = {}
    rows, skipped = [], []
    for ev in events:
        eid, city, date = ev["event_id"], ev["city"], ev["date"]
        tzname = CITY_TZ.get(city)
        if not tzname:
            skipped.append(f"{city}(no tz)")
            continue
        if city not in proto_cache:
            proto_cache[city] = con.execute(
                """SELECT data, fcst_valid, fetched_at FROM twc_probabilistic
                   WHERE city = ? AND product = 'prototypes' AND parameter = 'temperature'
                   ORDER BY fetched_at DESC LIMIT 1""", (city,)).fetchone()
        pr = proto_cache[city]
        if not pr:
            skipped.append(f"{city}(no prototypes)")
            continue
        samples_c, hours = _daily_high_samples(
            json.loads(pr["data"]), json.loads(pr["fcst_valid"]), ZoneInfo(tzname), date)
        if not samples_c:
            skipped.append(f"{city} {date}(date not in forecast window)")
            continue

        bins = con.execute(
            "SELECT contract_id, range_low, range_high, unit FROM bins WHERE event_id = ?",
            (eid,)).fetchall()
        if not bins:
            continue
        unit = (bins[0]["unit"] or "celsius")
        to_f = unit.lower().startswith("f")
        samples = [c_to_f(s) for s in samples_c] if to_f else samples_c
        ssorted = sorted(samples)
        dh10, dh50, dh90 = _pct(ssorted, 10), _pct(ssorted, 50), _pct(ssorted, 90)

        # latest market price per contract + normalisation across the event
        priced = []
        for b in bins:
            pr_row = con.execute(
                """SELECT yes_price, recorded_at FROM price_snapshots
                   WHERE contract_id = ? ORDER BY recorded_at DESC LIMIT 1""",
                (b["contract_id"],)).fetchone()
            priced.append((b, pr_row["yes_price"] if pr_row else None,
                           pr_row["recorded_at"] if pr_row else None))
        total = sum(p for _, p, _ in priced if p is not None) or None
        last_price_at = max((rt for _, _, rt in priced if rt), default=None)

        for b, price, _rt in priced:
            lo, hi = b["range_low"], b["range_high"]
            mp = _model_prob(samples, lo, hi)
            norm = (price / total) if (price is not None and total) else None
            rows.append((
                eid, city, date, b["contract_id"], unit, lo, hi, _bin_label(lo, hi),
                price, norm, round(mp, 4),
                round(mp - price, 4) if price is not None else None,
                dh10, dh50, dh90, len(samples), hours,
                pr["fetched_at"], last_price_at, now,
            ))

    con.executemany(
        """INSERT INTO edge (event_id, city, date, contract_id, unit, bin_low, bin_high,
               bin_label, market_prob, market_prob_norm, model_prob, edge,
               dh_p10, dh_median, dh_p90, n_samples, hours_covered,
               forecast_fetched_at, price_recorded_at, computed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.commit()
    con.execute("CREATE INDEX IF NOT EXISTS idx_edge_ev ON edge(event_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_edge_edge ON edge(edge)")
    con.commit()
    return len(rows), len({r[0] for r in rows}), skipped


def main():
    ap = argparse.ArgumentParser(description="Build the daily-max edge table")
    ap.add_argument("--db", default=DB_PATH, help="SQLite DB path (default: env DB_PATH)")
    args = ap.parse_args()
    con = sqlite3.connect(args.db, timeout=30)
    n_rows, n_events, skipped = build_edge(con)
    con.close()
    print(f"edge: wrote {n_rows} rows across {n_events} events -> {args.db}")
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped[:12])}"
              + (" ..." if len(skipped) > 12 else ""))


if __name__ == "__main__":
    main()
