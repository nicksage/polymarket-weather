"""
failure_segmentation.py — Workstream 0 audit.

Three-axis join over the last N days of resolved Polymarket weather
markets, categorizing every settled market into one of:

  bot_caching_gap        Bot observed_max disagrees with mesonet METAR
                          archive by enough to flip top-P bin.  Cheap
                          engineering fix (cache/refresh window).

  settle_divergence      Mesonet METAR max bin != winning bin.  The
                          smoking gun: the bot's observation is correct
                          per the source feed, but the market settled
                          against a different value.  Implies the
                          observation->settlement mapping is mis-
                          specified (DSM aggregation, primary-site
                          assignment, missing peak cycle).  If
                          non-trivial, W1 changes from "caching fix" to
                          "remap every market's resolution source."

  boundary_shape_wrong   Mesonet bin == winning bin, but our top-P bin
                          was ADJACENT to the winner AND we gave the
                          winning bin < 30% probability.  Distribution
                          shape near the edge.  W2/W3 territory.

  magnitude_shape_wrong  Mesonet bin == winning bin, but our top-P bin
                          was NON-adjacent (2+ bins away).  sigma too
                          tight in the wrong direction.  W2.

  calibration_overconfident
                          We gave >= 50% probability to the winning bin
                          but bought a DIFFERENT bin.  Top-P-only buy
                          rule is throwing away signal.

Three axes:
  A: bot_observed_max     EOD max of paper_predictor_signals.observed_max_c
                          for (city, event_date).  What the BOT saw.
  B: mesonet_archive_max  Iowa State ASOS archive daily max for the ICAO.
                          The METAR feed AS ARCHIVED (different latency
                          path than the bot, same underlying source).
  C: settled_interval     From resolutions.winning_range_low/high mapped
                          through bin_temp_range() semantics.  The bin
                          the market resolved against.

Output:
  - CSV with one row per (city, event_date, event_id) and all three axes
  - per-category counts (text summary)
  - Offending-tuple list for settle_divergence cases (for W1 deep-dive)

Usage:
    cd bot
    python -m scripts.failure_segmentation --days 60 --out audit.csv

Required:
    --collector-db   Path to the backtest-collector resolutions DB
                       (default: ~/apps/weather-data/backtest-collector/data/prices.db)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_BOT_DIR), ".env"), override=True)
except ImportError:
    pass

from config import DB_PATH  # type: ignore
from scripts.intraday_predictor import bin_temp_range  # type: ignore
from station_meta import CITY_STATIONS  # type: ignore

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("failure_segmentation")

DEFAULT_COLLECTOR_DB = os.path.expanduser(
    "~/apps/weather-data/backtest-collector/data/prices.db")

# Iowa State ASOS archive endpoint.  Bulk download per station per
# date range — small CSV, free, no auth.  Same NOAA METAR feed but
# archived independently of our bot's intraday cache.
IOWA_STATE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Boundary-loss threshold: we gave the winning bin < this probability
# AND it was adjacent to our top-P bin → "shape was wrong near the edge"
BOUNDARY_PROB_THRESHOLD = 0.30
# Calibration-overconfident threshold: we gave the winner >= this AND
# bought elsewhere → top-P-only rule cost us
OVERCONFIDENT_PROB_THRESHOLD = 0.50


# ============================================================
# Axis B: mesonet archive fetch + parse
# ============================================================

def fetch_mesonet_daily_max(icao: str, start_date: date, end_date: date,
                              ) -> dict[str, float]:
    """Return {YYYY-MM-DD: max_temp_f} for the date range, computed from
    the Iowa State ASOS METAR archive.  Local-time aggregation uses the
    station's IANA timezone from CITY_STATIONS — matches how Polymarket
    settles "highest temp today" (local day boundaries).

    Single bulk HTTP call per station.  Returns {} on fetch failure.
    """
    # Find the timezone from CITY_STATIONS by reverse-looking-up the ICAO.
    tz_str = None
    for _city, meta in CITY_STATIONS.items():
        if meta[0] == icao:
            tz_str = meta[2]
            break
    if not tz_str:
        log.warning(f"no tz mapping for {icao} — skipping mesonet fetch")
        return {}

    # Pull one extra day on each side to handle TZ overlap properly
    params = {
        "station": icao.lstrip("K") if len(icao) == 4 and icao.startswith("K") else icao,
        # NB: Iowa State expects 3-letter station IDs for US stations
        # (KORD → ORD).  Strip the K prefix.
        "data": "tmpf",
        "year1": start_date.year,  "month1": start_date.month,  "day1": start_date.day,
        "year2": end_date.year,    "month2": end_date.month,    "day2": end_date.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "empty",
        "trace": "T",
        "direct": "no",
        "report_type": "3,4",  # 3=MADIS, 4=Routine + Special (METAR/SPECI)
    }
    url = IOWA_STATE_URL + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url,
                                       headers={"User-Agent": "polymarket-weather/audit"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        log.warning(f"mesonet fetch failed for {icao}: {e}")
        return {}

    # CSV header: station,valid,tmpf
    # valid is UTC ISO-ish: "2026-06-11 18:00"
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_str)
    by_local_date: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("station,"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        _stn, ts_utc_str, tmpf_str = parts[0], parts[1], parts[2]
        if not tmpf_str or tmpf_str.upper() in ("M", "MISSING", "T"):
            continue
        try:
            tmpf = float(tmpf_str)
            ts_utc = datetime.strptime(ts_utc_str, "%Y-%m-%d %H:%M")
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        local_date = ts_utc.astimezone(tz).date().isoformat()
        prev = by_local_date.get(local_date)
        if prev is None or tmpf > prev:
            by_local_date[local_date] = tmpf
    return by_local_date


# ============================================================
# Bin geometry: which bin contains a given temperature?
# ============================================================

def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def find_bin_index(temp_c: float, bins: list[dict]) -> Optional[int]:
    """Return the index of the bin whose Celsius interval contains temp_c,
    or None if no bin matches.  Bins is sorted by range_low ascending and
    each element is {range_low, range_high, unit}."""
    for i, b in enumerate(bins):
        lo_c, hi_c = bin_temp_range(b)
        if lo_c is not None and temp_c < lo_c:
            continue
        if hi_c is not None and temp_c >= hi_c:
            continue
        return i
    return None


# ============================================================
# Main audit
# ============================================================

def run_audit(signals_db: str, collector_db: str, days: int,
                out_csv: str, include_unwatched: bool) -> int:
    cutoff_date = (date.today() - timedelta(days=days)).isoformat()

    # ---------- 1. Pull resolutions in window ----------
    with sqlite3.connect(collector_db) as cdb:
        cdb.row_factory = sqlite3.Row
        rs_rows = cdb.execute(
            """SELECT event_id, city, date, winning_contract_id,
                       winning_range_low, winning_range_high, resolved_at
               FROM resolutions
               WHERE date >= ?
               ORDER BY date DESC""",
            (cutoff_date,),
        ).fetchall()
    log.info(f"resolutions in window: {len(rs_rows)}")
    if not rs_rows:
        log.error("no resolutions found — check --collector-db path and --days")
        return 1

    # ---------- 2. Pull bot signals for those events ----------
    event_ids = [r["event_id"] for r in rs_rows]
    placeholders = ",".join("?" for _ in event_ids)
    with sqlite3.connect(signals_db) as sdb:
        sdb.row_factory = sqlite3.Row
        sig_rows = sdb.execute(
            f"""SELECT scanned_at_utc, mode, city, settlement_station, event_date,
                       event_id, contract_id, yes_token_id, bin_label,
                       bin_range_low, bin_range_high, unit,
                       our_prob, market_prob, edge, action,
                       recommended_stake_usd, observed_max_c
                FROM paper_predictor_signals
                WHERE event_id IN ({placeholders})""",
            event_ids,
        ).fetchall()
    log.info(f"signal rows for those events: {len(sig_rows)}")

    # ---------- 3. Sanity check: winning_range_low/high are native ----------
    # Quick spot-check: pick a resolution and confirm the bin range matches
    # what's in signals for the same contract.
    cross_check_passed = 0
    for r in rs_rows[:50]:
        wcid = r["winning_contract_id"]
        match = next((s for s in sig_rows if s["contract_id"] == wcid), None)
        if match is None:
            continue
        if (match["bin_range_low"] is not None
            and abs(float(match["bin_range_low"]) - float(r["winning_range_low"]))
                <= 0.01):
            cross_check_passed += 1
    if cross_check_passed == 0:
        log.warning("CROSS-CHECK FAILED: no winning_range_low matched a signal bin_range_low "
                     "in 50 sampled rows.  winning_range_* may be in different units than "
                     "signals.  Continuing but boundary detection may be wrong.")
    else:
        log.info(f"cross-check OK: {cross_check_passed}/50 winning_range matched signal bins")

    # ---------- 4. Group signals by event, compute end-of-day state ----------
    sigs_by_event: dict[str, list[dict]] = defaultdict(list)
    for s in sig_rows:
        sigs_by_event[s["event_id"]].append(dict(s))

    # ---------- 5. Fetch mesonet archives per ICAO ----------
    icao_dates_needed: dict[str, set[str]] = defaultdict(set)
    for r in rs_rows:
        sigs = sigs_by_event.get(r["event_id"])
        if not sigs:
            if not include_unwatched:
                continue
        icao = None
        if sigs:
            for s in sigs:
                if s.get("settlement_station"):
                    icao = s["settlement_station"]
                    break
        if not icao:
            # Fall back: look up by city
            meta = CITY_STATIONS.get(r["city"])
            icao = meta[0] if meta else None
        if icao and r["date"]:
            icao_dates_needed[icao].add(r["date"])

    log.info(f"fetching mesonet archives for {len(icao_dates_needed)} stations")
    mesonet: dict[tuple[str, str], float] = {}
    for icao, dates in icao_dates_needed.items():
        dlist = sorted(dates)
        start = date.fromisoformat(dlist[0]) - timedelta(days=1)
        end   = date.fromisoformat(dlist[-1]) + timedelta(days=1)
        per_day = fetch_mesonet_daily_max(icao, start, end)
        for d, max_f in per_day.items():
            mesonet[(icao, d)] = max_f
        log.info(f"  {icao}: {len(per_day)} days fetched (requested span "
                  f"{start} to {end})")

    # ---------- 6. Per-resolution categorization ----------
    rows_out: list[dict] = []
    cat_counts: Counter = Counter()
    missing_data_count = 0
    for r in rs_rows:
        event_id   = r["event_id"]
        city       = r["city"]
        event_date = r["date"]
        wcid       = r["winning_contract_id"]
        wr_lo      = r["winning_range_low"]
        wr_hi      = r["winning_range_high"]
        sigs = sigs_by_event.get(event_id, [])
        if not sigs and not include_unwatched:
            missing_data_count += 1
            continue

        # Get the unit + bin set from signal rows
        unit = None
        for s in sigs:
            if s.get("unit"):
                unit = s["unit"]; break
        if unit is None:
            unit = "fahrenheit"  # default for US

        # Build the bin set: one row per distinct contract_id
        bins_for_event: dict[str, dict] = {}
        for s in sigs:
            cid = s["contract_id"]
            if cid not in bins_for_event:
                bins_for_event[cid] = {
                    "contract_id": cid,
                    "range_low":  s["bin_range_low"],
                    "range_high": s["bin_range_high"],
                    "unit":       s.get("unit") or unit,
                    "bin_label":  s.get("bin_label"),
                }
        bins_list = sorted(bins_for_event.values(),
                            key=lambda b: (b["range_low"] if b["range_low"] is not None
                                            else -9999))
        winning_bin_idx = next((i for i, b in enumerate(bins_list)
                                  if b["contract_id"] == wcid), None)

        # Axis A: bot observed max (EOD)
        bot_max_c = None
        for s in sigs:
            v = s.get("observed_max_c")
            if v is None: continue
            if bot_max_c is None or v > bot_max_c:
                bot_max_c = float(v)

        # ICAO for mesonet lookup
        icao = None
        for s in sigs:
            if s.get("settlement_station"):
                icao = s["settlement_station"]; break
        if not icao:
            meta = CITY_STATIONS.get(city)
            icao = meta[0] if meta else None

        # Axis B: mesonet archive max
        mes_max_f = mesonet.get((icao, event_date))
        mes_max_c = f_to_c(mes_max_f) if mes_max_f is not None else None

        # Axis C: settled interval (the bin that won)
        if winning_bin_idx is None:
            # Couldn't find the winning contract in our bin set — likely
            # the bot never evaluated this market.  Skip with a flag.
            missing_data_count += 1
            continue

        # Find which bin contains bot_max_c, mes_max_c
        bot_bin_idx = (find_bin_index(bot_max_c, bins_list)
                        if bot_max_c is not None else None)
        mes_bin_idx = (find_bin_index(mes_max_c, bins_list)
                        if mes_max_c is not None else None)

        # Our top-P bin from the LAST scan before resolution
        sigs_sorted = sorted(sigs, key=lambda s: s.get("scanned_at_utc") or "",
                              reverse=True)
        last_ts = sigs_sorted[0]["scanned_at_utc"] if sigs_sorted else None
        last_scan = [s for s in sigs_sorted
                      if s.get("scanned_at_utc") == last_ts]
        top_p_row = max(last_scan, key=lambda s: s.get("our_prob") or 0,
                         default=None)
        top_p_contract = top_p_row["contract_id"] if top_p_row else None
        top_p_idx = next((i for i, b in enumerate(bins_list)
                           if b["contract_id"] == top_p_contract), None)
        our_prob_at_winner = next(
            (s.get("our_prob") for s in last_scan if s["contract_id"] == wcid),
            None)

        # What did we BUY (if anything)?
        buy_rows = [s for s in sigs if s["action"] in ("LIVE_BUY", "PAPER_BUY")]
        bought_contracts = {b["contract_id"] for b in buy_rows}
        bought_winner = wcid in bought_contracts

        # ---------- categorization ----------
        category = None
        category_detail = ""
        if mes_bin_idx is None or bot_bin_idx is None:
            category = "missing_data"
            category_detail = ("no_bot_obs" if bot_max_c is None
                                else "no_mesonet_obs")
        elif (bot_bin_idx != mes_bin_idx):
            # Bot's parsed max disagrees with mesonet's parsed max ENOUGH
            # to fall in a different bin.  Caching/timing issue.
            category = "bot_caching_gap"
            category_detail = (f"bot_bin={bot_bin_idx}({bins_list[bot_bin_idx].get('bin_label')}) "
                                f"mes_bin={mes_bin_idx}({bins_list[mes_bin_idx].get('bin_label')})")
        elif mes_bin_idx != winning_bin_idx:
            # Bot AND mesonet agree, but the market settled to a different
            # bin.  Smoking gun.
            category = "settle_divergence"
            category_detail = (f"mes_bin={mes_bin_idx}({bins_list[mes_bin_idx].get('bin_label')}) "
                                f"win_bin={winning_bin_idx}({bins_list[winning_bin_idx].get('bin_label')}) "
                                f"mes_max_f={mes_max_f:.1f}" if mes_max_f else "")
        else:
            # Read was right.  Did our distribution put the winning bin
            # where it belonged?
            if top_p_idx is None:
                category = "missing_data"
                category_detail = "no_top_p"
            elif top_p_idx == winning_bin_idx:
                category = "model_correct"
                category_detail = f"prob_at_winner={our_prob_at_winner}"
            else:
                gap = abs(top_p_idx - winning_bin_idx)
                if (our_prob_at_winner is not None
                    and our_prob_at_winner >= OVERCONFIDENT_PROB_THRESHOLD
                    and not bought_winner):
                    category = "calibration_overconfident"
                    category_detail = (f"prob_at_winner={our_prob_at_winner:.2f} "
                                        f"but bought top_p instead")
                elif (gap == 1 and our_prob_at_winner is not None
                      and our_prob_at_winner < BOUNDARY_PROB_THRESHOLD):
                    category = "boundary_shape_wrong"
                    category_detail = (f"adjacent bin (gap=1) "
                                        f"prob_at_winner={our_prob_at_winner:.2f}")
                else:
                    category = "magnitude_shape_wrong"
                    category_detail = (f"gap={gap} bins, "
                                        f"prob_at_winner={our_prob_at_winner}")
        cat_counts[category] += 1

        bot_caching_gap_f = None
        if bot_max_c is not None and mes_max_c is not None:
            bot_caching_gap_f = abs(bot_max_c - mes_max_c) * 9.0 / 5.0

        adjacency = None
        if top_p_idx is not None and winning_bin_idx is not None:
            adjacency = abs(top_p_idx - winning_bin_idx)

        rows_out.append({
            "event_date":       event_date,
            "city":             city,
            "event_id":         event_id,
            "icao":              icao,
            "category":         category,
            "category_detail":  category_detail,
            "bot_observed_max_c":  round(bot_max_c, 2) if bot_max_c is not None else None,
            "mesonet_max_f":       round(mes_max_f, 2) if mes_max_f is not None else None,
            "mesonet_max_c":       round(mes_max_c, 2) if mes_max_c is not None else None,
            "bot_caching_gap_f":   round(bot_caching_gap_f, 2) if bot_caching_gap_f is not None else None,
            "winning_bin_label":   bins_list[winning_bin_idx].get("bin_label"),
            "winning_range_low":   wr_lo,
            "winning_range_high":  wr_hi,
            "winning_bin_idx":     winning_bin_idx,
            "bot_max_bin_idx":     bot_bin_idx,
            "mesonet_max_bin_idx": mes_bin_idx,
            "top_p_bin_label":     bins_list[top_p_idx].get("bin_label") if top_p_idx is not None else None,
            "top_p_bin_idx":       top_p_idx,
            "top_p_prob":          top_p_row.get("our_prob") if top_p_row else None,
            "our_prob_at_winning_bin": our_prob_at_winner,
            "adjacency_to_winner":  adjacency,
            "bought_winner":        bought_winner,
            "bought_top_p_only":    (top_p_contract in bought_contracts) if top_p_contract else False,
            "n_bins":               len(bins_list),
            "resolved_at":          r["resolved_at"],
        })

    log.info(f"audited {len(rows_out)} resolved markets "
             f"({missing_data_count} skipped for missing sigs/bin lookups)")

    # ---------- 7. Emit CSV ----------
    if rows_out:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            cols = list(rows_out[0].keys())
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        log.info(f"wrote {len(rows_out)} rows to {out_csv}")

    # ---------- 8. Summary ----------
    total = sum(cat_counts.values())
    print()
    print(f"{'='*72}")
    print(f"FAILURE SEGMENTATION SUMMARY — last {days}d, {total} resolved markets")
    print(f"{'='*72}")
    for cat in ["model_correct", "boundary_shape_wrong", "magnitude_shape_wrong",
                 "calibration_overconfident", "bot_caching_gap", "settle_divergence",
                 "missing_data"]:
        n = cat_counts.get(cat, 0)
        pct = 100.0 * n / total if total else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {cat:<30s} {n:>5d}  {pct:>5.1f}%  {bar}")
    print()
    losses = total - cat_counts.get("model_correct", 0) - cat_counts.get("missing_data", 0)
    print(f"  signal-rows considered:           {len(sig_rows)}")
    print(f"  total resolved (model_correct=W): {total - cat_counts.get('missing_data', 0)}")
    print(f"  losses (errors of some kind):     {losses}")
    print()

    # ---------- 9. Settle-divergence drilldown ----------
    sd_rows = [r for r in rows_out if r["category"] == "settle_divergence"]
    if sd_rows:
        print(f"{'='*72}")
        print(f"SETTLE_DIVERGENCE TUPLES ({len(sd_rows)}) — W1 investigation list")
        print(f"{'='*72}")
        print(f"  These are markets where the bot+mesonet AGREE on the max")
        print(f"  but the market settled into a different bin.  Each one is a")
        print(f"  candidate for: DSM aggregation rule, primary-site mismatch,")
        print(f"  or missing-cycle-around-peak.")
        print()
        for r in sd_rows[:50]:  # cap output; CSV has all of them
            print(f"  {r['event_date']}  {r['city']:<14s}  {r['icao']}  "
                  f"mes_max={r['mesonet_max_f']:.1f}F  "
                  f"mes_bin='{r['winning_bin_label'] if r['mesonet_max_bin_idx']==r['winning_bin_idx'] else '?'}'  "
                  f"win_bin='{r['winning_bin_label']}'  "
                  f"event={r['event_id'][:24]}")
        if len(sd_rows) > 50:
            print(f"  ... and {len(sd_rows) - 50} more in {out_csv}")

    # ---------- 10. Per-city breakdown ----------
    by_city: dict[str, Counter] = defaultdict(Counter)
    for r in rows_out:
        by_city[r["city"]][r["category"]] += 1
    print()
    print(f"{'='*72}")
    print(f"PER-CITY BREAKDOWN (categories as % of city's resolved markets)")
    print(f"{'='*72}")
    cities_sorted = sorted(by_city.keys())
    print(f"  {'city':<14s} {'N':>4s}  {'corr%':>6s} {'bnd%':>6s} {'mag%':>6s} "
          f"{'cache%':>7s} {'settle%':>8s}")
    for c in cities_sorted:
        cc = by_city[c]
        n  = sum(cc.values())
        pct = lambda k: (100.0 * cc.get(k, 0) / n) if n else 0.0
        print(f"  {c:<14s} {n:>4d}  {pct('model_correct'):>5.1f}% "
              f"{pct('boundary_shape_wrong'):>5.1f}% "
              f"{pct('magnitude_shape_wrong'):>5.1f}% "
              f"{pct('bot_caching_gap'):>6.1f}% "
              f"{pct('settle_divergence'):>7.1f}%")
    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--days",         type=int, default=60,
                    help="lookback window in days (default 60)")
    p.add_argument("--signals-db",   default=None,
                    help="signals DB path (default: from config.DB_PATH)")
    p.add_argument("--collector-db", default=DEFAULT_COLLECTOR_DB,
                    help=f"resolutions DB path (default: {DEFAULT_COLLECTOR_DB})")
    p.add_argument("--out",          default="failure_segmentation.csv",
                    help="output CSV path")
    p.add_argument("--include-unwatched", action="store_true",
                    help="include resolved markets where the bot had no signals "
                         "(distribution-shape signal even without our buys)")
    args = p.parse_args()
    signals_db = args.signals_db or DB_PATH
    if not os.path.exists(signals_db):
        log.error(f"signals DB not found: {signals_db}")
        return 1
    if not os.path.exists(args.collector_db):
        log.error(f"collector DB not found: {args.collector_db}")
        return 1
    return run_audit(signals_db, args.collector_db, args.days,
                      args.out, args.include_unwatched)


if __name__ == "__main__":
    sys.exit(main())