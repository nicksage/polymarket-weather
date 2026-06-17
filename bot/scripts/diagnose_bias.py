#!/usr/bin/env python3
"""
diagnose_bias.py — quantify model centering error against ground truth.

Answers the question the critic put on the table after the 6/10-6/16
ledger review: is the model systematically cold, and is the raw NWS
forecast a more accurate predictor than the model's corrected mean?

Joins:
  paper_predictor_signals (first buy per contract)  → model mu_c, forecast_high_c,
                                                       at_buy_sigma_c, current_hour_local
  resolution_observations (settlement truth)         → actual_c (Wunderground high)
  positions                                          → only closed live trades

Outputs three tables to stdout (no files written):
  1. Per-city bias  — actual vs model vs forecast across all closed trades
  2. Per-city × time-of-day  — same but bucketed by current_hour_local at buy time
     (the critic's specific ask: is the bias mid-afternoon-only?)
  3. Disagreement bucket realized return — for losses, how often was the bot
     in the high-disagreement (model_p > 0.35, mkt_p < 0.15) zone?

Verdict column on tables 1 & 2:  "forecast better" if forecast MAE < model MAE.
That's the smoking gun for "trust the forecast more, correct less."

Run:
    cd bot
    python scripts/diagnose_bias.py
    python scripts/diagnose_bias.py --days 60 --city Houston

Uses resolution_observations.wunderground_high_c as the authoritative
settlement temperature.  Falls back to metar_peak_t_group_c (tenths)
then bot_observed_max_c (whole-deg).  Source per row is surfaced in
the per-event table so you know what the actual was derived from.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Optional

_HERE    = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
DEFAULT_DB = os.path.join(_BOT_DIR, "data", "signals.db")


def _bucket_hour(h: Optional[int]) -> str:
    if h is None:
        return "(unknown)"
    h = int(h)
    if h < 12:  return "morning  (<12)"
    if h < 14:  return "early PM (12-13)"
    if h < 16:  return "mid PM   (14-15)"
    if h < 18:  return "late PM  (16-17)"
    return       "evening  (>=18)"


_BUCKET_ORDER = ["morning  (<12)", "early PM (12-13)", "mid PM   (14-15)",
                  "late PM  (16-17)", "evening  (>=18)", "(unknown)"]


def fetch_closed_buys(conn: sqlite3.Connection, days: int,
                          city_filter: Optional[str]) -> list[dict]:
    """All closed live positions with their first_buy_signal model state
    + the resolution truth.  Mirrors the Analysis tab's first_buy_signal
    CTE so the numbers compare to what the dashboard shows."""
    sql = f"""
    WITH first_buy_signal AS (
        SELECT * FROM (
            SELECT
                s.contract_id, s.event_date, s.action, s.city,
                s.our_prob, s.market_prob,
                s.forecast_high_c, s.mu_c, s.sigma_c,
                s.observed_max_c, s.current_hour_local,
                s.scanned_at_utc,
                ROW_NUMBER() OVER (
                    PARTITION BY s.contract_id, s.event_date, s.action
                    ORDER BY s.scanned_at_utc ASC
                ) AS rn
            FROM paper_predictor_signals s
            WHERE s.action IN ('LIVE_BUY', 'PAPER_BUY')
              AND s.event_date >= date('now', ?)
        )
        WHERE rn = 1
    )
    SELECT
        pa.city                AS city,
        pa.date                AS event_date,
        pa.contract_id         AS contract_id,
        pa.range_low           AS bought_lo,
        pa.range_high          AS bought_hi,
        pa.unit                AS unit,
        pa.entry_price         AS entry_px,
        pa.exit_price          AS exit_px,
        pa.pnl_net             AS pnl,
        s.our_prob             AS model_p,
        s.market_prob          AS mkt_p,
        s.forecast_high_c      AS forecast_c,
        s.mu_c                 AS model_mu_c,
        s.sigma_c              AS at_buy_sigma_c,
        s.current_hour_local   AS buy_hour_local,
        r.wunderground_high_c  AS wunderground_c,
        r.metar_peak_t_group_c AS metar_t_c,
        r.bot_observed_max_c   AS bot_obs_c,
        r.winning_bin_label    AS winning_bin
    FROM positions pa
    JOIN first_buy_signal s
      ON s.contract_id = pa.contract_id
     AND s.event_date  = pa.date
     AND ((COALESCE(pa.is_paper, 0) = 0 AND s.action = 'LIVE_BUY')
       OR (COALESCE(pa.is_paper, 0) = 1 AND s.action = 'PAPER_BUY'))
    LEFT JOIN resolution_observations r
      ON r.city = pa.city AND r.event_date = pa.date
    WHERE pa.status = 'closed'
      AND COALESCE(pa.is_paper, 0) = 0
      AND pa.date >= date('now', ?)
    """
    # Two bindings: one for the first_buy_signal date cutoff, one for
    # the positions date cutoff.  City filter applied in Python below.
    window = f"-{days} days"
    rows = conn.execute(sql, (window, window)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if city_filter and d["city"] != city_filter:
            continue
        # Pick actual_c: Wunderground > T-group > bot_observed; record source
        if d["wunderground_c"] is not None:
            d["actual_c"]      = float(d["wunderground_c"])
            d["actual_source"] = "wunderground"
        elif d["metar_t_c"] is not None:
            d["actual_c"]      = float(d["metar_t_c"])
            d["actual_source"] = "metar_t_group"
        elif d["bot_obs_c"] is not None:
            d["actual_c"]      = float(d["bot_obs_c"])
            d["actual_source"] = "bot_observed_max"
        else:
            d["actual_c"]      = None
            d["actual_source"] = None
        out.append(d)
    return out


def mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def fmt_c(v: Optional[float], width: int = 6, signed: bool = False) -> str:
    if v is None:
        return f"{'--':>{width}}"
    s = "+" if signed and v >= 0 else ""
    return f"{s}{v:>{width-1 if signed and v >= 0 else width}.2f}"


def print_per_city_table(rows: list[dict]) -> None:
    by_city: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("actual_c") is None or r.get("model_mu_c") is None \
           or r.get("forecast_c") is None:
            continue
        by_city[r["city"]].append(r)

    print()
    print("=" * 108)
    print("TABLE 1 — Per-city bias (closed live trades only, last window)")
    print("=" * 108)
    print(f"{'city':<14} {'N':>3}   "
          f"{'model_bias':>10} {'fcst_bias':>10}   "
          f"{'model_MAE':>9} {'fcst_MAE':>9}   "
          f"{'verdict':<22}")
    print(f"{'':>14} {'':>3}   "
          f"{'(act-mu)':>10} {'(act-fc)':>10}   {'°C':>9} {'°C':>9}")
    print("-" * 108)

    tot_n = 0
    tot_model_b = []
    tot_fcst_b  = []
    tot_model_a = []
    tot_fcst_a  = []
    for city in sorted(by_city):
        rs = by_city[city]
        model_b = [r["actual_c"] - r["model_mu_c"] for r in rs]
        fcst_b  = [r["actual_c"] - r["forecast_c"]  for r in rs]
        model_a = [abs(x) for x in model_b]
        fcst_a  = [abs(x) for x in fcst_b]
        m_mae   = mean(model_a)
        f_mae   = mean(fcst_a)
        verdict = ""
        if m_mae is not None and f_mae is not None:
            if f_mae < m_mae - 0.1:
                verdict = "*** forecast better"
            elif m_mae < f_mae - 0.1:
                verdict = "    model better"
            else:
                verdict = "    ~tied"
        print(f"{city:<14} {len(rs):>3}   "
              f"{fmt_c(mean(model_b), signed=True):>10} "
              f"{fmt_c(mean(fcst_b),  signed=True):>10}   "
              f"{fmt_c(m_mae):>9} {fmt_c(f_mae):>9}   "
              f"{verdict:<22}")
        tot_n += len(rs)
        tot_model_b += model_b; tot_fcst_b  += fcst_b
        tot_model_a += model_a; tot_fcst_a  += fcst_a

    if tot_n > 0:
        print("-" * 108)
        m_mae = mean(tot_model_a); f_mae = mean(tot_fcst_a)
        verdict = ""
        if m_mae is not None and f_mae is not None:
            if f_mae < m_mae - 0.1:   verdict = "*** forecast better"
            elif m_mae < f_mae - 0.1: verdict = "    model better"
            else:                      verdict = "    ~tied"
        print(f"{'ALL':<14} {tot_n:>3}   "
              f"{fmt_c(mean(tot_model_b), signed=True):>10} "
              f"{fmt_c(mean(tot_fcst_b),  signed=True):>10}   "
              f"{fmt_c(m_mae):>9} {fmt_c(f_mae):>9}   "
              f"{verdict:<22}")
    print()
    print("  POSITIVE bias = model/forecast was too COLD relative to actual.")
    print("  Stars on 'forecast better' rows are the smoking gun the critic")
    print("  is asking about — those cities' corrections subtract value.")


def print_per_city_tod_table(rows: list[dict]) -> None:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("actual_c") is None or r.get("model_mu_c") is None \
           or r.get("forecast_c") is None:
            continue
        key = (r["city"], _bucket_hour(r.get("buy_hour_local")))
        by_key[key].append(r)

    print()
    print("=" * 108)
    print("TABLE 2 — Per-city × time-of-day  (was the bias mid-PM only?)")
    print("=" * 108)
    print(f"{'city':<14} {'bucket':<18} {'N':>3}   "
          f"{'model_bias':>10} {'fcst_bias':>10}   "
          f"{'model_MAE':>9} {'fcst_MAE':>9}   "
          f"{'verdict':<22}")
    print("-" * 108)

    cities = sorted({c for (c, _) in by_key})
    for city in cities:
        for bucket in _BUCKET_ORDER:
            rs = by_key.get((city, bucket), [])
            if not rs:
                continue
            model_b = [r["actual_c"] - r["model_mu_c"] for r in rs]
            fcst_b  = [r["actual_c"] - r["forecast_c"]  for r in rs]
            m_mae   = mean([abs(x) for x in model_b])
            f_mae   = mean([abs(x) for x in fcst_b])
            verdict = ""
            if m_mae is not None and f_mae is not None:
                if f_mae < m_mae - 0.1:   verdict = "*** forecast better"
                elif m_mae < f_mae - 0.1: verdict = "    model better"
                else:                      verdict = "    ~tied"
            print(f"{city:<14} {bucket:<18} {len(rs):>3}   "
                  f"{fmt_c(mean(model_b), signed=True):>10} "
                  f"{fmt_c(mean(fcst_b),  signed=True):>10}   "
                  f"{fmt_c(m_mae):>9} {fmt_c(f_mae):>9}   "
                  f"{verdict:<22}")
        # blank row between cities for readability
        print()


def print_disagreement_table(rows: list[dict]) -> None:
    """For closed trades, bucket by (model_p, mkt_p) zone and show win
    rate + avg PnL.  The 'high disagreement' bucket — model_p>=0.35 AND
    mkt_p<=0.15 — is the one the critic flagged as systematically losing.
    """
    print()
    print("=" * 108)
    print("TABLE 3 — Closed-trade outcome by disagreement bucket")
    print("=" * 108)
    print(f"{'bucket':<55} {'N':>4} {'wins':>5} {'win%':>6} {'avg PnL':>10} {'tot PnL':>11}")
    print("-" * 108)

    def bucket(r) -> str:
        mp = r.get("model_p"); kp = r.get("mkt_p")
        if mp is None or kp is None:
            return "(missing prob)"
        if mp >= 0.35 and kp <= 0.15:
            return "HIGH disagreement (our_p>=0.35, mkt_p<=0.15)"
        if mp >= 0.35 and kp >= 0.35:
            return "agreement-confident (both >=0.35)"
        if mp <  0.35 and kp <  0.35:
            return "agreement-skeptical (both <0.35)"
        if mp >= 0.35 and kp > 0.15 and kp < 0.35:
            return "mid disagreement"
        return "other"

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("exit_px") is None and r.get("pnl") is None:
            continue
        by_bucket[bucket(r)].append(r)

    for b, rs in sorted(by_bucket.items()):
        # Win: exit_px >= 0.99 or pnl > 0
        wins = 0
        pnl_list = []
        for r in rs:
            won = False
            if r.get("exit_px") is not None and r["exit_px"] >= 0.99:
                won = True
            elif r.get("pnl") is not None and r["pnl"] > 0:
                won = True
            if won:
                wins += 1
            if r.get("pnl") is not None:
                pnl_list.append(float(r["pnl"]))
        n = len(rs)
        wr = (100.0 * wins / n) if n else 0
        avg_pnl = mean(pnl_list)
        tot_pnl = sum(pnl_list) if pnl_list else None
        avg_s = f"{avg_pnl:+.2f}" if avg_pnl is not None else "--"
        tot_s = f"{tot_pnl:+.2f}" if tot_pnl is not None else "--"
        print(f"{b:<55} {n:>4} {wins:>5} {wr:>5.1f}% "
              f"{avg_s:>10} {tot_s:>11}")
    print()
    print("  If 'HIGH disagreement' has negative avg PnL, the loss-stopper")
    print("  gate (Stage 3) is justified — you're losing those trades")
    print("  systematically, the cold bias is manufacturing false edge.")


def print_sigma_distribution(rows: list[dict]) -> None:
    """How many entries had over-collapsed σ?  The σ floor was meant to
    catch ≤1.3°C but bin-lock can override it to ~0.19°C."""
    print()
    print("=" * 60)
    print("TABLE 4 — at_buy σ distribution")
    print("=" * 60)
    print(f"{'bucket':<28} {'N':>4} {'%':>6}")
    print("-" * 60)
    buckets = [
        ("<=0.30 (bin-lock collapse)", lambda s: s <= 0.30),
        ("0.30-1.30 (below floor)",    lambda s: 0.30 < s <= 1.30),
        ("1.30-2.00 (floored zone)",   lambda s: 1.30 < s <= 2.00),
        ("2.00-3.00 (normal)",          lambda s: 2.00 < s <= 3.00),
        (">3.00 (wide)",                lambda s: s > 3.00),
    ]
    sigmas = [float(r["at_buy_sigma_c"]) for r in rows
              if r.get("at_buy_sigma_c") is not None]
    n_total = len(sigmas)
    for label, pred in buckets:
        n = sum(1 for s in sigmas if pred(s))
        pct = (100.0 * n / n_total) if n_total else 0
        print(f"{label:<28} {n:>4} {pct:>5.1f}%")
    print(f"{'TOTAL':<28} {n_total:>4}")
    print()
    print("  '<=0.30' rows are bin-lock collapse — Stage 4a fix raises")
    print("  these to >=PREDICTOR_SIGMA_FLOOR_C (default 1.3°C).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",   default=DEFAULT_DB)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--city", default=None, help="limit to one city")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"FATAL: DB not found at {args.db}", file=sys.stderr)
        return 1

    print(f"db:     {args.db}")
    print(f"window: last {args.days} days")
    if args.city:
        print(f"city:   {args.city}")

    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        rows = fetch_closed_buys(conn, args.days, args.city)
    print(f"closed live trades found: {len(rows)}")
    n_with_actual = sum(1 for r in rows if r.get("actual_c") is not None)
    n_wund        = sum(1 for r in rows if r.get("actual_source") == "wunderground")
    n_metar       = sum(1 for r in rows if r.get("actual_source") == "metar_t_group")
    n_bot         = sum(1 for r in rows if r.get("actual_source") == "bot_observed_max")
    print(f"  with actual_c:       {n_with_actual}/{len(rows)}  "
          f"(wunderground={n_wund}, metar_t_group={n_metar}, "
          f"bot_observed_max={n_bot})")

    if n_with_actual == 0:
        print()
        print("No resolution_observations rows joined — either none of the")
        print("closed events have been captured, OR the capture script hasn't")
        print("backfilled the window.  Run capture before re-running this.")
        return 0

    print_per_city_table(rows)
    print_per_city_tod_table(rows)
    print_disagreement_table(rows)
    print_sigma_distribution(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())