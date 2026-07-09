"""Forecast/probability accuracy — MODEL vs MARKET, side by side.

For every resolved event, reconstruct at each historical forecast time:
  * the MODEL distribution over the market's bins (TWC probabilistic `prototypes`
    -> daily-max ensemble -> P(bin)), and
  * the MARKET distribution (Polymarket implied prob per bin, as-of that time),
and score both against the outcome (winning bin + measured high), vs lead time.

Metrics (proper scoring rules; lower is better):
  * RPS   — Ranked Probability Score (ordinal: one bin off beats five off)
  * Brier — multi-category
  * log-loss — ignorance on the winning bin (punishes overconfidence)
  * CRPS  — model only, continuous ensemble vs the exact measured high (Celsius)

Bin scores are unit-independent; CRPS and the deterministic error are computed
in Celsius so events in F and C markets aggregate consistently.

Writes a `forecast_scores` table (one row per event x snapshot) for your own
slicing/plotting, and prints skill by lead-time, a model-vs-market summary, a
reliability curve at a target lead, and deterministic bias.

    python -m analysis.accuracy                 # runs on db/snapshot.db
    python -m analysis.accuracy --db path --rel-lead 24
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, date as dt_date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from analysis.db import DEFAULT_DB
from config.cities import CITY_TZ

LEAD_EDGES = [0, 3, 6, 12, 24, 48, 1e9]
LEAD_LABELS = ["0-3h", "3-6h", "6-12h", "12-24h", "24-48h", "48h+"]
PEAK_HOUR = 15   # local reference for "when the daily high happens" (mid-afternoon)


def c_to_f(c):
    return c * 9 / 5 + 32


def _peak_ref_utc(date_str, tzname):
    """UTC instant of the resolution day's mid-afternoon peak reference. Lead is
    measured to this: a forecast can only 'forecast the daily high' before the
    high occurs, so snapshots issued after it are not scored."""
    tz = ZoneInfo(tzname)
    y, m, d = map(int, date_str.split("-"))
    return datetime(y, m, d, PEAK_HOUR, 0, tzinfo=tz).astimezone(timezone.utc)


def _daily_high_samples(proto_data, fcst_valid, tz, date_str):
    """Max over the resolution date's local hours, per ensemble trace (Celsius)."""
    idx = [i for i, t in enumerate(fcst_valid)
           if datetime.fromtimestamp(t, tz).strftime("%Y-%m-%d") == date_str]
    if not idx:
        return None
    out = []
    for tr in proto_data.get("forecast", []):
        vals = [tr[i] for i in idx if i < len(tr) and tr[i] is not None]
        if vals:
            out.append(max(vals))
    return np.asarray(out, dtype=float) if out else None


def _model_dist(samples, ranges):
    """P(bin) = fraction of daily-high samples in each bin's +/-0.5 range
    (samples and ranges in the market's unit)."""
    p = np.array([
        np.mean((samples > (lo - 0.5 if lo is not None else -1e9)) &
                (samples <= (hi + 0.5 if hi is not None else 1e9)))
        for lo, hi in ranges])
    s = p.sum()
    return p / s if s > 0 else p


def _scores_bins(p, k):
    """RPS (normalised), Brier, log-loss for distribution p and winning index k."""
    o = np.zeros_like(p)
    o[k] = 1.0
    brier = float(np.sum((p - o) ** 2))
    n = len(p)
    rps = float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2)) / (n - 1) if n > 1 else 0.0
    logloss = float(-np.log(max(p[k], 1e-6)))
    return brier, rps, logloss


def _crps(samples_c, y_c):
    """Energy-form CRPS (Celsius) of the empirical ensemble vs observed high."""
    s = np.asarray(samples_c, dtype=float)
    if s.size == 0:
        return None
    return float(np.mean(np.abs(s - y_c)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])))


def score_event(conn, ev):
    """Yield one score dict per forecast snapshot for a resolved event."""
    eid, city, date_str = ev["event_id"], ev["city"], ev["date"]
    tzname = CITY_TZ.get(city)
    if not tzname:
        return
    tz = ZoneInfo(tzname)

    bins = conn.execute(
        """SELECT contract_id, range_low, range_high, unit FROM bins
           WHERE event_id = ? ORDER BY range_low IS NULL DESC, range_low""", (eid,)).fetchall()
    if not bins:
        return
    ranges = [(b["range_low"], b["range_high"]) for b in bins]
    contracts = [b["contract_id"] for b in bins]
    unit = (bins[0]["unit"] or "celsius")
    to_f = unit.lower().startswith("f")
    try:
        k = contracts.index(ev["winning_contract_id"])
    except ValueError:
        return
    y_c = ev["actual_high_c"]                      # measured high, Celsius (for CRPS/bias)
    peak_utc = _peak_ref_utc(date_str, tzname)

    ph = pd.read_sql_query(
        "SELECT contract_id, yes_price, recorded_at FROM price_snapshots WHERE event_id = ?",
        conn, params=[eid])
    wide = None
    if not ph.empty:
        ph["recorded_at"] = pd.to_datetime(ph["recorded_at"], utc=True)
        wide = (ph.pivot_table(index="recorded_at", columns="contract_id",
                               values="yes_price", aggfunc="last").sort_index())

    protos = conn.execute(
        """SELECT data, fcst_valid, fetched_at FROM twc_probabilistic
           WHERE city = ? AND product = 'prototypes' AND parameter = 'temperature'
           ORDER BY fetched_at""", (city,)).fetchall()

    for pr in protos:
        s_c = _daily_high_samples(json.loads(pr["data"]), json.loads(pr["fcst_valid"]),
                                  tz, date_str)
        if s_c is None or s_c.size == 0:
            continue
        ft = pd.to_datetime(pr["fetched_at"], utc=True)
        lead = (peak_utc - ft.to_pydatetime()).total_seconds() / 3600.0
        if lead <= 0:                              # issued after the peak — nothing to forecast
            continue

        s_mkt = c_to_f(s_c) if to_f else s_c       # samples in market unit for bin membership
        mp = _model_dist(s_mkt, ranges)
        mb, mr, ml = _scores_bins(mp, k)
        crps = _crps(s_c, y_c) if y_c is not None else None

        market_dist = None
        qb = qr = qlg = qpw = None
        if wide is not None:
            pos = wide.index.searchsorted(ft, side="right") - 1
            if pos >= 0:
                prices = wide.iloc[pos].reindex(contracts).astype(float).fillna(0.0).to_numpy()
                tot = prices.sum()
                if tot > 0:
                    q = prices / tot
                    qb, qr, qlg = _scores_bins(q, k)
                    qpw = float(q[k])
                    market_dist = [round(float(x), 5) for x in q]

        yield {
            "event_id": eid, "city": city, "date": date_str,
            "fetched_at": pr["fetched_at"], "lead_hours": round(lead, 2),
            "n_bins": len(bins), "win_idx": k, "unit": unit,
            "model_high_c": float(np.median(s_c)), "measured_high_c": y_c,
            "model_brier": mb, "model_rps": mr, "model_logloss": ml, "model_crps": crps,
            "model_p_win": float(mp[k]),
            "market_brier": qb, "market_rps": qr, "market_logloss": qlg, "market_p_win": qpw,
            "model_dist": [round(float(x), 5) for x in mp], "market_dist": market_dist,
        }


def _reliability(points, tag):
    """points: array of (predicted_prob, outcome) over bins. Bucket by predicted
    prob and show mean-predicted vs observed frequency (calibration)."""
    d = pd.DataFrame(points, columns=["p", "o"])
    d["b"] = (d["p"] * 10).clip(0, 9).astype(int)
    print(f"  {tag}:   {'prob bucket':<12} {'mean pred':>9} {'observed':>9} {'n':>5}")
    for b, g in d.groupby("b"):
        print(f"      {'':<8}[{b/10:.1f},{(b+1)/10:.1f})  {g['p'].mean():>9.2f} "
              f"{g['o'].mean():>9.2f} {len(g):>5}")


def build(db_path, rel_lead):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    resolved = conn.execute(
        """SELECT event_id, city, date, winning_contract_id, actual_high_c
           FROM resolutions WHERE winning_contract_id IS NOT NULL""").fetchall()
    rows = []
    for ev in resolved:
        rows.extend(score_event(conn, ev))
    if not rows:
        print("No scored snapshots yet — need resolved events whose probabilistic forecast\n"
              "history covers the resolution day (populates as Jul 7+ events resolve).")
        conn.close()
        return
    df = pd.DataFrame(rows)

    # persist (JSON-encode the dist columns so SQLite accepts them)
    store = df.copy()
    store["model_dist"] = store["model_dist"].apply(json.dumps)
    store["market_dist"] = store["market_dist"].apply(lambda x: json.dumps(x) if x is not None else None)
    store.to_sql("forecast_scores", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    df["lead_bucket"] = pd.cut(df["lead_hours"], bins=LEAD_EDGES, labels=LEAD_LABELS, right=True)
    print(f"\nScored {len(df)} forecast snapshots across {df['event_id'].nunique()} "
          f"resolved event(s).  (wrote table `forecast_scores`)\n")

    # 1) skill by lead time
    print("=== Skill by lead time (mean; lower is better) ===")
    print(f"  {'lead':<8} {'n':>4} | {'RPS mdl':>8} {'RPS mkt':>8} | "
          f"{'LL mdl':>7} {'LL mkt':>7} | {'CRPS C':>7}")
    for lab in LEAD_LABELS:
        g = df[df["lead_bucket"] == lab]
        if g.empty:
            continue
        gm = g.dropna(subset=["market_rps"])
        mkt_rps = gm["market_rps"].mean() if len(gm) else float("nan")
        mkt_ll = gm["market_logloss"].mean() if len(gm) else float("nan")
        print(f"  {lab:<8} {len(g):>4} | {g['model_rps'].mean():>8.4f} {mkt_rps:>8.4f} | "
              f"{g['model_logloss'].mean():>7.3f} {mkt_ll:>7.3f} | {g['model_crps'].mean():>7.3f}")

    # 2) overall model vs market (where both exist)
    both = df.dropna(subset=["market_rps"])
    print("\n=== Model vs Market (snapshots where both exist) ===")
    if both.empty:
        print("  (no snapshots with both model and market prices yet)")
    else:
        for label, mcol, qcol in [("RPS", "model_rps", "market_rps"),
                                  ("Brier", "model_brier", "market_brier"),
                                  ("log-loss", "model_logloss", "market_logloss")]:
            m, q = both[mcol].mean(), both[qcol].mean()
            better = "model" if m < q else "market"
            print(f"  {label:<9} model={m:.4f}  market={q:.4f}  -> {better} better "
                  f"({(m-q)/q*100:+.1f}% vs market)")

    # 3) reliability at ~rel_lead (nearest snapshot per event), pooled over bins
    print(f"\n=== Reliability at ~{rel_lead:g}h lead (predicted prob -> observed frequency) ===")
    pick = (df.assign(d=(df["lead_hours"] - rel_lead).abs())
              .sort_values("d").groupby("event_id").head(1))
    m_pts, q_pts = [], []
    for _, r in pick.iterrows():
        o = np.zeros(r["n_bins"]); o[r["win_idx"]] = 1.0
        m_pts += list(zip(r["model_dist"], o))
        if r["market_dist"] is not None:
            q_pts += list(zip(r["market_dist"], o))
    if len(pick) >= 3:
        _reliability(m_pts, "MODEL ")
        if q_pts:
            _reliability(q_pts, "MARKET")
        print(f"  (from {len(pick)} event(s); well-calibrated = mean pred ~= observed)")
    else:
        print(f"  only {len(pick)} event(s) near {rel_lead:g}h lead — need more resolutions.")

    # 4) deterministic bias (Celsius)
    print("\n=== Forecast daily-high bias (model median - measured, Celsius) ===")
    df["err"] = df["model_high_c"] - df["measured_high_c"]
    for lab in LEAD_LABELS:
        g = df[df["lead_bucket"] == lab].dropna(subset=["err"])
        if g.empty:
            continue
        print(f"  {lab:<8} n={len(g):>4}  bias={g['err'].mean():+.2f}  "
              f"MAE={g['err'].abs().mean():.2f}  RMSE={np.sqrt((g['err']**2).mean()):.2f}")


def main():
    ap = argparse.ArgumentParser(description="Model vs market forecast accuracy")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--rel-lead", type=float, default=24.0,
                    help="target lead (hours) for the reliability curve")
    args = ap.parse_args()
    build(args.db, args.rel_lead)


if __name__ == "__main__":
    main()
