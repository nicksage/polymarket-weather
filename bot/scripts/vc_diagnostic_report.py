"""
vc_diagnostic_report.py — Retrospective analysis of VC forecast disagreement.

Reads vc_forecast_diagnostics + decision_snapshots + positions +
historical_observed_daily to answer:

  1. Does VC disagreement contain predictive signal about model error?
  2. Does large VC disagreement predict worse trade outcomes?
  3. Which past trades would a VC-disagreement filter have warned us away from?

Shadow-only — does not touch any trade logic.  Run on demand:

    python bot/scripts/vc_diagnostic_report.py --days 14
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from config import (
    DB_PATH,
    VC_DIAG_BIN_WIDTH_C,
    VC_DIAG_DISAGREE_LARGE_C,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("vc_diag_report")

BUCKETS_C = [(0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, float("inf"))]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_diags(conn: sqlite3.Connection, since: str) -> list[dict]:
    rows = conn.execute("""
        SELECT d.*, h.tempmax_c AS actual_tmax_c
        FROM vc_forecast_diagnostics d
        LEFT JOIN historical_observed_daily h
          ON ROUND(d.target_date, 2) = h.date   -- no-op; match on date below
        WHERE d.pulled_at_utc >= ?
    """, (since,)).fetchall()
    # Second pass to enrich with actuals (coarse join above was placeholder)
    out = []
    for r in rows:
        r = dict(r)
        # Pull actual tmax by (city, target_date) — historical_observed_daily
        # stores by (lat_key, lon_key, date).  Use city match as fallback.
        actual = conn.execute(
            "SELECT tempmax_c FROM historical_observed_daily "
            "WHERE city = ? AND date = ? ORDER BY fetched_at DESC LIMIT 1",
            (r.get("city"), r.get("target_date"))
        ).fetchone()
        r["actual_tmax_c"] = actual[0] if actual else None
        out.append(r)
    return out


def _load_snapshots(conn: sqlite3.Connection, since: str) -> list[dict]:
    return [dict(r) for r in conn.execute("""
        SELECT s.*, h.tempmax_c AS actual_tmax_c
        FROM decision_snapshots s
        LEFT JOIN historical_observed_daily h
          ON h.city = s.city AND h.date = s.date
        WHERE s.evaluated_at_utc >= ?
    """, (since,)).fetchall()]


def _load_closed_positions(conn: sqlite3.Connection, since: str) -> list[dict]:
    return [dict(r) for r in conn.execute("""
        SELECT p.*, s.vc_vs_blended_mu_c, s.vc_vs_adjusted_mu_c,
               s.vc_bins_apart, s.flag_vc_disagreement_large,
               s.flag_vc_warns_hotter, s.flag_vc_warns_colder,
               s.vc_projected_day_max_c,
               s.blended_mu_c, s.adjusted_mu_c
        FROM positions p
        LEFT JOIN decision_snapshots s
               ON s.contract_id = p.contract_id
              AND s.evaluated_at_utc <= p.entry_time
        WHERE p.status = 'closed' AND p.entry_time >= ?
        GROUP BY p.id
    """, (since,)).fetchall()]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bucket_for(abs_disagreement: float | None) -> str | None:
    if abs_disagreement is None:
        return None
    for lo, hi in BUCKETS_C:
        if lo <= abs_disagreement < hi:
            return f"[{lo:.1f}, {'∞' if hi == float('inf') else f'{hi:.1f}'})"
    return None


def cohort_analysis(diags: list[dict]) -> list[dict]:
    """Group by |vc_vs_blended_mu_c| bucket; report accuracy stats."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for d in diags:
        abs_dis = d.get("abs_vc_vs_blended")
        actual = d.get("actual_tmax_c")
        if abs_dis is None or actual is None:
            continue
        bkt = _bucket_for(abs_dis)
        if bkt:
            by_bucket[bkt].append(d)

    out = []
    for bkt, rows in sorted(by_bucket.items()):
        def _mae(a_key: str) -> float | None:
            vals = [abs(r[a_key] - r["actual_tmax_c"]) for r in rows
                    if r.get(a_key) is not None and r.get("actual_tmax_c") is not None]
            return (sum(vals) / len(vals)) if vals else None

        mae_blended = _mae("blended_mu_c")
        mae_adjusted = _mae("adjusted_mu_c")
        mae_vc = _mae("vc_projected_day_max_c")
        out.append({
            "bucket":        bkt,
            "n":             len(rows),
            "mae_blended":   mae_blended,
            "mae_adjusted":  mae_adjusted,
            "mae_vc":        mae_vc,
        })
    return out


def directional_lift(diags: list[dict]) -> dict:
    """Contingency table: VC hotter/colder vs actual hotter/colder (vs blended μ)."""
    a = b = c = d = 0
    for r in diags:
        vc_d   = r.get("vc_vs_blended_mu_c")
        actual = r.get("actual_tmax_c")
        blend  = r.get("blended_mu_c")
        if vc_d is None or actual is None or blend is None:
            continue
        vc_hotter     = vc_d > 0
        actual_hotter = actual > blend
        if   vc_hotter and actual_hotter:         a += 1
        elif not vc_hotter and actual_hotter:     b += 1
        elif vc_hotter and not actual_hotter:     c += 1
        else:                                     d += 1
    total = a + b + c + d
    if total == 0:
        return {"a": 0, "b": 0, "c": 0, "d": 0, "total": 0,
                "p_hot_given_vc_hot": None, "p_hot": None, "lift": None}
    p_hot              = (a + b) / total
    p_hot_given_vc_hot = (a / (a + c)) if (a + c) > 0 else None
    lift = (p_hot_given_vc_hot / p_hot) if (p_hot_given_vc_hot and p_hot) else None
    return {
        "a": a, "b": b, "c": c, "d": d, "total": total,
        "p_hot_given_vc_hot": p_hot_given_vc_hot,
        "p_hot":              p_hot,
        "lift":               lift,
    }


def trade_outcomes(positions: list[dict]) -> dict:
    """Compare closed-position outcomes by VC flag presence."""
    groups: dict[str, list[dict]] = {
        "no_warning":    [],
        "large":         [],
        "warns_hotter":  [],
        "warns_colder":  [],
    }
    for p in positions:
        if p.get("flag_vc_disagreement_large"):
            groups["large"].append(p)
        if p.get("flag_vc_warns_hotter"):
            groups["warns_hotter"].append(p)
        if p.get("flag_vc_warns_colder"):
            groups["warns_colder"].append(p)
        if not any([p.get("flag_vc_disagreement_large"),
                    p.get("flag_vc_warns_hotter"),
                    p.get("flag_vc_warns_colder")]):
            groups["no_warning"].append(p)

    out: dict[str, dict] = {}
    for g, ps in groups.items():
        if not ps:
            out[g] = {"n": 0}
            continue
        pnls = [p.get("pnl") or 0.0 for p in ps]
        wins = sum(1 for p in pnls if p > 0)
        out[g] = {
            "n":          len(ps),
            "wins":       wins,
            "win_rate":   wins / len(ps),
            "avg_pnl":    sum(pnls) / len(ps),
            "total_pnl":  sum(pnls),
        }
    return out


def warn_away_trades(positions: list[dict]) -> list[dict]:
    """List of closed trades that a VC-disagreement filter would have flagged
    at entry.  Sorted by P&L ascending so the biggest losers are at the top."""
    out = []
    for p in positions:
        if p.get("flag_vc_disagreement_large"):
            out.append(p)
    out.sort(key=lambda p: (p.get("pnl") or 0.0))
    return out


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    days: int,
    diags: list[dict], snapshots: list[dict], positions: list[dict],
) -> str:
    logs_dir = os.path.join(_BOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(logs_dir, f"vc_diagnostic_{ts}.md")

    cohort = cohort_analysis(diags)
    directional = directional_lift(diags)
    outcomes = trade_outcomes(positions)
    warn_list = warn_away_trades(positions)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# VC Forecast Diagnostic Report\n\n")
        f.write(f"Window: last {days} days  \n")
        f.write(f"Diagnostic rows: {len(diags)}  \n")
        f.write(f"Decision snapshots: {len(snapshots)}  \n")
        f.write(f"Closed positions: {len(positions)}  \n")
        f.write(f"Disagreement-large threshold: {VC_DIAG_DISAGREE_LARGE_C}°C  \n")
        f.write(f"Bin-width estimate: {VC_DIAG_BIN_WIDTH_C}°C  \n\n")

        # --- Section 1: cohort accuracy ---
        f.write("## 1. Forecast Accuracy by VC-Disagreement Bucket\n\n")
        f.write("For events where VC disagreed with our model by the indicated "
                "magnitude, how accurate was each predictor against the realized "
                "daily max?  Lower MAE = closer to truth.\n\n")
        if not cohort:
            f.write("_No rows available yet._\n\n")
        else:
            f.write("| Disagreement | N | MAE blended | MAE adjusted | MAE VC |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for c in cohort:
                def _fmt(v):
                    return f"{v:.2f}°C" if v is not None else "—"
                f.write(f"| {c['bucket']} | {c['n']} | {_fmt(c['mae_blended'])} | "
                        f"{_fmt(c['mae_adjusted'])} | {_fmt(c['mae_vc'])} |\n")
            f.write("\n")

        # --- Section 2: directional lift ---
        f.write("## 2. Directional Predictive Value\n\n")
        f.write("When VC said hotter than blended, was actual also hotter?  "
                "Lift > 1.0 means VC direction contains information.  "
                "Lift ≈ 1.0 means VC direction is noise.\n\n")
        if directional["total"] == 0:
            f.write("_Insufficient data._\n\n")
        else:
            d = directional
            f.write(f"| Contingency | Actual HOTTER | Actual COLDER |\n")
            f.write(f"|---|---:|---:|\n")
            f.write(f"| VC says HOTTER | {d['a']} | {d['c']} |\n")
            f.write(f"| VC says COLDER | {d['b']} | {d['d']} |\n\n")
            f.write(f"Base rate P(actual hotter) = {d['p_hot']:.3f}  \n")
            if d["p_hot_given_vc_hot"] is not None:
                f.write(f"P(actual hotter | VC says hotter) = "
                        f"{d['p_hot_given_vc_hot']:.3f}  \n")
            if d["lift"] is not None:
                f.write(f"**Lift = {d['lift']:.3f}** "
                        f"({'signal' if d['lift'] > 1.05 else 'no clear signal'})\n\n")

        # --- Section 3: trade outcomes by flag ---
        f.write("## 3. Trade Outcome by VC Flag (closed positions)\n\n")
        if not positions:
            f.write("_No closed positions in window._\n\n")
        else:
            f.write("| Group | N | Wins | Win rate | Avg P&L | Total P&L |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for g, s in outcomes.items():
                if s["n"] == 0:
                    f.write(f"| {g} | 0 | — | — | — | — |\n")
                else:
                    f.write(f"| {g} | {s['n']} | {s['wins']} | "
                            f"{s['win_rate']*100:.1f}% | "
                            f"${s['avg_pnl']:.2f} | ${s['total_pnl']:.2f} |\n")
            f.write("\n")

        # --- Section 4: warn-away list ---
        f.write("## 4. Trades a VC-Disagreement Filter Would Have Warned Against\n\n")
        f.write(f"Closed trades entered while `flag_vc_disagreement_large=1` "
                f"(|VC − blended_μ| ≥ {VC_DIAG_DISAGREE_LARGE_C}°C).  "
                f"Sorted by P&L ascending (biggest losers first).\n\n")
        if not warn_list:
            f.write("_No flagged closed trades in window._\n\n")
        else:
            f.write("| City | Date | Side | Entry px | P&L | VC Δ (°C) | "
                    "bins_apart | Warns |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---|\n")
            for p in warn_list[:50]:   # cap at 50
                warns = []
                if p.get("flag_vc_warns_hotter"): warns.append("hotter")
                if p.get("flag_vc_warns_colder"): warns.append("colder")
                vc_d = p.get("vc_vs_blended_mu_c")
                vcd_fmt = f"{vc_d:+.2f}" if vc_d is not None else "—"
                pnl = p.get("pnl") or 0.0
                f.write(f"| {p.get('city', '?')} | {p.get('date', '?')} | "
                        f"{p.get('side', '?')} | "
                        f"${(p.get('entry_price') or 0):.3f} | "
                        f"${pnl:+.2f} | {vcd_fmt} | "
                        f"{p.get('vc_bins_apart', '—')} | "
                        f"{','.join(warns) or '—'} |\n")
            if len(warn_list) > 50:
                f.write(f"\n_... and {len(warn_list) - 50} more._\n")
            # Aggregate impact
            total_flagged_pnl = sum(p.get("pnl") or 0.0 for p in warn_list)
            total_all_pnl = sum(p.get("pnl") or 0.0 for p in positions)
            f.write(f"\n**Aggregate P&L of flagged trades: ${total_flagged_pnl:+.2f}**  \n")
            f.write(f"Of total closed-position P&L ${total_all_pnl:+.2f}  \n")
            if total_all_pnl != 0:
                f.write(f"Flagged trades accounted for "
                        f"{total_flagged_pnl/total_all_pnl*100:.1f}% of total P&L  \n")
            f.write("\nIf a VC-filter had blocked these entries, total P&L "
                    f"would have been approximately "
                    f"${(total_all_pnl - total_flagged_pnl):+.2f}.\n\n")

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    diags     = _load_diags(conn, since)
    snaps     = _load_snapshots(conn, since)
    positions = _load_closed_positions(conn, since)

    log.info(f"Loaded: {len(diags)} diagnostics, {len(snaps)} snapshots, "
             f"{len(positions)} closed positions (last {args.days} days)")

    path = write_report(args.days, diags, snaps, positions)
    log.info(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
