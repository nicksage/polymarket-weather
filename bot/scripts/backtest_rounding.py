"""
backtest_rounding.py — Phase 1 of the cold-bias remediation plan.

QUESTION
========
The codebase uses HALF-UP rounding (e.g. 85.5°F -> 86°F) to map an
observed daily-max temperature to a Polymarket bin.  A third-party
source claimed Polymarket settles by TRUNCATION (e.g. 85.5°F -> 85°F).
These two conventions disagree on half-bin shifts, which silently
shifts every bin in the system by ~0.5°F.  Whichever convention
matches the observed settlement on ~100% of clean rows is the right
one.  This script answers the question definitively.

METHODOLOGY
===========
For every event the bot saw resolve (any bin had market_prob >= 0.99
on the latest scan), we:
  1. Load the actual hourly station temps from raw_metar_log
  2. Take the max-of-hourly = realized daily-max in °C
  3. Convert to the settlement unit (°F for US, °C for international)
  4. Apply each rounding convention and check which Polymarket bin
     the result falls into
  5. Compare to the bin Polymarket actually settled on
  6. Report per-convention match rate, overall and per-city

OUTPUT INTERPRETATION
=====================
For each convention:
  - "match rate" = fraction of events where our predicted bin == winner
  - the highest match rate (ideally ~100%) is the convention Polymarket
    uses
  - any event with NO match under any convention is anomalous (settled
    against a different data source than the station hourly, or has
    bad raw_metar_log data, or settlement rules use sub-integer
    precision we're not modeling).  These are listed separately.

USAGE
=====
    cd bot
    python scripts/backtest_rounding.py
    python scripts/backtest_rounding.py --days 30 --city Miami
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from config import DB_PATH  # type: ignore

from scripts.backtest_harness import (   # type: ignore
    Bin, Event, Method, assign_bin_for_integer,
    load_resolved_events, score_method_against_events,
)


# ============================================================
# Three rounding-convention methods
# Each is a deterministic single-bin prediction: assigns 100% to
# the bin containing the rounded actual_max, 0% elsewhere.
# ============================================================

def _convert_to_settlement_unit(temp_c: float, settlement_unit: str) -> float:
    """°C source temp -> settlement unit (°F if US, else °C)."""
    if (settlement_unit or "").lower() == "fahrenheit":
        return temp_c * 9.0 / 5.0 + 32.0
    return temp_c


def _truncate(x: float) -> int:
    """Truncation toward zero: 85.99 -> 85, -1.99 -> -1.  Matches the
    common 'integer cast' / 'floor for positives' convention."""
    return int(x)


def _round_half_up(x: float) -> int:
    """Half-up rounding: 85.5 -> 86, 85.49999 -> 85.  This is what the
    codebase currently uses (see boundary_watcher._round_half_up_int,
    twc_settlement_audit._round_half_up_int)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def _round_half_even(x: float) -> int:
    """Banker's rounding (Python's built-in round()): 85.5 -> 86,
    84.5 -> 84.  Included for completeness; rarely used in settlements
    but worth confirming we can rule it out."""
    return round(x)


class _RoundingMethod(Method):
    """Apply a rounding convention to the realized daily-max, then put
    100% probability on the bin containing that integer value (0% on
    all other bins).  If the integer falls outside every bin range,
    returns an empty dict — score functions handle that as a miss."""

    def __init__(self, name: str, round_fn):
        self.name = name
        self._round_fn = round_fn

    def predict(self, event: Event) -> dict[str, float]:
        m_c = event.actual_max_c
        if m_c is None:
            return {}
        m_settle = _convert_to_settlement_unit(m_c, event.settlement_unit)
        rounded = self._round_fn(m_settle)
        bin_hit = assign_bin_for_integer(rounded, event.bins)
        if bin_hit is None:
            return {}
        return {bin_hit.label: 1.0}


TruncationMethod  = _RoundingMethod("truncation",     _truncate)
HalfUpMethod      = _RoundingMethod("round_half_up",  _round_half_up)
HalfEvenMethod    = _RoundingMethod("round_half_even", _round_half_even)


# ============================================================
# Report
# ============================================================

def print_summary(results_by_method: dict[str, dict]) -> None:
    print()
    print("=" * 72)
    print("ROUNDING CONVENTION BACKTEST — OVERALL")
    print("=" * 72)
    print(f"{'method':<22} {'N':>5} {'top-correct':>14} {'mean Brier':>12}")
    print("-" * 72)
    for name, r in results_by_method.items():
        tc = r.get("top_correct_rate")
        mb = r.get("mean_brier")
        tc_s = f"{(tc * 100):.1f}%" if tc is not None else "--"
        mb_s = f"{mb:.3f}"           if mb is not None else "--"
        print(f"{name:<22} {r['n_events']:>5} {tc_s:>14} {mb_s:>12}")
    print()
    print("'top-correct' = fraction of events where the convention's "
          "predicted bin matched the winner.")
    print("Highest score wins.  Identical scores = both conventions "
          "produce the same bin on every event (i.e., no diff exists "
          "in this sample).")


def print_per_city(results_by_method: dict[str, dict]) -> None:
    print()
    print("=" * 72)
    print("PER-CITY MATCH RATE")
    print("=" * 72)

    by_city: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list))
    for method_name, r in results_by_method.items():
        for rec in r.get("per_event", []):
            if "error" in rec:
                continue
            ev = rec["event"]
            by_city[ev.city][method_name].append(rec["top_correct"])

    method_names = list(results_by_method.keys())
    header = f"{'city':<14} {'N':>5}  " + "  ".join(
        f"{m:>16}" for m in method_names)
    print(header)
    print("-" * len(header))
    for city in sorted(by_city):
        n = len(next(iter(by_city[city].values())))
        cells = []
        for m in method_names:
            outcomes = by_city[city][m]
            rate = (sum(outcomes) / len(outcomes)) if outcomes else 0
            cells.append(f"{rate*100:>14.1f}%")
        print(f"{city:<14} {n:>5}  " + "  ".join(cells))


def print_disagreements(results_by_method: dict[str, dict]) -> None:
    """Per-event detail for any event where the conventions DIVERGE on
    which bin they predict.  These are the rows that actually
    distinguish the conventions."""
    # Index per-event records by event_id so we can join across methods
    by_event: dict[str, dict] = {}
    for method_name, r in results_by_method.items():
        for rec in r.get("per_event", []):
            if "error" in rec:
                continue
            ev = rec["event"]
            entry = by_event.setdefault(ev.event_id, {
                "event": ev,
                "actual_max_in_unit": rec["actual_max_in_unit"],
                "method_predictions": {},
                "method_correct": {},
            })
            pred = rec["predicted"]
            top_label = max(pred, key=pred.get) if pred else "(no bin)"
            entry["method_predictions"][method_name] = top_label
            entry["method_correct"][method_name] = rec["top_correct"]

    disagreements = []
    misses_all = []
    for eid, e in by_event.items():
        labels = set(e["method_predictions"].values())
        if len(labels) > 1:
            disagreements.append(e)
        if not any(e["method_correct"].values()):
            misses_all.append(e)

    print()
    print("=" * 96)
    print(f"EVENTS WHERE CONVENTIONS DISAGREE ({len(disagreements)})")
    print("=" * 96)
    if not disagreements:
        print("  (none — every event in the sample predicts the same bin "
              "under all conventions)")
        print("  This means the sample doesn't span any half-degree edge "
              "cases.  Need more events with actual_max close to .5")
    else:
        print(f"{'date':<12} {'city':<14} "
              f"{'winner':<14} {'actual':>8} "
              + "  ".join(f"{m:>22}" for m in results_by_method))
        print("-" * 96)
        for d in disagreements[:50]:
            ev = d["event"]
            actual = d["actual_max_in_unit"]
            actual_s = f"{actual:.2f}" if actual is not None else "--"
            cells = []
            for m in results_by_method:
                pred = d["method_predictions"].get(m, "?")
                ok = "✓" if d["method_correct"].get(m) else "✗"
                cells.append(f"{pred} {ok:>1}")
            print(f"{ev.event_date:<12} {ev.city:<14} "
                  f"{ev.winning_bin.label:<14} {actual_s:>8} "
                  + "  ".join(f"{c:>22}" for c in cells))
        if len(disagreements) > 50:
            print(f"... {len(disagreements) - 50} more truncated")

    print()
    print("=" * 96)
    print(f"EVENTS NO CONVENTION GOT RIGHT ({len(misses_all)})")
    print("=" * 96)
    if not misses_all:
        print("  (none)")
    else:
        print("These events settled to a bin none of our rounding ")
        print("conventions predict.  Investigate: bad raw_metar_log "
              "data, settlement against a different source, or "
              "sub-integer precision in the settlement rule.")
        print(f"{'date':<12} {'city':<14} "
              f"{'winner':<14} {'actual':>8}  "
              + "  ".join(f"{m+' pred':<14}" for m in results_by_method))
        print("-" * 96)
        for d in misses_all[:20]:
            ev = d["event"]
            actual = d["actual_max_in_unit"]
            actual_s = f"{actual:.2f}" if actual is not None else "--"
            cells = [d["method_predictions"].get(m, "?")
                     for m in results_by_method]
            print(f"{ev.event_date:<12} {ev.city:<14} "
                  f"{ev.winning_bin.label:<14} {actual_s:>8}  "
                  + "  ".join(f"{c:<14}" for c in cells))
        if len(misses_all) > 20:
            print(f"... {len(misses_all) - 20} more truncated")


# ============================================================
# Main
# ============================================================

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 1 — definitively answer 'truncation vs half-up?'",
    )
    ap.add_argument("--db",   default=DB_PATH)
    ap.add_argument("--days", type=int, default=60,
                       help="lookback window in days (default 60)")
    ap.add_argument("--city", default=None,
                       help="optional city filter for diagnostics")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"FATAL: DB not found at {args.db}", file=sys.stderr)
        return 1

    print(f"db:     {args.db}")
    print(f"window: last {args.days} days")
    if args.city:
        print(f"city:   {args.city}")

    events = load_resolved_events(
        args.db, days_back=args.days, city_filter=args.city)
    if not events:
        print("\nNo resolved events with hourly station data found.")
        print("Possible causes:")
        print("  * No events have resolved in the window")
        print("  * raw_metar_log is empty / has no rows for the event dates")
        print("    -> try a smaller --days window covering recent events")
        return 0

    print(f"\nresolved events with hourly data: {len(events)}")

    methods = [TruncationMethod, HalfUpMethod, HalfEvenMethod]
    results_by_method = {}
    for m in methods:
        res = score_method_against_events(m, events)
        results_by_method[m.name] = res

    print_summary(results_by_method)
    print_per_city(results_by_method)
    print_disagreements(results_by_method)

    # Final verdict
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    best = max(results_by_method.items(),
               key=lambda kv: kv[1].get("top_correct_rate") or 0)
    best_name, best_res = best
    best_rate = best_res.get("top_correct_rate") or 0
    print(f"Highest match rate: {best_name} ({best_rate * 100:.1f}%)")
    if best_rate >= 0.99:
        print(f"=> Settlement convention is almost certainly {best_name.upper()}.")
    elif best_rate >= 0.90:
        print(f"=> {best_name.upper()} is the leading candidate, but "
              f"{(1-best_rate)*100:.0f}% of events miss — investigate "
              f"the no-convention-matched list above.")
    else:
        print(f"=> Neither rounding convention dominates.  Settlement "
              f"may use a different data source (Wunderground display "
              f"value, sub-integer precision, etc.) than the raw "
              f"station max.  The rounding question is secondary to "
              f"a settlement-source mismatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())