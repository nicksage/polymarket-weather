"""
backtest.py — Replay historical price data to simulate any trading strategy.

Reads from the standalone collector database (data/prices.db).
Uses price_snapshots for price paths and resolutions to know which bin won.

Usage:
    python backtest.py                    # full parameter sweep
    python backtest.py --sweep            # same as above
    python backtest.py --min-price 0.30 --max-price 0.45 --bins 2 --trail 12
"""

import argparse
import sqlite3
import os
from dataclasses import dataclass

import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prices.db")


@dataclass
class StrategyParams:
    min_price: float = 0.25
    max_price: float = 0.50
    max_bins: int = 2
    trail_pct: float = 0.12
    trail_activation: float = 0.03
    take_profit: float = 0.90
    cross_bin_exit: float = 0.75
    hard_stop_pct: float = 0.30
    bet_size: float = 500.0
    use_ratchet: bool = False


@dataclass
class SimTrade:
    event_id: str
    contract_id: str
    city: str
    date: str
    entry_price: float
    entry_time: str
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    pnl: float = 0.0
    peak_price: float = 0.0
    won_event: bool = False
    shares: float = 0.0


def load_backtestable_events() -> list[dict]:
    """Load all resolved events with full price histories."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    resolutions = conn.execute("""
        SELECT event_id, city, date, winning_contract_id,
               winning_range_low, winning_range_high
        FROM resolutions
        ORDER BY date
    """).fetchall()

    events = []
    for res in resolutions:
        eid = res["event_id"]
        winner_cid = res["winning_contract_id"]

        bins_data = conn.execute("""
            SELECT contract_id,
                   GROUP_CONCAT(yes_price, '|') as prices,
                   GROUP_CONCAT(recorded_at, '|') as times,
                   COUNT(*) as n_snaps
            FROM price_snapshots
            WHERE event_id = ? AND yes_price IS NOT NULL
            GROUP BY contract_id
            HAVING n_snaps >= 3
            ORDER BY MAX(yes_price) DESC
        """, (eid,)).fetchall()

        if not bins_data:
            continue

        bins = []
        for b in bins_data:
            prices = [float(p) for p in b["prices"].split("|") if p]
            times = b["times"].split("|")
            if len(prices) < 3:
                continue
            bins.append({
                "contract_id": b["contract_id"],
                "prices": prices,
                "times": times,
                "is_winner": b["contract_id"] == winner_cid,
                "initial_price": prices[0],
            })

        if not bins:
            continue

        events.append({
            "event_id": eid,
            "city": res["city"],
            "date": res["date"],
            "winner_cid": winner_cid,
            "bins": bins,
        })

    conn.close()
    return events


def simulate_event(event: dict, params: StrategyParams) -> list[SimTrade]:
    """Simulate trading one event with the given strategy parameters."""
    bins = event["bins"]

    candidates = [
        b for b in bins
        if params.min_price <= b["initial_price"] <= params.max_price
    ]
    candidates.sort(key=lambda b: b["initial_price"], reverse=True)
    selected = candidates[:params.max_bins]

    if not selected:
        return []

    trades: list[SimTrade] = []

    for b in selected:
        entry_price = b["initial_price"]
        shares = params.bet_size / entry_price if entry_price > 0 else 0
        prices = b["prices"]
        times = b["times"]

        trade = SimTrade(
            event_id=event["event_id"],
            contract_id=b["contract_id"],
            city=event["city"],
            date=event["date"],
            entry_price=entry_price,
            entry_time=times[0],
            shares=shares,
            won_event=b["is_winner"],
        )

        peak = entry_price
        exited = False

        for i, price in enumerate(prices[1:], 1):
            if price > peak:
                peak = price
            trade.peak_price = peak

            # 1. Take profit
            if price >= params.take_profit:
                trade.exit_price = price
                trade.exit_time = times[i]
                trade.exit_reason = "TAKE_PROFIT"
                exited = True
                break

            # 2. Cross-bin exit
            if params.cross_bin_exit < 1.0:
                for sibling in selected:
                    if sibling["contract_id"] == b["contract_id"]:
                        continue
                    if i < len(sibling["prices"]) and sibling["prices"][i] >= params.cross_bin_exit:
                        trade.exit_price = price
                        trade.exit_time = times[i]
                        trade.exit_reason = "CROSS_BIN_EXIT"
                        exited = True
                        break
                if exited:
                    break

            # 3. Trailing stop
            if peak >= entry_price + params.trail_activation:
                if params.use_ratchet:
                    if peak < 0.50:
                        trail = 0.15
                    elif peak < 0.65:
                        trail = 0.12
                    elif peak < 0.80:
                        trail = 0.08
                    else:
                        trail = 0.05
                else:
                    trail = params.trail_pct

                stop_level = peak * (1 - trail)
                if price <= stop_level:
                    trade.exit_price = stop_level
                    trade.exit_time = times[i]
                    trade.exit_reason = "TRAILING_STOP"
                    exited = True
                    break

            # 4. Hard stop
            hard_stop_level = entry_price * (1 - params.hard_stop_pct)
            if price <= hard_stop_level:
                trade.exit_price = hard_stop_level
                trade.exit_time = times[i]
                trade.exit_reason = "HARD_STOP"
                exited = True
                break

        if not exited:
            if b["is_winner"]:
                trade.exit_price = 1.0
                trade.exit_reason = "RESOLVED_WIN"
            else:
                trade.exit_price = 0.0
                trade.exit_reason = "RESOLVED_LOSS"
            trade.exit_time = times[-1]

        trade.pnl = round((trade.exit_price - trade.entry_price) * shares, 2)
        trades.append(trade)

    return trades


def run_backtest(events: list[dict], params: StrategyParams) -> dict:
    """Run a full backtest across all events."""
    all_trades: list[SimTrade] = []

    for event in events:
        trades = simulate_event(event, params)
        all_trades.extend(trades)

    if not all_trades:
        return {"n": 0, "trades": []}

    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in all_trades)

    from collections import Counter
    exit_counts = Counter(t.exit_reason for t in all_trades)

    equity = 0
    peak_equity = 0
    max_dd = 0
    for t in all_trades:
        equity += t.pnl
        if equity > peak_equity:
            peak_equity = equity
        dd = equity - peak_equity
        if dd < max_dd:
            max_dd = dd

    return {
        "n": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(all_trades),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(all_trades),
        "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
        "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0,
        "max_drawdown": max_dd,
        "exit_reasons": dict(exit_counts),
        "trades": all_trades,
        "events_traded": len(set(t.event_id for t in all_trades)),
    }


def print_result(label: str, result: dict, verbose: bool = False):
    if result["n"] == 0:
        print(f"  {label:<40} no trades")
        return
    wr = result["win_rate"] * 100
    print(f"  {label:<40} n={result['n']:<4} wins={result['wins']:<3} "
          f"({wr:4.0f}%)  avg=${result['avg_pnl']:>7,.0f}  "
          f"total=${result['total_pnl']:>9,.0f}  dd=${result['max_drawdown']:>8,.0f}")
    if verbose:
        for reason, count in sorted(result["exit_reasons"].items()):
            reason_trades = [t for t in result["trades"] if t.exit_reason == reason]
            reason_pnl = sum(t.pnl for t in reason_trades)
            print(f"    {reason:<25} n={count:<4} pnl=${reason_pnl:>9,.0f}")


def run_parameter_sweep(events: list[dict]):
    """Test many parameter combinations and find the best."""
    print(f"\n{'='*80}")
    print(f"  PARAMETER SWEEP — {len(events)} resolved events")
    print(f"{'='*80}\n")

    print("--- Entry Price Threshold (1 bin, 12% trail) ---")
    for min_p in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        params = StrategyParams(min_price=min_p, max_bins=1, trail_pct=0.12)
        result = run_backtest(events, params)
        print_result(f"entry>={min_p:.0%}", result)

    print(f"\n--- Number of Bins (entry>=30c, 12% trail) ---")
    for n_bins in [1, 2, 3]:
        params = StrategyParams(min_price=0.30, max_bins=n_bins, trail_pct=0.12)
        result = run_backtest(events, params)
        print_result(f"{n_bins} bin(s)", result)

    print(f"\n--- Trailing Stop % (entry>=30c, 1 bin) ---")
    for trail in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        params = StrategyParams(min_price=0.30, max_bins=1, trail_pct=trail)
        result = run_backtest(events, params)
        print_result(f"trail={trail:.0%}", result)

    print(f"\n--- Take Profit Level (entry>=30c, 1 bin, 12% trail) ---")
    for tp in [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
        label = f"TP={tp:.0%}" if tp < 1.0 else "hold to resolution"
        params = StrategyParams(min_price=0.30, max_bins=1, trail_pct=0.12, take_profit=tp)
        result = run_backtest(events, params)
        print_result(label, result)

    print(f"\n--- Cross-Bin Exit (entry>=30c, 2 bins, 12% trail) ---")
    for xb in [0.60, 0.70, 0.75, 0.80, 0.90, 1.0]:
        label = f"cross-exit@{xb:.0%}" if xb < 1.0 else "no cross-exit"
        params = StrategyParams(min_price=0.30, max_bins=2, trail_pct=0.12, cross_bin_exit=xb)
        result = run_backtest(events, params)
        print_result(label, result)

    # Exhaustive search
    print(f"\n{'='*80}")
    print(f"  TOP 10 COMBINATIONS")
    print(f"{'='*80}\n")

    results = []
    for min_p in [0.20, 0.25, 0.30, 0.35]:
        for n_bins in [1, 2]:
            for trail in [0.10, 0.12, 0.15]:
                for tp in [0.75, 0.90, 1.0]:
                    for xb in [0.75, 1.0]:
                        for hs in [0.30, 0.50, 1.0]:
                            params = StrategyParams(
                                min_price=min_p, max_bins=n_bins,
                                trail_pct=trail, take_profit=tp,
                                cross_bin_exit=xb, hard_stop_pct=hs,
                            )
                            result = run_backtest(events, params)
                            if result["n"] >= 5:
                                results.append((params, result))

    results.sort(key=lambda x: x[1]["total_pnl"], reverse=True)

    print(f"  {'Params':<55} {'N':>4} {'Win%':>5} {'Avg':>8} {'Total':>10}")
    print(f"  {'-'*85}")
    for params, result in results[:10]:
        label = (f"entry>={params.min_price:.0%} bins={params.max_bins} "
                 f"trail={params.trail_pct:.0%} tp={params.take_profit:.0%} "
                 f"xb={params.cross_bin_exit:.0%} hs={params.hard_stop_pct:.0%}")
        wr = result["win_rate"] * 100
        print(f"  {label:<55} {result['n']:>4} {wr:>4.0f}% ${result['avg_pnl']:>7,.0f} ${result['total_pnl']:>9,.0f}")

    if results:
        best_params, best_result = results[0]
        print(f"\n  BEST STRATEGY:")
        print(f"    Entry price:    >= {best_params.min_price:.0%}")
        print(f"    Bins per event: {best_params.max_bins}")
        print(f"    Trailing stop:  {best_params.trail_pct:.0%}")
        print(f"    Take profit:    {best_params.take_profit:.0%}")
        print(f"    Cross-bin exit: {best_params.cross_bin_exit:.0%}")
        print(f"    Hard stop:      {best_params.hard_stop_pct:.0%}")
        print(f"\n    Results: {best_result['n']} trades, {best_result['wins']} wins "
              f"({best_result['win_rate']*100:.0f}%), "
              f"total P&L: ${best_result['total_pnl']:,.0f}, "
              f"avg: ${best_result['avg_pnl']:,.0f}/trade")
        print(f"\n    Exit breakdown:")
        print_result("", best_result, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies on collected price data")
    parser.add_argument("--min-price", type=float, default=None)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--trail", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--db", type=str, default=None, help="Path to prices.db (default: data/prices.db)")
    args = parser.parse_args()

    global DB_PATH
    if args.db:
        DB_PATH = args.db

    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        print("Run collector.py first to start collecting data.")
        return

    events = load_backtestable_events()

    # Stats
    conn = sqlite3.connect(DB_PATH)
    n_snaps = conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    n_resolved = conn.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]
    n_bins = conn.execute("SELECT COUNT(DISTINCT contract_id) FROM price_snapshots").fetchone()[0]
    conn.close()

    print(f"\nDatabase: {DB_PATH}")
    print(f"  {n_resolved} resolved events | {n_bins} bins tracked | {n_snaps:,} price snapshots")
    print(f"  {len(events)} events with backtestable price paths")

    if not events:
        print("\nNo resolved events with sufficient price data yet.")
        print("The collector needs to run for 2-3 days to capture full event lifecycles.")
        return

    total_bins = sum(len(e["bins"]) for e in events)
    print(f"  {total_bins} total bin price paths")
    print(f"  Date range: {events[0]['date']} to {events[-1]['date']}")

    if args.sweep or (args.min_price is None and args.bins is None):
        run_parameter_sweep(events)
    else:
        params = StrategyParams(
            min_price=args.min_price or 0.25,
            max_price=args.max_price or 0.50,
            max_bins=args.bins or 2,
            trail_pct=(args.trail or 12) / 100,
        )
        result = run_backtest(events, params)
        print(f"\nCustom backtest: entry>={params.min_price:.0%}, "
              f"{params.max_bins} bins, {params.trail_pct:.0%} trail")
        print_result("Result", result, verbose=True)


if __name__ == "__main__":
    main()
