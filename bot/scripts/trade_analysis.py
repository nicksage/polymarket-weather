"""
trade_analysis.py — Analyze closed paper trades to find optimal entry thresholds.

Runs three analyses:
  1. Calibration: Is the model well-calibrated? (reliability diagram)
  2. Feature breakdown: Which features predict winning trades?
  3. Logistic regression: What combination of features best predicts wins?

Usage:
    cd bot
    python scripts/trade_analysis.py
"""

import sqlite3
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

def load_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    trades = pd.read_sql("""
        SELECT
            p.*,
            julianday(p.date) - julianday(date(p.entry_time)) as days_ahead
        FROM positions p
        WHERE p.status = 'closed'
          AND p.pnl IS NOT NULL
          AND p.model_prob IS NOT NULL
    """, conn)

    # Enrich with entry snapshot data (forecast_agreement, adjusted_mu, etc.)
    snapshots = pd.read_sql("""
        SELECT
            id, contract_id, forecast_agreement_c, blended_mu_c, blended_sigma_c,
            adjusted_mu_c, adjusted_sigma_c, observed_max_so_far_c,
            live_adjustment_score, current_temp_c
        FROM decision_snapshots
    """, conn)

    conn.close()

    trades["won"] = (trades["pnl"] > 0).astype(int)
    trades["days_ahead"] = trades["days_ahead"].clip(lower=0).round().astype(int)

    # Merge entry snapshot
    if not snapshots.empty and "entry_snapshot_id" in trades.columns:
        snap_entry = snapshots.rename(columns={
            "id": "entry_snapshot_id",
            "forecast_agreement_c": "entry_agreement_c",
            "blended_sigma_c": "entry_blended_sigma_c",
            "live_adjustment_score": "entry_live_adj_score",
        })
        trades = trades.merge(
            snap_entry[["entry_snapshot_id", "entry_agreement_c",
                         "entry_blended_sigma_c", "entry_live_adj_score"]],
            on="entry_snapshot_id", how="left"
        )

    return trades


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def overview(df):
    print_header("TRADE OVERVIEW")
    n = len(df)
    wins = df["won"].sum()
    losses = n - wins
    total_pnl = df["pnl"].sum()
    avg_win = df.loc[df["won"]==1, "pnl"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["won"]==0, "pnl"].mean() if losses > 0 else 0
    print(f"  Total closed trades:  {n}")
    print(f"  Wins:                 {wins} ({wins*100/n:.0f}%)")
    print(f"  Losses:               {losses} ({losses*100/n:.0f}%)")
    print(f"  Total P&L:            ${total_pnl:,.2f}")
    print(f"  Avg winning trade:    ${avg_win:,.2f}")
    print(f"  Avg losing trade:     ${avg_loss:,.2f}")
    print(f"  Win/Loss ratio:       {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print(f"  Expectancy per trade: ${total_pnl/n:,.2f}")

    # Categorize exit reasons
    df["exit_category"] = "other"
    df.loc[df["exit_reason"].str.contains("HARD_STOP|hard_stop", na=False), "exit_category"] = "HARD_STOP"
    df.loc[df["exit_reason"].str.contains("RT_HARD_STOP", na=False), "exit_category"] = "RT_STOP_LOSS"
    df.loc[df["exit_reason"].str.contains("INVALIDATED", na=False), "exit_category"] = "INVALIDATED"
    df.loc[df["exit_reason"].str.contains("DYING", na=False), "exit_category"] = "DYING"
    df.loc[df["exit_reason"].str.contains("TOP_BIN_CONFIRMED", na=False), "exit_category"] = "TOP_BIN_CONFIRMED"
    df.loc[df["exit_reason"].isna(), "exit_category"] = "RESOLVED"

    print(f"\n  Exit reason breakdown:")
    cats = df.groupby("exit_category").agg(
        count=("pnl", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).sort_values("count", ascending=False)
    for cat, row in cats.iterrows():
        wr = row["wins"] / row["count"] * 100
        print(f"    {cat:<20} n={int(row['count']):<4} wins={int(row['wins'])} ({wr:4.0f}%)  "
              f"total=${row['total_pnl']:>9,.2f}  avg=${row['avg_pnl']:>8,.2f}")


def calibration_analysis(df):
    print_header("CALIBRATION ANALYSIS")
    print("  Does model_prob match actual win rate?\n")

    bins = [0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.01]
    labels = ["<15%", "15-20%", "20-25%", "25-30%", "30-40%", "40-50%", "50-70%", "70%+"]
    df["prob_bin"] = pd.cut(df["model_prob"], bins=bins, labels=labels, right=False)

    print(f"  {'Bin':<10} {'N':>5} {'Wins':>5} {'Win%':>6} {'Model Avg':>10} {'Gap':>8} {'Total P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-'*67}")
    for label in labels:
        subset = df[df["prob_bin"] == label]
        if len(subset) == 0:
            continue
        n = len(subset)
        wins = subset["won"].sum()
        wr = wins / n * 100
        avg_mp = subset["model_prob"].mean() * 100
        gap = wr - avg_mp
        total = subset["pnl"].sum()
        avg = subset["pnl"].mean()
        flag = " <-- OVERCONFIDENT" if gap < -15 and n >= 5 else ""
        flag = " <-- UNDERCONFIDENT" if gap > 15 and n >= 5 else flag
        print(f"  {label:<10} {n:>5} {wins:>5} {wr:>5.0f}% {avg_mp:>9.1f}% {gap:>+7.1f}% ${total:>11,.2f} ${avg:>9,.2f}{flag}")

    print(f"\n  Interpretation:")
    print(f"  - If Win% is consistently below Model Avg, the model is OVERCONFIDENT")
    print(f"  - If Win% is consistently above Model Avg, the model is UNDERCONFIDENT")
    print(f"  - The 'Gap' column shows the calibration error per bin")


def feature_breakdown(df):
    print_header("FEATURE BREAKDOWN — Win Rate by Feature Bucket")

    features = [
        ("model_prob", "Model Probability", [0, 0.15, 0.20, 0.25, 0.30, 0.40, 1.01],
         ["<15%", "15-20%", "20-25%", "25-30%", "30-40%", "40%+"]),
        ("entry_price", "Entry Price (Market Prob)", [0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01],
         ["<15c", "15-20c", "20-25c", "25-30c", "30-40c", "40-50c", "50c+"]),
        ("edge", "Edge (Model - Market)", [-1, 0, 0.03, 0.05, 0.10, 0.20, 1.01],
         ["<0%", "0-3%", "3-5%", "5-10%", "10-20%", "20%+"]),
        ("forecast_sigma_c", "Forecast Sigma (Uncertainty)", [0, 1.5, 1.7, 1.9, 2.1, 10],
         ["<1.5", "1.5-1.7", "1.7-1.9", "1.9-2.1", "2.1+"]),
        ("days_ahead", "Days Ahead", [-0.5, 0.5, 1.5, 2.5, 10],
         ["D+0", "D+1", "D+2", "D+3+"]),
    ]

    for col, title, bins, labels in features:
        if col not in df.columns or df[col].isna().all():
            continue

        print(f"\n  {title}:")
        df["_bucket"] = pd.cut(df[col].astype(float), bins=bins, labels=labels, right=False)
        print(f"  {'Bucket':<10} {'N':>5} {'Wins':>5} {'Win%':>6} {'Avg P&L':>10} {'Total P&L':>12}")
        print(f"  {'-'*52}")
        for label in labels:
            subset = df[df["_bucket"] == label]
            if len(subset) == 0:
                continue
            n = len(subset)
            wins = subset["won"].sum()
            wr = wins / n * 100
            total = subset["pnl"].sum()
            avg = subset["pnl"].mean()
            marker = " ***" if wr > 30 and n >= 5 else ""
            print(f"  {label:<10} {n:>5} {wins:>5} {wr:>5.0f}% ${avg:>9,.2f} ${total:>11,.2f}{marker}")
        df.drop(columns=["_bucket"], inplace=True)

    # Forecast agreement (from snapshot)
    if "entry_agreement_c" in df.columns and not df["entry_agreement_c"].isna().all():
        print(f"\n  Forecast Agreement (ECMWF vs GFS gap in C):")
        bins_a = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 100]
        labels_a = ["<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0+"]
        df["_bucket"] = pd.cut(df["entry_agreement_c"].astype(float), bins=bins_a, labels=labels_a, right=False)
        print(f"  {'Bucket':<10} {'N':>5} {'Wins':>5} {'Win%':>6} {'Avg P&L':>10} {'Total P&L':>12}")
        print(f"  {'-'*52}")
        for label in labels_a:
            subset = df[df["_bucket"] == label]
            if len(subset) == 0:
                continue
            n = len(subset)
            wins = subset["won"].sum()
            wr = wins / n * 100
            total = subset["pnl"].sum()
            avg = subset["pnl"].mean()
            print(f"  {label:<10} {n:>5} {wins:>5} {wr:>5.0f}% ${avg:>9,.2f} ${total:>11,.2f}")
        df.drop(columns=["_bucket"], inplace=True)

    # City-level breakdown
    print(f"\n  Win Rate by City (top 15 by trade count):")
    city_stats = df.groupby("city").agg(
        count=("pnl", "count"),
        wins=("won", "sum"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).sort_values("count", ascending=False).head(15)
    print(f"  {'City':<20} {'N':>5} {'Wins':>5} {'Win%':>6} {'Avg P&L':>10} {'Total P&L':>12}")
    print(f"  {'-'*62}")
    for city, row in city_stats.iterrows():
        wr = row["wins"] / row["count"] * 100
        print(f"  {city:<20} {int(row['count']):>5} {int(row['wins']):>5} {wr:>5.0f}% "
              f"${row['avg_pnl']:>9,.2f} ${row['total_pnl']:>11,.2f}")


def logistic_regression_analysis(df):
    print_header("LOGISTIC REGRESSION — Which Features Predict Wins?")

    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        print("  scikit-learn not installed. Run: pip install scikit-learn")
        return

    feature_cols = ["model_prob", "entry_price", "edge", "forecast_sigma_c", "days_ahead"]
    if "entry_agreement_c" in df.columns:
        feature_cols.append("entry_agreement_c")

    valid = df.dropna(subset=feature_cols + ["won"])
    if len(valid) < 30:
        print(f"  Only {len(valid)} trades with complete data — need 30+ for regression")
        return

    X = valid[feature_cols].astype(float).values
    y = valid["won"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression with cross-validation
    model = LogisticRegressionCV(cv=5, penalty="l2", max_iter=1000, scoring="roc_auc")
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")

    print(f"  Model AUC (5-fold CV): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  (0.50 = random, 0.70+ = useful, 0.80+ = strong)\n")

    # Feature importance (standardized coefficients)
    coefs = model.coef_[0]
    importance = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Feature':<25} {'Coefficient':>12} {'Direction':>12}")
    print(f"  {'-'*51}")
    for feat, coef in importance:
        direction = "HELPS win" if coef > 0 else "HURTS win"
        bar = "+" * min(int(abs(coef) * 10), 20) if coef > 0 else "-" * min(int(abs(coef) * 10), 20)
        print(f"  {feat:<25} {coef:>+11.3f}  {direction:<12} {bar}")

    print(f"\n  Interpretation:")
    print(f"  - Positive coefficient = higher values of this feature increase win probability")
    print(f"  - Larger absolute value = stronger effect")
    print(f"  - Coefficients are standardized (comparable across features)")

    # Decision tree for interpretable rules
    print(f"\n  --- Decision Tree Rules (depth=3) ---\n")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, len(valid)//20))
    tree.fit(X, y)

    from sklearn.tree import export_text
    rules = export_text(tree, feature_names=feature_cols, decimals=3)
    print(f"  {rules}")

    # Tree-based feature importance
    tree_imp = sorted(zip(feature_cols, tree.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"  Tree feature importance:")
    for feat, imp in tree_imp:
        if imp > 0:
            bar = "#" * int(imp * 40)
            print(f"    {feat:<25} {imp:.3f} {bar}")


def threshold_simulation(df):
    print_header("THRESHOLD SIMULATION — What If We Changed Entry Criteria?")

    print(f"  Simulating different model_prob and market_prob thresholds:\n")

    model_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    market_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    print(f"  {'ModelP':>7} {'MktP':>7} {'Trades':>7} {'Wins':>5} {'Win%':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Expectancy':>11}")
    print(f"  {'-'*75}")

    best_pnl = -float("inf")
    best_combo = None

    for mp_thresh in model_thresholds:
        for mk_thresh in market_thresholds:
            subset = df[
                (df["model_prob"] >= mp_thresh) &
                (df["entry_price"] >= mk_thresh)
            ]
            if len(subset) < 5:
                continue
            n = len(subset)
            wins = subset["won"].sum()
            wr = wins / n * 100
            total = subset["pnl"].sum()
            avg = subset["pnl"].mean()
            marker = " <-- CURRENT" if mp_thresh == 0.15 and mk_thresh == 0.15 else ""
            if total > best_pnl and n >= 10:
                best_pnl = total
                best_combo = (mp_thresh, mk_thresh, n, wins, wr, total, avg)
            print(f"  {mp_thresh:>7.2f} {mk_thresh:>7.2f} {n:>7} {wins:>5} {wr:>5.0f}% "
                  f"${total:>11,.2f} ${avg:>9,.2f} ${avg:>10,.2f}{marker}")

    if best_combo:
        print(f"\n  Best combo (by total P&L, min 10 trades):")
        print(f"    model_prob >= {best_combo[0]:.2f}, market_prob >= {best_combo[1]:.2f}")
        print(f"    {best_combo[2]} trades, {best_combo[3]} wins ({best_combo[4]:.0f}%), "
              f"total=${best_combo[5]:,.2f}, avg=${best_combo[6]:,.2f}")

    # Edge threshold simulation
    print(f"\n  Edge threshold simulation (keeping current model/market thresholds):\n")
    print(f"  {'Min Edge':>9} {'Trades':>7} {'Wins':>5} {'Win%':>6} {'Total P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-'*53}")

    for edge_thresh in [0.0, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20]:
        subset = df[df["edge"] >= edge_thresh]
        if len(subset) < 3:
            continue
        n = len(subset)
        wins = subset["won"].sum()
        wr = wins / n * 100
        total = subset["pnl"].sum()
        avg = subset["pnl"].mean()
        print(f"  {edge_thresh:>8.0%} {n:>7} {wins:>5} {wr:>5.0f}% ${total:>11,.2f} ${avg:>9,.2f}")


def closing_line_analysis(df):
    print_header("CLOSING LINE VALUE (CLV) ANALYSIS")

    # For resolved trades (exit_reason is None = market resolved naturally),
    # we can compute CLV as: exit_price - entry_price for winners (exit=1.0)
    # For stopped-out trades, CLV is approximated by current_price at exit vs entry
    resolved = df[df["exit_reason"].isna()].copy()
    stopped = df[df["exit_reason"].notna()].copy()

    if len(resolved) > 0:
        print(f"  Resolved trades (market settled): {len(resolved)}")
        print(f"    Won: {resolved['won'].sum()}, Lost: {len(resolved) - resolved['won'].sum()}")
        print(f"    Avg entry_price: ${resolved['entry_price'].mean():.3f}")
        print(f"    Avg P&L: ${resolved['pnl'].mean():,.2f}")
    else:
        print(f"  No naturally resolved trades yet (all exited via stop/exit engine)")

    if len(stopped) > 0:
        # For stopped trades: did the market move toward or away from our entry?
        stopped["price_move"] = stopped["current_price"] - stopped["entry_price"]
        moved_against = (stopped["price_move"] < 0).sum()
        moved_toward = (stopped["price_move"] > 0).sum()
        print(f"\n  Stopped/exited trades: {len(stopped)}")
        print(f"    Market moved AGAINST entry: {moved_against} ({moved_against*100/len(stopped):.0f}%)")
        print(f"    Market moved TOWARD entry:  {moved_toward} ({moved_toward*100/len(stopped):.0f}%)")
        print(f"    Avg price move: {stopped['price_move'].mean():+.4f}")
        print(f"\n    If market consistently moves against you after entry,")
        print(f"    the model may be systematically wrong about direction.")


def mfe_mae_analysis(df):
    print_header("MFE / MAE ANALYSIS — Are Stop Losses Set Correctly?")

    has_mfe = df["max_favorable_excursion"].notna().sum()
    has_mae = df["max_adverse_excursion"].notna().sum()

    if has_mfe < 10 or has_mae < 10:
        print(f"  Only {has_mfe} trades with MFE data, {has_mae} with MAE — need more data")
        return

    valid = df.dropna(subset=["max_favorable_excursion", "max_adverse_excursion"]).copy()

    winners = valid[valid["won"] == 1]
    losers = valid[valid["won"] == 0]

    print(f"  Winners ({len(winners)}):")
    if len(winners) > 0:
        print(f"    Avg MFE (max unrealized gain):  ${winners['max_favorable_excursion'].mean():>8,.2f}")
        print(f"    Avg MAE (max unrealized loss):   ${winners['max_adverse_excursion'].mean():>8,.2f}")
        print(f"    Avg realized P&L:                ${winners['pnl'].mean():>8,.2f}")
        pct_captured = winners["pnl"].mean() / winners["max_favorable_excursion"].mean() * 100 if winners["max_favorable_excursion"].mean() != 0 else 0
        print(f"    % of MFE captured as profit:     {pct_captured:.0f}%")

    print(f"\n  Losers ({len(losers)}):")
    if len(losers) > 0:
        print(f"    Avg MFE (max unrealized gain):  ${losers['max_favorable_excursion'].mean():>8,.2f}")
        print(f"    Avg MAE (max unrealized loss):   ${losers['max_adverse_excursion'].mean():>8,.2f}")
        print(f"    Avg realized P&L:                ${losers['pnl'].mean():>8,.2f}")

        # How many losers were once profitable?
        once_profitable = (losers["max_favorable_excursion"] > 0).sum()
        print(f"    Losers that were once profitable: {once_profitable}/{len(losers)} ({once_profitable*100/len(losers):.0f}%)")
        if once_profitable > 0:
            avg_mfe_of_eventual_losers = losers.loc[losers["max_favorable_excursion"] > 0, "max_favorable_excursion"].mean()
            print(f"    Their avg peak profit:           ${avg_mfe_of_eventual_losers:>8,.2f}")
            print(f"\n    -> These trades turned from winners to losers.")
            print(f"       Consider adding a trailing stop or take-profit at ~${avg_mfe_of_eventual_losers/2:,.0f}")


def main():
    print("\n" + "=" * 70)
    print("  POLYMARKET WEATHER BOT — TRADE ANALYSIS")
    print("=" * 70)

    df = load_data()
    if len(df) < 5:
        print("Not enough closed trades for analysis")
        return

    overview(df)
    calibration_analysis(df)
    feature_breakdown(df)
    threshold_simulation(df)
    closing_line_analysis(df)
    mfe_mae_analysis(df)

    try:
        logistic_regression_analysis(df)
    except Exception as e:
        print(f"\n  Logistic regression failed: {e}")

    print_header("NEXT STEPS")
    print("""  1. If calibration shows overconfidence (Gap < -15%):
     -> The model thinks bins are more likely than they are
     -> Raise TBV_MIN_MODEL_PROB or recalibrate the ensemble weights

  2. If a feature strongly predicts wins (e.g., days_ahead=0 wins more):
     -> Consider adding it as an entry filter or sizing factor

  3. If threshold simulation shows a better combo than current (0.15/0.15):
     -> Test the new thresholds with walk-forward validation before deploying

  4. If MFE analysis shows losers were once profitable:
     -> Consider adding a trailing stop or take-profit level

  5. If CLV is negative (market moves against you after entry):
     -> The model's directional prediction may need improvement
""")


if __name__ == "__main__":
    main()
