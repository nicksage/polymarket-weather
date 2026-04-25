# ML Distribution Model — Evaluation Report

**City:** Wuhan
**Generated:** 2026-04-25T00:42:42.255555+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.440 °C** |
| Mean bias | -0.006 °C |
| Fallback σ | 1.440 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.440** | — | — |
| persistence_yesterday | 3.106 | +1.665 | ML wins |
| yoy_same_date_last_year | 6.034 | +4.594 | ML wins |
| lag_7d_ago | 6.208 | +4.768 | ML wins |
| climatology_doy_mean | 4.563 | +3.123 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.6% | 68.3% | well-calibrated |
| ±2σ | 95.0% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.89 | 1.85 | -0.42 |
| 02 | 508 | 1.57 | 1.57 | -0.12 |
| 03 | 546 | 1.42 | 1.42 | +0.07 |
| 04 | 480 | 1.57 | 1.57 | +0.05 |
| 05 | 496 | 1.44 | 1.44 | +0.06 |
| 06 | 480 | 1.36 | 1.35 | +0.07 |
| 07 | 496 | 1.22 | 1.22 | -0.06 |
| 08 | 496 | 1.18 | 1.18 | +0.08 |
| 09 | 480 | 1.30 | 1.28 | +0.23 |
| 10 | 496 | 1.31 | 1.30 | +0.11 |
| 11 | 486 | 1.31 | 1.31 | +0.07 |
| 12 | 558 | 1.49 | 1.48 | -0.13 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.611 | +0.059 |
| 12:00 | 3040 | 1.246 | -0.070 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.85 |
| 02 | 1.57 |
| 03 | 1.42 |
| 04 | 1.57 |
| 05 | 1.44 |
| 06 | 1.35 |
| 07 | 1.22 |
| 08 | 1.18 |
| 09 | 1.28 |
| 10 | 1.30 |
| 11 | 1.31 |
| 12 | 1.48 |
| _fallback_ | _1.44_ |
