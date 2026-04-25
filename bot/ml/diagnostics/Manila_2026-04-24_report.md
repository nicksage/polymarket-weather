# ML Distribution Model — Evaluation Report

**City:** Manila
**Generated:** 2026-04-25T00:39:46.817383+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.939 °C** |
| Mean bias | +0.034 °C |
| Fallback σ | 0.939 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.939** | — | — |
| persistence_yesterday | 1.575 | +0.636 | ML wins |
| yoy_same_date_last_year | 2.207 | +1.268 | ML wins |
| lag_7d_ago | 2.053 | +1.114 | ML wins |
| climatology_doy_mean | 1.659 | +0.720 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.8% | 68.3% | well-calibrated |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 0.87 | 0.87 | +0.04 |
| 02 | 508 | 0.99 | 0.98 | +0.19 |
| 03 | 546 | 0.96 | 0.96 | +0.07 |
| 04 | 480 | 0.92 | 0.92 | -0.03 |
| 05 | 496 | 0.89 | 0.88 | +0.05 |
| 06 | 480 | 0.97 | 0.97 | -0.00 |
| 07 | 496 | 0.92 | 0.92 | +0.01 |
| 08 | 496 | 0.94 | 0.94 | -0.03 |
| 09 | 480 | 0.90 | 0.90 | +0.01 |
| 10 | 496 | 0.95 | 0.95 | +0.03 |
| 11 | 486 | 0.96 | 0.96 | +0.08 |
| 12 | 558 | 1.00 | 1.00 | -0.01 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.040 | +0.075 |
| 12:00 | 3040 | 0.826 | -0.007 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 0.87 |
| 02 | 0.98 |
| 03 | 0.96 |
| 04 | 0.92 |
| 05 | 0.88 |
| 06 | 0.97 |
| 07 | 0.92 |
| 08 | 0.94 |
| 09 | 0.90 |
| 10 | 0.95 |
| 11 | 0.96 |
| 12 | 1.00 |
| _fallback_ | _0.94_ |
