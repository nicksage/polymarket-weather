# ML Distribution Model — Evaluation Report

**City:** Wellington
**Generated:** 2026-04-25T00:42:36.519070+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.115 °C** |
| Mean bias | +0.079 °C |
| Fallback σ | 1.112 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.115** | — | — |
| persistence_yesterday | 2.239 | +1.125 | ML wins |
| yoy_same_date_last_year | 3.180 | +2.065 | ML wins |
| lag_7d_ago | 3.135 | +2.020 | ML wins |
| climatology_doy_mean | 2.376 | +1.261 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.8% | 68.3% | well-calibrated |
| ±2σ | 94.4% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.72 | 1.69 | +0.31 |
| 02 | 508 | 1.58 | 1.56 | +0.24 |
| 03 | 546 | 1.15 | 1.12 | +0.25 |
| 04 | 480 | 0.90 | 0.90 | -0.03 |
| 05 | 496 | 0.82 | 0.82 | -0.07 |
| 06 | 480 | 0.82 | 0.82 | +0.07 |
| 07 | 496 | 0.80 | 0.80 | +0.04 |
| 08 | 496 | 0.73 | 0.73 | +0.01 |
| 09 | 480 | 0.83 | 0.83 | +0.03 |
| 10 | 496 | 0.95 | 0.95 | -0.04 |
| 11 | 486 | 1.05 | 1.05 | +0.06 |
| 12 | 558 | 1.32 | 1.32 | +0.03 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.167 | +0.112 |
| 12:00 | 3040 | 1.060 | +0.046 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.69 |
| 02 | 1.56 |
| 03 | 1.12 |
| 04 | 0.90 |
| 05 | 0.82 |
| 06 | 0.82 |
| 07 | 0.80 |
| 08 | 0.73 |
| 09 | 0.83 |
| 10 | 0.95 |
| 11 | 1.05 |
| 12 | 1.32 |
| _fallback_ | _1.11_ |
