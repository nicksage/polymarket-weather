# ML Distribution Model — Evaluation Report

**City:** San Francisco
**Generated:** 2026-04-25T00:40:48.363501+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.247 °C** |
| Mean bias | -0.024 °C |
| Fallback σ | 1.247 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.247** | — | — |
| persistence_yesterday | 2.555 | +1.308 | ML wins |
| yoy_same_date_last_year | 4.387 | +3.140 | ML wins |
| lag_7d_ago | 4.131 | +2.884 | ML wins |
| climatology_doy_mean | 3.284 | +2.037 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.1% | 68.3% | well-calibrated |
| ±2σ | 94.4% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.13 | 1.13 | +0.02 |
| 02 | 508 | 1.22 | 1.22 | +0.03 |
| 03 | 546 | 1.30 | 1.30 | +0.07 |
| 04 | 480 | 1.31 | 1.31 | -0.02 |
| 05 | 496 | 1.16 | 1.16 | -0.03 |
| 06 | 480 | 1.21 | 1.20 | -0.13 |
| 07 | 496 | 1.17 | 1.16 | -0.13 |
| 08 | 496 | 1.30 | 1.30 | -0.05 |
| 09 | 480 | 1.54 | 1.54 | +0.07 |
| 10 | 496 | 1.49 | 1.49 | +0.04 |
| 11 | 486 | 1.08 | 1.07 | -0.13 |
| 12 | 558 | 1.00 | 0.99 | -0.05 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.380 | +0.028 |
| 12:00 | 3040 | 1.099 | -0.076 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.13 |
| 02 | 1.22 |
| 03 | 1.30 |
| 04 | 1.31 |
| 05 | 1.16 |
| 06 | 1.20 |
| 07 | 1.16 |
| 08 | 1.30 |
| 09 | 1.54 |
| 10 | 1.49 |
| 11 | 1.07 |
| 12 | 0.99 |
| _fallback_ | _1.25_ |
