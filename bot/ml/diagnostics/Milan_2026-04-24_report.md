# ML Distribution Model — Evaluation Report

**City:** Milan
**Generated:** 2026-04-25T00:40:07.519748+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.434 °C** |
| Mean bias | -0.114 °C |
| Fallback σ | 1.430 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.434** | — | — |
| persistence_yesterday | 2.689 | +1.255 | ML wins |
| yoy_same_date_last_year | 4.857 | +3.423 | ML wins |
| lag_7d_ago | 4.311 | +2.877 | ML wins |
| climatology_doy_mean | 3.467 | +2.033 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.8% | 68.3% | well-calibrated |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.64 | 1.63 | -0.19 |
| 02 | 508 | 1.80 | 1.80 | -0.04 |
| 03 | 546 | 1.71 | 1.62 | -0.54 |
| 04 | 480 | 1.41 | 1.40 | -0.12 |
| 05 | 496 | 1.45 | 1.45 | -0.03 |
| 06 | 480 | 1.13 | 1.13 | -0.06 |
| 07 | 496 | 1.13 | 1.13 | +0.08 |
| 08 | 496 | 1.19 | 1.18 | -0.13 |
| 09 | 480 | 1.24 | 1.24 | -0.04 |
| 10 | 496 | 1.45 | 1.45 | -0.06 |
| 11 | 486 | 1.26 | 1.26 | -0.09 |
| 12 | 558 | 1.50 | 1.50 | -0.10 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.585 | -0.070 |
| 12:00 | 3040 | 1.265 | -0.158 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.63 |
| 02 | 1.80 |
| 03 | 1.62 |
| 04 | 1.40 |
| 05 | 1.45 |
| 06 | 1.13 |
| 07 | 1.13 |
| 08 | 1.18 |
| 09 | 1.24 |
| 10 | 1.45 |
| 11 | 1.26 |
| 12 | 1.50 |
| _fallback_ | _1.43_ |
