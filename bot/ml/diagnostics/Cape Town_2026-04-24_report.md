# ML Distribution Model — Evaluation Report

**City:** Cape Town
**Generated:** 2026-04-25T00:37:32.964326+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7260  (unique dates: 3630, OOF predictions: 6050)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.650 °C** |
| Mean bias | -0.055 °C |
| Fallback σ | 1.649 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.650** | — | — |
| persistence_yesterday | 3.705 | +2.056 | ML wins |
| yoy_same_date_last_year | 5.331 | +3.681 | ML wins |
| lag_7d_ago | 5.224 | +3.574 | ML wins |
| climatology_doy_mean | 3.961 | +2.312 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.8% | 68.3% | well-calibrated |
| ±2σ | 94.4% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 548 | 1.61 | 1.60 | +0.08 |
| 02 | 472 | 1.78 | 1.76 | +0.23 |
| 03 | 546 | 1.86 | 1.86 | -0.05 |
| 04 | 480 | 1.61 | 1.61 | -0.14 |
| 05 | 496 | 1.89 | 1.89 | -0.11 |
| 06 | 480 | 1.74 | 1.74 | -0.05 |
| 07 | 496 | 1.57 | 1.56 | -0.16 |
| 08 | 496 | 1.53 | 1.53 | -0.07 |
| 09 | 480 | 1.60 | 1.60 | -0.08 |
| 10 | 496 | 1.54 | 1.53 | -0.14 |
| 11 | 502 | 1.55 | 1.55 | -0.03 |
| 12 | 558 | 1.47 | 1.46 | -0.13 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3025 | 1.832 | -0.041 |
| 12:00 | 3025 | 1.444 | -0.070 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.60 |
| 02 | 1.76 |
| 03 | 1.86 |
| 04 | 1.61 |
| 05 | 1.89 |
| 06 | 1.74 |
| 07 | 1.56 |
| 08 | 1.53 |
| 09 | 1.60 |
| 10 | 1.53 |
| 11 | 1.55 |
| 12 | 1.46 |
| _fallback_ | _1.65_ |
