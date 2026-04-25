# ML Distribution Model — Evaluation Report

**City:** Mexico City
**Generated:** 2026-04-25T00:39:53.533315+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **2.355 °C** |
| Mean bias | -0.271 °C |
| Fallback σ | 2.339 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **2.355** | — | — |
| persistence_yesterday | 2.800 | +0.445 | ML wins |
| yoy_same_date_last_year | 3.698 | +1.343 | ML wins |
| lag_7d_ago | 3.523 | +1.168 | ML wins |
| climatology_doy_mean | 2.736 | +0.381 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 67.8% | 68.3% | well-calibrated |
| ±2σ | 95.5% | 95.4% | well-calibrated |
| ±3σ | 99.5% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.42 | 2.41 | -0.19 |
| 02 | 508 | 2.88 | 2.86 | +0.31 |
| 03 | 546 | 2.58 | 2.57 | +0.16 |
| 04 | 480 | 2.74 | 2.66 | -0.66 |
| 05 | 496 | 2.56 | 2.54 | -0.34 |
| 06 | 480 | 2.51 | 2.50 | -0.14 |
| 07 | 496 | 1.88 | 1.84 | -0.36 |
| 08 | 496 | 1.76 | 1.72 | -0.36 |
| 09 | 480 | 1.71 | 1.68 | -0.29 |
| 10 | 496 | 2.19 | 2.15 | -0.39 |
| 11 | 486 | 2.19 | 2.16 | -0.31 |
| 12 | 558 | 2.46 | 2.35 | -0.73 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 2.378 | -0.198 |
| 12:00 | 3040 | 2.332 | -0.345 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.41 |
| 02 | 2.86 |
| 03 | 2.57 |
| 04 | 2.66 |
| 05 | 2.54 |
| 06 | 2.50 |
| 07 | 1.84 |
| 08 | 1.72 |
| 09 | 1.68 |
| 10 | 2.15 |
| 11 | 2.16 |
| 12 | 2.35 |
| _fallback_ | _2.34_ |
