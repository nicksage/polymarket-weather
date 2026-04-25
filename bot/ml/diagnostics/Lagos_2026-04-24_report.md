# ML Distribution Model — Evaluation Report

**City:** Lagos
**Generated:** 2026-04-25T00:39:13.863706+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7148  (unique dates: 3574, OOF predictions: 5950)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.364 °C** |
| Mean bias | +0.063 °C |
| Fallback σ | 1.363 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.364** | — | — |
| persistence_yesterday | 2.026 | +0.662 | ML wins |
| yoy_same_date_last_year | 2.390 | +1.026 | ML wins |
| lag_7d_ago | 2.202 | +0.837 | ML wins |
| climatology_doy_mean | 1.746 | +0.382 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.7% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 522 | 1.29 | 1.28 | +0.19 |
| 02 | 500 | 1.39 | 1.35 | +0.33 |
| 03 | 538 | 1.40 | 1.40 | +0.06 |
| 04 | 480 | 1.15 | 1.15 | +0.02 |
| 05 | 496 | 1.15 | 1.15 | +0.03 |
| 06 | 474 | 1.43 | 1.43 | +0.03 |
| 07 | 496 | 1.07 | 1.07 | -0.04 |
| 08 | 496 | 1.11 | 1.09 | +0.24 |
| 09 | 480 | 1.49 | 1.49 | -0.02 |
| 10 | 496 | 1.43 | 1.42 | -0.22 |
| 11 | 478 | 1.95 | 1.95 | +0.04 |
| 12 | 494 | 1.30 | 1.30 | +0.08 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 2975 | 1.471 | +0.103 |
| 12:00 | 2975 | 1.249 | +0.024 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.28 |
| 02 | 1.35 |
| 03 | 1.40 |
| 04 | 1.15 |
| 05 | 1.15 |
| 06 | 1.43 |
| 07 | 1.07 |
| 08 | 1.09 |
| 09 | 1.49 |
| 10 | 1.42 |
| 11 | 1.95 |
| 12 | 1.30 |
| _fallback_ | _1.36_ |
