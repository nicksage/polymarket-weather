# ML Distribution Model — Evaluation Report

**City:** Seoul
**Generated:** 2026-04-25T00:41:12.233538+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.489 °C** |
| Mean bias | -0.018 °C |
| Fallback σ | 1.489 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.489** | — | — |
| persistence_yesterday | 3.273 | +1.784 | ML wins |
| yoy_same_date_last_year | 5.336 | +3.847 | ML wins |
| lag_7d_ago | 5.383 | +3.895 | ML wins |
| climatology_doy_mean | 4.024 | +2.536 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.4% | 68.3% | well-calibrated |
| ±2σ | 95.3% | 95.4% | well-calibrated |
| ±3σ | 99.3% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.80 | 1.79 | -0.17 |
| 02 | 508 | 1.57 | 1.57 | -0.03 |
| 03 | 546 | 1.54 | 1.54 | +0.07 |
| 04 | 480 | 1.68 | 1.68 | +0.07 |
| 05 | 496 | 1.40 | 1.40 | -0.04 |
| 06 | 480 | 1.32 | 1.32 | +0.06 |
| 07 | 496 | 1.42 | 1.42 | +0.11 |
| 08 | 496 | 1.23 | 1.22 | +0.15 |
| 09 | 480 | 1.32 | 1.32 | +0.02 |
| 10 | 496 | 1.45 | 1.41 | -0.32 |
| 11 | 486 | 1.43 | 1.43 | +0.06 |
| 12 | 558 | 1.54 | 1.53 | -0.17 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.706 | +0.119 |
| 12:00 | 3040 | 1.234 | -0.155 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.79 |
| 02 | 1.57 |
| 03 | 1.54 |
| 04 | 1.68 |
| 05 | 1.40 |
| 06 | 1.32 |
| 07 | 1.42 |
| 08 | 1.22 |
| 09 | 1.32 |
| 10 | 1.41 |
| 11 | 1.43 |
| 12 | 1.53 |
| _fallback_ | _1.49_ |
