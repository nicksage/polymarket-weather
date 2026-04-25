# ML Distribution Model — Evaluation Report

**City:** Toronto
**Generated:** 2026-04-25T00:42:17.375662+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.632 °C** |
| Mean bias | -0.200 °C |
| Fallback σ | 1.619 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.632** | — | — |
| persistence_yesterday | 3.809 | +2.177 | ML wins |
| yoy_same_date_last_year | 5.944 | +4.312 | ML wins |
| lag_7d_ago | 5.952 | +4.320 | ML wins |
| climatology_doy_mean | 4.524 | +2.892 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.9% | 68.3% | well-calibrated |
| ±2σ | 95.2% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.09 | 2.04 | -0.46 |
| 02 | 508 | 1.83 | 1.82 | -0.23 |
| 03 | 546 | 1.76 | 1.76 | -0.11 |
| 04 | 480 | 1.73 | 1.73 | -0.04 |
| 05 | 496 | 2.04 | 2.04 | -0.09 |
| 06 | 480 | 1.57 | 1.57 | -0.07 |
| 07 | 496 | 1.34 | 1.33 | -0.17 |
| 08 | 496 | 1.21 | 1.20 | -0.20 |
| 09 | 480 | 1.21 | 1.20 | -0.15 |
| 10 | 496 | 1.46 | 1.43 | -0.29 |
| 11 | 486 | 1.38 | 1.36 | -0.26 |
| 12 | 558 | 1.57 | 1.54 | -0.30 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.753 | -0.163 |
| 12:00 | 3040 | 1.500 | -0.237 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.04 |
| 02 | 1.82 |
| 03 | 1.76 |
| 04 | 1.73 |
| 05 | 2.04 |
| 06 | 1.57 |
| 07 | 1.33 |
| 08 | 1.20 |
| 09 | 1.20 |
| 10 | 1.43 |
| 11 | 1.36 |
| 12 | 1.54 |
| _fallback_ | _1.62_ |
