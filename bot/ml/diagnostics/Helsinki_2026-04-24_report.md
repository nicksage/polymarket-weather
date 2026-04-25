# ML Distribution Model — Evaluation Report

**City:** Helsinki
**Generated:** 2026-04-25T00:38:21.496388+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.343 °C** |
| Mean bias | +0.024 °C |
| Fallback σ | 1.343 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.343** | — | — |
| persistence_yesterday | 2.512 | +1.169 | ML wins |
| yoy_same_date_last_year | 5.313 | +3.970 | ML wins |
| lag_7d_ago | 4.763 | +3.420 | ML wins |
| climatology_doy_mean | 3.985 | +2.643 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.6% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 98.6% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.54 | 1.52 | -0.23 |
| 02 | 508 | 1.44 | 1.43 | -0.16 |
| 03 | 546 | 1.22 | 1.22 | -0.02 |
| 04 | 480 | 1.55 | 1.55 | +0.13 |
| 05 | 496 | 1.46 | 1.46 | +0.04 |
| 06 | 480 | 1.33 | 1.31 | +0.19 |
| 07 | 496 | 1.66 | 1.63 | +0.34 |
| 08 | 496 | 1.31 | 1.30 | +0.14 |
| 09 | 480 | 1.03 | 1.03 | -0.03 |
| 10 | 496 | 1.08 | 1.08 | +0.04 |
| 11 | 486 | 1.04 | 1.04 | -0.09 |
| 12 | 558 | 1.26 | 1.26 | -0.01 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.442 | +0.092 |
| 12:00 | 3040 | 1.235 | -0.045 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.52 |
| 02 | 1.43 |
| 03 | 1.22 |
| 04 | 1.55 |
| 05 | 1.46 |
| 06 | 1.31 |
| 07 | 1.63 |
| 08 | 1.30 |
| 09 | 1.03 |
| 10 | 1.08 |
| 11 | 1.04 |
| 12 | 1.26 |
| _fallback_ | _1.34_ |
