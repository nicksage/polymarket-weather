# ML Distribution Model — Evaluation Report

**City:** Chicago
**Generated:** 2026-04-25T00:37:45.259741+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.808 °C** |
| Mean bias | +0.026 °C |
| Fallback σ | 1.808 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.808** | — | — |
| persistence_yesterday | 4.635 | +2.827 | ML wins |
| yoy_same_date_last_year | 7.766 | +5.958 | ML wins |
| lag_7d_ago | 7.656 | +5.848 | ML wins |
| climatology_doy_mean | 5.888 | +4.081 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.1% | 68.3% | under-confident (bands too wide) |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.05 | 2.05 | -0.15 |
| 02 | 508 | 1.92 | 1.92 | +0.10 |
| 03 | 546 | 2.18 | 2.18 | +0.06 |
| 04 | 480 | 2.32 | 2.31 | +0.14 |
| 05 | 496 | 1.95 | 1.95 | -0.03 |
| 06 | 480 | 1.69 | 1.69 | +0.01 |
| 07 | 496 | 1.40 | 1.40 | +0.06 |
| 08 | 496 | 1.28 | 1.28 | -0.01 |
| 09 | 480 | 1.53 | 1.53 | +0.04 |
| 10 | 496 | 1.82 | 1.81 | +0.18 |
| 11 | 486 | 1.40 | 1.39 | -0.07 |
| 12 | 558 | 1.76 | 1.76 | -0.00 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.965 | +0.100 |
| 12:00 | 3040 | 1.636 | -0.047 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.05 |
| 02 | 1.92 |
| 03 | 2.18 |
| 04 | 2.31 |
| 05 | 1.95 |
| 06 | 1.69 |
| 07 | 1.40 |
| 08 | 1.28 |
| 09 | 1.53 |
| 10 | 1.81 |
| 11 | 1.39 |
| 12 | 1.76 |
| _fallback_ | _1.81_ |
