# ML Distribution Model — Evaluation Report

**City:** Istanbul
**Generated:** 2026-04-25T00:38:42.335164+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.058 °C** |
| Mean bias | +0.069 °C |
| Fallback σ | 1.056 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.058** | — | — |
| persistence_yesterday | 2.489 | +1.431 | ML wins |
| yoy_same_date_last_year | 4.858 | +3.800 | ML wins |
| lag_7d_ago | 4.567 | +3.509 | ML wins |
| climatology_doy_mean | 3.643 | +2.585 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.6% | 68.3% | under-confident (bands too wide) |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 98.8% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.07 | 1.07 | -0.04 |
| 02 | 508 | 1.03 | 1.03 | +0.04 |
| 03 | 546 | 1.26 | 1.25 | +0.04 |
| 04 | 480 | 1.30 | 1.28 | +0.22 |
| 05 | 496 | 1.25 | 1.25 | +0.01 |
| 06 | 480 | 0.98 | 0.96 | +0.17 |
| 07 | 496 | 1.06 | 1.04 | +0.19 |
| 08 | 496 | 0.86 | 0.85 | +0.13 |
| 09 | 480 | 0.81 | 0.81 | +0.05 |
| 10 | 496 | 0.90 | 0.89 | -0.03 |
| 11 | 486 | 1.02 | 1.02 | +0.03 |
| 12 | 558 | 1.03 | 1.03 | +0.03 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.160 | +0.106 |
| 12:00 | 3040 | 0.945 | +0.031 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.07 |
| 02 | 1.03 |
| 03 | 1.25 |
| 04 | 1.28 |
| 05 | 1.25 |
| 06 | 0.96 |
| 07 | 1.04 |
| 08 | 0.85 |
| 09 | 0.81 |
| 10 | 0.89 |
| 11 | 1.02 |
| 12 | 1.03 |
| _fallback_ | _1.06_ |
