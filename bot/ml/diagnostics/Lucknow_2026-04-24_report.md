# ML Distribution Model — Evaluation Report

**City:** Lucknow
**Generated:** 2026-04-25T00:39:33.685652+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.261 °C** |
| Mean bias | -0.287 °C |
| Fallback σ | 1.228 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.261** | — | — |
| persistence_yesterday | 2.052 | +0.791 | ML wins |
| yoy_same_date_last_year | 3.968 | +2.706 | ML wins |
| lag_7d_ago | 3.537 | +2.276 | ML wins |
| climatology_doy_mean | 2.969 | +1.708 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.1% | 68.3% | well-calibrated |
| ±2σ | 94.3% | 95.4% | well-calibrated |
| ±3σ | 98.7% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.91 | 1.81 | -0.63 |
| 02 | 508 | 1.03 | 1.02 | -0.14 |
| 03 | 546 | 1.13 | 1.08 | -0.35 |
| 04 | 480 | 1.09 | 1.05 | -0.30 |
| 05 | 496 | 1.26 | 1.23 | -0.28 |
| 06 | 480 | 1.27 | 1.25 | -0.19 |
| 07 | 496 | 1.07 | 1.07 | +0.01 |
| 08 | 496 | 1.07 | 1.05 | -0.24 |
| 09 | 480 | 1.17 | 1.15 | -0.21 |
| 10 | 496 | 0.98 | 0.92 | -0.35 |
| 11 | 486 | 1.16 | 1.10 | -0.36 |
| 12 | 558 | 1.55 | 1.51 | -0.33 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.382 | -0.292 |
| 12:00 | 3040 | 1.128 | -0.281 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.81 |
| 02 | 1.02 |
| 03 | 1.08 |
| 04 | 1.05 |
| 05 | 1.23 |
| 06 | 1.25 |
| 07 | 1.07 |
| 08 | 1.05 |
| 09 | 1.15 |
| 10 | 0.92 |
| 11 | 1.10 |
| 12 | 1.51 |
| _fallback_ | _1.23_ |
