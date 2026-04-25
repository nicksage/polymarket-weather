# ML Distribution Model — Evaluation Report

**City:** Jeddah
**Generated:** 2026-04-25T00:38:55.267087+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.183 °C** |
| Mean bias | +0.055 °C |
| Fallback σ | 1.181 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.183** | — | — |
| persistence_yesterday | 2.119 | +0.936 | ML wins |
| yoy_same_date_last_year | 3.216 | +2.034 | ML wins |
| lag_7d_ago | 3.178 | +1.995 | ML wins |
| climatology_doy_mean | 2.449 | +1.266 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 75.6% | 68.3% | under-confident (bands too wide) |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 98.7% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.12 | 1.12 | +0.01 |
| 02 | 508 | 1.08 | 1.08 | +0.07 |
| 03 | 546 | 1.19 | 1.19 | +0.04 |
| 04 | 480 | 1.21 | 1.21 | -0.01 |
| 05 | 496 | 1.40 | 1.40 | +0.10 |
| 06 | 480 | 1.39 | 1.39 | +0.04 |
| 07 | 496 | 1.26 | 1.26 | +0.03 |
| 08 | 496 | 1.04 | 1.03 | +0.11 |
| 09 | 480 | 1.38 | 1.38 | -0.00 |
| 10 | 496 | 1.23 | 1.22 | +0.16 |
| 11 | 486 | 0.84 | 0.84 | +0.02 |
| 12 | 558 | 0.96 | 0.95 | +0.09 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.386 | +0.110 |
| 12:00 | 3040 | 0.937 | +0.001 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.12 |
| 02 | 1.08 |
| 03 | 1.19 |
| 04 | 1.21 |
| 05 | 1.40 |
| 06 | 1.39 |
| 07 | 1.26 |
| 08 | 1.03 |
| 09 | 1.38 |
| 10 | 1.22 |
| 11 | 0.84 |
| 12 | 0.95 |
| _fallback_ | _1.18_ |
