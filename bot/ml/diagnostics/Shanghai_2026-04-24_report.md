# ML Distribution Model — Evaluation Report

**City:** Shanghai
**Generated:** 2026-04-25T00:41:22.107153+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.036 °C** |
| Mean bias | +0.032 °C |
| Fallback σ | 1.036 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.036** | — | — |
| persistence_yesterday | 3.211 | +2.174 | ML wins |
| yoy_same_date_last_year | 5.365 | +4.329 | ML wins |
| lag_7d_ago | 5.428 | +4.391 | ML wins |
| climatology_doy_mean | 4.048 | +3.012 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.7% | 68.3% | well-calibrated |
| ±2σ | 95.0% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.25 | 1.24 | -0.09 |
| 02 | 508 | 1.20 | 1.19 | -0.15 |
| 03 | 546 | 1.31 | 1.29 | +0.24 |
| 04 | 480 | 1.09 | 1.09 | +0.12 |
| 05 | 496 | 1.13 | 1.12 | +0.13 |
| 06 | 480 | 1.12 | 1.10 | +0.18 |
| 07 | 496 | 0.91 | 0.91 | -0.01 |
| 08 | 496 | 0.83 | 0.83 | +0.05 |
| 09 | 480 | 0.75 | 0.75 | +0.00 |
| 10 | 496 | 0.74 | 0.74 | +0.04 |
| 11 | 486 | 0.85 | 0.85 | -0.01 |
| 12 | 558 | 0.98 | 0.97 | -0.11 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.185 | +0.098 |
| 12:00 | 3040 | 0.863 | -0.034 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.24 |
| 02 | 1.19 |
| 03 | 1.29 |
| 04 | 1.09 |
| 05 | 1.12 |
| 06 | 1.10 |
| 07 | 0.91 |
| 08 | 0.83 |
| 09 | 0.75 |
| 10 | 0.74 |
| 11 | 0.85 |
| 12 | 0.97 |
| _fallback_ | _1.04_ |
