# ML Distribution Model — Evaluation Report

**City:** Miami
**Generated:** 2026-04-25T00:40:00.668896+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.942 °C** |
| Mean bias | -0.071 °C |
| Fallback σ | 0.939 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.942** | — | — |
| persistence_yesterday | 1.943 | +1.001 | ML wins |
| yoy_same_date_last_year | 3.036 | +2.095 | ML wins |
| lag_7d_ago | 2.960 | +2.018 | ML wins |
| climatology_doy_mean | 2.265 | +1.323 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 75.4% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.3% | 95.4% | well-calibrated |
| ±3σ | 98.6% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.26 | 1.23 | -0.28 |
| 02 | 508 | 0.89 | 0.89 | -0.05 |
| 03 | 546 | 0.93 | 0.93 | -0.02 |
| 04 | 480 | 0.84 | 0.84 | -0.04 |
| 05 | 496 | 0.87 | 0.86 | -0.13 |
| 06 | 480 | 0.94 | 0.94 | -0.01 |
| 07 | 496 | 0.82 | 0.82 | +0.05 |
| 08 | 496 | 0.82 | 0.82 | -0.01 |
| 09 | 480 | 0.83 | 0.83 | -0.02 |
| 10 | 496 | 0.83 | 0.83 | -0.05 |
| 11 | 486 | 0.82 | 0.82 | -0.02 |
| 12 | 558 | 1.22 | 1.20 | -0.22 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.040 | -0.047 |
| 12:00 | 3040 | 0.833 | -0.094 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.23 |
| 02 | 0.89 |
| 03 | 0.93 |
| 04 | 0.84 |
| 05 | 0.86 |
| 06 | 0.94 |
| 07 | 0.82 |
| 08 | 0.82 |
| 09 | 0.83 |
| 10 | 0.83 |
| 11 | 0.82 |
| 12 | 1.20 |
| _fallback_ | _0.94_ |
