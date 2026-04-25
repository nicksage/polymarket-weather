# ML Distribution Model — Evaluation Report

**City:** Chengdu
**Generated:** 2026-04-25T00:37:38.322449+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.638 °C** |
| Mean bias | -0.006 °C |
| Fallback σ | 1.638 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.638** | — | — |
| persistence_yesterday | 3.041 | +1.403 | ML wins |
| yoy_same_date_last_year | 5.451 | +3.814 | ML wins |
| lag_7d_ago | 5.175 | +3.537 | ML wins |
| climatology_doy_mean | 4.066 | +2.428 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 69.9% | 68.3% | well-calibrated |
| ±2σ | 95.0% | 95.4% | well-calibrated |
| ±3σ | 99.4% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.64 | 1.63 | -0.21 |
| 02 | 508 | 1.59 | 1.59 | +0.11 |
| 03 | 546 | 1.71 | 1.67 | +0.39 |
| 04 | 480 | 1.76 | 1.75 | +0.11 |
| 05 | 496 | 1.87 | 1.84 | -0.34 |
| 06 | 480 | 1.78 | 1.76 | +0.24 |
| 07 | 496 | 1.72 | 1.72 | -0.06 |
| 08 | 496 | 1.78 | 1.77 | +0.18 |
| 09 | 480 | 1.41 | 1.41 | +0.06 |
| 10 | 496 | 1.31 | 1.31 | -0.08 |
| 11 | 486 | 1.47 | 1.47 | -0.12 |
| 12 | 558 | 1.52 | 1.48 | -0.32 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.850 | +0.050 |
| 12:00 | 3040 | 1.394 | -0.062 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.63 |
| 02 | 1.59 |
| 03 | 1.67 |
| 04 | 1.75 |
| 05 | 1.84 |
| 06 | 1.76 |
| 07 | 1.72 |
| 08 | 1.77 |
| 09 | 1.41 |
| 10 | 1.31 |
| 11 | 1.47 |
| 12 | 1.48 |
| _fallback_ | _1.64_ |
