# ML Distribution Model — Evaluation Report

**City:** Jakarta
**Generated:** 2026-04-25T00:38:48.648707+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.827 °C** |
| Mean bias | +0.051 °C |
| Fallback σ | 0.825 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.827** | — | — |
| persistence_yesterday | 1.456 | +0.629 | ML wins |
| yoy_same_date_last_year | 1.887 | +1.060 | ML wins |
| lag_7d_ago | 1.798 | +0.971 | ML wins |
| climatology_doy_mean | 1.456 | +0.629 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.0% | 68.3% | well-calibrated |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.08 | 1.07 | -0.03 |
| 02 | 508 | 0.82 | 0.82 | -0.01 |
| 03 | 546 | 0.90 | 0.90 | +0.01 |
| 04 | 480 | 0.76 | 0.76 | +0.02 |
| 05 | 496 | 0.70 | 0.70 | +0.05 |
| 06 | 480 | 0.83 | 0.83 | -0.00 |
| 07 | 496 | 0.68 | 0.68 | +0.02 |
| 08 | 496 | 0.71 | 0.71 | +0.06 |
| 09 | 480 | 0.77 | 0.76 | +0.12 |
| 10 | 496 | 0.87 | 0.84 | +0.24 |
| 11 | 486 | 0.83 | 0.83 | +0.09 |
| 12 | 558 | 0.85 | 0.85 | +0.06 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 0.944 | +0.110 |
| 12:00 | 3040 | 0.691 | -0.008 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.07 |
| 02 | 0.82 |
| 03 | 0.90 |
| 04 | 0.76 |
| 05 | 0.70 |
| 06 | 0.83 |
| 07 | 0.68 |
| 08 | 0.71 |
| 09 | 0.76 |
| 10 | 0.84 |
| 11 | 0.83 |
| 12 | 0.85 |
| _fallback_ | _0.83_ |
