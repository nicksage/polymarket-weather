# ML Distribution Model — Evaluation Report

**City:** Beijing
**Generated:** 2026-04-25T00:37:13.326878+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.596 °C** |
| Mean bias | -0.155 °C |
| Fallback σ | 1.588 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.596** | — | — |
| persistence_yesterday | 3.562 | +1.967 | ML wins |
| yoy_same_date_last_year | 5.470 | +3.874 | ML wins |
| lag_7d_ago | 5.590 | +3.995 | ML wins |
| climatology_doy_mean | 4.128 | +2.532 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.6% | 68.3% | well-calibrated |
| ±2σ | 94.2% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.80 | 1.76 | -0.40 |
| 02 | 508 | 1.80 | 1.77 | -0.32 |
| 03 | 546 | 1.85 | 1.85 | -0.09 |
| 04 | 480 | 1.86 | 1.84 | -0.21 |
| 05 | 496 | 1.54 | 1.51 | -0.27 |
| 06 | 480 | 1.66 | 1.66 | +0.10 |
| 07 | 496 | 1.54 | 1.54 | +0.03 |
| 08 | 496 | 1.20 | 1.20 | -0.10 |
| 09 | 480 | 1.22 | 1.22 | -0.08 |
| 10 | 496 | 1.23 | 1.23 | +0.07 |
| 11 | 486 | 1.37 | 1.37 | +0.09 |
| 12 | 558 | 1.75 | 1.65 | -0.57 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.793 | -0.117 |
| 12:00 | 3040 | 1.370 | -0.192 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.76 |
| 02 | 1.77 |
| 03 | 1.85 |
| 04 | 1.84 |
| 05 | 1.51 |
| 06 | 1.66 |
| 07 | 1.54 |
| 08 | 1.20 |
| 09 | 1.22 |
| 10 | 1.23 |
| 11 | 1.37 |
| 12 | 1.65 |
| _fallback_ | _1.59_ |
