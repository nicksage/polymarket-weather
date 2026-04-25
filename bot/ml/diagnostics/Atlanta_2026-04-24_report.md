# ML Distribution Model — Evaluation Report

**City:** Atlanta
**Generated:** 2026-04-25T00:37:01.628047+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.659 °C** |
| Mean bias | -0.052 °C |
| Fallback σ | 1.658 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.659** | — | — |
| persistence_yesterday | 3.461 | +1.803 | ML wins |
| yoy_same_date_last_year | 5.734 | +4.075 | ML wins |
| lag_7d_ago | 5.619 | +3.960 | ML wins |
| climatology_doy_mean | 4.358 | +2.699 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.9% | 68.3% | well-calibrated |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.11 | 2.06 | -0.46 |
| 02 | 508 | 2.00 | 2.00 | +0.05 |
| 03 | 546 | 2.20 | 2.19 | +0.20 |
| 04 | 480 | 1.77 | 1.76 | -0.09 |
| 05 | 496 | 1.41 | 1.41 | +0.05 |
| 06 | 480 | 1.33 | 1.33 | +0.04 |
| 07 | 496 | 1.08 | 1.08 | -0.03 |
| 08 | 496 | 1.25 | 1.25 | +0.02 |
| 09 | 480 | 1.29 | 1.29 | -0.04 |
| 10 | 496 | 1.52 | 1.51 | -0.10 |
| 11 | 486 | 1.45 | 1.44 | -0.11 |
| 12 | 558 | 1.86 | 1.85 | -0.12 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.831 | +0.007 |
| 12:00 | 3040 | 1.467 | -0.111 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.06 |
| 02 | 2.00 |
| 03 | 2.19 |
| 04 | 1.76 |
| 05 | 1.41 |
| 06 | 1.33 |
| 07 | 1.08 |
| 08 | 1.25 |
| 09 | 1.29 |
| 10 | 1.51 |
| 11 | 1.44 |
| 12 | 1.85 |
| _fallback_ | _1.66_ |
