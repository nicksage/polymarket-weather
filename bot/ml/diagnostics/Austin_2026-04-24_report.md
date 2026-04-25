# ML Distribution Model — Evaluation Report

**City:** Austin
**Generated:** 2026-04-25T00:37:07.646157+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.919 °C** |
| Mean bias | -0.022 °C |
| Fallback σ | 1.919 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.919** | — | — |
| persistence_yesterday | 4.035 | +2.116 | ML wins |
| yoy_same_date_last_year | 6.765 | +4.845 | ML wins |
| lag_7d_ago | 6.564 | +4.645 | ML wins |
| climatology_doy_mean | 5.159 | +3.240 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.2% | 68.3% | well-calibrated |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 98.8% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.23 | 2.23 | -0.14 |
| 02 | 508 | 2.51 | 2.49 | -0.32 |
| 03 | 546 | 2.13 | 2.13 | +0.04 |
| 04 | 480 | 2.23 | 2.23 | +0.04 |
| 05 | 496 | 1.77 | 1.77 | -0.17 |
| 06 | 480 | 1.42 | 1.42 | +0.06 |
| 07 | 496 | 1.42 | 1.42 | -0.14 |
| 08 | 496 | 1.30 | 1.30 | +0.14 |
| 09 | 480 | 1.44 | 1.43 | +0.07 |
| 10 | 496 | 2.18 | 2.17 | +0.20 |
| 11 | 486 | 1.84 | 1.84 | -0.04 |
| 12 | 558 | 1.97 | 1.97 | +0.02 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 2.150 | +0.098 |
| 12:00 | 3040 | 1.657 | -0.142 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.23 |
| 02 | 2.49 |
| 03 | 2.13 |
| 04 | 2.23 |
| 05 | 1.77 |
| 06 | 1.42 |
| 07 | 1.42 |
| 08 | 1.30 |
| 09 | 1.43 |
| 10 | 2.17 |
| 11 | 1.84 |
| 12 | 1.97 |
| _fallback_ | _1.92_ |
