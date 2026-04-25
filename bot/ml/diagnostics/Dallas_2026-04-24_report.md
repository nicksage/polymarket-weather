# ML Distribution Model — Evaluation Report

**City:** Dallas
**Generated:** 2026-04-25T00:37:58.138778+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.931 °C** |
| Mean bias | +0.124 °C |
| Fallback σ | 1.927 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.931** | — | — |
| persistence_yesterday | 4.126 | +2.195 | ML wins |
| yoy_same_date_last_year | 7.043 | +5.113 | ML wins |
| lag_7d_ago | 6.767 | +4.837 | ML wins |
| climatology_doy_mean | 5.330 | +3.400 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.7% | 68.3% | well-calibrated |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.33 | 2.33 | +0.11 |
| 02 | 508 | 2.75 | 2.75 | +0.10 |
| 03 | 546 | 1.96 | 1.96 | +0.17 |
| 04 | 480 | 2.23 | 2.23 | -0.06 |
| 05 | 496 | 1.69 | 1.69 | +0.01 |
| 06 | 480 | 1.49 | 1.49 | +0.10 |
| 07 | 496 | 1.57 | 1.54 | +0.30 |
| 08 | 496 | 1.37 | 1.31 | +0.40 |
| 09 | 480 | 1.31 | 1.30 | +0.22 |
| 10 | 496 | 1.81 | 1.81 | +0.09 |
| 11 | 486 | 1.82 | 1.82 | -0.05 |
| 12 | 558 | 2.15 | 2.15 | +0.09 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 2.109 | +0.231 |
| 12:00 | 3040 | 1.734 | +0.017 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.33 |
| 02 | 2.75 |
| 03 | 1.96 |
| 04 | 2.23 |
| 05 | 1.69 |
| 06 | 1.49 |
| 07 | 1.54 |
| 08 | 1.31 |
| 09 | 1.30 |
| 10 | 1.81 |
| 11 | 1.82 |
| 12 | 2.15 |
| _fallback_ | _1.93_ |
