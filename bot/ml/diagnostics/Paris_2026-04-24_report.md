# ML Distribution Model — Evaluation Report

**City:** Paris
**Generated:** 2026-04-25T00:40:41.931696+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.487 °C** |
| Mean bias | -0.000 °C |
| Fallback σ | 1.487 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.487** | — | — |
| persistence_yesterday | 2.816 | +1.329 | ML wins |
| yoy_same_date_last_year | 5.584 | +4.097 | ML wins |
| lag_7d_ago | 5.285 | +3.798 | ML wins |
| climatology_doy_mean | 4.114 | +2.628 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 70.9% | 68.3% | well-calibrated |
| ±2σ | 95.3% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.22 | 1.22 | -0.01 |
| 02 | 508 | 1.43 | 1.43 | -0.02 |
| 03 | 546 | 1.43 | 1.41 | -0.24 |
| 04 | 480 | 1.55 | 1.53 | +0.23 |
| 05 | 496 | 1.61 | 1.60 | +0.14 |
| 06 | 480 | 1.69 | 1.69 | +0.08 |
| 07 | 496 | 1.74 | 1.73 | +0.02 |
| 08 | 496 | 1.65 | 1.65 | -0.07 |
| 09 | 480 | 1.59 | 1.58 | +0.06 |
| 10 | 496 | 1.53 | 1.52 | +0.13 |
| 11 | 486 | 1.15 | 1.14 | -0.16 |
| 12 | 558 | 1.18 | 1.17 | -0.13 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.646 | +0.111 |
| 12:00 | 3040 | 1.308 | -0.111 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.22 |
| 02 | 1.43 |
| 03 | 1.41 |
| 04 | 1.53 |
| 05 | 1.60 |
| 06 | 1.69 |
| 07 | 1.73 |
| 08 | 1.65 |
| 09 | 1.58 |
| 10 | 1.52 |
| 11 | 1.14 |
| 12 | 1.17 |
| _fallback_ | _1.49_ |
