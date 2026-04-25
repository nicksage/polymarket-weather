# ML Distribution Model — Evaluation Report

**City:** Munich
**Generated:** 2026-04-25T00:40:21.361993+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.700 °C** |
| Mean bias | -0.099 °C |
| Fallback σ | 1.697 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.700** | — | — |
| persistence_yesterday | 3.677 | +1.977 | ML wins |
| yoy_same_date_last_year | 6.892 | +5.192 | ML wins |
| lag_7d_ago | 6.418 | +4.718 | ML wins |
| climatology_doy_mean | 5.065 | +3.365 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.1% | 68.3% | under-confident (bands too wide) |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.57 | 1.57 | -0.13 |
| 02 | 508 | 1.76 | 1.76 | -0.03 |
| 03 | 546 | 1.58 | 1.56 | -0.25 |
| 04 | 480 | 1.79 | 1.78 | +0.20 |
| 05 | 496 | 1.60 | 1.60 | -0.09 |
| 06 | 480 | 1.53 | 1.53 | -0.01 |
| 07 | 496 | 1.49 | 1.49 | -0.09 |
| 08 | 496 | 1.60 | 1.60 | +0.02 |
| 09 | 480 | 1.46 | 1.45 | -0.13 |
| 10 | 496 | 1.91 | 1.91 | +0.00 |
| 11 | 486 | 1.92 | 1.89 | -0.35 |
| 12 | 558 | 2.03 | 2.01 | -0.30 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.909 | -0.039 |
| 12:00 | 3040 | 1.461 | -0.158 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.57 |
| 02 | 1.76 |
| 03 | 1.56 |
| 04 | 1.78 |
| 05 | 1.60 |
| 06 | 1.53 |
| 07 | 1.49 |
| 08 | 1.60 |
| 09 | 1.45 |
| 10 | 1.91 |
| 11 | 1.89 |
| 12 | 2.01 |
| _fallback_ | _1.70_ |
