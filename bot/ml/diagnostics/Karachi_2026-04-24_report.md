# ML Distribution Model — Evaluation Report

**City:** Karachi
**Generated:** 2026-04-25T00:39:01.413890+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.103 °C** |
| Mean bias | -0.009 °C |
| Fallback σ | 1.103 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.103** | — | — |
| persistence_yesterday | 1.657 | +0.554 | ML wins |
| yoy_same_date_last_year | 3.079 | +1.977 | ML wins |
| lag_7d_ago | 2.840 | +1.737 | ML wins |
| climatology_doy_mean | 2.308 | +1.205 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.2% | 68.3% | well-calibrated |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 98.8% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.17 | 1.16 | +0.09 |
| 02 | 508 | 1.21 | 1.20 | -0.05 |
| 03 | 546 | 1.31 | 1.31 | -0.01 |
| 04 | 480 | 1.27 | 1.27 | +0.09 |
| 05 | 496 | 1.30 | 1.30 | +0.07 |
| 06 | 480 | 0.89 | 0.89 | +0.03 |
| 07 | 496 | 0.93 | 0.92 | +0.13 |
| 08 | 496 | 0.99 | 0.97 | -0.17 |
| 09 | 480 | 0.97 | 0.97 | -0.06 |
| 10 | 496 | 0.97 | 0.97 | -0.02 |
| 11 | 486 | 1.00 | 1.00 | -0.03 |
| 12 | 558 | 1.08 | 1.06 | -0.17 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.240 | +0.015 |
| 12:00 | 3040 | 0.945 | -0.033 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.16 |
| 02 | 1.20 |
| 03 | 1.31 |
| 04 | 1.27 |
| 05 | 1.30 |
| 06 | 0.89 |
| 07 | 0.92 |
| 08 | 0.97 |
| 09 | 0.97 |
| 10 | 0.97 |
| 11 | 1.00 |
| 12 | 1.06 |
| _fallback_ | _1.10_ |
