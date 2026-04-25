# ML Distribution Model — Evaluation Report

**City:** Nyc
**Generated:** 2026-04-25T00:40:28.085413+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.598 °C** |
| Mean bias | -0.062 °C |
| Fallback σ | 1.597 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.598** | — | — |
| persistence_yesterday | 4.207 | +2.609 | ML wins |
| yoy_same_date_last_year | 6.352 | +4.754 | ML wins |
| lag_7d_ago | 6.348 | +4.750 | ML wins |
| climatology_doy_mean | 4.765 | +3.167 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.8% | 68.3% | well-calibrated |
| ±2σ | 94.8% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.97 | 1.94 | -0.32 |
| 02 | 508 | 1.84 | 1.83 | -0.14 |
| 03 | 546 | 2.01 | 2.01 | -0.12 |
| 04 | 480 | 2.05 | 2.05 | -0.03 |
| 05 | 496 | 1.66 | 1.65 | +0.16 |
| 06 | 480 | 1.31 | 1.31 | +0.07 |
| 07 | 496 | 1.28 | 1.27 | +0.12 |
| 08 | 496 | 1.19 | 1.19 | +0.10 |
| 09 | 480 | 1.26 | 1.26 | -0.04 |
| 10 | 496 | 1.35 | 1.34 | -0.13 |
| 11 | 486 | 1.35 | 1.33 | -0.20 |
| 12 | 558 | 1.45 | 1.44 | -0.15 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.772 | -0.027 |
| 12:00 | 3040 | 1.403 | -0.096 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.94 |
| 02 | 1.83 |
| 03 | 2.01 |
| 04 | 2.05 |
| 05 | 1.65 |
| 06 | 1.31 |
| 07 | 1.27 |
| 08 | 1.19 |
| 09 | 1.26 |
| 10 | 1.34 |
| 11 | 1.33 |
| 12 | 1.44 |
| _fallback_ | _1.60_ |
