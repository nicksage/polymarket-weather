# ML Distribution Model — Evaluation Report

**City:** Buenos Aires
**Generated:** 2026-04-25T00:37:20.233907+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.301 °C** |
| Mean bias | -0.001 °C |
| Fallback σ | 1.301 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.301** | — | — |
| persistence_yesterday | 2.927 | +1.626 | ML wins |
| yoy_same_date_last_year | 4.385 | +3.083 | ML wins |
| lag_7d_ago | 4.426 | +3.125 | ML wins |
| climatology_doy_mean | 3.287 | +1.986 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.7% | 68.3% | well-calibrated |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.30 | 1.30 | -0.01 |
| 02 | 508 | 1.30 | 1.30 | +0.13 |
| 03 | 546 | 1.21 | 1.21 | +0.03 |
| 04 | 480 | 1.04 | 1.04 | -0.05 |
| 05 | 496 | 1.17 | 1.16 | +0.04 |
| 06 | 480 | 1.31 | 1.31 | -0.07 |
| 07 | 496 | 1.35 | 1.35 | +0.01 |
| 08 | 496 | 1.33 | 1.33 | -0.03 |
| 09 | 480 | 1.36 | 1.36 | -0.08 |
| 10 | 496 | 1.40 | 1.40 | +0.09 |
| 11 | 486 | 1.33 | 1.33 | +0.07 |
| 12 | 558 | 1.44 | 1.43 | -0.13 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.432 | +0.088 |
| 12:00 | 3040 | 1.156 | -0.090 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.30 |
| 02 | 1.30 |
| 03 | 1.21 |
| 04 | 1.04 |
| 05 | 1.16 |
| 06 | 1.31 |
| 07 | 1.35 |
| 08 | 1.33 |
| 09 | 1.36 |
| 10 | 1.40 |
| 11 | 1.33 |
| 12 | 1.43 |
| _fallback_ | _1.30_ |
