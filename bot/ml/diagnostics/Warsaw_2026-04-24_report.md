# ML Distribution Model — Evaluation Report

**City:** Warsaw
**Generated:** 2026-04-25T00:42:27.827800+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.334 °C** |
| Mean bias | -0.003 °C |
| Fallback σ | 1.334 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.334** | — | — |
| persistence_yesterday | 3.210 | +1.876 | ML wins |
| yoy_same_date_last_year | 6.276 | +4.942 | ML wins |
| lag_7d_ago | 5.799 | +4.464 | ML wins |
| climatology_doy_mean | 4.695 | +3.360 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.4% | 68.3% | well-calibrated |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.15 | 1.15 | +0.04 |
| 02 | 508 | 1.39 | 1.39 | +0.00 |
| 03 | 546 | 1.38 | 1.37 | -0.11 |
| 04 | 480 | 1.46 | 1.45 | +0.11 |
| 05 | 496 | 1.33 | 1.32 | -0.18 |
| 06 | 480 | 1.46 | 1.46 | +0.03 |
| 07 | 496 | 1.44 | 1.44 | +0.06 |
| 08 | 496 | 1.28 | 1.28 | -0.01 |
| 09 | 480 | 1.34 | 1.34 | -0.01 |
| 10 | 496 | 1.50 | 1.50 | +0.15 |
| 11 | 486 | 1.11 | 1.10 | -0.09 |
| 12 | 558 | 1.15 | 1.15 | +0.00 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.507 | +0.056 |
| 12:00 | 3040 | 1.136 | -0.062 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.15 |
| 02 | 1.39 |
| 03 | 1.37 |
| 04 | 1.45 |
| 05 | 1.32 |
| 06 | 1.46 |
| 07 | 1.44 |
| 08 | 1.28 |
| 09 | 1.34 |
| 10 | 1.50 |
| 11 | 1.10 |
| 12 | 1.15 |
| _fallback_ | _1.33_ |
