# ML Distribution Model — Evaluation Report

**City:** Seattle
**Generated:** 2026-04-25T00:41:02.975008+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.393 °C** |
| Mean bias | -0.002 °C |
| Fallback σ | 1.393 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.393** | — | — |
| persistence_yesterday | 2.822 | +1.429 | ML wins |
| yoy_same_date_last_year | 4.806 | +3.413 | ML wins |
| lag_7d_ago | 4.719 | +3.326 | ML wins |
| climatology_doy_mean | 3.601 | +2.208 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.9% | 68.3% | well-calibrated |
| ±2σ | 94.6% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.31 | 1.31 | +0.07 |
| 02 | 508 | 1.14 | 1.14 | +0.02 |
| 03 | 546 | 1.38 | 1.36 | +0.20 |
| 04 | 480 | 1.38 | 1.37 | -0.13 |
| 05 | 496 | 1.44 | 1.44 | +0.02 |
| 06 | 480 | 1.68 | 1.68 | +0.06 |
| 07 | 496 | 1.41 | 1.39 | +0.20 |
| 08 | 496 | 1.44 | 1.44 | -0.10 |
| 09 | 480 | 1.44 | 1.44 | -0.04 |
| 10 | 496 | 1.46 | 1.45 | -0.15 |
| 11 | 486 | 1.24 | 1.23 | -0.15 |
| 12 | 558 | 1.36 | 1.36 | -0.07 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.541 | +0.084 |
| 12:00 | 3040 | 1.226 | -0.088 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.31 |
| 02 | 1.14 |
| 03 | 1.36 |
| 04 | 1.37 |
| 05 | 1.44 |
| 06 | 1.68 |
| 07 | 1.39 |
| 08 | 1.44 |
| 09 | 1.44 |
| 10 | 1.45 |
| 11 | 1.23 |
| 12 | 1.36 |
| _fallback_ | _1.39_ |
