# ML Distribution Model — Evaluation Report

**City:** Panama City
**Generated:** 2026-04-25T00:40:34.621372+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7294  (unique dates: 3647, OOF predictions: 6070)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.912 °C** |
| Mean bias | +0.028 °C |
| Fallback σ | 0.912 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.912** | — | — |
| persistence_yesterday | 1.570 | +0.658 | ML wins |
| yoy_same_date_last_year | 1.994 | +1.082 | ML wins |
| lag_7d_ago | 1.861 | +0.948 | ML wins |
| climatology_doy_mean | 1.489 | +0.577 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.6% | 68.3% | well-calibrated |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 99.4% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 0.86 | 0.85 | +0.12 |
| 02 | 502 | 0.90 | 0.89 | +0.17 |
| 03 | 546 | 0.91 | 0.91 | -0.07 |
| 04 | 480 | 0.98 | 0.97 | -0.09 |
| 05 | 496 | 1.18 | 1.18 | +0.05 |
| 06 | 480 | 0.84 | 0.83 | -0.14 |
| 07 | 496 | 0.94 | 0.94 | +0.08 |
| 08 | 496 | 0.90 | 0.89 | -0.07 |
| 09 | 478 | 0.88 | 0.87 | +0.05 |
| 10 | 496 | 0.76 | 0.76 | +0.06 |
| 11 | 484 | 0.89 | 0.88 | +0.08 |
| 12 | 558 | 0.86 | 0.86 | +0.06 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3035 | 1.015 | +0.058 |
| 12:00 | 3035 | 0.797 | -0.002 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 0.85 |
| 02 | 0.89 |
| 03 | 0.91 |
| 04 | 0.97 |
| 05 | 1.18 |
| 06 | 0.83 |
| 07 | 0.94 |
| 08 | 0.89 |
| 09 | 0.87 |
| 10 | 0.76 |
| 11 | 0.88 |
| 12 | 0.86 |
| _fallback_ | _0.91_ |
