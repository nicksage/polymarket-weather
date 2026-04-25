# ML Distribution Model — Evaluation Report

**City:** Singapore
**Generated:** 2026-04-25T00:41:39.254225+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.966 °C** |
| Mean bias | -0.111 °C |
| Fallback σ | 0.960 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.966** | — | — |
| persistence_yesterday | 1.702 | +0.736 | ML wins |
| yoy_same_date_last_year | 2.212 | +1.246 | ML wins |
| lag_7d_ago | 2.054 | +1.088 | ML wins |
| climatology_doy_mean | 1.608 | +0.642 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.9% | 68.3% | well-calibrated |
| ±2σ | 94.3% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.14 | 1.12 | -0.20 |
| 02 | 508 | 0.93 | 0.93 | +0.05 |
| 03 | 546 | 0.94 | 0.94 | -0.02 |
| 04 | 480 | 1.00 | 0.98 | -0.15 |
| 05 | 496 | 0.90 | 0.90 | -0.03 |
| 06 | 480 | 0.98 | 0.97 | -0.13 |
| 07 | 496 | 0.83 | 0.83 | -0.07 |
| 08 | 496 | 0.93 | 0.93 | -0.11 |
| 09 | 480 | 0.94 | 0.92 | -0.20 |
| 10 | 496 | 0.95 | 0.94 | -0.14 |
| 11 | 486 | 1.02 | 1.02 | -0.12 |
| 12 | 558 | 0.98 | 0.96 | -0.20 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.090 | -0.108 |
| 12:00 | 3040 | 0.823 | -0.113 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.12 |
| 02 | 0.93 |
| 03 | 0.94 |
| 04 | 0.98 |
| 05 | 0.90 |
| 06 | 0.97 |
| 07 | 0.83 |
| 08 | 0.93 |
| 09 | 0.92 |
| 10 | 0.94 |
| 11 | 1.02 |
| 12 | 0.96 |
| _fallback_ | _0.96_ |
