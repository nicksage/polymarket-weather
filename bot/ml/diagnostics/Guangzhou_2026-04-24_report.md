# ML Distribution Model — Evaluation Report

**City:** Guangzhou
**Generated:** 2026-04-25T00:38:12.581874+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.255 °C** |
| Mean bias | -0.011 °C |
| Fallback σ | 1.255 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.255** | — | — |
| persistence_yesterday | 2.729 | +1.474 | ML wins |
| yoy_same_date_last_year | 5.275 | +4.021 | ML wins |
| lag_7d_ago | 5.077 | +3.822 | ML wins |
| climatology_doy_mean | 3.858 | +2.604 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.1% | 68.3% | well-calibrated |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 99.4% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.52 | 1.52 | -0.02 |
| 02 | 508 | 1.39 | 1.39 | -0.07 |
| 03 | 546 | 1.37 | 1.37 | +0.05 |
| 04 | 480 | 1.23 | 1.23 | -0.06 |
| 05 | 496 | 1.12 | 1.12 | -0.00 |
| 06 | 480 | 1.13 | 1.13 | +0.11 |
| 07 | 496 | 1.12 | 1.11 | +0.16 |
| 08 | 496 | 1.12 | 1.11 | -0.11 |
| 09 | 480 | 1.01 | 1.01 | -0.01 |
| 10 | 496 | 1.07 | 1.07 | -0.03 |
| 11 | 486 | 1.31 | 1.31 | +0.09 |
| 12 | 558 | 1.44 | 1.42 | -0.22 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.391 | +0.063 |
| 12:00 | 3040 | 1.102 | -0.086 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.52 |
| 02 | 1.39 |
| 03 | 1.37 |
| 04 | 1.23 |
| 05 | 1.12 |
| 06 | 1.13 |
| 07 | 1.11 |
| 08 | 1.11 |
| 09 | 1.01 |
| 10 | 1.07 |
| 11 | 1.31 |
| 12 | 1.42 |
| _fallback_ | _1.25_ |
