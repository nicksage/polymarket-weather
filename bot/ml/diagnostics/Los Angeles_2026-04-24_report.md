# ML Distribution Model — Evaluation Report

**City:** Los Angeles
**Generated:** 2026-04-25T00:39:27.437990+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.065 °C** |
| Mean bias | +0.002 °C |
| Fallback σ | 1.065 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.065** | — | — |
| persistence_yesterday | 2.483 | +1.417 | ML wins |
| yoy_same_date_last_year | 4.966 | +3.901 | ML wins |
| lag_7d_ago | 4.655 | +3.590 | ML wins |
| climatology_doy_mean | 3.727 | +2.662 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.3% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.7% | 95.4% | well-calibrated |
| ±3σ | 98.8% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.07 | 1.06 | +0.09 |
| 02 | 508 | 1.06 | 1.05 | -0.10 |
| 03 | 546 | 1.15 | 1.14 | -0.18 |
| 04 | 480 | 1.11 | 1.11 | -0.04 |
| 05 | 496 | 0.96 | 0.96 | -0.05 |
| 06 | 480 | 0.94 | 0.94 | -0.02 |
| 07 | 496 | 1.05 | 1.05 | +0.03 |
| 08 | 496 | 0.84 | 0.83 | +0.12 |
| 09 | 480 | 1.19 | 1.18 | +0.10 |
| 10 | 496 | 1.08 | 1.08 | +0.14 |
| 11 | 486 | 1.04 | 1.04 | -0.07 |
| 12 | 558 | 1.22 | 1.22 | +0.02 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.250 | +0.012 |
| 12:00 | 3040 | 0.841 | -0.007 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.06 |
| 02 | 1.05 |
| 03 | 1.14 |
| 04 | 1.11 |
| 05 | 0.96 |
| 06 | 0.94 |
| 07 | 1.05 |
| 08 | 0.83 |
| 09 | 1.18 |
| 10 | 1.08 |
| 11 | 1.04 |
| 12 | 1.22 |
| _fallback_ | _1.07_ |
