# ML Distribution Model — Evaluation Report

**City:** Amsterdam
**Generated:** 2026-04-25T00:36:49.303277+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7304  (unique dates: 3652, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.287 °C** |
| Mean bias | -0.020 °C |
| Fallback σ | 1.286 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.287** | — | — |
| persistence_yesterday | 2.579 | +1.292 | ML wins |
| yoy_same_date_last_year | 4.865 | +3.578 | ML wins |
| lag_7d_ago | 4.713 | +3.427 | ML wins |
| climatology_doy_mean | 3.657 | +2.370 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.1% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.8% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.26 | 1.26 | -0.07 |
| 02 | 508 | 1.26 | 1.26 | +0.00 |
| 03 | 546 | 1.26 | 1.26 | -0.09 |
| 04 | 480 | 1.36 | 1.35 | +0.17 |
| 05 | 494 | 1.32 | 1.30 | -0.22 |
| 06 | 480 | 1.41 | 1.41 | -0.07 |
| 07 | 496 | 1.49 | 1.49 | +0.14 |
| 08 | 496 | 1.31 | 1.31 | +0.10 |
| 09 | 480 | 1.31 | 1.31 | +0.09 |
| 10 | 496 | 1.25 | 1.25 | -0.07 |
| 11 | 488 | 1.04 | 1.04 | -0.09 |
| 12 | 558 | 1.14 | 1.13 | -0.11 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.435 | +0.017 |
| 12:00 | 3040 | 1.119 | -0.057 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.26 |
| 02 | 1.26 |
| 03 | 1.26 |
| 04 | 1.35 |
| 05 | 1.30 |
| 06 | 1.41 |
| 07 | 1.49 |
| 08 | 1.31 |
| 09 | 1.31 |
| 10 | 1.25 |
| 11 | 1.04 |
| 12 | 1.13 |
| _fallback_ | _1.29_ |
