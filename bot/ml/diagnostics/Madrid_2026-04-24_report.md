# ML Distribution Model — Evaluation Report

**City:** Madrid
**Generated:** 2026-04-25T00:39:40.735264+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7304  (unique dates: 3652, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.504 °C** |
| Mean bias | +0.046 °C |
| Fallback σ | 1.503 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.504** | — | — |
| persistence_yesterday | 2.557 | +1.053 | ML wins |
| yoy_same_date_last_year | 5.277 | +3.773 | ML wins |
| lag_7d_ago | 4.805 | +3.301 | ML wins |
| climatology_doy_mean | 3.805 | +2.301 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.2% | 68.3% | well-calibrated |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.63 | 1.62 | -0.21 |
| 02 | 508 | 1.36 | 1.36 | +0.11 |
| 03 | 546 | 1.61 | 1.60 | -0.12 |
| 04 | 480 | 1.63 | 1.62 | +0.14 |
| 05 | 494 | 1.62 | 1.61 | +0.21 |
| 06 | 480 | 1.54 | 1.53 | +0.14 |
| 07 | 496 | 1.34 | 1.31 | +0.24 |
| 08 | 496 | 1.45 | 1.44 | +0.18 |
| 09 | 480 | 1.46 | 1.46 | -0.06 |
| 10 | 496 | 1.63 | 1.63 | +0.04 |
| 11 | 488 | 1.33 | 1.33 | -0.05 |
| 12 | 558 | 1.38 | 1.38 | +0.00 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.651 | +0.088 |
| 12:00 | 3040 | 1.341 | +0.004 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.62 |
| 02 | 1.36 |
| 03 | 1.60 |
| 04 | 1.62 |
| 05 | 1.61 |
| 06 | 1.53 |
| 07 | 1.31 |
| 08 | 1.44 |
| 09 | 1.46 |
| 10 | 1.63 |
| 11 | 1.33 |
| 12 | 1.38 |
| _fallback_ | _1.50_ |
