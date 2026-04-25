# ML Distribution Model — Evaluation Report

**City:** Tokyo
**Generated:** 2026-04-25T00:42:07.274114+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.079 °C** |
| Mean bias | +0.014 °C |
| Fallback σ | 1.079 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.079** | — | — |
| persistence_yesterday | 3.246 | +2.167 | ML wins |
| yoy_same_date_last_year | 4.562 | +3.482 | ML wins |
| lag_7d_ago | 4.602 | +3.523 | ML wins |
| climatology_doy_mean | 3.452 | +2.373 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.2% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.5% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.04 | 1.03 | -0.07 |
| 02 | 508 | 1.33 | 1.33 | -0.05 |
| 03 | 546 | 1.41 | 1.40 | +0.11 |
| 04 | 480 | 1.18 | 1.18 | -0.05 |
| 05 | 496 | 1.06 | 1.06 | -0.03 |
| 06 | 480 | 1.07 | 1.07 | -0.04 |
| 07 | 496 | 1.02 | 1.02 | +0.05 |
| 08 | 496 | 0.86 | 0.85 | +0.13 |
| 09 | 480 | 0.93 | 0.93 | +0.02 |
| 10 | 496 | 0.85 | 0.85 | +0.10 |
| 11 | 486 | 0.86 | 0.86 | +0.00 |
| 12 | 558 | 1.14 | 1.14 | -0.01 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.226 | +0.091 |
| 12:00 | 3040 | 0.910 | -0.064 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.03 |
| 02 | 1.33 |
| 03 | 1.40 |
| 04 | 1.18 |
| 05 | 1.06 |
| 06 | 1.07 |
| 07 | 1.02 |
| 08 | 0.85 |
| 09 | 0.93 |
| 10 | 0.85 |
| 11 | 0.86 |
| 12 | 1.14 |
| _fallback_ | _1.08_ |
