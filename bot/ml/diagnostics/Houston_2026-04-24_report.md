# ML Distribution Model — Evaluation Report

**City:** Houston
**Generated:** 2026-04-25T00:38:35.310113+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.626 °C** |
| Mean bias | -0.034 °C |
| Fallback σ | 1.626 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.626** | — | — |
| persistence_yesterday | 3.462 | +1.836 | ML wins |
| yoy_same_date_last_year | 5.945 | +4.319 | ML wins |
| lag_7d_ago | 5.713 | +4.087 | ML wins |
| climatology_doy_mean | 4.447 | +2.821 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 74.6% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.5% | 95.4% | well-calibrated |
| ±3σ | 98.7% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 2.20 | 2.19 | -0.16 |
| 02 | 508 | 2.15 | 2.15 | -0.19 |
| 03 | 546 | 1.77 | 1.77 | +0.00 |
| 04 | 480 | 1.59 | 1.59 | -0.07 |
| 05 | 496 | 1.37 | 1.36 | +0.03 |
| 06 | 480 | 1.25 | 1.25 | -0.01 |
| 07 | 496 | 1.29 | 1.29 | -0.07 |
| 08 | 496 | 1.44 | 1.43 | +0.18 |
| 09 | 480 | 1.28 | 1.28 | +0.02 |
| 10 | 496 | 1.45 | 1.45 | +0.08 |
| 11 | 486 | 1.48 | 1.47 | -0.21 |
| 12 | 558 | 1.73 | 1.73 | +0.01 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.756 | +0.041 |
| 12:00 | 3040 | 1.485 | -0.110 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 2.19 |
| 02 | 2.15 |
| 03 | 1.77 |
| 04 | 1.59 |
| 05 | 1.36 |
| 06 | 1.25 |
| 07 | 1.29 |
| 08 | 1.43 |
| 09 | 1.28 |
| 10 | 1.45 |
| 11 | 1.47 |
| 12 | 1.73 |
| _fallback_ | _1.63_ |
