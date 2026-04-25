# ML Distribution Model — Evaluation Report

**City:** Moscow
**Generated:** 2026-04-25T00:40:14.381205+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.342 °C** |
| Mean bias | +0.050 °C |
| Fallback σ | 1.341 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.342** | — | — |
| persistence_yesterday | 3.082 | +1.740 | ML wins |
| yoy_same_date_last_year | 6.763 | +5.420 | ML wins |
| lag_7d_ago | 6.082 | +4.740 | ML wins |
| climatology_doy_mean | 5.067 | +3.724 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.3% | 68.3% | well-calibrated |
| ±2σ | 94.4% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.39 | 1.38 | -0.12 |
| 02 | 508 | 1.32 | 1.30 | -0.20 |
| 03 | 546 | 1.48 | 1.48 | +0.07 |
| 04 | 480 | 1.57 | 1.57 | +0.13 |
| 05 | 496 | 1.51 | 1.51 | +0.04 |
| 06 | 480 | 1.22 | 1.21 | +0.18 |
| 07 | 496 | 1.33 | 1.32 | +0.15 |
| 08 | 496 | 1.30 | 1.29 | +0.16 |
| 09 | 480 | 1.35 | 1.34 | +0.07 |
| 10 | 496 | 1.29 | 1.28 | +0.19 |
| 11 | 486 | 1.15 | 1.14 | +0.08 |
| 12 | 558 | 1.12 | 1.12 | -0.10 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.493 | +0.121 |
| 12:00 | 3040 | 1.173 | -0.021 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.38 |
| 02 | 1.30 |
| 03 | 1.48 |
| 04 | 1.57 |
| 05 | 1.51 |
| 06 | 1.21 |
| 07 | 1.32 |
| 08 | 1.29 |
| 09 | 1.34 |
| 10 | 1.28 |
| 11 | 1.14 |
| 12 | 1.12 |
| _fallback_ | _1.34_ |
