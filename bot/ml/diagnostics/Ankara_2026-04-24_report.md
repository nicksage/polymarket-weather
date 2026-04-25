# ML Distribution Model — Evaluation Report

**City:** Ankara
**Generated:** 2026-04-25T00:36:55.543754+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.504 °C** |
| Mean bias | -0.010 °C |
| Fallback σ | 1.504 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.504** | — | — |
| persistence_yesterday | 2.815 | +1.311 | ML wins |
| yoy_same_date_last_year | 5.888 | +4.384 | ML wins |
| lag_7d_ago | 5.605 | +4.101 | ML wins |
| climatology_doy_mean | 4.392 | +2.887 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.5% | 68.3% | well-calibrated |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 99.1% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.60 | 1.60 | +0.05 |
| 02 | 508 | 1.78 | 1.77 | -0.13 |
| 03 | 546 | 1.83 | 1.83 | -0.09 |
| 04 | 480 | 1.53 | 1.52 | +0.11 |
| 05 | 496 | 1.65 | 1.64 | +0.14 |
| 06 | 480 | 1.25 | 1.25 | -0.06 |
| 07 | 496 | 1.14 | 1.14 | +0.04 |
| 08 | 496 | 1.17 | 1.16 | +0.15 |
| 09 | 480 | 1.23 | 1.23 | -0.06 |
| 10 | 496 | 1.51 | 1.51 | -0.06 |
| 11 | 486 | 1.53 | 1.53 | -0.02 |
| 12 | 558 | 1.55 | 1.54 | -0.18 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.712 | +0.057 |
| 12:00 | 3040 | 1.263 | -0.076 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.60 |
| 02 | 1.77 |
| 03 | 1.83 |
| 04 | 1.52 |
| 05 | 1.64 |
| 06 | 1.25 |
| 07 | 1.14 |
| 08 | 1.16 |
| 09 | 1.23 |
| 10 | 1.51 |
| 11 | 1.53 |
| 12 | 1.54 |
| _fallback_ | _1.50_ |
