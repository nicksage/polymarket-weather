# ML Distribution Model — Evaluation Report

**City:** Shenzhen
**Generated:** 2026-04-25T00:41:31.350381+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.969 °C** |
| Mean bias | +0.047 °C |
| Fallback σ | 0.968 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.969** | — | — |
| persistence_yesterday | 2.147 | +1.177 | ML wins |
| yoy_same_date_last_year | 3.991 | +3.022 | ML wins |
| lag_7d_ago | 3.880 | +2.911 | ML wins |
| climatology_doy_mean | 2.917 | +1.948 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.6% | 68.3% | well-calibrated |
| ±2σ | 95.7% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.16 | 1.16 | -0.04 |
| 02 | 508 | 1.24 | 1.24 | +0.14 |
| 03 | 546 | 1.19 | 1.19 | +0.08 |
| 04 | 480 | 0.97 | 0.97 | +0.03 |
| 05 | 496 | 0.89 | 0.88 | +0.06 |
| 06 | 480 | 0.74 | 0.74 | +0.10 |
| 07 | 496 | 0.81 | 0.80 | +0.10 |
| 08 | 496 | 0.82 | 0.81 | +0.04 |
| 09 | 480 | 0.87 | 0.86 | +0.09 |
| 10 | 496 | 0.85 | 0.85 | +0.03 |
| 11 | 486 | 0.84 | 0.84 | -0.03 |
| 12 | 558 | 1.01 | 1.01 | -0.01 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.078 | +0.095 |
| 12:00 | 3040 | 0.846 | -0.001 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.16 |
| 02 | 1.24 |
| 03 | 1.19 |
| 04 | 0.97 |
| 05 | 0.88 |
| 06 | 0.74 |
| 07 | 0.80 |
| 08 | 0.81 |
| 09 | 0.86 |
| 10 | 0.85 |
| 11 | 0.84 |
| 12 | 1.01 |
| _fallback_ | _0.97_ |
