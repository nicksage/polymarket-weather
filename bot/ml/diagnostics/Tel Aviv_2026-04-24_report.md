# ML Distribution Model — Evaluation Report

**City:** Tel Aviv
**Generated:** 2026-04-25T00:41:57.049718+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.151 °C** |
| Mean bias | +0.049 °C |
| Fallback σ | 1.150 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.151** | — | — |
| persistence_yesterday | 2.713 | +1.562 | ML wins |
| yoy_same_date_last_year | 4.231 | +3.081 | ML wins |
| lag_7d_ago | 4.195 | +3.044 | ML wins |
| climatology_doy_mean | 3.245 | +2.094 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 76.2% | 68.3% | under-confident (bands too wide) |
| ±2σ | 94.8% | 95.4% | well-calibrated |
| ±3σ | 98.6% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.08 | 1.08 | +0.04 |
| 02 | 508 | 1.18 | 1.18 | +0.01 |
| 03 | 546 | 1.40 | 1.40 | -0.10 |
| 04 | 480 | 1.47 | 1.47 | +0.05 |
| 05 | 496 | 1.81 | 1.77 | +0.38 |
| 06 | 480 | 0.86 | 0.86 | -0.00 |
| 07 | 496 | 0.92 | 0.89 | +0.25 |
| 08 | 496 | 0.66 | 0.66 | +0.09 |
| 09 | 480 | 0.75 | 0.74 | +0.12 |
| 10 | 496 | 1.01 | 1.01 | -0.01 |
| 11 | 486 | 1.10 | 1.09 | -0.15 |
| 12 | 558 | 1.02 | 1.02 | -0.06 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.316 | +0.182 |
| 12:00 | 3040 | 0.957 | -0.084 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.08 |
| 02 | 1.18 |
| 03 | 1.40 |
| 04 | 1.47 |
| 05 | 1.77 |
| 06 | 0.86 |
| 07 | 0.89 |
| 08 | 0.66 |
| 09 | 0.74 |
| 10 | 1.01 |
| 11 | 1.09 |
| 12 | 1.02 |
| _fallback_ | _1.15_ |
