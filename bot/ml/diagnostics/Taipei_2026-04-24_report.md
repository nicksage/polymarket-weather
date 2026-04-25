# ML Distribution Model — Evaluation Report

**City:** Taipei
**Generated:** 2026-04-25T00:41:47.317050+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.112 °C** |
| Mean bias | +0.080 °C |
| Fallback σ | 1.109 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.112** | — | — |
| persistence_yesterday | 3.248 | +2.136 | ML wins |
| yoy_same_date_last_year | 5.209 | +4.097 | ML wins |
| lag_7d_ago | 5.282 | +4.170 | ML wins |
| climatology_doy_mean | 3.863 | +2.751 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.8% | 68.3% | under-confident (bands too wide) |
| ±2σ | 95.1% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.26 | 1.26 | +0.12 |
| 02 | 508 | 1.40 | 1.40 | +0.07 |
| 03 | 546 | 1.43 | 1.41 | +0.22 |
| 04 | 480 | 1.24 | 1.24 | +0.09 |
| 05 | 496 | 1.10 | 1.09 | +0.16 |
| 06 | 480 | 1.04 | 1.04 | +0.10 |
| 07 | 496 | 0.91 | 0.90 | +0.12 |
| 08 | 496 | 1.00 | 1.00 | +0.07 |
| 09 | 480 | 0.95 | 0.95 | +0.03 |
| 10 | 496 | 0.93 | 0.93 | +0.00 |
| 11 | 486 | 0.80 | 0.80 | -0.03 |
| 12 | 558 | 1.02 | 1.02 | -0.01 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.267 | +0.133 |
| 12:00 | 3040 | 0.932 | +0.027 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.26 |
| 02 | 1.40 |
| 03 | 1.41 |
| 04 | 1.24 |
| 05 | 1.09 |
| 06 | 1.04 |
| 07 | 0.90 |
| 08 | 1.00 |
| 09 | 0.95 |
| 10 | 0.93 |
| 11 | 0.80 |
| 12 | 1.02 |
| _fallback_ | _1.11_ |
