# ML Distribution Model — Evaluation Report

**City:** Sao Paulo
**Generated:** 2026-04-25T00:40:55.027267+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.325 °C** |
| Mean bias | -0.077 °C |
| Fallback σ | 1.323 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.325** | — | — |
| persistence_yesterday | 3.404 | +2.079 | ML wins |
| yoy_same_date_last_year | 5.421 | +4.096 | ML wins |
| lag_7d_ago | 5.501 | +4.176 | ML wins |
| climatology_doy_mean | 4.058 | +2.733 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.1% | 68.3% | well-calibrated |
| ±2σ | 94.8% | 95.4% | well-calibrated |
| ±3σ | 99.0% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.25 | 1.25 | -0.07 |
| 02 | 508 | 1.14 | 1.14 | -0.07 |
| 03 | 546 | 1.08 | 1.08 | -0.03 |
| 04 | 480 | 1.20 | 1.19 | -0.11 |
| 05 | 496 | 1.20 | 1.20 | -0.06 |
| 06 | 480 | 1.21 | 1.20 | -0.17 |
| 07 | 496 | 1.43 | 1.43 | -0.02 |
| 08 | 496 | 1.44 | 1.44 | -0.08 |
| 09 | 480 | 1.67 | 1.65 | +0.21 |
| 10 | 496 | 1.54 | 1.54 | -0.15 |
| 11 | 486 | 1.31 | 1.29 | -0.21 |
| 12 | 558 | 1.34 | 1.33 | -0.16 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.541 | -0.019 |
| 12:00 | 3040 | 1.065 | -0.136 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.25 |
| 02 | 1.14 |
| 03 | 1.08 |
| 04 | 1.19 |
| 05 | 1.20 |
| 06 | 1.20 |
| 07 | 1.43 |
| 08 | 1.44 |
| 09 | 1.65 |
| 10 | 1.54 |
| 11 | 1.29 |
| 12 | 1.33 |
| _fallback_ | _1.32_ |
