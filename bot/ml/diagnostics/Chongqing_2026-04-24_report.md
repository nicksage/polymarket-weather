# ML Distribution Model — Evaluation Report

**City:** Chongqing
**Generated:** 2026-04-25T00:37:51.363124+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.513 °C** |
| Mean bias | -0.158 °C |
| Fallback σ | 1.504 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.513** | — | — |
| persistence_yesterday | 3.048 | +1.535 | ML wins |
| yoy_same_date_last_year | 5.921 | +4.409 | ML wins |
| lag_7d_ago | 5.685 | +4.172 | ML wins |
| climatology_doy_mean | 4.456 | +2.943 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 70.0% | 68.3% | well-calibrated |
| ±2σ | 94.8% | 95.4% | well-calibrated |
| ±3σ | 99.3% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.47 | 1.45 | -0.26 |
| 02 | 508 | 1.47 | 1.47 | -0.03 |
| 03 | 546 | 1.80 | 1.79 | -0.05 |
| 04 | 480 | 1.78 | 1.78 | +0.02 |
| 05 | 496 | 1.73 | 1.71 | -0.27 |
| 06 | 480 | 1.63 | 1.61 | -0.27 |
| 07 | 496 | 1.49 | 1.49 | -0.01 |
| 08 | 496 | 1.54 | 1.54 | -0.08 |
| 09 | 480 | 1.35 | 1.35 | +0.00 |
| 10 | 496 | 1.27 | 1.23 | -0.30 |
| 11 | 486 | 1.26 | 1.23 | -0.29 |
| 12 | 558 | 1.22 | 1.17 | -0.33 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.700 | -0.147 |
| 12:00 | 3040 | 1.298 | -0.168 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.45 |
| 02 | 1.47 |
| 03 | 1.79 |
| 04 | 1.78 |
| 05 | 1.71 |
| 06 | 1.61 |
| 07 | 1.49 |
| 08 | 1.54 |
| 09 | 1.35 |
| 10 | 1.23 |
| 11 | 1.23 |
| 12 | 1.17 |
| _fallback_ | _1.50_ |
