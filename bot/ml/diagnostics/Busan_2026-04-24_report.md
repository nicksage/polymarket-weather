# ML Distribution Model — Evaluation Report

**City:** Busan
**Generated:** 2026-04-25T00:37:26.694487+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.254 °C** |
| Mean bias | -0.111 °C |
| Fallback σ | 1.249 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.254** | — | — |
| persistence_yesterday | 2.690 | +1.435 | ML wins |
| yoy_same_date_last_year | 4.332 | +3.078 | ML wins |
| lag_7d_ago | 4.357 | +3.103 | ML wins |
| climatology_doy_mean | 3.273 | +2.019 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 70.5% | 68.3% | well-calibrated |
| ±2σ | 94.9% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.61 | 1.60 | -0.14 |
| 02 | 508 | 1.42 | 1.42 | -0.10 |
| 03 | 546 | 1.33 | 1.33 | -0.06 |
| 04 | 480 | 1.21 | 1.19 | -0.21 |
| 05 | 496 | 1.27 | 1.23 | -0.32 |
| 06 | 480 | 1.10 | 1.09 | -0.13 |
| 07 | 496 | 1.11 | 1.10 | -0.16 |
| 08 | 496 | 0.99 | 0.98 | -0.16 |
| 09 | 480 | 1.13 | 1.13 | -0.00 |
| 10 | 496 | 1.02 | 1.02 | +0.07 |
| 11 | 486 | 1.23 | 1.22 | +0.11 |
| 12 | 558 | 1.40 | 1.38 | -0.22 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.412 | -0.027 |
| 12:00 | 3040 | 1.074 | -0.196 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.60 |
| 02 | 1.42 |
| 03 | 1.33 |
| 04 | 1.19 |
| 05 | 1.23 |
| 06 | 1.09 |
| 07 | 1.10 |
| 08 | 0.98 |
| 09 | 1.13 |
| 10 | 1.02 |
| 11 | 1.22 |
| 12 | 1.38 |
| _fallback_ | _1.25_ |
