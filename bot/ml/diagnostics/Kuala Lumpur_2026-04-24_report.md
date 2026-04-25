# ML Distribution Model — Evaluation Report

**City:** Kuala Lumpur
**Generated:** 2026-04-25T00:39:07.332069+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.048 °C** |
| Mean bias | +0.061 °C |
| Fallback σ | 1.046 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.048** | — | — |
| persistence_yesterday | 1.654 | +0.606 | ML wins |
| yoy_same_date_last_year | 2.064 | +1.016 | ML wins |
| lag_7d_ago | 1.983 | +0.935 | ML wins |
| climatology_doy_mean | 1.545 | +0.497 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 70.0% | 68.3% | well-calibrated |
| ±2σ | 95.2% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.10 | 1.10 | -0.01 |
| 02 | 508 | 1.01 | 1.01 | +0.10 |
| 03 | 546 | 1.06 | 1.06 | +0.02 |
| 04 | 480 | 0.95 | 0.94 | -0.10 |
| 05 | 496 | 1.06 | 1.06 | +0.09 |
| 06 | 480 | 1.04 | 1.04 | +0.08 |
| 07 | 496 | 1.01 | 1.01 | +0.10 |
| 08 | 496 | 1.06 | 1.06 | +0.03 |
| 09 | 480 | 1.08 | 1.08 | +0.01 |
| 10 | 496 | 1.05 | 1.04 | +0.17 |
| 11 | 486 | 1.06 | 1.05 | +0.11 |
| 12 | 558 | 1.06 | 1.05 | +0.12 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.162 | +0.097 |
| 12:00 | 3040 | 0.920 | +0.025 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.10 |
| 02 | 1.01 |
| 03 | 1.06 |
| 04 | 0.94 |
| 05 | 1.06 |
| 06 | 1.04 |
| 07 | 1.01 |
| 08 | 1.06 |
| 09 | 1.08 |
| 10 | 1.04 |
| 11 | 1.05 |
| 12 | 1.05 |
| _fallback_ | _1.05_ |
