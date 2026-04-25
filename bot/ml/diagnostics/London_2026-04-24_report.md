# ML Distribution Model — Evaluation Report

**City:** London
**Generated:** 2026-04-25T00:39:20.876504+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.346 °C** |
| Mean bias | -0.125 °C |
| Fallback σ | 1.340 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.346** | — | — |
| persistence_yesterday | 2.570 | +1.224 | ML wins |
| yoy_same_date_last_year | 4.821 | +3.476 | ML wins |
| lag_7d_ago | 4.572 | +3.226 | ML wins |
| climatology_doy_mean | 3.552 | +2.206 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 72.4% | 68.3% | well-calibrated |
| ±2σ | 95.0% | 95.4% | well-calibrated |
| ±3σ | 98.9% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.23 | 1.22 | -0.13 |
| 02 | 508 | 1.34 | 1.33 | -0.15 |
| 03 | 546 | 1.43 | 1.40 | -0.26 |
| 04 | 480 | 1.38 | 1.38 | +0.04 |
| 05 | 496 | 1.43 | 1.41 | -0.27 |
| 06 | 480 | 1.51 | 1.51 | -0.04 |
| 07 | 496 | 1.59 | 1.59 | -0.03 |
| 08 | 496 | 1.51 | 1.51 | -0.05 |
| 09 | 480 | 1.35 | 1.35 | -0.10 |
| 10 | 496 | 1.23 | 1.23 | -0.14 |
| 11 | 486 | 0.98 | 0.96 | -0.18 |
| 12 | 558 | 1.07 | 1.06 | -0.17 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.509 | -0.100 |
| 12:00 | 3040 | 1.159 | -0.150 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.22 |
| 02 | 1.33 |
| 03 | 1.40 |
| 04 | 1.38 |
| 05 | 1.41 |
| 06 | 1.51 |
| 07 | 1.59 |
| 08 | 1.51 |
| 09 | 1.35 |
| 10 | 1.23 |
| 11 | 0.96 |
| 12 | 1.06 |
| _fallback_ | _1.34_ |
