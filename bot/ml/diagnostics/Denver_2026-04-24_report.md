# ML Distribution Model — Evaluation Report

**City:** Denver
**Generated:** 2026-04-25T00:38:05.062097+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **1.905 °C** |
| Mean bias | -0.051 °C |
| Fallback σ | 1.905 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **1.905** | — | — |
| persistence_yesterday | 5.779 | +3.874 | ML wins |
| yoy_same_date_last_year | 8.835 | +6.930 | ML wins |
| lag_7d_ago | 8.793 | +6.888 | ML wins |
| climatology_doy_mean | 6.701 | +4.796 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 73.2% | 68.3% | well-calibrated |
| ±2σ | 94.8% | 95.4% | well-calibrated |
| ±3σ | 98.8% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.82 | 1.82 | -0.01 |
| 02 | 508 | 2.29 | 2.29 | -0.10 |
| 03 | 546 | 2.07 | 2.07 | +0.11 |
| 04 | 480 | 2.12 | 2.12 | +0.16 |
| 05 | 496 | 1.88 | 1.87 | -0.08 |
| 06 | 480 | 1.93 | 1.93 | -0.08 |
| 07 | 496 | 1.52 | 1.52 | +0.02 |
| 08 | 496 | 1.56 | 1.56 | +0.10 |
| 09 | 480 | 1.65 | 1.65 | -0.04 |
| 10 | 496 | 2.13 | 2.13 | -0.11 |
| 11 | 486 | 1.78 | 1.75 | -0.36 |
| 12 | 558 | 1.91 | 1.90 | -0.22 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 2.184 | +0.011 |
| 12:00 | 3040 | 1.578 | -0.113 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.82 |
| 02 | 2.29 |
| 03 | 2.07 |
| 04 | 2.12 |
| 05 | 1.87 |
| 06 | 1.93 |
| 07 | 1.52 |
| 08 | 1.56 |
| 09 | 1.65 |
| 10 | 2.13 |
| 11 | 1.75 |
| 12 | 1.90 |
| _fallback_ | _1.90_ |
