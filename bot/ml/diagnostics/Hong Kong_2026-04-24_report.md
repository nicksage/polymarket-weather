# ML Distribution Model — Evaluation Report

**City:** Hong Kong
**Generated:** 2026-04-25T00:38:28.817243+00:00
**Feature version:** `v1.0`
**Training window:** 2016-03-25 → 2026-03-25  
**Training rows:** 7306  (unique dates: 3653, OOF predictions: 6080)

## Headline metrics (out-of-fold)

| Metric | Value |
|---|---|
| Point RMSE | **0.941 °C** |
| Mean bias | -0.062 °C |
| Fallback σ | 0.939 °C |

## Comparison vs naive baselines

All values are RMSE in °C.  ML model competes against trivial heuristics that need no training.

| Method | RMSE | Δ vs ML | Verdict |
|---|---|---|---|
| **ML model (this)** | **0.941** | — | — |
| persistence_yesterday | 2.060 | +1.119 | ML wins |
| yoy_same_date_last_year | 3.880 | +2.939 | ML wins |
| lag_7d_ago | 3.762 | +2.821 | ML wins |
| climatology_doy_mean | 2.808 | +1.867 | ML wins |

_(positive Δ means ML beats that baseline)_

## Calibration — do residuals respect the predicted σ?

σ comes from per-month residual std (fallback to city-wide).  For a well-calibrated Gaussian: 68% of |residual| ≤ 1σ, 95% ≤ 2σ, 99.7% ≤ 3σ.

| Window | Empirical | Expected | Verdict |
|---|---|---|---|
| ±1σ | 71.1% | 68.3% | well-calibrated |
| ±2σ | 95.5% | 95.4% | well-calibrated |
| ±3σ | 99.2% | 99.7% | well-calibrated |

## Per-month performance

Months with fewer than 10 OOF samples are omitted.

| Month | n | RMSE °C | σ °C | Bias °C |
|---|---|---|---|---|
| 01 | 558 | 1.19 | 1.19 | -0.09 |
| 02 | 508 | 1.19 | 1.19 | +0.08 |
| 03 | 546 | 1.07 | 1.07 | -0.07 |
| 04 | 480 | 0.93 | 0.92 | -0.06 |
| 05 | 496 | 0.84 | 0.84 | -0.05 |
| 06 | 480 | 0.72 | 0.71 | -0.07 |
| 07 | 496 | 0.78 | 0.78 | -0.03 |
| 08 | 496 | 0.85 | 0.85 | -0.10 |
| 09 | 480 | 0.89 | 0.89 | -0.03 |
| 10 | 496 | 0.86 | 0.85 | -0.11 |
| 11 | 486 | 0.81 | 0.81 | -0.08 |
| 12 | 558 | 0.95 | 0.94 | -0.11 |

## Per-decision-hour performance

Each training day produces two rows: 10:00 and 12:00 local.  The 12:00 row has more morning observations available, so should yield lower RMSE if the model is using observation features effectively.

| Hour | n | RMSE °C | Bias °C |
|---|---|---|---|
| 10:00 | 3040 | 1.029 | -0.020 |
| 12:00 | 3040 | 0.843 | -0.103 |

## σ per month (used at inference time)

| Month | σ °C |
|---|---|
| 01 | 1.19 |
| 02 | 1.19 |
| 03 | 1.07 |
| 04 | 0.92 |
| 05 | 0.84 |
| 06 | 0.71 |
| 07 | 0.78 |
| 08 | 0.85 |
| 09 | 0.89 |
| 10 | 0.85 |
| 11 | 0.81 |
| 12 | 0.94 |
| _fallback_ | _0.94_ |
