# W3 — Physical ceiling / remaining-upside model (spec stub)

**Status**: Spec stub only. Implementation is data-gated and won't start
until ~30 days of clean intraday signal data have accumulated.

**Companion code**: Will live in `bot/scripts/intraday_predictor.py`
(estimator) and a calibration script under `bot/scripts/` (offline).
Consumed via the existing `probability_in_bin` interface's
`truncate_at_hi` parameter, which W2 Phase A reserved exactly for this.

## What W3 does

The current day-high distribution has an asymmetric structural problem:
the floor rises through the day (observed_max ratchets up) but the
upper tail is symmetric to the lower in the Gaussian path and doesn't
contract asymmetrically. At 5pm on a clear day with sun about to set,
the σ-based distribution still assigns real probability to large
further rises that two hours of remaining insolation physically can't
produce. Near a bin edge, that mass spillover reads as a boundary loss.

W3 estimates a physical upper bound on how much more the day can warm
given current conditions:

```
P(day_high ≥ T | current_temp, solar_elevation, cloud_cover, dewpoint, hour)
```

and uses it as `truncate_at_hi` in `probability_in_bin`. The Gaussian
mass above the ceiling renormalizes into the still-physically-possible
range, which both contracts the upper tail asymmetrically AND smooths
the late-day pricing transitions that the current binary `bin_lock`
switch produces.

## Design — hierarchical pooling

Per-city quantile regression at the 98th percentile of "subsequent
rise from current temp" conditioned on (solar elevation, cloud cover,
dewpoint depression). Naive per-city fitting is sample-thin in exactly
the regime where it matters most (clear, dry, mid-afternoon — that's a
narrow conditioned subset of any one city's history).

To buy statistical power without sacrificing too much specificity:

1. **Climate-regime buckets** as the prior:
   - Marine subsidence: SF, LA, Seattle
   - Humid subtropical with sea-breeze cap: Miami
   - Continental dry: Denver, Austin, Dallas
   - Humid subtropical inland: Atlanta, Houston
   - Northern temperate: NYC, Chicago

2. **Pool calibration data within a regime** to fit the ceiling
   coefficients. Per-city offsets only where `N_samples > 200` for
   that city's conditioned cells.

3. **Quantile target: 98th percentile** (NOT 95th). Truncating off a
   bin that actually wins is a total loss on the position; leaving a
   slightly-fat upper tail is a mild misize. Asymmetric loss → use the
   conservative quantile.

4. **Per-regime safety buffer**: start at 0.5°F absolute on top of the
   98th-percentile estimate. Tunable from W0 results once they exist.

5. **Conditioning variable: solar elevation, NOT wall-clock hour.**
   The diurnal rise envelope shifts hard with solar declination
   through the season; wall-clock 3pm in June ≠ wall-clock 3pm in
   October. Solar elevation is already computed per scan and lives in
   `bot/solar.py`.

6. **Refit cadence: weekly cron, rolling 90-day window**. No fixed
   monthly cycle — the recalibration just runs.

## REGIME MIS-BUCKETING DIAGNOSTIC — load-bearing requirement

The climate-regime taxonomy above WILL misclassify at least two
cities, and the failure is silent. A wrong regime assignment degrades
the pooled fit invisibly with no error thrown. The taxonomy doesn't
need to be perfect; we need to detect when a specific city is
mis-bucketed.

**Required output of the W3 calibration script** (not optional, not a
future improvement):

For every city, after the pooled regime fit has been computed:

1. **Per-city residual bias**: compute the city's own observed rises
   against its regime's fitted 98th-percentile ceiling. The city's
   residuals should be approximately symmetric around the ceiling
   line. If the city sits SYSTEMATICALLY inside or outside its
   regime's fitted quantile, flag as mis-bucketed.

2. **Per-city exceedance rate**: count what fraction of the city's
   actual observed rises exceeded the regime-fitted ceiling. For a
   well-fit 98th percentile, this should be ~2% per city. If it's 5%+
   for a city, the city is mis-bucketed AND the bot has been clipping
   real wins from that city — flag with high priority.

**Priors for who'll trip the diagnostic** (informed expectations, not
predictions):

- **Miami** in `humid-subtropical-with-sea-breeze-cap` is currently
  the regime-of-one. The bucket exists specifically because Miami's
  upper-tail physics (afternoon thunderstorm cutoff, dewpoints in the
  70s) doesn't look like LA's marine-layer subsidence ceiling. If the
  diagnostic shows Miami's pooled ceiling is way off, Miami may need
  its own per-city fit even though `N_samples` is thin.
- **Denver** in `continental-dry` is right on average but has a
  downslope/Chinook regime that produces the fat warm tail
  specifically on the days the ceiling most needs to not clip. The
  exceedance rate is the metric that catches this — Denver may show a
  good mean-bias number but a high exceedance rate. Trust the
  exceedance rate.

Neither prior should drive the bucketing choice upfront. The whole
point of the diagnostic is to discover mis-bucketing empirically
rather than guess at the taxonomy. The priors only tell us where to
look first when the numbers come in.

## Wiring — non-changes from W2 Phase A

The integrator already supports two-sided truncation. W2 Phase A
introduced:

```python
probability_in_bin(
    bin_lo_c, bin_hi_c, cdf,
    truncate_at_lo = observed_max_c,    # the rising floor (current)
    truncate_at_hi = None,              # W3 fills this
)
```

W3's contribution is supplying `truncate_at_hi` from the physical
ceiling estimator. No call-site changes in `predict_bins` are needed —
the existing dispatch becomes a switch:

```python
ceiling_c = estimate_remaining_upside_ceiling(
    current_temp_c, solar_elevation_deg,
    cloud_cover_pct, dewpoint_c,
    city, climate_regime,
)
probability_in_bin(c_lo, c_hi, day_high_cdf,
                    truncate_at_lo=truncate_at,
                    truncate_at_hi=ceiling_c)
```

## Then-delete: old wall-clock σ branches

After W3 ships and the physical ceiling is proven calibrated, the
existing wall-clock σ narrowing branches in `estimate_day_high_dist`
(`bot/scripts/intraday_predictor.py:722-731`, `:747-765`) and the
binary `bin_lock` override (`:917-936`) become inert — the physical
ceiling will be tighter in every case where it was correct, and the
old branches just don't fire.

**Sequencing rule** (per the data-quality contract's "fail-down" rule):
the deletion happens in a SEPARATE PR from the W3 introduction. Both
systems run in parallel for ≥2 weeks. The physical ceiling takes
precedence only where it's tighter than the old narrowing. Once we've
proven the old branches are inert (count their fire-rate; should be
near zero), they get deleted. Don't couple the new-model introduction
with the old-safety-net removal.

## Data dependencies — what W3 needs that the wait is for

For each (city, scan) pair, the calibration needs:

| Field | Source | Currently captured? |
|---|---|---|
| current_temp_c | latest METAR temp | yes — in `raw_metar_log` and via the obs path |
| solar_elevation_deg | `bot/solar.py` | yes — computed per scan, not persisted |
| cloud_cover_pct | NWS hourly forecast `properties.skyCover` | not persisted |
| dewpoint_c | latest METAR dewpoint | yes — in `raw_metar_log` |
| subsequent_rise_c | EOD `observed_max_c` minus `current_temp_c` at scan time | yes — joinable |

Two fields (solar elevation, cloud cover) need to be persisted to
`paper_predictor_signals` before calibration data is usable. That's a
small migration analogous to `cooling_confidence` — defer until W3
implementation starts to avoid noise in the column until then. The
field names will be `solar_elevation_deg` and `cloud_cover_pct`.

## Open question — already deferred to "when W0 returns"

What if W0 segmentation reveals that boundary losses are dominantly
caused by `settle_divergence` rather than distribution-shape issues?
Then W3 is lower priority and W1 (observation-source remap) is the
first move. W3 still ships eventually, but the calendar shifts.

## What this stub DOESN'T cover

- Concrete coefficient initialization values for the quantile
  regression. Will be empirically derived from calibration data, not
  spec'd upfront.
- The exact form of `estimate_remaining_upside_ceiling` (linear in the
  conditioning variables, GBM, something else). Start with linear
  quantile regression as the baseline; revisit if Brier doesn't move
  enough.
- The exact form of how regime mis-bucketing triggers re-bucketing.
  For now: the diagnostic emits the flagged cities; a human reviews
  and decides whether to move them, NOT auto-reassignment.