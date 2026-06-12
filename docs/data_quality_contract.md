# Data-quality contract — distribution sources & degradation behavior

**Status**: Draft, not yet enforced in code.
**Owner**: Predictor / W2 workstream.
**Companion code**: `bot/scripts/intraday_predictor.py`, `bot/scheduled_predictor.py`.

## Why this exists

The predictor's day-high distribution can come from three sources arranged
as a fallback chain:

1. **Primary** — external ensemble percentiles (NBM via GRIB2, or
   Open-Meteo ensemble endpoint). Not yet wired (W2 Phase B).
2. **Secondary** — empirical residual CDF built from per-city historical
   `(forecast - observed)` residuals. Code shipped, dormant (W2 Phase C).
3. **Tertiary** — Gaussian `N(mu, sigma)` fit from per-city RMSE. Always
   available. Currently the only path in production.

Without an explicit contract, a source returning *partial garbage* —
50 ensemble members where 8 are NaN, an NBM cycle that's stale by 12h
but still returns 200 OK, a residual list whose newest entry is from
last quarter — silently produces a confident-looking distribution from
degraded inputs. The predictor's `our_p` would be presented to Kelly
sizing as if it were trustworthy, and the bot would deploy real capital
against a distribution it has no diagnostic basis to trust.

This document specifies, for each source: (a) the minimum trustability
threshold, (b) what happens when the threshold isn't met, (c) how the
fallback is observable downstream.

## Design principles

1. **Triggers, not just order.** Specifying the fallback chain
   (NBM → empirical → Gaussian) is necessary but insufficient. The
   contract is in the *triggers* — the specific thresholds that move
   us from one tier to the next.

2. **Degrade sizing, not the gate stack.** The existing gate stack
   (`MIN_MARKET_PROB`, `MIN_EDGE`, etc.) decides whether to act on a
   signal. Data quality decides *how much* to act. A degraded source
   doesn't veto the trade — it scales the stake down. Vetoing entirely
   is reserved for the no-trustworthy-source case.

3. **One persisted flag.** `paper_predictor_signals.data_quality_flag`
   already exists. It carries both the tier and the reason in a single
   string so audits can join on it without schema churn:
   - `"primary"` — NBM or external ensemble produced our_p
   - `"empirical"` — empirical residual CDF produced our_p
   - `"gaussian"` — fell back to Gaussian for any reason
   - `"empirical_fallback:samples=22"` — fell back to gaussian because
     the empirical path failed a quality check (sample count below
     threshold, in this case)
   - `"primary_fallback:stale_18h"` — primary returned but was rejected
     for staleness; emitted by the next tier that actually produced our_p

4. **Fail-down to gaussian, never up.** Gaussian is the safety net. It
   has a wide enough σ floor (`MIN_SIGMA_C=0.80°C`) that even a city
   with weak calibration gets a defensible distribution. The contract
   never silently substitutes one degraded source for another — it
   falls through to the next, simpler, more defensive source.

## Per-source trust thresholds

### Source 1 — External ensemble (Phase B, future)

The exact ingestion path will be either Open-Meteo's ensemble endpoint
or NBM GRIB2 from NODD/AWS. Either way, the response is a set of
ensemble members or percentile points that build the CDF.

| Check | Trigger | If failed |
|---|---|---|
| Cycle staleness | `now - cycle_issued > 9h` | Fall to empirical |
| Member count | `n_non_null_members < 20` (Open-Meteo) or `n_percentile_points < 5` (NBM) | Fall to empirical |
| NaN fraction | `n_null_members / n_total_members > 0.20` | Fall to empirical |
| Percentile monotonicity | Any P50 > P75 or similar inversion | Fall to empirical |
| Range plausibility | `max(members) - min(members) > 25°C` or `< 0.3°C` | Fall to empirical |

Rationale on each:

- **Cycle staleness**: NBM runs every 6h. A 9h-old cycle means we missed
  the next refresh; the source upstream is unhealthy. 9h is a hedge
  against narrow windows where a cycle is briefly delayed.
- **Member count**: Open-Meteo's blended ensemble returns ~50 members
  when healthy; we need ≥20 to construct a reasonable empirical CDF
  from members. NBM's percentile product has discrete bands (P10, P25,
  P50, P75, P90) — fewer than 5 means partial response.
- **NaN fraction**: If 20%+ of members are null, the response is
  structurally compromised. Better to use the empirical tier than to
  silently treat 40 valid members as if they were a full ensemble.
- **Monotonicity**: A CDF that isn't monotone is a parse bug or upstream
  corruption. Reject hard.
- **Range plausibility**: 25°C spread = forecast is degenerate.
  0.3°C spread = ensemble collapsed (all members agree to spurious
  precision). Both indicate the calibration upstream is broken.

### Source 2 — Empirical residual CDF (Phase C, shipped, dormant)

Already partially gated by `EMPIRICAL_MIN_SAMPLES=30` in
[bot/scripts/intraday_predictor.py](bot/scripts/intraday_predictor.py).
The full contract adds:

| Check | Trigger | If failed |
|---|---|---|
| Sample count | `len(centered_residuals) < 30` | Fall to gaussian (existing) |
| Calibration age | `now - calibration_generated_at > 45 days` | Fall to gaussian + log warning |
| Newest residual age | `now - newest_residual_date > 21 days` | Fall to gaussian + log warning |
| NaN in residuals | Any non-finite value in the list | Fall to gaussian |
| Spread sanity | `stdev(residuals) < 0.2°C` or `> 6°C` | Fall to gaussian |
| Zero-mean preservation | `abs(mean(residuals)) > 0.5°C` | Log warning, do NOT fall back (mean drift is recalibration latency, not corruption) |

Rationale:

- **Sample count**: Below 30 the empirical CDF is noise. Existing rule.
- **Calibration age 45d**: Forecast skill changes seasonally. A
  March calibration applied to August trading is dangerous. 45d is
  generous (allows monthly cron); tighten if Brier shows seasonal drift.
- **Newest residual 21d**: Catches the case where the calibration file
  exists but the cron stopped running. The file is stale even if its
  generated_at is fresh-looking.
- **NaN check**: Defends against partial calibration runs where some
  days produced null residuals; the existing calibration script's
  `residuals` list could pick those up if upstream changes.
- **Spread sanity**: Same logic as the ensemble — degenerate spread
  means calibration is broken upstream.
- **Zero-mean drift**: The residuals are *supposed* to be mean-zero
  by construction (we subtract the per-city bias before saving). If
  the mean has drifted, it means the bias correction is out of date
  but the underlying shape is still useful. Warn, don't fall back.

### Source 3 — Gaussian (always available)

The safety floor. No data-quality gate — if we got here, every other
source failed and we still need to produce a distribution. Existing
guards:

- `MIN_SIGMA_C=0.80°C` floor in
  [forecast_rmse_calibration.py](bot/scripts/forecast_rmse_calibration.py)
- `DEFAULT_FORECAST_SIGMA_C=2.0°C` ultimate fallback in
  [intraday_predictor.py](bot/scripts/intraday_predictor.py)

The only Gaussian-specific check: if even the per-city σ is missing
(no calibration file at all), the predictor uses the default. That
case sets `data_quality_flag = "gaussian_default_sigma"` so audits can
identify cities that have never been calibrated.

## Degradation behavior — sizing scalar

The contract maps each `data_quality_flag` to a Kelly sizing multiplier
applied at the stake-computation step. Defined as env-tunable constants
in [bot/scheduled_predictor.py](bot/scheduled_predictor.py):

```
DATA_QUALITY_SIZE_PRIMARY                 = 1.00   # relative-tier (informational)
DATA_QUALITY_SIZE_EMPIRICAL               = 1.00   # relative-tier (informational)
DATA_QUALITY_SIZE_GAUSSIAN                = 1.00   # relative-tier (informational)
DATA_QUALITY_SIZE_GAUSSIAN_DEFAULT_SIGMA  = 0.30   # absolute-trustability haircut
DATA_QUALITY_SIZE_COLD_START_SUSPECT      = 0.30   # absolute-trustability haircut
DATA_QUALITY_SIZE_BLOCK                   = 0.00   # don't trade
```

### Two different kinds of haircut

There's a deliberate split here, and it matters: the multipliers
encode two different things, and conflating them was the original
mistake.

**Relative-confidence tiers** (PRIMARY, EMPIRICAL, GAUSSIAN). These
exist to express "this distribution is less trustworthy than the
best-available one." A 40% Gaussian haircut would have meant something
in the original spec because a PRIMARY tier existed at 1.00 to
compare against. **Today no PRIMARY tier exists.** Every city is on
Gaussian. A 40% haircut on Gaussian right now isn't a confidence
statement — it's a unilateral 40% capital reduction wearing a
risk-management costume, on a model that has been live and (position-
tracking bugs aside) functioning.

These tiers ship at 1.00 across the board. The differentiation lights
up as a config change — no code change — the moment a real PRIMARY
tier exists to make "Gaussian" mean "the degraded option." Until then,
the flag carries provenance and the multiplier doesn't act.

**Absolute-trustability tiers** (GAUSSIAN_DEFAULT_SIGMA,
COLD_START_SUSPECT, BLOCK). These are *not* relative-confidence
statements. They are facts about the inputs that are true regardless
of what other tiers exist:

- **Default-σ Gaussian at 0.30**: We have literally never calibrated
  this station and are using a generic 2°C σ. Dangerous regardless of
  whether a better tier exists. 70% haircut.

- **Cold-start suspect at 0.30**: The first scan of the day for this
  city was after the city's climatological peak hour (`HH_local >=
  COLD_START_PEAK_HOUR_LOCAL`, default 14:00). The NWS hourly endpoint
  may have returned only evening-cooling periods on its very first
  fetch — the recovery helper has no higher prior value to draw on.
  `forecast_high_c` for this city today may be the evening cooling
  curve, not the actual day's high. 70% haircut. The exact same logic
  as default-σ: an absolute trustability statement true on its own
  merits.

- **Block at 0.00**: Reserved for cases where every source's
  trustability check failed AND the Gaussian fallback itself can't run
  (NaN propagation, catastrophic forecast failure). Should be
  effectively impossible. Included as defense-in-depth.

These absolute-trustability tiers carry their haircut today —
independent of whether a PRIMARY tier exists. They're real
information about the input quality, not comparative statements.

### Composable flags

A row can hit multiple conditions: e.g., a cold-start day on an
uncalibrated city would have `data_quality_flag =
"gaussian_default_sigma,cold_start_suspect"`. When multiple labels
apply, the size factor is the **minimum** of all applicable values
(most conservative). For the example: `min(0.30, 0.30) = 0.30`. For
a non-cold-start uncalibrated city: `min(1.00, 0.30) = 0.30`.

### The block tier is rare

With Gaussian as a safety floor that's always available, the only way
to hit `DATA_QUALITY_SIZE_BLOCK` is a catastrophic failure where
forecast itself returned no data. In that case the bot was already
going to skip the bin via existing logic. Tracking the size-zero
outcome explicitly via the data_quality_flag makes it auditable.

## How this composes with existing gates

The contract runs **after** the distribution is constructed and
**before** Kelly sizing. The order through `evaluate_gates` and the
sizing path is:

```
1. Fetch forecast + obs                   (existing)
2. Construct CDF via dispatch              (W2 Phase A; live)
3. Compute mu, sigma                       (existing)
4. Per-bin: probability_in_bin             (W2 Phase A; live)
5. Compute edge                            (existing)
6. evaluate_gates(...)                     (existing — MIN_MARKET_PROB,
                                              edge, liquidity, W4 cap)
7. ⭐ Compute Kelly stake                  (existing)
8. ⭐ Scale by data_quality_size_factor    (NEW — this contract)
9. Floor check (MIN_STAKE_USD)             (existing)
10. Place order                            (existing)
```

The contract does not bypass any existing gate. It only scales the
stake. A signal that fails the MIN_EDGE or W4 gates never reaches step
8 — data quality is only consulted on signals that have already passed
fundamentals.

## Observability

The single column `paper_predictor_signals.data_quality_flag` carries
the tier + reason:

```
primary
empirical
empirical_fallback:samples=22
empirical_fallback:newest_residual_age=34d
gaussian
gaussian_default_sigma
primary_fallback:stale_11h
primary_fallback:nan_fraction=0.24
```

This composes with the dashboard's existing filtering: a per-city
breakdown of `data_quality_flag` counts shows operationally which
cities are running on which tier, and a sudden shift in the
distribution flags an upstream regression in the calibration cron or
the ensemble source.

For paid auditing during W0 (when it eventually runs): every settled
market in the audit CSV will already carry `data_quality_flag` for the
last scan before resolution, so the segmentation can report whether
boundary_shape_wrong cases concentrate in `gaussian` rows
(distribution-shape was the problem) vs `empirical` rows (the empirical
shape itself was wrong).

## Failure modes this catches

| Failure mode | Caught by |
|---|---|
| External ensemble endpoint returns 200 with stale cycle | Cycle staleness check |
| NBM partial response (some percentiles NaN) | NaN fraction check |
| Open-Meteo ensemble collapses (all members agree to 0.1°C) | Range plausibility check |
| Calibration cron stopped 60 days ago | Calibration age + newest residual age checks |
| New city added without calibration entry | `gaussian_default_sigma` flag + 0.30 size factor |
| Bot first scanned a city's day post-peak; forecast_high_c may be evening cooling curve | `cold_start_suspect` flag + 0.30 size factor |
| Catastrophic NaN propagation reaches CDF construction | Block tier (0.00 size) |
| Live predictor uses Gaussian because of forecast bug | `gaussian` flag visible in audit; existing behavior |

## Failure modes this does NOT catch (accepted)

| Failure mode | Why not caught |
|---|---|
| External ensemble is systematically biased (every cycle 2°C low) | Bias correction is a separate per-station system; the contract treats the source as trusted if it passes structural checks |
| Empirical residuals are tight because calibration source has its own bug (the Open-Meteo `historical-forecast` issue we already hit) | Source-vs-deployment mismatch is detected by W0's segmentation, not by this contract — by design |
| Forecast data is correct but `observed_max` is wrong (settled_divergence) | W0 audit territory, not data quality |
| Gaussian σ floor (0.80°C) is too tight for a marine city | Calibration concern; this contract enforces *trustability* of sources, not optimality |

## Implementation checklist for W2 Phase B kick-off

Before any Phase B PR touches the dispatch in `predict_bins`:

- [x] **Contract accepted** (this revision, two modifications from
  draft: relative tiers at 1.00, cold_start_suspect added)
- [ ] `DATA_QUALITY_SIZE_*` constants added to `scheduled_predictor.py`
- [ ] Cold-start detection wired into the scan loop alongside the
  recovery helper: query `MIN(scanned_at_utc)` for `(city, event_date)`;
  if converted to city's local time it is >= `COLD_START_PEAK_HOUR_LOCAL`
  (default 14:00), append `cold_start_suspect` to the row's
  `data_quality_flag`
- [ ] Sizing path modified to apply the multiplier as a MIN over all
  flag components (one comma-split + min(), plus the env-var reads)
- [ ] Unit tests covering:
  - Each `data_quality_flag` value maps to the correct multiplier
  - Composable flag `gaussian,cold_start_suspect` resolves to
    `min(1.00, 0.30) = 0.30`
  - A stake of $10 at 0.30 multiplier produces $3 final stake
  - A `BLOCK` flag produces stake=0 with a clean SKIP reason
- [ ] Cold-start integration test: synthesize a fresh DB, insert a
  signal with `scanned_at_utc` = 18:00 local for a city whose
  COLD_START_PEAK_HOUR_LOCAL is 14:00, verify `cold_start_suspect`
  fires and the size factor is 0.30
- [ ] Dashboard adds a per-city `data_quality_flag` breakdown column
- [ ] One end-to-end test that synthesizes a "stale ensemble" response
  and verifies the dispatch falls through to empirical AND the row's
  flag reads `primary_fallback:stale_*`

## Open questions

1. **Should the size haircuts be per-city configurable?** A
   well-calibrated marine city on the gaussian path may deserve a
   smaller haircut than a sparsely-calibrated continental city on the
   same path. Defer until W0 + Brier results show whether per-city
   tuning matters.

2. **Should `BLOCK` actually skip the SCAN, or just zero out the
   stake?** Zero stake is observationally identical to a `SKIP` row in
   the dashboard but preserves the model output for later diagnostics.
   Recommend zero-stake-but-write-row for now.

3. **Does the contract apply to PAPER mode the same as LIVE?** Yes —
   paper mode should reflect the same degradation so the paper-shadow
   period (W2's promotion gate) gives a faithful preview of live
   behavior.

## Decision

Accepted? Y / N / Modify (paste edits inline). Once accepted, this
file becomes the source of truth and W2 Phase B can begin.