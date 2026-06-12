# Project state & 30-day roadmap

A self-contained handoff document. Designed to be fed into a fresh
Claude session (or read cold by a human) without requiring any prior
conversation context. If you're picking this up later, **start at
Section 0** and follow the document straight through; everything is
sequenced so earlier sections inform later ones.

---

## 0. How to use this document

- **You are mid-project.** The bot is running in production. Most of
  the immediate work is done. The next ~30 days are an enforced wait
  while clean calibration data accumulates. This document tells you
  what was built, what's accumulating, and what to do when the wait
  is over.
- **Don't make implementation decisions without reading Section 5
  (design rules) first.** Several rules were learned from specific
  failure modes (Section 4); ignoring them re-opens those failures.
- **Section 7 is the decision tree for after the wait.** When ~30
  days of data exist, start there. The branches name specific
  workstreams; concrete instructions follow.
- **Operational commands are in Section 8.** Deploy, test, verify —
  all the literal shell incantations.
- **When in doubt, check the source files.** Every reference is a
  link with a path; the code is the source of truth. This document
  paraphrases.

---

## 1. What the bot does — one paragraph

The bot scans Polymarket every ~2 minutes for active "highest
temperature today" markets in US cities. For each market, it builds a
per-bin probability distribution over the day's eventual high using
NWS forecasts, live METAR observations, neighbor-station signals, and
a per-city σ calibration. It compares its own probability to the
market's implied probability, computes an edge per bin, and — if a
stack of gates passes (edge, liquidity, hour-of-day, dedup, exposure
caps, market-anchored risk cap) — places a Kelly-sized LIVE_BUY
order against the top-P bin. All decisions are persisted: every
scan, every bin, every gate result, every order attempt.

---

## 2. Architecture as built

### 2.1 Prediction pipeline

Live code: [bot/scripts/intraday_predictor.py](../bot/scripts/intraday_predictor.py).

**Distribution model**: Truncated normal `N(μ, σ²)` conditioned on
`observed_max ≤ day_high`. μ starts at the NWS forecast high
(bias-corrected per station), then gets nudged through six stacking
branches in `estimate_day_high_dist()` based on time of day,
observation accumulation, neighbor signals, and ensemble disagreement.

**CDF-agnostic integrator (W2 Phase A)**: `probability_in_bin(lo, hi,
cdf, truncate_at_lo, truncate_at_hi)` accepts any CDF callable. Today
only `make_gaussian_cdf(mu, sigma)` is wired. The empirical-residual
CDF (`make_empirical_residual_cdf`) is implemented but dormant, gated
by `PREDICTOR_CDF_IMPL=empirical` env var + ≥30 centered residuals per
city. Future W3 work uses the `truncate_at_hi` parameter for the
physical ceiling.

**Dispatch**: `PREDICTOR_CDF_IMPL` env var picks the CDF
implementation. Default `gaussian`. Setting `empirical` requires the
calibration data described in Section 6.

### 2.2 Position-tracking three-signal model

This was the first major fix in the project. The dashboard previously
used a single signal (token-in-wallet + value filter) to answer three
different questions: "did a fill happen?", "how much cost basis?", and
"is this still tradeable?" Conflating these caused six surface bugs
from one architectural mistake.

The fix: three explicit signals.

| Signal | Source | Answers |
|---|---|---|
| `HELD` | Polymarket positions API (no value filter) | Did a fill happen? |
| `DEPLOYED` | `size × avg_price` per held token | Cost basis for topup decisions |
| `MARKET_OPEN` | Gamma `closed` flag per bin, persisted in `paper_predictor_signals.market_closed` | Still tradeable? Drives RESOLVED badge |

Companion fixes:
- Per-contract daily $ ceiling (`MAX_PER_CONTRACT_USD=15`, prevents
  runaway topup scaling)
- Reconciliation sweep at scan start (asks CLOB about `placed` orders,
  updates to `filled` / `cancelled` / `stale`)
- KPI dedup-by-contract (multiple topup signal rows for one position
  no longer inflate "n LIVE BUYS")

Dashboard renders `RESOLVED` pills for held-but-settled positions
distinct from `LIVE` pills for active ones, with final P&L pulled from
API for resolved positions.

### 2.3 Gate stack

Live code: [`evaluate_gates()` in scheduled_predictor.py](../bot/scheduled_predictor.py).

Order matters — gates are checked top-to-bottom; first failure stops:

1. `too_early` / `too_late` — hour bounds (default 13–22 local)
2. `market_too_skeptical` — `market_p < MIN_MARKET_PROB` (default 0.15)
3. `low_edge` — tiered (`MIN_EDGE=0.10` if market_p≥0.75, else
   `MIN_EDGE_LOW_MKT=0.05`)
4. **`liquid_market_strong_disagreement` (W4)** — `liquidity ≥ $10k AND edge ≥ 0.40`
5. `priced_in` — `market_p ≥ MAX_MARKET_PRICE` (default 0.95)
6. `thin_book` — `liquidity < MIN_LIQUIDITY_USD` (default $300)
7. `dedup_today`
8. `trades_cap` — `MAX_TRADES_PER_DAY=25`
9. `exposure_cap` — `MAX_DAILY_EXPOSURE=$200`

W4 sits after `low_edge` deliberately, so the reason ordering stays
informative — a tiny-edge signal is rejected as `low_edge`, not as
W4. W4 only catches the "huge edge on liquid book" case that's almost
certainly a stale model or bug rather than real edge.

### 2.4 Forecast-recovery helper

Live code: `recover_persisted_day_forecast()` in
[bot/scheduled_predictor.py](../bot/scheduled_predictor.py).

**The bug it fixes**: NWS `/forecastHourly` only returns periods from
"now" forward. By late afternoon, "today's" remaining periods describe
only the evening cooling curve. The bot's `forecast_high_c` therefore
ratchets DOWN through the afternoon as the actual peak periods drop
out of the result. Verified against SF 2026-06-11: 28.33°C stable from
07:04 UTC through 23:06 UTC, then declining to 17.22°C by 23:24 UTC
(16:24 PDT), tracking the cooling curve exactly.

**The fix**: at the cache-miss point in the scan loop, query
`MAX(forecast_high_c) WHERE city=? AND event_date=?`. If the persisted
max exceeds the fresh fetch, prefer the persisted value. Works because
the morning's fetch correctly captured the day's peak; the helper just
prevents it from being overwritten as the day winds down.

**Cold-start limitation**: days where the bot's FIRST scan is after
the local peak hour can't be recovered (no higher prior exists). These
get the `cold_start_suspect` flag (Section 2.6).

Verified firing in production. Log line: `forecast_high recovered for
{city}: {old}°C → {new}°C (NWS hourly evening-scan bug)`.

### 2.5 Invariant guards — observational forever

Live code: [bot/scripts/invariant_guards.py](../bot/scripts/invariant_guards.py).

**Design rule (load-bearing)**: Guards LOG, COUNT, and SURFACE. They
never GATE or REMEDIATE. The moment a guard suppresses a trade or
corrects a value, it becomes load-bearing infrastructure and needs its
own calibration and its own guards — exactly the silent-assumption
layer pattern that produced the forecast ratchet bug. If a guard
reveals a problem worth acting on, the action belongs in the explicit
gate stack or data-quality contract.

**Enforced by code**: `test_no_import_from_prediction_path` in
[bot/tests/test_invariant_guards.py](../bot/tests/test_invariant_guards.py)
uses AST parsing to assert that `invariant_guards` is never imported
by anything in the prediction or trading path. Any future PR that
adds such an import fails this test.

**Current guards** (each runs at end of every scan):

| Guard | Invariant | Caveat |
|---|---|---|
| `observed_max_monotone` | observed_max_c can only rise within a day | Tolerance ±0.05°C for float noise |
| `forecast_high_monotone` | forecast_high_c non-decreasing (recovery helper guarantees this) | Catches helper regressions |
| `sigma_monotone_post_peak` | σ non-increasing after `forecast_peak_hour` | Known false-positive: ensemble-disagreement inflation can legitimately re-widen σ. Operator de-noises via dashboard's "is-new" flag |
| `cooling_confidence_monotone_post_peak` | cooling_confidence non-decreasing once past peak | Oscillation here is bin-lock-discontinuity churn surfacing as data |
| `mu_jump_incoherent_with_neighbors` | μ jumped ≥1.5°C in <1h AND upwind neighbors did NOT move coherently | High-precision source-flip detector. Real weather is spatially coherent; parse errors and source-flips are station-specific |

Violations write to `guard_violations` table (Section 3.4). The
dashboard panel for "per-(city, guard-type) violations with is-new
flag" is specified but NOT YET BUILT — that's a deferred dashboard
task.

### 2.6 Data-quality contract & sizing scalar

Spec: [docs/data_quality_contract.md](data_quality_contract.md).
Live code in [bot/scheduled_predictor.py](../bot/scheduled_predictor.py).

The contract governs how degraded inputs translate to degraded sizing.
It's specified as triggers (when does a source become untrustworthy)
and multipliers (how much do we haircut Kelly when on a degraded
source).

**Sizing multipliers as shipped**:

```
DATA_QUALITY_SIZE_PRIMARY                = 1.00   # relative-tier
DATA_QUALITY_SIZE_EMPIRICAL              = 1.00   # relative-tier
DATA_QUALITY_SIZE_GAUSSIAN               = 1.00   # relative-tier
DATA_QUALITY_SIZE_GAUSSIAN_DEFAULT_SIGMA = 0.30   # absolute-trustability
DATA_QUALITY_SIZE_COLD_START_SUSPECT     = 0.30   # absolute-trustability
DATA_QUALITY_SIZE_BLOCK                  = 0.00   # don't trade
```

**Why the relative tiers are at 1.00**: There is no PRIMARY tier today
(every city is on Gaussian). A 40% haircut on Gaussian would be a
unilateral capital reduction, not a relative-confidence statement —
"arbitrarily haircutting without a quality reference point doesn't add
information." When a real PRIMARY tier exists (W2 Phase B), the
relative-tier multipliers light up via env var, NO code change.

**Why the absolute-trustability tiers fire NOW**: These are not
relative-confidence statements. They're facts about the inputs:
- `gaussian_default_sigma` (0.30) — city has never been calibrated
- `cold_start_suspect` (0.30) — first scan of the day was past local
  peak hour, so `forecast_high_c` may be evening curve
- `block` (0.00) — catastrophic failure, defense in depth

**Composable flags**: A signal row's `data_quality_flag` can carry
multiple labels comma-separated (e.g.
`gaussian,cold_start_suspect`). The sizing scalar is the MIN over
all components — most conservative wins.

**Wiring**: `compute_data_quality_size_factor()` is called per event,
once `predict_bins()` has run and the flag composition is known. The
factor multiplies the Kelly-derived stake BEFORE the `MIN_STAKE_USD`
floor check. If size_factor = 0, the row becomes a SKIP with reason
`data_quality_blocked`. If size_factor brings the stake below
`MIN_STAKE_USD`, it becomes SKIP with reason `stake_too_small`.

---

## 3. Data being collected

All on the VPS at `~/apps/polymarket-weather/bot/data/` (signals DB)
and `~/apps/weather-data/backtest-collector/data/` (resolutions DB).
SQLite throughout. No external storage, no cloud backups — flagged in
Section 9 as a hygiene gap.

### 3.1 `paper_predictor_signals` — central accumulation table

Live schema: [`_SCHEMA_SQL` in scheduled_predictor.py](../bot/scheduled_predictor.py).

One row per (bin, scan). With 11 US cities × ~11 bins/event ×
1 event/city × ~720 scans/day, the rate is ~87k rows/day. Recovery
helper protects `forecast_high_c` from the ratchet bug from June 12
onward; rows before that date have contaminated values.

Schema highlights:

| Column | Purpose |
|---|---|
| `scanned_at_utc`, `mode`, `city`, `event_date`, `event_id`, `contract_id`, `yes_token_id`, `bin_label` | Identity / join keys |
| `forecast_high_c`, `forecast_peak_hour` | NWS forecast (with recovery protection) — **the residual-calibration input** |
| `observed_max_c`, `observed_peak_hour` | EOD running max from METAR |
| `mu_c`, `sigma_c` | Distribution parameters at scan time |
| `cooling_confidence` | 3-signal weighted cooling score; consumed by `cooling_confidence_monotone_post_peak` guard |
| `our_prob`, `market_prob`, `edge` | Per-bin probabilities and edge |
| `action`, `recommended_stake_usd`, `recommended_limit_price`, `gate_blocked_by` | What the bot decided and why |
| `market_closed` | Gamma per-bin resolution state (drives RESOLVED rendering) |
| `data_quality_flag` | Which CDF path + any absolute-trustability flags (e.g. `gaussian,cold_start_suspect`) |
| `wind_octant`, `upwind_signal_strength`, `settlement_station` | Neighbor / station metadata |

### 3.2 `raw_metar_log` — forensic backup

One row per NWS observation cycle per ICAO, ~100–200 cycles per
ICAO per day. Captures `rawMessage` METAR string + parsed fields
(temp, dewpoint, wind, present_weather).

**What it's for**: When the W0 audit (Section 7.1) eventually flags a
`settle_divergence` tuple — meaning the bot's parsed `observed_max`
disagrees with what the market settled against — the raw METAR strings
around peak hour are needed to diagnose WHY. Possible causes are
distinguishable only from raw obs:
- DSM-aggregation-rule difference (Wunderground uses synoptic windows)
- Missing METAR cycle around peak (we under-counted)
- Unprocessed SPECI report

Idempotent insert via `UNIQUE(icao, cycle_timestamp_utc)`.

### 3.3 `live_predictor_orders` — execution forensics

One row per LIVE_BUY order placement. Carries `order_id`,
`position_id`, `status` (placed / filled / cancelled / stale / error),
the raw CLOB response.

**Reconciliation**: At each scan start, `reconcile_pending_orders()`
walks rows with `status='placed'` and asks the CLOB whether they
filled. Updates statuses accordingly. Orders older than
`RECONCILE_LOOKBACK_HOURS=12` that are still 'placed' get marked
`stale`.

### 3.4 `guard_violations` — invariant monitoring

Written by [bot/scripts/invariant_guards.py](../bot/scripts/invariant_guards.py)
at end of every scan. Pure observational record: scan timestamp, guard
name, city, prev/curr values, delta, detail string. Idempotent via
`UNIQUE(scan_at_utc, guard_name, city)`.

**Currently consumed by**: nobody. The dashboard panel for "per-(city,
guard, is-new)" is specified but unbuilt. Manual querying for now;
fresh-session work item.

### 3.5 `resolutions` — sibling DB, ground truth

Path: `~/apps/weather-data/backtest-collector/data/prices.db`.

Populated by a separate sidecar process (the backtest collector — not
in this repo). Schema:

```sql
CREATE TABLE resolutions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT UNIQUE NOT NULL,
    city            TEXT,
    date            TEXT,
    winning_contract_id TEXT,
    winning_range_low   REAL,
    winning_range_high  REAL,
    resolved_at     TEXT
)
```

**Verified clean of cross-venue contamination** as of June 12 (no
`venue` / `source` column, Polymarket-only). Joins to
`paper_predictor_signals` on `event_id` and `winning_contract_id`.

### 3.6 Supporting DBs

- `bot/data/station_obs.db` — daily-max observations, populated by
  [bot/scripts/station_obs_pull.py](../bot/scripts/station_obs_pull.py)
- `bot/data/neighbor_obs.db` — hourly neighbor ASOS station readings,
  populated by [bot/scripts/neighbor_obs_pull.py](../bot/scripts/neighbor_obs_pull.py),
  consumed by the neighbor signal in the predictor and the
  `mu_jump_incoherent_with_neighbors` invariant guard

---

## 4. Failure modes to remember

These are not hypothetical. Each happened. Each shaped a design rule.

### 4.1 The forecast ratchet bug — June 12

**Symptom**: `forecast_high_c` in the DB was 17.22°C for SF on a 90°F
day. Multiple cities showed exact °F-to-°C conversions of round
numbers (63°F, 70°F) at evening scan times.

**Root cause**: NWS `/forecastHourly` returns only future periods. At
5pm, "today's" remaining periods describe only the evening cooling
curve. `max(remaining_periods)` is the evening cooling temp, not the
day's high.

**Fix**: `recover_persisted_day_forecast()` helper. Look up the
session's max persisted value; prefer it over the buggy fresh fetch.

**Lesson**: A value that should be monotonic-or-stable within a day
silently moved the wrong direction because an assumption about an
external feed's shape was wrong. The fix forward is the recovery
helper; the fix permanent is the invariant guards (Section 2.5), which
will catch the next instance of this bug class automatically.

### 4.2 Position-tracking dust filter overload

**Symptom**: Dashboard showed "5 LIVE positions" when Polymarket only
had 1. Various downstream conflations between "did we fill?", "how
much cost basis?", and "is this market still open?".

**Root cause**: One signal (`LIVE_POS_BY_TOKEN` with a $0.50 dust
filter) used to answer three different questions. Every change to the
filter shifted one query and broke the others.

**Fix**: Three explicit signals — HELD, DEPLOYED, MARKET_OPEN — with
`market_closed` propagated through the Gamma API call chain.

**Lesson**: When one signal serves multiple semantically distinct
questions, the resulting bug surface is multiplicative. Separate
signals at the source.

### 4.3 The June 12 silent regression

**Symptom**: Commit `e19cef0` added new files (the guards module, the
contract spec) but ALSO silently deleted W4 risk-cap constants, the
recovery helper, and `MAX_MARKET_PRICE` from `scheduled_predictor.py`.
The deletion wasn't caught by tests because it manifested as test-
collection failure (the test files couldn't import the missing
symbols), not assertion failure.

**Recovery**: Manual restore from prior conversation state.

**Fix forward**: [bot/scripts/deploy_safety_check.py](../bot/scripts/deploy_safety_check.py)
+ a pre-commit hook (`bot/scripts/git_hooks/pre-commit`, installed via
`bot/scripts/install_hooks.py`). The check asserts that 25 specific
load-bearing names exist in `scheduled_predictor.py`. Verified to
catch this exact regression class.

**Lesson**: Tests are not enough — a commit can silently delete code
that tests reference, and pytest collection failure is easy to miss.
Cheap precondition checks at commit time catch what tests can't.

### 4.4 Open-Meteo calibration source mismatch

**Symptom**: Per-city RMSEs from
[bot/scripts/forecast_rmse_calibration.py](../bot/scripts/forecast_rmse_calibration.py)
came back at 0.5–1.1°C across 11 cities. Typical NWS day-ahead skill
is 1.5–2.5°C. The numbers were "too clean."

**Root cause**: Open-Meteo's `historical-forecast-api` endpoint
appears to serve short-lead-time (near-nowcast) forecasts rather than
day-ahead, AND it's a source mismatch (calibrated against Open-Meteo,
deployed against NWS).

**Status**: W2 Phase C empirical CDF code is shipped but DORMANT —
not activated against this calibration data. The real calibration
comes from 30 days of native NWS forecast-vs-observed pairs in
`paper_predictor_signals`. See Section 7.3.

**Lesson**: A clean number isn't proof of correctness. Verify the
source matches the deployment before trusting calibration output.

### 4.5 Backfill was the wrong instinct

**Almost-failure**: I proposed Open-Meteo backfill to "shorten the
wait." The user correctly identified this as pouring contaminated
calibration data into the column we're trying to calibrate against —
manufacturing the exact problem the contract's "does NOT catch" table
hands to W0.

**Lesson**: When offered a shortcut that trades data integrity for
speed, the right answer is usually to wait. The patience IS the work.

---

## 5. Design rules — non-negotiable

These rules emerged from the failures in Section 4. Future work must
respect them. If you're tempted to break one, re-read the
corresponding failure mode first.

### 5.1 Observational forever (invariant guards)

Guards LOG, COUNT, and SURFACE. They never GATE, REMEDIATE, or affect
the prediction or trading path. Enforced by
`test_no_import_from_prediction_path`. If a guard reveals a problem
worth acting on, the action belongs in the explicit gate stack or
data-quality contract, not buried in a monitoring module.

### 5.2 Fail-down, never up (data-quality contract)

When a source fails its trustability check, fall through to the next
simpler / more defensive source. Never silently substitute one
degraded source for another. Gaussian is the safety floor.

### 5.3 Don't blend market price into edge calc

`our_p` stays independent of `market_p`. Blending and then computing
`edge = our_p_blended - market_p` mechanically shrinks the signal —
you're partly comparing the market to itself. Market-anchored
considerations live in the gate stack (W4) as vetoes, not in the
distribution.

### 5.4 Patience is the work

When offered a shortcut that trades data integrity for speed, wait.
This came up twice — activating empirical CDF on flattering
Open-Meteo residuals, and backfilling from Open-Meteo to shorten the
calendar wait. Both wrong. The edge in these markets is calibration,
and calibration is only as honest as the cleanliness of the residuals
underneath it.

### 5.5 Required names check before deploy

Every commit must pass [bot/scripts/deploy_safety_check.py](../bot/scripts/deploy_safety_check.py).
Pre-commit hook installs via `python bot/scripts/install_hooks.py`.
If you intentionally remove a load-bearing name, update the
`REQUIRED_NAMES` list in the same commit. Don't bypass with
`--no-verify` except in genuine emergencies.

### 5.6 No `forecast_high_c` source change without recovery preservation

The recovery helper assumes `forecast_high_c` is non-decreasing within
a day (the `forecast_high_monotone` guard enforces this). If you ever
swap NWS for a different forecast source, ensure the new source
preserves this property OR update the helper. The bug class is
expensive to debug; don't reopen it.

---

## 6. The 30-day wait — what's accumulating

Starting from the recovery commit (~June 12), `paper_predictor_signals`
is accumulating CLEAN data: `forecast_high_c` no longer ratchets down,
`observed_max_c` end-of-day values are joinable to forecasts for
residual calibration, every row carries `data_quality_flag` so audits
can distinguish provenance.

**Approximate timeline**:
- Recovery commit + safety net live: June 12
- W0 segmentation needs ≥30 resolved markets — likely mid-July at
  current settlement cadence (~2-3 markets settle per city per week,
  depending on Polymarket's posting schedule)
- W2 Phase C residual calibration needs ~30 days of NWS-native
  forecast-vs-observed pairs — also mid-July
- W3 ceiling fit needs ~30 days of (scan, subsequent-rise,
  conditioning-state) data — also mid-July

The accumulation also includes:
- `raw_metar_log` — forensic backup for any future settle_divergence
  investigation
- `guard_violations` — invariant breaks. Even one persistent (city,
  guard) pair across multiple scans is worth investigating.
- `live_predictor_orders` — fill rates and slippage for execution
  cost calibration

---

## 7. The decision tree for after the wait

When ≥30 resolved markets exist in the resolutions DB, START HERE.

### 7.1 Step 1: Run W0 segmentation

Script: [bot/scripts/failure_segmentation.py](../bot/scripts/failure_segmentation.py).

```bash
cd ~/apps/polymarket-weather/bot && source ../venv/bin/activate
python -m scripts.failure_segmentation --days 60 --out audit_60d.csv
```

The script does a three-axis join over resolved markets:
- **Axis A**: bot's `observed_max_c` (what the bot's METAR feed saw)
- **Axis B**: Iowa State ASOS archive max (independent METAR feed)
- **Axis C**: settled bin interval from `resolutions.winning_range_low/high`

And categorizes every settled market into ONE of:

| Category | Meaning | What W1 / next step does |
|---|---|---|
| `bot_caching_gap` | A and B disagree by enough to flip top-P bin | Small caching fix in the bot's fetch path |
| `settle_divergence` | B's bin ≠ C (mesonet agrees with bot but market settled differently) | **Observation→settlement mapping is mis-specified.** Investigate the offending (city, date, market) tuples — see Section 7.2 |
| `boundary_shape_wrong` | B is in C, our top-P was the adjacent bin, our_prob in C < 30% | Distribution shape near edges — W2 Phase C + W3 |
| `magnitude_shape_wrong` | B is in C, our top-P was non-adjacent | σ too tight in wrong direction — W2 Phase C |
| `calibration_overconfident` | We assigned ≥50% to winning bin but bought a different one | Top-P-only buy rule is throwing away signal — revisit basket strategies |

**Output**: CSV with per-tuple data + summary stats + the offending-
tuple list for `settle_divergence` cases.

### 7.2 Step 2: Branch on the dominant category

**IMPORTANT**: Don't read the category percentages alone. Look at the
`settle_divergence` tuple list FIRST. Even a small number of confirmed
settle_divergence tuples reorders W1 from "small caching fix" to
"remap every market's resolution source" — a much bigger workstream.

User's exact framing (preserved verbatim because it matters): "When W0
returns, the thing I'll want to look at first is the settle_divergence
tuples, not the category percentages — even one confirmed tuple
reorders W1, and that existence-check is the highest-information
output of the whole audit."

**Decision tree**:

```
IF settle_divergence > 5% of resolved markets OR ≥3 confirmed tuples:
    → W1 = "remap every market's resolution source"
    → Investigate each (city, date, market) tuple using raw_metar_log
        to distinguish DSM-aggregation vs missing-cycle vs SPECI causes
    → Likely outcome: switch the bot's observation source from raw METAR
        max() to whatever Polymarket actually settles against
    → Defer W2 Phase C / W3 until W1 is resolved

ELIF bot_caching_gap dominates (>20%):
    → W1 = small caching fix in the bot's NWS fetch path
    → Quick (<1 week), then proceed to W2 Phase C / W3

ELIF boundary_shape_wrong dominates:
    → Skip W1. Go to Step 3 (W2 Phase C activation) and Step 4 (W3 fit).
        The distribution shape near edges is the problem.

ELIF magnitude_shape_wrong dominates:
    → Same as boundary — skip W1, do W2 Phase C with focus on σ tuning.

ELIF calibration_overconfident dominates:
    → Revisit top-P-only buy rule. Consider basket / per-bin sizing.
        This is NOT a distribution problem — it's a sizing-policy problem.
```

### 7.3 Step 3: Activate W2 Phase C (empirical residual CDF)

**Precondition**: ≥30 clean NWS-native (forecast, observed) pairs per
city in `paper_predictor_signals` (filtered to dates AFTER the recovery
commit + excluding cold-start days).

**Required work before activation**:

1. **Build a NWS-native calibration** to replace the contaminated
   Open-Meteo one. New script (does not exist yet):
   `bot/scripts/forecast_residuals_from_signals.py`. Joins
   `paper_predictor_signals.forecast_high_c` (filtered to non-cold-
   start, post-recovery rows) with EOD `observed_max_c` per (city,
   event_date). Writes to `bot/data/forecast_calibration.json` with
   the same schema as the Open-Meteo calibration (so the dispatch
   code in `intraday_predictor.py` consumes it unchanged).

2. **Verify the per-city sample counts** before flipping the flag:
   ```bash
   python -c "import json; d=json.load(open('bot/data/forecast_calibration.json')); \
       [print(c, len(v.get('centered_residuals') or [])) for c,v in d['by_city'].items()]"
   ```
   Every city must have ≥30 centered residuals. Below 30, the
   empirical path falls back to gaussian via `EMPIRICAL_MIN_SAMPLES`
   gate.

3. **Backtest BEFORE flipping live**: Use the calibrated residuals to
   replay W0's settled markets. Compute Brier score per (city,
   distribution) pair. The empirical path must show ≥2% Brier
   improvement over gaussian on at least the marine cities (SF, LA,
   Seattle) to justify activation. If it doesn't, the empirical
   distribution isn't fixing what we thought it was — investigate
   further.

4. **Flip the env var**: Set `PREDICTOR_CDF_IMPL=empirical` in
   `.env`. The dispatch in `predict_bins` will use the empirical CDF
   for cities with sufficient samples and fall back to gaussian
   elsewhere. `data_quality_flag` will reflect the path per row.

5. **Paper-shadow week**: Per the contract, the paper-shadow period
   is a REGRESSION check (does it crash, do sizes go absurd, does it
   deadlock the scan), NOT a P&L gate. The P&L gate was the backtest
   in step 3 — don't let one good or bad paper week veto a
   better-calibrated model.

### 7.4 Step 4: W3 implementation (physical ceiling)

**Spec**: [docs/w3_physical_ceiling.md](w3_physical_ceiling.md).

**Precondition**: Enough (scan, subsequent-rise, conditioning-state)
data for the per-regime quantile regression. At ~30 days × ~720
scans/day × 11 cities, that's ~240k data points — comfortably enough
for the hierarchical-pooled fit.

**Required work**:

1. **Persist solar elevation and cloud cover** per signal row. Two
   migrations:
   ```python
   _add_column("paper_predictor_signals", "solar_elevation_deg", "REAL")
   _add_column("paper_predictor_signals", "cloud_cover_pct", "REAL")
   ```
   Both fields are already computed per scan in `bot/solar.py` and
   from NWS forecast `properties.skyCover`; they just need to be
   written to the row. Without this migration, the calibration data
   isn't usable.

2. **Build the calibration script**:
   `bot/scripts/build_physical_ceiling.py`. Per the spec
   (Section 5 of [w3_physical_ceiling.md](w3_physical_ceiling.md)),
   it must:
   - Pool data by climate regime (marine / continental-dry /
     humid-subtropical / northern-temperate)
   - Fit 98th-percentile quantile regression of subsequent_rise
     against (solar_elevation, cloud_cover, dewpoint, hour)
   - Apply per-regime safety buffer (start 0.5°F)
   - **REQUIRED OUTPUT**: regime mis-bucketing diagnostic
     (Section 5.3 of [w3_physical_ceiling.md](w3_physical_ceiling.md)).
     Per-city residual bias against pooled ceiling AND per-city
     exceedance rate of the fitted 98th percentile. Flag any city
     whose own residuals are systematically biased OR whose
     exceedance rate exceeds 5%. Priors: Miami and Denver are likely
     to trip the diagnostic.

3. **Build the estimator**:
   `estimate_remaining_upside_ceiling()` in
   `bot/scripts/intraday_predictor.py`. Takes (current_temp,
   solar_elevation, cloud_cover, dewpoint, city, climate_regime),
   returns a temperature ceiling in °C.

4. **Wire into `predict_bins`**: Pass
   `truncate_at_hi=ceiling_c` to `probability_in_bin()` (the
   parameter is already there — W2 Phase A reserved it for this).

5. **Don't delete the old branches yet**: The wall-clock σ narrowing
   branches in `estimate_day_high_dist` (lines 722-731, 747-765) and
   the binary `bin_lock` override (lines 917-936) stay live. The
   physical ceiling uses `min(physical_ceiling, old_upper_bound)` as
   the effective `truncate_at_hi`. Run in parallel for ≥2 weeks,
   confirm the old branches are inert (count their fire-rate; should
   be near zero), then delete in a SEPARATE PR. **Sequencing rule
   (load-bearing)**: never couple new-model introduction with
   old-safety-net removal.

### 7.5 Step 5: W1 (only if W0 said so)

If Step 2's branch sent us here, follow the investigation steps from
Step 2's "IF settle_divergence" branch. The exact shape depends on
what the offending tuples reveal.

---

## 8. Operational reference

### 8.1 Deploy

```bash
cd ~/apps/polymarket-weather && git pull
sudo systemctl restart weather_bot
```

If the pre-commit hook isn't installed on the VPS:
```bash
python bot/scripts/install_hooks.py
```

### 8.2 Test

```bash
cd ~/apps/polymarket-weather/bot && source ../venv/bin/activate
python -m pytest tests/ -q
```

Current count: **92 tests** across:
- `test_predictor.py` — 41 (estimate_day_high_dist branches, CDF integrator equivalence, probability_in_bin, empirical CDF)
- `test_gates.py` — 8 (gate stack + W4 risk cap)
- `test_forecast_recovery.py` — 7 (recovery helper)
- `test_invariant_guards.py` — 15 (guards + observational-forever architecture)
- `test_data_quality.py` — 15 (cold-start + sizing scalar)
- Plus 6 empirical CDF tests in test_predictor.py

### 8.3 Deploy-safety check

```bash
python bot/scripts/deploy_safety_check.py
```

Should print `PASS (25 required names present in scheduled_predictor.py)`.
Exit code 1 with stderr report if any required name is missing.

### 8.4 Verifying recovery is firing

```bash
sleep 90
sudo journalctl -u weather_bot --since "3 minutes ago" --no-pager | \
    grep -E "forecast_high recovered|liquid_market_strong|intraday_scan done" | tail -10
```

Expected: `forecast_high recovered for {city}` lines for cities whose
morning value beats the current evening fetch. Normal scans complete
in 3–9 seconds.

### 8.5 Checking invariant violations

```bash
sqlite3 ~/apps/polymarket-weather/bot/data/signals.db <<'SQL'
SELECT guard_name, city, COUNT(*) AS n,
       MAX(detected_at_utc) AS latest
FROM guard_violations
WHERE detected_at_utc >= datetime('now', '-7 days')
GROUP BY guard_name, city
ORDER BY n DESC LIMIT 30;
SQL
```

Repeated (city, guard) pairs are worth investigating. A guard that
fires for the same city every day is either telling you about a real
standing issue you've been tolerating, or it's mis-calibrated and
needs its threshold tuned.

### 8.6 Checking forecast_high_c health

After deploying or at any time, run this to confirm the recovery
helper is working AND see the data accumulating:

```bash
sqlite3 ~/apps/polymarket-weather/bot/data/signals.db <<'SQL'
SELECT city, event_date,
       ROUND(MIN(forecast_high_c), 2) AS min_fc,
       ROUND(MAX(forecast_high_c), 2) AS max_fc,
       ROUND(MAX(forecast_high_c) - MIN(forecast_high_c), 2) AS spread_c,
       COUNT(*) AS n_scans
FROM paper_predictor_signals
WHERE event_date >= date('now', '-7 days')
  AND forecast_high_c IS NOT NULL
GROUP BY city, event_date
HAVING spread_c > 3.0
ORDER BY spread_c DESC LIMIT 20;
SQL
```

Any (city, date) where the spread is > 3°C indicates either real
weather change OR a recovery-helper regression. After June 12, spreads
should be small (<1°C typically). A sudden uptick is a smoke signal
worth investigating.

---

## 9. Deliberately deferred

These are known incomplete items, deferred with rationale:

### 9.1 Dashboard panel for guard violations

Spec was per-(city, guard, is-new) breakdown alongside the existing
`data_quality_flag` per-city panel. Designed but not built. The
dashboard file is
[bot/scripts/predictor_dashboard.py](../bot/scripts/predictor_dashboard.py).
The `guard_violations` table is already being populated, so this is
purely a frontend exercise. Half-day of work.

### 9.2 NBM ingestion (W2 Phase B)

Initially planned. Deferred because: (a) Open-Meteo's ensemble
endpoint is a reasonable first proof-of-concept that doesn't require
GRIB2 plumbing, (b) the right trigger is "Open-Meteo's tail
calibration is specifically poor on marine-layer cities AND NBM
percentiles demonstrably tighten those tails" — not a generic Brier
improvement threshold. Reopen once empirical (Phase C) ships and
we have a real comparison baseline.

### 9.3 NBM-specific data-quality triggers in contract

[docs/data_quality_contract.md](data_quality_contract.md) Section
"External ensemble" has trigger thresholds (staleness, member count,
NaN fraction, monotonicity, range plausibility). These are
spec-only; the corresponding code lives in the Phase B PR that
doesn't exist yet.

### 9.4 Off-VPS backups

Currently no cloud backup of `signals.db` or the sibling
`prices.db`. 30+ days of accumulation that's the entire calibration
input shouldn't live on one machine. Nightly `rsync` to S3 or
similar would close this. Not started.

### 9.5 Old wall-clock σ narrowing branches in `estimate_day_high_dist`

Will be deleted as part of W3's follow-up PR (separate from W3's
introduction PR). See Section 7.4 step 5. **Note (2026-06-12 update):**
the HRRR plateau signal is now another trigger for `day_has_likely_peaked`
alongside wall-clock and observed-vs-forecast. Wall-clock branch
stays as fallback for cold-start days and non-CAM cities; deletion
discipline same as before — introduce alongside, don't couple
removal with introduction.

### 9.6 Backfilling pre-June-12 contaminated `forecast_high_c`

NOT going to do this. Per Section 4.5, backfilling from Open-Meteo
or any synthetic source pollutes the calibration data with exactly
the failure mode W0 exists to detect. Pre-June-12 rows stay marked
as contaminated by their date; calibration filters them out.

### 9.7 W3 `solar_elevation_deg` / `cloud_cover_pct` columns

Will be added as part of the W3 implementation PR, not before. See
Section 7.4 step 1.

---

## 10. Open questions for future sessions

### 10.1 What's the right paper-shadow duration for W2 Phase C?

Currently spec'd at "one week, regression check only." Backtest is
the P&L gate. Worth revisiting if the backtest scope shrinks for any
reason.

### 10.2 What if W0 returns inconclusive?

Possible scenarios: not enough resolved markets even at 30 days
(slow settlement schedule), or categories are roughly evenly split.
Probably means we wait longer rather than guess. Worth stating up
front so we don't force a premature decision.

### 10.3 Should the sizing scalar's relative tiers ever go below 1.00?

Today they're all at 1.00 because no PRIMARY tier exists. When one
ships, the question becomes: how big a haircut for empirical vs.
gaussian vs. the new primary? Defer to data — Brier comparison on
real signals.

### 10.4 International °C markets

Mentioned as a possible edge source (thinner books) but deferred
behind W2 Phase C. The `bin_temp_range` rounding semantics need a
unit-aware audit before any live exposure to °C markets. Noted but
not prioritized.

---

## 11. Files index

### Documentation
- [docs/HANDOFF.md](HANDOFF.md) — this file
- [docs/data_quality_contract.md](data_quality_contract.md) — degradation triggers + sizing scalar
- [docs/w3_physical_ceiling.md](w3_physical_ceiling.md) — physical ceiling spec
- [docs/deploy_safety.md](deploy_safety.md) — pre-commit hook setup
- HRRR ceiling spec — Phase 0a (`capture_resolution_truth.py`) and
  Phase 1 (Open-Meteo HRRR/ICON-D2 dispatch behind `PREDICTOR_USE_HRRR_CEILING`)
  are shipped; activation has TWO gates: Phase 2 backtest improvement AND
  Phase 0b confirming the T-group fix closed the observed_max-vs-settlement gap

### Core bot code
- [bot/scheduled_predictor.py](../bot/scheduled_predictor.py) — scan loop, gate stack, recovery helper, cold-start detection, sizing scalar, data-quality flag composition
- [bot/scripts/intraday_predictor.py](../bot/scripts/intraday_predictor.py) — prediction pipeline, CDF integrator, empirical residual CDF, σ calibration loader
- [bot/scripts/invariant_guards.py](../bot/scripts/invariant_guards.py) — observational guards, "observational forever" enforcement
- [bot/polymarket.py](../bot/polymarket.py) — Gamma API market discovery (closed flag propagation)
- [bot/scripts/predictor_dashboard.py](../bot/scripts/predictor_dashboard.py) — dashboard with three-signal position rendering

### Scripts (one-shot tools)
- [bot/scripts/failure_segmentation.py](../bot/scripts/failure_segmentation.py) — W0 audit
- [bot/scripts/forecast_rmse_calibration.py](../bot/scripts/forecast_rmse_calibration.py) — current per-city σ calibration (Open-Meteo-based, to be replaced)
- [bot/scripts/deploy_safety_check.py](../bot/scripts/deploy_safety_check.py) — pre-deploy assertion
- [bot/scripts/install_hooks.py](../bot/scripts/install_hooks.py) — installs the pre-commit hook
- [bot/scripts/git_hooks/pre-commit](../bot/scripts/git_hooks/pre-commit) — hook template

### Tests
- [bot/tests/test_predictor.py](../bot/tests/test_predictor.py) — 41 tests
- [bot/tests/test_gates.py](../bot/tests/test_gates.py) — 8 tests
- [bot/tests/test_forecast_recovery.py](../bot/tests/test_forecast_recovery.py) — 7 tests
- [bot/tests/test_invariant_guards.py](../bot/tests/test_invariant_guards.py) — 15 tests
- [bot/tests/test_data_quality.py](../bot/tests/test_data_quality.py) — 15 tests

---

## 12. Quick context for a fresh Claude session

If you're a Claude instance reading this in a new session:

- **Verify the timeline first**: `ls -lt ~/apps/polymarket-weather/bot/data/signals.db`
  tells you when data last accumulated. If the bot's running, the
  mtime should be within the last 5 minutes.
- **Then check `SELECT MIN(event_date), MAX(event_date) FROM paper_predictor_signals WHERE forecast_high_c IS NOT NULL`**
  to see how much data exists. If `event_date` range is ≥30 days
  from June 12, the wait is over and Section 7 applies.
- **Read Sections 4 (failure modes) and 5 (design rules) before
  proposing any work.** These are the institutional knowledge that
  isn't in the code.
- **The user is technically sharp and patient.** Pressure-testing
  reasoning is welcome; pretending alignment isn't. If the user's
  framing is wrong, push back substantively. If they push back on
  yours, take it seriously.
- **Verify before recommending.** Memories from this session may be
  stale; the code is the source of truth. When in doubt, grep / read
  the file before claiming behavior.