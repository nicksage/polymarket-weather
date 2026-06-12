# Deploy safety check

A pre-commit hook that prevents accidental removal of load-bearing code
from `bot/scheduled_predictor.py`.

## Why it exists

On 2026-06-12, a commit silently removed three production features while
adding unrelated new files:

- W4 risk-cap constants and the `liquid_market_strong_disagreement` gate
- The forecast-recovery helper (`recover_persisted_day_forecast`) that
  defends against the NWS hourly evening-scan bug
- `MAX_MARKET_PRICE` constant — left a latent `NameError` in
  `evaluate_gates`

None of the tests caught it because the regression manifested as a
test-collection failure rather than an assertion failure — pytest
couldn't even import the test modules to discover them. The bot ran in
production for ~30 minutes without W4 protection and with the
forecast bug reactivated before the gap was found and manually
restored.

This check encodes the load-bearing names of `scheduled_predictor.py`
as a substring-presence smoke test that runs in <50ms before every
commit. It's not a substitute for unit tests; it's a cheap precondition
that fails fast and obviously when a load-bearing name disappears.

## Install (once per fresh clone)

```bash
python bot/scripts/install_hooks.py
```

That copies the versioned hook from `bot/scripts/git_hooks/pre-commit`
into `.git/hooks/pre-commit` and makes it executable. After install,
every `git commit` runs the check automatically.

Cross-platform — works on Linux, macOS, and Windows (via Git Bash).

## Manual run

```bash
python bot/scripts/deploy_safety_check.py
```

Returns exit code 0 if all required names are present, 1 otherwise
with a detailed report on stderr.

## Adding a required name

Edit `REQUIRED_NAMES` in `bot/scripts/deploy_safety_check.py`. Add the
name in the same commit as the source-file change that introduces it.

## Removing a required name

If a feature is intentionally deprecated, remove the entry from
`REQUIRED_NAMES` in the same commit that removes the source code. The
check fails closed: if you remove the source code but forget to update
the list, the next commit fails until the list catches up. That's the
point.

## Bypass (emergency only)

```bash
git commit --no-verify
```

Don't make a habit of it.

## What's checked

Three categories of names, all from `bot/scheduled_predictor.py`:

| Category | Examples | Why required |
|---|---|---|
| Risk-cap gate (W4) | `MARKET_DISAGREEMENT_LIQ_THRESHOLD`, `liquid_market_strong_disagreement` | Blowup prevention on liquid markets with extreme model-vs-market disagreement |
| Other gate constants | `MIN_MARKET_PROB`, `MAX_MARKET_PRICE`, `MIN_EDGE` | Referenced by `evaluate_gates`; removal causes `NameError` on signals |
| Forecast recovery | `def recover_persisted_day_forecast`, `forecast_high recovered for` | Defends against the NWS hourly evening-scan bug; removal silently corrupts `forecast_high_c` |
| Core entry points | `def run_intraday_scan`, `def evaluate_gates`, `def ensure_schema` | Removing any of these breaks the scan loop |
| Three-signal position model | `market_closed`, `data_quality_flag` | Columns required by the dashboard's HELD / DEPLOYED / MARKET_OPEN rendering |
| Invariant guards | `run_invariant_checks` | Called at scan end; removal silently stops the permanent monitoring |