# AGENTS.md

## Mission

Implement the repository exactly as defined in `implementation_spec.md`.

Your job is execution, not research redesign. The intellectual choices have already been made. Do not invent new architecture, new methodology, new evaluation logic, or new artifact contracts unless a real implementation blocker makes the spec impossible to satisfy.

## Absolute guardrails

1. **No fallback code.**
   - No alternate coordinate system if forward estimation fails.
   - No CPU fallback for GPU-required stages.
   - No alternate model family if a dependency is missing.
   - No silent downgrade from calcs files to quote-only files.

2. **Chronology safety is non-negotiable.**
   - No randomized train/validation/test splits.
   - No fitting scalers or model hyperparameters on future data.
   - No feature that uses date `t+1` information to predict date `t+1`.

3. **The source PDF is authoritative for equations.**
   - Use extracted markdown for search/navigation.
   - For any equation, penalty, notation, or table that affects the implementation, open the source PDF and transcribe carefully.
   - This is especially mandatory for the Zhang, Li, and Zhang (2023) arbitrage-penalty equations.

4. **Single supported implementation path.**
   - The canonical workflow is `ivs-forecast run --config ...`.
   - Stage subcommands may exist for debugging, but they must call the same core code, not alternate logic.

5. **Fail fast and explain why.**
   - Every fatal condition must raise a clear exception with a human-readable message.
   - Never catch and suppress a scientific or data-integrity failure.

## Repository conventions

- Use `src/` layout only.
- Put all business logic in importable modules under `src/ivs_forecast/`.
- Keep CLI thin.
- Keep scripts declarative; they must call package functions rather than embedding logic.
- Use explicit type annotations on public functions and classes.
- Prefer small, testable, pure functions for transformations.
- Keep stage outputs immutable once written.

## Dependency rules

Use only the approved stack unless the spec is formally revised:

- `polars`
- `pyarrow`
- `numpy`
- `scipy`
- `pydantic`
- `pyyaml`
- `typer`
- `rich`
- `torch`
- `pytest`
- `ruff`

Do not add:

- `pandas`
- `lightgbm`
- `catboost`
- `tensorflow`
- alternative experiment managers
- notebook-only workflow dependencies

If a new dependency appears necessary, stop and justify it explicitly.

## Python version rules

- Target Python **3.14.3** everywhere.
- Keep `pyproject.toml`, lockfile, CI config, README, and AGENTS consistent.
- Do not lower the version unless a real package blocker is proven.
- If a blocker appears, stop and report the exact dependency, version, and failure mode before changing Python.

## External data-root rules

- Raw files stay outside the repo.
- Never hardcode `D:\Options Data`.
- Read the raw root from config / env / CLI override according to the spec.
- Normalize Windows paths safely.
- Never copy raw ZIP files into the repository.
- Never mutate raw ZIPs.

## Vendor-schema and data-reconciliation rules

You must implement:

- file discovery for `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`;
- recursive discovery of canonical daily files under the configured raw root;
- ZIP readability checks;
- one-CSV-per-ZIP validation;
- documented-vs-observed schema reconciliation;
- machine-readable reconciliation reports;
- `raw_corpus_contract.json`.

You must stop and ask if observed data materially conflict with the design, especially if:

- 15:45 calc fields are absent,
- required columns are renamed in a nontrivial way,
- option files are quote-only rather than calcs-included,
- the selected underlying lacks enough valid data for the supported protocol.

## Chronology and leakage rules

- Forecast origin is **after close on date `t`**.
- Target is **15:45 surface on next valid date `t+1`**.
- Same-day EOD volume/OHLC/VWAP may be used as date-`t` features only because the forecast origin is after close.
- The calibrated daily SSVI surface state is always built from date-`t` 15:45 IVs.
- Contract-level evaluation may query the predicted surface at the realized next-day contract coordinates. That is allowed because it defines the evaluation domain, not the forecast values.

Explicitly test:

- all train dates < all validation dates < all test dates;
- feature windows stop at date `t`;
- no target column leaks into model input;
- target dates equal the immediate next trading dates in the raw/root calendar.

## Modeling rules

Implement only these model families:

- `state_last`
- `state_var1`
- `ssvi_tcn_direct`

Do not add:

- VAE forecaster
- transformer forecaster
- model averaging
- ensemble pooling
- deep RL hedger
- extra baselines

There is no shared reconstructor stage in the supported pipeline.

## CUDA, performance, and memory rules

GPU-required stages:

- `ssvi_tcn_direct`

Rules:

- if CUDA is unavailable for a GPU-required stage, abort;
- do not silently run that stage on CPU;
- do not bulk-load the full raw history into memory;
- stream ZIPs;
- write parquet early;
- batch contract-level evaluation;
- use `float64` for financial transforms and evaluation, `float32` for neural training.

No AMP / mixed precision in v1 unless the spec is revised.

## Artifact-management rules

Every stage must write:

- its main artifact(s),
- a stage manifest,
- resolved config snapshot,
- upstream artifact references / hashes,
- counts and exclusion diagnostics.

Do not overwrite artifacts by default.

If an artifact directory already exists for a run, fail unless overwrite was explicitly requested.

## Documentation rules

For every substantive change you make, update:

- code
- tests
- README if user-facing behavior changed
- artifact contract docs if schemas changed

Never let the implementation drift away from the spec silently.

## Test rules

Minimum required before considering a task complete:

- unit tests for the changed logic;
- integration coverage if the change touches multiple stages;
- smoke run compatibility maintained;
- no broken CLI command paths.

If you touch chronology logic, data cleaning, forward estimation, SSVI calibration, settlement timing, or evaluation metrics, add or update tests immediately.

## When you must stop and ask

Stop and ask if any of the following occurs:

1. The local dataset materially differs from the documented schema.
2. The selected supported underlying is missing or too sparse.
3. Forward estimation fails systematically on the supposedly supported index-underlying workflow.
4. CUDA is unavailable but the user still expects the full benchmark set.
5. A source-PDF equation is unreadable or conflicts with extracted markdown in a way that affects implementation.
6. A required dependency cannot be made to work on Python 3.14.3.
7. A change would require adding a new model family, new data source, or new evaluation protocol.

Do not ask broad open-ended questions. Ask only concrete blocker questions.

## Definition of done

A task is done only when all of the following are true:

- the implementation matches `implementation_spec.md`;
- the canonical `ivs-forecast run --config ...` path works for the targeted stage set;
- required artifacts are written with the expected schema;
- tests pass;
- chronology-safety checks pass;
- no fallback code was introduced;
- README and AGENTS remain accurate.

Anything less is not done.
