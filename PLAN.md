# Implementation Plan: Make IV_Forecast Trustworthy, Reproducible, and Full-Run Ready

## Summary

Yes, code changes are required.

Implement the next phase in gated order. Do not start the full `spx_1d` benchmark until phases 1 through 4 are complete and passing.

1. Restore a self-contained runnable repository.
2. Harden `verify-data` so it proves the real vendor corpus matches the protocol.
3. Add explicit chronology and leakage guards plus failure tests.
4. Correct and validate DM/MCS so research conclusions are defensible.
5. Refactor the heavy contract-level pipeline to stream by date instead of materializing the whole history.
6. Re-run smoke, then full benchmark, then generate the final research summary from the existing CLI path.

## Public Interfaces

- Keep the canonical CLI unchanged:
  - `ivs-forecast verify-data --config ...`
  - `ivs-forecast build-data --config ...`
  - `ivs-forecast run --config ...`
  - `ivs-forecast report --run-dir ...`
- Add a real `src/ivs_forecast/artifacts/` package with hashing and manifest utilities used by the existing pipeline imports.
- Keep `vendor_schema_reconciliation.json` as the machine-readable schema artifact, but expand its schema to include corpus-wide checks, offending filenames, column diffs, dtype summaries, and caveat counts.
- Add `data_audit_report.md` as a human-readable companion artifact for `verify-data`.
- Expand `dm_tests.json` and `mcs_results.json` with method metadata and deterministic outputs.
- Change contract-level intermediate artifacts from single monolithic files to partitioned parquet datasets for scalability.
  - Keep `sampled_surface_wide.parquet` and `features_targets.parquet` as single date-level files.
  - Update README and artifact docs to reflect the partitioned intermediate contract.

## Implementation Changes

### 1. Reproducibility and artifact infrastructure

- Add `artifacts.hashing.sha256_file`.
- Add `artifacts.manifests` helpers for JSON, YAML, parquet, and stage manifests.
- Make every stage write:
  - primary artifacts,
  - stage manifest,
  - stage-specific resolved config snapshot,
  - upstream artifact paths plus SHA256 hashes,
  - counts and exclusion diagnostics.
- Include runtime metadata in manifests:
  - git commit,
  - Python version,
  - installed package versions,
  - CUDA availability and versions,
  - device used per model family,
  - global seed.
- Treat inability to resolve required runtime metadata as fatal, not best-effort.
- Update the integration test to assert the missing artifact package is no longer a runtime blocker.

### 2. Vendor verification and dataset audit

- Expand `verify-data` from “first ZIP + header sample” to a corpus-wide audit over the configured date range.
- For every ZIP:
  - validate readable ZIP,
  - require exactly one CSV,
  - parse filename date,
  - read the `quote_date` column and require all rows to match the filename date,
  - fail with concrete offending filenames on any mismatch.
- Reconcile observed schema against expected v1 columns across the corpus:
  - exact missing columns,
  - extra columns,
  - per-column dtype summaries,
  - null-rate summaries,
  - files with header anomalies.
- Enforce calcs-required semantics explicitly:
  - fail on quote-only or calcs-missing files,
  - fail if required `*_1545` calc fields are absent.
- Add index caveat auditing:
  - report the fraction of rows with zero/missing underlying bid/ask,
  - record whether `active_underlying_price_1545` remains usable for the supported `^SPX` path.
- Add early-close awareness as an audit-only artifact using an in-repo curated half-day date table for the supported study window.
- Keep failures hard and human-readable; do not silently downgrade or skip bad files.

### 3. Chronology and leakage safety

- Add runtime assertions in feature construction and split/refit code:
  - sampled surfaces must be strictly ordered by `quote_date`,
  - `target_date` must be the next available modeling date,
  - no duplicate `quote_date` rows in the sampled-surface panel,
  - training/refit windows must end strictly before the chunk being forecast.
- Keep current expanding-window semantics:
  - validation chunks may use earlier realized validation dates,
  - test chunks may use earlier realized test dates,
  - neither may use same-chunk or future dates.
- Add explicit leak-prevention tests:
  - target-shift test with missing business days,
  - trailing-window test proving MA5/MA22 and return features stop at `t`,
  - feature/target separation test proving no `y_*` columns enter model inputs,
  - validation refit-boundary test,
  - test refit-boundary test,
  - leakage-trap synthetic dataset where only future information is predictive; generated features must remain non-informative while targets still vary.

### 4. DM and MCS correctness

- Replace the current DM implementation with a fully specified one:
  - two-sided test,
  - Newey-West HAC variance with bandwidth 5,
  - HLN finite-sample adjustment,
  - Student-t reference distribution for the adjusted statistic,
  - raw and Holm-adjusted p-values.
- Add hard validation in DM:
  - equal-length loss series,
  - minimum sample size,
  - finite numeric inputs only.
- Re-implement MCS using the actual Hansen-Lunde-Nason elimination logic, not worst-mean-loss removal.
- Support both required variants:
  - `Tmax`,
  - `TR`.
- Keep bootstrap policy fixed by default:
  - stationary bootstrap,
  - 5000 resamples,
  - block length 10,
  - deterministic seed.
- Record in the JSON outputs:
  - method,
  - alpha,
  - bootstrap settings,
  - elimination order,
  - included set,
  - excluded set,
  - p-values by elimination step.

### 5. Scalability refactor

- Remove the full-history “read everything then concat” pattern from the contract-level path.
- Process raw files one day at a time:
  - ingest selected underlying,
  - clean contracts,
  - estimate forward terms,
  - collapse nodes,
  - persist outputs immediately.
- Keep large artifacts partitioned by date/year so forecast and utility evaluation can load only the current and target dates on demand.
- Refactor evaluation code to use a date-to-path index instead of loading full `clean_contracts` and `surface_nodes` tables into memory.
- Keep date-level outputs compact and monolithic:
  - `sampled_surface_wide.parquet`,
  - `features_targets.parquet`,
  - summary evaluation panels.
- Update docs and tests so the artifact contract matches the streaming implementation exactly.

### 6. Benchmark execution and final report

- After phases 1 through 5 pass, run in this order:
  1. clean-clone install check,
  2. `verify-data`,
  3. smoke run,
  4. full `spx_1d` run,
  5. `ivs-forecast report --run-dir ...`.
- Expand the existing run-local `summary.md` so it is research-grade:
  - dataset coverage and caveats,
  - schema audit findings,
  - selected hyperparameters,
  - headline loss metrics,
  - DM and MCS conclusions,
  - arbitrage diagnostics,
  - pricing, hedged-PnL, and straddle summaries.
- Do not make novelty claims in docs or reports until the corrected full-run outputs exist.

## Test Plan and Acceptance Criteria

- Unit tests:
  - artifact hashing and manifest serialization,
  - ZIP/CSV/date verification edge cases,
  - schema reconciliation edge cases,
  - chronology/leakage guards,
  - DM numerical regression fixtures,
  - MCS synthetic-behavior fixtures,
  - fixed-seed MCS reproducibility.
- Integration tests:
  - `verify-data` on synthetic vendor files,
  - `build-data` on synthetic vendor files,
  - `run` and `report` on the smoke config,
  - artifact existence and schema assertions for the new manifests and audit outputs.
- Clean-clone acceptance:
  - `uv sync`
  - `uv run pytest -q`
  - `uv run ruff check`
  - `uv run ivs-forecast --help`
  - `uv run ivs-forecast run --config configs/experiments/spx_smoke.yaml`
- Full acceptance gate:
  - `verify-data` passes on the real corpus with no unresolved schema conflicts,
  - smoke run passes from the canonical CLI,
  - full `spx_1d` run completes without manual edits or OOM,
  - final `summary.md` is generated from the checked-in codebase only.

## Assumptions and Defaults

- No new Python dependencies will be added.
- The supported methodology remains exactly the current v1 scope:
  - `^SPX`,
  - 1-day horizon,
  - `rw_last`,
  - `pca_var1`,
  - `xgb_direct`,
  - `lstm_direct`,
  - shared reconstructor.
- Early-close awareness will be implemented with a curated in-repo date table because adding a trading-calendar dependency is out of scope.
- The existing `report` command remains the only supported reporting path; it will be expanded rather than replaced.
- Missing CUDA for GPU-required stages remains a fatal error.
- Missing git commit/runtime metadata remains a fatal reproducibility error rather than being silently recorded as unknown.
