# IVS Forecast

A reproducible framework for root-explicit, chronology-safe, next-trading-day implied-volatility-surface forecasting on Cboe Option EOD Summary files.

## What this repository does

The repo takes daily `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip` files, filters them to `^SPX` and option root `SPX`, calibrates one arbitrage-aware daily SSVI surface state, forecasts the next trading day’s surface state, and evaluates the resulting surface forecasts on realized next-day nodes and contracts.

The canonical representation is a direct daily SSVI state:

- 11 ATM total-variance knot values at `10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730` days
- 1 global `rho`
- 1 global `eta`
- 1 global `lambda`

Forecasting is done in the latent `state_z` coordinates, and evaluation always decodes those forecasts back to valid constrained SSVI parameters before querying the target-day node and contract coordinates.

## Supported scope

Version 1 is intentionally narrow:

- `underlying_symbol = "^SPX"`
- `option_root = "SPX"`
- 1-trading-day horizon only
- maximum maturity `730` calendar days
- calcs-included Cboe files only

There is no alternate representation path, no quote-only fallback, and no CPU fallback for the deep model stage.

## Method summary

The end-to-end workflow is:

1. Verify ZIP readability, one-CSV-per-ZIP integrity, filename/date agreement, and required `*_1545` calc fields.
2. Audit observed schema against the documented vendor contract and report option-root coverage by date.
3. Discover canonical daily ZIPs recursively under the configured raw root and fail fast if only grouped monthly/yearly archives are present.
4. Ingest `^SPX` rows and fail fast if the configured `option_root` is absent on any date in scope.
5. Clean contracts using the vendor 15:45 snapshot, or 12:45 on manifest-listed early-close dates.
6. Compute fractional ACT/365 time to the `SPX` expiration-date `AM_SOQ_PROXY` settlement proxy.
7. Estimate same-date forward terms by root and expiration using put-call parity.
8. Collapse calls and puts into one node panel per date in log-forward-moneyness and fractional maturity coordinates.
9. Calibrate one SSVI state per date with a deterministic Adam-then-LBFGS schedule and certify static arbitrage on a dense grid.
10. Build a narrow feature index with 22-day chronology-safe history windows, same-day scalar regime/liquidity features, a checked trading-date index, and explicit row exclusions when the immediate next trading day lacks a valid target state.
11. Train and compare exactly three models: `state_last`, `state_var1`, and `ssvi_tcn_direct`.
12. Decode forecasts directly onto realized next-day nodes and contracts for loss, pricing, hedging, straddle, DM, and MCS outputs.

## Model set

The supported model families are:

- `state_last`: tomorrow’s latent SSVI state equals today’s latent SSVI state
- `state_var1`: OLS VAR(1) with intercept on the latent SSVI state vector
- `ssvi_tcn_direct`: a TCN over 22 daily state-plus-scalar vectors, trained with vega-weighted Huber loss on target-day nodes

No other forecasting families are implemented.

## Data expectations

Raw ZIPs stay outside the repo. Configure the root with YAML, the `IVS_FORECAST_RAW_DATA_ROOT` env var, or `--raw-data-root`.

Canonical daily ZIPs may be stored flat or in nested folders under the configured raw root. The runtime does not silently ingest monthly/yearly grouped archives.

Canonical vendor columns include:

- `underlying_symbol`
- `quote_date`
- `root`
- `expiration`
- `strike`
- `option_type`
- `bid_1545`
- `ask_1545`
- `active_underlying_price_1545`
- `implied_volatility_1545`
- `delta_1545`
- `vega_1545`
- `trade_volume`
- `open_interest`

Important vendor notes:

- the `1545` suffix remains even on early-close days
- zero IV rows are invalid for surface modeling
- `root` is treated as the option-class key
- quote-only files are unsupported

Settlement note:

- standard `SPX` expiries are modeled with an explicit `AM_SOQ_PROXY` clock
- the default proxy time is `09:30` ET
- the proxy is versioned in artifacts and is not presented as an official exact settlement timestamp

## Configuration

```yaml
paths:
  raw_data_root: "D:/Options Data"
  artifact_root: "./artifacts"

study:
  underlying_symbol: "^SPX"
  option_root: "SPX"
  start_date: "2012-01-03"
  end_date: "2021-12-31"
  forecast_horizon_days: 1

settlement:
  settlement_style: "AM_SOQ_PROXY"
  proxy_time_eastern: "09:30:00"
  exact_clock: false
```

## Canonical commands

Verify raw files and schema:

```bash
uv run ivs-forecast verify-data --config configs/experiments/spx_public_2012plus.yaml
```

Build the immutable preprocessing artifacts:

```bash
uv run ivs-forecast build-data --config configs/experiments/spx_public_2012plus.yaml
```

Run the full experiment:

```bash
uv run ivs-forecast run --config configs/experiments/spx_public_2012plus.yaml
```

Regenerate the summary report for an existing run:

```bash
uv run ivs-forecast report --run-dir artifacts/runs/<run_id>
```

## Main artifacts

Build stage:

- `raw_inventory.parquet`
- `raw_inventory.json`
- `raw_corpus_contract.json`
- `vendor_schema_reconciliation.json`
- `data_audit_report.md`
- `clean_contracts/`
- `clean_contracts_index.parquet`
- `forward_terms.parquet`
- `surface_nodes/`
- `surface_nodes_index.parquet`
- `surface_date_quality.parquet`
- `ssvi_state.parquet`
- `ssvi_fit_diagnostics.parquet`
- `ssvi_certification.parquet`
- `trading_date_index.parquet`
- `feature_row_exclusions.parquet`
- `settlement_convention.json`
- `features_targets.parquet`

Run stage:

- `split_manifest.json`
- `models/<model_name>/forecast_ssvi_state.parquet`
- `models/<model_name>/forecast_node_panel.parquet`
- `models/<model_name>/forecast_contract_panel.parquet`
- `loss_panel.parquet`
- `forecast_ssvi_certification.parquet`
- `pricing_utility.parquet`
- `hedged_pnl_utility.parquet`
- `straddle_signal_utility.parquet`
- `dm_tests.json`
- `mcs_results.json`
- `summary.md`

Every stage also writes a manifest and resolved-config snapshot under `manifests/`.

## Environment

This project targets Python `3.14.3`.

Install with:

```bash
uv sync
```

The canonical `run` path requires CUDA for `ssvi_tcn_direct`. Verification, build-data, lint, import checks, and the CPU portions of the test suite remain runnable without CUDA.
