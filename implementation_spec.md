# Implementation specification

This document is the repository-level source of truth for the current implementation.

## 1. Scientific scope

Version 1 supports exactly one validated protocol:

- underlying symbol: `^SPX`
- option root: `SPX`
- canonical benchmark window: `2004-01-02` through `2021-04-09`
- horizon: one next valid trading day
- target timestamp: next-day 15:45 ET surface
- raw data source: `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`
- maximum supported maturity: `730` calendar days

There is no parallel legacy method, no alternate state representation, and no quote-only fallback path.

## 2. Core representation

The canonical daily state is a direct SSVI surface state in total-variance space:

`w(k, τ) = 0.5 * θ(τ) * [1 + ρ * φ(θ(τ)) * k + sqrt((φ(θ(τ)) * k + ρ)^2 + 1 - ρ^2)]`

with

`φ(θ) = η * θ^{-λ}`

and constraints:

- `θ(τ) > 0`
- `θ(τ)` monotone nondecreasing in `τ`
- `ρ ∈ (-1, 1)`
- `η > 0`
- `λ ∈ [0, 0.5]`

The 11 ATM total-variance knots are anchored at:

- `10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730` days

The daily constrained state therefore has 14 dimensions:

- `theta_d010 ... theta_d730`
- `rho`
- `eta`
- `lambda`

The canonical forecast target is the latent 14-vector:

- `state_z_000 ... state_z_010 = log(theta_knot)`
- `state_z_011 = atanh(rho)`
- `state_z_012 = log(eta)`
- `state_z_013 = logit(2 * lambda)`

All models forecast `state_z`. All evaluation paths decode that latent vector back to constrained SSVI parameters before querying the surface.

## 3. Forecast origin and chronology

Forecasts are made after the close on date `t`.

Allowed date-`t` inputs include:

- the vendor 15:45 IV and Greeks snapshot
- same-day end-of-day volume, OHLC, VWAP, and open-interest summaries

The target is the realized surface on the next trading date `t+1` in the raw/root inventory.

Chronology rules:

- train dates < validation dates < test dates
- target dates are always the next trading dates in the raw/root inventory
- history windows stop at date `t`
- normalization is fit only on rows available at the current refit point
- no target-day columns may enter origin-day features

## 4. Root and time-to-settlement policy

`root` is a first-class contract attribute. The implementation must filter to `option_root = "SPX"` before any modeling artifact is written.

Time-to-settlement is fractional ACT/365, not integer-day `dte / 365`.

Rules:

- normal-day snapshot timestamp: `15:45` ET
- manifest-listed early-close snapshot timestamp: `12:45` ET
- `SPX` settlement uses an explicit `AM_SOQ_PROXY` policy with a versioned proxy clock and `exact_clock = false`

If a post-filter date contains zero rows for the configured option root, preprocessing fails.

## 5. Raw-data and audit contract

Supported raw files:

- `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`

Discovery rules:

- recursive discovery of canonical daily files under the configured raw root is supported
- flat and nested daily directory layouts are supported
- monthly/yearly grouped archives must be detected explicitly and reported in the corpus contract

Fatal conditions:

- unreadable ZIP
- zero CSV members
- multiple CSV members
- filename/date mismatch
- missing required columns
- missing calc-required `*_1545` fields
- quote-only files without calcs files in scope

The verification stage must write:

- `raw_inventory.parquet`
- `raw_inventory.json`
- `raw_corpus_contract.json`
- `vendor_schema_reconciliation.json`
- `data_audit_report.md`

The reconciliation report must include root coverage by date for the selected underlying and configured option root.
The canonical benchmark must fail if the requested study window does not exactly match the observed raw daily-ZIP window.

## 6. Pre-modeling backbone

The preprocessing backbone is:

1. ingest filtered contracts
2. clean valid contracts
3. estimate parity-implied forward terms by `quote_date`, `root`, and `expiration`
4. collapse call/put observations into node panels
5. calibrate one daily SSVI state per valid date
6. certify static arbitrage on a dense grid
7. build a narrow feature index over the ordered daily state panel

These artifacts are immutable once written:

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
- `stage_loss_by_date.parquet`
- `stage_coverage_by_year.json`
- `forward_invalid_reasons.json`
- `benchmark_contract.json`

## 7. SSVI calibration

Daily calibration uses target-day node IVs and target-day node vegas.

Required optimizer schedule:

- 50 Adam steps
- then 75 LBFGS steps

Required calibration behavior:

- initialize `theta` from near-ATM nodes by maturity
- make initial `theta` monotone by cumulative maximum
- warm-start each day from the prior day’s fitted latent state
- use one deterministic schedule for every date

Certification must report at least:

- max negative calendar residual
- max negative butterfly residual
- calendar violation count
- butterfly violation count

## 8. Feature panel

`features_targets.parquet` is intentionally narrow. It must include:

- `quote_date`
- `target_date`
- `option_root`
- `history_start_index`
- `history_end_index`
- `surface_state_row_index`
- `target_state_row_index`
- origin-day scalar features

Required scalar features:

- underlying log return `1d`, `5d`, `22d`
- log total trade volume
- log total open interest
- median relative spread
- surface fit RMSE
- node count
- valid-expiry count

The full state history is assembled at training time from `ssvi_state.parquet`, not materialized into the parquet feature index.

`trading_date_index.parquet` must prove that each kept feature row targets the immediate next trading date in the raw/root inventory. If a trading date is missing a valid origin or target SSVI state, the origin row must be dropped and recorded in `feature_row_exclusions.parquet`; the pipeline must not skip forward to a later target date.

## 9. Model set

Exactly three model families are supported:

- `state_last`
- `state_var1`
- `ssvi_tcn_direct`

Definitions:

- `state_last`: tomorrow’s latent state equals today’s latent state
- `state_var1`: OLS VAR(1) with intercept on the latent state vector only
- `ssvi_tcn_direct`: a TCN over daily `[state_z | scalar_features]` vectors

The TCN contract is:

- input history length: tune over `[10, 22]`
- projection width: tune over `[32, 64]`
- dropout: tune over `[0.0, 0.1]`
- four residual causal conv blocks
- kernel size `3`
- dilations `[1, 2, 4, 8]`
- GELU activations
- LayerNorm after projection and inside each block
- MLP head to 14 latent outputs

Training loop:

- optimizer: AdamW
- learning rate: `3e-4`
- weight decay: `1e-4`
- max epochs: `150`
- patience: `15`
- gradient clipping: `1.0`
- batch size: `64`

Primary training loss:

- vega-weighted Huber loss on target-day nodes after decoding the predicted latent state onto the realized target-day node coordinates

There is no parameter-vector MSE objective as the primary training target.

## 10. Run-stage outputs

The canonical command is:

```bash
ivs-forecast run --config ...
```

The run stage must write:

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
- `artifact_contract_version.json`
- `bundle_manifest.json`

The summary must print:

- `underlying_symbol`
- `option_root`
- the direct SSVI representation family
- a short literature-positioning paragraph
- the root/snapshot policy

## 11. Evaluation

All models must be evaluated on the same forecast dates and target domains.

Node-level metrics:

- `rmse_iv`
- `mae_iv`
- `mape_iv_clipped`
- `vega_rmse_iv`

Downstream diagnostics:

- mark-to-mid pricing utility
- delta-hedged marking-error utility
- stylized ATM straddle utility

Formal comparisons:

- Diebold-Mariano tests
- Model Confidence Set

## 12. Non-goals

Version 1 does not implement:

- quote-only vendor files
- alternative state representations
- additional forecasting families
- multi-underlying pooled training
- reinforcement-learning policies
- ensemble averaging
- alternate evaluation protocols
- CPU fallback for the CUDA-required TCN run path
