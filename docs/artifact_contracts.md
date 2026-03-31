# Artifact Contracts

Every stage writes:

- primary artifact files
- a stage manifest
- a resolved-config snapshot
- upstream hashes
- counts and exclusion diagnostics

## Build stage

Required outputs:

- `raw_inventory.parquet`
- `raw_inventory.json`
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
- `features_targets.parquet`

Schema notes:

- all downstream modeling artifacts carry `option_root`
- `forward_terms.parquet` remains keyed by `quote_date`, `root`, and `expiration`
- `surface_nodes/` partitions preserve root-explicit node coordinates
- `ssvi_state.parquet` stores both constrained SSVI parameters and latent `state_z_*` columns
- `features_targets.parquet` stays narrow and stores row indices rather than lagged state copies

## Run stage

Required outputs:

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

The `models/<model_name>/` directory may also contain:

- `selected_params.json`
- `model_artifact.json`
- `model_checkpoint.pt` for `ssvi_tcn_direct`

## Removed artifacts

These artifacts are not part of the current repo contract:

- `grid_definition.parquet`
- legacy wide fixed-grid surface parquet
- any shared reconstruction-network artifact
- any sampled-grid forecast artifact

Run directories are immutable unless overwrite is explicitly requested.
