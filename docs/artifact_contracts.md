# Artifact Contracts

Every stage writes:

- its primary artifact(s);
- a stage manifest;
- a resolved config snapshot;
- upstream references and hashes;
- exclusion and count diagnostics.

Primary run-level artifacts:

- `raw_inventory.parquet`
- `raw_inventory.json`
- `vendor_schema_reconciliation.json`
- `clean_contracts.parquet`
- `forward_terms.parquet`
- `surface_nodes.parquet`
- `grid_definition.parquet`
- `sampled_surface_wide.parquet`
- `features_targets.parquet`
- `split_manifest.json`
- `scalers.json`
- `reconstructor_model.pt`
- `reconstructor_manifest.json`
- `loss_panel.parquet`
- `arbitrage_panel.parquet`
- `pricing_utility.parquet`
- `hedged_pnl_utility.parquet`
- `straddle_signal_utility.parquet`
- `dm_tests.json`
- `mcs_results.json`
- `summary.md`

Run directories are immutable unless overwrite is explicitly requested.
