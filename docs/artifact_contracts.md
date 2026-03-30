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
- `data_audit_report.md`
- `clean_contracts/` partitioned parquet dataset
- `clean_contracts_index.parquet`
- `forward_terms.parquet`
- `surface_nodes/` partitioned parquet dataset
- `surface_nodes_index.parquet`
- `surface_date_quality.parquet`
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

Each stage also writes `manifests/<stage_name>_manifest.json` plus
`manifests/<stage_name>_resolved_config.yaml`.

Run directories are immutable unless overwrite is explicitly requested.

Schema notes:

- `forward_terms.parquet` carries `quote_date`, `root`, and `expiration` keys for root-safe parity fits.
- `surface_nodes/` partitions carry `root` so node construction and downstream evaluation cannot silently mix contract classes.
- `straddle_signal_utility.parquet` records the root used for each selected straddle pair.
