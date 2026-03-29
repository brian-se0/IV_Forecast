from __future__ import annotations

from pathlib import Path

import polars as pl

from ivs_forecast.artifacts.manifests import write_json, write_polars, write_stage_bundle
from ivs_forecast.config import AppConfig, ensure_run_directory
from ivs_forecast.data.clean import clean_contracts_files
from ivs_forecast.data.collapse_nodes import build_surface_nodes
from ivs_forecast.data.discovery import (
    header_schema_report,
    inventory_raw_files,
    raw_inventory_frame,
    write_inventory_json,
)
from ivs_forecast.data.ingest import stream_ingest_selected_underlying
from ivs_forecast.data.parity_forward import estimate_forward_terms
from ivs_forecast.data.sampled_surface import build_sampled_surfaces
from ivs_forecast.features.dataset import build_features_targets


def verify_data_stage(config: AppConfig) -> dict[str, Path]:
    run_root = ensure_run_directory(config)
    records = inventory_raw_files(
        config.paths.raw_data_root,
        config.study.start_date,
        config.study.end_date,
    )
    if not all(record.readable and record.csv_member_count == 1 for record in records):
        invalid = [
            record.path for record in records if not record.readable or record.csv_member_count != 1
        ]
        raise ValueError(f"Raw ZIP validation failed for files including: {invalid[:5]}")
    inventory = raw_inventory_frame(records)
    inventory_path = run_root / "raw_inventory.parquet"
    write_polars(inventory_path, inventory)
    write_inventory_json(run_root / "raw_inventory.json", records)
    schema_report = header_schema_report(records[0].path)
    if not schema_report["pass_status"]:
        raise ValueError(
            f"Vendor schema reconciliation failed: {schema_report['missing_required_columns']}"
        )
    write_json(run_root / "vendor_schema_reconciliation.json", schema_report)
    write_stage_bundle(
        run_root / "manifests",
        "verify_data",
        config.model_config_dump(),
        counts={"raw_zip_count": len(records)},
        diagnostics={"schema_pass_status": int(bool(schema_report["pass_status"]))},
        upstream_paths=[],
    )
    return {
        "run_root": run_root,
        "inventory_path": inventory_path,
        "schema_report_path": run_root / "vendor_schema_reconciliation.json",
    }


def build_data_stage(config: AppConfig) -> dict[str, Path]:
    verify_outputs = verify_data_stage(config)
    run_root = verify_outputs["run_root"]
    inventory = pl.read_parquet(verify_outputs["inventory_path"])
    raw_records = inventory_raw_files(
        config.paths.raw_data_root,
        config.study.start_date,
        config.study.end_date,
    )
    subset_paths = stream_ingest_selected_underlying(config, raw_records)
    clean_contracts, clean_diagnostics = clean_contracts_files(config, subset_paths)
    clean_contracts_path = run_root / "clean_contracts.parquet"
    write_polars(clean_contracts_path, clean_contracts)
    forward_terms, forward_diagnostics = estimate_forward_terms(clean_contracts)
    if forward_terms.is_empty():
        raise ValueError("Forward estimation failed for all expiries in the selected date range.")
    forward_terms_path = run_root / "forward_terms.parquet"
    write_polars(forward_terms_path, forward_terms)
    surface_nodes, date_quality = build_surface_nodes(clean_contracts, forward_terms, config)
    if surface_nodes.is_empty():
        raise ValueError("No valid modeling dates remained after node construction.")
    surface_nodes_path = run_root / "surface_nodes.parquet"
    write_polars(surface_nodes_path, surface_nodes)
    write_polars(run_root / "surface_date_quality.parquet", date_quality)
    grid_definition, sampled_surface = build_sampled_surfaces(surface_nodes)
    grid_definition_path = run_root / "grid_definition.parquet"
    sampled_surface_path = run_root / "sampled_surface_wide.parquet"
    write_polars(grid_definition_path, grid_definition)
    write_polars(sampled_surface_path, sampled_surface)
    features_targets = build_features_targets(sampled_surface)
    features_targets_path = run_root / "features_targets.parquet"
    write_polars(features_targets_path, features_targets)
    diagnostics = {
        "clean_contracts_rows": clean_contracts.height,
        "forward_rows": forward_terms.height,
        "surface_node_rows": surface_nodes.height,
        "valid_dates": sampled_surface.height,
        "forward_invalid_count": sum(
            item.invalid_reason is not None for item in forward_diagnostics
        ),
    }
    write_stage_bundle(
        run_root / "manifests",
        "build_data",
        config.model_config_dump(),
        counts={
            "inventory_rows": inventory.height,
            "subset_files": len(subset_paths),
            "feature_rows": features_targets.height,
        },
        diagnostics={**diagnostics, **clean_diagnostics},
        upstream_paths=[
            verify_outputs["inventory_path"],
            verify_outputs["schema_report_path"],
        ],
    )
    return {
        "run_root": run_root,
        "clean_contracts_path": clean_contracts_path,
        "forward_terms_path": forward_terms_path,
        "surface_nodes_path": surface_nodes_path,
        "grid_definition_path": grid_definition_path,
        "sampled_surface_path": sampled_surface_path,
        "features_targets_path": features_targets_path,
    }
