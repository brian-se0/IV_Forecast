from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from ivs_forecast.artifacts.manifests import write_json, write_polars, write_stage_bundle
from ivs_forecast.config import AppConfig, ensure_run_directory
from ivs_forecast.data.clean import clean_contracts_day
from ivs_forecast.data.collapse_nodes import build_surface_nodes
from ivs_forecast.data.discovery import (
    audit_vendor_corpus,
    data_audit_markdown,
    inventory_raw_files,
    raw_inventory_frame,
    write_inventory_json,
)
from ivs_forecast.data.ingest import stream_ingest_selected_underlying
from ivs_forecast.data.parity_forward import estimate_forward_terms
from ivs_forecast.data.partitioned import (
    DatePartitionRecord,
    partition_index_frame,
    write_date_partition,
)
from ivs_forecast.data.sampled_surface import build_sampled_surfaces
from ivs_forecast.features.dataset import build_features_targets


def _subset_path_quote_date(path: Path) -> date:
    return date.fromisoformat(path.stem)


def verify_data_stage(config: AppConfig) -> dict[str, Path]:
    run_root = ensure_run_directory(config)
    records = inventory_raw_files(
        config.paths.raw_data_root,
        config.study.start_date,
        config.study.end_date,
    )
    inventory = raw_inventory_frame(records)
    inventory_path = run_root / "raw_inventory.parquet"
    inventory_json_path = run_root / "raw_inventory.json"
    schema_report_path = run_root / "vendor_schema_reconciliation.json"
    audit_report_path = run_root / "data_audit_report.md"
    write_polars(inventory_path, inventory)
    write_inventory_json(inventory_json_path, records)
    schema_report = audit_vendor_corpus(
        records=records,
        underlying_symbol=config.study.underlying_symbol,
        start_date=config.study.start_date,
        end_date=config.study.end_date,
    )
    write_json(schema_report_path, schema_report)
    audit_report_path.write_text(data_audit_markdown(schema_report), encoding="utf-8")
    write_stage_bundle(
        run_root / "manifests",
        "verify_data",
        config.model_config_dump(),
        global_seed=config.runtime.seed,
        primary_artifact_paths=[
            inventory_path,
            inventory_json_path,
            schema_report_path,
            audit_report_path,
        ],
        counts={"raw_zip_count": len(records)},
        diagnostics={
            "schema_pass_status": int(bool(schema_report["pass_status"])),
            "header_anomaly_count": int(schema_report["caveat_counts"]["header_anomaly_count"]),
            "early_close_count": int(schema_report["caveat_counts"]["early_close_count"]),
        },
        upstream_paths=[],
    )
    return {
        "run_root": run_root,
        "inventory_path": inventory_path,
        "schema_report_path": schema_report_path,
        "audit_report_path": audit_report_path,
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
    clean_contracts_root = run_root / "clean_contracts"
    surface_nodes_root = run_root / "surface_nodes"
    clean_partition_records: list[DatePartitionRecord] = []
    surface_partition_records: list[DatePartitionRecord] = []
    forward_frames: list[pl.DataFrame] = []
    date_quality_frames: list[pl.DataFrame] = []
    sampled_surface_frames: list[pl.DataFrame] = []
    forward_invalid_count = 0
    exact_duplicates_removed = 0
    clean_contract_rows = 0
    surface_node_rows = 0
    clean_dates_after_cleaning = 0
    grid_definition: pl.DataFrame | None = None
    for subset_path in sorted(subset_paths):
        subset_frame = pl.read_parquet(subset_path)
        quote_date = (
            subset_frame["quote_date"][0] if subset_frame.height > 0 else _subset_path_quote_date(subset_path)
        )
        cleaned_day, duplicate_summary = clean_contracts_day(subset_frame, config)
        exact_duplicates_removed += duplicate_summary.exact_duplicates_removed
        if cleaned_day.is_empty():
            continue
        clean_dates_after_cleaning += 1
        clean_contract_rows += cleaned_day.height
        clean_partition_records.append(
            write_date_partition(clean_contracts_root, quote_date, cleaned_day)
        )
        forward_day, forward_day_diagnostics = estimate_forward_terms(cleaned_day)
        forward_invalid_count += sum(item.invalid_reason is not None for item in forward_day_diagnostics)
        if forward_day.is_empty():
            continue
        forward_frames.append(forward_day)
        nodes_day, date_quality_day = build_surface_nodes(cleaned_day, forward_day, config)
        if not date_quality_day.is_empty():
            date_quality_frames.append(date_quality_day)
        if nodes_day.is_empty():
            continue
        surface_node_rows += nodes_day.height
        surface_partition_records.append(
            write_date_partition(surface_nodes_root, quote_date, nodes_day)
        )
        grid_day, sampled_surface_day = build_sampled_surfaces(nodes_day)
        if grid_definition is None:
            grid_definition = grid_day
        elif not grid_definition.equals(grid_day):
            raise ValueError("Grid definition changed across dates, which is not supported.")
        sampled_surface_frames.append(sampled_surface_day)
    clean_contracts_index_path = run_root / "clean_contracts_index.parquet"
    surface_nodes_index_path = run_root / "surface_nodes_index.parquet"
    write_polars(clean_contracts_index_path, partition_index_frame(clean_partition_records))
    write_polars(surface_nodes_index_path, partition_index_frame(surface_partition_records))
    if not forward_frames:
        raise ValueError("Forward estimation failed for all expiries in the selected date range.")
    forward_terms = pl.concat(forward_frames, how="vertical_relaxed").sort(["quote_date", "expiration"])
    if forward_terms.is_empty():
        raise ValueError("Forward estimation failed for all expiries in the selected date range.")
    forward_terms_path = run_root / "forward_terms.parquet"
    write_polars(forward_terms_path, forward_terms)
    date_quality = (
        pl.concat(date_quality_frames, how="vertical_relaxed").sort("quote_date")
        if date_quality_frames
        else pl.DataFrame(
            schema={
                "quote_date": pl.Date,
                "surface_node_count": pl.Int64,
                "valid_expiry_count": pl.Int64,
                "modeling_valid": pl.Boolean,
            }
        )
    )
    if not surface_partition_records:
        raise ValueError("No valid modeling dates remained after node construction.")
    write_polars(run_root / "surface_date_quality.parquet", date_quality)
    grid_definition_path = run_root / "grid_definition.parquet"
    sampled_surface_path = run_root / "sampled_surface_wide.parquet"
    if grid_definition is None:
        raise ValueError("No grid definition could be produced from the valid modeling dates.")
    sampled_surface = pl.concat(sampled_surface_frames, how="vertical_relaxed").sort("quote_date")
    write_polars(grid_definition_path, grid_definition)
    write_polars(sampled_surface_path, sampled_surface)
    features_targets = build_features_targets(sampled_surface)
    features_targets_path = run_root / "features_targets.parquet"
    write_polars(features_targets_path, features_targets)
    diagnostics = {
        "clean_contracts_rows": clean_contract_rows,
        "forward_rows": forward_terms.height,
        "surface_node_rows": surface_node_rows,
        "valid_dates": sampled_surface.height,
        "forward_invalid_count": forward_invalid_count,
    }
    write_stage_bundle(
        run_root / "manifests",
        "build_data",
        config.model_config_dump(),
        global_seed=config.runtime.seed,
        primary_artifact_paths=[
            clean_contracts_index_path,
            forward_terms_path,
            surface_nodes_index_path,
            run_root / "surface_date_quality.parquet",
            grid_definition_path,
            sampled_surface_path,
            features_targets_path,
        ],
        counts={
            "inventory_rows": inventory.height,
            "subset_files": len(subset_paths),
            "clean_contract_partitions": len(clean_partition_records),
            "surface_node_partitions": len(surface_partition_records),
            "feature_rows": features_targets.height,
        },
        diagnostics={
            **diagnostics,
            "rows_after_cleaning": clean_contract_rows,
            "exact_duplicates_removed": exact_duplicates_removed,
            "dates_after_cleaning": clean_dates_after_cleaning,
        },
        upstream_paths=[
            verify_outputs["inventory_path"],
            verify_outputs["schema_report_path"],
            verify_outputs["audit_report_path"],
        ],
    )
    return {
        "run_root": run_root,
        "clean_contracts_root": clean_contracts_root,
        "clean_contracts_index_path": clean_contracts_index_path,
        "forward_terms_path": forward_terms_path,
        "surface_nodes_root": surface_nodes_root,
        "surface_nodes_index_path": surface_nodes_index_path,
        "grid_definition_path": grid_definition_path,
        "sampled_surface_path": sampled_surface_path,
        "features_targets_path": features_targets_path,
    }
