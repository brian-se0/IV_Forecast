from __future__ import annotations

import json
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
    raw_corpus_contract,
    raw_inventory_frame,
    require_exact_window_coverage,
    write_inventory_json,
)
from ivs_forecast.data.ingest import stream_ingest_selected_underlying
from ivs_forecast.data.parity_forward import estimate_forward_terms
from ivs_forecast.data.partitioned import (
    DatePartitionRecord,
    partition_index_frame,
    write_date_partition,
)
from ivs_forecast.data.ssvi import (
    SsviCalibrationConfig,
    calibrate_daily_ssvi,
    static_arb_certification,
)
from ivs_forecast.data.time_to_settlement import settlement_policy_record
from ivs_forecast.features.dataset import (
    build_features_targets,
    build_trading_date_index,
    state_z_columns,
)
from ivs_forecast.features.scalars import add_state_scalar_features
from ivs_forecast.models.base import state_parameter_columns
from ivs_forecast.reporting.data_quality import (
    build_benchmark_contract,
    build_forward_invalid_reasons_summary,
    build_stage_coverage_by_year,
    build_stage_loss_by_date,
    initialize_daily_build_diagnostics,
    summarize_forward_diagnostics,
)


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
    raw_corpus_contract_path = run_root / "raw_corpus_contract.json"
    schema_report_path = run_root / "vendor_schema_reconciliation.json"
    audit_report_path = run_root / "data_audit_report.md"
    write_polars(inventory_path, inventory)
    write_inventory_json(inventory_json_path, records)
    corpus_contract = raw_corpus_contract(
        root=config.paths.raw_data_root,
        records=records,
        start_date=config.study.start_date,
        end_date=config.study.end_date,
        option_root=config.study.option_root,
    )
    write_json(
        raw_corpus_contract_path,
        corpus_contract,
    )
    schema_report = audit_vendor_corpus(
        records=records,
        underlying_symbol=config.study.underlying_symbol,
        option_root=config.study.option_root,
        start_date=config.study.start_date,
        end_date=config.study.end_date,
    )
    write_json(schema_report_path, schema_report)
    audit_report_path.write_text(data_audit_markdown(schema_report), encoding="utf-8")
    if config.runtime.require_exact_window_coverage:
        require_exact_window_coverage(corpus_contract["window_coverage"])
    write_stage_bundle(
        run_root / "manifests",
        "verify_data",
        config.model_config_dump(),
        global_seed=config.runtime.seed,
        primary_artifact_paths=[
            inventory_path,
            inventory_json_path,
            raw_corpus_contract_path,
            schema_report_path,
            audit_report_path,
        ],
        counts={"raw_zip_count": len(records)},
        diagnostics={
            "schema_pass_status": int(bool(schema_report["pass_status"])),
            "header_anomaly_count": int(schema_report["caveat_counts"]["header_anomaly_count"]),
            "early_close_count": int(schema_report["caveat_counts"]["early_close_count"]),
            "dates_missing_option_root": len(schema_report["selected_root_coverage"]["dates_missing_option_root"]),
            "exact_window_coverage": int(bool(corpus_contract["window_coverage"]["matches_requested_window"])),
        },
        upstream_paths=[],
    )
    return {
        "run_root": run_root,
        "inventory_path": inventory_path,
        "raw_corpus_contract_path": raw_corpus_contract_path,
        "schema_report_path": schema_report_path,
        "audit_report_path": audit_report_path,
    }


def _state_artifact_row(
    quote_date: date,
    option_root: str,
    nodes_day: pl.DataFrame,
    fit_rmse_iv: float,
    fit_vega_rmse_iv: float,
    raw_state: list[float],
    constrained_params: list[float],
) -> dict[str, object]:
    summary = nodes_day.select(
        pl.col("active_underlying_price_1545").mean().alias("active_underlying_price_1545"),
        pl.col("trade_volume").sum().alias("total_trade_volume"),
        pl.col("open_interest").sum().alias("total_open_interest"),
        pl.col("median_rel_spread_1545").median().alias("median_rel_spread_1545"),
        pl.len().alias("surface_node_count"),
        pl.col("expiration").n_unique().alias("valid_expiry_count"),
    ).row(0, named=True)
    row: dict[str, object] = {
        "quote_date": quote_date,
        "option_root": option_root,
        **summary,
        "surface_fit_rmse_iv": fit_rmse_iv,
        "surface_fit_vega_rmse_iv": fit_vega_rmse_iv,
    }
    for index, column in enumerate(state_z_columns()):
        row[column] = float(raw_state[index])
    for index, column in enumerate(state_parameter_columns()):
        row[column] = float(constrained_params[index])
    return row


def build_data_stage(config: AppConfig) -> dict[str, Path]:
    verify_outputs = verify_data_stage(config)
    run_root = verify_outputs["run_root"]
    inventory = pl.read_parquet(verify_outputs["inventory_path"])
    settlement_convention_path = run_root / "settlement_convention.json"
    write_json(
        settlement_convention_path,
        settlement_policy_record(config.study.option_root, config.settlement),
    )
    raw_records = inventory_raw_files(
        config.paths.raw_data_root,
        config.study.start_date,
        config.study.end_date,
    )
    subset_paths = stream_ingest_selected_underlying(config, raw_records)
    trading_dates = [_subset_path_quote_date(path) for path in sorted(subset_paths)]
    minimum_history_days = 22
    daily_diagnostics = initialize_daily_build_diagnostics(trading_dates, config.study.option_root)
    clean_contracts_root = run_root / "clean_contracts"
    surface_nodes_root = run_root / "surface_nodes"
    clean_partition_records: list[DatePartitionRecord] = []
    surface_partition_records: list[DatePartitionRecord] = []
    forward_frames: list[pl.DataFrame] = []
    all_forward_diagnostics = []
    date_quality_frames: list[pl.DataFrame] = []
    ssvi_state_rows: list[dict[str, object]] = []
    ssvi_fit_rows: list[dict[str, object]] = []
    ssvi_certification_rows: list[dict[str, object]] = []
    forward_invalid_count = 0
    exact_duplicates_removed = 0
    clean_contract_rows = 0
    surface_node_rows = 0
    calibration_config = SsviCalibrationConfig()
    previous_raw_state: list[float] | None = None
    for subset_path in sorted(subset_paths):
        subset_frame = pl.read_parquet(subset_path)
        quote_date = (
            subset_frame["quote_date"][0] if subset_frame.height > 0 else _subset_path_quote_date(subset_path)
        )
        day_diagnostics = daily_diagnostics[quote_date]
        day_diagnostics.subset_rows = subset_frame.height
        cleaned_day, duplicate_summary = clean_contracts_day(subset_frame, config)
        exact_duplicates_removed += duplicate_summary.exact_duplicates_removed
        day_diagnostics.clean_contract_rows = cleaned_day.height
        if cleaned_day.is_empty():
            continue
        clean_contract_rows += cleaned_day.height
        clean_partition_records.append(write_date_partition(clean_contracts_root, quote_date, cleaned_day))
        forward_day, forward_day_diagnostics = estimate_forward_terms(cleaned_day)
        (
            day_diagnostics.forward_total_expiries,
            day_diagnostics.forward_valid_expiries,
            day_diagnostics.forward_invalid_expiries,
            day_diagnostics.forward_invalid_reason_codes,
        ) = summarize_forward_diagnostics(forward_day_diagnostics)
        day_diagnostics.forward_term_rows = forward_day.height
        all_forward_diagnostics.extend(forward_day_diagnostics)
        forward_invalid_count += sum(item.invalid_reason is not None for item in forward_day_diagnostics)
        if forward_day.is_empty():
            continue
        forward_frames.append(forward_day)
        nodes_day, date_quality_day = build_surface_nodes(cleaned_day, forward_day, config)
        if not date_quality_day.is_empty():
            quality_row = date_quality_day.row(0, named=True)
            day_diagnostics.surface_node_count = int(quality_row["surface_node_count"])
            day_diagnostics.valid_expiry_count = int(quality_row["valid_expiry_count"])
            day_diagnostics.root_count = int(quality_row["root_count"])
            day_diagnostics.modeling_valid = bool(quality_row["modeling_valid"])
            date_quality_frames.append(date_quality_day)
        if nodes_day.is_empty():
            continue
        surface_node_rows += nodes_day.height
        surface_partition_records.append(write_date_partition(surface_nodes_root, quote_date, nodes_day))
        calibration = calibrate_daily_ssvi(nodes_day, previous_raw_state, calibration_config)
        certification = static_arb_certification(
            calibration.constrained_params,
            tol=calibration_config.certification_tolerance,
        )
        if not certification["passes_static_arb"]:
            raise ValueError(
                "Static-arbitrage certification failed for calibrated SSVI state on "
                f"{quote_date.isoformat()}."
            )
        previous_raw_state = calibration.raw_state.tolist()
        ssvi_state_rows.append(
            _state_artifact_row(
                quote_date=quote_date,
                option_root=config.study.option_root,
                nodes_day=nodes_day,
                fit_rmse_iv=calibration.fit_rmse_iv,
                fit_vega_rmse_iv=calibration.fit_vega_rmse_iv,
                raw_state=calibration.raw_state.tolist(),
                constrained_params=calibration.constrained_params.tolist(),
            )
        )
        ssvi_fit_rows.append(
            {
                "quote_date": quote_date,
                "option_root": config.study.option_root,
                "fit_rmse_iv": calibration.fit_rmse_iv,
                "fit_vega_rmse_iv": calibration.fit_vega_rmse_iv,
                "fit_mae_iv": calibration.fit_mae_iv,
                "final_loss": calibration.final_loss,
                "warm_start_used": calibration.warm_start_used,
                "node_count": calibration.node_count,
            }
        )
        ssvi_certification_rows.append(
            {
                "quote_date": quote_date,
                "option_root": config.study.option_root,
                **certification,
            }
        )
        day_diagnostics.has_surface_state = True
    clean_contracts_index_path = run_root / "clean_contracts_index.parquet"
    surface_nodes_index_path = run_root / "surface_nodes_index.parquet"
    write_polars(clean_contracts_index_path, partition_index_frame(clean_partition_records))
    write_polars(surface_nodes_index_path, partition_index_frame(surface_partition_records))
    if not forward_frames:
        raise ValueError("Forward estimation failed for all expiries in the selected date range.")
    forward_terms = pl.concat(forward_frames, how="vertical_relaxed").sort(
        ["quote_date", "root", "expiration"]
    )
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
                "root_count": pl.Int64,
                "modeling_valid": pl.Boolean,
            }
        )
    )
    if not surface_partition_records:
        raise ValueError("No valid modeling dates remained after node construction.")
    write_polars(run_root / "surface_date_quality.parquet", date_quality)
    if not ssvi_state_rows:
        raise ValueError("No daily SSVI states were calibrated for the configured date range.")
    ssvi_state = (
        pl.DataFrame(ssvi_state_rows)
        .sort("quote_date")
        .with_row_index(name="state_row_index")
    )
    ssvi_state = add_state_scalar_features(ssvi_state)
    ssvi_fit_diagnostics = pl.DataFrame(ssvi_fit_rows).sort("quote_date")
    ssvi_certification = pl.DataFrame(ssvi_certification_rows).sort("quote_date")
    ssvi_state_path = run_root / "ssvi_state.parquet"
    ssvi_fit_path = run_root / "ssvi_fit_diagnostics.parquet"
    ssvi_certification_path = run_root / "ssvi_certification.parquet"
    write_polars(ssvi_state_path, ssvi_state)
    write_polars(ssvi_fit_path, ssvi_fit_diagnostics)
    write_polars(ssvi_certification_path, ssvi_certification)
    trading_date_index = build_trading_date_index(
        trading_dates=trading_dates,
        option_root=config.study.option_root,
        ssvi_state=ssvi_state,
    )
    trading_date_index_path = run_root / "trading_date_index.parquet"
    write_polars(trading_date_index_path, trading_date_index)
    feature_artifacts = build_features_targets(
        ssvi_state,
        trading_date_index,
        minimum_history_days=minimum_history_days,
    )
    feature_row_exclusions_path = run_root / "feature_row_exclusions.parquet"
    write_polars(feature_row_exclusions_path, feature_artifacts.exclusions)
    features_targets = feature_artifacts.features_targets
    features_targets_path = run_root / "features_targets.parquet"
    write_polars(features_targets_path, features_targets)
    stage_loss_by_date = build_stage_loss_by_date(
        trading_date_index=trading_date_index,
        daily_diagnostics=daily_diagnostics,
        features_targets=features_targets,
        feature_exclusions=feature_artifacts.exclusions,
        min_surface_nodes=config.study.min_surface_nodes,
        min_valid_expiries=config.study.min_valid_expiries,
    )
    stage_loss_by_date_path = run_root / "stage_loss_by_date.parquet"
    write_polars(stage_loss_by_date_path, stage_loss_by_date)
    stage_coverage_by_year = build_stage_coverage_by_year(stage_loss_by_date)
    stage_coverage_by_year_path = run_root / "stage_coverage_by_year.json"
    write_json(stage_coverage_by_year_path, stage_coverage_by_year)
    forward_invalid_reasons_path = run_root / "forward_invalid_reasons.json"
    write_json(
        forward_invalid_reasons_path,
        build_forward_invalid_reasons_summary(all_forward_diagnostics),
    )
    benchmark_contract_path = run_root / "benchmark_contract.json"
    raw_corpus_contract = json.loads(
        verify_outputs["raw_corpus_contract_path"].read_text(encoding="utf-8")
    )
    benchmark_contract = build_benchmark_contract(
        raw_corpus_contract=raw_corpus_contract,
        trading_date_index=trading_date_index,
        ssvi_state=ssvi_state,
        features_targets=features_targets,
        feature_exclusions=feature_artifacts.exclusions,
        minimum_history_days=minimum_history_days,
    )
    write_json(benchmark_contract_path, benchmark_contract)
    exclusion_counts = (
        {
            str(row["exclusion_reason"]): int(row["count"])
            for row in feature_artifacts.exclusions.group_by("exclusion_reason")
            .len(name="count")
            .iter_rows(named=True)
        }
        if not feature_artifacts.exclusions.is_empty()
        else {}
    )
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
            ssvi_state_path,
            ssvi_fit_path,
            ssvi_certification_path,
            trading_date_index_path,
            feature_row_exclusions_path,
            settlement_convention_path,
            features_targets_path,
            stage_loss_by_date_path,
            stage_coverage_by_year_path,
            forward_invalid_reasons_path,
            benchmark_contract_path,
        ],
        counts={
            "inventory_rows": inventory.height,
            "subset_files": len(subset_paths),
            "clean_contract_partitions": len(clean_partition_records),
            "surface_node_partitions": len(surface_partition_records),
            "ssvi_state_rows": ssvi_state.height,
            "trading_dates": trading_date_index.height,
            "feature_rows": features_targets.height,
            "feature_row_exclusions": feature_artifacts.exclusions.height,
            "stage_loss_rows": stage_loss_by_date.height,
        },
        diagnostics={
            "clean_contracts_rows": clean_contract_rows,
            "forward_rows": forward_terms.height,
            "surface_node_rows": surface_node_rows,
            "forward_invalid_count": forward_invalid_count,
            "exact_duplicates_removed": exact_duplicates_removed,
            "feature_row_exclusion_counts": exclusion_counts,
            "first_ssvi_state_date": benchmark_contract["forecastable_window"]["ssvi_state_window"]["start_date"],
            "first_feature_origin_date": benchmark_contract["forecastable_window"]["feature_origin_window"]["start_date"],
            "first_feature_target_date": benchmark_contract["forecastable_window"]["feature_target_window"]["start_date"],
        },
        upstream_paths=[
            verify_outputs["inventory_path"],
            verify_outputs["raw_corpus_contract_path"],
            verify_outputs["schema_report_path"],
            verify_outputs["audit_report_path"],
        ],
    )
    return {
        "run_root": run_root,
        "raw_corpus_contract_path": verify_outputs["raw_corpus_contract_path"],
        "clean_contracts_root": clean_contracts_root,
        "clean_contracts_index_path": clean_contracts_index_path,
        "forward_terms_path": forward_terms_path,
        "surface_nodes_root": surface_nodes_root,
        "surface_nodes_index_path": surface_nodes_index_path,
        "ssvi_state_path": ssvi_state_path,
        "ssvi_fit_path": ssvi_fit_path,
        "ssvi_certification_path": ssvi_certification_path,
        "trading_date_index_path": trading_date_index_path,
        "feature_row_exclusions_path": feature_row_exclusions_path,
        "settlement_convention_path": settlement_convention_path,
        "features_targets_path": features_targets_path,
        "stage_loss_by_date_path": stage_loss_by_date_path,
        "stage_coverage_by_year_path": stage_coverage_by_year_path,
        "forward_invalid_reasons_path": forward_invalid_reasons_path,
        "benchmark_contract_path": benchmark_contract_path,
    }
