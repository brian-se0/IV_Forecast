from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from ivs_forecast.artifacts.manifests import (
    write_json,
    write_polars,
    write_stage_bundle,
    write_yaml,
)
from ivs_forecast.config import AppConfig
from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.evaluation.dm import pairwise_dm
from ivs_forecast.evaluation.mcs import run_mcs
from ivs_forecast.models.base import DailyStateStore, history_feature_columns
from ivs_forecast.models.ssvi_tcn_direct import SsviTcnDirectModel
from ivs_forecast.pipeline.build_data import build_data_stage
from ivs_forecast.pipeline.forecast import (
    evaluate_hedged_pnl_utility,
    evaluate_node_forecast,
    evaluate_pricing_utility,
    evaluate_straddle_signal,
    forecast_state_record,
    load_contracts_with_forward_for_date,
    summarize_straddle,
)
from ivs_forecast.pipeline.splits import (
    assert_refit_window_precedes_chunk,
    build_split_manifest,
    label_feature_rows,
    write_split_manifest,
)
from ivs_forecast.pipeline.train_models import instantiate_model, tune_model_family

MODEL_FAMILIES = ["state_last", "state_var1", "ssvi_tcn_direct"]
REPORT_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "loss_panel.parquet",
    "forecast_ssvi_certification.parquet",
    "pricing_utility.parquet",
    "hedged_pnl_utility.parquet",
    "straddle_signal_utility.parquet",
    "feature_row_exclusions.parquet",
    "selected_model_configs.json",
    "manifests/run_resolved_config.yaml",
)


def _partition_by_single_date(frame: pl.DataFrame, column: str) -> dict[object, pl.DataFrame]:
    mapping: dict[object, pl.DataFrame] = {}
    for key, partition in frame.partition_by(column, as_dict=True).items():
        normalized_key = key[0] if isinstance(key, tuple) else key
        mapping[normalized_key] = partition
    return mapping


def _loss_series(loss_panel: pl.DataFrame, metric: str) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for model_name, group in loss_panel.partition_by("model_name", as_dict=True).items():
        normalized_key = model_name[0] if isinstance(model_name, tuple) else model_name
        series[str(normalized_key)] = (
            group.sort("target_date")[metric].to_numpy().astype(np.float64)
        )
    return series


def _frame_or_empty(
    rows: list[dict[str, Any]],
    schema: dict[str, pl.DataType],
    sort_by: list[str],
) -> pl.DataFrame:
    if rows:
        return pl.DataFrame(rows).sort(sort_by)
    return pl.DataFrame(schema=schema)


def _save_model_outputs(
    run_root: Path,
    model_name: str,
    selected_params: dict[str, Any] | None,
    model_artifact: dict[str, Any],
    state_forecasts: list[dict[str, Any]],
    node_forecasts: list[pl.DataFrame],
    contract_forecasts: list[pl.DataFrame],
    checkpoint_payload: dict[str, Any] | None,
) -> None:
    model_root = run_root / "models" / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    write_json(model_root / "selected_params.json", selected_params or {})
    write_json(model_root / "model_artifact.json", model_artifact)
    write_polars(model_root / "forecast_ssvi_state.parquet", pl.DataFrame(state_forecasts))
    if node_forecasts:
        write_polars(
            model_root / "forecast_node_panel.parquet",
            pl.concat(node_forecasts, how="vertical_relaxed"),
        )
    if contract_forecasts:
        write_polars(
            model_root / "forecast_contract_panel.parquet",
            pl.concat(contract_forecasts, how="vertical_relaxed"),
        )
    if checkpoint_payload is not None:
        import torch

        torch.save(checkpoint_payload, model_root / "model_checkpoint.pt")


def _summary_markdown(
    config_dump: dict[str, Any],
    loss_panel: pl.DataFrame,
    certification_panel: pl.DataFrame,
    pricing_utility: pl.DataFrame,
    hedged_utility: pl.DataFrame,
    straddle_rows: pl.DataFrame,
    feature_exclusions: pl.DataFrame,
    selected_models: dict[str, dict[str, Any] | None],
) -> str:
    study = config_dump["study"]
    settlement = config_dump["settlement"]
    lines = ["# Experiment Summary", ""]
    lines.append("## Study")
    lines.append(
        f"- underlying_symbol: `{study['underlying_symbol']}`; option_root: `{study['option_root']}`; representation: `direct_ssvi_state`"
    )
    lines.append(
        "- snapshot policy: forecast origin is after the close on date `t`, using the vendor 15:45 ET surface snapshot, or 12:45 ET on manifest-listed early-close days."
    )
    lines.append(
        "- settlement policy: `SPX` maturities use fractional ACT/365 time from the snapshot to an explicit `AM_SOQ_PROXY` clock. "
        f"The configured proxy is `{settlement['proxy_time_eastern']}` ET with `exact_clock={settlement['exact_clock']}`."
    )
    lines.append("")
    lines.append("## Literature Positioning")
    lines.append(
        "This repo now implements a root-explicit, one-step, supervised, arbitrage-aware next-day IVS forecaster. It does not use the older two-step grid-then-decoder design, and it does not claim a generic nonlinear or generative novelty story."
    )
    lines.append("")
    if not feature_exclusions.is_empty():
        lines.append("## Feature Index")
        for row in (
            feature_exclusions.group_by("exclusion_reason")
            .len(name="count")
            .sort("exclusion_reason")
            .iter_rows(named=True)
        ):
            lines.append(f"- `{row['exclusion_reason']}`: `{int(row['count'])}`")
        lines.append("")
    lines.append("## Selected Models")
    for model_name in MODEL_FAMILIES:
        lines.append(
            f"- `{model_name}`: `{json.dumps(selected_models[model_name], sort_keys=True)}`"
        )
    lines.append("")
    lines.append("## Mean Node Metrics")
    metrics = (
        loss_panel.group_by("model_name")
        .agg(
            pl.col("rmse_iv").mean().alias("rmse_iv"),
            pl.col("vega_rmse_iv").mean().alias("vega_rmse_iv"),
            pl.col("mae_iv").mean().alias("mae_iv"),
        )
        .sort("vega_rmse_iv")
    )
    for row in metrics.iter_rows(named=True):
        lines.append(
            f"- `{row['model_name']}`: vega_rmse_iv={row['vega_rmse_iv']:.6f}, rmse_iv={row['rmse_iv']:.6f}, mae_iv={row['mae_iv']:.6f}"
        )
    lines.append("")
    lines.append("## Forecast Certification")
    summary = certification_panel.group_by("model_name").agg(
        pl.col("calendar_violation_count").mean().alias("calendar_violation_count"),
        pl.col("butterfly_violation_count").mean().alias("butterfly_violation_count"),
    )
    for row in summary.sort("model_name").iter_rows(named=True):
        lines.append(
            f"- `{row['model_name']}`: mean_calendar_violations={row['calendar_violation_count']:.4f}, mean_butterfly_violations={row['butterfly_violation_count']:.4f}"
        )
    if not pricing_utility.is_empty():
        lines.append("")
        lines.append("## Mean Pricing Utility")
        pricing = pricing_utility.group_by("model_name").agg(
            pl.col("price_rmse").mean().alias("price_rmse"),
            pl.col("inside_spread_rate").mean().alias("inside_spread_rate"),
        )
        for row in pricing.iter_rows(named=True):
            lines.append(
                f"- `{row['model_name']}`: price_rmse={row['price_rmse']:.6f}, inside_spread_rate={row['inside_spread_rate']:.4f}"
            )
    if not hedged_utility.is_empty():
        lines.append("")
        lines.append("## Mean Hedged PnL Utility")
        hedged = hedged_utility.group_by(["model_name", "bucket"]).agg(
            pl.col("hedged_pnl_rmse").mean().alias("hedged_pnl_rmse"),
            pl.col("hedged_pnl_mae").mean().alias("hedged_pnl_mae"),
        )
        for row in hedged.iter_rows(named=True):
            lines.append(
                f"- `{row['model_name']}` / `{row['bucket']}`: rmse={row['hedged_pnl_rmse']:.6f}, mae={row['hedged_pnl_mae']:.6f}"
            )
    if not straddle_rows.is_empty():
        lines.append("")
        lines.append("## Straddle Utility")
        straddle_summary = summarize_straddle(straddle_rows)
        for row in straddle_summary.iter_rows(named=True):
            lines.append(
                f"- `{int(row['anchor_days'])}d`: mean_net_return={row['mean_net_return']:.6f}, hit_rate={row['hit_rate']:.4f}, sharpe={row['sharpe_ratio']:.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def _require_report_inputs(run_dir: Path) -> None:
    missing = [
        relative_path
        for relative_path in REPORT_REQUIRED_ARTIFACTS
        if not (run_dir / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run directory is incomplete. `ivs-forecast report` requires a completed "
            f"`ivs-forecast run`. Missing artifacts under {run_dir}: {missing}"
        )


def write_summary_report(run_dir: Path) -> Path:
    _require_report_inputs(run_dir)
    loss_panel = pl.read_parquet(run_dir / "loss_panel.parquet")
    certification_panel = pl.read_parquet(run_dir / "forecast_ssvi_certification.parquet")
    pricing_utility = pl.read_parquet(run_dir / "pricing_utility.parquet")
    hedged_utility = pl.read_parquet(run_dir / "hedged_pnl_utility.parquet")
    straddle_rows = pl.read_parquet(run_dir / "straddle_signal_utility.parquet")
    feature_exclusions = pl.read_parquet(run_dir / "feature_row_exclusions.parquet")
    selected_models = json.loads(
        (run_dir / "selected_model_configs.json").read_text(encoding="utf-8")
    )
    config_dump = yaml.safe_load(
        (run_dir / "manifests" / "run_resolved_config.yaml").read_text(encoding="utf-8")
    )
    summary = _summary_markdown(
        config_dump=config_dump,
        loss_panel=loss_panel,
        certification_panel=certification_panel,
        pricing_utility=pricing_utility,
        hedged_utility=hedged_utility,
        straddle_rows=straddle_rows,
        feature_exclusions=feature_exclusions,
        selected_models=selected_models,
    )
    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    return summary_path


def run_experiment(config: AppConfig) -> Path:
    outputs = build_data_stage(config)
    run_root = outputs["run_root"]
    forward_terms = pl.read_parquet(outputs["forward_terms_path"])
    ssvi_state = pl.read_parquet(outputs["ssvi_state_path"])
    features_targets = pl.read_parquet(outputs["features_targets_path"])
    feature_exclusions = pl.read_parquet(outputs["feature_row_exclusions_path"])
    state_store = DailyStateStore.from_frame(ssvi_state)
    clean_contracts_store = DatePartitionIndex(outputs["clean_contracts_index_path"], "clean_contracts")
    surface_nodes_store = DatePartitionIndex(outputs["surface_nodes_index_path"], "surface_nodes")
    split_manifest = build_split_manifest(features_targets, config)
    labeled_features = label_feature_rows(features_targets, split_manifest)
    write_split_manifest(run_root / "split_manifest.json", split_manifest)
    selected_models: dict[str, dict[str, Any] | None] = {}
    for model_name in MODEL_FAMILIES:
        selected = tune_model_family(
            model_name=model_name,
            labeled_features=labeled_features,
            validation_target_dates=split_manifest.validation_target_dates,
            state_store=state_store,
            surface_nodes_store=surface_nodes_store,
            config=config,
        )
        selected_models[model_name] = selected.params
    write_json(run_root / "selected_model_configs.json", selected_models)
    state_forecasts_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in MODEL_FAMILIES}
    node_forecasts_by_model: dict[str, list[pl.DataFrame]] = {name: [] for name in MODEL_FAMILIES}
    contract_forecasts_by_model: dict[str, list[pl.DataFrame]] = {name: [] for name in MODEL_FAMILIES}
    latest_model_artifacts: dict[str, dict[str, Any]] = {name: {} for name in MODEL_FAMILIES}
    latest_checkpoints: dict[str, dict[str, Any] | None] = {name: None for name in MODEL_FAMILIES}
    loss_rows: list[dict[str, Any]] = []
    certification_rows: list[dict[str, Any]] = []
    pricing_rows: list[dict[str, Any]] = []
    hedged_rows: list[dict[str, Any]] = []
    straddle_rows: list[dict[str, Any]] = []
    target_node_cache: dict[object, pl.DataFrame] = {}
    contract_cache: dict[object, pl.DataFrame] = {}

    def _surface_nodes_for_date(quote_date: object) -> pl.DataFrame:
        nodes = target_node_cache.get(quote_date)
        if nodes is None:
            nodes = surface_nodes_store.load_date(quote_date)
            target_node_cache[quote_date] = nodes
            if len(target_node_cache) > 4:
                oldest = next(iter(target_node_cache))
                del target_node_cache[oldest]
        return nodes

    def _contracts_for_date(quote_date: object) -> pl.DataFrame:
        contracts = contract_cache.get(quote_date)
        if contracts is None:
            contracts = load_contracts_with_forward_for_date(
                clean_contracts_store=clean_contracts_store,
                forward_terms=forward_terms,
                quote_date=quote_date,
            )
            contract_cache[quote_date] = contracts
            if len(contract_cache) > 4:
                oldest = next(iter(contract_cache))
                del contract_cache[oldest]
        return contracts

    for offset in range(0, len(split_manifest.test_target_dates), config.split.refit_frequency):
        chunk_dates = split_manifest.test_target_dates[offset : offset + config.split.refit_frequency]
        chunk_start = chunk_dates[0]
        available_rows = labeled_features.filter(pl.col("target_date") < chunk_start).filter(
            pl.col("split_label") != "discard"
        ).sort("target_date")
        chunk_rows = labeled_features.filter(pl.col("target_date").is_in(chunk_dates)).sort("target_date")
        assert_refit_window_precedes_chunk(available_rows, chunk_rows, "test_refit")
        for model_name in MODEL_FAMILIES:
            model = instantiate_model(model_name, selected_models[model_name], config.runtime.seed)
            model.fit(available_rows, state_store, surface_nodes_store)
            predictions = model.predict(chunk_rows, state_store)
            latest_model_artifacts[model_name] = model.artifact().params
            if isinstance(model, SsviTcnDirectModel):
                latest_checkpoints[model_name] = {
                    "model_name": model.model_name,
                    "params": model.params.to_dict(),
                    "normalization": model.normalization.to_dict() if model.normalization else {},
                    "feature_columns": history_feature_columns(),
                    "state_dict": model.best_state_dict,
                }
            for row, prediction in zip(chunk_rows.iter_rows(named=True), predictions, strict=True):
                option_root = str(row["option_root"])
                state_forecasts_by_model[model_name].append(
                    forecast_state_record(
                        model_name=model_name,
                        quote_date=row["quote_date"],
                        target_date=row["target_date"],
                        option_root=option_root,
                        predicted_state_z=prediction,
                    )
                )
                target_nodes = _surface_nodes_for_date(row["target_date"])
                loss_row, certification_row, node_predictions = evaluate_node_forecast(
                    model_name=model_name,
                    quote_date=row["quote_date"],
                    target_date=row["target_date"],
                    predicted_state_z=prediction,
                    target_nodes=target_nodes,
                )
                loss_rows.append(loss_row)
                certification_rows.append(certification_row)
                node_forecasts_by_model[model_name].append(node_predictions)
                pricing_row, priced_contracts = evaluate_pricing_utility(
                    model_name=model_name,
                    quote_date=row["quote_date"],
                    target_date=row["target_date"],
                    predicted_state_z=prediction,
                    target_contracts=_contracts_for_date(row["target_date"]),
                )
                pricing_rows.append(pricing_row)
                contract_forecasts_by_model[model_name].append(priced_contracts)
                hedged_rows.extend(
                    evaluate_hedged_pnl_utility(
                        model_name=model_name,
                        quote_date=row["quote_date"],
                        target_date=row["target_date"],
                        current_contracts=_contracts_for_date(row["quote_date"]),
                        target_priced_contracts=priced_contracts,
                    )
                )
                current_params = state_store.parameters_by_indices(
                    np.asarray([int(row["surface_state_row_index"])], dtype=np.int64)
                )[0]
                straddle_rows.extend(
                    evaluate_straddle_signal(
                        model_name=model_name,
                        quote_date=row["quote_date"],
                        target_date=row["target_date"],
                        current_contracts=_contracts_for_date(row["quote_date"]),
                        target_contracts=_contracts_for_date(row["target_date"]),
                        current_state_params=current_params,
                        predicted_state_z=prediction,
                    )
                )
    loss_panel = pl.DataFrame(loss_rows).sort(["target_date", "model_name"])
    certification_panel = pl.DataFrame(certification_rows).sort(["target_date", "model_name"])
    pricing_utility = _frame_or_empty(
        pricing_rows,
        {
            "model_name": pl.Utf8,
            "forecast_origin": pl.Date,
            "target_date": pl.Date,
            "option_root": pl.Utf8,
            "price_rmse": pl.Float64,
            "price_mae": pl.Float64,
            "inside_spread_rate": pl.Float64,
        },
        ["target_date", "model_name"],
    )
    hedged_pnl_utility = _frame_or_empty(
        hedged_rows,
        {
            "model_name": pl.Utf8,
            "forecast_origin": pl.Date,
            "target_date": pl.Date,
            "option_root": pl.Utf8,
            "bucket": pl.Utf8,
            "hedged_pnl_rmse": pl.Float64,
            "hedged_pnl_mae": pl.Float64,
        },
        ["target_date", "model_name", "bucket"],
    )
    straddle_signal_utility = _frame_or_empty(
        straddle_rows,
        {
            "model_name": pl.Utf8,
            "forecast_origin": pl.Date,
            "target_date": pl.Date,
            "option_root": pl.Utf8,
            "root": pl.Utf8,
            "anchor_days": pl.Int64,
            "gross_return": pl.Float64,
            "net_return": pl.Float64,
        },
        ["target_date", "model_name", "root", "anchor_days"],
    )
    write_polars(run_root / "loss_panel.parquet", loss_panel)
    write_polars(run_root / "forecast_ssvi_certification.parquet", certification_panel)
    write_polars(run_root / "pricing_utility.parquet", pricing_utility)
    write_polars(run_root / "hedged_pnl_utility.parquet", hedged_pnl_utility)
    write_polars(run_root / "straddle_signal_utility.parquet", straddle_signal_utility)
    dm_results = {
        "method": {
            "alternative": "two_sided",
            "variance_estimator": "newey_west_hac",
            "bandwidth": 5,
            "finite_sample_adjustment": "harvey_leybourne_newbold",
            "reference_distribution": "student_t",
        },
        "metrics": {
            "rmse_iv_sq": [
                item.__dict__
                for item in pairwise_dm(
                    _loss_series(
                        loss_panel.with_columns((pl.col("rmse_iv") ** 2).alias("rmse_iv_sq")),
                        "rmse_iv_sq",
                    )
                )
            ],
            "vega_rmse_iv_sq": [
                item.__dict__
                for item in pairwise_dm(
                    _loss_series(
                        loss_panel.with_columns(
                            (pl.col("vega_rmse_iv") ** 2).alias("vega_rmse_iv_sq")
                        ),
                        "vega_rmse_iv_sq",
                    )
                )
            ],
        },
    }
    write_json(run_root / "dm_tests.json", dm_results)
    mcs_results = {
        "rmse_iv_sq": run_mcs(
            _loss_series(
                loss_panel.with_columns((pl.col("rmse_iv") ** 2).alias("rmse_iv_sq")), "rmse_iv_sq"
            )
        ),
        "vega_rmse_iv_sq": run_mcs(
            _loss_series(
                loss_panel.with_columns((pl.col("vega_rmse_iv") ** 2).alias("vega_rmse_iv_sq")),
                "vega_rmse_iv_sq",
            )
        ),
    }
    write_json(run_root / "mcs_results.json", mcs_results)
    for model_name in MODEL_FAMILIES:
        _save_model_outputs(
            run_root=run_root,
            model_name=model_name,
            selected_params=selected_models[model_name],
            model_artifact=latest_model_artifacts[model_name],
            state_forecasts=state_forecasts_by_model[model_name],
            node_forecasts=node_forecasts_by_model[model_name],
            contract_forecasts=contract_forecasts_by_model[model_name],
            checkpoint_payload=latest_checkpoints[model_name],
        )
    summary = _summary_markdown(
        config_dump=config.model_config_dump(),
        loss_panel=loss_panel,
        certification_panel=certification_panel,
        pricing_utility=pricing_utility,
        hedged_utility=hedged_pnl_utility,
        straddle_rows=straddle_signal_utility,
        feature_exclusions=feature_exclusions,
        selected_models=selected_models,
    )
    summary_path = run_root / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    write_stage_bundle(
        run_root / "manifests",
        "run",
        config.model_config_dump(),
        global_seed=config.runtime.seed,
        device_by_model_family={
            "state_last": "cpu",
            "state_var1": "cpu",
            "ssvi_tcn_direct": "cuda",
        },
        primary_artifact_paths=[
            run_root / "split_manifest.json",
            run_root / "selected_model_configs.json",
            run_root / "loss_panel.parquet",
            run_root / "forecast_ssvi_certification.parquet",
            run_root / "pricing_utility.parquet",
            run_root / "hedged_pnl_utility.parquet",
            run_root / "straddle_signal_utility.parquet",
            run_root / "dm_tests.json",
            run_root / "mcs_results.json",
            summary_path,
        ],
        counts={
            "loss_rows": loss_panel.height,
            "certification_rows": certification_panel.height,
            "pricing_rows": pricing_utility.height,
            "hedged_rows": hedged_pnl_utility.height,
            "straddle_rows": straddle_signal_utility.height,
        },
        diagnostics={"selected_models": selected_models},
        upstream_paths=[
            outputs["clean_contracts_index_path"],
            outputs["forward_terms_path"],
            outputs["surface_nodes_index_path"],
            outputs["ssvi_state_path"],
            outputs["features_targets_path"],
        ],
    )
    write_yaml(run_root / "resolved_config.yaml", config.model_config_dump())
    return run_root
