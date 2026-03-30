from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

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
from ivs_forecast.models.reconstructor import FittedReconstructor, train_reconstructor
from ivs_forecast.pipeline.build_data import build_data_stage
from ivs_forecast.pipeline.forecast import (
    evaluate_hedged_pnl_utility,
    evaluate_node_forecast,
    evaluate_pricing_utility,
    evaluate_straddle_signal,
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

MODEL_FAMILIES = ["rw_last", "pca_var1", "xgb_direct", "lstm_direct"]


def _atm_grid_map(grid_definition: pl.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for anchor in (30, 91):
        row = grid_definition.filter(
            (pl.col("moneyness_ratio") == 1.0) & (pl.col("maturity_days") == anchor)
        )
        if row.height != 1:
            raise ValueError(f"Could not locate ATM grid point for anchor {anchor} days.")
        mapping[anchor] = row["grid_id"][0]
    return mapping


def _partition_by_single_date(frame: pl.DataFrame, column: str) -> dict[object, pl.DataFrame]:
    mapping: dict[object, pl.DataFrame] = {}
    for key, partition in frame.partition_by(column, as_dict=True).items():
        normalized_key = key[0] if isinstance(key, tuple) else key
        mapping[normalized_key] = partition
    return mapping


def _save_model_outputs(
    run_root: Path,
    model_name: str,
    selected_params: dict[str, Any] | None,
    sampled_forecasts: list[dict[str, Any]],
    node_forecasts: list[pl.DataFrame],
) -> None:
    model_root = run_root / "models" / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    write_json(model_root / "selected_params.json", selected_params or {})
    write_polars(model_root / "sampled_forecasts.parquet", pl.DataFrame(sampled_forecasts))
    if node_forecasts:
        write_polars(
            model_root / "node_forecasts.parquet", pl.concat(node_forecasts, how="vertical_relaxed")
        )


def _loss_series(loss_panel: pl.DataFrame, metric: str) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for model_name, group in loss_panel.partition_by("model_name", as_dict=True).items():
        normalized_key = model_name[0] if isinstance(model_name, tuple) else model_name
        series[str(normalized_key)] = (
            group.sort("target_date")[metric].to_numpy().astype(np.float64)
        )
    return series


def _summary_markdown(
    loss_panel: pl.DataFrame,
    pricing_utility: pl.DataFrame,
    hedged_utility: pl.DataFrame,
    straddle_rows: pl.DataFrame,
    selected_models: dict[str, dict[str, Any] | None],
) -> str:
    lines = ["# Experiment Summary", ""]
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
        .sort("rmse_iv")
    )
    for row in metrics.iter_rows(named=True):
        lines.append(
            f"- `{row['model_name']}`: rmse_iv={row['rmse_iv']:.6f}, "
            f"vega_rmse_iv={row['vega_rmse_iv']:.6f}, mae_iv={row['mae_iv']:.6f}"
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
                f"- `{row['model_name']}`: price_rmse={row['price_rmse']:.6f}, "
                f"inside_spread_rate={row['inside_spread_rate']:.4f}"
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
                f"- `{row['model_name']}` / `{row['bucket']}`: "
                f"rmse={row['hedged_pnl_rmse']:.6f}, mae={row['hedged_pnl_mae']:.6f}"
            )
    if not straddle_rows.is_empty():
        lines.append("")
        lines.append("## Straddle Utility")
        summary = summarize_straddle(straddle_rows)
        for row in summary.iter_rows(named=True):
            lines.append(
                f"- `{int(row['anchor_days'])}d`: mean_net_return={row['mean_net_return']:.6f}, "
                f"hit_rate={row['hit_rate']:.4f}, sharpe={row['sharpe_ratio']:.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def _frame_or_empty(
    rows: list[dict[str, Any]], columns: list[str], sort_by: list[str]
) -> pl.DataFrame:
    if rows:
        return pl.DataFrame(rows).sort(sort_by)
    return pl.DataFrame(schema={column: pl.Utf8 for column in columns})


def write_summary_report(run_dir: Path) -> Path:
    loss_panel = pl.read_parquet(run_dir / "loss_panel.parquet")
    pricing_utility = pl.read_parquet(run_dir / "pricing_utility.parquet")
    hedged_utility = pl.read_parquet(run_dir / "hedged_pnl_utility.parquet")
    straddle_rows = pl.read_parquet(run_dir / "straddle_signal_utility.parquet")
    selected_models = json.loads(
        (run_dir / "selected_model_configs.json").read_text(encoding="utf-8")
    )
    summary = _summary_markdown(
        loss_panel, pricing_utility, hedged_utility, straddle_rows, selected_models
    )
    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    return summary_path


def run_experiment(config: AppConfig) -> Path:
    outputs = build_data_stage(config)
    run_root = outputs["run_root"]
    forward_terms = pl.read_parquet(outputs["forward_terms_path"])
    grid_definition = pl.read_parquet(outputs["grid_definition_path"])
    sampled_surface_wide = pl.read_parquet(outputs["sampled_surface_path"])
    features_targets = pl.read_parquet(outputs["features_targets_path"])
    clean_contracts_store = DatePartitionIndex(outputs["clean_contracts_index_path"], "clean_contracts")
    surface_nodes_store = DatePartitionIndex(outputs["surface_nodes_index_path"], "surface_nodes")
    split_manifest = build_split_manifest(features_targets, config)
    labeled_features = label_feature_rows(features_targets, split_manifest)
    write_split_manifest(run_root / "split_manifest.json", split_manifest)
    write_json(run_root / "scalers.json", {"scalers": "not_used_in_v1"})
    selected_models: dict[str, dict[str, Any] | None] = {}
    for model_name in MODEL_FAMILIES:
        selected = tune_model_family(
            model_name=model_name,
            labeled_features=labeled_features,
            validation_target_dates=split_manifest.validation_target_dates,
            sampled_surface_wide=sampled_surface_wide,
            surface_nodes_store=surface_nodes_store,
            config=config,
        )
        selected_models[model_name] = selected.params
    write_json(run_root / "selected_model_configs.json", selected_models)
    sampled_surface_by_date = _partition_by_single_date(sampled_surface_wide, "quote_date")
    atm_grid_map = _atm_grid_map(grid_definition)
    loss_rows: list[dict[str, Any]] = []
    arbitrage_rows: list[dict[str, Any]] = []
    pricing_rows: list[dict[str, Any]] = []
    hedged_rows: list[dict[str, Any]] = []
    straddle_rows: list[dict[str, Any]] = []
    sampled_forecasts_by_model: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_FAMILIES
    }
    node_forecasts_by_model: dict[str, list[pl.DataFrame]] = {name: [] for name in MODEL_FAMILIES}
    latest_reconstructor: FittedReconstructor | None = None
    latest_reconstructor_dates: list[str] = []
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
        chunk_dates = split_manifest.test_target_dates[
            offset : offset + config.split.refit_frequency
        ]
        chunk_start = chunk_dates[0]
        available_rows = labeled_features.filter(pl.col("target_date") < chunk_start).filter(
            pl.col("split_label") != "discard"
        ).sort("target_date")
        available_dates = sampled_surface_wide.filter(pl.col("quote_date") < chunk_start)[
            "quote_date"
        ].to_list()
        chunk_rows = labeled_features.filter(pl.col("target_date").is_in(chunk_dates)).sort(
            "target_date"
        )
        assert_refit_window_precedes_chunk(available_rows, chunk_rows, "test_refit")
        if not available_dates:
            raise ValueError(
                f"Reconstructor chronology violation: no sampled surfaces are available prior to {chunk_start}."
            )
        if max(available_dates) >= chunk_start:
            raise ValueError(
                "Reconstructor chronology violation: sampled surfaces for test refitting are not "
                f"strictly earlier than chunk start {chunk_start}."
            )
        reconstructor = train_reconstructor(
            sampled_surface_wide=sampled_surface_wide,
            surface_nodes=surface_nodes_store.load_many(available_dates),
            available_dates=available_dates,
            config_path=Path(config.models.reconstructor_config),
            seed=config.runtime.seed,
        )
        latest_reconstructor = reconstructor
        latest_reconstructor_dates = [str(value) for value in available_dates]
        for model_name in MODEL_FAMILIES:
            model = instantiate_model(model_name, selected_models[model_name], config.runtime.seed)
            model.fit(available_rows)
            predictions = model.predict(chunk_rows)
            for row, pred in zip(chunk_rows.iter_rows(named=True), predictions, strict=True):
                pred_iv = np.exp(pred)
                sampled_row: dict[str, Any] = {
                    "model_name": model_name,
                    "forecast_origin": row["quote_date"],
                    "target_date": row["target_date"],
                }
                for grid_index, value in enumerate(pred):
                    sampled_row[f"pred_logiv_g{grid_index:03d}"] = float(value)
                    sampled_row[f"pred_iv_g{grid_index:03d}"] = float(pred_iv[grid_index])
                sampled_forecasts_by_model[model_name].append(sampled_row)
                node_loss_row, arbitrage_row, node_predictions = evaluate_node_forecast(
                    model_name=model_name,
                    quote_date=row["quote_date"],
                    target_date=row["target_date"],
                    predicted_sampled_iv=pred_iv,
                    reconstructor=reconstructor,
                    target_nodes=_surface_nodes_for_date(row["target_date"]),
                )
                loss_rows.append(node_loss_row)
                arbitrage_rows.append(arbitrage_row)
                node_forecasts_by_model[model_name].append(node_predictions)
                pricing_row, priced_contracts = evaluate_pricing_utility(
                    model_name=model_name,
                    quote_date=row["quote_date"],
                    target_date=row["target_date"],
                    predicted_sampled_iv=pred_iv,
                    reconstructor=reconstructor,
                    target_contracts=_contracts_for_date(row["target_date"]),
                )
                pricing_rows.append(pricing_row)
                hedged_rows.extend(
                    evaluate_hedged_pnl_utility(
                        model_name=model_name,
                        quote_date=row["quote_date"],
                        target_date=row["target_date"],
                        current_contracts=_contracts_for_date(row["quote_date"]),
                        target_priced_contracts=priced_contracts,
                    )
                )
                straddle_rows.extend(
                    evaluate_straddle_signal(
                        model_name=model_name,
                        quote_date=row["quote_date"],
                        target_date=row["target_date"],
                        current_contracts=_contracts_for_date(row["quote_date"]),
                        target_contracts=_contracts_for_date(row["target_date"]),
                        predicted_sampled_iv=pred_iv,
                        current_sampled_surface=sampled_surface_by_date[row["quote_date"]],
                        reconstructor=reconstructor,
                        atm_grid_map=atm_grid_map,
                    )
                )
    loss_panel = pl.DataFrame(loss_rows).sort(["target_date", "model_name"])
    arbitrage_panel = pl.DataFrame(arbitrage_rows).sort(["target_date", "model_name"])
    pricing_utility = _frame_or_empty(
        pricing_rows,
        [
            "model_name",
            "forecast_origin",
            "target_date",
            "price_rmse",
            "price_mae",
            "inside_spread_rate",
        ],
        ["target_date", "model_name"],
    )
    hedged_pnl_utility = _frame_or_empty(
        hedged_rows,
        [
            "model_name",
            "forecast_origin",
            "target_date",
            "bucket",
            "hedged_pnl_rmse",
            "hedged_pnl_mae",
        ],
        ["target_date", "model_name", "bucket"],
    )
    straddle_signal_utility = _frame_or_empty(
        straddle_rows,
        [
            "model_name",
            "forecast_origin",
            "target_date",
            "root",
            "anchor_days",
            "gross_return",
            "net_return",
        ],
        ["target_date", "model_name", "root", "anchor_days"],
    )
    write_polars(run_root / "loss_panel.parquet", loss_panel)
    write_polars(run_root / "arbitrage_panel.parquet", arbitrage_panel)
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
    if latest_reconstructor is None:
        raise RuntimeError("No reconstructor was trained during the test pass.")
    torch.save(latest_reconstructor.network.state_dict(), run_root / "reconstructor_model.pt")
    write_json(
        run_root / "reconstructor_manifest.json",
        {
            "training_dates": latest_reconstructor_dates,
            "architecture": {
                "input_dim": 156,
                "hidden_widths": [256, 256, 128],
                "activation": "SiLU",
                "output_activation": "Softplus",
            },
        },
    )
    for model_name in MODEL_FAMILIES:
        _save_model_outputs(
            run_root=run_root,
            model_name=model_name,
            selected_params=selected_models[model_name],
            sampled_forecasts=sampled_forecasts_by_model[model_name],
            node_forecasts=node_forecasts_by_model[model_name],
        )
    summary = _summary_markdown(
        loss_panel=loss_panel,
        pricing_utility=pricing_utility,
        hedged_utility=hedged_pnl_utility,
        straddle_rows=straddle_signal_utility,
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
            "rw_last": "cpu",
            "pca_var1": "cpu",
            "reconstructor": "cuda",
            "xgb_direct": "cuda",
            "lstm_direct": "cuda",
        },
        primary_artifact_paths=[
            run_root / "split_manifest.json",
            run_root / "selected_model_configs.json",
            run_root / "loss_panel.parquet",
            run_root / "arbitrage_panel.parquet",
            run_root / "pricing_utility.parquet",
            run_root / "hedged_pnl_utility.parquet",
            run_root / "straddle_signal_utility.parquet",
            run_root / "dm_tests.json",
            run_root / "mcs_results.json",
            run_root / "reconstructor_model.pt",
            run_root / "reconstructor_manifest.json",
            summary_path,
        ],
        counts={
            "loss_rows": loss_panel.height,
            "pricing_rows": pricing_utility.height,
            "hedged_rows": hedged_pnl_utility.height,
            "straddle_rows": straddle_signal_utility.height,
        },
        diagnostics={"selected_models": selected_models},
        upstream_paths=[
            outputs["clean_contracts_index_path"],
            outputs["forward_terms_path"],
            outputs["surface_nodes_index_path"],
            outputs["sampled_surface_path"],
            outputs["features_targets_path"],
        ],
    )
    write_yaml(run_root / "resolved_config.yaml", config.model_config_dump())
    return run_root
