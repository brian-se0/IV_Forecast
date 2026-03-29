from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ivs_forecast.config import AppConfig
from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.models.base import SampledSurfaceModel
from ivs_forecast.models.lstm_direct import LstmDirectModel, LstmDirectParams
from ivs_forecast.models.lstm_direct import load_search_space as load_lstm_candidates
from ivs_forecast.models.pca_var1 import PcaVar1Model
from ivs_forecast.models.reconstructor import train_reconstructor
from ivs_forecast.models.rw_last import RwLastModel
from ivs_forecast.models.xgb_direct import XgbDirectModel, XgbDirectParams
from ivs_forecast.models.xgb_direct import load_search_space as load_xgb_candidates
from ivs_forecast.pipeline.forecast import evaluate_node_forecast
from ivs_forecast.pipeline.splits import assert_refit_window_precedes_chunk


@dataclass(frozen=True)
class SelectedModelConfig:
    model_name: str
    params: dict[str, Any] | None


@dataclass(frozen=True)
class CandidateEvaluation:
    params: dict[str, Any]
    mean_rmse_iv: float
    mean_vega_rmse_iv: float
    mean_violating_fraction: float


def instantiate_model(
    model_name: str, params: dict[str, Any] | None, seed: int
) -> SampledSurfaceModel:
    if model_name == "rw_last":
        return RwLastModel()
    if model_name == "pca_var1":
        return PcaVar1Model()
    if model_name == "xgb_direct":
        if params is None:
            raise ValueError("xgb_direct requires a parameter set.")
        return XgbDirectModel(XgbDirectParams(**params), seed=seed)
    if model_name == "lstm_direct":
        if params is None:
            raise ValueError("lstm_direct requires a parameter set.")
        return LstmDirectModel(LstmDirectParams(**params), seed=seed)
    raise ValueError(f"Unsupported model family: {model_name}")


def load_candidate_params(config: AppConfig, model_name: str) -> list[dict[str, Any]] | None:
    if model_name in {"rw_last", "pca_var1"}:
        return None
    if model_name == "xgb_direct":
        return [item.to_dict() for item in load_xgb_candidates(Path(config.models.xgboost_config))]
    if model_name == "lstm_direct":
        return [item.to_dict() for item in load_lstm_candidates(Path(config.models.lstm_config))]
    raise ValueError(f"Unsupported model family: {model_name}")


def _forecast_validation_chunk(
    model_name: str,
    params: dict[str, Any] | None,
    available_rows: pl.DataFrame,
    chunk_rows: pl.DataFrame,
    sampled_surface_wide: pl.DataFrame,
    surface_nodes_store: DatePartitionIndex,
    config: AppConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = instantiate_model(model_name, params, config.runtime.seed)
    model.fit(available_rows)
    chunk_start = chunk_rows["target_date"][0]
    available_dates = sampled_surface_wide.filter(pl.col("quote_date") < chunk_start)[
        "quote_date"
    ].to_list()
    if not available_dates:
        raise ValueError(
            f"Reconstructor chronology violation: no sampled surfaces are available prior to {chunk_start}."
        )
    if max(available_dates) >= chunk_start:
        raise ValueError(
            "Reconstructor chronology violation: sampled surfaces for refitting are not strictly "
            f"earlier than chunk start {chunk_start}."
        )
    reconstructor = train_reconstructor(
        sampled_surface_wide=sampled_surface_wide,
        surface_nodes=surface_nodes_store.load_many(available_dates),
        available_dates=available_dates,
        config_path=Path(config.models.reconstructor_config),
        seed=config.runtime.seed,
    )
    predictions = model.predict(chunk_rows)
    loss_rows: list[dict[str, Any]] = []
    arbitrage_rows: list[dict[str, Any]] = []
    target_node_cache: dict[object, pl.DataFrame] = {}
    for row, pred in zip(chunk_rows.iter_rows(named=True), predictions, strict=True):
        target_date = row["target_date"]
        target_nodes = target_node_cache.get(target_date)
        if target_nodes is None:
            target_nodes = surface_nodes_store.load_date(target_date)
            target_node_cache[target_date] = target_nodes
        loss_row, arbitrage_row, _ = evaluate_node_forecast(
            model_name=model_name,
            quote_date=row["quote_date"],
            target_date=row["target_date"],
            predicted_sampled_iv=np.exp(pred),
            reconstructor=reconstructor,
            target_nodes=target_nodes,
        )
        loss_rows.append(loss_row)
        arbitrage_rows.append(arbitrage_row)
    return loss_rows, arbitrage_rows


def walk_forward_validation_panel(
    model_name: str,
    params: dict[str, Any] | None,
    labeled_features: pl.DataFrame,
    validation_target_dates: list[object],
    sampled_surface_wide: pl.DataFrame,
    surface_nodes_store: DatePartitionIndex,
    config: AppConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    loss_rows: list[dict[str, Any]] = []
    arbitrage_rows: list[dict[str, Any]] = []
    for offset in range(0, len(validation_target_dates), config.split.refit_frequency):
        chunk_dates = validation_target_dates[offset : offset + config.split.refit_frequency]
        chunk_rows = labeled_features.filter(pl.col("target_date").is_in(chunk_dates)).sort(
            "target_date"
        )
        available_rows = labeled_features.filter(pl.col("target_date") < chunk_dates[0]).filter(
            pl.col("split_label").is_in(["train", "validation"])
        ).sort("target_date")
        assert_refit_window_precedes_chunk(available_rows, chunk_rows, "validation_refit")
        chunk_loss_rows, chunk_arbitrage_rows = _forecast_validation_chunk(
            model_name=model_name,
            params=params,
            available_rows=available_rows,
            chunk_rows=chunk_rows,
            sampled_surface_wide=sampled_surface_wide,
            surface_nodes_store=surface_nodes_store,
            config=config,
        )
        loss_rows.extend(chunk_loss_rows)
        arbitrage_rows.extend(chunk_arbitrage_rows)
    return pl.DataFrame(loss_rows), pl.DataFrame(arbitrage_rows)


def tune_model_family(
    model_name: str,
    labeled_features: pl.DataFrame,
    validation_target_dates: list[object],
    sampled_surface_wide: pl.DataFrame,
    surface_nodes_store: DatePartitionIndex,
    config: AppConfig,
) -> SelectedModelConfig:
    candidates = load_candidate_params(config, model_name)
    if candidates is None:
        return SelectedModelConfig(model_name=model_name, params=None)
    evaluations: list[CandidateEvaluation] = []
    for params in candidates:
        loss_panel, arbitrage_panel = walk_forward_validation_panel(
            model_name=model_name,
            params=params,
            labeled_features=labeled_features,
            validation_target_dates=validation_target_dates,
            sampled_surface_wide=sampled_surface_wide,
            surface_nodes_store=surface_nodes_store,
            config=config,
        )
        evaluations.append(
            CandidateEvaluation(
                params=params,
                mean_rmse_iv=float(loss_panel["rmse_iv"].mean()),
                mean_vega_rmse_iv=float(loss_panel["vega_rmse_iv"].mean()),
                mean_violating_fraction=float(arbitrage_panel["violating_fraction"].mean()),
            )
        )
    evaluations.sort(
        key=lambda item: (
            item.mean_rmse_iv,
            item.mean_vega_rmse_iv,
            item.mean_violating_fraction,
        )
    )
    return SelectedModelConfig(model_name=model_name, params=evaluations[0].params)
