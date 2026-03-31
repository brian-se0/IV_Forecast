from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from ivs_forecast.config import AppConfig
from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.models.base import DailyStateStore, SsviStateModel
from ivs_forecast.models.ssvi_tcn_direct import (
    SsviTcnDirectModel,
    SsviTcnParams,
)
from ivs_forecast.models.ssvi_tcn_direct import (
    load_search_space as load_tcn_candidates,
)
from ivs_forecast.models.state_last import StateLastModel
from ivs_forecast.models.state_var1 import StateVar1Model
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
    mean_calendar_violations: float
    mean_butterfly_violations: float


def instantiate_model(
    model_name: str,
    params: dict[str, Any] | None,
    seed: int,
) -> SsviStateModel:
    if model_name == "state_last":
        return StateLastModel()
    if model_name == "state_var1":
        return StateVar1Model()
    if model_name == "ssvi_tcn_direct":
        if params is None:
            raise ValueError("ssvi_tcn_direct requires a parameter set.")
        return SsviTcnDirectModel(SsviTcnParams(**params), seed=seed)
    raise ValueError(f"Unsupported model family: {model_name}")


def load_candidate_params(model_name: str) -> list[dict[str, Any]] | None:
    if model_name in {"state_last", "state_var1"}:
        return None
    if model_name == "ssvi_tcn_direct":
        return [item.to_dict() for item in load_tcn_candidates()]
    raise ValueError(f"Unsupported model family: {model_name}")


def _forecast_validation_chunk(
    model_name: str,
    params: dict[str, Any] | None,
    available_rows: pl.DataFrame,
    chunk_rows: pl.DataFrame,
    state_store: DailyStateStore,
    surface_nodes_store: DatePartitionIndex,
    config: AppConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = instantiate_model(model_name, params, config.runtime.seed)
    model.fit(available_rows, state_store, surface_nodes_store)
    predictions = model.predict(chunk_rows, state_store)
    loss_rows: list[dict[str, Any]] = []
    certification_rows: list[dict[str, Any]] = []
    target_node_cache: dict[object, pl.DataFrame] = {}
    for row, prediction in zip(chunk_rows.iter_rows(named=True), predictions, strict=True):
        target_date = row["target_date"]
        target_nodes = target_node_cache.get(target_date)
        if target_nodes is None:
            target_nodes = surface_nodes_store.load_date(target_date)
            target_node_cache[target_date] = target_nodes
        loss_row, certification_row, _ = evaluate_node_forecast(
            model_name=model_name,
            quote_date=row["quote_date"],
            target_date=row["target_date"],
            predicted_state_z=prediction,
            target_nodes=target_nodes,
        )
        loss_rows.append(loss_row)
        certification_rows.append(certification_row)
    return loss_rows, certification_rows


def walk_forward_validation_panel(
    model_name: str,
    params: dict[str, Any] | None,
    labeled_features: pl.DataFrame,
    validation_target_dates: list[object],
    state_store: DailyStateStore,
    surface_nodes_store: DatePartitionIndex,
    config: AppConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    loss_rows: list[dict[str, Any]] = []
    certification_rows: list[dict[str, Any]] = []
    for offset in range(0, len(validation_target_dates), config.split.refit_frequency):
        chunk_dates = validation_target_dates[offset : offset + config.split.refit_frequency]
        chunk_rows = labeled_features.filter(pl.col("target_date").is_in(chunk_dates)).sort("target_date")
        available_rows = labeled_features.filter(pl.col("target_date") < chunk_dates[0]).filter(
            pl.col("split_label").is_in(["train", "validation"])
        ).sort("target_date")
        assert_refit_window_precedes_chunk(available_rows, chunk_rows, "validation_refit")
        chunk_loss_rows, chunk_certification_rows = _forecast_validation_chunk(
            model_name=model_name,
            params=params,
            available_rows=available_rows,
            chunk_rows=chunk_rows,
            state_store=state_store,
            surface_nodes_store=surface_nodes_store,
            config=config,
        )
        loss_rows.extend(chunk_loss_rows)
        certification_rows.extend(chunk_certification_rows)
    return pl.DataFrame(loss_rows), pl.DataFrame(certification_rows)


def tune_model_family(
    model_name: str,
    labeled_features: pl.DataFrame,
    validation_target_dates: list[object],
    state_store: DailyStateStore,
    surface_nodes_store: DatePartitionIndex,
    config: AppConfig,
) -> SelectedModelConfig:
    candidates = load_candidate_params(model_name)
    if candidates is None:
        return SelectedModelConfig(model_name=model_name, params=None)
    evaluations: list[CandidateEvaluation] = []
    for params in candidates:
        loss_panel, certification_panel = walk_forward_validation_panel(
            model_name=model_name,
            params=params,
            labeled_features=labeled_features,
            validation_target_dates=validation_target_dates,
            state_store=state_store,
            surface_nodes_store=surface_nodes_store,
            config=config,
        )
        evaluations.append(
            CandidateEvaluation(
                params=params,
                mean_rmse_iv=float(loss_panel["rmse_iv"].mean()),
                mean_vega_rmse_iv=float(loss_panel["vega_rmse_iv"].mean()),
                mean_calendar_violations=float(certification_panel["calendar_violation_count"].mean()),
                mean_butterfly_violations=float(certification_panel["butterfly_violation_count"].mean()),
            )
        )
    evaluations.sort(
        key=lambda item: (
            item.mean_vega_rmse_iv,
            item.mean_rmse_iv,
            item.mean_calendar_violations,
            item.mean_butterfly_violations,
        )
    )
    return SelectedModelConfig(model_name=model_name, params=evaluations[0].params)
