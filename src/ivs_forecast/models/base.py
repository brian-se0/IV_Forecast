from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch

from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.data.ssvi import maturity_knots_days
from ivs_forecast.features.dataset import state_z_columns
from ivs_forecast.features.scalars import scalar_feature_columns


def theta_columns() -> list[str]:
    return [f"theta_d{int(days):03d}" for days in maturity_knots_days()]


def state_parameter_columns() -> list[str]:
    return theta_columns() + ["rho", "eta", "lambda"]


def history_feature_columns() -> list[str]:
    return state_z_columns() + scalar_feature_columns()


def assert_cuda_available(stage_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for {stage_name}, but no CUDA device is available.")


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(np.float64).tolist(),
            "std": self.std.astype(np.float64).tolist(),
        }

    def apply(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / self.std


@dataclass(frozen=True)
class ModelArtifact:
    model_name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DailyStateStore:
    frame: pl.DataFrame
    latent_matrix: np.ndarray
    parameter_matrix: np.ndarray
    daily_feature_matrix: np.ndarray

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> DailyStateStore:
        ordered = frame.sort("quote_date")
        return cls(
            frame=ordered,
            latent_matrix=ordered.select(state_z_columns()).to_numpy().astype(np.float64),
            parameter_matrix=ordered.select(state_parameter_columns()).to_numpy().astype(np.float64),
            daily_feature_matrix=ordered.select(history_feature_columns()).to_numpy().astype(np.float32),
        )

    def latent_by_indices(self, indices: list[int] | np.ndarray) -> np.ndarray:
        return self.latent_matrix[np.asarray(indices, dtype=np.int64)]

    def parameters_by_indices(self, indices: list[int] | np.ndarray) -> np.ndarray:
        return self.parameter_matrix[np.asarray(indices, dtype=np.int64)]

    def history_tensor(self, history_end_index: int, history_days: int) -> np.ndarray:
        start_index = history_end_index - history_days + 1
        if start_index < 0:
            raise ValueError(
                f"History window underflow: end_index={history_end_index}, history_days={history_days}."
            )
        return self.daily_feature_matrix[start_index : history_end_index + 1]


def unique_history_indices(feature_rows: pl.DataFrame, history_days: int) -> np.ndarray:
    indices: set[int] = set()
    for row in feature_rows.select("history_end_index").iter_rows(named=True):
        history_end = int(row["history_end_index"])
        start_index = history_end - history_days + 1
        if start_index < 0:
            raise ValueError(
                f"History window underflow while collecting normalization rows: {start_index}."
            )
        indices.update(range(start_index, history_end + 1))
    return np.asarray(sorted(indices), dtype=np.int64)


def fit_normalization(
    feature_rows: pl.DataFrame,
    state_store: DailyStateStore,
    history_days: int,
) -> NormalizationStats:
    indices = unique_history_indices(feature_rows, history_days)
    matrix = state_store.daily_feature_matrix[indices].astype(np.float64)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return NormalizationStats(mean=matrix.mean(axis=0), std=std)


class SsviStateModel(ABC):
    model_name: str

    @abstractmethod
    def fit(
        self,
        train_rows: pl.DataFrame,
        state_store: DailyStateStore,
        surface_nodes_store: DatePartitionIndex | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        feature_rows: pl.DataFrame,
        state_store: DailyStateStore,
    ) -> np.ndarray:
        raise NotImplementedError

    def artifact(self) -> ModelArtifact:
        return ModelArtifact(model_name=self.model_name, params={})
