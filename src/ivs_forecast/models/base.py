from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch

from ivs_forecast.features.dataset import grid_ids, x_feature_columns, y_target_columns


def x_curr_columns() -> list[str]:
    return [f"x_curr_{grid_id}" for grid_id in grid_ids()]


def x_ma5_columns() -> list[str]:
    return [f"x_ma5_{grid_id}" for grid_id in grid_ids()]


def x_ma22_columns() -> list[str]:
    return [f"x_ma22_{grid_id}" for grid_id in grid_ids()]


def scalar_feature_columns() -> list[str]:
    return [
        "underlying_logret_1",
        "underlying_logret_5",
        "underlying_logret_22",
        "log1p_total_trade_volume",
        "log1p_total_open_interest",
        "median_rel_spread_1545",
    ]


def features_matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(x_feature_columns()).to_numpy().astype(np.float32)


def current_surface_matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(x_curr_columns()).to_numpy().astype(np.float64)


def target_matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(y_target_columns()).to_numpy().astype(np.float64)


def sequence_matrix(frame: pl.DataFrame) -> np.ndarray:
    scalar = frame.select(scalar_feature_columns()).to_numpy().astype(np.float32)
    ma22 = frame.select(x_ma22_columns()).to_numpy().astype(np.float32)
    ma5 = frame.select(x_ma5_columns()).to_numpy().astype(np.float32)
    curr = frame.select(x_curr_columns()).to_numpy().astype(np.float32)
    step0 = np.concatenate([ma22, scalar], axis=1)
    step1 = np.concatenate([ma5, scalar], axis=1)
    step2 = np.concatenate([curr, scalar], axis=1)
    return np.stack([step0, step1, step2], axis=1)


def assert_cuda_available(stage_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for {stage_name}, but no CUDA device is available.")


@dataclass
class ModelArtifact:
    model_name: str
    params: dict[str, Any]


class SampledSurfaceModel(ABC):
    model_name: str

    @abstractmethod
    def fit(self, train_frame: pl.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, feature_frame: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def artifact(self) -> ModelArtifact:
        return ModelArtifact(model_name=self.model_name, params={})
