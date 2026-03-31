from __future__ import annotations

import numpy as np
import polars as pl

from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.models.base import DailyStateStore, ModelArtifact, SsviStateModel


class StateVar1Model(SsviStateModel):
    model_name = "state_var1"

    def __init__(self) -> None:
        self.intercept_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None

    def fit(
        self,
        train_rows: pl.DataFrame,
        state_store: DailyStateStore,
        surface_nodes_store: DatePartitionIndex | None = None,
    ) -> None:
        if train_rows.height < 2:
            raise ValueError("state_var1 requires at least two training rows.")
        x_indices = train_rows["surface_state_row_index"].to_numpy().astype(np.int64)
        y_indices = train_rows["target_state_row_index"].to_numpy().astype(np.int64)
        x = state_store.latent_by_indices(x_indices)
        y = state_store.latent_by_indices(y_indices)
        design = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_ = coefficients[0]
        self.transition_ = coefficients[1:].T

    def predict(
        self,
        feature_rows: pl.DataFrame,
        state_store: DailyStateStore,
    ) -> np.ndarray:
        if self.intercept_ is None or self.transition_ is None:
            raise RuntimeError("state_var1 must be fit before prediction.")
        x_indices = feature_rows["surface_state_row_index"].to_numpy().astype(np.int64)
        x = state_store.latent_by_indices(x_indices)
        return self.intercept_[None, :] + x @ self.transition_.T

    def artifact(self) -> ModelArtifact:
        if self.intercept_ is None or self.transition_ is None:
            return ModelArtifact(model_name=self.model_name, params={})
        return ModelArtifact(
            model_name=self.model_name,
            params={
                "intercept_shape": list(self.intercept_.shape),
                "transition_shape": list(self.transition_.shape),
            },
        )
