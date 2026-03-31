from __future__ import annotations

import numpy as np
import polars as pl

from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.models.base import DailyStateStore, ModelArtifact, SsviStateModel


class StateLastModel(SsviStateModel):
    model_name = "state_last"

    def fit(
        self,
        train_rows: pl.DataFrame,
        state_store: DailyStateStore,
        surface_nodes_store: DatePartitionIndex | None = None,
    ) -> None:
        if train_rows.is_empty():
            raise ValueError("state_last requires at least one training row.")

    def predict(
        self,
        feature_rows: pl.DataFrame,
        state_store: DailyStateStore,
    ) -> np.ndarray:
        indices = feature_rows["surface_state_row_index"].to_numpy().astype(np.int64)
        return state_store.latent_by_indices(indices)

    def artifact(self) -> ModelArtifact:
        return ModelArtifact(model_name=self.model_name, params={"strategy": "copy_current_state"})
