from __future__ import annotations

import numpy as np
import polars as pl

from ivs_forecast.models.base import SampledSurfaceModel, current_surface_matrix


class RwLastModel(SampledSurfaceModel):
    model_name = "rw_last"

    def fit(self, train_frame: pl.DataFrame) -> None:
        _ = train_frame

    def predict(self, feature_frame: pl.DataFrame) -> np.ndarray:
        return current_surface_matrix(feature_frame)
