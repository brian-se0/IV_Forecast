from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivs_forecast.models.base import (
    ModelArtifact,
    SampledSurfaceModel,
    current_surface_matrix,
    target_matrix,
)

FACTOR_COUNT = 3


@dataclass
class PcaVar1State:
    mean: np.ndarray
    components: np.ndarray
    intercept: np.ndarray
    transition_matrix: np.ndarray


class PcaVar1Model(SampledSurfaceModel):
    model_name = "pca_var1"

    def __init__(self) -> None:
        self.state: PcaVar1State | None = None

    def fit(self, train_frame: pl.DataFrame) -> None:
        x_train = current_surface_matrix(train_frame)
        y_train = target_matrix(train_frame)
        if x_train.shape[0] < FACTOR_COUNT + 1:
            raise ValueError(
                "pca_var1 requires at least four training rows to fit a 3-factor VAR(1)."
            )
        mean = x_train.mean(axis=0)
        centered = x_train - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:FACTOR_COUNT]
        current_scores = centered @ components.T
        next_scores = (y_train - mean) @ components.T
        design = np.column_stack([np.ones(current_scores.shape[0]), current_scores])
        coefficients, *_ = np.linalg.lstsq(design, next_scores, rcond=None)
        intercept = coefficients[0]
        transition_matrix = coefficients[1:]
        self.state = PcaVar1State(
            mean=mean,
            components=components,
            intercept=intercept,
            transition_matrix=transition_matrix,
        )

    def predict(self, feature_frame: pl.DataFrame) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("pca_var1 must be fit before prediction.")
        x_curr = current_surface_matrix(feature_frame)
        current_scores = (x_curr - self.state.mean) @ self.state.components.T
        next_scores_hat = self.state.intercept + current_scores @ self.state.transition_matrix
        return self.state.mean + next_scores_hat @ self.state.components

    def artifact(self) -> ModelArtifact:
        if self.state is None:
            return super().artifact()
        return ModelArtifact(
            model_name=self.model_name,
            params={
                "factor_count": FACTOR_COUNT,
                "has_intercept": True,
                "transition_shape": list(self.state.transition_matrix.shape),
            },
        )
