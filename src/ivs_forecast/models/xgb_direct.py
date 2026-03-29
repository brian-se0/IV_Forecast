from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
import yaml

from ivs_forecast.models.base import (
    ModelArtifact,
    SampledSurfaceModel,
    assert_cuda_available,
    features_matrix,
    target_matrix,
)


@dataclass(frozen=True)
class XgbDirectParams:
    max_depth: int
    learning_rate: float
    n_estimators: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    min_child_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
        }


def load_search_space(config_path: Path) -> list[XgbDirectParams]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grid = payload["search_space"]
    result = []
    for values in product(
        grid["max_depth"],
        grid["learning_rate"],
        grid["n_estimators"],
        grid["subsample"],
        grid["colsample_bytree"],
        grid["reg_lambda"],
        grid["min_child_weight"],
    ):
        result.append(XgbDirectParams(*values))
    return result


class XgbDirectModel(SampledSurfaceModel):
    model_name = "xgb_direct"

    def __init__(self, params: XgbDirectParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.models: list[xgb.Booster] = []

    def fit(self, train_frame: pl.DataFrame) -> None:
        assert_cuda_available(self.model_name)
        x_train = features_matrix(train_frame)
        y_train = target_matrix(train_frame).astype(np.float32)
        self.models = []
        base_params = {
            "objective": "reg:squarederror",
            "device": "cuda",
            "tree_method": "hist",
            "max_depth": self.params.max_depth,
            "eta": self.params.learning_rate,
            "subsample": self.params.subsample,
            "colsample_bytree": self.params.colsample_bytree,
            "lambda": self.params.reg_lambda,
            "min_child_weight": self.params.min_child_weight,
            "seed": self.seed,
            "verbosity": 0,
        }
        for target_index in range(y_train.shape[1]):
            dtrain = xgb.DMatrix(x_train, label=y_train[:, target_index])
            model = xgb.train(
                params=base_params,
                dtrain=dtrain,
                num_boost_round=self.params.n_estimators,
            )
            self.models.append(model)

    def predict(self, feature_frame: pl.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("xgb_direct must be fit before prediction.")
        x_features = features_matrix(feature_frame)
        dmatrix = xgb.DMatrix(x_features)
        preds = np.column_stack([model.predict(dmatrix) for model in self.models])
        return preds.astype(np.float64)

    def artifact(self) -> ModelArtifact:
        return ModelArtifact(model_name=self.model_name, params=self.params.to_dict())
