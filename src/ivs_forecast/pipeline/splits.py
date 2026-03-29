from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from ivs_forecast.artifacts.manifests import write_json
from ivs_forecast.config import AppConfig


@dataclass(frozen=True)
class SplitManifest:
    train_target_dates: list[date]
    validation_target_dates: list[date]
    test_target_dates: list[date]

    def to_dict(self) -> dict[str, object]:
        return {
            "train_target_dates": [item.isoformat() for item in self.train_target_dates],
            "validation_target_dates": [item.isoformat() for item in self.validation_target_dates],
            "test_target_dates": [item.isoformat() for item in self.test_target_dates],
        }


def build_split_manifest(features_targets: pl.DataFrame, config: AppConfig) -> SplitManifest:
    target_dates = sorted(features_targets["target_date"].unique().to_list())
    required = config.split.validation_size + config.split.test_size + config.split.min_train_size
    if len(target_dates) < required:
        raise ValueError(
            f"Too few valid dates after preprocessing: need at least {required}, found {len(target_dates)}."
        )
    test_target_dates = target_dates[-config.split.test_size :]
    validation_end = len(target_dates) - config.split.test_size
    validation_target_dates = target_dates[
        validation_end - config.split.validation_size : validation_end
    ]
    train_target_dates = target_dates[: validation_end - config.split.validation_size]
    if not (
        max(train_target_dates)
        < min(validation_target_dates)
        < max(validation_target_dates)
        < min(test_target_dates)
    ):
        raise ValueError(
            "Chronology split violation: train/validation/test target dates are not strictly ordered."
        )
    return SplitManifest(
        train_target_dates=train_target_dates,
        validation_target_dates=validation_target_dates,
        test_target_dates=test_target_dates,
    )


def label_feature_rows(features_targets: pl.DataFrame, manifest: SplitManifest) -> pl.DataFrame:
    return features_targets.with_columns(
        pl.when(pl.col("target_date").is_in(manifest.train_target_dates))
        .then(pl.lit("train"))
        .when(pl.col("target_date").is_in(manifest.validation_target_dates))
        .then(pl.lit("validation"))
        .when(pl.col("target_date").is_in(manifest.test_target_dates))
        .then(pl.lit("test"))
        .otherwise(pl.lit("discard"))
        .alias("split_label")
    )


def write_split_manifest(path: Path, manifest: SplitManifest) -> None:
    write_json(path, manifest.to_dict())
