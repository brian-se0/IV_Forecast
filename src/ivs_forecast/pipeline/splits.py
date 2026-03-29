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


def _validated_target_dates(features_targets: pl.DataFrame) -> list[date]:
    if features_targets["quote_date"].null_count() > 0 or features_targets["target_date"].null_count() > 0:
        raise ValueError("Feature rows must not contain null quote_date or target_date values.")
    quote_dates = features_targets["quote_date"].to_list()
    if quote_dates != sorted(quote_dates):
        raise ValueError("Feature rows must remain ordered by quote_date.")
    target_dates = features_targets["target_date"].to_list()
    if target_dates != sorted(target_dates):
        raise ValueError("Feature rows must remain ordered by target_date.")
    unique_target_dates = sorted(features_targets["target_date"].unique().to_list())
    if len(unique_target_dates) != len(target_dates):
        raise ValueError("Feature rows must not contain duplicate target_date values.")
    for quote_date, target_date in zip(quote_dates, target_dates, strict=True):
        if not quote_date < target_date:
            raise ValueError("Each feature row must forecast a strictly later target_date.")
    return unique_target_dates


def assert_refit_window_precedes_chunk(
    available_rows: pl.DataFrame,
    chunk_rows: pl.DataFrame,
    stage_name: str,
) -> None:
    if chunk_rows.is_empty():
        raise ValueError(f"{stage_name} received an empty forecast chunk.")
    chunk_start = chunk_rows["target_date"][0]
    if available_rows.is_empty():
        raise ValueError(f"{stage_name} received no rows strictly prior to chunk {chunk_start}.")
    available_max = available_rows["target_date"].max()
    if available_max is None or not available_max < chunk_start:
        raise ValueError(
            f"{stage_name} chronology violation: training/refit rows end at {available_max}, "
            f"which is not strictly earlier than chunk start {chunk_start}."
        )


def build_split_manifest(features_targets: pl.DataFrame, config: AppConfig) -> SplitManifest:
    target_dates = _validated_target_dates(features_targets)
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
