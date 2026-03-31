from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from ivs_forecast.features.scalars import scalar_feature_columns

STATE_Z_DIM = 14


@dataclass(frozen=True)
class FeatureTargetArtifacts:
    trading_date_index: pl.DataFrame
    features_targets: pl.DataFrame
    exclusions: pl.DataFrame


def state_z_columns() -> list[str]:
    return [f"state_z_{index:03d}" for index in range(STATE_Z_DIM)]


def origin_scalar_feature_columns() -> list[str]:
    return scalar_feature_columns()


def _validated_ssvi_state_panel(ssvi_state: pl.DataFrame) -> pl.DataFrame:
    if ssvi_state.is_empty():
        raise ValueError("SSVI state panel is empty.")
    required = {"quote_date", "option_root", "state_row_index", *state_z_columns(), *scalar_feature_columns()}
    missing = sorted(required - set(ssvi_state.columns))
    if missing:
        raise ValueError(f"SSVI state panel is missing required columns: {missing}")
    ordered = ssvi_state.sort("quote_date")
    quote_dates = ordered["quote_date"].to_list()
    if len(set(quote_dates)) != len(quote_dates):
        raise ValueError("SSVI state panel must not contain duplicate quote_date rows.")
    if quote_dates != sorted(quote_dates):
        raise ValueError("SSVI state panel must be strictly ordered by quote_date.")
    row_indices = ordered["state_row_index"].to_list()
    if row_indices != list(range(len(row_indices))):
        raise ValueError("state_row_index must be contiguous and zero-based.")
    return ordered


def _validated_trading_dates(trading_dates: Sequence[date]) -> list[date]:
    ordered = list(trading_dates)
    if not ordered:
        raise ValueError("Trading-date index cannot be empty.")
    if any(item is None for item in ordered):
        raise ValueError("Trading-date index cannot contain null dates.")
    if ordered != sorted(ordered):
        raise ValueError("Trading-date index must be strictly ordered by quote_date.")
    if len(set(ordered)) != len(ordered):
        raise ValueError("Trading-date index must not contain duplicate quote_date rows.")
    return ordered


def build_trading_date_index(
    trading_dates: Sequence[date],
    option_root: str,
    ssvi_state: pl.DataFrame,
) -> pl.DataFrame:
    ordered_dates = _validated_trading_dates(trading_dates)
    ordered_state = _validated_ssvi_state_panel(ssvi_state)
    state_row_by_date = {
        quote_date: int(state_row_index)
        for quote_date, state_row_index in zip(
            ordered_state["quote_date"].to_list(),
            ordered_state["state_row_index"].to_list(),
            strict=True,
        )
    }
    rows: list[dict[str, object]] = []
    for index, quote_date in enumerate(ordered_dates):
        next_trading_date = ordered_dates[index + 1] if index + 1 < len(ordered_dates) else None
        surface_state_row_index = state_row_by_date.get(quote_date)
        rows.append(
            {
                "quote_date": quote_date,
                "option_root": option_root,
                "trading_day_index": index,
                "next_trading_date": next_trading_date,
                "has_surface_state": surface_state_row_index is not None,
                "surface_state_row_index": surface_state_row_index,
            }
        )
    return pl.DataFrame(rows).sort("quote_date")


def assert_feature_target_separation() -> None:
    overlap = sorted(set(origin_scalar_feature_columns()) & {"target_state_row_index"})
    if overlap:
        raise ValueError(f"Feature/target leakage detected in shared column names: {overlap}")


def build_features_targets(
    ssvi_state: pl.DataFrame,
    trading_date_index: pl.DataFrame,
    minimum_history_days: int = 22,
) -> FeatureTargetArtifacts:
    ordered = _validated_ssvi_state_panel(ssvi_state)
    required_trading_columns = {
        "quote_date",
        "option_root",
        "trading_day_index",
        "next_trading_date",
        "has_surface_state",
        "surface_state_row_index",
    }
    missing_trading_columns = sorted(required_trading_columns - set(trading_date_index.columns))
    if missing_trading_columns:
        raise ValueError(
            f"Trading-date index is missing required columns: {missing_trading_columns}"
        )
    trading_dates = trading_date_index.sort("quote_date")
    if trading_dates["quote_date"].to_list() != sorted(trading_dates["quote_date"].to_list()):
        raise ValueError("Trading-date index must remain ordered by quote_date.")
    state_row_by_date = {
        quote_date: int(state_row_index)
        for quote_date, state_row_index in zip(
            ordered["quote_date"].to_list(),
            ordered["state_row_index"].to_list(),
            strict=True,
        )
    }
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for candidate in trading_dates.iter_rows(named=True):
        quote_date = candidate["quote_date"]
        target_date = candidate["next_trading_date"]
        if target_date is None:
            continue
        surface_state_row_index = state_row_by_date.get(quote_date)
        target_state_row_index = state_row_by_date.get(target_date)
        if surface_state_row_index is None:
            exclusions.append(
                {
                    "quote_date": quote_date,
                    "target_date": target_date,
                    "option_root": candidate["option_root"],
                    "exclusion_reason": "missing_origin_state",
                }
            )
            continue
        if target_state_row_index is None:
            exclusions.append(
                {
                    "quote_date": quote_date,
                    "target_date": target_date,
                    "option_root": candidate["option_root"],
                    "exclusion_reason": "missing_target_state",
                }
            )
            continue
        if surface_state_row_index < minimum_history_days - 1:
            exclusions.append(
                {
                    "quote_date": quote_date,
                    "target_date": target_date,
                    "option_root": candidate["option_root"],
                    "exclusion_reason": "missing_history_window",
                }
            )
            continue
        row: dict[str, object] = {
            "quote_date": quote_date,
            "target_date": target_date,
            "option_root": candidate["option_root"],
            "history_start_index": surface_state_row_index - minimum_history_days + 1,
            "history_end_index": surface_state_row_index,
            "surface_state_row_index": surface_state_row_index,
            "target_state_row_index": target_state_row_index,
        }
        for column in origin_scalar_feature_columns():
            row[column] = float(ordered[column][surface_state_row_index])
        rows.append(row)
    if not rows:
        raise ValueError("Too few valid SSVI dates to build features_targets.parquet.")
    assert_feature_target_separation()
    features = pl.DataFrame(rows).sort("quote_date")
    exclusion_frame = (
        pl.DataFrame(exclusions).sort(["quote_date", "target_date"])
        if exclusions
        else pl.DataFrame(
            schema={
                "quote_date": pl.Date,
                "target_date": pl.Date,
                "option_root": pl.String,
                "exclusion_reason": pl.String,
            }
        )
    )
    return FeatureTargetArtifacts(
        trading_date_index=trading_dates,
        features_targets=features,
        exclusions=exclusion_frame,
    )
