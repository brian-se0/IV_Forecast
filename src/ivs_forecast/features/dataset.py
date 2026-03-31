from __future__ import annotations

import polars as pl

from ivs_forecast.features.scalars import scalar_feature_columns

STATE_Z_DIM = 14


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


def assert_feature_target_separation() -> None:
    overlap = sorted(set(origin_scalar_feature_columns()) & {"target_state_row_index"})
    if overlap:
        raise ValueError(f"Feature/target leakage detected in shared column names: {overlap}")


def build_features_targets(ssvi_state: pl.DataFrame, minimum_history_days: int = 22) -> pl.DataFrame:
    ordered = _validated_ssvi_state_panel(ssvi_state)
    rows: list[dict[str, object]] = []
    for origin_index in range(minimum_history_days - 1, ordered.height - 1):
        target_index = origin_index + 1
        row: dict[str, object] = {
            "quote_date": ordered["quote_date"][origin_index],
            "target_date": ordered["quote_date"][target_index],
            "option_root": ordered["option_root"][origin_index],
            "history_start_index": origin_index - minimum_history_days + 1,
            "history_end_index": origin_index,
            "surface_state_row_index": int(ordered["state_row_index"][origin_index]),
            "target_state_row_index": int(ordered["state_row_index"][target_index]),
        }
        for column in origin_scalar_feature_columns():
            row[column] = float(ordered[column][origin_index])
        rows.append(row)
    if not rows:
        raise ValueError("Too few valid SSVI dates to build features_targets.parquet.")
    assert_feature_target_separation()
    features = pl.DataFrame(rows).sort("quote_date")
    expected_next_dates = ordered["quote_date"][minimum_history_days:].to_list()
    if features["target_date"].to_list() != expected_next_dates:
        raise ValueError("Each target_date must equal the next available modeling date.")
    return features
