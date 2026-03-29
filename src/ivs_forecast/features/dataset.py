from __future__ import annotations

import math

import polars as pl

from ivs_forecast.features.scalars import log_return
from ivs_forecast.features.windows import trailing_mean

GRID_SIZE = 154


def grid_ids() -> list[str]:
    return [f"g{index:03d}" for index in range(GRID_SIZE)]


def build_features_targets(sampled_surfaces: pl.DataFrame) -> pl.DataFrame:
    ordered = sampled_surfaces.sort("quote_date")
    ids = grid_ids()
    logiv_columns = [f"logiv_{grid_id}" for grid_id in ids]
    logiv_matrix = ordered.select(logiv_columns).to_numpy()
    current = logiv_matrix
    ma5 = trailing_mean(logiv_matrix, 5)
    ma22 = trailing_mean(logiv_matrix, 22)
    underlying_prices = ordered["active_underlying_price_1545"].to_numpy()
    underlying_logret_1 = log_return(underlying_prices, 1)
    underlying_logret_5 = log_return(underlying_prices, 5)
    underlying_logret_22 = log_return(underlying_prices, 22)
    volume = ordered["total_trade_volume"].to_numpy()
    oi = ordered["total_open_interest"].to_numpy()
    spread = ordered["median_rel_spread_1545"].to_numpy()
    rows: list[dict[str, object]] = []
    for index in range(22, ordered.height - 1):
        row: dict[str, object] = {
            "quote_date": ordered["quote_date"][index],
            "target_date": ordered["quote_date"][index + 1],
            "underlying_logret_1": float(underlying_logret_1[index]),
            "underlying_logret_5": float(underlying_logret_5[index]),
            "underlying_logret_22": float(underlying_logret_22[index]),
            "log1p_total_trade_volume": float(math.log1p(volume[index])),
            "log1p_total_open_interest": float(math.log1p(oi[index])),
            "median_rel_spread_1545": float(spread[index]),
        }
        for grid_index, grid_id in enumerate(ids):
            row[f"x_curr_{grid_id}"] = float(current[index, grid_index])
            row[f"x_ma5_{grid_id}"] = float(ma5[index, grid_index])
            row[f"x_ma22_{grid_id}"] = float(ma22[index, grid_index])
            row[f"y_{grid_id}"] = float(logiv_matrix[index + 1, grid_index])
        rows.append(row)
    if not rows:
        raise ValueError("Too few valid sampled-surface dates to build features/targets.")
    return pl.DataFrame(rows).sort("quote_date")


def x_feature_columns() -> list[str]:
    ids = grid_ids()
    scalar_columns = [
        "underlying_logret_1",
        "underlying_logret_5",
        "underlying_logret_22",
        "log1p_total_trade_volume",
        "log1p_total_open_interest",
        "median_rel_spread_1545",
    ]
    return (
        [f"x_curr_{grid_id}" for grid_id in ids]
        + [f"x_ma5_{grid_id}" for grid_id in ids]
        + [f"x_ma22_{grid_id}" for grid_id in ids]
        + scalar_columns
    )


def y_target_columns() -> list[str]:
    return [f"y_{grid_id}" for grid_id in grid_ids()]
