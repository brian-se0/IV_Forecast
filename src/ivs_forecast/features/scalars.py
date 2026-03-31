from __future__ import annotations

import numpy as np
import polars as pl

SCALAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "underlying_log_return_1d",
    "underlying_log_return_5d",
    "underlying_log_return_22d",
    "log_total_trade_volume",
    "log_total_open_interest",
    "median_rel_spread_1545",
    "surface_fit_rmse_iv",
    "surface_node_count",
    "valid_expiry_count",
)


def scalar_feature_columns() -> list[str]:
    return list(SCALAR_FEATURE_COLUMNS)


def log_return(series: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(series.shape, np.nan, dtype=np.float64)
    for index in range(horizon, series.shape[0]):
        if series[index] <= 0 or series[index - horizon] <= 0:
            raise ValueError("Log returns require strictly positive underlying prices.")
        result[index] = np.log(series[index] / series[index - horizon])
    return result


def add_state_scalar_features(ssvi_state: pl.DataFrame) -> pl.DataFrame:
    if ssvi_state.is_empty():
        raise ValueError("Cannot build scalar features from an empty SSVI state panel.")
    ordered = ssvi_state.sort("quote_date")
    prices = ordered["active_underlying_price_1545"].to_numpy().astype(np.float64)
    scalar_columns = {
        "underlying_log_return_1d": log_return(prices, 1),
        "underlying_log_return_5d": log_return(prices, 5),
        "underlying_log_return_22d": log_return(prices, 22),
        "log_total_trade_volume": np.log(
            np.maximum(ordered["total_trade_volume"].to_numpy().astype(np.float64), 1.0)
        ),
        "log_total_open_interest": np.log(
            np.maximum(ordered["total_open_interest"].to_numpy().astype(np.float64), 1.0)
        ),
    }
    enriched = ordered.with_columns(
        *[
            pl.Series(name, values)
            for name, values in scalar_columns.items()
        ]
    ).with_columns(
        pl.col("surface_fit_rmse_iv").fill_null(strategy="forward").fill_null(0.0),
        pl.col("median_rel_spread_1545").fill_null(0.0),
        pl.col("surface_node_count").cast(pl.Float64),
        pl.col("valid_expiry_count").cast(pl.Float64),
    )
    return enriched.with_columns(
        *[pl.col(column).fill_nan(0.0).fill_null(0.0) for column in scalar_feature_columns()]
    )
