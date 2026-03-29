from __future__ import annotations

import math

import polars as pl

from ivs_forecast.data.dfw import default_grid_definition, fit_dfw_surface, sample_dfw


def build_sampled_surfaces(surface_nodes: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    grid = default_grid_definition()
    sampled_rows: list[dict[str, object]] = []
    for nodes in surface_nodes.partition_by("quote_date", maintain_order=True):
        fit = fit_dfw_surface(nodes)
        sampled_iv = sample_dfw(fit.coefficients, grid)
        row: dict[str, object] = {
            "quote_date": fit.quote_date,
            "surface_node_count": nodes.height,
            "valid_expiry_count": int(nodes["expiration"].n_unique()),
            "active_underlying_price_1545": float(nodes["active_underlying_price_1545"].mean()),
            "total_trade_volume": int(nodes["trade_volume"].sum()),
            "total_open_interest": int(nodes["open_interest"].sum()),
            "median_rel_spread_1545": float(nodes["median_rel_spread_1545"].median()),
        }
        for coefficient_index, coefficient in enumerate(fit.coefficients.tolist()):
            row[f"dfw_a{coefficient_index}"] = float(coefficient)
        for item, iv_value in zip(grid.iter_rows(named=True), sampled_iv, strict=True):
            grid_id = item["grid_id"]
            row[f"iv_{grid_id}"] = float(iv_value)
            row[f"logiv_{grid_id}"] = float(math.log(iv_value))
        sampled_rows.append(row)
    return grid, pl.DataFrame(sampled_rows).sort("quote_date")
