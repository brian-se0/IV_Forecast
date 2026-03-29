from __future__ import annotations

import math

import polars as pl

from ivs_forecast.config import AppConfig


def build_surface_nodes(
    clean_contracts: pl.DataFrame,
    forward_terms: pl.DataFrame,
    config: AppConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    joined = clean_contracts.join(forward_terms, on=["quote_date", "expiration"], how="inner")
    joined = joined.with_columns(
        pl.struct(["strike", "forward_price"])
        .map_elements(
            lambda item: float(math.log(item["strike"] / item["forward_price"])),
            return_dtype=pl.Float64,
        )
        .alias("m")
    )
    aggregated = (
        joined.group_by(["quote_date", "expiration", "strike"])
        .agg(
            pl.col("tau").first().alias("tau"),
            pl.col("m").first().alias("m"),
            (
                (pl.col("implied_volatility_1545") * pl.col("vega_1545")).sum()
                / pl.col("vega_1545").sum()
            ).alias("node_iv"),
            pl.col("vega_1545").sum().alias("node_vega"),
            pl.col("active_underlying_price_1545").mean().alias("active_underlying_price_1545"),
            pl.col("trade_volume").sum().alias("trade_volume"),
            pl.col("open_interest").sum().alias("open_interest"),
            pl.col("rel_spread_1545").median().alias("median_rel_spread_1545"),
            pl.col("forward_price").first().alias("forward_price"),
            pl.col("discount_factor").first().alias("discount_factor"),
            pl.len().alias("collapsed_contract_count"),
        )
        .sort(["quote_date", "expiration", "strike"])
    )
    date_quality = (
        aggregated.group_by("quote_date")
        .agg(
            pl.len().alias("surface_node_count"),
            pl.col("expiration").n_unique().alias("valid_expiry_count"),
        )
        .with_columns(
            (
                (pl.col("surface_node_count") >= config.study.min_surface_nodes)
                & (pl.col("valid_expiry_count") >= config.study.min_valid_expiries)
            ).alias("modeling_valid")
        )
    )
    valid_dates = date_quality.filter(pl.col("modeling_valid"))["quote_date"].to_list()
    valid_nodes = aggregated.filter(pl.col("quote_date").is_in(valid_dates))
    return valid_nodes, date_quality.sort("quote_date")
