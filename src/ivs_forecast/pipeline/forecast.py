from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.evaluation.hedged_pnl import hedged_pnl_utility
from ivs_forecast.evaluation.metrics import compute_metrics
from ivs_forecast.evaluation.pricing_mark import black_scholes_price, pricing_utility
from ivs_forecast.evaluation.straddle_signal import straddle_utility
from ivs_forecast.models.reconstructor import FittedReconstructor


def contracts_with_forward(
    clean_contracts: pl.DataFrame, forward_terms: pl.DataFrame
) -> pl.DataFrame:
    joined = clean_contracts.join(
        forward_terms, on=["quote_date", "root", "expiration"], how="inner"
    )
    return joined.with_columns(
        pl.struct(["strike", "forward_price"])
        .map_elements(
            lambda item: float(math.log(item["strike"] / item["forward_price"])),
            return_dtype=pl.Float64,
        )
        .alias("m")
    )


def load_contracts_with_forward_for_date(
    clean_contracts_store: DatePartitionIndex,
    forward_terms: pl.DataFrame,
    quote_date: object,
) -> pl.DataFrame:
    clean_contracts = clean_contracts_store.load_date(quote_date)
    forward_for_date = forward_terms.filter(pl.col("quote_date") == quote_date)
    if forward_for_date.is_empty():
        raise ValueError(
            f"Forward terms are missing for quote_date {quote_date}; the date-partitioned contract "
            "evaluation path cannot proceed."
        )
    contracts = contracts_with_forward(clean_contracts, forward_for_date)
    if contracts.is_empty():
        raise ValueError(
            f"No forward-enriched contracts were available for quote_date {quote_date}."
        )
    return contracts


def evaluate_node_forecast(
    model_name: str,
    quote_date: object,
    target_date: object,
    predicted_sampled_iv: np.ndarray,
    reconstructor: FittedReconstructor,
    target_nodes: pl.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    predicted_node_iv = reconstructor.predict(
        predicted_sampled_iv,
        target_nodes["m"].to_numpy().astype(np.float64),
        target_nodes["tau"].to_numpy().astype(np.float64),
    )
    metrics = compute_metrics(
        target_nodes["node_iv"].to_numpy().astype(np.float64),
        predicted_node_iv,
        target_nodes["node_vega"].to_numpy().astype(np.float64),
    )
    node_predictions = target_nodes.select(
        "quote_date", "root", "expiration", "strike", "m", "tau", "node_iv", "node_vega"
    ).with_columns(
        pl.lit(model_name).alias("model_name"),
        pl.lit(quote_date).alias("forecast_origin"),
        pl.lit(target_date).alias("target_date"),
        pl.Series("predicted_iv", predicted_node_iv),
    )
    arbitrage = reconstructor.arbitrage_diagnostics(predicted_sampled_iv)
    loss_row = {
        "model_name": model_name,
        "forecast_origin": quote_date,
        "target_date": target_date,
        "rmse_iv": metrics.rmse_iv,
        "mae_iv": metrics.mae_iv,
        "mape_iv_clipped": metrics.mape_iv_clipped,
        "vega_rmse_iv": metrics.vega_rmse_iv,
    }
    arbitrage_row = {
        "model_name": model_name,
        "forecast_origin": quote_date,
        "target_date": target_date,
        **arbitrage,
    }
    return loss_row, arbitrage_row, node_predictions


def evaluate_pricing_utility(
    model_name: str,
    quote_date: object,
    target_date: object,
    predicted_sampled_iv: np.ndarray,
    reconstructor: FittedReconstructor,
    target_contracts: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame]:
    predicted_iv = reconstructor.predict(
        predicted_sampled_iv,
        target_contracts["m"].to_numpy().astype(np.float64),
        target_contracts["tau"].to_numpy().astype(np.float64),
    )
    predicted_price = black_scholes_price(
        target_contracts["forward_price"].to_numpy().astype(np.float64),
        target_contracts["strike"].to_numpy().astype(np.float64),
        target_contracts["tau"].to_numpy().astype(np.float64),
        predicted_iv,
        target_contracts["discount_factor"].to_numpy().astype(np.float64),
        target_contracts["option_type"].to_numpy(),
    )
    utility = pricing_utility(
        predicted_price,
        target_contracts["mid_1545"].to_numpy().astype(np.float64),
        target_contracts["bid_1545"].to_numpy().astype(np.float64),
        target_contracts["ask_1545"].to_numpy().astype(np.float64),
    )
    priced_contracts = target_contracts.with_columns(
        pl.lit(model_name).alias("model_name"),
        pl.lit(quote_date).alias("forecast_origin"),
        pl.lit(target_date).alias("target_date"),
        pl.Series("predicted_iv", predicted_iv),
        pl.Series("predicted_price", predicted_price),
    )
    row = {
        "model_name": model_name,
        "forecast_origin": quote_date,
        "target_date": target_date,
        **utility,
    }
    return row, priced_contracts


def _bucket_contracts(
    current_contracts: pl.DataFrame,
) -> dict[str, tuple[str, object, object, float, str]]:
    buckets: dict[str, tuple[str, object, object, float, str]] = {}
    rules = [
        ("atm_call_30d", "C", 30.0 / 365.0),
        ("atm_put_30d", "P", 30.0 / 365.0),
        ("atm_call_91d", "C", 91.0 / 365.0),
    ]
    for bucket_name, option_type, target_tau in rules:
        candidate = (
            current_contracts.filter(pl.col("option_type") == option_type)
            .with_columns(
                (pl.col("tau") - target_tau).abs().alias("tau_distance"),
                pl.col("m").abs().alias("m_abs"),
            )
            .sort(["tau_distance", "m_abs"])
            .head(1)
        )
        if candidate.height == 1:
            row = candidate.row(0, named=True)
            buckets[bucket_name] = (
                str(row["root"]),
                row["expiration"],
                row["strike"],
                row["tau"],
                row["option_type"],
            )
    return buckets


def evaluate_hedged_pnl_utility(
    model_name: str,
    quote_date: object,
    target_date: object,
    current_contracts: pl.DataFrame,
    target_priced_contracts: pl.DataFrame,
) -> list[dict[str, Any]]:
    join_keys = ["root", "expiration", "strike", "option_type"]
    joined = current_contracts.join(
        target_priced_contracts.select(
            join_keys + ["mid_1545", "active_underlying_price_1545", "predicted_price"]
        ),
        on=join_keys,
        how="inner",
        suffix="_t1",
    )
    if joined.is_empty():
        return []
    overall = hedged_pnl_utility(
        joined["mid_1545"].to_numpy().astype(np.float64),
        joined["mid_1545_t1"].to_numpy().astype(np.float64),
        joined["predicted_price"].to_numpy().astype(np.float64),
        joined["delta_1545"].to_numpy().astype(np.float64),
        joined["active_underlying_price_1545"].to_numpy().astype(np.float64),
        joined["active_underlying_price_1545_t1"].to_numpy().astype(np.float64),
    )
    rows = [
        {
            "model_name": model_name,
            "forecast_origin": quote_date,
            "target_date": target_date,
            "bucket": "all",
            **overall,
        }
    ]
    buckets = _bucket_contracts(current_contracts)
    for bucket_name, (root, expiration, strike, _tau, option_type) in buckets.items():
        bucket_joined = joined.filter(
            (pl.col("root") == root)
            & (pl.col("expiration") == expiration)
            & (pl.col("strike") == strike)
            & (pl.col("option_type") == option_type)
        )
        if bucket_joined.is_empty():
            continue
        utility = hedged_pnl_utility(
            bucket_joined["mid_1545"].to_numpy().astype(np.float64),
            bucket_joined["mid_1545_t1"].to_numpy().astype(np.float64),
            bucket_joined["predicted_price"].to_numpy().astype(np.float64),
            bucket_joined["delta_1545"].to_numpy().astype(np.float64),
            bucket_joined["active_underlying_price_1545"].to_numpy().astype(np.float64),
            bucket_joined["active_underlying_price_1545_t1"].to_numpy().astype(np.float64),
        )
        rows.append(
            {
                "model_name": model_name,
                "forecast_origin": quote_date,
                "target_date": target_date,
                "bucket": bucket_name,
                **utility,
            }
        )
    return rows


def evaluate_straddle_signal(
    model_name: str,
    quote_date: object,
    target_date: object,
    current_contracts: pl.DataFrame,
    target_contracts: pl.DataFrame,
    predicted_sampled_iv: np.ndarray,
    current_sampled_surface: pl.DataFrame,
    reconstructor: FittedReconstructor,
    atm_grid_map: dict[int, str],
) -> list[dict[str, Any]]:
    matched = current_contracts.join(
        current_contracts.filter(pl.col("option_type") == "P").select(
            ["root", "expiration", "strike"]
        ),
        on=["root", "expiration", "strike"],
        how="inner",
    )
    if matched.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    for anchor_days in (30, 91):
        atm_candidates = (
            current_contracts.group_by(["root", "expiration", "strike"])
            .agg(
                pl.col("option_type").sort().alias("option_types"),
                pl.col("mid_1545").sum().alias("gross_premium"),
                pl.col("spread_1545").sum().alias("total_spread"),
                pl.col("tau").first().alias("tau"),
                pl.col("m").abs().mean().alias("m_abs"),
            )
            .filter(
                pl.col("option_types").list.contains("C")
                & pl.col("option_types").list.contains("P")
            )
            .with_columns((pl.col("tau") - anchor_days / 365.0).abs().alias("tau_distance"))
            .sort(["tau_distance", "m_abs"])
            .head(1)
        )
        if atm_candidates.is_empty():
            continue
        pair = atm_candidates.row(0, named=True)
        next_pair = target_contracts.filter(
            (pl.col("root") == pair["root"])
            & (pl.col("expiration") == pair["expiration"])
            & (pl.col("strike") == pair["strike"])
        )
        if next_pair.height < 2:
            continue
        grid_id = atm_grid_map[anchor_days]
        current_atm_iv = float(current_sampled_surface[f"iv_{grid_id}"][0])
        predicted_atm_iv = float(
            reconstructor.predict(
                predicted_sampled_iv,
                np.array([0.0], dtype=np.float64),
                np.array([anchor_days / 365.0], dtype=np.float64),
            )[0]
        )
        signal = 1.0 if predicted_atm_iv - current_atm_iv > 0 else -1.0
        current_premium = float(pair["gross_premium"])
        next_premium = float(next_pair["mid_1545"].sum())
        gross_return = signal * (next_premium - current_premium) / max(current_premium, 1e-12)
        entry_cost = 0.5 * float(pair["total_spread"]) / max(current_premium, 1e-12)
        exit_cost = 0.5 * float(next_pair["spread_1545"].sum()) / max(current_premium, 1e-12)
        net_return = gross_return - entry_cost - exit_cost
        rows.append(
            {
                "model_name": model_name,
                "forecast_origin": quote_date,
                "target_date": target_date,
                "root": pair["root"],
                "anchor_days": anchor_days,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
    return rows


def summarize_straddle(rows: pl.DataFrame) -> pl.DataFrame:
    summaries = []
    for anchor_days, group in rows.partition_by("anchor_days", as_dict=True).items():
        if isinstance(anchor_days, tuple):
            anchor_value = anchor_days[0]
        else:
            anchor_value = anchor_days
        utility = straddle_utility(
            group["net_return"].to_numpy().astype(np.float64),
            group["gross_return"].to_numpy().astype(np.float64),
        )
        summaries.append({"anchor_days": anchor_value, **utility})
    return pl.DataFrame(summaries)
