from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ivs_forecast.config import AppConfig
from ivs_forecast.data.early_closes import early_close_date_set
from ivs_forecast.data.time_to_settlement import (
    settlement_timestamp_eastern,
    snapshot_timestamp_eastern,
    year_fraction_act365,
)

CONTRACT_KEY = (
    "quote_date",
    "underlying_symbol",
    "root",
    "expiration",
    "strike",
    "option_type",
)


@dataclass(frozen=True)
class DuplicateSummary:
    exact_duplicates_removed: int
    conflicting_duplicate_keys: list[tuple[str, ...]]


def _validate_duplicates(frame: pl.DataFrame) -> tuple[pl.DataFrame, DuplicateSummary]:
    groups = defaultdict(list)
    for row in frame.iter_rows(named=True):
        key = tuple(str(row[column]) for column in CONTRACT_KEY)
        groups[key].append(row)
    kept_rows: list[dict[str, object]] = []
    exact_duplicates_removed = 0
    conflicting: list[tuple[str, ...]] = []
    for key, rows in groups.items():
        first = rows[0]
        if all(candidate == first for candidate in rows[1:]):
            kept_rows.append(first)
            exact_duplicates_removed += len(rows) - 1
        else:
            conflicting.append(key)
    if conflicting:
        raise ValueError(f"Conflicting duplicate contracts detected for keys: {conflicting[:5]}")
    return pl.DataFrame(kept_rows, schema=frame.schema), DuplicateSummary(
        exact_duplicates_removed=exact_duplicates_removed,
        conflicting_duplicate_keys=conflicting,
    )


def clean_contracts_day(
    frame: pl.DataFrame, config: AppConfig
) -> tuple[pl.DataFrame, DuplicateSummary]:
    filtered_to_contract = frame.filter(pl.col("underlying_symbol") == config.study.underlying_symbol).filter(
        pl.col("root") == config.study.option_root
    )
    if filtered_to_contract.is_empty():
        return _validate_duplicates(filtered_to_contract)
    quote_dates = filtered_to_contract["quote_date"].unique().to_list()
    if len(quote_dates) != 1:
        raise ValueError(
            "Each daily contract frame must contain exactly one quote_date before cleaning."
        )
    quote_date = quote_dates[0]
    is_early_close = quote_date in early_close_date_set(quote_date, quote_date)
    snapshot_ts = snapshot_timestamp_eastern(quote_date, is_early_close)
    filtered = (
        filtered_to_contract.with_columns(
            (
                pl.col("expiration").cast(pl.Date).cast(pl.Int32)
                - pl.col("quote_date").cast(pl.Date).cast(pl.Int32)
            ).alias("dte_days")
        )
        .filter(pl.col("option_type").is_in(["C", "P"]))
        .filter(pl.col("strike") > 0)
        .filter(pl.col("quote_date") < pl.col("expiration"))
        .filter(pl.col("dte_days") >= config.study.min_dte_days)
        .filter(pl.col("dte_days") <= config.study.max_dte_days)
        .filter(pl.col("bid_1545") > 0)
        .filter(pl.col("ask_1545") > 0)
        .filter(pl.col("ask_1545") >= pl.col("bid_1545"))
        .filter(pl.col("implied_volatility_1545") > 0)
        .filter(pl.col("vega_1545") > 0)
        .filter(pl.col("active_underlying_price_1545") > 0)
        .with_columns(
            pl.lit(config.study.option_root).alias("option_root"),
            pl.lit(is_early_close).alias("is_early_close"),
            (0.5 * (pl.col("bid_1545") + pl.col("ask_1545"))).alias("mid_1545"),
            (pl.col("ask_1545") - pl.col("bid_1545")).alias("spread_1545"),
        )
        .with_columns(
            (pl.col("spread_1545") / pl.max_horizontal(pl.col("mid_1545"), pl.lit(1e-6))).alias(
                "rel_spread_1545"
            ),
            pl.col("expiration")
            .map_elements(
                lambda expiration: year_fraction_act365(
                    snapshot_ts,
                    settlement_timestamp_eastern(
                        expiration,
                        config.study.option_root,
                        config.settlement,
                    ),
                ),
                return_dtype=pl.Float64,
            )
            .alias("tau"),
        )
    )
    return _validate_duplicates(filtered)


def clean_contracts_files(
    config: AppConfig, subset_paths: list[Path]
) -> tuple[pl.DataFrame, dict[str, int]]:
    frames: list[pl.DataFrame] = []
    exact_removed = 0
    for path in sorted(subset_paths):
        day_frame = pl.read_parquet(path)
        cleaned, duplicate_summary = clean_contracts_day(day_frame, config)
        exact_removed += duplicate_summary.exact_duplicates_removed
        frames.append(cleaned)
    if not frames:
        raise ValueError("No ingested subset parquet files were available for cleaning.")
    result = pl.concat(frames, how="vertical_relaxed").sort(
        ["quote_date", "expiration", "strike", "option_type"]
    )
    diagnostics = {
        "rows_after_cleaning": result.height,
        "exact_duplicates_removed": exact_removed,
        "dates_after_cleaning": result.select(pl.col("quote_date").n_unique()).item(),
    }
    return result, diagnostics
