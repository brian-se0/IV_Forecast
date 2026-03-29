from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

CANONICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "underlying_symbol",
    "quote_date",
    "root",
    "expiration",
    "strike",
    "option_type",
    "bid_1545",
    "ask_1545",
    "active_underlying_price_1545",
    "implied_volatility_1545",
    "delta_1545",
    "vega_1545",
    "trade_volume",
    "open_interest",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "gamma_1545",
    "theta_1545",
    "rho_1545",
    "bid_eod",
    "ask_eod",
    "vwap",
    "open",
    "high",
    "low",
    "close",
    "implied_underlying_price_1545",
    "delivery_code",
    "bid_size_1545",
    "ask_size_1545",
    "bid_size_eod",
    "ask_size_eod",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "underlying_bid_eod",
    "underlying_ask_eod",
)

DOCUMENTED_COLUMNS: tuple[str, ...] = CANONICAL_REQUIRED_COLUMNS + OPTIONAL_COLUMNS

CSV_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "underlying_symbol": pl.Utf8,
    "quote_date": pl.Date,
    "root": pl.Utf8,
    "expiration": pl.Date,
    "strike": pl.Float64,
    "option_type": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "trade_volume": pl.Int64,
    "bid_size_1545": pl.Int64,
    "bid_1545": pl.Float64,
    "ask_size_1545": pl.Int64,
    "ask_1545": pl.Float64,
    "underlying_bid_1545": pl.Float64,
    "underlying_ask_1545": pl.Float64,
    "implied_underlying_price_1545": pl.Float64,
    "active_underlying_price_1545": pl.Float64,
    "implied_volatility_1545": pl.Float64,
    "delta_1545": pl.Float64,
    "gamma_1545": pl.Float64,
    "theta_1545": pl.Float64,
    "vega_1545": pl.Float64,
    "rho_1545": pl.Float64,
    "bid_size_eod": pl.Int64,
    "bid_eod": pl.Float64,
    "ask_size_eod": pl.Int64,
    "ask_eod": pl.Float64,
    "underlying_bid_eod": pl.Float64,
    "underlying_ask_eod": pl.Float64,
    "vwap": pl.Float64,
    "open_interest": pl.Int64,
    "delivery_code": pl.Utf8,
}


@dataclass(frozen=True)
class SchemaReconciliation:
    documented_columns: list[str]
    observed_columns: list[str]
    missing_required_columns: list[str]
    extra_columns: list[str]
    inferred_dtypes: dict[str, str]
    pass_status: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "documented_columns": self.documented_columns,
            "observed_columns": self.observed_columns,
            "missing_required_columns": self.missing_required_columns,
            "extra_columns": self.extra_columns,
            "inferred_dtypes": self.inferred_dtypes,
            "pass_status": self.pass_status,
            "notes": self.notes,
        }


def reconcile_schema(
    observed_columns: list[str],
    inferred_dtypes: dict[str, str],
) -> SchemaReconciliation:
    missing = sorted(set(CANONICAL_REQUIRED_COLUMNS) - set(observed_columns))
    extra = sorted(set(observed_columns) - set(DOCUMENTED_COLUMNS))
    notes = [
        "Canonical observed/vendor schema uses *_1545 fields.",
        "Any *_15453 references are treated as stale documentation aliases and not as required columns.",
    ]
    return SchemaReconciliation(
        documented_columns=list(DOCUMENTED_COLUMNS),
        observed_columns=observed_columns,
        missing_required_columns=missing,
        extra_columns=extra,
        inferred_dtypes=inferred_dtypes,
        pass_status=not missing,
        notes=notes,
    )
