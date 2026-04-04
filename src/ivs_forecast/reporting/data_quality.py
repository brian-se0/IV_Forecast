from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from ivs_forecast.data.parity_forward import ForwardEstimationDiagnostics

STAGE_LOSS_SCHEMA: dict[str, pl.DataType] = {
    "quote_date": pl.Date,
    "target_date": pl.Date,
    "option_root": pl.String,
    "trading_day_index": pl.Int64,
    "has_next_trading_date": pl.Boolean,
    "subset_rows": pl.Int64,
    "clean_contract_rows": pl.Int64,
    "forward_term_rows": pl.Int64,
    "forward_total_expiries": pl.Int64,
    "forward_valid_expiries": pl.Int64,
    "forward_invalid_expiries": pl.Int64,
    "forward_invalid_reason_codes": pl.List(pl.String),
    "surface_node_count": pl.Int64,
    "valid_expiry_count": pl.Int64,
    "root_count": pl.Int64,
    "modeling_valid": pl.Boolean,
    "has_surface_state": pl.Boolean,
    "feature_exclusion_reason": pl.String,
    "feature_row_kept": pl.Boolean,
    "first_failed_stage": pl.String,
    "first_failure_reason_codes": pl.List(pl.String),
}


@dataclass
class DailyBuildDiagnostics:
    quote_date: date
    option_root: str
    subset_rows: int = 0
    clean_contract_rows: int = 0
    forward_term_rows: int = 0
    forward_total_expiries: int = 0
    forward_valid_expiries: int = 0
    forward_invalid_expiries: int = 0
    forward_invalid_reason_codes: list[str] = field(default_factory=list)
    surface_node_count: int | None = None
    valid_expiry_count: int | None = None
    root_count: int | None = None
    modeling_valid: bool = False
    has_surface_state: bool = False


def initialize_daily_build_diagnostics(
    trading_dates: list[date], option_root: str
) -> dict[date, DailyBuildDiagnostics]:
    return {
        trading_date: DailyBuildDiagnostics(
            quote_date=trading_date,
            option_root=option_root,
        )
        for trading_date in trading_dates
    }


def summarize_forward_diagnostics(
    diagnostics: list[ForwardEstimationDiagnostics],
) -> tuple[int, int, int, list[str]]:
    valid_expiries = 0
    invalid_reasons: Counter[str] = Counter()
    for item in diagnostics:
        if item.invalid_reason is None:
            valid_expiries += 1
            continue
        invalid_reasons[str(item.invalid_reason)] += 1
    total_expiries = len(diagnostics)
    invalid_expiries = total_expiries - valid_expiries
    return (
        total_expiries,
        valid_expiries,
        invalid_expiries,
        sorted(invalid_reasons),
    )


def build_stage_loss_by_date(
    trading_date_index: pl.DataFrame,
    daily_diagnostics: dict[date, DailyBuildDiagnostics],
    features_targets: pl.DataFrame,
    feature_exclusions: pl.DataFrame,
    min_surface_nodes: int,
    min_valid_expiries: int,
) -> pl.DataFrame:
    exclusion_by_date = {
        row["quote_date"]: str(row["exclusion_reason"])
        for row in feature_exclusions.iter_rows(named=True)
    }
    kept_feature_dates = set(features_targets["quote_date"].to_list())
    rows: list[dict[str, object]] = []
    for trading_row in trading_date_index.sort("quote_date").iter_rows(named=True):
        quote_date = trading_row["quote_date"]
        diagnostics = daily_diagnostics.get(
            quote_date,
            DailyBuildDiagnostics(
                quote_date=quote_date,
                option_root=str(trading_row["option_root"]),
            ),
        )
        feature_exclusion_reason = exclusion_by_date.get(quote_date)
        has_next_trading_date = trading_row["next_trading_date"] is not None
        first_failed_stage: str | None = None
        first_failure_reason_codes: list[str] = []
        if diagnostics.clean_contract_rows <= 0:
            first_failed_stage = "cleaning"
            first_failure_reason_codes = ["cleaning_eliminated_all_contracts"]
        elif diagnostics.forward_valid_expiries <= 0:
            first_failed_stage = "forward_estimation"
            first_failure_reason_codes = (
                diagnostics.forward_invalid_reason_codes.copy()
                if diagnostics.forward_invalid_reason_codes
                else ["no_forward_terms_after_parity"]
            )
        elif not diagnostics.modeling_valid:
            first_failed_stage = "node_construction"
            if (
                diagnostics.surface_node_count is not None
                and diagnostics.surface_node_count < min_surface_nodes
            ):
                first_failure_reason_codes.append("below_min_surface_nodes")
            if (
                diagnostics.valid_expiry_count is not None
                and diagnostics.valid_expiry_count < min_valid_expiries
            ):
                first_failure_reason_codes.append("below_min_valid_expiries")
            if diagnostics.root_count is not None and diagnostics.root_count != 1:
                first_failure_reason_codes.append("mixed_option_roots")
            if not first_failure_reason_codes:
                first_failure_reason_codes.append("node_panel_not_modeling_valid")
        elif not diagnostics.has_surface_state:
            first_failed_stage = "ssvi_calibration"
            first_failure_reason_codes = ["missing_ssvi_state"]
        elif feature_exclusion_reason is not None:
            first_failed_stage = "feature_target_formation"
            first_failure_reason_codes = [feature_exclusion_reason]
        feature_row_kept = quote_date in kept_feature_dates
        if feature_row_kept and feature_exclusion_reason is not None:
            raise ValueError(
                f"quote_date {quote_date.isoformat()} cannot be both a kept feature row "
                "and an excluded feature row."
            )
        if has_next_trading_date and first_failed_stage is None and not feature_row_kept:
            raise ValueError(
                "Feature-row accounting became inconsistent: a candidate origin date was neither "
                f"kept nor excluded for quote_date {quote_date.isoformat()}."
            )
        rows.append(
            {
                "quote_date": quote_date,
                "target_date": trading_row["next_trading_date"],
                "option_root": str(trading_row["option_root"]),
                "trading_day_index": int(trading_row["trading_day_index"]),
                "has_next_trading_date": has_next_trading_date,
                "subset_rows": diagnostics.subset_rows,
                "clean_contract_rows": diagnostics.clean_contract_rows,
                "forward_term_rows": diagnostics.forward_term_rows,
                "forward_total_expiries": diagnostics.forward_total_expiries,
                "forward_valid_expiries": diagnostics.forward_valid_expiries,
                "forward_invalid_expiries": diagnostics.forward_invalid_expiries,
                "forward_invalid_reason_codes": diagnostics.forward_invalid_reason_codes.copy(),
                "surface_node_count": diagnostics.surface_node_count,
                "valid_expiry_count": diagnostics.valid_expiry_count,
                "root_count": diagnostics.root_count,
                "modeling_valid": diagnostics.modeling_valid,
                "has_surface_state": bool(trading_row["has_surface_state"]),
                "feature_exclusion_reason": feature_exclusion_reason,
                "feature_row_kept": feature_row_kept,
                "first_failed_stage": first_failed_stage,
                "first_failure_reason_codes": first_failure_reason_codes,
            }
        )
    return pl.DataFrame(rows, schema=STAGE_LOSS_SCHEMA).sort("quote_date")


def build_stage_coverage_by_year(stage_loss_by_date: pl.DataFrame) -> list[dict[str, Any]]:
    coverage_rows: list[dict[str, Any]] = []
    by_year = stage_loss_by_date.with_columns(pl.col("quote_date").dt.year().alias("year"))
    for year, group in by_year.partition_by("year", as_dict=True).items():
        normalized_year = int(year[0] if isinstance(year, tuple) else year)
        failure_stage_counts = Counter(
            str(item)
            for item in group["first_failed_stage"].to_list()
            if item is not None
        )
        failure_reason_counts: Counter[str] = Counter()
        for reasons in group["first_failure_reason_codes"].to_list():
            for reason in reasons or []:
                failure_reason_counts[str(reason)] += 1
        coverage_rows.append(
            {
                "year": normalized_year,
                "raw_trading_days": group.height,
                "days_after_cleaning": int(group.filter(pl.col("clean_contract_rows") > 0).height),
                "days_with_forward_terms": int(group.filter(pl.col("forward_term_rows") > 0).height),
                "days_with_node_quality": int(group.filter(pl.col("surface_node_count").is_not_null()).height),
                "modeling_valid_days": int(group.filter(pl.col("modeling_valid")).height),
                "ssvi_state_days": int(group.filter(pl.col("has_surface_state")).height),
                "feature_candidate_days": int(group.filter(pl.col("has_next_trading_date")).height),
                "feature_rows": int(group.filter(pl.col("feature_row_kept")).height),
                "terminal_days": int(group.filter(~pl.col("has_next_trading_date")).height),
                "forward_invalid_expiry_count": int(group["forward_invalid_expiries"].sum()),
                "first_failed_stage_counts": dict(sorted(failure_stage_counts.items())),
                "first_failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            }
        )
    return sorted(coverage_rows, key=lambda item: int(item["year"]))


def build_forward_invalid_reasons_summary(
    diagnostics: list[ForwardEstimationDiagnostics],
) -> dict[str, Any]:
    overall_counts: Counter[str] = Counter()
    yearly_counts: dict[int, Counter[str]] = defaultdict(Counter)
    invalid_expiry_count = 0
    for item in diagnostics:
        if item.invalid_reason is None:
            continue
        invalid_expiry_count += 1
        reason = str(item.invalid_reason)
        year = date.fromisoformat(item.quote_date).year
        overall_counts[reason] += 1
        yearly_counts[year][reason] += 1
    return {
        "invalid_expiry_count": invalid_expiry_count,
        "overall_by_reason": dict(sorted(overall_counts.items())),
        "by_year": [
            {
                "year": year,
                "invalid_expiry_count": int(sum(counts.values())),
                "reason_counts": dict(sorted(counts.items())),
            }
            for year, counts in sorted(yearly_counts.items())
        ],
    }


def build_benchmark_contract(
    raw_corpus_contract: dict[str, Any],
    trading_date_index: pl.DataFrame,
    ssvi_state: pl.DataFrame,
    features_targets: pl.DataFrame,
    feature_exclusions: pl.DataFrame,
    minimum_history_days: int,
) -> dict[str, Any]:
    exclusion_counts = (
        {
            str(row["exclusion_reason"]): int(row["count"])
            for row in feature_exclusions.group_by("exclusion_reason")
            .len(name="count")
            .iter_rows(named=True)
        }
        if not feature_exclusions.is_empty()
        else {}
    )

    def _window(frame: pl.DataFrame, column: str) -> dict[str, Any]:
        if frame.is_empty():
            return {"start_date": None, "end_date": None, "row_count": 0}
        start_date = frame.select(pl.col(column).min()).item()
        end_date = frame.select(pl.col(column).max()).item()
        return {
            "start_date": start_date.isoformat() if start_date is not None else None,
            "end_date": end_date.isoformat() if end_date is not None else None,
            "row_count": frame.height,
        }

    return {
        "raw_window": raw_corpus_contract["window_coverage"],
        "forecastable_window": {
            "history_window_days": minimum_history_days,
            "raw_trading_days": trading_date_index.height,
            "feature_candidate_days": int(
                trading_date_index.filter(pl.col("next_trading_date").is_not_null()).height
            ),
            "ssvi_state_window": _window(ssvi_state, "quote_date"),
            "feature_origin_window": _window(features_targets, "quote_date"),
            "feature_target_window": _window(features_targets, "target_date"),
            "feature_exclusion_count": feature_exclusions.height,
            "feature_exclusion_counts": exclusion_counts,
        },
    }
