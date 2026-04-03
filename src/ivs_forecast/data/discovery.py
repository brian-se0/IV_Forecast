from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import polars as pl

from ivs_forecast.artifacts.hashing import sha256_file
from ivs_forecast.data.early_closes import early_close_dates_in_range, load_early_close_calendar
from ivs_forecast.data.schema import (
    CALCS_REQUIRED_COLUMNS,
    CANONICAL_REQUIRED_COLUMNS,
    CSV_SCHEMA_OVERRIDES,
    DOCUMENTED_COLUMNS,
    reconcile_schema,
)

RAW_FILE_PATTERN = re.compile(r"^UnderlyingOptionsEODCalcs_(\d{4}-\d{2}-\d{2})\.zip$")
QUOTE_ONLY_PATTERN = re.compile(r"^UnderlyingOptionsEODQuotes_(\d{4}-\d{2}-\d{2})\.zip$")
GROUPED_CALCS_PATTERN = re.compile(r"^UnderlyingOptionsEODCalcs_(\d{4}|\d{4}-\d{2})\.zip$")
GROUPED_QUOTES_PATTERN = re.compile(r"^UnderlyingOptionsEODQuotes_(\d{4}|\d{4}-\d{2})\.zip$")


@dataclass(frozen=True)
class RawZipRecord:
    path: Path
    trade_date: date
    file_size: int
    sha256: str
    csv_member_name: str
    readable: bool
    csv_member_count: int


def parse_trade_date_from_filename(path: Path) -> date:
    match = RAW_FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unsupported raw filename: {path.name}")
    return date.fromisoformat(match.group(1))


def _iter_supported_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("UnderlyingOptionsEODCalcs_*.zip")):
        if RAW_FILE_PATTERN.match(path.name):
            yield path


def _iter_quote_only_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("UnderlyingOptionsEODQuotes_*.zip")):
        if QUOTE_ONLY_PATTERN.match(path.name):
            yield path


def _iter_grouped_archives(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("UnderlyingOptionsEODCalcs_*.zip")):
        if GROUPED_CALCS_PATTERN.match(path.name) and not RAW_FILE_PATTERN.match(path.name):
            yield path


def _iter_grouped_quote_only_archives(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("UnderlyingOptionsEODQuotes_*.zip")):
        if GROUPED_QUOTES_PATTERN.match(path.name) and not QUOTE_ONLY_PATTERN.match(path.name):
            yield path


def inspect_zip(path: Path) -> RawZipRecord:
    trade_date = parse_trade_date_from_filename(path)
    readable = False
    csv_member_name = ""
    csv_member_count = 0
    try:
        with zipfile.ZipFile(path) as handle:
            csv_members = [
                item.filename
                for item in handle.infolist()
                if item.filename.lower().endswith(".csv")
            ]
            csv_member_count = len(csv_members)
            if csv_member_count == 1:
                csv_member_name = csv_members[0]
                readable = True
    except zipfile.BadZipFile:
        readable = False
    return RawZipRecord(
        path=path,
        trade_date=trade_date,
        file_size=path.stat().st_size,
        sha256=sha256_file(path),
        csv_member_name=csv_member_name,
        readable=readable,
        csv_member_count=csv_member_count,
    )


def requested_window_coverage(
    records: list[RawZipRecord],
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    observed_start = min((record.trade_date for record in records), default=None)
    observed_end = max((record.trade_date for record in records), default=None)
    matches_requested_window = observed_start == start_date and observed_end == end_date
    prefix_gap = observed_start is not None and observed_start > start_date
    suffix_gap = observed_end is not None and observed_end < end_date
    if matches_requested_window:
        coverage_status = "exact"
    elif prefix_gap and suffix_gap:
        coverage_status = "short_both"
    elif prefix_gap:
        coverage_status = "short_prefix"
    elif suffix_gap:
        coverage_status = "short_suffix"
    else:
        coverage_status = "mismatch"
    return {
        "matches_requested_window": matches_requested_window,
        "coverage_status": coverage_status,
        "requested_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "observed_window": {
            "start_date": observed_start.isoformat() if observed_start else None,
            "end_date": observed_end.isoformat() if observed_end else None,
        },
        "missing_prefix_range": (
            {
                "start_date": start_date.isoformat(),
                "end_date": (observed_start - timedelta(days=1)).isoformat(),
            }
            if prefix_gap
            else None
        ),
        "missing_suffix_range": (
            {
                "start_date": (observed_end + timedelta(days=1)).isoformat(),
                "end_date": end_date.isoformat(),
            }
            if suffix_gap
            else None
        ),
    }


def require_exact_window_coverage(coverage: dict[str, Any]) -> None:
    if coverage["matches_requested_window"]:
        return
    raise ValueError(
        "Configured study window does not exactly match the observed raw daily ZIP coverage. "
        f"Requested {coverage['requested_window']['start_date']}..{coverage['requested_window']['end_date']} "
        f"but observed {coverage['observed_window']['start_date']}..{coverage['observed_window']['end_date']}. "
        f"Coverage status: {coverage['coverage_status']}."
    )


def raw_corpus_contract(
    root: Path,
    records: list[RawZipRecord],
    start_date: date,
    end_date: date,
    option_root: str,
) -> dict[str, Any]:
    grouped_archives = [path for path in _iter_grouped_archives(root)]
    relative_dirs = {
        str(record.path.parent.relative_to(root)) if record.path.parent != root else "."
        for record in records
    }
    grouping_mode = "flat_daily" if relative_dirs in [set(), {"."}] else "nested_daily"
    coverage = requested_window_coverage(records, start_date, end_date)
    return {
        "raw_data_root": str(root),
        "requested_window": coverage["requested_window"],
        "observed_window": coverage["observed_window"],
        "window_coverage": coverage,
        "daily_zip_count": len(records),
        "grouping_mode": grouping_mode,
        "relative_parent_directories": sorted(relative_dirs),
        "selected_option_root": option_root,
        "unsupported_grouped_archives": [str(path.relative_to(root)) for path in grouped_archives],
        "notes": [
            "Recursive discovery supports canonical daily UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip files.",
            "Daily files may be stored flat or in nested folders under the configured raw_data_root.",
            "Monthly/yearly grouped archives are detected and reported explicitly.",
        ],
    }


def inventory_raw_files(root: Path, start_date: date, end_date: date) -> list[RawZipRecord]:
    records: list[RawZipRecord] = []
    for path in _iter_supported_files(root):
        trade_date = parse_trade_date_from_filename(path)
        if start_date <= trade_date <= end_date:
            records.append(inspect_zip(path))
    if records:
        return records
    grouped_archives = [path for path in _iter_grouped_archives(root)]
    if grouped_archives:
        raise FileNotFoundError(
            "Only grouped monthly/yearly calcs archives were found in the configured window. "
            "The runtime currently requires canonical daily UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip "
            "files, but it will discover them recursively under nested folders once present. "
            f"Examples: {[str(path) for path in grouped_archives[:3]]}"
        )
    quote_only = [
        path
        for path in _iter_quote_only_files(root)
        if start_date
        <= date.fromisoformat(QUOTE_ONLY_PATTERN.match(path.name).group(1))
        <= end_date
    ]
    if quote_only:
        raise FileNotFoundError(
            "Only quote-only vendor ZIPs were found in the configured window. "
            "The v1 pipeline requires UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip files."
        )
    grouped_quote_only = [path for path in _iter_grouped_quote_only_archives(root)]
    if grouped_quote_only:
        raise FileNotFoundError(
            "Only grouped monthly/yearly quote-only archives were found in the configured window. "
            "The v1 pipeline requires daily calcs-included files."
        )
    raise FileNotFoundError(
        f"No matching UnderlyingOptionsEODCalcs_*.zip files found under {root} for "
        f"{start_date}..{end_date}."
    )


def raw_inventory_frame(records: list[RawZipRecord]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "file_path": [str(item.path) for item in records],
            "trade_date": [item.trade_date for item in records],
            "file_size": [item.file_size for item in records],
            "sha256": [item.sha256 for item in records],
            "csv_member_name": [item.csv_member_name for item in records],
            "csv_member_count": [item.csv_member_count for item in records],
            "readable": [item.readable for item in records],
        }
    ).sort("trade_date")


def _read_zip_csv(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as handle:
        csv_members = [
            item.filename for item in handle.infolist() if item.filename.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(f"Expected exactly one CSV in {path}, found {len(csv_members)}.")
        with handle.open(csv_members[0], "r") as csv_handle:
            return pl.read_csv(
                csv_handle,
                schema_overrides=CSV_SCHEMA_OVERRIDES,
                infer_schema_length=500,
                try_parse_dates=True,
                ignore_errors=False,
            )


def _null_rate_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def _selected_underlying_caveats(
    frame: pl.DataFrame,
    underlying_symbol: str,
    option_root: str,
) -> dict[str, Any]:
    if "underlying_symbol" not in frame.columns:
        return {
            "selected_underlying_rows": 0,
            "zero_or_missing_underlying_bid_ask_rows": 0,
            "positive_active_underlying_price_rows": 0,
            "distinct_root_count": 0,
            "root_row_counts": {},
            "mixed_root_date_count": 0,
            "mixed_root_dates": [],
            "selected_option_root_rows": 0,
            "selected_option_root_present": False,
            "root_counts_by_date": [],
        }
    subset = frame.filter(pl.col("underlying_symbol") == underlying_symbol)
    if subset.is_empty():
        return {
            "selected_underlying_rows": 0,
            "zero_or_missing_underlying_bid_ask_rows": 0,
            "positive_active_underlying_price_rows": 0,
            "distinct_root_count": 0,
            "root_row_counts": {},
            "mixed_root_date_count": 0,
            "mixed_root_dates": [],
            "selected_option_root_rows": 0,
            "selected_option_root_present": False,
            "root_counts_by_date": [],
        }
    if {"underlying_bid_1545", "underlying_ask_1545"}.issubset(subset.columns):
        zero_or_missing = subset.select(
            (
                pl.col("underlying_bid_1545").is_null()
                | (pl.col("underlying_bid_1545") <= 0)
                | pl.col("underlying_ask_1545").is_null()
                | (pl.col("underlying_ask_1545") <= 0)
            )
            .sum()
            .alias("count")
        ).item()
    else:
        zero_or_missing = subset.height
    positive_active = (
        subset.select(
            (
                pl.col("active_underlying_price_1545").is_not_null()
                & (pl.col("active_underlying_price_1545") > 0)
            )
            .sum()
            .alias("count")
        ).item()
        if "active_underlying_price_1545" in subset.columns
        else 0
    )
    root_row_counts = {
        str(row["root"]): int(row["row_count"])
        for row in subset.group_by("root").len(name="row_count").sort("root").iter_rows(named=True)
    }
    mixed_root_dates_frame = (
        subset.group_by("quote_date")
        .agg(pl.col("root").n_unique().alias("root_count"))
        .filter(pl.col("root_count") > 1)
        .sort("quote_date")
    )
    mixed_root_dates = [
        value.isoformat() for value in mixed_root_dates_frame["quote_date"].to_list()
    ]
    root_counts_by_date = []
    for quote_date, partition in subset.partition_by("quote_date", as_dict=True).items():
        normalized_date = quote_date[0] if isinstance(quote_date, tuple) else quote_date
        root_counts = {
            str(row["root"]): int(row["row_count"])
            for row in partition.group_by("root").len(name="row_count").sort("root").iter_rows(named=True)
        }
        root_counts_by_date.append(
            {
                "quote_date": normalized_date.isoformat(),
                "root_counts": root_counts,
                "selected_option_root_rows": int(root_counts.get(option_root, 0)),
                "selected_option_root_present": option_root in root_counts,
            }
        )
    return {
        "selected_underlying_rows": subset.height,
        "zero_or_missing_underlying_bid_ask_rows": int(zero_or_missing),
        "positive_active_underlying_price_rows": int(positive_active),
        "distinct_root_count": len(root_row_counts),
        "root_row_counts": root_row_counts,
        "mixed_root_date_count": mixed_root_dates_frame.height,
        "mixed_root_dates": mixed_root_dates,
        "selected_option_root_rows": int(root_row_counts.get(option_root, 0)),
        "selected_option_root_present": option_root in root_row_counts,
        "root_counts_by_date": sorted(root_counts_by_date, key=lambda item: item["quote_date"]),
    }


def audit_vendor_corpus(
    records: list[RawZipRecord],
    underlying_symbol: str,
    option_root: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    invalid_zip_files = [
        str(record.path) for record in records if (not record.readable or record.csv_member_count != 1)
    ]
    if invalid_zip_files:
        raise ValueError(
            "Raw ZIP validation failed because some files were unreadable or did not contain exactly "
            f"one CSV member: {invalid_zip_files[:5]}"
        )

    reference_columns: list[str] | None = None
    observed_columns_union: set[str] = set()
    missing_required_union: set[str] = set()
    extra_columns_union: set[str] = set()
    files_with_header_anomalies: list[str] = []
    files_with_quote_date_mismatch: list[str] = []
    files_with_missing_required: dict[str, list[str]] = {}
    files_with_missing_calcs: dict[str, list[str]] = {}
    files_with_extra_columns: dict[str, list[str]] = {}
    dtype_counts: dict[str, Counter[str]] = defaultdict(Counter)
    null_rates: dict[str, list[float]] = defaultdict(list)
    selected_underlying_rows = 0
    zero_or_missing_underlying_bid_ask_rows = 0
    positive_active_underlying_price_rows = 0
    root_row_counts_total: Counter[str] = Counter()
    mixed_root_dates: set[str] = set()
    file_reports: list[dict[str, Any]] = []
    root_coverage_by_date: list[dict[str, Any]] = []

    for record in records:
        frame = _read_zip_csv(record.path)
        reconciliation = reconcile_schema(
            frame.columns, {name: str(dtype) for name, dtype in frame.schema.items()}
        )
        quote_dates = (
            frame["quote_date"].drop_nulls().unique().sort().to_list() if "quote_date" in frame.columns else []
        )
        quote_date_match = (
            "quote_date" in frame.columns
            and frame["quote_date"].null_count() == 0
            and quote_dates == [record.trade_date]
        )
        if not quote_date_match:
            files_with_quote_date_mismatch.append(str(record.path))
        if reference_columns is None:
            reference_columns = list(frame.columns)
        header_anomaly = list(frame.columns) != reference_columns
        if header_anomaly or reconciliation.missing_required_columns or reconciliation.extra_columns:
            files_with_header_anomalies.append(str(record.path))
        if reconciliation.missing_required_columns:
            files_with_missing_required[str(record.path)] = reconciliation.missing_required_columns
        missing_calcs = [
            column for column in CALCS_REQUIRED_COLUMNS if column not in set(frame.columns)
        ]
        if missing_calcs:
            files_with_missing_calcs[str(record.path)] = missing_calcs
        if reconciliation.extra_columns:
            files_with_extra_columns[str(record.path)] = reconciliation.extra_columns
        observed_columns_union.update(frame.columns)
        missing_required_union.update(reconciliation.missing_required_columns)
        extra_columns_union.update(reconciliation.extra_columns)
        for column, dtype in frame.schema.items():
            dtype_counts[column][str(dtype)] += 1
            null_rates[column].append(frame[column].null_count() / max(frame.height, 1))
        caveats = _selected_underlying_caveats(frame, underlying_symbol, option_root)
        selected_underlying_rows += int(caveats["selected_underlying_rows"])
        zero_or_missing_underlying_bid_ask_rows += int(caveats["zero_or_missing_underlying_bid_ask_rows"])
        positive_active_underlying_price_rows += int(caveats["positive_active_underlying_price_rows"])
        root_row_counts_total.update(caveats["root_row_counts"])
        mixed_root_dates.update(caveats["mixed_root_dates"])
        root_coverage_by_date.extend(caveats["root_counts_by_date"])
        file_reports.append(
            {
                "file_path": str(record.path),
                "trade_date": record.trade_date.isoformat(),
                "row_count": frame.height,
                "quote_date_values": [value.isoformat() for value in quote_dates],
                "quote_date_match": quote_date_match,
                "missing_required_columns": reconciliation.missing_required_columns,
                "missing_calcs_columns": missing_calcs,
                "extra_columns": reconciliation.extra_columns,
                "header_anomaly": header_anomaly,
                "inferred_dtypes": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        )

    if files_with_quote_date_mismatch:
        raise ValueError(
            "Filename date and quote_date contents disagreed for files including: "
            f"{files_with_quote_date_mismatch[:5]}"
        )
    if files_with_missing_calcs:
        raise ValueError(
            "Calcs-required 15:45 fields were missing in files including: "
            f"{list(files_with_missing_calcs)[:5]}"
        )
    if files_with_missing_required:
        raise ValueError(
            "Vendor schema reconciliation failed because required columns were missing in files "
            f"including: {list(files_with_missing_required)[:5]}"
        )

    zero_or_missing_fraction = (
        zero_or_missing_underlying_bid_ask_rows / selected_underlying_rows
        if selected_underlying_rows
        else 0.0
    )
    active_price_positive_fraction = (
        positive_active_underlying_price_rows / selected_underlying_rows
        if selected_underlying_rows
        else 0.0
    )
    early_close_calendar = load_early_close_calendar()
    early_close_dates = early_close_dates_in_range(start_date, end_date)
    coverage = requested_window_coverage(records, start_date, end_date)
    return {
        "pass_status": True,
        "study_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "underlying_symbol": underlying_symbol,
            "option_root": option_root,
        },
        "window_coverage": coverage,
        "raw_zip_count": len(records),
        "documented_columns": list(DOCUMENTED_COLUMNS),
        "observed_columns_union": sorted(observed_columns_union),
        "missing_required_columns": sorted(missing_required_union),
        "extra_columns": sorted(extra_columns_union),
        "files_with_header_anomalies": files_with_header_anomalies,
        "files_with_quote_date_mismatch": files_with_quote_date_mismatch,
        "files_with_missing_required_columns": files_with_missing_required,
        "files_with_missing_calcs_columns": files_with_missing_calcs,
        "files_with_extra_columns": files_with_extra_columns,
        "column_summaries": {
            column: {
                "observed_file_count": int(sum(dtype_counts[column].values())),
                "dtype_counts": dict(sorted(dtype_counts[column].items())),
                "null_rate_summary": _null_rate_summary(null_rates[column]),
                "required": column in CANONICAL_REQUIRED_COLUMNS,
                "calcs_required": column in CALCS_REQUIRED_COLUMNS,
                "documented": column in DOCUMENTED_COLUMNS,
            }
            for column in sorted(observed_columns_union)
        },
        "selected_underlying_caveats": {
            "underlying_symbol": underlying_symbol,
            "option_root": option_root,
            "selected_underlying_rows": selected_underlying_rows,
            "zero_or_missing_underlying_bid_ask_rows": zero_or_missing_underlying_bid_ask_rows,
            "zero_or_missing_underlying_bid_ask_fraction": float(zero_or_missing_fraction),
            "positive_active_underlying_price_rows": positive_active_underlying_price_rows,
            "positive_active_underlying_price_fraction": float(active_price_positive_fraction),
            "distinct_root_count": len(root_row_counts_total),
            "root_row_counts": dict(sorted(root_row_counts_total.items())),
            "mixed_root_date_count": len(mixed_root_dates),
            "mixed_root_dates": sorted(mixed_root_dates),
            "active_underlying_price_1545_usable": active_price_positive_fraction == 1.0,
        },
        "selected_root_coverage": {
            "option_root": option_root,
            "dates_in_window": len(records),
            "dates_with_option_root": sum(
                1 for item in root_coverage_by_date if item["selected_option_root_present"]
            ),
            "dates_missing_option_root": [
                item["quote_date"] for item in root_coverage_by_date if not item["selected_option_root_present"]
            ],
            "root_counts_by_date": sorted(root_coverage_by_date, key=lambda item: item["quote_date"]),
        },
        "early_close_audit": {
            "calendar_name": early_close_calendar.calendar_name,
            "coverage_start": early_close_calendar.coverage_start.isoformat(),
            "coverage_end": early_close_calendar.coverage_end.isoformat(),
            "dates_in_range": [item.isoformat() for item in early_close_dates],
            "count": len(early_close_dates),
        },
        "caveat_counts": {
            "header_anomaly_count": len(files_with_header_anomalies),
            "extra_column_file_count": len(files_with_extra_columns),
            "early_close_count": len(early_close_dates),
        },
        "notes": [
            "The audit is corpus-wide across the configured date window.",
            "Calcs-required semantics are enforced via the required 15:45 vendor fields.",
            "Early-close dates come from the checked-in manifest and do not alter the 1545 column contract.",
            "Option-root coverage is reported by date before any modeling artifacts are built.",
        ],
        "files": file_reports,
    }


def data_audit_markdown(report: dict[str, Any]) -> str:
    lines = ["# Data Audit Report", ""]
    window = report["study_window"]
    lines.append(
        f"Study window: `{window['start_date']}` to `{window['end_date']}` for "
        f"`{window['underlying_symbol']}` with option root `{window['option_root']}`."
    )
    lines.append(f"Audited raw ZIP count: `{report['raw_zip_count']}`.")
    lines.append("")
    lines.append("## Coverage")
    coverage = report["window_coverage"]
    lines.append(
        f"Coverage status: `{coverage['coverage_status']}`; exact match: `{coverage['matches_requested_window']}`."
    )
    lines.append(
        f"Requested window: `{coverage['requested_window']['start_date']}` to `{coverage['requested_window']['end_date']}`."
    )
    lines.append(
        f"Observed window: `{coverage['observed_window']['start_date']}` to `{coverage['observed_window']['end_date']}`."
    )
    if coverage["missing_prefix_range"] is not None:
        lines.append(
            "Missing prefix range: "
            f"`{coverage['missing_prefix_range']['start_date']}` to `{coverage['missing_prefix_range']['end_date']}`."
        )
    if coverage["missing_suffix_range"] is not None:
        lines.append(
            "Missing suffix range: "
            f"`{coverage['missing_suffix_range']['start_date']}` to `{coverage['missing_suffix_range']['end_date']}`."
        )
    lines.append("")
    lines.append("## Schema")
    lines.append(
        f"Missing required columns across corpus: `{report['missing_required_columns']}`."
    )
    lines.append(f"Extra columns across corpus: `{report['extra_columns']}`.")
    lines.append(
        f"Header anomaly files: `{report['caveat_counts']['header_anomaly_count']}`."
    )
    lines.append("")
    lines.append("## Selected Underlying Caveats")
    caveats = report["selected_underlying_caveats"]
    lines.append(f"Selected underlying rows: `{caveats['selected_underlying_rows']}`.")
    lines.append(
        "Zero or missing underlying bid/ask fraction: "
        f"`{caveats['zero_or_missing_underlying_bid_ask_fraction']:.6f}`."
    )
    lines.append(
        "Positive active-underlying-price fraction: "
        f"`{caveats['positive_active_underlying_price_fraction']:.6f}`."
    )
    lines.append(f"Distinct option roots: `{caveats['distinct_root_count']}`.")
    lines.append(f"Root row counts: `{caveats['root_row_counts']}`.")
    lines.append(f"Mixed-root dates: `{caveats['mixed_root_date_count']}`.")
    if caveats["mixed_root_dates"]:
        lines.append(f"Mixed-root quote dates: `{', '.join(caveats['mixed_root_dates'])}`.")
    lines.append(
        "Active underlying price usable: "
        f"`{caveats['active_underlying_price_1545_usable']}`."
    )
    lines.append("")
    lines.append("## Root Coverage")
    coverage = report["selected_root_coverage"]
    lines.append(
        f"Dates with configured root `{coverage['option_root']}`: "
        f"`{coverage['dates_with_option_root']}` / `{coverage['dates_in_window']}`."
    )
    if coverage["dates_missing_option_root"]:
        lines.append(
            "Dates missing configured root: "
            f"`{', '.join(coverage['dates_missing_option_root'])}`."
        )
    lines.append("")
    lines.append("## Early Closes")
    early_close = report["early_close_audit"]
    lines.append(
        f"Manifest half-day count in range: `{early_close['count']}`."
    )
    lines.append(
        "Manifest coverage: "
        f"`{early_close['coverage_start']}` to `{early_close['coverage_end']}`."
    )
    if early_close["dates_in_range"]:
        lines.append(f"Half-days: `{', '.join(early_close['dates_in_range'])}`.")
    lines.append("")
    lines.append("## Notes")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_inventory_json(path: Path, records: list[RawZipRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "file_path": str(item.path),
            "trade_date": item.trade_date.isoformat(),
            "file_size": item.file_size,
            "sha256": item.sha256,
            "csv_member_name": item.csv_member_name,
            "csv_member_count": item.csv_member_count,
            "readable": item.readable,
        }
        for item in records
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
