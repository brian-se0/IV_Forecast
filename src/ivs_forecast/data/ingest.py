from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from ivs_forecast.config import AppConfig, underlying_to_key
from ivs_forecast.data.discovery import RawZipRecord
from ivs_forecast.data.schema import CANONICAL_REQUIRED_COLUMNS, CSV_SCHEMA_OVERRIDES


def _ingest_single_zip(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as handle:
        csv_members = [
            item.filename for item in handle.infolist() if item.filename.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(
                f"Expected exactly one CSV member in {path}, found {len(csv_members)}."
            )
        with handle.open(csv_members[0], "r") as csv_handle:
            frame = pl.read_csv(
                csv_handle,
                schema_overrides=CSV_SCHEMA_OVERRIDES,
                infer_schema_length=500,
                try_parse_dates=True,
                ignore_errors=False,
            )
    missing = sorted(set(CANONICAL_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return frame


def stream_ingest_selected_underlying(
    config: AppConfig,
    records: list[RawZipRecord],
) -> list[Path]:
    subset_root = config.subset_root
    subset_root.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    underlying_key = underlying_to_key(config.study.underlying_symbol)
    for record in records:
        frame = _ingest_single_zip(record.path).filter(
            pl.col("underlying_symbol") == config.study.underlying_symbol
        )
        year = record.trade_date.year
        day_path = subset_root / f"year={year}" / f"{record.trade_date.isoformat()}.parquet"
        day_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(day_path)
        written_paths.append(day_path)
    if not written_paths:
        raise ValueError(
            f"No rows found for {config.study.underlying_symbol} in the configured raw dataset."
        )
    expected_prefix = subset_root.parent / f"underlying_key={underlying_key}"
    if subset_root != expected_prefix:
        raise ValueError("Internal subset path mismatch.")
    return written_paths


def scan_subset_parquet(subset_root: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(subset_root / "**" / "*.parquet"))
