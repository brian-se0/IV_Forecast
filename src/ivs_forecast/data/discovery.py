from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from ivs_forecast.artifacts.hashing import sha256_file
from ivs_forecast.data.schema import reconcile_schema

RAW_FILE_PATTERN = re.compile(r"^UnderlyingOptionsEODCalcs_(\d{4}-\d{2}-\d{2})\.zip$")


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
    for path in sorted(root.glob("UnderlyingOptionsEODCalcs_*.zip")):
        if RAW_FILE_PATTERN.match(path.name):
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


def inventory_raw_files(root: Path, start_date: date, end_date: date) -> list[RawZipRecord]:
    records: list[RawZipRecord] = []
    for path in _iter_supported_files(root):
        trade_date = parse_trade_date_from_filename(path)
        if start_date <= trade_date <= end_date:
            records.append(inspect_zip(path))
    if not records:
        raise FileNotFoundError(
            f"No matching UnderlyingOptionsEODCalcs_*.zip files found under {root} for {start_date}..{end_date}."
        )
    return records


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


def header_schema_report(sample_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(sample_path) as handle:
        csv_members = [
            item.filename for item in handle.infolist() if item.filename.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(
                f"Expected exactly one CSV in {sample_path}, found {len(csv_members)}."
            )
        with handle.open(csv_members[0], "r") as csv_handle:
            sample = pl.read_csv(
                csv_handle,
                n_rows=100,
                infer_schema_length=100,
                try_parse_dates=True,
            )
    reconciliation = reconcile_schema(
        sample.columns, {name: str(dtype) for name, dtype in sample.schema.items()}
    )
    return reconciliation.to_dict()


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
