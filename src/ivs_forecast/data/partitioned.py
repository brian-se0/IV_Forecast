from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ivs_forecast.artifacts.hashing import sha256_file


@dataclass(frozen=True)
class DatePartitionRecord:
    quote_date: date
    year: int
    file_path: str
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote_date": self.quote_date,
            "year": self.year,
            "file_path": self.file_path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


def normalize_quote_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Unsupported quote_date value: {value!r}")


def write_date_partition(dataset_root: Path, quote_date: date, frame: pl.DataFrame) -> DatePartitionRecord:
    if frame.is_empty():
        raise ValueError("Cannot write an empty date partition.")
    partition_path = dataset_root / f"year={quote_date.year}" / f"quote_date={quote_date.isoformat()}.parquet"
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(partition_path)
    return DatePartitionRecord(
        quote_date=quote_date,
        year=quote_date.year,
        file_path=str(partition_path),
        row_count=frame.height,
        sha256=sha256_file(partition_path),
    )


def partition_index_frame(records: list[DatePartitionRecord]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(
            schema={
                "quote_date": pl.Date,
                "year": pl.Int32,
                "file_path": pl.Utf8,
                "row_count": pl.Int64,
                "sha256": pl.Utf8,
            }
        )
    return pl.DataFrame([record.to_dict() for record in records]).sort("quote_date")


class DatePartitionIndex:
    def __init__(self, index_path: Path, label: str, cache_size: int = 4) -> None:
        self.index_path = index_path
        self.label = label
        self.cache_size = cache_size
        self.index_frame = pl.read_parquet(index_path).sort("quote_date")
        quote_dates = self.index_frame["quote_date"].to_list()
        if len(set(quote_dates)) != len(quote_dates):
            raise ValueError(f"{label} index contains duplicate quote_date rows.")
        self._paths_by_date = {
            row["quote_date"]: Path(row["file_path"])
            for row in self.index_frame.iter_rows(named=True)
        }
        self._cache: OrderedDict[date, pl.DataFrame] = OrderedDict()

    def available_dates(self) -> list[date]:
        return self.index_frame["quote_date"].to_list()

    def load_date(self, quote_date: object) -> pl.DataFrame:
        normalized = normalize_quote_date(quote_date)
        if normalized in self._cache:
            self._cache.move_to_end(normalized)
            return self._cache[normalized]
        path = self._paths_by_date.get(normalized)
        if path is None:
            raise FileNotFoundError(
                f"{self.label} partition is missing for quote_date {normalized.isoformat()}."
            )
        frame = pl.read_parquet(path)
        self._cache[normalized] = frame
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame

    def load_many(self, quote_dates: list[object]) -> pl.DataFrame:
        if not quote_dates:
            raise ValueError(f"{self.label} load_many received no quote dates.")
        normalized_dates = [normalize_quote_date(value) for value in quote_dates]
        paths: list[str] = []
        for quote_date in normalized_dates:
            path = self._paths_by_date.get(quote_date)
            if path is None:
                raise FileNotFoundError(
                    f"{self.label} partition is missing for quote_date {quote_date.isoformat()}."
                )
            paths.append(str(path))
        frame = pl.read_parquet(paths)
        if "quote_date" in frame.columns:
            return frame.sort("quote_date")
        return frame
