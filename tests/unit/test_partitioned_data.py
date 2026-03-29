from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ivs_forecast.data.partitioned import (
    DatePartitionIndex,
    partition_index_frame,
    write_date_partition,
)


def test_date_partition_index_loads_by_date(tmp_path) -> None:
    dataset_root = tmp_path / "clean_contracts"
    first = write_date_partition(
        dataset_root,
        date(2020, 1, 2),
        pl.DataFrame({"quote_date": [date(2020, 1, 2)], "value": [1]}),
    )
    second = write_date_partition(
        dataset_root,
        date(2020, 1, 3),
        pl.DataFrame({"quote_date": [date(2020, 1, 3)], "value": [2]}),
    )
    index_path = tmp_path / "clean_contracts_index.parquet"
    partition_index_frame([first, second]).write_parquet(index_path)
    index = DatePartitionIndex(index_path, "clean_contracts", cache_size=2)
    loaded = index.load_date(date(2020, 1, 3))
    assert loaded["value"][0] == 2
    merged = index.load_many([date(2020, 1, 2), date(2020, 1, 3)])
    assert merged.height == 2


def test_date_partition_index_rejects_missing_date(tmp_path) -> None:
    dataset_root = tmp_path / "surface_nodes"
    record = write_date_partition(
        dataset_root,
        date(2020, 1, 2),
        pl.DataFrame({"quote_date": [date(2020, 1, 2)], "value": [1]}),
    )
    index_path = tmp_path / "surface_nodes_index.parquet"
    partition_index_frame([record]).write_parquet(index_path)
    index = DatePartitionIndex(index_path, "surface_nodes")
    with pytest.raises(FileNotFoundError, match="missing for quote_date"):
        index.load_date(date(2020, 1, 3))
