from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from tests.fixtures.synthetic_vendor import write_synthetic_vendor_dataset

from ivs_forecast.config import AppConfig
from ivs_forecast.pipeline.build_data import build_data_stage


def _config(raw_root: Path, artifact_root: Path, run_id: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "paths": {"raw_data_root": str(raw_root), "artifact_root": str(artifact_root)},
            "study": {
                "underlying_symbol": "^SPX",
                "option_root": "SPX",
                "start_date": "2020-01-02",
                "end_date": "2020-04-30",
                "forecast_horizon_days": 1,
            },
            "split": {
                "validation_size": 5,
                "test_size": 5,
                "min_train_size": 5,
                "refit_frequency": 5,
            },
            "runtime": {"seed": 20260329, "overwrite": False, "run_id": run_id},
        }
    )


def test_build_data_stage_writes_direct_ssvi_artifacts(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=35)
    outputs = build_data_stage(_config(raw_root, artifact_root, "direct_ssvi_build"))
    run_root = outputs["run_root"]
    assert (run_root / "ssvi_state.parquet").exists()
    assert (run_root / "ssvi_fit_diagnostics.parquet").exists()
    assert (run_root / "ssvi_certification.parquet").exists()
    legacy_surface_name = "_".join(["sampled", "surface", "wide.parquet"])
    assert not (run_root / legacy_surface_name).exists()
    features = pl.read_parquet(run_root / "features_targets.parquet")
    assert "option_root" in features.columns
    assert "history_start_index" in features.columns
    assert "history_end_index" in features.columns
    assert "surface_state_row_index" in features.columns
    assert "target_state_row_index" in features.columns
    assert not any(column.startswith("x_curr_") for column in features.columns)
    assert not any(column.startswith("y_g") for column in features.columns)


def test_build_data_stage_fails_if_option_root_missing_for_any_date(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=5, option_root="SPXW")
    with pytest.raises(ValueError, match="configured option_root was absent"):
        build_data_stage(_config(raw_root, artifact_root, "missing_root"))
