from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from tests.fixtures.synthetic_vendor import write_synthetic_vendor_dataset
from typer.testing import CliRunner

from ivs_forecast.cli import app

runner = CliRunner()


def _write_config(tmp_path: Path, raw_root: Path, artifact_root: Path, run_id: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    config = {
        "paths": {"raw_data_root": str(raw_root), "artifact_root": str(artifact_root)},
        "study": {
            "underlying_symbol": "^SPX",
            "start_date": "2020-01-02",
            "end_date": "2020-04-30",
            "forecast_horizon_days": 1,
        },
        "split": {
            "validation_size": 10,
            "test_size": 10,
            "min_train_size": 20,
            "refit_frequency": 5,
        },
        "runtime": {"seed": 20260329, "overwrite": False, "run_id": run_id},
        "models": {
            "reconstructor_config": str(repo_root / "configs/models/reconstructor_smoke.yaml"),
            "xgboost_config": str(repo_root / "configs/models/xgboost_smoke.yaml"),
            "lstm_config": str(repo_root / "configs/models/lstm_smoke.yaml"),
        },
    }
    config_path = tmp_path / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_cli_verify_data_on_synthetic_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=5)
    config_path = _write_config(tmp_path, raw_root, artifact_root, "verify_only")
    result = runner.invoke(app, ["verify-data", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    run_dir = artifact_root / "runs" / "verify_only"
    for name in (
        "raw_inventory.parquet",
        "raw_inventory.json",
        "vendor_schema_reconciliation.json",
        "data_audit_report.md",
    ):
        assert (run_dir / name).exists(), name
    assert (run_dir / "manifests" / "verify_data_manifest.json").exists()


def test_cli_build_data_on_synthetic_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=35)
    config_path = _write_config(tmp_path, raw_root, artifact_root, "build_only")
    result = runner.invoke(app, ["build-data", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    run_dir = artifact_root / "runs" / "build_only"
    for name in (
        "clean_contracts_index.parquet",
        "forward_terms.parquet",
        "surface_nodes_index.parquet",
        "surface_date_quality.parquet",
        "grid_definition.parquet",
        "sampled_surface_wide.parquet",
        "features_targets.parquet",
    ):
        assert (run_dir / name).exists(), name
    assert (run_dir / "clean_contracts").is_dir()
    assert (run_dir / "surface_nodes").is_dir()
    assert list((run_dir / "clean_contracts").glob("year=*/*.parquet"))
    assert list((run_dir / "surface_nodes").glob("year=*/*.parquet"))
    assert (run_dir / "manifests" / "build_data_manifest.json").exists()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the canonical run path."
)
def test_cli_run_and_report_on_synthetic_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=70)
    config_path = _write_config(tmp_path, raw_root, artifact_root, "synthetic_run")
    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    run_dir = artifact_root / "runs" / "synthetic_run"
    mandatory = [
        "raw_inventory.parquet",
        "raw_inventory.json",
        "vendor_schema_reconciliation.json",
        "data_audit_report.md",
        "clean_contracts_index.parquet",
        "forward_terms.parquet",
        "surface_nodes_index.parquet",
        "surface_date_quality.parquet",
        "sampled_surface_wide.parquet",
        "features_targets.parquet",
        "split_manifest.json",
        "reconstructor_model.pt",
        "loss_panel.parquet",
        "arbitrage_panel.parquet",
        "pricing_utility.parquet",
        "hedged_pnl_utility.parquet",
        "straddle_signal_utility.parquet",
        "dm_tests.json",
        "mcs_results.json",
        "summary.md",
    ]
    for name in mandatory:
        assert (run_dir / name).exists(), name
    assert (run_dir / "clean_contracts").is_dir()
    assert (run_dir / "surface_nodes").is_dir()
    for name in (
        "verify_data_manifest.json",
        "build_data_manifest.json",
        "run_manifest.json",
    ):
        assert (run_dir / "manifests" / name).exists(), name
    report = runner.invoke(app, ["report", "--run-dir", str(run_dir)])
    assert report.exit_code == 0, report.stdout
