from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from tests.fixtures.synthetic_vendor import write_synthetic_vendor_dataset
from typer.testing import CliRunner

from ivs_forecast.cli import app

runner = CliRunner()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the canonical run path."
)
def test_cli_run_and_report_on_synthetic_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=70)
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
        "runtime": {"seed": 20260329, "overwrite": False, "run_id": "synthetic_run"},
        "models": {
            "reconstructor_config": str(repo_root / "configs/models/reconstructor_smoke.yaml"),
            "xgboost_config": str(repo_root / "configs/models/xgboost_smoke.yaml"),
            "lstm_config": str(repo_root / "configs/models/lstm_smoke.yaml"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    run_dir = artifact_root / "runs" / "synthetic_run"
    mandatory = [
        "raw_inventory.parquet",
        "vendor_schema_reconciliation.json",
        "clean_contracts.parquet",
        "forward_terms.parquet",
        "surface_nodes.parquet",
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
    report = runner.invoke(app, ["report", "--run-dir", str(run_dir)])
    assert report.exit_code == 0, report.stdout
