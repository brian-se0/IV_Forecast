from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from tests.fixtures.synthetic_vendor import write_synthetic_vendor_dataset
from typer.testing import CliRunner

from ivs_forecast.artifacts.manifests import build_stage_manifest, write_json, write_yaml
from ivs_forecast.cli import app

runner = CliRunner()


def _write_config(
    tmp_path: Path,
    raw_root: Path,
    artifact_root: Path,
    run_id: str | None,
) -> Path:
    runtime = {"seed": 42, "overwrite": False, "require_exact_window_coverage": False}
    if run_id is not None:
        runtime["run_id"] = run_id
    config = {
        "paths": {"raw_data_root": str(raw_root), "artifact_root": str(artifact_root)},
        "study": {
            "underlying_symbol": "^SPX",
            "option_root": "SPX",
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
        "runtime": runtime,
    }
    config_name = run_id if run_id is not None else "generated_run_id"
    config_path = tmp_path / f"{config_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_exportable_run_tree(run_root: Path) -> None:
    top_level_files = [
        "raw_inventory.parquet",
        "raw_inventory.json",
        "raw_corpus_contract.json",
        "vendor_schema_reconciliation.json",
        "data_audit_report.md",
        "clean_contracts_index.parquet",
        "forward_terms.parquet",
        "surface_nodes_index.parquet",
        "surface_date_quality.parquet",
        "ssvi_state.parquet",
        "ssvi_fit_diagnostics.parquet",
        "ssvi_certification.parquet",
        "trading_date_index.parquet",
        "feature_row_exclusions.parquet",
        "settlement_convention.json",
        "features_targets.parquet",
        "stage_loss_by_date.parquet",
        "stage_coverage_by_year.json",
        "forward_invalid_reasons.json",
        "benchmark_contract.json",
        "split_manifest.json",
        "selected_model_configs.json",
        "loss_panel.parquet",
        "forecast_ssvi_certification.parquet",
        "pricing_utility.parquet",
        "hedged_pnl_utility.parquet",
        "straddle_signal_utility.parquet",
        "dm_tests.json",
        "mcs_results.json",
        "summary.md",
        "artifact_contract_version.json",
    ]
    for relative_path in top_level_files:
        path = run_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
    config_dump = {"study": {"underlying_symbol": "^SPX", "option_root": "SPX"}}
    manifests_root = run_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    for stage_name in ("verify_data", "build_data", "run"):
        manifest = build_stage_manifest(
            stage_name=stage_name,
            config_dump=config_dump,
            global_seed=42,
            primary_artifact_paths=[],
            upstream_paths=[],
        )
        write_json(manifests_root / f"{stage_name}_manifest.json", manifest.model_dump())
        write_yaml(manifests_root / f"{stage_name}_resolved_config.yaml", config_dump)
    for model_name in ("state_last", "state_var1", "ssvi_tcn_direct"):
        model_root = run_root / "models" / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        for relative_name in (
            "forecast_ssvi_state.parquet",
            "forecast_node_panel.parquet",
            "forecast_contract_panel.parquet",
            "selected_params.json",
            "model_artifact.json",
        ):
            (model_root / relative_name).write_text(relative_name, encoding="utf-8")
    (run_root / "models" / "ssvi_tcn_direct" / "model_checkpoint.pt").write_text(
        "checkpoint",
        encoding="utf-8",
    )


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
        "raw_corpus_contract.json",
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
        "ssvi_state.parquet",
        "ssvi_fit_diagnostics.parquet",
        "ssvi_certification.parquet",
        "trading_date_index.parquet",
        "feature_row_exclusions.parquet",
        "settlement_convention.json",
        "features_targets.parquet",
        "stage_loss_by_date.parquet",
        "stage_coverage_by_year.json",
        "forward_invalid_reasons.json",
        "benchmark_contract.json",
    ):
        assert (run_dir / name).exists(), name
    legacy_surface_name = "_".join(["sampled", "surface", "wide.parquet"])
    assert not (run_dir / legacy_surface_name).exists()
    assert not (run_dir / "grid_definition.parquet").exists()
    assert (run_dir / "clean_contracts").is_dir()
    assert (run_dir / "surface_nodes").is_dir()
    assert list((run_dir / "clean_contracts").glob("year=*/*.parquet"))
    assert list((run_dir / "surface_nodes").glob("year=*/*.parquet"))
    assert (run_dir / "manifests" / "build_data_manifest.json").exists()


def test_cli_build_data_without_explicit_run_id_uses_single_run_directory(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=35)
    config_path = _write_config(tmp_path, raw_root, artifact_root, None)
    result = runner.invoke(app, ["build-data", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    run_dirs = list((artifact_root / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "raw_inventory.parquet").exists()
    assert (run_dir / "features_targets.parquet").exists()
    assert list((run_dir / "subset").glob("underlying_key=*/*/year=*/*.parquet"))
    assert (run_dir / "manifests" / "verify_data_manifest.json").exists()
    assert (run_dir / "manifests" / "build_data_manifest.json").exists()


def test_cli_export_bundle_writes_evidence_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    output = tmp_path / "bundle.zip"
    run_dir.mkdir()
    _write_exportable_run_tree(run_dir)
    monkeypatch.setattr(
        "ivs_forecast.artifacts.bundles.resolve_git_worktree_status",
        lambda: {
            "repository_root": "D:/IV_Forecast",
            "dirty": False,
            "status_lines": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output.exists()
    import json
    import zipfile

    with zipfile.ZipFile(output) as handle:
        evidence = json.loads(handle.read("evidence_manifest.json"))
    assert evidence["git_commit"]
    assert evidence["git_worktree_dirty"] is False


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
        "raw_corpus_contract.json",
        "vendor_schema_reconciliation.json",
        "data_audit_report.md",
        "clean_contracts_index.parquet",
        "forward_terms.parquet",
        "surface_nodes_index.parquet",
        "surface_date_quality.parquet",
        "ssvi_state.parquet",
        "ssvi_fit_diagnostics.parquet",
        "ssvi_certification.parquet",
        "trading_date_index.parquet",
        "feature_row_exclusions.parquet",
        "settlement_convention.json",
        "features_targets.parquet",
        "stage_loss_by_date.parquet",
        "stage_coverage_by_year.json",
        "forward_invalid_reasons.json",
        "benchmark_contract.json",
        "split_manifest.json",
        "loss_panel.parquet",
        "forecast_ssvi_certification.parquet",
        "pricing_utility.parquet",
        "hedged_pnl_utility.parquet",
        "straddle_signal_utility.parquet",
        "dm_tests.json",
        "mcs_results.json",
        "summary.md",
        "artifact_contract_version.json",
        "bundle_manifest.json",
    ]
    for name in mandatory:
        assert (run_dir / name).exists(), name
    for model_name in ("state_last", "state_var1", "ssvi_tcn_direct"):
        model_root = run_dir / "models" / model_name
        assert (model_root / "forecast_ssvi_state.parquet").exists()
        assert (model_root / "forecast_node_panel.parquet").exists()
        assert (model_root / "forecast_contract_panel.parquet").exists()
    for name in (
        "verify_data_manifest.json",
        "build_data_manifest.json",
        "run_manifest.json",
    ):
        assert (run_dir / "manifests" / name).exists(), name
    report = runner.invoke(app, ["report", "--run-dir", str(run_dir)])
    assert report.exit_code == 0, report.stdout
