from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ivs_forecast.artifacts.hashing import sha256_file
from ivs_forecast.artifacts.manifests import write_json

ARTIFACT_CONTRACT_VERSION = "2026-04-04"
RUN_BUNDLE_REQUIRED_TOP_LEVEL: tuple[str, ...] = (
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
    "manifests/verify_data_manifest.json",
    "manifests/verify_data_resolved_config.yaml",
    "manifests/build_data_manifest.json",
    "manifests/build_data_resolved_config.yaml",
    "manifests/run_manifest.json",
    "manifests/run_resolved_config.yaml",
)
RUN_BUNDLE_OPTIONAL_TOP_LEVEL: tuple[str, ...] = (
    "resolved_config.yaml",
    "bundle_manifest.json",
)


def _required_model_artifacts(model_name: str) -> tuple[str, ...]:
    return (
        f"models/{model_name}/forecast_ssvi_state.parquet",
        f"models/{model_name}/forecast_node_panel.parquet",
        f"models/{model_name}/forecast_contract_panel.parquet",
    )


def _optional_model_artifacts(model_name: str) -> tuple[str, ...]:
    artifacts = (
        f"models/{model_name}/selected_params.json",
        f"models/{model_name}/model_artifact.json",
    )
    if model_name == "ssvi_tcn_direct":
        return artifacts + (f"models/{model_name}/model_checkpoint.pt",)
    return artifacts


def _artifact_record(path: Path, run_root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(run_root).as_posix()
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_artifact_contract_version(run_root: Path) -> Path:
    path = run_root / "artifact_contract_version.json"
    write_json(
        path,
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_bundle_manifest_version": 1,
        },
    )
    return path


def resolve_run_bundle_contract(
    run_root: Path,
    model_families: list[str],
) -> tuple[list[Path], list[Path]]:
    required_paths = [run_root / relative for relative in RUN_BUNDLE_REQUIRED_TOP_LEVEL]
    optional_paths = [run_root / relative for relative in RUN_BUNDLE_OPTIONAL_TOP_LEVEL]
    for model_name in model_families:
        required_paths.extend(run_root / relative for relative in _required_model_artifacts(model_name))
        optional_paths.extend(run_root / relative for relative in _optional_model_artifacts(model_name))
    missing = [path.relative_to(run_root).as_posix() for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run directory is missing required bundle artifacts under "
            f"{run_root}: {missing}"
        )
    existing_optional = [path for path in optional_paths if path.exists()]
    return required_paths, existing_optional


def write_run_bundle_manifest(run_root: Path, model_families: list[str]) -> Path:
    required_paths, optional_paths = resolve_run_bundle_contract(run_root, model_families)
    manifest_path = run_root / "bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_dir": str(run_root),
            "required_artifacts": [
                _artifact_record(path, run_root) for path in required_paths
            ],
            "optional_artifacts": [
                _artifact_record(path, run_root) for path in optional_paths
            ],
        },
    )
    return manifest_path


def export_run_bundle(
    run_root: Path,
    destination: Path,
    model_families: list[str],
    overwrite: bool = False,
) -> Path:
    manifest_path = write_run_bundle_manifest(run_root, model_families)
    required_paths, optional_paths = resolve_run_bundle_contract(run_root, model_families)
    export_paths = list(dict.fromkeys([*required_paths, *optional_paths, manifest_path]))
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Bundle output already exists: {destination}. Set overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in export_paths:
            handle.write(path, arcname=path.relative_to(run_root).as_posix())
    return destination
