from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ivs_forecast.artifacts.hashing import sha256_file
from ivs_forecast.artifacts.manifests import (
    StageManifest,
    resolve_git_worktree_status,
    write_json,
)

ARTIFACT_CONTRACT_VERSION = "2026-04-04"
EVIDENCE_MANIFEST_NAME = "evidence_manifest.json"
REQUIRED_STAGE_MANIFESTS: tuple[tuple[str, str], ...] = (
    ("verify_data", "manifests/verify_data_manifest.json"),
    ("build_data", "manifests/build_data_manifest.json"),
    ("run", "manifests/run_manifest.json"),
)
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


def _load_stage_manifests(run_root: Path) -> dict[str, tuple[StageManifest, Path]]:
    manifests: dict[str, tuple[StageManifest, Path]] = {}
    for stage_name, relative_path in REQUIRED_STAGE_MANIFESTS:
        manifest_path = run_root / relative_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Run directory is missing required stage manifest for evidence export under "
                f"{run_root}: {relative_path}"
            )
        manifest = StageManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        manifests[stage_name] = (manifest, manifest_path)
    return manifests


def _require_matching_stage_metadata(
    label: str,
    values_by_stage: dict[str, str],
) -> str:
    unique_values = {value for value in values_by_stage.values()}
    if len(unique_values) != 1:
        details = ", ".join(
            f"{stage_name}={value}" for stage_name, value in sorted(values_by_stage.items())
        )
        raise ValueError(f"Run stage manifests disagree on {label}: {details}")
    return next(iter(unique_values))


def resolve_run_bundle_evidence(run_root: Path) -> dict[str, Any]:
    manifests = _load_stage_manifests(run_root)
    git_commit = _require_matching_stage_metadata(
        "git_commit",
        {
            stage_name: manifest.git_commit
            for stage_name, (manifest, _manifest_path) in manifests.items()
        },
    )
    config_sha256 = _require_matching_stage_metadata(
        "config_sha256",
        {
            stage_name: manifest.config_sha256
            for stage_name, (manifest, _manifest_path) in manifests.items()
        },
    )
    run_manifest, _run_manifest_path = manifests["run"]
    return {
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "python_version": run_manifest.python_version,
        "package_versions": run_manifest.package_versions,
        "platform": run_manifest.platform,
        "cuda": run_manifest.cuda.model_dump(),
        "stage_manifests": {
            stage_name: {
                "relative_path": manifest_path.relative_to(run_root).as_posix(),
                "sha256": sha256_file(manifest_path),
                "created_at_utc": manifest.created_at_utc,
                "stage_name": manifest.stage_name,
            }
            for stage_name, (manifest, manifest_path) in manifests.items()
        },
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
    evidence = resolve_run_bundle_evidence(run_root)
    manifest_path = run_root / "bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_dir": str(run_root),
            "git_commit": evidence["git_commit"],
            "config_sha256": evidence["config_sha256"],
            "stage_manifests": evidence["stage_manifests"],
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
    allow_dirty_worktree: bool = False,
) -> Path:
    required_paths, optional_paths = resolve_run_bundle_contract(run_root, model_families)
    worktree_status = resolve_git_worktree_status()
    if worktree_status["dirty"] and not allow_dirty_worktree:
        raise ValueError(
            "Bundle export requires a clean git worktree. Commit or stash local changes, "
            "or rerun with allow_dirty_worktree=True to record the dirty state explicitly."
        )
    manifest_path = write_run_bundle_manifest(run_root, model_families)
    evidence_manifest = {
        **resolve_run_bundle_evidence(run_root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run_root),
        "repository_root": worktree_status["repository_root"],
        "git_worktree_dirty": worktree_status["dirty"],
        "dirty_worktree_allowed": allow_dirty_worktree,
        "git_status_porcelain": worktree_status["status_lines"],
    }
    export_paths = list(dict.fromkeys([*required_paths, *optional_paths, manifest_path]))
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Bundle output already exists: {destination}. Set overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in export_paths:
            handle.write(path, arcname=path.relative_to(run_root).as_posix())
        handle.writestr(
            EVIDENCE_MANIFEST_NAME,
            json.dumps(evidence_manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
    return destination
