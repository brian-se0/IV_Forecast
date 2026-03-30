from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml
from pydantic import BaseModel, Field

from ivs_forecast.artifacts.hashing import sha256_file, sha256_json


class UpstreamArtifact(BaseModel):
    path: str
    sha256: str


class CudaMetadata(BaseModel):
    available: bool
    torch_cuda_version: str | None
    cudnn_version: int | None
    device_count: int
    device_names: list[str] = Field(default_factory=list)


class StageManifest(BaseModel):
    stage_name: str
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)
    git_commit: str
    package_versions: dict[str, str]
    cuda: CudaMetadata
    global_seed: int
    device_by_model_family: dict[str, str] = Field(default_factory=dict)
    config_sha256: str
    primary_artifacts: list[UpstreamArtifact] = Field(default_factory=list)
    upstream_artifacts: list[UpstreamArtifact] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


RUNTIME_PACKAGE_NAMES: tuple[str, ...] = (
    "ivs-forecast",
    "numpy",
    "polars",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "rich",
    "scipy",
    "torch",
    "typer",
    "xgboost",
)


def resolve_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to resolve reproducibility metadata.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Unable to resolve git commit for reproducibility metadata.") from exc
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Git commit resolution returned an empty value.")
    return commit


def resolve_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in RUNTIME_PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required runtime package metadata is unavailable for {package_name}."
            ) from exc
    return versions


def resolve_cuda_metadata() -> CudaMetadata:
    available = torch.cuda.is_available()
    if not available:
        return CudaMetadata(
            available=False,
            torch_cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            device_count=0,
            device_names=[],
        )
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("CUDA reports available, but no CUDA devices were enumerated.")
    return CudaMetadata(
        available=True,
        torch_cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        device_count=device_count,
        device_names=[torch.cuda.get_device_name(index) for index in range(device_count)],
    )


def _artifacts_with_hashes(paths: list[Path] | None) -> list[UpstreamArtifact]:
    artifacts: list[UpstreamArtifact] = []
    for path in paths or []:
        artifacts.append(UpstreamArtifact(path=str(path), sha256=sha256_file(path)))
    return artifacts


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_stage_manifest(
    stage_name: str,
    config_dump: dict[str, Any],
    global_seed: int,
    device_by_model_family: dict[str, str] | None = None,
    primary_artifact_paths: list[Path] | None = None,
    counts: dict[str, int] | None = None,
    diagnostics: dict[str, Any] | None = None,
    upstream_paths: list[Path] | None = None,
) -> StageManifest:
    return StageManifest(
        stage_name=stage_name,
        git_commit=resolve_git_commit(),
        package_versions=resolve_package_versions(),
        cuda=resolve_cuda_metadata(),
        global_seed=global_seed,
        device_by_model_family=device_by_model_family or {},
        config_sha256=sha256_json(config_dump),
        primary_artifacts=_artifacts_with_hashes(primary_artifact_paths),
        counts=counts or {},
        diagnostics=diagnostics or {},
        upstream_artifacts=_artifacts_with_hashes(upstream_paths),
    )


def write_stage_bundle(
    stage_dir: Path,
    stage_name: str,
    config_dump: dict[str, Any],
    global_seed: int,
    device_by_model_family: dict[str, str] | None = None,
    primary_artifact_paths: list[Path] | None = None,
    counts: dict[str, int] | None = None,
    diagnostics: dict[str, Any] | None = None,
    upstream_paths: list[Path] | None = None,
) -> StageManifest:
    manifest = build_stage_manifest(
        stage_name=stage_name,
        config_dump=config_dump,
        global_seed=global_seed,
        device_by_model_family=device_by_model_family,
        primary_artifact_paths=primary_artifact_paths,
        counts=counts,
        diagnostics=diagnostics,
        upstream_paths=upstream_paths,
    )
    write_yaml(stage_dir / f"{stage_name}_resolved_config.yaml", config_dump)
    write_json(stage_dir / f"{stage_name}_manifest.json", manifest.model_dump())
    return manifest


def write_polars(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
