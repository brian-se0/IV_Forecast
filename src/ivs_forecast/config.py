from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def _normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def underlying_to_key(symbol: str) -> str:
    return "".join(ch.lower() for ch in symbol if ch.isalnum()) or "unknown"


class PathConfig(BaseModel):
    raw_data_root: Path
    artifact_root: Path = Path("artifacts")

    @field_validator("raw_data_root", "artifact_root", mode="before")
    @classmethod
    def _coerce_paths(cls, value: str | Path) -> Path:
        return _normalize_path(value)


class StudyConfig(BaseModel):
    underlying_symbol: Literal["^SPX"] = "^SPX"
    option_root: Literal["SPX"]
    start_date: date
    end_date: date
    forecast_horizon_days: int = 1
    min_dte_days: int = 10
    max_dte_days: int = 730
    min_surface_nodes: int = 40
    min_valid_expiries: int = 4

    @model_validator(mode="after")
    def _validate_range(self) -> StudyConfig:
        if self.forecast_horizon_days != 1:
            raise ValueError("Only 1-day-ahead forecasting is supported in v1.")
        if self.start_date >= self.end_date:
            raise ValueError("study.start_date must be earlier than study.end_date.")
        return self


class SplitConfig(BaseModel):
    validation_size: int = 252
    test_size: int = 504
    min_train_size: int = 756
    refit_frequency: int = 21


class RuntimeConfig(BaseModel):
    seed: int = 20260329
    overwrite: bool = False
    run_id: str | None = None

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return datetime.now().strftime("%Y%m%d_%H%M%S")


class AppConfig(BaseModel):
    paths: PathConfig
    study: StudyConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def model_config_dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def run_root(self) -> Path:
        return self.paths.artifact_root / "runs" / self.runtime.resolved_run_id()

    @property
    def subset_root(self) -> Path:
        key = underlying_to_key(self.study.underlying_symbol)
        return self.run_root / "subset" / f"underlying_key={key}" / f"option_root={self.study.option_root}"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path, raw_data_root_override: Path | None = None) -> AppConfig:
    yaml_config = load_yaml(config_path)
    env_override: dict[str, Any] = {}
    if env_raw := os.environ.get("IVS_FORECAST_RAW_DATA_ROOT"):
        env_override = {"paths": {"raw_data_root": env_raw}}
    cli_override: dict[str, Any] = {}
    if raw_data_root_override is not None:
        cli_override = {"paths": {"raw_data_root": str(raw_data_root_override)}}
    merged = deep_merge(yaml_config, env_override)
    merged = deep_merge(merged, cli_override)
    return AppConfig.model_validate(merged)


def ensure_run_directory(config: AppConfig) -> Path:
    run_root = config.run_root
    if run_root.exists() and not config.runtime.overwrite:
        raise FileExistsError(
            f"Artifact directory already exists: {run_root}. Set runtime.overwrite=true to replace it."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root
