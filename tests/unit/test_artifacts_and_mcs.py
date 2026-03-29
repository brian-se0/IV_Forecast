from __future__ import annotations

import numpy as np

from ivs_forecast.artifacts.hashing import sha256_file
from ivs_forecast.artifacts.manifests import build_stage_manifest
from ivs_forecast.evaluation.mcs import run_mcs


def test_build_stage_manifest_records_primary_artifacts(tmp_path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text('{"ok": true}', encoding="utf-8")
    manifest = build_stage_manifest(
        stage_name="unit_test",
        config_dump={"runtime": {"seed": 123}},
        global_seed=123,
        device_by_model_family={"rw_last": "cpu"},
        primary_artifact_paths=[artifact_path],
        counts={"rows": 1},
        diagnostics={"status": "ok"},
        upstream_paths=[],
    )
    assert manifest.git_commit
    assert manifest.package_versions["numpy"]
    assert manifest.global_seed == 123
    assert manifest.primary_artifacts[0].sha256 == sha256_file(artifact_path)


def test_run_mcs_excludes_dominated_model_and_is_reproducible() -> None:
    observations = 60
    loss_by_model = {
        "best": np.full(observations, 0.10, dtype=np.float64),
        "middle": np.full(observations, 0.20, dtype=np.float64),
        "worst": np.full(observations, 1.00, dtype=np.float64),
    }
    result_a = run_mcs(loss_by_model, bootstrap_draws=400, block_length=5, seed=7)
    result_b = run_mcs(loss_by_model, bootstrap_draws=400, block_length=5, seed=7)
    assert result_a == result_b
    assert "worst" in result_a["Tmax"]["excluded_models"]
    assert "worst" in result_a["TR"]["excluded_models"]
