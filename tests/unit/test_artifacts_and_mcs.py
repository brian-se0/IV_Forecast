from __future__ import annotations

import numpy as np
import pytest

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
    loss_by_model = {
        "best": np.array([0.10, 0.12, 0.11, 0.10, 0.13, 0.09, 0.11, 0.12], dtype=np.float64),
        "mid": np.array([0.20, 0.22, 0.21, 0.20, 0.23, 0.19, 0.21, 0.22], dtype=np.float64),
        "worst": np.array([0.80, 0.82, 0.81, 0.80, 0.83, 0.79, 0.81, 0.82], dtype=np.float64),
    }
    result_a = run_mcs(loss_by_model, bootstrap_draws=400, block_length=4, seed=7)
    result_b = run_mcs(loss_by_model, bootstrap_draws=400, block_length=4, seed=7)
    assert result_a == result_b
    assert result_a["Tmax"]["elimination_order"] == ["worst", "mid"]
    assert result_a["TR"]["elimination_order"] == ["worst", "mid"]
    assert result_a["Tmax"]["included_models"] == ["best"]
    assert result_a["TR"]["included_models"] == ["best"]
    assert result_a["Tmax"]["p_values_by_step"][0]["observed_statistic"] == pytest.approx(
        650000.0000000001
    )
    assert result_a["TR"]["p_values_by_step"][0]["observed_statistic"] == pytest.approx(
        700000.0000000001
    )


def test_run_mcs_keeps_equal_loss_models() -> None:
    loss_by_model = {
        "m1": np.array([0.20, 0.30, 0.25, 0.22, 0.28, 0.27], dtype=np.float64),
        "m2": np.array([0.20, 0.30, 0.25, 0.22, 0.28, 0.27], dtype=np.float64),
        "m3": np.array([0.20, 0.30, 0.25, 0.22, 0.28, 0.27], dtype=np.float64),
    }
    result = run_mcs(loss_by_model, bootstrap_draws=300, block_length=3, seed=11)
    for method in ("Tmax", "TR"):
        assert result[method]["included_models"] == ["m1", "m2", "m3"]
        assert result[method]["excluded_models"] == []
        assert result[method]["p_values_by_step"][0]["observed_statistic"] == pytest.approx(0.0)
        assert result[method]["p_values_by_step"][0]["p_value"] == pytest.approx(1.0)


def test_run_mcs_autocorrelated_fixture_matches_reference_values() -> None:
    loss_by_model = {
        "a": np.array([0.30, 0.35, 0.33, 0.36, 0.34, 0.37, 0.35, 0.38, 0.36, 0.39]),
        "b": np.array([0.32, 0.34, 0.36, 0.35, 0.37, 0.36, 0.38, 0.37, 0.39, 0.38]),
        "c": np.array([0.50, 0.52, 0.51, 0.53, 0.52, 0.54, 0.53, 0.55, 0.54, 0.56]),
    }
    result = run_mcs(loss_by_model, bootstrap_draws=500, block_length=5, seed=19)
    assert result["Tmax"]["elimination_order"] == ["c", "b"]
    assert result["TR"]["elimination_order"] == ["c", "b"]
    assert result["Tmax"]["p_values_by_step"][1]["p_value"] == pytest.approx(0.002)
    assert result["TR"]["p_values_by_step"][1]["p_value"] == pytest.approx(0.002)
    assert result["Tmax"]["p_values_by_step"][0]["observed_statistic"] == pytest.approx(
        100.99235854463049
    )
    assert result["TR"]["p_values_by_step"][0]["observed_statistic"] == pytest.approx(
        116.60371492801961
    )
