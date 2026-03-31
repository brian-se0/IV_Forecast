from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "implementation_spec.md",
    REPO_ROOT / "docs" / "artifact_contracts.md",
    REPO_ROOT / "docs" / "methodology.md",
    REPO_ROOT / "docs" / "vendor_dataset_contract.md",
]
FORBIDDEN_TERMS = [
    "rw_last",
    "pca_var1",
    "xgb_direct",
    "lstm_direct",
]
FORBIDDEN_PHRASES = [
    "The shared reconstructor is part of the pipeline and is not optional.",
    "reconstructor is fit only on the currently available expanding window.",
]


def test_live_docs_do_not_reference_legacy_runtime_models() -> None:
    for path in LIVE_DOCS:
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"Legacy runtime term {term!r} remained in {path}"
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in text, f"Legacy runtime phrase remained in {path}: {phrase!r}"


def test_stale_root_level_plan_is_removed() -> None:
    assert not (REPO_ROOT / "new_PLAN.md").exists()
