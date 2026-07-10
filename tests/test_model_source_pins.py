from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COSMOS_REPO_COMMIT = "661da4774b0ca41d082a0ecbeb47550bcf07e03f"
COSMOS_MODEL_COMMIT = "0d37c7498f54cee3c599d438d895a0a4a8608064"


def test_cosmos_bootstrap_requires_reviewed_detached_commit() -> None:
    script = (ROOT / "scripts" / "bootstrap_cosmos_official_repo.sh").read_text(encoding="utf-8")

    assert COSMOS_REPO_COMMIT in script
    assert "COSMOS_OFFICIAL_REPO_REF:-main" not in script
    assert "COSMOS_OFFICIAL_REPO_REF:-master" not in script
    assert "must be an immutable 40-hex commit" in script
    assert "is not in the reviewed allowlist" in script
    assert 'checkout --detach "$ref"' in script
    assert "pull --ff-only" not in script


def test_ml_installer_pins_optional_source_and_model_downloads() -> None:
    script = (ROOT / "scripts" / "install_ml_stack.sh").read_text(encoding="utf-8")

    for commit in (
        COSMOS_REPO_COMMIT,
        COSMOS_MODEL_COMMIT,
        "5dd401d1c5c1d5c3eedff06d41b77af824517619",
        "41736238f5bced4debf3f2a12375d2466874866d",
        "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        "f4d8f09d1eb8f758c89cf1795ae41f20533942d3",
    ):
        assert commit in script
    assert 'checkout --detach "$ref"' in script
    assert "pull --ff-only" not in script
    assert '--revision "${DA3_MODEL_REVISION}"' in script
    assert '--revision "${QWEN_IMAGE_EDIT_MODEL_REVISION}"' in script
    assert "require_reviewed_pin" in script
    assert "COSMOS_PREDICT_VERSION:-1.0.9" in script


def test_native_cosmos_surfaces_export_exact_model_revision() -> None:
    start_script = (ROOT / "scripts" / "start_native_runtime_vast.sh").read_text(encoding="utf-8")
    env_example = (ROOT / "configs" / "native_runtime_vast.env.example").read_text(encoding="utf-8")
    runbook = (ROOT / "docs" / "GPU_VM_RUNBOOK.md").read_text(encoding="utf-8")

    assert COSMOS_MODEL_COMMIT in start_script
    assert COSMOS_MODEL_COMMIT in env_example
    assert COSMOS_MODEL_COMMIT in runbook
    assert COSMOS_REPO_COMMIT in runbook
    assert 'COSMOS_OFFICIAL_REPO_REF="main"' not in runbook
