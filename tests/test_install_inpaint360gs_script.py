"""Regression guards for Inpaint360GS installer script."""

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "install_inpaint360gs.sh"
    return script_path.read_text(encoding="utf-8")


def test_install_script_exists() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "scripts" / "install_inpaint360gs.sh").is_file()


def test_install_script_clones_from_correct_repo() -> None:
    text = _script_text()
    assert "https://github.com/dfki-av/Inpaint360GS.git" in text


def test_install_script_builds_cuda_kernels() -> None:
    text = _script_text()
    assert "diff-gaussian-rasterization" in text
    assert "simple-knn" in text
    assert "--no-build-isolation" in text


def test_install_script_installs_lama_deps() -> None:
    text = _script_text()
    assert "LaMa/requirements.txt" in text
    assert "SKIP_LAMA" in text


def test_install_script_has_required_env_vars() -> None:
    text = _script_text()
    assert 'INPAINT360GS_DIR' in text
    assert 'INPAINT360GS_PYTHON' in text
    assert 'INPAINT360GS_REF' in text
    assert 'CUDA_HOME' in text


def test_install_script_verifies_core_scripts() -> None:
    text = _script_text()
    assert "train.py" in text
    assert "train_finetune.py" in text
    assert "edit_object_removal.py" in text
    assert "edit_object_inpaint.py" in text
    assert "--probe" in text


def test_install_script_has_skip_options() -> None:
    text = _script_text()
    assert "--skip-cuda-kernels" in text
    assert "--skip-lama" in text


def test_install_script_has_shebang_and_strict_mode() -> None:
    text = _script_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
