from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.ctrl_world_joint_position_reference_wam import MODEL_FREEZE


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "deploy/docker/policy_ranking_openpi_ctrl_world"


def test_combined_worker_preserves_separate_pinned_runtimes() -> None:
    dockerfile = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "FROM docker.io/nijelhunt/blueprint-openpi-policy-ranking@sha256:"
        "badde151b360cd5227ede3e4e037b5e12f36ed59ac09ca89470a22511e72c6c4"
    ) in dockerfile
    assert "BLUEPRINT_CTRL_WORLD_PYTHON=/.ctrl-world-venv/bin/python" in dockerfile
    assert "BLUEPRINT_CTRL_WORLD_MODEL_ROOT=/workspace/ctrl-world-models" in dockerfile
    assert "torch==2.7.1 torchvision==0.22.1" in dockerfile
    assert "https://download.pytorch.org/whl/cu128" in dockerfile
    assert "export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5" in dockerfile
    assert dockerfile.index("export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5") < dockerfile.index(
        "uv pip install --python /.ctrl-world-venv/bin/python"
    )
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in dockerfile
    assert "VIRTUAL_ENV=/.venv uv pip install --no-deps" in dockerfile
    assert "VIRTUAL_ENV=/.ctrl-world-venv uv pip install --no-deps" in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "HUGGING_FACE_HUB_TOKEN" not in dockerfile


def test_combined_worker_source_manifest_matches_runtime_freeze() -> None:
    manifest = json.loads(
        (IMAGE_DIR / "ctrl_world_source_manifest.json").read_text(encoding="utf-8")
    )
    freeze = MODEL_FREEZE["ctrl_world_source"]

    assert manifest["repository"] == freeze["repository"]
    assert manifest["revision"] == freeze["revision"]
    assert manifest["files"] == freeze["required_files"]


def test_ctrl_world_dependency_lock_is_exact_and_excludes_policy_environment() -> None:
    rows = [
        row.strip()
        for row in (IMAGE_DIR / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]

    assert len(rows) == len(set(rows))
    assert all(row.count("==") == 1 for row in rows)
    assert "diffusers==0.34.0" in rows
    assert "transformers==4.48.1" in rows
    assert not any(row.startswith(("torch==", "torchvision==", "jax==")) for row in rows)
