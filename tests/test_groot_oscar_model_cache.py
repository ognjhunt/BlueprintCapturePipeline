from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.groot_oscar_model_cache import (
    MANIFEST_NAME,
    MODEL_PINS,
    build_manifest,
    verify_model_cache,
)


def _cache(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for index, (name, _, _) in enumerate(MODEL_PINS):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "artifact.bin").write_bytes(f"model-{index}".encode())
    manifest = build_manifest(root)
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_external_model_cache_verifies_every_declared_byte(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    result = verify_model_cache(root)
    assert result["status"] == "passed"
    assert result["checks"]["models_cached_offline"] is True
    assert result["verified_file_count"] == 4
    assert str(result["model_manifest_digest"]).startswith("sha256:")


def test_external_model_cache_rejects_mutated_checkpoint(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    (root / "oscar/artifact.bin").write_bytes(b"tampered")
    result = verify_model_cache(root)
    assert result["status"] == "blocked"
    assert any(
        blocker.startswith("model_cache_file_") for blocker in result["blockers"]
    )


def test_external_model_cache_rejects_unmanifested_file(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    (root / "sonic/extra.bin").write_bytes(b"extra")
    result = verify_model_cache(root)
    assert result["status"] == "blocked"
    assert "model_cache_inventory_mismatch" in result["blockers"]


def test_external_model_cache_rejects_revision_rewrite(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    manifest_path = root / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["repositories"][0]["revision"] = "mutable-main"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    result = verify_model_cache(root)
    assert result["status"] == "blocked"
    assert "model_cache_repository_pins_mismatch" in result["blockers"]
    assert "model_cache_manifest_digest_mismatch" in result["blockers"]


def test_thin_image_contract_separates_foundation_models_and_release() -> None:
    root = Path(__file__).resolve().parents[1]
    docker_dir = root / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
    foundation = (docker_dir / "Foundation.Dockerfile").read_text(encoding="utf-8")
    release = (docker_dir / "Release.Dockerfile").read_text(encoding="utf-8")
    entrypoint = (docker_dir / "thin_release_entrypoint.sh").read_text(encoding="utf-8")

    assert "AS wbc-builder" in foundation
    assert "AS robot-env-builder" in foundation
    assert "/opt/robot-venv" in foundation
    assert foundation.count("uv venv /opt/robot-venv") == 1
    assert "uv venv /opt/gr00t-venv" not in foundation
    assert "ln -s /opt/robot-venv /opt/gr00t-venv" in foundation
    assert "target/release/g1_deploy_onnx_ref" in foundation
    assert "cp -a build target g1 scripts reference" in foundation
    assert ".blueprint-source-revision" in foundation
    assert "test ! -e /opt/blueprint/ckpts" in foundation
    assert "ARG FOUNDATION_IMAGE" in release
    assert "FROM ${FOUNDATION_IMAGE}" in release
    assert "snapshot_download" not in release
    assert "test ! -e /opt/blueprint/ckpts" in release
    assert "groot_oscar_model_cache activate" in entrypoint


def test_release_builder_enforces_digest_foundation_and_two_gib_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/build_push_groot_oscar_release_image.sh").read_text(
        encoding="utf-8"
    )
    assert "foundation image must be digest pinned" in script
    assert "2147483648" in script
    assert "thin release exceeds compressed budget" in script
    assert "--build-arg \"FOUNDATION_IMAGE=$foundation\"" in script
