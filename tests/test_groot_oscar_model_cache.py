from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

from blueprint_pipeline.groot_oscar_model_cache import (
    MANIFEST_NAME,
    MODEL_PINS,
    REQUIRED_MODEL_FILES,
    build_manifest,
    prepare_model_cache,
    verify_model_cache,
)


def _cache(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    assert set(REQUIRED_MODEL_FILES) == {name for name, _, _ in MODEL_PINS}
    for model_name, relatives in REQUIRED_MODEL_FILES.items():
        for relative in relatives:
            path = root / model_name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{model_name}:{relative}".encode())
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
    assert result["verified_file_count"] == sum(
        len(paths) for paths in REQUIRED_MODEL_FILES.values()
    )
    assert str(result["model_manifest_digest"]).startswith("sha256:")
    assert result["cache_root"] == str(root.resolve())


def test_external_model_cache_rejects_digest_different_from_admission(
    tmp_path: Path,
) -> None:
    root = _cache(tmp_path)
    result = verify_model_cache(
        root, expected_manifest_digest="sha256:" + "f" * 64
    )
    assert result["status"] == "blocked"
    assert "model_cache_manifest_digest_differs_from_admission" in result["blockers"]


def test_external_model_cache_verification_records_provider_volume_binding(
    tmp_path: Path,
) -> None:
    root = _cache(tmp_path)
    result = verify_model_cache(root, provider_volume_id="volume-1")
    assert result["status"] == "passed"
    assert result["provider_volume_id"] == "volume-1"


def test_external_model_cache_rejects_mutated_checkpoint(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    (root / "oscar/model/__0_0.distcp").write_bytes(b"tampered")
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


def test_manifest_builder_rejects_placeholder_roots_without_runtime_assets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    for name, _, _ in MODEL_PINS:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "artifact.bin").write_bytes(b"not-the-required-runtime-assets")
    try:
        build_manifest(root)
    except ValueError as exc:
        assert "model_cache_required_files_missing_or_empty" in str(exc)
    else:  # pragma: no cover - makes the fail-closed expectation explicit
        raise AssertionError("placeholder model roots must not produce a manifest")


def test_external_model_cache_names_missing_required_runtime_asset(
    tmp_path: Path,
) -> None:
    root = _cache(tmp_path)
    required = root / "gear_sonic/model_decoder.onnx"
    required.unlink()
    # The original signed manifest remains present; verification must report the
    # semantic runtime-asset blocker in addition to inventory/hash failures.
    result = verify_model_cache(root)
    assert result["status"] == "blocked"
    assert (
        "model_cache_required_file_missing_or_empty:gear_sonic/model_decoder.onnx"
        in result["blockers"]
    )


def test_prepare_uses_runtime_allowlists_and_atomically_replaces_cache(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / "old-cache-marker").write_text("keep-until-ready", encoding="utf-8")
    calls: list[tuple[str, tuple[str, ...]]] = []
    repo_to_name = {repo: name for name, repo, _ in MODEL_PINS}

    def snapshot_download(*, repo_id, revision, local_dir, allow_patterns, token):
        del revision, token
        calls.append((repo_id, tuple(allow_patterns)))
        destination = Path(local_dir)
        for relative in allow_patterns:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if repo_to_name[repo_id] == "sonic" and relative == "config.json":
                path.write_text(
                    json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B"}),
                    encoding="utf-8",
                )
            else:
                path.write_bytes(f"{repo_id}:{relative}".encode())
        metadata = destination / ".cache/huggingface/metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_bytes(b"download-cache-must-not-survive")
        return str(destination)

    def hf_hub_download(*, repo_id, revision, filename, token):
        del revision, token
        source = tmp_path / "downloads" / repo_id.replace("/", "-") / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"{repo_id}:{filename}".encode())
        return str(source)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=snapshot_download,
            hf_hub_download=hf_hub_download,
        ),
    )
    result = prepare_model_cache(root)
    assert result["models_embedded_in_release_image"] is False
    assert not (root / "old-cache-marker").exists()
    assert not list(root.rglob(".cache"))
    assert verify_model_cache(root)["status"] == "passed"
    assert json.loads((root / "sonic/config.json").read_text(encoding="utf-8"))[
        "model_name"
    ] == str(root / "cosmos")
    assert calls == [
        (repo_id, REQUIRED_MODEL_FILES[name])
        for name, repo_id, _ in MODEL_PINS
        if name != "gear_sonic"
    ]


def test_failed_prepare_preserves_previous_cache(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "models"
    root.mkdir()
    marker = root / "old-cache-marker"
    marker.write_text("still-live", encoding="utf-8")

    def fail_snapshot(**kwargs):
        raise RuntimeError(f"download_failed:{kwargs['repo_id']}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_snapshot, hf_hub_download=fail_snapshot),
    )
    try:
        prepare_model_cache(root)
    except RuntimeError as exc:
        assert "download_failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("failed model preparation must propagate")
    assert marker.read_text(encoding="utf-8") == "still-live"
    assert not list(tmp_path.glob(".models.staging-*"))
    assert not list(tmp_path.glob(".models.backup-*"))


def test_thin_image_contract_separates_foundation_models_and_release() -> None:
    root = Path(__file__).resolve().parents[1]
    docker_dir = root / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
    foundation = (docker_dir / "Foundation.Dockerfile").read_text(encoding="utf-8")
    release = (docker_dir / "Release.Dockerfile").read_text(encoding="utf-8")
    entrypoint = (docker_dir / "thin_release_entrypoint.sh").read_text(encoding="utf-8")

    assert "AS wbc-builder" in foundation
    assert "AS robot-env-builder" in foundation
    assert "/opt/robot-venv" not in foundation
    assert foundation.count("uv venv /opt/oscar-venv") == 1
    assert foundation.count("uv venv /opt/gr00t-venv") == 1
    assert "/tmp/oscar/requirements_minimal.txt" in foundation
    assert "requirements_oscar_foundation.lock" in foundation
    assert "target/release/g1_deploy_onnx_ref" in foundation
    assert "cp -a build target g1 scripts reference" not in foundation
    assert "install -m 0755 target/release/g1_deploy_onnx_ref" in foundation
    assert "cp -a /tmp/wbc/gear_sonic /opt/wbc-runtime/gear_sonic" not in foundation
    assert "cp -a /opt/onnxruntime/lib/libonnxruntime.so*" in foundation
    assert "test ! -d /opt/wbc/gear_sonic_deploy/build" in foundation
    assert "test ! -d /opt/onnxruntime/include" in foundation
    assert "libzmq5 libyaml-cpp0.8 zlib1g" in foundation
    assert ".blueprint-source-revision" in foundation
    assert "test ! -e /opt/blueprint/ckpts" in foundation
    assert "ARG FOUNDATION_IMAGE" in release
    assert "FROM ${FOUNDATION_IMAGE}" in release
    assert "snapshot_download" not in release
    assert "test ! -e /opt/blueprint/ckpts" in release
    assert "groot_oscar_model_cache activate" in entrypoint
    assert "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST" in entrypoint
    assert "--expected-manifest-digest" in entrypoint
    assert 'if [[ $# -eq 0 || "${1}" == -* ]]' in entrypoint
    assert 'set -- blueprint-run-robot-eval-worker "$@"' in entrypoint


def test_release_builder_enforces_digest_foundation_and_two_gib_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/build_push_groot_oscar_release_image.sh").read_text(
        encoding="utf-8"
    )
    assert "foundation image must be digest pinned" in script
    assert "2147483648" in script
    assert "thin release exceeds compressed budget" in script
    assert "--build-arg \"FOUNDATION_IMAGE=$foundation\"" in script
