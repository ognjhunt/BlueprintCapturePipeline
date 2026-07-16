from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import groot_oscar_model_cache_s3_remote_executor as executor
from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    RUNTIME_ARCHIVE_ROOTS,
)


SOURCE_REF = "docker.io/blueprint/release@sha256:" + "1" * 64
CARRIER_REF = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" + "2" * 64


class FakeDocker:
    def __init__(self, *, verify_digest: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.verify_digest = verify_digest

    def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        command = list(argv)
        self.calls.append(command)
        if command[1] == "pull":
            return SimpleNamespace(stdout="")
        if command[1:4] == ["image", "inspect", "--format"]:
            image_ref = command[-1]
            digest = image_ref.rpartition("@")[2] if self.verify_digest else "sha256:" + "f" * 64
            return SimpleNamespace(stdout=json.dumps(["docker.io/test/image@" + digest]))
        if command[1] == "create":
            return SimpleNamespace(stdout="a" * 64 + "\n")
        if command[1] == "cp":
            source = command[2].split(":", 1)[1]
            destination = Path(command[3])
            copied = destination / Path(source).name
            (copied / "bin").mkdir(parents=True)
            (copied / "bin/python3").write_text("runtime\n", encoding="utf-8")
            (copied / "bin/python").symlink_to("python3")
            if source == "/opt/blueprint":
                (copied / "ckpts/sonic").mkdir(parents=True)
                (copied / "ckpts/sonic/model.safetensors").write_bytes(b"model")
                (copied / "hf_home").mkdir()
                (copied / "hf_home/cached-model").write_bytes(b"model")
                (copied / "models").symlink_to("hf_home")
            return SimpleNamespace(stdout="")
        if command[1:3] == ["rm", "-f"]:
            return SimpleNamespace(stdout="")
        if command[1] == "run":
            return SimpleNamespace(stdout="runtime verified\n")
        raise AssertionError(command)


def _request() -> dict:
    return {
        "enabled": True,
        "source_release_image_ref": SOURCE_REF,
        "carrier_image_ref": CARRIER_REF,
    }


def test_prepare_runtime_bundle_copies_allowlist_and_builds_verified_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")
    docker = FakeDocker()
    runtime_root = tmp_path / "runtime"
    build_root = tmp_path / "build"

    result = executor.prepare_runtime_bundle(
        _request(),
        runtime_root=runtime_root,
        build_root=build_root,
        runner=docker,
        generated_at="2026-07-15T12:00:00Z",
    )

    assert result["status"] == "completed"
    assert result["source_and_carrier_registry_digests_verified"] is True
    assert len(result["additional_artifacts"]) == 2
    assert {row["remote_key"] for row in result["additional_artifacts"]} == {
        DEFAULT_RUNTIME_ARCHIVE_PATH.removeprefix("/workspace/"),
        DEFAULT_RUNTIME_MANIFEST_PATH.removeprefix("/workspace/"),
    }
    with tarfile.open(result["archive_path"], "r:gz") as archive:
        inventory = archive.getnames()
        for root in RUNTIME_ARCHIVE_ROOTS:
            assert root in inventory
            assert f"{root}/bin/python" in inventory
        assert not any(name.startswith("opt/blueprint/ckpts/") for name in inventory)
        assert not any(name.startswith("opt/blueprint/hf_home/") for name in inventory)
        assert "opt/blueprint/models" not in inventory
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["source_release_image_ref"] == SOURCE_REF
    assert manifest["carrier_image_ref"] == CARRIER_REF
    assert [call[:2] for call in docker.calls].count(["docker", "pull"]) == 2
    assert docker.calls[-1] == ["docker", "rm", "-f", "a" * 64]
    carrier_runs = [call for call in docker.calls if call[1] == "run"]
    assert len(carrier_runs) == 1
    assert "--network" in carrier_runs[0]
    assert "none" in carrier_runs[0]
    assert SOURCE_REF not in carrier_runs[0]
    assert CARRIER_REF in carrier_runs[0]
    assert Path(result["archive_path"]).parent == runtime_root
    assert not build_root.exists()


def test_prepare_runtime_bundle_rejects_registry_digest_mismatch_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")
    docker = FakeDocker(verify_digest=False)

    with pytest.raises(RuntimeError, match="typed_runtime_bundle_image_digest_unverified"):
        executor.prepare_runtime_bundle(
            _request(),
            runtime_root=tmp_path / "runtime",
            build_root=tmp_path / "build",
            runner=docker,
        )

    assert not any(call[1] == "cp" for call in docker.calls)


def test_prepare_runtime_bundle_not_requested_never_calls_docker(tmp_path: Path) -> None:
    docker = FakeDocker()
    result = executor.prepare_runtime_bundle(
        {"enabled": False},
        runtime_root=tmp_path / "runtime",
        build_root=tmp_path / "build",
        runner=docker,
    )
    assert result["status"] == "not_requested"
    assert result["additional_artifacts"] == []
    assert docker.calls == []
