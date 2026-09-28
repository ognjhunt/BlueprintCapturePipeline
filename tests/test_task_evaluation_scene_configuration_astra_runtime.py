"""Optional Blender component bytes materialize without downloads or dependency installs."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from blueprint_pipeline import production_blender_runtime as blender
from blueprint_pipeline import task_evaluation_scene_configuration_astra_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    archive = tmp_path / "retained-different-basename.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        info = tarfile.TarInfo(blender.EXECUTABLE_RELATIVE)
        body = b"#!/bin/sh\n# fixture is never executed\n"
        info.size, info.mode = len(body), 0o755
        bundle.addfile(info, io.BytesIO(body))
    monkeypatch.setattr(blender, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(blender.platform, "system", lambda: "Linux")
    monkeypatch.setattr(blender.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(blender.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    def no_network(*args, **kwargs):
        pytest.fail("packaged runtime must never download")
    monkeypatch.setattr(blender, "_download_archive", no_network)
    monkeypatch.setattr(blender.urllib.request, "urlopen", no_network)
    package = tmp_path / "package"
    manifest = runtime.stage_blender_runtime_archive(archive, package)
    return SimpleNamespace(root=tmp_path, package=package, archive=archive, manifest=manifest)


def version_runner(argv, **kwargs):
    assert argv[1:] == ["--background", "--factory-startup", "--version"]
    return subprocess.CompletedProcess(argv, 0, stdout="Blender 5.2.1 LTS\n", stderr="")


def test_packaged_archive_materializes_and_reuses_verified_version_receipt(packaged):
    destination = packaged.root / "runtime"
    result = runtime.materialize_packaged_blender_runtime(packaged.package, destination, runner=version_runner)
    assert result["materialization_source"] == "sealed_component_archive"
    assert result["runtime_root"] == str(destination)
    assert result["component_manifest_digest"] == packaged.manifest["manifest_digest"]
    receipt = json.loads((destination / blender.RECEIPT_NAME).read_text())
    assert receipt["archive_sha256"] == blender.ARCHIVE_SHA256
    assert receipt["executable_sha256"] == hashlib.sha256(Path(result["executable"]).read_bytes()).hexdigest()
    assert runtime.materialize_packaged_blender_runtime(packaged.package, destination, runner=version_runner) == result


def test_mutated_packaged_archive_refuses_before_install_or_execution(packaged):
    path = packaged.package / blender.ARCHIVE_NAME
    path.chmod(0o644)
    path.write_bytes(b"mutated")
    def forbidden(*args, **kwargs): pytest.fail("mutated archive must never execute")
    with pytest.raises(runtime.AstraRuntimePackageError, match="archive_digest_or_path_invalid"):
        runtime.materialize_packaged_blender_runtime(packaged.package, packaged.root / "runtime", runner=forbidden)
    assert not (packaged.root / "runtime").exists()


def test_wrong_official_archive_refuses_before_staging(tmp_path):
    archive = tmp_path / "wrong.tar.xz"
    archive.write_bytes(b"not the official archive")
    with pytest.raises(runtime.AstraRuntimePackageError, match="archive_digest_or_path_invalid"):
        runtime.stage_blender_runtime_archive(archive, tmp_path / "package")
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("field,value", [("python_profile", "generic-nvidia"), ("provider_download_required", True)])
def test_recomputed_manifest_cannot_change_runtime_policy(packaged, field, value):
    path = packaged.package / runtime.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest[field] = value
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    path.write_text(json.dumps(manifest))
    with pytest.raises(runtime.AstraRuntimePackageError, match="manifest_invalid"):
        runtime.validate_packaged_blender_runtime(packaged.package)


def test_archive_symlink_and_output_collision_are_not_followed(packaged):
    other = packaged.root / "other"
    other.mkdir()
    (other / blender.ARCHIVE_NAME).symlink_to(packaged.root / "never-created")
    with pytest.raises(runtime.AstraRuntimePackageError, match="output_exists_or_unsafe"):
        runtime.stage_blender_runtime_archive(packaged.archive, other)
    assert not (packaged.root / "never-created").exists()
    package_archive = packaged.package / blender.ARCHIVE_NAME
    package_archive.unlink()
    package_archive.symlink_to(packaged.archive)
    with pytest.raises(runtime.AstraRuntimePackageError, match="archive_digest_or_path_invalid"):
        runtime.validate_packaged_blender_runtime(packaged.package)
