"""Byte-bound synthetic runtime packages for CPU-only construction rehearsals."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from blueprint_pipeline import production_blender_runtime as blender
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_astra_runtime import stage_blender_runtime_archive
from blueprint_pipeline.task_evaluation_scene_configuration_component_package import SCHEMA_VERSION
from scripts.build_task_evaluation_scene_configuration_toolchain import build_published_scene_configuration_toolchain
from tests.test_build_task_evaluation_scene_configuration_toolchain import _component_packages
from tests.test_task_evaluation_scene_configuration_python_runtime import _build_astra


def _reseal(package):
    manifest_path = package / f"{SCHEMA_VERSION}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [{"relative_path": str(path.relative_to(package)),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size, "executable": bool(path.stat().st_mode & 0o111)}
        for path in sorted(package.rglob("*")) if path.is_file() and path != manifest_path]
    manifest["package_digest"] = canonical_digest(manifest, digest_field="package_digest")
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest))
    for path in package.rglob("*"):
        path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
    package.chmod(0o555)


def astra_toolchain_fixture(root: Path, commit: str, monkeypatch):
    """Exercise real Astra gates with sealed fixture bytes, never loosen the gate."""
    workspace = root.parent / "astra-toolchain-fixture"
    packages = _component_packages(workspace)
    removal = packages["artifixer3d_observed_object_removal"]
    removal.chmod(0o755)
    previous = removal / "python_wheelhouse"
    for path in previous.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    previous.chmod(0o755)
    shutil.rmtree(previous)  # Only the fixture tree created immediately above.
    python_fixture = workspace / "python-runtime"
    python_fixture.mkdir()
    wheelhouse, _ = _build_astra(python_fixture)
    shutil.copytree(wheelhouse, previous)
    _reseal(removal)
    authoring = packages["content_agents_rigid_replacement"]
    authoring.chmod(0o755)
    archive = workspace / "blender-fixture.tar.xz"
    with tarfile.open(archive, "w:xz") as stream:
        info = tarfile.TarInfo(blender.EXECUTABLE_RELATIVE)
        body = b"#!/bin/sh\n# Synthetic runtime archive; never executed.\n"
        info.size, info.mode = len(body), 0o755
        stream.addfile(info, io.BytesIO(body))
    monkeypatch.setattr(blender, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    stage_blender_runtime_archive(archive, authoring)
    _reseal(authoring)
    build_published_scene_configuration_toolchain(source_commit=commit, output_root=root,
        readback=lambda path: path.read_bytes(), readback_actor="service-account:test", component_packages=packages)
    return root
