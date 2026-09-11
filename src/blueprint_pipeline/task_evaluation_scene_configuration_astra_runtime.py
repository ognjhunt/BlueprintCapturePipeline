"""Seal and materialize official Blender bytes; no provider download or pip install."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from . import production_blender_runtime as blender
from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA_VERSION = "task_evaluation_scene_configuration_astra_runtime.v1"
MANIFEST_NAME = "astra_runtime_manifest.v1.json"
PYTHON_PROFILE = "astra_asset_authoring"
PYTHON_REQUIREMENTS = ("langgraph==0.2.76",)


class AstraRuntimePackageError(RuntimeError):
    """A sealed runtime prerequisite is absent or differs from its admitted bytes."""


def _checked_archive(path: Path) -> dict[str, Any]:
    if (path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= blender.MAX_ARCHIVE_BYTES
            or blender._digest(path) != blender.ARCHIVE_SHA256):
        raise AstraRuntimePackageError("astra_blender_archive_digest_or_path_invalid")
    return {"relative_path": blender.ARCHIVE_NAME, "sha256": "sha256:" + blender.ARCHIVE_SHA256,
            "size_bytes": path.stat().st_size}


def _python_contract() -> dict[str, Any]:
    return {"python_version": "3.12", "implementation": "cpython", "platform": "linux-x86_64",
            "required_requirements": list(PYTHON_REQUIREMENTS),
            "required_modules": ["agents", "build123d", "cadpy", "langgraph", "pxr", "trimesh"],
            "provisioning_owner": "parent_verified_python_wheelhouse_profile",
            "dependencies_packaged_here": False, "provider_network_install_required": False}


def declared_python_profile(toolchain_root: str | Path) -> str:
    """Select the closed wheel profile from the already sealed component manifest."""
    path = Path(toolchain_root) / 'components/content_agents_rigid_replacement/package' / MANIFEST_NAME
    if not path.exists():
        return 'base'
    if path.is_symlink() or not path.is_file():
        raise AstraRuntimePackageError('astra_runtime_profile_manifest_invalid')
    value = json.loads(path.read_text())
    if (value.get('schema_version') != SCHEMA_VERSION or value.get('python_profile') != PYTHON_PROFILE
            or value.get('authoring_backend') != 'astra_cad_blender_v1'
            or value.get('python_runtime') != _python_contract()
            or value.get('manifest_digest') != canonical_digest(value, digest_field='manifest_digest')):
        raise AstraRuntimePackageError('astra_runtime_profile_manifest_invalid')
    return PYTHON_PROFILE


def stage_blender_runtime_archive(archive_path: str | Path, package_root: str | Path) -> dict[str, Any]:
    """Optional release-build step: verify supplied bytes, copy, then seal their manifest."""
    source, destination = Path(archive_path).expanduser(), Path(package_root)
    archive_record = _checked_archive(source)
    destination.mkdir(parents=True, exist_ok=True)
    target, manifest_path = destination / blender.ARCHIVE_NAME, destination / MANIFEST_NAME
    if destination.is_symlink() or target.exists() or target.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise AstraRuntimePackageError("astra_runtime_package_output_exists_or_unsafe")
    shutil.copyfile(source, target)
    if _checked_archive(target) != archive_record:
        raise AstraRuntimePackageError("astra_blender_archive_copy_mismatch")
    # The immutable component publisher can hardlink these already read-only bytes.
    target.chmod(0o444)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "sealed_runtime_prerequisites",
                "authoring_backend": "astra_cad_blender_v1", "python_profile": PYTHON_PROFILE,
                "blender": {"version": blender.VERSION, "platform": "linux-x86_64",
                            "archive_url": blender.ARCHIVE_URL, "archive": archive_record,
                            "executable_relative_path": blender.EXECUTABLE_RELATIVE},
                "python_runtime": _python_contract(), "scene_specific_source": False,
                "provider_download_required": False, "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def validate_packaged_blender_runtime(package_root: str | Path) -> tuple[dict[str, Any], Path]:
    package = Path(package_root)
    path = package / MANIFEST_NAME
    if package.is_symlink() or path.is_symlink() or not path.is_file():
        raise AstraRuntimePackageError("astra_runtime_package_manifest_missing")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise AstraRuntimePackageError("astra_runtime_package_manifest_invalid") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "sealed_runtime_prerequisites"
            or manifest.get("authoring_backend") != "astra_cad_blender_v1"
            or manifest.get("python_profile") != PYTHON_PROFILE
            or manifest.get("python_runtime") != _python_contract()
            or manifest.get("scene_specific_source") is not False
            or manifest.get("provider_download_required") is not False
            or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")):
        raise AstraRuntimePackageError("astra_runtime_package_manifest_invalid")
    archive = package / blender.ARCHIVE_NAME
    record = _checked_archive(archive)
    expected = {"version": blender.VERSION, "platform": "linux-x86_64", "archive_url": blender.ARCHIVE_URL,
                "archive": record, "executable_relative_path": blender.EXECUTABLE_RELATIVE}
    if manifest.get("blender") != expected:
        raise AstraRuntimePackageError("astra_runtime_package_blender_identity_mismatch")
    return manifest, archive


def materialize_packaged_blender_runtime(package_root: str | Path, destination_root: str | Path,
                                        *, runner: Callable = subprocess.run) -> dict[str, Any]:
    """Use the production extractor/version receipt with a local archive-copy callback."""
    manifest, archive = validate_packaged_blender_runtime(package_root)

    def local_archive_copy(url: str, destination: Path) -> None:
        if url != blender.ARCHIVE_URL:
            raise AstraRuntimePackageError("astra_blender_unexpected_download_request")
        # No network operation exists on this path. The official installer rechecks copied bytes.
        shutil.copyfile(archive, destination)

    result = blender.install_runtime(Path(destination_root), downloader=local_archive_copy, runner=runner)
    return {**result, "runtime_root": str(Path(destination_root).resolve()),
            "component_manifest_digest": manifest["manifest_digest"],
            "materialization_source": "sealed_component_archive"}
