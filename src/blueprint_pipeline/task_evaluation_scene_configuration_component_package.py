"""Immutable package contract for one production scene-configuration component.

The Website selects a capability identity.  It never supplies a command, source
checkout, or executable path.  A release instead publishes exactly one package
for every admitted GPU-backed adapter.  The package contains its driver and any
runtime/source bytes needed on the already-allocated worker, with a complete
byte inventory and public-source identity.  Scene-specific inputs remain in the
construction envelope and are never embedded in this platform package.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
)


SCHEMA_VERSION = "task_evaluation_scene_configuration_component_package.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_NETWORK_POLICY_BY_ADAPTER = {
    "artifixer3d_observed_object_removal": "provider_and_openai_api",
    "content_agents_rigid_replacement": "provider_and_openai_api",
    "simready_native_import_qualification": "disabled",
}


class TaskEvaluationSceneConfigurationComponentPackageError(ValueError):
    """A component package was incomplete, mutable, or identity-mismatched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _relative_file(root: Path, value: Any) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_path_invalid"
        )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_path_invalid"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_path_invalid"
        )
    return path


def validate_scene_configuration_component_package(
    *,
    root: str | Path,
    expected_adapter_id: str,
    require_read_only: bool = True,
) -> dict[str, Any]:
    """Validate one exhaustive, digest-bound component package directory."""

    package_root = Path(root).expanduser().resolve()
    manifest_path = package_root / f"{SCHEMA_VERSION}.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_manifest_invalid"
        ) from exc
    admitted = {
        identity.adapter_id: identity for identity in ADMITTED_PRODUCER_IDENTITIES
    }
    identity = admitted.get(expected_adapter_id)
    source = value.get("source_identity") if isinstance(value, Mapping) else None
    files = value.get("files") if isinstance(value, Mapping) else None
    if (
        identity is None
        or not isinstance(value, Mapping)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "immutable_component_ready"
        or value.get("adapter_id") != expected_adapter_id
        or value.get("adapter_version") != identity.version
        or value.get("capability") != identity.capability
        or not isinstance(source, Mapping)
        or not str(source.get("repository") or "").strip()
        or not _COMMIT.fullmatch(str(source.get("commit") or ""))
        or not str(source.get("license") or "").strip()
        or source.get("scene_specific_source") is not False
        or value.get("driver_protocol")
        != "task_evaluation_scene_configuration_component_driver.v1"
        or value.get("network_policy")
        != _NETWORK_POLICY_BY_ADAPTER.get(expected_adapter_id)
        or value.get("secrets_via_files_only") is not True
        or value.get("raw_secret_values_in_argv_or_logs") is not False
        or not isinstance(files, list)
        or not files
        or value.get("package_digest")
        != canonical_digest(value, digest_field="package_digest")
    ):
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_manifest_invalid"
        )
    observed: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationComponentPackageError(
                "scene_configuration_component_package_inventory_invalid"
            )
        relative = str(row.get("relative_path") or "")
        path = _relative_file(package_root, relative)
        if (
            relative in observed
            or not _DIGEST.fullmatch(str(row.get("sha256") or ""))
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
            or bool(path.stat().st_mode & 0o111) is not bool(row.get("executable"))
            or (require_read_only and path.stat().st_mode & 0o222)
        ):
            raise TaskEvaluationSceneConfigurationComponentPackageError(
                "scene_configuration_component_package_inventory_invalid"
            )
        observed.add(relative)
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != observed:
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_inventory_incomplete"
        )
    driver = _relative_file(package_root, value.get("driver_entrypoint"))
    if not driver.stat().st_mode & 0o111:
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_driver_invalid"
        )
    return dict(value)


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationComponentPackageError",
    "validate_scene_configuration_component_package",
]
