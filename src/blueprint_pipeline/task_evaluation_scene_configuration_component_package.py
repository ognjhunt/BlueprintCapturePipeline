"""Immutable package contract for one production scene-configuration component.

The Website selects a capability identity.  It never supplies a command, source
checkout, or executable path.  A release instead publishes exactly one package
for every admitted GPU-backed adapter.  The package contains its driver and any
runtime/source bytes needed on the already-allocated worker, with a complete
byte inventory and public-source identity.  Scene-specific inputs remain in the
construction envelope and are never embedded in this platform package.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .immutable_file_digest import sha256_file
from .immutable_tree_walk import walk_tree
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


def _inventoried_file(root: Path, value: str, directories: dict[str, str]) -> tuple[Path, os.stat_result]:
    """``_relative_file`` and the file's stat, resolving each directory only once.

    Resolving every parent of every file was much of this validation, which a deploy
    runs three times over the same 38,505 files. A symlink goes the long way.
    """

    relative = Path(value)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_path_invalid"
        )
    directory, name = os.path.split(os.path.join(root, relative))
    resolved = directories.get(directory)
    if resolved is None:
        resolved = directories[directory] = os.path.realpath(directory)
    leaf = os.path.join(resolved, name)
    try:
        info = os.lstat(leaf)
    except OSError:
        info = None
    if info is None or stat.S_ISLNK(info.st_mode):
        path = _relative_file(root, value)
        return path, path.stat()
    if not stat.S_ISREG(info.st_mode) or not leaf.startswith(f"{root}{os.sep}"):
        raise TaskEvaluationSceneConfigurationComponentPackageError(
            "scene_configuration_component_package_path_invalid"
        )
    return Path(leaf), info


def _sha256(path: Path) -> str:
    return sha256_file(path)


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
    directories: dict[str, str] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationComponentPackageError(
                "scene_configuration_component_package_inventory_invalid"
            )
        relative = str(row.get("relative_path") or "")
        path, info = _inventoried_file(package_root, relative, directories)
        if (
            relative in observed
            or not _DIGEST.fullmatch(str(row.get("sha256") or ""))
            or info.st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
            or bool(info.st_mode & 0o111) is not bool(row.get("executable"))
            or (require_read_only and info.st_mode & 0o222)
        ):
            raise TaskEvaluationSceneConfigurationComponentPackageError(
                "scene_configuration_component_package_inventory_invalid"
            )
        observed.add(relative)
    actual = {relative for relative, entry in walk_tree(package_root) if entry.is_file()}
    actual.discard(manifest_path.name)
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
