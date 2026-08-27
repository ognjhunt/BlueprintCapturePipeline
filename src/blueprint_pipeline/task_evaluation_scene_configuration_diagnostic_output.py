"""Validation of diagnostic checkpoint references extracted from provider ZIPs."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    SCHEMA_VERSION as DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION,
    validate_scene_configuration_diagnostic_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validated_advanced_checkpoint_reference(
    *,
    extraction_root: Path,
    result: Mapping[str, Any],
    checkpoint_validator: Callable[..., Mapping[str, Any]] = (
        validate_scene_configuration_diagnostic_checkpoint
    ),
) -> tuple[dict[str, Any] | None, str | None]:
    reference = result.get("advanced_checkpoint")
    if not isinstance(reference, Mapping):
        return None, "scene_configuration_diagnostic_advanced_checkpoint_missing"
    relative_root = str(reference.get("provider_output_relative_root") or "")
    relative_manifest = str(reference.get("manifest_relative_path") or "")
    if (
        not relative_root
        or not relative_manifest
        or Path(relative_root).is_absolute()
        or Path(relative_manifest).is_absolute()
        or ".." in Path(relative_root).parts
        or ".." in Path(relative_manifest).parts
    ):
        return None, "scene_configuration_diagnostic_advanced_checkpoint_unsafe"
    root = (extraction_root / relative_root).resolve()
    manifest = (extraction_root / relative_manifest).resolve()
    try:
        root.relative_to(extraction_root)
        manifest.relative_to(root)
    except ValueError:
        return None, "scene_configuration_diagnostic_advanced_checkpoint_unsafe"
    if manifest != root / f"{DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION}.json":
        return None, "scene_configuration_diagnostic_advanced_checkpoint_unsafe"
    try:
        checkpoint = checkpoint_validator(checkpoint_root=root)
    except (OSError, RuntimeError, ValueError):
        return None, "scene_configuration_diagnostic_advanced_checkpoint_invalid"
    files = [path for path in root.rglob("*") if path.is_file()]
    if (
        _sha256(manifest) != reference.get("manifest_sha256")
        or checkpoint.get("checkpoint_digest") != reference.get("checkpoint_digest")
        or checkpoint.get("completed_stage_prefix_count")
        != reference.get("completed_stage_prefix_count")
        or len(files) != reference.get("file_count")
        or sum(path.stat().st_size for path in files) != reference.get("total_bytes")
    ):
        return None, "scene_configuration_diagnostic_advanced_checkpoint_invalid"
    return {
        **dict(reference),
        "checkpoint_root": str(root),
        "manifest_path": str(manifest),
    }, None


__all__ = ["validated_advanced_checkpoint_reference"]
