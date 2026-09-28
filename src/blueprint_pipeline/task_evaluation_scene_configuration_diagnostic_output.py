"""Validation of diagnostic checkpoint references extracted from provider ZIPs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
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


def seal_validated_diagnostic_checkpoint_reference(
    *,
    checkpoint_root: str | Path,
    destination: str | Path,
    source_provider_result_digest: str,
    checkpoint_validator: Callable[..., Mapping[str, Any]] = (
        validate_scene_configuration_diagnostic_checkpoint
    ),
) -> dict[str, Any]:
    """Seal an extracted completed prefix for a later cold diagnostic retry."""

    root = Path(checkpoint_root).expanduser().resolve()
    requested_destination = Path(destination).expanduser().absolute()
    target = requested_destination.parent.resolve() / requested_destination.name
    manifest = root / f"{DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION}.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
        or target.exists()
        or target.is_symlink()
        or not target.parent.is_dir()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_provider_result_digest)
        is None
    ):
        raise ValueError("scene_configuration_diagnostic_checkpoint_reference_invalid")
    checkpoint = dict(checkpoint_validator(checkpoint_root=root))
    prefix = checkpoint.get("completed_stage_prefix_count")
    if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 1:
        raise ValueError("scene_configuration_diagnostic_checkpoint_reference_invalid")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError("scene_configuration_diagnostic_checkpoint_reference_invalid")
    reference: dict[str, Any] = {
        "schema_version": (
            "task_evaluation_scene_configuration_advanced_checkpoint_reference.v1"
        ),
        "status": "validated_diagnostic_checkpoint_ready_for_next_retry",
        "checkpoint_root": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "completed_stage_prefix_count": prefix,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "source_provider_result_digest": source_provider_result_digest,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "reference_digest": "",
    }
    reference["reference_digest"] = canonical_digest(
        reference, digest_field="reference_digest"
    )
    encoded = (canonical_json(reference) + "\n").encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o440)
        if target.read_bytes() != encoded or json.loads(encoded) != reference:
            raise ValueError(
                "scene_configuration_diagnostic_checkpoint_reference_readback_failed"
            )
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if target.is_file() and not target.is_symlink():
            target.unlink()
        raise
    return reference


__all__ = [
    "seal_validated_diagnostic_checkpoint_reference",
    "validated_advanced_checkpoint_reference",
]
