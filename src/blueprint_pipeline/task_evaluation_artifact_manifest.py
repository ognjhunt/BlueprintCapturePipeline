"""Digest-bound artifact manifests for production Task Evaluation attempts.

The provider worker may describe artifacts in its own terminal receipt, but the
canonical allocator is responsible for independently inventorying the bytes it
actually retained.  This module deliberately accepts only paths beneath one
attempt root so a profile cannot turn manifest generation into an arbitrary
filesystem reader.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_artifact_manifest.v1"


class TaskEvaluationArtifactManifestError(ValueError):
    """Raised when allocator-retained evidence cannot be safely inventoried."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under_attempt_root(path: Path, attempt_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != attempt_root and attempt_root not in resolved.parents:
        raise TaskEvaluationArtifactManifestError(
            "task_evaluation_artifact_path_outside_attempt_root"
        )
    return resolved


def _role_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def build_task_evaluation_artifact_manifest(
    *,
    attempt_root: str | Path,
    artifact_roots: Mapping[str, str | Path],
    required_roles: Sequence[str],
    binding: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory allocator-retained evidence and write one immutable manifest."""

    root = Path(attempt_root).expanduser().resolve()
    if not root.is_dir():
        raise TaskEvaluationArtifactManifestError(
            "task_evaluation_artifact_attempt_root_missing"
        )
    destination = _under_attempt_root(
        Path(output_path) if output_path is not None else root / "artifact_manifest.json",
        root,
    )
    normalized_roles = {
        str(role): _under_attempt_root(Path(path), root)
        for role, path in artifact_roots.items()
        if str(role).strip()
    }
    required = sorted(set(str(role) for role in required_roles if str(role).strip()))
    missing_roles = sorted(
        role for role in required if not _role_files(normalized_roles.get(role, root / ".missing"))
    )

    roles_by_path: dict[Path, set[str]] = defaultdict(set)
    for role, path in sorted(normalized_roles.items()):
        for artifact in _role_files(path):
            resolved = _under_attempt_root(artifact, root)
            if resolved != destination:
                roles_by_path[resolved].add(role)

    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "roles": sorted(roles),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path, roles in sorted(roles_by_path.items(), key=lambda item: item[0].as_posix())
    ]
    blockers = [f"task_evaluation_artifact_role_missing:{role}" for role in missing_roles]
    if not files:
        blockers.append("task_evaluation_artifact_manifest_empty")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "binding": dict(binding),
        "required_roles": required,
        "observed_roles": sorted(
            role for role, path in normalized_roles.items() if _role_files(path)
        ),
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    payload["manifest_digest"] = canonical_digest(
        payload, digest_field="manifest_digest"
    )
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationArtifactManifestError(
                "task_evaluation_artifact_manifest_existing_invalid"
            ) from exc
        if existing != payload:
            raise TaskEvaluationArtifactManifestError(
                "task_evaluation_artifact_manifest_immutable_conflict"
            )
        return payload
    write_json(destination, payload)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationArtifactManifestError",
    "build_task_evaluation_artifact_manifest",
]
