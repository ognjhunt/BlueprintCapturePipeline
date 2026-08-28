"""Lease-aware retention for remotely verified scene-configuration artifacts."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json


ACTIVE_STATES = {"pending", "in_flight"}
EXPIRING_ACTIVE_STATES = {"retained_for_retry"}
APPLY_ACK = "reap-remotely-verified-scene-artifacts"


class TaskEvaluationSceneArtifactRetentionError(RuntimeError):
    """A retention plan or mutation failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _parse_time(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_retention_time_invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_retention_time_invalid"
        )
    return parsed.astimezone(UTC)


def _valid_remote_reference(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("schema_version")
        == "task_evaluation_scene_artifact_reference.v1"
        and value.get("status") == "remote_verified"
        and value.get("remote_identity_verified") is True
        and value.get("full_byte_service_account_readback_passed") is True
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("digest") or ""))
        is not None
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and int(value["size_bytes"]) > 0
        and str(value.get("uri") or "").startswith("s3://")
        and value.get("raw_secret_values_recorded") is False
    )


def seal_scene_artifact_lease(
    *,
    destination: str | Path,
    run_id: str,
    lifecycle_state: str,
    artifact_references: Sequence[Mapping[str, Any]],
    expires_at: str | None = None,
) -> dict[str, Any]:
    """Seal a small lease; pending and in-flight leases never expire silently."""

    state = str(lifecycle_state or "")
    if state not in ACTIVE_STATES | EXPIRING_ACTIVE_STATES | {"completed"}:
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_lease_state_invalid"
        )
    if not run_id.strip() or not artifact_references or not all(
        _valid_remote_reference(item) for item in artifact_references
    ):
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_lease_reference_invalid"
        )
    if state in EXPIRING_ACTIVE_STATES:
        _parse_time(str(expires_at or ""))
    elif expires_at is not None:
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_lease_expiration_invalid"
        )
    references = [
        {
            "artifact_kind": item["artifact_kind"],
            "digest": item["digest"],
            "size_bytes": item["size_bytes"],
            "uri": item["uri"],
        }
        for item in artifact_references
    ]
    lease: dict[str, Any] = {
        "schema_version": "task_evaluation_scene_artifact_lease.v1",
        "run_id": run_id,
        "lifecycle_state": state,
        "expires_at": expires_at,
        "artifact_references": references,
        "raw_secret_values_recorded": False,
        "lease_digest": "",
    }
    lease["lease_digest"] = canonical_digest(lease, digest_field="lease_digest")
    target = Path(destination).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    encoded = (canonical_json(lease) + "\n").encode()
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o440)
    if target.read_bytes() != encoded:
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_lease_readback_failed"
        )
    return lease


def _lease_is_active(lease: Mapping[str, Any], *, now: datetime) -> bool:
    if lease.get("schema_version") != "task_evaluation_scene_artifact_lease.v1":
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_lease_invalid"
        )
    if lease.get("lease_digest") != canonical_digest(
        lease, digest_field="lease_digest"
    ):
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_lease_invalid"
        )
    state = str(lease.get("lifecycle_state") or "")
    if state in ACTIVE_STATES:
        return True
    if state in EXPIRING_ACTIVE_STATES:
        return _parse_time(str(lease.get("expires_at") or "")) > now
    if state == "completed":
        return False
    raise TaskEvaluationSceneArtifactRetentionError(
        "scene_artifact_lease_invalid"
    )


def plan_scene_artifact_retention(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    leases: Sequence[Mapping[str, Any]],
    protected_roots: Sequence[str | Path],
    now: str,
) -> dict[str, Any]:
    """Return exact safe candidates; any ambiguous row blocks the whole plan."""

    observed_at = _parse_time(now)
    roots = [Path(root).expanduser().resolve() for root in protected_roots]
    active_digests: set[str] = set()
    for lease in leases:
        if _lease_is_active(lease, now=observed_at):
            for reference in lease.get("artifact_references") or []:
                if isinstance(reference, Mapping):
                    active_digests.add(str(reference.get("digest") or ""))
    blockers: list[str] = []
    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for row in artifacts:
        path = Path(str(row.get("local_path") or "")).expanduser().absolute()
        reference = row.get("remote_reference")
        reason = ""
        if not _valid_remote_reference(reference):
            blockers.append(f"scene_artifact_remote_reference_invalid:{path}")
            reason = "remote_identity_unverified"
        elif path.is_symlink() or not path.is_file():
            blockers.append(f"scene_artifact_local_file_invalid:{path}")
            reason = "local_file_invalid"
        else:
            resolved = path.resolve()
            if any(resolved == root or root in resolved.parents for root in roots):
                reason = "protected_root"
            elif reference["digest"] in active_digests:
                reason = "active_lease"
            elif (
                path.stat().st_size != reference["size_bytes"]
                or _sha256(path) != reference["digest"]
            ):
                blockers.append(f"scene_artifact_local_identity_mismatch:{path}")
                reason = "local_identity_mismatch"
        item = {
            "local_path": str(path),
            "digest": reference.get("digest") if isinstance(reference, Mapping) else None,
            "size_bytes": reference.get("size_bytes") if isinstance(reference, Mapping) else None,
        }
        if reason:
            protected.append({**item, "reason": reason})
        else:
            candidates.append(item)
    if blockers:
        candidates = []
    plan: dict[str, Any] = {
        "schema_version": "task_evaluation_scene_artifact_retention_plan.v1",
        "status": "completed" if not blockers else "blocked",
        "observed_at": now,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in candidates),
        "candidates": candidates,
        "protected": protected,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def apply_scene_artifact_retention(
    *, plan: Mapping[str, Any], ack: str
) -> dict[str, Any]:
    """Apply only a fully validated immutable plan and recheck every byte."""

    if (
        ack != APPLY_ACK
        or plan.get("schema_version")
        != "task_evaluation_scene_artifact_retention_plan.v1"
        or plan.get("status") != "completed"
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
        or plan.get("blockers") != []
    ):
        raise TaskEvaluationSceneArtifactRetentionError(
            "scene_artifact_retention_plan_invalid"
        )
    removed: list[dict[str, Any]] = []
    for row in plan.get("candidates") or []:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneArtifactRetentionError(
                "scene_artifact_retention_candidate_invalid"
            )
        path = Path(str(row.get("local_path") or "")).absolute()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("digest")
        ):
            raise TaskEvaluationSceneArtifactRetentionError(
                "scene_artifact_retention_candidate_changed"
            )
        path.unlink()
        removed.append(dict(row))
    return {
        "schema_version": "task_evaluation_scene_artifact_retention_result.v1",
        "status": "completed",
        "source_plan_digest": plan["plan_digest"],
        "removed_count": len(removed),
        "removed_bytes": sum(int(row["size_bytes"]) for row in removed),
        "removed": removed,
        "raw_secret_values_recorded": False,
    }


__all__ = [
    "APPLY_ACK",
    "TaskEvaluationSceneArtifactRetentionError",
    "apply_scene_artifact_retention",
    "plan_scene_artifact_retention",
    "seal_scene_artifact_lease",
]
