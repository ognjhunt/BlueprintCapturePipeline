from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_artifact_retention import (
    APPLY_ACK,
    TaskEvaluationSceneArtifactRetentionError,
    apply_scene_artifact_retention,
    plan_scene_artifact_retention,
    seal_scene_artifact_lease,
)


def _reference(path: Path, *, kind: str = "provider-output") -> dict[str, object]:
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "task_evaluation_scene_artifact_reference.v1",
        "status": "remote_verified",
        "artifact_kind": kind,
        "uri": f"s3://bucket/prefix/sha256/{digest.removeprefix('sha256:')}/{path.name}",
        "digest": digest,
        "size_bytes": path.stat().st_size,
        "remote_identity_verified": True,
        "full_byte_service_account_readback_passed": True,
        "raw_secret_values_recorded": False,
    }


def _now() -> datetime:
    return datetime(2026, 8, 28, 18, 0, tzinfo=UTC)


def test_pending_and_inflight_leases_never_expire_silently(tmp_path: Path) -> None:
    artifact = tmp_path / "provider-output.zip"
    artifact.write_bytes(b"provider output")
    reference = _reference(artifact)
    leases = [
        seal_scene_artifact_lease(
            destination=tmp_path / f"{state}.lease.json",
            run_id=f"run-{state}",
            lifecycle_state=state,
            artifact_references=[reference],
        )
        for state in ("pending", "in_flight")
    ]

    plan = plan_scene_artifact_retention(
        artifacts=[{"local_path": artifact, "remote_reference": reference}],
        leases=leases,
        protected_roots=[],
        now=(_now() + timedelta(days=365)).isoformat(),
    )

    assert plan["status"] == "completed"
    assert plan["candidates"] == []
    assert plan["protected"][0]["reason"] == "active_lease"


def test_expired_retry_lease_allows_exact_remote_verified_local_copy(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "checkpoint.zip"
    artifact.write_bytes(b"checkpoint")
    reference = _reference(artifact, kind="diagnostic-checkpoint")
    lease = seal_scene_artifact_lease(
        destination=tmp_path / "retry.lease.json",
        run_id="run-1",
        lifecycle_state="retained_for_retry",
        artifact_references=[reference],
        expires_at=(_now() - timedelta(seconds=1)).isoformat(),
    )

    plan = plan_scene_artifact_retention(
        artifacts=[{"local_path": artifact, "remote_reference": reference}],
        leases=[lease],
        protected_roots=[],
        now=_now().isoformat(),
    )

    assert plan["status"] == "completed"
    assert plan["candidate_count"] == 1
    assert plan["candidate_bytes"] == len(b"checkpoint")


def test_unuploaded_artifact_blocks_every_deletion(tmp_path: Path) -> None:
    safe = tmp_path / "safe.zip"
    unsafe = tmp_path / "unsafe.zip"
    safe.write_bytes(b"safe")
    unsafe.write_bytes(b"unsafe")
    invalid_reference = _reference(unsafe)
    invalid_reference["remote_identity_verified"] = False

    plan = plan_scene_artifact_retention(
        artifacts=[
            {"local_path": safe, "remote_reference": _reference(safe)},
            {"local_path": unsafe, "remote_reference": invalid_reference},
        ],
        leases=[],
        protected_roots=[],
        now=_now().isoformat(),
    )

    assert plan["status"] == "blocked"
    assert plan["candidates"] == []
    assert plan["blockers"] == [
        f"scene_artifact_remote_reference_invalid:{unsafe.absolute()}"
    ]


def test_active_release_or_run_root_is_never_candidate(tmp_path: Path) -> None:
    active = tmp_path / "active-release" / "bundle.zip"
    active.parent.mkdir()
    active.write_bytes(b"bundle")

    plan = plan_scene_artifact_retention(
        artifacts=[{"local_path": active, "remote_reference": _reference(active)}],
        leases=[],
        protected_roots=[active.parent],
        now=_now().isoformat(),
    )

    assert plan["candidates"] == []
    assert plan["protected"][0]["reason"] == "protected_root"


def test_apply_rechecks_identity_and_requires_explicit_ack(tmp_path: Path) -> None:
    artifact = tmp_path / "provider-output.zip"
    artifact.write_bytes(b"provider output")
    reference = _reference(artifact)
    plan = plan_scene_artifact_retention(
        artifacts=[{"local_path": artifact, "remote_reference": reference}],
        leases=[],
        protected_roots=[],
        now=_now().isoformat(),
    )

    with pytest.raises(
        TaskEvaluationSceneArtifactRetentionError,
        match="scene_artifact_retention_plan_invalid",
    ):
        apply_scene_artifact_retention(plan=plan, ack="wrong")
    artifact.write_bytes(b"changed")
    with pytest.raises(
        TaskEvaluationSceneArtifactRetentionError,
        match="scene_artifact_retention_candidate_changed",
    ):
        apply_scene_artifact_retention(plan=plan, ack=APPLY_ACK)
    assert artifact.is_file()


def test_apply_removes_only_exact_planned_file(tmp_path: Path) -> None:
    artifact = tmp_path / "provider-output.zip"
    sibling = tmp_path / "receipt.json"
    artifact.write_bytes(b"provider output")
    sibling.write_text("{}\n", encoding="utf-8")
    plan = plan_scene_artifact_retention(
        artifacts=[{"local_path": artifact, "remote_reference": _reference(artifact)}],
        leases=[],
        protected_roots=[],
        now=_now().isoformat(),
    )

    result = apply_scene_artifact_retention(plan=plan, ack=APPLY_ACK)

    assert result["removed_count"] == 1
    assert not artifact.exists()
    assert sibling.is_file()
