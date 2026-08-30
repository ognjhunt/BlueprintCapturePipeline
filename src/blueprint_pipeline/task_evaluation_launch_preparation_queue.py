"""Immutable no-spend queue for customer Task Evaluation preparation requests."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError,
    launch_preparation_request_digest,
    validate_launch_preparation_request,
)
from .task_evaluation_release_reference_lock import release_reference_lock


ENVELOPE_SCHEMA_VERSION = "task_evaluation_launch_preparation_envelope.v1"
IDENTITY_SCHEMA_VERSION = "task_evaluation_launch_preparation_identity.v1"
INTAKE_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_preparation_intake_receipt.v1"
QUEUE_STATES = ("pending", "processing", "materialized", "completed", "blocked")


class TaskEvaluationLaunchPreparationQueueError(ValueError):
    """The immutable preparation request could not be staged safely."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def ensure_launch_preparation_queue_root(queue_root: str | Path) -> Path:
    root = Path(queue_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_root_unsafe"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        resolved = root.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_root_unavailable"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_root_unsafe"
        )
    for state in QUEUE_STATES:
        child = resolved / state
        if child.is_symlink():
            raise TaskEvaluationLaunchPreparationQueueError(
                "launch_preparation_queue_state_unsafe"
            )
        child.mkdir(mode=0o750, exist_ok=True)
    identities = resolved / "identities"
    if identities.is_symlink():
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_identity_root_unsafe"
        )
    identities.mkdir(mode=0o750, exist_ok=True)
    return resolved


def _write_launch_preparation_record_exclusive_locked(
    path: Path, value: Mapping[str, Any]
) -> None:
    payload = _canonical_bytes(value)
    path_token = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    temporary_path = path.with_name(
        f".queue-{path_token}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable preparation queue write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary_path, path, follow_symlinks=False)
        temporary_path.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        raise
    except OSError as exc:
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_write_failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def write_launch_preparation_record_exclusive(
    path: Path, value: Mapping[str, Any]
) -> None:
    with release_reference_lock(path.parents[2], exclusive=False):
        _write_launch_preparation_record_exclusive_locked(path, value)


def stage_launch_preparation_request(
    *,
    value: Mapping[str, Any],
    queue_root: str | Path,
    submitted_by: str,
) -> dict[str, Any]:
    """Validate and immutably queue one no-spend preparation request."""

    try:
        request = validate_launch_preparation_request(value)
    except TaskEvaluationLaunchPreparationContractError:
        raise
    preparation_id = str(request["preparation_id"])
    request_digest = launch_preparation_request_digest(request)
    root = ensure_launch_preparation_queue_root(queue_root)
    identity_path = root / "identities" / f"{preparation_id}.json"
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "preparation_id": preparation_id,
        "request_digest": request_digest,
        "identity_digest": "",
    }
    identity["identity_digest"] = canonical_digest(
        identity, digest_field="identity_digest"
    )
    try:
        write_launch_preparation_record_exclusive(identity_path, identity)
    except FileExistsError:
        try:
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationLaunchPreparationQueueError(
                "launch_preparation_queue_identity_invalid"
            ) from exc
        if (
            identity_path.is_symlink()
            or not isinstance(existing_identity, Mapping)
            or existing_identity.get("schema_version") != IDENTITY_SCHEMA_VERSION
            or existing_identity.get("identity_digest")
            != canonical_digest(existing_identity, digest_field="identity_digest")
            or existing_identity.get("preparation_id") != preparation_id
        ):
            raise TaskEvaluationLaunchPreparationQueueError(
                "launch_preparation_queue_identity_invalid"
            )
        if existing_identity.get("request_digest") != request_digest:
            raise TaskEvaluationLaunchPreparationQueueError(
                "launch_preparation_id_immutable_conflict"
            )
    matches = [
        path
        for state in QUEUE_STATES
        for path in (root / state).glob(f"{preparation_id}-*.json")
    ]
    if matches:
        exact_name = f"{preparation_id}-{request_digest.removeprefix('sha256:')}.json"
        exact = [path for path in matches if path.name == exact_name]
        if len(exact) == 1 and len(matches) == 1:
            return _intake_receipt(
                request=request,
                request_digest=request_digest,
                queue_path=exact[0],
                already_exists=True,
            )
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_identity_ambiguous"
        )
    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "request_digest": request_digest,
        "request": request,
        "submitted_by": submitted_by,
        "submitted_at_iso": datetime.now(timezone.utc).isoformat(),
        "provider_mutation_performed_inside_intake": False,
        "catalog_mutation_performed_inside_intake": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    destination = (
        root
        / "pending"
        / f"{preparation_id}-{request_digest.removeprefix('sha256:')}.json"
    )
    try:
        write_launch_preparation_record_exclusive(destination, envelope)
    except FileExistsError:
        return _intake_receipt(
            request=request,
            request_digest=request_digest,
            queue_path=destination,
            already_exists=True,
        )
    return _intake_receipt(
        request=request,
        request_digest=request_digest,
        queue_path=destination,
        already_exists=False,
    )


def _intake_receipt(
    *,
    request: Mapping[str, Any],
    request_digest: str,
    queue_path: Path,
    already_exists: bool,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": INTAKE_RECEIPT_SCHEMA_VERSION,
        "status": "queued_for_no_spend_preparation",
        "accepted": True,
        "already_exists": already_exists,
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "request_digest": request_digest,
        "queue_path": str(queue_path),
        "provider_mutation_performed_inside_http_request": False,
        "catalog_mutation_performed_inside_http_request": False,
        "paid_execution_requested": False,
        "canonical_allocator_required_for_later_execution": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def launch_preparation_status(
    *, preparation_id: str, queue_root: str | Path
) -> dict[str, Any]:
    """Return the one immutable queue state without exposing request bytes."""

    root = ensure_launch_preparation_queue_root(queue_root)
    matches = [
        (state, path)
        for state in QUEUE_STATES
        for path in (root / state).glob(f"{preparation_id}-*.json")
    ]
    if not matches:
        return {
            "schema_version": "task_evaluation_launch_preparation_status.v1",
            "status": "not_found",
            "preparation_id": preparation_id,
            "provider_mutation_performed_by_status_read": False,
        }
    if len(matches) != 1:
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_identity_ambiguous"
        )
    state, path = matches[0]
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_envelope_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(envelope, Mapping)
        or envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
    ):
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_queue_envelope_invalid"
        )
    status: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_preparation_status.v1",
        "status": state,
        "preparation_id": preparation_id,
        "run_id": envelope["request"]["run_id"],
        "team_namespace": envelope["request"]["team_namespace"],
        "expected_production_commit": envelope["request"][
            "expected_production_commit"
        ],
        "run_mode": envelope["request"]["run_mode"],
        "request_digest": envelope["request_digest"],
        "provider_mutation_performed_by_status_read": False,
    }
    result_path = root / "results" / path.name
    conflict_paths = sorted(
        (root / "results" / "conflicts").glob(f"{path.stem}-*.json")
    ) if (root / "results" / "conflicts").is_dir() else []
    if len(conflict_paths) > 1:
        raise TaskEvaluationLaunchPreparationQueueError(
            "launch_preparation_result_conflict_ambiguous"
        )
    if conflict_paths:
        result_path = conflict_paths[0]
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationLaunchPreparationQueueError(
                "launch_preparation_result_invalid"
            ) from exc
        if (
            result_path.is_symlink()
            or not isinstance(result, Mapping)
            or result.get("schema_version")
            != "task_evaluation_launch_preparation_result.v1"
            or result.get("result_digest")
            != canonical_digest(result, digest_field="result_digest")
            or result.get("preparation_id") != preparation_id
        ):
            raise TaskEvaluationLaunchPreparationQueueError(
                "launch_preparation_result_invalid"
            )
        status.update(
            {
                "worker_status": result.get("status"),
                "source_commit": result.get("source_commit"),
                "result_digest": result.get("result_digest"),
                "reference_count": result.get("reference_count"),
                "full_byte_service_account_readback_passed": result.get(
                    "full_byte_service_account_readback_passed"
                ),
                "blockers": list(result.get("blockers") or []),
                "provider_mutation_performed_by_worker": result.get(
                    "provider_mutation_performed"
                ),
                "catalog_mutation_performed_by_worker": result.get(
                    "catalog_mutation_performed"
                ),
                "paid_execution_requested": result.get(
                    "paid_execution_requested"
                ),
            }
        )
        for field in (
            "construction_orchestration_id",
            "construction_queue_envelope_digest",
            "automatic_progression_required",
            "configured_scene_revision_digest",
            "configured_scene_bundle_digest",
            "episode_compilation_id",
            "episode_compilation_queue_envelope_digest",
        ):
            if result.get(field) is not None:
                status[field] = result[field]
        policy_plan = result.get("policy_run_plan")
        if isinstance(policy_plan, Mapping):
            status["policy_run"] = {
                "status": policy_plan.get("status"),
                "preset_id": policy_plan.get("preset_id"),
                "counts": dict(policy_plan.get("counts") or {}),
                "configuration_digest": policy_plan.get(
                    "configuration_digest"
                ),
                "plan_digest": policy_plan.get("plan_digest"),
                "provider_mutation_performed": policy_plan.get(
                    "provider_mutation_performed"
                ),
                "paid_execution_requested": policy_plan.get(
                    "paid_execution_requested"
                ),
            }
    return status


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "INTAKE_RECEIPT_SCHEMA_VERSION",
    "TaskEvaluationLaunchPreparationQueueError",
    "ensure_launch_preparation_queue_root",
    "launch_preparation_status",
    "stage_launch_preparation_request",
    "write_launch_preparation_record_exclusive",
]
