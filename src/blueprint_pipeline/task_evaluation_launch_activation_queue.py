"""Immutable queue for authority-gated Task Evaluation profile activation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_activation_contract import (
    launch_activation_request_digest,
    validate_launch_activation_request,
)
from .task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)


ENVELOPE_SCHEMA_VERSION = "task_evaluation_launch_activation_envelope.v1"
IDENTITY_SCHEMA_VERSION = "task_evaluation_launch_activation_identity.v1"
INTAKE_RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_launch_activation_intake_receipt.v1"
)
STATUS_SCHEMA_VERSION = "task_evaluation_launch_activation_status.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_launch_activation_result.v1"
QUEUE_STATES = ("pending", "processing", "prepared", "blocked")
_MAX_COMPONENT_BYTES = 255


class TaskEvaluationLaunchActivationQueueError(ValueError):
    """One immutable activation could not be queued or reopened safely."""


def _queue_filename(*, activation_id: str, request_digest: str) -> str:
    digest = request_digest.removeprefix("sha256:")
    readable = f"{activation_id}-{digest}.json"
    if len(readable.encode("utf-8")) <= _MAX_COMPONENT_BYTES:
        return readable
    identity = hashlib.sha256(activation_id.encode("utf-8")).hexdigest()
    return f"activation-{identity}-{digest}.json"


def _queue_matches(
    *, root: Path, activation_id: str, request_digest: str
) -> list[Path]:
    filename = _queue_filename(
        activation_id=activation_id, request_digest=request_digest
    )
    return [
        candidate
        for state in QUEUE_STATES
        if (candidate := root / state / filename).is_file()
    ]


def ensure_launch_activation_queue_root(queue_root: str | Path) -> Path:
    root = Path(queue_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_root_unsafe"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise OSError("activation queue root is not a directory")
        for name in (*QUEUE_STATES, "identities", "results"):
            child = resolved / name
            if child.is_symlink():
                raise OSError("activation queue child is a symlink")
            child.mkdir(mode=0o750, exist_ok=True)
    except OSError as exc:
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_root_unavailable"
        ) from exc
    return resolved


def _load_sealed(path: Path, *, schema_version: str, digest_field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_record_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != schema_version
        or value.get(digest_field)
        != canonical_digest(value, digest_field=digest_field)
    ):
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_record_invalid"
        )
    return dict(value)


def stage_launch_activation_request(
    *, value: Mapping[str, Any], queue_root: str | Path, submitted_by: str
) -> dict[str, Any]:
    """Validate and immutably queue one profile/authority activation request."""

    request = validate_launch_activation_request(value)
    activation_id = str(request["activation_id"])
    request_digest = launch_activation_request_digest(request)
    root = ensure_launch_activation_queue_root(queue_root)
    identity_path = root / "identities" / f"{activation_id}.json"
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "activation_id": activation_id,
        "request_digest": request_digest,
        "identity_digest": "",
    }
    identity["identity_digest"] = canonical_digest(
        identity, digest_field="identity_digest"
    )
    try:
        write_launch_preparation_record_exclusive(identity_path, identity)
    except FileExistsError:
        existing = _load_sealed(
            identity_path,
            schema_version=IDENTITY_SCHEMA_VERSION,
            digest_field="identity_digest",
        )
        if (
            existing.get("activation_id") != activation_id
            or existing.get("request_digest") != request_digest
        ):
            raise TaskEvaluationLaunchActivationQueueError(
                "launch_activation_id_immutable_conflict"
            )
    filename = _queue_filename(
        activation_id=activation_id, request_digest=request_digest
    )
    matches = _queue_matches(
        root=root,
        activation_id=activation_id,
        request_digest=request_digest,
    )
    if matches:
        exact = [path for path in matches if path.name == filename]
        if len(matches) != 1 or len(exact) != 1:
            raise TaskEvaluationLaunchActivationQueueError(
                "launch_activation_queue_identity_ambiguous"
            )
        return _intake_receipt(
            request=request, request_digest=request_digest, already_exists=True
        )
    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "request_digest": request_digest,
        "request": request,
        "submitted_by": submitted_by,
        "submitted_at_iso": datetime.now(timezone.utc).isoformat(),
        "provider_mutation_performed_inside_intake": False,
        "catalog_mutation_performed_inside_intake": False,
        "standing_authorization_published_inside_intake": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    try:
        write_launch_preparation_record_exclusive(
            root / "pending" / filename, envelope
        )
    except FileExistsError:
        return _intake_receipt(
            request=request, request_digest=request_digest, already_exists=True
        )
    return _intake_receipt(
        request=request, request_digest=request_digest, already_exists=False
    )


def _intake_receipt(
    *, request: Mapping[str, Any], request_digest: str, already_exists: bool
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": INTAKE_RECEIPT_SCHEMA_VERSION,
        "status": "queued_for_authority_gated_activation",
        "accepted": True,
        "already_exists": already_exists,
        "activation_id": request["activation_id"],
        "preparation_id": request["preparation"]["preparation_id"],
        "team_namespace": request["team_namespace"],
        "lane": request["lane"],
        "expected_production_commit": request["expected_production_commit"],
        "request_digest": request_digest,
        "provider_mutation_performed_inside_http_request": False,
        "catalog_mutation_performed_inside_http_request": False,
        "standing_authorization_published_inside_http_request": False,
        "paid_execution_requested": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def launch_activation_status(
    *, activation_id: str, queue_root: str | Path
) -> dict[str, Any]:
    """Read a safe activation state without exposing request bytes or host paths."""

    root = ensure_launch_activation_queue_root(queue_root)
    identity_path = root / "identities" / f"{activation_id}.json"
    if not identity_path.is_file():
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": "not_found",
            "activation_id": activation_id,
            "provider_mutation_performed_by_status_read": False,
        }
    identity = _load_sealed(
        identity_path,
        schema_version=IDENTITY_SCHEMA_VERSION,
        digest_field="identity_digest",
    )
    if identity.get("activation_id") != activation_id:
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_record_invalid"
        )
    request_digest = str(identity.get("request_digest") or "")
    paths = _queue_matches(
        root=root,
        activation_id=activation_id,
        request_digest=request_digest,
    )
    matches = [
        (state, path)
        for state in QUEUE_STATES
        for path in paths
        if path.parent.name == state
    ]
    if not matches:
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_record_invalid"
        )
    if len(matches) != 1:
        raise TaskEvaluationLaunchActivationQueueError(
            "launch_activation_queue_identity_ambiguous"
        )
    state, path = matches[0]
    envelope = _load_sealed(
        path,
        schema_version=ENVELOPE_SCHEMA_VERSION,
        digest_field="envelope_digest",
    )
    request = envelope["request"]
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": state,
        "activation_id": activation_id,
        "preparation_id": request["preparation"]["preparation_id"],
        "team_namespace": request["team_namespace"],
        "lane": request["lane"],
        "expected_production_commit": request["expected_production_commit"],
        "request_digest": envelope["request_digest"],
        "provider_mutation_performed_by_status_read": False,
    }
    result_path = root / "results" / path.name
    if result_path.exists():
        result = _load_sealed(
            result_path,
            schema_version=RESULT_SCHEMA_VERSION,
            digest_field="result_digest",
        )
        if result.get("activation_id") != activation_id:
            raise TaskEvaluationLaunchActivationQueueError(
                "launch_activation_queue_record_invalid"
            )
        status.update(
            {
                "worker_status": result.get("status"),
                "result_digest": result.get("result_digest"),
                "profile_id": result.get("profile_id"),
                "profile_digest": result.get("profile_digest"),
                "profile_publication_receipt_digest": result.get(
                    "profile_publication_receipt_digest"
                ),
                "standing_authorization_digest": result.get(
                    "standing_authorization_digest"
                ),
                "policy_campaign_activation_digest": result.get(
                    "policy_campaign_activation_digest"
                ),
                "campaign_unit_count": result.get("campaign_unit_count"),
                "blockers": list(result.get("blockers") or []),
                "provider_mutation_performed_by_worker": result.get(
                    "provider_mutation_performed"
                ),
                "paid_execution_requested": result.get(
                    "paid_execution_requested"
                ),
            }
        )
    return status


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "INTAKE_RECEIPT_SCHEMA_VERSION",
    "QUEUE_STATES",
    "RESULT_SCHEMA_VERSION",
    "STATUS_SCHEMA_VERSION",
    "TaskEvaluationLaunchActivationQueueError",
    "ensure_launch_activation_queue_root",
    "launch_activation_status",
    "stage_launch_activation_request",
]
