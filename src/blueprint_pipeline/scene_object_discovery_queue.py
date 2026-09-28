"""Durable, idempotent queue and selection store for scene object discovery."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_digest
from .scene_object_discovery_contract import (
    scene_object_discovery_request_digest,
    validate_scene_object_discovery_request,
)


ENVELOPE_SCHEMA = "scene_object_discovery_envelope.v1"
INTAKE_RECEIPT_SCHEMA = "scene_object_discovery_intake_receipt.v1"
STATUS_SCHEMA = "scene_object_discovery_status.v1"
SELECTION_REQUEST_SCHEMA = "scene_object_discovery_selection_request.v1"
SELECTION_RECEIPT_SCHEMA = "scene_object_discovery_selection_receipt.v1"
QUEUE_STATES = ("pending", "processing", "blocked", "results")


class SceneObjectDiscoveryQueueError(ValueError):
    """Queue state is unsafe, conflicting, or invalid."""


def ensure_scene_object_discovery_queue_root(queue_root: str | Path) -> Path:
    root = Path(queue_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_queue_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve(strict=True)
    for name in (*QUEUE_STATES, "identities", "selections"):
        child = root / name
        if child.is_symlink():
            raise SceneObjectDiscoveryQueueError("scene_object_discovery_queue_state_unsafe")
        child.mkdir(parents=True, exist_ok=True, mode=0o750)
    return root


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise SceneObjectDiscoveryQueueError(
                "scene_object_discovery_queue_immutable_conflict"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_queue_record_invalid") from exc
    if not isinstance(value, Mapping):
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_queue_record_invalid")
    return dict(value)


def stage_scene_object_discovery_request(
    *,
    value: Mapping[str, Any],
    queue_root: str | Path,
    submitted_by: str,
) -> dict[str, Any]:
    request = validate_scene_object_discovery_request(value)
    discovery_id = strict_identifier(request["discovery_id"], field="discovery_id", max_length=192)
    request_digest = scene_object_discovery_request_digest(request)
    root = ensure_scene_object_discovery_queue_root(queue_root)
    identity_path = root / "identities" / f"{discovery_id}.json"
    identity = _read_mapping(identity_path)
    already_exists = bool(identity)
    if identity and identity.get("request_digest") != request_digest:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_id_immutable_conflict")
    if not identity:
        identity = {
            "schema_version": "scene_object_discovery_identity.v1",
            "discovery_id": discovery_id,
            "request_digest": request_digest,
            "team_namespace": request["team_namespace"],
            "expected_production_commit": request["expected_production_commit"],
        }
        _write_json_exclusive(identity_path, identity)
    envelope = {
        "schema_version": ENVELOPE_SCHEMA,
        "discovery_id": discovery_id,
        "request_digest": request_digest,
        "request": request,
        "submitted_by": str(submitted_by or "unknown")[:192],
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    pending_path = root / "pending" / f"{discovery_id}.json"
    if not already_exists and not pending_path.exists():
        _write_json_exclusive(pending_path, envelope)
    receipt = {
        "schema_version": INTAKE_RECEIPT_SCHEMA,
        "status": "queued_for_no_spend_discovery_preparation",
        "accepted": True,
        "already_exists": already_exists,
        "discovery_id": discovery_id,
        "team_namespace": request["team_namespace"],
        "request_digest": request_digest,
        "expected_production_commit": request["expected_production_commit"],
        "provider_mutation_performed_inside_http_request": False,
        "paid_execution_requested": False,
        "canonical_allocator_required_for_provider_execution": True,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def seal_scene_object_discovery_result(
    *,
    queue_root: str | Path,
    discovery_id: str,
    request_digest: str,
    source_commit: str,
    discovery: Mapping[str, Any],
    source_object_artifact: Mapping[str, Any] | None = None,
    paid_execution_performed: bool = False,
) -> dict[str, Any]:
    root = ensure_scene_object_discovery_queue_root(queue_root)
    normalized_id = strict_identifier(discovery_id, field="discovery_id", max_length=192)
    identity = _read_mapping(root / "identities" / f"{normalized_id}.json")
    if not identity or identity.get("request_digest") != request_digest:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_result_identity_mismatch")
    output = {
        "schema_version": "scene_object_discovery_queue_result.v1",
        "discovery_id": normalized_id,
        "team_namespace": identity["team_namespace"],
        "expected_production_commit": identity["expected_production_commit"],
        "request_digest": request_digest,
        "source_commit": source_commit,
        "discovery": dict(discovery),
        "source_object_artifact": dict(source_object_artifact or {}),
        "paid_execution_performed": bool(paid_execution_performed),
        "provider_mutation_performed_by_worker": False,
    }
    output["result_digest"] = canonical_digest(output, digest_field="result_digest")
    result_path = root / "results" / f"{normalized_id}.json"
    existing = _read_mapping(result_path)
    if existing:
        if existing != output:
            raise SceneObjectDiscoveryQueueError("scene_object_discovery_result_immutable_conflict")
        return existing
    _write_json_exclusive(result_path, output)
    (root / "pending" / f"{normalized_id}.json").unlink(missing_ok=True)
    (root / "processing" / f"{normalized_id}.json").unlink(missing_ok=True)
    return output


def claim_scene_object_discovery_request(
    *, queue_root: str | Path, pending_path: str | Path
) -> tuple[Path, dict[str, Any]] | None:
    """Atomically claim one pending envelope for the no-spend worker."""

    root = ensure_scene_object_discovery_queue_root(queue_root)
    source = Path(pending_path)
    try:
        source = source.resolve(strict=True)
    except OSError:
        return None
    pending_root = (root / "pending").resolve(strict=True)
    if source.parent != pending_root or source.suffix != ".json" or source.is_symlink():
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_pending_path_unsafe")
    claimed = root / "processing" / source.name
    try:
        descriptor = os.open(claimed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    else:
        os.close(descriptor)
    try:
        os.replace(source, claimed)
        envelope = _read_mapping(claimed)
    except Exception:
        claimed.unlink(missing_ok=True)
        raise
    if (
        envelope.get("schema_version") != ENVELOPE_SCHEMA
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or not isinstance(envelope.get("request"), Mapping)
    ):
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_envelope_invalid")
    return claimed, envelope


def seal_scene_object_discovery_blocked(
    *,
    queue_root: str | Path,
    discovery_id: str,
    request_digest: str,
    source_commit: str,
    blockers: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Seal a typed terminal blocker without losing the claimed envelope."""

    root = ensure_scene_object_discovery_queue_root(queue_root)
    normalized_id = strict_identifier(discovery_id, field="discovery_id", max_length=192)
    identity = _read_mapping(root / "identities" / f"{normalized_id}.json")
    normalized_blockers = sorted({str(item).strip() for item in blockers if str(item).strip()})
    if not identity or identity.get("request_digest") != request_digest or not normalized_blockers:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_blocked_identity_invalid")
    output = {
        "schema_version": "scene_object_discovery_blocked.v1",
        "status": "blocked",
        "discovery_id": normalized_id,
        "request_digest": request_digest,
        "source_commit": source_commit,
        "blockers": normalized_blockers,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }
    output["blocked_digest"] = canonical_digest(output, digest_field="blocked_digest")
    destination = root / "blocked" / f"{normalized_id}.json"
    existing = _read_mapping(destination)
    if existing and existing != output:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_blocked_immutable_conflict")
    if not existing:
        _write_json_exclusive(destination, output)
    (root / "processing" / f"{normalized_id}.json").unlink(missing_ok=True)
    (root / "pending" / f"{normalized_id}.json").unlink(missing_ok=True)
    return output


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "candidate_id": candidate.get("candidate_id"),
        "label": candidate.get("label"),
        "backend": candidate.get("backend"),
        "confidence": candidate.get("confidence"),
        "task_match_score": candidate.get("task_match_score"),
        "eligible_for_automatic_source_object": candidate.get(
            "eligible_for_automatic_source_object"
        )
        is True,
        "candidate_claim_boundary": candidate.get("candidate_claim_boundary"),
    }
    preview = candidate.get("preview")
    if (
        isinstance(preview, Mapping)
        and set(preview) == {"uri", "digest", "size_bytes"}
        and isinstance(preview.get("uri"), str)
        and str(preview["uri"]).startswith(("gs://", "s3://", "https://"))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(preview.get("digest") or ""))
        and isinstance(preview.get("size_bytes"), int)
        and not isinstance(preview.get("size_bytes"), bool)
        and int(preview["size_bytes"]) > 0
    ):
        row["preview"] = dict(preview)
    return row


def scene_object_discovery_status(*, discovery_id: str, queue_root: str | Path) -> dict[str, Any]:
    root = ensure_scene_object_discovery_queue_root(queue_root)
    normalized_id = strict_identifier(discovery_id, field="discovery_id", max_length=192)
    identity = _read_mapping(root / "identities" / f"{normalized_id}.json")
    base = {
        "schema_version": STATUS_SCHEMA,
        "status": "not_found",
        "discovery_id": normalized_id,
        "provider_mutation_performed_by_status_read": False,
    }
    if not identity:
        return base
    base.update(
        {
            "team_namespace": identity["team_namespace"],
            "expected_production_commit": identity["expected_production_commit"],
            "request_digest": identity["request_digest"],
        }
    )
    selection = _read_mapping(root / "selections" / f"{normalized_id}.json")
    if selection:
        base.update(
            {
                "status": "ready_auto_selected",
                "discovery_digest": selection["discovery_digest"],
                "source_commit": selection["source_commit"],
                "candidates": selection["candidates"],
                "selected_candidate_id": selection["candidate_id"],
                "source_object": selection["source_object"],
                "unseen_regions": selection["unseen_regions"],
                "blockers": [],
                "paid_execution_performed": selection["paid_execution_performed"],
            }
        )
        return base
    result = _read_mapping(root / "results" / f"{normalized_id}.json")
    if result:
        discovery = result.get("discovery")
        if not isinstance(discovery, Mapping):
            raise SceneObjectDiscoveryQueueError("scene_object_discovery_queue_result_invalid")
        source_object = discovery.get("source_object")
        artifact = result.get("source_object_artifact")
        public_source_object = None
        if isinstance(source_object, Mapping) and isinstance(artifact, Mapping) and artifact:
            public_source_object = {
                "object_id": source_object.get("object_id"),
                "label": source_object.get("label"),
                "metric_geometry_authority": source_object.get("metric_geometry_authority"),
                "metric_geometry_evidence_digest": source_object.get(
                    "metric_geometry_evidence_digest"
                ),
                "source_object_artifact": dict(artifact),
            }
        base.update(
            {
                "status": discovery.get("status"),
                "discovery_digest": discovery.get("discovery_digest"),
                "source_commit": result.get("source_commit"),
                "candidates": [
                    _public_candidate(row)
                    for row in discovery.get("candidates", [])
                    if isinstance(row, Mapping)
                ],
                "selected_candidate_id": discovery.get("selected_candidate_id"),
                "source_object": public_source_object,
                "unseen_regions": list(
                    (discovery.get("coverage") or {}).get("unseen_regions") or []
                ),
                "blockers": [],
                "paid_execution_performed": result.get("paid_execution_performed") is True,
            }
        )
        return base
    if (root / "blocked" / f"{normalized_id}.json").is_file():
        blocked = _read_mapping(root / "blocked" / f"{normalized_id}.json")
        base.update({"status": "blocked", "blockers": list(blocked.get("blockers") or [])})
    elif (root / "processing" / f"{normalized_id}.json").is_file():
        base["status"] = "processing"
    else:
        base["status"] = "pending"
    return base


def select_scene_object_candidate(
    *, value: Mapping[str, Any], queue_root: str | Path
) -> dict[str, Any]:
    required = {
        "schema_version",
        "discovery_id",
        "expected_production_commit",
        "request_digest",
        "discovery_digest",
        "candidate_id",
        "confirm_selection",
    }
    if set(value) != required or value.get("schema_version") != SELECTION_REQUEST_SCHEMA:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_selection_request_invalid")
    discovery_id = strict_identifier(
        value.get("discovery_id"), field="discovery_id", max_length=192
    )
    candidate_id = strict_identifier(
        value.get("candidate_id"), field="candidate_id", max_length=192
    )
    if value.get("confirm_selection") is not True:
        raise SceneObjectDiscoveryQueueError(
            "scene_object_discovery_selection_confirmation_required"
        )
    root = ensure_scene_object_discovery_queue_root(queue_root)
    result = _read_mapping(root / "results" / f"{discovery_id}.json")
    discovery = result.get("discovery") if isinstance(result, Mapping) else None
    if not isinstance(discovery, Mapping) or (
        result.get("request_digest") != value.get("request_digest")
        or result.get("expected_production_commit") != value.get("expected_production_commit")
        or discovery.get("discovery_digest") != value.get("discovery_digest")
        or discovery.get("status") != "selection_required"
    ):
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_selection_identity_mismatch")
    matches = [
        row
        for row in discovery.get("candidates", [])
        if isinstance(row, Mapping)
        and row.get("candidate_id") == candidate_id
        and row.get("eligible_for_automatic_source_object") is True
    ]
    if len(matches) != 1:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_candidate_not_selectable")
    artifact = matches[0].get("source_object_artifact")
    if not isinstance(artifact, Mapping):
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_candidate_artifact_missing")
    metric = matches[0].get("metric_geometry")
    if not isinstance(metric, Mapping):
        raise SceneObjectDiscoveryQueueError(
            "scene_object_discovery_candidate_metric_geometry_missing"
        )
    source_object = {
        "object_id": candidate_id,
        "label": matches[0].get("label"),
        "metric_geometry_authority": matches[0].get("metric_geometry_authority"),
        "metric_geometry_evidence_digest": metric.get("evidence_digest"),
        "source_object_artifact": dict(artifact),
    }
    selection = {
        "schema_version": "scene_object_discovery_selection.v1",
        "discovery_id": discovery_id,
        "request_digest": value["request_digest"],
        "discovery_digest": value["discovery_digest"],
        "candidate_id": candidate_id,
        "source_commit": result["source_commit"],
        "source_object": source_object,
        "candidates": [
            _public_candidate(row)
            for row in discovery.get("candidates", [])
            if isinstance(row, Mapping)
        ],
        "unseen_regions": list((discovery.get("coverage") or {}).get("unseen_regions") or []),
        "paid_execution_performed": result.get("paid_execution_performed") is True,
    }
    selection["selection_digest"] = canonical_digest(dict(value), digest_field="selection_digest")
    path = root / "selections" / f"{discovery_id}.json"
    existing = _read_mapping(path)
    if existing and existing != selection:
        raise SceneObjectDiscoveryQueueError("scene_object_discovery_selection_immutable_conflict")
    if not existing:
        _write_json_exclusive(path, selection)
    receipt = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "status": "selection_sealed",
        "discovery_id": discovery_id,
        "request_digest": value["request_digest"],
        "discovery_digest": value["discovery_digest"],
        "candidate_id": candidate_id,
        "selection_digest": selection["selection_digest"],
        "provider_mutation_performed_inside_http_request": False,
        "paid_execution_requested": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "SceneObjectDiscoveryQueueError",
    "ensure_scene_object_discovery_queue_root",
    "scene_object_discovery_status",
    "seal_scene_object_discovery_result",
    "select_scene_object_candidate",
    "stage_scene_object_discovery_request",
]
