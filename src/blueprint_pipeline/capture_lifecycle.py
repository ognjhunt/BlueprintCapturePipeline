"""Fail-closed revocation and retention lifecycle for completed capture uploads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import fcntl
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_digest, canonical_json


LIFECYCLE_ACTIONS = {
    "consent_revoked",
    "operator_deletion_request",
    "retention_expired",
}
PROVIDER_DELETION_VERIFICATION_METHODS = {
    "provider_api_receipt",
    "provider_signed_receipt",
    "operator_console_receipt",
}
EXTERNAL_REVOCATION_VERIFICATION_METHODS = {
    "signed_webapp_receipt",
    "storage_access_revocation_receipt",
}


class CaptureLifecycleError(ValueError):
    def __init__(self, code: str, *, status_code: int = 422) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read(path: Path, *, code: str, status_code: int = 404) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureLifecycleError(code, status_code=status_code) from exc
    if not isinstance(value, Mapping):
        raise CaptureLifecycleError(code)
    return dict(value)


def _safe_child(root: Path, relative: str, *, code: str) -> Path:
    value = PurePosixPath(str(relative).replace("\\", "/"))
    if not relative or value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise CaptureLifecycleError(code)
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / Path(*value.parts)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise CaptureLifecycleError(code)
    return candidate


def _write_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = _read(path, code=f"lifecycle_artifact_invalid:{path.name}")
        if canonical_json(existing) != canonical_json(normalized):
            raise CaptureLifecycleError(
                f"lifecycle_artifact_conflict:{path.name}", status_code=409
            )
        return existing
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _read(path, code=f"lifecycle_artifact_invalid:{path.name}")
            if canonical_json(existing) != canonical_json(normalized):
                raise CaptureLifecycleError(
                    f"lifecycle_artifact_conflict:{path.name}", status_code=409
                )
            return existing
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _lifecycle_key(capture_session_id: str, intake_id: str) -> str:
    return hashlib.sha256(f"{capture_session_id}\0{intake_id}".encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _parse_time(value: Any, *, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureLifecycleError(code) from exc
    if parsed.tzinfo is None:
        raise CaptureLifecycleError(code)
    return parsed.astimezone(timezone.utc)


def _delete_tree(root: Path) -> tuple[int, str, list[str]]:
    if not root.exists():
        return 0, canonical_digest({"files": []}), []
    if not root.is_dir() or root.is_symlink():
        raise CaptureLifecycleError("lifecycle_delete_root_invalid")
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CaptureLifecycleError("lifecycle_delete_symlink_forbidden")
        if path.is_file():
            inventory.append(
                {
                    "relative_path_digest": "sha256:"
                    + hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest(),
                    "content_digest": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    inventory_digest = canonical_digest({"files": inventory})
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()
    return (
        len(inventory),
        inventory_digest,
        sorted({str(row["content_digest"]) for row in inventory}),
    )


def _other_active_references(
    store_root: Path, *, target_artifact_root: Path, capture_digest: str
) -> list[str]:
    references: list[str] = []
    intake_root = store_root / "intakes"
    if not intake_root.is_dir():
        return references
    for path in sorted(intake_root.glob("*/*/capture_intake_object_manifest.json")):
        if path.parent.resolve() == target_artifact_root.resolve():
            continue
        try:
            manifest = _read(path, code="shared_object_manifest_invalid", status_code=422)
        except CaptureLifecycleError:
            # An unreadable possible reference must preserve the shared object.
            references.append("unverifiable_reference")
            continue
        if any(
            isinstance(row, Mapping) and row.get("sha256") == capture_digest
            for row in manifest.get("objects", [])
        ):
            references.append(
                "sha256:" + hashlib.sha256(str(path.parent.relative_to(store_root)).encode()).hexdigest()
            )
    return sorted(set(references))


def _provider_obligations(work_root: Path, *, intake_id: str, capture_digest: str) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    plans = work_root / "reconstruction_control_plane" / "plans"
    if not plans.is_dir():
        return obligations
    for result_path in sorted(plans.glob("*/artifacts/execution_result.json")):
        execution = _read(result_path, code="reconstruction_execution_result_invalid", status_code=422)
        for result in execution.get("results", []):
            if not isinstance(result, Mapping):
                continue
            provider = str(result.get("provider_identity") or "").strip()
            if (
                result.get("intake_id") != intake_id
                or result.get("capture_digest") != capture_digest
                or provider in {"", "local"}
            ):
                continue
            obligation = {
                "provider_identity": provider,
                "reconstruction_result_digest": result.get("reconstruction_result_digest"),
                "provider_receipt_digest": canonical_digest(
                    _mapping(result.get("provider_receipt"))
                ),
                "deletion_evidence_present_at_revocation": isinstance(
                    result.get("deletion_evidence"), Mapping
                ),
            }
            obligation["obligation_digest"] = canonical_digest(
                obligation, digest_field="obligation_digest"
            )
            obligations.append(obligation)
    return sorted(obligations, key=lambda row: row["obligation_digest"])


def _contains_binding(value: Any, *, values: set[str]) -> bool:
    if isinstance(value, str):
        return value in values
    if isinstance(value, Mapping):
        return any(_contains_binding(item, values=values) for item in value.values())
    if isinstance(value, list):
        return any(_contains_binding(item, values=values) for item in value)
    return False


def _purge_bound_work_products(
    work_root: Path,
    *,
    capture_session_id: str,
    intake_id: str,
    capture_digest: str,
) -> tuple[int, list[str], list[str]]:
    purged_files = 0
    inventory_digests: list[str] = []
    content_digests: list[str] = []
    plans = work_root / "reconstruction_control_plane" / "plans"
    deletion_roots: set[Path] = set()
    if plans.is_dir():
        for context_path in sorted(plans.glob("*/artifacts/context.json")):
            context = _read(context_path, code="reconstruction_context_invalid", status_code=422)
            if context.get("intake_id") == intake_id and context.get(
                "capture_digest"
            ) == capture_digest:
                deletion_roots.add(context_path.parents[1])
    task_session = (
        work_root
        / "task_candidate_control_plane"
        / "sessions"
        / capture_session_id
    )
    if task_session.is_dir():
        deletion_roots.add(task_session)
    runs = work_root / "task_evaluation_runs" / "runs"
    if runs.is_dir():
        for context_path in sorted(runs.glob("*/artifacts/run_context.json")):
            context = _read(context_path, code="task_evaluation_run_context_invalid", status_code=422)
            if context.get("capture_session_id") == capture_session_id and context.get(
                "intake_id"
            ) == intake_id:
                deletion_roots.add(context_path.parents[1])
    testbeds = work_root / "maintained_site_task_testbeds"
    if testbeds.is_dir():
        for path in sorted(testbeds.glob("*/*/*.json")):
            document = _read(path, code="maintained_testbed_artifact_invalid", status_code=422)
            if _contains_binding(
                document, values={intake_id, capture_digest, capture_session_id}
            ):
                deletion_roots.add(path.parent)
    for deletion_root in sorted(deletion_roots):
        count, digest, deleted_content_digests = _delete_tree(deletion_root)
        purged_files += count
        inventory_digests.append(digest)
        content_digests.extend(deleted_content_digests)
    return (
        purged_files,
        sorted(inventory_digests),
        sorted(set(content_digests)),
    )


def _apply_capture_lifecycle_action_locked(
    *,
    store_root: str | Path,
    work_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    capture_digest: str,
    envelope_digest: str,
    action: str,
    actor: Mapping[str, Any],
    idempotency_key: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    try:
        session = strict_identifier(capture_session_id, field="capture_session_id", max_length=192)
        intake = strict_identifier(intake_id, field="intake_id", max_length=192)
        key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
    except ValueError as exc:
        raise CaptureLifecycleError(str(exc)) from exc
    if action not in LIFECYCLE_ACTIONS:
        raise CaptureLifecycleError("lifecycle_action_unsupported")
    if not _is_digest(capture_digest) or not _is_digest(envelope_digest):
        raise CaptureLifecycleError("capture_lifecycle_digest_invalid")
    root = Path(store_root).expanduser().resolve()
    working = Path(work_root).expanduser().resolve()
    lifecycle_key = _lifecycle_key(session, intake)
    marker_path = root / "lifecycle_markers" / f"{lifecycle_key}.json"
    tombstone_path = root / "lifecycle_tombstones" / f"{lifecycle_key}.json"
    if tombstone_path.is_file():
        existing = _read(tombstone_path, code="capture_lifecycle_tombstone_invalid")
        if (
            existing.get("capture_digest") != capture_digest
            or existing.get("envelope_digest") != envelope_digest
            or existing.get("action") != action
        ):
            raise CaptureLifecycleError("capture_lifecycle_terminal_conflict", status_code=409)
        return {**existing, "already_exists": True}
    receipt_path = root / "transfer_receipts" / f"{lifecycle_key}.json"
    receipt = _read(receipt_path, code="capture_upload_receipt_not_found")
    if (
        receipt.get("capture_session_id") != session
        or receipt.get("intake_id") != intake
        or receipt.get("capture_digest") != capture_digest
        or receipt.get("envelope_digest") != envelope_digest
    ):
        raise CaptureLifecycleError("capture_lifecycle_source_mismatch", status_code=409)
    artifact_reference = _mapping(receipt.get("artifact_reference"))
    artifact_root = _safe_child(
        root,
        str(artifact_reference.get("uri") or ""),
        code="capture_artifact_reference_unsafe",
    )
    envelope = _read(
        artifact_root / "capture_intake_envelope.json",
        code="capture_intake_envelope_not_found",
    )
    governance = _mapping(envelope.get("governance"))
    revocation = _mapping(governance.get("revocation"))
    retention = _mapping(governance.get("retention"))
    if action in {"consent_revoked", "operator_deletion_request"} and revocation.get(
        "supported"
    ) is not True:
        raise CaptureLifecycleError("capture_revocation_not_supported")
    if retention.get("legal_hold") is True:
        raise CaptureLifecycleError("capture_legal_hold_prevents_deletion", status_code=409)
    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    retention_deadline: str | None = None
    if action == "retention_expired":
        received = _parse_time(
            receipt.get("capture_upload_received_at"),
            code="capture_retention_start_time_missing",
        )
        try:
            max_days = int(retention.get("max_days"))
        except (TypeError, ValueError) as exc:
            raise CaptureLifecycleError("capture_retention_max_days_invalid") from exc
        if max_days < 0:
            raise CaptureLifecycleError("capture_retention_max_days_invalid")
        deadline = received + timedelta(days=max_days)
        retention_deadline = deadline.isoformat()
        if effective_now < deadline:
            raise CaptureLifecycleError("capture_retention_not_expired", status_code=409)
    actor_value = dict(actor)
    if any(
        value not in (None, "", [], {})
        and (
            str(name).lower() in {"authorization", "credential", "credentials", "password", "secret", "token"}
            or str(name).lower().endswith(("_token", "_secret", "_password"))
        )
        for name, value in actor_value.items()
    ):
        raise CaptureLifecycleError("capture_lifecycle_actor_secret_forbidden")
    actor_identity = str(actor_value.get("identity") or "")
    marker = {
        "schema_version": "capture_lifecycle_marker.v1",
        "capture_session_id_digest": "sha256:" + hashlib.sha256(session.encode()).hexdigest(),
        "intake_id_digest": "sha256:" + hashlib.sha256(intake.encode()).hexdigest(),
        "capture_digest": capture_digest,
        "envelope_digest": envelope_digest,
        "action": action,
        "idempotency_key_digest": "sha256:" + hashlib.sha256(key.encode()).hexdigest(),
        "actor": {
            "role": str(actor_value.get("role") or "authenticated_service"),
            "identity_digest": "sha256:" + hashlib.sha256(actor_identity.encode()).hexdigest(),
        },
        "serve_allowed": False,
        "future_processing_allowed": False,
        "created_at": effective_now.isoformat(),
    }
    marker["marker_digest"] = canonical_digest(marker, digest_field="marker_digest")
    _write_once(marker_path, marker)

    manifest = _read(
        artifact_root / "capture_intake_object_manifest.json",
        code="capture_object_manifest_not_found",
    )
    objects = [dict(row) for row in manifest.get("objects", []) if isinstance(row, Mapping)]
    target_objects = [row for row in objects if row.get("sha256") == capture_digest]
    if len(objects) != 1 or len(target_objects) != 1:
        raise CaptureLifecycleError("capture_object_binding_ambiguous")
    object_path = _safe_child(
        root,
        str(target_objects[0].get("object_path") or ""),
        code="capture_object_reference_unsafe",
    )
    shared_references = _other_active_references(
        root, target_artifact_root=artifact_root, capture_digest=capture_digest
    )
    provider_obligations = _provider_obligations(
        working, intake_id=intake, capture_digest=capture_digest
    )
    if (
        not shared_references
        and object_path.is_file()
        and _sha256_file(object_path) != capture_digest
    ):
        raise CaptureLifecycleError("capture_object_digest_mismatch", status_code=409)
    work_product_count, work_product_inventories, work_product_digests = (
        _purge_bound_work_products(
            working,
            capture_session_id=session,
            intake_id=intake,
            capture_digest=capture_digest,
        )
    )
    artifact_count, artifact_inventory_digest, artifact_content_digests = _delete_tree(
        artifact_root
    )
    raw_object_deleted = False
    if not shared_references and object_path.is_file():
        object_path.unlink()
        raw_object_deleted = True
    receipt_digest = canonical_digest(receipt)
    receipt_path.unlink(missing_ok=True)
    tombstone = {
        "schema_version": "capture_lifecycle_tombstone.v1",
        "intake_id_digest": marker["intake_id_digest"],
        "capture_session_id_digest": marker["capture_session_id_digest"],
        "capture_digest": capture_digest,
        "envelope_digest": envelope_digest,
        "receipt_digest": receipt_digest,
        "qa_report_digest": _mapping(receipt.get("capture_qa_report")).get(
            "qa_report_digest"
        ),
        "object_manifest_digest": manifest.get("manifest_digest"),
        "action": action,
        "retention_deadline": retention_deadline,
        "completed_at": effective_now.isoformat(),
        "serve_allowed": False,
        "future_processing_allowed": False,
        "training_use_allowed": False,
        "raw_object_deleted": raw_object_deleted,
        "raw_object_preserved_for_shared_active_references": bool(shared_references),
        "shared_active_reference_digests": shared_references,
        "deleted_capture_artifact_file_count": artifact_count,
        "deleted_capture_artifact_inventory_digest": artifact_inventory_digest,
        "deleted_capture_artifact_content_digests": artifact_content_digests,
        "deleted_bound_work_product_file_count": work_product_count,
        "deleted_bound_work_product_inventory_digests": work_product_inventories,
        "deleted_bound_work_product_content_digests": work_product_digests,
        "provider_deletion_obligations": provider_obligations,
        "provider_deletion_status": (
            "required" if provider_obligations else "not_required"
        ),
        "external_revocation_actions": [
            {
                "action": "sync_webapp_revocation_verdict",
                "status": "required_not_executed",
            },
            {
                "action": "disable_signed_download_access",
                "status": "required_not_executed",
            },
        ],
        "local_payload_deletion_complete": True,
        "external_revocation_complete": False,
        "historical_digest_binding_preserved": True,
        "non_sensitive_tombstone": True,
        "proof_boundary": {
            "local_payload_deletion_is_provider_deletion": False,
            "provider_deletion_evidence_verified": False,
            "prior_decisions_rewritten": False,
            "physical_task_success_established": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
        "already_exists": False,
    }
    tombstone["tombstone_digest"] = canonical_digest(
        tombstone, digest_field="tombstone_digest"
    )
    return _write_once(tombstone_path, tombstone)


def apply_capture_lifecycle_action(
    *,
    store_root: str | Path,
    work_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    capture_digest: str,
    envelope_digest: str,
    action: str,
    actor: Mapping[str, Any],
    idempotency_key: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Serialize one destructive lifecycle action for an exact capture binding."""

    try:
        session = strict_identifier(capture_session_id, field="capture_session_id", max_length=192)
        intake = strict_identifier(intake_id, field="intake_id", max_length=192)
    except ValueError as exc:
        raise CaptureLifecycleError(str(exc)) from exc
    root = Path(store_root).expanduser().resolve()
    lock_path = root / "lifecycle_locks" / f"{_lifecycle_key(session, intake)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _apply_capture_lifecycle_action_locked(
            store_root=root,
            work_root=work_root,
            capture_session_id=session,
            intake_id=intake,
            capture_digest=capture_digest,
            envelope_digest=envelope_digest,
            action=action,
            actor=actor,
            idempotency_key=idempotency_key,
            now=now,
        )


def record_provider_deletion_evidence(
    *,
    store_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    obligation_digest: str,
    deletion_receipt_digest: str,
    provider_identity: str,
    deleted_at: str,
    verification_method: str,
    idempotency_key: str,
) -> dict[str, Any]:
    try:
        session = strict_identifier(capture_session_id, field="capture_session_id", max_length=192)
        intake = strict_identifier(intake_id, field="intake_id", max_length=192)
        key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
        provider = strict_identifier(
            provider_identity, field="provider_identity", max_length=192
        )
    except ValueError as exc:
        raise CaptureLifecycleError(str(exc)) from exc
    if not _is_digest(obligation_digest) or not _is_digest(deletion_receipt_digest):
        raise CaptureLifecycleError("provider_deletion_digest_invalid")
    if verification_method not in PROVIDER_DELETION_VERIFICATION_METHODS:
        raise CaptureLifecycleError("provider_deletion_verification_method_unsupported")
    root = Path(store_root).expanduser().resolve()
    lifecycle_key = _lifecycle_key(session, intake)
    tombstone = _read(
        root / "lifecycle_tombstones" / f"{lifecycle_key}.json",
        code="capture_lifecycle_tombstone_not_found",
    )
    obligations = {
        str(row.get("obligation_digest") or ""): dict(row)
        for row in tombstone.get("provider_deletion_obligations", [])
        if isinstance(row, Mapping)
    }
    if obligation_digest not in obligations:
        raise CaptureLifecycleError("provider_deletion_obligation_not_found")
    if obligations[obligation_digest].get("provider_identity") != provider:
        raise CaptureLifecycleError("provider_deletion_provider_mismatch", status_code=409)
    _parse_time(deleted_at, code="provider_deletion_timestamp_invalid")
    evidence = {
        "schema_version": "capture_provider_deletion_evidence.v1",
        "tombstone_digest": tombstone["tombstone_digest"],
        "obligation_digest": obligation_digest,
        "provider_identity": provider,
        "deletion_receipt_digest": deletion_receipt_digest,
        "deleted_at": deleted_at,
        "verification_method": verification_method,
        "idempotency_key_digest": "sha256:" + hashlib.sha256(key.encode()).hexdigest(),
        "proof_boundary": {
            "receipt_metadata_is_independent_provider_verification": False,
            "physical_task_success_established": False,
        },
    }
    evidence["provider_deletion_evidence_digest"] = canonical_digest(
        evidence, digest_field="provider_deletion_evidence_digest"
    )
    return _write_once(
        root
        / "provider_deletion_evidence"
        / lifecycle_key
        / f"{obligation_digest.removeprefix('sha256:')}.json",
        evidence,
    )


def record_external_revocation_evidence(
    *,
    store_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    action: str,
    target_system: str,
    receipt_digest: str,
    completed_at: str,
    verification_method: str,
    idempotency_key: str,
) -> dict[str, Any]:
    try:
        session = strict_identifier(capture_session_id, field="capture_session_id", max_length=192)
        intake = strict_identifier(intake_id, field="intake_id", max_length=192)
        target = strict_identifier(target_system, field="target_system", max_length=192)
        key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
    except ValueError as exc:
        raise CaptureLifecycleError(str(exc)) from exc
    if not _is_digest(receipt_digest):
        raise CaptureLifecycleError("external_revocation_receipt_digest_invalid")
    if verification_method not in EXTERNAL_REVOCATION_VERIFICATION_METHODS:
        raise CaptureLifecycleError("external_revocation_verification_method_unsupported")
    _parse_time(completed_at, code="external_revocation_timestamp_invalid")
    root = Path(store_root).expanduser().resolve()
    lifecycle_key = _lifecycle_key(session, intake)
    tombstone = _read(
        root / "lifecycle_tombstones" / f"{lifecycle_key}.json",
        code="capture_lifecycle_tombstone_not_found",
    )
    required = {
        str(row.get("action") or "")
        for row in tombstone.get("external_revocation_actions", [])
        if isinstance(row, Mapping)
    }
    if action not in required:
        raise CaptureLifecycleError("external_revocation_action_not_required")
    evidence = {
        "schema_version": "capture_external_revocation_evidence.v1",
        "tombstone_digest": tombstone["tombstone_digest"],
        "action": action,
        "target_system": target,
        "receipt_digest": receipt_digest,
        "completed_at": completed_at,
        "verification_method": verification_method,
        "idempotency_key_digest": "sha256:" + hashlib.sha256(key.encode()).hexdigest(),
        "proof_boundary": {
            "receipt_is_scientific_or_physical_success": False,
            "prior_decisions_rewritten": False,
        },
    }
    evidence["external_revocation_evidence_digest"] = canonical_digest(
        evidence, digest_field="external_revocation_evidence_digest"
    )
    return _write_once(
        root
        / "external_revocation_evidence"
        / lifecycle_key
        / f"{action}.json",
        evidence,
    )


def inspect_capture_lifecycle(
    *, store_root: str | Path, capture_session_id: str, intake_id: str
) -> dict[str, Any]:
    try:
        session = strict_identifier(capture_session_id, field="capture_session_id", max_length=192)
        intake = strict_identifier(intake_id, field="intake_id", max_length=192)
    except ValueError as exc:
        raise CaptureLifecycleError(str(exc)) from exc
    root = Path(store_root).expanduser().resolve()
    lifecycle_key = _lifecycle_key(session, intake)
    tombstone_path = root / "lifecycle_tombstones" / f"{lifecycle_key}.json"
    marker_path = root / "lifecycle_markers" / f"{lifecycle_key}.json"
    tombstone = _read(tombstone_path, code="capture_lifecycle_not_found") if tombstone_path.is_file() else None
    marker = _read(marker_path, code="capture_lifecycle_marker_invalid") if marker_path.is_file() else None
    if tombstone is None and marker is None:
        raise CaptureLifecycleError("capture_lifecycle_not_found", status_code=404)
    evidence: list[dict[str, Any]] = []
    evidence_root = root / "provider_deletion_evidence" / lifecycle_key
    if evidence_root.is_dir():
        evidence = [
            _read(path, code="provider_deletion_evidence_invalid")
            for path in sorted(evidence_root.glob("*.json"))
        ]
    obligation_count = len(tombstone.get("provider_deletion_obligations", [])) if tombstone else 0
    external_evidence: list[dict[str, Any]] = []
    external_root = root / "external_revocation_evidence" / lifecycle_key
    if external_root.is_dir():
        external_evidence = [
            _read(path, code="external_revocation_evidence_invalid")
            for path in sorted(external_root.glob("*.json"))
        ]
    required_external_actions = (
        {
            str(row.get("action") or "")
            for row in tombstone.get("external_revocation_actions", [])
            if isinstance(row, Mapping)
        }
        if tombstone
        else set()
    )
    recorded_external_actions = {
        str(row.get("action") or "") for row in external_evidence
    }
    provider_complete = bool(tombstone) and len(evidence) == obligation_count
    external_complete = bool(tombstone) and recorded_external_actions == required_external_actions
    return {
        "schema_version": "capture_lifecycle_inspection.v1",
        "state": "tombstoned" if tombstone else "deletion_in_progress_or_retry_required",
        "marker": marker,
        "tombstone": tombstone,
        "provider_deletion_evidence": evidence,
        "provider_deletion_complete": provider_complete,
        "external_revocation_evidence": external_evidence,
        "local_payload_deletion_complete": bool(tombstone),
        "external_revocation_complete": external_complete,
        "lifecycle_complete": bool(tombstone) and provider_complete and external_complete,
        "serve_allowed": False,
        "future_processing_allowed": False,
    }


__all__ = [
    "CaptureLifecycleError",
    "apply_capture_lifecycle_action",
    "inspect_capture_lifecycle",
    "record_external_revocation_evidence",
    "record_provider_deletion_evidence",
]
