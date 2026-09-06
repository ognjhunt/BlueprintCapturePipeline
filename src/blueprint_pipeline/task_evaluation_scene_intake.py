"""Persistent owner intent, distinct from immutable release-bound execution attempts.

Only an authenticated intake service may issue these records. A record copied
directly into a database is not an execution capability. This module never
allocates resources and does not substitute user declarations for source proof.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as canonical_digest
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)

REQUEST_SCHEMA = "task_evaluation_scene_intake_request.v1"
INTENT_SCHEMA = "task_evaluation_scene_intent.v1"
ATTEMPT_SCHEMA = "task_evaluation_scene_attempt.v1"
ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"
CLIENTS_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
#: The two frozen policy candidates this ADP-009D run actually supports end to
#: end (scene setup CANDIDATE_IDS, policy-canary handoff, dispatch). Intake
#: rejects any other pair up front instead of accepting it and failing late,
#: after construction spend, at the handoff (A10). Do not broaden this here.
SUPPORTED_POLICY_CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")


class SceneIntakeError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneIntakeError("scene_intake_" + code)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_digest(result, digest_field=field)
    return result


def validate_request(value: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    _require(set(value) == {"schema_version", "submission_id", "owner", "source", "task",
                            "execution", "consent"}, "request_fields_invalid")
    _require(value.get("schema_version") == REQUEST_SCHEMA, "schema_invalid")
    _require(_identifier(value.get("submission_id")), "submission_id_invalid")
    owner = value.get("owner")
    _require(isinstance(owner, Mapping) and set(owner) == {"user_id", "organization_id"}
             and all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._@-]{0,127}", v)
                     for v in owner.values()), "owner_invalid")
    source = value.get("source")
    _require(isinstance(source, Mapping) and set(source) - {"collision_mesh"} == {"kind", "binding_id", "content_digest"},
             "source_invalid")
    _require(source["kind"] in {"capture_bundle", "mesh", "gaussian_splat", "public_scene"}
             and _identifier(source["binding_id"])
             and isinstance(source["content_digest"], str)
             and _DIGEST.fullmatch(source["content_digest"]) is not None, "source_invalid")
    if "collision_mesh" in source:
        companion = source["collision_mesh"]
        _require(source["kind"] == "gaussian_splat" and isinstance(companion, Mapping)
                 and set(companion) == {"binding_id", "content_digest", "rights_reference", "frame_relation"}
                 and _identifier(companion.get("binding_id")) and companion["binding_id"] != source["binding_id"]
                 and isinstance(companion.get("content_digest"), str)
                 and _DIGEST.fullmatch(companion["content_digest"]) is not None
                 and isinstance(companion.get("rights_reference"), str)
                 and _DIGEST.fullmatch(companion["rights_reference"]) is not None
                 and companion.get("frame_relation") == "owner_declared_common_frame", "collision_mesh_binding_invalid")
    task = value.get("task")
    _require(isinstance(task, Mapping) and task.get("strategy") == "pick_and_place"
             and _identifier(task.get("task_id")), "task_invalid")
    for key in ("subject", "support", "destination", "success"):
        _require(isinstance(task.get(key), Mapping) and bool(task[key]), "task_" + key + "_missing")
    execution = value.get("execution")
    _require(isinstance(execution, Mapping) and set(execution) == {
        "max_total_spend_usd", "max_paid_attempts", "max_retries", "expires_at_epoch",
        "allowed_providers", "policy_candidates", "claim_scope"}, "execution_invalid")
    _require(_number(execution["max_total_spend_usd"])
             and 0 < execution["max_total_spend_usd"] <= 1000, "spend_invalid")
    _require(type(execution["max_paid_attempts"]) is int and 1 <= execution["max_paid_attempts"] <= 32
             and type(execution["max_retries"]) is int and 0 <= execution["max_retries"] <= 3,
             "attempt_bounds_invalid")
    _require(_number(execution["expires_at_epoch"])
             and now < execution["expires_at_epoch"] <= now + 7 * 86400, "authority_expiry_invalid")
    providers = execution["allowed_providers"]
    _require(isinstance(providers, list) and bool(providers)
             and all(isinstance(p, str) and p in {"vast", "runpod", "openai"} for p in providers)
             and len(providers) == len(set(providers)), "providers_invalid")
    policies = execution["policy_candidates"]
    _require(isinstance(policies, list) and len(policies) == 2, "two_policies_required")
    for policy in policies:
        _require(isinstance(policy, Mapping) and set(policy) == {"id", "artifact_digest"}
                 and _identifier(policy["id"]) and isinstance(policy["artifact_digest"], str)
                 and _DIGEST.fullmatch(policy["artifact_digest"]) is not None, "policy_identity_invalid")
    _require(policies[0]["id"] != policies[1]["id"], "two_distinct_policies_required")
    _require([policy["id"] for policy in policies] == list(SUPPORTED_POLICY_CANDIDATE_IDS),
             "policy_candidates_unsupported")
    _require(execution["claim_scope"] == "development_only", "claim_scope_invalid")
    consent = value.get("consent")
    _require(isinstance(consent, Mapping) and set(consent) == {
        "accepted_by", "accepted_at_epoch", "rights_reference", "provider_terms_reference",
        "private_processing_authorized", "provider_training_authorized", "task_confirmed",
        "spend_authorized"}, "consent_invalid")
    _require(consent["accepted_by"] == owner["user_id"]
             and _number(consent["accepted_at_epoch"])
             and now - 86400 <= consent["accepted_at_epoch"] <= now, "consent_actor_or_time_invalid")
    _require(all(isinstance(consent[k], str) and 1 <= len(consent[k]) <= 1000
                 for k in ("rights_reference", "provider_terms_reference")), "consent_references_missing")
    _require(consent["private_processing_authorized"] is True
             and consent["provider_training_authorized"] is False
             and consent["task_confirmed"] is True and consent["spend_authorized"] is True,
             "consent_missing")
    # Detach mutable caller state and reject non-JSON/NaN task values.
    try:
        detached = json.loads(json.dumps(value, allow_nan=False))
        canonical_digest(detached)
        return detached
    except (TypeError, ValueError) as exc:
        raise SceneIntakeError("scene_intake_json_invalid") from exc


def _root(root: Path) -> Path:
    _require(not root.is_symlink(), "root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    return root.resolve(strict=True)


@contextmanager
def _lock(root: Path):
    descriptor = os.open(root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read(path: Path, field: str) -> dict[str, Any]:
    _require(not path.is_symlink(), "record_unsafe")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SceneIntakeError("scene_intake_record_unreadable") from exc
    _require(isinstance(value, dict) and value.get(field) == canonical_digest(value, digest_field=field),
             "record_digest_invalid")
    return value


def stage_scene_intent(*, value: Mapping[str, Any], queue_root: str | Path,
                       authenticated_client: str, trusted_clients: set[str],
                       now: float | None = None) -> dict[str, Any]:
    _require(bool(authenticated_client) and authenticated_client in trusted_clients,
             "issuer_not_authorized")
    moment = datetime.now(timezone.utc).timestamp() if now is None else now
    request = validate_request(value, now=moment)
    root = _root(Path(queue_root))
    # Tenant plus user plus idempotency key: another owner cannot alias this intent.
    identity = canonical_digest({"owner": request["owner"], "submission_id": request["submission_id"]})
    intent_id = "scene-" + identity.removeprefix("sha256:")
    with _lock(root):
        directory = root / intent_id
        _require(not directory.is_symlink(), "record_unsafe")
        directory.mkdir(mode=0o750, exist_ok=True)
        path = directory / "intent.json"
        if path.exists():
            intent = _read(path, "intent_digest")
            _require(intent["request"] == request, "idempotency_conflict")
        else:
            intent = _seal({"schema_version": INTENT_SCHEMA, "intent_id": intent_id,
                "request": request, "authenticated_issuer": authenticated_client,
                "accepted_at_epoch": moment, "source_content_digest": request["source"]["content_digest"],
                "task_content_digest": canonical_digest(request["task"]),
                "provider_mutation_performed": False}, "intent_digest")
            write_exclusive(path, intent)
    return _seal({"schema_version": "task_evaluation_scene_intake_receipt.v1",
                  "status": "accepted", "intent_id": intent_id, "intent_digest": intent["intent_digest"],
                  "request_digest": canonical_digest(request),
                  "provider_mutation_performed_inside_http_request": False}, "receipt_digest")


def reserve_scene_attempt(*, queue_root: str | Path, intent_id: str, attempt_id: str,
                          source_commit: str, runtime_digest: str, input_digest: str,
                          provider: str, maximum_spend_usd: float,
                          now: float | None = None,
                          recovery_from_attempt_id: str | None = None,
                          recovery_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Debit maximum exposure before dispatch; retries never reset the owner's cap."""
    _require(_identifier(intent_id) and _identifier(attempt_id), "attempt_id_invalid")
    _require(_COMMIT.fullmatch(source_commit) is not None and _DIGEST.fullmatch(runtime_digest) is not None
             and _DIGEST.fullmatch(input_digest) is not None, "attempt_identity_invalid")
    _require(_number(maximum_spend_usd) and maximum_spend_usd > 0, "attempt_spend_invalid")
    root = _root(Path(queue_root))
    moment = datetime.now(timezone.utc).timestamp() if now is None else now
    with _lock(root):
        directory = root / intent_id
        _require(not directory.is_symlink(), "record_unsafe")
        intent = _read(directory / "intent.json", "intent_digest")
        execution = intent["request"]["execution"]
        _require(not (directory / "revoked.json").exists(), "authority_revoked")
        _require(moment < execution["expires_at_epoch"], "authority_expired")
        _require(provider in execution["allowed_providers"], "provider_not_authorized")
        attempts = directory / "attempts"
        _require(not attempts.is_symlink(), "record_unsafe")
        attempts.mkdir(mode=0o750, exist_ok=True)
        body = {"schema_version": ATTEMPT_SCHEMA, "intent_id": intent_id,
                "intent_digest": intent["intent_digest"], "attempt_id": attempt_id,
                "source_commit": source_commit, "runtime_digest": runtime_digest,
                "input_digest": input_digest, "provider": provider,
                "maximum_spend_usd": maximum_spend_usd, "status": "reserved"}
        path = attempts / (attempt_id + ".json")
        _require((recovery_from_attempt_id is None) == (recovery_evidence is None),
                 "recovery_lineage_required")
        if recovery_from_attempt_id is not None:
            _require(_identifier(recovery_from_attempt_id) and recovery_from_attempt_id != attempt_id,
                     "recovery_new_attempt_required")
            prior = _read(attempts / (recovery_from_attempt_id + ".json"), "attempt_digest")
            _require(prior["intent_digest"] == intent["intent_digest"] and prior["provider"] == provider,
                     "recovery_prior_attempt_mismatch")
            # An idempotent read must not require fresh inventory after the
            # reservation has already been durably debited.
            if path.exists():
                existing = _read(path, "attempt_digest")
                _require(all(existing.get(k) == v for k, v in body.items())
                         and existing.get("recovery", {}).get("prior_attempt_id") == recovery_from_attempt_id
                         and existing["recovery"].get("evidence") == recovery_evidence,
                         "attempt_immutable_conflict")
                return existing
            rows = [_read(p, "attempt_digest") for p in attempts.glob("*.json")]
            _require(sum("recovery" in row for row in rows) < execution["max_retries"],
                     "retry_cap_exhausted")
            _require(not any(row.get("recovery", {}).get("prior_attempt_id") == recovery_from_attempt_id
                             for row in rows), "recovery_successor_already_reserved")
            from .task_evaluation_scene_recovery import validate_recovery_evidence
            body["recovery"] = validate_recovery_evidence(recovery_evidence,
                prior_attempt=prior, provider=provider, now=moment)
        if path.exists():
            existing = _read(path, "attempt_digest")
            _require(all(existing.get(k) == v for k, v in body.items())
                     and ("recovery" in existing) == ("recovery" in body), "attempt_immutable_conflict")
            return existing
        rows = [_read(p, "attempt_digest") for p in attempts.glob("*.json")]
        _require(len(rows) < execution["max_paid_attempts"], "attempt_cap_exhausted")
        exposure = sum((Decimal(str(row["maximum_spend_usd"])) for row in rows), Decimal(0))
        _require(exposure + Decimal(str(maximum_spend_usd))
                 <= Decimal(str(execution["max_total_spend_usd"])), "spend_cap_exhausted")
        result = _seal({**body, "reserved_at_epoch": moment}, "attempt_digest")
        write_exclusive(path, result)
        return result


def scene_intent_status(*, queue_root: str | Path, intent_id: str,
                        now: float | None = None) -> dict[str, Any]:
    """Public projection from retained records; a status read never changes state."""
    _require(_identifier(intent_id), "intent_id_invalid")
    root = Path(queue_root)
    directory = root / intent_id
    _require(root.is_dir() and not root.is_symlink() and not directory.is_symlink(), "root_unsafe")
    intent = _read(directory / "intent.json", "intent_digest")
    moment = datetime.now(timezone.utc).timestamp() if now is None else now
    status, phase, blockers, result_reference = "accepted", None, [], None
    progress_path = directory / "progression.json"
    if progress_path.exists():
        progress = _read(progress_path, "progression_digest")
        _require(progress.get("intent_digest") == intent["intent_digest"], "progression_binding_invalid")
        permitted = {"accepted", "preparing", "awaiting_source", "awaiting_execution", "running",
                     "completed", "needs_input", "blocked"}
        _require(progress.get("status") in permitted, "progression_status_invalid")
        status = progress["status"]
        phase = progress.get("phase")
        _require(phase is None or _identifier(phase), "progression_phase_invalid")
        raw_blockers = progress.get("blockers", [])
        _require(isinstance(raw_blockers, list), "progression_blockers_invalid")
        # Internal diagnostics stay in worker receipts; never return paths,
        # secrets, or arbitrary exception text as a customer-facing blocker.
        blockers = sorted({str(b).split(":", 1)[0] for b in raw_blockers
                           if re.fullmatch(r"[a-z][a-z0-9_:-]{0,255}", str(b))})
        result_reference = progress.get("result_reference")
        if result_reference is not None:
            _require(isinstance(result_reference, Mapping)
                     and set(result_reference) == {"uri", "digest", "size_bytes"}
                     and isinstance(result_reference["digest"], str)
                     and _DIGEST.fullmatch(result_reference["digest"]) is not None
                     and type(result_reference["size_bytes"]) is int and result_reference["size_bytes"] > 0
                     and isinstance(result_reference["uri"], str)
                     and result_reference["uri"].startswith(("https://", "s3://", "b2://", "gs://", "r2://"))
                     and "?" not in result_reference["uri"], "result_reference_invalid")
        _require(status != "completed" or result_reference is not None, "completed_result_missing")
    attempts_path = directory / "attempts"
    _require(not attempts_path.is_symlink(), "record_unsafe")
    attempts = []
    for path in sorted(attempts_path.glob("*.json")):
        row = _read(path, "attempt_digest")
        _require(row.get("intent_digest") == intent["intent_digest"], "attempt_binding_invalid")
        attempts.append({key: row[key] for key in (
            "attempt_id", "source_commit", "runtime_digest", "input_digest", "provider",
            "maximum_spend_usd", "status")})
    if status != "completed":
        if (directory / "revoked.json").exists():
            status, blockers = "revoked", ["scene_intake_authority_revoked"]
        elif moment >= intent["request"]["execution"]["expires_at_epoch"]:
            status, blockers = "expired", ["scene_intake_authority_expired"]
    return _seal({"schema_version": "task_evaluation_scene_intent_status.v1", "intent_id": intent_id,
        "intent_digest": intent["intent_digest"], "request_digest": canonical_digest(intent["request"]),
        "owner": intent["request"]["owner"], "status": status, "phase": phase, "blockers": blockers,
        "attempts": attempts, "result_reference": result_reference,
        "provider_mutation_performed_by_status_read": False}, "status_digest")


def revoke_scene_intent(*, queue_root: str | Path, intent_id: str, intent_digest: str,
                        owner: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    """Revoke future admissions without deleting evidence or pretending to stop a GPU."""
    _require(_identifier(intent_id), "intent_id_invalid")
    root = Path(queue_root)
    directory = root / intent_id
    _require(root.is_dir() and not root.is_symlink() and not directory.is_symlink(), "root_unsafe")
    with _lock(root):
        intent = _read(directory / "intent.json", "intent_digest")
        _require(intent["intent_digest"] == intent_digest and intent["request"]["owner"] == owner,
                 "owner_or_intent_mismatch")
        path = directory / "revoked.json"
        if path.exists():
            return _read(path, "receipt_digest")
        receipt = _seal({"schema_version": "task_evaluation_scene_intent_revocation.v1",
            "intent_id": intent_id, "intent_digest": intent_digest, "owner": dict(owner),
            "status": "revoked", "revoked_at_epoch": datetime.now(timezone.utc).timestamp() if now is None else now,
            "scope": "future_execution", "provider_mutation_performed": False}, "receipt_digest")
        write_exclusive(path, receipt)
        return receipt
