"""Durable control plane for inferred task review.

Pipeline owns discovery and approval truth. WebApp may relay a customer command,
but it cannot approve a task or compile a Decision/Evidence Request. This store
keeps immutable discovery and decision artifacts, uses an inter-process lock,
and supports exact idempotent replay across service restarts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_json
from .task_candidate_discovery import (
    TaskCandidateContractError,
    record_task_candidate_decision,
    validate_task_candidate_discovery,
)
from .webapp_sync import _pipeline_sync_headers


CONTROL_PLANE_SCHEMA_VERSION = "task_candidate_control_plane_state.v1"
SUBMISSION_SCHEMA_VERSION = "task_candidate_decision_submission.v1"
RESULT_SCHEMA_VERSION = "task_candidate_decision_processing_result.v1"
DISCOVERY_PUBLICATION_SCHEMA_VERSION = "task_candidate_discovery_publication.v1"
DISCOVERY_WEBAPP_URL_ENV = "PIPELINE_TASK_DISCOVERY_WEBAPP_URL"


class TaskCandidateControlPlaneError(ValueError):
    """A stable fail-closed error suitable for an API response."""

    def __init__(self, code: str, *, status_code: int = 422):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _identifier(value: Any, *, field: str) -> str:
    try:
        return strict_identifier(value, field=field, max_length=192)
    except ValueError as exc:
        raise TaskCandidateControlPlaneError(f"{field}:invalid") from exc


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json_any(path)
    if not isinstance(value, Mapping):
        raise TaskCandidateControlPlaneError("stored_artifact:not_object", status_code=500)
    return dict(value)


def _session_dir(state_root: Path, capture_session_id: str) -> Path:
    return state_root / "sessions" / capture_session_id


def _fingerprint(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def publish_task_candidate_discovery(
    *,
    state_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an immutable discovery and advance the session projection."""

    session_id = _identifier(capture_session_id, field="capture_session_id")
    expected_intake_id = _identifier(intake_id, field="intake_id")
    try:
        verified = validate_task_candidate_discovery(discovery)
    except TaskCandidateContractError as exc:
        raise TaskCandidateControlPlaneError(f"discovery_invalid:{exc}") from exc
    if _text(_mapping(verified.get("source_capture")).get("intake_id")) != expected_intake_id:
        raise TaskCandidateControlPlaneError("discovery_intake_mismatch", status_code=409)

    root = Path(state_root).expanduser().resolve()
    session_dir = _session_dir(root, session_id)
    ensure_dir(session_dir / "discoveries")
    lock_path = session_dir / ".lock"
    digest = _text(verified.get("discovery_digest"))
    digest_name = digest.removeprefix("sha256:")
    discovery_path = session_dir / "discoveries" / f"{digest_name}.json"
    state_path = session_dir / "state.json"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _read_mapping(state_path)
        existing = _read_mapping(discovery_path)
        if existing and existing != verified:
            raise TaskCandidateControlPlaneError(
                "discovery_digest_content_conflict", status_code=409
            )
        if not existing:
            write_json(discovery_path, verified)
        history = [
            item
            for item in current.get("discovery_history", [])
            if isinstance(item, str) and item
        ]
        if digest not in history:
            history.append(digest)
        next_state = {
            "schema_version": CONTROL_PLANE_SCHEMA_VERSION,
            "capture_session_id": session_id,
            "intake_id": expected_intake_id,
            "current_discovery_digest": digest,
            "current_discovery_path": str(discovery_path),
            "discovery_history": history,
            "latest_decision_id": current.get("latest_decision_id"),
            "latest_decision_status": current.get("latest_decision_status"),
            "latest_decision_discovery_digest": current.get(
                "latest_decision_discovery_digest"
            ),
            "updated_at_iso": utc_now_iso(),
            "proof_boundary": {
                "candidate_is_customer_intent": False,
                "decision_evidence_request_compiled": False,
                "task_success_established": False,
            },
        }
        write_json(state_path, next_state)
    return {
        "schema_version": "task_candidate_discovery_publication_result.v1",
        "status": "published",
        "already_exists": bool(existing),
        "capture_session_id": session_id,
        "intake_id": expected_intake_id,
        "discovery_digest": digest,
        "discovery": verified,
        "proof_boundary": next_state["proof_boundary"],
    }


def build_task_discovery_webapp_publication(
    *,
    capture_session_id: str,
    intake_id: str,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    session_id = _identifier(capture_session_id, field="capture_session_id")
    expected_intake_id = _identifier(intake_id, field="intake_id")
    try:
        verified = validate_task_candidate_discovery(discovery)
    except TaskCandidateContractError as exc:
        raise TaskCandidateControlPlaneError(f"discovery_invalid:{exc}") from exc
    if _text(_mapping(verified.get("source_capture")).get("intake_id")) != expected_intake_id:
        raise TaskCandidateControlPlaneError("discovery_intake_mismatch", status_code=409)
    return {
        "schema_version": DISCOVERY_PUBLICATION_SCHEMA_VERSION,
        "capture_session_id": session_id,
        "intake_id": expected_intake_id,
        "discovery_digest": verified["discovery_digest"],
        "pipeline_task_discovery": verified,
        "proof_boundary": {
            "candidate_is_customer_intent": False,
            "decision_evidence_request_compiled": False,
            "task_success_established": False,
        },
    }


def sync_task_candidate_discovery_to_webapp(
    *,
    capture_session_id: str,
    intake_id: str,
    discovery: Mapping[str, Any],
    endpoint_url: str | None = None,
    token: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Publish a safe discovery projection through WebApp's signed route."""

    payload = build_task_discovery_webapp_publication(
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        discovery=discovery,
    )
    resolved_url = _text(endpoint_url) or _text(os.getenv(DISCOVERY_WEBAPP_URL_ENV))
    resolved_token = _text(token) or _text(os.getenv("PIPELINE_SYNC_TOKEN"))
    if not resolved_url or not resolved_token:
        return {
            "schema_version": "task_candidate_discovery_webapp_sync_result.v1",
            "status": "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
            "capture_session_id": payload["capture_session_id"],
            "discovery_digest": payload["discovery_digest"],
            "proof_boundary": payload["proof_boundary"],
        }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    attempts = max(1, min(int(max_attempts), 10))
    last_reason = "sync_unknown_failure"
    for attempt in range(1, attempts + 1):
        outbound = urllib_request.Request(
            resolved_url,
            data=body,
            headers=_pipeline_sync_headers(resolved_token, body),
            method="POST",
        )
        try:
            with urllib_request.urlopen(outbound, timeout=max(0.1, timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_reason = f"http_error:{exc.code}"
        except urllib_error.URLError as exc:
            last_reason = f"url_error:{exc.reason}"
        except (TimeoutError, ValueError) as exc:
            last_reason = exc.__class__.__name__.lower()
        else:
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                last_reason = "invalid_json"
            else:
                response_value = _mapping(parsed)
                response_matches = (
                    response_value.get("schema_version")
                    == "capture_task_discovery_publication_receipt.v1"
                    and response_value.get("status") == "published"
                    and isinstance(response_value.get("already_exists"), bool)
                    and response_value.get("capture_session_id")
                    == payload["capture_session_id"]
                    and response_value.get("intake_id") == payload["intake_id"]
                    and response_value.get("discovery_digest")
                    == payload["discovery_digest"]
                    and response_value.get("proof_boundary")
                    == payload["proof_boundary"]
                )
                if response_matches:
                    return {
                        "schema_version": "task_candidate_discovery_webapp_sync_result.v1",
                        "status": "succeeded",
                        "attempts": attempt,
                        "capture_session_id": payload["capture_session_id"],
                        "discovery_digest": payload["discovery_digest"],
                        "response": response_value,
                        "proof_boundary": payload["proof_boundary"],
                    }
                last_reason = "response_binding_mismatch"
        if attempt < attempts and retry_delay_seconds > 0:
            time.sleep(min(float(retry_delay_seconds), 5.0))
    return {
        "schema_version": "task_candidate_discovery_webapp_sync_result.v1",
        "status": "failed",
        "reason": last_reason,
        "attempts": attempts,
        "capture_session_id": payload["capture_session_id"],
        "discovery_digest": payload["discovery_digest"],
        "proof_boundary": payload["proof_boundary"],
    }


def publish_and_sync_task_candidate_discovery(
    *,
    state_root: str | Path,
    capture_session_id: str,
    intake_id: str,
    discovery: Mapping[str, Any],
    endpoint_url: str | None = None,
    token: str | None = None,
    sync_required: bool = False,
) -> dict[str, Any]:
    """Durably publish in Pipeline, then project to WebApp without conflating proof."""

    publication = publish_task_candidate_discovery(
        state_root=state_root,
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        discovery=discovery,
    )
    sync = sync_task_candidate_discovery_to_webapp(
        capture_session_id=capture_session_id,
        intake_id=intake_id,
        discovery=discovery,
        endpoint_url=endpoint_url,
        token=token,
    )
    sync_path = (
        _session_dir(Path(state_root).expanduser().resolve(), publication["capture_session_id"])
        / "webapp_sync"
        / f"{publication['discovery_digest'].removeprefix('sha256:')}.json"
    )
    write_json(sync_path, sync)
    if sync_required and sync.get("status") != "succeeded":
        raise TaskCandidateControlPlaneError(
            f"task_discovery_webapp_sync_required:{sync.get('reason') or sync.get('status')}",
            status_code=502,
        )
    return {"publication": publication, "webapp_sync": sync}


def process_task_candidate_decision_submission(
    *, state_root: str | Path, submission: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and append one WebApp-relayed customer/operator decision."""

    value = _mapping(submission)
    if value.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        raise TaskCandidateControlPlaneError(
            f"schema_version:must_be:{SUBMISSION_SCHEMA_VERSION}"
        )
    session_id = _identifier(value.get("capture_session_id"), field="capture_session_id")
    intake_id = _identifier(value.get("intake_id"), field="intake_id")
    command = _mapping(value.get("command"))
    if command.get("schema_version") != "task_candidate_decision_command_record.v1":
        raise TaskCandidateControlPlaneError(
            "command.schema_version:must_be:task_candidate_decision_command_record.v1"
        )
    command_request_id = _identifier(
        command.get("command_request_id"), field="command_request_id"
    )
    if _text(command.get("capture_session_id")) != session_id:
        raise TaskCandidateControlPlaneError("command_capture_session_mismatch", status_code=409)
    if _text(command.get("intake_id")) != intake_id:
        raise TaskCandidateControlPlaneError("command_intake_mismatch", status_code=409)
    requester_user_id = _identifier(
        command.get("requester_user_id"), field="requester_user_id"
    )
    actor = _mapping(command.get("actor"))
    if _text(actor.get("identity")) != f"firebase:{requester_user_id}":
        raise TaskCandidateControlPlaneError("command_actor_requester_mismatch", status_code=409)
    fingerprint = _fingerprint(value)

    root = Path(state_root).expanduser().resolve()
    session_dir = _session_dir(root, session_id)
    state_path = session_dir / "state.json"
    lock_path = session_dir / ".lock"
    decision_dir = session_dir / "decisions" / command_request_id
    result_path = decision_dir / "result.json"
    ensure_dir(decision_dir)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _read_mapping(result_path)
        if existing:
            if existing.get("submission_fingerprint_sha256") != fingerprint:
                raise TaskCandidateControlPlaneError(
                    "command_request_id_idempotency_conflict", status_code=409
                )
            return {**existing, "already_exists": True}

        state = _read_mapping(state_path)
        if not state:
            raise TaskCandidateControlPlaneError("task_discovery_not_published", status_code=404)
        if state.get("intake_id") != intake_id:
            raise TaskCandidateControlPlaneError("session_intake_mismatch", status_code=409)
        discovery_digest = _text(command.get("discovery_digest"))
        if state.get("current_discovery_digest") != discovery_digest:
            raise TaskCandidateControlPlaneError("task_discovery_stale", status_code=409)
        discovery_path = Path(_text(state.get("current_discovery_path")))
        discovery = _read_mapping(discovery_path)
        candidates = [
            item
            for item in discovery.get("task_candidates", [])
            if isinstance(item, Mapping)
            and item.get("task_candidate_id") == command.get("task_candidate_id")
        ]
        if len(candidates) != 1 or candidates[0].get("candidate_digest") != command.get(
            "candidate_digest"
        ):
            raise TaskCandidateControlPlaneError("task_candidate_stale", status_code=409)
        if (
            state.get("latest_decision_status") == "approved"
            and state.get("latest_decision_discovery_digest") == discovery_digest
        ):
            raise TaskCandidateControlPlaneError(
                "task_discovery_already_approved", status_code=409
            )
        try:
            decision, approved = record_task_candidate_decision(
                discovery,
                task_candidate_id=_text(command.get("task_candidate_id")),
                action=_text(command.get("action")),
                actor=actor,
                idempotency_key=_text(command.get("idempotency_key")),
                rationale=_text(command.get("rationale")),
                edited_task=(
                    _mapping(command.get("edited_task"))
                    if isinstance(command.get("edited_task"), Mapping)
                    else None
                ),
            )
        except TaskCandidateContractError as exc:
            raise TaskCandidateControlPlaneError(f"decision_invalid:{exc}") from exc
        action = _text(command.get("action"))
        approval_status = {
            "approve": "approved",
            "edit_and_approve": "approved",
            "reject": "rejected",
            "request_more_capture": "recapture_requested",
        }[action]
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "processed",
            "accepted": True,
            "already_exists": False,
            "capture_session_id": session_id,
            "intake_id": intake_id,
            "command_request_id": command_request_id,
            "submission_fingerprint_sha256": fingerprint,
            "pipeline_approval_status": approval_status,
            "pipeline_task_decision": decision,
            "approved_task_definition": approved,
            "decision_evidence_request": None,
            "processed_at_iso": utc_now_iso(),
            "proof_boundary": {
                "webapp_command_is_pipeline_approval": False,
                "pipeline_decision_recorded": True,
                "approved_task_exists": approved is not None,
                "decision_evidence_request_compiled": False,
                "testbed_required_before_request_compilation": True,
                "task_success_established": False,
                "physical_success_established": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
        }
        write_json(decision_dir / "submission.json", value)
        write_json(decision_dir / "decision.json", decision)
        if approved is not None:
            write_json(decision_dir / "approved_task.json", approved)
        write_json(result_path, result)
        write_json(
            state_path,
            {
                **state,
                "latest_decision_id": decision["decision_id"],
                "latest_decision_status": approval_status,
                "latest_decision_discovery_digest": discovery_digest,
                "latest_command_request_id": command_request_id,
                "updated_at_iso": utc_now_iso(),
                "proof_boundary": result["proof_boundary"],
            },
        )
    return result


def load_task_candidate_control_plane_state(
    *, state_root: str | Path, capture_session_id: str
) -> dict[str, Any]:
    session_id = _identifier(capture_session_id, field="capture_session_id")
    return _read_mapping(_session_dir(Path(state_root).expanduser().resolve(), session_id) / "state.json")
