"""Pure intake contract for an exact terminal-provider release request.

The HTTP intake service may validate and immutably queue this request, but it
must not import provider clients.  The separately-run release worker owns the
sole provider mutation after canonical allocator admission.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .task_evaluation_launch_dispatcher import canonical_digest

REQUEST_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_request.v1"
QUEUE_RECEIPT_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_queue_receipt.v1"
RECEIPT_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_receipt.v1"
QUEUE_RUN_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_queue_run.v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_INSTANCE_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
_LABEL_RE = re.compile(r"^blueprint-adp009d-[1-9][0-9]{9,}$")


class TerminalResourceReleaseError(ValueError):
    """A release-only request failed a fail-closed contract."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _is_digest(value: Any) -> bool:
    return bool(_DIGEST_RE.fullmatch(_string(value)))


def _is_identifier(value: Any) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(_string(value)))


def validate_terminal_resource_release_request(value: Mapping[str, Any]) -> list[str]:
    """Validate the immutable, zero-spend Website recovery capability."""

    request = _mapping(value)
    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("terminal_resource_release_schema_version_mismatch")
    for field in ("release_id", "launch_id", "run_id"):
        if not _is_identifier(request.get(field)):
            blockers.append(f"terminal_resource_release_{field}_invalid")
    if not _is_digest(request.get("request_digest")):
        blockers.append("terminal_resource_release_launch_request_digest_invalid")
    if request.get("provider") != "vast":
        blockers.append("terminal_resource_release_provider_must_be_vast")
    if not _INSTANCE_ID_RE.fullmatch(_string(request.get("instance_id"))):
        blockers.append("terminal_resource_release_instance_id_invalid")
    if not _LABEL_RE.fullmatch(_string(request.get("expected_label"))):
        blockers.append("terminal_resource_release_expected_label_invalid")
    blocker = _mapping(request.get("control_plane_terminal_blocker"))
    expected_blocker = {
        "schema_version": "task_evaluation_launch_control_plane_blocker.v1",
        "status": "blocked",
        "code": "control_plane_terminal_receipt_missing_after_spend_authority_expiry",
        "pipeline_terminal_receipt_observed": False,
        "provider_mutation_performed_by_webapp": False,
        "paid_execution_retry_performed": False,
        "execution_result": "not_observed",
        "scripted_positive_controls_result": "not_observed",
        "learned_policy_result": "not_observed",
    }
    for field, expected in expected_blocker.items():
        if blocker.get(field) != expected:
            blockers.append(f"terminal_resource_release_blocker_mismatch:{field}")
    for field in ("launch_id", "run_id", "request_digest"):
        if blocker.get(field) != request.get(field):
            blockers.append(f"terminal_resource_release_blocker_binding_mismatch:{field}")
    authorization = _mapping(request.get("authorization"))
    actor = _mapping(authorization.get("actor"))
    if authorization.get("action") != "terminal_provider_record_release":
        blockers.append("terminal_resource_release_action_invalid")
    if authorization.get("approved") is not True:
        blockers.append("terminal_resource_release_authority_missing")
    if authorization.get("max_additional_spend_usd") != 0:
        blockers.append("terminal_resource_release_spend_must_be_zero")
    if authorization.get("retry_cap") != 0:
        blockers.append("terminal_resource_release_retry_cap_must_be_zero")
    if actor.get("role") not in {"admin", "ops"} or not _is_identifier(actor.get("id")):
        blockers.append("terminal_resource_release_actor_invalid")
    if not _string(authorization.get("authorized_at")):
        blockers.append("terminal_resource_release_authorized_at_missing")
    if request.get("provider_mutation_performed_inside_web_request") is not False:
        blockers.append("terminal_resource_release_webapp_mutation_forbidden")
    if request.get("automatic_retry_performed") is not False:
        blockers.append("terminal_resource_release_automatic_retry_forbidden")
    if request.get("claim_ceiling") != "operational_resource_release_only":
        blockers.append("terminal_resource_release_claim_ceiling_invalid")
    if request.get("terminal_resource_release_digest") != canonical_digest(
        request, digest_field="terminal_resource_release_digest"
    ):
        blockers.append("terminal_resource_release_digest_mismatch")
    return sorted(set(blockers))


def _write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TerminalResourceReleaseError(
                f"immutable_terminal_resource_release_conflict:{path.name}"
            )
        return False


RECEIPT_FILENAME = "terminal_resource_release_receipt.json"


def release_redrive_admission(
    receipt: Mapping[str, Any] | None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether a blocked release may be re-armed by an operator.

    The invariant that matters is that the provider was never contacted. A
    release that failed before the canonical allocator reached a provider -- an
    unimportable allocator, an unhandled dispatcher error -- left nothing
    behind to reconcile, so re-running it can neither double-spend nor destroy
    an unexpected resource.

    When no outcome was retained the question cannot be answered from evidence,
    but for one action class it does not need to be. The release worker
    re-inspects the provider before any mutation and proceeds only on an
    API-confirmed exact ``instance_id``, a matching ``expected_label`` and a
    terminal status; an instance already absent is confirmed rather than
    touched. With zero additional spend and a zero retry cap, re-driving a
    release-only action can neither double-spend nor destroy an unexpected
    resource. Any other action without a retained receipt stays refused.
    """
    if not isinstance(receipt, Mapping):
        authorization = _mapping((request or {}).get("authorization"))
        if (
            isinstance(request, Mapping)
            and authorization.get("action") == "terminal_provider_record_release"
            and authorization.get("max_additional_spend_usd") == 0
            and authorization.get("retry_cap") == 0
        ):
            return {
                "admitted": True,
                "blockers": [],
                "admitted_without_retained_receipt": True,
                "evidence": "release_worker_reverifies_exact_instance_before_mutation",
            }
        return {"admitted": False, "blockers": ["redrive_refused_no_retained_receipt"]}

    blockers: list[str] = []
    if _string(receipt.get("status")) != "blocked":
        blockers.append("redrive_refused_release_not_blocked")
    if receipt.get("provider_mutation_attempted") not in (None, False):
        blockers.append("redrive_refused_provider_mutation_attempted")
    if list(receipt.get("provider_mutations_performed") or []):
        blockers.append("redrive_refused_provider_mutation_performed")
    return {"admitted": not blockers, "blockers": blockers}


def _retained_receipt(state_root: str | Path, release_id: str) -> dict[str, Any] | None:
    path = Path(state_root).expanduser().resolve() / release_id / RECEIPT_FILENAME
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def stage_terminal_resource_release_request(
    *, value: Mapping[str, Any], queue_root: str | Path,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate then append one immutable, provider-free queue record.

    Passing ``state_root`` lets an explicit operator re-submit re-arm a release
    that blocked before the provider was contacted. Re-arming is never
    automatic: only this call, driven by a website request, can move a blocked
    record back to ``pending``.
    """

    request = dict(value)
    blockers = validate_terminal_resource_release_request(request)
    if blockers:
        raise TerminalResourceReleaseError(",".join(blockers))
    queue = Path(queue_root).expanduser().resolve()
    digest = _string(request["terminal_resource_release_digest"])
    filename = f"{request['release_id']}-{digest[7:23]}.json"
    existing: Path | None = None
    for state in ("pending", "processing", "completed", "blocked"):
        candidate = queue / state / filename
        if candidate.exists():
            if existing is not None:
                raise TerminalResourceReleaseError(
                    f"duplicate_terminal_resource_release_queue_state:{filename}"
                )
            existing = candidate

    if existing is not None and existing.parent.name == "blocked" and state_root is not None:
        admission = release_redrive_admission(
            _retained_receipt(state_root, _string(request["release_id"])),
            request,
        )
        if not admission["admitted"]:
            raise TerminalResourceReleaseError(",".join(admission["blockers"]))
        rearmed = queue / "pending" / filename
        rearmed.parent.mkdir(parents=True, exist_ok=True)
        existing.replace(rearmed)
        return {
            "schema_version": QUEUE_RECEIPT_SCHEMA_VERSION,
            "status": "requeued",
            "already_exists": True,
            "release_id": request["release_id"],
            "launch_id": request["launch_id"],
            "terminal_resource_release_digest": digest,
            "provider_mutation_performed": False,
            "automatic_retry_performed": False,
        }

    path = existing or queue / "pending" / filename
    created = _write_immutable(path, request)
    return {
        "schema_version": QUEUE_RECEIPT_SCHEMA_VERSION,
        "status": "queued" if created else path.parent.name,
        "already_exists": not created,
        "release_id": request["release_id"],
        "launch_id": request["launch_id"],
        "terminal_resource_release_digest": digest,
        "provider_mutation_performed": False,
    }


__all__ = [
    "QUEUE_RECEIPT_SCHEMA_VERSION",
    "QUEUE_RUN_SCHEMA_VERSION",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "TerminalResourceReleaseError",
    "release_redrive_admission",
    "stage_terminal_resource_release_request",
    "validate_terminal_resource_release_request",
]
