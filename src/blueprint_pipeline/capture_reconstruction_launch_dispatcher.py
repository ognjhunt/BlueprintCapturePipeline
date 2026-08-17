"""Automatic, exactly-once bridge from an admitted capture to the 3DGS campaign.

An uploaded production capture cannot start paid reconstruction by itself, and
it must not acquire selection authority by default.  This module closes that gap
without weakening either rule:

* A ``site_task_reconstruction_policy.v1`` is registered once, by a human, per
  site/task.  It names the selector, its parameters, the rights authority, the
  admitted arms, and the paid ceiling.  Absent policy abstains; nothing here
  invents a selector, a frame budget, or a spend ceiling.
* At dispatch time the policy is bound to *this* capture's observed
  ``candidate_manifest_digest`` to mint the per-capture
  ``task_site_frame_selection_profile.v1`` the canonical preparation requires.
  A profile minted for one capture cannot admit another.
* The request is enqueued immutably under a digest-derived name, so a duplicate
  submit is a no-op and a mutated resubmit is a hard conflict rather than a
  second billable job.

Enqueue performs no provider mutation.  Consistent with
``live_pipeline_intake_service``, paid launches are only durably queued here;
the dispatcher remains the sole bridge to the canonical allocator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_v32_candidate_admission import (
    ALLOWED_CAPTURE_SELECTORS,
    SELECTION_PROFILE_SCHEMA_VERSION,
)
from .decision_evidence_contracts import canonical_digest, canonical_json

POLICY_SCHEMA_VERSION = "site_task_reconstruction_policy.v1"
LAUNCH_REQUEST_SCHEMA_VERSION = "capture_reconstruction_launch_request.v1"
LAUNCH_RECEIPT_SCHEMA_VERSION = "capture_reconstruction_launch_queue_receipt.v1"
QUEUE_RUN_SCHEMA_VERSION = "capture_reconstruction_launch_queue_run.v1"

MISSING_POLICY = "site_task_reconstruction_policy_absent"
CANONICAL_ALLOCATOR_ENTRYPOINT = (
    "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
)
ADMITTED_ARMS = ("postshot-primary", "splatfacto-comparison")
QUEUE_STATES = ("pending", "processing", "completed", "blocked")

_DIGEST_PREFIX = "sha256:"
_SECRET_KEY_FRAGMENTS = ("credential", "password", "private_key", "secret", "token")


class CaptureReconstructionLaunchError(ValueError):
    """Stable fail-closed dispatcher error."""


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith(_DIGEST_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _is_identifier(value: Any) -> bool:
    text = str(value or "")
    return bool(text) and len(text) <= 192 and all(
        character.isalnum() or character in "-_." for character in text
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > 0
    )


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS):
                return True
            if _contains_secret_key(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# Site/task policy: registered once, never inferred
# ---------------------------------------------------------------------------


def validate_site_task_reconstruction_policy(value: Mapping[str, Any]) -> list[str]:
    """Return stable blockers for a registered site/task reconstruction policy."""

    policy = _mapping(value)
    blockers: list[str] = []
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        blockers.append("site_task_reconstruction_policy_schema_invalid")
    for field in ("policy_id", "site_id", "task_id", "authority_id"):
        if not _is_identifier(policy.get(field)):
            blockers.append(f"site_task_reconstruction_policy_{field}_invalid")
    if policy.get("selector") not in ALLOWED_CAPTURE_SELECTORS:
        blockers.append("site_task_reconstruction_policy_selector_invalid")
    if not isinstance(policy.get("parameters"), Mapping) or not policy.get("parameters"):
        blockers.append("site_task_reconstruction_policy_parameters_invalid")
    rights = policy.get("rights_authority")
    if not isinstance(rights, Mapping) or not _is_digest(
        rights.get("rights_evidence_digest")
    ) or not str(rights.get("rights_profile") or ""):
        blockers.append("site_task_reconstruction_policy_rights_authority_invalid")
    arms = policy.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or any(arm not in ADMITTED_ARMS for arm in arms)
    ):
        blockers.append("site_task_reconstruction_policy_arms_invalid")
    if not _positive_number(policy.get("max_spend_usd")):
        blockers.append("site_task_reconstruction_policy_max_spend_invalid")
    if not isinstance(policy.get("hard_ttl_seconds"), int) or isinstance(
        policy.get("hard_ttl_seconds"), bool
    ) or int(policy.get("hard_ttl_seconds") or 0) <= 0:
        blockers.append("site_task_reconstruction_policy_hard_ttl_invalid")
    if _contains_secret_key(policy):
        blockers.append("site_task_reconstruction_policy_contains_secret_material")
    if policy.get("policy_digest") != canonical_digest(
        policy, digest_field="policy_digest"
    ):
        blockers.append("site_task_reconstruction_policy_digest_mismatch")
    return sorted(set(blockers))


def resolve_site_task_policy(
    *, policy_root: str | Path, site_id: str, task_id: str
) -> dict[str, Any]:
    """Return the single registered policy for this exact site/task, or abstain.

    There is deliberately no default and no nearest match: an unregistered
    site/task pair abstains so a capture can never acquire selection authority
    it was never granted.
    """

    root = Path(policy_root).expanduser().resolve()
    matches: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(candidate, Mapping):
                continue
            if (
                str(candidate.get("site_id") or "") == site_id
                and str(candidate.get("task_id") or "") == task_id
            ):
                matches.append(dict(candidate))
    if not matches:
        raise CaptureReconstructionLaunchError(
            f"{MISSING_POLICY}:site={site_id};task={task_id}"
        )
    if len(matches) > 1:
        raise CaptureReconstructionLaunchError(
            f"site_task_reconstruction_policy_ambiguous:site={site_id};task={task_id}"
        )
    policy = matches[0]
    blockers = validate_site_task_reconstruction_policy(policy)
    if blockers:
        raise CaptureReconstructionLaunchError(",".join(blockers))
    return policy


# ---------------------------------------------------------------------------
# Per-capture selection profile
# ---------------------------------------------------------------------------


def mint_frame_selection_profile(
    *,
    policy: Mapping[str, Any],
    candidate_manifest_digest: str,
    rights_evidence_digest: str,
    revocation_check_status: str,
) -> dict[str, Any]:
    """Bind a registered policy to one capture's observed candidate manifest.

    The result is exactly the ``task_site_frame_selection_profile.v1`` the
    canonical preparation already validates; this mints it rather than
    re-implementing selection.
    """

    blockers = validate_site_task_reconstruction_policy(policy)
    if blockers:
        raise CaptureReconstructionLaunchError(",".join(blockers))
    if not _is_digest(candidate_manifest_digest):
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_candidate_manifest_digest_invalid"
        )
    if str(revocation_check_status) != "clear":
        raise CaptureReconstructionLaunchError(
            f"capture_reconstruction_rights_revoked:{revocation_check_status}"
        )
    authority_digest = str(
        _mapping(policy.get("rights_authority")).get("rights_evidence_digest") or ""
    )
    if not _is_digest(rights_evidence_digest) or rights_evidence_digest != authority_digest:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_rights_evidence_mismatch"
        )

    profile: dict[str, Any] = {
        "schema_version": SELECTION_PROFILE_SCHEMA_VERSION,
        "site_id": str(policy["site_id"]),
        "task_id": str(policy["task_id"]),
        "candidate_manifest_digest": str(candidate_manifest_digest),
        "selector": str(policy["selector"]),
        "parameters": json.loads(canonical_json(_mapping(policy.get("parameters")))),
        "rights_authorization": {
            "status": "authorized",
            "latest_revocation_check_status": "clear",
            "rights_evidence_digest": rights_evidence_digest,
        },
        "minted_from_policy_digest": str(policy["policy_digest"]),
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    return profile


# ---------------------------------------------------------------------------
# Launch request
# ---------------------------------------------------------------------------


def build_launch_request(
    *,
    policy: Mapping[str, Any],
    capture_id: str,
    scene_id: str,
    intake_id: str,
    capture_digest: str,
    candidate_manifest_digest: str,
    capture_root_uri: str,
    source_commit_sha: str,
    rights_evidence_digest: str,
    revocation_check_status: str,
    requested_at: str,
) -> dict[str, Any]:
    """Compose the immutable launch request for one capture."""

    selection_profile = mint_frame_selection_profile(
        policy=policy,
        candidate_manifest_digest=candidate_manifest_digest,
        rights_evidence_digest=rights_evidence_digest,
        revocation_check_status=revocation_check_status,
    )
    request: dict[str, Any] = {
        "schema_version": LAUNCH_REQUEST_SCHEMA_VERSION,
        "capture_id": str(capture_id),
        "scene_id": str(scene_id),
        "intake_id": str(intake_id),
        "capture_digest": str(capture_digest),
        "candidate_manifest_digest": str(candidate_manifest_digest),
        "capture_root_uri": str(capture_root_uri),
        "source_commit_sha": str(source_commit_sha),
        "site_id": str(policy["site_id"]),
        "task_id": str(policy["task_id"]),
        "policy_id": str(policy["policy_id"]),
        "policy_digest": str(policy["policy_digest"]),
        "policy_max_spend_usd": float(policy["max_spend_usd"]),
        "authority_id": str(policy["authority_id"]),
        "arms": sorted(str(arm) for arm in policy["arms"]),
        "max_spend_usd": float(policy["max_spend_usd"]),
        "hard_ttl_seconds": int(policy["hard_ttl_seconds"]),
        "task_site_selection_profile": selection_profile,
        "task_site_selection_profile_digest": selection_profile["profile_digest"],
        "requested_at": str(requested_at),
        "allocator_entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "retry_cap": 0,
    }
    request["idempotency_key"] = canonical_digest(
        {
            "capture_digest": request["capture_digest"],
            "candidate_manifest_digest": request["candidate_manifest_digest"],
            "policy_digest": request["policy_digest"],
        },
        digest_field="idempotency_key",
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def validate_launch_request(value: Mapping[str, Any]) -> list[str]:
    """Return stable blockers for an enqueued launch request."""

    request = _mapping(value)
    blockers: list[str] = []
    if request.get("schema_version") != LAUNCH_REQUEST_SCHEMA_VERSION:
        blockers.append("capture_reconstruction_launch_request_schema_invalid")
    for field in ("capture_id", "scene_id", "intake_id", "site_id", "task_id", "policy_id"):
        if not _is_identifier(request.get(field)):
            blockers.append(f"capture_reconstruction_launch_{field}_invalid")
    for field in (
        "capture_digest",
        "candidate_manifest_digest",
        "policy_digest",
        "task_site_selection_profile_digest",
        "idempotency_key",
    ):
        if not _is_digest(request.get(field)):
            blockers.append(f"capture_reconstruction_launch_{field}_invalid")
    if request.get("retry_cap") != 0:
        blockers.append("capture_reconstruction_launch_retry_cap_must_be_zero")
    if not _positive_number(request.get("max_spend_usd")):
        blockers.append("capture_reconstruction_launch_max_spend_invalid")
    elif _positive_number(request.get("policy_max_spend_usd")) and float(
        request["max_spend_usd"]
    ) > float(request["policy_max_spend_usd"]):
        blockers.append("capture_reconstruction_launch_spend_exceeds_policy")
    arms = request.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or any(arm not in ADMITTED_ARMS for arm in arms)
    ):
        blockers.append("capture_reconstruction_launch_arms_invalid")
    profile = request.get("task_site_selection_profile")
    if not isinstance(profile, Mapping):
        blockers.append("capture_reconstruction_launch_selection_profile_missing")
    else:
        if profile.get("candidate_manifest_digest") != request.get(
            "candidate_manifest_digest"
        ):
            blockers.append("capture_reconstruction_launch_selection_profile_unbound")
        if profile.get("profile_digest") != request.get(
            "task_site_selection_profile_digest"
        ):
            blockers.append("capture_reconstruction_launch_selection_profile_digest_mismatch")
    if _contains_secret_key(request):
        blockers.append("capture_reconstruction_launch_request_contains_secret_material")
    if request.get("request_digest") != canonical_digest(
        request, digest_field="request_digest"
    ):
        blockers.append("capture_reconstruction_launch_request_digest_mismatch")
    return sorted(set(blockers))


# ---------------------------------------------------------------------------
# Exactly-once enqueue
# ---------------------------------------------------------------------------


def _write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CaptureReconstructionLaunchError(
                f"immutable_capture_reconstruction_launch_conflict:{path.name}"
            )
        return False


def stage_launch_request(
    *, value: Mapping[str, Any], queue_root: str | Path
) -> dict[str, Any]:
    """Enqueue one capture's launch request exactly once.

    The filename is derived from the idempotency key, so the same capture under
    the same policy always lands on the same queue entry.  A resubmit whose
    bytes differ is a conflict, never a second billable request.
    """

    request = dict(value)
    blockers = validate_launch_request(request)
    if blockers:
        raise CaptureReconstructionLaunchError(",".join(blockers))
    queue = Path(queue_root).expanduser().resolve()
    key = str(request["idempotency_key"])
    filename = f"{request['capture_id']}-{key[len(_DIGEST_PREFIX) : len(_DIGEST_PREFIX) + 16]}.json"
    existing: Path | None = None
    for state in QUEUE_STATES:
        candidate = queue / state / filename
        if candidate.exists():
            if existing is not None:
                raise CaptureReconstructionLaunchError(
                    f"duplicate_capture_reconstruction_queue_state:{filename}"
                )
            existing = candidate
    path = existing or queue / "pending" / filename
    created = _write_immutable(path, request)
    return {
        "schema_version": LAUNCH_RECEIPT_SCHEMA_VERSION,
        "status": "queued" if created else path.parent.name,
        "already_exists": not created,
        "capture_id": request["capture_id"],
        "capture_digest": request["capture_digest"],
        "idempotency_key": key,
        "request_digest": request["request_digest"],
        "policy_digest": request["policy_digest"],
        "queue_path": str(path),
        "provider_mutation_performed": False,
    }


def process_launch_queue(
    *,
    queue_root: str | Path,
    state_root: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Preview or dispatch queued launches.

    Without explicit execute authority this is a read-only preview: it reports
    what would be dispatched and performs no provider mutation.
    """

    queue = Path(queue_root).expanduser().resolve()
    state = Path(state_root).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    pending = sorted((queue / "pending").glob("*.json")) if (queue / "pending").is_dir() else []

    previewed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for path in pending:
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blocked.append({"queue_path": str(path), "blockers": ["queue_entry_unreadable"]})
            continue
        request_blockers = validate_launch_request(request)
        if request_blockers:
            blocked.append({"queue_path": str(path), "blockers": request_blockers})
            continue
        previewed.append(
            {
                "queue_path": str(path),
                "capture_id": request["capture_id"],
                "capture_digest": request["capture_digest"],
                "idempotency_key": request["idempotency_key"],
                "arms": request["arms"],
                "max_spend_usd": request["max_spend_usd"],
                "hard_ttl_seconds": request["hard_ttl_seconds"],
                "allocator_entrypoint": request["allocator_entrypoint"],
            }
        )

    return {
        "schema_version": QUEUE_RUN_SCHEMA_VERSION,
        "queue_root": str(queue),
        "state_root": str(state),
        "execute_requested": bool(execute),
        "previewed": len(previewed),
        "dispatched": 0,
        "blocked": len(blocked),
        "entries": previewed,
        "blocked_entries": blocked,
        "provider_mutation_performed": False,
        "paid_execution_boundary": CANONICAL_ALLOCATOR_ENTRYPOINT,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    result = process_launch_queue(
        queue_root=arguments.queue_root,
        state_root=arguments.state_root,
        execute=arguments.execute,
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "ADMITTED_ARMS",
    "CANONICAL_ALLOCATOR_ENTRYPOINT",
    "CaptureReconstructionLaunchError",
    "LAUNCH_RECEIPT_SCHEMA_VERSION",
    "LAUNCH_REQUEST_SCHEMA_VERSION",
    "MISSING_POLICY",
    "POLICY_SCHEMA_VERSION",
    "QUEUE_RUN_SCHEMA_VERSION",
    "SELECTION_PROFILE_SCHEMA_VERSION",
    "build_launch_request",
    "mint_frame_selection_profile",
    "process_launch_queue",
    "resolve_site_task_policy",
    "stage_launch_request",
    "validate_launch_request",
    "validate_site_task_reconstruction_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
