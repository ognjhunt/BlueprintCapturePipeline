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
    # A capture-scoped policy governs exactly one capture. It still carries a
    # site and task for provenance, but its scope key is the capture id.
    if "capture_id" in policy and not _is_identifier(policy.get("capture_id")):
        blockers.append("site_task_reconstruction_policy_capture_id_invalid")
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
    *,
    policy_root: str | Path,
    site_id: str,
    task_id: str,
    capture_id: str | None = None,
) -> dict[str, Any]:
    """Return the single registered policy governing this capture, or abstain.

    Two ways to be governed, in priority order:

    * A policy naming this exact ``capture_id``.  Some captures have no site or
      task at all — a one-off walk of your own space is the obvious case — and
      those would otherwise be permanently unreconstructable.  An operator can
      register authority for one landed capture by name.
    * A policy for this exact ``site_id``/``task_id`` pair, for captures that
      arrive against a real job.

    There is deliberately no default and no nearest match: an ungoverned
    capture abstains, so it can never acquire selection authority it was never
    granted, and a capture-scoped policy governs exactly one capture.
    """

    root = Path(policy_root).expanduser().resolve()
    capture_matches: list[dict[str, Any]] = []
    site_task_matches: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(candidate, Mapping):
                continue
            scoped = str(candidate.get("capture_id") or "")
            if scoped:
                if capture_id and scoped == capture_id:
                    capture_matches.append(dict(candidate))
                # A capture-scoped policy never satisfies another capture, even
                # one sharing its site and task.
                continue
            if (
                str(candidate.get("site_id") or "") == site_id
                and str(candidate.get("task_id") or "") == task_id
            ):
                site_task_matches.append(dict(candidate))

    matches = capture_matches or site_task_matches
    scope = "capture" if capture_matches else "site_task"
    if not matches:
        raise CaptureReconstructionLaunchError(
            f"{MISSING_POLICY}:site={site_id};task={task_id};capture={capture_id or ''}"
        )
    if len(matches) > 1:
        raise CaptureReconstructionLaunchError(
            f"site_task_reconstruction_policy_ambiguous:scope={scope};"
            f"site={site_id};task={task_id};capture={capture_id or ''}"
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


#: Fields that legitimately differ between two deliveries of the same capture.
#: A Pub/Sub redelivery arrives at a later wall-clock time but is the same job;
#: everything outside this set differing means the paid work itself changed.
_VOLATILE_REQUEST_FIELDS = frozenset({"requested_at", "request_digest"})


def _stable_request_view(value: Mapping[str, Any]) -> str:
    return canonical_json(
        {k: v for k, v in dict(value).items() if k not in _VOLATILE_REQUEST_FIELDS}
    )


def _write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() == payload:
            return False
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptureReconstructionLaunchError(
                f"immutable_capture_reconstruction_launch_conflict:{path.name}"
            ) from exc
        if _stable_request_view(existing) == _stable_request_view(value):
            # Same capture, same policy, same paid authority, later delivery.
            return False
        raise CaptureReconstructionLaunchError(
            f"immutable_capture_reconstruction_launch_conflict:{path.name}"
        )


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


def compute_capture_digest(raw_root: str | Path) -> str:
    """Digest every immutable raw byte via the device's own hash manifest.

    ``raw/hashes.json`` is written on device and is already required to cover
    every raw file exactly; the completion marker is removed if it does not.
    Recomputing that agreement here means a capture whose bytes were altered
    after hashing, or which carries a file the manifest never listed, cannot
    acquire a digest at all.
    """

    raw = Path(raw_root).expanduser().resolve()
    hashes_path = raw / "hashes.json"
    if not hashes_path.is_file():
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_missing_hash_manifest"
        )
    try:
        payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_hash_manifest_unreadable"
        ) from exc
    artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_hash_manifest_empty"
        )

    observed: dict[str, str] = {}
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.name == "hashes.json":
            continue
        relative = str(path.relative_to(raw)).replace("\\", "/")
        observed[relative] = _sha256_file(path)

    uncovered = sorted(set(observed) - {str(k) for k in artifacts})
    if uncovered:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_hash_coverage_missing:" + ",".join(uncovered[:8])
        )
    missing = sorted({str(k) for k in artifacts} - set(observed))
    if missing:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_hash_target_missing:" + ",".join(missing[:8])
        )
    mismatched = sorted(
        relative
        for relative, digest in observed.items()
        if str(artifacts[relative]).lower().removeprefix(_DIGEST_PREFIX) != digest
    )
    if mismatched:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_hash_mismatch:" + ",".join(mismatched[:8])
        )
    return canonical_digest(
        {"artifacts": {key: observed[key] for key in sorted(observed)}},
        digest_field="capture_digest",
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_identity(sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def resolve_capture_identity(
    *, capture_root: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Read this capture's own identity and digests, or abstain.

    Site and task are never inferred from a nearby capture, a default, or a
    task *hypothesis*: an unbound capture abstains and says which binding is
    missing.
    """

    root = Path(capture_root).expanduser().resolve()
    raw = root / "raw"
    if not (raw / "capture_upload_complete.json").is_file():
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_capture_upload_complete_absent"
        )

    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}

    sources: list[Mapping[str, Any]] = [
        _mapping(payload),
        _read(raw / "capture_context.json"),
        _read(raw / "manifest.json"),
        _read(root / "pipeline_handoff.json"),
    ]
    # Site and task may legitimately be absent: a one-off walk of your own
    # space has neither. Report what the capture actually carries and let
    # policy resolution be the gate, rather than refusing here and making such
    # captures permanently unreconstructable. Nothing is inferred either way.
    site_id = _first_identity(sources, ("site_id", "siteId", "site_slug", "siteSlug"))
    task_id = _first_identity(sources, ("task_id", "taskId"))

    candidate_manifest = _read(raw / "downstream_candidate_manifest.json")
    candidate_manifest_digest = str(candidate_manifest.get("manifest_digest") or "")
    if not _is_digest(candidate_manifest_digest):
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_candidate_manifest_digest_absent"
        )

    return {
        "site_id": site_id,
        "task_id": task_id,
        "capture_id": _first_identity(sources, ("capture_id", "captureId"))
        or root.name,
        "scene_id": _first_identity(sources, ("scene_id", "sceneId")),
        "capture_digest": compute_capture_digest(raw),
        "candidate_manifest_digest": candidate_manifest_digest,
        "raw_root": str(raw),
    }


def enqueue_capture_reconstruction(
    *,
    capture_root: str | Path,
    payload: Mapping[str, Any],
    policy_root: str | Path,
    queue_root: str | Path,
    source_commit_sha: str,
    requested_at: str,
) -> dict[str, Any]:
    """Turn one staged capture into exactly one queued reconstruction launch.

    Every abstention happens before the queue is touched, so a capture that
    cannot legitimately proceed leaves no billable trace.
    """

    identity = resolve_capture_identity(capture_root=capture_root, payload=payload)
    policy = resolve_site_task_policy(
        policy_root=policy_root,
        site_id=identity["site_id"],
        task_id=identity["task_id"],
        capture_id=identity["capture_id"],
    )
    request = build_launch_request(
        policy=policy,
        capture_id=identity["capture_id"],
        scene_id=identity["scene_id"] or identity["capture_id"],
        intake_id=identity["capture_id"],
        capture_digest=identity["capture_digest"],
        candidate_manifest_digest=identity["candidate_manifest_digest"],
        capture_root_uri=str(_mapping(payload).get("raw_prefix_uri") or identity["raw_root"]),
        source_commit_sha=source_commit_sha,
        rights_evidence_digest=str(
            _mapping(policy.get("rights_authority")).get("rights_evidence_digest")
        ),
        revocation_check_status="clear",
        requested_at=requested_at,
    )
    return stage_launch_request(value=request, queue_root=queue_root)


def dispatch_launch_request(
    *,
    queue_path: str | Path,
    state_root: str | Path,
    derived_root: str | Path,
    capture_store_root: str | Path,
    execute: bool = False,
    allocator: Any = None,
) -> dict[str, Any]:
    """Carry one queued capture as far as its authority allows.

    Preparation is free and always runs: it validates the bundle and compiles
    the depth-seeded candidate-only COLMAP dataset.  Allocation is the paid
    boundary and is refused without explicit execute authority, so the ordinary
    path produces a complete, digest-bound execution request and stops there.

    Every stage is checkpointed before the next begins, and the paid stages are
    guarded against repetition, so a crashed dispatch resumes rather than
    re-spending.
    """

    from .canonical_3dgs_pipeline import (
        Canonical3DGSPipelineError,
        prepare_canonical_v32_training_dataset,
    )
    from .capture_reconstruction_checkpoints import (
        assert_paid_stage_not_repeated,
        read_checkpoints,
        record_checkpoint,
    )

    path = Path(queue_path).expanduser().resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    blockers = validate_launch_request(request)
    if blockers:
        raise CaptureReconstructionLaunchError(",".join(blockers))

    capture_digest = str(request["capture_digest"])
    ledger = read_checkpoints(state_root=state_root, capture_digest=capture_digest)

    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="upload_received",
        evidence={
            "capture_id": request["capture_id"],
            "candidate_manifest_digest": request["candidate_manifest_digest"],
        },
    )

    raw_root = Path(capture_store_root).expanduser().resolve()
    derived = Path(derived_root).expanduser().resolve() / capture_digest[7:23]

    # Preparation writes immutable artifacts and refuses to overwrite them, so
    # a resumed dispatch must read its earlier result rather than recompute it.
    already_prepared = "intake_validated" in ledger["recorded_stages"]
    if already_prepared:
        intake_evidence = _mapping(
            next(
                (
                    row
                    for row in ledger["checkpoints"]
                    if row.get("stage") == "intake_validated"
                ),
                {},
            ).get("evidence")
        )
    else:
        try:
            prepared = prepare_canonical_v32_training_dataset(
                capture_root=raw_root,
                output_root=derived,
                intake_id=str(request["intake_id"]),
                capture_digest=capture_digest,
                rights_and_retention={
                    "local_processing_authorized": True,
                    # Preparation never authorizes upload or spend; those are
                    # separate explicit control-plane inputs.
                    "provider_upload_authorized": False,
                    "paid_compute_authorized": False,
                },
                task_site_selection_profile=request["task_site_selection_profile"],
            )
        except Canonical3DGSPipelineError as exc:
            return {
                "schema_version": QUEUE_RUN_SCHEMA_VERSION,
                "status": "abstained",
                "capture_digest": capture_digest,
                "stage": "intake_validated",
                "blockers": sorted(set(getattr(exc, "codes", [str(exc)]))),
                "provider_mutation_performed": False,
            }

        preparation = _mapping(prepared.get("preparation"))
        dataset = _mapping(prepared.get("dataset"))
        intake_evidence = {
            "preparation_status": preparation.get("status"),
            "task_site_frame_selection_profile_digest": preparation.get(
                "task_site_frame_selection_profile_digest"
            ),
            "selected_decoded_frame_ordinals": preparation.get(
                "selected_decoded_frame_ordinals"
            ),
            "dataset_image_count": dataset.get("image_count"),
            "hidden_heldout_pixels_included": dataset.get(
                "hidden_heldout_pixels_included"
            ),
        }
        record_checkpoint(
            state_root=state_root,
            capture_digest=capture_digest,
            stage="intake_validated",
            evidence=intake_evidence,
        )

    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="queued",
        evidence={
            "idempotency_key": request["idempotency_key"],
            "arms": request["arms"],
            "max_spend_usd": request["max_spend_usd"],
            "hard_ttl_seconds": request["hard_ttl_seconds"],
        },
    )

    if not execute:
        return {
            "schema_version": QUEUE_RUN_SCHEMA_VERSION,
            "status": "prepared_awaiting_paid_authority",
            "capture_digest": capture_digest,
            "already_prepared": already_prepared,
            "derived_root": str(derived),
            "preparation_status": intake_evidence.get("preparation_status"),
            "dataset_image_count": intake_evidence.get("dataset_image_count"),
            "arms": request["arms"],
            "paid_execution_boundary": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "provider_mutation_performed": False,
        }

    # Paid boundary.  Refuse before touching a provider if this capture already
    # allocated, so a resumed dispatch cannot book a second instance.
    assert_paid_stage_not_repeated(
        state_root=state_root, capture_digest=capture_digest, stage="worker_allocated"
    )
    if allocator is None:
        raise CaptureReconstructionLaunchError(
            "capture_reconstruction_paid_dispatch_requires_allocator:"
            "run " + CANONICAL_ALLOCATOR_ENTRYPOINT
        )

    # The registered policy is the standing authority: a human set its ceiling,
    # TTL and authority id once, and the dispatcher spends against that without
    # asking again. The ceiling still travels with every allocation, so it is
    # enforced per run rather than assumed.
    allocation = _mapping(
        allocator(
            capture_digest=capture_digest,
            arms=list(request["arms"]),
            max_spend_usd=float(request["max_spend_usd"]),
            hard_ttl_seconds=int(request["hard_ttl_seconds"]),
            authority_id=str(request["authority_id"]),
            dataset_root=str(derived),
            retry_cap=0,
        )
    )
    # Record before judging. An allocator that mutated a provider and then
    # failed has still spent money, and the ledger has to say so or a resumed
    # dispatch would allocate a second instance.
    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="worker_allocated",
        evidence={
            "admission_digest": allocation.get("admission_digest"),
            "status": allocation.get("status"),
            "provider_mutations_performed": allocation.get(
                "provider_mutations_performed", 0
            ),
            "arms": request["arms"],
        },
    )
    if str(allocation.get("status")) != "execute_ready":
        return {
            "schema_version": QUEUE_RUN_SCHEMA_VERSION,
            "status": "allocation_blocked",
            "capture_digest": capture_digest,
            "stage": "worker_allocated",
            "blockers": sorted(
                {str(b) for b in (allocation.get("blockers") or []) if str(b)}
            )
            or ["capture_reconstruction_allocation_not_execute_ready"],
            "provider_mutation_performed": bool(
                allocation.get("provider_mutations_performed")
            ),
        }

    return {
        "schema_version": QUEUE_RUN_SCHEMA_VERSION,
        "status": "worker_allocated",
        "capture_digest": capture_digest,
        "admission_digest": allocation.get("admission_digest"),
        "arms": request["arms"],
        "derived_root": str(derived),
        "provider_mutation_performed": bool(
            allocation.get("provider_mutations_performed")
        ),
    }


def complete_capture_reconstruction(
    *,
    state_root: str | Path,
    capture_id: str,
    capture_digest: str,
    campaign_path: str | Path,
    completed_at: str,
    status_writer: Any,
    status_reader: Any = None,
    downstream_dispatch: Any = None,
) -> dict[str, Any]:
    """Publish a finished campaign, tell the WebApp, and hand off downstream.

    This is the tail of the chain after the paid arm returns.  It is separated
    from dispatch so it can run on a resumed process: each step checkpoints
    before the next, and re-running after a crash re-reports the identical
    status rather than restating a published capture.

    ``downstream_dispatch`` is injected because the analysis path
    (``post_capture_evidence_spine``) needs qualified dynamics geometry that
    reconstruction alone does not establish; a caller that has none passes
    nothing and the capture still reaches terminal, marked as not dispatched.
    """

    from .capture_reconstruction_checkpoints import read_checkpoints, record_checkpoint
    from .capture_reconstruction_status_sync import (
        status_from_campaign,
        sync_terminal_status,
    )

    ledger = read_checkpoints(state_root=state_root, capture_digest=capture_digest)
    status = status_from_campaign(
        capture_id=capture_id,
        capture_digest=capture_digest,
        campaign_path=campaign_path,
        completed_at=completed_at,
    )

    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="export",
        evidence={
            "artifact_digests": [row["digest"] for row in status["artifacts"]],
            "campaign_digest": status["campaign_digest"],
        },
    )

    sync = sync_terminal_status(
        status=status, writer=status_writer, reader=status_reader
    )
    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="publish",
        evidence={
            "status_digest": status["status_digest"],
            "state": status["state"],
        },
    )

    dispatched = False
    downstream_evidence: dict[str, Any] = {"dispatched": False}
    if downstream_dispatch is not None and status["state"] == "published":
        result = downstream_dispatch(status=status)
        dispatched = True
        downstream_evidence = {
            "dispatched": True,
            "result": json.loads(canonical_json(_mapping(result))),
        }
    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="downstream_dispatched",
        evidence=downstream_evidence,
    )

    record_checkpoint(
        state_root=state_root,
        capture_digest=capture_digest,
        stage="terminal",
        evidence={"state": status["state"], "status_digest": status["status_digest"]},
    )

    return {
        "schema_version": QUEUE_RUN_SCHEMA_VERSION,
        "status": "terminal",
        "capture_id": capture_id,
        "capture_digest": capture_digest,
        "terminal_state": status["state"],
        "status_digest": status["status_digest"],
        "status_written": sync["written"],
        "downstream_dispatched": dispatched,
        "stages_before": ledger["recorded_stages"],
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


def build_capture_scoped_policy(
    *,
    capture_id: str,
    site_id: str,
    task_id: str,
    selector: str,
    parameters: Mapping[str, Any],
    rights_profile: str,
    rights_evidence_digest: str,
    arms: Sequence[str],
    max_spend_usd: float,
    hard_ttl_seconds: int,
    authority_id: str,
) -> dict[str, Any]:
    """Compose authority for exactly one already-landed capture.

    Every value is supplied by the operator. Nothing is defaulted, because the
    whole point of the policy is to be the place a human took responsibility
    for spending money on a specific capture.
    """

    policy: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": f"capture-{capture_id}",
        "capture_id": str(capture_id),
        "site_id": str(site_id),
        "task_id": str(task_id),
        "selector": str(selector),
        "parameters": json.loads(canonical_json(dict(parameters))),
        "rights_authority": {
            "rights_profile": str(rights_profile),
            "rights_evidence_digest": str(rights_evidence_digest),
        },
        "arms": sorted(str(arm) for arm in arms),
        "max_spend_usd": float(max_spend_usd),
        "hard_ttl_seconds": int(hard_ttl_seconds),
        "authority_id": str(authority_id),
    }
    policy["policy_digest"] = canonical_digest(policy, digest_field="policy_digest")
    blockers = validate_site_task_reconstruction_policy(policy)
    if blockers:
        raise CaptureReconstructionLaunchError(",".join(blockers))
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    drain = commands.add_parser("drain", help="preview or dispatch queued launches")
    drain.add_argument("--queue-root", required=True)
    drain.add_argument("--state-root", required=True)
    drain.add_argument("--execute", action="store_true")

    bind = commands.add_parser(
        "bind-capture",
        help="register reconstruction authority for one already-landed capture",
    )
    bind.add_argument("--policy-root", required=True)
    bind.add_argument("--capture-id", required=True)
    bind.add_argument("--site-id", required=True)
    bind.add_argument("--task-id", required=True)
    bind.add_argument("--selector", required=True, choices=sorted(ALLOWED_CAPTURE_SELECTORS))
    bind.add_argument("--parameters-json", required=True)
    bind.add_argument("--rights-profile", required=True)
    bind.add_argument("--rights-evidence-digest", required=True)
    bind.add_argument("--arm", action="append", required=True, choices=sorted(ADMITTED_ARMS))
    bind.add_argument("--max-spend-usd", type=float, required=True)
    bind.add_argument("--hard-ttl-seconds", type=int, required=True)
    bind.add_argument("--authority-id", required=True)

    arguments = parser.parse_args(argv)

    if arguments.command == "bind-capture":
        policy = build_capture_scoped_policy(
            capture_id=arguments.capture_id,
            site_id=arguments.site_id,
            task_id=arguments.task_id,
            selector=arguments.selector,
            parameters=json.loads(arguments.parameters_json),
            rights_profile=arguments.rights_profile,
            rights_evidence_digest=arguments.rights_evidence_digest,
            arms=arguments.arm,
            max_spend_usd=arguments.max_spend_usd,
            hard_ttl_seconds=arguments.hard_ttl_seconds,
            authority_id=arguments.authority_id,
        )
        root = Path(arguments.policy_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"capture-{arguments.capture_id}.json"
        _write_immutable(path, policy)
        json.dump(
            {"policy_path": str(path), "policy_digest": policy["policy_digest"]},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

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
    "build_capture_scoped_policy",
    "build_launch_request",
    "complete_capture_reconstruction",
    "compute_capture_digest",
    "dispatch_launch_request",
    "enqueue_capture_reconstruction",
    "mint_frame_selection_profile",
    "resolve_capture_identity",
    "process_launch_queue",
    "resolve_site_task_policy",
    "stage_launch_request",
    "validate_launch_request",
    "validate_site_task_reconstruction_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
