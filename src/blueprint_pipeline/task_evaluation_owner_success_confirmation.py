"""The owner's confirmation of a website job's built success thresholds, sealed under its intent.

A website job's task template carries Blueprint's fixed development controls
as its success thresholds, marked ``proposal_only``
(``website_native_submission``): nothing translated or confirmed the owner's
own bar, so every owner-scored step (the team packet, screening) refuses the
job with ``configured_owner_success_contract_authority_missing``. The owner can
accept those exact thresholds through the operator door
(``task_evaluation_owner_success_confirmation_door``, kind
``confirm-owner-success-contract``): a preview names the published job, the
thresholds and an ``apply_ack`` digest; an apply repeating that digest writes
one record here, under ``<intent>/owner-success-confirmations/<digest>.json``.

The record is bound to the intent (id, digest, owner, issuer), the published
configured scene revision, the exact success-criteria document the job was
built with (its published reference), the thresholds that document puts in the
task (``configured_success_criteria``, with a copy kept in the record) and the
``proposal_only`` authority it confirms. ``confirmed_owner_authority`` reads it
back for the team packet, which recomputes every binding from what it already
reads and only then compiles the owner contract as the owner's. A changed
revision, document, threshold or proposal, another owner or a revoked intent
finds no confirmation, and the job refuses exactly as before.

A confirmation changes no threshold: it records only that the owner accepted
the thresholds as built. Nothing here starts compute, calls a provider or moves
money, and nothing else is ever read as a confirmation.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "task_evaluation_owner_success_contract_confirmation.v1"
DIRECTORY = "owner-success-confirmations"
ACK = "confirm-owner-success-contract"
SCOPE = "built_success_thresholds_acceptance_only"
# The confirmed authority's reference; the sealed owner contract carries it as its author.
AUTHORITY_PREFIX = "owner-success-confirmation:"
CONFIRMED_BASIS = "owner_confirmed_built_thresholds_via_operator_door"
CHANNEL = "operator_door"
# Both native adapters drop exactly these from the success document (``configured_success_criteria``).
_UNSCORED_SUCCESS_FIELDS = frozenset({"schema_version", "status"})
_BINDING_FIELDS = ("intent_id", "intent_digest", "owner", "authenticated_issuer",
                   "configured_scene_revision_digest", "task_id", "success_criteria_document",
                   "configured_success_criteria", "configured_success_criteria_digest",
                   "proposal_authority", "proposal_authority_digest")
RECORD_FIELDS = frozenset({"schema_version", "scope", *_BINDING_FIELDS, "approved_by", "approval_evidence",
                           "authorization_reference", "issuer_channel", "operator_uid", "confirmed_at",
                           "thresholds_changed", "provider_mutation_performed", "confirmation_digest"})


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("owner_success_confirmation_" + code)


def _safe(path: Path) -> None:
    _require(path.is_absolute() and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), "store_unsafe")


def configured_success_criteria(success_document: Mapping[str, Any]) -> dict[str, Any]:
    """The thresholds a success document puts in the task, as both native adapters copy them."""

    return {key: json.loads(json.dumps(value)) for key, value in success_document.items()
            if key not in _UNSCORED_SUCCESS_FIELDS}


def binding(*, intent: Mapping[str, Any], revision: Mapping[str, Any], criteria: Mapping[str, Any],
            proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a confirmation is bound to, from the intent, the published revision and the task.

    The door builds it from the retained published documents; the team packet
    from the task it compiled. Only a ``proposal_only`` authority over thresholds
    that require an owner contract can be confirmed.
    """

    from .task_evaluation_scene_intent_contracts import canonical_digest

    reference = (revision.get("task_template") or {}).get("success_criteria")
    _require(isinstance(reference, Mapping) and isinstance(reference.get("uri"), str)
             and isinstance(reference.get("digest"), str) and type(reference.get("size_bytes")) is int,
             "revision_success_reference_invalid")
    _require(isinstance(proposal, Mapping) and proposal.get("confirmation_status") == "proposal_only",
             "not_a_proposal")
    _require(isinstance(criteria, Mapping) and criteria.get("owner_success_contract_required") is True,
             "owner_contract_not_required")
    thresholds = json.loads(json.dumps(dict(criteria)))
    authority = json.loads(json.dumps(dict(proposal)))
    return {"intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
            "owner": dict(intent["request"]["owner"]), "authenticated_issuer": intent["authenticated_issuer"],
            "configured_scene_revision_digest": revision["revision_digest"],
            "task_id": str(revision["task_template"]["identity"]["id"]),
            "success_criteria_document": {key: reference[key] for key in ("uri", "digest", "size_bytes")},
            "configured_success_criteria": thresholds,
            "configured_success_criteria_digest": canonical_digest(thresholds),
            "proposal_authority": authority, "proposal_authority_digest": canonical_digest(authority)}


def _valid_record(path: Path, intent: Mapping[str, Any]) -> dict[str, Any] | None:
    """One sealed confirmation of this intent's owner, or None for anything else in the store."""

    from .task_evaluation_scene_intent_contracts import canonical_digest

    if path.is_symlink() or not path.is_file() or path.suffix != ".json":
        return None
    try:
        row = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if not (isinstance(row, dict) and set(row) == RECORD_FIELDS and row.get("schema_version") == SCHEMA
            and row.get("scope") == SCOPE
            and row.get("confirmation_digest") == canonical_digest(row, digest_field="confirmation_digest")
            and path.name == str(row["confirmation_digest"]).removeprefix("sha256:") + ".json"
            and row["intent_id"] == intent["intent_id"] and row["intent_digest"] == intent["intent_digest"]
            and row["owner"] == intent["request"]["owner"] and row["approved_by"] == row["owner"]["user_id"]
            and row["authenticated_issuer"] == intent["authenticated_issuer"]
            and isinstance(row["approval_evidence"], str) and bool(row["approval_evidence"].strip())
            and isinstance(row["authorization_reference"], str) and bool(row["authorization_reference"].strip())
            and row["issuer_channel"] == CHANNEL and type(row["operator_uid"]) is int
            and row["thresholds_changed"] is False and row["provider_mutation_performed"] is False
            and isinstance(row["configured_success_criteria"], dict)
            and row["configured_success_criteria_digest"] == canonical_digest(row["configured_success_criteria"])
            and isinstance(row["proposal_authority"], dict)
            and row["proposal_authority_digest"] == canonical_digest(row["proposal_authority"])):
        return None
    return row


def confirmations(directory: str | Path, intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every valid sealed confirmation of this intent's owner, earliest first; none once revoked."""

    directory = Path(directory)
    _safe(directory)
    store = directory / DIRECTORY
    if (directory / "revoked.json").exists() or store.is_symlink() or not store.is_dir():
        return []
    rows = [row for row in (_valid_record(path, intent) for path in sorted(store.glob("*.json"))) if row]
    return sorted(rows, key=lambda row: (row["confirmed_at"], row["confirmation_digest"]))


def find_confirmation(directory: Path, intent: Mapping[str, Any],
                      expected: Mapping[str, Any]) -> dict[str, Any] | None:
    """The earliest valid record that confirms exactly ``expected`` for this intent, else None.

    Records for another revision, document, threshold set or proposal are kept
    and ignored; a revoked intent has no confirmation.
    """

    return next((row for row in confirmations(directory, intent)
                 if all(row[field] == expected[field] for field in _BINDING_FIELDS)), None)


def confirmed_owner_authority(*, task_spec: Mapping[str, Any], revision: Mapping[str, Any],
                              intent: Mapping[str, Any], directory: str | Path) -> dict[str, Any] | None:
    """The owner's confirmed authority for this exact compiled task, or None (the task is unchanged).

    Applies only to a ``proposal_only`` authority, and only through a sealed
    record bound to everything ``binding`` names, recomputed here from the task
    and revision the caller compiled. The original proposal is kept under
    ``proposal_authority``.
    """

    authority = task_spec.get("configured_owner_authority")
    criteria = task_spec.get("configured_success_criteria")
    if not isinstance(authority, Mapping) or authority.get("confirmation_status") != "proposal_only":
        return None
    try:
        expected = binding(intent=intent, revision=revision, criteria=criteria or {}, proposal=authority)
        record = find_confirmation(Path(directory), intent, expected)
    except (KeyError, TypeError, ValueError):
        return None
    if record is None:
        return None
    return {"author_source": "task_owner", "confirmation_status": "confirmed",
            "accepted_by": record["owner"]["user_id"],
            "authority_reference": AUTHORITY_PREFIX + record["confirmation_digest"],
            "basis": CONFIRMED_BASIS, "confirmation_digest": record["confirmation_digest"],
            "proposal_authority": dict(authority)}


def published_job(directory: Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    """The intent's published revision, thresholds and proposal, from retained host records only.

    The progression's completion names the launch; the launch's validated
    terminal result names the revision, whose bytes must be the progression's
    published result. The success-criteria and template documents are the
    copies the completion join fetched and retained beside the launch, each
    checked against the revision's own reference. Nothing is fetched.
    """

    import hashlib

    from . import task_evaluation_scene_intake as intake
    from .robot_free_readiness_retention import WORK_DIRECTORY
    from .task_evaluation_configured_controls_deferred_inputs import _document_name
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .task_evaluation_configured_scene_revision import validate_configured_scene_revision
    from .task_evaluation_rigid_relocation_native_adapter import (
        DEFINITION_CONTRACT_PATH,
        SUCCESS_CONTRACT_PATH,
    )

    def sha(payload: bytes) -> str:
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def read_bound(path: Path, reference: Mapping[str, Any], code: str) -> bytes:
        _require(not path.is_symlink() and path.is_file(), code)
        payload = path.read_bytes()
        _require(sha(payload) == reference.get("digest") and len(payload) == reference.get("size_bytes"), code)
        return payload

    progression = intake._read(directory / "progression.json", "progression_digest")
    completion = (progression.get("state") or {}).get("scene_preparation_completion") or {}
    published = progression.get("result_reference") or {}
    receipt = completion.get("launch_receipt") or {}
    _require(progression.get("intent_id") == intent["intent_id"]
             and progression.get("intent_digest") == intent["intent_digest"]
             and isinstance(published.get("digest"), str) and isinstance(receipt.get("path"), str),
             "job_not_published")
    receipt_path = Path(receipt["path"])
    _safe(receipt_path)
    read_bound(receipt_path, {"digest": receipt.get("sha256"), "size_bytes": receipt.get("size_bytes")},
               "launch_receipt_changed")
    run_root = receipt_path.parent
    try:
        result, _receipt, _zero = _validate_source(run_root)
    except RuntimeError as exc:  # the source validator's own typed refusal
        raise ValueError("owner_success_confirmation_launch_result_invalid:" + str(exc)[:160]) from exc
    revision_path = Path(str(result.get("configured_scene_revision_path") or ""))
    _safe(revision_path)
    _require(revision_path.is_file(), "revision_unavailable")
    payload = revision_path.read_bytes()
    _require(sha(payload) == published["digest"], "revision_not_the_published_result")
    revision = validate_configured_scene_revision(json.loads(payload))
    _require(revision.get("revision_digest") == result.get("configured_scene_revision_digest"),
             "revision_digest_mismatch")
    documents = run_root / WORK_DIRECTORY / "documents"
    success = json.loads(read_bound(documents / _document_name(SUCCESS_CONTRACT_PATH),
                                    revision["task_template"]["success_criteria"], "success_document_unavailable"))
    template = json.loads(read_bound(documents / _document_name(DEFINITION_CONTRACT_PATH),
                                     revision["task_template"]["definition"], "template_document_unavailable"))
    _require(isinstance(success, dict) and isinstance(template, dict), "documents_invalid")
    return {"revision": revision, "revision_path": str(revision_path), "launch_run_root": str(run_root),
            "criteria": configured_success_criteria(success),
            "proposal": template.get("owner_success_contract_authority")}


def preview_owner_success_confirmation(*, queue_root: str | Path, intent_id: str, owner: Mapping[str, Any],
                                       approved_by: str, approval_evidence: str) -> dict[str, Any]:
    """What an apply would confirm, and its ``apply_ack``; writes nothing."""

    from . import task_evaluation_scene_intake as intake

    _require(intake._identifier(intent_id), "not_authorized")
    root = Path(queue_root)
    _safe(root)
    directory = root / intent_id
    _safe(directory)
    intent = intake._read(directory / "intent.json", "intent_digest")
    evidence = approval_evidence.strip() if isinstance(approval_evidence, str) else ""
    _require(intent["intent_id"] == intent_id and intent["request"]["owner"] == dict(owner)
             and approved_by == owner.get("user_id") and bool(evidence)
             and not (directory / "revoked.json").exists(), "owner_or_revocation_mismatch")
    job = published_job(directory, intent)
    bound = binding(intent=intent, revision=job["revision"], criteria=job["criteria"],
                    proposal=job["proposal"] or {})
    preview = {**bound, "approved_by": approved_by, "approval_evidence": evidence}
    existing = find_confirmation(directory, intent, bound)
    return {"preview": preview, "apply_ack": intake.canonical_digest(preview),
            "existing_confirmation_digest": existing["confirmation_digest"] if existing else None,
            "revision_path": job["revision_path"], "launch_run_root": job["launch_run_root"]}


def confirm_owner_success_contract(*, queue_root: str | Path, intent_id: str, owner: Mapping[str, Any],
                                   approved_by: str, approval_evidence: str, apply_ack: str,
                                   authorization_reference: str, ack: str,
                                   now: datetime | None = None) -> dict[str, Any]:
    """Write one confirmation under the intake lock, exclusively; an existing one is returned as is.

    The preview is recomputed under the lock and must still equal ``apply_ack``,
    so the record confirms exactly what the owner saw. Never overwrites.
    """

    from . import task_evaluation_scene_intake as intake

    _require(ack == ACK and isinstance(authorization_reference, str)
             and bool(authorization_reference.strip()), "not_authorized")
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = Path(queue_root)
    _safe(root)
    root = intake._root(root)
    with intake._lock(root):
        directory = root / intent_id
        _safe(directory)
        # A record must be readable by the service that compiles the team half (see the budget grant).
        _require(os.geteuid() == directory.stat().st_uid, "issuer_user_mismatch")
        planned = preview_owner_success_confirmation(queue_root=root, intent_id=intent_id, owner=owner,
                                                     approved_by=approved_by, approval_evidence=approval_evidence)
        _require(planned["apply_ack"] == apply_ack, "apply_ack_mismatch")
        intent = intake._read(directory / "intent.json", "intent_digest")
        preview = planned["preview"]
        bound = {field: preview[field] for field in _BINDING_FIELDS}
        existing = find_confirmation(directory, intent, bound)
        if existing is not None:
            return {"status": "already_confirmed", "record_path": str(directory / DIRECTORY / (
                existing["confirmation_digest"].removeprefix("sha256:") + ".json")), **existing}
        value = intake._seal({"schema_version": SCHEMA, "scope": SCOPE, **preview,
                              "authorization_reference": authorization_reference,
                              "issuer_channel": CHANNEL, "operator_uid": os.geteuid(),
                              "confirmed_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "thresholds_changed": False, "provider_mutation_performed": False},
                             "confirmation_digest")
        path = directory / DIRECTORY / (value["confirmation_digest"].removeprefix("sha256:") + ".json")
        path.parent.mkdir(mode=0o750, exist_ok=True)
        _safe(path.parent)
        intake.write_exclusive(path, value)
        _require(find_confirmation(directory, intent, bound) is not None, "record_invalid")
    return {"status": "confirmed", "record_path": str(path), **value}


__all__ = [
    "ACK",
    "AUTHORITY_PREFIX",
    "DIRECTORY",
    "SCHEMA",
    "binding",
    "configured_success_criteria",
    "confirm_owner_success_contract",
    "confirmations",
    "confirmed_owner_authority",
    "find_confirmation",
    "preview_owner_success_confirmation",
    "published_job",
]
