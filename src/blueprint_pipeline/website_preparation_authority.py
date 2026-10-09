"""Bounded website preparation evidence; never robot evaluation permission.

These pure checks validate retained or signed evidence. A context or caller-
supplied grant does not itself confer authenticated issuance or paid admission.
"""

from collections.abc import Mapping
import math
from time import time as current_time

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


def validate_preparation_authority(*, task_context, authority, now):
    from .website_task_context import validate_website_task_context
    from .website_assessment_resume import _proposal_admitted

    validate_website_task_context(
        task_context,
        request_id=task_context["request_id"],
        scene_id=task_context["scene_id"],
        capture_id=task_context["capture_id"],
        purpose="scene_preparation",
    )
    rights = task_context.get("capture_rights") or {}
    owner = authority.get("owner") or {}
    consent = authority.get("consent") or {}
    issued = consent.get("accepted_at_epoch")
    expiry = authority.get("expires_at_epoch")
    limits = [
        authority.get(k)
        for k in (
            "preparation_max_total_spend_usd",
            "upstream_max_spend_usd",
            "max_total_spend_usd",
        )
    ]
    if (
        authority.get("schema_version") != "website_scene_sponsorship.v1"
        or authority.get("sponsor") != "blueprint"
        or authority.get("purpose") != "scene_preparation"
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or any(
            authority.get(k) != task_context[k] for k in ("request_id", "scene_id", "capture_id")
        )
        or authority.get("task_context_digest") != task_context["context_digest"]
        or rights.get("derived_scene_generation_allowed") is not True
        or rights.get("consent_revoked") is not False
        or not isinstance(owner, Mapping)
        or set(owner) != {"user_id", "organization_id"}
        or any(not isinstance(v, str) or not v.strip() for v in owner.values())
        or consent.get("accepted_by") != owner["user_id"]
        or consent.get("task_confirmed") is not task_context["confirmed"]
        or consent.get("private_processing_authorized") is not True
        or consent.get("provider_training_authorized") is not False
        or consent.get("spend_authorized") is not True
        or consent.get("rights_reference") != cross_runtime_canonical_digest(rights)
        or not isinstance(consent.get("provider_terms_reference"), str)
        or not consent["provider_terms_reference"]
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in limits
        )
        or limits[1] + limits[2] > limits[0]
        or type(authority.get("max_paid_attempts")) is not int
        or not 1 <= authority["max_paid_attempts"] <= 32
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (issued, expiry, now)
        )
        or issued > now + 60
        or not now < expiry
        or not 0 < expiry - issued <= 86400
    ):
        raise ValueError("website_preparation_authority_invalid")
    _proposal_admitted(authority, task_context)
    return dict(authority)


def preparation_consent_valid(request):
    """Structural scope only; consequential consumers must reopen authority."""
    from .task_evaluation_scene_execution_scope import scene_preparation_only

    consent = request.get("consent") or {}
    task = request.get("task") or {}
    return (
        scene_preparation_only(request)
        and request["execution"].get("policy_candidates") == []
        and "robot_binding_id" not in task
        and "evaluation_source" not in task
        and consent.get("task_confirmed") is False
    )


def validate_preparation_request(*, request, task_context, authority, now):
    validate_preparation_authority(task_context=task_context, authority=authority, now=now)
    execution = request.get("execution") or {}
    source = request.get("source") or {}
    expected_providers = ["vast", "openai"]
    if authority.get("authoring_provider") == "anthropic":
        expected_providers.append("anthropic")
    if (
        not preparation_consent_valid(request)
        or request.get("submission_id") != task_context["capture_id"]
        or request.get("owner") != authority["owner"]
        or request.get("consent") != authority["consent"]
        or source.get("kind") != "gaussian_splat"
        or not isinstance(source.get("content_digest"), str)
        or source.get("binding_id") != "website-splat-" + source["content_digest"][7:39]
        or request.get("task", {}).get("subject", {}).get("geometry_origin")
        != "removed_before_reconstruction"
        or request.get("task", {}).get("task_id")
        != "website-" + task_context["context_digest"][7:27]
        or any(
            execution.get(k) != authority[k]
            for k in ("max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")
        )
        or execution.get("max_retries") != 0
        or execution.get("claim_scope") != "development_only"
        or execution.get("allowed_providers") != expected_providers
    ):
        raise ValueError("website_preparation_request_authority_invalid")
    return dict(authority)


def require_retained_preparation_authority(*, request, queue_root, now):
    """Reopen server-owned source references; context-only consent is insufficient.

    Signed Web intake and current source registration are both required. The
    read-only current grant check cannot create or renew an allowance.
    """
    from .website_scene_dispatch import binding_root, _index_path
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .website_task_context import (
        load_current_website_task_context,
        load_website_scene_sponsorship,
    )

    root = binding_root({"intent_root": str(queue_root)})
    path = _index_path(root, request)
    if not path.is_file() or path.is_symlink():
        raise ValueError("website_preparation_source_missing")
    registration = read(path, digest_field="registration_digest")
    if (
        registration.get("schema_version") != "website_scene_source_registration.v1"
        or registration.get("request_digest") != cross_runtime_canonical_digest(request)
        or registration.get("execution_authority_granted") is not False
    ):
        raise ValueError("website_preparation_source_invalid")
    refs = registration["references"]
    if not {"preparation", "runtime_inputs", "task_context"}.issubset(refs):
        raise ValueError("website_preparation_source_references_missing")
    for ref in refs.values():
        checked_file(ref["path"], ref)
    prepared = read(
        checked_file(refs["preparation"]["path"], refs["preparation"]), digest_field="digest"
    )
    context = read(
        checked_file(refs["task_context"]["path"], refs["task_context"]),
        digest_field="context_digest",
    )
    if (
        prepared.get("intake_request") != request
        or prepared.get("binding", {}).get("task_context_digest") != context["context_digest"]
    ):
        raise ValueError("website_preparation_source_request_changed")
    authority = validate_preparation_request(
        request=request,
        task_context=context,
        authority=prepared.get("website_preparation_authority") or {},
        now=now,
    )
    current = load_current_website_task_context(
        request_id=context["request_id"],
        scene_id=context["scene_id"],
        capture_id=context["capture_id"],
        purpose="scene_preparation",
    )
    if current != context:
        raise ValueError("website_preparation_source_context_changed")
    fresh = load_website_scene_sponsorship(task_context=current, now=now, create=False)
    if fresh != authority:
        raise ValueError("website_preparation_source_authority_changed")
    # Signed reads may cross the expiry boundary. Recheck the actual clock
    # and return it for the caller's effective-window/action checks.
    moment = current_time()
    validate_preparation_request(request=request, task_context=current, authority=fresh, now=moment)
    return moment
