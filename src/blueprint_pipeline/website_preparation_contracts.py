"""Pure preparation scope and proposal data checks; never execution authority.

ADP-010/day14 compatibility: read-only controls may inspect these structures
without reaching intake, current-grant reopening, workers, or paid dispatch.
"""

from collections.abc import Mapping
import math
import re

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .website_task_context_contracts import validate_website_task_context

from .task_evaluation_scene_execution_scope import scene_preparation_only


def _require(condition):
    if not condition:
        raise ValueError("website_assessment_resume_binding_invalid")


def preparation_consent_valid(request):
    """Structural scope only; consequential consumers must reopen authority."""
    consent = request.get("consent") or {}
    task = request.get("task") or {}
    return (
        scene_preparation_only(request)
        and request["execution"].get("policy_candidates") == []
        and "robot_binding_id" not in task
        and "evaluation_source" not in task
        and consent.get("task_confirmed") is False
    )


def validate_preparation_proposal(authority, context):
    proposal = authority.get("assessment_preparation_proposal")
    _require(type(proposal) is dict and set(proposal) == {
        "schema_version", "request_id", "capture_id", "job_id", "run_id", "source_key", "context_digest",
        "packet_sha256", "questions_pending", "scope", "robot_suitability_verified", "physical_trial_authorized"})
    _require(proposal["schema_version"] == "site_assessment_preparation_proposal.v1"
        and proposal["request_id"] == context["request_id"] and proposal["capture_id"] == context["capture_id"]
        and proposal["scope"] == "scene_preparation_only" and proposal["robot_suitability_verified"] is False
        and proposal["physical_trial_authorized"] is False and type(proposal["questions_pending"]) is bool)
    _require(re.fullmatch(r"advisory-[a-f0-9]{64}", str(proposal["job_id"])) is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", str(proposal["run_id"])) is not None
        and re.fullmatch(r"sha256:[a-f0-9]{64}", str(proposal["source_key"])) is not None
        and all(re.fullmatch(r"[a-f0-9]{64}", str(proposal[key])) is not None
                for key in ("context_digest", "packet_sha256")))
    _require(all(type(proposal[key]) is str for key in ("job_id", "run_id", "source_key", "context_digest", "packet_sha256")))


def validate_preparation_authority(*, task_context, authority, now):
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
    validate_preparation_proposal(authority, task_context)
    return dict(authority)


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


