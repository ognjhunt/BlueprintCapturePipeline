"""Read the website's confirmed task immediately before scene preparation.

The original upload manifest is never rewritten to impersonate a later
confirmation. The authenticated snapshot is retained as derived run evidence.
"""
from __future__ import annotations

import json
import hmac
import hashlib
import os
import re
from collections.abc import Callable, Mapping
from typing import Any
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .safe_outbound_http import pinned_api_policy
from .safe_outbound_http import request as safe_request
from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url
from .website_task_context_contracts import validate_website_task_context as validate_website_task_context


def load_current_website_task_context(
    *, request_id: str, scene_id: str, capture_id: str,
    purpose: str | None = None,
) -> dict[str, Any]:
    """One authenticated bounded read; errors hold preparation without fallback."""
    payload = {"request_id": request_id, "scene_id": scene_id}
    if purpose is not None:
        if purpose != "scene_preparation":
            raise ValueError("website_task_context_purpose_invalid")
        payload["purpose"] = purpose
    value = website_webapp_request(capture_id=capture_id, operation="task-context", payload=payload)
    return validate_website_task_context(value, request_id=request_id,
                                         scene_id=scene_id, capture_id=capture_id, purpose=purpose)


def website_webapp_request(*, capture_id: str, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Use the existing signed transport for private preparation control data."""
    if operation in {"prepared-scene", "preparation-spend", "task-item-evidence", "atlas-preview"} or (
            operation in {"scene-sponsorship", "atlas-configuration"} and payload.get("create") is not False):
        from .core.handoff_lease import handoff_lease_checkpoint
        handoff_lease_checkpoint("website_control_effect:" + operation)
    if operation not in {"task-context", "scene-sponsorship", "prepared-scene",
                         "preparation-spend", "preparation-settlement", "atlas-configuration", "atlas-preview",
                         "agent-execution-offer", "task-item-evidence", "preparation-status",
                         "task-target-choice"}:
        raise ValueError("website_control_operation_invalid")
    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        raise ValueError("website_task_context_webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    token = load_pipeline_sync_token()
    body = json.dumps(payload, separators=(",", ":")).encode()
    # The noncreating Atlas read can include 128 preparation reservations and
    # 64 capture inference calls. Keep that response finite without widening
    # ordinary configuration, authority or other control responses.
    response_limit = 200_000 if operation == "atlas-configuration" and payload.get("create") is False else 100_000
    try:
        response = safe_request(
            f"{origin}/api/internal/pipeline/creator-captures/{quote(capture_id, safe='')}/{operation}",
            method="POST", data=body, headers=_pipeline_sync_headers(token, body),
            timeout_seconds=10, policy=pinned_api_policy(origin, max_response_bytes=response_limit),
            max_response_bytes=response_limit,
        )
    except HTTPError as exc:
        # Preserve the owning service's typed refusal, never its raw body or
        # request headers. In particular, an exhausted request allowance must
        # not look like an unexplained model/provider failure.
        code = None
        try:
            failure = json.loads(exc.read(4096))
            if isinstance(failure, dict):
                code = failure.get("code")
        except (ValueError, OSError):
            pass
        finally:
            exc.close()
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,99}", code):
            code = "unclassified"
        raise ValueError(f"website_control_{operation}_http_{exc.code}:{code}") from None
    value = json.loads(response.body)
    if not isinstance(value, Mapping):
        raise ValueError("website_task_context_response_invalid")
    return dict(value)


def reserve_website_sam_spend(*, task_context: Mapping[str, Any], binding_digest: str,
                            maximum_cost_usd: float, request_count: int, source_video_digest: str | None = None) -> tuple[dict[str, Any], Any]:
    return reserve_website_preparation_spend(task_context=task_context, binding_digest=binding_digest,
        maximum_cost_usd=maximum_cost_usd, request_count=request_count,
        resource_class="evaluator_api", provider="meta", source_video_digest=source_video_digest,
        atlas_auxiliary=source_video_digest is not None)


def reserve_website_preparation_spend(*, task_context: Mapping[str, Any], binding_digest: str,
                                     maximum_cost_usd: float, request_count: int,
                                     resource_class: str, provider: str,
                                     source_video_digest: str | None = None,
                                     atlas_auxiliary: bool = False,
                                     retained_admission: Mapping[str, Any] | None = None,
                                     unallocated_continuation_path: str | Path | None = None,
                                     retain_admission: Callable[[Mapping[str, Any]], None] | None = None) -> tuple[dict[str, Any], Any]:
    """The controller obtains an exact, one-dispatch grant from the shared cap."""
    import time

    from .paid_resource_admission import require_paid_resource_admission, require_paid_resource_budget
    if (resource_class, provider) not in {("evaluator_api", "meta"), ("evaluator_api", "google"), ("openai_api_candidate", "openai"),
                                         ("provider_reconstruction_api", "world_labs_atlas"), ("gpu_render", "vast")}:
        raise ValueError("website_preparation_provider_resource_mismatch")
    command = {"task_context_digest": task_context["context_digest"], "allocation_binding_digest": binding_digest,
               "resource_class": resource_class, "provider": provider, "maximum_cost_usd": maximum_cost_usd,
               "request_count": request_count}
    if source_video_digest is not None:
        command["source_video_digest"] = source_video_digest
    if atlas_auxiliary:
        if provider not in {"meta", "google"} or source_video_digest is None:
            raise ValueError("website_atlas_auxiliary_binding_invalid")
        command["atlas_auxiliary"] = True
    # Avoid an orphaned remote hold when the local monitor already refuses.
    # A report can still expire during the RPC, so this is only a precheck.
    require_paid_resource_budget(resource_class=resource_class)
    operator_file = os.getenv("BLUEPRINT_ATLAS_OPERATOR_ADMISSION_FILE", "").strip()
    if operator_file:
        value = load_atlas_operator_admission(Path(operator_file), command=command, task_context=task_context)
    else:
        value = website_webapp_request(capture_id=task_context["capture_id"], operation="preparation-spend",
            payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"], "spend": command})
    if (any(value.get(key) != expected for key, expected in command.items())
            or value.get("external_disclosure_allowed") is not True
            or not isinstance(value.get("expires_at_epoch"), (int, float)) or value["expires_at_epoch"] <= time.time()):
        raise ValueError("website_preparation_spend_receipt_invalid")
    if value.get("status") == "already_reserved" and retained_admission is not None:
        # Reuse requires the original authenticated admission. Atlas retains
        # its upload/submission intent; GPU recovery additionally proves that
        # the previous dispatch never called the provider or allocated a worker.
        retained = {key: item for key, item in retained_admission.items() if key != "source_commit"}
        if retained != {**value, "status": "admitted"}:
            raise ValueError("website_preparation_retained_admission_mismatch")
        if (resource_class, provider) == ("gpu_render", "vast") and unallocated_continuation_path is not None:
            from .policy_canary_unallocated_resume import validate_unallocated_funding_reuse
            validate_unallocated_funding_reuse(unallocated_continuation_path, funding_admission=retained)
        elif (resource_class, provider) != ("provider_reconstruction_api", "world_labs_atlas"):
            raise ValueError("website_preparation_retained_admission_mismatch")
        value = retained
    if (retain_admission is not None and value.get("status") == "admitted"
            and value.get("schema_version") == "paid_lane_admission.v1"
            and value.get("blockers") in ([], None)):
        # Preserve the original authenticated transaction before rechecking
        # local budget. A restart must never turn a bare already_reserved
        # answer into authority or overwrite the original receipt with it.
        retain_admission(dict(value))
    grant = require_paid_resource_admission(value, resource_class=resource_class, expected_schema_version="paid_lane_admission.v1")
    return value, grant


def load_atlas_operator_admission(path: Path, *, command: Mapping[str, Any],
                                  task_context: Mapping[str, Any]) -> dict[str, Any]:
    """A short-lived signed handoff from the actual WebApp reservation transaction.

    The operator runs WebApp's committed utility; this never invents authority
    or reserves locally. The same canonical admission check still issues the
    in-process grant, and the same allocator/input/release gates still apply.
    """
    import time
    if command.get("provider") != "world_labs_atlas" or command.get("resource_class") != "provider_reconstruction_api":
        raise ValueError("atlas_operator_admission_provider_invalid")
    value = json.loads(path.read_text())
    signature = value.pop("signature", None)
    expected = hmac.new(load_pipeline_sync_token().encode(), cross_runtime_canonical_digest(value).encode(), hashlib.sha256).hexdigest()
    admission = value.get("admission")
    if (not isinstance(signature, str) or not hmac.compare_digest(signature, expected)
            or value.get("schema_version") != "website_atlas_operator_admission.v1"
            or value.get("request_id") != task_context["request_id"]
            or value.get("capture_id") != task_context["capture_id"]
            or not isinstance(value.get("expires_at_epoch"), (int, float))
            or value["expires_at_epoch"] <= time.time()
            or not isinstance(value.get("issued_at_epoch"), (int, float))
            or value["expires_at_epoch"] - value["issued_at_epoch"] > 600
            or not isinstance(admission, Mapping)
            or any(admission.get(k) != v for k, v in command.items())
            or admission.get("status") not in {"admitted", "already_reserved"}):
        raise ValueError("atlas_operator_admission_invalid")
    return dict(admission)


def load_website_scene_sponsorship(*, task_context: Mapping[str, Any], now: float, create: bool = True) -> dict[str, Any]:
    if type(create) is not bool:
        raise ValueError("website_scene_sponsorship_create_invalid")
    payload = {"request_id": task_context["request_id"], "scene_id": task_context["scene_id"]}
    if task_context.get("purpose") == "scene_preparation":
        validate_website_task_context(task_context, request_id=task_context["request_id"],
            scene_id=task_context["scene_id"], capture_id=task_context["capture_id"], purpose="scene_preparation")
        payload.update(purpose="scene_preparation", expected_task_context_digest=task_context["context_digest"])
        if create is False:
            payload["create"] = False
    elif create is not True:
        raise ValueError("website_scene_sponsorship_read_purpose_required")
    value = website_webapp_request(capture_id=task_context["capture_id"], operation="scene-sponsorship",
        payload=payload)
    from math import isfinite
    issued = (value.get("consent") or {}).get("accepted_at_epoch")
    authoring_provider = value.get("authoring_provider", "openai")
    anthropic_terms = value.get("anthropic_provider_terms_reference")
    authoring_runtime = value.get("authoring_agent_runtime")
    agents_policy = value.get("agents_api_policy")
    if (authoring_provider not in {"openai", "anthropic"}
            or (authoring_provider == "openai" and anthropic_terms is not None)
            or (authoring_provider == "anthropic" and (
                not isinstance(anthropic_terms, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", anthropic_terms) is None))):
        raise ValueError("website_scene_sponsorship_authoring_provider_invalid")
    if authoring_runtime is not None or value.get("authoring_model") is not None or agents_policy is not None:
        if (authoring_runtime != "openai_agents_api" or authoring_provider != "openai"
                or value.get("authoring_model") != "gpt-6.1-sol"
                or not isinstance(agents_policy, Mapping)
                or agents_policy.get("schema_version") != "scene_configuration_agents_api_policy.v1"
                or agents_policy.get("disclosure_scope") != "task_asset_source_frames_and_metric_envelope"
                or agents_policy.get("session_retention") != "until_deleted"
                or agents_policy.get("trace_retention") != "provider_default"
                or agents_policy.get("region") != "us"
                or agents_policy.get("budget_policy") != "project_guard_accepted_uncertainty"
                or re.fullmatch(r"sha256:[0-9a-f]{64}",
                                str(agents_policy.get("project_guard_receipt_digest") or "")) is None
                or type(agents_policy.get("ttl_seconds")) is not int
                or not 60 <= agents_policy["ttl_seconds"] <= 1800
                or type(agents_policy.get("maximum_review_cycles")) is not int
                or not 1 <= agents_policy["maximum_review_cycles"] <= 3):
            raise ValueError("website_scene_sponsorship_agents_api_policy_invalid")
    amounts = [value.get(key) for key in ("preparation_max_total_spend_usd", "upstream_max_spend_usd", "max_total_spend_usd")]
    if (value.get("schema_version") != "website_scene_sponsorship.v1" or value.get("sponsor") != "blueprint"
            or value.get("authority_digest") != canonical_digest(value, digest_field="authority_digest")
            or any(value.get(key) != task_context[key] for key in ("request_id", "scene_id", "capture_id"))
            or value.get("task_context_digest") != task_context["context_digest"]
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not isfinite(x) or x <= 0 for x in amounts)
            or amounts[1] + amounts[2] > amounts[0]
            or not isinstance(value.get("expires_at_epoch"), (int, float))
            or isinstance(issued, bool) or not isinstance(issued, (int, float)) or not isfinite(issued)
            or issued > now + 60
            or not now < value["expires_at_epoch"]
            or not 0 < value["expires_at_epoch"] - issued <= 86400):
        raise ValueError("website_scene_sponsorship_binding_invalid")
    if task_context.get("purpose") == "scene_preparation":
        from .website_preparation_contracts import validate_preparation_authority
        validate_preparation_authority(task_context=task_context, authority=value, now=now)
    return value


# The WebApp's typed refusal of a create=False read once the retained grant's
# window has passed. It is answered only after the WebApp has checked the capture
# binding, the assessment proposal, the expected context digest and the retained
# grant's digest, policy, purpose and context.
_SPONSORSHIP_EXPIRED = "website_control_scene-sponsorship_http_409:consent_expired"
_SPONSORSHIP_MISSING = "website_control_scene-sponsorship_http_409:website_scene_sponsorship_missing"


def website_scene_sponsorship_standing(*, task_context: Mapping[str, Any], now: float,
                                       allow_missing: bool = False) -> str:
    """Status evidence only: "current", "expired" or, when allowed, "missing";
    any other answer raises.

    The read never creates or renews a grant, and its answer authorizes nothing.
    """
    try:
        load_website_scene_sponsorship(task_context=task_context, now=now, create=False)
    except ValueError as exc:
        if str(exc) == _SPONSORSHIP_EXPIRED:
            return "expired"
        if allow_missing and str(exc) == _SPONSORSHIP_MISSING:
            return "missing"
        raise
    return "current"


def enqueue_website_prepared_scene(*, task_context: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    value = website_webapp_request(capture_id=task_context["capture_id"], operation="prepared-scene",
        payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"], "request": request})
    if (value.get("request_digest") != cross_runtime_canonical_digest(request) or value.get("state") != "forward_pending"
            or not isinstance(value.get("id"), str) or not value["id"].startswith("scene-")):
        raise ValueError("website_scene_outbox_receipt_invalid")
    return value
