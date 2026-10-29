"""Project a bounded authoring receipt onto the existing preparation callback.

Callers provide the trusted running Pipeline commit, never a capture-supplied
version. This pure projection does not send, load credentials, reopen raw paths,
change job facts, or create execution authority.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest, cross_runtime_canonical_json
from .website_task_context_contracts import website_task_context_digest

SCHEMA = "website_family_authoring_status.v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_OWNERS = {"task_fact": "task_owner", "source_measurement": "scene_preparation",
           "runtime_integration": "runtime_operator"}


def _require(condition, reason):
    if not condition:
        raise ValueError("website_family_authoring_status_" + reason)


def _text(value, limit, reason):
    _require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit, reason)
    return value


def _digest(value, reason):
    _require(isinstance(value, str) and _SHA.fullmatch(value), reason)
    return value


def _requirement(row):
    _require(isinstance(row, Mapping) and isinstance(row.get("category"), str) and row["category"] in _OWNERS
        and row.get("owner") == _OWNERS[row["category"]]
        and row.get("status") == "required" and row.get("authority_created") is False,
        "requirement_invalid")
    value = {"requirement_id": _text(row.get("requirement_id"), 200, "requirement_id_invalid"),
             "category": row["category"], "owner": row["owner"], "status": "required",
             "question": _text(row.get("question"), 500, "question_invalid"), "authority_created": False}
    for key, limit in (("stage_id", 200), ("quantity", 200), ("unit", 48)):
        value[key] = _text(row[key], limit, key + "_invalid") if row.get(key) is not None else None
    return value


def project_family_authoring_status(*, preparation, task_context, pipeline_source_commit):
    """Return the source-bound, path-free payload for preparation-status v4.

    Status-sender integration belongs to the existing callback/outbox owner.
    Missing or oversized evidence refuses this projection; callers must retain
    that delivery gap without blocking or reclassifying scene preparation.
    """
    _require(isinstance(pipeline_source_commit, str) and _COMMIT.fullmatch(pipeline_source_commit),
             "pipeline_commit_required")
    _require(isinstance(task_context, Mapping)
        and task_context.get("schema_version") == "website_site_task_context.v1"
        and task_context.get("context_digest") == website_task_context_digest(task_context),
        "context_changed")
    _require(isinstance(preparation, Mapping)
        and preparation.get("schema_version") == "website_scene_preparation.v1"
        and preparation.get("digest") == canonical_digest(preparation, digest_field="digest"), "preparation_changed")
    inputs = preparation.get("task_evaluation_inputs") or {}
    _require(isinstance(inputs, Mapping) and isinstance(inputs.get("authoring"), Mapping), "receipt_required")
    request = inputs["authoring"].get("request")
    _require(isinstance(request, Mapping)
        and request.get("schema_version") == "website_task_family_authoring_requirements.v1"
        and request.get("request_digest") == canonical_digest(request, digest_field="request_digest")
        and request.get("claim_ceiling") == "development_only"
        and request.get("provider_mutation_performed") is False
        and request.get("automatic_paid_execution_authorized") is False, "receipt_changed")
    binding = request.get("binding") or {}
    prepared_binding = preparation.get("binding") or {}
    _require(isinstance(binding, Mapping) and isinstance(prepared_binding, Mapping), "binding_invalid")
    _require(binding.get("task_context_digest") == task_context["context_digest"], "context_binding_changed")
    for key in ("task_context_digest", "source_geometry_digest", "task_masks_digest"):
        _require(binding.get(key) == prepared_binding.get(key), "source_binding_changed")
        _digest(binding.get(key), key + "_required")
    video = _digest(binding.get("source_video_digest"), "source_video_required")
    operations = request.get("operations")
    targets = request.get("source_targets")
    requirements = request.get("requirements")
    _require(isinstance(operations, list) and len(operations) <= 8, "operation_limit")
    _require(isinstance(targets, list) and len(targets) <= 32, "target_limit")
    _require(isinstance(requirements, list) and len(requirements) <= 64, "requirement_limit")
    projected_operations = []
    for order, row in enumerate(operations, 1):
        _require(isinstance(row, Mapping), "operation_invalid")
        source = row.get("source_operation") or {}
        _require(isinstance(source, Mapping) and row.get("order") == order, "operation_order_invalid")
        _require(len(operations) == 1 or bool(source.get("task_quote")), "operation_quote_required")
        quote = source.get("task_quote") or row.get("description") or task_context.get("description")
        projected_operations.append({"operation_id": _text(row.get("stage_id"), 200, "operation_id_invalid"),
            "order": order, "task_family": _text(row.get("task_family"), 48, "operation_family_invalid"),
            "source_quote": _text(quote, 1000, "operation_quote_invalid")})
    projected_targets = []
    for row in targets:
        _require(isinstance(row, Mapping), "target_invalid")
        target = _text(row.get("target_id"), 200, "target_id_invalid")
        projected_targets.append({"target_id": target,
            "label": _text(row.get("semantic_label") or target, 200, "target_label_invalid")})
    for rows, key in ((projected_operations, "operation_id"), (projected_targets, "target_id")):
        _require(len({row[key] for row in rows}) == len(rows), key + "_duplicate")
    projected_requirements = [_requirement(row) for row in requirements]
    _require(len({row["requirement_id"] for row in projected_requirements}) == len(requirements), "requirement_id_duplicate")
    next_action = request.get("next_action")
    _require(next_action is None or next_action in requirements, "next_action_changed")
    next_action = projected_requirements[requirements.index(next_action)] if next_action is not None else None
    result = {"schema_version": SCHEMA, "pipeline_source_commit": pipeline_source_commit,
        **{key: _text(task_context.get(key), 200, key + "_required") for key in ("request_id", "scene_id", "capture_id")},
        **{key: binding[key] for key in ("task_context_digest", "source_geometry_digest", "task_masks_digest")},
        "source_video_digest": video, "authoring_request_digest": request["request_digest"],
        "task_description_digest": cross_runtime_canonical_digest({"description": task_context["description"]}),
        "task_success_criteria_digest": cross_runtime_canonical_digest({"success_criteria": task_context.get("success_criteria")}),
        "operations": projected_operations, "source_targets": projected_targets,
        "requirements": projected_requirements, "next_action": next_action,
        "claim_ceiling": "development_only", "automatic_paid_execution_authorized": False}
    result["projection_digest"] = cross_runtime_canonical_digest(result, digest_field="projection_digest")
    _require(len(cross_runtime_canonical_json(result).encode("utf-8")) <= 48 * 1024, "payload_limit")
    return result
