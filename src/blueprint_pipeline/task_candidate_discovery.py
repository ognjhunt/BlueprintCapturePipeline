"""Deterministic task discovery and explicit customer-intent approval.

Task proposals are hypotheses grounded in capture observations.  Confidence does
not make a proposal customer intent: every inferred candidate remains blocked
until a customer or authorized operator records an immutable approval decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    canonical_digest,
    canonical_json,
)


DISCOVERY_SCHEMA_VERSION = "task_candidate_discovery.v1"
DECISION_SCHEMA_VERSION = "task_candidate_decision.v1"
APPROVED_TASK_SCHEMA_VERSION = "approved_task_definition.v1"

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_APPROVAL_ACTIONS = {"approve", "edit_and_approve", "reject", "request_more_capture"}
_APPROVER_ROLES = {"customer", "operator"}
_MODEL_PROPOSAL_ORIGINS = {"model", "provider", "model_provider"}
_PROPOSAL_ORIGINS = _MODEL_PROPOSAL_ORIGINS | {"local_rule"}
_SECRET_KEYS = {"api_key", "authorization", "credential", "credentials", "password", "secret", "token"}


class TaskCandidateContractError(ValueError):
    """Fail-closed error with stable, sorted identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TaskCandidateContractError(["artifact:not_json_serializable"]) from exc


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({_text(item) for item in value if _text(item)})


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_clone(row) for row in value if isinstance(row, Mapping)]


def _is_digest(value: Any) -> bool:
    return bool(_SHA256.fullmatch(_text(value)))


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _secret_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            if (
                lowered in _SECRET_KEYS
                or any(lowered.endswith(f"_{suffix}") for suffix in _SECRET_KEYS)
            ) and nested not in (None, "", [], {}):
                paths.append(path)
            paths.extend(_secret_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_secret_paths(nested, prefix=f"{prefix}[{index}]"))
    return paths


def _reject_secrets(value: Any) -> None:
    paths = _secret_paths(value)
    if paths:
        raise TaskCandidateContractError(
            [f"secret_value_forbidden:{path}" for path in paths]
        )


def _normalize_grounded_rows(
    value: Any,
    *,
    row_kind: str,
    required_description: bool = True,
) -> list[dict[str, Any]]:
    rows = _rows(value)
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    identifier_key = {
        "observed_site_facts": "fact_id",
        "inferred_objects_and_affordances": "inference_id",
        "unsupported_or_occluded_regions": "region_id",
        "hazards": "hazard_id",
        "privacy_sensitive_areas": "area_id",
    }[row_kind]
    for index, row in enumerate(rows):
        description = _text(row.get("description"))
        if required_description and not description:
            errors.append(f"{row_kind}[{index}].description:missing")
        if not _text(row.get(identifier_key)):
            errors.append(f"{row_kind}[{index}].{identifier_key}:missing")
        confidence = row.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append(f"{row_kind}[{index}].confidence:invalid")
            confidence = 0.0
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{row_kind}[{index}].confidence:invalid")
        item = {
            key: _clone(nested)
            for key, nested in row.items()
            if key not in {"row_digest", "confidence", "supporting_frames", "supporting_3d_regions"}
        }
        item["description"] = description
        item["confidence"] = round(float(confidence), 6)
        item["supporting_frames"] = _strings(row.get("supporting_frames"))
        item["supporting_3d_regions"] = _strings(row.get("supporting_3d_regions"))
        if row_kind == "observed_site_facts":
            item["observation_status"] = "directly_observed"
        elif row_kind == "inferred_objects_and_affordances":
            item["observation_status"] = "inferred"
        else:
            status = _text(row.get("observation_status")) or "directly_observed"
            if status not in {"directly_observed", "inferred"}:
                errors.append(f"{row_kind}[{index}].observation_status:unsupported")
            item["observation_status"] = status
        item["row_digest"] = canonical_digest(item, digest_field="row_digest")
        normalized.append(item)
    if errors:
        raise TaskCandidateContractError(errors)
    return sorted(normalized, key=lambda row: (row["row_digest"], row["description"]))


def _normalize_success_condition(value: Any, *, prefix: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskCandidateContractError([f"{prefix}:missing_or_invalid"])
    condition = _clone(value)
    errors: list[str] = []
    for key in ("metric", "operator", "units"):
        if not _text(condition.get(key)):
            errors.append(f"{prefix}.{key}:missing")
    if "threshold" not in condition:
        errors.append(f"{prefix}.threshold:missing")
    if errors:
        raise TaskCandidateContractError(errors)
    return condition


def _normalize_candidate(
    proposal: Mapping[str, Any],
    *,
    discovery_identity: Mapping[str, Any],
    proposal_method: Mapping[str, Any],
    observed_fact_ids: set[str],
    index: int,
) -> dict[str, Any]:
    errors: list[str] = []
    description = _text(proposal.get("description"))
    task_family = _text(proposal.get("likely_task_family"))
    reset = _text(proposal.get("required_site_reset"))
    for key, text in (
        ("description", description),
        ("likely_task_family", task_family),
        ("required_site_reset", reset),
    ):
        if not text:
            errors.append(f"candidate_proposals[{index}].{key}:missing")
    confidence = proposal.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"candidate_proposals[{index}].confidence:invalid")
        confidence = 0.0
    coverage = proposal.get("coverage")
    if not isinstance(coverage, Mapping) or not coverage:
        errors.append(f"candidate_proposals[{index}].coverage:missing_or_empty")
        coverage = {}
    observed_objects = sorted(
        _rows(proposal.get("observed_objects")), key=canonical_json
    )
    target_regions = sorted(_rows(proposal.get("target_regions")), key=canonical_json)
    if not observed_objects:
        errors.append(f"candidate_proposals[{index}].observed_objects:missing_or_invalid")
    for object_index, observed_object in enumerate(observed_objects):
        for key in ("object_id", "label"):
            if not _text(observed_object.get(key)):
                errors.append(
                    f"candidate_proposals[{index}].observed_objects[{object_index}].{key}:missing"
                )
        fact_ids = set(_strings(observed_object.get("observation_fact_ids")))
        if not fact_ids:
            errors.append(
                f"candidate_proposals[{index}].observed_objects[{object_index}].observation_fact_ids:missing"
            )
        elif not fact_ids.issubset(observed_fact_ids):
            errors.append(
                f"candidate_proposals[{index}].observed_objects[{object_index}].observation_fact_ids:unknown"
            )
    if not target_regions:
        errors.append(f"candidate_proposals[{index}].target_regions:missing_or_invalid")
    for region_index, target_region in enumerate(target_regions):
        for key in ("region_id", "label"):
            if not _text(target_region.get(key)):
                errors.append(
                    f"candidate_proposals[{index}].target_regions[{region_index}].{key}:missing"
                )
    required_capabilities = _strings(proposal.get("required_robot_capabilities"))
    if not required_capabilities:
        errors.append(f"candidate_proposals[{index}].required_robot_capabilities:missing")
    supporting_frames = _strings(proposal.get("supporting_frames"))
    supporting_regions = _strings(proposal.get("supporting_3d_regions"))
    if not supporting_frames and not supporting_regions:
        errors.append(f"candidate_proposals[{index}].supporting_evidence:missing")
    if errors:
        raise TaskCandidateContractError(errors)
    condition = _normalize_success_condition(
        proposal.get("proposed_measurable_success_condition"),
        prefix=f"candidate_proposals[{index}].proposed_measurable_success_condition",
    )
    estimated_cost = proposal.get("estimated_evaluation_cost_usd")
    if isinstance(estimated_cost, bool) or not isinstance(estimated_cost, (int, float)) or float(estimated_cost) < 0:
        raise TaskCandidateContractError(
            [f"candidate_proposals[{index}].estimated_evaluation_cost_usd:invalid"]
        )
    expected_value = proposal.get("expected_customer_value")
    if expected_value is not None and (
        not isinstance(expected_value, Mapping)
        or _text(expected_value.get("source")) != "customer"
    ):
        raise TaskCandidateContractError(
            [f"candidate_proposals[{index}].expected_customer_value:not_customer_supplied"]
        )
    candidate = {
        "description": description,
        "observed_objects": observed_objects,
        "target_regions": target_regions,
        "required_robot_capabilities": required_capabilities,
        "likely_task_family": task_family,
        "proposed_measurable_success_condition": condition,
        "required_site_reset": reset,
        "supporting_frames": supporting_frames,
        "supporting_3d_regions": supporting_regions,
        "confidence": round(float(confidence), 6),
        "coverage": _clone(coverage),
        "assumptions": _strings(proposal.get("assumptions")),
        "missing_evidence": _strings(proposal.get("missing_evidence")),
        "prohibited_claims": _strings(proposal.get("prohibited_claims")),
        "estimated_evaluation_cost_usd": round(float(estimated_cost), 6),
        "expected_customer_value": _clone(expected_value),
        "proposal_method": _clone(proposal_method),
        "approval_status": "approval_required",
    }
    identity = {**discovery_identity, "candidate": candidate}
    candidate["task_candidate_id"] = _stable_id("task-candidate", identity)
    candidate["candidate_digest"] = canonical_digest(candidate, digest_field="candidate_digest")
    return candidate


def build_task_candidate_discovery(
    *,
    discovery_id: str,
    source_capture: Mapping[str, Any],
    capture_qa_report_digest: str,
    scene_analysis: Mapping[str, Any],
    candidate_proposals: Sequence[Mapping[str, Any]],
    proposal_method: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic discovery artifact without approving intent."""

    errors: list[str] = []
    if not _IDENTIFIER.fullmatch(_text(discovery_id)):
        errors.append("discovery_id:invalid")
    if not isinstance(source_capture, Mapping):
        errors.append("source_capture:missing")
        source_capture = {}
    for key in ("intake_id", "capture_authority_profile"):
        if not _text(source_capture.get(key)):
            errors.append(f"source_capture.{key}:missing")
    if not _is_digest(source_capture.get("capture_digest")):
        errors.append("source_capture.capture_digest:invalid_sha256")
    if not _is_digest(capture_qa_report_digest):
        errors.append("capture_qa_report_digest:invalid_sha256")
    if not isinstance(scene_analysis, Mapping):
        errors.append("scene_analysis:missing")
        scene_analysis = {}
    for key in (
        "observed_site_facts",
        "inferred_objects_and_affordances",
        "unsupported_or_occluded_regions",
        "hazards",
        "privacy_sensitive_areas",
    ):
        if not isinstance(scene_analysis.get(key), list):
            errors.append(f"scene_analysis.{key}:must_be_list")
    if not isinstance(proposal_method, Mapping):
        errors.append("proposal_method:missing")
        proposal_method = {}
    for key in ("method_id", "version", "implementation_digest", "proposer_identity", "origin"):
        if not _text(proposal_method.get(key)):
            errors.append(f"proposal_method.{key}:missing")
    if proposal_method.get("implementation_digest") and not _is_digest(
        proposal_method.get("implementation_digest")
    ):
        errors.append("proposal_method.implementation_digest:invalid_sha256")
    if proposal_method.get("origin") and proposal_method.get("origin") not in _PROPOSAL_ORIGINS:
        errors.append("proposal_method.origin:unsupported")
    if not isinstance(candidate_proposals, (list, tuple)) or not all(
        isinstance(proposal, Mapping) for proposal in candidate_proposals
    ):
        errors.append("candidate_proposals:must_be_mapping_list")
    if errors:
        raise TaskCandidateContractError(errors)

    analysis = {
        "observed_site_facts": _normalize_grounded_rows(
            scene_analysis.get("observed_site_facts"), row_kind="observed_site_facts"
        ),
        "inferred_objects_and_affordances": _normalize_grounded_rows(
            scene_analysis.get("inferred_objects_and_affordances"),
            row_kind="inferred_objects_and_affordances",
        ),
        "unsupported_or_occluded_regions": _normalize_grounded_rows(
            scene_analysis.get("unsupported_or_occluded_regions"),
            row_kind="unsupported_or_occluded_regions",
        ),
        "hazards": _normalize_grounded_rows(scene_analysis.get("hazards"), row_kind="hazards"),
        "privacy_sensitive_areas": _normalize_grounded_rows(
            scene_analysis.get("privacy_sensitive_areas"),
            row_kind="privacy_sensitive_areas",
        ),
    }
    identity = {
        "discovery_id": discovery_id,
        "source_capture": _clone(source_capture),
        "capture_qa_report_digest": capture_qa_report_digest,
        "scene_analysis": analysis,
        "proposal_method": _clone(proposal_method),
    }
    candidates = [
        _normalize_candidate(
            proposal,
            discovery_identity=identity,
            proposal_method=proposal_method,
            observed_fact_ids={
                _text(row.get("fact_id"))
                for row in analysis["observed_site_facts"]
                if _text(row.get("fact_id"))
            },
            index=index,
        )
        for index, proposal in enumerate(candidate_proposals)
    ]
    candidates.sort(key=lambda candidate: candidate["task_candidate_id"])
    artifact = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        **identity,
        "task_candidates": candidates,
        "approval_state": "task_approval_required" if candidates else "no_candidates",
        "claim_boundaries": {
            "candidate_is_customer_intent": False,
            "candidate_is_task_success_evidence": False,
            "generated_or_inferred_content_upgrades_capture_authority": False,
        },
    }
    _reject_secrets(artifact)
    artifact["discovery_digest"] = canonical_digest(artifact, digest_field="discovery_digest")
    return artifact


def _validate_discovery(discovery: Mapping[str, Any]) -> dict[str, Any]:
    value = _clone(discovery)
    errors: list[str] = []
    if value.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        errors.append(f"schema_version:must_be:{DISCOVERY_SCHEMA_VERSION}")
    if canonical_digest(value, digest_field="discovery_digest") != value.get("discovery_digest"):
        errors.append("discovery_digest:mismatch")
    candidates = _rows(value.get("task_candidates"))
    if len(candidates) != len(value.get("task_candidates") or []):
        errors.append("task_candidates:invalid")
    for index, candidate in enumerate(candidates):
        if canonical_digest(candidate, digest_field="candidate_digest") != candidate.get(
            "candidate_digest"
        ):
            errors.append(f"task_candidates[{index}].candidate_digest:mismatch")
        if candidate.get("approval_status") != "approval_required":
            errors.append(f"task_candidates[{index}].approval_status:must_be:approval_required")
    if errors:
        raise TaskCandidateContractError(errors)
    return value


def _approved_task_body(candidate: Mapping[str, Any], edited_task: Mapping[str, Any] | None) -> dict[str, Any]:
    if edited_task is None:
        return {
            "description": candidate["description"],
            "task_family": candidate["likely_task_family"],
            "measurable_success_conditions": [
                _clone(candidate["proposed_measurable_success_condition"])
            ],
            "reset_contract": {"instructions": candidate["required_site_reset"]},
            "task_objects": _clone(candidate["observed_objects"]),
            "target_regions": _clone(candidate["target_regions"]),
            "required_robot_capabilities": _clone(candidate["required_robot_capabilities"]),
        }
    body = _clone(edited_task)
    errors: list[str] = []
    for key in ("description", "task_family"):
        if not _text(body.get(key)):
            errors.append(f"edited_task.{key}:missing")
    conditions = _rows(body.get("measurable_success_conditions"))
    if not conditions:
        errors.append("edited_task.measurable_success_conditions:missing_or_invalid")
    else:
        for index, condition in enumerate(conditions):
            _normalize_success_condition(
                condition,
                prefix=f"edited_task.measurable_success_conditions[{index}]",
            )
    if not isinstance(body.get("reset_contract"), Mapping) or not body.get("reset_contract"):
        errors.append("edited_task.reset_contract:missing_or_empty")
    if errors:
        raise TaskCandidateContractError(errors)
    return body


def record_task_candidate_decision(
    discovery: Mapping[str, Any],
    *,
    task_candidate_id: str,
    action: str,
    actor: Mapping[str, Any],
    idempotency_key: str,
    rationale: str,
    edited_task: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Record an append-only decision and return an approved task when applicable."""

    value = _validate_discovery(discovery)
    errors: list[str] = []
    if action not in _APPROVAL_ACTIONS:
        errors.append("action:unsupported")
    if _text(actor.get("role")) not in _APPROVER_ROLES:
        errors.append("actor.role:not_authorized")
    if not _text(actor.get("identity")):
        errors.append("actor.identity:missing")
    if not _text(idempotency_key):
        errors.append("idempotency_key:missing")
    if not _text(rationale):
        errors.append("rationale:missing")
    if action == "edit_and_approve" and edited_task is None:
        errors.append("edited_task:required")
    if action != "edit_and_approve" and edited_task is not None:
        errors.append("edited_task:only_allowed_for_edit_and_approve")
    candidates = [
        candidate
        for candidate in value["task_candidates"]
        if candidate.get("task_candidate_id") == task_candidate_id
    ]
    if len(candidates) != 1:
        errors.append("task_candidate_id:not_found")
    if errors:
        raise TaskCandidateContractError(errors)
    candidate = candidates[0]
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "discovery_id": value["discovery_id"],
        "discovery_digest": value["discovery_digest"],
        "task_candidate_id": task_candidate_id,
        "candidate_digest": candidate["candidate_digest"],
        "action": action,
        "actor": _clone(actor),
        "idempotency_key": idempotency_key,
        "rationale": rationale,
        "edited_task": _clone(edited_task),
    }
    _reject_secrets(decision)
    decision["decision_id"] = _stable_id("task-decision", decision)
    decision["decision_digest"] = canonical_digest(decision, digest_field="decision_digest")
    if action not in {"approve", "edit_and_approve"}:
        return decision, None

    task_body = _approved_task_body(candidate, edited_task)
    proposer_identity = _text(value["proposal_method"].get("proposer_identity"))
    proposal_origin = _text(value["proposal_method"].get("origin")) or "local_rule"
    prohibited_evaluators = [proposer_identity] if proposal_origin in _MODEL_PROPOSAL_ORIGINS else []
    approved = {
        "schema_version": APPROVED_TASK_SCHEMA_VERSION,
        "approved_task_id": _stable_id(
            "approved-task", {"decision_digest": decision["decision_digest"], "task": task_body}
        ),
        "source_capture": _clone(value["source_capture"]),
        "discovery_id": value["discovery_id"],
        "discovery_digest": value["discovery_digest"],
        "task_candidate_id": task_candidate_id,
        "candidate_digest": candidate["candidate_digest"],
        "approval_decision_id": decision["decision_id"],
        "approval_decision_digest": decision["decision_digest"],
        "approval_actor": _clone(actor),
        "intent_source": "customer_edited_candidate" if edited_task is not None else "customer_approved_candidate",
        "task": task_body,
        "proposer_identity": proposer_identity,
        "prohibited_evaluator_identities": prohibited_evaluators,
        "approval_status": "approved",
    }
    approved["approved_task_digest"] = canonical_digest(
        approved, digest_field="approved_task_digest"
    )
    return decision, approved


def record_customer_supplied_task(
    *,
    source_capture: Mapping[str, Any],
    task: Mapping[str, Any],
    actor: Mapping[str, Any],
    idempotency_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record exact customer-supplied intent without inventing missing thresholds."""

    errors: list[str] = []
    if not _is_digest(source_capture.get("capture_digest")):
        errors.append("source_capture.capture_digest:invalid_sha256")
    for key in ("intake_id", "capture_authority_profile"):
        if not _text(source_capture.get(key)):
            errors.append(f"source_capture.{key}:missing")
    if _text(actor.get("role")) not in _APPROVER_ROLES:
        errors.append("actor.role:not_authorized")
    if not _text(actor.get("identity")):
        errors.append("actor.identity:missing")
    if not _text(idempotency_key):
        errors.append("idempotency_key:missing")
    if errors:
        raise TaskCandidateContractError(errors)
    task_body = _approved_task_body({}, task)
    receipt = {
        "schema_version": "customer_supplied_task_receipt.v1",
        "source_capture": _clone(source_capture),
        "task": task_body,
        "actor": _clone(actor),
        "idempotency_key": idempotency_key,
    }
    _reject_secrets(receipt)
    receipt["receipt_id"] = _stable_id("task-receipt", receipt)
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    approved = {
        "schema_version": APPROVED_TASK_SCHEMA_VERSION,
        "approved_task_id": _stable_id(
            "approved-task", {"receipt_digest": receipt["receipt_digest"], "task": task_body}
        ),
        "source_capture": _clone(source_capture),
        "discovery_id": None,
        "discovery_digest": None,
        "task_candidate_id": None,
        "candidate_digest": None,
        "approval_decision_id": receipt["receipt_id"],
        "approval_decision_digest": receipt["receipt_digest"],
        "approval_actor": _clone(actor),
        "intent_source": "customer_supplied",
        "task": task_body,
        "proposer_identity": _text(actor.get("identity")),
        "prohibited_evaluator_identities": [],
        "approval_status": "approved",
    }
    approved["approved_task_digest"] = canonical_digest(
        approved, digest_field="approved_task_digest"
    )
    return receipt, approved


def compile_approved_task_decision_request(
    approved_task: Mapping[str, Any],
    *,
    testbed: Mapping[str, Any],
    request_id: str,
    decision_id: str,
    candidates: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    deadline: str,
    permitted_evidence_methods: Sequence[str],
    restrictions: Mapping[str, Any],
    requested_result_audience: str,
    caller_identity: str,
    idempotency_key: str,
    proposed_evaluator_identities: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile approved intent into the existing provider-neutral request contract."""

    task = _clone(approved_task)
    errors: list[str] = []
    if task.get("schema_version") != APPROVED_TASK_SCHEMA_VERSION:
        errors.append(f"schema_version:must_be:{APPROVED_TASK_SCHEMA_VERSION}")
    if task.get("approval_status") != "approved":
        errors.append("approval_status:must_be:approved")
    if canonical_digest(task, digest_field="approved_task_digest") != task.get(
        "approved_task_digest"
    ):
        errors.append("approved_task_digest:mismatch")
    for key in ("testbed_id", "version", "testbed_digest"):
        if not _text(testbed.get(key)):
            errors.append(f"testbed.{key}:missing")
    if testbed.get("testbed_digest") and not _is_digest(testbed.get("testbed_digest")):
        errors.append("testbed.testbed_digest:invalid_sha256")
    task_binding = testbed.get("approved_task_definition")
    if not isinstance(task_binding, Mapping) or task_binding.get("digest") != task.get(
        "approved_task_digest"
    ):
        errors.append("testbed.approved_task_definition:digest_mismatch")
    prohibited = set(_strings(task.get("prohibited_evaluator_identities")))
    attempted = {_text(identity) for identity in proposed_evaluator_identities if _text(identity)}
    if prohibited.intersection(attempted):
        errors.append("task_proposer_self_grading_forbidden")
    if errors:
        raise TaskCandidateContractError(errors)

    request_restrictions = _clone(restrictions)
    request_restrictions["prohibited_evaluator_identities"] = sorted(prohibited)
    task_body = task["task"]
    request = DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": request_id,
            "decision_id": decision_id,
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": f"Which claims are supported for: {task_body['description']}",
            "candidates": _clone(list(candidates)),
            "claims": _clone(list(claims)),
            "budget": _clone(budget),
            "deadline": deadline,
            "available_physical_evidence": [],
            "permitted_evidence_methods": sorted({_text(method) for method in permitted_evidence_methods if _text(method)}),
            "restrictions": request_restrictions,
            "requested_result_audience": requested_result_audience,
            "provenance": {
                "caller_identity": caller_identity,
                "approved_task_id": task["approved_task_id"],
                "approved_task_digest": task["approved_task_digest"],
                "approval_decision_digest": task["approval_decision_digest"],
                "source_capture_digest": task["source_capture"]["capture_digest"],
            },
            "idempotency_key": idempotency_key,
        }
    )
    return request.to_mapping()
