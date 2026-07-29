"""Deprecated product-input translation into a Task Evaluation Run request.

Policy Improvement Run and Post-Training Data Package contracts remain readable
for compatibility.  They are not primary products or router authority.  This
module extracts only bounded, non-secret decision constraints from legacy
inputs and emits the canonical provider-neutral request contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .decision_evidence_contracts import DecisionEvidenceRequest, MaintainedSiteTaskTestbed


@dataclass(frozen=True)
class LegacyTaskEvaluationTranslation:
    request: DecisionEvidenceRequest
    metadata: Mapping[str, Any]


def _scope(testbed: Mapping[str, Any]) -> dict[str, Any]:
    bindings = dict(testbed.get("robot_sensor_controller_bindings") or {})
    distribution = dict(testbed.get("task_distribution") or {})
    return {
        "task_family": distribution.get("task_family"),
        "site_domain_conditions": dict(testbed.get("supported_condition_ranges") or {}),
        "embodiment": dict(bindings.get("embodiment") or {}),
        "sensors": dict(bindings.get("sensors") or {}),
        "controller_action_representation": dict(
            bindings.get("controller_action_representation") or {}
        ),
    }


def _claim(claim_id: str, claim_type: str, subject: str, *, consequence: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "subject": subject,
        "measurable_threshold": None,
        "false_safe_consequence": consequence,
        "acceptable_false_safe_risk": 0.05 if consequence != "critical" else 0.001,
        "desired_confidence_or_coverage": {
            "minimum_coverage": 0.8,
            "minimum_independent_methods": 1,
        },
        "permitted_abstention_behavior": {"allowed": True, "required_when_unqualified": True},
    }


def _translate(
    legacy: Mapping[str, Any],
    testbed_value: Mapping[str, Any],
    *,
    legacy_kind: str,
) -> LegacyTaskEvaluationTranslation:
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    legacy_id = str(
        legacy.get("run_id")
        or legacy.get("request_id")
        or legacy.get("package_id")
        or legacy.get("policy_improvement_run_id")
        or "legacy-request"
    ).strip()
    candidates = legacy.get("candidates") or legacy.get("policies") or []
    if not isinstance(candidates, list):
        candidates = []
    safe_candidates = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        safe_candidates.append(
            {
                "candidate_id": str(
                    candidate.get("candidate_id")
                    or candidate.get("policy_id")
                    or candidate.get("checkpoint_id")
                    or f"legacy-candidate-{index + 1}"
                ),
                "checkpoint_digest": candidate.get("checkpoint_digest"),
            }
        )
    if legacy_kind == "policy_improvement_run":
        claims = [
            _claim(
                "legacy-policy-comparison",
                "comparative_policy_ranking",
                "legacy policy candidates",
                consequence="moderate",
            )
        ]
        question = "Does qualified evidence support a comparative decision among the legacy policy candidates?"
    else:
        claims = [
            _claim(
                "legacy-evidence-use",
                "post_training_evidence_use_eligibility",
                "rights-cleared evidence produced within this Task Evaluation Run",
                consequence="critical",
            )
        ]
        question = "Is evidence from this Task Evaluation Run eligible for the requested evaluation or post-training use?"
    scope = _scope(testbed)
    claims = [{**claim, **scope} for claim in claims]
    max_cost = legacy.get("max_cost_usd", legacy.get("budget_usd", 0.0))
    max_latency = legacy.get("max_latency_seconds", 3600.0)
    request = DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": f"translated-{legacy_kind}-{legacy_id}",
            "decision_id": f"decision-{legacy_id}",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": question,
            "candidates": safe_candidates,
            "claims": claims,
            "budget": {
                "max_cost_usd": float(max_cost or 0.0),
                "max_latency_seconds": float(max_latency or 0.0),
                "delay_cost_per_second": 0.0,
            },
            "deadline": str(legacy.get("deadline") or "unspecified"),
            "available_physical_evidence": [],
            "permitted_evidence_methods": [
                "analytic_geometry_kinematics",
                "captured_real_observation",
                "traditional_simulation",
                "learned_world_model",
                "external_provider_tool",
                "physical_evidence",
                "owner_attested_operational_input",
            ],
            "restrictions": {
                "external_processing_allowed": bool(
                    legacy.get("external_processing_allowed", False)
                ),
                "max_data_retention_days": int(legacy.get("max_data_retention_days", 0) or 0),
            },
            "requested_result_audience": str(legacy.get("audience") or "legacy_caller"),
            "provenance": {
                "caller_identity": str(legacy.get("caller_identity") or "legacy-translator"),
                "legacy_contract_kind": legacy_kind,
                "legacy_contract_id": legacy_id,
            },
            "idempotency_key": str(
                legacy.get("idempotency_key") or f"translate-{legacy_kind}-{legacy_id}"
            ),
        }
    )
    metadata = {
        "schema_version": "legacy_task_evaluation_translation.v1",
        "legacy_contract_kind": legacy_kind,
        "legacy_contract_id": legacy_id,
        "deprecated": True,
        "replacement_product": "Task Evaluation Run",
        "translation_grants_qualification": False,
        "translation_implies_training_occurred": False,
        "translation_implies_policy_improved": False,
        "secret_values_copied": False,
        "request_digest": request.digest,
    }
    return LegacyTaskEvaluationTranslation(request=request, metadata=metadata)


def translate_policy_improvement_request(
    value: Mapping[str, Any], testbed_value: Mapping[str, Any]
) -> LegacyTaskEvaluationTranslation:
    return _translate(value, testbed_value, legacy_kind="policy_improvement_run")


def translate_post_training_data_request(
    value: Mapping[str, Any], testbed_value: Mapping[str, Any]
) -> LegacyTaskEvaluationTranslation:
    return _translate(value, testbed_value, legacy_kind="post_training_data_package")


__all__ = [
    "LegacyTaskEvaluationTranslation",
    "translate_policy_improvement_request",
    "translate_post_training_data_request",
]
