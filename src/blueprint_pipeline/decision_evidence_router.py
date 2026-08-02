"""Deterministic claim-level Decision/Evidence Router.

The router minimizes evidence cost, delay, and expected decision loss subject
to exact qualification, false-safe, coverage, rights, reproducibility, input,
budget, and availability gates.  It produces a plan only; provider execution
remains behind explicit adapters and the canonical EvaluationRunSpec authority.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
    canonical_digest,
)
from .evaluation_run_contract import EVALUATION_RUN_SCHEMA_VERSION, validate_evaluation_run_spec


_BASE_AUTHORITY = {
    "reachability": 1,
    "kinematic_feasibility": 1,
    "perception_visibility": 1,
    "collision_contact": 2,
    "comparative_controller_ranking": 2,
    "comparative_policy_ranking": 3,
    "physical_task_success": 4,
    "deployment_readiness": 4,
    "safety_certification": 4,
}
_CONSEQUENCE_WEIGHT = {
    "low": 1.0,
    "moderate": 10.0,
    "high": 100.0,
    "critical": 1000.0,
}
_EXECUTION_LEAF_FAMILIES = {
    "traditional_simulation",
    "learned_world_model",
    "external_provider_tool",
}
_PHYSICAL_CLAIMS = {
    "physical_task_success",
    "deployment_readiness",
    "safety_certification",
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item) for item in value if _string(item)]


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_id(*parts: str) -> str:
    rendered = "-".join(_string(part) for part in parts if _string(part))
    rendered = re.sub(r"[^A-Za-z0-9._:-]+", "-", rendered).strip("-")
    return rendered[:192] or "decision-evidence-step"


def _required_authority(claim: Mapping[str, Any]) -> int:
    claim_type = _string(claim.get("claim_type"))
    base = _BASE_AUTHORITY.get(claim_type, 2)
    consequence = _string(claim.get("false_safe_consequence")).lower()
    risk = _number(claim.get("acceptable_false_safe_risk"), 1.0)
    if consequence == "critical" and risk <= 0.001:
        return max(base, 4)
    if consequence in {"high", "critical"} and risk <= 0.01:
        return max(base, 3)
    return base


def _evidence_inventory(testbed: Mapping[str, Any], request: Mapping[str, Any]) -> set[str]:
    available: set[str] = set()
    for item in list(testbed.get("evidence_inventory") or []) + list(
        request.get("available_physical_evidence") or []
    ):
        if isinstance(item, Mapping):
            identity = _string(
                item.get("evidence_id") or item.get("artifact_id") or item.get("id")
            )
            if identity:
                available.add(identity)
        elif _string(item):
            available.add(_string(item))
    return available


def _scope_value(claim: Mapping[str, Any], testbed: Mapping[str, Any], key: str) -> Any:
    if key in claim:
        return claim.get(key)
    if key == "task_family":
        distribution = testbed.get("task_distribution")
        if isinstance(distribution, Mapping):
            return distribution.get("task_family")
    if key == "site_domain_conditions":
        return claim.get("conditions") or testbed.get("supported_condition_ranges")
    bindings = testbed.get("robot_sensor_controller_bindings")
    if isinstance(bindings, Mapping):
        return bindings.get(key)
    return None


def _qualification_scope_matches(
    qualification: Mapping[str, Any],
    claim: Mapping[str, Any],
    testbed: Mapping[str, Any],
) -> bool:
    for key in (
        "task_family",
        "site_domain_conditions",
        "embodiment",
        "sensors",
        "controller_action_representation",
    ):
        expected = _scope_value(claim, testbed, key)
        if expected not in (None, "", {}, []) and qualification.get(key) != expected:
            return False
    return True


def _matching_qualifications(
    profile: Mapping[str, Any],
    claim: Mapping[str, Any],
    testbed: Mapping[str, Any],
    qualifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for qualification in qualifications:
        if _string(qualification.get("status")) != "qualified":
            continue
        if _string(qualification.get("method_id")) != _string(profile.get("method_id")):
            continue
        if _string(qualification.get("method_version")) != _string(profile.get("version")):
            continue
        if qualification.get("method_profile_digest") != profile.get("method_profile_digest"):
            continue
        if qualification.get("implementation_digest") != profile.get("implementation_digest"):
            continue
        if _string(qualification.get("claim_type")) != _string(claim.get("claim_type")):
            continue
        if not _qualification_scope_matches(qualification, claim, testbed):
            continue
        matches.append(dict(qualification))
    return sorted(matches, key=lambda row: _string(row.get("qualification_digest")))


def _method_permitted(
    profile: Mapping[str, Any], request: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    permitted = set(_strings(request.get("permitted_evidence_methods")))
    method_id = _string(profile.get("method_id"))
    family = _string(profile.get("method_family"))
    if method_id not in permitted and family not in permitted:
        reasons.append("method_not_permitted")
    restrictions = request.get("restrictions")
    restrictions = dict(restrictions) if isinstance(restrictions, Mapping) else {}
    if method_id in set(_strings(restrictions.get("prohibited_method_ids"))):
        reasons.append("method_restricted")
    allowed_families = set(_strings(restrictions.get("allowed_method_families")))
    if allowed_families and family not in allowed_families:
        reasons.append("method_family_restricted")
    constraints = profile.get("constraints")
    constraints = dict(constraints) if isinstance(constraints, Mapping) else {}
    if constraints.get("external_processing") is True and restrictions.get(
        "external_processing_allowed"
    ) is not True:
        reasons.append("external_processing_rights_incompatible")
    prohibited_providers = set(_strings(restrictions.get("prohibited_provider_ids")))
    provider_id = _string(constraints.get("provider_id"))
    if provider_id and provider_id in prohibited_providers:
        reasons.append("provider_restricted")
    request_retention = _number(restrictions.get("max_data_retention_days"), -1.0)
    method_retention = _number(constraints.get("data_retention_days"), 0.0)
    if request_retention >= 0 and method_retention > request_retention:
        reasons.append("data_retention_incompatible")
    return reasons


def _applicability_reasons(
    profile: Mapping[str, Any],
    claim: Mapping[str, Any],
    testbed: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    envelope = profile.get("applicability_envelope")
    envelope = dict(envelope) if isinstance(envelope, Mapping) else {}
    checks = {
        "testbed_ids": testbed.get("testbed_id"),
        "testbed_versions": testbed.get("version"),
        "task_families": _scope_value(claim, testbed, "task_family"),
    }
    for allowed_key, actual in checks.items():
        allowed = envelope.get(allowed_key)
        if isinstance(allowed, list) and allowed and actual not in allowed:
            reasons.append(f"unsupported_domain:{allowed_key}")
    excluded_conditions = envelope.get("excluded_conditions")
    conditions = _scope_value(claim, testbed, "site_domain_conditions")
    if isinstance(excluded_conditions, list) and isinstance(conditions, Mapping):
        if any(condition in conditions for condition in excluded_conditions):
            reasons.append("unsupported_domain:excluded_condition")
    return reasons


def _leaf_spec(
    *,
    profile: Mapping[str, Any],
    qualification: Mapping[str, Any],
    claim: Mapping[str, Any],
    request: Mapping[str, Any],
    testbed: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    family = _string(profile.get("method_family"))
    if family not in _EXECUTION_LEAF_FAMILIES:
        return None, []
    template = profile.get("evaluation_run_template")
    if not isinstance(template, Mapping):
        return None, ["evaluation_run_template_missing"]
    leaf = copy.deepcopy(dict(template))
    leaf["schema_version"] = EVALUATION_RUN_SCHEMA_VERSION
    leaf["run_id"] = _stable_id(
        _string(request.get("request_id")),
        _string(claim.get("claim_id")),
        _string(profile.get("method_id")),
    )
    metadata = leaf.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata.update(
        {
            "decision_request_digest": request.get("request_digest"),
            "testbed_digest": testbed.get("testbed_digest"),
            "claim_id": claim.get("claim_id"),
            "method_profile_digest": profile.get("method_profile_digest"),
            "qualification_digest": qualification.get("qualification_digest"),
            "router_selected_provider_as_qualification": False,
        }
    )
    leaf["metadata"] = metadata
    validation = validate_evaluation_run_spec(leaf)
    if validation["status"] != "passed":
        return None, [
            f"evaluation_run_template_invalid:{error}"
            for error in validation.get("errors", [])
        ]
    return leaf, []


@dataclass(frozen=True)
class _Candidate:
    profile: dict[str, Any]
    qualification: dict[str, Any] | None
    reasons: tuple[str, ...]
    objective: float
    leaf_spec: dict[str, Any] | None

    @property
    def eligible(self) -> bool:
        return not self.reasons and self.qualification is not None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method_id": self.profile.get("method_id"),
            "method_version": self.profile.get("version"),
            "method_family": self.profile.get("method_family"),
            "method_profile_digest": self.profile.get("method_profile_digest"),
            "qualification_digest": (
                self.qualification.get("qualification_digest")
                if self.qualification
                else None
            ),
            "qualification_result": "qualified" if self.eligible else "rejected",
            "rejection_reasons": list(self.reasons),
            "authority_tier": self.profile.get("authority_tier"),
            "correlation_group": self.profile.get("correlation_group"),
            "shared_dependencies": list(self.profile.get("shared_dependencies") or []),
            "expected_cost_usd": self.profile.get("expected_cost_usd"),
            "expected_latency_seconds": self.profile.get("expected_latency_seconds"),
            "selection_objective": self.objective,
            "compiles_evaluation_run_spec": self.leaf_spec is not None,
        }


def _candidate(
    *,
    profile: Mapping[str, Any],
    claim: Mapping[str, Any],
    request: Mapping[str, Any],
    testbed: Mapping[str, Any],
    qualifications: Sequence[Mapping[str, Any]],
    available_inputs: set[str],
    remaining_budget: float,
) -> _Candidate:
    reasons = _method_permitted(profile, request)
    claim_type = _string(claim.get("claim_type"))
    if claim_type not in set(_strings(profile.get("supported_claim_types"))):
        reasons.append("claim_type_not_supported")
    availability = profile.get("provider_availability")
    availability = dict(availability) if isinstance(availability, Mapping) else {}
    if _string(availability.get("status")) != "available":
        reasons.append("method_unavailable")
    if _string(profile.get("reproducibility_level")) in {"none", "unrepeatable"}:
        reasons.append("reproducibility_insufficient")
    reasons.extend(_applicability_reasons(profile, claim, testbed))
    missing_inputs = sorted(set(_strings(profile.get("required_inputs"))) - available_inputs)
    reasons.extend(f"required_input_missing:{value}" for value in missing_inputs)
    if _number(profile.get("expected_cost_usd")) > remaining_budget:
        reasons.append("over_budget")
    max_latency = _number(dict(request.get("budget") or {}).get("max_latency_seconds"), -1)
    if max_latency >= 0 and _number(profile.get("expected_latency_seconds")) > max_latency:
        reasons.append("over_latency_budget")
    if _number(profile.get("authority_tier")) < _required_authority(claim):
        reasons.append("authority_tier_insufficient")

    matches = _matching_qualifications(profile, claim, testbed, qualifications)
    qualification = matches[0] if matches else None
    if qualification is None:
        reasons.append("unqualified_or_out_of_scope")
    else:
        if _number(qualification.get("false_safe_rate"), 1.0) > _number(
            claim.get("acceptable_false_safe_risk"), 0.0
        ):
            reasons.append("false_safe_limit_exceeded")
        desired = claim.get("desired_confidence_or_coverage")
        desired = dict(desired) if isinstance(desired, Mapping) else {}
        if _number(qualification.get("coverage")) < _number(
            desired.get("minimum_coverage"), 0.0
        ):
            reasons.append("minimum_coverage_not_met")

    leaf: dict[str, Any] | None = None
    if qualification is not None:
        leaf, leaf_reasons = _leaf_spec(
            profile=profile,
            qualification=qualification,
            claim=claim,
            request=request,
            testbed=testbed,
        )
        reasons.extend(leaf_reasons)

    consequence = _string(claim.get("false_safe_consequence")).lower()
    expected_loss = _number(
        profile.get("expected_decision_loss"),
        _number(qualification.get("false_safe_rate"), 1.0)
        * _CONSEQUENCE_WEIGHT.get(consequence, 10.0)
        if qualification
        else _CONSEQUENCE_WEIGHT.get(consequence, 10.0),
    )
    delay_cost = _number(dict(request.get("budget") or {}).get("delay_cost_per_second"))
    objective = (
        _number(profile.get("expected_cost_usd"))
        + delay_cost * _number(profile.get("expected_latency_seconds"))
        + expected_loss
    )
    return _Candidate(
        profile=dict(profile),
        qualification=qualification,
        reasons=tuple(sorted(set(reasons))),
        objective=round(objective, 12),
        leaf_spec=leaf,
    )


def _prune_dominated(candidates: list[_Candidate]) -> list[_Candidate]:
    updated: list[_Candidate] = []
    eligible = [candidate for candidate in candidates if candidate.eligible]
    for candidate in candidates:
        reasons = list(candidate.reasons)
        if candidate.eligible:
            cost = _number(candidate.profile.get("expected_cost_usd"))
            latency = _number(candidate.profile.get("expected_latency_seconds"))
            authority = _number(candidate.profile.get("authority_tier"))
            for other in eligible:
                if other is candidate:
                    continue
                other_cost = _number(other.profile.get("expected_cost_usd"))
                other_latency = _number(other.profile.get("expected_latency_seconds"))
                other_authority = _number(other.profile.get("authority_tier"))
                candidate_false_safe = _number(
                    (candidate.qualification or {}).get("false_safe_rate"), 1.0
                )
                other_false_safe = _number(
                    (other.qualification or {}).get("false_safe_rate"), 1.0
                )
                candidate_coverage = _number(
                    (candidate.qualification or {}).get("coverage"), 0.0
                )
                other_coverage = _number((other.qualification or {}).get("coverage"), 0.0)
                weakly_better = (
                    other_cost <= cost
                    and other_latency <= latency
                    and other_authority >= authority
                    and other_false_safe <= candidate_false_safe
                    and other_coverage >= candidate_coverage
                )
                strictly_better = (
                    other_cost < cost
                    or other_latency < latency
                    or other_authority > authority
                    or other_false_safe < candidate_false_safe
                    or other_coverage > candidate_coverage
                )
                if weakly_better and strictly_better:
                    reasons.append(
                        f"dominated_by:{_string(other.profile.get('method_id'))}"
                    )
                    break
        updated.append(
            _Candidate(
                profile=candidate.profile,
                qualification=candidate.qualification,
                reasons=tuple(sorted(set(reasons))),
                objective=candidate.objective,
                leaf_spec=candidate.leaf_spec,
            )
        )
    return updated


def route_decision_evidence(
    request_value: Mapping[str, Any],
    testbed_value: Mapping[str, Any],
    method_values: Sequence[Mapping[str, Any]],
    qualification_values: Sequence[Mapping[str, Any]],
) -> EvidencePlan:
    """Build one deterministic, inspectable evidence plan before execution."""

    request = DecisionEvidenceRequest.from_mapping(request_value).to_mapping()
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    if request["testbed_id"] != testbed["testbed_id"]:
        raise ValueError("decision_request_testbed_id_mismatch")
    if request["testbed_version"] != testbed["version"]:
        raise ValueError("decision_request_testbed_version_mismatch")
    if request["testbed_digest"] != testbed["testbed_digest"]:
        raise ValueError("decision_request_testbed_digest_mismatch")

    methods = [EvidenceMethodProfile.from_mapping(value).to_mapping() for value in method_values]
    methods.sort(key=lambda value: (_string(value.get("method_id")), _string(value.get("version"))))
    qualifications = [
        QualificationRecord.from_mapping(value).to_mapping()
        for value in qualification_values
    ]
    qualifications.sort(key=lambda value: _string(value.get("qualification_digest")))
    available_inputs = _evidence_inventory(testbed, request)
    budget = dict(request.get("budget") or {})
    max_budget = _number(budget.get("max_cost_usd"))
    projected_cost = 0.0
    projected_latency = 0.0

    claim_plans: list[dict[str, Any]] = []
    execution_order: list[str] = []
    leaf_specs: list[dict[str, Any]] = []
    non_leaf_steps: list[dict[str, Any]] = []
    physical_requests: list[dict[str, Any]] = []
    shared_warnings: list[dict[str, Any]] = []
    prohibited_claims = sorted(_PHYSICAL_CLAIMS | {"policy_ranking_thesis_upgrade"})

    claims = sorted(_rows(request.get("claims")), key=lambda value: _string(value.get("claim_id")))
    for claim in claims:
        remaining_budget = max(0.0, max_budget - projected_cost)
        candidates = [
            _candidate(
                profile=profile,
                claim=claim,
                request=request,
                testbed=testbed,
                qualifications=qualifications,
                available_inputs=available_inputs,
                remaining_budget=remaining_budget,
            )
            for profile in methods
        ]
        candidates = _prune_dominated(candidates)
        eligible = sorted(
            (candidate for candidate in candidates if candidate.eligible),
            key=lambda candidate: (
                candidate.objective,
                _number(candidate.profile.get("expected_cost_usd")),
                _number(candidate.profile.get("expected_latency_seconds")),
                _string(candidate.profile.get("method_id")),
            ),
        )
        desired = claim.get("desired_confidence_or_coverage")
        desired = dict(desired) if isinstance(desired, Mapping) else {}
        independent_count = max(1, int(_number(desired.get("minimum_independent_methods"), 1)))
        selected: list[_Candidate] = []
        correlation_groups: set[str] = set()
        for candidate in eligible:
            group = _string(candidate.profile.get("correlation_group"))
            if group and group in correlation_groups:
                shared_warnings.append(
                    {
                        "claim_id": claim.get("claim_id"),
                        "method_id": candidate.profile.get("method_id"),
                        "correlation_group": group,
                        "counted_as_independent": False,
                    }
                )
                continue
            selected.append(candidate)
            if group:
                correlation_groups.add(group)
            if len(selected) >= independent_count:
                break

        status = "planned"
        rationale = "cheapest_qualified_sufficient_evidence"
        next_experiment = "none_required"
        if len(selected) < independent_count:
            selected = []
            status = "abstention_planned"
            rationale = "no_qualified_sufficient_plan"
            rejected = sorted(
                candidates,
                key=lambda candidate: (
                    candidate.objective,
                    _string(candidate.profile.get("method_id")),
                ),
            )
            # An abstention should identify the cheapest experiment that could
            # actually answer this claim.  A globally cheaper profile for a
            # different claim type is not a legal escalation candidate.
            claim_compatible = [
                candidate
                for candidate in rejected
                if "claim_type_not_supported" not in candidate.reasons
            ]
            next_candidate = next(
                (candidate for candidate in claim_compatible if candidate.reasons),
                None,
            )
            if next_candidate is not None:
                next_experiment = (
                    f"qualify_or_supply:{next_candidate.profile.get('method_id')}:"
                    f"{','.join(next_candidate.reasons)}"
                )
            if _string(claim.get("claim_type")) in _PHYSICAL_CLAIMS:
                evidence_request = {
                    "request_id": _stable_id(
                        _string(request.get("request_id")),
                        _string(claim.get("claim_id")),
                        "physical-evidence",
                    ),
                    "claim_id": claim.get("claim_id"),
                    "bounded_request": True,
                    "robot_run_initiated": False,
                    "required_binding": {
                        "testbed_digest": testbed.get("testbed_digest"),
                        "claim_subject": claim.get("subject"),
                        "task_family": _scope_value(claim, testbed, "task_family"),
                        "site_domain_conditions": _scope_value(
                            claim, testbed, "site_domain_conditions"
                        ),
                    },
                }
                physical_requests.append(evidence_request)
                next_experiment = f"collect_physical_evidence:{evidence_request['request_id']}"

        selected_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(selected):
            method_id = _string(candidate.profile.get("method_id"))
            step_id = _stable_id(_string(claim.get("claim_id")), method_id)
            execution_order.append(step_id)
            projected_cost += _number(candidate.profile.get("expected_cost_usd"))
            projected_latency += _number(candidate.profile.get("expected_latency_seconds"))
            row = {
                "step_id": step_id,
                "claim_id": claim.get("claim_id"),
                "method_id": method_id,
                "method_profile_digest": candidate.profile.get("method_profile_digest"),
                "qualification_digest": candidate.qualification.get("qualification_digest")
                if candidate.qualification
                else None,
                "execution_rank": index,
                "stop_when_sufficient": True,
                "escalate_on": [
                    "invalid",
                    "uncertain",
                    "contradictory",
                    "coverage_below_required",
                    "domain_mismatch",
                ],
            }
            selected_rows.append(row)
            if candidate.leaf_spec is not None:
                leaf_specs.append(candidate.leaf_spec)
            else:
                non_leaf_steps.append(
                    {
                        **row,
                        "adapter_reference": candidate.profile.get("adapter_reference"),
                        "method_family": candidate.profile.get("method_family"),
                    }
                )

        escalation_rows: list[dict[str, Any]] = []
        if selected:
            selected_digests = {
                _string(candidate.profile.get("method_profile_digest")) for candidate in selected
            }
            for index, candidate in enumerate(
                candidate
                for candidate in eligible
                if _string(candidate.profile.get("method_profile_digest")) not in selected_digests
            ):
                method_id = _string(candidate.profile.get("method_id"))
                step_id = _stable_id(_string(claim.get("claim_id")), method_id, "escalation")
                execution_order.append(step_id)
                row = {
                    "step_id": step_id,
                    "claim_id": claim.get("claim_id"),
                    "method_id": method_id,
                    "method_profile_digest": candidate.profile.get("method_profile_digest"),
                    "qualification_digest": candidate.qualification.get("qualification_digest")
                    if candidate.qualification
                    else None,
                    "execution_rank": len(selected) + index,
                    "execution_role": "conditional_escalation",
                    "execute_only_after": [
                        "invalid",
                        "uncertain",
                        "contradictory",
                        "unavailable",
                        "coverage_below_required",
                    ],
                    "stop_when_sufficient": True,
                }
                escalation_rows.append(row)
                if candidate.leaf_spec is not None:
                    leaf_specs.append(candidate.leaf_spec)
                else:
                    non_leaf_steps.append(
                        {
                            **row,
                            "adapter_reference": candidate.profile.get("adapter_reference"),
                            "method_family": candidate.profile.get("method_family"),
                        }
                    )

        claim_plans.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim_type": claim.get("claim_type"),
                "required_authority_tier": _required_authority(claim),
                "candidate_methods_considered": [
                    candidate.to_mapping()
                    for candidate in sorted(
                        candidates,
                        key=lambda candidate: _string(candidate.profile.get("method_id")),
                    )
                ],
                "selected_methods": selected_rows,
                "escalation_methods": escalation_rows,
                "status": status,
                "selection_rationale": rationale,
                "next_cheapest_experiment": next_experiment,
                "expected_cost_usd": sum(
                    _number(candidate.profile.get("expected_cost_usd"))
                    for candidate in selected
                ),
                "expected_latency_seconds": sum(
                    _number(candidate.profile.get("expected_latency_seconds"))
                    for candidate in selected
                ),
            }
        )

    plan_value = {
        "schema_version": "evidence_plan.v1",
        "plan_id": _stable_id(_string(request.get("request_id")), "plan"),
        "request_id": request.get("request_id"),
        "decision_id": request.get("decision_id"),
        "request_digest": request.get("request_digest"),
        "testbed_id": testbed.get("testbed_id"),
        "testbed_version": testbed.get("version"),
        "testbed_digest": testbed.get("testbed_digest"),
        "claim_plans": claim_plans,
        "execution_order": execution_order,
        "stop_conditions": [
            "claim_evidence_requirement_satisfied",
            "marginal_experiment_not_justified",
            "budget_exhausted",
        ],
        "escalation_conditions": [
            "invalid_result",
            "excessive_uncertainty",
            "cross_method_disagreement",
            "coverage_below_required",
            "domain_mismatch",
        ],
        "physical_evidence_requests": physical_requests,
        "compiled_evaluation_run_specs": sorted(
            leaf_specs, key=lambda value: _string(value.get("run_id"))
        ),
        "non_evaluation_run_steps": sorted(
            non_leaf_steps, key=lambda value: _string(value.get("step_id"))
        ),
        "budget_status": {
            "max_cost_usd": max_budget,
            "projected_cost_usd": round(projected_cost, 12),
            "within_budget": projected_cost <= max_budget,
            "projected_latency_seconds": round(projected_latency, 12),
        },
        "prohibited_claims": prohibited_claims,
        "shared_dependency_warnings": sorted(
            shared_warnings,
            key=lambda value: (
                _string(value.get("claim_id")), _string(value.get("method_id"))
            ),
        ),
        "router_policy": {
            "deterministic": True,
            "provider_identity_is_qualification": False,
            "visual_realism_is_qualification": False,
            "agreement_is_independence": False,
            "uncalibrated_methods_are_debug_only": True,
            "cross_domain_transfer_enabled": False,
            "policy_ranking_thesis_verdict": "thesis_not_supported",
        },
    }
    return EvidencePlan.from_mapping(plan_value)


def plan_fingerprint(
    request_value: Mapping[str, Any],
    testbed_value: Mapping[str, Any],
    method_values: Sequence[Mapping[str, Any]],
    qualification_values: Sequence[Mapping[str, Any]],
) -> str:
    """Convenience proof that repeated routing inputs serialize identically."""

    return canonical_digest(
        route_decision_evidence(
            request_value, testbed_value, method_values, qualification_values
        ).to_mapping(),
        digest_field="plan_digest",
    )


__all__ = ["plan_fingerprint", "route_decision_evidence"]
