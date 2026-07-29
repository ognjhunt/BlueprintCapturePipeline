"""Stable method-adapter execution and Decision Envelope aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .decision_evidence_contracts import (
    DecisionEnvelope,
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    NormalizedEvidenceResult,
    QualificationRecord,
)
from .evaluation_run_contract import validate_evaluation_run_spec


class EvidenceMethodAdapter(Protocol):
    adapter_reference: str

    def execute(
        self,
        *,
        step: Mapping[str, Any],
        claim: Mapping[str, Any],
        request: Mapping[str, Any],
        testbed: Mapping[str, Any],
        method_profile: Mapping[str, Any],
        qualification: Mapping[str, Any],
        evaluation_run_spec: Mapping[str, Any] | None,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class EvidenceMethodAdapterRegistry:
    def __init__(self, adapters: Sequence[EvidenceMethodAdapter] = ()) -> None:
        self._adapters: dict[str, EvidenceMethodAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: EvidenceMethodAdapter) -> None:
        reference = str(getattr(adapter, "adapter_reference", "") or "").strip()
        if not reference:
            raise ValueError("evidence_method_adapter_reference_missing")
        if reference in self._adapters:
            raise ValueError(f"duplicate_evidence_method_adapter:{reference}")
        self._adapters[reference] = adapter

    def resolve(self, reference: str) -> EvidenceMethodAdapter | None:
        return self._adapters.get(str(reference or "").strip())

    def manifest(self) -> list[str]:
        return sorted(self._adapters)


@dataclass(frozen=True)
class EvidenceExecution:
    results: tuple[NormalizedEvidenceResult, ...]
    execution_manifest: Mapping[str, Any]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item) for item in value if _string(item)]


def _leaf_by_claim_and_method(plan: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    leaves: dict[tuple[str, str], dict[str, Any]] = {}
    for leaf in _rows(plan.get("compiled_evaluation_run_specs")):
        metadata = leaf.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        key = (_string(metadata.get("claim_id")), _string(metadata.get("method_profile_digest")))
        leaves[key] = leaf
    return leaves


def _normalize(
    raw: Mapping[str, Any],
    *,
    step: Mapping[str, Any],
    claim: Mapping[str, Any],
    request: Mapping[str, Any],
    testbed: Mapping[str, Any],
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    qualification: Mapping[str, Any],
    leaf: Mapping[str, Any] | None,
) -> NormalizedEvidenceResult:
    status = _string(raw.get("status")) or "invalid"
    if status not in {
        "valid",
        "invalid",
        "uncertain",
        "contradictory",
        "unavailable",
        "evidence_requested",
    }:
        status = "invalid"
    blockers = _strings(raw.get("blockers"))
    invalid_reasons = _strings(raw.get("invalid_rollout_reasons"))
    if status == "invalid" and not invalid_reasons:
        invalid_reasons = ["adapter_result_invalid"]
    leaf_digest = None
    leaf_run_id = None
    if leaf is not None:
        validation = validate_evaluation_run_spec(leaf)
        leaf_digest = validation.get("spec_digest")
        leaf_run_id = leaf.get("run_id")
    family = _string(profile.get("method_family"))
    raw_ceiling = dict(raw.get("claim_ceiling") or {}) if isinstance(
        raw.get("claim_ceiling"), Mapping
    ) else {}
    result = {
        "schema_version": "normalized_evidence_result.v1",
        "result_id": f"result-{step['step_id']}",
        "request_id": request.get("request_id"),
        "request_digest": request.get("request_digest"),
        "plan_id": plan.get("plan_id"),
        "plan_digest": plan.get("plan_digest"),
        "claim_id": claim.get("claim_id"),
        "testbed_id": testbed.get("testbed_id"),
        "testbed_version": testbed.get("version"),
        "testbed_digest": testbed.get("testbed_digest"),
        "method_profile_snapshot": profile,
        "method_profile_digest": profile.get("method_profile_digest"),
        "qualification_digest": qualification.get("qualification_digest"),
        "evaluation_run_id": leaf_run_id,
        "evaluation_run_spec_digest": leaf_digest,
        "status": status,
        "validity": status == "valid" and not blockers and not invalid_reasons,
        "observed_value": raw.get("observed_value"),
        "categorical_finding": raw.get("categorical_finding"),
        "supports_claim": raw.get("supports_claim"),
        "uncertainty": _number(raw.get("uncertainty"), 1.0),
        "coverage": _number(raw.get("coverage"), 0.0),
        "applicability_envelope": dict(
            raw.get("applicability_envelope")
            if isinstance(raw.get("applicability_envelope"), Mapping)
            else profile.get("applicability_envelope") or {}
        ),
        "raw_artifact_references": _rows(raw.get("raw_artifact_references")),
        "provenance": {
            **dict(raw.get("provenance") if isinstance(raw.get("provenance"), Mapping) else {}),
            "adapter_reference": profile.get("adapter_reference"),
            "implementation_digest": profile.get("implementation_digest"),
            "qualification_digest": qualification.get("qualification_digest"),
            "raw_capture_authority_rewritten": False,
        },
        "cost_usd": _number(raw.get("cost_usd"), _number(profile.get("expected_cost_usd"))),
        "duration_seconds": _number(
            raw.get("duration_seconds"), _number(profile.get("expected_latency_seconds"))
        ),
        "blockers": blockers,
        "invalid_rollout_reasons": invalid_reasons,
        "claim_ceiling": {
            **raw_ceiling,
            "method_family": family,
            "authority_tier": profile.get("authority_tier"),
            "physical_success": family == "physical_evidence"
            and raw_ceiling.get("physical_success") is True,
            "deployment_readiness": False,
            "safety_certification": False,
            "generated_artifact_upgrades_raw_or_physical_claim": False,
        },
        "false_safe_risk": _number(
            raw.get("false_safe_risk"), _number(qualification.get("false_safe_rate"), 1.0)
        ),
        "false_reject_estimate": _number(
            raw.get("false_reject_estimate"),
            _number(qualification.get("false_reject_rate"), 1.0),
        ),
        "raw_policy_values_persisted": False,
        "raw_secret_values_persisted": False,
    }
    return NormalizedEvidenceResult.from_mapping(result)


def execute_evidence_plan(
    plan_value: Mapping[str, Any],
    request_value: Mapping[str, Any],
    testbed_value: Mapping[str, Any],
    method_values: Sequence[Mapping[str, Any]],
    qualification_values: Sequence[Mapping[str, Any]],
    *,
    registry: EvidenceMethodAdapterRegistry,
    context: Mapping[str, Any] | None = None,
) -> EvidenceExecution:
    """Execute selected plan steps through registered, stable method ports.

    The caller supplies adapters explicitly. This function never discovers or
    launches providers from defaults and never initiates a physical robot run.
    """

    plan = EvidencePlan.from_mapping(plan_value).to_mapping()
    request = DecisionEvidenceRequest.from_mapping(request_value).to_mapping()
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    if plan["request_digest"] != request["request_digest"]:
        raise ValueError("evidence_plan_request_digest_mismatch")
    if plan["testbed_digest"] != testbed["testbed_digest"]:
        raise ValueError("evidence_plan_testbed_digest_mismatch")
    profiles = {
        profile.digest: profile.to_mapping()
        for profile in (EvidenceMethodProfile.from_mapping(value) for value in method_values)
    }
    qualifications = {
        qualification.digest: qualification.to_mapping()
        for qualification in (
            QualificationRecord.from_mapping(value) for value in qualification_values
        )
    }
    claims = {_string(row.get("claim_id")): row for row in _rows(request.get("claims"))}
    leaf_specs = _leaf_by_claim_and_method(plan)
    selected_steps: dict[str, dict[str, Any]] = {}
    for claim_plan in _rows(plan.get("claim_plans")):
        for step in _rows(claim_plan.get("selected_methods")) + _rows(
            claim_plan.get("escalation_methods")
        ):
            selected_steps[_string(step.get("step_id"))] = step

    results: list[NormalizedEvidenceResult] = []
    execution_rows: list[dict[str, Any]] = []
    ephemeral_context = dict(context or {})
    claim_has_sufficient_result: dict[str, bool] = {}
    for step_id in _strings(plan.get("execution_order")):
        step = selected_steps.get(step_id)
        if step is None:
            raise ValueError(f"evidence_plan_execution_step_missing:{step_id}")
        profile = profiles.get(_string(step.get("method_profile_digest")))
        qualification = qualifications.get(_string(step.get("qualification_digest")))
        if profile is None or qualification is None:
            raise ValueError(f"evidence_plan_binding_missing:{step_id}")
        claim = claims.get(_string(step.get("claim_id")))
        if claim is None:
            raise ValueError(f"evidence_plan_claim_missing:{step_id}")
        claim_id = _string(claim.get("claim_id"))
        if _string(step.get("execution_role")) == "conditional_escalation" and claim_has_sufficient_result.get(
            claim_id, False
        ):
            execution_rows.append(
                {
                    "step_id": step_id,
                    "result_digest": None,
                    "status": "skipped_evidence_already_sufficient",
                    "adapter_reference": profile.get("adapter_reference"),
                }
            )
            continue
        adapter = registry.resolve(_string(profile.get("adapter_reference")))
        leaf = leaf_specs.get(
            (_string(claim.get("claim_id")), _string(profile.get("method_profile_digest")))
        )
        if _string(profile.get("method_family")) == "physical_evidence" and (
            adapter is None or getattr(adapter, "physical_evidence_mode", None) != "read_only"
        ):
            raw: Mapping[str, Any] = {
                "status": "evidence_requested",
                "supports_claim": None,
                "uncertainty": 1.0,
                "coverage": 0.0,
                "blockers": ["physical_evidence_execution_not_permitted"],
                "invalid_rollout_reasons": [],
                "raw_artifact_references": [],
                "provenance": {"physical_robot_run_initiated": False},
            }
        elif adapter is None:
            raw = {
                "status": "unavailable",
                "supports_claim": None,
                "uncertainty": 1.0,
                "coverage": 0.0,
                "blockers": ["evidence_method_adapter_unavailable"],
                "invalid_rollout_reasons": [],
                "raw_artifact_references": [],
                "provenance": {},
            }
        else:
            raw = adapter.execute(
                step=step,
                claim=claim,
                request=request,
                testbed=testbed,
                method_profile=profile,
                qualification=qualification,
                evaluation_run_spec=leaf,
                context=ephemeral_context,
            )
        result = _normalize(
            raw,
            step=step,
            claim=claim,
            request=request,
            testbed=testbed,
            plan=plan,
            profile=profile,
            qualification=qualification,
            leaf=leaf,
        )
        results.append(result)
        result_mapping = result.to_mapping()
        desired = dict(claim.get("desired_confidence_or_coverage") or {})
        claim_has_sufficient_result[claim_id] = (
            result_mapping["validity"] is True
            and _number(result_mapping.get("coverage"))
            >= _number(desired.get("minimum_coverage"), 0.0)
            and _number(result_mapping.get("false_safe_risk"), 1.0)
            <= _number(claim.get("acceptable_false_safe_risk"), 0.0)
        )
        execution_rows.append(
            {
                "step_id": step_id,
                "result_digest": result.digest,
                "status": result.to_mapping()["status"],
                "adapter_reference": profile.get("adapter_reference"),
            }
        )

    manifest = {
        "schema_version": "evidence_plan_execution.v1",
        "plan_id": plan.get("plan_id"),
        "plan_digest": plan.get("plan_digest"),
        "request_digest": request.get("request_digest"),
        "testbed_digest": testbed.get("testbed_digest"),
        "status": "completed"
        if all(
            row["status"] in {"valid", "skipped_evidence_already_sufficient"}
            for row in execution_rows
        )
        else "completed_with_abstentions",
        "steps": execution_rows,
        "registered_adapters": registry.manifest(),
        "context_keys_supplied": sorted(str(key) for key in ephemeral_context),
        "context_values_persisted": False,
        "provider_discovery_from_defaults": False,
        "physical_robot_run_initiated": False,
    }
    return EvidenceExecution(results=tuple(results), execution_manifest=manifest)


def _find_disagreement(results: Sequence[Mapping[str, Any]]) -> bool:
    valid = [row for row in results if row.get("validity") is True]
    if len(valid) < 2:
        return False
    support = {row.get("supports_claim") for row in valid if row.get("supports_claim") is not None}
    if len(support) > 1:
        return True
    categorical = {
        _string(row.get("categorical_finding"))
        for row in valid
        if _string(row.get("categorical_finding"))
    }
    return len(categorical) > 1


def build_decision_envelope(
    request_value: Mapping[str, Any],
    testbed_value: Mapping[str, Any],
    plan_value: Mapping[str, Any],
    result_values: Sequence[Mapping[str, Any]],
) -> DecisionEnvelope:
    """Aggregate normalized evidence into a decision or explicit abstention."""

    request = DecisionEvidenceRequest.from_mapping(request_value).to_mapping()
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    plan = EvidencePlan.from_mapping(plan_value).to_mapping()
    results = [NormalizedEvidenceResult.from_mapping(value).to_mapping() for value in result_values]
    for result in results:
        if result["request_digest"] != request["request_digest"]:
            raise ValueError("evidence_result_request_digest_mismatch")
        if result["plan_digest"] != plan["plan_digest"]:
            raise ValueError("evidence_result_plan_digest_mismatch")
        if result["testbed_digest"] != testbed["testbed_digest"]:
            raise ValueError("evidence_result_testbed_digest_mismatch")

    by_claim: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_claim.setdefault(_string(result.get("claim_id")), []).append(result)
    plan_by_claim = {
        _string(row.get("claim_id")): row for row in _rows(plan.get("claim_plans"))
    }
    verdicts: list[dict[str, Any]] = []
    accepted: list[str] = []
    rejected: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    physical_required: list[dict[str, Any]] = list(plan.get("physical_evidence_requests") or [])
    next_experiments: list[str] = []
    false_safe_risks: list[float] = []
    false_reject_estimates: list[float] = []
    coverages: list[float] = []

    for claim in sorted(_rows(request.get("claims")), key=lambda row: _string(row.get("claim_id"))):
        claim_id = _string(claim.get("claim_id"))
        claim_results = sorted(
            by_claim.get(claim_id, []), key=lambda row: _string(row.get("result_digest"))
        )
        desired = claim.get("desired_confidence_or_coverage")
        desired = dict(desired) if isinstance(desired, Mapping) else {}
        minimum_coverage = _number(desired.get("minimum_coverage"), 0.0)
        valid = [
            row
            for row in claim_results
            if row.get("validity") is True
            and _number(row.get("coverage")) >= minimum_coverage
            and _number(row.get("false_safe_risk"), 1.0)
            <= _number(claim.get("acceptable_false_safe_risk"), 0.0)
        ]
        for row in valid:
            accepted.append(_string(row.get("result_digest")))
            false_safe_risks.append(_number(row.get("false_safe_risk")))
            false_reject_estimates.append(_number(row.get("false_reject_estimate"), 1.0))
            coverages.append(_number(row.get("coverage")))
        for row in claim_results:
            if row not in valid:
                rejected.append(
                    {
                        "result_digest": row.get("result_digest"),
                        "claim_id": claim_id,
                        "status": row.get("status"),
                        "blockers": row.get("blockers"),
                    }
                )
        disagreement = _find_disagreement(valid)
        if disagreement:
            disagreements.append(
                {
                    "claim_id": claim_id,
                    "result_digests": sorted(_string(row.get("result_digest")) for row in valid),
                    "resolution": "abstain_and_escalate",
                }
            )
        claim_plan = plan_by_claim.get(claim_id, {})
        if disagreement or not valid:
            verdict = "abstention"
            rationale = "cross_method_disagreement" if disagreement else "qualified_evidence_missing_or_insufficient"
            next_experiment = _string(claim_plan.get("next_cheapest_experiment")) or "collect_more_qualified_evidence"
            next_experiments.append(next_experiment)
        else:
            support = {row.get("supports_claim") for row in valid}
            if support == {True}:
                verdict = "supported"
                rationale = "qualified_evidence_satisfies_claim"
            elif support == {False}:
                verdict = "not_supported"
                rationale = "qualified_evidence_contradicts_claim"
            else:
                verdict = "abstention"
                rationale = "evidence_does_not_resolve_claim"
                next_experiment = _string(claim_plan.get("next_cheapest_experiment")) or "collect_disambiguating_evidence"
                next_experiments.append(next_experiment)
        verdicts.append(
            {
                "claim_id": claim_id,
                "claim_type": claim.get("claim_type"),
                "verdict": verdict,
                "rationale": rationale,
                "accepted_result_digests": sorted(
                    _string(row.get("result_digest")) for row in valid
                ),
                "claim_ceiling": {
                    "physical_success": _string(claim.get("claim_type")) == "physical_task_success"
                    and any(
                        row.get("method_profile_snapshot", {}).get("method_family")
                        == "physical_evidence"
                        for row in valid
                    ),
                    "deployment_readiness": False,
                    "safety_certification": False,
                },
            }
        )

    decided = sum(row["verdict"] != "abstention" for row in verdicts)
    abstained = len(verdicts) - decided
    if decided == 0:
        overall = "abstention"
    elif abstained:
        overall = "partial_decision"
    else:
        overall = "decision"
    next_cheapest = sorted(set(next_experiments))[0] if next_experiments else "none_required"
    envelope = {
        "schema_version": "decision_envelope.v1",
        "decision_id": request.get("decision_id"),
        "request_id": request.get("request_id"),
        "request_digest": request.get("request_digest"),
        "plan_digest": plan.get("plan_digest"),
        "testbed_digest": testbed.get("testbed_digest"),
        "decision_question": request.get("decision_question"),
        "overall_outcome": overall,
        "per_claim_verdicts": verdicts,
        "evidence_accepted": sorted(set(accepted)),
        "evidence_rejected": rejected,
        "validation_envelope": testbed.get("validation_envelope"),
        "unsupported_conditions": list(testbed.get("known_unsupported_conditions") or []),
        "uncertainty": {
            "maximum": max((_number(row.get("uncertainty")) for row in results), default=1.0),
            "ranking_science_boundary": "thesis_not_supported",
        },
        "severity_weighted_false_safe_risk": max(false_safe_risks, default=1.0),
        "false_reject_estimate": max(false_reject_estimates)
        if false_reject_estimates
        else None,
        "evidence_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "abstention_rate": abstained / len(verdicts) if verdicts else 1.0,
        "cross_method_disagreements": disagreements,
        "shared_dependency_warnings": list(plan.get("shared_dependency_warnings") or []),
        "claim_ceiling": {
            "task_evaluation_run_decision": overall != "abstention",
            "physical_success": any(
                row["claim_type"] == "physical_task_success" and row["verdict"] == "supported"
                for row in verdicts
            ),
            "deployment_readiness": False,
            "safety_certification": False,
            "generated_artifact_upgrades_raw_or_physical_claim": False,
        },
        "decision_rationale": "claim_level_qualified_evidence_with_fail_closed_abstention",
        "next_cheapest_experiment": next_cheapest,
        "physical_evidence_still_required": physical_required,
        "input_run_result_testbed_digests": sorted(
            {
                request.get("request_digest"),
                plan.get("plan_digest"),
                testbed.get("testbed_digest"),
                *(_string(row.get("result_digest")) for row in results),
                *(
                    _string(row.get("evaluation_run_spec_digest"))
                    for row in results
                    if _string(row.get("evaluation_run_spec_digest"))
                ),
            }
        ),
        "deployment_approval": False,
        "safety_certification": False,
        "raw_policy_values_persisted": False,
        "raw_secret_values_persisted": False,
    }
    return DecisionEnvelope.from_mapping(envelope)


__all__ = [
    "EvidenceExecution",
    "EvidenceMethodAdapter",
    "EvidenceMethodAdapterRegistry",
    "build_decision_envelope",
    "execute_evidence_plan",
]
