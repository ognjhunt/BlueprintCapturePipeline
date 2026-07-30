"""Capability contracts and deterministic evaluation baselines.

The product path uses OpenAI Agents SDK capabilities. The deterministic
implementations in this module are retained only as reproducible baselines and
fixture oracles for evaluating the agents; they are not the default harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from ..decision_evidence_contracts import (
    DecisionEnvelope,
    DecisionEvidenceRequest,
    MaintainedSiteTaskTestbed,
    NormalizedEvidenceResult,
    canonical_digest,
)
from ..decision_evidence_router import route_decision_evidence
from .contracts import (
    ActionProposal,
    CapabilityKind,
    CapabilityResult,
    ProposalDisposition,
)


@dataclass(frozen=True)
class SupervisorContext:
    run_id: str
    customer_question: str
    capture_build: Mapping[str, Any] | None = None
    decision_request: Mapping[str, Any] | None = None
    testbed: Mapping[str, Any] | None = None
    method_profiles: Sequence[Mapping[str, Any]] = ()
    qualifications: Sequence[Mapping[str, Any]] = ()
    evidence_plan: Mapping[str, Any] | None = None
    evidence_results: Sequence[Mapping[str, Any]] = ()
    decision_envelope: Mapping[str, Any] | None = None
    clarification_request: Mapping[str, Any] | None = None
    clarification_receipt: Mapping[str, Any] | None = None
    targeted_recapture_request: Mapping[str, Any] | None = None
    targeted_recapture_receipt: Mapping[str, Any] | None = None
    recapture_reinspection: Mapping[str, Any] | None = None
    autonomy_mode: str | None = None
    authority_envelope: Mapping[str, Any] | None = None
    # Internal execution scope for deterministic registered tools. This path is
    # never serialized into an agent prompt and cannot be selected by the model.
    supervisor_output_dir: str | None = None
    recovery_controller: Any | None = None


class SupervisorCapability(Protocol):
    kind: CapabilityKind
    adapter_id: str
    adapter_version: str
    instruction: str

    def propose(self, context: SupervisorContext) -> CapabilityResult: ...


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _proposal(
    *,
    context: SupervisorContext,
    capability: CapabilityKind,
    ordinal: int,
    action_type: str,
    reasons: Sequence[str],
    tool_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    evidence_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return ActionProposal.from_mapping(
        {
            "schema_version": "task_evaluation_supervisor_action_proposal.v1",
            "proposal_id": f"{context.run_id}-{capability.value}-{ordinal}",
            "run_id": context.run_id,
            "capability": capability.value,
            "action_type": action_type,
            "tool_id": tool_id,
            "parameters": dict(parameters or {}),
            "reasons": sorted(set(str(reason) for reason in reasons if str(reason))),
            "evidence_refs": [dict(row) for row in evidence_refs],
            "estimated_cost_usd": 0.0,
            "requested_proof_effect": "none",
            "disposition": ProposalDisposition.SHADOW_ONLY.value,
        }
    ).to_mapping()


def _result(
    *,
    context: SupervisorContext,
    capability: CapabilityKind,
    status: str,
    artifact: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]] = (),
    blockers: Sequence[str] = (),
    evidence_refs: Sequence[Mapping[str, Any]] = (),
) -> CapabilityResult:
    return CapabilityResult.from_mapping(
        {
            "schema_version": "task_evaluation_supervisor_capability_result.v1",
            "result_id": f"{context.run_id}-{capability.value}",
            "run_id": context.run_id,
            "capability": capability.value,
            "status": status,
            "artifact": dict(artifact),
            "proposals": [dict(row) for row in proposals],
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "evidence_refs": [dict(row) for row in evidence_refs],
            "authoritative": False,
            "proof_booleans_mutable": False,
            "proof_effect": "none",
        }
    )


class DeterministicClaimTaskInterpreter:
    kind = CapabilityKind.CLAIM_TASK_INTERPRETER
    adapter_id = "deterministic_claim_task_interpreter"
    adapter_version = "1"
    instruction = (
        "Propose a claim graph from customer intent. Never select evidence, set proof, "
        "or invent measurable thresholds. Return clarification when the supplied "
        "DecisionEvidenceRequest is absent or invalid."
    )

    def propose(self, context: SupervisorContext) -> CapabilityResult:
        if context.decision_request is None:
            clarification_response = dict(context.clarification_receipt or {})
            clarification = _proposal(
                context=context,
                capability=self.kind,
                ordinal=0,
                action_type="request_claim_contract_clarification",
                reasons=[
                    (
                        "validated_decision_evidence_request_missing_after_clarification"
                        if clarification_response
                        else "validated_decision_evidence_request_missing"
                    )
                ],
                parameters={
                    "customer_question": context.customer_question,
                    "required_fields": [
                        "testbed_binding",
                        "candidate_scope",
                        "measurable_claims",
                        "false_safe_tolerance",
                        "permitted_abstention",
                    ],
                },
            )
            return _result(
                context=context,
                capability=self.kind,
                status="abstained",
                artifact={
                    "schema_version": "proposed_claim_graph.v1",
                    "customer_question": context.customer_question,
                    "claims": [],
                    "clarification_required": True,
                    "validated_by_deterministic_contract": False,
                    "clarification_response_received": bool(clarification_response),
                    "clarification_response_keys": sorted(
                        str(key) for key in (clarification_response.get("responses") or {})
                    ),
                    "clarification_response_requires_validated_decision_request": bool(
                        clarification_response
                    ),
                    "clarification_response_is_proof": False,
                },
                proposals=[clarification],
                blockers=[
                    (
                        "validated_decision_evidence_request_missing_after_clarification"
                        if clarification_response
                        else "validated_decision_evidence_request_missing"
                    )
                ],
                evidence_refs=(
                    [
                        {
                            "artifact": "clarification_receipt",
                            "digest": clarification_response.get("clarification_receipt_digest"),
                        }
                    ]
                    if clarification_response
                    else []
                ),
            )

        request = DecisionEvidenceRequest.from_mapping(context.decision_request).to_mapping()
        claims = []
        for claim in sorted(
            _rows(request.get("claims")), key=lambda row: _string(row.get("claim_id"))
        ):
            claims.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "claim_type": claim.get("claim_type"),
                    "subject": claim.get("subject"),
                    "measurable_threshold": claim.get("measurable_threshold"),
                    "false_safe_consequence": claim.get("false_safe_consequence"),
                    "acceptable_false_safe_risk": claim.get("acceptable_false_safe_risk"),
                    "permitted_abstention_behavior": claim.get("permitted_abstention_behavior"),
                    "source": "validated_decision_evidence_request",
                    "agent_inferred": False,
                }
            )
        proposal = _proposal(
            context=context,
            capability=self.kind,
            ordinal=0,
            action_type="validate_proposed_claim_graph",
            reasons=["customer_request_has_contract_valid_seed_claims"],
            tool_id="validate_proposed_claim_graph",
            parameters={"request_digest": request["request_digest"]},
            evidence_refs=[
                {"artifact": "decision_evidence_request", "digest": request["request_digest"]}
            ],
        )
        return _result(
            context=context,
            capability=self.kind,
            status="proposed",
            artifact={
                "schema_version": "proposed_claim_graph.v1",
                "customer_question": context.customer_question,
                "claims": claims,
                "clarification_required": False,
                "validated_by_deterministic_contract": True,
                "request_digest": request["request_digest"],
                "agent_may_change_thresholds": False,
            },
            proposals=[proposal],
            evidence_refs=[
                {"artifact": "decision_evidence_request", "digest": request["request_digest"]}
            ],
        )


class DeterministicCaptureTestbedSupervisor:
    kind = CapabilityKind.CAPTURE_TESTBED_SUPERVISOR
    adapter_id = "deterministic_capture_testbed_supervisor"
    adapter_version = "1"
    instruction = (
        "Inspect validated capture/testbed facts and propose the smallest follow-up. "
        "Never infer missing samples, collision quality, calibration, or rights clearance."
    )

    def propose(self, context: SupervisorContext) -> CapabilityResult:
        if context.testbed is None:
            recapture_receipt = dict(context.targeted_recapture_receipt or {})
            return _result(
                context=context,
                capability=self.kind,
                status="blocked",
                artifact={
                    "schema_version": "capture_testbed_inspection.v1",
                    "inspection_completed": False,
                    "targeted_recapture_required": False,
                    "targeted_recapture_received": bool(recapture_receipt),
                    "original_blocker_resolution": recapture_receipt.get(
                        "original_blocker_resolution"
                    ),
                    "recapture_requires_testbed_recompilation": bool(recapture_receipt),
                },
                blockers=["maintained_site_task_testbed_missing"],
                evidence_refs=(
                    [
                        {
                            "artifact": "targeted_recapture_receipt",
                            "digest": recapture_receipt.get("targeted_recapture_receipt_digest"),
                        }
                    ]
                    if recapture_receipt
                    else []
                ),
            )
        testbed = MaintainedSiteTaskTestbed.from_mapping(context.testbed).to_mapping()
        recapture_reinspection = dict(context.recapture_reinspection or {})
        inventory = {
            _string(row.get("evidence_id")) for row in _rows(testbed.get("evidence_inventory"))
        }
        governance = dict(testbed.get("governance") or {})
        governance_blockers = [
            f"governance_not_accepted:{key}"
            for key in ("rights", "consent", "privacy")
            if _string(governance.get(key)).lower() not in {"accepted", "cleared", "approved"}
        ]
        recapture_blockers = (
            [
                "targeted_recapture_not_deterministically_resolved:"
                f"{recapture_reinspection.get('status')}"
            ]
            if recapture_reinspection
            and recapture_reinspection.get("status")
            != "resolved_by_deterministic_testbed_reinspection"
            else []
        )
        claim_types: set[str] = set()
        if context.decision_request is not None:
            request = DecisionEvidenceRequest.from_mapping(context.decision_request).to_mapping()
            claim_types = {_string(row.get("claim_type")) for row in _rows(request.get("claims"))}
        missing: list[str] = []
        if (
            claim_types & {"reachability", "collision_contact"}
            and "metric_geometry" not in inventory
        ):
            missing.append("metric_geometry")
        if "perception_visibility" in claim_types and "captured_rgb_frames" not in inventory:
            missing.append("captured_rgb_frames")
        proposals: list[dict[str, Any]] = []
        if governance_blockers:
            proposals.append(
                _proposal(
                    context=context,
                    capability=self.kind,
                    ordinal=0,
                    action_type="request_governance_resolution",
                    reasons=governance_blockers,
                    parameters={"testbed_digest": testbed["testbed_digest"]},
                )
            )
        elif recapture_blockers:
            proposals.append(
                _proposal(
                    context=context,
                    capability=self.kind,
                    ordinal=0,
                    action_type="request_testbed_rebuild_or_recapture_gap_resolution",
                    reasons=recapture_blockers,
                    parameters={
                        "recapture_reinspection_digest": recapture_reinspection.get(
                            "recapture_reinspection_digest"
                        ),
                        "unresolved_missing_evidence": recapture_reinspection.get(
                            "unresolved_missing_evidence"
                        )
                        or [],
                    },
                )
            )
        elif missing:
            proposals.append(
                _proposal(
                    context=context,
                    capability=self.kind,
                    ordinal=0,
                    action_type="request_targeted_recapture",
                    reasons=[f"required_evidence_missing:{item}" for item in missing],
                    tool_id="propose_targeted_recapture",
                    parameters={
                        "testbed_digest": testbed["testbed_digest"],
                        "missing_evidence": sorted(missing),
                        "full_site_recapture_requested": False,
                    },
                )
            )
        else:
            proposals.append(
                _proposal(
                    context=context,
                    capability=self.kind,
                    ordinal=0,
                    action_type="continue_to_evidence_planning",
                    reasons=["declared_testbed_inventory_covers_seed_claim_inputs"],
                    tool_id="inspect_site_task_testbed",
                    parameters={"testbed_digest": testbed["testbed_digest"]},
                )
            )
        return _result(
            context=context,
            capability=self.kind,
            status="proposed",
            artifact={
                "schema_version": "capture_testbed_inspection.v1",
                "testbed_id": testbed["testbed_id"],
                "testbed_version": testbed["version"],
                "testbed_digest": testbed["testbed_digest"],
                "declared_evidence_inventory": sorted(inventory),
                "governance_blockers": governance_blockers,
                "missing_claim_relevant_evidence": sorted(missing),
                "targeted_recapture_required": bool(missing) and not governance_blockers,
                "full_site_recapture_required": False,
                "raw_capture_truth_rewritten": False,
                "rights_clearance_inferred": False,
                "targeted_recapture_received": bool(context.targeted_recapture_receipt),
                "recapture_reinspection_status": recapture_reinspection.get("status"),
                "recapture_reinspection_digest": recapture_reinspection.get(
                    "recapture_reinspection_digest"
                ),
                "recapture_gap_resolution_claimed_by_agent": False,
            },
            proposals=proposals,
            blockers=[*governance_blockers, *recapture_blockers],
            evidence_refs=[
                {"artifact": "maintained_site_task_testbed", "digest": testbed["testbed_digest"]},
                *(
                    [
                        {
                            "artifact": "recapture_reinspection",
                            "digest": recapture_reinspection.get("recapture_reinspection_digest"),
                        }
                    ]
                    if recapture_reinspection
                    else []
                ),
            ],
        )


class DeterministicEvaluationMethodRouter:
    kind = CapabilityKind.EVALUATION_METHOD_ROUTER
    adapter_id = "deterministic_evaluation_method_router"
    adapter_version = "1"
    instruction = (
        "Invoke Blueprint's deterministic Decision/Evidence router and summarize its plan. "
        "Provider identity and agent preference never qualify a method."
    )

    def propose(self, context: SupervisorContext) -> CapabilityResult:
        missing = []
        if context.decision_request is None:
            missing.append("decision_request")
        if context.testbed is None:
            missing.append("testbed")
        if not context.method_profiles:
            missing.append("method_profiles")
        if not context.qualifications:
            missing.append("qualifications")
        if missing:
            return _result(
                context=context,
                capability=self.kind,
                status="abstained",
                artifact={
                    "schema_version": "evidence_method_selection.v1",
                    "deterministic_plan_compiled": False,
                    "missing_inputs": missing,
                },
                blockers=[f"routing_input_missing:{item}" for item in missing],
            )
        plan = route_decision_evidence(
            context.decision_request or {},
            context.testbed or {},
            context.method_profiles,
            context.qualifications,
        ).to_mapping()
        abstentions = [
            row["claim_id"]
            for row in _rows(plan.get("claim_plans"))
            if row.get("status") == "abstention_planned"
        ]
        proposal = _proposal(
            context=context,
            capability=self.kind,
            ordinal=0,
            action_type="adopt_deterministic_evidence_plan",
            reasons=["plan_compiled_by_qualified_method_router"],
            tool_id="compile_deterministic_evidence_plan",
            parameters={"plan_digest": plan["plan_digest"]},
            evidence_refs=[{"artifact": "evidence_plan", "digest": plan["plan_digest"]}],
        )
        return _result(
            context=context,
            capability=self.kind,
            status="proposed",
            artifact={
                "schema_version": "evidence_method_selection.v1",
                "deterministic_plan_compiled": True,
                "plan": plan,
                "planned_claim_count": len(_rows(plan.get("claim_plans"))),
                "planned_abstention_claim_ids": sorted(abstentions),
                "agent_selected_provider": False,
                "agent_qualified_method": False,
            },
            proposals=[proposal],
            evidence_refs=[{"artifact": "evidence_plan", "digest": plan["plan_digest"]}],
        )


def _failure_type(result: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            _string(result.get("status")),
            *[str(item) for item in result.get("blockers") or []],
            *[str(item) for item in result.get("invalid_rollout_reasons") or []],
        ]
    ).lower()
    rules = (
        ("budget", "budget_exhaustion"),
        ("conflict", "conflicting_evidence"),
        ("capacity", "provider_capacity"),
        ("admission", "infrastructure_admission"),
        ("container", "container_startup"),
        ("checkpoint", "checkpoint_acquisition"),
        ("timeout", "timeout"),
        ("malformed", "malformed_non_scientific_output"),
        ("causal", "invalid_causal_signal"),
        ("incompatible", "permanent_incompatibility"),
        ("invalid", "invalid_scientific_output"),
    )
    return next((kind for token, kind in rules if token in text), "unclassified_failure")


class DeterministicRuntimeFailureRecovery:
    kind = CapabilityKind.RUNTIME_FAILURE_RECOVERY
    adapter_id = "deterministic_runtime_failure_recovery"
    adapter_version = "1"
    instruction = (
        "Classify normalized failures and propose bounded registered recovery actions. "
        "Preserve failures and never retry past budget, scientific validity, or authority."
    )

    def propose(self, context: SupervisorContext) -> CapabilityResult:
        diagnoses: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        for index, raw in enumerate(context.evidence_results):
            result = NormalizedEvidenceResult.from_mapping(raw).to_mapping()
            if result.get("validity") is True:
                continue
            failure = _failure_type(result)
            recovery = {
                "provider_capacity": "request_bounded_retry_or_qualified_host",
                "infrastructure_admission": "preserve_and_request_authorization",
                "container_startup": "inspect_immutable_runtime_then_bounded_retry",
                "checkpoint_acquisition": "verify_digest_and_reuse_qualified_cache",
                "timeout": "stop_or_request_bounded_retry",
                "malformed_non_scientific_output": "regenerate_supporting_artifact",
                "invalid_causal_signal": "preserve_and_abstain",
                "permanent_incompatibility": "stop_as_incompatible",
                "budget_exhaustion": "request_authorization_or_abstain",
                "invalid_scientific_output": "preserve_and_abstain",
                "conflicting_evidence": "preserve_disagreement_and_request_adjudication",
                "unclassified_failure": "request_operator_diagnosis",
            }[failure]
            diagnoses.append(
                {
                    "result_id": result["result_id"],
                    "result_digest": result["result_digest"],
                    "failure_type": failure,
                    "recommended_recovery": recovery,
                    "failed_evidence_preserved": True,
                }
            )
            proposals.append(
                _proposal(
                    context=context,
                    capability=self.kind,
                    ordinal=index,
                    action_type=recovery,
                    reasons=[f"typed_failure:{failure}"],
                    tool_id="inspect_normalized_evidence_results",
                    parameters={
                        "result_digest": result["result_digest"],
                        "execution_requested": False,
                    },
                    evidence_refs=[
                        {
                            "artifact": "normalized_evidence_result",
                            "digest": result["result_digest"],
                        }
                    ],
                )
            )
        status = "proposed" if diagnoses else "abstained"
        return _result(
            context=context,
            capability=self.kind,
            status=status,
            artifact={
                "schema_version": "typed_failure_diagnosis.v1",
                "diagnoses": diagnoses,
                "recovery_executed": False,
                "failed_evidence_suppressed": False,
                "additional_authority_inferred": False,
            },
            proposals=proposals,
            blockers=[] if diagnoses else ["no_failed_normalized_results_supplied"],
            evidence_refs=[
                {"artifact": "normalized_evidence_result", "digest": row["result_digest"]}
                for row in diagnoses
            ],
        )


class DeterministicScenarioAdversarialProposer:
    kind = CapabilityKind.SCENARIO_ADVERSARIAL_PROPOSER
    adapter_id = "deterministic_scenario_adversarial_proposer"
    adapter_version = "1"
    instruction = (
        "Propose task-relevant adversarial scenarios before held-out evaluation. "
        "Never access hidden labels, change frozen scenarios, or react to candidate ranking."
    )

    _SCENARIOS = {
        "perception_visibility": ("occluded_target", "conflicting_sensor_signal"),
        "reachability": ("shifted_object", "restricted_route"),
        "collision_contact": ("moving_person", "occupied_destination"),
        "cycle_time": ("partial_completion", "restricted_route"),
        "comparative_policy_ranking": ("ambiguous_instruction", "failed_grasp"),
        "deployment_readiness": ("tool_unavailable", "moving_person"),
    }

    def propose(self, context: SupervisorContext) -> CapabilityResult:
        if context.decision_request is None:
            return _result(
                context=context,
                capability=self.kind,
                status="abstained",
                artifact={
                    "schema_version": "scenario_proposal_set.v1",
                    "scenarios": [],
                    "generated_before_heldout": True,
                    "frozen": False,
                    "hidden_labels_accessed": False,
                },
                blockers=["validated_claims_missing"],
            )
        request = DecisionEvidenceRequest.from_mapping(context.decision_request).to_mapping()
        proposed: dict[tuple[str, str], dict[str, Any]] = {}
        for claim in _rows(request.get("claims")):
            claim_id = _string(claim.get("claim_id"))
            claim_type = _string(claim.get("claim_type"))
            for scenario_type in self._SCENARIOS.get(claim_type, ("ambiguous_instruction",)):
                proposed[(claim_id, scenario_type)] = {
                    "scenario_id": f"proposal-{claim_id}-{scenario_type}",
                    "claim_id": claim_id,
                    "scenario_type": scenario_type,
                    "failure_mode_target": f"challenge:{claim_type}:{scenario_type}",
                    "accepted_into_frozen_pack": False,
                }
        scenarios = [proposed[key] for key in sorted(proposed)]
        proposal = _proposal(
            context=context,
            capability=self.kind,
            ordinal=0,
            action_type="review_and_freeze_scenario_proposals",
            reasons=["claim_relevant_variations_proposed_before_heldout"],
            tool_id="propose_adversarial_scenarios",
            parameters={
                "request_digest": request["request_digest"],
                "scenarios": scenarios,
                "candidate_results_observed": False,
            },
        )
        return _result(
            context=context,
            capability=self.kind,
            status="proposed",
            artifact={
                "schema_version": "scenario_proposal_set.v1",
                "request_digest": request["request_digest"],
                "scenarios": scenarios,
                "generated_before_heldout": True,
                "frozen": False,
                "hidden_labels_accessed": False,
                "candidate_results_observed": False,
                "agent_may_freeze_scenarios": False,
            },
            proposals=[proposal],
        )


class DeterministicPostRunDiagnostician:
    kind = CapabilityKind.POST_RUN_DIAGNOSTICIAN
    adapter_id = "deterministic_post_run_diagnostician"
    adapter_version = "1"
    instruction = (
        "Explain an already validated DecisionEnvelope. Separate decisive evidence, "
        "correlation, missing proof, and next experiments without changing the verdict."
    )

    def propose(self, context: SupervisorContext) -> CapabilityResult:
        if context.decision_envelope is None:
            return _result(
                context=context,
                capability=self.kind,
                status="abstained",
                artifact={
                    "schema_version": "post_run_diagnosis.v1",
                    "diagnosis_available": False,
                    "deterministic_verdict_changed": False,
                },
                blockers=["decision_envelope_missing"],
            )
        envelope = DecisionEnvelope.from_mapping(context.decision_envelope).to_mapping()
        per_claim = []
        for row in _rows(envelope.get("per_claim_verdicts")):
            per_claim.append(
                {
                    "claim_id": row.get("claim_id"),
                    "verdict": row.get("verdict") or row.get("outcome"),
                    "decisive_evidence": list(row.get("accepted_result_digests") or []),
                    "missing_or_rejected_evidence": list(row.get("rejected_result_digests") or []),
                    "explanatory_only": True,
                }
            )
        proposal = _proposal(
            context=context,
            capability=self.kind,
            ordinal=0,
            action_type="publish_claim_bounded_explanation",
            reasons=["deterministic_decision_envelope_available"],
            tool_id="explain_deterministic_decision",
            parameters={"decision_envelope_digest": envelope["decision_envelope_digest"]},
            evidence_refs=[
                {"artifact": "decision_envelope", "digest": envelope["decision_envelope_digest"]}
            ],
        )
        return _result(
            context=context,
            capability=self.kind,
            status="proposed",
            artifact={
                "schema_version": "post_run_diagnosis.v1",
                "diagnosis_available": True,
                "overall_outcome": envelope["overall_outcome"],
                "per_claim": per_claim,
                "decision_rationale": envelope["decision_rationale"],
                "next_cheapest_experiment": envelope["next_cheapest_experiment"],
                "claim_ceiling": envelope["claim_ceiling"],
                "deployment_approval": False,
                "safety_certification": False,
                "deterministic_verdict_changed": False,
            },
            proposals=[proposal],
            evidence_refs=[
                {"artifact": "decision_envelope", "digest": envelope["decision_envelope_digest"]}
            ],
        )


def deterministic_baseline_capabilities() -> tuple[SupervisorCapability, ...]:
    return (
        DeterministicClaimTaskInterpreter(),
        DeterministicCaptureTestbedSupervisor(),
        DeterministicEvaluationMethodRouter(),
        DeterministicRuntimeFailureRecovery(),
        DeterministicScenarioAdversarialProposer(),
        DeterministicPostRunDiagnostician(),
    )


def capability_instruction_digest(capability: SupervisorCapability) -> str:
    return canonical_digest(
        {
            "adapter_id": capability.adapter_id,
            "adapter_version": capability.adapter_version,
            "instruction": capability.instruction,
        }
    )


__all__ = [
    "DeterministicCaptureTestbedSupervisor",
    "DeterministicClaimTaskInterpreter",
    "DeterministicEvaluationMethodRouter",
    "DeterministicPostRunDiagnostician",
    "DeterministicRuntimeFailureRecovery",
    "DeterministicScenarioAdversarialProposer",
    "SupervisorCapability",
    "SupervisorContext",
    "capability_instruction_digest",
    "deterministic_baseline_capabilities",
]
