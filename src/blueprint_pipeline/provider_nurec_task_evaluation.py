"""Compile a live provider NuRec receipt into a bounded Task Evaluation Run.

The compiler reuses already-paid, digest-bound Isaac evidence where it is
qualified and routes every larger claim independently. In particular, a
successful scene render, point-contact probe, or robot-depth render cannot
become formal placement, articulated policy execution, comparative ranking,
or physical task success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
    canonical_digest,
    canonical_json,
)
from .decision_evidence_execution import build_decision_envelope, execute_evidence_plan
from .decision_evidence_router import route_decision_evidence
from .external_provider_nurec import (
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_runtime_result,
)
from .local_evidence_adapters import (
    SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER,
    SIGNED_ISAAC_POINT_CONTACT_ADAPTER,
    SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER,
    SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER,
    authorized_local_evidence_adapter_registry,
)


SCHEMA_VERSION = "provider_nurec_task_evaluation.v1"
TASK_FAMILY = "fixed_base_scene_inspection"


class ProviderNuRecTaskEvaluationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ProviderNuRecTaskEvaluationError(["artifact_not_json_serializable"]) from exc
    return cloned


def _validate_digest_artifact(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    digest_field: str,
    status_field: str | None = None,
    accepted_status: str | None = None,
) -> dict[str, Any]:
    artifact = _clone(value)
    errors: list[str] = []
    if artifact.get("schema_version") != schema_version:
        errors.append(f"{digest_field}_schema_invalid")
    if artifact.get(digest_field) != canonical_digest(artifact, digest_field=digest_field):
        errors.append(f"{digest_field}_mismatch")
    if status_field and artifact.get(status_field) != accepted_status:
        errors.append(f"{digest_field}_status_invalid")
    if errors:
        raise ProviderNuRecTaskEvaluationError(errors)
    return artifact


def _validated_policy_trace_pair(runtime: Mapping[str, Any]) -> dict[str, Any] | None:
    value = runtime.get("articulated_policy_trace_pair")
    if not isinstance(value, Mapping) or value.get("requested") is not True:
        return None
    pair = _clone(value)
    errors: list[str] = []
    if pair.get("schema_version") != "franka_articulated_policy_trace_pair.v1":
        errors.append("articulated_policy_trace_pair_schema_invalid")
    if pair.get("status") != "completed" or pair.get("blockers") not in ([], (None,)):
        errors.append("articulated_policy_trace_pair_not_completed")
    if pair.get("robot_id") != "franka_panda":
        errors.append("articulated_policy_trace_pair_robot_invalid")
    controller_id = pair.get("controller_id")
    if controller_id not in {
        "deterministic_franka_joint_position_pair.v1",
        "deterministic_franka_inspection_cohort.v1",
    }:
        errors.append("articulated_policy_trace_pair_controller_invalid")
    if pair.get("articulated_policy_execution_observed") is not True:
        errors.append("articulated_policy_trace_execution_not_observed")
    if pair.get("comparative_policy_ranking_proven") is not False:
        errors.append("articulated_policy_trace_ranking_boundary_invalid")
    if pair.get("physical_success_claimed") is not False:
        errors.append("articulated_policy_trace_physical_boundary_invalid")
    if controller_id == "deterministic_franka_inspection_cohort.v1" and not str(
        pair.get("inspection_outcome_contract_digest") or ""
    ).startswith("sha256:"):
        errors.append("articulated_policy_trace_inspection_contract_missing")
    if pair.get("articulated_policy_trace_pair_digest") != canonical_digest(
        pair, digest_field="articulated_policy_trace_pair_digest"
    ):
        errors.append("articulated_policy_trace_pair_digest_mismatch")
    traces = pair.get("candidate_traces")
    expected_ids = (
        ["franka-fixed-hold-v1", "franka-inspection-sweep-v1"]
        if controller_id == "deterministic_franka_joint_position_pair.v1"
        else [
            "franka-inspection-center-hold-v1",
            "franka-inspection-left-narrow-v1",
            "franka-inspection-right-narrow-v1",
            "franka-inspection-left-wide-v1",
            "franka-inspection-right-wide-v1",
        ]
    )
    if (
        not isinstance(traces, list)
        or len(traces) != len(expected_ids)
        or [row.get("policy_id") for row in traces if isinstance(row, Mapping)] != expected_ids
    ):
        errors.append("articulated_policy_trace_candidates_invalid")
        traces = []
    for index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            errors.append(f"articulated_policy_trace_{index}_invalid")
            continue
        if trace.get("status") != "completed" or trace.get(
            "policy_trace_digest"
        ) != canonical_digest(trace, digest_field="policy_trace_digest"):
            errors.append(f"articulated_policy_trace_{index}_digest_or_status_invalid")
        if not isinstance(trace.get("samples"), list) or not trace.get("samples"):
            errors.append(f"articulated_policy_trace_{index}_samples_missing")
        observation = trace.get("egocentric_observation")
        if (
            not isinstance(observation, Mapping)
            or observation.get("robot_relative_mount") is not True
            or observation.get("nonblank") is not True
            or not str(observation.get("artifact_reference") or "")
            or not str(observation.get("digest") or "").startswith("sha256:")
        ):
            errors.append(f"articulated_policy_trace_{index}_egocentric_observation_invalid")
        if controller_id == "deterministic_franka_inspection_cohort.v1":
            target_views = trace.get("target_view_observations")
            if (
                trace.get("action_source") != "scripted_controller"
                or trace.get("contact_reporting_active") is not True
                or not isinstance(trace.get("collision_free_observed"), bool)
                or not isinstance(target_views, list)
                or not target_views
                or any(not isinstance(row, Mapping) for row in target_views)
            ):
                errors.append(f"articulated_policy_trace_{index}_inspection_evidence_invalid")
    assessment = pair.get("trace_pair_assessment")
    if not isinstance(assessment, Mapping):
        errors.append("articulated_policy_trace_pair_assessment_missing")
    else:
        if (
            assessment.get("status") != "completed"
            or assessment.get("distinct") is not True
            or assessment.get("identical_frozen_start_observed") is not True
            or assessment.get("robot_relative_egocentric_camera") is not True
            or assessment.get("trace_pair_digest")
            != canonical_digest(assessment, digest_field="trace_pair_digest")
        ):
            errors.append("articulated_policy_trace_pair_assessment_invalid")
    if errors:
        raise ProviderNuRecTaskEvaluationError(errors)
    return pair


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ProviderNuRecTaskEvaluationError([f"immutable_artifact_conflict:{path.name}"])


def _claim(
    claim_id: str,
    claim_type: str,
    *,
    subject: Any,
    scope: Mapping[str, Any],
    consequence: str = "moderate",
    risk: float = 0.05,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "subject": subject,
        "measurable_threshold": {"operator": "==", "value": True, "units": "boolean"},
        "false_safe_consequence": consequence,
        "acceptable_false_safe_risk": risk,
        "desired_confidence_or_coverage": {
            "minimum_coverage": 1.0,
            "minimum_independent_methods": 1,
        },
        "permitted_abstention_behavior": {"allowed": True},
        **dict(scope),
    }


def _method_profiles(
    *, testbed: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bindings = testbed["robot_sensor_controller_bindings"]
    source_bundle = testbed["source_capture_bundles"][0]
    bundle_id = str(source_bundle["bundle_id"])
    scene_entrypoint = str(
        testbed["supported_condition_ranges"].get("scene_entrypoint") or "scene.usdz"
    )
    task_id = str(testbed["task_distribution"]["tasks"][0])
    task_family = str(testbed["task_distribution"]["task_family"])

    def replay_leaf(method_id: str, required_evidence: str) -> dict[str, Any]:
        return {
            "schema_version": "evaluation_run.v1",
            "run_id": "router-replaces-this-id",
            "mode": "evaluate",
            "scene_bundle": {
                "adapter_id": "openusd_scene_bundle",
                "adapter_version": "1",
                "bundle_id": bundle_id,
                "uri": f"artifact://{bundle_id}",
                "entrypoint": scene_entrypoint,
                "content_digest": testbed["source_capture_bundles"][0]["digest"],
            },
            "robot_adapter": {
                "adapter_id": "robot_profile_adapter",
                "adapter_version": "1",
                "robot_profile_id": "franka_panda",
                "asset_ref": str(bindings["embodiment"]["robot_asset"]),
            },
            "task_scenario_pack": {
                "adapter_id": "manifest_task_scenario_pack",
                "adapter_version": "1",
                "pack_id": f"{task_id}-exact-scene-pack",
                "tasks": [{"task_id": task_id}],
                "scenarios": [
                    {
                        "scenario_id": "exact-signed-run-replay",
                        "task_id": task_id,
                    }
                ],
            },
            "policy_adapter": {
                "adapter_id": "robot_eval_policy_package",
                "adapter_version": "1",
                "policy_id": "no-policy-static-evidence-replay",
                "observation_schema_ref": "signed_isaac_runtime_result.v1",
                "action_schema_ref": "no_action_replay.v1",
            },
            "runtime_provider_profile": {
                "adapter_id": "robot_eval_runtime_provider",
                "adapter_version": "1",
                "profile_id": method_id,
                "providers": ["local_digest_bound_replay"],
                "simulator": "isaac_sim",
                "max_spend_usd": 0.0,
            },
            "proof_contract": {
                "adapter_id": "robot_eval_proof_contract",
                "adapter_version": "1",
                "contract_id": f"{method_id}-proof",
                "required_evidence": [required_evidence],
                "claim_ceiling": {"level": "sim_observation_only"},
                "prohibited_claims": [
                    "formal_robot_placement",
                    "kinematic_feasibility",
                    "policy_execution",
                    "physical_success",
                    "deployment_readiness",
                ],
            },
            "metadata": {"live_provider_execution": False},
        }

    def trace_replay_leaf() -> dict[str, Any]:
        leaf = replay_leaf(
            "signed-isaac-policy-trace-pair-replay",
            "signed_isaac_articulated_policy_trace_pair",
        )
        leaf["policy_adapter"] = {
            "adapter_id": "robot_eval_policy_package",
            "adapter_version": "1",
            "policy_id": "franka-frozen-candidate-pair-v1",
            "observation_schema_ref": "franka_articulated_policy_trace_pair.v1",
            "action_schema_ref": "deterministic_franka_joint_position_pair.v1",
        }
        leaf["proof_contract"] = {
            "adapter_id": "robot_eval_proof_contract",
            "adapter_version": "1",
            "contract_id": "signed-isaac-policy-trace-pair-replay-proof",
            "required_evidence": ["signed_isaac_articulated_policy_trace_pair"],
            "claim_ceiling": {"level": "simulated_policy_trace_distinguishability_only"},
            "prohibited_claims": [
                "comparative_policy_ranking",
                "metric_task_success",
                "physical_success",
                "deployment_readiness",
            ],
        }
        return leaf

    def inspection_ranking_replay_leaf() -> dict[str, Any]:
        leaf = replay_leaf(
            "signed-isaac-inspection-ranking-replay",
            "signed_isaac_inspection_candidate_ranking",
        )
        leaf["policy_adapter"] = {
            "adapter_id": "robot_eval_policy_package",
            "adapter_version": "1",
            "policy_id": "franka-scripted-inspection-controller-cohort-v1",
            "observation_schema_ref": "external_scene_inspection_candidate_ranking.v2",
            "action_schema_ref": "deterministic_franka_inspection_cohort.v1",
        }
        leaf["proof_contract"] = {
            "adapter_id": "robot_eval_proof_contract",
            "adapter_version": "1",
            "contract_id": "signed-isaac-inspection-ranking-replay-proof",
            "required_evidence": ["signed_isaac_inspection_candidate_ranking"],
            "claim_ceiling": {"level": "comparative_controller_ranking_only"},
            "prohibited_claims": [
                "comparative_policy_ranking",
                "metric_task_success",
                "physical_success",
                "deployment_readiness",
            ],
        }
        return leaf

    common = {
        "schema_version": "evidence_method_profile.v1",
        "version": "1",
        "applicability_envelope": {
            "testbed_ids": [testbed["testbed_id"]],
            "testbed_versions": [testbed["version"]],
            "task_families": [task_family],
        },
        "calibration_evidence_references": [],
        "constraints": {"external_processing": False, "data_retention_days": 0},
        "expected_latency_seconds": 0.01,
        "reproducibility_level": "digest_bound_replay",
        "failure_modes": ["bound_evidence_missing_or_tampered"],
        "abstention_modes": ["missing_input", "unqualified", "out_of_scope"],
        "disqualifying_conditions": [],
        "self_qualified": False,
    }
    definitions = [
        {
            "method_id": "signed-isaac-visual-placement-replay",
            "implementation_digest": canonical_digest(
                {"adapter": SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER, "version": "1"}
            ),
            "adapter_reference": SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER,
            "method_family": "traditional_simulation",
            "supported_claim_types": ["perception_visibility"],
            "required_inputs": ["signed_isaac_visual_placement"],
            "authority_tier": 1,
            "proof_tier": "isaac_visual_placement_only",
            "correlation_group": "exact-isaac-runtime",
            "shared_dependencies": ["exact-scene-package", "exact-isaac-runtime"],
            "expected_cost_usd": 0.0,
            "provider_availability": {"status": "available"},
            "evaluation_run_template": replay_leaf(
                "signed-isaac-visual-placement-replay",
                "signed_isaac_visual_placement",
            ),
        },
        {
            "method_id": "signed-isaac-point-contact-replay",
            "implementation_digest": canonical_digest(
                {"adapter": SIGNED_ISAAC_POINT_CONTACT_ADAPTER, "version": "1"}
            ),
            "adapter_reference": SIGNED_ISAAC_POINT_CONTACT_ADAPTER,
            "method_family": "traditional_simulation",
            "supported_claim_types": ["collision_contact"],
            "required_inputs": ["signed_isaac_point_contact"],
            "authority_tier": 2,
            "proof_tier": "isaac_single_point_contact_only",
            "correlation_group": "exact-isaac-runtime",
            "shared_dependencies": ["exact-scene-package", "exact-isaac-runtime"],
            "expected_cost_usd": 0.0,
            "provider_availability": {"status": "available"},
            "evaluation_run_template": replay_leaf(
                "signed-isaac-point-contact-replay",
                "signed_isaac_point_contact",
            ),
        },
        {
            "method_id": "signed-isaac-policy-trace-pair-replay",
            "implementation_digest": canonical_digest(
                {"adapter": SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER, "version": "1"}
            ),
            "adapter_reference": SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER,
            "method_family": "traditional_simulation",
            "supported_claim_types": ["simulated_policy_trace_distinguishability"],
            "required_inputs": ["signed_isaac_articulated_policy_trace_pair"],
            "authority_tier": 2,
            "proof_tier": "isaac_articulated_trace_pair_only",
            "correlation_group": "exact-isaac-runtime",
            "shared_dependencies": ["exact-scene-package", "exact-isaac-runtime"],
            "expected_cost_usd": 0.0,
            "provider_availability": {"status": "available"},
            "evaluation_run_template": trace_replay_leaf(),
        },
        {
            "method_id": "signed-isaac-inspection-ranking-replay",
            "implementation_digest": canonical_digest(
                {"adapter": SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER, "version": "1"}
            ),
            "adapter_reference": SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER,
            "method_family": "traditional_simulation",
            "supported_claim_types": ["comparative_controller_ranking"],
            "required_inputs": ["signed_isaac_inspection_candidate_ranking"],
            "authority_tier": 2,
            "proof_tier": "isaac_task_bound_controller_ranking_only",
            "correlation_group": "exact-isaac-runtime",
            "shared_dependencies": ["exact-scene-package", "exact-isaac-runtime"],
            "expected_cost_usd": 0.0,
            "provider_availability": {"status": "available"},
            "evaluation_run_template": inspection_ranking_replay_leaf(),
        },
        {
            "method_id": "analytic-franka-kinematics-candidate",
            "implementation_digest": canonical_digest(
                {"candidate": "analytic-franka-kinematics", "version": "1"}
            ),
            "adapter_reference": "local://analytic-reachability-v1",
            "method_family": "analytic_geometry_kinematics",
            "supported_claim_types": ["kinematic_feasibility"],
            "required_inputs": ["formal_robot_placement", "metric_task_geometry"],
            "authority_tier": 1,
            "proof_tier": "analytic_only",
            "correlation_group": "metric-kinematics",
            "shared_dependencies": ["formal-placement", "metric-task-target"],
            "expected_cost_usd": 0.0,
            "provider_availability": {"status": "available"},
        },
        {
            "method_id": "isaac-articulated-franka-task-candidate",
            "implementation_digest": canonical_digest(
                {"candidate": "isaac-articulated-franka-task", "version": "1"}
            ),
            "adapter_reference": "provider://isaac-articulated-franka-task-v1",
            "method_family": "traditional_simulation",
            "supported_claim_types": [
                "kinematic_feasibility",
                "comparative_policy_ranking",
            ],
            "required_inputs": [
                "formal_robot_placement",
                "metric_task_geometry",
                "articulated_policy_trace_pair",
            ],
            "authority_tier": 3,
            "proof_tier": "sim_only_candidate",
            "correlation_group": "isaac-articulated-runtime",
            "shared_dependencies": ["exact-scene-package", "formal-placement"],
            "expected_cost_usd": 0.15,
            "provider_availability": {"status": "available"},
        },
        {
            "method_id": "learned-world-model-policy-ranking-candidate",
            "implementation_digest": canonical_digest(
                {"candidate": "learned-world-model-ranking", "version": "1"}
            ),
            "adapter_reference": "provider://qualified-world-model-not-bound",
            "method_family": "learned_world_model",
            "supported_claim_types": ["comparative_policy_ranking"],
            "required_inputs": [
                "qualified_site_observations",
                "articulated_policy_trace_pair",
            ],
            "authority_tier": 3,
            "proof_tier": "learned_sim_candidate",
            "correlation_group": "learned-site-model",
            "shared_dependencies": ["exact-scene-package"],
            "expected_cost_usd": 0.3,
            "provider_availability": {"status": "unavailable"},
        },
        {
            "method_id": "accepted-physical-task-outcome",
            "implementation_digest": canonical_digest(
                {"candidate": "accepted-physical-task-outcome", "version": "1"}
            ),
            "adapter_reference": "physical://read-only-outcome-required",
            "method_family": "physical_evidence",
            "supported_claim_types": ["physical_task_success"],
            "required_inputs": ["accepted_physical_outcome"],
            "authority_tier": 4,
            "proof_tier": "physical",
            "correlation_group": "physical-outcome",
            "shared_dependencies": [],
            "expected_cost_usd": 0.0,
            "provider_availability": {"status": "unavailable"},
        },
    ]
    profiles = [
        EvidenceMethodProfile.from_mapping({**common, **definition}).to_mapping()
        for definition in definitions
    ]
    scope = {
        "task_family": task_family,
        "site_domain_conditions": testbed["supported_condition_ranges"],
        "embodiment": bindings["embodiment"],
        "sensors": bindings["sensors"],
        "controller_action_representation": bindings["controller_action_representation"],
    }
    visual, contact, trace_pair_profile, inspection_ranking_profile = profiles[:4]
    visual_evidence = next(
        row
        for row in testbed["evidence_inventory"]
        if row["evidence_id"] == "signed_isaac_visual_placement"
    )
    contact_evidence = next(
        row
        for row in testbed["evidence_inventory"]
        if row["evidence_id"] == "signed_isaac_point_contact"
    )
    trace_pair_evidence = next(
        (
            row
            for row in testbed["evidence_inventory"]
            if row["evidence_id"] == "signed_isaac_articulated_policy_trace_pair"
        ),
        None,
    )
    inspection_ranking_evidence = next(
        (
            row
            for row in testbed["evidence_inventory"]
            if row["evidence_id"] == "signed_isaac_inspection_candidate_ranking"
        ),
        None,
    )
    visual_digest = str(
        visual_evidence.get("visual_placement_evidence_digest")
        or visual_evidence.get("provider_robot_placement_evidence_digest")
        or ""
    )

    def qualification(
        *,
        profile: Mapping[str, Any],
        qualification_id: str,
        claim_type: str,
        evaluator_id: str,
        evaluator_digest: str,
        owner_digest: str,
    ) -> dict[str, Any]:
        return QualificationRecord.from_mapping(
            {
                "schema_version": "evidence_method_qualification.v1",
                "qualification_id": qualification_id,
                "method_id": profile["method_id"],
                "method_version": profile["version"],
                "method_profile_digest": profile["method_profile_digest"],
                "implementation_digest": profile["implementation_digest"],
                "claim_type": claim_type,
                **scope,
                "evaluator": {"evaluator_id": evaluator_id, "version": "1"},
                "evaluator_digest": evaluator_digest,
                "predictions": [{"prediction_id": "exact-bound-run", "value": True}],
                "accepted_real_outcomes": [
                    {
                        "outcome_id": "independent-digital-artifact-verification",
                        "value": True,
                        "physical_robot_outcome": False,
                        "exact_digest_scope_only": True,
                    }
                ],
                "calibration_partition": "heldout",
                "confidence_intervals": {"level": 1.0, "lower": 1.0, "upper": 1.0},
                "coverage": 1.0,
                "abstention_rate": 0.0,
                "false_safe_rate": 0.0,
                "false_reject_rate": 0.0,
                "provenance": {
                    "source": "independent_digest_bound_replay",
                    "exact_testbed_only": True,
                    "physical_outcome": False,
                },
                "owner_evidence": [{"uri": f"artifact://{evaluator_id}", "digest": owner_digest}],
                "status": "qualified",
                "self_grading": False,
                "subject_provider_id": "nvidia-isaac-runtime",
                "evaluator_provider_id": evaluator_id,
            }
        ).to_mapping()

    qualifications = []
    if visual_evidence.get("independently_rehashed") is True and visual_digest.startswith(
        "sha256:"
    ):
        qualifications.append(
            qualification(
                profile=visual,
                qualification_id="qualification-exact-visual-placement-v1",
                claim_type="perception_visibility",
                evaluator_id="blueprint-independent-depth-rehasher",
                evaluator_digest=visual_digest,
                owner_digest=visual_digest,
            )
        )
    qualifications.append(
        qualification(
            profile=contact,
            qualification_id="qualification-exact-point-contact-v1",
            claim_type="collision_contact",
            evaluator_id="blueprint-independent-isaac-qualifier",
            evaluator_digest=contact_evidence["independent_qualification_digest"],
            owner_digest=contact_evidence["independent_qualification_digest"],
        )
    )
    if trace_pair_evidence is not None:
        qualifications.append(
            qualification(
                profile=trace_pair_profile,
                qualification_id="qualification-exact-policy-trace-pair-v1",
                claim_type="simulated_policy_trace_distinguishability",
                evaluator_id="blueprint-independent-trace-pair-validator",
                evaluator_digest=trace_pair_evidence["articulated_policy_trace_pair_digest"],
                owner_digest=trace_pair_evidence["articulated_policy_trace_pair_digest"],
            )
        )
    if inspection_ranking_evidence is not None:
        qualifications.append(
            qualification(
                profile=inspection_ranking_profile,
                qualification_id="qualification-exact-inspection-controller-ranking-v1",
                claim_type="comparative_controller_ranking",
                evaluator_id="blueprint-deterministic-inspection-scorer",
                evaluator_digest=inspection_ranking_evidence["ranking_digest"],
                owner_digest=inspection_ranking_evidence["ranking_digest"],
            )
        )
    return profiles, qualifications


def compile_provider_nurec_task_evaluation(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    independent_qualification: Mapping[str, Any],
    visual_placement_evidence: Mapping[str, Any],
    task_definition: Mapping[str, Any],
    robot_placement_result: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile and execute one evidence-bounded, site-specific base evaluation."""

    request_input = build_provider_nurec_isaac_request(verification_request)
    runtime = build_provider_nurec_isaac_runtime_result(
        runtime_result, verification_request=request_input
    )
    policy_trace_pair = _validated_policy_trace_pair(runtime)
    independent = _validate_digest_artifact(
        independent_qualification,
        schema_version="reconstruction_isaac_independent_qualification.v1",
        digest_field="qualification_digest",
        status_field="status",
        accepted_status="verified_compatibility_only",
    )
    visual = _validate_digest_artifact(
        visual_placement_evidence,
        schema_version="provider_robot_placement_evidence.v1",
        digest_field="provider_robot_placement_evidence_digest",
        status_field="status",
        accepted_status="verified_visual_placement_only",
    )
    task = _validate_digest_artifact(
        task_definition,
        schema_version="provider_nurec_site_task_definition.v1",
        digest_field="approved_task_digest",
    )
    placement = _validate_digest_artifact(
        robot_placement_result,
        schema_version="robot_placement_result.v1",
        digest_field="robot_placement_digest",
        status_field="status",
        accepted_status="abstained",
    )
    errors: list[str] = []
    if runtime.get("status") != "completed":
        errors.append("isaac_runtime_not_completed")
    if independent.get("blockers") not in ([], ()):
        errors.append("independent_qualification_has_blockers")
    if independent.get("isaac_verification_request_digest") != request_input.get(
        "isaac_verification_request_digest"
    ):
        errors.append("independent_qualification_request_mismatch")
    if independent.get("runtime_result_digest") != runtime.get("isaac_runtime_result_digest"):
        errors.append("independent_qualification_runtime_mismatch")
    if independent.get("claim_ceiling") != "isaac_load_render_compatibility":
        errors.append("independent_qualification_ceiling_invalid")
    if visual.get("blockers") not in ([], ()):
        errors.append("visual_placement_has_blockers")
    if visual.get("isaac_verification_request_digest") != request_input.get(
        "isaac_verification_request_digest"
    ):
        errors.append("visual_placement_request_mismatch")
    if visual.get("isaac_runtime_result_digest") != runtime.get("isaac_runtime_result_digest"):
        errors.append("visual_placement_runtime_mismatch")
    if visual.get("package_digest") != request_input.get("package_digest"):
        errors.append("visual_placement_package_mismatch")
    if visual.get("visual_robot_placement_observed") is not True:
        errors.append("visual_robot_placement_not_observed")
    if visual.get("claim_ceiling") != "isaac_visual_robot_placement":
        errors.append("visual_placement_ceiling_invalid")
    if task.get("source_asset", {}).get("asset_digest") != request_input.get("package_digest"):
        errors.append("task_package_mismatch")
    if placement.get("approved_task_digest") != task.get("approved_task_digest"):
        errors.append("placement_task_mismatch")
    robot = runtime.get("robot")
    physics = runtime.get("physics_probe")
    camera_evidence = visual.get("camera_evidence")
    robot_pose = visual.get("robot_pose")
    if not isinstance(robot, Mapping):
        errors.append("isaac_runtime_robot_evidence_missing")
    if not isinstance(physics, Mapping):
        errors.append("isaac_runtime_physics_evidence_missing")
    if not isinstance(camera_evidence, list) or not camera_evidence:
        errors.append("visual_placement_camera_evidence_missing")
    if not isinstance(robot_pose, list) or len(robot_pose) != 4:
        errors.append("visual_placement_robot_pose_invalid")
    if errors:
        raise ProviderNuRecTaskEvaluationError(errors)

    assert isinstance(robot, Mapping)
    assert isinstance(physics, Mapping)
    assert isinstance(camera_evidence, list)
    assert isinstance(robot_pose, list)
    task_digest = task["approved_task_digest"]
    reset_digest = canonical_digest(
        {
            "reset": "reload_exact_provider_package_and_franka_start_pose",
            "package_digest": request_input["package_digest"],
            "robot_pose": robot_pose,
        }
    )
    bindings = {
        "embodiment": {
            "robot_id": "franka_panda",
            "robot_prim_path": robot.get("prim_path"),
            "robot_asset": robot.get("robot_usd"),
        },
        "sensors": {
            "fixed_camera_ids": [row["id"] for row in camera_evidence],
            "robot_relative_camera": (
                {
                    "camera_id": "franka-wrist-egocentric",
                    "parent_link": "panda_hand",
                    "modality": "isaac_rgb",
                    "trace_pair_digest": policy_trace_pair["articulated_policy_trace_pair_digest"],
                }
                if policy_trace_pair is not None
                else None
            ),
            "modality": "isaac_rgb_distance_to_camera",
        },
        "controller_action_representation": {
            "type": "joint_position",
            "controller_id": "deterministic_franka_joint_position_pair.v1",
        },
        "selected_robot_placement": {
            "candidate_id": "franka-at-verified-ground-probe",
            "base_position_site_m": robot_pose[:3],
            "formal_status": placement["status"],
            "visual_status": visual["status"],
            "method_qualification_status": "visual_only",
        },
    }
    testbed = MaintainedSiteTaskTestbed.from_mapping(
        {
            "schema_version": "maintained_site_task_testbed.v1",
            "testbed_id": "public-reference-ethel-franka-inspection",
            "version": "1",
            "predecessor_testbed_digest": None,
            "supersedes": [],
            "source_capture_bundles": [
                {
                    "bundle_id": "ethel-provider-nurec-package",
                    "version": "1",
                    "digest": request_input["package_digest"],
                }
            ],
            "artifact_references": {
                "site_card": {"uri": "artifact://ethel-site", "digest": task_digest},
                "task_cards": [{"uri": "artifact://ethel-task", "digest": task_digest}],
                "scenario_cards": [
                    {
                        "uri": "artifact://ethel-ground-probe-scenario",
                        "digest": request_input["isaac_verification_request_digest"],
                    }
                ],
                "eval_cards": [
                    {
                        "uri": "artifact://ethel-base-eval",
                        "digest": independent["qualification_digest"],
                    }
                ],
                "evaluator": {
                    "uri": "artifact://signed-evidence-router",
                    "digest": visual["provider_robot_placement_evidence_digest"],
                },
                "reset": {"uri": "artifact://ethel-reset", "digest": reset_digest},
            },
            "task_distribution": {
                "task_family": TASK_FAMILY,
                "tasks": [task["task_id"]],
            },
            "supported_condition_ranges": {
                "site_id": task["site_id"],
                "package_digest": request_input["package_digest"],
                "robot_pose_xyzyaw_site": robot_pose,
                "fixed_camera_ids": [row["id"] for row in camera_evidence],
            },
            "robot_sensor_controller_bindings": bindings,
            "governance": {
                "rights": "public_provider_sample_evaluation_only",
                "consent": "not_applicable_public_provider_sample",
                "privacy": "public_reference_asset",
                "revocation": "new_version_invalidates_testbed",
                "allowed_uses": ["internal_evaluation", "public_reference_harness"],
            },
            "evidence_inventory": [
                {
                    "evidence_id": "signed_isaac_visual_placement",
                    "independently_rehashed": True,
                    "provider_robot_placement_evidence_digest": visual[
                        "provider_robot_placement_evidence_digest"
                    ],
                    "camera_evidence": camera_evidence,
                    "exact_view_coverage": 1.0,
                    "formal_robot_placement": False,
                },
                {
                    "evidence_id": "signed_isaac_point_contact",
                    "independently_qualified": True,
                    "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
                    "independent_qualification_digest": independent["qualification_digest"],
                    "contact_event_count": physics["contact_event_count"],
                    "test_body_fell_through_floor": physics["test_body_fell_through_floor"],
                    "probe_scope": "single_precommitted_point",
                },
                {
                    "evidence_id": "metric_stage_semantics",
                    "meters_per_unit": runtime["stage"]["meters_per_unit"],
                    "up_axis": runtime["stage"]["up_axis"],
                    "independent_known_distance_anchor": False,
                },
            ]
            + (
                [
                    {
                        "evidence_id": "signed_isaac_articulated_policy_trace_pair",
                        "independently_validated": True,
                        "articulated_policy_trace_pair_digest": policy_trace_pair[
                            "articulated_policy_trace_pair_digest"
                        ],
                        "policy_trace_request_digest": policy_trace_pair[
                            "policy_trace_request_digest"
                        ],
                        "controller_id": policy_trace_pair["controller_id"],
                        "identical_frozen_start_observed": policy_trace_pair[
                            "trace_pair_assessment"
                        ]["identical_frozen_start_observed"],
                        "distinct": policy_trace_pair["trace_pair_assessment"]["distinct"],
                        "maximum_end_joint_delta_rad": policy_trace_pair["trace_pair_assessment"][
                            "maximum_end_joint_delta_rad"
                        ],
                        "candidate_traces": [
                            {
                                "policy_id": row["policy_id"],
                                "status": row["status"],
                                "policy_trace_digest": row["policy_trace_digest"],
                                "egocentric_observation_digest": row["egocentric_observation"][
                                    "digest"
                                ],
                            }
                            for row in policy_trace_pair["candidate_traces"]
                        ],
                        "comparative_policy_ranking": False,
                        "metric_task_success": False,
                        "physical_task_success": False,
                    }
                ]
                if policy_trace_pair is not None
                else []
            ),
            "validation_envelope": {
                "exact_digest_scope": True,
                "isaac_verification_request_digest": request_input[
                    "isaac_verification_request_digest"
                ],
                "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
                "independent_qualification_digest": independent["qualification_digest"],
                "visual_placement_evidence_digest": visual[
                    "provider_robot_placement_evidence_digest"
                ],
                "formal_robot_placement_digest": placement["robot_placement_digest"],
                "articulated_policy_trace_pair_digest": (
                    policy_trace_pair["articulated_policy_trace_pair_digest"]
                    if policy_trace_pair is not None
                    else None
                ),
            },
            "target_regions": [
                {
                    "region_id": task["target_region_id"],
                    "position_site_m": task["target_position_site_m"],
                    "supporting_frames": [row["id"] for row in camera_evidence],
                    "captured_coverage": 0.0,
                    "simulated_view_coverage": 1.0,
                }
            ],
            "known_unsupported_conditions": [
                "formal_robot_placement",
                "complete_robot_footprint_clearance",
                "kinematic_reachability",
                "comparative_policy_ranking",
                "physical_task_success",
                "deployment_readiness",
            ]
            + ([] if policy_trace_pair is not None else ["articulated_policy_execution"]),
            "invalidation_triggers": [
                "package_digest_change",
                "robot_asset_change",
                "robot_pose_change",
                "camera_change",
                "task_change",
            ],
            "physical_outcome_history_refs": [],
            "lifecycle_state": "active",
        }
    ).to_mapping()
    scope = {
        "task_family": TASK_FAMILY,
        "site_domain_conditions": testbed["supported_condition_ranges"],
        "embodiment": bindings["embodiment"],
        "sensors": bindings["sensors"],
        "controller_action_representation": bindings["controller_action_representation"],
    }
    claims = [
        _claim(
            "exact-sim-robot-visibility",
            "perception_visibility",
            subject={"target_region_id": task["target_region_id"]},
            scope=scope,
        ),
        _claim(
            "exact-sim-point-contact",
            "collision_contact",
            subject={"probe_scope": "single_precommitted_point"},
            scope=scope,
        ),
        _claim(
            "franka-kinematic-feasibility",
            "kinematic_feasibility",
            subject={"target_position_site_m": task["target_position_site_m"]},
            scope=scope,
        ),
        _claim(
            "franka-policy-trace-distinguishability",
            "simulated_policy_trace_distinguishability",
            subject={
                "candidate_policy_ids": [
                    "franka-fixed-hold-v1",
                    "franka-inspection-sweep-v1",
                ],
                "trace_status": (
                    "collected_distinct_pair" if policy_trace_pair is not None else "not_collected"
                ),
            },
            scope=scope,
        ),
        _claim(
            "franka-candidate-policy-ranking",
            "comparative_policy_ranking",
            subject={
                "candidate_policy_ids": [
                    "franka-fixed-hold-v1",
                    "franka-inspection-sweep-v1",
                ],
                "trace_status": (
                    "collected_distinct_pair" if policy_trace_pair is not None else "not_collected"
                ),
            },
            scope=scope,
            consequence="high",
            risk=0.01,
        ),
        _claim(
            "franka-physical-task-success",
            "physical_task_success",
            subject=task["task_id"],
            scope=scope,
            consequence="high",
            risk=0.01,
        ),
    ]
    decision_request = DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": "ethel-franka-base-eval-request-v1",
            "decision_id": "ethel-franka-base-eval-decision-v1",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": (
                "Which claims are supported for the exact Ethel NuRec scene, Franka pose, "
                "and proposed fixed-base inspection task?"
            ),
            "candidates": [
                {
                    "robot_id": "franka_panda",
                    "policy_id": "franka-fixed-hold-v1",
                    "policy_trace_status": (
                        "collected" if policy_trace_pair is not None else "not_collected"
                    ),
                    **(
                        {
                            "policy_trace_digest": policy_trace_pair["candidate_traces"][0][
                                "policy_trace_digest"
                            ]
                        }
                        if policy_trace_pair is not None
                        else {}
                    ),
                },
                {
                    "robot_id": "franka_panda",
                    "policy_id": "franka-inspection-sweep-v1",
                    "policy_trace_status": (
                        "collected" if policy_trace_pair is not None else "not_collected"
                    ),
                    **(
                        {
                            "policy_trace_digest": policy_trace_pair["candidate_traces"][1][
                                "policy_trace_digest"
                            ]
                        }
                        if policy_trace_pair is not None
                        else {}
                    ),
                },
            ],
            "claims": claims,
            "budget": {
                "max_cost_usd": 1.0,
                "max_latency_seconds": 3600.0,
                "delay_cost_per_second": 0.0,
            },
            "deadline": "2026-12-31T00:00:00Z",
            "available_physical_evidence": [],
            "permitted_evidence_methods": [
                "analytic_geometry_kinematics",
                "traditional_simulation",
                "learned_world_model",
                "physical_evidence",
            ],
            "restrictions": {
                "external_processing_allowed": False,
                "max_data_retention_days": 0,
                "live_robot_execution_allowed": False,
            },
            "requested_result_audience": "blueprint_internal_base_evaluation",
            "provenance": {"caller_identity": "pipeline:provider-nurec-task-evaluation"},
            "idempotency_key": "ethel-franka-base-eval-request-v1",
        }
    ).to_mapping()
    profiles, qualifications = _method_profiles(testbed=testbed)
    plan = route_decision_evidence(decision_request, testbed, profiles, qualifications).to_mapping()
    authorized_adapters = [
        SIGNED_ISAAC_POINT_CONTACT_ADAPTER,
        SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER,
    ] + ([SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER] if policy_trace_pair is not None else [])
    authorization = {
        "schema_version": "provider_nurec_task_evaluation_authorization.v1",
        "plan_digest": plan["plan_digest"],
        "authorized_adapter_references": sorted(authorized_adapters),
        "live_provider_execution": False,
        "paid_compute_authorized": False,
        "physical_robot_run_authorized": False,
    }
    authorization["authorization_digest"] = canonical_digest(
        authorization, digest_field="authorization_digest"
    )
    execution = execute_evidence_plan(
        plan,
        decision_request,
        testbed,
        profiles,
        qualifications,
        registry=authorized_local_evidence_adapter_registry(authorized_adapters),
        context={"authorization_digest": authorization["authorization_digest"]},
    )
    results = [result.to_mapping() for result in execution.results]
    decision = build_decision_envelope(decision_request, testbed, plan, results).to_mapping()
    verdicts = {row["claim_id"]: row["verdict"] for row in decision["per_claim_verdicts"]}
    if (
        decision["overall_outcome"] != "partial_decision"
        or verdicts.get("exact-sim-robot-visibility") != "supported"
        or verdicts.get("exact-sim-point-contact") != "supported"
        or verdicts.get("franka-kinematic-feasibility") != "abstention"
        or verdicts.get("franka-policy-trace-distinguishability")
        != ("supported" if policy_trace_pair is not None else "abstention")
        or verdicts.get("franka-candidate-policy-ranking") != "abstention"
        or verdicts.get("franka-physical-task-success") != "abstention"
        or decision["deployment_approval"] is not False
    ):
        raise ProviderNuRecTaskEvaluationError(["decision_claim_boundary_upgraded"])

    output = Path(output_root).expanduser().resolve()
    if output.is_symlink():
        raise ProviderNuRecTaskEvaluationError(["output_root_symlink_forbidden"])
    artifact_root = output / "provider_nurec_task_evaluation"
    artifacts: dict[str, Mapping[str, Any]] = {
        "testbed.json": testbed,
        "decision_evidence_request.json": decision_request,
        "method_profiles.json": {"method_profiles": profiles},
        "qualifications.json": {"qualifications": qualifications},
        "evidence_plan.json": plan,
        "execution_authorization.json": authorization,
        "execution_manifest.json": execution.execution_manifest,
        "decision_envelope.json": decision,
    }
    for index, result in enumerate(results, start=1):
        artifacts[f"evidence_result_{index}.json"] = result
    for name, value in sorted(artifacts.items()):
        _write_immutable(artifact_root / name, value)

    route_summary = {
        row["claim_id"]: {
            "status": row["status"],
            "selected_method_ids": [step["method_id"] for step in row["selected_methods"]],
            "next_cheapest_experiment": row["next_cheapest_experiment"],
        }
        for row in plan["claim_plans"]
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "site_id": task["site_id"],
        "task_id": task["task_id"],
        "robot_id": "franka_panda",
        "package_digest": request_input["package_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "testbed_digest": testbed["testbed_digest"],
        "request_digest": decision_request["request_digest"],
        "plan_digest": plan["plan_digest"],
        "decision_envelope_digest": decision["decision_envelope_digest"],
        "overall_outcome": decision["overall_outcome"],
        "per_claim_verdicts": decision["per_claim_verdicts"],
        "route_summary": route_summary,
        "candidate_policies": decision_request["candidates"],
        "cost_usd": 0.0,
        "paid_compute_reused_not_relaunched": policy_trace_pair is None,
        "claim_flags": {
            "metric_stage_semantics": True,
            "independent_known_distance_scale_anchor": False,
            "visual_robot_placement": True,
            "single_point_scene_contact": True,
            "formal_robot_placement": False,
            "kinematic_feasibility": False,
            "articulated_policy_execution": policy_trace_pair is not None,
            "policy_trace_distinguishability": policy_trace_pair is not None,
            "comparative_policy_ranking": False,
            "physical_task_success": False,
            "deployment_readiness": False,
        },
    }
    summary["summary_digest"] = canonical_digest(summary, digest_field="summary_digest")
    _write_immutable(artifact_root / "summary.json", summary)
    return summary


__all__ = [
    "ProviderNuRecTaskEvaluationError",
    "SCHEMA_VERSION",
    "compile_provider_nurec_task_evaluation",
]
