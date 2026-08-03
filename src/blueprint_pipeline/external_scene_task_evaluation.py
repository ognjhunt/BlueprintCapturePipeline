"""Compile an external reconstruction into a claim-bounded Task Evaluation Run.

The compiler is site and task specific, but engine selection remains evidence
driven. Isaac replay methods are qualified only for the exact signed artifacts
they can support; metric reach, policy ranking, and physical success abstain
until an independently qualified method and its required inputs exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    MaintainedSiteTaskTestbed,
    canonical_digest,
)
from .decision_evidence_execution import build_decision_envelope, execute_evidence_plan
from .decision_evidence_router import route_decision_evidence
from .external_scene_isaac_verification import (
    build_external_scene_isaac_verification_request,
)
from .external_scene_inspection_outcome import (
    build_franka_inspection_outcome_contract,
    rank_franka_inspection_candidates,
)
from .isaac_reconstruction_verification import build_isaac_runtime_result_v3
from .local_evidence_adapters import (
    SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER,
    SIGNED_ISAAC_POINT_CONTACT_ADAPTER,
    SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER,
    SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER,
    authorized_local_evidence_adapter_registry,
)
from .provider_nurec_task_evaluation import (
    ProviderNuRecTaskEvaluationError,
    _claim,
    _method_profiles,
    _validate_digest_artifact,
    _validated_policy_trace_pair,
    _write_immutable,
)


SCHEMA_VERSION = "external_scene_task_evaluation.v1"


class ExternalSceneTaskEvaluationError(ProviderNuRecTaskEvaluationError):
    pass


def compile_external_scene_task_evaluation(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    independent_qualification: Mapping[str, Any],
    visual_placement_evidence: Mapping[str, Any],
    target_analysis: Mapping[str, Any],
    robot_placement_result: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Route exact external-scene evidence without upgrading its claim ceiling."""

    request = build_external_scene_isaac_verification_request(verification_request)
    runtime = build_isaac_runtime_result_v3(runtime_result)
    independent = _validate_digest_artifact(
        independent_qualification,
        schema_version="reconstruction_isaac_independent_qualification.v1",
        digest_field="qualification_digest",
    )
    visual = _validate_digest_artifact(
        visual_placement_evidence,
        schema_version="external_scene_robot_placement_evidence.v1",
        digest_field="visual_placement_evidence_digest",
    )
    target = _validate_digest_artifact(
        target_analysis,
        schema_version="scene_task_target_analysis_result.v1",
        digest_field="target_analysis_digest",
        status_field="status",
        accepted_status="target_ready_for_bounded_sim",
    )
    placement = _validate_digest_artifact(
        robot_placement_result,
        schema_version="external_scene_robot_placement_candidate.v1",
        digest_field="placement_proposal_digest",
    )
    errors: list[str] = []
    if runtime.get("status") not in {"completed", "blocked"}:
        errors.append("external_scene_runtime_unavailable")
    source_probe_blockers = {
        "isaac_ground_contact_surface_missing",
        "isaac_physics_probe_not_executed",
        "isaac_test_body_contact_not_observed",
        "isaac_test_body_fell_through_floor",
        "isaac_test_body_pose_unavailable",
    }
    runtime_blockers = sorted(str(code) for code in (runtime.get("blockers") or []) if str(code))
    proxy = runtime.get("proxy_composed_evaluation")
    proxy = proxy if isinstance(proxy, Mapping) else {}
    pair_value = runtime.get("articulated_policy_trace_pair")
    pair_value = pair_value if isinstance(pair_value, Mapping) else {}
    source_probe_only_abstention = bool(
        runtime.get("status") == "blocked"
        and runtime_blockers
        and set(runtime_blockers).issubset(source_probe_blockers)
        and proxy.get("configured") is True
        and proxy.get("source_collision_restored_for_independent_probe") is True
        and pair_value.get("status") == "completed"
    )
    policy_only_abstention = bool(
        runtime.get("status") == "blocked"
        and runtime_blockers == ["isaac_articulated_policy_trace_pair_incomplete"]
    )
    if runtime.get("status") == "blocked" and not (
        source_probe_only_abstention or policy_only_abstention
    ):
        errors.append("external_scene_runtime_static_evidence_not_qualifiable")
    if independent.get("status") not in {
        "verified_compatibility_only",
        "verified_proxy_composed_policy_only",
    }:
        errors.append("external_scene_independent_qualification_status_invalid")
    if independent.get("blockers") not in ([], ()):
        errors.append("external_scene_independent_qualification_has_blockers")
    if independent.get("isaac_verification_request_digest") != request.get(
        "isaac_verification_request_digest"
    ):
        errors.append("external_scene_independent_request_mismatch")
    if independent.get("runtime_result_digest") != runtime.get("isaac_runtime_result_digest"):
        errors.append("external_scene_independent_runtime_mismatch")
    expected_independent_ceiling = (
        "exact_proxy_composed_simulation_policy_trace_only"
        if independent.get("status") == "verified_proxy_composed_policy_only"
        else "isaac_load_render_compatibility"
    )
    if independent.get("claim_ceiling") != expected_independent_ceiling:
        errors.append("external_scene_independent_ceiling_invalid")
    if target.get("target_analysis_digest") != request.get("target_analysis_digest"):
        errors.append("external_scene_target_request_mismatch")
    if placement.get("placement_proposal_digest") != request.get("placement_proposal_digest"):
        errors.append("external_scene_placement_request_mismatch")
    selected = target.get("selected_target")
    robot = runtime.get("robot")
    physics = runtime.get("physics_probe")
    if not isinstance(selected, Mapping):
        errors.append("external_scene_selected_target_missing")
    if not isinstance(robot, Mapping):
        errors.append("external_scene_robot_evidence_missing")
    if not isinstance(physics, Mapping):
        errors.append("external_scene_physics_evidence_missing")
    if visual.get("status") not in {"verified_visual_placement_only", "blocked"}:
        errors.append("external_scene_visual_evidence_status_invalid")
    if placement.get("status") not in {"runtime_visualization_candidate_only", "abstained"}:
        errors.append("external_scene_placement_status_invalid")
    if placement.get("status") == "abstained" and placement.get("proof_effect") != (
        "external_scene_runtime_robot_visualization_candidate"
    ):
        errors.append("external_scene_abstained_placement_not_runtime_candidate")
    if errors:
        raise ExternalSceneTaskEvaluationError(errors)

    assert isinstance(selected, Mapping)
    assert isinstance(robot, Mapping)
    assert isinstance(physics, Mapping)
    policy_pair = None
    pair_value = runtime.get("articulated_policy_trace_pair")
    if isinstance(pair_value, Mapping) and pair_value.get("status") == "completed":
        policy_pair = _validated_policy_trace_pair(runtime)
    inspection_outcome_contract = build_franka_inspection_outcome_contract(
        target_analysis=target,
        placement_proposal_digest=placement["placement_proposal_digest"],
        target_position_stage=placement["target_position_collision_stage"],
        scene_frame_binding_digest=placement["scene_frame_binding_digest"],
    )
    inspection_ranking = None
    if (
        policy_pair is not None
        and policy_pair.get("controller_id") == "deterministic_franka_inspection_cohort.v1"
    ):
        if (
            policy_pair.get("inspection_outcome_contract_digest")
            != inspection_outcome_contract["contract_digest"]
        ):
            raise ExternalSceneTaskEvaluationError(
                ["external_scene_inspection_outcome_contract_mismatch"]
            )
        inspection_ranking = rank_franka_inspection_candidates(
            contract=inspection_outcome_contract,
            candidates=[
                {
                    "candidate_id": row["policy_id"],
                    "action_source": row["action_source"],
                    "checkpoint_provenance_digest": row.get("checkpoint_provenance_digest"),
                    "stable_reset_observed": row["reset_stability"]["status"] == "completed",
                    "collision_free_observed": row["collision_free_observed"],
                    "terminal_egocentric_nonblank": row["egocentric_observation"]["nonblank"],
                    "target_view_observations": row["target_view_observations"],
                }
                for row in policy_pair["candidate_traces"]
            ],
        )
    visual_qualified = bool(
        visual.get("status") == "verified_visual_placement_only"
        and visual.get("visual_robot_placement_observed") is True
        and not visual.get("blockers")
    )
    source_point_contact_qualified = bool(
        independent.get("status") == "verified_compatibility_only"
        and physics.get("ground_contact_surface_present") is True
        and physics.get("live_rigid_body_pose_observed") is True
        and physics.get("test_body_fell_through_floor") is False
        and isinstance(physics.get("contact_event_count"), int)
        and not isinstance(physics.get("contact_event_count"), bool)
        and int(physics.get("contact_event_count") or 0) >= 1
    )
    camera_evidence = visual.get("camera_evidence")
    camera_evidence = camera_evidence if isinstance(camera_evidence, list) else []
    pose = placement["robot_pose_xyzyaw_collision_stage"]
    task_id = str(selected["task_family"])
    target_region_id = str(selected["proposal_id"])
    target_position = list(selected["target_position_scene"])
    task_family = str(selected["task_family"])
    site_id = str(target["scene_id"])
    bundle_id = f"{site_id}-external-scene-package"
    reset_digest = canonical_digest(
        {
            "reset": "reload_exact_external_scene_package_and_robot_pose",
            "package_digest": request["package_digest"],
            "robot_pose": pose,
        }
    )
    bindings = {
        "embodiment": {
            "robot_id": request["robot_id"],
            "robot_prim_path": robot.get("prim_path"),
            "robot_asset": robot.get("robot_usd"),
        },
        "sensors": {
            "fixed_camera_ids": list(request["fixed_camera_ids"]),
            "robot_relative_camera": (
                {
                    "camera_id": "franka-wrist-egocentric",
                    "parent_link": "panda_hand",
                    "modality": "isaac_rgb",
                    "trace_pair_digest": policy_pair["articulated_policy_trace_pair_digest"],
                }
                if policy_pair is not None
                else None
            ),
            "modality": "isaac_rgb_distance_to_camera",
        },
        "controller_action_representation": {
            "type": "joint_position",
            "controller_id": (
                policy_pair["controller_id"]
                if policy_pair is not None
                else "deterministic_franka_inspection_cohort.v1"
            ),
        },
        "selected_robot_placement": {
            "candidate_id": "external-scene-placement-candidate",
            "base_position_site_m": pose[:3],
            "formal_status": "candidate_only",
            "visual_status": visual["status"],
            "method_qualification_status": "visual_only" if visual_qualified else "unqualified",
        },
    }
    evidence_inventory: list[dict[str, Any]] = [
        {
            "evidence_id": "signed_isaac_visual_placement",
            "independently_rehashed": visual_qualified,
            "visual_placement_evidence_digest": visual["visual_placement_evidence_digest"],
            "camera_evidence": camera_evidence if visual_qualified else [],
            "exact_view_coverage": 1.0 if visual_qualified else 0.0,
            "formal_robot_placement": False,
            "blockers": list(visual.get("blockers") or []),
        },
        {
            "evidence_id": "signed_isaac_point_contact",
            "independently_qualified": source_point_contact_qualified,
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
    if policy_pair is not None:
        evidence_inventory.append(
            {
                "evidence_id": "signed_isaac_articulated_policy_trace_pair",
                "independently_validated": True,
                "articulated_policy_trace_pair_digest": policy_pair[
                    "articulated_policy_trace_pair_digest"
                ],
                "policy_trace_request_digest": policy_pair["policy_trace_request_digest"],
                "controller_id": policy_pair["controller_id"],
                "identical_frozen_start_observed": policy_pair["trace_pair_assessment"][
                    "identical_frozen_start_observed"
                ],
                "distinct": policy_pair["trace_pair_assessment"]["distinct"],
                "maximum_end_joint_delta_rad": policy_pair["trace_pair_assessment"][
                    "maximum_end_joint_delta_rad"
                ],
                "candidate_traces": [
                    {
                        "policy_id": row["policy_id"],
                        "status": row["status"],
                        "policy_trace_digest": row["policy_trace_digest"],
                        "egocentric_observation_digest": row["egocentric_observation"]["digest"],
                    }
                    for row in policy_pair["candidate_traces"]
                ],
                "comparative_policy_ranking": False,
                "metric_task_success": False,
                "physical_task_success": False,
            }
        )
    if inspection_ranking is not None:
        evidence_inventory.append(
            {
                "evidence_id": "signed_isaac_inspection_candidate_ranking",
                **inspection_ranking,
            }
        )
    testbed = MaintainedSiteTaskTestbed.from_mapping(
        {
            "schema_version": "maintained_site_task_testbed.v1",
            "testbed_id": f"{site_id}-franka-external-scene-testbed",
            "version": "1",
            "predecessor_testbed_digest": None,
            "supersedes": [],
            "source_capture_bundles": [
                {"bundle_id": bundle_id, "version": "1", "digest": request["package_digest"]}
            ],
            "artifact_references": {
                "site_card": {
                    "uri": f"artifact://{site_id}",
                    "digest": target["target_analysis_digest"],
                },
                "task_cards": [
                    {"uri": f"artifact://{task_id}", "digest": target["target_analysis_digest"]}
                ],
                "scenario_cards": [
                    {
                        "uri": f"artifact://{target_region_id}",
                        "digest": request["target_binding_digest"],
                    }
                ],
                "eval_cards": [
                    {
                        "uri": "artifact://external-scene-base-eval",
                        "digest": independent["qualification_digest"],
                    }
                ],
                "evaluator": {
                    "uri": "artifact://signed-evidence-router",
                    "digest": independent["qualification_digest"],
                },
                "reset": {"uri": "artifact://external-scene-reset", "digest": reset_digest},
            },
            "task_distribution": {"task_family": task_family, "tasks": [task_id]},
            "supported_condition_ranges": {
                "site_id": site_id,
                "package_digest": request["package_digest"],
                "scene_entrypoint": request["package_artifact_reference"],
                "robot_pose_xyzyaw_site": pose,
                "fixed_camera_ids": list(request["fixed_camera_ids"]),
            },
            "robot_sensor_controller_bindings": bindings,
            "governance": {
                "rights": "operator_authorized_private_evaluation_only",
                "consent": "explicit_remote_processing_authorization",
                "privacy": "confidential_private_site",
                "revocation": "delete_on_operator_request_or_new_version",
                "allowed_uses": ["internal_evaluation"],
            },
            "evidence_inventory": evidence_inventory,
            "validation_envelope": {
                "exact_digest_scope": True,
                "source_video_available": bool(request["source_video_available"]),
                "source_video_required_for_candidate_execution": False,
                "independent_metric_scale_proven": False,
                "isaac_verification_request_digest": request["isaac_verification_request_digest"],
                "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
                "independent_qualification_digest": independent["qualification_digest"],
                "visual_placement_evidence_digest": visual["visual_placement_evidence_digest"],
                "formal_robot_placement_digest": placement["placement_proposal_digest"],
                "articulated_policy_trace_pair_digest": (
                    policy_pair["articulated_policy_trace_pair_digest"] if policy_pair else None
                ),
                "inspection_outcome_contract_digest": inspection_outcome_contract[
                    "contract_digest"
                ],
                "inspection_candidate_ranking_digest": (
                    inspection_ranking["ranking_digest"] if inspection_ranking else None
                ),
            },
            "target_regions": [
                {
                    "region_id": target_region_id,
                    "position_site_m": target_position,
                    "supporting_frames": list(selected["supporting_view_ids"]),
                    "captured_coverage": 0.0,
                    "simulated_view_coverage": 1.0,
                }
            ],
            "known_unsupported_conditions": [
                "independent_metric_scale",
                "formal_robot_placement",
                "complete_robot_footprint_clearance",
                "kinematic_reachability",
                "comparative_policy_ranking",
                "physical_task_success",
                "deployment_readiness",
            ]
            + ([] if visual_qualified else ["visual_robot_placement"])
            + ([] if source_point_contact_qualified else ["source_collision_contact"])
            + ([] if policy_pair else ["articulated_policy_execution"]),
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
        "task_family": task_family,
        "site_domain_conditions": testbed["supported_condition_ranges"],
        "embodiment": bindings["embodiment"],
        "sensors": bindings["sensors"],
        "controller_action_representation": bindings["controller_action_representation"],
    }
    candidate_ids = (
        [row["policy_id"] for row in policy_pair["candidate_traces"]]
        if policy_pair is not None
        else []
    )
    claims = [
        _claim(
            "exact-sim-robot-visibility",
            "perception_visibility",
            subject={"target_region_id": target_region_id},
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
            subject={"target_position_site_m": target_position},
            scope=scope,
        ),
        _claim(
            "franka-policy-trace-distinguishability",
            "simulated_policy_trace_distinguishability",
            subject={"candidate_policy_ids": candidate_ids},
            scope=scope,
        ),
        _claim(
            "franka-controller-ranking",
            "comparative_controller_ranking",
            subject={"candidate_controller_ids": candidate_ids},
            scope=scope,
        ),
        _claim(
            "franka-candidate-policy-ranking",
            "comparative_policy_ranking",
            subject={"candidate_policy_ids": candidate_ids},
            scope=scope,
            consequence="high",
            risk=0.01,
        ),
        _claim(
            "franka-physical-task-success",
            "physical_task_success",
            subject=task_id,
            scope=scope,
            consequence="high",
            risk=0.01,
        ),
    ]
    decision_request = DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": f"{site_id}-franka-base-eval-request-v1",
            "decision_id": f"{site_id}-franka-base-eval-decision-v1",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": f"Which claims are supported for {site_id}, {task_id}, and the exact bound Franka placement?",
            "candidates": [
                {
                    "robot_id": "franka_panda",
                    "policy_id": candidate_id,
                    "candidate_kind": "scripted_controller_baseline",
                    "policy_trace_status": "collected" if policy_pair else "not_collected",
                }
                for candidate_id in candidate_ids
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
            "provenance": {"caller_identity": "pipeline:external-scene-task-evaluation"},
            "idempotency_key": f"{site_id}-franka-base-eval-request-v1",
        }
    ).to_mapping()
    profiles, qualifications = _method_profiles(testbed=testbed)
    plan = route_decision_evidence(decision_request, testbed, profiles, qualifications).to_mapping()
    authorized_adapters = []
    if source_point_contact_qualified:
        authorized_adapters.append(SIGNED_ISAAC_POINT_CONTACT_ADAPTER)
    if visual_qualified:
        authorized_adapters.append(SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER)
    if policy_pair is not None:
        authorized_adapters.append(SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER)
    if inspection_ranking is not None and inspection_ranking.get("status") == "completed":
        authorized_adapters.append(SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER)
    authorization = {
        "schema_version": "external_scene_task_evaluation_authorization.v1",
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
    results = [row.to_mapping() for row in execution.results]
    decision = build_decision_envelope(decision_request, testbed, plan, results).to_mapping()
    verdicts = {row["claim_id"]: row["verdict"] for row in decision["per_claim_verdicts"]}
    expected = {
        "exact-sim-robot-visibility": "supported" if visual_qualified else "abstention",
        "exact-sim-point-contact": (
            "supported" if source_point_contact_qualified else "abstention"
        ),
        "franka-kinematic-feasibility": "abstention",
        "franka-policy-trace-distinguishability": "supported" if policy_pair else "abstention",
        "franka-controller-ranking": (
            "supported"
            if inspection_ranking is not None and inspection_ranking["status"] == "completed"
            else "abstention"
        ),
        "franka-candidate-policy-ranking": "abstention",
        "franka-physical-task-success": "abstention",
    }
    if verdicts != expected or decision["deployment_approval"] is not False:
        raise ExternalSceneTaskEvaluationError(["external_scene_decision_claim_boundary_upgraded"])
    artifact_root = Path(output_root).expanduser().resolve() / "external_scene_task_evaluation"
    artifacts = {
        "inspection_outcome_contract.json": inspection_outcome_contract,
        "testbed.json": testbed,
        "decision_evidence_request.json": decision_request,
        "method_profiles.json": {"method_profiles": profiles},
        "qualifications.json": {"qualifications": qualifications},
        "evidence_plan.json": plan,
        "execution_authorization.json": authorization,
        "execution_manifest.json": execution.execution_manifest,
        "decision_envelope.json": decision,
    }
    if inspection_ranking is not None:
        artifacts["inspection_candidate_ranking.json"] = inspection_ranking
    for index, result in enumerate(results, start=1):
        artifacts[f"evidence_result_{index}.json"] = result
    for name, value in sorted(artifacts.items()):
        _write_immutable(artifact_root / name, value)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "site_id": site_id,
        "task_id": task_id,
        "target_region_id": target_region_id,
        "robot_id": request["robot_id"],
        "package_digest": request["package_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "testbed_digest": testbed["testbed_digest"],
        "plan_digest": plan["plan_digest"],
        "decision_envelope_digest": decision["decision_envelope_digest"],
        "overall_outcome": decision["overall_outcome"],
        "per_claim_verdicts": decision["per_claim_verdicts"],
        "selected_methods_by_claim": {
            row["claim_id"]: [step["method_id"] for step in row["selected_methods"]]
            for row in plan["claim_plans"]
        },
        "source_video_available": bool(request["source_video_available"]),
        "source_video_missing_is_not_pipeline_blocker": True,
        "independent_metric_scale_proven": False,
        "visual_robot_placement_qualified": visual_qualified,
        "single_point_contact_qualified": source_point_contact_qualified,
        "policy_trace_pair_qualified": policy_pair is not None,
        "controller_candidate_ranking_proven": bool(
            inspection_ranking and inspection_ranking["controller_candidate_ranking_proven"] is True
        ),
        "comparative_policy_ranking_proven": False,
        "physical_task_success_proven": False,
        "deployment_readiness_proven": False,
        "cost_usd": 0.0,
    }
    summary["summary_digest"] = canonical_digest(summary, digest_field="summary_digest")
    _write_immutable(artifact_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "verification-request",
        "runtime-result",
        "independent-qualification",
        "visual-placement-evidence",
        "target-analysis",
        "robot-placement-result",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-out")
    args = parser.parse_args(argv)

    def load(name: str) -> Mapping[str, Any]:
        value = json.loads(Path(getattr(args, name)).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ExternalSceneTaskEvaluationError([f"{name}_not_json_object"])
        return value

    summary = compile_external_scene_task_evaluation(
        verification_request=load("verification_request"),
        runtime_result=load("runtime_result"),
        independent_qualification=load("independent_qualification"),
        visual_placement_evidence=load("visual_placement_evidence"),
        target_analysis=load("target_analysis"),
        robot_placement_result=load("robot_placement_result"),
        output_root=args.output_root,
    )
    if args.summary_out:
        write_json(Path(args.summary_out), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = [
    "ExternalSceneTaskEvaluationError",
    "SCHEMA_VERSION",
    "compile_external_scene_task_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
