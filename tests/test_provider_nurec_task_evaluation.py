from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_provider_nurec import (
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_runtime_result,
)
from blueprint_pipeline.provider_nurec_robot_placement import (
    build_provider_nurec_robot_placement_packet,
)
from blueprint_pipeline.provider_nurec_task_evaluation import (
    ProviderNuRecTaskEvaluationError,
    compile_provider_nurec_task_evaluation,
)


D = ["sha256:" + character * 64 for character in "abcdef"]


def _request() -> dict:
    return build_provider_nurec_isaac_request(
        {
            "stable_run_identity": "ethel-task-evaluation-test",
            "source_commit_sha": "1" * 40,
            "package_digest": D[0],
            "package_artifact_reference": "source/ethel_sim.usdz",
            "external_import_receipt_digest": D[1],
            "qualification_report_digest": D[2],
            "fixed_camera_spec_digest": D[3],
            "fixed_camera_ids": ["ground-probe-local"],
            "runtime_implementation_digest": D[4],
            "render_options_digest": D[5],
            "runtime_container_image_digest": "registry.test/isaac@" + D[5],
            "expected_prim_paths": {
                "appearance": "/World/gauss/gauss",
                "collision": "/World/gauss/mesh",
            },
            "physics_probe_request": {
                "ground_collider_prim": "/World/gauss/mesh",
                "ground_height_m": -0.06,
                "probe_xy_m": [31.2, -9.65],
                "selection_status": "cpu_geometry_candidate_unverified_in_isaac",
                "manufacture_ground_plane": False,
                "require_contact_event": True,
                "steps": 240,
            },
            "timeout_seconds": 3600,
            "spend_controls": {
                "authorized": False,
                "estimated_max_spend_usd": 2.0,
                "hard_ttl_seconds": 3600,
                "teardown_required": True,
                "provider_zero_required_before_and_after": True,
            },
            "provider_authored_package": True,
            "exact_package_required": True,
            "headless": True,
            "display_attached": False,
            "execution_status": "awaiting_explicit_paid_runtime_authorization",
            "provider_allocation_performed": False,
            "expected_runtime_schema": "provider_nurec_isaac_runtime_result.v1",
            "proof_effect": "none",
            "claim_ceiling": "request_only",
        }
    )


def _runtime(request: dict) -> dict:
    return build_provider_nurec_isaac_runtime_result(
        {
            "schema_version": "provider_nurec_isaac_runtime_result.v1",
            "status": "completed",
            "isaac_verification_request_digest": request["isaac_verification_request_digest"],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_identity": {
                "runtime": "isaac_sim",
                "renderer": "RayTracedLighting",
                "python_version": "3.11",
                "headless": True,
            },
            "raw_secret_values_recorded": False,
            "cost_usd": 0.1,
            "duration_seconds": 120.0,
            "stage": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "transforms_valid": True,
                "dependency_inspection_available": True,
                "missing_asset_count": 0,
                "particlefield_prim_count": 3,
                "active_collision_prim_count": 1,
                "obvious_scale_mismatch_detected": False,
                "expected_prim_paths": request["expected_prim_paths"],
            },
            "physics_probe": {
                "ground_contact_surface_present": True,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 12,
                "steps_executed": 240,
                "probe_configuration": {},
                "ground_surface": {
                    "prim_path": "/World/gauss/mesh",
                    "probe_height_m": -0.06,
                },
            },
            "cameras": [
                {
                    "id": "ground-probe-local",
                    "artifact_reference": "frames/ground-probe-local.png",
                    "digest": D[1],
                    "width": 1280,
                    "height": 960,
                    "pixel_mean": 100.0,
                    "pixel_std": 10.0,
                    "nonblank": True,
                }
            ],
            "robot": {
                "requested": True,
                "composited": True,
                "geometry_streamed": True,
                "mesh_point_total": 49244,
                "prim_path": "/World/Franka",
                "robot_usd": "Robots/FrankaRobotics/FrankaPanda/franka.usd",
                "resolved_usd": "omniverse://assets/Franka/franka.usd",
                "robot_pose": [31.2, -9.65, -0.06, 0.0],
                "world_bound_min": [31.05, -9.79, -0.06],
                "world_bound_max": [31.39, -9.53, 1.05],
            },
            "proof_boundary": {
                "isaac_load_render_physics_presence_compatibility": True,
                "simulator_task_success_proven": False,
                "physics_navigation_control_proven": False,
                "physical_success_proven": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
            },
        },
        verification_request=request,
    )


def _independent(request: dict, runtime: dict) -> dict:
    value = {
        "schema_version": "reconstruction_isaac_independent_qualification.v1",
        "status": "verified_compatibility_only",
        "isaac_verification_request_digest": request["isaac_verification_request_digest"],
        "runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "qualified_result_digest": D[4],
        "blockers": [],
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": "isaac_load_render_physics_presence_only",
        "claim_ceiling": "isaac_load_render_compatibility",
    }
    value["qualification_digest"] = canonical_digest(value, digest_field="qualification_digest")
    return value


def _visual(request: dict, runtime: dict) -> dict:
    value = {
        "schema_version": "provider_robot_placement_evidence.v1",
        "status": "verified_visual_placement_only",
        "blockers": [],
        "isaac_verification_request_digest": request["isaac_verification_request_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "package_digest": request["package_digest"],
        "render_options_digest": request["render_options_digest"],
        "robot_prim_path": "/World/Franka",
        "robot_usd": "Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "resolved_robot_usd": "omniverse://assets/Franka/franka.usd",
        "robot_pose": [31.2, -9.65, -0.06, 0.0],
        "world_bound_min": [31.05, -9.79, -0.06],
        "world_bound_max": [31.39, -9.53, 1.05],
        "mesh_point_total": 49244,
        "camera_evidence": [
            {
                "id": "ground-probe-local",
                "rgb_artifact_reference": "frames_robot_only/ground-probe-local.png",
                "rgb_digest": D[2],
                "distance_artifact_reference": (
                    "frames_robot_only/ground-probe-local_distance.npy"
                ),
                "distance_digest": D[3],
                "depth_foreground_pixel_count": 1842,
                "depth_foreground_fraction": 0.0015,
                "min_distance_m": 7.4,
                "max_distance_m": 7.8,
                "visual_geometry_observed": True,
            }
        ],
        "visual_robot_placement_observed": True,
        "collision_free_placement_proven": False,
        "kinematic_reachability_proven": False,
        "navigation_or_task_success_proven": False,
        "physical_robot_readiness_proven": False,
        "proof_effect": "visual_robot_placement_evidence_only",
        "claim_ceiling": "isaac_visual_robot_placement",
        "raw_secret_values_recorded": False,
    }
    value["provider_robot_placement_evidence_digest"] = canonical_digest(
        value, digest_field="provider_robot_placement_evidence_digest"
    )
    return value


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    request = _request()
    runtime = _runtime(request)
    independent = _independent(request, runtime)
    packet = build_provider_nurec_robot_placement_packet(
        verification_request=request,
        runtime_result=runtime,
        independent_qualification=independent,
        site_id="public_reference_ethel_sim",
        task_id="inspect-ground-probe-waypoint",
    )
    return request, runtime, independent, _visual(request, runtime), packet


def test_compile_routes_exact_evidence_and_abstains_on_policy_claims(tmp_path) -> None:
    request, runtime, independent, visual, packet = _inputs()
    result = compile_provider_nurec_task_evaluation(
        verification_request=request,
        runtime_result=runtime,
        independent_qualification=independent,
        visual_placement_evidence=visual,
        task_definition=packet["task_definition"],
        robot_placement_result=packet["robot_placement_result"],
        output_root=tmp_path,
    )

    assert result["overall_outcome"] == "partial_decision"
    verdicts = {row["claim_id"]: row["verdict"] for row in result["per_claim_verdicts"]}
    assert verdicts["exact-sim-robot-visibility"] == "supported"
    assert verdicts["exact-sim-point-contact"] == "supported"
    assert verdicts["franka-kinematic-feasibility"] == "abstention"
    assert verdicts["franka-candidate-policy-ranking"] == "abstention"
    assert verdicts["franka-physical-task-success"] == "abstention"
    assert result["route_summary"]["exact-sim-robot-visibility"]["selected_method_ids"] == [
        "signed-isaac-visual-placement-replay"
    ]
    assert result["paid_compute_reused_not_relaunched"] is True
    assert result["candidate_policies"] == [
        {
            "robot_id": "franka_panda",
            "policy_id": "franka-fixed-hold-v1",
            "policy_trace_status": "not_collected",
        },
        {
            "robot_id": "franka_panda",
            "policy_id": "franka-inspection-sweep-v1",
            "policy_trace_status": "not_collected",
        },
    ]
    assert (tmp_path / "provider_nurec_task_evaluation/summary.json").is_file()


def test_compile_rejects_tampered_visual_evidence(tmp_path) -> None:
    request, runtime, independent, visual, packet = _inputs()
    tampered = copy.deepcopy(visual)
    tampered["camera_evidence"][0]["depth_foreground_pixel_count"] += 1

    with pytest.raises(
        ProviderNuRecTaskEvaluationError,
        match="provider_robot_placement_evidence_digest_mismatch",
    ):
        compile_provider_nurec_task_evaluation(
            verification_request=request,
            runtime_result=runtime,
            independent_qualification=independent,
            visual_placement_evidence=tampered,
            task_definition=packet["task_definition"],
            robot_placement_result=packet["robot_placement_result"],
            output_root=tmp_path,
        )
