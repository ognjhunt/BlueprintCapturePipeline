from __future__ import annotations

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_provider_nurec import (
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_runtime_result,
)
from blueprint_pipeline.provider_nurec_robot_placement import (
    ProviderNuRecRobotPlacementError,
    build_default_franka_policy_trace_request,
    build_franka_inspection_controller_cohort_request,
    build_provider_nurec_robot_placement_packet,
    write_provider_nurec_robot_placement_packet,
)

import pytest


D = ["sha256:" + character * 64 for character in "abcdef"]


def _request():
    return build_provider_nurec_isaac_request(
        {
            "stable_run_identity": "ethel-placement-test",
            "source_commit_sha": "1" * 40,
            "package_digest": D[0],
            "package_artifact_reference": "source/ethel_sim.usdz",
            "external_import_receipt_digest": D[1],
            "qualification_report_digest": D[2],
            "fixed_camera_spec_digest": D[3],
            "fixed_camera_ids": ["ground-probe-local"],
            "runtime_implementation_digest": D[4],
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


def _runtime(request):
    return build_provider_nurec_isaac_runtime_result(
        {
            "schema_version": "provider_nurec_isaac_runtime_result.v1",
            "status": "completed",
            "isaac_verification_request_digest": request["isaac_verification_request_digest"],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_identity": {"runtime": "isaac_sim", "headless": True},
            "raw_secret_values_recorded": False,
            "stage": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "transforms_valid": True,
                "dependency_inspection_available": True,
                "missing_asset_count": 0,
                "obvious_scale_mismatch_detected": False,
                "expected_prim_paths": request["expected_prim_paths"],
                "particlefield_prim_count": 3,
                "active_collision_prim_count": 1,
            },
            "physics_probe": {
                "ground_contact_surface_present": True,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 12,
                "steps_executed": 240,
                "ground_surface": {
                    "prim_path": "/World/gauss/mesh",
                    "probe_height_m": -0.06,
                },
            },
            "cameras": [
                {
                    "id": "ground-probe-local",
                    "digest": D[5],
                    "pixel_std": 20.0,
                    "nonblank": True,
                }
            ],
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


def _qualification(request, runtime):
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


def test_packet_uses_franka_default_but_formal_placement_abstains(tmp_path):
    request = _request()
    runtime = _runtime(request)
    packet = build_provider_nurec_robot_placement_packet(
        verification_request=request,
        runtime_result=runtime,
        independent_qualification=_qualification(request, runtime),
        site_id="public_reference_ethel_sim",
        task_id="inspect-ground-probe-waypoint",
    )

    assert packet["render_options"]["robot_id"] == "franka_panda"
    assert packet["render_options"]["robot_ground_z"] == -0.06
    assert packet["placement_proposal"]["status"] == "runtime_visualization_candidate_only"
    support = packet["placement_proposal"]["known_support"]
    assert support["metric_stage_semantics_declared"] is True
    assert support["independent_known_distance_scale_anchor"] is False
    assert "metric_scale_verified" not in support
    assert (
        "independent_known_distance_scale_anchor_missing"
        in packet["placement_proposal"]["formal_gaps"]
    )
    placement = packet["robot_placement_result"]
    assert placement["status"] == "abstained"
    assert placement["selected_candidate_id"] is None
    reasons = placement["rejected_candidates"][0]["reasons"]
    assert "captured_coverage_insufficient" in reasons
    assert "collision_free_failed" in reasons
    assert packet["task_definition"]["source_asset"]["blueprint_raw_capture_truth"] is False

    receipt = write_provider_nurec_robot_placement_packet(output_dir=tmp_path, packet=packet)
    assert receipt["formal_placement_status"] == "abstained"
    assert receipt["physical_robot_execution_authorized"] is False
    assert (tmp_path / "render_options.json").is_file()


def test_packet_can_bind_reusable_franka_trace_pair_without_upgrading_placement():
    request = _request()
    runtime = _runtime(request)
    packet = build_provider_nurec_robot_placement_packet(
        verification_request=request,
        runtime_result=runtime,
        independent_qualification=_qualification(request, runtime),
        site_id="public_reference_ethel_sim",
        task_id="inspect-ground-probe-waypoint",
        include_articulated_policy_trace_pair=True,
    )

    trace = packet["render_options"]["articulated_policy_trace_request"]
    assert trace == build_default_franka_policy_trace_request(robot_prim_path="/World/Franka")
    assert [row["policy_id"] for row in trace["candidates"]] == [
        "franka-fixed-hold-v1",
        "franka-inspection-sweep-v1",
    ]
    assert packet["robot_placement_result"]["status"] == "abstained"
    assert trace["physical_success_claimed"] is False


def test_builds_five_scripted_inspection_controllers_without_policy_claim() -> None:
    trace = build_franka_inspection_controller_cohort_request(
        robot_prim_path="/World/Franka",
        target_position_stage=[0.1, -4.2, -0.1],
    )

    assert trace["controller_id"] == "deterministic_franka_inspection_cohort.v1"
    assert trace["candidate_kind"] == "scripted_controller_baseline"
    assert len(trace["candidates"]) == 5
    assert all(row["action_source"] == "scripted_controller" for row in trace["candidates"])
    assert trace["comparative_policy_ranking_claimed"] is False
    assert trace["physical_success_claimed"] is False


def test_packet_rejects_unqualified_runtime_binding():
    request = _request()
    runtime = _runtime(request)
    qualification = _qualification(request, runtime)
    qualification["blockers"] = ["not-actually-qualified"]
    qualification["qualification_digest"] = canonical_digest(
        qualification, digest_field="qualification_digest"
    )
    with pytest.raises(ProviderNuRecRobotPlacementError, match="has_blockers"):
        build_provider_nurec_robot_placement_packet(
            verification_request=request,
            runtime_result=runtime,
            independent_qualification=qualification,
            site_id="public_reference_ethel_sim",
            task_id="inspect-ground-probe-waypoint",
        )
