from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.reconstruction_capability import (
    ReconstructionContractError,
    build_reconstruction_method_profile,
    decide_simready_assets,
    normalize_reconstruction_result,
    plan_reconstruction_methods,
    score_robot_placements,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _beta_fixture(case_id: str) -> dict:
    matrix = json.loads(
        (
            Path(__file__).parents[1]
            / "tests"
            / "fixtures"
            / "design_partner_beta_v1"
            / "fixture_matrix.json"
        ).read_text(encoding="utf-8")
    )
    return next(row for row in matrix["cases"] if row["case_id"] == case_id)


def _method(
    method_id: str,
    kind: str,
    outputs: list[str],
    *,
    cost: float,
    qualified: str = "qualified",
    authorized: bool = True,
    required_flags: list[str] | None = None,
) -> dict:
    return {
        "method_id": method_id,
        "version": "1",
        "implementation_digest": SHA_B,
        "method_kind": kind,
        "provider_identity": "local",
        "execution_mode": "hermetic_local",
        "outputs": outputs,
        "required_capture_authority_profiles": [],
        "required_claim_ceiling_flags": required_flags or [],
        "qualified_claim_types": [],
        "execution_authorized": authorized,
        "qualification_status": qualified,
        "expected_cost_usd": cost,
        "provider_constraints": {},
        "rights_constraints": {},
        "failure_modes": [],
    }


def test_reconstruction_planner_selects_cheapest_sufficient_not_always_3dgs() -> None:
    plan = plan_reconstruction_methods(
        intake_id="intake-1",
        capture_digest=SHA_A,
        capture_authority_profile="iphone_arkit_lidar",
        claim_ceiling={"metric_geometry": True, "calibrated_camera_poses": True},
        requested_claim_types=["perception_visibility", "reachability"],
        permitted_provider_identities=["local"],
        method_profiles=[
            _method(
                "calibrated-frames",
                "pose_sfm_estimation",
                ["calibrated_frames"],
                cost=0.01,
            ),
            _method(
                "lidar-scaffold",
                "lidar_depth_fusion",
                ["metric_reference_layer"],
                cost=0.05,
                required_flags=["metric_geometry"],
            ),
            _method(
                "expensive-splat",
                "gaussian_splat_3d",
                ["appearance_layer"],
                cost=4.0,
            ),
        ],
    )

    assert plan["status"] == "planned"
    assert [row["method_id"] for row in plan["selected_methods"]] == [
        "calibrated-frames",
        "lidar-scaffold",
    ]
    assert "appearance_layer" not in plan["required_representations"]
    assert plan["proof_boundary"]["physical_task_success_established"] is False


def test_reconstruction_planner_minimizes_total_method_set_cost() -> None:
    plan = plan_reconstruction_methods(
        intake_id="intake-1",
        capture_digest=SHA_A,
        capture_authority_profile="iphone_arkit_lidar",
        claim_ceiling={"metric_geometry": True},
        requested_claim_types=["perception_visibility", "reachability"],
        permitted_provider_identities=["local"],
        method_profiles=[
            _method(
                "frames-only",
                "pose_sfm_estimation",
                ["calibrated_frames"],
                cost=0.04,
            ),
            _method(
                "metric-only",
                "metric_scaffold",
                ["metric_reference_layer"],
                cost=0.04,
            ),
            _method(
                "combined-local",
                "lidar_depth_fusion",
                ["calibrated_frames", "metric_reference_layer"],
                cost=0.07,
            ),
        ],
    )

    assert plan["estimated_cost_usd"] == 0.07
    assert plan["selected_methods"] == [
        {
            "representations": ["calibrated_frames", "metric_reference_layer"],
            "method_id": "combined-local",
            "method_version": "1",
            "method_profile_digest": plan["selected_methods"][0]["method_profile_digest"],
            "provider_identity": "local",
            "expected_cost_usd": 0.07,
        }
    ]


def test_physics_plan_rejects_generated_or_unqualified_collision_output() -> None:
    plan = plan_reconstruction_methods(
        intake_id="intake-1",
        capture_digest=SHA_A,
        capture_authority_profile="monocular_video",
        claim_ceiling={"metric_geometry": False},
        requested_claim_types=["collision_contact"],
        permitted_provider_identities=["local"],
        method_profiles=[
            _method(
                "generated-completion",
                "generated_visual_completion",
                ["collision_geometry", "physics_layer"],
                cost=0.0,
            ),
            _method(
                "debug-collider",
                "collision_proxy",
                ["collision_geometry", "physics_layer"],
                cost=0.01,
                qualified="debug_only",
            ),
            _method(
                "metric-from-lidar",
                "lidar_depth_fusion",
                ["metric_reference_layer"],
                cost=0.02,
                required_flags=["metric_geometry"],
            ),
        ],
    )

    assert plan["status"] == "partial_plan"
    assert plan["selected_methods"] == []
    reasons = {
        rejected["reason"]
        for missing in plan["missing_representations"]
        for rejected in missing["rejected_candidates"]
    }
    assert "method_not_qualified_for_physics_output" in reasons
    assert "required_capture_evidence_missing" in reasons


def _reconstruction_result() -> dict:
    return {
        "result_id": "reconstruction-1",
        "intake_id": "intake-1",
        "capture_digest": SHA_A,
        "method_id": "local-splat",
        "method_version": "1",
        "method_profile_digest": SHA_B,
        "implementation_digest": SHA_B,
        "provider_identity": "local",
        "runtime_identity": "fixture-runtime",
        "runtime_digest": SHA_C,
        "outputs": ["appearance_layer"],
        "source_frames": {"frame_ids": ["frame-1"]},
        "camera_solution": {"status": "calibrated"},
        "coordinate_system": {"up_axis": "Y", "scale_status": "not_authoritative"},
        "asset_references": {"appearance": {"uri": "fixture://splat", "digest": SHA_C}},
        "coverage_map": {"covered_fraction": 0.8},
        "observed_regions": [{"region_id": "observed-1"}],
        "generated_regions": [],
        "uncertainty_map": {"uri": "fixture://uncertainty", "digest": SHA_C},
        "invalid_regions": [],
        "validation_metrics": {"held_out_psnr": 20.0},
        "cost_usd": 0.0,
        "duration_seconds": 1.0,
        "provider_receipt": None,
        "rights_and_retention": {"external_processing": False},
        "deletion_evidence": None,
        "claim_ceiling": {
            "appearance_review": True,
            "metric_geometry": False,
            "collision_geometry": False,
            "physical_task_success": False,
        },
    }


def test_reconstruction_result_requires_generated_region_masks_and_physics_exclusion() -> None:
    first = normalize_reconstruction_result(_reconstruction_result())
    second = normalize_reconstruction_result(_reconstruction_result())
    assert first == second

    unsafe = _reconstruction_result()
    unsafe["outputs"] = ["appearance_layer", "collision_geometry"]
    unsafe["generated_regions"] = [{"region_id": "hidden-rear"}]
    with pytest.raises(
        ReconstructionContractError,
        match="generated_regions:mask_reference_required",
    ):
        normalize_reconstruction_result(unsafe)

    masked = copy.deepcopy(unsafe)
    masked["generated_regions"][0]["mask_reference"] = "fixture://generated-mask"
    with pytest.raises(
        ReconstructionContractError,
        match="generated_regions:physics_exclusion_required",
    ):
        normalize_reconstruction_result(masked)

    stale_digest = _reconstruction_result()
    stale_digest["reconstruction_result_digest"] = SHA_A
    with pytest.raises(ReconstructionContractError, match="reconstruction_result_digest:mismatch"):
        normalize_reconstruction_result(stale_digest)


def test_generated_only_trajectory_intersection_forces_next_experiment_and_no_physics() -> None:
    fixture = _beta_fixture("generated_only_gap_intersects_trajectory")
    assert fixture["generated_region_mask_intersects_trajectory"] is True
    intersecting = _reconstruction_result()
    intersecting["generated_regions"] = [{
        "region_id": "hidden-rear",
        "mask_reference": "fixture://generated-mask",
        "intersects_planned_trajectory": True,
    }]
    intersecting["next_cheapest_experiment"] = {
        "kind": fixture["expected_next_experiment"],
        "instructions": "Capture the hidden rear or provide a verified owner asset.",
    }
    normalized = normalize_reconstruction_result(intersecting)
    assert normalized["generated_trajectory_intersection"] == {
        "intersects_planned_trajectory": True,
        "region_ids": ["hidden-rear"],
        "physics_use_allowed": False,
    }
    assert normalized["claim_ceiling"]["trajectory_clearance_established"] is False
    assert normalized["claim_ceiling"]["generated_trajectory_intersection_physics_use"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "reconstruction_capability.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(normalized)

    unsafe = copy.deepcopy(intersecting)
    unsafe["outputs"] = ["appearance_layer", "collision_geometry"]
    unsafe["claim_ceiling"]["generated_regions_excluded_from_physics"] = True
    with pytest.raises(
        ReconstructionContractError,
        match="trajectory_intersection_forbids_physics_output",
    ):
        normalize_reconstruction_result(unsafe)

    missing_experiment = copy.deepcopy(intersecting)
    missing_experiment.pop("next_cheapest_experiment")
    with pytest.raises(
        ReconstructionContractError,
        match="trajectory_intersection_experiment_required",
    ):
        normalize_reconstruction_result(missing_experiment)


def test_simready_is_per_object_per_claim_and_generated_assets_do_not_self_qualify() -> None:
    fixture = _beta_fixture("unqualified_simready_asset")
    assert fixture["asset_required_for_claim"] is True
    visual_only = decide_simready_assets(
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        requested_claim_types=["perception_visibility"],
        task_objects=[{"object_id": "item-1"}],
        asset_candidates=[],
    )
    assert visual_only["object_decisions"][0]["status"] == "not_required"

    physics = decide_simready_assets(
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        requested_claim_types=["collision_contact"],
        task_objects=[{"object_id": "item-1"}],
        asset_candidates=[
            {
                "object_id": "item-1",
                "asset_uri": "fixture://looks-good.usd",
                "asset_digest": SHA_C,
                "source_capture_digest": SHA_A,
                "validation_status": "passed",
                "generated_only": True,
            }
        ],
    )
    assert physics["status"] == "blocked_missing_asset"
    assert physics["object_decisions"][0]["rejected_asset_digests"] == [SHA_C]
    assert fixture["expected_physics_use"] is False

    independently_verified = decide_simready_assets(
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        requested_claim_types=["collision_contact"],
        task_objects=[{"object_id": "item-1"}],
        asset_candidates=[
            {
                "object_id": "item-1",
                "asset_uri": "fixture://verified.usd",
                "asset_digest": SHA_C,
                "source_capture_digest": SHA_A,
                "provider_identity": "provider:asset-maker",
                "validator_identity": "local:independent-validator",
                "validation_status": "passed",
                "generated_only": False,
                "site_to_object_transform": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ],
                "independent_validation": {
                    "scale": True,
                    "site_to_object_transform": True,
                    "support_surface": True,
                    "orientation": True,
                    "penetration": True,
                    "reprojection": True,
                    "physics_properties": True,
                },
            }
        ],
    )
    assert independently_verified["status"] == "complete"
    assert independently_verified["object_decisions"][0]["status"] == (
        "verified_asset_selected"
    )


def _robot_binding() -> dict:
    return {
        "robot_id": "fixture-arm",
        "embodiment_version": "1",
        "base_footprint": {"shape": "circle", "radius_m": 0.4},
        "sensors": {"camera": "rgb-v1"},
        "controller_id": "joint-position-v1",
        "end_effector_id": "parallel-gripper-v1",
    }


def _placement(candidate_id: str, *, coverage: float, reach: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "site_from_robot_base": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "floor_support_valid": True,
        "footprint_clear": True,
        "access_path_clear": True,
        "collision_free": True,
        "reset_feasible": True,
        "human_clearance_valid": True,
        "captured_coverage": coverage,
        "reachability_score": reach,
        "manipulability_score": 0.8,
        "sensor_visibility_score": 0.8,
        "approach_direction_score": 0.8,
        "cable_controller_score": 0.8,
        "stability_score": 0.8,
        "calibration_uncertainty_m": 0.01,
        "method_qualification_status": "qualified",
        "evidence_digests": [SHA_C],
    }


def test_robot_placement_is_deterministic_and_coverage_aware() -> None:
    result = score_robot_placements(
        robot_binding=_robot_binding(),
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        task_object_id="item-1",
        target_region_id="tote-1",
        candidates=[
            _placement("outside-coverage", coverage=0.5, reach=1.0),
            _placement("candidate-b", coverage=0.9, reach=0.8),
            _placement("candidate-a", coverage=0.9, reach=0.8),
        ],
    )

    assert result["selected_candidate_id"] == "candidate-a"
    assert result["rejected_candidates"] == [{
        "candidate_id": "outside-coverage",
        "reasons": ["captured_coverage_insufficient"],
    }]
    assert result["proof_boundary"]["placement_is_physical_deployment_approval"] is False


def test_robot_placement_abstains_and_requests_capture_when_no_candidate_is_covered() -> None:
    fixture = _beta_fixture("robot_placement_outside_captured_coverage")
    assert fixture["placement_area_covered"] is False
    result = score_robot_placements(
        robot_binding=_robot_binding(),
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        task_object_id="item-1",
        target_region_id="tote-1",
        candidates=[_placement("outside-coverage", coverage=0.5, reach=1.0)],
    )

    assert result["status"] == "abstained"
    assert fixture["expected_verdict"] == "abstention"
    assert result["next_cheapest_experiment"]["kind"] == "targeted_recapture_or_measurement"


def test_reconstruction_schema_accepts_all_emitted_artifact_kinds() -> None:
    profile = build_reconstruction_method_profile(
        _method(
            "metric-method",
            "metric_scaffold",
            ["metric_reference_layer"],
            cost=0.0,
        )
    )
    plan = plan_reconstruction_methods(
        intake_id="intake-1",
        capture_digest=SHA_A,
        capture_authority_profile="iphone_arkit_lidar",
        claim_ceiling={},
        requested_claim_types=["reachability"],
        permitted_provider_identities=["local"],
        method_profiles=[profile],
    )
    result = normalize_reconstruction_result(_reconstruction_result())
    simready = decide_simready_assets(
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        requested_claim_types=["perception_visibility"],
        task_objects=[{"object_id": "item-1"}],
        asset_candidates=[],
    )
    placement = score_robot_placements(
        robot_binding=_robot_binding(),
        approved_task_digest=SHA_B,
        capture_digest=SHA_A,
        task_object_id="item-1",
        target_region_id="tote-1",
        candidates=[_placement("candidate-1", coverage=0.9, reach=0.8)],
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "reconstruction_capability.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for artifact in (profile, plan, result, simready, placement):
        validator.validate(artifact)
