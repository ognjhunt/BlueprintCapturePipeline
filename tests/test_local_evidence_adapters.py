from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_evidence_adapters import (
    ANALYTIC_REACHABILITY_ADAPTER,
    CAPTURED_VISIBILITY_ADAPTER,
    PROCESSED_OBSERVATION_VISIBILITY_ADAPTER,
    SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER,
    SWEPT_AABB_COLLISION_SIMULATION_ADAPTER,
    AnalyticReachabilityAdapter,
    CapturedVisibilityAdapter,
    ProcessedObservationVisibilityAdapter,
    SignedIsaacInspectionRankingReplayAdapter,
    SweptAabbCollisionSimulationAdapter,
    authorized_local_evidence_adapter_registry,
)


def _testbed() -> dict:
    scene = {
        "schema_version": "collision_scene_aabb.v1",
        "source_capture_digest": "sha256:" + "f" * 64,
        "coordinate_frame": "site",
        "scale_status": "metric_verified",
        "generated_geometry": False,
        "primitives": [
            {
                "primitive_id": "table-obstacle",
                "object_id": "table-obstacle",
                "minimum_site_m": [0.4, -0.2, 0.0],
                "maximum_site_m": [0.6, 0.2, 0.8],
            },
            {
                "primitive_id": "item-1",
                "object_id": "item-1",
                "minimum_site_m": [0.0, 0.0, 0.0],
                "maximum_site_m": [0.1, 0.1, 0.1],
            },
        ],
        "validation": {
            "status": "qualified",
            "independent_validation": True,
            "coverage": 0.9,
            "maximum_spatial_uncertainty_m": 0.01,
        },
    }
    scene["collision_scene_digest"] = canonical_digest(scene, digest_field="collision_scene_digest")
    return {
        "source_capture_bundles": [
            {"bundle_id": "capture-1", "version": "3", "digest": "sha256:" + "f" * 64}
        ],
        "robot_sensor_controller_bindings": {
            "embodiment": {
                "reach_envelope": {"minimum_m": 0.1, "maximum_m": 1.0},
            },
            "selected_robot_placement": {
                "candidate_id": "base-1",
                "base_position_site_m": [0.0, 0.0, 0.0],
                "captured_coverage": 0.95,
                "calibration_uncertainty_m": 0.01,
                "method_qualification_status": "analytic_only",
            },
        },
        "target_regions": [
            {
                "region_id": "tote-1",
                "position_site_m": [0.6, 0.1, 0.7],
                "supporting_frames": ["frame-2", "frame-1", "frame-1"],
                "captured_coverage": 0.9,
            }
        ],
        "validation_envelope": {
            "robot_placement_digest": "sha256:" + "a" * 64,
            "reconstruction_layers": {
                "physics_layer": [
                    {
                        "output": "collision_geometry",
                        "result_id": "collision-result-1",
                        "result_digest": "sha256:" + "e" * 64,
                        "asset_references": {"collision_scene": scene},
                        "generated_regions": [],
                        "claim_ceiling": {"collision_geometry": True},
                    }
                ],
            },
        },
    }


def test_analytic_reachability_uses_only_explicit_metric_inputs() -> None:
    result = AnalyticReachabilityAdapter().execute(
        claim={
            "claim_type": "analytic_reachability",
            "subject": "tote-1",
        },
        testbed=_testbed(),
    )
    assert result["status"] == "valid"
    assert result["supports_claim"] is True
    assert result["claim_ceiling"]["physical_success"] is False
    assert result["provenance"]["physical_robot_run_initiated"] is False

    missing = _testbed()
    missing["robot_sensor_controller_bindings"]["selected_robot_placement"].pop(
        "base_position_site_m"
    )
    abstention = AnalyticReachabilityAdapter().execute(
        claim={
            "claim_type": "analytic_reachability",
            "subject": "tote-1",
        },
        testbed=missing,
    )
    assert abstention["status"] == "unavailable"
    assert abstention["supports_claim"] is None
    assert "robot_base_metric_position_missing" in abstention["blockers"]


def test_analytic_reachability_abstains_at_uncertain_boundary() -> None:
    result = AnalyticReachabilityAdapter().execute(
        claim={
            "claim_type": "reachability",
            "subject": {"target_position_site_m": [0.995, 0.0, 0.0]},
        },
        testbed=_testbed(),
    )
    assert result["status"] == "uncertain"
    assert result["supports_claim"] is None
    assert result["categorical_finding"] == "reach_boundary_uncertain"


def test_captured_visibility_binds_exact_region_and_retained_frames() -> None:
    result = CapturedVisibilityAdapter().execute(
        claim={
            "claim_type": "captured_visibility",
            "subject": {"target_region_id": "tote-1"},
        },
        testbed=_testbed(),
    )
    assert result["status"] == "valid"
    assert result["coverage"] == 0.9
    assert result["raw_artifact_references"] == [
        {"uri": "capture-frame://frame-1", "frame_id": "frame-1"},
        {"uri": "capture-frame://frame-2", "frame_id": "frame-2"},
    ]
    assert result["claim_ceiling"]["metric_geometry"] is False

    missing = CapturedVisibilityAdapter().execute(
        claim={
            "claim_type": "captured_visibility",
            "subject": {"target_region_id": "occluded-region"},
        },
        testbed=_testbed(),
    )
    assert missing["status"] == "unavailable"
    assert missing["supports_claim"] is None


def test_processed_observation_visibility_never_claims_raw_capture() -> None:
    testbed = _testbed()
    testbed["validation_envelope"]["capture_authority_profile"] = (
        "public_processed_rgbd_pose_sequence"
    )
    testbed["evidence_inventory"] = [
        {
            "evidence_id": "processed_capture_observations",
            "digest": "sha256:" + "f" * 64,
            "raw_capture_authority": False,
        }
    ]

    result = ProcessedObservationVisibilityAdapter().execute(
        claim={
            "claim_type": "perception_visibility",
            "subject": {"target_region_id": "tote-1"},
        },
        testbed=testbed,
    )

    assert result["status"] == "valid"
    assert result["categorical_finding"] == ("target_region_visible_in_processed_dataset_views")
    assert result["provenance"]["raw_capture_authority"] is False
    assert result["claim_ceiling"]["processed_captured_observation_visibility"] is True
    assert result["claim_ceiling"]["metric_geometry"] is False
    assert result["claim_ceiling"]["physical_success"] is False

    wrong_profile = _testbed()
    abstention = ProcessedObservationVisibilityAdapter().execute(
        claim={"claim_type": "perception_visibility", "subject": "tote-1"},
        testbed=wrong_profile,
    )
    assert abstention["status"] == "unavailable"
    assert "processed_observation_profile_required" in abstention["blockers"]


def test_collision_simulation_is_metric_qualified_and_sim_only() -> None:
    clear = SweptAabbCollisionSimulationAdapter().execute(
        claim={
            "claim_type": "collision_contact",
            "subject": {
                "trajectory_points_site_m": [[0.0, 0.8, 1.0], [1.0, 0.8, 1.0]],
                "swept_radius_m": 0.05,
                "excluded_collision_object_ids": ["item-1"],
            },
        },
        testbed=_testbed(),
    )
    assert clear["status"] == "valid"
    assert clear["supports_claim"] is True
    assert clear["categorical_finding"] == "modeled_trajectory_collision_free"
    assert clear["claim_ceiling"]["sim_only_modeled_collision_clearance"] is True
    assert clear["claim_ceiling"]["physical_success"] is False

    collision = SweptAabbCollisionSimulationAdapter().execute(
        claim={
            "claim_type": "collision_contact",
            "subject": {
                "trajectory_points_site_m": [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
                "swept_radius_m": 0.05,
                "excluded_collision_object_ids": ["item-1"],
            },
        },
        testbed=_testbed(),
    )
    assert collision["status"] == "valid"
    assert collision["supports_claim"] is False
    assert collision["provenance"]["contact_object_ids"] == ["table-obstacle"]


def test_collision_simulation_rejects_generated_or_tampered_physics() -> None:
    generated = _testbed()
    generated["validation_envelope"]["reconstruction_layers"]["physics_layer"][0][
        "generated_regions"
    ] = [{"region_id": "hidden", "mask_reference": "fixture://mask"}]
    result = SweptAabbCollisionSimulationAdapter().execute(
        claim={
            "claim_type": "collision_contact",
            "subject": {
                "trajectory_points_site_m": [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
                "swept_radius_m": 0.05,
            },
        },
        testbed=generated,
    )
    assert result["status"] == "unavailable"
    assert "generated_region_cannot_supply_collision_geometry" in result["blockers"]

    tampered = _testbed()
    scene = tampered["validation_envelope"]["reconstruction_layers"]["physics_layer"][0][
        "asset_references"
    ]["collision_scene"]
    scene["validation"]["coverage"] = 1.0
    result = SweptAabbCollisionSimulationAdapter().execute(
        claim={
            "claim_type": "collision_contact",
            "subject": {
                "trajectory_points_site_m": [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
                "swept_radius_m": 0.05,
            },
        },
        testbed=tampered,
    )
    assert result["status"] == "unavailable"
    assert "collision_scene_digest_invalid" in result["blockers"]


def test_inspection_ranking_replay_supports_controllers_but_not_policy_ranking() -> None:
    ranking = {
        "schema_version": "external_scene_inspection_candidate_ranking.v2",
        "contract_digest": "sha256:" + "a" * 64,
        "status": "completed",
        "candidate_results": [
            {"candidate_id": "controller-a", "status": "qualified"},
            {"candidate_id": "controller-b", "status": "qualified"},
        ],
        "ranking": [
            {"rank": 1, "candidate_id": "controller-a", "score": 0.9},
            {"rank": 2, "candidate_id": "controller-b", "score": 0.8},
        ],
        "controller_candidate_ranking_proven": True,
        "learned_policy_ranking_proven": False,
        "blockers": [],
    }
    ranking["ranking_digest"] = canonical_digest(ranking, digest_field="ranking_digest")
    testbed = _testbed()
    testbed["evidence_inventory"] = [
        {"evidence_id": "signed_isaac_inspection_candidate_ranking", **ranking}
    ]
    claim = {
        "claim_type": "comparative_controller_ranking",
        "subject": {"candidate_controller_ids": ["controller-a", "controller-b"]},
    }

    result = SignedIsaacInspectionRankingReplayAdapter().execute(
        claim=claim,
        testbed=testbed,
    )

    assert result["status"] == "valid"
    assert result["supports_claim"] is True
    assert result["claim_ceiling"]["comparative_controller_ranking"] is True
    assert result["claim_ceiling"]["comparative_policy_ranking"] is False

    testbed["evidence_inventory"][0]["ranking"][0]["score"] = 1.0
    tampered = SignedIsaacInspectionRankingReplayAdapter().execute(
        claim=claim,
        testbed=testbed,
    )
    assert tampered["status"] == "unavailable"


def test_local_registry_is_empty_by_default_and_rejects_unknown_authority() -> None:
    empty = authorized_local_evidence_adapter_registry([])
    assert empty.manifest() == []
    authorized = authorized_local_evidence_adapter_registry(
        [
            CAPTURED_VISIBILITY_ADAPTER,
            PROCESSED_OBSERVATION_VISIBILITY_ADAPTER,
            SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER,
            ANALYTIC_REACHABILITY_ADAPTER,
            SWEPT_AABB_COLLISION_SIMULATION_ADAPTER,
        ]
    )
    assert authorized.manifest() == [
        ANALYTIC_REACHABILITY_ADAPTER,
        CAPTURED_VISIBILITY_ADAPTER,
        PROCESSED_OBSERVATION_VISIBILITY_ADAPTER,
        SIGNED_ISAAC_INSPECTION_RANKING_ADAPTER,
        SWEPT_AABB_COLLISION_SIMULATION_ADAPTER,
    ]
    with pytest.raises(ValueError, match="local_evidence_adapter_not_registered"):
        authorized_local_evidence_adapter_registry(["provider://live-not-authorized"])
