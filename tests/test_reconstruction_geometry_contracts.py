from __future__ import annotations

import pytest

from blueprint_pipeline.reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_isaac_asset_verification_result,
    build_metric_geometry_manifest,
    build_nurec_openusd_packaging_result,
)


D = ["sha256:" + str(i) * 64 for i in range(1, 7)]


def _base(**updates):
    value = {
        "stable_run_identity": "run-1",
        "source_capture_identity": "capture-1",
        "source_capture_digest": D[0],
        "original_file_references": [{"artifact_id": "raw", "digest": D[1]}],
        "producing_method": "fixture",
        "implementation_version": "1",
        "source_commit_sha": "a" * 40,
        "deterministic_configuration_digest": D[2],
        "input_digests": [{"artifact_id": "input", "digest": D[3]}],
        "output_digests": [],
        "train_heldout_split_digest": D[4],
        "camera_calibration_binding": {"digest": D[1]},
        "coordinate_frame_declaration": {"frame": "world", "up": "Z"},
        "units": "meters",
        "provider_runtime_identity": {"provider": "local"},
        "cost_usd": 0.0,
        "duration_seconds": 1.0,
        "authority_used": {"mode": "execute_non_spend"},
        "warnings": [],
        "blockers": [],
        "parent_artifact_or_event": {"digest": D[0]},
        "timestamp": "2026-07-30T12:00:00Z",
    }
    value.update(updates)
    return value


def _metric(**updates):
    value = _base(
        metric_scale_status="validated",
        generated_fill_used=False,
        appearance_asset_used_as_geometry_truth=False,
        observed_region_ids=["floor", "wall-a"],
        unsupported_region_ids=["behind-cabinet"],
        confidence_filter={"minimum_confidence": 2, "missing_depth_preserved": True},
        geometry_asset_digest=D[5],
        proof_effect="metric_reference_candidate_only",
        claim_ceiling="metric_reference_geometry",
    )
    value.update(updates)
    return build_metric_geometry_manifest(value)


def _candidate(**updates):
    value = _base(
        metric_geometry_manifest_digest=_metric()["metric_geometry_manifest_digest"],
        collider_asset_digest=D[5],
        unobserved_regions_filled=False,
        collision_validated=False,
        component_statistics={"count": 2, "disconnected_count": 1},
        hole_statistics={"count": 3, "area_m2": 0.2},
        proof_effect="collision_candidate_only",
        claim_ceiling="collision_geometry_candidate",
    )
    value.update(updates)
    return build_collider_candidate_manifest(value)


def _qualification(**updates):
    measurements = {
        "scale_error_fraction": 0.01,
        "gravity_alignment_error_deg": 1.0,
        "floor_height_residual_m": 0.01,
        "wall_offset_residual_m": 0.02,
        "visual_to_collider_disagreement_m": 0.02,
        "clearance_error_m": 0.03,
        "mesh_coverage_fraction": 0.95,
        "minimum_obstacle_thickness_m": 0.04,
    }
    thresholds = {
        "scale_error_fraction": 0.03,
        "gravity_alignment_error_deg": 3.0,
        "floor_height_residual_m": 0.03,
        "wall_offset_residual_m": 0.05,
        "visual_to_collider_disagreement_m": 0.05,
        "clearance_error_m": 0.05,
        "mesh_coverage_fraction": 0.9,
        "minimum_obstacle_thickness_m": 0.03,
    }
    value = _base(
        collider_candidate_manifest_digest=_candidate()["collider_candidate_manifest_digest"],
        qa_thresholds_digest=D[5],
        measurements=measurements,
        thresholds=thresholds,
        metric_scale_status="validated",
        robot_footprint_navigability_checked=True,
        decision="accepted_bounded_navigation",
        unsupported_claims=["grasping", "articulation", "contact_force", "deployment"],
        proof_effect="bounded_navigation_collision_qualification",
        claim_ceiling="bounded_navigation_simulation",
    )
    value.update(updates)
    return build_collider_qualification_report(value)


def _package(**updates):
    value = _base(
        appearance_asset_digest=D[0],
        metric_geometry_manifest_digest=_metric()["metric_geometry_manifest_digest"],
        collider_candidate_manifest_digest=_candidate()["collider_candidate_manifest_digest"],
        package_digest=D[5],
        stage_meters_per_unit=1.0,
        up_axis="Z",
        shared_visual_physics_frame=True,
        appearance_prim_present=True,
        collision_prim_present=True,
        collision_api_configured=True,
        proof_effect="packaging_compatibility_candidate_only",
        claim_ceiling="openusd_package",
    )
    value.update(updates)
    return build_nurec_openusd_packaging_result(value)


def test_metric_geometry_preserves_unobserved_regions_and_rejects_generated_fill():
    manifest = _metric()
    assert manifest["unsupported_region_ids"] == ["behind-cabinet"]
    with pytest.raises(ReconstructionGeometryContractError, match="generated_or_unseen_fill_forbidden"):
        _metric(generated_fill_used=True)


def test_appearance_asset_cannot_be_promoted_to_metric_truth():
    with pytest.raises(ReconstructionGeometryContractError, match="appearance_cannot_be_geometry_truth"):
        _metric(appearance_asset_used_as_geometry_truth=True)


def test_collider_is_a_candidate_until_independently_qualified():
    candidate = _candidate()
    assert candidate["collision_validated"] is False
    with pytest.raises(ReconstructionGeometryContractError, match="candidate_cannot_declare_collision_valid"):
        _candidate(collision_validated=True)


def test_collider_qualification_is_deterministic_and_navigation_bounded():
    report = _qualification()
    assert report["decision"] == "accepted_bounded_navigation"
    assert "grasping" in report["unsupported_claims"]
    failed = dict(report["measurements"])
    failed["clearance_error_m"] = 0.5
    with pytest.raises(ReconstructionGeometryContractError, match="collider_decision_not_deterministic"):
        _qualification(measurements=failed)


def test_unknown_metric_scale_cannot_accept_collider():
    with pytest.raises(ReconstructionGeometryContractError, match="collider_decision_not_deterministic"):
        _qualification(metric_scale_status="anchor_required")


def test_openusd_package_requires_visual_and_collision_prims_in_one_metric_frame():
    package = _package()
    assert package["stage_meters_per_unit"] == 1.0
    with pytest.raises(ReconstructionGeometryContractError, match="visual_physics_frame_mismatch"):
        _package(shared_visual_physics_frame=False)


def test_isaac_verification_requires_render_and_active_collision_without_claim_promotion():
    checks = {
        "exact_package_opened": True,
        "expected_prims_present": True,
        "stage_units_valid": True,
        "transforms_valid": True,
        "missing_assets_detected": False,
        "particlefield_loaded": True,
        "collision_geometry_active": True,
        "ground_contact_surface_present": True,
        "test_body_fell_through_floor": False,
        "fixed_camera_renders_nonblank": True,
        "nan_or_corrupt_render_detected": False,
        "obvious_scale_mismatch_detected": False,
    }
    result = build_isaac_asset_verification_result(
        _base(
            packaging_result_digest=_package()["packaging_result_digest"],
            checks=checks,
            fixed_camera_render_references=[{"artifact_id": "camera-1", "digest": D[1]}],
            status="verified_compatibility_only",
            simulator_task_success_proven=False,
            physical_success_proven=False,
            deployment_readiness_proven=False,
            proof_effect="isaac_load_render_physics_presence_only",
            claim_ceiling="isaac_load_render_compatibility",
        )
    )
    assert result["claim_ceiling"] == "isaac_load_render_compatibility"
    checks["collision_geometry_active"] = False
    with pytest.raises(ReconstructionGeometryContractError, match="isaac_required_checks_failed"):
        build_isaac_asset_verification_result(
            _base(
                packaging_result_digest=_package()["packaging_result_digest"],
                checks=checks,
                fixed_camera_render_references=[{"artifact_id": "camera-1", "digest": D[1]}],
                status="verified_compatibility_only",
                simulator_task_success_proven=False,
                physical_success_proven=False,
                deployment_readiness_proven=False,
                proof_effect="isaac_load_render_physics_presence_only",
                claim_ceiling="isaac_load_render_compatibility",
            )
        )
