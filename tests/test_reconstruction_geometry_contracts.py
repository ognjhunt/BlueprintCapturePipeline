from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.isaac_reconstruction_verification import (
    IsaacReconstructionVerificationError,
    normalize_isaac_reconstruction_verification,
)
from blueprint_pipeline.reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_isaac_asset_verification_result,
    build_metric_geometry_manifest,
    build_nurec_openusd_packaging_result,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry, non_spend_tool_bindings


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


def test_phase5_tools_are_registered_digest_only_and_metric_runtime_cannot_change_proof(tmp_path):
    registry = ToolRegistry.default()
    expected = {
        "compile_metric_geometry": "source_artifact_digest",
        "compile_collision_candidate": "metric_geometry_manifest_digest",
        "qualify_collision_candidate": "collider_candidate_manifest_digest",
        "package_nurec_openusd": "packaging_request_digest",
        "verify_isaac_asset": "isaac_verification_request_digest",
    }
    for tool_id, field in expected.items():
        descriptor = registry.resolve(tool_id)
        assert descriptor is not None
        assert set(descriptor.to_mapping()["input_schema"]["properties"]) == {field}

    source = {"source_artifact_digest": D[0]}

    def compiler(*, source_artifact, output_root):
        assert source_artifact == source
        assert output_root.name == "compile_metric_geometry"
        return _metric()

    context = SupervisorContext(
        run_id="metric-geometry-tool",
        customer_question="Compile metric geometry",
        supervisor_output_dir=str(tmp_path),
        metric_geometry_source=source,
        metric_geometry_compiler=compiler,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[D[0]],
    ).to_mapping()
    bindings = {
        item.tool_id: item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }
    observation = bindings["compile_metric_geometry"].invoke(
        {"source_artifact_digest": D[0]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["claim_ceiling"] == "metric_reference_geometry"
    assert observation["typed_result"]["proof_state_changed"] is False
    assert observation["proof_effect"] == "none"


def test_phase5_request_tools_refuse_tampered_request_digest(tmp_path):
    request = {"packaging_result_digest": D[0]}
    request["isaac_verification_request_digest"] = canonical_digest(
        request, digest_field="isaac_verification_request_digest"
    )

    def verifier(**_kwargs):
        raise AssertionError("tampered request must stop before runtime")

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="isaac-request-binding",
        customer_question="Verify Isaac package",
        supervisor_output_dir=str(tmp_path),
        isaac_verification_request=request,
        isaac_asset_verifier=verifier,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[D[0]],
    ).to_mapping()
    bindings = {
        item.tool_id: item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }
    refused = bindings["verify_isaac_asset"].invoke(
        {"isaac_verification_request_digest": D[1]}
    )
    assert refused["status"] == "refused"
    assert "source_digest_mismatch" in refused["typed_failure"]["reason"]


def _isaac_runtime_v2():
    return {
        "schema_version": "isaac_splat_nurec_render_result.v2",
        "status": "completed",
        "package_digest": D[5],
        "raw_secret_values_recorded": False,
        "stage": {
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "transforms_valid": True,
            "dependency_inspection_available": True,
            "missing_asset_count": 0,
            "particlefield_prim_count": 1,
            "active_collision_prim_count": 2,
            "obvious_scale_mismatch_detected": False,
        },
        "physics_probe": {
            "ground_contact_surface_present": True,
            "steps_executed": 120,
            "live_rigid_body_pose_observed": True,
            "test_body_fell_through_floor": False,
            "contact_event_count": 3,
        },
        "cameras": [
            {"id": "fixed-1", "digest": D[1], "pixel_std": 12.0, "nonblank": True}
        ],
        "proof_boundary": {
            "isaac_load_render_physics_presence_compatibility": True,
            "simulator_task_success_proven": False,
            "physics_navigation_control_proven": False,
            "physical_success_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }


def test_isaac_runtime_v2_normalizer_requires_real_render_and_physics_presence():
    result = normalize_isaac_reconstruction_verification(
        packaging_result=_package(), runtime_result=_isaac_runtime_v2(), lineage=_base()
    )
    assert result["status"] == "verified_compatibility_only"
    assert result["checks"]["collision_geometry_active"] is True
    assert result["simulator_task_success_proven"] is False


def test_visual_only_v1_runtime_cannot_pass_physics_verification():
    runtime = _isaac_runtime_v2()
    runtime["schema_version"] = "isaac_splat_nurec_render_result.v1"
    runtime.pop("physics_probe")
    with pytest.raises(
        IsaacReconstructionVerificationError, match="isaac_runtime_result_v2_required"
    ):
        normalize_isaac_reconstruction_verification(
            packaging_result=_package(), runtime_result=runtime, lineage=_base()
        )


def test_falling_through_floor_and_missing_contact_fail_closed():
    runtime = _isaac_runtime_v2()
    runtime["physics_probe"]["test_body_fell_through_floor"] = True
    runtime["physics_probe"]["contact_event_count"] = 0
    with pytest.raises(IsaacReconstructionVerificationError) as error:
        normalize_isaac_reconstruction_verification(
            packaging_result=_package(), runtime_result=runtime, lineage=_base()
        )
    assert "isaac_test_body_fell_through_floor" in str(error.value)
    assert "isaac_test_body_contact_not_observed" in str(error.value)


def test_forged_v2_label_cannot_bypass_runtime_evidence_or_claim_boundary():
    runtime = _isaac_runtime_v2()
    runtime["raw_secret_values_recorded"] = True
    runtime["stage"]["dependency_inspection_available"] = False
    runtime["stage"]["obvious_scale_mismatch_detected"] = True
    runtime["physics_probe"]["live_rigid_body_pose_observed"] = False
    runtime["cameras"][0]["digest"] = "not-a-digest"
    runtime["proof_boundary"]["physical_success_proven"] = True
    with pytest.raises(IsaacReconstructionVerificationError) as error:
        normalize_isaac_reconstruction_verification(
            packaging_result=_package(), runtime_result=runtime, lineage=_base()
        )
    message = str(error.value)
    assert "isaac_runtime_secret_recording_state_invalid" in message
    assert "isaac_dependency_inspection_unavailable" in message
    assert "isaac_obvious_scale_mismatch" in message
    assert "isaac_test_body_pose_unavailable" in message
    assert "isaac_fixed_render_invalid:0" in message
    assert "isaac_forbidden_claim_promotion:physical_success_proven" in message
