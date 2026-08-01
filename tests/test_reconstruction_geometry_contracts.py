from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.isaac_reconstruction_verification import (
    IsaacReconstructionVerificationError,
    build_isaac_asset_verification_request,
    build_isaac_runtime_result_v3,
    normalize_isaac_reconstruction_verification,
)
from blueprint_pipeline.reconstruction_isaac_worker_bundle import (
    IsaacWorkerBundleError,
    compile_isaac_verification_worker_bundle,
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


D = ["sha256:" + str(i) * 64 for i in range(1, 10)]
IMAGE = "registry.example/blueprint/isaac@sha256:" + "a" * 64


def _sha256(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
        appearance_asset_manifest_digest=D[1],
        metric_geometry_manifest_digest=_metric()["metric_geometry_manifest_digest"],
        collider_candidate_manifest_digest=_candidate()["collider_candidate_manifest_digest"],
        collider_qualification_digest=_qualification()["collider_qualification_digest"],
        collider_qualification_decision="accepted_bounded_navigation",
        packaging_request_digest=D[4],
        package_digest=D[5],
        package_artifact_reference="package/reconstruction.usdz",
        package_format="usdz",
        self_contained=True,
        deterministic_archive=True,
        package_member_count=3,
        particlefield_prim_count=1,
        collision_api_prim_count=1,
        missing_asset_count=0,
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


def test_independent_measurement_blocker_forces_collider_rejection():
    with pytest.raises(ReconstructionGeometryContractError, match="collider_decision_not_deterministic"):
        _qualification(blockers=["task_region_measurement_incomplete"])
    report = _qualification(
        blockers=["task_region_measurement_incomplete"], decision="rejected"
    )
    assert report["decision"] == "rejected"


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
            package_digest=D[5],
            isaac_verification_request_digest=D[6],
            isaac_runtime_result_digest=D[7],
            runtime_container_image_digest=IMAGE,
            runtime_implementation_digest=D[8],
            fixed_camera_spec_digest=D[2],
            exact_package_rehash_verified=True,
            runtime_artifact_rehash_verified=True,
            checks=checks,
            fixed_camera_render_references=[{"artifact_id": "camera-1", "digest": D[1]}],
            physics_probe={
                "ground_contact_surface_present": True,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 1,
            },
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
                package_digest=D[5],
                isaac_verification_request_digest=D[6],
                isaac_runtime_result_digest=D[7],
                runtime_container_image_digest=IMAGE,
                runtime_implementation_digest=D[8],
                fixed_camera_spec_digest=D[2],
                exact_package_rehash_verified=True,
                runtime_artifact_rehash_verified=True,
                checks=checks,
                fixed_camera_render_references=[{"artifact_id": "camera-1", "digest": D[1]}],
                physics_probe={
                    "ground_contact_surface_present": True,
                    "live_rigid_body_pose_observed": True,
                    "test_body_fell_through_floor": False,
                    "contact_event_count": 1,
                },
                status="verified_compatibility_only",
                simulator_task_success_proven=False,
                physical_success_proven=False,
                deployment_readiness_proven=False,
                proof_effect="isaac_load_render_physics_presence_only",
                claim_ceiling="isaac_load_render_compatibility",
            )
        )


def _isaac_request(package=None, *, camera_spec_digest=D[6], runner_digest=D[7]):
    package = package or _package()
    request = build_isaac_asset_verification_request(
        _base(
            packaging_result=package,
            packaging_result_digest=package["packaging_result_digest"],
            package_artifact_reference=package["package_artifact_reference"],
            package_digest=package["package_digest"],
            fixed_camera_spec_digest=camera_spec_digest,
            fixed_camera_ids=["fixed-1"],
            runtime_container_image_digest=IMAGE,
            runtime_implementation_digest=runner_digest,
            expected_prim_paths={
                "appearance": "/World/BlueprintReconstruction/Appearance",
                "collision": "/World/BlueprintReconstruction/Collision",
            },
            physics_probe_request={
                "steps": 120,
                "manufacture_ground_plane": False,
                "require_contact_event": True,
                "test_body": {
                    "shape": "cube",
                    "size_m": 0.1,
                    "mass_kg": 1.0,
                    "spawn_height_above_ground_m": 0.5,
                },
                "gravity_m_s2": -9.81,
                "physics_dt_seconds": 1.0 / 60.0,
            },
            headless=True,
            display_attached=False,
            timeout_seconds=1800,
            resource_request={"gpu_count": 1, "minimum_vram_gb": 24},
            input_digests=[
                {"artifact_id": "package_result", "digest": package["packaging_result_digest"]},
                {"artifact_id": "package", "digest": package["package_digest"]},
                {"artifact_id": "cameras", "digest": camera_spec_digest},
                {"artifact_id": "runner", "digest": runner_digest},
            ],
            output_digests=[],
            proof_effect="none",
            claim_ceiling="request_only",
        )
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/isaac_asset_verification_request.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(request)
    return request


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
    request = _isaac_request()

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


def _isaac_fixture(tmp_path):
    package_root = tmp_path / "packages"
    package_path = package_root / "package/reconstruction.usdz"
    package_path.parent.mkdir(parents=True)
    package_path.write_bytes(b"exact-package-bytes")
    package = _package(package_digest=_sha256(package_path))
    request = _isaac_request(package)

    runtime_root = tmp_path / "runtime"
    render = runtime_root / "frames/fixed-1.png"
    render.parent.mkdir(parents=True)
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    Image.fromarray(pixels, mode="RGB").save(render)
    measured = np.asarray(Image.open(render).convert("RGB"), dtype=np.float32)
    runtime = build_isaac_runtime_result_v3(
        {
            "schema_version": "isaac_splat_nurec_render_result.v3",
            "status": "completed",
            "isaac_verification_request_digest": request[
                "isaac_verification_request_digest"
            ],
            "package_digest": package["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": IMAGE,
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_identity": {
                "runtime": "isaac_sim",
                "version": "6.0.0",
                "renderer": "RayTracedLighting",
                "python_version": "3.11.0",
                "headless": True,
            },
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
                "expected_prim_paths": request["expected_prim_paths"],
            },
            "physics_probe": {
                "ground_contact_surface_present": True,
                "steps_executed": 120,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 3,
                "probe_configuration": {
                    "test_body": request["physics_probe_request"]["test_body"],
                    "gravity_m_s2": -9.81,
                    "physics_dt_seconds": 1.0 / 60.0,
                },
            },
            "cameras": [
                {
                    "id": "fixed-1",
                    "artifact_reference": "frames/fixed-1.png",
                    "digest": _sha256(render),
                    "width": 16,
                    "height": 12,
                    "pixel_mean": float(measured.mean()),
                    "pixel_std": float(measured.std()),
                    "nonblank": True,
                }
            ],
            "cost_usd": 0.4,
            "duration_seconds": 60.0,
            "proof_boundary": {
                "isaac_load_render_physics_presence_compatibility": True,
                "simulator_task_success_proven": False,
                "physics_navigation_control_proven": False,
                "physical_success_proven": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
            },
        }
    )
    return request, runtime, package_root, runtime_root


def test_isaac_v3_normalizer_rehashes_exact_package_and_png(tmp_path):
    request, runtime, package_root, runtime_root = _isaac_fixture(tmp_path)
    result = normalize_isaac_reconstruction_verification(
        verification_request=request,
        runtime_result=runtime,
        package_artifact_root=package_root,
        runtime_artifact_root=runtime_root,
    )
    assert result["status"] == "verified_compatibility_only"
    assert result["exact_package_rehash_verified"] is True
    assert result["runtime_artifact_rehash_verified"] is True
    assert result["simulator_task_success_proven"] is False
    result_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/isaac_asset_verification_result.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(result_schema).validate(result)


def test_isaac_worker_bundle_binds_exact_package_cameras_runner_without_spend(tmp_path):
    package_root = tmp_path / "packages"
    package_path = package_root / "package/reconstruction.usdz"
    package_path.parent.mkdir(parents=True)
    package_path.write_bytes(b"exact-package-bytes")
    cameras = tmp_path / "fixed_cameras.json"
    cameras.write_text('[{"id":"fixed-1","spec":{}}]\n', encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("print('fixture')\n", encoding="utf-8")
    package = _package(package_digest=_sha256(package_path))
    request = _isaac_request(
        package,
        camera_spec_digest=_sha256(cameras),
        runner_digest=_sha256(runner),
    )
    receipt = compile_isaac_verification_worker_bundle(
        verification_request=request,
        package_artifact_root=package_root,
        fixed_camera_spec_path=cameras,
        runner_path=runner,
        output_root=tmp_path / "bundles",
    )
    replay = compile_isaac_verification_worker_bundle(
        verification_request=request,
        package_artifact_root=package_root,
        fixed_camera_spec_path=cameras,
        runner_path=runner,
        output_root=tmp_path / "bundles",
    )
    assert replay == receipt
    assert receipt["provider_allocation_performed"] is False
    assert receipt["paid_execution_authorized_by_bundle"] is False
    assert receipt["expected_runtime_schema"] == "isaac_splat_nurec_render_result.v3"
    assert receipt["canonical_allocator_command"].endswith("gpu-canary")
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/isaac_verification_worker_bundle.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)

    bundle_path = (
        tmp_path
        / "bundles"
        / request["isaac_verification_request_digest"][7:]
        / "isaac_verification_worker_bundle.zip"
    )
    bundle_path.unlink()
    with pytest.raises(IsaacWorkerBundleError, match="existing_output_tampered"):
        compile_isaac_verification_worker_bundle(
            verification_request=request,
            package_artifact_root=package_root,
            fixed_camera_spec_path=cameras,
            runner_path=runner,
            output_root=tmp_path / "bundles",
        )


def test_visual_only_v2_runtime_cannot_pass_v3_verification(tmp_path):
    request, runtime, package_root, runtime_root = _isaac_fixture(tmp_path)
    runtime.pop("isaac_runtime_result_digest")
    runtime["schema_version"] = "isaac_splat_nurec_render_result.v2"
    runtime.pop("physics_probe")
    with pytest.raises(
        IsaacReconstructionVerificationError, match="isaac_runtime_result_v3_required"
    ):
        normalize_isaac_reconstruction_verification(
            verification_request=request,
            runtime_result=runtime,
            package_artifact_root=package_root,
            runtime_artifact_root=runtime_root,
        )


def test_falling_through_floor_missing_contact_and_tampered_png_fail_closed(tmp_path):
    request, runtime, package_root, runtime_root = _isaac_fixture(tmp_path)
    runtime.pop("isaac_runtime_result_digest")
    runtime["physics_probe"]["test_body_fell_through_floor"] = True
    runtime["physics_probe"]["contact_event_count"] = 0
    runtime = build_isaac_runtime_result_v3(runtime)
    (runtime_root / "frames/fixed-1.png").write_bytes(b"tampered")
    with pytest.raises(IsaacReconstructionVerificationError) as error:
        normalize_isaac_reconstruction_verification(
            verification_request=request,
            runtime_result=runtime,
            package_artifact_root=package_root,
            runtime_artifact_root=runtime_root,
        )
    assert "isaac_test_body_fell_through_floor" in str(error.value)
    assert "isaac_test_body_contact_not_observed" in str(error.value)
    assert "isaac_fixed_render:0_digest_mismatch" in str(error.value)


def test_forged_v3_claim_and_secret_state_are_rejected_before_artifact_use(tmp_path):
    _request, runtime, _package_root, _runtime_root = _isaac_fixture(tmp_path)
    runtime.pop("isaac_runtime_result_digest")
    runtime["raw_secret_values_recorded"] = True
    runtime["proof_boundary"]["physical_success_proven"] = True
    with pytest.raises(IsaacReconstructionVerificationError) as error:
        build_isaac_runtime_result_v3(runtime)
    message = str(error.value)
    assert "isaac_runtime_secret_recording_state_invalid" in message
    assert "isaac_forbidden_claim_promotion:physical_success_proven" in message
