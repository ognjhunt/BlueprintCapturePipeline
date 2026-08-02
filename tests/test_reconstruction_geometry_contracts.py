from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import reconstruction_isaac_bootstrap as isaac_bootstrap
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.isaac_reconstruction_verification import (
    IsaacReconstructionVerificationError,
    build_isaac_asset_verification_request,
    build_isaac_runtime_result_v3,
    normalize_isaac_reconstruction_verification,
)
from blueprint_pipeline.reconstruction_isaac_worker_bundle import (
    IsaacWorkerBundleError,
    compile_isaac_verification_worker_bundle,
    extract_isaac_verification_worker_bundle,
    validate_isaac_verification_worker_bundle_receipt,
)
from blueprint_pipeline.reconstruction_isaac_output_bundle import (
    IsaacVerificationOutputBundleError,
    compile_isaac_verification_output_bundle,
    validate_and_extract_isaac_verification_output_bundle,
)
from blueprint_pipeline.reconstruction_isaac_vast_operation import (
    ReconstructionIsaacVastError,
    replay_reconstruction_isaac_vast_operation,
    run_reconstruction_isaac_vast_operation,
)
from blueprint_pipeline.reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_isaac_asset_verification_result,
    build_metric_geometry_manifest,
    build_nurec_openusd_packaging_result,
)
from blueprint_pipeline.safe_outbound_http import SafeHttpFileTransfer
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
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


def _isaac_bundle_receipt_for_request(request):
    value = {
        "schema_version": "isaac_verification_worker_bundle.v1",
        "isaac_verification_request_digest": request[
            "isaac_verification_request_digest"
        ],
        "package_digest": request["package_digest"],
        "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
        "runtime_implementation_digest": request["runtime_implementation_digest"],
        "runtime_container_image_digest": request[
            "runtime_container_image_digest"
        ],
        "source_commit_sha": request["source_commit_sha"],
        "fixed_camera_ids": request["fixed_camera_ids"],
        "command": [
            "/isaac-sim/python.sh",
            "/workspace/bundle/run_isaac_splat_nurec_render.py",
            *[f"arg-{index}" for index in range(19)],
        ],
        "expected_runtime_schema": "isaac_splat_nurec_render_result.v3",
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "canonical_allocator_command": (
            "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
        ),
        "bundle_manifest_digest": D[7],
        "bundle_digest": D[8],
        "bundle_artifact_reference": (
            f"{request['isaac_verification_request_digest'][7:]}/"
            "isaac_verification_worker_bundle.zip"
        ),
        "bundle_member_count": 5,
        "bundle_bytes": 100,
        "cost_usd": 0.0,
        "proof_effect": "none",
        "claim_ceiling": "request_only",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _isaac_bootstrap_fixture(tmp_path):
    package_root = tmp_path / "bootstrap-packages"
    package_path = package_root / "package/reconstruction.usdz"
    package_path.parent.mkdir(parents=True)
    package_path.write_bytes(b"exact-bootstrap-package")
    cameras = tmp_path / "bootstrap-fixed-cameras.json"
    cameras.write_text('[{"id":"fixed-1","spec":{}}]\n', encoding="utf-8")
    runner = tmp_path / "bootstrap-runner.py"
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
        output_root=tmp_path / "bootstrap-bundles",
    )
    bundle_path = (
        tmp_path
        / "bootstrap-bundles"
        / request["isaac_verification_request_digest"][7:]
        / "isaac_verification_worker_bundle.zip"
    )
    receipt_bytes = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return request, receipt, bundle_path, receipt_bytes


def _isaac_bootstrap_environment(request, receipt, receipt_bytes, **updates):
    value = {
        "BLUEPRINT_ISAAC_INPUT_BUNDLE_GET_URL": (
            "https://objects.example/input.zip?signature=input-secret"
        ),
        "BLUEPRINT_ISAAC_INPUT_RECEIPT_GET_URL": (
            "https://objects.example/receipt.json?signature=receipt-secret"
        ),
        "BLUEPRINT_ISAAC_OUTPUT_BUNDLE_PUT_URL": (
            "https://objects.example/output.zip?signature=output-secret"
        ),
        "BLUEPRINT_ISAAC_INPUT_BUNDLE_DIGEST": receipt["bundle_digest"],
        "BLUEPRINT_ISAAC_INPUT_RECEIPT_FILE_DIGEST": (
            "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
        ),
        "BLUEPRINT_ISAAC_VERIFICATION_REQUEST_DIGEST": request[
            "isaac_verification_request_digest"
        ],
        "BLUEPRINT_CONTAINER_IMAGE_DIGEST": request[
            "runtime_container_image_digest"
        ],
        "BLUEPRINT_SOURCE_COMMIT": request["source_commit_sha"],
        "BLUEPRINT_RECONSTRUCTION_HARD_TTL_SECONDS": "3600",
    }
    value.update(updates)
    return value


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


def test_isaac_runtime_output_bundle_is_complete_deterministic_and_independent(
    tmp_path,
):
    request, runtime, package_root, runtime_root = _isaac_fixture(tmp_path)
    (runtime_root / "isaac_runtime_result.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    receipt = _isaac_bundle_receipt_for_request(request)
    first = tmp_path / "isaac-output-first.zip"
    second = tmp_path / "isaac-output-second.zip"
    first_receipt = compile_isaac_verification_output_bundle(
        bundle_receipt=receipt,
        runtime_output_root=runtime_root,
        output_path=first,
    )
    second_receipt = compile_isaac_verification_output_bundle(
        bundle_receipt=receipt,
        runtime_output_root=runtime_root,
        output_path=second,
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["output_bundle_digest"] == second_receipt[
        "output_bundle_digest"
    ]
    validated, replayed_runtime, extracted = (
        validate_and_extract_isaac_verification_output_bundle(
            bundle_path=first,
            expected_input_receipt=receipt,
            expected_source_commit_sha=request["source_commit_sha"],
            output_root=tmp_path / "validated-output",
        )
    )
    assert replayed_runtime == runtime
    assert validated["scientific_qualification_inferred"] is False
    assert validated["simulator_task_success_proven"] is False
    output_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/isaac_verification_output_bundle.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(output_schema).validate(validated)
    assert (extracted / "frames/fixed-1.png").is_file()
    qualified = normalize_isaac_reconstruction_verification(
        verification_request=request,
        runtime_result=replayed_runtime,
        package_artifact_root=package_root,
        runtime_artifact_root=extracted,
    )
    assert qualified["status"] == "verified_compatibility_only"
    assert qualified["simulator_task_success_proven"] is False

    tampered = tmp_path / "isaac-output-tampered.zip"
    with zipfile.ZipFile(first, "r") as source_archive, zipfile.ZipFile(
        tampered, "w", allowZip64=True
    ) as destination_archive:
        for member in source_archive.infolist():
            payload = source_archive.read(member.filename)
            if member.filename == "artifacts/frames/fixed-1.png":
                payload = b"tampered"
            destination_archive.writestr(member, payload)
    with pytest.raises(
        IsaacVerificationOutputBundleError, match="artifact_digest_mismatch"
    ):
        validate_and_extract_isaac_verification_output_bundle(
            bundle_path=tampered,
            expected_input_receipt=receipt,
            expected_source_commit_sha=request["source_commit_sha"],
            output_root=tmp_path / "tampered-output",
        )


def test_isaac_bootstrap_binds_downloads_preserves_typed_blocker_and_uploads(
    tmp_path, monkeypatch
):
    request, receipt, input_bundle, receipt_bytes = _isaac_bootstrap_fixture(tmp_path)
    uploaded: dict[str, bytes] = {}

    def fake_download(url, *, output_path, expected_sha256, **_kwargs):
        payload = receipt_bytes if "receipt.json" in url else input_bundle.read_bytes()
        observed = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert observed == expected_sha256
        Path(output_path).write_bytes(payload)
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=observed,
            host="objects.example",
        )

    def fake_process(command, root, log_path, timeout_seconds):
        assert command[0] == "/isaac-sim/python.sh"
        assert command[1] == str(root / "bundle/run_isaac_splat_nurec_render.py")
        assert all(not item.startswith("/workspace/") for item in command[1:])
        assert timeout_seconds == 3480
        runtime = build_isaac_runtime_result_v3(
            {
                "schema_version": "isaac_splat_nurec_render_result.v3",
                "status": "blocked",
                "isaac_verification_request_digest": request[
                    "isaac_verification_request_digest"
                ],
                "package_digest": request["package_digest"],
                "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
                "runtime_container_image_digest": request[
                    "runtime_container_image_digest"
                ],
                "runtime_implementation_digest": request[
                    "runtime_implementation_digest"
                ],
                "runtime_identity": {
                    "runtime": "isaac_sim",
                    "renderer": "RayTracedLighting",
                    "python_version": "3.11.0",
                    "headless": True,
                },
                "raw_secret_values_recorded": False,
                "blockers": ["isaacsim_module_unavailable"],
            }
        )
        (root / "out/isaac_runtime_result.json").write_text(
            json.dumps(runtime), encoding="utf-8"
        )
        log_path.write_text("typed blocker preserved\n", encoding="utf-8")
        return 2

    def fake_upload(url, *, input_path, expected_sha256, **_kwargs):
        payload = Path(input_path).read_bytes()
        observed = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert observed == expected_sha256
        uploaded["output"] = payload
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=observed,
            host="objects.example",
        )

    monkeypatch.setattr(isaac_bootstrap, "download_file", fake_download)
    monkeypatch.setattr(isaac_bootstrap, "upload_file", fake_upload)
    result = isaac_bootstrap.run_reconstruction_isaac_bootstrap(
        environment=_isaac_bootstrap_environment(
            request, receipt, receipt_bytes
        ),
        work_root=tmp_path / "bootstrap-worker",
        process_runner=fake_process,
    )
    assert result["status"] == "output_uploaded"
    assert result["isaac_runtime_exit_code"] == 2
    assert result["scientific_qualification_inferred"] is False
    assert result["simulator_task_success_proven"] is False
    assert result["bootstrap_receipt_digest"] == canonical_digest(
        result, digest_field="bootstrap_receipt_digest"
    )
    bootstrap_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_isaac_bootstrap.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(bootstrap_schema).validate(result)
    encoded = json.dumps(result)
    assert "input-secret" not in encoded
    assert "receipt-secret" not in encoded
    assert "output-secret" not in encoded

    retrieved = tmp_path / "retrieved-isaac-output.zip"
    retrieved.write_bytes(uploaded["output"])
    validated, runtime, _root = validate_and_extract_isaac_verification_output_bundle(
        bundle_path=retrieved,
        expected_input_receipt=receipt,
        expected_source_commit_sha=request["source_commit_sha"],
        output_root=tmp_path / "validated-bootstrap-output",
    )
    assert validated["status"] == "validated"
    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["isaacsim_module_unavailable"]


def test_isaac_bootstrap_fails_closed_before_runtime_on_binding_ttl_or_missing_result(
    tmp_path, monkeypatch
):
    request, receipt, input_bundle, receipt_bytes = _isaac_bootstrap_fixture(tmp_path)
    downloads: list[str] = []

    def fake_download(url, *, output_path, expected_sha256, **_kwargs):
        downloads.append(url.split("?", maxsplit=1)[0])
        payload = receipt_bytes if "receipt.json" in url else input_bundle.read_bytes()
        Path(output_path).write_bytes(payload)
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=expected_sha256,
            host="objects.example",
        )

    monkeypatch.setattr(isaac_bootstrap, "download_file", fake_download)
    with pytest.raises(
        isaac_bootstrap.ReconstructionIsaacBootstrapError,
        match="receipt_binding_mismatch",
    ):
        isaac_bootstrap.run_reconstruction_isaac_bootstrap(
            environment=_isaac_bootstrap_environment(
                request,
                receipt,
                receipt_bytes,
                BLUEPRINT_ISAAC_VERIFICATION_REQUEST_DIGEST=D[8],
            ),
            work_root=tmp_path / "binding-worker",
            process_runner=lambda *_args: pytest.fail("runtime must not start"),
        )
    assert downloads == ["https://objects.example/receipt.json"]

    downloads.clear()
    with pytest.raises(
        isaac_bootstrap.ReconstructionIsaacBootstrapError,
        match="ttl_invalid",
    ):
        isaac_bootstrap.run_reconstruction_isaac_bootstrap(
            environment=_isaac_bootstrap_environment(
                request,
                receipt,
                receipt_bytes,
                BLUEPRINT_RECONSTRUCTION_HARD_TTL_SECONDS="60",
            ),
            work_root=tmp_path / "ttl-worker",
            process_runner=lambda *_args: pytest.fail("runtime must not start"),
        )
    assert downloads == []

    with pytest.raises(
        isaac_bootstrap.ReconstructionIsaacBootstrapError,
        match="runtime_result_missing",
    ):
        isaac_bootstrap.run_reconstruction_isaac_bootstrap(
            environment=_isaac_bootstrap_environment(
                request, receipt, receipt_bytes
            ),
            work_root=tmp_path / "missing-result-worker",
            process_runner=lambda *_args: 0,
        )


def test_isaac_bootstrap_uploads_typed_blocker_after_abnormal_runtime_exit(
    tmp_path, monkeypatch
):
    request, receipt, input_bundle, receipt_bytes = _isaac_bootstrap_fixture(tmp_path)
    uploaded = {}

    def fake_download(url, *, output_path, expected_sha256, **_kwargs):
        payload = receipt_bytes if "receipt.json" in url else input_bundle.read_bytes()
        Path(output_path).write_bytes(payload)
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=expected_sha256,
            host="objects.example",
        )

    def abnormal_process(_command, root, log_path, _timeout_seconds):
        partial = {
            "schema_version": receipt["expected_runtime_schema"],
            "status": "running",
            "phase": "runner_stage_opened",
            "isaac_verification_request_digest": receipt[
                "isaac_verification_request_digest"
            ],
            "package_digest": receipt["package_digest"],
            "fixed_camera_spec_digest": receipt["fixed_camera_spec_digest"],
            "runtime_container_image_digest": receipt[
                "runtime_container_image_digest"
            ],
            "runtime_implementation_digest": receipt[
                "runtime_implementation_digest"
            ],
            "runtime_identity": {
                "runtime": "isaac_sim",
                "renderer": "RayTracedLighting",
                "python_version": "3.11.0",
                "headless": True,
            },
            "raw_secret_values_recorded": False,
        }
        (root / "out/isaac_runtime_result.json").write_text(
            json.dumps(partial), encoding="utf-8"
        )
        log_path.write_text("abnormal runtime marker\n", encoding="utf-8")
        return 1

    def fake_upload(url, *, input_path, expected_sha256, **_kwargs):
        del url
        payload = Path(input_path).read_bytes()
        observed = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert observed == expected_sha256
        uploaded["output"] = payload
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=observed,
            host="objects.example",
        )

    monkeypatch.setattr(isaac_bootstrap, "download_file", fake_download)
    monkeypatch.setattr(isaac_bootstrap, "upload_file", fake_upload)
    bootstrap = isaac_bootstrap.run_reconstruction_isaac_bootstrap(
        environment=_isaac_bootstrap_environment(request, receipt, receipt_bytes),
        work_root=tmp_path / "abnormal-worker",
        process_runner=abnormal_process,
    )

    assert bootstrap["status"] == "output_uploaded"
    retrieved = tmp_path / "abnormal-output.zip"
    retrieved.write_bytes(uploaded["output"])
    _validated, runtime, _root = validate_and_extract_isaac_verification_output_bundle(
        bundle_path=retrieved,
        expected_input_receipt=receipt,
        expected_source_commit_sha=request["source_commit_sha"],
        output_root=tmp_path / "validated-abnormal-output",
    )
    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["isaac_runtime_process_exit_status_mismatch"]
    diagnostic = runtime["runtime_process_diagnostic"]
    assert diagnostic["exit_code"] == 1
    assert diagnostic["partial_phase"] == "runner_stage_opened"
    assert diagnostic["log_tail"] == "abnormal runtime marker\n"
    assert diagnostic["transfer_urls_removed_from_runner_environment"] is True


def test_isaac_default_process_runner_caps_live_log_and_strips_transfer_urls(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(isaac_bootstrap, "MAX_LOG_BYTES", 512)
    for name in (
        "BLUEPRINT_ISAAC_INPUT_BUNDLE_GET_URL",
        "BLUEPRINT_ISAAC_INPUT_RECEIPT_GET_URL",
        "BLUEPRINT_ISAAC_OUTPUT_BUNDLE_PUT_URL",
    ):
        monkeypatch.setenv(name, "https://objects.example/secret")
    clean_log = tmp_path / "clean.log"
    exit_code = isaac_bootstrap._default_process_runner(
        [
            sys.executable,
            "-c",
            (
                "import os; print(any(os.environ.get(k) for k in "
                "['BLUEPRINT_ISAAC_INPUT_BUNDLE_GET_URL',"
                "'BLUEPRINT_ISAAC_INPUT_RECEIPT_GET_URL',"
                "'BLUEPRINT_ISAAC_OUTPUT_BUNDLE_PUT_URL']))"
            ),
        ],
        tmp_path,
        clean_log,
        10,
    )
    assert exit_code == 0
    assert clean_log.read_text(encoding="utf-8").strip() == "False"

    capped_log = tmp_path / "capped.log"
    with pytest.raises(
        isaac_bootstrap.ReconstructionIsaacBootstrapError,
        match="log_oversized",
    ):
        isaac_bootstrap._default_process_runner(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            tmp_path,
            capped_log,
            10,
        )
    assert capped_log.stat().st_size == 512


def _isaac_bound_paid_request(request, receipt):
    value = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "isaac_canary",
        "capture_profile": "iphone_arkit_lidar",
        "source_commit_sha": request["source_commit_sha"],
        "worker_image_digest": request["runtime_container_image_digest"],
        "worker_stack_manifest_digest": D[0],
        "reconstruction_dataset_digest": D[1],
        "frozen_split_digest": D[2],
        "calibration_digest": D[3],
        "deterministic_configuration_digest": D[4],
        "operation_request_digest": request["isaac_verification_request_digest"],
        "operation_input_bundle_digest": receipt["bundle_digest"],
        "expected_runtime_result_schema": "isaac_splat_nurec_render_result.v3",
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": 18.0,
        "hard_ttl_seconds": 3600,
        "retry_cap": 1,
        "authority_id": "user-authorized-18usd-60min-1retry-two-gpu-ceiling",
        "proof_effect": "none",
        "request_digest": D[5],
        "bound_provider": "vast",
        "bound_preflight_digest": D[6],
        "bound_checkout_source_commit": request["source_commit_sha"],
        "bound_checkout_clean": True,
        "isaac_image_release_digest": D[7],
        "provider_mutation_authorized": True,
    }
    value["bound_request_digest"] = canonical_digest(
        value, digest_field="bound_request_digest"
    )
    return value


class _IsaacVastProvider:
    name = "vast"

    def __init__(self, *, external_live=False):
        self.external_live = external_live
        self.launched = False
        self.requests = []

    def billable_inventory(self, *, name_prefix):
        count = int(self.launched or (self.external_live and name_prefix == ""))
        return {"api_confirmed": True, "live_resource_count": count, "resources": []}

    def build_request(self, spec, job_dir):
        assert (Path(job_dir) / "prelaunch_staging_receipt.json").is_file()
        assert spec.image == IMAGE
        assert spec.env["BLUEPRINT_RECONSTRUCTION_HARD_TTL_SECONDS"] == "3600"
        assert spec.env["ACCEPT_EULA"] == "Y"
        assert spec.env["PRIVACY_CONSENT"] == "Y"
        assert spec.env["CUDA_VISIBLE_DEVICES"] == "0"
        bootstrap = " ".join(spec.bootstrap_argv)
        assert (
            "exec /isaac-sim/python.sh -m "
            "blueprint_pipeline.reconstruction_isaac_bootstrap"
        ) in bootstrap
        assert "exec python3 -m blueprint_pipeline.reconstruction_isaac_bootstrap" in bootstrap
        assert "BLUEPRINT_RECONSTRUCTION_ISAAC_BLOCKED:python_runtime_missing" in bootstrap
        return {"create_payload": {"env": dict(spec.env)}}

    def launch(self, job_dir, request, **_kwargs):
        del job_dir
        self.requests.append(request)
        self.launched = True
        return {"status": "launched", "instance_id": "isaac-42"}

    def terminate(self, instance_id):
        assert instance_id == "isaac-42"
        self.launched = False
        return {"status": "stopped", "instance_id": instance_id}


def test_isaac_vast_lifecycle_stages_before_launch_abstains_and_proves_zero(
    tmp_path, monkeypatch
):
    request, receipt, input_bundle, receipt_bytes = _isaac_bootstrap_fixture(tmp_path)
    runtime_root = tmp_path / "provider-runtime"
    runtime_root.mkdir()
    runtime = build_isaac_runtime_result_v3(
        {
            "schema_version": "isaac_splat_nurec_render_result.v3",
            "status": "blocked",
            "isaac_verification_request_digest": request[
                "isaac_verification_request_digest"
            ],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": request[
                "runtime_container_image_digest"
            ],
            "runtime_implementation_digest": request[
                "runtime_implementation_digest"
            ],
            "runtime_identity": {
                "runtime": "isaac_sim",
                "renderer": "RayTracedLighting",
                "python_version": "3.11.0",
                "headless": True,
            },
            "raw_secret_values_recorded": False,
            "blockers": ["isaacsim_module_unavailable"],
        }
    )
    (runtime_root / "isaac_runtime_result.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    provider_output = tmp_path / "provider-output.zip"
    compile_isaac_verification_output_bundle(
        bundle_receipt=receipt,
        runtime_output_root=runtime_root,
        output_path=provider_output,
    )

    def exact_fetch(url, destination, expected_digest, _maximum_bytes):
        payload = receipt_bytes if "receipt.json" in url else input_bundle.read_bytes()
        destination.write_bytes(payload)
        observed = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert observed == expected_digest
        return SafeHttpFileTransfer(200, len(payload), observed, "objects.example")

    def output_fetch(_url, destination):
        payload = provider_output.read_bytes()
        destination.write_bytes(payload)
        return SafeHttpFileTransfer(
            200,
            len(payload),
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            "objects.example",
        )

    provider = _IsaacVastProvider()
    grant = require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_reconstruction_isaac_vast_operation(
        bound_request=_isaac_bound_paid_request(request, receipt),
        bundle_receipt=receipt,
        preflight={
            "watchdog": {
                "status": "armed",
                "independent_process": True,
                "pid": 123,
                "deadline_epoch": 10_000,
                "name_prefix": "blueprint-reconstruction-",
            },
            "gpu_memory_bytes": 48 * 1024**3,
            "container_disk_bytes": 120 * 1024**3,
            "on_demand_price_usd_per_hour": 0.58,
        },
        job_dir=tmp_path / "isaac-vast",
        input_bundle_get_url="https://objects.example/input.zip?sig=secret",
        input_receipt_get_url="https://objects.example/receipt.json?sig=secret",
        output_bundle_put_url="https://objects.example/output-put.zip?sig=secret",
        output_bundle_get_url="https://objects.example/output-get.zip?sig=secret",
        provider=provider,
        paid_resource_admission_grant=grant,
        exact_fetcher=exact_fetch,
        output_fetcher=output_fetch,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "completed"
    assert result["gpu_count"] == 1
    assert result["runtime_status"] == "blocked"
    assert result["independent_qualification_status"] == "abstained"
    assert result["provider_zero_verified"] is True
    assert result["output_retrieved_before_teardown"] is True
    assert result["simulator_task_success_proven"] is False
    assert result["physical_success_proven"] is False
    assert provider.launched is False
    assert provider.requests[0]["prelaunch_spend_guard"]["gpu_count"] == 1
    assert provider.requests[0]["prelaunch_spend_guard"]["max_spend_usd"] == 18.0
    assert "secret" not in json.dumps(result)
    qualification = json.loads(
        (tmp_path / "isaac-vast/independent_isaac_qualification.json").read_text()
    )
    assert qualification["blockers"] == ["isaac_runtime_not_completed"]
    replay = replay_reconstruction_isaac_vast_operation(
        job_dir=tmp_path / "isaac-vast",
        bound_request=_isaac_bound_paid_request(request, receipt),
        bundle_receipt=receipt,
    )
    assert replay["status"] == "replay_verified"
    assert replay["live_provider_accessed"] is False
    assert replay["live_agent_accessed"] is False
    assert replay["independent_qualification_status"] == "abstained"
    lifecycle_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_isaac_vast_operation.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    for artifact in (
        result,
        replay,
        qualification,
        json.loads((tmp_path / "isaac-vast/teardown_receipt.json").read_text()),
        json.loads(
            (tmp_path / "isaac-vast/provider_zero_verification.json").read_text()
        ),
    ):
        jsonschema.Draft202012Validator(lifecycle_schema).validate(artifact)


def test_isaac_vast_second_gpu_authority_does_not_override_global_zero(
    tmp_path,
):
    request, receipt, input_bundle, receipt_bytes = _isaac_bootstrap_fixture(tmp_path)

    def exact_fetch(url, destination, expected_digest, _maximum_bytes):
        payload = receipt_bytes if "receipt.json" in url else input_bundle.read_bytes()
        destination.write_bytes(payload)
        return SafeHttpFileTransfer(200, len(payload), expected_digest, "objects.example")

    provider = _IsaacVastProvider(external_live=True)
    grant = require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    with pytest.raises(
        ReconstructionIsaacVastError, match="provider_not_zero_before_launch"
    ):
        run_reconstruction_isaac_vast_operation(
            bound_request=_isaac_bound_paid_request(request, receipt),
            bundle_receipt=receipt,
            preflight={"watchdog": {}},
            job_dir=tmp_path / "blocked-isaac-vast",
            input_bundle_get_url="https://objects.example/input.zip",
            input_receipt_get_url="https://objects.example/receipt.json",
            output_bundle_put_url="https://objects.example/output-put.zip",
            output_bundle_get_url="https://objects.example/output-get.zip",
            provider=provider,
            paid_resource_admission_grant=grant,
            exact_fetcher=exact_fetch,
            watchdog_validator=lambda _watchdog, _now, _ttl: True,
            clock=lambda: 1000.0,
        )
    assert provider.launched is False


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
    assert validate_isaac_verification_worker_bundle_receipt(receipt) == receipt
    extraction = extract_isaac_verification_worker_bundle(
        bundle_path=bundle_path,
        bundle_receipt=receipt,
        output_root=tmp_path / "extracted",
    )
    assert (
        extract_isaac_verification_worker_bundle(
            bundle_path=bundle_path,
            bundle_receipt=receipt,
            output_root=tmp_path / "extracted",
        )
        == extraction
    )
    extraction_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/isaac_verification_worker_bundle_extraction.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(extraction_schema).validate(extraction)
    extracted_package = (
        tmp_path
        / "extracted"
        / receipt["bundle_digest"][7:]
        / "reconstruction.usdz"
    )
    extracted_package.write_bytes(b"tampered")
    with pytest.raises(IsaacWorkerBundleError, match="extraction_replay_tampered"):
        extract_isaac_verification_worker_bundle(
            bundle_path=bundle_path,
            bundle_receipt=receipt,
            output_root=tmp_path / "extracted",
        )

    compressed = tmp_path / "compressed-isaac-bundle.zip"
    with zipfile.ZipFile(bundle_path, "r") as source_archive, zipfile.ZipFile(
        compressed, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination_archive:
        for member in source_archive.infolist():
            destination_archive.writestr(
                member.filename, source_archive.read(member.filename)
            )
    drifted = dict(receipt)
    drifted["bundle_digest"] = _sha256(compressed)
    drifted["receipt_digest"] = canonical_digest(
        drifted, digest_field="receipt_digest"
    )
    with pytest.raises(IsaacWorkerBundleError, match="archive_member_unsafe"):
        extract_isaac_verification_worker_bundle(
            bundle_path=compressed,
            bundle_receipt=drifted,
            output_root=tmp_path / "compressed-extracted",
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
