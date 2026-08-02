from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from PIL import Image

from blueprint_pipeline import external_scene_isaac_verification as verification

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_scene_isaac_verification import (
    ExternalSceneIsaacVerificationError,
    build_external_scene_isaac_verification_request,
    normalize_external_scene_isaac_verification,
)
from blueprint_pipeline.reconstruction_isaac_worker_bundle import (
    compile_isaac_verification_worker_bundle,
    extract_isaac_verification_worker_bundle,
    validate_isaac_verification_worker_bundle_receipt,
)


DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/isaac@sha256:" + "b" * 64
SHA = "c" * 40


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization() -> dict:
    value = {
        "schema_version": "blueprint_remote_processing_authorization.v1",
        "authorization_id": "private-site-vast-evaluation",
        "provider_scope": ["vast"],
        "purpose_scope": [
            "isaac_sim_scene_ingest",
            "collision_candidate_compilation",
            "scene_task_target_analysis",
            "franka_articulated_policy_evaluation",
        ],
        "asset_digests": [DIGEST],
        "remote_upload_authorized": True,
        "paid_compute_authorized": True,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "commercial_benchmarking_authorized": False,
        "retention_policy": "bounded_to_evaluation_then_provider_zero",
    }
    value["authorization_digest"] = canonical_digest(value, digest_field="authorization_digest")
    return value


def _request(*, package: Path, cameras: Path, runner: Path, options: Path) -> dict:
    authorization = _authorization()
    return build_external_scene_isaac_verification_request(
        {
            "schema_version": "external_scene_isaac_verification_request.v1",
            "source_commit_sha": SHA,
            "robot_id": "franka_panda",
            "package_digest": _sha(package),
            "package_result_digest": DIGEST,
            "package_artifact_reference": package.name,
            "appearance_scene_digest": DIGEST,
            "collision_candidate_digest": DIGEST,
            "scene_frame_binding_digest": DIGEST,
            "target_analysis_digest": DIGEST,
            "target_binding_digest": DIGEST,
            "placement_proposal_digest": DIGEST,
            "render_options_digest": _sha(options),
            "fixed_camera_spec_digest": _sha(cameras),
            "fixed_camera_ids": ["task_focus"],
            "runtime_implementation_digest": _sha(runner),
            "runtime_container_image_digest": IMAGE,
            "expected_prim_paths": {
                "appearance": "/World/BlueprintReconstruction/Appearance",
                "collision": "/World/BlueprintReconstruction/Collision",
            },
            "physics_probe_request": {
                "ground_collider_prim": "/World/BlueprintReconstruction/Collision/ExternalSceneMesh",
                "ground_height_m": 0.0,
                "probe_xy_m": [0.0, 0.0],
                "selection_status": "derived_geometry_candidate_unverified_in_isaac",
                "manufacture_ground_plane": False,
                "require_contact_event": True,
                "steps": 240,
                "test_body": {
                    "shape": "cube",
                    "size_m": 0.1,
                    "mass_kg": 1.0,
                    "spawn_height_above_ground_m": 0.5,
                },
                "gravity_m_s2": -9.81,
                "physics_dt_seconds": 1.0 / 60.0,
            },
            "remote_processing_authorization": authorization,
            "remote_processing_authorization_digest": authorization["authorization_digest"],
            "timeout_seconds": 900,
            "spend_controls": {
                "authorized": False,
                "estimated_max_spend_usd": 2.0,
                "hard_ttl_seconds": 900,
                "teardown_required": True,
                "provider_zero_required_before_and_after": True,
            },
            "external_derived_support_asset": True,
            "source_relationship_to_blueprint_raw_capture": "none",
            "blueprint_raw_capture_truth": False,
            "source_video_available": False,
            "source_video_required_for_candidate_execution": False,
            "independent_metric_scale_proven": False,
            "provider_authored_package": False,
            "blueprint_compiled_package": True,
            "exact_package_required": True,
            "headless": True,
            "display_attached": False,
            "execution_status": "awaiting_canonical_paid_runtime_authorization",
            "provider_allocation_performed": False,
            "expected_runtime_schema": "isaac_splat_nurec_render_result.v3",
            "proof_effect": "none",
            "claim_ceiling": "request_only",
        }
    )


def test_external_scene_request_and_worker_bundle_preserve_authorization_boundary(
    tmp_path: Path,
) -> None:
    package = tmp_path / "scene.usdz"
    package.write_bytes(b"test package")
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(
            [
                {
                    "id": "task_focus",
                    "spec": {
                        "pos": [1, 0, 1],
                        "target": [0, 0, 0],
                        "fov": 52,
                        "up": [0, 0, 1],
                    },
                }
            ]
        )
    )
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n")
    options = tmp_path / "render_options.json"
    options.write_text(
        json.dumps(
            {
                "robot_id": "franka_panda",
                "robot_usd": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                "robot_prim_path": "/World/Franka",
                "robot_pose": [0, 0, 0, 0],
                "robot_ground_z": 0,
                "robot_placement_digest": DIGEST,
                "placement_proposal_digest": DIGEST,
                "robot_only_pass": True,
            }
        )
    )
    request = _request(package=package, cameras=cameras, runner=runner, options=options)
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/external_scene_isaac_verification_request.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(request, schema)
    receipt = compile_isaac_verification_worker_bundle(
        verification_request=request,
        package_artifact_root=tmp_path,
        fixed_camera_spec_path=cameras,
        runner_path=runner,
        render_options_path=options,
        output_root=tmp_path / "bundles",
    )
    assert receipt["verification_request_member"] == (
        "external_scene_isaac_verification_request.v1.json"
    )
    assert validate_isaac_verification_worker_bundle_receipt(receipt) == receipt
    extraction = extract_isaac_verification_worker_bundle(
        bundle_path=(
            tmp_path
            / "bundles"
            / request["isaac_verification_request_digest"][7:]
            / "isaac_verification_worker_bundle.zip"
        ),
        bundle_receipt=receipt,
        output_root=tmp_path / "extracted",
    )
    assert any(
        row["archive_path"] == "external_scene_isaac_verification_request.v1.json"
        for row in extraction["extracted_members"]
    )


def test_external_scene_request_rejects_missing_video_as_blocker_or_fabricated_scale(
    tmp_path: Path,
) -> None:
    package = tmp_path / "scene.usdz"
    package.write_bytes(b"x")
    cameras = tmp_path / "cameras.json"
    cameras.write_text("[]")
    runner = tmp_path / "runner.py"
    runner.write_text("pass\n")
    options = tmp_path / "render_options.json"
    options.write_text("{}")
    request = _request(package=package, cameras=cameras, runner=runner, options=options)
    request["source_video_required_for_candidate_execution"] = True
    request["independent_metric_scale_proven"] = True
    request.pop("isaac_verification_request_digest")
    with pytest.raises(ExternalSceneIsaacVerificationError) as exc:
        build_external_scene_isaac_verification_request(request)
    assert (
        "external_scene_isaac_request_boundary_invalid:independent_metric_scale_proven"
        in exc.value.codes
    )
    assert (
        "external_scene_isaac_request_boundary_invalid:source_video_required_for_candidate_execution"
        in exc.value.codes
    )


def test_policy_only_abstention_preserves_static_scene_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "package"
    runtime_root = tmp_path / "runtime"
    package_root.mkdir()
    (runtime_root / "frames").mkdir(parents=True)
    package = package_root / "scene.usdz"
    package.write_bytes(b"exact external scene package")
    frame = runtime_root / "frames/task_focus.png"
    image = Image.new("RGB", (8, 8))
    image.putdata(
        [(index * 3 % 255, index * 7 % 255, index * 13 % 255) for index in range(64)]
    )
    image.save(frame)
    width, height, mean, std = verification._render_measurements(frame)
    request = {
        "isaac_verification_request_digest": DIGEST,
        "package_digest": _sha(package),
        "package_result_digest": DIGEST,
        "package_artifact_reference": "scene.usdz",
        "fixed_camera_spec_digest": DIGEST,
        "runtime_container_image_digest": IMAGE,
        "runtime_implementation_digest": DIGEST,
        "expected_prim_paths": {
            "appearance": "/World/BlueprintReconstruction/Appearance",
            "collision": "/World/BlueprintReconstruction/Collision",
        },
        "physics_probe_request": {
            "steps": 240,
            "test_body": {"shape": "cube"},
            "gravity_m_s2": -9.81,
            "physics_dt_seconds": 1.0 / 60.0,
        },
        "robot_id": "franka_panda",
        "fixed_camera_ids": ["task_focus"],
        "remote_processing_authorization_digest": DIGEST,
        "target_analysis_digest": DIGEST,
        "target_binding_digest": DIGEST,
        "placement_proposal_digest": DIGEST,
    }
    runtime = {
        "status": "blocked",
        "blockers": ["isaac_articulated_policy_trace_pair_incomplete"],
        "isaac_verification_request_digest": DIGEST,
        "package_digest": request["package_digest"],
        "fixed_camera_spec_digest": DIGEST,
        "runtime_container_image_digest": IMAGE,
        "runtime_implementation_digest": DIGEST,
        "isaac_runtime_result_digest": DIGEST,
        "stage": {
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "transforms_valid": True,
            "dependency_inspection_available": True,
            "missing_asset_count": 0,
            "obvious_scale_mismatch_detected": False,
            "particlefield_prim_count": 1,
            "active_collision_prim_count": 1,
            "expected_prim_paths": request["expected_prim_paths"],
        },
        "physics_probe": {
            "ground_contact_surface_present": True,
            "live_rigid_body_pose_observed": True,
            "test_body_fell_through_floor": False,
            "contact_event_count": 2,
            "steps_executed": 240,
            "probe_configuration": {
                "test_body": {"shape": "cube"},
                "gravity_m_s2": -9.81,
                "physics_dt_seconds": 1.0 / 60.0,
            },
        },
        "robot": {
            "robot_id": "franka_panda",
            "composited": True,
            "geometry_streamed": True,
            "resolved_usd": (
                "https://assets.example/Isaac/Robots/FrankaRobotics/"
                "FrankaPanda/franka.usd"
            ),
        },
        "cameras": [
            {
                "id": "task_focus",
                "artifact_reference": "frames/task_focus.png",
                "digest": _sha(frame),
                "width": width,
                "height": height,
                "pixel_mean": mean,
                "pixel_std": std,
            }
        ],
        "articulated_policy_trace_pair": {"status": "blocked"},
    }
    monkeypatch.setattr(
        verification, "build_external_scene_isaac_verification_request", lambda _value: request
    )
    monkeypatch.setattr(verification, "build_isaac_runtime_result_v3", lambda _value: runtime)

    result = normalize_external_scene_isaac_verification(
        verification_request=request,
        runtime_result=runtime,
        package_artifact_root=package_root,
        runtime_artifact_root=runtime_root,
    )

    assert result["status"] == "verified_derived_scene_compatibility_only"
    assert result["policy_lane_abstained_without_invalidating_static_evidence"] is True
    assert result["articulated_policy_trace_pair_qualified"] is False
    assert result["checks"]["live_contact_observed"] is True
