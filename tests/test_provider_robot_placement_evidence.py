from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from blueprint_pipeline.external_provider_nurec import (
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_runtime_result,
)
from blueprint_pipeline.provider_robot_placement_evidence import (
    build_provider_robot_placement_evidence,
)


D = ["sha256:" + character * 64 for character in "123456789abcdef"]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _request() -> dict:
    return build_provider_nurec_isaac_request(
        {
            "stable_run_identity": "robot-placement-evidence-test",
            "source_commit_sha": "a" * 40,
            "package_digest": D[0],
            "package_artifact_reference": "public/source.usdz",
            "external_import_receipt_digest": D[1],
            "qualification_report_digest": D[2],
            "fixed_camera_spec_digest": D[3],
            "fixed_camera_ids": ["near", "wide"],
            "runtime_implementation_digest": D[4],
            "runtime_container_image_digest": "registry.test/isaac@" + D[5],
            "render_options_digest": D[6],
            "expected_prim_paths": {
                "appearance": "/World/gauss/gauss",
                "collision": "/World/gauss/mesh",
            },
            "physics_probe_request": {
                "ground_collider_prim": "/World/gauss/mesh",
                "ground_height_m": 0.0,
                "probe_xy_m": [1.0, 2.0],
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


def _runtime(request: dict, root: Path) -> dict:
    robot_rows = []
    for index, camera_id in enumerate(request["fixed_camera_ids"]):
        rgb = root / f"frames_robot_only/{camera_id}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(b"PNG" + camera_id.encode())
        depth = root / f"frames_robot_only/{camera_id}_distance.npy"
        array = np.full((64, 64), np.inf, dtype=np.float32)
        array[4:12, 8 + index : 16 + index] = 2.0 + index
        np.save(depth, array, allow_pickle=False)
        robot_rows.append(
            {
                "id": camera_id,
                "rgb_artifact_reference": f"frames_robot_only/{camera_id}.png",
                "rgb_digest": _sha256(rgb),
                "distance_artifact_reference": (f"frames_robot_only/{camera_id}_distance.npy"),
                "distance_digest": _sha256(depth),
            }
        )
    cameras = [
        {
            "id": camera_id,
            "artifact_reference": f"frames/{camera_id}.png",
            "digest": D[7 + index],
            "pixel_std": 10.0,
            "nonblank": True,
        }
        for index, camera_id in enumerate(request["fixed_camera_ids"])
    ]
    return build_provider_nurec_isaac_runtime_result(
        {
            "schema_version": "provider_nurec_isaac_runtime_result.v1",
            "status": "completed",
            "isaac_verification_request_digest": request["isaac_verification_request_digest"],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "raw_secret_values_recorded": False,
            "stage": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "transforms_valid": True,
                "dependency_inspection_available": True,
                "missing_asset_count": 0,
                "obvious_scale_mismatch_detected": False,
                "expected_prim_paths": request["expected_prim_paths"],
                "particlefield_prim_count": 1,
                "active_collision_prim_count": 1,
            },
            "physics_probe": {
                "ground_contact_surface_present": True,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 1,
                "steps_executed": 240,
            },
            "cameras": cameras,
            "robot": {
                "requested": True,
                "composited": True,
                "geometry_streamed": True,
                "mesh_point_total": 1024,
                "prim_path": "/World/Franka",
                "robot_usd": "Isaac/Robots/Franka/franka.usd",
                "resolved_usd": "https://assets.example/franka.usd",
                "robot_pose": [1.0, 2.0, 0.0, 0.0],
                "world_bound_min": [0.8, 1.8, 0.0],
                "world_bound_max": [1.2, 2.2, 1.0],
                "robot_only_environment_hidden": True,
                "robot_only_hidden_environment_prim_paths": ["/World/gauss/gauss"],
                "robot_only_pass": robot_rows,
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


def test_depth_foreground_qualifies_small_robot_on_uniform_background(
    tmp_path: Path,
) -> None:
    request = _request()
    runtime = _runtime(request, tmp_path)

    evidence = build_provider_robot_placement_evidence(
        verification_request=request,
        runtime_result=runtime,
        runtime_artifact_root=tmp_path,
    )

    assert evidence["status"] == "verified_visual_placement_only"
    assert evidence["visual_robot_placement_observed"] is True
    assert [row["depth_foreground_pixel_count"] for row in evidence["camera_evidence"]] == [
        64,
        64,
    ]
    assert evidence["collision_free_placement_proven"] is False
    assert evidence["navigation_or_task_success_proven"] is False


def test_depth_artifact_tamper_blocks_visual_placement(tmp_path: Path) -> None:
    request = _request()
    runtime = _runtime(request, tmp_path)
    (tmp_path / "frames_robot_only/near_distance.npy").write_bytes(b"tampered")

    evidence = build_provider_robot_placement_evidence(
        verification_request=request,
        runtime_result=runtime,
        runtime_artifact_root=tmp_path,
    )

    assert evidence["status"] == "blocked"
    assert "provider_robot_placement_artifact_digest_mismatch" in evidence["blockers"]
