from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.robot_eval_execution import build_robot_pov_observation_bundle
from blueprint_pipeline.robot_initial_observation import (
    build_initial_observation_source_resolution,
    build_owner_robot_camera_calibration_request,
    build_robot_camera_profile_registry,
    build_robot_camera_profile_launch_readiness,
)


GENERATED_AT = "2026-06-25T00:00:00+00:00"
IDENTITY_POSE = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_png(path: Path, *, size: tuple[int, int] = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (90, 130, 180))
    image.save(path)
    return path


def _write_depth(path: Path, *, size: tuple[int, int] = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full(size, 2.0, dtype=np.float32))
    return path


def _write_object_grounding(capture_root: Path, *, object_id: str = "bin_0001") -> None:
    _write_json(
        capture_root / "raw" / "object_grounding_hints.json",
        {
            "backend_status": "ok",
            "grounded_objects": [
                {
                    "object_id": object_id,
                    "label": "bin",
                    "boundingBox": {
                        "center": [0.0, 0.0, 0.5],
                        "extents": [0.4, 0.3, 0.2],
                    },
                    "provenance": {
                        "grounding_level": "observed",
                        "canonical_truth": True,
                    },
                }
            ],
        },
    )


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (root / "pipeline" / "robot_eval_dataset").mkdir(parents=True, exist_ok=True)
    return root


def _scenario_cards() -> dict[str, Any]:
    return {
        "cards": [
            {
                "task_id": "inspect_bin",
                "scenario_id": "scenario-inspect-bin",
                "robot_profile_id": "unitree_g1",
            }
        ]
    }


def _owner_robot_profile(index: int) -> dict[str, Any]:
    profile_id = f"owner_robot_{index:02d}"
    return {
        "robot_profile_id": profile_id,
        "display_name": f"Owner robot {index:02d}",
        "embodiment_type": "mobile_manipulator",
        "source": "owner_provided_robot_team_camera_calibration",
        "primary_camera_id": "front_rgbd",
        "cameras": [
            {
                "camera_id": "front_rgbd",
                "display_name": "Front RGB-D",
                "modalities": ["rgb", "depth"],
                "mount": "front_mast",
                "frame_id": f"{profile_id}_front_rgbd",
                "horizontal_fov_degrees": 72.0 + index * 0.1,
                "vertical_fov_degrees": 47.5 + index * 0.1,
                "intrinsics": {
                    "width": 1280,
                    "height": 720,
                    "fx": 930.0 + index,
                    "fy": 928.0 + index,
                    "cx": 640.0,
                    "cy": 360.0,
                    "camera_model": "pinhole",
                    "source": "owner_provided_calibration_file",
                },
                "extrinsics": {
                    "reference_frame": "robot_base",
                    "child_frame": f"{profile_id}_front_rgbd",
                    "xyz_m": [0.28 + index * 0.01, 0.0, 1.12],
                    "rpy_rad": [0.0, -0.08, 0.0],
                    "source": "owner_provided_calibration_file",
                },
            }
        ],
    }


def _unitree_g1_owner_camera(
    *,
    camera_id: str,
    mount: str,
    frame_id: str,
    xyz_m: list[float],
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "display_name": camera_id.replace("_", " ").title(),
        "modalities": ["rgb", "depth"],
        "mount": mount,
        "frame_id": frame_id,
        "horizontal_fov_degrees": horizontal_fov_degrees,
        "vertical_fov_degrees": vertical_fov_degrees,
        "intrinsics": {
            "width": 1280,
            "height": 720,
            "fx": 920.0,
            "fy": 918.0,
            "cx": 640.0,
            "cy": 360.0,
            "camera_model": "pinhole",
            "source": "owner_provided_calibration_file",
        },
        "extrinsics": {
            "reference_frame": "robot_base",
            "child_frame": frame_id,
            "xyz_m": xyz_m,
            "rpy_rad": [0.0, -0.08, 0.0],
            "source": "owner_provided_calibration_file",
        },
    }


def _unitree_g1_owner_profile(*, cameras: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "robot_profile_id": "unitree_g1",
        "display_name": "Owner Unitree G1",
        "embodiment_type": "humanoid",
        "source": "owner_provided_robot_team_camera_calibration",
        "primary_camera_id": "head_rgbd",
        "cameras": cameras
        if cameras is not None
        else [
            _unitree_g1_owner_camera(
                camera_id="head_rgbd",
                mount="head",
                frame_id="unitree_g1_head_camera",
                xyz_m=[0.18, 0.0, 1.42],
                horizontal_fov_degrees=75.0,
                vertical_fov_degrees=46.7,
            ),
            _unitree_g1_owner_camera(
                camera_id="chest_rgbd",
                mount="torso",
                frame_id="unitree_g1_chest_camera",
                xyz_m=[0.12, 0.0, 1.05],
                horizontal_fov_degrees=82.0,
                vertical_fov_degrees=52.1,
            ),
        ],
    }


def _task_cards() -> dict[str, Any]:
    return {
        "cards": [
            {
                "task_id": "inspect_bin",
                "task_statement": "Inspect the returns bin",
            }
        ]
    }


def _scenario_eval_matrix() -> dict[str, Any]:
    return {
        "runs": [
            {
                "scenario_eval_run_id": "run-inspect-bin-0001",
                "task_id": "inspect_bin",
                "scenario_id": "scenario-inspect-bin",
                "variation_name": "base_capture_layout",
                "spawn_pose": [0.0, 0.0, 0.8],
                "concrete_mutation": {"spawn_pose": [0.0, 0.0, 0.8]},
            }
        ]
    }


def test_robot_camera_profile_registry_normalizes_multiple_profiles() -> None:
    registry = build_robot_camera_profile_registry(
        job_request={
            "robot_profile": {
                "robot_profile_id": "unitree_g1",
                "display_name": "G1 customer config",
                "primary_camera_id": "head_rgbd",
                "cameras": [
                    {
                        "camera_id": "head_rgbd",
                        "modalities": ["rgb", "depth"],
                        "horizontal_fov_degrees": 90,
                        "intrinsics": {"width": 1280, "height": 720},
                        "extrinsics": {
                            "reference_frame": "robot_base",
                            "xyz_m": [0.2, 0.0, 1.45],
                            "rpy_rad": [0.0, -0.1, 0.0],
                        },
                    }
                ],
            },
            "robot_profiles": [
                {
                    "robot_profile_id": "inspection_bot_v2",
                    "cameras": [
                        {
                            "camera_id": "front_rgb",
                            "intrinsics": {
                                "width": 640,
                                "height": 480,
                                "fx": 500,
                                "fy": 510,
                                "cx": 320,
                                "cy": 240,
                            },
                            "extrinsics": {"xyz_m": [0.3, 0.0, 0.9]},
                        }
                    ],
                }
            ],
        },
        scenario_cards=_scenario_cards(),
        generated_at=GENERATED_AT,
    )

    assert registry["schema_version"] == "robot_camera_profile_registry.v1"
    assert registry["profile_count"] == 2
    profile_ids = {profile["robot_profile_id"] for profile in registry["profiles"]}
    assert profile_ids == {"unitree_g1", "inspection_bot_v2"}
    g1 = next(profile for profile in registry["profiles"] if profile["robot_profile_id"] == "unitree_g1")
    camera = g1["cameras"][0]
    assert camera["camera_id"] == "head_rgbd"
    assert camera["intrinsics"]["width"] == 1280
    assert camera["intrinsics"]["fx"] > 600
    assert camera["extrinsics"]["xyz_m"] == [0.2, 0.0, 1.45]
    assert camera["horizontal_fov_degrees"] == 90.0


def test_robot_camera_profile_launch_readiness_accepts_default_profiles_for_sim_only() -> None:
    registry = build_robot_camera_profile_registry(
        job_request={},
        scenario_cards={},
        generated_at=GENERATED_AT,
    )
    readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=GENERATED_AT,
        launch_mode=True,
    )

    assert registry["profile_count"] == 1
    assert registry["profiles"][0]["smoke_only"] is True
    assert readiness["status"] == "ready"
    assert readiness["launch_mode"] is True
    assert readiness["launch_scope"] == "sim_only"
    assert readiness["defaults_are_smoke_only"] is True
    assert readiness["default_smoke_only_profile_count"] == 1
    assert readiness["launch_ready_profile_count"] == 1
    assert readiness["sim_only_launch_ready_profile_count"] == 1
    assert readiness["physical_robot_launch_ready_profile_count"] == 0
    assert readiness["ready_for_launch"] is True
    assert readiness["owner_provided_camera_calibration_required_for_launch"] is False
    assert readiness["owner_provided_camera_calibration_required_for_physical_robot_launch"] is True
    assert readiness["owner_calibration_request_packet_path"] is None
    assert readiness["blockers"] == []
    assert "default_robot_camera_profile_smoke_only:unitree_g1" in (
        readiness["physical_robot_calibration_blockers"]
    )
    missing = readiness["physical_robot_calibration_inputs_needed_for_physical_launch"]
    assert {
        (item["robot_profile_id"], item["camera_id"]) for item in missing
    } == {
        ("unitree_g1", "head_rgbd"),
        ("unitree_g1", "chest_rgbd"),
    }
    head_missing = next(item for item in missing if item["camera_id"] == "head_rgbd")
    assert head_missing["missing_launch_fields"] == [
        "owner_provided_intrinsics",
        "owner_provided_extrinsics",
        "owner_provided_fov",
    ]
    assert {
        field["path"] for field in head_missing["required_owner_fields"]
    } >= {
        "cameras[?camera_id=='head_rgbd'].intrinsics.fx",
        "cameras[?camera_id=='head_rgbd'].intrinsics.fy",
        "cameras[?camera_id=='head_rgbd'].extrinsics.reference_frame",
        "cameras[?camera_id=='head_rgbd'].extrinsics.child_frame",
        "cameras[?camera_id=='head_rgbd'].extrinsics.xyz_m",
        "cameras[?camera_id=='head_rgbd'].horizontal_fov_degrees",
        "cameras[?camera_id=='head_rgbd'].vertical_fov_degrees",
    }
    assert head_missing["required_file_names"] == {
        "combined_owner_profile_file": "unitree_g1_owner_robot_camera_profile.json",
        "camera_intrinsics_file": "unitree_g1_head_rgbd_intrinsics.json",
        "camera_fov_file": "unitree_g1_head_rgbd_fov.json",
        "camera_extrinsics_file": "unitree_g1_head_rgbd_extrinsics.json",
    }
    assert "missing_owner_provided_extrinsics:unitree_g1:head_rgbd" in (
        readiness["physical_robot_calibration_blockers"]
    )


def test_unitree_g1_owner_camera_calibration_accepts_head_and_chest_rgbd_profile() -> None:
    registry = build_robot_camera_profile_registry(
        job_request={"robot_profile": _unitree_g1_owner_profile()},
        scenario_cards=_scenario_cards(),
        generated_at=GENERATED_AT,
    )
    readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=GENERATED_AT,
        launch_mode=True,
    )
    request = build_owner_robot_camera_calibration_request(
        registry=registry,
        launch_readiness=readiness,
        generated_at=GENERATED_AT,
    )

    profile = registry["profiles"][0]
    assert profile["robot_profile_id"] == "unitree_g1"
    assert profile["smoke_only"] is False
    assert profile["calibration_contract"]["physical_robot_camera_shape_valid"] is True
    assert profile["calibration_contract"]["required_physical_camera_ids"] == [
        "head_rgbd",
        "chest_rgbd",
    ]
    assert readiness["status"] == "ready"
    assert readiness["ready_for_launch"] is True
    assert readiness["physical_robot_launch_ready_profile_count"] == 1
    assert readiness["physical_robot_calibration_blockers"] == []
    assert request["status"] == "not_required"
    assert request["artifact_is_calibration_proof"] is False
    assert request["physical_robot_claim_upgrade_proven_by_this_artifact"] is False
    for camera in profile["cameras"]:
        contract = camera["calibration_contract"]
        assert camera["owner_input_context"] is True
        assert camera["intrinsics"]["owner_provided"] is True
        assert camera["extrinsics"]["owner_provided"] is True
        assert contract["owner_provided_intrinsics"] is True
        assert contract["owner_provided_extrinsics"] is True
        assert contract["owner_provided_fov"] is True
        assert contract["missing_launch_fields"] == []


def test_unitree_g1_owner_camera_calibration_flags_partial_profile_shape() -> None:
    head_only = _unitree_g1_owner_camera(
        camera_id="head_rgbd",
        mount="head",
        frame_id="unitree_g1_head_camera",
        xyz_m=[0.18, 0.0, 1.42],
        horizontal_fov_degrees=75.0,
        vertical_fov_degrees=46.7,
    )
    registry = build_robot_camera_profile_registry(
        job_request={"robot_profile": _unitree_g1_owner_profile(cameras=[head_only])},
        scenario_cards={},
        generated_at=GENERATED_AT,
    )
    readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=GENERATED_AT,
        launch_mode=True,
    )

    profile = registry["profiles"][0]
    assert readiness["status"] == "ready"
    assert readiness["ready_for_launch"] is True
    assert readiness["blockers"] == []
    assert readiness["physical_robot_launch_ready_profile_count"] == 0
    assert profile["smoke_only"] is True
    assert profile["calibration_contract"]["physical_robot_camera_shape_valid"] is False
    assert profile["calibration_contract"]["missing_required_physical_camera_ids"] == [
        "chest_rgbd"
    ]
    assert (
        "missing_required_unitree_g1_rgbd_camera:unitree_g1:chest_rgbd"
        in readiness["physical_robot_calibration_blockers"]
    )
    missing = readiness["physical_robot_calibration_inputs_needed_for_physical_launch"]
    chest_missing = next(item for item in missing if item["camera_id"] == "chest_rgbd")
    assert chest_missing["missing_launch_fields"] == [
        "owner_provided_camera_profile",
        "owner_provided_intrinsics",
        "owner_provided_extrinsics",
        "owner_provided_fov",
    ]
    assert {
        field["path"] for field in chest_missing["required_owner_fields"]
    } >= {
        "cameras[?camera_id=='chest_rgbd'].modalities",
        "cameras[?camera_id=='chest_rgbd'].intrinsics.fx",
        "cameras[?camera_id=='chest_rgbd'].extrinsics.reference_frame",
        "cameras[?camera_id=='chest_rgbd'].horizontal_fov_degrees",
    }


def test_physical_launch_blocks_without_owner_robot_base_to_camera_extrinsics() -> None:
    cameras = _unitree_g1_owner_profile()["cameras"]
    cameras[0]["extrinsics"] = {
        **cameras[0]["extrinsics"],
        "reference_frame": "map",
    }
    registry = build_robot_camera_profile_registry(
        job_request={"robot_profile": _unitree_g1_owner_profile(cameras=cameras)},
        scenario_cards={},
        generated_at=GENERATED_AT,
    )
    readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=GENERATED_AT,
        launch_mode=True,
    )
    request = build_owner_robot_camera_calibration_request(
        registry=registry,
        launch_readiness=readiness,
        generated_at=GENERATED_AT,
    )

    profile = registry["profiles"][0]
    head = next(camera for camera in profile["cameras"] if camera["camera_id"] == "head_rgbd")
    assert readiness["status"] == "ready"
    assert readiness["ready_for_launch"] is True
    assert readiness["blockers"] == []
    assert readiness["physical_robot_launch_ready_profile_count"] == 0
    assert head["calibration_contract"]["owner_provided_extrinsics"] is False
    assert head["calibration_contract"]["missing_launch_fields"] == [
        "owner_provided_extrinsics"
    ]
    assert "missing_owner_provided_extrinsics:unitree_g1:head_rgbd" in (
        readiness["physical_robot_calibration_blockers"]
    )
    missing = request["optional_physical_robot_calibration_inputs"]
    head_missing = next(item for item in missing if item["camera_id"] == "head_rgbd")
    assert head_missing["missing_launch_fields"] == ["owner_provided_extrinsics"]
    assert {
        field["path"] for field in head_missing["required_owner_fields"]
    } >= {
        "cameras[?camera_id=='head_rgbd'].extrinsics.reference_frame",
        "cameras[?camera_id=='head_rgbd'].extrinsics.child_frame",
        "cameras[?camera_id=='head_rgbd'].extrinsics.xyz_m",
        "cameras[?camera_id=='head_rgbd'].extrinsics.rpy_rad",
    }


def test_owner_camera_calibration_request_packet_documents_required_inputs() -> None:
    registry = build_robot_camera_profile_registry(
        job_request={},
        scenario_cards={},
        generated_at=GENERATED_AT,
    )
    readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=GENERATED_AT,
        launch_mode=False,
    )
    request = build_owner_robot_camera_calibration_request(
        registry=registry,
        launch_readiness=readiness,
        generated_at=GENERATED_AT,
    )

    assert request["schema_version"] == "owner_robot_camera_calibration_request.v1"
    assert request["artifact_is_calibration_proof"] is False
    assert request["request_packet_is_not_owner_calibration"] is True
    assert request["accepted_owner_calibration_evidence_in_this_artifact"] is False
    assert request["physical_robot_claim_upgrade_proven_by_this_artifact"] is False
    assert request["status"] == "not_required_for_sim_only"
    assert request["ready_for_launch"] is True
    assert request["required_profile_count"] == 0
    assert request["profiles"] == []
    assert request["missing_owner_calibration_inputs"] == []
    assert request["physical_robot_calibration_profiles"][0]["canonical_owner_profile_path"] == (
        "pipeline/robot_eval_inputs/robot_camera_profile_calibration/"
        "unitree_g1_owner_robot_camera_profile.json"
    )
    template = request["physical_robot_calibration_profiles"][0][
        "owner_profile_schema_template"
    ]
    assert template["robot_profile_id"] == "unitree_g1"
    assert template["source"] == "owner_provided_robot_team_camera_calibration"
    assert {camera["camera_id"] for camera in template["cameras"]} == {
        "head_rgbd",
        "chest_rgbd",
    }
    assert template["cameras"][0]["intrinsics"]["fx"] is None
    assert "Export intrinsics in pixel units" in " ".join(request["capture_procedure"])
    assert request["claim_boundary"]["request_packet_is_not_owner_calibration"] is True
    assert request["claim_boundary"]["request_packet_is_shape_contract_not_evidence"] is True
    assert request["claim_boundary"]["artifact_is_calibration_proof"] is False
    assert request["claim_boundary"]["owner_calibration_not_required_for_sim_only_launch"] is True
    readiness_key = "_".join(["physical", "robot", "readiness", "proven"])
    assert readiness_key not in request["claim_boundary"]
    assert request["claim_boundary"]["safety_validation_proven"] is False


def test_robot_camera_profile_launch_readiness_validates_ten_owner_profiles() -> None:
    profiles = [_owner_robot_profile(index) for index in range(1, 11)]
    registry = build_robot_camera_profile_registry(
        job_request={"robot_profiles": profiles},
        scenario_cards={
            "cards": [
                {
                    "task_id": "inspect_bin",
                    "scenario_id": "scenario-inspect-bin",
                    "robot_profile_id": "owner_robot_01",
                }
            ]
        },
        generated_at=GENERATED_AT,
    )
    readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=GENERATED_AT,
        launch_mode=True,
    )

    assert registry["profile_count"] == 10
    assert readiness["status"] == "ready"
    assert readiness["profile_count"] == 10
    assert readiness["launch_ready_profile_count"] == 10
    assert readiness["ready_for_launch"] is True
    assert readiness["owner_calibration_request_packet_path"] is None
    assert readiness["missing_owner_calibration_inputs"] == []
    assert readiness["smoke_only_profile_count"] == 0
    assert readiness["blockers"] == []
    assert {profile["robot_profile_id"] for profile in readiness["profiles"]} == {
        f"owner_robot_{index:02d}" for index in range(1, 11)
    }
    for profile in registry["profiles"]:
        assert profile["smoke_only"] is False
        assert profile["calibration_contract"]["launch_ready"] is True
        camera = profile["cameras"][0]
        assert camera["calibration_contract"]["owner_provided_intrinsics"] is True
        assert camera["calibration_contract"]["owner_provided_extrinsics"] is True
        assert camera["calibration_contract"]["owner_provided_fov"] is True


def test_initial_observation_resolver_selects_direct_capture_frame(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    frame = _write_png(capture_root / "raw" / "frames" / "head_0001.png")
    depth = _write_depth(capture_root / "raw" / "depth" / "head_0001.npy")
    _write_object_grounding(capture_root)
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "capture_frame_index.json",
        {
            "frames": [
                {
                    "frame_id": "head-0001",
                    "image_path": str(frame.relative_to(capture_root)),
                    "depth_path": str(depth.relative_to(capture_root)),
                    "camera_id": "head_rgbd",
                    "robot_profile_id": "unitree_g1",
                    "source_kind": "robot_pov_capture_frame",
                    "intrinsics": {"width": 64, "height": 64, "fx": 60, "fy": 60, "cx": 32, "cy": 32},
                    "T_world_camera": IDENTITY_POSE,
                }
            ]
        },
    )

    result = build_initial_observation_source_resolution(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"robot_profile": {"robot_profile_id": "unitree_g1"}},
        scenario_cards=_scenario_cards(),
        scenario_eval_matrix=_scenario_eval_matrix(),
        generated_at=GENERATED_AT,
    )

    candidate_set = result["candidate_set"]
    selected = result["selected_initial_policy_observation"]
    assert (job_dir / "robot_pov_observation_candidate_set.json").is_file()
    assert (job_dir / "selected_initial_policy_observation.json").is_file()
    assert candidate_set["selected_source_kind"] == "direct_capture_frame"
    assert selected["selection_source_kind"] == "direct_capture_frame"
    assert selected["visual_observation"]["capture_truth"] is True
    assert selected["provenance"]["capture_truth"] is True
    assert selected["camera"]["intrinsics"]["width"] == 960
    assert selected["camera"]["extrinsics"]["reference_frame"] == "robot_base"
    assert selected["source_qa"]["status"] == "ready"
    assert Path(selected["source_qa_path"]).is_file()
    assert Path(selected["contact_sheet_path"]).is_file()
    assert Path(selected["recapture_guidance_path"]).is_file()
    assert candidate_set["paid_provider_calls_performed"] is False


def test_initial_observation_resolver_uses_depth_splat_and_indexes_3dgs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    frame = _write_png(capture_root / "raw" / "frames" / "walkthrough_0001.png", size=(32, 32))
    depth = np.full((32, 32), 2.0, dtype=np.float32)
    depth_path = capture_root / "raw" / "depth" / "walkthrough_0001.npy"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, depth)
    _write_object_grounding(capture_root)
    gs_path = capture_root / "pipeline" / "advanced_geometry" / "3dgs_compressed.ply"
    gs_path.parent.mkdir(parents=True, exist_ok=True)
    gs_path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "capture_frame_index.json",
        {
            "frames": [
                {
                    "frame_id": "walkthrough-0001",
                    "image_path": str(frame.relative_to(capture_root)),
                    "depth_path": str(depth_path.relative_to(capture_root)),
                    "camera_id": "walkthrough_rgb",
                    "intrinsics": {"width": 32, "height": 32, "fx": 30, "fy": 30, "cx": 16, "cy": 16},
                    "T_world_camera": IDENTITY_POSE,
                }
            ]
        },
    )

    result = build_initial_observation_source_resolution(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "robot_profile": {"robot_profile_id": "unitree_g1"},
            "initial_observation": {
                "camera_id": "head_rgbd",
                "target_camera_pose": IDENTITY_POSE,
            },
        },
        scenario_cards=_scenario_cards(),
        scenario_eval_matrix=_scenario_eval_matrix(),
        generated_at=GENERATED_AT,
    )

    candidate_set = result["candidate_set"]
    selected = result["selected_initial_policy_observation"]
    source_kinds = {candidate["source_kind"] for candidate in candidate_set["candidates"]}
    assert "capture_derived_depth_splat" in source_kinds
    assert "capture_derived_3dgs" in source_kinds
    assert "direct_capture_frame" not in source_kinds
    assert candidate_set["selected_source_kind"] == "capture_derived_depth_splat"
    assert selected["visual_observation"]["capture_derived"] is True
    assert selected["visual_observation"]["capture_truth"] is False
    assert Path(selected["camera_frame_path"]).is_file()
    assert selected["provenance"]["paid_provider_call_performed"] is False

    three_dgs = next(
        candidate
        for candidate in candidate_set["candidates"]
        if candidate["source_kind"] == "capture_derived_3dgs"
    )
    assert three_dgs["status"] == "renderer_required"
    assert three_dgs["synthesis"]["renderer_required"] is True


def test_initial_observation_resolver_ingests_pipeline_geometry_records(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    geometry_dir = capture_root / "pipeline" / "geometry"
    frame_path = geometry_dir / "frames" / "images" / "frame_000000.npy"
    depth_path = geometry_dir / "depth" / "depth_000000.npy"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(frame_path, np.full((24, 32, 3), 120, dtype=np.float32))
    np.save(depth_path, np.full((24, 32), 1.5, dtype=np.float32))
    _write_json(
        geometry_dir / "camera" / "intrinsics.json",
        {
            "image_width": 32,
            "image_height": 24,
            "fx": 30.0,
            "fy": 30.0,
            "cx": 16.0,
            "cy": 12.0,
        },
    )
    (geometry_dir / "camera" / "poses.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (geometry_dir / "camera" / "poses.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "000000",
                "frame_index": 0,
                "world_from_camera": IDENTITY_POSE,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (geometry_dir / "frames" / "frame_index.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (geometry_dir / "frames" / "frame_index.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "000000",
                "frame_index": 0,
                "image_path": str(frame_path),
                "depth_path": str(depth_path),
                "intrinsics_present": True,
                "pose_present": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_object_grounding(capture_root, object_id="bin_0001")

    result = build_initial_observation_source_resolution(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "robot_profile": {"robot_profile_id": "unitree_g1"},
            "initial_observation": {
                "camera_id": "head_rgbd",
                "target_camera_pose": IDENTITY_POSE,
                "target_object_ids": ["bin_0001"],
            },
        },
        scenario_cards=_scenario_cards(),
        scenario_eval_matrix=_scenario_eval_matrix(),
        generated_at=GENERATED_AT,
    )

    candidate_set = result["candidate_set"]
    selected = result["selected_initial_policy_observation"]
    assert candidate_set["status"] == "ready"
    assert candidate_set["selected_source_kind"] == "capture_derived_depth_splat"
    assert candidate_set["source_qa_summary"]["local_depth_count"] == 1
    assert selected["status"] == "ready"
    assert selected["selection_source_kind"] == "capture_derived_depth_splat"
    assert selected["source_qa"]["status"] == "ready"
    assert selected["object_grounding"]["target_object_ids"] == ["bin_0001"]
    assert Path(selected["camera_frame_path"]).is_file()
    assert Path(selected["contact_sheet_path"]).is_file()


def test_initial_observation_resolver_fails_closed_without_required_capture_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"

    result = build_initial_observation_source_resolution(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"robot_profile": {"robot_profile_id": "mobile_manipulator_rgb_v1"}},
        scenario_cards=_scenario_cards(),
        scenario_eval_matrix=_scenario_eval_matrix(),
        generated_at=GENERATED_AT,
    )

    candidate_set = result["candidate_set"]
    selected = result["selected_initial_policy_observation"]
    assert candidate_set["status"] == "blocked"
    assert candidate_set["synthetic_fallback_allowed"] is False
    assert candidate_set["selected_source_kind"] is None
    assert selected["status"] == "blocked"
    assert selected["selection_source_kind"] is None
    assert selected["claim_boundary"]["selected_synthetic_fallback"] is False
    assert selected["claim_boundary"]["capture_truth"] is False
    assert selected["claim_boundary"]["geometry_truth"] is False
    assert selected["claim_boundary"]["visually_useful_rollout"] is False
    assert selected["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False
    assert "capture_frame_index_missing" in selected["blockers"]
    assert "depth_map_missing" in selected["blockers"]
    assert "camera_pose_missing" in selected["blockers"]
    assert "camera_intrinsics_missing" in selected["blockers"]
    assert "object_grounding_missing" in selected["blockers"]
    assert "no_capture_backed_initial_policy_observation_candidate" in selected["blockers"]
    assert Path(selected["source_qa_path"]).is_file()
    assert Path(selected["contact_sheet_path"]).is_file()
    guidance = _read_json(Path(selected["recapture_guidance_path"]))
    assert guidance["status"] == "recapture_required"


def test_robot_pov_bundle_emits_blocked_default_profile_launch_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    capture_root = _capture_root(tmp_path)
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(robot_eval_dir / "scenario_cards.json", _scenario_cards())
    _write_json(robot_eval_dir / "task_cards.json", _task_cards())
    job_dir = tmp_path / "job"

    manifest = build_robot_pov_observation_bundle(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "robot_profile": {"robot_profile_id": "unitree_g1"},
            "requested_tasks": [{"task_id": "inspect_bin", "scenario_ids": ["scenario-inspect-bin"]}],
        },
        scenario_eval_matrix=_scenario_eval_matrix(),
        generated_at=GENERATED_AT,
    )

    assert (job_dir / "robot_pov_observation_candidate_set.json").is_file()
    assert (job_dir / "selected_initial_policy_observation.json").is_file()
    assert (job_dir / "initial_policy_observation_source_qa.json").is_file()
    assert (job_dir / "initial_policy_observation_contact_sheet.jpg").is_file()
    assert (job_dir / "initial_policy_observation_recapture_guidance.json").is_file()
    assert (job_dir / "owner_robot_camera_calibration_request.json").is_file()
    resolver = manifest["initial_observation_source_resolver"]
    assert resolver["candidate_set_path"] == "robot_pov_observation_candidate_set.json"
    assert resolver["selected_initial_policy_observation_path"] == (
        "selected_initial_policy_observation.json"
    )
    assert resolver["source_qa_path"] == "initial_policy_observation_source_qa.json"
    assert resolver["contact_sheet_path"] == "initial_policy_observation_contact_sheet.jpg"
    assert resolver["recapture_guidance_path"] == (
        "initial_policy_observation_recapture_guidance.json"
    )
    selected = _read_json(job_dir / "selected_initial_policy_observation.json")
    readiness = _read_json(job_dir / "robot_camera_profile_launch_readiness.json")
    assert selected["schema_version"] == "selected_initial_policy_observation.v1"
    assert selected["status"] == "blocked"
    assert selected["selection_source_kind"] is None
    assert readiness["status"] == "ready"
    assert readiness["launch_mode"] is True
    assert readiness["ready_for_launch"] is True
    assert readiness["default_smoke_only_profile_count"] == 1
    assert resolver["camera_profile_launch_readiness_status"] == "ready"
    assert resolver["owner_robot_camera_calibration_request_path"] == (
        "owner_robot_camera_calibration_request.json"
    )
    request = _read_json(job_dir / "owner_robot_camera_calibration_request.json")
    assert request["status"] == "not_required_for_sim_only"
    assert request["ready_for_launch"] is True
    assert request["physical_robot_calibration_profiles"][0]["missing_camera_count"] == 2


def test_robot_pov_bundle_writes_launch_ready_ten_profile_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    capture_root = _capture_root(tmp_path)
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    scenario_cards = {
        "cards": [
            {
                "task_id": "inspect_bin",
                "scenario_id": "scenario-inspect-bin",
                "robot_profile_id": "owner_robot_01",
            }
        ]
    }
    _write_json(robot_eval_dir / "scenario_cards.json", scenario_cards)
    _write_json(robot_eval_dir / "task_cards.json", _task_cards())
    job_dir = tmp_path / "job"
    profiles = [_owner_robot_profile(index) for index in range(1, 11)]

    manifest = build_robot_pov_observation_bundle(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "robot_profiles": profiles,
            "robot_profile_id": "owner_robot_01",
            "requested_tasks": [
                {"task_id": "inspect_bin", "scenario_ids": ["scenario-inspect-bin"]}
            ],
        },
        scenario_eval_matrix=_scenario_eval_matrix(),
        generated_at=GENERATED_AT,
    )

    registry = _read_json(job_dir / "robot_camera_profile_registry.json")
    readiness = _read_json(job_dir / "robot_camera_profile_launch_readiness.json")
    request = _read_json(job_dir / "owner_robot_camera_calibration_request.json")
    candidate_set = _read_json(job_dir / "robot_pov_observation_candidate_set.json")
    assert registry["profile_count"] == 10
    assert readiness["status"] == "ready"
    assert readiness["launch_mode"] is True
    assert readiness["profile_count"] == 10
    assert readiness["launch_ready_profile_count"] == 10
    assert readiness["ready_for_launch"] is True
    assert request["status"] == "not_required"
    assert request["ready_for_launch"] is True
    assert candidate_set["camera_profile_registry_path"] == "robot_camera_profile_registry.json"
    assert (
        candidate_set["camera_profile_launch_readiness_path"]
        == "robot_camera_profile_launch_readiness.json"
    )
    resolver = manifest["initial_observation_source_resolver"]
    assert resolver["camera_profile_count"] == 10
    assert resolver["camera_profile_launch_readiness_status"] == "ready"
    assert manifest["robot_camera_profile_launch_readiness"]["all_profiles_launch_ready"] is True
