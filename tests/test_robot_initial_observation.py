from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from blueprint_pipeline.robot_eval_execution import build_robot_pov_observation_bundle
from blueprint_pipeline.robot_initial_observation import (
    build_initial_observation_source_resolution,
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


def test_robot_camera_profile_launch_readiness_blocks_default_profiles() -> None:
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
    assert readiness["status"] == "blocked"
    assert readiness["launch_mode"] is True
    assert readiness["defaults_are_smoke_only"] is True
    assert readiness["default_smoke_only_profile_count"] == 1
    assert readiness["launch_ready_profile_count"] == 0
    assert "default_robot_camera_profile_smoke_only:unitree_g1" in readiness["blockers"]


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
    assert selected["claim_boundary"]["physical_robot_readiness_proven"] is False
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
    assert readiness["status"] == "blocked"
    assert readiness["launch_mode"] is True
    assert readiness["default_smoke_only_profile_count"] == 1
    assert resolver["camera_profile_launch_readiness_status"] == "blocked"


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
    candidate_set = _read_json(job_dir / "robot_pov_observation_candidate_set.json")
    assert registry["profile_count"] == 10
    assert readiness["status"] == "ready"
    assert readiness["launch_mode"] is True
    assert readiness["profile_count"] == 10
    assert readiness["launch_ready_profile_count"] == 10
    assert candidate_set["camera_profile_registry_path"] == "robot_camera_profile_registry.json"
    assert (
        candidate_set["camera_profile_launch_readiness_path"]
        == "robot_camera_profile_launch_readiness.json"
    )
    resolver = manifest["initial_observation_source_resolver"]
    assert resolver["camera_profile_count"] == 10
    assert resolver["camera_profile_launch_readiness_status"] == "ready"
    assert manifest["robot_camera_profile_launch_readiness"]["all_profiles_launch_ready"] is True
