from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
    _run_wrist_camera_mount_sweep,
)
from blueprint_pipeline.native_task_wrist_camera_mount_sweep import (
    WristCameraMountSweepError,
    resolve_wrist_camera_mount_eyes,
    select_wrist_camera_mount_candidate,
    validate_wrist_camera_mount_registry,
)


REGISTRY = (
    Path(__file__).parents[1]
    / "docs/arm_decision_proof_v1/manifests/"
    "franka_robotiq_policy_camera_mount_registry.v1.json"
)


def _registry() -> dict:
    return json.loads(REGISTRY.read_text())


def _observation(row: dict, *, task_pixels: int, robot_fraction: float) -> dict:
    return {
        **row,
        "frame_png": {"path": f"{row['candidate_id']}.png", "sha256": "sha256:" + "a" * 64},
        "task_object": {
            "pixel_count": task_pixels,
            "pixel_fraction": task_pixels / (640 * 360),
        },
        "robot": {"pixel_fraction": robot_fraction},
        "frame_structure_passed": True,
    }


def test_registry_resolves_mirrored_task_facing_mounts() -> None:
    registry = validate_wrist_camera_mount_registry(_registry())
    rows = resolve_wrist_camera_mount_eyes(
        registry=registry,
        controlled_body_position_world_m=[0.0, -1.0, 1.0],
        task_target_position_world_m=[0.0, 0.0, 0.0],
    )

    assert len(rows) == 6
    assert rows[0]["eye_position_world_m"][0] == pytest.approx(0.1)
    assert rows[1]["eye_position_world_m"][0] == pytest.approx(-0.1)
    assert rows[0]["target_position_world_m"] == [0.0, 0.0, 0.0]


def test_selection_uses_real_task_pixels_then_robot_occlusion() -> None:
    registry = _registry()
    resolved = resolve_wrist_camera_mount_eyes(
        registry=registry,
        controlled_body_position_world_m=[0.0, -1.0, 1.0],
        task_target_position_world_m=[0.0, 0.0, 0.0],
    )
    observations = [
        _observation(row, task_pixels=500 + index, robot_fraction=0.1)
        for index, row in enumerate(resolved)
    ]
    observations[-1]["robot"]["pixel_fraction"] = 0.8

    result = select_wrist_camera_mount_candidate(
        registry=registry, observations=observations
    )

    assert result["status"] == "selected"
    assert result["selected_candidate"]["candidate_id"] == resolved[-2][
        "candidate_id"
    ]
    assert result["observations"][-1]["admitted"] is False
    assert "wrist_camera_mount_robot_occlusion_above_ceiling" in result[
        "observations"
    ][-1]["blockers"]


def test_selection_blocks_when_no_candidate_sees_the_task() -> None:
    registry = _registry()
    resolved = resolve_wrist_camera_mount_eyes(
        registry=registry,
        controlled_body_position_world_m=[0.0, -1.0, 1.0],
        task_target_position_world_m=[0.0, 0.0, 0.0],
    )
    result = select_wrist_camera_mount_candidate(
        registry=registry,
        observations=[
            _observation(row, task_pixels=0, robot_fraction=0.0)
            for row in resolved
        ],
    )

    assert result["status"] == "blocked"
    assert result["selected_candidate"] is None


def test_registry_tamper_is_refused() -> None:
    registry = _registry()
    registry["candidates"][0]["forward_offset_m"] = 0.2

    with pytest.raises(WristCameraMountSweepError):
        validate_wrist_camera_mount_registry(registry)


def test_runtime_sweep_uses_isaac_view_pose_and_reapplies_selected(
    tmp_path: Path,
) -> None:
    registry = _registry()

    class Camera:
        def __init__(self) -> None:
            self.calls: list[tuple[list, list]] = []
            self.updates = 0

        def set_world_poses_from_view(self, *, eyes, targets) -> None:
            self.calls.append((eyes, targets))

        def update(self, _dt: float, *, force_recompute: bool) -> None:
            assert force_recompute is True
            self.updates += 1

    class App:
        def __init__(self) -> None:
            self.updates = 0

        def update(self) -> None:
            self.updates += 1

    camera = Camera()
    app = App()
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(scene={"robot": object(), "wrist_camera": camera})
    )
    built = SimpleNamespace(camera_scene_names={"wrist": "wrist_camera"})

    def snapshot(*, output_root: Path, **_kwargs) -> dict:
        candidate_id = output_root.name
        index = int(candidate_id.rsplit("_", 1)[-1])
        frame = output_root / "construction_frames/wrist/candidate.png"
        frame.parent.mkdir(parents=True)
        Image.fromarray(
            np.full((18, 32, 3), 20 + index, dtype=np.uint8), mode="RGB"
        ).save(frame)
        return {
            "cameras": [
                {
                    "rgb_png": {
                        "path": "construction_frames/wrist/candidate.png",
                        "sha256": "sha256:" + hashlib.sha256(frame.read_bytes()).hexdigest(),
                    },
                    "semantic_label_pixels": {
                        "task_object": {
                            "pixel_count": 500 + index,
                            "pixel_fraction": (500 + index) / (640 * 360),
                        },
                        "robot": {
                            "pixel_fraction": 0.8 if index == 6 else 0.1,
                        },
                    },
                    "observability": {"render_passed": True},
                }
            ]
        }

    result = _run_wrist_camera_mount_sweep(
        simulation_app=app,
        env=env,
        built=built,
        packet_request={
            "wrist_camera_mount_registry": registry,
            "cameras": [
                {
                    "role": "wrist",
                    "parent_prim_path": (
                        "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
                    ),
                }
            ],
        },
        plan={"task_spec": {"start_pose_world": [0, 0, 0, 0, 0, 0, 1]}},
        output_root=tmp_path,
        torch=object(),
        body_pose_reader=lambda *_args, **_kwargs: [0, -1, 1, 0, 0, 0, 1],
        camera_snapshot=snapshot,
    )

    assert result is not None
    assert result["status"] == "selected"
    assert result["selected_candidate"]["candidate_id"].endswith("05")
    assert len(camera.calls) == 7
    assert camera.updates == 7
    assert app.updates == 42
    assert (tmp_path / "wrist_mount_sweep/contact_sheet.png").is_file()
