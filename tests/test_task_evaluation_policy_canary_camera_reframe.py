from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
    build_droid_observation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_camera_reframe import (
    OVERVIEW_RENDER_RESOLUTION,
    POLICY_RENDER_RESOLUTION,
    PolicyCanaryCameraReframeError,
    materialize_policy_canary_camera_reframe,
)


def _camera(role: str) -> dict:
    return {
        "role": role,
        "policy_input": role != "overview",
        "review_only": role == "overview",
        "scoring_input": False,
        "pose_frame": "robot_body" if role == "wrist" else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if role == "wrist"
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, -1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "intrinsics": {
            "fx": 172.8,
            "fy": 172.8,
            "cx": 159.5,
            "cy": 89.5,
            "width": 320,
            "height": 180,
        },
    }


def _request() -> dict:
    value = {
        "schema_version": "native_task_arena_packet_request.v1",
        "task_spec": {
            "start_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        },
        "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def _preflight() -> dict:
    return {
        "schema_version": "native_task_arena_runtime_preflight.v1",
        "status": "blocked",
        "phase_reached": "environment_built",
        "candidate_policy_queried": False,
        "camera_snapshot": {
            "cameras": [
                {
                    "role": role,
                    "position_world_m": [0.0, -1.0, 1.0],
                    "quaternion_world_opengl_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
                for role in ("external", "wrist", "overview")
            ]
        },
    }


def test_reaims_only_wrist_rotation_and_increases_capture_resolution(
    tmp_path: Path,
) -> None:
    base = _request()
    result = materialize_policy_canary_camera_reframe(
        base_request=base,
        runtime_preflight=_preflight(),
        output_path=tmp_path / "request.json",
    )

    by_role = {row["role"]: row for row in result["cameras"]}
    for role in ("external", "wrist"):
        intrinsics = by_role[role]["intrinsics"]
        assert (intrinsics["width"], intrinsics["height"]) == POLICY_RENDER_RESOLUTION
        assert intrinsics["fx"] == pytest.approx(345.6)
        assert intrinsics["cx"] == pytest.approx(319.5)
    overview = by_role["overview"]["intrinsics"]
    assert (overview["width"], overview["height"]) == OVERVIEW_RENDER_RESOLUTION
    assert overview["fx"] == pytest.approx(691.2)
    assert overview["cx"] == pytest.approx(639.5)

    assert by_role["external"]["frame_from_camera_matrix"] == base["cameras"][0][
        "frame_from_camera_matrix"
    ]
    wrist = by_role["wrist"]["frame_from_camera_matrix"]
    assert [wrist[3], wrist[7], wrist[11]] == pytest.approx([0.0, -1.0, 1.0])
    forward = [wrist[2], wrist[6], wrist[10]]
    expected = np.asarray([0.0, 1.0, -1.0]) / np.sqrt(2.0)
    assert forward == pytest.approx(expected.tolist())
    assert result["request_digest"] == canonical_digest(
        result, digest_field="request_digest"
    )
    assert result["camera_reframe"]["fresh_native_render_required"] is True


@pytest.mark.parametrize(
    ("candidate_id", "expected_shape"),
    [("pi05_droid", (224, 224, 3)), ("groot_n17_droid", (180, 320, 3))],
)
def test_high_resolution_policy_frames_keep_candidate_input_contract(
    candidate_id: str, expected_shape: tuple[int, int, int]
) -> None:
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    result = build_droid_observation(
        candidate_id=candidate_id,
        camera_rgb={DROID_EXTERIOR_VIEW_1: rgb, DROID_WRIST_VIEW: rgb},
        joint_position=[0.0] * 7,
        gripper_position=0.0,
        prompt="Push the mug forward.",
        eef_9d=[0.0] * 9 if candidate_id == "groot_n17_droid" else None,
    )

    assert result[DROID_EXTERIOR_VIEW_1].shape == expected_shape
    assert result[DROID_WRIST_VIEW].shape == expected_shape


def test_reframe_refuses_incomplete_native_camera_readback(tmp_path: Path) -> None:
    preflight = _preflight()
    preflight["camera_snapshot"]["cameras"].pop()

    with pytest.raises(
        PolicyCanaryCameraReframeError,
        match="policy_canary_camera_reframe_roles_invalid",
    ):
        materialize_policy_canary_camera_reframe(
            base_request=_request(),
            runtime_preflight=preflight,
            output_path=tmp_path / "request.json",
        )
