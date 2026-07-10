from __future__ import annotations

import sys
from pathlib import Path

import pytest

from blueprint_pipeline.robot_kinematics import solve_robot_forward_kinematics


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _write_urdf(path: Path) -> None:
    path.write_text(
        """<robot name="golden_robot">
  <link name="base_link"/>
  <link name="tool_link"/>
  <joint name="tool_slide" type="prismatic">
    <parent link="base_link"/>
    <child link="tool_link"/>
    <origin xyz="0 0 2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def _golden_state() -> dict[str, object]:
    return {
        "angle_unit": "radians",
        "linear_unit": "meters",
        "timestamp_unit": "seconds",
        "reference_frame": "world",
        "base_frame": "base_link",
        "reference_tolerance_m": 1e-6,
        "max_link_step_m": 0.2,
        "joint_state_frames": [
            {
                "timestamp": 0.0,
                "joint_names": ["tool_slide"],
                "joint_positions": [0.0],
                "world_from_robot_base": IDENTITY,
                "expected_link_positions": {
                    "base_link": [0.0, 0.0, 0.0],
                    "tool_link": [0.0, 0.0, 2.0],
                },
            },
            {
                "timestamp": 0.1,
                "joint_names": ["tool_slide"],
                "joint_positions": [0.1],
                "world_from_robot_base": IDENTITY,
                "expected_link_positions": {
                    "base_link": [0.0, 0.0, 0.0],
                    "tool_link": [0.1, 0.0, 2.0],
                },
            },
        ],
    }


def test_urdf_fk_solves_every_aligned_step_and_matches_reference(tmp_path: Path) -> None:
    model = tmp_path / "robot.urdf"
    _write_urdf(model)

    result = solve_robot_forward_kinematics(
        model_path=model,
        state_payload=_golden_state(),
        expected_reference_frame="world",
    )

    assert result["status"] == "completed"
    assert result["solver_executed"] is True
    assert result["solver_name"] == "blueprint_urdf_tree_fk.v1"
    assert result["frame_count"] == 2
    assert result["frames"][1]["link_positions"]["tool_link"] == [0.1, 0.0, 2.0]
    assert result["frames"][1]["reference_max_error_m"] == pytest.approx(0.0)


def test_fk_blocks_wrong_joint_order_limits_transform_time_and_reference(tmp_path: Path) -> None:
    model = tmp_path / "robot.urdf"
    _write_urdf(model)
    state = _golden_state()
    frames = state["joint_state_frames"]
    frames[0]["joint_names"] = ["wrong_joint"]
    frames[1]["joint_positions"] = [0.8]
    frames[1]["timestamp"] = 0.0
    frames[1]["world_from_robot_base"] = [
        [1, 0.2, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    frames[1]["expected_link_positions"] = {"unknown_link": [0, 0, 0]}

    result = solve_robot_forward_kinematics(
        model_path=model,
        state_payload=state,
        expected_reference_frame="world",
    )
    blockers = result["blockers"]

    assert result["status"] == "blocked"
    assert result["solver_executed"] is False
    assert "robot_joint_name_order_mismatch:frame_0" in blockers
    assert "robot_joint_above_limit:tool_slide:frame_1" in blockers
    assert "robot_timestamps_not_strictly_monotonic:frame_1" in blockers
    assert any("world_from_robot_base_rotation_not_orthonormal" in reason for reason in blockers)


def test_cartesian_landmark_copy_cannot_substitute_for_fk(tmp_path: Path) -> None:
    model = tmp_path / "robot.urdf"
    _write_urdf(model)

    result = solve_robot_forward_kinematics(
        model_path=model,
        state_payload={
            "right_end_effector_xyz": [0.1, 0.0, 2.0],
            "fk_landmarks": {"tool_link": [0.1, 0.0, 2.0]},
        },
        expected_reference_frame="world",
    )

    assert result["status"] == "blocked"
    assert result["solver_executed"] is False
    assert "robot_joint_state_sequence_missing" in result["blockers"]


def test_robot_xml_rejects_external_entity_payload(tmp_path: Path) -> None:
    model = tmp_path / "entity.urdf"
    model.write_text(
        """<!DOCTYPE robot [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<robot name="&xxe;"><link name="base_link"/></robot>
""",
        encoding="utf-8",
    )

    result = solve_robot_forward_kinematics(
        model_path=model,
        state_payload=_golden_state(),
        expected_reference_frame="world",
    )

    assert result["status"] == "blocked"
    assert result["solver_executed"] is False
    assert result["blockers"] == ["robot_model_xml_invalid"]


def test_mjcf_uses_real_mujoco_forward_solver_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "darwin":
        monkeypatch.delenv("MUJOCO_GL", raising=False)
    pytest.importorskip("mujoco")
    model = tmp_path / "robot.xml"
    model.write_text(
        """<mujoco model="golden_mjcf">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base_link" pos="0 0 2">
      <joint name="tool_slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>
      <geom type="sphere" size="0.01" mass="1"/>
      <body name="tool_link" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    state = _golden_state()
    for index, frame in enumerate(state["joint_state_frames"]):
        x = index * 0.1
        frame["expected_link_positions"] = {
            "base_link": [x, 0.0, 2.0],
            "tool_link": [x, 0.0, 2.0],
        }

    result = solve_robot_forward_kinematics(
        model_path=model,
        state_payload=state,
        expected_reference_frame="world",
    )

    assert result["status"] == "completed"
    assert result["solver_name"] == "mujoco_mj_forward"
    assert result["model_format"] == "mjcf"
