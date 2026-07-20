from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from blueprint_pipeline import g1_microwave_reach_seed as reach
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    PINNED_WBC_SOURCE_REVISION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gear_sonic_g1_min"
    / "g1_29dof_with_hand_min.xml"
)


def _joint_limited_model(tmp_path: Path) -> Path:
    tree = ET.parse(FIXTURE)
    root = tree.getroot()
    names = set(reach.RIGHT_ARM_JOINT_NAMES)
    for joint in root.findall(".//joint"):
        if joint.get("name") in names:
            joint.set("limited", "true")
            joint.set("range", "-3 3")
    model = tmp_path / "g1_29dof_with_hand_test.xml"
    tree.write(model, encoding="unicode")
    return model


def _write_inputs(tmp_path: Path, model_path: Path) -> tuple[Path, Path]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    target_values = (-0.35, 0.1, -0.1, 0.4, 0.05, 0.1, -0.05)
    for joint_name, value in zip(reach.RIGHT_ARM_JOINT_NAMES, target_values):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(model, data)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    effector_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, reach.DEFAULT_EFFECTOR
    )
    target_relative = (
        np.asarray(data.xpos[effector_id]) - np.asarray(data.xpos[pelvis_id])
    ).tolist()
    standing = tmp_path / "standing.json"
    standing.write_text(
        json.dumps(
            {
                "status": "passed",
                "surrogate": False,
                "pinned_wbc_source_revision": PINNED_WBC_SOURCE_REVISION,
                "measured_full_joint_positions": [
                    0.0 for _ in PROTOCOL_V4_FULL_JOINT_ORDER
                ],
            }
        ),
        encoding="utf-8",
    )
    observation = tmp_path / "observation.json"
    observation.write_text(
        json.dumps(
            {
                "target_prim_path": "/root/Microwave017/Microwave017_Door",
                "camera_projection_context": {
                    "status": "captured_from_live_persistent_isaac_session",
                    "live_isaac_pelvis_world_pose": {
                        "position_xyz": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "camera_contract": {
                        "calibration_target_world_xyz_m": target_relative
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return standing, observation


def _write_focus_report(tmp_path: Path, target_world_xyz: list[float]) -> Path:
    hinge_world_xyz = list(target_world_xyz)
    hinge_world_xyz[1] -= 0.10
    report = tmp_path / "focus.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "articulated_handle_focus.v1",
                "status": "resolved_disconnected_articulated_handle",
                "target_prim_path": "/root/Microwave017/Microwave017_Door",
                "joint_prim_path": (
                    "/root/Microwave017/Microwave017_Door/RevoluteJoint"
                ),
                "target_world_xyz_m": target_world_xyz,
                "hinge_world_xyz_m": hinge_world_xyz,
                "joint_axis_token": "Z",
                "joint_world_axis_xyz": [0.0, 0.0, 1.0],
                "joint_lower_limit_degrees": -90.0,
                "joint_upper_limit_degrees": 0.0,
                "selected_component_count": 2,
            }
        ),
        encoding="utf-8",
    )
    return report


def test_minimum_jerk_trajectory_has_exact_endpoints() -> None:
    trajectory = reach.minimum_jerk_trajectory(
        [0.0, 1.0], [2.0, -1.0], frame_count=11
    )
    np.testing.assert_allclose(trajectory[0], [0.0, 1.0])
    np.testing.assert_allclose(trajectory[-1], [2.0, -1.0])
    np.testing.assert_allclose(trajectory[5], [1.0, 0.0])


def test_rotate_point_around_axis_preserves_hinge_radius() -> None:
    rotated = reach.rotate_point_around_axis(
        [1.0, 0.0, 0.0],
        origin=[0.0, 0.0, 0.0],
        axis=[0.0, 0.0, 1.0],
        angle_rad=-np.pi / 2.0,
    )

    np.testing.assert_allclose(rotated, [0.0, -1.0, 0.0], atol=1e-12)


def test_reach_seed_solves_only_right_arm_against_exact_input_geometry(
    tmp_path: Path,
) -> None:
    model = _joint_limited_model(tmp_path)
    standing, observation = _write_inputs(tmp_path, model)
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    target = json.loads(observation.read_text(encoding="utf-8"))[
        "camera_projection_context"
    ]["camera_contract"]["calibration_target_world_xyz_m"]
    focus = _write_focus_report(tmp_path, target)

    trajectory, report = reach.solve_reach_seed(
        model_path=model,
        standing_initialization_path=standing,
        initial_policy_observation_path=observation,
        target_focus_report_path=focus,
        expected_model_sha256=model_sha256,
        frame_count=51,
        maximum_final_distance_m=0.05,
    )

    assert trajectory.shape == (51, 43)
    assert report["status"] == "qualified_right_arm_reach_seed"
    assert report["geometry"]["progress_m"] >= 0.015
    assert report["geometry"]["final_distance_m"] <= 0.05
    assert report["trajectory"]["only_right_arm_changes"] is True
    assert report["geometry"]["target_source"]["source"] == (
        "disconnected_articulated_handle_focus"
    )
    assert report["claim_boundary"]["task_success_proven"] is False


def test_reach_seed_follows_door_hinge_arc_without_non_arm_joint_drift(
    tmp_path: Path,
) -> None:
    model = _joint_limited_model(tmp_path)
    standing, observation = _write_inputs(tmp_path, model)
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    target = json.loads(observation.read_text(encoding="utf-8"))[
        "camera_projection_context"
    ]["camera_contract"]["calibration_target_world_xyz_m"]
    focus = _write_focus_report(tmp_path, target)

    trajectory, report = reach.solve_reach_seed(
        model_path=model,
        standing_initialization_path=standing,
        initial_policy_observation_path=observation,
        target_focus_report_path=focus,
        expected_model_sha256=model_sha256,
        frame_count=51,
        pull_frame_count=11,
        door_open_angle_rad=0.10,
        maximum_final_distance_m=0.05,
    )

    assert trajectory.shape == (61, 43)
    assert report["status"] == (
        "qualified_right_arm_reach_and_kinematic_pull_seed"
    )
    assert report["pull"]["requested_door_open_observable_transition_rad"] == 0.10
    assert report["pull"]["signed_joint_angle_rad"] == -0.10
    assert report["pull"]["maximum_ik_error_m"] <= 0.05
    assert report["pull"]["kinematic_arc_following_proven"] is True
    assert report["pull"]["physics_validated_contact"] is False
    assert report["trajectory"]["only_right_arm_changes"] is True
    assert report["claim_boundary"]["kinematic_door_arc_following_proven"] is True
    assert report["claim_boundary"]["microwave_door_articulation_transition_proven"] is False


def test_reach_seed_rejects_model_hash_drift(tmp_path: Path) -> None:
    model = _joint_limited_model(tmp_path)
    standing, observation = _write_inputs(tmp_path, model)
    with pytest.raises(
        ValueError,
        match="g1_microwave_reach_seed_model_missing_or_sha256_mismatch",
    ):
        reach.solve_reach_seed(
            model_path=model,
            standing_initialization_path=standing,
            initial_policy_observation_path=observation,
            expected_model_sha256="0" * 64,
        )
