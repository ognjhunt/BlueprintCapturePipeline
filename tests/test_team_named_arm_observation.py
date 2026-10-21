"""Named TCP proprioception remains bounded and separate from evaluator state."""
from copy import deepcopy

import numpy as np
import pytest

from blueprint_pipeline.team_policy_https_client import encode_team_observation


def observation():
    return {"images": {"overhead": np.zeros((8, 8, 3), dtype=np.uint8)}, "prompt": "move mug",
            "state": {"joint_position": [0.1, 0.2], "joint_velocity": [0, 0],
                      "gripper_position": [0.3, 0.4], "gripper_names": ["left_tool", "right_tool"],
                      "joint_names": ["left_joint", "right_joint"], "arm_names": ["left", "right"],
                      "arm_joint_names": {"left": ["left_joint"], "right": ["right_joint"]},
                      "arm_gripper_names": {"left": "left_tool", "right": "right_tool"},
                      "arm_eef_pose": {arm: {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}
                                       for arm in ("left", "right")},
                      "base": {"position_world_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
                               "linear_velocity_world_mps": [0, 0, 0], "angular_velocity_world_radps": [0, 0, 0]}}}


def test_exact_named_arm_state_reaches_generic_controller_bridge():
    value = observation()
    wire = encode_team_observation(value, camera_names=["overhead"])
    assert wire["state"] == value["state"]


@pytest.mark.parametrize("mutation", ["missing", "foreign_joint", "overlap", "tool", "quaternion", "nonfinite", "self_grade"])
def test_malformed_arm_state_and_evaluator_leakage_refused(mutation):
    value = deepcopy(observation())
    state = value["state"]
    if mutation == "missing":
        del state["joint_names"]
    elif mutation == "foreign_joint":
        state["arm_joint_names"]["left"] = ["other_joint"]
    elif mutation == "overlap":
        state["arm_joint_names"]["right"] = ["left_joint"]
    elif mutation == "tool":
        state["arm_gripper_names"]["left"] = "right_tool"
    elif mutation == "quaternion":
        state["arm_eef_pose"]["left"]["orientation_xyzw"] = [0, 0, 0, 0]
    elif mutation == "nonfinite":
        state["arm_eef_pose"]["left"]["position_m"][0] = float("nan")
    else:
        state["task_success"] = True
    with pytest.raises(ValueError):
        encode_team_observation(value, camera_names=["overhead"])
