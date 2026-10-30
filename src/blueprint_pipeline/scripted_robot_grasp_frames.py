"""Audited source-tool bases for the owned scripted manipulation baseline.

The task convention is +Z approach, +Y jaw separation, +X right handed.
These are fixed simulation grasp references, not publisher calibration or a
measurement of the eventual contact point. Mesh-derived references explicitly
approximate a pad centre; articulated finger pads move as the jaws close.
Source/import verification remains owned by the prepared-model registration.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import numpy as np

from .team_arm_kinematics import pose_matrix, quaternion_xyzw
from .team_embodiment import validate_embodiment_manifest

SCHEMA = "blueprint.scripted_robot_grasp_binding.v1"
CONVENTION = "+Z approach into feature; +Y jaw separation; +X right handed"

# Position and axes are expressed in the source gripper body, not the
# manifest's TCP. In particular YAM/PiPER/Axol expose a body-origin TCP.
# The open/closed geometry was read from the immutable source recipes; the
# profile corrections for TRLC/Axol must precede registration of this script.
_FRAMES = {
    "yam": {"tool": "gripper", "contacts": ("tip_left", "tip_right"), "dof": 6,
        "position": (0., 0., .138017), "approach": (0., 0., 1.), "jaw": (1., 0., 0.),
        "jaws": {"joint7": (.04695, 0.), "joint8": (.04695, 0.)},
        "url": "https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf",
        "basis": "Distal tip visual mesh bounds, source joint7/8 transforms: Z[.1291924,.1468425], jaw travel X.",
        "assumptions": ["Symmetric centre of distal 20mm mesh vertex envelopes; visual geometry is not a measured collision pad."]},
    "piper": {"tool": "gripper_base", "contacts": ("gripper_link1", "gripper_link2"), "dof": 6,
        "position": (0., 0., .126073), "approach": (0., 0., 1.), "jaw": (0., 1., 0.),
        "jaws": {"gripper_joint1": (.05, 0.), "gripper_joint2": (-.05, 0.)},
        "url": "https://github.com/agilexrobotics/Agilex-College/tree/c2688be41e1bc99a9addd555f237a16bc839d936/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf/piper",
        "basis": "gripper_joint1/2 origins Z=.1358 with Rx(pi/2); distal collision bounds Z[.1163463,.1358], jaw travel Y.",
        "assumptions": ["Symmetric centre of distal 20mm collision-mesh vertex envelopes, not the tip-end frame at135.8mm."]},
    "fr3": {"tool": "fr3_hand", "contacts": ("fr3_leftfinger", "fr3_rightfinger"), "dof": 7,
        "position": (0., 0., .1034), "approach": (0., 0., 1.), "jaw": (0., 1., 0.),
        "jaws": {"fr3_finger_joint1": (.04, 0.), "fr3_finger_joint2": (.04, 0.)},
        "url": "https://github.com/frankarobotics/franka_description/tree/7aeeddc449edf8d62b594f9e36a81da53e7796f9/end_effectors/franka_hand",
        "basis": "Authored fr3_hand_tcp at103.4mm; opposing prismatic axes Y; distal pad boxes span Z[.0944,.1129].",
        "assumptions": ["Publisher-authored nominal TCP, without site/tool calibration or measured contact proof."]},
    "robotiq_ur": {"tool": "robotiq_85_base_link", "contacts": ("robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"), "dof": 6,
        "position": (0., 0., .134), "approach": (0., 0., 1.), "jaw": (1., 0., 0.),
        "jaws": {"robotiq_85_left_knuckle_joint": (0., .7929), "robotiq_85_right_knuckle_joint": (0., -.7929),
                 "robotiq_85_left_inner_knuckle_joint": (0., .7929), "robotiq_85_right_inner_knuckle_joint": (0., -.7929),
                 "robotiq_85_left_finger_tip_joint": (0., -.7929), "robotiq_85_right_finger_tip_joint": (0., .7929)},
        "url": "https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/a74d007d8f2f06dc6a503ad21038ba869d4999a6/robotiq_description",
        "basis": "Opposing Robotiq finger tips separate along source X. Position follows the pinned PickNik134mm grasp_link reference; its yaw is not the task's Y-jaw convention.",
        "additional_source": "https://github.com/PickNikRobotics/moveit_pro_ur_ws/blob/bab77dd0862bd6dea828ec41fb60b39d22c6f2c0/src/ur_mock/description/picknik_ur.xacro",
        "assumptions": ["Nominal reference from a separately pinned example, not calibrated Dream hardware.", "Fixed grasp position ignores about13.5mm of finger-pad Z travel during closure."]},
    "kinova": {"tool": "robotiq_arg2f_base_link", "contacts": ("left_inner_finger_pad", "right_inner_finger_pad"), "dof": 7,
        "position": (0., 0., .130324), "approach": (0., 0., 1.), "jaw": (0., 1., 0.),
        "jaws": {"finger_joint": (0., .8), "left_inner_knuckle_joint": (0., .8), "left_inner_finger_joint": (0., -.8),
                 "right_outer_knuckle_joint": (0., .8), "right_inner_knuckle_joint": (0., .8), "right_inner_finger_joint": (0., -.8)},
        "url": "https://github.com/Kinovarobotics/ros_kortex/tree/c3280d8a6a6b5d96590d51538e3f265710a6c77d/kortex_description/grippers/robotiq_2f_85",
        "basis": "Open pad-box centre from source transforms: .054904-.0041+.0471+.03242=.130324m; opposing pad centres Y=+/- .0461807553m.",
        "assumptions": ["Fixed open-pad midpoint; pad centre advances about13.9mm in Z during closure.", "The source tool_frame at120mm and yaw-1.57rad is separately transformed; it is not the pad midpoint."]},
    "xarm": {"tool": "xarm_gripper_base_link", "contacts": ("left_finger", "right_finger"), "dof": 7,
        "position": (0., 0., .158331), "approach": (0., 0., 1.), "jaw": (0., 1., 0.),
        "jaws": {name: (0., .85) for name in ("drive_joint", "left_finger_joint", "left_inner_knuckle_joint",
                                             "right_outer_knuckle_joint", "right_finger_joint", "right_inner_knuckle_joint")},
        "url": "https://github.com/xArm-Developer/xarm_ros2/tree/ac736c92b02de7ca7e0be6f737d77d46707464d8/xarm_description",
        "basis": "Opposing STL fingers travel Y. Distal20mm envelope common to open/closed spans Z[.1545213,.1621402]; its centre is158.331mm.",
        "assumptions": ["Symmetric centre of the open/closed distal-mesh overlap, not calibrated contact geometry.", "Authored link_tcp at172mm lies beyond the open finger tips; script uses a pad reference instead."]},
    "axol": {"tool": "gripper", "contacts": ("finger_1", "finger_2"), "dof": 7,
        "position": (0., 0., -.14254), "approach": (0., 0., -1.), "jaw": (1., 0., 0.),
        "jaws": {"finger_1_joint": (-.044, 0.), "finger_2_joint": (-.044, 0.)},
        "url": "https://github.com/almond-bot/axol/blob/3ccc69cf3adaea09845456148d6a4f16ef059f27/almond_axol/kinematics/urdf/axol_standalone.urdf",
        "basis": "Standalone collision tips extend toward -Z, distal bounds Z[-.1495394,-.1355384]; prismatic jaws travel X. q=-.044 opens, q=0 closes.",
        "assumptions": ["Symmetric distal pad-envelope centre rounded to10micrometres; small source left/right asymmetries are ignored."]},
    "trlc": {"tool": "link6_7", "contacts": ("finger_left", "finger_right"), "dof": 6,
        "position": (.15285, 0., 0.), "approach": (1., 0., 0.), "jaw": (0., 1., 0.),
        "jaws": {"gripper_left": (0., -.045), "gripper_right": (0., -.045)},
        "url": "https://github.com/robot-learning-co/trlc-dk1/blob/12f5368aefd0381461f2c7ffbb5611b4e8c90de9/urdf/follower/TRLC-DK1-Follower.urdf",
        "basis": "Distal collision pads X[.1480064,.1576931], jaws Y. Source tool0 atX=.158 has its own rotated basis; q=0 opens, negative q closes.",
        "assumptions": ["Symmetric distal-mesh envelope centre, not calibrated contact geometry.", "Full closed command overlaps distal envelopes by about10mm; native contact response and nominal width are unmeasured."]},
}

_IDENTITIES = {
    "i2rt_yam_bimanual": "yam", "agilex_piper_bimanual": "piper", "franka_research_3_bimanual": "fr3",
    "universal_robots_ur5e_bimanual": "robotiq_ur", "kinova_gen3": "kinova", "kinova_gen3_bimanual": "kinova",
    "ufactory_xarm7": "xarm", "ufactory_xarm7_bimanual": "xarm", "almond_axol": "axol", "trlc_dk1_bimanual": "trlc",
}


def grasp_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return TCP-from-task-grasp transforms for the exact declared source tools.

    Named links, actual jaw mappings and arm count must match the audited
    source; a changed tool declaration cannot silently reuse this approximation.
    The manifest's sealed TCP may be calibrated differently: composition uses
    its full transform instead of assuming its position/orientation is zero.
    """
    declared = validate_embodiment_manifest(manifest)
    identity = declared["embodiment_id"]
    if identity not in _IDENTITIES:
        raise ValueError("scripted_grasp_source_profile_unsupported:" + identity)
    profile = _FRAMES[_IDENTITIES[identity]]
    arms, channels = declared.get("arms") or [], declared["grippers"]
    arm_names = ["arm"] if identity in {"kinova_gen3", "ufactory_xarm7"} else ["left", "right"]
    if [arm["name"] for arm in arms] != arm_names or len(channels) != len(arm_names):
        raise ValueError("scripted_grasp_source_arm_binding_mismatch")
    channel_map = {channel["name"]: channel for channel in channels}
    result = {}
    for arm in arms:
        prefix = "" if arm["name"] == "arm" else (arm["name"] + "_")
        if identity == "ufactory_xarm7_bimanual":
            prefix = {"left": "L_", "right": "R_"}[arm["name"]]
        channel = channel_map[arm["gripper_name"]]
        expected_jaws = {prefix + name: values for name, values in profile["jaws"].items()}
        if (arm["tool_link"] != prefix + profile["tool"] or len(arm["joint_names"]) != profile["dof"]
                or set(channel["contact_links"]) != {prefix + name for name in profile["contacts"]}
                or {joint["name"] for joint in channel["joints"]} != set(expected_jaws)):
            raise ValueError("scripted_grasp_source_tool_binding_mismatch:" + arm["name"])
        if any(not math.isclose(joint[field], expected_jaws[joint["name"]][index], abs_tol=1e-8)
               for joint in channel["joints"] for index, field in enumerate(("open", "closed"))):
            raise ValueError("scripted_grasp_source_closure_mismatch:" + arm["name"])
        z, y = np.array(profile["approach"]), np.array(profile["jaw"])
        tool_from_grasp = np.eye(4)
        tool_from_grasp[:3, :3] = np.column_stack((np.cross(y, z), y, z))
        tool_from_grasp[:3, 3] = profile["position"]
        tcp = arm["tool_from_tcp"]
        tcp_from_grasp = np.linalg.inv(pose_matrix(tcp["position_m"], tcp["orientation_xyzw"])) @ tool_from_grasp
        result[arm["name"]] = {
            "schema_version": SCHEMA, "tcp_from_grasp": {"position_m": tcp_from_grasp[:3, 3].tolist(),
                "orientation_xyzw": quaternion_xyzw(tcp_from_grasp[:3, :3])},
            "basis_reference": profile["url"], "source_geometry_basis": profile["basis"],
            "task_grasp_convention": CONVENTION,
            "additional_source_references": [profile["additional_source"]] if "additional_source" in profile else [],
            "simulation_assumptions": deepcopy(profile["assumptions"]),
            "publisher_calibration_claimed": False, "physical_contact_point_measured": False,
            "native_task_success_claimed": False,
        }
    return result
