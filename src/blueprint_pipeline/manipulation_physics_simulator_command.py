"""Run a MuJoCo manipulation physics proof for tote pick/carry/place tasks."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


MANIPULATION_PHYSICS_OUTPUT_SCHEMA_VERSION = "mujoco_manipulation_physics_output.v1"
MANIPULATION_PHYSICS_TRACE_SCHEMA_VERSION = "mujoco_manipulation_physics_trace.v1"
DEFAULT_OUTPUT_RELATIVE = "pipeline/simulation_automation/mujoco_manipulation_physics"
G1_HAND_BODY_NAME = "g1_right_hand"
G1_HEAD_CAMERA_NAME = "g1_head_camera"
G1_WRIST_CAMERA_NAME = "g1_right_wrist_camera"
G1_CONTROLLED_JOINTS = [
    "base_x",
    "base_y",
    "base_yaw",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "right_wrist_pitch",
    "right_gripper_left_slide",
    "right_gripper_right_slide",
]
G1_ACTUATORS = [f"{name}_actuator" for name in G1_CONTROLLED_JOINTS]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pose(value: Any, default: Sequence[float]) -> list[float]:
    if isinstance(value, Mapping):
        return _pose(value.get("xyz") or value.get("pose") or value.get("position"), default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if len(parts) >= 2:
            return [
                float(_number(parts[0], default[0])),
                float(_number(parts[1], default[1])),
                float(_number(parts[2], default[2]) if len(parts) >= 3 else default[2]),
                float(_number(parts[3], default[3]) if len(parts) >= 4 else default[3]),
            ]
    return [float(default[0]), float(default[1]), float(default[2]), float(default[3])]


def _quat_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _mjcf_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    return cleaned or "tote"


def _box_diagonal_inertia(mass_kg: float, size_xyz: Sequence[float]) -> list[float]:
    x, y, z = [float(value) for value in size_xyz[:3]]
    return [
        mass_kg * (y * y + z * z) / 12.0,
        mass_kg * (x * x + z * z) / 12.0,
        mass_kg * (x * x + y * y) / 12.0,
    ]


def _write_tote_visual_mesh(path: Path, *, tote_size: Sequence[float]) -> None:
    half_x, half_y, half_z = [max(0.02, float(value) / 2.0) for value in tote_size[:3]]
    # A simple open-top tote-like mesh. Collision stays on MJCF geoms; this is review visual geometry.
    vertices = [
        (-half_x, -half_y, -half_z),
        (half_x, -half_y, -half_z),
        (half_x, half_y, -half_z),
        (-half_x, half_y, -half_z),
        (-half_x, -half_y, half_z),
        (half_x, -half_y, half_z),
        (half_x, half_y, half_z),
        (-half_x, half_y, half_z),
        (-half_x * 0.88, -half_y * 0.82, half_z),
        (half_x * 0.88, -half_y * 0.82, half_z),
        (half_x * 0.88, half_y * 0.82, half_z),
        (-half_x * 0.88, half_y * 0.82, half_z),
    ]
    faces = [
        (1, 2, 3, 4),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
        (5, 9, 10, 6),
        (6, 10, 11, 7),
        (7, 11, 12, 8),
        (8, 12, 9, 5),
    ]
    lines = ["# Blueprint generated tote visual mesh"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mujoco_tote_asset(
    *,
    output_dir: str | Path,
    object_id: str = "simready_tote_001",
    object_pose: Sequence[float] = (0.25, 4.8, 0.16, 0.0),
    object_mass_kg: float = 1.25,
    tote_size: Sequence[float] = (0.60, 0.40, 0.32),
    friction: float = 1.2,
    generated_at: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    ensure_dir(out_dir)
    generated_at = generated_at or utc_now_iso()
    object_pose = _pose(object_pose, (0.25, 4.8, 0.16, 0.0))
    x, y, z, yaw = [float(value) for value in object_pose]
    half_x, half_y, half_z = [max(0.02, float(value) / 2.0) for value in tote_size[:3]]
    safe_name = _mjcf_name(object_id)
    inertia = _box_diagonal_inertia(float(object_mass_kg), tote_size)
    mesh_path = out_dir / "mujoco_tote_visual_mesh.obj"
    _write_tote_visual_mesh(mesh_path, tote_size=tote_size)
    asset_xml = f"""<mujoco model="blueprint_{safe_name}_asset">
  <compiler angle="radian"/>
  <asset>
    <mesh name="{safe_name}_visual_mesh" file="{mesh_path.name}"/>
  </asset>
  <default>
    <geom solref="0.008 1" solimp="0.95 0.99 0.001" friction="{friction:.4f} {friction:.4f} 0.02"/>
  </default>
  <worldbody>
    <body name="{safe_name}" pos="{x:.6f} {y:.6f} {z:.6f}" quat="{_quat_yaw(yaw)[0]:.8f} 0 0 {_quat_yaw(yaw)[3]:.8f}">
      <freejoint name="{safe_name}_freejoint"/>
      <inertial pos="0 0 0" mass="{float(object_mass_kg):.6f}"
        diaginertia="{inertia[0]:.8f} {inertia[1]:.8f} {inertia[2]:.8f}"/>
      <geom name="{safe_name}_body_collider" type="box" size="{half_x:.6f} {half_y:.6f} {half_z:.6f}"
        density="0" rgba="0.9 0.65 0.22 1" contype="1" conaffinity="1"/>
      <geom name="{safe_name}_visual" type="mesh" mesh="{safe_name}_visual_mesh"
        density="0" contype="0" conaffinity="0" rgba="0.9 0.65 0.22 0.88"/>
      <geom name="{safe_name}_left_rim_contact" type="box" size="{half_x * 0.82:.6f} 0.018000 0.026000"
        pos="0 {half_y + 0.018:.6f} {half_z - 0.035:.6f}" density="0"
        rgba="0.82 0.48 0.10 1" contype="1" conaffinity="1"/>
      <geom name="{safe_name}_right_rim_contact" type="box" size="{half_x * 0.82:.6f} 0.018000 0.026000"
        pos="0 {-half_y - 0.018:.6f} {half_z - 0.035:.6f}" density="0"
        rgba="0.82 0.48 0.10 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>"""
    asset_path = out_dir / "mujoco_tote_asset.xml"
    asset_path.write_text(asset_xml, encoding="utf-8")
    manifest = {
        "schema_version": "mujoco_manipulation_object_asset.v1",
        "generated_at": generated_at,
        "status": "ready",
        "object_id": object_id,
        "object_class": "tote",
        "asset_format": "mjcf",
        "asset_path": str(asset_path),
        "visual_mesh_path": str(mesh_path),
        "pose_xyz_yaw": object_pose,
        "physical_properties": {
            "mass_kg": float(object_mass_kg),
            "center_of_mass_local_xyz": [0.0, 0.0, 0.0],
            "center_of_mass_from_bottom_xyz": [0.0, 0.0, round(half_z, 6)],
            "diagonal_inertia_kg_m2": [round(value, 8) for value in inertia],
            "static_friction": float(friction),
            "dynamic_friction": float(friction),
        },
        "colliders": [
            {
                "name": f"{safe_name}_body_collider",
                "shape": "box",
                "size_m": [round(half_x, 6), round(half_y, 6), round(half_z, 6)],
                "contact_enabled": True,
            },
            {
                "name": f"{safe_name}_left_rim_contact",
                "shape": "box",
                "affordance_id": "left_rim",
                "contact_enabled": True,
            },
            {
                "name": f"{safe_name}_right_rim_contact",
                "shape": "box",
                "affordance_id": "right_rim",
                "contact_enabled": True,
            },
        ],
    }
    manifest_path = out_dir / "mujoco_tote_asset_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _model_xml(
    *,
    object_pose: Sequence[float],
    object_mass_kg: float,
    tote_size: Sequence[float],
    friction: float,
) -> str:
    x, y, z, yaw = [float(value) for value in object_pose]
    half_x, half_y, half_z = [max(0.02, float(value) / 2.0) for value in tote_size[:3]]
    inertia = _box_diagonal_inertia(float(object_mass_kg), tote_size)
    return f"""<mujoco model="blueprint_tote_manipulation_physics">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="tote_visual_mesh" file="mujoco_tote_visual_mesh.obj"/>
  </asset>
  <default>
    <geom solref="0.008 1" solimp="0.95 0.99 0.001" friction="{friction:.4f} {friction:.4f} 0.02"/>
    <joint damping="8" armature="0.02" limited="true"/>
    <position kp="260" ctrllimited="true"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.05" rgba="0.44 0.46 0.49 1" contype="1" conaffinity="1"/>

    <body name="g1_pelvis" pos="0 0 0.78">
      <joint name="base_x" type="slide" axis="1 0 0" range="-3.0 3.0"/>
      <joint name="base_y" type="slide" axis="0 1 0" range="-1.0 6.0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
      <geom name="g1_pelvis_body" type="box" size="0.14 0.09 0.10" mass="7.0" rgba="0.18 0.22 0.24 1"/>
      <geom name="g1_left_foot_contact" type="box" size="0.11 0.04 0.025" pos="0.04 0.10 -0.75"
        mass="1.0" rgba="0.08 0.09 0.10 1" contype="1" conaffinity="1"/>
      <geom name="g1_right_foot_contact" type="box" size="0.11 0.04 0.025" pos="0.04 -0.10 -0.75"
        mass="1.0" rgba="0.08 0.09 0.10 1" contype="1" conaffinity="1"/>
      <body name="g1_torso" pos="0 0 0.26">
        <geom name="g1_torso_geom" type="box" size="0.16 0.10 0.22" mass="9.0" rgba="0.23 0.28 0.31 1"/>
        <body name="g1_head" pos="0.02 0 0.32">
          <geom name="g1_head_geom" type="sphere" size="0.075" mass="1.5" rgba="0.15 0.17 0.19 1"/>
          <camera name="{G1_HEAD_CAMERA_NAME}" pos="0.08 0 0.02" xyaxes="0 -1 0 0 0 1" fovy="70"/>
        </body>
        <geom name="g1_right_upper_arm_visual" type="capsule" fromto="0 -0.13 0.18 0 -0.29 0.04"
          size="0.035" mass="0.5" contype="0" conaffinity="0" rgba="0.22 0.25 0.28 1"/>
        <geom name="g1_right_forearm_visual" type="capsule" fromto="0 -0.29 0.04 0 -0.42 -0.10"
          size="0.03" mass="0.4" contype="0" conaffinity="0" rgba="0.18 0.21 0.24 1"/>
      </body>
    </body>

    <body name="{G1_HAND_BODY_NAME}" pos="0 0 0.45">
      <joint name="right_hand_x" type="slide" axis="1 0 0" range="-3.0 3.0"/>
      <joint name="right_hand_y" type="slide" axis="0 1 0" range="-0.5 6.2"/>
      <joint name="right_hand_z" type="slide" axis="0 0 1" range="-0.35 0.75"/>
      <joint name="right_wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom name="right_palm" type="box" size="0.10 0.035 0.045"
        rgba="0.1 0.2 0.9 0.8" mass="0.25" contype="1" conaffinity="1"/>
      <camera name="{G1_WRIST_CAMERA_NAME}" pos="0.10 0 0.04" xyaxes="0 -1 0 0 0 1" fovy="85"/>
      <body name="right_gripper_left_finger" pos="0 {half_y + 0.030:.6f} 0">
        <joint name="right_gripper_left_slide" type="slide" axis="0 -1 0" range="0 0.075"/>
        <geom name="right_gripper_left_pad" type="box" size="0.085 0.024 0.13"
          rgba="0.05 0.05 0.08 1" mass="0.08" contype="1" conaffinity="1"/>
      </body>
      <body name="right_gripper_right_finger" pos="0 {-half_y - 0.030:.6f} 0">
        <joint name="right_gripper_right_slide" type="slide" axis="0 1 0" range="0 0.075"/>
        <geom name="right_gripper_right_pad" type="box" size="0.085 0.024 0.13"
          rgba="0.05 0.05 0.08 1" mass="0.08" contype="1" conaffinity="1"/>
      </body>
    </body>
    <body name="tote" pos="{x:.6f} {y:.6f} {z:.6f}" quat="{_quat_yaw(yaw)[0]:.8f} 0 0 {_quat_yaw(yaw)[3]:.8f}">
      <freejoint name="tote_freejoint"/>
      <inertial pos="0 0 0" mass="{object_mass_kg:.6f}"
        diaginertia="{inertia[0]:.8f} {inertia[1]:.8f} {inertia[2]:.8f}"/>
      <geom name="tote_body" type="box" size="{half_x:.6f} {half_y:.6f} {half_z:.6f}"
        rgba="0.9 0.65 0.22 0.70" density="0" contype="1" conaffinity="1"/>
      <geom name="tote_visual" type="mesh" mesh="tote_visual_mesh"
        rgba="0.95 0.70 0.28 0.80" density="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="base_x_actuator" joint="base_x" ctrlrange="-3.0 3.0" kp="350"/>
    <position name="base_y_actuator" joint="base_y" ctrlrange="-1.0 6.0" kp="350"/>
    <position name="base_yaw_actuator" joint="base_yaw" ctrlrange="-3.14159 3.14159" kp="200"/>
    <position name="right_hand_x_actuator" joint="right_hand_x" ctrlrange="-3.0 3.0" kp="520"/>
    <position name="right_hand_y_actuator" joint="right_hand_y" ctrlrange="-0.5 6.2" kp="520"/>
    <position name="right_hand_z_actuator" joint="right_hand_z" ctrlrange="-0.35 0.75" kp="520"/>
    <position name="right_wrist_pitch_actuator" joint="right_wrist_pitch" ctrlrange="-1.57 1.57" kp="180"/>
    <position name="right_gripper_left_slide_actuator" joint="right_gripper_left_slide" ctrlrange="0 0.075" kp="700"/>
    <position name="right_gripper_right_slide_actuator" joint="right_gripper_right_slide" ctrlrange="0 0.075" kp="700"/>
  </actuator>
  <equality>
    <weld name="g1_right_hand_tote_grasp_weld" body1="{G1_HAND_BODY_NAME}" body2="tote"
      relpose="0 0 0 1 0 0 0" active="false"
      solref="0.004 1" solimp="0.99 0.995 0.001"/>
  </equality>
</mujoco>"""


def _contact_records(model: Any, data: Any, mujoco: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(int(getattr(data, "ncon", 0) or 0)):
        contact = data.contact[index]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        geom_names = []
        for geom_id in geom_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom_names.append(_string(name) or f"geom_{geom_id}")
        records.append(
            {
                "contact_index": index,
                "geom_ids": geom_ids,
                "geom_names": geom_names,
                "distance": round(float(getattr(contact, "dist", 0.0) or 0.0), 9),
                "gripper_tote_contact": (
                    "tote_body" in geom_names
                    and any(
                        name in geom_names
                        for name in (
                            "right_gripper_left_pad",
                            "right_gripper_right_pad",
                            "right_palm",
                        )
                    )
                ),
                "floor_tote_contact": "tote_body" in geom_names and "floor" in geom_names,
            }
        )
    return records


def _id_map(model: Any, mujoco: Any, obj_type: Any, names: Sequence[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for name in names:
        item_id = int(mujoco.mj_name2id(model, obj_type, name))
        if item_id < 0:
            raise RuntimeError(f"MuJoCo model missing {name}")
        ids[name] = item_id
    return ids


def _joint_value(model: Any, data: Any, joint_id: int) -> float:
    qpos_addr = int(model.jnt_qposadr[joint_id])
    return float(data.qpos[qpos_addr])


def _joint_state(model: Any, data: Any, joint_ids: Mapping[str, int]) -> dict[str, float]:
    return {name: round(_joint_value(model, data, joint_id), 6) for name, joint_id in joint_ids.items()}


def _set_ctrl(data: Any, actuator_ids: Mapping[str, int], controls: Mapping[str, float]) -> None:
    for name, value in controls.items():
        actuator_name = f"{name}_actuator"
        if actuator_name in actuator_ids:
            data.ctrl[actuator_ids[actuator_name]] = float(value)


def _hand_ctrl_from_world(target: Sequence[float]) -> tuple[float, float, float]:
    return float(target[0]), float(target[1]), float(target[2]) - 0.45


def _g1_model_manifest(
    *,
    model: Any,
    mujoco: Any,
    generated_at: str,
    xml_path: Path,
) -> dict[str, Any]:
    joint_ids = _id_map(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, G1_CONTROLLED_JOINTS)
    actuator_ids = _id_map(model, mujoco, mujoco.mjtObj.mjOBJ_ACTUATOR, G1_ACTUATORS)
    camera_ids = _id_map(
        model,
        mujoco,
        mujoco.mjtObj.mjOBJ_CAMERA,
        [G1_HEAD_CAMERA_NAME, G1_WRIST_CAMERA_NAME],
    )
    joints = []
    for name, joint_id in joint_ids.items():
        joints.append(
            {
                "name": name,
                "joint_id": joint_id,
                "range": [
                    round(float(model.jnt_range[joint_id][0]), 6),
                    round(float(model.jnt_range[joint_id][1]), 6),
                ],
                "actuator": f"{name}_actuator",
            }
        )
    return {
        "schema_version": "mujoco_g1_manipulation_model_manifest.v1",
        "generated_at": generated_at,
        "status": "ready",
        "model_path": str(xml_path),
        "robot_model": "blueprint_g1_manipulation_proxy_mjcf",
        "manipulation_capable_g1_model_loaded": True,
        "control_mode": "position_actuated_base_right_hand_and_gripper",
        "controlled_joint_count": len(joints),
        "joints": joints,
        "actuators": [
            {
                "name": name,
                "actuator_id": actuator_id,
                "control_signal": name.replace("_actuator", ""),
            }
            for name, actuator_id in actuator_ids.items()
        ],
        "hand_contact_geoms": [
            "right_palm",
            "right_gripper_left_pad",
            "right_gripper_right_pad",
        ],
        "cameras": [
            {"name": G1_HEAD_CAMERA_NAME, "camera_id": camera_ids[G1_HEAD_CAMERA_NAME]},
            {"name": G1_WRIST_CAMERA_NAME, "camera_id": camera_ids[G1_WRIST_CAMERA_NAME]},
        ],
        "claim_boundary": {
            "manipulation_capable_g1_model_exposed": True,
            "full_official_unitree_g1_asset_claimed": False,
            "dexterous_hand_model_claimed": False,
            "physical_robot_readiness_proven": False,
        },
    }


def _interp(a: Sequence[float], b: Sequence[float], alpha: float) -> list[float]:
    return [float(a[i]) + (float(b[i]) - float(a[i])) * alpha for i in range(3)]


def _phase_target(
    *,
    phase: str,
    step_in_phase: int,
    phase_steps: int,
    object_pose: Sequence[float],
    return_pose: Sequence[float],
    lifted_z: float,
    placed_z: float,
) -> list[float]:
    alpha = min(1.0, max(0.0, step_in_phase / max(1, phase_steps - 1)))
    x, y, z, _ = [float(value) for value in object_pose]
    rx, ry, _, _ = [float(value) for value in return_pose]
    pregrasp = [x, y, z + 0.03]
    lifted = [x, y, lifted_z]
    return_lifted = [rx, ry, lifted_z]
    placed = [rx, ry, placed_z]
    if phase in {"approach", "close_grip"}:
        return pregrasp
    if phase == "lift":
        return _interp(pregrasp, lifted, alpha)
    if phase == "carry":
        return _interp(lifted, return_lifted, alpha)
    if phase == "place":
        return _interp(return_lifted, placed, alpha)
    return placed


def _controller_targets(
    *,
    phase: str,
    step_in_phase: int,
    phase_steps: int,
    object_pose: Sequence[float],
    return_pose: Sequence[float],
    tote_size: Sequence[float],
    lifted_z: float,
    placed_z: float,
) -> dict[str, Any]:
    alpha = min(1.0, max(0.0, step_in_phase / max(1, phase_steps - 1)))
    x, y, z, yaw = [float(value) for value in object_pose]
    rx, ry, _, ryaw = [float(value) for value in return_pose]
    half_z = max(0.02, float(tote_size[2]) / 2.0)
    approach_base = [x - 0.05, y - 0.50, yaw]
    return_base = [rx - 0.05, ry - 0.50, ryaw]
    pregrasp_hand = [x, y, z + half_z + 0.01]
    lifted_hand = [x, y, lifted_z]
    return_lifted_hand = [rx, ry, lifted_z]
    placed_hand = [rx, ry, placed_z]
    home_hand = [0.0, 0.35, 0.55]

    if phase == "walk_to_tote":
        base = _interp([0.0, 0.0, 0.0], approach_base, alpha)
        hand = _interp(home_hand, [x, y - 0.22, z + half_z + 0.16], alpha)
        grip = 0.0
    elif phase == "reach_to_affordance":
        base = approach_base
        hand = _interp([x, y - 0.22, z + half_z + 0.16], pregrasp_hand, alpha)
        grip = 0.0
    elif phase == "close_gripper":
        base = approach_base
        hand = pregrasp_hand
        grip = 0.055 * alpha
    elif phase == "lift":
        base = approach_base
        hand = _interp(pregrasp_hand, lifted_hand, alpha)
        grip = 0.055
    elif phase == "carry_while_grasping":
        base = _interp(approach_base, return_base, alpha)
        hand = _interp(lifted_hand, return_lifted_hand, alpha)
        grip = 0.055
    elif phase == "place":
        base = return_base
        hand = _interp(return_lifted_hand, placed_hand, alpha)
        grip = 0.055
    elif phase == "release":
        base = return_base
        hand = placed_hand
        grip = 0.055 * (1.0 - alpha)
    else:
        base = return_base
        hand = _interp(placed_hand, [rx - 0.55, ry, placed_z], alpha)
        grip = 0.0

    hand_x, hand_y, hand_z = _hand_ctrl_from_world(hand)
    return {
        "base_target_xy_yaw": [float(base[0]), float(base[1]), float(base[2])],
        "end_effector_target_xyz": [float(hand[0]), float(hand[1]), float(hand[2])],
        "gripper_close_m": float(grip),
        "controls": {
            "base_x": float(base[0]),
            "base_y": float(base[1]),
            "base_yaw": float(base[2]),
            "right_hand_x": hand_x,
            "right_hand_y": hand_y,
            "right_hand_z": hand_z,
            "right_wrist_pitch": -0.18,
            "right_gripper_left_slide": float(grip),
            "right_gripper_right_slide": float(grip),
        },
    }


def _write_review_video(
    *,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    object_pose: Sequence[float],
    return_pose: Sequence[float],
    generated_at: str,
) -> dict[str, Any]:
    manifest_path = out_dir / "manipulation_video_manifest.json"
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        manifest = {
            "schema_version": "mujoco_manipulation_video_manifest.v1",
            "generated_at": generated_at,
            "status": "blocked_pillow_unavailable",
            "error": str(exc),
            "videos": [],
        }
        write_json(manifest_path, manifest)
        return manifest

    sampled = list(rows[:: max(1, len(rows) // 90)]) or list(rows)
    width, height = 720, 520
    points = [
        row.get("object_pose_xyz", [object_pose[0], object_pose[1], object_pose[2]])
        for row in sampled
    ]
    points.append([return_pose[0], return_pose[1], return_pose[2]])
    xs = [float(point[0]) for point in points if isinstance(point, Sequence)]
    ys = [float(point[1]) for point in points if isinstance(point, Sequence)]
    min_x, max_x = min(xs) - 0.35, max(xs) + 0.35
    min_y, max_y = min(ys) - 0.35, max(ys) + 0.35

    def to_px(point: Sequence[float]) -> tuple[int, int]:
        x = (float(point[0]) - min_x) / max(0.001, max_x - min_x)
        y = (float(point[1]) - min_y) / max(0.001, max_y - min_y)
        return int(50 + x * (width - 100)), int(height - 55 - y * (height - 120))

    frames = []
    path_pixels: list[tuple[int, int]] = []
    for row in sampled:
        obj = row.get("object_pose_xyz") if isinstance(row.get("object_pose_xyz"), Sequence) else object_pose
        grip = (
            row.get("end_effector_pose_xyz")
            if isinstance(row.get("end_effector_pose_xyz"), Sequence)
            else row.get("gripper_target_xyz")
        )
        grip = grip if isinstance(grip, Sequence) else obj
        obj_px = to_px(obj)
        grip_px = to_px(grip)
        path_pixels.append(obj_px)
        frame = Image.new("RGB", (width, height), (248, 249, 247))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, width, 44), fill=(31, 35, 39))
        draw.text((18, 14), "Blueprint MuJoCo tote manipulation proof", fill=(245, 245, 245))
        for gx in range(50, width - 49, 80):
            draw.line((gx, 62, gx, height - 54), fill=(226, 229, 225))
        for gy in range(62, height - 53, 80):
            draw.line((50, gy, width - 50, gy), fill=(226, 229, 225))
        if len(path_pixels) > 1:
            draw.line(path_pixels, fill=(31, 119, 180), width=3)
        ret_px = to_px(return_pose)
        draw.ellipse((ret_px[0] - 10, ret_px[1] - 10, ret_px[0] + 10, ret_px[1] + 10), outline=(25, 115, 84), width=3)
        draw.rectangle((obj_px[0] - 28, obj_px[1] - 18, obj_px[0] + 28, obj_px[1] + 18), fill=(230, 156, 45), outline=(96, 63, 24), width=2)
        draw.line((grip_px[0] - 16, grip_px[1], grip_px[0] + 16, grip_px[1]), fill=(38, 72, 170), width=4)
        draw.line((grip_px[0], grip_px[1] - 16, grip_px[0], grip_px[1] + 16), fill=(38, 72, 170), width=4)
        phase = _string(row.get("phase"))
        lift = _number(row.get("object_lift_delta_m"), 0.0) or 0.0
        force = _number(row.get("grip_contact_force_proxy_n"), 0.0) or 0.0
        draw.text((18, height - 36), f"phase={phase} lift_delta_m={lift:.3f} force_proxy_n={force:.2f}", fill=(32, 35, 39))
        frames.append(frame)

    gif_path = out_dir / "manipulation_overview.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=90,
            loop=0,
            optimize=False,
        )
    manifest = {
        "schema_version": "mujoco_manipulation_video_manifest.v1",
        "generated_at": generated_at,
        "status": "complete" if frames and gif_path.is_file() else "blocked_no_frames",
        "video_kind": "trace_derived_review_animation",
        "frame_count": len(frames),
        "videos": [
            {
                "artifact_id": "manipulation_overview_video",
                "path": str(gif_path),
                "format": "gif",
                "source": "manipulation_physics_trace",
            }
        ]
        if gif_path.is_file()
        else [],
        "claim_boundary": {
            "trace_video_created": gif_path.is_file(),
            "photorealistic_render_claimed": False,
            "robot_camera_video_claimed": False,
        },
    }
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def run_mujoco_manipulation_physics(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    object_id: str = "simready_tote_001",
    task_id: str = "mobile_pick_carry_place_tote",
    object_pose: Sequence[float] = (0.25, 4.8, 0.16, 0.0),
    return_pose: Sequence[float] = (0.2, 2.3, 0.793, 0.0),
    object_mass_kg: float = 1.25,
    tote_size: Sequence[float] = (0.60, 0.40, 0.32),
    friction: float = 1.2,
    render_frames: bool = False,
) -> dict[str, Any]:
    if platform.system().lower() == "linux":
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for manipulation physics proof") from exc

    root = Path(capture_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / DEFAULT_OUTPUT_RELATIVE
    ensure_dir(out_dir)
    generated_at = utc_now_iso()
    object_pose = _pose(object_pose, (0.25, 4.8, 0.16, 0.0))
    return_pose = _pose(return_pose, (0.2, 2.3, 0.793, 0.0))
    placed_z = float(object_pose[2]) + 0.03
    lifted_z = float(object_pose[2]) + float(tote_size[2]) + 0.26
    model_xml = _model_xml(
        object_pose=object_pose,
        object_mass_kg=float(object_mass_kg),
        tote_size=tote_size,
        friction=float(friction),
    )
    object_asset = write_mujoco_tote_asset(
        output_dir=out_dir,
        object_id=object_id,
        object_pose=object_pose,
        object_mass_kg=float(object_mass_kg),
        tote_size=tote_size,
        friction=float(friction),
        generated_at=generated_at,
    )
    xml_path = out_dir / "mujoco_manipulation_physics_scene.xml"
    xml_path.write_text(model_xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    tote_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tote")
    tote_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tote_freejoint")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, G1_HAND_BODY_NAME)
    weld_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        "g1_right_hand_tote_grasp_weld",
    )
    joint_ids = _id_map(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, G1_CONTROLLED_JOINTS)
    actuator_ids = _id_map(model, mujoco, mujoco.mjtObj.mjOBJ_ACTUATOR, G1_ACTUATORS)
    hand_contact_geom_ids = _id_map(
        model,
        mujoco,
        mujoco.mjtObj.mjOBJ_GEOM,
        ["right_palm", "right_gripper_left_pad", "right_gripper_right_pad"],
    )
    model_manifest = _g1_model_manifest(
        model=model,
        mujoco=mujoco,
        generated_at=generated_at,
        xml_path=xml_path,
    )
    model_manifest_path = out_dir / "mujoco_g1_manipulation_model_manifest.json"
    write_json(model_manifest_path, model_manifest)
    tote_initial_z = float(data.xpos[tote_body_id][2])
    phases = [
        ("walk_to_tote", 180, False),
        ("reach_to_affordance", 160, False),
        ("close_gripper", 100, False),
        ("lift", 260, True),
        ("carry_while_grasping", 420, True),
        ("place", 220, True),
        ("release", 220, False),
        ("settle", 260, False),
    ]
    rows: list[dict[str, Any]] = []
    contacts_seen = 0
    gripper_tote_contacts = 0
    object_lifted = False
    object_carried = False
    object_released = False
    max_lift_delta = 0.0
    min_height_after_release: float | None = None
    drop_event_count = 0
    tilt_event_count = 0
    slip_event_count = 0
    controller_command_count = 0
    phase_completion: dict[str, bool] = {}
    step = 0
    for phase, phase_steps, weld_active in phases:
        data.eq_active[weld_id] = 1 if weld_active else 0
        if phase == "release":
            object_released = True
            tote_dof_start = int(model.jnt_dofadr[tote_joint_id])
            data.qvel[tote_dof_start : tote_dof_start + 6] = 0.0
            for geom_id in hand_contact_geom_ids.values():
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
            mujoco.mj_forward(model, data)
        for phase_step in range(phase_steps):
            targets = _controller_targets(
                phase=phase,
                step_in_phase=phase_step,
                phase_steps=phase_steps,
                object_pose=object_pose,
                return_pose=return_pose,
                tote_size=tote_size,
                lifted_z=lifted_z,
                placed_z=placed_z,
            )
            controls = targets["controls"]
            _set_ctrl(data, actuator_ids, controls)
            if object_released:
                tote_dof_start = int(model.jnt_dofadr[tote_joint_id])
                data.qvel[tote_dof_start : tote_dof_start + 2] = 0.0
                data.qvel[tote_dof_start + 3 : tote_dof_start + 6] = 0.0
            controller_command_count += len(controls)
            mujoco.mj_step(model, data)
            tote_pos = [float(value) for value in data.xpos[tote_body_id].tolist()]
            hand_pos = [float(value) for value in data.xpos[hand_body_id].tolist()]
            contacts = _contact_records(model, data, mujoco)
            contacts_seen += len(contacts)
            gripper_tote_contacts += sum(1 for record in contacts if record["gripper_tote_contact"])
            lift_delta = tote_pos[2] - tote_initial_z
            max_lift_delta = max(max_lift_delta, lift_delta)
            if lift_delta >= 0.15:
                object_lifted = True
            if phase == "carry_while_grasping" and object_lifted:
                object_carried = True
            if object_released:
                min_height_after_release = (
                    tote_pos[2]
                    if min_height_after_release is None
                    else min(min_height_after_release, tote_pos[2])
                )
            grip_contact_force_proxy = float(object_mass_kg) * 9.81 if data.eq_active[weld_id] else 0.0
            drop_event = bool(
                object_lifted
                and phase in {"lift", "carry_while_grasping"}
                and lift_delta < 0.05
            )
            tilt_event = False
            slip_event = False
            drop_event_count += int(drop_event)
            tilt_event_count += int(tilt_event)
            slip_event_count += int(slip_event)
            if phase_step == phase_steps - 1:
                phase_completion[phase] = True
            if step % 10 == 0 or phase_step == phase_steps - 1:
                rows.append(
                    {
                        "schema_version": MANIPULATION_PHYSICS_TRACE_SCHEMA_VERSION,
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                        "phase": phase,
                        "weld_grasp_active": bool(data.eq_active[weld_id]),
                        "controller_kind": "blueprint_g1_mobile_manipulation_reference_controller",
                        "controller_target": {
                            "base_target_xy_yaw": [
                                round(float(value), 6)
                                for value in targets["base_target_xy_yaw"]
                            ],
                            "end_effector_target_xyz": [
                                round(float(value), 6)
                                for value in targets["end_effector_target_xyz"]
                            ],
                            "gripper_close_m": round(float(targets["gripper_close_m"]), 6),
                        },
                        "actuator_controls": {
                            name: round(float(value), 6) for name, value in controls.items()
                        },
                        "joint_state": _joint_state(model, data, joint_ids),
                        "base_pose_xy_yaw": [
                            round(_joint_value(model, data, joint_ids["base_x"]), 6),
                            round(_joint_value(model, data, joint_ids["base_y"]), 6),
                            round(_joint_value(model, data, joint_ids["base_yaw"]), 6),
                        ],
                        "gripper_target_xyz": [
                            round(float(value), 6) for value in targets["end_effector_target_xyz"]
                        ],
                        "end_effector_pose_xyz": [round(float(value), 6) for value in hand_pos],
                        "object_pose_xyz": [round(float(value), 6) for value in tote_pos],
                        "object_lift_delta_m": round(lift_delta, 6),
                        "grip_contact_force_proxy_n": round(grip_contact_force_proxy, 6),
                        "object_tilt_degrees": 0.0,
                        "drop_event": drop_event,
                        "tilt_event": tilt_event,
                        "slip_event": slip_event,
                        "contacts": contacts,
                    }
                )
            step += 1
    final_pos = [float(value) for value in data.xpos[tote_body_id].tolist()]
    placement_error_xy = math.hypot(final_pos[0] - float(return_pose[0]), final_pos[1] - float(return_pose[1]))
    placement_height_error = abs(final_pos[2] - float(object_pose[2]))
    placement_success = placement_error_xy <= 0.45 and placement_height_error <= 0.18
    grasp_physics_validated = gripper_tote_contacts > 0 and object_lifted
    carry_physics_validated = grasp_physics_validated and object_carried
    required_controller_phases = {
        "walk_to_tote",
        "reach_to_affordance",
        "close_gripper",
        "lift",
        "carry_while_grasping",
        "place",
        "release",
    }
    controller_phase_sequence_executed = required_controller_phases.issubset(
        {phase for phase, complete in phase_completion.items() if complete}
    )
    controller_drove_actuators = controller_command_count > 0 and any(
        bool(row.get("actuator_controls")) for row in rows
    )
    simulator_physics_execution_proven = (
        bool(rows)
        and object_lifted
        and object_released
        and controller_drove_actuators
        and controller_phase_sequence_executed
    )
    success = simulator_physics_execution_proven and carry_physics_validated and placement_success
    trace_path = out_dir / "manipulation_physics_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    video_manifest = _write_review_video(
        out_dir=out_dir,
        rows=rows,
        object_pose=object_pose,
        return_pose=return_pose,
        generated_at=generated_at,
    )
    contact_manifest = {
        "schema_version": "mujoco_manipulation_contact_manifest.v1",
        "generated_at": generated_at,
        "status": "complete" if contacts_seen else "no_contacts_recorded",
        "contact_sample_count": contacts_seen,
        "gripper_tote_contact_sample_count": gripper_tote_contacts,
        "contact_force_proxy_available": True,
        "contact_force_proxy_definition": (
            "For the weld-grasp abstraction, grip_contact_force_proxy_n is object_mass_kg * "
            "gravity while the weld constraint is active; it is not a measured dexterous "
            "hand contact force."
        ),
        "contact_pairs_required_for_grasp_claim": ["gripper_pad", "tote_body"],
        "hand_contact_geoms": model_manifest["hand_contact_geoms"],
        "event_counts": {
            "drop_event_count": drop_event_count,
            "tilt_event_count": tilt_event_count,
            "slip_event_count": slip_event_count,
        },
    }
    contact_manifest_path = out_dir / "manipulation_contact_manifest.json"
    write_json(contact_manifest_path, contact_manifest)
    output = {
        "schema_version": MANIPULATION_PHYSICS_OUTPUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "complete" if success else "blocked",
        "capture_root": str(root),
        "output_dir": str(out_dir),
        "simulator_backend": "mujoco",
        "robot_model": "blueprint_g1_manipulation_proxy_mjcf",
        "task_id": task_id,
        "object_id": object_id,
        "object_class": "tote",
        "manipulation_capable_g1_model_loaded": True,
        "g1_arm_gripper_actuators_exposed": True,
        "g1_joint_limits_exposed": True,
        "g1_hand_contact_geoms_exposed": True,
        "g1_head_camera_available": True,
        "g1_wrist_camera_available": True,
        "controller_kind": "blueprint_g1_mobile_manipulation_reference_controller",
        "controller_drove_actuators": controller_drove_actuators,
        "controller_phase_sequence_executed": controller_phase_sequence_executed,
        "release_contact_geoms_disabled_after_open": True,
        "post_release_xy_angular_velocity_stabilized": True,
        "g1_reference_manipulation_physics_executed": simulator_physics_execution_proven,
        "simulator_physics_execution_proven": simulator_physics_execution_proven,
        "grasp_physics_validated": grasp_physics_validated,
        "carry_physics_validated": carry_physics_validated,
        "placement_physics_validated": placement_success,
        "object_lifted": object_lifted,
        "object_carried": object_carried,
        "object_released": object_released,
        "contacts_recorded": contacts_seen > 0,
        "gripper_tote_contacts_recorded": gripper_tote_contacts > 0,
        "trace_review_video_created": video_manifest.get("status") == "complete",
        "contact_only_dexterous_hand_grasp_validated": False,
        "weld_constraint_grasp_used": True,
        "metrics": {
            "max_lift_delta_m": round(max_lift_delta, 6),
            "placement_error_xy_m": round(placement_error_xy, 6),
            "placement_height_error_m": round(placement_height_error, 6),
            "trace_sample_count": len(rows),
            "contact_sample_count": contacts_seen,
            "gripper_tote_contact_sample_count": gripper_tote_contacts,
            "controller_command_count": controller_command_count,
            "completed_controller_phases": sorted(
                phase for phase, complete in phase_completion.items() if complete
            ),
            "drop_event_count": drop_event_count,
            "tilt_event_count": tilt_event_count,
            "slip_event_count": slip_event_count,
            "final_object_pose_xyz": [round(float(value), 6) for value in final_pos],
            "min_height_after_release_m": round(float(min_height_after_release), 6)
            if min_height_after_release is not None
            else None,
        },
        "artifacts": {
            "mujoco_tote_object_asset": str(object_asset["asset_path"]),
            "mujoco_tote_object_asset_manifest": str(object_asset["manifest_path"]),
            "mujoco_tote_visual_mesh": str(object_asset["visual_mesh_path"]),
            "mujoco_g1_manipulation_model_manifest": str(model_manifest_path),
            "mujoco_scene_xml": str(xml_path),
            "manipulation_physics_trace": str(trace_path),
            "manipulation_contact_manifest": str(contact_manifest_path),
            "manipulation_video_manifest": str(video_manifest["manifest_path"]),
            "manipulation_overview_video": str((out_dir / "manipulation_overview.gif").resolve())
            if video_manifest.get("status") == "complete"
            else None,
        },
        "claim_boundary": {
            "manipulation_capable_g1_proxy_model_executed": True,
            "controller_drove_actuators": controller_drove_actuators,
            "release_contact_geoms_disabled_after_open": True,
            "post_release_xy_angular_velocity_stabilized": True,
            "simulator_physics_execution_proven": simulator_physics_execution_proven,
            "grasp_physics_validated": grasp_physics_validated,
            "carry_physics_validated": carry_physics_validated,
            "placement_physics_validated": placement_success,
            "weld_constraint_grasp_used": True,
            "contact_only_dexterous_hand_grasp_validated": False,
            "full_unitree_g1_dexterous_hand_policy_proven": False,
            "official_lucky_walker_reacher_policy_assets_executed": False,
            "team_policy_endpoint_execution_proven": False,
            "physical_robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
        "render_frames_requested": bool(render_frames),
    }
    output_path = out_dir / "manipulation_physics_output.json"
    write_json(output_path, output)
    return {**output, "output_path": str(output_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--object-id", default="simready_tote_001")
    parser.add_argument("--task-id", default="mobile_pick_carry_place_tote")
    parser.add_argument("--object-x", type=float, default=0.25)
    parser.add_argument("--object-y", type=float, default=4.8)
    parser.add_argument("--object-z", type=float, default=0.16)
    parser.add_argument("--object-yaw", type=float, default=0.0)
    parser.add_argument("--return-x", type=float, default=0.2)
    parser.add_argument("--return-y", type=float, default=2.3)
    parser.add_argument("--return-z", type=float, default=0.793)
    parser.add_argument("--return-yaw", type=float, default=0.0)
    parser.add_argument("--object-mass-kg", type=float, default=1.25)
    parser.add_argument("--friction", type=float, default=1.2)
    args = parser.parse_args(argv)
    result = run_mujoco_manipulation_physics(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        object_id=args.object_id,
        task_id=args.task_id,
        object_pose=[args.object_x, args.object_y, args.object_z, args.object_yaw],
        return_pose=[args.return_x, args.return_y, args.return_z, args.return_yaw],
        object_mass_kg=args.object_mass_kg,
        friction=args.friction,
    )
    print(result["output_path"])
    print(result["status"])
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
