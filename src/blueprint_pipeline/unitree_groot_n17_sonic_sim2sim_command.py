"""Simulator-only GR00T N1.7 + UNITREE_G1_SONIC action chunk consumer.

This command consumes a real Blueprint GR00T/SONIC policy action artifact and
applies a bounded, normalized upper-body/hand target bridge to a MuJoCo Unitree
G1 scene. It is not an official GR00T-WholeBodyControl deployment path and it
never sends commands to physical robot hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import POLICY_ID


SCHEMA_VERSION = "unitree_groot_n17_sonic_sim2sim_execution.v1"
TRACE_SCHEMA_VERSION = "unitree_groot_n17_sonic_sim2sim_trace_row.v1"
PROJECTED_SKELETON_SCHEMA_VERSION = "blueprint.mujoco_g1.projected_upper_body_skeleton.v1"
PROJECTED_SKELETON_MANIFEST_SCHEMA_VERSION = "policy_action_projected_skeleton_trace_manifest.v1"
DEFAULT_STEPS = 40
DEFAULT_ACTION_HOLD_STEPS = 1
OBJECT_DISPLACEMENT_SUCCESS_M = 0.015
SONIC_ACTION_DIM = 78
FIXED_G1_CAMERA_NAMES = {
    "head_pov": "blueprint_g1_head_pov",
    "torso_pov": "blueprint_g1_torso_pov",
    "robot_pov": "blueprint_g1_head_pov",
}
G1_UPPER_BODY_LANDMARK_SPECS = (
    {"landmark_id": "left_shoulder", "body_name": "left_shoulder_pitch_link"},
    {"landmark_id": "left_elbow", "body_name": "left_elbow_link"},
    {"landmark_id": "left_wrist", "body_name": "left_wrist_yaw_link"},
    {
        "landmark_id": "left_hand",
        "body_name": "left_hand_palm_link",
        "fallback_body_name": "left_wrist_yaw_link",
        "fallback_local_offset_m": [0.082, 0.003, 0.0],
    },
    {"landmark_id": "right_shoulder", "body_name": "right_shoulder_pitch_link"},
    {"landmark_id": "right_elbow", "body_name": "right_elbow_link"},
    {"landmark_id": "right_wrist", "body_name": "right_wrist_yaw_link"},
    {
        "landmark_id": "right_hand",
        "body_name": "right_hand_palm_link",
        "fallback_body_name": "right_wrist_yaw_link",
        "fallback_local_offset_m": [0.082, -0.003, 0.0],
    },
)
G1_UPPER_BODY_SKELETON_SEGMENTS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)
UPPER_BODY_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
ACTION_VALUE_KEYS = (
    "action_chunk",
    "sonic_latent_action",
    "sonic_latents",
    "latent_action_tokens",
    "actions",
    "action_vector",
    "joint_targets",
    "joint_positions",
)
OBJECT_FREEJOINT_NAME = "blueprint_light_object_freejoint"
OBJECT_GEOM_NAME = "blueprint_light_object_geom"
OBJECT_BODY_NAME = "blueprint_light_object"
HAND_CONTACT_BODY_MARKERS = (
    "hand",
    "wrist",
    "thumb",
    "index",
    "middle",
    "finger",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _flatten_numbers(value: Any) -> list[float]:
    numbers: list[float] = []

    def visit(child: Any) -> None:
        if isinstance(child, Mapping):
            for item in child.values():
                visit(item)
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            for item in child:
                visit(item)
        elif isinstance(child, np.ndarray):
            visit(child.tolist())
        elif isinstance(child, np.generic):
            visit(child.item())
        elif isinstance(child, (int, float)) and not isinstance(child, bool):
            value = float(child)
            if math.isfinite(value):
                numbers.append(value)

    visit(value)
    return numbers


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def _scene_xml_from_job(job_dir: Path) -> Path | None:
    manifest_path = job_dir / "mujoco_scene_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return None
    scene_xml = str(manifest.get("scene_xml") or "").strip()
    return Path(scene_xml).expanduser() if scene_xml else None


def _object_initial_pose_from_job(job_dir: Path) -> list[float] | None:
    manifest_path = job_dir / "manipulation_scene_object_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return None
    raw_pose = manifest.get("initial_pose")
    if not isinstance(raw_pose, Sequence) or isinstance(raw_pose, (str, bytes, bytearray)):
        return None
    pose = [float(value) for value in raw_pose[:7]]
    return pose if len(pose) == 7 else None


def _extract_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    action = payload.get("action") or payload.get("normalized_action")
    return dict(action) if isinstance(action, Mapping) else dict(payload)


def _extract_action_vector(action: Mapping[str, Any]) -> tuple[str | None, list[float]]:
    for key in ACTION_VALUE_KEYS:
        values = _flatten_numbers(action.get(key))
        if values:
            return key, values
    return None, []


def _chunk_action_frames(values: Sequence[float]) -> tuple[int, list[list[float]]]:
    if not values:
        return 0, []
    if len(values) >= SONIC_ACTION_DIM and len(values) % SONIC_ACTION_DIM == 0:
        frames = [
            [float(value) for value in values[index : index + SONIC_ACTION_DIM]]
            for index in range(0, len(values), SONIC_ACTION_DIM)
        ]
        return SONIC_ACTION_DIM, frames
    return len(values), [[float(value) for value in values]]


def _control_target_from_action_value(value: float, low: float, high: float) -> float:
    center = (float(low) + float(high)) / 2.0
    span = max(0.0, float(high) - float(low))
    target = center + math.tanh(float(value)) * span * 0.25
    return min(float(high), max(float(low), target))


def _actuator_bindings(model: Any, mujoco_module: Any) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for action_index, joint_name in enumerate(UPPER_BODY_JOINT_NAMES):
        actuator_id = int(
            mujoco_module.mj_name2id(
                model,
                mujoco_module.mjtObj.mjOBJ_ACTUATOR,
                joint_name,
            )
        )
        joint_id = int(
            mujoco_module.mj_name2id(
                model,
                mujoco_module.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
        )
        if actuator_id < 0 or joint_id < 0:
            continue
        ctrl_range = model.actuator_ctrlrange[actuator_id]
        qpos_addr = int(model.jnt_qposadr[joint_id])
        bindings.append(
            {
                "joint_name": joint_name,
                "actuator_id": actuator_id,
                "joint_id": joint_id,
                "qpos_addr": qpos_addr,
                "action_dim_index": action_index,
                "ctrl_range": [float(ctrl_range[0]), float(ctrl_range[1])],
            }
        )
    return bindings


def _initialize_joint_holds(model: Any, data: Any) -> None:
    for actuator_index in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator_index][0])
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_index] = data.qpos[qpos_addr]


def _freejoint_qpos_addr(model: Any, mujoco_module: Any, joint_name: str) -> int | None:
    joint_id = int(
        mujoco_module.mj_name2id(
            model,
            mujoco_module.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
    )
    if joint_id < 0:
        return None
    return int(model.jnt_qposadr[joint_id])


def _object_pose(data: Any, qpos_addr: int | None) -> dict[str, Any]:
    if qpos_addr is None:
        return {"available": False}
    return {
        "available": True,
        "position": [round(float(value), 6) for value in data.qpos[qpos_addr : qpos_addr + 3]],
        "quaternion_wxyz": [
            round(float(value), 6) for value in data.qpos[qpos_addr + 3 : qpos_addr + 7]
        ],
    }


def _body_name(model: Any, mujoco_module: Any, body_id: int) -> str:
    return (
        mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_BODY, int(body_id))
        or ""
    )


def _geom_body_name(model: Any, mujoco_module: Any, geom_id: int) -> str:
    if geom_id < 0:
        return ""
    body_id = int(model.geom_bodyid[int(geom_id)])
    return _body_name(model, mujoco_module, body_id)


def _object_contact_state(model: Any, data: Any, mujoco_module: Any) -> dict[str, Any]:
    object_robot_contact_count = 0
    object_any_contact_count = 0
    sampled_contacts: list[dict[str, Any]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        body_names = [
            _geom_body_name(model, mujoco_module, geom_id) for geom_id in geom_ids
        ]
        object_in_contact = OBJECT_BODY_NAME in body_names
        if not object_in_contact:
            continue
        object_any_contact_count += 1
        other_body_names = [name for name in body_names if name != OBJECT_BODY_NAME]
        robot_hand_contact = any(
            any(marker in name.lower() for marker in HAND_CONTACT_BODY_MARKERS)
            for name in other_body_names
        )
        object_robot_contact_count += int(robot_hand_contact)
        if len(sampled_contacts) < 25:
            sampled_contacts.append(
                {
                    "contact_index": index,
                    "geom_ids": geom_ids,
                    "body_names": body_names,
                    "robot_hand_contact": robot_hand_contact,
                    "distance": round(float(contact.dist), 9),
                    "position": [round(float(value), 6) for value in contact.pos],
                }
            )
    return {
        "object_any_contact_count": object_any_contact_count,
        "object_robot_contact_count": object_robot_contact_count,
        "sampled_object_contacts": sampled_contacts,
    }


def _nearest_hand_body_distance_to_object(
    model: Any,
    data: Any,
    mujoco_module: Any,
) -> dict[str, Any]:
    object_body_id = int(
        mujoco_module.mj_name2id(
            model,
            mujoco_module.mjtObj.mjOBJ_BODY,
            OBJECT_BODY_NAME,
        )
    )
    if object_body_id < 0:
        return {"available": False}
    object_pos = np.asarray(data.xpos[object_body_id], dtype=float)
    nearest_name: str | None = None
    nearest_distance: float | None = None
    for body_id in range(int(model.nbody)):
        name = _body_name(model, mujoco_module, body_id)
        if not any(marker in name.lower() for marker in HAND_CONTACT_BODY_MARKERS):
            continue
        distance = float(np.linalg.norm(np.asarray(data.xpos[body_id], dtype=float) - object_pos))
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_name = name
    return {
        "available": nearest_distance is not None,
        "nearest_body_name": nearest_name,
        "distance_m": round(float(nearest_distance), 6) if nearest_distance is not None else None,
    }


def _mujoco_object_id(mujoco_module: Any, model: Any, object_type: Any, name: str) -> int:
    return int(mujoco_module.mj_name2id(model, object_type, name))


def _matrix3_rows(flat: Sequence[float]) -> list[list[float]]:
    values = [float(value) for value in flat[:9]]
    if len(values) < 9:
        values.extend([0.0] * (9 - len(values)))
    return [values[index : index + 3] for index in range(0, 9, 3)]


def _mat3_mul_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    ]


def _g1_body_landmark_position(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    body_name: str,
    local_offset_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    body_id = _mujoco_object_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_BODY,
        body_name,
    )
    if body_id < 0:
        return {
            "available": False,
            "body_name": body_name,
            "blockers": [f"missing_g1_body:{body_name}"],
        }
    origin = [float(value) for value in data.xpos[body_id][:3]]
    offset = [float(value) for value in (local_offset_m or [0.0, 0.0, 0.0])[:3]]
    if len(offset) < 3:
        offset.extend([0.0] * (3 - len(offset)))
    world = origin
    if any(abs(value) > 1e-12 for value in offset):
        body_xmat = _matrix3_rows(data.xmat[body_id])
        world_offset = _mat3_mul_vec(body_xmat, offset)
        world = [origin[index] + world_offset[index] for index in range(3)]
    return {
        "available": True,
        "body_name": body_name,
        "body_id": body_id,
        "local_offset_m": [round(value, 6) for value in offset],
        "world_xyz_m": [round(value, 6) for value in world],
    }


def _g1_body_landmark_position_from_spec(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    primary = _g1_body_landmark_position(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        body_name=str(spec["body_name"]),
        local_offset_m=spec.get("local_offset_m"),
    )
    if primary.get("available"):
        primary["landmark_source"] = "primary_g1_body"
        return primary
    fallback_body = str(spec.get("fallback_body_name") or "")
    if not fallback_body:
        primary["landmark_source"] = "primary_g1_body_missing"
        return primary
    fallback = _g1_body_landmark_position(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        body_name=fallback_body,
        local_offset_m=spec.get("fallback_local_offset_m"),
    )
    if fallback.get("available"):
        fallback["landmark_source"] = "fallback_g1_body_with_local_offset"
        fallback["preferred_body_name"] = str(spec["body_name"])
        fallback["fallback_for_missing_preferred_body"] = True
        return fallback
    fallback["blockers"] = sorted(
        set(list(primary.get("blockers") or []) + list(fallback.get("blockers") or []))
    )
    fallback["landmark_source"] = "preferred_and_fallback_g1_bodies_missing"
    fallback["preferred_body_name"] = str(spec["body_name"])
    return fallback


def _fixed_camera_projection_context(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    camera_id: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    fixed_camera_name = FIXED_G1_CAMERA_NAMES.get(camera_id)
    if not fixed_camera_name:
        return {
            "available": False,
            "camera_id": camera_id,
            "blockers": [f"camera_not_fixed_g1_camera:{camera_id}"],
        }
    camera_obj_id = _mujoco_object_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_CAMERA,
        fixed_camera_name,
    )
    if camera_obj_id < 0:
        return {
            "available": False,
            "camera_id": camera_id,
            "fixed_mujoco_camera_name": fixed_camera_name,
            "blockers": [f"missing_fixed_g1_camera:{fixed_camera_name}"],
        }
    try:
        fovy_deg = float(model.cam_fovy[camera_obj_id])
        cam_xpos = [float(value) for value in data.cam_xpos[camera_obj_id][:3]]
        cam_xmat = _matrix3_rows(data.cam_xmat[camera_obj_id])
    except Exception as exc:
        return {
            "available": False,
            "camera_id": camera_id,
            "fixed_mujoco_camera_name": fixed_camera_name,
            "camera_obj_id": camera_obj_id,
            "blockers": ["fixed_g1_camera_projection_metadata_unavailable"],
            "error": str(exc),
        }
    focal_px = 0.5 * float(image_height) / math.tan(math.radians(fovy_deg) / 2.0)
    return {
        "available": True,
        "camera_id": camera_id,
        "fixed_mujoco_camera_name": fixed_camera_name,
        "camera_obj_id": camera_obj_id,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "fovy_deg": round(fovy_deg, 6),
        "focal_length_px": round(focal_px, 6),
        "camera_world_xyz_m": [round(value, 6) for value in cam_xpos],
        "camera_xmat_row_major": [[round(value, 8) for value in row] for row in cam_xmat],
        "projection_method": "mujoco_fixed_camera_pinhole_from_data_cam_xpos_xmat",
    }


def _project_world_xyz_to_camera_pixel(
    *,
    world_xyz_m: Sequence[float],
    projection_context: Mapping[str, Any],
) -> dict[str, Any]:
    if not projection_context.get("available"):
        return {
            "available": False,
            "blockers": list(
                projection_context.get("blockers") or ["projection_context_unavailable"]
            ),
        }
    cam_pos = [float(value) for value in projection_context.get("camera_world_xyz_m", [])[:3]]
    if len(cam_pos) < 3:
        return {"available": False, "blockers": ["camera_world_position_unavailable"]}
    cam_xmat = projection_context.get("camera_xmat_row_major")
    if not (
        isinstance(cam_xmat, Sequence)
        and len(cam_xmat) >= 3
        and all(isinstance(row, Sequence) and len(row) >= 3 for row in cam_xmat[:3])
    ):
        return {"available": False, "blockers": ["camera_orientation_unavailable"]}
    world = [float(value) for value in world_xyz_m[:3]]
    if len(world) < 3:
        return {"available": False, "blockers": ["world_point_unavailable"]}
    delta = [world[index] - cam_pos[index] for index in range(3)]
    rows = [[float(value) for value in row[:3]] for row in cam_xmat[:3]]
    columns = [[rows[0][index], rows[1][index], rows[2][index]] for index in range(3)]
    camera_local = [
        sum(delta[index] * columns[axis][index] for index in range(3)) for axis in range(3)
    ]
    depth = abs(float(camera_local[2]))
    if depth <= 1e-9:
        return {
            "available": False,
            "camera_local_xyz": [round(value, 6) for value in camera_local],
            "blockers": ["camera_projection_depth_near_zero"],
        }
    width = int(projection_context.get("image_width") or 0)
    height = int(projection_context.get("image_height") or 0)
    focal_px = float(projection_context.get("focal_length_px") or 0.0)
    if width <= 0 or height <= 0 or focal_px <= 0.0:
        return {
            "available": False,
            "camera_local_xyz": [round(value, 6) for value in camera_local],
            "blockers": ["camera_projection_intrinsics_unavailable"],
        }
    u = width * 0.5 + focal_px * float(camera_local[0]) / depth
    v = height * 0.5 - focal_px * float(camera_local[1]) / depth
    return {
        "available": True,
        "u_px": round(u, 3),
        "v_px": round(v, 3),
        "depth_m_abs": round(depth, 6),
        "camera_local_xyz": [round(value, 6) for value in camera_local],
        "inside_image": bool(0.0 <= u < width and 0.0 <= v < height),
        "projection_depth_sign_abs_used": True,
    }


def _build_policy_action_projected_skeleton_trace_row(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    generated_at: str,
    step: int,
    action_frame_index: int,
    source_action_key: str | None,
    camera_id: str = "head_pov",
    image_width: int = 640,
    image_height: int = 480,
) -> dict[str, Any]:
    projection_context = _fixed_camera_projection_context(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        camera_id=camera_id,
        image_width=image_width,
        image_height=image_height,
    )
    landmarks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for spec in G1_UPPER_BODY_LANDMARK_SPECS:
        landmark = {
            "landmark_id": spec["landmark_id"],
            **_g1_body_landmark_position_from_spec(
                mujoco_module=mujoco_module,
                model=model,
                data=data,
                spec=spec,
            ),
        }
        if landmark.get("available"):
            landmark["image_projection"] = _project_world_xyz_to_camera_pixel(
                world_xyz_m=landmark.get("world_xyz_m", []),
                projection_context=projection_context,
            )
        else:
            blockers.extend(str(item) for item in landmark.get("blockers") or [])
        landmarks.append(landmark)
    projected_count = sum(
        1 for landmark in landmarks if _mapping(landmark.get("image_projection")).get("available")
    )
    if not projection_context.get("available"):
        blockers.extend(str(item) for item in projection_context.get("blockers") or [])
    if projected_count <= 0:
        blockers.append("no_g1_upper_body_landmarks_projected_into_camera")
    return {
        "schema_version": PROJECTED_SKELETON_SCHEMA_VERSION,
        "status": "completed" if not blockers else "warning_partial_projection",
        "generated_at": generated_at,
        "step": int(step),
        "sim_time_s": round(float(data.time), 9),
        "action_frame_index": int(action_frame_index),
        "source_action_key": source_action_key,
        "camera_id": camera_id,
        "projection_context": projection_context,
        "landmarks": landmarks,
        "segments": [{"from": start, "to": end} for start, end in G1_UPPER_BODY_SKELETON_SEGMENTS],
        "available_landmark_count": sum(1 for landmark in landmarks if landmark.get("available")),
        "projected_landmark_count": projected_count,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "policy_derived_action_conditioning": True,
            "not_a_learned_robot_policy_action": False,
            "projected_skeleton_trace_derived_from_seed_render_geometry": False,
            "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": False,
            "nominal_kinematic_projection_without_scene_or_wbc_bridge": False,
            "official_wbc_or_sim_bridge_used": True,
            "official_groot_wholebodycontrol_sim2sim_used": False,
            "simulator_only_mujoco_action_trace_bridge_used": True,
            "uses_unitree_g1_mujoco_body_transforms": True,
            "uses_simulated_fixed_head_or_torso_camera_projection": bool(
                projection_context.get("available")
            ),
            "simulated_state_not_physical_robot_sensor_evidence": True,
            "not_task_success_proof": True,
            "not_physical_robot_sensor_proof": True,
        },
    }


def _projected_skeleton_manifest(
    *,
    generated_at: str,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    projectable_rows = [
        row
        for row in rows
        if row.get("status") == "completed" or int(row.get("projected_landmark_count") or 0) > 0
    ]
    blockers: list[str] = []
    if not rows:
        blockers.append("blocked_no_policy_action_sim2sim_rows")
    if rows and not projectable_rows:
        blockers.append("blocked_no_policy_action_projected_skeleton_rows")
    return {
        "schema_version": PROJECTED_SKELETON_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if projectable_rows else "blocked",
        "trace_jsonl": str(output_path),
        "row_count": len(rows),
        "projectable_row_count": len(projectable_rows),
        "landmark_ids": [str(spec["landmark_id"]) for spec in G1_UPPER_BODY_LANDMARK_SPECS],
        "segments": [{"from": start, "to": end} for start, end in G1_UPPER_BODY_SKELETON_SEGMENTS],
        "blockers": blockers,
        "claim_boundary": {
            "derived_from_policy_action_sim2sim_bridge": True,
            "derived_from_unitree_g1_mujoco_body_transforms": True,
            "derived_from_head_or_torso_sim_camera_metadata": True,
            "official_groot_wholebodycontrol_sim2sim_used": False,
            "simulated_g1_arm_hand_state_available_for_wam_conditioning": bool(projectable_rows),
            "not_physical_robot_sensor_proof": True,
            "not_wam_generated_output": True,
            "not_success_review_label": True,
        },
    }


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_video_from_frames(*, frames_dir: Path, output_path: Path, fps: int) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    frame_count = len(sorted(frames_dir.glob("frame_*.png")))
    if not ffmpeg:
        return {
            "path": str(output_path),
            "status": "blocked",
            "frame_count": frame_count,
            "blockers": ["ffmpeg_unavailable"],
        }
    ensure_dir(output_path.parent)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "path": str(output_path),
        "status": "completed" if result.returncode == 0 and output_path.is_file() else "blocked",
        "frame_count": frame_count,
        "fps": int(fps),
        "ffmpeg_exit_code": result.returncode,
        "stderr_size_bytes": len(result.stderr or ""),
        "blockers": []
        if result.returncode == 0 and output_path.is_file()
        else ["ffmpeg_video_encode_failed"],
    }


def _ffprobe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"path": str(path), "status": "not_checked", "blockers": ["ffprobe_unavailable"]}
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {
            "path": str(path),
            "status": "blocked",
            "ffprobe_exit_code": result.returncode,
            "blockers": ["ffprobe_failed"],
        }
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"path": str(path), "status": "blocked", "blockers": ["ffprobe_json_failed"]}
    stream = (payload.get("streams") or [{}])[0]
    return {
        "path": str(path),
        "status": "completed",
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "r_frame_rate": stream.get("r_frame_rate"),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
    }


def _blocked(
    *,
    job_dir: Path,
    generated_at: str,
    blockers: Sequence[str],
    policy_action_output: Path | None = None,
    scene_xml: Path | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "unitree_groot_n17_sonic_sim2sim_command_ran": False,
        "unitree_groot_n17_sonic_action_chunk_consumed": False,
        "policy_action_output_path": str(policy_action_output) if policy_action_output else None,
        "scene_xml": str(scene_xml) if scene_xml else None,
        "blockers": sorted(set(blockers)),
        "error_type": error_type,
        "claim_boundary": _claim_boundary(),
    }
    write_json(job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json", payload)
    return payload


def _claim_boundary() -> dict[str, Any]:
    return {
        "simulator_only": True,
        "blueprint_mujoco_action_consumption_bridge": True,
        "official_groot_wholebodycontrol_sim2sim_used": False,
        "official_sonic_wbc_mapping_proven": False,
        "not_physical_robot_command": True,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
    }


def run_unitree_groot_n17_sonic_sim2sim(
    *,
    job_dir: str | Path,
    policy_action_output: str | Path | None = None,
    scene_xml: str | Path | None = None,
    steps: int = DEFAULT_STEPS,
    action_hold_steps: int = DEFAULT_ACTION_HOLD_STEPS,
    render_video: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser()
    ensure_dir(job)
    action_path = (
        Path(policy_action_output).expanduser()
        if policy_action_output is not None
        else job / "policy_action_model_command_output.json"
    )
    scene_path = Path(scene_xml).expanduser() if scene_xml is not None else _scene_xml_from_job(job)
    if not action_path.is_file():
        return _blocked(
            job_dir=job,
            generated_at=generated_at,
            blockers=["blocked_missing_policy_action_model_command_output"],
            policy_action_output=action_path,
            scene_xml=scene_path,
        )
    if scene_path is None or not scene_path.is_file():
        return _blocked(
            job_dir=job,
            generated_at=generated_at,
            blockers=["blocked_missing_mujoco_scene_xml_for_unitree_groot_n17_sonic_sim2sim"],
            policy_action_output=action_path,
            scene_xml=scene_path,
        )
    try:
        policy_output = _read_json(action_path)
        action = _extract_action(policy_output)
        source_key, action_values = _extract_action_vector(action)
        source_dim, action_frames = _chunk_action_frames(action_values)
        if not action_frames:
            return _blocked(
                job_dir=job,
                generated_at=generated_at,
                blockers=["blocked_policy_action_output_missing_numeric_sonic_action_chunk"],
                policy_action_output=action_path,
                scene_xml=scene_path,
            )
        import mujoco  # type: ignore[import-not-found]

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        object_qpos_addr = _freejoint_qpos_addr(model, mujoco, OBJECT_FREEJOINT_NAME)
        object_initial_pose = _object_initial_pose_from_job(job)
        if object_qpos_addr is not None and object_initial_pose is not None:
            data.qpos[object_qpos_addr : object_qpos_addr + 7] = object_initial_pose
        _initialize_joint_holds(model, data)
        mujoco.mj_forward(model, data)
        initial_object_pose = _object_pose(data, object_qpos_addr)
        initial_nearest_hand_object_distance = _nearest_hand_body_distance_to_object(
            model,
            data,
            mujoco,
        )
        bindings = _actuator_bindings(model, mujoco)
        if not bindings:
            return _blocked(
                job_dir=job,
                generated_at=generated_at,
                blockers=["blocked_no_unitree_g1_upper_body_or_hand_actuators_in_mujoco_scene"],
                policy_action_output=action_path,
                scene_xml=scene_path,
            )
        trace_rows: list[dict[str, Any]] = []
        projected_skeleton_rows: list[dict[str, Any]] = []
        frames_dir = job / "unitree_groot_n17_sonic_sim2sim_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        ensure_dir(frames_dir)
        renderer = None
        image_module = None
        if render_video:
            try:
                from PIL import Image

                renderer = mujoco.Renderer(model, height=360, width=640)
                image_module = Image
            except Exception:
                renderer = None
                image_module = None
        initial_qpos = data.qpos.copy()
        requested_action_frame_count = max(1, int(steps))
        hold_steps = max(1, int(action_hold_steps))
        step_count = requested_action_frame_count * hold_steps
        frame_count = len(action_frames)
        total_object_any_contact_count = 0
        total_object_robot_contact_count = 0
        minimum_nearest_hand_object_distance: dict[str, Any] = dict(
            initial_nearest_hand_object_distance
        )
        sampled_object_contacts: list[dict[str, Any]] = []
        for step in range(step_count):
            action_frame_index = min(frame_count - 1, step // hold_steps)
            action_frame = action_frames[action_frame_index]
            applied_targets: list[dict[str, Any]] = []
            for binding in bindings:
                dim_index = int(binding["action_dim_index"])
                value = float(action_frame[dim_index % len(action_frame)])
                low, high = binding["ctrl_range"]
                target = _control_target_from_action_value(value, low, high)
                actuator_id = int(binding["actuator_id"])
                data.ctrl[actuator_id] = target
                applied_targets.append(
                    {
                        "joint_name": binding["joint_name"],
                        "actuator_id": actuator_id,
                        "action_dim_index": dim_index,
                        "source_action_value": round(value, 9),
                        "target_rad": round(target, 9),
                    }
                )
            mujoco.mj_step(model, data)
            object_contact_state = _object_contact_state(model, data, mujoco)
            total_object_any_contact_count += int(
                object_contact_state["object_any_contact_count"]
            )
            total_object_robot_contact_count += int(
                object_contact_state["object_robot_contact_count"]
            )
            nearest_hand_object_distance = _nearest_hand_body_distance_to_object(
                model,
                data,
                mujoco,
            )
            if (
                nearest_hand_object_distance.get("available")
                and (
                    not minimum_nearest_hand_object_distance.get("available")
                    or float(nearest_hand_object_distance.get("distance_m") or 0.0)
                    < float(minimum_nearest_hand_object_distance.get("distance_m") or 0.0)
                )
            ):
                minimum_nearest_hand_object_distance = nearest_hand_object_distance
            if len(sampled_object_contacts) < 50:
                for contact in object_contact_state["sampled_object_contacts"]:
                    if len(sampled_object_contacts) >= 50:
                        break
                    sampled_object_contacts.append(
                        {
                            **contact,
                            "step": step,
                            "action_frame_index": action_frame_index,
                            "sim_time_s": round(float(data.time), 9),
                        }
                    )
            joint_positions = {
                str(binding["joint_name"]): round(float(data.qpos[int(binding["qpos_addr"])]), 9)
                for binding in bindings
            }
            trace_rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "step": step,
                    "sim_time_s": round(float(data.time), 9),
                    "action_frame_index": action_frame_index,
                    "source_action_key": source_key,
                    "source_action_dim": source_dim,
                    "action_hold_step_index": step % hold_steps,
                    "object_any_contact_count": int(
                        object_contact_state["object_any_contact_count"]
                    ),
                    "object_robot_contact_count": int(
                        object_contact_state["object_robot_contact_count"]
                    ),
                    "nearest_hand_object_distance": nearest_hand_object_distance,
                    "applied_target_count": len(applied_targets),
                    "applied_targets": applied_targets,
                    "joint_positions": joint_positions,
                }
            )
            projected_skeleton_rows.append(
                _build_policy_action_projected_skeleton_trace_row(
                    mujoco_module=mujoco,
                    model=model,
                    data=data,
                    generated_at=generated_at,
                    step=step,
                    action_frame_index=action_frame_index,
                    source_action_key=source_key,
                )
            )
            if renderer is not None and image_module is not None:
                renderer.update_scene(data)
                frame = renderer.render()
                image_module.fromarray(frame).save(frames_dir / f"frame_{step:04d}.png")
        if renderer is not None:
            renderer.close()
        trace_path = job / "unitree_groot_n17_sonic_sim2sim_action_trace.jsonl"
        _jsonl(trace_path, trace_rows)
        projected_skeleton_trace_path = job / "policy_action_projected_skeleton_trace.jsonl"
        _jsonl(projected_skeleton_trace_path, projected_skeleton_rows)
        projected_skeleton_manifest = _projected_skeleton_manifest(
            generated_at=generated_at,
            rows=projected_skeleton_rows,
            output_path=projected_skeleton_trace_path,
        )
        write_json(job / "policy_action_projected_skeleton_manifest.json", projected_skeleton_manifest)
        qpos_delta = np.abs(data.qpos - initial_qpos)
        moved_upper_body_joint_count = sum(
            1 for binding in bindings if qpos_delta[int(binding["qpos_addr"])] > 1e-5
        )
        max_upper_body_joint_delta_rad = max(
            [float(qpos_delta[int(binding["qpos_addr"])]) for binding in bindings] or [0.0]
        )
        final_object_pose = _object_pose(data, object_qpos_addr)
        object_displacement_m = 0.0
        object_horizontal_displacement_m = 0.0
        if initial_object_pose.get("available") and final_object_pose.get("available"):
            initial_position = np.asarray(initial_object_pose["position"], dtype=float)
            final_position = np.asarray(final_object_pose["position"], dtype=float)
            object_displacement_m = float(
                np.linalg.norm(final_position - initial_position)
            )
            object_horizontal_displacement_m = float(
                np.linalg.norm(final_position[:2] - initial_position[:2])
            )
        final_nearest_hand_object_distance = _nearest_hand_body_distance_to_object(
            model,
            data,
            mujoco,
        )
        policy_chunk_integrated_contact_rollout_success = bool(
            total_object_robot_contact_count > 0
            and object_horizontal_displacement_m >= OBJECT_DISPLACEMENT_SUCCESS_M
        )
        video_path = job / "unitree_groot_n17_sonic_sim2sim_review.mp4"
        video = (
            _write_video_from_frames(frames_dir=frames_dir, output_path=video_path, fps=24)
            if render_video and any(frames_dir.glob("frame_*.png"))
            else {
                "path": str(video_path),
                "status": "skipped",
                "blockers": ["render_video_disabled_or_renderer_unavailable"],
            }
        )
        ffprobe = (
            _ffprobe(video_path)
            if video.get("status") == "completed" and video_path.is_file()
            else {
                "path": str(video_path),
                "status": "not_checked",
                "blockers": list(video.get("blockers", [])),
            }
        )
        completed = bool(moved_upper_body_joint_count > 0 and trace_rows)
        contact_rollout_blockers: list[str] = []
        if not total_object_robot_contact_count:
            contact_rollout_blockers.append(
                "blocked_no_robot_hand_object_contact_from_gr00t_sonic_action_chunk"
            )
        if object_horizontal_displacement_m < OBJECT_DISPLACEMENT_SUCCESS_M:
            contact_rollout_blockers.append(
                "blocked_horizontal_object_displacement_below_success_threshold_from_gr00t_sonic_action_chunk"
            )
        execution_blockers = (
            contact_rollout_blockers
            if completed
            else [
                "blocked_unitree_groot_n17_sonic_sim2sim_no_joint_motion",
                *contact_rollout_blockers,
            ]
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if completed else "blocked",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "unitree_groot_n17_sonic_sim2sim_command_ran": completed,
            "unitree_groot_n17_sonic_action_chunk_consumed": completed,
            "unitree_policy_action_command_ran": bool(
                policy_output.get("unitree_policy_action_command_ran")
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_output.get("unitree_groot_n17_sonic_policy_action_command_ran")
            ),
            "policy_action_output_path": str(action_path),
            "scene_xml": str(scene_path),
            "source_action_key": source_key,
            "source_action_value_count": len(action_values),
            "source_action_dim": source_dim,
            "source_action_frame_count": frame_count,
            "sim_step_count": step_count,
            "requested_action_frame_count": requested_action_frame_count,
            "action_hold_steps": hold_steps,
            "mujoco_version": getattr(mujoco, "__version__", None),
            "mujoco_g1_scene_loaded": True,
            "upper_body_or_hand_actuator_count": len(bindings),
            "moved_upper_body_or_hand_joint_count": moved_upper_body_joint_count,
            "max_upper_body_or_hand_joint_delta_rad": round(max_upper_body_joint_delta_rad, 9),
            "object_freejoint_initialized": bool(
                object_qpos_addr is not None and object_initial_pose is not None
            ),
            "object_initial_pose": initial_object_pose,
            "object_final_pose": final_object_pose,
            "object_displacement_m": round(float(object_displacement_m), 9),
            "object_horizontal_displacement_m": round(
                float(object_horizontal_displacement_m),
                9,
            ),
            "object_displacement_success_threshold_m": OBJECT_DISPLACEMENT_SUCCESS_M,
            "object_displacement_success_axis": "xy",
            "object_any_contact_count": total_object_any_contact_count,
            "object_robot_contact_count": total_object_robot_contact_count,
            "object_displacement_without_robot_contact": bool(
                object_displacement_m > 0.0 and total_object_robot_contact_count == 0
            ),
            "object_horizontal_displacement_without_robot_contact": bool(
                object_horizontal_displacement_m > 0.0
                and total_object_robot_contact_count == 0
            ),
            "sampled_object_contacts": sampled_object_contacts,
            "initial_nearest_hand_object_distance": initial_nearest_hand_object_distance,
            "final_nearest_hand_object_distance": final_nearest_hand_object_distance,
            "minimum_nearest_hand_object_distance": minimum_nearest_hand_object_distance,
            "policy_chunk_integrated_contact_rollout_success": (
                policy_chunk_integrated_contact_rollout_success
            ),
            "policy_action_chunk_integrated_into_contact_rollout": (
                policy_chunk_integrated_contact_rollout_success
            ),
            "action_trace_jsonl": str(trace_path),
            "policy_action_projected_skeleton_trace_jsonl": str(projected_skeleton_trace_path),
            "policy_action_projected_skeleton_trace_path": str(projected_skeleton_trace_path),
            "policy_action_projected_skeleton_manifest": str(
                job / "policy_action_projected_skeleton_manifest.json"
            ),
            "policy_action_projected_skeleton_status": projected_skeleton_manifest["status"],
            "policy_action_projected_skeleton_projectable_row_count": (
                projected_skeleton_manifest["projectable_row_count"]
            ),
            "policy_derived_projected_skeleton_trace_present": bool(
                projected_skeleton_manifest["status"] == "completed"
            ),
            "video": video,
            "ffprobe": ffprobe,
            "blockers": sorted(set(execution_blockers)),
            "contact_rollout_blockers": contact_rollout_blockers,
            "claim_boundary": _claim_boundary(),
        }
        write_json(job / "unitree_groot_n17_sonic_sim2sim_execution.json", payload)
        write_json(
            job / "unitree_groot_n17_sonic_sim2sim_controller_truth.json",
            {
                "schema_version": "unitree_groot_n17_sonic_sim2sim_controller_truth.v1",
                "generated_at": generated_at,
                "status": payload["status"],
                "unitree_groot_n17_sonic_sim2sim_command_ran": payload[
                    "unitree_groot_n17_sonic_sim2sim_command_ran"
                ],
                "unitree_groot_n17_sonic_action_chunk_consumed": payload[
                    "unitree_groot_n17_sonic_action_chunk_consumed"
                ],
                "mujoco_g1_scene_loaded": True,
                "upper_body_or_hand_actuator_count": len(bindings),
                "moved_upper_body_or_hand_joint_count": moved_upper_body_joint_count,
                "object_freejoint_initialized": payload["object_freejoint_initialized"],
                "object_displacement_m": payload["object_displacement_m"],
                "object_horizontal_displacement_m": payload[
                    "object_horizontal_displacement_m"
                ],
                "object_robot_contact_count": total_object_robot_contact_count,
                "object_displacement_without_robot_contact": payload[
                    "object_displacement_without_robot_contact"
                ],
                "object_horizontal_displacement_without_robot_contact": payload[
                    "object_horizontal_displacement_without_robot_contact"
                ],
                "minimum_nearest_hand_object_distance": (
                    minimum_nearest_hand_object_distance
                ),
                "policy_action_chunk_integrated_into_contact_rollout": (
                    policy_chunk_integrated_contact_rollout_success
                ),
                "policy_action_projected_skeleton_trace_jsonl": str(
                    projected_skeleton_trace_path
                ),
                "policy_action_projected_skeleton_trace_path": str(
                    projected_skeleton_trace_path
                ),
                "policy_action_projected_skeleton_manifest": str(
                    job / "policy_action_projected_skeleton_manifest.json"
                ),
                "policy_action_projected_skeleton_status": projected_skeleton_manifest[
                    "status"
                ],
                "policy_action_projected_skeleton_projectable_row_count": (
                    projected_skeleton_manifest["projectable_row_count"]
                ),
                "official_groot_wholebodycontrol_sim2sim_used": False,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
                "accepted_anchor_manipulation_success_proven": False,
                "claim_boundary": _claim_boundary(),
                "blockers": payload["blockers"],
            },
        )
        return payload
    except Exception as exc:
        return _blocked(
            job_dir=job,
            generated_at=generated_at,
            blockers=["blocked_unitree_groot_n17_sonic_sim2sim_command_failed"],
            policy_action_output=action_path,
            scene_xml=scene_path,
            error_type=type(exc).__name__,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--policy-action-output", type=Path)
    parser.add_argument("--scene-xml", type=Path)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--action-hold-steps", type=int, default=DEFAULT_ACTION_HOLD_STEPS)
    parser.add_argument("--no-render-video", action="store_true")
    args = parser.parse_args(argv)
    summary = run_unitree_groot_n17_sonic_sim2sim(
        job_dir=args.job_dir,
        policy_action_output=args.policy_action_output
        or os.getenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_POLICY_OUTPUT"),
        scene_xml=args.scene_xml or os.getenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_SCENE_XML"),
        steps=args.steps,
        action_hold_steps=args.action_hold_steps,
        render_video=not args.no_render_video,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("status") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
