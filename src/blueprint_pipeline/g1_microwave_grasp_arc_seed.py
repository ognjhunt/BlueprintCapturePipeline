"""Build an oriented G1 microwave grasp/hinge-arc qualification seed.

This local, no-spend qualification binds the exact GEAR-SONIC G1 model to a
collision proxy reconstructed from the attested microwave door mesh bounds.
It solves the wrist position *and* orientation, closes the right hand, and
checks thumb/index/middle collision distances throughout a prescribed hinge
arc.  The door angle is prescribed for contact qualification, so this module
does not claim contact-driven articulation, a trained checkpoint, or task
success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .g1_microwave_reach_seed import (
    PINNED_G1_MODEL_SHA256,
    _finite_vector,
    _load_mapping,
    _rotation_wxyz,
    _sha256,
    minimum_jerk_trajectory,
)
from .g1_sonic_motion_token_conversion import (
    RIGHT_HAND_JOINT_NAMES,
    SOURCE_ACTION_JOINT_NAMES,
)
from .gear_sonic_joint_order_contract import (
    PINNED_WBC_SOURCE_REVISION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    validate_model_joint_names,
)
from .isaac_task_review_renderer import ARTICULATED_HANDLE_FOCUS_SCHEMA_VERSION


SCHEMA_VERSION = "g1_microwave_grasp_arc_seed.v3"
RIGHT_ARM_JOINT_NAMES = SOURCE_ACTION_JOINT_NAMES[29:36]
UPPER_BODY_SOLVE_JOINT_NAMES = (
    *SOURCE_ACTION_JOINT_NAMES[12:15],
    *RIGHT_ARM_JOINT_NAMES,
)
WRIST_BODY_NAME = "right_wrist_yaw_link"
DEFAULT_HANDLE_IN_WRIST_XYZ_M = (0.19, 0.04, 0.0)
DEFAULT_HAND_AXIS_POLARITY = 1.0
DEFAULT_GRASP_YAW_RAD = math.pi / 4.0
DEFAULT_RIGHT_HAND_GRASP = (0.9, 0.3, 0.9, 0.3, 0.0, -1.0, -0.6)
EGOCENTRIC_CAMERA_NAME = "g1_head_egocentric"
# The exact head mesh spans torso-local x=[-0.0643, 0.0750] and
# z=[0.2687, 0.4764].  Place the optical center just ahead of the face and at
# eye height.  MuJoCo cameras look down local -Z.  The task-directed head gaze
# is 41 degrees right and 12 degrees down from torso-forward, centering the
# handle at the qualified grasp instead of placing it outside the image.
EGOCENTRIC_CAMERA_POSITION_TORSO_M = (0.08, 0.0, 0.39)
EGOCENTRIC_CAMERA_XYAXES = (
    -0.6560590289905073,
    -0.754709580222772,
    0.0,
    0.1569129449004779,
    -0.13640234199367376,
    0.9781476007338057,
)
EGOCENTRIC_CAMERA_FOVY_DEG = 70.0


def axis_angle_rotation(axis: Any, angle_rad: float) -> np.ndarray:
    """Return a finite 3x3 Rodrigues rotation matrix."""

    direction = _finite_vector(axis, size=3, name="grasp_arc_axis")
    norm = float(np.linalg.norm(direction))
    angle = float(angle_rad)
    if norm <= 1e-12 or not math.isfinite(angle):
        raise ValueError("g1_microwave_grasp_arc_axis_angle_invalid")
    x, y, z = direction / norm
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def oriented_grasp_basis(
    *,
    pelvis_model_xyz_m: Any,
    handle_model_xyz_m: Any,
    hinge_axis_model_xyz: Any,
    hand_axis_polarity: float = DEFAULT_HAND_AXIS_POLARITY,
    grasp_yaw_rad: float = DEFAULT_GRASP_YAW_RAD,
) -> np.ndarray:
    """Orient hand Z along the handle and hand X toward the appliance."""

    pelvis = _finite_vector(pelvis_model_xyz_m, size=3, name="model_pelvis")
    handle = _finite_vector(handle_model_xyz_m, size=3, name="model_handle")
    hinge_axis = _finite_vector(
        hinge_axis_model_xyz, size=3, name="model_hinge_axis"
    )
    hinge_axis /= np.linalg.norm(hinge_axis)
    polarity = float(hand_axis_polarity)
    if polarity not in {-1.0, 1.0}:
        raise ValueError("g1_microwave_grasp_arc_hand_axis_polarity_invalid")
    approach = handle - pelvis
    approach -= hinge_axis * float(np.dot(approach, hinge_axis))
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm <= 1e-9:
        raise ValueError("g1_microwave_grasp_arc_approach_axis_invalid")
    hand_x = approach / approach_norm
    hand_z = polarity * hinge_axis
    hand_y = np.cross(hand_z, hand_x)
    hand_y /= np.linalg.norm(hand_y)
    base = np.column_stack((hand_x, hand_y, hand_z))
    cosine = math.cos(float(grasp_yaw_rad))
    sine = math.sin(float(grasp_yaw_rad))
    local_yaw = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    return base @ local_yaw


def _mesh_geometry(focus: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    panel = focus.get("panel_component")
    if not isinstance(panel, Mapping):
        raise ValueError("g1_microwave_grasp_arc_panel_component_missing")
    panel_center = _finite_vector(
        panel.get("center_xyz_m"), size=3, name="door_panel_center"
    )
    panel_extent = _finite_vector(
        panel.get("extent_xyz_m"), size=3, name="door_panel_extent"
    )
    handle_minimum = _finite_vector(
        focus.get("handle_bbox_min_xyz_m"), size=3, name="handle_bbox_minimum"
    )
    handle_maximum = _finite_vector(
        focus.get("handle_bbox_max_xyz_m"), size=3, name="handle_bbox_maximum"
    )
    handle_extent = handle_maximum - handle_minimum
    if np.any(panel_extent <= 0.0) or np.any(handle_extent <= 0.0):
        raise ValueError("g1_microwave_grasp_arc_mesh_extent_invalid")
    # The handle rail has a rectangular cross-section.  A capsule proxy must
    # use the inscribed (smaller) half-extent; using the larger half-extent
    # makes the proxy nearly 3 mm too thick and creates false penetrations.
    radius = float(min(handle_extent[0], handle_extent[1]) / 2.0)
    half_segment = float(max(0.005, handle_extent[2] / 2.0 - radius))
    return {
        "panel_center_world": panel_center,
        "panel_half_extent": panel_extent / 2.0,
        "handle_radius": radius,
        "handle_half_segment": half_segment,
    }


def _proxy_model(
    *,
    mujoco: Any,
    model_path: Path,
    pelvis_world: np.ndarray,
    pelvis_rotation: np.ndarray,
    handle_world: np.ndarray,
    hinge_world: np.ndarray,
    hinge_axis_world: np.ndarray,
    focus: Mapping[str, Any],
) -> tuple[Any, dict[str, np.ndarray | int | float]]:
    base_model = mujoco.MjModel.from_xml_path(str(model_path))
    pelvis_id = mujoco.mj_name2id(
        base_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    pelvis_model = np.asarray(base_model.body_pos[pelvis_id], dtype=np.float64)
    world_to_model = pelvis_rotation.T
    handle_model = pelvis_model + world_to_model @ (handle_world - pelvis_world)
    hinge_model = pelvis_model + world_to_model @ (hinge_world - pelvis_world)
    hinge_axis_model = world_to_model @ hinge_axis_world
    hinge_axis_model /= np.linalg.norm(hinge_axis_model)
    geometry = _mesh_geometry(focus)
    panel_center_world = np.asarray(geometry["panel_center_world"])
    panel_center_model = pelvis_model + world_to_model @ (
        panel_center_world - pelvis_world
    )
    model_z = world_to_model @ np.asarray((0.0, 0.0, 1.0))
    model_x = world_to_model @ np.asarray((1.0, 0.0, 0.0))
    model_y = world_to_model @ np.asarray((0.0, 1.0, 0.0))

    spec = mujoco.MjSpec.from_file(str(model_path))
    door = spec.worldbody.add_body(name="microwave_door", pos=hinge_model.tolist())
    door.add_joint(
        name="microwave_door_hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=hinge_axis_model.tolist(),
        limited=True,
        range=[-math.pi / 2.0, 0.0],
        damping=1.0,
    )
    door.add_geom(
        name="microwave_door_panel",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(panel_center_model - hinge_model).tolist(),
        xyaxes=[*model_x.tolist(), *model_y.tolist()],
        size=np.asarray(geometry["panel_half_extent"]).tolist(),
        mass=2.0,
        friction=[1.0, 0.005, 0.0001],
        margin=0.001,
    )
    handle_local = handle_model - hinge_model
    half_segment = float(geometry["handle_half_segment"])
    door.add_geom(
        name="microwave_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[
            *(handle_local - model_z * half_segment).tolist(),
            *(handle_local + model_z * half_segment).tolist(),
        ],
        size=[float(geometry["handle_radius"])],
        mass=0.1,
        friction=[1.5, 0.005, 0.0001],
        margin=0.004,
    )
    torso = spec.body("torso_link")
    if torso is None:
        raise ValueError("g1_microwave_grasp_arc_torso_body_missing")
    torso.add_camera(
        name=EGOCENTRIC_CAMERA_NAME,
        pos=list(EGOCENTRIC_CAMERA_POSITION_TORSO_M),
        xyaxes=list(EGOCENTRIC_CAMERA_XYAXES),
        fovy=EGOCENTRIC_CAMERA_FOVY_DEG,
    )
    proxy = spec.compile()
    return proxy, {
        "pelvis_model_xyz_m": pelvis_model,
        "handle_model_xyz_m": handle_model,
        "hinge_model_xyz_m": hinge_model,
        "hinge_axis_model_xyz": hinge_axis_model,
        "handle_radius_m": float(geometry["handle_radius"]),
        "handle_half_segment_m": half_segment,
        "egocentric_camera_name": EGOCENTRIC_CAMERA_NAME,
        "egocentric_camera_position_torso_m": np.asarray(
            EGOCENTRIC_CAMERA_POSITION_TORSO_M
        ),
    }


def solve_grasp_arc_seed(
    *,
    model_path: str | Path,
    standing_initialization_path: str | Path,
    initial_policy_observation_path: str | Path,
    target_focus_report_path: str | Path,
    reach_frame_count: int = 101,
    closure_frame_count: int = 26,
    pull_frame_count: int = 51,
    door_open_angle_rad: float = 0.35,
    expected_model_sha256: str = PINNED_G1_MODEL_SHA256,
    maximum_position_error_m: float = 0.008,
    maximum_orientation_error_rad: float = 0.04,
    maximum_handle_penetration_m: float = 0.003,
    maximum_panel_penetration_m: float = 0.001,
    maximum_contact_gap_m: float = 0.002,
    handle_in_wrist_xyz_m: Sequence[float] = DEFAULT_HANDLE_IN_WRIST_XYZ_M,
    hand_axis_polarity: float = DEFAULT_HAND_AXIS_POLARITY,
    grasp_yaw_rad: float = DEFAULT_GRASP_YAW_RAD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a 43D oriented reach, hand closure, and prescribed pull arc."""

    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("g1_microwave_grasp_arc_mujoco_missing") from exc

    model_file = Path(model_path).expanduser().resolve()
    standing_file = Path(standing_initialization_path).expanduser().resolve()
    observation_file = Path(initial_policy_observation_path).expanduser().resolve()
    focus_file = Path(target_focus_report_path).expanduser().resolve()
    if not model_file.is_file() or _sha256(model_file) != str(
        expected_model_sha256
    ).lower():
        raise ValueError("g1_microwave_grasp_arc_model_missing_or_sha256_mismatch")
    if min(int(reach_frame_count), int(closure_frame_count), int(pull_frame_count)) < 2:
        raise ValueError("g1_microwave_grasp_arc_frame_count_invalid")
    angle = float(door_open_angle_rad)
    if not 0.0 < angle <= math.pi / 2.0:
        raise ValueError("g1_microwave_grasp_arc_door_angle_invalid")

    standing = _load_mapping(standing_file, name="standing_initialization")
    observation = _load_mapping(observation_file, name="initial_policy_observation")
    focus = _load_mapping(focus_file, name="target_focus_report")
    if (
        standing.get("status") != "passed"
        or standing.get("surrogate") is not False
        or standing.get("pinned_wbc_source_revision") != PINNED_WBC_SOURCE_REVISION
    ):
        raise ValueError("g1_microwave_grasp_arc_standing_attestation_invalid")
    if (
        focus.get("schema_version") != ARTICULATED_HANDLE_FOCUS_SCHEMA_VERSION
        or focus.get("status") != "resolved_disconnected_articulated_handle"
        or focus.get("target_prim_path") != observation.get("target_prim_path")
    ):
        raise ValueError("g1_microwave_grasp_arc_focus_report_invalid")
    context = observation.get("camera_projection_context")
    if not isinstance(context, Mapping) or context.get("status") != (
        "captured_from_live_persistent_isaac_session"
    ):
        raise ValueError("g1_microwave_grasp_arc_live_context_invalid")
    pelvis = context.get("live_isaac_pelvis_world_pose")
    if not isinstance(pelvis, Mapping):
        raise ValueError("g1_microwave_grasp_arc_pelvis_pose_missing")
    pelvis_world = _finite_vector(
        pelvis.get("position_xyz"), size=3, name="live_pelvis_world"
    )
    pelvis_rotation = _rotation_wxyz(pelvis.get("quaternion_wxyz"))
    handle_world = _finite_vector(
        focus.get("target_world_xyz_m"), size=3, name="handle_world"
    )
    hinge_world = _finite_vector(
        focus.get("hinge_world_xyz_m"), size=3, name="hinge_world"
    )
    hinge_axis_world = _finite_vector(
        focus.get("joint_world_axis_xyz"), size=3, name="hinge_axis_world"
    )

    original = mujoco.MjModel.from_xml_path(str(model_file))
    model_joint_names = [
        str(mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, index) or "")
        for index in range(original.njnt)
        if int(original.jnt_type[index]) != int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    validate_model_joint_names(model_joint_names)
    model, geometry = _proxy_model(
        mujoco=mujoco,
        model_path=model_file,
        pelvis_world=pelvis_world,
        pelvis_rotation=pelvis_rotation,
        handle_world=handle_world,
        hinge_world=hinge_world,
        hinge_axis_world=hinge_axis_world,
        focus=focus,
    )
    data = mujoco.MjData(model)
    measured = _finite_vector(
        standing.get("measured_full_joint_positions"),
        size=len(PROTOCOL_V4_FULL_JOINT_ORDER),
        name="standing_joint_positions",
    )
    for name, value in zip(PROTOCOL_V4_FULL_JOINT_ORDER, measured):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    joint_limits: list[tuple[float, float]] = []
    for name in UPPER_BODY_SOLVE_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            raise ValueError(
                "g1_microwave_grasp_arc_upper_body_joint_missing_or_unlimited"
            )
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        joint_limits.append(tuple(float(v) for v in model.jnt_range[joint_id]))
    wrist_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY_NAME
    )
    if wrist_id < 0:
        raise ValueError("g1_microwave_grasp_arc_wrist_body_missing")

    basis = oriented_grasp_basis(
        pelvis_model_xyz_m=geometry["pelvis_model_xyz_m"],
        handle_model_xyz_m=geometry["handle_model_xyz_m"],
        hinge_axis_model_xyz=geometry["hinge_axis_model_xyz"],
        hand_axis_polarity=hand_axis_polarity,
        grasp_yaw_rad=grasp_yaw_rad,
    )
    handle_in_wrist = _finite_vector(
        handle_in_wrist_xyz_m,
        size=3,
        name="handle_in_wrist",
    )
    wrist_position = np.asarray(geometry["handle_model_xyz_m"]) - basis @ handle_in_wrist
    initial_solver_positions = np.asarray(
        [data.qpos[address] for address in qpos_addresses]
    )

    def solve_pose(
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        *,
        max_iterations: int,
    ) -> tuple[int, float, float, np.ndarray]:
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            mujoco.mj_forward(model, data)
            current_rotation = np.asarray(data.xmat[wrist_id]).reshape(3, 3)
            position_error = target_position - np.asarray(data.xpos[wrist_id])
            rotation_error = 0.5 * sum(
                (
                    np.cross(current_rotation[:, axis], target_rotation[:, axis])
                    for axis in range(3)
                ),
                start=np.zeros(3),
            )
            if (
                float(np.linalg.norm(position_error)) <= 0.001
                and float(np.linalg.norm(rotation_error)) <= 0.01
            ):
                break
            jacobian_position = np.zeros((3, model.nv))
            jacobian_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacBody(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                wrist_id,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, dof_addresses],
                    jacobian_rotation[:, dof_addresses],
                )
            )
            error = np.concatenate((position_error, rotation_error))
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.0025 * np.eye(6), error
            )
            regularization = np.asarray((0.004, 0.008, 0.008, *([0.001] * 7)))
            delta -= regularization * (
                np.asarray([data.qpos[address] for address in qpos_addresses])
                - initial_solver_positions
            )
            delta = np.clip(delta, -0.04, 0.04)
            for index, (address, (lower, upper)) in enumerate(
                zip(qpos_addresses, joint_limits)
            ):
                data.qpos[address] = np.clip(
                    data.qpos[address] + 0.8 * delta[index],
                    lower + 1e-4,
                    upper - 1e-4,
                )
        mujoco.mj_forward(model, data)
        current_rotation = np.asarray(data.xmat[wrist_id]).reshape(3, 3)
        position_distance = float(
            np.linalg.norm(target_position - np.asarray(data.xpos[wrist_id]))
        )
        orientation_distance = float(
            np.linalg.norm(
                0.5
                * sum(
                    (
                        np.cross(
                            current_rotation[:, axis], target_rotation[:, axis]
                        )
                        for axis in range(3)
                    ),
                    start=np.zeros(3),
                )
            )
        )
        joints = np.asarray([data.qpos[address] for address in qpos_addresses])
        return iterations, position_distance, orientation_distance, joints

    opening_angles = minimum_jerk_trajectory(
        [0.0], [-angle], frame_count=int(pull_frame_count)
    )[:, 0]
    upper_body_rows: list[np.ndarray] = []
    pose_rows: list[dict[str, Any]] = []
    hinge_model = np.asarray(geometry["hinge_model_xyz_m"])
    hinge_axis_model = np.asarray(geometry["hinge_axis_model_xyz"])
    for index, opening_angle in enumerate(opening_angles):
        rotation = axis_angle_rotation(hinge_axis_model, float(opening_angle))
        target_position = hinge_model + rotation @ (wrist_position - hinge_model)
        target_rotation = rotation @ basis
        iterations, position_error, orientation_error, upper_body = solve_pose(
            target_position,
            target_rotation,
            max_iterations=800 if index == 0 else 350,
        )
        if position_error > float(maximum_position_error_m):
            raise RuntimeError(
                "g1_microwave_grasp_arc_position_gate_failed:"
                f"frame={index}:error_m={position_error:.9f}:"
                f"maximum_m={float(maximum_position_error_m):.9f}"
            )
        if orientation_error > float(maximum_orientation_error_rad):
            raise RuntimeError(
                "g1_microwave_grasp_arc_orientation_gate_failed:"
                f"frame={index}:error_rad={orientation_error:.9f}:"
                f"maximum_rad={float(maximum_orientation_error_rad):.9f}"
            )
        upper_body_rows.append(upper_body)
        pose_rows.append(
            {
                "frame_index": index,
                "door_angle_rad": float(opening_angle),
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "solver_iterations": iterations,
            }
        )

    hand_addresses: list[int] = []
    for name, target in zip(RIGHT_HAND_JOINT_NAMES, DEFAULT_RIGHT_HAND_GRASP):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            raise ValueError("g1_microwave_grasp_arc_hand_joint_missing_or_unlimited")
        lower, upper = (float(v) for v in model.jnt_range[joint_id])
        if not lower <= float(target) <= upper:
            raise ValueError("g1_microwave_grasp_arc_hand_target_out_of_range")
        hand_addresses.append(int(model.jnt_qposadr[joint_id]))

    handle_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "microwave_handle"
    )
    panel_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "microwave_door_panel"
    )
    door_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "microwave_door_hinge"
    )
    door_address = int(model.jnt_qposadr[door_joint])
    hand_geoms: dict[str, list[int]] = {}
    for geom_id in range(model.ngeom):
        body_name = str(
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            )
            or ""
        )
        if body_name.startswith("right_hand_") and int(model.geom_contype[geom_id]):
            hand_geoms.setdefault(body_name, []).append(geom_id)
    if not hand_geoms:
        raise ValueError("g1_microwave_grasp_arc_hand_collision_geoms_missing")

    def signed_group_distances() -> dict[str, float]:
        mujoco.mj_forward(model, data)
        distances = {
            body_name: min(
                float(
                    mujoco.mj_geomDistance(
                        model, data, handle_geom, geom_id, 0.05, None
                    )
                )
                for geom_id in geom_ids
            )
            for body_name, geom_ids in hand_geoms.items()
        }
        return {
            group: min(
                distance
                for body_name, distance in distances.items()
                if group in body_name
            )
            for group in ("thumb", "index", "middle")
        }

    # The rail-to-finger distance changes slightly as the wrist solver follows
    # the hinge.  A fixed hand target therefore alternates between a gap and an
    # excessive preload.  Solve two bounded one-dimensional closure controls
    # per frame: a shared index/middle proximal flexion and thumb distal
    # flexion.  This stays within the exact hand joint limits and makes the
    # intended force-closure supervision explicit instead of silently losing
    # the handle mid-pull.
    target_preload_distance_m = -0.0004
    hand_rows: list[np.ndarray] = []
    closure_search_rows: list[dict[str, Any]] = []
    for frame_index, (opening_angle, upper_body) in enumerate(
        zip(opening_angles, upper_body_rows)
    ):
        for address, value in zip(qpos_addresses, upper_body):
            data.qpos[address] = float(value)
        data.qpos[door_address] = float(opening_angle)
        base_hand = np.asarray(DEFAULT_RIGHT_HAND_GRASP, dtype=np.float64)
        for address, value in zip(hand_addresses, base_hand):
            data.qpos[address] = float(value)

        best_finger: tuple[float, float, dict[str, float]] | None = None
        for finger_flexion in np.linspace(0.82, 0.98, 65):
            data.qpos[hand_addresses[0]] = float(finger_flexion)
            data.qpos[hand_addresses[2]] = float(finger_flexion)
            distances = signed_group_distances()
            finger_distances = (distances["index"], distances["middle"])
            maximum_penetration = max(0.0, -min(finger_distances))
            score = sum(
                abs(distance - target_preload_distance_m)
                for distance in finger_distances
            ) + 100.0 * max(0.0, maximum_penetration - 0.002)
            candidate = (score, float(finger_flexion), distances)
            if best_finger is None or candidate[0] < best_finger[0]:
                best_finger = candidate
        assert best_finger is not None
        base_hand[0] = best_finger[1]
        base_hand[2] = best_finger[1]
        data.qpos[hand_addresses[0]] = best_finger[1]
        data.qpos[hand_addresses[2]] = best_finger[1]

        best_thumb: tuple[float, float, dict[str, float]] | None = None
        for thumb_distal_flexion in np.linspace(-0.82, -0.48, 69):
            data.qpos[hand_addresses[6]] = float(thumb_distal_flexion)
            distances = signed_group_distances()
            thumb_penetration = max(0.0, -distances["thumb"])
            score = abs(
                distances["thumb"] - target_preload_distance_m
            ) + 100.0 * max(0.0, thumb_penetration - 0.002)
            candidate = (score, float(thumb_distal_flexion), distances)
            if best_thumb is None or candidate[0] < best_thumb[0]:
                best_thumb = candidate
        assert best_thumb is not None
        base_hand[6] = best_thumb[1]
        for address, value in zip(hand_addresses, base_hand):
            data.qpos[address] = float(value)
        final_distances = signed_group_distances()
        hand_rows.append(base_hand)
        closure_search_rows.append(
            {
                "frame_index": frame_index,
                "door_angle_rad": float(opening_angle),
                "shared_index_middle_proximal_target_rad": best_finger[1],
                "thumb_distal_target_rad": best_thumb[1],
                "group_signed_distances_m": final_distances,
            }
        )

    contact_rows: list[dict[str, Any]] = []
    for frame_index, (opening_angle, arm, hand) in enumerate(
        zip(opening_angles, upper_body_rows, hand_rows)
    ):
        for address, value in zip(qpos_addresses, arm):
            data.qpos[address] = float(value)
        for address, value in zip(hand_addresses, hand):
            data.qpos[address] = float(value)
        data.qpos[door_address] = float(opening_angle)
        mujoco.mj_forward(model, data)
        distances = {
            body_name: min(
                float(
                    mujoco.mj_geomDistance(
                        model, data, handle_geom, geom_id, 0.05, None
                    )
                )
                for geom_id in geom_ids
            )
            for body_name, geom_ids in hand_geoms.items()
        }
        group_distances = {
            group: min(
                distance
                for body_name, distance in distances.items()
                if group in body_name
            )
            for group in ("thumb", "index", "middle")
        }
        panel_penetration = max(
            0.0,
            max(
                -float(
                    mujoco.mj_geomDistance(
                        model, data, panel_geom, geom_id, 0.05, None
                    )
                )
                for geom_ids in hand_geoms.values()
                for geom_id in geom_ids
            ),
        )
        handle_penetration = max(
            0.0, -min(group_distances.values())
        )
        if handle_penetration > float(maximum_handle_penetration_m):
            raise RuntimeError("g1_microwave_grasp_arc_handle_penetration_gate_failed")
        if panel_penetration > float(maximum_panel_penetration_m):
            raise RuntimeError("g1_microwave_grasp_arc_panel_penetration_gate_failed")
        contact_rows.append(
            {
                "frame_index": frame_index,
                "door_angle_rad": float(opening_angle),
                "group_signed_distances_m": group_distances,
                "all_three_groups_touching": all(
                    distance <= 0.0 for distance in group_distances.values()
                ),
                "all_three_groups_within_1mm": all(
                    distance <= 0.001 for distance in group_distances.values()
                ),
                "all_three_groups_within_contact_gap": all(
                    distance <= float(maximum_contact_gap_m)
                    for distance in group_distances.values()
                ),
                "maximum_handle_penetration_m": handle_penetration,
                "maximum_panel_penetration_m": panel_penetration,
            }
        )

    within_1mm_count = sum(
        row["all_three_groups_within_1mm"] is True for row in contact_rows
    )
    within_contact_gap_count = sum(
        row["all_three_groups_within_contact_gap"] is True
        for row in contact_rows
    )
    if within_contact_gap_count != len(contact_rows):
        maximum_group_gap = max(
            max(row["group_signed_distances_m"].values())
            for row in contact_rows
        )
        raise RuntimeError(
            "g1_microwave_grasp_arc_contact_coverage_gate_failed:"
            f"{within_contact_gap_count}/{len(contact_rows)}:"
            f"maximum_group_gap_m={maximum_group_gap:.9f}"
        )

    initial_by_name = dict(zip(PROTOCOL_V4_FULL_JOINT_ORDER, measured))
    initial_source = np.asarray(
        [initial_by_name[name] for name in SOURCE_ACTION_JOINT_NAMES]
    )
    source_index = {name: index for index, name in enumerate(SOURCE_ACTION_JOINT_NAMES)}
    oriented_open = initial_source.copy()
    for name, value in zip(UPPER_BODY_SOLVE_JOINT_NAMES, upper_body_rows[0]):
        oriented_open[source_index[name]] = float(value)
    grasped = oriented_open.copy()
    for name, value in zip(RIGHT_HAND_JOINT_NAMES, hand_rows[0]):
        grasped[source_index[name]] = float(value)
    reach = minimum_jerk_trajectory(
        initial_source, oriented_open, frame_count=int(reach_frame_count)
    )
    closure = minimum_jerk_trajectory(
        oriented_open, grasped, frame_count=int(closure_frame_count)
    )[1:]
    pull_rows: list[np.ndarray] = []
    for upper_body, hand in zip(upper_body_rows[1:], hand_rows[1:]):
        row = grasped.copy()
        for name, value in zip(UPPER_BODY_SOLVE_JOINT_NAMES, upper_body):
            row[source_index[name]] = float(value)
        for name, value in zip(RIGHT_HAND_JOINT_NAMES, hand):
            row[source_index[name]] = float(value)
        pull_rows.append(row)
    trajectory = np.concatenate((reach, closure, np.asarray(pull_rows))).astype(
        np.float32
    )
    trajectory_sha256 = hashlib.sha256(trajectory.tobytes(order="C")).hexdigest()
    allowed_to_change = {
        *range(12, 15),
        *range(29, 43),
    }
    unchanged = tuple(
        index for index in range(len(SOURCE_ACTION_JOINT_NAMES))
        if index not in allowed_to_change
    )
    upper_body_and_hand_only = bool(
        np.allclose(
            trajectory[:, unchanged],
            np.tile(initial_source[list(unchanged)], (trajectory.shape[0], 1)),
            rtol=0.0,
            atol=1e-7,
        )
    )
    if not upper_body_and_hand_only:
        raise AssertionError("g1_microwave_grasp_arc_non_upper_body_drift")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified_waist_assisted_grasp_and_prescribed_contact_arc_seed",
        "blockers": [
            "contact_driven_door_articulation_not_proven",
            "trained_microwave_checkpoint_not_produced",
            "semantic_episode_success_not_proven",
        ],
        "model": {
            "path": str(model_file),
            "sha256": _sha256(model_file),
            "wbc_source_revision": PINNED_WBC_SOURCE_REVISION,
        },
        "inputs": {
            "standing_initialization": {
                "path": str(standing_file),
                "sha256": _sha256(standing_file),
            },
            "initial_policy_observation": {
                "path": str(observation_file),
                "sha256": _sha256(observation_file),
            },
            "target_focus_report": {
                "path": str(focus_file),
                "sha256": _sha256(focus_file),
            },
        },
        "egocentric_review_camera": {
            "name": EGOCENTRIC_CAMERA_NAME,
            "mounted_body": "torso_link",
            "mount_semantics": "robot_head_mesh_eye_height_task_directed",
            "position_torso_xyz_m": list(EGOCENTRIC_CAMERA_POSITION_TORSO_M),
            "xyaxes": list(EGOCENTRIC_CAMERA_XYAXES),
            "vertical_field_of_view_deg": EGOCENTRIC_CAMERA_FOVY_DEG,
            "review_resolution_width": 640,
            "review_resolution_height": 480,
            "third_person_observer_camera": False,
        },
        "grasp_contract": {
            "wrist_body": WRIST_BODY_NAME,
            "handle_in_wrist_xyz_m": handle_in_wrist.tolist(),
            "hand_axis_polarity": float(hand_axis_polarity),
            "grasp_yaw_rad": float(grasp_yaw_rad),
            "right_hand_joint_names": list(RIGHT_HAND_JOINT_NAMES),
            "right_hand_joint_targets": list(DEFAULT_RIGHT_HAND_GRASP),
            "right_hand_initial_solved_targets": hand_rows[0].tolist(),
            "adaptive_force_closure_during_pull": True,
            "target_preload_distance_m": target_preload_distance_m,
            "shared_index_middle_proximal_target_range_rad": [
                float(min(row[0] for row in hand_rows)),
                float(max(row[0] for row in hand_rows)),
            ],
            "thumb_distal_target_range_rad": [
                float(min(row[6] for row in hand_rows)),
                float(max(row[6] for row in hand_rows)),
            ],
            "closure_search_frames": closure_search_rows,
            "upper_body_solve_joint_names": list(UPPER_BODY_SOLVE_JOINT_NAMES),
            "waist_assistance_enabled": True,
            "position_and_orientation_constrained": True,
        },
        "pose_solver": {
            "method": "waist_plus_right_arm_damped_least_squares_full_wrist_pose",
            "maximum_position_error_m": max(
                row["position_error_m"] for row in pose_rows
            ),
            "maximum_orientation_error_rad": max(
                row["orientation_error_rad"] for row in pose_rows
            ),
            "frames": pose_rows,
        },
        "contact_proxy": {
            "source": "exact_g1_collision_meshes_plus_usd_bound_derived_door_proxy",
            "handle_radius_m": geometry["handle_radius_m"],
            "handle_half_segment_m": geometry["handle_half_segment_m"],
            "frame_count": len(contact_rows),
            "all_three_groups_touching_frame_count": sum(
                row["all_three_groups_touching"] is True for row in contact_rows
            ),
            "all_three_groups_within_1mm_frame_count": within_1mm_count,
            "maximum_contact_gap_m": float(maximum_contact_gap_m),
            "all_three_groups_within_contact_gap_frame_count": (
                within_contact_gap_count
            ),
            "maximum_handle_penetration_m": max(
                row["maximum_handle_penetration_m"] for row in contact_rows
            ),
            "maximum_panel_penetration_m": max(
                row["maximum_panel_penetration_m"] for row in contact_rows
            ),
            "frames": contact_rows,
        },
        "pull": {
            "requested_door_open_observable_transition_rad": angle,
            "signed_prescribed_door_angle_rad": -angle,
            "frame_count_including_grasp_endpoint": int(pull_frame_count),
            "full_wrist_pose_arc_following_proven": True,
            "contact_proxy_coverage_preserved_within_2mm": True,
            "door_angle_was_prescribed_not_contact_driven": True,
        },
        "trajectory": {
            "frame_count": int(trajectory.shape[0]),
            "joint_count": int(trajectory.shape[1]),
            "fps": 50.0,
            "duration_seconds": float(trajectory.shape[0] / 50.0),
            "joint_names": list(SOURCE_ACTION_JOINT_NAMES),
            "sha256": trajectory_sha256,
            "only_waist_right_arm_and_right_hand_change": upper_body_and_hand_only,
        },
        "claim_boundary": {
            "full_wrist_pose_reach_proven": True,
            "waist_assisted_upper_body_route_proven": True,
            "right_hand_closure_supervision_present": True,
            "prescribed_arc_contact_proxy_coverage_proven": True,
            "contact_driven_door_articulation_proven": False,
            "isaac_usd_dynamic_episode_qualification_proven": False,
            "trained_checkpoint_produced": False,
            "task_success_proven": False,
        },
    }
    json.dumps(report, sort_keys=True)
    return trajectory, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an oriented G1 microwave grasp/contact-arc seed."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--standing-initialization", required=True)
    parser.add_argument("--initial-policy-observation", required=True)
    parser.add_argument("--target-focus-report", required=True)
    parser.add_argument("--trajectory-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--reach-frame-count", type=int, default=101)
    parser.add_argument("--closure-frame-count", type=int, default=26)
    parser.add_argument("--pull-frame-count", type=int, default=51)
    parser.add_argument("--door-open-angle-rad", type=float, default=0.35)
    args = parser.parse_args(list(argv) if argv is not None else None)
    trajectory_out = Path(args.trajectory_out).expanduser().resolve()
    report_out = Path(args.report_out).expanduser().resolve()
    trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        trajectory, report = solve_grasp_arc_seed(
            model_path=args.model,
            standing_initialization_path=args.standing_initialization,
            initial_policy_observation_path=args.initial_policy_observation,
            target_focus_report_path=args.target_focus_report,
            reach_frame_count=args.reach_frame_count,
            closure_frame_count=args.closure_frame_count,
            pull_frame_count=args.pull_frame_count,
            door_open_angle_rad=args.door_open_angle_rad,
        )
        np.save(trajectory_out, trajectory, allow_pickle=False)
        report["trajectory"]["path"] = str(trajectory_out)
        report["trajectory"]["file_sha256"] = _sha256(trajectory_out)
        exit_code = 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [f"grasp_arc_seed_failed:{type(exc).__name__}:{exc}"],
        }
        exit_code = 1
    report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
