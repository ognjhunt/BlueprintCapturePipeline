"""Derive articulated task scoring state only from native numeric readback.

The scorer needs booleans such as ``task_contact_active`` and
``retreat_completed``.  Accepting those booleans from a runtime caller would
let the component being evaluated grade itself, so this module accepts only
joint state, filtered contact-force vectors, root poses, and measured reference
positions.  It deterministically derives the scorer sample or fails closed on
a missing sensor.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "native_articulated_task_state.v1"


class NativeArticulatedTaskStateError(ValueError):
    """Stable native-state failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _vector(value: Any, *, length: int, error: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeArticulatedTaskStateError([error]) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise NativeArticulatedTaskStateError([error])
    return result


def _force_peak(value: Any, *, sensor_id: str) -> float:
    if value is None or isinstance(value, (str, bytes)):
        raise NativeArticulatedTaskStateError(
            [f"native_articulated_task_contact_readback_missing:{sensor_id}"]
        )
    try:
        rows = list(value)
    except TypeError as exc:
        raise NativeArticulatedTaskStateError(
            [f"native_articulated_task_contact_readback_missing:{sensor_id}"]
        ) from exc
    peak = 0.0
    for index, raw in enumerate(rows):
        vector = _vector(
            raw,
            length=3,
            error=f"native_articulated_task_contact_readback_invalid:{sensor_id}:{index}",
        )
        peak = max(peak, math.sqrt(sum(component * component for component in vector)))
    return peak


def _quaternion_angle(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _vector(a, length=4, error="native_articulated_task_root_pose_invalid")
    qb = _vector(b, length=4, error="native_articulated_task_root_pose_invalid")
    norm_a = math.sqrt(sum(value * value for value in qa))
    norm_b = math.sqrt(sum(value * value for value in qb))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise NativeArticulatedTaskStateError(
            ["native_articulated_task_root_pose_invalid"]
        )
    dot = abs(sum(qa[index] * qb[index] for index in range(4)) / (norm_a * norm_b))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def compile_native_articulated_task_sample(
    *,
    task_spec: Mapping[str, Any],
    task_sample_binding: Mapping[str, Any],
    task_state_binding: Mapping[str, Any],
    native_joint_names: Sequence[str],
    native_joint_positions_rad: Sequence[float],
    native_joint_velocities_rad_s: Sequence[float],
    task_robot_contact_forces_w_n: Sequence[Sequence[float]] | None,
    task_scene_contact_forces_w_n: Sequence[Sequence[float]] | None,
    robot_scene_contact_forces_w_n: Sequence[Sequence[float]] | None,
    task_root_pose_world: Sequence[float],
    task_root_reset_pose_world: Sequence[float],
    grasp_frame_position_world_m: Sequence[float],
    handle_reference_position_world_m: Sequence[float],
) -> dict[str, Any]:
    """Compile one exact scorer sample from native readback quantities."""

    if task_spec.get("task_kind") != "articulated_open_close":
        raise NativeArticulatedTaskStateError(
            ["native_articulated_task_kind_invalid"]
        )
    names = [str(name) for name in native_joint_names]
    positions = _vector(
        native_joint_positions_rad,
        length=len(names),
        error="native_articulated_task_joint_positions_invalid",
    )
    velocities = _vector(
        native_joint_velocities_rad_s,
        length=len(names),
        error="native_articulated_task_joint_velocities_invalid",
    )
    if len(set(names)) != len(names):
        raise NativeArticulatedTaskStateError(
            ["native_articulated_task_joint_names_duplicated"]
        )
    index = {name: offset for offset, name in enumerate(names)}
    prim_paths = dict(task_sample_binding.get("joint_prim_paths") or {})
    expected_ids = list(task_sample_binding.get("joint_ids") or [])
    joint_positions: dict[str, float] = {}
    joint_velocities: dict[str, float] = {}
    errors: list[str] = []
    for joint_id in expected_ids:
        native_name = str(prim_paths.get(joint_id) or "").rsplit("/", 1)[-1]
        if not native_name or native_name not in index:
            errors.append(f"native_articulated_task_joint_unresolved:{joint_id}")
            continue
        joint_positions[str(joint_id)] = positions[index[native_name]]
        joint_velocities[str(joint_id)] = velocities[index[native_name]]
    reset_ids = set((task_spec.get("joint_reset_positions_rad") or {}).keys())
    if set(joint_positions) != reset_ids:
        errors.append("native_articulated_task_joint_set_mismatch")
    if errors:
        raise NativeArticulatedTaskStateError(errors)

    task_contact_peak = _force_peak(
        task_robot_contact_forces_w_n, sensor_id="task_robot_contact"
    )
    task_scene_peak = _force_peak(
        task_scene_contact_forces_w_n, sensor_id="task_scene_contact"
    )
    robot_scene_peak = _force_peak(
        robot_scene_contact_forces_w_n, sensor_id="robot_scene_contact"
    )
    task_contact = task_contact_peak >= float(
        task_state_binding["task_contact_minimum_force_n"]
    )
    collision_threshold = float(
        task_state_binding["collision_failure_minimum_force_n"]
    )

    root = _vector(
        task_root_pose_world,
        length=7,
        error="native_articulated_task_root_pose_invalid",
    )
    reset_root = _vector(
        task_root_reset_pose_world,
        length=7,
        error="native_articulated_task_root_pose_invalid",
    )
    translation_delta = math.dist(root[:3], reset_root[:3])
    orientation_delta = _quaternion_angle(root[3:], reset_root[3:])
    containment_violation = (
        translation_delta
        > float(task_state_binding["root_translation_tolerance_m"])
        or orientation_delta
        > float(task_state_binding["root_orientation_tolerance_rad"])
    )

    hard_limits = task_spec.get("joint_hard_limits_rad") or {}
    joint_limit_violation = any(
        joint_positions[joint_id] < float(hard_limits[joint_id][0]) - 1.0e-6
        or joint_positions[joint_id] > float(hard_limits[joint_id][1]) + 1.0e-6
        for joint_id in joint_positions
    )
    grasp = _vector(
        grasp_frame_position_world_m,
        length=3,
        error="native_articulated_task_grasp_frame_position_invalid",
    )
    handle = _vector(
        handle_reference_position_world_m,
        length=3,
        error="native_articulated_task_handle_position_invalid",
    )
    retreat_separation = math.dist(grasp, handle)
    retreat_completed = (
        not task_contact
        and retreat_separation
        >= float(task_state_binding["retreat_minimum_separation_m"])
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "joint_positions_rad": joint_positions,
        "joint_velocities_rad_s": joint_velocities,
        "task_contact_active": task_contact,
        "joint_limit_violation": joint_limit_violation,
        "containment_violation": containment_violation,
        "robot_collision_failure": robot_scene_peak >= collision_threshold,
        "scene_collision_failure": task_scene_peak >= collision_threshold,
        "retreat_completed": retreat_completed,
        "native_readback": {
            "task_robot_contact_peak_force_n": task_contact_peak,
            "task_scene_contact_peak_force_n": task_scene_peak,
            "robot_scene_contact_peak_force_n": robot_scene_peak,
            "root_translation_delta_m": translation_delta,
            "root_orientation_delta_rad": orientation_delta,
            "grasp_frame_to_handle_separation_m": retreat_separation,
            "caller_asserted_booleans_used": False,
        },
    }


__all__ = [
    "NativeArticulatedTaskStateError",
    "SCHEMA_VERSION",
    "compile_native_articulated_task_sample",
]
