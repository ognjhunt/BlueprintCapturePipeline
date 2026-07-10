"""Small, deterministic URDF/MJCF forward-kinematics validation substrate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

import numpy as np

from .camera_geometry_validation import matrix4, validate_se3_matrix


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _numbers(text: Any, count: int, default: Sequence[float]) -> list[float]:
    if isinstance(text, str):
        values: Sequence[Any] = text.replace(",", " ").split()
    elif isinstance(text, Sequence) and not isinstance(text, (bytes, bytearray)):
        values = text
    else:
        values = default
    try:
        parsed = [float(value) for value in values]
    except (TypeError, ValueError):
        parsed = list(default)
    return parsed if len(parsed) == count and all(math.isfinite(value) for value in parsed) else list(default)


def _translation(xyz: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("joint_axis_zero_length")
    x, y, z = vector / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    rotation = np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _axis_translation(axis: Sequence[float], distance: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("joint_axis_zero_length")
    return _translation((vector / norm * distance).tolist())


def _parse_urdf(model_path: Path) -> dict[str, Any]:
    root = ET.parse(model_path).getroot()
    if root.tag.rsplit("}", 1)[-1] != "robot":
        raise ValueError("not_urdf")
    links = [str(node.get("name") or "").strip() for node in root.findall("link")]
    links = [name for name in links if name]
    joints: list[dict[str, Any]] = []
    child_links: set[str] = set()
    for node in root.findall("joint"):
        name = str(node.get("name") or "").strip()
        joint_type = str(node.get("type") or "fixed").strip().lower()
        parent_node = node.find("parent")
        child_node = node.find("child")
        parent = str(parent_node.get("link") or "").strip() if parent_node is not None else ""
        child = str(child_node.get("link") or "").strip() if child_node is not None else ""
        if not name or not parent or not child or parent not in links or child not in links:
            raise ValueError("urdf_joint_link_reference_invalid")
        origin = node.find("origin")
        axis_node = node.find("axis")
        limit_node = node.find("limit")
        lower = None
        upper = None
        if joint_type not in {"fixed", "continuous"}:
            if limit_node is None or limit_node.get("lower") is None or limit_node.get("upper") is None:
                raise ValueError(f"joint_limit_missing:{name}")
            lower = float(limit_node.get("lower"))
            upper = float(limit_node.get("upper"))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"joint_limit_invalid:{name}")
        joints.append(
            {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _translation(_numbers(origin.get("xyz") if origin is not None else None, 3, [0, 0, 0]))
                @ _rpy(_numbers(origin.get("rpy") if origin is not None else None, 3, [0, 0, 0])),
                "axis": _numbers(axis_node.get("xyz") if axis_node is not None else None, 3, [1, 0, 0]),
                "lower": lower,
                "upper": upper,
            }
        )
        child_links.add(child)
    roots = [link for link in links if link not in child_links]
    if len(roots) != 1:
        raise ValueError("urdf_must_have_one_root_link")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        by_parent.setdefault(joint["parent"], []).append(joint)
    ordered: list[dict[str, Any]] = []

    def visit(link: str) -> None:
        for joint in by_parent.get(link, []):
            ordered.append(joint)
            visit(joint["child"])

    visit(roots[0])
    if len(ordered) != len(joints):
        raise ValueError("urdf_joint_tree_disconnected_or_cyclic")
    movable = [joint for joint in ordered if joint["type"] != "fixed"]
    unsupported = [joint["name"] for joint in movable if joint["type"] not in {"revolute", "continuous", "prismatic"}]
    if unsupported:
        raise ValueError("unsupported_urdf_joint_types:" + ",".join(unsupported))
    return {
        "format": "urdf",
        "base_frame": roots[0],
        "links": links,
        "joints": ordered,
        "movable_joint_names": [joint["name"] for joint in movable],
    }


def _parse_mjcf(model_path: Path) -> dict[str, Any]:
    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("mujoco_python_runtime_unavailable") from exc
    try:
        mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as exc:
        raise ValueError("mjcf_model_load_failed") from exc
    joints: list[dict[str, Any]] = []
    joint_names: list[str] = []
    for joint_id in range(int(mj_model.njnt)):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        name = str(name or "").strip()
        joint_type = int(mj_model.jnt_type[joint_id])
        if not name:
            raise ValueError("mjcf_joint_name_missing")
        if joint_type not in {2, 3}:
            raise ValueError(f"unsupported_mjcf_joint_type:{name}")
        limited = bool(mj_model.jnt_limited[joint_id])
        lower = float(mj_model.jnt_range[joint_id][0]) if limited else None
        upper = float(mj_model.jnt_range[joint_id][1]) if limited else None
        joints.append(
            {
                "name": name,
                "type": "prismatic" if joint_type == 2 else "revolute",
                "lower": lower,
                "upper": upper,
                "joint_id": joint_id,
                "qpos_address": int(mj_model.jnt_qposadr[joint_id]),
            }
        )
        joint_names.append(name)
    body_names: list[str] = []
    for body_id in range(1, int(mj_model.nbody)):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            body_names.append(str(name))
    if not body_names:
        raise ValueError("mjcf_named_bodies_missing")
    return {
        "format": "mjcf",
        "base_frame": body_names[0],
        "links": body_names,
        "joints": joints,
        "movable_joint_names": joint_names,
        "_mj_model": mj_model,
        "_mujoco": mujoco,
    }


def _urdf_fk(model: Mapping[str, Any], positions: Mapping[str, float], world_from_base: np.ndarray) -> dict[str, list[float]]:
    transforms: dict[str, np.ndarray] = {str(model["base_frame"]): world_from_base}
    for joint in model["joints"]:
        parent_transform = transforms[joint["parent"]]
        motion = np.eye(4, dtype=np.float64)
        value = float(positions.get(joint["name"], 0.0))
        if joint["type"] in {"revolute", "continuous"}:
            motion = _axis_rotation(joint["axis"], value)
        elif joint["type"] == "prismatic":
            motion = _axis_translation(joint["axis"], value)
        transforms[joint["child"]] = parent_transform @ joint["origin"] @ motion
    return {
        name: [round(float(value), 9) for value in transform[:3, 3]]
        for name, transform in transforms.items()
    }


def _mujoco_fk(model: Mapping[str, Any], positions: Mapping[str, float], world_from_base: np.ndarray) -> dict[str, list[float]]:
    mujoco = model["_mujoco"]
    mj_model = model["_mj_model"]
    data = mujoco.MjData(mj_model)
    for joint in model["joints"]:
        data.qpos[int(joint["qpos_address"])] = float(positions[joint["name"]])
    mujoco.mj_forward(mj_model, data)
    solved: dict[str, list[float]] = {}
    for body_id in range(1, int(mj_model.nbody)):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            continue
        homogeneous = np.asarray([*data.xpos[body_id], 1.0], dtype=np.float64)
        world_point = world_from_base @ homogeneous
        solved[str(name)] = [round(float(value), 9) for value in world_point[:3]]
    return solved


def _extract_state_frames(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("joint_state_frames", "states", "trajectory"):
        rows = state.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    if isinstance(state.get("joint_names"), list) and isinstance(state.get("joint_positions"), list):
        return [dict(state)]
    return []


def _validate_state_frames(
    *,
    state: Mapping[str, Any],
    model: Mapping[str, Any],
    expected_reference_frame: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    frames = _extract_state_frames(state)
    if not frames:
        return [], ["robot_joint_state_sequence_missing"]
    expected_names = list(model.get("movable_joint_names") or [])
    if not expected_names:
        blockers.append("robot_model_has_no_supported_movable_joints")
    angle_unit = str(state.get("angle_unit") or "").strip().lower()
    linear_unit = str(state.get("linear_unit") or "").strip().lower()
    timestamp_unit = str(state.get("timestamp_unit") or "").strip().lower()
    reference_frame = str(state.get("reference_frame") or "").strip()
    base_frame = str(state.get("base_frame") or "").strip()
    if angle_unit not in {"rad", "radian", "radians"}:
        blockers.append("robot_joint_angle_unit_missing_or_not_radians")
    if linear_unit not in {"m", "meter", "meters", "metre", "metres"}:
        blockers.append("robot_joint_linear_unit_missing_or_not_meters")
    if timestamp_unit not in {"s", "sec", "second", "seconds"}:
        blockers.append("robot_timestamp_unit_missing_or_not_seconds")
    if not reference_frame:
        blockers.append("robot_reference_frame_missing")
    if expected_reference_frame and reference_frame != expected_reference_frame:
        blockers.append("robot_camera_reference_frame_mismatch")
    if base_frame != str(model.get("base_frame") or ""):
        blockers.append("robot_base_frame_mismatch")
    normalized: list[dict[str, Any]] = []
    previous_timestamp: float | None = None
    for frame_index, frame in enumerate(frames):
        try:
            timestamp = float(frame.get("timestamp"))
        except (TypeError, ValueError):
            blockers.append(f"robot_timestamp_non_numeric:frame_{frame_index}")
            timestamp = float("nan")
        if not math.isfinite(timestamp):
            blockers.append(f"robot_timestamp_nonfinite:frame_{frame_index}")
        elif previous_timestamp is not None and timestamp <= previous_timestamp:
            blockers.append(f"robot_timestamps_not_strictly_monotonic:frame_{frame_index}")
        if math.isfinite(timestamp):
            previous_timestamp = timestamp
        names = frame.get("joint_names") if isinstance(frame.get("joint_names"), list) else state.get("joint_names")
        positions = frame.get("joint_positions") if isinstance(frame.get("joint_positions"), list) else None
        if names != expected_names:
            blockers.append(f"robot_joint_name_order_mismatch:frame_{frame_index}")
            continue
        if not isinstance(positions, list) or len(positions) != len(expected_names):
            blockers.append(f"robot_joint_position_width_mismatch:frame_{frame_index}")
            continue
        try:
            numeric_positions = [float(value) for value in positions]
        except (TypeError, ValueError):
            blockers.append(f"robot_joint_state_non_numeric:frame_{frame_index}")
            continue
        if not all(math.isfinite(value) for value in [*numeric_positions, timestamp]):
            blockers.append(f"robot_joint_state_nonfinite:frame_{frame_index}")
            continue
        position_map = dict(zip(expected_names, numeric_positions))
        for joint in model.get("joints", []):
            if joint.get("name") not in position_map:
                continue
            value = position_map[joint["name"]]
            if joint.get("lower") is not None and value < float(joint["lower"]) - 1e-9:
                blockers.append(f"robot_joint_below_limit:{joint['name']}:frame_{frame_index}")
            if joint.get("upper") is not None and value > float(joint["upper"]) + 1e-9:
                blockers.append(f"robot_joint_above_limit:{joint['name']}:frame_{frame_index}")
        base_raw = frame.get("world_from_robot_base", state.get("world_from_robot_base"))
        base_validation = validate_se3_matrix(base_raw, field="world_from_robot_base")
        if not base_validation["valid"]:
            blockers.extend(f"{reason}:frame_{frame_index}" for reason in base_validation["blockers"])
            continue
        expected_positions = frame.get("expected_link_positions")
        if not isinstance(expected_positions, Mapping) or not expected_positions:
            blockers.append(f"fk_reference_landmarks_missing:frame_{frame_index}")
            continue
        normalized.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "positions": position_map,
                "world_from_robot_base": matrix4(base_validation["matrix"]),
                "expected_link_positions": dict(expected_positions),
            }
        )
    if len(normalized) != len(frames):
        blockers.append("not_all_robot_state_steps_validated")
    return normalized, list(dict.fromkeys(blockers))


def _reference_error(
    solved: Mapping[str, Sequence[float]],
    expected: Mapping[str, Any],
) -> tuple[float | None, list[str]]:
    errors: list[float] = []
    blockers: list[str] = []
    for link_name, raw_point in expected.items():
        if link_name not in solved:
            blockers.append(f"fk_reference_link_missing:{link_name}")
            continue
        point = _numbers(raw_point, 3, [float("nan")] * 3)
        if not all(math.isfinite(value) for value in point):
            blockers.append(f"fk_reference_point_invalid:{link_name}")
            continue
        errors.append(float(np.linalg.norm(np.asarray(solved[link_name]) - np.asarray(point))))
    return (max(errors) if errors else None), blockers


def solve_robot_forward_kinematics(
    *,
    model_path: Path | None,
    state_payload: Mapping[str, Any],
    expected_reference_frame: str | None,
) -> dict[str, Any]:
    """Solve every aligned state and verify it against known link landmarks."""

    blockers: list[str] = []
    if model_path is None or not model_path.is_file():
        return {
            "status": "blocked",
            "solver_executed": False,
            "model_format": None,
            "frames": [],
            "blockers": ["missing_robot_urdf_or_mjcf"],
        }
    try:
        root_tag = ET.parse(model_path).getroot().tag.rsplit("}", 1)[-1]
    except (ET.ParseError, DefusedXmlException, OSError):
        return {
            "status": "blocked",
            "solver_executed": False,
            "model_format": None,
            "frames": [],
            "blockers": ["robot_model_xml_invalid"],
        }
    if root_tag not in {"robot", "mujoco"}:
        return {
            "status": "blocked",
            "solver_executed": False,
            "model_format": None,
            "frames": [],
            "blockers": ["robot_model_format_not_urdf_or_mjcf"],
        }
    try:
        model = _parse_urdf(model_path) if root_tag == "robot" else _parse_mjcf(model_path)
    except (
        ET.ParseError,
        DefusedXmlException,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "status": "blocked",
            "solver_executed": False,
            "model_format": "urdf" if root_tag == "robot" else "mjcf",
            "frames": [],
            "blockers": [f"robot_model_parse_or_validation_failed:{exc}"],
        }
    state_frames, state_blockers = _validate_state_frames(
        state=state_payload,
        model=model,
        expected_reference_frame=expected_reference_frame,
    )
    blockers.extend(state_blockers)
    solved_frames: list[dict[str, Any]] = []
    solver_executed = False
    tolerance = state_payload.get("reference_tolerance_m", 0.01)
    try:
        tolerance_m = float(tolerance)
    except (TypeError, ValueError):
        tolerance_m = -1.0
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0 or tolerance_m > 0.1:
        blockers.append("fk_reference_tolerance_invalid")
    if not blockers:
        for frame in state_frames:
            try:
                solved = (
                    _urdf_fk(model, frame["positions"], frame["world_from_robot_base"])
                    if model.get("format") == "urdf"
                    else _mujoco_fk(model, frame["positions"], frame["world_from_robot_base"])
                )
                solver_executed = True
            except (KeyError, TypeError, ValueError) as exc:
                blockers.append(f"fk_solver_failed:frame_{frame['frame_index']}:{exc}")
                continue
            max_error, reference_blockers = _reference_error(solved, frame["expected_link_positions"])
            blockers.extend(reference_blockers)
            if max_error is None or max_error > tolerance_m:
                blockers.append(f"fk_reference_error_exceeded:frame_{frame['frame_index']}")
            solved_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "link_positions": solved,
                    "reference_max_error_m": max_error,
                }
            )
    if len(solved_frames) != len(state_frames):
        blockers.append("fk_not_solved_for_every_aligned_step")
    max_step_m = 0.0
    for previous, current in zip(solved_frames, solved_frames[1:]):
        common = set(previous["link_positions"]) & set(current["link_positions"])
        for link_name in common:
            delta = float(
                np.linalg.norm(
                    np.asarray(current["link_positions"][link_name])
                    - np.asarray(previous["link_positions"][link_name])
                )
            )
            max_step_m = max(max_step_m, delta)
    continuity_limit = float(state_payload.get("max_link_step_m", 1.0) or 1.0)
    if solved_frames and (not math.isfinite(continuity_limit) or continuity_limit <= 0 or max_step_m > continuity_limit):
        blockers.append("fk_link_continuity_failed")
    blockers = list(dict.fromkeys(blockers))
    completed = bool(solver_executed and solved_frames and not blockers)
    return {
        "status": "completed" if completed else "blocked",
        "solver_executed": solver_executed,
        "solver_name": (
            "blueprint_urdf_tree_fk.v1"
            if solver_executed and model.get("format") == "urdf"
            else "mujoco_mj_forward"
            if solver_executed
            else None
        ),
        "model_format": model.get("format"),
        "base_frame": model.get("base_frame"),
        "joint_names": model.get("movable_joint_names"),
        "frame_count": len(solved_frames),
        "input_frame_count": len(state_frames),
        "frames": solved_frames,
        "reference_tolerance_m": tolerance_m,
        "max_link_step_m": max_step_m,
        "blockers": blockers,
    }


__all__ = ["solve_robot_forward_kinematics"]
