"""Pure family scene/source validation for read-only and execution callers.

No provider access, file writes, reconstruction, or execution imports belong in
this module. Runtime materialization and episode effects remain in the adapter.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .task_family_scoring import canonical_task_family, validate_task_family_contract

RUNTIME_SCHEMA = "task_family_native_runtime_contract.v1"


_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


_ASSET_ROLES = {"pouring": {"source", "receiver", "material"}, "wiping": {"surface"},
                "insertion": {"peg", "socket"}, "stacking": {"object", "support"},
                "button_press": {"button"}, "cloth_folding": {"cloth"}}


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("task_family_native_" + code)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def validate_task_family_native_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "runtime_contract_missing")
    value = _json(value)
    _require(value.get("schema_version") == RUNTIME_SCHEMA, "runtime_schema_invalid")
    _require(value.get("contract_digest") == canonical_digest(value, digest_field="contract_digest"),
             "runtime_digest_invalid")
    spec = value.get("task_spec")
    _require(isinstance(spec, dict), "task_spec_invalid")
    task = validate_task_family_contract(spec.get("task_success_contract"))
    _require(canonical_task_family(spec.get("task_kind")) == task["task_family"], "task_family_mismatch")
    _require(all(value.get(key) == task["scope"][key] for key in ("scene_id", "task_id")),
             "task_scope_mismatch")
    _require(value.get("task_family") == task["task_family"] and value.get("task_kind") == task["task_family"]
             and value.get("runtime_identity_digest") == task["runtime_identity_digest"], "runtime_task_binding_invalid")
    binding = value.get("task_state_binding")
    _require(isinstance(binding, dict) and binding.get("source") == "native_simulator_state"
             and isinstance(binding.get("engine_id"), str) and bool(binding["engine_id"])
             and binding.get("runtime_identity_digest") == task["runtime_identity_digest"],
             "state_binding_invalid")
    assets = value.get("assets")
    _require(isinstance(assets, list) and bool(assets) and all(isinstance(row, dict) for row in assets),
             "assets_invalid")
    ids = [row.get("asset_id") for row in assets]
    _require(all(isinstance(key, str) and _ID.fullmatch(key) for key in ids) and len(set(ids)) == len(ids),
             "asset_ids_invalid")
    for asset in assets:
        _require(bool(_SHA.fullmatch(str(asset.get("content_digest", ""))))
                 and isinstance(asset.get("reset_state"), dict) and bool(asset["reset_state"]),
                 "asset_identity_or_reset_missing")
    stages = binding.get("stage_bindings")
    _require(isinstance(stages, dict) and set(stages) == {row["stage_id"] for row in task["stages"]},
             "stage_bindings_invalid")
    for stage in task["stages"]:
        row = stages[stage["stage_id"]]
        _require(isinstance(row, dict) and row.get("native_readback_kind") == stage["task_family"]
                 and isinstance(row.get("asset_ids"), list) and bool(row["asset_ids"])
                 and set(row["asset_ids"]) <= set(ids), "stage_asset_binding_invalid")
        roles = row.get("asset_roles")
        _require(isinstance(roles, dict) and set(roles) == _ASSET_ROLES[stage["task_family"]]
                 and set(roles.values()) <= set(row["asset_ids"]), "stage_asset_roles_invalid")
        _validate_measurement_geometry(stage["task_family"], row.get("measurement_geometry"))
    base, joints, cameras = (value.get(key) for key in
                             ("robot_base_pose_world", "robot_joint_reset_positions_rad", "cameras"))
    _require(isinstance(base, dict) and bool(base) and isinstance(joints, dict) and bool(joints)
             and all(isinstance(key, str) and key and _number(position) for key, position in joints.items()),
             "robot_reset_invalid")
    _pose(base)
    configuration = value.get("robot_configuration")
    _require(isinstance(configuration, dict) and all(isinstance(configuration.get(key), str) and configuration[key]
             for key in ("embodiment_id", "controller", "observation_schema_id", "action_schema_id")),
             "robot_configuration_missing")
    _require(isinstance(cameras, list) and bool(cameras) and all(isinstance(row, dict) and row for row in cameras),
             "cameras_missing")
    camera_ids = [row.get("camera_id") for row in cameras]
    _require(all(isinstance(key, str) and _ID.fullmatch(key) for key in camera_ids)
             and len(set(camera_ids)) == len(camera_ids), "camera_ids_invalid")
    for camera in cameras:
        _pose(camera.get("pose_world"))
        intrinsics = _array(camera.get("intrinsics"), columns=3)
        resolution = camera.get("resolution")
        _require(intrinsics.shape == (3, 3) and intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0
                 and np.allclose(intrinsics[2], [0, 0, 1], atol=1e-12, rtol=0)
                 and isinstance(resolution, list) and len(resolution) == 2
                 and all(type(size) is int and size > 0 for size in resolution), "camera_calibration_invalid")
    _require(isinstance(value.get("scenario_cell_id"), str) and bool(value["scenario_cell_id"])
             and bool(_SHA.fullmatch(str(value.get("scenario_instance_digest", "")))), "scenario_invalid")
    _require(type(value.get("seed")) is int and value["seed"] >= 0, "seed_invalid")
    _require(value.get("scenario_context_kind") in {"evaluation_cell", "construction_canary", "canonical_anchor", "control_cell"},
             "scenario_context_invalid")
    _require(value.get("scoring_source") == "deterministic_native_state"
             and value.get("physical_validation") is False, "claim_boundary_invalid")
    return value


def _array(value: Any, *, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    _require(result.dtype.kind in "fiu" and result.size > 0 and np.all(np.isfinite(result)),
             "readback_array_invalid")
    result = result.astype(np.float64)
    if columns is not None:
        _require(result.ndim == 2 and result.shape[1] == columns, "readback_array_shape_invalid")
    return result


def _vector(value: Any) -> np.ndarray:
    result = _array(value)
    _require(result.shape == (3,), "readback_vector_invalid")
    return result


def _unit(value: Any) -> np.ndarray:
    vector = _vector(value)
    norm = float(np.linalg.norm(vector))
    _require(norm > 0, "readback_axis_invalid")
    return vector / norm


def _pose(value: Any) -> tuple[np.ndarray, np.ndarray]:
    _require(isinstance(value, Mapping), "pose_invalid")
    position = _vector(value.get("position_world_m"))
    q = _array(value.get("orientation_xyzw"))
    _require(q.shape == (4,) and abs(float(q @ q) - 1) <= 1e-6, "pose_quaternion_invalid")
    x, y, z, w = q
    rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    return position, rotation


def _validate_measurement_geometry(kind: str, value: Any) -> None:
    _require(isinstance(value, Mapping), "measurement_geometry_missing")
    if kind == "pouring":
        for key in ("source_local_bounds_m", "receiver_local_bounds_m", "transfer_workspace_bounds_world_m"):
            bounds = _array(value.get(key), columns=3)
            _require(bounds.shape == (2, 3) and np.all(bounds[1] > bounds[0]), "container_geometry_invalid")
    elif kind == "wiping":
        areas = _array(value.get("cell_areas_m2"))
        _require(areas.ndim == 1 and np.all(areas > 0), "residue_cell_geometry_invalid")
    elif kind == "insertion":
        for key in ("peg_tip_local_m", "socket_entry_local_m"):
            _vector(value.get(key))
        for key in ("peg_axis_local", "socket_axis_local"):
            _unit(value.get(key))
    elif kind == "stacking":
        _polygon(value.get("support_polygon_local_xy_m"))
    elif kind == "button_press":
        _require(_number(value.get("joint_rest_position_m")) and _number(value.get("press_direction"))
                 and value["press_direction"] in {-1, 1},
                 "button_geometry_invalid")
    elif kind == "cloth_folding":
        _pose(value.get("fold_plane_pose_world"))
        targets = _array(value.get("target_keypoint_positions_m"), columns=3)
        for key in ("keypoint_vertex_indices", "first_fold_vertex_indices", "second_fold_vertex_indices"):
            indices = value.get(key)
            _require(isinstance(indices, list) and bool(indices) and all(type(i) is int and i >= 0 for i in indices)
                     and len(set(indices)) == len(indices), "cloth_indices_invalid")
        _require(len(value["keypoint_vertex_indices"]) == len(targets), "cloth_keypoint_geometry_invalid")


def _polygon(value: Any) -> np.ndarray:
    polygon = _array(value, columns=2)
    _require(len(polygon) >= 3, "polygon_invalid")
    signed = sum(np.cross(polygon[(i + 1) % len(polygon)] - polygon[i],
                          polygon[(i + 2) % len(polygon)] - polygon[(i + 1) % len(polygon)])
                 for i in range(len(polygon)))
    if signed < 0:
        polygon = polygon[::-1]
    edges = np.roll(polygon, -1, axis=0) - polygon
    _require(np.all(np.linalg.norm(edges, axis=1) > 0), "polygon_degenerate")
    _require(all(float(np.cross(edges[i], edges[(i + 1) % len(edges)])) >= -1e-12
                 for i in range(len(edges))), "polygon_not_convex")
    _require(_area(polygon) > 0, "polygon_degenerate")
    return polygon


def _area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                     - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2)


def family_source_blockers(*, native: Mapping[str, Any], source_bindings: Mapping[str, Any],
                           source_scene: Mapping[str, Any]) -> list[str]:
    """Apply the same source/frame requirements at compile and retained read."""
    blockers: list[str] = []
    unit = source_scene.get("meters_per_unit")
    if (isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit <= 0
            or source_scene.get("up_axis") not in {"Y", "-Y", "Z"}):
        raise ValueError("website_task_family_source_frame_invalid")
    binding = native["task_state_binding"].get("source_scene_binding")
    if not isinstance(binding, Mapping):
        blockers.append("website_task_family_source_scene_binding_required")
    elif any(binding.get(key) != source_bindings.get(key) for key in (
            "task_context_digest", "source_geometry_digest", "task_masks_digest", "splat_digest", "collision_mesh_digest")):
        raise ValueError("website_task_family_source_scene_binding_mismatch")
    else:
        frame = binding.get("frame_registration")
        if not isinstance(frame, Mapping):
            blockers.append("website_task_family_frame_registration_required")
        else:
            transform = np.asarray(frame.get("source_to_runtime"), dtype=float)
            linear = transform[:3, :3] if transform.shape == (4, 4) else np.zeros((3, 3))
            gram = linear.T @ linear
            if (frame.get("source_geometry_digest") != source_bindings.get("source_geometry_digest")
                    or frame.get("collision_mesh_digest") != source_scene["collision_mesh_digest"]
                    or frame.get("runtime_up_axis") != source_scene["up_axis"]
                    or frame.get("runtime_meters_per_unit") != source_scene.get("meters_per_unit")
                    or frame.get("registration_digest") != canonical_digest(frame, digest_field="registration_digest")
                    or transform.shape != (4, 4) or not np.all(np.isfinite(transform))
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or np.linalg.det(linear) <= 1e-12
                    or not np.allclose(gram, np.eye(3) * np.trace(gram) / 3, rtol=1e-8, atol=1e-10)):
                raise ValueError("website_task_family_frame_registration_invalid")
    for role, digest in (("scene_appearance", source_scene["splat_digest"]),
                         ("scene_collision", source_scene["collision_mesh_digest"])):
        if not any(row.get("semantic_role") == role and row.get("content_digest") == digest for row in native["assets"]):
            blockers.append("website_task_family_" + role + "_asset_required")
    return blockers
