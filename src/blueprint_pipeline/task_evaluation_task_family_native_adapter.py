"""Native readback, independent scoring, and episode execution for beta families.

This adapter consumes engine state through a trusted runtime object. It does not
load arbitrary code, allocate resources, or turn an unimplemented material model
into a simulation. Its scene contract keeps source assets, reset, robot and
camera identities separate from task success, including for humanoid runtimes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import finalize_visual_evidence, persist_observation_frame
from .task_family_scoring import (
    TELEMETRY_SCHEMA,
    canonical_task_family,
    score_task_family_episode,
    validate_task_family_contract,
)

RUNTIME_SCHEMA = "task_family_native_runtime_contract.v1"
RECEIPT_SCHEMA = "task_family_native_episode.v1"
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


def materialize_task_family_native_runtime_contract(
    *, scene_id: str, task_id: str, task_spec: Mapping[str, Any],
    task_state_binding: Mapping[str, Any] | None, assets: Sequence[Mapping[str, Any]],
    robot_base_pose_world: Mapping[str, Any], robot_joint_reset_positions_rad: Mapping[str, float],
    cameras: Sequence[Mapping[str, Any]], scenario_cell_id: str, scenario_instance_digest: str,
    seed: int, robot_configuration: Mapping[str, Any] | None = None,
    scenario_context_kind: str = "evaluation_cell", destination: str | Path | None = None,
    **additional_bindings: Any,
) -> dict[str, Any]:
    """Freeze family-native inputs without applying rigid-only asset assumptions."""
    value = {"schema_version": RUNTIME_SCHEMA, "scene_id": scene_id, "task_id": task_id,
             "task_family": task_spec.get("task_kind"), "task_kind": task_spec.get("task_kind"),
             "runtime_identity_digest": (task_state_binding or {}).get("runtime_identity_digest"),
             "task_spec": task_spec, "task_state_binding": task_state_binding,
             "assets": assets, "robot_base_pose_world": robot_base_pose_world,
             "robot_joint_reset_positions_rad": robot_joint_reset_positions_rad,
             "robot_configuration": robot_configuration, "cameras": cameras,
             "scenario_cell_id": scenario_cell_id, "scenario_instance_digest": scenario_instance_digest,
             "seed": seed, "scenario_context_kind": scenario_context_kind,
             "additional_bindings": additional_bindings,
             "scoring_source": "deterministic_native_state", "physical_validation": False}
    value = _json(value)
    value["contract_digest"] = canonical_digest(value, digest_field="contract_digest")
    value = validate_task_family_native_runtime_contract(value)
    if destination is not None:
        write_json(Path(destination), value)
    return value


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


def _forces(row: Mapping[str, Any]) -> float:
    if np.asarray(row["contact_forces_n"]).shape in {(0,), (0, 3)}:
        return 0.0
    forces = _array(row["contact_forces_n"], columns=3)
    return float(np.linalg.norm(forces, axis=1).sum())


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


def _intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        points, output = output, []
        if not points:
            break
        start = points[-1]
        for end in points:
            s = float(np.cross(b - a, start - a))
            e = float(np.cross(b - a, end - a))
            if (e >= 0) != (s >= 0):
                output.append(start + (end - start) * s / (s - e))
            if e >= 0:
                output.append(end)
            start = end
    return np.asarray(output)


def _cloth_overlap_fraction(first: Any, second: Any) -> float:
    """Score finite intermediate fold projections conservatively.

    An edge-on or concave cloth half is a normal intermediate state. The
    convex clipping method cannot establish overlap there, so it contributes
    zero progress. Malformed or nonfinite native geometry remains invalid.
    """
    projected = [_array(value, columns=2) for value in (first, second)]
    _require(all(len(polygon) >= 3 for polygon in projected), "polygon_invalid")
    polygons = []
    for polygon in projected:
        extent = float(np.ptp(polygon, axis=0).max())
        if _area(polygon) <= 1e-12 * extent * extent:
            return 0.0
        try:
            polygons.append(_polygon(polygon))
        except ValueError as exc:
            if str(exc) not in {"task_family_native_polygon_degenerate", "task_family_native_polygon_not_convex"}:
                raise
            return 0.0
    return min(1.0, _area(_intersection(*polygons)) / max(_area(polygon) for polygon in polygons))


def _stage_measurements(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "pouring":
        positions, volumes = _array(row["particle_positions_m"], columns=3), _array(row["particle_volumes_ml"])
        _require(volumes.shape == (len(positions),) and np.all(volumes > 0), "particle_volumes_invalid")
        def inside(bounds: Any, pose: Any) -> np.ndarray:
            bounds = _array(bounds, columns=3)
            _require(bounds.shape == (2, 3) and np.all(bounds[1] > bounds[0]), "container_bounds_invalid")
            position, rotation = _pose(pose)
            local = (positions - position) @ rotation
            return np.all((local >= bounds[0]) & (local <= bounds[1]), axis=1)
        source = inside(row["source_local_bounds_m"], row["source_pose_world"])
        receiver = inside(row["receiver_local_bounds_m"], row["receiver_pose_world"])
        _require(not np.any(source & receiver), "container_membership_ambiguous")
        workspace = _array(row["transfer_workspace_bounds_world_m"], columns=3)
        transit_region = np.all((positions >= workspace[0]) & (positions <= workspace[1]), axis=1)
        in_flight = ~(source | receiver) & transit_region
        spilled = ~(source | receiver | in_flight)
        return {"source_volume_ml": float(volumes[source].sum()),
                "receiver_volume_ml": float(volumes[receiver].sum()),
                "in_flight_volume_ml": float(volumes[in_flight].sum()),
                "spilled_volume_ml": float(volumes[spilled].sum())}
    if kind == "wiping":
        areas, dirt = _array(row["cell_areas_m2"]), _array(row["cell_dirt_fractions"])
        _require(areas.ndim == 1 and areas.shape == dirt.shape and np.all(areas > 0)
                 and np.all((dirt >= 0) & (dirt <= 1)), "residue_grid_invalid")
        return {"dirty_area_m2": float(np.dot(areas, dirt))}
    if kind == "insertion":
        axis = _unit(row["socket_axis_world"])
        delta = _vector(row["peg_tip_position_m"]) - _vector(row["socket_entry_position_m"])
        depth = float(delta @ axis)
        return {"insertion_depth_m": max(0.0, depth),
                "lateral_error_m": float(np.linalg.norm(delta - depth * axis)),
                "alignment_error_rad": math.acos(float(np.clip(_unit(row["peg_axis_world"]) @ axis, -1, 1))),
                "contact_force_n": _forces(row)}
    if kind == "stacking":
        polygon = _polygon(row["support_polygon_xy_m"])
        center = _vector(row["center_of_mass_m"])[:2]
        edges = np.roll(polygon, -1, axis=0) - polygon
        margin = min(float(np.cross(edge, center - point) / np.linalg.norm(edge))
                     for edge, point in zip(edges, polygon))
        contacts = row["support_normal_forces_n"]
        _require(isinstance(contacts, list) and all(_number(force) and force >= 0 for force in contacts),
                 "support_contacts_invalid")
        _require(type(row.get("robot_contact_active")) is bool, "robot_contact_readback_missing")
        return {"support_contact": any(force > 0 for force in contacts), "com_support_margin_m": margin,
                "robot_contact_active": row["robot_contact_active"],
                "linear_speed_m_s": float(np.linalg.norm(_vector(row["linear_velocity_m_s"]))),
                "angular_speed_rad_s": float(np.linalg.norm(_vector(row["angular_velocity_rad_s"])))}
    if kind == "button_press":
        _require(_number(row["joint_position_m"]) and _number(row["joint_rest_position_m"])
                 and type(row["native_button_sensor_active"]) is bool, "button_readback_invalid")
        return {"button_travel_m": max(0.0, (row["joint_position_m"] - row["joint_rest_position_m"]) * row["press_direction"]),
                "button_activated": row["native_button_sensor_active"], "contact_force_n": _forces(row)}
    if kind == "cloth_folding":
        points, target = _array(row["keypoint_positions_m"], columns=3), _array(row["target_keypoint_positions_m"], columns=3)
        _require(points.shape == target.shape, "cloth_keypoints_invalid")
        overlap = _cloth_overlap_fraction(row["first_fold_polygon_xy_m"], row["second_fold_polygon_xy_m"])
        _require(type(row.get("robot_contact_active")) is bool, "robot_contact_readback_missing")
        return {"fold_keypoint_error_m": float(np.linalg.norm(points - target, axis=1).max()),
                "fold_overlap_fraction": overlap,
                "robot_contact_active": row["robot_contact_active"],
                "linear_speed_m_s": float(np.linalg.norm(_array(row["vertex_velocities_m_s"], columns=3), axis=1).max())}
    raise ValueError("task_family_native_unknown_readback")


def _bound_stage_readback(kind: str, binding: Mapping[str, Any], assets: Mapping[str, Any]) -> dict[str, Any]:
    """Read only selected native assets; measurement targets come from the seal."""
    roles, geometry = binding["asset_roles"], binding["measurement_geometry"]
    _require(all(isinstance(assets.get(key), Mapping) for key in roles.values()), "bound_asset_readback_missing")
    states = {role: assets[key] for role, key in roles.items()}
    if kind == "pouring":
        row = dict(states["material"])
        row.update(geometry, source_pose_world=states["source"]["pose_world"],
                   receiver_pose_world=states["receiver"]["pose_world"])
    elif kind == "wiping":
        row = {**states["surface"], **geometry}
    elif kind == "insertion":
        peg_position, peg_rotation = _pose(states["peg"]["pose_world"])
        socket_position, socket_rotation = _pose(states["socket"]["pose_world"])
        row = {**states["peg"],
               "peg_tip_position_m": (peg_position + peg_rotation @ _vector(geometry["peg_tip_local_m"])).tolist(),
               "peg_axis_world": (peg_rotation @ _unit(geometry["peg_axis_local"])).tolist(),
               "socket_entry_position_m": (socket_position + socket_rotation @ _vector(geometry["socket_entry_local_m"])).tolist(),
               "socket_axis_world": (socket_rotation @ _unit(geometry["socket_axis_local"])).tolist()}
    elif kind == "stacking":
        position, rotation = _pose(states["support"]["pose_world"])
        row = {**states["object"], "support_polygon_xy_m": geometry["support_polygon_local_xy_m"],
               "center_of_mass_m": ((_vector(states["object"]["center_of_mass_m"]) - position) @ rotation).tolist()}
    elif kind == "button_press":
        row = {**states["button"], **geometry}
    else:
        row = dict(states["cloth"])
        vertices = _array(row["vertex_positions_m"], columns=3)
        plane_origin, plane_rotation = _pose(geometry["fold_plane_pose_world"])
        plane_vertices = (vertices - plane_origin) @ plane_rotation
        selected = {key: geometry[key] for key in
                    ("keypoint_vertex_indices", "first_fold_vertex_indices", "second_fold_vertex_indices")}
        _require(all(max(indices) < len(vertices) for indices in selected.values()), "cloth_vertex_index_out_of_range")
        row.update(keypoint_positions_m=vertices[selected["keypoint_vertex_indices"]].tolist(),
                   target_keypoint_positions_m=geometry["target_keypoint_positions_m"],
                   first_fold_polygon_xy_m=plane_vertices[selected["first_fold_vertex_indices"], :2].tolist(),
                   second_fold_polygon_xy_m=plane_vertices[selected["second_fold_vertex_indices"], :2].tolist())
    return row


def derive_task_family_measurements(contract: Mapping[str, Any], native_state: Mapping[str, Any], *,
                                    native_runtime_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Derive task quantities from engine geometry/contact/particle readback."""
    contract = validate_task_family_contract(contract)
    runtime_contract = validate_task_family_native_runtime_contract(native_runtime_contract)
    _require(runtime_contract["task_spec"]["task_success_contract"] == contract, "success_contract_mismatch")
    _require(isinstance(native_state, Mapping) and native_state.get("source") == "native_simulator_state",
             "readback_source_invalid")
    provenance = native_state.get("native_provenance")
    _require(isinstance(provenance, Mapping) and type(provenance.get("seed")) is int
             and provenance == native_runtime_provenance(runtime_contract),
             "readback_scene_binding_mismatch")
    _require(type(native_state.get("safety_violation")) is bool, "safety_readback_missing")
    assets = native_state.get("asset_states")
    _require(isinstance(assets, Mapping), "asset_readback_missing")
    result = {}
    for stage in contract["stages"]:
        try:
            binding = runtime_contract["task_state_binding"]["stage_bindings"][stage["stage_id"]]
            row = _bound_stage_readback(stage["task_family"], binding, assets)
            result[stage["stage_id"]] = _stage_measurements(stage["task_family"], row)
        except KeyError as exc:
            raise ValueError("task_family_native_readback_missing:" + str(exc.args[0])) from None
    return result


def native_runtime_provenance(runtime_contract: Mapping[str, Any]) -> dict[str, Any]:
    """The exact scene/scenario identity every native readback must observe."""
    return {"native_runtime_contract_digest": runtime_contract["contract_digest"],
            "scenario_cell_id": runtime_contract["scenario_cell_id"],
            "scenario_instance_digest": runtime_contract["scenario_instance_digest"],
            "seed": runtime_contract["seed"]}


def execute_task_family_episode(
    *, runtime: Any, policy: Any, contract: Mapping[str, Any], maximum_steps: int,
    control_frequency_hz: float, output_dir: Path, episode_id: str, seed: int = 0,
    native_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded episode; retain inputs and score only independent state.

    The trusted runtime implements reset(seed), native_state(),
    policy_observation(), step(action, duration_seconds), close(), and
    runtime_identity_digest. Native state includes simulation_time_seconds.
    Observations are exactly ``{rgb, proprioception}``; privileged engine state
    stays in the scorer. The policy implements act(observation).
    """
    samples, frames, scoring, failure, closed = [], [], None, None, False
    artifacts: list[dict[str, Any]] = []
    visual: dict[str, Any] = {"status": "missing", "media_gap": {"type": "before_first_observation"}}
    task, runtime_contract = None, None
    try:
        task = validate_task_family_contract(contract)
        runtime_contract = validate_task_family_native_runtime_contract(
            native_runtime_contract if native_runtime_contract is not None else getattr(runtime, "native_runtime_contract", None))
        _require(runtime_contract["task_spec"]["task_success_contract"] == task
                 and runtime_contract["seed"] == seed, "success_contract_mismatch")
        _require(getattr(runtime, "runtime_identity_digest", None) == task["runtime_identity_digest"],
                 "runtime_identity_mismatch")
        _require(getattr(runtime, "engine_id", None) == runtime_contract["task_state_binding"]["engine_id"],
                 "engine_identity_mismatch")
        _require(getattr(runtime, "native_runtime_contract_digest", None) == runtime_contract["contract_digest"],
                 "applied_runtime_contract_mismatch")
        _require(type(maximum_steps) is int and maximum_steps > 0 and _number(control_frequency_hz)
                 and control_frequency_hz > 0 and 1 / control_frequency_hz <= task["maximum_sample_gap_seconds"],
                 "episode_bounds_invalid")
        _require(type(seed) is int and seed >= 0 and isinstance(episode_id, str) and bool(_ID.fullmatch(episode_id)),
                 "episode_identity_invalid")
        runtime.reset(seed=seed)
        for index in range(maximum_steps + 1):
            _require(getattr(runtime, "native_runtime_contract_digest", None) == runtime_contract["contract_digest"],
                     "applied_runtime_contract_mismatch")
            state = runtime.native_state()
            _require(getattr(runtime, "native_runtime_contract_digest", None) == runtime_contract["contract_digest"],
                     "applied_runtime_contract_mismatch")
            measurements = derive_task_family_measurements(task, state, native_runtime_contract=runtime_contract)
            moment = state.get("simulation_time_seconds")
            _require(_number(moment) and abs(moment - index / control_frequency_hz) <= 1e-8,
                     "simulation_clock_mismatch")
            samples.append({"time_seconds": moment, "measurements": measurements,
                            "safety_violation": state["safety_violation"],
                            "native_provenance": dict(state["native_provenance"])})
            if index == 0:
                reset_trace = {"schema_version": TELEMETRY_SCHEMA, "scope": task["scope"],
                               "contract_digest": task["contract_digest"], "runtime_identity_digest": task["runtime_identity_digest"],
                               "source": "native_simulator_state", "samples": samples}
                reset_trace["telemetry_digest"] = canonical_digest(reset_trace, digest_field="telemetry_digest")
                score_task_family_episode(contract=task, telemetry=reset_trace)
            observation = runtime.policy_observation()
            _require(isinstance(observation, Mapping) and set(observation) == {"rgb", "proprioception"},
                     "policy_observation_fields_invalid")
            proprio = _array(observation["proprioception"])
            _require(proprio.ndim == 1, "policy_proprioception_invalid")
            kind = "terminal-observation" if index == maximum_steps else "policy-input"
            frame = persist_observation_frame(observation["rgb"], output_dir=Path(output_dir),
                episode_id=episode_id, frame_index=index, kind=kind)
            frame["proprioception"] = proprio.tolist()
            frame["simulation_time_seconds"] = index / control_frequency_hz
            if index == maximum_steps:
                visual, artifacts = finalize_visual_evidence(output_dir=Path(output_dir), episode_id=episode_id,
                    identity={"scope": task["scope"], "contract_digest": task["contract_digest"], "seed": seed,
                              "native_provenance": native_runtime_provenance(runtime_contract)},
                    policy_input_frames=frames, terminal_observation=frame, frames_per_second=control_frequency_hz)
                break
            frames.append(frame)
            action = policy.act(copy.deepcopy(observation))
            action_array = _array(action)
            _require(action_array.ndim == 1, "policy_action_invalid")
            runtime.step(action_array.tolist(), duration_seconds=1 / control_frequency_hz)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "reason": str(exc)}
        visual = {"status": "incomplete", "media_gap": {"type": "after_first_observation" if frames else
                  "before_first_observation", "reason": str(exc)}, "retained_policy_input_frame_count": len(frames)}
        if frames:
            manifest = {"schema_version": "task_family_failed_observation_manifest.v1", "episode_id": episode_id,
                        "native_provenance": native_runtime_provenance(runtime_contract),
                        "policy_input_frames": frames, "failure": failure}
            manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
            manifest_path = Path(output_dir) / "media" / episode_id / "failed_frame_manifest.json"
            write_json(manifest_path, manifest)
            visual["frame_manifest_digest"] = manifest["manifest_digest"]
            visual["frame_manifest_path"] = str(manifest_path)
    finally:
        try:
            runtime.close()
            closed = True
        except Exception as exc:
            failure = {"type": type(exc).__name__, "reason": "runtime_close_failed:" + str(exc)}
    telemetry = None
    if task is not None and samples:
        telemetry = {"schema_version": TELEMETRY_SCHEMA, "scope": task["scope"],
                     "contract_digest": task["contract_digest"], "runtime_identity_digest": task["runtime_identity_digest"],
                     "native_provenance": native_runtime_provenance(runtime_contract),
                     "source": "native_simulator_state", "samples": samples}
        telemetry["telemetry_digest"] = canonical_digest(telemetry, digest_field="telemetry_digest")
        try:
            scoring = score_task_family_episode(contract=task, telemetry=telemetry)
        except ValueError as exc:
            failure = failure or {"type": type(exc).__name__, "reason": str(exc)}
    status = "completed" if failure is None and visual.get("status") == "complete" and scoring is not None else "invalid"
    receipt = {"schema_version": RECEIPT_SCHEMA, "episode_id": episode_id, "status": status,
               "task_succeeded": status == "completed" and scoring["task_succeeded"],
               "telemetry": telemetry, "scoring": scoring, "visual_evidence": visual,
               "artifacts": artifacts, "failure": failure, "runtime_closed": closed,
               "scoring_source": "deterministic_native_state", "physical_validation": False,
               "native_provenance": native_runtime_provenance(runtime_contract) if runtime_contract else None,
               "maximum_steps": maximum_steps, "control_frequency_hz": control_frequency_hz,
               "claim_ceiling": task["claim_ceiling"] if task else "development_only"}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Materialize a typed local contract without launching a runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize", help="Validate and seal native runtime input JSON")
    materialize.add_argument("--inputs", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
        _require(isinstance(inputs, dict) and "destination" not in inputs, "materialization_inputs_invalid")
        result = materialize_task_family_native_runtime_contract(**inputs, destination=args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"schema_version": result["schema_version"], "contract_digest": result["contract_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
