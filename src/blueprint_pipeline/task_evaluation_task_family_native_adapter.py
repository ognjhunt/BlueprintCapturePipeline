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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import finalize_failed_policy_visual_evidence, finalize_visual_evidence, persist_observation_frame
from .task_family_native_contracts import (
    RUNTIME_SCHEMA as RUNTIME_SCHEMA,
    _ASSET_ROLES as _ASSET_ROLES,
    _ID,
    _SHA as _SHA,
    _area,
    _array,
    _json,
    _number,
    _polygon,
    _pose,
    _require,
    _unit,
    _validate_measurement_geometry as _validate_measurement_geometry,
    _vector,
    validate_task_family_native_runtime_contract as validate_task_family_native_runtime_contract,
)
from .task_family_scoring import (
    TELEMETRY_SCHEMA,
    canonical_task_family as canonical_task_family,
    score_task_family_episode,
    validate_task_family_contract,
)

RECEIPT_SCHEMA = "task_family_native_episode.v1"


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


def _forces(row: Mapping[str, Any]) -> float:
    if np.asarray(row["contact_forces_n"]).shape in {(0,), (0, 3)}:
        return 0.0
    forces = _array(row["contact_forces_n"], columns=3)
    return float(np.linalg.norm(forces, axis=1).sum())


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
    phase = "native_runtime_validation"
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
        phase = "native_reset"
        runtime.reset(seed=seed)
        for index in range(maximum_steps + 1):
            phase = "native_state_readback"
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
            phase = "policy_observation_readback"
            observation = runtime.policy_observation()
            _require(isinstance(observation, Mapping) and set(observation) == {"rgb", "proprioception"},
                     "policy_observation_fields_invalid")
            proprio = _array(observation["proprioception"])
            _require(proprio.ndim == 1, "policy_proprioception_invalid")
            kind = "terminal-observation" if index == maximum_steps else "policy-input"
            phase = "policy_input_media_retention"
            frame = persist_observation_frame(observation["rgb"], output_dir=Path(output_dir),
                episode_id=episode_id, frame_index=index, kind=kind)
            frame["proprioception"] = proprio.tolist()
            frame["simulation_time_seconds"] = index / control_frequency_hz
            if index == maximum_steps:
                phase = "episode_media_finalization"
                visual, artifacts = finalize_visual_evidence(output_dir=Path(output_dir), episode_id=episode_id,
                    identity={"scope": task["scope"], "contract_digest": task["contract_digest"], "seed": seed,
                              "native_provenance": native_runtime_provenance(runtime_contract)},
                    policy_input_frames=frames, terminal_observation=frame, frames_per_second=control_frequency_hz)
                break
            frames.append(frame)
            phase = "policy_inference"
            action = policy.act(copy.deepcopy(observation))
            action_array = _array(action)
            _require(action_array.ndim == 1, "policy_action_invalid")
            phase = "native_action_step"
            runtime.step(action_array.tolist(), duration_seconds=1 / control_frequency_hz)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "reason": str(exc), "origin": phase}
        visual = {"status": "incomplete", "media_gap": {"type": "after_first_observation" if frames else
                  "before_first_observation", "reason": str(exc)}, "retained_policy_input_frame_count": len(frames)}
        if frames:
            exact = []
            for frame in frames:
                retained = {**frame, "candidate_exact_policy_input": True}
                retained["frame_manifest_digest"] = canonical_digest(retained, digest_field="frame_manifest_digest")
                exact.append(retained)
            try:
                visual, artifacts = finalize_failed_policy_visual_evidence(output_dir=Path(output_dir),
                    episode_id=episode_id, identity={"scope": task["scope"], "contract_digest": task["contract_digest"],
                        "seed": seed, "native_provenance": native_runtime_provenance(runtime_contract)},
                    exact_policy_input_frames=exact, failure_reason=str(exc), frames_per_second=control_frequency_hz)
                visual["retained_policy_input_frame_count"] = len(frames)
            except (OSError, RuntimeError, ValueError) as media_exc:
                visual["media_gap"]["finalization_failure"] = type(media_exc).__name__
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
