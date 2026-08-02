"""MuJoCo peg-insertion boundary development worker.

This worker executes synthetic square-peg insertion cases with public geometry
twice. It exercises collision/interference and insertion-result plumbing only;
it is not an instrumented insertion benchmark or physical qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-mujoco-insertion-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "mujoco_square_peg_insertion_boundary.v1"


def implementation_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"mujoco_insertion_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"mujoco_insertion_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"mujoco_insertion_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"mujoco_insertion_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_insertion_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_insertion_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("mujoco_insertion_protocol_invalid")
    if point.get("geometry_origin") != "synthetic_public_primitives":
        raise MeasurementAdapterExecutionError("mujoco_insertion_geometry_origin_invalid")
    if point.get("instrumented_force_measurement") is not False:
        raise MeasurementAdapterExecutionError("mujoco_insertion_instrumented_force_scope_invalid")
    timestep = _number(point.get("timestep_s"), name="timestep_s", minimum=1e-6, maximum=0.05)
    duration = _number(point.get("duration_s"), name="duration_s", minimum=timestep, maximum=10.0)
    peg_half_width = _number(point.get("peg_half_width_m"), name="peg_half_width_m", minimum=1e-4)
    hole_half_width = _number(
        point.get("hole_half_width_m"), name="hole_half_width_m", minimum=1e-4
    )
    insertion_depth = _number(
        point.get("target_insertion_depth_m"),
        name="target_insertion_depth_m",
        minimum=1e-4,
    )
    return {
        "peg_half_width_m": peg_half_width,
        "hole_half_width_m": hole_half_width,
        "lateral_offset_m": _number(
            point.get("lateral_offset_m"), name="lateral_offset_m", minimum=0.0
        ),
        "peg_half_height_m": _number(
            point.get("peg_half_height_m"), name="peg_half_height_m", minimum=1e-4
        ),
        "peg_mass_kg": _number(point.get("peg_mass_kg"), name="peg_mass_kg", minimum=1e-6),
        "wall_thickness_m": _number(
            point.get("wall_thickness_m"), name="wall_thickness_m", minimum=1e-4
        ),
        "wall_half_height_m": _number(
            point.get("wall_half_height_m"),
            name="wall_half_height_m",
            minimum=1e-4,
        ),
        "initial_center_height_m": _number(
            point.get("initial_center_height_m"),
            name="initial_center_height_m",
            minimum=1e-4,
        ),
        "target_insertion_depth_m": insertion_depth,
        "success_tolerance_m": _number(
            point.get("success_tolerance_m"), name="success_tolerance_m", minimum=0.0
        ),
        "applied_force_n": _number(
            point.get("applied_force_n"), name="applied_force_n", minimum=1e-6
        ),
        "joint_damping": _number(point.get("joint_damping"), name="joint_damping", minimum=0.0),
        "penetration_unsafe_threshold_m": _number(
            point.get("penetration_unsafe_threshold_m"),
            name="penetration_unsafe_threshold_m",
            minimum=0.0,
        ),
        "timestep_s": timestep,
        "duration_s": duration,
    }


def _xml(point: Mapping[str, Any], solver: Mapping[str, Any]) -> str:
    gap = point["hole_half_width_m"]
    wall = point["wall_thickness_m"]
    wall_z = point["wall_half_height_m"]
    offset = point["lateral_offset_m"]
    peg = point["peg_half_width_m"]
    peg_z = point["peg_half_height_m"]
    depth = point["target_insertion_depth_m"]
    force = point["applied_force_n"]
    return f"""<mujoco model="blueprint_measurement_peg_insertion">
  <compiler autolimits="true"/>
  <option timestep="{point["timestep_s"]:.17g}" gravity="0 0 0"
          integrator="{solver["integrator"]}" solver="{solver["solver"]}"
          iterations="{int(solver["iterations"])}"
          tolerance="{float(solver["tolerance"]):.17g}"/>
  <worldbody>
    <geom name="left" type="box" pos="{-gap - wall:.17g} 0 {wall_z:.17g}"
          size="{wall:.17g} 0.2 {wall_z:.17g}"/>
    <geom name="right" type="box" pos="{gap + wall:.17g} 0 {wall_z:.17g}"
          size="{wall:.17g} 0.2 {wall_z:.17g}"/>
    <geom name="front" type="box" pos="0 {-gap - wall:.17g} {wall_z:.17g}"
          size="{gap:.17g} {wall:.17g} {wall_z:.17g}"/>
    <geom name="back" type="box" pos="0 {gap + wall:.17g} {wall_z:.17g}"
          size="{gap:.17g} {wall:.17g} {wall_z:.17g}"/>
    <body name="peg" pos="{offset:.17g} 0 {point["initial_center_height_m"]:.17g}">
      <joint name="insert" type="slide" axis="0 0 1" range="{-depth:.17g} 0"
             damping="{point["joint_damping"]:.17g}"/>
      <geom name="peg_geom" type="box" size="{peg:.17g} {peg:.17g} {peg_z:.17g}"
            mass="{point["peg_mass_kg"]:.17g}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="insert_drive" joint="insert" gear="1" ctrllimited="true"
           ctrlrange="-{force:.17g} {force:.17g}"/>
  </actuator>
</mujoco>
"""


def _simulate(mujoco: Any, model_xml: str, point: Mapping[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    data.ctrl[0] = -point["applied_force_n"]
    step_count = int(math.ceil(point["duration_s"] / model.opt.timestep))
    first_contact_step: int | None = None
    minimum_contact_distance = 0.0
    maximum_contact_count = 0
    for step in range(step_count):
        mujoco.mj_step(model, data)
        maximum_contact_count = max(maximum_contact_count, int(data.ncon))
        if data.ncon and first_contact_step is None:
            first_contact_step = step
        for index in range(data.ncon):
            minimum_contact_distance = min(
                minimum_contact_distance, float(data.contact[index].dist)
            )
    final_depth = max(0.0, -float(data.qpos[0]))
    penetration = max(0.0, -minimum_contact_distance)
    success = final_depth >= (point["target_insertion_depth_m"] - point["success_tolerance_m"])
    signed_clearance = point["hole_half_width_m"] - (
        point["peg_half_width_m"] + abs(point["lateral_offset_m"])
    )
    trace = {
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "maximum_contact_count": maximum_contact_count,
        "final_insertion_depth_m": final_depth,
        "signed_minimum_clearance_m": signed_clearance,
        "penetration_m": penetration,
        "insertion_succeeded": success,
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_mujoco_insertion_measurement_request(
    request_value: Mapping[str, Any],
) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    observations = {
        "engine_version": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    implementation = request["implementation"]
    identities = (
        ("implementation_id", IMPLEMENTATION_ID, "implementation_id_mismatch"),
        ("implementation_version", IMPLEMENTATION_VERSION, "implementation_version_mismatch"),
        ("implementation_digest", implementation_digest(), "implementation_digest_mismatch"),
    )
    for key, expected, code in identities:
        if implementation[key] != expected:
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations=observations,
                failure_codes=[f"mujoco_insertion_{code}"],
            )
    try:
        import mujoco
    except ImportError:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["mujoco_insertion_package_unavailable"],
        )
    engine_version = str(mujoco.__version__)
    observations["engine_version"] = engine_version
    if engine_version != runtime["target_engine_version"]:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["mujoco_insertion_target_version_mismatch"],
        )
    solver = dict(runtime["solver_settings"])
    if set(solver) != {"integrator", "solver", "iterations", "tolerance"}:
        raise MeasurementAdapterExecutionError("mujoco_insertion_solver_settings_invalid")
    if solver["integrator"] != "implicitfast":
        raise MeasurementAdapterExecutionError("mujoco_insertion_integrator_invalid")
    if solver["solver"] != "Newton":
        raise MeasurementAdapterExecutionError("mujoco_insertion_solver_invalid")
    iterations = solver["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise MeasurementAdapterExecutionError("mujoco_insertion_iterations_invalid")
    _number(solver["tolerance"], name="tolerance", minimum=0.0)
    point = _operating_point(request)
    model_xml = _xml(point, solver)
    model_digest = "sha256:" + hashlib.sha256(model_xml.encode()).hexdigest()
    first = _simulate(mujoco, model_xml, point)
    second = _simulate(mujoco, model_xml, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics = {
        "minimum_clearance_error": first["signed_minimum_clearance_m"],
        "contact_sequence": (
            "side_contact" if first["first_contact_step"] is not None else "no_side_contact"
        ),
        "penetration": first["penetration_m"],
        "insertion_success_boundary": first["insertion_succeeded"],
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations.update(
        {
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "model_digest": model_digest,
            "solver_settings_digest": runtime["solver_settings_digest"],
            "applied_force_n": point["applied_force_n"],
            "timestep_s": point["timestep_s"],
            **{key: value for key, value in first.items() if key != "trace_digest"},
            "trace_digest": first["trace_digest"],
            "repeat_trace_digest": second["trace_digest"],
            "deterministic_replay_match": replay_match,
        }
    )
    if not replay_match:
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics=metrics,
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["mujoco_insertion_replay_mismatch"],
        )
    unsafe = (not first["insertion_succeeded"]) or (
        first["penetration_m"] > point["penetration_unsafe_threshold_m"]
    )
    return build_measurement_adapter_worker_result(
        request,
        status="completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=unsafe,
        runtime_observations=observations,
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("mujoco_insertion_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_insertion_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MuJoCo peg-insertion development case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_mujoco_insertion_measurement_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "implementation_digest",
    "main",
    "run_mujoco_insertion_measurement_request",
]
