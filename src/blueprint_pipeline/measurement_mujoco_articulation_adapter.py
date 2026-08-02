"""MuJoCo door/drawer articulation development worker.

The worker executes a digest-bound synthetic hinge or slide case twice and
emits development predictions only after exact replay agreement. It does not
measure a captured joint, instrumented force, physical task success, or R5-R7
qualification evidence.
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


IMPLEMENTATION_ID = "blueprint-mujoco-articulation-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "mujoco_articulated_joint_travel.v1"


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
        raise MeasurementAdapterExecutionError(f"mujoco_articulation_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"mujoco_articulation_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"mujoco_articulation_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"mujoco_articulation_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_articulation_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_articulation_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("mujoco_articulation_protocol_invalid")
    articulation_type = str(point.get("articulation_type", "")).strip()
    if articulation_type not in {"door_hinge", "drawer_slide"}:
        raise MeasurementAdapterExecutionError("mujoco_articulation_type_invalid")
    if point.get("reference_origin") != "synthetic_public_target":
        raise MeasurementAdapterExecutionError("mujoco_articulation_reference_origin_invalid")
    if point.get("physical_force_measurement") is not False:
        raise MeasurementAdapterExecutionError("mujoco_articulation_physical_force_scope_invalid")
    timestep = _number(point.get("timestep_s"), name="timestep_s", minimum=1e-6, maximum=0.05)
    duration = _number(point.get("duration_s"), name="duration_s", minimum=timestep, maximum=10.0)
    limit = _number(point.get("joint_limit"), name="joint_limit", minimum=1e-4)
    target = _number(
        point.get("target_joint_position"),
        name="target_joint_position",
        minimum=0.0,
        maximum=limit,
    )
    return {
        "articulation_type": articulation_type,
        "joint_limit": limit,
        "target_joint_position": target,
        "maximum_travel_error": _number(
            point.get("maximum_travel_error"),
            name="maximum_travel_error",
            minimum=0.0,
        ),
        "applied_effort": _number(point.get("applied_effort"), name="applied_effort", minimum=1e-6),
        "body_mass_kg": _number(point.get("body_mass_kg"), name="body_mass_kg", minimum=1e-6),
        "joint_damping": _number(point.get("joint_damping"), name="joint_damping", minimum=0.0),
        "timestep_s": timestep,
        "duration_s": duration,
    }


def _xml(point: Mapping[str, Any], solver_settings: Mapping[str, Any]) -> str:
    if point["articulation_type"] == "door_hinge":
        joint = (
            '<joint name="joint" type="hinge" axis="0 0 1" '
            f'range="0 {point["joint_limit"]:.17g}" '
            f'damping="{point["joint_damping"]:.17g}"/>'
        )
        geom = (
            '<geom name="moving_geom" type="box" pos="0.25 0 0" '
            f'size="0.25 0.02 0.35" mass="{point["body_mass_kg"]:.17g}"/>'
        )
    else:
        joint = (
            '<joint name="joint" type="slide" axis="1 0 0" '
            f'range="0 {point["joint_limit"]:.17g}" '
            f'damping="{point["joint_damping"]:.17g}"/>'
        )
        geom = (
            '<geom name="moving_geom" type="box" size="0.2 0.15 0.08" '
            f'mass="{point["body_mass_kg"]:.17g}"/>'
        )
    effort = point["applied_effort"]
    return f"""<mujoco model="blueprint_measurement_articulation">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{point["timestep_s"]:.17g}" gravity="0 0 0"
          integrator="{solver_settings["integrator"]}"
          solver="{solver_settings["solver"]}"
          iterations="{int(solver_settings["iterations"])}"
          tolerance="{float(solver_settings["tolerance"]):.17g}"/>
  <worldbody>
    <body name="moving" pos="0 0 0.5">
      {joint}
      {geom}
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="joint" gear="1" ctrllimited="true"
           ctrlrange="-{effort:.17g} {effort:.17g}"/>
  </actuator>
</mujoco>
"""


def _simulate(mujoco: Any, model_xml: str, point: Mapping[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    data.ctrl[0] = point["applied_effort"]
    step_count = int(math.ceil(point["duration_s"] / model.opt.timestep))
    positions: list[float] = []
    velocities: list[float] = []
    for _ in range(step_count):
        mujoco.mj_step(model, data)
        positions.append(float(data.qpos[0]))
        velocities.append(float(data.qvel[0]))
    final_position = positions[-1]
    trace = {
        "step_count": step_count,
        "final_joint_position": final_position,
        "maximum_joint_position": max(positions),
        "peak_absolute_joint_velocity": max(abs(item) for item in velocities),
        "joint_limit_reached": max(positions) >= point["joint_limit"] - 1e-6,
        "travel_error": abs(final_position - point["target_joint_position"]),
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_mujoco_articulation_measurement_request(
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
    for key, expected, code in (
        ("implementation_id", IMPLEMENTATION_ID, "implementation_id_mismatch"),
        (
            "implementation_version",
            IMPLEMENTATION_VERSION,
            "implementation_version_mismatch",
        ),
        (
            "implementation_digest",
            implementation_digest(),
            "implementation_digest_mismatch",
        ),
    ):
        if implementation[key] != expected:
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations=observations,
                failure_codes=[f"mujoco_articulation_{code}"],
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
            failure_codes=["mujoco_articulation_package_unavailable"],
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
            failure_codes=["mujoco_articulation_target_version_mismatch"],
        )
    solver_settings = dict(runtime["solver_settings"])
    if set(solver_settings) != {"integrator", "solver", "iterations", "tolerance"}:
        raise MeasurementAdapterExecutionError("mujoco_articulation_solver_settings_invalid")
    if solver_settings["integrator"] != "implicitfast":
        raise MeasurementAdapterExecutionError("mujoco_articulation_integrator_invalid")
    if solver_settings["solver"] != "Newton":
        raise MeasurementAdapterExecutionError("mujoco_articulation_solver_invalid")
    iterations = solver_settings["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise MeasurementAdapterExecutionError("mujoco_articulation_iterations_invalid")
    _number(solver_settings["tolerance"], name="tolerance", minimum=0.0)
    point = _operating_point(request)
    model_xml = _xml(point, solver_settings)
    model_digest = "sha256:" + hashlib.sha256(model_xml.encode()).hexdigest()
    first = _simulate(mujoco, model_xml, point)
    second = _simulate(mujoco, model_xml, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics = {
        "drawer_door_force_travel_error": first["travel_error"],
        "contact_sequence": (
            "joint_limit_reached" if first["joint_limit_reached"] else "articulation_free_travel"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations.update(
        {
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "model_digest": model_digest,
            "solver_settings_digest": runtime["solver_settings_digest"],
            "articulation_type": point["articulation_type"],
            "applied_effort": point["applied_effort"],
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
            failure_codes=["mujoco_articulation_replay_mismatch"],
        )
    return build_measurement_adapter_worker_result(
        request,
        status="completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=(first["travel_error"] > point["maximum_travel_error"]),
        runtime_observations=observations,
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("mujoco_articulation_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_articulation_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MuJoCo articulation development case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_mujoco_articulation_measurement_request(_load_object(args.request))
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
    "run_mujoco_articulation_measurement_request",
]
