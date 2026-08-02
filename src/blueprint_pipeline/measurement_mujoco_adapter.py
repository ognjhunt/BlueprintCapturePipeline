"""MuJoCo rigid-contact development worker for measurement benchmarks.

This is a real local MuJoCo execution port, but intentionally a narrow one. It
runs a digest-bound rigid-body drop/contact case twice, records exact runtime
identity and replay agreement, and emits only development predictions. It does
not represent a captured-site qualification, physical benchmark, R6 decision,
or R7 route.
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


IMPLEMENTATION_ID = "blueprint-mujoco-rigid-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "mujoco_rigid_drop.v1"


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
        raise MeasurementAdapterExecutionError(f"mujoco_adapter_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"mujoco_adapter_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"mujoco_adapter_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"mujoco_adapter_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_adapter_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_adapter_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("mujoco_adapter_protocol_invalid")
    shape = str(point.get("body_shape", "")).strip()
    if shape not in {"sphere", "box"}:
        raise MeasurementAdapterExecutionError("mujoco_adapter_body_shape_invalid")
    if shape == "sphere":
        size = [_number(point.get("radius_m"), name="radius_m", minimum=1e-4)]
        resting_height = size[0]
    else:
        raw_size = point.get("half_size_m")
        if not isinstance(raw_size, list) or len(raw_size) != 3:
            raise MeasurementAdapterExecutionError("mujoco_adapter_half_size_invalid")
        size = [_number(item, name="half_size_m", minimum=1e-4) for item in raw_size]
        resting_height = size[2]
    friction = point.get("friction")
    if not isinstance(friction, list) or len(friction) != 3:
        raise MeasurementAdapterExecutionError("mujoco_adapter_friction_invalid")
    friction_values = [_number(item, name="friction", minimum=0.0) for item in friction]
    timestep = _number(point.get("timestep_s"), name="timestep_s", minimum=1e-6, maximum=0.05)
    duration = _number(point.get("duration_s"), name="duration_s", minimum=timestep, maximum=60.0)
    initial_height = _number(
        point.get("initial_height_m"),
        name="initial_height_m",
        minimum=resting_height + 1e-4,
    )
    return {
        "shape": shape,
        "size": size,
        "resting_height": resting_height,
        "mass_kg": _number(point.get("mass_kg"), name="mass_kg", minimum=1e-6),
        "initial_height_m": initial_height,
        "gravity_m_s2": _number(point.get("gravity_m_s2", -9.81), name="gravity_m_s2", maximum=0.0),
        "timestep_s": timestep,
        "duration_s": duration,
        "friction": friction_values,
        "penetration_unsafe_threshold_m": _number(
            point.get("penetration_unsafe_threshold_m", 0.001),
            name="penetration_unsafe_threshold_m",
            minimum=0.0,
        ),
    }


def _xml(point: Mapping[str, Any], solver_settings: Mapping[str, Any]) -> str:
    size = " ".join(f"{item:.17g}" for item in point["size"])
    friction = " ".join(f"{item:.17g}" for item in point["friction"])
    iterations = int(solver_settings["iterations"])
    tolerance = float(solver_settings["tolerance"])
    integrator = str(solver_settings["integrator"])
    solver = str(solver_settings["solver"])
    return f"""<mujoco model="blueprint_measurement_rigid_drop">
  <option timestep="{point["timestep_s"]:.17g}"
          gravity="0 0 {point["gravity_m_s2"]:.17g}"
          integrator="{integrator}" solver="{solver}"
          iterations="{iterations}" tolerance="{tolerance:.17g}"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1" friction="{friction}"/>
    <body name="test_body" pos="0 0 {point["initial_height_m"]:.17g}">
      <freejoint/>
      <geom name="test_geom" type="{point["shape"]}" size="{size}"
            mass="{point["mass_kg"]:.17g}" friction="{friction}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _simulate(mujoco: Any, model_xml: str, duration_s: float) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_body")
    step_count = int(math.ceil(duration_s / model.opt.timestep))
    first_contact_step: int | None = None
    minimum_contact_distance = 0.0
    for step in range(step_count):
        mujoco.mj_step(model, data)
        if data.ncon and first_contact_step is None:
            first_contact_step = step
        for contact_index in range(data.ncon):
            minimum_contact_distance = min(
                minimum_contact_distance, float(data.contact[contact_index].dist)
            )
    final_position = [float(item) for item in data.xpos[body_id]]
    final_velocity = [float(item) for item in data.qvel[:6]]
    trace = {
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "minimum_contact_distance_m": minimum_contact_distance,
        "final_position_m": final_position,
        "final_velocity": final_velocity,
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_mujoco_measurement_request(
    request_value: Mapping[str, Any],
) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    base_observations = {
        "engine_version": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    if request["implementation"]["implementation_id"] != IMPLEMENTATION_ID:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["mujoco_adapter_implementation_id_mismatch"],
        )
    if request["implementation"]["implementation_version"] != IMPLEMENTATION_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["mujoco_adapter_implementation_version_mismatch"],
        )
    if request["implementation"]["implementation_digest"] != implementation_digest():
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["mujoco_adapter_implementation_digest_mismatch"],
        )
    try:
        import mujoco
    except ImportError:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["mujoco_adapter_package_unavailable"],
        )
    engine_version = str(mujoco.__version__)
    base_observations["engine_version"] = engine_version
    if engine_version != runtime["target_engine_version"]:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["mujoco_adapter_target_version_mismatch"],
        )
    solver_settings = dict(runtime["solver_settings"])
    required_settings = {"integrator", "solver", "iterations", "tolerance"}
    if set(solver_settings) != required_settings:
        raise MeasurementAdapterExecutionError("mujoco_adapter_solver_settings_invalid")
    if solver_settings["integrator"] not in {"Euler", "implicit", "implicitfast", "RK4"}:
        raise MeasurementAdapterExecutionError("mujoco_adapter_integrator_invalid")
    if solver_settings["solver"] not in {"PGS", "CG", "Newton"}:
        raise MeasurementAdapterExecutionError("mujoco_adapter_solver_invalid")
    iterations = solver_settings["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise MeasurementAdapterExecutionError("mujoco_adapter_iterations_invalid")
    _number(solver_settings["tolerance"], name="tolerance", minimum=0.0)
    point = _operating_point(request)
    model_xml = _xml(point, solver_settings)
    model_digest = "sha256:" + hashlib.sha256(model_xml.encode()).hexdigest()
    first = _simulate(mujoco, model_xml, point["duration_s"])
    second = _simulate(mujoco, model_xml, point["duration_s"])
    replay_match = first["trace_digest"] == second["trace_digest"]
    penetration = max(0.0, -float(first["minimum_contact_distance_m"]))
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "penetration": penetration,
        "contact_sequence": (
            "ground_contact" if first["first_contact_step"] is not None else "no_contact"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations = {
        **base_observations,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_digest": implementation_digest(),
        "model_digest": model_digest,
        "solver_settings_digest": runtime["solver_settings_digest"],
        "timestep_s": point["timestep_s"],
        "step_count": first["step_count"],
        "first_contact_step": first["first_contact_step"],
        "final_position_m": first["final_position_m"],
        "final_velocity": first["final_velocity"],
        "trace_digest": first["trace_digest"],
        "repeat_trace_digest": second["trace_digest"],
        "deterministic_replay_match": replay_match,
    }
    if not replay_match:
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics=metrics,
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["mujoco_adapter_replay_mismatch"],
        )
    return build_measurement_adapter_worker_result(
        request,
        status="completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=(penetration > point["penetration_unsafe_threshold_m"]),
        runtime_observations=observations,
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("mujoco_adapter_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_adapter_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MuJoCo development measurement case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_mujoco_measurement_request(_load_object(args.request))
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
    "run_mujoco_measurement_request",
]
