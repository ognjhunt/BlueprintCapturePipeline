"""Drake 1.55 CPU SAP rigid-contact development worker.

The worker executes the shared sphere/box drop corpus with point contact and
the SAP discrete contact approximation. Drake 1.55 does not support the
repository's macOS Python 3.12 environment, so the checked suite may launch
this module through an explicit external Python 3.13/3.14 interpreter. The
argv-only execution receipt binds that interpreter command and this module's
source identity. This is development evidence, never qualification or R7.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-drake-sap-rigid-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "drake_sap_point_rigid_drop.v1"
DRAKE_VERSION = "1.55.0"
WORKER_SCRIPT = Path(__file__).parents[2] / "scripts/measurement_drake_rigid_worker.py"


def implementation_digest() -> str:
    hasher = hashlib.sha256()
    for label, path in (("adapter", Path(__file__)), ("worker", WORKER_SCRIPT)):
        hasher.update(label.encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"drake_rigid_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"drake_rigid_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"drake_rigid_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"drake_rigid_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"drake_rigid_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("drake_rigid_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("drake_rigid_protocol_invalid")
    if point.get("protocol_family") != "rigid_body_drop":
        raise MeasurementAdapterExecutionError("drake_rigid_protocol_family_invalid")
    shape = str(point.get("body_shape", "")).strip()
    if shape not in {"sphere", "box"}:
        raise MeasurementAdapterExecutionError("drake_rigid_body_shape_invalid")
    if shape == "sphere":
        size = [_number(point.get("radius_m"), name="radius_m", minimum=1e-4)]
        resting_height = size[0]
    else:
        raw_size = point.get("half_size_m")
        if not isinstance(raw_size, list) or len(raw_size) != 3:
            raise MeasurementAdapterExecutionError("drake_rigid_half_size_invalid")
        size = [_number(item, name="half_size_m", minimum=1e-4) for item in raw_size]
        resting_height = size[2]
    friction = point.get("friction")
    if not isinstance(friction, list) or len(friction) != 3:
        raise MeasurementAdapterExecutionError("drake_rigid_friction_invalid")
    friction_values = [_number(item, name="friction", minimum=0.0) for item in friction]
    timestep = _number(point.get("timestep_s"), name="timestep_s", minimum=1e-6, maximum=0.05)
    duration = _number(point.get("duration_s"), name="duration_s", minimum=timestep, maximum=60)
    return {
        "shape": shape,
        "size": size,
        "resting_height": resting_height,
        "mass_kg": _number(point.get("mass_kg"), name="mass_kg", minimum=1e-6),
        "initial_height_m": _number(
            point.get("initial_height_m"),
            name="initial_height_m",
            minimum=resting_height + 1e-4,
        ),
        "gravity_m_s2": _number(point.get("gravity_m_s2", -9.81), name="gravity_m_s2", maximum=0),
        "timestep_s": timestep,
        "duration_s": duration,
        "friction": friction_values,
        "penetration_unsafe_threshold_m": _number(
            point.get("penetration_unsafe_threshold_m", 0.001),
            name="penetration_unsafe_threshold_m",
            minimum=0,
        ),
    }


def _simulate(drake: Any, point: Mapping[str, Any], solver: Mapping[str, Any]) -> dict[str, Any]:
    builder = drake.DiagramBuilder()
    plant, _scene_graph = drake.AddMultibodyPlantSceneGraph(builder, time_step=point["timestep_s"])
    plant.set_contact_model(drake.ContactModel.kPoint)
    plant.set_discrete_contact_approximation(drake.DiscreteContactApproximation.kSap)
    plant.set_penetration_allowance(solver["penetration_allowance_m"])
    plant.set_stiction_tolerance(solver["stiction_tolerance_m_s"])
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, point["gravity_m_s2"]])
    material = drake.CoulombFriction(point["friction"][0], point["friction"][0])
    plant.RegisterCollisionGeometry(
        plant.world_body(), drake.RigidTransform(), drake.HalfSpace(), "ground", material
    )
    if point["shape"] == "sphere":
        geometry = drake.Sphere(point["size"][0])
        inertia = drake.UnitInertia.SolidSphere(point["size"][0])
    else:
        dimensions = [2.0 * item for item in point["size"]]
        geometry = drake.Box(*dimensions)
        inertia = drake.UnitInertia.SolidBox(*dimensions)
    body = plant.AddRigidBody(
        "test_body",
        drake.SpatialInertia(point["mass_kg"], [0.0, 0.0, 0.0], inertia),
    )
    plant.RegisterCollisionGeometry(body, drake.RigidTransform(), geometry, "body", material)
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetFreeBodyPose(
        plant_context,
        body,
        drake.RigidTransform([0.0, 0.0, point["initial_height_m"]]),
    )
    simulator = drake.Simulator(diagram, context)
    simulator.Initialize()
    step_count = int(math.ceil(point["duration_s"] / point["timestep_s"]))
    first_contact_step: int | None = None
    maximum_contact_count = 0
    maximum_point_pair_depth = 0.0
    minimum_center_height = point["initial_height_m"]
    for step in range(step_count):
        simulator.AdvanceTo((step + 1) * point["timestep_s"])
        contacts = plant.get_contact_results_output_port().Eval(plant_context)
        contact_count = contacts.num_point_pair_contacts() + contacts.num_hydroelastic_contacts()
        maximum_contact_count = max(maximum_contact_count, contact_count)
        if contact_count and first_contact_step is None:
            first_contact_step = step
        for index in range(contacts.num_point_pair_contacts()):
            pair = contacts.point_pair_contact_info(index).point_pair()
            maximum_point_pair_depth = max(maximum_point_pair_depth, float(pair.depth))
        height = float(plant.EvalBodyPoseInWorld(plant_context, body).translation()[2])
        minimum_center_height = min(minimum_center_height, height)
    pose = plant.EvalBodyPoseInWorld(plant_context, body)
    velocity = plant.EvalBodySpatialVelocityInWorld(plant_context, body)
    penetration = max(
        0.0,
        point["resting_height"] - minimum_center_height,
        maximum_point_pair_depth,
    )
    trace = {
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "maximum_contact_count": maximum_contact_count,
        "maximum_point_pair_depth_m": maximum_point_pair_depth,
        "minimum_center_height_m": minimum_center_height,
        "final_position_m": [float(item) for item in pose.translation()],
        "final_velocity": [
            *[float(item) for item in velocity.translational()],
            *[float(item) for item in velocity.rotational()],
        ],
        "observed_mass_kg": float(body.default_mass()),
        "penetration_m": penetration,
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_drake_rigid_measurement_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
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
        ("implementation_version", IMPLEMENTATION_VERSION, "implementation_version_mismatch"),
        ("implementation_digest", implementation_digest(), "implementation_digest_mismatch"),
    ):
        if implementation[key] != expected:
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations=observations,
                failure_codes=[f"drake_rigid_{code}"],
            )
    if runtime["target_engine_version"] != DRAKE_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["drake_rigid_target_version_mismatch"],
        )
    if runtime["backend_id"] != "drake-multibody-cpu-sap-point":
        raise MeasurementAdapterExecutionError("drake_rigid_backend_invalid")
    if runtime["precision"] != "float64":
        raise MeasurementAdapterExecutionError("drake_rigid_precision_invalid")
    solver = dict(runtime["solver_settings"])
    if set(solver) != {
        "discrete_contact_approximation",
        "contact_model",
        "penetration_allowance_m",
        "stiction_tolerance_m_s",
    }:
        raise MeasurementAdapterExecutionError("drake_rigid_solver_settings_invalid")
    if solver["discrete_contact_approximation"] != "sap":
        raise MeasurementAdapterExecutionError("drake_rigid_contact_approximation_invalid")
    if solver["contact_model"] != "point":
        raise MeasurementAdapterExecutionError("drake_rigid_contact_model_invalid")
    solver["penetration_allowance_m"] = _number(
        solver["penetration_allowance_m"], name="penetration_allowance_m", minimum=1e-9
    )
    solver["stiction_tolerance_m_s"] = _number(
        solver["stiction_tolerance_m_s"], name="stiction_tolerance_m_s", minimum=1e-9
    )
    try:
        if importlib.metadata.version("drake") != DRAKE_VERSION:
            raise importlib.metadata.PackageNotFoundError
        import pydrake.all as drake
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["drake_rigid_package_or_version_unavailable"],
        )
    observations["engine_version"] = DRAKE_VERSION
    point = _operating_point(request)
    first = _simulate(drake, point, solver)
    second = _simulate(drake, point, solver)
    replay_match = first["trace_digest"] == second["trace_digest"]
    available_metrics = {
        "penetration": first["penetration_m"],
        "contact_sequence": (
            "ground_contact" if first["first_contact_step"] is not None else "no_contact"
        ),
    }
    requested = set(request["case_manifest"]["requested_metric_ids"])
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations.update(
        {
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "solver_settings_digest": runtime["solver_settings_digest"],
            "device": "cpu",
            "discrete_contact_approximation": "sap",
            "contact_model": "point",
            "scene_graph_renderer_used": False,
            "drake_visualizer_used": False,
            "friction_mapping": "shared_static_dynamic_only",
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
            failure_codes=["drake_rigid_replay_mismatch"],
        )
    return build_measurement_adapter_worker_result(
        request,
        status="completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=(
            first["penetration_m"] > point["penetration_unsafe_threshold_m"]
        ),
        runtime_observations=observations,
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("drake_rigid_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("drake_rigid_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Drake rigid-contact development case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_drake_rigid_measurement_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DRAKE_VERSION",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "WORKER_SCRIPT",
    "implementation_digest",
    "main",
    "run_drake_rigid_measurement_request",
]
