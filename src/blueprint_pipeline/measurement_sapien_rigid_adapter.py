"""SAPIEN 3.0.3 / PhysX CPU rigid-contact development worker.

This port runs the shared method-neutral sphere/box drop cases without a
renderer or ManiSkill task layer. It records exact SAPIEN/PhysX/runtime identity
and requires full replay agreement before emitting development predictions.
It is not a physical benchmark, ManiSkill qualification, or R7 route.
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


IMPLEMENTATION_ID = "blueprint-sapien-physx-cpu-rigid-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "sapien_physx_cpu_rigid_drop.v1"
SAPIEN_VERSION = "3.0.3"


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
        raise MeasurementAdapterExecutionError(f"sapien_rigid_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"sapien_rigid_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"sapien_rigid_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"sapien_rigid_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"sapien_rigid_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("sapien_rigid_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("sapien_rigid_protocol_invalid")
    if point.get("protocol_family") != "rigid_body_drop":
        raise MeasurementAdapterExecutionError("sapien_rigid_protocol_family_invalid")
    shape = str(point.get("body_shape", "")).strip()
    if shape not in {"sphere", "box"}:
        raise MeasurementAdapterExecutionError("sapien_rigid_body_shape_invalid")
    if shape == "sphere":
        size = [_number(point.get("radius_m"), name="radius_m", minimum=1e-4)]
        resting_height = size[0]
        volume = 4.0 / 3.0 * math.pi * size[0] ** 3
    else:
        raw_size = point.get("half_size_m")
        if not isinstance(raw_size, list) or len(raw_size) != 3:
            raise MeasurementAdapterExecutionError("sapien_rigid_half_size_invalid")
        size = [_number(item, name="half_size_m", minimum=1e-4) for item in raw_size]
        resting_height = size[2]
        volume = 8.0 * math.prod(size)
    friction = point.get("friction")
    if not isinstance(friction, list) or len(friction) != 3:
        raise MeasurementAdapterExecutionError("sapien_rigid_friction_invalid")
    friction_values = [_number(item, name="friction", minimum=0.0) for item in friction]
    timestep = _number(point.get("timestep_s"), name="timestep_s", minimum=1e-6, maximum=0.05)
    duration = _number(point.get("duration_s"), name="duration_s", minimum=timestep, maximum=60.0)
    mass = _number(point.get("mass_kg"), name="mass_kg", minimum=1e-6)
    return {
        "shape": shape,
        "size": size,
        "resting_height": resting_height,
        "density": mass / volume,
        "mass_kg": mass,
        "initial_height_m": _number(
            point.get("initial_height_m"),
            name="initial_height_m",
            minimum=resting_height + 1e-4,
        ),
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


def _simulate(
    sapien: Any,
    point: Mapping[str, Any],
    solver_settings: Mapping[str, Any],
) -> dict[str, Any]:
    sapien.physx.set_scene_config(
        gravity=[0.0, 0.0, point["gravity_m_s2"]],
        enable_pcm=True,
        enable_tgs=solver_settings["enable_tgs"],
        enable_ccd=False,
        enable_enhanced_determinism=solver_settings["enhanced_determinism"],
        enable_friction_every_iteration=True,
        cpu_workers=solver_settings["cpu_workers"],
    )
    physics = sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([physics])
    scene.set_timestep(point["timestep_s"])
    material = scene.create_physical_material(point["friction"][0], point["friction"][0], 0.0)
    scene.add_ground(0.0, render=False, material=material)
    builder = scene.create_actor_builder()
    if point["shape"] == "sphere":
        builder.add_sphere_collision(
            radius=point["size"][0], material=material, density=point["density"]
        )
    else:
        builder.add_box_collision(
            half_size=point["size"], material=material, density=point["density"]
        )
    actor = builder.build(name="test_body")
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, point["initial_height_m"]]))
    component = next(
        item
        for item in actor.components
        if isinstance(item, sapien.physx.PhysxRigidDynamicComponent)
    )
    component.set_solver_position_iterations(solver_settings["position_iterations"])
    component.set_solver_velocity_iterations(solver_settings["velocity_iterations"])
    step_count = int(math.ceil(point["duration_s"] / point["timestep_s"]))
    first_contact_step: int | None = None
    maximum_contact_count = 0
    minimum_separation = 0.0
    minimum_center_height = point["initial_height_m"]
    for step in range(step_count):
        scene.step()
        minimum_center_height = min(minimum_center_height, float(actor.pose.p[2]))
        contacts = scene.get_contacts()
        maximum_contact_count = max(maximum_contact_count, len(contacts))
        if contacts and first_contact_step is None:
            first_contact_step = step
        for contact in contacts:
            for contact_point in contact.points:
                minimum_separation = min(minimum_separation, float(contact_point.separation))
    final_position = [float(item) for item in actor.pose.p]
    final_velocity = [
        *[float(item) for item in component.linear_velocity],
        *[float(item) for item in component.angular_velocity],
    ]
    penetration = max(
        0.0,
        point["resting_height"] - minimum_center_height,
        -minimum_separation,
    )
    trace = {
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "maximum_contact_count": maximum_contact_count,
        "minimum_contact_separation_m": minimum_separation,
        "minimum_center_height_m": minimum_center_height,
        "final_position_m": final_position,
        "final_velocity": final_velocity,
        "observed_mass_kg": float(component.mass),
        "penetration_m": penetration,
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_sapien_rigid_measurement_request(
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
                failure_codes=[f"sapien_rigid_{code}"],
            )
    try:
        if importlib.metadata.version("sapien") != SAPIEN_VERSION:
            raise importlib.metadata.PackageNotFoundError
        import sapien
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["sapien_rigid_package_or_version_unavailable"],
        )
    observations["engine_version"] = SAPIEN_VERSION
    observations["physx_version"] = str(sapien.physx.version())
    if runtime["target_engine_version"] != SAPIEN_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["sapien_rigid_target_version_mismatch"],
        )
    if runtime["backend_id"] != "sapien-physx-cpu":
        raise MeasurementAdapterExecutionError("sapien_rigid_backend_invalid")
    if runtime["precision"] != "float32":
        raise MeasurementAdapterExecutionError("sapien_rigid_precision_invalid")
    solver = dict(runtime["solver_settings"])
    if set(solver) != {
        "enhanced_determinism",
        "enable_tgs",
        "cpu_workers",
        "position_iterations",
        "velocity_iterations",
    }:
        raise MeasurementAdapterExecutionError("sapien_rigid_solver_settings_invalid")
    if solver["enhanced_determinism"] is not True or solver["enable_tgs"] is not True:
        raise MeasurementAdapterExecutionError("sapien_rigid_determinism_invalid")
    if solver["cpu_workers"] != 0:
        raise MeasurementAdapterExecutionError("sapien_rigid_cpu_workers_invalid")
    for key in ("position_iterations", "velocity_iterations"):
        value = solver[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise MeasurementAdapterExecutionError(f"sapien_rigid_{key}_invalid")
    point = _operating_point(request)
    first = _simulate(sapien, point, solver)
    second = _simulate(sapien, point, solver)
    replay_match = first["trace_digest"] == second["trace_digest"]
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics = {
        "penetration": first["penetration_m"],
        "contact_sequence": (
            "ground_contact" if first["first_contact_step"] is not None else "no_contact"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations.update(
        {
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "solver_settings_digest": runtime["solver_settings_digest"],
            "device": "cpu",
            "renderer_created": False,
            "maniskill_runtime_used": False,
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
            failure_codes=["sapien_rigid_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("sapien_rigid_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("sapien_rigid_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a SAPIEN rigid-contact development case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_sapien_rigid_measurement_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "SAPIEN_VERSION",
    "implementation_digest",
    "main",
    "run_sapien_rigid_measurement_request",
]
