"""Newton 1.4 XPBD rigid-contact development worker.

The worker runs the same method-neutral sphere/box drop cases used by the
MuJoCo development suite. It verifies exact Newton and Warp identities, forces
the CPU XPBD backend into run-to-run deterministic mode, and emits only
development predictions after exact replay agreement. It grants no physical,
R5, R6, R7, or production authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-newton-xpbd-rigid-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "newton_xpbd_rigid_drop.v1"
NEWTON_VERSION = "1.4.0"
WARP_VERSION = "1.15.0"


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
        raise MeasurementAdapterExecutionError(f"newton_rigid_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"newton_rigid_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"newton_rigid_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"newton_rigid_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"newton_rigid_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("newton_rigid_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("newton_rigid_protocol_invalid")
    if point.get("protocol_family") != "rigid_body_drop":
        raise MeasurementAdapterExecutionError("newton_rigid_protocol_family_invalid")
    shape = str(point.get("body_shape", "")).strip()
    if shape not in {"sphere", "box"}:
        raise MeasurementAdapterExecutionError("newton_rigid_body_shape_invalid")
    if shape == "sphere":
        size = [_number(point.get("radius_m"), name="radius_m", minimum=1e-4)]
        resting_height = size[0]
        volume = 4.0 / 3.0 * math.pi * size[0] ** 3
    else:
        raw_size = point.get("half_size_m")
        if not isinstance(raw_size, list) or len(raw_size) != 3:
            raise MeasurementAdapterExecutionError("newton_rigid_half_size_invalid")
        size = [_number(item, name="half_size_m", minimum=1e-4) for item in raw_size]
        resting_height = size[2]
        volume = 8.0 * math.prod(size)
    friction = point.get("friction")
    if not isinstance(friction, list) or len(friction) != 3:
        raise MeasurementAdapterExecutionError("newton_rigid_friction_invalid")
    friction_values = [_number(item, name="friction", minimum=0.0) for item in friction]
    timestep = _number(point.get("timestep_s"), name="timestep_s", minimum=1e-6, maximum=0.05)
    duration = _number(point.get("duration_s"), name="duration_s", minimum=timestep, maximum=60.0)
    initial_height = _number(
        point.get("initial_height_m"),
        name="initial_height_m",
        minimum=resting_height + 1e-4,
    )
    mass = _number(point.get("mass_kg"), name="mass_kg", minimum=1e-6)
    return {
        "shape": shape,
        "size": size,
        "resting_height": resting_height,
        "density": mass / volume,
        "mass_kg": mass,
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


def _simulate(
    newton: Any,
    wp: Any,
    point: Mapping[str, Any],
    solver_settings: Mapping[str, Any],
) -> dict[str, Any]:
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder(gravity=point["gravity_m_s2"], up_axis=newton.Axis.Z)
        config = newton.ModelBuilder.ShapeConfig(
            density=point["density"],
            mu=point["friction"][0],
            mu_torsional=point["friction"][1],
            mu_rolling=point["friction"][2],
            restitution=0.0,
        )
        builder.add_ground_plane(cfg=config)
        body_id = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, point["initial_height_m"]),
                q=wp.quat_identity(),
            ),
            label="test_body",
        )
        if point["shape"] == "sphere":
            builder.add_shape_sphere(body_id, radius=point["size"][0], cfg=config)
        else:
            builder.add_shape_box(
                body_id,
                hx=point["size"][0],
                hy=point["size"][1],
                hz=point["size"][2],
                cfg=config,
            )
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=solver_settings["iterations"],
            rigid_contact_relaxation=solver_settings["rigid_contact_relaxation"],
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        first_contact_step: int | None = None
        maximum_contact_count = 0
        minimum_center_height = point["initial_height_m"]
        step_count = int(math.ceil(point["duration_s"] / point["timestep_s"]))
        for step in range(step_count):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            maximum_contact_count = max(maximum_contact_count, contact_count)
            if contact_count and first_contact_step is None:
                first_contact_step = step
            solver.step(
                state_0,
                state_1,
                control,
                contacts,
                point["timestep_s"],
            )
            state_0, state_1 = state_1, state_0
            minimum_center_height = min(
                minimum_center_height,
                float(state_0.body_q.numpy()[body_id, 2]),
            )
        final_position = [float(item) for item in state_0.body_q.numpy()[body_id, :3]]
        final_velocity = [float(item) for item in state_0.body_qd.numpy()[body_id]]
    penetration = max(0.0, point["resting_height"] - minimum_center_height)
    trace = {
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "maximum_contact_count": maximum_contact_count,
        "minimum_center_height_m": minimum_center_height,
        "final_position_m": final_position,
        "final_velocity": final_velocity,
        "penetration_m": penetration,
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_newton_rigid_measurement_request(
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
    )
    for key, expected, code in identities:
        if implementation[key] != expected:
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations=observations,
                failure_codes=[f"newton_rigid_{code}"],
            )
    try:
        if importlib.metadata.version("newton") != NEWTON_VERSION:
            raise importlib.metadata.PackageNotFoundError
        if importlib.metadata.version("warp-lang") != WARP_VERSION:
            raise importlib.metadata.PackageNotFoundError
        import newton
        import warp as wp
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["newton_rigid_package_or_version_unavailable"],
        )
    observations["engine_version"] = NEWTON_VERSION
    observations["warp_version"] = WARP_VERSION
    if runtime["target_engine_version"] != NEWTON_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["newton_rigid_target_version_mismatch"],
        )
    if runtime["backend_id"] != "newton-warp-cpu-xpbd":
        raise MeasurementAdapterExecutionError("newton_rigid_backend_invalid")
    if runtime["precision"] != "float32":
        raise MeasurementAdapterExecutionError("newton_rigid_precision_invalid")
    solver_settings = dict(runtime["solver_settings"])
    if set(solver_settings) != {
        "solver",
        "iterations",
        "rigid_contact_relaxation",
        "deterministic_mode",
    }:
        raise MeasurementAdapterExecutionError("newton_rigid_solver_settings_invalid")
    if solver_settings["solver"] != "XPBD":
        raise MeasurementAdapterExecutionError("newton_rigid_solver_invalid")
    if solver_settings["deterministic_mode"] != "RUN_TO_RUN":
        raise MeasurementAdapterExecutionError("newton_rigid_deterministic_mode_invalid")
    iterations = solver_settings["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise MeasurementAdapterExecutionError("newton_rigid_iterations_invalid")
    relaxation = _number(
        solver_settings["rigid_contact_relaxation"],
        name="rigid_contact_relaxation",
        minimum=0.0,
        maximum=1.0,
    )
    solver_settings["rigid_contact_relaxation"] = relaxation
    point = _operating_point(request)
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "warp-kernel-cache"
    wp.config.kernel_cache_dir = str(cache_root)
    wp.config.log_level = wp.LOG_WARNING
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
    first = _simulate(newton, wp, point, solver_settings)
    second = _simulate(newton, wp, point, solver_settings)
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
            "solver": "XPBD",
            "device": "cpu",
            "deterministic_mode": "RUN_TO_RUN",
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
            failure_codes=["newton_rigid_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("newton_rigid_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("newton_rigid_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Newton rigid-contact development case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_newton_rigid_measurement_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "NEWTON_VERSION",
    "PROTOCOL_ID",
    "WARP_VERSION",
    "implementation_digest",
    "main",
    "run_newton_rigid_measurement_request",
]
