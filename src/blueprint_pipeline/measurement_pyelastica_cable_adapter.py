"""PyElastica cable/rod deformation development benchmark worker.

This worker executes a bounded, fixed-free Cosserat-rod cantilever under a
known gravity load.  It is a real PyElastica simulation port for the Q-DLO
cable lane, not a claim that a captured rope, hose, or cable has been
characterized.  The worker runs each public development case twice and emits a
prediction only when the complete sampled trace is identical.
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


IMPLEMENTATION_ID = "blueprint-pyelastica-cable-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "pyelastica_cantilever_cable.v1"
EXPECTED_ENGINE_VERSION = "0.3.3.post2"


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
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid")
    return result


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid")
    return value


def _vector(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise MeasurementAdapterExecutionError(f"pyelastica_adapter_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("pyelastica_adapter_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("pyelastica_adapter_protocol_invalid")
    if point.get("length_unit") != "meters":
        raise MeasurementAdapterExecutionError("pyelastica_adapter_length_unit_invalid")
    if point.get("mass_unit") != "kilograms":
        raise MeasurementAdapterExecutionError("pyelastica_adapter_mass_unit_invalid")
    if point.get("time_unit") != "seconds":
        raise MeasurementAdapterExecutionError("pyelastica_adapter_time_unit_invalid")
    start = _vector(point.get("start_m"), name="start")
    direction = _vector(point.get("direction"), name="direction")
    normal = _vector(point.get("normal"), name="normal")
    gravity = _vector(point.get("gravity_m_s2"), name="gravity")
    if start != [0.0, 0.0, 0.0]:
        raise MeasurementAdapterExecutionError("pyelastica_adapter_start_invalid")
    if direction != [1.0, 0.0, 0.0] or normal != [0.0, 0.0, 1.0]:
        raise MeasurementAdapterExecutionError("pyelastica_adapter_frame_convention_invalid")
    direction_norm = math.sqrt(sum(item * item for item in direction))
    normal_norm = math.sqrt(sum(item * item for item in normal))
    dot = sum(a * b for a, b in zip(direction, normal, strict=True))
    if not (
        math.isclose(direction_norm, 1.0, abs_tol=1e-12)
        and math.isclose(normal_norm, 1.0, abs_tol=1e-12)
        and math.isclose(dot, 0.0, abs_tol=1e-12)
    ):
        raise MeasurementAdapterExecutionError("pyelastica_adapter_frame_orthonormality_invalid")
    duration = _number(point.get("duration_s"), name="duration", minimum=1e-4, maximum=2.0)
    timestep = _number(point.get("timestep_s"), name="timestep", minimum=1e-7, maximum=1e-3)
    step_count_float = duration / timestep
    step_count = round(step_count_float)
    if step_count < 10 or not math.isclose(step_count_float, step_count, rel_tol=0.0, abs_tol=1e-9):
        raise MeasurementAdapterExecutionError("pyelastica_adapter_timestep_duration_mismatch")
    return {
        "n_elements": _integer(point.get("n_elements"), name="n_elements", minimum=6, maximum=128),
        "start_m": start,
        "direction": direction,
        "normal": normal,
        "base_length_m": _number(
            point.get("base_length_m"),
            name="base_length",
            minimum=0.05,
            maximum=5.0,
        ),
        "base_radius_m": _number(
            point.get("base_radius_m"),
            name="base_radius",
            minimum=1e-4,
            maximum=0.1,
        ),
        "density_kg_m3": _number(
            point.get("density_kg_m3"),
            name="density",
            minimum=1.0,
            maximum=50000.0,
        ),
        "youngs_modulus_pa": _number(
            point.get("youngs_modulus_pa"),
            name="youngs_modulus",
            minimum=1e3,
            maximum=1e12,
        ),
        "gravity_m_s2": gravity,
        "duration_s": duration,
        "timestep_s": timestep,
        "step_count": step_count,
        "damping_constant": _number(
            point.get("damping_constant"),
            name="damping_constant",
            minimum=0.0,
            maximum=1e4,
        ),
        "maximum_tip_displacement_m": _number(
            point.get("maximum_tip_displacement_m"),
            name="maximum_tip_displacement",
            minimum=0.0,
            maximum=10.0,
        ),
        "maximum_segment_strain": _number(
            point.get("maximum_segment_strain"),
            name="maximum_segment_strain",
            minimum=0.0,
            maximum=10.0,
        ),
    }


def _simulate(elastica: Any, np: Any, point: Mapping[str, Any]) -> dict[str, Any]:
    class Simulator(
        elastica.BaseSystemCollection,
        elastica.Constraints,
        elastica.Forcing,
        elastica.Damping,
        elastica.CallBacks,
    ):
        pass

    trace: list[dict[str, Any]] = []
    sample_stride = max(1, point["step_count"] // 20)

    class TraceCallback(elastica.CallBackBaseClass):
        def make_callback(self, system: Any, time: Any, current_step: int) -> None:
            if current_step % sample_stride == 0:
                trace.append(
                    {
                        "step": current_step,
                        "time_s": float(time),
                        "tip_position_m": [
                            float(item) for item in system.position_collection[:, -1]
                        ],
                    }
                )

    simulator = Simulator()
    rod = elastica.CosseratRod.straight_rod(
        point["n_elements"],
        np.asarray(point["start_m"], dtype=np.float64),
        np.asarray(point["direction"], dtype=np.float64),
        np.asarray(point["normal"], dtype=np.float64),
        point["base_length_m"],
        point["base_radius_m"],
        point["density_kg_m3"],
        youngs_modulus=point["youngs_modulus_pa"],
    )
    simulator.append(rod)
    simulator.constrain(rod).using(
        elastica.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    simulator.add_forcing_to(rod).using(
        elastica.GravityForces,
        acc_gravity=np.asarray(point["gravity_m_s2"], dtype=np.float64),
    )
    simulator.dampen(rod).using(
        elastica.AnalyticalLinearDamper,
        uniform_damping_constant=point["damping_constant"],
        time_step=point["timestep_s"],
    )
    simulator.collect_diagnostics(rod).using(TraceCallback)
    simulator.finalize()
    elastica.integrate(
        elastica.PositionVerlet(),
        simulator,
        final_time=point["duration_s"],
        n_steps=point["step_count"],
        progress_bar=False,
    )
    initial_tip = np.asarray([point["base_length_m"], 0.0, 0.0], dtype=np.float64)
    final_tip = rod.position_collection[:, -1].astype(np.float64)
    tip_displacement = float(np.linalg.norm(final_tip - initial_tip))
    segment_lengths = np.linalg.norm(np.diff(rod.position_collection, axis=1), axis=0)
    rest_length = point["base_length_m"] / point["n_elements"]
    maximum_segment_strain = float(np.max(np.abs(segment_lengths / rest_length - 1.0)))
    total_mass = float(np.sum(rod.mass))
    applied_gravity_force = float(total_mass * np.linalg.norm(np.asarray(point["gravity_m_s2"])))
    result = {
        "sample_stride": sample_stride,
        "sample_count": len(trace),
        "trace": trace,
        "final_tip_position_m": [float(item) for item in final_tip],
        "tip_displacement_m": tip_displacement,
        "maximum_segment_strain": maximum_segment_strain,
        "total_mass_kg": total_mass,
        "applied_gravity_force_n": applied_gravity_force,
    }
    result["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return result


def run_pyelastica_cable_request(
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
    implementation = request["implementation"]
    if implementation["implementation_id"] != IMPLEMENTATION_ID:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["pyelastica_adapter_implementation_id_mismatch"],
        )
    if implementation["implementation_version"] != IMPLEMENTATION_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["pyelastica_adapter_implementation_version_mismatch"],
        )
    if implementation["implementation_digest"] != implementation_digest():
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["pyelastica_adapter_implementation_digest_mismatch"],
        )
    try:
        import elastica
        import numpy as np
    except ImportError:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["pyelastica_adapter_runtime_unavailable"],
        )
    engine_version = importlib.metadata.version("pyelastica")
    base_observations["engine_version"] = engine_version
    if (
        engine_version != EXPECTED_ENGINE_VERSION
        or engine_version != runtime["target_engine_version"]
    ):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["pyelastica_adapter_target_version_mismatch"],
        )
    settings = dict(runtime["solver_settings"])
    if set(settings) != {"integrator", "replay_count"}:
        raise MeasurementAdapterExecutionError("pyelastica_adapter_solver_settings_invalid")
    if settings["integrator"] != "PositionVerlet":
        raise MeasurementAdapterExecutionError("pyelastica_adapter_integrator_invalid")
    if settings["replay_count"] != 2:
        raise MeasurementAdapterExecutionError("pyelastica_adapter_replay_count_invalid")
    point = _operating_point(request)
    first = _simulate(elastica, np, point)
    second = _simulate(elastica, np, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    unsafe = any(
        (
            first["tip_displacement_m"] > point["maximum_tip_displacement_m"],
            first["maximum_segment_strain"] > point["maximum_segment_strain"],
        )
    )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "state_trajectory": first["tip_displacement_m"],
        "force": first["applied_gravity_force_n"],
        "task_outcome": (
            "deformation_envelope_exceeded" if unsafe else "within_deformation_envelope"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations = {
        **base_observations,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_digest": implementation_digest(),
        "adapter_protocol": PROTOCOL_ID,
        "solver_settings_digest": runtime["solver_settings_digest"],
        "element_count": point["n_elements"],
        "step_count": point["step_count"],
        "sample_count": first["sample_count"],
        "final_tip_position_m": first["final_tip_position_m"],
        "tip_displacement_m": first["tip_displacement_m"],
        "maximum_segment_strain": first["maximum_segment_strain"],
        "total_mass_kg": first["total_mass_kg"],
        "applied_gravity_force_n": first["applied_gravity_force_n"],
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
            failure_codes=["pyelastica_adapter_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("pyelastica_adapter_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("pyelastica_adapter_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a PyElastica cable development measurement case"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_pyelastica_cable_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ENGINE_VERSION",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "implementation_digest",
    "main",
    "run_pyelastica_cable_request",
]
