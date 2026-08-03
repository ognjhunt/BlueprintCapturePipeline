"""Exact-source Chrono 10.0.0 Chrono::DEM CUDA development worker.

The Python boundary validates a public synthetic case, launches the separately
built exact-source C++ probe twice, and binds its CUDA/runtime observations to
the uniform measurement-adapter receipt. It never creates characterized-
material, Q-GRAN, R5-R7, production, physical-success, or safety authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-chrono-dem-cuda-synthetic-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "chrono_dem_cuda_synthetic_particle_settling.v1"
EXPECTED_ENGINE_VERSION = "10.0.0"
EXPECTED_SOURCE_COMMIT = "9faf13dd8f1128dd75ed233a9627027b0422c3f7"
BINARY_NAME = "measurement_chrono_dem_cuda_probe"
WORKER_SCRIPT = Path(__file__).parents[2] / "scripts/measurement_chrono_dem_cuda_worker.py"
PROBE_SOURCE = Path(__file__).parents[2] / "scripts/measurement_chrono_dem_cuda_probe.cpp"
PROBE_CMAKE = Path(__file__).parents[2] / "scripts/measurement_chrono_dem_cuda_probe.CMakeLists.txt"
PROBE_RESULT_SCHEMA_VERSION = "measurement_chrono_dem_cuda_probe_result.v1"


def implementation_digest() -> str:
    hasher = hashlib.sha256()
    for label, path in (
        ("adapter", Path(__file__)),
        ("worker", WORKER_SCRIPT),
        ("probe_source", PROBE_SOURCE),
        ("probe_cmake", PROBE_CMAKE),
    ):
        hasher.update(label.encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, float]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_operating_point_invalid")
    point = dict(raw)
    exact = {
        "adapter_protocol": PROTOCOL_ID,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "module": "chrono_dem",
        "native_length_unit": "centimeters",
        "native_mass_unit": "grams",
        "native_time_unit": "seconds",
        "output_length_unit": "meters",
        "output_force_unit": "newtons",
        "particle_shape": "sphere",
        "particle_size_distribution": "monodisperse",
        "cohesion_model": "none",
        "material_characterization_scope": "synthetic_parameters_only",
        "count_x": 3,
        "count_y": 3,
        "count_z": 3,
        "particle_radius_cm": 1.0,
        "gravity_cm_s2": -980.0,
    }
    for key, expected in exact.items():
        if point.get(key) != expected:
            raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_{key}_invalid")
    duration = _number(point.get("duration_s"), name="duration", minimum=0.1, maximum=2)
    timestep = _number(point.get("timestep_s"), name="timestep", minimum=1e-6, maximum=1e-3)
    step_count = duration / timestep
    if step_count < 100 or not math.isclose(step_count, round(step_count), abs_tol=1e-3):
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_timestep_duration_mismatch")
    return {
        "density_g_cm3": _number(
            point.get("density_g_cm3"), name="density", minimum=0.1, maximum=20
        ),
        "friction": _number(point.get("friction"), name="friction", minimum=0, maximum=2),
        "rolling_friction": _number(
            point.get("rolling_friction"), name="rolling_friction", minimum=0, maximum=1
        ),
        "duration_s": duration,
        "timestep_s": timestep,
        "settle_speed_threshold_cm_s": _number(
            point.get("settle_speed_threshold_cm_s"),
            name="settle_speed_threshold",
            minimum=0.01,
            maximum=100,
        ),
        "minimum_settled_fraction": _number(
            point.get("minimum_settled_fraction"),
            name="minimum_settled_fraction",
            minimum=0,
            maximum=1,
        ),
        "minimum_spread_ratio": _number(
            point.get("minimum_spread_ratio"), name="minimum_spread_ratio", minimum=0.5
        ),
        "maximum_spread_ratio": _number(
            point.get("maximum_spread_ratio"), name="maximum_spread_ratio", minimum=0.5
        ),
        "maximum_penetration_m": _number(
            point.get("maximum_penetration_m"), name="maximum_penetration", minimum=0
        ),
        "maximum_static_weight_relative_error": _number(
            point.get("maximum_static_weight_relative_error"),
            name="maximum_static_weight_relative_error",
            minimum=0,
            maximum=1,
        ),
    }


def _validate_probe_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    exact = {
        "schema_version": PROBE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "chrono_version": EXPECTED_ENGINE_VERSION,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "chrono_dem_module_used": True,
        "cuda_device_count": 1,
        "particle_count": 27,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            errors.append(f"chrono_dem_cuda_probe_{key}_invalid")
    for key in (
        "density_g_cm3",
        "friction",
        "rolling_friction",
        "duration_s",
        "timestep_s",
        "initial_horizontal_span_m",
        "final_horizontal_span_m",
        "spread_ratio",
        "final_settled_fraction",
        "final_maximum_speed_m_s",
        "maximum_contact_count",
        "expected_static_weight_n",
        "final_ground_reaction_force_n",
        "maximum_ground_reaction_force_n",
        "penetration_m",
    ):
        try:
            _number(result.get(key), name=f"probe_{key}")
        except MeasurementAdapterExecutionError as exc:
            errors.extend(exc.codes)
    if not str(result.get("cuda_device_name", "")).strip():
        errors.append("chrono_dem_cuda_probe_device_name_missing")
    if not str(result.get("cuda_compute_capability", "")).strip():
        errors.append("chrono_dem_cuda_probe_compute_capability_missing")
    trace = result.get("trace")
    if (
        not isinstance(trace, list)
        or len(trace) != 20
        or not all(isinstance(row, Mapping) for row in trace)
    ):
        errors.append("chrono_dem_cuda_probe_trace_invalid")
    else:
        for index, row in enumerate(trace):
            if not isinstance(row.get("centroid_m"), list) or len(row["centroid_m"]) != 3:
                errors.append(f"chrono_dem_cuda_probe_trace_centroid_invalid:{index}")
            for key in (
                "time_s",
                "horizontal_span_m",
                "maximum_speed_m_s",
                "settled_fraction",
                "contact_count",
                "kinetic_energy_native",
                "ground_reaction_force_n",
            ):
                try:
                    _number(row.get(key), name=f"probe_trace_{key}")
                except MeasurementAdapterExecutionError as exc:
                    errors.extend(exc.codes)
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    result["trace_digest"] = _digest({"trace": trace})
    result["probe_result_digest"] = _digest(result)
    return result


def _run_probe(binary: Path, point: Mapping[str, float]) -> dict[str, Any]:
    argv = [
        str(binary),
        "--density-g-cm3",
        str(point["density_g_cm3"]),
        "--friction",
        str(point["friction"]),
        "--rolling-friction",
        str(point["rolling_friction"]),
        "--duration-s",
        str(point["duration_s"]),
        "--timestep-s",
        str(point["timestep_s"]),
        "--settle-speed-threshold-cm-s",
        str(point["settle_speed_threshold_cm_s"]),
    ]
    completed = subprocess.run(  # nosec B603 - exact binary and numeric argv, no shell
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise MeasurementAdapterExecutionError(
            f"chrono_dem_cuda_probe_exit_nonzero:{completed.returncode}"
        )
    if len(completed.stdout.encode()) > 512_000:
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_probe_output_too_large")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_probe_output_invalid") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_probe_output_not_object")
    result = _validate_probe_result(value)
    for key in (
        "density_g_cm3",
        "friction",
        "rolling_friction",
        "duration_s",
        "timestep_s",
    ):
        if not math.isclose(
            float(result[key]),
            float(point[key]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise MeasurementAdapterExecutionError(f"chrono_dem_cuda_probe_{key}_binding_mismatch")
    return result


def run_chrono_dem_cuda_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    observations: dict[str, Any] = {
        "engine_version": "unavailable",
        "source_commit": EXPECTED_SOURCE_COMMIT,
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
                failure_codes=[f"chrono_dem_cuda_{code}"],
            )
    if runtime["target_engine_version"] != EXPECTED_ENGINE_VERSION:
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_target_version_mismatch")
    if runtime["backend_id"] != "chrono-dem-cuda" or runtime["precision"] != "float32":
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_runtime_invalid")
    settings = dict(runtime["solver_settings"])
    if settings != {
        "binary_name": BINARY_NAME,
        "cuda_device_count": 1,
        "module": "chrono_dem",
        "replay_count": 2,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }:
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_solver_settings_invalid")
    point = _operating_point(request)
    if point["maximum_spread_ratio"] < point["minimum_spread_ratio"]:
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_spread_envelope_invalid")
    raw_binary = shutil.which(BINARY_NAME)
    if not raw_binary:
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_probe_binary_unavailable")
    binary = Path(raw_binary).resolve()
    if not binary.is_file() or binary.is_symlink():
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_probe_binary_invalid")
    first = _run_probe(binary, point)
    second = _run_probe(binary, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    force_error = abs(
        first["final_ground_reaction_force_n"] - first["expected_static_weight_n"]
    ) / max(first["expected_static_weight_n"], 1e-12)
    contact_observed = first["maximum_contact_count"] > 0
    unsafe = any(
        (
            not contact_observed,
            first["spread_ratio"] < point["minimum_spread_ratio"],
            first["spread_ratio"] > point["maximum_spread_ratio"],
            first["final_settled_fraction"] < point["minimum_settled_fraction"],
            first["penetration_m"] > point["maximum_penetration_m"],
            force_error > point["maximum_static_weight_relative_error"],
        )
    )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "state_trajectory": first["spread_ratio"],
        "topology_contact": (
            "particle_contact_observed" if contact_observed else "particle_contact_missing"
        ),
        "force": first["final_ground_reaction_force_n"],
        "task_outcome": (
            "chrono_dem_cuda_synthetic_envelope_exceeded"
            if unsafe
            else "within_chrono_dem_cuda_synthetic_envelope"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations.update(
        {
            "engine_version": first["chrono_version"],
            "source_commit": first["source_commit"],
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "adapter_protocol": PROTOCOL_ID,
            "solver_settings_digest": runtime["solver_settings_digest"],
            "chrono_dem_module_used": first["chrono_dem_module_used"],
            "cuda_available": True,
            "cuda_device_count": first["cuda_device_count"],
            "cuda_device_name": first["cuda_device_name"],
            "cuda_compute_capability": first["cuda_compute_capability"],
            "cpu_fallback_used": False,
            "binary_digest": _file_digest(binary),
            "probe_source_digest": _file_digest(PROBE_SOURCE),
            "probe_cmake_digest": _file_digest(PROBE_CMAKE),
            "particle_count": first["particle_count"],
            "material_characterization_scope": "synthetic_parameters_only",
            "pouring_or_tool_interaction_included": False,
            "spread_ratio": first["spread_ratio"],
            "final_settled_fraction": first["final_settled_fraction"],
            "maximum_contact_count": first["maximum_contact_count"],
            "expected_static_weight_n": first["expected_static_weight_n"],
            "final_ground_reaction_force_n": first["final_ground_reaction_force_n"],
            "static_weight_relative_error": force_error,
            "penetration_m": first["penetration_m"],
            "trace_digest": first["trace_digest"],
            "repeat_trace_digest": second["trace_digest"],
            "deterministic_replay_match": replay_match,
            "q_gran_qualification_created": False,
            "r5_evidence_created": False,
            "r6_decision_created": False,
            "r7_admission_created": False,
            "physical_success_established": False,
        }
    )
    if not replay_match:
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics=metrics,
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["chrono_dem_cuda_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("chrono_dem_cuda_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_chrono_dem_cuda_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BINARY_NAME",
    "EXPECTED_ENGINE_VERSION",
    "EXPECTED_SOURCE_COMMIT",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROBE_CMAKE",
    "PROBE_SOURCE",
    "PROTOCOL_ID",
    "WORKER_SCRIPT",
    "implementation_digest",
    "main",
    "run_chrono_dem_cuda_request",
]
