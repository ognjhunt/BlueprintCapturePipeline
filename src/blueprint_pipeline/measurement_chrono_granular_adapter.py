"""PyChrono NSC spherical-granular development worker.

This adapter executes a bounded, synthetic 27-sphere column-collapse protocol
with Chrono's nonsmooth contact formulation.  It binds the official conda
package record, environment-owned OpenMP preload, solver, collision system,
material parameters, public case, implementation bytes, and exact replay.

It is deliberately not Chrono::Granular GPU evidence, a DEM material
calibration, a pouring/tool-interaction benchmark, physical evidence, or R7.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-chrono-nsc-spherical-granular-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "chrono_nsc_spherical_particle_column_collapse.v1"
EXPECTED_ENGINE_VERSION = "10.0.0"
EXPECTED_CHANNEL = "https://conda.anaconda.org/projectchrono/label/release"
WORKER_SCRIPT = Path(__file__).parents[2] / "scripts/measurement_chrono_granular_worker.py"


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
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid")
    return result


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid")
    return value


def _vector(value: Any, *, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MeasurementAdapterExecutionError(f"chrono_granular_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("chrono_granular_operating_point_invalid")
    point = dict(raw)
    exact = {
        "adapter_protocol": PROTOCOL_ID,
        "length_unit": "meters",
        "mass_unit": "kilograms",
        "time_unit": "seconds",
        "particle_shape": "sphere",
        "particle_size_distribution": "monodisperse",
        "cohesion_model": "none",
        "material_characterization_scope": "synthetic_parameters_only",
        "contact_method": "nonsmooth_contact",
        "collision_system": "bullet",
    }
    for key, expected in exact.items():
        if point.get(key) != expected:
            raise MeasurementAdapterExecutionError(f"chrono_granular_{key}_invalid")
    gravity = _vector(point.get("gravity_m_s2"), name="gravity", length=3)
    if gravity[2] >= 0:
        raise MeasurementAdapterExecutionError("chrono_granular_gravity_invalid")
    duration = _number(point.get("duration_s"), name="duration", minimum=0.05, maximum=10)
    timestep = _number(point.get("timestep_s"), name="timestep", minimum=1e-6, maximum=0.01)
    step_count_float = duration / timestep
    step_count = round(step_count_float)
    if step_count < 50 or not math.isclose(
        step_count_float, step_count, rel_tol=0.0, abs_tol=1e-9
    ):
        raise MeasurementAdapterExecutionError("chrono_granular_timestep_duration_mismatch")
    count_x = _integer(point.get("count_x"), name="count_x", minimum=2, maximum=8)
    count_y = _integer(point.get("count_y"), name="count_y", minimum=2, maximum=8)
    count_z = _integer(point.get("count_z"), name="count_z", minimum=2, maximum=8)
    particle_count = count_x * count_y * count_z
    if particle_count > 256:
        raise MeasurementAdapterExecutionError("chrono_granular_particle_count_invalid")
    minimum_spread = _number(
        point.get("minimum_spread_ratio"), name="minimum_spread_ratio", minimum=1, maximum=20
    )
    maximum_spread = _number(
        point.get("maximum_spread_ratio"), name="maximum_spread_ratio", minimum=1, maximum=20
    )
    if maximum_spread < minimum_spread:
        raise MeasurementAdapterExecutionError("chrono_granular_spread_envelope_invalid")
    return {
        "count_x": count_x,
        "count_y": count_y,
        "count_z": count_z,
        "particle_count": particle_count,
        "particle_radius_m": _number(
            point.get("particle_radius_m"), name="particle_radius", minimum=0.001, maximum=0.1
        ),
        "particle_density_kg_m3": _number(
            point.get("particle_density_kg_m3"),
            name="particle_density",
            minimum=10,
            maximum=20000,
        ),
        "contact_friction": _number(
            point.get("contact_friction"), name="contact_friction", minimum=0, maximum=2
        ),
        "rolling_friction": _number(
            point.get("rolling_friction"), name="rolling_friction", minimum=0, maximum=1
        ),
        "spinning_friction": _number(
            point.get("spinning_friction"), name="spinning_friction", minimum=0, maximum=1
        ),
        "restitution": _number(
            point.get("restitution"), name="restitution", minimum=0, maximum=1
        ),
        "spacing_factor": _number(
            point.get("spacing_factor"), name="spacing_factor", minimum=2, maximum=3
        ),
        "vertical_spacing_factor": _number(
            point.get("vertical_spacing_factor"),
            name="vertical_spacing_factor",
            minimum=2,
            maximum=3,
        ),
        "layer_stagger_x_fraction": _number(
            point.get("layer_stagger_x_fraction"),
            name="layer_stagger_x_fraction",
            minimum=0,
            maximum=0.99,
        ),
        "layer_stagger_y_fraction": _number(
            point.get("layer_stagger_y_fraction"),
            name="layer_stagger_y_fraction",
            minimum=0,
            maximum=0.99,
        ),
        "initial_ground_clearance_m": _number(
            point.get("initial_ground_clearance_m"),
            name="initial_ground_clearance",
            minimum=0,
            maximum=0.1,
        ),
        "ground_height_m": _number(
            point.get("ground_height_m"), name="ground_height", minimum=-5, maximum=5
        ),
        "gravity_m_s2": gravity,
        "duration_s": duration,
        "timestep_s": timestep,
        "step_count": step_count,
        "settle_speed_threshold_m_s": _number(
            point.get("settle_speed_threshold_m_s"),
            name="settle_speed_threshold",
            minimum=1e-6,
            maximum=10,
        ),
        "minimum_settled_fraction": _number(
            point.get("minimum_settled_fraction"),
            name="minimum_settled_fraction",
            minimum=0,
            maximum=1,
        ),
        "minimum_spread_ratio": minimum_spread,
        "maximum_spread_ratio": maximum_spread,
        "maximum_penetration_m": _number(
            point.get("maximum_penetration_m"), name="maximum_penetration", minimum=0
        ),
        "maximum_normal_contact_force_n": _number(
            point.get("maximum_normal_contact_force_n"),
            name="maximum_normal_contact_force",
            minimum=0,
        ),
    }


def _runtime_identity() -> tuple[Any, dict[str, Any]]:
    prefix = Path(sys.prefix).absolute()
    records = sorted((prefix / "conda-meta").glob("pychrono-*.json"))
    if len(records) != 1:
        raise MeasurementAdapterExecutionError("chrono_granular_conda_record_not_unique")
    try:
        record = json.loads(records[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("chrono_granular_conda_record_invalid") from exc
    subdir = str(record.get("subdir", ""))
    channel = str(record.get("channel", "")).rstrip("/")
    if subdir and channel.endswith(f"/{subdir}"):
        channel = channel[: -(len(subdir) + 1)]
    if (
        record.get("name") != "pychrono"
        or record.get("version") != EXPECTED_ENGINE_VERSION
        or channel != EXPECTED_CHANNEL
        or not str(record.get("build", "")).strip()
        or not subdir
    ):
        raise MeasurementAdapterExecutionError("chrono_granular_conda_identity_invalid")
    candidates = (
        prefix / "lib/libiomp5.dylib",
        prefix / "lib/libiomp5.so",
        prefix / "Library/bin/libiomp5md.dll",
    )
    openmp = next((path for path in candidates if path.is_file()), None)
    if openmp is not None:
        ctypes.CDLL(str(openmp), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    try:
        import pychrono.core as chrono
    except ImportError as exc:
        raise MeasurementAdapterExecutionError("chrono_granular_runtime_unavailable") from exc
    return chrono, {
        "engine_version": EXPECTED_ENGINE_VERSION,
        "package_build": str(record["build"]),
        "package_channel": channel,
        "package_subdir": subdir,
        "package_metadata_source": "conda-meta",
        "openmp_library": str(openmp) if openmp else None,
        "openmp_preload_used": openmp is not None,
    }


def _simulate(chrono: Any, point: Mapping[str, Any]) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(*point["gravity_m_s2"]))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetSolverType(chrono.ChSolver.Type_PSOR)
    material = chrono.ChContactMaterialNSC()
    material.SetFriction(point["contact_friction"])
    material.SetRollingFriction(point["rolling_friction"])
    material.SetSpinningFriction(point["spinning_friction"])
    material.SetRestitution(point["restitution"])
    ground = chrono.ChBodyEasyBox(2.0, 2.0, 0.1, 1000.0, False, True, material)
    ground.SetName("ground")
    ground.SetFixed(True)
    ground.SetPos(chrono.ChVector3d(0, 0, point["ground_height_m"] - 0.05))
    system.Add(ground)
    particles: list[Any] = []
    radius = point["particle_radius_m"]
    spacing = point["spacing_factor"] * radius
    for z_index in range(point["count_z"]):
        offset_x = point["layer_stagger_x_fraction"] * spacing if z_index % 2 else 0.0
        offset_y = point["layer_stagger_y_fraction"] * spacing if z_index % 3 else 0.0
        for y_index in range(point["count_y"]):
            for x_index in range(point["count_x"]):
                particle = chrono.ChBodyEasySphere(
                    radius,
                    point["particle_density_kg_m3"],
                    False,
                    True,
                    material,
                )
                particle.SetName(f"particle_{len(particles)}")
                particle.SetPos(
                    chrono.ChVector3d(
                        (x_index - (point["count_x"] - 1) / 2) * spacing + offset_x,
                        (y_index - (point["count_y"] - 1) / 2) * spacing + offset_y,
                        point["ground_height_m"]
                        + radius
                        + point["initial_ground_clearance_m"]
                        + z_index * point["vertical_spacing_factor"] * radius,
                    )
                )
                system.Add(particle)
                particles.append(particle)

    class ContactReporter(chrono.ReportContactCallback):
        def __init__(self) -> None:
            super().__init__()
            self.ground = False
            self.interparticle = False
            self.maximum_force = 0.0
            self.minimum_distance = 0.0

        def OnReportContact(
            self,
            _point_a: Any,
            _point_b: Any,
            _plane: Any,
            distance: float,
            _effective_radius: float,
            forces: Any,
            _torques: Any,
            contact_a: Any,
            contact_b: Any,
            _offset: int,
        ) -> bool:
            names = {
                contact_a.GetPhysicsItem().GetName(),
                contact_b.GetPhysicsItem().GetName(),
            }
            self.ground |= "ground" in names
            self.interparticle |= "ground" not in names
            self.maximum_force = max(self.maximum_force, float(forces.Length()))
            self.minimum_distance = min(self.minimum_distance, float(distance))
            return True

    def span() -> float:
        x = [float(body.GetPos().x) for body in particles]
        y = [float(body.GetPos().y) for body in particles]
        return max(max(x) - min(x), max(y) - min(y))

    initial_span = span()
    if initial_span <= 0:
        raise MeasurementAdapterExecutionError("chrono_granular_initial_span_invalid")
    reporter = ContactReporter()
    sample_stride = max(1, point["step_count"] // 20)
    trace: list[dict[str, Any]] = []
    maximum_contact_count = 0
    for step in range(point["step_count"]):
        system.DoStepDynamics(point["timestep_s"])
        system.GetContactContainer().ReportAllContacts(reporter)
        maximum_contact_count = max(
            maximum_contact_count, int(system.GetContactContainer().GetNumContacts())
        )
        if step % sample_stride == 0 or step == point["step_count"] - 1:
            speeds = [float(body.GetLinVel().Length()) for body in particles]
            positions = [body.GetPos() for body in particles]
            trace.append(
                {
                    "step": step,
                    "time_s": float(system.GetChTime()),
                    "horizontal_span_m": span(),
                    "centroid_m": [
                        sum(float(pos.x) for pos in positions) / len(positions),
                        sum(float(pos.y) for pos in positions) / len(positions),
                        sum(float(pos.z) for pos in positions) / len(positions),
                    ],
                    "settled_fraction": sum(
                        speed < point["settle_speed_threshold_m_s"] for speed in speeds
                    )
                    / len(speeds),
                    "contact_count": int(system.GetContactContainer().GetNumContacts()),
                }
            )
    speeds = [float(body.GetLinVel().Length()) for body in particles]
    positions = [body.GetPos() for body in particles]
    final_span = span()
    result = {
        "particle_count": len(particles),
        "particle_mass_kg": float(particles[0].GetMass()),
        "total_particle_mass_kg": sum(float(body.GetMass()) for body in particles),
        "sample_stride": sample_stride,
        "sample_count": len(trace),
        "trace": trace,
        "initial_horizontal_span_m": initial_span,
        "final_horizontal_span_m": final_span,
        "spread_ratio": final_span / initial_span,
        "final_maximum_height_m": max(float(pos.z) for pos in positions)
        + radius
        - point["ground_height_m"],
        "final_settled_fraction": sum(
            speed < point["settle_speed_threshold_m_s"] for speed in speeds
        )
        / len(speeds),
        "final_maximum_speed_m_s": max(speeds),
        "maximum_contact_count": maximum_contact_count,
        "maximum_normal_contact_force_n": reporter.maximum_force,
        "penetration_m": max(0.0, -reporter.minimum_distance),
        "ground_contact_observed": reporter.ground,
        "interparticle_contact_observed": reporter.interparticle,
    }
    result["trace_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def run_chrono_granular_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    observations: dict[str, Any] = {
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
                failure_codes=[f"chrono_granular_{code}"],
            )
    if runtime["target_engine_version"] != EXPECTED_ENGINE_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["chrono_granular_target_version_mismatch"],
        )
    if runtime["backend_id"] != "chrono-nsc-cpu-bullet-psor":
        raise MeasurementAdapterExecutionError("chrono_granular_backend_invalid")
    if runtime["precision"] != "float64":
        raise MeasurementAdapterExecutionError("chrono_granular_precision_invalid")
    settings = dict(runtime["solver_settings"])
    if settings != {
        "collision_system": "bullet",
        "contact_method": "nsc",
        "replay_count": 2,
        "solver": "psor",
        "timestepper": "euler_implicit_linearized",
    }:
        raise MeasurementAdapterExecutionError("chrono_granular_solver_settings_invalid")
    chrono, identity = _runtime_identity()
    observations.update(identity)
    point = _operating_point(request)
    first = _simulate(chrono, point)
    second = _simulate(chrono, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    contact_scope = first["ground_contact_observed"] and first["interparticle_contact_observed"]
    unsafe = any(
        (
            not contact_scope,
            first["spread_ratio"] < point["minimum_spread_ratio"],
            first["spread_ratio"] > point["maximum_spread_ratio"],
            first["final_settled_fraction"] < point["minimum_settled_fraction"],
            first["penetration_m"] > point["maximum_penetration_m"],
            first["maximum_normal_contact_force_n"]
            > point["maximum_normal_contact_force_n"],
        )
    )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "state_trajectory": first["spread_ratio"],
        "topology_contact": (
            "particle_ground_and_interparticle_contact"
            if contact_scope
            else "required_contact_scope_missing"
        ),
        "force": first["maximum_normal_contact_force_n"],
        "task_outcome": (
            "chrono_nsc_spherical_particle_envelope_exceeded"
            if unsafe
            else "within_chrono_nsc_spherical_particle_envelope"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations.update(
        {
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "adapter_protocol": PROTOCOL_ID,
            "solver_settings_digest": runtime["solver_settings_digest"],
            "contact_method": "nsc",
            "collision_system": "bullet",
            "solver": "psor",
            "timestepper": "euler_implicit_linearized",
            "chrono_granular_gpu_module_used": False,
            "particle_shape": "sphere",
            "particle_size_distribution": "monodisperse",
            "cohesion_model": "none",
            "material_characterization_scope": "synthetic_parameters_only",
            "particle_count": first["particle_count"],
            "particle_mass_kg": first["particle_mass_kg"],
            "total_particle_mass_kg": first["total_particle_mass_kg"],
            "step_count": point["step_count"],
            "sample_count": first["sample_count"],
            "initial_horizontal_span_m": first["initial_horizontal_span_m"],
            "final_horizontal_span_m": first["final_horizontal_span_m"],
            "spread_ratio": first["spread_ratio"],
            "final_maximum_height_m": first["final_maximum_height_m"],
            "final_settled_fraction": first["final_settled_fraction"],
            "final_maximum_speed_m_s": first["final_maximum_speed_m_s"],
            "maximum_contact_count": first["maximum_contact_count"],
            "maximum_normal_contact_force_n": first["maximum_normal_contact_force_n"],
            "penetration_m": first["penetration_m"],
            "ground_contact_observed": first["ground_contact_observed"],
            "interparticle_contact_observed": first["interparticle_contact_observed"],
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
            failure_codes=["chrono_granular_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("chrono_granular_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("chrono_granular_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_chrono_granular_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ENGINE_VERSION",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "WORKER_SCRIPT",
    "implementation_digest",
    "main",
    "run_chrono_granular_request",
]
