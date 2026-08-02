"""MuJoCo spherical-particle granular development benchmark worker.

The worker executes a bounded unconfined column-collapse protocol for identical
noncohesive spheres.  It binds the exact MuJoCo version, particle geometry,
synthetic material parameters, solver, timestep, and public case, then runs the
complete trajectory twice.  This is development evidence for one rigid-sphere
contact model.  It is not DEM qualification, physical material
characterization, nonspherical/cohesive-grain evidence, or a production
granular route.
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


IMPLEMENTATION_ID = "blueprint-mujoco-spherical-granular-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "mujoco_spherical_particle_column_collapse.v1"
EXPECTED_ENGINE_VERSION = "3.11.0"


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
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid")
    return result


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid")
    return value


def _vector(value: Any, *, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MeasurementAdapterExecutionError(f"mujoco_granular_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_granular_operating_point_invalid")
    point = dict(raw)
    exact_values = {
        "adapter_protocol": PROTOCOL_ID,
        "length_unit": "meters",
        "mass_unit": "kilograms",
        "time_unit": "seconds",
        "particle_shape": "sphere",
        "particle_size_distribution": "monodisperse",
        "cohesion_model": "none",
        "material_characterization_scope": "synthetic_parameters_only",
        "restitution_characterization": "not_measured_contact_damping_only",
    }
    for key, expected in exact_values.items():
        if point.get(key) != expected:
            raise MeasurementAdapterExecutionError(f"mujoco_granular_{key}_invalid")
    gravity = _vector(point.get("gravity_m_s2"), name="gravity", length=3)
    if gravity[2] >= 0:
        raise MeasurementAdapterExecutionError("mujoco_granular_gravity_invalid")
    friction = _vector(point.get("contact_friction"), name="contact_friction", length=3)
    if any(value < 0 for value in friction):
        raise MeasurementAdapterExecutionError("mujoco_granular_contact_friction_invalid")
    solimp = _vector(point.get("contact_solimp"), name="contact_solimp", length=3)
    if not 0 < solimp[0] <= solimp[1] <= 1 or solimp[2] <= 0:
        raise MeasurementAdapterExecutionError("mujoco_granular_contact_solimp_invalid")
    duration = _number(point.get("duration_s"), name="duration", minimum=0.05, maximum=10.0)
    timestep = _number(point.get("timestep_s"), name="timestep", minimum=1e-6, maximum=0.01)
    step_count_float = duration / timestep
    step_count = round(step_count_float)
    if step_count < 50 or not math.isclose(step_count_float, step_count, rel_tol=0.0, abs_tol=1e-9):
        raise MeasurementAdapterExecutionError("mujoco_granular_timestep_duration_mismatch")
    count_x = _integer(point.get("count_x"), name="count_x", minimum=2, maximum=8)
    count_y = _integer(point.get("count_y"), name="count_y", minimum=2, maximum=8)
    count_z = _integer(point.get("count_z"), name="count_z", minimum=2, maximum=8)
    particle_count = count_x * count_y * count_z
    if particle_count > 256:
        raise MeasurementAdapterExecutionError("mujoco_granular_particle_count_invalid")
    minimum_spread_ratio = _number(
        point.get("minimum_spread_ratio"), name="minimum_spread_ratio", minimum=1.0, maximum=20.0
    )
    maximum_spread_ratio = _number(
        point.get("maximum_spread_ratio"), name="maximum_spread_ratio", minimum=1.0, maximum=20.0
    )
    if maximum_spread_ratio < minimum_spread_ratio:
        raise MeasurementAdapterExecutionError("mujoco_granular_spread_envelope_invalid")
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
            minimum=10.0,
            maximum=20000.0,
        ),
        "spacing_factor": _number(
            point.get("spacing_factor"), name="spacing_factor", minimum=2.0, maximum=3.0
        ),
        "layer_stagger_x_fraction": _number(
            point.get("layer_stagger_x_fraction"),
            name="layer_stagger_x_fraction",
            minimum=0.0,
            maximum=0.99,
        ),
        "layer_stagger_y_fraction": _number(
            point.get("layer_stagger_y_fraction"),
            name="layer_stagger_y_fraction",
            minimum=0.0,
            maximum=0.99,
        ),
        "initial_ground_clearance_m": _number(
            point.get("initial_ground_clearance_m"),
            name="initial_ground_clearance",
            minimum=0.0,
            maximum=0.1,
        ),
        "ground_height_m": _number(
            point.get("ground_height_m"), name="ground_height", minimum=-5.0, maximum=5.0
        ),
        "free_joint_damping": _number(
            point.get("free_joint_damping"), name="free_joint_damping", minimum=0.0, maximum=10.0
        ),
        "gravity_m_s2": gravity,
        "contact_friction": friction,
        "contact_time_constant_s": _number(
            point.get("contact_time_constant_s"),
            name="contact_time_constant",
            minimum=1e-5,
            maximum=0.1,
        ),
        "contact_damping_ratio": _number(
            point.get("contact_damping_ratio"),
            name="contact_damping_ratio",
            minimum=0.01,
            maximum=10.0,
        ),
        "contact_solimp": solimp,
        "duration_s": duration,
        "timestep_s": timestep,
        "step_count": step_count,
        "settle_speed_threshold_m_s": _number(
            point.get("settle_speed_threshold_m_s"),
            name="settle_speed_threshold",
            minimum=1e-6,
            maximum=10.0,
        ),
        "minimum_settled_fraction": _number(
            point.get("minimum_settled_fraction"),
            name="minimum_settled_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_spread_ratio": minimum_spread_ratio,
        "maximum_spread_ratio": maximum_spread_ratio,
        "maximum_penetration_m": _number(
            point.get("maximum_penetration_m"),
            name="maximum_penetration",
            minimum=0.0,
            maximum=0.1,
        ),
        "maximum_normal_contact_force_n": _number(
            point.get("maximum_normal_contact_force_n"),
            name="maximum_normal_contact_force",
            minimum=0.0,
            maximum=1e6,
        ),
    }


def _xml(point: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    radius = point["particle_radius_m"]
    spacing = point["spacing_factor"] * radius
    bodies: list[str] = []
    index = 0
    for z_index in range(point["count_z"]):
        offset_x = point["layer_stagger_x_fraction"] * spacing if z_index % 2 else 0.0
        offset_y = point["layer_stagger_y_fraction"] * spacing if z_index % 3 else 0.0
        for y_index in range(point["count_y"]):
            for x_index in range(point["count_x"]):
                x = (x_index - (point["count_x"] - 1) / 2) * spacing + offset_x
                y = (y_index - (point["count_y"] - 1) / 2) * spacing + offset_y
                z = (
                    point["ground_height_m"]
                    + radius
                    + point["initial_ground_clearance_m"]
                    + z_index * 1.98 * radius
                )
                bodies.append(
                    f'''    <body name="particle_{index}" pos="{x:.17g} {y:.17g} {z:.17g}">
      <joint type="free" damping="{point["free_joint_damping"]:.17g}"/>
      <geom name="particle_geom_{index}" type="sphere" size="{radius:.17g}"
            density="{point["particle_density_kg_m3"]:.17g}"/>
    </body>'''
                )
                index += 1
    gravity = " ".join(f"{value:.17g}" for value in point["gravity_m_s2"])
    friction = " ".join(f"{value:.17g}" for value in point["contact_friction"])
    solimp = " ".join(f"{value:.17g}" for value in point["contact_solimp"])
    return f'''<mujoco model="blueprint_measurement_spherical_granular">
  <option timestep="{point["timestep_s"]:.17g}" gravity="{gravity}"
          integrator="{settings["integrator"]}" solver="{settings["solver"]}"
          iterations="{settings["iterations"]}" cone="{settings["cone"]}"
          tolerance="{float(settings["tolerance"]):.17g}"/>
  <size memory="20M"/>
  <default>
    <geom friction="{friction}"
          solref="{point["contact_time_constant_s"]:.17g} {point["contact_damping_ratio"]:.17g}"
          solimp="{solimp}"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 {point["ground_height_m"]:.17g}"
          size="2 2 .1"/>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
'''


def _horizontal_span(np: Any, positions: Any) -> float:
    return float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])))


def _simulate(mujoco: Any, np: Any, model_xml: str, point: Mapping[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(model_xml)
    if model.nbody != point["particle_count"] + 1 or model.njnt != point["particle_count"]:
        raise MeasurementAdapterExecutionError("mujoco_granular_compiled_topology_invalid")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"particle_{index}")
            for index in range(point["particle_count"])
        ]
    )
    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    initial_positions = np.array(data.xpos[body_ids], copy=True)
    initial_span = _horizontal_span(np, initial_positions)
    if initial_span <= 0:
        raise MeasurementAdapterExecutionError("mujoco_granular_initial_span_invalid")
    sample_stride = max(1, point["step_count"] // 20)
    trace: list[dict[str, Any]] = []
    maximum_contact_count = 0
    maximum_normal_contact_force = 0.0
    minimum_contact_distance = 0.0
    ground_contact_observed = False
    interparticle_contact_observed = False
    contact_force = np.zeros(6)
    particle_mass = float(model.body_mass[body_ids[0]])
    for step in range(point["step_count"]):
        mujoco.mj_step(model, data)
        positions = np.asarray(data.xpos[body_ids])
        velocities = np.asarray(data.qvel).reshape(point["particle_count"], 6)[:, :3]
        speeds = np.linalg.norm(velocities, axis=1)
        maximum_contact_count = max(maximum_contact_count, int(data.ncon))
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            minimum_contact_distance = min(minimum_contact_distance, float(contact.dist))
            if ground_geom_id in {int(contact.geom1), int(contact.geom2)}:
                ground_contact_observed = True
            else:
                interparticle_contact_observed = True
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            maximum_normal_contact_force = max(
                maximum_normal_contact_force, abs(float(contact_force[0]))
            )
        if step % sample_stride == 0 or step == point["step_count"] - 1:
            trace.append(
                {
                    "step": step,
                    "time_s": float(data.time),
                    "horizontal_span_m": _horizontal_span(np, positions),
                    "centroid_m": [float(value) for value in np.mean(positions, axis=0)],
                    "translational_kinetic_energy_j": float(
                        0.5 * particle_mass * np.sum(speeds**2)
                    ),
                    "settled_fraction": float(
                        np.mean(speeds < point["settle_speed_threshold_m_s"])
                    ),
                    "contact_count": int(data.ncon),
                }
            )
    final_positions = np.asarray(data.xpos[body_ids])
    final_velocities = np.asarray(data.qvel).reshape(point["particle_count"], 6)[:, :3]
    final_speeds = np.linalg.norm(final_velocities, axis=1)
    final_span = _horizontal_span(np, final_positions)
    warnings = [
        {"warning_index": index, "count": int(row.number), "last_info": int(row.lastinfo)}
        for index, row in enumerate(data.warning)
        if row.number
    ]
    result = {
        "particle_count": point["particle_count"],
        "particle_mass_kg": particle_mass,
        "total_particle_mass_kg": particle_mass * point["particle_count"],
        "sample_stride": sample_stride,
        "sample_count": len(trace),
        "trace": trace,
        "initial_horizontal_span_m": initial_span,
        "final_horizontal_span_m": final_span,
        "spread_ratio": final_span / initial_span,
        "final_maximum_height_m": float(np.max(final_positions[:, 2]) - point["ground_height_m"]),
        "final_settled_fraction": float(
            np.mean(final_speeds < point["settle_speed_threshold_m_s"])
        ),
        "final_maximum_speed_m_s": float(np.max(final_speeds)),
        "maximum_contact_count": maximum_contact_count,
        "maximum_normal_contact_force_n": maximum_normal_contact_force,
        "penetration_m": max(0.0, -minimum_contact_distance),
        "ground_contact_observed": ground_contact_observed,
        "interparticle_contact_observed": interparticle_contact_observed,
        "warnings": warnings,
    }
    result["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return result


def run_mujoco_granular_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    base_observations = {
        "engine_version": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    implementation = request["implementation"]
    mismatch = next(
        (
            code
            for actual, expected, code in (
                (
                    implementation["implementation_id"],
                    IMPLEMENTATION_ID,
                    "mujoco_granular_implementation_id_mismatch",
                ),
                (
                    implementation["implementation_version"],
                    IMPLEMENTATION_VERSION,
                    "mujoco_granular_implementation_version_mismatch",
                ),
                (
                    implementation["implementation_digest"],
                    implementation_digest(),
                    "mujoco_granular_implementation_digest_mismatch",
                ),
            )
            if actual != expected
        ),
        None,
    )
    if mismatch:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=[mismatch],
        )
    try:
        import mujoco
        import numpy as np
    except ImportError:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["mujoco_granular_runtime_unavailable"],
        )
    engine_version = str(mujoco.__version__)
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
            failure_codes=["mujoco_granular_target_version_mismatch"],
        )
    settings = dict(runtime["solver_settings"])
    if set(settings) != {
        "cone",
        "integrator",
        "iterations",
        "particle_model",
        "replay_count",
        "solver",
        "tolerance",
    }:
        raise MeasurementAdapterExecutionError("mujoco_granular_solver_settings_invalid")
    if settings["particle_model"] != "rigid_sphere_contact":
        raise MeasurementAdapterExecutionError("mujoco_granular_particle_model_invalid")
    if settings["integrator"] != "Euler":
        raise MeasurementAdapterExecutionError("mujoco_granular_integrator_invalid")
    if settings["solver"] != "Newton" or settings["cone"] != "elliptic":
        raise MeasurementAdapterExecutionError("mujoco_granular_solver_invalid")
    _integer(settings["iterations"], name="iterations", minimum=1, maximum=1000)
    _number(settings["tolerance"], name="tolerance", minimum=0.0, maximum=1.0)
    if settings["replay_count"] != 2:
        raise MeasurementAdapterExecutionError("mujoco_granular_replay_count_invalid")
    point = _operating_point(request)
    model_xml = _xml(point, settings)
    model_digest = "sha256:" + hashlib.sha256(model_xml.encode()).hexdigest()
    first = _simulate(mujoco, np, model_xml, point)
    second = _simulate(mujoco, np, model_xml, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    contact_scope_satisfied = (
        first["ground_contact_observed"] and first["interparticle_contact_observed"]
    )
    unsafe = any(
        (
            bool(first["warnings"]),
            not contact_scope_satisfied,
            first["spread_ratio"] < point["minimum_spread_ratio"],
            first["spread_ratio"] > point["maximum_spread_ratio"],
            first["final_settled_fraction"] < point["minimum_settled_fraction"],
            first["penetration_m"] > point["maximum_penetration_m"],
            first["maximum_normal_contact_force_n"] > point["maximum_normal_contact_force_n"],
        )
    )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "state_trajectory": first["spread_ratio"],
        "topology_contact": (
            "particle_ground_and_interparticle_contact"
            if contact_scope_satisfied
            else "required_contact_scope_missing"
        ),
        "force": first["maximum_normal_contact_force_n"],
        "task_outcome": (
            "spherical_particle_envelope_exceeded"
            if unsafe
            else "within_spherical_particle_envelope"
        ),
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations = {
        **base_observations,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_digest": implementation_digest(),
        "adapter_protocol": PROTOCOL_ID,
        "model_digest": model_digest,
        "solver_settings_digest": runtime["solver_settings_digest"],
        "particle_model": "rigid_sphere_contact",
        "particle_shape": "sphere",
        "particle_size_distribution": "monodisperse",
        "cohesion_model": "none",
        "material_characterization_scope": "synthetic_parameters_only",
        "restitution_characterization": "not_measured_contact_damping_only",
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
        "warning_count": len(first["warnings"]),
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
            failure_codes=["mujoco_granular_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("mujoco_granular_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_granular_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a MuJoCo spherical-particle granular development case"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_mujoco_granular_request(_load_object(args.request))
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
    "run_mujoco_granular_request",
]
