"""MuJoCo flex-cloth development benchmark worker.

The worker executes a narrow 2D ``flexcomp`` stretch model with four pinned
corners, gravity sag, and optional ground contact.  It binds the exact MuJoCo
version, flex material settings, topology, solver, timestep, and public case,
then runs the complete trajectory twice.  This is Q-CLOTH development evidence
for one flex formulation, not generic garment, bending, self-contact, topology
change, captured-material, or physical qualification evidence.
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


IMPLEMENTATION_ID = "blueprint-mujoco-flex-cloth-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "mujoco_flex_cloth_sag.v1"
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
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid")
    return result


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid")
    return value


def _vector(value: Any, *, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MeasurementAdapterExecutionError(f"mujoco_cloth_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_cloth_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("mujoco_cloth_protocol_invalid")
    if point.get("length_unit") != "meters":
        raise MeasurementAdapterExecutionError("mujoco_cloth_length_unit_invalid")
    if point.get("mass_unit") != "kilograms":
        raise MeasurementAdapterExecutionError("mujoco_cloth_mass_unit_invalid")
    if point.get("time_unit") != "seconds":
        raise MeasurementAdapterExecutionError("mujoco_cloth_time_unit_invalid")
    if point.get("pin_pattern") != "four_corners":
        raise MeasurementAdapterExecutionError("mujoco_cloth_pin_pattern_invalid")
    if point.get("elastic_formulation") != "stretch_only":
        raise MeasurementAdapterExecutionError("mujoco_cloth_elastic_formulation_invalid")
    if point.get("self_collision") != "none":
        raise MeasurementAdapterExecutionError("mujoco_cloth_self_collision_scope_invalid")
    gravity = _vector(point.get("gravity_m_s2"), name="gravity", length=3)
    if gravity[2] >= 0:
        raise MeasurementAdapterExecutionError("mujoco_cloth_gravity_invalid")
    contact_friction = _vector(point.get("contact_friction"), name="contact_friction", length=3)
    if any(value < 0 for value in contact_friction):
        raise MeasurementAdapterExecutionError("mujoco_cloth_contact_friction_invalid")
    duration = _number(point.get("duration_s"), name="duration", minimum=0.01, maximum=10.0)
    timestep = _number(point.get("timestep_s"), name="timestep", minimum=1e-6, maximum=0.01)
    step_count_float = duration / timestep
    step_count = round(step_count_float)
    if step_count < 10 or not math.isclose(step_count_float, step_count, rel_tol=0.0, abs_tol=1e-9):
        raise MeasurementAdapterExecutionError("mujoco_cloth_timestep_duration_mismatch")
    count_x = _integer(point.get("count_x"), name="count_x", minimum=4, maximum=32)
    count_y = _integer(point.get("count_y"), name="count_y", minimum=4, maximum=32)
    initial_height = _number(
        point.get("initial_height_m"),
        name="initial_height",
        minimum=0.01,
        maximum=5.0,
    )
    ground_height = _number(
        point.get("ground_height_m"),
        name="ground_height",
        minimum=-5.0,
        maximum=5.0,
    )
    radius = _number(
        point.get("collision_radius_m"),
        name="collision_radius",
        minimum=0.0,
        maximum=0.05,
    )
    if initial_height <= ground_height + radius:
        raise MeasurementAdapterExecutionError("mujoco_cloth_initial_ground_clearance_invalid")
    return {
        "count_x": count_x,
        "count_y": count_y,
        "spacing_m": _number(point.get("spacing_m"), name="spacing", minimum=0.005, maximum=0.5),
        "initial_height_m": initial_height,
        "ground_height_m": ground_height,
        "total_mass_kg": _number(
            point.get("total_mass_kg"), name="total_mass", minimum=1e-5, maximum=100.0
        ),
        "collision_radius_m": radius,
        "youngs_modulus_pa": _number(
            point.get("youngs_modulus_pa"),
            name="youngs_modulus",
            minimum=1.0,
            maximum=1e9,
        ),
        "poisson_ratio": _number(
            point.get("poisson_ratio"), name="poisson_ratio", minimum=0.0, maximum=0.49
        ),
        "thickness_m": _number(
            point.get("thickness_m"), name="thickness", minimum=1e-5, maximum=0.1
        ),
        "edge_damping": _number(
            point.get("edge_damping"), name="edge_damping", minimum=0.0, maximum=1e5
        ),
        "gravity_m_s2": gravity,
        "contact_friction": contact_friction,
        "duration_s": duration,
        "timestep_s": timestep,
        "step_count": step_count,
        "maximum_sag_m": _number(
            point.get("maximum_sag_m"), name="maximum_sag", minimum=0.0, maximum=10.0
        ),
        "maximum_edge_strain": _number(
            point.get("maximum_edge_strain"),
            name="maximum_edge_strain",
            minimum=0.0,
            maximum=10.0,
        ),
        "maximum_penetration_m": _number(
            point.get("maximum_penetration_m"),
            name="maximum_penetration",
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _xml(point: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    friction = " ".join(f"{value:.17g}" for value in point["contact_friction"])
    gravity = " ".join(f"{value:.17g}" for value in point["gravity_m_s2"])
    last_x = point["count_x"] - 1
    last_y = point["count_y"] - 1
    return f"""<mujoco model="blueprint_measurement_flex_cloth">
  <option timestep="{point["timestep_s"]:.17g}" gravity="{gravity}"
          integrator="{settings["integrator"]}" solver="{settings["solver"]}"
          iterations="{settings["iterations"]}"
          tolerance="{float(settings["tolerance"]):.17g}"/>
  <size memory="10M"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 {point["ground_height_m"]:.17g}"
          size="2 2 .1" friction="{friction}"/>
    <flexcomp name="cloth" type="grid" dim="2"
              count="{point["count_x"]} {point["count_y"]} 1"
              spacing="{point["spacing_m"]:.17g} {point["spacing_m"]:.17g} {point["spacing_m"]:.17g}"
              pos="0 0 {point["initial_height_m"]:.17g}"
              mass="{point["total_mass_kg"]:.17g}"
              radius="{point["collision_radius_m"]:.17g}">
      <contact condim="3" solref=".01 1" solimp=".95 .99 .0001"
               friction="{friction}" selfcollide="none"/>
      <edge equality="false" damping="{point["edge_damping"]:.17g}"/>
      <elasticity young="{point["youngs_modulus_pa"]:.17g}"
                  poisson="{point["poisson_ratio"]:.17g}"
                  thickness="{point["thickness_m"]:.17g}"
                  elastic2d="stretch"/>
      <pin grid="0 0"/>
      <pin grid="{last_x} 0"/>
      <pin grid="0 {last_y}"/>
      <pin grid="{last_x} {last_y}"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""


def _simulate(mujoco: Any, np: Any, model_xml: str, point: Mapping[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(model_xml)
    if model.nflex != 1 or model.flex_dim[0] != 2:
        raise MeasurementAdapterExecutionError("mujoco_cloth_compiled_topology_invalid")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_edges = np.array(data.flexedge_length, copy=True)
    initial_min_z = float(np.min(data.flexvert_xpos[:, 2]))
    sample_stride = max(1, point["step_count"] // 20)
    trace: list[dict[str, Any]] = []
    maximum_sag = 0.0
    maximum_edge_strain = 0.0
    maximum_contact_count = 0
    minimum_contact_distance = 0.0
    for step in range(point["step_count"]):
        mujoco.mj_step(model, data)
        current_min_z = float(np.min(data.flexvert_xpos[:, 2]))
        sag = max(0.0, initial_min_z - current_min_z)
        edge_strain = float(np.max(np.abs(data.flexedge_length / initial_edges - 1.0)))
        maximum_sag = max(maximum_sag, sag)
        maximum_edge_strain = max(maximum_edge_strain, edge_strain)
        maximum_contact_count = max(maximum_contact_count, int(data.ncon))
        for contact_index in range(data.ncon):
            minimum_contact_distance = min(
                minimum_contact_distance, float(data.contact[contact_index].dist)
            )
        if step % sample_stride == 0 or step == point["step_count"] - 1:
            trace.append(
                {
                    "step": step,
                    "time_s": float(data.time),
                    "minimum_vertex_height_m": current_min_z,
                    "maximum_edge_strain": edge_strain,
                    "contact_count": int(data.ncon),
                }
            )
    warnings = [
        {"warning_index": index, "count": int(row.number), "last_info": int(row.lastinfo)}
        for index, row in enumerate(data.warning)
        if row.number
    ]
    penetration = max(0.0, -minimum_contact_distance)
    result = {
        "vertex_count": int(model.nflexvert),
        "edge_count": int(model.flex_edgenum[0]),
        "element_count": int(model.flex_elemnum[0]),
        "sample_stride": sample_stride,
        "sample_count": len(trace),
        "trace": trace,
        "final_minimum_vertex_height_m": float(np.min(data.flexvert_xpos[:, 2])),
        "maximum_sag_m": maximum_sag,
        "maximum_edge_strain": maximum_edge_strain,
        "maximum_contact_count": maximum_contact_count,
        "penetration_m": penetration,
        "warnings": warnings,
    }
    result["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return result


def run_mujoco_flex_cloth_request(
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
    mismatch = next(
        (
            code
            for actual, expected, code in (
                (
                    implementation["implementation_id"],
                    IMPLEMENTATION_ID,
                    "mujoco_cloth_implementation_id_mismatch",
                ),
                (
                    implementation["implementation_version"],
                    IMPLEMENTATION_VERSION,
                    "mujoco_cloth_implementation_version_mismatch",
                ),
                (
                    implementation["implementation_digest"],
                    implementation_digest(),
                    "mujoco_cloth_implementation_digest_mismatch",
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
            failure_codes=["mujoco_cloth_runtime_unavailable"],
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
            failure_codes=["mujoco_cloth_target_version_mismatch"],
        )
    settings = dict(runtime["solver_settings"])
    if set(settings) != {
        "elastic2d",
        "integrator",
        "iterations",
        "replay_count",
        "solver",
        "tolerance",
    }:
        raise MeasurementAdapterExecutionError("mujoco_cloth_solver_settings_invalid")
    if settings["elastic2d"] != "stretch":
        raise MeasurementAdapterExecutionError("mujoco_cloth_elastic2d_invalid")
    if settings["integrator"] not in {"Euler", "implicitfast"}:
        raise MeasurementAdapterExecutionError("mujoco_cloth_integrator_invalid")
    if settings["solver"] not in {"CG", "Newton"}:
        raise MeasurementAdapterExecutionError("mujoco_cloth_solver_invalid")
    _integer(settings["iterations"], name="iterations", minimum=1, maximum=1000)
    _number(settings["tolerance"], name="tolerance", minimum=0.0, maximum=1.0)
    if settings["replay_count"] != 2:
        raise MeasurementAdapterExecutionError("mujoco_cloth_replay_count_invalid")
    point = _operating_point(request)
    model_xml = _xml(point, settings)
    model_digest = "sha256:" + hashlib.sha256(model_xml.encode()).hexdigest()
    first = _simulate(mujoco, np, model_xml, point)
    second = _simulate(mujoco, np, model_xml, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    unsafe = any(
        (
            bool(first["warnings"]),
            first["maximum_sag_m"] > point["maximum_sag_m"],
            first["maximum_edge_strain"] > point["maximum_edge_strain"],
            first["penetration_m"] > point["maximum_penetration_m"],
        )
    )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "state_trajectory": first["maximum_sag_m"],
        "topology_contact": (
            "ground_contact" if first["maximum_contact_count"] else "no_external_contact"
        ),
        "force": point["total_mass_kg"] * abs(point["gravity_m_s2"][2]),
        "task_outcome": "cloth_envelope_exceeded" if unsafe else "within_cloth_envelope",
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
        "elastic_formulation": "stretch_only",
        "self_collision": "none",
        "vertex_count": first["vertex_count"],
        "edge_count": first["edge_count"],
        "element_count": first["element_count"],
        "step_count": point["step_count"],
        "sample_count": first["sample_count"],
        "maximum_sag_m": first["maximum_sag_m"],
        "maximum_edge_strain": first["maximum_edge_strain"],
        "maximum_contact_count": first["maximum_contact_count"],
        "penetration_m": first["penetration_m"],
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
            failure_codes=["mujoco_cloth_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("mujoco_cloth_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("mujoco_cloth_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a MuJoCo flex-cloth development measurement case"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_mujoco_flex_cloth_request(_load_object(args.request))
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
    "run_mujoco_flex_cloth_request",
]
