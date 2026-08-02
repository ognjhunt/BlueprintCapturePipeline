"""Isaac Sim 6.0.1 CPU PhysX rigid-contact development worker.

The adapter executes the method-neutral sphere/box drop corpus on a fresh USD
stage using PhysX TGS.  Isaac is intentionally imported only after
``SimulationApp`` starts.  The request, worker source, engine version, solver
settings, live body trace, contact report, and exact replay result are bound in
the normal development execution receipt.  This is not held-out qualification,
R7 admission, physical evidence, or policy-ranking evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-isaac-physx-rigid-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "isaac_physx_tgs_rigid_drop.v1"
ISAAC_VERSION = "6.0.1"
WORKER_SCRIPT = Path(__file__).parents[2] / "scripts/measurement_isaac_physx_rigid_worker.py"


def implementation_digest() -> str:
    hasher = hashlib.sha256()
    for label, path in (("adapter", Path(__file__)), ("worker", WORKER_SCRIPT)):
        hasher.update(label.encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def _import_simulation_app() -> Any:
    """Resolve a callable Isaac launcher across supported packaging layouts."""

    for module_name in (
        "isaacsim.simulation_app",
        "isaacsim",
        "omni.isaac.kit",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - try the next supported packaging layout
            continue
        simulation_app = getattr(module, "SimulationApp", None)
        if callable(simulation_app):
            return simulation_app
    raise ImportError("isaac_physx_rigid_simulation_app_not_callable")


def _observe_isaac_runtime_identity(
    simulation_app: Any,
    *,
    version_getter: Callable[[], Any] | None = None,
) -> dict[str, str]:
    """Read Isaac identity from the running app rather than pip metadata.

    NVIDIA's container distribution does not necessarily install a normal
    ``isaacsim`` dist-info record.  Isaac's supported version API reads the
    application's own VERSION file, while the live Kit application supplies
    useful secondary app/build observations.  The VERSION-file value is the
    authoritative engine version used by this adapter's fail-closed gate.
    """

    package_version = "unavailable"
    try:
        package_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        pass

    if version_getter is None:
        from isaacsim.core.version import get_version  # type: ignore

        version_getter = get_version
    raw_version = version_getter()
    if not isinstance(raw_version, (tuple, list)) or len(raw_version) < 1:
        raise RuntimeError("isaac_physx_rigid_runtime_version_observation_invalid")
    engine_version = str(raw_version[0]).strip()
    if not engine_version:
        raise RuntimeError("isaac_physx_rigid_runtime_version_observation_invalid")

    app = getattr(simulation_app, "app", None)
    app_version = "unavailable"
    build_version = "unavailable"
    if app is not None:
        get_app_version = getattr(app, "get_app_version", None)
        if callable(get_app_version):
            app_version = str(get_app_version()).strip() or "unavailable"
        get_build_version = getattr(app, "get_build_version", None)
        if callable(get_build_version):
            build_version = str(get_build_version()).strip() or "unavailable"
    return {
        "engine_version": engine_version,
        "engine_version_source": "isaacsim.core.version.get_version_app_VERSION_file",
        "observed_package_version": package_version,
        "observed_app_version": app_version,
        "observed_build_version": build_version,
    }


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"isaac_physx_rigid_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"isaac_physx_rigid_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"isaac_physx_rigid_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"isaac_physx_rigid_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"isaac_physx_rigid_{name}_invalid")
    return result


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_protocol_invalid")
    if point.get("protocol_family") != "rigid_body_drop":
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_protocol_family_invalid")
    shape = str(point.get("body_shape", "")).strip()
    if shape not in {"sphere", "box"}:
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_body_shape_invalid")
    if shape == "sphere":
        size = [_number(point.get("radius_m"), name="radius_m", minimum=1e-4)]
        resting_height = size[0]
    else:
        raw_size = point.get("half_size_m")
        if not isinstance(raw_size, list) or len(raw_size) != 3:
            raise MeasurementAdapterExecutionError("isaac_physx_rigid_half_size_invalid")
        size = [_number(item, name="half_size_m", minimum=1e-4) for item in raw_size]
        resting_height = size[2]
    friction = point.get("friction")
    if not isinstance(friction, list) or len(friction) != 3:
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_friction_invalid")
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
        "friction": [_number(item, name="friction", minimum=0.0) for item in friction],
        "penetration_unsafe_threshold_m": _number(
            point.get("penetration_unsafe_threshold_m", 0.03),
            name="penetration_unsafe_threshold_m",
            minimum=0,
        ),
    }


def _live_position(omni_physx: Any, prim_path: str) -> list[float]:
    state = omni_physx.get_physx_interface().get_rigidbody_transformation(prim_path)
    if not hasattr(state, "get") or state.get("ret_val") is not True:
        raise RuntimeError("isaac_physx_live_rigid_body_pose_unavailable")
    position = state.get("position")
    values = [float(position[index]) for index in range(3)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("isaac_physx_live_rigid_body_position_nonfinite")
    return values


def _contact_count(omni_physx: Any, physics_schema_tools: Any, body_path: str) -> int:
    report = omni_physx.get_physx_simulation_interface().get_contact_report()
    headers = report[0] if report else []
    count = 0
    for header in headers:
        paths: list[str] = []
        for name in ("actor0", "actor1", "collider0", "collider1"):
            value = getattr(header, name, "")
            try:
                paths.append(str(physics_schema_tools.intToSdfPath(int(value))))
            except (TypeError, ValueError):
                paths.append(str(value))
        if body_path in paths or any(path.startswith(body_path + "/") for path in paths):
            count += 1
    return count


def _bind_material(
    stage: Any,
    prim: Any,
    *,
    friction: list[float],
    sdf: Any,
    usd_physics: Any,
    usd_shade: Any,
) -> None:
    material = usd_shade.Material.Define(stage, sdf.Path("/World/PhysicsMaterial"))
    api = usd_physics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(friction[0])
    api.CreateDynamicFrictionAttr().Set(friction[0])
    api.CreateRestitutionAttr().Set(0.0)
    usd_shade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        usd_shade.Tokens.weakerThanDescendants,
        "physics",
    )


def _simulate(point: Mapping[str, Any], solver: Mapping[str, Any]) -> dict[str, Any]:
    import omni.physx as omni_physx  # type: ignore
    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # type: ignore

    clear_instance = getattr(SimulationContext, "clear_instance", None)
    if callable(clear_instance):
        clear_instance()
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("isaac_physx_stage_creation_failed")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(abs(point["gravity_m_s2"]))
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateEnableEnhancedDeterminismAttr().Set(True)
    scene_api.CreateEnableGPUDynamicsAttr().Set(False)
    scene_api.CreateBroadphaseTypeAttr().Set("SAP")

    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
    ground.CreateSizeAttr(2.0)
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    ground_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.025))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    body_path = "/World/TestBody"
    if point["shape"] == "sphere":
        body = UsdGeom.Sphere.Define(stage, Sdf.Path(body_path))
        body.CreateRadiusAttr(point["size"][0])
        body_prim = body.GetPrim()
    else:
        body = UsdGeom.Cube.Define(stage, Sdf.Path(body_path))
        body.CreateSizeAttr(2.0)
        body_xform = UsdGeom.Xformable(body.GetPrim())
        body_xform.AddScaleOp().Set(Gf.Vec3f(*point["size"]))
        body_prim = body.GetPrim()
    UsdGeom.Xformable(body_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, point["initial_height_m"]))
    UsdPhysics.CollisionAPI.Apply(body_prim)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr().Set(point["mass_kg"])
    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
    contact_api.CreateThresholdAttr().Set(0.0)
    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
    rigid_api.CreateSolverPositionIterationCountAttr().Set(solver["position_iterations"])
    rigid_api.CreateSolverVelocityIterationCountAttr().Set(solver["velocity_iterations"])
    _bind_material(
        stage,
        body_prim,
        friction=point["friction"],
        sdf=Sdf,
        usd_physics=UsdPhysics,
        usd_shade=UsdShade,
    )
    _bind_material(
        stage,
        ground.GetPrim(),
        friction=point["friction"],
        sdf=Sdf,
        usd_physics=UsdPhysics,
        usd_shade=UsdShade,
    )

    context = SimulationContext(
        physics_dt=point["timestep_s"],
        rendering_dt=point["timestep_s"],
        stage_units_in_meters=1.0,
    )
    physics_context = context.get_physics_context()
    for name, argument in (
        ("set_solver_type", "TGS"),
        ("set_broadphase_type", "SAP"),
        ("enable_gpu_dynamics", False),
        ("enable_enhanced_determinism", True),
    ):
        method = getattr(physics_context, name, None)
        if callable(method):
            method(argument)
    context.initialize_physics()
    context.play()
    step_count = int(math.ceil(point["duration_s"] / point["timestep_s"]))
    first_contact_step: int | None = None
    contact_report_event_count = 0
    minimum_center_height = point["initial_height_m"]
    positions: list[list[float]] = []
    for step in range(step_count):
        try:
            context.step(render=False)
        except TypeError:
            context.step()
        position = _live_position(omni_physx, body_path)
        positions.append(position)
        minimum_center_height = min(minimum_center_height, position[2])
        event_count = _contact_count(omni_physx, PhysicsSchemaTools, body_path)
        contact_report_event_count += event_count
        if event_count and first_contact_step is None:
            first_contact_step = step
    final_position = positions[-1]
    context.stop()
    penetration = max(0.0, point["resting_height"] - minimum_center_height)
    trace = {
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "contact_report_event_count": contact_report_event_count,
        "minimum_center_height_m": minimum_center_height,
        "final_position_m": final_position,
        "penetration_m": penetration,
        "position_trace_m": positions,
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    if callable(clear_instance):
        clear_instance()
    return trace


def run_isaac_physx_rigid_measurement_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
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
                failure_codes=[f"isaac_physx_rigid_{code}"],
            )
    if runtime["target_engine_version"] != ISAAC_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["isaac_physx_rigid_target_version_mismatch"],
        )
    if runtime["backend_id"] != "isaac-physx-cpu-tgs-rigid":
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_backend_invalid")
    if runtime["precision"] != "float32":
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_precision_invalid")
    solver = dict(runtime["solver_settings"])
    if set(solver) != {
        "solver_type",
        "broadphase_type",
        "gpu_dynamics",
        "enhanced_determinism",
        "position_iterations",
        "velocity_iterations",
    }:
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_solver_settings_invalid")
    if (
        solver["solver_type"] != "TGS"
        or solver["broadphase_type"] != "SAP"
        or solver["gpu_dynamics"] is not False
        or solver["enhanced_determinism"] is not True
    ):
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_solver_configuration_invalid")
    for key in ("position_iterations", "velocity_iterations"):
        value = solver[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
            raise MeasurementAdapterExecutionError(f"isaac_physx_rigid_{key}_invalid")
    point = _operating_point(request)
    simulation_app: Any | None = None
    try:
        SimulationApp = _import_simulation_app()
        simulation_app = SimulationApp({"headless": True})
        try:
            runtime_identity = _observe_isaac_runtime_identity(simulation_app)
        except Exception as exc:  # noqa: BLE001
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations={
                    **observations,
                    "runtime_identity_error_type": type(exc).__name__,
                    "runtime_identity_error": repr(exc)[:800],
                },
                failure_codes=["isaac_physx_rigid_runtime_identity_unavailable"],
            )
        observations.update(runtime_identity)
        if runtime_identity["engine_version"] != ISAAC_VERSION:
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations=observations,
                failure_codes=["isaac_physx_rigid_runtime_version_mismatch"],
            )
        first = _simulate(point, solver)
        second = _simulate(point, solver)
    except Exception as exc:  # noqa: BLE001
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations={
                **observations,
                "execution_error_type": type(exc).__name__,
                "execution_error": repr(exc)[:800],
            },
            failure_codes=["isaac_physx_rigid_execution_failed"],
        )
    finally:
        if simulation_app is not None:
            simulation_app.close()
    replay_match = first["trace_digest"] == second["trace_digest"]
    contact_observed = first["first_contact_step"] is not None
    available_metrics = {
        "penetration": first["penetration_m"],
        "contact_sequence": "ground_contact" if contact_observed else "no_contact",
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
            "solver_type": "TGS",
            "broadphase_type": "SAP",
            "gpu_dynamics": False,
            "enhanced_determinism": True,
            "renderer_used": False,
            "rtx_sensor_used": False,
            "contact_source": "physx_contact_report",
            "penetration_source": "rest_height_minus_minimum_live_center_height",
            "timestep_s": point["timestep_s"],
            **{
                key: value
                for key, value in first.items()
                if key not in {"trace_digest", "position_trace_m"}
            },
            "trace_digest": first["trace_digest"],
            "repeat_trace_digest": second["trace_digest"],
            "deterministic_replay_match": replay_match,
        }
    )
    failure_codes = []
    if not replay_match:
        failure_codes.append("isaac_physx_rigid_replay_mismatch")
    if not contact_observed:
        failure_codes.append("isaac_physx_rigid_contact_not_observed")
    return build_measurement_adapter_worker_result(
        request,
        status="failed" if failure_codes else "completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=(
            first["penetration_m"] > point["penetration_unsafe_threshold_m"]
            if not failure_codes
            else None
        ),
        runtime_observations=observations,
        failure_codes=failure_codes,
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("isaac_physx_rigid_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an Isaac/PhysX rigid-contact development case"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_isaac_physx_rigid_measurement_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "ISAAC_VERSION",
    "PROTOCOL_ID",
    "WORKER_SCRIPT",
    "implementation_digest",
    "main",
    "run_isaac_physx_rigid_measurement_request",
]
