"""Native Isaac driver for one Website scene-configuration component stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .measurement_isaac_physx_rigid_adapter import (
    ISAAC_VERSION,
    _bind_isaac_runtime_environment,
    _import_simulation_app,
    _observe_isaac_runtime_identity,
)
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_stage_configuration import (
    native_import_checks_valid,
)


ADAPTER_ID = "simready_native_import_qualification"
RUNTIME_RESULT_SCHEMA_VERSION = "task_evaluation_replacement_native_import_result.v1"
_STAGE_INPUT_SCHEMA = "task_evaluation_scene_configuration_stage_production_input.v1"
_COMPONENT_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
_STAGE_INPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
_DEPENDENCIES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
_OUTPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
NativeObservationConsumer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
NativeRunner = Callable[..., Mapping[str, Any]]


class TaskEvaluationSceneConfigurationNativeImportDriverError(RuntimeError):
    """The exact asset could not be qualified in the native runtime."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(code) from exc
    if path.is_symlink():
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(code)
    return value


def _required_path(environment: Mapping[str, str], name: str) -> Path:
    unresolved = str(environment.get(name) or "").strip()
    if not unresolved:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            f"scene_configuration_native_import_environment_missing:{name}"
        )
    return Path(unresolved).expanduser().resolve()


def _artifact(dependencies: Any, *, role: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(dependencies, list):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_dependencies_invalid"
        )
    matches = [
        row
        for result in dependencies
        if isinstance(result, Mapping)
        for row in result.get("output_artifacts") or []
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            f"scene_configuration_native_import_dependency_missing:{role}"
        )
    record = dict(matches[0])
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("digest")
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            f"scene_configuration_native_import_dependency_invalid:{role}"
        )
    return record, path


def _angle_between_xyzw(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _live_pose(omni_physx: Any, prim_path: str) -> tuple[list[float], list[float]]:
    state = omni_physx.get_physx_interface().get_rigidbody_transformation(prim_path)
    if not hasattr(state, "get") or state.get("ret_val") is not True:
        raise RuntimeError("scene_configuration_native_import_live_pose_unavailable")
    position = [float(state["position"][index]) for index in range(3)]
    rotation = [float(state["rotation"][index]) for index in range(4)]
    if not all(math.isfinite(value) for value in position + rotation):
        raise RuntimeError("scene_configuration_native_import_live_pose_nonfinite")
    return position, rotation


def _subscribe_body_contact_reports(
    *,
    omni_physx: Any,
    physics_schema_tools: Any,
    body_path: str,
    event_count: list[int],
) -> Any:
    """Count body contacts through PhysX's supported simulation callback."""

    def _on_contact_report(contact_headers: Any, _contact_data: Any) -> None:
        try:
            for header in contact_headers:
                paths: list[str] = []
                for name in ("actor0", "actor1", "collider0", "collider1"):
                    encoded = getattr(header, name, 0)
                    try:
                        paths.append(str(physics_schema_tools.intToSdfPath(int(encoded))))
                    except (TypeError, ValueError):
                        paths.append(str(encoded))
                if body_path in paths or any(path.startswith(body_path + "/") for path in paths):
                    event_count[0] += 1
        except Exception:  # noqa: BLE001 - absence of proof fails qualification
            return

    interface = omni_physx.get_physx_simulation_interface()
    return interface.subscribe_contact_report_events(_on_contact_report)


def _one_native_settle(
    *,
    asset_path: Path,
    duration_seconds: float,
    timestep_seconds: float,
) -> dict[str, Any]:
    import omni.physx as omni_physx  # type: ignore
    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore
    from pxr import (  # type: ignore
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
    )

    clear_instance = getattr(SimulationContext, "clear_instance", None)
    if callable(clear_instance):
        clear_instance()
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("scene_configuration_native_import_stage_creation_failed")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateEnableEnhancedDeterminismAttr().Set(True)
    scene_api.CreateEnableGPUDynamicsAttr().Set(False)
    scene_api.CreateBroadphaseTypeAttr().Set("SAP")
    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
    ground.CreateSizeAttr(2.0)
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    ground_xform.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.025))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    placement = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Placement"))
    replacement = stage.DefinePrim("/World/Placement/Replacement", "Xform")
    replacement.GetReferences().AddReference(str(asset_path), "/Asset")
    stage.Load()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = cache.ComputeWorldBound(replacement).ComputeAlignedRange()
    lower = bounds.GetMin()
    if not all(math.isfinite(float(lower[index])) for index in range(3)):
        raise RuntimeError("scene_configuration_native_import_bounds_invalid")
    placement.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.002 - float(lower[2])))
    rigid_paths = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    collision_paths = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(rigid_paths) != 1 or not collision_paths:
        raise RuntimeError("scene_configuration_native_import_structure_invalid")
    body_path = rigid_paths[0]
    body = stage.GetPrimAtPath(body_path)
    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(body)
    contact_api.CreateThresholdAttr().Set(0.0)
    contact_event_count = [0]
    contact_subscription = _subscribe_body_contact_reports(
        omni_physx=omni_physx,
        physics_schema_tools=PhysicsSchemaTools,
        body_path=body_path,
        event_count=contact_event_count,
    )
    simulation = SimulationContext(
        physics_dt=timestep_seconds,
        rendering_dt=timestep_seconds,
        stage_units_in_meters=1.0,
    )
    physics_context = simulation.get_physics_context()
    for name, argument in (
        ("set_solver_type", "TGS"),
        ("set_broadphase_type", "SAP"),
        ("enable_gpu_dynamics", False),
        ("enable_enhanced_determinism", True),
    ):
        method = getattr(physics_context, name, None)
        if callable(method):
            method(argument)
    simulation.initialize_physics()
    simulation.play()
    initial_position, initial_rotation = _live_pose(omni_physx, body_path)
    trace: list[list[float]] = []
    step_count = int(math.ceil(duration_seconds / timestep_seconds))
    for _step in range(step_count):
        try:
            simulation.step(render=False)
        except TypeError:
            simulation.step()
        position, _rotation = _live_pose(omni_physx, body_path)
        trace.append(position)
    final_position, final_rotation = _live_pose(omni_physx, body_path)
    simulation.stop()
    del contact_subscription
    if callable(clear_instance):
        clear_instance()
    translation = math.dist(initial_position, final_position)
    rotation = _angle_between_xyzw(initial_rotation, final_rotation)
    state = {
        "position_m": [round(value, 7) for value in final_position],
        "orientation_xyzw": [round(value, 7) for value in final_rotation],
    }
    return {
        "asset_imported": True,
        "rigid_body_paths": rigid_paths,
        "collision_paths": collision_paths,
        "support_contact_observed": contact_event_count[0] > 0,
        "contact_report_event_count": contact_event_count[0],
        "settle_translation_m": translation,
        "settle_rotation_rad": rotation,
        "step_count": step_count,
        "final_state": state,
        "final_state_digest": canonical_digest(state),
        "position_trace_digest": canonical_digest(
            {"position_trace_m": [[round(v, 7) for v in row] for row in trace]}
        ),
    }


def _run_native_import(
    *,
    asset_path: Path,
    required_checks: Mapping[str, Any],
    observation_consumer: NativeObservationConsumer,
) -> dict[str, Any]:
    _bind_isaac_runtime_environment()
    SimulationApp = _import_simulation_app()
    app = SimulationApp({"headless": True, "fast_shutdown": True})
    try:
        runtime_identity = _observe_isaac_runtime_identity(app)
        repeats = [
            _one_native_settle(
                asset_path=asset_path,
                duration_seconds=float(required_checks["gravity_settle_seconds"]),
                timestep_seconds=1.0 / 60.0,
            )
            for _ in range(int(required_checks["state_digest_repeat_count"]))
        ]
        return dict(
            observation_consumer({"runtime_identity": runtime_identity, "repeats": repeats})
        )
    finally:
        app.close()


def execute_native_import_component(
    *,
    environment: Mapping[str, str] | None = None,
    native_runner: NativeRunner = _run_native_import,
) -> dict[str, Any]:
    """Execute and seal three deterministic native import/reset observations."""

    values = dict(os.environ if environment is None else environment)
    stage_input_path = _required_path(values, _STAGE_INPUT_ENV)
    dependencies_path = _required_path(values, _DEPENDENCIES_ENV)
    output_root = _required_path(values, _OUTPUT_ENV)
    component_result_path = _required_path(values, _COMPONENT_RESULT_ENV)
    production_input = _read(
        stage_input_path, code="scene_configuration_native_import_input_invalid"
    )
    dependencies = _read(
        dependencies_path,
        code="scene_configuration_native_import_dependencies_invalid",
    )
    stage = production_input.get("stage") if isinstance(production_input, Mapping) else None
    configuration = (
        production_input.get("configuration") if isinstance(production_input, Mapping) else None
    )
    checks = configuration.get("required_checks") if isinstance(configuration, Mapping) else None
    if (
        not isinstance(production_input, Mapping)
        or production_input.get("schema_version") != _STAGE_INPUT_SCHEMA
        or not isinstance(stage, Mapping)
        or stage.get("adapter", {}).get("id") != ADAPTER_ID
        or not isinstance(configuration, Mapping)
        or configuration.get("schema_version")
        != "replacement_native_import_qualification_configuration.v1"
        or not native_import_checks_valid(checks)
        or component_result_path.exists()
        or component_result_path.parent != output_root
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_input_invalid"
        )
    asset_record, asset_path = _artifact(
        dependencies, role="statically_qualified_replacement_asset"
    )
    static_record, _static_path = _artifact(dependencies, role="static_qualification_receipt")

    def _seal_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        observed = dict(observation)
        runtime_identity = observed.get("runtime_identity")
        if (
            not isinstance(runtime_identity, Mapping)
            or runtime_identity.get("engine_version") != ISAAC_VERSION
        ):
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_runtime_identity_invalid"
            )
        repeats = observed.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != 3:
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_execution_invalid"
            )
        state_digests = [str(row.get("final_state_digest") or "") for row in repeats]
        maximum_translation = max(
            float(row.get("settle_translation_m", math.inf)) for row in repeats
        )
        maximum_rotation = max(float(row.get("settle_rotation_rad", math.inf)) for row in repeats)
        qualified = (
            len(set(state_digests)) == 1
            and all(row.get("asset_imported") is True for row in repeats)
            and all(row.get("support_contact_observed") is True for row in repeats)
            and all(len(row.get("rigid_body_paths") or []) == 1 for row in repeats)
            and all(bool(row.get("collision_paths")) for row in repeats)
            and maximum_translation <= float(checks["maximum_settle_translation_m"])
            and maximum_rotation <= float(checks["maximum_settle_rotation_rad"])
        )
        if not qualified:
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_qualification_failed"
            )
        runtime_result: dict[str, Any] = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "qualified",
            "replacement_identity": configuration["replacement_identity"],
            "asset_digest": asset_record["digest"],
            "static_qualification_digest": static_record["digest"],
            "native_isaac_executed": True,
            "native_simulator_import_qualified": True,
            "support_contact_observed": True,
            "deterministic_reset_state_digest_repeat_count": 3,
            "deterministic_reset_state_digest": state_digests[0],
            "maximum_observed_settle_translation_m": maximum_translation,
            "maximum_observed_settle_rotation_rad": maximum_rotation,
            "runtime_identity": dict(runtime_identity),
            "repeats": repeats,
            "physical_equivalence_claimed": False,
            "evaluation_episode_executed": False,
            "blockers": [],
            "result_digest": "",
        }
        runtime_result["result_digest"] = canonical_digest(
            runtime_result, digest_field="result_digest"
        )
        artifact_path = output_root / RUNTIME_RESULT_SCHEMA_VERSION
        artifact_path.write_text(canonical_json(runtime_result) + "\n", encoding="utf-8")
        artifact = {
            "role": "native_import_runtime_result",
            "path": str(artifact_path),
            "digest": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        }
        component: dict[str, Any] = {
            "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "adapter_id": ADAPTER_ID,
            "stage_id": stage["stage_id"],
            "provider_mutations_performed": 0,
            "nested_paid_execution_requested": False,
            "artifacts": [artifact],
            "result_digest": "",
        }
        component["result_digest"] = canonical_digest(component, digest_field="result_digest")
        component_result_path.write_text(canonical_json(component) + "\n", encoding="utf-8")
        return component

    return dict(
        native_runner(
            asset_path=asset_path,
            required_checks=checks,
            observation_consumer=_seal_observation,
        )
    )


def main() -> int:
    result = execute_native_import_component()
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ADAPTER_ID",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationNativeImportDriverError",
    "execute_native_import_component",
    "main",
]
