"""Record the rigid robot-free probe in Isaac Sim, on a GPU host; never score.

Beta journey step: L5 team policy evaluation. The scene contains registered
support boxes, the qualified replacement asset at the task's reset pose, and
any exact qualified supplemental destination asset. There is no robot and no
other scene collision geometry. Physics runs at the 120 Hz robot teams are scored
at, with the same solver settings as the native import qualification
(``task_evaluation_scene_configuration_native_import_driver``), whose stage
setup and live pose readback this reuses. The probe's proof boundary says so:
``scene_context`` identifies the support-box context and any supplemental
destination, and ``scene_collisions_modeled`` stays false, as the articulated
probe's ground-plane context does.

``prepare_isaac_probe`` is hermetic: it fetches and verifies the revision's
documents, adapts the task exactly as an evaluation would, binds the asset's
bytes to the revision's native import qualification, and returns the plan.
``run_isaac_probe`` needs Isaac. Its trace is retained beside the source
launch as ``rigid_robot_free_probe_trace.v1.json`` and scored by
``robot_free_readiness_retention``. Scene construction also records this probe
inside its existing native-import allocation, without a second GPU launch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_plan import PHYSICS_HZ, physics_plan, probe_context
from .rigid_robot_free_probe_recorder import record_rigid_robot_free_probe

SCENE_CONTEXT = "registered_support_box_and_replacement_only"
_SUPPORT_PATH = "/World/RegisteredSupport"
#: A task that ends on another surface (floor to desk) models that one too.
_DESTINATION_SUPPORT_PATH = "/World/RegisteredDestinationSupport"
_ROOT_PATH = "/World/Placement/Replacement"
_DESTINATION_ROOT_PATH = "/World/DestinationPlacement/Destination"


class RigidRobotFreeProbeIsaacError(ValueError):
    """The probe cannot be bound to the revision; nothing was simulated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _rotate(vector: Sequence[float], q: Sequence[float]) -> list[float]:
    x, y, z, w = q
    a, b, c = vector
    tx, ty, tz = 2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a)
    return [a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx]


def prepare_isaac_probe(*, revision: Mapping[str, Any], replacement_asset: str | Path, output_root: str | Path,
                        fetcher: Callable[[Mapping[str, Any]], bytes] | None = None) -> dict[str, Any]:
    """Everything the GPU run needs, bound to the revision before any simulator starts.

    Deprecated for team scenes: this binds the configured revision's task
    specification, so its receipt would refuse every team evaluation with
    ``task_spec_mismatch``. Teams' scenes are probed by the capped lane
    (``rigid_robot_free_probe_runner``), bound to the specification they are
    scored with (``team_robot_free_probe.probe_request``).
    """

    from .task_evaluation_rigid_relocation_native_adapter import (
        DEFINITION_CONTRACT_PATH,
        SUPPORT_PLANE_CONTRACT_PATH,
    )
    from .task_evaluation_scene_rigid_robot_free_readiness import fetch_revision_documents, rigid_task_for_revision

    documents = fetch_revision_documents(revision=revision, output_root=output_root, fetcher=fetcher)
    task = rigid_task_for_revision(revision=revision, documents=documents)
    if task["unsupported_reason"] is not None:
        raise RigidRobotFreeProbeIsaacError("rigid_probe_unsupported:" + task["unsupported_reason"])
    asset = Path(replacement_asset)
    if asset.is_symlink() or not asset.is_file() or task["replacement_asset_digest"] is None \
            or _sha256(asset) != task["replacement_asset_digest"]:
        raise RigidRobotFreeProbeIsaacError("rigid_probe_replacement_asset_not_the_qualified_asset")
    return prepare_probe_scene(task=task, replacement_asset=asset,
        support=json.loads(documents[SUPPORT_PLANE_CONTRACT_PATH].read_text()),
        destination=json.loads(documents[DEFINITION_CONTRACT_PATH].read_text()).get("destination_support_plane"),
        revision_digest=revision["revision_digest"])


def prepare_probe_scene(*, task: Mapping[str, Any], replacement_asset: str | Path,
                        support: Mapping[str, Any], destination: Mapping[str, Any] | None,
                        revision_digest: str | None) -> dict[str, Any]:
    """Shared exact geometry/plan derivation for published and construction inputs."""
    asset = Path(replacement_asset)
    if asset.is_symlink() or not asset.is_file() or task.get("replacement_asset_digest") is None \
            or _sha256(asset) != task["replacement_asset_digest"]:
        raise RigidRobotFreeProbeIsaacError("rigid_probe_replacement_asset_not_the_qualified_asset")
    destination_probe = task.get("destination_probe")
    if destination_probe is not None:
        destination_asset = Path(destination_probe["asset_path"])
        if destination_asset.is_symlink() or not destination_asset.is_file() or (
                _sha256(destination_asset) != destination_probe["asset_digest"]):
            raise RigidRobotFreeProbeIsaacError("rigid_probe_destination_asset_not_the_qualified_asset")
    spec = task["task_spec"]
    transform = spec["interaction_affordance"]["asset_root_from_scoring_frame"]
    if [float(v) for v in transform["orientation_xyzw"]] != [0.0, 0.0, 0.0, 1.0]:
        raise RigidRobotFreeProbeIsaacError("rigid_probe_scoring_frame_rotation_unsupported")
    bounds = []
    for surface in [support] + ([destination] if destination is not None else []):
        lower, upper = [float(v) for v in surface["bounds_min_xyz_m"]], [float(v) for v in surface["bounds_max_xyz_m"]]
        if any(a >= b for a, b in zip(lower, upper, strict=True)):
            raise RigidRobotFreeProbeIsaacError("rigid_probe_support_bounds_invalid")
        bounds.append({"minimum": lower, "maximum": upper})
    pose = task["task_object_pose_world"]
    return {
        "plan": physics_plan(probe_context(spec)), "asset_path": str(asset),
        **({"destination_probe": dict(destination_probe)} if destination_probe is not None else {}),
        "support_bounds_world_m": bounds[0],
        **({"destination_support_bounds_world_m": bounds[1]} if destination is not None else {}),
        "root_pose_world": [*map(float, pose["position_world_m"]), *map(float, pose["orientation_xyzw"])],
        "scoring_offset_m": [float(v) for v in transform["position_m"]],
        "binding": {"configured_scene_revision_digest": revision_digest,
                    "task_spec_digest": canonical_digest(spec),
                    "replacement_asset_digest": task["replacement_asset_digest"],
                    "scene_context": ("registered_support_boxes_and_qualified_destination_and_replacement"
                                      if destination_probe is not None else SCENE_CONTEXT),
                    "scene_collisions_modeled": False,
                    **({"destination_asset_digest": destination_probe["asset_digest"]}
                       if destination_probe is not None else {})},
    }


class IsaacRigidProbeScene:
    """The recorder seam over one live Isaac stage: pose setting, stepping, readback."""

    def __init__(self, *, simulation: Any, omni_physx: Any, body: Any, body_path: str, offset: Sequence[float],
                 contact: dict[str, Any], destination_body_path: str | None = None):
        self._simulation, self._physx, self._body, self._path = simulation, omni_physx, body, body_path
        self._offset, self._contact = list(offset), contact
        self._destination_path = destination_body_path
        self._carrier_used = False

    def set_scoring_pose(self, pose: Sequence[float]) -> None:
        import numpy as np  # type: ignore

        self._carrier_used = True
        q = [float(v) for v in pose[3:]]
        shift = _rotate(self._offset, q)
        self._body.set_world_pose(position=np.asarray([pose[i] - shift[i] for i in range(3)], dtype=np.float64),
                                  orientation=np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64))
        self._body.set_linear_velocity(np.zeros(3))
        self._body.set_angular_velocity(np.zeros(3))

    def step(self) -> None:
        from .task_evaluation_scene_configuration_native_import_driver import _physics_step

        self._contact["found"] = self._contact["lost"] = False
        _physics_step(self._simulation)
        if self._contact["found"]:
            self._contact["touching"] = True
        elif self._contact["lost"]:
            self._contact["touching"] = False

    def read_sample(self) -> dict[str, Any]:
        from .task_evaluation_scene_configuration_native_import_driver import _live_pose

        position, rotation = _live_pose(self._physx, self._path)
        shift = _rotate(self._offset, rotation)
        destination_pose = None
        if self._destination_path is not None:
            destination_position, destination_rotation = _live_pose(self._physx, self._destination_path)
            destination_pose = [*destination_position, *destination_rotation]
        if "source_pairs" in self._contact:
            touching = bool(self._contact["destination_pairs"])
            if self._destination_path is None or not self._carrier_used:
                touching |= bool(self._contact["source_pairs"])
        else:
            touching = bool(self._contact["touching"])
        return {"task_scoring_pose_world": [*(position[i] + shift[i] for i in range(3)), *rotation],
                "support_contact_active": touching, "scene_collision_force_n": 0.0,
                **({"destination_pose_world": destination_pose} if destination_pose is not None else {})}


def destination_probe_support_paths(*, source_root: str, source_paths: Sequence[str]) -> set[str]:
    """Map exact qualified support bodies through USD's default-prim reference."""
    paths = set()
    for path in source_paths:
        if path != source_root and not path.startswith(source_root + "/"):
            raise RuntimeError("rigid_probe_destination_support_body_invalid")
        paths.add(_DESTINATION_ROOT_PATH + path.removeprefix(source_root))
    return paths


def _build_stage(prepared: Mapping[str, Any]) -> dict[str, Any]:
    import omni.physx as omni_physx  # type: ignore
    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore

    clear_instance = getattr(SimulationContext, "clear_instance", None)
    if callable(clear_instance):
        clear_instance()
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
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
    supports = {_SUPPORT_PATH: prepared["support_bounds_world_m"],
                **({_DESTINATION_SUPPORT_PATH: prepared["destination_support_bounds_world_m"]}
                   if "destination_support_bounds_world_m" in prepared else {})}
    destination_probe = prepared.get("destination_probe")
    if destination_probe is not None and destination_probe.get("placement_support_bounds_world_m") is not None:
        supports[_DESTINATION_SUPPORT_PATH] = destination_probe["placement_support_bounds_world_m"]
    for path, bounds in supports.items():
        lower, upper = bounds["minimum"], bounds["maximum"]
        support = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        support.CreateSizeAttr(1.0)
        support_xform = UsdGeom.Xformable(support.GetPrim())
        support_xform.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2.0 for a, b in zip(lower, upper, strict=True)]))
        support_xform.AddScaleOp().Set(Gf.Vec3f(*[b - a for a, b in zip(lower, upper, strict=True)]))
        UsdPhysics.CollisionAPI.Apply(support.GetPrim())
    root = prepared["root_pose_world"]
    placement = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Placement"))
    placement.AddTranslateOp().Set(Gf.Vec3d(*root[:3]))
    placement.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(root[6], Gf.Vec3d(*root[3:6])))
    stage.DefinePrim(_ROOT_PATH, "Xform").GetReferences().AddReference(prepared["asset_path"])
    if destination_probe is not None:
        destination_pose = destination_probe["pose_world"]
        destination_placement = UsdGeom.Xform.Define(stage, Sdf.Path("/World/DestinationPlacement"))
        destination_placement.AddTranslateOp().Set(Gf.Vec3d(*destination_pose[:3]))
        destination_placement.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(destination_pose[6], Gf.Vec3d(*destination_pose[3:6])))
        destination_source = Usd.Stage.Open(destination_probe["asset_path"])
        destination_default = destination_source.GetDefaultPrim() if destination_source is not None else None
        if destination_default is None or not destination_default.IsValid():
            raise RuntimeError("rigid_probe_destination_default_prim_missing")
        destination_source_root = str(destination_default.GetPath())
        stage.DefinePrim(_DESTINATION_ROOT_PATH, "Xform").GetReferences().AddReference(
            destination_probe["asset_path"])
    stage.Load()
    bodies = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
              and (str(prim.GetPath()) == _ROOT_PATH or str(prim.GetPath()).startswith(_ROOT_PATH + "/"))]
    destination_body_path = None
    support_paths = set(supports)
    destination_support_paths: set[str] = set()
    if destination_probe is not None:
        destination_bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (str(prim.GetPath()) == _DESTINATION_ROOT_PATH
                 or str(prim.GetPath()).startswith(_DESTINATION_ROOT_PATH + "/"))]
        if len(destination_bodies) != 1:
            raise RuntimeError("rigid_probe_destination_structure_invalid")
        destination_body_path = str(destination_bodies[0].GetPath())
        destination_world = UsdGeom.Xformable(destination_bodies[0]).ComputeLocalToWorldTransform(0)
        if any(abs(float(destination_world.ExtractTranslation()[i]) - destination_pose[i]) > 1e-6 for i in range(3)):
            raise RuntimeError("rigid_probe_destination_body_not_at_asset_root")
        declared_supports = destination_probe_support_paths(source_root=destination_source_root,
            source_paths=destination_probe["intended_support_prim_paths"])
        if destination_body_path not in declared_supports:
            raise RuntimeError("rigid_probe_destination_support_body_invalid")
        destination_support_paths = declared_supports
    if len(bodies) != 1:
        raise RuntimeError("rigid_probe_replacement_structure_invalid")
    world = UsdGeom.Xformable(stage.GetPrimAtPath(bodies[0])).ComputeLocalToWorldTransform(0)
    if any(abs(float(world.ExtractTranslation()[i]) - root[i]) > 1e-6 for i in range(3)):
        # The scoring offset is authored from the asset root; a nested body would shift it.
        raise RuntimeError("rigid_probe_body_not_at_asset_root")
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(bodies[0])).CreateThresholdAttr().Set(0.0)
    contact = {"found": False, "lost": False, "touching": False,
               "source_pairs": set(), "destination_pairs": set()}

    def _on_contact(headers: Any, _data: Any) -> None:
        for header in headers:
            paths = {str(PhysicsSchemaTools.intToSdfPath(int(getattr(header, name, 0))))
                     for name in ("actor0", "actor1", "collider0", "collider1")}
            if not any(p == bodies[0] or p.startswith(bodies[0] + "/") for p in paths):
                continue
            pair = tuple(sorted(path for path in paths if path))
            for label, candidates in (("source_pairs", support_paths),
                                      ("destination_pairs", destination_support_paths)):
                if any(path == support or path.startswith(support + "/")
                       for path in paths for support in candidates):
                    if str(getattr(header, "type", "")).endswith("LOST"):
                        contact[label].discard(pair)
                    else:
                        contact[label].add(pair)

    subscription = omni_physx.get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
    simulation = SimulationContext(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ,
                                   stage_units_in_meters=1.0)
    physics_context = simulation.get_physics_context()
    for name, argument in (("set_solver_type", "TGS"), ("set_broadphase_type", "SAP"),
                           ("enable_gpu_dynamics", False), ("enable_enhanced_determinism", True)):
        method = getattr(physics_context, name, None)
        if callable(method):
            method(argument)
    simulation.initialize_physics()
    simulation.play()
    body = SingleRigidPrim(prim_path=bodies[0], name="blueprint_rigid_robot_free_probe")
    body.initialize()
    return {"simulation": simulation, "omni_physx": omni_physx, "body": body, "body_path": bodies[0],
            "contact": contact, "subscription": subscription, "destination_body_path": destination_body_path}


def run_isaac_probe(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Start headless Isaac, record the plan, shut down; the trace records any failure."""

    from .measurement_isaac_physx_rigid_adapter import _bind_isaac_runtime_environment, _import_simulation_app

    _bind_isaac_runtime_environment()
    app = _import_simulation_app()({"headless": True, "fast_shutdown": True})
    try:
        return record_in_active_isaac(prepared)
    finally:
        app.close()


def record_in_active_isaac(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Record in construction's already-owned SimulationApp; never allocate."""
    live: dict[str, Any] = {}
    trace: dict[str, Any] | None = None
    try:
        try:
            live = _build_stage(prepared)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed probe, never raised
            scene: Any = _FailedScene(f"{type(exc).__name__}:{str(exc)[:200]}")
        else:
            scene = IsaacRigidProbeScene(simulation=live["simulation"], omni_physx=live["omni_physx"],
                                         body=live["body"], body_path=live["body_path"],
                                         offset=prepared["scoring_offset_m"], contact=live["contact"],
                                         destination_body_path=live.get("destination_body_path"))
        trace = record_rigid_robot_free_probe(scene=scene, plan=prepared["plan"], binding=prepared["binding"])
        return trace
    finally:
        errors = []
        simulation = live.get("simulation")
        simulation_type = None
        try:
            from isaacsim.core.api import SimulationContext  # type: ignore
            simulation_type = SimulationContext
            if simulation is None and callable(getattr(SimulationContext, "instance", None)):
                simulation = SimulationContext.instance()
        except ImportError:
            pass
        try:
            if simulation is not None:
                simulation.stop()
        except Exception as exc:  # noqa: BLE001 - retained alongside failed evidence
            errors.append(type(exc).__name__)
        finally:
            subscription = live.pop("subscription", None)
            unsubscribe = getattr(subscription, "unsubscribe", None)
            try:
                if callable(unsubscribe):
                    unsubscribe()
            except Exception as exc:  # noqa: BLE001 - continue closing the simulator
                errors.append(type(exc).__name__)
            finally:
                del subscription
                clear_instance = getattr(simulation_type, "clear_instance", None)
                try:
                    if callable(clear_instance):
                        clear_instance()
                except Exception as exc:  # noqa: BLE001 - retained cleanup failure
                    errors.append(type(exc).__name__)
        if errors and trace is not None:
            trace.update(status="failed", shutdown_errors=errors)
            trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")


class _FailedScene:
    def __init__(self, reason: str):
        self.reason = reason

    def set_scoring_pose(self, pose: Sequence[float]) -> None:
        raise RuntimeError(self.reason)

    def step(self) -> None:
        raise RuntimeError(self.reason)

    def read_sample(self) -> dict[str, Any]:
        raise RuntimeError(self.reason)


def main(argv: list[str] | None = None) -> int:
    """Record one revision's rigid robot-free probe on this GPU host; writes the raw trace."""

    import argparse

    from .robot_free_readiness_retention import TRACE_FILENAME
    from .task_evaluation_configured_scene_revision import validate_configured_scene_revision

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--revision", required=True, help="configured_scene_revision.v1.json")
    parser.add_argument("--replacement-asset", required=True, help="the qualified replacement USD")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    revision = validate_configured_scene_revision(json.loads(Path(args.revision).read_text()))
    prepared = prepare_isaac_probe(revision=revision, replacement_asset=args.replacement_asset,
                                   output_root=args.output_root)
    trace = run_isaac_probe(prepared)
    path = Path(args.output_root) / TRACE_FILENAME
    text = json.dumps(trace, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise SystemExit("rigid_robot_free_probe_trace_conflict")
    path.write_text(text)
    print(json.dumps({key: trace.get(key) for key in ("status", "error", "trace_digest")}, sort_keys=True))
    return 0 if trace["status"] == "recorded" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IsaacRigidProbeScene",
    "RigidRobotFreeProbeIsaacError",
    "SCENE_CONTEXT",
    "main",
    "prepare_isaac_probe",
    "run_isaac_probe",
]
