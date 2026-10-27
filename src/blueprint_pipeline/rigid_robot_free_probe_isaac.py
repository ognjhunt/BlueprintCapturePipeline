"""Record the rigid robot-free probe in Isaac Sim, on a GPU host; never score.

Beta journey step: L5 team policy evaluation. The scene is the configured
revision's registered support (its published box, a static collider) and the
qualified replacement asset at the task's reset pose, nothing else: no robot,
no other scene collisions. Physics runs at the 120 Hz robot teams are scored
at, with the same solver settings as the native import qualification
(``task_evaluation_scene_configuration_native_import_driver``), whose stage
setup and live pose readback this reuses. The probe's proof boundary says so:
``scene_context`` is ``registered_support_box_and_replacement_only`` and
``scene_collisions_modeled`` is false, as the articulated probe's ground
plane context is.

``prepare_isaac_probe`` is hermetic: it fetches and verifies the revision's
documents, adapts the task exactly as an evaluation would, binds the asset's
bytes to the revision's native import qualification, and returns the plan.
``run_isaac_probe`` needs Isaac. Its trace is retained beside the source
launch as ``rigid_robot_free_probe_trace.v1.json`` and scored by
``robot_free_readiness_retention``. Launching it on a rented GPU is a paid
path and is not wired here.
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
    """Everything the GPU run needs, bound to the revision before any simulator starts."""

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
    spec = task["task_spec"]
    transform = spec["interaction_affordance"]["asset_root_from_scoring_frame"]
    if [float(v) for v in transform["orientation_xyzw"]] != [0.0, 0.0, 0.0, 1.0]:
        raise RigidRobotFreeProbeIsaacError("rigid_probe_scoring_frame_rotation_unsupported")
    support = json.loads(documents[SUPPORT_PLANE_CONTRACT_PATH].read_text())
    destination = json.loads(documents[DEFINITION_CONTRACT_PATH].read_text()).get("destination_support_plane")
    bounds = []
    for surface in [support] + ([destination] if destination is not None else []):
        lower, upper = [float(v) for v in surface["bounds_min_xyz_m"]], [float(v) for v in surface["bounds_max_xyz_m"]]
        if any(a >= b for a, b in zip(lower, upper, strict=True)):
            raise RigidRobotFreeProbeIsaacError("rigid_probe_support_bounds_invalid")
        bounds.append({"minimum": lower, "maximum": upper})
    pose = task["task_object_pose_world"]
    return {
        "plan": physics_plan(probe_context(spec)), "asset_path": str(asset),
        "support_bounds_world_m": bounds[0],
        **({"destination_support_bounds_world_m": bounds[1]} if destination is not None else {}),
        "root_pose_world": [*map(float, pose["position_world_m"]), *map(float, pose["orientation_xyzw"])],
        "scoring_offset_m": [float(v) for v in transform["position_m"]],
        "binding": {"configured_scene_revision_digest": revision["revision_digest"],
                    "task_spec_digest": canonical_digest(spec),
                    "replacement_asset_digest": task["replacement_asset_digest"],
                    "scene_context": SCENE_CONTEXT, "scene_collisions_modeled": False},
    }


class IsaacRigidProbeScene:
    """The recorder seam over one live Isaac stage: pose setting, stepping, readback."""

    def __init__(self, *, simulation: Any, omni_physx: Any, body: Any, body_path: str, offset: Sequence[float],
                 contact: dict[str, bool]):
        self._simulation, self._physx, self._body, self._path = simulation, omni_physx, body, body_path
        self._offset, self._contact = list(offset), contact

    def set_scoring_pose(self, pose: Sequence[float]) -> None:
        import numpy as np  # type: ignore

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
        return {"task_scoring_pose_world": [*(position[i] + shift[i] for i in range(3)), *rotation],
                "support_contact_active": bool(self._contact["touching"]), "scene_collision_force_n": 0.0}


def _build_stage(prepared: Mapping[str, Any]) -> dict[str, Any]:
    import omni.physx as omni_physx  # type: ignore
    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics  # type: ignore

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
    stage.DefinePrim(_ROOT_PATH, "Xform").GetReferences().AddReference(prepared["asset_path"], "/Asset")
    stage.Load()
    bodies = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise RuntimeError("rigid_probe_replacement_structure_invalid")
    world = UsdGeom.Xformable(stage.GetPrimAtPath(bodies[0])).ComputeLocalToWorldTransform(0)
    if any(abs(float(world.ExtractTranslation()[i]) - root[i]) > 1e-6 for i in range(3)):
        # The scoring offset is authored from the asset root; a nested body would shift it.
        raise RuntimeError("rigid_probe_body_not_at_asset_root")
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(bodies[0])).CreateThresholdAttr().Set(0.0)
    contact = {"found": False, "lost": False, "touching": False}

    def _on_contact(headers: Any, _data: Any) -> None:
        for header in headers:
            paths = {str(PhysicsSchemaTools.intToSdfPath(int(getattr(header, name, 0))))
                     for name in ("actor0", "actor1", "collider0", "collider1")}
            if any(p == bodies[0] or p.startswith(bodies[0] + "/") for p in paths) and not paths.isdisjoint(supports):
                # ContactEventType: FOUND and PERSIST mean touching, LOST means apart.
                contact["lost" if str(getattr(header, "type", "")).endswith("LOST") else "found"] = True

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
            "contact": contact, "subscription": subscription}


def run_isaac_probe(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Start headless Isaac, record the plan, shut down; the trace records any failure."""

    from .measurement_isaac_physx_rigid_adapter import _bind_isaac_runtime_environment, _import_simulation_app

    _bind_isaac_runtime_environment()
    app = _import_simulation_app()({"headless": True, "fast_shutdown": True})
    try:
        live: dict[str, Any] = {}
        try:
            live = _build_stage(prepared)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed probe, never raised
            scene: Any = _FailedScene(f"{type(exc).__name__}:{str(exc)[:200]}")
        else:
            scene = IsaacRigidProbeScene(simulation=live["simulation"], omni_physx=live["omni_physx"],
                                         body=live["body"], body_path=live["body_path"],
                                         offset=prepared["scoring_offset_m"], contact=live["contact"])
        trace = record_rigid_robot_free_probe(scene=scene, plan=prepared["plan"], binding=prepared["binding"])
        if live:
            live["simulation"].stop()
        return trace
    finally:
        app.close()


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
