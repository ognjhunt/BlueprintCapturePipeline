"""Bind the USD asset frame to an explicit physics root before simulation.

An ArticulationRootAPI on a namespace lets PhysX choose a central link. That
link need not be the frame used by the scene planner. Keep the source asset
unchanged, select the root on the composed stage, and convert poses at the
native boundary. Fixed-base joints retain their authored semantics.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .rigid_frame_transforms import (
    apply_rigid_offset,
    normalize_quaternion_xyzw,
    quaternion_conjugate_xyzw,
    rotate_vector_xyzw,
)

SCHEMA_VERSION = "native_articulation_root_frame.v1"


class NativeArticulationRootFrameError(ValueError):
    """An articulation cannot be bound to one known native root frame."""


def validate_root_frame_binding(binding: Mapping[str, Any]) -> None:
    if (
        binding.get("schema_version") != SCHEMA_VERSION
        or binding.get("binding_digest")
        != canonical_digest(binding, digest_field="binding_digest")
    ):
        raise NativeArticulationRootFrameError("native_articulation_root_binding_invalid")


def derive_root_frame_binding(asset_path: str | Path) -> dict[str, Any]:
    """Resolve a single root from authored APIs/world joints/directed topology."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = Path(asset_path)
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim() if stage else None
    if not root or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise NativeArticulationRootFrameError("native_articulation_root_asset_invalid")
    prims = list(Usd.PrimRange(root))
    root_path = str(root.GetPath())
    bodies = {
        str(prim.GetPath()): prim for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
    }
    api_paths = [str(prim.GetPath()) for prim in prims
                 if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not bodies or len(api_paths) > 1:
        raise NativeArticulationRootFrameError("native_articulation_root_count_invalid")
    edges: list[tuple[str, str]] = []
    world_joints: list[tuple[str, str]] = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if (joint.GetJointEnabledAttr().Get() is False
                or joint.GetExcludeFromArticulationAttr().Get() is True):
            continue
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(a) > 1 or len(b) > 1 or not (a or b):
            raise NativeArticulationRootFrameError("native_articulation_joint_body_invalid")
        left, right = str(a[0]) if a else None, str(b[0]) if b else None
        if any(body is not None and body not in bodies for body in (left, right)):
            raise NativeArticulationRootFrameError("native_articulation_joint_body_invalid")
        if left is None or right is None:
            world_joints.append((str(prim.GetPath()), left or right))
        else:
            edges.append((left, right))
    if len(world_joints) > 1:
        raise NativeArticulationRootFrameError("native_articulation_world_root_ambiguous")
    # One connected tree is required. No USD namespace heuristic chooses a link.
    neighbours: dict[str, set[str]] = {body: set() for body in bodies}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    visited: set[str] = set()
    pending = [next(iter(bodies))]
    while pending:
        body = pending.pop()
        if body not in visited:
            visited.add(body)
            pending.extend(neighbours[body] - visited)
    if visited != set(bodies) or len(edges) != len(bodies) - 1:
        raise NativeArticulationRootFrameError("native_articulation_topology_ambiguous")
    authored_api = api_paths[0] if api_paths else None
    if world_joints:
        api_target, base = world_joints[0]
        mode = "authored_world_joint"
    elif authored_api in bodies:
        base = api_target = authored_api
        mode = "authored_body_root"
    else:
        candidates = set(bodies) - {b for _, b in edges}
        if len(candidates) != 1:
            raise NativeArticulationRootFrameError("native_articulation_directed_root_ambiguous")
        base = api_target = candidates.pop()
        mode = "directed_body_root"
    if authored_api and not (api_target == authored_api
                            or api_target.startswith(authored_api + "/")
                            or mode == "authored_body_root"):
        raise NativeArticulationRootFrameError("native_articulation_root_scope_invalid")
    matrix, reset_stack = UsdGeom.XformCache().ComputeRelativeTransform(bodies[base], root)
    # Root/body scale, reflection and shearing need a separate geometry adapter.
    rows = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    if reset_stack or any(
        not math.isclose(sum(rows[i][k] * rows[j][k] for k in range(3)),
                         float(i == j), abs_tol=1e-6)
        for i in range(3) for j in range(3)
    ) or matrix.GetDeterminant() <= 0:
        raise NativeArticulationRootFrameError("native_articulation_root_transform_not_rigid")
    quat = Gf.Transform(matrix).GetRotation().GetQuat()
    binding = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_root_prim_path": root_path,
        "authored_root_api_prim_paths": api_paths,
        "native_root_body_prim_path": base,
        "runtime_root_api_prim_path": api_target,
        "selection": mode,
        "asset_from_native_root": {
            "position_world_m": list(matrix.ExtractTranslation()),
            "orientation_xyzw": normalize_quaternion_xyzw(
                [*quat.GetImaginary(), quat.GetReal()]
            ),
        },
        "candidate_bytes_modified": False,
    }
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    return binding


def convert_root_pose(
    pose: Mapping[str, Sequence[float]],
    binding: Mapping[str, Any] | None,
    *,
    to_native: bool,
) -> dict[str, list[float]]:
    """Compose world<-asset<-native; invert only the constant rigid offset."""
    if binding is None:
        return {key: list(pose[key]) for key in ("position_world_m", "orientation_xyzw")}
    validate_root_frame_binding(binding)
    offset = binding["asset_from_native_root"]
    position, orientation = offset["position_world_m"], offset["orientation_xyzw"]
    if not to_native:
        orientation = quaternion_conjugate_xyzw(orientation)
        position = rotate_vector_xyzw(orientation, [-value for value in position])
    position, orientation = apply_rigid_offset(
        body_position_world=pose["position_world_m"],
        body_quaternion_world_xyzw=pose["orientation_xyzw"],
        offset_position_body=position,
        offset_quaternion_body_xyzw=orientation,
    )
    return {"position_world_m": position, "orientation_xyzw": orientation}


def rebind_root_frame_after_overlay(binding: Mapping[str, Any], asset_path: str | Path) -> dict[str, Any]:
    """Admit an existing material overlay only when its root/frame is identical."""
    validate_root_frame_binding(binding)
    derived = derive_root_frame_binding(asset_path)
    ignored = {"source_sha256", "binding_digest"}
    if ({key: value for key, value in binding.items() if key not in ignored}
            != {key: value for key, value in derived.items() if key not in ignored}):
        raise NativeArticulationRootFrameError("native_articulation_overlay_frame_changed")
    return derived


def apply_root_api_binding(stage: Any, spawned_root_path: str, binding: Mapping[str, Any]) -> None:
    """Override only the root API in the composed spawn, never the source file."""
    from pxr import UsdPhysics

    validate_root_frame_binding(binding)
    source_root = binding["source_root_prim_path"]

    def spawned(source: str) -> Any:
        prim = stage.GetPrimAtPath(spawned_root_path + source[len(source_root):])
        if not prim:
            raise NativeArticulationRootFrameError("native_articulation_spawn_root_missing")
        return prim

    target = spawned(binding["runtime_root_api_prim_path"])
    for source in binding["authored_root_api_prim_paths"]:
        if source != binding["runtime_root_api_prim_path"]:
            spawned(source).RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(target)


def root_bound_usd_spawner(binding: Mapping[str, Any]) -> Any:
    """Keep namespace spawn and native reset poses consistent, including clones."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import clone

    validate_root_frame_binding(binding)

    @clone
    def spawn(prim_path: str, cfg: Any, translation: Any = None,
              orientation: Any = None, **kwargs: Any) -> Any:
        if derive_root_frame_binding(cfg.usd_path) != binding:
            raise NativeArticulationRootFrameError("native_articulation_spawn_binding_mismatch")
        logical_pose = convert_root_pose(
            {"position_world_m": translation or (0.0, 0.0, 0.0),
             "orientation_xyzw": orientation or (0.0, 0.0, 0.0, 1.0)},
            binding, to_native=False,
        )
        prim = spawn_from_usd(
            prim_path, cfg, translation=tuple(logical_pose["position_world_m"]),
            orientation=tuple(logical_pose["orientation_xyzw"]), **kwargs,
        )
        apply_root_api_binding(prim.GetStage(), str(prim.GetPath()), binding)
        return prim

    return spawn
