"""Qualify articulated task colliders for native GPU rigid-body simulation.

PhysX cannot use an implicit triangle mesh below a dynamic rigid body and may
silently cook very oblong convex hulls on the CPU.  This module detects both
conditions before paid launch and can emit a new, digest-bound candidate that
uses explicit supported approximations.  It never edits the source asset.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_gpu_collision_qualification.v1"
DEFAULT_MAXIMUM_CONVEX_HULL_ASPECT_RATIO = 100.0
SUPPORTED_DYNAMIC_MESH_APPROXIMATIONS = frozenset(
    {"convexHull", "convexDecomposition", "boundingCube"}
)


class NativeTaskGpuCollisionQualificationError(ValueError):
    """Stable local qualification or authoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stage(path: Path) -> Any:
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(path))
    except (ImportError, RuntimeError) as exc:
        raise NativeTaskGpuCollisionQualificationError(
            ["native_task_gpu_collision_usd_unreadable"]
        ) from exc
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise NativeTaskGpuCollisionQualificationError(
            ["native_task_gpu_collision_usd_unreadable"]
        )
    return stage


def audit_native_task_gpu_collisions(
    usd_path: str | Path,
    *,
    maximum_convex_hull_aspect_ratio: float = (
        DEFAULT_MAXIMUM_CONVEX_HULL_ASPECT_RATIO
    ),
) -> dict[str, Any]:
    """Return exhaustive dynamic-mesh findings without changing the USD."""

    path = Path(usd_path).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or not math.isfinite(maximum_convex_hull_aspect_ratio)
        or maximum_convex_hull_aspect_ratio <= 1.0
    ):
        raise NativeTaskGpuCollisionQualificationError(
            ["native_task_gpu_collision_input_invalid"]
        )
    stage = _stage(path)
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):
            continue
        body = prim
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body.IsValid():
            continue
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        approximation = (
            str(mesh_collision.GetApproximationAttr().Get() or "")
            if mesh_collision
            else ""
        )
        local_range = cache.ComputeLocalBound(prim).ComputeAlignedRange()
        low = local_range.GetMin()
        high = local_range.GetMax()
        dimensions = [max(float(high[i] - low[i]), 0.0) for i in range(3)]
        positive_dimensions = [value for value in dimensions if value > 1e-9]
        aspect_ratio = (
            max(positive_dimensions) / min(positive_dimensions)
            if len(positive_dimensions) == 3
            else math.inf
        )
        prim_path = str(prim.GetPath())
        row_blockers: list[str] = []
        if approximation not in SUPPORTED_DYNAMIC_MESH_APPROXIMATIONS:
            row_blockers.append(
                f"native_task_dynamic_mesh_approximation_unsupported:{prim_path}"
            )
        elif (
            approximation == "convexHull"
            and aspect_ratio > maximum_convex_hull_aspect_ratio
        ):
            row_blockers.append(
                f"native_task_dynamic_convex_hull_gpu_oblong:{prim_path}"
            )
        blockers.extend(row_blockers)
        rows.append(
            {
                "prim_path": prim_path,
                "rigid_body_prim_path": str(body.GetPath()),
                "approximation": approximation or None,
                "local_dimensions_m": dimensions,
                "aspect_ratio": aspect_ratio,
                "blockers": row_blockers,
            }
        )
    if not rows:
        blockers.append("native_task_dynamic_mesh_colliders_missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified" if not blockers else "blocked",
        "usd_path": str(path),
        "usd_sha256": _sha256(path),
        "maximum_convex_hull_aspect_ratio": maximum_convex_hull_aspect_ratio,
        "dynamic_mesh_colliders": rows,
        "blockers": sorted(set(blockers)),
        "runtime_gpu_cooking_readback_still_required": True,
        "simulator_execution_is_not_physical_truth": True,
    }


def author_native_task_gpu_qualified_collisions(
    *,
    source_usd_path: str | Path,
    destination_usd_path: str | Path,
    receipt_path: str | Path,
    maximum_convex_hull_aspect_ratio: float = (
        DEFAULT_MAXIMUM_CONVEX_HULL_ASPECT_RATIO
    ),
) -> dict[str, Any]:
    """Create a new explicit-approximation candidate and bind every mutation."""

    source = Path(source_usd_path).expanduser().resolve()
    destination = Path(destination_usd_path).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    if source == destination or destination.is_symlink() or receipt.is_symlink():
        raise NativeTaskGpuCollisionQualificationError(
            ["native_task_gpu_collision_destination_invalid"]
        )
    before = audit_native_task_gpu_collisions(
        source,
        maximum_convex_hull_aspect_ratio=maximum_convex_hull_aspect_ratio,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    stage = _stage(destination)
    from pxr import UsdPhysics

    changed: list[dict[str, str]] = []
    rows = {
        row["prim_path"]: row for row in before["dynamic_mesh_colliders"]
    }
    for prim_path, row in rows.items():
        prim = stage.GetPrimAtPath(prim_path)
        old = str(row["approximation"] or "")
        new = old
        if old not in SUPPORTED_DYNAMIC_MESH_APPROXIMATIONS:
            new = "convexHull"
        if (
            new == "convexHull"
            and float(row["aspect_ratio"])
            > maximum_convex_hull_aspect_ratio
        ):
            new = "boundingCube"
        if new != old:
            UsdPhysics.MeshCollisionAPI.Apply(
                prim
            ).CreateApproximationAttr().Set(new)
            changed.append(
                {"prim_path": prim_path, "old_approximation": old, "new_approximation": new}
            )
    stage.GetRootLayer().Save()
    after = audit_native_task_gpu_collisions(
        destination,
        maximum_convex_hull_aspect_ratio=maximum_convex_hull_aspect_ratio,
    )
    if after["status"] != "qualified":
        destination.unlink(missing_ok=True)
        raise NativeTaskGpuCollisionQualificationError(after["blockers"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "authored_and_locally_qualified",
        "source_usd_path": str(source),
        "source_usd_sha256": before["usd_sha256"],
        "destination_usd_path": str(destination),
        "destination_usd_sha256": after["usd_sha256"],
        "maximum_convex_hull_aspect_ratio": maximum_convex_hull_aspect_ratio,
        "changes": changed,
        "before_blockers": before["blockers"],
        "after_audit": after,
        "source_bytes_mutated": False,
        "native_gpu_cooking_readback_still_required": True,
        "physical_equivalence_claimed": False,
        "receipt_path": str(receipt),
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    write_json(receipt, result)
    return json.loads(json.dumps(result))


__all__ = [
    "DEFAULT_MAXIMUM_CONVEX_HULL_ASPECT_RATIO",
    "NativeTaskGpuCollisionQualificationError",
    "SCHEMA_VERSION",
    "audit_native_task_gpu_collisions",
    "author_native_task_gpu_qualified_collisions",
]
