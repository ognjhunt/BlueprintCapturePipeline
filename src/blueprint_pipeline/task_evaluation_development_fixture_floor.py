"""Add an explicit floor to the robot-placement screen of a development fixture.

The published scene collider remains untouched. This derived placement input
retains its desk/obstacles and is never promoted as captured-room geometry.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Sequence

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_development_fixture_placement_floor(
    *, source_collision: Path, qualified_asset: Path,
    qualified_asset_digest: str, target_world_m: Sequence[float],
    output_root: Path,
) -> Path:
    """Derive a z=0 base support only when the qualified asset rests at z=0."""
    if _digest(qualified_asset) != qualified_asset_digest:
        raise ValueError("development_fixture_floor_asset_digest_mismatch")
    asset_stage = Usd.Stage.Open(str(qualified_asset))
    if asset_stage is None or not asset_stage.GetDefaultPrim().IsValid():
        raise ValueError("development_fixture_floor_asset_invalid")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    ).ComputeWorldBound(asset_stage.GetDefaultPrim()).ComputeAlignedRange()
    if abs(float(bounds.GetMin()[2])) > 0.005:
        raise ValueError("development_fixture_floor_asset_not_at_zero")
    if len(target_world_m) != 3 or not all(
        isinstance(value, (float, int)) for value in target_world_m
    ):
        raise ValueError("development_fixture_floor_target_invalid")
    stage = Usd.Stage.Open(str(source_collision))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise ValueError("development_fixture_floor_collision_invalid")
    floor_path = str(stage.GetDefaultPrim().GetPath()) + "/development_fixture_floor"
    if stage.GetPrimAtPath(floor_path).IsValid():
        raise ValueError("development_fixture_floor_prim_conflict")
    x, y = float(target_world_m[0]), float(target_world_m[1])
    mesh = UsdGeom.Mesh.Define(stage, floor_path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(x - 2.5, y - 2.5, 0.0),
        Gf.Vec3f(x + 2.5, y - 2.5, 0.0),
        Gf.Vec3f(x - 2.5, y + 2.5, 0.0),
        Gf.Vec3f(x + 2.5, y + 2.5, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    stage.GetRootLayer().customLayerData = {
        "blueprintPurpose": "development_fixture_robot_placement_only",
        "blueprintSourceCollisionDigest": _digest(source_collision),
        "blueprintQualifiedAssetDigest": qualified_asset_digest,
    }
    destination = output_root / "development_fixture_placement_collision.usda"
    with tempfile.NamedTemporaryFile(dir=output_root, suffix=".usda", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if not stage.GetRootLayer().Export(str(temporary_path)):
            raise ValueError("development_fixture_floor_export_failed")
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != temporary_path.read_bytes():
                raise ValueError("development_fixture_floor_conflict")
        else:
            os.link(temporary_path, destination)
            destination.chmod(0o440)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
