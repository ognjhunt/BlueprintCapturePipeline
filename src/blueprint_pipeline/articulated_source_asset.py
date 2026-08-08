"""Materialize one exact SAGE object mesh as a Joint Agent source asset.

The extractor is deterministic and scene-neutral.  It requires one qualified
whole-object InteriorGS/SAGE collision match, transforms the exact source mesh
into a local asset frame, retains topology, and reports all disconnected mesh
components.  It does not split, name, rig, simplify, or physically qualify any
component; those are downstream released-code and native-simulator gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .sage_collision_identity import (
    SageCollisionIdentityError,
    inspect_sage_collision_identity,
)


SCHEMA_VERSION = "articulated_source_asset.v1"


class ArticulatedSourceAssetError(ValueError):
    """Stable, sorted target-extraction failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _connected_components(
    points: Sequence[Sequence[float]],
    counts: Sequence[int],
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    parent = list(range(len(points)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    faces: list[list[int]] = []
    offset = 0
    for raw_count in counts:
        count = int(raw_count)
        face = [int(value) for value in indices[offset : offset + count]]
        offset += count
        if count < 3 or any(value < 0 or value >= len(points) for value in face):
            raise ArticulatedSourceAssetError(["source_mesh_topology_invalid"])
        for vertex in face[1:]:
            union(face[0], vertex)
        faces.append(face)
    if offset != len(indices) or not faces:
        raise ArticulatedSourceAssetError(["source_mesh_topology_invalid"])
    grouped_faces: dict[int, list[list[int]]] = {}
    for face in faces:
        grouped_faces.setdefault(find(face[0]), []).append(face)
    rows: list[dict[str, Any]] = []
    for face_group in grouped_faces.values():
        vertex_ids = sorted({vertex for face in face_group for vertex in face})
        component_points = [points[index] for index in vertex_ids]
        minimum = [min(float(point[axis]) for point in component_points) for axis in range(3)]
        maximum = [max(float(point[axis]) for point in component_points) for axis in range(3)]
        rows.append(
            {
                "vertex_count": len(vertex_ids),
                "face_count": len(face_group),
                "aabb_min_asset_m": [round(value, 9) for value in minimum],
                "aabb_max_asset_m": [round(value, 9) for value in maximum],
                "aabb_extent_m": [
                    round(maximum[axis] - minimum[axis], 9) for axis in range(3)
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["face_count"]),
            -int(row["vertex_count"]),
            row["aabb_min_asset_m"],
        )
    )
    for index, row in enumerate(rows):
        row["component_index"] = index
    return rows


def materialize_articulated_source_asset(
    *,
    labels_path: str | Path,
    target_instance_id: str,
    sage_collision_usd_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Extract the unique qualified SAGE target mesh into a local-frame USDA."""

    try:
        from pxr import Gf, Usd, UsdGeom, Vt
    except ImportError as exc:
        raise ArticulatedSourceAssetError(["openusd_runtime_missing"]) from exc

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ArticulatedSourceAssetError(["articulated_source_output_not_empty"])
    destination.mkdir(parents=True, exist_ok=True)
    try:
        identity = inspect_sage_collision_identity(
            labels_path=labels_path,
            target_instance_id=target_instance_id,
            sage_collision_usd_path=sage_collision_usd_path,
        )
    except SageCollisionIdentityError as exc:
        raise ArticulatedSourceAssetError(exc.errors) from exc
    whole_matches = identity["whole_object_matches"]
    if len(whole_matches) != 1:
        raise ArticulatedSourceAssetError(["unique_whole_object_collision_match_missing"])

    collision = Path(sage_collision_usd_path).expanduser().resolve()
    source_stage = Usd.Stage.Open(str(collision), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise ArticulatedSourceAssetError(["sage_collision_usd_open_failed"])
    source_prim_path = str(whole_matches[0]["prim_path"])
    source_prim = source_stage.GetPrimAtPath(source_prim_path)
    if not source_prim or not source_prim.IsA(UsdGeom.Mesh):
        raise ArticulatedSourceAssetError(["matched_source_prim_not_mesh"])
    source_mesh = UsdGeom.Mesh(source_prim)
    points = source_mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
    counts = source_mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
    indices = source_mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
    if not points or not counts or not indices:
        raise ArticulatedSourceAssetError(["matched_source_mesh_topology_missing"])

    target = identity["target"]
    target_min = [float(value) for value in target["world_aabb_min_m"]]
    target_max = [float(value) for value in target["world_aabb_max_m"]]
    origin_world = [
        0.5 * (target_min[0] + target_max[0]),
        0.5 * (target_min[1] + target_max[1]),
        target_min[2],
    ]
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        source_prim
    )
    local_points: list[tuple[float, float, float]] = []
    for raw_point in points:
        world = transform.Transform(Gf.Vec3d(*[float(value) for value in raw_point]))
        local_points.append(
            tuple(float(world[index]) - origin_world[index] for index in range(3))
        )
    components = _connected_components(local_points, counts, indices)

    output_usd = destination / "articulated_source_mesh.usda"
    output_stage = Usd.Stage.CreateNew(str(output_usd))
    if output_stage is None:
        raise ArticulatedSourceAssetError(["articulated_source_stage_create_failed"])
    UsdGeom.SetStageUpAxis(output_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(output_stage, 1.0)
    root = UsdGeom.Xform.Define(output_stage, "/Asset")
    output_stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(output_stage, "/Asset/source_mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in local_points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(value) for value in counts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(value) for value in indices]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.58, 0.62)]))
    extent = UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get())
    mesh.CreateExtentAttr(extent)
    root.GetPrim().SetCustomDataByKey("blueprint:sourcePrimPath", source_prim_path)
    root.GetPrim().SetCustomDataByKey(
        "blueprint:sourceCollisionSha256",
        identity["source_files"]["sage_collision_usd"]["sha256"],
    )
    output_stage.GetRootLayer().documentation = (
        "Blueprint deterministic SAGE target extraction; construction candidate only"
    )
    output_stage.GetRootLayer().Save()

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "target": target,
        "source_collision_identity_receipt_digest": identity["receipt_digest"],
        "source_collision_prim_path": source_prim_path,
        "source_files": identity["source_files"],
        "asset_frame": {
            "name": "articulated_asset_local_right_back_up",
            "units": "meters",
            "up_axis": "Z",
            "origin_world_m": [round(value, 9) for value in origin_world],
            "T_asset_world": [
                [1.0, 0.0, 0.0, -origin_world[0]],
                [0.0, 1.0, 0.0, -origin_world[1]],
                [0.0, 0.0, 1.0, -origin_world[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "T_world_asset": [
                [1.0, 0.0, 0.0, origin_world[0]],
                [0.0, 1.0, 0.0, origin_world[1]],
                [0.0, 0.0, 1.0, origin_world[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "output_asset": {
            "relative_path": output_usd.name,
            "size_bytes": output_usd.stat().st_size,
            "sha256": _sha256(output_usd),
            "default_prim": "/Asset",
            "mesh_prim": "/Asset/source_mesh",
            "point_count": len(local_points),
            "face_count": len(counts),
        },
        "connected_component_count": len(components),
        "connected_components": components,
        "joint_agent_0_5_2_input": {
            "usd_path_ready": True,
            "default_prim_valid": True,
            "single_source_mesh_requires_split_meshes": len(components) > 1,
            "predicted_split_prim_count": len(components),
            "topology_inference_executed": False,
        },
        "claim_boundary": {
            "source_topology_preserved": True,
            "connected_components_are_not_rigid_links": True,
            "component_names_or_roles_inferred": False,
            "joint_topology_inferred": False,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = destination / "articulated_source_asset_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _resolved_under(path: str | Path, approved_roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in approved_roots]
    if not roots or not any(resolved == root or root in resolved.parents for root in roots):
        raise ArticulatedSourceAssetError([f"path_outside_approved_roots:{resolved}"])
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--target-instance-id", required=True)
    parser.add_argument("--sage-collision-usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    args = parser.parse_args(argv)
    result = materialize_articulated_source_asset(
        labels_path=_resolved_under(args.labels, args.approved_root),
        target_instance_id=args.target_instance_id,
        sage_collision_usd_path=_resolved_under(
            args.sage_collision_usd, args.approved_root
        ),
        output_dir=_resolved_under(args.output_dir, args.approved_root),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "receipt_digest": result["receipt_digest"],
                "connected_component_count": result["connected_component_count"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ArticulatedSourceAssetError",
    "SCHEMA_VERSION",
    "materialize_articulated_source_asset",
]


if __name__ == "__main__":
    raise SystemExit(main())
