"""Digest-bound InteriorGS instance to SAGE collision-mesh identity analysis.

The publisher scene-ID join is necessary but not sufficient for object removal.
This module opens every SAGE mesh, computes its world AABB, and ranks exact
overlap with one InteriorGS publisher instance.  A whole-object match and a
contained submesh are reported separately; neither is promoted to an observed
moving link, joint, contact region, or dynamic articulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_placement.interiorgs_index import load_interiorgs_labels


SCHEMA_VERSION = "interiorgs_sage_collision_identity.v1"


class SageCollisionIdentityError(ValueError):
    """Stable, sorted SAGE collision identity failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _box_metrics(
    target_min: Sequence[float],
    target_max: Sequence[float],
    mesh_min: Sequence[float],
    mesh_max: Sequence[float],
) -> tuple[float, float, float]:
    intersection_extents = [
        max(0.0, min(float(target_max[i]), float(mesh_max[i])) - max(float(target_min[i]), float(mesh_min[i])))
        for i in range(3)
    ]
    intersection = intersection_extents[0] * intersection_extents[1] * intersection_extents[2]
    target_volume = 1.0
    mesh_volume = 1.0
    for index in range(3):
        target_volume *= max(0.0, float(target_max[index]) - float(target_min[index]))
        mesh_volume *= max(0.0, float(mesh_max[index]) - float(mesh_min[index]))
    union = target_volume + mesh_volume - intersection
    iou = intersection / union if union > 0.0 else 0.0
    target_coverage = intersection / target_volume if target_volume > 0.0 else 0.0
    mesh_coverage = intersection / mesh_volume if mesh_volume > 0.0 else 0.0
    return iou, target_coverage, mesh_coverage


def inspect_sage_collision_identity(
    *,
    labels_path: str | Path,
    target_instance_id: str,
    sage_collision_usd_path: str | Path,
    minimum_whole_object_iou: float = 0.85,
    minimum_part_mesh_coverage: float = 0.95,
    minimum_part_target_coverage: float = 0.05,
    maximum_part_target_coverage: float = 0.8,
) -> dict[str, Any]:
    """Rank SAGE meshes against one InteriorGS instance's conservative AABB."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise SageCollisionIdentityError(["sage_collision_openusd_runtime_missing"]) from exc

    labels = Path(labels_path).expanduser().resolve()
    collision = Path(sage_collision_usd_path).expanduser().resolve()
    if not labels.is_file() or labels.is_symlink():
        raise SageCollisionIdentityError(["interiorgs_labels_missing"])
    if not collision.is_file() or collision.is_symlink():
        raise SageCollisionIdentityError(["sage_collision_usd_missing"])
    matches = [
        item for item in load_interiorgs_labels(labels) if item.id == str(target_instance_id)
    ]
    if len(matches) != 1:
        raise SageCollisionIdentityError(["interiorgs_target_instance_not_exactly_one"])
    target = matches[0]
    stage = Usd.Stage.Open(str(collision), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SageCollisionIdentityError(["sage_collision_usd_open_failed"])
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise SageCollisionIdentityError(["sage_collision_stage_not_z_up"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise SageCollisionIdentityError(["sage_collision_stage_not_meter_units"])

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    mesh_rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
        if not points or not counts:
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        raw_min = aligned.GetMin()
        raw_max = aligned.GetMax()
        mesh_min = [float(raw_min[index]) for index in range(3)]
        mesh_max = [float(raw_max[index]) for index in range(3)]
        iou, target_coverage, mesh_coverage = _box_metrics(
            target.bbox_min, target.bbox_max, mesh_min, mesh_max
        )
        if target_coverage <= 0.0:
            continue
        mesh_rows.append(
            {
                "prim_path": str(prim.GetPath()),
                "world_aabb_min_m": [round(value, 9) for value in mesh_min],
                "world_aabb_max_m": [round(value, 9) for value in mesh_max],
                "point_count": len(points),
                "face_count": len(counts),
                "collision_api_applied": prim.HasAPI(UsdPhysics.CollisionAPI),
                "aabb_iou": round(iou, 9),
                "target_coverage_fraction": round(target_coverage, 9),
                "mesh_coverage_fraction": round(mesh_coverage, 9),
            }
        )
    mesh_rows.sort(
        key=lambda row: (
            -float(row["aabb_iou"]),
            -float(row["target_coverage_fraction"]),
            str(row["prim_path"]),
        )
    )
    whole_matches = [
        row
        for row in mesh_rows
        if row["collision_api_applied"]
        and float(row["aabb_iou"]) >= float(minimum_whole_object_iou)
    ]
    part_candidates = [
        row
        for row in mesh_rows
        if row["collision_api_applied"]
        and float(row["mesh_coverage_fraction"]) >= float(minimum_part_mesh_coverage)
        and float(row["target_coverage_fraction"]) >= float(minimum_part_target_coverage)
        and float(row["target_coverage_fraction"]) <= float(maximum_part_target_coverage)
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_files": {
            "interiorgs_labels": {
                "path": labels.name,
                "size_bytes": labels.stat().st_size,
                "sha256": _sha256(labels),
            },
            "sage_collision_usd": {
                "path": collision.name,
                "size_bytes": collision.stat().st_size,
                "sha256": _sha256(collision),
            },
        },
        "coordinate_frame": {
            "up_axis": "Z",
            "meters_per_unit": meters_per_unit,
            "transform_applied": "identity",
        },
        "target": {
            "interiorgs_instance_id": target.id,
            "semantic_label": target.label,
            "world_aabb_min_m": [round(float(value), 9) for value in target.bbox_min],
            "world_aabb_max_m": [round(float(value), 9) for value in target.bbox_max],
        },
        "thresholds": {
            "minimum_whole_object_iou": float(minimum_whole_object_iou),
            "minimum_part_mesh_coverage": float(minimum_part_mesh_coverage),
            "minimum_part_target_coverage": float(minimum_part_target_coverage),
            "maximum_part_target_coverage": float(maximum_part_target_coverage),
        },
        "overlapping_meshes": mesh_rows,
        "whole_object_matches": whole_matches,
        "candidate_subpart_meshes": part_candidates,
        "whole_object_collision_identity_passed": len(whole_matches) == 1,
        "candidate_subpart_count": len(part_candidates),
        "claim_boundary": {
            "candidate_subpart_is_not_moving_link_proof": True,
            "joint_axis_or_limits_proven": False,
            "source_visual_partition_proven": False,
            "independent_dynamic_articulation_proven": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def _resolved_under(path: str | Path, approved_roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in approved_roots]
    if not roots or not any(resolved == root or root in resolved.parents for root in roots):
        raise SageCollisionIdentityError([f"path_outside_approved_roots:{resolved}"])
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--target-instance-id", required=True)
    parser.add_argument("--sage-collision-usd", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    labels = _resolved_under(args.labels, args.approved_root)
    collision = _resolved_under(args.sage_collision_usd, args.approved_root)
    output = _resolved_under(args.out, args.approved_root)
    result = inspect_sage_collision_identity(
        labels_path=labels,
        target_instance_id=args.target_instance_id,
        sage_collision_usd_path=collision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "receipt_digest": result["receipt_digest"],
                "whole_object_collision_identity_passed": result[
                    "whole_object_collision_identity_passed"
                ],
                "candidate_subpart_count": result["candidate_subpart_count"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "SageCollisionIdentityError",
    "inspect_sage_collision_identity",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
