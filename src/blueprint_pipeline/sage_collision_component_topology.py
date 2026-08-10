"""Connected-component and open-cavity evidence for SAGE collision meshes.

SAGE sometimes groups many disconnected scene objects into one USD mesh prim.
Prim-level bounds therefore cannot establish the collision identity of an
InteriorGS instance.  This module applies the authored USD transforms, splits
overlapping meshes by face connectivity, ranks each component against exact
publisher label bounds, and optionally probes a component with vertical rays.

The result is geometric public-dataset evidence only.  An open collision cavity
does not prove hidden appearance, material, physical equivalence, or native
simulator behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_placement.interiorgs_index import SceneObject, load_interiorgs_labels


SCHEMA_VERSION = "interiorgs_sage_collision_component_topology.v1"


class SageCollisionComponentTopologyError(ValueError):
    """Stable, sorted topology-inspection failures."""

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
    candidate_min: Sequence[float],
    candidate_max: Sequence[float],
) -> tuple[float, float, float]:
    intersection_extent = [
        max(
            0.0,
            min(float(target_max[index]), float(candidate_max[index]))
            - max(float(target_min[index]), float(candidate_min[index])),
        )
        for index in range(3)
    ]
    intersection = (
        intersection_extent[0] * intersection_extent[1] * intersection_extent[2]
    )
    target_volume = 1.0
    candidate_volume = 1.0
    for index in range(3):
        target_volume *= max(0.0, float(target_max[index]) - float(target_min[index]))
        candidate_volume *= max(
            0.0, float(candidate_max[index]) - float(candidate_min[index])
        )
    union = target_volume + candidate_volume - intersection
    return (
        intersection / union if union > 0.0 else 0.0,
        intersection / target_volume if target_volume > 0.0 else 0.0,
        intersection / candidate_volume if candidate_volume > 0.0 else 0.0,
    )


def _boxes_overlap(
    first_min: Sequence[float],
    first_max: Sequence[float],
    second_min: Sequence[float],
    second_max: Sequence[float],
) -> bool:
    return all(
        min(float(first_max[index]), float(second_max[index]))
        > max(float(first_min[index]), float(second_min[index]))
        for index in range(3)
    )


def _face_rows(counts: Sequence[int], indices: Sequence[int]) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    cursor = 0
    for raw_count in counts:
        count = int(raw_count)
        face = tuple(int(value) for value in indices[cursor : cursor + count])
        cursor += count
        if count >= 3:
            rows.append(face)
    if cursor != len(indices):
        raise SageCollisionComponentTopologyError(
            ["sage_collision_face_index_count_mismatch"]
        )
    return rows


def _connected_face_components(
    *, point_count: int, faces: Sequence[tuple[int, ...]]
) -> list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]]:
    parent = list(range(point_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    for face in faces:
        anchor = face[0]
        if anchor < 0 or anchor >= point_count:
            raise SageCollisionComponentTopologyError(
                ["sage_collision_face_vertex_index_invalid"]
            )
        for value in face[1:]:
            if value < 0 or value >= point_count:
                raise SageCollisionComponentTopologyError(
                    ["sage_collision_face_vertex_index_invalid"]
                )
            union(anchor, value)

    grouped_faces: dict[int, list[tuple[int, ...]]] = {}
    for face in faces:
        grouped_faces.setdefault(find(face[0]), []).append(face)
    result = []
    for root, component_faces in grouped_faces.items():
        vertices = tuple(sorted({value for face in component_faces for value in face}))
        result.append((vertices, tuple(component_faces)))
    result.sort(key=lambda row: (row[0][0], len(row[0]), len(row[1])))
    return result


def _triangles(faces: Iterable[Sequence[int]]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        for index in range(1, len(face) - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
    return triangles


def _vertical_first_hit(
    x: float,
    y: float,
    *,
    vertices: Sequence[Sequence[float]],
    triangles: Sequence[tuple[int, int, int]],
    start_z: float,
) -> float | None:
    hits: list[float] = []
    for first, second, third in triangles:
        a = vertices[first]
        b = vertices[second]
        c = vertices[third]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
            a[1] - c[1]
        )
        if abs(denominator) <= 1e-14:
            continue
        weight_a = (
            (b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])
        ) / denominator
        weight_b = (
            (c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])
        ) / denominator
        weight_c = 1.0 - weight_a - weight_b
        if min(weight_a, weight_b, weight_c) < -1e-10:
            continue
        z = weight_a * a[2] + weight_b * b[2] + weight_c * c[2]
        if z <= start_z + 1e-10:
            hits.append(float(z))
    return max(hits) if hits else None


def _opening_probe(
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    grid_size: int,
    margin_fraction: float,
    floor_band_fraction: float,
    cap_band_fraction: float,
) -> dict[str, Any]:
    if grid_size < 3 or grid_size % 2 == 0:
        raise SageCollisionComponentTopologyError(
            ["opening_probe_grid_size_must_be_odd_and_at_least_three"]
        )
    if not 0.0 <= margin_fraction < 0.5:
        raise SageCollisionComponentTopologyError(
            ["opening_probe_margin_fraction_invalid"]
        )
    extent = [float(bounds_max[i]) - float(bounds_min[i]) for i in range(3)]
    if min(extent) <= 0.0:
        raise SageCollisionComponentTopologyError(["opening_probe_bounds_degenerate"])
    x_min = float(bounds_min[0]) + extent[0] * margin_fraction
    x_max = float(bounds_max[0]) - extent[0] * margin_fraction
    y_min = float(bounds_min[1]) + extent[1] * margin_fraction
    y_max = float(bounds_max[1]) - extent[1] * margin_fraction
    x_values = [
        x_min + (x_max - x_min) * index / (grid_size - 1)
        for index in range(grid_size)
    ]
    y_values = [
        y_min + (y_max - y_min) * index / (grid_size - 1)
        for index in range(grid_size)
    ]
    triangles = _triangles(faces)
    start_z = float(bounds_max[2]) + max(0.01, extent[2] * 0.1)
    rows: list[dict[str, Any]] = []
    floor_cells: list[tuple[float, float, float]] = []
    cap_count = 0
    missing_count = 0
    for y in y_values:
        for x in x_values:
            hit = _vertical_first_hit(
                x,
                y,
                vertices=vertices,
                triangles=triangles,
                start_z=start_z,
            )
            if hit is None:
                band = "no_hit"
                normalized_height = None
                missing_count += 1
            else:
                normalized_height = (hit - float(bounds_min[2])) / extent[2]
                if normalized_height <= floor_band_fraction:
                    band = "floor"
                    floor_cells.append((x, y, hit))
                elif normalized_height >= cap_band_fraction:
                    band = "cap"
                    cap_count += 1
                else:
                    band = "mid"
            rows.append(
                {
                    "x_m": round(x, 9),
                    "y_m": round(y, 9),
                    "first_hit_z_m": None if hit is None else round(hit, 9),
                    "normalized_height": (
                        None
                        if normalized_height is None
                        else round(normalized_height, 9)
                    ),
                    "band": band,
                }
            )
    center = rows[len(rows) // 2]
    sample_count = len(rows)
    floor_fraction = len(floor_cells) / sample_count
    cap_fraction = cap_count / sample_count
    clear_bounds = None
    if floor_cells:
        step_x = (x_max - x_min) / (grid_size - 1)
        step_y = (y_max - y_min) / (grid_size - 1)
        clear_min_x = max(
            float(bounds_min[0]), min(row[0] for row in floor_cells) - step_x / 2.0
        )
        clear_max_x = min(
            float(bounds_max[0]), max(row[0] for row in floor_cells) + step_x / 2.0
        )
        clear_min_y = max(
            float(bounds_min[1]), min(row[1] for row in floor_cells) - step_y / 2.0
        )
        clear_max_y = min(
            float(bounds_max[1]), max(row[1] for row in floor_cells) + step_y / 2.0
        )
        clear_bounds = {
            "world_xy_min_m": [round(clear_min_x, 9), round(clear_min_y, 9)],
            "world_xy_max_m": [round(clear_max_x, 9), round(clear_max_y, 9)],
            "size_xy_m": [
                round(clear_max_x - clear_min_x, 9),
                round(clear_max_y - clear_min_y, 9),
            ],
        }
    median_floor_z = None
    if floor_cells:
        ordered = sorted(row[2] for row in floor_cells)
        median_floor_z = ordered[len(ordered) // 2]
    open_cavity = (
        center["band"] == "floor"
        and floor_fraction >= 0.2
        and cap_fraction < 0.5
    )
    return {
        "grid_size": grid_size,
        "sample_count": sample_count,
        "margin_fraction": margin_fraction,
        "floor_band_fraction": floor_band_fraction,
        "cap_band_fraction": cap_band_fraction,
        "center_first_hit_band": center["band"],
        "floor_hit_count": len(floor_cells),
        "floor_hit_fraction": round(floor_fraction, 9),
        "cap_hit_count": cap_count,
        "cap_hit_fraction": round(cap_fraction, 9),
        "no_hit_count": missing_count,
        "median_floor_z_m": (
            None if median_floor_z is None else round(median_floor_z, 9)
        ),
        "cavity_depth_m": (
            None
            if median_floor_z is None
            else round(float(bounds_max[2]) - median_floor_z, 9)
        ),
        "conservative_clear_opening": clear_bounds,
        "open_collision_cavity_passed": open_cavity,
        "samples": rows,
        "claim_boundary": {
            "hidden_appearance_observed": False,
            "material_properties_observed": False,
            "native_collision_qualified": False,
            "physical_equivalence_proven": False,
        },
    }


def _target_map(
    labels_path: Path, target_instance_ids: Sequence[str]
) -> dict[str, SceneObject]:
    labels = load_interiorgs_labels(labels_path)
    requested = [str(value) for value in target_instance_ids]
    if not requested or len(set(requested)) != len(requested):
        raise SageCollisionComponentTopologyError(
            ["target_instance_ids_missing_or_duplicate"]
        )
    targets = {row.id: row for row in labels if row.id in set(requested)}
    missing = sorted(set(requested) - set(targets))
    if missing:
        raise SageCollisionComponentTopologyError(
            [f"interiorgs_target_instance_missing:{value}" for value in missing]
        )
    return targets


def inspect_sage_collision_component_topology(
    *,
    labels_path: str | Path,
    target_instance_ids: Sequence[str],
    sage_collision_usd_path: str | Path,
    opening_probe_instance_ids: Sequence[str] = (),
    minimum_component_iou: float = 0.85,
    opening_grid_size: int = 9,
    opening_margin_fraction: float = 0.1,
) -> dict[str, Any]:
    """Match exact label boxes to transformed connected collision components."""

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise SageCollisionComponentTopologyError(
            ["sage_collision_openusd_runtime_missing"]
        ) from exc

    labels = Path(labels_path).expanduser().resolve()
    collision = Path(sage_collision_usd_path).expanduser().resolve()
    if not labels.is_file() or labels.is_symlink():
        raise SageCollisionComponentTopologyError(["interiorgs_labels_missing"])
    if not collision.is_file() or collision.is_symlink():
        raise SageCollisionComponentTopologyError(["sage_collision_usd_missing"])
    targets = _target_map(labels, target_instance_ids)
    opening_targets = {str(value) for value in opening_probe_instance_ids}
    if not opening_targets.issubset(targets):
        raise SageCollisionComponentTopologyError(
            ["opening_probe_target_not_in_requested_targets"]
        )
    if not 0.0 < minimum_component_iou <= 1.0:
        raise SageCollisionComponentTopologyError(["minimum_component_iou_invalid"])

    stage = Usd.Stage.Open(str(collision), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SageCollisionComponentTopologyError(["sage_collision_usd_open_failed"])
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise SageCollisionComponentTopologyError(["sage_collision_stage_not_z_up"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise SageCollisionComponentTopologyError(
            ["sage_collision_stage_not_meter_units"]
        )

    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        target_id: [] for target_id in targets
    }
    geometry_by_key: dict[
        tuple[str, int], tuple[list[list[float]], tuple[tuple[int, ...], ...]]
    ] = {}
    inspected_mesh_count = 0
    inspected_component_count = 0

    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        aligned = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        prim_min = [float(aligned.GetMin()[index]) for index in range(3)]
        prim_max = [float(aligned.GetMax()[index]) for index in range(3)]
        overlapping_targets = {
            target_id: target
            for target_id, target in targets.items()
            if _boxes_overlap(target.bbox_min, target.bbox_max, prim_min, prim_max)
        }
        if not overlapping_targets:
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
        if not raw_points or not counts or not indices:
            continue
        inspected_mesh_count += 1
        world_matrix = transform_cache.GetLocalToWorldTransform(prim)
        points: list[list[float]] = []
        for point in raw_points:
            world = world_matrix.Transform(
                Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            )
            points.append([float(world[0]), float(world[1]), float(world[2])])
        faces = _face_rows([int(value) for value in counts], [int(value) for value in indices])
        components = _connected_face_components(point_count=len(points), faces=faces)
        inspected_component_count += len(components)
        for component_index, (vertex_ids, component_faces) in enumerate(components):
            component_points = [points[index] for index in vertex_ids]
            component_min = [min(row[index] for row in component_points) for index in range(3)]
            component_max = [max(row[index] for row in component_points) for index in range(3)]
            matched_targets = {
                target_id: target
                for target_id, target in overlapping_targets.items()
                if _boxes_overlap(
                    target.bbox_min, target.bbox_max, component_min, component_max
                )
            }
            if not matched_targets:
                continue
            geometry_key = (str(prim.GetPath()), component_index)
            geometry_by_key[geometry_key] = (points, component_faces)
            geometry_digest = canonical_digest(
                {
                    "vertices_world_m": [
                        [round(value, 9) for value in points[index]]
                        for index in vertex_ids
                    ],
                    "faces_source_vertex_indices": [list(face) for face in component_faces],
                }
            )
            for target_id, target in matched_targets.items():
                iou, target_coverage, component_coverage = _box_metrics(
                    target.bbox_min,
                    target.bbox_max,
                    component_min,
                    component_max,
                )
                candidate_rows[target_id].append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "component_index": component_index,
                        "world_aabb_min_m": [round(value, 9) for value in component_min],
                        "world_aabb_max_m": [round(value, 9) for value in component_max],
                        "world_aabb_size_m": [
                            round(component_max[i] - component_min[i], 9)
                            for i in range(3)
                        ],
                        "vertex_count": len(vertex_ids),
                        "face_count": len(component_faces),
                        "collision_api_applied": prim.HasAPI(UsdPhysics.CollisionAPI),
                        "aabb_iou": round(iou, 9),
                        "target_coverage_fraction": round(target_coverage, 9),
                        "component_coverage_fraction": round(component_coverage, 9),
                        "geometry_digest": geometry_digest,
                    }
                )

    target_rows: list[dict[str, Any]] = []
    for target_id in [str(value) for value in target_instance_ids]:
        target = targets[target_id]
        ranked = sorted(
            candidate_rows[target_id],
            key=lambda row: (
                -float(row["aabb_iou"]),
                -float(row["target_coverage_fraction"]),
                str(row["prim_path"]),
                int(row["component_index"]),
            ),
        )
        qualified = [
            row
            for row in ranked
            if row["collision_api_applied"]
            and float(row["aabb_iou"]) >= minimum_component_iou
        ]
        best = dict(ranked[0]) if ranked else None
        opening_probe = None
        if best is not None and target_id in opening_targets:
            key = (str(best["prim_path"]), int(best["component_index"]))
            points, faces = geometry_by_key[key]
            opening_probe = _opening_probe(
                vertices=points,
                faces=faces,
                bounds_min=best["world_aabb_min_m"],
                bounds_max=best["world_aabb_max_m"],
                grid_size=opening_grid_size,
                margin_fraction=opening_margin_fraction,
                floor_band_fraction=0.25,
                cap_band_fraction=0.75,
            )
        target_rows.append(
            {
                "interiorgs_instance_id": target_id,
                "semantic_label": target.label,
                "label_world_aabb_min_m": [
                    round(float(value), 9) for value in target.bbox_min
                ],
                "label_world_aabb_max_m": [
                    round(float(value), 9) for value in target.bbox_max
                ],
                "candidate_component_count": len(ranked),
                "qualified_component_count": len(qualified),
                "component_collision_identity_passed": len(qualified) == 1,
                "best_component": best,
                "opening_probe": opening_probe,
            }
        )

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
            "authored_prim_local_to_world_transforms_applied": True,
            "interiorgs_to_sage_world_transform": "identity_after_authored_prim_transforms",
        },
        "thresholds": {
            "minimum_component_iou": minimum_component_iou,
            "opening_grid_size": opening_grid_size,
            "opening_margin_fraction": opening_margin_fraction,
        },
        "inspected_overlapping_mesh_count": inspected_mesh_count,
        "inspected_connected_component_count": inspected_component_count,
        "targets": target_rows,
        "all_component_collision_identities_passed": all(
            row["component_collision_identity_passed"] for row in target_rows
        ),
        "claim_boundary": {
            "publisher_frame_registration_consistency_only": True,
            "measurement_authoritative_surface_truth": False,
            "hidden_appearance_observed": False,
            "native_simulator_contacts_qualified": False,
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
        raise SageCollisionComponentTopologyError(
            [f"path_outside_approved_roots:{resolved}"]
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--target-instance-id", action="append", required=True)
    parser.add_argument("--opening-probe-instance-id", action="append", default=[])
    parser.add_argument("--sage-collision-usd", required=True)
    parser.add_argument("--minimum-component-iou", type=float, default=0.85)
    parser.add_argument("--opening-grid-size", type=int, default=9)
    parser.add_argument("--opening-margin-fraction", type=float, default=0.1)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    labels = _resolved_under(args.labels, args.approved_root)
    collision = _resolved_under(args.sage_collision_usd, args.approved_root)
    output = _resolved_under(args.out, args.approved_root)
    result = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=args.target_instance_id,
        opening_probe_instance_ids=args.opening_probe_instance_id,
        sage_collision_usd_path=collision,
        minimum_component_iou=args.minimum_component_iou,
        opening_grid_size=args.opening_grid_size,
        opening_margin_fraction=args.opening_margin_fraction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "receipt_digest": result["receipt_digest"],
                "all_component_collision_identities_passed": result[
                    "all_component_collision_identities_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "SageCollisionComponentTopologyError",
    "inspect_sage_collision_component_topology",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
