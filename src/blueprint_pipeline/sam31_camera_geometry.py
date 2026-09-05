"""CPU screening of calibrated SAM views using observed InteriorGS room geometry.

Conservative label/wall bounds can reject a view, but cannot qualify captured
appearance or visibility. Rendered contribution and fidelity gates remain required.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_placement.interiorgs_index import InteriorGSSceneSpatialIndex
from .task_evaluation_scene_configuration_submission_inputs import read, require, sha

SCREEN_SCHEMA = "sam31_camera_geometry_screen.v1"
GENERATOR = "interiorgs_room_occlusion_screen_v1"


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _segment_hits(start: Sequence[float], end: Sequence[float], lower: Sequence[float],
                  upper: Sequence[float]) -> bool:
    """Closed slab intersection, excluding the target endpoint itself."""
    entry, leave = 0.0, 1.0 - 1e-7
    for axis in range(3):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not lower[axis] <= start[axis] <= upper[axis]:
                return False
            continue
        first, second = sorted(((lower[axis] - start[axis]) / delta,
                                (upper[axis] - start[axis]) / delta))
        entry, leave = max(entry, first), min(leave, second)
        if entry > leave:
            return False
    return entry <= leave


def _unit(vector: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector]


def _fits_frame(position: Sequence[float], center: Sequence[float],
                corners: Sequence[Sequence[float]]) -> bool:
    forward = _unit([center[i] - position[i] for i in range(3)])
    right = _unit([forward[1], -forward[0], 0.0])
    up = [right[1] * forward[2], -right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]]
    limit = math.tan(math.radians(55.0) / 2.0) * 0.95
    for corner in corners:
        ray = [corner[i] - position[i] for i in range(3)]
        depth = sum(ray[i] * forward[i] for i in range(3))
        if depth <= 0 or any(abs(sum(ray[i] * axis[i] for i in range(3))) > depth * limit
                             for axis in (right, up)):
            return False
    return True


def _offset_inventory(extent: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()

    def append(angle: float, radius: float, height: float, legacy_id: str | None = None) -> None:
        offset = tuple(round(extent * v, 9) for v in
                       (radius * math.cos(angle), radius * math.sin(angle), height))
        if offset in seen:
            return
        seen.add(offset)
        rows.append({"candidate_id": f"candidate-{len(rows) + 1:04d}",
                     "position_offset_m": list(offset), "legacy_camera_id": legacy_id})

    for i in range(16):
        append(2 * math.pi * i / 16, (1.5, 1.8, 2.1, 2.4)[i % 4],
               (0.65, 0.95, 1.3, 1.7)[i % 4], f"source-{i + 1:02d}")
    for radius, height, angle_index in itertools.product((1.5, 1.8, 2.1, 2.4),
                                                        (0.65, 0.95, 1.3, 1.7), range(32)):
        append(2 * math.pi * angle_index / 32, radius, height)
    return rows


def select_geometry_aware_camera_policy(
    *, labels_path: Path, structure_path: Path, collision_identity_path: Path,
    target_instance_id: str, source_min: Sequence[float], source_max: Sequence[float],
) -> dict[str, Any]:
    """Select sixteen views and up to sixteen sealed reserve poses without rendering."""
    paths = {"labels": Path(labels_path), "structure": Path(structure_path),
             "collision_identity": Path(collision_identity_path)}
    sources = {name: _record(path) for name, path in paths.items()}
    identity = read(paths["collision_identity"], digest_field="receipt_digest")
    require(identity.get("schema_version") == "interiorgs_sage_collision_identity.v1"
            and identity.get("whole_object_collision_identity_passed") is True
            and str(identity.get("target", {}).get("interiorgs_instance_id")) == str(target_instance_id)
            and identity.get("source_files", {}).get("interiorgs_labels", {}).get("sha256")
            == sources["labels"]["sha256"], "sam31_camera_geometry_identity_invalid")
    index = InteriorGSSceneSpatialIndex(paths["labels"], paths["structure"])
    target = index.object_by_instance(target_instance_id)
    require(target is not None and index.structure is not None, "sam31_camera_geometry_target_missing")
    lower, upper = list(source_min), list(source_max)
    require(len(lower) == len(upper) == 3 and
            all(math.isfinite(v) for v in lower + upper) and
            all(lo < hi for lo, hi in zip(lower, upper, strict=True)) and
            all(abs(a - b) <= 1e-6 for a, b in zip(lower + upper,
                list(target.bbox_min) + list(target.bbox_max), strict=True)),
            "sam31_camera_geometry_bounds_mismatch")
    require(all(abs(a - b) <= 1e-6 for a, b in zip(lower + upper,
        identity["target"]["world_aabb_min_m"] + identity["target"]["world_aabb_max_m"], strict=True)),
        "sam31_camera_geometry_identity_bounds_mismatch")
    center = [(lo + hi) / 2 for lo, hi in zip(lower, upper, strict=True)]
    extent = max(hi - lo for lo, hi in zip(lower, upper, strict=True))
    room = index.structure.room_index_of_point(tuple(center[:2]))
    require(room is not None, "sam31_camera_geometry_target_room_missing")
    obstacles = [box for box in index.obstacle_boxes() if box.id != str(target_instance_id)]
    corners = list(itertools.product(*zip(lower, upper, strict=True)))
    # Views are above the target; sample the complete exposed upper face. Every
    # sample must have clear conservative sight lines, not just the center point.
    targets = [(x, y, upper[2]) for x, y in itertools.product(
        (lower[0], center[0], upper[0]), (lower[1], center[1], upper[1]))]
    clearance = max(0.01, extent * 0.1)
    candidates = _offset_inventory(extent)
    accepted: list[dict[str, Any]] = []
    for row in candidates:
        position = [round(center[i] + row["position_offset_m"][i], 9) for i in range(3)]
        reasons: list[str] = []
        camera_room = index.structure.room_index_of_point(tuple(position[:2]))
        if camera_room != room:
            reasons.append("camera_outside_target_room")
        if not index.floor_z + clearance < position[2] < index.floor_z + index.structure.wall_height_m - clearance:
            reasons.append("camera_outside_vertical_room_bounds")
        containing = [box.id for box in obstacles if all(
            box.bbox_min[i] - clearance <= position[i] <= box.bbox_max[i] + clearance for i in range(3))]
        if containing:
            reasons.append("camera_inside_obstacle_clearance")
        if not _fits_frame(position, center, corners):
            reasons.append("target_bounds_outside_calibrated_frame")
        # Do not waste ray tests on poses already rejected by containment.
        occluders = sorted({box.id for point in targets for box in obstacles
                            if _segment_hits(position, point, box.bbox_min, box.bbox_max)}) if not reasons else []
        if occluders:
            reasons.append("target_sight_line_intersects_observed_bounds")
        row.update({"position_world_m": position, "camera_room_index": camera_room,
                    "status": "rejected" if reasons else "screened_candidate",
                    "rejection_reasons": reasons, "containing_obstacle_ids": containing,
                    "occluding_obstacle_ids": occluders})
        if not reasons:
            accepted.append(row)
    require(len(accepted) >= 16, "sam31_camera_geometry_insufficient_clear_candidates")
    # Deterministic farthest-point sampling retains translation/elevation diversity
    # within the actually clear sector rather than forcing a full 360-degree orbit.
    def radius(row: dict[str, Any]) -> float:
        return round(math.sqrt(sum(v * v for v in row["position_offset_m"])), 3)

    require(len({round(r["position_offset_m"][2], 3) for r in accepted}) >= 3
            and len({radius(r) for r in accepted}) >= 3,
            "sam31_camera_geometry_translation_baselines_insufficient")
    selected = [min(accepted, key=lambda r: (sum(v * v for v in r["position_offset_m"]), r["candidate_id"]))]
    remaining = [r for r in accepted if r is not selected[0]]
    while len(selected) < min(32, len(accepted)):
        heights = {round(r["position_offset_m"][2], 3) for r in selected}
        radii = {radius(r) for r in selected}
        best = max(remaining, key=lambda r: (
            len(heights) < 3 and round(r["position_offset_m"][2], 3) not in heights,
            len(radii) < 3 and radius(r) not in radii,
            min(sum((r["position_offset_m"][i] - s["position_offset_m"][i]) ** 2
                    for i in range(3)) for s in selected),
            -int(r["candidate_id"].split("-")[-1])))
        selected.append(best)
        remaining.remove(best)
    def view(row: dict[str, Any], camera_id: str) -> dict[str, Any]:
        return {"camera_id": camera_id, "position_offset_m": row["position_offset_m"],
                "target_offset_m": [0.0, 0.0, 0.0]}
    screen = {"schema_version": SCREEN_SCHEMA, "generator": GENERATOR, "source_files": sources,
              "target_instance_id": str(target_instance_id), "target_room_index": room,
              "target_bounds_min_m": lower, "target_bounds_max_m": upper,
              "camera_clearance_m": clearance, "vertical_fov_deg": 55.0,
              "visibility_sample_count": len(targets), "candidates": candidates,
              "selected_candidate_ids": [r["candidate_id"] for r in selected[:16]],
              "replacement_candidate_ids": [r["candidate_id"] for r in selected[16:]],
              "claim_boundary": {"appearance_fidelity_qualified": False,
                                 "rendered_visibility_qualified": False,
                                 "camera_calibration_qualified": False,
                                 "candidate_policy_queried": False},
              "screen_digest": ""}
    screen["screen_digest"] = canonical_digest(screen, digest_field="screen_digest")
    return {"generator": "translated_target_coverage_v1", "orbit_only_forbidden": True,
            "views": [view(r, f"source-{i + 1:02d}") for i, r in enumerate(selected[:16])],
            "replacement_views": [view(r, f"reserve-{i + 1:02d}") for i, r in enumerate(selected[16:])],
            "geometry_screen": screen}
