"""InteriorGS labeled-splat scene backend for the dynamic robot-placement pipeline.

InteriorGS scenes (a 3DGS ``.ply`` + sidecar ``labels.json`` / ``structure.json``)
carry ground-truth object annotations, so — unlike an unlabeled splat capture — we
do NOT need the SAM3/DA3 perception path to enumerate objects. This module turns
those sidecars into the same :class:`SceneObject` catalog the USD backend produces,
so the entire downstream chain (task→target resolution, stance solving, placement
validation, camera framing) runs unchanged on a splat scene:

* ``labels.json`` — a list of ``{ins_id, label, bounding_box: [8 corners]}``
  instances. Each becomes one :class:`SceneObject` with a canonical world AABB.
* ``structure.json`` — room floor polygons, wall segments (with thickness/height),
  and door/window holes. Walls become obstacle boxes; room polygons give a
  point-in-room test so a stance can be constrained to the target's room (a splat
  has no collision shell, so without this the probe would happily step through a
  wall into the neighboring room).

The perception backend (:mod:`perception_index`) remains the fallback for splat
scenes WITHOUT label sidecars. Geometry convention matches the rest of the
package: world coordinates, ``z`` up, floor near ``z = floor_z``.

Everything here is stdlib-only and hermetic: no numpy, no pxr, no network.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .robot_profile import RobotProfile
from .types import Probe, SceneObject, Vec3

INTERIORGS_LABELS_SOURCE = "interiorgs_labels"
INTERIORGS_STRUCTURE_SOURCE = "interiorgs_structure"

# An obstacle only blocks WHERE THE ROBOT STANDS if it actually sticks up out of
# the floor. Anything whose top is below this ankle band (carpets, rugs, floor
# mats, paint-thin decals) is walk-over-able and must not block placement.
DEFAULT_ANKLE_CLEARANCE_M = 0.06

# Ordered from the user's preregistered preferred task form to progressively
# less self-contained open/close assemblies.  These are discovery semantics,
# not inferred joints: an InteriorGS label never establishes a moving link,
# handle, axis, limits, collider separation, or simulator articulation.
ARTICULATED_OPEN_CLOSE_SEMANTICS: tuple[tuple[str, str], ...] = (
    ("drawer", "explicit_moving_link"),
    ("oven", "appliance_assembly"),
    ("dishwasher", "appliance_assembly"),
    ("microwave_oven", "appliance_assembly"),
    ("microwave", "appliance_assembly"),
    ("refrigerator", "appliance_assembly"),
    ("fridge", "appliance_assembly"),
    ("door", "explicit_moving_link"),
)

ARTICULATED_AGGREGATE_SEMANTICS: frozenset[str] = frozenset(
    {
        "basin_cabinet",
        "cabinet",
        "cupboard",
        "display_cabinet",
        "laundry_cabinets",
        "mirror_cabinet",
        "shoe_cabinet",
        "tv_cabinet",
        "wall_cabinet",
        "wardrobe",
        "wine_cabinet",
    }
)


# ----------------------------- labels.json -----------------------------

def _corners_to_aabb(corners: Sequence[dict]) -> Tuple[Vec3, Vec3]:
    """Canonical AABB (min <= max on every axis) from the 8-corner label box."""
    xs = [float(c["x"]) for c in corners]
    ys = [float(c["y"]) for c in corners]
    zs = [float(c["z"]) for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _normalized_box_corners(corners: Sequence[dict]) -> List[List[float]]:
    """Return the source OBB corners as finite ``[x, y, z]`` meter rows."""

    if len(corners) != 8:
        raise ValueError("InteriorGS oriented boxes require exactly eight corners")
    normalized = [
        [float(corner["x"]), float(corner["y"]), float(corner["z"])]
        for corner in corners
    ]
    if not all(math.isfinite(value) for corner in normalized for value in corner):
        raise ValueError("InteriorGS oriented box contains a non-finite coordinate")
    return normalized


def _normalize_label(label: str) -> str:
    return "_".join(str(label or "").strip().lower().split())


def load_interiorgs_labels(path: str | Path) -> List[SceneObject]:
    """Read an InteriorGS ``labels.json`` into a :class:`SceneObject` catalog.

    Object ``id`` is the raw ``ins_id`` string (matches the ``<label>_<ins_id>``
    instance names InteriorGS task prompts use, e.g. ``pot_88`` -> id ``"88"``).
    Instances with a missing/degenerate bounding box are skipped rather than
    emitting a NaN/inverted AABB into the placement math.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"interiorgs labels file is not a list: {path}")
    objects: List[SceneObject] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        corners = entry.get("bounding_box")
        ins_id = str(entry.get("ins_id", "")).strip()
        label = str(entry.get("label", "")).strip()
        if not ins_id or not isinstance(corners, list) or len(corners) != 8:
            continue
        try:
            normalized_corners = _normalized_box_corners(corners)
            bbox_min, bbox_max = _corners_to_aabb(corners)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (*bbox_min, *bbox_max)):
            continue
        centroid = (
            0.5 * (bbox_min[0] + bbox_max[0]),
            0.5 * (bbox_min[1] + bbox_max[1]),
            0.5 * (bbox_min[2] + bbox_max[2]),
        )
        norm = _normalize_label(label)
        objects.append(
            SceneObject(
                id=ins_id,
                label=norm or ins_id,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                centroid=centroid,
                category="",
                source=INTERIORGS_LABELS_SOURCE,
                confidence=1.0,
                extra={
                    "ins_id": ins_id,
                    "raw_label": label,
                    "instance_name": f"{norm}_{ins_id}" if norm else ins_id,
                    "oriented_bounding_box": {
                        "corners_world_m": normalized_corners,
                        "coordinate_frame": "interiorgs_world_right_back_up",
                        "units": "meters",
                        "source": "dataset_author_sidecar",
                    },
                    "placement_bounds_kind": "conservative_world_aabb",
                },
            )
        )
    return objects


def inventory_articulated_open_close_candidates(
    objects: Sequence[SceneObject],
) -> dict[str, Any]:
    """Inventory label-derived open/close candidates without inventing a joint.

    The result deliberately separates an explicit leaf or appliance assembly
    label from aggregate storage labels.  A cabinet label may contain several
    doors or drawers, so it cannot enter the target-closeup queue until a
    separately observed moving member is bound.  Likewise, an ``oven`` label is
    only an assembly candidate; visual and collision evidence must still prove
    a fixed link, moving link, handle/contact region, and one admissible joint.
    """

    semantic_priority = {
        semantic: (index, candidate_kind)
        for index, (semantic, candidate_kind) in enumerate(
            ARTICULATED_OPEN_CLOSE_SEMANTICS
        )
    }
    candidates: list[dict[str, Any]] = []
    aggregate_only: list[dict[str, Any]] = []
    for item in objects:
        semantic = _normalize_label(item.extra.get("raw_label", item.label))
        size = [round(float(value), 9) for value in item.size()]
        row = {
            "ins_id": item.id,
            "semantic_label": semantic,
            "raw_label": item.extra.get("raw_label", item.label),
            "centroid_world_m": [round(float(value), 9) for value in item.centroid],
            "aabb_size_m": size,
            "oriented_bounding_box": item.extra.get("oriented_bounding_box"),
        }
        if semantic in semantic_priority:
            priority, candidate_kind = semantic_priority[semantic]
            candidates.append(
                {
                    **row,
                    "semantic_priority": priority,
                    "candidate_kind": candidate_kind,
                    "closeup_admission": "pending_observed_link_and_interface_evidence",
                    "articulation_qualified": False,
                }
            )
        elif semantic in ARTICULATED_AGGREGATE_SEMANTICS:
            aggregate_only.append(
                {
                    **row,
                    "closeup_admission": "blocked_aggregate_label_has_no_separate_moving_member",
                    "articulation_qualified": False,
                }
            )
    candidates.sort(
        key=lambda row: (
            int(row["semantic_priority"]),
            len(str(row["ins_id"])),
            str(row["ins_id"]),
        )
    )
    aggregate_only.sort(
        key=lambda row: (
            str(row["semantic_label"]),
            len(str(row["ins_id"])),
            str(row["ins_id"]),
        )
    )
    return {
        "schema_version": "interiorgs_articulated_open_close_inventory.v1",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "aggregate_only_count": len(aggregate_only),
        "aggregate_only": aggregate_only,
        "selection_authority": "publisher_semantic_labels_and_bounds_only",
        "claim_boundary": {
            "joint_or_articulation_inferred": False,
            "handle_or_contact_region_observed": False,
            "fixed_and_moving_links_separated": False,
            "collision_identity_established": False,
            "reachability_established": False,
            "task_selected": False,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_interiorgs_object_index(
    labels_path: str | Path,
    *,
    splat_path: str | Path,
    structure_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize InteriorGS author sidecars into deterministic ``object_index.v2``.

    The eight source corners are retained exactly for semantic inspection.  The
    existing world AABB remains alongside them because robot-placement filtering
    intentionally uses a conservative axis-aligned bound.  Neither representation
    is promoted to customer-capture, collision, physics, or physical authority.
    """

    labels = Path(labels_path).expanduser().resolve()
    splat = Path(splat_path).expanduser().resolve()
    structure = Path(structure_path).expanduser().resolve() if structure_path else None
    for source in (labels, splat, structure):
        if source is not None and (not source.is_file() or source.stat().st_size <= 0):
            raise ValueError(f"InteriorGS source file missing or empty: {source}")

    objects = sorted(load_interiorgs_labels(labels), key=lambda item: (item.label, item.id))
    rows: List[dict[str, Any]] = []
    for item in objects:
        oriented = dict(item.extra.get("oriented_bounding_box") or {})
        rows.append(
            {
                "id": item.id,
                "label": item.label,
                "boundingBox": {
                    "center": [float(value) for value in item.centroid],
                    "extents": [float(value) for value in item.size()],
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
                    "kind": "conservative_world_aabb",
                },
                "orientedBoundingBox": oriented,
                "mean_confidence": 1.0,
                "n_total_detections": 0,
                "n_frame_detections": 0,
                "evidence_frames": [],
                "source_prompts": [],
                "provenance": {
                    "source": INTERIORGS_LABELS_SOURCE,
                    "annotation_authority": "dataset_author_sidecar",
                    "raw_customer_capture_authority": False,
                    "model_inferred_from_customer_capture": False,
                },
            }
        )

    source_files: dict[str, Any] = {
        "labels": {"sha256": _sha256_file(labels), "size_bytes": labels.stat().st_size},
        "splat": {"sha256": _sha256_file(splat), "size_bytes": splat.stat().st_size},
    }
    structure_summary: dict[str, Any] | None = None
    if structure is not None:
        parsed_structure = load_interiorgs_structure(structure)
        source_files["structure"] = {
            "sha256": _sha256_file(structure),
            "size_bytes": structure.stat().st_size,
        }
        structure_summary = {
            "room_count": len(parsed_structure.rooms),
            "wall_count": len(parsed_structure.wall_boxes),
            "hole_count": len(parsed_structure.holes),
            "source_digest": source_files["structure"]["sha256"],
        }

    return {
        "schema_version": "object_index.v2",
        "objects": rows,
        "coordinate_frame": {
            "name": "interiorgs_world_right_back_up",
            "axes": {"x": "right", "y": "back", "z": "up"},
            "handedness": "right_handed",
            "units": "meters",
        },
        "scene_structure": structure_summary,
        "provenance": {
            "source_profile": "precomputed_external_reconstruction",
            "dataset_profile": "interiorgs.v2",
            "source_files": source_files,
            "deterministic_order": "normalized_label_then_instance_id",
        },
        "claim_boundary": {
            "dataset_annotations_are_not_customer_capture_observations": True,
            "raw_capture_authority": False,
            "metric_authority_requires_separate_source_and_transform_validation": True,
            "collision_or_physics_authority": False,
            "physical_task_success": False,
            "deployment_readiness": False,
            "safety_certification": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }


# ----------------------------- structure.json -----------------------------

@dataclass
class InteriorGSStructure:
    """Parsed room/wall/hole geometry from an InteriorGS ``structure.json``."""

    rooms: List[List[Tuple[float, float]]] = field(default_factory=list)
    wall_boxes: List[SceneObject] = field(default_factory=list)
    holes: List[dict] = field(default_factory=list)
    wall_height_m: float = 2.6

    def room_index_of_point(self, xy: Tuple[float, float]) -> Optional[int]:
        """Index of the room polygon containing ``xy`` (None = in a wall band/outside)."""
        for idx, poly in enumerate(self.rooms):
            if point_in_polygon(xy, poly):
                return idx
        return None


def point_in_polygon(pt: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    """Even-odd ray-casting containment test (boundary points count as inside-ish).

    Room profiles are simple rectilinear polygons; the standard crossing test is
    exact enough for a stance-in-room gate (the footprint clearance margins dwarf
    any boundary ambiguity).
    """
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _wall_segment_to_box(
    location: Sequence[Sequence[float]],
    thickness: float,
    height: float,
    *,
    floor_z: float,
    wall_id: str,
) -> Optional[SceneObject]:
    """One wall segment -> an obstacle AABB.

    Segments in InteriorGS are wall CENTERLINES. Axis-aligned segments (the
    overwhelmingly common case) get the exact box: half the thickness on each
    side of the centerline, and the ends extended by half the thickness so
    corners seal. A skew segment falls back to the segment's enclosing AABB
    inflated by half the thickness — conservative (over-blocks the corner
    triangles) but never lets the probe step through a wall.
    """
    try:
        (x1, y1), (x2, y2) = (float(location[0][0]), float(location[0][1])), (
            float(location[1][0]),
            float(location[1][1]),
        )
        half_t = 0.5 * float(thickness)
        height = float(height)
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, half_t, height)):
        return None
    axis_aligned = abs(x1 - x2) < 1e-6 or abs(y1 - y2) < 1e-6
    bbox_min = (min(x1, x2) - half_t, min(y1, y2) - half_t, floor_z)
    bbox_max = (max(x1, x2) + half_t, max(y1, y2) + half_t, floor_z + height)
    centroid = (
        0.5 * (bbox_min[0] + bbox_max[0]),
        0.5 * (bbox_min[1] + bbox_max[1]),
        0.5 * (bbox_min[2] + bbox_max[2]),
    )
    return SceneObject(
        id=wall_id,
        label="wall",
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        centroid=centroid,
        category="structural",
        source=INTERIORGS_STRUCTURE_SOURCE,
        confidence=1.0,
        extra={"axis_aligned": axis_aligned, "thickness_m": float(thickness)},
    )


def load_interiorgs_structure(
    path: str | Path,
    *,
    floor_z: float = 0.0,
) -> InteriorGSStructure:
    """Read an InteriorGS ``structure.json`` (rooms + walls + holes)."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"interiorgs structure file is not an object: {path}")
    rooms: List[List[Tuple[float, float]]] = []
    for room in payload.get("rooms") or []:
        profile = room.get("profile") if isinstance(room, dict) else None
        if not isinstance(profile, list) or len(profile) < 3:
            continue
        try:
            poly = [(float(p[0]), float(p[1])) for p in profile]
        except (IndexError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for xy in poly for v in xy):
            rooms.append(poly)
    wall_boxes: List[SceneObject] = []
    heights: List[float] = []
    for idx, wall in enumerate(payload.get("walls") or []):
        if not isinstance(wall, dict):
            continue
        box = _wall_segment_to_box(
            wall.get("location") or [],
            wall.get("thickness", 0.1),
            wall.get("height", 2.6),
            floor_z=floor_z,
            wall_id=f"wall_{idx}",
        )
        if box is not None:
            wall_boxes.append(box)
            heights.append(float(wall.get("height", 2.6)))
    holes = [h for h in (payload.get("holes") or []) if isinstance(h, dict)]
    wall_height = max(heights) if heights else 2.6
    return InteriorGSStructure(
        rooms=rooms, wall_boxes=wall_boxes, holes=holes, wall_height_m=wall_height
    )


def estimate_labels_floor_z(
    objects: Sequence[SceneObject],
    *,
    cluster_band_m: float = 0.25,
    snap_zero_tol_m: float = 0.02,
) -> float:
    """Floor height from the label boxes: the MEDIAN of the floor-contact cluster.

    The raw minimum over box bottoms is one outlier away from wrong (a single
    label dipping 8 cm below the floor shrinks the walk-over ankle band and
    turns every carpet into a phantom placement blocker). Most floor-standing
    furniture bottoms pile up at the true floor plane, so we take the median of
    all bottoms within ``cluster_band_m`` of the minimum, and snap to exactly
    0.0 when within ``snap_zero_tol_m`` (InteriorGS exports put the floor at
    z=0 with rounding noise).
    """
    bottoms = sorted(obj.min_z() for obj in objects)
    if not bottoms:
        return 0.0
    lowest = bottoms[0]
    cluster = [b for b in bottoms if b <= lowest + cluster_band_m]
    mid = cluster[len(cluster) // 2] if len(cluster) % 2 else 0.5 * (
        cluster[len(cluster) // 2 - 1] + cluster[len(cluster) // 2]
    )
    return 0.0 if abs(mid) < snap_zero_tol_m else float(mid)


# ----------------------------- the spatial index -----------------------------

class InteriorGSSceneSpatialIndex:
    """:class:`SceneSpatialIndex` over InteriorGS label + structure sidecars.

    ``objects()`` returns every labeled instance (doors/windows included — they are
    open/close task targets). ``obstacle_boxes()`` additionally appends the wall
    boxes derived from ``structure.json`` so validation sees the room envelope a
    splat scene otherwise lacks.

    ``floor_z`` defaults to the lowest labeled-box bottom clamped toward the
    structure's implicit floor plane (InteriorGS exports put the floor at z=0 with
    tiny negative rounding noise); pass it explicitly to override.
    """

    def __init__(
        self,
        labels_path: str | Path,
        structure_path: str | Path | None = None,
        *,
        floor_z: float | None = None,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.structure_path = Path(structure_path) if structure_path else None
        self._objects = load_interiorgs_labels(self.labels_path)
        if floor_z is None:
            floor_z = estimate_labels_floor_z(self._objects)
        self.floor_z = float(floor_z)
        self.structure: InteriorGSStructure | None = None
        if self.structure_path is not None:
            self.structure = load_interiorgs_structure(
                self.structure_path, floor_z=self.floor_z
            )

    def objects(self) -> List[SceneObject]:
        return list(self._objects)

    def obstacle_boxes(self) -> List[SceneObject]:
        boxes = list(self._objects)
        if self.structure is not None:
            boxes.extend(self.structure.wall_boxes)
        return boxes

    def scene_bounds(self) -> Tuple[Vec3, Vec3] | None:
        boxes = self.obstacle_boxes()
        if not boxes:
            return None
        mins = tuple(min(b.bbox_min[i] for b in boxes) for i in range(3))
        maxs = tuple(max(b.bbox_max[i] for b in boxes) for i in range(3))
        return mins, maxs  # type: ignore[return-value]

    def object_by_instance(self, ins_id: str) -> Optional[SceneObject]:
        ins_id = str(ins_id).strip()
        for obj in self._objects:
            if obj.id == ins_id:
                return obj
        return None


# ----------------------------- probe + fixtures -----------------------------

def _yawed_footprint_half_extent(
    half_extent_xy: Tuple[float, float], yaw: float
) -> Tuple[float, float]:
    """Axis-aligned half extents enclosing the yaw-rotated footprint rectangle."""
    hx, hy = float(half_extent_xy[0]), float(half_extent_xy[1])
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (c * hx + s * hy, s * hx + c * hy)


def _xy_overlaps(
    a_min: Tuple[float, float],
    a_max: Tuple[float, float],
    b_min: Tuple[float, float],
    b_max: Tuple[float, float],
) -> bool:
    return (
        a_min[0] < b_max[0]
        and a_max[0] > b_min[0]
        and a_min[1] < b_max[1]
        and a_max[1] > b_min[1]
    )


def build_interiorgs_probe(
    index: InteriorGSSceneSpatialIndex,
    *,
    target: SceneObject | None = None,
    robot_profile: RobotProfile | None = None,
    footprint_half_extent_xy: Tuple[float, float] | None = None,
    foot_clearance: float | None = None,
    ankle_clearance: float = DEFAULT_ANKLE_CLEARANCE_M,
    clearance_margin: float | None = None,
    require_room: bool = True,
    standoff_obstacles: Sequence[SceneObject] | None = None,
    min_standoff_gap: float | None = None,
    obstacle_boxes: Sequence[SceneObject] | None = None,
    region_of=None,
) -> Probe:
    """Floor-occupancy probe over the labeled boxes + structure walls.

    Mirrors the validator's floor-occupancy model (an obstacle blocks a stance iff
    it reaches the floor band under the footprint) with two splat-specific twists:

    * WALK-OVER RULE — anything whose top is below ``ankle_clearance`` (carpets,
      rugs, mats) never blocks, even though it touches the floor. Height-driven,
      not label-driven, so unlabeled floor decals behave too.
    * SAME-ROOM RULE — a splat has no collision shell, so probing outward from a
      target near a wall would otherwise step THROUGH the wall and report clear
      floor in the neighboring room. When ``require_room`` and the structure
      resolves the target's room, candidates outside that room polygon (or inside
      a wall band, where no room contains the point) count as blocked.
    * STANDOFF-FLOOR RULE — ``standoff_obstacles`` (the target's supporting
      fixtures, e.g. the sideboard a pot sits on) are reach surfaces the
      validator later measures the standoff against, so a candidate hugging one
      laterally (gap below ``min_standoff_gap``) would pass a bare clip test but
      fail validation; reject it here so the solver keeps searching.

    The returned callable follows the package :data:`Probe` contract
    (``(pose, yaw) -> hit_count``, 0 == clear).
    """
    if footprint_half_extent_xy is None:
        fp = (
            robot_profile.footprint_half_extent_xyz
            if robot_profile is not None
            else (0.28, 0.28, 0.62)
        )
        footprint_half_extent_xy = (float(fp[0]), float(fp[1]))
    if foot_clearance is None:
        foot_clearance = (
            robot_profile.foot_clearance_m if robot_profile is not None else 0.40
        )
    if clearance_margin is None:
        clearance_margin = (
            robot_profile.min_obstacle_clearance_m if robot_profile is not None else 0.08
        )
    if min_standoff_gap is None:
        min_standoff_gap = (
            float(robot_profile.standoff_range_m[0]) if robot_profile is not None else 0.4
        )
    floor_z = index.floor_z
    floor_ceiling = floor_z + float(foot_clearance)
    walk_over_top = floor_z + float(ankle_clearance)
    target_id = target.id if target is not None else None
    fixtures = list(standoff_obstacles or ())

    # ``obstacle_boxes`` lets callers substitute a refined catalog (e.g. jumbo
    # label AABBs carved into splat-occupancy columns) without rebuilding the index.
    candidate_boxes = (
        list(obstacle_boxes) if obstacle_boxes is not None else index.obstacle_boxes()
    )
    blockers: List[SceneObject] = []
    for obs in candidate_boxes:
        if target_id is not None and obs.id == target_id and obs.source == INTERIORGS_LABELS_SOURCE:
            continue  # standing is judged against everything EXCEPT the target itself
        if obs.min_z() >= floor_ceiling:
            continue  # overhead: wall cabinet, ceiling fixture — stand under, reach over
        if obs.max_z() <= walk_over_top:
            continue  # carpet/rug/mat: walk over it
        blockers.append(obs)

    # Region lookup priority: an explicit ``region_of`` callable (e.g. a splat
    # free-space component map for scenes with no structure.json) supersedes the
    # structure room polygons. Both answer the same question — "which contiguous
    # floor region is this point in?" — with None meaning blocked/occupied.
    structure = index.structure
    if region_of is None and structure is not None and structure.rooms:
        region_of = structure.room_index_of_point
    target_room: Optional[int] = None
    if require_room and region_of is not None and target is not None:
        target_room = region_of((float(target.centroid[0]), float(target.centroid[1])))

    def probe(pose: Vec3, yaw: float) -> int:
        px, py = float(pose[0]), float(pose[1])
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(float(yaw))):
            return 1
        hx, hy = _yawed_footprint_half_extent(footprint_half_extent_xy, float(yaw))
        hx += float(clearance_margin)
        hy += float(clearance_margin)
        f_min = (px - hx, py - hy)
        f_max = (px + hx, py + hy)
        hits = 0
        for obs in blockers:
            if _xy_overlaps(
                f_min, f_max,
                (obs.bbox_min[0], obs.bbox_min[1]),
                (obs.bbox_max[0], obs.bbox_max[1]),
            ):
                hits += 1
        if require_room and region_of is not None:
            room = region_of((px, py))
            if room is None:
                hits += 1  # inside a wall band / occupied cell / outside the plan
            elif target_room is not None and room != target_room:
                hits += 1  # would stand in a different floor region than the target
        for fixture in fixtures:
            gap_x = max(
                fixture.bbox_min[0] - f_max[0], f_min[0] - fixture.bbox_max[0], 0.0
            )
            gap_y = max(
                fixture.bbox_min[1] - f_max[1], f_min[1] - fixture.bbox_max[1], 0.0
            )
            gap = math.hypot(gap_x, gap_y)
            if 0.0 < gap < float(min_standoff_gap):
                hits += 1  # hugging a reach surface: validation would reject it
        return hits

    return probe


def supporting_fixtures_for(
    target: SceneObject,
    obstacles: Sequence[SceneObject],
    *,
    top_tolerance_m: float = 0.15,
) -> List[SceneObject]:
    """Fixtures the target sits ON/IN — the reach surfaces for the standoff check.

    A pot on a sideboard is only ~0.13 m wide, so the raw footprint→target gap
    exceeds the standoff ceiling even at the correct stance; the validator must
    measure the gap to the NEAREST of {target, its supporting fixture} (exactly
    how the USD kitchen flow passes the counter). A fixture qualifies when its
    xy box contains the target's footprint center and its top is within
    ``top_tolerance_m`` of the target's bottom.
    """
    cx, cy = target.footprint_center()
    out: List[SceneObject] = []
    for obs in obstacles:
        if obs.id == target.id and obs.source == target.source:
            continue
        if not (
            obs.bbox_min[0] <= cx <= obs.bbox_max[0]
            and obs.bbox_min[1] <= cy <= obs.bbox_max[1]
        ):
            continue
        if abs(obs.max_z() - target.min_z()) <= top_tolerance_m:
            out.append(obs)
    return out


__all__ = [
    "ARTICULATED_AGGREGATE_SEMANTICS",
    "ARTICULATED_OPEN_CLOSE_SEMANTICS",
    "DEFAULT_ANKLE_CLEARANCE_M",
    "INTERIORGS_LABELS_SOURCE",
    "INTERIORGS_STRUCTURE_SOURCE",
    "InteriorGSSceneSpatialIndex",
    "InteriorGSStructure",
    "build_interiorgs_probe",
    "estimate_labels_floor_z",
    "inventory_articulated_open_close_candidates",
    "load_interiorgs_labels",
    "load_interiorgs_structure",
    "point_in_polygon",
    "supporting_fixtures_for",
]
