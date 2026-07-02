"""Placement validation: does a stance pose actually stand the robot correctly?

Deterministic, geometry-only checks (no GPU, no render) that catch the placement bugs we kept
missing by eyeballing grainy frames: the robot footprint overlapping a counter/cabinet (clipping
*into* furniture), standing the wrong distance from the target, facing away from it, or floating
off the floor. Given a stance pose + the scene's object AABBs (the same ``UsdSceneSpatialIndex``
catalog used to place the robot), it returns a structured pass/fail verdict with concrete reasons.

Two intended uses — and the difference is the whole point:
  1. Validate the INTENDED pose (``compute_stand_pose`` output) BEFORE rendering — reject a pose that
     would clip or face wrong, instead of discovering it in a frame.
  2. Validate the ACTUAL placed pose (the robot's queried world position AFTER `_place_root`) — if the
     intent validates but the actual position does NOT, that isolates a TRANSLATION / position-application
     bug (the robot's xy/z didn't land where the pose said, so it now clips or fails standoff) vs. a
     bad-pose bug. That separation is exactly what we lacked when a pose read fine in the data but the
     robot rendered inside the counter.

     Rotation-frame bugs are checked when the caller supplies a non-zero
     ``forward_axis_yaw_offset_deg``. That keeps the default runner convention unchanged while making a
     flipped/rotated forward-axis convention fail deterministically before render.

Pure + stdlib-only, so it unit-tests with synthetic boxes — no torch, no GPU, no network.

NOTE on obstacles: for clipping, prefer fine obstacle boxes such as
``UsdSceneSpatialIndex.obstacle_boxes()``. The grouped object catalog from
``SceneSpatialIndex.objects()`` is right for target resolution, but a whole cabinet/counter assembly
can collapse into a broad AABB that covers open aisle floor. Still exclude floor/ground/ceiling from
the obstacle list; if you pass the raw floor, it will correctly but unhelpfully report the robot as
overlapping the floor.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .robot_profile import RobotProfile
from .types import SceneObject, StandPose, Vec3


def _resolve_profile_scalar(
    explicit: Optional[float],
    profile: Optional[RobotProfile],
    profile_field: str,
    default: float,
) -> float:
    """Explicit kwarg > robot profile field > historical default."""
    if explicit is not None:
        return float(explicit)
    if profile is not None:
        return float(getattr(profile, profile_field))
    return float(default)


PLACEMENT_VALIDATION_SCHEMA_VERSION = "placement_validation.v1"
DEFAULT_VALIDATION_FOOTPRINT_HALF_EXTENT: Tuple[float, float, float] = (0.28, 0.28, 0.62)
DEFAULT_VALIDATION_PELVIS_HEIGHT_M = 0.79
DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG = 30.0
DEFAULT_VALIDATION_STANDOFF_RANGE: Tuple[float, float] = (0.4, 1.2)
DEFAULT_VALIDATION_CLIP_AREA_EPS_M2 = 0.005
DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M = 0.08


def _yaw_rotated_aabb_half_extent(
    footprint_half_extent: Tuple[float, float, float],
    yaw: float,
) -> Tuple[float, float]:
    """World-axis half extent of a yawed local robot footprint.

    The local x half extent is robot depth/front-back, and local y is lateral width. Validation
    still uses cheap axis-aligned rectangle checks, but this keeps a narrow front-facing robot from
    being treated as equally deep and wide at a counter.
    """
    hx_local, hy_local, _hz = (abs(float(v)) for v in footprint_half_extent)
    c = abs(math.cos(float(yaw)))
    s = abs(math.sin(float(yaw)))
    return (
        c * hx_local + s * hy_local,
        s * hx_local + c * hy_local,
    )


@dataclass
class PlacementVerdict:
    """Structured result of validating a stance pose. ``ok`` is the AND of every check."""

    ok: bool
    failures: List[str] = field(default_factory=list)          # human-readable failed checks
    clipping: List[Tuple[str, float]] = field(default_factory=list)  # (obstacle_id, xy overlap m^2)
    near_clearance: List[Tuple[str, float]] = field(default_factory=list)  # (obstacle_id, xy gap m)
    outside_boundary: List[str] = field(default_factory=list)
    min_obstacle_clearance_m: float | None = None
    facing_error_deg: float = 0.0
    standoff_m: float = 0.0
    on_floor: bool = True
    notes: str = ""


def _xy_overlap_area(
    a_min: Tuple[float, float], a_max: Tuple[float, float],
    b_min: Tuple[float, float], b_max: Tuple[float, float],
) -> float:
    """Area of the xy-plane overlap of two axis-aligned rectangles (0 when disjoint)."""
    dx = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
    dy = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def _xy_box_gap(
    a_min: Tuple[float, float], a_max: Tuple[float, float],
    b_min: Tuple[float, float], b_max: Tuple[float, float],
) -> float:
    """Minimum xy distance between two axis-aligned rectangles (0 when they overlap/touch)."""
    dx = max(0.0, b_min[0] - a_max[0], a_min[0] - b_max[0])
    dy = max(0.0, b_min[1] - a_max[1], a_min[1] - b_max[1])
    return math.hypot(dx, dy)


def _angle_diff_deg(a: float, b: float) -> float:
    """Absolute smallest-arc difference between two headings (radians), in degrees [0, 180].

    Returns ``nan`` for a non-finite input (the caller must treat that as a failure, not a pass —
    a silent ``False`` comparison against ``nan`` would let a garbage heading slip through).
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(d))


def _all_finite(*values: float) -> bool:
    """True iff every scalar is a finite real (no nan, no inf)."""
    return all(math.isfinite(v) for v in values)


def _obj_bbox_finite(obj: SceneObject) -> bool:
    """True iff every coordinate of an object's AABB + centroid is finite."""
    return _all_finite(
        obj.bbox_min[0], obj.bbox_min[1], obj.bbox_min[2],
        obj.bbox_max[0], obj.bbox_max[1], obj.bbox_max[2],
        obj.centroid[0], obj.centroid[1], obj.centroid[2],
    )


def _is_structural_boundary_obstacle(obj: SceneObject) -> bool:
    """Walls/windows are room boundaries even when their mesh AABB starts above the foot band."""
    source = str(obj.source or "").lower()
    if not source.startswith("usd"):
        return False
    text = f"{obj.id} {obj.label}".lower()
    if "wall" in text or "wallcollider" in text:
        return True
    # Window leaf meshes are usually mullions/trim, not the room envelope. They can still
    # participate in clearance when close to the footprint, but only shell-derived windows
    # should decide that a pose is outside the room or across a boundary from the target.
    return "window" in text and source == "usd_shell"


def _crosses_structural_boundary_xy(
    *,
    pose_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    boundary: SceneObject,
    footprint_half_extent_xy: Tuple[float, float],
) -> bool:
    """True when the pose-to-target xy segment crosses a thin wall/window AABB.

    A side-of-plane test alone is too broad for authored kitchens: any decorative window
    segment with a thin AABB can sit between the pose x and target x while being nowhere near
    the actual line of approach. Requiring the line segment to pierce the AABB's long-span
    interval keeps real "wall/window side" rejections while not treating unrelated panes as
    global room separators.
    """
    px, py = pose_xy
    tx, ty = target_xy
    hx, hy = footprint_half_extent_xy
    min_x, min_y = float(boundary.bbox_min[0]), float(boundary.bbox_min[1])
    max_x, max_y = float(boundary.bbox_max[0]), float(boundary.bbox_max[1])
    size_x = abs(max_x - min_x)
    size_y = abs(max_y - min_y)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    eps = 1e-9
    if size_x <= min(size_y, 0.35):
        if (px - cx) * (tx - cx) >= 0.0 or abs(tx - px) <= eps:
            return False
        t = (cx - px) / (tx - px)
        if not 0.0 <= t <= 1.0:
            return False
        y_at_boundary = py + t * (ty - py)
        return (min_y - hy) <= y_at_boundary <= (max_y + hy)
    if size_y <= min(size_x, 0.35):
        if (py - cy) * (ty - cy) >= 0.0 or abs(ty - py) <= eps:
            return False
        t = (cy - py) / (ty - py)
        if not 0.0 <= t <= 1.0:
            return False
        x_at_boundary = px + t * (tx - px)
        return (min_x - hx) <= x_at_boundary <= (max_x + hx)
    return False


def validate_stand_pose(
    position: Vec3,
    yaw: float,
    target: SceneObject,
    obstacles: Sequence[SceneObject],
    floor_z: float,
    *,
    footprint_half_extent: Optional[Tuple[float, float, float]] = None,
    pelvis_height: Optional[float] = None,
    max_facing_error_deg: Optional[float] = None,
    standoff_range: Optional[Tuple[float, float]] = None,
    floor_tol: Optional[float] = None,
    foot_clearance: Optional[float] = None,
    clip_area_eps: float = DEFAULT_VALIDATION_CLIP_AREA_EPS_M2,
    min_obstacle_clearance_m: Optional[float] = None,
    standoff_obstacles: Optional[Sequence[SceneObject]] = None,
    forward_axis_yaw_offset_deg: float = 0.0,
    robot_profile: Optional[RobotProfile] = None,
) -> PlacementVerdict:
    """Validate that ``position``/``yaw`` stands the robot correctly for acting on ``target``.

    ``floor_z`` is REQUIRED (not defaulted): the validator works in the floor frame, and the pelvis
    target, the on-floor band, and the clip z-gate all hinge on it. A wrong ``floor_z`` silently
    corrupts every check (e.g. an overhead-cabinet clip gets mis-classified as "stood under" and the
    clip is missed), so the caller must state it explicitly rather than inherit a guessed 0.0.

    Four independent checks, all must pass for ``ok``:

    1. NO CLIP — where the robot may STAND is governed by what occupies the FLOOR under its footprint,
       not by what sits on the counter or wall above it. An obstacle blocks the stance iff its xy box
       overlaps the robot footprint AND it actually reaches the floor — ``min_z < floor_z + foot_clearance``
       (a cabinet base, a fridge, a chair, a floor-standing planter). Anything that starts ABOVE the foot
       band — a vase/plant or fruit bowl ON the counter, a wall cabinet, a range hood, draping foliage — is
       deliberately NOT a clip: the robot stands under / in front of it and reaches over, exactly as a
       person stands at a cluttered counter. Whole-body collision with overhead clutter belongs to the
       manipulation/motion-planning layer, not this standing-placement gate.
    2. ON FLOOR — the pelvis z is within ``floor_tol`` of ``floor_z + pelvis_height`` (not floating/sunk).
    3. FACING — ``yaw`` points within ``max_facing_error_deg`` of the direction to the target centroid.
    4. STANDOFF — the xy gap between the footprint and the nearest reach surface lies within
       ``standoff_range`` (not standing inside the surface, not unreachably far). The reach surface is
       the target's box by default; pass ``standoff_obstacles`` (e.g. the counter/fixture the target is
       recessed in) to measure the gap to the NEAREST of the target and those fixtures, so a small target
       recessed behind a deep counter is judged reachable when the robot is at the counter's front edge.

    ``clip_area_eps`` is a small AABB sliver tolerance for visual meshes split into thin component
    boxes. It prevents millimeter-scale face-box overlaps from rejecting an otherwise clear stance
    while still failing real footprint/object intersections. ``min_obstacle_clearance_m`` handles
    the mirror-image bug: a footprint can be technically non-overlapping but only millimeters from
    a base cabinet/wall AABB, which renders as clipping once the full G1 mesh and reconstruction
    noise are involved.

    ``obstacles`` should be the shell-excluded object catalog (``SceneSpatialIndex.objects()``); the
    target itself may be present and is treated as a clip obstacle too (standing inside it = clip).

    All numeric inputs must be finite. A non-finite ``position``/``yaw``, or an obstacle/target with a
    non-finite AABB, yields ``ok=False`` with an explicit ``non_finite_*`` reason rather than a
    silently-passing verdict (every IEEE comparison against ``nan`` is ``False``, which would otherwise
    let a garbage pose slip through facing/clip/standoff).

    ``forward_axis_yaw_offset_deg`` is a render-free convention cross-check. Keep it at ``0`` for the
    runner's normal +x-forward pelvis frame. Passing a non-zero offset models a caller applying yaw in a
    rotated/flipped frame; the validator then fails with ``forward_frame_mismatch`` if that applied
    forward axis would not look at the target.

    ``robot_profile`` supplies robot-specific defaults (footprint, pelvis height,
    tolerances) for any knob the caller leaves unset; an explicit kwarg always wins
    over the profile. With neither, the historical G1-scale defaults apply unchanged.
    """
    if footprint_half_extent is None:
        footprint_half_extent = (
            robot_profile.footprint_half_extent_xyz
            if robot_profile is not None
            else DEFAULT_VALIDATION_FOOTPRINT_HALF_EXTENT
        )
    if standoff_range is None:
        standoff_range = (
            robot_profile.standoff_range_m
            if robot_profile is not None
            else DEFAULT_VALIDATION_STANDOFF_RANGE
        )
    pelvis_height = _resolve_profile_scalar(
        pelvis_height, robot_profile, "pelvis_height_m", DEFAULT_VALIDATION_PELVIS_HEIGHT_M
    )
    max_facing_error_deg = _resolve_profile_scalar(
        max_facing_error_deg, robot_profile, "max_facing_error_deg",
        DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG,
    )
    floor_tol = _resolve_profile_scalar(floor_tol, robot_profile, "floor_tol_m", 0.08)
    foot_clearance = _resolve_profile_scalar(
        foot_clearance, robot_profile, "foot_clearance_m", 0.40
    )
    min_obstacle_clearance_m = _resolve_profile_scalar(
        min_obstacle_clearance_m, robot_profile, "min_obstacle_clearance_m",
        DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M,
    )
    px, py, pz = (float(v) for v in position)
    yaw = float(yaw)
    pelvis_z = floor_z + pelvis_height
    floor_obstacle_ceiling = floor_z + foot_clearance
    failures: List[str] = []

    # 0. FINITE INPUTS — guard before any comparison, since `x > nan` etc. are all False (silent pass).
    if not _all_finite(px, py, pz, yaw, floor_z, forward_axis_yaw_offset_deg):
        failures.append("non_finite_pose")
    if not _obj_bbox_finite(target):
        failures.append(f"non_finite_target:{target.id}")
    bad_boxes = [o.id for o in obstacles if not _obj_bbox_finite(o)]
    if standoff_obstacles:
        bad_boxes += [o.id for o in standoff_obstacles if not _obj_bbox_finite(o)]
    if bad_boxes:
        failures.append("non_finite_box:" + ",".join(bad_boxes))
    if failures:
        # Bail out early: with non-finite geometry the downstream metrics are meaningless and would
        # report misleading (nan/0.0) diagnostics. Fail loud with the concrete reason instead.
        return PlacementVerdict(
            ok=False, failures=failures, clipping=[], near_clearance=[], outside_boundary=[],
            min_obstacle_clearance_m=min_obstacle_clearance_m,
            facing_error_deg=float("nan"), standoff_m=float("nan"),
            on_floor=False, notes="INVALID: " + "; ".join(failures),
        )

    # The yawed footprint extent needs a finite yaw (cos/sin of inf raise), so it is computed
    # only after the finite-inputs guard above has had its chance to fail loud.
    hx, hy = _yaw_rotated_aabb_half_extent(footprint_half_extent, yaw)
    f_min = (px - hx, py - hy)
    f_max = (px + hx, py + hy)

    # 1. CLIP — floor-occupancy model: an obstacle blocks the stance iff it reaches the floor under
    #    the footprint. Skip anything above the foot band or entirely below the declared floor.
    clipping: List[Tuple[str, float]] = []
    near_clearance: List[Tuple[str, float]] = []
    outside_boundary: List[str] = []
    for obs in obstacles:
        structural_boundary = _is_structural_boundary_obstacle(obs)
        if (
            not structural_boundary
            and (obs.min_z() >= floor_obstacle_ceiling or obs.max_z() < floor_z)
        ):
            continue
        if structural_boundary:
            if _crosses_structural_boundary_xy(
                pose_xy=(px, py),
                target_xy=(float(target.centroid[0]), float(target.centroid[1])),
                boundary=obs,
                footprint_half_extent_xy=(hx, hy),
            ):
                outside_boundary.append(obs.id)
        area = _xy_overlap_area(
            f_min, f_max,
            (obs.bbox_min[0], obs.bbox_min[1]), (obs.bbox_max[0], obs.bbox_max[1]),
        )
        if area > clip_area_eps:
            clipping.append((obs.id, round(area, 4)))
            continue
        gap = _xy_box_gap(
            f_min, f_max,
            (obs.bbox_min[0], obs.bbox_min[1]), (obs.bbox_max[0], obs.bbox_max[1]),
        )
        if 0.0 < gap < min_obstacle_clearance_m:
            near_clearance.append((obs.id, round(gap, 4)))
    if clipping:
        failures.append("clips:" + ",".join(oid for oid, _ in clipping))
    if near_clearance:
        failures.append("clearance:" + ",".join(oid for oid, _ in near_clearance))
    if outside_boundary:
        failures.append("outside_boundary:" + ",".join(outside_boundary))

    # 2. ON FLOOR
    on_floor = abs(pz - pelvis_z) <= floor_tol
    if not on_floor:
        failures.append(f"off_floor(z={pz:.2f},expected={pelvis_z:.2f})")

    # 3. FACING
    dir_to_target = math.atan2(target.centroid[1] - py, target.centroid[0] - px)
    facing_err = _angle_diff_deg(yaw, dir_to_target)
    if facing_err > max_facing_error_deg:
        failures.append(f"facing_off({facing_err:.0f}deg)")
    applied_forward_err = _angle_diff_deg(
        yaw + math.radians(float(forward_axis_yaw_offset_deg)),
        dir_to_target,
    )
    if applied_forward_err > max_facing_error_deg:
        failures.append(f"forward_frame_mismatch({applied_forward_err:.0f}deg)")

    # 4. STANDOFF — xy gap to the nearest reach surface (target box, plus any supporting fixtures).
    standoff = _xy_box_gap(
        f_min, f_max,
        (target.bbox_min[0], target.bbox_min[1]), (target.bbox_max[0], target.bbox_max[1]),
    )
    for fixture in (standoff_obstacles or ()):
        standoff = min(standoff, _xy_box_gap(
            f_min, f_max,
            (fixture.bbox_min[0], fixture.bbox_min[1]), (fixture.bbox_max[0], fixture.bbox_max[1]),
        ))
    lo, hi = standoff_range
    if not (lo <= standoff <= hi):
        failures.append(f"standoff_out({standoff:.2f}m,range=[{lo},{hi}])")

    ok = not failures
    notes = "placement valid" if ok else "INVALID: " + "; ".join(failures)
    return PlacementVerdict(
        ok=ok, failures=failures, clipping=clipping, near_clearance=near_clearance,
        outside_boundary=outside_boundary,
        min_obstacle_clearance_m=round(float(min_obstacle_clearance_m), 6),
        facing_error_deg=round(facing_err, 1), standoff_m=round(standoff, 3),
        on_floor=on_floor, notes=notes,
    )


def validate_placement(
    stand_pose: StandPose,
    target: SceneObject,
    obstacles: Sequence[SceneObject],
    **kwargs,
) -> PlacementVerdict:
    """Convenience: validate a :class:`StandPose` (from ``compute_stand_pose``) against the scene.

    ``floor_z`` is required by :func:`validate_stand_pose`, so pass it through ``kwargs``
    (``validate_placement(sp, target, obstacles, floor_z=0.05)``).
    """
    return validate_stand_pose(stand_pose.position, stand_pose.yaw, target, obstacles, **kwargs)


def scene_object_to_dict(obj: SceneObject) -> dict[str, Any]:
    """Serialize a scene object AABB into the placement-validation artifact shape."""
    return {
        "id": obj.id,
        "label": obj.label,
        "bbox_min_xyz": [round(float(v), 6) for v in obj.bbox_min],
        "bbox_max_xyz": [round(float(v), 6) for v in obj.bbox_max],
        "centroid_xyz": [round(float(v), 6) for v in obj.centroid],
        "category": obj.category,
        "source": obj.source,
        "confidence": round(float(obj.confidence), 6),
        "extra": dict(obj.extra or {}),
    }


def placement_verdict_to_dict(verdict: PlacementVerdict) -> dict[str, Any]:
    """Serialize a :class:`PlacementVerdict` for stable JSON artifacts."""
    return {
        "ok": bool(verdict.ok),
        "failures": list(verdict.failures),
        "clipping": [
            {"object_id": oid, "overlap_area_xy_m2": round(float(area), 6)}
            for oid, area in verdict.clipping
        ],
        "near_clearance": [
            {"object_id": oid, "gap_m": round(float(gap), 6)}
            for oid, gap in verdict.near_clearance
        ],
        "outside_boundary": list(verdict.outside_boundary),
        "min_obstacle_clearance_m": (
            round(float(verdict.min_obstacle_clearance_m), 6)
            if verdict.min_obstacle_clearance_m is not None
            else None
        ),
        "facing_error_deg": round(float(verdict.facing_error_deg), 6)
        if math.isfinite(float(verdict.facing_error_deg))
        else str(verdict.facing_error_deg),
        "standoff_m": round(float(verdict.standoff_m), 6)
        if math.isfinite(float(verdict.standoff_m))
        else str(verdict.standoff_m),
        "on_floor": bool(verdict.on_floor),
        "notes": verdict.notes,
    }


def _footprint_box_dict(
    position: Vec3,
    footprint_half_extent: Tuple[float, float, float],
) -> dict[str, list[float]]:
    px, py, pz = (float(v) for v in position)
    hx, hy, hz = (abs(float(v)) for v in footprint_half_extent)
    return {
        "bbox_min_xyz": [round(px - hx, 6), round(py - hy, 6), round(pz - hz, 6)],
        "bbox_max_xyz": [round(px + hx, 6), round(py + hy, 6), round(pz + hz, 6)],
        "center_xyz": [round(px, 6), round(py, 6), round(pz, 6)],
    }


def build_placement_validation_report(
    *,
    position: Vec3,
    yaw: float,
    target: SceneObject,
    scene_objects: Sequence[SceneObject],
    floor_z: float,
    footprint_half_extent: Tuple[float, float, float] = DEFAULT_VALIDATION_FOOTPRINT_HALF_EXTENT,
    pelvis_height: float = DEFAULT_VALIDATION_PELVIS_HEIGHT_M,
    max_facing_error_deg: float = DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG,
    standoff_range: Tuple[float, float] = DEFAULT_VALIDATION_STANDOFF_RANGE,
    floor_tol: float = 0.08,
    min_obstacle_clearance_m: float = DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M,
    standoff_obstacles: Optional[Sequence[SceneObject]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the hermetic placement-validation JSON payload.

    This is the artifact contract for deterministic, no-GPU placement proof: the robot footprint
    box at the stance pose must not overlap scene-object AABBs in xy, the pelvis/root z must be on
    the declared floor frame, yaw must face the target, and the standoff must be reachable. ``PASS``
    is emitted only when every deterministic check passes.
    """
    verdict = validate_stand_pose(
        position,
        yaw,
        target,
        scene_objects,
        floor_z,
        footprint_half_extent=footprint_half_extent,
        pelvis_height=pelvis_height,
        max_facing_error_deg=max_facing_error_deg,
        standoff_range=standoff_range,
        floor_tol=floor_tol,
        min_obstacle_clearance_m=min_obstacle_clearance_m,
        standoff_obstacles=standoff_obstacles,
    )
    payload: dict[str, Any] = {
        "schema_version": PLACEMENT_VALIDATION_SCHEMA_VERSION,
        "status": "PASS" if verdict.ok else "FAIL",
        "deterministic_geometry": placement_verdict_to_dict(verdict),
        "stance_pose": {
            "position_xyz": [round(float(v), 6) for v in position],
            "yaw_rad": round(float(yaw), 6),
        },
        "floor_z": round(float(floor_z), 6),
        "expected_pelvis_z": round(float(floor_z) + float(pelvis_height), 6),
        "robot_footprint_half_extent_xyz": [round(float(v), 6) for v in footprint_half_extent],
        "robot_footprint_box_at_pose": _footprint_box_dict(position, footprint_half_extent),
        "max_facing_error_deg": round(float(max_facing_error_deg), 6),
        "standoff_range_m": [round(float(standoff_range[0]), 6), round(float(standoff_range[1]), 6)],
        "min_obstacle_clearance_m": round(float(min_obstacle_clearance_m), 6),
        "target_object": scene_object_to_dict(target),
        "scene_object_count": len(scene_objects),
        "scene_objects": [scene_object_to_dict(obj) for obj in scene_objects],
        "claim_boundary": (
            "Deterministic scene-AABB placement validation only. This does not prove dynamic "
            "locomotion, manipulation success, safety validation, or physical robot readiness."
        ),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def write_placement_validation_report(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Build and write ``placement_validation.json``; return the written payload."""
    payload = build_placement_validation_report(**kwargs)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_VALIDATION_FOOTPRINT_HALF_EXTENT",
    "DEFAULT_VALIDATION_CLIP_AREA_EPS_M2",
    "DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M",
    "DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG",
    "DEFAULT_VALIDATION_PELVIS_HEIGHT_M",
    "DEFAULT_VALIDATION_STANDOFF_RANGE",
    "PLACEMENT_VALIDATION_SCHEMA_VERSION",
    "PlacementVerdict",
    "build_placement_validation_report",
    "placement_verdict_to_dict",
    "scene_object_to_dict",
    "validate_stand_pose",
    "validate_placement",
    "write_placement_validation_report",
]
