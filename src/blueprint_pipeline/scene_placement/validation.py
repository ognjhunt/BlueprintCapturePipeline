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

     LIMITATION: this only isolates *position* bugs. A pure-ROTATION frame bug (the robot lands at the
     right xy but ``yaw`` is applied in a flipped/rotated frame) that stays under ``max_facing_error_deg``
     passes BOTH the intent and the actual checks, so it is NOT isolated here. The facing check only
     catches gross (> tolerance) heading errors; sub-threshold frame rotations are a known blind spot.

Pure + stdlib-only, so it unit-tests with synthetic boxes — no torch, no GPU, no network.

NOTE on obstacles: pass the object catalog from ``SceneSpatialIndex.objects()`` (which already
excludes the structural shell — floor/walls/ceiling). The clip test treats every floor-standing
catalog object as something the robot must NOT overlap; if you pass the raw floor/walls it will
(correctly, but unhelpfully) report the robot as overlapping the floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .types import SceneObject, StandPose, Vec3


@dataclass
class PlacementVerdict:
    """Structured result of validating a stance pose. ``ok`` is the AND of every check."""

    ok: bool
    failures: List[str] = field(default_factory=list)          # human-readable failed checks
    clipping: List[Tuple[str, float]] = field(default_factory=list)  # (obstacle_id, xy overlap m^2)
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


def validate_stand_pose(
    position: Vec3,
    yaw: float,
    target: SceneObject,
    obstacles: Sequence[SceneObject],
    floor_z: float,
    *,
    footprint_half_extent: Tuple[float, float, float] = (0.28, 0.28, 0.62),
    pelvis_height: float = 0.79,
    max_facing_error_deg: float = 35.0,
    standoff_range: Tuple[float, float] = (0.30, 1.30),
    floor_tol: float = 0.08,
    clip_area_eps: float = 1e-4,
    standoff_obstacles: Optional[Sequence[SceneObject]] = None,
) -> PlacementVerdict:
    """Validate that ``position``/``yaw`` stands the robot correctly for acting on ``target``.

    ``floor_z`` is REQUIRED (not defaulted): the validator works in the floor frame, and the pelvis
    target, the on-floor band, and the clip z-gate all hinge on it. A wrong ``floor_z`` silently
    corrupts every check (e.g. an overhead-cabinet clip gets mis-classified as "stood under" and the
    clip is missed), so the caller must state it explicitly rather than inherit a guessed 0.0.

    Four independent checks, all must pass for ``ok``:

    1. NO CLIP — the robot footprint box (``footprint_half_extent`` xy, centered at the pose) does not
       overlap any obstacle whose vertical AABB span intersects the robot's COLLISION z-interval
       ``[pelvis_z - hz, pelvis_z + hz]`` (``hz`` is the z half-extent of ``footprint_half_extent``).
       That interval models the whole body (feet→head), so an obstacle is ignored only when it clears
       the robot's head (``min_z > pelvis_z + hz``) or sits entirely below the feet — a wall/ceiling box
       high enough to stand *under* — and an upper cabinet / range hood whose box dips into the torso/head
       band IS reported. This is the test that catches the robot clipping a counter/cabinet *or* an
       overhead fixture.
    2. ON FLOOR — the pelvis z is within ``floor_tol`` of ``floor_z + pelvis_height`` (not floating/sunk).
    3. FACING — ``yaw`` points within ``max_facing_error_deg`` of the direction to the target centroid.
    4. STANDOFF — the xy gap between the footprint and the nearest reach surface lies within
       ``standoff_range`` (not standing inside the surface, not unreachably far). The reach surface is
       the target's box by default; pass ``standoff_obstacles`` (e.g. the counter/fixture the target is
       recessed in) to measure the gap to the NEAREST of the target and those fixtures, so a small target
       recessed behind a deep counter is judged reachable when the robot is at the counter's front edge.

    ``obstacles`` should be the shell-excluded object catalog (``SceneSpatialIndex.objects()``); the
    target itself may be present and is treated as a clip obstacle too (standing inside it = clip).

    All numeric inputs must be finite. A non-finite ``position``/``yaw``, or an obstacle/target with a
    non-finite AABB, yields ``ok=False`` with an explicit ``non_finite_*`` reason rather than a
    silently-passing verdict (every IEEE comparison against ``nan`` is ``False``, which would otherwise
    let a garbage pose slip through facing/clip/standoff).
    """
    px, py, pz = (float(v) for v in position)
    yaw = float(yaw)
    hx, hy, hz = (abs(float(v)) for v in footprint_half_extent)
    f_min = (px - hx, py - hy)
    f_max = (px + hx, py + hy)
    pelvis_z = floor_z + pelvis_height
    robot_top_z = pelvis_z + hz       # top of the collision box (head)
    robot_bottom_z = pelvis_z - hz    # bottom of the collision box (feet)
    failures: List[str] = []

    # 0. FINITE INPUTS — guard before any comparison, since `x > nan` etc. are all False (silent pass).
    if not _all_finite(px, py, pz, yaw, floor_z):
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
            ok=False, failures=failures, clipping=[],
            facing_error_deg=float("nan"), standoff_m=float("nan"),
            on_floor=False, notes="INVALID: " + "; ".join(failures),
        )

    # 1. CLIP — an obstacle clips iff its z-span overlaps the robot collision z-interval AND its xy
    #    footprint overlaps the robot footprint. Skip only obstacles wholly above the head or below
    #    the feet (`>`/`<` so an obstacle merely touching a boundary is excluded, not clipped).
    clipping: List[Tuple[str, float]] = []
    for obs in obstacles:
        if obs.min_z() > robot_top_z or obs.max_z() < robot_bottom_z:
            continue
        area = _xy_overlap_area(
            f_min, f_max,
            (obs.bbox_min[0], obs.bbox_min[1]), (obs.bbox_max[0], obs.bbox_max[1]),
        )
        if area > clip_area_eps:
            clipping.append((obs.id, round(area, 4)))
    if clipping:
        failures.append("clips:" + ",".join(oid for oid, _ in clipping))

    # 2. ON FLOOR
    on_floor = abs(pz - pelvis_z) <= floor_tol
    if not on_floor:
        failures.append(f"off_floor(z={pz:.2f},expected={pelvis_z:.2f})")

    # 3. FACING
    dir_to_target = math.atan2(target.centroid[1] - py, target.centroid[0] - px)
    facing_err = _angle_diff_deg(yaw, dir_to_target)
    if facing_err > max_facing_error_deg:
        failures.append(f"facing_off({facing_err:.0f}deg)")

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
        ok=ok, failures=failures, clipping=clipping,
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


__all__ = ["PlacementVerdict", "validate_stand_pose", "validate_placement"]
