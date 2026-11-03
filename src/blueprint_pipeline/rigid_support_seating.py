"""Where a rigid task object rests on the scene's own collider, at its start and at its goal (CPU, pure reads).

Beta journey step: L5 team policy evaluation. The scorer reads an episode only
when its first sample lies within 2 mm of the task's ``start_pose_world``, and
counts a placement only inside the success region's height window. Both
heights were taken from the registered support plane, whose top is the highest
face of the whole support (capture-447ae4f4: 0.8166 m over a 1.3 m oval table
whose fitted top under the tray was 0.8019 m). Isaac spawns the object at its
root pose and physics rests it on the collision mesh, so the start the scorer
expected was neither where the object was spawned (14.7 mm above it) nor where
it would rest.

This module measures the collider itself, with the same triangles the physics
run collides with (``scene_physics_preflight.load_background_triangles`` through
the compile's binary read copy, ``usd_binary_read_cache``):

- ``footprint_top_z``: the highest point of a triangle soup over an axis-aligned
  plan rectangle, within a height window. It is exact for the triangles: a
  triangle's highest point over the rectangle is a vertex of their
  intersection, so it is a triangle vertex inside the rectangle, an edge's
  crossing of the rectangle's boundary, or a rectangle corner inside it;
- ``collider_support_surface``: that query bound to a scene collider, for the
  native adapter (``task_evaluation_rigid_relocation_native_adapter``), which
  seats the object's qualified collision bottom on the start reading and
  records the goal reading against the success window;
- ``start_spawn_disagreement_m``: how far a compiled plan's start pose lies from
  the scoring frame of the object it spawns. Beyond the scorer's reset
  tolerance no episode of that plan can be scored.

The highest point is an estimate of the rest height: an object on uneven
triangles can tilt and settle lower. Physics (the robot-free probe) decides.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Below the registered support top, the collider is searched this far at least (the registration's
#: placement uncertainty widens it): capture-447ae4f4's table faces span 5 cm under their highest one.
SEARCH_BELOW_SUPPORT_TOP_M = 0.03
#: Triangles kept around the start and goal footprints when the collider is read, beyond their reach.
READ_MARGIN_M = 0.1
#: A collider reading this close to the registered seating agrees with it (a 32-bit collider alone moves a
#: 0.75 m top by 3e-8 m): the task is then exactly the one its revision adapts, so a receipt recorded on
#: the revision's own task still binds it.
SEATING_AGREEMENT_M = 1e-4

SupportSurface = Callable[..., "float | None"]


def footprint_top_z(triangles: Any, lower_xy: Sequence[float], upper_xy: Sequence[float], *,
                    floor_z: float, ceiling_z: float) -> float | None:
    """The highest triangle point over the rectangle ``lower_xy``..``upper_xy`` with height in the window.

    ``triangles`` is ``(n, 3, 3)`` world coordinates, Z up. Points outside
    ``floor_z``..``ceiling_z`` are not read: below is another surface, above is
    clear of the object. None when no triangle reaches the rectangle there.
    """

    tris = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    x0, y0 = (float(value) for value in lower_xy)
    x1, y1 = (float(value) for value in upper_xy)
    if not (x0 < x1 and y0 < y1 and floor_z <= ceiling_z) or not len(tris):
        return None
    low, high = tris.min(axis=1), tris.max(axis=1)
    near = ((high[:, 0] >= x0) & (low[:, 0] <= x1) & (high[:, 1] >= y0) & (low[:, 1] <= y1)
            & (high[:, 2] >= floor_z) & (low[:, 2] <= ceiling_z))
    tris = tris[near]
    if not len(tris):
        return None
    heights: list[np.ndarray] = []
    # Triangle vertices inside the rectangle.
    points = tris.reshape(-1, 3)
    inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    heights.append(points[inside, 2])
    # Edges crossing the rectangle's four sides.
    for first, second in ((0, 1), (1, 2), (2, 0)):
        a, b = tris[:, first], tris[:, second]
        for axis, value, other, (lo, hi) in ((0, x0, 1, (y0, y1)), (0, x1, 1, (y0, y1)),
                                              (1, y0, 0, (x0, x1)), (1, y1, 0, (x0, x1))):
            span = b[:, axis] - a[:, axis]
            moving = np.abs(span) > 1e-15
            t = np.divide(value - a[:, axis], span, out=np.full(len(span), -1.0), where=moving)
            hit = moving & (t >= 0.0) & (t <= 1.0)
            along = a[:, other] + t * (b[:, other] - a[:, other])
            hit &= (along >= lo) & (along <= hi)
            heights.append((a[:, 2] + t * (b[:, 2] - a[:, 2]))[hit])
    # Rectangle corners inside a triangle, on its plane.
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    safe = np.abs(determinant) > 1e-18
    for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        ox, oy = x - a[:, 0], y - a[:, 1]
        u = np.divide(ox * e2[:, 1] - oy * e2[:, 0], determinant, out=np.full(len(a), -1.0), where=safe)
        v = np.divide(e1[:, 0] * oy - e1[:, 1] * ox, determinant, out=np.full(len(a), -1.0), where=safe)
        hit = safe & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1.0 + 1e-12)
        heights.append((a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2]))[hit])
    values = np.concatenate(heights)
    values = values[np.isfinite(values) & (values >= floor_z) & (values <= ceiling_z)]
    return float(values.max()) if len(values) else None


def triangle_support_surface(triangles: Any) -> SupportSurface:
    """``footprint_top_z`` over these triangles, in the adapter's ``support_surface`` shape."""

    held = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)

    def surface(lower_xy: Sequence[float], upper_xy: Sequence[float], *, floor_z: float,
                ceiling_z: float) -> float | None:
        return footprint_top_z(held, lower_xy, upper_xy, floor_z=floor_z, ceiling_z=ceiling_z)

    return surface


def collider_support_surface(collision_path: str | Path, *, points_xy: Sequence[Sequence[float]],
                             reach_m: float, height_z: float) -> SupportSurface:
    """The scene collider's own triangles around ``points_xy`` (start and goal), as a support query.

    ``reach_m`` is how far from each point a footprint extends. Read through the
    caller's binary read scope; raises ``ScenePhysicsPreflightError`` when the
    collider is not a static Z-up metre stage.
    """

    from .scene_physics_preflight import load_background_triangles

    xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    middle = xy.mean(axis=0)
    radius = float(np.max(np.linalg.norm(xy - middle, axis=1))) + float(reach_m) + READ_MARGIN_M
    triangles = load_background_triangles(collision_path, centre=[*middle.tolist(), float(height_z)], radius=radius)
    return triangle_support_surface(triangles)


def start_spawn_disagreement_m(plan: Mapping[str, Any]) -> float | None:
    """Distance from a scene plan's ``start_pose_world`` to the scoring frame of the task object it spawns.

    None when the plan is not a rigid task with one spawned task object and a
    scoring-frame transform (nothing to compare).
    """

    from .adp_rigid_task_scoring import _rotate_xyzw

    spec = plan.get("task_spec") if isinstance(plan, Mapping) else None
    if not isinstance(spec, Mapping) or spec.get("task_kind") != "rigid_pick_place":
        return None
    rows = [row for row in plan.get("objects") or [] if isinstance(row, Mapping) and row.get("task_subject") is True]
    transform = (spec.get("interaction_affordance") or {}).get("asset_root_from_scoring_frame") or {}
    try:
        (row,) = rows
        pose = (row.get("reset_state") or {}).get("root_pose_world") or row["pose_world"]
        root = [float(value) for value in pose["position_world_m"]]
        rotation = [float(value) for value in pose["orientation_xyzw"]]
        offset = [float(value) for value in transform["position_m"]]
        start = [float(value) for value in spec["start_pose_world"][:3]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(root) != 3 or len(rotation) != 4 or len(offset) != 3 or len(start) != 3:
        return None
    rotated = _rotate_xyzw(offset, rotation)
    scoring = [root[axis] + rotated[axis] for axis in range(3)]
    distance = math.dist(scoring, start)
    return distance if math.isfinite(distance) else None


def start_spawn_mismatch(plan: Mapping[str, Any]) -> str | None:
    """The typed reason no episode of ``plan`` can be scored from its spawn, or None."""

    distance = start_spawn_disagreement_m(plan)
    try:
        tolerance = float(plan["task_spec"]["reset_translation_tolerance_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if distance is None or not distance > tolerance:
        return None
    return f"rigid_task_start_pose_not_spawn_pose:{distance * 1000:.1f}mm>{tolerance * 1000:.1f}mm"


__all__ = [
    "READ_MARGIN_M",
    "SEARCH_BELOW_SUPPORT_TOP_M",
    "SEATING_AGREEMENT_M",
    "collider_support_surface",
    "footprint_top_z",
    "start_spawn_disagreement_m",
    "start_spawn_mismatch",
    "triangle_support_surface",
]
