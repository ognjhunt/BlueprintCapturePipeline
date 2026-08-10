"""Spot colliders whose convex hull would swallow the things inside them.

A convex hull is the smallest convex shape containing a mesh. For a hollow
shell that is the whole enclosed volume, so a fridge cabinet authored as a box
with a cavity becomes, to the solver, a solid block of fridge. Anything that
belongs *in* the cavity - shelves, door bins - is then permanently
interpenetrating it.

On this twin that cost five hypotheses and several runs. The cabinet shell is
0.713 x 0.528 x 1.61 m built from 95 points, and both door bins sit entirely
inside its hull. PhysX resolves the overlap every step, and because the door
hangs on a free hinge the resolution becomes rotation: 35 degrees before the
arm arrives.

The tell is a large enclosing volume described by very few points. A dense mesh
of that size is a real solid; a sparse one is a shell. Thin slabs are excluded
however large, because a shelf spans the cavity and encloses nothing.

Nothing here inspects mesh topology - it reads extents and vertex counts, which
is enough to flag a candidate and not enough to prove concavity. The
recommendation is to author a concave approximation, which is safe whether or
not the part turned out to be hollow.
"""

from __future__ import annotations

from typing import Any, Sequence


HOLLOW_COLLIDER_SCHEMA_VERSION = "hollow_collider_approximation.v1"
CONCAVE_APPROXIMATIONS = frozenset({"convexDecomposition", "sdf", "none", "meshSimplification"})
# Below this an enclosing volume is too small to swallow anything meaningful.
MINIMUM_SUSPECT_VOLUME_M3 = 0.05
# A box needs 8 points; anything under this for that volume is a shell, not a
# solid. A genuinely solid part of this size carries far more geometry.
MAXIMUM_SHELL_POINT_COUNT = 512
# A slab encloses nothing regardless of footprint.
MINIMUM_SUSPECT_THICKNESS_M = 0.05


class HollowColliderApproximationError(ValueError):
    """Stable, sorted collider-approximation failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def classify_collider_approximation(
    *,
    aabb_extents_m: Sequence[float],
    point_count: int,
    current_approximation: str,
) -> dict[str, Any]:
    """Whether this collider's hull is safe, and what it should be."""

    extents = [float(value) for value in aabb_extents_m]
    if len(extents) != 3 or any(value <= 0.0 for value in extents):
        raise HollowColliderApproximationError(
            ["hollow_collider_extents_invalid:" + ",".join(str(v) for v in extents)]
        )

    volume = extents[0] * extents[1] * extents[2]
    smallest = min(extents)
    already_concave = str(current_approximation) in CONCAVE_APPROXIMATIONS

    unsafe = (
        not already_concave
        and volume >= MINIMUM_SUSPECT_VOLUME_M3
        and smallest >= MINIMUM_SUSPECT_THICKNESS_M
        and int(point_count) <= MAXIMUM_SHELL_POINT_COUNT
    )

    return {
        "schema_version": HOLLOW_COLLIDER_SCHEMA_VERSION,
        "hull_is_unsafe": unsafe,
        "enclosed_volume_m3": volume,
        "smallest_extent_m": smallest,
        "point_count": int(point_count),
        "current_approximation": str(current_approximation),
        "recommended_approximation": (
            "convexDecomposition" if unsafe else str(current_approximation)
        ),
        "claim_boundary": {
            "extents_and_point_count_flag_a_candidate_not_a_proof": True,
            "a_concave_approximation_is_safe_either_way": True,
        },
    }


__all__ = [
    "CONCAVE_APPROXIMATIONS",
    "HOLLOW_COLLIDER_SCHEMA_VERSION",
    "HollowColliderApproximationError",
    "classify_collider_approximation",
]
