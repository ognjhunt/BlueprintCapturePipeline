"""Clear the sub-millimetre interference a solver turns into motion.

A rigid-body solver has no notion of "close enough". Two parts overlapping by
0.8 mm at the rest pose are two parts it must separate, and if one of them
hangs on a free hinge the separation becomes rotation: on this twin, 0.8 mm
between a door trim component and the cabinet face opened the door 35 degrees
before anything touched it.

The shift goes along the shallowest overlapping axis. The same pair overlapped
13 mm in x and 6 mm in z, and pushing along either of those would move the part
visibly to fix a fault under a millimetre deep.

Depth is the difference between a rest-pose gap and a modelling error, so
anything past the shallow bound refuses. Nudging a part a centimetre puts it
somewhere it does not belong and hides the real fault behind a part that no
longer looks wrong.
"""

from __future__ import annotations

from typing import Any, Sequence


REST_POSE_INTERFERENCE_SCHEMA_VERSION = "rest_pose_interference.v1"
# Past this an overlap is a modelling error, not a rest-pose gap.
DEFAULT_MAXIMUM_SHALLOW_OVERLAP_M = 0.005


class RestPoseInterferenceError(ValueError):
    """Stable, sorted interference-planning failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def plan_axis_clearance(
    *,
    moving_min: Sequence[float],
    moving_max: Sequence[float],
    blocking_min: Sequence[float],
    blocking_max: Sequence[float],
    clearance_m: float = 0.001,
    maximum_shallow_overlap_m: float = DEFAULT_MAXIMUM_SHALLOW_OVERLAP_M,
) -> dict[str, Any]:
    """How far to move one part so it stops intersecting another."""

    overlaps = [
        min(float(moving_max[i]), float(blocking_max[i]))
        - max(float(moving_min[i]), float(blocking_min[i]))
        for i in range(3)
    ]
    if any(value <= 0.0 for value in overlaps):
        return {
            "schema_version": REST_POSE_INTERFERENCE_SCHEMA_VERSION,
            "already_clear": True,
            "axis": None,
            "overlap_m": 0.0,
            "shift_m": 0.0,
        }

    axis = min(range(3), key=lambda i: overlaps[i])
    overlap = overlaps[axis]
    if overlap > float(maximum_shallow_overlap_m):
        raise RestPoseInterferenceError(
            [
                "rest_pose_interference_overlap_too_deep:"
                f"axis={axis}:overlap_m={overlap:.5g}:"
                f"limit_m={float(maximum_shallow_overlap_m):.5g}"
            ]
        )

    # Move away from the blocker along the axis the parts are least committed
    # to sharing: whichever side of the blocker's centre the part sits on.
    moving_centre = (float(moving_min[axis]) + float(moving_max[axis])) / 2.0
    blocking_centre = (float(blocking_min[axis]) + float(blocking_max[axis])) / 2.0
    direction = 1.0 if moving_centre >= blocking_centre else -1.0
    shift = direction * (overlap + float(clearance_m))

    return {
        "schema_version": REST_POSE_INTERFERENCE_SCHEMA_VERSION,
        "already_clear": False,
        "axis": axis,
        "overlap_m": overlap,
        "shift_m": shift,
        "clearance_m": float(clearance_m),
        "claim_boundary": {
            "aabb_overlap_is_not_mesh_overlap": True,
            "a_shift_moves_geometry_it_does_not_fix_a_model": True,
        },
    }


__all__ = [
    "DEFAULT_MAXIMUM_SHALLOW_OVERLAP_M",
    "REST_POSE_INTERFERENCE_SCHEMA_VERSION",
    "RestPoseInterferenceError",
    "plan_axis_clearance",
]
