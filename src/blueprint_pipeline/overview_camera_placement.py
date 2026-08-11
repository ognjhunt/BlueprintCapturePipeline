"""Place a static overview camera that frames a whole task episode.

The review stream exists for a human. The policy cameras cannot serve it:
they are robot-mounted, tightly framed on the manipulanda, and oriented for
the policy's benefit. An overview is the opposite contract - one fixed,
upright, wide viewpoint that keeps the robot, the task object, and the space
between them in frame for the entire episode.

Three guarantees, each the residue of a paid lesson:

- **everything in frame**: the pose is derived from the scene's bounding
  sphere and the camera's own field of view, then every input point is
  checked against the view cone before the pose is returned;
- **horizon upright**: the rotation is built from world-up by construction
  and refuses configurations where roll could creep in (a review camera
  nobody can watch sideways is not review evidence);
- **deterministic**: same points, same answer, no randomness - the pose is
  part of the run's provenance.

The rotation is returned in Isaac's OffsetCfg convention: quaternion
(w, x, y, z), OpenGL camera axes (forward -Z, up +Y).
"""

from __future__ import annotations

import math
from typing import Any, Sequence


OVERVIEW_CAMERA_PLACEMENT_SCHEMA_VERSION = "overview_camera_placement.v1"
DEFAULT_MARGIN_FRACTION = 1.35
# A geometrically valid pose half a metre from the action at knee height is
# not an overview a human can review. Wide rigs on small scenes hit this.
DEFAULT_MINIMUM_DISTANCE_M = 1.5
DEFAULT_ELEVATION_DEG = 24.0
DEFAULT_AZIMUTH_OFFSET_DEG = 135.0
_MINIMUM_UP_DOT = 0.9


class OverviewCameraPlacementError(ValueError):
    """Stable, sorted placement failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _unit(vector: Sequence[float], error: str) -> list[float]:
    length = _norm(vector)
    if length <= 1e-9:
        raise OverviewCameraPlacementError([error])
    return [value / length for value in vector]


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _rotation_to_wxyz(columns: Sequence[Sequence[float]]) -> list[float]:
    """Quaternion (w,x,y,z) from a rotation given as three column vectors."""

    m = [[columns[col][row] for col in range(3)] for row in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return [w, x, y, z]


def plan_overview_camera(
    *,
    scene_points_world_m: Sequence[Sequence[float]],
    fov_horizontal_rad: float,
    image_aspect: float,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    azimuth_offset_deg: float = DEFAULT_AZIMUTH_OFFSET_DEG,
) -> dict[str, Any]:
    """One fixed, upright pose with every scene point inside the view cone."""

    points: list[list[float]] = []
    for index, raw in enumerate(scene_points_world_m):
        try:
            point = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise OverviewCameraPlacementError(
                [f"overview_camera_point_invalid:{index}"]
            ) from exc
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise OverviewCameraPlacementError(
                [f"overview_camera_point_invalid:{index}"]
            )
        points.append(point)
    if len(points) < 2:
        raise OverviewCameraPlacementError(["overview_camera_needs_two_points"])
    fov = float(fov_horizontal_rad)
    aspect = float(image_aspect)
    if not 0.1 < fov < math.pi:
        raise OverviewCameraPlacementError(["overview_camera_fov_invalid"])
    if not 0.1 < aspect < 10.0:
        raise OverviewCameraPlacementError(["overview_camera_aspect_invalid"])

    centre = [sum(point[i] for point in points) / len(points) for i in range(3)]
    radius = max(_norm([p[i] - centre[i] for i in range(3)]) for p in points)
    radius = max(radius, 0.25)

    # The vertical FOV is the tighter cone for wide images; the distance is
    # set so the bounding sphere fits the tighter one with margin.
    fov_vertical = 2.0 * math.atan(math.tan(fov / 2.0) / aspect)
    tightest = min(fov, fov_vertical)
    distance = (radius * float(margin_fraction)) / math.tan(tightest / 2.0)
    distance = max(distance, DEFAULT_MINIMUM_DISTANCE_M)

    # Viewing direction: start from the bearing of the scene's dominant axis
    # (first point toward last, typically robot toward task), swing by the
    # azimuth offset so the camera looks across the action rather than down
    # its throat, then elevate.
    baseline = [points[-1][i] - points[0][i] for i in range(2)] + [0.0]
    if _norm(baseline) <= 1e-6:
        baseline = [1.0, 0.0, 0.0]
    bearing = math.atan2(baseline[1], baseline[0])
    azimuth = bearing + math.radians(float(azimuth_offset_deg))
    elevation = math.radians(float(elevation_deg))
    offset = [
        math.cos(azimuth) * math.cos(elevation) * distance,
        math.sin(azimuth) * math.cos(elevation) * distance,
        math.sin(elevation) * distance,
    ]
    position = [centre[i] + offset[i] for i in range(3)]

    forward = _unit(
        [centre[i] - position[i] for i in range(3)],
        "overview_camera_degenerate_forward",
    )
    world_up = [0.0, 0.0, 1.0]
    right = _unit(_cross(forward, world_up), "overview_camera_looking_straight_down")
    camera_up = _cross(right, forward)
    # OpenGL columns: X = right, Y = up, Z = -forward.
    rotation_wxyz = _rotation_to_wxyz(
        [right, camera_up, [-value for value in forward]]
    )

    half_tightest = tightest / 2.0
    worst_angle = 0.0
    for point in points:
        ray = [point[i] - position[i] for i in range(3)]
        cosine = sum(forward[i] * ray[i] for i in range(3)) / _norm(ray)
        worst_angle = max(worst_angle, math.acos(max(-1.0, min(1.0, cosine))))
    if worst_angle >= half_tightest:
        raise OverviewCameraPlacementError(
            [
                "overview_camera_point_outside_view_cone:"
                f"worst_{math.degrees(worst_angle):.1f}deg_of_{math.degrees(half_tightest):.1f}deg"
            ]
        )
    if camera_up[2] < _MINIMUM_UP_DOT:
        raise OverviewCameraPlacementError(["overview_camera_horizon_not_upright"])

    return {
        "schema_version": OVERVIEW_CAMERA_PLACEMENT_SCHEMA_VERSION,
        "position_world_m": position,
        "rotation_wxyz_opengl": rotation_wxyz,
        "receipt": {
            "scene_point_count": len(points),
            "bounding_centre_world_m": centre,
            "bounding_radius_m": radius,
            "camera_distance_m": distance,
            "worst_point_angle_deg": math.degrees(worst_angle),
            "view_cone_half_angle_deg": math.degrees(half_tightest),
            "horizon_upright": True,
            "claim_boundary": {
                "frame_containment_is_geometric_not_rendered": True,
                "occlusion_between_points_is_not_modelled": True,
            },
        },
    }


__all__ = [
    "DEFAULT_AZIMUTH_OFFSET_DEG",
    "DEFAULT_ELEVATION_DEG",
    "DEFAULT_MARGIN_FRACTION",
    "DEFAULT_MINIMUM_DISTANCE_M",
    "OVERVIEW_CAMERA_PLACEMENT_SCHEMA_VERSION",
    "OverviewCameraPlacementError",
    "plan_overview_camera",
]
