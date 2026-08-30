"""Derive per-camera task-object framing expectations from sealed geometry.

The semantic framing gate historically asserted fixed pixel minimums
(``CAMERA_THRESHOLDS``) that were calibrated on one scene.  Construction run
r18 for scene 839873 measured the task object correctly segmented and centred
at 93 pixels on the static external camera while the gate demanded 200 -- a
number the geometry cannot produce: a 12.8 cm object at 1.43 m depth through
fx=172.9 on a 320x180 frame projects to a ~15x16 px bounding box.  The gate
therefore failed a camera that observed the object as well as physics allows,
and it would have failed every future paid run of that scene identically.

This module computes what the sealed scene geometry *can* produce, so the
framing minimum can be scaled down to a fraction of that expectation instead
of asserting an unreachable constant.  Three properties are deliberate:

* The expectation is computed from sealed inputs only -- the staged task
  object's authored USD extent, the sealed camera matrix and intrinsics, and
  the sealed task start/destination positions.  Nothing is read back from the
  provider at gate time.
* Scaling only ever *lowers* the configured minimum, never raises it, and
  never below an absolute floor.  A camera that cannot even meet the floor is
  refused at plan time, before any paid execution.
* When no expectation is available (robot-parented cameras, legacy plans
  compiled before this contract) the configured constants apply unchanged.

Empirical anchor for the constants below: r18's external camera measured 93
segmented pixels against a projected bounding-box area of ~240 px^2 (a 39%
fill after self-occlusion and the raked viewing angle).  ``0.20`` demands half
of that observed fill, tolerating partial occlusion by the arm without
accepting an absent or unrendered object, and ``24`` pixels is the smallest
mask this gate will ever accept as "observed".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "native_task_camera_framing_expectation.v1"

#: Fraction of the projected bounding-box area the effective minimum demands.
FRAMING_EXPECTED_AREA_SCALE = 0.20

#: The smallest segmented-pixel minimum the scaling may ever produce.
FRAMING_MINIMUM_FLOOR_PIXELS = 24

#: Projected bounding-box areas below this cannot reliably satisfy the floor
#: (0.4 fill of 60 px^2 is the floor itself); plans refuse instead of paying
#: to discover it.
FRAMING_MINIMUM_EXPECTED_BBOX_AREA_PX = 60.0

#: A camera closer than this to the object cannot be reasoned about with a
#: pinhole projection of the whole extent.
_MINIMUM_DEPTH_M = 0.05

#: Authored extents outside this band are treated as unmeasured geometry
#: rather than a manipulable task object.
_MINIMUM_EXTENT_M = 0.001
_MAXIMUM_EXTENT_M = 10.0


class NativeTaskCameraFramingExpectationError(RuntimeError):
    """Sealed camera framing expectations could not be derived fail-closed."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = [str(value) for value in errors]
        super().__init__("; ".join(self.errors))


def measure_task_object_extent_m(asset_path: str | Path) -> list[float]:
    """Measure the authored world-axis-aligned extent of one staged asset.

    Uses the authored default-time bound across the render-relevant purposes,
    scaled by the stage's metersPerUnit.  Refuses rather than guessing when
    the stage is unreadable or carries no measurable geometry: a rigid task
    object without geometry cannot be manipulated, observed, or scored.
    """

    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:  # noqa: BLE001 - exact import failure is evidence
        raise NativeTaskCameraFramingExpectationError(
            [f"native_task_camera_framing_pxr_unavailable:{type(exc).__name__}"]
        ) from exc
    path = Path(asset_path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_task_object_unreadable"]
        )
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
    )
    bound = cache.ComputeWorldBound(stage.GetPseudoRoot())
    aligned = bound.ComputeAlignedRange()
    if aligned.IsEmpty():
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_task_object_extent_unmeasurable"]
        )
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_stage_units_invalid"]
        )
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    extent = [
        (float(maximum[index]) - float(minimum[index])) * meters_per_unit
        for index in range(3)
    ]
    if any(
        not math.isfinite(value)
        or value < _MINIMUM_EXTENT_M
        or value > _MAXIMUM_EXTENT_M
        for value in extent
    ):
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_task_object_extent_unmeasurable"]
        )
    return extent


def _camera_world_pose(matrix: Sequence[Any]) -> tuple[list[float], list[float]]:
    """Return (position, optical z axis) from a flat row-major 4x4 matrix."""

    values = [float(value) for value in matrix]
    if len(values) != 16 or any(not math.isfinite(value) for value in values):
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_camera_matrix_invalid"]
        )
    position = [values[3], values[7], values[11]]
    z_axis = [values[2], values[6], values[10]]
    norm = math.sqrt(sum(value * value for value in z_axis))
    if not 0.5 < norm < 2.0:
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_camera_matrix_invalid"]
        )
    return position, [value / norm for value in z_axis]


def camera_framing_expectation(
    *,
    camera: Mapping[str, Any],
    object_extent_m: Sequence[float],
    object_positions_world: Sequence[Sequence[float]],
) -> dict[str, Any] | None:
    """Project the object's extent through one sealed world-frame camera.

    Returns ``None`` for cameras this expectation cannot honestly describe:
    robot-parented cameras move with the arm, so their framing is the robot's
    responsibility and the configured constants stay authoritative.
    """

    if str(camera.get("pose_frame") or "") != "world":
        return None
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_intrinsics_invalid"]
        )
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_intrinsics_invalid"]
        ) from exc
    if (
        not math.isfinite(fx)
        or not math.isfinite(fy)
        or fx <= 0.0
        or fy <= 0.0
        or width < 1
        or height < 1
    ):
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_intrinsics_invalid"]
        )
    extent = [float(value) for value in object_extent_m]
    if len(extent) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in extent
    ):
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_object_extent_invalid"]
        )
    position, z_axis = _camera_world_pose(camera.get("frame_from_camera_matrix") or [])
    # The two largest extents bound what any viewing angle can show; the
    # smallest is always at least partially foreshortened away.
    e_max, e_mid = sorted(extent, reverse=True)[:2]
    positions_evidence: list[dict[str, Any]] = []
    for row in object_positions_world:
        point = [float(value) for value in list(row)[:3]]
        if len(point) != 3 or any(not math.isfinite(value) for value in point):
            raise NativeTaskCameraFramingExpectationError(
                ["native_task_camera_framing_object_position_invalid"]
            )
        delta = [point[index] - position[index] for index in range(3)]
        depth = sum(delta[index] * z_axis[index] for index in range(3))
        if depth < _MINIMUM_DEPTH_M:
            raise NativeTaskCameraFramingExpectationError(
                ["native_task_camera_framing_object_behind_camera"]
            )
        bbox_width_px = fx * e_max / depth
        bbox_height_px = fy * e_mid / depth
        positions_evidence.append(
            {
                "object_position_world_m": point,
                "depth_m": depth,
                "projected_bbox_width_px": bbox_width_px,
                "projected_bbox_height_px": bbox_height_px,
                "projected_bbox_area_px": bbox_width_px * bbox_height_px,
            }
        )
    if not positions_evidence:
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_object_position_invalid"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "object_extent_m": extent,
        "frame_resolution_hw": [height, width],
        "positions": positions_evidence,
        "expected_bbox_area_px": min(
            row["projected_bbox_area_px"] for row in positions_evidence
        ),
    }


def effective_framing_minimums(
    *,
    minimum_pixels: int,
    minimum_pixel_fraction: float,
    frame_width: int,
    frame_height: int,
    expected_bbox_area_px: float,
) -> dict[str, Any]:
    """Scale the configured framing minimums to the geometric expectation.

    Scaling only lowers, never raises: a scene whose geometry supports the
    configured constants keeps them bit-exactly.  The floor keeps "observed"
    meaning a resolvable mask rather than speckle.
    """

    if (
        not math.isfinite(float(expected_bbox_area_px))
        or float(expected_bbox_area_px) <= 0.0
    ):
        raise NativeTaskCameraFramingExpectationError(
            ["native_task_camera_framing_expectation_invalid"]
        )
    scaled = math.ceil(FRAMING_EXPECTED_AREA_SCALE * float(expected_bbox_area_px))
    effective_pixels = max(
        FRAMING_MINIMUM_FLOOR_PIXELS, min(int(minimum_pixels), scaled)
    )
    effective_fraction = min(
        float(minimum_pixel_fraction),
        effective_pixels / float(frame_width * frame_height),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "configured_minimum_pixels": int(minimum_pixels),
        "configured_minimum_pixel_fraction": float(minimum_pixel_fraction),
        "expected_bbox_area_px": float(expected_bbox_area_px),
        "expected_area_scale": FRAMING_EXPECTED_AREA_SCALE,
        "floor_pixels": FRAMING_MINIMUM_FLOOR_PIXELS,
        "effective_minimum_pixels": int(effective_pixels),
        "effective_minimum_pixel_fraction": float(effective_fraction),
    }


__all__ = [
    "FRAMING_EXPECTED_AREA_SCALE",
    "FRAMING_MINIMUM_EXPECTED_BBOX_AREA_PX",
    "FRAMING_MINIMUM_FLOOR_PIXELS",
    "NativeTaskCameraFramingExpectationError",
    "SCHEMA_VERSION",
    "camera_framing_expectation",
    "effective_framing_minimums",
    "measure_task_object_extent_m",
]
