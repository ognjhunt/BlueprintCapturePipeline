"""Derive task-framing cameras from a solved stance — the render side of placement.

Once the placement solver has stood the robot in front of the target, the render
(Isaac RTX for USD scenes, the reference Spark splat renderer for 3DGS scenes)
needs cameras that actually SHOW the manipulation: the robot's POV looking at the
target, a third-person view of robot+target together, an overhead sanity view, and
a close task-focus view. This module derives all four from ``(StandPose, target)``
with zero scene-specific coordinates, in the same camera schema the perception
path already uses (``{eye, target, up, vfov, width, height}``, ``vfov`` in
RADIANS — see :mod:`perception_views`).

``to_splat_render_specs`` converts that schema into the ``--cameras`` JSON the
headless Spark harness (``tools/splat_render/render_splat.mjs``) consumes
(``{id, spec: {pos, target, fov(deg), up}}``), so a solved stance can be rendered
against the real splat locally with no GPU.

Pure + stdlib-only (``math``); hermetic to unit-test.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

from .perception_views import Camera
from .robot_profile import RobotProfile
from .types import SceneObject, StandPose, Vec3

# Optional camera-eye clearance test: world point -> True when the eye position
# is in open space (not inside a wall band or furniture volume).
EyeClearFn = Callable[[Vec3], bool]

DEFAULT_STANCE_CAMERA_IDS = ("head_pov", "third_person", "overhead", "task_focus")


def _rotate(rotation: Sequence[float], vector: Sequence[float]) -> Vec3:
    if len(rotation) != 9 or len(vector) != 3:
        raise ValueError("rotation_must_be_3x3_and_vector_must_be_3d")
    return tuple(
        sum(float(rotation[row * 3 + col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _unit(vector: Sequence[float]) -> Vec3:
    if len(vector) != 3:
        raise ValueError("camera_vector_must_be_3d")
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("camera_vector_must_be_finite_and_nonzero")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def link_mounted_camera_spec(
    *,
    parent_translation: Sequence[float],
    parent_rotation_row_major: Sequence[float],
    mount_translation: Sequence[float],
    mount_forward: Sequence[float],
    mount_up: Sequence[float] = (0.0, 0.0, 1.0),
    look_distance_m: float = 0.45,
    fov_deg: float = 72.0,
) -> dict:
    """Derive a render camera from a robot-link pose and a fixed link mount.

    The result uses the Spark camera schema, but the transform is renderer-neutral.
    A simulator must call this with the current link pose on every frame; freezing a
    single link pose is valid only for an initial-observation registration preview.
    """
    if len(parent_translation) != 3 or len(mount_translation) != 3:
        raise ValueError("camera_translations_must_be_3d")
    if not math.isfinite(float(look_distance_m)) or float(look_distance_m) <= 0:
        raise ValueError("look_distance_m_must_be_positive")
    if not math.isfinite(float(fov_deg)) or not 1.0 <= float(fov_deg) < 179.0:
        raise ValueError("fov_deg_out_of_range")

    rotated_offset = _rotate(parent_rotation_row_major, mount_translation)
    position = tuple(
        float(parent_translation[index]) + rotated_offset[index] for index in range(3)
    )
    forward = _unit(_rotate(parent_rotation_row_major, mount_forward))
    up = _unit(_rotate(parent_rotation_row_major, mount_up))
    if abs(sum(forward[index] * up[index] for index in range(3))) > 0.995:
        raise ValueError("camera_forward_and_up_are_collinear")
    target = tuple(
        position[index] + forward[index] * float(look_distance_m) for index in range(3)
    )
    return {
        "pos": [float(value) for value in position],
        "target": [float(value) for value in target],
        "fov": float(fov_deg),
        "up": [float(value) for value in up],
    }


def _eye_height_for_profile(robot_profile: Optional[RobotProfile]) -> float:
    """Approximate head-camera height above the floor for the robot embodiment.

    Pelvis + shoulder rise + a small neck/head offset. Falls back to the G1-scale
    ~1.23 m when no profile is given. Deliberately approximate — this frames a
    preview POV, it does not claim the exact head-link transform.
    """
    if robot_profile is None:
        return 0.79 + 0.29 + 0.15
    return (
        float(robot_profile.pelvis_height_m)
        + float(robot_profile.shoulder_above_root_m)
        + 0.15
    )


def _camera(eye: Vec3, target: Vec3, *, vfov_deg: float, width: int, height: int) -> Camera:
    return {
        "eye": tuple(float(v) for v in eye),
        "target": tuple(float(v) for v in target),
        "up": (0.0, 0.0, 1.0),
        "vfov": math.radians(float(vfov_deg)),
        "width": int(width),
        "height": int(height),
    }


def stance_task_cameras(
    pose: StandPose,
    target: SceneObject,
    *,
    floor_z: float = 0.0,
    robot_profile: Optional[RobotProfile] = None,
    eye_height: Optional[float] = None,
    vfov_deg: float = 60.0,
    width: int = 1280,
    height: int = 960,
    ceiling_z: Optional[float] = None,
    eye_clear_fn: Optional[EyeClearFn] = None,
) -> Dict[str, Camera]:
    """Four named cameras framing the solved stance's manipulation moment.

    * ``head_pov`` — from the robot's head position at the stance, looking at the
      target centroid (the WAM/VLA policy seed view).
    * ``third_person`` — behind and to the side of the robot, looking at the
      midpoint between robot and target (shows both in frame).
    * ``overhead`` — above the robot↔target midpoint looking straight down
      (placement/clearance sanity). Clamped under ``ceiling_z`` when given.
    * ``task_focus`` — a close-up of the target from the robot's approach
      direction, backed off proportionally to the target size.

    Everything derives from the pose, the target AABB, and the robot profile —
    no scene-specific coordinates.
    """
    px, py, _pz = (float(v) for v in pose.position)
    yaw = float(pose.yaw)
    tx, ty, tz = (float(v) for v in target.centroid)
    eye_h = float(eye_height) if eye_height is not None else _eye_height_for_profile(robot_profile)
    head = (px, py, floor_z + eye_h)

    fwd = (math.cos(yaw), math.sin(yaw))
    left = (-fwd[1], fwd[0])
    mid = (0.5 * (px + tx), 0.5 * (py + ty), 0.5 * (head[2] + tz))

    dx, dy, dz = target.size()
    half_diag = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
    focus_back = max(3.0 * half_diag, 0.6)
    focus_eye = (
        tx - fwd[0] * focus_back,
        ty - fwd[1] * focus_back,
        max(tz + 0.4 * focus_back, floor_z + 0.3),
    )

    # Third-person eye: behind-left by default, but an offset eye can land inside
    # a wall band or furniture (the stance is validated; the camera spot is not).
    # With an ``eye_clear_fn``, try behind-left / behind-right / straight-behind /
    # a closer high vantage, and take the first CLEAR one.
    third_z = floor_z + eye_h + 0.5
    third_candidates = [
        (px - fwd[0] * 1.6 + left[0] * 1.1, py - fwd[1] * 1.6 + left[1] * 1.1, third_z),
        (px - fwd[0] * 1.6 - left[0] * 1.1, py - fwd[1] * 1.6 - left[1] * 1.1, third_z),
        (px - fwd[0] * 2.0, py - fwd[1] * 2.0, third_z),
        (px - fwd[0] * 1.0 + left[0] * 0.7, py - fwd[1] * 1.0 + left[1] * 0.7, third_z + 0.4),
        (px - fwd[0] * 1.0 - left[0] * 0.7, py - fwd[1] * 1.0 - left[1] * 0.7, third_z + 0.4),
    ]
    third_eye = third_candidates[0]
    if eye_clear_fn is not None:
        for candidate in third_candidates:
            if eye_clear_fn(candidate):
                third_eye = candidate
                break

    overhead_z = floor_z + 3.2
    if ceiling_z is not None and math.isfinite(float(ceiling_z)):
        overhead_z = min(overhead_z, float(ceiling_z) - 0.15)
    overhead_eye = (mid[0], mid[1], max(overhead_z, floor_z + 1.5))
    # Looking straight down makes (0,0,1) up degenerate; nudge the look-at point
    # toward the target so the view direction keeps a horizontal component.
    overhead_target = (
        mid[0] + 0.15 * (tx - px + 1e-3),
        mid[1] + 0.15 * (ty - py + 1e-3),
        floor_z,
    )

    return {
        "head_pov": _camera(head, (tx, ty, tz), vfov_deg=vfov_deg, width=width, height=height),
        "third_person": _camera(third_eye, mid, vfov_deg=vfov_deg, width=width, height=height),
        "overhead": _camera(overhead_eye, overhead_target, vfov_deg=vfov_deg, width=width, height=height),
        "task_focus": _camera(focus_eye, (tx, ty, tz), vfov_deg=max(vfov_deg - 15.0, 30.0), width=width, height=height),
    }


def to_splat_render_specs(cameras: Dict[str, Camera]) -> List[dict]:
    """Convert package cameras to the Spark harness ``--cameras`` JSON shape.

    The harness (``tools/splat_render/render_splat.mjs``) expects
    ``[{"id", "spec": {"pos": [..], "target": [..], "fov": deg, "up": [..]}}]``
    with ``fov`` in DEGREES (three.js convention), whereas the package schema
    stores ``vfov`` in radians — this converter owns that unit flip.
    """
    specs: List[dict] = []
    for cam_id, cam in cameras.items():
        specs.append(
            {
                "id": str(cam_id),
                "spec": {
                    "pos": [float(v) for v in cam["eye"]],  # type: ignore[union-attr]
                    "target": [float(v) for v in cam["target"]],  # type: ignore[union-attr]
                    "fov": math.degrees(float(cam["vfov"])),  # type: ignore[arg-type]
                    "up": [float(v) for v in cam.get("up", (0.0, 0.0, 1.0))],  # type: ignore[union-attr]
                },
            }
        )
    return specs


__all__ = [
    "DEFAULT_STANCE_CAMERA_IDS",
    "link_mounted_camera_spec",
    "stance_task_cameras",
    "to_splat_render_specs",
]
