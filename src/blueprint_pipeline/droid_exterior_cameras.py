"""DROID's right exterior camera, placed clear of a scene for the policies that read it.

Beta journey step: candidate screening of a completed site scene. DROID policies
that read both over-the-shoulder exteriors (Cosmos 3 Policy, FLUX 3 Action) need
the right one as well, but a scene's DROID station carries one exterior, the
wrist camera and a review-only overview. Arena's DROID rig mounts its exteriors
at (0.05, +-0.57, 0.66) m on ``panda_link0`` (``DroidCameraCfg``), and in a
kitchen the right mount can sit inside a base cabinet. The right exterior is
therefore placed only where it sees the task clear of the captured scene:
Arena's own mount first, then nearby DROID-like poses aimed at the task. With
none, the scene has no right exterior and the candidates that read it are
skipped with the recorded reason; the two-view candidates are unchanged.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "droid_exterior_right_placement.v1"
# The runtime renders it through an added Arena slot (``native_task_robot_registry``).
ROLE = "exterior_right"
# ``isaaclab_arena.embodiments.droid.droid.DroidCameraCfg.external_camera_2``: offset on
# panda_link0, rotation XYZW (Isaac Lab 3), OpenGL optics. Its optical axis meets the base
# plane 0.73 m ahead of the arm, 4 cm right of centre.
ARENA_MOUNT_POSITION_M = (0.05, -0.57, 0.66)
ARENA_MOUNT_ROTATION_XYZW = (0.399, -0.195, -0.393, 0.805)
# Its optics (1280 x 720, 2.1 mm over a 5.376 x 3.024 mm aperture: 104 degrees across), the
# same as the left exterior a DROID canary keeps, so the two exterior views share one size. The
# principal point is the pixel-centre one the runtime's pinhole conversion accepts, (size - 1) / 2.
ARENA_EXTERIOR_INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 639.5, "cy": 359.5, "width": 1280, "height": 720}
# Alternatives stay in DROID's own exterior range: beside the shoulder, a little ahead or
# behind, at a tripod's height. Base-frame steps from Arena's mount, nearest first.
ALTERNATIVE_STEPS_M = ((0.0, -0.10, 0.15), (0.0, 0.15, -0.15), (0.0, 0.25, -0.15))
UNAVAILABLE_NO_CLEAR_VIEW = "droid_exterior_right_no_clear_view"
UNAVAILABLE_UNMEASURED = "droid_exterior_right_unmeasured"

_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def _rotation_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("droid_exterior_rotation_invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _world_from_base(base_pose_world: Mapping[str, Any]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation_xyzw(base_pose_world["orientation_xyzw"])
    matrix[:3, 3] = [float(value) for value in base_pose_world["position_world_m"]]
    return matrix


def arena_mount_world(base_pose_world: Mapping[str, Any]) -> np.ndarray:
    """World-from-camera (OpenCV optics) of Arena's right exterior for a base pose."""

    mount = np.eye(4)
    mount[:3, :3] = _rotation_xyzw(ARENA_MOUNT_ROTATION_XYZW) @ _OPENGL_TO_OPENCV
    mount[:3, 3] = ARENA_MOUNT_POSITION_M
    return _world_from_base(base_pose_world) @ mount


def _aimed(position: np.ndarray, aim: np.ndarray) -> np.ndarray:
    forward = aim - position
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("droid_exterior_aim_invalid")
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("droid_exterior_aim_invalid")
    right = right / np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, np.cross(forward, right), forward])
    matrix[:3, 3] = position
    return matrix


def candidate_poses(base_pose_world: Mapping[str, Any], aim_world: Sequence[float]) -> list[tuple[str, np.ndarray]]:
    """Arena's mount, then the alternatives aimed at the task, nearest to the mount first."""

    world_from_base = _world_from_base(base_pose_world)
    aim = np.asarray([float(value) for value in aim_world])
    steps = sorted((step for step in itertools.product(*ALTERNATIVE_STEPS_M) if any(step)),
                   key=lambda step: (math.dist(step, (0.0, 0.0, 0.0)), step))
    poses = [("arena_mount", arena_mount_world(base_pose_world))]
    for step in steps:
        local = np.append(np.add(ARENA_MOUNT_POSITION_M, step), 1.0)
        poses.append(("offset_m:" + ",".join(f"{value:+.2f}" for value in step),
                      _aimed((world_from_base @ local)[:3], aim)))
    return poses


def camera_row(world_from_camera: np.ndarray, intrinsics: Mapping[str, Any]) -> dict[str, Any]:
    """The scene plan's world camera row for the right exterior."""

    return {"role": ROLE, "policy_input": True, "scoring_input": False, "pose_frame": "world",
            "parent_prim_path": "{ENV_REGEX_NS}", "optical_convention": "opencv",
            "frame_from_camera_matrix": [float(value) for value in np.asarray(world_from_camera).reshape(-1)],
            "intrinsics": dict(intrinsics)}


def place_exterior_right(*, base_pose_world: Mapping[str, Any], aim_world: Sequence[float],
                         intrinsics: Mapping[str, Any],
                         evaluate: Callable[[Mapping[str, Any]], Sequence[str]]) -> tuple[dict | None, dict]:
    """The first candidate pose ``evaluate`` finds no problem with, and the sealed placement receipt.

    ``evaluate`` returns the problems that keep one camera row from seeing the task through
    the scene (``scene_camera_visibility.packet_view_evaluator``); an empty list places it.
    """

    attempts: list[dict[str, Any]] = []
    placed: dict[str, Any] | None = None
    for candidate_id, matrix in candidate_poses(base_pose_world, aim_world):
        row = camera_row(matrix, intrinsics)
        problems = sorted(str(value) for value in evaluate(row))
        attempts.append({"candidate": candidate_id, "position_world_m": matrix[:3, 3].tolist(),
                         "problems": problems})
        if not problems:
            placed = row
            break
    receipt = {"schema_version": SCHEMA_VERSION, "role": ROLE,
               "status": "placed" if placed is not None else "unavailable",
               "reason": None if placed is not None else UNAVAILABLE_NO_CLEAR_VIEW,
               "placement": attempts[-1]["candidate"] if placed is not None else None,
               "aim_world_m": [float(value) for value in aim_world], "attempts": attempts}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return placed, receipt


def unmeasured_receipt(code: str) -> dict[str, Any]:
    """The receipt when the scene cannot be measured: no right exterior, the reason recorded."""

    receipt = {"schema_version": SCHEMA_VERSION, "role": ROLE, "status": "unavailable",
               "reason": f"{UNAVAILABLE_UNMEASURED}:{code}", "placement": None, "aim_world_m": None,
               "attempts": []}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ARENA_EXTERIOR_INTRINSICS",
    "ARENA_MOUNT_POSITION_M",
    "ARENA_MOUNT_ROTATION_XYZW",
    "ROLE",
    "SCHEMA_VERSION",
    "UNAVAILABLE_NO_CLEAR_VIEW",
    "UNAVAILABLE_UNMEASURED",
    "arena_mount_world",
    "camera_row",
    "candidate_poses",
    "place_exterior_right",
    "unmeasured_receipt",
]
