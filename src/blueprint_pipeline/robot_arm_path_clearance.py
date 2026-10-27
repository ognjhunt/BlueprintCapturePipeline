"""Whether the arm can hold a task part's handle through the task's motion without the part striking it.

Placement measured reach (the hand's position along the task path) and the robot's fixed base and
shoulder against the moving part. Neither sees the moving arm. On 2026-10-26 the one base pose that
passed for the Atlas dishwasher, 45 cm to the side of the handle, had the elbow 57 mm inside the door at
49 degrees: the planned grip held the middle of the handle, and from that base the arm had to cross
under the panel as it came down. Holding the handle 10-20 cm toward the robot cleared the door at every
grip rotation tried. A paid run would have scored the strike as the policy colliding with the task.

So the question a person answers by looking is asked here: is there a way to hold this handle, from this
base, that keeps the arm out of the part's way? Grips vary where along the handle the hand closes and how
far it is turned about the handle. Each grip is carried with the part from the task's reset to its
success threshold. The arm's full pose is solved at every step from the previous step's solution with
the published Franka kinematics (``franka_kinematics``, whose link frames the runtime asset's match to
3e-7 m), and every arm link -- the links the collision rule forbids from touching the task object -- is
measured against the moving part's colliders at the same joint value. A grip clears when the arm
reaches every step and keeps ``margin_m`` from the part throughout. A base passes when any grip clears.

The handle is taken to run along the hinge of a revolute part (a dishwasher's, an oven's or a fridge's
handle does) and across the slide of a prismatic one, as far as the part's collider under the contact
reaches. The gripper is meant to touch the handle and is not measured; link 0 is measured, with link 1's
housing, by the fixed-volume check in ``scene_physics_preflight``. Nothing here runs a simulator, and a
pass does not mean a policy will find the grip: it means one exists, so a policy whose arm the part
strikes failed the task rather than met an impossible setup.

The room around the task can block the arm as surely as the part: the counter over a dishwasher, the
wall beside a fridge, the floor under a low drawer. Given the static scene (``scene_obstacles``), the
same links are measured against it, and against the task asset's own fixed bodies, with the same margin;
a grip clears only when it keeps clear of both.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .franka_kinematics import FRANKA_FLANGE_OFFSET_M, FRANKA_JOINT_LIMITS_RAD, FRANKA_MODIFIED_DH
from .scene_physics_preflight import (
    MAXIMUM_BACKGROUND_SAMPLES,
    ScenePhysicsPreflightError,
    _clip_large_triangles,
    _collider_surface,
    _depths,
    _file_sha256,
    _movable_joint,
    _pose_matrix,
    body_poses,
    descendants,
    load_asset_model,
    load_background_triangles,
    sample_surface,
    task_motion_ranges,
)

SCHEMA_VERSION = "robot_arm_path_clearance.v1"
PACKET_REPORT_FILENAME = "native_task_robot_arm_path_clearance.v1.json"
DEFAULT_MARGIN_M = 0.02
# Where along the handle's free half-length the hand closes, and how far it is turned about the handle.
GRIP_SLIDE_FRACTIONS = (0.0, -0.4, 0.4, -0.8, 0.8)
GRIP_ROLLS_DEG = (0.0, -15.0, 15.0, -30.0, 30.0)
FINGER_HALF_WIDTH_M = 0.02
REVOLUTE_STEP_RAD = math.radians(3.0)
PRISMATIC_STEP_M = 0.02
MAXIMUM_JOINT_STEP_RAD = 0.05
PART_SAMPLE_SPACING_M = 0.01
SCENE_SAMPLE_SPACING_M = 0.01
IK_POSITION_TOLERANCE_M = 0.002
IK_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
IK_ORIENTATION_WEIGHT_M = 0.3
IK_MAX_ITERATIONS = 200
# Arm poses each grip's path is also solved from, after any supplied seed: a mid-range pose, and the
# runtime asset's rest pose. A seven-axis arm holds one grip with its elbow in different places, and
# the path keeps the elbow where the first solve put it; on 2026-10-26 the dishwasher's middle grip
# struck the door from the reach check's solution and cleared it by 9 cm from either of these.
NOMINAL_SEED_RAD = (0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.785)
REST_SEED_RAD = {"franka_panda": (0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0)}


class RobotArmPathClearanceError(ValueError):
    """Stable refusal codes."""


class RobotArmPathAdmissionBlocked(RobotArmPathClearanceError):
    """A paid native run would start where the task's moving part strikes the arm."""

    def __init__(self, blockers: Sequence[str]):
        super().__init__(",".join(blockers))
        self.blockers = list(blockers)


def _directions() -> np.ndarray:
    """The 48 outward directions each link bound is measured along: the axes, an icosahedron's
    vertices and its edge midpoints."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    ico = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
                    (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    gap = np.linalg.norm(ico[:, None] - ico[None], axis=2)
    edge = gap[gap > 1e-9].min()
    mids = np.asarray([ico[a] + ico[b] for a in range(12) for b in range(a + 1, 12)
                       if abs(gap[a, b] - edge) < 1e-9])
    return np.vstack([np.eye(3), -np.eye(3), ico, mids / np.linalg.norm(mids, axis=1, keepdims=True)])


DIRECTIONS = _directions()

# Each arm link's outer bound in its link frame: its collision meshes' largest extent along each of
# ``DIRECTIONS``, rounded up to 0.1 mm. Measured with ``measure_arm_link_bounds`` from the runtime USD named
# by ``measured_from_sha256``; a bound holds its link and is at most a few millimetres larger. The grasp
# point is the midpoint of the open fingertip pads in the flange (link 8) frame.
ARM_LINK_BOUNDS: dict[str, dict[str, Any]] = {
    "franka_panda": {
        "measured_from_sha256": "sha256:c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c",
        "grasp_point_flange_m": (0.0, 0.0, 0.1493),
        # Every link bound lies within this distance of the base origin at any joint value: 1.279 m was the
        # farthest over 20,000 joint values drawn uniformly within the limits.
        "reach_radius_m": 1.35,
        "links": {
            "panda_link1": (1, (0.0550, 0.0551, 0.0550, 0.0551, 0.1294, 0.1921, 0.0550, 0.0549, 0.1191, 0.1189,
                                0.0957, 0.0466, 0.1923, 0.1923, 0.1477, 0.0550, 0.1478, 0.0550, 0.0551, 0.0320,
                                0.1436, 0.1116, 0.0474, 0.0319, 0.1437, 0.1116, 0.0474, 0.1294, 0.1163, 0.1437,
                                0.1116, 0.0937, 0.1162, 0.1436, 0.1116, 0.0937, 0.0550, 0.0773, 0.0773, 0.0522,
                                0.0522, 0.1921, 0.1876, 0.1877, 0.1877, 0.1877, 0.0550, 0.0551)),
            "panda_link2": (2, (0.0550, 0.0549, 0.1294, 0.0551, 0.1941, 0.0551, 0.0549, 0.0549, 0.1940, 0.1939,
                                0.1488, 0.1191, 0.1488, 0.0285, 0.0550, 0.0957, 0.0550, 0.0958, 0.0549, 0.0936,
                                0.0474, 0.0521, 0.0773, 0.0937, 0.0473, 0.0521, 0.0772, 0.1941, 0.1892, 0.1893,
                                0.1447, 0.1446, 0.1893, 0.1892, 0.1446, 0.1446, 0.1294, 0.1160, 0.1162, 0.1160,
                                0.1162, 0.0551, 0.1122, 0.1122, 0.0319, 0.0319, 0.0550, 0.0551)),
            "panda_link3": (3, (0.1375, 0.1111, 0.0550, 0.0551, 0.0551, 0.1211, 0.0618, 0.1485, 0.0550, 0.0719,
                                0.0465, 0.0876, 0.1319, 0.1319, 0.1252, 0.1252, 0.1105, 0.0190, 0.1111, 0.0777,
                                0.1082, 0.0897, 0.0310, 0.1288, 0.1287, 0.1526, 0.1527, 0.0551, 0.0204, 0.1081,
                                0.0897, 0.0310, 0.0574, 0.1081, 0.1142, 0.1141, 0.0550, 0.0934, 0.0109, 0.1133,
                                0.0310, 0.1211, 0.1303, 0.1302, 0.1303, 0.1303, 0.1375, 0.0551)),
            "panda_link4": (4, (0.0550, 0.1240, 0.1111, 0.1376, 0.0549, 0.0551, 0.1778, 0.0910, 0.0550, 0.0550,
                                0.1051, 0.1120, 0.0285, 0.1120, 0.0466, 0.0876, 0.1252, 0.1252, 0.1240, 0.1581,
                                0.1581, 0.1764, 0.1764, 0.1071, 0.1071, 0.0522, 0.0722, 0.0549, 0.0859, 0.0473,
                                0.0871, 0.0871, 0.0860, 0.0474, 0.0521, 0.0720, 0.1111, 0.1032, 0.1032, 0.1032,
                                0.1319, 0.0551, 0.0319, 0.0723, 0.0494, 0.1319, 0.0550, 0.1376)),
            "panda_link5": (5, (0.0550, 0.1300, 0.0522, 0.0550, 0.0550, 0.2590, 0.1212, 0.1212, 0.0551, 0.0551,
                                0.0325, 0.0961, 0.2493, 0.2493, 0.1830, 0.0515, 0.1830, 0.0515, 0.1300, 0.1186,
                                0.1771, 0.1323, 0.0951, 0.1186, 0.1771, 0.1323, 0.0951, 0.0550, 0.0154, 0.1771,
                                0.1324, 0.0336, 0.0154, 0.1771, 0.1324, 0.0336, 0.0522, 0.0407, 0.0407, 0.0800,
                                0.0800, 0.2590, 0.2419, 0.2419, 0.2418, 0.2418, 0.0550, 0.0550)),
            "panda_link6": (6, (0.1321, 0.0817, 0.0563, 0.0480, 0.0512, 0.0440, 0.0479, 0.1206, 0.0480, 0.1130,
                                0.0644, 0.0686, 0.0644, 0.0686, 0.1189, 0.1189, 0.0482, 0.0551, 0.0817, 0.0558,
                                0.0557, 0.0502, 0.0535, 0.1043, 0.1043, 0.1341, 0.1341, 0.0512, 0.0550, 0.0489,
                                0.0502, 0.0535, 0.0945, 0.0945, 0.1350, 0.1349, 0.0563, 0.1017, 0.0563, 0.0993,
                                0.0563, 0.0440, 0.1017, 0.0402, 0.0993, 0.0402, 0.1321, 0.0480)),
            "panda_link7": (7, (0.0814, 0.0814, 0.1069, 0.0441, 0.0441, -0.0520, 0.0496, 0.0900, 0.0532, 0.0440,
                                0.1069, 0.1224, -0.0211, -0.0211, 0.0272, 0.1173, 0.0101, 0.0882, 0.0814, 0.1003,
                                0.0146, 0.0257, 0.0716, 0.1233, 0.0377, 0.0603, 0.1132, 0.0441, 0.0877, 0.0120,
                                0.0258, 0.0777, 0.0863, 0.0121, 0.0257, 0.0756, 0.1069, 0.1056, 0.1044, 0.1279,
                                0.1044, -0.0520, -0.0163, -0.0163, -0.0108, -0.0163, 0.0814, 0.0441)),
            "panda_link8": (8, (0.0376, 0.0375, 0.0181, 0.0375, 0.0375, -0.0000, 0.0375, 0.0375, 0.0375, 0.0375,
                                0.0351, 0.0351, 0.0198, 0.0198, 0.0319, 0.0414, 0.0319, 0.0414, 0.0375, 0.0415,
                                0.0325, 0.0357, 0.0413, 0.0415, 0.0325, 0.0357, 0.0413, 0.0375, 0.0415, 0.0325,
                                0.0357, 0.0413, 0.0415, 0.0325, 0.0357, 0.0413, 0.0181, 0.0366, 0.0366, 0.0366,
                                0.0366, -0.0000, 0.0221, 0.0221, 0.0221, 0.0221, 0.0376, 0.0375)),
        },
    },
}


def measure_arm_link_bounds(robot_usd: str | Path, links: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Each named link's outer bound from a robot USD's collision meshes, in that link's frame.

    How ``ARM_LINK_BOUNDS`` was measured: every Mesh or Cylinder under the link's prim (instance
    proxies included) is expressed in the link frame, and its largest extent along each of
    ``DIRECTIONS`` is rounded up to 0.1 mm.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(robot_usd))
    root = stage.GetDefaultPrim() if stage else None
    if not root:
        raise RobotArmPathClearanceError("robot_arm_path_robot_usd_invalid")
    cache = UsdGeom.XformCache()

    def matrix(prim: Any) -> np.ndarray:
        return np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T

    bounds = {}
    for name in links:
        link = next((p for p in Usd.PrimRange(root) if p.GetName() == name), None)
        if link is None:
            raise RobotArmPathClearanceError("robot_arm_path_link_missing:" + name)
        link_from_world = np.linalg.inv(matrix(link))
        parts = []
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
            elif prim.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(prim)
                radius, half = float(cylinder.GetRadiusAttr().Get()), 0.5 * float(cylinder.GetHeightAttr().Get())
                angles = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
                ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
                axis = "XYZ".index(str(cylinder.GetAxisAttr().Get()))
                points = np.concatenate([np.insert(ring, axis, z, axis=1) for z in (-half, half)])
            else:
                continue
            frame = link_from_world @ matrix(prim)
            parts.append(points @ frame[:3, :3].T + frame[:3, 3])
        if not parts:
            raise RobotArmPathClearanceError("robot_arm_path_link_geometry_missing:" + name)
        extent = (np.concatenate(parts) @ DIRECTIONS.T).max(axis=0)
        bounds[name] = tuple(float(value) for value in np.ceil(extent * 1e4) / 1e4)
    return bounds


@lru_cache(maxsize=None)
def _arm_model(robot_id: str) -> tuple[tuple[tuple[str, int, np.ndarray, np.ndarray, float], ...], np.ndarray]:
    """Per link: name, kinematic frame index, plane offsets, bound centre and radius; and the grasp point."""
    model = ARM_LINK_BOUNDS.get(robot_id)
    if model is None:
        raise RobotArmPathClearanceError("robot_arm_path_robot_unmeasured:" + str(robot_id))
    links = []
    for name, (frame, offsets) in model["links"].items():
        offsets = np.asarray(offsets, dtype=float)
        low, high = -offsets[3:6], offsets[0:3]
        centre = (low + high) / 2.0
        links.append((name, int(frame), offsets, centre, float(np.linalg.norm(high - centre))))
    return tuple(links), np.asarray(model["grasp_point_flange_m"], dtype=float)


_DH = np.asarray(FRANKA_MODIFIED_DH, dtype=float)
_FLANGE = np.eye(4)
_FLANGE[2, 3] = FRANKA_FLANGE_OFFSET_M


def _frames(joints: np.ndarray) -> np.ndarray:
    """The base, link 1-7 and flange frames: ``franka_kinematics._chain`` in array arithmetic.

    The solver evaluates it at every iteration, where the list version cost a third of the gate.
    """
    ct, st = np.cos(np.asarray(joints, dtype=float)), np.sin(np.asarray(joints, dtype=float))
    a, d, ca, sa = _DH[:, 0], _DH[:, 1], np.cos(_DH[:, 2]), np.sin(_DH[:, 2])
    links = np.zeros((7, 4, 4))
    links[:, 0, 0], links[:, 0, 1], links[:, 0, 3] = ct, -st, a
    links[:, 1, 0], links[:, 1, 1], links[:, 1, 2], links[:, 1, 3] = st * ca, ct * ca, -sa, -d * sa
    links[:, 2, 0], links[:, 2, 1], links[:, 2, 2], links[:, 2, 3] = st * sa, ct * sa, ca, d * ca
    links[:, 3, 3] = 1.0
    frames = np.empty((9, 4, 4))
    frames[0] = np.eye(4)
    for index in range(7):
        frames[index + 1] = frames[index] @ links[index]
    frames[8] = frames[7] @ _FLANGE
    return frames


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """A rotation matrix's axis times its angle."""
    cos = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    sin = float(np.linalg.norm(skew)) / 2.0
    angle = math.atan2(sin, cos)
    if angle < 1e-6:
        return skew / 2.0
    if math.pi - angle > 1e-3:
        return skew * (angle / (2.0 * sin))
    # Near a half turn the skew part vanishes; the axis is the symmetric part's dominant column.
    symmetric = (rotation + rotation.T) / 2.0 - cos * np.eye(3)
    column = int(np.argmax(np.diag(symmetric)))
    axis = symmetric[:, column] / math.sqrt(max(symmetric[column, column] * (1.0 - cos), 1e-300))
    return axis * angle if float(axis @ skew) >= 0.0 else -axis * angle


def solve_grasp_ik(*, base_from_world: np.ndarray, position_world: np.ndarray, rotation_world: np.ndarray,
                   seed: Sequence[float], grasp_point_flange: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Damped least-squares full-pose IK of the grasp point; returns joints and position/angle errors.

    ``rotation_world`` is the flange orientation: its z axis the approach, its y axis the fingers'
    closing direction.
    """
    target_p = base_from_world[:3, :3] @ position_world + base_from_world[:3, 3]
    target_r = base_from_world[:3, :3] @ rotation_world
    limits = np.asarray(FRANKA_JOINT_LIMITS_RAD, dtype=float)
    q = np.clip(np.asarray(seed, dtype=float), limits[:, 0], limits[:, 1])

    def state(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = _frames(joints)
        flange = frames[8]
        point = flange[:3, 3] + flange[:3, :3] @ grasp_point_flange
        error = np.r_[target_p - point, IK_ORIENTATION_WEIGHT_M * _rotation_vector(target_r @ flange[:3, :3].T)]
        return frames, point, error

    frames, point, error = state(q)
    damping = 0.05
    for _ in range(IK_MAX_ITERATIONS):
        if (np.linalg.norm(error[:3]) < IK_POSITION_TOLERANCE_M
                and np.linalg.norm(error[3:]) / IK_ORIENTATION_WEIGHT_M < IK_ORIENTATION_TOLERANCE_RAD):
            break
        axes, origins = frames[1:8, :3, 2], frames[1:8, :3, 3]
        jacobian = np.vstack([np.cross(axes, point - origins).T, IK_ORIENTATION_WEIGHT_M * axes.T])
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(6), error)
        largest = float(np.abs(step).max())
        if largest > 0.2:
            step *= 0.2 / largest
        candidate = np.clip(q + step, limits[:, 0], limits[:, 1])
        candidate_frames, candidate_point, candidate_error = state(candidate)
        if np.linalg.norm(candidate_error) < np.linalg.norm(error):
            q, frames, point, error = candidate, candidate_frames, candidate_point, candidate_error
            damping = max(damping * 0.7, 1e-3)
        else:
            damping = min(damping * 2.0, 1.0)
            if damping >= 1.0 and np.linalg.norm(step) < 1e-6:
                break
    return q, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]) / IK_ORIENTATION_WEIGHT_M)


def _unit(vector: np.ndarray, code: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise RobotArmPathClearanceError(code)
    return vector / norm


def _about(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _grip_rows(free: tuple[float, float]) -> list[tuple[float, float]]:
    """Slides along the handle and turns about it, nearest the planned grip first."""
    rows = []
    for fraction in GRIP_SLIDE_FRACTIONS:
        slide = fraction * (free[1] if fraction > 0 else free[0])
        if fraction and abs(slide) < 0.01:
            continue
        rows.extend((slide, roll) for roll in GRIP_ROLLS_DEG)
    return rows


@dataclass(frozen=True)
class SceneObstacles:
    """The static scene around a robot, as surface samples the arm links are measured against."""

    points: np.ndarray
    tree: Any
    digest: str

    def near(self, centre: np.ndarray, radius: float) -> np.ndarray:
        return self.points[self.tree.query_ball_point(centre, radius)]


def scene_radius_m(robot_id: str) -> float:
    """How far from the robot's base origin the scene can come within the margin of an arm link."""
    model = ARM_LINK_BOUNDS.get(robot_id)
    if model is None:
        raise RobotArmPathClearanceError("robot_arm_path_robot_unmeasured:" + str(robot_id))
    return float(model["reach_radius_m"]) + DEFAULT_MARGIN_M


def sample_scene_obstacles(triangles: np.ndarray, *, centre: Sequence[float], radius: float,
                    spacing: float = SCENE_SAMPLE_SPACING_M) -> SceneObstacles:
    """Samples no further apart than ``spacing`` on ``triangles``, within ``radius`` of ``centre``.

    A big triangle is cut to the region first, so a floor many metres across is not sampled in full.
    """
    from scipy.spatial import cKDTree

    centre = np.asarray(centre, dtype=float)
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    inside = (np.all(triangles.max(axis=1) >= centre - radius, axis=1)
              & np.all(triangles.min(axis=1) <= centre + radius, axis=1))
    try:
        points, _ = sample_surface(_clip_large_triangles(triangles[inside], centre - radius, centre + radius),
                                   spacing, maximum_samples=MAXIMUM_BACKGROUND_SAMPLES)
    except ScenePhysicsPreflightError as exc:
        raise RobotArmPathClearanceError(f"robot_arm_path_scene_unmeasurable:{exc}") from exc
    points = np.ascontiguousarray(points[np.linalg.norm(points - centre, axis=1) <= radius])
    return SceneObstacles(points=points, tree=cKDTree(points),
                          digest="sha256:" + hashlib.sha256(points.tobytes()).hexdigest())


def arm_path_clearance(*, robot_id: str, base_pose_world: Mapping[str, Sequence[float]], asset_usd: str | Path,
                       asset_pose_world: Mapping[str, Sequence[float]], joint_id: str, task_range: Sequence[float],
                       reset_joint_positions: Mapping[str, float] | None, contact_point_world: Sequence[float],
                       approach_point_world: Sequence[float], seed_joint_positions: Sequence[float] | None = None,
                       margin_m: float = DEFAULT_MARGIN_M,
                       scene_obstacles: SceneObstacles | None = None) -> dict[str, Any]:
    """Measure the grip family from one base pose; see the module docstring."""
    links, grasp_point = _arm_model(robot_id)
    try:
        model = load_asset_model(asset_usd)
        joint = _movable_joint(model, joint_id)
    except ScenePhysicsPreflightError as exc:
        raise RobotArmPathClearanceError(f"robot_arm_path_task_part_unmeasurable:{exc}") from exc
    reset = {str(k): float(v) for k, v in (reset_joint_positions or {}).items()}
    start = float(reset.get(joint_id, 0.0))
    low, high = sorted(float(v) for v in task_range)
    goal = high if abs(high - start) >= abs(low - start) else low
    world_from_asset = _pose_matrix(asset_pose_world)
    world_from_base = _pose_matrix(base_pose_world)
    base_from_world = np.linalg.inv(world_from_base)
    moving = descendants(model, joint_id)

    def world_poses(value: float) -> dict[str, np.ndarray]:
        return {b: world_from_asset @ p for b, p in body_poses(model, {**reset, joint_id: value}).items()}

    rest = world_poses(start)
    hinge = (world_from_asset @ model.asset_from_body[joint.parent] @ joint.parent_from_joint)[:3, :3] @ joint.axis
    bar = _unit(hinge if joint.kind == "revolute" else np.cross([0.0, 0.0, 1.0], hinge),
                "robot_arm_path_handle_axis_undefined")
    contact = np.asarray(contact_point_world, dtype=float)
    approach_out = np.asarray(approach_point_world, dtype=float) - contact
    standoff = float(np.linalg.norm(approach_out))
    approach_out = _unit(approach_out - (approach_out @ bar) * bar, "robot_arm_path_approach_invalid")
    # The handle's free length: the moving collider holding the contact, along the handle, less a finger.
    colliders = [(c, rest[c.body] @ c.body_from_collider) for c in model.colliders if c.body in moving]
    if not colliders:
        raise RobotArmPathClearanceError("robot_arm_path_task_part_colliders_missing")
    held, frame = max(colliders, key=lambda row: float(_depths(row[0], ((contact - row[1][:3, 3]) @ row[1][:3, :3])[None])[0]))
    extent = (held.local_corners() @ frame[:3, :3].T + frame[:3, 3] - contact) @ bar
    free = (max(0.0, -float(extent.min()) - FINGER_HALF_WIDTH_M), max(0.0, float(extent.max()) - FINGER_HALF_WIDTH_M))
    # The part's surface in each moving body's frame, placed per joint value.
    local: dict[str, list[np.ndarray]] = {}
    for collider in model.colliders:
        if collider.body in moving:
            points = _collider_surface(collider, PART_SAMPLE_SPACING_M)
            local.setdefault(collider.body, []).append(
                points @ collider.body_from_collider[:3, :3].T + collider.body_from_collider[:3, 3])
    local_points = {body: np.concatenate(parts) for body, parts in local.items()}

    def part_points(value: float) -> np.ndarray:
        poses = world_poses(value)
        return np.concatenate([p @ poses[b][:3, :3].T + poses[b][:3, 3] for b, p in local_points.items()])

    # The static scene: what the caller sampled, and the task asset's own fixed bodies -- the cabinet
    # around a door -- which only this gate knows apart from the parts that move.
    static, static_count = [], 0
    if scene_obstacles is not None:
        from scipy.spatial import cKDTree

        fixed = [(c, rest[c.body] @ c.body_from_collider) for c in model.colliders if c.body not in moving]
        fixed_points = np.concatenate([_collider_surface(c, PART_SAMPLE_SPACING_M) @ frame[:3, :3].T + frame[:3, 3]
                                       for c, frame in fixed]) if fixed else np.zeros((0, 3))
        fixed_tree = cKDTree(fixed_points)
        static = [scene_obstacles.near, lambda centre, radius: fixed_points[fixed_tree.query_ball_point(centre, radius)]]
        static_count = len(scene_obstacles.points) + len(fixed_points)

    def clearance(joints: np.ndarray, value: float) -> tuple[float, str | None, str | None]:
        """The least gap from an arm link to the part or the scene, the link, and which it was."""
        frames = world_from_base @ _frames(joints)
        points = part_points(value)
        least, which, against = math.inf, None, None
        for name, index, offsets, centre, radius in links:
            frame = frames[index]
            middle = frame[:3, :3] @ centre + frame[:3, 3]
            clouds = [("task_part", points[np.linalg.norm(points - middle, axis=1) <= radius + margin_m])]
            clouds += [("scene", lookup(middle, radius + margin_m)) for lookup in static]
            for obstacle, near in clouds:
                if not len(near):
                    continue
                gap = float(np.min(np.max(((near - frame[:3, 3]) @ frame[:3, :3]) @ DIRECTIONS.T - offsets, axis=1)))
                if gap < least:
                    least, which, against = gap, name, obstacle
        return least, which, against

    step = REVOLUTE_STEP_RAD if joint.kind == "revolute" else PRISMATIC_STEP_M
    values = [float(v) for v in np.linspace(start, goal, max(2, int(math.ceil(abs(goal - start) / step)) + 1))]
    carried = {v: world_poses(v)[joint.child] @ np.linalg.inv(rest[joint.child]) for v in values}
    flip = np.diag([-1.0, -1.0, 1.0])  # the same grip with the hand turned half a turn about the approach
    seeds = []
    for seed in (seed_joint_positions, NOMINAL_SEED_RAD, REST_SEED_RAD.get(robot_id)):
        if seed is not None and tuple(float(v) for v in seed) not in seeds:
            seeds.append(tuple(float(v) for v in seed))

    def follow(poses: list, seed: Sequence[float]) -> tuple[list, float | None]:
        """Solve each pose from the previous solution; the joint value where the arm first falls short."""
        path, previous = [], np.asarray(seed, dtype=float)
        for position, orientation, value in poses:
            joints, position_error, angle_error = solve_grasp_ik(
                base_from_world=base_from_world, position_world=position, rotation_world=orientation,
                seed=previous, grasp_point_flange=grasp_point)
            if position_error >= IK_POSITION_TOLERANCE_M or angle_error >= IK_ORIENTATION_TOLERANCE_RAD:
                return path, value
            path.append((joints, value))
            previous = joints
        return path, None

    def measure(path: list) -> tuple[float | None, str | None, float | None, str | None]:
        least, which, where, what = None, None, None, None
        checks = [path[0]]
        for (qa, va), (qb, vb) in zip(path, path[1:]):
            count = max(1, int(math.ceil(float(np.abs(qb - qa).max()) / MAXIMUM_JOINT_STEP_RAD)))
            checks += [(qa + t * (qb - qa), va + t * (vb - va)) for t in np.linspace(1.0 / count, 1.0, count)]
        for joints, value in checks:
            gap, link, against = clearance(joints, value)
            if math.isfinite(gap) and (least is None or gap < least):
                least, which, where, what = round(gap, 4), link, round(float(value), 4), against
        return least, which, where, what

    grips, best = [], None
    for slide, roll_deg in _grip_rows(free):
        approach_in = _about(bar, math.radians(roll_deg)) @ -approach_out
        grip = contact + slide * bar
        row = {"slide_m": round(slide, 4), "roll_deg": roll_deg, "half_turn": False, "reached": False,
               "clear": False, "minimum_clearance_m": None, "at_joint_value": None, "link": None, "obstacle": None}
        for half_turn, turn in ((False, np.eye(3)), (True, flip)):
            rotation = np.column_stack([bar, np.cross(approach_in, bar), approach_in]) @ turn
            poses = [(grip - standoff * approach_in, rotation, start)]
            poses += [(carried[v][:3, :3] @ grip + carried[v][:3, 3], carried[v][:3, :3] @ rotation, v) for v in values]
            for seed in seeds:
                path, short = follow(poses, seed)
                if short is not None:
                    if not row["reached"]:
                        row["at_joint_value"] = round(short, 4)
                    continue
                least, link, where, against = measure(path)
                clear = least is None or least >= margin_m
                if not row["reached"] or clear or (least is not None and row["minimum_clearance_m"] is not None
                                                   and least > row["minimum_clearance_m"]):
                    row.update(reached=True, clear=clear, minimum_clearance_m=least, link=link, at_joint_value=where,
                               half_turn=half_turn, obstacle=against)
                if clear:
                    break
            # The hand turned half a turn holds the same grip with the wrist elsewhere: try it whenever the
            # first turn did not both reach and clear.
            if row["clear"]:
                break
        grips.append(row)
        if row["clear"]:
            best = row
            break

    blockers = []
    if best is None:
        reached = [row for row in grips if row["reached"]]
        if not reached:
            blockers.append("robot_arm_path_grip_unreachable")
        else:
            nearest = max(reached, key=lambda row: row["minimum_clearance_m"])
            value = nearest["at_joint_value"]
            where = "%.0fdeg" % math.degrees(value) if joint.kind == "revolute" else "%.3fm" % value
            kind = "robot_arm_in_scene_path" if nearest["obstacle"] == "scene" else "robot_arm_in_task_part_path"
            blockers.append(f"{kind}:{joint_id}@{where}:{nearest['link']}")
    result = {
        "schema_version": SCHEMA_VERSION, "robot_id": robot_id,
        "measured_from_sha256": ARM_LINK_BOUNDS[robot_id]["measured_from_sha256"],
        "status": "passed" if best is not None else "blocked", "blockers": blockers,
        "margin_m": margin_m, "joint_id": joint_id, "joint_span": [start, goal],
        "handle_axis_world": [float(v) for v in bar], "handle_free_length_m": [round(v, 4) for v in free],
        "grips_tried": len(grips), "best_grip": best, "grips": grips,
        "scene_checked": scene_obstacles is not None, "scene_sample_count": static_count,
        "scene_obstacles_digest": scene_obstacles.digest if scene_obstacles is not None else None,
        "simulator_executed": False, "report_digest": "",
    }
    result["report_digest"] = canonical_digest(result, digest_field="report_digest")
    return result


#: A rigid object is carried along the robot-free probe's path: straight up from its start, across
#: above the higher of its two ends by this much, and down onto its target.
RIGID_CARRY_CLEARANCE_M = 0.08
RIGID_PATH_STEP_M = 0.02
RIGID_GRIP_ROLLS_DEG = (0.0, 45.0, -45.0, 90.0)
#: A larger change in any joint between two path samples is a jump between arm configurations.
RIGID_MAXIMUM_JOINT_STEP_RAD = 0.35


def _segment(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    count = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / RIGID_PATH_STEP_M)))
    return [a + (b - a) * (k / count) for k in range(1, count + 1)]


def rigid_pick_place_clearance(*, robot_id: str, base_pose_world: Mapping[str, Sequence[float]],
                               start_pose_world: Sequence[float], target_position_world: Sequence[float],
                               contact_point_m: Sequence[float], approach_unit: Sequence[float],
                               gripper_orientation_xyzw: Sequence[float], pregrasp_clearance_m: float,
                               seeds: Sequence[Sequence[float]], margin_m: float = DEFAULT_MARGIN_M,
                               scene_obstacles: SceneObstacles | None = None) -> dict[str, Any]:
    """Whether the arm can carry a rigid object from its start to its target, clear of the static scene.

    The hand holds the task's own grasp (``contact_point_m`` and ``gripper_orientation_xyzw`` in the
    object's scoring frame, backing off along ``approach_unit``), turned about the approach by a few
    rolls. It comes down to the object, lifts it straight up, carries it across above the higher end,
    lowers it onto the target and backs off. The arm's pose is solved every ``RIGID_PATH_STEP_M`` from
    the previous solution, from each seed, and every arm link keeps ``margin_m`` from the scene. A
    pass means one such motion exists, not that a policy finds it; the object itself is not measured.
    """
    links, grasp_point = _arm_model(robot_id)
    world_from_base = _pose_matrix(base_pose_world)
    base_from_world = np.linalg.inv(world_from_base)
    pose = [float(v) for v in start_pose_world]
    held = _pose_matrix({"position_world_m": pose[:3], "orientation_xyzw": pose[3:7]})[:3, :3]
    contact = held @ np.asarray(contact_point_m, dtype=float)
    outward = _unit(held @ np.asarray(approach_unit, dtype=float), "robot_rigid_reach_approach_invalid")
    grip = held @ _pose_matrix({"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": gripper_orientation_xyzw})[:3, :3]
    pick = np.asarray(pose[:3], dtype=float) + contact
    place = np.asarray(target_position_world, dtype=float) + contact
    height = max(float(pick[2]), float(place[2])) + RIGID_CARRY_CLEARANCE_M
    corners = [pick + pregrasp_clearance_m * outward, pick, np.array([pick[0], pick[1], height]),
               np.array([place[0], place[1], height]), place, place + pregrasp_clearance_m * outward]
    phases = ("approach", "lift", "carry", "lower", "retreat")
    path = [(phases[0], corners[0])] + [(phase, point) for phase, a, b in zip(phases, corners, corners[1:])
                                        for point in _segment(a, b)]
    lookups = [scene_obstacles.near] if scene_obstacles is not None else []

    def gap(joints: np.ndarray) -> tuple[float, str | None]:
        frames = world_from_base @ _frames(joints)
        least, which = math.inf, None
        for name, index, offsets, centre, radius in links:
            frame = frames[index]
            middle = frame[:3, :3] @ centre + frame[:3, 3]
            for lookup in lookups:
                near = lookup(middle, radius + margin_m)
                if len(near):
                    value = float(np.min(np.max(((near - frame[:3, 3]) @ frame[:3, :3]) @ DIRECTIONS.T - offsets, axis=1)))
                    if value < least:
                        least, which = value, name
        return least, which

    # The attempt that got furthest names the refusal; a pose reached but struck outranks one not reached.
    best: tuple[int, bool, str] = (-1, False, "unreachable:approach")
    for roll in RIGID_GRIP_ROLLS_DEG:
        rotation = _about(grip[:, 2], math.radians(roll)) @ grip
        for seed_index, seed in enumerate(seeds):
            q, least, which, failure = np.asarray(seed, dtype=float), math.inf, None, None
            for index, (phase, point) in enumerate(path):
                solved, position_error, angle_error = solve_grasp_ik(
                    base_from_world=base_from_world, position_world=point, rotation_world=rotation, seed=q,
                    grasp_point_flange=grasp_point)
                if (position_error > IK_POSITION_TOLERANCE_M or angle_error > IK_ORIENTATION_TOLERANCE_RAD
                        or (index and float(np.abs(solved - q).max()) > RIGID_MAXIMUM_JOINT_STEP_RAD)):
                    failure = (index, False, "unreachable:" + phase)
                    break
                value, link = gap(solved)
                if value < margin_m:
                    failure = (index, True, f"scene_strike:{link}:{phase}")
                    break
                q = solved
                if value < least:
                    least, which = value, link
            if failure is None:
                return {"schema_version": SCHEMA_VERSION, "status": "clear", "task_kind": "rigid_pick_place",
                        "robot_id": robot_id, "margin_m": margin_m, "blockers": [],
                        "cleared_grip": {"roll_deg": roll, "seed_index": seed_index,
                                         "least_scene_gap_m": None if math.isinf(least) else least,
                                         "least_scene_gap_link": which},
                        "path_samples": len(path), "scene_obstacles_measured": scene_obstacles is not None}
            best = max(best, failure)
    return {"schema_version": SCHEMA_VERSION, "status": "blocked", "task_kind": "rigid_pick_place",
            "robot_id": robot_id, "margin_m": margin_m, "blockers": ["robot_rigid_pick_place_" + best[2]],
            "furthest_sample": best[0], "path_samples": len(path), "cleared_grip": None,
            "scene_obstacles_measured": scene_obstacles is not None}


def _packet_rigid_clearance(root: Path, plan: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The rigid gate on a packet; ``None`` for a robot this module has no arm model of (a team's own)."""
    robot = plan.get("robot") or {}
    robot_id = str(robot.get("robot_id"))
    if robot_id not in ARM_LINK_BOUNDS:
        return None
    try:
        affordance = spec["interaction_affordance"]
        reset = robot["joint_reset_positions_rad"]
        seeds = [[float(reset[f"panda_joint{i}"]) for i in range(1, 8)], list(NOMINAL_SEED_RAD),
                 list(REST_SEED_RAD.get(robot_id, NOMINAL_SEED_RAD))]
        inputs = {"start_pose_world": spec["start_pose_world"], "target_position_world": spec["target_position_world_m"],
                  "contact_point_m": affordance["contact_point_scoring_frame_m"],
                  "approach_unit": affordance["approach_unit_scoring_frame"],
                  "gripper_orientation_xyzw": affordance["gripper_orientation_scoring_frame_xyzw"],
                  "pregrasp_clearance_m": float(affordance["pregrasp_clearance_m"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise RobotArmPathClearanceError("robot_rigid_reach_task_affordance_invalid") from exc
    scenes = [o for o in plan.get("objects") or [] if isinstance(o, Mapping) and o.get("semantic_role") == "scene_collision"]
    scene_usd = (root / str(scenes[0].get("usd_path") or "")).resolve() if len(scenes) == 1 else None
    if (scene_usd is None or root.resolve() not in scene_usd.parents or not scene_usd.is_file()
            or _file_sha256(scene_usd) != scenes[0].get("sha256")):
        raise RobotArmPathClearanceError("robot_arm_path_scene_collision_unbound")
    centre = _pose_matrix(robot["base_pose_world"])[:3, 3]
    radius = scene_radius_m(robot_id)
    try:
        triangles = load_background_triangles(scene_usd, centre=centre, radius=radius,
                                              world_from_stage=_pose_matrix(scenes[0]["pose_world"]))
    except (ScenePhysicsPreflightError, KeyError, TypeError, ValueError) as exc:
        raise RobotArmPathClearanceError(f"robot_arm_path_scene_unmeasurable:{exc}") from exc
    return rigid_pick_place_clearance(robot_id=robot_id, base_pose_world=robot["base_pose_world"], seeds=seeds,
                                      scene_obstacles=sample_scene_obstacles(triangles, centre=centre, radius=radius),
                                      **inputs)


def packet_arm_path_clearance(packet_dir: str | Path) -> dict[str, Any]:
    """The gate on a compiled native packet, read from its scene plan alone.

    The base is the plan's robot pose and the part is the task subject at its reset. The contact and
    approach are the task's interaction affordance: the point on the contact link, the outward approach
    and the pre-contact clearance, the same contact the placement trajectory plans. The arm starts from
    the plan's reset pose. The static scene is the plan's one scene collision asset, the one the
    simulator loads. A plan that moves no part is not applicable.
    """
    import json

    from .articulated_control_planner import DEFAULT_APPROACH_STANDOFF_M
    from .decision_evidence_contracts import canonical_digest as digest

    root = Path(packet_dir)
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())

    def not_applicable(reason: str) -> dict[str, Any]:
        result = {"schema_version": SCHEMA_VERSION, "status": "not_applicable", "reason": reason,
                  "arena_scene_plan_digest": plan.get("plan_digest"), "blockers": [], "report_digest": ""}
        result["report_digest"] = digest(result, digest_field="report_digest")
        return result

    articulated = [o for o in plan.get("objects") or [] if isinstance(o, Mapping) and o.get("object_type") == "ARTICULATION"]
    if not articulated:
        spec = plan.get("task_spec") or {}
        if spec.get("manipulation_strategy") != "pick_and_place":
            return not_applicable("no_articulated_object")
        report = _packet_rigid_clearance(root, plan, spec)
        if report is None:
            return not_applicable("robot_arm_unmodeled")
        report = {**report, "arena_scene_plan_digest": plan.get("plan_digest"), "report_digest": ""}
        report["report_digest"] = digest(report, digest_field="report_digest")
        return report
    try:
        ranges = task_motion_ranges(plan)
    except ScenePhysicsPreflightError as exc:
        raise RobotArmPathClearanceError(f"robot_arm_path_task_range_unknown:{exc}") from exc
    if not ranges:
        return not_applicable("task_moves_no_part")
    subjects = [o for o in articulated if o.get("task_subject") is True or o.get("semantic_role") == "task_object"]
    if len(subjects) != 1 or len(ranges) != 1:
        raise RobotArmPathClearanceError("robot_arm_path_task_subject_ambiguous")
    subject, ((joint_id, span),) = subjects[0], ranges.items()
    asset = (root / str(subject.get("usd_path") or "")).resolve()
    if root.resolve() not in asset.parents or not asset.is_file() or _file_sha256(asset) != subject.get("sha256"):
        raise RobotArmPathClearanceError("robot_arm_path_task_asset_unbound")
    reset = subject.get("reset_state") or {}
    pose = reset.get("root_pose_world") or subject["pose_world"]
    affordance = (plan.get("task_spec") or {}).get("interaction_affordance") or {}
    try:
        (link,) = affordance["contact_body_prim_paths"]
        point = np.asarray(affordance["contact_point_link_m"], dtype=float)
        outward = np.asarray(affordance["approach_unit_asset_root"], dtype=float)
        standoff = float(affordance.get("precontact_clearance_m", DEFAULT_APPROACH_STANDOFF_M))
        robot = plan["robot"]
        seed = [float(robot["joint_reset_positions_rad"][f"panda_joint{i}"]) for i in range(1, 8)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RobotArmPathClearanceError("robot_arm_path_contact_affordance_invalid") from exc
    try:
        model = load_asset_model(asset)
        world_from_link = _pose_matrix(pose) @ body_poses(model, reset.get("joint_positions"))[str(link)]
    except (ScenePhysicsPreflightError, KeyError) as exc:
        raise RobotArmPathClearanceError(f"robot_arm_path_contact_link_unmeasurable:{exc}") from exc
    contact = world_from_link[:3, :3] @ point + world_from_link[:3, 3]
    approach = contact + standoff * (_pose_matrix(pose)[:3, :3] @ outward)
    scenes = [o for o in plan.get("objects") or [] if isinstance(o, Mapping) and o.get("semantic_role") == "scene_collision"]
    scene_usd = (root / str(scenes[0].get("usd_path") or "")).resolve() if len(scenes) == 1 else None
    if (scene_usd is None or root.resolve() not in scene_usd.parents or not scene_usd.is_file()
            or _file_sha256(scene_usd) != scenes[0].get("sha256")):
        raise RobotArmPathClearanceError("robot_arm_path_scene_collision_unbound")
    centre = _pose_matrix(robot["base_pose_world"])[:3, 3]
    radius = scene_radius_m(str(robot.get("robot_id")))
    try:
        triangles = load_background_triangles(scene_usd, centre=centre, radius=radius,
                                              world_from_stage=_pose_matrix(scenes[0]["pose_world"]))
    except (ScenePhysicsPreflightError, KeyError, TypeError, ValueError) as exc:
        raise RobotArmPathClearanceError(f"robot_arm_path_scene_unmeasurable:{exc}") from exc
    report = arm_path_clearance(
        robot_id=str(robot.get("robot_id")), base_pose_world=robot["base_pose_world"], asset_usd=asset,
        asset_pose_world=pose, joint_id=joint_id, task_range=span, reset_joint_positions=reset.get("joint_positions"),
        contact_point_world=contact, approach_point_world=approach, seed_joint_positions=seed,
        scene_obstacles=sample_scene_obstacles(triangles, centre=centre, radius=radius))
    report = {**report, "arena_scene_plan_digest": plan.get("plan_digest"), "report_digest": ""}
    report["report_digest"] = digest(report, digest_field="report_digest")
    return report


def require_robot_arm_path_admission(packet_dir: str | Path) -> dict[str, Any]:
    """Refuse a paid native run whose packet the gate blocks or cannot measure."""
    try:
        report = packet_arm_path_clearance(packet_dir)
    except RobotArmPathClearanceError as exc:
        raise RobotArmPathAdmissionBlocked(["robot_arm_path_refused:" + str(exc)]) from exc
    if report["status"] == "blocked":
        raise RobotArmPathAdmissionBlocked(report["blockers"])
    return report
