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
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .franka_kinematics import FRANKA_JOINT_LIMITS_RAD, _chain
from .scene_physics_preflight import (
    ScenePhysicsPreflightError,
    _collider_surface,
    _depths,
    _movable_joint,
    _pose_matrix,
    body_poses,
    descendants,
    load_asset_model,
)

SCHEMA_VERSION = "robot_arm_path_clearance.v1"
DEFAULT_MARGIN_M = 0.02
# Where along the handle's free half-length the hand closes, and how far it is turned about the handle.
GRIP_SLIDE_FRACTIONS = (0.0, -0.4, 0.4, -0.8, 0.8)
GRIP_ROLLS_DEG = (0.0, -15.0, 15.0, -30.0, 30.0)
FINGER_HALF_WIDTH_M = 0.02
REVOLUTE_STEP_RAD = math.radians(3.0)
PRISMATIC_STEP_M = 0.02
MAXIMUM_JOINT_STEP_RAD = 0.05
PART_SAMPLE_SPACING_M = 0.01
IK_POSITION_TOLERANCE_M = 0.002
IK_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
IK_ORIENTATION_WEIGHT_M = 0.3
IK_MAX_ITERATIONS = 200
# A fixed mid-range arm pose, tried after any supplied seed for the first grip pose.
NOMINAL_SEED_RAD = (0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.785)


class RobotArmPathClearanceError(ValueError):
    """Stable refusal codes."""


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


def _frames(joints: np.ndarray) -> np.ndarray:
    return np.asarray(_chain([float(value) for value in joints]), dtype=float)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotation).as_rotvec()


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
        jacobian = np.zeros((6, 7))
        for index in range(7):
            axis, origin = frames[index + 1][:3, 2], frames[index + 1][:3, 3]
            jacobian[:3, index] = np.cross(axis, point - origin)
            jacobian[3:, index] = IK_ORIENTATION_WEIGHT_M * axis
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


def arm_path_clearance(*, robot_id: str, base_pose_world: Mapping[str, Sequence[float]], asset_usd: str | Path,
                       asset_pose_world: Mapping[str, Sequence[float]], joint_id: str, task_range: Sequence[float],
                       reset_joint_positions: Mapping[str, float] | None, contact_point_world: Sequence[float],
                       approach_point_world: Sequence[float], seed_joint_positions: Sequence[float] | None = None,
                       margin_m: float = DEFAULT_MARGIN_M) -> dict[str, Any]:
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

    def clearance(joints: np.ndarray, value: float) -> tuple[float, str | None]:
        frames = world_from_base @ _frames(joints)
        points = part_points(value)
        least, which = math.inf, None
        for name, index, offsets, centre, radius in links:
            frame = frames[index]
            near = points[np.linalg.norm(points - (frame[:3, :3] @ centre + frame[:3, 3]), axis=1) <= radius + margin_m]
            if not len(near):
                continue
            gap = float(np.min(np.max(((near - frame[:3, 3]) @ frame[:3, :3]) @ DIRECTIONS.T - offsets, axis=1)))
            if gap < least:
                least, which = gap, name
        return least, which

    step = REVOLUTE_STEP_RAD if joint.kind == "revolute" else PRISMATIC_STEP_M
    values = [float(v) for v in np.linspace(start, goal, max(2, int(math.ceil(abs(goal - start) / step)) + 1))]
    carried = {v: world_poses(v)[joint.child] @ np.linalg.inv(rest[joint.child]) for v in values}
    flip = np.diag([-1.0, -1.0, 1.0])  # the same grip with the hand turned half a turn about the approach
    seeds = [s for s in (seed_joint_positions, NOMINAL_SEED_RAD) if s is not None]
    grips, best = [], None
    for slide, roll_deg in _grip_rows(free):
        approach_in = _about(bar, math.radians(roll_deg)) @ -approach_out
        grip = contact + slide * bar
        row = {"slide_m": round(slide, 4), "roll_deg": roll_deg, "reached": False, "clear": False,
               "minimum_clearance_m": None, "at_joint_value": None, "link": None}
        for turn in (np.eye(3), flip):
            rotation = np.column_stack([bar, np.cross(approach_in, bar), approach_in]) @ turn
            poses = [(grip - standoff * approach_in, rotation, start)]
            poses += [(carried[v][:3, :3] @ grip + carried[v][:3, 3], carried[v][:3, :3] @ rotation, v) for v in values]
            path, previous = [], None
            for position, orientation, value in poses:
                for seed in ([previous] if previous is not None else seeds):
                    joints, position_error, angle_error = solve_grasp_ik(
                        base_from_world=base_from_world, position_world=position, rotation_world=orientation,
                        seed=seed, grasp_point_flange=grasp_point)
                    if position_error < IK_POSITION_TOLERANCE_M and angle_error < IK_ORIENTATION_TOLERANCE_RAD:
                        break
                else:
                    row["at_joint_value"] = round(value, 4)
                    break
                path.append((joints, value))
                previous = joints
            if len(path) == len(poses):
                break
        if len(path) == len(poses):
            row["reached"], row["at_joint_value"] = True, None
            checks = [path[0]]
            for (qa, va), (qb, vb) in zip(path, path[1:]):
                count = max(1, int(math.ceil(float(np.abs(qb - qa).max()) / MAXIMUM_JOINT_STEP_RAD)))
                checks += [(qa + t * (qb - qa), va + t * (vb - va)) for t in np.linspace(1.0 / count, 1.0, count)]
            for joints, value in checks:
                gap, link = clearance(joints, value)
                if math.isfinite(gap) and (row["minimum_clearance_m"] is None or gap < row["minimum_clearance_m"]):
                    row.update(minimum_clearance_m=round(gap, 4), link=link, at_joint_value=round(float(value), 4))
            row["clear"] = row["minimum_clearance_m"] is None or row["minimum_clearance_m"] >= margin_m
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
            blockers.append(f"robot_arm_in_task_part_path:{joint_id}@{where}:{nearest['link']}")
    result = {
        "schema_version": SCHEMA_VERSION, "robot_id": robot_id,
        "measured_from_sha256": ARM_LINK_BOUNDS[robot_id]["measured_from_sha256"],
        "status": "passed" if best is not None else "blocked", "blockers": blockers,
        "margin_m": margin_m, "joint_id": joint_id, "joint_span": [start, goal],
        "handle_axis_world": [float(v) for v in bar], "handle_free_length_m": [round(v, 4) for v in free],
        "grips_tried": len(grips), "best_grip": best, "grips": grips,
        "simulator_executed": False, "report_digest": "",
    }
    result["report_digest"] = canonical_digest(result, digest_field="report_digest")
    return result
