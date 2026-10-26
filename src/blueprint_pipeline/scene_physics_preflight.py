"""CPU physics preflight for a composed scene, run before any paid simulator time.

Every fault the dishwasher scene showed on paid runs on 2026-10-25 was a geometry
fault a CPU can see in seconds: the appliance's colliders started inside the
reconstructed background (a 9.4 cm push under about 490 kN of contact), bay walls
stood inside its volume, and the door met the floor at its 90 degree limit. This
module reads the scene's static background collision and the placed articulated
asset from their USD files and measures two things:

* rest overlap: background surface inside the asset's solid colliders at the reset
  pose, with the shortest translation that clears it (the push a physics engine
  would apply to a free asset);
* motion clearance: each movable joint swept through its limits with every other
  joint at reset, reporting where its moving colliders first meet the background.

A collider is a solid (box, cylinder, sphere, or the convex hull of a mesh, which
is what PhysX uses for a dynamic mesh). Background surfaces are sampled densely
and each sample inside a solid counts with its depth to the solid's boundary.
Nothing here runs a simulator, calls a provider or changes an input file.

Every paid native bundle passes ``require_scene_physics_admission``. Rest overlap
and background inside the asset's envelope block it. A moving part's contact blocks
only inside the task's range: the target joint between its reset position and the
success threshold. Contact beyond that range, or on a joint the task does not
command, is reported without blocking.

The same sweep measures the robot: its base and shoulder, the parts no policy
action can move, block when they sit inside the asset at reset or in the target
joint's path before it reaches the success threshold, because then no successful
episode avoids striking them. Contact beyond the threshold is reported: a policy
can stop short of it, and runtime counterbalancing keeps a released door where it
is left. On 2026-10-26 the dishwasher door, free-falling after release at 30
degrees, came to rest on the arm's shoulder at 60 degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "scene_physics_preflight.v1"
DEFAULT_SAMPLE_SPACING_M = 0.005
DEFAULT_REST_TOLERANCE_M = 0.005
DEFAULT_MOTION_TOLERANCE_M = 0.002
DEFAULT_REVOLUTE_STEP_RAD = math.radians(2.0)
DEFAULT_PRISMATIC_STEP_M = 0.01
_PUSH_DIRECTIONS = 1024


class ScenePhysicsPreflightError(ValueError):
    """The scene cannot be measured: a missing, unsupported or malformed input."""


class ScenePhysicsAdmissionBlocked(ScenePhysicsPreflightError):
    """A paid native run would start from a physically invalid scene."""

    def __init__(self, blockers: Sequence[str]):
        super().__init__(",".join(blockers))
        self.blockers = list(blockers)


@dataclass(frozen=True)
class Collider:
    """One solid collider of the asset, in the frame of the body it belongs to."""

    prim_path: str
    body: str
    kind: str  # box | cylinder | sphere | hull
    body_from_collider: np.ndarray  # 4x4, rigid
    half_extents: np.ndarray | None = None  # box
    radius: float | None = None  # cylinder, sphere
    half_height: float | None = None  # cylinder (local z axis)
    planes: np.ndarray | None = None  # hull: rows (nx, ny, nz, offset) with n.x <= offset inside
    vertices: np.ndarray | None = None  # hull: its vertices, local
    bound_radius: float = 0.0

    def local_corners(self) -> np.ndarray:
        """Points whose box bounds the solid, in the collider's own frame."""
        if self.kind == "hull":
            return self.vertices
        half = (self.half_extents if self.kind == "box"
                else np.array([self.radius, self.radius, self.half_height if self.kind == "cylinder" else self.radius]))
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        return signs * half


@dataclass(frozen=True)
class Joint:
    joint_id: str
    kind: str  # revolute | prismatic | fixed
    parent: str
    child: str
    parent_from_joint: np.ndarray  # 4x4: the joint frame in the parent body
    axis: np.ndarray  # unit vector in the joint frame
    lower: float  # radians or metres
    upper: float


@dataclass(frozen=True)
class AssetModel:
    root_body: str
    asset_from_body: Mapping[str, np.ndarray]  # authored reset pose of every body
    colliders: tuple[Collider, ...]
    joints: tuple[Joint, ...]
    mounted: bool  # the root body is fixed to the world by an authored joint
    skipped_colliders: tuple[str, ...] = ()  # flat or empty hulls: no volume to overlap


def _rigid(matrix: np.ndarray, code: str) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 into a rigid transform and its per-axis scale; refuse shear."""
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= 0) or not np.isfinite(matrix).all():
        raise ScenePhysicsPreflightError(code)
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise ScenePhysicsPreflightError(code)
    rigid = np.eye(4)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = matrix[:3, 3]
    return rigid, scale


def _usd_matrix(cache: Any, prim: Any) -> np.ndarray:
    return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T


def _quaternion_matrix(quat: Any) -> np.ndarray:
    """A Gf quaternion (real first) as a 3x3 rotation."""
    w = float(quat.GetReal())
    x, y, z = (float(v) for v in quat.GetImaginary())
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ScenePhysicsPreflightError("scene_physics_joint_frame_invalid")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy.spatial import ConvexHull

    if len(points) < 4:
        raise ScenePhysicsPreflightError("scene_physics_mesh_collider_degenerate")
    try:
        hull = ConvexHull(points)
    except Exception as exc:  # qhull refuses flat or degenerate point sets
        raise ScenePhysicsPreflightError("scene_physics_mesh_collider_degenerate") from exc
    planes = np.column_stack([hull.equations[:, :3], -hull.equations[:, 3]])
    vertices = points[hull.vertices]
    return planes, vertices, float(np.linalg.norm(vertices, axis=1).max())


def load_asset_model(usd_path: str | Path) -> AssetModel:
    """Read bodies, solid colliders and joints of an articulated asset (Z up, metres)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim() if stage else None
    if not root or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ScenePhysicsPreflightError("scene_physics_asset_frame_invalid")
    cache = UsdGeom.XformCache()
    world_from_root = _usd_matrix(cache, root)
    # A scale above a body would change every collider and joint offset under it;
    # measuring it unscaled would check a different shape, so refuse it instead.
    if not np.allclose(_rigid(world_from_root, "scene_physics_asset_frame_invalid")[1], 1.0, atol=1e-6):
        raise ScenePhysicsPreflightError("scene_physics_asset_scale_unsupported:" + str(root.GetPath()))
    root_from_world = np.linalg.inv(world_from_root)
    prims = list(Usd.PrimRange(root))
    bodies = {str(p.GetPath()): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if not bodies:
        raise ScenePhysicsPreflightError("scene_physics_asset_bodies_missing")
    asset_from_body = {}
    for path, prim in bodies.items():
        asset_from_body[path], scale = _rigid(root_from_world @ _usd_matrix(cache, prim), "scene_physics_body_transform_invalid")
        if not np.allclose(scale, 1.0, atol=1e-6):
            raise ScenePhysicsPreflightError("scene_physics_asset_scale_unsupported:" + path)

    def owner(path: str) -> str:
        best = max((b for b in bodies if path == b or path.startswith(b + "/")), key=len, default=None)
        if best is None:
            raise ScenePhysicsPreflightError("scene_physics_collider_without_body")
        return best

    colliders, skipped = [], []
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        body = owner(str(prim.GetPath()))
        body_from_prim = np.linalg.inv(root_from_world @ _usd_matrix(cache, bodies[body])) @ (root_from_world @ _usd_matrix(cache, prim))
        frame, scale = _rigid(body_from_prim, "scene_physics_collider_transform_invalid")
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Cube):
            half = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * scale
            colliders.append(Collider(path, body, "box", frame, half_extents=half, bound_radius=float(np.linalg.norm(half))))
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            axis = "XYZ".index(str(cylinder.GetAxisAttr().Get()))
            radial = [i for i in range(3) if i != axis]
            if not math.isclose(scale[radial[0]], scale[radial[1]], rel_tol=1e-6):
                raise ScenePhysicsPreflightError("scene_physics_cylinder_scale_nonuniform")
            # Re-express the cylinder about its frame's local z axis.
            align = np.eye(4)
            if axis == 0:
                align[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
            elif axis == 1:
                align[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
            radius = float(cylinder.GetRadiusAttr().Get()) * float(scale[radial[0]])
            half_height = 0.5 * float(cylinder.GetHeightAttr().Get()) * float(scale[axis])
            colliders.append(Collider(path, body, "cylinder", frame @ align, radius=radius, half_height=half_height,
                                      bound_radius=math.hypot(radius, half_height)))
        elif prim.IsA(UsdGeom.Sphere):
            if not np.allclose(scale, scale[0], rtol=1e-6):
                raise ScenePhysicsPreflightError("scene_physics_sphere_scale_nonuniform")
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * float(scale[0])
            colliders.append(Collider(path, body, "sphere", frame, radius=radius, bound_radius=radius))
        elif prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3) * scale
            try:
                planes, vertices, bound = _hull(points)
            except ScenePhysicsPreflightError:
                skipped.append(path)
                continue
            colliders.append(Collider(path, body, "hull", frame, planes=planes, vertices=vertices, bound_radius=bound))
        else:
            raise ScenePhysicsPreflightError("scene_physics_collider_type_unsupported:" + prim.GetTypeName())

    joints = []
    children: set[str] = set()
    mounted = False
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if prim.IsA(UsdPhysics.FixedJoint) and len(body0) + len(body1) == 1:
            # One body and the world: an installed fixture's mount, not a link.
            mounted = True
            continue
        if len(body0) != 1 or len(body1) != 1 or str(body0[0]) not in bodies or str(body1[0]) not in bodies:
            raise ScenePhysicsPreflightError("scene_physics_joint_bodies_invalid")
        parent, child = str(body0[0]), str(body1[0])
        frame = np.eye(4)
        frame[:3, :3] = _quaternion_matrix(joint.GetLocalRot0Attr().Get())
        frame[:3, 3] = np.asarray(joint.GetLocalPos0Attr().Get(), dtype=float)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed = UsdPhysics.RevoluteJoint(prim)
            kind, scale_limit = "revolute", math.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed = UsdPhysics.PrismaticJoint(prim)
            kind, scale_limit = "prismatic", 1.0
        elif prim.IsA(UsdPhysics.FixedJoint):
            joints.append(Joint(prim.GetName(), "fixed", parent, child, frame, np.zeros(3), 0.0, 0.0))
            children.add(child)
            continue
        else:
            raise ScenePhysicsPreflightError("scene_physics_joint_type_unsupported:" + prim.GetTypeName())
        lower, upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
        if lower is None or upper is None or not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
            raise ScenePhysicsPreflightError("scene_physics_joint_limits_invalid:" + prim.GetName())
        axis = np.eye(3)["XYZ".index(str(typed.GetAxisAttr().Get()))]
        joints.append(Joint(prim.GetName(), kind, parent, child, frame, axis, float(lower) * scale_limit, float(upper) * scale_limit))
        children.add(child)
    roots = [b for b in bodies if b not in children]
    if len(roots) != 1:
        raise ScenePhysicsPreflightError("scene_physics_asset_root_ambiguous")
    return AssetModel(roots[0], asset_from_body, tuple(colliders), tuple(joints), mounted, tuple(skipped))


def body_poses(model: AssetModel, joint_values: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
    """Every body's pose in the asset frame with the named joints moved from reset."""
    values = dict(joint_values or {})
    poses = {model.root_body: model.asset_from_body[model.root_body]}
    pending = list(model.joints)
    while pending:
        progressed = False
        for joint in list(pending):
            if joint.parent not in poses:
                continue
            asset_from_joint = model.asset_from_body[joint.parent] @ joint.parent_from_joint
            joint_from_child = np.linalg.inv(asset_from_joint) @ model.asset_from_body[joint.child]
            motion = np.eye(4)
            q = float(values.get(joint.joint_id, 0.0))
            if q and joint.kind == "revolute":
                from scipy.spatial.transform import Rotation

                motion[:3, :3] = Rotation.from_rotvec(joint.axis * q).as_matrix()
            elif q and joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * q
            # The child keeps its authored offset from the joint frame, carried by
            # the parent's current pose and the joint's motion.
            current_joint = poses[joint.parent] @ joint.parent_from_joint
            poses[joint.child] = current_joint @ motion @ joint_from_child
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ScenePhysicsPreflightError("scene_physics_joint_tree_disconnected")
    return poses


def descendants(model: AssetModel, joint_id: str) -> set[str]:
    joint = next((j for j in model.joints if j.joint_id == joint_id), None)
    if joint is None:
        raise ScenePhysicsPreflightError("scene_physics_joint_unknown:" + joint_id)
    moving = {joint.child}
    grew = True
    while grew:
        grew = False
        for row in model.joints:
            if row.parent in moving and row.child not in moving:
                moving.add(row.child)
                grew = True
    return moving


def load_background_triangles(usd_path: str | Path, *, centre: Sequence[float], radius: float,
                              world_from_stage: np.ndarray | None = None) -> np.ndarray:
    """World triangles of every static collision mesh within ``radius`` of ``centre``."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if not stage or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ScenePhysicsPreflightError("scene_physics_background_frame_invalid")
    cache = UsdGeom.XformCache()
    centre = np.asarray(centre, dtype=float)
    kept = []
    meshes = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ScenePhysicsPreflightError("scene_physics_background_not_static")
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        meshes += 1
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if not len(points) or not len(counts):
            continue
        if np.all(counts == 3):
            faces = indices.reshape(-1, 3)
        else:
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            faces = np.asarray([[indices[s], indices[s + i], indices[s + i + 1]]
                                for s, c in zip(starts, counts) for i in range(1, c - 1)], dtype=np.int64)
        matrix = _usd_matrix(cache, prim)
        if world_from_stage is not None:
            matrix = world_from_stage @ matrix
        world = points @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = world[faces]
        near = np.linalg.norm(triangles.mean(axis=1) - centre, axis=1) <= radius + np.linalg.norm(
            triangles - triangles.mean(axis=1, keepdims=True), axis=2).max(axis=1)
        kept.append(triangles[near])
    if meshes == 0:
        raise ScenePhysicsPreflightError("scene_physics_background_collision_missing")
    return np.concatenate(kept) if kept else np.zeros((0, 3, 3))


def sample_surface(triangles: np.ndarray, spacing: float = DEFAULT_SAMPLE_SPACING_M) -> tuple[np.ndarray, np.ndarray]:
    """Points on every triangle no further apart than ``spacing``, and the triangle each came from."""
    if len(triangles) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    longest = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                               np.linalg.norm(a - c, axis=1)]), axis=0)
    divisions = np.maximum(1, np.ceil(longest / spacing)).astype(int)
    rows, owners = [], []
    for n in np.unique(divisions):
        chosen = np.flatnonzero(divisions == n)
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u, v = (i[keep] / n), (j[keep] / n)
        w = 1.0 - u - v
        ta, tb, tc = a[chosen], b[chosen], c[chosen]
        rows.append((w[None, :, None] * ta[:, None] + u[None, :, None] * tb[:, None]
                     + v[None, :, None] * tc[:, None]).reshape(-1, 3))
        owners.append(np.repeat(chosen, len(u)))
    return np.concatenate(rows), np.concatenate(owners)


def triangle_normals(triangles: np.ndarray) -> np.ndarray:
    """Unit normals; a degenerate triangle gets zeros."""
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1, keepdims=True)
    return np.divide(cross, length, out=np.zeros_like(cross), where=length > 0)


def _depths(collider: Collider, local: np.ndarray) -> np.ndarray:
    """Depth of each local point inside the solid (positive inside, negative outside)."""
    if collider.kind == "box":
        return np.min(collider.half_extents - np.abs(local), axis=1)
    if collider.kind == "cylinder":
        return np.minimum(collider.radius - np.linalg.norm(local[:, :2], axis=1), collider.half_height - np.abs(local[:, 2]))
    if collider.kind == "sphere":
        return collider.radius - np.linalg.norm(local, axis=1)
    return np.min(collider.planes[:, 3][None, :] - local @ collider.planes[:, :3].T, axis=1)


def _exit_distances(collider: Collider, local: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """How far each inside point must move along ``direction`` (local) to leave the solid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if collider.kind == "box":
            steps = np.where(direction != 0, (np.sign(direction) * collider.half_extents - local) / direction, np.inf)
            return np.min(np.where(steps >= 0, steps, np.inf), axis=1)
        if collider.kind == "cylinder":
            axial = (np.sign(direction[2]) * collider.half_height - local[:, 2]) / direction[2] if direction[2] else np.full(len(local), np.inf)
            a = float(direction[:2] @ direction[:2])
            if a <= 0:
                radial = np.full(len(local), np.inf)
            else:
                b = local[:, :2] @ direction[:2]
                c = np.sum(local[:, :2] ** 2, axis=1) - collider.radius ** 2
                radial = (-b + np.sqrt(np.maximum(b * b - a * c, 0))) / a
            return np.minimum(axial, radial)
        if collider.kind == "sphere":
            b = local @ direction
            c = np.sum(local ** 2, axis=1) - collider.radius ** 2
            return -b + np.sqrt(np.maximum(b * b - c, 0))
        rate = collider.planes[:, :3] @ direction
        slack = collider.planes[:, 3][None, :] - local @ collider.planes[:, :3].T
        steps = np.where(rate[None, :] > 0, slack / rate[None, :], np.inf)
        return np.min(steps, axis=1)


def _placed(colliders: Iterable[Collider], world_from_body: Mapping[str, np.ndarray]):
    for collider in colliders:
        world = world_from_body[collider.body] @ collider.body_from_collider
        yield collider, world


def penetration(points: np.ndarray, placed: Sequence[tuple[Collider, np.ndarray]], *, tree: Any = None,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Deepest penetration of each point into any placed collider, and which collider."""
    from scipy.spatial import cKDTree

    tree = tree if tree is not None else cKDTree(points)
    depth = np.full(len(points), -np.inf)
    owner = np.full(len(points), -1, dtype=np.int64)
    for index, (collider, world) in enumerate(placed):
        near = np.asarray(tree.query_ball_point(world[:3, 3], collider.bound_radius), dtype=np.int64)
        if not len(near):
            continue
        local = (points[near] - world[:3, 3]) @ world[:3, :3]
        d = _depths(collider, local)
        better = d > depth[near]
        depth[near[better]] = d[better]
        owner[near[better]] = index
    return depth, owner


def clearing_translation(points: np.ndarray, placed: Sequence[tuple[Collider, np.ndarray]],
                         inside: np.ndarray, *, directions: int = _PUSH_DIRECTIONS) -> dict[str, Any]:
    """Shortest move of the whole asset that takes every inside point out of every solid it is in.

    A first-order estimate of the push: it considers the surfaces the asset is in now,
    not ones a long move would reach. The engine applies at least this much.
    """
    if not np.any(inside):
        return {"distance_m": 0.0, "direction_world": [0.0, 0.0, 0.0]}
    golden = math.pi * (3 - math.sqrt(5))
    k = np.arange(directions)
    z = 1 - 2 * (k + 0.5) / directions
    r = np.sqrt(1 - z * z)
    candidates = np.column_stack([r * np.cos(golden * k), r * np.sin(golden * k), z])
    chosen = points[inside]
    # Which chosen points each collider holds does not depend on the direction.
    held = []
    for collider, world in placed:
        local = (chosen - world[:3, 3]) @ world[:3, :3]
        inner = np.flatnonzero(_depths(collider, local) > 0)
        if len(inner):
            held.append((collider, world[:3, :3], local[inner], inner))
    best = (math.inf, None)
    for direction in candidates:
        # Moving the asset by s*d moves each static point by -s*d relative to it.
        need = np.zeros(len(chosen))
        for collider, rotation, local, inner in held:
            steps = _exit_distances(collider, local, -(rotation.T @ direction))
            need[inner] = np.maximum(need[inner], steps)
        distance = float(need.max())
        if distance < best[0]:
            best = (distance, direction)
    return {"distance_m": best[0], "direction_world": [float(v) for v in best[1]]}


def _pose_matrix(pose: Mapping[str, Sequence[float]]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
    matrix[:3, 3] = pose["position_world_m"]
    return matrix


def _summary(points: np.ndarray, depth: np.ndarray, owner: np.ndarray, placed, tolerance: float) -> dict[str, Any]:
    inside = depth > tolerance
    by_body: dict[str, dict[str, float]] = {}
    for index in np.unique(owner[inside]):
        collider = placed[int(index)][0]
        rows = inside & (owner == index)
        entry = by_body.setdefault(collider.body.rsplit("/", 1)[-1], {"points": 0, "max_depth_m": 0.0})
        entry["points"] += int(rows.sum())
        entry["max_depth_m"] = max(entry["max_depth_m"], float(depth[rows].max()))
    heights = points[inside, 2]
    return {"penetrating_points": int(inside.sum()),
            "max_depth_m": float(depth.max()) if len(depth) else 0.0,
            "penetration_by_body": dict(sorted(by_body.items())),
            "penetration_height_range_m": [float(heights.min()), float(heights.max())] if len(heights) else None}


_FACE_NAMES = ("x-", "x+", "y-", "y+", "z-", "z+")


def asset_envelope(model: AssetModel, joint_values: Mapping[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Box in the asset frame bounding every collider at the reset pose."""
    poses = body_poses(model, joint_values)
    corners = []
    for collider in model.colliders:
        frame = poses[collider.body] @ collider.body_from_collider
        corners.append(collider.local_corners() @ frame[:3, :3].T + frame[:3, 3])
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def envelope_intrusion(points: np.ndarray, normals: np.ndarray, world_from_asset: np.ndarray, low: np.ndarray,
                       high: np.ndarray, tolerance: float) -> dict[str, Any]:
    """Background inside the space the asset occupies, by the face each surface stands behind.

    A shell of thin panels hides a wall standing well inside its volume from the
    solid test, which only sees where the wall crosses a panel. Each sample is
    attributed by its surface normal: a wall facing x stands behind an x face (the
    one on its side of the centre), a floor or shelf behind a z face, and its
    depth is its distance in from that face.
    """
    local = (points - world_from_asset[:3, 3]) @ world_from_asset[:3, :3]
    facing = np.abs(normals @ world_from_asset[:3, :3])
    axis = facing.argmax(axis=1)
    rows = np.arange(len(points))
    coordinate = local[rows, axis]
    upper = coordinate > (low[axis] + high[axis]) / 2
    depth = np.where(upper, high[axis] - coordinate, coordinate - low[axis])
    # Inside means clear of every face by the tolerance: a surface grazing one face
    # (a preserved cabinet panel at the carve boundary) is not an intrusion.
    clearance = np.minimum(local - low, high - local).min(axis=1)
    inside = (clearance > tolerance) & (facing.max(axis=1) > 0)
    face = 2 * axis + upper.astype(int)
    faces = {}
    for index in np.unique(face[inside]):
        chosen = inside & (face == index)
        # A wall standing inside the volume reads as a pile-up at one depth; the
        # median says where it stands, the maximum only where its deepest scrap is.
        faces[_FACE_NAMES[int(index)]] = {"points": int(chosen.sum()), "median_depth_m": float(np.median(depth[chosen])),
                                          "p90_depth_m": float(np.percentile(depth[chosen], 90)),
                                          "max_depth_m": float(depth[chosen].max())}
    return {"tolerance_m": tolerance, "envelope_asset_m": {"minimum": low.tolist(), "maximum": high.tolist()},
            "intruding_points": int(inside.sum()), "max_depth_m": float(depth[inside].max()) if np.any(inside) else 0.0,
            "by_face": dict(sorted(faces.items()))}


_CLIP_LONGEST_EDGE_M = 0.5


def _clip_to_box(triangle: np.ndarray, low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """The part of one triangle inside an axis-aligned box, as a fan of triangles."""
    polygon = [np.asarray(v, dtype=float) for v in triangle]
    for axis in range(3):
        for bound, keep_above in ((low[axis], True), (high[axis], False)):
            kept = []
            for index, current in enumerate(polygon):
                previous = polygon[index - 1]
                inside_now = current[axis] >= bound if keep_above else current[axis] <= bound
                inside_before = previous[axis] >= bound if keep_above else previous[axis] <= bound
                if inside_now != inside_before:
                    fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                    kept.append(previous + fraction * (current - previous))
                if inside_now:
                    kept.append(current)
            polygon = kept
            if len(polygon) < 3:
                return []
    return [np.stack([polygon[0], polygon[i], polygon[i + 1]]) for i in range(1, len(polygon) - 1)]


def _clip_large_triangles(triangles: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Cut big triangles down to the reachable box so a floor 100 m across is not sampled in full."""
    if not len(triangles):
        return triangles
    edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max(axis=1)
    large = edges > _CLIP_LONGEST_EDGE_M
    pieces = [piece for triangle in triangles[large] for piece in _clip_to_box(triangle, low, high)]
    return np.concatenate([triangles[~large], np.asarray(pieces).reshape(-1, 3, 3)])


def _reachable_triangles(triangles: np.ndarray, placements: Sequence[tuple[Collider, np.ndarray]]) -> np.ndarray:
    """Triangles that come within reach of any placed collider."""
    from scipy.spatial import cKDTree

    if not len(triangles):
        return triangles
    centroids = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centroids[:, None], axis=2).max(axis=1)
    tree = cKDTree(centroids)
    keep = np.zeros(len(triangles), dtype=bool)
    reach = float(radii.max())
    for collider, world in placements:
        near = np.asarray(tree.query_ball_point(world[:3, 3], collider.bound_radius + reach), dtype=np.int64)
        if len(near):
            close = np.linalg.norm(centroids[near] - world[:3, 3], axis=1) <= collider.bound_radius + radii[near]
            keep[near[close]] = True
    return triangles[keep]


def _sweep_values(joint: Joint, span: Sequence[float], revolute_step: float, prismatic_step: float) -> np.ndarray:
    step = revolute_step if joint.kind == "revolute" else prismatic_step
    values = np.linspace(joint.lower, joint.upper, max(2, int(math.ceil((joint.upper - joint.lower) / step)) + 1))
    # Sample the task's own endpoints too: a goal between two sweep steps is
    # exactly the position the task must reach.
    return np.unique(np.concatenate([values, [float(v) for v in span if joint.lower <= float(v) <= joint.upper]]))


def _joint_label(motion: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    return ("%.0fdeg" % math.degrees(sample["value"]) if motion["joint_type"] == "revolute"
            else "%.0fcm" % (100 * sample["value"]))


def scene_physics_preflight(*, background_usd: str | Path, asset_usd: str | Path,
                            asset_pose_world: Mapping[str, Sequence[float]],
                            reset_joint_positions: Mapping[str, float] | None = None,
                            background_pose_world: Mapping[str, Sequence[float]] | None = None,
                            movable_joint_ids: Sequence[str] | None = None,
                            task_ranges: Mapping[str, Sequence[float]] | None = None,
                            sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
                            rest_tolerance_m: float = DEFAULT_REST_TOLERANCE_M,
                            motion_tolerance_m: float = DEFAULT_MOTION_TOLERANCE_M,
                            revolute_step_rad: float = DEFAULT_REVOLUTE_STEP_RAD,
                            prismatic_step_m: float = DEFAULT_PRISMATIC_STEP_M) -> dict[str, Any]:
    """Measure one placed asset against the static background; raises only on unusable input.

    ``task_ranges`` maps a joint to the span the task drives it through. With it,
    contact inside a span blocks and every other contact is an advisory; without
    it, every contact blocks.
    """
    from scipy.spatial import cKDTree

    model = load_asset_model(asset_usd)
    world_from_asset = _pose_matrix(asset_pose_world)
    reset = dict(reset_joint_positions or {})
    known_joints = {j.joint_id for j in model.joints}
    unknown = (set(reset) | set(task_ranges or {})) - known_joints
    if unknown:
        raise ScenePhysicsPreflightError("scene_physics_joint_unknown:" + ",".join(sorted(unknown)))
    pose = {k: [float(v) for v in asset_pose_world[k]] for k in ("position_world_m", "orientation_xyzw")}

    def nothing_to_measure(notice: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "status": "passed", "blockers": [], "advisories": [],
                "notices": [notice], "asset_pose_world": pose, "asset_mounted": model.mounted,
                "skipped_colliders": list(model.skipped_colliders),
                "simulator_executed": False, "provider_mutation_performed": False}

    if not model.colliders:
        return nothing_to_measure("scene_physics_asset_solid_colliders_absent")

    def placed_at(values: Mapping[str, float] | None, bodies: set[str] | None = None):
        poses = {body: world_from_asset @ pose for body, pose in body_poses(model, {**reset, **(values or {})}).items()}
        return [(c, w) for c, w in _placed(model.colliders, poses) if bodies is None or c.body in bodies]

    movable = [j for j in model.joints if j.kind != "fixed"]
    if movable_joint_ids is not None:
        unknown = sorted(set(movable_joint_ids) - {j.joint_id for j in movable})
        if unknown:
            raise ScenePhysicsPreflightError("scene_physics_joint_unknown:" + ",".join(unknown))
        movable = [j for j in movable if j.joint_id in set(movable_joint_ids)]
    rest = placed_at(None)
    sweeps = []
    for joint in movable:
        moving = descendants(model, joint.joint_id)
        values = _sweep_values(joint, (task_ranges or {}).get(joint.joint_id) or [], revolute_step_rad, prismatic_step_m)
        sweeps.append((joint, moving, values, [placed_at({joint.joint_id: float(v)}, moving) for v in values]))
    every = rest + [pair for _, _, _, frames in sweeps for placed in frames for pair in placed]
    centres = np.asarray([w[:3, 3] for _, w in every])
    centre = centres.mean(axis=0)
    radius = float(np.linalg.norm(centres - centre, axis=1).max() + max(c.bound_radius for c in model.colliders))
    nearby = load_background_triangles(background_usd, centre=centre, radius=radius,
                                       world_from_stage=None if background_pose_world is None else _pose_matrix(background_pose_world))
    bounds = np.asarray([c.bound_radius for c, _ in every])
    triangles = _clip_large_triangles(_reachable_triangles(nearby, every),
                                      (centres - bounds[:, None]).min(axis=0), (centres + bounds[:, None]).max(axis=0))
    points, owners = sample_surface(triangles, sample_spacing_m)
    if not len(points):
        return nothing_to_measure("scene_physics_background_empty_near_asset")
    tree = cKDTree(points)

    depth, owner = penetration(points, rest, tree=tree)
    rest_report = _summary(points, depth, owner, rest, rest_tolerance_m)
    rest_report["tolerance_m"] = rest_tolerance_m
    rest_report["clearing_translation"] = clearing_translation(points, rest, depth > rest_tolerance_m)
    low, high = asset_envelope(model, reset)
    envelope = envelope_intrusion(points, triangle_normals(triangles)[owners], world_from_asset, low, high, rest_tolerance_m)

    motions = []
    for joint, moving, values, frames in sweeps:
        # Contact the moving parts already have at reset belongs to the rest
        # overlap; a motion reports only the contact it adds.
        base, _ = penetration(points, [(c, w) for c, w in rest if c.body in moving], tree=tree)
        samples, first_contact = [], None
        for value, placed in zip(values, frames):
            d, o = penetration(points, placed, tree=tree)
            added = (d > motion_tolerance_m) & ~(base > motion_tolerance_m)
            row = {"value": float(value), "contact_points": int(added.sum()),
                   "max_depth_m": float(d[added].max()) if np.any(added) else 0.0}
            if np.any(added):
                row["contact_height_range_m"] = [float(points[added, 2].min()), float(points[added, 2].max())]
                row["contact_bodies"] = sorted({placed[int(i)][0].body.rsplit("/", 1)[-1] for i in np.unique(o[added])})
                first_contact = first_contact or row
            samples.append(row)
        motions.append({"joint_id": joint.joint_id, "joint_type": joint.kind, "limits": [joint.lower, joint.upper],
                        "moving_bodies": sorted(b.rsplit("/", 1)[-1] for b in moving), "sample_count": len(values),
                        "first_contact": first_contact,
                        "contact_sample_count": sum(1 for s in samples if s["contact_points"]), "samples": samples})

    blockers, advisories = [], []
    if rest_report["penetrating_points"]:
        blockers.append("scene_physics_rest_overlap:%.1fcm" % (100 * rest_report["max_depth_m"]))
    for face, row in envelope["by_face"].items():
        blockers.append("scene_physics_envelope_intrusion:%s:%.1fcm" % (face, 100 * row["median_depth_m"]))
    for motion in motions:
        contacts = [s for s in motion["samples"] if s["contact_points"]]
        span = None if task_ranges is None else task_ranges.get(motion["joint_id"])
        if span is not None:
            low_value, high_value = sorted(float(v) for v in span)
            motion["task_range"] = [low_value, high_value]
        inside = [s for s in contacts if task_ranges is None
                  or (span is not None and low_value - 1e-6 <= s["value"] <= high_value + 1e-6)]
        outside = [s for s in contacts if s not in inside]
        if inside:
            blockers.append("scene_physics_motion_contact:%s@%s" % (motion["joint_id"], _joint_label(motion, inside[0])))
        if outside:
            advisories.append("scene_physics_motion_contact_outside_task_range:%s@%s"
                              % (motion["joint_id"], _joint_label(motion, outside[0])))
    return {"schema_version": SCHEMA_VERSION, "status": "blocked" if blockers else "passed", "blockers": blockers,
            "advisories": advisories, "notices": [], "asset_pose_world": pose,
            "skipped_colliders": list(model.skipped_colliders),
            "asset_mounted": model.mounted, "background_triangles_reached": int(len(triangles)),
            "background_samples": int(len(points)), "sample_spacing_m": sample_spacing_m,
            "rest_overlap": rest_report, "envelope_intrusion": envelope, "motion_clearance": motions,
            "simulator_executed": False, "provider_mutation_performed": False}



# The robot parts no policy action can move out of a moving part's way, in the
# robot's base frame, measured from the runtime USD named by
# ``measured_from_sha256``: link 0's bounding box, and the cylinder about the base
# axis that holds link 1 at every joint-1 angle and link 2's housing around the
# joint-2 axis at every joint-2 angle.
ROBOT_FIXED_VOLUMES: dict[str, dict[str, Any]] = {
    "franka_panda": {
        "measured_from_sha256": "sha256:c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c",
        "boxes": (((-0.154, -0.095, 0.0), (0.072, 0.095, 0.140)),),
        "cylinders": ((0.13, 0.140, 0.391),),  # radius, base-frame z low, z high
    },
}


def robot_fixed_surface(robot_id: str, spacing: float = DEFAULT_SAMPLE_SPACING_M) -> np.ndarray:
    """Surface samples of a robot's fixed volumes, in its base frame."""
    volumes = ROBOT_FIXED_VOLUMES.get(robot_id)
    if volumes is None:
        raise ScenePhysicsPreflightError("scene_physics_robot_model_unmeasured:" + str(robot_id))

    def steps(low: float, high: float) -> np.ndarray:
        return np.linspace(low, high, max(2, int(math.ceil((high - low) / spacing)) + 1))

    parts = []
    for low, high in volumes["boxes"]:
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        grid = np.stack(np.meshgrid(*(steps(low[i], high[i]) for i in range(3)), indexing="ij"), axis=-1).reshape(-1, 3)
        parts.append(grid[np.isclose(grid, low).any(axis=1) | np.isclose(grid, high).any(axis=1)])
    for radius, z_low, z_high in volumes["cylinders"]:
        rings = []
        for ring_radius in steps(0.0, radius):
            angles = np.linspace(0.0, 2 * math.pi, max(1, int(math.ceil(2 * math.pi * ring_radius / spacing))),
                                 endpoint=False)
            rings.append(np.c_[ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
        heights, rim = steps(z_low, z_high), rings[-1]
        parts.append(np.c_[np.tile(rim, (len(heights), 1)), np.repeat(heights, len(rim))])
        parts.extend(np.c_[ring, np.full(len(ring), z)] for ring in rings for z in (z_low, z_high))
    return np.concatenate(parts)


def robot_motion_clearance(*, asset_usd: str | Path, asset_pose_world: Mapping[str, Sequence[float]],
                           reset_joint_positions: Mapping[str, float] | None, joint_id: str,
                           task_range: Sequence[float], robot_id: str,
                           robot_pose_world: Mapping[str, Sequence[float]],
                           sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
                           rest_tolerance_m: float = DEFAULT_REST_TOLERANCE_M,
                           motion_tolerance_m: float = DEFAULT_MOTION_TOLERANCE_M,
                           revolute_step_rad: float = DEFAULT_REVOLUTE_STEP_RAD,
                           prismatic_step_m: float = DEFAULT_PRISMATIC_STEP_M) -> dict[str, Any]:
    """Measure the robot's fixed volumes against the placed asset at reset and along one joint.

    Contact at reset or inside ``task_range`` blocks; contact elsewhere in the
    joint's limits is an advisory.
    """
    from scipy.spatial import cKDTree

    model = load_asset_model(asset_usd)
    joint = next((j for j in model.joints if j.joint_id == joint_id and j.kind != "fixed"), None)
    if joint is None:
        raise ScenePhysicsPreflightError("scene_physics_joint_unknown:" + joint_id)
    world_from_asset = _pose_matrix(asset_pose_world)
    world_from_robot = _pose_matrix(robot_pose_world)
    points = robot_fixed_surface(robot_id, sample_spacing_m) @ world_from_robot[:3, :3].T + world_from_robot[:3, 3]
    tree = cKDTree(points)
    reset = dict(reset_joint_positions or {})
    moving = descendants(model, joint_id)

    def placed_at(values: Mapping[str, float] | None, bodies: set[str] | None = None):
        poses = {body: world_from_asset @ pose for body, pose in body_poses(model, {**reset, **(values or {})}).items()}
        return [(c, w) for c, w in _placed(model.colliders, poses) if bodies is None or c.body in bodies]

    rest = placed_at(None)
    depth, _ = penetration(points, rest, tree=tree)
    base, _ = penetration(points, [(c, w) for c, w in rest if c.body in moving], tree=tree)
    motion = {"joint_id": joint_id, "joint_type": joint.kind, "limits": [joint.lower, joint.upper],
              "task_range": sorted(float(v) for v in task_range), "samples": []}
    for value in _sweep_values(joint, task_range, revolute_step_rad, prismatic_step_m):
        d, _ = penetration(points, placed_at({joint_id: float(value)}, moving), tree=tree)
        added = (d > motion_tolerance_m) & ~(base > motion_tolerance_m)
        motion["samples"].append({"value": float(value), "contact_points": int(added.sum()),
                                  "max_depth_m": float(d[added].max()) if np.any(added) else 0.0})
    low, high = motion["task_range"]
    contacts = [s for s in motion["samples"] if s["contact_points"]]
    inside = [s for s in contacts if low - 1e-6 <= s["value"] <= high + 1e-6]
    outside = [s for s in contacts if s not in inside]
    blockers, advisories = [], []
    overlap = depth > rest_tolerance_m
    rest_depth = float(depth[overlap].max()) if np.any(overlap) else 0.0
    if rest_depth:
        blockers.append("scene_physics_robot_rest_overlap:%.1fcm" % (100 * rest_depth))
    if inside:
        blockers.append("scene_physics_robot_in_motion_path:%s@%s" % (joint_id, _joint_label(motion, inside[0])))
    if outside:
        advisories.append("scene_physics_robot_in_motion_path_outside_task_range:%s@%s"
                          % (joint_id, _joint_label(motion, outside[0])))
    return {"robot_id": robot_id, "robot_pose_world": {k: [float(v) for v in robot_pose_world[k]]
                                                       for k in ("position_world_m", "orientation_xyzw")},
            "measured_from_sha256": ROBOT_FIXED_VOLUMES[robot_id]["measured_from_sha256"],
            "status": "blocked" if blockers else "passed", "blockers": blockers, "advisories": advisories,
            "notices": [], "robot_samples": int(len(points)), "rest_max_depth_m": rest_depth,
            "motion": motion}


def robot_fixed_volume_mask(robot_id: str, points_base: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
    """Which base-frame points lie in a robot's fixed volumes grown by ``margin_m``."""
    volumes = ROBOT_FIXED_VOLUMES.get(robot_id)
    if volumes is None:
        raise ScenePhysicsPreflightError("scene_physics_robot_model_unmeasured:" + str(robot_id))
    inside = np.zeros(len(points_base), dtype=bool)
    for low, high in volumes["boxes"]:
        inside |= np.all((points_base >= np.asarray(low) - margin_m) & (points_base <= np.asarray(high) + margin_m),
                         axis=1)
    for radius, z_low, z_high in volumes["cylinders"]:
        inside |= ((np.hypot(points_base[:, 0], points_base[:, 1]) <= radius + margin_m)
                   & (points_base[:, 2] >= z_low - margin_m) & (points_base[:, 2] <= z_high + margin_m))
    return inside


def robot_fixed_volume_extent(robot_id: str) -> float:
    """The farthest horizontal reach of a robot's fixed volumes from its base axis."""
    volumes = ROBOT_FIXED_VOLUMES.get(robot_id)
    if volumes is None:
        raise ScenePhysicsPreflightError("scene_physics_robot_model_unmeasured:" + str(robot_id))
    corners = [math.hypot(x, y) for low, high in volumes["boxes"] for x in (low[0], high[0]) for y in (low[1], high[1])]
    return max(corners + [radius for radius, _, _ in volumes["cylinders"]])


def _voxel_thin(points: np.ndarray, spacing: float) -> np.ndarray:
    """One point per ``spacing`` cube, kept in input order."""
    if not len(points):
        return points
    _, keep = np.unique(np.floor(points / spacing).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(keep)]


def _collider_surface(collider: Collider, spacing: float) -> np.ndarray:
    """Surface samples of one collider in its own frame; a box bounds a cylinder or sphere."""
    if collider.kind == "hull":
        from scipy.spatial import ConvexHull

        return sample_surface(collider.vertices[ConvexHull(collider.vertices).simplices], spacing)[0]
    corners = collider.local_corners()  # index = 4*(x>0) + 2*(y>0) + (z>0)
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    triangles = np.asarray([corners[[a, b, c]] for a, b, c, d in faces] + [corners[[a, c, d]] for a, b, c, d in faces])
    return sample_surface(triangles, spacing)[0]


def task_motion_surface(*, asset_usd: str | Path, asset_pose_world: Mapping[str, Sequence[float]], joint_id: str,
                        task_range: Sequence[float], reset_joint_positions: Mapping[str, float] | None = None,
                        sample_spacing_m: float = 0.02, revolute_step_rad: float = DEFAULT_REVOLUTE_STEP_RAD,
                        prismatic_step_m: float = DEFAULT_PRISMATIC_STEP_M) -> np.ndarray:
    """World surface samples of every part ``joint_id`` moves, at each sweep step inside ``task_range``."""
    model = load_asset_model(asset_usd)
    joint = next((j for j in model.joints if j.joint_id == joint_id and j.kind != "fixed"), None)
    if joint is None:
        raise ScenePhysicsPreflightError("scene_physics_joint_unknown:" + joint_id)
    low, high = sorted(float(v) for v in task_range)
    moving = descendants(model, joint_id)
    local = [(c, _voxel_thin(_collider_surface(c, sample_spacing_m), sample_spacing_m))
             for c in model.colliders if c.body in moving]
    world_from_asset = _pose_matrix(asset_pose_world)
    reset = dict(reset_joint_positions or {})
    parts = [np.zeros((0, 3))]
    for value in _sweep_values(joint, [low, high], revolute_step_rad, prismatic_step_m):
        if not low - 1e-9 <= value <= high + 1e-9:
            continue
        poses = body_poses(model, {**reset, joint_id: float(value)})
        for collider, points in local:
            world = world_from_asset @ poses[collider.body] @ collider.body_from_collider
            parts.append(points @ world[:3, :3].T + world[:3, 3])
    return _voxel_thin(np.concatenate(parts), sample_spacing_m)


PACKET_REPORT_SCHEMA_VERSION = "native_task_scene_physics_preflight.v1"
PACKET_REPORT_FILENAME = "native_task_scene_physics_preflight.v1.json"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def task_motion_ranges(plan: Mapping[str, Any]) -> dict[str, list[float]]:
    """The task subject's commanded span: its target joint from reset to the success threshold.

    An articulated task whose span cannot be read is refused rather than treated as
    commanding nothing, which would turn every contact into an advisory.
    """
    spec = plan.get("task_spec") or {}
    if spec.get("task_kind") != "articulated_open_close":
        return {}
    threshold = spec.get("executable_opening_threshold")
    try:
        if threshold is not None:
            target = str(threshold["target_joint_id"])
            reset, goal = float(threshold["reset_position"]), float(threshold["success_interval"][0])
        else:
            # Pre-v2 task specs command the scripted target from the scorer's reset.
            target = str(spec["target_joint_id"])
            reset = float(spec["joint_reset_positions_rad"][target])
            goal = float(spec["scripted_positive_target_rad"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ScenePhysicsPreflightError("scene_physics_task_range_unknown") from exc
    if not target or not (math.isfinite(reset) and math.isfinite(goal)):
        raise ScenePhysicsPreflightError("scene_physics_task_range_unknown")
    # The plan names the joint by graph id; the asset's joint prim carries the USD name.
    prim = ((plan.get("articulation") or {}).get("task_joint_prim_paths") or {}).get(target)
    return {str(prim).rsplit("/", 1)[-1] if prim else target: sorted([reset, goal])}


def preflight_native_packet(packet_dir: str | Path, **options: Any) -> dict[str, Any]:
    """Run the preflight for every articulated object of a native task arena packet.

    The background is the plan's one ``scene_collision`` object at its pose; each
    ``ARTICULATION`` object is checked at its reset root pose and joint positions.
    The task subject's target joint blocks only inside the task's range; other
    articulated objects and joints report contact without blocking. The robot's
    fixed volumes, at the plan's base pose, block inside the subject at reset or in
    its target joint's path before the success threshold, and report beyond it.
    Every file is bound to the digest the plan recorded for it. A plan with no
    articulated object is ``not_applicable``.
    """
    import json

    from .decision_evidence_contracts import canonical_digest

    root = Path(packet_dir)
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())
    objects = plan.get("objects", [])
    if not isinstance(objects, list):
        raise ScenePhysicsPreflightError("scene_physics_packet_plan_objects_invalid")
    articulated = [o for o in objects if isinstance(o, Mapping) and o.get("object_type") == "ARTICULATION"]
    if not articulated:
        result = {"schema_version": PACKET_REPORT_SCHEMA_VERSION, "arena_scene_plan_digest": plan.get("plan_digest"),
                  "status": "not_applicable", "reason": "no_articulated_object", "blockers": [], "advisories": [],
                  "objects": [], "simulator_executed": False, "provider_mutation_performed": False}
        result["report_digest"] = canonical_digest(result, digest_field="report_digest")
        return result

    def bound(row: Mapping[str, Any]) -> Path:
        path = (root / str(row.get("usd_path") or "")).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ScenePhysicsPreflightError("scene_physics_packet_asset_missing:" + str(row.get("name")))
        if _file_sha256(path) != row.get("sha256"):
            raise ScenePhysicsPreflightError("scene_physics_packet_asset_digest_mismatch:" + str(row.get("name")))
        return path

    collisions = [o for o in objects if o.get("semantic_role") == "scene_collision"]
    if len(collisions) != 1:
        raise ScenePhysicsPreflightError("scene_physics_packet_scene_collision_count_invalid")
    background = bound(collisions[0])
    ranges = task_motion_ranges(plan)
    subjects = [o for o in articulated if o.get("task_subject") is True or o.get("semantic_role") == "task_object"]
    rows, blockers, advisories = [], [], []
    for row in articulated:
        reset = row.get("reset_state") or {}
        report = scene_physics_preflight(
            background_usd=background, asset_usd=bound(row),
            asset_pose_world=reset.get("root_pose_world") or row["pose_world"],
            reset_joint_positions=reset.get("joint_positions"),
            background_pose_world=collisions[0].get("pose_world"),
            task_ranges=ranges if len(subjects) == 1 and row is subjects[0] else {}, **options)
        rows.append({"name": row.get("name"), "usd_path": row.get("usd_path"), "sha256": row.get("sha256"), **report})
        blockers.extend(f"{row.get('name')}:{b}" for b in report["blockers"])
        advisories.extend(f"{row.get('name')}:{a}" for a in report["advisories"])
    robot_report = None
    reach = ranges if len(subjects) == 1 else {}
    if reach:
        robot = plan.get("robot") or {}
        if not isinstance(robot.get("base_pose_world"), Mapping):
            raise ScenePhysicsPreflightError("scene_physics_robot_pose_missing")
        subject, ((joint_id, span),) = subjects[0], reach.items()
        reset = subject.get("reset_state") or {}
        robot_report = robot_motion_clearance(
            asset_usd=bound(subject), asset_pose_world=reset.get("root_pose_world") or subject["pose_world"],
            reset_joint_positions=reset.get("joint_positions"), joint_id=joint_id, task_range=span,
            robot_id=str(robot.get("robot_id")), robot_pose_world=robot["base_pose_world"], **options)
        runtime = ((plan.get("articulation") or {}).get("robot_contact_topology") or {}).get("runtime_asset") or {}
        if runtime.get("sha256") and runtime["sha256"] != robot_report["measured_from_sha256"]:
            robot_report["notices"].append("scene_physics_robot_volumes_measured_on_other_asset")
        blockers.extend(f"{subject.get('name')}:{b}" for b in robot_report["blockers"])
        advisories.extend(f"{subject.get('name')}:{a}" for a in robot_report["advisories"])
    result = {"schema_version": PACKET_REPORT_SCHEMA_VERSION, "arena_scene_plan_digest": plan.get("plan_digest"),
              "scene_collision_sha256": collisions[0].get("sha256"), "status": "blocked" if blockers else "passed",
              "blockers": blockers, "advisories": advisories, "task_ranges": ranges, "objects": rows,
              "robot_clearance": robot_report, "simulator_executed": False, "provider_mutation_performed": False}
    result["report_digest"] = canonical_digest(result, digest_field="report_digest")
    return result


def require_scene_physics_admission(packet_dir: str | Path, **options: Any) -> dict[str, Any]:
    """Refuse a paid native run whose packet the preflight blocks or cannot measure."""
    try:
        report = preflight_native_packet(packet_dir, **options)
    except ScenePhysicsPreflightError as exc:
        raise ScenePhysicsAdmissionBlocked(["scene_physics_preflight_refused:" + str(exc)]) from exc
    if report["status"] == "blocked":
        raise ScenePhysicsAdmissionBlocked(report["blockers"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CPU physics preflight of a native task arena packet.")
    parser.add_argument("--packet", required=True, help="native task arena packet directory")
    parser.add_argument("--output", help="write the full report here (default: stdout summary only)")
    args = parser.parse_args(argv)
    try:
        report = preflight_native_packet(args.packet)
    except ScenePhysicsPreflightError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "blockers": report["blockers"],
                      "advisories": report["advisories"], "report_digest": report["report_digest"]}, sort_keys=True))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
