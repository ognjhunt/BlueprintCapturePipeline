"""Any registered body's arm on the CPU: its kinematic chain, forward and inverse kinematics, link bounds
and clearance, read from the body's own prepared model.

Blueprint places and checks its own DROID Franka before any GPU run with the published Franka model
(``robot_arm_path_clearance``); every other body had only a straight-line ``reach_m``. This module reads
any body's arm from the robot USD Blueprint simulates (the prepared ``robot.usda``) and the team
manifest's arm (``base_link``, ``tool_link``, ``joint_names``, ``tool_from_tcp``, the ``joints`` limits
and ``reset_joint_positions``), so every body can get the same CPU checks.

**Chain.** Bodies are the model's rigid bodies; a prim that is not one moves with the nearest rigid body
above it, and a joint end on no rigid body is the static world (as ``team_embodiment._world_anchor``
reads it). The enabled UsdPhysics joints between them are walked from the base link's body to the tool
link's: revolute and prismatic joints with their ``localPos``/``localRot`` frames on both ends, ``axis``
and limits (degrees for revolute), and fixed joints in between, in either authoring direction. Names are
matched as the manifest carries them -- the prim names an importer authored (Isaac Sim 6 may rename
``link6-7`` to ``tn__link67_sG``) -- or else the source name it recorded as ``displayName``. A chain that
is missing, ambiguous (a loop or a parallel linkage gives two routes), branches off the declared joints,
or holds a mimic or an unsupported joint is refused with a typed code; it is never replaced by a guess.
Joint limits are the manifest's command window intersected with the model's own limits. The chain's
bodies are those on that route plus any welded by fixed joints to a moving one (a wrist camera); the base's
pose in the robot root (``root_from_base``) is read from the authored model.

**Kinematics.** ``forward`` gives every chain body's frame, each joint frame, the tool link and the TCP
(``tool_from_tcp`` applied) in the arm's base frame. ``solve`` is damped least-squares (Levenberg-
Marquardt damping) IK of a 6-D TCP target with a configurable orientation weight and tolerances, joint
limits held by clipping and an active set, from the caller's seeds, the manifest reset, the middle of the
limits and a fixed-seed set of further starts. A miss names its reason: ``ik_unreachable`` (beyond the
chain's reach bound, or no solution from any start even ignoring joint limits) or ``ik_joint_limit``
(a start reached it only outside the limits). Numpy only; no randomness without a fixed seed.

**Bounds and clearance.** ``link_bounds`` measures each chain body's collision geometry (visual where a
body has none) in its own frame as ``ARM_LINK_BOUNDS`` are measured -- the largest extent along each of
``DIRECTIONS``, rounded up to 0.1 mm -- and the hand beyond the tool link as one more bound at its
authored pose, contact pads excluded on request. ``clearance`` returns the least gap and the link that
has it between a posed arm and scene points (the ``SceneObstacles`` samples ``robot_arm_path_clearance``
measures against) or oriented boxes. Both gaps are lower bounds on the true distance. A body with no
geometry stays unmeasured and is reported as unassessed, never as clear.

Everything here is a CPU bounded estimate from the model: no simulator runs, and a reachable, clear pose
means one exists in the model, not that a policy or the physical robot finds it.

**How the station search and the gates call it (steps 3 and 4).** Read the arms once per prepared model:
``body_arm_chains(robot_usd, manifest)`` returns each arm's ``ArmChain`` and a typed reason for any arm it
cannot assess (report that arm unassessed; do not fall back to ``reach_m``). For a candidate stance,
``world_from_base = world_from_root @ chain.root_from_base`` with the robot root placed at the stance.
Express each grasp, carry and place TCP pose in the base frame (``inv(world_from_base) @ world_from_tcp``)
and call ``solve(chain, base_from_tcp, seeds=[previous])`` along a path, passing the previous solution so
the arm keeps its configuration; ``extra_seeds=False`` and ``classify=False`` keep each step fast. Measure
the bounds once (``link_bounds``) and call ``clearance(chain, bounds, q, world_from_base=...,
points=scene_obstacles, boxes=...)`` at each solved pose; skip the fixed base as the Franka gate does.
Carry ``chain.summary()`` and every IK and clearance record into the report as a bounded estimate.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .robot_arm_path_clearance import DIRECTIONS
from .team_arm_kinematics import pose_matrix

SCHEMA_VERSION = "team_body_kinematics.v1"
CLAIM_CEILING = "cpu_bounded_estimate"
STATIC = "<static>"
IK_POSITION_TOLERANCE_M = 0.002
IK_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
IK_ORIENTATION_WEIGHT_M = 0.3
IK_MAX_ITERATIONS = 80
IK_MAX_REVOLUTE_STEP_RAD = 0.4
IK_MAX_PRISMATIC_STEP_M = 0.05
IK_EXTRA_SEEDS = 24
IK_SEED = 20261031
# A joint whose frames disagree with the authored body poses by more than this is noted (the solver
# snaps the bodies to the joint, so the joint frames still decide the kinematics).
FRAME_TOLERANCE_M = 1e-4
FRAME_TOLERANCE_RAD = 1e-3
CLEARANCE_SEARCH_M = 0.10
# Model limits are single precision (degrees for revolute joints): a manifest limit this close agrees.
LIMIT_TOLERANCE = 1e-5
_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")
_AXES = {"X": np.array([[0.0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
         "Y": np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
         "Z": np.eye(4)}  # each rotates z onto the named axis


def _finite_or_none(values: Any) -> list[float | None]:
    """JSON-safe numbers: an unlimited or unmeasured value is None, never Infinity."""
    return [float(v) if math.isfinite(float(v)) else None for v in values]


class BodyKinematicsError(ValueError):
    """A typed refusal: ``code`` says why the arm stays unassessed. Never a pass, never a fallback."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, eq=False)
class ArmSpec:
    """One arm as a manifest declares it. ``base_link`` None: the robot root, through its world anchor."""

    name: str
    base_link: str | None
    tool_link: str
    joint_names: tuple[str, ...]
    tool_from_tcp: np.ndarray
    limits: Mapping[str, tuple[float, float]]
    reset: Mapping[str, float]
    exact_joints: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class ChainBody:
    """A rigid body on the chain, fixed in the moving frame of joint ``joint`` (-1: the base frame)."""

    name: str
    source_name: str
    path: str
    joint: int
    joint_from_body: np.ndarray


@dataclass(frozen=True, eq=False)
class ArmChain:
    name: str
    robot_usd: str
    robot_usd_sha256: str | None
    base_link: str | None
    tool_link: str
    tool_path: str
    joint_names: tuple[str, ...]  # the joint vector's order: the manifest's
    joint_kinds: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    reset: np.ndarray
    authored: np.ndarray  # joint values at the model's authored pose
    bodies: tuple[ChainBody, ...]
    tool_from_tcp: np.ndarray
    root_from_base: np.ndarray  # the base frame in the robot's default prim, at the authored pose
    reach_bound_m: float
    notes: tuple[str, ...]
    order: np.ndarray  # chain position -> joint vector index
    sign: np.ndarray
    prismatic: np.ndarray
    before: np.ndarray  # (n, 4, 4): previous moving frame (or base) -> this joint's frame, axis along z
    joint_from_tool: np.ndarray  # last moving frame (or base) -> tool link frame
    max_step: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "claim_ceiling": CLAIM_CEILING, "arm": self.name,
                "robot_usd_sha256": self.robot_usd_sha256, "base_link": self.base_link, "tool_link": self.tool_link,
                "joint_names": list(self.joint_names), "joint_kinds": list(self.joint_kinds),
                "lower": _finite_or_none(self.lower), "upper": _finite_or_none(self.upper),
                "bodies": [body.name for body in self.bodies], "reach_bound_m": round(self.reach_bound_m, 4),
                "root_from_base": np.round(self.root_from_base, 9).tolist(), "notes": list(self.notes)}


@dataclass(frozen=True, eq=False)
class ArmPose:
    """``forward``'s frames, all in the arm's base frame (or the world, given ``world_from_base``)."""

    bodies: np.ndarray  # (len(chain.bodies), 4, 4)
    joints: np.ndarray  # (n, 4, 4) moving frames, chain order
    tool: np.ndarray
    tcp: np.ndarray


@dataclass(frozen=True, eq=False)
class IkResult:
    ok: bool
    joints: np.ndarray  # the solution, or the closest attempt within the limits
    position_error_m: float
    orientation_error_rad: float
    failure: str | None  # ik_unreachable | ik_joint_limit | ik_target_invalid
    detail: str | None
    seed_index: int | None
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "claim_ceiling": CLAIM_CEILING, "simulator_executed": False,
                "ok": self.ok, "joints": [float(v) for v in self.joints],
                "position_error_m": _finite_or_none([self.position_error_m])[0],
                "orientation_error_rad": _finite_or_none([self.orientation_error_rad])[0],
                "failure": self.failure, "detail": self.detail, "seed_index": self.seed_index,
                "iterations": self.iterations}


@dataclass(frozen=True, eq=False)
class LinkBound:
    """One link's outer bound: extents along ``DIRECTIONS`` in the frame of ``chain.bodies[body]``."""

    name: str
    body: int
    offsets: np.ndarray | None
    vertices: np.ndarray | None  # the bound's corners, same frame
    centre: np.ndarray | None
    radius: float
    source: str | None  # collision | visual
    parts: tuple[str, ...]
    fixed: bool  # rides on the base: never moves with the joints
    unmeasured: str | None = None


@dataclass(frozen=True, eq=False)
class OrientedBox:
    world_from_box: np.ndarray
    half_extents: np.ndarray

    @classmethod
    def from_pose(cls, position_m: Sequence[float], orientation_xyzw: Sequence[float],
                  half_extents_m: Sequence[float]) -> OrientedBox:
        half = np.asarray(half_extents_m, dtype=float)
        if half.shape != (3,) or not np.isfinite(half).all() or (half < 0).any():
            raise BodyKinematicsError("clearance_box_invalid")
        return cls(pose_matrix(position_m, orientation_xyzw), half)


# --- manifest arms -------------------------------------------------------------------------------------

def arm_specs(manifest: Mapping[str, Any]) -> list[ArmSpec]:
    """Each arm a validated manifest declares, in order.

    A manifest without ``arms`` (a bare arm or a legacy v1 manifest) gives one arm from its ``tool_link``
    to the robot's world anchor, over whichever declared position joints lie on that route; its TCP is
    the tool link's origin. Both are noted on the spec.
    """
    try:
        joints = {str(row["name"]): row for row in manifest["joints"]}
        limits = {name: (float(row["lower"]), float(row["upper"])) for name, row in joints.items()
                  if row.get("control") == "position"}
        reset = {str(key): float(value) for key, value in (manifest.get("reset_joint_positions") or {}).items()}
        arms = manifest.get("arms") or []
        if arms:
            return [ArmSpec(name=str(arm["name"]), base_link=str(arm["base_link"]), tool_link=str(arm["tool_link"]),
                            joint_names=tuple(str(name) for name in arm["joint_names"]),
                            tool_from_tcp=pose_matrix(arm["tool_from_tcp"]["position_m"],
                                                      arm["tool_from_tcp"]["orientation_xyzw"]),
                            limits=limits, reset=reset) for arm in arms]
        channels = manifest.get("grippers") or ([manifest["gripper"]] if manifest.get("gripper") else [])
        fingers = {str(row["name"]) for channel in channels for row in channel["joints"]}
        names = tuple(name for name in limits if name not in fingers)
        return [ArmSpec(name="arm", base_link=None, tool_link=str(manifest["tool_link"]), joint_names=names,
                        tool_from_tcp=np.eye(4), limits=limits, reset=reset, exact_joints=False,
                        notes=("arm_spec_inferred_from_tool_link_and_world_anchor", "tcp_is_tool_link_origin"))]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise BodyKinematicsError("arm_spec_invalid") from exc


# --- model reading -------------------------------------------------------------------------------------

def _identifier(value: str) -> str:
    return _IDENTIFIER.sub("_", value)


def _source_name(prim: Any) -> str | None:
    name = prim.GetMetadata("displayName") if prim.HasAuthoredMetadata("displayName") else None
    return name if isinstance(name, str) and name and name != prim.GetName() else None


def _names(prim: Any) -> set[str]:
    source = _source_name(prim)
    return {prim.GetName()} | ({source, _identifier(source)} if source else set())


def _applied_schemas(prim: Any) -> set[str]:
    """Every API schema applied in any layer, registered here or not (PhysX's and Newton's mimic schemas
    are not part of usd-core, so ``GetAppliedSchemas`` leaves them out)."""
    names = {str(name) for name in prim.GetAppliedSchemas()}
    for spec in prim.GetPrimStack():
        listed = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else None
        if listed is not None:
            names.update(str(name) for items in (listed.explicitItems, listed.prependedItems, listed.appendedItems,
                                                 listed.addedItems) for name in items)
    return names


def _rigid(matrix: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
    """A 4x4 split into a rigid transform and its per-axis scale; shear is refused."""
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if not np.isfinite(matrix).all() or np.any(scale <= 1e-12):
        raise BodyKinematicsError("arm_chain_transform_invalid:" + path)
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise BodyKinematicsError("arm_chain_transform_invalid:" + path)
    rigid = np.eye(4)
    rigid[:3, :3], rigid[:3, 3] = rotation, matrix[:3, 3]
    return rigid, scale


def _quaternion(value: Any) -> np.ndarray:
    w, (x, y, z) = float(value.GetReal()), (float(v) for v in value.GetImaginary())
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


@dataclass(frozen=True, eq=False)
class _Edge:
    name: str
    path: str
    prim: Any
    kind: str  # revolute | prismatic | fixed | unsupported:<type>
    ends: tuple[str, str]  # owning bodies (or STATIC) of body0 and body1
    frames: tuple[np.ndarray, np.ndarray]  # each end's joint frame in its owning body's frame (stage units)
    axis: str
    limits: tuple[float, float]  # radians or stage units; +-inf when unauthored
    mimic: bool


class _Model:
    """The rigid bodies, their authored frames and the joints between them, from one opened stage."""

    def __init__(self, robot_usd: Any):
        from pxr import Usd, UsdGeom, UsdPhysics

        if isinstance(robot_usd, Usd.Stage):
            self.stage, self.path, self.sha256 = robot_usd, str(robot_usd.GetRootLayer().realPath), None
        else:
            self.path = str(robot_usd)
            try:
                self.stage = Usd.Stage.Open(self.path)
                self.sha256 = "sha256:" + hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
            except Exception as exc:  # pxr raises Tf errors on unreadable bytes
                raise BodyKinematicsError("arm_chain_robot_usd_unreadable") from exc
        root = self.stage.GetDefaultPrim() if self.stage else None
        if not root:
            raise BodyKinematicsError("arm_chain_robot_usd_invalid")
        self.root = root
        self.notes: list[str] = []
        # USD's fallback unit is the centimetre; a robot asset that never says is taken to be in metres, as
        # Isaac's importers author it and the rest of the pipeline reads it.
        self.metres = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        if not UsdGeom.StageHasAuthoredMetersPerUnit(self.stage):
            self.metres = 1.0
            self.notes.append("meters_per_unit_unauthored_metres_assumed")
        self.cache = UsdGeom.XformCache()
        self.prims = list(Usd.PrimRange(root))
        self.bodies = {str(p.GetPath()): p for p in self.prims if p.HasAPI(UsdPhysics.RigidBodyAPI)
                       and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False}
        self.world_root = self.rigid(root)
        self.edges = [edge for prim in self.prims if prim.IsA(UsdPhysics.Joint) and (edge := self._edge(prim))]

    def matrix(self, prim: Any) -> np.ndarray:
        return np.asarray(self.cache.GetLocalToWorldTransform(prim), dtype=float).T

    def rigid(self, prim: Any) -> np.ndarray:
        return _rigid(self.matrix(prim), str(prim.GetPath()))[0]

    def owner(self, prim: Any) -> str:
        """The rigid body a prim moves with, or STATIC."""
        while prim and not prim.IsPseudoRoot():
            if str(prim.GetPath()) in self.bodies:
                return str(prim.GetPath())
            prim = prim.GetParent()
        return STATIC

    def frame(self, node: str) -> np.ndarray:
        """A node's authored world frame (stage units); STATIC is the stage's world."""
        return np.eye(4) if node == STATIC else self.rigid(self.bodies[node])

    def _edge(self, prim: Any) -> _Edge | None:
        from pxr import UsdPhysics

        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            return None
        ends, frames = [], []
        for rel, position, rotation in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                                        (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())):
            targets = list(rel.GetTargets()) if rel else []
            target = self.stage.GetPrimAtPath(targets[0]) if len(targets) == 1 else None
            if len(targets) > 1 or (targets and not (target and target.IsValid())):
                self.notes.append("joint_target_invalid:" + prim.GetName())
                return None
            local = np.eye(4)
            try:
                local[:3, :3] = _quaternion(rotation.Get()) if rotation.Get() is not None else np.eye(3)
            except ValueError:
                self.notes.append("joint_frame_invalid:" + prim.GetName())
                return None
            local[:3, 3] = np.asarray(position.Get() if position.Get() is not None else (0, 0, 0), dtype=float)
            node = self.owner(target) if target is not None else STATIC
            if target is None:
                frames.append(local)
            else:
                # localPos is in the target prim's own space, scale included.
                world = self.matrix(target)
                placed = np.eye(4)
                placed[:3, :3] = _rigid(world, str(target.GetPath()))[0][:3, :3] @ local[:3, :3]
                placed[:3, 3] = world[:3, :3] @ local[:3, 3] + world[:3, 3]
                frames.append(_inverse(self.frame(node)) @ placed)
            ends.append(node)
        if ends[0] == ends[1]:
            return None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed, kind, unit = UsdPhysics.RevoluteJoint(prim), "revolute", math.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed, kind, unit = UsdPhysics.PrismaticJoint(prim), "prismatic", 1.0
        elif prim.IsA(UsdPhysics.FixedJoint):
            typed, kind, unit = None, "fixed", 1.0
        else:
            typed, kind, unit = None, "unsupported:" + prim.GetTypeName(), 1.0
        axis, limits = "Z", (-math.inf, math.inf)
        if typed is not None:
            axis = str(typed.GetAxisAttr().Get() or "X")
            values = [typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()]
            limits = tuple(float(v) * unit if v is not None and math.isfinite(float(v)) else bound
                           for v, bound in zip(values, (-math.inf, math.inf)))
        align = _AXES.get(axis)
        if align is None:
            kind = "unsupported:axis_" + axis
            align = np.eye(4)
        enabled = prim.GetAttribute("newton:mimicEnabled")
        mimic = any(name.startswith("PhysxMimicJointAPI") or (name == "NewtonMimicAPI" and not (
            enabled and enabled.Get() is False)) for name in _applied_schemas(prim))
        return _Edge(prim.GetName(), str(prim.GetPath()), prim, kind, (ends[0], ends[1]),
                     (frames[0] @ align, frames[1] @ align), axis, limits, mimic)

    def resolve(self, name: str, *, joints: bool = False) -> Any:
        """The one prim the manifest's name means; exact prim names win over recorded source names."""
        from pxr import UsdGeom, UsdPhysics

        def usable(prim: Any) -> bool:
            return prim.IsA(UsdPhysics.Joint) if joints else not (prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdGeom.Gprim))

        def unique(found: list) -> Any:
            if len(found) <= 1:
                return found[0] if found else None
            bodies = [p for p in found if str(p.GetPath()) in self.bodies]
            if len(bodies) == 1:
                return bodies[0]
            paths = {str(p.GetPath()) for p in found}
            outer = [p for p in found if not any(str(p.GetPath()).startswith(other + "/") for other in paths)]
            return outer[0] if len(outer) == 1 else False

        kind = "joint" if joints else "link"
        for found in ([p for p in self.prims if usable(p) and p.GetName() == name],
                      [p for p in self.prims if usable(p) and name in _names(p)]):
            pick = unique(found)
            if pick is False:
                raise BodyKinematicsError(f"arm_chain_{kind}_ambiguous:{name}")
            if pick is not None:
                return pick
        raise BodyKinematicsError(f"arm_chain_{kind}_missing:{name}")


def _route(edges: Sequence[_Edge], start: str, goal: str, banned: int | None = None) -> list[tuple[_Edge, bool]] | None:
    """One route of edges from ``start`` to ``goal`` (each with whether it is walked body0 to body1)."""
    links: dict[str, list[tuple[int, str, bool]]] = {}
    for index, edge in enumerate(edges):
        if index != banned:
            links.setdefault(edge.ends[0], []).append((index, edge.ends[1], True))
            links.setdefault(edge.ends[1], []).append((index, edge.ends[0], False))
    previous: dict[str, tuple[str, int, bool] | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            route = []
            while previous[node] is not None:
                node_before, index, forward = previous[node]
                route.append((edges[index], forward))
                node = node_before
            return route[::-1]
        for index, other, forward in links.get(node, []):
            if other not in previous:
                previous[other] = (node, index, forward)
                queue.append(other)
    return None


def _motion(kind: str, value: float) -> np.ndarray:
    motion = np.eye(4)
    if kind == "revolute":
        c, s = math.cos(value), math.sin(value)
        motion[:2, :2] = [[c, -s], [s, c]]
    elif kind == "prismatic":
        motion[2, 3] = value
    return motion


def _authored_value(kind: str, relative: np.ndarray) -> tuple[float, float, float]:
    """The joint value a relative joint-frame transform holds, and its off-axis translation and rotation."""
    value = 0.0
    if kind == "revolute":
        value = math.atan2(relative[1, 0], relative[0, 0])
    elif kind == "prismatic":
        value = float(relative[2, 3])
    residual = _inverse(_motion(kind, value)) @ relative
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(residual[:3, :3])) - 1.0) / 2.0)))
    return value, float(np.linalg.norm(residual[:3, 3])), angle


def arm_chain(robot_usd: Any, spec: ArmSpec, *, model: _Model | None = None) -> ArmChain:
    """The arm's kinematic chain from the robot USD (a path or an opened ``Usd.Stage``), or a typed refusal."""
    model = model or _Model(robot_usd)
    notes = list(spec.notes) + list(model.notes)
    tool_prim = model.resolve(spec.tool_link)
    tool_node = model.owner(tool_prim)
    if tool_node == STATIC:
        raise BodyKinematicsError("arm_chain_tool_not_on_body:" + spec.tool_link)
    if spec.base_link is None:
        base_node, base_world = STATIC, model.world_root
        base_name = None
    else:
        base_prim = model.resolve(spec.base_link)
        base_node, base_world, base_name = model.owner(base_prim), model.rigid(base_prim), base_prim.GetName()
    if base_node == tool_node:
        raise BodyKinematicsError("arm_chain_empty:" + spec.tool_link)
    route = _route(model.edges, base_node, tool_node)
    if route is None:
        raise BodyKinematicsError("arm_chain_base_unanchored" if spec.base_link is None
                                  else "arm_chain_disconnected:" + spec.tool_link)
    for index, (edge, _) in enumerate(route):
        # A second route exists exactly when some edge of this one is not needed: a loop or parallel linkage.
        if _route(model.edges, base_node, tool_node, banned=model.edges.index(edge)) is not None:
            raise BodyKinematicsError("arm_chain_ambiguous:" + edge.name)
        if edge.kind.startswith("unsupported"):
            raise BodyKinematicsError(f"arm_chain_joint_type_unsupported:{edge.name}:{edge.kind.split(':', 1)[1]}")
        if edge.mimic and edge.kind != "fixed":
            raise BodyKinematicsError("arm_chain_mimic_joint_unsupported:" + edge.name)
    movable = [edge for edge, _ in route if edge.kind != "fixed"]
    declared: dict[str, _Edge] = {}
    for name in spec.joint_names:
        found = [edge for edge in movable if name in _names(edge.prim)]
        if len(found) > 1:
            raise BodyKinematicsError("arm_chain_joint_ambiguous:" + name)
        if found:
            declared[name] = found[0]
        elif spec.exact_joints:
            raise BodyKinematicsError("arm_chain_joint_missing:" + name)
    for edge in movable:
        if edge not in declared.values():
            raise BodyKinematicsError("arm_chain_joint_undeclared:" + edge.name)
    names = tuple(name for name in spec.joint_names if name in declared)
    if not names:
        raise BodyKinematicsError("arm_chain_joints_missing:" + spec.tool_link)
    if not spec.exact_joints and len(names) < len(spec.joint_names):
        notes.append("declared_joints_off_this_arm:" + ",".join(n for n in spec.joint_names if n not in declared))
    position = {name: index for index, name in enumerate(names)}
    scale = model.metres

    def metric(transform: np.ndarray) -> np.ndarray:
        result = transform.copy()
        result[:3, 3] *= scale
        return result

    # Walk the route keeping ``current_from_node``: the node's frame in the latest moving frame.
    base_from_node = _inverse(base_world) @ model.frame(base_node)
    current = base_from_node
    placed, before, order, sign, kinds, authored = [], [], [], [], [], []
    if base_node != STATIC:
        placed.append((base_node, -1, current))
    lower, upper = [], []
    for edge, forward in route:
        near, far = (0, 1) if forward else (1, 0)
        a_node, b_node = edge.ends[near], edge.ends[far]
        a_from_ja, b_from_jb = edge.frames[near], edge.frames[far]
        relative = _inverse(model.frame(a_node) @ a_from_ja) @ (model.frame(b_node) @ b_from_jb)
        relative[:3, 3] *= scale
        value, offset, turn = _authored_value(edge.kind, relative)
        if offset > FRAME_TOLERANCE_M or turn > FRAME_TOLERANCE_RAD:
            notes.append("joint_frames_inconsistent_at_authored_pose:" + edge.name)
        if edge.kind == "fixed":
            current = current @ a_from_ja @ _inverse(b_from_jb)
        else:
            before.append(metric(current @ a_from_ja))
            name = next(n for n in names if declared[n] is edge)
            order.append(position[name])
            sign.append(1.0 if forward else -1.0)
            kinds.append(edge.kind)
            authored.append((name, value if forward else -value))
            current = _inverse(b_from_jb)
            low, high = edge.limits
            if edge.kind == "prismatic":
                low, high = low * scale, high * scale
            if spec.limits.get(name) is not None:
                window = spec.limits[name]
                if window[0] < low - LIMIT_TOLERANCE or window[1] > high + LIMIT_TOLERANCE:
                    notes.append("manifest_limits_beyond_model:" + name)
                low, high = max(low, window[0]), min(high, window[1])
            if not low <= high:
                raise BodyKinematicsError("arm_chain_joint_limits_disjoint:" + name)
            if not (math.isfinite(low) and math.isfinite(high)):
                notes.append("joint_unlimited:" + name)
            lower.append((position[name], low))
            upper.append((position[name], high))
        if b_node != STATIC:
            placed.append((b_node, len(before) - 1, current))
    tool_from_node = _inverse(model.rigid(tool_prim)) @ model.frame(tool_node)
    # A body welded by fixed joints to a moving link (a wrist camera, a sensor mount) moves with it.
    on_route = {node for node, _, _ in placed} | {STATIC}
    bodies = []
    for node, joint, node_frame in placed:
        queue = deque([(node, node_frame)])
        while queue:
            here, here_frame = queue.popleft()
            prim = model.bodies[here]
            bodies.append(ChainBody(prim.GetName(), _source_name(prim) or prim.GetName(), here, joint, metric(here_frame)))
            for edge in (model.edges if joint >= 0 else ()):
                if edge.kind != "fixed" or here not in edge.ends:
                    continue
                near = edge.ends.index(here)
                other = edge.ends[1 - near]
                if other not in on_route:
                    on_route.add(other)
                    queue.append((other, here_frame @ edge.frames[near] @ _inverse(edge.frames[1 - near])))
    joint_from_tool = metric(current @ _inverse(tool_from_node))
    n = len(names)
    low, high = np.full(n, -math.inf), np.full(n, math.inf)
    for index, value in lower:
        low[index] = value
    for index, value in upper:
        high[index] = value
    values = dict(authored)
    authored_q = np.array([values[name] for name in names])
    reset = np.array([spec.reset.get(name, math.nan) for name in names], dtype=float)
    if not np.isfinite(reset).all():
        notes.append("reset_unavailable_mid_range_used")
        reset = _middle(low, high)
    if np.any(reset < low - 1e-9) or np.any(reset > high + 1e-9):
        notes.append("reset_outside_model_limits_clipped")
    reset = np.clip(reset, low, high)
    before_array = np.asarray(before)
    tcp_offset = float(np.linalg.norm((joint_from_tool @ spec.tool_from_tcp)[:3, 3]))
    travel = sum(max(abs(low[o]), abs(high[o])) for o, kind in zip(order, kinds) if kind == "prismatic")
    reach = float(sum(np.linalg.norm(before_array[1:, :3, 3], axis=1)) + tcp_offset + travel)
    root_from_base = _inverse(model.world_root) @ base_world
    root_from_base[:3, 3] *= scale
    # Is the base itself moved by a joint between it and the world anchor? Then its root pose is authored.
    anchor = _route(model.edges, STATIC, base_node) if base_node != STATIC else []
    if anchor is None:
        notes.append("base_not_anchored_to_world")
    elif any(edge.kind != "fixed" for edge, _ in anchor):
        notes.append("base_moves_relative_to_world_authored_pose_used")
    steps = np.array([IK_MAX_PRISMATIC_STEP_M if kind == "prismatic" else IK_MAX_REVOLUTE_STEP_RAD
                      for kind in kinds])
    max_step = np.empty(n)
    max_step[np.asarray(order)] = steps
    kinds_by_q = [""] * n
    for o, kind in zip(order, kinds):
        kinds_by_q[o] = kind
    return ArmChain(name=spec.name, robot_usd=model.path, robot_usd_sha256=model.sha256, base_link=base_name,
                    tool_link=tool_prim.GetName(), tool_path=str(tool_prim.GetPath()), joint_names=names,
                    joint_kinds=tuple(kinds_by_q), lower=low, upper=high, reset=reset, authored=authored_q,
                    bodies=tuple(bodies), tool_from_tcp=np.asarray(spec.tool_from_tcp, dtype=float),
                    root_from_base=root_from_base, reach_bound_m=reach, notes=tuple(dict.fromkeys(notes)),
                    order=np.asarray(order), sign=np.asarray(sign), prismatic=np.asarray([k == "prismatic" for k in kinds]),
                    before=before_array, joint_from_tool=joint_from_tool, max_step=max_step)


def body_arm_chains(robot_usd: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Every manifest arm's chain, or the typed reason it stays unassessed; one arm never stops another."""
    chains, unassessed = {}, {}
    try:
        specs = arm_specs(manifest)
        model = _Model(robot_usd)
    except BodyKinematicsError as exc:
        return {"chains": {}, "unassessed": {"*": exc.code}}
    for spec in specs:
        try:
            chains[spec.name] = arm_chain(robot_usd, spec, model=model)
        except BodyKinematicsError as exc:
            unassessed[spec.name] = exc.code
    return {"chains": chains, "unassessed": unassessed}


# --- forward kinematics --------------------------------------------------------------------------------

def _vector(chain: ArmChain, joints: Any) -> np.ndarray:
    q = np.asarray(joints, dtype=float).reshape(-1)
    if q.shape != (len(chain.joint_names),) or not np.isfinite(q).all():
        raise BodyKinematicsError("joint_vector_invalid")
    return q


def _moving_frames(chain: ArmChain, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each joint's moving frame in the base frame (chain order), and the tool link's frame."""
    frames = np.empty((len(chain.order), 4, 4))
    current = np.eye(4)
    for index in range(len(chain.order)):
        current = current @ chain.before[index]
        value = chain.sign[index] * q[chain.order[index]]
        if chain.prismatic[index]:
            current[:, 3] += value * current[:, 2]
        else:
            c, s = math.cos(value), math.sin(value)
            x, y = current[:, 0].copy(), current[:, 1].copy()
            current[:, 0], current[:, 1] = c * x + s * y, c * y - s * x
        frames[index] = current
    return frames, current @ chain.joint_from_tool


def forward(chain: ArmChain, joints: Any, *, world_from_base: np.ndarray | None = None) -> ArmPose:
    """Every chain body's frame, the joint frames, the tool link and the TCP for one joint vector."""
    frames, tool = _moving_frames(chain, _vector(chain, joints))
    bodies = np.asarray([(frames[body.joint] if body.joint >= 0 else np.eye(4)) @ body.joint_from_body
                         for body in chain.bodies]).reshape(-1, 4, 4)
    tcp = tool @ chain.tool_from_tcp
    if world_from_base is not None:
        frame = np.asarray(world_from_base, dtype=float)
        return ArmPose(frame @ bodies, frame @ frames, frame @ tool, frame @ tcp)
    return ArmPose(bodies, frames, tool, tcp)


# --- inverse kinematics --------------------------------------------------------------------------------

def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """A rotation matrix's axis times its angle, stable at small angles and at a half turn."""
    cos = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    sin = float(np.linalg.norm(skew)) / 2.0
    angle = math.atan2(sin, cos)
    if angle < 1e-6:
        return skew / 2.0
    if math.pi - angle > 1e-3:
        return skew * (angle / (2.0 * sin))
    symmetric = (rotation + rotation.T) / 2.0 - cos * np.eye(3)
    column = int(np.argmax(np.diag(symmetric)))
    axis = symmetric[:, column] / math.sqrt(max(symmetric[column, column] * (1.0 - cos), 1e-300))
    return axis * angle if float(axis @ skew) >= 0.0 else -axis * angle


def _middle(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """The middle of each joint's limits; its one finite limit, or 0, when it has fewer."""
    finite_low, finite_high = np.isfinite(low), np.isfinite(high)
    low, high = np.where(finite_low, low, 0.0), np.where(finite_high, high, 0.0)
    return np.where(finite_low & finite_high, (low + high) / 2.0, np.where(finite_low, low, high))


def _wrapped(chain: ArmChain, q: np.ndarray) -> np.ndarray:
    """Each revolute joint turned by whole turns toward its limits' middle, then clipped into them."""
    turns = np.where(chain.prismatic[np.argsort(chain.order)], 0.0,
                     np.round((_middle(chain.lower, chain.upper) - q) / (2.0 * math.pi)))
    return np.clip(q + 2.0 * math.pi * turns, chain.lower, chain.upper)


def ik_seeds(chain: ArmChain, seeds: Sequence[Sequence[float]] | None = None, *, extra_seeds: bool = True) -> list[np.ndarray]:
    """The starts ``solve`` tries, in order: the caller's, the manifest reset, the middle of the limits, and
    ``IK_EXTRA_SEEDS`` drawn within the limits from a fixed seed."""
    rows = [_vector(chain, seed) for seed in (seeds or [])]
    rows.append(chain.reset)
    if extra_seeds:
        low = np.where(np.isfinite(chain.lower), chain.lower, -math.pi)
        high = np.where(np.isfinite(chain.upper), chain.upper, math.pi)
        rows.append(_middle(chain.lower, chain.upper))
        generator = np.random.default_rng(IK_SEED)
        rows.extend(low + (high - low) * generator.random(len(low)) for _ in range(IK_EXTRA_SEEDS))
    unique: list[np.ndarray] = []
    for row in rows:
        row = np.clip(row, chain.lower, chain.upper)
        if not any(np.allclose(row, other) for other in unique):
            unique.append(row)
    return unique


def _dls(chain: ArmChain, seed: np.ndarray, target: np.ndarray, weight: float, position_tolerance: float,
         orientation_tolerance: float, iterations: int, limited: bool) -> tuple[np.ndarray, float, float, int, bool]:
    """Damped least squares from one seed; returns joints, position and angle errors, iterations, success."""
    low, high = (chain.lower, chain.upper) if limited else (np.full_like(chain.lower, -np.inf), np.full_like(chain.upper, np.inf))
    rows = 6 if weight > 0 else 3
    tcp_from_tool = chain.tool_from_tcp

    def state(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        frames, tool = _moving_frames(chain, q)
        tcp = tool @ tcp_from_tool
        error = np.empty(rows)
        error[:3] = target[:3, 3] - tcp[:3, 3]
        turn = _rotation_vector(target[:3, :3] @ tcp[:3, :3].T)
        if rows == 6:
            error[3:] = weight * turn
        return frames, tcp, error, float(np.linalg.norm(turn))

    q = np.clip(seed, low, high)
    frames, tcp, error, angle = state(q)
    damping, count = 0.05, 0
    for count in range(1, iterations + 1):
        position = float(np.linalg.norm(error[:3]))
        if position < position_tolerance and (rows == 3 or angle < orientation_tolerance):
            return q, position, angle, count - 1, True
        axes = frames[:, :3, 2] * chain.sign[:, None]
        jacobian = np.zeros((rows, len(q)))
        revolute = ~chain.prismatic
        linear = np.where(revolute[:, None], np.cross(axes, tcp[:3, 3] - frames[:, :3, 3]), axes)
        jacobian[:3, chain.order] = linear.T
        if rows == 6:
            jacobian[3:, chain.order] = (weight * axes * revolute[:, None]).T
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(rows), error)
        blocked = ((q <= low + 1e-12) & (step < 0)) | ((q >= high - 1e-12) & (step > 0))
        if blocked.any():
            jacobian[:, blocked] = 0.0
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(rows), error)
        ratio = float(np.max(np.abs(step) / chain.max_step))
        if ratio > 1.0:
            step /= ratio
        candidate = np.clip(q + step, low, high)
        candidate_state = state(candidate)
        if float(candidate_state[2] @ candidate_state[2]) < float(error @ error):
            q, (frames, tcp, error, angle) = candidate, candidate_state
            damping = max(damping * 0.5, 1e-4)
        else:
            damping = min(damping * 3.0, 2.0)
            if damping >= 2.0 and float(np.linalg.norm(step)) < 1e-9:
                break
    position = float(np.linalg.norm(error[:3]))
    return q, position, angle, count, position < position_tolerance and (rows == 3 or angle < orientation_tolerance)


def solve(chain: ArmChain, base_from_tcp: Any, *, seeds: Sequence[Sequence[float]] | None = None,
          extra_seeds: bool = True, classify: bool = True, orientation_weight_m: float = IK_ORIENTATION_WEIGHT_M,
          position_tolerance_m: float = IK_POSITION_TOLERANCE_M,
          orientation_tolerance_rad: float = IK_ORIENTATION_TOLERANCE_RAD,
          max_iterations: int = IK_MAX_ITERATIONS) -> IkResult:
    """The arm's joints holding the TCP at ``base_from_tcp`` (4x4, base frame), or a typed miss.

    ``orientation_weight_m`` 0 solves position only. A miss is ``ik_unreachable`` with detail
    ``beyond_reach_bound`` (provable from the chain) or ``no_solution_from_seeds`` (a search result: no
    start reached it even ignoring joint limits), or ``ik_joint_limit`` (a start reached it only outside
    the limits). ``classify=False`` skips that second search and reports ``ik_unreachable``
    ``not_classified``.
    """
    target = np.asarray(base_from_tcp, dtype=float)
    rotation = target[:3, :3] if target.shape == (4, 4) else None
    if (rotation is None or not np.isfinite(target).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        return IkResult(False, chain.reset.copy(), math.inf, math.inf, "ik_target_invalid", None, None, 0)
    weight = max(0.0, float(orientation_weight_m))
    starts = ik_seeds(chain, seeds, extra_seeds=extra_seeds)
    origin = chain.before[0][:3, 3]
    if float(np.linalg.norm(target[:3, 3] - origin)) > chain.reach_bound_m + position_tolerance_m:
        q = starts[0]
        tcp = forward(chain, q).tcp
        return IkResult(False, q, float(np.linalg.norm(target[:3, 3] - tcp[:3, 3])),
                        float(np.linalg.norm(_rotation_vector(target[:3, :3] @ tcp[:3, :3].T))),
                        "ik_unreachable", "beyond_reach_bound", None, 0)
    best, used = None, 0
    for index, seed in enumerate(starts):
        q, position, angle, count, ok = _dls(chain, seed, target, weight, position_tolerance_m,
                                             orientation_tolerance_rad, max_iterations, True)
        used += count
        if ok:
            return IkResult(True, q, position, angle, None, None, index, used)
        score = position + weight * angle
        if best is None or score < best[0]:
            best = (score, q, position, angle)
    detail, failure = "not_classified", "ik_unreachable"
    if classify:
        detail = "no_solution_from_seeds"
        for index, seed in enumerate(starts):
            free, *_, count, ok = _dls(chain, seed, target, weight, position_tolerance_m, orientation_tolerance_rad,
                                       max_iterations, False)
            used += count
            if not ok:
                continue
            # A revolute joint a whole turn out holds the same pose a turn back; start from there, within limits.
            q, position, angle, count, ok = _dls(chain, _wrapped(chain, free), target, weight, position_tolerance_m,
                                                 orientation_tolerance_rad, max_iterations, True)
            used += count
            if ok:
                return IkResult(True, q, position, angle, None, "solved_from_unlimited_start", index, used)
            failure, detail = "ik_joint_limit", "found_only_outside_joint_limits"
            break
    return IkResult(False, best[1], best[2], best[3], failure, detail, None, used)


# --- link bounds ---------------------------------------------------------------------------------------

def _support(prim: Any, body_from_prim: np.ndarray) -> np.ndarray | None:
    """The prim's largest extent along each of ``DIRECTIONS`` in the body frame (stage units), exactly."""
    from pxr import UsdGeom

    linear, origin = body_from_prim[:3, :3], body_from_prim[:3, 3]
    local = DIRECTIONS @ linear  # each direction, in the prim's own (scaled) coordinates

    def lateral(axis: int) -> np.ndarray:
        return np.sqrt(np.maximum(0.0, (local ** 2).sum(axis=1) - local[:, axis] ** 2))

    if prim.IsA(UsdGeom.Mesh):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
        if not len(points):
            return None
        return ((points @ linear.T + origin) @ DIRECTIONS.T).max(axis=0)
    if prim.IsA(UsdGeom.Cube):
        extent = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * np.abs(local).sum(axis=1)
    elif prim.IsA(UsdGeom.Sphere):
        extent = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * np.linalg.norm(local, axis=1)
    elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
        shape = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
        axis = "XYZ".index(str(shape.GetAxisAttr().Get()))
        radius, half = float(shape.GetRadiusAttr().Get()), 0.5 * float(shape.GetHeightAttr().Get())
        across = lateral(axis) if prim.IsA(UsdGeom.Cylinder) else np.linalg.norm(local, axis=1)
        extent = half * np.abs(local[:, axis]) + radius * across
    elif prim.IsA(UsdGeom.Cone):
        cone = UsdGeom.Cone(prim)
        axis = "XYZ".index(str(cone.GetAxisAttr().Get()))
        radius, half = float(cone.GetRadiusAttr().Get()), 0.5 * float(cone.GetHeightAttr().Get())
        extent = np.maximum(half * local[:, axis], -half * local[:, axis] + radius * lateral(axis))
    else:
        return None
    return DIRECTIONS @ origin + extent


def _geometry(model: _Model, body: str, frame_from_world: np.ndarray) -> tuple[list[np.ndarray], list[str], str | None, list[str]]:
    """The supports of the geometry that moves with ``body`` (collision preferred), in another frame."""
    from pxr import Usd, UsdGeom, UsdPhysics

    collision, visual, names, skipped = [], [], [], []
    iterator = iter(Usd.PrimRange(model.bodies[body], Usd.TraverseInstanceProxies()))
    for prim in iterator:
        path = str(prim.GetPath())
        if path != body and path in model.bodies:
            iterator.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        world = model.matrix(prim)
        support = _support(prim, frame_from_world @ world)
        if support is None:
            skipped.append(prim.GetTypeName())
            continue
        enabled = prim.HasAPI(UsdPhysics.CollisionAPI) and \
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        (collision if enabled else visual).append(support)
        names.append(prim.GetName())
    if collision:
        return collision, names, "collision", skipped
    return visual, names, ("visual" if visual else None), skipped


def _corners(offsets: np.ndarray) -> np.ndarray:
    """The vertices of the polytope ``DIRECTIONS @ x <= offsets``."""
    from scipy.optimize import linprog
    from scipy.spatial import HalfspaceIntersection

    for pad in (0.0, 1e-6, 1e-5):
        bound = offsets + pad
        norms = np.linalg.norm(DIRECTIONS, axis=1)
        centre = linprog(np.r_[0.0, 0.0, 0.0, -1.0], A_ub=np.c_[DIRECTIONS, norms], b_ub=bound,
                         bounds=[(None, None)] * 3 + [(0.0, None)])
        if centre.status == 0 and centre.x[3] > 1e-9:
            hull = HalfspaceIntersection(np.c_[DIRECTIONS, -bound], centre.x[:3])
            return np.unique(np.round(hull.intersections, 9), axis=0)
    raise BodyKinematicsError("link_bound_degenerate")


def _bound(name: str, body: int, supports: list[np.ndarray], parts: Sequence[str], source: str | None,
           fixed: bool, metres: float, missing: str) -> LinkBound:
    if not supports:
        return LinkBound(name, body, None, None, None, 0.0, None, tuple(parts), fixed, missing)
    offsets = np.ceil(np.max(supports, axis=0) * metres * 1e4) / 1e4
    vertices = _corners(offsets)
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    return LinkBound(name, body, offsets, vertices, centre, float(np.linalg.norm(vertices - centre, axis=1).max()),
                     source, tuple(parts), fixed)


def link_bounds(robot_usd: Any, chain: ArmChain, *, hand: bool = True,
                hand_exclude: Sequence[str] = ()) -> tuple[LinkBound, ...]:
    """Each chain body's bound in its own frame, and (``hand``) one bound for every body beyond the tool link.

    The hand is measured at its authored pose (its finger joints as the model holds them), in the tool
    link's body frame; ``hand_exclude`` names bodies left out, such as the contact pads meant to touch the
    object. A body with no geometry gets ``unmeasured``; it is never treated as empty space.
    """
    model = _Model(robot_usd)
    bounds = []
    for index, body in enumerate(chain.bodies):
        supports, parts, source, skipped = _geometry(model, body.path, _inverse(model.frame(body.path)))
        missing = "link_geometry_missing" + (":skipped_" + ",".join(sorted(set(skipped))) if skipped else "")
        bounds.append(_bound(body.name, index, supports, parts, source, body.joint < 0, model.metres, missing))
    if not hand:
        return tuple(bounds)
    tool_index = next(index for index, body in enumerate(chain.bodies) if body.path == model.owner(
        model.stage.GetPrimAtPath(chain.tool_path)))
    on_chain = {body.path for body in chain.bodies} | {STATIC}
    tool = chain.bodies[tool_index].path
    seen, queue, hand_bodies = {tool}, deque([tool]), []
    while queue:
        node = queue.popleft()
        for edge in model.edges:
            if node in edge.ends:
                other = edge.ends[1] if edge.ends[0] == node else edge.ends[0]
                if other not in seen and other not in on_chain:
                    seen.add(other)
                    queue.append(other)
                    hand_bodies.append(other)
    excluded = set(hand_exclude)
    frame = _inverse(model.frame(tool))
    supports, parts, sources = [], [], set()
    for path in hand_bodies:
        prim = model.bodies[path]
        if _names(prim) & excluded:
            continue
        found, _, source, _ = _geometry(model, path, frame)
        if found:
            supports.extend(found)
            parts.append(prim.GetName())
            sources.add(source)
    if hand_bodies:
        source = "collision" if sources == {"collision"} else ("visual" if sources == {"visual"} else
                                                               ("mixed" if sources else None))
        bounds.append(_bound("hand", tool_index, supports, parts, source, False, model.metres, "hand_geometry_missing"))
    return tuple(bounds)


# --- clearance -----------------------------------------------------------------------------------------

def _box_gap(vertices: np.ndarray, frame: np.ndarray, box: OrientedBox) -> float:
    """A lower bound on the distance from a placed bound to a box (separating axes); negative when they
    overlap along every tested axis."""
    corners = vertices @ frame[:3, :3].T + frame[:3, 3]
    axes = np.vstack([DIRECTIONS @ frame[:3, :3].T, box.world_from_box[:3, :3].T])
    projected = corners @ axes.T
    centre = axes @ box.world_from_box[:3, 3]
    reach = np.abs(axes @ box.world_from_box[:3, :3]) @ box.half_extents
    return float(np.max(np.maximum(centre - reach - projected.max(axis=0), projected.min(axis=0) - centre - reach)))


def clearance(chain: ArmChain, bounds: Sequence[LinkBound], joints: Any, *, world_from_base: np.ndarray | None = None,
              points: Any = None, boxes: Sequence[OrientedBox] = (), margin_m: float = 0.0,
              search_m: float = CLEARANCE_SEARCH_M, include_fixed: bool = False,
              skip: Sequence[str] = ()) -> dict[str, Any]:
    """The least gap from the posed arm's bounds to scene ``points`` and ``boxes``, and the link that has it.

    ``points`` is an (N, 3) world array or anything with ``near(centre, radius)`` (``SceneObstacles``).
    Points farther than ``search_m`` from a bound are not measured; a link with nothing that near has
    gap None (clear by at least ``search_m``). Gaps are lower bounds on the true distance. A bound
    without geometry is listed in ``unassessed_links``, never counted clear.
    """
    pose = forward(chain, joints, world_from_base=world_from_base)
    cloud = None
    if points is not None and not hasattr(points, "near"):
        cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    reach = max(float(search_m), float(margin_m))
    least, which, against, per_link, unassessed = None, None, None, {}, {}
    for bound in bounds:
        if bound.name in skip or (bound.fixed and not include_fixed):
            continue
        if bound.offsets is None:
            unassessed[bound.name] = bound.unmeasured
            continue
        frame = pose.bodies[bound.body]
        gaps = []
        if points is not None:
            middle = frame[:3, :3] @ bound.centre + frame[:3, 3]
            near = (points.near(middle, bound.radius + reach) if cloud is None
                    else cloud[np.linalg.norm(cloud - middle, axis=1) <= bound.radius + reach])
            if len(near):
                local = (np.asarray(near, dtype=float) - frame[:3, 3]) @ frame[:3, :3]
                gaps.append((float(np.min(np.max(local @ DIRECTIONS.T - bound.offsets, axis=1))), "points"))
        gaps += [(_box_gap(bound.vertices, frame, box), f"box:{index}") for index, box in enumerate(boxes)]
        gap = min(gaps, default=None)
        per_link[bound.name] = None if gap is None else round(gap[0], 6)
        if gap is not None and (least is None or gap[0] < least):
            least, which, against = gap[0], bound.name, gap[1]
    return {"schema_version": SCHEMA_VERSION, "claim_ceiling": CLAIM_CEILING, "simulator_executed": False,
            "arm": chain.name, "minimum_clearance_m": None if least is None else round(least, 6),
            "link": which, "obstacle": against, "margin_m": margin_m,
            "colliding": least is not None and least < margin_m, "search_m": reach, "per_link": per_link,
            "unassessed_links": unassessed}
