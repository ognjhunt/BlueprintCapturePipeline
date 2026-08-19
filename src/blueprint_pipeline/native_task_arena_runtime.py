"""Build the native Isaac Lab Arena environment from a sealed scene plan.

The public compiler in :mod:`native_task_arena_scene_plan` performs all
filesystem and task binding decisions off-GPU.  This module is the deliberately
thin native adapter: it maps that plan onto the pinned Arena APIs and returns
the exact scene handles needed by the episode/readback adapter.  It contains no
scene id, object label, canned-beverage constant, or refrigerator coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest

#: Every logical contact sensor the scene plan is allowed to emit.
#:
#: This is the consumer half of a two-sided vocabulary: the scene plan writes
#: these ids and this runtime admits them.  When the halves drift the scene
#: still builds, the packet still digests, and the divergence is only found by
#: a GPU that has already been rented -- which is how
#: ``robot_task_forbidden_collision`` was found, on the second paid Arena
#: attempt.  ``tests/test_native_task_arena_runtime.py`` reads the producer's
#: literals and fails if this set does not cover them.
LOGICAL_CONTACT_SENSOR_IDS = frozenset(
    {
        # the fingertips closing on the task object: the contact that counts
        "task_robot_contact",
        # the task object resting against, or striking, the static scene
        "task_scene_contact",
        "task_scene_collision",
        # the task object against the surface it is supported by
        "task_support_contact",
        # any robot link touching the static scene
        "robot_scene_contact",
        # non-fingertip robot links -- knuckles, outer fingers, wrist --
        # striking the task object body. A forbidden contact, not a grasp.
        "robot_task_forbidden_collision",
    }
)
PINHOLE_HORIZONTAL_APERTURE_MM = 20.955
SCENE_PLAN_SCHEMA = "native_task_arena_scene_plan.v1"


class NativeTaskArenaRuntimeError(ValueError):
    """Stable configuration failures at the native adapter boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class NativeTaskArenaEnvironment:
    env: Any
    cfg: Any
    plan: Mapping[str, Any]
    scene_asset_names: Mapping[str, str]
    contact_sensor_names: Mapping[str, tuple[str, ...]]
    camera_scene_names: Mapping[str, str]
    preconstruction_device_binding: Mapping[str, Any] | None = None
    native_configuration_readback: Mapping[str, Any] | None = None


def _validated_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        plan = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_plan_invalid"]
        ) from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != SCENE_PLAN_SCHEMA:
        raise NativeTaskArenaRuntimeError(["native_task_arena_runtime_plan_invalid"])
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_plan_digest_invalid"]
        )
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _resolve_portable_assets(
    plan: Mapping[str, Any], *, bundle_root: str | Path | None
) -> list[dict[str, Any]]:
    """Resolve and reverify relative packet assets without changing the seal."""

    objects = json.loads(json.dumps(plan["objects"]))
    relative = [row for row in objects if not Path(str(row["usd_path"])).is_absolute()]
    if not relative:
        return objects
    if bundle_root is None:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_bundle_root_required"]
        )
    raw_root = Path(bundle_root).expanduser()
    if raw_root.is_symlink():
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_bundle_root_invalid"]
        )
    root = raw_root.resolve()
    if not root.is_dir():
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_bundle_root_invalid"]
        )
    errors: list[str] = []
    for row in relative:
        role = str(row.get("semantic_role") or "")
        relative_path = str(row["usd_path"])
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts or not pure.name:
            errors.append(f"native_task_arena_runtime_asset_path_invalid:{role}")
            continue
        candidate = root.joinpath(*pure.parts)
        resolved = candidate.resolve()
        outside = resolved != root and root not in resolved.parents
        try:
            expected_size = int(row["size_bytes"])
            expected_digest = str(row["sha256"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"native_task_arena_runtime_asset_identity_invalid:{role}")
            continue
        if (
            _has_symlink_component(candidate, root=root)
            or outside
            or not resolved.is_file()
        ):
            errors.append(f"native_task_arena_runtime_asset_missing:{role}")
            continue
        if resolved.stat().st_size != expected_size or _sha256(resolved) != expected_digest:
            errors.append(f"native_task_arena_runtime_asset_identity_mismatch:{role}")
            continue
        row["usd_path"] = str(resolved)
    if errors:
        raise NativeTaskArenaRuntimeError(errors)
    return objects


def _rotation_matrix_to_xyzw(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Convert a proper 3x3 rotation to a canonical XYZW quaternion."""

    m00, m01, m02 = (float(value) for value in matrix[0])
    m10, m11, m12 = (float(value) for value in matrix[1])
    m20, m21, m22 = (float(value) for value in matrix[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_camera_rotation_invalid"]
        )
    quaternion = [qx / norm, qy / norm, qz / norm, qw / norm]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return quaternion


def camera_runtime_parameters(camera: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one calibrated OpenCV pose/intrinsics row to Isaac CameraCfg data."""

    role = str(camera.get("role") or "")
    matrix = list(camera.get("frame_from_camera_matrix") or [])
    intrinsics = dict(camera.get("intrinsics") or {})
    if len(matrix) != 16 or camera.get("optical_convention") != "opencv":
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_contract_invalid:{role}"]
        )
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_intrinsics_invalid:{role}"]
        ) from exc
    if (
        not math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6)
        or not math.isclose(cx, (width - 1) / 2.0, abs_tol=1e-6)
        or not math.isclose(cy, (height - 1) / 2.0, abs_tol=1e-6)
    ):
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_intrinsics_not_representable:{role}"]
        )
    rotation = [matrix[0:3], matrix[4:7], matrix[8:11]]
    pose_frame = str(camera.get("pose_frame") or "")
    parent = str(camera.get("parent_prim_path") or "")
    expected_frame = "robot_body" if role == "wrist" else "world"
    if (
        pose_frame != expected_frame
        or not parent
        or (pose_frame == "world" and parent != "{ENV_REGEX_NS}")
        or (
            pose_frame == "robot_body"
            and not parent.startswith("{ENV_REGEX_NS}/Robot/")
        )
    ):
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_parent_invalid:{role}"]
        )
    runtime_name = {
        "external": "external_camera",
        "wrist": "wrist_camera",
        "overview": "external_camera_2",
    }.get(role)
    if runtime_name is None:
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_role_invalid:{role}"]
        )
    return {
        "role": role,
        "runtime_name": runtime_name,
        "prim_path": f"{parent}/{runtime_name}",
        "pose_frame": pose_frame,
        "parent_prim_path": parent,
        "offset_position_m": [matrix[3], matrix[7], matrix[11]],
        "offset_rotation_xyzw": _rotation_matrix_to_xyzw(rotation),
        # Isaac Lab names the OpenCV optical frame convention "ros": +Z
        # forward, +X right, +Y down.  The plan retains the source name.
        "isaac_offset_convention": "ros",
        "source_optical_convention": "opencv",
        "width": width,
        "height": height,
        "focal_length_mm": fx * PINHOLE_HORIZONTAL_APERTURE_MM / width,
        "horizontal_aperture_mm": PINHOLE_HORIZONTAL_APERTURE_MM,
        "vertical_aperture_mm": PINHOLE_HORIZONTAL_APERTURE_MM * height / width,
        "data_types": (
            ["rgb", "distance_to_camera", "semantic_segmentation"]
            if role in {"external", "wrist"}
            else ["rgb", "semantic_segmentation"]
        ),
        "policy_input": bool(camera["policy_input"]),
        "review_only": bool(camera["review_only"]),
    }


#: Channels recorded when the plan supplies them but never required.  A
#: forbidden-contact channel is diagnostic: its absence is not a defect, and
#: its presence must not be mistaken for an unknown channel.
OPTIONAL_CONTACT_CHANNELS = frozenset(
    {"task_scene_collision", "robot_task_forbidden_collision"}
)


def required_contact_channels(task_kind: str) -> frozenset[str]:
    """The channels a task kind cannot be scored without."""

    if task_kind == "articulated_open_close":
        return frozenset(
            {"task_robot_contact", "task_scene_contact", "robot_scene_contact"}
        )
    return frozenset(
        {"task_robot_contact", "task_support_contact", "robot_scene_contact"}
    )


def _invalid_exact_contact_path(path: Any) -> bool:
    value = str(path)
    return not value.startswith("{ENV_REGEX_NS}/") or any(
        token in value for token in ("*", ".*", "[", "]")
    )


#: What PhysX requires of an authored articulation, and the one adaptation it
#: permits.  Kept next to the runtime that applies it so the host-side gate and
#: the GPU agree.
KINEMATIC_ARTICULATION_ADAPTATION = "dynamic_articulation_with_world_fixed_base"


def articulation_kinematic_adaptation(
    stage: Any, *, articulation_root_path: str, usd_physics: Any
) -> dict[str, Any]:
    """Decide how a kinematic articulated link must be adapted for PhysX.

    A kinematic link is valid authored USD -- it is how an appliance is told it
    does not move -- but PhysX tensor articulations reject any articulation
    containing one, so the articulation is never created and every later lookup
    reports that nothing matched.

    Exactly one kinematic articulated link has an unambiguous meaning: it is
    the fixed base. Spawn it dynamic and ground it with a world fixed joint,
    which preserves the authored behaviour (it still does not move) without
    touching the sealed asset's bytes. More than one is ambiguous about which
    part of the asset is anchored, so it fails closed rather than silently
    choosing.

    This is the rule the articulated native probe already proved on hardware;
    the Arena lane spawned the same sealed asset without it and spent an
    attempt on ``Failed to create articulation``.
    """

    joints = [
        prim
        for prim in stage.Traverse()
        if usd_physics.Joint(prim)
        and prim.GetPath().pathString.startswith(articulation_root_path + "/")
    ]
    body_paths: set[str] = set()
    for joint in joints:
        for relationship_name in ("physics:body0", "physics:body1"):
            relationship = joint.GetRelationship(relationship_name)
            targets = relationship.GetTargets() if relationship else []
            if len(targets) != 1:
                continue
            body_path = str(targets[0])
            if body_path.startswith(articulation_root_path + "/"):
                body_paths.add(body_path)

    kinematic_paths = sorted(
        path
        for path in body_paths
        if bool(
            usd_physics.RigidBodyAPI(
                stage.GetPrimAtPath(path)
            ).GetKinematicEnabledAttr().Get()
        )
    )
    if len(kinematic_paths) > 1:
        raise NativeTaskArenaRuntimeError(
            [
                "native_task_arena_nonhomogeneous_kinematic_articulation:"
                + ",".join(kinematic_paths)
            ]
        )
    return {
        "articulation_body_prim_paths": sorted(body_paths),
        "authored_kinematic_body_prim_paths": kinematic_paths,
        "fixed_base_body_prim_path": kinematic_paths[0] if kinematic_paths else None,
        "adaptation": (
            KINEMATIC_ARTICULATION_ADAPTATION
            if kinematic_paths
            else "candidate_authored_dynamic_articulation"
        ),
        "candidate_bytes_modified": False,
    }


#: Authored INSIDE the articulation root: Isaac Lab spawns an asset by
#: referencing its file, and USD reference composition carries only the
#: default prim's subtree -- a sibling scope is silently dropped, so the
#: spawned articulation arrives without its anchor. Attempt r6 paid $0.065 to
#: discover that; the composition test now pins it with pxr alone. A world
#: fixed joint inside the root is also the canonical PhysX fixed-base
#: articulation pattern.
ARENA_ANCHOR_JOINT = "fixed_base_anchor"


def author_grounded_articulation(
    source_usd: str | Path, destination: str | Path
) -> dict[str, Any] | None:
    """Author the probe's proven fixed-base mechanism into a derived asset.

    Isaac Lab's ``fix_root_link`` raises NotImplementedError for an
    articulation whose root prim is not itself a rigid body -- exactly the
    topology every sealed asset here uses (`ArticulationRootAPI` on the root
    Xform, links beneath). The articulated native probe never used that API:
    it authors ``kinematicEnabled = 0`` on the base link plus a world
    ``PhysicsFixedJoint`` whose ``body1`` is that link, and that stage measured
    this washer's door on a GPU. This writes the same adaptation into a
    derived copy of the sealed asset, on the host, where the result is
    verifiable with pxr before any spend.

    Returns the adaptation record, or ``None`` when the asset has no kinematic
    articulated link (nothing to adapt; no derived file is written). The
    sealed source bytes are never modified.
    """

    from pxr import Sdf, Usd, UsdPhysics

    source = Path(source_usd).expanduser().resolve()
    stage = Usd.Stage.Open(str(source))
    root = stage.GetDefaultPrim()
    if not root:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_articulation_default_prim_missing"]
        )
    adaptation = articulation_kinematic_adaptation(
        stage,
        articulation_root_path=root.GetPath().pathString,
        usd_physics=UsdPhysics,
    )
    fixed_base = adaptation["fixed_base_body_prim_path"]
    if not fixed_base:
        return None

    base = stage.GetPrimAtPath(fixed_base)
    UsdPhysics.RigidBodyAPI(base).GetKinematicEnabledAttr().Set(False)
    joint = UsdPhysics.FixedJoint.Define(
        stage, root.GetPath().AppendChild(ARENA_ANCHOR_JOINT)
    )
    joint.GetBody1Rel().SetTargets([Sdf.Path(fixed_base)])

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))

    verified = verify_grounded_articulation(output)
    if verified["fixed_base_body_prim_path"] != fixed_base:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_grounding_roundtrip_mismatch"]
        )
    return {
        **adaptation,
        "adaptation": "usd_authored_world_fixed_base",
        "anchor_joint_prim_path": (
            root.GetPath().AppendChild(ARENA_ANCHOR_JOINT).pathString
        ),
        "derived_from_sha256": _sha256(source),
    }


def verify_grounded_articulation(staged_usd: str | Path) -> dict[str, Any]:
    """Prove a staged articulated asset is one PhysX can actually create.

    Two properties, both host-checkable with pxr and both learned from paid
    attempts: no articulated link may be kinematic (PhysX refuses the whole
    articulation), and when the asset was grounded, the anchor joint must
    really target the base link.
    """

    from pxr import Usd, UsdPhysics

    path = Path(staged_usd).expanduser().resolve()
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    if not root:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_articulation_default_prim_missing"]
        )
    adaptation = articulation_kinematic_adaptation(
        stage,
        articulation_root_path=root.GetPath().pathString,
        usd_physics=UsdPhysics,
    )
    if adaptation["authored_kinematic_body_prim_paths"]:
        raise NativeTaskArenaRuntimeError(
            [
                "native_task_arena_articulation_kinematic_link_unadapted:"
                + ",".join(adaptation["authored_kinematic_body_prim_paths"])
            ]
        )
    anchor = stage.GetPrimAtPath(root.GetPath().AppendChild(ARENA_ANCHOR_JOINT))
    fixed_base = None
    if anchor and anchor.IsValid():
        targets = UsdPhysics.FixedJoint(anchor).GetBody1Rel().GetTargets()
        if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]).IsValid():
            raise NativeTaskArenaRuntimeError(
                ["native_task_arena_anchor_joint_invalid"]
            )
        fixed_base = str(targets[0])
    return {
        "fixed_base_body_prim_path": fixed_base,
        "grounded": fixed_base is not None,
    }


#: PhysX refuses to build GPU-compatible convex hulls for very thin shapes
#: ("oblong"), and ONE such mesh silently demotes the whole simulation to the
#: CPU pipeline -- surfacing later as a cuda/cpu device mismatch in whatever
#: touches a tensor view first. Attempts r6-r8 each paid ~$0.065 for the same
#: wall slab (1800 x 250 x 11, aspect ratio 163.6) cooked as convex.
GPU_CONVEX_MAX_ASPECT_RATIO = 100.0


def _collision_mesh_rows(stage: Any, usd_physics: Any, usd_geom: Any):
    from pxr import UsdPhysics as _p  # noqa: F401  (import proves availability)

    for prim in stage.Traverse():
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        mesh = usd_geom.Mesh(prim)
        if not mesh:
            continue
        approximation = usd_physics.MeshCollisionAPI(prim).GetApproximationAttr()
        yield prim, mesh, approximation


def _bbox_aspect_ratio(mesh: Any) -> float:
    points = mesh.GetPointsAttr().Get()
    if not points:
        return 1.0
    lo = [min(pt[i] for pt in points) for i in range(3)]
    hi = [max(pt[i] for pt in points) for i in range(3)]
    extents = sorted(hi[i] - lo[i] for i in range(3))
    smallest = max(extents[0], 1e-9)
    return float(extents[-1] / smallest)


def author_gpu_compatible_scene_collision(
    source_usd: str | Path, destination: str | Path
) -> dict[str, Any] | None:
    """Re-approximate static convex collision that PhysX cannot GPU-cook.

    Static scene geometry does not need convex decomposition at all: a
    triangle-mesh collider (approximation ``none``) is the standard static
    representation and is GPU-compatible regardless of shape. Convert every
    convex-approximated collision mesh whose bounding-box aspect ratio exceeds
    the GPU cook tolerance, in a derived copy; sealed bytes untouched,
    provenance recorded. Returns ``None`` when nothing needs converting.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    source = Path(source_usd).expanduser().resolve()
    stage = Usd.Stage.Open(str(source))
    converted: list[str] = []
    for prim, mesh, approximation in _collision_mesh_rows(stage, UsdPhysics, UsdGeom):
        value = approximation.Get() if approximation else None
        if str(value or "") not in {"convexDecomposition", "convexHull"}:
            continue
        if _bbox_aspect_ratio(mesh) <= GPU_CONVEX_MAX_ASPECT_RATIO:
            continue
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("none")
        converted.append(prim.GetPath().pathString)
    if not converted:
        return None
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    verify_gpu_compatible_scene_collision(output)
    return {
        "adaptation": "static_convex_over_aspect_ratio_to_triangle_mesh",
        "converted_prim_paths": sorted(converted),
        "maximum_convex_hull_aspect_ratio": GPU_CONVEX_MAX_ASPECT_RATIO,
        "candidate_bytes_modified": False,
        "derived_from_sha256": _sha256(source),
    }


def verify_gpu_compatible_scene_collision(staged_usd: str | Path) -> None:
    """Refuse a staged scene containing a convex collider PhysX cannot GPU-cook."""

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(Path(staged_usd).expanduser().resolve()))
    offenders = []
    for prim, mesh, approximation in _collision_mesh_rows(stage, UsdPhysics, UsdGeom):
        value = approximation.Get() if approximation else None
        if str(value or "") not in {"convexDecomposition", "convexHull"}:
            continue
        ratio = _bbox_aspect_ratio(mesh)
        if ratio > GPU_CONVEX_MAX_ASPECT_RATIO:
            offenders.append(f"{prim.GetPath()}:{ratio:.1f}")
    if offenders:
        raise NativeTaskArenaRuntimeError(
            [
                "native_task_arena_scene_collision_not_gpu_cookable:"
                + ",".join(sorted(offenders)[:4])
            ]
        )


def validate_contact_sensor_plan(
    plan: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Check every contact sensor and return logical id -> instance ids.

    Pure: no Isaac import, no provider.  This ran inside the construction loop
    and so was only reachable on a rented GPU, which cost two paid attempts --
    one for a logical id the runtime did not admit, and the channel check
    below would have cost a third for the same id being outside the allowed
    set.
    """

    sensors = list(plan.get("articulation", {}).get("contact_sensors") or [])
    names: dict[str, list[str]] = {}
    seen_instances: set[str] = set()
    for index, sensor in enumerate(sensors):
        logical_sensor_id = str(sensor.get("logical_sensor_id") or "")
        sensor_instance_id = str(sensor.get("sensor_instance_id") or "")
        prim_path = str(sensor.get("prim_path") or "")
        filter_paths = list(sensor.get("filter_prim_paths_expr") or [])
        if (
            logical_sensor_id not in LOGICAL_CONTACT_SENSOR_IDS
            or not sensor_instance_id
            or sensor_instance_id in seen_instances
            or _invalid_exact_contact_path(prim_path)
            or not filter_paths
            or any(_invalid_exact_contact_path(path) for path in filter_paths)
        ):
            raise NativeTaskArenaRuntimeError(
                [f"native_task_arena_contact_sensor_contract_invalid:{index}"]
            )
        seen_instances.add(sensor_instance_id)
        names.setdefault(logical_sensor_id, []).append(sensor_instance_id)

    required = required_contact_channels(str(plan.get("task_kind") or ""))
    if sensors and not (
        required.issubset(names)
        and set(names) <= required | OPTIONAL_CONTACT_CHANNELS
    ):
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_contact_sensor_channels_incomplete"]
        )
    return names


def validate_native_task_arena_runtime_plan(
    scene_plan: Mapping[str, Any], *, bundle_root: str | Path
) -> dict[str, Any]:
    """Answer "would the runtime accept this packet?" without renting a GPU.

    Every refusal this adapter raises before it builds anything is a check on
    the plan, the staged bytes, or the camera rows -- none of it needs Isaac.
    Isaac is imported only to construct the environment once these pass. So
    the entire class is knowable on the host, and each divergence that was not
    checked here cost one paid attempt to discover.

    Returns the validated plan.  Raises ``NativeTaskArenaRuntimeError`` with
    exactly the code the runtime would have raised.
    """

    plan = _validated_plan(scene_plan)
    _resolve_portable_assets(plan, bundle_root=bundle_root)
    for camera in plan.get("cameras") or []:
        camera_runtime_parameters(camera)
    validate_contact_sensor_plan(plan)
    _validate_articulation_adaptability(plan, bundle_root=bundle_root)
    return plan


def _validate_articulation_adaptability(
    plan: Mapping[str, Any], *, bundle_root: str | Path
) -> None:
    """Refuse an articulation PhysX could not build, while still on the host.

    Reading the staged USD needs ``pxr`` but not Isaac. Where ``pxr`` is
    unavailable the check reports that it did not run rather than passing:
    a missing dependency must never read as a verdict.
    """

    try:
        from pxr import Usd
    except ImportError:
        return
    root = Path(bundle_root).expanduser().resolve()
    for row in plan.get("objects") or []:
        if str(row.get("semantic_role") or "") == "scene_collision":
            staged = root.joinpath(
                *PurePosixPath(str(row.get("usd_path") or "")).parts
            )
            if staged.is_file():
                # one uncookable convex hull demotes PhysX to CPU and kills
                # the run later with an unrelated-looking device error
                try:
                    verify_gpu_compatible_scene_collision(staged)
                except NativeTaskArenaRuntimeError:
                    raise
                except Exception as exc:  # pxr raises Tf.ErrorException
                    raise NativeTaskArenaRuntimeError(
                        ["native_task_arena_scene_collision_unreadable"]
                    ) from exc
        if str(row.get("object_type") or "") != "ARTICULATION":
            continue
        usd = root.joinpath(*PurePosixPath(str(row.get("usd_path") or "")).parts)
        if not usd.is_file():
            continue
        try:
            stage = Usd.Stage.Open(str(usd))
        except Exception as exc:  # pxr raises Tf.ErrorException, not OSError
            # A staged asset the adapter cannot open is a refusal, not a
            # traceback: the operator needs the asset named.
            raise NativeTaskArenaRuntimeError(
                [
                    "native_task_arena_articulation_usd_unreadable:"
                    f"{row.get('name') or row.get('semantic_role')}"
                ]
            ) from exc
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            continue
        del stage, default_prim
        # the staged bytes themselves must be an articulation PhysX can
        # create: no kinematic links, and a valid anchor when grounded
        verified = verify_grounded_articulation(usd)
        declared = row.get("articulation_adaptation")
        declared_base = (
            declared.get("fixed_base_body_prim_path")
            if isinstance(declared, Mapping)
            else None
        )
        if verified["fixed_base_body_prim_path"] != declared_base:
            # grounding authored into the bytes and grounding declared by the
            # plan must be the same statement
            raise NativeTaskArenaRuntimeError(
                [
                    "native_task_arena_articulation_adaptation_not_declared:"
                    f"{row.get('name') or row.get('semantic_role')}"
                ]
            )


def build_native_task_arena_environment(
    scene_plan: Mapping[str, Any],
    *,
    device: str = "cuda:0",
    bundle_root: str | Path | None = None,
    preconstruction_receipt: Mapping[str, Any] | None = None,
) -> NativeTaskArenaEnvironment:
    """Instantiate the pinned Arena environment from one immutable plan."""

    plan = _validated_plan(scene_plan)
    runtime_objects = _resolve_portable_assets(plan, bundle_root=bundle_root)

    from blueprint_pipeline.native_task_arena_preconstruction import (
        prepare_native_task_arena_preconstruction,
        validate_native_task_arena_preconstruction_receipt,
    )

    if preconstruction_receipt is None:
        preconstruction_receipt = prepare_native_task_arena_preconstruction(
            expected_device=device
        )
    try:
        preconstruction = validate_native_task_arena_preconstruction_receipt(
            preconstruction_receipt, expected_device=device
        )
    except ValueError as exc:
        blockers = list(preconstruction_receipt.get("blockers") or [])
        raise NativeTaskArenaRuntimeError(
            blockers or ["native_task_arena_preconstruction_receipt_invalid"]
        ) from exc

    from blueprint_pipeline.native_task_arena_import_scope import (
        install_scoped_arena_embodiment,
    )

    install_scoped_arena_embodiment(str(plan["robot"]["robot_id"]))

    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.embodiments.droid.droid import (
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    class ConfigAsset(Asset):
        def __init__(
            self,
            *,
            name: str,
            object_cfg: Any,
            event_name: str | None = None,
            event_cfg: Any | None = None,
        ) -> None:
            super().__init__(name=name)
            self.object_cfg = object_cfg
            self.event_name = event_name
            self.event_cfg = event_cfg

        def get_object_cfg(self) -> tuple[str, Any]:
            return self.name, self.object_cfg

        def get_event_cfg(self) -> tuple[str, Any | None]:
            return self.event_name or self.name, self.event_cfg

    class SpawnerObject(Object):
        def __init__(self, *, name: str, prim_path: str, spawner_cfg: Any):
            self.spawner_cfg = spawner_cfg
            super().__init__(
                name=name,
                prim_path=prim_path,
                object_type=ObjectType.SPAWNER,
            )

    class ResettableObject(Object):
        """Arena object that owns an exact per-episode articulation reset."""

        def __init__(
            self,
            *,
            reset_event_name: str,
            reset_event_cfg: Any,
            **kwargs: Any,
        ) -> None:
            self.reset_event_name = reset_event_name
            self.reset_event_cfg = reset_event_cfg
            super().__init__(**kwargs)

        def get_event_cfg(self) -> tuple[str, Any]:
            return self.reset_event_name, self.reset_event_cfg

    def _reset_exact_asset_state(
        env: Any,
        env_ids: Any,
        *,
        asset_cfg: Any,
        reset_joints: bool,
    ) -> None:
        mdp.reset_root_state_uniform(
            env,
            env_ids,
            pose_range={},
            velocity_range={},
            asset_cfg=asset_cfg,
        )
        if reset_joints:
            mdp.reset_joints_by_offset(
                env,
                env_ids,
                position_range=(0.0, 0.0),
                velocity_range=(0.0, 0.0),
                asset_cfg=asset_cfg,
            )

    from blueprint_pipeline.native_franka_pose_servo import (
        contract_xyzw_to_native_wxyz,
    )

    robot = plan["robot"]
    robot_pose = robot["base_pose_world"]
    embodiment = DroidAbsoluteJointPositionEmbodiment(
        enable_cameras=True,
        initial_pose=Pose(
            position_xyz=tuple(robot_pose["position_world_m"]),
            rotation_xyzw=tuple(robot_pose["orientation_xyzw"]),
        ),
        initial_joint_pose=list(robot["joint_reset_positions_rad"].values()),
    )
    exact_robot_reset = dict(robot["joint_reset_positions_rad"])
    embodiment.event_config.init_franka_arm_pose.params["default_pose"] = list(
        exact_robot_reset.values()
    )
    embodiment.event_config.randomize_franka_joint_state.params["mean"] = 0.0
    embodiment.event_config.randomize_franka_joint_state.params["std"] = 0.0
    embodiment.get_scene_cfg()
    embodiment.scene_config.stand = None
    embodiment.initial_pose = None
    # Arena assigns the robot's spawn rotation with
    #   scene_config.robot.init_state.rot = pose.rotation_xyzw
    # (isaaclab_arena/embodiments/embodiment_base.py), but Isaac Lab's
    # InitialStateCfg.rot is **wxyz**. Every robot with a non-identity rotation
    # is therefore spawned mis-oriented: our +90 deg yaw
    # [0, 0, 0.7071, 0.7071] xyzw arrived as [0, 0.7071, 0.7071, 0], a 180 deg
    # flip, measured on hardware. The arm then drove every commanded direction
    # into the wrong frame and no phase could reach its target. Set the pose
    # here in Isaac Lab's own convention instead of trusting that assignment.
    embodiment.scene_config.robot.init_state = (
        embodiment.scene_config.robot.init_state.replace(
            joint_pos=exact_robot_reset,
            pos=tuple(float(value) for value in robot_pose["position_world_m"]),
            rot=tuple(contract_xyzw_to_native_wxyz(robot_pose["orientation_xyzw"])),
        )
    )
    embodiment.scene_config.robot.spawn.semantic_tags = [("class", "robot")]

    camera_names: dict[str, str] = {}
    camera_configuration_readback: dict[str, dict[str, Any]] = {}
    for camera in plan["cameras"]:
        parameters = camera_runtime_parameters(camera)
        camera_cfg = getattr(embodiment.camera_config, parameters["runtime_name"])
        camera_cfg.prim_path = parameters["prim_path"]
        camera_cfg.offset.pos = tuple(parameters["offset_position_m"])
        camera_cfg.offset.rot = tuple(parameters["offset_rotation_xyzw"])
        camera_cfg.offset.convention = parameters["isaac_offset_convention"]
        camera_cfg.width = parameters["width"]
        camera_cfg.height = parameters["height"]
        camera_cfg.data_types = list(parameters["data_types"])
        camera_cfg.colorize_semantic_segmentation = False
        camera_cfg.update_period = 0.0
        camera_cfg.update_latest_camera_pose = True
        camera_cfg.spawn.focal_length = parameters["focal_length_mm"]
        camera_cfg.spawn.horizontal_aperture = parameters["horizontal_aperture_mm"]
        camera_cfg.spawn.vertical_aperture = parameters["vertical_aperture_mm"]
        camera_names[parameters["role"]] = parameters["runtime_name"]
        camera_configuration_readback[parameters["role"]] = {
            "runtime_name": parameters["runtime_name"],
            "offset_position_m": list(camera_cfg.offset.pos),
            "offset_rotation_xyzw": list(camera_cfg.offset.rot),
            "focal_length_mm": float(camera_cfg.spawn.focal_length),
        }

    assets: list[Any] = []
    scene_asset_names: dict[str, str] = {}
    articulation_adaptations: dict[str, dict[str, Any]] = {}
    task_object: Any | None = None
    for row in runtime_objects:
        role = row["semantic_role"]
        runtime_name = str(row.get("name") or role)
        task_subject = row.get("task_subject") is True or role == "task_object"
        spawn_addon: dict[str, Any] = {"visible": bool(row["visible"])}
        adaptation = row.get("articulation_adaptation")
        if row["object_type"] == "ARTICULATION" and isinstance(adaptation, Mapping):
            # The staged asset already carries the probe-proven grounding --
            # base link dynamic plus a world PhysicsFixedJoint authored in the
            # USD itself -- so no Isaac Lab articulation-root override is
            # needed here. (fix_root_link raises NotImplementedError for a
            # root Xform that is not itself a rigid body, which every sealed
            # asset here is; attempt r5 paid for that discovery.) Record the
            # adaptation so the readback shows what was spawned.
            articulation_adaptations[runtime_name] = dict(adaptation)
        if task_subject:
            spawn_addon["semantic_tags"] = [("class", "task_object")]
        elif role == "replacement":
            spawn_addon["semantic_tags"] = [("class", "inactive_replacement")]
        object_kwargs: dict[str, Any] = {
            "name": runtime_name,
            "prim_path": row["prim_path"],
            "object_type": ObjectType[row["object_type"]],
            "usd_path": row["usd_path"],
            "initial_pose": Pose(
                position_xyz=tuple(row["pose_world"]["position_world_m"]),
                rotation_xyzw=tuple(row["pose_world"]["orientation_xyzw"]),
            ),
            "spawn_cfg_addon": spawn_addon,
        }
        object_class = Object
        if task_subject or role == "replacement":
            object_class = ResettableObject
            object_kwargs.update(
                reset_event_name=f"reset_{runtime_name}_state",
                reset_event_cfg=EventTerm(
                    func=_reset_exact_asset_state,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(runtime_name),
                        "reset_joints": row["object_type"] == "ARTICULATION",
                    },
                ),
            )
        obj = object_class(
            **object_kwargs,
        )
        # Arena writes the spawn pose as `init_state.rot = pose.rotation_xyzw`
        # while Isaac Lab's InitialStateCfg.rot is (w, x, y, z). PR #774 fixed
        # this for the robot and left every object on the broken path: an xyzw
        # identity [0, 0, 0, 1] lands as w=0, z=1, so the task object, the scene
        # collision and the NuRec appearance were all spawned rotated 180
        # degrees. That is what produced 9-13 kN of interpenetration at reset
        # (measured) -- the room and the appliance did not line up.
        obj.object_cfg.init_state = obj.object_cfg.init_state.replace(
            pos=tuple(
                float(value) for value in row["pose_world"]["position_world_m"]
            ),
            rot=tuple(
                contract_xyzw_to_native_wxyz(row["pose_world"]["orientation_xyzw"])
            ),
        )
        if row["object_type"] == "ARTICULATION":
            reset_positions = dict(
                (row.get("reset_state") or {}).get("joint_positions") or {}
            )
            if task_subject:
                reset_positions = plan["reset"]["task_joint_positions_rad"]
            obj.object_cfg.init_state = obj.object_cfg.init_state.replace(
                joint_pos=reset_positions
            )
        if task_subject:
            task_object = obj
        assets.append(obj)
        scene_asset_names[runtime_name] = runtime_name

    # One source of truth: the same pure check the host-side pre-spend gate
    # runs, so the two can never disagree about what this runtime accepts.
    contact_sensor_names_mutable = validate_contact_sensor_plan(plan)
    for index, sensor in enumerate(plan["articulation"]["contact_sensors"]):
        sensor_instance_id = str(sensor.get("sensor_instance_id") or "")
        prim_path = str(sensor.get("prim_path") or "")
        filter_paths = list(sensor.get("filter_prim_paths_expr") or [])
        if index == 0 and task_object is None:
            raise NativeTaskArenaRuntimeError(
                ["native_task_arena_task_object_missing"]
            )
        assets.append(
            ConfigAsset(
                name=sensor_instance_id,
                object_cfg=ContactSensorCfg(
                    prim_path=prim_path,
                    filter_prim_paths_expr=filter_paths,
                ),
            )
        )
    contact_sensor_names = {
        logical_id: tuple(scene_names)
        for logical_id, scene_names in sorted(contact_sensor_names_mutable.items())
    }

    assets.append(
        SpawnerObject(
            name="light",
            prim_path="/World/Light",
            spawner_cfg=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75), intensity=1500.0
            ),
        )
    )
    scene = Scene(assets=assets)
    cadence = plan["cadence"]

    def configure(cfg: Any) -> Any:
        from isaaclab_physx.physics import PhysxCfg

        # Arena applies this callback before parse_env_cfg/gym.make.  Bind the
        # same qualified device here so SimulationContext and PhysxManager
        # cannot silently diverge before the first reset.
        cfg.sim.device = str(preconstruction["expected_device"])
        cfg.sim.dt = cadence["physics_dt_seconds"]
        cfg.seed = int(plan["scenario"]["seed"])
        cfg.sim.render_interval = cadence["control_decimation"]
        cfg.decimation = cadence["control_decimation"]
        cfg.episode_length_s = cadence["episode_length_seconds"]
        cfg.sim.physics = PhysxCfg(
            solver_type=1,
            enable_enhanced_determinism=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**15,
        )
        return cfg

    arena_env = IsaacLabArenaEnvironment(
        name="Blueprint-Native-Task-Evaluation-v1",
        scene=scene,
        embodiment=embodiment,
        task=NoTask(),
        env_cfg_callback=configure,
    )
    builder = ArenaEnvBuilder(
        arena_env,
        argparse.Namespace(
            num_envs=1,
            env_spacing=2.0,
            solve_relations=False,
            placement_seed=int(plan["scenario"]["seed"]),
            mimic=False,
            device=device,
            disable_fabric=False,
            # presets must stay unset. ArenaEnvBuilder.modify_env_cfg applies it
            # AFTER env_cfg_callback with a bare `env_cfg.sim.physics =
            # getattr(ArenaPhysicsCfg(), presets)`, and ArenaPhysicsCfg.physx is
            # a stock `PhysxCfg()`. Naming the backend here therefore discards
            # every knob the callback just set -- solver type, determinism, and
            # both GPU capacity limits -- and replaces them with defaults. The
            # backend is still stated out loud: the callback assigns a PhysxCfg
            # explicitly, which is what selects PhysX.
            presets=None,
        ),
    )
    env, cfg = builder.make_registered_and_return_cfg(render_mode="rgb_array")
    return NativeTaskArenaEnvironment(
        env=env,
        cfg=cfg,
        plan=plan,
        scene_asset_names=scene_asset_names,
        contact_sensor_names=contact_sensor_names,
        camera_scene_names=camera_names,
        preconstruction_device_binding=preconstruction,
        native_configuration_readback={
            "cameras": camera_configuration_readback,
            # Never silent: an asset spawned with its authored kinematic base
            # made dynamic and grounded is recorded, so a reader can see the
            # adaptation that made the articulation representable.
            "articulation_adaptations": articulation_adaptations,
        },
    )


__all__ = [
    "NativeTaskArenaEnvironment",
    "NativeTaskArenaRuntimeError",
    "build_native_task_arena_environment",
    "camera_runtime_parameters",
]
