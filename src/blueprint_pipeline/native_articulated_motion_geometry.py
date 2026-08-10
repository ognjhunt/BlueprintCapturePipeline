"""Derive a hinged task's motion geometry from its exact OpenUSD bytes.

The scorer knows which joint matters and the task-state binding knows which
moving body and handle point are observed.  The remaining facts needed by an
arm trajectory -- hinge origin, hinge axis, authored limits, and the handle's
closed world position -- must come from the joint actually shipped to Isaac.
Keeping that derivation here prevents a refrigerator coordinate (or a guess
about a world-vertical hinge) from entering the reusable runtime contract.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_articulated_motion_geometry.v1"


class NativeArticulatedMotionGeometryError(ValueError):
    """Stable USD/geometry failures raised before a native launch."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _vector(value: Any, *, length: int, error: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeArticulatedMotionGeometryError([error]) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise NativeArticulatedMotionGeometryError([error])
    return result


def _normalize(value: Sequence[float], *, error: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise NativeArticulatedMotionGeometryError([error])
    return [float(item) / norm for item in value]


def _quaternion_rotate_xyzw(
    quaternion: Sequence[float], vector: Sequence[float]
) -> list[float]:
    x, y, z, w = _vector(
        quaternion, length=4, error="native_articulated_motion_task_pose_invalid"
    )
    vx, vy, vz = _vector(
        vector, length=3, error="native_articulated_motion_vector_invalid"
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > 1.0e-6:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_task_pose_invalid"]
        )
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def derive_native_articulated_motion_geometry(
    *,
    task_object_usd_path: str | Path,
    task_object_sha256: str,
    target_joint_id: str,
    target_joint_prim_path: str,
    moving_link_prim_path: str,
    handle_grasp_point_moving_link_m: Sequence[float],
    task_object_pose_world: Mapping[str, Any],
    reset_angle_rad: float,
    scripted_target_angle_rad: float,
) -> dict[str, Any]:
    """Read the target revolute joint and express its handle arc in world space."""

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - guarded runtime dependency
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_openusd_runtime_missing"]
        ) from exc

    source = Path(task_object_usd_path).expanduser().resolve()
    if not source.is_file():
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_task_object_missing"]
        )
    if _sha256(source) != str(task_object_sha256):
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_task_object_digest_mismatch"]
        )
    try:
        stage = Usd.Stage.Open(str(source))
    except Exception as exc:  # noqa: BLE001 - pxr raises a versioned Tf exception
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_task_object_unreadable"]
        ) from exc
    if stage is None:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_task_object_unreadable"]
        )
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_default_prim_missing"]
        )
    source_root = str(default_prim.GetPath())
    for prim_path, error in (
        (target_joint_prim_path, "native_articulated_motion_target_joint_outside_root"),
        (moving_link_prim_path, "native_articulated_motion_moving_link_outside_root"),
    ):
        if str(prim_path) != source_root and not str(prim_path).startswith(
            source_root + "/"
        ):
            raise NativeArticulatedMotionGeometryError([error])
    joint_prim = stage.GetPrimAtPath(str(target_joint_prim_path))
    if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_target_joint_invalid"]
        )
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    body0_targets = list(joint.GetBody0Rel().GetTargets())
    body1_targets = list(joint.GetBody1Rel().GetTargets())
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_joint_body_binding_invalid"]
        )
    body0 = stage.GetPrimAtPath(body0_targets[0])
    body1 = stage.GetPrimAtPath(body1_targets[0])
    if (
        not body0.IsValid()
        or not body1.IsValid()
        or str(body1.GetPath()) != str(moving_link_prim_path)
    ):
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_moving_link_mismatch"]
        )

    local_position = _vector(
        joint.GetLocalPos0Attr().Get(),
        length=3,
        error="native_articulated_motion_joint_origin_missing",
    )
    axis_token = str(joint.GetAxisAttr().Get() or "")
    axis_basis = {
        "X": [1.0, 0.0, 0.0],
        "Y": [0.0, 1.0, 0.0],
        "Z": [0.0, 0.0, 1.0],
    }.get(axis_token)
    if axis_basis is None:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_joint_axis_invalid"]
        )
    local_rotation = joint.GetLocalRot0Attr().Get()
    if local_rotation is None:
        local_rotation = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    try:
        joint_axis_body = Gf.Rotation(local_rotation).TransformDir(
            Gf.Vec3d(*axis_basis)
        )
    except Exception as exc:  # noqa: BLE001 - pxr type variants are versioned
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_joint_rotation_invalid"]
        ) from exc

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body0_to_asset = cache.GetLocalToWorldTransform(body0)
    body1_to_asset = cache.GetLocalToWorldTransform(body1)
    hinge_asset = body0_to_asset.Transform(Gf.Vec3d(*local_position))
    axis_asset = body0_to_asset.TransformDir(joint_axis_body)
    handle_local = _vector(
        handle_grasp_point_moving_link_m,
        length=3,
        error="native_articulated_motion_handle_point_invalid",
    )
    handle_asset = body1_to_asset.Transform(Gf.Vec3d(*handle_local))

    position_world = _vector(
        task_object_pose_world.get("position_world_m"),
        length=3,
        error="native_articulated_motion_task_pose_invalid",
    )
    orientation_world = _vector(
        task_object_pose_world.get("orientation_xyzw"),
        length=4,
        error="native_articulated_motion_task_pose_invalid",
    )

    def _point_world(point: Any) -> list[float]:
        rotated = _quaternion_rotate_xyzw(
            orientation_world, [float(value) for value in point]
        )
        return [position_world[index] + rotated[index] for index in range(3)]

    hinge_world = _point_world(hinge_asset)
    handle_world = _point_world(handle_asset)
    axis_world = _normalize(
        _quaternion_rotate_xyzw(
            orientation_world, [float(value) for value in axis_asset]
        ),
        error="native_articulated_motion_joint_axis_degenerate",
    )

    try:
        lower_degrees = float(joint.GetLowerLimitAttr().Get())
        upper_degrees = float(joint.GetUpperLimitAttr().Get())
        reset = float(reset_angle_rad)
        target = float(scripted_target_angle_rad)
    except (TypeError, ValueError) as exc:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_joint_limits_invalid"]
        ) from exc
    if not all(
        math.isfinite(value) for value in (lower_degrees, upper_degrees, reset, target)
    ) or lower_degrees > upper_degrees:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_joint_limits_invalid"]
        )
    target_degrees = math.degrees(target)
    if target_degrees < lower_degrees - 1.0e-6 or target_degrees > upper_degrees + 1.0e-6:
        raise NativeArticulatedMotionGeometryError(
            ["native_articulated_motion_scripted_target_outside_limits"]
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target_joint_id": str(target_joint_id),
        "source_asset_root_prim_path": source_root,
        "target_joint_prim_path": str(target_joint_prim_path),
        "body0_prim_path": str(body0.GetPath()),
        "body1_prim_path": str(body1.GetPath()),
        "axis_token": axis_token,
        "hinge_point_task_object_m": [float(value) for value in hinge_asset],
        "hinge_axis_task_object_unit": _normalize(
            [float(value) for value in axis_asset],
            error="native_articulated_motion_joint_axis_degenerate",
        ),
        "handle_grasp_point_moving_link_m": handle_local,
        "handle_grasp_point_task_object_m": [float(value) for value in handle_asset],
        "hinge_point_world_m": hinge_world,
        "hinge_axis_world_unit": axis_world,
        "handle_grasp_point_closed_world_m": handle_world,
        "authored_limits_degrees": [lower_degrees, upper_degrees],
        "reset_angle_rad": reset,
        "scripted_target_angle_rad": target,
        "scripted_sweep_angle_degrees": math.degrees(target - reset),
        "source": {
            "task_object_sha256": str(task_object_sha256),
            "joint_local_position_0_m": local_position,
            "joint_local_rotation_0_real": float(local_rotation.GetReal()),
            "joint_local_rotation_0_imaginary": [
                float(value) for value in local_rotation.GetImaginary()
            ],
            "derived_from_openusd_joint_and_body_transforms": True,
            "caller_authored_world_hinge_forbidden": True,
        },
        "motion_geometry_digest": "",
    }
    result["motion_geometry_digest"] = canonical_digest(
        result, digest_field="motion_geometry_digest"
    )
    return result


__all__ = [
    "NativeArticulatedMotionGeometryError",
    "SCHEMA_VERSION",
    "derive_native_articulated_motion_geometry",
]
