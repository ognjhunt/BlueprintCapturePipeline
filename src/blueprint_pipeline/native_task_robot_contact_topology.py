"""Exact robot-body topology used to author filtered native contact sensors.

Isaac Lab's filtered contact view supports one sensor body against many exact
filter bodies.  It does not support a many-body sensor expression against a
many-body filter expression.  These profiles therefore bind each admitted
embodiment to the exact rigid-body paths of its released runtime asset.  Scene
and task compilers consume the profile; no scene id or task object appears
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "native_task_robot_contact_topology.v1"
ENV_ROOT = "{ENV_REGEX_NS}"


class NativeTaskRobotContactTopologyError(ValueError):
    """Stable admission failures for an unknown or malformed robot profile."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


_DROID_ASSET_ROOT = f"{ENV_ROOT}/Robot"
_DROID_GRIPPER_ROOT = f"{_DROID_ASSET_ROOT}/Gripper/Robotiq_2F_85"
_DROID_ARM_BODIES = tuple(
    f"{_DROID_ASSET_ROOT}/panda_link{index}" for index in range(9)
)
_DROID_GRIPPER_BODIES = tuple(
    f"{_DROID_GRIPPER_ROOT}/{name}"
    for name in (
        "base_link",
        "left_outer_knuckle",
        "left_outer_finger",
        "left_inner_finger",
        "left_inner_knuckle",
        "right_outer_knuckle",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_knuckle",
    )
)


_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "franka_panda": {
        "schema_version": SCHEMA_VERSION,
        "robot_id": "franka_panda",
        "runtime_asset": {
            "source": (
                "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/6.0/Isaac/IsaacLab/Arena/assets/robot_library/"
                "droid/franka_robotiq_2f_85_flattened.usd"
            ),
            "sha256": (
                "sha256:c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c"
            ),
            "content_length_bytes": 14161197,
            "last_modified": "Mon, 10 Aug 2026 07:09:58 GMT",
            "source_default_prim": "/panda",
            "spawn_prim_path": _DROID_ASSET_ROOT,
            "identity_observed_not_runtime_bytes_reverified": True,
        },
        "task_contact_body_paths": (
            f"{_DROID_GRIPPER_ROOT}/left_inner_finger",
            f"{_DROID_GRIPPER_ROOT}/right_inner_finger",
        ),
        "protected_collision_body_paths": (
            *_DROID_ARM_BODIES,
            *_DROID_GRIPPER_BODIES,
        ),
    }
}


def _asset_rigid_body_paths(asset_path: Path) -> list[str]:
    """Read exact contact-sensor body paths from the sealed robot USD."""

    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(asset_path))
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise ValueError("default_prim_missing")
        source_root = str(stage.GetDefaultPrim().GetPath())
        bodies = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
    except Exception as exc:  # USD may raise pxr.Tf.ErrorException on bad asset bytes.
        raise NativeTaskRobotContactTopologyError(
            ["native_task_robot_contact_asset_unreadable"]
        ) from exc
    prefix = f"{ENV_ROOT}/Robot"
    return sorted(
        {
            prefix + path[len(source_root) :]
            for path in bodies
            if path == source_root or path.startswith(source_root + "/")
        }
    )


def resolve_native_task_robot_contact_topology(
    robot_id: str, robot: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a detached, validated exact-body profile for one embodiment."""

    profile = _PROFILES.get(str(robot_id))
    if profile is None:
        if robot is None or robot.get("robot_id") != robot_id:
            raise NativeTaskRobotContactTopologyError(
                [f"native_task_robot_contact_topology_unavailable:{robot_id}"]
            )
        from .native_task_robot_registry import validate_native_robot_plan

        try:
            validate_native_robot_plan(robot)
            asset_path = Path(str(robot["usd_path"]))
            requested = robot["task_contact_body_paths"]
            if not isinstance(requested, (list, tuple)):
                raise ValueError("task_contact_body_paths")
            protected = _asset_rigid_body_paths(asset_path)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            if isinstance(exc, NativeTaskRobotContactTopologyError):
                raise
            raise NativeTaskRobotContactTopologyError(
                [f"native_task_robot_contact_topology_unavailable:{robot_id}"]
            ) from exc
        profile = {
            "schema_version": SCHEMA_VERSION,
            "robot_id": robot_id,
            "runtime_asset": {
                "source": str(asset_path),
                "sha256": robot["usd_sha256"],
                "spawn_prim_path": f"{ENV_ROOT}/Robot",
            },
            "task_contact_body_paths": requested,
            "protected_collision_body_paths": protected,
        }
    value = json.loads(json.dumps(profile))
    task_bodies = list(value["task_contact_body_paths"])
    protected_bodies = list(value["protected_collision_body_paths"])
    errors: list[str] = []
    for field, paths in (
        ("task_contact_body_paths", task_bodies),
        ("protected_collision_body_paths", protected_bodies),
    ):
        if not paths or len(paths) != len(set(paths)):
            errors.append(f"native_task_robot_contact_topology_invalid:{field}")
        for path in paths:
            if not isinstance(path, str) or not path.startswith(f"{ENV_ROOT}/Robot/") or any(
                token in path for token in ("*", ".*", "[", "]")
            ):
                errors.append(f"native_task_robot_contact_body_not_exact:{field}")
    if not set(task_bodies).issubset(protected_bodies):
        errors.append("native_task_robot_contact_topology_task_bodies_unprotected")
    if not set(protected_bodies) - set(task_bodies):
        errors.append("native_task_robot_contact_topology_forbidden_bodies_missing")
    if errors:
        raise NativeTaskRobotContactTopologyError(errors)
    return value


__all__ = [
    "NativeTaskRobotContactTopologyError",
    "SCHEMA_VERSION",
    "resolve_native_task_robot_contact_topology",
]
