"""Read back native Arena/PhysX tensor devices after environment creation."""

from __future__ import annotations

from typing import Any

from .native_task_arena_runtime import NativeTaskArenaEnvironment


SCHEMA_VERSION = "native_task_arena_device_readback.v1"


def _device(value: Any) -> str | None:
    raw = getattr(value, "device", value)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text == "cuda":
        return "cuda:0"
    return text or None


def read_native_task_arena_device_binding(
    built: NativeTaskArenaEnvironment,
    *,
    expected_device: str,
) -> dict[str, Any]:
    """Require config, environment, assets, and PhysX views on one device."""

    expected = _device(expected_device)
    env = built.env.unwrapped
    scene = env.scene
    robot = scene["robot"]
    task_object = scene[built.scene_asset_names["task_object"]]
    observed: dict[str, str | None] = {
        "requested": expected,
        "config": _device(getattr(built.cfg.sim, "device", None)),
        "environment": _device(getattr(env, "device", None)),
        "robot_asset": _device(getattr(robot, "device", None)),
        "robot_data": _device(getattr(robot.data, "device", None)),
        "task_asset": _device(getattr(task_object, "device", None)),
        "task_data": _device(getattr(task_object.data, "device", None)),
        "physics_manager": None,
        "robot_physx_joint_velocity": None,
        "task_physx_joint_velocity": None,
    }
    probe_errors: list[dict[str, str]] = []
    try:
        from isaaclab_physx.physics import PhysxManager

        observed["physics_manager"] = _device(PhysxManager.get_device())
    except Exception as exc:  # noqa: BLE001 - retained native capability gap
        probe_errors.append(
            {"probe": "physics_manager", "type": type(exc).__name__, "message": str(exc)}
        )
    for key, asset in (
        ("robot_physx_joint_velocity", robot),
        ("task_physx_joint_velocity", task_object),
    ):
        try:
            root_view = getattr(asset, "root_physx_view", None)
            if root_view is None:
                root_view = getattr(asset, "root_view")
            observed[key] = _device(root_view.get_dof_velocities())
        except Exception as exc:  # noqa: BLE001 - retained native capability gap
            probe_errors.append(
                {"probe": key, "type": type(exc).__name__, "message": str(exc)}
            )

    required = tuple(observed)
    mismatches = [
        key
        for key in required
        if key != "requested" and observed.get(key) != expected
    ]
    blockers = [f"native_task_arena_device_mismatch:{key}" for key in mismatches]
    blockers.extend(
        f"native_task_arena_device_probe_failed:{row['probe']}"
        for row in probe_errors
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_device": expected,
        "observed_devices": observed,
        "probe_errors": probe_errors,
        "all_required_read_back": not probe_errors
        and all(observed.get(key) is not None for key in required),
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
    }


__all__ = ["SCHEMA_VERSION", "read_native_task_arena_device_binding"]
