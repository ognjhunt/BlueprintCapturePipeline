from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from blueprint_pipeline.native_task_arena_device_readback import (
    read_native_task_arena_device_binding,
)
from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaEnvironment


class _ArticulationView:
    def __init__(self, device: str):
        self.device = device

    def get_dof_velocities(self):
        return SimpleNamespace(device=self.device)


class _RigidBodyView:
    def __init__(self, device: str):
        self.device = device

    def get_velocities(self):
        return SimpleNamespace(device=self.device)


def _built(*, task_view_device: str = "cuda:0") -> NativeTaskArenaEnvironment:
    def asset(view):
        return SimpleNamespace(
            device="cuda:0",
            data=SimpleNamespace(device="cuda:0"),
            root_physx_view=view,
        )

    unwrapped = SimpleNamespace(
        device="cuda:0",
        scene={
            "robot": asset(_ArticulationView("cuda:0")),
            "task_object": asset(_RigidBodyView(task_view_device)),
        },
    )
    return NativeTaskArenaEnvironment(
        env=SimpleNamespace(unwrapped=unwrapped),
        cfg=SimpleNamespace(sim=SimpleNamespace(device="cuda:0")),
        plan={},
        scene_asset_names={"task_object": "task_object"},
        contact_sensor_names={},
        camera_scene_names={},
    )


def _physics_manager(monkeypatch, device: str = "cuda:0") -> None:
    package = types.ModuleType("isaaclab_physx")
    physics = types.ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = type(
        "PhysxManager", (), {"get_device": classmethod(lambda cls: device)}
    )
    monkeypatch.setitem(sys.modules, "isaaclab_physx", package)
    monkeypatch.setitem(sys.modules, "isaaclab_physx.physics", physics)


def test_all_native_tensor_devices_must_match_requested_cuda(monkeypatch) -> None:
    _physics_manager(monkeypatch)

    receipt = read_native_task_arena_device_binding(
        _built(), expected_device="cuda:0"
    )

    assert receipt["passed"] is True
    assert set(receipt["observed_devices"].values()) == {"cuda:0"}
    assert receipt["probe_methods"] == {
        "robot_physx_joint_velocity": "get_dof_velocities",
        "task_physx_joint_velocity": "get_velocities",
    }


def test_cpu_physx_array_in_cuda_environment_is_a_typed_blocker(monkeypatch) -> None:
    _physics_manager(monkeypatch)

    receipt = read_native_task_arena_device_binding(
        _built(task_view_device="cpu"), expected_device="cuda:0"
    )

    assert receipt["passed"] is False
    assert receipt["blockers"] == [
        "native_task_arena_device_mismatch:task_physx_joint_velocity"
    ]


def test_a_navigation_scene_probes_only_the_robot_and_still_fails_on_a_wrong_device(monkeypatch) -> None:
    """A scene kept as filmed has no task object: its probes are the robot's alone."""
    _physics_manager(monkeypatch)
    built = _built()
    del built.env.unwrapped.scene["task_object"]
    navigation = NativeTaskArenaEnvironment(
        env=built.env, cfg=built.cfg, plan={"task_kind": "goal_region_navigation"},
        scene_asset_names={"scene_collision": "scene_collision"}, contact_sensor_names={},
        camera_scene_names={},
    )
    receipt = read_native_task_arena_device_binding(navigation, expected_device="cuda:0")
    assert receipt["passed"] is True and receipt["all_required_read_back"] is True
    assert "task_asset" not in receipt["observed_devices"]
    assert "task_physx_joint_velocity" not in receipt["observed_devices"]
    assert receipt["probe_methods"] == {"robot_physx_joint_velocity": "get_dof_velocities"}
    built.env.unwrapped.scene["robot"].data.device = "cpu"
    assert read_native_task_arena_device_binding(navigation, expected_device="cuda:0")["blockers"] == [
        "native_task_arena_device_mismatch:robot_data"]


def test_a_rigid_scene_without_a_task_object_still_fails(monkeypatch) -> None:
    import pytest

    _physics_manager(monkeypatch)
    built = _built()
    rigid = NativeTaskArenaEnvironment(env=built.env, cfg=built.cfg, plan={"task_kind": "rigid_pick_place"},
                                       scene_asset_names={}, contact_sensor_names={}, camera_scene_names={})
    with pytest.raises(KeyError):
        read_native_task_arena_device_binding(rigid, expected_device="cuda:0")
