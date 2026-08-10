from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from blueprint_pipeline.native_task_arena_device_readback import (
    read_native_task_arena_device_binding,
)
from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaEnvironment


class _View:
    def __init__(self, device: str):
        self.device = device

    def get_dof_velocities(self):
        return SimpleNamespace(device=self.device)


def _built(*, task_view_device: str = "cuda:0") -> NativeTaskArenaEnvironment:
    def asset(view_device: str):
        return SimpleNamespace(
            device="cuda:0",
            data=SimpleNamespace(device="cuda:0"),
            root_physx_view=_View(view_device),
        )

    unwrapped = SimpleNamespace(
        device="cuda:0",
        scene={"robot": asset("cuda:0"), "task_object": asset(task_view_device)},
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


def test_cpu_physx_array_in_cuda_environment_is_a_typed_blocker(monkeypatch) -> None:
    _physics_manager(monkeypatch)

    receipt = read_native_task_arena_device_binding(
        _built(task_view_device="cpu"), expected_device="cuda:0"
    )

    assert receipt["passed"] is False
    assert receipt["blockers"] == [
        "native_task_arena_device_mismatch:task_physx_joint_velocity"
    ]
