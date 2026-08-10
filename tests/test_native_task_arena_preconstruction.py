from __future__ import annotations

import sys
import types

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_preconstruction import (
    SCHEMA_VERSION,
    UPSTREAM_OWNERSHIP_FIX,
    prepare_native_task_arena_preconstruction,
    validate_native_task_arena_preconstruction_receipt,
)


class _Cuda:
    current = 0

    @classmethod
    def set_device(cls, value: int) -> None:
        cls.current = int(value)

    @classmethod
    def current_device(cls) -> int:
        return cls.current


class _WarpDevice:
    is_cuda = True

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class _OriginalSimulationManager:
    disabled_with = None

    @classmethod
    def enable_all_default_callbacks(cls, enabled: bool) -> None:
        cls.disabled_with = enabled


class _PhysxManager:
    pass


def _install_runtime(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = _Cuda
    warp = types.ModuleType("warp")
    warp.get_device = lambda name: _WarpDevice(name)

    public = types.ModuleType("isaacsim.core.simulation_manager")
    public.SimulationManager = _OriginalSimulationManager
    implementation = types.ModuleType(
        "isaacsim.core.simulation_manager.impl.simulation_manager"
    )
    implementation.SimulationManager = _OriginalSimulationManager
    physics = types.ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = _PhysxManager
    physics.IsaacEvents = object()
    package = types.ModuleType("isaaclab_physx")
    package.physics = physics
    package.__path__ = []
    package._patch_isaacsim_simulation_manager = (
        lambda: setattr(public, "SimulationManager", _PhysxManager)
    )
    for name, module in {
        "torch": torch,
        "warp": warp,
        "isaacsim.core.simulation_manager": public,
        "isaacsim.core.simulation_manager.impl.simulation_manager": implementation,
        "isaaclab_physx": package,
        "isaaclab_physx.physics": physics,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return public


def test_preconstruction_disables_original_owner_and_binds_one_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _install_runtime(monkeypatch)
    _OriginalSimulationManager.disabled_with = None

    receipt = prepare_native_task_arena_preconstruction(expected_device="cuda:2")

    assert receipt["passed"] is True
    assert receipt["expected_device"] == "cuda:2"
    assert receipt["observed"]["torch_device"] == "cuda:2"
    assert receipt["observed"]["warp_device"] == "cuda:2"
    assert receipt["observed"]["callbacks_disabled"] is True
    assert _OriginalSimulationManager.disabled_with is False
    assert public.SimulationManager is _PhysxManager
    assert receipt["upstream_contract"]["simulation_lifecycle_ownership_fix"] == (
        UPSTREAM_OWNERSHIP_FIX
    )
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_preconstruction_receipt_rejects_device_or_byte_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    receipt = prepare_native_task_arena_preconstruction(expected_device="cuda:0")

    with pytest.raises(
        ValueError, match="native_task_arena_preconstruction_receipt_invalid"
    ):
        validate_native_task_arena_preconstruction_receipt(
            receipt, expected_device="cuda:1"
        )

    receipt["schema_version"] = SCHEMA_VERSION + ".mutated"
    with pytest.raises(
        ValueError, match="native_task_arena_preconstruction_receipt_invalid"
    ):
        validate_native_task_arena_preconstruction_receipt(
            receipt, expected_device="cuda:0"
        )

