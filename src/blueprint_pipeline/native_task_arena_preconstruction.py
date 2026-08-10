"""Qualify one task-neutral CUDA/PhysX ownership boundary before Arena build.

IsaacLab-Arena's pinned April 2026 IsaacLab source predates the released
simulation-lifecycle ownership fix in IsaacLab PR #6588.  On Isaac Sim 6 the
original ``SimulationManager`` can therefore retain callbacks that invalidate
the PhysX tensor view during the first reset.  This module applies the same
released, scene-neutral ownership rule before ``ArenaEnvBuilder`` reaches
``gym.make`` and proves that Torch and external Warp resolve the requested
CUDA device.

It does not construct a scene, inspect a task, or claim that the later native
view will be coherent.  Post-construction native view readback remains a
separate required gate.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_arena_preconstruction.v1"
UPSTREAM_OWNERSHIP_FIX = "03904ab49152d1bae929513529913b9be2e06808"
UPSTREAM_WARP_EXCLUSION_FIX = "c4169b2f1c41117b67154c569668b8834519a5ee"
_CALLBACK_ATTRIBUTES = (
    "_default_callback_warm_start",
    "_default_callback_on_stop",
    "_default_callback_stage_open",
    "_default_callback_stage_close",
)


def _normalize_cuda_device(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "cuda":
        return "cuda:0"
    if not text.startswith("cuda:"):
        return None
    try:
        index = int(text.split(":", 1)[1])
    except ValueError:
        return None
    if index < 0:
        return None
    return f"cuda:{index}"


def _disable_original_callbacks(
    original_class: type[Any],
) -> tuple[bool, str]:
    try:
        original_class.enable_all_default_callbacks(False)
        return True, "enable_all_default_callbacks"
    except Exception:  # noqa: BLE001 - compatibility fallback mirrors upstream
        present = [name for name in _CALLBACK_ATTRIBUTES if hasattr(original_class, name)]
        for name in present:
            setattr(original_class, name, None)
        disabled = bool(present) and all(
            getattr(original_class, name, None) is None for name in present
        )
        return disabled, "subscription_handles_cleared" if disabled else "unavailable"


def prepare_native_task_arena_preconstruction(
    *, expected_device: str
) -> dict[str, Any]:
    """Bind CUDA libraries and make PhysxManager the sole lifecycle owner.

    The returned receipt is intentionally available before any scene bytes are
    loaded.  A caller must stop before Arena construction unless ``passed`` is
    true, and must still perform the native tensor-view readback afterwards.
    """

    expected = _normalize_cuda_device(expected_device)
    blockers: list[str] = []
    observed: dict[str, Any] = {
        "requested_device": expected,
        "torch_device": None,
        "warp_device": None,
        "isaacsim_simulation_manager_module_present": False,
        "original_simulation_manager_present": False,
        "callbacks_disabled": None,
        "callback_disable_method": "not_required",
        "public_simulation_manager_is_physx_manager": None,
    }
    if expected is None:
        blockers.append("native_task_arena_preconstruction_cuda_device_invalid")
    else:
        device_index = int(expected.split(":", 1)[1])
        try:
            import torch

            torch.cuda.set_device(device_index)
            observed["torch_device"] = f"cuda:{int(torch.cuda.current_device())}"
        except Exception:  # noqa: BLE001 - retained as a typed preflight gap
            blockers.append("native_task_arena_preconstruction_torch_cuda_unavailable")
        if observed["torch_device"] != expected:
            blockers.append("native_task_arena_preconstruction_torch_device_mismatch")

        try:
            import warp as wp

            warp_device = wp.get_device(expected)
            observed["warp_device"] = _normalize_cuda_device(warp_device)
            if not bool(getattr(warp_device, "is_cuda", False)):
                blockers.append("native_task_arena_preconstruction_warp_cuda_unavailable")
        except Exception:  # noqa: BLE001 - retained as a typed preflight gap
            blockers.append("native_task_arena_preconstruction_warp_cuda_unavailable")
        if observed["warp_device"] != expected:
            blockers.append("native_task_arena_preconstruction_warp_device_mismatch")

    try:
        import isaaclab_physx
        from isaaclab_physx.physics import PhysxManager

        public_module = sys.modules.get("isaacsim.core.simulation_manager")
        implementation_module = sys.modules.get(
            "isaacsim.core.simulation_manager.impl.simulation_manager"
        )
        observed["isaacsim_simulation_manager_module_present"] = public_module is not None
        original_class = None
        if implementation_module is not None:
            original_class = getattr(implementation_module, "SimulationManager", None)
        if original_class is None and public_module is not None:
            candidate = getattr(public_module, "SimulationManager", None)
            if candidate is not PhysxManager:
                original_class = candidate
        observed["original_simulation_manager_present"] = original_class is not None
        if original_class is not None and original_class is not PhysxManager:
            disabled, method = _disable_original_callbacks(original_class)
            observed["callbacks_disabled"] = disabled
            observed["callback_disable_method"] = method
            if not disabled:
                blockers.append(
                    "native_task_arena_preconstruction_lifecycle_callbacks_not_disabled"
                )
        else:
            observed["callbacks_disabled"] = True

        patch = getattr(isaaclab_physx, "_patch_isaacsim_simulation_manager", None)
        if callable(patch):
            patch()
        if public_module is not None:
            public_module.SimulationManager = PhysxManager
            if hasattr(isaaclab_physx.physics, "IsaacEvents"):
                public_module.IsaacEvents = isaaclab_physx.physics.IsaacEvents
            observed["public_simulation_manager_is_physx_manager"] = (
                getattr(public_module, "SimulationManager", None) is PhysxManager
            )
            if not observed["public_simulation_manager_is_physx_manager"]:
                blockers.append(
                    "native_task_arena_preconstruction_physics_owner_mismatch"
                )
        else:
            # The compatible Kit experience excludes the original extension;
            # absence is a valid single-owner state.
            observed["public_simulation_manager_is_physx_manager"] = True
    except Exception:  # noqa: BLE001 - retained as a typed preflight gap
        blockers.append("native_task_arena_preconstruction_physics_owner_unavailable")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "upstream_contract": {
            "simulation_lifecycle_ownership_fix": UPSTREAM_OWNERSHIP_FIX,
            "warp_extension_exclusion_fix": UPSTREAM_WARP_EXCLUSION_FIX,
        },
        "expected_device": expected,
        "observed": observed,
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "postconstruction_native_view_readback_still_required": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def validate_native_task_arena_preconstruction_receipt(
    value: Mapping[str, Any], *, expected_device: str
) -> dict[str, Any]:
    """Fail closed on a mutated, failed, or differently targeted receipt."""

    receipt = dict(value)
    expected = _normalize_cuda_device(expected_device)
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("passed") is not True
        or receipt.get("expected_device") != expected
    ):
        raise ValueError("native_task_arena_preconstruction_receipt_invalid")
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "UPSTREAM_OWNERSHIP_FIX",
    "UPSTREAM_WARP_EXCLUSION_FIX",
    "prepare_native_task_arena_preconstruction",
    "validate_native_task_arena_preconstruction_receipt",
]
