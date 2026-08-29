"""Select the immutable native Task Arena bundle validator by execution mode."""

from __future__ import annotations

from typing import Any

from .native_task_arena_construction_bundle import (
    load_verified_native_task_arena_construction_bundle,
)
from .native_task_arena_controls_bundle import (
    load_verified_native_task_arena_controls_bundle,
)
from .native_task_arena_policy_bundle import load_verified_native_task_arena_policy_bundle
from .native_task_arena_policy_diagnostic_bundle import (
    load_verified_native_task_arena_policy_diagnostic_bundle,
)


def native_task_arena_bundle_loader(mode: str) -> Any:
    """Return the fail-closed validator for one admitted execution mode."""

    return {
        "construction_canary": load_verified_native_task_arena_construction_bundle,
        "controls": load_verified_native_task_arena_controls_bundle,
        "policy": load_verified_native_task_arena_policy_bundle,
        "policy_diagnostic": load_verified_native_task_arena_policy_diagnostic_bundle,
    }[mode]
