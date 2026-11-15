"""Select the immutable native Task Arena bundle validator by execution mode."""

from __future__ import annotations

from typing import Any

from .native_task_arena_construction_bundle import (
    load_verified_native_task_arena_construction_bundle,
)
from .native_task_arena_controls_bundle import (
    load_verified_native_task_arena_controls_bundle,
)
from .native_task_arena_destination_qualification_bundle import (
    load_verified_native_task_arena_destination_qualification_bundle,
)


def native_task_arena_bundle_loader(mode: str) -> Any:
    """Return the fail-closed validator for one admitted execution mode.

    The policy and policy-diagnostic modes are retired: their bundles are refused here.
    """

    loaders = {
        "destination_qualification": (
            load_verified_native_task_arena_destination_qualification_bundle
        ),
        "construction_canary": load_verified_native_task_arena_construction_bundle,
        "controls": load_verified_native_task_arena_controls_bundle,
    }
    if mode not in loaders:
        raise ValueError("native_task_arena_bundle_mode_not_admitted:" + str(mode))
    return loaders[mode]
