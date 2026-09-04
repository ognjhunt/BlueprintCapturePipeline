"""Small closed dispatch table for Native Task Arena allocator modes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .native_task_arena_construction_bundle import (
    PROBE_KIND as CONSTRUCTION_PROBE_KIND,
    load_verified_native_task_arena_construction_bundle,
)
from .native_task_arena_controls_bundle import (
    PROBE_KIND as CONTROLS_PROBE_KIND,
    load_verified_native_task_arena_controls_bundle,
)
from .native_task_arena_destination_qualification_bundle import (
    PROBE_KIND as DESTINATION_QUALIFICATION_PROBE_KIND,
    load_verified_native_task_arena_destination_qualification_bundle,
)
from .native_task_arena_policy_bundle import (
    PROBE_KIND as POLICY_PROBE_KIND,
    load_verified_native_task_arena_policy_bundle,
)
from .native_task_arena_policy_diagnostic_bundle import (
    PROBE_KIND as POLICY_DIAGNOSTIC_PROBE_KIND,
    load_verified_native_task_arena_policy_diagnostic_bundle,
)
from .native_task_arena_runtime_preflight_bundle import (
    PROBE_KIND as RUNTIME_PREFLIGHT_PROBE_KIND,
    load_verified_native_task_arena_runtime_preflight_bundle,
)
from .native_task_arena_vast import (
    POLICY_PROVIDER_RUNTIME_ENVIRONMENT_NAMES,
    run_native_task_arena_controls_vast,
    run_native_task_arena_destination_qualification_vast,
    run_native_task_arena_policy_diagnostic_vast,
    run_native_task_arena_policy_vast,
    run_native_task_arena_runtime_preflight_vast,
    run_native_task_arena_vast,
)


_MODE_BY_PROBE_KIND = {
    RUNTIME_PREFLIGHT_PROBE_KIND: "runtime_preflight",
    DESTINATION_QUALIFICATION_PROBE_KIND: "destination_qualification",
    CONSTRUCTION_PROBE_KIND: "construction_canary",
    CONTROLS_PROBE_KIND: "controls",
    POLICY_PROBE_KIND: "policy",
    POLICY_DIAGNOSTIC_PROBE_KIND: "policy_diagnostic",
}
NATIVE_TASK_ARENA_PROBE_KINDS = tuple(_MODE_BY_PROBE_KIND)
_LOADER_BY_PROBE_KIND: dict[str, Callable[..., dict[str, Any]]] = {
    RUNTIME_PREFLIGHT_PROBE_KIND: load_verified_native_task_arena_runtime_preflight_bundle,
    DESTINATION_QUALIFICATION_PROBE_KIND: load_verified_native_task_arena_destination_qualification_bundle,
    CONSTRUCTION_PROBE_KIND: load_verified_native_task_arena_construction_bundle,
    CONTROLS_PROBE_KIND: load_verified_native_task_arena_controls_bundle,
    POLICY_PROBE_KIND: load_verified_native_task_arena_policy_bundle,
    POLICY_DIAGNOSTIC_PROBE_KIND: load_verified_native_task_arena_policy_diagnostic_bundle,
}
_RUNNER_BY_PROBE_KIND: dict[str, Callable[..., dict[str, Any]]] = {
    RUNTIME_PREFLIGHT_PROBE_KIND: run_native_task_arena_runtime_preflight_vast,
    DESTINATION_QUALIFICATION_PROBE_KIND: run_native_task_arena_destination_qualification_vast,
    CONSTRUCTION_PROBE_KIND: run_native_task_arena_vast,
    CONTROLS_PROBE_KIND: run_native_task_arena_controls_vast,
    POLICY_PROBE_KIND: run_native_task_arena_policy_vast,
    POLICY_DIAGNOSTIC_PROBE_KIND: run_native_task_arena_policy_diagnostic_vast,
}


def native_task_arena_probe_mode(probe_kind: str) -> str:
    return _MODE_BY_PROBE_KIND[probe_kind]


def native_task_arena_verified_bundle_loader(
    probe_kind: str,
) -> Callable[..., dict[str, Any]]:
    return _LOADER_BY_PROBE_KIND[probe_kind]


def native_task_arena_vast_runner(probe_kind: str) -> Callable[..., dict[str, Any]]:
    return _RUNNER_BY_PROBE_KIND[probe_kind]


__all__ = [
    "NATIVE_TASK_ARENA_PROBE_KINDS",
    "POLICY_PROVIDER_RUNTIME_ENVIRONMENT_NAMES",
    "native_task_arena_probe_mode",
    "native_task_arena_vast_runner",
    "native_task_arena_verified_bundle_loader",
]
