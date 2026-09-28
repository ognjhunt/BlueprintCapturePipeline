"""Fail-closed allocation bindings for retained scene diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .task_evaluation_scene_configuration_warm_overlay import (
    SceneConfigurationWarmDiagnosticError,
)


def validate_warm_claim_boundary(value: Mapping[str, Any], *, code: str) -> None:
    if (
        value.get("diagnostic_only") is not True
        or value.get("development_only") is not True
        or value.get("qualification_eligible") is not False
        or value.get("configured_revision_publication_permitted") is not False
        or value.get("offering_publication_permitted") is not False
        or value.get("terminal_e2e_completion_permitted") is not False
        or value.get("arbitrary_command_permitted") is not False
        or value.get("raw_secret_values_recorded") is not False
    ):
        raise SceneConfigurationWarmDiagnosticError(code)


def scene_configuration_warm_iteration_allocation_binding(
    *, session: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": "task-evaluation-scene-configuration",
        "action": "warm_iteration",
        "provider": "vast",
        "session_digest": session.get("session_digest"),
        "provider_instance_id": session.get("provider_instance_id"),
        "iteration_authority_digest": authority.get("authority_digest"),
        "watchdog_deadline_epoch": session.get("watchdog_deadline_epoch"),
        "maximum_provider_allocations": 0,
        "maximum_instance_lifecycle_mutations": 0,
        "maximum_remote_workload_dispatches": 1,
    }


def scene_configuration_warm_closeout_allocation_binding(
    *, session: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": "task-evaluation-scene-configuration",
        "action": "warm_closeout",
        "provider": "vast",
        "session_digest": session.get("session_digest"),
        "provider_instance_id": session.get("provider_instance_id"),
        "watchdog_deadline_epoch": session.get("watchdog_deadline_epoch"),
        "maximum_provider_allocations": 0,
        "maximum_instance_lifecycle_mutations": 1,
        "maximum_remote_workload_dispatches": 0,
    }
