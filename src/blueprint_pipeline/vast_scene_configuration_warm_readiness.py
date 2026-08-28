"""Typed Vast heartbeat evidence for retained scene-configuration runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


WARM_RUNTIME_READY_MARKER = "BLUEPRINT_VAST_SCENE_CONFIGURATION_WARM_RUNTIME_READY"
ARTIFIXER_WARM_RUNTIME_READY_MARKER = (
    "BLUEPRINT_VAST_SCENE_CONFIGURATION_ARTIFIXER_WARM_RUNTIME_READY"
)


def observed_scene_configuration_warm_readiness(
    heartbeat_text: str,
) -> tuple[bool, bool]:
    """Return the general and specialized readiness observations independently."""

    return (
        WARM_RUNTIME_READY_MARKER in heartbeat_text,
        ARTIFIXER_WARM_RUNTIME_READY_MARKER in heartbeat_text,
    )


def scene_configuration_warm_validation_fields(
    provider_command: Mapping[str, Any],
) -> dict[str, bool]:
    """Require started scene runtime plus an exact typed readiness marker."""

    is_scene_runtime = (
        provider_command.get("provider_bundle_kind")
        == "task_evaluation_scene_configuration"
    )
    is_started_scene_runtime = (
        is_scene_runtime
        and provider_command.get("provider_bundle_downloaded") is True
        and provider_command.get("provider_entrypoint_started") is True
    )
    artifixer_ready = (
        provider_command.get("scene_configuration_artifixer_warm_runtime_ready") is True
    )
    return {
        "scene_configuration_runtime_root_ready": is_started_scene_runtime
        and (
            provider_command.get("scene_configuration_warm_runtime_ready") is True
            or artifixer_ready
        ),
        "scene_configuration_artifixer_warm_runtime_ready": (
            is_scene_runtime and artifixer_ready
        ),
    }


__all__ = [
    "ARTIFIXER_WARM_RUNTIME_READY_MARKER",
    "WARM_RUNTIME_READY_MARKER",
    "observed_scene_configuration_warm_readiness",
    "scene_configuration_warm_validation_fields",
]
