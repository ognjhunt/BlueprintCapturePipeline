"""Closed adapter registry for production scene-configuration stages.

Recipes select a stable capability plus a versioned adapter identity.  They
never provide Python imports, shell commands, or provider-specific launch
arguments.  Production owns this registry and therefore owns every executable
selected after an authenticated Website submission.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


StageAdapter = Callable[..., Mapping[str, Any]]


class TaskEvaluationSceneConfigurationAdapterError(RuntimeError):
    """A recipe requested an unavailable or mismatched production adapter."""


@dataclass(frozen=True, slots=True)
class SceneConfigurationAdapterIdentity:
    capability: str
    adapter_id: str
    version: str
    execution_class: str


ADMITTED_STAGE_ADAPTER_IDENTITIES = (
    SceneConfigurationAdapterIdentity(
        "observed_appearance_object_removal",
        "artifixer3d_observed_object_removal",
        "v1",
        "gpu_canary",
    ),
    SceneConfigurationAdapterIdentity(
        "collision_object_excision",
        "sage_exact_prim_excision",
        "v1",
        "no_spend",
    ),
    SceneConfigurationAdapterIdentity(
        "rigid_replacement_authoring",
        "content_agents_rigid_replacement",
        "v1",
        "gpu_canary",
    ),
    SceneConfigurationAdapterIdentity(
        "replacement_static_qualification",
        "simready_static_rigid_qualification",
        "v1",
        "no_spend",
    ),
    SceneConfigurationAdapterIdentity(
        "replacement_native_import_qualification",
        "simready_native_import_qualification",
        "v1",
        "gpu_canary",
    ),
    SceneConfigurationAdapterIdentity(
        "scene_assembly",
        "native_task_scene_assembly",
        "v1",
        "no_spend",
    ),
)


def _identity_from_stage(
    stage: Mapping[str, Any],
) -> SceneConfigurationAdapterIdentity:
    adapter = stage.get("adapter")
    if not isinstance(adapter, Mapping):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "scene_configuration_stage_adapter_identity_invalid"
        )
    return SceneConfigurationAdapterIdentity(
        capability=str(stage.get("capability") or ""),
        adapter_id=str(adapter.get("id") or ""),
        version=str(adapter.get("version") or ""),
        execution_class=str(stage.get("execution_class") or ""),
    )


class SceneConfigurationAdapterRegistry:
    """Resolve only explicitly admitted repository-owned stage handlers."""

    def __init__(
        self,
        handlers: Mapping[SceneConfigurationAdapterIdentity, StageAdapter],
        *,
        admitted: Sequence[SceneConfigurationAdapterIdentity] = (
            ADMITTED_STAGE_ADAPTER_IDENTITIES
        ),
    ) -> None:
        admitted_set = frozenset(admitted)
        if len(admitted_set) != len(admitted):
            raise TaskEvaluationSceneConfigurationAdapterError(
                "scene_configuration_adapter_admission_duplicate"
            )
        if not handlers or any(
            identity not in admitted_set or not callable(handler)
            for identity, handler in handlers.items()
        ):
            raise TaskEvaluationSceneConfigurationAdapterError(
                "scene_configuration_adapter_registry_invalid"
            )
        self._admitted = admitted_set
        self._handlers = dict(handlers)

    def execute(
        self,
        *,
        stage: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        identity = _identity_from_stage(stage)
        if identity not in self._admitted:
            raise TaskEvaluationSceneConfigurationAdapterError(
                "scene_configuration_stage_adapter_not_admitted:"
                f"{identity.adapter_id}:{identity.version}"
            )
        handler = self._handlers.get(identity)
        if handler is None:
            raise TaskEvaluationSceneConfigurationAdapterError(
                "scene_configuration_stage_adapter_not_installed:"
                f"{identity.adapter_id}:{identity.version}"
            )
        result = handler(stage=stage, **kwargs)
        if not isinstance(result, Mapping):
            raise TaskEvaluationSceneConfigurationAdapterError(
                "scene_configuration_stage_adapter_result_invalid:"
                f"{identity.adapter_id}:{identity.version}"
            )
        return result


__all__ = [
    "ADMITTED_STAGE_ADAPTER_IDENTITIES",
    "SceneConfigurationAdapterIdentity",
    "SceneConfigurationAdapterRegistry",
    "StageAdapter",
    "TaskEvaluationSceneConfigurationAdapterError",
]
