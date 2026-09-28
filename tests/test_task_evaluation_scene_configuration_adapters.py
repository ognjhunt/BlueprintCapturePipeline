from __future__ import annotations

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
    SceneConfigurationAdapterIdentity,
    SceneConfigurationAdapterRegistry,
    TaskEvaluationSceneConfigurationAdapterError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import (
    builtin_scene_configuration_adapter_handlers,
)


def stage(identity: SceneConfigurationAdapterIdentity) -> dict[str, object]:
    return {
        "stage_id": "stage-1",
        "capability": identity.capability,
        "adapter": {"id": identity.adapter_id, "version": identity.version},
        "execution_class": identity.execution_class,
    }


def test_registry_selects_exact_typed_adapter_without_customer_command() -> None:
    identity = ADMITTED_STAGE_ADAPTER_IDENTITIES[0]
    calls: list[dict[str, object]] = []

    def handler(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    registry = SceneConfigurationAdapterRegistry({identity: handler})
    assert registry.execute(stage=stage(identity), configuration={"scene": "x"}) == {
        "status": "completed"
    }
    assert calls[0]["configuration"] == {"scene": "x"}
    assert "command" not in calls[0]


def test_registry_rejects_unadmitted_adapter_before_handler() -> None:
    admitted = ADMITTED_STAGE_ADAPTER_IDENTITIES[0]
    registry = SceneConfigurationAdapterRegistry(
        {admitted: lambda **_kwargs: {"status": "completed"}}
    )
    requested = SceneConfigurationAdapterIdentity(
        admitted.capability,
        "customer_shell_adapter",
        "v1",
        admitted.execution_class,
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="scene_configuration_stage_adapter_not_admitted",
    ):
        registry.execute(stage=stage(requested))


def test_registry_rejects_admitted_but_uninstalled_adapter() -> None:
    installed = ADMITTED_STAGE_ADAPTER_IDENTITIES[0]
    missing = ADMITTED_STAGE_ADAPTER_IDENTITIES[1]
    registry = SceneConfigurationAdapterRegistry(
        {installed: lambda **_kwargs: {"status": "completed"}}
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="scene_configuration_stage_adapter_not_installed",
    ):
        registry.execute(stage=stage(missing))


def test_registry_rejects_execution_class_substitution() -> None:
    identity = ADMITTED_STAGE_ADAPTER_IDENTITIES[0]
    registry = SceneConfigurationAdapterRegistry(
        {identity: lambda **_kwargs: {"status": "completed"}}
    )
    mutated = stage(identity)
    mutated["execution_class"] = "no_spend"
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="scene_configuration_stage_adapter_not_admitted",
    ):
        registry.execute(stage=mutated)


def test_builtin_registry_installs_every_admitted_stage() -> None:
    handlers = builtin_scene_configuration_adapter_handlers()
    assert set(handlers) == set(ADMITTED_STAGE_ADAPTER_IDENTITIES)
    SceneConfigurationAdapterRegistry(handlers)
