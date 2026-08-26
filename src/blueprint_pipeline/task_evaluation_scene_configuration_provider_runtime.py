"""Single-allocation provider runtime for a scene-configuration run.

The paid parent launch owns the one provider mutation.  This runtime executes
the six admitted configuration stages in order inside that already allocated
host.  No stage may invoke the allocator, create another provider resource, or
execute a robot episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_adapters import (
    SceneConfigurationAdapterRegistry,
    TaskEvaluationSceneConfigurationAdapterError,
)
from .task_evaluation_scene_configuration_builtin_adapters import (
    builtin_scene_configuration_adapter_handlers,
)
from .task_evaluation_scene_configuration_builtin_producers import (
    builtin_scene_configuration_stage_producer_registry,
)
from .task_evaluation_scene_configuration_orchestrator import (
    STAGE_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_stage_producers import (
    SceneConfigurationStageProducerRegistry,
)


RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_provider_stage_chain.v1"
)


class TaskEvaluationSceneConfigurationProviderRuntimeError(RuntimeError):
    """The already-allocated provider could not execute the typed chain."""


def execute_scene_configuration_stage_chain(
    *,
    envelope: Mapping[str, Any],
    configurations: Mapping[str, tuple[Mapping[str, Any], Path]],
    output_root: str | Path,
    registry: SceneConfigurationAdapterRegistry | None = None,
    producer_registry: SceneConfigurationStageProducerRegistry | None = None,
) -> dict[str, Any]:
    """Execute all six stages once without any nested paid mutation."""

    recipe = envelope.get("recipe")
    stages = recipe.get("stage_sequence") if isinstance(recipe, Mapping) else None
    if (
        not isinstance(stages, list)
        or len(stages) != 6
        or set(configurations) != {
            str(stage.get("stage_id") or "")
            for stage in stages
            if isinstance(stage, Mapping)
        }
    ):
        raise TaskEvaluationSceneConfigurationProviderRuntimeError(
            "scene_configuration_provider_stage_set_invalid"
        )
    runtime_registry = registry or SceneConfigurationAdapterRegistry(
        builtin_scene_configuration_adapter_handlers()
    )
    runtime_producers = producer_registry or (
        builtin_scene_configuration_stage_producer_registry(
            expected_source_commit=str(envelope.get("expected_production_commit") or "")
        )
    )
    root = Path(output_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise TaskEvaluationSceneConfigurationProviderRuntimeError(
            "scene_configuration_provider_output_root_invalid"
        )
    results: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise TaskEvaluationSceneConfigurationProviderRuntimeError(
                "scene_configuration_provider_stage_set_invalid"
            )
        stage_id = str(stage["stage_id"])
        expected_dependencies = [] if index == 0 else [stages[index - 1]["stage_id"]]
        if stage.get("depends_on") != expected_dependencies:
            raise TaskEvaluationSceneConfigurationProviderRuntimeError(
                f"scene_configuration_provider_dependency_invalid:{stage_id}"
            )
        configuration, configuration_path = configurations[stage_id]
        stage_output = root / stage_id
        stage_output.mkdir(mode=0o750, exist_ok=False)
        execution_class = str(stage.get("execution_class") or "")
        if execution_class == "gpu_canary":
            producer_output = stage_output / "producer"
            producer_output.mkdir(mode=0o750)
            produced_artifacts = runtime_producers.execute(
                stage=stage,
                envelope=envelope,
                configuration=configuration,
                configuration_path=configuration_path,
                dependency_results=tuple(results),
                output_root=producer_output,
            )
        elif execution_class == "no_spend":
            produced_artifacts = ()
        else:
            raise TaskEvaluationSceneConfigurationProviderRuntimeError(
                f"scene_configuration_provider_execution_class_invalid:{stage_id}"
            )
        adapter_output = stage_output / "adapter"
        adapter_output.mkdir(mode=0o750)
        try:
            value = runtime_registry.execute(
                stage=stage,
                envelope=envelope,
                configuration=configuration,
                configuration_path=configuration_path,
                dependency_results=tuple(results),
                output_root=adapter_output,
                provider_runtime_artifacts=produced_artifacts,
            )
        except TaskEvaluationSceneConfigurationAdapterError:
            raise
        result = dict(value)
        if (
            result.get("schema_version") != STAGE_RESULT_SCHEMA_VERSION
            or result.get("status") != "completed"
            or result.get("stage_id") != stage_id
            or result.get("canonical_allocator") is not None
            or result.get("provider_mutations_performed") != 0
            or result.get("paid_execution_requested") is not False
            or result.get("executed_inside_parent_configuration_run") is not True
            or result.get("stage_result_digest")
            != canonical_digest(result, digest_field="stage_result_digest")
        ):
            raise TaskEvaluationSceneConfigurationProviderRuntimeError(
                f"scene_configuration_provider_stage_result_invalid:{stage_id}"
            )
        results.append(result)
    chain: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "run_id": envelope["run_id"],
        "stage_result_digests": [
            result["stage_result_digest"] for result in results
        ],
        "stage_results": results,
        "stage_count": len(results),
        "executed_inside_one_parent_provider_run": True,
        "nested_provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "evaluation_episode_executed": False,
        "retry_cap": 0,
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(
        chain, digest_field="result_digest"
    )
    return chain


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationProviderRuntimeError",
    "execute_scene_configuration_stage_chain",
]
