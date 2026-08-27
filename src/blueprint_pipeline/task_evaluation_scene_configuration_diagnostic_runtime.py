"""Execute a checkpointed scene-configuration retry without qualifying it.

This runtime intentionally has a different schema and terminal status from the
production six-stage provider runtime.  Stage 1 resumes *after* the sealed
render and semantic-teacher prefix; later stages may then run normally inside a
new, bounded diagnostic allocation.  The resulting artifacts are debugging
evidence only and can never be published as a configured scene revision.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_adapters import (
    SceneConfigurationAdapterRegistry,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    diagnostic_checkpoint_scientific_binding_digest,
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_orchestrator import (
    STAGE_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    required_remaining_stage_seconds,
)
from .task_evaluation_scene_configuration_stage_producers import (
    SceneConfigurationStageProducerRegistry,
)


RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_stage_chain.v1"
)
STATUS = "completed_diagnostic_only_not_qualification_eligible"
StageOneResumeProducer = Callable[..., Sequence[Mapping[str, Any]]]


class TaskEvaluationSceneConfigurationDiagnosticRuntimeError(RuntimeError):
    """The bounded diagnostic retry could not honor its claim boundary."""


def _mark_diagnostic(result: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(result)
    value.update(
        {
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
        }
    )
    value["stage_result_digest"] = canonical_digest(
        value, digest_field="stage_result_digest"
    )
    return value


def _validate_diagnostic_stage_result(
    result: Mapping[str, Any], *, stage_id: str
) -> dict[str, Any]:
    if (
        result.get("schema_version") != STAGE_RESULT_SCHEMA_VERSION
        or result.get("status") != "completed"
        or result.get("stage_id") != stage_id
        or result.get("canonical_allocator") is not None
        or result.get("provider_mutations_performed") != 0
        or result.get("paid_execution_requested") is not False
        or result.get("executed_inside_parent_configuration_run") is not True
        or result.get("diagnostic_only") is not True
        or result.get("qualification_eligible") is not False
        or result.get("executed_inside_one_parent_provider_run") is not False
        or result.get("configured_revision_publication_permitted") is not False
        or result.get("offering_publication_permitted") is not False
        or result.get("terminal_e2e_completion_permitted") is not False
        or result.get("stage_result_digest")
        != canonical_digest(result, digest_field="stage_result_digest")
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
            f"scene_configuration_diagnostic_stage_result_invalid:{stage_id}"
        )
    return dict(result)


def execute_scene_configuration_diagnostic_stage_chain(
    *,
    checkpoint_root: str | Path,
    envelope: Mapping[str, Any],
    configurations: Mapping[str, tuple[Mapping[str, Any], Path]],
    output_root: str | Path,
    registry: SceneConfigurationAdapterRegistry,
    producer_registry: SceneConfigurationStageProducerRegistry,
    stage_one_resume_producer: StageOneResumeProducer,
    toolchain_digest: str,
    parent_deadline_epoch: float,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Resume after semantic outputs and execute the remaining typed chain."""

    recipe = envelope.get("recipe")
    stages = recipe.get("stage_sequence") if isinstance(recipe, Mapping) else None
    render_inputs = envelope.get("render_inputs_result")
    if (
        not isinstance(stages, list)
        or len(stages) != 6
        or not isinstance(render_inputs, Mapping)
        or set(configurations)
        != {
            str(stage.get("stage_id") or "")
            for stage in stages
            if isinstance(stage, Mapping)
        }
        or not isinstance(parent_deadline_epoch, (int, float))
        or isinstance(parent_deadline_epoch, bool)
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
            "scene_configuration_diagnostic_stage_set_invalid"
        )
    first_stage_id = str(stages[0].get("stage_id") or "")
    first_configuration, first_configuration_path = configurations[first_stage_id]
    stage_input = {
        "configuration": dict(first_configuration),
        "configuration_sha256": "sha256:"
        + hashlib.sha256(first_configuration_path.read_bytes()).hexdigest(),
        "toolchain_digest": toolchain_digest,
        "construction_envelope": dict(envelope),
    }
    expected_binding = diagnostic_checkpoint_scientific_binding_digest(
        stage_input=stage_input,
        render_inputs=render_inputs,
    )
    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root,
        expected_scientific_binding_digest=expected_binding,
    )
    root = Path(output_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
            "scene_configuration_diagnostic_output_root_invalid"
        )
    results: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
                "scene_configuration_diagnostic_stage_set_invalid"
            )
        stage_id = str(stage["stage_id"])
        remaining_seconds = parent_deadline_epoch - clock()
        required_seconds = required_remaining_stage_seconds(stages, start_index=index)
        if remaining_seconds < required_seconds:
            raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
                "scene_configuration_diagnostic_runtime_budget_insufficient:"
                f"{stage_id}:{required_seconds}:{max(0, int(remaining_seconds))}"
            )
        expected_dependencies = [] if index == 0 else [stages[index - 1]["stage_id"]]
        if stage.get("depends_on") != expected_dependencies:
            raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
                f"scene_configuration_diagnostic_dependency_invalid:{stage_id}"
            )
        configuration, configuration_path = configurations[stage_id]
        stage_output = root / stage_id
        stage_output.mkdir(mode=0o750, exist_ok=False)
        execution_class = str(stage.get("execution_class") or "")
        if execution_class == "gpu_canary":
            producer_output = stage_output / "producer"
            producer_output.mkdir(mode=0o750)
            if index == 0:
                produced_artifacts = tuple(
                    stage_one_resume_producer(
                        checkpoint=checkpoint,
                        checkpoint_root=Path(checkpoint_root).expanduser().resolve(),
                        stage=stage,
                        envelope=envelope,
                        configuration=configuration,
                        configuration_path=configuration_path,
                        dependency_results=(),
                        output_root=producer_output,
                    )
                )
            else:
                produced_artifacts = producer_registry.execute(
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
            raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
                f"scene_configuration_diagnostic_execution_class_invalid:{stage_id}"
            )
        adapter_output = stage_output / "adapter"
        adapter_output.mkdir(mode=0o750)
        result = registry.execute(
            stage=stage,
            envelope=envelope,
            configuration=configuration,
            configuration_path=configuration_path,
            dependency_results=tuple(results),
            output_root=adapter_output,
            provider_runtime_artifacts=produced_artifacts,
        )
        results.append(
            _validate_diagnostic_stage_result(
                _mark_diagnostic(result), stage_id=stage_id
            )
        )
    if parent_deadline_epoch - clock() < OUTPUT_AND_CLOSURE_RESERVE_SECONDS:
        raise TaskEvaluationSceneConfigurationDiagnosticRuntimeError(
            "scene_configuration_diagnostic_runtime_budget_insufficient:"
            f"output_closure:{OUTPUT_AND_CLOSURE_RESERVE_SECONDS}:"
            f"{max(0, int(parent_deadline_epoch - clock()))}"
        )
    chain: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": STATUS,
        "run_id": envelope["run_id"],
        "source_checkpoint_digest": checkpoint["checkpoint_digest"],
        "stage_result_digests": [row["stage_result_digest"] for row in results],
        "stage_results": results,
        "stage_count": 6,
        "resumed_after_stage_one_semantic_teacher": True,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "executed_inside_one_parent_provider_run": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "nested_provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "evaluation_episode_executed": False,
        "retry_cap": 0,
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    return chain


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "STATUS",
    "TaskEvaluationSceneConfigurationDiagnosticRuntimeError",
    "execute_scene_configuration_diagnostic_stage_chain",
]
