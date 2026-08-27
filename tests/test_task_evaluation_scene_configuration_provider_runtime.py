from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
    SceneConfigurationAdapterRegistry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_orchestrator import (
    STAGE_RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import (
    TaskEvaluationSceneConfigurationProviderRuntimeError,
    execute_scene_configuration_stage_chain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    SERIAL_GPU_STAGE_TIMEOUT_SECONDS,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
    SceneConfigurationStageProducerRegistry,
)
from blueprint_pipeline.task_evaluation_scene_construction_recipe import (
    CAPABILITY_ORDER,
)


def _inputs(tmp_path: Path):
    stages = []
    configurations = {}
    for index, (capability, identity) in enumerate(
        zip(CAPABILITY_ORDER, ADMITTED_STAGE_ADAPTER_IDENTITIES, strict=True),
        start=1,
    ):
        stage_id = f"stage-{index}"
        stage = {
            "stage_id": stage_id,
            "capability": capability,
            "adapter": {"id": identity.adapter_id, "version": identity.version},
            "execution_class": identity.execution_class,
            "depends_on": [] if index == 1 else [f"stage-{index - 1}"],
        }
        stages.append(stage)
        path = tmp_path / f"configuration-{index}.json"
        path.write_text(f'{{"stage":{index}}}\n', encoding="utf-8")
        configurations[stage_id] = ({"stage": index}, path)
    envelope = {
        "run_id": "configure-scene-v1",
        "recipe": {"stage_sequence": stages},
    }
    return envelope, configurations


def _registry(observed: list[str], *, nested_mutation: bool = False):
    handlers = {}
    for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES:
        def execute(
            *,
            stage,
            configuration_path,
            output_root,
            dependency_results,
            identity=identity,
            **_kwargs,
        ):
            assert stage["capability"] == identity.capability
            assert len(dependency_results) == len(observed)
            observed.append(stage["stage_id"])
            artifact = output_root / "artifact.json"
            artifact.write_text("{}\n", encoding="utf-8")
            result = {
                "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                "status": "completed",
                "stage_id": stage["stage_id"],
                "capability": stage["capability"],
                "execution_class": stage["execution_class"],
                "configuration_digest": "sha256:"
                + hashlib.sha256(configuration_path.read_bytes()).hexdigest(),
                "canonical_allocator": None,
                "provider_mutations_performed": 1 if nested_mutation else 0,
                "paid_execution_requested": False,
                "executed_inside_parent_configuration_run": True,
                "retry_cap": 0,
                "raw_secret_values_recorded": False,
                "output_artifacts": [],
                "stage_result_digest": "",
            }
            result["stage_result_digest"] = canonical_digest(
                result, digest_field="stage_result_digest"
            )
            return result

        handlers[identity] = execute
    return SceneConfigurationAdapterRegistry(handlers)


def _diagnostic_registry(observed: list[str]):
    registry = _registry(observed)
    original = registry.execute

    def execute(**kwargs):
        result = dict(original(**kwargs))
        result["diagnostic_only"] = True
        result["qualification_eligible"] = False
        result["executed_inside_one_parent_provider_run"] = False
        result["stage_result_digest"] = canonical_digest(
            result, digest_field="stage_result_digest"
        )
        return result

    registry.execute = execute
    return registry


def _producers():
    handlers = {}
    for identity in ADMITTED_PRODUCER_IDENTITIES:
        def produce(*, stage, output_root, identity=identity, **_kwargs):
            assert stage["capability"] == identity.capability
            artifact = output_root / "producer.json"
            artifact.write_text("{}\n", encoding="utf-8")
            result = {
                "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
                "status": "completed",
                "stage_id": stage["stage_id"],
                "capability": stage["capability"],
                "provider_mutations_performed": 0,
                "paid_execution_requested": False,
                "executed_inside_parent_configuration_run": True,
                "artifacts": [
                    {
                        "role": "producer_result",
                        "path": str(artifact),
                        "digest": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
                "production_result_digest": "",
            }
            result["production_result_digest"] = canonical_digest(
                result, digest_field="production_result_digest"
            )
            return result

        handlers[identity] = produce
    return SceneConfigurationStageProducerRegistry(handlers)


def test_runs_all_six_stages_inside_one_parent_allocation(tmp_path: Path) -> None:
    envelope, configurations = _inputs(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    observed: list[str] = []

    result = execute_scene_configuration_stage_chain(
        envelope=envelope,
        configurations=configurations,
        output_root=outputs,
        registry=_registry(observed),
        producer_registry=_producers(),
    )

    assert observed == [f"stage-{index}" for index in range(1, 7)]
    assert result["stage_count"] == 6
    assert result["executed_inside_one_parent_provider_run"] is True
    assert result["nested_provider_mutations_performed"] == 0
    assert result["evaluation_episode_executed"] is False


def test_rejects_any_stage_that_claims_a_nested_provider_mutation(
    tmp_path: Path,
) -> None:
    envelope, configurations = _inputs(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    with pytest.raises(
        TaskEvaluationSceneConfigurationProviderRuntimeError,
        match="scene_configuration_provider_stage_result_invalid:stage-1",
    ):
        execute_scene_configuration_stage_chain(
            envelope=envelope,
            configurations=configurations,
            output_root=outputs,
            registry=_registry([], nested_mutation=True),
            producer_registry=_producers(),
        )


def test_production_runtime_refuses_diagnostic_stage_results(tmp_path: Path) -> None:
    envelope, configurations = _inputs(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    with pytest.raises(
        TaskEvaluationSceneConfigurationProviderRuntimeError,
        match="scene_configuration_provider_stage_result_invalid:stage-1",
    ):
        execute_scene_configuration_stage_chain(
            envelope=envelope,
            configurations=configurations,
            output_root=outputs,
            registry=_diagnostic_registry([]),
            producer_registry=_producers(),
        )


def test_refuses_before_stage_when_parent_cannot_cover_remaining_chain(
    tmp_path: Path,
) -> None:
    envelope, configurations = _inputs(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    observed: list[str] = []
    now = 1_000.0
    required = SERIAL_GPU_STAGE_TIMEOUT_SECONDS + OUTPUT_AND_CLOSURE_RESERVE_SECONDS

    with pytest.raises(
        TaskEvaluationSceneConfigurationProviderRuntimeError,
        match=(
            "scene_configuration_parent_runtime_budget_insufficient:"
            f"stage-1:{required}:{required - 1}"
        ),
    ):
        execute_scene_configuration_stage_chain(
            envelope=envelope,
            configurations=configurations,
            output_root=outputs,
            registry=_registry(observed),
            producer_registry=_producers(),
            parent_deadline_epoch=now + required - 1,
            clock=lambda: now,
        )

    assert observed == []
