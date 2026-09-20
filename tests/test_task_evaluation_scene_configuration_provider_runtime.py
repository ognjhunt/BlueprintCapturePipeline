from __future__ import annotations

import json
import time

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
    # The registry also admits alternative mesh adapters; this fixture exercises
    # the canonical six-stage ArtiFixer construction recipe.
    identities = [next(identity for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
                       if identity.capability == capability) for capability in CAPABILITY_ORDER]
    for index, (capability, identity) in enumerate(
        zip(CAPABILITY_ORDER, identities, strict=True), start=1,
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


def test_runs_all_six_stages_inside_one_parent_allocation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    assert capsys.readouterr().out.splitlines() == [
        marker
        for index in range(1, 7)
        for marker in (
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_STARTED:"
            f" index={index}/6 stage_id=stage-{index}",
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_COMPLETED:"
            f" index={index}/6 stage_id=stage-{index}",
        )
    ]


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


def test_completed_stage_checkpoint_survives_later_producer_failure(tmp_path, monkeypatch):
    import zipfile
    from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import preserve_stage_prefix

    envelope, configurations = _inputs(tmp_path)
    output = tmp_path / 'runtime-output'
    stages = output / 'stages'
    stages.mkdir(parents=True)
    checkpoint = tmp_path / 'stage-checkpoint.zip'
    producers = _producers()
    original = producers.execute

    def produce(**kwargs):
        if kwargs['stage']['stage_id'] == 'stage-3':
            raise RuntimeError('actual-later-stage-failure')
        if kwargs['stage']['stage_id'] == 'stage-1':
            (kwargs['output_root'] / 'ckpt_30000.pt').write_bytes(b'completed-training')
        return original(**kwargs)

    monkeypatch.setattr(producers, 'execute', produce)
    with pytest.raises(RuntimeError, match='actual-later-stage-failure'):
        execute_scene_configuration_stage_chain(
            envelope=envelope, configurations=configurations, output_root=stages,
            registry=_registry([]), producer_registry=producers,
            checkpoint_callback=lambda results: preserve_stage_prefix(
                output_root=output, completed_results=results, checkpoint_path=checkpoint),
        )
    with zipfile.ZipFile(checkpoint) as archive:
        assert archive.read('stages/stage-1/producer/ckpt_30000.pt') == b'completed-training'
        marker = json.loads(archive.read('completed_stage_checkpoint.json'))
        assert marker['completed_stage_ids'] == ['stage-1', 'stage-2']
        assert marker['whole_run_completed'] is False
        assert not any(name.startswith('stages/stage-3/') for name in archive.namelist())



def test_failed_new_checkpoint_preserves_previous_completed_prefix(tmp_path):
    import zipfile
    from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import preserve_stage_prefix
    output = tmp_path / "output"
    stage1 = output / "stages/stage-1"
    stage1.mkdir(parents=True)
    (stage1 / "checkpoint.pt").write_bytes(b"completed-stage-one")
    checkpoint = tmp_path / "checkpoint.zip"
    preserve_stage_prefix(output_root=output, completed_results=[{"stage_id": "stage-1"}],
                          checkpoint_path=checkpoint)
    original = checkpoint.read_bytes()
    stage2 = output / "stages/stage-2"
    stage2.mkdir()
    (stage2 / "untrusted-link").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="symlink_forbidden"):
        preserve_stage_prefix(output_root=output,
            completed_results=[{"stage_id": "stage-1"}, {"stage_id": "stage-2"}],
            checkpoint_path=checkpoint)
    assert checkpoint.read_bytes() == original
    with zipfile.ZipFile(checkpoint) as archive:
        assert archive.read("stages/stage-1/checkpoint.pt") == b"completed-stage-one"


def _astra_inputs(tmp_path: Path):
    """The canonical chain, with stage 3 declaring the Astra backend so checkpoints are bound."""
    envelope, configurations = _inputs(tmp_path)
    path = tmp_path / "configuration-3.json"
    path.write_text('{"stage":3,"authoring_backend":"astra_cad_blender_v1"}\n', encoding="utf-8")
    configurations["stage-3"] = ({"stage": 3, "authoring_backend": "astra_cad_blender_v1"}, path)
    return envelope, configurations


def test_stage_limit_executes_a_cpu_prefix_that_a_paid_run_elsewhere_adopts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stages 1-4 are CPU work; only native import and assembly need the Isaac host.

    A prestage on the control plane executes the prefix into ``.../stages``;
    the paid run receives that tree under its own root and, with its own
    deadline, adopts the four checkpoints and executes only stages 5 and 6.
    """
    import shutil

    envelope, configurations = _astra_inputs(tmp_path)
    host = tmp_path / "control-plane" / "runtime_output" / "stages"
    host.mkdir(parents=True)
    observed_host: list[str] = []
    prefix = execute_scene_configuration_stage_chain(
        envelope=envelope, configurations=configurations, output_root=host,
        registry=_registry(observed_host), producer_registry=_producers(), stage_limit="stage-4",
    )
    assert observed_host == ["stage-1", "stage-2", "stage-3", "stage-4"]
    assert prefix["status"] == "completed_prefix" and prefix["whole_run_completed"] is False
    assert prefix["stage_limit"] == "stage-4" and prefix["stage_count"] == 4
    assert (host / "astra_same_run_resume_binding.json").is_file()
    assert all((host / f"stage-{i}" / "completed_stage_checkpoint.json").is_file() for i in range(1, 5))
    binding = json.loads((host / "astra_same_run_resume_binding.json").read_text())
    assert binding["output_root"] == "stages" and "parent_deadline_epoch" not in binding
    from blueprint_pipeline.task_evaluation_astra_stage_resume import bind_same_root_resume
    # The same run bound under a different root and deadline yields the same seal.
    assert bind_same_root_resume(tmp_path / "elsewhere" / "stages", envelope, configurations,
                                 time.time() + 3600) == binding

    container = tmp_path / "workspace" / "runtime_output" / "stages"
    shutil.copytree(host, container)
    capsys.readouterr()
    # The registry stub asserts each stage sees as many dependency results as
    # stages executed so far; the four adopted stages count as executed.
    observed_paid = ["stage-1", "stage-2", "stage-3", "stage-4"]
    full = execute_scene_configuration_stage_chain(
        envelope=envelope, configurations=configurations, output_root=container,
        registry=_registry(observed_paid), producer_registry=_producers(),
        parent_deadline_epoch=time.time() + 10 * 3600,
    )
    assert observed_paid == [f"stage-{i}" for i in range(1, 7)]
    assert full["status"] == "completed" and full["stage_count"] == 6
    assert full["stage_result_digests"][:4] == prefix["stage_result_digests"]
    out = capsys.readouterr().out
    assert out.count("BLUEPRINT_SCENE_CONFIGURATION_STAGE_ADOPTED") == 4
    assert "stage_id=stage-5" in out and "stage_id=stage-6" in out


def test_stage_limit_must_name_a_stage_in_the_recipe(tmp_path: Path) -> None:
    envelope, configurations = _inputs(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with pytest.raises(TaskEvaluationSceneConfigurationProviderRuntimeError,
                       match="scene_configuration_provider_stage_limit_invalid:stage-9"):
        execute_scene_configuration_stage_chain(
            envelope=envelope, configurations=configurations, output_root=outputs,
            registry=_registry([]), producer_registry=_producers(), stage_limit="stage-9",
        )
