from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest


def _provider_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/task_evaluation_scene_configuration_provider_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scene_configuration_provider_runner_redaction_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provider_runner_redacts_failure_before_retaining_result(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _provider_runner()
    runtime = tmp_path / "provider_runtime"
    output = tmp_path / "runtime_output"
    result_path = output / "task_evaluation_scene_configuration_provider_result.v1.json"
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT", str(output))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_PROVIDER_RESULT", str(result_path))
    monkeypatch.setenv(
        "BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH",
        str(time.time() + 27_000),
    )
    monkeypatch.setattr(
        runner,
        "_read",
        lambda _path: {"envelope_digest": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_hydrate_envelope",
        lambda _runtime, _portable: {
            "run_id": "configure-scene",
            "expected_production_commit": "b" * 40,
            "control_plane_envelope_digest": "sha256:" + "c" * 64,
            "stage_configuration_references": [],
        },
    )

    def fail_before_stage_chain(**_kwargs):
        raise RuntimeError(
            "request failed sk-provider-secret-value "
            "https://object.invalid/out?X-Amz-Signature=signed-provider-value"
        )

    monkeypatch.setattr(
        runner, "execute_scene_configuration_stage_chain", fail_before_stage_chain
    )

    assert runner.main() == 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert len(result["blockers"]) == 1
    blocker = result["blockers"][0]
    assert blocker.startswith("scene_configuration_provider_failed:RuntimeError:")
    assert "sk-provider-secret-value" not in blocker
    assert "signed-provider-value" not in blocker
    assert blocker.count("<redacted>") == 2


@pytest.mark.parametrize("runtime_mode", ["refused", "packaged", "configured"])
def test_astra_runtime_preflight_admits_one_runtime_before_stages(tmp_path, monkeypatch, capsys, runtime_mode):
    from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as astra
    from blueprint_pipeline import task_evaluation_scene_configuration_builtin_producers as producers
    runner = _provider_runner()
    runtime, output = tmp_path / 'runtime', tmp_path / 'output'
    monkeypatch.setenv('BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT', str(runtime))
    monkeypatch.setenv('BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT', str(output))
    monkeypatch.setenv('BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH', str(time.time()+27000))
    monkeypatch.setattr(runner, '_read', lambda p: {'authoring_backend': 'astra_cad_blender_v1'}
        if p.name == 'configuration.json' else {'envelope_digest': 'sha256:'+'a'*64})
    monkeypatch.setattr(runner, '_hydrate_envelope', lambda *_: {
        'run_id': 'run', 'expected_production_commit': 'b'*40,
        'stage_configuration_references': [{'stage_id': 'stage-3',
            'materialized_path': str(tmp_path/'configuration.json')}]})
    monkeypatch.setattr(producers, '_validate_toolchain', lambda **_: ({'stages': {
        'content_agents_rigid_replacement': {'component_entrypoint': 'component/run.sh'}}}, []))
    calls = []
    configured = str(tmp_path / 'installed-blender')
    monkeypatch.setenv('BLUEPRINT_BLENDER_RUNTIME_ROOT', configured if runtime_mode == 'configured' else '')

    def preflight(**kwargs):
        calls.append(kwargs)
        if runtime_mode == 'refused':
            raise RuntimeError('astra_sandboxed_cad_runtime_preflight_failed')

    def stages(**kwargs):
        import os
        calls.append(os.environ['BLUEPRINT_BLENDER_RUNTIME_ROOT'])
        raise RuntimeError('test_stops_before_stage_execution')

    monkeypatch.setattr(astra, 'preflight_astra_execution_runtime', preflight)
    monkeypatch.setattr(runner, 'execute_scene_configuration_stage_chain', stages)
    assert runner.main() == 2
    assert calls[0]['package'] == runtime/'toolchain/component'
    logged = capsys.readouterr().out
    if runtime_mode == 'refused':
        assert len(calls) == 1
        assert 'astra_sandboxed_cad_runtime_preflight_failed' in logged
    else:
        assert calls[1] == (configured if runtime_mode == 'configured'
                            else str(output / 'astra_runtime_preflight/packaged_blender'))
        assert 'BLUEPRINT_SCENE_CONFIGURATION_ASTRA_RUNTIME_PREFLIGHT_PASSED' in logged


@pytest.mark.parametrize("changed_asset,stage_limit", [(False, None), (True, None), (False, "stage-4")])
def test_native_runner_validates_real_prefix_before_skipping_authoring_tools(tmp_path, monkeypatch, changed_asset, stage_limit):
    from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as astra
    from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import execute_scene_configuration_stage_chain
    from tests.test_task_evaluation_scene_configuration_provider_runtime import _astra_inputs, _registry, _producers
    runner = _provider_runner()
    envelope, configurations = _astra_inputs(tmp_path)
    envelope.update(expected_production_commit="b" * 40, stage_configuration_references=[
        {"stage_id": name, "materialized_path": str(path)} for name, (_, path) in configurations.items()])
    output = tmp_path / "runtime_output"
    stages = output / "stages"
    stages.mkdir(parents=True)
    prefix_deadline = time.time() + 27000
    execute_scene_configuration_stage_chain(envelope=envelope, configurations=configurations, output_root=stages,
        registry=_registry([], real_artifacts=True), producer_registry=_producers(), stage_limit="stage-4",
        parent_deadline_epoch=prefix_deadline)
    # The transferred prefix also contains this directory from the CPU run.
    # Repeating tool setup would fail before adoption, even with valid assets.
    (output / "astra_runtime_preflight").mkdir()
    if changed_asset:
        (stages / "stage-3/adapter/artifact.json").write_text("changed")
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT", str(output))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH",
                       str(prefix_deadline if stage_limit else time.time() + 27000))
    if stage_limit:
        monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT", stage_limit)
    else:
        monkeypatch.delenv("BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT", raising=False)
    original_read = runner._read
    monkeypatch.setattr(runner, "_read", lambda path: original_read(path) if path.is_file()
                        else {"envelope_digest": "sha256:" + "a" * 64})
    monkeypatch.setattr(runner, "_hydrate_envelope", lambda *_: envelope)
    monkeypatch.setattr(astra, "preflight_astra_execution_runtime", lambda **_: pytest.fail("authoring tools repeated"))
    observed = [f"stage-{i}" for i in range(1, 5)]
    monkeypatch.setattr(runner, "execute_scene_configuration_stage_chain", lambda **kwargs:
        execute_scene_configuration_stage_chain(**kwargs, registry=_registry(observed), producer_registry=_producers()))
    assert runner.main() == (2 if changed_asset else 0)
    result = json.loads((output / (runner.RESULT_SCHEMA_VERSION + ".json")).read_text())
    if changed_asset:
        assert result["status"] == "blocked"
        assert "astra_stage_resume_artifact_changed" in result["blockers"][0]
        assert len(observed) == 4
    elif stage_limit:
        assert result["status"] == "completed_prefix" and len(observed) == 4
        assert result["stage_chain"]["whole_run_completed"] is False
    else:
        assert result["status"] == "completed" and len(observed) == 6
        assert result["stage_chain"]["stage_count"] == 6
