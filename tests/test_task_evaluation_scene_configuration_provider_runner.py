from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path


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


def test_astra_runtime_refusal_is_reported_before_training(tmp_path, monkeypatch, capsys):
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

    def refusal(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('astra_sandboxed_cad_runtime_preflight_failed')

    monkeypatch.setattr(astra, 'preflight_astra_execution_runtime', refusal)
    monkeypatch.setattr(runner, 'execute_scene_configuration_stage_chain',
                        lambda **_: calls.append('training-must-not-run'))
    assert runner.main() == 2
    assert len(calls) == 1 and calls[0]['package'] == runtime/'toolchain/component'
    logged = capsys.readouterr().out
    assert 'BLUEPRINT_SCENE_CONFIGURATION_FAILURE:' in logged
    assert 'astra_sandboxed_cad_runtime_preflight_failed' in logged
