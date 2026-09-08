from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_artifixer_pretraining as prep
from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_driver as driver
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_scene_configuration_artifixer_driver import _inputs


def _environment(tmp_path):
    envelope, configuration = _inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    value = {"stage": {"stage_id": "stage-1", "adapter": {"id": "artifixer3d_observed_object_removal"}},
             "configuration": configuration, "construction_envelope": envelope,
             "run_id": "run", "source_commit": "a" * 40}
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(value))
    dependencies = tmp_path / "dependencies.json"
    dependencies.write_text("[]")
    return {driver._INPUT_ENV: str(path), driver._DEPENDENCIES_ENV: str(dependencies),
            driver._OUTPUT_ENV: str(output), driver._PACKAGE_ENV: str(tmp_path),
            driver._RESULT_ENV: str(output / "result.json")}, value


def test_cpu_preparation_never_enters_training(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path)
    env[prep.CPU_PREPARATION_ENV] = "1"
    monkeypatch.setattr(driver, "_prepare_semantic_prefix", lambda **_k: ({"teacher": "admitted"}, "secret-not-serialized"))
    monkeypatch.setattr(prep, "_binding", lambda _s: "sha256:" + "1" * 64)
    monkeypatch.setattr(driver, "_run_artifixer_training_round", lambda **_k: pytest.fail("GPU training called"))
    result = driver.execute_artifixer_component(environment=env)
    assert result["status"] == "admitted_for_gpu_training"
    assert result["gpu_execution_performed"] is False
    assert "secret-not-serialized" not in Path(env[driver._RESULT_ENV]).read_text()


def test_gpu_required_admission_refuses_before_inline_api(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path)
    env[prep.GPU_PREPARATION_REQUIRED_ENV] = "1"
    monkeypatch.setattr(driver, "_prepare_semantic_prefix", lambda **_k: pytest.fail("inline API preparation called"))
    with pytest.raises(driver.TaskEvaluationSceneConfigurationArtifixerError, match="pretraining_admission_missing"):
        driver.execute_artifixer_component(environment=env)


def test_stage_key_context_does_not_select_general_file_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY_FILE", "/general-key")
    monkeypatch.setenv("OPENAI_API_KEY", "previous")
    import os
    with driver._temporary_openai_key("stage-key"):
        assert "OPENAI_API_KEY_FILE" not in os.environ
        assert os.environ["OPENAI_API_KEY"] == "stage-key"
    assert os.environ["OPENAI_API_KEY_FILE"] == "/general-key"
    assert os.environ["OPENAI_API_KEY"] == "previous"


def test_capsule_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="archive_path_invalid"):
        prep._extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_capsule_url_is_private_and_redacted():
    from blueprint_pipeline.vast_scene_private_startup_environment import scene_configuration_startup_environments
    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import _secret_values
    url = "https://objects.example/capsule?X-Amz-Signature=private-signature"
    env = {prep.CAPSULE_URL_ENV: url, prep.GPU_PREPARATION_REQUIRED_ENV: "1"}
    public, private = scene_configuration_startup_environments(env)
    assert prep.CAPSULE_URL_ENV not in public
    assert private[prep.CAPSULE_URL_ENV] == url
    assert url in _secret_values(env)


@pytest.mark.parametrize("fault", [None, "archive", "inventory", "binding"])
def test_provider_restores_exact_prepared_data_without_api(tmp_path, monkeypatch, fault):
    logical = tmp_path / "logical"
    monkeypatch.setattr(prep, "LOGICAL_ROOT", logical)
    root = logical / ("a" * 64)
    staged = tmp_path / "staged"
    (staged / "output/released_artifixer_runtime").mkdir(parents=True)
    (staged / "output/diagnostic_checkpoint").mkdir()
    frame = staged / "output/released_artifixer_runtime/teacher.png"
    frame.write_bytes(b"exact prepared image bytes")
    (staged / "output/diagnostic_checkpoint/checkpoint.json").write_text("{}")
    stage = {"run_id": "run", "source_commit": "b" * 40}
    binding = "sha256:" + "c" * 64
    monkeypatch.setattr(prep, "_binding", lambda _s: binding)
    state_path = staged / "output/pretraining_state.json"
    prep.write_pretraining_state(state={"teacher_receipt_path": str(root / "output/teacher.json")},
                                 stage_input=stage, output_path=state_path)
    files = [{"path": str(p.relative_to(staged)), "sha256": prep._sha(p), "size_bytes": p.stat().st_size}
             for p in staged.rglob("*") if p.is_file()]
    if fault == "inventory":
        files[0]["sha256"] = "sha256:" + "0" * 64
    capsule = {"schema_version": prep.CAPSULE_SCHEMA, "status": "admitted_for_gpu_training",
               "logical_root": str(root), **stage,
               "scientific_binding_digest": binding if fault != "binding" else "sha256:" + "d" * 64,
               "state_path": "output/pretraining_state.json", "files": files,
               "gpu_execution_performed": False, "appearance_repair_qualified": False}
    capsule["capsule_digest"] = canonical_digest(capsule, digest_field="capsule_digest")
    (staged / "capsule_manifest.json").write_text(json.dumps(capsule))
    archive = tmp_path / "capsule.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in staged.rglob("*"):
            if p.is_file():
                z.write(p, str(p.relative_to(staged)))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(archive.read_bytes()))
    from blueprint_pipeline import task_evaluation_scene_configuration_diagnostic_checkpoint as checkpoints
    monkeypatch.setattr(checkpoints, "validate_scene_configuration_diagnostic_checkpoint", lambda **_k: {"validated": True})
    out = tmp_path / "provider-output"
    out.mkdir()
    env = {prep.CAPSULE_URL_ENV: "https://objects.example/capsule.zip",
           prep.CAPSULE_SHA_ENV: prep._sha(archive) if fault != "archive" else "sha256:" + "0" * 64,
           prep.CAPSULE_BYTES_ENV: str(archive.stat().st_size), driver._OUTPUT_ENV: str(out)}
    if fault:
        with pytest.raises(ValueError, match="artifixer_pretraining_"):
            prep.consume_pretraining_capsule(environment=env, stage_input=stage)
    else:
        state = prep.consume_pretraining_capsule(environment=env, stage_input=stage)
        assert state["semantic_checkpoint"] == {"validated": True}
        assert (out / "api_pretraining/teacher.png").read_bytes() == frame.read_bytes()


@pytest.mark.parametrize("changed", [False, True])
def test_real_review_cache_requires_identical_multimodal_input(tmp_path, monkeypatch, changed):
    from blueprint_pipeline import task_evaluation_artifixer_ai_visual_review as review
    original = {"review_phase": "pre_training_semantic_targets", "tasks": []}
    original["receipt_digest"] = canonical_digest(original, digest_field="receipt_digest")
    model_input = [{"role": "user", "content": "exact multimodal fixture identity"}]
    execution = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review_execution.v1",
        "status": "completed", "provider_called": True, "review_phase": original["review_phase"],
        "final_composite_receipt_digest": original["receipt_digest"],
        "reviewer": {"model": review.AI_REVIEW_MODEL, "runtime": "openai_agents_sdk"},
        "usage": {"provider_response_id": "resp_fixture"}, "response_store": False,
        "tracing_disabled": True, "raw_secret_values_recorded": False, "decision": "rejected",
        "input_digest": canonical_digest({"input": model_input}),
        "frames": [{"camera_id": "view", "frame_sha256": "sha256:" + "1" * 64}],
    }
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")
    cache = {}
    for key, value in [("review_input", original), ("review_execution", execution)]:
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        cache[key] = {"path": str(path), "sha256": prep._sha(path)}
    cache["cache_digest"] = canonical_digest(cache, digest_field="cache_digest")
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    current_input = [{"role": "user", "content": "changed pixels"}] if changed else model_input
    monkeypatch.setattr(review, "build_artifixer_ai_visual_review_input", lambda **_k: (
        current_input, "task", [{"camera_id": "view", "sha256": "sha256:" + "1" * 64}], {}))
    output = tmp_path / "out"
    output.mkdir()
    result = prep.reuse_real_pretraining_review(cache_path=path, current_input_path=path, output_root=output)
    if changed:
        assert result is None
    else:
        assert result["review"]["decision"] == "rejected"
        assert json.loads((output / "pretraining_review_reuse.json").read_text())["provider_call_performed"] is False
