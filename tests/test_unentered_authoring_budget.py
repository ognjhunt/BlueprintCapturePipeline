"""Early admission failures release no GPU allowance and no uncertain model spend."""
import copy
import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_unentered_authoring_budget as budget
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS


def seal(value, field="result_digest"):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def evidence(tmp_path, *, before_producer=False):
    identity = {"run_id": "website-run", "source_commit": "a" * 40}
    request = {"run_id": identity["run_id"], "expected_production_commit": identity["source_commit"],
        "spend": {"external_service_caps": {"openai": {"stage_max_cost_usd": {
            "artifixer_semantic_teacher": 0, "artifixer_visual_review": 0, "content_agents": 5}}}}}
    prefix = "stages/stage-3/producer/"
    files = {
        "provider_output_zip_exclusions.json": {"schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
            "excluded_directory_names": sorted(EXCLUDED_PARTS)},
        "task_evaluation_scene_configuration_provider_result.v1.json": seal({**identity, "status": "blocked", "first_stage_started": True}),
        prefix + "stage_production_input.v1.json": {**identity,
            "configuration": {"authoring_backend": "astra_cad_blender_v1", "source_observation_kind": "website_capture_frames"},
            "construction_envelope": {}},
        prefix + "dependency_results.v1.json": [seal({"stage_id": stage, "status": "completed", "execution_class": "no_spend",
            "paid_execution_requested": False, "provider_mutations_performed": 0}, "stage_result_digest") for stage in ("stage-1", "stage-2")],
        prefix + ".astra-component.lock": "",
        prefix + "stage_producer.log": '  File "astra_driver.py", line 108, in build_authoring_request\n'
            "blueprint_pipeline.task_evaluation_scene_configuration_astra_driver.AstraStageError: astra_derived_disclosure_not_admitted\n",
    }
    result = {**identity, "status": "blocked", "retry_cap": 0,
        "provider_runtime_output_zip_path": str(tmp_path / "output.zip")}
    if before_producer:
        files.pop(prefix + ".astra-component.lock")
        files.pop(prefix + "stage_producer.log")
        refusal = ("scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:"
                   "scene_configuration_raw_secret_environment_forbidden")
        if before_producer == "secret_file":
            refusal = refusal.replace("scene_configuration_raw_secret_environment_forbidden",
                                      "scene_configuration_secret_file_invalid:OPENAI_API_KEY_FILE")
        provider = files["task_evaluation_scene_configuration_provider_result.v1.json"]
        provider["blockers"] = [refusal]
        seal(provider)
        result["blockers"] = ["provider_result_blocker:" + refusal]
    return request, files, result


def archive(files, result):
    path = result["provider_runtime_output_zip_path"]
    with zipfile.ZipFile(path, "w") as zipped:
        for name, value in files.items():
            zipped.writestr(name, value if isinstance(value, str) else json.dumps(value))
    with open(path, "rb") as stream:
        result["provider_runtime_output_zip_sha256"] = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    return seal(result)


@pytest.mark.parametrize("before_producer", [False, True, "secret_file"])
def test_exact_initial_admission_refusal_proves_unentered_authoring(tmp_path, before_producer):
    request, files, result = evidence(tmp_path, before_producer=before_producer)
    assert budget.authoring_never_entered(archive(files, result), request)


@pytest.mark.parametrize("mutation", ["model_attempt", "later_stage", "partial_adoption", "pretraining",
    "wrong_run", "unknown_error", "dependency_paid", "exclusion", "archive_changed", "seal_changed", "other_api_cap"])
@pytest.mark.parametrize("before_producer", [False, True, "secret_file"])
def test_incomplete_ambiguous_or_previously_paid_work_keeps_allowance(tmp_path, mutation, before_producer):
    request, files, result = evidence(tmp_path, before_producer=before_producer)
    prefix = "stages/stage-3/producer/"
    if mutation == "model_attempt":
        files[prefix + "astra_cad_blender_runtime/reservations.json"] = {}
    elif mutation == "later_stage":
        files["stages/stage-4/partial.json"] = {}
    elif mutation == "partial_adoption":
        files[prefix + "stage_production_input.v1.json"]["construction_envelope"]["partial_astra_successor"] = {}
    elif mutation == "pretraining":
        result["api_pretraining"] = {"status": "completed"}
    elif mutation == "wrong_run":
        files[prefix + "stage_production_input.v1.json"]["run_id"] = "other"
    elif mutation == "unknown_error":
        if before_producer:
            provider = files["task_evaluation_scene_configuration_provider_result.v1.json"]
            provider["blockers"] = ["timeout"]
            seal(provider)
        else:
            files[prefix + "stage_producer.log"] = "timeout after model call"
    elif mutation == "dependency_paid":
        row = files[prefix + "dependency_results.v1.json"][0]
        row["paid_execution_requested"] = True
        seal(row, "stage_result_digest")
    elif mutation == "exclusion":
        files["provider_output_zip_exclusions.json"]["excluded_directory_names"].append("costs")
    elif mutation == "other_api_cap":
        request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]["artifixer_visual_review"] = 1
    archive(files, result)
    if mutation == "seal_changed":
        result["status"] = "completed"
    elif mutation == "archive_changed":
        with open(result["provider_runtime_output_zip_path"], "ab") as stream:
            stream.write(b"changed")
    assert not budget.authoring_never_entered(result, request)


def test_pretraining_refusal_is_specific_and_before_any_paid_phase():
    result = {"provider_mutations_performed": 0,
        "blockers": ["vast_adapter_failed:ValueError:artifixer_pretraining_first_stage_invalid"]}
    assert budget.pretraining_never_entered(result)
    for key, value in (("api_pretraining", {}), ("cpu_prestage", {}), ("provider_mutations_performed", 1),
                       ("provider_runtime_output_zip_path", "/output.zip"), ("blockers", ["provider_timeout"])):
        changed = copy.deepcopy(result)
        changed[key] = value
        assert not budget.pretraining_never_entered(changed)
