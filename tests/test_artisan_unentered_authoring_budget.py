"""ARTISAN refusals before its model cost gate keep only the GPU allowance.

capture-447ae4f4, 2026-11-01: ARTISAN's sandbox preflight failed on a rented
Vast host (Permission denied: '/isaac-sim/kit/python') before any model call.
The rule only knew Astra's shapes, so the row kept its whole $20 hold and the
owner's $20 scene cap could fund no retry.
"""
import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_unentered_authoring_budget as budget
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS

PREFIX = "stages/stage-3/producer/"
STAGE3_REFUSAL = ("scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:"
                  "scene_configuration_stage_producer_failed:content_agents_rigid_replacement:1")
SANDBOX_LOG = (
    '  File "/workspace/.../task_evaluation_scene_configuration_artisan_driver.py", line 194, in _execute\n'
    "    preflight_runner.preflight()\n"
    '  File "/workspace/.../asset_authoring_sandbox.py", line 78, in preflight\n'
    "    raise AssetSandboxError('asset_sandbox_unavailable')\n"
    "blueprint_pipeline.asset_authoring_sandbox.AssetSandboxError: asset_sandbox_unavailable\n")


def seal(value, field="result_digest"):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def evidence(tmp_path, *, shape="sandbox"):
    identity = {"run_id": "website-run", "source_commit": "a" * 40}
    request = {"run_id": identity["run_id"], "expected_production_commit": identity["source_commit"],
               "spend": {"external_service_caps": {"openai": {"stage_max_cost_usd": {
                   "artifixer_semantic_teacher": 0, "artifixer_visual_review": 0, "content_agents": 15}}}}}
    refusal = STAGE3_REFUSAL
    files = {"provider_output_zip_exclusions.json": {
        "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
        "excluded_directory_names": sorted(EXCLUDED_PARTS)}}
    if shape == "early":
        refusal = ("scene_configuration_provider_failed:RuntimeError:artisan_sandboxed_interpreter_unavailable:"
                   "126:ASSET_KERNEL_SANDBOX_FAILED:[Errno 13] Permission denied: '/isaac-sim/kit/python'")
        first_stage_started = False
    elif shape == "early_controller":
        refusal = ("scene_configuration_provider_failed:RuntimeError:artisan_controller_runtime_unavailable:127:"
                   "python3: error while loading shared libraries: libpython3.12.so.1.0: cannot open shared object file")
        first_stage_started = False
    else:
        first_stage_started = True
        files.update({
            "stages/stage-1/adapter/result.json": {}, "stages/stage-2/adapter/result.json": {},
            PREFIX + "stage_production_input.v1.json": {**identity, "configuration": {
                "authoring_backend": "artisan_v1", "source_observation_kind": "website_capture_frames"},
                "construction_envelope": {}},
            PREFIX + "dependency_results.v1.json": [seal({
                "stage_id": stage, "status": "completed", "execution_class": "no_spend",
                "paid_execution_requested": False, "provider_mutations_performed": 0}, "stage_result_digest")
                for stage in ("stage-1", "stage-2")],
            PREFIX + "artisan_component.lock": "",
            PREFIX + "artisan_stage_deadline.v1.json": {},
            PREFIX + "artisan_runtime/blueprint_request.v1.json": {},
            PREFIX + "artisan_runtime/bridge_context.v1.json": {},
            PREFIX + "artisan_runtime/sandbox_preflight.sandbox_preflight.json": {"status": "blocked"},
            PREFIX + "stage_producer.log": SANDBOX_LOG if shape == "sandbox" else
            "ValueError: artisan_confined_runtime_preflight_failed\n",
        })
    files["task_evaluation_scene_configuration_provider_result.v1.json"] = seal({
        **identity, "status": "blocked", "first_stage_started": first_stage_started, "blockers": [refusal]})
    result = {**identity, "status": "blocked", "retry_cap": 0, "provider_mutations_performed": 1,
              "blockers": ["provider_result_blocker:" + refusal, "scene_configuration_provider_not_completed"],
              "provider_runtime_output_zip_path": str(tmp_path / "output.zip")}
    return request, files, result


def archive(files, result):
    path = result["provider_runtime_output_zip_path"]
    with zipfile.ZipFile(path, "w") as zipped:
        for name, value in files.items():
            zipped.writestr(name, value if isinstance(value, str) else json.dumps(value))
    with open(path, "rb") as stream:
        result["provider_runtime_output_zip_sha256"] = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    return seal(result)


@pytest.mark.parametrize("shape", ["sandbox", "confined_import", "early", "early_controller"])
def test_artisan_refusal_before_the_model_gate_proves_unentered_authoring(tmp_path, shape):
    request, files, result = evidence(tmp_path, shape=shape)
    assert budget.authoring_never_entered(archive(files, result), request)


@pytest.mark.parametrize("mutation", ["cost_gate_reserved", "usage_ledger", "artisan_config", "terminal",
                                      "later_stage", "wrong_blocker", "log_without_refusal", "dependency_paid",
                                      "wrong_run", "archive_changed", "early_stage_started"])
def test_anything_past_the_gate_or_ambiguous_keeps_the_allowance(tmp_path, mutation):
    shape = "early" if mutation == "early_stage_started" else "sandbox"
    request, files, result = evidence(tmp_path, shape=shape)
    if mutation == "cost_gate_reserved":
        files[PREFIX + "artisan_runtime/official_openai_cost/reservation.json"] = {}
    elif mutation == "usage_ledger":
        files[PREFIX + "artisan_runtime/blueprint_usage.json"] = {"calls": []}
    elif mutation == "artisan_config":
        files[PREFIX + "artisan_runtime/artisan.toml"] = "[artisan]\n"
    elif mutation == "terminal":
        files[PREFIX + "artisan_runtime/terminal.v1.json"] = {}
    elif mutation == "later_stage":
        files["stages/stage-4/adapter/result.json"] = {}
    elif mutation == "wrong_blocker":
        provider = files["task_evaluation_scene_configuration_provider_result.v1.json"]
        provider["blockers"] = ["scene_configuration_parent_runtime_budget_exhausted_during_stage_chain"]
        seal(provider)
        result["blockers"] = ["provider_result_blocker:" + provider["blockers"][0]]
    elif mutation == "log_without_refusal":
        files[PREFIX + "stage_producer.log"] = "openai.RateLimitError: slow down\n"
    elif mutation == "dependency_paid":
        row = files[PREFIX + "dependency_results.v1.json"][0]
        row["paid_execution_requested"] = True
        seal(row, "stage_result_digest")
    elif mutation == "wrong_run":
        files[PREFIX + "stage_production_input.v1.json"]["run_id"] = "other"
    elif mutation == "early_stage_started":
        provider = files["task_evaluation_scene_configuration_provider_result.v1.json"]
        provider["first_stage_started"] = True
        seal(provider)
    archive(files, result)
    if mutation == "archive_changed":
        with open(result["provider_runtime_output_zip_path"], "ab") as stream:
            stream.write(b"changed")
    assert not budget.authoring_never_entered(result, request)
