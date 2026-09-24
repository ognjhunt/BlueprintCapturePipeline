"""Release only model allowances proven unentered by retained terminal evidence.

Native billing stays at its full authorized allowance. These early refusal
contracts precede the first model gate; an ordinary failure, a partial authoring
directory, or an incomplete archive is not evidence of zero model spending.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from .decision_evidence_contracts import canonical_digest


def pretraining_never_entered(result):
    return (result.get("api_pretraining") is None
            and result.get("cpu_prestage") is None
            and result.get("provider_mutations_performed") == 0
            and result.get("provider_runtime_output_zip_path") is None
            and "vast_adapter_failed:ValueError:artifixer_pretraining_first_stage_invalid"
            in result.get("blockers", []))


def _require(condition):
    if not condition:
        raise ValueError("unentered_authoring_evidence_invalid")


def _sealed(value, field):
    _require(value.get(field) == canonical_digest(value, digest_field=field))
    return value


def authoring_never_entered(result, request):
    """Recognize initial credential/disclosure refusals, never a later failure.

Read the digest-bound complete output archive, not mutable extracted files.
Only the input and dependencies may exist before producer entry; the lock and
initial refusal log may also exist for the disclosure check. A cost ledger,
resumed authoring, or any later stage keeps the hold.
"""
    try:
        _sealed(result, "result_digest")
        _require(result.get("status") == "blocked" and result.get("retry_cap") == 0
                 and result.get("api_pretraining") is None and result.get("cpu_prestage") is None)
        caps = request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]
        _require(set(caps) == {"artifixer_semantic_teacher", "artifixer_visual_review", "content_agents"}
                 and caps["artifixer_semantic_teacher"] == caps["artifixer_visual_review"] == 0)
        path = Path(result["provider_runtime_output_zip_path"])
        _require(not path.is_symlink() and path.is_file())
        with path.open("rb") as stream:
            digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        _require(digest == result["provider_runtime_output_zip_sha256"])
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)))
            prefix = "stages/stage-3/producer/"
            initial = {prefix + name for name in (
                "dependency_results.v1.json", "stage_production_input.v1.json")}
            expected = initial | {prefix + ".astra-component.lock", prefix + "stage_producer.log"}
            stage3 = {name for name in names if name.startswith("stages/stage-3/")}
            before_producer = stage3 == initial
            _require((before_producer or stage3 == expected) and not any(
                name.startswith("stages/stage-") and not name.startswith(
                    ("stages/stage-1/", "stages/stage-2/", "stages/stage-3/")) for name in names))

            def read(name):
                _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                return archive.read(name)

            exclusions = json.loads(read("provider_output_zip_exclusions.json"))
            from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
            _require(exclusions == {
                "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
                "excluded_directory_names": sorted(EXCLUDED_PARTS)})
            provider = _sealed(json.loads(read("task_evaluation_scene_configuration_provider_result.v1.json")), "result_digest")
            stage_input = json.loads(read(prefix + "stage_production_input.v1.json"))
            _require(provider.get("status") == "blocked" and provider.get("first_stage_started") is True)
            for value in (provider, stage_input):
                _require(value.get("run_id") == result["run_id"] == request["run_id"]
                         and value.get("source_commit") == result["source_commit"] == request["expected_production_commit"])
            config = stage_input["configuration"]
            envelope = stage_input["construction_envelope"]
            _require(config.get("authoring_backend") == "astra_cad_blender_v1"
                     and config.get("source_observation_kind") == "website_capture_frames"
                     and config.get("astra_phase_adoption") is None
                     and envelope.get("partial_astra_successor") is None)
            dependencies = json.loads(read(prefix + "dependency_results.v1.json"))
            _require([row.get("stage_id") for row in dependencies] == ["stage-1", "stage-2"])
            for row in dependencies:
                _sealed(row, "stage_result_digest")
                _require(row.get("status") == "completed" and row.get("execution_class") == "no_spend"
                         and row.get("paid_execution_requested") is False and row.get("provider_mutations_performed") == 0)
            if before_producer:
                from .task_evaluation_scene_configuration_builtin_producers import _SECRET_ENVIRONMENT_FILES
                prefix = "scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:"
                refusals = {prefix + "scene_configuration_raw_secret_environment_forbidden"}
                refusals.update(prefix + "scene_configuration_secret_file_invalid:" + name
                                for name in _SECRET_ENVIRONMENT_FILES)
                blockers = provider.get("blockers") or []
                _require(len(blockers) == 1 and blockers[0] in refusals
                         and "provider_result_blocker:" + blockers[0] in result.get("blockers", []))
            else:
                log = read(prefix + "stage_producer.log").decode("utf-8")
                _require("in build_authoring_request\n" in log and
                         "blueprint_pipeline.task_evaluation_scene_configuration_astra_driver.AstraStageError: astra_derived_disclosure_not_admitted\n" in log)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile):
        return False


def prestage_before_first_stage(result, request):
    """Prove a CPU preflight stopped before any stage or model allowance began."""
    try:
        _sealed(result, "result_digest")
        _require(result.get("status") == "blocked" and result.get("retry_cap") == 0
                 and result.get("provider_mutations_performed") == 0
                 and result.get("api_pretraining") is None and result.get("cpu_prestage") is None)
        path = Path(result["provider_runtime_output_zip_path"])
        _require(path.is_file() and not path.is_symlink())
        with path.open("rb") as stream:
            digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        _require(digest == result["provider_runtime_output_zip_sha256"])
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)) and not any(name.startswith("stages/") for name in names))
            def read(name):
                _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                return archive.read(name)
            exclusions = json.loads(read("provider_output_zip_exclusions.json"))
            from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
            _require(exclusions == {
                "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
                "excluded_directory_names": sorted(EXCLUDED_PARTS)})
            provider = _sealed(json.loads(read("task_evaluation_scene_configuration_provider_result.v1.json")),
                               "result_digest")
            blockers = provider.get("blockers") or []
            runtime_setup_failure = blockers == ["scene_configuration_provider_python_runtime_invalid"]
            early_runtime_identity = (runtime_setup_failure
                and set(provider) == {"schema_version", "status", "blockers", "first_stage_started",
                                      "evaluation_episode_executed", "candidate_policy_queried",
                                      "provider_zero_required_after_return", "result_digest"}
                and provider.get("provider_zero_required_after_return") is True)
            _require(provider.get("status") == "blocked" and provider.get("first_stage_started") is False
                     and provider.get("evaluation_episode_executed") is False
                     and provider.get("candidate_policy_queried") is False
                     and result.get("run_id") == request["run_id"]
                     and result.get("source_commit") == request["expected_production_commit"]
                     and (early_runtime_identity or
                          (provider.get("run_id") == result["run_id"]
                           and provider.get("source_commit") == result["source_commit"])))
            from .core.common import redacted_failure_text
            detail = " ".join(redacted_failure_text(blockers[0]).split()) if len(blockers) == 1 else ""
            if len(detail) > 300:
                detail = detail[:297] + "..."
            if runtime_setup_failure:
                _require(read("provider_python_runtime_setup.log") in {
                    b"BLUEPRINT_SCENE_CONFIGURATION_BLOCKED:scene_configuration_python_import_preflight_failed\n",
                    b"BLUEPRINT_SCENE_CONFIGURATION_BLOCKED:scene_configuration_python_import_preflight_timed_out\n",
                })
            _require(len(blockers) == 1
                     and (runtime_setup_failure or blockers[0].startswith("scene_configuration_provider_failed:"))
                     and "provider_result_blocker:" + detail in result.get("blockers", []))
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile):
        return False


def prestage_authoring_cap_upper_bound(result, request):
    """Retain the full model cap when bounded CPU authoring failed before GPU.

    This is an upper bound, not a claim of final API billing. The archived
    official reservation must prove that the stage actually used the cap in
    the signed request; no later stage or provider allocation may have run.
    """
    try:
        _sealed(result, "result_digest")
        _require(result.get("status") == "blocked" and result.get("retry_cap") == 0
                 and result.get("provider_mutations_performed") == 0
                 and result.get("api_pretraining") is None and result.get("cpu_prestage") is None)
        caps = request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]
        _require(set(caps) == {"artifixer_semantic_teacher", "artifixer_visual_review", "content_agents"}
                 and caps["artifixer_semantic_teacher"] == caps["artifixer_visual_review"] == 0)
        cap = float(caps["content_agents"])
        _require(0 < cap <= request["spend"]["external_service_caps"]["openai"]["maximum_cost_usd"])
        path = Path(result["provider_runtime_output_zip_path"])
        _require(path.is_file() and not path.is_symlink())
        with path.open("rb") as stream:
            _require("sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
                     == result["provider_runtime_output_zip_sha256"])
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)) and not any(
                name.startswith("stages/stage-") and not name.startswith(
                    ("stages/stage-1/", "stages/stage-2/", "stages/stage-3/")) for name in names))

            def read(name):
                _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                return archive.read(name)

            exclusions = json.loads(read("provider_output_zip_exclusions.json"))
            from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
            _require(exclusions == {
                "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
                "excluded_directory_names": sorted(EXCLUDED_PARTS)})
            provider = _sealed(json.loads(read("task_evaluation_scene_configuration_provider_result.v1.json")),
                               "result_digest")
            blocker = "scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:" \
                      "scene_configuration_stage_producer_failed:content_agents_rigid_replacement:1"
            _require(provider.get("status") == "blocked" and provider.get("first_stage_started") is True
                     and provider.get("evaluation_episode_executed") is False
                     and provider.get("candidate_policy_queried") is False
                     and provider.get("run_id") == result.get("run_id") == request["run_id"]
                     and provider.get("source_commit") == result.get("source_commit")
                     == request["expected_production_commit"]
                     and provider.get("blockers") == [blocker]
                     and "provider_result_blocker:" + blocker in result.get("blockers", []))
            prefix = "stages/stage-3/producer/astra_cad_blender_runtime/official_openai_cost/"
            reservation = _sealed(json.loads(read(prefix + "openai_official_cost_run_reservation.v1.json")),
                                  "reservation_receipt_digest")
            completion = _sealed(json.loads(read(prefix + "openai_official_cost_run_completion.v1.json")),
                                 "completion_receipt_digest")
            _require(reservation.get("schema_version") == "openai_official_cost_run_reservation.v1"
                     and reservation.get("status") == "reserved_before_openai_call"
                     and reservation.get("run_id") == request["run_id"]
                     and reservation.get("lane_id") == "task_evaluation_scene_configuration_content_agents"
                     and reservation.get("maximum_cost_usd") == cap
                     and completion.get("schema_version") == "openai_official_cost_run_completion.v1"
                     and completion.get("run_id") == request["run_id"]
                     and completion.get("reservation_receipt_digest") == reservation["reservation_receipt_digest"]
                     and completion.get("provider_call_performed") is True)
            log = read("stages/stage-3/producer/stage_producer.log").decode("utf-8")
            failure_markers = {
                "AgentsSDKInvocationBlocked": "agents_sdk_inference_budget_ceiling_exceeded",
                "RateLimitError": "credit_balance_exhausted",
                "AssetAuthoringError": "AssetAuthoringError",
            }
            marker = failure_markers.get(completion.get("runtime_exception_type"))
            _require(marker is not None and marker in log)
        return cap
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError, zipfile.BadZipFile):
        return None
