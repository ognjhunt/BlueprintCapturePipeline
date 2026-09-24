"""Release only model allowances proven unentered by retained terminal evidence.

Native billing stays at its full authorized allowance. These early refusal
contracts precede the first model gate; an ordinary failure, a partial authoring
directory, or an incomplete archive is not evidence of zero model spending.
"""
from __future__ import annotations

import hashlib
import json
import math
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


def _anthropic_cpu_authoring_archive(read, names, *, run_id, cap, max_requests):
    """Prove a bounded Claude authoring call without treating usage as a final bill."""
    prefix = "stages/stage-3/producer/astra_cad_blender_runtime/"
    manifest = _sealed(json.loads(read(prefix + "inference_audit.json")),
                       "inference_reservation_manifest_digest")
    rows = manifest.get("reservations")
    _require(manifest.get("schema_version") == "task_evaluation_inference_reservation_manifest.v1"
             and manifest.get("run_id") == run_id and manifest.get("proof_effect") == "none"
             and isinstance(rows, list) and 0 < len(rows) <= max_requests
             and manifest.get("reservation_count") == len(rows)
             and manifest.get("in_flight_unknown_count") == 0
             and isinstance(manifest.get("reserved_max_cost_usd"), (int, float))
             and 0 <= manifest["reserved_max_cost_usd"] <= cap)
    expected = set()
    retained = 0.0
    for row in rows:
        reservation_id = row.get("reservation_id")
        _require(isinstance(reservation_id, str) and reservation_id.startswith("sha256:")
                 and len(reservation_id) == 71 and row.get("status") == "completed")
        token = reservation_id[7:]
        reserved_name = "inference_reservations/reserved/" + token + ".json"
        completed_name = "inference_reservations/completed/" + token + ".json"
        _require(row.get("reservation_path") == reserved_name
                 and row.get("completion_path") == completed_name)
        expected.update((prefix + "inference/" + reserved_name,
                         prefix + "inference/" + completed_name))
        reservation = _sealed(json.loads(read(prefix + "inference/" + reserved_name)),
                              "inference_reservation_digest")
        completion = _sealed(json.loads(read(prefix + "inference/" + completed_name)),
                             "inference_completion_digest")
        projected = reservation.get("projected_max_cost_usd")
        reconciled = completion.get("reconciled_actual_cost_usd")
        _require(reservation.get("schema_version") == "task_evaluation_inference_reservation.v1"
                 and completion.get("schema_version") == "task_evaluation_inference_completion.v1"
                 and reservation.get("run_id") == completion.get("run_id") == run_id
                 and reservation.get("reservation_id") == completion.get("reservation_id") == reservation_id
                 and reservation.get("provider") == completion.get("provider") == "anthropic"
                 and reservation.get("model") == completion.get("model") == "claude-opus-5-5"
                 and reservation.get("authority_digest") == completion.get("authority_digest")
                 and reservation.get("provider_terms_digest") == completion.get("provider_terms_digest")
                 and completion.get("status") == "completed"
                 and completion.get("proof_effect") == "none"
                 and row.get("reservation_digest") == reservation["inference_reservation_digest"]
                 and row.get("completion_digest") == completion["inference_completion_digest"]
                 and isinstance(projected, (int, float)) and not isinstance(projected, bool)
                 and isinstance(reconciled, (int, float)) and not isinstance(reconciled, bool)
                 and math.isfinite(projected) and math.isfinite(reconciled)
                 and 0 <= reconciled <= projected
                 and row.get("projected_max_cost_usd") == projected
                 and row.get("reconciled_actual_cost_usd") == reconciled)
        retained += reconciled
    actual = {name for name in names if name.startswith(prefix + "inference/inference_reservations/")
              and name.endswith(".json")}
    _require(len(expected) == 2 * len(rows) and actual == expected
             and math.isclose(retained, manifest["reserved_max_cost_usd"],
                                                  rel_tol=0, abs_tol=1e-8))


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
        services = request["spend"]["external_service_caps"]
        openai = services["openai"]
        caps = openai["stage_max_cost_usd"]
        _require(set(caps) == {"artifixer_semantic_teacher", "artifixer_visual_review", "content_agents"}
                 and caps["artifixer_semantic_teacher"] == caps["artifixer_visual_review"] == 0)
        anthropic = services.get("anthropic")
        claude = (caps["content_agents"] == 0 and openai["maximum_cost_usd"] == 0
                  and isinstance(anthropic, dict))
        cap = float(anthropic["maximum_cost_usd"] if claude else caps["content_agents"])
        _require(0 < cap <= (anthropic["maximum_cost_usd"] if claude else openai["maximum_cost_usd"])
                 and (not claude or 0 < anthropic["maximum_requests"] <= 32))
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
            log = read("stages/stage-3/producer/stage_producer.log").decode("utf-8")
            reservation_name = prefix + "openai_official_cost_run_reservation.v1.json"
            completion_name = prefix + "openai_official_cost_run_completion.v1.json"
            if not claude:
                reservation = _sealed(json.loads(read(reservation_name)), "reservation_receipt_digest")
                completion = _sealed(json.loads(read(completion_name)), "completion_receipt_digest")
                _require(reservation.get("schema_version") == "openai_official_cost_run_reservation.v1"
                         and reservation.get("status") == "reserved_before_openai_call"
                         and reservation.get("run_id") == request["run_id"]
                         and reservation.get("lane_id") == "task_evaluation_scene_configuration_content_agents"
                         and reservation.get("maximum_cost_usd") == cap
                         and completion.get("schema_version") == "openai_official_cost_run_completion.v1"
                         and completion.get("run_id") == request["run_id"]
                         and completion.get("reservation_receipt_digest") == reservation["reservation_receipt_digest"]
                         and completion.get("provider_call_performed") is True)
                failure_markers = {
                    "AgentsSDKInvocationBlocked": "agents_sdk_inference_budget_ceiling_exceeded",
                    "RateLimitError": "credit_balance_exhausted",
                    "AssetAuthoringError": "AssetAuthoringError",
                }
                marker = failure_markers.get(completion.get("runtime_exception_type"))
                _require(marker is not None and marker in log)
            else:
                _require(reservation_name not in names and completion_name not in names)
                _anthropic_cpu_authoring_archive(read, names, run_id=request["run_id"], cap=cap,
                                                  max_requests=anthropic["maximum_requests"])
                _require("AssetAuthoringError" in log)
        return cap
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError, zipfile.BadZipFile):
        return None
