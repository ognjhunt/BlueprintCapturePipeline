"""Release only model allowances proven unentered, or spent, by retained terminal evidence.

Native billing stays at its full authorized allowance. These early refusal
contracts precede the first model gate; an ordinary failure, a partial authoring
directory, or an incomplete archive is not evidence of zero model spending.

``artisan_proven_authoring_spend`` (B-04) covers ARTISAN runs past the gate: a
completed official cost gate that agrees with the usage meter proves what the
authoring stage spent, so the allowance it did not spend can be released.
"""
from __future__ import annotations

import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
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


#: What ARTISAN writes under ``stages/stage-3/producer/`` before its model cost
#: gate (``official_openai_cost`` reservation, then ``artisan.toml`` and the
#: usage ledger). Anything else keeps the allowance.
_ARTISAN_PRE_GATE_FILES = frozenset({
    "dependency_results.v1.json", "stage_production_input.v1.json", "artisan_component.lock",
    "artisan_stage_deadline.v1.json", "stage_producer.log",
    "artisan_runtime/blueprint_request.v1.json", "artisan_runtime/bridge_context.v1.json",
    "artisan_runtime/runtime_code_access.json", "artisan_runtime/sandbox_preflight.sandbox_preflight.json",
})
#: The sandboxed preflight's own scratch root: no model call runs there.
_ARTISAN_PREFLIGHT_SCRATCH = "artisan_runtime/sandbox_preflight/"
_ARTISAN_PRE_GATE_REFUSALS = (
    # ARTISAN's own sandbox preflight (capture-447ae4f4, 2026-11-01).
    ("    preflight_runner.preflight()\n",
     "blueprint_pipeline.asset_authoring_sandbox.AssetSandboxError: asset_sandbox_unavailable\n"),
    # Its confined import check, which also precedes the cost gate.
    ("artisan_confined_runtime_preflight_failed",),
    # Restoring a continued source's workspace, the driver's first step
    # (capture-447ae4f4, 2026-11-02). A restore that succeeded leaves its
    # workspace in artisan_runtime/, which the file list above refuses.
    ("    continuation = _restore_continuation(envelope, runtime, request)\n", ": artisan_continuation_"),
)
#: The runner's checks before stage 1 (#3228, #3230 and the controller
#: import check): no stage, no credential read, no model call.
_EARLY_ISOLATION_REFUSALS = (
    "scene_configuration_provider_failed:AssetSandboxError:asset_sandbox_unavailable",
    "scene_configuration_provider_failed:RuntimeError:artisan_sandboxed_interpreter_unavailable:",
    "scene_configuration_provider_failed:RuntimeError:artisan_blender_runtime_unavailable:",
    "scene_configuration_provider_failed:RuntimeError:artisan_controller_runtime_unavailable:",
)


def _artisan_refused_before_model_gate(names, read, result, request):
    """ARTISAN stopped before its model cost gate: in stage 3, or before stage 1."""

    from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
    exclusions = json.loads(read("provider_output_zip_exclusions.json"))
    _require(exclusions == {
        "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
        "excluded_directory_names": sorted(EXCLUDED_PARTS)})
    provider = _sealed(json.loads(read("task_evaluation_scene_configuration_provider_result.v1.json")),
                       "result_digest")
    _require(provider.get("status") == "blocked"
             and provider.get("run_id") == result["run_id"] == request["run_id"]
             and provider.get("source_commit") == result["source_commit"] == request["expected_production_commit"])
    blockers = provider.get("blockers") or []
    _require(len(blockers) == 1 and "provider_result_blocker:" + blockers[0] in result.get("blockers", []))
    if not any(name.startswith("stages/") for name in names):
        _require(provider.get("first_stage_started") is False
                 and blockers[0].startswith(_EARLY_ISOLATION_REFUSALS))
        return
    prefix = "stages/stage-3/producer/"
    _require(provider.get("first_stage_started") is True and blockers[0] == (
        "scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:"
        "scene_configuration_stage_producer_failed:content_agents_rigid_replacement:1"))
    stage3 = {name[len(prefix):] for name in names if name.startswith("stages/stage-3/")
              and name.startswith(prefix)}
    _require(all(name.startswith("stages/stage-3/producer/") for name in names if name.startswith("stages/stage-3/"))
             and not any(name.startswith("stages/stage-") and not name.startswith(
                 ("stages/stage-1/", "stages/stage-2/", "stages/stage-3/")) for name in names))
    _require({"dependency_results.v1.json", "stage_production_input.v1.json", "stage_producer.log"} <= stage3
             and all(name in _ARTISAN_PRE_GATE_FILES or name.startswith(_ARTISAN_PREFLIGHT_SCRATCH)
                     for name in stage3))
    stage_input = json.loads(read(prefix + "stage_production_input.v1.json"))
    _require(stage_input.get("run_id") == result["run_id"]
             and stage_input.get("source_commit") == result["source_commit"]
             and stage_input["configuration"].get("authoring_backend") == "artisan_v1"
             and stage_input["configuration"].get("source_observation_kind") == "website_capture_frames")
    dependencies = json.loads(read(prefix + "dependency_results.v1.json"))
    _require([row.get("stage_id") for row in dependencies] == ["stage-1", "stage-2"])
    for row in dependencies:
        _sealed(row, "stage_result_digest")
        _require(row.get("status") == "completed" and row.get("execution_class") == "no_spend"
                 and row.get("paid_execution_requested") is False and row.get("provider_mutations_performed") == 0)
    log = read(prefix + "stage_producer.log").decode("utf-8")
    _require(any(all(part in log for part in refusal) for refusal in _ARTISAN_PRE_GATE_REFUSALS))


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
            if _artisan_run(archive, names):
                def read_artisan(name):
                    _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                    return archive.read(name)

                _artisan_refused_before_model_gate(names, read_artisan, result, request)
                return True
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


def _artisan_run(archive, names):
    """Whether this archive is an ARTISAN run (its stage-3 input, or an early ARTISAN check)."""

    name = "stages/stage-3/producer/stage_production_input.v1.json"
    if name in names:
        _require(archive.getinfo(name).file_size <= 2 * 1024**2)
        configuration = json.loads(archive.read(name)).get("configuration") or {}
        return configuration.get("authoring_backend") == "artisan_v1"
    if any(item.startswith("stages/") for item in names):
        return False
    result = "task_evaluation_scene_configuration_provider_result.v1.json"
    if result not in names:
        return False
    _require(archive.getinfo(result).file_size <= 2 * 1024**2)
    blockers = json.loads(archive.read(result)).get("blockers") or []
    return len(blockers) == 1 and str(blockers[0]).startswith(_EARLY_ISOLATION_REFUSALS)


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


# ---------------------------------------------------------------------------
# B-04: a terminal ARTISAN attempt keeps its proven authoring spend, not its cap.

#: Recorded with every proof, so each settled figure names the rule that produced it.
PROVEN_AUTHORING_SPEND_POLICY = "artisan_proven_authoring_spend.v1"
_ARTISAN_RUNTIME = "stages/stage-3/producer/artisan_runtime/"
_COST_GATE = _ARTISAN_RUNTIME + "official_openai_cost/"
_GATE_RESERVATION = "openai_official_cost_run_reservation.v1.json"
_GATE_COMPLETION = "openai_official_cost_run_completion.v1.json"
_GATE_ATTESTATION = "openai_cost_scope_attestation_content_agents.json"
#: The driver's failure snapshots: one directory per failed controller run, numbered in order.
_GATE_FAILURE_SNAPSHOT = re.compile(r"attempts/(0|[1-9][0-9]{0,3})/("
                                    + re.escape(_GATE_RESERVATION) + "|" + re.escape(_GATE_COMPLETION) + ")")
_CONTINUATION_ROOT = _ARTISAN_RUNTIME + "continuation/"
_CONTINUATION_ADOPTION = _CONTINUATION_ROOT + "source/adoption.v1.json"
_CONTINUATION_SOURCE_USAGE = _CONTINUATION_ROOT + "source/blueprint_usage.json"
_ARTISAN_STAGE_FAILURE = ("scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:"
                          "scene_configuration_stage_producer_failed:content_agents_rigid_replacement:1")
#: Float noise only. An official figure above the meter by more is a disagreement.
_AGREEMENT_TOLERANCE_USD = 1e-6
#: The uncaught exception that ends a controller run the provider refused for
#: want of credit. OpenAI never charges a request it refuses this way.
_QUOTA_REFUSAL = re.compile(r"openai\.RateLimitError: Error code: 429 - (\{.*\})")
_QUOTA_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted"})


def _quota_refusal_termination(read, read_json, names, *, snapshots, main_completion, latest, calls, issued,
                               context):
    """Prove a run with no terminal record ended on the provider's credit refusal, with nothing after it.

    The driver writes ``controller-<n>.log`` only after the controller process
    has exited, then its failure snapshot; one log per snapshot means every
    controller run ended and was snapshotted. The latest log must end on the
    uncaught 429 ``insufficient_quota`` refusal, and the meter's last call must
    be that refused request, priced at $0. Any other ending keeps the hold.
    """
    logs = {name for name in names if name.startswith(_ARTISAN_RUNTIME + "controller-") and name.endswith(".log")}
    expected = [f"{_ARTISAN_RUNTIME}controller-{index}.log" for index in range(snapshots)]
    _require(not main_completion and snapshots > 0 and logs == set(expected)
             and latest.get("runtime_result_digest") is None
             and latest.get("runtime_exception_type") == "AuthoringContractError"
             and latest.get("provider_call_performed") is True)
    failure = read_json(_ARTISAN_RUNTIME + "last_failure.json")
    _require(failure == {"status": "failed_or_uncertain", "exception_type": "AuthoringContractError",
                         "request_digest": issued["request_digest"], "upstream_run_id": context.get("upstream_run_id"),
                         "completed_generation_adoptable": False})
    lines = [line for line in read(expected[-1]).decode("utf-8").splitlines() if line.strip()]
    match = _QUOTA_REFUSAL.fullmatch(lines[-1])
    _require(match is not None)
    error = ast.literal_eval(match[1])["error"]
    refused = calls[-1]
    failed = refused.get("failure") or {}
    _require(error.get("type") == "insufficient_quota" and error.get("code") in _QUOTA_CODES
             and refused.get("cost_usd") == 0 and refused.get("cost_origin") == "upstream_rejected_request"
             and failed.get("http_status") == 429 and failed.get("error_type") == "RateLimitError")
    return {"kind": "provider_credit_refusal", "http_status": 429, "error_type": error["type"],
            "error_code": error["code"], "controller_log": expected[-1],
            "refused_call": {"id": refused["id"], "request_id": failed.get("request_id")}}


def _usd(value):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0)
    return float(value)


def artisan_proven_authoring_spend(result, request):
    """What a terminal ARTISAN stage provably spent through its model cost gate, else ``None``.

    The owner approved settling failed attempts at their proven cost
    (2026-11-02, B-04): capture-447ae4f4 held about $58 against about $18
    spent, because every attempt that entered ARTISAN kept its whole authoring
    allowance. Proof comes from the digest-bound output archive only: the GPU
    run's, or the CPU prestage's own when the prestage failed before any GPU.

    - the official cost gate reserved before any model call, for this run, this
      stage input, this candidate and this authority, at most the signed
      ``content_agents`` allowance;
    - every controller run under that reservation left a completion or failure
      snapshot, and the latest one binds the ARTISAN terminal record whose usage
      is exactly the retained meter, so no metered call follows it. With no
      terminal record, only a run that ended on the provider's 429
      ``insufficient_quota`` refusal qualifies, as its controller log and meter
      both end on that uncharged request (``_quota_refusal_termination``);
    - every metered call is priced (a call that raised is priced at its own
      upper bound) under the gate's own cap;
    - the official spend observed since the reservation's baseline is no more
      than the meter's total. The Costs endpoint lags, so the official figure
      may fall short of the meter; above it, something spent unmetered.

    The proven spend is the larger of the two figures. Anything missing,
    malformed, disagreeing, or past stage 3 returns ``None`` and the caller
    keeps the full allowance. A continuation meters only its own increment:
    the spend it carries stays settled under its source attempt, so it is
    recorded here and never counted (``task_evaluation_artisan_continuation``).
    """
    try:
        _sealed(result, "result_digest")
        _require(result.get("status") == "blocked" and result.get("retry_cap") == 0
                 and result.get("api_pretraining") is None and result.get("cpu_prestage") is None)
        allowance = _usd(request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]["content_agents"])
        _require(allowance > 0)
        path = Path(result["provider_runtime_output_zip_path"])
        _require(not path.is_symlink() and path.is_file())
        with path.open("rb") as stream:
            archive_digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        _require(archive_digest == result["provider_runtime_output_zip_sha256"])
        records = {}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)))

            def read(name):
                _require(archive.getinfo(name).file_size <= 2 * 1024**2)
                data = archive.read(name)
                records[name] = "sha256:" + hashlib.sha256(data).hexdigest()
                return data

            def read_json(name):
                value = json.loads(read(name))
                _require(isinstance(value, dict))
                return value

            from .task_evaluation_scene_configuration_output_archive import EXCLUDED_PARTS
            _require(json.loads(read("provider_output_zip_exclusions.json")) == {
                "schema_version": "task_evaluation_scene_configuration_provider_output_zip_exclusions.v1",
                "excluded_directory_names": sorted(EXCLUDED_PARTS)})
            provider = _sealed(read_json("task_evaluation_scene_configuration_provider_result.v1.json"),
                               "result_digest")
            _require(provider.get("status") == "blocked" and provider.get("first_stage_started") is True
                     and provider.get("run_id") == result["run_id"] == request["run_id"]
                     and provider.get("source_commit") == result["source_commit"]
                     == request["expected_production_commit"]
                     and provider.get("blockers") == [_ARTISAN_STAGE_FAILURE]
                     and "provider_result_blocker:" + _ARTISAN_STAGE_FAILURE in result.get("blockers", []))
            # Stage 3 failed and nothing after it ran under this attempt's allowances.
            _require(not any(name.startswith("stages/stage-") and not name.startswith(
                ("stages/stage-1/", "stages/stage-2/", "stages/stage-3/producer/")) for name in names))
            prefix = "stages/stage-3/producer/"
            stage_input_bytes = read(prefix + "stage_production_input.v1.json")
            stage_input = json.loads(stage_input_bytes)
            _require(stage_input.get("run_id") == result["run_id"]
                     and stage_input.get("source_commit") == result["source_commit"]
                     and stage_input["configuration"].get("authoring_backend") == "artisan_v1"
                     and stage_input["configuration"].get("source_observation_kind") == "website_capture_frames")
            dependencies = json.loads(read(prefix + "dependency_results.v1.json"))
            _require([row.get("stage_id") for row in dependencies] == ["stage-1", "stage-2"])
            for row in dependencies:
                _sealed(row, "stage_result_digest")
                _require(row.get("status") == "completed" and row.get("execution_class") == "no_spend"
                         and row.get("paid_execution_requested") is False
                         and row.get("provider_mutations_performed") == 0)
            issued = _sealed(read_json(_ARTISAN_RUNTIME + "blueprint_request.v1.json"), "request_digest")
            _require(issued.get("run_id") == request["run_id"] and issued.get("backend") == "artisan_v1")
            context = read_json(_ARTISAN_RUNTIME + "bridge_context.v1.json")

            main, snapshots = set(), {}
            for name in names:
                if not name.startswith(_COST_GATE):
                    continue
                relative = name[len(_COST_GATE):]
                if relative in (_GATE_RESERVATION, _GATE_COMPLETION, _GATE_ATTESTATION):
                    main.add(relative)
                    continue
                match = _GATE_FAILURE_SNAPSHOT.fullmatch(relative)
                _require(match is not None)
                snapshots.setdefault(int(match[1]), set()).add(match[2])
            _require(_GATE_RESERVATION in main and sorted(snapshots) == list(range(len(snapshots)))
                     and all(files == {_GATE_RESERVATION, _GATE_COMPLETION} for files in snapshots.values()))
            reservation_bytes = read(_COST_GATE + _GATE_RESERVATION)
            reservation = _sealed(json.loads(reservation_bytes), "reservation_receipt_digest")
            held = reservation["provider_reservation"]
            cap = _usd(reservation["maximum_cost_usd"])
            baseline = _usd(held["baseline_cost_snapshot"]["total_cost_usd"])
            identity = {"lane_id": "task_evaluation_scene_configuration_content_agents",
                        "run_id": request["run_id"],
                        "request_digest": "sha256:" + hashlib.sha256(stage_input_bytes).hexdigest(),
                        "candidate_digest": issued["source_candidate"]["digest"],
                        "authorization_receipt_digest": result["authority_digest"]}
            _require(reservation.get("schema_version") == "openai_official_cost_run_reservation.v1"
                     and reservation.get("status") == "reserved_before_openai_call"
                     and all(reservation.get(key) == value for key, value in identity.items())
                     and 0 < cap <= allowance
                     and held.get("candidate_id") == request["run_id"]
                     and held.get("reserved_max_cost_usd") == cap)
            completions = [_COST_GATE + _GATE_COMPLETION] if _GATE_COMPLETION in main else []
            for index in sorted(snapshots):
                # One original reservation governs every run; each failure snapshot copies it.
                _require(read(f"{_COST_GATE}attempts/{index}/{_GATE_RESERVATION}") == reservation_bytes)
                completions.append(f"{_COST_GATE}attempts/{index}/{_GATE_COMPLETION}")
            _require(completions)  # A reservation without a snapshot: the gate never completed.
            observed, ordered = [], []
            for name in completions:
                completion = _sealed(read_json(name), "completion_receipt_digest")
                snapshot = completion["official_completion_snapshot"]
                cost = _usd(completion["provider_observed_cost_usd"])
                completed_at = datetime.fromisoformat(completion["completed_at"])
                _require(completion.get("schema_version") == "openai_official_cost_run_completion.v1"
                         and completion.get("status") == "official_cost_reporting_pending"
                         and all(completion.get(key) == value for key, value in identity.items())
                         and completion.get("reservation_receipt_digest") == reservation["reservation_receipt_digest"]
                         and completion.get("cost_reservation_digest") == held["cost_reservation_digest"]
                         and snapshot.get("project_id") == held.get("project_id")
                         and snapshot.get("api_key_id") == held.get("api_key_id")
                         and completed_at.tzinfo is not None
                         and math.isclose(cost, round(_usd(snapshot["total_cost_usd"]) - baseline, 8),
                                          rel_tol=0, abs_tol=1e-8))
                observed.append(cost)
                ordered.append((completed_at, completion))
            ordered.sort(key=lambda pair: pair[0])
            _require(len(ordered) == 1 or ordered[-1][0] > ordered[-2][0])
            latest = ordered[-1][1]

            usage = read_json(_ARTISAN_RUNTIME + "blueprint_usage.json")
            calls = usage.get("calls")
            _require(usage.get("schema_version") == "artisan_upstream_usage.v1"
                     and usage.get("maximum_cost_usd") == context.get("maximum_cost_usd") == cap
                     and usage.get("maximum_requests") == context.get("maximum_requests")
                     and usage.get("deadline_epoch") == context.get("deadline_epoch")
                     and str(context.get("usage_file")).endswith("/" + _ARTISAN_RUNTIME + "blueprint_usage.json")
                     and isinstance(calls, list) and len(calls) <= usage["maximum_requests"])
            termination = None
            if _ARTISAN_RUNTIME + "terminal.v1.json" in names:
                # The latest snapshot binds the record its run wrote last, and that
                # record's usage is the retained meter: no call was metered after it.
                terminal = read_json(_ARTISAN_RUNTIME + "terminal.v1.json")
                _require(terminal.get("schema_version") == "blueprint_artisan_terminal.v1"
                         and latest.get("runtime_result_digest") == canonical_digest(terminal)
                         and terminal.get("usage") == usage)
            else:
                termination = _quota_refusal_termination(
                    read, read_json, names, snapshots=len(snapshots), main_completion=_GATE_COMPLETION in main,
                    latest=latest, calls=calls, issued=issued, context=context)
            identities, costs = set(), []
            for call in calls:
                _require(isinstance(call, dict) and call.get("status") == "priced"
                         and isinstance(call.get("id"), str))
                identities.add(call["id"])
                costs.append(_usd(call["cost_usd"]))
            _require(len(identities) == len(calls))
            metered = math.fsum(costs)
            official = max(observed)
            _require(official <= metered + _AGREEMENT_TOLERANCE_USD)
            proven = max(official, metered)
            _require(proven <= cap + _AGREEMENT_TOLERANCE_USD)

            continuation = None
            if _CONTINUATION_ADOPTION in names:
                adoption = read_json(_CONTINUATION_ADOPTION)
                carried_calls = read_json(_CONTINUATION_SOURCE_USAGE).get("calls")
                _require(adoption.get("schema_version") == "task_evaluation_artisan_continuation_record.v1"
                         and adoption.get("record_digest") == canonical_digest(adoption, digest_field="record_digest")
                         and adoption.get("successor_run_id") == request["run_id"]
                         and adoption.get("successor_request_digest") == issued["request_digest"]
                         # The meter and the gate received only the increment.
                         and adoption.get("increment_usd") == cap
                         and adoption.get("carried_spend_settled_under_source_attempt") is True
                         and adoption.get("new_spend_authorized") is False
                         and isinstance(carried_calls, list)
                         and not identities & {row.get("id") for row in carried_calls})
                continuation = {"source_run_id": adoption["source_run_id"],
                                "carried_cost_usd": _usd(adoption["carried_cost_usd"]),
                                "carried_spend_settled_under_source_attempt": True,
                                "carried_spend_counted_here": False}
            else:
                _require(not any(name.startswith(_CONTINUATION_ROOT) for name in names))
        return {"policy_version": PROVEN_AUTHORING_SPEND_POLICY,
                "output_archive": {"path": str(path), "sha256": archive_digest},
                "records": [{"name": name, "sha256": digest} for name, digest in sorted(records.items())],
                "authoring_allowance_usd": allowance, "gate_maximum_cost_usd": cap,
                "official_observed_cost_usd": official, "official_cost_final": False,
                "metered_cost_usd": metered, "metered_call_count": len(calls),
                "proven_authoring_spend_usd": proven, "continuation": continuation, "termination": termination}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, UnicodeError,
            zipfile.BadZipFile):
        return None
