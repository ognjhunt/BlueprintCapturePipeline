"""Validate and optionally stage live external inputs for the control plane.

The intake command is a preflight for real external handoffs. It can inspect a
WebApp decision-evidence queue envelope or ``robot_eval_job_request.v1`` file and an owner-system Arena result
directory against the current live control-plane manifest. It never runs live
simulators, calls providers, uploads storage, or promotes proof claims.
"""

from __future__ import annotations

import argparse
import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .live_pipeline_control_plane import (
    ARENA_RESULT_ARTIFACT_NAMES,
    JOB_REQUEST_INBOX_ENV,
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
    LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
    WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
    WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
    WEBAPP_UPSTREAM_REQUIRED_FIELDS,
)
from .live_robot_eval_closure import LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION


LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION = "blueprint_live_pipeline_input_intake.v1"
LIVE_CLOSURE_EVIDENCE_ARTIFACT_NAME = "live_eval_closure_evidence.json"
POLICY_PACKAGE_ARTIFACT_NAME = "policy_package.json"
LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND = "local_first_gpu_rehearsal_request"
DECISION_EVIDENCE_QUEUE_CONTRACT = "blueprint.decision_evidence_request_inbox.v1"
DECISION_EVIDENCE_REQUEST_SCHEMA_VERSION = "blueprint.decision_evidence_request.v1"
POLICY_MODALITY_ORDER = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "passed", "success", "succeeded"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _section(payload: Mapping[str, Any], *aliases: str) -> Dict[str, Any]:
    for alias in aliases:
        value = payload.get(alias)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _status_ok(value: Any) -> bool:
    return _string(value).lower() in {
        "accepted",
        "complete",
        "completed",
        "passed",
        "ready",
        "succeeded",
        "validated",
        "verified",
    }


def _attestation_ok(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    return bool(
        _string(
            value.get("attested_by")
            or value.get("attestedBy")
            or value.get("operator_id")
            or value.get("operatorId")
            or value.get("reviewer")
        )
        and _string(
            value.get("attestation")
            or value.get("statement")
            or value.get("accepted_claim_boundary")
            or value.get("acceptedClaimBoundary")
        )
    )


def _delivery_access_ready(section: Mapping[str, Any]) -> bool:
    if not section:
        return True
    if (
        _status_ok(section.get("status"))
        or _boolish(section.get("accepted"))
        or bool(section.get("artifact_refs") or section.get("artifactRefs"))
    ):
        return True
    signed_urls = section.get("signed_urls") or section.get("signedUrls")
    signed_access = section.get("signed_access") or section.get("signedAccess")
    storage_upload = section.get("storage_upload_performed") or section.get(
        "storageUploadPerformed"
    )
    entitlement = section.get("entitlement_verified") or section.get("entitlementVerified")
    return bool(
        (isinstance(signed_urls, list) and signed_urls)
        or (isinstance(signed_access, list) and signed_access)
        or _boolish(storage_upload)
        or _boolish(entitlement)
    )


def _safe_job_id(value: Any) -> bool:
    text = _string(value)
    return bool(text) and text not in {".", ".."} and "/" not in text and "\\" not in text


def _read_mapping(path: Path) -> Dict[str, Any]:
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def translate_decision_evidence_envelope_to_legacy_execution_request(
    payload: Mapping[str, Any], *, expected_capture_root: Path | None
) -> Dict[str, Any] | None:
    """Translate the WebApp product envelope into the legacy execution adapter input.

    The original decision request remains the provenance authority. This bounded
    adapter only supplies the legacy simulator scheduler fields while the Task
    Evaluation Run control plane replaces that execution spine.
    """

    if payload.get("queue_contract") != DECISION_EVIDENCE_QUEUE_CONTRACT:
        return None
    decision_request = _mapping(payload.get("decision_request"))
    if decision_request.get("schema_version") != DECISION_EVIDENCE_REQUEST_SCHEMA_VERSION:
        return None
    routing = _mapping(decision_request.get("routing_authority"))
    authorization = _mapping(decision_request.get("authorization"))
    if (
        routing.get("system") != "BlueprintCapturePipeline"
        or routing.get("webapp_backend_selection_allowed") is not False
        or authorization.get("access_state") != "provisioned"
        or expected_capture_root is None
    ):
        return None
    request_id = _string(decision_request.get("request_id") or payload.get("request_id"))
    decision_id = _string(decision_request.get("decision_id") or payload.get("decision_id"))
    site_task = _mapping(decision_request.get("site_task"))
    testbed = _mapping(decision_request.get("testbed"))
    candidates = decision_request.get("candidates")
    candidate = _mapping(candidates[0]) if isinstance(candidates, list) and candidates else {}
    candidate_reference = _mapping(candidate.get("reference"))
    task_id = _string(site_task.get("task_id"))
    site_slug = _string(site_task.get("site_name") or site_task.get("site_id"))
    if not all((request_id, decision_id, task_id, site_slug)) or not _safe_job_id(request_id):
        return None
    capture_id = expected_capture_root.name
    policy_id = _string(
        candidate.get("candidate_id")
        or candidate_reference.get("external_id")
        or "decision_evidence_candidate"
    )
    return {
        "schema_version": WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
        "job_id": request_id,
        "request_id": request_id,
        "buyer_request_id": decision_id,
        "requested_tasks": [{"task_id": task_id, "scenario_ids": []}],
        "site_package": {
            "capture_root": str(expected_capture_root),
            "capture_id": capture_id,
            "site_slug": site_slug,
            "site_submission_id": _string(site_task.get("site_id")),
            "buyer_request_id": decision_id,
            "capture_job_id": request_id,
            "package_uri": _string(testbed.get("manifest_uri")) or None,
        },
        "owner_system": {
            "name": "Blueprint-WebApp",
            "request_id": request_id,
            "buyer_request_id": decision_id,
            "site_submission_id": _string(site_task.get("site_id")),
            "capture_job_id": request_id,
            "capture_id": capture_id,
        },
        "policy": {"policy_id": policy_id},
        "source": {
            "system": "Blueprint-WebApp",
            "source_kind": "decision_evidence_request_legacy_execution_adapter",
            "selection_state": {
                "source_kind": "decision_evidence_request_legacy_execution_adapter",
                "task_id": task_id,
                "policy_id": policy_id,
            },
        },
        "decision_evidence_request": decision_request,
        "proof_boundary": {
            "translation_grants_method_qualification": False,
            "translation_proves_decision": False,
            "translation_proves_physical_success": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _request_from_payload(
    payload: Mapping[str, Any], *, expected_capture_root: Path | None = None
) -> Dict[str, Any] | None:
    decision_request = translate_decision_evidence_envelope_to_legacy_execution_request(
        payload, expected_capture_root=expected_capture_root
    )
    if decision_request is not None:
        return decision_request
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        request = payload.get("job_request")
        if isinstance(request, Mapping) and request.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
            return dict(request)
        return None
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return dict(payload)
    return None


def _source_kind_from_request(request: Mapping[str, Any]) -> str:
    source = _mapping(request.get("source"))
    selection = _mapping(source.get("selection_state"))
    for candidate in (request, source, selection):
        source_kind = _string(candidate.get("source_kind"))
        if source_kind:
            return source_kind
    return ""


def _field_value(request: Mapping[str, Any], field: str) -> str | None:
    source = _mapping(request.get("source"))
    source_selection = _mapping(source.get("selection_state"))
    owner_system = _mapping(request.get("owner_system"))
    site_package = _mapping(request.get("site_package"))
    for candidate in (request, source, source_selection, owner_system, site_package):
        value = _string(candidate.get(field))
        if value:
            return value
    return None


def _path_matches(value: str | None, expected: Path | None) -> bool:
    if not value or expected is None:
        return False
    try:
        return Path(value).resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def _load_control_plane_manifest(path: Path) -> Dict[str, Any]:
    manifest = _read_mapping(path)
    if manifest.get("schema_version") != LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION:
        raise ValueError(f"Expected {LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION} at {path}")
    return manifest


def _audit_webapp_request(
    *,
    request_path: Path | None,
    expected_capture_root: Path | None,
    configured_inbox: Path | None,
    allow_request_capture_root: bool = False,
) -> Dict[str, Any]:
    if request_path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "path": None,
            "blockers": ["webapp_job_request_not_provided"],
        }
    if not request_path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(request_path),
            "blockers": ["webapp_job_request_missing"],
        }
    try:
        payload = _read_mapping(request_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(request_path),
            "blockers": [f"webapp_job_request_read_failed:{type(exc).__name__}"],
        }
    request = _request_from_payload(
        payload, expected_capture_root=expected_capture_root
    )
    if request is None:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(request_path),
            "blockers": ["not_robot_eval_job_request_v1_or_queue_envelope"],
            "sha256": _sha_file(request_path),
        }
    source_kind = _string(payload.get("source_kind")) or _source_kind_from_request(request)
    local_rehearsal_only = (
        bool(payload.get("local_rehearsal_only"))
        or bool(request.get("local_rehearsal_only"))
        or source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    )
    site_package = _mapping(request.get("site_package"))
    request_capture_root = _string(site_package.get("capture_root")) or None
    request_capture_root_path = Path(request_capture_root).resolve() if request_capture_root else None
    request_capture_root_exists = (
        request_capture_root_path.is_dir() if request_capture_root_path else False
    )
    fields_present = {
        field: bool(_field_value(request, field))
        for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS
    }
    missing_fields = [
        field for field, present in fields_present.items() if not present
    ]
    capture_root_matches = _path_matches(request_capture_root, expected_capture_root)
    blockers: List[str] = []
    if missing_fields:
        blockers.append("missing_required_webapp_ids")
    capture_root_accepted_by_request = (
        allow_request_capture_root and request_capture_root_exists
    )
    if not capture_root_matches and not capture_root_accepted_by_request:
        blockers.append("request_capture_root_does_not_match_control_plane")
    job_id = _string(request.get("job_id")) or request_path.stem
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "path": str(request_path),
        "sha256": _sha_file(request_path),
        "job_id": job_id,
        "schema_version": request.get("schema_version"),
        "fields_present": fields_present,
        "missing_fields": missing_fields,
        "request_capture_root_configured": bool(request_capture_root),
        "request_capture_root": str(request_capture_root_path) if request_capture_root_path else None,
        "request_capture_root_exists": request_capture_root_exists,
        "request_capture_root_matches_control_plane": capture_root_matches,
        "request_capture_root_accepted_for_multi_capture_inbox": capture_root_accepted_by_request,
        "configured_capture_root": str(expected_capture_root) if expected_capture_root else None,
        "configured_inbox": str(configured_inbox) if configured_inbox else None,
        "source_kind": source_kind or None,
        "local_rehearsal_only": local_rehearsal_only,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Valid WebApp request metadata proves handoff shape only; local rehearsal "
            "requests are not live WebApp forwarding proof. The control plane still owns "
            "scheduling and proof-boundary enforcement."
        ),
    }


def _audit_arena_results(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "arena_results_dir": None,
            "blockers": ["arena_results_dir_not_provided"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    if not path.is_dir():
        return {
            "status": "blocked",
            "ready": False,
            "arena_results_dir": str(path),
            "blockers": ["arena_results_dir_missing"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    json_artifacts = sorted(item for item in path.rglob("*.json") if item.is_file())
    recognized_names = set(ARENA_RESULT_ARTIFACT_NAMES)
    recognized = [
        str(item.relative_to(path))
        for item in json_artifacts
        if item.name in recognized_names
    ]
    blockers: List[str] = []
    if not json_artifacts:
        blockers.append("arena_results_dir_has_no_json_artifacts")
    return {
        "status": "ready_for_ingest" if not blockers else "blocked",
        "ready": not blockers,
        "arena_results_dir": str(path),
        "blockers": blockers,
        "json_artifact_count": len(json_artifacts),
        "recognized_artifacts": recognized,
        "artifact_sample": [str(item.relative_to(path)) for item in json_artifacts[:20]],
        "truncated_artifact_sample": len(json_artifacts) > 20,
        "proof_boundary": (
            "Arena result artifacts are ingest inputs only; they are not simulator execution, "
            "robot policy, contact, safety, or readiness proof by themselves."
        ),
    }


def _evidence_job_id(payload: Mapping[str, Any]) -> str | None:
    for field in ("job_id", "jobId", "robot_eval_job_id", "robotEvalJobId"):
        value = _string(payload.get(field))
        if value:
            return value
    return None


def _records(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = payload.get("records") or payload.get("outcomes") or payload.get("observations")
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _record_id(record: Mapping[str, Any], *, prefix: str, index: int) -> str:
    return (
        _string(
            record.get("outcome_id")
            or record.get("outcomeId")
            or record.get("evidence_id")
            or record.get("evidenceId")
            or record.get("record_id")
            or record.get("recordId")
            or record.get("id")
        )
        or f"{prefix}_{index + 1:04d}"
    )


def _record_has_owner_evidence(record: Mapping[str, Any]) -> bool:
    evidence = (
        record.get("evidence_refs")
        or record.get("evidenceRefs")
        or record.get("owner_evidence_refs")
        or record.get("ownerEvidenceRefs")
        or record.get("operator_attestation")
        or record.get("operatorAttestation")
        or record.get("owner_attestation")
        or record.get("ownerAttestation")
    )
    if isinstance(evidence, Mapping):
        return bool(evidence)
    if isinstance(evidence, list):
        return bool(evidence)
    return bool(_string(evidence))


def _record_has_actual_result(record: Mapping[str, Any]) -> bool:
    for key in (
        "actual_success",
        "actualSuccess",
        "actual_result",
        "actualResult",
        "result",
        "outcome",
        "status",
        "failure_mode_ids",
        "failureModeIds",
    ):
        if key in record and record.get(key) not in (None, "", []):
            return True
    return False


def _audit_deployment_outcomes(
    *,
    path: Path | None,
    expected_job_id: str | None,
) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "path": None,
            "blockers": ["deployment_outcomes_not_provided"],
            "job_id": expected_job_id,
            "record_count": 0,
        }
    if not path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": ["deployment_outcomes_missing"],
            "job_id": expected_job_id,
            "record_count": 0,
        }
    try:
        payload = _read_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": [f"deployment_outcomes_read_failed:{type(exc).__name__}"],
            "sha256": _sha_file(path),
            "job_id": expected_job_id,
            "record_count": 0,
        }

    evidence_job_id = _evidence_job_id(payload)
    resolved_job_id = expected_job_id or evidence_job_id
    records = _records(payload)
    blockers: List[str] = []
    if payload.get("schema_version") != "deployment_outcome_manifest.v1":
        blockers.append("deployment_outcomes_schema_mismatch")
    if not resolved_job_id:
        blockers.append("deployment_outcomes_job_id_missing")
    elif not _safe_job_id(resolved_job_id):
        blockers.append("deployment_outcomes_job_id_unsafe")
    if expected_job_id and evidence_job_id and evidence_job_id != expected_job_id:
        blockers.append("deployment_outcomes_job_id_mismatch")
    if not records:
        blockers.append("deployment_outcomes_no_records")

    missing_task_or_scenario: List[str] = []
    missing_actual_signal: List[str] = []
    missing_prediction_keys: List[str] = []
    missing_owner_evidence: List[str] = []
    prediction_key_count = 0
    owner_evidence_count = 0
    for index, record in enumerate(records):
        record_id = _record_id(record, prefix="deployment_outcome", index=index)
        has_task = bool(_string(record.get("task_id") or record.get("taskId")))
        has_scenario = bool(_string(record.get("scenario_id") or record.get("scenarioId")))
        has_run = bool(_string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId")))
        has_variation = bool(
            _string(
                record.get("scenario_variation_instance_id")
                or record.get("scenarioVariationInstanceId")
            )
        )
        if not (has_task and has_scenario):
            missing_task_or_scenario.append(record_id)
        if not _record_has_actual_result(record):
            missing_actual_signal.append(record_id)
        if has_run and has_variation:
            prediction_key_count += 1
        else:
            missing_prediction_keys.append(record_id)
        if _record_has_owner_evidence(record):
            owner_evidence_count += 1
        else:
            missing_owner_evidence.append(record_id)

    if missing_task_or_scenario:
        blockers.append("deployment_outcomes_missing_task_or_scenario")
    if missing_actual_signal:
        blockers.append("deployment_outcomes_missing_actual_result_signal")

    ready = not blockers
    return {
        "status": "ready_for_real_world_validation" if ready else "blocked",
        "ready": ready,
        "path": str(path),
        "sha256": _sha_file(path),
        "schema_version": payload.get("schema_version"),
        "job_id": resolved_job_id,
        "evidence_job_id": evidence_job_id,
        "expected_job_id": expected_job_id,
        "record_count": len(records),
        "records_ready_for_calibration": bool(records) and not missing_prediction_keys,
        "prediction_match_keys_ready": bool(records) and not missing_prediction_keys,
        "prediction_match_key_record_count": prediction_key_count,
        "missing_prediction_match_key_record_ids": missing_prediction_keys,
        "owner_evidence_ready": bool(records) and not missing_owner_evidence,
        "owner_evidence_record_count": owner_evidence_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Deployment outcome intake validates owner records for later predicted-vs-actual "
            "joins; it does not prove calibration until the robot-eval job ingests them."
        ),
    }


def _real_pov_video(record: Mapping[str, Any]) -> str:
    return _string(
        record.get("robot_camera_video_uri")
        or record.get("robotCameraVideoUri")
        or record.get("camera_video_uri")
        or record.get("cameraVideoUri")
        or record.get("video_uri")
        or record.get("videoUri")
    )


def _real_pov_action_log(record: Mapping[str, Any]) -> str:
    return _string(
        record.get("action_log_uri")
        or record.get("actionLogUri")
        or record.get("actions_uri")
        or record.get("actionsUri")
        or record.get("action_trace_uri")
        or record.get("actionTraceUri")
    )


def _audit_real_robot_pov(
    *,
    path: Path | None,
    expected_job_id: str | None,
) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "path": None,
            "blockers": ["real_robot_pov_not_provided"],
            "job_id": expected_job_id,
            "record_count": 0,
        }
    if not path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": ["real_robot_pov_missing"],
            "job_id": expected_job_id,
            "record_count": 0,
        }
    try:
        payload = _read_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": [f"real_robot_pov_read_failed:{type(exc).__name__}"],
            "sha256": _sha_file(path),
            "job_id": expected_job_id,
            "record_count": 0,
        }

    evidence_job_id = _evidence_job_id(payload)
    resolved_job_id = expected_job_id or evidence_job_id
    records = _records(payload)
    blockers: List[str] = []
    if payload.get("schema_version") != "real_robot_pov_manifest.v1":
        blockers.append("real_robot_pov_schema_mismatch")
    if resolved_job_id and not _safe_job_id(resolved_job_id):
        blockers.append("real_robot_pov_job_id_unsafe")
    if expected_job_id and evidence_job_id and evidence_job_id != expected_job_id:
        blockers.append("real_robot_pov_job_id_mismatch")
    if not records:
        blockers.append("real_robot_pov_no_records")

    missing_exact_keys: List[str] = []
    missing_camera_video: List[str] = []
    missing_action_logs: List[str] = []
    missing_timestamp_alignment: List[str] = []
    missing_evidence: List[str] = []
    exact_key_count = 0
    camera_video_count = 0
    action_log_count = 0
    timestamp_alignment_count = 0
    evidence_count = 0
    for index, record in enumerate(records):
        record_id = _record_id(record, prefix="real_robot_pov", index=index)
        has_task = bool(_string(record.get("task_id") or record.get("taskId")))
        has_scenario = bool(_string(record.get("scenario_id") or record.get("scenarioId")))
        has_run = bool(_string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId")))
        has_variation = bool(
            _string(
                record.get("scenario_variation_instance_id")
                or record.get("scenarioVariationInstanceId")
            )
        )
        if has_task and has_scenario and has_run and has_variation:
            exact_key_count += 1
        else:
            missing_exact_keys.append(record_id)
        if _real_pov_video(record):
            camera_video_count += 1
        else:
            missing_camera_video.append(record_id)
        if _real_pov_action_log(record):
            action_log_count += 1
        else:
            missing_action_logs.append(record_id)
        if _string(record.get("timestamp_alignment") or record.get("timestampAlignment")):
            timestamp_alignment_count += 1
        else:
            missing_timestamp_alignment.append(record_id)
        if _record_has_owner_evidence(record):
            evidence_count += 1
        else:
            missing_evidence.append(record_id)

    if missing_exact_keys:
        blockers.append("real_robot_pov_missing_exact_keys")
    if missing_camera_video:
        blockers.append("real_robot_pov_missing_camera_video")
    if missing_action_logs:
        blockers.append("real_robot_pov_missing_action_logs")
    if missing_timestamp_alignment:
        blockers.append("real_robot_pov_missing_timestamp_alignment")
    if missing_evidence:
        blockers.append("real_robot_pov_missing_owner_evidence")

    ready = not blockers
    return {
        "status": "ready_for_robot_eval_job" if ready else "blocked",
        "ready": ready,
        "path": str(path),
        "sha256": _sha_file(path),
        "schema_version": payload.get("schema_version"),
        "job_id": resolved_job_id,
        "evidence_job_id": evidence_job_id,
        "expected_job_id": expected_job_id,
        "record_count": len(records),
        "exact_key_record_count": exact_key_count,
        "camera_video_record_count": camera_video_count,
        "action_log_record_count": action_log_count,
        "timestamp_alignment_record_count": timestamp_alignment_count,
        "evidence_record_count": evidence_count,
        "missing_exact_key_record_ids": missing_exact_keys,
        "missing_camera_video_record_ids": missing_camera_video,
        "missing_action_log_record_ids": missing_action_logs,
        "missing_timestamp_alignment_record_ids": missing_timestamp_alignment,
        "missing_evidence_record_ids": missing_evidence,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Real robot POV intake validates owner-supplied camera/action evidence pointers "
            "only; the job ingest and closure audit must join them to scenario runs."
        ),
    }


def _field(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _policy_modality_payload(policy_package: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    camel = {
        "policy_api_endpoint": "policyApiEndpoint",
        "docker_container": "dockerContainer",
        "recorded_action_trace": "recordedActionTrace",
        "high_level_skill_trace": "highLevelSkillTrace",
        "teleop_demo": "teleopDemo",
        "sim_controller_plugin": "simControllerPlugin",
    }[modality]
    return _mapping(policy_package.get(modality) or policy_package.get(camel))


def _policy_package_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    package = _mapping(payload.get("policy_package") or payload.get("policyPackage"))
    if package:
        return package
    return {
        modality: modality_payload
        for modality in POLICY_MODALITY_ORDER
        if (modality_payload := _policy_modality_payload(payload, modality))
    }


def _audit_policy_modality(*, modality: str, payload: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    if not payload:
        return missing
    if modality == "policy_api_endpoint":
        endpoint = _string(_field(payload, "endpoint_url", "endpointUrl", "url"))
        if not (endpoint.startswith("https://") or endpoint.startswith("http://")):
            missing.append("policy_package.policy_api_endpoint.endpoint_url")
    elif modality == "docker_container":
        if not _string(_field(payload, "image_ref", "imageRef")):
            missing.append("policy_package.docker_container.image_ref")
        digest = _string(_field(payload, "digest", "digestChecksum"))
        if not digest.startswith("sha256:"):
            missing.append("policy_package.docker_container.digest")
    elif modality == "recorded_action_trace":
        if not _string(_field(payload, "trace_manifest_uri", "traceManifestUri")):
            missing.append("policy_package.recorded_action_trace.trace_manifest_uri")
        if not _string(_field(payload, "timestamp_alignment", "timestampAlignment")):
            missing.append("policy_package.recorded_action_trace.timestamp_alignment")
    elif modality == "high_level_skill_trace":
        sequence = payload.get("ordered_skill_sequence") or payload.get("orderedSkillSequence")
        if not (isinstance(sequence, list) and sequence):
            missing.append("policy_package.high_level_skill_trace.ordered_skill_sequence")
    elif modality == "teleop_demo":
        if not _string(_field(payload, "demo_artifact_uri", "demoArtifactUri")):
            missing.append("policy_package.teleop_demo.demo_artifact_uri")
        if not _string(_field(payload, "rights_privacy_attestation", "rightsPrivacyAttestation")):
            missing.append("policy_package.teleop_demo.rights_privacy_attestation")
    elif modality == "sim_controller_plugin":
        if not _string(_field(payload, "simulator_framework", "simulatorFramework")):
            missing.append("policy_package.sim_controller_plugin.simulator_framework")
        if not _string(_field(payload, "plugin_uri", "pluginUri")):
            missing.append("policy_package.sim_controller_plugin.plugin_uri")
    return missing


def _audit_policy_package(
    *,
    path: Path | None,
    expected_job_id: str | None,
) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "path": None,
            "blockers": ["policy_package_not_provided"],
            "job_id": expected_job_id,
            "selected_modalities": [],
        }
    if not path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": ["policy_package_missing"],
            "job_id": expected_job_id,
            "selected_modalities": [],
        }
    try:
        payload = _read_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": [f"policy_package_read_failed:{type(exc).__name__}"],
            "sha256": _sha_file(path),
            "job_id": expected_job_id,
            "selected_modalities": [],
        }
    evidence_job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    ) or None
    resolved_job_id = expected_job_id or evidence_job_id
    package = _policy_package_from_payload(payload)
    blockers: List[str] = []
    if payload.get("schema_version") not in {
        "robot_team_policy_package.v1",
        "robot_eval_policy_package.v1",
        "robot_eval_job_request.v1",
    }:
        blockers.append("policy_package_schema_mismatch")
    if not resolved_job_id:
        blockers.append("policy_package_job_id_missing")
    elif not _safe_job_id(resolved_job_id):
        blockers.append("policy_package_job_id_unsafe")
    if expected_job_id and evidence_job_id and evidence_job_id != expected_job_id:
        blockers.append("policy_package_job_id_mismatch")
    selected_modalities: List[str] = []
    modality_missing: Dict[str, List[str]] = {}
    for modality in POLICY_MODALITY_ORDER:
        modality_payload = _policy_modality_payload(package, modality)
        if not modality_payload:
            continue
        selected_modalities.append(modality)
        missing = _audit_policy_modality(modality=modality, payload=modality_payload)
        if missing:
            modality_missing[modality] = missing
    if not selected_modalities:
        blockers.append("policy_package_no_supported_modality")
    for missing in modality_missing.values():
        blockers.extend(missing)
    return {
        "status": "ready_for_robot_eval_job" if not blockers else "blocked",
        "ready": not blockers,
        "path": str(path),
        "sha256": _sha_file(path),
        "schema_version": payload.get("schema_version"),
        "job_id": resolved_job_id,
        "evidence_job_id": evidence_job_id,
        "expected_job_id": expected_job_id,
        "selected_modalities": selected_modalities,
        "modality_missing_inputs": modality_missing,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Policy package intake validates robot-team references only; execution proof requires "
            "the gated robot-eval job policy execution bundle."
        ),
    }


def _audit_live_closure_evidence(
    *,
    path: Path | None,
    expected_job_id: str | None,
) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "path": None,
            "blockers": ["live_closure_evidence_not_provided"],
            "job_id": expected_job_id,
        }
    if not path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": ["live_closure_evidence_missing"],
            "job_id": expected_job_id,
        }
    try:
        payload = _read_mapping(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": [f"live_closure_evidence_read_failed:{type(exc).__name__}"],
            "sha256": _sha_file(path),
            "job_id": expected_job_id,
        }

    evidence_job_id = _evidence_job_id(payload)
    resolved_job_id = expected_job_id or evidence_job_id
    blockers: List[str] = []
    if payload.get("schema_version") != LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION:
        blockers.append("live_closure_evidence_schema_mismatch")
    if not resolved_job_id:
        blockers.append("live_closure_evidence_job_id_missing")
    elif not _safe_job_id(resolved_job_id):
        blockers.append("live_closure_evidence_job_id_unsafe")
    if expected_job_id and evidence_job_id and evidence_job_id != expected_job_id:
        blockers.append("live_closure_evidence_job_id_mismatch")

    delivery_access = _section(payload, "delivery_access", "deliveryAccess", "delivery")
    rights = _section(payload, "rights_privacy", "rightsPrivacy")
    webapp = _section(payload, "webapp_upstream", "webappUpstream")

    delivery_access_ready = _delivery_access_ready(delivery_access)
    rights_ready = (
        not rights
        or _status_ok(rights.get("status"))
        or _boolish(rights.get("accepted"))
        or _boolish(rights.get("external_use_allowed") or rights.get("externalUseAllowed"))
    )
    webapp_id_count = sum(
        1 for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS if _string(webapp.get(field))
    )

    if not delivery_access_ready:
        blockers.append("delivery_access_evidence_incomplete")
    if not rights_ready:
        blockers.append("rights_privacy_evidence_blocked")

    return {
        "status": "ready_for_closure_audit" if not blockers else "blocked",
        "ready": not blockers,
        "path": str(path),
        "sha256": _sha_file(path),
        "schema_version": payload.get("schema_version"),
        "job_id": resolved_job_id,
        "evidence_job_id": evidence_job_id,
        "expected_job_id": expected_job_id,
        "sections": {
            "delivery_access_ready": delivery_access_ready,
            "rights_privacy_ready": rights_ready,
            "webapp_upstream_id_count": webapp_id_count,
        },
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Closure evidence is staged for the deterministic live closure audit only; it does "
            "not prove package closure until live_eval_closure_manifest.json passes."
        ),
    }


def _stage_webapp_request(
    *,
    request_path: Path,
    audit: Mapping[str, Any],
    inbox: Path | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if not audit.get("ready"):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["webapp_request_not_ready_for_staging"],
        }
    if inbox is None:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": [f"missing_env_or_manifest_{JOB_REQUEST_INBOX_ENV}"],
        }
    job_id = _string(audit.get("job_id")) or request_path.stem
    target = inbox / f"{job_id}.json"
    ensure_dir(inbox)
    blockers: List[str] = []
    if target.exists() and not overwrite:
        blockers.append("target_request_already_exists")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": blockers,
        }
    payload = _read_mapping(request_path)
    if payload.get("queue_contract") == DECISION_EVIDENCE_QUEUE_CONTRACT:
        normalized_request = _request_from_payload(
            payload,
            expected_capture_root=Path(_string(audit.get("request_capture_root"))).resolve(),
        )
        if normalized_request is None:
            return {
                "status": "blocked",
                "performed": False,
                "target_path": str(target),
                "blockers": ["webapp_request_normalization_failed"],
            }
        staged_payload: Mapping[str, Any] = {
            "queue_contract": WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
            "source_kind": "decision_evidence_request_legacy_execution_adapter",
            "job_request": normalized_request,
        }
        write_json(target, staged_payload)
    else:
        shutil.copy2(request_path, target)
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target),
        "sha256": _sha_file(target),
        "blockers": [],
        "proof_boundary": "staging copies an input request only and does not process the job",
    }


def _stage_live_closure_evidence(
    *,
    evidence_path: Path,
    audit: Mapping[str, Any],
    capture_root: Path | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if not audit.get("ready"):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["live_closure_evidence_not_ready_for_staging"],
        }
    if capture_root is None:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["missing_control_plane_capture_root"],
        }
    job_id = _string(audit.get("job_id"))
    if not job_id:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["live_closure_evidence_job_id_missing"],
        }
    if not _safe_job_id(job_id):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["live_closure_evidence_job_id_unsafe"],
        }
    target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / LIVE_CLOSURE_EVIDENCE_ARTIFACT_NAME
    )
    ensure_dir(target.parent)
    blockers: List[str] = []
    if target.exists() and not overwrite:
        blockers.append("target_live_closure_evidence_already_exists")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": blockers,
        }
    shutil.copy2(evidence_path, target)
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target),
        "job_id": job_id,
        "sha256": _sha_file(target),
        "blockers": [],
        "proof_boundary": "staging copies closure evidence only and does not run the closure audit",
    }


def _stage_deployment_outcomes(
    *,
    outcome_path: Path,
    audit: Mapping[str, Any],
    capture_root: Path | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if not audit.get("ready"):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["deployment_outcomes_not_ready_for_staging"],
        }
    if capture_root is None:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["missing_control_plane_capture_root"],
        }
    job_id = _string(audit.get("job_id"))
    if not job_id:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["deployment_outcomes_job_id_missing"],
        }
    if not _safe_job_id(job_id):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["deployment_outcomes_job_id_unsafe"],
        }
    payload = _read_mapping(outcome_path)
    records = _records(payload)
    target_dir = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / "deployment_outcomes"
        / "inbox"
    )
    ensure_dir(target_dir)
    target_paths = [
        target_dir / f"{_record_id(record, prefix='deployment_outcome', index=index)}.json"
        for index, record in enumerate(records)
    ]
    existing = [str(path) for path in target_paths if path.exists()]
    if existing and not overwrite:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": existing[0],
            "blockers": ["target_deployment_outcome_already_exists"],
        }
    for index, record in enumerate(records):
        target = target_paths[index]
        write_json(
            target,
            {
                "schema_version": "deployment_outcome_record.v1",
                "job_id": job_id,
                "source_manifest_path": str(outcome_path),
                "record": record,
            },
        )
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target_paths[0]) if target_paths else None,
        "target_dir": str(target_dir),
        "job_id": job_id,
        "record_count": len(records),
        "sha256": _sha_file(target_paths[0]) if target_paths else None,
        "blockers": [],
        "proof_boundary": (
            "staging copies owner deployment outcome records only and does not run "
            "predicted-vs-actual calibration"
        ),
    }


def _stage_real_robot_pov(
    *,
    pov_path: Path,
    audit: Mapping[str, Any],
    capture_root: Path | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if not audit.get("ready"):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["real_robot_pov_not_ready_for_staging"],
        }
    if capture_root is None:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["missing_control_plane_capture_root"],
        }
    target = capture_root / "pipeline" / "robot_eval_inputs" / "real_robot_pov_manifest.json"
    ensure_dir(target.parent)
    if target.exists() and not overwrite:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": ["target_real_robot_pov_already_exists"],
        }
    shutil.copy2(pov_path, target)
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target),
        "job_id": audit.get("job_id"),
        "record_count": int(audit.get("record_count") or 0),
        "sha256": _sha_file(target),
        "blockers": [],
        "proof_boundary": "staging copies real robot POV evidence only and does not execute policy",
    }


def _stage_policy_package(
    *,
    policy_path: Path,
    audit: Mapping[str, Any],
    capture_root: Path | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if not audit.get("ready"):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["policy_package_not_ready_for_staging"],
        }
    if capture_root is None:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["missing_control_plane_capture_root"],
        }
    job_id = _string(audit.get("job_id"))
    if not job_id:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["policy_package_job_id_missing"],
        }
    if not _safe_job_id(job_id):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["policy_package_job_id_unsafe"],
        }
    target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / POLICY_PACKAGE_ARTIFACT_NAME
    )
    ensure_dir(target.parent)
    blockers: List[str] = []
    if target.exists() and not overwrite:
        blockers.append("target_policy_package_already_exists")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": blockers,
        }
    shutil.copy2(policy_path, target)
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target),
        "job_id": job_id,
        "selected_modalities": list(audit.get("selected_modalities") or []),
        "sha256": _sha_file(target),
        "blockers": [],
        "proof_boundary": "staging copies policy package references only and does not execute policy",
    }


def _write_staged_inputs(
    *,
    path: Path,
    manifest_path: Path,
    capture_root: Path | None,
    webapp_audit: Mapping[str, Any],
    webapp_staging: Mapping[str, Any],
    arena_audit: Mapping[str, Any],
    stage_arena_results: bool,
    live_closure_evidence_audit: Mapping[str, Any],
    live_closure_evidence_staging: Mapping[str, Any],
    stage_live_closure_evidence: bool,
    deployment_outcomes_audit: Mapping[str, Any],
    deployment_outcomes_staging: Mapping[str, Any],
    stage_deployment_outcomes: bool,
    policy_package_audit: Mapping[str, Any],
    policy_package_staging: Mapping[str, Any],
    stage_policy_package: bool,
    real_robot_pov_audit: Mapping[str, Any],
    real_robot_pov_staging: Mapping[str, Any],
    stage_real_robot_pov: bool,
) -> Dict[str, Any]:
    arena_ready = bool(arena_audit.get("ready"))
    webapp_ready = bool(webapp_audit.get("ready"))
    webapp_staged = bool(webapp_staging.get("performed"))
    closure_ready = bool(live_closure_evidence_audit.get("ready"))
    closure_staged = bool(live_closure_evidence_staging.get("performed"))
    outcomes_ready = bool(deployment_outcomes_audit.get("ready"))
    outcomes_staged = bool(deployment_outcomes_staging.get("performed"))
    policy_ready = bool(policy_package_audit.get("ready"))
    policy_staged = bool(policy_package_staging.get("performed"))
    real_pov_ready = bool(real_robot_pov_audit.get("ready"))
    real_pov_staged = bool(real_robot_pov_staging.get("performed"))
    blockers: List[str] = []
    if stage_arena_results and not arena_ready:
        blockers.append("arena_results_not_ready_for_staging")
    if stage_live_closure_evidence and not closure_ready:
        blockers.append("live_closure_evidence_not_ready_for_staging")
    if stage_deployment_outcomes and not outcomes_ready:
        blockers.append("deployment_outcomes_not_ready_for_staging")
    if stage_policy_package and not policy_ready:
        blockers.append("policy_package_not_ready_for_staging")
    if stage_real_robot_pov and not real_pov_ready:
        blockers.append("real_robot_pov_not_ready_for_staging")
    if webapp_staging.get("status") == "blocked":
        blockers.append("webapp_request_not_staged")
    if live_closure_evidence_staging.get("status") == "blocked":
        blockers.append("live_closure_evidence_not_staged")
    if deployment_outcomes_staging.get("status") == "blocked":
        blockers.append("deployment_outcomes_not_staged")
    if policy_package_staging.get("status") == "blocked":
        blockers.append("policy_package_not_staged")
    if real_robot_pov_staging.get("status") == "blocked":
        blockers.append("real_robot_pov_not_staged")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "path": str(path),
            "blockers": blockers,
        }
    if (
        not stage_arena_results
        and not webapp_staged
        and not closure_staged
        and not outcomes_staged
        and not policy_staged
        and not real_pov_staged
    ):
        return {
            "status": "not_requested",
            "performed": False,
            "path": str(path),
            "blockers": [],
        }
    payload = {
        "schema_version": LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_intake_manifest_path": str(manifest_path),
        "configured_capture_root": str(capture_root) if capture_root else None,
        "source_kind": webapp_audit.get("source_kind"),
        "local_rehearsal_only": bool(webapp_audit.get("local_rehearsal_only")),
        "webapp_request": {
            "ready": webapp_ready and webapp_staged,
            "staged": webapp_staged,
            "job_id": webapp_audit.get("job_id"),
            "path": webapp_audit.get("path"),
            "target_path": webapp_staging.get("target_path"),
            "sha256": webapp_staging.get("sha256") or webapp_audit.get("sha256"),
            "source_kind": webapp_audit.get("source_kind"),
            "local_rehearsal_only": bool(webapp_audit.get("local_rehearsal_only")),
        },
        "arena_results": {
            "ready": arena_ready if stage_arena_results else False,
            "arena_results_dir": arena_audit.get("arena_results_dir")
            if stage_arena_results
            else None,
            "json_artifact_count": arena_audit.get("json_artifact_count", 0)
            if stage_arena_results
            else 0,
            "recognized_artifacts": arena_audit.get("recognized_artifacts", [])
            if stage_arena_results
            else [],
        },
        "live_closure_evidence": {
            "ready": closure_ready and closure_staged,
            "staged": closure_staged,
            "job_id": live_closure_evidence_audit.get("job_id")
            or live_closure_evidence_staging.get("job_id"),
            "path": live_closure_evidence_audit.get("path"),
            "target_path": live_closure_evidence_staging.get("target_path"),
            "sha256": live_closure_evidence_staging.get("sha256")
            or live_closure_evidence_audit.get("sha256"),
            "sections": live_closure_evidence_audit.get("sections", {}),
        },
        "deployment_outcomes": {
            "ready": outcomes_ready and outcomes_staged,
            "staged": outcomes_staged,
            "job_id": deployment_outcomes_audit.get("job_id")
            or deployment_outcomes_staging.get("job_id"),
            "path": deployment_outcomes_audit.get("path"),
            "target_path": deployment_outcomes_staging.get("target_path"),
            "target_dir": deployment_outcomes_staging.get("target_dir"),
            "sha256": deployment_outcomes_staging.get("sha256")
            or deployment_outcomes_audit.get("sha256"),
            "record_count": deployment_outcomes_audit.get("record_count", 0),
            "records_ready_for_calibration": bool(
                deployment_outcomes_audit.get("records_ready_for_calibration")
            ),
            "prediction_match_keys_ready": bool(
                deployment_outcomes_audit.get("prediction_match_keys_ready")
            ),
            "prediction_match_key_record_count": deployment_outcomes_audit.get(
                "prediction_match_key_record_count",
                0,
            ),
            "missing_prediction_match_key_record_ids": deployment_outcomes_audit.get(
                "missing_prediction_match_key_record_ids",
                [],
            ),
            "owner_evidence_ready": bool(deployment_outcomes_audit.get("owner_evidence_ready")),
            "owner_evidence_record_count": deployment_outcomes_audit.get(
                "owner_evidence_record_count",
                0,
            ),
            "missing_owner_evidence_record_ids": deployment_outcomes_audit.get(
                "missing_owner_evidence_record_ids",
                [],
            ),
        },
        "policy_package": {
            "ready": policy_ready and policy_staged,
            "staged": policy_staged,
            "job_id": policy_package_audit.get("job_id") or policy_package_staging.get("job_id"),
            "path": policy_package_audit.get("path"),
            "target_path": policy_package_staging.get("target_path"),
            "sha256": policy_package_staging.get("sha256") or policy_package_audit.get("sha256"),
            "selected_modalities": policy_package_audit.get("selected_modalities", []),
        },
        "real_robot_pov": {
            "ready": real_pov_ready and real_pov_staged,
            "staged": real_pov_staged,
            "job_id": real_robot_pov_audit.get("job_id")
            or real_robot_pov_staging.get("job_id"),
            "path": real_robot_pov_audit.get("path"),
            "target_path": real_robot_pov_staging.get("target_path"),
            "sha256": real_robot_pov_staging.get("sha256") or real_robot_pov_audit.get("sha256"),
            "record_count": real_robot_pov_audit.get("record_count", 0),
            "exact_key_record_count": real_robot_pov_audit.get("exact_key_record_count", 0),
            "camera_video_record_count": real_robot_pov_audit.get(
                "camera_video_record_count",
                0,
            ),
            "action_log_record_count": real_robot_pov_audit.get("action_log_record_count", 0),
            "timestamp_alignment_record_count": real_robot_pov_audit.get(
                "timestamp_alignment_record_count",
                0,
            ),
            "evidence_record_count": real_robot_pov_audit.get("evidence_record_count", 0),
            "missing_exact_key_record_ids": real_robot_pov_audit.get(
                "missing_exact_key_record_ids",
                [],
            ),
            "missing_evidence_record_ids": real_robot_pov_audit.get(
                "missing_evidence_record_ids",
                [],
            ),
        },
        "proof_boundary": {
            "staged_inputs_are_pointers_only": True,
            "local_webapp_rehearsal_only": bool(webapp_audit.get("local_rehearsal_only")),
            "live_webapp_forwarding_proven": False,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "policy_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    ensure_dir(path.parent)
    write_json(path, payload)
    return {
        "status": "staged",
        "performed": True,
        "path": str(path),
        "blockers": [],
        "arena_results_staged": bool(stage_arena_results and arena_ready),
        "webapp_request_staged": webapp_staged,
        "live_closure_evidence_staged": closure_staged,
        "deployment_outcomes_staged": outcomes_staged,
        "policy_package_staged": policy_staged,
        "real_robot_pov_staged": real_pov_staged,
    }


def build_live_pipeline_input_intake(
    *,
    manifest_path: str | Path,
    webapp_job_request: str | Path | None = None,
    arena_results_dir: str | Path | None = None,
    live_closure_evidence: str | Path | None = None,
    deployment_outcomes: str | Path | None = None,
    policy_package: str | Path | None = None,
    real_robot_pov: str | Path | None = None,
    stage_webapp_request: bool = False,
    stage_arena_results: bool = False,
    stage_live_closure_evidence: bool = False,
    stage_deployment_outcomes: bool = False,
    stage_policy_package: bool = False,
    stage_real_robot_pov: bool = False,
    overwrite: bool = False,
    allow_request_capture_root: bool = False,
    output_path: str | Path | None = None,
    staged_inputs_path: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = _load_control_plane_manifest(resolved_manifest_path)
    capture_root = Path(manifest["capture_root"]).resolve() if manifest.get("capture_root") else None
    inbox = Path(manifest["job_request_inbox"]).resolve() if manifest.get("job_request_inbox") else None
    request_path = Path(webapp_job_request).resolve() if webapp_job_request else None
    results_path = Path(arena_results_dir).resolve() if arena_results_dir else None
    live_closure_evidence_path = (
        Path(live_closure_evidence).resolve() if live_closure_evidence else None
    )
    deployment_outcomes_path = Path(deployment_outcomes).resolve() if deployment_outcomes else None
    policy_package_path = Path(policy_package).resolve() if policy_package else None
    real_robot_pov_path = Path(real_robot_pov).resolve() if real_robot_pov else None
    generated_at = utc_now_iso()

    webapp_audit = _audit_webapp_request(
        request_path=request_path,
        expected_capture_root=capture_root,
        configured_inbox=inbox,
        allow_request_capture_root=allow_request_capture_root,
    )
    arena_audit = _audit_arena_results(results_path)
    live_closure_evidence_audit = _audit_live_closure_evidence(
        path=live_closure_evidence_path,
        expected_job_id=_string(webapp_audit.get("job_id")) or None,
    )
    deployment_outcomes_audit = _audit_deployment_outcomes(
        path=deployment_outcomes_path,
        expected_job_id=_string(webapp_audit.get("job_id")) or None,
    )
    policy_package_audit = _audit_policy_package(
        path=policy_package_path,
        expected_job_id=_string(webapp_audit.get("job_id")) or None,
    )
    real_robot_pov_audit = _audit_real_robot_pov(
        path=real_robot_pov_path,
        expected_job_id=_string(webapp_audit.get("job_id")) or None,
    )
    staging = (
        _stage_webapp_request(
            request_path=request_path or Path(),
            audit=webapp_audit,
            inbox=inbox,
            overwrite=overwrite,
        )
        if stage_webapp_request
        else {
            "status": "not_requested",
            "performed": False,
            "blockers": [],
        }
    )
    live_closure_staging = (
        _stage_live_closure_evidence(
            evidence_path=live_closure_evidence_path or Path(),
            audit=live_closure_evidence_audit,
            capture_root=capture_root,
            overwrite=overwrite,
        )
        if stage_live_closure_evidence
        else {
            "status": "not_requested",
            "performed": False,
            "blockers": [],
        }
    )
    deployment_outcomes_staging = (
        _stage_deployment_outcomes(
            outcome_path=deployment_outcomes_path or Path(),
            audit=deployment_outcomes_audit,
            capture_root=capture_root,
            overwrite=overwrite,
        )
        if stage_deployment_outcomes
        else {
            "status": "not_requested",
            "performed": False,
            "blockers": [],
        }
    )
    policy_package_staging = (
        _stage_policy_package(
            policy_path=policy_package_path or Path(),
            audit=policy_package_audit,
            capture_root=capture_root,
            overwrite=overwrite,
        )
        if stage_policy_package
        else {
            "status": "not_requested",
            "performed": False,
            "blockers": [],
        }
    )
    real_robot_pov_staging = (
        _stage_real_robot_pov(
            pov_path=real_robot_pov_path or Path(),
            audit=real_robot_pov_audit,
            capture_root=capture_root,
            overwrite=overwrite,
        )
        if stage_real_robot_pov
        else {
            "status": "not_requested",
            "performed": False,
            "blockers": [],
        }
    )
    input_blockers: List[str] = []
    if webapp_job_request and not webapp_audit.get("ready"):
        input_blockers.extend(f"webapp:{blocker}" for blocker in webapp_audit.get("blockers", []))
    if arena_results_dir and not arena_audit.get("ready"):
        input_blockers.extend(f"arena:{blocker}" for blocker in arena_audit.get("blockers", []))
    if live_closure_evidence and not live_closure_evidence_audit.get("ready"):
        input_blockers.extend(
            f"live_closure_evidence:{blocker}"
            for blocker in live_closure_evidence_audit.get("blockers", [])
        )
    if deployment_outcomes and not deployment_outcomes_audit.get("ready"):
        input_blockers.extend(
            f"deployment_outcomes:{blocker}"
            for blocker in deployment_outcomes_audit.get("blockers", [])
        )
    if policy_package and not policy_package_audit.get("ready"):
        input_blockers.extend(
            f"policy_package:{blocker}" for blocker in policy_package_audit.get("blockers", [])
        )
    if real_robot_pov and not real_robot_pov_audit.get("ready"):
        input_blockers.extend(
            f"real_robot_pov:{blocker}"
            for blocker in real_robot_pov_audit.get("blockers", [])
        )
    if staging.get("blockers"):
        input_blockers.extend(f"staging:{blocker}" for blocker in staging.get("blockers", []))
    if live_closure_staging.get("blockers"):
        input_blockers.extend(
            f"staging:{blocker}" for blocker in live_closure_staging.get("blockers", [])
        )
    if deployment_outcomes_staging.get("blockers"):
        input_blockers.extend(
            f"staging:{blocker}" for blocker in deployment_outcomes_staging.get("blockers", [])
        )
    if policy_package_staging.get("blockers"):
        input_blockers.extend(
            f"staging:{blocker}" for blocker in policy_package_staging.get("blockers", [])
        )
    if real_robot_pov_staging.get("blockers"):
        input_blockers.extend(
            f"staging:{blocker}" for blocker in real_robot_pov_staging.get("blockers", [])
        )

    status = "ready_for_control_plane"
    if input_blockers:
        status = "blocked"
    elif (
        not webapp_job_request
        and not arena_results_dir
        and not live_closure_evidence
        and not deployment_outcomes
        and not policy_package
        and not real_robot_pov
    ):
        status = "waiting_for_inputs"
    elif stage_webapp_request and staging.get("performed"):
        status = "staged_for_control_plane"
    elif stage_arena_results and arena_audit.get("ready"):
        status = "staged_for_control_plane"
    elif stage_live_closure_evidence and live_closure_staging.get("performed"):
        status = "staged_for_control_plane"
    elif stage_deployment_outcomes and deployment_outcomes_staging.get("performed"):
        status = "staged_for_control_plane"
    elif stage_policy_package and policy_package_staging.get("performed"):
        status = "staged_for_control_plane"
    elif stage_real_robot_pov and real_robot_pov_staging.get("performed"):
        status = "staged_for_control_plane"

    webapp_request_metadata_valid = bool(webapp_audit.get("ready"))
    local_webapp_rehearsal_only = bool(webapp_audit.get("local_rehearsal_only"))
    webapp_truth_proven = webapp_request_metadata_valid and not local_webapp_rehearsal_only
    if output_path:
        path = Path(output_path).resolve()
    else:
        path = resolved_manifest_path.parent / "live_pipeline_input_intake_audit.json"
    staged_path = (
        Path(staged_inputs_path).resolve()
        if staged_inputs_path
        else resolved_manifest_path.parent / "live_pipeline_staged_inputs.json"
    )
    staged_inputs = _write_staged_inputs(
        path=staged_path,
        manifest_path=path,
        capture_root=capture_root,
        webapp_audit=webapp_audit,
        webapp_staging=staging,
        arena_audit=arena_audit,
        stage_arena_results=stage_arena_results,
        live_closure_evidence_audit=live_closure_evidence_audit,
        live_closure_evidence_staging=live_closure_staging,
        stage_live_closure_evidence=stage_live_closure_evidence,
        deployment_outcomes_audit=deployment_outcomes_audit,
        deployment_outcomes_staging=deployment_outcomes_staging,
        stage_deployment_outcomes=stage_deployment_outcomes,
        policy_package_audit=policy_package_audit,
        policy_package_staging=policy_package_staging,
        stage_policy_package=stage_policy_package,
        real_robot_pov_audit=real_robot_pov_audit,
        real_robot_pov_staging=real_robot_pov_staging,
        stage_real_robot_pov=stage_real_robot_pov,
    )
    if staged_inputs.get("blockers"):
        input_blockers.extend(
            f"staged_inputs:{blocker}" for blocker in staged_inputs.get("blockers", [])
        )
        status = "blocked"

    intake = {
        "schema_version": LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "manifest_path": str(resolved_manifest_path),
        "configured_capture_root": str(capture_root) if capture_root else None,
        "configured_job_request_inbox": str(inbox) if inbox else None,
        "webapp_request_metadata_valid": webapp_request_metadata_valid,
        "local_webapp_rehearsal_only": local_webapp_rehearsal_only,
        "webapp_truth_proven": webapp_truth_proven,
        "webapp_job_request": webapp_audit,
        "arena_results": arena_audit,
        "live_closure_evidence": live_closure_evidence_audit,
        "deployment_outcomes": deployment_outcomes_audit,
        "policy_package": policy_package_audit,
        "real_robot_pov": real_robot_pov_audit,
        "webapp_staging": staging,
        "live_closure_evidence_staging": live_closure_staging,
        "deployment_outcomes_staging": deployment_outcomes_staging,
        "policy_package_staging": policy_package_staging,
        "real_robot_pov_staging": real_robot_pov_staging,
        "staged_inputs": staged_inputs,
        "input_blockers": input_blockers,
        "next_steps": [
            "Run blueprint-run-live-pipeline-control-plane after staging a WebApp request.",
            "Run blueprint-run-live-pipeline-control-plane after staging owner Arena artifacts.",
            "Run blueprint-audit-live-pipeline-proof-boundary after the control-plane pass.",
        ],
        "proof_boundary": {
            "intake_performs_live_actions": False,
            "webapp_request_metadata_valid": webapp_request_metadata_valid,
            "webapp_truth_proven": webapp_truth_proven,
            "local_webapp_rehearsal_only": local_webapp_rehearsal_only,
            "live_webapp_forwarding_proven": False,
            "arena_results_ready_for_ingest": bool(arena_audit.get("ready")),
            "live_closure_evidence_ready_for_closure_audit": bool(
                live_closure_evidence_audit.get("ready")
            ),
            "deployment_outcomes_ready_for_real_world_validation": bool(
                deployment_outcomes_audit.get("ready")
            ),
            "policy_package_ready_for_robot_eval_job": bool(policy_package_audit.get("ready")),
            "real_robot_pov_ready_for_job_ingest": bool(
                real_robot_pov_audit.get("ready")
            ),
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "robot_pov_evidence_proven": False,
            "real_world_outcome_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    ensure_dir(path.parent)
    intake["output_path"] = str(path)
    write_json(path, intake)
    return intake


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and optionally stage live WebApp/Arena inputs for the control plane."
    )
    parser.add_argument(
        "--manifest-path",
        default="/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json",
    )
    parser.add_argument("--webapp-job-request")
    parser.add_argument("--arena-results-dir")
    parser.add_argument("--live-closure-evidence")
    parser.add_argument("--deployment-outcomes")
    parser.add_argument("--policy-package")
    parser.add_argument("--real-robot-pov")
    parser.add_argument("--stage-webapp-request", action="store_true")
    parser.add_argument("--stage-arena-results", action="store_true")
    parser.add_argument("--stage-live-closure-evidence", action="store_true")
    parser.add_argument("--stage-deployment-outcomes", action="store_true")
    parser.add_argument("--stage-policy-package", action="store_true")
    parser.add_argument("--stage-real-robot-pov", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-path")
    parser.add_argument("--staged-inputs-path")
    args = parser.parse_args(argv)
    result = build_live_pipeline_input_intake(
        manifest_path=args.manifest_path,
        webapp_job_request=args.webapp_job_request,
        arena_results_dir=args.arena_results_dir,
        live_closure_evidence=args.live_closure_evidence,
        deployment_outcomes=args.deployment_outcomes,
        policy_package=args.policy_package,
        real_robot_pov=args.real_robot_pov,
        stage_webapp_request=args.stage_webapp_request,
        stage_arena_results=args.stage_arena_results,
        stage_live_closure_evidence=args.stage_live_closure_evidence,
        stage_deployment_outcomes=args.stage_deployment_outcomes,
        stage_policy_package=args.stage_policy_package,
        stage_real_robot_pov=args.stage_real_robot_pov,
        overwrite=args.overwrite,
        output_path=args.output_path,
        staged_inputs_path=args.staged_inputs_path,
    )
    print(f"[live-pipeline-input-intake] audit={result['output_path']}")
    print(f"[live-pipeline-input-intake] status={result['status']}")
    print(
        "[live-pipeline-input-intake] webapp_request_metadata_valid="
        f"{str(bool(result.get('webapp_request_metadata_valid'))).lower()}"
    )
    print(
        "[live-pipeline-input-intake] local_webapp_rehearsal_only="
        f"{str(bool(result.get('local_webapp_rehearsal_only'))).lower()}"
    )
    print(
        "[live-pipeline-input-intake] webapp_truth_proven="
        f"{str(bool(result.get('webapp_truth_proven'))).lower()}"
    )
    if result["input_blockers"]:
        print(f"[live-pipeline-input-intake] blockers={len(result['input_blockers'])}")
    return 0 if not result["input_blockers"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
