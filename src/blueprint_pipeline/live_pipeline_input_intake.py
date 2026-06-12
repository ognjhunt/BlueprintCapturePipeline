"""Validate and optionally stage live external inputs for the control plane.

The intake command is a preflight for real external handoffs. It can inspect a
WebApp ``robot_eval_job_request.v1`` file and an owner-system Arena result
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
DEPLOYMENT_OUTCOME_ARTIFACT_NAME = "deployment_outcome.json"
POLICY_PACKAGE_ARTIFACT_NAME = "policy_package.json"
REAL_ROBOT_POV_ARTIFACT_NAME = "real_robot_pov_manifest.json"
LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND = "local_first_gpu_rehearsal_request"
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


def _request_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any] | None:
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
    request = _request_from_payload(payload)
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
    if not capture_root_matches:
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
        "request_capture_root_matches_control_plane": capture_root_matches,
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


def _records_from_outcome_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in (
        "records",
        "outcomes",
        "actual_outcomes",
        "actualOutcomes",
        "deployment_outcomes",
        "deploymentOutcomes",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    if _string(payload.get("task_id") or payload.get("taskId")) or _string(
        payload.get("scenario_id") or payload.get("scenarioId")
    ):
        return [dict(payload)]
    return []


def _outcome_job_id(payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str | None:
    for field in ("job_id", "jobId", "robot_eval_job_id", "robotEvalJobId"):
        value = _string(payload.get(field))
        if value:
            return value
    record_job_ids = {
        value
        for record in records
        if (
            value := _string(
                record.get("job_id")
                or record.get("jobId")
                or record.get("robot_eval_job_id")
                or record.get("robotEvalJobId")
            )
        )
    }
    if len(record_job_ids) == 1:
        return next(iter(record_job_ids))
    return None


def _actual_signal_present(record: Mapping[str, Any]) -> bool:
    for key in (
        "actual_success",
        "actualSuccess",
        "success",
        "passed",
        "actual_status",
        "actualStatus",
        "status",
    ):
        if record.get(key) is not None and _string(record.get(key)) != "":
            return True
    for key in ("failure_mode_ids", "actual_failures", "actualFailures", "failures"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _prediction_match_key_present(record: Mapping[str, Any]) -> bool:
    return bool(
        _string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId"))
        and _string(
            record.get("scenario_variation_instance_id")
            or record.get("scenarioVariationInstanceId")
        )
    )


def _record_id(record: Mapping[str, Any], index: int) -> str:
    value = _string(record.get("outcome_id") or record.get("record_id") or record.get("id"))
    return value or f"deployment-outcome-{index:04d}"


def _outcome_owner_evidence_present(record: Mapping[str, Any]) -> bool:
    evidence_refs = _mapping(
        record.get("evidence_refs")
        or record.get("evidenceRefs")
        or record.get("owner_evidence_refs")
        or record.get("ownerEvidenceRefs")
    )
    owner_evidence_uri = _string(
        record.get("evidence_uri")
        or record.get("evidenceUri")
        or record.get("pilot_log_uri")
        or record.get("pilotLogUri")
        or record.get("owner_system_proof_uri")
        or record.get("ownerSystemProofUri")
    )
    attestation = (
        record.get("operator_attestation")
        or record.get("operatorAttestation")
        or record.get("owner_attestation")
        or record.get("ownerAttestation")
    )
    return bool(evidence_refs or owner_evidence_uri or _attestation_ok(attestation))


def _records_from_real_robot_pov_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in (
        "records",
        "robot_pov_records",
        "robotPovRecords",
        "evidence_records",
        "evidenceRecords",
        "observations",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    if _string(
        payload.get("robot_camera_video_uri")
        or payload.get("robotCameraVideoUri")
        or payload.get("action_log_uri")
        or payload.get("actionLogUri")
    ):
        return [dict(payload)]
    return []


def _real_robot_pov_job_id(
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> str | None:
    for field in ("job_id", "jobId", "robot_eval_job_id", "robotEvalJobId"):
        value = _string(payload.get(field))
        if value:
            return value
    record_job_ids = {
        value
        for record in records
        if (
            value := _string(
                record.get("job_id")
                or record.get("jobId")
                or record.get("robot_eval_job_id")
                or record.get("robotEvalJobId")
            )
        )
    }
    if len(record_job_ids) == 1:
        return next(iter(record_job_ids))
    return None


def _pov_record_id(record: Mapping[str, Any], index: int) -> str:
    value = _string(
        record.get("evidence_id")
        or record.get("evidenceId")
        or record.get("record_id")
        or record.get("recordId")
        or record.get("id")
    )
    return value or f"real-robot-pov-{index:04d}"


def _pov_exact_key_present(record: Mapping[str, Any]) -> bool:
    return bool(
        _string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId"))
        and _string(
            record.get("scenario_variation_instance_id")
            or record.get("scenarioVariationInstanceId")
        )
    )


def _pov_camera_video_present(record: Mapping[str, Any]) -> bool:
    return bool(
        _string(
            record.get("robot_camera_video_uri")
            or record.get("robotCameraVideoUri")
            or record.get("pov_video_uri")
            or record.get("povVideoUri")
            or record.get("camera_video_uri")
            or record.get("cameraVideoUri")
            or record.get("video_uri")
            or record.get("videoUri")
        )
    )


def _pov_action_log_present(record: Mapping[str, Any]) -> bool:
    return bool(
        _string(
            record.get("action_log_uri")
            or record.get("actionLogUri")
            or record.get("recorded_action_trace_uri")
            or record.get("recordedActionTraceUri")
            or record.get("action_trace_uri")
            or record.get("actionTraceUri")
        )
    )


def _pov_timestamp_alignment_present(
    *,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> bool:
    return bool(
        _string(
            record.get("timestamp_alignment")
            or record.get("timestampAlignment")
            or payload.get("timestamp_alignment")
            or payload.get("timestampAlignment")
        )
    )


def _pov_owner_evidence_present(record: Mapping[str, Any]) -> bool:
    evidence_refs = _mapping(
        record.get("owner_evidence_refs")
        or record.get("ownerEvidenceRefs")
        or record.get("evidence_refs")
        or record.get("evidenceRefs")
    )
    evidence_uri = _string(
        record.get("owner_evidence_uri")
        or record.get("ownerEvidenceUri")
        or record.get("evidence_uri")
        or record.get("evidenceUri")
    )
    attestation = (
        record.get("operator_attestation")
        or record.get("operatorAttestation")
        or record.get("owner_attestation")
        or record.get("ownerAttestation")
    )
    return bool(evidence_refs or evidence_uri or _attestation_ok(attestation))


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
            "record_ids": [],
        }
    if not path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": ["real_robot_pov_missing"],
            "job_id": expected_job_id,
            "record_count": 0,
            "record_ids": [],
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
            "record_ids": [],
        }
    records = _records_from_real_robot_pov_payload(payload)
    evidence_job_id = _real_robot_pov_job_id(payload, records)
    resolved_job_id = expected_job_id or evidence_job_id
    blockers: List[str] = []
    if payload.get("schema_version") not in {
        "real_robot_pov_manifest.v1",
        "robot_pov_evidence_manifest.v1",
        "real_robot_pov_evidence_manifest.v1",
    }:
        blockers.append("real_robot_pov_schema_mismatch")
    if evidence_job_id and not _safe_job_id(evidence_job_id):
        blockers.append("real_robot_pov_job_id_unsafe")
    if expected_job_id and evidence_job_id and evidence_job_id != expected_job_id:
        blockers.append("real_robot_pov_job_id_mismatch")
    if not records:
        blockers.append("real_robot_pov_empty")

    missing_exact_keys = [
        _pov_record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _pov_exact_key_present(record)
    ]
    missing_camera_video = [
        _pov_record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _pov_camera_video_present(record)
    ]
    missing_action_logs = [
        _pov_record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _pov_action_log_present(record)
    ]
    missing_timestamp_alignment = [
        _pov_record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _pov_timestamp_alignment_present(record=record, payload=payload)
    ]
    missing_evidence = [
        _pov_record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _pov_owner_evidence_present(record)
    ]
    if missing_exact_keys:
        blockers.append("real_robot_pov_missing_exact_keys")
    if missing_camera_video:
        blockers.append("real_robot_pov_missing_camera_videos")
    if missing_action_logs:
        blockers.append("real_robot_pov_missing_action_logs")
    if missing_timestamp_alignment:
        blockers.append("real_robot_pov_missing_timestamp_alignment")
    if missing_evidence:
        blockers.append("real_robot_pov_missing_owner_evidence")

    exact_key_record_count = len(records) - len(missing_exact_keys)
    camera_video_record_count = len(records) - len(missing_camera_video)
    action_log_record_count = len(records) - len(missing_action_logs)
    timestamp_alignment_record_count = len(records) - len(missing_timestamp_alignment)
    evidence_record_count = len(records) - len(missing_evidence)
    return {
        "status": "ready_for_robot_eval_job" if not blockers else "blocked",
        "ready": not blockers,
        "path": str(path),
        "sha256": _sha_file(path),
        "schema_version": payload.get("schema_version"),
        "job_id": resolved_job_id,
        "evidence_job_id": evidence_job_id,
        "expected_job_id": expected_job_id,
        "record_count": len(records),
        "record_ids": [
            _pov_record_id(record, index)
            for index, record in enumerate(records, start=1)
        ],
        "exact_key_record_count": exact_key_record_count,
        "camera_video_record_count": camera_video_record_count,
        "action_log_record_count": action_log_record_count,
        "timestamp_alignment_record_count": timestamp_alignment_record_count,
        "evidence_record_count": evidence_record_count,
        "missing_exact_key_record_ids": missing_exact_keys,
        "missing_camera_video_record_ids": missing_camera_video,
        "missing_action_log_record_ids": missing_action_logs,
        "missing_timestamp_alignment_record_ids": missing_timestamp_alignment,
        "missing_evidence_record_ids": missing_evidence,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Real robot POV intake validates owner-supplied camera/action evidence references only; "
            "the job must ingest the manifest before robot POV proof is allowed."
        ),
    }


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
    records = _records_from_outcome_payload(payload)
    evidence_job_id = _outcome_job_id(payload, records)
    resolved_job_id = expected_job_id or evidence_job_id
    blockers: List[str] = []
    if payload.get("schema_version") not in {
        "deployment_outcome.v1",
        "deployment_outcome_manifest.v1",
        "actual_outcome_manifest.v1",
        "deployment_outcome_inbox.v1",
    }:
        blockers.append("deployment_outcomes_schema_mismatch")
    if not records:
        blockers.append("deployment_outcomes_empty")
    if not resolved_job_id:
        blockers.append("deployment_outcomes_job_id_missing")
    elif not _safe_job_id(resolved_job_id):
        blockers.append("deployment_outcomes_job_id_unsafe")
    if expected_job_id and evidence_job_id and evidence_job_id != expected_job_id:
        blockers.append("deployment_outcomes_job_id_mismatch")
    missing_task_or_scenario = [
        _record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _string(record.get("task_id") or record.get("taskId"))
        or not _string(record.get("scenario_id") or record.get("scenarioId"))
    ]
    missing_actual_signal = [
        _record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _actual_signal_present(record)
    ]
    if missing_task_or_scenario:
        blockers.append("deployment_outcomes_missing_task_or_scenario")
    if missing_actual_signal:
        blockers.append("deployment_outcomes_missing_actual_result_signal")
    missing_owner_evidence = [
        _record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _outcome_owner_evidence_present(record)
    ]
    missing_prediction_match_keys = [
        _record_id(record, index)
        for index, record in enumerate(records, start=1)
        if not _prediction_match_key_present(record)
    ]
    owner_evidence_record_count = len(records) - len(missing_owner_evidence)
    prediction_match_key_record_count = len(records) - len(missing_prediction_match_keys)
    records_ready_for_calibration = (
        bool(records)
        and not blockers
        and not missing_prediction_match_keys
    )
    return {
        "status": "ready_for_real_world_validation" if not blockers else "blocked",
        "ready": not blockers,
        "records_ready_for_calibration": records_ready_for_calibration,
        "prediction_match_keys_ready": records_ready_for_calibration,
        "owner_evidence_ready": bool(records) and not missing_owner_evidence and not blockers,
        "path": str(path),
        "sha256": _sha_file(path),
        "schema_version": payload.get("schema_version"),
        "job_id": resolved_job_id,
        "evidence_job_id": evidence_job_id,
        "expected_job_id": expected_job_id,
        "record_count": len(records),
        "record_ids": [_record_id(record, index) for index, record in enumerate(records, start=1)],
        "prediction_match_key_record_count": prediction_match_key_record_count,
        "missing_prediction_match_key_record_ids": missing_prediction_match_keys,
        "owner_evidence_record_count": owner_evidence_record_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence,
        "missing_task_or_scenario_record_ids": missing_task_or_scenario,
        "missing_actual_signal_record_ids": missing_actual_signal,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Deployment outcomes are owner-supplied real-world validation inputs; they do not "
            "prove readiness until every outcome has owner evidence, the job pairs them with "
            "predictions, and closure passes."
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

    review = _section(payload, "review_acceptance", "reviewAcceptance")
    delivery = _section(payload, "delivery", "signed_delivery", "signedDelivery")
    safety = _section(
        payload,
        "safety_contact_physics",
        "safetyContactPhysics",
        "robot_readiness",
        "robotReadiness",
    )
    rights = _section(payload, "rights_privacy", "rightsPrivacy")
    webapp = _section(payload, "webapp_upstream", "webappUpstream")

    review_ready = (
        _boolish(review.get("accepted"))
        or _status_ok(review.get("status"))
    ) and (
        _attestation_ok(
            review.get("operator_attestation")
            or review.get("operatorAttestation")
            or review.get("owner_attestation")
            or review.get("ownerAttestation")
        )
        or bool(_string(review.get("reviewer")))
    )
    signed_urls = delivery.get("signed_urls") or delivery.get("signedUrls") or []
    delivery_ready = (
        bool(signed_urls)
        and _boolish(
            delivery.get("storage_upload_performed")
            or delivery.get("storageUploadPerformed")
        )
        and _boolish(delivery.get("entitlement_verified") or delivery.get("entitlementVerified"))
    )
    safety_ref_fields = (
        "methodology_uri_or_path",
        "methodologyUriOrPath",
        "contact_validation_uri_or_path",
        "contactValidationUriOrPath",
        "safety_validation_uri_or_path",
        "safetyValidationUriOrPath",
    )
    safety_refs = [field for field in safety_ref_fields if _string(safety.get(field))]
    safety_ready = (
        _boolish(safety.get("physics_contact_validated") or safety.get("physicsContactValidated"))
        and _boolish(safety.get("safety_validated") or safety.get("safetyValidated"))
        and _boolish(safety.get("robot_readiness_proven") or safety.get("robotReadinessProven"))
        and bool(safety_refs)
        and _attestation_ok(
            safety.get("operator_attestation")
            or safety.get("operatorAttestation")
            or safety.get("owner_attestation")
            or safety.get("ownerAttestation")
        )
    )
    rights_ready = (
        not rights
        or _status_ok(rights.get("status"))
        or _boolish(rights.get("accepted"))
        or _boolish(rights.get("external_use_allowed") or rights.get("externalUseAllowed"))
    )
    webapp_id_count = sum(
        1 for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS if _string(webapp.get(field))
    )

    if not review_ready:
        blockers.append("review_acceptance_evidence_incomplete")
    if not delivery_ready:
        blockers.append("delivery_evidence_incomplete")
    if not safety_ready:
        blockers.append("safety_contact_physics_evidence_incomplete")
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
            "review_acceptance_ready": review_ready,
            "delivery_ready": delivery_ready,
            "safety_contact_physics_ready": safety_ready,
            "rights_privacy_ready": rights_ready,
            "webapp_upstream_id_count": webapp_id_count,
        },
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Closure evidence is staged for the deterministic live closure audit only; it does "
            "not prove robot readiness until live_eval_closure_manifest.json passes."
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
    target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / REAL_ROBOT_POV_ARTIFACT_NAME
    )
    ensure_dir(target.parent)
    blockers: List[str] = []
    if target.exists() and not overwrite:
        blockers.append("target_real_robot_pov_already_exists")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": blockers,
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
        "proof_boundary": "staging copies real robot POV references only and does not run the job",
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
    digest = _string(audit.get("sha256"))[:12] or _sha_file(outcome_path)[:12]
    record_ids = [
        _string(record_id)
        for record_id in audit.get("record_ids", [])
        if _string(record_id)
    ]
    stem = record_ids[0] if len(record_ids) == 1 else f"{job_id}-{digest}"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in stem).strip(".-")
    safe_stem = safe_stem[:120] or f"{job_id}-{digest}"
    target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / "deployment_outcomes"
        / "inbox"
        / f"{safe_stem}.json"
    )
    ensure_dir(target.parent)
    blockers: List[str] = []
    if target.exists() and not overwrite:
        blockers.append("target_deployment_outcome_already_exists")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": blockers,
        }
    shutil.copy2(outcome_path, target)
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target),
        "job_id": job_id,
        "record_count": int(audit.get("record_count") or 0),
        "sha256": _sha_file(target),
        "blockers": [],
        "proof_boundary": "staging copies deployment outcomes only and does not run the job",
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
            "records_ready_for_calibration": bool(
                deployment_outcomes_audit.get("records_ready_for_calibration")
            )
            and outcomes_staged,
            "prediction_match_keys_ready": bool(
                deployment_outcomes_audit.get("prediction_match_keys_ready")
            )
            and outcomes_staged,
            "owner_evidence_ready": bool(
                deployment_outcomes_audit.get("owner_evidence_ready")
            )
            and outcomes_staged,
            "staged": outcomes_staged,
            "job_id": deployment_outcomes_audit.get("job_id")
            or deployment_outcomes_staging.get("job_id"),
            "path": deployment_outcomes_audit.get("path"),
            "target_path": deployment_outcomes_staging.get("target_path"),
            "sha256": deployment_outcomes_staging.get("sha256")
            or deployment_outcomes_audit.get("sha256"),
            "record_count": deployment_outcomes_audit.get("record_count", 0),
            "record_ids": deployment_outcomes_audit.get("record_ids", []),
            "prediction_match_key_record_count": deployment_outcomes_audit.get(
                "prediction_match_key_record_count", 0
            ),
            "missing_prediction_match_key_record_ids": deployment_outcomes_audit.get(
                "missing_prediction_match_key_record_ids", []
            ),
            "owner_evidence_record_count": deployment_outcomes_audit.get(
                "owner_evidence_record_count", 0
            ),
            "missing_owner_evidence_record_ids": deployment_outcomes_audit.get(
                "missing_owner_evidence_record_ids", []
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
            "job_id": real_robot_pov_audit.get("job_id") or real_robot_pov_staging.get("job_id"),
            "path": real_robot_pov_audit.get("path"),
            "target_path": real_robot_pov_staging.get("target_path"),
            "sha256": real_robot_pov_staging.get("sha256")
            or real_robot_pov_audit.get("sha256"),
            "record_count": real_robot_pov_audit.get("record_count", 0),
            "record_ids": real_robot_pov_audit.get("record_ids", []),
            "exact_key_record_count": real_robot_pov_audit.get("exact_key_record_count", 0),
            "camera_video_record_count": real_robot_pov_audit.get("camera_video_record_count", 0),
            "action_log_record_count": real_robot_pov_audit.get("action_log_record_count", 0),
            "timestamp_alignment_record_count": real_robot_pov_audit.get(
                "timestamp_alignment_record_count", 0
            ),
            "evidence_record_count": real_robot_pov_audit.get("evidence_record_count", 0),
            "missing_exact_key_record_ids": real_robot_pov_audit.get(
                "missing_exact_key_record_ids", []
            ),
            "missing_camera_video_record_ids": real_robot_pov_audit.get(
                "missing_camera_video_record_ids", []
            ),
            "missing_action_log_record_ids": real_robot_pov_audit.get(
                "missing_action_log_record_ids", []
            ),
            "missing_timestamp_alignment_record_ids": real_robot_pov_audit.get(
                "missing_timestamp_alignment_record_ids", []
            ),
            "missing_evidence_record_ids": real_robot_pov_audit.get(
                "missing_evidence_record_ids", []
            ),
        },
        "proof_boundary": {
            "staged_inputs_are_pointers_only": True,
            "local_webapp_rehearsal_only": bool(webapp_audit.get("local_rehearsal_only")),
            "live_webapp_forwarding_proven": False,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "robot_pov_evidence_proven": False,
            "real_world_outcome_proven": False,
            "policy_execution_proven": False,
            "robot_readiness_proven": False,
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
            "robot_readiness_proven": False,
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
