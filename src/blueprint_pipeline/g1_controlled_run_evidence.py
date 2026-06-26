"""Assemble physical Unitree G1 run evidence into live robot-eval input manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json, write_text
from .g1_controlled_proof_setup import (
    DEFAULT_POLICY_ID,
    DEFAULT_ROBOT_MAKE_MODEL,
    DEFAULT_ROBOT_PROFILE_ID,
)


G1_CONTROLLED_RUN_EVIDENCE_SCHEMA_VERSION = "g1_controlled_run_evidence_assembly.v1"
EVIDENCE_INPUT_TEMPLATE_SCHEMA_VERSION = "g1_controlled_run_inputs.v1"
CONTROLLED_FIELD_ANCHOR_REQUEST_SCHEMA_VERSION = "controlled_field_anchor_request_packet.v1"
ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION = "accepted_real_world_anchor.v1"
CONTROLLED_ANCHOR_JOIN_KEYS = (
    "scenario_eval_run_id",
    "policy_id",
    "task_id",
    "scenario_variation_instance_id",
)


REQUIRED_EVIDENCE_FILES = {
    "robot_camera_video": ("robot_camera_video.mp4", "pov_video.mp4", "g1_pov.mp4"),
    "timestamp_alignment": ("timestamp_alignment.json",),
    "action_log": ("action_log.jsonl", "action_log.json"),
    "robot_state_log": ("robot_state_log.jsonl", "robot_state_log.json"),
    "command_log": ("command_log.jsonl", "command_log.json"),
    "contact_collision_log": ("contact_collision_log.json", "contact_collision_log.jsonl"),
    "hardware_validation": ("hardware_validation.json", "non_ranking_operational_claim.json"),
    "policy_execution_trace": ("policy_execution_trace.jsonl", "policy_execution_trace.json"),
    "policy_metrics": ("policy_metrics.json",),
    "robot_team_review": ("robot_team_review.json", "policy_owner_review.json"),
}
OWNER_EVIDENCE_CLAIM_REQUIREMENTS = (
    "robot_camera_video",
    "action_log",
    "timestamp_alignment",
    "hardware_validation",
    "contact_collision_log",
    "policy_metrics",
    "robot_team_review",
)
REQUIRED_ATTESTATION_FIELDS = {
    "operator": {
        "actor_field": "operator_id",
        "statement_field": "operator_statement",
        "signed_field": "operator_attestation_signed",
        "signed_at_field": "operator_attestation_signed_at_utc",
    },
    "hardware_owner": {
        "actor_field": "hardware_owner_id",
        "statement_field": "hardware_owner_statement",
        "signed_field": "hardware_owner_attestation_signed",
        "signed_at_field": "hardware_owner_attestation_signed_at_utc",
    },
    "safety_reviewer": {
        "actor_field": "safety_reviewer_id",
        "statement_field": "safety_reviewer_statement",
        "signed_field": "safety_reviewer_attestation_signed",
        "signed_at_field": "safety_reviewer_attestation_signed_at_utc",
    },
    "robot_team_reviewer": {
        "actor_field": "robot_team_reviewer_id",
        "statement_field": "robot_team_review_statement",
        "signed_field": "robot_team_review_attestation_signed",
        "signed_at_field": "robot_team_review_attestation_signed_at_utc",
    },
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "passed", "success"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _safe_id(value: str, fallback: str) -> str:
    text = _string(value)
    if not text:
        return fallback
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text).strip("-") or fallback


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_ref(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "uri": str(path),
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _discover_evidence_files(evidence_dir: Path) -> tuple[dict[str, Path], list[str]]:
    discovered: dict[str, Path] = {}
    blockers: list[str] = []
    for evidence_id, candidates in REQUIRED_EVIDENCE_FILES.items():
        for candidate in candidates:
            path = evidence_dir / candidate
            if path.is_file():
                discovered[evidence_id] = path.resolve()
                break
        if evidence_id not in discovered:
            blockers.append(f"missing_evidence_file:{evidence_id}")
    return discovered, blockers


def _input_template(context: Mapping[str, Any]) -> dict[str, Any]:
    job_id = _string(context.get("job_id")) or "<robot-eval-job-id>"
    task_id = _string(context.get("task_id")) or "<task-id>"
    scenario_id = _string(context.get("scenario_id")) or "<scenario-id>"
    scenario_eval_run_id = _string(context.get("scenario_eval_run_id")) or "<scenario-eval-run-id>"
    variation_id = (
        _string(context.get("scenario_variation_instance_id"))
        or "<scenario-variation-instance-id>"
    )
    return {
        "schema_version": EVIDENCE_INPUT_TEMPLATE_SCHEMA_VERSION,
        "job_id": job_id,
        "run_id": f"unitree-g1-controlled-run-{_safe_id(job_id, 'job')}",
        "sim_only_mode": True,
        "physical_evidence_required": False,
        "policy_id": DEFAULT_POLICY_ID,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "scenario_eval_run_id": scenario_eval_run_id,
        "scenario_variation_instance_id": variation_id,
        "robot_serial_or_fleet_id": "<unitree-g1-serial-or-fleet-id>",
        "site_or_lab_location_id": "<controlled-test-site-or-lab-id>",
        "operator_id": "<operator-id>",
        "hardware_owner_id": "<hardware-owner-id>",
        "safety_reviewer_id": "<safety-reviewer-id>",
        "robot_team_reviewer_id": "<robot-team-reviewer-id>",
        "start_time_utc": "<timestamp>",
        "end_time_utc": "<timestamp>",
        "actual_status": "passed",
        "actual_success": True,
        "actual_outcome": {
            "actual_status": "passed",
            "actual_success": True,
            "failure_mode_ids": [],
            "cycle_time_seconds": "<measured-cycle-time>",
            "intervention_count": 0,
        },
        "cycle_time_seconds": "<measured-cycle-time>",
        "intervention_count": 0,
        "operator_site_checklist": {
            "controlled_area_verified": False,
            "floor_clear_for_g1_walk": False,
            "bystander_exclusion_zone_marked": False,
            "emergency_stop_operator_present": True,
            "site_or_lab_location_id": "<controlled-test-site-or-lab-id>",
        },
        "allowed_task_set": [
            {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "scenario_variation_instance_id": variation_id,
                "policy_id": DEFAULT_POLICY_ID,
                "allowed": True,
            }
        ],
        "exclusion_and_abort_criteria": {
            "excluded_task_ids": ["<any-task-not-listed-in-allowed_task_set>"],
            "abort_conditions": [
                "loss_of_comms",
                "unexpected_human_entry",
                "fall_detected",
                "contact_force_exceeds_threshold",
                "operator_estop",
            ],
            "loose_or_inferred_anchor_matches_allowed": False,
        },
        "robot_calibration_refs": {
            "robot_state_log": "robot_state_log.jsonl",
            "hardware_validation": "hardware_validation.json",
            "contact_collision_log": "contact_collision_log.json",
        },
        "camera_calibration_refs": {
            "robot_camera_video": "robot_camera_video.mp4",
            "timestamp_alignment": "timestamp_alignment.json",
            "camera_mount_or_sensor_ids": ["<g1-head-or-body-camera-id>"],
        },
        "accepted_safety_thresholds": {
            "max_speed_mps": "<reviewed-threshold>",
            "min_human_clearance_m": "<reviewed-threshold>",
            "max_contact_force_n": "<reviewed-threshold>",
            "emergency_stop_required": True,
        },
        "review_decision": "accepted",
        "reviewer_decision": {
            "safety_review_decision": "not_reviewed",
            "policy_review_decision": "not_reviewed",
            "accepted_for_calibration": False,
        },
        "owner_evidence_refs": {
            "robot_camera_video": "robot_camera_video.mp4",
            "action_log": "action_log.jsonl",
            "timestamp_alignment": "timestamp_alignment.json",
        },
        "storage_upload_performed": False,
        "entitlement_verified": False,
        "signed_customer_delivery_url": "<signed-customer-delivery-url>",
        "rights_privacy_status": "not_reviewed",
        "external_use_allowed": False,
        "production_webapp_request_id": "<production-webapp-request-id>",
        "pipeline_intake_request_id": "<pipeline-intake-request-id>",
        "production_forward_url": "<production-forward-url>",
        "webapp_response_status_code": "<202>",
        "sync_status": "<succeeded>",
        "operator_statement": (
            "I attest that the referenced files were captured from the physical Unitree G1 "
            "controlled run for this job, task, and scenario."
        ),
        "operator_attestation_signed": False,
        "operator_attestation_signed_at_utc": "<operator-attestation-signed-at>",
        "hardware_owner_statement": "I attest that the robot identifier and hardware run are accurate.",
        "hardware_owner_attestation_signed": False,
        "hardware_owner_attestation_signed_at_utc": "<hardware-owner-attestation-signed-at>",
        "safety_reviewer_statement": "I attest that the safety package was reviewed for this run.",
        "safety_reviewer_attestation_signed": False,
        "safety_reviewer_attestation_signed_at_utc": "<safety-reviewer-attestation-signed-at>",
        "robot_team_review_statement": (
            "I accept this non-default Unitree G1 policy package for this controlled run."
        ),
        "robot_team_review_attestation_signed": False,
        "robot_team_review_attestation_signed_at_utc": "<robot-team-review-attestation-signed-at>",
    }


def _attestation(
    *,
    role: str,
    actor_id: str,
    statement: str,
    signed: bool,
    signed_at_utc: str,
) -> dict[str, Any]:
    status = (
        "signed"
        if signed
        and not _is_placeholder(actor_id)
        and not _is_placeholder(statement)
        and not _is_placeholder(signed_at_utc)
        else "not_signed"
    )
    return {
        "status": status,
        "role": role,
        "attested_by": actor_id,
        "statement": statement,
        "accepted_claim_boundary": (
            "This attestation applies only to the referenced physical Unitree G1 run evidence."
        ),
        "signed_at_utc": signed_at_utc,
    }


def _attestation_from_config(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    fields = REQUIRED_ATTESTATION_FIELDS[role]
    return _attestation(
        role=role,
        actor_id=_string(config.get(fields["actor_field"])),
        statement=_string(config.get(fields["statement_field"])),
        signed=_bool(config.get(fields["signed_field"])),
        signed_at_utc=_string(config.get(fields["signed_at_field"])),
    )


def _attestation_config_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for role, fields in REQUIRED_ATTESTATION_FIELDS.items():
        signed_field = fields["signed_field"]
        signed_at_field = fields["signed_at_field"]
        if not _bool(config.get(signed_field)):
            blockers.append(f"unsigned_attestation:{role}")
        signed_at = _string(config.get(signed_at_field))
        if not signed_at or "<" in signed_at or ">" in signed_at:
            blockers.append(f"missing_or_placeholder_attestation_signed_at:{role}")
    return blockers


def _required_config_blockers(config: Mapping[str, Any]) -> list[str]:
    required = [
        "run_id",
        "policy_id",
        "task_id",
        "scenario_eval_run_id",
        "scenario_variation_instance_id",
        "robot_serial_or_fleet_id",
        "site_or_lab_location_id",
        "operator_id",
        "hardware_owner_id",
        "safety_reviewer_id",
        "robot_team_reviewer_id",
        "start_time_utc",
        "end_time_utc",
        "cycle_time_seconds",
        "production_webapp_request_id",
        "pipeline_intake_request_id",
        "production_forward_url",
        "webapp_response_status_code",
    ]
    blockers: list[str] = []
    for field in required:
        value = _string(config.get(field))
        if not value or "<" in value or ">" in value:
            blockers.append(f"missing_or_placeholder_config:{field}")
    for field in (
        "operator_statement",
        "hardware_owner_statement",
        "safety_reviewer_statement",
        "robot_team_review_statement",
    ):
        value = _string(config.get(field))
        if not value or "<" in value or ">" in value:
            blockers.append(f"missing_or_placeholder_config:{field}")
    blockers.extend(_attestation_config_blockers(config))
    thresholds = _mapping(config.get("accepted_safety_thresholds"))
    for field in ("max_speed_mps", "min_human_clearance_m", "max_contact_force_n"):
        value = _string(thresholds.get(field))
        if not value or "<" in value or ">" in value:
            blockers.append(f"missing_or_placeholder_safety_threshold:{field}")
        elif _number(value) is None:
            blockers.append(f"non_numeric_safety_threshold:{field}")
    if _number(config.get("cycle_time_seconds")) is None:
        blockers.append("non_numeric_config:cycle_time_seconds")
    if _string(config.get("actual_status")).lower() != "passed":
        blockers.append("physical_run_status_not_passed")
    if not _bool(config.get("actual_success")):
        blockers.append("physical_run_actual_success_not_true")
    if _string(config.get("review_decision")).lower() not in {"accepted", "approved", "passed"}:
        blockers.append("safety_review_decision_not_accepted")
    if _string(config.get("sync_status")).lower() not in {"succeeded", "success", "synced"}:
        blockers.append("webapp_sync_status_not_succeeded")
    if not _string(config.get("signed_customer_delivery_url")) or "<" in _string(
        config.get("signed_customer_delivery_url")
    ):
        blockers.append("missing_or_placeholder_config:signed_customer_delivery_url")
    if not _bool(config.get("storage_upload_performed")):
        blockers.append("storage_upload_not_performed")
    if not _bool(config.get("entitlement_verified")):
        blockers.append("entitlement_not_verified")
    if not _bool(config.get("external_use_allowed")):
        blockers.append("rights_privacy_external_use_not_allowed")
    return blockers


def _configured_policy_id(config: Mapping[str, Any]) -> str:
    return _string(config.get("policy_id")) or DEFAULT_POLICY_ID


def _physical_evidence_required(config: Mapping[str, Any]) -> bool:
    reviewer_decision = _mapping(config.get("reviewer_decision"))
    return bool(
        _bool(config.get("physical_evidence_required"))
        or _bool(config.get("field_evidence_required"))
        or _bool(config.get("accepted_for_calibration"))
        or _bool(reviewer_decision.get("accepted_for_calibration"))
    )


def _expected_anchor_values(context: Mapping[str, Any]) -> dict[str, str]:
    return {
        "policy_id": DEFAULT_POLICY_ID,
        "task_id": _string(context.get("task_id")),
        "scenario_eval_run_id": _string(context.get("scenario_eval_run_id")),
        "scenario_variation_instance_id": _string(
            context.get("scenario_variation_instance_id")
        ),
    }


def _anchor_join_key(config: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, str]:
    expected = _expected_anchor_values(context)
    return {
        "scenario_eval_run_id": _string(config.get("scenario_eval_run_id"))
        or expected["scenario_eval_run_id"],
        "policy_id": _configured_policy_id(config),
        "task_id": _string(config.get("task_id")) or expected["task_id"],
        "scenario_variation_instance_id": _string(
            config.get("scenario_variation_instance_id")
        )
        or expected["scenario_variation_instance_id"],
    }


def _anchor_context_blockers(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected = _expected_anchor_values(context)
    for field, expected_value in expected.items():
        configured = _string(config.get(field))
        if not configured or "<" in configured or ">" in configured:
            continue
        if expected_value and configured != expected_value:
            blockers.append(f"mismatched_anchor_join_key:{field}")
    allowed_rows = config.get("allowed_task_set")
    if not isinstance(allowed_rows, list) or not allowed_rows:
        blockers.append("missing_allowed_task_set")
    else:
        join_key = _anchor_join_key(config, context)
        if not any(
            isinstance(row, Mapping)
            and all(_string(row.get(key)) == value for key, value in join_key.items())
            and row.get("allowed") is True
            for row in allowed_rows
        ):
            blockers.append("allowed_task_set_missing_exact_anchor_join_key")
    if not isinstance(config.get("operator_site_checklist"), Mapping):
        blockers.append("missing_operator_site_checklist")
    if not isinstance(config.get("exclusion_and_abort_criteria"), Mapping):
        blockers.append("missing_exclusion_and_abort_criteria")
    if not isinstance(config.get("robot_calibration_refs"), Mapping):
        blockers.append("missing_robot_calibration_refs")
    if not isinstance(config.get("camera_calibration_refs"), Mapping):
        blockers.append("missing_camera_calibration_refs")
    return blockers


def _read_json_file(path: Path, evidence_id: str) -> tuple[Any, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception:
        return None, [f"invalid_json_evidence:{evidence_id}"]


def _read_json_records(path: Path, evidence_id: str) -> tuple[list[Any], list[str]]:
    if path.suffix == ".json":
        payload, blockers = _read_json_file(path, evidence_id)
        if blockers:
            return [], blockers
        if isinstance(payload, list):
            return payload, []
        if isinstance(payload, Mapping):
            return [payload], []
        return [], [f"invalid_json_evidence:{evidence_id}"]
    records: list[Any] = []
    blockers: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], [f"unreadable_evidence_file:{evidence_id}"]
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            records.append(json.loads(text))
        except Exception:
            blockers.append(f"invalid_jsonl_evidence:{evidence_id}:line_{line_number}")
            break
    if not records and not blockers:
        blockers.append(f"empty_evidence_records:{evidence_id}")
    return records, blockers


def _is_placeholder(value: Any) -> bool:
    text = _string(value)
    return not text or "<" in text or ">" in text


def _contains_placeholder_value(value: Any) -> bool:
    if isinstance(value, str):
        return "<" in value or ">" in value
    if isinstance(value, Mapping):
        return any(_contains_placeholder_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder_value(item) for item in value)
    return False


def _placeholder_evidence_blocker(evidence_id: str, value: Any) -> list[str]:
    if _contains_placeholder_value(value):
        return [f"placeholder_evidence_value:{evidence_id}"]
    return []


def _accepted_status(value: Any) -> bool:
    return _string(value).lower() in {"accepted", "approved", "passed", "succeeded", "complete"}


def _evidence_content_blockers(
    discovered_files: Mapping[str, Path],
    config: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    expected_policy_id = _configured_policy_id(config)
    video_path = discovered_files.get("robot_camera_video")
    if video_path and video_path.stat().st_size <= 0:
        blockers.append("empty_evidence_file:robot_camera_video")

    alignment_path = discovered_files.get("timestamp_alignment")
    if alignment_path:
        alignment, alignment_blockers = _read_json_file(alignment_path, "timestamp_alignment")
        blockers.extend(alignment_blockers)
        if not alignment_blockers:
            blockers.extend(_placeholder_evidence_blocker("timestamp_alignment", alignment))
            alignment_map = _mapping(alignment)
            max_error = _number(alignment_map.get("max_alignment_error_ms"))
            if max_error is None:
                blockers.append("timestamp_alignment_missing_max_alignment_error_ms")
            elif max_error > 250:
                blockers.append("timestamp_alignment_error_exceeds_250_ms")

    for evidence_id in ("action_log", "robot_state_log", "command_log", "policy_execution_trace"):
        path = discovered_files.get(evidence_id)
        if not path:
            continue
        records, record_blockers = _read_json_records(path, evidence_id)
        blockers.extend(record_blockers)
        if not record_blockers:
            blockers.extend(_placeholder_evidence_blocker(evidence_id, records))
        if evidence_id == "action_log" and records:
            action_record_fields = {
                "action_id",
                "action",
                "action_vector",
                "motor_command",
                "motor_targets",
                "joint_targets",
                "velocity_command",
                "locomotion_command",
            }
            has_action_record = any(
                bool(action_record_fields.intersection(_mapping(record).keys()))
                for record in records
            )
            if not has_action_record:
                blockers.append("action_log_missing_robot_action_record")
        if evidence_id == "command_log" and records:
            completed = [
                _mapping(record)
                for record in records
                if _string(_mapping(record).get("kind")) == "policy_command_completed"
            ]
            if not completed:
                blockers.append("command_log_missing_policy_command_completed")
            elif any(_number(record.get("exit_code")) != 0 for record in completed):
                blockers.append("command_log_policy_command_exit_nonzero")
        if evidence_id == "policy_execution_trace" and records:
            policy_ids = [
                _string(_mapping(record).get("policy_id"))
                for record in records
                if _string(_mapping(record).get("policy_id"))
            ]
            if not policy_ids:
                blockers.append("policy_execution_trace_missing_policy_id")
            elif expected_policy_id not in policy_ids:
                blockers.append("policy_execution_trace_policy_id_mismatch")

    contact_path = discovered_files.get("contact_collision_log")
    if contact_path:
        contact, contact_blockers = _read_json_file(contact_path, "contact_collision_log")
        blockers.extend(contact_blockers)
        contact_map = _mapping(contact)
        if not contact_blockers:
            blockers.extend(_placeholder_evidence_blocker("contact_collision_log", contact))
            if _string(contact_map.get("status")).lower() == "operator_review_required":
                blockers.append("contact_collision_log_still_operator_review_required")
            max_contact_force = _number(contact_map.get("max_contact_force_n"))
            threshold = _number(_mapping(config.get("accepted_safety_thresholds")).get("max_contact_force_n"))
            if max_contact_force is None:
                blockers.append("contact_collision_log_missing_max_contact_force_n")
            elif threshold is not None and max_contact_force > threshold:
                blockers.append("contact_collision_log_exceeds_accepted_threshold")

    hardware_path = discovered_files.get("hardware_validation")
    if hardware_path:
        hardware, hardware_blockers = _read_json_file(hardware_path, "hardware_validation")
        blockers.extend(hardware_blockers)
        hardware_map = _mapping(hardware)
        if not hardware_blockers:
            blockers.extend(_placeholder_evidence_blocker("hardware_validation", hardware))
            if hardware_map.get("hardware_ready") is not True:
                blockers.append("hardware_validation_not_ready")
            if hardware_map.get("estop_verified") is not True:
                blockers.append("hardware_validation_estop_not_verified")
            if not _accepted_status(hardware_map.get("status")):
                blockers.append("hardware_validation_status_not_accepted")

    metrics_path = discovered_files.get("policy_metrics")
    if metrics_path:
        metrics, metrics_blockers = _read_json_file(metrics_path, "policy_metrics")
        blockers.extend(metrics_blockers)
        metrics_map = _mapping(metrics)
        if not metrics_blockers:
            blockers.extend(_placeholder_evidence_blocker("policy_metrics", metrics))
            episode_count = _number(metrics_map.get("episode_count"))
            if episode_count is None or episode_count <= 0:
                blockers.append("policy_metrics_missing_episode_count")
            if _number(metrics_map.get("success_rate")) is None:
                blockers.append("policy_metrics_missing_success_rate")
            if _number(metrics_map.get("intervention_count")) is None:
                blockers.append("policy_metrics_missing_intervention_count")
            if not _accepted_status(metrics_map.get("status")):
                blockers.append("policy_metrics_status_not_accepted")

    review_path = discovered_files.get("robot_team_review")
    if review_path:
        review, review_blockers = _read_json_file(review_path, "robot_team_review")
        blockers.extend(review_blockers)
        review_map = _mapping(review)
        if not review_blockers:
            blockers.extend(_placeholder_evidence_blocker("robot_team_review", review))
            if review_map.get("accepted") is not True:
                blockers.append("robot_team_review_not_accepted")
            if not _accepted_status(review_map.get("review_decision")):
                blockers.append("robot_team_review_decision_not_accepted")
            if _is_placeholder(review_map.get("reviewer_id")):
                blockers.append("robot_team_review_missing_reviewer_id")
            elif _string(review_map.get("reviewer_id")) != _string(
                config.get("robot_team_reviewer_id")
            ):
                blockers.append("robot_team_review_reviewer_id_mismatch")

    return blockers


def _attestation_is_signed(attestation: Mapping[str, Any]) -> bool:
    return (
        _string(attestation.get("status")) == "signed"
        and not _is_placeholder(attestation.get("attested_by"))
        and not _is_placeholder(attestation.get("statement"))
        and not _is_placeholder(attestation.get("signed_at_utc"))
    )


def _controlled_anchor_request_packet(
    *,
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    evidence_root: Path,
    evidence_file_refs: Mapping[str, Any],
    blockers: Sequence[str],
    ready: bool,
    physical_evidence_required: bool,
) -> dict[str, Any]:
    join_key = _anchor_join_key(config, context)
    status = (
        "ready_for_calibration_intake"
        if ready
        else "blocked_missing_evidence"
        if physical_evidence_required
        else "not_requested_for_sim_only"
    )
    return {
        "schema_version": CONTROLLED_FIELD_ANCHOR_REQUEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_root_job_context": dict(context),
        "output_dir": str(output_root),
        "evidence_dir": str(evidence_root),
        "accepted_anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
        "required_exact_join_keys": list(CONTROLLED_ANCHOR_JOIN_KEYS),
        "anchor_join_key": join_key,
        "loose_or_inferred_matches_allowed_for_calibration": False,
        "operator_site_checklist": _mapping(config.get("operator_site_checklist")),
        "allowed_task_set": config.get("allowed_task_set") or [],
        "exclusion_and_abort_criteria": _mapping(config.get("exclusion_and_abort_criteria")),
        "robot_calibration_refs": _mapping(config.get("robot_calibration_refs")),
        "camera_calibration_refs": _mapping(config.get("camera_calibration_refs")),
        "policy_id": join_key["policy_id"],
        "task_id": join_key["task_id"],
        "scenario_eval_run_id": join_key["scenario_eval_run_id"],
        "scenario_variation_instance_id": join_key["scenario_variation_instance_id"],
        "actual_outcome": {
            **_mapping(config.get("actual_outcome")),
            "actual_status": _string(config.get("actual_status"))
            or _string(_mapping(config.get("actual_outcome")).get("actual_status")),
            "actual_success": _bool(
                config.get("actual_success")
                if config.get("actual_success") is not None
                else _mapping(config.get("actual_outcome")).get("actual_success")
            ),
            "cycle_time_seconds": _number(config.get("cycle_time_seconds")),
            "intervention_count": _number(config.get("intervention_count"), 0),
        },
        "reviewer_decision": {
            **_mapping(config.get("reviewer_decision")),
            "safety_review_decision": _string(config.get("review_decision")) or "not_reviewed",
            "policy_review_decision": "accepted" if ready else "not_accepted",
            "accepted_for_calibration": ready,
        },
        "owner_evidence": {
            "operator_id": _string(config.get("operator_id")),
            "hardware_owner_id": _string(config.get("hardware_owner_id")),
            "required_before_physical_claim": list(OWNER_EVIDENCE_CLAIM_REQUIREMENTS),
            "owner_evidence_refs": _mapping(config.get("owner_evidence_refs")),
            "evidence_file_refs": dict(evidence_file_refs),
            "all_required_owner_evidence_present": all(
                evidence_file_refs.get(evidence_id)
                for evidence_id in OWNER_EVIDENCE_CLAIM_REQUIREMENTS
            ),
        },
        "timestamps": {
            "start_time_utc": _string(config.get("start_time_utc")),
            "end_time_utc": _string(config.get("end_time_utc")),
        },
        "artifact_provenance": {
            "input_config_path": str(evidence_root / "g1_controlled_run_inputs.json"),
            "assembly_output_dir": str(output_root),
            "evidence_file_refs": dict(evidence_file_refs),
            "claim_boundary": (
                "This packet can carry controlled field evidence when explicitly requested. "
                "For sim-only runs it is not a blocker and does not request physical evidence."
            ),
        },
        "blockers": (
            sorted(set(_string(item) for item in blockers if _string(item)))
            if physical_evidence_required
            else []
        ),
        "diagnostic_blockers": sorted(set(_string(item) for item in blockers if _string(item))),
        "proof_boundary": {
            **_proof_boundary(),
            "accepted_anchors_can_calibrate_evaluator_ranking_against_supplied_outcomes": ready,
            "broad_deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "future_real_world_success_proven": False,
            "sim_only_beta_ranking_blocked": False,
            "physical_evidence_not_requested_for_sim_only": not physical_evidence_required,
        },
    }


def _proof_boundary() -> dict[str, bool]:
    return {
        "assembly_is_metadata_only": True,
        "generated_world_rank_fidelity_result_proven": False,
        "non_ranking_operational_claim_validated": False,
        "real_robot_pov_evidence_proven": False,
        "robot_team_policy_performance_proven": False,
        "production_runpod_worker_execution_proven": False,
        "customer_through_website_testing_ready": False,
        "public_claim_upgrade_allowed": False,
    }


def write_g1_controlled_run_input_template(
    *,
    output_path: str | Path,
    capture_root: str | Path,
    job_id: str | None = None,
) -> dict[str, Any]:
    from .g1_controlled_proof_setup import _job_context

    root = Path(capture_root).expanduser().resolve()
    context = _job_context(root, job_id)
    template = _input_template(context)
    path = Path(output_path).expanduser().resolve()
    ensure_dir(path.parent)
    write_json(path, template)
    return template


def assemble_g1_controlled_run_evidence(
    *,
    capture_root: str | Path,
    evidence_dir: str | Path,
    job_id: str | None = None,
    input_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    from .g1_controlled_proof_setup import _job_context

    root = Path(capture_root).expanduser().resolve()
    evidence_root = Path(evidence_dir).expanduser().resolve()
    context = _job_context(root, job_id)
    config_path = (
        Path(input_config_path).expanduser().resolve()
        if input_config_path
        else evidence_root / "g1_controlled_run_inputs.json"
    )
    config = optional_read_json(config_path) or {}
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "pipeline" / "g1_controlled_proof_setup" / "assembled_live_inputs"
    )
    ensure_dir(output_root)

    discovered_files, discovered_missing_files = _discover_evidence_files(evidence_root)
    physical_evidence_required = _physical_evidence_required(config)
    if physical_evidence_required:
        file_blockers = discovered_missing_files
        config_blockers = [
            *_required_config_blockers(config),
            *_anchor_context_blockers(config, context),
        ]
        content_blockers = _evidence_content_blockers(discovered_files, config)
    else:
        file_blockers = []
        config_blockers = []
        content_blockers = []
    blockers = [*file_blockers, *config_blockers, *content_blockers]
    ready = bool(physical_evidence_required and not blockers)
    status = (
        "ready_for_live_input_staging"
        if ready
        else "blocked_missing_evidence"
        if physical_evidence_required
        else "not_requested_for_sim_only"
    )
    job_id_value = _string(context.get("job_id"))
    anchor_join_key = _anchor_join_key(config, context)
    policy_id = anchor_join_key["policy_id"]
    task_id = anchor_join_key["task_id"]
    scenario_id = _string(context.get("scenario_id"))
    scenario_variation_id = anchor_join_key["scenario_variation_instance_id"]
    scenario_eval_run_id = anchor_join_key["scenario_eval_run_id"]
    run_id = _string(config.get("run_id")) or f"unitree-g1-controlled-run-{_safe_id(job_id_value, 'job')}"
    operator_attestation = _attestation_from_config(config, "operator")
    hardware_owner_attestation = _attestation_from_config(config, "hardware_owner")
    safety_attestation = _attestation_from_config(config, "safety_reviewer")
    robot_team_attestation = _attestation_from_config(config, "robot_team_reviewer")
    refs = {name: _file_ref(path) for name, path in discovered_files.items()}

    def path_uri(name: str) -> str:
        return _string(_mapping(refs.get(name)).get("uri"))

    owner_evidence_presence = {
        evidence_id: bool(refs.get(evidence_id))
        for evidence_id in OWNER_EVIDENCE_CLAIM_REQUIREMENTS
    }
    attestations_signed = {
        "operator": _attestation_is_signed(operator_attestation),
        "hardware_owner": _attestation_is_signed(hardware_owner_attestation),
        "safety_reviewer": _attestation_is_signed(safety_attestation),
        "robot_team_reviewer": _attestation_is_signed(robot_team_attestation),
    }
    owner_evidence_ready = (
        ready
        and all(owner_evidence_presence.values())
        and all(attestations_signed.values())
    )
    owner_evidence_refs = {
        "physical_robot_run_manifest": str(output_root / "physical_robot_run_manifest.json"),
        "operator_log": path_uri("action_log"),
        "video_review": path_uri("robot_camera_video"),
        "timestamp_alignment": path_uri("timestamp_alignment"),
        "hardware_validation": path_uri("hardware_validation"),
        "contact_collision_log": path_uri("contact_collision_log"),
        "policy_metrics": path_uri("policy_metrics"),
        "robot_team_review": path_uri("robot_team_review"),
        **_mapping(config.get("owner_evidence_refs")),
    }

    physical_robot_run = {
        "schema_version": "physical_robot_run_package.v1",
        "status": status,
        "job_id": job_id_value,
        "run_id": run_id,
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "robot_serial_or_fleet_id": _string(config.get("robot_serial_or_fleet_id")),
        "site_or_lab_location_id": _string(config.get("site_or_lab_location_id")),
        "operator_attestation": operator_attestation,
        "hardware_owner_attestation": hardware_owner_attestation,
        "safety_reviewer_attestation": safety_attestation,
        "robot_team_review_attestation": robot_team_attestation,
        "start_time_utc": _string(config.get("start_time_utc")),
        "end_time_utc": _string(config.get("end_time_utc")),
        "policy_id": policy_id,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "scenario_variation_id": scenario_variation_id,
        "scenario_eval_run_id": scenario_eval_run_id,
        "accepted_anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
        "anchor_join_key": anchor_join_key,
        "allowed_task_set": config.get("allowed_task_set") or [],
        "operator_site_checklist": _mapping(config.get("operator_site_checklist")),
        "exclusion_and_abort_criteria": _mapping(config.get("exclusion_and_abort_criteria")),
        "robot_calibration_refs": _mapping(config.get("robot_calibration_refs")),
        "camera_calibration_refs": _mapping(config.get("camera_calibration_refs")),
        "action_log_refs": {
            "robot_state_log_uri": path_uri("robot_state_log"),
            "command_log_uri": path_uri("command_log"),
            "action_trace_uri": path_uri("action_log"),
        },
        "outcome_ledger_ref": str(output_root / "deployment_outcome_manifest.json"),
        "evidence_file_refs": refs,
        "required_owner_evidence": list(OWNER_EVIDENCE_CLAIM_REQUIREMENTS),
        "owner_evidence_presence": owner_evidence_presence,
        "attestations_signed": attestations_signed,
        "owner_evidence_ready_for_physical_claim": owner_evidence_ready,
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "proof_boundary": _proof_boundary(),
    }
    physical_path = output_root / "physical_robot_run_manifest.json"
    write_json(physical_path, physical_robot_run)

    deployment_records = []
    if physical_evidence_required:
        deployment_records.append(
            {
                "outcome_id": f"unitree-g1-outcome-{_safe_id(job_id_value, 'job')}",
                "job_id": job_id_value,
                "policy_id": policy_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_variation_instance_id": scenario_variation_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
                "anchor_join_key": anchor_join_key,
                "anchor_status": "accepted" if ready else "blocked",
                "review_status": "accepted" if ready else "blocked",
                "actual_status": _string(config.get("actual_status")) or "unknown",
                "actual_success": _bool(config.get("actual_success")),
                "actual_outcome": _mapping(config.get("actual_outcome")),
                "failure_mode_ids": config.get("failure_mode_ids") or [],
                "cycle_time_seconds": _number(config.get("cycle_time_seconds"), 0),
                "intervention_count": _number(config.get("intervention_count"), 0),
                "reviewer_decision": {
                    "safety_review_decision": _string(config.get("review_decision"))
                    or "not_reviewed",
                    "policy_review_decision": "accepted" if ready else "not_accepted",
                    "accepted_for_calibration": ready,
                },
                "evidence_refs": owner_evidence_refs if owner_evidence_ready else {},
                "owner_evidence_refs": owner_evidence_refs if owner_evidence_ready else {},
                "operator_attestation": operator_attestation if owner_evidence_ready else {},
                "hardware_owner_attestation": hardware_owner_attestation
                if owner_evidence_ready
                else {},
                "safety_reviewer_attestation": safety_attestation
                if owner_evidence_ready
                else {},
                "robot_team_review_attestation": robot_team_attestation
                if owner_evidence_ready
                else {},
                "owner_evidence_present": owner_evidence_ready,
                "required_owner_evidence": list(OWNER_EVIDENCE_CLAIM_REQUIREMENTS),
                "owner_evidence_presence": owner_evidence_presence,
                "attestations_signed": attestations_signed,
                "timestamps": {
                    "start_time_utc": _string(config.get("start_time_utc")),
                    "end_time_utc": _string(config.get("end_time_utc")),
                },
                "artifact_provenance": {
                    "physical_robot_run_manifest": str(physical_path),
                    "source_evidence_dir": str(evidence_root),
                    "evidence_file_refs": refs,
                },
            }
        )

    deployment_outcomes = {
        "schema_version": "deployment_outcome_manifest.v1",
        "status": status,
        "job_id": job_id_value,
        "records": deployment_records,
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "proof_boundary": _proof_boundary(),
    }
    deployment_path = output_root / "deployment_outcome_manifest.json"
    write_json(deployment_path, deployment_outcomes)

    pov_records = []
    if physical_evidence_required:
        pov_records.append(
            {
                "evidence_id": f"unitree-g1-pov-{_safe_id(job_id_value, 'job')}",
                "job_id": job_id_value,
                "policy_id": policy_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_variation_instance_id": scenario_variation_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "robot_camera_video_uri": path_uri("robot_camera_video"),
                "camera_mount_or_sensor_ids": config.get("camera_mount_or_sensor_ids")
                or ["unitree-g1-operator-pov"],
                "action_log_uri": path_uri("action_log"),
                "timestamp_alignment": path_uri("timestamp_alignment"),
                "owner_evidence_refs": {
                    "physical_robot_run_manifest": str(physical_path),
                    "deployment_outcome_manifest": str(deployment_path),
                },
                "operator_attestation": operator_attestation,
            }
        )

    real_robot_pov = {
        "schema_version": "real_robot_pov_manifest.v1",
        "status": status,
        "job_id": job_id_value,
        "run_id": run_id,
        "timestamp_alignment": path_uri("timestamp_alignment"),
        "records": pov_records,
        "claim_boundary": (
            "Real robot POV evidence is optional for sim-only runs and must not block "
            "simulator evaluation."
        ),
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "proof_boundary": _proof_boundary(),
    }
    pov_path = output_root / "real_robot_pov_manifest.json"
    write_json(pov_path, real_robot_pov)

    safety_package = {
        "schema_version": "reviewed_non_ranking_operational_claim_package.v1",
        "status": status,
        "job_id": job_id_value,
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "robot_id": _string(config.get("robot_serial_or_fleet_id")),
        "policy_id": policy_id,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "scenario_variation_id": scenario_variation_id,
        "reviewer_id": _string(config.get("safety_reviewer_id")),
        "accepted_safety_thresholds": _mapping(config.get("accepted_safety_thresholds")),
        "stop_conditions": config.get("stop_conditions")
        or [
            "loss_of_comms",
            "unexpected_human_entry",
            "fall_detected",
            "contact_force_exceeds_threshold",
            "operator_estop",
        ],
        "contact_or_collision_log_refs": [path_uri("contact_collision_log")],
        "physics_or_hardware_validation_refs": [path_uri("hardware_validation")],
        "review_decision": _string(config.get("review_decision")) or "not_reviewed",
        "review_timestamp_utc": _string(config.get("review_timestamp_utc")) or utc_now_iso(),
        "operator_attestation": safety_attestation,
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "proof_boundary": _proof_boundary(),
    }
    safety_path = output_root / "reviewed_non_ranking_operational_claim_package.json"
    write_json(safety_path, safety_package)
    legacy_safety_path = output_root / "reviewed_safety_validation_package.json"
    write_json(legacy_safety_path, {**safety_package, "schema_version": "reviewed_safety_validation_package.v1"})

    policy_package = {
        "schema_version": "robot_team_policy_package.v1",
        "status": status,
        "job_id": job_id_value,
        "policy_id": policy_id,
        "policy_owner": "Unitree G1 controlled proof operator",
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "policy_package": {
            "sim_controller_plugin": {
                "simulator_framework": "mujoco",
                "plugin_uri": "https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_mujoco",
                "expected_config": "g1.yaml",
                "execution_trace_refs": [path_uri("policy_execution_trace")],
                "metric_refs": [path_uri("policy_metrics")],
                "owner_acceptance_or_review": path_uri("robot_team_review"),
            },
            "recorded_action_trace": {
                "trace_manifest_uri": path_uri("policy_execution_trace"),
                "timestamp_alignment": path_uri("timestamp_alignment"),
            },
        },
        "scenario_variation_ids": [scenario_variation_id],
        "owner_attestation": robot_team_attestation,
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "proof_boundary": _proof_boundary(),
    }
    policy_path = output_root / "robot_team_policy_package.json"
    write_json(policy_path, policy_package)

    live_closure_evidence = {
        "schema_version": "live_robot_eval_closure_evidence.v1",
        "status": status,
        "job_id": job_id_value,
        "review_acceptance": {
            "status": "accepted" if ready else "blocked",
            "accepted": ready,
            "reviewer": _string(config.get("safety_reviewer_id")),
            "operator_attestation": safety_attestation,
        },
        "delivery": {
            "signed_urls": [_string(config.get("signed_customer_delivery_url"))],
            "storage_upload_performed": _bool(config.get("storage_upload_performed")),
            "entitlement_verified": _bool(config.get("entitlement_verified")),
        },
        "safety_contact_physics": {
            "physics_contact_validated": ready,
            "non_ranking_operational_claim_validated": ready,
            "rank_fidelity_result_proven": False,
            "non_ranking_operational_claim_uri_or_path": str(safety_path),
            "contact_validation_uri_or_path": path_uri("contact_collision_log"),
            "operator_attestation": safety_attestation,
        },
        "rights_privacy": {
            "status": _string(config.get("rights_privacy_status")) or "not_reviewed",
            "external_use_allowed": _bool(config.get("external_use_allowed")),
            "evidence_uri": str(physical_path),
        },
        "webapp_upstream": {
            "site_submission_id": context.get("site_submission_id"),
            "request_id": _string(config.get("production_webapp_request_id")),
            "buyer_request_id": context.get("buyer_request_id"),
            "capture_job_id": context.get("capture_job_id"),
            "pipeline_intake_request_id": _string(config.get("pipeline_intake_request_id")),
            "sync_status": _string(config.get("sync_status")) or "not_proven",
            "production_forward_url": _string(config.get("production_forward_url")),
            "request_timestamp_utc": _string(config.get("request_timestamp_utc")) or utc_now_iso(),
            "response_status_code": _string(config.get("webapp_response_status_code")),
        },
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "proof_boundary": _proof_boundary(),
    }
    closure_path = output_root / "live_eval_closure_evidence.json"
    write_json(closure_path, live_closure_evidence)

    anchor_request_packet = _controlled_anchor_request_packet(
        context=context,
        config=config,
        output_root=output_root,
        evidence_root=evidence_root,
        evidence_file_refs=refs,
        blockers=blockers,
        ready=ready,
        physical_evidence_required=physical_evidence_required,
    )
    anchor_request_path = output_root / "controlled_field_anchor_request_packet.json"
    write_json(anchor_request_path, anchor_request_packet)

    stage_script = output_root / "stage_assembled_g1_live_inputs.sh"
    write_text(
        stage_script,
        f"""#!/usr/bin/env bash
set -euo pipefail

ASSEMBLY_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
CAPTURE_ROOT="${{BLUEPRINT_CAPTURE_ROOT:-{root}}}"
MANIFEST_PATH="${{BLUEPRINT_LIVE_PIPELINE_MANIFEST:-$CAPTURE_ROOT/pipeline/live_pipeline_control_plane/live_pipeline_control_plane_manifest.json}}"
WEBAPP_JOB_REQUEST="${{BLUEPRINT_WEBAPP_JOB_REQUEST:-{context.get('job_request_path')}}}"

if [[ "${{BLUEPRINT_ALLOW_STAGING_G1_CONTROLLED_RUN_INPUTS:-}}" != "true" ]]; then
  echo "Set BLUEPRINT_ALLOW_STAGING_G1_CONTROLLED_RUN_INPUTS=true after reviewing assembled evidence." >&2
  exit 2
fi

python - "$ASSEMBLY_DIR/g1_controlled_run_evidence_assembly_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "ready_for_live_input_staging":
    raise SystemExit(f"G1 evidence assembly is not ready: {{payload.get('blockers')}}")
PY

blueprint-intake-live-pipeline-inputs \\
  --manifest-path "$MANIFEST_PATH" \\
  --webapp-job-request "$WEBAPP_JOB_REQUEST" \\
  --policy-package "$ASSEMBLY_DIR/robot_team_policy_package.json" \\
  --real-robot-pov "$ASSEMBLY_DIR/real_robot_pov_manifest.json" \\
  --deployment-outcomes "$ASSEMBLY_DIR/deployment_outcome_manifest.json" \\
  --live-closure-evidence "$ASSEMBLY_DIR/live_eval_closure_evidence.json" \\
  --stage-policy-package \\
  --stage-real-robot-pov \\
  --stage-deployment-outcomes \\
  --stage-live-closure-evidence \\
  --overwrite \\
  --output-path "$ASSEMBLY_DIR/live_pipeline_input_intake_audit.json" \\
  --staged-inputs-path "$CAPTURE_ROOT/pipeline/live_pipeline_staged_inputs.json"
""",
    )
    stage_script.chmod(0o755)

    manifest = {
        "schema_version": G1_CONTROLLED_RUN_EVIDENCE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_root": str(root),
        "evidence_dir": str(evidence_root),
        "input_config_path": str(config_path),
        "output_dir": str(output_root),
        "job_context": context,
        "evidence_file_refs": refs,
        "accepted_anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
        "required_exact_join_keys": list(CONTROLLED_ANCHOR_JOIN_KEYS),
        "anchor_join_key": anchor_join_key,
        "loose_or_inferred_matches_allowed_for_calibration": False,
        "physical_evidence_required": physical_evidence_required,
        "sim_only_beta_ranking_blocked": False,
        "required_owner_evidence": list(OWNER_EVIDENCE_CLAIM_REQUIREMENTS),
        "owner_evidence_presence": owner_evidence_presence,
        "attestations_signed": attestations_signed,
        "owner_evidence_ready_for_physical_claim": owner_evidence_ready,
        "file_blockers": file_blockers,
        "config_blockers": config_blockers,
        "content_blockers": content_blockers,
        "blockers": blockers if physical_evidence_required else [],
        "diagnostic_blockers": blockers,
        "artifacts": {
            "controlled_field_anchor_request_packet": str(anchor_request_path),
            "physical_robot_run_manifest": str(physical_path),
            "deployment_outcome_manifest": str(deployment_path),
            "real_robot_pov_manifest": str(pov_path),
            "reviewed_non_ranking_operational_claim_package": str(safety_path),
            "reviewed_safety_validation_package": str(legacy_safety_path),
            "robot_team_policy_package": str(policy_path),
            "live_closure_evidence": str(closure_path),
            "stage_script": str(stage_script),
        },
        "proof_boundary": {
            **_proof_boundary(),
            "sim_only_beta_ranking_blocked": False,
            "physical_evidence_not_requested_for_sim_only": not physical_evidence_required,
        },
    }
    manifest_path = output_root / "g1_controlled_run_evidence_assembly_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    template_parser = subparsers.add_parser("write-template")
    template_parser.add_argument("--capture-root", required=True, type=Path)
    template_parser.add_argument("--job-id")
    template_parser.add_argument("--output-path", required=True, type=Path)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--capture-root", required=True, type=Path)
    assemble_parser.add_argument("--evidence-dir", required=True, type=Path)
    assemble_parser.add_argument("--job-id")
    assemble_parser.add_argument("--input-config", type=Path)
    assemble_parser.add_argument("--output-dir", type=Path)
    assemble_parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "write-template":
        payload = write_g1_controlled_run_input_template(
            output_path=args.output_path,
            capture_root=args.capture_root,
            job_id=args.job_id,
        )
        print(json.dumps({"status": "template_written", "schema_version": payload["schema_version"]}))
        return 0
    if args.command == "assemble":
        manifest = assemble_g1_controlled_run_evidence(
            capture_root=args.capture_root,
            evidence_dir=args.evidence_dir,
            job_id=args.job_id,
            input_config_path=args.input_config,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "manifest": str(
                        Path(manifest["output_dir"])
                        / "g1_controlled_run_evidence_assembly_manifest.json"
                    ),
                }
            )
        )
        return 0 if manifest["status"] == "ready_for_live_input_staging" or not args.require_ready else 1
    parser.error("Provide a command: write-template or assemble")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
