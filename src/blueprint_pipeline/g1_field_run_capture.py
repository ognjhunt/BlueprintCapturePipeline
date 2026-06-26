"""Generate the operator-side Unitree G1 field-run capture kit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json, write_text
from .g1_controlled_proof_setup import (
    DEFAULT_LOW_COST_GPU_TYPE_ID,
    DEFAULT_POLICY_ID,
    DEFAULT_ROBOT_MAKE_MODEL,
    DEFAULT_ROBOT_PROFILE_ID,
    _job_context,
    _safe_id,
)


G1_FIELD_RUN_CAPTURE_KIT_SCHEMA_VERSION = "g1_field_run_capture_kit.v1"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _first_string(*values: Any, default: str = "") -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return default


def _webapp_route_proof_score(payload: Mapping[str, Any] | None) -> tuple[int, int, int, int, str]:
    if not payload:
        return (-1, -1, -1, -1, "")
    boundary = _mapping(payload.get("proof_boundary"))
    pipeline_forward = _mapping(payload.get("pipeline_forward"))
    pipeline_intake = _mapping(payload.get("pipeline_intake"))
    blockers = _as_list(pipeline_intake.get("input_blockers"))
    return (
        1 if payload.get("status") == "forwarded_to_pipeline_intake" else 0,
        1 if boundary.get("production_live_webapp_forwarding_proven") is True else 0,
        1 if pipeline_forward.get("accepted") is True else 0,
        1 if pipeline_intake.get("accepted") is True and not blockers else 0,
        _string(payload.get("generated_at")),
    )


def _select_webapp_route_forwarding_proof(root: Path) -> dict[str, Any] | None:
    proof_dir = root / "pipeline" / "webapp_route_forwarding_proof"
    default_path = proof_dir / "webapp_route_forwarding_proof.json"
    candidates = [default_path]
    if proof_dir.is_dir():
        for candidate in sorted(proof_dir.glob("webapp_route_forwarding_proof*.json")):
            if candidate not in candidates:
                candidates.append(candidate)

    best_payload = optional_read_json(default_path)
    best_score = _webapp_route_proof_score(best_payload)
    for candidate in candidates[1:]:
        payload = optional_read_json(candidate)
        score = _webapp_route_proof_score(payload)
        if score > best_score:
            best_payload = payload
            best_score = score
    return best_payload


def _webapp_route_prefill(proof: Mapping[str, Any] | None) -> dict[str, str]:
    if not proof:
        return {}
    boundary = _mapping(proof.get("proof_boundary"))
    webapp_route = _mapping(proof.get("webapp_route"))
    pipeline_forward = _mapping(proof.get("pipeline_forward"))
    pipeline_intake = _mapping(proof.get("pipeline_intake"))
    durable_store = _mapping(proof.get("durable_store"))
    firestore = _mapping(durable_store.get("firestore"))
    job_request = _mapping(proof.get("job_request"))
    accepted = bool(
        proof.get("status") == "forwarded_to_pipeline_intake"
        and boundary.get("production_live_webapp_forwarding_proven") is True
        and pipeline_forward.get("accepted") is True
        and pipeline_intake.get("accepted") is True
        and not _as_list(pipeline_intake.get("input_blockers"))
    )
    if not accepted:
        return {}
    request_id = _first_string(firestore.get("doc_id"), job_request.get("job_id"))
    route_url = _first_string(webapp_route.get("route_url"), webapp_route.get("remote_webapp_url"))
    return {
        "production_webapp_request_id": request_id,
        "pipeline_intake_request_id": _first_string(job_request.get("job_id"), request_id),
        "production_forward_url": route_url,
        "webapp_response_status_code": _string(webapp_route.get("http_status")),
        "sync_status": "succeeded",
    }


def _is_placeholder(value: Any) -> bool:
    text = _string(value)
    return not text or "<" in text or ">" in text


def _refresh_stale_webapp_input_fields(
    *,
    evidence_dir: Path,
    webapp_prefill: Mapping[str, str],
    context: Mapping[str, Any],
) -> str | None:
    if not webapp_prefill:
        return None
    input_path = evidence_dir / "g1_controlled_run_inputs.json"
    payload = optional_read_json(input_path)
    if not isinstance(payload, Mapping):
        return None
    updated = dict(payload)
    changed = False
    placeholder_packet = any(
        _is_placeholder(updated.get(field))
        for field in (
            "robot_serial_or_fleet_id",
            "site_or_lab_location_id",
            "operator_id",
            "hardware_owner_id",
        )
    )
    context_job_id = _string(context.get("job_id"))
    if context_job_id and placeholder_packet and _string(updated.get("job_id")) != context_job_id:
        updated["job_id"] = context_job_id
        changed = True
    for field in (
        "production_webapp_request_id",
        "pipeline_intake_request_id",
        "production_forward_url",
        "webapp_response_status_code",
        "sync_status",
    ):
        value = _string(webapp_prefill.get(field))
        if value and _is_placeholder(updated.get(field)):
            updated[field] = value
            changed = True
    if changed:
        write_json(input_path, updated)
    return str(input_path)


def _bound_job_context(context: Mapping[str, Any]) -> dict[str, Any]:
    job_id = _string(context.get("job_id")) or "robot-eval-job-id"
    return {
        "job_id": job_id,
        "job_request_found": context.get("job_request_found") is True,
        "job_request_source": _string(context.get("job_request_source")),
        "site_slug": _string(context.get("site_slug")),
        "site_submission_id": _string(context.get("site_submission_id")),
        "buyer_request_id": _string(context.get("buyer_request_id")),
        "capture_job_id": _string(context.get("capture_job_id")),
        "capture_id": _string(context.get("capture_id")),
        "task_id": _string(context.get("task_id")),
        "scenario_id": _string(context.get("scenario_id")),
        "scenario_variation_instance_id": _string(context.get("scenario_variation_instance_id")),
        "scenario_eval_run_id": _string(context.get("scenario_eval_run_id")),
        "robot_profile_id": _string(context.get("robot_profile_id")) or DEFAULT_ROBOT_PROFILE_ID,
    }


def _field_run_config(
    context: Mapping[str, Any],
    *,
    webapp_route_prefill: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    job_id = _string(context.get("job_id")) or "robot-eval-job-id"
    webapp = dict(webapp_route_prefill or {})
    return {
        "schema_version": "g1_field_run_capture_config.v1",
        "job_id": job_id,
        "job_context": _bound_job_context(context),
        "run_id": f"unitree-g1-controlled-run-{_safe_id(job_id, 'job')}",
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "robot_ip": "192.168.123.164",
        "robot_serial_or_fleet_id": "<unitree-g1-serial-or-fleet-id>",
        "site_or_lab_location_id": "<controlled-test-site-or-lab-id>",
        "operator_id": "<operator-id>",
        "hardware_owner_id": "<hardware-owner-id>",
        "safety_reviewer_id": "<safety-reviewer-id>",
        "robot_team_reviewer_id": "<robot-team-reviewer-id>",
        "max_duration_seconds": 120,
        "camera": {
            "source_uri": "<rtsp-or-device-or-zmq-camera-source>",
            "camera_mount_or_sensor_ids": ["<g1-head-or-body-camera-id>"],
            "output_filename": "robot_camera_video.mp4",
            "record_command": [
                "ffmpeg",
                "-y",
                "-t",
                "${BLUEPRINT_G1_RUN_DURATION_SECONDS:-120}",
                "-i",
                "${BLUEPRINT_G1_CAMERA_SOURCE}",
                "robot_camera_video.mp4",
            ],
        },
        "commands": {
            "required_env": [
                "BLUEPRINT_G1_CAMERA_SOURCE",
                "BLUEPRINT_G1_POLICY_COMMAND",
                "BLUEPRINT_G1_ACTION_LOG_COMMAND",
                "BLUEPRINT_G1_STATE_COMMAND",
                "BLUEPRINT_G1_CONTACT_COLLISION_COMMAND",
            ],
            "policy_command_env": "BLUEPRINT_G1_POLICY_COMMAND",
            "action_log_command_env": "BLUEPRINT_G1_ACTION_LOG_COMMAND",
            "state_sample_command_env": "BLUEPRINT_G1_STATE_COMMAND",
            "contact_collision_command_env": "BLUEPRINT_G1_CONTACT_COLLISION_COMMAND",
            "official_unitree_real_policy_templates": {
                "source_repo": "https://github.com/unitreerobotics/unitree_rl_gym",
                "python_deploy_real": (
                    "python deploy/deploy_real/deploy_real.py "
                    "${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} g1.yaml"
                ),
                "cpp_g1_deploy_run": "./g1_deploy_run ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0}",
                "claim_boundary": (
                    "Template only. Operator must run on the controlled G1 lab machine and "
                    "capture action/state/contact logs before readiness can be proven."
                ),
            },
            "blueprint_dds_logger_templates": {
                "action_log_jsonl": (
                    "python $KIT_DIR/record_g1_dds_logs.py --mode action "
                    "--net ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} "
                    "--duration ${BLUEPRINT_G1_RUN_DURATION_SECONDS:-120}"
                ),
                "robot_state_jsonl": (
                    "python $KIT_DIR/record_g1_dds_logs.py --mode state "
                    "--net ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} "
                    "--duration ${BLUEPRINT_G1_RUN_DURATION_SECONDS:-120}"
                ),
                "contact_collision_json": (
                    "python $KIT_DIR/record_g1_dds_logs.py --mode contact "
                    "--net ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} "
                    "--duration ${BLUEPRINT_G1_RUN_DURATION_SECONDS:-120}"
                ),
            },
            "review_command_templates": {
                "blocked_dry_review": (
                    "blueprint-review-g1-field-run-evidence "
                    "--evidence-dir $EVIDENCE_DIR"
                ),
                "accepted_review_after_human_signoff": (
                    "blueprint-review-g1-field-run-evidence "
                    "--evidence-dir $EVIDENCE_DIR --accept-safety --accept-policy --require-ready"
                ),
            },
        },
        "timestamp_alignment": {
            "max_alignment_error_ms": 100,
            "alignment_method": "operator_workstation_monotonic_clock",
            "camera_timebase": "ffmpeg_capture_wall_clock",
            "robot_timebase": "policy_command_wall_clock",
            "robot_action_log_source": "action_log.jsonl",
            "timestamp_alignment_output": "timestamp_alignment.json",
        },
        "real_robot_pov_contract": {
            "required": True,
            "camera_video_file": "robot_camera_video.mp4",
            "action_log_file": "action_log.jsonl",
            "timestamp_alignment_file": "timestamp_alignment.json",
            "max_alignment_error_ms": 100,
            "physical_source_required": True,
            "simulator_frames_count_as_real_pov": False,
            "test_fixture_policy": {
                "synthetic_media_allowed_in_unit_tests": True,
                "synthetic_media_can_upgrade_readiness": False,
                "required_fixture_marker": "synthetic_fixture_not_physical_robot_pov",
            },
        },
        "accepted_safety_thresholds": {
            "max_speed_mps": "<reviewed-threshold>",
            "min_human_clearance_m": "<reviewed-threshold>",
            "max_contact_force_n": "<reviewed-threshold>",
            "emergency_stop_required": True,
        },
        "stop_conditions": [
            "loss_of_comms",
            "unexpected_human_entry",
            "fall_detected",
            "contact_force_exceeds_threshold",
            "operator_estop",
        ],
        "policy": {
            "policy_id": DEFAULT_POLICY_ID,
            "source_repo": "https://github.com/unitreerobotics/unitree_rl_gym",
            "source_path": "deploy/deploy_mujoco",
            "fallback_source_repo": "https://github.com/unitreerobotics/unitree_rl_lab",
            "sim_bridge_repo": "https://github.com/unitreerobotics/unitree_mujoco",
            "source_commit_or_version": "<git-sha-or-release-reviewed-for-this-run>",
            "execution_mode": "physical_g1_controlled_run",
        },
        "runpod": {
            "preferred_gpu_type_id": DEFAULT_LOW_COST_GPU_TYPE_ID,
            "max_budget_usd": 2.0,
            "hard_timeout_seconds": 120,
        },
        "review_decision": "not_reviewed",
        "storage_upload_performed": False,
        "entitlement_verified": False,
        "signed_customer_delivery_url": "<signed-customer-delivery-url>",
        "rights_privacy_status": "not_reviewed",
        "external_use_allowed": False,
        "production_webapp_request_id": webapp.get(
            "production_webapp_request_id", "<production-webapp-request-id>"
        ),
        "pipeline_intake_request_id": webapp.get(
            "pipeline_intake_request_id", "<pipeline-intake-request-id>"
        ),
        "production_forward_url": webapp.get("production_forward_url", "<production-forward-url>"),
        "webapp_response_status_code": webapp.get("webapp_response_status_code", "<202>"),
        "sync_status": webapp.get("sync_status", "not_proven"),
    }


def _real_robot_pov_capture_contract(context: Mapping[str, Any], paths: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "g1_real_robot_pov_capture_contract.v1",
        "status": "physical_capture_required",
        "job_id": context["job_id"],
        "job_context": _bound_job_context(context),
        "evidence_dir": paths["evidence_dir"],
        "required_files": [
            "robot_camera_video.mp4",
            "timestamp_alignment.json",
            "action_log.jsonl",
        ],
        "required_alignment": {
            "max_alignment_error_ms": 100,
            "camera_timebase": "ffmpeg_capture_wall_clock",
            "robot_action_log_source": "action_log.jsonl",
        },
        "test_fixture_policy": {
            "synthetic_media_allowed_for_parser_tests": True,
            "synthetic_media_marker": "synthetic_fixture_not_physical_robot_pov",
            "synthetic_media_can_prove_real_robot_pov": False,
        },
        "proof_boundary": {
            "contract_is_not_pov_evidence": True,
            "simulator_pov_frames_count": False,
            "requires_physical_robot_camera_or_sensor_evidence": True,
            "real_robot_pov_evidence_proven": False,
        },
    }


def _safety_review_checklist(context: Mapping[str, Any], paths: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "g1_safety_review_checklist.v1",
        "status": "review_required",
        "job_id": context["job_id"],
        "job_context": _bound_job_context(context),
        "evidence_dir": paths["evidence_dir"],
        "required_thresholds": [
            "max_speed_mps",
            "min_human_clearance_m",
            "max_contact_force_n",
            "emergency_stop_required",
        ],
        "required_files": [
            "contact_collision_log.json",
            "hardware_validation.json",
            "robot_state_log.jsonl",
            "command_log.jsonl",
        ],
        "required_review_decisions": [
            "hardware_ready:true",
            "estop_verified:true",
            "hardware_validation.status:accepted",
            "contact_collision_log.max_contact_force_n <= accepted_safety_thresholds.max_contact_force_n",
            "explicit --accept-safety reviewer action",
        ],
        "stop_conditions": [
            "loss_of_comms",
            "unexpected_human_entry",
            "fall_detected",
            "contact_force_exceeds_threshold",
            "operator_estop",
        ],
        "proof_boundary": {
            "checklist_is_not_non_ranking_operational_claim": True,
            "requires_human_safety_review": True,
            "requires_physical_contact_or_hardware_logs": True,
            "non_ranking_operational_claim_validated": False,
        },
    }


def _capture_script(context: Mapping[str, Any], paths: Mapping[str, str]) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
EVIDENCE_DIR="${{BLUEPRINT_G1_EVIDENCE_DIR:-{paths['evidence_dir']}}}"
CONFIG_PATH="${{BLUEPRINT_G1_FIELD_RUN_CONFIG:-$KIT_DIR/g1_field_run_config.json}}"
RUN_DURATION="${{BLUEPRINT_G1_RUN_DURATION_SECONDS:-120}}"
POLICY_COMMAND="${{BLUEPRINT_G1_POLICY_COMMAND:-}}"
ACTION_LOG_COMMAND="${{BLUEPRINT_G1_ACTION_LOG_COMMAND:-}}"
STATE_COMMAND="${{BLUEPRINT_G1_STATE_COMMAND:-}}"
CONTACT_COMMAND="${{BLUEPRINT_G1_CONTACT_COLLISION_COMMAND:-}}"

if [[ "${{BLUEPRINT_ALLOW_G1_PHYSICAL_RUN:-}}" != "true" ]]; then
  echo "Set BLUEPRINT_ALLOW_G1_PHYSICAL_RUN=true only on the controlled G1 lab machine." >&2
  exit 2
fi
if [[ -z "${{BLUEPRINT_G1_CAMERA_SOURCE:-}}" ]]; then
  echo "Missing BLUEPRINT_G1_CAMERA_SOURCE for physical robot POV capture." >&2
  exit 2
fi
if [[ -z "$POLICY_COMMAND" ]]; then
  echo "Missing BLUEPRINT_G1_POLICY_COMMAND for non-default policy execution." >&2
  exit 2
fi
if [[ -z "$ACTION_LOG_COMMAND" ]]; then
  echo "Missing BLUEPRINT_G1_ACTION_LOG_COMMAND for physical robot action logs." >&2
  exit 2
fi
if [[ -z "$STATE_COMMAND" ]]; then
  echo "Missing BLUEPRINT_G1_STATE_COMMAND for physical robot state/action evidence." >&2
  exit 2
fi
if [[ -z "$CONTACT_COMMAND" ]]; then
  echo "Missing BLUEPRINT_G1_CONTACT_COLLISION_COMMAND for contact/collision evidence." >&2
  exit 2
fi

mkdir -p "$EVIDENCE_DIR"
cd "$EVIDENCE_DIR"

python - "$CONFIG_PATH" "$EVIDENCE_DIR/field_run_start.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(sys.argv[2]).write_text(json.dumps({{
    "schema_version": "g1_field_run_start.v1",
    "job_id": "{context['job_id']}",
    "config": config,
    "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}}, indent=2, sort_keys=True), encoding="utf-8")
PY

ffmpeg -y -t "$RUN_DURATION" -i "$BLUEPRINT_G1_CAMERA_SOURCE" robot_camera_video.mp4 > camera_stdout.txt 2> camera_stderr.txt &
CAMERA_PID=$!
bash -lc "$ACTION_LOG_COMMAND" > action_log.jsonl &
ACTION_LOG_PID=$!
bash -lc "$STATE_COMMAND" > robot_state_log.jsonl &
STATE_PID=$!
bash -lc "$CONTACT_COMMAND" > contact_collision_log.json &
CONTACT_PID=$!

python - "$POLICY_COMMAND" <<'PY'
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
command = sys.argv[1]
started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
Path("command_log.jsonl").write_text(json.dumps({{
    "kind": "policy_command_started",
    "command": command,
    "started_at_utc": started,
}}, sort_keys=True) + "\\n", encoding="utf-8")
completed = None
try:
    result = subprocess.run(
        shlex.split(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("BLUEPRINT_G1_POLICY_TIMEOUT_SECONDS", "150")),
    )
    completed = {{
        "kind": "policy_command_completed",
        "exit_code": result.returncode,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }}
    Path("policy_command_stdout.txt").write_text(result.stdout, encoding="utf-8")
    Path("policy_command_stderr.txt").write_text(result.stderr, encoding="utf-8")
except Exception as exc:  # pragma: no cover - exercised on field machine
    completed = {{
        "kind": "policy_command_failed",
        "error": str(exc),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }}
with Path("command_log.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(completed, sort_keys=True) + "\\n")
Path("policy_execution_trace.jsonl").write_text(json.dumps({{
    "policy_id": "{DEFAULT_POLICY_ID}",
    "command": command,
    **completed,
}}, sort_keys=True) + "\\n", encoding="utf-8")
PY

wait "$CAMERA_PID"
wait "$ACTION_LOG_PID"
wait "$STATE_PID"
wait "$CONTACT_PID"

python - "$CONFIG_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
alignment = config.get("timestamp_alignment", {{}})
now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
Path("timestamp_alignment.json").write_text(json.dumps({{
    "schema_version": "g1_timestamp_alignment.v1",
    "alignment_method": alignment.get("alignment_method", "operator_workstation_monotonic_clock"),
    "camera_timebase": alignment.get("camera_timebase", "ffmpeg_capture_wall_clock"),
    "robot_timebase": alignment.get("robot_timebase", "policy_command_wall_clock"),
    "robot_action_log_source": alignment.get("robot_action_log_source", "action_log.jsonl"),
    "max_alignment_error_ms": alignment.get("max_alignment_error_ms", config.get("max_alignment_error_ms", 100)),
    "generated_at_utc": now,
}}, indent=2, sort_keys=True), encoding="utf-8")
if not Path("robot_state_log.jsonl").exists():
    Path("robot_state_log.jsonl").write_text("", encoding="utf-8")
if not Path("contact_collision_log.json").exists():
    Path("contact_collision_log.json").write_text(json.dumps({{
        "schema_version": "g1_contact_collision_log.v1",
        "events": [],
        "max_contact_force_n": None,
        "status": "operator_review_required",
    }}, indent=2, sort_keys=True), encoding="utf-8")
Path("hardware_validation.json").write_text(json.dumps({{
    "schema_version": "g1_hardware_validation.v1",
    "hardware_ready": False,
    "estop_verified": False,
    "status": "operator_review_required",
    "generated_at_utc": now,
}}, indent=2, sort_keys=True), encoding="utf-8")
Path("policy_metrics.json").write_text(json.dumps({{
    "schema_version": "g1_policy_metrics.v1",
    "episode_count": 1,
    "success_rate": 0,
    "intervention_count": None,
    "status": "operator_review_required",
}}, indent=2, sort_keys=True), encoding="utf-8")
Path("robot_team_review.json").write_text(json.dumps({{
    "schema_version": "g1_robot_team_review.v1",
    "review_decision": "not_reviewed",
    "accepted": False,
    "reviewer_id": config.get("robot_team_reviewer_id"),
}}, indent=2, sort_keys=True), encoding="utf-8")
input_path = Path("g1_controlled_run_inputs.json")
if not input_path.exists():
    input_path.write_text(json.dumps({{
        "schema_version": "g1_controlled_run_inputs.v1",
        "job_id": "{context['job_id']}",
        "run_id": config.get("run_id"),
        "robot_serial_or_fleet_id": config.get("robot_serial_or_fleet_id"),
        "site_or_lab_location_id": config.get("site_or_lab_location_id"),
        "operator_id": config.get("operator_id"),
        "hardware_owner_id": config.get("hardware_owner_id"),
        "safety_reviewer_id": config.get("safety_reviewer_id"),
        "robot_team_reviewer_id": config.get("robot_team_reviewer_id"),
        "start_time_utc": json.loads(Path("field_run_start.json").read_text(encoding="utf-8")).get("started_at_utc"),
        "end_time_utc": now,
        "actual_status": "operator_review_required",
        "actual_success": False,
        "cycle_time_seconds": None,
        "intervention_count": None,
        "accepted_safety_thresholds": config.get("accepted_safety_thresholds", {{}}),
        "review_decision": "not_reviewed",
        "storage_upload_performed": False,
        "entitlement_verified": False,
        "signed_customer_delivery_url": config.get("signed_customer_delivery_url"),
        "rights_privacy_status": config.get("rights_privacy_status", "not_reviewed"),
        "external_use_allowed": False,
        "production_webapp_request_id": config.get("production_webapp_request_id"),
        "pipeline_intake_request_id": config.get("pipeline_intake_request_id"),
        "production_forward_url": config.get("production_forward_url"),
        "webapp_response_status_code": config.get("webapp_response_status_code"),
        "sync_status": config.get("sync_status", "not_proven"),
        "camera_mount_or_sensor_ids": config.get("camera", {{}}).get("camera_mount_or_sensor_ids", []),
        "operator_statement": "Operator must sign after reviewing physical G1 evidence files.",
        "hardware_owner_statement": "Hardware owner must sign after confirming the G1 identity and run.",
        "safety_reviewer_statement": "Safety reviewer must sign after accepting thresholds and logs.",
        "robot_team_review_statement": "Robot team reviewer must accept the non-default G1 policy package.",
    }}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "Field evidence written to $EVIDENCE_DIR"
echo "Run blueprint-review-g1-field-run-evidence after human safety/policy review before assembly."
"""


def _dds_logger_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def motor_cmd_record(cmd: Any) -> dict[str, Any]:
    return {
        "q": number(getattr(cmd, "q", None)),
        "qd": number(getattr(cmd, "qd", None)),
        "kp": number(getattr(cmd, "kp", None)),
        "kd": number(getattr(cmd, "kd", None)),
        "tau": number(getattr(cmd, "tau", None)),
    }


def motor_state_record(state: Any) -> dict[str, Any]:
    return {
        "q": number(getattr(state, "q", None)),
        "dq": number(getattr(state, "dq", None)),
        "tau_est": number(getattr(state, "tau_est", None)),
    }


def records_from_sequence(values: Any, mapper) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in list(values or []):
        result.append(mapper(item))
    return result


def load_unitree_sdk():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "g1_dds_logger_error.v1",
                    "status": "unitree_sdk2py_unavailable",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    return ChannelFactoryInitialize, ChannelSubscriber, LowCmdHG, LowStateHG


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Unitree G1 DDS evidence for Blueprint.")
    parser.add_argument("--mode", choices=["action", "state", "contact"], required=True)
    parser.add_argument("--net", required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--sample-period", type=float, default=0.1)
    args = parser.parse_args()

    channel_factory_initialize, channel_subscriber, low_cmd_type, low_state_type = load_unitree_sdk()
    channel_factory_initialize(0, args.net)
    latest: dict[str, Any] = {}

    def lowcmd_handler(msg: Any) -> None:
        latest["lowcmd"] = msg

    def lowstate_handler(msg: Any) -> None:
        latest["lowstate"] = msg

    if args.mode == "action":
        subscriber = channel_subscriber(TOPIC_LOWCMD, low_cmd_type)
        subscriber.Init(lowcmd_handler, 10)
    else:
        subscriber = channel_subscriber(TOPIC_LOWSTATE, low_state_type)
        subscriber.Init(lowstate_handler, 10)

    deadline = time.time() + args.duration
    max_tau_est = 0.0
    wrote_record = False
    while time.time() < deadline:
        if args.mode == "action" and latest.get("lowcmd") is not None:
            msg = latest["lowcmd"]
            print(
                json.dumps(
                    {
                        "schema_version": "g1_action_log_record.v1",
                        "kind": "action",
                        "captured_at_utc": utc_now(),
                        "action_id": f"lowcmd-{int(time.time() * 1000)}",
                        "motor_targets": records_from_sequence(
                            getattr(msg, "motor_cmd", []), motor_cmd_record
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            wrote_record = True
        elif args.mode == "state" and latest.get("lowstate") is not None:
            msg = latest["lowstate"]
            print(
                json.dumps(
                    {
                        "schema_version": "g1_robot_state_log_record.v1",
                        "kind": "state",
                        "captured_at_utc": utc_now(),
                        "motor_state": records_from_sequence(
                            getattr(msg, "motor_state", []), motor_state_record
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            wrote_record = True
        elif args.mode == "contact" and latest.get("lowstate") is not None:
            msg = latest["lowstate"]
            for item in list(getattr(msg, "motor_state", []) or []):
                tau = abs(number(getattr(item, "tau_est", None)) or 0.0)
                max_tau_est = max(max_tau_est, tau)
        time.sleep(args.sample_period)

    if args.mode == "contact":
        print(
            json.dumps(
                {
                    "schema_version": "g1_contact_collision_log.v1",
                    "status": "operator_review_required",
                    "events": [],
                    "max_contact_force_n": max_tau_est,
                    "proxy_metric": "max_abs_motor_tau_est",
                    "claim_boundary": "DDS torque estimate is a review input, not standalone off-scope validation.",
                    "captured_at_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    return 0 if wrote_record else 3


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _readme(context: Mapping[str, Any], paths: Mapping[str, str]) -> str:
    job_source = _string(context.get("job_request_source")) or "not_found"
    site_slug = _string(context.get("site_slug")) or "unknown"
    task_id = _string(context.get("task_id")) or "unknown"
    scenario_id = _string(context.get("scenario_id")) or "unknown"
    return f"""# Unitree G1 Field Run Capture Kit

This is the operator-side packet for the real controlled G1 run for job `{context['job_id']}`.

It is intentionally fail-closed:

- `BLUEPRINT_ALLOW_G1_PHYSICAL_RUN=true` is required before any capture command runs.
- Real POV requires `BLUEPRINT_G1_CAMERA_SOURCE`; simulator frames do not count.
- Non-default policy execution requires `BLUEPRINT_G1_POLICY_COMMAND`.
- Robot action evidence requires `BLUEPRINT_G1_ACTION_LOG_COMMAND`.
- Generated review files start as `operator_review_required` or `not_reviewed`; a human reviewer must update them before assembly can pass.

## Bound Job Context

- Job request source: `{job_source}`
- Site: `{site_slug}`
- Task: `{task_id}`
- Scenario: `{scenario_id}`

## Run

1. Review `{paths['real_robot_pov_capture_contract']}` and `{paths['safety_review_checklist']}`.
2. Fill `{paths['config']}` with robot ID, lab/site ID, operator IDs, safety thresholds, and reviewed policy source commit.
3. Export `BLUEPRINT_G1_CAMERA_SOURCE`, `BLUEPRINT_G1_POLICY_COMMAND`,
   `BLUEPRINT_G1_ACTION_LOG_COMMAND`, `BLUEPRINT_G1_STATE_COMMAND`, and
   `BLUEPRINT_G1_CONTACT_COLLISION_COMMAND`.
4. On the controlled lab machine, run `{paths['capture_script']}`. The script requires
   camera, policy, action-log, robot-state, and contact/collision commands before it starts.
5. Review/update the evidence files in `{paths['evidence_dir']}`.
6. Run `blueprint-review-g1-field-run-evidence --evidence-dir {paths['evidence_dir']}` to see review blockers.
7. After human safety and policy signoff, run `blueprint-review-g1-field-run-evidence --evidence-dir {paths['evidence_dir']} --accept-safety --accept-policy --require-ready`.
8. Run `{paths['assemble_script']}` from the parent G1 proof packet.

Official default policy candidates:

- Unitree RL Gym: https://github.com/unitreerobotics/unitree_rl_gym
- Unitree RL Lab: https://github.com/unitreerobotics/unitree_rl_lab
- Unitree MuJoCo bridge: https://github.com/unitreerobotics/unitree_mujoco

Official Unitree real-deploy command templates:

- `python deploy/deploy_real/deploy_real.py ${{BLUEPRINT_G1_NET_INTERFACE:-enp3s0}} g1.yaml`
- `./g1_deploy_run ${{BLUEPRINT_G1_NET_INTERFACE:-enp3s0}}`

LeRobot also documents physical G1 camera and teleoperation commands, including the fixed Ethernet robot IP and G1 camera wiring, but using it still requires a real robot run and owner review.
"""


def build_g1_field_run_capture_kit(
    *,
    capture_root: str | Path,
    job_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    context = _job_context(root, job_id)
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "pipeline" / "g1_controlled_proof_setup" / "field_run_capture_kit"
    )
    ensure_dir(output_root)
    evidence_dir = output_root.parent / "physical_g1_evidence_drop"
    ensure_dir(evidence_dir)
    webapp_prefill = _webapp_route_prefill(_select_webapp_route_forwarding_proof(root))
    refreshed_input_path = _refresh_stale_webapp_input_fields(
        evidence_dir=evidence_dir,
        webapp_prefill=webapp_prefill,
        context=context,
    )
    paths = {
        "config": str(output_root / "g1_field_run_config.json"),
        "capture_script": str(output_root / "run_g1_field_capture.sh"),
        "dds_logger_script": str(output_root / "record_g1_dds_logs.py"),
        "readme": str(output_root / "README.md"),
        "evidence_manifest": str(output_root / "expected_evidence_files.json"),
        "real_robot_pov_capture_contract": str(output_root / "real_robot_pov_capture_contract.json"),
        "safety_review_checklist": str(output_root / "safety_review_checklist.json"),
        "evidence_dir": str(evidence_dir),
        "review_manifest": str(evidence_dir / "g1_field_run_review_manifest.json"),
        "assemble_script": str(output_root.parent / "assemble_g1_evidence.sh"),
        "refreshed_input_config": refreshed_input_path,
    }
    write_json(Path(paths["config"]), _field_run_config(context, webapp_route_prefill=webapp_prefill))
    write_json(
        Path(paths["evidence_manifest"]),
        {
            "schema_version": "g1_field_run_expected_evidence_files.v1",
            "status": "operator_capture_required",
            "canonical_evidence_dir": str(evidence_dir),
            "required_files": [
                "robot_camera_video.mp4",
                "timestamp_alignment.json",
                "action_log.jsonl",
                "robot_state_log.jsonl",
                "command_log.jsonl",
                "contact_collision_log.json",
                "hardware_validation.json",
                "policy_execution_trace.jsonl",
                "policy_metrics.json",
                "robot_team_review.json",
                "g1_controlled_run_inputs.json",
            ],
            "required_live_commands": [
                "BLUEPRINT_G1_CAMERA_SOURCE",
                "BLUEPRINT_G1_POLICY_COMMAND",
                "BLUEPRINT_G1_ACTION_LOG_COMMAND",
                "BLUEPRINT_G1_STATE_COMMAND",
                "BLUEPRINT_G1_CONTACT_COLLISION_COMMAND",
            ],
            "contracts": {
                "real_robot_pov_capture_contract": paths["real_robot_pov_capture_contract"],
                "safety_review_checklist": paths["safety_review_checklist"],
            },
        },
    )
    write_json(
        Path(paths["real_robot_pov_capture_contract"]),
        _real_robot_pov_capture_contract(context, paths),
    )
    write_json(Path(paths["safety_review_checklist"]), _safety_review_checklist(context, paths))
    write_text(Path(paths["capture_script"]), _capture_script(context, paths))
    Path(paths["capture_script"]).chmod(0o755)
    write_text(Path(paths["dds_logger_script"]), _dds_logger_script())
    Path(paths["dds_logger_script"]).chmod(0o755)
    write_text(Path(paths["readme"]), _readme(context, paths))
    manifest = {
        "schema_version": G1_FIELD_RUN_CAPTURE_KIT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "field_run_capture_ready_operator_inputs_required",
        "capture_root": str(root),
        "output_dir": str(output_root),
        "evidence_dir": str(evidence_dir),
        "job_context": context,
        "default_robot": {
            "make_model": DEFAULT_ROBOT_MAKE_MODEL,
            "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
            "proof_target": "physical Unitree G1 controlled run",
        },
        "artifacts": paths,
        "proof_boundary": {
            "kit_is_not_physical_robot_proof": True,
            "requires_real_g1_hardware": True,
            "requires_real_robot_pov": True,
            "requires_human_safety_review": True,
            "public_claim_upgrade_allowed": False,
        },
    }
    manifest_path = output_root / "g1_field_run_capture_kit_manifest.json"
    write_json(manifest_path, manifest)
    manifest["artifacts"]["manifest"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    manifest = build_g1_field_run_capture_kit(
        capture_root=args.capture_root,
        job_id=args.job_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": manifest["status"], "manifest": manifest["artifacts"]["manifest"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
