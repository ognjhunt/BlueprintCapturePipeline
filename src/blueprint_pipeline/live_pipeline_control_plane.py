"""Always-on live pipeline control-plane runner.

The control plane is intentionally thin. It audits local/live readiness, then
optionally consumes a WebApp-style robot-eval job request inbox through the
existing deterministic orchestrator. It does not promote proof claims and it
does not turn on simulator, vision-labeling, delivery, or live agent operators
unless the caller supplies the matching CLI and environment gates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV, LIVE_CODEX_SDK_ENV
from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .live_pipeline_setup import (
    CONTROL_PLANE_NOT_PROOF,
    build_live_pipeline_setup_manifest,
)
from .robot_eval_job_orchestrator import (
    CPU_BACKENDS,
    CLAIM_BOUNDARY,
    AgentsSdkRobotEvalJobAdapter,
    FakeRobotEvalJobAgentAdapter,
    RobotEvalJobAgentAdapter,
    run_robot_eval_job_request_inbox,
)
from .safe_env import load_env_files


LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION = "blueprint_live_pipeline_control_plane_run.v1"
LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION = (
    "blueprint_live_pipeline_external_input_packet.v1"
)

WEBAPP_UPSTREAM_REQUIRED_FIELDS = (
    "site_submission_id",
    "request_id",
    "buyer_request_id",
    "capture_job_id",
)

WEBAPP_UPSTREAM_ACCEPTED_SOURCES = (
    "capture_descriptor.json",
    "raw/manifest.json",
    "pipeline/opportunity_handoff.json",
    "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX robot_eval_job_request.v1 files for scheduling",
)

ARENA_RESULT_ARTIFACT_NAMES = (
    "rollout_manifest.json",
    "shard_manifest.json",
    "artifact_manifest.json",
    "metrics.json",
    "results.json",
    "any additional owner-system *.json result artifacts",
)

WEBAPP_JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
WEBAPP_JOB_REQUEST_QUEUE_CONTRACT = "robot_eval_job_request_inbox.v1"

CAPTURE_ROOT_ENV = "BLUEPRINT_PIPELINE_CAPTURE_ROOT"
JOB_REQUEST_INBOX_ENV = "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX"
PACKAGE_DIR_ENV = "BLUEPRINT_PIPELINE_PACKAGE_DIR"
ARENA_RESULTS_DIR_ENV = "BLUEPRINT_ARENA_RESULTS_DIR"
SIMULATOR_AUDIT_COMMAND_ENV = "BLUEPRINT_SIMULATOR_COMMAND"
VISION_LABELING_COMMAND_ENV = "BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND"
DELIVERY_COMMAND_ENV = "BLUEPRINT_PACKAGE_DELIVERY_UPLOAD_COMMAND"
CONTROL_PLANE_AGENT_MODE_ENV = "BLUEPRINT_CONTROL_PLANE_AGENT_MODE"
CONTROL_PLANE_ARENA_OPERATOR_MODE_ENV = "BLUEPRINT_CONTROL_PLANE_ARENA_OPERATOR_MODE"
CONTROL_PLANE_OUTPUT_PATH_ENV = "BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH"
CONTROL_PLANE_ALLOW_LIVE_AGENT_OPERATOR_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_AGENT_OPERATOR"
CONTROL_PLANE_ALLOW_DIGITALOCEAN_READ_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_DIGITALOCEAN_READ"
CONTROL_PLANE_ALLOW_GPU_PROVISIONING_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_GPU_PROVISIONING"
CONTROL_PLANE_ALLOW_SIMULATOR_EXECUTION_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_SIMULATOR_EXECUTION"
CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_CPU_PREFLIGHT"
CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_RENDER_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_RENDER"
CONTROL_PLANE_ALLOW_TRAINING_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_TRAINING"
CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING_ENV = (
    "BLUEPRINT_CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING"
)
CONTROL_PLANE_ALLOW_DELIVERY_UPLOAD_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_DELIVERY_UPLOAD"
CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK"
CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK"
CONTROL_PLANE_SIMULATOR_ENV = "BLUEPRINT_CONTROL_PLANE_SIMULATOR"
CONTROL_PLANE_PROVISIONER_ENV = "BLUEPRINT_CONTROL_PLANE_PROVISIONER"
CONTROL_PLANE_TIMEOUT_SECONDS_ENV = "BLUEPRINT_CONTROL_PLANE_TIMEOUT_SECONDS"
ISAAC_LAB_ARENA_COMMAND_ENV = "BLUEPRINT_ISAAC_LAB_ARENA_COMMAND"
DIGITALOCEAN_DROPLET_NAME_ENV = "BLUEPRINT_DIGITALOCEAN_DROPLET_NAME"
DIGITALOCEAN_DROPLET_IP_ENV = "BLUEPRINT_DIGITALOCEAN_DROPLET_IP"

SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "WORLDLABS_API_KEY",
    "PIPELINE_SYNC_TOKEN",
    "DIGITALOCEAN_ACCESS_TOKEN",
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return _truthy(os.getenv(name))


def _env_value(name: str, explicit: str | Path | None = None) -> str | None:
    value = _string(explicit)
    if value:
        return value
    env_value = _string(os.getenv(name))
    return env_value or None


def _env_int(name: str, default: int) -> int:
    value = _string(os.getenv(name))
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _restore_env(original_env: Mapping[str, str]) -> None:
    for key in list(os.environ):
        if key not in original_env:
            os.environ.pop(key, None)
    for key, value in original_env.items():
        os.environ[key] = value


def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _output_path(capture_root: Path | None, output_path: str | Path | None) -> Path:
    if output_path:
        return Path(output_path).resolve()
    if capture_root:
        return (
            capture_root
            / "pipeline"
            / "live_pipeline_control_plane"
            / "live_pipeline_control_plane_manifest.json"
        )
    return Path.cwd().resolve() / "live_pipeline_control_plane_manifest.json"


def _agent_adapter_from_mode(mode: str, *, allow_live_operator: bool) -> RobotEvalJobAgentAdapter | None:
    if mode == "fake":
        return FakeRobotEvalJobAgentAdapter()
    if mode == "agents-sdk":
        return AgentsSdkRobotEvalJobAdapter(allow_live_operator=allow_live_operator)
    return None


def _parse_simulator_commands(values: Sequence[str] | None) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values or []:
        text = _string(value)
        if not text:
            continue
        framework, sep, command = text.partition("=")
        framework = framework.strip()
        command = command.strip()
        if not sep or not framework or not command:
            raise ValueError("simulator commands must be formatted as <framework>=<command>")
        commands[framework] = command
    env_command = _string(os.getenv(ISAAC_LAB_ARENA_COMMAND_ENV))
    if env_command and "isaac_lab_arena" not in commands:
        commands["isaac_lab_arena"] = env_command
    return commands


def _secret_values() -> List[str]:
    values: List[str] = []
    for name in SECRET_ENV_NAMES:
        value = _string(os.getenv(name))
        if len(value) >= 8 and value.lower() not in {"placeholder", "changeme", "example"}:
            values.append(value)
    return values


def _manifest_leaks_secret(manifest: Mapping[str, Any], secret_values: Sequence[str]) -> bool:
    if not secret_values:
        return False
    serialized = json.dumps(manifest, sort_keys=True)
    return any(value in serialized for value in secret_values)


def _inbox_status_not_configured(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_configured",
        "processed": False,
        "processed_count": 0,
        "blockers": [reason],
        "manifest_path": None,
    }


def _overall_status(
    *,
    capture_root: Path | None,
    inbox: Mapping[str, Any],
    setup_manifest: Mapping[str, Any],
) -> str:
    if capture_root is None:
        return "blocked"
    if inbox.get("status") == "completed":
        return "processed_jobs"
    if inbox.get("status") == "empty":
        return "waiting_for_jobs"
    if setup_manifest.get("status") == "ready_for_live_external_execution":
        return "ready_for_live_external_execution"
    if setup_manifest.get("status") == "local_ready_live_external_blocked":
        return "local_ready_live_external_blocked"
    return "blocked"


def _setup_section_ready(setup_manifest: Mapping[str, Any], section_name: str) -> bool:
    sections = setup_manifest.get("sections") if isinstance(setup_manifest.get("sections"), Mapping) else {}
    section = sections.get(section_name) if isinstance(sections, Mapping) else {}
    if not isinstance(section, Mapping):
        return False
    status = _string(section.get("status"))
    return bool(section.get("ready")) or status.startswith("ready")


def _control_plane_next_inputs_needed(
    *,
    capture_root: Path | None,
    job_request_inbox: Path | None,
    setup_manifest: Mapping[str, Any],
    webapp_upstream_truth_ready: bool | None = None,
) -> List[str]:
    next_inputs: List[str] = []
    webapp_truth_ready = (
        bool(webapp_upstream_truth_ready)
        if webapp_upstream_truth_ready is not None
        else _setup_section_ready(setup_manifest, "webapp_upstream_truth")
    )
    if capture_root is None:
        next_inputs.append("Set BLUEPRINT_PIPELINE_CAPTURE_ROOT to a real capture root.")
    elif not webapp_truth_ready:
        next_inputs.append(
            "Provide a real WebApp capture root with site, request, buyer, and capture job IDs."
        )
    if job_request_inbox is None:
        next_inputs.append("Set BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX to the WebApp job request inbox path.")
    if not _setup_section_ready(setup_manifest, "real_arena_execution"):
        next_inputs.append(
            "Provide a real owner-system Isaac Lab-Arena command or result directory before "
            "claiming simulator execution."
        )
    if not _setup_section_ready(setup_manifest, "rollout_vision_labeling"):
        next_inputs.append(
            "Provide a vision-labeling command and gate before model labels can be generated."
        )
    if not _setup_section_ready(setup_manifest, "delivery_upload"):
        next_inputs.append(
            "Provide a delivery command and gate before package uploads or signed links can be "
            "created."
        )
    if not (
        _setup_section_ready(setup_manifest, "live_agents_operator")
        and _setup_section_ready(setup_manifest, "live_codex_operator")
    ):
        next_inputs.append(
            "Configure gated Agents SDK and Codex SDK or host-OAuth operator credentials before "
            "running live repo operators."
        )
    return next_inputs


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _setup_section(setup_manifest: Mapping[str, Any], section_name: str) -> Dict[str, Any]:
    sections = _as_mapping(setup_manifest.get("sections"))
    return _as_mapping(sections.get(section_name))


def _section_blockers(section: Mapping[str, Any]) -> List[str]:
    blockers = section.get("blockers")
    return list(blockers) if isinstance(blockers, list) else []


def _field_value_from_sources(
    payload: Mapping[str, Any],
    field: str,
    sources: Sequence[Mapping[str, Any]],
) -> str | None:
    for source in sources:
        value = _string(source.get(field))
        if value:
            return value
    if field == "request_id":
        owner_system = _as_mapping(payload.get("owner_system"))
        value = _string(owner_system.get("request_id"))
        if value:
            return value
    return None


def _request_from_webapp_payload(payload: Mapping[str, Any]) -> Dict[str, Any] | None:
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        request = payload.get("job_request")
        if isinstance(request, Mapping) and request.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
            return dict(request)
        return None
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return dict(payload)
    return None


def _path_matches_configured_capture_root(
    request_capture_root: str | None,
    capture_root: Path | None,
) -> bool:
    if not request_capture_root or capture_root is None:
        return False
    try:
        return Path(request_capture_root).resolve() == capture_root.resolve()
    except (OSError, RuntimeError):
        return False


def _webapp_job_request_inbox_truth(
    *,
    inbox_path: Path | None,
    capture_root: Path | None,
) -> Dict[str, Any]:
    if inbox_path is None:
        return {
            "status": "not_configured",
            "ready": False,
            "inbox_path": None,
            "blockers": ["job_request_inbox_not_provided"],
            "request_count": 0,
            "accepted_request_count": 0,
            "accepted_request_ids": [],
            "candidates": [],
        }
    if not inbox_path.is_dir():
        return {
            "status": "blocked",
            "ready": False,
            "inbox_path": str(inbox_path),
            "blockers": ["job_request_inbox_missing"],
            "request_count": 0,
            "accepted_request_count": 0,
            "accepted_request_ids": [],
            "candidates": [],
        }
    candidates: List[Dict[str, Any]] = []
    invalid_json_count = 0
    for request_path in sorted(inbox_path.glob("*.json")):
        if request_path.name.startswith("."):
            continue
        try:
            payload = read_json_any(request_path)
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_json_count += 1
            continue
        if not isinstance(payload, Mapping):
            continue
        request = _request_from_webapp_payload(payload)
        if request is None:
            continue
        source = _as_mapping(request.get("source"))
        site_package = _as_mapping(request.get("site_package"))
        top_level_sources = (request, source)
        fields_present: Dict[str, bool] = {}
        for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS:
            fields_present[field] = bool(_field_value_from_sources(request, field, top_level_sources))
        request_capture_root = _string(site_package.get("capture_root")) or None
        capture_root_matches = _path_matches_configured_capture_root(request_capture_root, capture_root)
        missing_fields = [
            field for field, present in fields_present.items() if not present
        ]
        accepted = capture_root_matches and not missing_fields
        candidates.append(
            {
                "path": str(request_path),
                "schema_version": _string(request.get("schema_version")) or None,
                "job_id": _string(request.get("job_id")) or None,
                "fields_present": fields_present,
                "missing_fields": missing_fields,
                "capture_root_matches_control_plane": capture_root_matches,
                "request_capture_root_configured": bool(request_capture_root),
                "accepted_as_webapp_truth": accepted,
            }
        )
    accepted_candidates = [
        candidate for candidate in candidates if candidate["accepted_as_webapp_truth"]
    ]
    blockers: List[str] = []
    if not candidates:
        blockers.append("no_robot_eval_job_request_v1_files")
    if candidates and not accepted_candidates:
        if not any(candidate["capture_root_matches_control_plane"] for candidate in candidates):
            blockers.append("no_job_request_matches_configured_capture_root")
        if any(candidate["missing_fields"] for candidate in candidates):
            blockers.append("job_request_missing_required_webapp_ids")
    warnings = ["invalid_json_files_ignored"] if invalid_json_count else []
    return {
        "status": "ready" if accepted_candidates else "blocked",
        "ready": bool(accepted_candidates),
        "inbox_path": str(inbox_path),
        "blockers": blockers,
        "warnings": warnings,
        "invalid_json_count": invalid_json_count,
        "request_count": len(candidates),
        "accepted_request_count": len(accepted_candidates),
        "accepted_request_ids": [
            candidate["job_id"] for candidate in accepted_candidates if candidate.get("job_id")
        ],
        "candidates": candidates[:20],
        "truncated_candidates": len(candidates) > 20,
        "proof_boundary": (
            "Queued WebApp job requests prove upstream linkage only when they contain real "
            "WebApp IDs and point at the configured capture root."
        ),
    }


def _missing_webapp_fields(section: Mapping[str, Any]) -> List[str]:
    fields_present = _as_mapping(section.get("fields_present"))
    if not fields_present:
        return list(WEBAPP_UPSTREAM_REQUIRED_FIELDS)
    missing = [
        field
        for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS
        if not bool(fields_present.get(field))
    ]
    return missing


def _input_packet_status(
    *,
    required_inputs: Sequence[Mapping[str, Any]],
    enablement_inputs: Sequence[Mapping[str, Any]],
) -> str:
    if required_inputs:
        return "waiting_for_external_inputs"
    if enablement_inputs:
        return "core_external_inputs_ready_enablement_missing"
    return "all_external_inputs_configured"


def _gateable_section_input(
    *,
    input_id: str,
    title: str,
    section: Mapping[str, Any],
    required_env: Sequence[str],
    command_env: str | None = None,
    accepted_artifact: str | None = None,
    proof_boundary: str,
) -> Dict[str, Any] | None:
    if bool(section.get("ready")):
        return None
    command = _as_mapping(section.get("command"))
    configured = bool(command.get("configured")) if command else None
    return {
        "id": input_id,
        "title": title,
        "status": _string(section.get("status")) or "blocked",
        "current_blockers": _section_blockers(section),
        "required_env": list(required_env),
        "command_env": command_env,
        "command_configured": configured,
        "accepted_artifact": accepted_artifact,
        "proof_boundary": proof_boundary,
    }


def _build_external_input_packet(
    *,
    generated_at: str,
    capture_root: Path | None,
    job_request_inbox: Path | None,
    package_dir: Path | None,
    arena_results_dir: Path | None,
    output_path: Path,
    setup_manifest_path: Path,
    setup_manifest: Mapping[str, Any],
    inbox_run: Mapping[str, Any],
    webapp_inbox_truth: Mapping[str, Any],
) -> Dict[str, Any]:
    webapp_section = _setup_section(setup_manifest, "webapp_upstream_truth")
    arena_section = _setup_section(setup_manifest, "real_arena_execution")
    webapp_truth_ready = _setup_section_ready(
        setup_manifest,
        "webapp_upstream_truth",
    ) or bool(webapp_inbox_truth.get("ready"))
    required_inputs: List[Dict[str, Any]] = []
    if not webapp_truth_ready:
        required_inputs.append(
            {
                "id": "webapp_upstream_truth",
                "title": "Real WebApp capture/job IDs",
                "status": _string(webapp_section.get("status")) or "blocked",
                "required_fields": list(WEBAPP_UPSTREAM_REQUIRED_FIELDS),
                "missing_fields": _missing_webapp_fields(webapp_section),
                "accepted_sources": list(WEBAPP_UPSTREAM_ACCEPTED_SOURCES),
                "configured_capture_root": str(capture_root) if capture_root else None,
                "configured_job_request_inbox": str(job_request_inbox)
                if job_request_inbox
                else None,
                "webapp_inbox_truth": {
                    "status": webapp_inbox_truth.get("status"),
                    "request_count": webapp_inbox_truth.get("request_count"),
                    "accepted_request_count": webapp_inbox_truth.get("accepted_request_count"),
                    "blockers": webapp_inbox_truth.get("blockers", []),
                },
                "current_blockers": _section_blockers(webapp_section),
                "proof_boundary": (
                    "IDs prove upstream WebApp linkage only when sourced from real capture/job "
                    "records; placeholders do not satisfy production proof."
                ),
            }
        )
    if not _setup_section_ready(setup_manifest, "real_arena_execution"):
        required_inputs.append(
            {
                "id": "isaac_lab_arena_owner_evidence",
                "title": "Owner-system Isaac Lab-Arena evidence",
                "status": _string(arena_section.get("status")) or "blocked",
                "accepted_paths": [
                    {
                        "kind": "owner_result_directory",
                        "env": ARENA_RESULTS_DIR_ENV,
                        "configured_path": str(arena_results_dir) if arena_results_dir else None,
                        "required_artifacts": list(ARENA_RESULT_ARTIFACT_NAMES),
                        "accepted_section_status": "ready_for_result_ingest",
                    },
                    {
                        "kind": "gated_simulator_command",
                        "gate_env": "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
                        "audit_command_env": SIMULATOR_AUDIT_COMMAND_ENV,
                        "isaac_lab_arena_command_env": ISAAC_LAB_ARENA_COMMAND_ENV,
                        "accepted_section_status": "ready",
                    },
                ],
                "current_blockers": _section_blockers(arena_section),
                "proof_boundary": (
                    "Arena results or commands are ingest/execution inputs only; robot policy, "
                    "contact, safety, and readiness claims require accepted owner-system evidence."
                ),
            }
        )

    enablement_inputs: List[Dict[str, Any]] = []
    for item in (
        _gateable_section_input(
            input_id="rollout_vision_labeling",
            title="Rollout vision labeling hook",
            section=_setup_section(setup_manifest, "rollout_vision_labeling"),
            required_env=("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING",),
            command_env=VISION_LABELING_COMMAND_ENV,
            accepted_artifact="rollout_vision_labels.command.json",
            proof_boundary="Model labels remain review-required support evidence.",
        ),
        _gateable_section_input(
            input_id="delivery_upload",
            title="Package delivery hook",
            section=_setup_section(setup_manifest, "delivery_upload"),
            required_env=("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD",),
            command_env=DELIVERY_COMMAND_ENV,
            accepted_artifact="delivery_upload.command.json or signed_access.command.json",
            proof_boundary="Delivery artifacts do not upgrade robot or simulator proof claims.",
        ),
        _gateable_section_input(
            input_id="live_agents_operator",
            title="Gated Agents SDK operator",
            section=_setup_section(setup_manifest, "live_agents_operator"),
            required_env=(LIVE_AGENTS_SDK_ENV, "OPENAI_API_KEY"),
            proof_boundary=(
                "Agents may choose or summarize next steps but cannot mutate proof booleans."
            ),
        ),
        _gateable_section_input(
            input_id="live_codex_operator",
            title="Gated Codex SDK or Codex CLI host-OAuth operator",
            section=_setup_section(setup_manifest, "live_codex_operator"),
            required_env=(LIVE_CODEX_SDK_ENV, "BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH"),
            proof_boundary=(
                "Codex operators may edit or inspect repo artifacts only behind explicit gates."
            ),
        ),
    ):
        if item is not None:
            enablement_inputs.append(item)

    packet = {
        "schema_version": LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": _input_packet_status(
            required_inputs=required_inputs,
            enablement_inputs=enablement_inputs,
        ),
        "configured_paths": {
            "capture_root": str(capture_root) if capture_root else None,
            "job_request_inbox": str(job_request_inbox) if job_request_inbox else None,
            "package_dir": str(package_dir) if package_dir else None,
            "arena_results_dir": str(arena_results_dir) if arena_results_dir else None,
            "control_plane_manifest_path": str(output_path),
            "setup_manifest_path": str(setup_manifest_path),
            "inbox_run_manifest_path": inbox_run.get("manifest_path"),
        },
        "webapp_upstream_truth": {
            "ready": webapp_truth_ready,
            "capture_root_section_status": _string(webapp_section.get("status")) or "blocked",
            "job_request_inbox_status": webapp_inbox_truth.get("status"),
            "accepted_request_ids": webapp_inbox_truth.get("accepted_request_ids", []),
        },
        "required_inputs": required_inputs,
        "enablement_inputs": enablement_inputs,
        "example_robot_eval_job_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "REPLACE_WITH_WEBAPP_JOB_ID",
            "customer": {
                "id": "REPLACE_WITH_ROBOT_TEAM_ID",
                "name": "REPLACE_WITH_ROBOT_TEAM_NAME",
            },
            "site_package": {
                "capture_root": "REPLACE_WITH_CAPTURE_ROOT",
                "site_id": "REPLACE_WITH_SITE_ID",
                "package_uri": "REPLACE_WITH_SITE_PACKAGE_URI",
            },
            "requested_tasks": [
                {
                    "task_id": "REPLACE_WITH_TASK_ID",
                    "scenario_ids": ["REPLACE_WITH_SCENARIO_ID"],
                }
            ],
            "robot_profile": {
                "robot_profile_id": "REPLACE_WITH_ROBOT_PROFILE_ID",
                "embodiment": "REPLACE_WITH_EMBODIMENT",
                "sensors": ["REPLACE_WITH_SENSOR"],
            },
            "policy_package": {
                "policy_api_endpoint": {"endpoint_url": "REPLACE_WITH_POLICY_ENDPOINT"},
                "docker_container": {"image_ref": "REPLACE_WITH_IMAGE_REF"},
                "recorded_action_trace": {"trace_manifest_uri": "REPLACE_WITH_TRACE_URI"},
                "high_level_skill_trace": {"ordered_skill_sequence": ["REPLACE_WITH_SKILL"]},
                "teleop_demo": {"demo_artifact_uri": "REPLACE_WITH_DEMO_URI"},
                "sim_controller_plugin": {"plugin_uri": "REPLACE_WITH_PLUGIN_URI"},
            },
            "operation": "evaluate_only",
            "simulator_preference": "isaac_lab_arena",
            "rights_privacy_scope": {
                "status": "REPLACE_WITH_CLEARED_STATUS",
                "external_use_allowed": "REPLACE_WITH_BOOLEAN",
            },
            "source": {
                "system": "Blueprint-WebApp",
                "site_submission_id": "REPLACE_WITH_SITE_SUBMISSION_ID",
                "request_id": "REPLACE_WITH_REQUEST_ID",
                "buyer_request_id": "REPLACE_WITH_BUYER_REQUEST_ID",
                "capture_job_id": "REPLACE_WITH_CAPTURE_JOB_ID",
            },
        },
        "proof_boundary": {
            **CONTROL_PLANE_NOT_PROOF,
            "packet_is_request_contract_only": True,
            "public_claim_upgrade_allowed": False,
        },
    }
    return packet


def _external_input_packet_markdown(packet: Mapping[str, Any]) -> str:
    required_inputs = packet.get("required_inputs")
    enablement_inputs = packet.get("enablement_inputs")
    paths = _as_mapping(packet.get("configured_paths"))
    lines = [
        "# Live Pipeline External Input Packet",
        "",
        f"- Schema: `{packet.get('schema_version')}`",
        f"- Status: `{packet.get('status')}`",
        f"- Generated: `{packet.get('generated_at')}`",
        "- Boundary: request/contract artifact only; not simulator, robot policy, safety, or readiness proof.",
        "",
        "## Configured Paths",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Required Inputs"])
    if isinstance(required_inputs, list) and required_inputs:
        for item in required_inputs:
            if not isinstance(item, Mapping):
                continue
            blockers = ", ".join(str(blocker) for blocker in item.get("current_blockers", []))
            lines.append(f"- `{item.get('id')}`: `{item.get('status')}`")
            if item.get("missing_fields"):
                lines.append(f"  - Missing fields: `{', '.join(item['missing_fields'])}`")
            if blockers:
                lines.append(f"  - Current blockers: `{blockers}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Enablement Inputs"])
    if isinstance(enablement_inputs, list) and enablement_inputs:
        for item in enablement_inputs:
            if not isinstance(item, Mapping):
                continue
            blockers = ", ".join(str(blocker) for blocker in item.get("current_blockers", []))
            lines.append(f"- `{item.get('id')}`: `{item.get('status')}`")
            if blockers:
                lines.append(f"  - Current blockers: `{blockers}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## WebApp Request Shape",
            "",
            "See `example_robot_eval_job_request` in the JSON packet. Values are placeholders only.",
            "",
        ]
    )
    return "\n".join(lines)


def run_live_pipeline_control_plane(
    *,
    capture_root: str | Path | None = None,
    job_request_inbox: str | Path | None = None,
    package_dir: str | Path | None = None,
    arena_results_dir: str | Path | None = None,
    simulator_audit_command: str | None = None,
    vision_labeling_command: str | None = None,
    delivery_command: str | None = None,
    process_inbox: bool = True,
    load_local_env: bool = True,
    allow_digitalocean_read: bool | None = None,
    digitalocean_token_env: str = "DIGITALOCEAN_ACCESS_TOKEN",
    digitalocean_droplet_name: str | None = None,
    digitalocean_droplet_ip: str | None = None,
    agent_mode: str | None = None,
    allow_live_agent_operator: bool | None = None,
    provisioner: str | None = None,
    simulator: str | None = None,
    allow_gpu_provisioning: bool | None = None,
    allow_simulator_execution: bool | None = None,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Sequence[str] = (),
    allow_cpu_simulator_preflight: bool | None = None,
    cpu_preflight_backends: Sequence[str] = CPU_BACKENDS,
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool | None = None,
    allow_training: bool | None = None,
    training_command: str | None = None,
    timeout_seconds: int | None = None,
    budget_usd: float | None = None,
    arena_scenario_count: int = 500,
    arena_shard_size: int = 50,
    arena_num_envs: int = 16,
    arena_retry_budget: int = 2,
    allow_rollout_vision_labeling: bool | None = None,
    allow_delivery_upload: bool | None = None,
    arena_operator_mode: str | None = None,
    allow_live_agents_sdk: bool | None = None,
    allow_live_codex_sdk: bool | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    original_env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        initial_capture_text = _env_value(CAPTURE_ROOT_ENV, capture_root)
        initial_capture_path = Path(initial_capture_text).resolve() if initial_capture_text else None
        env_roots = _unique_paths(
            [repo_root, Path.cwd(), initial_capture_path]
            if initial_capture_path
            else [repo_root, Path.cwd()]
        )
        env_summary = (
            load_env_files(env_roots)
            if load_local_env
            else {
                "files": [],
                "loaded_keys": [],
                "skipped_existing_keys": [],
                "skipped_placeholder_keys": [],
            }
        )
        capture_text = _env_value(CAPTURE_ROOT_ENV, capture_root)
        capture_path = Path(capture_text).resolve() if capture_text else None
        if load_local_env and capture_path and capture_path.resolve() not in set(env_roots):
            capture_env_summary = load_env_files([capture_path])
            env_summary = {
                "files": env_summary["files"] + capture_env_summary["files"],
                "loaded_keys": sorted(
                    set(env_summary["loaded_keys"]) | set(capture_env_summary["loaded_keys"])
                ),
                "skipped_existing_keys": sorted(
                    set(env_summary["skipped_existing_keys"])
                    | set(capture_env_summary["skipped_existing_keys"])
                ),
                "skipped_placeholder_keys": sorted(
                    set(env_summary["skipped_placeholder_keys"])
                    | set(capture_env_summary["skipped_placeholder_keys"])
                ),
            }
        inbox_text = _env_value(JOB_REQUEST_INBOX_ENV, job_request_inbox)
        inbox_path = Path(inbox_text).resolve() if inbox_text else None
        package_text = _env_value(PACKAGE_DIR_ENV, package_dir)
        package_path = Path(package_text).resolve() if package_text else None
        arena_results_text = _env_value(ARENA_RESULTS_DIR_ENV, arena_results_dir)
        arena_results_path = Path(arena_results_text).resolve() if arena_results_text else None
        output_text = _env_value(CONTROL_PLANE_OUTPUT_PATH_ENV, output_path)
        output = _output_path(capture_path, output_text)
        secret_values = _secret_values()

        resolved_agent_mode = _string(agent_mode or os.getenv(CONTROL_PLANE_AGENT_MODE_ENV)) or "none"
        resolved_arena_operator_mode = (
            _string(arena_operator_mode or os.getenv(CONTROL_PLANE_ARENA_OPERATOR_MODE_ENV))
            or "none"
        )
        resolved_provisioner = (
            _string(provisioner or os.getenv(CONTROL_PLANE_PROVISIONER_ENV)) or "fixture_local"
        )
        resolved_simulator = _string(simulator or os.getenv(CONTROL_PLANE_SIMULATOR_ENV)) or "fixture"
        resolved_timeout = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else _env_int(CONTROL_PLANE_TIMEOUT_SECONDS_ENV, 120)
        )
        resolved_simulator_audit_command = (
            _string(simulator_audit_command or os.getenv(SIMULATOR_AUDIT_COMMAND_ENV)) or None
        )
        resolved_vision_command = (
            _string(vision_labeling_command or os.getenv(VISION_LABELING_COMMAND_ENV)) or None
        )
        resolved_delivery_command = _string(delivery_command or os.getenv(DELIVERY_COMMAND_ENV)) or None
        digitalocean_read_allowed = (
            bool(allow_digitalocean_read)
            if allow_digitalocean_read is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_DIGITALOCEAN_READ_ENV)
        )
        resolved_digitalocean_name = (
            _string(digitalocean_droplet_name or os.getenv(DIGITALOCEAN_DROPLET_NAME_ENV)) or None
        )
        resolved_digitalocean_ip = (
            _string(digitalocean_droplet_ip or os.getenv(DIGITALOCEAN_DROPLET_IP_ENV)) or None
        )
        live_agent_operator_allowed = (
            bool(allow_live_agent_operator)
            if allow_live_agent_operator is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_LIVE_AGENT_OPERATOR_ENV)
        )
        gpu_allowed = (
            bool(allow_gpu_provisioning)
            if allow_gpu_provisioning is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_GPU_PROVISIONING_ENV)
        )
        simulator_execution_allowed = (
            bool(allow_simulator_execution)
            if allow_simulator_execution is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_SIMULATOR_EXECUTION_ENV)
        )
        cpu_preflight_allowed = (
            bool(allow_cpu_simulator_preflight)
            if allow_cpu_simulator_preflight is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_ENV)
        )
        cpu_preflight_render_allowed = (
            bool(allow_cpu_preflight_render)
            if allow_cpu_preflight_render is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_RENDER_ENV)
        )
        training_allowed = (
            bool(allow_training)
            if allow_training is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_TRAINING_ENV)
        )
        vision_allowed = (
            bool(allow_rollout_vision_labeling)
            if allow_rollout_vision_labeling is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING_ENV)
        )
        delivery_allowed = (
            bool(allow_delivery_upload)
            if allow_delivery_upload is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_DELIVERY_UPLOAD_ENV)
        )
        live_agents_allowed = (
            bool(allow_live_agents_sdk)
            if allow_live_agents_sdk is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK_ENV)
            or _env_truthy(LIVE_AGENTS_SDK_ENV)
        )
        live_codex_allowed = (
            bool(allow_live_codex_sdk)
            if allow_live_codex_sdk is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK_ENV)
            or _env_truthy(LIVE_CODEX_SDK_ENV)
        )
        parsed_simulator_commands = _parse_simulator_commands(simulator_commands)

        setup_output = (
            capture_path / "pipeline" / "live_pipeline_setup" / "live_pipeline_setup_manifest.json"
            if capture_path
            else output.parent / "live_pipeline_setup_manifest.json"
        )
        try:
            setup_manifest = build_live_pipeline_setup_manifest(
                capture_root=capture_path,
                package_dir=package_path,
                arena_results_dir=arena_results_path,
                simulator_command=resolved_simulator_audit_command,
                vision_labeling_command=resolved_vision_command,
                delivery_command=resolved_delivery_command,
                load_local_env=False,
                allow_digitalocean_read=digitalocean_read_allowed,
                digitalocean_token_env=digitalocean_token_env,
                digitalocean_droplet_name=resolved_digitalocean_name,
                digitalocean_droplet_ip=resolved_digitalocean_ip,
                output_path=setup_output,
                timeout_seconds=min(resolved_timeout, 30),
            )
        except Exception as exc:  # pragma: no cover - exact exception varies by bad path
            setup_manifest = {
                "schema_version": "blueprint_live_pipeline_setup_blocked.v1",
                "generated_at": utc_now_iso(),
                "status": "blocked",
                "capture_root": str(capture_path) if capture_path else None,
                "blockers": [f"setup_audit_failed:{type(exc).__name__}"],
                "error_type": type(exc).__name__,
            }
            ensure_dir(setup_output.parent)
            write_json(setup_output, setup_manifest)

        inbox_run: Dict[str, Any]
        if not process_inbox:
            inbox_run = _inbox_status_not_configured("inbox_processing_disabled")
        elif capture_path is None:
            inbox_run = _inbox_status_not_configured("missing_capture_root")
        elif inbox_path is None:
            inbox_run = _inbox_status_not_configured("missing_job_request_inbox")
        else:
            ensure_dir(inbox_path)
            try:
                inbox_result = run_robot_eval_job_request_inbox(
                    capture_root=capture_path,
                    inbox_dir=inbox_path,
                    agent_adapter=_agent_adapter_from_mode(
                        resolved_agent_mode,
                        allow_live_operator=live_agent_operator_allowed,
                    ),
                    provisioner=resolved_provisioner,
                    simulator=resolved_simulator,
                    allow_gpu_provisioning=gpu_allowed,
                    allow_simulator_execution=simulator_execution_allowed,
                    allowed_simulators=allowed_simulators,
                    simulator_commands=parsed_simulator_commands,
                    allow_cpu_simulator_preflight=cpu_preflight_allowed,
                    cpu_preflight_backends=cpu_preflight_backends,
                    cpu_preflight_smoke_steps=cpu_preflight_smoke_steps,
                    allow_cpu_preflight_render=cpu_preflight_render_allowed,
                    allow_training=training_allowed,
                    training_command=training_command,
                    timeout_seconds=resolved_timeout,
                    budget_usd=budget_usd,
                    arena_results_dir=arena_results_path,
                    arena_scenario_count=arena_scenario_count,
                    arena_shard_size=arena_shard_size,
                    arena_num_envs=arena_num_envs,
                    arena_retry_budget=arena_retry_budget,
                    allow_rollout_vision_labeling=vision_allowed,
                    vision_labeling_command=resolved_vision_command,
                    allow_delivery_upload=delivery_allowed,
                    delivery_command=resolved_delivery_command,
                    arena_operator_mode=resolved_arena_operator_mode,
                    allow_live_agents_sdk=live_agents_allowed,
                    allow_live_codex_sdk=live_codex_allowed,
                )
                inbox_run = {
                    **inbox_result,
                    "processed": True,
                    "manifest_path": str(
                        capture_path
                        / "pipeline"
                        / "robot_eval_job_requests"
                        / "inbox_run_manifest.json"
                    ),
                    "blockers": [],
                }
            except Exception as exc:  # pragma: no cover - exact exception varies by bad path
                inbox_run = {
                    "status": "blocked",
                    "processed": False,
                    "processed_count": 0,
                    "blockers": [f"inbox_run_failed:{type(exc).__name__}"],
                    "error_type": type(exc).__name__,
                    "manifest_path": None,
                }

        blockers: List[str] = []
        if capture_path is None:
            blockers.append("missing_capture_root")
        for blocker in inbox_run.get("blockers") or []:
            blockers.append(f"inbox:{blocker}")

        webapp_inbox_truth = _webapp_job_request_inbox_truth(
            inbox_path=inbox_path,
            capture_root=capture_path,
        )
        webapp_upstream_truth_ready = _setup_section_ready(
            setup_manifest,
            "webapp_upstream_truth",
        ) or bool(webapp_inbox_truth.get("ready"))
        generated_at = utc_now_iso()
        external_input_packet_path = output.parent / "live_pipeline_external_input_packet.json"
        external_input_packet_markdown_path = (
            output.parent / "live_pipeline_external_input_packet.md"
        )
        external_input_packet = _build_external_input_packet(
            generated_at=generated_at,
            capture_root=capture_path,
            job_request_inbox=inbox_path,
            package_dir=package_path,
            arena_results_dir=arena_results_path,
            output_path=output,
            setup_manifest_path=setup_output,
            setup_manifest=setup_manifest,
            inbox_run=inbox_run,
            webapp_inbox_truth=webapp_inbox_truth,
        )
        external_input_packet["secrets_leaked"] = _manifest_leaks_secret(
            external_input_packet,
            secret_values,
        )
        write_json(external_input_packet_path, external_input_packet)
        write_text(
            external_input_packet_markdown_path,
            _external_input_packet_markdown(external_input_packet),
        )

        manifest: Dict[str, Any] = {
            "schema_version": LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": _overall_status(
                capture_root=capture_path,
                inbox=inbox_run,
                setup_manifest=setup_manifest,
            ),
            "capture_root": str(capture_path) if capture_path else None,
            "job_request_inbox": str(inbox_path) if inbox_path else None,
            "output_path": str(output),
            "env_files": env_summary,
            "setup_manifest_path": str(setup_output),
            "setup_status": setup_manifest.get("status"),
            "setup_blockers": setup_manifest.get("blockers", []),
            "inbox_run": inbox_run,
            "webapp_inbox_truth": webapp_inbox_truth,
            "effective_webapp_upstream_truth_ready": webapp_upstream_truth_ready,
            "external_input_packet": {
                "schema_version": LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION,
                "status": external_input_packet["status"],
                "path": str(external_input_packet_path),
                "markdown_path": str(external_input_packet_markdown_path),
                "required_input_count": len(external_input_packet["required_inputs"]),
                "enablement_input_count": len(external_input_packet["enablement_inputs"]),
                "secrets_leaked": external_input_packet["secrets_leaked"],
            },
            "operator_config": {
                "agent_mode": resolved_agent_mode,
                "arena_operator_mode": resolved_arena_operator_mode,
                "live_agent_operator_allowed_by_control_plane": live_agent_operator_allowed,
                "live_agents_sdk_allowed_by_control_plane": live_agents_allowed,
                "live_codex_sdk_allowed_by_control_plane": live_codex_allowed,
            },
            "digitalocean_config": {
                "read_allowed_by_control_plane": digitalocean_read_allowed,
                "droplet_name": resolved_digitalocean_name,
                "droplet_ip": resolved_digitalocean_ip,
                "token_env": digitalocean_token_env,
            },
            "execution_config": {
                "provisioner": resolved_provisioner,
                "simulator": resolved_simulator,
                "allowed_simulators": list(allowed_simulators),
                "simulator_commands_configured": sorted(parsed_simulator_commands),
                "allow_gpu_provisioning": gpu_allowed,
                "allow_simulator_execution": simulator_execution_allowed,
                "allow_cpu_simulator_preflight": cpu_preflight_allowed,
                "allow_cpu_preflight_render": cpu_preflight_render_allowed,
                "allow_training": training_allowed,
                "allow_rollout_vision_labeling": vision_allowed,
                "allow_delivery_upload": delivery_allowed,
                "arena_scenario_count": arena_scenario_count,
                "arena_shard_size": arena_shard_size,
                "arena_num_envs": arena_num_envs,
                "arena_retry_budget": arena_retry_budget,
                "timeout_seconds": resolved_timeout,
            },
            "control_plane_boundary": {
                **CONTROL_PLANE_NOT_PROOF,
                "public_claim_upgrade_allowed": False,
                "proof_boundary_authority": "deterministic_artifacts_and_owner_system_evidence",
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "blockers": blockers,
            "next_inputs_needed": _control_plane_next_inputs_needed(
                capture_root=capture_path,
                job_request_inbox=inbox_path,
                setup_manifest=setup_manifest,
                webapp_upstream_truth_ready=webapp_upstream_truth_ready,
            ),
        }
        manifest["secrets_leaked"] = _manifest_leaks_secret(manifest, secret_values)
        ensure_dir(output.parent)
        write_json(output, manifest)
        return manifest
    finally:
        _restore_env(original_env)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Blueprint live pipeline control plane once: audit readiness and optionally "
            "consume a robot-eval job request inbox."
        )
    )
    parser.add_argument("--capture-root")
    parser.add_argument("--job-request-inbox")
    parser.add_argument("--package-dir")
    parser.add_argument("--arena-results-dir")
    parser.add_argument("--simulator-audit-command")
    parser.add_argument("--vision-labeling-command")
    parser.add_argument("--delivery-command")
    parser.add_argument("--no-process-inbox", action="store_true")
    parser.add_argument("--no-load-env-files", action="store_true")
    parser.add_argument("--allow-digitalocean-read", action="store_true", default=None)
    parser.add_argument("--digitalocean-token-env", default="DIGITALOCEAN_ACCESS_TOKEN")
    parser.add_argument("--digitalocean-droplet-name")
    parser.add_argument("--digitalocean-droplet-ip")
    parser.add_argument("--agent-mode", choices=("none", "fake", "agents-sdk"), default=None)
    parser.add_argument("--allow-live-agent-operator", action="store_true", default=None)
    parser.add_argument("--provisioner", default=None)
    parser.add_argument("--simulator", default=None)
    parser.add_argument("--allow-gpu-provisioning", action="store_true", default=None)
    parser.add_argument("--allow-simulator-execution", action="store_true", default=None)
    parser.add_argument("--allow-simulator", action="append", default=[])
    parser.add_argument("--simulator-command", action="append", default=[])
    parser.add_argument("--allow-cpu-simulator-preflight", action="store_true", default=None)
    parser.add_argument("--cpu-preflight-backend", action="append", default=[])
    parser.add_argument("--cpu-preflight-smoke-steps", type=int, default=10)
    parser.add_argument("--allow-cpu-preflight-render", action="store_true", default=None)
    parser.add_argument("--allow-training", action="store_true", default=None)
    parser.add_argument("--training-command")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--arena-scenario-count", type=int, default=500)
    parser.add_argument("--arena-shard-size", type=int, default=50)
    parser.add_argument("--arena-num-envs", type=int, default=16)
    parser.add_argument("--arena-retry-budget", type=int, default=2)
    parser.add_argument("--allow-rollout-vision-labeling", action="store_true", default=None)
    parser.add_argument("--allow-delivery-upload", action="store_true", default=None)
    parser.add_argument("--arena-operator-mode", choices=("none", "fake", "agents-sdk"), default=None)
    parser.add_argument("--allow-live-agents-sdk", action="store_true", default=None)
    parser.add_argument("--allow-live-codex-sdk", action="store_true", default=None)
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    result = run_live_pipeline_control_plane(
        capture_root=args.capture_root,
        job_request_inbox=args.job_request_inbox,
        package_dir=args.package_dir,
        arena_results_dir=args.arena_results_dir,
        simulator_audit_command=args.simulator_audit_command,
        vision_labeling_command=args.vision_labeling_command,
        delivery_command=args.delivery_command,
        process_inbox=not args.no_process_inbox,
        load_local_env=not args.no_load_env_files,
        allow_digitalocean_read=args.allow_digitalocean_read,
        digitalocean_token_env=args.digitalocean_token_env,
        digitalocean_droplet_name=args.digitalocean_droplet_name,
        digitalocean_droplet_ip=args.digitalocean_droplet_ip,
        agent_mode=args.agent_mode,
        allow_live_agent_operator=args.allow_live_agent_operator,
        provisioner=args.provisioner,
        simulator=args.simulator,
        allow_gpu_provisioning=args.allow_gpu_provisioning,
        allow_simulator_execution=args.allow_simulator_execution,
        allowed_simulators=args.allow_simulator,
        simulator_commands=args.simulator_command,
        allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
        cpu_preflight_backends=args.cpu_preflight_backend or CPU_BACKENDS,
        cpu_preflight_smoke_steps=args.cpu_preflight_smoke_steps,
        allow_cpu_preflight_render=args.allow_cpu_preflight_render,
        allow_training=args.allow_training,
        training_command=args.training_command,
        timeout_seconds=args.timeout_seconds,
        budget_usd=args.budget_usd,
        arena_scenario_count=args.arena_scenario_count,
        arena_shard_size=args.arena_shard_size,
        arena_num_envs=args.arena_num_envs,
        arena_retry_budget=args.arena_retry_budget,
        allow_rollout_vision_labeling=args.allow_rollout_vision_labeling,
        allow_delivery_upload=args.allow_delivery_upload,
        arena_operator_mode=args.arena_operator_mode,
        allow_live_agents_sdk=args.allow_live_agents_sdk,
        allow_live_codex_sdk=args.allow_live_codex_sdk,
        output_path=args.output_path,
    )
    print(f"[live-pipeline-control-plane] manifest={result['output_path']}")
    print(f"[live-pipeline-control-plane] status={result['status']}")
    if result["blockers"]:
        print(f"[live-pipeline-control-plane] blockers={len(result['blockers'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
