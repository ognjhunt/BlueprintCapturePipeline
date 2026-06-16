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
import shlex
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
    REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION,
    RobotEvalJobAgentAdapter,
    run_robot_eval_job_request_inbox,
)
from .safe_env import load_env_files


LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION = "blueprint_live_pipeline_control_plane_run.v1"
LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION = (
    "blueprint_live_pipeline_external_input_packet.v1"
)
LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION = "blueprint_live_pipeline_staged_inputs.v1"

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

LIVE_CLOSURE_EVIDENCE_ARTIFACT_NAMES = (
    "live_eval_closure_evidence.json",
    "package_closure_evidence.json",
)

POLICY_PACKAGE_ARTIFACT_NAMES = (
    "policy_package.json",
    "robot_team_policy_package.v1",
)

POLICY_MODALITY_ORDER = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)

WEBAPP_JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
WEBAPP_JOB_REQUEST_QUEUE_CONTRACT = "robot_eval_job_request_inbox.v1"

CAPTURE_ROOT_ENV = "BLUEPRINT_PIPELINE_CAPTURE_ROOT"
JOB_REQUEST_INBOX_ENV = "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX"
PACKAGE_DIR_ENV = "BLUEPRINT_PIPELINE_PACKAGE_DIR"
ARENA_RESULTS_DIR_ENV = "BLUEPRINT_ARENA_RESULTS_DIR"
STAGED_INPUTS_ENV = "BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"
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


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    try:
        return max(int(str(value)), 0)
    except (TypeError, ValueError):
        return 0


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
    real_robot_pov_ready: bool = False,
    live_closure_evidence_ready: bool = False,
    deployment_outcomes_ready: bool = False,
    deployment_prediction_match_keys_ready: bool = False,
    deployment_owner_evidence_ready: bool = False,
    policy_package_ready: bool = False,
    followup_request_queues: Mapping[str, Any] | None = None,
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
    if not real_robot_pov_ready:
        next_inputs.append(
            "Provide real robot POV evidence with exact run/variation keys, camera video, "
            "action logs, timestamp alignment, and owner evidence."
        )
    if not live_closure_evidence_ready:
        next_inputs.append(
            "Provide live closure evidence for the job, including package delivery/access, "
            "rights/privacy, and WebApp lineage."
        )
    if not deployment_outcomes_ready:
        next_inputs.append(
            "Provide deployment outcome records with task/scenario IDs and actual result "
            "signals for real-world validation."
        )
    elif not deployment_prediction_match_keys_ready:
        next_inputs.append(
            "Provide deployment outcome exact prediction join keys: scenario_eval_run_id and "
            "scenario_variation_instance_id."
        )
    elif not deployment_owner_evidence_ready:
        next_inputs.append(
            "Provide owner evidence for each deployment outcome record before claiming "
            "real-world validation closure."
        )
    if not policy_package_ready:
        next_inputs.append(
            "Provide a robot-team policy package through one supported modality before "
            "running policy execution."
        )
    followup_queues = _as_mapping(followup_request_queues)
    if followup_queues.get("ready"):
        for queue in followup_queues.get("queues") or []:
            if not isinstance(queue, Mapping):
                continue
            command = _string(queue.get("safe_processing_command"))
            if command:
                next_inputs.append(command)
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


def _field(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _policy_modality_missing_inputs(modality: str, reference: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    if modality == "policy_api_endpoint":
        endpoint = _string(_field(reference, "endpoint_url", "endpointUrl", "url"))
        if not (endpoint.startswith("https://") or endpoint.startswith("http://")):
            missing.append("policy_package.policy_api_endpoint.endpoint_url")
    elif modality == "docker_container":
        if not _string(_field(reference, "image_ref", "imageRef")):
            missing.append("policy_package.docker_container.image_ref")
        digest = _string(_field(reference, "digest", "digestChecksum"))
        if not digest.startswith("sha256:"):
            missing.append("policy_package.docker_container.digest")
    elif modality == "recorded_action_trace":
        if not _string(_field(reference, "trace_manifest_uri", "traceManifestUri")):
            missing.append("policy_package.recorded_action_trace.trace_manifest_uri")
        if not _string(_field(reference, "timestamp_alignment", "timestampAlignment")):
            missing.append("policy_package.recorded_action_trace.timestamp_alignment")
    elif modality == "high_level_skill_trace":
        sequence = reference.get("ordered_skill_sequence") or reference.get("orderedSkillSequence")
        if not (isinstance(sequence, list) and sequence):
            missing.append("policy_package.high_level_skill_trace.ordered_skill_sequence")
    elif modality == "teleop_demo":
        if not _string(_field(reference, "demo_artifact_uri", "demoArtifactUri")):
            missing.append("policy_package.teleop_demo.demo_artifact_uri")
        if not _string(_field(reference, "rights_privacy_attestation", "rightsPrivacyAttestation")):
            missing.append("policy_package.teleop_demo.rights_privacy_attestation")
    elif modality == "sim_controller_plugin":
        if not _string(_field(reference, "simulator_framework", "simulatorFramework")):
            missing.append("policy_package.sim_controller_plugin.simulator_framework")
        if not _string(_field(reference, "plugin_uri", "pluginUri")):
            missing.append("policy_package.sim_controller_plugin.plugin_uri")
    return missing


def _request_policy_package_audit(request: Mapping[str, Any]) -> Dict[str, Any]:
    policy_package = _as_mapping(request.get("policy_package") or request.get("policyPackage"))
    camel = {
        "policy_api_endpoint": "policyApiEndpoint",
        "docker_container": "dockerContainer",
        "recorded_action_trace": "recordedActionTrace",
        "high_level_skill_trace": "highLevelSkillTrace",
        "teleop_demo": "teleopDemo",
        "sim_controller_plugin": "simControllerPlugin",
    }
    selected: List[str] = []
    ready: List[str] = []
    missing_by_modality: Dict[str, List[str]] = {}
    for modality in POLICY_MODALITY_ORDER:
        payload = _as_mapping(policy_package.get(modality) or policy_package.get(camel[modality]))
        if not payload:
            continue
        selected.append(modality)
        missing = _policy_modality_missing_inputs(modality, payload)
        if missing:
            missing_by_modality[modality] = missing
        else:
            ready.append(modality)
    return {
        "selected_modalities": selected,
        "ready_modalities": ready,
        "missing_inputs": missing_by_modality,
        "ready": bool(ready),
    }


def _safe_followup_processing_command(capture_root: Path, inbox_dir: Path) -> str:
    return (
        "blueprint-run-robot-eval-job "
        f"--capture-root {shlex.quote(str(capture_root.resolve()))} "
        f"--job-request-inbox {shlex.quote(str(inbox_dir.resolve()))}"
    )


def _real_world_validation_followup_request_queues(
    capture_root: Path | None,
) -> Dict[str, Any]:
    if capture_root is None:
        return {
            "status": "not_configured",
            "ready": False,
            "capture_root": None,
            "queue_count": 0,
            "ready_queue_count": 0,
            "queued_request_count": 0,
            "queues": [],
            "blockers": ["missing_capture_root"],
        }
    jobs_dir = capture_root / "pipeline" / "robot_eval_jobs"
    queue_paths = (
        sorted(jobs_dir.glob("*/real_world_validation_followup_request_queue.json"))
        if jobs_dir.is_dir()
        else []
    )
    queues: List[Dict[str, Any]] = []
    total_queued = 0
    ready_count = 0
    aggregate_blockers: List[str] = []
    for queue_path in queue_paths:
        blockers: List[str] = []
        try:
            payload = read_json_any(queue_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            payload = {}
            blockers.append(f"followup_request_queue_read_failed:{type(exc).__name__}")
        if not isinstance(payload, Mapping):
            payload = {}
            blockers.append("followup_request_queue_not_json_object")
        schema_version = _string(payload.get("schema_version")) or None
        if schema_version != REAL_WORLD_VALIDATION_FOLLOWUP_REQUEST_QUEUE_SCHEMA_VERSION:
            blockers.append("followup_request_queue_schema_mismatch")
        status = _string(payload.get("status")) or "unknown"
        inbox_dir_text = _string(payload.get("inbox_dir")) or None
        inbox_dir = Path(inbox_dir_text).resolve() if inbox_dir_text else None
        queued_request_paths = [
            str(Path(str(path)).resolve())
            for path in payload.get("queued_request_paths") or []
            if _string(path)
        ]
        queued_count = _count(payload.get("queued_request_count")) or len(queued_request_paths)
        total_queued += queued_count
        if status == "ready_for_inbox_processing":
            if inbox_dir is None:
                blockers.append("followup_request_queue_inbox_missing")
            elif not inbox_dir.is_dir():
                blockers.append("followup_request_queue_inbox_dir_missing")
            if queued_count <= 0:
                blockers.append("followup_request_queue_empty")
            for request_path in queued_request_paths[:20]:
                if not Path(request_path).is_file():
                    blockers.append("followup_request_queue_request_file_missing")
                    break
        ready = status == "ready_for_inbox_processing" and not blockers
        if ready:
            ready_count += 1
        aggregate_blockers.extend(blockers)
        queues.append(
            {
                "job_id": _string(payload.get("parent_job_id")) or queue_path.parent.name,
                "path": str(queue_path.resolve()),
                "schema_version": schema_version,
                "status": status,
                "ready_for_inbox_processing": ready,
                "inbox_dir": str(inbox_dir) if inbox_dir else None,
                "queued_request_count": queued_count,
                "queued_request_paths": queued_request_paths[:20],
                "truncated_queued_request_paths": len(queued_request_paths) > 20,
                "safe_processing_command": (
                    _safe_followup_processing_command(capture_root, inbox_dir)
                    if ready and inbox_dir
                    else None
                ),
                "blockers": blockers,
            }
        )
    if ready_count:
        status = "ready_for_inbox_processing"
    elif aggregate_blockers:
        status = "blocked"
    elif queues:
        status = "no_followup_requests_queued"
    else:
        status = "no_followup_request_queues"
    return {
        "status": status,
        "ready": ready_count > 0,
        "capture_root": str(capture_root),
        "queue_count": len(queues),
        "ready_queue_count": ready_count,
        "queued_request_count": total_queued,
        "queues": queues[:20],
        "truncated_queues": len(queues) > 20,
        "blockers": sorted(set(aggregate_blockers)),
        "proof_boundary": (
            "Follow-up request queues are draft job-request inputs generated from "
            "predicted-vs-actual validation; processing them creates new local job "
            "artifacts but does not prove real-world rerun success."
        ),
    }


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
        source_selection = _as_mapping(source.get("selection_state"))
        owner_system = _as_mapping(request.get("owner_system"))
        site_package = _as_mapping(request.get("site_package"))
        top_level_sources = (request, source, source_selection, owner_system, site_package)
        fields_present: Dict[str, bool] = {}
        for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS:
            fields_present[field] = bool(_field_value_from_sources(request, field, top_level_sources))
        request_capture_root = _string(site_package.get("capture_root")) or None
        capture_root_matches = _path_matches_configured_capture_root(request_capture_root, capture_root)
        policy_package_audit = _request_policy_package_audit(request)
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
                "policy_package_ready": bool(policy_package_audit["ready"]),
                "policy_package_selected_modalities": policy_package_audit["selected_modalities"],
                "policy_package_ready_modalities": policy_package_audit["ready_modalities"],
                "policy_package_missing_inputs": policy_package_audit["missing_inputs"],
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
        "accepted_policy_package_request_count": sum(
            1 for candidate in accepted_candidates if candidate.get("policy_package_ready")
        ),
        "candidates": candidates[:20],
        "truncated_candidates": len(candidates) > 20,
        "proof_boundary": (
            "Queued WebApp job requests prove upstream linkage only when they contain real "
            "WebApp IDs and point at the configured capture root."
        ),
    }


def _staged_inputs_path(output_path: Path) -> Path:
    configured = _string(os.getenv(STAGED_INPUTS_ENV))
    if configured:
        return Path(configured).resolve()
    return output_path.parent / "live_pipeline_staged_inputs.json"


def _load_staged_inputs(path: Path, *, capture_root: Path | None) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "status": "not_configured",
            "ready": False,
            "path": str(path),
            "blockers": ["staged_inputs_missing"],
            "arena_results_dir": None,
            "webapp_request_path": None,
            "live_closure_evidence_path": None,
            "deployment_outcomes_path": None,
            "policy_package_path": None,
            "real_robot_pov_path": None,
        }
    try:
        payload = read_json_any(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": [f"staged_inputs_read_failed:{type(exc).__name__}"],
            "arena_results_dir": None,
            "webapp_request_path": None,
            "live_closure_evidence_path": None,
            "deployment_outcomes_path": None,
            "policy_package_path": None,
            "real_robot_pov_path": None,
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "blocked",
            "ready": False,
            "path": str(path),
            "blockers": ["staged_inputs_not_json_object"],
            "arena_results_dir": None,
            "webapp_request_path": None,
            "live_closure_evidence_path": None,
            "deployment_outcomes_path": None,
            "policy_package_path": None,
            "real_robot_pov_path": None,
        }
    blockers: List[str] = []
    if payload.get("schema_version") != LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION:
        blockers.append("staged_inputs_schema_mismatch")
    configured_capture_root = _string(payload.get("configured_capture_root")) or None
    capture_root_matches = _path_matches_configured_capture_root(
        configured_capture_root,
        capture_root,
    )
    if configured_capture_root and not capture_root_matches:
        blockers.append("staged_inputs_capture_root_mismatch")
    arena = _as_mapping(payload.get("arena_results"))
    webapp = _as_mapping(payload.get("webapp_request"))
    closure = _as_mapping(payload.get("live_closure_evidence"))
    outcomes = _as_mapping(payload.get("deployment_outcomes"))
    policy = _as_mapping(payload.get("policy_package"))
    real_pov = _as_mapping(payload.get("real_robot_pov"))
    arena_results_dir = _string(arena.get("arena_results_dir")) or None
    webapp_request_path = _string(webapp.get("target_path") or webapp.get("path")) or None
    live_closure_evidence_path = (
        _string(closure.get("target_path") or closure.get("path")) or None
    )
    deployment_outcomes_path = (
        _string(outcomes.get("target_path") or outcomes.get("path")) or None
    )
    policy_package_path = _string(policy.get("target_path") or policy.get("path")) or None
    real_robot_pov_path = _string(real_pov.get("target_path") or real_pov.get("path")) or None
    arena_ready = bool(arena.get("ready")) and bool(arena_results_dir)
    webapp_ready = bool(webapp.get("ready") or webapp.get("staged")) and bool(webapp_request_path)
    closure_ready = (
        bool(closure.get("ready") or closure.get("staged")) and bool(live_closure_evidence_path)
    )
    outcomes_ready = (
        bool(outcomes.get("ready") or outcomes.get("staged")) and bool(deployment_outcomes_path)
    )
    outcomes_records_ready = (
        bool(outcomes.get("records_ready_for_calibration"))
        and bool(deployment_outcomes_path)
    )
    outcomes_owner_evidence_ready = (
        bool(outcomes.get("owner_evidence_ready")) and bool(deployment_outcomes_path)
    )
    policy_ready = bool(policy.get("ready") or policy.get("staged")) and bool(policy_package_path)
    real_pov_ready = bool(real_pov.get("ready") or real_pov.get("staged")) and bool(real_robot_pov_path)
    if arena_results_dir and not Path(arena_results_dir).is_dir():
        blockers.append("staged_arena_results_dir_missing")
        arena_ready = False
    if webapp_request_path and not Path(webapp_request_path).is_file():
        blockers.append("staged_webapp_request_missing")
        webapp_ready = False
    if live_closure_evidence_path and not Path(live_closure_evidence_path).is_file():
        blockers.append("staged_live_closure_evidence_missing")
        closure_ready = False
    if deployment_outcomes_path and not Path(deployment_outcomes_path).is_file():
        blockers.append("staged_deployment_outcomes_missing")
        outcomes_ready = False
        outcomes_records_ready = False
        outcomes_owner_evidence_ready = False
    if policy_package_path and not Path(policy_package_path).is_file():
        blockers.append("staged_policy_package_missing")
        policy_ready = False
    if real_robot_pov_path and not Path(real_robot_pov_path).is_file():
        blockers.append("staged_real_robot_pov_missing")
        real_pov_ready = False
    status = (
        "ready"
        if (
            arena_ready
            or webapp_ready
            or closure_ready
            or outcomes_ready
            or policy_ready
            or real_pov_ready
        )
        and not blockers
        else "blocked"
    )
    if (
        not arena_ready
        and not webapp_ready
        and not closure_ready
        and not outcomes_ready
        and not policy_ready
        and not real_pov_ready
        and not blockers
    ):
        status = "empty"
    return {
        "status": status,
        "ready": status == "ready",
        "path": str(path),
        "schema_version": payload.get("schema_version"),
        "configured_capture_root": configured_capture_root,
        "capture_root_matches_control_plane": capture_root_matches,
        "arena_results_dir": arena_results_dir,
        "arena_results_ready": arena_ready,
        "webapp_request_path": webapp_request_path,
        "webapp_request_ready": webapp_ready,
        "live_closure_evidence_path": live_closure_evidence_path,
        "live_closure_evidence_ready": closure_ready,
        "live_closure_evidence_job_id": closure.get("job_id"),
        "deployment_outcomes_path": deployment_outcomes_path,
        "deployment_outcomes_ready": outcomes_ready,
        "deployment_outcomes_records_ready_for_calibration": outcomes_records_ready,
        "deployment_outcomes_prediction_match_keys_ready": bool(
            outcomes.get("prediction_match_keys_ready")
        )
        and bool(deployment_outcomes_path),
        "deployment_outcomes_owner_evidence_ready": outcomes_owner_evidence_ready,
        "deployment_outcomes_job_id": outcomes.get("job_id"),
        "deployment_outcome_record_count": int(outcomes.get("record_count") or 0),
        "deployment_outcome_prediction_match_key_record_count": int(
            outcomes.get("prediction_match_key_record_count") or 0
        ),
        "deployment_outcome_missing_prediction_match_key_record_ids": list(
            outcomes.get("missing_prediction_match_key_record_ids") or []
        ),
        "deployment_outcome_owner_evidence_record_count": int(
            outcomes.get("owner_evidence_record_count") or 0
        ),
        "deployment_outcome_missing_owner_evidence_record_ids": list(
            outcomes.get("missing_owner_evidence_record_ids") or []
        ),
        "policy_package_path": policy_package_path,
        "policy_package_ready": policy_ready,
        "policy_package_job_id": policy.get("job_id"),
        "policy_package_selected_modalities": list(policy.get("selected_modalities") or []),
        "real_robot_pov_path": real_robot_pov_path,
        "real_robot_pov_ready": real_pov_ready,
        "real_robot_pov_job_id": real_pov.get("job_id"),
        "real_robot_pov_record_count": int(real_pov.get("record_count") or 0),
        "real_robot_pov_exact_key_record_count": int(
            real_pov.get("exact_key_record_count") or 0
        ),
        "real_robot_pov_camera_video_record_count": int(
            real_pov.get("camera_video_record_count") or 0
        ),
        "real_robot_pov_action_log_record_count": int(
            real_pov.get("action_log_record_count") or 0
        ),
        "real_robot_pov_timestamp_alignment_record_count": int(
            real_pov.get("timestamp_alignment_record_count") or 0
        ),
        "real_robot_pov_evidence_record_count": int(
            real_pov.get("evidence_record_count") or 0
        ),
        "real_robot_pov_missing_exact_key_record_ids": list(
            real_pov.get("missing_exact_key_record_ids") or []
        ),
        "real_robot_pov_missing_evidence_record_ids": list(
            real_pov.get("missing_evidence_record_ids") or []
        ),
        "blockers": blockers,
        "proof_boundary": "staged inputs are pointers to validated inputs, not proof claims",
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


BLOCKER_PACKET_TEMPLATES: Dict[str, Dict[str, str]] = {
    "webapp_upstream_truth": {
        "owner": "Blueprint-WebApp operator",
        "required_input": (
            "A real robot_eval_job_request.v1 or queue envelope with site_submission_id, "
            "request_id, buyer_request_id, capture_job_id, and the configured capture root."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--webapp-job-request <robot_eval_job_request.json> --stage-webapp-request --overwrite"
        ),
        "retry_condition": (
            "Re-run after the WebApp request has real IDs and the request capture_root matches "
            "BLUEPRINT_PIPELINE_CAPTURE_ROOT."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not invent placeholder upstream IDs, copy capture IDs into WebApp IDs, or mark "
            "webapp_upstream_truth ready without a real request source."
        ),
    },
    "isaac_lab_arena_owner_evidence": {
        "owner": "Simulator owner/operator",
        "required_input": (
            "Owner-system Isaac Lab-Arena result artifacts or an explicitly gated simulator "
            "command that produces normalized run evidence."
        ),
        "safe_proof_command": (
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true blueprint-run-live-pipeline-control-plane "
            "--capture-root <capture-root> --job-request-inbox <job-request-inbox> "
            "--simulator isaac_lab_arena --allow-simulator isaac_lab_arena "
            "--allow-simulator-execution --simulator-command "
            "'isaac_lab_arena=<owner-arena-command>' --output-path <control-plane-manifest>"
        ),
        "retry_condition": (
            "Re-run after the owner result directory exists or the gated simulator command exits "
            "successfully and writes expected artifacts."
        ),
        "resume_target": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--arena-results-dir <owner-arena-results-dir> --stage-arena-results --overwrite"
        ),
        "disallowed_workaround": (
            "Do not treat fixture runs, CPU smoke checks, advisory agent plans, or generated "
            "variation manifests as live simulator execution proof."
        ),
    },
    "live_robot_eval_closure_evidence": {
        "owner": "Blueprint delivery/review operator",
        "required_input": (
            "Job-specific package closure evidence covering delivery/access, rights/privacy, "
            "and WebApp lineage where applicable."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--live-closure-evidence <live_eval_closure_evidence.json> "
            "--stage-live-closure-evidence --overwrite"
        ),
        "retry_condition": (
            "Re-run after the staged closure file job_id matches a robot-eval job and includes "
            "local evidence refs or owner proof URIs for package closure claims."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not upgrade package access, rights, or WebApp lineage claims from local "
            "package generation alone."
        ),
    },
    "robot_team_policy_package": {
        "owner": "Robot team",
        "required_input": (
            "A policy package using at least one supported modality: policy API endpoint, Docker "
            "container, recorded action traces, high-level skill traces, teleop demos, or sim "
            "controller plugin."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--policy-package <policy_package.json> --stage-policy-package --overwrite"
        ),
        "retry_condition": (
            "Re-run after one selected modality has all required fields and points to real "
            "policy or trace artifacts."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not use placeholder policy endpoints, missing Docker digests, or unsupported "
            "modalities to satisfy policy execution input readiness."
        ),
    },
    "real_robot_pov_evidence": {
        "owner": "Robot team",
        "required_input": (
            "Real robot POV evidence for exact scenario run/variation keys, including camera "
            "video, action logs, timestamp alignment, and owner evidence."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--real-robot-pov <real_robot_pov_manifest.json> --stage-real-robot-pov --overwrite"
        ),
        "retry_condition": (
            "Re-run after real_robot_pov_manifest.v1 contains exact run/variation keys and "
            "camera/action owner evidence."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not use generated simulator POV, storyboards, or local support media as real "
            "robot POV evidence."
        ),
    },
    "real_world_deployment_outcomes": {
        "owner": "Robot team deployment operator",
        "required_input": (
            "Deployment outcome records with task/scenario IDs and actual result signals from "
            "a real robot-team deployment or pilot."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--deployment-outcomes <deployment_outcome_manifest.json> "
            "--stage-deployment-outcomes --overwrite"
        ),
        "retry_condition": (
            "Re-run after deployment_outcome_manifest.v1 records exist for the job."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not infer actual deployment outcomes from simulator predictions or local "
            "evaluation rows."
        ),
    },
    "predicted_vs_actual_exact_match_keys": {
        "owner": "Robot team deployment operator",
        "required_input": (
            "Exact deployment outcome join keys for predicted-vs-actual calibration."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--deployment-outcomes <deployment_outcome_manifest.json> "
            "--stage-deployment-outcomes --overwrite"
        ),
        "retry_condition": (
            "Re-run after every deployment outcome record includes scenario_eval_run_id and "
            "scenario_variation_instance_id."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not calibrate predicted-vs-actual outcomes from task/scenario names alone."
        ),
    },
    "real_world_deployment_outcome_owner_evidence": {
        "owner": "Robot team deployment operator",
        "required_input": (
            "Owner evidence references or attestations for each deployment outcome record."
        ),
        "safe_proof_command": (
            "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
            "--deployment-outcomes <deployment_outcome_manifest.json> "
            "--stage-deployment-outcomes --overwrite"
        ),
        "retry_condition": (
            "Re-run after every deployment outcome has evidence_refs, owner_evidence_refs, "
            "owner_evidence_uri, or an operator attestation."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not treat un-attested result rows as owner deployment evidence."
        ),
    },
    "rollout_vision_labeling": {
        "owner": "Blueprint model-ops operator",
        "required_input": "A gated rollout vision-labeling command and enablement env.",
        "safe_proof_command": (
            "BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true "
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --allow-rollout-vision-labeling "
            "--vision-labeling-command '<vision-label-command>' --output-path <control-plane-manifest>"
        ),
        "retry_condition": (
            "Re-run after the command is configured and produces review-required label artifacts."
        ),
        "resume_target": (
            "blueprint-audit-live-pipeline-proof-boundary --manifest-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not treat model labels as accepted failure labels without review resolution."
        ),
    },
    "delivery_upload": {
        "owner": "Blueprint delivery operator",
        "required_input": "A gated package delivery command that uploads artifacts and returns signed access.",
        "safe_proof_command": (
            "BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true "
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --allow-delivery-upload "
            "--delivery-command '<delivery-upload-command>' --output-path <control-plane-manifest>"
        ),
        "retry_condition": (
            "Re-run after delivery_upload.command.json or signed_access.command.json contains "
            "non-placeholder signed URLs and storage proof."
        ),
        "resume_target": (
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not claim delivery from local export manifests without storage upload and signed "
            "access evidence."
        ),
    },
    "live_agents_operator": {
        "owner": "Blueprint operator",
        "required_input": "Gated Agents SDK credentials and explicit live-operator enablement.",
        "safe_proof_command": (
            "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK=true "
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --allow-live-agents-sdk "
            "--agent-mode agents-sdk --output-path <control-plane-manifest>"
        ),
        "retry_condition": (
            "Re-run after the Agents SDK is installed, OPENAI_API_KEY is configured, and the "
            "operator gate is explicitly enabled."
        ),
        "resume_target": (
            "blueprint-audit-live-pipeline-proof-boundary --manifest-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not let agent recommendations mutate proof booleans or replace deterministic artifacts."
        ),
    },
    "live_codex_operator": {
        "owner": "Blueprint operator",
        "required_input": "Gated Codex SDK or host-OAuth Codex CLI operator credentials.",
        "safe_proof_command": (
            "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK=true "
            "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
            "--job-request-inbox <job-request-inbox> --allow-live-codex-sdk "
            "--output-path <control-plane-manifest>"
        ),
        "retry_condition": (
            "Re-run after Codex SDK or Codex CLI host OAuth is available and explicitly gated."
        ),
        "resume_target": (
            "blueprint-audit-live-pipeline-proof-boundary --manifest-path <control-plane-manifest>"
        ),
        "disallowed_workaround": (
            "Do not treat Codex edits or advisory plans as simulator, policy, safety, or delivery proof."
        ),
    },
}


def _blocker_packet_for_input(item: Mapping[str, Any]) -> Dict[str, str]:
    input_id = _string(item.get("id"))
    template = BLOCKER_PACKET_TEMPLATES.get(
        input_id,
        {
            "owner": "Blueprint operator",
            "required_input": f"Resolve external input `{input_id}` with real evidence.",
            "safe_proof_command": (
                "blueprint-run-live-pipeline-control-plane --capture-root <capture-root> "
                "--job-request-inbox <job-request-inbox> --output-path <control-plane-manifest>"
            ),
            "retry_condition": "Re-run after the required external input is staged.",
            "resume_target": (
                "blueprint-audit-live-pipeline-proof-boundary --manifest-path <control-plane-manifest>"
            ),
            "disallowed_workaround": (
                "Do not satisfy this blocker with placeholders, inferred evidence, or proof "
                "boolean edits."
            ),
        },
    )
    blockers = item.get("current_blockers")
    packet = {
        "id": input_id,
        **template,
        "current_blockers": ", ".join(str(blocker) for blocker in blockers)
        if isinstance(blockers, list) and blockers
        else "none",
    }
    return packet


def _with_blocker_packets(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in items:
        enriched.append({**item, "blocker_packet": _blocker_packet_for_input(item)})
    return enriched


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
    staged_inputs: Mapping[str, Any],
    followup_request_queues: Mapping[str, Any],
) -> Dict[str, Any]:
    webapp_section = _setup_section(setup_manifest, "webapp_upstream_truth")
    arena_section = _setup_section(setup_manifest, "real_arena_execution")
    webapp_truth_ready = _setup_section_ready(
        setup_manifest,
        "webapp_upstream_truth",
    ) or bool(webapp_inbox_truth.get("ready"))
    live_closure_evidence_ready = bool(staged_inputs.get("live_closure_evidence_ready"))
    real_robot_pov_ready = bool(staged_inputs.get("real_robot_pov_ready"))
    deployment_outcomes_ready = bool(staged_inputs.get("deployment_outcomes_ready"))
    deployment_prediction_match_keys_ready = bool(
        staged_inputs.get("deployment_outcomes_prediction_match_keys_ready")
    )
    deployment_owner_evidence_ready = bool(
        staged_inputs.get("deployment_outcomes_owner_evidence_ready")
    )
    policy_package_ready = bool(staged_inputs.get("policy_package_ready")) or bool(
        webapp_inbox_truth.get("accepted_policy_package_request_count")
    )
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
    if not real_robot_pov_ready:
        required_inputs.append(
            {
                "id": "real_robot_pov_evidence",
                "title": "Real robot POV and action evidence",
                "status": _string(staged_inputs.get("status")) or "not_staged",
                "accepted_paths": [
                    {
                        "kind": "global_real_robot_pov_manifest",
                        "target": (
                            "<capture_root>/pipeline/robot_eval_inputs/"
                            "real_robot_pov_manifest.json"
                        ),
                        "required_artifacts": ["real_robot_pov_manifest.v1"],
                    }
                ],
                "required_record_fields": [
                    "task_id",
                    "scenario_id",
                    "scenario_eval_run_id",
                    "scenario_variation_instance_id",
                    "robot_camera_video_uri",
                    "action_log_uri",
                    "timestamp_alignment",
                    "owner_evidence_refs",
                ],
                "staged_inputs": {
                    "status": staged_inputs.get("status"),
                    "real_robot_pov_path": staged_inputs.get("real_robot_pov_path"),
                    "real_robot_pov_record_count": staged_inputs.get(
                        "real_robot_pov_record_count"
                    ),
                    "missing_exact_key_record_ids": staged_inputs.get(
                        "real_robot_pov_missing_exact_key_record_ids"
                    ),
                    "missing_evidence_record_ids": staged_inputs.get(
                        "real_robot_pov_missing_evidence_record_ids"
                    ),
                    "blockers": staged_inputs.get("blockers", []),
                },
                "proof_boundary": (
                    "Generated robot POV or simulator camera media does not satisfy real robot "
                    "POV evidence; this input must come from owner robot camera/action logs."
                ),
            }
        )
    if not live_closure_evidence_ready:
        required_inputs.append(
            {
                "id": "live_robot_eval_closure_evidence",
                "title": "Job-specific package closure evidence",
                "status": _string(staged_inputs.get("status")) or "not_staged",
                "accepted_paths": [
                    {
                        "kind": "job_specific_closure_evidence",
                        "target": (
                            "<capture_root>/pipeline/robot_eval_inputs/<job_id>/"
                            "live_eval_closure_evidence.json"
                        ),
                        "required_artifacts": list(LIVE_CLOSURE_EVIDENCE_ARTIFACT_NAMES),
                    }
                ],
                "required_sections": [],
                "optional_sections": [
                    "delivery_access",
                    "rights_privacy",
                    "webapp_upstream",
                ],
                "staged_inputs": {
                    "status": staged_inputs.get("status"),
                    "live_closure_evidence_path": staged_inputs.get(
                        "live_closure_evidence_path"
                    ),
                    "live_closure_evidence_job_id": staged_inputs.get(
                        "live_closure_evidence_job_id"
                    ),
                    "blockers": staged_inputs.get("blockers", []),
                },
                "proof_boundary": (
                    "Closure evidence is an input to live_eval_closure_manifest.json; it does "
                    "not upgrade proof until the job-level closure audit passes."
                ),
            }
        )
    if not deployment_outcomes_ready:
        required_inputs.append(
            {
                "id": "real_world_deployment_outcomes",
                "title": "Real-world deployment outcome records",
                "status": _string(staged_inputs.get("status")) or "not_staged",
                "accepted_paths": [
                    {
                        "kind": "job_specific_deployment_outcome_inbox",
                        "target": (
                            "<capture_root>/pipeline/robot_eval_inputs/<job_id>/"
                            "deployment_outcomes/inbox/*.json"
                        ),
                        "required_artifacts": ["deployment_outcome_manifest.v1"],
                    }
                ],
                "required_record_fields": [
                    "task_id",
                    "scenario_id",
                    "actual_success or actual_result",
                ],
                "staged_inputs": {
                    "status": staged_inputs.get("status"),
                    "deployment_outcomes_path": staged_inputs.get(
                        "deployment_outcomes_path"
                    ),
                    "deployment_outcome_record_count": staged_inputs.get(
                        "deployment_outcome_record_count"
                    ),
                    "blockers": staged_inputs.get("blockers", []),
                },
                "proof_boundary": (
                    "Deployment outcomes must be owner-supplied actual result records; "
                    "simulator predictions cannot satisfy this input."
                ),
            }
        )
    elif not deployment_prediction_match_keys_ready:
        required_inputs.append(
            {
                "id": "predicted_vs_actual_exact_match_keys",
                "title": "Exact prediction-to-actual join keys",
                "status": _string(staged_inputs.get("status")) or "not_staged",
                "required_record_fields": [
                    "scenario_eval_run_id",
                    "scenario_variation_instance_id",
                ],
                "current_blockers": ["deployment_outcomes_missing_exact_prediction_join_keys"],
                "staged_inputs": {
                    "deployment_outcomes_path": staged_inputs.get(
                        "deployment_outcomes_path"
                    ),
                    "missing_prediction_match_key_record_ids": staged_inputs.get(
                        "deployment_outcome_missing_prediction_match_key_record_ids"
                    ),
                },
                "proof_boundary": (
                    "Task/scenario names are insufficient for predicted-vs-actual calibration; "
                    "exact run and variation keys are required."
                ),
            }
        )
    elif not deployment_owner_evidence_ready:
        required_inputs.append(
            {
                "id": "real_world_deployment_outcome_owner_evidence",
                "title": "Owner evidence for deployment outcomes",
                "status": _string(staged_inputs.get("status")) or "not_staged",
                "required_record_fields": [
                    "evidence_refs",
                    "owner_evidence_refs",
                    "owner_evidence_uri",
                    "operator_attestation",
                ],
                "current_blockers": ["deployment_outcomes_missing_owner_evidence"],
                "staged_inputs": {
                    "deployment_outcomes_path": staged_inputs.get(
                        "deployment_outcomes_path"
                    ),
                    "missing_owner_evidence_record_ids": staged_inputs.get(
                        "deployment_outcome_missing_owner_evidence_record_ids"
                    ),
                },
                "proof_boundary": (
                    "Outcome rows without owner evidence remain actual-result inputs only and "
                    "do not close real-world validation."
                ),
            }
        )
    if not policy_package_ready:
        required_inputs.append(
            {
                "id": "robot_team_policy_package",
                "title": "Robot-team policy package or trace modality",
                "status": _string(staged_inputs.get("status")) or "not_staged",
                "accepted_paths": [
                    {
                        "kind": "job_specific_policy_package",
                        "target": (
                            "<capture_root>/pipeline/robot_eval_inputs/<job_id>/"
                            "policy_package.json"
                        ),
                        "required_artifacts": list(POLICY_PACKAGE_ARTIFACT_NAMES),
                    },
                    {
                        "kind": "webapp_job_request_policy_package",
                        "source": "robot_eval_job_request.v1 policy_package",
                    },
                ],
                "supported_modalities": list(POLICY_MODALITY_ORDER),
                "staged_inputs": {
                    "status": staged_inputs.get("status"),
                    "policy_package_path": staged_inputs.get("policy_package_path"),
                    "policy_package_job_id": staged_inputs.get("policy_package_job_id"),
                    "policy_package_selected_modalities": staged_inputs.get(
                        "policy_package_selected_modalities"
                    ),
                    "blockers": staged_inputs.get("blockers", []),
                },
                "webapp_inbox_truth": {
                    "accepted_policy_package_request_count": webapp_inbox_truth.get(
                        "accepted_policy_package_request_count"
                    ),
                },
                "proof_boundary": (
                    "Policy package references are execution inputs only; policy proof requires "
                    "the gated policy execution bundle to produce attempts."
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

    required_inputs = _with_blocker_packets(required_inputs)
    enablement_inputs = _with_blocker_packets(enablement_inputs)

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
            "staged_inputs_path": staged_inputs.get("path"),
            "live_closure_evidence_path": staged_inputs.get("live_closure_evidence_path"),
            "deployment_outcomes_path": staged_inputs.get("deployment_outcomes_path"),
            "policy_package_path": staged_inputs.get("policy_package_path"),
            "real_robot_pov_path": staged_inputs.get("real_robot_pov_path"),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "arena_results_ready": staged_inputs.get("arena_results_ready"),
            "webapp_request_ready": staged_inputs.get("webapp_request_ready"),
            "live_closure_evidence_ready": staged_inputs.get("live_closure_evidence_ready"),
            "live_closure_evidence_job_id": staged_inputs.get("live_closure_evidence_job_id"),
            "deployment_outcomes_ready": staged_inputs.get("deployment_outcomes_ready"),
            "deployment_outcomes_records_ready_for_calibration": staged_inputs.get(
                "deployment_outcomes_records_ready_for_calibration"
            ),
            "deployment_outcomes_prediction_match_keys_ready": staged_inputs.get(
                "deployment_outcomes_prediction_match_keys_ready"
            ),
            "deployment_outcomes_owner_evidence_ready": staged_inputs.get(
                "deployment_outcomes_owner_evidence_ready"
            ),
            "deployment_outcomes_job_id": staged_inputs.get("deployment_outcomes_job_id"),
            "deployment_outcome_record_count": staged_inputs.get(
                "deployment_outcome_record_count"
            ),
            "deployment_outcome_prediction_match_key_record_count": staged_inputs.get(
                "deployment_outcome_prediction_match_key_record_count"
            ),
            "deployment_outcome_missing_prediction_match_key_record_ids": staged_inputs.get(
                "deployment_outcome_missing_prediction_match_key_record_ids"
            ),
            "deployment_outcome_owner_evidence_record_count": staged_inputs.get(
                "deployment_outcome_owner_evidence_record_count"
            ),
            "deployment_outcome_missing_owner_evidence_record_ids": staged_inputs.get(
                "deployment_outcome_missing_owner_evidence_record_ids"
            ),
            "policy_package_ready": staged_inputs.get("policy_package_ready"),
            "policy_package_job_id": staged_inputs.get("policy_package_job_id"),
            "policy_package_selected_modalities": staged_inputs.get(
                "policy_package_selected_modalities"
            ),
            "real_robot_pov_ready": staged_inputs.get("real_robot_pov_ready"),
            "real_robot_pov_job_id": staged_inputs.get("real_robot_pov_job_id"),
            "real_robot_pov_record_count": staged_inputs.get("real_robot_pov_record_count"),
            "real_robot_pov_exact_key_record_count": staged_inputs.get(
                "real_robot_pov_exact_key_record_count"
            ),
            "real_robot_pov_camera_video_record_count": staged_inputs.get(
                "real_robot_pov_camera_video_record_count"
            ),
            "real_robot_pov_action_log_record_count": staged_inputs.get(
                "real_robot_pov_action_log_record_count"
            ),
            "real_robot_pov_timestamp_alignment_record_count": staged_inputs.get(
                "real_robot_pov_timestamp_alignment_record_count"
            ),
            "real_robot_pov_evidence_record_count": staged_inputs.get(
                "real_robot_pov_evidence_record_count"
            ),
            "real_robot_pov_missing_exact_key_record_ids": staged_inputs.get(
                "real_robot_pov_missing_exact_key_record_ids"
            ),
            "real_robot_pov_missing_evidence_record_ids": staged_inputs.get(
                "real_robot_pov_missing_evidence_record_ids"
            ),
            "blockers": staged_inputs.get("blockers", []),
        },
        "real_world_validation_followup_request_queues": followup_request_queues,
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
    followup_queues = _as_mapping(packet.get("real_world_validation_followup_request_queues"))
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
    lines.extend(["", "## Real-World Follow-Up Request Queues"])
    queues = followup_queues.get("queues")
    lines.append(f"- Status: `{followup_queues.get('status')}`")
    lines.append(f"- Ready queues: `{followup_queues.get('ready_queue_count', 0)}`")
    lines.append(f"- Queued draft requests: `{followup_queues.get('queued_request_count', 0)}`")
    if isinstance(queues, list) and queues:
        for queue in queues:
            if not isinstance(queue, Mapping):
                continue
            lines.append(f"- `{queue.get('job_id')}`: `{queue.get('status')}`")
            if queue.get("inbox_dir"):
                lines.append(f"  - Draft inbox: `{queue.get('inbox_dir')}`")
            if queue.get("safe_processing_command"):
                lines.append(
                    f"  - Safe processing command: `{queue.get('safe_processing_command')}`"
                )
            if queue.get("blockers"):
                lines.append(
                    "  - Current blockers: "
                    f"`{', '.join(str(blocker) for blocker in queue.get('blockers') or [])}`"
                )
    else:
        lines.append("- None.")
    if followup_queues.get("proof_boundary"):
        lines.append(f"- Boundary: {followup_queues.get('proof_boundary')}")
    lines.extend(["", "## Required Inputs"])
    if isinstance(required_inputs, list) and required_inputs:
        for item in required_inputs:
            if not isinstance(item, Mapping):
                continue
            blockers = ", ".join(str(blocker) for blocker in item.get("current_blockers", []))
            blocker_packet = _as_mapping(item.get("blocker_packet"))
            lines.append(f"- `{item.get('id')}`: `{item.get('status')}`")
            if item.get("missing_fields"):
                lines.append(f"  - Missing fields: `{', '.join(item['missing_fields'])}`")
            if blockers:
                lines.append(f"  - Current blockers: `{blockers}`")
            if blocker_packet:
                lines.append(f"  - Owner: {blocker_packet.get('owner')}")
                lines.append(
                    f"  - Safe proof command: `{blocker_packet.get('safe_proof_command')}`"
                )
                lines.append(f"  - Retry condition: {blocker_packet.get('retry_condition')}")
                lines.append(
                    f"  - Disallowed workaround: {blocker_packet.get('disallowed_workaround')}"
                )
    else:
        lines.append("- None.")
    lines.extend(["", "## Enablement Inputs"])
    if isinstance(enablement_inputs, list) and enablement_inputs:
        for item in enablement_inputs:
            if not isinstance(item, Mapping):
                continue
            blockers = ", ".join(str(blocker) for blocker in item.get("current_blockers", []))
            blocker_packet = _as_mapping(item.get("blocker_packet"))
            lines.append(f"- `{item.get('id')}`: `{item.get('status')}`")
            if blockers:
                lines.append(f"  - Current blockers: `{blockers}`")
            if blocker_packet:
                lines.append(f"  - Owner: {blocker_packet.get('owner')}")
                lines.append(
                    f"  - Safe proof command: `{blocker_packet.get('safe_proof_command')}`"
                )
                lines.append(f"  - Retry condition: {blocker_packet.get('retry_condition')}")
                lines.append(
                    f"  - Disallowed workaround: {blocker_packet.get('disallowed_workaround')}"
                )
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
        output_text = _env_value(CONTROL_PLANE_OUTPUT_PATH_ENV, output_path)
        output = _output_path(capture_path, output_text)
        staged_inputs = _load_staged_inputs(
            _staged_inputs_path(output),
            capture_root=capture_path,
        )
        arena_results_text = _env_value(ARENA_RESULTS_DIR_ENV, arena_results_dir)
        if not arena_results_text and staged_inputs.get("arena_results_ready"):
            arena_results_text = _string(staged_inputs.get("arena_results_dir")) or None
        arena_results_path = Path(arena_results_text).resolve() if arena_results_text else None
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
        followup_request_queues = _real_world_validation_followup_request_queues(capture_path)
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
            staged_inputs=staged_inputs,
            followup_request_queues=followup_request_queues,
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
            "staged_inputs": staged_inputs,
            "webapp_inbox_truth": webapp_inbox_truth,
            "real_world_validation_followup_request_queues": followup_request_queues,
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
                real_robot_pov_ready=bool(staged_inputs.get("real_robot_pov_ready")),
                live_closure_evidence_ready=bool(
                    staged_inputs.get("live_closure_evidence_ready")
                ),
                deployment_outcomes_ready=bool(staged_inputs.get("deployment_outcomes_ready")),
                deployment_prediction_match_keys_ready=bool(
                    staged_inputs.get("deployment_outcomes_prediction_match_keys_ready")
                ),
                deployment_owner_evidence_ready=bool(
                    staged_inputs.get("deployment_outcomes_owner_evidence_ready")
                ),
                policy_package_ready=bool(staged_inputs.get("policy_package_ready"))
                or bool(webapp_inbox_truth.get("accepted_policy_package_request_count")),
                followup_request_queues=followup_request_queues,
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
