"""Read-only readiness audit for the first owner-GPU E2E attempt."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .live_pipeline_control_plane import (
    LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
    STAGED_INPUTS_ENV,
    WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
    WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
    WEBAPP_UPSTREAM_REQUIRED_FIELDS,
)
from .preflight_capture import build_capture_preflight_report
from .simulation_automation import SIMULATOR_FRAMEWORKS, validate_owner_gpu_system_proof


FIRST_GPU_E2E_READINESS_SCHEMA_VERSION = "first_gpu_e2e_readiness.v1"
OWNER_GPU_BLOCKER = "owner_gpu_simulator_execution_not_run"
FORWARD_URL_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"
FORWARD_TOKEN_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN"
FORWARD_CAPTURE_ROOT_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"
FORWARD_CAPTURE_ROOT_BY_SITE_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"
FORWARD_PREFLIGHT_REPORT_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT"
FORWARD_PREFLIGHT_SCHEMA_VERSION = "blueprint.webapp.robot_eval_forwarding_readiness.v1"
SIMULATOR_EXECUTION_ENV = "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"
GPU_PROVISIONING_ENV = "BLUEPRINT_ALLOW_GPU_PROVISIONING"
SIMULATOR_COMMAND_ENV = "BLUEPRINT_SIMULATOR_COMMAND"
ISAAC_LAB_ARENA_COMMAND_ENV = "BLUEPRINT_ISAAC_LAB_ARENA_COMMAND"

PROVISIONERS = ("fixture_local", "local_process", "docker_local", "vast", "runpod", "gcp")
SIMULATOR_COMMAND_LOCATIONS = ("local", "remote")
LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND = "local_first_gpu_rehearsal_request"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "first_gpu_e2e_readiness_audit_only",
    "live_provider_calls_performed": False,
    "simulator_execution_performed": False,
    "gpu_provisioning_performed": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "non_ranking_operational_claim_validated": False,
    "rank_fidelity_result_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return _truthy(os.getenv(name))


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _append_unique(target: List[str], values: Iterable[str]) -> None:
    for value in values:
        text = _string(value)
        if text and text not in target:
            target.append(text)


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    _append_unique(out, (_string(item) for item in values))
    return out


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "ready": path.is_file(),
    }


def _placeholder_like(value: str, *, scene_id: str, capture_id: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    if lowered in {scene_id.lower(), capture_id.lower(), "placeholder", "unknown"}:
        return True
    return any(token in lowered for token in ("replace_with", "todo", "fake-", "sample-"))


def _first_executable(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    if not parts:
        return ""
    if "=" in parts[0] and len(parts) > 1:
        return parts[1]
    return parts[0]


def _command_status(command: str, *, command_location: str = "local") -> Dict[str, Any]:
    executable = _first_executable(command)
    if not executable:
        return {
            "configured": False,
            "executable": None,
            "executable_found": False,
            "command_location": command_location,
            "executable_check_performed": False,
            "blockers": ["missing_simulator_command"],
        }
    if command_location == "remote":
        return {
            "configured": True,
            "executable": executable,
            "executable_found": None,
            "command_location": "remote",
            "executable_check_performed": False,
            "blockers": [],
            "warnings": ["simulator_command_executable_not_checked_remote_vm"],
        }
    found = bool(shutil.which(executable)) if "/" not in executable else Path(executable).exists()
    return {
        "configured": True,
        "executable": executable,
        "executable_found": found,
        "command_location": "local",
        "executable_check_performed": True,
        "blockers": [] if found else ["simulator_command_executable_missing"],
        "warnings": [],
    }


def _capture_preflight_stage(capture_root: Path) -> Dict[str, Any]:
    try:
        report = build_capture_preflight_report(capture_root)
    except Exception as exc:  # pragma: no cover - defensive, surfaced in manifest
        return {
            "status": "blocked",
            "ready": False,
            "blockers": ["capture_preflight_exception"],
            "error": str(exc),
        }
    missing = _string_list(report.get("missing_required_inputs"))
    blockers = [f"missing_raw_input:{item}" for item in missing]
    return {
        "status": report.get("status"),
        "mode_decision": report.get("mode_decision"),
        "human_review_required": bool(report.get("human_review_required")),
        "ready": not blockers,
        "blockers": blockers,
        "notes": report.get("notes") or [],
    }


def _requested_outputs_stage(capture_root: Path) -> Dict[str, Any]:
    sources = [
        _read_optional_mapping(capture_root / "raw" / "manifest.json"),
        _read_optional_mapping(capture_root / "capture_descriptor.json"),
        _read_optional_mapping(capture_root / "pipeline_handoff.json"),
        _read_optional_mapping(capture_root / "pipeline" / "opportunity_handoff.json"),
    ]
    requested: List[str] = []
    for source in sources:
        _append_unique(requested, _string_list(source.get("requested_outputs")))
        _append_unique(requested, _string_list(source.get("requested_lanes")))
    required = ("robot_eval_dataset", "task_evaluation_run")
    blockers = [f"missing_requested_output:{item}" for item in required if item not in requested]
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "requested_outputs_or_lanes": requested,
        "required_outputs": list(required),
        "blockers": blockers,
    }


def _webapp_upstream_truth_stage(
    capture_root: Path,
    *,
    scene_id: str,
    capture_id: str,
    staged_inputs_path: str | Path | None = None,
) -> Dict[str, Any]:
    sources = [
        _read_optional_mapping(capture_root / "capture_descriptor.json"),
        _read_optional_mapping(capture_root / "raw" / "manifest.json"),
        _read_optional_mapping(capture_root / "pipeline_handoff.json"),
        _read_optional_mapping(capture_root / "pipeline" / "opportunity_handoff.json"),
    ]
    fields = ("site_submission_id", "request_id", "buyer_request_id", "capture_job_id")
    values: Dict[str, str] = {}
    source_artifacts: Dict[str, str | None] = {field: None for field in fields}
    for field in fields:
        for index, source in enumerate(sources):
            candidate = _string(source.get(field))
            if candidate:
                values[field] = candidate
                source_artifacts[field] = (
                    "capture_descriptor.json",
                    "raw/manifest.json",
                    "pipeline_handoff.json",
                    "pipeline/opportunity_handoff.json",
                )[index]
                break
    owner_system = _mapping(_read_optional_mapping(capture_root / "pipeline_handoff.json").get("owner_system"))
    if "request_id" not in values:
        values["request_id"] = _string(owner_system.get("request_id"))
        if values["request_id"]:
            source_artifacts["request_id"] = "pipeline_handoff.json owner_system"

    staged_path = _default_staged_inputs_path(capture_root, staged_inputs_path)
    staged_request_used = False
    staged_request_warnings: List[str] = []
    if staged_path.is_file() and any(
        _placeholder_like(values.get(field, ""), scene_id=scene_id, capture_id=capture_id)
        for field in fields
    ):
        try:
            staged_payload = read_json_any(staged_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            staged_request_warnings.append(f"staged_webapp_request_read_failed:{type(exc).__name__}")
        else:
            if isinstance(staged_payload, Mapping):
                source_kind = _string(
                    _mapping(staged_payload.get("webapp_request")).get("source_kind")
                    or staged_payload.get("source_kind")
                )
                local_rehearsal_only = (
                    bool(staged_payload.get("local_rehearsal_only"))
                    or source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
                )
                if local_rehearsal_only:
                    staged_request_warnings.append("staged_webapp_request_local_rehearsal_only")
                else:
                    webapp = _mapping(staged_payload.get("webapp_request"))
                    request_path_text = _string(webapp.get("target_path") or webapp.get("path"))
                    if request_path_text:
                        request_path = Path(request_path_text).expanduser()
                        if request_path.is_file():
                            try:
                                request_payload = read_json_any(request_path)
                            except (OSError, ValueError, json.JSONDecodeError) as exc:
                                staged_request_warnings.append(
                                    f"staged_webapp_request_payload_read_failed:{type(exc).__name__}"
                                )
                            else:
                                request = (
                                    _request_from_webapp_payload(request_payload)
                                    if isinstance(request_payload, Mapping)
                                    else {}
                                )
                                site_package = _mapping(request.get("site_package"))
                                if request and _path_matches(
                                    _string(site_package.get("capture_root")),
                                    capture_root,
                                ):
                                    for field in fields:
                                        if _placeholder_like(
                                            values.get(field, ""),
                                            scene_id=scene_id,
                                            capture_id=capture_id,
                                        ):
                                            candidate = _nested_webapp_source(request, field)
                                            if candidate:
                                                values[field] = candidate
                                                source_artifacts[field] = (
                                                    "pipeline/live_pipeline_staged_inputs.json "
                                                    "robot_eval_job_request.v1"
                                                )
                                                staged_request_used = True
                                elif request:
                                    staged_request_warnings.append(
                                        "staged_webapp_request_capture_root_mismatch"
                                    )
                        else:
                            staged_request_warnings.append("staged_webapp_request_file_missing")
                    else:
                        staged_request_warnings.append("staged_webapp_request_path_missing")
            else:
                staged_request_warnings.append("staged_webapp_inputs_not_json_object")
    blockers: List[str] = []
    for field in fields:
        value = values.get(field, "")
        if _placeholder_like(value, scene_id=scene_id, capture_id=capture_id):
            blockers.append(f"missing_or_placeholder_webapp_{field}")
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "fields": {field: bool(values.get(field)) for field in fields},
        "values_redacted": {field: bool(values.get(field)) for field in fields},
        "source_artifacts": source_artifacts,
        "staged_webapp_request_used": staged_request_used,
        "staged_webapp_request_path": str(staged_path),
        "blockers": blockers,
        "warnings": staged_request_warnings,
    }


def _parse_by_site_override() -> Dict[str, Any]:
    raw = _string(os.getenv(FORWARD_CAPTURE_ROOT_BY_SITE_ENV))
    if not raw:
        return {"status": "not_configured", "overrides": {}, "blockers": []}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "status": "blocked",
            "overrides": {},
            "blockers": [f"invalid_env_{FORWARD_CAPTURE_ROOT_BY_SITE_ENV}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "blocked",
            "overrides": {},
            "blockers": [f"invalid_env_{FORWARD_CAPTURE_ROOT_BY_SITE_ENV}"],
        }
    overrides = {str(key): _string(value) for key, value in payload.items() if _string(value)}
    return {"status": "configured", "overrides": overrides, "blockers": []}


def _default_forwarding_preflight_path(explicit: str | Path | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = _string(os.getenv(FORWARD_PREFLIGHT_REPORT_ENV))
    if configured:
        return Path(configured).expanduser().resolve()
    return None


def _webapp_forwarding_preflight_stage(
    *,
    webapp_site_slug: str,
    require_webapp_forwarding: bool,
    preflight_report_path: str | Path | None,
) -> Dict[str, Any]:
    path = _default_forwarding_preflight_path(preflight_report_path)
    if path is None:
        return {
            "configured": False,
            "ready": False,
            "path": None,
            "status": "not_configured",
            "blockers": [],
            "warnings": [],
            "site_slug_covered": False,
            "single_capture_root_override_configured": False,
        }
    if not path.is_file():
        return {
            "configured": True,
            "ready": False,
            "path": str(path),
            "status": "blocked",
            "blockers": ["webapp_forwarding_preflight_report_missing"],
            "warnings": [],
            "site_slug_covered": False,
            "single_capture_root_override_configured": False,
        }
    try:
        payload = read_json_any(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "configured": True,
            "ready": False,
            "path": str(path),
            "status": "blocked",
            "blockers": [f"webapp_forwarding_preflight_report_read_failed:{type(exc).__name__}"],
            "warnings": [],
            "site_slug_covered": False,
            "single_capture_root_override_configured": False,
        }
    if not isinstance(payload, Mapping):
        return {
            "configured": True,
            "ready": False,
            "path": str(path),
            "status": "blocked",
            "blockers": ["webapp_forwarding_preflight_report_not_json_object"],
            "warnings": [],
            "site_slug_covered": False,
            "single_capture_root_override_configured": False,
        }

    blockers: List[str] = []
    warnings: List[str] = []
    if payload.get("schema_version") != FORWARD_PREFLIGHT_SCHEMA_VERSION:
        blockers.append("webapp_forwarding_preflight_schema_mismatch")
    status = _string(payload.get("status"))
    ready_statuses = {
        "ready_for_required_forwarding",
        "ready_for_required_forwarding_with_probe",
    }
    if not require_webapp_forwarding:
        ready_statuses.update(
            {
                "ready_for_optional_forwarding",
                "ready_for_optional_forwarding_with_probe",
            }
        )
    if status not in ready_statuses:
        blockers.append(f"webapp_forwarding_preflight_status:{status or 'unknown'}")
    if require_webapp_forwarding and payload.get("forwarding_required") is not True:
        blockers.append("webapp_forwarding_preflight_not_required_mode")
    if payload.get("endpoint_configured") is not True:
        blockers.append("webapp_forwarding_preflight_endpoint_not_configured")

    configured_env = _mapping(payload.get("configured_env"))
    forward_url = _mapping(configured_env.get("forward_url"))
    if forward_url.get("valid") is not True:
        blockers.append("webapp_forwarding_preflight_forward_url_invalid")
    forward_token = _mapping(configured_env.get("forward_token"))
    if forward_token.get("configured") is not True:
        blockers.append("webapp_forwarding_preflight_token_not_configured")
    if forward_token.get("redacted") is not True:
        blockers.append("webapp_forwarding_preflight_token_not_redacted")
    timeout = _mapping(configured_env.get("forward_timeout_ms"))
    if timeout and timeout.get("valid") is not True:
        blockers.append("webapp_forwarding_preflight_timeout_invalid")

    capture_root_by_site = _mapping(configured_env.get("capture_root_by_site_json"))
    if capture_root_by_site.get("configured") and capture_root_by_site.get("valid") is not True:
        blockers.append("webapp_forwarding_preflight_capture_root_map_invalid")
    site_slugs = {
        _string(item)
        for item in capture_root_by_site.get("site_slugs") or []
        if _string(item)
    }
    single_override = _mapping(configured_env.get("single_capture_root_override"))
    single_override_configured = single_override.get("configured") is True
    site_slug_covered = bool(webapp_site_slug and webapp_site_slug in site_slugs)
    if webapp_site_slug and not site_slug_covered and not single_override_configured:
        blockers.append("webapp_forwarding_preflight_missing_site_slug")
    if not webapp_site_slug and not single_override_configured:
        warnings.append("webapp_forwarding_preflight_site_slug_not_checked")

    report_blockers = _string_list(payload.get("blockers"))
    if report_blockers:
        blockers.append("webapp_forwarding_preflight_report_has_blockers")
    proof_boundary = _mapping(payload.get("proof_boundary"))
    required_boundaries = (
        "command_is_read_only",
        "no_job_queued",
        "no_pipeline_mutation_requested",
        "no_gpu_allocated",
        "no_simulator_execution_proven",
        "no_rank_fidelity_result_proven",
        "no_public_claim_upgrade_allowed",
    )
    for field in required_boundaries:
        if proof_boundary.get(field) is not True:
            blockers.append(f"webapp_forwarding_preflight_boundary_missing:{field}")

    probe = _mapping(payload.get("probe"))
    if probe.get("requested") is True and probe.get("status") != "reachable":
        blockers.append("webapp_forwarding_preflight_probe_not_reachable")
    if probe.get("requested") is not True:
        warnings.append("webapp_forwarding_preflight_not_network_probed")

    return {
        "configured": True,
        "ready": not blockers,
        "path": str(path),
        "status": "ready" if not blockers else "blocked",
        "preflight_status": status or None,
        "forwarding_required": bool(payload.get("forwarding_required")),
        "endpoint_configured": bool(payload.get("endpoint_configured")),
        "site_slug_covered": site_slug_covered,
        "site_slugs": sorted(site_slugs),
        "single_capture_root_override_configured": single_override_configured,
        "probe_status": probe.get("status"),
        "blockers": blockers,
        "warnings": warnings,
        "proof_boundary": (
            "WebApp forwarding preflight proves configuration and optional intake-audit "
            "reachability only; it does not submit a job, allocate GPU workers, run a "
            "simulator, or prove generated-world rank fidelity."
        ),
    }


def _request_from_webapp_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        request = payload.get("job_request")
        return dict(request) if isinstance(request, Mapping) else {}
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return dict(payload)
    return {}


def _nested_webapp_source(request: Mapping[str, Any], field: str) -> str:
    source = _mapping(request.get("source"))
    selection = _mapping(source.get("selection_state"))
    owner_system = _mapping(request.get("owner_system"))
    site_package = _mapping(request.get("site_package"))
    for candidate in (request, source, selection, owner_system, site_package):
        value = _string(candidate.get(field))
        if value:
            return value
    return ""


def _path_matches(value: str, expected: Path) -> bool:
    if not value:
        return False
    try:
        return Path(value).expanduser().resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def _default_staged_inputs_path(capture_root: Path, explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = _string(os.getenv(STAGED_INPUTS_ENV))
    if configured:
        return Path(configured).expanduser().resolve()
    return capture_root / "pipeline" / "live_pipeline_staged_inputs.json"


def _webapp_staged_request_stage(
    capture_root: Path,
    *,
    staged_inputs_path: str | Path | None,
    require_webapp_staged_request: bool,
    allow_local_webapp_rehearsal: bool,
) -> Dict[str, Any]:
    resolved_path = _default_staged_inputs_path(capture_root, staged_inputs_path)
    if not require_webapp_staged_request:
        return {
            "status": "not_required",
            "ready": True,
            "required": False,
            "path": str(resolved_path),
            "blockers": [],
        }
    if not resolved_path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "required": True,
            "path": str(resolved_path),
            "blockers": ["missing_webapp_staged_inputs"],
        }
    try:
        payload = read_json_any(resolved_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "required": True,
            "path": str(resolved_path),
            "blockers": [f"webapp_staged_inputs_read_failed:{type(exc).__name__}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "blocked",
            "ready": False,
            "required": True,
            "path": str(resolved_path),
            "blockers": ["webapp_staged_inputs_not_json_object"],
        }
    blockers: List[str] = []
    if payload.get("schema_version") != LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION:
        blockers.append("webapp_staged_inputs_schema_mismatch")
    configured_capture_root = _string(payload.get("configured_capture_root"))
    if not configured_capture_root:
        blockers.append("webapp_staged_inputs_missing_configured_capture_root")
    elif not _path_matches(configured_capture_root, capture_root):
        blockers.append("webapp_staged_inputs_capture_root_mismatch")

    warnings: List[str] = []
    webapp = _mapping(payload.get("webapp_request"))
    source_kind = _string(webapp.get("source_kind") or payload.get("source_kind"))
    local_rehearsal_only = bool(payload.get("local_rehearsal_only")) or (
        source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    )
    local_rehearsal_blocker_recorded = False
    local_rehearsal_warning_recorded = False

    def record_local_rehearsal_boundary() -> None:
        nonlocal local_rehearsal_blocker_recorded, local_rehearsal_warning_recorded
        if not allow_local_webapp_rehearsal and not local_rehearsal_blocker_recorded:
            blockers.append("webapp_staged_inputs_local_rehearsal_only")
            local_rehearsal_blocker_recorded = True
        if not local_rehearsal_warning_recorded:
            warnings.append("local_webapp_rehearsal_not_live_forwarding_proof")
            local_rehearsal_warning_recorded = True

    if local_rehearsal_only:
        record_local_rehearsal_boundary()
    staged = bool(webapp.get("staged"))
    ready = bool(webapp.get("ready"))
    request_path_text = _string(webapp.get("target_path") or webapp.get("path"))
    if not staged:
        blockers.append("webapp_request_not_staged")
    if not ready:
        blockers.append("webapp_request_not_ready")
    if not request_path_text:
        blockers.append("webapp_request_path_missing")

    fields_present = {field: False for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS}
    job_id = _string(webapp.get("job_id"))
    request_capture_root = ""
    request_path = Path(request_path_text).expanduser() if request_path_text else None
    if request_path is not None:
        if not request_path.is_file():
            blockers.append("webapp_request_file_missing")
        else:
            try:
                request_payload = read_json_any(request_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"webapp_request_read_failed:{type(exc).__name__}")
            else:
                if isinstance(request_payload, Mapping):
                    source_kind = source_kind or _string(request_payload.get("source_kind"))
                    if source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND:
                        local_rehearsal_only = True
                        record_local_rehearsal_boundary()
                    request = _request_from_webapp_payload(request_payload)
                else:
                    request = {}
                if not request:
                    blockers.append("webapp_request_not_robot_eval_job_request_v1")
                else:
                    source_kind = source_kind or _string(request.get("source_kind"))
                    if source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND:
                        local_rehearsal_only = True
                        record_local_rehearsal_boundary()
                    job_id = job_id or _string(request.get("job_id"))
                    site_package = _mapping(request.get("site_package"))
                    request_capture_root = _string(site_package.get("capture_root"))
                    if not _path_matches(request_capture_root, capture_root):
                        blockers.append("webapp_request_capture_root_mismatch")
                    fields_present = {
                        field: bool(_nested_webapp_source(request, field))
                        for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS
                    }
                    missing = [field for field, present in fields_present.items() if not present]
                    if missing:
                        blockers.append("webapp_request_missing_required_upstream_ids")

    if not job_id:
        blockers.append("webapp_request_job_id_missing")
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "required": True,
        "path": str(resolved_path),
        "job_id": job_id or None,
        "request_path": request_path_text or None,
        "source_kind": source_kind or None,
        "local_rehearsal_only": local_rehearsal_only,
        "local_rehearsal_allowed": allow_local_webapp_rehearsal,
        "fields_present": fields_present,
        "request_capture_root_configured": bool(request_capture_root),
        "blockers": blockers,
        "warnings": warnings,
        "proof_boundary": (
            "Staged WebApp request evidence proves request handoff shape only; it does not run "
            "the job, simulator, policy, contact, safety, or readiness proof."
        ),
    }


def _webapp_forwarding_stage(
    capture_root: Path,
    *,
    webapp_site_slug: str,
    require_webapp_forwarding: bool,
    webapp_forwarding_preflight_path: str | Path | None = None,
) -> Dict[str, Any]:
    blockers: List[str] = []
    url_present = bool(_string(os.getenv(FORWARD_URL_ENV)))
    token_present = bool(_string(os.getenv(FORWARD_TOKEN_ENV)))
    preflight = _webapp_forwarding_preflight_stage(
        webapp_site_slug=webapp_site_slug,
        require_webapp_forwarding=require_webapp_forwarding,
        preflight_report_path=webapp_forwarding_preflight_path,
    )
    _append_unique(blockers, preflight.get("blockers") or [])
    preflight_ready = bool(preflight.get("ready"))
    url_evidence_present = url_present or preflight_ready
    token_evidence_present = token_present or preflight_ready
    if require_webapp_forwarding and not url_evidence_present:
        blockers.append(f"missing_env_{FORWARD_URL_ENV}")
    if require_webapp_forwarding and not token_evidence_present:
        blockers.append(f"missing_env_{FORWARD_TOKEN_ENV}")

    by_site = _parse_by_site_override()
    _append_unique(blockers, by_site.get("blockers") or [])
    global_override = _string(os.getenv(FORWARD_CAPTURE_ROOT_ENV))
    expected = str(capture_root.resolve())
    site_override = ""
    if webapp_site_slug:
        site_override = _string(_mapping(by_site.get("overrides")).get(webapp_site_slug))
    elif require_webapp_forwarding:
        blockers.append("missing_webapp_site_slug_for_capture_root_override")

    override_source = None
    override_value = ""
    if site_override:
        override_source = FORWARD_CAPTURE_ROOT_BY_SITE_ENV
        override_value = site_override
    elif global_override:
        override_source = FORWARD_CAPTURE_ROOT_ENV
        override_value = global_override
    elif preflight_ready and (
        bool(preflight.get("site_slug_covered"))
        or bool(preflight.get("single_capture_root_override_configured"))
    ):
        override_source = FORWARD_PREFLIGHT_REPORT_ENV
        override_value = "<redacted-preflight-report>"
    elif require_webapp_forwarding:
        blockers.append("missing_pipeline_capture_root_override_for_webapp_synced_artifact")

    if (
        override_value
        and override_source != FORWARD_PREFLIGHT_REPORT_ENV
        and str(Path(override_value).expanduser().resolve()) != expected
    ):
        blockers.append("pipeline_capture_root_override_does_not_match_capture_root")

    warnings = _string_list(preflight.get("warnings"))
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "required": require_webapp_forwarding,
        "webapp_site_slug": webapp_site_slug or None,
        "forward_url_configured": url_present,
        "forward_token_configured": token_present,
        "forward_url_evidence_present": url_evidence_present,
        "forward_token_evidence_present": token_evidence_present,
        "capture_root_override_source": override_source,
        "capture_root_override_configured": bool(override_value),
        "forwarding_preflight": preflight,
        "blockers": blockers,
        "warnings": warnings,
    }


def _pipeline_handoff_stage(capture_root: Path) -> Dict[str, Any]:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    artifacts = {
        "gpu_handoff_packet": _artifact(automation_dir / "gpu_handoff_packet.json"),
        "gpu_owner_system_proof_schema": _artifact(
            automation_dir / "gpu_owner_system_proof_schema.json"
        ),
        "gpu_run_checklist": _artifact(automation_dir / "gpu_run_checklist.md"),
        "owner_gpu_blocked_manifest": _artifact(
            automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json"
        ),
        "simulator_engine_plugin_registry": _artifact(
            automation_dir / "simulator_engine_plugin_registry.json"
        ),
    }
    blockers = [
        f"missing_artifact:{name}" for name, status in artifacts.items() if not status["ready"]
    ]
    gpu_handoff = _read_optional_mapping(automation_dir / "gpu_handoff_packet.json")
    if gpu_handoff:
        if gpu_handoff.get("status") != "ready_for_owner_gpu_preflight_handoff":
            blockers.append("gpu_handoff_packet_not_ready")
        if bool(gpu_handoff.get("rank_fidelity_result_proven")):
            blockers.append("gpu_handoff_illegally_marks_rank_fidelity")
        if bool(gpu_handoff.get("public_claim_upgrade_allowed")):
            blockers.append("gpu_handoff_illegally_allows_public_claim_upgrade")
        for blocker in _string_list(gpu_handoff.get("blockers")):
            if blocker != OWNER_GPU_BLOCKER:
                blockers.append(blocker)
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "artifacts": artifacts,
        "expected_pre_gpu_blocker": OWNER_GPU_BLOCKER,
        "hard_preflight_blockers": _string_list(gpu_handoff.get("hard_preflight_blockers")),
        "spawn_validation_summary": _mapping(gpu_handoff.get("spawn_validation_summary")),
        "pre_gpu_blocker_details": [
            dict(item)
            for item in gpu_handoff.get("pre_gpu_blocker_details") or []
            if isinstance(item, Mapping)
        ],
        "blockers": blockers,
    }


def _simulator_runtime_stage(
    *,
    simulator: str,
    provisioner: str,
    simulator_command: str,
    simulator_command_location: str,
    require_gpu_gates: bool,
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []
    command_status = _command_status(
        simulator_command,
        command_location=simulator_command_location,
    )
    _append_unique(blockers, command_status.get("blockers") or [])
    _append_unique(warnings, command_status.get("warnings") or [])
    simulator_gate = _env_truthy(SIMULATOR_EXECUTION_ENV)
    provisioning_gate = _env_truthy(GPU_PROVISIONING_ENV)
    if require_gpu_gates and not simulator_gate:
        blockers.append(f"missing_env_{SIMULATOR_EXECUTION_ENV}")
    if provisioner != "fixture_local" and not provisioning_gate:
        warnings.append(f"missing_env_{GPU_PROVISIONING_ENV}")
    if provisioner in {"runpod", "vast", "gcp"}:
        warnings.append(f"{provisioner}_allocation_is_external_or_request_manifest_only")
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "simulator": simulator,
        "provisioner": provisioner,
        "simulator_execution_gate_enabled": simulator_gate,
        "gpu_provisioning_gate_enabled": provisioning_gate,
        "command": {
            "configured": command_status["configured"],
            "executable": command_status["executable"],
            "executable_found": command_status["executable_found"],
            "command_location": command_status["command_location"],
            "executable_check_performed": command_status["executable_check_performed"],
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def _owner_gpu_proof_stage(capture_root: Path) -> Dict[str, Any]:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    proof_path = automation_dir / "gpu_owner_system_proof.json"
    manifest_path = automation_dir / "owner_gpu_simulator_execution_proof_manifest.json"
    proof = validate_owner_gpu_system_proof(
        proof_path=proof_path,
        capture_root=capture_root,
        output_path=manifest_path,
    )
    missing_is_expected = proof.get("status") == "missing" and proof.get("blockers") == [
        OWNER_GPU_BLOCKER
    ]
    blockers = []
    if proof.get("status") == "blocked" and proof_path.is_file():
        blockers.append("owner_gpu_proof_present_but_blocked")
    return {
        "status": proof.get("status"),
        "ready": bool(proof.get("owner_gpu_simulator_execution_proven")),
        "missing_is_expected_before_first_gpu_run": missing_is_expected,
        "proof_path": str(proof_path),
        "proof_manifest_path": str(manifest_path),
        "blockers": blockers,
        "proof_blockers": proof.get("blockers") or [],
        "claim_boundary": proof.get("claim_boundary") or CLAIM_BOUNDARY,
    }


def _default_simulator_command(simulator: str, explicit: str | None) -> str:
    if explicit is not None:
        return _string(explicit)
    if simulator == "isaac_lab_arena":
        return _string(os.getenv(ISAAC_LAB_ARENA_COMMAND_ENV) or os.getenv(SIMULATOR_COMMAND_ENV))
    return _string(os.getenv(SIMULATOR_COMMAND_ENV))


def build_first_gpu_e2e_readiness(
    *,
    capture_root: str | Path,
    webapp_site_slug: str = "",
    webapp_staged_inputs_path: str | Path | None = None,
    webapp_forwarding_preflight_path: str | Path | None = None,
    simulator: str = "isaac_sim",
    provisioner: str = "runpod",
    simulator_command: str | None = None,
    simulator_command_location: str = "local",
    require_webapp_forwarding: bool = True,
    require_webapp_staged_request: bool = True,
    allow_local_webapp_rehearsal: bool = False,
    require_gpu_gates: bool = True,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    selected_simulator = _string(simulator) or "isaac_sim"
    selected_provisioner = _string(provisioner) or "runpod"
    selected_command_location = _string(simulator_command_location) or "local"
    if selected_command_location not in SIMULATOR_COMMAND_LOCATIONS:
        selected_command_location = "local"
    command = _default_simulator_command(selected_simulator, simulator_command)
    generated_at = utc_now_iso()

    stages = {
        "capture_preflight": _capture_preflight_stage(context.capture_root),
        "requested_outputs": _requested_outputs_stage(context.capture_root),
        "webapp_upstream_truth": _webapp_upstream_truth_stage(
            context.capture_root,
            scene_id=context.scene_id,
            capture_id=context.capture_id,
            staged_inputs_path=webapp_staged_inputs_path,
        ),
        "webapp_forwarding": _webapp_forwarding_stage(
            context.capture_root,
            webapp_site_slug=webapp_site_slug,
            require_webapp_forwarding=require_webapp_forwarding,
            webapp_forwarding_preflight_path=webapp_forwarding_preflight_path,
        ),
        "webapp_staged_request": _webapp_staged_request_stage(
            context.capture_root,
            staged_inputs_path=webapp_staged_inputs_path,
            require_webapp_staged_request=require_webapp_staged_request,
            allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
        ),
        "pipeline_gpu_handoff": _pipeline_handoff_stage(context.capture_root),
        "simulator_runtime": _simulator_runtime_stage(
            simulator=selected_simulator,
            provisioner=selected_provisioner,
            simulator_command=command,
            simulator_command_location=selected_command_location,
            require_gpu_gates=require_gpu_gates,
        ),
        "owner_gpu_proof": _owner_gpu_proof_stage(context.capture_root),
    }

    blockers: List[str] = []
    warnings: List[str] = []
    for stage_name, stage in stages.items():
        for blocker in _string_list(stage.get("blockers")):
            blockers.append(f"{stage_name}:{blocker}")
        for warning in _string_list(stage.get("warnings")):
            warnings.append(f"{stage_name}:{warning}")

    proof_stage = _mapping(stages["owner_gpu_proof"])
    owner_gpu_proof_ready = bool(proof_stage.get("ready"))
    ready_for_attempt = not blockers
    if blockers:
        status = "blocked"
    elif owner_gpu_proof_ready:
        status = "owner_gpu_proof_present_audit_closure_next"
    else:
        status = "ready_for_owner_gpu_attempt"

    return {
        "schema_version": FIRST_GPU_E2E_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "capture_root": str(context.capture_root),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": status,
        "ready_for_first_gpu_attempt": ready_for_attempt,
        "owner_gpu_proof_ready": owner_gpu_proof_ready,
        "simulator": selected_simulator,
        "provisioner": selected_provisioner,
        "simulator_command_location": selected_command_location,
        "stages": stages,
        "blockers": blockers,
        "warnings": warnings,
        "next_commands": {
            "local_pipeline": (
                f"blueprint-run-e2e --capture-root {context.capture_root} --provider local "
                "--pipeline-lane current --run-evaluation-prep --evaluation-prep-provider manual"
            ),
            "simulation_automation": f"blueprint-run-simulation-automation --capture-root {context.capture_root}",
            "gpu_simulation": (
                "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
                f"blueprint-run-simulation-automation --capture-root {context.capture_root} "
                f"--allow-simulator-execution --allow-simulator {selected_simulator} "
                f"--simulator-command \"{selected_simulator}=<owner proof command>\""
            ),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit readiness for the first sample-video to owner-GPU E2E attempt"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--webapp-site-slug", default="")
    parser.add_argument("--webapp-staged-inputs", default=None)
    parser.add_argument("--webapp-forwarding-preflight", default=None)
    parser.add_argument("--simulator", choices=SIMULATOR_FRAMEWORKS, default="isaac_sim")
    parser.add_argument("--provisioner", choices=PROVISIONERS, default="runpod")
    parser.add_argument("--simulator-command", default=None)
    parser.add_argument(
        "--simulator-command-location",
        choices=SIMULATOR_COMMAND_LOCATIONS,
        default="local",
        help=(
            "Use local to require the command executable on this machine; use remote when "
            "the command is expected to exist only inside the owner GPU VM."
        ),
    )
    parser.add_argument("--no-require-webapp-forwarding", action="store_true")
    parser.add_argument("--no-require-webapp-staged-request", action="store_true")
    parser.add_argument(
        "--allow-local-webapp-rehearsal",
        action="store_true",
        help=(
            "Allow a staged local rehearsal WebApp request to satisfy the staged-request "
            "gate. This is not live forwarding proof."
        ),
    )
    parser.add_argument("--no-require-gpu-gates", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = build_first_gpu_e2e_readiness(
        capture_root=args.capture_root,
        webapp_site_slug=args.webapp_site_slug,
        webapp_staged_inputs_path=args.webapp_staged_inputs,
        webapp_forwarding_preflight_path=args.webapp_forwarding_preflight,
        simulator=args.simulator,
        provisioner=args.provisioner,
        simulator_command=args.simulator_command,
        simulator_command_location=args.simulator_command_location,
        require_webapp_forwarding=not args.no_require_webapp_forwarding,
        require_webapp_staged_request=not args.no_require_webapp_staged_request,
        allow_local_webapp_rehearsal=args.allow_local_webapp_rehearsal,
        require_gpu_gates=not args.no_require_gpu_gates,
    )
    output = Path(args.output) if args.output else Path(args.capture_root) / "pipeline" / "first_gpu_e2e_readiness_manifest.json"
    ensure_dir(output.parent)
    write_json(output, result)
    print(f"[first-gpu-e2e-readiness] status={result['status']}")
    print(f"[first-gpu-e2e-readiness] manifest={output}")
    if result["blockers"]:
        print("[first-gpu-e2e-readiness] blockers=" + ",".join(result["blockers"]))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
