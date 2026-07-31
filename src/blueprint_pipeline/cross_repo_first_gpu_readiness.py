"""Cross-repo readiness audit for the first owner-GPU E2E run."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .artifact_storage import default_artifact_cache_root
from .first_gpu_e2e_readiness import (
    PROVISIONERS,
    SIMULATOR_COMMAND_LOCATIONS,
    build_first_gpu_e2e_readiness,
)
from .simulation_automation import SIMULATOR_FRAMEWORKS


CROSS_REPO_FIRST_GPU_READINESS_SCHEMA_VERSION = "cross_repo_first_gpu_readiness.v1"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _default_pipeline_repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_adjacent_repo(name: str) -> Path:
    return _default_pipeline_repo().parent / name


def _resolve_repo(path: str | Path | None, *, default: Path) -> Path:
    return Path(path).expanduser().resolve() if path else default.resolve()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _append_unique(target: List[str], values: Iterable[str]) -> None:
    for value in values:
        text = _string(value)
        if text and text not in target:
            target.append(text)


def _string_list_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    _append_unique(result, (str(value) for value in values))
    return result


def _read_json_mapping(path: Path) -> tuple[Dict[str, Any] | None, str | None]:
    try:
        payload = read_json_any(path)
    except Exception as exc:  # pragma: no cover - surfaced as manifest evidence
        return None, f"invalid_json:{path.name}:{exc.__class__.__name__}"
    if not isinstance(payload, Mapping):
        return None, f"invalid_json_payload:{path.name}:{type(payload).__name__}"
    return dict(payload), None


def _runtime_preflight_result_summary(result_path: str | Path | None) -> Dict[str, Any]:
    path_text = _string(result_path)
    if not path_text:
        return {
            "path": None,
            "exists": False,
            "status": None,
            "ready_for_owner_command_attempt": False,
            "blockers": ["gpu_vm_runtime_preflight_result_path_missing"],
        }
    path = Path(path_text)
    if not path.is_file():
        return {
            "path": path_text,
            "exists": False,
            "status": None,
            "ready_for_owner_command_attempt": False,
            "blockers": ["gpu_vm_runtime_preflight_result_missing"],
        }
    payload, error = _read_json_mapping(path)
    if error or payload is None:
        return {
            "path": path_text,
            "exists": True,
            "status": None,
            "ready_for_owner_command_attempt": False,
            "blockers": [error or "gpu_vm_runtime_preflight_result_invalid"],
        }
    status = _string(payload.get("status"))
    blockers = [
        f"gpu_vm_runtime_preflight_result_blocker:{item}"
        for item in payload.get("blockers") or []
    ]
    if status != "ready_for_owner_command_attempt":
        blockers.append(
            f"gpu_vm_runtime_preflight_result_status:{status or 'unknown'}"
        )
    return {
        "path": path_text,
        "exists": True,
        "status": status or None,
        "ready_for_owner_command_attempt": not blockers,
        "blockers": _string_list_unique(blockers),
    }


def _file_contains_check(
    root: Path,
    relative_path: str,
    *,
    required: Sequence[tuple[str, str]],
) -> Dict[str, Any]:
    path = root / relative_path
    blockers: List[str] = []
    matched: Dict[str, bool] = {}
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "ready": False,
            "matched": {label: False for label, _needle in required},
            "blockers": [f"missing_file:{relative_path}"],
        }
    text = _read_text(path)
    for label, needle in required:
        present = needle in text
        matched[label] = present
        if not present:
            blockers.append(f"missing_contract_text:{relative_path}:{label}")
    return {
        "path": str(path),
        "exists": True,
        "ready": not blockers,
        "matched": matched,
        "blockers": blockers,
    }


def _phase(name: str, checks: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    blockers: List[str] = []
    for check_name, check in checks.items():
        for blocker in check.get("blockers", []) or []:
            blockers.append(f"{check_name}:{blocker}")
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "checks": dict(checks),
        "blockers": blockers,
        "proof_boundary": (
            f"{name} audits source and artifact contracts only; it does not prove a live "
            "capture, WebApp request, simulator run, GPU provisioning, policy execution, "
            "safety, or generated-world rank fidelity."
        ),
    }


def _repo_exists_check(path: Path) -> Dict[str, Any]:
    blockers = [] if path.is_dir() else [f"missing_repo:{path}"]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "ready": not blockers,
        "blockers": blockers,
    }


def _capture_to_pipeline_phase(capture_repo: Path) -> Dict[str, Any]:
    checks: Dict[str, Mapping[str, Any]] = {
        "repo": _repo_exists_check(capture_repo),
        "ios_upload_contract": _file_contains_check(
            capture_repo,
            "BlueprintCapture/Services/CaptureUploadService.swift",
            required=[
                ("completion_marker_filename", 'completionMarkerFilename = "capture_upload_complete.json"'),
                ("robot_eval_dataset_requested", '"robot_eval_dataset"'),
                ("task_evaluation_run_requested", '"task_evaluation_run"'),
                ("requested_outputs_written", '"requested_outputs"'),
                ("site_submission_id_written", '"site_submission_id"'),
                ("buyer_request_id_written", '"buyer_request_id"'),
                ("capture_job_id_written", '"capture_job_id"'),
            ],
        ),
        "raw_contract_validator": _file_contains_check(
            capture_repo,
            "BlueprintCapture/Services/CaptureRawContractV3Validator.swift",
            required=[
                ("completion_marker_required", '"capture_upload_complete.json"'),
                ("manifest_required", '"manifest.json"'),
            ],
        ),
        "bridge_trigger_contract": _file_contains_check(
            capture_repo,
            "cloud/extract-frames/src/index.ts",
            required=[
                ("raw_upload_complete_event", "capture.raw_upload_complete.v1"),
                ("descriptor_output", "capture_descriptor.json"),
                ("handoff_output", "pipeline_handoff.json"),
                ("robot_eval_dataset_lane", "robot_eval_dataset"),
                ("task_evaluation_run_lane", "task_evaluation_run"),
                ("site_submission_id_forwarded", "site_submission_id"),
                ("buyer_request_id_forwarded", "buyer_request_id"),
                ("capture_job_id_forwarded", "capture_job_id"),
            ],
        ),
        "bridge_regressions": _file_contains_check(
            capture_repo,
            "cloud/extract-frames/src/index.test.ts",
            required=[
                ("placeholder_site_id_rejected", "invalid_site_submission_id_placeholder"),
                ("capture_id_as_job_id_rejected", "invalid_capture_job_id_matches_capture_id"),
                ("handoff_uri_asserted", "pipeline_handoff_uri"),
                ("robot_eval_requested_asserted", "robot_eval_dataset_requested"),
            ],
        ),
    }
    return _phase("capture_to_pipeline", checks)


def _webapp_request_phase(webapp_repo: Path) -> Dict[str, Any]:
    checks: Dict[str, Mapping[str, Any]] = {
        "repo": _repo_exists_check(webapp_repo),
        "request_builder": _file_contains_check(
            webapp_repo,
            "server/utils/robotEvalJobRequests.ts",
            required=[
                ("job_request_schema", "robot_eval_job_request.v1"),
                ("queue_contract", "robot_eval_job_request_inbox.v1"),
                ("forward_capture_root_env", "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"),
                (
                    "forward_capture_root_by_site_env",
                    "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
                ),
                ("capture_root_override_provenance", "capture_root_override_source"),
                ("claim_boundary_false", "public_claim_upgrade_allowed: false"),
                ("pre_gpu_ready_field", "ready_for_owner_gpu_preflight"),
            ],
        ),
        "request_route": _file_contains_check(
            webapp_repo,
            "server/routes/robot-eval-job-requests.ts",
            required=[
                ("route_validates_request", "Invalid robot_eval_job_request.v1"),
                ("local_inbox_fallback", "ROBOT_EVAL_JOB_REQUEST_INBOX_DIR"),
                ("optional_pipeline_forward", "forwardRobotEvalJobRequestToPipeline"),
            ],
        ),
        "client_request_builder": _file_contains_check(
            webapp_repo,
            "client/src/lib/robotEvalJobRequest.ts",
            required=[
                ("job_request_schema", "robot_eval_job_request.v1"),
                ("cpu_pre_gpu_context", "cpu_pre_gpu_preflight"),
                ("ready_for_owner_gpu_preflight", "ready_for_owner_gpu_preflight"),
            ],
        ),
        "state_machine_boundary": _file_contains_check(
            webapp_repo,
            "server/utils/pipelineStateMachine.ts",
            required=[
                ("job_request_uri_tracked", "robot_eval_job_request_uri"),
                ("pre_gpu_ready_preserved", "ready_for_owner_gpu_preflight"),
            ],
        ),
        "webapp_regressions": _file_contains_check(
            webapp_repo,
            "server/tests/robot-eval-job-requests.test.ts",
            required=[
                ("creates_robot_eval_request", "creates a durable Pipeline robot_eval_job_request.v1"),
                ("forward_override_tested", "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"),
                ("default_pre_gpu_false", "ready_for_owner_gpu_preflight: false"),
                (
                    "webapp_synced_artifact_blocks_without_override",
                    "missing_pipeline_capture_root_override_for_webapp_synced_artifact",
                ),
            ],
        ),
        "webapp_local_rehearsal_exporter": _file_contains_check(
            webapp_repo,
            "scripts/pipeline/export-first-gpu-webapp-rehearsal-request.ts",
            required=[
                ("uses_webapp_request_builder", "buildRobotEvalJobRequest"),
                ("validates_webapp_request", "validateRobotEvalJobRequest"),
                ("local_rehearsal_marker", "local_first_gpu_rehearsal_request"),
                ("live_forwarding_blocked", "live_webapp_forwarding_proven: false"),
            ],
        ),
        "webapp_forwarding_preflight": _file_contains_check(
            webapp_repo,
            "scripts/pipeline/audit-robot-eval-forwarding-readiness.ts",
            required=[
                (
                    "preflight_schema",
                    "blueprint.webapp.robot_eval_forwarding_readiness.v1",
                ),
                (
                    "token_env_checked",
                    "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN",
                ),
                ("read_only_probe_flag", "probe-intake-audit"),
                ("token_redacted", "redacted: true"),
                ("no_job_queued_boundary", "no_job_queued"),
                ("no_gpu_boundary", "no_gpu_allocated"),
                ("no_simulator_boundary", "no_simulator_execution_proven"),
            ],
        ),
    }
    return _phase("webapp_to_pipeline", checks)


def _pipeline_return_phase(pipeline_repo: Path) -> Dict[str, Any]:
    checks: Dict[str, Mapping[str, Any]] = {
        "repo": _repo_exists_check(pipeline_repo),
        "entrypoints": _file_contains_check(
            pipeline_repo,
            "pyproject.toml",
            required=[
                ("first_gpu_readiness_cli", "blueprint-audit-first-gpu-e2e-readiness"),
                ("first_gpu_stage_cli", "blueprint-stage-first-gpu-sample-video"),
                ("owner_gpu_proof_cli", "blueprint-run-owner-gpu-proof"),
                (
                    "owner_default_smoke_artifacts_cli",
                    "blueprint-write-owner-gpu-default-smoke-artifacts",
                ),
                ("first_gpu_run_packet_cli", "blueprint-build-first-gpu-run-packet"),
            ],
        ),
        "webapp_intake": _file_contains_check(
            pipeline_repo,
            "src/blueprint_pipeline/live_pipeline_input_intake.py",
            required=[
                ("staged_inputs_schema", "LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION"),
                ("stage_webapp_request", "stage_webapp_request"),
                ("webapp_request_audit", "_audit_webapp_request"),
                ("required_upstream_fields", "WEBAPP_UPSTREAM_REQUIRED_FIELDS"),
            ],
        ),
        "first_gpu_readiness": _file_contains_check(
            pipeline_repo,
            "src/blueprint_pipeline/first_gpu_e2e_readiness.py",
            required=[
                ("requires_staged_request", "missing_webapp_staged_inputs"),
                ("blocks_local_rehearsal_by_default", "webapp_staged_inputs_local_rehearsal_only"),
                (
                    "webapp_forwarding_preflight_env",
                    "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT",
                ),
                ("owner_gpu_blocker", "owner_gpu_simulator_execution_not_run"),
                ("gpu_gate", "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"),
            ],
        ),
        "first_gpu_run_packet": _file_contains_check(
            pipeline_repo,
            "src/blueprint_pipeline/first_gpu_run_packet.py",
            required=[
                ("gpu_provider_bootstrap_file", "gpu_provider_bootstrap.md"),
                ("gpu_provider_bootstrap_manifest", "gpu_provider_bootstrap.json"),
                (
                    "webapp_handoff_verification_script",
                    "webapp_handoff_verification_commands",
                ),
                ("nvidia_nim_boundary", "nvidia_nim_boundary"),
                ("isaac_sim_gpu_avoid_list", "avoid_for_isaac_sim"),
                (
                    "owner_default_smoke_helper",
                    "blueprint-write-owner-gpu-default-smoke-artifacts",
                ),
                ("owner_command_binding_template", "owner_default_smoke_command_binding.sh"),
                ("live_policy_execution_contract", "live_policy_execution_contract.md"),
                (
                    "default_test_job_request_template",
                    "default_test_robot_eval_job_request.template.json",
                ),
                ("real_robot_pov_template", "real_robot_pov_manifest.template.json"),
                ("live_input_staging_script", "stage_first_gpu_live_inputs.sh"),
            ],
        ),
        "owner_gpu_proof_runner": _file_contains_check(
            pipeline_repo,
            "src/blueprint_pipeline/owner_gpu_proof_runner.py",
            required=[
                ("scene_load_trace_env", "BLUEPRINT_SCENE_LOAD_TRACE"),
                ("spawn_trace_env", "BLUEPRINT_SPAWN_TRACE"),
                ("action_trace_env", "BLUEPRINT_ACTION_OR_POLICY_TRACE"),
                ("default_policy_env", "BLUEPRINT_DEFAULT_SMOKE_POLICY"),
                ("policy_execution_trace_env", "BLUEPRINT_POLICY_EXECUTION_TRACE"),
                ("sim_robot_pov_env", "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"),
                ("proof_manifest", "owner_gpu_simulator_execution_proof_manifest.json"),
            ],
        ),
        "first_gpu_runbook": _file_contains_check(
            pipeline_repo,
            "docs/FIRST_GPU_E2E_RUNBOOK.md",
            required=[
                ("runpod_provider", "--provisioner runpod"),
                ("local_rehearsal_boundary", "--allow-local-webapp-rehearsal"),
                ("owner_gpu_proof_wrapper", "blueprint-run-owner-gpu-proof"),
                ("nim_not_primary", "Phase 3: GPU VM Bring-Up"),
            ],
        ),
    }
    return _phase("pipeline_return", checks)


def _runtime_capture_phase(
    *,
    capture_root: str | Path | None,
    webapp_site_slug: str,
    webapp_staged_inputs_path: str | Path | None,
    webapp_forwarding_preflight_path: str | Path | None,
    simulator: str,
    provisioner: str,
    simulator_command: str | None,
    simulator_command_location: str,
    require_webapp_forwarding: bool,
    require_webapp_staged_request: bool,
    allow_local_webapp_rehearsal: bool,
    require_gpu_gates: bool,
) -> Dict[str, Any]:
    if not capture_root:
        return {
            "status": "blocked",
            "ready": False,
            "required": True,
            "capture_root": None,
            "readiness": None,
            "blockers": ["missing_capture_root_for_runtime_first_gpu_readiness"],
            "proof_boundary": (
                "Repo contracts are not runtime proof; a concrete capture root must pass "
                "first-GPU readiness before owner GPU time is useful."
            ),
        }
    try:
        readiness = build_first_gpu_e2e_readiness(
            capture_root=capture_root,
            webapp_site_slug=webapp_site_slug,
            webapp_staged_inputs_path=webapp_staged_inputs_path,
            webapp_forwarding_preflight_path=webapp_forwarding_preflight_path,
            simulator=simulator,
            provisioner=provisioner,
            simulator_command=simulator_command,
            simulator_command_location=simulator_command_location,
            require_webapp_forwarding=require_webapp_forwarding,
            require_webapp_staged_request=require_webapp_staged_request,
            allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
            require_gpu_gates=require_gpu_gates,
        )
    except Exception as exc:  # pragma: no cover - defensive, surfaced in manifest
        return {
            "status": "blocked",
            "ready": False,
            "required": True,
            "capture_root": str(Path(capture_root).expanduser()),
            "readiness": None,
            "blockers": [f"first_gpu_readiness_exception:{exc.__class__.__name__}"],
            "error": str(exc),
        }
    blockers = [str(item) for item in readiness.get("blockers") or []]
    return {
        "status": readiness.get("status"),
        "ready": bool(readiness.get("ready_for_first_gpu_attempt")),
        "required": True,
        "capture_root": readiness.get("capture_root"),
        "readiness": readiness,
        "blockers": blockers,
        "warnings": readiness.get("warnings") or [],
        "proof_boundary": (
            "This wraps the authoritative first-GPU readiness audit. Passing it means ready "
            "for an owner GPU attempt, not that owner GPU proof or generated-world rank fidelity exists."
        ),
    }


def _runtime_has_local_webapp_rehearsal(runtime_phase: Mapping[str, Any]) -> bool:
    readiness = runtime_phase.get("readiness")
    if not isinstance(readiness, Mapping):
        return False
    stages = readiness.get("stages")
    if not isinstance(stages, Mapping):
        return False
    staged = stages.get("webapp_staged_request")
    return isinstance(staged, Mapping) and bool(staged.get("local_rehearsal_only"))


def _run_packet_phase(*, capture_root: str | Path | None) -> Dict[str, Any]:
    if not capture_root:
        return {
            "status": "not_checked",
            "ready": False,
            "required": False,
            "capture_root": None,
            "packet_dir": None,
            "checks": {},
            "blockers": [],
            "proof_boundary": (
                "Run-packet launch consistency is checked only after a concrete capture root "
                "is supplied; this phase does not generate packets, copy files, provision "
                "GPUs, run simulators, or prove generated-world rank fidelity."
            ),
        }

    root = Path(capture_root).expanduser()
    packet_dir = root / "pipeline" / "first_gpu_e2e_run_packet"
    required_files = {
        "first_gpu_run_packet": packet_dir / "first_gpu_run_packet.json",
        "blocker_resolution": packet_dir / "first_gpu_blocker_resolution.json",
        "webapp_handoff": packet_dir / "first_gpu_webapp_handoff.json",
        "scene_asset_acquisition": packet_dir / "first_gpu_scene_asset_acquisition.json",
        "launch_order": packet_dir / "first_gpu_launch_order.json",
        "gpu_vm_runtime_preflight_plan": packet_dir / "gpu_vm_runtime_preflight_plan.json",
        "gpu_vm_sync_manifest": packet_dir / "gpu_vm_sync_manifest.json",
    }
    missing_blockers = {
        "first_gpu_run_packet": "missing_first_gpu_run_packet",
        "blocker_resolution": "missing_first_gpu_blocker_resolution",
        "webapp_handoff": "missing_first_gpu_webapp_handoff",
        "scene_asset_acquisition": "missing_first_gpu_scene_asset_acquisition",
        "launch_order": "missing_first_gpu_launch_order",
        "gpu_vm_runtime_preflight_plan": "missing_gpu_vm_runtime_preflight_plan",
        "gpu_vm_sync_manifest": "missing_gpu_vm_sync_manifest",
    }
    checks: Dict[str, Any] = {}
    blockers: List[str] = []
    payloads: Dict[str, Dict[str, Any]] = {}
    for name, path in required_files.items():
        if not path.is_file():
            blocker = missing_blockers[name]
            checks[name] = {
                "path": str(path),
                "exists": False,
                "ready": False,
                "blockers": [blocker],
            }
            blockers.append(blocker)
            continue
        payload, error = _read_json_mapping(path)
        check_blockers = [error] if error else []
        if payload is not None:
            payloads[name] = payload
        checks[name] = {
            "path": str(path),
            "exists": True,
            "ready": not check_blockers,
            "blockers": check_blockers,
        }
        _append_unique(blockers, check_blockers)

    packet = payloads.get("first_gpu_run_packet", {})
    packet_blockers = [str(item) for item in packet.get("blockers") or []]
    if packet and not bool(packet.get("ready_for_first_gpu_attempt")):
        blockers.append("first_gpu_run_packet_not_ready_for_attempt")
        checks["first_gpu_run_packet"]["blockers"].append(
            "first_gpu_run_packet_not_ready_for_attempt"
        )
    for packet_blocker in packet_blockers:
        blocker = f"first_gpu_run_packet_blocker:{packet_blocker}"
        blockers.append(blocker)
        checks["first_gpu_run_packet"]["blockers"].append(blocker)
    if packet:
        generated_files = (
            packet.get("generated_files")
            if isinstance(packet.get("generated_files"), Mapping)
            else {}
        )
        checks["first_gpu_run_packet"].update(
            {
                "readiness_status": packet.get("readiness_status"),
                "ready_for_first_gpu_attempt": bool(packet.get("ready_for_first_gpu_attempt")),
                "webapp_upstream_truth_verification_script_path": generated_files.get(
                    "webapp_upstream_truth_verification_commands"
                ),
            }
        )
        owner_binding_template_path = _string(
            generated_files.get("owner_command_binding_template")
        )
        if not owner_binding_template_path:
            owner_binding_template_path = str(
                packet_dir / "owner_default_smoke_command_binding.sh"
            )
        owner_binding_template_exists = bool(owner_binding_template_path) and Path(
            owner_binding_template_path
        ).is_file()
        checks["first_gpu_run_packet"].update(
            {
                "owner_command_binding_template_path": owner_binding_template_path,
                "owner_command_binding_template_exists": owner_binding_template_exists,
            }
        )
        if not owner_binding_template_exists:
            blocker = "owner_command_binding_template_missing"
            blockers.append(blocker)
            checks["first_gpu_run_packet"]["blockers"].append(blocker)
        live_policy_contract_path = _string(
            generated_files.get("live_policy_execution_contract")
        )
        if not live_policy_contract_path:
            live_policy_contract_path = str(packet_dir / "live_policy_execution_contract.md")
        live_policy_contract_exists = bool(live_policy_contract_path) and Path(
            live_policy_contract_path
        ).is_file()
        checks["first_gpu_run_packet"].update(
            {
                "live_policy_execution_contract_path": live_policy_contract_path,
                "live_policy_execution_contract_exists": live_policy_contract_exists,
            }
        )
        if not live_policy_contract_exists:
            blocker = "live_policy_execution_contract_missing"
            blockers.append(blocker)
            checks["first_gpu_run_packet"]["blockers"].append(blocker)
        generated_required_files = {
            "default_test_robot_eval_job_request_template": (
                "default_test_robot_eval_job_request.template.json",
                "default_test_robot_eval_job_request_template_missing",
            ),
            "real_robot_pov_manifest_template": (
                "real_robot_pov_manifest.template.json",
                "real_robot_pov_manifest_template_missing",
            ),
            "live_input_staging_commands": (
                "stage_first_gpu_live_inputs.sh",
                "live_input_staging_commands_missing",
            ),
        }
        for key, (fallback_name, missing_blocker) in generated_required_files.items():
            generated_path = _string(generated_files.get(key))
            if not generated_path:
                generated_path = str(packet_dir / fallback_name)
            generated_exists = bool(generated_path) and Path(generated_path).is_file()
            checks["first_gpu_run_packet"].update(
                {
                    f"{key}_path": generated_path,
                    f"{key}_exists": generated_exists,
                }
            )
            if not generated_exists:
                blockers.append(missing_blocker)
                checks["first_gpu_run_packet"]["blockers"].append(missing_blocker)

    blocker_resolution = payloads.get("blocker_resolution", {})
    operator_actions: list[Dict[str, Any]] = []
    if blocker_resolution:
        action_count = int(blocker_resolution.get("action_count") or 0)
        blocked_action_count = int(blocker_resolution.get("blocked_action_count") or 0)
        raw_actions = blocker_resolution.get("actions") or []
        if isinstance(raw_actions, Sequence) and not isinstance(raw_actions, (str, bytes)):
            operator_actions = [
                dict(item)
                for item in raw_actions
                if isinstance(item, Mapping)
            ]
        if len(operator_actions) != action_count:
            blockers.append("blocker_resolution_action_count_mismatch")
            checks["blocker_resolution"]["blockers"].append(
                "blocker_resolution_action_count_mismatch"
            )
        if packet and not bool(packet.get("ready_for_first_gpu_attempt")) and action_count == 0:
            blockers.append("blocker_resolution_missing_actions_for_blocked_packet")
            checks["blocker_resolution"]["blockers"].append(
                "blocker_resolution_missing_actions_for_blocked_packet"
            )
        checks["blocker_resolution"].update(
            {
                "readiness_status": blocker_resolution.get("readiness_status"),
                "ready_for_first_gpu_attempt": bool(
                    blocker_resolution.get("ready_for_first_gpu_attempt")
                ),
                "action_count": action_count,
                "blocked_action_count": blocked_action_count,
                "action_category_ids": [
                    str(item.get("category_id"))
                    for item in operator_actions
                    if item.get("category_id")
                ],
            }
        )

    webapp_handoff = payloads.get("webapp_handoff", {})
    if webapp_handoff:
        status = _string(webapp_handoff.get("status"))
        handoff_blockers = [str(item) for item in webapp_handoff.get("blockers") or []]
        verification = (
            webapp_handoff.get("verification")
            if isinstance(webapp_handoff.get("verification"), Mapping)
            else {}
        )
        verification_script = (
            verification.get("script")
            if isinstance(verification.get("script"), Mapping)
            else {}
        )
        if status != "ready_for_webapp_handoff_verification":
            blockers.append("webapp_handoff_not_ready")
            checks["webapp_handoff"]["blockers"].append("webapp_handoff_not_ready")
        for handoff_blocker in handoff_blockers:
            blocker = f"webapp_handoff_blocker:{handoff_blocker}"
            blockers.append(blocker)
            checks["webapp_handoff"]["blockers"].append(blocker)
        checks["webapp_handoff"].update(
            {
                "status": status or None,
                "blocker_count": len(handoff_blockers),
                "upstream_id_requirement_count": len(
                    [
                        item
                        for item in webapp_handoff.get("upstream_id_requirements") or []
                        if isinstance(item, Mapping)
                    ]
                ),
                "verification_script_path": verification_script.get("path"),
                "verification_script_safe_to_run_now": bool(
                    verification_script.get("safe_to_run_now")
                ),
                "verification_runs_live_webapp_call": bool(
                    verification_script.get("runs_live_webapp_call")
                ),
                "verification_missing_env": [
                    str(item) for item in verification.get("missing_env") or []
                ],
            }
        )

    scene_asset_acquisition = payloads.get("scene_asset_acquisition", {})
    if scene_asset_acquisition:
        status = _string(scene_asset_acquisition.get("status"))
        provider_submission = (
            scene_asset_acquisition.get("provider_submission")
            if isinstance(scene_asset_acquisition.get("provider_submission"), Mapping)
            else {}
        )
        provider_submission_script = (
            provider_submission.get("script")
            if isinstance(provider_submission.get("script"), Mapping)
            else {}
        )
        scene_asset_blockers = [
            str(item) for item in scene_asset_acquisition.get("blockers") or []
        ]
        if status != "ready_for_scene_preflight_rerun":
            blockers.append("scene_asset_acquisition_not_ready")
            checks["scene_asset_acquisition"]["blockers"].append(
                "scene_asset_acquisition_not_ready"
            )
        for scene_asset_blocker in scene_asset_blockers:
            blocker = f"scene_asset_acquisition_blocker:{scene_asset_blocker}"
            blockers.append(blocker)
            checks["scene_asset_acquisition"]["blockers"].append(blocker)
        checks["scene_asset_acquisition"].update(
            {
                "status": status or None,
                "blocker_count": len(scene_asset_blockers),
                "source_video_preflight_status": (
                    (scene_asset_acquisition.get("source_video_preflight") or {}).get(
                        "status"
                    )
                    if isinstance(scene_asset_acquisition.get("source_video_preflight"), Mapping)
                    else None
                ),
                "worldlabs_request_manifest_exists": (
                    (scene_asset_acquisition.get("provider_preview") or {}).get(
                        "worldlabs_request_manifest_exists"
                    )
                    if isinstance(scene_asset_acquisition.get("provider_preview"), Mapping)
                    else None
                ),
                "worldlabs_world_manifest_exists": (
                    (scene_asset_acquisition.get("provider_preview") or {}).get(
                        "worldlabs_world_manifest_exists"
                    )
                    if isinstance(scene_asset_acquisition.get("provider_preview"), Mapping)
                    else None
                ),
                "materialization_manifest_exists": (
                    (scene_asset_acquisition.get("materialization") or {}).get(
                        "materialization_manifest_exists"
                    )
                    if isinstance(scene_asset_acquisition.get("materialization"), Mapping)
                    else None
                ),
                "provider_submission_status": provider_submission.get("status"),
                "provider_submission_input_status": provider_submission.get("input_status"),
                "ready_for_worldlabs_request_inputs": bool(
                    provider_submission.get("ready_for_worldlabs_request_inputs")
                ),
                "ready_to_submit_worldlabs_request": bool(
                    provider_submission.get("ready_to_submit_worldlabs_request")
                ),
                "safe_to_submit_before_gpu_spend": bool(
                    provider_submission.get("safe_to_submit_before_gpu_spend")
                ),
                "provider_submission_requires_env": [
                    str(item) for item in provider_submission.get("requires_env") or []
                ],
                "provider_submission_missing_env": [
                    str(item) for item in provider_submission.get("missing_env") or []
                ],
                "provider_submission_required_env_status": (
                    dict(provider_submission.get("required_env_status"))
                    if isinstance(provider_submission.get("required_env_status"), Mapping)
                    else {}
                ),
                "provider_submission_requires_gpu": bool(
                    provider_submission.get("requires_gpu")
                ),
                "provider_submission_script_path": provider_submission_script.get("path"),
                "provider_submission_script_safe_to_run_now": bool(
                    provider_submission_script.get("safe_to_run_now")
                ),
                "provider_submission_script_requires_allow_env": provider_submission_script.get(
                    "requires_explicit_allow_env"
                ),
            }
        )

    launch_order = payloads.get("launch_order", {})
    if launch_order:
        blocked_steps = [str(item) for item in launch_order.get("blocked_step_ids") or []]
        gpu_execution_allowed = bool(launch_order.get("gpu_execution_allowed"))
        if not gpu_execution_allowed:
            blockers.append("launch_order_blocks_gpu_execution")
            checks["launch_order"]["blockers"].append("launch_order_blocks_gpu_execution")
        for step_id in blocked_steps:
            blocker = f"launch_order_blocked_step:{step_id}"
            blockers.append(blocker)
            checks["launch_order"]["blockers"].append(blocker)
        checks["launch_order"].update(
            {
                "status": launch_order.get("status"),
                "gpu_execution_allowed": gpu_execution_allowed,
                "blocked_step_ids": blocked_steps,
                "next_action_step_ids": launch_order.get("next_action_step_ids") or [],
            }
        )

    runtime_preflight = payloads.get("gpu_vm_runtime_preflight_plan", {})
    if runtime_preflight:
        runtime_script = (
            runtime_preflight.get("script")
            if isinstance(runtime_preflight.get("script"), Mapping)
            else {}
        )
        runtime_related_artifacts = (
            runtime_preflight.get("related_artifacts")
            if isinstance(runtime_preflight.get("related_artifacts"), Mapping)
            else {}
        )
        safe_to_run = bool(
            runtime_script.get("safe_to_run_on_gpu_vm")
        )
        hard_stops = [
            str(item) for item in runtime_preflight.get("hard_stop_blockers") or []
        ]
        plan_result = (
            runtime_preflight.get("result")
            if isinstance(runtime_preflight.get("result"), Mapping)
            else {}
        )
        result_summary = dict(plan_result) if plan_result else _runtime_preflight_result_summary(
            runtime_script.get("default_result_path")
        )
        if not safe_to_run:
            blockers.append("gpu_vm_runtime_preflight_plan_blocks_vm_preflight")
            checks["gpu_vm_runtime_preflight_plan"]["blockers"].append(
                "gpu_vm_runtime_preflight_plan_blocks_vm_preflight"
            )
        for hard_stop in hard_stops:
            blocker = f"gpu_vm_runtime_preflight_hard_stop:{hard_stop}"
            blockers.append(blocker)
            checks["gpu_vm_runtime_preflight_plan"]["blockers"].append(blocker)
        if not bool(result_summary.get("ready_for_owner_command_attempt")):
            blockers.append("gpu_vm_runtime_preflight_result_not_ready")
            checks["gpu_vm_runtime_preflight_plan"]["blockers"].append(
                "gpu_vm_runtime_preflight_result_not_ready"
            )
            for result_blocker in result_summary.get("blockers") or []:
                blocker = f"gpu_vm_runtime_preflight_result:{result_blocker}"
                blockers.append(blocker)
                checks["gpu_vm_runtime_preflight_plan"]["blockers"].append(blocker)
        checks["gpu_vm_runtime_preflight_plan"].update(
            {
                "status": runtime_preflight.get("status"),
                "script_path": runtime_script.get("path"),
                "script_default_result_path": runtime_script.get("default_result_path"),
                "result": result_summary,
                "result_ready_for_owner_command_attempt": bool(
                    result_summary.get("ready_for_owner_command_attempt")
                ),
                "safe_to_run_on_gpu_vm": safe_to_run,
                "runs_owner_simulator_command": bool(
                    runtime_script.get("runs_owner_simulator_command")
                ),
                "hard_stop_blockers": hard_stops,
                "gpu_vm_sync_status": runtime_preflight.get("gpu_vm_sync_status"),
                "gpu_vm_commands_path": runtime_related_artifacts.get(
                    "gpu_vm_commands"
                ),
            }
        )

    sync_manifest = payloads.get("gpu_vm_sync_manifest", {})
    if sync_manifest:
        sync_status = _string(sync_manifest.get("status"))
        sync_blockers = [str(item) for item in sync_manifest.get("blockers") or []]
        if sync_status != "ready":
            blockers.append("gpu_vm_sync_manifest_not_ready")
            checks["gpu_vm_sync_manifest"]["blockers"].append(
                "gpu_vm_sync_manifest_not_ready"
            )
        for sync_blocker in sync_blockers:
            blocker = f"gpu_vm_sync_manifest_blocker:{sync_blocker}"
            blockers.append(blocker)
            checks["gpu_vm_sync_manifest"]["blockers"].append(blocker)
        checks["gpu_vm_sync_manifest"].update(
            {
                "status": sync_status or None,
                "missing_required_file_count": sync_manifest.get(
                    "missing_required_file_count"
                ),
                "blocker_count": len(sync_blockers),
            }
        )

    blockers = _string_list_unique(blockers)
    for check in checks.values():
        check["blockers"] = _string_list_unique(check.get("blockers") or [])
        check["ready"] = not check["blockers"]
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "required": True,
        "capture_root": str(root),
        "packet_dir": str(packet_dir),
        "checks": checks,
        "operator_actions": operator_actions,
        "operator_action_count": len(operator_actions),
        "blocked_operator_action_count": sum(
            1 for item in operator_actions if item.get("must_clear_before_gpu_spend")
        ),
        "blockers": blockers,
        "proof_boundary": (
            "This phase reads the generated first-GPU packet launch order, VM sync "
            "manifest, VM runtime preflight plan, and blocker-resolution actions. "
            "Passing it means the operator packet no longer forbids the owner GPU "
            "attempt; it does not copy files, provision GPUs, run simulators, or "
            "prove generated-world rank fidelity."
        ),
    }


def _remediation_for_blocker(blocker: str) -> Dict[str, Any]:
    action = {
        "blocker": blocker,
        "category": "repo_contract_or_unknown",
        "next_action": "Inspect the named source contract or readiness stage and repair the missing proof boundary.",
        "evidence_required": "A rerun of blueprint-audit-first-gpu-cross-repo-readiness no longer reports this blocker.",
        "safe_command": None,
        "can_be_rehearsed_locally": True,
        "proof_boundary": (
            "Clearing this blocker only satisfies the named audit gate; it does not prove "
            "owner GPU simulator execution or generated-world rank fidelity."
        ),
    }
    if "missing_capture_root_for_runtime_first_gpu_readiness" in blocker:
        action.update(
            {
                "category": "sample_capture",
                "next_action": (
                    "Stage the selected collected video as a capture root, or pass the real "
                    "Capture-produced capture root to this audit."
                ),
                "evidence_required": (
                    "A concrete scenes/<scene_id>/captures/<capture_id> root with raw/"
                    "capture_upload_complete.json and raw walkthrough video."
                ),
                "safe_command": (
                    "blueprint-stage-first-gpu-sample-video --source-video <video> "
                    "--storage-root output/first-gpu-sample-storage --scene-id <scene> "
                    "--capture-id <capture> --run-simulation-automation"
                ),
            }
        )
    elif "missing_or_placeholder_webapp_site_submission_id" in blocker:
        action.update(
            {
                "category": "webapp_upstream_truth",
                "next_action": "Supply the real WebApp site_submission_id from the request/site submission record.",
                "evidence_required": "capture_descriptor.json or raw/manifest.json contains a non-placeholder site_submission_id.",
                "safe_command": None,
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_or_placeholder_webapp_request_id" in blocker:
        action.update(
            {
                "category": "webapp_upstream_truth",
                "next_action": "Supply the real request_id from WebApp/request owner-system truth.",
                "evidence_required": "capture_descriptor.json, raw/manifest.json, or pipeline_handoff.json owner_system.request_id contains a non-placeholder request_id.",
                "safe_command": None,
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_or_placeholder_webapp_buyer_request_id" in blocker:
        action.update(
            {
                "category": "webapp_upstream_truth",
                "next_action": "Supply the real buyer_request_id from the robot-team/WebApp request.",
                "evidence_required": "capture_descriptor.json or raw/manifest.json contains a non-placeholder buyer_request_id.",
                "safe_command": None,
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_or_placeholder_webapp_capture_job_id" in blocker:
        action.update(
            {
                "category": "webapp_upstream_truth",
                "next_action": "Supply the real capture_job_id from the Capture/WebApp job assignment.",
                "evidence_required": "capture_descriptor.json or raw/manifest.json contains a non-placeholder capture_job_id.",
                "safe_command": None,
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL" in blocker:
        action.update(
            {
                "category": "webapp_forwarding_env",
                "next_action": "Set the WebApp-to-Pipeline intake URL for robot-eval job forwarding.",
                "evidence_required": "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL is set to the authenticated Pipeline intake endpoint.",
                "safe_command": "export ROBOT_EVAL_JOB_REQUEST_FORWARD_URL=https://<pipeline-host>/api/live-pipeline/job-requests",
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN" in blocker:
        action.update(
            {
                "category": "webapp_forwarding_env",
                "next_action": "Set the WebApp-to-Pipeline forwarding token in the shell or deployment environment.",
                "evidence_required": "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN is configured without writing the secret into artifacts.",
                "safe_command": "export ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN=<redacted>",
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_pipeline_capture_root_override_for_webapp_synced_artifact" in blocker:
        action.update(
            {
                "category": "webapp_forwarding_env",
                "next_action": (
                    "Map the public WebApp site slug capture root to the exact Pipeline "
                    "control-plane capture root."
                ),
                "evidence_required": "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON maps the WebApp site slug to the local capture root.",
                "safe_command": "export ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON='{\"<site-slug>\":\"<capture-root>\"}'",
                "can_be_rehearsed_locally": False,
            }
        )
    elif "webapp_forwarding_preflight" in blocker:
        action.update(
            {
                "category": "webapp_forwarding_env",
                "next_action": (
                    "Regenerate the WebApp forwarding preflight report with the intended "
                    "URL, token, site slug, and capture-root mapping."
                ),
                "evidence_required": (
                    "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT points at a ready, "
                    "redacted WebApp forwarding preflight report covering the selected site slug."
                ),
                "safe_command": (
                    "npm run pipeline:forwarding:preflight -- --require-forwarding "
                    "--probe-intake-audit --output <capture-root>/pipeline/webapp_forwarding_preflight.json"
                ),
                "can_be_rehearsed_locally": True,
            }
        )
    elif "missing_webapp_staged_inputs" in blocker:
        action.update(
            {
                "category": "webapp_staged_request",
                "next_action": "Stage a validated WebApp robot_eval_job_request.v1 through Pipeline intake.",
                "evidence_required": "pipeline/live_pipeline_staged_inputs.json points at a staged robot_eval_job_request.v1 or queue envelope matching this capture root.",
                "safe_command": (
                    "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
                    "--webapp-job-request <robot_eval_job_request.json> --stage-webapp-request "
                    "--staged-inputs-path <capture-root>/pipeline/live_pipeline_staged_inputs.json"
                ),
                "can_be_rehearsed_locally": True,
            }
        )
    elif "webapp_staged_inputs_local_rehearsal_only" in blocker:
        action.update(
            {
                "category": "webapp_staged_request",
                "next_action": (
                    "Use --allow-local-webapp-rehearsal only for local request-shape rehearsal, "
                    "or replace the local rehearsal request with a real WebApp-forwarded request."
                ),
                "evidence_required": "For live proof, staged inputs are not marked local_first_gpu_rehearsal_request.",
                "safe_command": "blueprint-audit-first-gpu-cross-repo-readiness ... --allow-local-webapp-rehearsal",
                "can_be_rehearsed_locally": True,
            }
        )
    elif "local_webapp_rehearsal_not_live_forwarding_proof" in blocker:
        action.update(
            {
                "category": "webapp_live_forwarding_proof",
                "next_action": (
                    "Replace the local rehearsal request with a real WebApp-forwarded "
                    "robot_eval_job_request.v1 before treating this as the full E2E path."
                ),
                "evidence_required": (
                    "Staged inputs are not marked local_first_gpu_rehearsal_request, and "
                    "the WebApp forwarding result is tied to real upstream IDs."
                ),
                "safe_command": (
                    "Submit the request through the WebApp route, then rerun "
                    "blueprint-audit-first-gpu-cross-repo-readiness without "
                    "--allow-local-webapp-rehearsal."
                ),
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_first_gpu_webapp_handoff" in blocker or "webapp_handoff_not_ready" in blocker:
        action.update(
            {
                "category": "webapp_handoff_packet",
                "next_action": (
                    "Regenerate the first-GPU run packet after WebApp upstream IDs, "
                    "forwarding environment, and staged request inputs are ready."
                ),
                "evidence_required": (
                    "first_gpu_webapp_handoff.json exists with "
                    "status=ready_for_webapp_handoff_verification."
                ),
                "safe_command": (
                    "blueprint-build-first-gpu-run-packet --capture-root <capture-root> "
                    "--webapp-site-slug <site-slug> --owner-command <remote-owner-command> "
                    "--owner-command-location remote"
                ),
            }
        )
    elif "webapp_handoff_blocker:" in blocker:
        action.update(
            {
                "category": "webapp_handoff_packet",
                "next_action": (
                    "Clear the WebApp handoff blocker named in first_gpu_webapp_handoff.json."
                ),
                "evidence_required": (
                    "first_gpu_webapp_handoff.json reports no blockers and preserves the "
                    "local-rehearsal boundary."
                ),
                "safe_command": (
                    "blueprint-build-first-gpu-run-packet --capture-root <capture-root> "
                    "--webapp-site-slug <site-slug> --owner-command <remote-owner-command> "
                    "--owner-command-location remote"
                ),
            }
        )
    elif (
        "missing_first_gpu_scene_asset_acquisition" in blocker
        or "scene_asset_acquisition_not_ready" in blocker
        or "scene_asset_acquisition_blocker:worldlabs_request_manifest_missing" in blocker
        or "scene_asset_acquisition_blocker:worldlabs_world_manifest_missing" in blocker
    ):
        action.update(
            {
                "category": "scene_asset_acquisition",
                "next_action": (
                    "Complete the provider-preview scene-generation path for the sample "
                    "video before trying to spend GPU time."
                ),
                "evidence_required": (
                    "pipeline/worldlabs_request_manifest.json and "
                    "pipeline/worldlabs_world_manifest.json exist for this capture root."
                ),
                "safe_command": (
                    "BLUEPRINT_PREVIEW_PROVIDER=world_labs WORLDLABS_API_KEY=<set-in-shell-not-artifact> "
                    "blueprint-run-e2e --capture-root <capture-root> --provider local "
                    "--pipeline-lane current --run-evaluation-prep --evaluation-prep-provider manual"
                ),
                "can_be_rehearsed_locally": False,
            }
        )
    elif (
        "scene_asset_acquisition_blocker:worldlabs_asset_materialization_manifest_missing"
        in blocker
        or "scene_asset_acquisition_blocker:materialized_scene_asset_missing" in blocker
    ):
        action.update(
            {
                "category": "scene_asset_acquisition",
                "next_action": (
                    "Materialize the already-generated World Labs scene assets and rerun "
                    "simulation automation with the local scene asset."
                ),
                "evidence_required": (
                    "pipeline/worldlabs_assets/materialized_assets_manifest.json and "
                    "pipeline/worldlabs_export_manifest.json exist with at least one local asset."
                ),
                "safe_command": (
                    "blueprint-materialize-worldlabs-assets --capture-root <capture-root> "
                    "--include-visual-assets"
                ),
                "can_be_rehearsed_locally": False,
            }
        )
    elif "pipeline_gpu_handoff:missing_artifact" in blocker or "gpu_handoff_packet_not_ready" in blocker:
        action.update(
            {
                "category": "pipeline_gpu_handoff",
                "next_action": (
                    "Rerun simulation automation after scene asset, scene frame, and spawn "
                    "validation blockers are fixed."
                ),
                "evidence_required": "gpu_handoff_packet.json status is ready_for_owner_gpu_preflight_handoff.",
                "safe_command": "blueprint-run-simulation-automation --capture-root <capture-root>",
            }
        )
    elif "spawn_validation_blocked" in blocker:
        action.update(
            {
                "category": "scene_spawn_preflight",
                "next_action": "Provide materialized scene geometry with finite bounds, then rerun simulation automation.",
                "evidence_required": "spawn_pose_validation_manifest.json has finite scene bounds and at least one valid/reviewable spawn candidate.",
                "safe_command": "blueprint-stage-first-gpu-sample-video ... --scene-asset <materialized-scene> --run-simulation-automation",
            }
        )
    elif "simulator_command_executable_missing" in blocker:
        action.update(
            {
                "category": "owner_gpu_command",
                "next_action": "Install or provide the actual owner simulator command on the host where readiness is being audited.",
                "evidence_required": "The executable in --simulator-command exists and is the real wrapper/command that writes owner proof traces.",
                "safe_command": "blueprint-build-first-gpu-run-packet --capture-root <capture-root> --owner-command <real-command>",
                "can_be_rehearsed_locally": False,
            }
        )
    elif "owner_command_binding_template_missing" in blocker:
        action.update(
            {
                "category": "owner_gpu_command",
                "next_action": (
                    "Regenerate the first-GPU run packet so it includes the "
                    "fail-closed owner command binding template."
                ),
                "evidence_required": (
                    "pipeline/first_gpu_e2e_run_packet/"
                    "owner_default_smoke_command_binding.sh exists and is included "
                    "in generated_files."
                ),
                "safe_command": (
                    "blueprint-build-first-gpu-run-packet --capture-root "
                    "<capture-root> --owner-command <real-command>"
                ),
                "can_be_rehearsed_locally": True,
            }
        )
    elif "missing_simulator_command" in blocker:
        action.update(
            {
                "category": "owner_gpu_command",
                "next_action": "Pass the owner simulator command for the selected backend.",
                "evidence_required": "--simulator-command names the command that will run inside the GPU VM.",
                "safe_command": "blueprint-audit-first-gpu-cross-repo-readiness ... --simulator-command <real-command>",
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION" in blocker:
        action.update(
            {
                "category": "owner_gpu_gate",
                "next_action": "Set the explicit simulator execution gate only when ready to run the owner GPU command.",
                "evidence_required": "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true is present for the actual owner GPU attempt.",
                "safe_command": "export BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true",
                "can_be_rehearsed_locally": False,
            }
        )
    elif "missing_local_scene_asset" in blocker:
        action.update(
            {
                "category": "scene_spawn_preflight",
                "next_action": "Provide a local materialized scene asset for the selected sample capture.",
                "evidence_required": "scene_asset_preflight.json references local geometry or scene asset bounds.",
                "safe_command": "blueprint-run-simulation-automation --capture-root <capture-root> --scene-asset <materialized-scene>",
            }
        )
    elif "missing_scene_frame_estimate" in blocker or "scene_bounds_missing_or_invalid" in blocker:
        action.update(
            {
                "category": "scene_spawn_preflight",
                "next_action": "Generate finite scene frame/bounds from a materialized scene asset before renting GPU time.",
                "evidence_required": "scene_frame_estimate.json contains finite min/max bounds and floor_z_estimate.",
                "safe_command": "blueprint-run-simulation-automation --capture-root <capture-root> --scene-asset <materialized-scene>",
            }
        )
    elif (
        "missing_gpu_vm_runtime_preflight_plan" in blocker
        or "gpu_vm_runtime_preflight_plan_blocks_vm_preflight" in blocker
        or "gpu_vm_runtime_preflight_hard_stop:" in blocker
        or "gpu_vm_runtime_preflight_result_not_ready" in blocker
        or "gpu_vm_runtime_preflight_result:" in blocker
        or "launch_order_blocked_step:gpu_vm_runtime_preflight" in blocker
    ):
        action.update(
            {
                "category": "gpu_vm_runtime_preflight",
                "next_action": (
                    "Run the GPU VM runtime preflight script after syncing the packet "
                    "onto the selected GPU VM, then regenerate the packet/audit so the "
                    "result is bound into launch readiness."
                ),
                "evidence_required": (
                    "gpu_vm_runtime_preflight_result.json has "
                    "status=ready_for_owner_command_attempt."
                ),
                "safe_command": (
                    "GPU_VM_PREFLIGHT_OUTPUT=<result-json> "
                    "bash <packet-dir>/gpu_vm_runtime_preflight.sh"
                ),
            }
        )
    elif (
        "missing_first_gpu_run_packet" in blocker
        or "missing_first_gpu_blocker_resolution" in blocker
        or "missing_first_gpu_launch_order" in blocker
        or "blocker_resolution_action_count_mismatch" in blocker
        or "blocker_resolution_missing_actions_for_blocked_packet" in blocker
        or "first_gpu_run_packet_not_ready_for_attempt" in blocker
        or "first_gpu_run_packet_blocker:" in blocker
        or "launch_order_blocks_gpu_execution" in blocker
        or "launch_order_blocked_step:" in blocker
    ):
        action.update(
            {
                "category": "first_gpu_run_packet",
                "next_action": (
                    "Regenerate the first-GPU run packet after the upstream WebApp, "
                    "scene, GPU handoff, owner-command, and gate blockers are cleared."
                ),
                "evidence_required": (
                    "pipeline/first_gpu_e2e_run_packet/first_gpu_launch_order.json has "
                    "gpu_execution_allowed=true for the owner GPU attempt."
                ),
                "safe_command": (
                    "blueprint-build-first-gpu-run-packet --capture-root <capture-root> "
                    "--webapp-site-slug <site-slug> --simulator isaac_sim --provisioner runpod "
                    "--owner-command <remote-owner-command> --owner-command-location remote"
                ),
            }
        )
    elif (
        "missing_gpu_vm_sync_manifest" in blocker
        or "gpu_vm_sync_manifest_not_ready" in blocker
        or "gpu_vm_sync_manifest_blocker:" in blocker
    ):
        action.update(
            {
                "category": "gpu_vm_sync",
                "next_action": (
                    "Make every required raw, simulation-automation, and run-packet file "
                    "available, then regenerate the GPU VM sync manifest."
                ),
                "evidence_required": (
                    "gpu_vm_sync_manifest.json has status=ready and zero missing required files."
                ),
                "safe_command": (
                    "blueprint-build-first-gpu-run-packet --capture-root <capture-root> "
                    "--webapp-site-slug <site-slug> --owner-command <remote-owner-command> "
                    "--owner-command-location remote"
                ),
            }
        )
    return action


def _build_remediation_plan(blockers: Sequence[str]) -> Dict[str, Any]:
    actions = [_remediation_for_blocker(blocker) for blocker in blockers]
    categories: Dict[str, Dict[str, Any]] = {}
    for action in actions:
        category = str(action["category"])
        bucket = categories.setdefault(
            category,
            {
                "blocker_count": 0,
                "blockers": [],
                "next_actions": [],
                "evidence_required": [],
                "safe_commands": [],
            },
        )
        bucket["blocker_count"] += 1
        _append_unique(bucket["blockers"], [str(action["blocker"])])
        _append_unique(bucket["next_actions"], [str(action["next_action"])])
        _append_unique(bucket["evidence_required"], [str(action["evidence_required"])])
        if action.get("safe_command"):
            _append_unique(bucket["safe_commands"], [str(action["safe_command"])])
    priority = [
        "sample_capture",
        "webapp_upstream_truth",
        "webapp_forwarding_env",
        "webapp_staged_request",
        "webapp_live_forwarding_proof",
        "webapp_handoff_packet",
        "scene_asset_acquisition",
        "scene_spawn_preflight",
        "pipeline_gpu_handoff",
        "first_gpu_run_packet",
        "gpu_vm_sync",
        "gpu_vm_runtime_preflight",
        "owner_gpu_command",
        "owner_gpu_gate",
        "repo_contract_or_unknown",
    ]
    ordered_categories = {
        category: categories[category]
        for category in priority
        if category in categories
    }
    for category, value in categories.items():
        if category not in ordered_categories:
            ordered_categories[category] = value
    return {
        "status": "ready" if not blockers else "blocked",
        "action_count": len(actions),
        "categories": ordered_categories,
        "actions": actions,
        "proof_boundary": (
            "The remediation plan maps blockers to required evidence. It does not run "
            "WebApp, provision GPUs, execute simulators, or upgrade readiness proof."
        ),
    }


def _build_gpu_spend_decision(
    *,
    blockers: Sequence[str],
    remediation_plan: Mapping[str, Any],
    runtime_phase: Mapping[str, Any],
    simulator: str,
    provisioner: str,
) -> Dict[str, Any]:
    simulator_label = "Isaac" if simulator == "isaac_sim" else simulator
    categories = remediation_plan.get("categories")
    category_names = list(categories.keys()) if isinstance(categories, Mapping) else []
    runtime_ready = bool(runtime_phase.get("ready"))
    local_webapp_rehearsal_only = _runtime_has_local_webapp_rehearsal(runtime_phase)
    if local_webapp_rehearsal_only and "webapp_live_forwarding_proof" not in category_names:
        category_names.append("webapp_live_forwarding_proof")
    gpu_rental_recommended = runtime_ready and not blockers and not local_webapp_rehearsal_only
    status = "ready_to_rent_gpu_vm_for_owner_attempt" if gpu_rental_recommended else "do_not_rent_gpu_yet"
    minimum_evidence = [
        "A concrete capture root passes first-GPU readiness for the selected sample video.",
        "Real WebApp upstream IDs are present and are not placeholders.",
        "WebApp forwarding env and capture-root-by-site override are configured outside artifacts.",
        "A validated WebApp robot_eval_job_request.v1 is staged for this capture root.",
        "Scene asset, scene frame, spawn validation, and gpu_handoff_packet.json are ready.",
        "The first-GPU run packet launch order allows GPU execution, VM sync is ready, and VM preflight is safe.",
        "The owner simulator command is known for the GPU VM and the execution gate is explicit.",
    ]
    if gpu_rental_recommended:
        next_actions = [
            "Allocate an interactive GPU VM or pod for the owner attempt.",
            "Sync the packet and capture roots, then run the GPU VM runtime preflight before simulator execution.",
        ]
        must_not_do: list[str] = []
    else:
        next_actions = [
            "Clear the blocker categories in order before allocating paid GPU time.",
            "Regenerate the first-GPU run packet and this cross-repo audit after each material blocker is cleared.",
        ]
        must_not_do = [
            "do_not_allocate_runpod_or_equivalent_gpu_vm",
            "do_not_run_gpu_vm_commands",
            "do_not_claim_webapp_live_forwarding",
            "do_not_claim_scene_asset_or_gpu_handoff_ready",
            "do_not_claim_owner_gpu_or_rank_fidelity",
        ]
    return {
        "status": status,
        "gpu_rental_recommended_now": gpu_rental_recommended,
        "full_e2e_webapp_live_forwarding_required_evidence_present": (
            runtime_ready and not local_webapp_rehearsal_only
        ),
        "local_webapp_rehearsal_only_observed": local_webapp_rehearsal_only,
        "selected_simulator": simulator,
        "selected_provisioner": provisioner,
        "recommended_first_gpu_environment": "interactive_gpu_vm_or_pod",
        "runpod_fit": (
            "RunPod Pod or equivalent full-control GPU VM is the preferred first debug "
            "environment once this decision becomes ready."
        ),
        "nvidia_nim_role": (
            "NVIDIA NIM can support model inference services later; it is not the primary "
            f"{simulator_label}/physics simulator runtime for this first owner-runtime smoke."
        ),
        "pre_spend_blocker_categories": category_names,
        "minimum_evidence_before_gpu_spend": minimum_evidence,
        "next_actions": next_actions,
        "must_not_do_until_ready": must_not_do,
        "first_gpu_scope": [
            "scene load",
            "robot spawn",
            "task or action trace proof through the owner GPU proof wrapper",
        ],
        "claim_boundary": {
            "artifact_purpose": "first_gpu_gpu_spend_decision",
            "gpu_provisioning_performed": False,
            "simulator_execution_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _external_input_catalog(
    *,
    capture_root: str | Path | None,
    webapp_site_slug: str,
    simulator: str,
    provisioner: str,
) -> Dict[str, Dict[str, Any]]:
    root_text = str(Path(capture_root).expanduser()) if capture_root else "<capture-root>"
    slug_text = webapp_site_slug or "<webapp-site-slug>"
    return {
        "sample_capture": {
            "title": "Concrete Sample Capture Root",
            "required_inputs": [
                {
                    "name": "capture_root",
                    "kind": "path",
                    "secret": False,
                    "expected_value": root_text,
                    "source": "Capture app upload or strict sample-video staging",
                },
            ],
        },
        "webapp_upstream_truth": {
            "title": "Real WebApp Upstream IDs",
            "required_inputs": [
                {
                    "name": field,
                    "kind": "upstream_id",
                    "secret": False,
                    "source": "real WebApp/Capture request path",
                }
                for field in (
                    "site_submission_id",
                    "request_id",
                    "buyer_request_id",
                    "capture_job_id",
                )
            ],
        },
        "webapp_forwarding_env": {
            "title": "WebApp Forwarding Environment",
            "required_inputs": [
                {
                    "name": "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL",
                    "kind": "env",
                    "secret": False,
                    "source": "WebApp deployment or operator shell",
                },
                {
                    "name": "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN",
                    "kind": "env",
                    "secret": True,
                    "source": "WebApp deployment or operator shell",
                },
                {
                    "name": "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
                    "kind": "env",
                    "secret": False,
                    "expected_value": f'{{"{slug_text}":"{root_text}"}}',
                    "source": "WebApp deployment or operator shell",
                },
            ],
        },
        "webapp_staged_request": {
            "title": "Staged WebApp Robot-Eval Request",
            "required_inputs": [
                {
                    "name": "robot_eval_job_request.v1",
                    "kind": "artifact",
                    "secret": False,
                    "source": "real WebApp request route or Pipeline intake service",
                },
                {
                    "name": "pipeline/live_pipeline_staged_inputs.json",
                    "kind": "artifact",
                    "secret": False,
                    "source": "blueprint-intake-live-pipeline-inputs",
                },
            ],
        },
        "webapp_live_forwarding_proof": {
            "title": "Real WebApp Forwarding Proof",
            "required_inputs": [
                {
                    "name": "non_rehearsal_webapp_staged_request",
                    "kind": "artifact",
                    "secret": False,
                    "source": "WebApp-forwarded request not marked local_first_gpu_rehearsal_request",
                },
            ],
        },
        "webapp_handoff_packet": {
            "title": "WebApp Handoff Packet",
            "required_inputs": [
                {
                    "name": "first_gpu_webapp_handoff.json",
                    "kind": "artifact",
                    "secret": False,
                    "source": "blueprint-build-first-gpu-run-packet",
                },
            ],
        },
        "scene_asset_acquisition": {
            "title": "Scene Asset Acquisition",
            "required_inputs": [
                {
                    "name": "WORLDLABS_API_KEY",
                    "kind": "env",
                    "secret": True,
                    "source": "operator shell or provider secret store",
                },
                {
                    "name": "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION",
                    "kind": "env",
                    "secret": False,
                    "expected_value": "true",
                    "source": "operator shell for the intentional provider call",
                },
                {
                    "name": "pipeline/worldlabs_world_manifest.json",
                    "kind": "artifact",
                    "secret": False,
                    "source": "World Labs provider request completion",
                },
                {
                    "name": "pipeline/worldlabs_assets/materialized_assets_manifest.json",
                    "kind": "artifact",
                    "secret": False,
                    "source": "World Labs asset materialization",
                },
            ],
        },
        "scene_spawn_preflight": {
            "title": "Scene And Spawn Preflight",
            "required_inputs": [
                {
                    "name": "materialized_scene_asset",
                    "kind": "path",
                    "secret": False,
                    "source": "World Labs materialization or owner-provided scene asset",
                },
                {
                    "name": "spawn_pose_validation_manifest.json",
                    "kind": "artifact",
                    "secret": False,
                    "source": "simulation automation pre-GPU pass",
                },
            ],
        },
        "pipeline_gpu_handoff": {
            "title": "Pipeline GPU Handoff",
            "required_inputs": [
                {
                    "name": "simulation_automation/gpu_handoff_packet.json",
                    "kind": "artifact",
                    "secret": False,
                    "expected_value": "status=ready_for_owner_gpu_preflight_handoff",
                    "source": "blueprint-run-simulation-automation",
                },
            ],
        },
        "first_gpu_run_packet": {
            "title": "First-GPU Run Packet",
            "required_inputs": [
                {
                    "name": "pipeline/first_gpu_e2e_run_packet/first_gpu_run_packet.json",
                    "kind": "artifact",
                    "secret": False,
                    "source": "blueprint-build-first-gpu-run-packet",
                },
            ],
        },
        "gpu_vm_sync": {
            "title": "GPU VM Sync",
            "required_inputs": [
                {
                    "name": "gpu_vm_sync_manifest.json",
                    "kind": "artifact",
                    "secret": False,
                    "expected_value": "status=ready",
                    "source": "first-GPU run packet",
                },
            ],
        },
        "gpu_vm_runtime_preflight": {
            "title": "GPU VM Runtime Preflight",
            "required_inputs": [
                {
                    "name": "nvidia-smi",
                    "kind": "gpu_vm_command",
                    "secret": False,
                    "source": "RunPod or equivalent GPU VM",
                },
                {
                    "name": "gpu_vm_runtime_preflight_result.json",
                    "kind": "artifact",
                    "secret": False,
                    "expected_value": "status=ready_for_owner_command_attempt",
                    "source": "gpu_vm_runtime_preflight.sh",
                },
            ],
        },
        "owner_gpu_command": {
            "title": "Owner GPU Simulator Command",
            "required_inputs": [
                {
                    "name": "OWNER_SIMULATOR_COMMAND",
                    "kind": "env_or_cli_arg",
                    "secret": False,
                    "source": f"owner {simulator} runtime on {provisioner}",
                },
                {
                    "name": "owner_default_smoke_command_binding.sh",
                    "kind": "artifact",
                    "secret": False,
                    "source": "first-GPU run packet generated owner command binding template",
                },
                {
                    "name": "OWNER_SCENE_LOAD_COMMAND",
                    "kind": "env",
                    "secret": False,
                    "source": (
                        "owner simulator runtime command that writes "
                        "BLUEPRINT_SCENE_LOAD_TRACE"
                    ),
                },
                {
                    "name": "OWNER_ROBOT_SPAWN_COMMAND",
                    "kind": "env",
                    "secret": False,
                    "source": (
                        "owner simulator runtime command that writes "
                        "BLUEPRINT_SPAWN_TRACE"
                    ),
                },
                {
                    "name": "OWNER_WALK_TO_TARGET_COMMAND",
                    "kind": "env",
                    "secret": False,
                    "source": (
                        "owner simulator runtime command that runs the default "
                        "walk_to_target policy"
                    ),
                },
                {
                    "name": "SIM_ROBOT_POV_FRAME_PATH or SIM_ROBOT_POV_VIDEO_PATH",
                    "kind": "path",
                    "secret": False,
                    "source": "simulator robot camera evidence emitted by the owner command",
                },
            ],
        },
        "owner_gpu_gate": {
            "title": "Owner GPU Execution Gate",
            "required_inputs": [
                {
                    "name": "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
                    "kind": "env",
                    "secret": False,
                    "expected_value": "true",
                    "source": "operator shell for the intentional owner GPU attempt",
                },
            ],
        },
        "repo_contract_or_unknown": {
            "title": "Repo Contract Repair",
            "required_inputs": [
                {
                    "name": "repo_contract_fix",
                    "kind": "code_or_contract",
                    "secret": False,
                    "source": "source repo audit",
                },
            ],
        },
    }


def _first_gpu_proof_scope(
    run_packet_phase: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    checks = (
        run_packet_phase.get("checks")
        if isinstance(run_packet_phase, Mapping)
        and isinstance(run_packet_phase.get("checks"), Mapping)
        else {}
    )
    packet_check = (
        checks.get("first_gpu_run_packet")
        if isinstance(checks.get("first_gpu_run_packet"), Mapping)
        else {}
    )
    owner_binding_path = _string(packet_check.get("owner_command_binding_template_path"))
    live_policy_contract_path = _string(
        packet_check.get("live_policy_execution_contract_path")
    )
    default_job_template_path = _string(
        packet_check.get("default_test_robot_eval_job_request_template_path")
    )
    real_robot_pov_template_path = _string(
        packet_check.get("real_robot_pov_manifest_template_path")
    )
    live_input_staging_script_path = _string(
        packet_check.get("live_input_staging_commands_path")
    )
    return {
        "default_simulator_smoke": {
            "status": "not_run_by_this_audit",
            "policy": "walk_to_target",
            "can_prove_after_successful_owner_gpu_run": [
                "scene loaded in the owner simulator",
                "robot spawned in the owner simulator",
                "default walk_to_target smoke policy executed",
                "simulator robot POV frame or video captured",
            ],
            "required_artifacts_after_owner_gpu_run": [
                (
                    "pipeline/simulation_automation/owner_gpu_proof/"
                    "owner_default_smoke_policy.json"
                ),
                (
                    "pipeline/simulation_automation/owner_gpu_proof/"
                    "owner_policy_execution_trace.json"
                ),
                (
                    "pipeline/simulation_automation/owner_gpu_proof/"
                    "owner_sim_robot_pov_evidence_manifest.json"
                ),
                (
                    "pipeline/simulation_automation/"
                    "owner_gpu_simulator_execution_proof_manifest.json"
                ),
            ],
            "owner_binding_template_path": owner_binding_path or None,
            "owner_binding_template_exists": bool(
                owner_binding_path and Path(owner_binding_path).is_file()
            ),
        },
        "not_proven_by_first_gpu_smoke": [
            {
                "claim": "live_robot_team_policy_execution",
                "reason": (
                    "The first smoke executes Blueprint's default walk_to_target "
                    "policy, not a robot-team policy package or API."
                ),
                "proof_required": [
                    "pipeline/robot_eval_jobs/<job_id>/policy_execution_manifest.json",
                    "pipeline/robot_eval_jobs/<job_id>/policy_execution_trace.json",
                    "BLUEPRINT_ALLOW_POLICY_EXECUTION=true for the gated job execution",
                ],
            },
            {
                "claim": "real_robot_pov_evidence",
                "reason": (
                    "Simulator camera evidence is not physical robot camera evidence. "
                    "Generated POV support and simulator POV stay separate from real POV."
                ),
                "proof_required": [
                    "pipeline/robot_eval_inputs/real_robot_pov_manifest.json",
                    "robot_camera_video_uri for each required scenario eval run",
                    "action_log_uri aligned to each required scenario eval run",
                ],
            },
        ],
        "contract_artifacts": {
            "live_policy_execution_contract_path": live_policy_contract_path or None,
            "live_policy_execution_contract_exists": bool(
                live_policy_contract_path and Path(live_policy_contract_path).is_file()
            ),
        },
        "live_input_templates": {
            "default_test_robot_eval_job_request_template_path": (
                default_job_template_path or None
            ),
            "default_test_robot_eval_job_request_template_exists": bool(
                default_job_template_path and Path(default_job_template_path).is_file()
            ),
            "real_robot_pov_manifest_template_path": real_robot_pov_template_path or None,
            "real_robot_pov_manifest_template_exists": bool(
                real_robot_pov_template_path and Path(real_robot_pov_template_path).is_file()
            ),
            "live_input_staging_script_path": live_input_staging_script_path or None,
            "live_input_staging_script_exists": bool(
                live_input_staging_script_path
                and Path(live_input_staging_script_path).is_file()
            ),
            "staging_gate": "BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS=true",
        },
    }


def _append_guarded_command(
    target: Dict[str, list[Dict[str, Any]]],
    category: str,
    *,
    name: str,
    path: str,
    command: str | None = None,
    safe_to_run_now: bool,
    runs_live_provider_call: bool = False,
    runs_live_webapp_call: bool = False,
    runs_owner_simulator_command: bool = False,
    requires_explicit_allow_env: str | None = None,
    proof_boundary: str,
) -> None:
    path_text = _string(path)
    if not path_text:
        return
    entry = {
        "name": name,
        "path": path_text,
        "command": command or f"bash {shlex.quote(path_text)}",
        "safe_to_run_now": safe_to_run_now,
        "runs_live_provider_call": runs_live_provider_call,
        "runs_live_webapp_call": runs_live_webapp_call,
        "runs_owner_simulator_command": runs_owner_simulator_command,
        "proof_boundary": proof_boundary,
    }
    if requires_explicit_allow_env:
        entry["requires_explicit_allow_env"] = requires_explicit_allow_env
    bucket = target.setdefault(category, [])
    if not any(
        item.get("name") == entry["name"] and item.get("path") == entry["path"]
        for item in bucket
    ):
        bucket.append(entry)


def _guarded_commands_by_category(
    run_packet_phase: Mapping[str, Any] | None,
) -> Dict[str, list[Dict[str, Any]]]:
    if not isinstance(run_packet_phase, Mapping):
        return {}
    checks = (
        run_packet_phase.get("checks")
        if isinstance(run_packet_phase.get("checks"), Mapping)
        else {}
    )
    guarded: Dict[str, list[Dict[str, Any]]] = {}

    packet_check = (
        checks.get("first_gpu_run_packet")
        if isinstance(checks.get("first_gpu_run_packet"), Mapping)
        else {}
    )
    upstream_script_path = _string(
        packet_check.get("webapp_upstream_truth_verification_script_path")
    )
    if not upstream_script_path and _string(run_packet_phase.get("packet_dir")):
        fallback_upstream_script_path = (
            Path(_string(run_packet_phase.get("packet_dir")))
            / "webapp_upstream_truth_verification_commands.sh"
        )
        if fallback_upstream_script_path.is_file():
            upstream_script_path = str(fallback_upstream_script_path)
    _append_guarded_command(
        guarded,
        "webapp_upstream_truth",
        name="webapp_upstream_truth_verification_commands",
        path=upstream_script_path,
        safe_to_run_now=bool(upstream_script_path)
        and Path(upstream_script_path).is_file(),
        runs_live_provider_call=False,
        runs_live_webapp_call=False,
        runs_owner_simulator_command=False,
        proof_boundary=(
            "Verifies that real non-placeholder WebApp upstream IDs are present in "
            "accepted capture, descriptor, or handoff artifacts; it does not mutate "
            "artifacts or submit a WebApp request."
        ),
    )

    scene_check = (
        checks.get("scene_asset_acquisition")
        if isinstance(checks.get("scene_asset_acquisition"), Mapping)
        else {}
    )
    _append_guarded_command(
        guarded,
        "scene_asset_acquisition",
        name="worldlabs_provider_submission_commands",
        path=_string(scene_check.get("provider_submission_script_path")),
        safe_to_run_now=bool(scene_check.get("provider_submission_script_safe_to_run_now")),
        runs_live_provider_call=True,
        requires_explicit_allow_env=_string(
            scene_check.get("provider_submission_script_requires_allow_env")
        )
        or None,
        proof_boundary=(
            "Submits the intentional World Labs provider request only after the API key, "
            "source-video preflight, and explicit provider-submission gate are present; "
            "it does not provision GPUs or run the simulator."
        ),
    )

    webapp_check = (
        checks.get("webapp_handoff")
        if isinstance(checks.get("webapp_handoff"), Mapping)
        else {}
    )
    webapp_verification_path = _string(webapp_check.get("verification_script_path"))
    for category in (
        "webapp_forwarding_env",
        "webapp_staged_request",
        "webapp_live_forwarding_proof",
        "webapp_handoff_packet",
    ):
        _append_guarded_command(
            guarded,
            category,
            name="webapp_handoff_verification_commands",
            path=webapp_verification_path,
            safe_to_run_now=bool(
                webapp_check.get("verification_script_safe_to_run_now")
            ),
            runs_live_webapp_call=bool(
                webapp_check.get("verification_runs_live_webapp_call")
            ),
            proof_boundary=(
                "Verifies forwarding environment, capture-root override, staged request "
                "shape, and real upstream IDs from artifacts; it does not submit a "
                "WebApp request or call the simulator."
            ),
        )

    runtime_check = (
        checks.get("gpu_vm_runtime_preflight_plan")
        if isinstance(checks.get("gpu_vm_runtime_preflight_plan"), Mapping)
        else {}
    )
    runtime_script_path = _string(runtime_check.get("script_path"))
    _append_guarded_command(
        guarded,
        "gpu_vm_runtime_preflight",
        name="gpu_vm_runtime_preflight",
        path=runtime_script_path,
        safe_to_run_now=bool(runtime_check.get("safe_to_run_on_gpu_vm")),
        runs_owner_simulator_command=False,
        proof_boundary=(
            "Checks the GPU VM, owner command executable, and synced file hashes; it "
            "does not run the owner simulator command."
        ),
    )

    owner_binding_template_path = _string(
        packet_check.get("owner_command_binding_template_path")
    )
    _append_guarded_command(
        guarded,
        "owner_gpu_command",
        name="owner_command_binding_template_syntax_check",
        path=owner_binding_template_path,
        command=f"bash -n {shlex.quote(owner_binding_template_path)}"
        if owner_binding_template_path
        else None,
        safe_to_run_now=bool(owner_binding_template_path)
        and Path(owner_binding_template_path).is_file(),
        runs_owner_simulator_command=False,
        proof_boundary=(
            "Checks the generated fail-closed owner command binding template syntax only; "
            "it does not run owner scene-load, spawn, policy, simulator, or GPU commands."
        ),
    )

    launch_check = (
        checks.get("launch_order")
        if isinstance(checks.get("launch_order"), Mapping)
        else {}
    )
    gpu_vm_commands_path = _string(runtime_check.get("gpu_vm_commands_path"))
    if not gpu_vm_commands_path and _string(run_packet_phase.get("packet_dir")):
        gpu_vm_commands_path = str(
            Path(_string(run_packet_phase.get("packet_dir"))) / "gpu_vm_commands.sh"
        )
    gpu_command_safe = bool(launch_check.get("gpu_execution_allowed")) and bool(
        runtime_check.get("safe_to_run_on_gpu_vm")
    )
    for category in ("owner_gpu_command", "owner_gpu_gate", "first_gpu_run_packet"):
        _append_guarded_command(
            guarded,
            category,
            name="gpu_vm_commands",
            path=gpu_vm_commands_path,
            safe_to_run_now=gpu_command_safe,
            runs_owner_simulator_command=True,
            requires_explicit_allow_env="BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
            proof_boundary=(
                "Runs the owner GPU proof wrapper and simulator command. It must stay "
                "blocked until WebApp, scene, GPU handoff, VM preflight, owner command, "
                "and explicit execution gate are ready."
            ),
        )

    return guarded


def _build_first_gpu_external_input_packet(
    *,
    capture_root: str | Path | None,
    webapp_site_slug: str,
    simulator: str,
    provisioner: str,
    remediation_plan: Mapping[str, Any],
    gpu_spend_decision: Mapping[str, Any],
    run_packet_phase: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    categories = (
        remediation_plan.get("categories")
        if isinstance(remediation_plan.get("categories"), Mapping)
        else {}
    )
    catalog = _external_input_catalog(
        capture_root=capture_root,
        webapp_site_slug=webapp_site_slug,
        simulator=simulator,
        provisioner=provisioner,
    )
    guarded_commands = _guarded_commands_by_category(run_packet_phase)
    missing_items: list[Dict[str, Any]] = []
    for category_id, details in categories.items():
        if not isinstance(details, Mapping):
            continue
        catalog_entry = catalog.get(str(category_id), {})
        missing_items.append(
            {
                "category_id": str(category_id),
                "title": catalog_entry.get("title") or str(category_id),
                "blocker_count": int(details.get("blocker_count") or 0),
                "blockers": [str(item) for item in details.get("blockers") or []],
                "required_inputs": [
                    dict(item)
                    for item in catalog_entry.get("required_inputs") or []
                    if isinstance(item, Mapping)
                ],
                "next_actions": [str(item) for item in details.get("next_actions") or []],
                "evidence_required": [
                    str(item) for item in details.get("evidence_required") or []
                ],
                "safe_commands": [str(item) for item in details.get("safe_commands") or []],
                "guarded_commands": [
                    dict(item)
                    for item in guarded_commands.get(str(category_id), [])
                    if isinstance(item, Mapping)
                ],
            }
        )
    return {
        "schema_version": "first_gpu_external_input_packet.v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if not missing_items else "blocked",
        "gpu_rental_recommended_now": bool(
            gpu_spend_decision.get("gpu_rental_recommended_now")
        ),
        "selected_simulator": simulator,
        "selected_provisioner": provisioner,
        "missing_input_category_count": len(missing_items),
        "missing_input_count": sum(len(item["required_inputs"]) for item in missing_items),
        "next_missing_category_id": (
            missing_items[0]["category_id"] if missing_items else None
        ),
        "missing_inputs": missing_items,
        "secret_handling": {
            "secrets_are_named_but_values_are_not_serialized": True,
            "secret_input_names": sorted(
                {
                    str(input_item.get("name"))
                    for item in missing_items
                    for input_item in item.get("required_inputs") or []
                    if isinstance(input_item, Mapping) and input_item.get("secret")
                }
            ),
        },
        "forbidden_actions_until_ready": list(
            gpu_spend_decision.get("must_not_do_until_ready") or []
        ),
        "first_gpu_proof_scope": _first_gpu_proof_scope(run_packet_phase),
        "claim_boundary": {
            "artifact_purpose": "first_gpu_external_input_packet",
            "external_inputs_collected": False,
            "live_provider_calls_performed": False,
            "webapp_requests_submitted": False,
            "gpu_provisioning_performed": False,
            "simulator_execution_performed": False,
            "default_sim_policy_execution_proven": False,
            "sim_robot_pov_evidence_proven": False,
            "robot_policy_execution_proven": False,
            "real_robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _first_gpu_external_input_packet_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# First GPU External Input Packet",
        "",
        f"- Schema: `{packet.get('schema_version')}`",
        f"- Status: `{packet.get('status')}`",
        f"- Generated: `{packet.get('generated_at')}`",
        f"- GPU rental recommended now: `{packet.get('gpu_rental_recommended_now')}`",
        f"- Selected simulator: `{packet.get('selected_simulator')}`",
        f"- Selected provisioner: `{packet.get('selected_provisioner')}`",
        f"- Missing input categories: `{packet.get('missing_input_category_count')}`",
        f"- Missing input count: `{packet.get('missing_input_count')}`",
        f"- Next missing category: `{packet.get('next_missing_category_id')}`",
        "",
        "This packet names required external inputs only. It does not collect secrets, submit WebApp requests, call providers, provision GPUs, run simulators, or prove generated-world rank fidelity.",
        "",
    ]
    secret_handling = (
        packet.get("secret_handling")
        if isinstance(packet.get("secret_handling"), Mapping)
        else {}
    )
    secret_names = [str(item) for item in secret_handling.get("secret_input_names") or []]
    lines.extend(["## Secret Handling", ""])
    lines.append(
        "- Secrets are named but values are not serialized: "
        f"`{secret_handling.get('secrets_are_named_but_values_are_not_serialized')}`"
    )
    if secret_names:
        lines.append("- Secret input names:")
        lines.extend(f"  - `{item}`" for item in secret_names)
    else:
        lines.append("- Secret input names: none")
    forbidden = [str(item) for item in packet.get("forbidden_actions_until_ready") or []]
    if forbidden:
        lines.extend(["", "## Forbidden Until Ready", ""])
        lines.extend(f"- `{item}`" for item in forbidden)
    proof_scope = (
        packet.get("first_gpu_proof_scope")
        if isinstance(packet.get("first_gpu_proof_scope"), Mapping)
        else {}
    )
    default_smoke = (
        proof_scope.get("default_simulator_smoke")
        if isinstance(proof_scope.get("default_simulator_smoke"), Mapping)
        else {}
    )
    lines.extend(["", "## Proof Scope", ""])
    lines.append(
        "- Default owner-GPU smoke status: "
        f"`{default_smoke.get('status') or 'not_run_by_this_audit'}`"
    )
    if default_smoke.get("policy"):
        lines.append(f"- Default smoke policy: `{default_smoke.get('policy')}`")
    can_prove = [
        str(item)
        for item in default_smoke.get("can_prove_after_successful_owner_gpu_run") or []
    ]
    if can_prove:
        lines.append("- Can prove after a successful gated owner-GPU run:")
        lines.extend(f"  - {item}" for item in can_prove)
    required_after_run = [
        str(item)
        for item in default_smoke.get("required_artifacts_after_owner_gpu_run") or []
    ]
    if required_after_run:
        lines.append("- Required artifacts after owner-GPU run:")
        lines.extend(f"  - `{item}`" for item in required_after_run)
    if default_smoke.get("owner_binding_template_path"):
        lines.append(
            "- Owner binding template: "
            f"`{default_smoke.get('owner_binding_template_path')}` "
            f"(exists=`{default_smoke.get('owner_binding_template_exists')}`)"
        )
    contract_artifacts = (
        proof_scope.get("contract_artifacts")
        if isinstance(proof_scope.get("contract_artifacts"), Mapping)
        else {}
    )
    if contract_artifacts.get("live_policy_execution_contract_path"):
        lines.append(
            "- Live policy contract: "
            f"`{contract_artifacts.get('live_policy_execution_contract_path')}` "
            f"(exists=`{contract_artifacts.get('live_policy_execution_contract_exists')}`)"
        )
    live_input_templates = (
        proof_scope.get("live_input_templates")
        if isinstance(proof_scope.get("live_input_templates"), Mapping)
        else {}
    )
    if live_input_templates:
        lines.append("- Live input templates:")
        template_fields = (
            (
                "default_test_robot_eval_job_request_template_path",
                "default_test_robot_eval_job_request_template_exists",
            ),
            (
                "real_robot_pov_manifest_template_path",
                "real_robot_pov_manifest_template_exists",
            ),
            ("live_input_staging_script_path", "live_input_staging_script_exists"),
        )
        for path_field, exists_field in template_fields:
            if live_input_templates.get(path_field):
                lines.append(
                    f"  - `{live_input_templates.get(path_field)}` "
                    f"(exists=`{live_input_templates.get(exists_field)}`)"
                )
        if live_input_templates.get("staging_gate"):
            lines.append(
                "- Live input staging gate: "
                f"`{live_input_templates.get('staging_gate')}`"
            )
    not_proven = [
        item
        for item in proof_scope.get("not_proven_by_first_gpu_smoke") or []
        if isinstance(item, Mapping)
    ]
    if not_proven:
        lines.append("- Not proven by the first-GPU default smoke:")
        for item in not_proven:
            lines.append(f"  - `{item.get('claim')}`: {item.get('reason')}")
            proof_required = [str(value) for value in item.get("proof_required") or []]
            if proof_required:
                lines.append("    Required proof:")
                lines.extend(f"    - `{value}`" for value in proof_required)
    lines.extend(["", "## Missing Inputs", ""])
    missing_inputs = [
        item for item in packet.get("missing_inputs") or [] if isinstance(item, Mapping)
    ]
    if not missing_inputs:
        lines.append("- None.")
    for item in missing_inputs:
        lines.extend(
            [
                f"### {item.get('title') or item.get('category_id')}",
                "",
                f"- Category: `{item.get('category_id')}`",
                f"- Blocker count: `{item.get('blocker_count')}`",
            ]
        )
        required_inputs = [
            value
            for value in item.get("required_inputs") or []
            if isinstance(value, Mapping)
        ]
        if required_inputs:
            lines.append("- Required inputs:")
            for required in required_inputs:
                bits = [
                    f"name=`{required.get('name')}`",
                    f"kind=`{required.get('kind')}`",
                    f"secret=`{required.get('secret')}`",
                ]
                if required.get("expected_value"):
                    bits.append(f"expected=`{required.get('expected_value')}`")
                if required.get("source"):
                    bits.append(f"source=`{required.get('source')}`")
                lines.append(f"  - {'; '.join(bits)}")
        evidence = [str(value) for value in item.get("evidence_required") or []]
        if evidence:
            lines.append("- Evidence required:")
            lines.extend(f"  - {value}" for value in evidence)
        next_actions = [str(value) for value in item.get("next_actions") or []]
        if next_actions:
            lines.append("- Next actions:")
            lines.extend(f"  - {value}" for value in next_actions)
        safe_commands = [str(value) for value in item.get("safe_commands") or []]
        if safe_commands:
            lines.extend(["- Safe commands:", "", "```bash"])
            lines.extend(safe_commands)
            lines.extend(["```", ""])
        guarded_commands = [
            value
            for value in item.get("guarded_commands") or []
            if isinstance(value, Mapping)
        ]
        if guarded_commands:
            lines.append("- Guarded packet scripts:")
            for guarded in guarded_commands:
                bits = [
                    f"name=`{guarded.get('name')}`",
                    f"path=`{guarded.get('path')}`",
                    f"safe_to_run_now=`{guarded.get('safe_to_run_now')}`",
                    (
                        "runs_live_provider_call="
                        f"`{guarded.get('runs_live_provider_call')}`"
                    ),
                    f"runs_live_webapp_call=`{guarded.get('runs_live_webapp_call')}`",
                    (
                        "runs_owner_simulator_command="
                        f"`{guarded.get('runs_owner_simulator_command')}`"
                    ),
                ]
                if guarded.get("requires_explicit_allow_env"):
                    bits.append(
                        "requires_explicit_allow_env="
                        f"`{guarded.get('requires_explicit_allow_env')}`"
                    )
                lines.append(f"  - {'; '.join(bits)}")
            command_lines = [
                str(guarded.get("command"))
                for guarded in guarded_commands
                if guarded.get("command")
            ]
            if command_lines:
                lines.extend(["- Guarded command lines:", "", "```bash"])
                lines.extend(command_lines)
                lines.extend(["```", ""])
        blockers = [str(value) for value in item.get("blockers") or []]
        if blockers:
            lines.append("- Current blockers:")
            lines.extend(f"  - `{value}`" for value in blockers)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_cross_repo_first_gpu_readiness(
    *,
    pipeline_repo: str | Path | None = None,
    capture_repo: str | Path | None = None,
    webapp_repo: str | Path | None = None,
    capture_root: str | Path | None = None,
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
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    selected_pipeline_repo = _resolve_repo(pipeline_repo, default=_default_pipeline_repo())
    selected_capture_repo = _resolve_repo(
        capture_repo,
        default=_default_adjacent_repo("BlueprintCapture"),
    )
    selected_webapp_repo = _resolve_repo(
        webapp_repo,
        default=_default_adjacent_repo("Blueprint-WebApp"),
    )
    phases = {
        "capture_to_pipeline": _capture_to_pipeline_phase(selected_capture_repo),
        "webapp_to_pipeline": _webapp_request_phase(selected_webapp_repo),
        "pipeline_return": _pipeline_return_phase(selected_pipeline_repo),
        "runtime_capture": _runtime_capture_phase(
            capture_root=capture_root,
            webapp_site_slug=webapp_site_slug,
            webapp_staged_inputs_path=webapp_staged_inputs_path,
            webapp_forwarding_preflight_path=webapp_forwarding_preflight_path,
            simulator=simulator,
            provisioner=provisioner,
            simulator_command=simulator_command,
            simulator_command_location=simulator_command_location,
            require_webapp_forwarding=require_webapp_forwarding,
            require_webapp_staged_request=require_webapp_staged_request,
            allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
            require_gpu_gates=require_gpu_gates,
        ),
        "run_packet": _run_packet_phase(capture_root=capture_root),
    }
    blockers: List[str] = []
    warnings: List[str] = []
    for phase_name, phase in phases.items():
        _append_unique(blockers, (f"{phase_name}:{item}" for item in phase.get("blockers", [])))
        _append_unique(warnings, (f"{phase_name}:{item}" for item in phase.get("warnings", [])))
    local_webapp_rehearsal_only = _runtime_has_local_webapp_rehearsal(
        phases["runtime_capture"],
    )
    if local_webapp_rehearsal_only:
        _append_unique(
            blockers,
            [
                (
                    "full_e2e_webapp_live_forwarding:"
                    "local_webapp_rehearsal_not_live_forwarding_proof"
                )
            ],
        )
    runtime_ready = bool(phases["runtime_capture"].get("ready"))
    status = (
        "ready_for_owner_gpu_attempt"
        if not blockers and runtime_ready and not local_webapp_rehearsal_only
        else "blocked"
    )
    remediation_plan = _build_remediation_plan(blockers)
    gpu_spend_decision = _build_gpu_spend_decision(
        blockers=blockers,
        remediation_plan=remediation_plan,
        runtime_phase=phases["runtime_capture"],
        simulator=simulator,
        provisioner=provisioner,
    )
    simulator_label = "Isaac" if simulator == "isaac_sim" else simulator
    first_gpu_external_input_packet = _build_first_gpu_external_input_packet(
        capture_root=capture_root,
        webapp_site_slug=webapp_site_slug,
        simulator=simulator,
        provisioner=provisioner,
        remediation_plan=remediation_plan,
        gpu_spend_decision=gpu_spend_decision,
        run_packet_phase=phases["run_packet"],
    )
    first_gpu_operator_actions = [
        dict(item)
        for item in phases["run_packet"].get("operator_actions", []) or []
        if isinstance(item, Mapping)
    ]
    result = {
        "schema_version": CROSS_REPO_FIRST_GPU_READINESS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "ready_for_owner_gpu_attempt": status == "ready_for_owner_gpu_attempt",
        "full_e2e_webapp_live_forwarding_required_evidence_present": (
            runtime_ready and not local_webapp_rehearsal_only
        ),
        "local_webapp_rehearsal_only_observed": local_webapp_rehearsal_only,
        "repos": {
            "pipeline": str(selected_pipeline_repo),
            "capture": str(selected_capture_repo),
            "webapp": str(selected_webapp_repo),
        },
        "provider_guidance": {
            "recommended_first_gpu_provisioner": "runpod_or_equivalent_gpu_vm",
            "selected_provisioner": provisioner,
            "selected_simulator": simulator,
            "nvidia_nim_role": (
                "optional model inference microservices; not the primary "
                f"{simulator_label}/physics simulator runtime for the first owner-runtime smoke"
            ),
            "first_smoke_scope": (
                "scene load, robot spawn, and task/action trace proof through the owner GPU "
                "proof wrapper"
            ),
        },
        "phases": phases,
        "blockers": blockers,
        "warnings": warnings,
        "remediation_plan": remediation_plan,
        "gpu_spend_decision": gpu_spend_decision,
        "first_gpu_external_input_packet": first_gpu_external_input_packet,
        "first_gpu_operator_action_count": len(first_gpu_operator_actions),
        "blocked_first_gpu_operator_action_count": sum(
            1 for item in first_gpu_operator_actions if item.get("must_clear_before_gpu_spend")
        ),
        "first_gpu_operator_actions": first_gpu_operator_actions,
        "claim_boundary": {
            "artifact_purpose": "cross_repo_first_gpu_readiness_audit",
            "live_provider_calls_performed": False,
            "webapp_requests_submitted": False,
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "default_sim_policy_execution_proven": False,
            "sim_robot_pov_evidence_proven": False,
            "robot_policy_execution_proven": False,
            "real_robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    if output_path:
        output = Path(output_path).expanduser()
        ensure_dir(output.parent)
        write_json(output, result)
        result["output_path"] = str(output)
        external_input_packet_markdown_path = (
            output.parent / "first_gpu_external_input_packet.md"
        )
        write_text(
            external_input_packet_markdown_path,
            _first_gpu_external_input_packet_markdown(first_gpu_external_input_packet),
        )
        result["first_gpu_external_input_packet"]["markdown_path"] = str(
            external_input_packet_markdown_path
        )
        write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Capture -> Pipeline -> WebApp -> Pipeline readiness for first GPU E2E"
    )
    parser.add_argument("--pipeline-repo", default=None)
    parser.add_argument("--capture-repo", default=None)
    parser.add_argument("--webapp-repo", default=None)
    parser.add_argument("--capture-root", default=None)
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
    )
    parser.add_argument("--no-require-webapp-forwarding", action="store_true")
    parser.add_argument("--no-require-webapp-staged-request", action="store_true")
    parser.add_argument("--allow-local-webapp-rehearsal", action="store_true")
    parser.add_argument("--no-require-gpu-gates", action="store_true")
    parser.add_argument(
        "--output",
        default=str(default_artifact_cache_root() / "first_gpu_cross_repo_readiness_manifest.json"),
    )
    args = parser.parse_args(argv)

    result = build_cross_repo_first_gpu_readiness(
        pipeline_repo=args.pipeline_repo,
        capture_repo=args.capture_repo,
        webapp_repo=args.webapp_repo,
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
        output_path=args.output,
    )
    print(f"[first-gpu-cross-repo-readiness] status={result['status']}")
    print(f"[first-gpu-cross-repo-readiness] manifest={result['output_path']}")
    gpu_spend_decision = (
        result.get("gpu_spend_decision")
        if isinstance(result.get("gpu_spend_decision"), Mapping)
        else {}
    )
    external_input_packet = (
        result.get("first_gpu_external_input_packet")
        if isinstance(result.get("first_gpu_external_input_packet"), Mapping)
        else {}
    )
    print(
        "[first-gpu-cross-repo-readiness] gpu_spend_decision="
        + str(gpu_spend_decision.get("status"))
    )
    print(
        "[first-gpu-cross-repo-readiness] gpu_rental_recommended_now="
        + str(gpu_spend_decision.get("gpu_rental_recommended_now"))
    )
    print(
        "[first-gpu-cross-repo-readiness] external_input_packet_status="
        + str(external_input_packet.get("status"))
    )
    if external_input_packet.get("next_missing_category_id"):
        print(
            "[first-gpu-cross-repo-readiness] next_missing_category="
            + str(external_input_packet.get("next_missing_category_id"))
        )
    if external_input_packet.get("markdown_path"):
        print(
            "[first-gpu-cross-repo-readiness] external_input_packet_markdown="
            + str(external_input_packet.get("markdown_path"))
        )
    if result["blockers"]:
        print("[first-gpu-cross-repo-readiness] blockers=" + ",".join(result["blockers"]))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
