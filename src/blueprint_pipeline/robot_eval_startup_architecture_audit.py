"""Read-only audit for robot-eval startup architecture artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json


ROBOT_EVAL_STARTUP_ARCHITECTURE_AUDIT_SCHEMA_VERSION = (
    "robot_eval_startup_architecture_audit.v1"
)

EXPECTED_JOB_ARTIFACTS = {
    "job_request": "job_request.json",
    "job_run_manifest": "job_run_manifest.json",
    "scheduler_decision": "scheduler_decision.json",
    "worker_launch_plan": "worker_launch_plan.json",
    "worker_manifest": "worker_manifest.json",
    "gpu_provider_launch_request": "gpu_provider_launch_request.json",
    "gpu_cost_control_ledger": "gpu_cost_control_ledger.json",
}

OPTIONAL_JOB_ARTIFACTS = {
    "worker_runtime_manifest": "worker_runtime_manifest.json",
    "worker_runtime_preflight": "worker_runtime_preflight.json",
    "gpu_provider_launcher_result": "gpu_provider_launcher_result.json",
    "runpod_provider_adapter_result": "runpod_provider_adapter_result.json",
}

EXPECTED_OUTPUTS = {
    "scheduler_decision",
    "worker_launch_plan",
    "worker_manifest",
    "gpu_provider_launch_request",
    "gpu_provider_launcher_result",
    "runpod_provider_adapter_result",
    "gpu_cost_control_ledger",
    "startup_architecture_audit",
    "worker_runtime_manifest",
    "worker_runtime_preflight",
    "job_run_manifest",
    "proof_boundary",
    "metrics",
    "trace",
    "simulator_pov",
    "stdout_log",
    "stderr_log",
}

REQUIRED_PREFLIGHT_ARTIFACTS = {
    "scene_asset_inventory",
    "scene_asset_dependency_audit",
    "cpu_preflight_scorecard",
    "episode_spec_manifest",
    "gpu_handoff_packet",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _schema(payload: Mapping[str, Any]) -> str:
    return _string(payload.get("schema_version"))


def _bool(payload: Mapping[str, Any], path: Sequence[str]) -> bool | None:
    value = _get(payload, path)
    return value if isinstance(value, bool) else None


def _artifact_path(job_dir: Path, artifact_name: str) -> Path:
    return job_dir / EXPECTED_JOB_ARTIFACTS[artifact_name]


def _read_artifacts(job_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    for artifact_name, relative_path in EXPECTED_JOB_ARTIFACTS.items():
        path = job_dir / relative_path
        if not path.is_file():
            issues.append(
                {
                    "id": f"missing_{artifact_name}",
                    "artifact": artifact_name,
                    "path": str(path),
                    "message": f"Missing required startup artifact {relative_path}",
                }
            )
            artifacts[artifact_name] = {}
            continue
        try:
            payload = read_json_any(path)
        except Exception as exc:  # pragma: no cover - defensive around corrupted JSON
            issues.append(
                {
                    "id": f"invalid_json_{artifact_name}",
                    "artifact": artifact_name,
                    "path": str(path),
                    "message": f"Could not parse {relative_path}: {exc}",
                }
            )
            artifacts[artifact_name] = {}
            continue
        if not isinstance(payload, Mapping):
            issues.append(
                {
                    "id": f"non_object_{artifact_name}",
                    "artifact": artifact_name,
                    "path": str(path),
                    "message": f"{relative_path} must be a JSON object",
                }
            )
            artifacts[artifact_name] = {}
            continue
        artifacts[artifact_name] = dict(payload)
    for artifact_name, relative_path in OPTIONAL_JOB_ARTIFACTS.items():
        path = job_dir / relative_path
        if not path.is_file():
            artifacts[artifact_name] = {}
            continue
        try:
            payload = read_json_any(path)
        except Exception as exc:  # pragma: no cover - defensive around corrupted JSON
            issues.append(
                {
                    "id": f"invalid_json_{artifact_name}",
                    "artifact": artifact_name,
                    "path": str(path),
                    "message": f"Could not parse optional {relative_path}: {exc}",
                }
            )
            artifacts[artifact_name] = {}
            continue
        if not isinstance(payload, Mapping):
            issues.append(
                {
                    "id": f"non_object_{artifact_name}",
                    "artifact": artifact_name,
                    "path": str(path),
                    "message": f"Optional {relative_path} must be a JSON object",
                }
            )
            artifacts[artifact_name] = {}
            continue
        artifacts[artifact_name] = dict(payload)
    return artifacts, issues


def _append_check(
    checks: list[dict[str, Any]],
    blockers: list[str],
    *,
    check_id: str,
    passed: bool,
    message: str,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "passed" if passed else "blocked",
            "message": message,
            "evidence": dict(evidence or {}),
        }
    )
    if not passed:
        blockers.append(check_id)


def _append_schema_check(
    checks: list[dict[str, Any]],
    blockers: list[str],
    *,
    artifact_name: str,
    payload: Mapping[str, Any],
    expected_schema: str,
) -> None:
    actual = _schema(payload)
    _append_check(
        checks,
        blockers,
        check_id=f"{artifact_name}:schema",
        passed=actual == expected_schema,
        message=f"{artifact_name} uses expected schema {expected_schema}",
        evidence={"expected": expected_schema, "actual": actual},
    )


def _startup_artifact_checks(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_issues: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    for issue in artifact_issues:
        check_id = _string(issue.get("id")) or "artifact_issue"
        _append_check(
            checks,
            blockers,
            check_id=check_id,
            passed=False,
            message=_string(issue.get("message")) or "Required startup artifact issue",
            evidence=issue,
        )

    job_request = artifacts.get("job_request", {})
    run_manifest = artifacts.get("job_run_manifest", {})
    scheduler = artifacts.get("scheduler_decision", {})
    worker = artifacts.get("worker_launch_plan", {})
    worker_manifest = artifacts.get("worker_manifest", {})
    provider = artifacts.get("gpu_provider_launch_request", {})
    ledger = artifacts.get("gpu_cost_control_ledger", {})
    worker_runtime_manifest = artifacts.get("worker_runtime_manifest", {})
    worker_runtime_preflight = artifacts.get("worker_runtime_preflight", {})
    provider_launcher_result = artifacts.get("gpu_provider_launcher_result", {})
    runpod_adapter_result = artifacts.get("runpod_provider_adapter_result", {})

    _append_schema_check(
        checks,
        blockers,
        artifact_name="job_request",
        payload=job_request,
        expected_schema="robot_eval_job_request.v1",
    )
    _append_schema_check(
        checks,
        blockers,
        artifact_name="job_run_manifest",
        payload=run_manifest,
        expected_schema="robot_eval_job_run_manifest.v1",
    )
    _append_schema_check(
        checks,
        blockers,
        artifact_name="scheduler_decision",
        payload=scheduler,
        expected_schema="robot_eval_execution_scheduler_decision.v1",
    )
    _append_schema_check(
        checks,
        blockers,
        artifact_name="worker_launch_plan",
        payload=worker,
        expected_schema="robot_eval_worker_launch_plan.v1",
    )
    _append_schema_check(
        checks,
        blockers,
        artifact_name="worker_manifest",
        payload=worker_manifest,
        expected_schema="robot_eval_worker_manifest.v1",
    )
    _append_schema_check(
        checks,
        blockers,
        artifact_name="gpu_provider_launch_request",
        payload=provider,
        expected_schema="robot_eval_gpu_provider_launch_request.v1",
    )
    _append_schema_check(
        checks,
        blockers,
        artifact_name="gpu_cost_control_ledger",
        payload=ledger,
        expected_schema="robot_eval_gpu_cost_control_ledger.v1",
    )
    worker_runtime_manifest_present = bool(worker_runtime_manifest)
    if worker_runtime_manifest_present:
        _append_schema_check(
            checks,
            blockers,
            artifact_name="worker_runtime_manifest",
            payload=worker_runtime_manifest,
            expected_schema="robot_eval_worker_runtime_manifest.v1",
        )
        worker_runtime_preflight_required = (
            worker_runtime_manifest.get("runtime_preflight_required_before_scene_load")
            is True
        )
        worker_runtime_preflight_present = bool(worker_runtime_preflight)
        _append_check(
            checks,
            blockers,
            check_id="worker_runtime:preflight_artifact_present",
            passed=not worker_runtime_preflight_required
            or worker_runtime_preflight_present,
            message="Worker runtime manifest references a preflight artifact when pre-scene preflight is required",
            evidence={
                "runtime_preflight_required_before_scene_load": worker_runtime_preflight_required,
                "runtime_preflight_manifest_path": worker_runtime_manifest.get(
                    "runtime_preflight_manifest_path"
                ),
                "worker_runtime_preflight_present": worker_runtime_preflight_present,
                "runtime_preflight_status": worker_runtime_manifest.get(
                    "runtime_preflight_status"
                ),
            },
        )
        if worker_runtime_preflight_present:
            _append_schema_check(
                checks,
                blockers,
                artifact_name="worker_runtime_preflight",
                payload=worker_runtime_preflight,
                expected_schema="robot_eval_worker_runtime_preflight.v1",
            )
            _append_check(
                checks,
                blockers,
                check_id="worker_runtime:preflight_status_consistent",
                passed=_string(worker_runtime_manifest.get("runtime_preflight_status"))
                == _string(worker_runtime_preflight.get("status"))
                and worker_runtime_preflight.get("simulator_execution_proven") is False
                and worker_runtime_preflight.get("robot_readiness_proven") is False
                and worker_runtime_preflight.get("public_claim_upgrade_allowed") is False,
                message="Worker runtime preflight artifact matches the runtime manifest and cannot upgrade proof",
                evidence={
                    "runtime_manifest_status": worker_runtime_manifest.get(
                        "runtime_preflight_status"
                    ),
                    "preflight_status": worker_runtime_preflight.get("status"),
                    "simulator_execution_proven": worker_runtime_preflight.get(
                        "simulator_execution_proven"
                    ),
                    "robot_readiness_proven": worker_runtime_preflight.get(
                        "robot_readiness_proven"
                    ),
                    "public_claim_upgrade_allowed": worker_runtime_preflight.get(
                        "public_claim_upgrade_allowed"
                    ),
                },
            )
            preflight_command = _mapping(worker_runtime_preflight.get("command"))
            preflight_command_attempted = (
                worker_runtime_preflight.get("execution_performed") is True
                or bool(preflight_command)
            )
            _append_check(
                checks,
                blockers,
                check_id="worker_runtime:preflight_command_redacted",
                passed="raw_command" not in worker_runtime_preflight
                and worker_runtime_preflight.get("secret_values_in_artifact") is False
                and (
                    not preflight_command_attempted
                    or (
                        preflight_command.get("raw_command_stored") is False
                        and preflight_command.get("shell") is False
                        and worker_runtime_preflight.get(
                            "stdout_stderr_secret_redaction_enabled"
                        )
                        is True
                    )
                ),
                message="Worker runtime preflight artifacts store no raw commands or secrets",
                evidence={
                    "execution_performed": worker_runtime_preflight.get(
                        "execution_performed"
                    ),
                    "secret_values_in_artifact": worker_runtime_preflight.get(
                        "secret_values_in_artifact"
                    ),
                    "raw_command_present": "raw_command" in worker_runtime_preflight,
                    "stdout_stderr_secret_redaction_enabled": worker_runtime_preflight.get(
                        "stdout_stderr_secret_redaction_enabled"
                    ),
                    "command": preflight_command,
                },
            )
    if provider_launcher_result:
        _append_schema_check(
            checks,
            blockers,
            artifact_name="gpu_provider_launcher_result",
            payload=provider_launcher_result,
            expected_schema="robot_eval_gpu_provider_launcher_result.v1",
        )
        _append_check(
            checks,
            blockers,
            check_id="provider_launcher:no_secret_or_proof_upgrade",
            passed=provider_launcher_result.get("secret_values_in_artifact") is False
            and provider_launcher_result.get("simulator_execution_proven") is False
            and provider_launcher_result.get("robot_readiness_proven") is False
            and provider_launcher_result.get("public_claim_upgrade_allowed") is False
            and "raw_command" not in provider_launcher_result,
            message="Provider launcher result stores no secrets and cannot upgrade simulator, robot-readiness, or public proof",
            evidence={
                "status": provider_launcher_result.get("status"),
                "execution_performed": provider_launcher_result.get(
                    "execution_performed"
                ),
                "provider_launcher_command_executed": provider_launcher_result.get(
                    "provider_launcher_command_executed"
                ),
                "secret_values_in_artifact": provider_launcher_result.get(
                    "secret_values_in_artifact"
                ),
                "simulator_execution_proven": provider_launcher_result.get(
                    "simulator_execution_proven"
                ),
                "robot_readiness_proven": provider_launcher_result.get(
                    "robot_readiness_proven"
                ),
                "public_claim_upgrade_allowed": provider_launcher_result.get(
                    "public_claim_upgrade_allowed"
                ),
                "raw_command_present": "raw_command" in provider_launcher_result,
            },
        )
        command = _mapping(provider_launcher_result.get("command"))
        command_executed = provider_launcher_result.get(
            "provider_launcher_command_executed"
        ) is True
        _append_check(
            checks,
            blockers,
            check_id="provider_launcher:command_redacted_when_executed",
            passed=not command_executed
            or (
                command.get("raw_command_stored") is False
                and command.get("shell") is False
                and provider_launcher_result.get(
                    "stdout_stderr_secret_redaction_enabled"
                )
                is True
                and bool(_string(provider_launcher_result.get("stdout_path")))
                and bool(_string(provider_launcher_result.get("stderr_path")))
            ),
            message="Executed provider launcher commands are argv-only, redacted in artifacts/logs, and capture stdout/stderr paths",
            evidence={
                "provider_launcher_command_executed": command_executed,
                "command": command,
                "stdout_stderr_secret_redaction_enabled": provider_launcher_result.get(
                    "stdout_stderr_secret_redaction_enabled"
                ),
                "stdout_path": provider_launcher_result.get("stdout_path"),
                "stderr_path": provider_launcher_result.get("stderr_path"),
            },
        )
    if runpod_adapter_result:
        _append_schema_check(
            checks,
            blockers,
            artifact_name="runpod_provider_adapter_result",
            payload=runpod_adapter_result,
            expected_schema="runpod_provider_adapter_result.v1",
        )
        _append_check(
            checks,
            blockers,
            check_id="runpod_adapter:no_secret_or_proof_upgrade",
            passed=runpod_adapter_result.get("secret_values_in_artifact") is False
            and runpod_adapter_result.get("raw_api_key_stored") is False
            and runpod_adapter_result.get("simulator_execution_proven") is False
            and runpod_adapter_result.get("robot_readiness_proven") is False
            and runpod_adapter_result.get("public_claim_upgrade_allowed") is False,
            message="RunPod adapter result stores no API keys and cannot upgrade simulator, robot-readiness, or public proof",
            evidence={
                "status": runpod_adapter_result.get("status"),
                "mode": runpod_adapter_result.get("mode"),
                "api_call_performed": runpod_adapter_result.get("api_call_performed"),
                "runpod_side_effects_may_have_occurred": runpod_adapter_result.get(
                    "runpod_side_effects_may_have_occurred"
                ),
                "secret_values_in_artifact": runpod_adapter_result.get(
                    "secret_values_in_artifact"
                ),
                "raw_api_key_stored": runpod_adapter_result.get("raw_api_key_stored"),
                "simulator_execution_proven": runpod_adapter_result.get(
                    "simulator_execution_proven"
                ),
                "robot_readiness_proven": runpod_adapter_result.get(
                    "robot_readiness_proven"
                ),
                "public_claim_upgrade_allowed": runpod_adapter_result.get(
                    "public_claim_upgrade_allowed"
                ),
            },
        )
        cost_policy = _mapping(runpod_adapter_result.get("cost_control_policy"))
        serverless_controls = _mapping(cost_policy.get("serverless_endpoint_controls"))
        on_demand_controls = _mapping(cost_policy.get("on_demand_pod_controls"))
        policy_boundary = _mapping(cost_policy.get("proof_boundary"))
        endpoint_settings = {
            str(item) for item in _list(serverless_controls.get("endpoint_level_settings_required"))
        }
        adapter_hard_timeout = _number(cost_policy.get("hard_timeout_seconds")) or 0
        adapter_idle_timeout = _number(cost_policy.get("idle_timeout_seconds")) or 0
        adapter_watchdog_ttl = _number(cost_policy.get("external_watchdog_ttl_seconds")) or 0
        adapter_max_workers = _number(cost_policy.get("max_active_workers")) or 0
        _append_check(
            checks,
            blockers,
            check_id="runpod_adapter:cost_control_policy",
            passed=adapter_hard_timeout > 0
            and adapter_idle_timeout > 0
            and adapter_watchdog_ttl > adapter_hard_timeout
            and adapter_max_workers > 0
            and serverless_controls.get("idle_timeout_set_by_run_request") is False
            and serverless_controls.get("max_workers_set_by_run_request") is False
            and {"active_workers", "max_workers", "idle_timeout"}.issubset(endpoint_settings)
            and on_demand_controls.get("external_watchdog_or_owner_terminator_required")
            is True
            and policy_boundary.get("provider_allocation_proven") is False
            and policy_boundary.get("simulator_execution_proven") is False,
            message="RunPod adapter separates /run policy from endpoint/pod lifecycle controls and keeps provider/runtime proof bounded",
            evidence={
                "hard_timeout_seconds": cost_policy.get("hard_timeout_seconds"),
                "idle_timeout_seconds": cost_policy.get("idle_timeout_seconds"),
                "external_watchdog_ttl_seconds": cost_policy.get(
                    "external_watchdog_ttl_seconds"
                ),
                "max_active_workers": cost_policy.get("max_active_workers"),
                "endpoint_level_settings_required": sorted(endpoint_settings),
                "idle_timeout_set_by_run_request": serverless_controls.get(
                    "idle_timeout_set_by_run_request"
                ),
                "max_workers_set_by_run_request": serverless_controls.get(
                    "max_workers_set_by_run_request"
                ),
                "external_watchdog_or_owner_terminator_required": on_demand_controls.get(
                    "external_watchdog_or_owner_terminator_required"
                ),
                "provider_allocation_proven": policy_boundary.get(
                    "provider_allocation_proven"
                ),
                "simulator_execution_proven": policy_boundary.get(
                    "simulator_execution_proven"
                ),
            },
        )

    queueing = _mapping(scheduler.get("queueing"))
    _append_check(
        checks,
        blockers,
        check_id="queueing:async_web_request",
        passed=queueing.get("mode") == "async_job"
        and queueing.get("customer_response") == "job_id_and_status_only"
        and queueing.get("web_request_must_not_wait_for_simulator") is True,
        message="Customer request is represented as an async job, not a held web request",
        evidence={
            "mode": queueing.get("mode"),
            "customer_response": queueing.get("customer_response"),
            "web_request_must_not_wait_for_simulator": queueing.get(
                "web_request_must_not_wait_for_simulator"
            ),
        },
    )
    _append_check(
        checks,
        blockers,
        check_id="ownership:webapp_queues_pipeline_schedules",
        passed=scheduler.get("webapp_role") == "queue_and_forward_only"
        and scheduler.get("scheduler_owner") == "BlueprintCapturePipeline",
        message="WebApp queues/forwards only and Pipeline owns scheduling",
        evidence={
            "webapp_role": scheduler.get("webapp_role"),
            "scheduler_owner": scheduler.get("scheduler_owner"),
        },
    )

    cpu_gate = _mapping(scheduler.get("cpu_preflight_gate"))
    required_status = _mapping(cpu_gate.get("required_artifact_status"))
    present_artifacts = {
        key
        for key, item in required_status.items()
        if isinstance(item, Mapping) and item.get("present") is True
    }
    _append_check(
        checks,
        blockers,
        check_id="cpu_preflight:gate_before_gpu",
        passed=cpu_gate.get("required_before_gpu") is True
        and cpu_gate.get("blocks_gpu_when_missing") is True
        and REQUIRED_PREFLIGHT_ARTIFACTS.issubset(present_artifacts),
        message="CPU preflight gates GPU allocation and exposes required support artifacts",
        evidence={
            "required_before_gpu": cpu_gate.get("required_before_gpu"),
            "blocks_gpu_when_missing": cpu_gate.get("blocks_gpu_when_missing"),
            "present_artifacts": sorted(present_artifacts),
            "required_artifacts": sorted(REQUIRED_PREFLIGHT_ARTIFACTS),
        },
    )

    gpu_allocation = _mapping(scheduler.get("gpu_allocation"))
    _append_check(
        checks,
        blockers,
        check_id="gpu_allocation:webapp_cannot_approve_spend",
        passed=gpu_allocation.get("allocation_allowed_by_webapp") is False
        and gpu_allocation.get("gpu_spend_approved_by_webapp") is False,
        message="WebApp request does not approve GPU allocation or spend",
        evidence={
            "allocation_allowed_by_webapp": gpu_allocation.get(
                "allocation_allowed_by_webapp"
            ),
            "gpu_spend_approved_by_webapp": gpu_allocation.get(
                "gpu_spend_approved_by_webapp"
            ),
        },
    )

    artifact_contract = _mapping(scheduler.get("artifact_contract"))
    expected_outputs = set(str(item) for item in _list(artifact_contract.get("expected_outputs")))
    _append_check(
        checks,
        blockers,
        check_id="artifact_contract:startup_outputs_listed",
        passed=EXPECTED_OUTPUTS.issubset(expected_outputs)
        and artifact_contract.get("simulator_execution_proven_by_webapp") is False
        and artifact_contract.get("public_claim_upgrade_allowed") is False,
        message="Startup artifact contract lists required outputs without proof upgrades",
        evidence={
            "missing_outputs": sorted(EXPECTED_OUTPUTS.difference(expected_outputs)),
            "simulator_execution_proven_by_webapp": artifact_contract.get(
                "simulator_execution_proven_by_webapp"
            ),
            "public_claim_upgrade_allowed": artifact_contract.get(
                "public_claim_upgrade_allowed"
            ),
        },
    )

    launch_mode = _mapping(worker.get("launch_mode"))
    _append_check(
        checks,
        blockers,
        check_id="worker:scale_to_zero_with_idle_timeout",
        passed=launch_mode.get("mode") == "on_demand_with_optional_warm_pool"
        and launch_mode.get("scale_to_zero_default") is True
        and launch_mode.get("idle_shutdown_required") is True
        and (_number(launch_mode.get("idle_timeout_seconds")) or 0) > 0
        and (_number(launch_mode.get("hard_timeout_seconds")) or 0) > 0,
        message="Worker plan supports on-demand GPUs, optional warm pool, idle shutdown, and timeout",
        evidence={
            "mode": launch_mode.get("mode"),
            "scale_to_zero_default": launch_mode.get("scale_to_zero_default"),
            "idle_shutdown_required": launch_mode.get("idle_shutdown_required"),
            "idle_timeout_seconds": launch_mode.get("idle_timeout_seconds"),
            "hard_timeout_seconds": launch_mode.get("hard_timeout_seconds"),
        },
    )

    scheduler_selection = _mapping(scheduler.get("selection"))
    worker_image = _mapping(worker.get("worker_image"))
    cache_plan = _mapping(worker.get("cache_plan"))
    worker_simulator = _string(worker.get("simulator")) or _string(
        scheduler_selection.get("simulator")
    )
    _append_check(
        checks,
        blockers,
        check_id="worker:prebuilt_no_runtime_install_or_guessing",
        passed=worker_image.get("entrypoint") == "blueprint-run-robot-eval-worker"
        and worker_image.get("runtime_dependency_install_disallowed") is True
        and worker_image.get("runtime_asset_guessing_disallowed") is True
        and cache_plan.get("install_simulator_during_customer_job") is False
        and cache_plan.get("install_python_dependencies_during_customer_job") is False,
        message="Worker uses a prepared entrypoint and forbids runtime installs or asset guessing",
        evidence={
            "entrypoint": worker_image.get("entrypoint"),
            "runtime_dependency_install_disallowed": worker_image.get(
                "runtime_dependency_install_disallowed"
            ),
            "runtime_asset_guessing_disallowed": worker_image.get(
                "runtime_asset_guessing_disallowed"
            ),
            "install_simulator_during_customer_job": cache_plan.get(
                "install_simulator_during_customer_job"
            ),
            "install_python_dependencies_during_customer_job": cache_plan.get(
                "install_python_dependencies_during_customer_job"
            ),
        },
    )
    runtime_preflight = _mapping(worker.get("runtime_preflight_contract"))
    runtime_preflight_required = worker_simulator != "fixture"
    runtime_preflight_checks = set(
        str(item) for item in _list(runtime_preflight.get("required_checks"))
    )
    isaac_required_checks = {
        "nvidia_smi_gpu_inventory",
        "driver_version",
        "vulkan_device",
        "rtx_renderer_available",
        "isaac_headless_launch",
        "blank_scene_load",
        "test_frame_render",
    }
    _append_check(
        checks,
        blockers,
        check_id="worker:runtime_preflight_before_scene_load",
        passed=not runtime_preflight_required
        or (
            runtime_preflight.get("required_before_scene_load") is True
            and runtime_preflight.get("worker_blocks_scene_load_on_failed_preflight")
            is True
            and runtime_preflight.get("run_before")
            == "scene_load_and_policy_execution"
            and runtime_preflight.get("result_artifact")
            == "worker_runtime_preflight.json"
            and bool(runtime_preflight_checks)
            and runtime_preflight.get("runtime_preflight_is_not_simulator_proof")
            is True
        ),
        message="Worker runtime preflight is required before scene load and cannot upgrade simulator proof by itself",
        evidence={
            "simulator": worker_simulator,
            "required_before_scene_load": runtime_preflight.get(
                "required_before_scene_load"
            ),
            "worker_blocks_scene_load_on_failed_preflight": runtime_preflight.get(
                "worker_blocks_scene_load_on_failed_preflight"
            ),
            "run_before": runtime_preflight.get("run_before"),
            "result_artifact": runtime_preflight.get("result_artifact"),
            "required_checks": sorted(runtime_preflight_checks),
            "runtime_preflight_is_not_simulator_proof": runtime_preflight.get(
                "runtime_preflight_is_not_simulator_proof"
            ),
        },
    )
    _append_check(
        checks,
        blockers,
        check_id="worker:isaac_runtime_preflight_checks",
        passed=worker_simulator not in {"isaac_sim", "isaac_lab_arena"}
        or (
            isaac_required_checks.issubset(runtime_preflight_checks)
            and runtime_preflight.get("vulkan_required") is True
            and runtime_preflight.get("test_frame_render_required") is True
        ),
        message="Isaac worker startup requires NVIDIA inventory, driver, Vulkan/RTX, headless launch, blank scene, and test-frame preflight",
        evidence={
            "simulator": worker_simulator,
            "missing_checks": sorted(isaac_required_checks.difference(runtime_preflight_checks)),
            "vulkan_required": runtime_preflight.get("vulkan_required"),
            "test_frame_render_required": runtime_preflight.get(
                "test_frame_render_required"
            ),
        },
    )
    published_image_required = worker_image.get("published_image_ref_required") is True
    _append_check(
        checks,
        blockers,
        check_id="worker:published_image_ref_when_live_provider",
        passed=not published_image_required
        or (
            worker_image.get("configured_image_ref_present") is True
            and worker_image.get("configured_image_ref_is_versioned") is True
            and bool(_string(worker_image.get("configured_image_ref")))
        ),
        message="Live provider worker plans require a configured versioned image ref, not only a Dockerfile path",
        evidence={
            "published_image_ref_required": published_image_required,
            "image_ref_env_var": worker_image.get("image_ref_env_var"),
            "configured_image_ref_present": worker_image.get(
                "configured_image_ref_present"
            ),
            "configured_image_ref_is_versioned": worker_image.get(
                "configured_image_ref_is_versioned"
            ),
            "configured_image_ref": worker_image.get("configured_image_ref"),
        },
    )

    entrypoint_contract = _mapping(worker.get("worker_entrypoint_contract"))
    worker_manifest_job_request = worker_manifest.get("job_request")
    _append_check(
        checks,
        blockers,
        check_id="worker_manifest:strict_manifest_payload",
        passed=bool(_string(worker_manifest.get("capture_root")))
        and bool(_string(worker_manifest.get("job_id")))
        and bool(_string(worker_manifest.get("provisioner")))
        and bool(_string(worker_manifest.get("simulator")))
        and isinstance(worker_manifest_job_request, Mapping),
        message="Worker manifest is a strict queued payload with capture root, job id, provider, simulator, and embedded job request",
        evidence={
            "capture_root": worker_manifest.get("capture_root"),
            "job_id": worker_manifest.get("job_id"),
            "provisioner": worker_manifest.get("provisioner"),
            "simulator": worker_manifest.get("simulator"),
            "job_request_present": isinstance(worker_manifest_job_request, Mapping),
        },
    )
    worker_manifest_artifact_required = (
        worker_manifest.get("artifact_output_uri_required") is True
    )
    _append_check(
        checks,
        blockers,
        check_id="worker_manifest:artifact_output_when_required",
        passed=not worker_manifest_artifact_required
        or bool(_string(worker_manifest.get("artifact_output_uri"))),
        message="Worker manifest carries an artifact output URI whenever the provider finalizer requires one",
        evidence={
            "artifact_output_uri_required": worker_manifest_artifact_required,
            "artifact_output_uri": worker_manifest.get("artifact_output_uri"),
            "artifact_output_uri_env_var": worker_manifest.get("artifact_output_uri_env_var"),
            "blockers": worker_manifest.get("blockers"),
        },
    )
    worker_manifest_uri_required = (
        worker_manifest.get("worker_manifest_uri_required") is True
    )
    _append_check(
        checks,
        blockers,
        check_id="worker_manifest:fetch_uri_when_required",
        passed=not worker_manifest_uri_required
        or (
            bool(_string(worker_manifest.get("worker_manifest_uri")))
            and worker_manifest.get("worker_manifest_uri_fetchable_by_provider") is True
        ),
        message="Worker manifest records a provider-fetchable manifest URI whenever a remote worker is required",
        evidence={
            "worker_manifest_uri_required": worker_manifest_uri_required,
            "worker_manifest_uri": worker_manifest.get("worker_manifest_uri"),
            "worker_manifest_uri_env_var": worker_manifest.get(
                "worker_manifest_uri_env_var"
            ),
            "worker_manifest_uri_scheme": worker_manifest.get(
                "worker_manifest_uri_scheme"
            ),
            "worker_manifest_uri_fetchable_by_provider": worker_manifest.get(
                "worker_manifest_uri_fetchable_by_provider"
            ),
            "blockers": worker_manifest.get("blockers"),
        },
    )
    _append_check(
        checks,
        blockers,
        check_id="worker:strict_manifest_entrypoint",
        passed=entrypoint_contract.get("job_manifest_env") == "BLUEPRINT_EVAL_MANIFEST_URI"
        and entrypoint_contract.get("package_console_script") == "blueprint-run-robot-eval-worker"
        and entrypoint_contract.get("web_request_waits_for_worker") is False,
        message="Worker starts from a strict manifest command and never blocks the web request",
        evidence={
            "job_manifest_env": entrypoint_contract.get("job_manifest_env"),
            "package_console_script": entrypoint_contract.get("package_console_script"),
            "web_request_waits_for_worker": entrypoint_contract.get(
                "web_request_waits_for_worker"
            ),
        },
    )

    upload_contract = _mapping(worker.get("artifact_upload_contract"))
    upload_outputs = set(str(item) for item in _list(upload_contract.get("expected_outputs")))
    _append_check(
        checks,
        blockers,
        check_id="worker:artifact_upload_before_shutdown",
        passed=upload_contract.get("upload_before_shutdown_required") is True
        and EXPECTED_OUTPUTS.issubset(upload_outputs),
        message="Worker upload contract requires startup/result artifacts before shutdown",
        evidence={
            "upload_before_shutdown_required": upload_contract.get(
                "upload_before_shutdown_required"
            ),
            "missing_outputs": sorted(EXPECTED_OUTPUTS.difference(upload_outputs)),
        },
    )

    provider_shape = _mapping(provider.get("provider_request_shape"))
    provider_name = (
        _string(provider.get("provider"))
        or _string(provider_shape.get("provider_api"))
        or "fixture_local"
    )
    provider_environment = _mapping(provider_shape.get("environment"))
    provider_inputs = _mapping(provider_shape.get("inputs"))
    provider_limits = _mapping(provider_shape.get("limits"))
    provider_runtime_preflight = _mapping(provider_shape.get("runtime_preflight"))
    _append_check(
        checks,
        blockers,
        check_id="provider:dry_run_shape_no_secrets",
        passed=provider.get("live_provider_calls_performed") is False
        and provider_shape.get("api_payload_is_provider_adapter_template") is True
        and provider_environment.get("secret_values_in_artifact") is False,
        message="Provider launch request is a dry-run adapter shape and stores no secrets",
        evidence={
            "live_provider_calls_performed": provider.get("live_provider_calls_performed"),
            "api_payload_is_provider_adapter_template": provider_shape.get(
                "api_payload_is_provider_adapter_template"
            ),
            "secret_values_in_artifact": provider_environment.get(
                "secret_values_in_artifact"
            ),
            "secret_env_var_names": provider_environment.get("secret_env_var_names"),
        },
    )
    provider_image = _mapping(provider_shape.get("image"))
    provider_image_required = provider_image.get("owner_published_image_ref_required") is True
    _append_check(
        checks,
        blockers,
        check_id="provider:published_image_ref_when_live_provider",
        passed=not provider_image_required
        or (
            provider_image.get("configured_image_ref_present") is True
            and provider_image.get("configured_image_ref_is_versioned") is True
            and bool(_string(provider_image.get("configured_image_ref")))
        ),
        message="Provider launch request carries the exact versioned worker image ref for live GPU launchers",
        evidence={
            "owner_published_image_ref_required": provider_image_required,
            "image_ref_env_var": provider_image.get("image_ref_env_var"),
            "configured_image_ref_present": provider_image.get(
                "configured_image_ref_present"
            ),
            "configured_image_ref_is_versioned": provider_image.get(
                "configured_image_ref_is_versioned"
            ),
            "configured_image_ref": provider_image.get("configured_image_ref"),
        },
    )
    provider_manifest_required = (
        provider_inputs.get("manifest_uri_required_for_provider") is True
    )
    _append_check(
        checks,
        blockers,
        check_id="provider:worker_manifest_uri_when_required",
        passed=not provider_manifest_required
        or (
            provider_inputs.get("manifest_uri_configured") is True
            and provider_inputs.get("manifest_uri_fetchable_by_provider") is True
            and bool(_string(provider_inputs.get("manifest_uri")))
        ),
        message="Provider launch request carries a fetchable worker manifest URI when launching a remote worker",
        evidence={
            "manifest_uri_required_for_provider": provider_manifest_required,
            "manifest_uri_env_var": provider_inputs.get("manifest_uri_env_var"),
            "manifest_uri_configured": provider_inputs.get("manifest_uri_configured"),
            "manifest_uri_fetchable_by_provider": provider_inputs.get(
                "manifest_uri_fetchable_by_provider"
            ),
            "manifest_uri_scheme": provider_inputs.get("manifest_uri_scheme"),
            "manifest_uri": provider_inputs.get("manifest_uri"),
        },
    )
    provider_runtime_required = bool(
        provider_runtime_preflight.get("required_for_provider") is True
        or (provider_name != "fixture_local" and worker_simulator != "fixture")
    )
    _append_check(
        checks,
        blockers,
        check_id="provider:runtime_preflight_before_scene_load",
        passed=not provider_runtime_required
        or (
            provider_runtime_preflight.get("required_before_scene_load") is True
            and provider_runtime_preflight.get(
                "worker_blocks_scene_load_on_failed_preflight"
            )
            is True
            and provider_runtime_preflight.get("run_before")
            == "scene_load_and_policy_execution"
            and provider_runtime_preflight.get("result_artifact")
            == "worker_runtime_preflight.json"
            and bool(_list(provider_runtime_preflight.get("required_checks")))
            and provider_runtime_preflight.get(
                "runtime_preflight_is_not_simulator_proof"
            )
            is True
        ),
        message="Provider launch request carries a pre-scene runtime preflight contract for remote workers",
        evidence={
            "required_for_provider": provider_runtime_required,
            "required_before_scene_load": provider_runtime_preflight.get(
                "required_before_scene_load"
            ),
            "worker_blocks_scene_load_on_failed_preflight": provider_runtime_preflight.get(
                "worker_blocks_scene_load_on_failed_preflight"
            ),
            "run_before": provider_runtime_preflight.get("run_before"),
            "result_artifact": provider_runtime_preflight.get("result_artifact"),
            "required_checks": provider_runtime_preflight.get("required_checks"),
            "runtime_preflight_is_not_simulator_proof": provider_runtime_preflight.get(
                "runtime_preflight_is_not_simulator_proof"
            ),
        },
    )
    provider_command = _string(provider_shape.get("command"))
    provider_command_ok = provider_command == (
        "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
    ) or (
        provider_command.startswith("blueprint-run-robot-eval-worker ")
        and "--allow-simulator-execution" in provider_command
        and "--simulator-command" in provider_command
    )
    provider_artifact_contract_ok = provider_inputs.get("artifact_output_uri_required") is True or (
        provider_inputs.get("artifact_output_uri_required") is False
        and provider_inputs.get("runtime_manifest_signed_put_required") is True
    )
    _append_check(
        checks,
        blockers,
        check_id="provider:manifest_and_artifact_contract",
        passed=provider_command_ok
        and provider_inputs.get("manifest_uri_required") is True
        and provider_artifact_contract_ok
        and provider_limits.get("idle_shutdown_required") is True
        and (_number(provider_limits.get("hard_timeout_seconds")) or 0) > 0,
        message="Provider launch request includes manifest input, artifact output or signed runtime manifest, timeout, and idle shutdown",
        evidence={
            "command": provider_command,
            "command_ok": provider_command_ok,
            "manifest_uri_required": provider_inputs.get("manifest_uri_required"),
            "artifact_output_uri_required": provider_inputs.get("artifact_output_uri_required"),
            "runtime_manifest_signed_put_required": provider_inputs.get(
                "runtime_manifest_signed_put_required"
            ),
            "artifact_contract_ok": provider_artifact_contract_ok,
            "idle_shutdown_required": provider_limits.get("idle_shutdown_required"),
            "hard_timeout_seconds": provider_limits.get("hard_timeout_seconds"),
        },
    )
    provider_watchdog_required = provider_limits.get("external_watchdog_ttl_required") is True
    provider_hard_timeout = _number(provider_limits.get("hard_timeout_seconds")) or 0
    provider_watchdog_ttl = _number(
        provider_limits.get("external_watchdog_ttl_seconds")
    ) or 0
    _append_check(
        checks,
        blockers,
        check_id="provider:external_watchdog_ttl",
        passed=not provider_watchdog_required
        or (
            provider_watchdog_ttl > provider_hard_timeout
            and bool(_string(provider_limits.get("external_watchdog_owner")))
        ),
        message="Provider launch request has a concrete watchdog TTL longer than the hard timeout",
        evidence={
            "external_watchdog_ttl_required": provider_watchdog_required,
            "hard_timeout_seconds": provider_hard_timeout,
            "external_watchdog_ttl_seconds": provider_watchdog_ttl,
            "external_watchdog_owner": provider_limits.get("external_watchdog_owner"),
        },
    )

    ledger_budget = _mapping(ledger.get("budget"))
    ledger_limits = _mapping(ledger.get("worker_limits"))
    gpu_time = _mapping(ledger.get("gpu_time"))
    _append_check(
        checks,
        blockers,
        check_id="cost:budget_timeout_worker_limits",
        passed=ledger_budget.get("gpu_spend_approved_by_webapp") is False
        and ledger_budget.get("allocation_allowed_by_webapp") is False
        and ledger_limits.get("customer_concurrency_limit_required") is True
        and ledger_limits.get("idle_shutdown_required") is True
        and (_number(ledger_limits.get("hard_timeout_seconds")) or 0) > 0
        and (_number(ledger_limits.get("max_billable_gpu_seconds")) or 0) > 0
        and (
            ledger_limits.get("external_watchdog_ttl_required") is not True
            or (
                (_number(ledger_limits.get("external_watchdog_ttl_seconds")) or 0)
                > (_number(ledger_limits.get("hard_timeout_seconds")) or 0)
            )
        ),
        message="Cost ledger records spend denial by WebApp plus timeout, concurrency, and idle-shutdown controls",
        evidence={
            "gpu_spend_approved_by_webapp": ledger_budget.get(
                "gpu_spend_approved_by_webapp"
            ),
            "allocation_allowed_by_webapp": ledger_budget.get("allocation_allowed_by_webapp"),
            "customer_concurrency_limit_required": ledger_limits.get(
                "customer_concurrency_limit_required"
            ),
            "idle_shutdown_required": ledger_limits.get("idle_shutdown_required"),
            "hard_timeout_seconds": ledger_limits.get("hard_timeout_seconds"),
            "max_billable_gpu_seconds": ledger_limits.get("max_billable_gpu_seconds"),
            "external_watchdog_ttl_required": ledger_limits.get(
                "external_watchdog_ttl_required"
            ),
            "external_watchdog_ttl_seconds": ledger_limits.get(
                "external_watchdog_ttl_seconds"
            ),
        },
    )
    estimated_gpu_seconds = _number(gpu_time.get("estimated_gpu_seconds")) or 0
    gpu_time_pending_until_provider_runtime = (
        ledger.get("status") == "ready_for_explicit_provider_launcher"
        and ledger.get("live_provider_calls_performed") is False
        and gpu_time.get("actual_gpu_time_source") == "not_observed"
        and estimated_gpu_seconds > 0
    )
    _append_check(
        checks,
        blockers,
        check_id="cost:gpu_time_recorded_or_blocked",
        passed=gpu_time.get("actual_gpu_time_record_required") is True
        and (
            gpu_time.get("actual_gpu_time_record_present") is True
            or gpu_time_pending_until_provider_runtime
        ),
        message="Cost ledger records actual GPU time, no-GPU/blocked zero time, or a pending provider-runtime GPU-time requirement",
        evidence={
            "estimated_gpu_seconds": gpu_time.get("estimated_gpu_seconds"),
            "actual_gpu_seconds": gpu_time.get("actual_gpu_seconds"),
            "actual_gpu_time_source": gpu_time.get("actual_gpu_time_source"),
            "actual_gpu_time_record_required": gpu_time.get(
                "actual_gpu_time_record_required"
            ),
            "actual_gpu_time_record_present": gpu_time.get(
                "actual_gpu_time_record_present"
            ),
            "gpu_time_pending_until_provider_runtime": (
                gpu_time_pending_until_provider_runtime
            ),
            "ledger_status": ledger.get("status"),
            "live_provider_calls_performed": ledger.get("live_provider_calls_performed"),
        },
    )

    claim_boundary = _mapping(run_manifest.get("claim_boundary"))
    proof_boundary = _mapping(run_manifest.get("proof_boundary"))
    if not claim_boundary and proof_boundary:
        claim_boundary = proof_boundary
    simulator_execution_earned = (
        _bool(claim_boundary, ("simulator_execution_proven",)) is True
        and _string(run_manifest.get("status")) == "simulator_command_completed"
        and _string(run_manifest.get("simulator_service_status")) == "completed"
    )
    _append_check(
        checks,
        blockers,
        check_id="proof:no_unearned_claim_upgrades",
        passed=(
            _bool(claim_boundary, ("simulator_execution_proven",)) is False
            or simulator_execution_earned
        )
        and _bool(claim_boundary, ("robot_readiness_proven",)) is False
        and _bool(claim_boundary, ("public_claim_upgrade_allowed",)) is False,
        message="Startup path does not upgrade unearned simulator, robot readiness, or public claims",
        evidence={
            "simulator_execution_proven": claim_boundary.get("simulator_execution_proven"),
            "simulator_execution_earned": simulator_execution_earned,
            "job_status": run_manifest.get("status"),
            "simulator_service_status": run_manifest.get("simulator_service_status"),
            "robot_readiness_proven": claim_boundary.get("robot_readiness_proven"),
            "public_claim_upgrade_allowed": claim_boundary.get(
                "public_claim_upgrade_allowed"
            ),
        },
    )

    simulator = _string(scheduler.get("selection", {}).get("simulator") if isinstance(scheduler.get("selection"), Mapping) else "")
    if simulator in {"isaac_sim", "isaac_lab_arena"}:
        provider_gpu = _mapping(provider_shape.get("gpu"))
        disallowed = {str(item).lower() for item in _list(provider_gpu.get("disallowed_gpu_classes"))}
        _append_check(
            checks,
            blockers,
            check_id="gpu_selection:isaac_requires_rt_class",
            passed=bool(disallowed.intersection({"a100", "h100"})),
            message="Isaac startup request explicitly disallows A100/H100-style non-RT render targets",
            evidence={
                "simulator": simulator,
                "preferred_gpu_class": provider_gpu.get("preferred_gpu_class"),
                "disallowed_gpu_classes": provider_gpu.get("disallowed_gpu_classes"),
            },
        )
    elif not simulator:
        warnings.append("scheduler_selection_missing_simulator")

    return checks, blockers, warnings


def build_robot_eval_startup_architecture_audit(
    *,
    job_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_job_dir = Path(job_dir).resolve()
    artifacts, artifact_issues = _read_artifacts(resolved_job_dir)
    checks, blockers, warnings = _startup_artifact_checks(
        artifacts=artifacts,
        artifact_issues=artifact_issues,
    )

    run_manifest = _mapping(artifacts.get("job_run_manifest"))
    job_request = _mapping(artifacts.get("job_request"))
    scheduler = _mapping(artifacts.get("scheduler_decision"))
    ledger = _mapping(artifacts.get("gpu_cost_control_ledger"))
    claim_boundary = _mapping(run_manifest.get("claim_boundary"))
    run_proof_boundary = _mapping(run_manifest.get("proof_boundary"))
    if not claim_boundary and run_proof_boundary:
        claim_boundary = run_proof_boundary
    simulator_execution_earned = (
        _bool(claim_boundary, ("simulator_execution_proven",)) is True
        and _string(run_manifest.get("status")) == "simulator_command_completed"
        and _string(run_manifest.get("simulator_service_status")) == "completed"
    )
    job_id = (
        _string(run_manifest.get("job_id"))
        or _string(job_request.get("job_id"))
        or resolved_job_dir.name
    )
    output = (
        Path(output_path).resolve()
        if output_path
        else resolved_job_dir / "startup_architecture_audit.json"
    )
    artifact_paths = {
        name: str(resolved_job_dir / relative_path)
        for name, relative_path in {
            **EXPECTED_JOB_ARTIFACTS,
            **OPTIONAL_JOB_ARTIFACTS,
        }.items()
    }
    result = {
        "schema_version": ROBOT_EVAL_STARTUP_ARCHITECTURE_AUDIT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "architecture_compliant": not blockers,
        "job_id": job_id,
        "job_dir": str(resolved_job_dir),
        "job_status": run_manifest.get("status"),
        "scheduler_decision_status": scheduler.get("status"),
        "gpu_cost_control_ledger_status": ledger.get("status"),
        "artifact_paths": artifact_paths,
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": sum(1 for item in checks if item.get("status") == "passed"),
        "blocked_check_count": len(blockers),
        "blockers": blockers,
        "warnings": warnings,
        "proof_boundary": {
            "read_only_audit": True,
            "live_provider_calls_performed_by_audit": False,
            "simulator_execution_performed_by_audit": False,
            "startup_architecture_compliant": not blockers,
            "simulator_execution_proven": simulator_execution_earned and not blockers,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
        "output_path": str(output),
    }
    ensure_dir(output.parent)
    write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit an existing robot-eval job directory against startup architecture rules."
    )
    parser.add_argument("--job-dir", help="Path to pipeline/robot_eval_jobs/<job_id>")
    parser.add_argument("--capture-root", help="Capture root containing pipeline/robot_eval_jobs")
    parser.add_argument("--job-id", help="Robot-eval job id under --capture-root")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)

    if args.job_dir:
        job_dir = Path(args.job_dir)
    elif args.capture_root and args.job_id:
        job_dir = Path(args.capture_root) / "pipeline" / "robot_eval_jobs" / args.job_id
    else:
        parser.error("Provide either --job-dir or both --capture-root and --job-id")

    result = build_robot_eval_startup_architecture_audit(
        job_dir=job_dir,
        output_path=args.output_path,
    )
    print(f"[robot-eval-startup-audit] audit={result['output_path']}")
    print(f"[robot-eval-startup-audit] status={result['status']}")
    print(f"[robot-eval-startup-audit] job_id={result['job_id']}")
    if result["blockers"]:
        print("[robot-eval-startup-audit] blockers=" + ",".join(result["blockers"]))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
