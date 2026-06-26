"""Read-only audit for optional GPU/provider closure evidence.

The audit inspects local artifacts only. It never provisions GPUs, reads raw
secret values, or calls provider APIs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


PROVIDER_CLOSURE_AUDIT_SCHEMA_VERSION = "provider_closure_audit_report.v1"
PROVIDER_CLOSURE_AUDIT_REPORT_NAME = "provider_closure_audit_report.json"

RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
RUNPOD_API_KEY_FILE_ENV = "RUNPOD_API_KEY_FILE"
RUNPOD_CONFIG_FILE_ENV = "RUNPOD_CONFIG_FILE"
DEFAULT_RUNPOD_CONFIG_FILE = "~/.runpod/config.toml"
VAST_API_KEY_FILE_ENV = "VAST_API_KEY_FILE"
DEFAULT_VAST_API_KEY_FILE = "~/.blueprint-secrets/vast_api_key"

FINALIZER_ARTIFACT_NAMES = (
    "provider_runtime_finalizer_proof.json",
)
SPEND_LEDGER_ARTIFACT_NAMES = (
    "sim_only_provider_cost_ledger.json",
    "gpu_cost_control_ledger.json",
    "wam_provider_cost_control_ledger.json",
    "isaac_gpu_cost_control_ledger.json",
    "vast_budget_ledger.json",
    "vast_session_cost_summary.json",
    "runpod_runtime_cost_teardown_summary.json",
)
WATCHDOG_ARTIFACT_NAMES = (
    "gpu_provider_launch_request.json",
    "runpod_provider_readiness_manifest.json",
    "sim_only_provider_cost_ledger.json",
    "vast_provider_plan.json",
    "vast_session_budget_guard.json",
    "vast_final_validation.json",
)
ARTIFACT_OUTPUT_CLOSURE_NAMES = (
    "provider_runtime_finalizer_proof.json",
    "wam_provider_artifact_upload_proof.json",
    "vast_provider_command_result.json",
)
TEARDOWN_ARTIFACT_NAMES = (
    "provider_shutdown_proof.json",
    "runpod_live_execution_proof.json",
    "runpod_teardown.json",
    "runpod_post_teardown_status.json",
    "runpod_runtime_cost_teardown_summary.json",
    "vast_teardown_manifest.json",
    "vast_final_validation.json",
)
PROVIDER_RESULT_ARTIFACT_NAMES = (
    "runpod_provider_adapter_result.json",
    "vast_provider_adapter_result.json",
    "vast_provider_command_result.json",
    "vast_final_validation.json",
)
REMOTE_EXECUTION_CLOSURE_ARTIFACT_NAMES = (
    "remote_cloud_execution_closure_manifest.json",
)
PROVIDER_READINESS_ARTIFACT_NAMES = (
    "runpod_provider_readiness_manifest.json",
)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_json_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"{type(exc).__name__}:{str(exc)[:160]}"
    if not isinstance(payload, Mapping):
        return {}, f"expected_json_object:{type(payload).__name__}"
    return dict(payload), None


def _path_status(path: Path) -> dict[str, Any]:
    try:
        is_file = path.is_file()
        size = path.stat().st_size if is_file else 0
    except OSError as exc:
        return {
            "path": str(path),
            "present": False,
            "nonempty": False,
            "stat_error": type(exc).__name__,
        }
    return {
        "path": str(path),
        "present": is_file,
        "nonempty": size > 0,
    }


def _find_named_artifacts(job_dir: Path, names: Sequence[str]) -> list[Path]:
    if not job_dir.exists():
        return []
    found: dict[str, Path] = {}
    for name in names:
        root_candidate = job_dir / name
        if root_candidate.is_file():
            found[str(root_candidate.resolve())] = root_candidate
        for candidate in job_dir.rglob(name):
            if candidate.is_file():
                found[str(candidate.resolve())] = candidate
    return sorted(found.values(), key=lambda path: _relative(path, job_dir))


def _find_teardown_artifacts(job_dir: Path) -> list[Path]:
    found: dict[str, Path] = {
        str(path.resolve()): path
        for path in _find_named_artifacts(job_dir, TEARDOWN_ARTIFACT_NAMES)
    }
    if job_dir.exists():
        for candidate in job_dir.rglob("*teardown*.json"):
            if candidate.is_file():
                found[str(candidate.resolve())] = candidate
    return sorted(found.values(), key=lambda path: _relative(path, job_dir))


def _artifact_summary(path: Path, job_dir: Path) -> dict[str, Any]:
    payload, error = _read_json_mapping(path)
    summary = {
        "path": str(path),
        "relative_path": _relative(path, job_dir),
        "present": path.is_file(),
        "parse_error": error,
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status") or payload.get("teardown_status"),
    }
    if payload.get("provider"):
        summary["provider"] = payload.get("provider")
    if payload.get("blockers"):
        summary["blockers"] = _string_list(payload.get("blockers"))
    return summary


def _load_payloads(paths: Sequence[Path]) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        payload, error = _read_json_mapping(path)
        if error is None:
            payloads.append((path, payload))
    return payloads


def _read_phase_names(job_dir: Path) -> tuple[set[str], list[str]]:
    phase_names: set[str] = set()
    errors: list[str] = []
    for phase_path in _find_named_artifacts(job_dir, ("vast_runtime_phase_log.jsonl",)):
        for line_number, line in enumerate(
            phase_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                errors.append(
                    f"{_relative(phase_path, job_dir)}:"
                    f"{line_number}:{type(exc).__name__}"
                )
                continue
            if isinstance(payload, Mapping) and _string(payload.get("phase")):
                phase_names.add(_string(payload.get("phase")))
    return phase_names, errors


def _infer_provider(
    *,
    explicit_provider: str | None,
    payloads: Sequence[tuple[Path, Mapping[str, Any]]],
    job_dir: Path,
) -> str:
    if _string(explicit_provider):
        return _string(explicit_provider)
    for path, payload in payloads:
        provider = _string(payload.get("provider"))
        if provider:
            return provider
        relative = _relative(path, job_dir)
        if relative.startswith("vast_") or "/vast_" in relative:
            return "vast"
        if relative.startswith("runpod_") or "/runpod_" in relative:
            return "runpod"
    return "provider_unknown"


def _runpod_credentials_configured() -> dict[str, Any]:
    api_key_env_present = bool(_string(os.getenv(RUNPOD_API_KEY_ENV)))
    key_file_env = _string(os.getenv(RUNPOD_API_KEY_FILE_ENV))
    config_file_env = _string(os.getenv(RUNPOD_CONFIG_FILE_ENV))
    key_file_status = (
        _path_status(Path(key_file_env).expanduser())
        if key_file_env
        else {"path": None, "present": False, "nonempty": False}
    )
    config_file_status = _path_status(
        Path(config_file_env or DEFAULT_RUNPOD_CONFIG_FILE).expanduser()
    )
    configured = bool(
        api_key_env_present
        or key_file_status.get("nonempty")
        or config_file_status.get("present")
    )
    return {
        "provider": "runpod",
        "credential_configured": configured,
        "env_value_present": api_key_env_present,
        "api_key_file": key_file_status,
        "config_file": config_file_status,
        "raw_secret_values_read": False,
        "raw_secret_values_recorded": False,
    }


def _vast_credentials_configured() -> dict[str, Any]:
    key_file_env = _string(os.getenv(VAST_API_KEY_FILE_ENV))
    key_file_status = _path_status(
        Path(key_file_env or DEFAULT_VAST_API_KEY_FILE).expanduser()
    )
    return {
        "provider": "vast",
        "credential_configured": bool(key_file_status.get("nonempty")),
        "api_key_file": key_file_status,
        "raw_secret_values_read": False,
        "raw_secret_values_recorded": False,
    }


def _credential_audit(provider: str) -> dict[str, Any]:
    if provider == "runpod":
        return _runpod_credentials_configured()
    if provider == "vast":
        return _vast_credentials_configured()
    runpod = _runpod_credentials_configured()
    vast = _vast_credentials_configured()
    return {
        "provider": provider,
        "credential_configured": bool(
            runpod.get("credential_configured") or vast.get("credential_configured")
        ),
        "runpod": runpod,
        "vast": vast,
        "raw_secret_values_read": False,
        "raw_secret_values_recorded": False,
    }


def _check_result(
    *,
    check_id: str,
    passed: bool,
    evidence: Sequence[dict[str, Any]],
    blockers: Sequence[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "evidence": list(evidence),
        "blockers": _dedupe(blockers),
        "details": dict(details or {}),
    }


def _watchdog_check(
    *,
    job_dir: Path,
    paths: Sequence[Path],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path, payload in _load_payloads(paths):
        relative = _relative(path, job_dir)
        provider_shape = _mapping(payload.get("provider_request_shape"))
        limits = _mapping(provider_shape.get("limits") or payload.get("limits"))
        hard_timeout = _number(limits.get("hard_timeout_seconds"))
        watchdog_timeout = _number(
            limits.get("external_watchdog_ttl_seconds")
            or payload.get("watchdog_timeout_seconds")
        )
        watchdog_owner = _string(limits.get("external_watchdog_owner"))
        if watchdog_timeout and (hard_timeout is None or watchdog_timeout > hard_timeout):
            evidence.append(
                {
                    "path": relative,
                    "source": "provider_limits",
                    "hard_timeout_seconds": hard_timeout,
                    "external_watchdog_ttl_seconds": watchdog_timeout,
                    "external_watchdog_owner_present": bool(watchdog_owner),
                }
            )
        readiness = _mapping(payload.get("watchdog_and_teardown"))
        if (
            readiness.get("external_watchdog_ttl_exceeds_hard_timeout") is True
            and readiness.get("provider_shutdown_evidence_required_after_live_attempt") is True
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "runpod_provider_readiness_manifest",
                    "external_watchdog_owner_present": bool(
                        _string(readiness.get("external_watchdog_owner"))
                    ),
                }
            )
        if payload.get("watchdog_timeout_seconds"):
            watchdog = _number(payload.get("watchdog_timeout_seconds"))
            hard = _number(payload.get("hard_timeout_seconds"))
            if watchdog and (hard is None or watchdog > hard):
                evidence.append(
                    {
                        "path": relative,
                        "source": "sim_only_provider_cost_ledger",
                        "hard_timeout_seconds": hard,
                        "watchdog_timeout_seconds": watchdog,
                    }
                )
        budget = _mapping(payload.get("budget"))
        if _number(budget.get("hard_cap_usd")) is not None and _number(
            budget.get("max_live_minutes")
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "vast_provider_plan_budget_guard",
                    "hard_cap_usd": budget.get("hard_cap_usd"),
                    "max_live_minutes": budget.get("max_live_minutes"),
                }
            )
    phase_names, phase_errors = _read_phase_names(job_dir)
    if {"vast_instance_teardown_started", "vast_instance_teardown_completed"} <= phase_names:
        evidence.append(
            {
                "path": "vast_runtime_phase_log.jsonl",
                "source": "vast_phase_watchdog_teardown_path",
                "teardown_phases_present": True,
            }
        )
    if phase_errors:
        blockers.append("vast_runtime_phase_log_parse_failed")
    if not evidence:
        blockers.append("provider_watchdog_evidence_missing")
    return _check_result(
        check_id="watchdog",
        passed=bool(evidence) and not phase_errors,
        evidence=evidence,
        blockers=blockers,
    )


def _spend_ledger_check(*, job_dir: Path, paths: Sequence[Path]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path, payload in _load_payloads(paths):
        relative = _relative(path, job_dir)
        if payload.get("continuing_spend_from_this_run") is True:
            blockers.append("continuing_provider_spend_detected")
        has_budget = any(
            _number(payload.get(key)) is not None
            for key in (
                "max_budget_per_job_usd",
                "estimated_cost_usd",
                "budget_cap_usd",
                "hard_cap_usd",
                "spend_hard_cap_usd",
            )
        )
        attempts = payload.get("attempts")
        if isinstance(attempts, list) and attempts:
            has_budget = True
        if has_budget or "cost" in path.name or "ledger" in path.name:
            evidence.append(
                {
                    "path": relative,
                    "schema_version": payload.get("schema_version"),
                    "status": payload.get("status") or payload.get("teardown_status"),
                    "estimated_cost_usd": payload.get("estimated_cost_usd"),
                    "continuing_spend_from_this_run": payload.get(
                        "continuing_spend_from_this_run"
                    ),
                }
            )
    if not evidence:
        blockers.append("provider_spend_ledger_missing")
    return _check_result(
        check_id="spend_ledger",
        passed=bool(evidence) and "continuing_provider_spend_detected" not in blockers,
        evidence=evidence,
        blockers=blockers,
    )


def _artifact_output_closure_check(*, job_dir: Path, paths: Sequence[Path]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path, payload in _load_payloads(paths):
        relative = _relative(path, job_dir)
        finalizer_uploaded = bool(
            payload.get("artifact_upload_completed_before_shutdown") is True
            or payload.get("worker_artifacts_finalized_before_shutdown") is True
            or payload.get("finalizer_refresh_upload_completed_before_shutdown") is True
            or payload.get("worker_runtime_manifest_upload_completed_before_shutdown") is True
        )
        if payload.get("status") == "completed" and finalizer_uploaded:
            evidence.append(
                {
                    "path": relative,
                    "source": "provider_runtime_finalizer_proof",
                    "artifact_upload_completed_before_shutdown": payload.get(
                        "artifact_upload_completed_before_shutdown"
                    ),
                    "worker_artifacts_finalized_before_shutdown": payload.get(
                        "worker_artifacts_finalized_before_shutdown"
                    ),
                }
            )
        if (
            path.name == "wam_provider_artifact_upload_proof.json"
            and payload.get("status") == "completed"
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "wam_provider_artifact_upload_proof",
                    "uploaded_file_count": payload.get("uploaded_file_count"),
                }
            )
        if (
            path.name == "vast_provider_command_result.json"
            and payload.get("provider_output_upload_ok") is True
            and (
                payload.get("provider_runtime_output_zip_received") is True
                or payload.get("provider_runtime_output_zip_produced") is True
            )
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "vast_provider_command_result",
                    "provider_output_upload_ok": True,
                    "provider_runtime_output_zip_received": payload.get(
                        "provider_runtime_output_zip_received"
                    ),
                }
            )
        if _string_list(payload.get("blockers")):
            blockers.append("provider_artifact_output_evidence_has_blockers")
    if not evidence:
        blockers.append("provider_artifact_output_closure_missing")
    return _check_result(
        check_id="artifact_output_closure",
        passed=bool(evidence),
        evidence=evidence,
        blockers=blockers,
    )


def _teardown_check(*, job_dir: Path, paths: Sequence[Path]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path, payload in _load_payloads(paths):
        relative = _relative(path, job_dir)
        if payload.get("continuing_spend_from_this_run") is True:
            blockers.append("continuing_provider_spend_detected")
        if payload.get("provider_shutdown_proven") is True or payload.get(
            "clean_shutdown_proven"
        ) is True:
            evidence.append(
                {
                    "path": relative,
                    "source": "provider_runtime_finalizer_proof",
                    "provider_shutdown_proven": payload.get("provider_shutdown_proven"),
                    "clean_shutdown_proven": payload.get("clean_shutdown_proven"),
                }
            )
        shutdown_evidence = _mapping(payload.get("provider_shutdown_evidence"))
        if shutdown_evidence.get("zero_active_workers_after_run") is True:
            evidence.append(
                {
                    "path": relative,
                    "source": "provider_shutdown_evidence",
                    "zero_active_workers_after_run": True,
                }
            )
        if payload.get("shutdown_or_termination_proof") is True and (
            payload.get("active_pod_count_after") in {0, "0", None}
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "runpod_live_execution_proof",
                    "shutdown_or_termination_proof": True,
                    "active_pod_count_after": payload.get("active_pod_count_after"),
                }
            )
        teardown_status = _string(payload.get("teardown_status")).lower()
        if any(token in teardown_status for token in ("terminated", "deleted", "no_pod", "zero")):
            evidence.append(
                {
                    "path": relative,
                    "source": "runpod_runtime_cost_teardown_summary",
                    "teardown_status": payload.get("teardown_status"),
                }
            )
        if (
            path.name == "vast_teardown_manifest.json"
            and payload.get("status") == "completed"
            and payload.get("continuing_spend_from_this_run") is False
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "vast_teardown_manifest",
                    "runner_gpu_teardown_completed": payload.get(
                        "runner_gpu_teardown_completed"
                    ),
                    "continuing_spend_from_this_run": False,
                }
            )
        if (
            path.name == "vast_final_validation.json"
            and payload.get("all_vast_instances_destroyed_by_adapter") is True
            and payload.get("continuing_spend_from_this_run") is False
            and payload.get("vast_instance_ids")
        ):
            evidence.append(
                {
                    "path": relative,
                    "source": "vast_final_validation",
                    "all_vast_instances_destroyed_by_adapter": True,
                }
            )
    if not evidence:
        blockers.append("provider_teardown_evidence_missing")
    return _check_result(
        check_id="teardown",
        passed=bool(evidence) and "continuing_provider_spend_detected" not in blockers,
        evidence=evidence,
        blockers=blockers,
    )


def _latest_payload_summary(
    *,
    job_dir: Path,
    paths: Sequence[Path],
    artifact_kind: str,
) -> dict[str, Any]:
    if not paths:
        return {
            "status": "not_available",
            "artifact_kind": artifact_kind,
            "path": None,
            "present": False,
        }
    path = sorted(paths, key=lambda item: item.stat().st_mtime if item.exists() else 0)[-1]
    payload, error = _read_json_mapping(path)
    if error:
        return {
            "status": "parse_error",
            "artifact_kind": artifact_kind,
            "path": str(path),
            "relative_path": _relative(path, job_dir),
            "present": True,
            "parse_error": error,
        }
    return {
        "status": _string(payload.get("status")) or "present",
        "artifact_kind": artifact_kind,
        "path": str(path),
        "relative_path": _relative(path, job_dir),
        "present": True,
        "payload": payload,
    }


def _remote_execution_summary(*, job_dir: Path, paths: Sequence[Path]) -> dict[str, Any]:
    summary = _latest_payload_summary(
        job_dir=job_dir,
        paths=paths,
        artifact_kind="remote_cloud_execution_closure",
    )
    payload = _mapping(summary.pop("payload", {}))
    if not payload:
        return summary
    return {
        **summary,
        "provider": payload.get("provider"),
        "simulator": payload.get("simulator"),
        "remote_cloud_execution_proven": bool(payload.get("remote_cloud_execution_proven")),
        "clean_shutdown_proven": bool(payload.get("clean_shutdown_proven")),
        "live_provider_calls_performed": bool(payload.get("live_provider_calls_performed")),
        "phase": payload.get("phase"),
        "provider_job_id": payload.get("provider_job_id"),
        "pod_id": payload.get("pod_id"),
        "provider_runtime_started_at": payload.get("provider_runtime_started_at"),
        "max_wait_seconds": payload.get("max_wait_seconds"),
        "watchdog_boundary": _mapping(payload.get("watchdog_boundary")),
        "max_spend_usd": payload.get("max_spend_usd"),
        "output_uri": payload.get("output_uri"),
        "output_zip_or_object_size_bytes": payload.get("output_zip_or_object_size_bytes"),
        "artifact_manifest": _mapping(payload.get("artifact_manifest")),
        "teardown_status": payload.get("teardown_status"),
        "continuing_spend_from_this_run": bool(
            payload.get("continuing_spend_from_this_run")
        ),
        "contract_blockers": _string_list(payload.get("contract_blockers")),
        "runtime_blockers": _string_list(payload.get("runtime_blockers")),
        "checks": _mapping(payload.get("checks")),
    }


def _provider_readiness_summary(*, job_dir: Path, paths: Sequence[Path]) -> dict[str, Any]:
    summary = _latest_payload_summary(
        job_dir=job_dir,
        paths=paths,
        artifact_kind="provider_readiness",
    )
    payload = _mapping(summary.pop("payload", {}))
    if not payload:
        return summary
    return {
        **summary,
        "provider": payload.get("provider"),
        "mode": payload.get("mode"),
        "api_call_performed": bool(payload.get("api_call_performed")),
        "live_provider_call_authorized": bool(payload.get("live_provider_call_authorized")),
        "blockers": _string_list(payload.get("blockers")),
        "spend_limits": _mapping(payload.get("spend_limits")),
        "artifact_output": _mapping(payload.get("artifact_output")),
        "watchdog_and_teardown": _mapping(payload.get("watchdog_and_teardown")),
        "no_secret_artifact_policy": _mapping(payload.get("no_secret_artifact_policy")),
        "claim_boundary": _mapping(payload.get("claim_boundary")),
    }


def _job_dir(
    capture_root: str | Path | None,
    job_id: str | None,
    job_dir: str | Path | None,
) -> Path:
    if job_dir:
        return Path(job_dir).expanduser().resolve()
    if not capture_root or not job_id:
        raise ValueError("job_dir or both capture_root and job_id are required")
    return (
        Path(capture_root).expanduser().resolve()
        / "pipeline"
        / "robot_eval_jobs"
        / job_id
    )


def audit_provider_closure(
    *,
    job_dir: str | Path | None = None,
    capture_root: str | Path | None = None,
    job_id: str | None = None,
    provider: str | None = None,
    output_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write and return a read-only provider closure audit report."""

    resolved_job_dir = _job_dir(capture_root, job_id, job_dir)
    resolved_generated_at = generated_at or utc_now_iso()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / PROVIDER_CLOSURE_AUDIT_REPORT_NAME
    )
    ensure_dir(resolved_output.parent)

    finalizer_paths = _find_named_artifacts(resolved_job_dir, FINALIZER_ARTIFACT_NAMES)
    spend_paths = _find_named_artifacts(resolved_job_dir, SPEND_LEDGER_ARTIFACT_NAMES)
    watchdog_paths = _find_named_artifacts(resolved_job_dir, WATCHDOG_ARTIFACT_NAMES)
    artifact_output_paths = _find_named_artifacts(
        resolved_job_dir,
        ARTIFACT_OUTPUT_CLOSURE_NAMES,
    )
    teardown_paths = _find_teardown_artifacts(resolved_job_dir)
    provider_result_paths = _find_named_artifacts(
        resolved_job_dir,
        PROVIDER_RESULT_ARTIFACT_NAMES,
    )
    remote_execution_paths = _find_named_artifacts(
        resolved_job_dir,
        REMOTE_EXECUTION_CLOSURE_ARTIFACT_NAMES,
    )
    provider_readiness_paths = _find_named_artifacts(
        resolved_job_dir,
        PROVIDER_READINESS_ARTIFACT_NAMES,
    )
    all_payload_paths = _dedupe(
        str(path)
        for path in (
            finalizer_paths
            + spend_paths
            + watchdog_paths
            + artifact_output_paths
            + teardown_paths
            + provider_result_paths
            + remote_execution_paths
            + provider_readiness_paths
        )
    )
    all_payloads = _load_payloads([Path(path) for path in all_payload_paths])
    resolved_provider = _infer_provider(
        explicit_provider=provider,
        payloads=all_payloads,
        job_dir=resolved_job_dir,
    )
    credentials = _credential_audit(resolved_provider)

    checks = {
        "watchdog": _watchdog_check(job_dir=resolved_job_dir, paths=watchdog_paths),
        "spend_ledger": _spend_ledger_check(job_dir=resolved_job_dir, paths=spend_paths),
        "artifact_output_closure": _artifact_output_closure_check(
            job_dir=resolved_job_dir,
            paths=artifact_output_paths,
        ),
        "teardown": _teardown_check(
            job_dir=resolved_job_dir,
            paths=teardown_paths + finalizer_paths,
        ),
    }
    blockers = _dedupe(
        blocker
        for check in checks.values()
        for blocker in _string_list(check.get("blockers"))
    )
    closure_artifacts_present = bool(
        finalizer_paths or artifact_output_paths or teardown_paths or provider_result_paths
    )
    if not closure_artifacts_present:
        blockers.append("provider_closure_artifacts_unavailable")
    if blockers and not credentials.get("credential_configured"):
        blockers.append("provider_credentials_missing_for_remote_artifact_recovery")
    blockers = _dedupe(blockers)
    passed = all(bool(check.get("passed")) for check in checks.values()) and not blockers

    report = {
        "schema_version": PROVIDER_CLOSURE_AUDIT_SCHEMA_VERSION,
        "generated_at": resolved_generated_at,
        "status": "passed" if passed else "blocked_optional_provider_closure",
        "job_id": job_id or resolved_job_dir.name,
        "job_dir": str(resolved_job_dir),
        "provider": resolved_provider,
        "read_only_audit": True,
        "live_provider_calls_performed": False,
        "api_call_performed": False,
        "optional_provider_closure_verified": passed,
        "provider_credentials": credentials,
        "remote_execution": _remote_execution_summary(
            job_dir=resolved_job_dir,
            paths=remote_execution_paths,
        ),
        "provider_readiness": _provider_readiness_summary(
            job_dir=resolved_job_dir,
            paths=provider_readiness_paths,
        ),
        "checks": checks,
        "observed_artifacts": {
            "finalizer_proofs": [
                _artifact_summary(path, resolved_job_dir) for path in finalizer_paths
            ],
            "spend_ledgers": [
                _artifact_summary(path, resolved_job_dir) for path in spend_paths
            ],
            "watchdog_sources": [
                _artifact_summary(path, resolved_job_dir) for path in watchdog_paths
            ],
            "artifact_output_closure_sources": [
                _artifact_summary(path, resolved_job_dir) for path in artifact_output_paths
            ],
            "teardown_sources": [
                _artifact_summary(path, resolved_job_dir) for path in teardown_paths
            ],
            "provider_results": [
                _artifact_summary(path, resolved_job_dir) for path in provider_result_paths
            ],
            "remote_execution_closure": [
                _artifact_summary(path, resolved_job_dir) for path in remote_execution_paths
            ],
            "provider_readiness": [
                _artifact_summary(path, resolved_job_dir) for path in provider_readiness_paths
            ],
        },
        "artifacts": {
            "provider_closure_audit_report": str(resolved_output),
        },
        "blockers": blockers,
        "claim_boundary": {
            "optional_provider_closure_required_for_local_sim_only_beta": False,
            "optional_provider_closure_is_remote_runtime_evidence_only": True,
            "rank_fidelity_result_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            "field_success_proven": False,
            "public_claim_upgrade_allowed": False,
            "local_sim_only_beta_can_remain_blocker_separate": True,
        },
    }
    write_json(resolved_output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only audit of optional provider closure artifacts."
    )
    parser.add_argument("--job-dir")
    parser.add_argument("--capture-root")
    parser.add_argument("--job-id")
    parser.add_argument("--provider")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    report = audit_provider_closure(
        job_dir=args.job_dir,
        capture_root=args.capture_root,
        job_id=args.job_id,
        provider=args.provider,
        output_path=args.output_path,
    )
    print(f"status={report['status']}")
    print(f"report={report['artifacts']['provider_closure_audit_report']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
