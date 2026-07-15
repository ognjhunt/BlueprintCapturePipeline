"""Live-gated serial failover launcher for robot-eval provider-race handoffs."""

from __future__ import annotations

import argparse
import fcntl
import functools
import json
import logging
import os
import signal
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .logging_utils import log_event


PROVIDER_RACE_LAUNCHER_RESULT_SCHEMA_VERSION = (
    "robot_eval_gpu_provider_race_launcher_result.v1"
)
PROVIDER_RACE_HANDOFF_SCHEMA_VERSION = "robot_eval_gpu_provider_race_handoff.v1"
PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provider_launch_request.v1"
ALLOW_PROVIDER_RACE_LAUNCH_ENV = "BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH"
SENSITIVE_ENV_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
PROVIDER_ADAPTER_REGISTRY: dict[str, dict[str, str]] = {
    "runpod": {
        "adapter_id": "runpod_provider_adapter.v1",
        "executable": "blueprint-run-runpod-provider-adapter",
        "operation": "enqueue_runpod_serverless_or_on_demand_worker",
        "result_filename": "runpod_provider_adapter_result.json",
    },
    "vast": {
        "adapter_id": "vast_provider_adapter.v1",
        "executable": "blueprint-run-vast-provider-adapter",
        "operation": "create_vast_instance_and_run_worker",
        "result_filename": "vast_provider_adapter_result.json",
    },
    "lambda_cloud": {
        "adapter_id": "lambda_provider_adapter.v1",
        "executable": "blueprint-run-lambda-provider-adapter",
        "operation": "launch_lambda_cloud_instance_and_run_worker",
        "result_filename": "lambda_provider_adapter_result.json",
    },
}
logger = logging.getLogger(__name__)


def _exclusive_provider_race(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        request_value = kwargs.get("provider_launch_request_path")
        if request_value is None:
            raise ValueError("provider_launch_request_path is required")
        request_path = Path(request_value).resolve()
        ensure_dir(request_path.parent)
        lock_path = request_path.parent / ".provider_race_launcher.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return wrapped


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
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
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _has_cli_option(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _secret_values_from_env(env: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, value in env.items():
        if not value or len(value) < 4:
            continue
        if any(marker in name.upper() for marker in SENSITIVE_ENV_NAME_MARKERS):
            values.setdefault(value, name)
    return values


def _redact_text(value: Any, secret_values: Mapping[str, str]) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    for secret_value, env_name in sorted(
        secret_values.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = text.replace(secret_value, f"<redacted:{env_name}>")
    return text


def _read_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = read_json_any(path)
    except Exception as exc:  # noqa: BLE001 - launcher artifacts must fail closed
        return {}, type(exc).__name__
    if not isinstance(payload, Mapping):
        return {}, "not_mapping"
    return dict(payload), None


def _provider_race_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    prelaunch_guard = _mapping(request.get("prelaunch_spend_guard"))
    return _mapping(prelaunch_guard.get("provider_race") or request.get("provider_race"))


def _resolve_handoff_path(
    *,
    request_path: Path,
    request: Mapping[str, Any],
    handoff_path: str | Path | None,
) -> Path:
    provider_race = _provider_race_contract(request)
    raw_path = (
        str(handoff_path)
        if handoff_path is not None
        else _string(request.get("provider_race_handoff_path"))
        or _string(provider_race.get("provider_race_handoff_path"))
        or "gpu_provider_race_handoff.json"
    )
    path = Path(raw_path)
    return path if path.is_absolute() else request_path.parent / path


def _candidate_count(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _base_result(
    *,
    request_path: Path,
    handoff_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    provider_race = _provider_race_contract(request)
    return {
        "schema_version": PROVIDER_RACE_LAUNCHER_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "provider_launch_request_sha256": _sha_file(request_path)
        if request_path.is_file()
        else None,
        "provider_race_handoff_path": str(handoff_path),
        "output_path": str(output_path),
        "job_id": _string(request.get("job_id") or handoff.get("job_id")),
        "provider": _string(request.get("provider")) or None,
        "provider_race": provider_race or None,
        "provider_race_required_for_customer_path": bool(
            provider_race.get("race_required_for_customer_path")
            or handoff.get("provider_race_required_for_customer_path")
        ),
        "provider_race_launcher_available": True,
        "live_provider_calls_performed": False,
        "provider_race_execution_performed": False,
        "provider_race_execution_proven": False,
        "remote_cloud_execution_proven": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            "provider_race_launcher_result_is_not_provider_execution": True,
            "live_provider_calls_performed": False,
            "provider_race_execution_proven": False,
            "remote_cloud_execution_proven": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
        },
    }


def _handoff_blockers(
    *,
    request: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if request.get("schema_version") != PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION:
        blockers.append("invalid_provider_launch_request_schema")
    if handoff.get("schema_version") != PROVIDER_RACE_HANDOFF_SCHEMA_VERSION:
        blockers.append("invalid_provider_race_handoff_schema")
    request_job_id = _string(request.get("job_id"))
    handoff_job_id = _string(handoff.get("job_id"))
    if (
        not request_job_id
        or len(request_job_id) > 128
        or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in request_job_id
        )
    ):
        blockers.append("provider_race_job_id_invalid")
    if request_job_id and handoff_job_id and request_job_id != handoff_job_id:
        blockers.append("provider_race_handoff_job_id_mismatch")
    if handoff.get("provider_race_required_for_customer_path") is not True:
        blockers.append("provider_race_handoff_does_not_require_customer_race")
    if handoff.get("live_provider_calls_performed") is True:
        blockers.append("provider_race_handoff_unexpected_live_provider_calls")
    if _candidate_count(handoff.get("race_candidate_count")) < 2:
        blockers.append("provider_race_handoff_requires_two_race_candidates")
    if _candidate_count(handoff.get("runnable_candidate_count")) < 2:
        blockers.append("provider_race_handoff_requires_two_runnable_candidates")
    if handoff.get("provider_race_runtime_launcher_available") is not True:
        blockers.append("provider_race_launcher_command_not_declared")
    if not _string(handoff.get("launcher_command")):
        blockers.append("provider_race_launcher_command_missing")
    seen_providers: set[str] = set()
    for candidate in _runnable_candidates(handoff):
        provider = _string(candidate.get("provider"))
        adapter = PROVIDER_ADAPTER_REGISTRY.get(provider)
        if adapter is None:
            blockers.append(f"provider_race_adapter_not_registered:{provider or 'missing'}")
            continue
        if provider in seen_providers:
            blockers.append(f"provider_race_duplicate_candidate:{provider}")
        seen_providers.add(provider)
        if _string(candidate.get("operation")) != adapter["operation"]:
            blockers.append(f"provider_race_operation_mismatch:{provider}")
        supplied_command = _string(candidate.get("adapter_command"))
        if supplied_command and supplied_command != adapter["executable"]:
            blockers.append(f"provider_race_noncanonical_adapter_command:{provider}")
        supplied_adapter_id = _string(candidate.get("adapter_id"))
        if supplied_adapter_id and supplied_adapter_id != adapter["adapter_id"]:
            blockers.append(f"provider_race_adapter_id_mismatch:{provider}")
    return blockers


def _runnable_candidates(handoff: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = handoff.get("runnable_candidates")
    if not isinstance(candidates, Sequence) or isinstance(
        candidates,
        (str, bytes, bytearray),
    ):
        return []
    return [dict(item) for item in candidates if isinstance(item, Mapping)]


def _candidate_adapter_argv(
    candidate: Mapping[str, Any],
    *,
    request_path: Path,
    output_path: Path,
) -> list[str]:
    provider = _string(candidate.get("provider"))
    adapter = PROVIDER_ADAPTER_REGISTRY.get(provider)
    if adapter is None:
        return []
    argv = [adapter["executable"]]
    if provider == "vast":
        argv.extend(["--job-dir", str(output_path.parent)])
        argv.extend(["--mode", "live-startup-probe"])
        argv.append("--allow-vast-api-call")
        argv.append("--allow-vast-instance-launch")
    else:
        argv.extend(["--provider-launch-request", str(request_path)])
        argv.extend(["--output-path", str(output_path)])
        argv.extend(["--mode", "auto"])
        if provider == "runpod":
            argv.append("--allow-runpod-api-call")
        if provider == "lambda_cloud":
            argv.append("--allow-lambda-api-call")
    return argv


def _default_timeout_seconds(request: Mapping[str, Any], explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    shape = _mapping(request.get("provider_request_shape"))
    limits = _mapping(shape.get("limits"))
    timeout = _number(limits.get("hard_timeout_seconds"))
    return max(60, int(timeout or 900))


def _execute_serial_failover(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    handoff: Mapping[str, Any],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    candidates = _runnable_candidates(handoff)
    if len(candidates) < 2:
        return {
            "status": "blocked",
            "reason": "provider_race_runtime_candidates_missing",
            "blockers": ["provider_race_runtime_candidates_missing"],
            "attempts": [],
        }
    timeout = _default_timeout_seconds(request, timeout_seconds)
    env = os.environ.copy()
    env["BLUEPRINT_PROVIDER_RACE_LAUNCH_REQUEST_PATH"] = str(request_path)
    env["BLUEPRINT_PROVIDER_RACE_HANDOFF_PATH"] = str(
        output_path.parent / "gpu_provider_race_handoff.json"
    )
    secret_values = _secret_values_from_env(env)
    attempts: list[dict[str, Any]] = []
    job_id = _string(request.get("job_id"))
    attempts_root = output_path.parent / "provider_race_attempts"
    ensure_dir(attempts_root)

    for index, candidate in enumerate(candidates):
        provider = _string(candidate.get("provider"))
        adapter = PROVIDER_ADAPTER_REGISTRY.get(provider)
        if adapter is None:
            attempts.append(
                {
                    "provider": provider or None,
                    "status": "permanent_invalid",
                    "reason": "provider_adapter_not_registered",
                    "phases": [],
                }
            )
            continue
        attempt_dir = attempts_root / f"{index + 1:02d}-{provider}"
        ensure_dir(attempt_dir)
        if provider == "vast":
            write_json(attempt_dir / "gpu_provider_launch_request.json", request)
        adapter_result_path = attempt_dir / adapter["result_filename"]
        adapter_result_path.unlink(missing_ok=True)
        stdout_path = attempt_dir / "provider_adapter.stdout.log"
        stderr_path = attempt_dir / "provider_adapter.stderr.log"
        argv = _candidate_adapter_argv(
            candidate,
            request_path=request_path,
            output_path=adapter_result_path,
        )
        launched_at_ns = time.time_ns()
        command_result = _run_provider_adapter(
            argv=argv,
            env=env,
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            secret_values=secret_values,
        )
        adapter_result, result_blockers = _fresh_adapter_result(
            adapter_result_path,
            launched_at_ns=launched_at_ns,
        )
        resource_ids = _provider_resource_ids(provider, adapter_result)
        artifact_validation = _terminal_artifact_validation(
            adapter_result,
            expected_job_id=job_id,
        )
        teardown = _ensure_attempt_teardown(
            provider=provider,
            adapter_result=adapter_result,
            attempt_dir=attempt_dir,
            request_path=request_path,
            side_effects_possible=command_result.get("started") is True,
        )
        adapter_terminal = _string(adapter_result.get("status")) == "completed"
        command_completed = (
            command_result.get("exit_code") == 0
            and command_result.get("timed_out") is not True
        )
        won = bool(
            command_completed
            and adapter_terminal
            and artifact_validation.get("status") == "validated"
            and teardown.get("teardown_verified") is True
        )
        phases = [
            {
                "phase": "launch",
                "status": "completed"
                if command_result.get("started") is True
                else "blocked",
            },
            {
                "phase": "resource_id",
                "status": "observed" if resource_ids else "not_observed",
                "resource_ids": resource_ids,
            },
            {
                "phase": "startup",
                "status": _string(
                    adapter_result.get("startup_status")
                    or adapter_result.get("provider_phase")
                )
                or "not_proven",
            },
            {
                "phase": "execution",
                "status": "completed" if adapter_terminal else "not_completed",
            },
            {
                "phase": "artifact_validation",
                "status": artifact_validation.get("status"),
            },
            {
                "phase": "teardown",
                "status": "verified"
                if teardown.get("teardown_verified") is True
                else "unverified",
            },
        ]
        blockers = _dedupe(
            [
                *result_blockers,
                *_string_list(command_result.get("blockers")),
                *_string_list(artifact_validation.get("blockers")),
                *_string_list(teardown.get("blockers")),
            ]
        )
        attempt = {
            "provider": provider,
            "adapter_id": adapter["adapter_id"],
            "status": "won" if won else "failed",
            "reason": "fresh_terminal_artifact_and_teardown_verified"
            if won
            else "provider_candidate_did_not_reach_verified_terminal_state",
            "exit_code": command_result.get("exit_code"),
            "timed_out": command_result.get("timed_out") is True,
            "adapter_result_path": str(adapter_result_path),
            "adapter_result_sha256": _sha_file(adapter_result_path)
            if adapter_result
            else None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "resource_ids": resource_ids,
            "artifact_validation": artifact_validation,
            "teardown": teardown,
            "phases": phases,
            "blockers": blockers,
            "command": {
                "shell": False,
                "adapter_registry_owned": True,
                "adapter_id": adapter["adapter_id"],
                "executable": adapter["executable"],
                "argv_count": len(argv),
                "raw_candidate_command_executed": False,
            },
        }
        attempts.append(attempt)
        if teardown.get("teardown_verified") is not True:
            return {
                "status": "failed",
                "reason": "provider_teardown_unverified_failover_stopped",
                "blockers": ["provider_teardown_unverified_failover_stopped", *blockers],
                "attempts": attempts,
                "provider_adapter_commands_executed": True,
                "timeout_seconds": timeout,
            }
        if won:
            return {
                "status": "completed",
                "reason": "provider_race_serial_failover_completed",
                "blockers": [],
                "attempts": attempts,
                "winning_provider": provider,
                "winning_artifact": artifact_validation.get("artifact"),
                "provider_adapter_commands_executed": True,
                "timeout_seconds": timeout,
            }
    return {
        "status": "failed",
        "reason": "all_provider_failover_candidates_failed",
        "blockers": ["all_provider_failover_candidates_failed"],
        "attempts": attempts,
        "provider_adapter_commands_executed": bool(attempts),
        "timeout_seconds": timeout,
    }


def _run_provider_adapter(
    *,
    argv: Sequence[str],
    env: Mapping[str, str],
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    secret_values: Mapping[str, str],
) -> dict[str, Any]:
    if not argv:
        return {
            "started": False,
            "exit_code": None,
            "timed_out": False,
            "blockers": ["provider_adapter_registry_argv_missing"],
        }
    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    blockers: list[str] = []
    try:
        process = subprocess.Popen(  # noqa: S603 - argv comes only from fixed registry
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(env),
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            blockers.append("provider_adapter_command_timeout")
            stdout = _redact_text(exc.stdout, secret_values)
            stderr = _redact_text(exc.stderr, secret_values)
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=1)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)
    except FileNotFoundError as exc:
        blockers.append("provider_adapter_command_not_found")
        stderr = str(exc)
    finally:
        write_text(stdout_path, _redact_text(stdout, secret_values))
        write_text(stderr_path, _redact_text(stderr, secret_values))
    return {
        "started": process is not None,
        "exit_code": process.returncode if process is not None else None,
        "timed_out": timed_out,
        "blockers": blockers,
    }


def _fresh_adapter_result(
    path: Path,
    *,
    launched_at_ns: int,
) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["provider_adapter_result_missing"]
    try:
        if path.stat().st_mtime_ns < launched_at_ns:
            return {}, ["provider_adapter_result_stale"]
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["provider_adapter_result_invalid"]
    if not isinstance(payload, Mapping):
        return {}, ["provider_adapter_result_not_mapping"]
    return dict(payload), []


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provider_resource_ids(provider: str, result: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    if provider == "runpod":
        response = _mapping(result.get("runpod_response"))
        values.extend(
            [
                result.get("pod_id"),
                result.get("runpod_job_id"),
                response.get("id"),
                response.get("pod_id"),
            ]
        )
    elif provider == "vast":
        values.extend(result.get("vast_instance_ids") or [])
    elif provider == "lambda_cloud":
        values.extend(result.get("lambda_instance_ids") or [])
    return _dedupe(_string(value) for value in values)


def _terminal_artifact_validation(
    result: Mapping[str, Any],
    *,
    expected_job_id: str,
) -> dict[str, Any]:
    artifact = _mapping(result.get("terminal_artifact"))
    blockers: list[str] = []
    if not artifact:
        blockers.append("provider_terminal_artifact_missing")
    if _string(artifact.get("status")) != "validated":
        blockers.append("provider_terminal_artifact_not_validated")
    if _string(artifact.get("job_id")) != expected_job_id:
        blockers.append("provider_terminal_artifact_job_id_mismatch")
    digest = _string(artifact.get("sha256"))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        blockers.append("provider_terminal_artifact_digest_invalid")
    if not _string(artifact.get("artifact_uri")):
        blockers.append("provider_terminal_artifact_uri_missing")
    return {
        "status": "validated" if not blockers else "blocked",
        "artifact": artifact or None,
        "blockers": blockers,
    }


def _teardown_evidence(result: Mapping[str, Any]) -> bool:
    teardown = _mapping(result.get("teardown"))
    return bool(
        result.get("provider_teardown_proven") is True
        or (
            result.get("continuing_spend_from_this_run") is False
            and _string(result.get("teardown_status")) == "completed"
        )
        or teardown.get("provider_api_terminal_status_confirmed") is True
        or teardown.get("teardown_verified") is True
    )


def _ensure_attempt_teardown(
    *,
    provider: str,
    adapter_result: Mapping[str, Any],
    attempt_dir: Path,
    request_path: Path,
    side_effects_possible: bool,
) -> dict[str, Any]:
    if _teardown_evidence(adapter_result):
        return {
            "status": "verified",
            "teardown_verified": True,
            "source": "provider_adapter_result",
            "blockers": [],
        }
    resource_ids = _provider_resource_ids(provider, adapter_result)
    side_effects_reported = any(
        adapter_result.get(key) is True
        for key in (
            "api_call_performed",
            "provider_job_submitted",
            "runpod_side_effects_may_have_occurred",
            "vast_side_effects_may_have_occurred",
            "lambda_side_effects_may_have_occurred",
        )
    )
    side_effect_fields = (
        "api_call_performed",
        "provider_job_submitted",
        "runpod_side_effects_may_have_occurred",
        "vast_side_effects_may_have_occurred",
        "lambda_side_effects_may_have_occurred",
    )
    explicit_no_side_effects = bool(adapter_result) and any(
        key in adapter_result for key in side_effect_fields
    ) and all(adapter_result.get(key) is not True for key in side_effect_fields)
    if (
        not side_effects_reported
        and not resource_ids
        and (not side_effects_possible or explicit_no_side_effects)
    ):
        return {
            "status": "not_required",
            "teardown_verified": True,
            "source": "adapter_not_started",
            "blockers": [],
        }
    if provider == "runpod" and resource_ids:
        from .runpod_wam_async_runner import _delete_pod, _read_runpod_api_key

        api_key, _ = _read_runpod_api_key()
        if api_key:
            manifests = [
                _delete_pod(
                    job_dir=attempt_dir,
                    pod_id=resource_id,
                    api_key=api_key,
                    generated_at=utc_now_iso(),
                )
                for resource_id in resource_ids
            ]
            verified = all(
                manifest.get("terminal_state_api_confirmed") is True
                for manifest in manifests
            )
            return {
                "status": "verified" if verified else "blocked",
                "teardown_verified": verified,
                "source": "runpod_delete_and_readback",
                "manifests": manifests,
                "blockers": [] if verified else ["runpod_teardown_not_verified"],
            }
    if provider == "lambda_cloud" and resource_ids:
        from .lambda_provider_adapter import run_lambda_provider_adapter

        teardown_result = run_lambda_provider_adapter(
            provider_launch_request_path=request_path,
            output_path=attempt_dir / "lambda_teardown_result.json",
            mode="terminate-instances",
            allow_lambda_api_call=True,
            instance_ids=resource_ids,
        )
        verified = teardown_result.get("provider_teardown_proven") is True
        return {
            "status": "verified" if verified else "blocked",
            "teardown_verified": verified,
            "source": "lambda_terminate_and_readback",
            "result": teardown_result,
            "blockers": [] if verified else ["lambda_teardown_not_verified"],
        }
    return {
        "status": "blocked",
        "teardown_verified": False,
        "source": "no_verified_cleanup_path",
        "resource_ids": resource_ids,
        "blockers": ["provider_teardown_not_verified"],
    }


def _runtime_blockers(handoff: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    handoff_status = _string(handoff.get("status"))
    if handoff_status != "ready_for_customer_provider_race_runtime":
        blockers.append("provider_race_handoff_not_ready")
    if handoff.get("customer_path_provider_failover_runtime_wired") is not True:
        blockers.append("customer_path_provider_failover_runtime_not_wired")
    blockers.extend(_string_list(handoff.get("blockers")))
    blockers.extend(
        _string_list(handoff.get("customer_path_provider_failover_runtime_blockers"))
    )
    blockers.extend(
        _string_list(handoff.get("provider_race_runtime_launcher_blockers"))
    )
    return blockers


@_exclusive_provider_race
def run_robot_eval_provider_race_launcher(
    *,
    provider_launch_request_path: str | Path,
    handoff_path: str | Path | None = None,
    output_path: str | Path | None = None,
    allow_live_provider_race: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Validate or execute a provider-race handoff.

    By default the command stops before provider API calls and proves only that
    the customer path has a runnable failover-launcher artifact contract. Live
    serial failover requires both ``allow_live_provider_race=True`` and
    ``BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH=true``.
    """

    request_path = Path(provider_launch_request_path).resolve()
    request, request_error = _read_mapping(request_path)
    job_dir = request_path.parent.resolve()
    requested_handoff_path = _resolve_handoff_path(
        request_path=request_path,
        request=request,
        handoff_path=handoff_path,
    )
    requested_output = (
        Path(output_path).resolve()
        if output_path
        else request_path.parent / "gpu_provider_race_launcher_result.json"
    )
    path_blockers: list[str] = []
    try:
        resolved_handoff_path = requested_handoff_path.resolve()
        resolved_handoff_path.relative_to(job_dir)
    except ValueError:
        resolved_handoff_path = job_dir / "gpu_provider_race_handoff.json"
        path_blockers.append("provider_race_handoff_path_outside_job_dir")
    try:
        resolved_output = requested_output.resolve()
        resolved_output.relative_to(job_dir)
    except ValueError:
        resolved_output = job_dir / "gpu_provider_race_launcher_result.json"
        path_blockers.append("provider_race_output_path_outside_job_dir")
    ensure_dir(resolved_output.parent)
    previous_result, _ = _read_mapping(resolved_output)

    handoff, handoff_error = _read_mapping(resolved_handoff_path)
    result = _base_result(
        request_path=request_path,
        handoff_path=resolved_handoff_path,
        output_path=resolved_output,
        request=request,
        handoff=handoff,
    )

    structural_blockers: list[str] = list(path_blockers)
    if request_error:
        structural_blockers.append(f"provider_launch_request_{request_error}")
    if handoff_error:
        structural_blockers.append(f"provider_race_handoff_{handoff_error}")
    if not request_error and not handoff_error:
        structural_blockers.extend(
            _handoff_blockers(request=request, handoff=handoff)
        )

    runtime_blockers = [] if structural_blockers else _runtime_blockers(handoff)
    blockers = _dedupe([*structural_blockers, *runtime_blockers])
    ready = not blockers
    live_gate_open = bool(
        allow_live_provider_race and _env_truthy(ALLOW_PROVIDER_RACE_LAUNCH_ENV)
    )
    if live_gate_open:
        runtime_blockers.append(
            "legacy_provider_race_launcher_disabled_use_paid_resource_allocator"
        )
        blockers = _dedupe([*structural_blockers, *runtime_blockers])
        ready = False
        live_gate_open = False
    result.update(
        {
            "status": "ready_for_live_provider_race" if ready else "blocked",
            "reason": "provider_race_launcher_ready"
            if ready
            else "provider_race_launcher_gate_blocked",
            "blockers": blockers,
            "structural_blockers": _dedupe(structural_blockers),
            "runtime_blockers": _dedupe(runtime_blockers),
            "provider_race_handoff_status": handoff.get("status"),
            "provider_race_handoff_ready": ready,
            "provider_race_runtime_launcher_available": bool(
                handoff.get("provider_race_runtime_launcher_available")
            ),
            "launcher_command": handoff.get("launcher_command"),
            "race_candidate_count": _candidate_count(handoff.get("race_candidate_count")),
            "runnable_candidate_count": _candidate_count(
                handoff.get("runnable_candidate_count")
            ),
            "allow_live_provider_race_env": ALLOW_PROVIDER_RACE_LAUNCH_ENV,
            "allow_live_provider_race_env_present": _env_truthy(
                ALLOW_PROVIDER_RACE_LAUNCH_ENV
            ),
            "cli_allow_live_provider_race_present": bool(allow_live_provider_race),
            "live_provider_race_gate_open": live_gate_open,
        }
    )
    if (
        ready
        and live_gate_open
        and previous_result.get("status") == "completed"
        and previous_result.get("provider_race_execution_proven") is True
        and previous_result.get("provider_launch_request_sha256")
        == result.get("provider_launch_request_sha256")
        and _string(previous_result.get("job_id")) == _string(result.get("job_id"))
    ):
        return {
            **previous_result,
            "idempotent_replay": True,
            "provider_adapter_commands_executed_on_replay": False,
        }
    if ready and live_gate_open:
        execution = _execute_serial_failover(
            request_path=request_path,
            output_path=resolved_output,
            request=request,
            handoff=handoff,
            timeout_seconds=timeout_seconds,
        )
        completed = execution.get("status") == "completed"
        result.update(
            {
                "status": "completed" if completed else execution.get("status") or "failed",
                "reason": execution.get("reason"),
                "blockers": _string_list(execution.get("blockers")),
                "provider_race_execution_performed": True,
                "provider_race_execution_proven": completed,
                "provider_adapter_commands_executed": execution.get(
                    "provider_adapter_commands_executed"
                )
                is True,
                "live_provider_calls_performed": True,
                "remote_cloud_execution_proven": False,
                "public_claim_upgrade_allowed": False,
                "winning_provider": execution.get("winning_provider"),
                "failover_attempts": execution.get("attempts") or [],
                "timeout_seconds": execution.get("timeout_seconds"),
                "claim_boundary": {
                    **_mapping(result.get("claim_boundary")),
                    "provider_race_launcher_result_is_not_provider_execution": False,
                    "provider_race_execution_proven": completed,
                    "remote_cloud_execution_proven": False,
                    "simulator_execution_proven": False,
                    "rank_fidelity_result_proven": False,
                },
            }
        )
    result["revision"] = int(previous_result.get("revision") or 0) + 1
    write_json(resolved_output, result)
    log_event(
        logger,
        logging.INFO if ready else logging.WARNING,
        "robot_eval_provider_race_launcher.ready"
        if result.get("status") == "ready_for_live_provider_race"
        else "robot_eval_provider_race_launcher.completed"
        if result.get("status") == "completed"
        else "robot_eval_provider_race_launcher.blocked",
        output_path=str(resolved_output),
        job_id=result.get("job_id"),
        status=result.get("status"),
        blocker_count=len(blockers),
        blockers=blockers,
        live_provider_calls_performed=result.get("live_provider_calls_performed") is True,
    )
    return result


def _request_path_from_args(args: argparse.Namespace) -> Path:
    if args.provider_launch_request:
        return Path(args.provider_launch_request)
    if args.job_dir:
        return Path(args.job_dir) / "gpu_provider_launch_request.json"
    raise ValueError("Provide --provider-launch-request or --job-dir")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a no-spend robot-eval GPU provider-race launcher handoff."
    )
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--job-dir")
    parser.add_argument("--handoff")
    parser.add_argument("--output-path")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument(
        "--allow-live-provider-race",
        action="store_true",
        help=(
            "Execute runnable provider adapter commands. Also requires "
            f"{ALLOW_PROVIDER_RACE_LAUNCH_ENV}=true."
        ),
    )
    args = parser.parse_args(argv)
    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=args.handoff,
        output_path=args.output_path,
        allow_live_provider_race=args.allow_live_provider_race,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[robot-eval-provider-race-launcher] result={result['output_path']}")
    print(f"[robot-eval-provider-race-launcher] status={result['status']}")
    print(f"[robot-eval-provider-race-launcher] job_id={result.get('job_id')}")
    blockers = result.get("blockers")
    if blockers:
        print(
            "[robot-eval-provider-race-launcher] blockers="
            + ",".join(str(item) for item in blockers)
        )
    return 0 if result["status"] in {"ready_for_live_provider_race", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
