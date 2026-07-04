"""Command-gated launcher for prepared robot-eval GPU provider requests."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event


GPU_PROVIDER_LAUNCHER_RESULT_SCHEMA_VERSION = (
    "robot_eval_gpu_provider_launcher_result.v1"
)
ALLOW_PROVIDER_LAUNCH_ENV = "BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH"
PROVIDER_LAUNCH_COMMAND_ENV = "BLUEPRINT_GPU_PROVIDER_LAUNCH_COMMAND"
SENSITIVE_ENV_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
logger = logging.getLogger(__name__)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _env_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _mapping(value: Any) -> Dict[str, Any]:
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


def _secret_env_var_names(request: Mapping[str, Any]) -> list[str]:
    provider_shape = _mapping(request.get("provider_request_shape"))
    environment = _mapping(provider_shape.get("environment"))
    explicit_names = _string_list(environment.get("secret_env_var_names"))
    ambient_names = [
        name
        for name in os.environ
        if any(marker in name.upper() for marker in SENSITIVE_ENV_NAME_MARKERS)
    ]
    return _dedupe([*explicit_names, *ambient_names])


def _secret_values_from_env(env: Mapping[str, str], names: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        value = env.get(name)
        if value and len(value) >= 4:
            values.setdefault(value, name)
    return values


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _redact_text(value: Any, secret_values: Mapping[str, str]) -> str:
    text = _output_text(value)
    for secret_value, env_name in sorted(
        secret_values.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = text.replace(secret_value, f"<redacted:{env_name}>")
    return text


def _log_redaction_summary(secret_values: Mapping[str, str]) -> dict[str, Any]:
    return {
        "stdout_stderr_secret_redaction_enabled": True,
        "redacted_secret_env_var_names": sorted(set(secret_values.values())),
        "redacted_secret_value_count": len(secret_values),
    }


def _command_summary(argv: Sequence[str]) -> dict[str, Any]:
    return {
        "shell": False,
        "executable": Path(argv[0]).name if argv else "",
        "argv_count": len(argv),
        "argument_count": max(len(argv) - 1, 0),
        "arguments_redacted": max(len(argv) - 1, 0),
        "raw_command_stored": False,
    }


def _provider_context(request: Mapping[str, Any]) -> dict[str, Any]:
    provider_shape = _mapping(request.get("provider_request_shape"))
    image = _mapping(provider_shape.get("image"))
    inputs = _mapping(provider_shape.get("inputs"))
    limits = _mapping(provider_shape.get("limits"))
    prelaunch_spend_guard = _mapping(request.get("prelaunch_spend_guard"))
    return {
        "worker_image_ref_present": bool(_string(image.get("configured_image_ref"))),
        "worker_image_ref_is_versioned": image.get("configured_image_ref_is_versioned")
        is True,
        "manifest_uri_present": bool(_string(inputs.get("manifest_uri"))),
        "manifest_uri_fetchable_by_provider": inputs.get("manifest_uri_fetchable_by_provider")
        is True,
        "artifact_output_uri_present": bool(_string(inputs.get("artifact_output_uri"))),
        "hard_timeout_seconds": limits.get("hard_timeout_seconds"),
        "idle_timeout_seconds": limits.get("idle_timeout_seconds"),
        "external_watchdog_ttl_seconds": limits.get("external_watchdog_ttl_seconds"),
        "max_active_workers": limits.get("max_active_workers"),
        "prelaunch_spend_guard_required": prelaunch_spend_guard.get(
            "required_before_provider_launch"
        )
        is True,
        "prelaunch_spend_guard_can_launch": prelaunch_spend_guard.get("can_launch")
        is True,
    }


def _launcher_env(
    *,
    request_path: Path,
    output_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    request: Mapping[str, Any],
) -> dict[str, str]:
    provider_shape = _mapping(request.get("provider_request_shape"))
    image = _mapping(provider_shape.get("image"))
    inputs = _mapping(provider_shape.get("inputs"))
    limits = _mapping(provider_shape.get("limits"))
    provider = _string(request.get("provider"))
    env = os.environ.copy()
    env["BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST"] = str(request_path)
    env["BLUEPRINT_GPU_PROVIDER_LAUNCHER_OUTPUT"] = str(output_path)
    env["BLUEPRINT_GPU_PROVIDER_LAUNCHER_STDOUT"] = str(stdout_path)
    env["BLUEPRINT_GPU_PROVIDER_LAUNCHER_STDERR"] = str(stderr_path)
    env["BLUEPRINT_GPU_PROVIDER_ADAPTER_OUTPUT"] = str(
        request_path.parent / f"{provider or 'gpu'}_provider_adapter_result.json"
    )
    env["BLUEPRINT_ROBOT_EVAL_JOB_ID"] = _string(request.get("job_id"))
    env["BLUEPRINT_GPU_PROVIDER"] = provider
    env["BLUEPRINT_WORKER_IMAGE_REF"] = _string(image.get("configured_image_ref"))
    env["BLUEPRINT_EVAL_MANIFEST_URI"] = _string(inputs.get("manifest_uri"))
    env["BLUEPRINT_ARTIFACT_OUTPUT_URI"] = _string(inputs.get("artifact_output_uri"))
    env["BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS"] = _env_value(
        limits.get("hard_timeout_seconds")
    )
    env["BLUEPRINT_GPU_PROVIDER_IDLE_TIMEOUT_SECONDS"] = _env_value(
        limits.get("idle_timeout_seconds")
    )
    env["BLUEPRINT_GPU_PROVIDER_MAX_ACTIVE_WORKERS"] = _env_value(
        limits.get("max_active_workers")
    )
    env["BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS"] = _env_value(
        limits.get("external_watchdog_ttl_seconds")
    )
    return env


def _default_timeout_seconds(
    request: Mapping[str, Any],
    explicit_timeout_seconds: int | None,
) -> int:
    if explicit_timeout_seconds and explicit_timeout_seconds > 0:
        return explicit_timeout_seconds
    provider_shape = _mapping(request.get("provider_request_shape"))
    limits = _mapping(provider_shape.get("limits"))
    configured = _number(
        limits.get("external_watchdog_ttl_seconds") or limits.get("hard_timeout_seconds")
    )
    return int(configured or 300)


def _base_result(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    prelaunch_spend_guard = _mapping(request.get("prelaunch_spend_guard"))
    provider_race = _mapping(
        prelaunch_spend_guard.get("provider_race") or request.get("provider_race")
    )
    return {
        "schema_version": GPU_PROVIDER_LAUNCHER_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "provider_launch_request_path": str(request_path),
        "output_path": str(output_path),
        "job_id": _string(request.get("job_id")),
        "provider": _string(request.get("provider")) or "fixture_local",
        "provider_launch_request_status": _string(request.get("status")),
        "execution_performed": False,
        "provider_launcher_command_executed": False,
        "live_provider_calls_performed_by_launcher_module": False,
        "live_provider_call_proven": False,
        "provider_allocation_proven": False,
        "provider_side_effects_may_have_occurred": False,
        "actual_gpu_seconds": None,
        "actual_gpu_time_record_present": False,
        "secret_values_in_artifact": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "provider_context": _provider_context(request),
        "prelaunch_spend_guard": prelaunch_spend_guard,
        "provider_race": provider_race,
    }


def _result_event_name(status: str) -> str:
    if status == "blocked":
        return "robot_eval_provider_launcher.blocked"
    if status == "failed":
        return "robot_eval_provider_launcher.failed"
    return "robot_eval_provider_launcher.completed"


def _write_result(output_path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    persisted = dict(result)
    write_json(output_path, persisted)
    blockers = _string_list(persisted.get("blockers"))
    status = _string(persisted.get("status"))
    log_event(
        logger,
        logging.WARNING if status in {"blocked", "failed"} else logging.INFO,
        _result_event_name(status),
        output_path=str(output_path),
        job_id=persisted.get("job_id"),
        provider=persisted.get("provider"),
        status=status,
        reason=persisted.get("reason"),
        blocker_count=len(blockers),
        blockers=blockers,
        execution_performed=persisted.get("execution_performed"),
        provider_launcher_command_executed=persisted.get(
            "provider_launcher_command_executed"
        ),
        exit_code=persisted.get("exit_code"),
        timeout_seconds=persisted.get("timeout_seconds"),
    )
    return persisted


def _request_blockers(
    *,
    request: Mapping[str, Any],
    allow_provider_launch: bool,
    command_text: str,
) -> list[str]:
    blockers: list[str] = []
    provider = _string(request.get("provider")) or "fixture_local"
    provider_shape = _mapping(request.get("provider_request_shape"))
    image = _mapping(provider_shape.get("image"))
    inputs = _mapping(provider_shape.get("inputs"))
    environment = _mapping(provider_shape.get("environment"))
    prelaunch_spend_guard = _mapping(request.get("prelaunch_spend_guard"))
    external_provider = provider != "fixture_local"
    if request.get("status") != "request_manifest_ready":
        blockers.append("provider_launch_request_not_ready")
    if external_provider:
        if (
            prelaunch_spend_guard.get("required_before_provider_launch") is True
            and prelaunch_spend_guard.get("can_launch") is not True
        ):
            blockers.append("provider_prelaunch_spend_guard_not_passed")
            blockers.extend(_string_list(prelaunch_spend_guard.get("blockers")))
        if not _env_truthy(ALLOW_PROVIDER_LAUNCH_ENV):
            blockers.append(f"missing_env_{ALLOW_PROVIDER_LAUNCH_ENV}")
        if not allow_provider_launch:
            blockers.append("missing_cli_allow_provider_launch")
        if provider != "vast" and not command_text:
            blockers.append("missing_gpu_provider_launch_command")
        if provider_shape.get("api_payload_is_provider_adapter_template") is not True:
            blockers.append("provider_launch_request_not_adapter_template")
        if environment.get("secret_values_in_artifact") is not False:
            blockers.append("provider_launch_request_secret_values_in_artifact")
        if image.get("owner_published_image_ref_required") is True and not _string(
            image.get("configured_image_ref")
        ):
            blockers.append("missing_provider_worker_image_ref")
        if inputs.get("manifest_uri_required_for_provider") is True and not _string(
            inputs.get("manifest_uri")
        ):
            blockers.append("missing_provider_worker_manifest_uri")
        if inputs.get("manifest_uri_required_for_provider") is True and inputs.get(
            "manifest_uri_fetchable_by_provider"
        ) is not True:
            blockers.append("provider_worker_manifest_uri_not_fetchable")
        if inputs.get("artifact_output_uri_required") is True and not _string(
            inputs.get("artifact_output_uri")
        ):
            blockers.append("missing_provider_artifact_output_uri")
    return _dedupe(blockers)


def _run_builtin_vast_provider_adapter(
    *,
    request_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    from .vast_provider_adapter import run_vast_provider_adapter

    provider_shape = _mapping(request.get("provider_request_shape"))
    image = _mapping(provider_shape.get("image"))
    inputs = _mapping(provider_shape.get("inputs"))
    limits = _mapping(provider_shape.get("limits"))
    max_live_minutes = max(
        1,
        int((_number(limits.get("hard_timeout_seconds")) or 300) // 60),
    )
    adapter_result = run_vast_provider_adapter(
        job_dir=request_path.parent,
        mode="live-startup-probe",
        allow_vast_api_call=True,
        allow_instance_launch=True,
        public_image=_string(image.get("configured_image_ref")) or None,
        provider_bundle_url=_string(inputs.get("capture_root_bundle_uri")) or None,
        provider_output_put_url=_string(
            _mapping(inputs.get("artifact_output_write_auth")).get("signed_put_url")
        )
        or None,
        provider_output_get_url=_string(inputs.get("artifact_output_uri")) or None,
        max_live_minutes=max_live_minutes,
        startup_timeout_seconds=max(60, int(_number(limits.get("hard_timeout_seconds")) or 300)),
        poll_interval_seconds=max(
            1,
            int(_number(os.getenv("BLUEPRINT_GPU_PROVIDER_POLL_INTERVAL_SECONDS")) or 10),
        ),
    )
    status = _string(adapter_result.get("status"))
    blockers = _string_list(adapter_result.get("blockers"))
    completed = status in {"completed", "success", "succeeded"}
    persisted = dict(result)
    persisted.update(
        {
            "status": "completed" if completed else status or "failed",
            "reason": (
                "builtin_vast_provider_adapter_completed"
                if completed
                else _string(adapter_result.get("reason"))
                or "builtin_vast_provider_adapter_failed"
            ),
            "blockers": [] if completed else blockers or ["vast_provider_adapter_failed"],
            "execution_performed": True,
            "provider_launcher_command_executed": False,
            "builtin_provider_adapter_executed": True,
            "live_provider_calls_performed_by_launcher_module": True,
            "live_provider_call_proven": bool(adapter_result.get("live_provider_call_proven")),
            "provider_allocation_proven": bool(
                adapter_result.get("provider_allocation_proven")
                or adapter_result.get("vast_instance_ids")
            ),
            "provider_side_effects_may_have_occurred": True,
            "adapter_result_path": str(request_path.parent / "vast_provider_adapter_result.json"),
            "adapter_status": status,
            "adapter_blockers": blockers,
            "vast_instance_ids": adapter_result.get("vast_instance_ids") or [],
            "all_vast_instances_destroyed_by_adapter": adapter_result.get(
                "all_vast_instances_destroyed_by_adapter"
            ),
            "timeout_seconds": timeout_seconds,
        }
    )
    return _write_result(output_path, persisted)


def run_gpu_provider_launcher(
    *,
    provider_launch_request_path: str | Path,
    output_path: str | Path | None = None,
    allow_provider_launch: bool = False,
    provider_launch_command: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run an explicitly supplied GPU provider launcher command.

    This function never calls provider APIs directly. It validates the prepared
    dry-run provider request, requires an env gate plus a CLI gate for live
    providers, executes only the supplied command argv, and writes a result
    artifact that cannot upgrade simulator or rank-fidelity proof.
    """

    request_path = Path(provider_launch_request_path).resolve()
    resolved_output = (
        Path(output_path).resolve()
        if output_path
        else request_path.parent / "gpu_provider_launcher_result.json"
    )
    ensure_dir(resolved_output.parent)
    generated_at = utc_now_iso()
    payload = read_json_any(request_path)
    request = dict(payload) if isinstance(payload, Mapping) else {}
    log_event(
        logger,
        logging.INFO,
        "robot_eval_provider_launcher.started",
        request_path=str(request_path),
        output_path=str(resolved_output),
        job_id=request.get("job_id"),
        provider=request.get("provider"),
        request_status=request.get("status"),
        allow_provider_launch=allow_provider_launch,
        command_provided=bool(provider_launch_command or os.getenv(PROVIDER_LAUNCH_COMMAND_ENV)),
    )
    if not request:
        result = _base_result(
            request_path=request_path,
            output_path=resolved_output,
            request={},
            generated_at=generated_at,
        )
        result.update(
            {
                "status": "blocked",
                "reason": "invalid_provider_launch_request",
                "blockers": ["invalid_provider_launch_request_json"],
            }
        )
        return _write_result(resolved_output, result)

    result = _base_result(
        request_path=request_path,
        output_path=resolved_output,
        request=request,
        generated_at=generated_at,
    )
    if request.get("schema_version") != "robot_eval_gpu_provider_launch_request.v1":
        result.update(
            {
                "status": "blocked",
                "reason": "invalid_provider_launch_request_schema",
                "blockers": ["invalid_provider_launch_request_schema"],
            }
        )
        return _write_result(resolved_output, result)

    provider = _string(request.get("provider")) or "fixture_local"
    command_text = (
        _string(provider_launch_command)
        or _string(os.getenv(PROVIDER_LAUNCH_COMMAND_ENV))
    )
    if provider == "fixture_local":
        result.update(
            {
                "status": "not_required_for_fixture_local",
                "reason": "fixture_local_does_not_require_provider_launcher",
                "blockers": [],
            }
        )
        return _write_result(resolved_output, result)

    blockers = _request_blockers(
        request=request,
        allow_provider_launch=allow_provider_launch,
        command_text=command_text,
    )
    if blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "provider_launcher_gate_blocked",
                "blockers": blockers,
                "command": {
                    "provided": bool(command_text),
                    "raw_command_stored": False,
                },
            }
        )
        return _write_result(resolved_output, result)

    if provider == "vast" and not command_text:
        return _run_builtin_vast_provider_adapter(
            request_path=request_path,
            output_path=resolved_output,
            request=request,
            result=result,
            timeout_seconds=timeout_seconds,
        )

    try:
        argv = shlex.split(command_text)
    except ValueError as exc:
        result.update(
            {
                "status": "blocked",
                "reason": "invalid_gpu_provider_launch_command",
                "blockers": ["invalid_gpu_provider_launch_command"],
                "command_parse_error": str(exc),
            }
        )
        return _write_result(resolved_output, result)
    if not argv:
        result.update(
            {
                "status": "blocked",
                "reason": "missing_gpu_provider_launch_command",
                "blockers": ["missing_gpu_provider_launch_command"],
            }
        )
        return _write_result(resolved_output, result)

    stdout_path = resolved_output.parent / "gpu_provider_launcher.stdout.log"
    stderr_path = resolved_output.parent / "gpu_provider_launcher.stderr.log"
    timeout = _default_timeout_seconds(request, timeout_seconds)
    command_summary = _command_summary(argv)
    env = _launcher_env(
        request_path=request_path,
        output_path=resolved_output,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        request=request,
    )
    secret_values = _secret_values_from_env(env, _secret_env_var_names(request))
    redaction_summary = _log_redaction_summary(secret_values)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        stdout_path.write_text(_redact_text(completed.stdout, secret_values), encoding="utf-8")
        stderr_path.write_text(_redact_text(completed.stderr, secret_values), encoding="utf-8")
        success = completed.returncode == 0
        result.update(
            {
                "status": "completed" if success else "failed",
                "reason": "provider_launcher_command_completed"
                if success
                else "provider_launcher_command_failed",
                "blockers": [] if success else ["gpu_provider_launch_command_failed"],
                "execution_performed": True,
                "provider_launcher_command_executed": True,
                "provider_side_effects_may_have_occurred": True,
                "command": command_summary,
                "exit_code": completed.returncode,
                "timeout_seconds": timeout,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                **redaction_summary,
            }
        )
    except FileNotFoundError as exc:
        result.update(
            {
                "status": "blocked",
                "reason": "gpu_provider_launch_command_not_found",
                "blockers": ["gpu_provider_launch_command_not_found"],
                "command": command_summary,
                "command_error": str(exc),
                "timeout_seconds": timeout,
            }
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_redact_text(exc.stdout, secret_values), encoding="utf-8")
        stderr_path.write_text(_redact_text(exc.stderr, secret_values), encoding="utf-8")
        result.update(
            {
                "status": "failed",
                "reason": "provider_launcher_command_timeout",
                "blockers": ["gpu_provider_launch_command_timeout"],
                "execution_performed": True,
                "provider_launcher_command_executed": True,
                "provider_side_effects_may_have_occurred": True,
                "command": command_summary,
                "timeout_seconds": timeout,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                **redaction_summary,
            }
        )
    return _write_result(resolved_output, result)


def _request_path_from_args(args: argparse.Namespace) -> Path:
    if args.provider_launch_request:
        return Path(args.provider_launch_request)
    if args.job_dir:
        return Path(args.job_dir) / "gpu_provider_launch_request.json"
    raise ValueError("Provide --provider-launch-request or --job-dir")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an explicitly gated GPU provider launcher command."
    )
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--job-dir")
    parser.add_argument("--output-path")
    parser.add_argument("--provider-launch-command")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument(
        "--allow-provider-launch",
        action="store_true",
        help="Required with BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true for live providers.",
    )
    args = parser.parse_args(argv)

    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        output_path=args.output_path,
        allow_provider_launch=args.allow_provider_launch,
        provider_launch_command=args.provider_launch_command,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[robot-eval-provider-launcher] result={result['output_path']}")
    print(f"[robot-eval-provider-launcher] status={result['status']}")
    print(f"[robot-eval-provider-launcher] job_id={result.get('job_id')}")
    print(f"[robot-eval-provider-launcher] provider={result.get('provider')}")
    blockers = result.get("blockers")
    if blockers:
        print("[robot-eval-provider-launcher] blockers=" + ",".join(blockers))
    return 0 if result["status"] in {"completed", "not_required_for_fixture_local"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
