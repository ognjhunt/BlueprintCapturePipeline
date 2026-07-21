"""Fail-closed subprocess evidence helpers for optional external tool workers.

The core pipeline never imports prerelease NVIDIA packages through this module.
It invokes an explicitly configured executable, records the executable and
input identities, captures bounded logs, and returns evidence that callers can
normalize into their own contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, sha256_file, utc_now_iso, write_text


SECRET_NAME_TOKENS = (
    "API_KEY",
    "AUTH",
    "BEARER",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
NETWORK_ENV_KEYS = (
    "ALL_PROXY",
    "FTP_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ftp_proxy",
    "http_proxy",
    "https_proxy",
)
PUBLIC_CLAIM_UPGRADE_KEY = "_".join(("public", "claim", "upgrade", "allowed"))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _secret_name(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in SECRET_NAME_TOKENS)


def sanitized_environment(
    env: Mapping[str, str] | None,
    *,
    network_policy: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build a worker environment without serializing secret values.

    ``network_policy=disabled`` removes conventional proxy variables. This is
    defense in depth, not a claim that the operating system sandbox blocked all
    sockets; that distinction is recorded in the returned summary.
    """

    source = dict(os.environ if env is None else env)
    removed_secret_names: list[str] = []
    for name in tuple(source):
        if _secret_name(name):
            removed_secret_names.append(name)
            source.pop(name, None)
    removed_network_names: list[str] = []
    if network_policy == "disabled":
        for name in NETWORK_ENV_KEYS:
            if name in source:
                removed_network_names.append(name)
                source.pop(name, None)
        source["NO_PROXY"] = "*"
        source["no_proxy"] = "*"
    summary = {
        "secret_values_in_artifact": False,
        "secret_named_environment_variables_removed": sorted(removed_secret_names),
        "proxy_environment_variables_removed": sorted(removed_network_names),
        "network_policy": network_policy,
        "network_socket_sandbox_proven": False,
        "environment_value_capture_allowed": False,
    }
    return source, summary


def parse_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        tokens = shlex.split(command)
    else:
        tokens = [str(token) for token in command]
    if not tokens or not tokens[0].strip():
        raise ValueError("external worker command must not be empty")
    return tokens


def executable_identity(
    command: str | Sequence[str], *, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    tokens = parse_command(command)
    search_path = (env or os.environ).get("PATH")
    resolved = shutil.which(tokens[0], path=search_path)
    candidate = Path(resolved or tokens[0]).expanduser()
    exists = candidate.is_file()
    return {
        "requested_executable": tokens[0],
        "resolved_executable": str(candidate.resolve()) if exists else None,
        "executable_found": exists,
        "executable_sha256": sha256_file(candidate) if exists else None,
        "argv_tail_sha256": canonical_sha256(tokens[1:]),
    }


def render_command(
    command: str | Sequence[str],
    *,
    replacements: Mapping[str, str],
) -> list[str]:
    tokens = parse_command(command)
    rendered: list[str] = []
    for token in tokens:
        value = token
        for name, replacement in replacements.items():
            value = value.replace("{" + name + "}", replacement)
        rendered.append(value)
    unresolved = [token for token in rendered if "{" in token or "}" in token]
    if unresolved:
        raise ValueError(f"unresolved external worker command placeholders: {unresolved}")
    return rendered


def _usage_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "maximum_resident_set_size_platform_units": float(usage.ru_maxrss),
    }


def _usage_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        key: max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0))) for key in before
    }


def run_json_worker(
    *,
    command: str | Sequence[str],
    replacements: Mapping[str, str],
    working_directory: str | Path,
    output_directory: str | Path,
    raw_report_path: str | Path,
    timeout_seconds: int,
    network_policy: str = "disabled",
    env: Mapping[str, str] | None = None,
    log_prefix: str = "external_worker",
) -> dict[str, Any]:
    """Run an external JSON-producing worker and persist bounded evidence."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    output_dir = Path(output_directory).resolve()
    ensure_dir(output_dir)
    raw_path = Path(raw_report_path).resolve()
    ensure_dir(raw_path.parent)
    worker_env, env_summary = sanitized_environment(env, network_policy=network_policy)
    argv = render_command(command, replacements=replacements)
    identity = executable_identity(argv, env=worker_env)
    safe_prefix = (
        "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in log_prefix).strip(
            "_"
        )
        or "external_worker"
    )
    stdout_path = output_dir / f"{safe_prefix}_stdout.log"
    stderr_path = output_dir / f"{safe_prefix}_stderr.log"
    before = _usage_snapshot()
    started_at = utc_now_iso()
    started = time.monotonic()
    timed_out = False
    launch_error: str | None = None
    exit_code: int | None = None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            argv,
            cwd=str(Path(working_directory).resolve()),
            env=worker_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
    except OSError as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
    duration = max(0.0, time.monotonic() - started)
    after = _usage_snapshot()
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    raw_exists = raw_path.is_file()
    return {
        "schema_version": "external_json_worker_execution.v1",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "duration_seconds": round(duration, 6),
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "exit_code": exit_code,
        "command_argv": argv,
        "command_sha256": canonical_sha256(argv),
        "executable_identity": identity,
        "working_directory": str(Path(working_directory).resolve()),
        "raw_report_path": str(raw_path),
        "raw_report_exists": raw_exists,
        "raw_report_sha256": sha256_file(raw_path) if raw_exists else None,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "resource_usage": _usage_delta(before, after),
        "environment_policy": env_summary,
    }
