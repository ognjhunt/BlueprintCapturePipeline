"""Fail-closed admission and systemd trigger controls for live intake."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess  # nosec B404 - fixed systemctl argv with strict unit allowlists
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Mapping

from fastapi import Depends, HTTPException, Request, status

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .core.security_controls import json_shape_within_limits
from .live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV


DEFAULT_MANIFEST_PATH = (
    "/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json"
)
INTAKE_WORK_DIR_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR"
INTAKE_TRIGGER_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_TRIGGER_COMMAND"
INTAKE_ALLOW_TRIGGER_ENV = "BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER"
INTAKE_TRIGGER_SYSTEMD_UNIT_ENV = "BLUEPRINT_LIVE_PIPELINE_TRIGGER_SYSTEMD_UNIT"
TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_DIR"
TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT"
)
TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV = "BLUEPRINT_ALLOW_TASK_EVALUATION_LAUNCH_TRIGGER"
TASK_EVALUATION_LAUNCH_TRIGGER_MODE_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_TRIGGER_MODE"
TASK_EVALUATION_LAUNCH_PATH_UNIT = "blueprint-task-evaluation-launch-dispatcher.path"
TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER_SYSTEMD_UNIT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER_SYSTEMD_UNIT"
)
TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_ALLOW_TRIGGER_ENV = (
    "BLUEPRINT_ALLOW_TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER"
)
INTAKE_MAX_BODY_BYTES_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_BODY_BYTES"
INTAKE_MAX_JSON_DEPTH_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_JSON_DEPTH"
INTAKE_MAX_JSON_ITEMS_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_JSON_ITEMS"
INTAKE_RATE_LIMIT_PER_MINUTE_ENV = "BLUEPRINT_LIVE_PIPELINE_RATE_LIMIT_PER_MINUTE"
INTAKE_MAX_CONCURRENT_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_CONCURRENT"
INTAKE_MAX_QUEUE_FILES_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_QUEUE_FILES"
INTAKE_MAX_STORAGE_BYTES_ENV = "BLUEPRINT_LIVE_PIPELINE_MAX_STORAGE_BYTES"
DEFAULT_INTAKE_MAX_BODY_BYTES = 2 * 1024 * 1024
DEFAULT_INTAKE_MAX_JSON_DEPTH = 32
DEFAULT_INTAKE_MAX_JSON_ITEMS = 100_000
DEFAULT_INTAKE_RATE_LIMIT_PER_MINUTE = 120
DEFAULT_INTAKE_MAX_CONCURRENT = 8
DEFAULT_INTAKE_MAX_QUEUE_FILES = 10_000
DEFAULT_INTAKE_MAX_STORAGE_BYTES = 20 * 1024 * 1024 * 1024


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(_string(os.getenv(name)))
    except ValueError:
        return default
    return value if value > 0 else default


def _read_mapping_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _work_root() -> Path:
    manifest_path = Path(
        os.getenv(CONTROL_PLANE_OUTPUT_PATH_ENV) or DEFAULT_MANIFEST_PATH
    ).expanduser()
    configured = _string(os.getenv(INTAKE_WORK_DIR_ENV))
    if configured:
        return Path(configured).expanduser().resolve()
    return (manifest_path.parent / "incoming_webapp_job_requests").resolve()


def _trigger_control_plane() -> Dict[str, Any]:
    unit = _string(os.getenv(INTAKE_TRIGGER_SYSTEMD_UNIT_ENV))
    allowed = _truthy(os.getenv(INTAKE_ALLOW_TRIGGER_ENV))
    if not unit:
        return {
            "status": "not_configured",
            "performed": False,
            "allowed": allowed,
            "systemd_unit_configured": False,
        }
    if not allowed:
        return {
            "status": "blocked",
            "performed": False,
            "allowed": False,
            "systemd_unit_configured": True,
            "blockers": [f"missing_env_{INTAKE_ALLOW_TRIGGER_ENV}"],
        }
    if not re.fullmatch(r"[A-Za-z0-9@_.-]+\.service", unit):
        return {
            "status": "blocked",
            "performed": False,
            "allowed": True,
            "systemd_unit_configured": True,
            "blockers": ["intake_trigger_systemd_unit_invalid"],
        }
    command_argv = ["systemctl", "start", "--no-block", unit]
    completed = subprocess.run(  # nosec B603
        command_argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "triggered" if completed.returncode == 0 else "failed",
        "performed": completed.returncode == 0,
        "allowed": True,
        "systemd_unit_configured": True,
        "systemd_unit": unit,
        "command_argv_count": len(command_argv),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _trigger_task_evaluation_launch_dispatcher() -> Dict[str, Any]:
    unit = _string(os.getenv(TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV))
    mode = _string(os.getenv(TASK_EVALUATION_LAUNCH_TRIGGER_MODE_ENV)) or "systemctl"
    allowed = _truthy(os.getenv(TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV))
    profile_dir = _string(os.getenv(TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV))
    blockers: list[str] = []
    if not profile_dir:
        blockers.append(f"missing_env_{TASK_EVALUATION_LAUNCH_PROFILE_DIR_ENV}")
    elif not Path(profile_dir).expanduser().resolve().is_dir():
        blockers.append("task_evaluation_launch_profile_dir_missing")
    if mode not in {"systemctl", "systemd_path"}:
        blockers.append(f"invalid_env_{TASK_EVALUATION_LAUNCH_TRIGGER_MODE_ENV}")
    if mode == "systemctl" and not unit:
        blockers.append(f"missing_env_{TASK_EVALUATION_LAUNCH_TRIGGER_SYSTEMD_UNIT_ENV}")
    elif mode == "systemctl" and not re.fullmatch(r"[A-Za-z0-9@_.-]+\.service", unit):
        blockers.append("task_evaluation_launch_trigger_systemd_unit_invalid")
    if not allowed:
        blockers.append(f"missing_env_{TASK_EVALUATION_LAUNCH_ALLOW_TRIGGER_ENV}")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "allowed": allowed,
            "blockers": sorted(set(blockers)),
        }
    if mode == "systemd_path":
        observed: dict[str, str] = {}
        for probe in ("is-enabled", "is-active"):
            completed = subprocess.run(  # nosec B603 B607 - fixed read-only argv
                ["systemctl", probe, TASK_EVALUATION_LAUNCH_PATH_UNIT],
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            observed[probe] = completed.stdout.strip() or (
                "disabled" if probe == "is-enabled" else "inactive"
            )
        if observed["is-active"] != "active":
            return {
                "status": "blocked",
                "performed": False,
                "allowed": True,
                "trigger_mode": mode,
                "systemd_path_unit": TASK_EVALUATION_LAUNCH_PATH_UNIT,
                "systemd_path_enabled_state": observed["is-enabled"],
                "systemd_path_active_state": observed["is-active"],
                "blockers": ["task_evaluation_launch_systemd_path_inactive"],
                "provider_mutation_performed": False,
            }
        return {
            "status": "armed_by_systemd_path",
            "performed": True,
            "allowed": True,
            "trigger_mode": mode,
            "systemd_path_unit": TASK_EVALUATION_LAUNCH_PATH_UNIT,
            "systemd_path_enabled_state": observed["is-enabled"],
            "systemd_path_active_state": observed["is-active"],
            "provider_mutation_performed": False,
        }
    command_argv = ["systemctl", "start", "--no-block", unit]
    completed = subprocess.run(  # nosec B603 - fixed executable plus strict unit allowlist
        command_argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "triggered" if completed.returncode == 0 else "failed",
        "performed": completed.returncode == 0,
        "allowed": True,
        "systemd_unit": unit,
        "command_argv_count": len(command_argv),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _trigger_task_evaluation_terminal_resource_release_dispatcher() -> Dict[str, Any]:
    """Start the independent release-only worker; no provider work occurs in HTTP."""

    unit = _string(os.getenv(TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER_SYSTEMD_UNIT_ENV))
    allowed = _truthy(os.getenv(TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_ALLOW_TRIGGER_ENV))
    if not unit:
        return {
            "status": "blocked",
            "performed": False,
            "allowed": allowed,
            "blockers": [
                f"missing_env_{TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER_SYSTEMD_UNIT_ENV}"
            ],
        }
    if not allowed:
        return {
            "status": "blocked",
            "performed": False,
            "allowed": False,
            "blockers": [
                f"missing_env_{TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_ALLOW_TRIGGER_ENV}"
            ],
        }
    if not re.fullmatch(r"[A-Za-z0-9@_.-]+\.service", unit):
        return {
            "status": "blocked",
            "performed": False,
            "allowed": True,
            "blockers": ["terminal_resource_release_trigger_systemd_unit_invalid"],
        }
    command_argv = ["systemctl", "start", "--no-block", unit]
    completed = subprocess.run(  # nosec B603 - fixed executable and strict unit allowlist
        command_argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "triggered" if completed.returncode == 0 else "failed",
        "performed": completed.returncode == 0,
        "allowed": True,
        "systemd_unit": unit,
        "command_argv_count": len(command_argv),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "provider_mutation_performed": False,
    }


def _intake_storage_usage(root: Path) -> tuple[int, int]:
    file_count = 0
    size_bytes = 0
    if not root.exists():
        return 0, 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        file_count += 1
        try:
            size_bytes += path.stat().st_size
        except OSError:
            continue
    return file_count, size_bytes


def _admission_state_paths() -> tuple[Path, Path]:
    root = _work_root() / ".admission"
    ensure_dir(root)
    root.chmod(0o700)
    return root / "state.json", root / "state.lock"


def _claim_intake_admission(client_id: str) -> str:
    state_path, lock_path = _admission_state_paths()
    work_root = _work_root()
    file_count, storage_bytes = _intake_storage_usage(work_root)
    if file_count >= _positive_int_env(INTAKE_MAX_QUEUE_FILES_ENV, DEFAULT_INTAKE_MAX_QUEUE_FILES):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="intake queue file quota exceeded",
        )
    if storage_bytes >= _positive_int_env(
        INTAKE_MAX_STORAGE_BYTES_ENV, DEFAULT_INTAKE_MAX_STORAGE_BYTES
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="intake storage quota exceeded",
        )
    now = time.time()
    lease_id = f"lease-{uuid.uuid4().hex}"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_payload = _read_mapping_file(state_path)
        rates = {
            str(key): [
                float(item)
                for item in value
                if isinstance(item, (int, float)) and float(item) > now - 60.0
            ]
            for key, value in _mapping(state_payload.get("rate_windows")).items()
            if isinstance(value, list)
        }
        active = {
            str(key): dict(value)
            for key, value in _mapping(state_payload.get("active_leases")).items()
            if isinstance(value, Mapping)
            and float(value.get("started_at_epoch") or 0.0) > now - 600.0
        }
        client_window = rates.setdefault(client_id, [])
        if len(client_window) >= _positive_int_env(
            INTAKE_RATE_LIMIT_PER_MINUTE_ENV,
            DEFAULT_INTAKE_RATE_LIMIT_PER_MINUTE,
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="intake client rate limit exceeded",
            )
        if len(active) >= _positive_int_env(
            INTAKE_MAX_CONCURRENT_ENV,
            DEFAULT_INTAKE_MAX_CONCURRENT,
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="intake concurrency quota exceeded",
            )
        client_window.append(now)
        active[lease_id] = {
            "client_id_sha256": sha256(client_id.encode("utf-8")).hexdigest(),
            "started_at_epoch": now,
        }
        write_json(
            state_path,
            {
                "schema_version": "blueprint_live_intake_admission_state.v1",
                "updated_at": utc_now_iso(),
                "rate_windows": rates,
                "active_leases": active,
            },
        )
    return lease_id


def _release_intake_admission(lease_id: str) -> None:
    state_path, lock_path = _admission_state_paths()
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_payload = _read_mapping_file(state_path)
        active = _mapping(state_payload.get("active_leases"))
        active.pop(lease_id, None)
        write_json(
            state_path,
            {
                **state_payload,
                "schema_version": "blueprint_live_intake_admission_state.v1",
                "updated_at": utc_now_iso(),
                "active_leases": active,
            },
        )


def build_require_admission(
    require_token: Callable[..., Awaitable[str]],
) -> Callable[..., AsyncIterator[str]]:
    """Bind admission limits to the service's authenticated-client dependency."""

    async def require_admission(
        request: Request,
        client_id: str = Depends(require_token),
    ) -> AsyncIterator[str]:
        body = await request.body()
        if len(body) > _positive_int_env(INTAKE_MAX_BODY_BYTES_ENV, DEFAULT_INTAKE_MAX_BODY_BYTES):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="intake request body exceeds byte limit",
            )
        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None and not json_shape_within_limits(
                parsed,
                max_depth=_positive_int_env(
                    INTAKE_MAX_JSON_DEPTH_ENV, DEFAULT_INTAKE_MAX_JSON_DEPTH
                ),
                max_items=_positive_int_env(
                    INTAKE_MAX_JSON_ITEMS_ENV, DEFAULT_INTAKE_MAX_JSON_ITEMS
                ),
            ):
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="intake JSON depth or item limit exceeded",
                )
        lease_id = _claim_intake_admission(client_id)
        try:
            yield client_id
        finally:
            _release_intake_admission(lease_id)

    return require_admission
