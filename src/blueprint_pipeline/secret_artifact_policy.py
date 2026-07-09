"""Release-safe metadata helpers for local secret files.

Provider and launch-readiness artifacts often need to prove that a local
operator has configured a credential file and that its permissions are sane.
Those artifacts must not publish absolute workstation paths to the credential
files. The helpers here keep the useful diagnostics while redacting the path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SECRET_PATH_DISCLOSURE_POLICY = "local_secret_file_paths_redacted_from_release_artifacts"


def secret_path_disclosure_policy() -> dict[str, Any]:
    return {
        "local_secret_file_paths_recorded": False,
        "path_disclosure_policy": SECRET_PATH_DISCLOSURE_POLICY,
        "publishable_release_artifacts_must_not_include_local_secret_paths": True,
    }


def redacted_secret_file_status(
    path: str | Path,
    *,
    env_name: str | None = None,
    env_field: str = "env_name",
    configured_by_env: bool | None = None,
    path_source: str | None = None,
    raw_secret_field: str = "raw_secret_values_recorded",
) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    mode = None
    if resolved.exists():
        try:
            mode = oct(resolved.stat().st_mode & 0o777)
        except OSError:
            mode = None
    status: dict[str, Any] = {
        "path_redacted": True,
        "path_disclosure_policy": SECRET_PATH_DISCLOSURE_POLICY,
        "present": resolved.is_file(),
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        raw_secret_field: False,
    }
    if env_name is not None:
        status[env_field] = env_name
    if configured_by_env is not None:
        status["configured_by_env"] = bool(configured_by_env)
    if path_source:
        status["path_source"] = path_source
    return status


def redacted_secret_file_status_from_env(
    env_name: str,
    default_path: str,
    *,
    env_field: str = "env_name",
    raw_secret_field: str = "raw_secret_values_recorded",
) -> dict[str, Any]:
    import os

    configured = str(os.getenv(env_name, "")).strip()
    selected = configured or default_path
    return redacted_secret_file_status(
        selected,
        env_name=env_name,
        env_field=env_field,
        configured_by_env=bool(configured),
        path_source="env" if configured else "default_blueprint_secret_file_path",
        raw_secret_field=raw_secret_field,
    )
