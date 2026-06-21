"""Helpers for normalizing model-access environment variables."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Sequence


HF_TOKEN_ENV_ALIASES = (
    "HUGGINGFACE_HUB_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
HF_TOKEN_FILE_ENV_ALIASES = (
    "HUGGINGFACE_HUB_TOKEN_FILE",
    "HUGGINGFACE_TOKEN_FILE",
    "HUGGING_FACE_HUB_TOKEN_FILE",
    "HF_TOKEN_FILE",
)
DEFAULT_HF_TOKEN_FILES = (
    "~/.blueprint-secrets/huggingface_token",
    "~/.blueprint-secrets/hf_token",
    "~/.blueprint-secrets/huggingface_token.txt",
    "~/.blueprint-secrets/hf_token.txt",
)
NGC_TOKEN_ENV_ALIASES = (
    "NGC_API_KEY",
    "NVIDIA_NGC_API_KEY",
)
NGC_TOKEN_FILE_ENV_ALIASES = (
    "NGC_API_KEY_FILE",
    "NVIDIA_NGC_API_KEY_FILE",
)
DEFAULT_NGC_TOKEN_FILES = ("~/.blueprint-secrets/ngc_api_key",)
DISABLE_DEFAULT_SECRET_FILES_ENV = "BLUEPRINT_DISABLE_DEFAULT_MODEL_SECRET_FILES"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _first_env_value(names: Sequence[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return name, value
    return None, None


def _candidate_file_paths(
    *,
    file_env_names: Sequence[str],
    default_paths: Sequence[str],
) -> list[tuple[str | None, Path, bool]]:
    candidates: list[tuple[str | None, Path, bool]] = []
    for env_name in file_env_names:
        configured = str(os.getenv(env_name, "")).strip()
        if configured:
            candidates.append((env_name, Path(configured).expanduser(), True))
    if not _truthy(os.getenv(DISABLE_DEFAULT_SECRET_FILES_ENV)):
        for item in default_paths:
            candidates.append((None, Path(item).expanduser(), False))
    return candidates


def _read_first_secret_file(
    *,
    file_env_names: Sequence[str],
    default_paths: Sequence[str],
) -> tuple[str | None, Path | None, str | None]:
    for env_name, path, _configured in _candidate_file_paths(
        file_env_names=file_env_names,
        default_paths=default_paths,
    ):
        if not path.is_file():
            continue
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value, path, env_name
    return None, None, None


def _file_status(
    *,
    path: Path,
    env_name: str | None,
    configured_by_env: bool,
) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    mode = None
    size_bytes = None
    if exists:
        try:
            st = path.stat()
            mode = oct(stat.S_IMODE(st.st_mode))
            size_bytes = st.st_size if is_file else None
        except OSError:
            mode = None
            size_bytes = None
    return {
        "env_name": env_name,
        "configured_by_env": configured_by_env,
        "path": str(path),
        "present": bool(is_file),
        "mode": mode,
        "size_bytes": size_bytes,
        "permission_recommended": "0600",
        "secret_value_written_to_artifacts": False,
        "secret_hash_written_to_artifacts": False,
    }


def _secret_group_status(
    *,
    group_id: str,
    env_names: Sequence[str],
    file_env_names: Sequence[str],
    default_paths: Sequence[str],
) -> dict[str, Any]:
    env_name, _env_value = _first_env_value(env_names)
    file_candidates = [
        _file_status(path=path, env_name=file_env_name, configured_by_env=configured)
        for file_env_name, path, configured in _candidate_file_paths(
            file_env_names=file_env_names,
            default_paths=default_paths,
        )
    ]
    ready_file = next((row for row in file_candidates if row["present"]), None)
    return {
        "group_id": group_id,
        "env_aliases": list(env_names),
        "env_value_configured": bool(env_name),
        "configured_env": env_name,
        "file_env_aliases": list(file_env_names),
        "file_candidates": file_candidates,
        "file_secret_configured": bool(ready_file),
        "selected_file_path": ready_file.get("path") if ready_file else None,
        "auth_ready": bool(env_name or ready_file),
        "raw_secret_written_to_artifacts": False,
        "secret_hash_written_to_artifacts": False,
    }


def normalize_model_access_env() -> None:
    """Bridge common auth env aliases used across VM bootstrap and runtime paths."""
    _hf_env_name, hf_token = _first_env_value(HF_TOKEN_ENV_ALIASES)
    if not hf_token:
        hf_token, _hf_path, _hf_file_env = _read_first_secret_file(
            file_env_names=HF_TOKEN_FILE_ENV_ALIASES,
            default_paths=DEFAULT_HF_TOKEN_FILES,
        )
    if hf_token:
        for name in HF_TOKEN_ENV_ALIASES:
            if os.getenv(name):
                os.environ[name] = hf_token
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    _ngc_env_name, ngc_token = _first_env_value(NGC_TOKEN_ENV_ALIASES)
    if not ngc_token:
        ngc_token, _ngc_path, _ngc_file_env = _read_first_secret_file(
            file_env_names=NGC_TOKEN_FILE_ENV_ALIASES,
            default_paths=DEFAULT_NGC_TOKEN_FILES,
        )
    if ngc_token:
        for name in NGC_TOKEN_ENV_ALIASES:
            if os.getenv(name):
                os.environ[name] = ngc_token
        os.environ.setdefault("NGC_API_KEY", ngc_token)
        os.environ.setdefault("NVIDIA_NGC_API_KEY", ngc_token)


def model_access_secret_status() -> dict[str, Any]:
    """Return non-secret readiness details for model download/container auth."""
    hf_status = _secret_group_status(
        group_id="huggingface",
        env_names=HF_TOKEN_ENV_ALIASES,
        file_env_names=HF_TOKEN_FILE_ENV_ALIASES,
        default_paths=DEFAULT_HF_TOKEN_FILES,
    )
    ngc_status = _secret_group_status(
        group_id="ngc",
        env_names=NGC_TOKEN_ENV_ALIASES,
        file_env_names=NGC_TOKEN_FILE_ENV_ALIASES,
        default_paths=DEFAULT_NGC_TOKEN_FILES,
    )
    return {
        "schema_version": "model_access_secret_status.v1",
        "status": "ready" if hf_status["auth_ready"] or ngc_status["auth_ready"] else "blocked",
        "huggingface": hf_status,
        "ngc": ngc_status,
        "raw_secret_written_to_artifacts": False,
        "secret_hash_written_to_artifacts": False,
        "claim_boundary": {
            "tokens_provide_model_or_container_access_only": True,
            "tokens_do_not_provide_gpu_compute": True,
        },
    }
