"""Safe local roots and admission checks for generated pipeline artifacts.

Source checkouts are durable code. Generated run output is disposable cache or
explicit evidence and must not silently accumulate beside the checkout.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ARTIFACT_CACHE_ROOT_ENV = "BLUEPRINT_ARTIFACT_CACHE_ROOT"
EVIDENCE_ROOT_ENV = "BLUEPRINT_EVIDENCE_ROOT"
ALLOW_REPO_OUTPUT_ENV = "BLUEPRINT_ALLOW_REPO_OUTPUT"
ALLOW_LARGE_ARTIFACTS_ENV = "BLUEPRINT_ALLOW_LARGE_ARTIFACTS"
REVIEW_THRESHOLD_BYTES = 25 * 1024**3
HARD_STOP_BYTES = 50 * 1024**3
LARGE_ARTIFACT_BYTES = 1 * 1024**3


class ArtifactStorageError(RuntimeError):
    """Raised when an artifact would violate the local storage policy."""


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _platform_cache_base() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def _platform_data_base() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")).expanduser()


def default_artifact_cache_root() -> Path:
    """Return the disposable generated-artifact cache root."""

    return (_env_path(ARTIFACT_CACHE_ROOT_ENV) or (_platform_cache_base() / "BlueprintCapturePipeline")).resolve()


def default_evidence_root() -> Path:
    """Return the separate retained-manifest/evidence root."""

    return (_env_path(EVIDENCE_ROOT_ENV) or (_platform_data_base() / "BlueprintCapturePipeline" / "evidence")).resolve()


def repo_output_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd().resolve()
    return root / "output"


def repo_output_allowed() -> bool:
    return os.environ.get(ALLOW_REPO_OUTPUT_ENV, "").strip().lower() in {"1", "true", "yes"}


def large_artifacts_allowed() -> bool:
    return os.environ.get(ALLOW_LARGE_ARTIFACTS_ENV, "").strip().lower() in {"1", "true", "yes"}


def path_is_within(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(Path(root).expanduser().resolve())
    except ValueError:
        return False
    return True


def directory_size(root: str | Path) -> int:
    path = Path(root).expanduser()
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def storage_status(root: str | Path, *, incoming_bytes: int = 0) -> str:
    total = directory_size(root) + max(0, incoming_bytes)
    if total >= HARD_STOP_BYTES:
        return "hard_stop"
    if total >= REVIEW_THRESHOLD_BYTES:
        return "review"
    return "ok"


def assert_artifact_write_allowed(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    estimated_bytes: int = 0,
    allow_repo_output: bool | None = None,
    allow_large_artifacts: bool | None = None,
) -> Path:
    """Validate a generated-artifact destination before creating it.

    Explicit ``--output`` paths remain supported. Repo-root ``output/`` and
    artifacts at or above one GiB require an explicit opt-in so old workflows
    fail clearly instead of filling a developer disk silently.
    """

    destination = Path(path).expanduser().resolve()
    repo_root_path = Path(repo_root).expanduser().resolve() if repo_root else None
    if repo_root_path is not None and path_is_within(destination, repo_output_root(repo_root_path)):
        permitted = repo_output_allowed() if allow_repo_output is None else allow_repo_output
        if not permitted:
            raise ArtifactStorageError(
                f"artifact destination is inside repo output/: {destination}; "
                f"use {ARTIFACT_CACHE_ROOT_ENV} or set {ALLOW_REPO_OUTPUT_ENV}=1 for legacy output"
            )
    if estimated_bytes >= LARGE_ARTIFACT_BYTES:
        permitted = large_artifacts_allowed() if allow_large_artifacts is None else allow_large_artifacts
        if not permitted:
            raise ArtifactStorageError(
                f"large artifact requires explicit opt-in ({estimated_bytes} bytes); "
                f"set {ALLOW_LARGE_ARTIFACTS_ENV}=1"
            )
    root = default_artifact_cache_root()
    if path_is_within(destination, root) and storage_status(root, incoming_bytes=estimated_bytes) == "hard_stop":
        raise ArtifactStorageError(
            f"artifact cache hard stop reached at {root}; run the retention tool before writing more"
        )
    return destination


def cache_run_root(run_name: str) -> Path:
    """Return a namespaced disposable root for one local run."""

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name.strip()).strip("-.")
    if not safe_name:
        raise ValueError("run_name must not be empty")
    return default_artifact_cache_root() / safe_name


__all__ = [
    "ALLOW_LARGE_ARTIFACTS_ENV",
    "ALLOW_REPO_OUTPUT_ENV",
    "ARTIFACT_CACHE_ROOT_ENV",
    "ArtifactStorageError",
    "EVIDENCE_ROOT_ENV",
    "HARD_STOP_BYTES",
    "LARGE_ARTIFACT_BYTES",
    "REVIEW_THRESHOLD_BYTES",
    "assert_artifact_write_allowed",
    "cache_run_root",
    "default_artifact_cache_root",
    "default_evidence_root",
    "directory_size",
    "large_artifacts_allowed",
    "path_is_within",
    "repo_output_allowed",
    "repo_output_root",
    "storage_status",
]
