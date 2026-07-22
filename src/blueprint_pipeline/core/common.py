"""Provider- and campaign-neutral filesystem and contract helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Tuple


# Shared readiness thresholds. Kept here so the numpy-free reviewer path
# (agent_runtime.orchestrator) and the qualification capability envelope
# stay in lock-step on a single source of truth.
MAXIMUM_HIDDEN_ZONE_BOUND = 0.35

__all__ = [
    "GCSUri",
    "MAXIMUM_HIDDEN_ZONE_BOUND",
    "PipelineError",
    "StageError",
    "ensure_dir",
    "ensure_local_uri_path",
    "flatten_scene_paths",
    "has_nonempty_file",
    "infer_storage_root_from_scene_path",
    "is_gs_uri",
    "join_gs_uri",
    "maybe_as_dict",
    "maybe_as_list",
    "optional_read_json",
    "parse_bool",
    "parse_gs_uri",
    "read_json",
    "read_json_any",
    "relative_scene_path",
    "resolve_gs_uri_to_path",
    "sha256_file",
    "to_capture_prefix",
    "to_pipeline_prefix",
    "to_scene_prefix",
    "try_parse_float",
    "try_parse_int",
    "utc_now_iso",
    "write_json",
    "write_text",
]


@dataclass(frozen=True)
class GCSUri:
    """Parsed ``gs://`` URI."""

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.key}"


class PipelineError(RuntimeError):
    """Raised for fatal orchestration failures."""


class StageError(PipelineError):
    """Raised when a specific pipeline stage fails."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gs_uri(uri: str) -> GCSUri:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    remainder = uri[5:]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return GCSUri(bucket=bucket, key=key)


def is_gs_uri(value: str) -> bool:
    return value.startswith("gs://")


def resolve_gs_uri_to_path(uri: str, gcs_root: Path) -> Path:
    """Resolve a ``gs://`` URI against mounted GCS roots.

    Supports both mount layouts:
    1) ``/mnt/gcs/<bucket>/<key>``
    2) ``/mnt/gcs/<key>`` (single-bucket mount)
    """

    parsed = parse_gs_uri(uri)
    candidate_bucket = gcs_root / parsed.bucket / parsed.key
    candidate_flat = gcs_root / parsed.key

    if candidate_flat.exists():
        return candidate_flat
    if candidate_bucket.exists():
        # If gcs_root is already bucket-root (for example /mnt/gcs/<bucket>),
        # stale nested directories can make candidate_bucket "exist" incorrectly.
        if gcs_root.name == parsed.bucket:
            return candidate_flat
        return candidate_bucket

    bucket_dir = gcs_root / parsed.bucket
    if gcs_root.name == parsed.bucket:
        return candidate_flat
    # Prefer bucket layout only when the provided root is a mount that already
    # has per-bucket subdirectories. Otherwise prefer flat/single-bucket layout.
    if bucket_dir.exists() and bucket_dir.is_dir():
        return candidate_bucket
    return candidate_flat


def ensure_local_uri_path(
    uri: str,
    *,
    gcs_root: Path,
    scratch_dir: Path | None = None,
) -> Path:
    """Return a local path for a URI, downloading from GCS when needed."""

    if is_gs_uri(uri):
        candidate = resolve_gs_uri_to_path(uri, gcs_root)
        if candidate.exists():
            return candidate
        parsed = parse_gs_uri(uri)
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise FileNotFoundError(
                f"GCS URI is not mounted locally and google-cloud-storage is unavailable: {uri}"
            ) from exc

        ext = Path(parsed.key).suffix
        target_root = scratch_dir or (gcs_root / ".downloads")
        ensure_dir(target_root)
        with tempfile.NamedTemporaryFile(
            dir=target_root,
            suffix=ext,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        client = gcs_storage.Client()
        client.bucket(parsed.bucket).blob(parsed.key).download_to_filename(str(temp_path))
        return temp_path

    return Path(uri)


def infer_storage_root_from_scene_path(path: Path) -> Path:
    """Infer mounted bucket root from a concrete scene path."""

    parts = path.parts
    if "scenes" not in parts:
        raise ValueError(f"Cannot infer storage root from non-scene path: {path}")
    idx = parts.index("scenes")
    if idx == 0:
        return Path("/")
    return Path(*parts[:idx])


def ensure_dir(path: Path) -> None:
    # This primitive intentionally creates the exact caller-authorized path;
    # request-facing callers must apply their own containment boundary first.
    path.mkdir(parents=True, exist_ok=True)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for a directory entry update."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        # Some filesystems/platforms do not support fsync on directories. The
        # file itself has already been flushed before replace in that case.
        pass
    finally:
        os.close(directory_fd)


def _atomic_write_text(path: Path, content: str) -> None:
    """Commit UTF-8 text without ever exposing a partially truncated target."""

    ensure_dir(path.parent)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), indent=2)
    content = content.replace("<redacted:secret-field>", "REDACTED_SECRET_FIELD")
    content = content.replace("<redacted:secret>", "REDACTED_SECRET")
    _atomic_write_text(path, content)


def write_text(path: Path, content: str) -> None:
    _atomic_write_text(path, content)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data).__name__}")
    return data


def read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def optional_read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def has_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def to_scene_prefix(scene_id: str) -> str:
    return f"scenes/{scene_id}"


def to_capture_prefix(scene_id: str, capture_id: str) -> str:
    return f"scenes/{scene_id}/captures/{capture_id}"


def to_pipeline_prefix(scene_id: str, capture_id: str) -> str:
    return f"scenes/{scene_id}/captures/{capture_id}/pipeline"


def join_gs_uri(prefix_uri: str, relative_path: str) -> str:
    parsed = parse_gs_uri(prefix_uri)
    base = parsed.key.rstrip("/")
    rel = relative_path.lstrip("/")
    return f"gs://{parsed.bucket}/{base}/{rel}"


def try_parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def try_parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def relative_scene_path(path: Path, storage_root: Path) -> str:
    return path.relative_to(storage_root).as_posix()


def maybe_as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def maybe_as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n"}:
            return False
    return default


def flatten_scene_paths(storage_root: Path, *relative_paths: str) -> Tuple[Path, ...]:
    return tuple(storage_root / rel for rel in relative_paths)
