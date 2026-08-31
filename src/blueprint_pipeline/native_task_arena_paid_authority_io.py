"""Byte/path helpers shared by native Task Arena paid authority contracts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir
from .task_evaluation_immutable_input_resolver import (
    ImmutableInputResolutionError,
    resolve_immutable_input,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def lexical_absolute_path(value: Any, code: str) -> Path:
    raw = str(value or "")
    expanded = Path(raw).expanduser()
    if not raw or not expanded.is_absolute():
        raise ValueError(code)
    return Path(os.path.abspath(str(expanded)))


def recorded_path(value: Mapping[str, Any], code: str) -> Path:
    """Return the source path sealed by one byte-verified record."""

    return lexical_absolute_path(value.get("path"), code)


def read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def bound_record(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    try:
        path = resolve_immutable_input(
            str(value.get("path") or ""),
            expected_digest=str(value.get("sha256") or ""),
            expected_size_bytes=value.get("size_bytes"),
        )
    except ImmutableInputResolutionError as exc:
        raise ValueError(code) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def bound_record_matches_observed(
    bound: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    """Compare a sealed source record with a validator's staged readback."""

    normalized_observed = dict(observed)
    normalized_observed["path"] = bound.get("path")
    return dict(bound) == normalized_observed


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one immutable closeout member without replacing existing evidence."""

    ensure_dir(path.parent)
    payload = (json.dumps(dict(value), indent=1, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def lower_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "bound_record",
    "bound_record_matches_observed",
    "lexical_absolute_path",
    "lower_hex",
    "read",
    "record",
    "recorded_path",
    "sha256",
    "write_exclusive_json",
]
