"""Validation for artifacts emitted by injected supervisor runtimes."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any


def validated_emitted_artifact(
    *, root: str | Path, relative_path: Any, expected_digest: Any
) -> Path:
    root_path = Path(root).resolve(strict=True)
    text = str(relative_path or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ValueError("emitted_artifact_relative_path_unsafe")
    candidate = root_path.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ValueError("emitted_artifact_symlink_forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("emitted_artifact_missing") from exc
    if root_path not in resolved.parents or not resolved.is_file():
        raise ValueError("emitted_artifact_escape_or_not_file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if expected_digest != f"sha256:{digest.hexdigest()}":
        raise ValueError("emitted_artifact_digest_mismatch")
    return resolved


__all__ = ["validated_emitted_artifact"]
