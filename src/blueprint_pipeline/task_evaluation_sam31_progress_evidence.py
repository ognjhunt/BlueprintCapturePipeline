"""Immutable SAM precursor progress and exact-file evidence readers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

PROGRESS_SCHEMA = "task_evaluation_sam31_preparation_progress.v1"


class Sam31PreparationQueueError(ValueError):
    """A precursor checkpoint or resume signal failed its immutable bindings."""


def _require(value: bool, code: str) -> None:
    if not value:
        raise Sam31PreparationQueueError("sam31_preparation_" + code)


def _read(path: Path) -> dict:
    _require(not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file() and path.stat().st_size <= 4 * 1024 * 1024,
             "record_path_invalid")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def verify_evidence_reference(row: Mapping[str, Any], roots: Sequence[Path]) -> Path:
    _require(isinstance(row, Mapping), "evidence_invalid")
    raw = row.get("path")
    _require(isinstance(raw, str) and Path(raw).is_absolute() and ".." not in Path(raw).parts,
             "evidence_path_invalid")
    path = Path(raw)
    _require(not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file()
             and any(path.resolve().is_relative_to(root.resolve()) for root in roots),
             "evidence_path_invalid")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    size = row.get("size_bytes")
    _require(type(size) is int and size > 0 and count == size
             and "sha256:" + digest.hexdigest() == row.get("sha256", row.get("digest")),
             "evidence_readback_mismatch")
    return path


def load_progress(root: Path, filename: str, request_digest: str) -> dict | None:
    directory = root / "source-progress" / Path(filename).stem
    if not directory.exists():
        return None
    _require(not directory.is_symlink(), "progress_path_invalid")
    prior = None
    for index, path in enumerate(sorted(directory.glob("*.json")), 1):
        value = _read(path)
        _require(value.get("schema_version") == PROGRESS_SCHEMA
                 and value.get("request_digest") == request_digest
                 and value.get("sequence") == index
                 and value.get("previous_progress_digest") == (prior["progress_digest"] if prior else None)
                 and value.get("progress_digest") == canonical_digest(value, digest_field="progress_digest"),
                 "progress_chain_invalid")
        prior = value
    return prior
