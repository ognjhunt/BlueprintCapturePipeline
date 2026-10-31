"""Sealed file records for beta screening inputs.

A record names an absolute path, its byte size and SHA-256. ``_bound_path``
reopens the exact bytes a registration or plan sealed and refuses a symlink or
any change. Screening launch, planning and the effective catalog share this one
implementation; it imports nothing that could launch, allocate or call a
provider.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("beta_screening_record_invalid")
    return value


def _record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "size_bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}


def _bound_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path") or ""))
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or _record(path)["sha256"] != record.get("sha256")
            or ("size_bytes" in record and path.stat().st_size != record["size_bytes"])):
        raise ValueError("beta_screening_input_bytes_changed")
    return path
