"""Append-only normalized index for immutable G1 kitchen evidence."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso


SCHEMA_VERSION = "g1_kitchen_run_index_event.v1"
EVENT_TYPES = frozenset(
    {"attempt_allocated", "attempt_terminalized", "selection_superseded", "bundle_ineligible"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_run_index_event(
    *,
    run_root: str | Path,
    event_type: str,
    run_id: str,
    attempt_id: str | None = None,
    artifact_paths: Sequence[str | Path] = (),
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one normalized event while retaining all raw evidence unchanged."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported run index event type:{event_type}")
    root = Path(run_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, Any]] = []
    for raw in artifact_paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"indexed artifact outside run root:{path}") from exc
        refs.append(
            {
                "relative_path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": utc_now_iso(),
        "event_type": event_type,
        "run_id": str(run_id),
        "attempt_id": str(attempt_id) if attempt_id else None,
        "artifact_refs": refs,
        "detail": dict(detail or {}),
        "retention": {
            "raw_evidence_retained": True,
            "normalization_rewrites_raw_evidence": False,
            "stale_bundles_marked_ineligible_not_deleted": True,
        },
    }
    index_path = root / "g1_kitchen_run_index.jsonl"
    with index_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if event_type == "attempt_terminalized":
            handle.seek(0)
            for line in handle:
                if not line.strip():
                    continue
                prior = json.loads(line)
                if (
                    prior.get("event_type") == "attempt_terminalized"
                    and prior.get("run_id") == str(run_id)
                    and prior.get("attempt_id") == str(attempt_id)
                ):
                    raise ValueError(
                        f"duplicate terminal attempt event:{run_id}:{attempt_id}"
                    )
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {**payload, "index_path": str(index_path)}


def load_run_index(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_terminal: set[tuple[str, str]] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"run index schema mismatch at line {number}")
        record = dict(value)
        if record.get("event_type") == "attempt_terminalized":
            key = (str(record.get("run_id") or ""), str(record.get("attempt_id") or ""))
            if key in seen_terminal:
                raise ValueError(f"duplicate terminal attempt event:{key[0]}:{key[1]}")
            seen_terminal.add(key)
        records.append(record)
    return records
