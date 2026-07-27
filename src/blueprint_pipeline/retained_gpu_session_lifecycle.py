"""Provider-neutral, append-only lifecycle evidence for retained GPU sessions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso


SCHEMA_VERSION = "blueprint.retained_gpu_session_lifecycle.v1"
JOURNAL_NAME = "retained_gpu_session_lifecycle.jsonl"
MANIFEST_NAME = "retained_gpu_session_manifest.json"
STATES = (
    "allocated",
    "container_starting",
    "image_pulling",
    "model_downloading",
    "model_loading",
    "healthy",
    "retained_owned",
    "refresh_in_progress",
    "experiment_running",
    "terminal_success",
    "terminal_failure",
    "teardown_requested",
    "provider_absent",
)
_TRANSITIONS: Mapping[str | None, frozenset[str]] = {
    None: frozenset({"allocated"}),
    "allocated": frozenset({"container_starting", "terminal_failure", "teardown_requested"}),
    "container_starting": frozenset(
        {"image_pulling", "healthy", "terminal_failure", "teardown_requested"}
    ),
    "image_pulling": frozenset(
        {"model_downloading", "healthy", "terminal_failure", "teardown_requested"}
    ),
    "model_downloading": frozenset(
        {"model_loading", "healthy", "terminal_failure", "teardown_requested"}
    ),
    "model_loading": frozenset({"healthy", "terminal_failure", "teardown_requested"}),
    "healthy": frozenset(
        {
            "retained_owned",
            "experiment_running",
            "terminal_success",
            "terminal_failure",
            "teardown_requested",
        }
    ),
    "retained_owned": frozenset(
        {"refresh_in_progress", "experiment_running", "terminal_failure", "teardown_requested"}
    ),
    "refresh_in_progress": frozenset(
        {
            "healthy",
            "retained_owned",
            "experiment_running",
            "terminal_failure",
            "teardown_requested",
        }
    ),
    "experiment_running": frozenset(
        {"healthy", "retained_owned", "terminal_success", "terminal_failure", "teardown_requested"}
    ),
    "terminal_success": frozenset({"teardown_requested"}),
    "terminal_failure": frozenset({"teardown_requested"}),
    "teardown_requested": frozenset({"provider_absent"}),
    "provider_absent": frozenset(),
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("retained_gpu_session_manifest_not_object")
    return value


def record_retained_gpu_state(
    root: str | Path,
    state: str,
    *,
    evidence: Mapping[str, Any] | None = None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Validate one transition and atomically append its hash-bound evidence."""

    if state not in STATES:
        raise ValueError(f"unsupported_retained_gpu_state:{state}")
    resolved = Path(root).expanduser().resolve()
    ensure_dir(resolved)
    manifest_path = resolved / MANIFEST_NAME
    journal_path = resolved / JOURNAL_NAME
    manifest = _read_manifest(manifest_path)
    previous_state = manifest.get("state") if manifest else None
    if previous_state not in _TRANSITIONS or state not in _TRANSITIONS[previous_state]:
        raise ValueError(f"invalid_retained_gpu_transition:{previous_state}->{state}")
    previous_hash = str(manifest.get("journal_tail_sha256") or "0" * 64)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": int(manifest.get("sequence") or 0) + 1,
        "recorded_at": recorded_at or utc_now_iso(),
        "previous_state": previous_state,
        "state": state,
        "previous_record_sha256": previous_hash,
        "evidence": dict(evidence or {}),
    }
    row["record_sha256"] = hashlib.sha256(_canonical_bytes(row)).hexdigest()
    descriptor = os.open(journal_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, _canonical_bytes(row) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    next_manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "sequence": row["sequence"],
        "journal_path": str(journal_path),
        "journal_tail_sha256": row["record_sha256"],
        "updated_at": row["recorded_at"],
        "states": list(STATES),
        "terminal": state == "provider_absent",
        "evidence": row["evidence"],
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_bytes(
        json.dumps(next_manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, manifest_path)
    return next_manifest


__all__ = [
    "JOURNAL_NAME",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "STATES",
    "record_retained_gpu_state",
]
