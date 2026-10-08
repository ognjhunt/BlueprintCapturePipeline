"""ADP-010/day14: local handoff durability reads shared by intake and workers.

This boundary intentionally imports no execution, provider or orchestration
modules. The worker keeps its public helper names through direct reexports.
"""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

JOB_LEDGER_FILENAME = "pipeline_job_ledger.json"
JOB_OUTPUT_COMMIT_FILENAME = "pipeline_job_output_commit.json"
JOB_LEDGER_SCHEMA_VERSION = "pipeline_job_ledger.v1"
JOB_OUTPUT_COMMIT_SCHEMA_VERSION = "pipeline_job_output_commit.v1"


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    """The JSON object at path, or {} when it is missing or unreadable.

    ValueError covers both invalid JSON and bytes that are not UTF-8.
    """

    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stage_result_status(value: Any) -> str | None:
    if isinstance(value, Mapping):
        status = _string(value.get("status"))
        return status or None
    return None


def required_stage_result_blocker(stage: str, value: Any) -> str | None:
    """Required execution results must prove success before completion or reuse.

    Readiness and optional trust outputs may truthfully remain blocked. These
    two stages, however, are always executed by run_end_to_end.
    """
    successful = {
        "capture_pipeline": {"completed"},
        "task_evaluation_supervisor": {
            "non_spend_complete", "advise_complete", "shadow_complete",
            "preauthorized_complete",
        },
    }
    if stage not in successful:
        return None
    status = _stage_result_status(value) or "missing_status"
    if status not in successful[stage]:
        return f"required_stage_not_complete:{stage}:{status}"
    return None


def _read_job_ledger(capture_root: Path) -> dict[str, Any]:
    ledger_path = capture_root / JOB_LEDGER_FILENAME
    if not ledger_path.is_file():
        return {}
    try:
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # ValueError: invalid JSON or not UTF-8
        return {
            "schema_version": JOB_LEDGER_SCHEMA_VERSION,
            "status": "corrupt",
            "ledger_read_error": type(exc).__name__,
        }
    if not isinstance(loaded, dict):
        return {
            "schema_version": JOB_LEDGER_SCHEMA_VERSION,
            "status": "corrupt",
            "ledger_read_error": "not_mapping",
        }
    return loaded


@contextmanager
def _existing_job_ledger_lock(capture_root: Path) -> Iterator[str]:
    """Hold the ledger lock of a capture that already exists, creating nothing.

    Yields "ledger_present" while the lock is held and the ledger exists,
    otherwise "capture_absent" or "ledger_absent". A capture whose workspace was
    retired (or never staged) must not be brought back by recording something
    about it.

    Contract: anything that deletes a capture root (scene workspace retirement)
    must hold this same flock, on the capture's .pipeline_job_ledger.json.lock,
    for the whole deletion. A writer here then either finishes before the
    deletion starts or finds the ledger gone once it gets the lock.
    """

    def absence() -> str:
        return "ledger_absent" if capture_root.is_dir() else "capture_absent"

    try:
        descriptor: int | None = os.open(capture_root / f".{JOB_LEDGER_FILENAME}.lock", os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        descriptor = None
    if descriptor is None:
        yield absence()
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield "ledger_present" if (capture_root / JOB_LEDGER_FILENAME).is_file() else absence()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _retained_required_stage_blockers(capture_root: Path) -> list[str]:
    retained = _read_optional_json_object(
        capture_root / "pipeline" / "run_e2e_stage_ledger.json"
    )
    blockers = []
    for stage, entry in _mapping(retained.get("stages")).items():
        if "result_snapshot" not in _mapping(entry):
            continue
        blocker = required_stage_result_blocker(stage, entry["result_snapshot"])
        if blocker:
            blockers.append(blocker)
    return blockers


def _output_commit(
    capture_root: Path,
    *,
    scene_id: str,
    capture_id: str,
) -> dict[str, Any]:
    commit = _read_optional_json_object(capture_root / JOB_OUTPUT_COMMIT_FILENAME)
    if (
        commit.get("schema_version") != JOB_OUTPUT_COMMIT_SCHEMA_VERSION
        or commit.get("status") != "committed"
        or commit.get("scene_id") != scene_id
        or commit.get("capture_id") != capture_id
        or not _string(commit.get("result_sha256"))
        or _retained_required_stage_blockers(capture_root)
    ):
        return {}
    return commit

