"""Provider-free typed intake for immutable SAM preparation phase jobs.

The parent control plane may enqueue work without importing execution services,
SDK operators, GPU adapters, or their dependency graphs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive

PHASES = (
    "source_selections", "standard_splat_conversion", "calibrated_views", "sam31_inputs",
    "sam31_tracking", "sam31_review", "calibrated_masks", "removal_freezes",
    "contribution_sweep", "segment_cutout",
)
JOB_SCHEMA = "task_evaluation_sam31_preparation_execution_job.v1"
STATES = ("pending", "processing", "waiting_external", "completed", "failed")


class Sam31PhaseExecutionError(ValueError):
    """A closed phase could not be advanced from verified parent evidence."""


def _require(value: bool, reason: str) -> None:
    if not value:
        raise Sam31PhaseExecutionError("sam31_phase_" + reason)


def _read(path: Path) -> dict:
    _require(not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file() and path.stat().st_size <= 4 * 1024 * 1024, "record_path_invalid")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_launch_preparation_record_exclusive(path, value)
    except FileExistsError:
        _require(_read(path) == value, "immutable_record_conflict")


def _ensure(root: Path) -> None:
    _require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
             "queue_path_invalid")
    root.mkdir(parents=True, exist_ok=True)
    for state in (*STATES, "results", "started", "progress", "wake-pending", "wake-completed", "worker-lock"):
        path = root / state
        _require(not path.is_symlink(), "queue_path_invalid")
        path.mkdir(exist_ok=True)


def _ref(value: Mapping[str, Any]) -> dict:
    _require(isinstance(value, Mapping) and set(value) in (
        {"path", "sha256", "size_bytes"}, {"path", "digest", "size_bytes"}), "file_reference_invalid")
    return {"path": value["path"], "sha256": value.get("sha256", value.get("digest")),
            "size_bytes": value["size_bytes"]}


def enqueue_sam31_phase(
    *, queue_root: str | Path, parent_preparation_id: str, parent_request_digest: str,
    expected_source_commit: str, plan_ref: Mapping[str, Any], phase: str,
    inputs: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Queue one stable phase; never accept commands or execute provider work."""
    _require(phase in PHASES and isinstance(inputs, Mapping) and len(inputs) <= 64, "phase_or_inputs_invalid")
    plan = _ref(plan_ref)
    normalized = {name: _ref(value) for name, value in inputs.items()}
    _require(all(isinstance(name, str) and name and "/" not in name and "\\" not in name
                 for name in normalized), "input_name_invalid")
    identities = {name: {key: value[key] for key in ("sha256", "size_bytes")}
                  for name, value in normalized.items()}
    inputs_digest = canonical_digest(identities)
    key = {"parent_request_digest": parent_request_digest, "plan_digest": plan["sha256"],
           "phase": phase, "inputs_digest": inputs_digest}
    child_id = "sam31-" + canonical_digest(key).removeprefix("sha256:")
    job = {"schema_version": JOB_SCHEMA, "child_id": child_id,
           "parent_preparation_id": parent_preparation_id, **key,
           "expected_source_commit": expected_source_commit, "plan_ref": plan, "inputs": normalized}
    job["job_digest"] = canonical_digest(job, digest_field="job_digest")
    root = Path(queue_root)
    _ensure(root)
    existing = [root / state / f"{child_id}.json" for state in STATES
                if (root / state / f"{child_id}.json").exists()]
    _require(len(existing) <= 1, "job_identity_ambiguous")
    path = existing[0] if existing else root / "pending" / f"{child_id}.json"
    if existing:
        _require(_read(path) == job, "job_identity_conflict")
    else:
        _write(path, job)
    return {"schema_version": "task_evaluation_sam31_preparation_execution_intake.v1",
            "status": "already_exists" if existing else "queued", "child_id": child_id,
            "job_digest": job["job_digest"], "phase": phase,
            "parent_request_digest": parent_request_digest, "plan_digest": plan["sha256"],
            "job_path": str(path), "result_path": str(root / "results" / f"{child_id}.json")}


