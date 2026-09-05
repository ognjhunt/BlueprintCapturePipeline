"""Closed, production-owned executors for the SAM preparation phase queue."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_sam31_plan import PROFILE_ENV, PROFILE_SCHEMA
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, sha

SCHEMA = "task_evaluation_sam31_phase_execution_receipt.v1"
CPU_PHASES = {"source_selections", "standard_splat_conversion", "calibrated_views", "sam31_inputs"}
REVIEW_PHASES = {"sam31_review", "calibrated_masks", "removal_freezes", "segment_cutout"}
PAID_PHASES = {"sam31_tracking", "contribution_sweep"}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _paid_stages():
    """Load the provider-touching paid stages only when a paid phase executes.

    The intake service reaches this module through the preparation queue; a
    static import of the paid stages would drag the Vast hot lane into the
    CPU-only intake process, so the module is resolved by name at call time.
    """

    return importlib.import_module(
        "blueprint_pipeline.task_evaluation_sam31_preparation_paid_stages"
    )


def execute_stage(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run a closed phase, with immutable completion and no blind paid retry."""
    phase = job.get("phase")
    require(phase in CPU_PHASES | REVIEW_PHASES | PAID_PHASES, "sam31_phase_not_supported")
    plan = job["plan"]
    profile_path = Path(os.environ.get(PROFILE_ENV, ""))
    require(profile_path.is_absolute() and sha(profile_path) == plan["server_profile_sha256"],
            "sam31_server_profile_missing_or_changed")
    profile = read(profile_path, digest_field="profile_digest")
    require(profile.get("schema_version") == PROFILE_SCHEMA and
            profile.get("source_commit") == job["expected_source_commit"] == plan["source_commit"],
            "sam31_phase_release_mismatch")
    root = Path(job["output_root"])
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
            "sam31_phase_output_invalid")
    receipt_path = root / "phase_execution_receipt.v1.json"
    if receipt_path.exists():
        receipt = read(receipt_path, digest_field="receipt_digest")
        require(receipt.get("schema_version") == SCHEMA and
                receipt.get("job_digest") == job["job_digest"] and
                receipt.get("source_commit") == job["expected_source_commit"] and
                receipt.get("phase") == phase, "sam31_phase_receipt_conflict")
        outcome = receipt["outcome"]
        for row in outcome.get("artifacts", {}).values():
            checked_file(row["path"], row)
        if phase == "calibrated_views" and profile.get("calibrated_views", {}).get("hardware_required") is True:
            from .sam31_source_calibration_stage import validate_retained_source_calibration_stage
            validate_retained_source_calibration_stage(outcome)
        if phase in PAID_PHASES and outcome.get("status") == "completed":
            _paid_stages().validate_retained_paid_stage(outcome, stage_id=str(phase))
        return outcome
    context = {
        **job, "stage_id": phase, "server_profile": profile,
        "output_root": str(root / "artifacts"),
        "repo_root": profile.get("repo_root", "/opt/blueprint/BlueprintCapturePipeline"),
        "server_data_root": profile.get("server_data_root", "/var/lib/blueprint/task-evaluation-inputs"),
        "runtime_root": profile.get("runtime_root"),
        "ffmpeg_executable": profile.get("ffmpeg_executable", "/usr/bin/ffmpeg"),
    }
    if phase == "calibrated_views" and profile.get("calibrated_views", {}).get("hardware_required") is True:
        from .sam31_source_calibration_stage import execute_source_calibration_stage
        outcome = execute_source_calibration_stage(context)
    elif phase in CPU_PHASES:
        from .task_evaluation_sam31_preparation_cpu_stages import execute_cpu_stage
        outcome = execute_cpu_stage(context)
    elif phase in REVIEW_PHASES:
        # An uncertain model execution may not be repeated under a zero-retry
        # budget. Its retained audit is the source for diagnosis and closeout.
        require(not job.get("resume_only") or not Path(context["output_root"]).exists(),
                "sam31_review_prior_execution_requires_reconciliation")
        from .task_evaluation_sam31_preparation_review_stages import execute_review_stage
        outcome = execute_review_stage(context)
    else:
        outcome = _paid_stages().execute_paid_stage(context)
    require(isinstance(outcome, dict), "sam31_phase_result_invalid")
    if outcome.get("status") == "blocked":
        outcome = {**outcome, "status": "failed"}
    require(outcome.get("status") in {"completed", "failed", "waiting_for_external_result"},
            "sam31_phase_status_invalid")
    if outcome["status"] != "waiting_for_external_result":
        receipt = {"schema_version": SCHEMA, "job_digest": job["job_digest"],
                   "source_commit": job["expected_source_commit"], "phase": phase,
                   "outcome": outcome, "receipt_digest": ""}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        _write(receipt_path, receipt)
    return outcome
