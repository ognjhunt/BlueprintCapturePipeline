"""Advance an automatic SAM precursor through immutable production child jobs.

This no-spend driver only queues typed work and consumes verified child receipts.
The child execution service owns CPU/model work; GPU work still uses the normal
launch dispatcher and canonical paid-resource allocator.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_sam31_plan import (
    HOST_ROOTS, PHASES, PROFILE_ENV, PROFILE_SCHEMA, file_record,
    validate_sam31_preparation_plan,
)
from .task_evaluation_scene_configuration_submission_inputs import (
    checked_file, read, require, sha,
)

CHILD_QUEUE_ENV = "BLUEPRINT_TASK_EVALUATION_SAM31_EXECUTION_QUEUE_ROOT"
DEFAULT_CHILD_QUEUE = Path("/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions")
EVIDENCE_NAMES = ("calibrated_mask_set", "segment_cutout_set", "track_selection_review",
                  "selection_inputs", "standard_splat_conversion")


def _reference(value: dict[str, Any], roots: tuple[Path, ...]) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"},
            "sam31_child_artifact_invalid")
    path = Path(value["path"])
    require(path.is_absolute() and any(path.resolve().is_relative_to(root.resolve())
                                      for root in roots), "sam31_child_artifact_outside_roots")
    checked_file(path, value)
    return dict(value)


def _context_plan(context: dict[str, Any], *, roots: tuple[Path, ...]) -> tuple[dict, dict]:
    expected = context["stage_one_configuration"]["sam31_preparation_plan"]
    rows = [row for row in context["materialized_references"]
            if all(row.get(k) == expected[k] for k in ("uri", "digest", "size_bytes"))]
    require(bool(rows), "sam31_plan_not_materialized")
    for row in rows:
        checked_file(Path(row["materialized_path"]), {
            "sha256": expected["digest"], "size_bytes": expected["size_bytes"]})
    paths = {str(Path(row["materialized_path"]).resolve()) for row in rows}
    require(len(paths) == 1, "sam31_plan_ambiguous")
    path = Path(next(iter(paths)))
    ref = {"path": str(path), "sha256": expected["digest"], "size_bytes": expected["size_bytes"]}
    checked_file(path, ref)
    plan = validate_sam31_preparation_plan(
        read(path, digest_field="plan_digest"), source_commit=context["expected_source_commit"],
        approved_roots=roots)
    request = context["request"]
    require(plan["scene_identity"] == request["scene"]["identity"]
            and plan["task_identity"] == request["task"]["identity"], "sam31_plan_task_mismatch")
    return plan, ref


def _server_profile(plan: dict[str, Any], roots: tuple[Path, ...]) -> dict[str, Any]:
    path_text = os.environ.get(PROFILE_ENV, "")
    require(bool(path_text), "sam31_server_profile_missing")
    path = Path(path_text)
    require(path.is_absolute() and sha(path) == plan["server_profile_sha256"],
            "sam31_server_profile_changed")
    profile = read(path, digest_field="profile_digest")
    require(profile.get("schema_version") == PROFILE_SCHEMA
            and profile.get("source_commit") == plan["source_commit"], "sam31_server_profile_invalid")
    references = profile.get("artifact_references", {})
    require(isinstance(references, dict), "sam31_server_profile_references_invalid")
    for ref in references.values():
        _reference(ref, roots)
    return profile


def advance_sam31_preparation(
    context: dict[str, Any], *, approved_roots: tuple[Path, ...] = HOST_ROOTS,
    enqueue_phase: Any | None = None,
) -> dict[str, Any]:
    if enqueue_phase is None:
        from .task_evaluation_sam31_phase_queue import enqueue_sam31_phase
        enqueue_phase = enqueue_sam31_phase
    request = context["request"]
    commit = context["expected_source_commit"]
    plan, plan_ref = _context_plan(context, roots=approved_roots)
    profile = _server_profile(plan, approved_roots)
    inputs = dict(plan["host_inputs"])
    profile_refs = profile.get("artifact_references", {})
    require(not set(inputs).intersection(profile_refs), "sam31_profile_overrides_host_inputs")
    inputs.update(profile_refs)
    results = []
    child_queue = Path(os.environ.get(CHILD_QUEUE_ENV, str(DEFAULT_CHILD_QUEUE)))
    adoption = None
    phases = PHASES
    if profile.get("completed_prefix_adoption") is not None:
        from .task_evaluation_sam31_prefix_adoption import validate_completed_prefix_adoption
        ref = _reference(profile["completed_prefix_adoption"], approved_roots)
        adopted = validate_completed_prefix_adoption(ref["path"], expected_source_commit=commit,
            approved_roots=approved_roots + (DEFAULT_CHILD_QUEUE, Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations")),
            current_plan=plan, current_provider_profile_path=profile_refs["sam31_provider_profile"]["path"])
        for name, artifact in adopted["artifacts"].items():
            verified = _reference(artifact, approved_roots)
            require(name not in inputs or inputs[name] == verified, "sam31_adoption_input_conflict:" + name)
            inputs[name] = verified
        phases = PHASES[adopted["phase_count"]:]
        adoption = {"receipt": ref, "original_execution_commit": adopted["record"]["original_execution_commit"],
                    "through_phase": adopted["record"]["through_phase"],
                    "original_phase_result_receipts": [row["result"] for row in adopted["record"]["phase_records"]]}
    for phase in phases:
        intake = enqueue_phase(
            queue_root=child_queue, parent_preparation_id=request["preparation_id"],
            parent_request_digest=context["request_digest"], expected_source_commit=commit,
            plan_ref=plan_ref, phase=phase, inputs=inputs)
        result_path = Path(intake["result_path"])
        if not result_path.exists():
            # The child intake itself is evidence of queued work, not of
            # completed measurements or a paid allocation.
            queued = file_record(intake["job_path"])
            return {"status": "waiting_for_child", "phase": phase,
                    "child_id": intake["child_id"], "child_job_digest": intake["job_digest"],
                    "evidence_refs": [queued], "human_review_required": False,
                    "candidate_policy_queried": False, **({"completed_prefix_adoption": adoption} if adoption else {})}
        result = read(result_path, digest_field="result_digest")
        require(result.get("schema_version") == "task_evaluation_sam31_preparation_execution_result.v1"
                and result.get("source_commit") == commit
                and result.get("job_digest") == intake["job_digest"]
                and result.get("child_id") == intake["child_id"]
                and result.get("phase") == phase
                and result.get("parent_request_digest") == context["request_digest"]
                and result.get("plan_digest") == plan_ref["sha256"],
                "sam31_child_result_binding_invalid")
        require(result.get("status") == "completed", "sam31_child_stage_failed:" + phase)
        artifacts = result.get("artifacts")
        require(isinstance(artifacts, dict) and bool(artifacts), "sam31_child_artifacts_missing")
        for name, ref in artifacts.items():
            verified = _reference(ref, approved_roots)
            require(name not in inputs or inputs[name] == verified,
                    "sam31_child_artifact_identity_changed:" + name)
            inputs[name] = verified
        if phase == "standard_splat_conversion":
            require("standard_splat_conversion_receipt" in artifacts,
                    "sam31_conversion_receipt_missing")
            inputs["standard_splat_conversion"] = inputs["standard_splat_conversion_receipt"]
        # Stages themselves have exact scientific validators; this chain binds
        # each stage to all previous exact inputs and the same immutable plan.
        results.append(file_record(result_path))
    require(all(name in inputs for name in EVIDENCE_NAMES), "sam31_final_evidence_missing")
    evidence = {name: inputs[name] for name in EVIDENCE_NAMES}
    sealed = {"schema_version": "task_evaluation_sam31_preparation_result.v1",
              "status": "exact_mask_inputs_ready", "source_commit": commit,
              # Consumer binds the immutable plan file reference, not a
              # caller-supplied claim that any mask or render passed.
              "plan_digest": plan_ref["sha256"], "evidence": evidence,
              "stage_result_receipts": results, "review_kind": "ai",
              "human_review_required": False, "candidate_policy_queried": False,
              "result_digest": ""}
    if adoption is not None:
        sealed["completed_prefix_adoption"] = adoption
    sealed["result_digest"] = canonical_digest(sealed, digest_field="result_digest")
    # The renderer consumer independently reopens and validates all five
    # artifacts and their renderer/track/contribution chains before handoff.
    return {"status": "ready", "evidence_refs": list(evidence.values()),
            "sam31_exact_mask_inputs": evidence, "sam31_preparation_result": sealed,
            "human_review_required": False, "candidate_policy_queried": False}
