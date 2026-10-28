"""Clarify a retained operation reading without rebuilding its captured scene.

The existing Atlas planner allocator owns any model call, admission, settlement
and bounded repair. This module never creates a grant or reconstructs geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .website_operation_coverage import (
    clarification_request, load_operation_coverage, retain_operation_coverage,
)
from .website_task_semantics import load_task_semantics, task_semantics_from_receipt


def _selected(job_dir, task_context, source_video_digest):
    if task_context.get("task_family") != "general_task":
        return None, None, None
    current = load_task_semantics(job_dir=Path(job_dir), task_context=task_context,
                                  source_video_digest=source_video_digest)
    if current is None:
        return None, None, None  # A legacy completion is never repurchased.
    parent = task_semantics_from_receipt(receipt=current["planner_receipt"], task_context=task_context,
                                         source_video_digest=source_video_digest)
    request = clarification_request(parent, task_context=task_context, source_video_digest=source_video_digest)
    return current, parent, request


def _retain(job_dir, parent, receipt, task_context, source_video_digest):
    # The first completed clarification remains authoritative for this parent,
    # including an honest unresolved reading. Later scale reviews cannot replace it.
    return load_operation_coverage(job_dir=job_dir, parent_semantics=parent, task_context=task_context,
        source_video_digest=source_video_digest) or retain_operation_coverage(job_dir=job_dir,
        parent_semantics=parent, receipt=receipt, task_context=task_context,
        source_video_digest=source_video_digest)


def scale_operation_coverage_options(*, job_dir, task_context, source_video_digest):
    """Offer clarification only as part of an otherwise necessary scale review.

    The scale controller retains its exact choice before its first call. An
    earlier completed/in-flight scale request keeps its original input and cache.
    """
    current, parent, request = _selected(job_dir, task_context, source_video_digest)
    if request is None:
        return {}
    return {"operation_coverage_request": request,
            "operation_coverage_needed": "operation_coverage_clarification" not in current,
            "operation_coverage_consumer": lambda receipt: _retain(
                Path(job_dir), parent, receipt, task_context, source_video_digest)}


def clarify_operation_coverage(*, job_dir, task_context, source_video_digest,
                               configuration=None, planner_runner=None, experimental=False,
                               invoker=None, checkpoint=None):
    """Resolve one ambiguous candidate after geometry, adopting every saved stage.

    The fallback carries task text and the selected semantic reading, no pixels.
    Its existing SDK lifecycle allows a USD 0.25 call and one schema repair;
    exhausted/abstained results never buy another automatic clarification.
    """
    job_dir = Path(job_dir)
    current, parent, request = _selected(job_dir, task_context, source_video_digest)
    if request is None or "operation_coverage_clarification" in current:
        return {"status": "adopted_or_not_required", "task_semantics": current}
    from .launch_immutable_input_writer import write_exclusive_private_bytes
    root = job_dir / "task_semantics" / "operation_coverage_reviews" / canonical_digest(request)[7:]
    survey = {"source_video_digest": source_video_digest, "heldout_pixels_included": False,
              "display_rotation_degrees": 0, "frames": [],
              "content_basis": "retained_source_bound_task_text_only"}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    survey_path = root / "survey.json"
    write_exclusive_private_bytes(survey_path, (json.dumps(survey, sort_keys=True) + "\n").encode())
    if configuration is None:
        from .website_atlas_pipeline import load_configuration
        configuration = load_configuration(task_context, source_video_digest, create=False)
    if planner_runner is None:
        from .paid_resource_allocator import run_sponsored_website_atlas_plan
        def planner_runner(**kw):
            return run_sponsored_website_atlas_plan(experimental=experimental, **kw)
    if checkpoint is not None:
        checkpoint("operation_coverage_clarification")
    try:
        receipt = planner_runner(survey_path=survey_path, task_context=task_context,
            configuration=configuration, output_root=root / "planner",
            recovery={"scope": "operation_coverage", "previews": [], "operation_coverage_request": request},
            **({"invoker": invoker} if invoker is not None else {}))
    except ValueError as exc:
        if str(exc) not in {"atlas_scene_planner_output_invalid",
                            "atlas_scene_planner_output_truncated_retry_unavailable"}:
            raise
        # Keep the reconstructed scene and original unresolved reading. The
        # retained SDK receipts own the charges and the precise failed answer.
        return {"status": "needs_input", "reason": "website_task_operation_coverage_review_invalid",
                "task_semantics": current}
    clarified = _retain(job_dir, parent, receipt, task_context, source_video_digest)
    return {"status": "clarification_retained", "task_semantics": clarified}
