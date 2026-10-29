"""Reuse a retained Atlas choice for candidate preparation, never owner truth.

ADP-040 / day 28: the same source review must not be followed by a repeated
question from an older analysis. Original uncertainty and all physical limits
stay in the derived record. No provider or spending operation runs here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .atlas_automation_plan import AutomationPlan
from .common import StageError
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .task_profile_contracts import task_profile_for
from .website_atlas_preview_store import SCHEMA as JOB_SCHEMA
from .website_task_context_contracts import validate_website_task_context, website_task_context_digest


def _read(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    ancestors = [root]
    current = root
    for name in Path(relative).parts:
        current /= name
        ancestors.append(current)
    if (not relative or Path(relative).is_absolute() or ".." in Path(relative).parts
            or not path.resolve().is_relative_to(root.resolve())
            or any(part.is_symlink() for part in ancestors)):
        raise ValueError("website_atlas_target_selection_path_invalid")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("website_atlas_target_selection_record_invalid")
    return value


def _source_binding(plan, task_context, source_video):
    validate_website_task_context(task_context, request_id=task_context.get("request_id"),
        scene_id=task_context.get("scene_id"), capture_id=task_context.get("capture_id"), purpose="scene_preparation")
    context_digest = website_task_context_digest(task_context)
    source_digest = _sha256_file(source_video)
    raw_context_hash = hashlib.sha256(json.dumps(dict(task_context), sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
    if (task_context.get("context_digest") != context_digest
            or plan.get("input_video_sha256") != source_digest.removeprefix("sha256:")
            or plan.get("task_context_sha256") != raw_context_hash):
        return None
    return context_digest, source_digest


def load_retained_uncertain_analysis(*, capture_root: Path, task_context: Mapping[str, Any],
                                     source_video: Path) -> dict[str, Any] | None:
    """Reuse an exact old stage result even when it predates model-call receipts."""
    root = capture_root / "pipeline" / "clean_plate"
    if not (root / "removal_plan.json").exists():
        return None
    if root.parent.is_symlink():
        raise ValueError("website_atlas_target_selection_path_invalid")
    value = _read(root, "removal_plan.json")
    selection = value.get("retained_task_selection")
    if selection is not None:
        original = value.get("original_uncertain_analysis")
        if (not isinstance(original, dict) or not isinstance(selection, dict)
                or selection.get("original_analysis_digest") != canonical_digest(original)):
            raise ValueError("website_atlas_target_selection_original_changed")
        value = original
    blockers = value.get("blockers") or []
    if (value.get("status") != "blocked" or not blockers
            or not all(isinstance(code, str) and (code.startswith("task_target_requires_clarification:")
                       or code == "task_relevant_target_not_observed") for code in blockers)
            or _source_binding(value, task_context, source_video) is None):
        return None
    from .clean_plate_removal_analysis_gemini import validate_removal_plan
    if validate_removal_plan(value):
        raise ValueError("website_atlas_target_selection_original_invalid")
    return value


def resolve_retained_target_selection(
    *, plan: Mapping[str, Any], capture_root: Path, task_context: Mapping[str, Any], source_video: Path,
) -> dict[str, Any]:
    """Resolve one uncertain rigid subject from its own ready Atlas selection.

    A current processing job enters the existing automatic reconstruction wait.
    Missing/abstained jobs keep the original refusal. Nothing resets prior
    attempts, grants, uncertainty, source bytes or a provider result.
    """
    original = deepcopy(dict(plan))
    targets = original.get("targets") or []
    if not isinstance(targets, list) or any(not isinstance(row, Mapping) for row in targets):
        return original
    uncertain = [row for row in targets if row.get("task_effect") == "uncertain"]
    profile = task_profile_for(task_context.get("task_family"))
    if (original.get("status") != "blocked" or len(uncertain) != 1
            or profile is None or profile.profile_id != "rigid_pick_place"):
        return original
    target = uncertain[0]
    allowed = {"task_target_requires_clarification:" + str(target.get("target_id")),
               "task_relevant_target_not_observed"}
    if (target.get("target_role") != "task_object" or target.get("target_class") != "movable_object"
            or target.get("articulated_part") or not original.get("blockers")
            or not set(original["blockers"]) <= allowed
            or any(row.get("task_effect") == "manipulated" for row in targets)):
        return original
    identity = _source_binding(original, task_context, source_video)
    if identity is None:
        raise ValueError("website_atlas_target_selection_source_changed")
    context_digest, source_digest = identity
    root = capture_root / "pipeline" / "atlas"
    if root.parent.is_symlink() or root.is_symlink():
        raise ValueError("website_atlas_target_selection_path_invalid")
    if not (root / "preview_job.json").exists():
        return original
    job = _read(root, "preview_job.json")
    if (job.get("schema_version") != JOB_SCHEMA
            or any(job.get(key) != task_context.get(key) for key in ("request_id", "scene_id", "capture_id"))
            or not isinstance(job.get("source_video"), str)
            or Path(job["source_video"]).is_absolute() or ".." in Path(job["source_video"]).parts
            or (capture_root / job["source_video"]).resolve() != source_video.resolve()
            or not source_video.resolve().is_relative_to(capture_root.resolve())):
        raise ValueError("website_atlas_target_selection_source_changed")
    if job.get("state") == "processing":
        raise StageError("website_scene_preparation", "website_atlas_job_processing")
    if job.get("state") != "ready":
        return original
    result = _read(root, "result.json")
    result_digest = canonical_digest(result, digest_field="digest")
    if (result.get("schema_version") != "website_atlas_pipeline_result.v1"
            or result.get("digest") != result_digest or job.get("result_digest") != result_digest
            or result.get("source_video_digest") != source_digest
            or result.get("task_context_digest") != context_digest):
        raise ValueError("website_atlas_target_selection_result_changed")
    selection = _read(root, "selection_binding.json")
    receipt = selection.get("planner_receipt") or {}
    binding, proposal = receipt.get("binding") or {}, receipt.get("proposal") or {}
    if (receipt.get("status") != "completed" or receipt.get("binding_digest") != canonical_digest(binding)
            or receipt.get("proposal_digest") != canonical_digest(proposal)
            or binding.get("source_video_digest") != source_digest
            or binding.get("task_context_digest") != context_digest
            or proposal.get("status") != "proposed" or proposal.get("subject_origin") != "captured"):
        raise ValueError("website_atlas_target_selection_planner_changed")
    survey = _read(root, selection.get("survey_relative_path", ""))
    if (survey.get("digest") != canonical_digest(survey, digest_field="digest")
            or survey.get("digest") != binding.get("survey_digest")
            or survey.get("source_video_digest") != source_digest):
        raise ValueError("website_atlas_target_selection_survey_changed")
    automation = AutomationPlan.model_validate(proposal.get("automation")).model_dump(mode="json", exclude_none=True)
    review = selection.get("review") or {}
    if (review.get("automation") != automation or review.get("survey_digest") != survey["digest"]
            or review.get("subject_origin") != "captured"):
        raise ValueError("website_atlas_target_selection_review_changed")
    removed = [row for row in automation["objects"] if row["disposition"] == "remove"]
    if len(removed) != 1:
        return original
    chosen = removed[0]
    frames = {row["frame_id"]: row for row in survey["frames"]}
    unsafe = set(proposal.get("unsafe_frame_ids") or [])
    admitted = {key for row in proposal.get("coverage_requirements") or []
                for key in row.get("required_frame_ids") or []} | set(proposal.get("additional_input_frame_ids") or [])
    evidence = []
    for region in chosen["regions"]:
        frame = frames.get(region["frame_id"])
        timestamp = frame.get("timestamp_seconds") if frame else None
        if (region["frame_id"] not in admitted or region["frame_id"] in unsafe or isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("website_atlas_target_selection_observation_missing")
        evidence.append({"timestamp_seconds": timestamp,
                         "box_xywh_normalized": region["box_xywh_normalized"]})
    source_pose = result.get("source_pose") or {}
    if source_pose.get("status") != "completed" or not isinstance(source_pose.get("pose_relative_path"), str):
        raise ValueError("website_atlas_target_selection_pose_changed")
    pose = _read(root, source_pose["pose_relative_path"] + "/atlas_posed_rgbd_result.json")
    pose_binding = pose.get("binding") or {}
    if (pose.get("digest") != canonical_digest(pose, digest_field="digest")
            or pose.get("digest") != source_pose.get("result_digest")
            or pose_binding.get("source_video_digest") != source_digest
            or pose_binding.get("survey_digest") != survey["digest"]
            or pose_binding.get("scene_state_review_digest") != canonical_digest(review)):
        raise ValueError("website_atlas_target_selection_pose_changed")
    derived_target = {**target, "segmentation_prompt": chosen["segmentation_prompt"],
                      "task_effect": "manipulated", "disposition": "remove",
                      "rebuild_intent": "rebuild_and_compose", "collision_required": True,
                      "spatial_evidence": evidence,
                      "decision_reason": "Retained Atlas source review selected this candidate; owner identity is unconfirmed."}
    derived = {**original, "status": "completed", "blockers": [], "movable_removal_count": 1,
               "targets": [derived_target if row is target else row for row in targets],
               "provider": "retained_atlas_source_selection", "model": receipt.get("model"),
               "processing": "retained_source_selection", "original_uncertain_analysis": original}
    provenance = {"schema_version": "website_atlas_task_selection.v1", "claim_ceiling": "development_only",
                  "basis": "source_bound_candidate_not_owner_confirmation",
                  "source_video_digest": source_digest, "task_context_digest": context_digest,
                  "original_analysis_digest": canonical_digest(original), "atlas_result_digest": result_digest,
                  "planner_receipt_digest": canonical_digest(receipt), "planner_binding_digest": receipt["binding_digest"],
                  "planner_proposal_digest": receipt["proposal_digest"], "atlas_target_id": chosen["target_id"],
                  "source_pose_digest": pose["digest"],
                  "owner_confirmation": False, "metric_authority": False, "physical_evidence": False}
    provenance["digest"] = canonical_digest(provenance, digest_field="digest")
    derived["retained_task_selection"] = provenance
    if (_read(root, "preview_job.json") != job or _read(root, "selection_binding.json") != selection
            or _read(root, "result.json") != result
            or _read(root, source_pose["pose_relative_path"] + "/atlas_posed_rgbd_result.json") != pose):
        raise ValueError("website_atlas_target_selection_snapshot_changed")
    return derived
