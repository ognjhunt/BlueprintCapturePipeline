"""Typed post-SAM preparation stages with independent Agents SDK review.

No GPU allocation occurs here. Exact review/cost artifacts are retained on
rejection or failure; no caller assertion can accept masks or Gaussian cutouts.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np

from .decision_evidence_contracts import canonical_digest
from .fresh_scene_removal_freezes import materialize_fresh_scene_removal_freezes
from .fresh_scene_supervisor_bindings import materialize_fresh_scene_removal_freeze_request
from .gaussian_splat_decode import read_standard_3dgs_ply
from .public_scene_removal_selection import validate_removal_task_selection, validate_removal_scene_selection
from .public_scene_sam31_ai_visual_reviewer import run_sam31_ai_visual_review
from .public_scene_sam31_track_selection_review import (
    load_validated_sam31_track_selection_inputs,
    materialize_sam31_track_selection_inputs, materialize_sam31_track_selection_review_candidate,
    validate_sam31_ai_visual_review_rights, validate_sam31_track_selection_review,
)
from .public_scene_calibrated_object_masks import materialize_calibrated_object_mask_set
from .public_scene_segment_contribution_cutout import materialize_segment_contribution_cutout_set
from .task_evaluation_sam31_preparation_cpu_stages import _input, _resident, _record
from .task_evaluation_scene_configuration_submission_inputs import beneath, checked_file, read
from .task_evaluation_sam31_preparation_review_authority import resolve_sam31_review_rights

STAGES = {"sam31_review", "calibrated_masks", "removal_freezes", "segment_cutout"}


class Sam31PreparationReviewStageError(ValueError):
    """An independently reviewed source-preparation stage cannot advance."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Sam31PreparationReviewStageError("sam31_preparation_review_" + code)


def _profile_file(profile: Mapping[str, Any], name: str) -> Path:
    record = profile.get(name)
    _require(isinstance(record, Mapping), "operator_profile_file_missing:" + name)
    return checked_file(str(record.get("path") or ""), dict(record))


def _secret_path(value: Any) -> Path:
    # This field is supplied only by the server profile, never by the task.
    path = Path(str(value or ""))
    _require(path.is_absolute() and path.is_file()
             and not any(item.is_symlink() for item in (path, *path.parents)),
             "operator_secret_file_invalid")
    return path


def _inventory(output: Path) -> dict[str, Any]:
    return {path.relative_to(output).as_posix(): _record(path)
            for path in sorted(output.rglob("*")) if path.is_file() and not path.is_symlink()}


def _failure_blocker(exc: Exception) -> str:
    from .public_scene_sam31_frame_inventory import FRAME_REGISTRY_ERROR, FRAME_BINDING_ERROR
    safe_codes = {FRAME_REGISTRY_ERROR, FRAME_BINDING_ERROR, "sam31_review_camera_frame_set_invalid",
                  "sam31_review_source_image_invalid", "sam31_review_prepared_inputs_invalid"}
    detail = str(exc) if str(exc) in safe_codes else type(exc).__name__
    return "sam31_preparation_review_stage_failed:" + detail


def execute_review_stage(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run one server-derived stage; preserve typed outputs even when review fails."""
    stage_id = job.get("stage_id")
    _require(stage_id in STAGES, "stage_invalid")
    request, plan, profile = job.get("request"), job.get("plan"), job.get("server_profile")
    _require(all(isinstance(value, Mapping) for value in (request, plan, profile)), "job_invalid")
    commit = request.get("expected_production_commit")
    _require(profile.get("source_commit") == commit and plan.get("source_commit") == commit
             and profile.get("profile_digest") == canonical_digest(dict(profile), digest_field="profile_digest"),
             "operator_profile_binding_invalid")
    root = Path(job.get("server_data_root") or "/var/lib/blueprint/task-evaluation-inputs")
    _require(root.is_absolute() and root.is_dir(), "server_root_invalid")
    root = _resident(root, root.resolve())
    output = _resident(str(job.get("output_root") or ""), root)
    _require(output != root and not output.exists(), "output_not_fresh")
    inputs = job.get("inputs") or {}
    _require(isinstance(inputs, Mapping), "input_inventory_invalid")
    selection_path = _input(inputs, "task_selection", root)
    selection = validate_removal_task_selection(read(selection_path))
    task_id = selection["task_id"]
    artifacts: dict[str, Any] = {}
    output.mkdir(parents=True)
    try:
        if stage_id == "sam31_review":
            packet_path = _input(inputs, "sam31_task_input_packet", root)
            packet = read(packet_path, digest_field="receipt_digest")
            tracks_path = _input(inputs, "sam31_source_tracks", root)
            tracks = read(tracks_path, digest_field="result_digest")
            _require(packet.get("task_id") == task_id, "task_packet_mismatch")
            labels = {str(row.get("output_label") or row.get("text") or "").strip().casefold()
                      for row in packet.get("prompts") or []}
            labels.discard("")
            selected = sorted({str(row["track_id"]) for row in tracks.get("track_registry") or []
                               if str(row.get("label") or "").strip().casefold() in labels
                               and str(row.get("track_id") or "").strip()})
            _require(bool(labels) and bool(selected), "no_source_prompt_tracks")
            prepared_root = output / "selection-inputs"
            materialize_sam31_track_selection_inputs(
                task_input_packet_paths=[packet_path],
                source_track_result_paths_by_task={task_id: tracks_path},
                selected_track_ids_by_task={task_id: selected}, output_root=prepared_root,
            )
            prepared = prepared_root / "public_scene_sam31_track_selection_inputs.v1.json"
            freezes, task_inputs, selected_ids = load_validated_sam31_track_selection_inputs(prepared)
            _require(freezes == [str(selection_path)], "prepared_selection_mismatch")
            candidate_root = output / "candidate"
            materialize_sam31_track_selection_review_candidate(
                task_freeze_paths=freezes, task_inputs=task_inputs,
                selected_track_ids_by_task=selected_ids, output_root=candidate_root,
                prepared_inputs_path=prepared,
            )
            candidate = candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
            artifacts.update(selection_inputs=_record(prepared), track_selection_candidate=_record(candidate))
            review_profile = profile.get("sam31_visual_review")
            _require(isinstance(review_profile, Mapping), "visual_review_profile_missing")
            authority = _profile_file(review_profile, "rights_attestation")
            task_request = _input(plan["host_inputs"], "task_request", root)
            rights = resolve_sam31_review_rights(
                authority_path=authority, task_request_path=task_request,
                candidate_path=candidate, output_path=output / "review-rights.json",
                completed_prefix_adoption_path=(
                    _profile_file(profile, "completed_prefix_adoption")
                    if profile.get("completed_prefix_adoption") is not None else None
                ),
            )
            artifacts["review_rights"] = _record(rights)
            derivation = rights.with_suffix(".derivation.json")
            if derivation.is_file():
                artifacts["review_rights_derivation"] = _record(derivation)
            validate_sam31_ai_visual_review_rights(candidate_path=candidate, rights_attestation_path=rights)
            scope = _profile_file(review_profile, "openai_cost_scope_attestation")
            admin = _secret_path(review_profile.get("openai_admin_api_key_file"))
            result = run_sam31_ai_visual_review(
                candidate_path=candidate, rights_attestation_path=rights, output_root=output / "sdk-review",
                openai_cost_scope_attestation_path=scope, openai_admin_api_key_file=admin,
                openai_project_id=str(review_profile.get("openai_project_id") or ""),
                openai_api_key_id=str(review_profile.get("openai_api_key_id") or ""),
            )
            for key, result_key in (("track_selection_review", "review_receipt"),
                                    ("review_execution", "execution_receipt")):
                record = result[result_key]
                artifacts[key] = _record(checked_file(str(record["path"]), record))
            _require(result.get("decision") == "accepted", "independent_sdk_rejected_selection")
            validate_sam31_track_selection_review(
                receipt_path=artifacts["track_selection_review"]["path"], task_freeze_paths=freezes,
                task_inputs=task_inputs, selected_track_ids_by_task=selected_ids,
            )
        elif stage_id == "calibrated_masks":
            prepared = _input(inputs, "selection_inputs", root)
            review = _input(inputs, "track_selection_review", root)
            review_value = read(review, digest_field="receipt_digest")
            _require(review_value.get("schema_version") ==
                     "public_scene_sam31_track_selection_ai_visual_review.v1",
                     "independent_sdk_review_required")
            freezes, task_inputs, selected = load_validated_sam31_track_selection_inputs(prepared)
            _require(freezes == [str(selection_path)], "prepared_selection_mismatch")
            produced = output / "masks"
            result = materialize_calibrated_object_mask_set(
                task_freeze_paths=freezes, task_inputs=task_inputs, selected_track_ids_by_task=selected,
                reviewed_track_selection_receipt_path=review, output_root=produced,
            )
            _require(result["selection_authority"]["all_selected_tracks_ai_visual_review_accepted"] is True,
                     "mask_review_not_accepted")
            artifacts["calibrated_mask_set"] = _record(produced / "public_scene_calibrated_object_mask_set.v1.json")
        elif stage_id == "removal_freezes":
            source = _input(inputs, "standard_splat", root)
            collision = _input(inputs, "source_collision", root)
            frame = _input(inputs, "registered_frame", root)
            masks = _input(inputs, "calibrated_mask_set", root)
            mask_receipt = read(masks, digest_field="receipt_digest")
            _require(mask_receipt.get("selection_authority", {}).get(
                "all_selected_tracks_ai_visual_review_accepted") is True, "mask_review_not_accepted")
            scene = validate_removal_scene_selection(read(_input(inputs, "scene_selection", root)))
            bounds = selection["source_object"]["observed_bounds_world_m"]
            splat = read_standard_3dgs_ply(source)
            count = int(np.count_nonzero(np.all(
                (splat.xyz >= bounds["minimum"]) & (splat.xyz <= bounds["maximum"]), axis=1,
            )))
            _require(count > 0, "diagnostic_baseline_empty")
            policy = profile.get("gaussian_excision_policy")
            _require(isinstance(policy, Mapping) and policy.get("deterministic_repetitions") == 2,
                     "contribution_policy_missing")
            task = {
                "target_collision_prim_path": selection["removal_plan"]["source_collider_prim_path"],
                "scene": {"publisher_scene_id": scene["selected_scene_id"], "task_id": task_id,
                          "target_instance_id": selection["source_object"]["instance_id"],
                          "target_semantic_label": selection["source_object"]["semantic_label"],
                          "removal_id": selection["removal_plan"]["removal_id"],
                          "mask_set_id": selection["removal_plan"]["mask_set_id"]},
                "policy": dict(policy),
                "historical_baseline": {"method": "center_inside_registered_target_aabb",
                                        "center_aabb_min_m": bounds["minimum"],
                                        "center_aabb_max_m": bounds["maximum"],
                                        "selected_gaussian_count": count},
            }
            request_path = output / "fresh_scene_removal_freeze_tool_request.v1.json"
            freeze_request = materialize_fresh_scene_removal_freeze_request(
                source_standard_splat_path=source, source_collision_path=collision,
                registered_frame_receipt_path=frame, calibrated_mask_set_receipt_path=masks,
                tasks={task_id: task}, output_path=request_path, roots=(root,),
            )
            # Exact calibrated SAM images/masks are reopened through the mask
            # receipt. The older render receipt's AABB masks are not substituted.
            produced = output / "freezes"
            result = materialize_fresh_scene_removal_freezes(request=freeze_request, output_root=produced)
            set_path = produced / "fresh_scene_removal_freeze_set.v1.json"
            artifacts.update(excision_freeze_set=_record(set_path), segment_sweep_freeze_set=_record(set_path),
                             removal_freeze_request=_record(request_path))
            _require(len(result["tasks"]) == 1 and result["tasks"][0]["task_id"] == task_id,
                     "freeze_task_mismatch")
            for key in ("excision_freeze", "segment_sweep_freeze"):
                row = result["tasks"][0][key]
                artifacts[key] = _record(checked_file(beneath(produced, row["relative_path"]), row))
        else:
            source = _input(inputs, "standard_splat", root)
            sweep = _input(inputs, "segment_sweep_freeze", root)
            contribution = _input(inputs, "gaussian_contribution_evidence", root)
            produced = output / "cutout"
            result = materialize_segment_contribution_cutout_set(
                source_standard_splat_path=source, task_freeze_paths=[selection_path],
                sweep_freeze_paths_by_task={task_id: sweep},
                contribution_manifest_paths_by_task={task_id: contribution}, output_root=produced,
            )
            artifacts["segment_cutout_set"] = _record(produced / "adp009d_segment_contribution_cutout_set.v1.json")
            for key, row in result["shared_scene_union"]["outputs"].items():
                artifacts[key] = _record(checked_file(beneath(produced, row["relative_path"]), row))
    except Exception as exc:
        # Official-cost receipts, SDK responses and negative decisions stay in
        # the stage root. Do not copy exception text that may contain secrets.
        return {"status": "blocked", "stage_id": stage_id, "artifacts": artifacts,
                "blockers": [_failure_blocker(exc)],
                "retained_artifacts": _inventory(output), "candidate_policy_queried": False,
                "provider_compute_allocated": False}
    return {"status": "completed", "stage_id": stage_id, "artifacts": artifacts,
            "retained_artifacts": _inventory(output), "candidate_policy_queried": False,
            "provider_compute_allocated": False, "evaluation_authorized": False}
