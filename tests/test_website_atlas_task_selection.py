"""One source-bound derived choice clears the retained refusal without new calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
from blueprint_pipeline.common import StageError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_atlas_preview_store import SCHEMA
from blueprint_pipeline.website_atlas_task_selection import load_retained_uncertain_analysis, resolve_retained_target_selection
from blueprint_pipeline.website_reconstruction_resume import atlas_wait


def _save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def retained(tmp_path):
    capture = tmp_path / "bucket/scenes/s1/captures/c1"
    source = capture / "raw/walkthrough.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"retained-source-fixture")
    video_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    context = {"schema_version": "website_site_task_context.v1", "purpose": "scene_preparation",
               "confirmed": False, "confirmed_at": None, "request_id": "r1", "scene_id": "s1", "capture_id": "c1",
               "task_family": "rigid_pick_and_place", "description": "Move one cardboard box on its shelf."}
    context["context_digest"] = canonical_digest(context)
    target = {"target_id": "cardboard_box_candidate", "semantic_label": "cardboard box on shelf",
              "segmentation_prompt": "cardboard box", "target_class": "movable_object",
              "target_role": "task_object", "task_effect": "uncertain", "disposition": "keep",
              "rebuild_intent": "none", "articulated_part": "", "collision_required": False,
              "confidence": 0.5, "clarification_question": "Which box?",
              "spatial_evidence": [{"timestamp_seconds": 14.5,
                                    "box_xywh_normalized": [0.038, 0.001, 0.207, 0.404]}]}
    support = {**target, "target_id": "shelf", "target_role": "support", "target_class": "fixed_clutter",
               "task_effect": "static_contact", "collision_required": True, "clarification_question": ""}
    plan = build_removal_plan(targets=[target, support], status="blocked", model="retained-model",
                              processing="static", video_path=source, video_digest=video_hash,
                              blockers=["task_target_requires_clarification:cardboard_box_candidate"])
    plan["task_context_sha256"] = hashlib.sha256(json.dumps(context, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest()
    root = capture / "pipeline/atlas"
    survey = {"source_video_digest": "sha256:" + video_hash,
              "frames": [{"frame_id": "observed-1", "timestamp_seconds": 12.5}]}
    survey["digest"] = canonical_digest(survey)
    automation = {"scale_references": [], "objects": [{"target_id": "selected-box", "label": "cardboard box",
                  "segmentation_prompt": "cardboard box at left", "disposition": "remove",
                  "regions": [{"frame_id": "observed-1", "box_xywh_normalized": [0.1, 0.2, 0.3, 0.4]}]}],
                  "removal_instruction": "Remove only the selected box.", "mask_margin_fraction": 0.01}
    proposal = {"status": "proposed", "subject_origin": "captured", "automation": automation,
                "coverage_requirements": [{"required_frame_ids": ["observed-1"]}], "unsafe_frame_ids": []}
    binding = {"survey_digest": survey["digest"], "source_video_digest": "sha256:" + video_hash,
               "task_context_digest": context["context_digest"]}
    receipt = {"status": "completed", "model": "retained-planner", "binding": binding,
               "binding_digest": canonical_digest(binding), "proposal": proposal,
               "proposal_digest": canonical_digest(proposal)}
    selection = {"survey_relative_path": "survey/atlas_video_survey.json", "planner_receipt": receipt,
                 "review": {"automation": automation, "survey_digest": survey["digest"], "subject_origin": "captured"}}
    pose = {"binding": {"source_video_digest": "sha256:" + video_hash, "survey_digest": survey["digest"],
                         "scene_state_review_digest": canonical_digest(selection["review"])}}
    pose["digest"] = canonical_digest(pose)
    result = {"schema_version": "website_atlas_pipeline_result.v1", "source_video_digest": "sha256:" + video_hash,
              "task_context_digest": context["context_digest"], "physical_evidence": False,
              "source_pose": {"status": "completed", "result_digest": pose["digest"], "pose_relative_path": "pose/source"}}
    result["digest"] = canonical_digest(result)
    job = {"schema_version": SCHEMA, "state": "ready", "source_video": "raw/walkthrough.mp4",
           **{key: context[key] for key in ("request_id", "scene_id", "capture_id")}, "result_digest": result["digest"]}
    for name, value in [("survey/atlas_video_survey.json", survey), ("selection_binding.json", selection),
                        ("result.json", result), ("preview_job.json", job),
                        ("pose/source/atlas_posed_rgbd_result.json", pose)]:
        _save(root, name, value)
    return SimpleNamespace(capture=capture, source=source, context=context, plan=plan, root=root,
                           job=job, selection=selection, survey=survey, result=result)


def _resolve(r):
    return resolve_retained_target_selection(plan=r.plan, capture_root=r.capture,
                                            task_context=r.context, source_video=r.source)


def test_exact_retained_refusal_uses_atlas_choice_and_preserves_original(retained):
    before = deepcopy(retained.plan)
    result = _resolve(retained)
    assert result["status"] == "completed" and result["blockers"] == []
    chosen, support = result["targets"]
    assert chosen["target_id"] == "cardboard_box_candidate"
    assert chosen["task_effect"] == "manipulated" and chosen["disposition"] == "remove"
    assert chosen["spatial_evidence"] == [{"timestamp_seconds": 12.5, "box_xywh_normalized": [0.1, 0.2, 0.3, 0.4]}]
    assert support == before["targets"][1]
    assert result["original_uncertain_analysis"] == retained.plan == before
    provenance = result["retained_task_selection"]
    assert provenance["original_analysis_digest"] == canonical_digest(before)
    assert provenance["basis"] == "source_bound_candidate_not_owner_confirmation"
    assert provenance["owner_confirmation"] is provenance["physical_evidence"] is provenance["metric_authority"] is False
    assert result["authority_boundary"] == before["authority_boundary"]
    assert _resolve(retained) == result  # Stable reuse, no provider/model/budget hook.


def test_processing_enters_the_existing_typed_automatic_wait(retained):
    _save(retained.root, "preview_job.json", {**retained.job, "state": "processing"})
    with pytest.raises(StageError) as raised:
        _resolve(retained)
    assert atlas_wait(raised.value)
    assert retained.plan["targets"][0]["task_effect"] == "uncertain"
    _save(retained.root, "preview_job.json", retained.job)
    assert _resolve(retained)["status"] == "completed"


@pytest.mark.parametrize("state", ["needs_input", "failed", "withdrawn"])
def test_a_terminal_source_limit_is_not_an_automatic_retry(retained, state):
    _save(retained.root, "preview_job.json", {**retained.job, "state": state})
    assert _resolve(retained) == retained.plan


@pytest.mark.parametrize("record,key,value,error", [
    ("job", "capture_id", "other", "source_changed"),
    ("job", "result_digest", "sha256:" + "0" * 64, "result_changed"),
    ("result", "source_video_digest", "sha256:" + "0" * 64, "result_changed"),
    ("result", "task_context_digest", "sha256:" + "0" * 64, "result_changed"),
    ("selection", "survey_relative_path", "../outside.json", "path_invalid"),
    ("survey", "source_video_digest", "sha256:" + "0" * 64, "survey_changed"),
])
def test_changed_or_escaping_binding_is_refused(retained, record, key, value, error):
    data = deepcopy(getattr(retained, record))
    data[key] = value
    names = {"job": "preview_job.json", "result": "result.json", "selection": "selection_binding.json",
             "survey": "survey/atlas_video_survey.json"}
    _save(retained.root, names[record], data)
    with pytest.raises(ValueError, match=error):
        _resolve(retained)


@pytest.mark.parametrize("which", ["receipt", "review", "unsafe", "unseen"])
def test_planner_or_observation_substitution_is_refused(retained, which):
    selection = deepcopy(retained.selection)
    if which == "receipt":
        selection["planner_receipt"]["binding_digest"] = "sha256:" + "0" * 64
    elif which == "review":
        selection["review"]["automation"]["removal_instruction"] = "Different removal."
    else:
        proposal = selection["planner_receipt"]["proposal"]
        if which == "unsafe":
            proposal["unsafe_frame_ids"] = ["observed-1"]
        else:
            proposal["coverage_requirements"] = []
        selection["planner_receipt"]["proposal_digest"] = canonical_digest(proposal)
    _save(retained.root, "selection_binding.json", selection)
    with pytest.raises(ValueError, match="planner_changed|review_changed|observation_missing"):
        _resolve(retained)


def test_other_failures_and_other_task_families_keep_their_original_limits(retained):
    retained.plan["blockers"].append("privacy_pipeline_failed_closed")
    assert _resolve(retained) == retained.plan
    retained.plan["blockers"].pop()
    retained.context["task_family"] = "articulated_open_close"
    assert _resolve(retained) == retained.plan


def test_changed_video_and_symlinked_survey_never_choose_a_target(retained):
    path = retained.root / "survey/atlas_video_survey.json"
    outside = retained.capture / "outside.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="path_invalid"):
        _resolve(retained)
    retained.source.write_bytes(b"different-source")
    with pytest.raises(ValueError, match="source_changed"):
        _resolve(retained)


def test_real_stage_reuses_old_analysis_before_another_model_call(retained, monkeypatch):
    from blueprint_pipeline import clean_plate_stage as stage
    root = retained.capture / "pipeline/clean_plate"
    _save(root, "removal_plan.json", retained.plan)
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **kw: pytest.fail("repeat model call"))
    monkeypatch.setattr(stage, "prepare_website_source_frames", lambda **kw: {"status": "completed"})
    selected = []
    def masks(**kw):
        selected.append(kw["plan"])
        return {"status": "completed"}
    monkeypatch.setattr(stage, "run_website_task_masks", masks)
    result = stage.run_clean_plate_stage(capture_root=retained.capture,
        website_source_video=retained.source, task_context=retained.context,
        privacy_processing={"status": "pending_website_review"}, policy=stage.CleanPlatePolicy(enabled=True))
    assert result["status"] == "objects_removed" and result["privacy_verified"] is True
    assert selected[0]["original_uncertain_analysis"] == retained.plan
    assert json.loads((root / "removal_plan.json").read_text())["retained_task_selection"]["owner_confirmation"] is False
    assert load_retained_uncertain_analysis(capture_root=retained.capture, task_context=retained.context,
                                           source_video=retained.source) == retained.plan
    assert retained.source.read_bytes() == b"retained-source-fixture"


def test_real_stage_enters_wait_without_model_or_mask_dispatch(retained, monkeypatch):
    from blueprint_pipeline import clean_plate_stage as stage
    _save(retained.capture / "pipeline/clean_plate", "removal_plan.json", retained.plan)
    _save(retained.root, "preview_job.json", {**retained.job, "state": "processing"})
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **kw: pytest.fail("repeat model call"))
    monkeypatch.setattr(stage, "run_website_task_masks", lambda **kw: pytest.fail("premature paid masks"))
    with pytest.raises(StageError) as raised:
        stage.run_clean_plate_stage(capture_root=retained.capture, website_source_video=retained.source,
            task_context=retained.context, privacy_processing={"status": "pending_website_review"},
            policy=stage.CleanPlatePolicy(enabled=True))
    assert atlas_wait(raised.value)


@pytest.mark.parametrize("changed", ["source", "context"])
def test_old_analysis_cannot_be_reused_for_changed_source_or_context(retained, changed):
    root = retained.capture / "pipeline/clean_plate"
    _save(root, "removal_plan.json", retained.plan)
    assert load_retained_uncertain_analysis(capture_root=retained.capture, task_context=retained.context,
                                           source_video=retained.source) == retained.plan
    if changed == "source":
        retained.source.write_bytes(b"replacement-video")
    else:
        retained.context["description"] = "Move a different object."
        retained.context["context_digest"] = canonical_digest(retained.context, digest_field="context_digest")
    assert load_retained_uncertain_analysis(capture_root=retained.capture, task_context=retained.context,
                                           source_video=retained.source) is None


def test_two_removed_objects_do_not_become_a_single_arbitrary_choice(retained):
    selection = deepcopy(retained.selection)
    proposal = selection["planner_receipt"]["proposal"]
    proposal["automation"]["objects"].append({**proposal["automation"]["objects"][0], "target_id": "second-box"})
    selection["planner_receipt"]["proposal_digest"] = canonical_digest(proposal)
    _save(retained.root, "selection_binding.json", selection)
    assert _resolve(retained) == retained.plan


def test_withdrawal_during_selection_refuses_before_dispatch(retained, monkeypatch):
    from blueprint_pipeline import website_atlas_task_selection as module
    real_read = module._read
    changed = False
    def read(root, relative):
        nonlocal changed
        value = real_read(root, relative)
        if relative == "selection_binding.json" and not changed:
            changed = True
            _save(retained.root, "preview_job.json", {**retained.job, "state": "withdrawn"})
        return value
    monkeypatch.setattr(module, "_read", read)
    with pytest.raises(ValueError, match="snapshot_changed"):
        _resolve(retained)


@pytest.mark.parametrize("fault", ["purpose", "confirmation"])
def test_reuse_retains_the_existing_task_context_validation(retained, fault):
    if fault == "purpose":
        retained.context.pop("purpose")
    else:
        retained.context["confirmed_at"] = "2026-10-28T00:00:00Z"
    retained.context["context_digest"] = canonical_digest(retained.context, digest_field="context_digest")
    retained.plan["task_context_sha256"] = hashlib.sha256(json.dumps(retained.context, sort_keys=True,
                                                                   separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="purpose_mismatch|confirmation_invalid"):
        _resolve(retained)


def test_finished_geometry_cannot_be_joined_to_another_same_source_selection(retained):
    selection = deepcopy(retained.selection)
    proposal = selection["planner_receipt"]["proposal"]
    proposal["automation"]["objects"][0]["regions"][0]["box_xywh_normalized"] = [0.6, 0.2, 0.2, 0.4]
    selection["planner_receipt"]["proposal_digest"] = canonical_digest(proposal)
    _save(retained.root, "selection_binding.json", selection)
    with pytest.raises(ValueError, match="pose_changed"):
        _resolve(retained)
