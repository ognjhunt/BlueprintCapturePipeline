# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_masks.py
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_company_simulation.py
"""The job's object was seen but could not be outlined: no stop (test job itest-ff80a41b, 16 Nov).

Job 3's analysis gave an anchor box rounded past the frame's bottom edge (y 0.7371 + h 0.263 = 1.0001), and the
clean plate stopped on ``task_target_spatial_anchor_invalid``. Such a box is now clipped to the frame. What still
cannot be located (an unreadable anchor, no instance to pick, no depth inside its outline) goes on flagged
``task_target_not_located``: nothing removed, no masks, the room package that says so. A person to mask, a
refusal about money, a refusal that may concern another target, or a video whose frames cannot be read still
stops. Synthetic only.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_task_masks as masks
from tests.test_website_nothing_found_in_the_video import PERSON, _context

NOT_LOCATED = "task_target_not_located"
DRAWER = {"target_id": "drawer_cabinet_assembly", "semantic_label": "drawer cabinet", "segmentation_prompt": "cabinet",
          "target_class": "movable_object", "target_role": "task_object", "task_effect": "manipulated",
          "disposition": "remove", "rebuild_intent": "rebuild_and_compose", "collision_required": True,
          "confidence": 0.95, "articulated_part": "top drawer", "articulation_kind": "prismatic",
          "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": [0.551, 0.7371, 0.203, 0.263]}]}


def _track(box_rows=(16, 20), box_cols=(11, 15)):
    """One instance in a 20 by 20 mask, as run lengths over the flattened rows."""
    runs = [{"start": row * 20 + box_cols[0], "length": box_cols[1] - box_cols[0]} for row in range(*box_rows)]
    return {"track_id": "t1", "label": DRAWER["target_id"],
            "observations": [{"source_frame_id": "f7", "height": 20, "width": 20, "runs": runs}]}


@pytest.mark.parametrize(("box", "scored"), [
    ([0.551, 0.7371, 0.203, 0.263], True),     # job 3's box: 0.0001 past the bottom edge, rounding
    ([0.551, 0.7371, 0.203, 0.27], False),     # 0.0071 past the edge: not rounding
    ([-0.003, 0.8, 0.25, 0.2], True),         # just left of the frame, over an instance at the left edge
    ([0.5, 0.5, 0.0, 0.2], False), ([0.5, 0.5, float("nan"), 0.2], False), ([1.002, 0.5, 0.001, 0.2], False)])
def test_an_anchor_box_rounded_past_the_frame_is_clipped_and_a_malformed_one_still_refused(box, scored):
    target = {**DRAWER, "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": box}]}
    frames = [{"frame_id": "f7", "timestamp_seconds": 3.5}]
    instance = _track(box_cols=(0, 4)) if box[0] < 0 else _track()
    if scored:
        [(score, track)] = masks.anchor_scores(target=target, tracks=[instance], frames=frames)
        assert track["track_id"] == "t1" and 0.0 <= score <= 1.0
        x, y, w, h = box
        inside = [max(x, 0.0), max(y, 0.0), min(x + w, 1.0) - max(x, 0.0), min(y + h, 1.0) - max(y, 0.0)]
        clipped = {**target, "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": inside}]}
        [(same, _)] = masks.anchor_scores(target=clipped, tracks=[instance], frames=frames)
        assert score == pytest.approx(same, abs=1e-12) and score > 0  # the box is read as its part inside the frame
    else:
        with pytest.raises(ValueError, match="task_target_spatial_anchor_invalid"):
            masks.anchor_scores(target=target, tracks=[_track()], frames=frames)


def _clean_plate(tmp_path, monkeypatch, *, targets, frames_fail=None, masks_fail=None):
    from blueprint_pipeline import clean_plate_stage as stage
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
    from tests.test_clean_plate_stage import _make_capture
    capture = _make_capture(tmp_path)
    plan = build_removal_plan(targets=[dict(row) for row in targets], status="completed", blockers=[],
                              model="test", processing="agentic")
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **_: plan)

    def frames(**_):
        if frames_fail is not None:
            raise frames_fail
        return {"status": "estimated", "frames": [{"frame_id": "f0"}]}

    def sam(**_):
        raise masks_fail
    monkeypatch.setattr(stage, "prepare_website_source_frames", frames)
    monkeypatch.setattr(stage, "run_website_task_masks", sam)
    result = stage.run_clean_plate_stage(capture_root=capture, website_source_video=capture / "raw/walkthrough.mp4",
                                         task_context=_context(), privacy_processing={"status": "pending_website_review"})
    return capture, result


def _ambiguous():
    from blueprint_pipeline.website_task_target_choice import TaskTargetAmbiguous
    return TaskTargetAmbiguous(DRAWER["target_id"])


GROUNDING_CODES = ["website_task_grounding_uncertain", "website_task_grounding_incomplete",
                   "website_task_grounding_frame_missing", "website_task_grounding_anchor_missing",
                   "website_gemini_unanswered_dispatches_exhausted"]


@pytest.mark.parametrize("failure", [
    {"masks_fail": ValueError("task_target_spatial_anchor_invalid")},
    {"masks_fail": ValueError("task_target_spatial_anchor_missing")},
    {"masks_fail": ValueError("task_target_has_no_estimated_geometry")},
    {"masks_fail": "ambiguous"},
    # The exact-frame look the segmenter falls back on (test job itest-917ca852, 19 Nov).
    *({"masks_fail": ValueError(code)} for code in GROUNDING_CODES)],
    ids=["anchor_invalid", "anchor_missing", "no_estimated_geometry", "ambiguous_left_over", *GROUNDING_CODES])
def test_an_object_that_cannot_be_outlined_goes_on_with_nothing_removed(tmp_path, monkeypatch, failure):
    failure = {key: (_ambiguous() if value == "ambiguous" else value) for key, value in failure.items()}
    capture, result = _clean_plate(tmp_path, monkeypatch, targets=[DRAWER],
                                   **({"masks_fail": ValueError("unused")} | failure))
    assert (result["status"], result["blockers"], result["flags"]) == ("noop", [], [NOT_LOCATED])
    assert result["privacy_verified"] is True and result["task_masks"] is None
    assert result["movable_removal_count"] == 0
    manifest = json.loads((capture / "pipeline/clean_plate/removal_manifest.json").read_text())
    assert manifest.get("entries") == []
    plan = json.loads((capture / "pipeline/clean_plate/removal_plan.json").read_text())
    assert [row["disposition"] for row in plan["targets"]] == ["keep"] and plan["flags"] == [NOT_LOCATED]


@pytest.mark.parametrize("case", ["a_person_to_mask", "money", "another_refusal", "no_source_frames", "no_tracks",
                                  "look_in_progress", "look_key_missing", "look_video_changed", "look_unreserved",
                                  "look_credit_depleted", "person_and_unfinished_look"])
def test_privacy_money_and_other_refusals_still_stop(tmp_path, monkeypatch, case):
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    from blueprint_pipeline.website_gemini_receipts import CreditDepleted
    targets = [DRAWER, PERSON] if case in {"a_person_to_mask", "person_and_unfinished_look"} else [DRAWER]
    failure = {"a_person_to_mask": {"masks_fail": ValueError("task_target_spatial_anchor_invalid")},
               "money": {"masks_fail": PaidResourceAdmissionBlocked(["paid_resource_budget_exhausted"])},
               "another_refusal": {"masks_fail": ValueError("website_sam31_retained_tracks_changed")},
               "no_tracks": {"masks_fail": ValueError("website_sam31_tracks_unavailable")},
               "no_source_frames": {"frames_fail": ValueError("website_video_undecodable")},
               # The look another run holds, a configuration fault, a changed input, a send with no reservation
               # on record and Gemini's credits running out are no answer about the object: they still stop.
               "look_in_progress": {"masks_fail": ValueError("website_gemini_request_in_progress")},
               "look_key_missing": {"masks_fail": ValueError("website_task_grounding_key_missing")},
               "look_video_changed": {"masks_fail": ValueError("website_task_grounding_video_changed")},
               "look_unreserved": {"masks_fail": ValueError("website_gemini_unanswered_call_spend_unreserved")},
               "look_credit_depleted": {"masks_fail": CreditDepleted()},
               "person_and_unfinished_look": {"masks_fail": ValueError("website_task_grounding_incomplete")}}[case]
    if case == "money":
        # Money stops by its type, whatever its code says, even if a code of its ever looked like a localisation.
        from blueprint_pipeline import clean_plate_stage as stage
        monkeypatch.setattr(stage, "LOCALISATION_CODES", stage.LOCALISATION_CODES | {"paid_resource_admission_blocked"})
    _capture, result = _clean_plate(tmp_path, monkeypatch, targets=targets,
                                    **({"masks_fail": ValueError("unused")} | failure))
    assert result["status"] == "blocked" and result["blockers"] and "flags" not in result


def test_an_object_not_outlined_ships_the_room_package_and_says_so(tmp_path, monkeypatch):
    from blueprint_pipeline.website_scene_handoff import room_flags
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job
    job, _context_, result = room_job(tmp_path, monkeypatch, family="rigid_pick_place", not_found=True,
                                      clean_plate_flag=NOT_LOCATED)
    assert (result["status"], result["blockers"]) == ("room_package_written", [])
    record = json.loads(open(result["room_package"]["path"]).read())
    assert record["flags"] == [NOT_LOCATED] and record["success_rule_defined"] is False
    _status, members = _published(job)
    assert sorted(members) == ["README.txt", "scene.usda"]
    assert _said(members["README.txt"]).endswith(f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} {sim.NOT_LOCATED}")
    assert record["job_objects"] is None
    assert sim.NOT_FOUND not in _said(members["README.txt"])
    both = {"flags": ["task_relevant_target_not_observed", NOT_LOCATED]}
    assert room_flags(["website_registration_ambiguous"], both) == [
        "website_registration_ambiguous", "task_relevant_target_not_observed", NOT_LOCATED]


def test_a_retained_not_located_plan_retries_once_on_the_answers_already_bought(tmp_path, monkeypatch):
    """A $0 re-run of a job floored as not located tries the outlining again on the original analysis.

    The segmentation answers are keyed by the task targets; the floor's ``keep`` targets would key a new
    request and buy the same masks again. Two runs: one reservation, the second run's answers retained.
    """
    import hashlib

    from blueprint_pipeline import clean_plate_stage as stage
    from blueprint_pipeline import website_task_masks as task_masks
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
    from tests.test_clean_plate_stage import _make_capture
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    capture = _make_capture(tmp_path)
    context = {key: value for key, value in _context().items() if key != "context_digest"} | {
        "confirmed_at": "2026-11-16T00:00:00Z"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    outside = {**DRAWER, "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": [0.5, 0.9, 0.2, 0.2]}]}
    plan = build_removal_plan(targets=[outside], status="completed", blockers=[], model="test", processing="agentic")
    plan.update(input_video_sha256=hashlib.sha256(b"RAWVIDEO").hexdigest(), task_context_sha256=hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    analyses, calls, reservations = [], [], []
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **_: analyses.append(1) or json.loads(json.dumps(plan)))
    geometry = {"digest": "sha256:geometry", "geometry_available": False, "binding": {"source_video_digest": "video"},
                "frames": [{"frame_id": "f7", "timestamp_seconds": 3.5, "width": 20, "height": 20}]}
    monkeypatch.setattr(stage, "prepare_website_source_frames", lambda **_: geometry)
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(task_masks, "prepare_continuous_video", lambda **_: (
        [{"source_frame_id": "f7", "decoded_pts_seconds": 3.5, "width": 20, "height": 20}], {"path": "clip.mp4"}))

    def sam(*, output_root, prompts, **_):
        calls.append(output_root)
        answers = output_root / "answers.json"  # stands in for run_meta_sam31's retained responses
        if not answers.is_file():
            reservations.append([prompt["text"] for prompt in prompts])
            answers.write_text("bought")
        return {"tracks": [_track()]}
    monkeypatch.setattr(task_masks, "run_meta_sam31", sam)

    def run():
        return stage.run_clean_plate_stage(capture_root=capture, website_source_video=capture / "raw/walkthrough.mp4",
                                           task_context=context, privacy_processing={"status": "pending_website_review"})
    first = run()
    assert (first["status"], first["flags"], len(calls), len(reservations)) == ("noop", [NOT_LOCATED], 1, 1)
    # The once-per-release $0 re-run (timed retries spent): the retained analysis, never a new purchase.
    (capture / "pipeline_job_ledger.json").write_text(json.dumps({"preparation_retry": {"kind": "retries_spent"}}))
    second = run()
    assert (second["status"], second["flags"], len(analyses)) == ("noop", [NOT_LOCATED], 1)
    assert len(calls) == 2 and calls[1] == calls[0] and len(reservations) == 1


CHAIR = {"target_id": "chair", "target_class": "movable_object", "target_role": "obstacle", "task_effect": "static_obstacle",
         "disposition": "keep", "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.1, 0.1, 0.2, 0.2]}]}
LID = {**DRAWER, "target_id": "lid"}


@pytest.mark.parametrize(("targets", "failure", "floored"), [
    ([DRAWER, CHAIR], "ambiguous:drawer_cabinet_assembly", True),
    ([DRAWER, CHAIR], "ambiguous:chair", False),            # about the chair: the drawer may be located fine
    ([DRAWER, CHAIR], "task_target_has_no_estimated_geometry", True),   # the kept chair is deferred, never selected
    ([DRAWER, {**CHAIR, "disposition": "remove"}], "task_target_has_no_estimated_geometry", False),
    ([DRAWER, CHAIR], "task_target_spatial_anchor_missing", True),
    ([DRAWER, {**CHAIR, "spatial_evidence": []}], "task_target_spatial_anchor_missing", False),
    ([DRAWER, CHAIR], "website_task_targets_resolve_to_same_instance", False),  # one object to move: not between them
    ([DRAWER, LID, CHAIR], "website_task_targets_resolve_to_same_instance", True),
    ([CHAIR], "task_target_spatial_anchor_invalid", False),  # no object to move at all
    ([DRAWER], "website_sam31_tracks_unavailable", False)])
def test_only_a_refusal_about_the_object_to_move_is_floored(targets, failure, floored):
    from blueprint_pipeline import clean_plate_stage as stage
    from blueprint_pipeline.website_task_target_choice import TaskTargetAmbiguous
    exc = TaskTargetAmbiguous(failure.split(":", 1)[1]) if failure.startswith("ambiguous:") else ValueError(failure)
    plan = {"targets": [dict(row) for row in targets], "movable_removal_count": 1}
    result = stage._not_located(plan, exc)
    assert (result is not None) == floored
    if floored:
        assert [row["disposition"] for row in result["targets"]] == ["keep"] * len(targets)
        assert result["flags"] == [NOT_LOCATED] and result[stage.FLOORED_FROM] == plan


def test_the_planner_removals_the_atlas_job_sealed_are_counted(tmp_path, monkeypatch):
    from blueprint_pipeline import website_scene_handoff as handoff
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job
    record = tmp_path / "c" / "pipeline" / "atlas" / "job_object_rule.json"
    record.parent.mkdir(parents=True)
    assert handoff.planner_removals_applied(tmp_path / "c") == 0
    for applied, count in ((False, 1), (True, 0)):
        decided = {"schema_version": "atlas_job_object_rule.v1", "applied": applied, "clean_plate_status": "blocked",
                   "planner_remove_count": 1}
        record.write_text(json.dumps({**decided, "digest": canonical_digest(decided, digest_field="digest")}))
        assert handoff.planner_removals_applied(tmp_path / "c") == count
    record.write_text(json.dumps({**decided, "applied": False, "digest": "sha256:" + "0" * 64}))
    assert handoff.planner_removals_applied(tmp_path / "c") == 0
    # The Atlas job took the drawer out as its planner asked: the package says so, after the not-located sentence.
    monkeypatch.setattr(handoff, "planner_removals_applied", lambda capture_root: 1)
    job, _context_, result = room_job(tmp_path / "job", monkeypatch, family="rigid_pick_place", not_found=True,
                                      clean_plate_flag=NOT_LOCATED)
    _status, members = _published(job)
    assert _said(members["README.txt"]).endswith(
        f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} {sim.NOT_LOCATED} {sim.OBJECTS_TAKEN_OUT.format(count=1)}")


@pytest.mark.parametrize("look", ["unfinished", "refined_noun"])
def test_job_917_the_segmenter_finds_nothing_for_the_analysis_noun_and_the_job_still_goes_on(
        tmp_path, monkeypatch, look):
    """Test job itest-917ca852 (19 Nov), its real path: the video segmenter found no track for the analysis noun,
    so the clean plate fell back on one exact-frame look (``website_task_grounding``). Its answer never finished and
    the job stopped at the clean plate with nothing shipped. Now an unfinished look is kept, read again at $0, and
    the job goes on flagged ``task_target_not_located``; a finished look's own noun is proved on its one frame and
    then outlines the object through the whole video. Only Gemini, SAM, ffmpeg and the website's reservations are
    faked; the clean plate, the task masks, the look and its receipts are the real ones."""
    import hashlib

    from PIL import Image

    from blueprint_pipeline import clean_plate_stage as stage
    from blueprint_pipeline import website_task_grounding as grounding
    from blueprint_pipeline import website_task_masks as task_masks
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from tests.test_clean_plate_stage import _make_capture
    from tests.test_website_task_grounding import Provider, answer
    capture = _make_capture(tmp_path)
    context = {key: value for key, value in _context().items() if key != "context_digest"} | {
        "confirmed_at": "2026-11-19T00:00:00Z", "description": "Open the top drawer under the hob."}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    box = [0.5, 0.7, 0.3, 0.3]
    target = {**DRAWER, "segmentation_prompt": "cabinet unit with drawers",
              "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": box}]}
    plan = build_removal_plan(targets=[target], status="completed", blockers=[], model="test", processing="agentic")
    plan.update(input_video_sha256=hashlib.sha256(b"RAWVIDEO").hexdigest(), task_context_sha256=hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **_: json.loads(json.dumps(plan)))
    geometry = {"digest": "sha256:geometry", "geometry_available": False, "binding": {"source_video_digest": "video"},
                "frames": [{"frame_id": "f7", "timestamp_seconds": 3.5, "width": 20, "height": 20}]}
    monkeypatch.setattr(stage, "prepare_website_source_frames", lambda **_: geometry)
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"continuous clip")
    monkeypatch.setattr(task_masks, "prepare_continuous_video", lambda **_: (
        [{"source_frame_id": "f7", "model_frame_index": 7, "decoded_pts_seconds": 3.5, "width": 20, "height": 20}],
        {"path": str(clip), "sha256": _sha256_file(clip)}))
    monkeypatch.setattr(grounding.subprocess, "run",
                        lambda args, **_: Image.new("RGB", (20, 20), "black").save(args[-1]))
    segmented = []

    def sam(*, prompts, video_artifact=None, **_):
        concept, label = prompts[0]["text"], prompts[0]["output_label"]
        segmented.append((concept, video_artifact is not None))
        if concept != "black drawer cabinet":
            return {"tracks": []}  # the analysis noun: no track anywhere in the clip, as on the host
        runs = [{"start": row * 20 + 10, "length": 6} for row in range(14, 20)]
        return {"tracks": [{"track_id": f"{label}-0", "label": label,
                            "observations": [{"source_frame_id": "f7", "width": 20, "height": 20, "runs": runs}]}]}
    monkeypatch.setattr(task_masks, "run_meta_sam31", sam)
    provider = Provider(monkeypatch, answer("MAX_TOKENS", text="") if look == "unfinished" else answer(
        text=json.dumps({"visible": True, "confidence": 0.9, "box_xywh_normalized": box,
                         "segmentation_prompt": "black drawer cabinet", "reason": "the drawers under the hob"})))

    def run():
        return stage.run_clean_plate_stage(capture_root=capture, website_source_video=capture / "raw/walkthrough.mp4",
                                           task_context=context, privacy_processing={"status": "pending_website_review"})
    first, second = run(), run()
    # One look bought, under one reservation; the second run reads it again.
    assert len(provider.sent) == 1 and len(provider.reservations) == 1
    if look == "unfinished":
        for result in (first, second):
            assert (result["status"], result["blockers"], result["flags"]) == ("noop", [], [NOT_LOCATED])
            assert result["task_masks"] is None and result["movable_removal_count"] == 0
        assert segmented == [("cabinet unit with drawers", True)] * 2
    else:
        for result in (first, second):
            assert (result["status"], result["blockers"]) == ("objects_removed", [])
            [selected] = result["task_masks"]["targets"]
            assert selected["segmentation_prompt"] == "black drawer cabinet"
            assert selected["grounding"]["source_frame_id"] == "f7"
        # Per run: the analysis noun on the clip, the look's noun on its one frame, then on the clip.
        assert segmented == [("cabinet unit with drawers", True), ("black drawer cabinet", False),
                             ("black drawer cabinet", True)] * 2
