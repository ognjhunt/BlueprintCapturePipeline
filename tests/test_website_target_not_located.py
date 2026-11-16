# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_masks.py
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_company_simulation.py
"""The job's object was seen but could not be outlined: no stop (test job itest-ff80a41b, 16 Nov).

Job 3's analysis gave an anchor box rounded past the frame's bottom edge (y 0.7371 + h 0.263 = 1.0001), and the
clean plate stopped on ``task_target_spatial_anchor_invalid``. Such a box is now clipped to the frame. What still
cannot be outlined (an unreadable anchor, no tracks, no instance to pick) goes on flagged
``task_target_not_located``: nothing removed, no masks, the room package that says so. A person to mask, a
refusal about money, or a video whose frames cannot be read still stops. Synthetic only.
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
    ([-0.003, 0.2, 0.3, 0.3], True),           # just left of the frame, rounding
    ([0.551, 0.7371, 0.203, 0.27], False),     # 0.0071 past the edge: not rounding
    ([0.5, 0.5, 0.0, 0.2], False), ([0.5, 0.5, float("nan"), 0.2], False), ([1.002, 0.5, 0.001, 0.2], False)])
def test_an_anchor_box_rounded_past_the_frame_is_clipped_and_a_malformed_one_still_refused(box, scored):
    target = {**DRAWER, "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": box}]}
    frames = [{"frame_id": "f7", "timestamp_seconds": 3.5}]
    if scored:
        [(score, track)] = masks.anchor_scores(target=target, tracks=[_track()], frames=frames)
        assert track["track_id"] == "t1" and 0.0 <= score <= 1.0
        x, y, w, h = box
        inside = [max(x, 0.0), max(y, 0.0), min(x + w, 1.0) - max(x, 0.0), min(y + h, 1.0) - max(y, 0.0)]
        clipped = {**target, "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": inside}]}
        [(same, _)] = masks.anchor_scores(target=clipped, tracks=[_track()], frames=frames)
        assert score == pytest.approx(same, abs=1e-12)  # the box is read as its part inside the frame
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


@pytest.mark.parametrize("failure", [
    {"masks_fail": ValueError("task_target_spatial_anchor_invalid")},
    {"masks_fail": ValueError("task_target_spatial_anchor_missing")},
    {"masks_fail": ValueError("website_sam31_tracks_unavailable")},
    {"masks_fail": "ambiguous"}],
    ids=["anchor_invalid", "anchor_missing", "no_tracks", "ambiguous_left_over"])
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


@pytest.mark.parametrize("case", ["a_person_to_mask", "money", "another_refusal", "no_source_frames"])
def test_privacy_money_and_other_refusals_still_stop(tmp_path, monkeypatch, case):
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    targets = [DRAWER, PERSON] if case == "a_person_to_mask" else [DRAWER]
    failure = {"a_person_to_mask": {"masks_fail": ValueError("task_target_spatial_anchor_invalid")},
               "money": {"masks_fail": PaidResourceAdmissionBlocked(["paid_resource_budget_exhausted"])},
               "another_refusal": {"masks_fail": ValueError("website_sam31_retained_tracks_changed")},
               "no_source_frames": {"frames_fail": ValueError("website_video_undecodable")}}[case]
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
    assert sim.NOT_FOUND not in _said(members["README.txt"])
    both = {"flags": ["task_relevant_target_not_observed", NOT_LOCATED]}
    assert room_flags(["website_registration_ambiguous"], both) == [
        "website_registration_ambiguous", "task_relevant_target_not_observed", NOT_LOCATED]
