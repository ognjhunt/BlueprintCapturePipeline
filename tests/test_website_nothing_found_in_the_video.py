# Covers (for impacted-test selection):
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/website_capture_preparation.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_company_simulation.py
#   src/blueprint_pipeline/website_atlas_task_selection.py
#   src/blueprint_pipeline/website_recapture_required.py
"""A video in which the analysis finds nothing the job is about is no stop (owner rule 2026-11-16).

Test job itest-8a15e775 (16 Nov, a navigation sentence on a dining-room clip) ended in 26 seconds asking the site
to film again. Now the clean plate goes on with nothing removed and no masks bought, flagged
(``clean_plate_stage.NOT_FOUND``), and the handoff ships the room package, saying so in the sealed record and the
README. Any other analysis blocker still stops exactly as before, also next to this one. People in frame are
erased from the views on the website path, as for every job; that is unchanged. Synthetic only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

NOT_FOUND = "task_relevant_target_not_observed"
TABLE = {"target_id": "dining_table", "semantic_label": "dining table", "segmentation_prompt": "table",
         "target_class": "fixed_clutter", "target_role": "destination", "task_effect": "uncertain",
         "disposition": "keep", "rebuild_intent": "none", "collision_required": False, "confidence": 0.4,
         "clarification_question": "Which table?", "decision_reason": "The job names a table to go to.",
         "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0.1, 0.5, 0.4, 0.3]}]}
PERSON = {"target_id": "person_1", "semantic_label": "person", "segmentation_prompt": "person",
          "target_class": "person", "target_role": "person", "task_effect": "privacy", "disposition": "remove",
          "rebuild_intent": "none", "collision_required": False, "confidence": 0.9,
          "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0.6, 0.2, 0.2, 0.6]}]}


def _context():
    value = {"schema_version": "website_site_task_context.v1", "request_id": "s1", "scene_id": "s1",
             "capture_id": "c1", "task_family": "navigation", "description": "Go from the doorway to the table.",
             "confirmed": True}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def _clean_plate(tmp_path, monkeypatch, *, targets, blockers):
    """The real clean-plate stage on a website capture whose analysis returns ``targets`` and ``blockers``."""
    from blueprint_pipeline import clean_plate_stage as stage
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
    from tests.test_clean_plate_stage import _make_capture
    capture = _make_capture(tmp_path)
    bought = []
    plan = build_removal_plan(targets=[dict(row) for row in targets], status="blocked" if blockers else "completed",
                              blockers=list(blockers), model="test", processing="agentic")
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **_: plan)
    monkeypatch.setattr(stage, "prepare_website_source_frames",
                        lambda **_: {"status": "estimated", "frames": [{"frame_id": "f0"}]})
    monkeypatch.setattr(stage, "run_website_task_masks", lambda **_: bought.append("sam_masks") or {"targets": []})
    result = stage.run_clean_plate_stage(capture_root=capture, website_source_video=capture / "raw/walkthrough.mp4",
                                         task_context=_context(),
                                         privacy_processing={"status": "pending_website_review"})
    return capture, result, bought


@pytest.mark.parametrize("found", ["nothing", "an_uncertain_table", "a_person_and_nothing_else"])
def test_the_clean_plate_goes_on_with_nothing_removed_and_buys_no_masks(tmp_path, monkeypatch, found):
    targets = {"nothing": [], "an_uncertain_table": [TABLE], "a_person_and_nothing_else": [PERSON]}[found]
    blockers = [NOT_FOUND] + (["task_target_requires_clarification:dining_table"] if found == "an_uncertain_table"
                              else [])
    capture, result, bought = _clean_plate(tmp_path, monkeypatch, targets=targets, blockers=blockers)
    assert (result["status"], result["blockers"], result["flags"]) == ("noop", [], [NOT_FOUND])
    assert result["privacy_verified"] is True and result["task_masks"] is None and bought == []
    assert result["person_target_count"] == (1 if found == "a_person_and_nothing_else" else 0)
    plan = json.loads((capture / "pipeline/clean_plate/removal_plan.json").read_text())
    assert (plan["status"], plan["flags"]) == ("completed", [NOT_FOUND])
    manifest = json.loads((capture / "pipeline/clean_plate/removal_manifest.json").read_text())
    assert not [row for row in manifest.get("entries") or [] if row.get("task_effect") == "manipulated"]


@pytest.mark.parametrize("blockers", [[NOT_FOUND, "removal_plan_target_class_invalid"], ["website_task_not_named"]],
                         ids=["beside_another_blocker", "another_blocker_alone"])
def test_any_other_analysis_blocker_still_stops_as_before(tmp_path, monkeypatch, blockers):
    _capture, result, bought = _clean_plate(tmp_path, monkeypatch, targets=[], blockers=blockers)
    assert result["status"] == "blocked" and result["privacy_verified"] is False and bought == []
    assert [code for code in blockers if code != NOT_FOUND][0] in result["blockers"] and "flags" not in result


def test_the_capture_stage_stops_on_a_blocked_clean_plate_and_asks_for_no_new_video(tmp_path):
    from types import SimpleNamespace

    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.common import StageError
    from tests.test_qualification_coverage_edges import _descriptor
    from tests.test_website_task_sentence_wait import NAMED
    descriptor = _descriptor(metadata={"site_task_context": NAMED})
    context = {**NAMED, "scene_id": descriptor.scene_id, "capture_id": descriptor.capture_id}
    descriptor = _descriptor(metadata={"site_task_context": context})
    capture = tmp_path / "scenes" / descriptor.scene_id / "captures" / descriptor.capture_id
    (capture / "pipeline").mkdir(parents=True)
    with pytest.raises(StageError):
        run_website_capture_preparation(
            descriptor=descriptor, capture_root=capture, pipeline_dir=capture / "pipeline", source_video=None,
            derived_scene_generation_allowed=True, gates=[],
            prepare_media=lambda **_: {"status": "blocked", "blockers": [NOT_FOUND, "x_other"]},
            check_privacy=lambda **_: SimpleNamespace(passed=True),
            make_gate=lambda _name, passed, _detail: SimpleNamespace(passed=passed))
    assert not (capture / "pipeline/website_scene_preparation/handoff.json").exists()


@pytest.mark.parametrize("family", ["navigation", "rigid_pick_place"])
def test_nothing_found_ships_the_room_package_and_says_so(tmp_path, monkeypatch, family):
    """The navigation sentence, and a pick-and-place one whose object is not in the video: the room, no rule from
    Blueprint, the finding said; no provider choice is made or recorded for a room package."""
    from blueprint_pipeline import artisan_provider_choice as choice
    from blueprint_pipeline import website_scene_handoff as handoff
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job
    asked = []
    monkeypatch.setattr(handoff, "artisan_provider_choice_for", lambda **kw: asked.append(kw))
    job, _context, result = room_job(tmp_path, monkeypatch, family=family, not_found=True)
    assert (result["status"], result["blockers"], result["job_disposition"]) == ("room_package_written", [], "accepted")
    record = json.loads(Path(result["room_package"]["path"]).read_text())
    assert record["flags"] == [NOT_FOUND] and record["success_rule_defined"] is False and record["job_objects"] is None
    assert record["task_kind"] == ("goal_region_navigation" if family == "navigation" else "not_defined_by_blueprint")
    if family == "navigation":
        assert record["goal_region"] == {"basis": "goal_not_placed:no_single_goal_mask"}
    _status, members = _published(job)
    assert sorted(members) == ["README.txt", "scene.usda"]
    assert _said(members["README.txt"]).endswith(f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} {sim.NOT_FOUND}")
    assert asked == [] and not list(job.capture.rglob(choice.CHOICE_FILE))


def test_a_nothing_found_package_holds_over_later_passes(tmp_path, monkeypatch):
    """Later ticks read what this one wrote: the handoff again (a release in between recompiles it), the
    builder twice, withdrawal."""
    from blueprint_pipeline import website_capture_withdrawal as withdrawal
    from tests.test_website_company_simulation import World
    from tests.test_website_scene_handoff_navigation import room_job
    job, context, result = room_job(tmp_path, monkeypatch, not_found=True)
    assert job.rerun() == result
    world = World(job)
    assert world.publish()["status"] == "published" and world.publish()["status"] == "current"
    assert job.rerun() == result and world.publish()["status"] == "current"
    withdrawal.acknowledge_withdrawal(capture_root=job.capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": context["request_id"],
        "scene_id": context["scene_id"], "capture_id": context["capture_id"], "withdrawal_id": "withdrawal-room",
        "requested_at_iso": "2026-11-16T00:00:00Z"})
    planned = {row["path"] for row in withdrawal.plan_local_cleanup(capture_root=job.capture)["files"]}
    room = [path for path in Path(result["room_package"]["path"]).parent.rglob("*") if path.is_file()]
    assert room and all(any(str(path).endswith(entry) for entry in planned) for path in room)


def test_a_rerun_keeps_the_analysis_that_found_nothing(tmp_path, monkeypatch):
    """The $0 re-run of a job held on this before 2026-11-16 keeps the analysis it has; it buys none again."""
    from blueprint_pipeline import website_atlas_task_selection as selection
    from tests.test_clean_plate_stage import _make_capture
    capture = _make_capture(tmp_path)
    plan = {"schema_version": "clean_plate_removal_plan.v1", "status": "blocked", "blockers": [NOT_FOUND],
            "targets": []}
    (capture / "pipeline/clean_plate").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(selection, "rerun_at_no_cost", lambda _root: "registration_reevaluation")
    monkeypatch.setattr(selection, "_source_binding", lambda *_: True)
    for value, kept in ((plan, True), ({**plan, "blockers": [NOT_FOUND, "x_other"]}, False)):
        (capture / "pipeline/clean_plate/removal_plan.json").write_text(json.dumps(value))
        found = selection.load_rerun_analysis(capture_root=capture, task_context=_context(),
                                              source_video=capture / "raw/walkthrough.mp4")
        assert found == (value if kept else None)


def test_a_hold_an_earlier_release_wrote_still_reads_as_held(tmp_path):
    """Until the release re-evaluates it, the status the site reads is the recapture hold, as before."""
    from blueprint_pipeline.website_recapture_required import FIELD, NOT_OBSERVED_CODE, record, retains_recapture_required
    context = _context()
    handoff = {"schema_version": "website_scene_handoff.v1", "scene_id": "s1", "capture_id": "c1",
               "status": "needs_input", "blockers": [NOT_OBSERVED_CODE],
               FIELD: record(NOT_OBSERVED_CODE, task_context_digest=context["context_digest"]),
               "simulator_ready": False, "provider_mutation_performed": False, "claim_ceiling": "development_only"}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    assert retains_recapture_required(handoff, context)


def test_a_room_package_keeps_a_caller_s_flags_and_the_clean_plate_s_in_order_once():
    from blueprint_pipeline.website_scene_handoff import room_flags
    caller = ["website_registration_ambiguous", "website_task_object_rigidity_unknown"]
    assert room_flags(caller, {"flags": [NOT_FOUND]}) == [*caller, NOT_FOUND]
    assert room_flags([*caller, NOT_FOUND, caller[0]], {"flags": [NOT_FOUND]}) == [*caller, NOT_FOUND]
    assert room_flags(caller, {}) == caller and room_flags((), {"flags": [NOT_FOUND]}) == [NOT_FOUND]
