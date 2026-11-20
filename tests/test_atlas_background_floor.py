"""An object the editor cannot take out cleanly stays in the room as filmed (gap G, owner 2026-11-16).

Job itest-ff80 (open a drawer, 16 Nov): the Atlas planner asked to remove ``candidate_top_drawer``; SAM left ten
selected views without its outline, so each got one exact-frame look. Nine answered "visible" (confidence 0.90-0.95);
the tenth (``decoded-000000132``) came back cut off, which raised ``atlas_removal_visibility_incomplete`` and ended the
Atlas job ``needs_input`` with no retry. Now a cut-off look is kept as "not seen" and never bought again, and when the
editor still cannot take the object out (that stop, or a review that never passes) the job takes the existing
as-filmed route (``_preserved_scene`` with ``job_object_rule``): the object stays in the scan, the handoff ships the
room package with the job's objects part of it and builds no separate object. People are never left in: with a
person in view nothing here applies.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from blueprint_pipeline import atlas_removal_visibility as visibility
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_gemini_receipts as receipts
from blueprint_pipeline.atlas_automation_plan import JOB_OBJECT_RULE
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.test_website_atlas_pipeline import _job
from tests.test_website_image_completion import _grant

FLOOR = "atlas_removal_visibility_incomplete"
DRAWER = [{"target_id": "drawer_cabinet_assembly", "semantic_label": "drawer cabinet", "task_effect": "manipulated",
           "disposition": "remove"}]


def _review(*, transients=()):
    """Job itest-ff80's retained Atlas review in shape: one object to remove, the rest of the kitchen kept."""
    objects = [{"target_id": key, "label": key, "segmentation_prompt": key, "disposition": disposition,
                "regions": [{"frame_id": "f0", "box_xywh_normalized": [.1, .1, .2, .2]}]}
               for key, disposition in (("candidate_top_drawer", "remove"), ("supporting_cabinetry", "keep"),
                                        ("countertop", "keep"))]
    return {"subject_origin": "captured", "uncaptured_regions": [],
            "coverage_requirements": [{"role": "task_object", "label": "t", "required_frame_ids": ["f0"]}],
            "automation": {"scale_references": [], "objects": objects, "removal_instruction": "the planner's words",
                           "mask_margin_fraction": .01},
            **({"transient_frame_ids": list(transients),
                "transients": [{"target_id": "transient-0", "label": "person", "segmentation_prompt": "person"}]}
               if transients else {})}


# --- the look at one view: a cut-off answer is "not seen", kept, never bought again ----------------------------

class Look:
    def __init__(self, monkeypatch, finish="MAX_TOKENS", text="{}"):
        self.calls, self.reservations, self.finish, self.text = 0, [], finish, text
        look = self

        class Client:
            def __init__(self, **_):
                self.models = self

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def generate_content(self, **_):
                look.calls += 1
                return NS(candidates=[NS(finish_reason=look.finish)], text=look.text)

        types = NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=lambda **kw: NS(**kw),
                   HttpOptions=lambda **kw: NS(**kw), HttpRetryOptions=lambda **kw: NS(**kw))
        monkeypatch.setitem(sys.modules, "google.genai", NS(Client=Client, types=types))
        monkeypatch.setattr(visibility, "_api_key", lambda: ("fixture-key", "fixture"))

        def reserve(**kwargs):
            self.reservations.append(kwargs)
            admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted",
                         "resource_class": kwargs["resource_class"], "blockers": [],
                         "allocation_binding_digest": kwargs["binding_digest"]}
            return admission, _grant(admission)
        monkeypatch.setattr(receipts, "reserve_website_preparation_spend", reserve)
        monkeypatch.setattr(receipts, "require_current_website_source", lambda _context: None)


def _look_args(tmp_path):
    image = tmp_path / "decoded-000000132.png"
    Image.new("RGB", (16, 9), "grey").save(image)
    task = {"schema_version": "website_site_task_context.v1", "request_id": "itest", "scene_id": "site-itest",
            "capture_id": "walkthrough-itest", "confirmed": True, "confirmed_at": "2026-11-16",
            "description": "Open the top drawer."}
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    return dict(target={"target_id": "candidate_top_drawer", "label": "top drawer", "segmentation_prompt": "drawer"},
                frame={"frame_id": "decoded-000000132", "image_path": str(image), "image_digest": _sha256_file(image)},
                task_context=task, output_root=tmp_path / "visibility_checks", source_video_digest="sha256:" + "1" * 64)


@pytest.mark.parametrize("finish,text", [("MAX_TOKENS", "{"), ("STOP", "not json")])
def test_two_unusable_looks_buy_exactly_two_and_then_the_view_is_not_seen_for_good(tmp_path, monkeypatch,
                                                                                  finish, text):
    """Owner's rule: a cut-off call is counted at its full quote and made again, once; the second answer is final."""
    look = Look(monkeypatch, finish=finish, text=text)
    args = _look_args(tmp_path)
    for _ in range(3):
        assert visibility.locate_target(**args) is None
    assert look.calls == 2 and len(look.reservations) == 2
    names = sorted(p.name.split(".", 1)[1] for p in args["output_root"].glob("*.json"))
    assert names == ["json", "resend-1.json", "unanswered-1.json"]
    final = json.loads(next(args["output_root"].glob("*.resend-1.json")).read_text())
    assert final["status"] == "completed" and final["result"]["observation"] is None


def test_one_resend_then_its_answer_is_final(tmp_path, monkeypatch):
    look = Look(monkeypatch)  # the first answer is cut off
    args = _look_args(tmp_path)
    real = visibility.retained_gemini_call

    def answered_on_resend(**kwargs):
        if look.calls == 1:
            look.finish, look.text = "STOP", json.dumps({"visible": True, "confidence": .95,
                                                          "box_xywh_normalized": [.6, .7, .2, .2]})
        return real(**kwargs)
    monkeypatch.setattr(visibility, "retained_gemini_call", answered_on_resend)
    assert visibility.locate_target(**args) == [.6, .7, .2, .2]
    assert visibility.locate_target(**args) == [.6, .7, .2, .2]
    assert look.calls == 2 and len(look.reservations) == 2


def test_a_good_look_is_never_bought_again(tmp_path, monkeypatch):
    look = Look(monkeypatch, finish="STOP", text=json.dumps({"visible": True, "confidence": .95,
                                                              "box_xywh_normalized": [.6, .7, .2, .2]}))
    args = _look_args(tmp_path)
    for _ in range(3):
        assert visibility.locate_target(**args) == [.6, .7, .2, .2]
    assert look.calls == 1 and len(look.reservations) == 1


def test_job_three_view_132_is_bought_once_more_at_most_and_then_kept(tmp_path, monkeypatch):
    """Its receipt as the host has it: one ``submitting`` dispatch (the old release raised after the answer)."""
    look = Look(monkeypatch)
    args = _look_args(tmp_path)
    real = visibility.retained_gemini_call

    def old_release(**kwargs):
        return real(**{**kwargs, "invoke": lambda: (_ for _ in ()).throw(ValueError(FLOOR))})
    monkeypatch.setattr(visibility, "retained_gemini_call", old_release)
    with pytest.raises(ValueError, match=FLOOR):
        visibility.locate_target(**args)
    assert [json.loads(p.read_text())["status"] for p in args["output_root"].glob("*.json")] == ["submitting"]
    monkeypatch.setattr(visibility, "retained_gemini_call", real)
    assert visibility.locate_target(**args) is None
    assert len(look.reservations) == 2 and look.calls == 1  # one resend, under its own $0.02 quote
    assert visibility.locate_target(**args) is None and len(look.reservations) == 2


def test_a_complete_confident_look_still_outlines_the_view(tmp_path, monkeypatch):
    Look(monkeypatch, finish="STOP", text=json.dumps({"visible": True, "confidence": .95,
                                                       "box_xywh_normalized": [.6, .7, .2, .2]}))
    assert visibility.locate_target(**_look_args(tmp_path)) == [.6, .7, .2, .2]


# --- the Atlas job: the editor's stop becomes the scene as filmed -----------------------------------------------

def _stopped(monkeypatch, automation):
    from blueprint_pipeline import paid_resource_allocator as allocator
    calls = []
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation",
                        lambda **kw: calls.append(kw) or automation)
    return calls


@pytest.mark.parametrize("automation,reason", [
    ({"status": "needs_input", "background": {"status": "needs_input", "reason": FLOOR}}, FLOOR),
    ({"status": "needs_input", "background": {"status": "needs_input",
                                              "reason": "atlas_object_mask_view_coverage_incomplete"}},
     "atlas_object_mask_view_coverage_incomplete"),
    ({"status": "needs_input", "background": {"status": "needs_input", "reason": "atlas_removal_masks_empty"}},
     "atlas_removal_masks_empty"),
    ({"status": "quality_rejected", "background": {"status": "quality_rejected"}}, "atlas_background_review_required"),
])
def test_job_three_goes_on_as_filmed_when_its_object_cannot_be_taken_out(tmp_path, monkeypatch, automation, reason):
    # Its clean plate ended blocked (gap F), so the planner's removal stood (``job_object_rule.json``: not applied).
    root, run, _automated, posed = _job(tmp_path, monkeypatch, _review(), clean_plate=("blocked", []))
    calls = _stopped(monkeypatch, automation)
    assert run() == {"status": "processing", "stage": "source_pose"}
    assert len(calls) == 1 and len(posed) == 1
    consumed = posed[0]["review"]
    assert [row["disposition"] for row in consumed["automation"]["objects"]] == ["keep"] * 3
    assert consumed["automation"]["removal_instruction"] == ""
    assert consumed["job_object_rule"] == {"code": JOB_OBJECT_RULE, "planner_remove_count": 1, "removed_count": 0,
                                           "background_not_cleaned": reason}
    assert json.loads((root / "job_object_rule.json").read_text())["applied"] is False
    sealed = json.loads((root / pipeline.BACKGROUND_FLOOR_RECORD).read_text())
    assert sealed["reason"] == reason and sealed["digest"] == canonical_digest(sealed, digest_field="digest")
    # A later step reads the decision and never runs the editor again.
    assert run() == {"status": "processing", "stage": "source_pose"} and len(calls) == 1 and len(posed) == 2
    assert posed[1]["review"] == consumed


@pytest.mark.parametrize("has_pose,has_room", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("raised", [False, True])
def test_spend_cap_uses_only_retained_room_evidence(tmp_path, monkeypatch, has_pose, has_room, raised):
    from blueprint_pipeline import atlas_depth_splat, atlas_result_bundle, atlas_source_pose_inputs
    root, run, _automated, posed = _job(tmp_path, monkeypatch, _review(), clean_plate=("blocked", []))
    calls = _stopped(monkeypatch, {"status": "needs_input", "background": {"reason": "spending_cap_reached"}})
    source = root / "pose/full-view-source-v1"
    source.mkdir(parents=True)
    if raised:
        from blueprint_pipeline import paid_resource_allocator
        from blueprint_pipeline.atlas_video_inputs import GEOMETRY_INPUTS_FILE
        from blueprint_pipeline.hard_stops import MoneyStop
        (source.parent / GEOMETRY_INPUTS_FILE).write_text(json.dumps(
            {"binding": {"source_video_digest": _sha256_file(tmp_path / "source.mov")}}))

        def refuse(**kwargs):
            calls.append(kwargs)
            raise MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")

        monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_automation", refuse)
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda _: (source, {}))
    pose = {"digest": "pose", "operation": {"id": "retained"}}
    if has_room:
        scale = {"status": "estimated", "meters_per_unit_estimate": 1, "binding": {"pose_digest": "pose"}}
        scale["digest"] = canonical_digest(scale, digest_field="digest")
        (source / "scene_measurements.json").write_text(json.dumps(scale))
        monkeypatch.setattr(atlas_result_bundle, "find_retained_mesh_root", lambda **_: root / "mesh")
        monkeypatch.setattr(atlas_result_bundle, "retained_mesh_for_splat", lambda **_: {})
    if has_pose:
        (source / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
        monkeypatch.setattr(atlas_depth_splat, "load_posed_rgbd", lambda _: (pose, []))
        monkeypatch.setattr(atlas_depth_splat, "write_depth_splat",
                            lambda **_: {"splat_kind": "depth_fused_draft", "digest": "draft"})
        monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle", lambda **_: {"zip_path": "retained.zip"})
    for _ in range(2):
        result = run()
        assert result["reason"] == "spending_cap_reached"
        assert bool(result.get("download")) is has_pose
        assert bool(result.get("mesh_available")) is has_room
        if has_room:
            assert result["source_pose"]["status"] == "completed"
            assert result["source_pose"]["pose_relative_path"] == "pose/full-view-source-v1"
        assert not posed and len(calls) == 1


@pytest.mark.parametrize("automation", [
    {"status": "needs_input", "background": {"status": "needs_input", "reason":
        "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"}},
    {"status": "needs_input", "background": {"status": "needs_input",
                                             "reason": "website_image_completion_requires_reconciliation"}},
    {"status": "needs_input", "background": {"status": "needs_input", "reason": "atlas_edit_source_changed"}},
    {"status": "needs_input", "background": {"status": "needs_input", "reason": "something_nobody_classified"}},
    {"status": "needs_input", "background": {"status": "quality_rejected"}},
    {"status": "some_new_status", "background": {"status": "some_new_status", "reason": FLOOR}},
])
def test_any_other_editor_stop_is_unchanged(tmp_path, monkeypatch, automation):
    root, run, _automated, posed = _job(tmp_path, monkeypatch, _review(), clean_plate=("blocked", []))
    _stopped(monkeypatch, automation)
    result = run()
    assert not posed and result["status"] == "needs_input" and "job_object_rule" not in result
    assert not (root / pipeline.BACKGROUND_FLOOR_RECORD).exists()


@pytest.mark.parametrize("transients", [["f0"], ["f9"]])
def test_a_person_who_cannot_be_erased_still_stops_the_job(tmp_path, monkeypatch, transients):
    root, run, _automated, posed = _job(tmp_path, monkeypatch, _review(transients=transients),
                                        clean_plate=("blocked", []))
    _stopped(monkeypatch, {"status": "quality_rejected", "background": {"status": "quality_rejected"}})
    result = run()
    assert not posed and result["blockers"] == ["atlas_background_review_required"]
    assert not (root / pipeline.BACKGROUND_FLOOR_RECORD).exists()


def test_a_tampered_decision_is_refused(tmp_path, monkeypatch):
    root, run, _automated, _posed = _job(tmp_path, monkeypatch, _review(), clean_plate=("blocked", []))
    _stopped(monkeypatch, {"status": "quality_rejected", "background": {"status": "quality_rejected"}})
    run()
    sealed = json.loads((root / pipeline.BACKGROUND_FLOOR_RECORD).read_text())
    (root / pipeline.BACKGROUND_FLOOR_RECORD).write_text(json.dumps({**sealed, "reason": FLOOR}))
    with pytest.raises(ValueError, match="atlas_background_floor_record_invalid"):
        run()


def test_only_a_website_captures_own_atlas_job_takes_the_floor(tmp_path):
    from blueprint_pipeline.owned_capture_atlas_context import SCHEMA as OWNED
    stop = {"status": "quality_rejected", "background": {"status": "quality_rejected"}}
    owned = tmp_path / "pipeline/atlas"
    owned.mkdir(parents=True)
    assert pipeline.background_floor(tmp_path / "elsewhere", _review(), {}, stop) is None
    assert pipeline.background_floor(owned, _review(), {"schema_version": OWNED}, stop) is None
    assert not (owned / pipeline.BACKGROUND_FLOOR_RECORD).exists()
    assert pipeline.background_floor(owned, _review(), {}, stop)["job_object_rule"]["code"] == JOB_OBJECT_RULE


# --- the handoff: the object is part of the scan, never also built on its own ----------------------------------

@pytest.mark.parametrize("not_cleaned", [True, False])
def test_the_room_package_holds_the_job_object_as_filmed_and_no_separate_object(tmp_path, monkeypatch, not_cleaned):
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    from tests.test_website_scene_handoff_navigation import _published, _record, _said, room_job

    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    rule = {"code": JOB_OBJECT_RULE, "planner_remove_count": 1, "removed_count": 0,
            **({"background_not_cleaned": FLOOR} if not_cleaned else {})}
    job, _context, result = room_job(tmp_path, monkeypatch, family="articulated_open_close",
                                     description="Open the top drawer.", removal=DRAWER, all_targets=True,
                                     job_object_rule=rule)
    if not not_cleaned:
        # A plain job-object rule never names an object the clean plate took (gap C1): not this route.
        assert result["status"] != "room_package_written" or _record(result)["job_objects"]["in_package"] != (
            "part_of_room_scan")
        return
    assert (result["status"], result["job_disposition"]) == ("room_package_written", "accepted")
    assert "website_intake_outbox" not in result and "source_registration" not in result  # no object built
    record = _record(result)
    assert record["job_objects"] == {"count": 1, "in_package": "part_of_room_scan"}
    assert record["flags"] == [JOB_OBJECT_RULE] and record["task_profile"] == "articulated_open_close"
    assert job.rerun() == result
    said = _said(_published(job)[1]["README.txt"])
    assert said.endswith(" ".join([sim.ROOM_NOT_RUN, sim.NO_RULE, sim.OBJECTS_IN_SCAN, sim.NOTHING_TAKEN_OUT]))


def test_a_job_with_no_object_of_its_own_keeps_its_route_when_the_scene_is_as_filmed(tmp_path, monkeypatch):
    # A navigation job: nothing the clean plate took, so nothing is in the scan twice; its goal region stays.
    from tests.test_website_scene_handoff_navigation import _record, room_job
    rule = {"code": JOB_OBJECT_RULE, "planner_remove_count": 1, "removed_count": 0, "background_not_cleaned": FLOOR}
    _job_, _context, result = room_job(tmp_path, monkeypatch, job_object_rule=rule)
    record = _record(result)
    assert record["task_kind"] == "goal_region_navigation" and "centre_world_m" in record["goal_region"]
    assert record["job_objects"] is None and record["flags"] == [JOB_OBJECT_RULE]


# --- one truth for "was anything erased from the scan that ships" (with gap F's not-located room) -------------

@pytest.mark.parametrize("floored", [True, False])
def test_the_package_says_what_the_shipped_scan_lost_with_every_sentence_together(tmp_path, monkeypatch, floored):
    """The Atlas job sealed its planner's removal (``applied: false``), the clean plate found nothing and could not
    outline the object (gap F; it is the clean plate's other as-filmed outcome beside "not found", never both), and
    the gap views were not filled in. Floored: the scan shipped as filmed, so nothing was
    taken out, whatever the sealed decision said. Not floored: the planner's removal stands and is counted."""
    from blueprint_pipeline import website_atlas_scene_source as source
    from blueprint_pipeline import website_scene_handoff as handoff
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    from tests.test_website_scene_handoff_navigation import ATLAS, _published, _record, _said, room_job

    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    monkeypatch.setattr(handoff, "planner_removals_applied", lambda capture_root: 1)  # the sealed `applied: false`
    rule = {"code": JOB_OBJECT_RULE, "planner_remove_count": 1, "removed_count": 0, "background_not_cleaned": FLOOR}
    job, _context, _result = room_job(tmp_path, monkeypatch, family="rigid_pick_place", not_found=True,
                                      clean_plate_flag="task_target_not_located",
                                      job_object_rule=rule if floored else None)
    atlas = source.atlas_scene_source()
    monkeypatch.setattr(ATLAS, lambda **_: {**atlas, "flags": ["scene_completion_not_admitted"]})
    result = job.rerun()
    record = _record(result)
    said = _said(_published(job)[1]["README.txt"])
    if floored:
        assert record["job_objects"] is None
        assert record["flags"][-3:] == ["task_target_not_located", JOB_OBJECT_RULE, "scene_completion_not_admitted"]
        assert said.endswith(" ".join([sim.ROOM_NOT_RUN, sim.NO_RULE, sim.NOT_LOCATED, sim.NOTHING_TAKEN_OUT, sim.NOT_FILLED_IN]))
        assert "taken out of the room scan and are not in this package" not in said
        assert sim.OBJECTS_IN_SCAN not in said
    else:
        assert record["job_objects"] == {"count": 1, "in_package": "taken_out_of_room_scan"}
        assert said.endswith(" ".join([sim.ROOM_NOT_RUN, sim.NO_RULE, sim.NOT_LOCATED, sim.OBJECTS_TAKEN_OUT_ONE, sim.NOT_FILLED_IN]))
        assert sim.NOTHING_TAKEN_OUT not in said


def test_a_nested_analysis_bound_to_another_source_is_refused(tmp_path, monkeypatch):
    """Gap F keeps the analysis a not-located plan was floored from; it is held to the same binding as the plan."""
    import hashlib

    from blueprint_pipeline import clean_plate_stage as stage
    from blueprint_pipeline import website_task_masks as task_masks
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import build_removal_plan
    from tests.test_clean_plate_stage import _make_capture
    from tests.test_website_nothing_found_in_the_video import _context
    from tests.test_website_target_not_located import DRAWER as LOCATED_DRAWER, _track
    capture = _make_capture(tmp_path)
    context = {key: value for key, value in _context().items() if key != "context_digest"} | {
        "confirmed_at": "2026-11-16T00:00:00Z"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    outside = {**LOCATED_DRAWER, "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": [.5, .9, .2, .2]}]}
    plan = build_removal_plan(targets=[outside], status="completed", blockers=[], model="test", processing="agentic")
    plan.update(input_video_sha256=hashlib.sha256(b"RAWVIDEO").hexdigest(), task_context_sha256=hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    monkeypatch.setattr(stage, "analyze_removal_targets", lambda **_: json.loads(json.dumps(plan)))
    geometry = {"digest": "sha256:geometry", "geometry_available": False, "binding": {"source_video_digest": "video"},
                "frames": [{"frame_id": "f7", "timestamp_seconds": 3.5, "width": 20, "height": 20}]}
    monkeypatch.setattr(stage, "prepare_website_source_frames", lambda **_: geometry)
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(task_masks, "prepare_continuous_video", lambda **_: (
        [{"source_frame_id": "f7", "decoded_pts_seconds": 3.5, "width": 20, "height": 20}], {"path": "clip.mp4"}))
    monkeypatch.setattr(task_masks, "run_meta_sam31", lambda **_: {"tracks": [_track()]})

    def run():
        return stage.run_clean_plate_stage(capture_root=capture, website_source_video=capture / "raw/walkthrough.mp4",
                                           task_context=context, privacy_processing={"status": "pending_website_review"})
    assert run()["flags"] == ["task_target_not_located"]
    (capture / "pipeline_job_ledger.json").write_text(json.dumps({"preparation_retry": {"kind": "retries_spent"}}))
    path = capture / "pipeline/clean_plate/removal_plan.json"
    retained = json.loads(path.read_text())
    assert retained[stage.FLOORED_FROM]["input_video_sha256"] == plan["input_video_sha256"]
    assert run()["flags"] == ["task_target_not_located"]  # bound: the re-run reads it
    retained[stage.FLOORED_FROM]["input_video_sha256"] = "0" * 64
    path.write_text(json.dumps(retained))
    with pytest.raises(ValueError, match="website_not_located_analysis_binding_changed"):
        run()
