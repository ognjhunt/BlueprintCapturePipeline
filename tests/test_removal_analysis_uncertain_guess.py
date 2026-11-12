"""An uncertain target is a recorded guess, not a stop; an analysis from an older instruction is redone once.

Fixtures keep the shape of the retained analyses of capture-c33427c3 (a box the task let the
analysis choose, left uncertain at confidence 0.5) and capture-98a79ce3 (a vase left uncertain as
an obstacle), with identifiers and site details replaced.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
from blueprint_pipeline import website_gemini_receipts
from blueprint_pipeline.website_atlas_task_selection import (
    load_retained_uncertain_analysis, resolve_retained_target_selection)
from tests.test_website_atlas_task_selection import _save, retained  # noqa: F401 - fixture

#: The retained analyses' own template digests (removal_plan.json ``prompt_template_sha256``).
BEFORE_CHOICE_RULE = "2bf3145b3e2db5924668890a794e1ab114ae82fd7bd29869f2ff790d7ddd42cf"  # c33427c3
OLDER_WITH_CHOICE_RULE = "81b37f534a4db29b622d7b7bc12031cc2815a43e0adb4c6f53340a729c8e853b"  # 98a79ce3, f88f1bd0


def _box(target_id="cardboard_box_candidate", confidence=0.5, evidence=((14.5, [0.038, 0.001, 0.207, 0.404]),
                                                                       (29.5, [0.015, 0.589, 0.165, 0.236]))):
    return {"target_id": target_id, "semantic_label": "cardboard box on shelf", "segmentation_prompt": "cardboard box",
            "target_class": "movable_object", "target_role": "task_object", "task_effect": "uncertain",
            "disposition": "keep", "rebuild_intent": "none", "collision_required": False, "confidence": confidence,
            "clarification_question": "Which box?", "decision_reason": "Several boxes fit the task.",
            "task_basis_quote": "Move one cardboard box from a shelf to an empty spot on the same shelf.",
            "spatial_evidence": [{"timestamp_seconds": t, "box_xywh_normalized": b} for t, b in evidence]}


SHELF = {"target_id": "storage_rack_shelf", "semantic_label": "warehouse shelf rack", "segmentation_prompt": "metal rack",
         "target_class": "fixed_clutter", "target_role": "support", "task_effect": "static_contact",
         "placement_relation": "on", "disposition": "keep", "rebuild_intent": "none", "collision_required": True,
         "confidence": 0.85, "spatial_evidence": [{"timestamp_seconds": 14.5, "box_xywh_normalized": [0, 0, 0.38, 0.99]}]}


def _plan(targets, blockers, template=None):
    plan = analysis.build_removal_plan(targets=targets, status="blocked", model="m", processing="static",
                                       blockers=blockers)
    if template is not None:
        plan["prompt_template_sha256"] = template
        plan.pop("prompt_revision")
    return plan


def test_revisions_are_named_and_only_complete_current_answers_stand():
    assert analysis.analysis_revision(_plan([], [], BEFORE_CHOICE_RULE)) == analysis.OLDER_PROMPT_REVISION
    # Its retained prompt has the choice rule and no destination region: r1, read as what it is.
    assert analysis.analysis_revision(_plan([], [], OLDER_WITH_CHOICE_RULE)) == analysis.PRIOR_PROMPT_REVISION
    assert analysis.analysis_revision(_plan([], [])) == analysis.PROMPT_REVISION
    completed = {"status": "completed", "blockers": []}
    assert analysis.binds_current_job({**completed, "prompt_revision": analysis.REGION_PROMPT_REVISION})
    assert not analysis.binds_current_job({**completed, "prompt_revision": analysis.PRIOR_PROMPT_REVISION})
    assert not analysis.binds_current_job({"status": "blocked", "blockers": ["x"],
                                           "prompt_revision": analysis.REGION_PROMPT_REVISION})
    assert analysis.binds_current_job({"status": "blocked", "blockers": ["x"], "prompt_revision": analysis.PROMPT_REVISION})


def test_only_the_region_revisions_completed_answer_is_reused(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(website_gemini_receipts, "retained_gemini_call", lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr(analysis, "_video_processing", lambda path, mode: ("static", 14.4))
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video")
    context = {"capture_id": "walkthrough-capture-x", "purpose": "scene_preparation", "confirmed": False,
               "description": "Move one cardboard box from a shelf to an empty spot on the same shelf."}
    analysis.analyze_removal_targets(video_path=video, output_root=tmp_path / "r", task_context=context)
    (region, _), (prior, _) = seen["earlier_requests"]
    accept = seen["accept_earlier"]
    assert accept(region, {"status": "completed", "blockers": []})
    assert not accept(region, {"status": "blocked", "blockers": ["task_target_requires_clarification:x"]})
    assert not accept(prior, {"status": "completed", "blockers": []})


def test_retained_receipts_gate_honours_the_acceptance(tmp_path, monkeypatch):
    from tests.test_website_gemini_receipts import context, setup_call
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    earlier = {"kind": "earlier"}
    _, digest = website_gemini_receipts._request(binding=earlier, task_context=context(), maximum_cost_usd=.1,
                                                 source_video_digest=None)
    root = tmp_path / "receipts"
    root.mkdir()
    old = {"status": "blocked", "targets": []}
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    (root / f"{digest[7:]}.json").write_text(json.dumps({"status": "completed", "request_digest": digest,
                                                          "result": old, "result_digest": canonical_digest(old)}))
    common = dict(output_root=root, binding={"kind": "current"}, task_context=context(), maximum_cost_usd=.1,
                  invoke=lambda: {"status": "completed", "targets": ["box"]}, earlier_requests=[(earlier, .1)])
    refused = []

    def preflight():
        refused.append("preflight")
        raise ValueError("website_gemini_runtime_not_configured")
    assert website_gemini_receipts.retained_gemini_call(preflight=preflight, **common) == old
    with pytest.raises(ValueError, match="runtime_not_configured"):
        website_gemini_receipts.retained_gemini_call(preflight=preflight, accept_earlier=lambda b, r: False, **common)
    assert refused == ["preflight"] and calls == []


def test_retained_blocked_analysis_from_an_older_instruction_is_bought_again(retained):  # noqa: F811
    old = {**retained.plan, "prompt_template_sha256": BEFORE_CHOICE_RULE}
    old.pop("prompt_revision")
    _save(retained.capture / "pipeline/clean_plate", "removal_plan.json", old)
    assert load_retained_uncertain_analysis(capture_root=retained.capture, task_context=retained.context,
                                           source_video=retained.source) is None
    _save(retained.capture / "pipeline/clean_plate", "removal_plan.json", retained.plan)
    assert load_retained_uncertain_analysis(capture_root=retained.capture, task_context=retained.context,
                                           source_video=retained.source) == retained.plan


def test_observed_room_job_without_selection_binding_keeps_the_analysis(retained):  # noqa: F811
    (retained.root / "selection_binding.json").unlink()
    assert resolve_retained_target_selection(plan=retained.plan, capture_root=retained.capture,
        task_context=retained.context, source_video=retained.source) == retained.plan


def test_uncertain_object_to_move_becomes_a_recorded_guess():
    plan = _plan([_box(), SHELF], ["task_target_requires_clarification:cardboard_box_candidate"])
    resolved = analysis.resolve_uncertain_targets(plan)
    assert resolved["status"] == "completed" and resolved["blockers"] == []
    box, shelf = resolved["targets"]
    assert (box["task_effect"], box["disposition"], box["rebuild_intent"]) == (
        "manipulated", "remove", "rebuild_and_compose")
    assert shelf == SHELF and resolved["movable_removal_count"] == 1
    guess = box["uncertain_guess"]
    assert guess["code"] == analysis.TASK_OBJECT_GUESSED and guess["physical_evidence"] is False
    assert guess["analysis_before_guess"]["task_effect"] == "uncertain"
    assert guess["analysis_before_guess"]["clarification_question"] == "Which box?"
    assert resolved["uncertain_target_guesses"] == [guess]
    assert analysis.validate_removal_plan(resolved) == []
    assert analysis.resolve_uncertain_targets(plan) == resolved  # The same answer gives the same guess.


def test_the_best_supported_uncertain_candidate_is_the_guess():
    plan = _plan([_box("box_a", 0.5), _box("box_b", 0.6, evidence=((3.0, [0.1, 0.1, 0.1, 0.1]),)), SHELF],
                 ["task_target_requires_clarification:box_a", "task_target_requires_clarification:box_b"])
    resolved = analysis.resolve_uncertain_targets(plan)
    effects = {row["target_id"]: row["task_effect"] for row in resolved["targets"]}
    assert effects == {"box_a": "unrelated", "box_b": "manipulated", "storage_rack_shelf": "static_contact"}
    assert resolved["targets"][1]["uncertain_guess"]["alternatives"] == ["box_a"]


def test_an_uncertain_target_in_another_role_keeps_that_role_flagged():
    vase = {**_box("flower_vase", 0.85), "target_role": "obstacle", "semantic_label": "glass vase"}
    table = {**SHELF, "target_id": "dining_table", "target_role": "support"}
    resolved = analysis.resolve_uncertain_targets(_plan([vase, table], ["task_target_requires_clarification:flower_vase"]))
    kept = resolved["targets"][0]
    assert (kept["task_effect"], kept["disposition"]) == ("static_obstacle", "keep")
    assert kept["uncertain_guess"]["code"] == analysis.TARGET_ROLE_KEPT
    assert resolved["status"] == "completed" and resolved["movable_removal_count"] == 0


def test_nothing_the_task_is_about_is_left_as_the_not_observed_stop():
    plan = _plan([], ["task_relevant_target_not_observed"])
    assert analysis.resolve_uncertain_targets(plan) == plan
    other = _plan([_box()], ["website_task_not_named"])
    assert analysis.resolve_uncertain_targets(other) == other


def test_guess_codes_reach_the_scene_estimate_flags(tmp_path):
    from blueprint_pipeline.atlas_scene_estimates import CODES, analysis_guess_flags
    resolved = analysis.resolve_uncertain_targets(
        _plan([_box(), SHELF], ["task_target_requires_clarification:cardboard_box_candidate"]))
    _save(tmp_path / "pipeline/clean_plate", "removal_plan.json", resolved)
    assert analysis_guess_flags(tmp_path) == {"scene_estimate:task_object_guessed"} <= CODES
    from blueprint_pipeline.beta_policy_screening_publication_flags import flags_valid
    assert flags_valid(sorted(CODES))


def test_not_observed_ends_as_a_hold_that_asks_for_a_new_video(tmp_path):
    from types import SimpleNamespace
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_preparation_outcome import read_preparation_outcome
    from blueprint_pipeline.website_recapture_required import NOT_OBSERVED_CODE, retains_recapture_required
    from tests.test_qualification_coverage_edges import _descriptor
    from tests.test_website_task_sentence_wait import NAMED
    descriptor = _descriptor(metadata={"site_task_context": NAMED})
    context = {**NAMED, "scene_id": descriptor.scene_id, "capture_id": descriptor.capture_id}
    descriptor = _descriptor(metadata={"site_task_context": context})
    capture = tmp_path / "scenes" / descriptor.scene_id / "captures" / descriptor.capture_id
    (capture / "pipeline").mkdir(parents=True)
    result = run_website_capture_preparation(
        descriptor=descriptor, capture_root=capture, pipeline_dir=capture / "pipeline", source_video=None,
        derived_scene_generation_allowed=True, gates=[],
        prepare_media=lambda **_: {"status": "blocked", "blockers": ["task_relevant_target_not_observed"]},
        check_privacy=lambda **_: SimpleNamespace(passed=True),
        make_gate=lambda _name, passed, _detail: SimpleNamespace(passed=passed))
    handoff = result["website_scene_preparation"]
    assert result["status"] == "completed" and handoff["blockers"] == [NOT_OBSERVED_CODE]
    assert retains_recapture_required(handoff, context)
    def outcome():
        return read_preparation_outcome(capture_root=capture, task_context_digest=context["context_digest"],
                                        state="awaiting_inputs", recapture=True)
    # Until the website lists the new reason, the nearest reason it already words travels instead.
    import os
    from blueprint_pipeline.website_preparation_outcome import NEW_REASONS_ENV
    os.environ[NEW_REASONS_ENV] = "false"
    assert outcome()["reason"] == "task_evidence_incomplete"
    os.environ[NEW_REASONS_ENV] = "true"
    try:
        assert outcome() == {"category": "insufficient_evidence", "reason": NOT_OBSERVED_CODE,
                             "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    finally:
        os.environ.pop(NEW_REASONS_ENV, None)


def test_a_guessed_object_is_asked_about_even_when_its_anchor_decides(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_masks as masks
    from blueprint_pipeline import website_task_target_choice as choice
    from tests.test_website_task_target_choice import GEOMETRY, _track
    # Only the left instance lies under the analysis's box: the anchor decides, yet the object was a guess.
    target = {"target_id": "target-1", "semantic_label": "cardboard box", "task_effect": "manipulated",
              "disposition": "remove", "target_role": "task_object",
              "spatial_evidence": [{"timestamp_seconds": 4.0, "box_xywh_normalized": [0.0, 0.4, 0.5, 0.3]}],
              "uncertain_guess": {"code": analysis.TASK_OBJECT_GUESSED}}
    registry = [{"source_frame_id": "f-4", "decoded_pts_seconds": 4.0, "width": 10, "height": 10}]
    tracks = [_track("sam-left", 1), _track("sam-right", 6)]
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    monkeypatch.setattr(masks, "run_meta_sam31", lambda **kw: {"tracks": tracks})

    def run(row, **extra):
        return masks.run_website_task_masks(plan={"targets": [row], "task_context_sha256": "task"},
            source_geometry={"digest": "source", "geometry_available": False, "binding": {"source_video_digest": "video"},
                             "frames": [{"frame_id": "f-4", "timestamp_seconds": 4.0, "width": 10, "height": 10}]},
            source_video=tmp_path / "source.mov", output_root=tmp_path / "masks", **extra)
    plain = []
    run({key: value for key, value in target.items() if key != "uncertain_guess"}, target_guesses=plain)
    assert plain == []  # An object the analysis was sure of is not asked about.
    guesses = []
    result = run(target, target_guesses=guesses)
    assert result["targets"][0]["track"]["track_id"] == "sam-left"
    assert [row["guessed_track_id"] for row in guesses] == ["sam-left"]
    question = choice.build_question(target=target, candidates=guesses[0]["candidates"], frames=guesses[0]["frames"],
        guessed_track_id="sam-left", scene_id="s", capture_id="c", source_geometry=GEOMETRY)
    assert question is not None and len(question["candidates"]) == 2
    other = next(row["candidate_id"] for row in question["candidates"] if row["candidate_id"] != question["guessed_candidate_id"])
    corrected = run(target, target_guesses=[], target_choices={"target-1": {
        "target_id": "target-1", "candidate_set_digest": question["candidate_set_digest"], "candidate_id": other,
        "question_digest": question["question_digest"]}})
    assert corrected["targets"][0]["track"]["track_id"] == "sam-right"
    assert corrected["targets"][0]["source_track"]["selection_basis"] == "site_choice"


@pytest.mark.parametrize("shape", ["uncertain_supports_only", "object_without_box"])
def test_with_nothing_to_move_the_guess_is_the_ask_for_a_new_video(shape):
    """Review 2026-11-12: a kept support must not hide that nothing to move was found (capture-4b60431e's first
    analysis), and an uncertain object with no box is not in the video."""
    table = {**SHELF, "target_id": "dining_table", "task_effect": "uncertain", "clarification_question": "q?",
             "decision_reason": "d"}
    table.pop("placement_relation")
    if shape == "uncertain_supports_only":
        targets = [table, {**table, "target_id": "kitchen_island"}]
        blockers = [analysis.NOT_OBSERVED, "task_target_requires_clarification:dining_table",
                    "task_target_requires_clarification:kitchen_island"]
    else:
        box = _box()
        box["spatial_evidence"] = [{"timestamp_seconds": 1.0, "box_xywh_normalized": None}]
        targets, blockers = [box, SHELF], ["task_target_requires_clarification:cardboard_box_candidate"]
    resolved = analysis.resolve_uncertain_targets(_plan(targets, blockers))
    assert resolved["status"] == "blocked" and resolved["blockers"] == [analysis.NOT_OBSERVED]
    assert not any(row["task_effect"] == "manipulated" for row in resolved["targets"])


def test_a_published_scene_keeps_the_analysis_it_was_built_on(retained):  # noqa: F811
    from blueprint_pipeline.website_atlas_task_selection import load_published_analysis
    built = {**retained.plan, "status": "completed", "blockers": [], "prompt_template_sha256": "ad87"}
    built.pop("prompt_revision")
    _save(retained.capture / "pipeline/clean_plate", "removal_plan.json", built)
    assert load_published_analysis(capture_root=retained.capture, task_context=retained.context,
                                   source_video=retained.source) is None  # Not staged: may be redone.
    _save(retained.capture / "pipeline/website_scene_preparation", "handoff.json",
          {"status": "intake_ready", "website_intake_outbox": {"state": "forward_pending"}})
    assert load_published_analysis(capture_root=retained.capture, task_context=retained.context,
                                   source_video=retained.source) == built


def test_a_replaced_plan_is_kept_beside_the_new_one(tmp_path):
    from blueprint_pipeline.clean_plate_stage import _keep_superseded_plan
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    old = _plan([_box(), SHELF], ["task_target_requires_clarification:cardboard_box_candidate"], BEFORE_CHOICE_RULE)
    _save(tmp_path, "removal_plan.json", old)
    new = analysis.resolve_uncertain_targets(_plan([_box(), SHELF], ["task_target_requires_clarification:cardboard_box_candidate"]))
    _keep_superseded_plan(tmp_path, new)
    kept = json.loads((tmp_path / "superseded" / (canonical_digest(old)[7:] + ".json")).read_text())
    assert kept["plan"] == old and kept["previous_revision"] == analysis.OLDER_PROMPT_REVISION
    assert kept["replacing_revision"] == analysis.PROMPT_REVISION
    assert kept["reason"] == "analysis_bought_again_under_current_instructions"


@pytest.mark.parametrize("rerun", ["registration_reevaluation", "spent_retry"])
def test_a_rerun_at_no_cost_keeps_the_analysis_the_job_has(retained, rerun):  # noqa: F811
    """Review 2026-11-12: a registration hold looked at again, or a spent retry on a new release, bought a new
    analysis under newer instructions (its retained one was an older revision). It keeps the one it has."""
    from blueprint_pipeline.website_atlas_task_selection import load_published_analysis, load_rerun_analysis
    from blueprint_pipeline.website_recapture_required import FIELD, record
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    built = {**retained.plan, "status": "completed", "blockers": [], "prompt_template_sha256": "ad87"}
    built.pop("prompt_revision")
    assert not analysis.binds_current_job(built)  # an older revision: an ordinary rerun buys it again
    _save(retained.capture / "pipeline/clean_plate", "removal_plan.json", built)
    kwargs = dict(capture_root=retained.capture, task_context=retained.context, source_video=retained.source)
    assert load_rerun_analysis(**kwargs) is None  # no $0 re-run under way
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
               "blockers": ["website_registration_ambiguous"],
               FIELD: record("website_registration_ambiguous", task_context_digest=retained.context["context_digest"])}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    _save(retained.capture / "pipeline/website_scene_preparation", "handoff.json", handoff)
    if rerun == "registration_reevaluation":
        _save(retained.capture / "pipeline/website_scene_preparation", "held_handoff_replay.json",
              {"fact": {"kind": "release", "recapture_code": "website_registration_ambiguous"},
               "handoff_digest": handoff["digest"]})
    else:
        _save(retained.capture, "pipeline_job_ledger.json", {"preparation_retry": {"kind": "retries_spent"}})
    assert load_published_analysis(**kwargs) is None and load_rerun_analysis(**kwargs) == built



def test_a_boxed_task_object_of_another_class_is_kept_as_the_object_the_job_acts_on():
    """Review 2026-11-12: an uncertain built-in drawer classed ``fixed_clutter``, with its box, was sent to
    "film again". It is in the video: it is the assembly the job acts on, its class recorded as it was."""
    box = {**_box(), "target_class": "fixed_clutter"}
    resolved = analysis.resolve_uncertain_targets(_plan([box, SHELF], ["task_target_requires_clarification:cardboard_box_candidate"]))
    assert resolved["status"] == "completed" and resolved["blockers"] == []
    chosen = next(row for row in resolved["targets"] if row["target_id"] == box["target_id"])
    assert chosen["task_effect"] == "manipulated" and chosen["target_class"] == "movable_object"
    assert chosen["uncertain_guess"]["analysis_before_guess"]["target_class"] == "fixed_clutter"
    assert not analysis.validate_removal_plan(resolved)
