"""A preserved capture whose generated gap views were not admitted ships as filmed (gap D, owner 2026-11-16).

Two parts. A paid review answer is kept on its receipt whatever it says: a content check that fails blocks the
review with a code, so the same review is never bought twice; on the preserved route an empty frame-id list is the
pass, as its prompt says. And a completion stopped by its review (or by a review nothing more may be bought for)
goes on without the generated part: the refined splat and mesh run on the captured scene, the scene source hands it
on flagged ``scene_completion_not_admitted``, and the room package's README says the gaps were not filled in. Any
other stop (money, privacy, a changed binding, a source recovery) is unchanged. Shapes are internal test job 1's
(``itest-8a15…``, 16 Nov): three generated wide views, a retained generation and two ``submitting`` review receipts.
"""
from __future__ import annotations

import json
import shutil
import sys
from hashlib import sha256
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from blueprint_pipeline import atlas_depth_splat as splats
from blueprint_pipeline import atlas_mesh_registration as mesh_registration
from blueprint_pipeline import atlas_result_bundle as bundles
from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_atlas_scene_source as scene_source
from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_gemini_receipts as receipts
from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.test_website_atlas_scene_source import digest, job_fixture, write
from tests.test_website_image_completion import _grant

FLAG = pipeline.SCENE_COMPLETION_NOT_ADMITTED
REVIEW_REQUIRED = "atlas_completion_visual_review_required"
EXHAUSTED = receipts.EXHAUSTED
NO_CONTEXT_DEPTH = "atlas_completion_no_context_depth"
# Its survey and aiming, estimated from counts before they start, cannot fit in a step (Austin, 25 Nov).
WORK_EXCEEDS_STEP = "atlas_completion_work_exceeds_step"
# Job 1's generated views and two of its capture-context frames (ids only).
VIEWS = [f"completion-completion-decoded-0000000{n}-wide" for n in (15, 21, 35)]
CONTEXT = ["decoded-000000015", "decoded-000000018"]
KEEP = [{"target_id": name, "semantic_label": name, "task_effect": "support", "disposition": "keep"}
        for name in ("dining_table", "floor")]


# --- the review: a paid answer is never thrown away ------------------------------------------------------------

class Review:
    """Gemini as the review sees it: each call answers ``answer`` (a dict, or raw text)."""

    def __init__(self, monkeypatch, answer):
        self.answer, self.calls, self.reservations = answer, 0, []
        review = self

        class Client:
            def __init__(self, **_):
                self.models = self

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def generate_content(self, **_):
                review.calls += 1
                text = review.answer if isinstance(review.answer, str) else json.dumps(review.answer)
                return NS(candidates=[NS(finish_reason="STOP")], text=text)

        types = NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=lambda **kw: NS(**kw),
                   HttpOptions=lambda **kw: NS(**kw), HttpRetryOptions=lambda **kw: NS(**kw))
        monkeypatch.setitem(sys.modules, "google.genai", NS(Client=Client, types=types))
        monkeypatch.setattr(completion, "_api_key", lambda: ("fixture-key", "fixture"))

        def reserve(**kwargs):
            self.reservations.append(kwargs)
            admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted",
                         "resource_class": kwargs["resource_class"], "blockers": [],
                         "allocation_binding_digest": kwargs["binding_digest"]}
            return admission, _grant(admission)
        monkeypatch.setattr(receipts, "reserve_website_preparation_spend", reserve)
        monkeypatch.setattr(receipts, "require_current_website_source", lambda _context: None)


def _review_args(tmp_path, *, preserved=True):
    task = {"schema_version": "website_site_task_context.v1", "request_id": "itest", "scene_id": "site-itest",
            "capture_id": "walkthrough-itest", "confirmed": True, "confirmed_at": "2026-11-16",
            "description": "Go from the doorway to the table."}
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    frames, originals = [], []
    for name in [*VIEWS, *CONTEXT]:
        path = tmp_path / "images" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 9), "grey").save(path)
        row = {"frame_id": name, "image_path": str(path), "image_digest": _sha256_file(path)}
        if name in VIEWS:
            frames.append({**row, "novel_view": True, "generated_pixels_present": True})
        originals.append(row)
    plan = {"targets": KEEP if preserved else [{**KEEP[0], "task_effect": "manipulated", "disposition": "remove"}],
            "task_context_sha256": sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    if preserved:
        plan["context_basis"] = "preserved_capture"
    return dict(frames=frames, original_frames=originals, plan=plan, output_root=tmp_path / "review",
                task_context=task, source_video_digest="sha256:" + "5" * 64)


def _answer(**fields):
    return {"consistent_background": True, "task_objects_removed": True, "unrelated_objects_preserved": True,
            "remaining_task_object_frame_ids": [], "unrelated_object_loss_frame_ids": [], **fields}


@pytest.mark.parametrize("removed", [True, False])
def test_a_preserved_review_passes_on_an_empty_list_whatever_it_says_of_removal(tmp_path, monkeypatch, removed):
    review = Review(monkeypatch, _answer(task_objects_removed=removed))
    assert completion.verify_completed_background(**_review_args(tmp_path))["status"] == "passed"
    assert review.calls == 1


@pytest.mark.parametrize("answer,code", [
    (_answer(remaining_task_object_frame_ids=["decoded-000000015"]), completion.REVIEW_FRAME_IDS_INVALID),
    (_answer(remaining_task_object_frame_ids=[VIEWS[0], VIEWS[0]]), completion.REVIEW_FRAME_IDS_INVALID),
    (_answer(unrelated_object_loss_frame_ids=["decoded-000000018"]), completion.REVIEW_UNRELATED_FRAME_IDS_INVALID),
    ("not json", completion.REVIEW_NOT_JSON),
])
def test_an_unusable_preserved_answer_is_kept_blocked_and_never_bought_again(tmp_path, monkeypatch, answer, code):
    review = Review(monkeypatch, answer)
    args = _review_args(tmp_path)
    first = completion.verify_completed_background(**args)
    assert (first["status"], first["blocker"]) == ("blocked", code)
    stored = [json.loads(path.read_text()) for path in (args["output_root"] / "gemini_reviews").glob("*.json")]
    assert [row["status"] for row in stored] == ["completed"] and stored[0]["result"] == first
    assert completion.verify_completed_background(**args) == first
    assert review.calls == 1 and len(review.reservations) == 1


def test_a_preserved_answer_naming_a_view_still_blocks(tmp_path, monkeypatch):
    Review(monkeypatch, _answer(remaining_task_object_frame_ids=[VIEWS[1]], task_objects_removed=False))
    result = completion.verify_completed_background(**_review_args(tmp_path))
    assert result["status"] == "blocked" and "blocker" not in result


@pytest.mark.parametrize("answer", [_answer(task_objects_removed=False), "not json"])
def test_the_editor_route_still_stops_on_an_unusable_answer_but_never_buys_it_again(tmp_path, monkeypatch, answer):
    review = Review(monkeypatch, answer)
    args = _review_args(tmp_path, preserved=False)
    for _ in range(2):
        with pytest.raises(ValueError, match="website_image_completion_review_(frame_ids_invalid|not_json)"):
            completion.verify_completed_background(**args)
    assert review.calls == 1 and len(review.reservations) == 1
    assert [json.loads(p.read_text())["status"] for p in (args["output_root"] / "gemini_reviews").glob("*.json")] \
        == ["completed"]


@pytest.mark.parametrize("old_release_raises", [2, 3])
def test_job_one_after_the_release_buys_at_most_the_one_review_its_cap_allows(tmp_path, monkeypatch,
                                                                                old_release_raises):
    """Its receipts as the host has them (dispatch and resend-1 ``submitting``, one counted unanswered), or one
    more if the old release spent resend-2 first. The new release buys at most that one; the answer is kept."""
    review = Review(monkeypatch, _answer(remaining_task_object_frame_ids=["decoded-000000015"],
                                         task_objects_removed=False))
    args = _review_args(tmp_path)
    real = completion._verify_completed_background

    def old_release(**_):
        review.calls += 1
        raise ValueError(completion.REVIEW_FRAME_IDS_INVALID)
    monkeypatch.setattr(completion, "_verify_completed_background", old_release)
    for _ in range(old_release_raises):
        with pytest.raises(ValueError, match="frame_ids_invalid"):
            completion.verify_completed_background(**args)
    names = sorted(path.name.split(".", 1)[-1] for path in (args["output_root"] / "gemini_reviews").glob("*.json"))
    if old_release_raises == 2:
        assert names == ["json", "resend-1.json", "unanswered-1.json"]  # the host's files on 16 Nov
    monkeypatch.setattr(completion, "_verify_completed_background", real)
    bought = len(review.reservations)
    if old_release_raises == 3:
        with pytest.raises(ValueError, match=EXHAUSTED):
            completion.verify_completed_background(**args)
        assert len(review.reservations) == bought == 3 and review.calls == 3
        return
    result = completion.verify_completed_background(**args)
    assert (result["status"], result["blocker"]) == ("blocked", completion.REVIEW_FRAME_IDS_INVALID)
    assert len(review.reservations) == bought + 1 == 3 and review.calls == 3
    assert completion.verify_completed_background(**args) == result and len(review.reservations) == 3


# --- the controller: the floor ----------------------------------------------------------------------------------

def _scene(tmp_path, monkeypatch, outcomes, *, review=None):
    """``_preserved_scene`` on job 1's shape: each call takes the next completion outcome."""
    pose_root = tmp_path / "pose"
    source_root = pose_root / "full-view-source-v1"
    source_root.mkdir(parents=True, exist_ok=True)
    pose = {"digest": "source-pose"}
    write(source_root / "atlas_posed_rgbd_result.json", pose)
    write(tmp_path / "atlas_scene_review.json", {})
    write(source_root / "scene_measurements.json", {})
    baseline = {"digest": "baseline", "splat_kind": "depth_fused_draft"}
    write(tmp_path / "splat/atlas_depth_splat_manifest.json", baseline)
    calls, exports = [], []
    monkeypatch.setattr(pipeline, "_source_pose", lambda **_: {"status": "completed"})
    monkeypatch.setattr(source_inputs, "prepare_source_pose", lambda _: (source_root, {}))
    monkeypatch.setattr(splats, "load_posed_rgbd", lambda _: (pose, ["source-depth"]))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *args: {"remaining_cost_usd": 3})

    def complete(**_):
        calls.append("completion")
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation", complete)
    monkeypatch.setattr(splats, "splats2mesh_attempt", lambda root, output: ({"request": {}, "uploads": {}}, output))

    def mesh(**options):
        calls.append("mesh")
        write(options["output_root"] / "result.json", {"outputs": [{"role": "mesh", "path": "scene.glb"}]})
        return {"status": "completed"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", mesh)
    monkeypatch.setattr(mesh_registration, "register_atlas_mesh", lambda **_: None)
    monkeypatch.setattr(bundles, "export_pose_bundle",
                        lambda **options: exports.append(options) or {"zip_path": "scene.zip"})

    def run():
        return pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={},
            review=review or {"subject_origin": "no_manipulated_subject",
                              "automation": {"objects": [], "removal_instruction": ""}, "uncaptured_regions": []},
            policy={"mesh_export_cost_usd": .5, "rgbd_view_cost_usd": .1, "rate_card_reference": "test",
                    "reviewed_by": "test", "reviewed_on": "test"},
            task_context={"context_digest": "task"}, source_digest="capture", configuration={},
            check=lambda _: None, experimental=False, max_polls=1, model="test",
            source_survey_path=tmp_path / "source-survey.json")
    return run, calls, exports, baseline


def _stopped(reason, **extra):
    return {"status": "needs_input", "reason": reason, "stage": "preserved_scene_completion",
            "failure_class": "missing_evidence", "generated_roots": ["native"],
            "remaining_unknowns": ["floor behind the camera"], **extra}


@pytest.mark.parametrize("outcome,reason", [
    (_stopped(REVIEW_REQUIRED), REVIEW_REQUIRED),
    (_stopped("atlas_generated_view_registration_unresolved"), "atlas_generated_view_registration_unresolved"),
    (ValueError(EXHAUSTED), EXHAUSTED),
    # Job itest-7c42 (16 Nov, 21:53Z): the completion planner, bought once ($0.258, settled), abstained; its
    # retained answer is re-read at $0 on every retry. Its siblings: the planner's answer is unusable.
    *[(ValueError(code), code) for code in ("atlas_completion_planner_abstained",
                                            "atlas_completion_camera_selection_invalid",
                                            "atlas_completion_task_gap_plan_missing",
                                            "atlas_coverage_environment_evidence_invalid")],
    # No view has any valid depth to place a completion camera on (a view without it is only left out).
    (ValueError(NO_CONTEXT_DEPTH), NO_CONTEXT_DEPTH),
    (ValueError(WORK_EXCEEDS_STEP), WORK_EXCEEDS_STEP),
])
def test_a_completion_its_review_did_not_admit_goes_on_without_the_generated_part(tmp_path, monkeypatch,
                                                                                outcome, reason):
    run, calls, exports, baseline = _scene(tmp_path, monkeypatch, [outcome, outcome])
    result = run()
    assert calls == ["completion", "mesh"] and result["mesh_available"]
    assert result["splat_manifest_digest"] == baseline["digest"] and not result["generated_geometry_admitted"]
    assert "generated_roots" not in exports[0] and exports[0]["splat_root"] == tmp_path / "splat"
    assert reason not in result["blockers"]
    scene = result["scene_completion"]
    assert (scene["status"], scene["generated_geometry_rejected"], scene["not_admitted_reason"]) == (
        "completed", FLAG, reason)
    state = json.loads((tmp_path / "preserved_completion/controller_state.json").read_text())
    assert state["not_admitted_reason"] == reason and state["generated_roots"] == []
    assert run() == result  # a second step: the same result


@pytest.mark.parametrize("outcome", [
    _stopped("atlas_completion_request_budget_insufficient:required=6,remaining=2"),
    _stopped("something_nobody_classified"),
    {"status": "source_recovery_required", "reason": "atlas_recorded_source_views_must_be_recovered_before_generation"},
    {"status": "source_recovery_required", "reason": REVIEW_REQUIRED},  # a floor code, but not a stopped completion
])
def test_any_other_completion_stop_is_unchanged(tmp_path, monkeypatch, outcome):
    run, calls, _exports, _ = _scene(tmp_path, monkeypatch, [outcome])
    result = run()
    assert calls == ["completion"] and not result["mesh_available"]
    assert result["scene_completion"]["status"] == outcome["status"]
    assert "not_admitted_reason" not in result["scene_completion"]


@pytest.mark.parametrize("code", [
    "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted",
    "atlas_preserved_completion_review_not_bound", "website_image_completion_review_source_changed",
    # The completion's other raises stay stops: no canonical planner, the release check, a money check.
    "atlas_completion_canonical_planner_required", "atlas_completion_release_not_admitted:host_checkout_dirty",
    "atlas_completion_review_release_not_admitted:host_checkout_dirty",
    "atlas_completion_request_budget_insufficient:required=6,remaining=2",
])
def test_a_raised_money_or_integrity_stop_is_raised_as_before(tmp_path, monkeypatch, code):
    run, calls, _exports, _ = _scene(tmp_path, monkeypatch, [ValueError(code)])
    with pytest.raises(ValueError, match=code.split(":")[0]):
        run()
    assert calls == ["completion"]


def test_a_review_with_people_in_its_frames_never_reaches_completion_or_the_floor(tmp_path, monkeypatch):
    run, calls, _exports, _ = _scene(tmp_path, monkeypatch, [_stopped(REVIEW_REQUIRED)], review={
        "subject_origin": "no_manipulated_subject", "automation": {"objects": [], "removal_instruction": ""},
        "uncaptured_regions": [], "transient_frame_ids": ["decoded-000000018"]})
    with pytest.raises(ValueError, match="atlas_scene_requires_preserved_capture"):
        run()
    assert calls == []


def test_a_completion_that_later_passes_replaces_the_unfilled_scene_once(tmp_path, monkeypatch):
    completed_root = tmp_path / "preserved_completion/splat-bound"
    admitted = {"status": "completed", "generated_geometry_admitted": True, "splat_root": str(completed_root),
                "splat_manifest": {"digest": "completed-export", "splat_kind": "depth_fused_draft"},
                "completion": {"generated_roots": ["native"]}, "generated_roots": [str(tmp_path / "native")],
                "analysis_paths": [], "admission": {"digest": "admission"},
                "admission_path": str(tmp_path / "admission.json")}
    run, calls, _exports, _ = _scene(tmp_path, monkeypatch, [_stopped(REVIEW_REQUIRED), admitted, admitted])
    floor, passed, again = run(), run(), run()
    assert floor["scene_completion"]["generated_geometry_rejected"] == FLAG
    assert passed["generated_geometry_admitted"] and "not_admitted_reason" not in passed["scene_completion"]
    assert passed["splat_manifest_digest"] == "completed-export" and again == passed


# --- the scene source, the handoff, the package and the preview -------------------------------------------------

def _floored_result(tmp_path, *, reason=REVIEW_REQUIRED, admitted=False, flag=FLAG):
    root = job_fixture(tmp_path, same_pose=True)
    result = json.loads((root / "result.json").read_text())
    background = result.pop("automation")["background"]
    result.update(subject_origin="no_manipulated_subject", captured_scene_preserved=True,
        object_removal_performed=False, image_inpainting_performed=False,
        generated_completion_performed=True, generated_geometry_admitted=admitted,
        mesh_available=True, splat_manifest_digest=background["splat_manifest_digest"],
        scene_completion={"status": "completed", "reason": None, "failure_class": None,
                          "generated_geometry_admitted": False, "remaining_unknowns": [],
                          "generated_geometry_rejected": flag, "not_admitted_reason": reason})
    shutil.move(root / "automation/background_splat", root / "splat")
    shutil.move(root / "automation/background_mesh", root / "mesh")
    write(root / "result.json", digest(result))
    write(root / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})


@pytest.mark.parametrize("tamper", [None, "reason", "admitted", "flag"])
def test_the_scene_source_hands_on_the_unfilled_scene_flagged(tmp_path, tamper):
    _floored_result(tmp_path, reason="something_else" if tamper == "reason" else REVIEW_REQUIRED,
                    admitted=tamper == "admitted", flag="something_else" if tamper == "flag" else FLAG)
    outcome = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    if tamper is None:
        assert outcome["status"] == "ready" and outcome["flags"] == [FLAG]
        assert outcome["scene_content"] == "captured_scene_preserved"
        # Named on the base scene too, for a package that builds an object on it (``base_scene_notes``).
        assert outcome["base_scene"]["scene_completion"] == {"generated_geometry_rejected": FLAG,
                                                             "not_admitted_reason": REVIEW_REQUIRED}
        assert scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep") == outcome
    else:
        assert outcome == {"status": "needs_input", "reason": "atlas_preserved_completion_not_admitted"}


def test_the_room_package_carries_the_flag_and_says_the_gaps_were_not_filled(tmp_path, monkeypatch):
    from tests.test_website_scene_handoff_navigation import ATLAS, _published, _record, _said, room_job
    from blueprint_pipeline import website_atlas_scene_source as source

    job, _context, plain = room_job(tmp_path, monkeypatch)
    assert sim.NOT_FILLED_IN not in _said(_published(job)[1]["README.txt"])
    atlas = source.atlas_scene_source()
    monkeypatch.setattr(ATLAS, lambda **_: {**atlas, "flags": [FLAG]})
    result = job.rerun()
    assert _record(result)["flags"] == [FLAG] and result != plain
    assert job.rerun() == result  # the handoff reading the result twice
    said = _said(_published(job)[1]["README.txt"])
    assert said.count(sim.NOT_FILLED_IN) == 1
    assert sim.NOT_FILLED_IN == "Parts of the room the video did not show were not filled in."


def test_the_preview_job_ends_ready_on_the_unfilled_scene(monkeypatch, tmp_path):
    from blueprint_pipeline import website_atlas_preview as preview
    from tests.test_website_atlas_preview import World, _completed, _enqueue, _job

    monkeypatch.delenv(preview.ALLOWLIST_ENV, raising=False)
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    site = NS(storage=tmp_path / "storage", capture=capture, pipeline=capture / "pipeline",
              job=capture / "pipeline/atlas" / preview.JOB_FILE)
    _enqueue(site)

    def floored(kwargs):
        return {**_completed(kwargs), "scene_completion": {"status": "completed", "generated_geometry_admitted": False,
                "generated_geometry_rejected": FLAG, "not_admitted_reason": EXHAUSTED}}
    world = World(monkeypatch, site, [floored])
    assert world.advance() == "ready" and _job(site)["archive"] is not None


def test_an_unreadable_consistency_diagnosis_is_kept_and_never_bought_again(tmp_path, monkeypatch):
    review = Review(monkeypatch, "not json")
    args = _review_args(tmp_path, preserved=False)
    failed = {"status": "blocked", "request_digest": "sha256:prior", "review": {
        "consistent_background": False, "task_objects_removed": True, "unrelated_objects_preserved": True,
        "remaining_task_object_frame_ids": []}}
    call = dict(frames=args["frames"], original_frames=args["original_frames"], plan=args["plan"],
                failed_review=failed, output_root=args["output_root"], task_context=args["task_context"])
    for _ in range(2):
        with pytest.raises(ValueError, match="website_background_consistency_diagnosis_invalid"):
            completion.diagnose_inconsistent_background(**call)
    assert review.calls == 1 and len(review.reservations) == 1


def test_three_flags_on_one_job_read_in_a_pinned_order(tmp_path, monkeypatch):
    """Not found (gap A), the planner's removals not applied (gap C1) and the gaps not filled in (gap D) on one job:
    the record and the README in the same order: what the job is about, what is in the scan, what was not filled."""
    from blueprint_pipeline import website_atlas_scene_source as source
    from blueprint_pipeline import website_scene_handoff as handoff
    from blueprint_pipeline.atlas_automation_plan import JOB_OBJECT_RULE
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    from tests.test_website_scene_handoff_navigation import ATLAS, _published, _record, _said, room_job

    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    rule = {"code": JOB_OBJECT_RULE, "planner_remove_count": 2, "removed_count": 0}
    job, _context, _result = room_job(tmp_path, monkeypatch, family="wiping", description="Wipe the table.",
                                      all_targets=True, not_found=True, job_object_rule=rule)
    atlas = source.atlas_scene_source()
    monkeypatch.setattr(ATLAS, lambda **_: {**atlas, "flags": [FLAG]})
    result = job.rerun()
    assert _record(result)["flags"][-3:] == [handoff.NOT_FOUND, JOB_OBJECT_RULE, FLAG]
    assert job.rerun() == result
    said = _said(_published(job)[1]["README.txt"])
    assert said.endswith(" ".join([sim.NOT_FOUND, sim.NOTHING_TAKEN_OUT, sim.NOT_FILLED_IN]))
    assert said.count(sim.NOT_FILLED_IN) == 1 and said.count("taken out of the room scan") == 1



def test_the_planner_codes_the_floor_takes_are_the_ones_the_completion_raises():
    """Pinned to the raise sites, so a renamed code can never fall out of the floor unnoticed."""
    import inspect

    from blueprint_pipeline import atlas_scene_completion
    source = inspect.getsource(atlas_scene_completion.complete_scene)
    planner = pipeline.COMPLETION_FLOOR_ERRORS - {EXHAUSTED, NO_CONTEXT_DEPTH, WORK_EXCEEDS_STEP}
    assert len(planner) == 4
    assert all(f'raise ValueError("{code}")' in source or f"raise ValueError('{code}')" in source for code in planner)
    # The camera pool's own refusal, raised only when no view has any valid depth.
    pool = inspect.getsource(atlas_scene_completion.camera_candidates)
    assert NO_CONTEXT_DEPTH in pipeline.COMPLETION_FLOOR_ERRORS and f'raise ValueError("{NO_CONTEXT_DEPTH}")' in pool
    # The work bound's, raised from counts before the survey and before placement scoring.
    from blueprint_pipeline import atlas_completion_work
    assert atlas_completion_work.WORK_EXCEEDS_STEP == WORK_EXCEEDS_STEP in pipeline.COMPLETION_FLOOR_ERRORS
    assert 'raise ValueError(WORK_EXCEEDS_STEP)' in inspect.getsource(atlas_completion_work.CompletionWork._admit)
    assert 'work.admit_survey()' in source and "'placement_bound': work.admit_placements" in source
    for stop in ("atlas_completion_canonical_planner_required", "atlas_completion_release_not_admitted"):
        assert stop in source and stop not in pipeline.COMPLETION_FLOOR_ERRORS


def test_job_two_after_the_planner_abstained_hands_on_the_room_flagged(tmp_path):
    _floored_result(tmp_path, reason="atlas_completion_planner_abstained")
    outcome = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert outcome["status"] == "ready" and outcome["flags"] == [FLAG]


# --- the captured path: a job that moves a filmed object (move-the-lamp-2, 24 Nov) --------------------------------
#
# Its editor took the lamp out of every selected view and its review passed; the completion's generated views only
# fill areas no frame saw (``atlas_scene_completion``: "ONE reviewed object-absent environment", known geometry
# immutable). Their review refused one view (invented clutter by the shower), and the whole job stopped. Now the room
# the editor cleaned goes on without them, flagged the same way, and the completion is never run again for it.

LAMP_REVIEW = "Invented clutter and tools appear on the floor near the shower enclosure."


def _captured(tmp_path, monkeypatch, outcomes, *, refined=False):
    """``run_automation`` on the captured path, the editor's cleaned views posed again; each completion call takes
    the next outcome. Returns the run, the calls made and what each paid or derived step received."""
    from blueprint_pipeline import (
        atlas_background_automation,
        atlas_native_splat,
        atlas_result_bundle,
        atlas_scene_completion,
        atlas_scene_coverage,
        atlas_scene_measurements,
        atlas_scene_recovery,
        atlas_scene_task_regions,
    )
    from blueprint_pipeline import atlas_scene_automation as worker
    from tests.test_atlas_scene_completion import frame

    cleaned = frame("decoded-000000030")
    pose = {"digest": "sha256:fallback-pose", "binding": {"source_video_digest": "source"}}
    calls, seen = [], {"splats": [], "exports": [], "meshes": []}
    monkeypatch.setattr(atlas_scene_measurements, "load_views", lambda *_: (pose, [cleaned]))
    monkeypatch.setattr(atlas_scene_measurements, "estimate_sizes", lambda **_: {})
    monkeypatch.setattr(atlas_background_automation, "run_background", lambda **_: {
        "status": "fallback_prepared", "frames": [], "review": {"status": "passed"},
        "stage_roots": [str(tmp_path / "editor")]})
    monkeypatch.setattr(worker, "_fallback_pose", lambda *_: ((pose, [cleaned], tmp_path), {"status": "completed"}))
    monkeypatch.setattr(splats, "load_posed_rgbd", lambda _root: (pose, [cleaned]))
    write(tmp_path / "atlas_pose_inputs.json", {})

    def complete(**options):
        calls.append("completion")
        options["root"].mkdir(parents=True, exist_ok=True)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome.get("digest"):
            write(options["root"] / "completion.json", outcome)
        return None, outcome
    monkeypatch.setattr(atlas_scene_completion, "complete_scene", complete)
    for module, name in ((atlas_scene_task_regions, "plan_task_regions"),
                         (atlas_scene_recovery, "recover_reviewed_geometry"),
                         (atlas_scene_coverage, "refine_scene_coverage")):
        monkeypatch.setattr(module, name, lambda name=name, **_: calls.append(name))

    def fuse(**options):
        seen["splats"].append(options)
        return {"digest": "sha256:" + "b" * 64, "generated_result_digests": options["generated_result_digests"],
                "frames": [{"frame_id": f["frame_id"]} for f in options["frames"]]}
    monkeypatch.setattr(splats, "write_depth_splat", fuse)

    def native(**options):
        calls.append("refined_splat")
        return {"status": "completed", "splat_root": str(options["output_root"]),
                "splat_manifest": {"digest": "sha256:" + "c" * 64}, "refined_splat": {"form": "minimal"}}
    monkeypatch.setattr(atlas_native_splat, "run_native_splat", native)
    monkeypatch.setattr(splats, "splats2mesh_attempt", lambda root, output: (
        seen["meshes"].append(output) or {"request": {}, "uploads": {}}, output))

    def task(**options):
        calls.append(options["task"])
        return {"status": "completed", "outputs": [{"role": "mesh", "path": "scene.glb"}]}
    monkeypatch.setattr(mesh_registration, "register_atlas_mesh", lambda **_: None)
    monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle",
                        lambda **options: seen["exports"].append(options) or {"zip_path": "scene.zip"})
    policy = {"rgbd_view_cost_usd": .05, "mesh_export_cost_usd": .5, "rate_card_reference": "test",
              "reviewed_by": "test", "reviewed_on": "test", **({"refined_splat_cost_usd": 5} if refined else {})}

    def run():
        return worker.run_automation(pose_root=None, inputs={"frames": []}, review={"automation": {
            "objects": [{"target_id": "lamp", "label": "Lamp to move from bedside table to glass wall shelf",
                         "segmentation_prompt": "pleated shade lamp", "disposition": "remove", "regions": [
                             {"frame_id": "decoded-000000030", "box_xywh_normalized": [.823, .281, .165, .337]}]}],
            "scale_references": [], "removal_instruction": "Remove the lamp", "mask_margin_fraction": .01}},
            output_root=tmp_path / "automation", task_context={}, configuration={"policy": policy},
            task_runner=task, pose_runner=None, planner_runner=object(), release_blockers=lambda: ("commit", []),
            source_survey_path=tmp_path / "source.json")
    monkeypatch.setattr(worker, "_unposed_source_views", lambda _inputs: (pose, [cleaned]))
    return run, calls, seen


def _refused(reason=REVIEW_REQUIRED):
    report = {"status": "needs_input", "reason": reason, "generation_result_digest": "sha256:" + "a" * 64,
              "generated_roots": ["scene_completion-v14/generated/0"], "review": {"status": "blocked", "review": {
                  "consistent_background": False, "reason": LAMP_REVIEW}},
              "remaining_unknowns": ["Glass shelf depth and rear mounting contact remain unresolved."]}
    return digest(report)


@pytest.mark.parametrize("outcome,reason", [
    (_refused(), REVIEW_REQUIRED),
    (ValueError(EXHAUSTED), EXHAUSTED),
    *[(ValueError(code), code) for code in ("atlas_completion_planner_abstained",
                                            "atlas_completion_camera_selection_invalid",
                                            "atlas_completion_task_gap_plan_missing",
                                            "atlas_coverage_environment_evidence_invalid")],
    (ValueError(NO_CONTEXT_DEPTH), NO_CONTEXT_DEPTH),
    (ValueError(WORK_EXCEEDS_STEP), WORK_EXCEEDS_STEP),
])
@pytest.mark.parametrize("refined", [False, True])
def test_a_captured_job_whose_gap_views_were_refused_goes_on_with_the_room_its_editor_cleaned(
        tmp_path, monkeypatch, outcome, reason, refined):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline.atlas_scene_completion import COMPLETION_REVISION

    run, calls, seen = _captured(tmp_path, monkeypatch, [outcome], refined=refined)
    result = run()
    # The cleaned room is fused from the editor's views alone, then the refined splat and the mesh are bought on it;
    # nothing more is bought for the refused views (no task regions, recovery or coverage repair).
    assert result["status"] == "completed" and result["background"]["status"] == "completed"
    assert calls == ["completion", *(["refined_splat"] if refined else []), "splats2Mesh"]
    [fused] = seen["splats"]
    assert [f["frame_id"] for f in fused["frames"]] == ["decoded-000000030"]
    assert "sha256:" + "a" * 64 not in fused["generated_result_digests"]
    assert fused["output_root"].name == f"background_splat-{COMPLETION_REVISION}-not-admitted"
    assert seen["meshes"][0].name.startswith(f"background_mesh-{COMPLETION_REVISION}-not-admitted")
    floor = result["background"]["scene_completion"]
    assert (floor["status"], floor["generated_geometry_admitted"], floor["generated_geometry_rejected"],
            floor["not_admitted_reason"]) == ("completed", False, FLAG, reason)
    assert "completion" not in result["background"]
    root = tmp_path / "automation"
    record = json.loads((root / worker.COMPLETION_FLOOR_RECORD).read_text())
    assert record["digest"] == floor["floor_digest"] and record["not_admitted_reason"] == reason
    # The export holds the editor's roots and the floor's evidence, never a refused view.
    export = seen["exports"][0]
    assert export["generated_roots"] == [tmp_path / "editor"]
    assert root / worker.COMPLETION_FLOOR_RECORD in export["analysis_paths"]
    assert all("generated" not in str(path.relative_to(tmp_path)) for path in export["analysis_paths"])
    if isinstance(outcome, dict):
        assert root / f"scene_completion-{COMPLETION_REVISION}/completion.json" in export["analysis_paths"]
        assert floor["rejected_generation_result_digest"] == "sha256:" + "a" * 64
        assert floor["remaining_unknowns"] == outcome["remaining_unknowns"]
    # The next step reads the sealed floor: the completion's planner, generation and review are not run again.
    before = len(calls)
    assert run() == result
    assert "completion" not in calls[before:]


@pytest.mark.parametrize("outcome,code", [
    (digest({"status": "needs_input", "reason": "atlas_completion_request_budget_insufficient:required=6,remaining=2"}),
     "atlas_completion_request_budget_insufficient:required=6,remaining=2"),
    (digest({"status": "needs_input", "reason": "something_nobody_classified"}), "something_nobody_classified"),
    (ValueError("atlas_completion_canonical_planner_required"), "atlas_completion_canonical_planner_required"),
    (ValueError("website_image_completion_review_source_changed"), "website_image_completion_review_source_changed"),
])
def test_any_other_captured_completion_stop_still_stops_the_background(tmp_path, monkeypatch, outcome, code):
    from blueprint_pipeline import atlas_scene_automation as worker

    run, calls, seen = _captured(tmp_path, monkeypatch, [outcome, outcome])
    result = run()
    assert result["status"] == "needs_input" and result["background"]["reason"] == code
    assert calls == ["completion"] and not seen["splats"] and not seen["exports"]
    assert not (tmp_path / "automation" / worker.COMPLETION_FLOOR_RECORD).exists()


@pytest.mark.parametrize("code", ["account_monthly_spend_limit_reached",
                                  "atlas_completion_release_not_admitted:host_checkout_dirty",
                                  "atlas_completion_review_release_not_admitted:host_checkout_dirty"])
def test_a_money_or_release_refusal_inside_the_completion_is_raised_as_before(tmp_path, monkeypatch, code):
    from blueprint_pipeline import atlas_scene_automation as worker

    run, calls, _ = _captured(tmp_path, monkeypatch, [ValueError(code)])
    with pytest.raises(ValueError, match=code.split(":")[0]):
        run()
    assert calls == ["completion"] and not (tmp_path / "automation" / worker.COMPLETION_FLOOR_RECORD).exists()


def test_a_floor_sealed_for_another_completion_decides_nothing_for_this_one(tmp_path, monkeypatch):
    """Sealed for a refused report that is no longer the one on file (or another pose): the completion decides again."""
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline.atlas_scene_completion import COMPLETION_REVISION

    first = _refused()
    run, calls, _ = _captured(tmp_path, monkeypatch, [first, first])
    run()
    report = tmp_path / f"automation/scene_completion-{COMPLETION_REVISION}/completion.json"
    write(report, digest({**json.loads(report.read_text()), "reason": "atlas_completion_visual_review_required",
                          "remaining_unknowns": []}))
    run()
    assert calls.count("completion") == 2
    record = tmp_path / "automation" / worker.COMPLETION_FLOOR_RECORD
    write(record, {**json.loads(record.read_text()), "not_admitted_reason": "tampered"})
    result = run()
    assert result["background"]["reason"] == "atlas_completion_floor_record_invalid"


def test_the_job_result_and_the_preview_say_the_cleaned_room_went_on_without_the_views(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker

    run, _calls, _ = _captured(tmp_path, monkeypatch, [_refused()])
    automation = run()
    result = pipeline.finalize_background_result(output_root=tmp_path, source_digest="sha256:" + "5" * 64,
        task_context={"context_digest": "task"}, automation=automation, source_pose={"status": "completed"},
        model="test")
    assert result["scene_completion"] == automation["background"]["scene_completion"]
    assert result["scene_completion"]["generated_geometry_rejected"] == FLAG
    assert result["blockers"] == ["metric_scene_registration_unproven"] and result["splat_available"]
    # The preview step reads ``scene_completion``: a completed one never parks the job.
    assert result["scene_completion"]["status"] == "completed"
    record = json.loads((tmp_path / "automation" / worker.COMPLETION_FLOOR_RECORD).read_text())
    assert record["completion_relative_path"].startswith("scene_completion-")


def _captured_floor_result(tmp_path, *, summary=None, provenance=()):
    """The scene source's captured fixture, its background carrying a floor summary."""
    root = job_fixture(tmp_path)
    manifest_path = root / "automation/background_splat/atlas_depth_splat_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest = digest({**{k: v for k, v in manifest.items() if k != "digest"},
                       "generated_result_digests": list(provenance)})
    write(manifest_path, manifest)
    registration_path = root / "automation/background_mesh/registered/atlas_mesh_registration.json"
    registration = json.loads(registration_path.read_text())
    write(registration_path, digest({**{k: v for k, v in registration.items() if k != "digest"},
                                     "splat_manifest_digest": manifest["digest"]}))
    result = json.loads((root / "result.json").read_text())
    background = result["automation"]["background"]
    background["splat_manifest_digest"] = manifest["digest"]
    background["scene_completion"] = summary if summary is not None else {
        "status": "completed", "generated_geometry_admitted": False, "generated_geometry_rejected": FLAG,
        "not_admitted_reason": REVIEW_REQUIRED, "remaining_unknowns": [], "floor_digest": "sha256:" + "f" * 64,
        "rejected_generation_result_digest": "sha256:" + "a" * 64}
    result = digest({k: v for k, v in result.items() if k != "digest"})
    write(root / "result.json", result)
    write(root / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})


def test_the_scene_source_hands_on_the_cleaned_room_flagged_and_names_it_on_the_base_scene(tmp_path):
    _captured_floor_result(tmp_path, provenance=["sha256:editor"])
    outcome = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert outcome["status"] == "ready" and outcome["flags"] == [FLAG]
    assert "scene_content" not in outcome  # a captured object's job: the cleaned room, not the scene as filmed
    assert outcome["base_scene"]["scene_completion"] == {"generated_geometry_rejected": FLAG,
                                                         "not_admitted_reason": REVIEW_REQUIRED}
    from blueprint_pipeline import website_half_preparation as half
    assert half.base_scene_notes(outcome["base_scene"]) == {"not_filled_in": True}
    assert scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep") == outcome


@pytest.mark.parametrize("tamper", ["reason", "admitted", "flag", "status", "refused_view_in_the_splat"])
def test_a_captured_floor_that_does_not_hold_is_refused(tmp_path, tamper):
    summary = {"status": "completed", "generated_geometry_admitted": False, "generated_geometry_rejected": FLAG,
               "not_admitted_reason": REVIEW_REQUIRED, "rejected_generation_result_digest": "sha256:" + "a" * 64}
    summary.update({"reason": {"not_admitted_reason": "something_else"}, "admitted": {"generated_geometry_admitted": True},
                    "flag": {"generated_geometry_rejected": "something_else"}, "status": {"status": "needs_input"},
                    }.get(tamper, {}))
    _captured_floor_result(tmp_path, summary=summary,
                           provenance=["sha256:" + "a" * 64] if tamper == "refused_view_in_the_splat" else [])
    with pytest.raises(ValueError, match="website_atlas_scene_background_changed"):
        scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")


def test_a_captured_room_with_no_floor_carries_no_flag_and_no_marker(tmp_path):
    job_fixture(tmp_path)
    outcome = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert outcome["status"] == "ready" and "flags" not in outcome and "scene_completion" not in outcome["base_scene"]
