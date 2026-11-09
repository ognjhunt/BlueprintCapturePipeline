"""A new job's one windowed re-pose before its refined splat (owner, 2026-11-09 15:27:24Z: "yes").

Synthetic poses only: cameras along a line, a time per view. The first pose's own motion analysis
(``select_splat_frames``) decides the window; the re-pose is bought once, under the term the job
recorded before its first purchase, and only an accepted re-pose replaces the first pose for the splat.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import atlas_splat_rebuild_window as windows
from blueprint_pipeline import website_atlas_windowed_repose as reposes
from blueprint_pipeline.atlas_refined_splat_forms import SPLAT_POSE_INPUT_CAPACITY
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_splat_rebuild_window import _review, _views, _window


def _policy(**changes):
    return {"maximum_review_cost_usd": 4, "pose_input_cost_usd": .01, "rgbd_view_cost_usd": .05,
            "mesh_export_cost_usd": .5, "refined_splat_cost_usd": 5,
            "rate_card_reference": "https://example.test/rates", "reviewed_by": "test",
            "reviewed_on": datetime.now(timezone.utc).date().isoformat(), **changes}


def test_a_new_job_records_its_repose_term_once_before_its_first_purchase(tmp_path):
    policy = _policy()
    term = reposes.decide_term(tmp_path, policy=policy, purchased=False, available_usd=10, minimum_usd=.03)
    assert term["enabled"] is True and term["owner_decision"]["decided_at"] == "2026-11-09T15:27:24Z"
    # The extra pose at the first pose's per-view price and input bound: in the maximum, not the expectation.
    assert term["per_view_cost_usd"] == .01 and term["max_input_images"] == SPLAT_POSE_INPUT_CAPACITY
    assert term["maximum_cost_usd"] == pytest.approx(SPLAT_POSE_INPUT_CAPACITY * .01)
    assert term["expected_cost_usd"] == 0 and "first_pose_refinement_not_accepted" in term["conditional_on"]
    assert term["price_review"]["reviewed_by"] == "test"
    assert reposes.held_back_usd(term) == term["maximum_cost_usd"]
    # Read back, never re-decided: not by a later price, a purchase or a smaller allowance.
    assert reposes.decide_term(tmp_path, policy=_policy(pose_input_cost_usd=.02), purchased=True,
                               available_usd=0, minimum_usd=1) == term


def test_an_in_flight_job_gets_no_term_and_is_never_re_priced(tmp_path):
    assert reposes.decide_term(tmp_path, policy=_policy(), purchased=True, available_usd=10, minimum_usd=.03) is None
    assert not (tmp_path / reposes.TERM_FILE).exists() and reposes.held_back_usd(None) == 0
    # A route that buys no refined splat gets none either.
    policy = {key: value for key, value in _policy().items() if key != "refined_splat_cost_usd"}
    assert reposes.decide_term(tmp_path, policy=policy, purchased=False, available_usd=10, minimum_usd=.03) is None


def test_an_allowance_that_cannot_hold_the_repose_back_records_it_disabled_and_holds_nothing(tmp_path):
    term = reposes.decide_term(tmp_path, policy=_policy(), purchased=False, available_usd=1.0, minimum_usd=.03)
    assert term["enabled"] is False and term["reason"] == "allowance_cannot_hold_back_the_repose"
    assert term["maximum_cost_usd"] == 0 and reposes.held_back_usd(term) == 0


@pytest.mark.parametrize("bought", [False, True])
def test_the_selection_budget_holds_the_repose_maximum_back_only_for_a_new_job(tmp_path, monkeypatch, bought):
    from blueprint_pipeline import atlas_scene_planner, paid_resource_allocator as allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline, website_atlas_pose as pose_api
    video = tmp_path / "source.mov"
    video.write_bytes(b"retained-source")
    root = tmp_path / "job"
    (root / "survey").mkdir(parents=True)
    if bought:
        (root / "scene_planner").mkdir()  # an in-flight job: its first review was already bought
    (root / "survey/atlas_video_survey.json").write_text(json.dumps(
        {"source_video_digest": pipeline._sha256_file(video), "frames": []}))
    saved = {"authority": {"provider_retention": "project_library_retention_approved"}, "policy": _policy(),
             "remaining_cost_usd": 20.0}
    if bought:
        (root / "configuration.json").write_text(json.dumps(saved))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **kw: deepcopy(saved))
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("url", "test-non-secret"))
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan",
                        lambda **_: {"proposal": {"status": "proposed"}, "reference_sources": []})

    class Reviewed(Exception):
        pass

    def review(*_, pose_budget, **__):
        raise Reviewed(pose_budget)
    monkeypatch.setattr(atlas_scene_planner, "proposal_review", review)
    with pytest.raises(Reviewed) as reviewed:
        pipeline.run_atlas_pipeline(source_video=video, output_root=root,
                                    task_context={"capture_id": "c", "context_digest": "ctx"})
    before = 20.0 - 2 * 4 - pipeline.final_output_cost(_policy())
    if bought:
        assert reviewed.value.args[0] == pytest.approx(before) and not (root / reposes.TERM_FILE).exists()
    else:
        term = json.loads((root / reposes.TERM_FILE).read_text())
        assert reviewed.value.args[0] == pytest.approx(before - term["maximum_cost_usd"])


FRAMES = _views([("still", 20), ("jump", 0), ("move", 30)])


def _job(tmp_path, *, frames=FRAMES, accepted=False, term=True, remaining=20.0):
    """A new job whose first pose came back: its records, its term and its first pose's selection profile."""
    job = tmp_path / "pipeline/atlas"
    first_root = job / "pose"
    source = first_root / "task-coverage-1/full-view-source-v1"
    source.mkdir(parents=True)
    pose = {"operation": {"response": {"poseRefinementAccepted": accepted}},
            "operation_receipt": {"input_usage": {"frames": [{"frame_id": row["frame_id"]} for row in frames]}}}
    pose["digest"] = canonical_digest(pose, digest_field="digest")
    (source / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    (first_root / "atlas_view_selection.json").write_text(json.dumps(
        {"frame_policy": {"splat_frame_target": 96, "relative_sharpness_fraction": .6}}))
    (job / "survey").mkdir()
    (job / "survey/atlas_video_survey.json").write_text(json.dumps({"frames": [
        {"frame_id": row["frame_id"], "timestamp_seconds": row["timestamp_seconds"]} for row in frames]}))
    review = _review(frames)
    (job / "selection_binding.json").write_text(json.dumps(
        {"survey_relative_path": "survey/atlas_video_survey.json", "review": review}))
    if term:
        reposes.decide_term(job, policy=_policy(), purchased=False, available_usd=10, minimum_usd=.03)
    return job, first_root, source, review, pose


class _Stage:
    """The paid and local stages the re-pose goes through, recorded; the pose is World Labs' answer."""

    def __init__(self, monkeypatch, job, *, frames=FRAMES, accepted=True, count=40, remaining=20.0,
                 scale="estimated_from_named_assumptions"):
        from blueprint_pipeline import atlas_source_pose_inputs, atlas_video_inputs
        from blueprint_pipeline import website_atlas_pipeline as pipeline
        self.selected, self.posed, self.job = [], [], job
        self.retry_source = job / reposes.ROOT / "task-coverage-2/full-view-source-v1"
        monkeypatch.setattr(windows, "selection_on_pose", lambda *_: (
            dict(_window(frames, route=["v000", "v030"])[0], pose_result_digest="sha256:" + "e" * 64),
            {row["frame_id"]: row["timestamp_seconds"] for row in frames}))
        monkeypatch.setattr(atlas_video_inputs, "prepare_video_inputs",
                            lambda **kw: self.selected.append(kw) or {"frames": [{"frame_id": "x"}] * count})

        def prepare(root):
            assert root == job / reposes.ROOT
            self.retry_source.mkdir(parents=True, exist_ok=True)
            return self.retry_source, {"frames": [{"frame_id": "x"}] * count}
        monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", prepare)
        monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: {"remaining_cost_usd": remaining})

        def source_pose(**kw):
            self.posed.append(kw)
            path = self.retry_source / "atlas_posed_rgbd_result.json"
            if not path.is_file():
                value = {"operation": {"response": {"poseRefinementAccepted": accepted}}}
                value["digest"] = canonical_digest(value, digest_field="digest")
                path.write_text(json.dumps(value))
            return {"status": "completed", "result_digest": json.loads(path.read_text())["digest"],
                    "scale_status": scale}
        monkeypatch.setattr(pipeline, "_source_pose", source_pose)


def _repose(job, first_root, source, review, **changes):
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    options = dict(output_root=job, first_pose_root=first_root, source_root=source, review=review,
                   source_pose={"status": "completed", "scale_status": "estimated_from_named_assumptions"},
                   policy=_policy(), task_context={"capture_id": "c", "context_digest": "ctx"},
                   source_digest="sha256:" + "1" * 64, configuration={}, check=lambda *_a, **_k: None,
                   experimental=False, max_polls=1)
    return pipeline._windowed_repose(**{**options, **changes})


def test_a_jump_reposes_the_window_once_and_an_accepted_repose_carries_the_splat(tmp_path, monkeypatch):
    job, first_root, source, review, first = _job(tmp_path)
    stage = _Stage(monkeypatch, job)
    value = _repose(job, first_root, source, review)
    assert value["adopted"] is True and value["source_root"] == stage.retry_source
    assert value["pose_root"] == job / reposes.ROOT
    window = json.loads((job / reposes.ROOT / "atlas_pose_window.json").read_text())
    assert window["first_pose_result_digest"] == first["digest"] and window["status"] == "windowed"
    # The selection draws only from the window, with the profile the first pose recorded.
    assert value["review"] == window["review"] and stage.selected[0]["review"] == window["review"]
    assert stage.selected[0]["splat_frame_target"] == 96 and stage.selected[0]["relative_sharpness_fraction"] == .6
    # One pose, with no scale review bought for it.
    assert len(stage.posed) == 1 and stage.posed[0]["allow_scale_review"] is False
    outcome = json.loads((job / reposes.ROOT / reposes.OUTCOME_FILE).read_text())
    assert outcome["status"] == "adopted" and outcome["input_count"] == 40
    assert value["disclosure"]["status"] == "adopted"
    assert value["disclosure"]["left_out_required_frame_ids"] == ["v000", "v018"]
    # A later tick reads it all back: the same outcome, nothing recorded again.
    again = _repose(job, first_root, source, review)
    assert again["adopted"] is True and again["source_root"] == stage.retry_source
    assert json.loads((job / reposes.ROOT / reposes.OUTCOME_FILE).read_text()) == outcome


def test_a_repose_world_labs_again_does_not_refine_leaves_the_first_pose_and_never_a_third(tmp_path, monkeypatch):
    job, first_root, source, review, _ = _job(tmp_path)
    stage = _Stage(monkeypatch, job, accepted=False)
    value = _repose(job, first_root, source, review)
    assert value["adopted"] is False and value["disclosure"]["status"] == "not_accepted"
    assert len(stage.posed) == 1
    # Later ticks go on with the first pose: no third pose is ever bought.
    assert _repose(job, first_root, source, review)["adopted"] is False and len(stage.posed) == 1


@pytest.mark.parametrize("shape", ["clean", "tripod"])
def test_a_shot_with_no_window_buys_nothing_extra(tmp_path, monkeypatch, shape):
    frames = _views([("move", 50)]) if shape == "clean" else _views([("still", 50)])
    job, first_root, source, review, _ = _job(tmp_path, frames=frames)
    stage = _Stage(monkeypatch, job, frames=frames)
    value = _repose(job, first_root, source, review)
    assert value["adopted"] is False and value["disclosure"]["status"] == "no_window"
    assert stage.selected == [] and stage.posed == []


def test_an_allowance_short_of_the_repose_stops_it_by_name_and_buys_nothing(tmp_path, monkeypatch):
    job, first_root, source, review, _ = _job(tmp_path)
    # 40 views at $0.01 plus the $5 splat and the $0.50 mesh still to buy: $5.90 needed.
    stage = _Stage(monkeypatch, job, remaining=5.5)
    value = _repose(job, first_root, source, review)
    outcome = json.loads((job / reposes.ROOT / reposes.OUTCOME_FILE).read_text())
    assert value["adopted"] is False and outcome["status"] == "allowance_insufficient"
    assert outcome["needed_usd"] == pytest.approx(5.9) and stage.posed == []


def test_a_repose_over_the_terms_input_bound_buys_nothing(tmp_path, monkeypatch):
    job, first_root, source, review, _ = _job(tmp_path)
    stage = _Stage(monkeypatch, job, count=len(FRAMES) + 1)
    value = _repose(job, first_root, source, review)
    assert value["disclosure"]["status"] == "inputs_over_term_bound" and stage.posed == []


@pytest.mark.parametrize("case", ["in_flight", "refined", "disabled"])
def test_no_term_a_refined_first_pose_or_a_disabled_term_tries_nothing(tmp_path, monkeypatch, case):
    job, first_root, source, review, _ = _job(tmp_path, term=case != "in_flight", accepted=case == "refined")
    if case == "disabled":
        (job / reposes.TERM_FILE).unlink()
        reposes.decide_term(job, policy=_policy(), purchased=False, available_usd=.5, minimum_usd=.03)
    stage = _Stage(monkeypatch, job)
    assert _repose(job, first_root, source, review) is None
    assert not (job / reposes.ROOT).exists() and stage.posed == []


def test_an_unscaled_repose_is_not_used_when_the_first_pose_had_a_scale(tmp_path, monkeypatch):
    job, first_root, source, review, _ = _job(tmp_path)
    _Stage(monkeypatch, job, scale="unscaled_provider_world_units")
    value = _repose(job, first_root, source, review)
    assert value["adopted"] is False and value["disclosure"]["status"] == "scale_unresolved"


@pytest.mark.parametrize("adopted", [True, False])
def test_the_scene_buys_its_splat_from_the_adopted_repose_and_names_it(tmp_path, monkeypatch, adopted):
    from blueprint_pipeline import atlas_depth_splat as splats, atlas_mesh_registration as registration
    from blueprint_pipeline import atlas_result_bundle as bundles, atlas_source_pose_inputs as source_inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pipeline as pipeline
    first_source = tmp_path / "pose/full-view-source-v1"
    retry_source = tmp_path / "pose-windowed/full-view-source-v1"
    for root, digest in ((first_source, "first"), (retry_source, "retry")):
        root.mkdir(parents=True)
        (root / "atlas_posed_rgbd_result.json").write_text(json.dumps(
            {"digest": digest, "operation": {"response": {"poseRefinementAccepted": digest == "retry"}}}))
        (root / "scene_measurements.json").write_text("{}")
    (tmp_path / "atlas_scene_review.json").write_text("{}")
    first_review = {"subject_origin": "no_manipulated_subject", "uncaptured_regions": ["Behind."],
                    "automation": {"objects": [], "removal_instruction": ""}}
    derived = {**first_review, "uncaptured_regions": ["Behind.", "Start area: frames v000 left out."]}
    monkeypatch.setattr(pipeline, "_source_pose", lambda **_: {"status": "completed"})
    monkeypatch.setattr(source_inputs, "prepare_source_pose", lambda _: (first_source, {}))
    disclosure = {"status": "adopted" if adopted else "not_accepted", "outcome_digest": "o"}
    monkeypatch.setattr(pipeline, "_windowed_repose", lambda **kw: {
        "adopted": adopted, "root": "pose-windowed", "records": ("atlas_pose_window.json",),
        "disclosure": disclosure,
        **({"pose_root": tmp_path / "pose-windowed", "inputs": {"frames": []}, "review": derived,
            "source_pose": {"status": "completed", "measurements_relative_path": "scene_measurements.json"},
            "source_root": retry_source} if adopted else {})})
    (tmp_path / "pose-windowed/atlas_pose_window.json").write_text("{}")
    loaded = []
    monkeypatch.setattr(splats, "load_posed_rgbd", lambda root: loaded.append(root) or (
        json.loads((root / "atlas_posed_rgbd_result.json").read_text()), [{"frame_id": "a"}]))
    monkeypatch.setattr(splats, "write_depth_splat", lambda **_: {"digest": "draft", "splat_kind": "depth_fused_draft"})
    seen = []
    refined = {"digest": "sha256:" + "6" * 64, "splat_kind": "atlas_refined"}
    monkeypatch.setattr(pipeline, "_preserved_appearance", lambda **kw: seen.append(kw) or {
        "status": "completed", "basis": "atlas_refined_splat", "splat_root": str(tmp_path / "refined"),
        "splat_manifest": refined})
    monkeypatch.setattr(splats, "splats2mesh_attempt", lambda root, output: ({"request": {}, "uploads": {}}, output))

    def mesh(**kw):
        kw["output_root"].mkdir(parents=True)
        (kw["output_root"] / "result.json").write_text(json.dumps({"outputs": [{"role": "mesh", "path": "m.glb"}]}))
        return {"status": "completed"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", mesh)
    monkeypatch.setattr(registration, "register_atlas_mesh", lambda **_: None)
    bundled = []
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **kw: bundled.append(kw) or {"zip_path": "s.zip"})
    result = pipeline._preserved_scene(pose_root=tmp_path / "pose", output_root=tmp_path, inputs={},
        review=first_review, policy={"mesh_export_cost_usd": .5, "refined_splat_cost_usd": 5,
                                     "pose_input_cost_usd": .01, "rate_card_reference": "r", "reviewed_by": "r",
                                     "reviewed_on": "r"},
        task_context={"context_digest": "t"}, source_digest="s", configuration={}, check=lambda *_: None,
        experimental=False, max_polls=1, model="m")
    chosen = retry_source if adopted else first_source
    assert loaded == [chosen] and seen[0]["source_root"] == chosen
    assert seen[0]["review"] == (derived if adopted else first_review)
    assert result["uncaptured_regions"] == (derived if adopted else first_review)["uncaptured_regions"]
    assert result["windowed_repose"] == disclosure and bundled[0]["pose_root"] == chosen
    assert tmp_path / "pose-windowed/atlas_pose_window.json" in bundled[0]["analysis_paths"]
