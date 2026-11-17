"""A room step that stopped for the site rests no more: it is run again on the refusal schedule and buys nothing twice.

Owner direction 2026-11-17: a stuck job retries by itself. These stops showed ``needs_input`` with no retry time,
so the job was ended and its provider assets deleted at once. They still show as before; the step is now run
again on ``job_retry_policy``'s refusal schedule and, once that is spent, rests exactly as it did.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
from tests.test_website_atlas_preview import World, _enqueue, _job


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    return SimpleNamespace(storage=tmp_path / "storage", capture=capture, pipeline=capture / "pipeline",
                           job=capture / "pipeline/atlas" / preview.JOB_FILE)


HELD = [
    ("atlas_scene_planner_abstained:people move through every frame", "atlas_scene_planner_abstained"),
    ("atlas_scene_planner_abstained:final", "atlas_scene_planner_abstained"),
    ("website_task_object_not_observed", "website_task_object_not_observed"),
    ("website_capture_frames_unusable", "website_capture_frames_unusable"),
    ("atlas_scene_subject_origin_uncertain", "atlas_scene_subject_origin_uncertain"),
    ("atlas_scene_subject_origin_changed", "atlas_scene_subject_origin_changed"),
    ("atlas_synthetic_subject_requires_preserved_capture", "atlas_synthetic_subject_requires_preserved_capture"),
    ("atlas_scene_requires_preserved_capture", "atlas_scene_requires_preserved_capture"),
    ("atlas_insufficient_overlapping_distinct_views", "atlas_insufficient_overlapping_distinct_views"),
    ("atlas_coverage_anchor_quality_insufficient", "atlas_coverage_anchor_quality_insufficient"),
]


@pytest.mark.parametrize("raised,reason", HELD)
def test_a_stop_for_the_site_shows_at_once_retries_on_the_schedule_then_rests_as_before(monkeypatch, site,
                                                                                        raised, reason):
    _enqueue(site)
    world = World(monkeypatch, site, [ValueError(raised)] * (len(REFUSAL_RETRY_DELAYS_SECONDS) + 1))
    (site.pipeline / "atlas/pose/atlas_pose").mkdir(parents=True)
    (site.pipeline / "atlas/pose/atlas_pose/submission.json").write_text('{"created_assets": ["upload1"]}')
    assert world.advance() == "needs_input"
    for attempt, delay in enumerate(REFUSAL_RETRY_DELAYS_SECONDS, start=1):
        job = _job(site)
        assert job["reason"] == reason and job["retry_at_epoch"] == world.clock + delay
        assert job["refusal_retries"] == attempt and job["provider_cleanup"] is None and world.deleted == []
        world.clock += delay - 1
        world.advance()
        assert len(world.runs) == attempt  # not before its time
        world.clock += 1
        assert world.advance() == "needs_input" and len(world.runs) == attempt + 1
    job = _job(site)
    # The schedule is spent: the job rests as it did before this rule, and only now is it cleaned up.
    assert (job["state"], job["reason"], job["retry_at_epoch"]) == ("needs_input", reason, None)
    assert job["provider_cleanup"]["state"] == "complete" and world.deleted == ["upload1"]
    # The website saw one state the whole time: the same ask it got before.
    assert [(row["state"], row["reason"]) for row in world.pushed] == [("needs_input", reason)]
    world.clock += preview.MAXIMUM_JOB_SECONDS
    assert world.advance() == "needs_input" and len(world.runs) == len(REFUSAL_RETRY_DELAYS_SECONDS) + 1


def test_a_retry_that_gets_further_goes_on_and_one_that_succeeds_is_ready(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [ValueError("atlas_insufficient_overlapping_distinct_views"),
                                      {"status": "processing", "stage": "pose"}])
    assert world.advance() == "needs_input"
    world.clock = _job(site)["retry_at_epoch"]
    assert world.advance() == "processing"
    assert [(row["state"], row["reason"]) for row in world.pushed] == [
        ("needs_input", "atlas_insufficient_overlapping_distinct_views"), ("processing", None)]


def test_the_lifetime_allowance_is_shared_with_every_other_refusal(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [ValueError("atlas_scene_subject_origin_uncertain")])
    job = {**_job(site), "refusal_retries_total": preview.MAXIMUM_REFUSAL_RETRIES}
    site.job.write_text(json.dumps(job))
    assert world.advance() == "needs_input"
    assert _job(site)["retry_at_epoch"] is None  # nothing left: it rests at once, as before


# --- A retry buys nothing twice --------------------------------------------------------------------


def _pipeline_world(tmp_path, monkeypatch, *, selection_refusal, second=None):
    """The real scene review (``atlas_scene_planner.review_survey``) behind a stop raised after it.

    The model, the reservation and the settlement are fakes that count; everything between the
    preview controller and the review's receipts under ``scene_planner/`` is the production code.
    """
    from blueprint_pipeline import atlas_scene_planner as planner
    from blueprint_pipeline import atlas_video_extension as extension
    from blueprint_pipeline import atlas_video_inputs as inputs
    from blueprint_pipeline import paid_resource_admission as admission
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline import website_atlas_pose as pose_api
    from blueprint_pipeline import website_task_context as control
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_atlas_scene_planner import fixture

    video = tmp_path / "source.mov"
    video.write_bytes(b"capture")
    _path, survey, proposal, policy = fixture(tmp_path)
    survey["source_video_digest"] = pipeline._sha256_file(video)
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    policy = {**policy, "reviewed_on": datetime.now(timezone.utc).date().isoformat(), "rgbd_view_cost_usd": .05}
    config = {"authority": {"provider_retention": "account_does_not_retain"}, "remaining_cost_usd": 10,
              "policy": policy}
    counts = {"invoked": 0, "reserved": 0, "settled": 0, "selected": 0}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: json.loads(json.dumps(config)))
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "test-non-secret"))
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend",
                        lambda **_: (counts.__setitem__("reserved", counts["reserved"] + 1) or {}, object()))

    def settle(**kwargs):
        assert kwargs["operation"] == "preparation-settlement"
        counts["settled"] += 1
        return {"status": "settled"}
    monkeypatch.setattr(control, "website_webapp_request", settle)

    def survey_video(**kwargs):
        (kwargs["output_root"] / "source").mkdir(parents=True, exist_ok=True)
        (kwargs["output_root"] / "source/decoded_observation_index.json").write_text("{}")
        (kwargs["output_root"] / "atlas_video_survey.json").write_text(json.dumps(survey))
    monkeypatch.setattr(inputs, "survey_video", survey_video)

    dense = {**survey, "frames": [*survey["frames"], {**survey["frames"][0], "frame_id": "3",
                                                         "timestamp_seconds": 0.5}]}
    dense["digest"] = canonical_digest(dense, digest_field="digest")

    def extend(**kwargs):  # the denser still interval, written once
        kwargs["output_root"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_root"] / "atlas_video_survey.json").write_text(json.dumps(dense))
    monkeypatch.setattr(extension, "extend_survey", extend)

    class Invoker:
        def configure_reservation_audit(self, **_):
            pass

        def invoke(self, spec, value):
            counts["invoked"] += 1
            answer = {**proposal, **(second or {})} if counts["invoked"] == 2 else proposal
            return SimpleNamespace(output=answer, cost_usd=.03, usage={"input_tokens": 100}, model=spec.model,
                                   sdk_version="test")
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", lambda *, experimental=False, **kw: (
        planner.review_survey(**{**kw, "invoker": Invoker()}, release_blockers=lambda: ("c" * 40, []))))

    def select(**_):
        counts["selected"] += 1
        raise ValueError(selection_refusal)
    monkeypatch.setattr(inputs, "prepare_video_inputs", select)
    context = {"description": "Move the box to the shelf", "context_digest": "sha256:" + "b" * 64,
               "request_id": "r", "scene_id": "s", "capture_id": "c"}
    return video, context, counts


@pytest.mark.parametrize("refusal", ["atlas_insufficient_overlapping_distinct_views",
                                     "atlas_coverage_anchor_quality_insufficient"])
def test_a_retried_selection_stop_reuses_both_settled_scene_reviews(tmp_path, monkeypatch, refusal):
    video, context, counts = _pipeline_world(tmp_path, monkeypatch, selection_refusal=refusal)
    job_dir = tmp_path / "job"
    clock = [1_000_000]
    job = {"schema_version": "website_atlas_preview_job.v1", "state": "processing", "reason": None, "revision": 1,
           "enqueued_at_epoch": clock[0], "next_attempt_at_epoch": clock[0], "stage": "queued"}
    step = {"job_dir": job_dir, "source_video": video, "task_context": context, "checkpoint": lambda *_a, **_k: None,
                "now": lambda: clock[0]}
    job = preview.advance_preparation_step(job, **step)
    assert (job["state"], job["reason"]) == ("needs_input", refusal)
    # The two scene reviews were bought and settled once each.
    assert counts == {"invoked": 2, "reserved": 2, "settled": 2, "selected": 1}
    receipts = sorted(p.name for p in (job_dir / "scene_planner").glob("*.json"))
    assert len([name for name in receipts if not name.endswith(".settlement.json")]) == 2
    for retry in range(len(REFUSAL_RETRY_DELAYS_SECONDS)):
        clock[0] = job["retry_at_epoch"]
        job = preview.advance_preparation_step(job, **step)
        assert (job["state"], job["reason"]) == ("needs_input", refusal)
        # Each retry reads the settled receipts back: no model call, no reservation, no settlement.
        assert counts == {"invoked": 2, "reserved": 2, "settled": 2, "selected": retry + 2}
    assert job["retry_at_epoch"] is None
    assert sorted(p.name for p in (job_dir / "scene_planner").glob("*.json")) == receipts


def test_a_retried_review_stop_reuses_the_settled_review(tmp_path, monkeypatch):
    """A stop the review's own answer decides (here: an uncertain subject) is read back from its receipt."""
    video, context, counts = _pipeline_world(tmp_path, monkeypatch, selection_refusal="unused")
    from blueprint_pipeline import paid_resource_allocator as allocator
    planned = allocator.run_sponsored_website_atlas_plan

    def uncertain(**kwargs):
        receipt = planned(**kwargs)
        return {**receipt, "proposal": {**receipt["proposal"], "subject_origin": "uncertain"}}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", uncertain)
    clock = [1_000_000]
    job = {"schema_version": "website_atlas_preview_job.v1", "state": "processing", "reason": None, "revision": 1,
           "enqueued_at_epoch": clock[0], "next_attempt_at_epoch": clock[0], "stage": "queued"}
    step = {"job_dir": tmp_path / "job", "source_video": video, "task_context": context,
                "checkpoint": lambda *_a, **_k: None, "now": lambda: clock[0]}
    job = preview.advance_preparation_step(job, **step)
    assert (job["state"], job["reason"]) == ("needs_input", "atlas_scene_subject_origin_uncertain")
    assert counts["invoked"] == counts["reserved"] == counts["settled"] == 1
    clock[0] = job["retry_at_epoch"]
    job = preview.advance_preparation_step(job, **step)
    assert (job["state"], job["reason"]) == ("needs_input", "atlas_scene_subject_origin_uncertain")
    assert counts["invoked"] == counts["reserved"] == counts["settled"] == 1
    assert job["retry_at_epoch"] == clock[0] + REFUSAL_RETRY_DELAYS_SECONDS[1]


@pytest.mark.parametrize("second,reason", [
    ({"subject_origin": "synthetic_design", "coverage_requirements": [
        {"role": "task_region", "label": "display", "required_frame_ids": ["1"]},
        {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["0"]}]},
     "atlas_synthetic_subject_requires_preserved_capture"),
])
def test_a_stop_decided_by_the_second_review_reuses_both_settled_reviews(tmp_path, monkeypatch, second, reason):
    video, context, counts = _pipeline_world(tmp_path, monkeypatch, selection_refusal="unused", second=second)
    clock = [1_000_000]
    job = {"schema_version": "website_atlas_preview_job.v1", "state": "processing", "reason": None, "revision": 1,
           "enqueued_at_epoch": clock[0], "next_attempt_at_epoch": clock[0], "stage": "queued"}
    step = {"job_dir": tmp_path / "job", "source_video": video, "task_context": context,
                "checkpoint": lambda *_a, **_k: None, "now": lambda: clock[0]}
    for _ in range(3):
        job = preview.advance_preparation_step(job, **step)
        assert (job["state"], job["reason"]) == ("needs_input", reason)
        assert counts["invoked"] == counts["reserved"] == counts["settled"] == 2
        clock[0] = job["retry_at_epoch"]


def test_a_job_that_passes_writes_the_same_record_as_before(monkeypatch, site):
    """Only a stop gains fields: a run that goes straight through keeps every key and value of main."""
    from tests.test_website_atlas_preview import _completed
    _enqueue(site)
    world = World(monkeypatch, site, [_completed])
    assert world.advance() == "ready"
    job = _job(site)
    assert not {"refusal_retries", "refusal_retries_total", "last_refusal_reason", "last_error"} & job.keys()
    assert job["retry_at_epoch"] is None and Path(site.job).is_file()
