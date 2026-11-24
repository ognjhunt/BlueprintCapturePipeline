"""A built room never waits for a scale review its provider cannot answer now (owner, 2026-11-23).

Live test job, 2026-11-23: OpenAI answered the room build's scale review 429 ``insufficient_quota`` three times. With
an outage read as an outage (pull request 3703), the source-pose step would wait the whole room build for as long as
the provider is out, and nothing would ship. Now the source pose records the review as waiting on its provider
(``atlas_unscaled_room.SCALE_REVIEW_WAITING``), the built room ships unscaled and flagged at once, and the Atlas job
asks the review again on the outage schedule (2 min doubling to 1 h); when it answers, the job's result is the
measured room.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_atlas_scene_source as source
from tests.test_atlas_unscaled_room import SCALE_NOT_MEASURED, unscaled_job
from tests.test_website_atlas_preview import World, _completed, _enqueue, _job
from tests.test_website_atlas_preview import site  # noqa: F401 - the fixture

WAITING = "atlas_scale_review_waiting_on_provider"


def _no_credit():
    import httpx
    import openai
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    return openai.RateLimitError("Error code: 429", response=response, body={
        "message": "You have no credits remaining.", "type": "insufficient_quota", "code": "insufficient_quota"})


def _posed_then_scale_review(monkeypatch, root, answer):
    """The original views are posed and measured without a scale; the scale review then ``answer``s."""
    from blueprint_pipeline import (atlas_context_fov, atlas_reference_masks, atlas_scale_review,
                                    atlas_scene_measurements, atlas_source_pose_inputs,
                                    paid_resource_allocator, website_operation_coverage_review)
    from blueprint_pipeline import atlas_scene_automation as automation
    from blueprint_pipeline.website_atlas_pose import ATLAS_PRICE_REVIEW_ENV
    inputs = {"frames": [{"frame_id": "a"}], "binding": {"source_video_digest": "sha256:v"}, "digest": "sha256:i"}
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda pose_root: (pose_root, inputs))
    monkeypatch.setenv(ATLAS_PRICE_REVIEW_ENV, "{}")
    (root / "atlas_posed_rgbd_result.json").write_text("{}")
    monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_pose",
                        lambda **_: {"status": "completed", "operation_id": "source"})
    monkeypatch.setattr(atlas_scene_measurements, "load_views",
                        lambda *_: ({"digest": "sha256:pose", "operation": {"id": "s"}}, ["views"]))
    monkeypatch.setattr(atlas_scene_measurements, "surfaces_on_posed_views", lambda *_: [])
    monkeypatch.setattr(atlas_scene_measurements, "estimate_sizes", lambda **kw: {
        "digest": "sha256:unscaled", "status": "relative_sizes_only", "scale_status": "unscaled_provider_world_units"})
    monkeypatch.setattr(atlas_context_fov, "verify_pose_context_fov", lambda **_: None)
    monkeypatch.setattr(automation, "measurement_plan", lambda review: {"scale_references": []})
    monkeypatch.setattr(atlas_reference_masks, "segment_scale_references", lambda **_: {})
    monkeypatch.setattr(atlas_scale_review, "needs_scale_review", lambda _: True)
    monkeypatch.setattr(website_operation_coverage_review, "scale_operation_coverage_options", lambda **_: {})

    def review(**_):
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr(atlas_scale_review, "repair_source_scale", review)
    return inputs


def _source_pose(root, inputs, **options):
    from datetime import datetime, timezone
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    policy = {"pose_input_cost_usd": .01, "rate_card_reference": "r", "reviewed_by": "o",
              "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    return pipeline._source_pose(pose_root=root, inputs=inputs, review={}, policy=policy,
                                 task_context={"capture_id": "c"}, check=lambda *_a, **_k: None, experimental=False,
                                 max_polls=1, configuration={"policy": {}}, **options)


def test_a_built_background_records_a_scale_review_its_provider_cannot_answer_and_goes_on(tmp_path, monkeypatch):
    root = tmp_path / "pose"
    root.mkdir()
    inputs = _posed_then_scale_review(monkeypatch, root, _no_credit())
    assert _source_pose(root, inputs, scale_outage_ships=True) == {
        "status": "needs_input", "reason": WAITING, "scene_measurements_digest": "sha256:unscaled"}
    # A scene kept as filmed cannot ship unscaled: it still waits on the outage, as before.
    import openai
    with pytest.raises(openai.RateLimitError):
        _source_pose(root, inputs)


@pytest.mark.parametrize("error", [ValueError("atlas_scene_planner_invoker_budget_exhausted"),
                                   RuntimeError("a defect in the review")])
def test_only_a_provider_outage_is_recorded_as_waiting(tmp_path, monkeypatch, error):
    root = tmp_path / "pose"
    root.mkdir()
    inputs = _posed_then_scale_review(monkeypatch, root, error)
    if isinstance(error, ValueError):
        # A refusal is held under its own code (money here), as before; it never reads as the provider waiting.
        held = _source_pose(root, inputs, scale_outage_ships=True)
        assert held["reason"] == str(error) and held["wall"] == "money"
    else:
        with pytest.raises(RuntimeError):
            _source_pose(root, inputs, scale_outage_ships=True)


def test_a_room_whose_scale_review_waits_on_its_provider_ships_flagged(tmp_path):
    unscaled_job(tmp_path, reason=WAITING)
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready" and value["flags"] == [SCALE_NOT_MEASURED]
    assert value["not_prepared"] == [WAITING] and value["base_scene"]["meters_per_unit"] is None


def _built(kwargs, *, waiting=True, digest="d"):
    return {**_completed(kwargs), "digest": "sha256:" + digest * 64,
            "source_pose": ({"status": "needs_input", "reason": WAITING} if waiting
                            else {"status": "completed", "scale_status": "estimated"})}


def test_the_job_asks_a_waiting_scale_review_again_on_the_outage_schedule_and_keeps_the_room(monkeypatch, site):  # noqa: F811 - the fixture
    _enqueue(site)
    world = World(monkeypatch, site, [
        lambda kw: _built(kw),                       # the room ships unscaled; the review waits
        lambda kw: _built(kw),                       # asked again: still waiting
        ValueError("atlas_scene_review_changed"),    # asked again: an ordinary refusal keeps the room
        lambda kw: _built(kw, waiting=False, digest="f"),  # asked again: measured
    ])
    assert world.advance() == "ready"
    shipped = _job(site)
    assert shipped["retry_at_epoch"] is None and shipped["scale_review_due_at_epoch"] == world.clock + 120
    # Final for every reader until the review is due: no step runs before.
    world.clock += 119
    assert world.advance() == "ready" and len(world.runs) == 1
    world.clock += 1
    assert world.advance() == "ready" and len(world.runs) == 2
    assert _job(site)["scale_review_due_at_epoch"] == world.clock + 240 and _job(site)["retry_at_epoch"] is None
    world.clock += 240
    assert world.advance() == "ready" and len(world.runs) == 3
    kept = _job(site)
    assert (kept["archive"], kept["result_digest"], kept["revision"]) == (
        shipped["archive"], shipped["result_digest"], shipped["revision"])
    assert kept["scale_review_last_outcome"]["reason"] == "atlas_scene_review_changed"
    assert kept["retry_at_epoch"] is None and kept["state"] == "ready"
    assert kept["scale_review_due_at_epoch"] == world.clock + 480
    world.clock += 480
    assert world.advance() == "ready" and len(world.runs) == 4
    measured = _job(site)
    assert measured["result_digest"] == "sha256:" + "f" * 64
    assert measured["scale_review_due_at_epoch"] is None and measured["retry_at_epoch"] is None
    world.clock += 7200
    assert world.advance() == "ready" and len(world.runs) == 4


def test_a_withdrawal_while_the_review_waits_still_stops_the_job(monkeypatch, site):  # noqa: F811 - the fixture
    from tests.test_website_atlas_preview import _withdraw
    _enqueue(site)
    world = World(monkeypatch, site, [lambda kw: _built(kw)])
    assert world.advance() == "ready"
    _withdraw(site)
    world.clock += 120
    assert world.advance() == "withdrawn" and len(world.runs) == 1


def test_a_money_stop_while_the_review_waits_keeps_the_room_and_stops_asking(monkeypatch, site):  # noqa: F811 - the fixture
    from blueprint_pipeline.hard_stops import MoneyStop
    _enqueue(site)
    world = World(monkeypatch, site, [lambda kw: _built(kw), MoneyStop("job_spend_cap_reached")])
    assert world.advance() == "ready"
    shipped = _job(site)
    world.clock += 120
    assert world.advance() == "ready" and len(world.runs) == 2
    kept = _job(site)
    assert kept["archive"] == shipped["archive"] and kept["scale_review_due_at_epoch"] is None
    world.clock += 7200
    assert world.advance() == "ready" and len(world.runs) == 2
    assert json.loads(site.job.read_text())["state"] == "ready"
