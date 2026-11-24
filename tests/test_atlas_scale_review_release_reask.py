"""A ready room whose scale review ended with no valid answer is asked again by the job itself, once per release.

Before 2026-11-23 every scale review the provider refused for no credit ended as
``atlas_scene_planner_output_invalid``: the room was built, archived and shipped unscaled, and nothing asked the review
again. The planner reopens such a call once on a changed release (``atlas_scene_planner.review_survey``), but only if
the job's step runs, and nothing ran a ready job's step for that code (``website_atlas_preview``'s release reopen
covers failed and needs-input jobs only; the outage re-ask covers ``atlas_scale_review_waiting_on_provider`` only).
Now the ready job asks once per release, through the same admissions, keeps the shipped room through any answer that
is no privacy or rights stop, and never asks after a wall, a withdrawal, a money stop or its window's end.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_atlas_preview.py
from __future__ import annotations

import json

from blueprint_pipeline import website_preparation_retry as retry
from tests.test_atlas_unscaled_room import unscaled_job
from tests.test_website_atlas_preview import World, _enqueue, _job, _live_grant, _withdraw
from tests.test_website_atlas_preview import site  # noqa: F401 - the fixture

LEGACY = "atlas_scene_planner_output_invalid"
A, B, C = "a" * 40, "b" * 40, "c" * 40


def _shipping(place, *, reason=LEGACY, wall=None):
    """The stored unscaled room as the Atlas job's own result, its scale review ended on ``reason``."""
    job_dir = unscaled_job(place.capture, reason=reason)
    if wall:
        result = json.loads((job_dir / "result.json").read_text())
        result["source_pose"]["wall"] = wall
        from tests.test_website_atlas_scene_source import digest, write
        result.pop("digest")
        write(job_dir / "result.json", digest(result))
    (job_dir / "preview_job.json").unlink()
    _enqueue(place)
    return json.loads((job_dir / "result.json").read_text())


def _answers(base, source_pose):
    """A step that writes the room's result as ``finalize_background_result`` does, its source pose ``source_pose``."""
    import hashlib
    import zipfile
    from blueprint_pipeline.task_object_authoring_contracts import write_json
    from blueprint_pipeline.website_atlas_pipeline import canonical_digest

    def step(kwargs):
        bundle = kwargs["output_root"] / "downloads/atlas-scene-0123.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr(zipfile.ZipInfo("manifest.json", date_time=(2026, 10, 22, 0, 0, 0)),
                             json.dumps(source_pose, sort_keys=True))
        value = {key: item for key, item in base.items() if key != "digest"}
        value["download"] = {"zip_path": str(bundle),
                             "zip_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
                             "manifest_digest": "sha256:" + "e" * 64}
        value["source_pose"] = source_pose
        value["digest"] = canonical_digest(value, digest_field="digest")
        write_json(kwargs["output_root"] / "result.json", value)
        return value
    return step


def _released(monkeypatch, commit):
    release = {"commit": commit}
    monkeypatch.setattr(retry, "_release", lambda: release["commit"])
    return release


def test_a_ready_room_on_the_legacy_code_is_asked_once_per_release_until_measured(monkeypatch, site):  # noqa: F811
    base = _shipping(site)
    world = World(monkeypatch, site, [
        _answers(base, base["source_pose"]),                       # the room ships unscaled on release A
        _answers(base, {**base["source_pose"], "asked": 2}),       # release B: asked, still no valid answer
        _answers(base, {"status": "completed", "scale_status": "estimated"}),  # release C: measured
    ])
    _live_grant(site, world.clock + 5 * 86400)
    release = _released(monkeypatch, A)
    assert world.advance() == "ready" and len(world.runs) == 1
    shipped = _job(site)
    assert shipped["retry_at_epoch"] is None and shipped.get("scale_review_due_at_epoch") is None
    # The release it shipped on decided it already: not asked again on it.
    world.clock += 7200
    assert world.advance() == "ready" and len(world.runs) == 1
    # A new release reopens the planner call once, so the job asks once.
    release["commit"] = B
    assert world.advance() == "ready" and len(world.runs) == 2
    again = _job(site)
    assert again["state"] == "ready" and again["retry_at_epoch"] is None
    assert again.get("scale_review_due_at_epoch") is None and again["scale_review_release"] == B
    world.clock += 7200
    assert world.advance() == "ready" and len(world.runs) == 2
    release["commit"] = C
    assert world.advance() == "ready" and len(world.runs) == 3
    measured = _job(site)
    assert measured["result_digest"] != shipped["result_digest"]
    assert json.loads((site.job.parent / "result.json").read_text())["source_pose"]["status"] == "completed"
    # Measured: never asked again, whatever the release.
    release["commit"] = "d" * 40
    world.clock += 7200
    assert world.advance() == "ready" and len(world.runs) == 3


def test_an_archived_room_from_before_is_asked_once_on_the_running_release(monkeypatch, site):  # noqa: F811
    base = _shipping(site)
    world = World(monkeypatch, site, [_answers(base, base["source_pose"]),
                                      _answers(base, {**base["source_pose"], "asked": 2})])
    _live_grant(site, world.clock + 5 * 86400)
    _released(monkeypatch, A)
    assert world.advance() == "ready"
    # Archived by a release before this one: no release is recorded as having asked.
    archived = {key: value for key, value in _job(site).items() if key != "scale_review_release"}
    site.job.write_text(json.dumps(archived))
    assert world.advance() == "ready" and len(world.runs) == 2
    assert world.advance() == "ready" and len(world.runs) == 2


def test_a_money_stop_keeps_the_shipped_room_and_no_later_release_asks_again(monkeypatch, site):  # noqa: F811
    from blueprint_pipeline.hard_stops import MoneyStop
    base = _shipping(site)
    world = World(monkeypatch, site, [_answers(base, base["source_pose"]), MoneyStop("job_spend_cap_reached")])
    _live_grant(site, world.clock + 5 * 86400)
    release = _released(monkeypatch, A)
    assert world.advance() == "ready"
    shipped = _job(site)
    release["commit"] = B
    assert world.advance() == "ready" and len(world.runs) == 2
    kept = _job(site)
    assert (kept["archive"], kept["result_digest"]) == (shipped["archive"], shipped["result_digest"])
    assert kept.get("scale_review_due_at_epoch") is None and kept["retry_at_epoch"] is None
    release["commit"] = C
    world.clock += 7200
    assert world.advance() == "ready" and len(world.runs) == 2


def test_a_wall_on_the_pose_is_never_asked_again(monkeypatch, site):  # noqa: F811
    base = _shipping(site, wall="money")
    world = World(monkeypatch, site, [_answers(base, base["source_pose"])])
    _live_grant(site, world.clock + 5 * 86400)
    release = _released(monkeypatch, A)
    world.advance()
    runs = len(world.runs)
    release["commit"] = B
    world.advance()
    assert len(world.runs) == runs


def test_a_withdrawal_stops_the_asking(monkeypatch, site):  # noqa: F811
    base = _shipping(site)
    world = World(monkeypatch, site, [_answers(base, base["source_pose"])])
    _live_grant(site, world.clock + 5 * 86400)
    release = _released(monkeypatch, A)
    assert world.advance() == "ready"
    _withdraw(site)
    release["commit"] = B
    world.advance()
    assert len(world.runs) == 1


def test_no_ask_once_the_jobs_window_or_grant_has_ended(monkeypatch, site):  # noqa: F811
    base = _shipping(site)
    world = World(monkeypatch, site, [_answers(base, base["source_pose"])])
    _live_grant(site, world.clock + 3600)
    release = _released(monkeypatch, A)
    assert world.advance() == "ready"
    world.clock += 7200  # the grant has expired
    release["commit"] = B
    assert world.advance() == "ready" and len(world.runs) == 1


def test_a_review_that_answered_without_a_scale_is_not_asked_again(monkeypatch, site):  # noqa: F811
    # ``atlas_scale_review_unresolved``: the review answered; its receipt is reused, so a new ask changes nothing.
    base = _shipping(site, reason="atlas_scale_review_unresolved")
    world = World(monkeypatch, site, [_answers(base, base["source_pose"])])
    _live_grant(site, world.clock + 5 * 86400)
    release = _released(monkeypatch, A)
    assert world.advance() == "ready"
    release["commit"] = B
    assert world.advance() == "ready" and len(world.runs) == 1
