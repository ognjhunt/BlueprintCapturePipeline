"""The clock's reopens of a parked Atlas job (``website_atlas_preview._reopen_parked``), per stop and under one bound
(the $0 stress run of round 2, 2026-11-24).

The Atlas preview tests' own stand-ins (``tests/test_website_atlas_preview.World``: the task context, World Labs and the
WebApp as each answers), a live grant, a fake clock; nothing paid for, no network.
"""
# Covers: src/blueprint_pipeline/atlas_preparation_job_control.py
# Covers: src/blueprint_pipeline/website_atlas_preview.py
from __future__ import annotations

import json

from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline import website_preparation_retry as retry
from blueprint_pipeline.atlas_preparation_job_control import MAX_AUTOMATIC_REOPENS, automatic_reopens, reopens_spent
from tests import test_website_atlas_preview as atlas_tests
from tests.test_website_atlas_preview import _HOST_STOP, World, _completed, _enqueue, _job, _live_grant, _withdraw

#: The Atlas preview tests' capture, as a fixture of this module too.
site = atlas_tests.site

#: A later step's stop, on no wall either: the scene completion's own audit after the pose's audit passed.
_LATER_STOP = {"status": "needs_input", "download": None, "blockers": ["atlas_completion_gap_survey_unresolved"],
               "digest": "sha256:" + "4" * 64}


def _run_schedule(world, site, count):
    """Let the clock bring ``count`` reopens, each at its due time."""
    for _ in range(count):
        delay = preview.REFUSAL_RETRY_DELAYS_SECONDS[automatic_reopens(_job(site))]
        world.clock = _job(site)["revised_at_epoch"] + delay
        world.advance()


def test_a_job_that_moves_on_to_a_new_stop_gets_reopens_for_that_stop(monkeypatch, site):
    """The pose audit refused five reopens running; the sixth got past it, to a new stop at a later step. Its count
    was read from every automatic reopen in a row, whatever each stopped on, so the new stop was the job's end at
    once, with no reopen of its own. Now the count is the reopens that stopped on the stop the job is on."""
    _enqueue(site)
    world = World(monkeypatch, site, [_HOST_STOP] * 6 + [_LATER_STOP] * 7)
    _live_grant(site, world.clock + 7 * 86400)
    monkeypatch.setattr(retry, "_release", lambda: None)
    world.advance()
    _run_schedule(world, site, 6)
    job = _job(site)
    assert job["reason"] == "atlas_completion_gap_survey_unresolved" and len(world.runs) == 7
    assert automatic_reopens(job) == 0 and not reopens_spent(job)
    _run_schedule(world, site, 6)  # the new stop's own six
    job = _job(site)
    assert len(world.runs) == 13 and reopens_spent(job)
    assert [row.get("stopped_on") for row in job["retries"]][-6:] == [job["reason"]] * 6
    world.clock = job["revised_at_epoch"] + 7 * 86400
    world.advance()
    assert len(world.runs) == 13  # spent: its stop is its end


def test_a_job_whose_stops_keep_changing_is_reopened_at_most_the_overall_bound():
    """However its stops change, nothing loops for ever: past ``MAX_AUTOMATIC_REOPENS`` automatic reopens since an
    operator's, the job is not reopened by itself, whatever stop it is on."""
    rows = [{"reason": "parked_retry", "at_epoch": index, "stopped_on": f"atlas_stop_{index // 6}"}
            for index in range(MAX_AUTOMATIC_REOPENS)]
    job = {"state": "needs_input", "reason": "atlas_stop_new", "retries": rows}
    assert automatic_reopens(job) == 0 and reopens_spent(job)
    fewer = {**job, "retries": rows[:-1]}
    assert not reopens_spent(fewer)
    # An operator's reopen starts the count afresh.
    again = {**job, "retries": [*rows, {"reason": "operator_retry", "at_epoch": 99}]}
    assert not reopens_spent(again)


def test_an_operator_retry_starts_the_count_afresh_and_the_job_still_ends(monkeypatch, site):
    """Control: six reopens spent, an operator's ``retry_failed_job`` runs it again and the clock has six more; then
    it ends again. It never loops by itself."""
    _enqueue(site)
    world = World(monkeypatch, site, [_HOST_STOP] * 14)
    _live_grant(site, world.clock + 7 * 86400)
    monkeypatch.setattr(retry, "_release", lambda: None)
    world.advance()
    _run_schedule(world, site, 6)
    assert reopens_spent(_job(site)) and len(world.runs) == 7
    monkeypatch.setattr(preview, "time", type("T", (), {"time": staticmethod(lambda: world.clock)}))
    preview.retry_failed_job(site.job, now=lambda: world.clock)
    assert automatic_reopens(_job(site)) == 0
    world.advance()
    _run_schedule(world, site, 6)
    assert reopens_spent(_job(site)) and len(world.runs) == 14
    world.clock += 7 * 86400
    world.advance()
    assert len(world.runs) == 14


def test_a_release_reopen_counts_and_the_clock_never_runs_one_twice(monkeypatch, site):
    """Control: a release reopen counts as one; a release seen again and the clock's due time on the same tick give
    one run, not two."""
    _enqueue(site)
    world = World(monkeypatch, site, [_HOST_STOP] * 4)
    _live_grant(site, world.clock + 7 * 86400)
    release = {"commit": "a" * 40}
    monkeypatch.setattr(retry, "_release", lambda: release["commit"])
    world.advance()
    world.advance()  # the release that parked it: one reopen at once
    assert automatic_reopens(_job(site)) == 1 and len(world.runs) == 2
    release["commit"] = "b" * 40
    world.clock = _job(site)["revised_at_epoch"] + preview.REFUSAL_RETRY_DELAYS_SECONDS[1]  # due and a new release
    world.advance()
    assert automatic_reopens(_job(site)) == 2 and len(world.runs) == 3


def test_a_withdrawn_capture_is_never_reopened_by_the_clock(monkeypatch, site):
    """Withdrawal: a parked job whose site withdrew is not run again, by the clock or a release."""
    _enqueue(site)
    world = World(monkeypatch, site, [_HOST_STOP, _HOST_STOP, _completed])
    _live_grant(site, world.clock + 7 * 86400)
    monkeypatch.setattr(retry, "_release", lambda: "a" * 40)
    world.advance()
    world.advance()
    runs = len(world.runs)
    _withdraw(site)
    monkeypatch.setattr(retry, "_release", lambda: "c" * 40)
    world.clock = _job(site)["revised_at_epoch"] + 7 * 3600
    world.advance()
    assert len(world.runs) == runs, "a withdrawn capture's parked job ran again"
    assert json.loads(site.job.read_text())["state"] in {"needs_input", "withdrawn", "failed"}
