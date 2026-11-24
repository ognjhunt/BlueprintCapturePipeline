"""A job at the $40 cap mid-run: its README says the job reached its limit on paid work. Real handoff and real
company builder.

From the $0 stress round of 2026-11-24 (the test reproduced a gap, now closed). Offline, no provider, nothing paid
for (``tests/test_website_scene_handoff_navigation.room_job``: the real handoff on fixture geometry; the Atlas lane
stood in).
"""
# Covers: src/blueprint_pipeline/website_company_simulation.py
from __future__ import annotations

import json
import zipfile

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import website_scene_handoff as handoff  # noqa: E402
from blueprint_pipeline.website_job_spend_cap import SPENDING_CAP_REACHED  # noqa: E402
from tests.test_website_company_simulation import World, _package  # noqa: E402
from tests.test_website_scene_handoff_navigation import ATLAS, room_job  # noqa: E402


def _readme(job) -> str:
    with zipfile.ZipFile(_package(job)) as archive:
        return " ".join(archive.read("README.txt").decode().split())


@pytest.mark.parametrize("family", ["rigid_pick_place", "articulated_open_close"])
def test_a_job_stopped_at_the_cap_ships_the_room_and_says_why(tmp_path, monkeypatch, family):
    """The Atlas lane hit the job's $40 cap (its flags carry ``spending_cap_reached``). The handoff ships the room
    package, buys nothing more and waits for no one (owner, 2026-11-19). The README says the object is missing
    because the job reached its limit on paid work (AGENTS.md), not leave the company guessing."""
    job, _context, first = room_job(tmp_path, monkeypatch, family=family, description="Move the bin to the shelf")
    import blueprint_pipeline.website_atlas_scene_source as source
    atlas = source.atlas_scene_source()  # the stand-in room_job installed
    monkeypatch.setattr(ATLAS, lambda **_: {**atlas, "flags": [SPENDING_CAP_REACHED]})
    result = job.rerun()
    assert result["status"] == handoff.ROOM_PACKAGE, result
    record = json.loads(open(result["room_package"]["path"]).read())
    assert record["not_prepared"] == [{"piece": "object", "code": SPENDING_CAP_REACHED}]
    published = World(job).publish()
    assert published["status"] == "published", published
    readme = _readme(job)
    assert "Blueprint could not prepare the job's object: the job reached its limit on paid work." in readme
    assert "Blueprint could not prepare the job's object." not in readme
    assert "limit" in readme.split("How it was made", 1)[1], "the README never says the job stopped at its spending limit"
