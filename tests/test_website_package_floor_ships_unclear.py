"""A room floor whose object path the pass cannot classify never ships, though the room is ready and the live
rights are checked separately: no privacy, rights or money reason stops it. It should ship flagged, like an
object still being built (#3686)."""
# Covers: src/blueprint_pipeline/website_company_simulation.py
from __future__ import annotations

import pytest

from blueprint_pipeline import website_company_simulation as sim
from tests.test_website_company_simulation import World, _manifest
from tests.test_website_package_floor import DAY, UNCLEAR, Floor


@pytest.mark.parametrize("state", sorted(UNCLEAR))
@pytest.mark.parametrize("age", [0, DAY])
def test_a_ready_room_ships_flagged_when_its_object_path_cannot_be_classified(tmp_path, monkeypatch, state, age):
    floor = Floor(tmp_path, monkeypatch)
    monkeypatch.setattr(sim, "SCREENING_ROOT", tmp_path / "no-screening")
    UNCLEAR[state](floor)
    if state == "intent_missing":
        floor.age(age)
    world = World(floor.job)
    result = world.publish()
    assert result["status"] == "published", result
    assert _manifest(floor.job)["source_kind"] == "floor"
