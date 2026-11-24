"""A room that is built but whose source pose failed, so its scale was never measured, ships flagged (owner, 2026-11-23).

Shape of internal test job 2 (``itest-917ca852…``, 20 Nov): the object-removed background's splat and mesh are
completed, but World Labs failed both pose attempts on the original views (code 13, ``EXECUTION_FAILED``), so the
result holds ``source_pose`` ``needs_input`` (``website_atlas_pose_attempt_failed``) and measurements of relative
sizes only. Before, the scene source answered ``website_atlas_source_pose_unavailable`` on every hourly retry and the
job never shipped. Now the scene source hands on the background levelled in its own units, ``meters_per_unit`` None
and flagged ``scale_not_measured``; the room package records relative units and no metres value; the company package
says so. A wall in the pose (privacy, rights, money), the spending cap, a withdrawal, a takedown or revoked rights
still ship nothing, and a measured room is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_atlas_scene_source as source
from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.website_job_spend_cap import SPENDING_CAP_REACHED
from tests.test_website_atlas_scene_source import digest, job_fixture, write

POSE_FAILED = "website_atlas_pose_attempt_failed"
#: The flag and the README sentence (``atlas_unscaled_room``), as literals so this file runs where it does not exist.
SCALE_NOT_MEASURED = "scale_not_measured"
SCALE_NOT_MEASURED_NOTE = "Sizes in this room are relative; its scale was not measured."


def unscaled_job(capture_root: Path, *, reason: str = POSE_FAILED, extra: dict | None = None) -> Path:
    """The job fixture rewritten to the stored shape: no source pose, relative sizes, the background completed."""
    job_dir = job_fixture(capture_root)
    # The original views' own pose never completed: nothing of it is on disk.
    (job_dir / "pose" / "atlas_posed_rgbd_result.json").unlink()
    (job_dir / "pose" / "scene_measurements.json").unlink()
    result = json.loads((job_dir / "result.json").read_text())
    result.pop("digest")
    result.update(source_pose={"status": "needs_input", "reason": reason},
                  scene_measurements={"schema_version": "atlas_scene_measurements.v1", "status": "relative_sizes_only",
                                      "meters_per_unit_estimate": None, "meters_per_unit_range": None,
                                      "scale_status": "unscaled_provider_world_units"},
                  splat_available=True, mesh_available=True, blockers=["metric_scene_registration_unproven"],
                  scale_status="unscaled_provider_world_units", **(extra or {}))
    result = write(job_dir / "result.json", digest(result))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    return job_dir


def _metres_values(value, path=""):
    """Every non-zero number under a key that names metres."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _metres_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from _metres_values(item, path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0 and (
            path.endswith("_m") or "meters" in path.rsplit(".", 1)[-1] or "metres" in path.rsplit(".", 1)[-1]):
        yield path, value


@pytest.mark.parametrize("reason", ["atlas_scale_review_unresolved", "atlas_scene_planner_output_invalid"])
def test_a_built_room_whose_scale_review_ended_without_a_scale_is_handed_on_flagged_too(tmp_path, reason):
    unscaled_job(tmp_path, reason=reason)
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready" and value["flags"] == [SCALE_NOT_MEASURED]
    assert value["source_geometry"] is None and value["not_prepared"] == [reason]
    assert value["base_scene"]["meters_per_unit"] is None


def test_a_built_room_with_no_source_pose_is_handed_on_levelled_in_its_own_units_and_flagged(tmp_path):
    unscaled_job(tmp_path)
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready" and value["flags"] == [SCALE_NOT_MEASURED]
    assert value["source_geometry"] is None and value["not_prepared"] == [POSE_FAILED]
    base = value["base_scene"]
    assert base["meters_per_unit"] is None and base["scale_status"] == SCALE_NOT_MEASURED
    assert base["anchor"] is None and base["operation_id"] == "op_background"
    assert base["levelling"]["up_basis"] == "observed_floor" and base["ground_plane_offset_m"] == 0.0
    assert base["collision_mesh_path"].endswith("atlas_mesh_levelled.glb")
    assert "camera_path_levelled_m" not in value
    assert base["levelling"]["provider_to_levelled"] and "scale_authority" not in base
    # Read twice, it is the same scene.
    assert source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep") == value


def test_a_measured_room_is_handed_on_exactly_as_before(tmp_path):
    job_fixture(tmp_path)
    value = source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready" and "flags" not in value and "not_prepared" not in value
    assert value["base_scene"]["meters_per_unit"] == pytest.approx(0.625)
    assert "scale_status" not in value["base_scene"] and value["source_geometry"]["digest"]


@pytest.mark.parametrize("reason", [
    # A wall the pose controller types (``hard_stops``) and records as such (``website_atlas_pipeline._source_pose``).
    ("website_atlas_provider_retention_admission_mismatch", {"wall": "privacy_rights"}),
    # Walls by their words, as stored before the type was recorded.
    ("website_capture_consent_revoked", {}),
    ("account_monthly_spend_limit_reached", {}),
    ("website_job_spend_cap_reached", {}),
])
def test_a_pose_stopped_by_a_wall_ships_nothing(tmp_path, reason):
    code, recorded = reason
    job_dir = unscaled_job(tmp_path)
    result = json.loads((job_dir / "result.json").read_text())
    result.pop("digest")
    result["source_pose"] = {"status": "needs_input", "reason": code, **recorded}
    result = write(job_dir / "result.json", digest(result))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    assert source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep") == {
        "status": "needs_input", "reason": "website_atlas_source_pose_unavailable", "job_reason": code}


def test_the_spending_cap_still_answers_as_before(tmp_path):
    unscaled_job(tmp_path, reason=SPENDING_CAP_REACHED, extra={"reason": SPENDING_CAP_REACHED})
    assert source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep") == {
        "status": "needs_input", "reason": SPENDING_CAP_REACHED, "observed_scene_available": True,
        "simulation_qualified": False}


def shipped_room(tmp_path, monkeypatch):
    """The unscaled scene source through the real handoff onto the company builder's own test capture."""
    from tests.test_website_scene_handoff_navigation import ATLAS, room_job

    unscaled_job(tmp_path / "atlas-job")
    atlas = source.atlas_scene_source(capture_root=tmp_path / "atlas-job", output_root=tmp_path / "prep")
    job, _context, _plain = room_job(tmp_path, monkeypatch, family="wiping", description="Wipe the table.")
    monkeypatch.setattr(ATLAS, lambda **_: atlas)
    return job, job.rerun()


def test_the_unscaled_room_ships_to_companies_flagged_with_no_metres_value_in_its_record(tmp_path, monkeypatch):
    from tests.test_website_scene_handoff_navigation import _published, _record, _said

    job, result = shipped_room(tmp_path, monkeypatch)
    assert result["status"] == handoff.ROOM_PACKAGE and result["job_disposition"] == "accepted"
    assert result["geometry_backend"] is None and result["source_geometry_digest"] is None
    record = _record(result)
    assert record["flags"] == [SCALE_NOT_MEASURED]
    assert record["not_prepared"] == [{"piece": "object", "code": POSE_FAILED}]
    assert job.rerun() == result  # the handoff reading the result twice
    # The sealed half the company builder reads: its record and plan hold relative units and no metres value.
    room = job.capture / "pipeline/website_scene_preparation/room_package/half"
    for path in room.rglob("*.json"):
        assert list(_metres_values(json.loads(path.read_text()))) == [], path
    from blueprint_pipeline.team_scene_robot_absent import PLAN_FILENAME
    plan = json.loads(next(room.rglob(PLAN_FILENAME)).read_text())
    assert plan["task_kind"] == "not_defined_by_blueprint"
    published, files = _published(job)
    assert published["status"] == "published"
    readme = files["README.txt"]
    said = _said(readme)
    assert said.count(SCALE_NOT_MEASURED_NOTE) == 1
    from blueprint_pipeline import atlas_unscaled_room as unscaled
    assert (unscaled.SCALE_NOT_MEASURED, unscaled.SCALE_NOT_MEASURED_NOTE) == (SCALE_NOT_MEASURED, SCALE_NOT_MEASURED_NOTE)
    assert sim.NOT_PREPARED.format(piece="object") in said
    assert "scene.usda is in metres" not in readme
    assert "- scene.usda states 1 unit as 1 metre only so that tools can open it; the room's scale was not" in readme


def test_takedown_and_withdrawal_still_stop_the_unscaled_room(tmp_path, monkeypatch):
    from tests.test_website_company_simulation import World, _withdraw, _write

    job, _result = shipped_room(tmp_path, monkeypatch)
    world = World(job)
    assert world.publish()["status"] == "published"
    takedown = _write(job.capture / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
    assert world.publish() == {"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"}
    assert len(world.retracted) == 1
    takedown.unlink()
    assert world.publish()["status"] == "published"
    _withdraw(job.capture)
    assert world.publish()["status"] == "withdrawn"


@pytest.mark.parametrize("rights", ["refused", "not_sharing"])
def test_the_unscaled_room_follows_the_live_rights_like_any_package(tmp_path, monkeypatch, rights):
    from tests.test_website_company_simulation import OLDER, World

    job, _result = shipped_room(tmp_path, monkeypatch)
    world = World(job)
    world.rights = ({"consent_status": "revoked", "consent_revoked": True, "statement_version": None}
                    if rights == "refused" else {"consent_status": "granted", "consent_revoked": False,
                                                 "statement_version": OLDER})
    result = world.publish()
    assert result["status"] in ({"refused"} if rights == "refused" else {"not_permitted"}), result
    assert world.delivered == []


#: Typed walls of the pose controller (``website_atlas_pose``) whose codes name no wall by their words, and one code
#: no list knows. A pose result stored before the wall's type was recorded carries the code alone.
UNRECOGNISED_STOPS = ["website_atlas_provider_retention_admission_mismatch",
                      "website_atlas_provider_retention_decision_required",
                      "website_atlas_pose_admission_expired", "website_atlas_pose_code_never_seen_before"]


@pytest.mark.parametrize("recorded", [{}, {"wall": "privacy_rights"}], ids=["stored_before", "stored_now"])
@pytest.mark.parametrize("code", UNRECOGNISED_STOPS)
def test_only_a_known_ordinary_pose_failure_ships_unscaled(tmp_path, monkeypatch, code, recorded):
    from blueprint_pipeline.atlas_unscaled_room import ships_unscaled
    from tests.test_website_scene_handoff_navigation import ATLAS, room_job

    pose = {"status": "needs_input", "reason": code, **recorded}
    assert ships_unscaled(pose) is False
    assert ships_unscaled({"status": "needs_input", "reason": POSE_FAILED}) is True
    # The scale review ending without a scale is ordinary too (the live job of 2026-11-23 ended this way); a wall
    # recorded beside either code still ships nothing.
    for ordinary in ("atlas_scale_review_unresolved", "atlas_scene_planner_output_invalid"):
        assert ships_unscaled({"status": "needs_input", "reason": ordinary}) is True
        assert ships_unscaled({"status": "needs_input", "reason": ordinary, "wall": "money"}) is False
    job_dir = unscaled_job(tmp_path / "atlas-job")
    result = json.loads((job_dir / "result.json").read_text())
    result.pop("digest")
    result["source_pose"] = pose
    result = write(job_dir / "result.json", digest(result))
    write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    atlas = source.atlas_scene_source(capture_root=tmp_path / "atlas-job", output_root=tmp_path / "prep")
    assert atlas == {"status": "needs_input", "reason": "website_atlas_source_pose_unavailable", "job_reason": code}
    job, _context, _plain = room_job(tmp_path, monkeypatch, family="wiping", description="Wipe the table.")
    monkeypatch.setattr(ATLAS, lambda **_: atlas)
    # Through the handoff it is no shipped room: the job waits, as for any pose stopped by a wall.
    assert job.rerun()["status"] == "needs_input"


#: The README's "How it was made" paragraph for a measured room, word for word as it has always been.
MEASURED_MADE = ("Built from one phone video of the site. Sizes, distances and positions are estimates from that video, "
                 "not measurements. A result in this simulation is not a promise of what a robot does at the site, "
                 "and says nothing about safety. No robot is included: bring your own.")
#: The same paragraph for a room whose scale was not measured: shapes from the video, sizes relative, not metres.
UNSCALED_MADE = ("Built from one phone video of the site. Shapes and proportions come from that video; sizes, "
                 "distances and positions are relative, not in metres. A result in this simulation is not a promise "
                 "of what a robot does at the site, and says nothing about safety. No robot is included: bring your "
                 "own.")


@pytest.mark.parametrize("scale", ["measured", "unscaled"])
def test_the_readme_never_calls_an_unscaled_rooms_sizes_estimates_and_a_measured_room_reads_as_before(
        tmp_path, monkeypatch, scale):
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job

    if scale == "unscaled":
        job, _result = shipped_room(tmp_path, monkeypatch)
    else:
        job, _context, _result = room_job(tmp_path, monkeypatch, family="wiping", description="Wipe the table.")
    _status, files = _published(job)
    readme = files["README.txt"]
    said = _said(readme)
    if scale == "measured":
        assert said.startswith(MEASURED_MADE + " ")
        assert SCALE_NOT_MEASURED_NOTE not in said and "scene.usda is in metres (1 unit is 1 metre)" in readme
    else:
        assert said.startswith(UNSCALED_MADE + " ")
        assert "estimates from that video" not in " ".join(readme.split())
        assert said.count(SCALE_NOT_MEASURED_NOTE) == 1
