"""Any job reaches a company package (owner direction 2026-11-16): the room package.

A navigation job, a job with no task profile, a family job whose runtime is not installed, one whose reading and
object disagree and one with several or soft task objects are no longer held at the scene handoff. Each gets the room
package (``website_scene_handoff.write_room_package``): the room's collision and look and, for navigation with one goal
mask, the goal region with an assumed radius and dwell. The site's own words stay on the job's page (the WebApp redacts
them there) and never enter the package. The record and the README say what is assumed, that the package holds no
rule from Blueprint when it holds none, and where the job's objects are. Real
geometry (``tests.test_website_task_preparation`` and ``tests.test_website_native_appearance``), the real handoff
and the real company builder; only the Atlas lane, the mask binding and the WebApp are stand-ins.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation, retains_room_package
from blueprint_pipeline.website_task_preparation import MULTI_OBJECT_GAP

ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"
NAVIGATE = "Walk to the tray and stop by it."
CUPS = [{"target_id": f"cup-{i}", "semantic_label": "cup", "task_effect": "manipulated", "disposition": "remove"}
        for i in (1, 2)]


def room_job(tmp_path: Path, patches, *, family: str = "navigation", description: str = NAVIGATE,
             removal=(), goal: bool = True, all_targets: bool = False, route: str | None = None,
             articulated: bool = False, second_goal: bool = False, not_found: bool = False,
             job_object_rule=None, clean_plate_flag: str = "task_relevant_target_not_observed", job_objects_kept=None):
    """One website job through the real handoff onto the company builder's own test capture (``Job``).

    The capture's team scene status is removed: no half is compiled for a room package; the builder reads the
    handoff's. ``patches`` is a ``pytest.MonkeyPatch``. Returns the job, its task context and the handoff result.
    """
    from tests.test_website_company_simulation import CAPTURE, REQUEST, SITE, Job
    from tests.test_website_native_appearance import inputs

    job = Job(tmp_path / "download")
    # Atlas keeps its levelled source under the capture. The company package only reads look sources there.
    source_root = job.capture / "pipeline" / "website_scene_preparation" / "atlas" / "level"
    source_root.mkdir(parents=True)
    args, _preparation, _data = inputs(source_root)
    for status in (job.queue / "status").glob("*.json"):
        status.unlink()
    context = {**{k: v for k, v in args["task_context"].items() if k != "context_digest"}, "request_id": REQUEST,
               "scene_id": SITE, "capture_id": CAPTURE, "task_family": family, "description": description}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    drawer = {"semantic_label": "three-drawer cabinet", "articulated_part": "middle drawer",
              "articulation_kind": "prismatic"} if articulated else {}
    masks = {**args["task_masks"], "targets": [{**row, **(drawer if row["target_id"] == "cup-1" else {})}
                                               for row in args["task_masks"]["targets"] if all_targets or (
        goal and row.get("target_role") == "destination")]}
    if second_goal:
        tray = next(row for row in masks["targets"] if row.get("target_role") == "destination")
        masks["targets"].append({**tray, "target_id": "tray-2"})
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    removal_path = job.capture / "pipeline/removal.json"
    write_json(removal_path, {"entries": [dict(row) for row in removal]})
    atlas = {"status": "ready", "base_scene": args["base_scene"], "alignment": "anchored",
             "source_geometry": {**args["source_geometry"], "geometry_backend": "world_labs_atlas"},
             "source_pose_digest": "sha256:" + "1" * 64, "background_pose_digest": "sha256:" + "2" * 64,
             **({"atlas_route": route} if route else {}),
             **({"job_object_rule": job_object_rule} if job_object_rule else {}),
             **({"job_objects_kept": job_objects_kept} if job_objects_kept else {})}
    patches.setattr(ATLAS, lambda **_: atlas)
    patches.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    descriptor = {"capture_id": CAPTURE, "scene_id": SITE, "metadata": {"site_task_context": context}}
    clean_plate = {"privacy_verified": True, "status": "noop", "source_geometry": None, "task_masks": masks,
                   "removal_manifest_path": str(removal_path)}
    if not_found:
        # The clean plate went on with nothing of the task found or outlined (``clean_plate_stage.NOT_FOUND``,
        # ``NOT_LOCATED``): no masks.
        clean_plate.update(task_masks=None, flags=[clean_plate_flag])
    job.rerun = lambda: handoff.prepare_website_scene_handoff(descriptor=descriptor, clean_plate=clean_plate,
                                                              capture_root=job.capture, now=args["now"])
    return job, context, job.rerun()


def _published(job) -> tuple[dict, dict[str, str]]:
    from tests.test_website_company_simulation import World, _package
    published = World(job).publish()
    assert published["status"] == "published", published
    with zipfile.ZipFile(_package(job)) as archive:
        return published, {name: archive.read(name).decode() for name in archive.namelist()
                           if not name.endswith((".usdc", ".ply"))}


def _said(readme: str) -> str:
    """The README's "How it was made" section as one line: its fixed paragraph, then each note in order, without the
    notes' own heading or their bullet marks."""
    section = readme.split("\nHow it was made\n", 1)[1].split("\nTerms\n", 1)[0]
    lines = [line[2:] if line.startswith("- ") else line for line in section.splitlines()
             if line != "Notes on this simulation"]
    return " ".join(" ".join(lines).split())


def _record(result) -> dict:
    return json.loads(Path(result["room_package"]["path"]).read_text())


def test_a_navigation_job_ships_the_room_and_an_assumed_goal_rule(tmp_path, monkeypatch):
    from tests.test_website_company_simulation import _package

    job, context, result = room_job(tmp_path, monkeypatch)
    assert (result["status"], result["blockers"], result["job_disposition"]) == ("room_package_written", [], "accepted")
    assert "native_task_preparation_gap" not in result and not holds_native_task_preparation(result, context)
    assert retains_room_package(result, context)
    record = _record(result)
    assert (record["task_kind"], record["success_rule_defined"], record["success_rule_assumed"]) == (
        "goal_region_navigation", True, True)
    goal = record["goal_region"]
    assert goal["basis"] == "goal_mask_bounds_registered_footprint_centre_at_its_lowest_point"
    assert (goal["radius_m"], goal["dwell_seconds"], goal["radius_and_dwell"]) == (0.5, 2.0, "assumed")
    # The tray's own place as compile lifts it (x, y of ``_grounded_destination``), at the bottom of its bounds.
    assert goal["centre_world_m"] == pytest.approx([2.5, -4.5, 3.5])
    assert record["robot_included"] is False and record["simulator_run"] is False
    assert job.rerun() == result  # a second handoff writes the same package
    _status, members = _published(job)
    assert sorted(members) == ["README.txt", "scene.usda", "success_rule.json"]
    with zipfile.ZipFile(_package(job)) as archive:
        assert "scene/room.ply" in archive.namelist()
    rule = json.loads(members["success_rule.json"])
    assert rule["criteria"] == {"surface_target": {"surface_position_world_m": goal["centre_world_m"], "radius_m": 0.5},
                                "settling": {"stable_seconds": 2.0}}
    assert all(NAVIGATE not in text for text in members.values())  # the site's words stay on the job's page
    said = _said(members["README.txt"])
    assert said.endswith(f"{sim.ROOM_NOT_RUN} {sim.GOAL_ESTIMATED}")
    assert sim.LOOK_AS_FIRST_WRITTEN not in said
    # The centre is the goal mask's lowest point as filmed (a tray on a counter: the counter top), and says so.
    files = " ".join(members["README.txt"].split())
    assert ("radius_m around surface_position_world_m, the middle of the goal's footprint at the lowest point of "
            "it the video shows" in files and "on the floor" not in files)
    assert sim.NO_RULE not in said and sim.NO_SIMULATOR not in said and "/World/Object" not in members["scene.usda"]


@pytest.mark.parametrize("masks", ["none", "two"])
def test_a_navigation_job_without_one_goal_mask_ships_the_room_with_no_rule(tmp_path, monkeypatch, masks):
    job, _context, result = room_job(tmp_path, monkeypatch, goal=masks == "two", second_goal=masks == "two")
    record = _record(result)
    assert (record["success_rule_defined"], record["success_rule_assumed"]) == (False, False)
    assert record["goal_region"] == {"basis": "goal_not_placed:no_single_goal_mask"}
    _status, members = _published(job)
    assert "success_rule.json" not in members
    assert _said(members["README.txt"]).endswith(f"{sim.ROOM_NOT_RUN} {sim.NO_RULE}")


@pytest.mark.parametrize(("family", "removal", "route", "articulated", "profile", "flags", "objects"), [
    ("restock_shelves", CUPS, None, False, "none", None, sim.OBJECTS_TAKEN_OUT.format(count=2)),
    ("restock_shelves", CUPS, "geometry_only", False, "none", None, sim.OBJECTS_IN_SCAN),
    ("rigid_pick_place", CUPS, None, False, "rigid_pick_place", [MULTI_OBJECT_GAP],
     sim.OBJECTS_TAKEN_OUT.format(count=2)),
    ("wiping", (), None, False, "wiping", [handoff.RUNTIME_GAP], None),
    ("rigid_pick_place", CUPS[:1], None, True, "rigid_pick_place", [handoff.MISMATCH],
     sim.OBJECTS_TAKEN_OUT_ONE)],
    ids=["unknown_kind", "unknown_kind_scene_as_filmed", "several_objects", "family_without_runtime",
         "profile_mechanism_mismatch"])
def test_a_job_blueprint_has_no_rule_for_ships_the_room_and_says_so(tmp_path, monkeypatch, family, removal, route,
                                                                    articulated, profile, flags, objects):
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)  # no family runtime installed on this host
    job, _context, result = room_job(tmp_path, monkeypatch, family=family, description="Restock the shelves.",
                                     removal=removal, all_targets=True, route=route, articulated=articulated)
    assert (result["status"], result["job_disposition"]) == ("room_package_written", "accepted")
    assert "website_intake_outbox" not in result and "source_registration" not in result  # nothing dispatched
    record = _record(result)
    assert (record["task_kind"], record["task_profile"], record["success_rule_defined"]) == (
        "not_defined_by_blueprint", profile, False)
    assert "goal_region" not in record and record.get("flags") == flags
    _status, members = _published(job)
    assert sorted(members) == ["README.txt", "scene.usda"]
    assert all("Restock" not in text for text in members.values())
    assert _said(members["README.txt"]).endswith(" ".join([sim.ROOM_NOT_RUN, sim.NO_RULE, *([objects] if objects else [])]))


def test_the_site_words_never_enter_the_package_and_a_tampered_record_is_no_source(tmp_path, monkeypatch):
    from tests.test_website_company_simulation import STREET, _site_records
    job, _context, result = room_job(tmp_path, monkeypatch, description=f"Walk to the tray at {STREET}.")
    _site_records(job)
    _status, members = _published(job)  # the description names the site's street: it is not shipped, so no refusal
    assert all(STREET not in text and "Quayside" not in text for text in members.values())
    record_path = Path(result["room_package"]["path"])
    record = json.loads(record_path.read_text())
    assert all(STREET not in json.dumps(value) for value in record.values())
    record_path.write_text(json.dumps({**record, "success_rule_assumed": False}))
    assert sim.simulation_source(job.capture, queue_root=job.queue) is None


def test_the_room_package_meets_every_later_reader_and_a_second_pass(tmp_path, monkeypatch):
    """Later ticks read what this one wrote: each reader accepts the room package, or names why it is not its."""
    from blueprint_pipeline import website_capture_withdrawal as withdrawal
    from blueprint_pipeline import website_held_handoff_replay as replay
    from blueprint_pipeline import website_preparation_renewal as renewal
    from blueprint_pipeline import website_reconstruction_resume as resume
    from tests.test_website_company_simulation import World
    job, context, result = room_job(tmp_path, monkeypatch)
    pipeline = job.capture / "pipeline"
    assert job.rerun() == result  # the handoff again: the same record, the same package
    world = World(job)
    assert world.publish()["status"] == "published" and world.publish()["status"] == "current"
    # Not a held job: the held-handoff drain keeps no bytes and has no fact for it; the reconstruction drain
    # arms only on an Atlas wait, which a written room package is not.
    assert replay.held_kind(pipeline) is None and replay.arrived_fact(pipeline) is None
    assert replay.replay_record(None, payload=b"", payload_digest="sha256:" + "0" * 64, producer_delivery_key="k",
                                capture_root=job.capture) is None
    assert resume.resume_record(None, exc=ValueError("website_room_package_written"), payload=b"",
                                payload_digest="sha256:" + "0" * 64, producer_delivery_key="k",
                                capture_root=job.capture) is None
    # Renewal re-issues an intake under a renewed grant: a room package has none, so it refuses by name.
    with pytest.raises(ValueError) as refused:
        renewal.reissue_renewed_preparation(capture_root=job.capture, queue_root=tmp_path / "intake", now=0)
    assert str(refused.value) == "website_preparation_renewal_handoff_not_completed"
    assert renewal.REFUSAL.fullmatch(str(refused.value))
    # Withdrawal finds and deletes the room package with the rest of the site's files.
    withdrawal.acknowledge_withdrawal(capture_root=job.capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": context["request_id"],
        "scene_id": context["scene_id"], "capture_id": context["capture_id"], "withdrawal_id": "withdrawal-room",
        "requested_at_iso": "2026-11-16T00:00:00Z"})
    planned = {row["path"] for row in withdrawal.plan_local_cleanup(capture_root=job.capture)["files"]}
    record = Path(result["room_package"]["path"])
    room = [path for path in record.parent.rglob("*") if path.is_file()]
    assert room and all(any(str(path).endswith(entry) for entry in planned) for path in room)


def test_the_capture_stage_completes_on_a_room_package(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    job, context, result = room_job(tmp_path, monkeypatch)
    monkeypatch.setattr(handoff, "prepare_website_scene_handoff", lambda **_: result)
    descriptor = SimpleNamespace(metadata={"site_task_context": context}, requested_outputs=["preview_simulation"],
                                 to_dict=lambda: {"capture_id": context["capture_id"], "metadata": {}})
    value = run_website_capture_preparation(
        descriptor=descriptor, capture_root=job.capture, pipeline_dir=job.capture / "pipeline", source_video=None,
        derived_scene_generation_allowed=True, gates=[], make_gate=lambda *_: None,
        prepare_media=lambda **_: {"status": "noop", "privacy_status": "verified", "privacy_verified": True},
        check_privacy=lambda **_: SimpleNamespace(passed=True))
    assert value["status"] == "completed" and value["website_scene_preparation"] == result


def test_a_tampered_handoff_names_no_room_package(tmp_path, monkeypatch):
    job, _context, result = room_job(tmp_path, monkeypatch)
    path = job.capture / "pipeline/website_scene_preparation/handoff.json"
    # A field no other check reads: only the seal refuses it.
    path.write_text(json.dumps({**result, "geometry_backend": "another_backend"}))  # not resealed
    assert sim.simulation_source(job.capture, queue_root=job.queue) is None
    path.write_text(json.dumps(result))
    assert sim.simulation_source(job.capture, queue_root=job.queue)["room"] is not None


@pytest.mark.parametrize("change", ["no_rule", "no_object"])
def test_a_rigid_half_never_ships_without_its_rule_or_its_object(tmp_path, change):
    """Only a real room package (its own plan, no object of its own) may ship with no rule and no /World/Object;
    a rigid half that records itself as a room is refused all the same."""
    from tests.test_website_company_simulation import Job, World, _write
    from tests.test_website_scene_package_robot_absent import _sealed_doc
    job = Job(tmp_path)
    plan_path = job.packet / sim.PLAN_FILE
    plan = json.loads(plan_path.read_text())
    if change == "no_rule":
        plan["task_spec"].pop("task_success_contract")
    else:
        plan["objects"] = [row for row in plan["objects"] if row["semantic_role"] != "task_object"]
    _write(plan_path, plan)
    _write(job.scene / "team_scene_inputs.v1.json", _sealed_doc(
        {"schema_version": sim.ROOM_RECORD_SCHEMA, "success_rule_defined": False}, "inputs_digest"))
    result = World(job).publish()
    assert (result["status"], result["reason"]) == ("refused", "website_company_simulation_room_record_mismatch")


@pytest.mark.parametrize("change", ["record_says_no_rule", "record_names_another_plan", "object_row_in_the_room"])
def test_a_room_package_that_contradicts_its_record_is_refused(tmp_path, monkeypatch, change):
    """Each kept consistent and resealed, so only the cross-check can refuse it: the record names the plan beside
    it, says there is a rule exactly when the plan has one, and a room package places no object of its own."""
    from blueprint_pipeline.team_scene_robot_absent import DIRECTORY, PLAN_FILENAME
    from tests.test_website_company_simulation import World
    job, _context, result = room_job(tmp_path, monkeypatch)
    record_path = Path(result["room_package"]["path"])
    record = json.loads(record_path.read_text())
    plan_path = record_path.parent / DIRECTORY / PLAN_FILENAME
    plan = json.loads(plan_path.read_text())
    if change == "record_says_no_rule":
        record.update(success_rule_defined=False, success_rule_assumed=False)
    elif change == "record_names_another_plan":
        record["plan_digest"] = "sha256:" + "e" * 64
    else:
        collision = next(row for row in plan["objects"] if row["semantic_role"] == "scene_collision")
        plan["objects"].append({**collision, "name": "task_object", "semantic_role": "task_object", "visible": True})
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        plan_path.write_text(json.dumps(plan))
        record["plan_digest"] = plan["plan_digest"]
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    record_path.write_text(json.dumps(record))
    handoff_path = job.capture / "pipeline/website_scene_preparation/handoff.json"
    value = {**result, "room_package": {**result["room_package"], "inputs_digest": record["inputs_digest"]}}
    value["digest"] = canonical_digest(value, digest_field="digest")
    handoff_path.write_text(json.dumps(value))
    published = World(job).publish()
    assert (published["status"], published["reason"]) == ("refused", "website_company_simulation_room_record_mismatch")


def test_a_job_that_brings_no_object_says_nothing_was_taken_out_of_the_room_scan(tmp_path, monkeypatch):
    # Job itest-7c42 (2026-11-16, wiping): the clean plate took nothing, so the Atlas planner's two
    # removals were not applied and the room is as filmed (``website_atlas_pipeline.job_object_review``).
    from blueprint_pipeline.atlas_automation_plan import JOB_OBJECT_RULE
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    rule = {"code": JOB_OBJECT_RULE, "planner_remove_count": 2, "removed_count": 0}
    job, _context, result = room_job(tmp_path, monkeypatch, family="wiping", description="Wipe the table.",
                                     all_targets=True, job_object_rule=rule)
    record = _record(result)
    assert record["flags"] == [handoff.RUNTIME_GAP, JOB_OBJECT_RULE] and record["job_objects"] is None
    _status, members = _published(job)
    assert _said(members["README.txt"]).endswith(" ".join([sim.ROOM_NOT_RUN, sim.NO_RULE, sim.NOTHING_TAKEN_OUT]))


def test_a_job_with_both_flags_reads_as_two_plain_sentences_in_order(tmp_path, monkeypatch):
    # Gap A (the analysis found nothing) and the job-object rule on one job: no contradiction, nothing twice.
    from blueprint_pipeline.atlas_automation_plan import JOB_OBJECT_RULE
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    rule = {"code": JOB_OBJECT_RULE, "planner_remove_count": 2, "removed_count": 0}
    job, _context, result = room_job(tmp_path, monkeypatch, family="wiping", description="Wipe the table.",
                                     all_targets=True, not_found=True, job_object_rule=rule)
    assert _record(result)["flags"][-2:] == [handoff.NOT_FOUND, JOB_OBJECT_RULE]  # caller's flags first
    _status, members = _published(job)
    said = _said(members["README.txt"])
    assert said.endswith(" ".join([sim.NOT_FOUND, sim.NOTHING_TAKEN_OUT]))
    assert said.count("taken out of the room scan") == 1
