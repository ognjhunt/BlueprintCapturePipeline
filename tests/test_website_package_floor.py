"""The package floor: a routed job's room ships once its object path stops, and gives way to the half.

The job's own scene intent decides (``website_company_simulation.object_path_stopped``); the floor record is the
real room writer's (``website_scene_handoff.write_room_package``) on the shared appearance fixture, and the company
builder is the real one on the package tests' capture (``tests/test_website_company_simulation``).
"""
# Covers: src/blueprint_pipeline/website_company_simulation.py
# Covers: src/blueprint_pipeline/website_scene_handoff.py
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_outcome import REASONS, outcome_record
from tests.test_website_company_simulation import (
    CAPTURE, NOW, OLDER, REQUEST, SITE, Job, World, _at, _manifest, _owned_job, _package, _withdraw, _write)

INTENT = "scene-" + "f" * 64
REQUEST_BODY = {"task": {"task_id": "website-floor", "strategy": "pick_and_place"},
                "execution": {"expires_at_epoch": NOW - 60}}
DAY = 24 * 3600


class Floor:
    """A capture whose handoff is ``intake_ready`` with its room floor written, and its own scene intent."""

    def __init__(self, tmp_path: Path, monkeypatch, *, job: Job | None = None, half: bool = False):
        from tests.test_website_native_appearance import inputs
        self.job = job or Job(tmp_path / "download")
        if not half:
            for status in (self.job.queue / "status").glob("*.json"):
                status.unlink()
        (tmp_path / "fixture").mkdir(parents=True)
        args, _preparation, _data = inputs(tmp_path / "fixture")
        context = {**{k: v for k, v in args["task_context"].items() if k != "context_digest"},
                   "request_id": REQUEST, "scene_id": SITE, "capture_id": CAPTURE}
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
        self.workspace = self.job.capture / "pipeline/website_scene_preparation"
        self.context, self.args = context, args
        self.write = lambda now=NOW: handoff.write_room_floor(
            now=now, task_context=context, base_scene=args["base_scene"], source_geometry=args["source_geometry"],
            removal={"entries": [{"target_id": "cup-1", "task_effect": "manipulated"}]},
            output_root=self.workspace / handoff.ROOM_FLOOR, task_profile=None, flags=[handoff.ROOM_FLOOR_FLAG])
        record = self.write()
        self.record = self.workspace / handoff.ROOM_FLOOR / "half" / handoff.ROOM_RECORD
        value = {"schema_version": "website_scene_handoff.v1", "status": "intake_ready", "blockers": [],
                 "capture_id": CAPTURE, "scene_id": SITE,
                 "room_floor": {"path": str(self.record), "inputs_digest": record["inputs_digest"]},
                 "website_intake_outbox": {"id": INTENT, "state": "forward_pending",
                                           "request_digest": cross_runtime_canonical_digest(REQUEST_BODY)}}
        value["digest"] = canonical_digest(value, digest_field="digest")
        _write(self.workspace / "handoff.json", value)
        self.intents = tmp_path / "intents"
        monkeypatch.setattr(sim, "INTENT_ROOT", self.intents)
        self.directory = self.intents / INTENT
        self.intent = {"intent_id": INTENT, "intent_digest": "sha256:" + "1" * 64, "request": REQUEST_BODY}
        self.age(0)

    def age(self, seconds: int) -> None:
        """The floor record first written ``seconds`` before the builder's clock."""
        first = self.workspace / handoff.ROOM_FLOOR / handoff.ROOM_FLOOR_FIRST_WRITTEN
        _write(first, {**json.loads(first.read_text()), "first_written_epoch": NOW - seconds})

    def started(self, *, progression: dict | None = None, reason: str | None = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _write(self.directory / "intent.json", self.intent)
        _write(self.directory / "progression.json", progression or {"status": "running", "phase": "preparation",
                                                                   "blockers": []})
        if reason is not None:
            kind, terminal, category = REASONS[reason]
            _write(self.directory / "outcome.json", outcome_record(self.directory, self.intent, {
                "kind": kind, "terminal": terminal, "reason_category": category, "reason_code": reason}, now=NOW))

    def completed(self, team: dict) -> None:
        self.started(progression={"status": "completed", "phase": "scene_prepared", "blockers": [],
                                  "state": {"scene_preparation_completion": {"team_scene_request": team}}})


def _readme(job: Job) -> str:
    with zipfile.ZipFile(_package(job)) as archive:
        return " ".join(archive.read("README.txt").decode().split())


# --- when the floor ships -----------------------------------------------------------------------------------------

STOPPED = {
    "spend_cap": lambda floor: floor.started(reason="spend_cap_reached"),
    "authority_expired": lambda floor: floor.started(reason="authority_expired"),
    "retry_cap": lambda floor: floor.started(reason="retry_cap_exhausted"),
    "held": lambda floor: floor.started(reason="task_family_unsupported"),
    "task_brief": lambda floor: floor.started(reason="task_brief_required"),
    "pipeline_fault": lambda floor: floor.started(reason="pipeline_fault"),
    "no_half_route": lambda floor: floor.completed({"status": "not_configured"}),
    "probe_failed": lambda floor: floor.completed({"status": "not_enqueued", "reason": "robot_free_probe_failed"}),
}
REASON = {"spend_cap": "spend_cap_reached", "authority_expired": "authority_expired",
          "retry_cap": "retry_cap_exhausted", "held": "task_family_unsupported", "task_brief": "task_brief_required",
          "pipeline_fault": "pipeline_fault", "no_half_route": "not_configured",
          "probe_failed": "robot_free_probe_failed"}


@pytest.mark.parametrize("state", sorted(STOPPED))
def test_a_stopped_object_path_ships_the_room_floor_once(tmp_path, monkeypatch, state):
    floor = Floor(tmp_path, monkeypatch)
    STOPPED[state](floor)
    world = World(floor.job)
    first = world.publish()
    assert first["status"] == "published", first
    manifest = _manifest(floor.job)
    assert manifest["source_kind"] == "floor"
    assert manifest["floor_reason"] == "object_not_built:" + REASON[state]
    readme = _readme(floor.job)
    assert sim.OBJECTS_TAKEN_OUT.format(count=1) in readme and sim.NO_RULE in readme
    assert world.publish()["status"] == "current" and len(world.delivered) == 1


MOVING = {
    "running": lambda floor: floor.started(),
    "intent_not_yet_made": lambda floor: floor.age(DAY - 60),
    # Queued for a day: ``enqueue_team_scene_request`` writes only ``pending/``; the worker writes its status when
    # it claims the request.
    "half_queued": lambda floor: (floor.completed({"status": "enqueued", "request_key": "k" * 32}),
                                  _write(floor.job.queue / "pending" / f"{'k' * 32}.json", {}),
                                  os.utime(floor.directory / "progression.json", (NOW - DAY, NOW - DAY))),
    "half_claimed": lambda floor: (floor.completed({"status": "enqueued", "request_key": "k" * 32}),
                                   _write(floor.job.queue / "status" / f"{'k' * 32}.json", {"status": "processing"})),
    "half_retry_armed": lambda floor: (floor.completed({"status": "enqueued", "request_key": "k" * 32}),
                                       _write(floor.job.queue / "status" / f"{'k' * 32}.json",
                                              {"status": "blocked", "blockers": ["team_scene_packet_failed:x"]})),
    "withdrawn": lambda floor: floor.started(reason="authority_revoked"),
    "site_consent": lambda floor: floor.started(reason="site_consent_required"),
}


@pytest.mark.parametrize("state", sorted(MOVING))
def test_an_object_path_still_moving_or_ended_by_rights_ships_nothing(tmp_path, monkeypatch, state):
    floor = Floor(tmp_path, monkeypatch)
    MOVING[state](floor)
    world = World(floor.job)
    assert world.publish() == {"status": "no_source"}
    assert world.delivered == [] and not _package(floor.job).exists()


UNCLEAR = {
    "intent_missing": lambda floor: floor.age(DAY),
    "half_not_enqueued": lambda floor: (floor.completed({"status": "enqueued", "request_key": "k" * 32}),
                                        os.utime(floor.directory / "progression.json", (NOW - DAY, NOW - DAY))),
    "team_scene_blocked_unheld": lambda floor: (
        floor.completed({"status": "enqueued", "request_key": "k" * 32}),
        _write(floor.job.queue / "status" / f"{'k' * 32}.json", {"status": "blocked", "blockers": ["x"]}),
        os.utime(floor.job.queue / "status" / f"{'k' * 32}.json", (NOW - DAY, NOW - DAY))),
}


@pytest.mark.parametrize("state", sorted(UNCLEAR))
def test_a_path_we_cannot_tell_is_stopped_is_listed_after_a_day_and_never_ships(tmp_path, monkeypatch, state):
    """No intent made, a half neither queued nor started, a block the scene stage never recorded: the floor
    ships only for a reason that can be named, so these are listed (``floor_unclassified``) and nothing ships."""
    floor = Floor(tmp_path, monkeypatch)
    monkeypatch.setattr(sim, "SCREENING_ROOT", tmp_path / "no-screening")
    UNCLEAR[state](floor)
    world = World(floor.job)
    assert world.publish() == {"status": "no_source", "reason": "floor_unclassified:" + state}
    assert world.delivered == []


def test_a_handoff_that_runs_again_keeps_the_floors_first_writing(tmp_path, monkeypatch):
    """Written at hour 0, written again at hour 23 (a release, a replay): its wait ends at hour 24, not 47."""
    floor = Floor(tmp_path, monkeypatch)
    first = floor.workspace / handoff.ROOM_FLOOR / handoff.ROOM_FLOOR_FIRST_WRITTEN
    first.unlink()  # the floor as its first handoff writes it, a day ago
    floor.write(NOW - DAY)
    floor.write(NOW - 3600)
    os.utime(floor.record, (NOW - 3600, NOW - 3600))
    assert json.loads(first.read_text())["first_written_epoch"] == NOW - DAY
    assert World(floor.job).publish() == {"status": "no_source", "reason": "floor_unclassified:intent_missing"}
    assert sim.FLOOR_FIRST_WRITTEN == handoff.ROOM_FLOOR_FIRST_WRITTEN


@pytest.mark.parametrize("fault", ["unbound", "outcome_invalid"])
def test_a_state_that_cannot_be_read_or_bound_ships_nothing_and_says_so(tmp_path, monkeypatch, fault):
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    if fault == "unbound":  # another job's request under this intent id
        _write(floor.directory / "intent.json", {**floor.intent, "request": {"task": {"task_id": "website-other"}}})
    else:
        _write(floor.directory / "outcome.json", {"reason_code": "spend_cap_reached"})
    result = World(floor.job).publish()
    assert result["status"] == "no_source"
    assert result["reason"] == {"unbound": "floor_unclassified:intent_unbound",
                                "outcome_invalid": "floor_unclassified:scene_outcome_record_invalid"}[fault]


def test_a_final_team_scene_block_ships_the_floor_and_a_new_release_changes_nothing_delivered(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch)
    floor.completed({"status": "enqueued", "request_key": "k" * 32})
    _write(floor.job.queue / "status" / f"{'k' * 32}.json",
           {"status": "blocked", "blockers": ["team_scene_robot_access_measurement_held:x"]})
    _hold(floor, monkeypatch, "held")  # the scene stage's pass found the block final and told the website
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    assert _manifest(floor.job)["floor_reason"] == "object_not_built:team_scene_packet_blocked"
    _hold(floor, monkeypatch, "released")  # a new release tries the half again: nothing new, nothing taken back
    assert world.publish() == {"status": "no_source"}
    assert world.retracted == [] and len(world.delivered) == 1
    _hold(floor, monkeypatch, "held")
    assert world.publish()["status"] == "current" and len(world.delivered) == 1


def _hold(floor: Floor, monkeypatch, status: str, intent: str = INTENT) -> None:
    """The scene stage's hold for the intent, as ``scene_stage_holds.settle_hold`` keeps it."""
    from blueprint_pipeline import scene_stage_holds as holds
    root = floor.job.tmp / "screening"
    monkeypatch.setattr(sim, "SCREENING_ROOT", root)
    publication = {"status": status, "reason_code": "team_scene_packet_blocked" if status == "held" else None,
                   "source": {"scene_intent_id": intent}, "hold_digest": ""}
    publication["hold_digest"] = cross_runtime_canonical_digest(publication, digest_field="hold_digest")
    _write(root / holds.HOLDS_DIRNAME / f"{holds.hold_id(intent)}.json", {"publication": publication})


def test_the_hold_store_read_is_the_scene_stages_own(tmp_path, monkeypatch):
    """The builder reads the hold store as a file (it never imports the GPU lane the scene stage reaches)."""
    from blueprint_pipeline import scene_stage_holds as holds
    assert (sim.SCREENING_ROOT_ENV, sim.DEFAULT_SCREENING_ROOT) == (holds.QUEUE_ROOT_ENV, holds.DEFAULT_QUEUE_ROOT)
    floor = Floor(tmp_path, monkeypatch)
    _hold(floor, monkeypatch, "held", intent="scene-" + "e" * 64)  # another intent's hold says nothing here
    floor.completed({"status": "enqueued", "request_key": "k" * 32})
    _write(floor.job.queue / "status" / f"{'k' * 32}.json", {"status": "blocked", "blockers": ["x"]})
    assert World(floor.job).publish() == {"status": "no_source"}


# --- the half, before and after -----------------------------------------------------------------------------------

def test_the_half_arriving_after_the_floor_replaces_it_with_one_revision(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    first = world.publish()
    assert first["status"] == "published"
    floor.job.seal()  # the owner raised the cap: the half compiled
    second = world.publish()
    assert second["status"] == "published" and second["simulation_revision"] != first["simulation_revision"]
    assert _manifest(floor.job)["source_kind"] == "half" and "floor_reason" not in _manifest(floor.job)
    assert len(world.delivered) == 2 and len(world.retracted) == 1
    assert world.publish()["status"] == "current"


def test_a_delivered_half_is_never_replaced_by_a_floor(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    assert _manifest(floor.job)["source_kind"] == "half"
    for status in (floor.job.queue / "status").glob("*.json"):
        status.unlink()  # the half's status is gone: still no floor over the delivered half
    assert world.publish() == {"status": "no_source"}
    assert world.retracted == [] and len(world.delivered) == 1


def test_a_half_refused_for_its_own_shape_gives_way_to_the_floor_and_is_not_rebuilt(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    plan = json.loads((floor.job.packet / sim.PLAN_FILE).read_text())
    plan["task_spec"].pop("task_success_contract")  # no rule anywhere: ``scene_stage_success_rule_missing``
    _write(floor.job.packet / sim.PLAN_FILE, plan)
    _write(floor.job.scene / "team_scene_inputs.v1.json", {})
    world = World(floor.job)
    refused = world.publish()
    assert refused["status"] == "refused" and refused["reason"] == "website_company_simulation_success_rule_missing"
    assert _manifest(floor.job)["refused_half"]["reason"] == refused["reason"]
    builds = []
    real = sim.build_package
    monkeypatch.setattr(sim, "build_package", lambda source, *a, **k: builds.append(source["source_kind"])
                        or real(source, *a, **k))
    assert world.publish()["status"] == "published"
    assert _manifest(floor.job)["floor_reason"] == "object_not_built:website_company_simulation_success_rule_missing"
    assert _at(world, 7 * 3600)["status"] == "current"  # past the 6-hour window: the same half is not built again
    assert builds == ["floor"]


@pytest.mark.parametrize("refusal", ["website_company_simulation_file_names_the_site",
                                     sim.KIND_NOT_SHIPPED, "website_company_simulation_outside_reference"])
def test_a_scan_or_reference_refusal_never_falls_back(tmp_path, monkeypatch, refusal):
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")

    def refuse(source, *_, **__):
        raise sim.PackageRefused(refusal, "x")
    monkeypatch.setattr(sim, "build_package", refuse)
    world = World(floor.job)
    for seconds in (0, 60, 7 * 3600):
        assert _at(world, seconds)["status"] == "refused"
    assert "refused_half" not in _manifest(floor.job) and world.delivered == []


# --- what the floor never gets past ---------------------------------------------------------------------------

def test_takedown_rights_and_withdrawal_stop_the_floor(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    takedown = _write(floor.job.capture / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
    assert world.publish() == {"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"}
    assert len(world.retracted) == 1
    takedown.unlink()
    assert world.publish()["status"] == "published"
    _withdraw(floor.job.capture)  # a withdrawal after the floor takes it back
    assert world.publish()["status"] == "withdrawn"


@pytest.mark.parametrize("rights", ["refused", "not_sharing"])
def test_a_floor_follows_the_live_rights_like_any_package(tmp_path, monkeypatch, rights):
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    world.rights = ({"consent_status": "revoked", "consent_revoked": True, "statement_version": None}
                    if rights == "refused" else {"consent_status": "granted", "consent_revoked": False,
                                                 "statement_version": OLDER})
    result = world.publish()
    assert result["status"] in ({"refused"} if rights == "refused" else {"not_permitted"}), result
    assert world.delivered == []


def test_no_floor_is_built_for_blueprints_own_capture_the_website_would_refuse(tmp_path, monkeypatch):
    """39ec's shape: on Blueprint's list under a statement that does not share. The half follows today's rule."""
    job = _owned_job(tmp_path / "owned", monkeypatch)
    floor = Floor(tmp_path, monkeypatch, job=job)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    world.rights = {"consent_status": "granted", "consent_revoked": False, "statement_version": OLDER}
    builds = []
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: builds.append(1))
    assert world.publish() == {"status": "not_permitted"}
    assert builds == [] and world.delivered == []


# --- the record -----------------------------------------------------------------------------------------------

def test_the_floor_record_is_the_same_on_every_handoff(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch)
    first = json.loads(floor.record.read_text())
    assert floor.write()["inputs_digest"] == first["inputs_digest"]
    assert handoff.ROOM_FLOOR_FLAG == sim.FLOOR_FLAG and first["flags"] == [sim.FLOOR_FLAG]
    assert first["success_rule_defined"] is False and first["job_objects"]["count"] == 1


@pytest.mark.parametrize("tamper", ["another_sealed_record", "resealed_without_its_flag"])
def test_a_floor_record_that_is_not_the_handoffs_floor_never_ships(tmp_path, monkeypatch, tamper):
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    record = json.loads(floor.record.read_text())
    record = {**record, "flags": []} if tamper == "resealed_without_its_flag" else {**record, "task_profile": "x"}
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    if tamper == "resealed_without_its_flag":  # a room package record, handed in as the floor
        value = json.loads((floor.workspace / "handoff.json").read_text())
        value["room_floor"]["inputs_digest"] = record["inputs_digest"]
        value["digest"] = canonical_digest(value, digest_field="digest")
        _write(floor.workspace / "handoff.json", value)
    _write(floor.record, record)
    assert World(floor.job).publish() == {"status": "no_source"}


def test_a_floor_reads_its_object_out_of_the_scan_in_the_pinned_room_order(tmp_path, monkeypatch):
    """A floor's README: no rule, then its object taken out of the scan, then the scene source's notes, in the
    room package's order (``tests/test_website_scene_completion_floor``): the floor adds no sentence of its own."""
    floor = Floor(tmp_path, monkeypatch)
    record = handoff.write_room_floor(
        now=NOW, task_context=json.loads(json.dumps(floor.context)), base_scene=floor.args["base_scene"],
        source_geometry=floor.args["source_geometry"],
        removal={"entries": [{"target_id": "cup-1", "task_effect": "manipulated"}]},
        output_root=floor.workspace / handoff.ROOM_FLOOR, task_profile=None,
        flags=[handoff.ROOM_FLOOR_FLAG, "scene_completion_not_admitted"])
    value = json.loads((floor.workspace / "handoff.json").read_text())
    value["room_floor"]["inputs_digest"] = record["inputs_digest"]
    value["digest"] = canonical_digest(value, digest_field="digest")
    _write(floor.workspace / "handoff.json", value)
    floor.started(reason="spend_cap_reached")
    assert World(floor.job).publish()["status"] == "published"
    readme = _readme(floor.job)
    said = [sim.NO_RULE, sim.OBJECTS_TAKEN_OUT.format(count=1), sim.NOT_FILLED_IN]
    assert readme.index(said[0]) < readme.index(said[1]) < readme.index(said[2])
    assert readme.count("taken out of the room scan") == 1


# --- v3: what is out keeps following its rights; the checks that keep the floor this capture's own -----------

REVOKED = {"consent_status": "revoked", "consent_revoked": True, "statement_version": None}


@pytest.mark.parametrize("moves", ["half_requeued", "money_granted"])
def test_a_served_floor_whose_path_moves_again_still_follows_revoked_rights(tmp_path, monkeypatch, moves):
    """A release releases the hold, or the owner grants money: nothing to build, and the floor out is still
    taken back when the site's consent is revoked (the review's probe, passes at 7 h, 13 h, 30 h and 72 h)."""
    floor = Floor(tmp_path, monkeypatch)
    if moves == "half_requeued":
        floor.completed({"status": "enqueued", "request_key": "k" * 32})
        _write(floor.job.queue / "status" / f"{'k' * 32}.json", {"status": "blocked", "blockers": ["x:y"]})
        _hold(floor, monkeypatch, "held")
    else:
        floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    if moves == "half_requeued":
        _hold(floor, monkeypatch, "released")
    else:
        (floor.directory / "outcome.json").unlink()  # the row moves again: its outcome is removed
    assert _at(world, 3600) == {"status": "no_source"}  # rights still hold: it stays out
    assert world.retracted == []
    world.rights = REVOKED
    assert _at(world, 7 * 3600)["status"] == "refused"
    assert len(world.retracted) == 1
    for hours in (13, 30, 72):
        assert _at(world, hours * 3600) == {"status": "no_source"}
    assert len(world.retracted) == 1


def test_a_served_floor_is_taken_back_when_no_basis_is_left_for_its_replacement(tmp_path, monkeypatch):
    """The half arrives, and the fresh live read for its build says the site no longer shares (Blueprint has no
    list entry): nothing is built and the floor out goes."""
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    floor.job.seal()  # the owner raised the cap: the half compiled
    world.rights = {"consent_status": "granted", "consent_revoked": False, "statement_version": OLDER}
    assert world.publish() == {"status": "not_permitted"}
    assert len(world.retracted) == 1 and len(world.delivered) == 1


@pytest.mark.parametrize("tamper", ["handoff_unsealed", "another_captures_floor"])
def test_a_floor_the_handoff_does_not_seal_inside_its_capture_never_ships(tmp_path, monkeypatch, tamper):
    floor = Floor(tmp_path, monkeypatch)
    floor.started(reason="spend_cap_reached")
    value = json.loads((floor.workspace / "handoff.json").read_text())
    if tamper == "handoff_unsealed":
        value["digest"] = "sha256:" + "0" * 64
    else:  # another site's floor record, copied outside this capture, named by this capture's handoff
        other = tmp_path / "other-site" / "room_floor"
        import shutil
        shutil.copytree(floor.workspace / handoff.ROOM_FLOOR, other)
        value["room_floor"]["path"] = str(other / "half" / handoff.ROOM_RECORD)
        value["digest"] = canonical_digest(value, digest_field="digest")
    _write(floor.workspace / "handoff.json", value)
    assert World(floor.job).publish() == {"status": "no_source"}


def test_a_delivered_half_whose_status_went_still_follows_revoked_rights(tmp_path, monkeypatch):
    """The older case the same lines close: a delivered half whose team-scene status disappeared."""
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    world = World(floor.job)
    assert world.publish()["status"] == "published"
    for status in (floor.job.queue / "status").glob("*.json"):
        status.unlink()
    world.rights = REVOKED
    assert _at(world, 7 * 3600)["status"] == "refused"
    assert len(world.retracted) == 1


def test_a_hold_that_does_not_seal_itself_holds_nothing(tmp_path, monkeypatch):
    """A hand-edited or torn hold file reads as no hold: the half counts as moving (and is listed after a day)."""
    from blueprint_pipeline import scene_stage_holds as holds
    floor = Floor(tmp_path, monkeypatch)
    floor.completed({"status": "enqueued", "request_key": "k" * 32})
    _write(floor.job.queue / "status" / f"{'k' * 32}.json", {"status": "blocked", "blockers": ["x"]})
    _hold(floor, monkeypatch, "held")
    path = floor.job.tmp / "screening" / holds.HOLDS_DIRNAME / f"{holds.hold_id(INTENT)}.json"
    record = json.loads(path.read_text())
    record["publication"]["reason_code"] = "edited_by_hand"
    _write(path, record)
    assert World(floor.job).publish() == {"status": "no_source"}


def test_a_floor_the_pass_cannot_decide_is_named_in_its_journal_and_counted(tmp_path, monkeypatch, caplog):
    """The review's round-2 probe: a listed floor reason reaches the operator, by capture id and code only, under
    its own count; a plain ``no_source`` stays a bare count."""
    import logging
    floor = Floor(tmp_path, monkeypatch)
    floor.age(2 * DAY)  # the intent was never made: listed ``floor_unclassified:intent_missing``
    world = World(floor.job)
    options = {"now": lambda: NOW, "check_rights": world.check_rights, "archive": world.archive,
               "retract": world.retract, "deliver": world.deliver, "queue_root": floor.job.queue}
    caplog.set_level(logging.WARNING, logger=sim.logger.name)
    assert sim.tick(floor.job.capture.parents[4], **options) == {"floor_unclassified": 1}
    said = [record.getMessage() for record in caplog.records]
    assert said == [f"website_company_simulation_outcome capture={CAPTURE} status=no_source "
                    "reason=floor_unclassified:intent_missing"]
    caplog.clear()
    floor.age(60)  # within the day: still waiting, a bare count, nothing named
    assert sim.tick(floor.job.capture.parents[4], **options) == {"no_source": 1}
    assert caplog.records == []


# --- a half that times out never costs the job its floor -----------------------------------------------------------

def _half_times_out(monkeypatch) -> list[str]:
    builds: list[str] = []
    real = sim.build_package

    def build(source, *a, **k):
        builds.append(source["source_kind"])
        if source["source_kind"] == "half":
            raise ValueError(sim.BUILD_TIMED_OUT)
        return real(source, *a, **k)
    monkeypatch.setattr(sim, "build_package", build)
    return builds


def test_an_old_rule_half_refusal_whose_rebuild_times_out_keeps_the_floor(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    real = sim.build_package
    monkeypatch.setattr(sim, "build_package", lambda source, *a, **k: (_ for _ in ()).throw(ValueError(sim.OVER_CAP))
                        if source["source_kind"] == "half" else real(source, *a, **k))
    world = World(floor.job)
    assert world.publish() == {"status": "over_size_cap"}
    assert world.publish()["status"] == "published" and _manifest(floor.job)["source_kind"] == "floor"
    # As the earlier builder left it: the half refused for its files' size, the floor archived but not delivered.
    manifest = _manifest(floor.job)
    manifest["refused_half"].pop("size_rule")
    manifest["delivered"] = None
    _write(floor.job.capture.joinpath(*sim.DIRECTORY) / sim.MANIFEST_FILE, manifest)
    floor_digest = manifest["package"]["digest"]
    builds = _half_times_out(monkeypatch)
    assert world.publish() == {"status": "build_timed_out"} and builds == ["half"]
    kept = _manifest(floor.job)
    assert kept["source_kind"] == "floor" and kept["package"]["digest"] == floor_digest
    assert kept["refused_half"]["reason"] == sim.BUILD_TIMED_OUT and kept["refused_half"]["size_rule"] == sim.SIZE_RULE
    assert _package(floor.job).is_file() and world.retracted == [] and not kept["archive"].get("retraction")
    assert world.publish()["status"] == "published" and builds == ["half"]  # the floor goes out; no half build
    # The next window tries the half again; it times out again and the floor stays.
    assert _at(world, 7 * 3600) == {"status": "build_timed_out"} and builds == ["half", "half"]
    assert _manifest(floor.job)["package"]["digest"] == floor_digest and world.retracted == []


def test_a_large_half_that_times_out_ends_on_the_floor(tmp_path, monkeypatch):
    from tests.test_website_company_simulation import _large_look
    floor = Floor(tmp_path, monkeypatch, job=_large_look(tmp_path / "download"), half=True)
    floor.started(reason="spend_cap_reached")
    half = sim.simulation_source(floor.job.capture, queue_root=floor.job.queue)
    large = sum(row["size_bytes"] for row in half["files"].values()) - 1  # its files are over the limit given
    builds = _half_times_out(monkeypatch)
    world = World(floor.job)
    assert world.publish(max_bytes=large) == {"status": "build_timed_out"}
    assert _manifest(floor.job)["refused_half"]["reason"] == sim.BUILD_TIMED_OUT
    assert world.publish(max_bytes=large)["status"] == "published" and _manifest(floor.job)["source_kind"] == "floor"
    later = sim.publish_company_simulation(floor.job.capture, now=lambda: NOW + 7 * 3600, max_bytes=large,
                                           check_rights=world.check_rights, archive=world.archive,
                                           retract=world.retract, deliver=world.deliver, queue_root=floor.job.queue)
    assert later == {"status": "build_timed_out"} and builds == ["half", "floor", "half"]  # once in the next window
    assert _manifest(floor.job)["source_kind"] == "floor" and _package(floor.job).is_file()


def test_a_floor_kept_through_a_half_build_is_taken_back_if_the_site_withdraws_meanwhile(tmp_path, monkeypatch):
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    builds = _half_times_out(monkeypatch)
    world = World(floor.job)
    world.publish()
    assert world.publish()["status"] == "published"
    manifest = _manifest(floor.job)
    manifest["refused_half"]["simulation_revision"] = "sha256:" + "0" * 64  # a new half revision arrived
    _write(floor.job.capture.joinpath(*sim.DIRECTORY) / sim.MANIFEST_FILE, manifest)
    real = sim.build_package

    def build_then_withdraw(source, *a, **k):
        _withdraw(floor.job.capture)
        return real(source, *a, **k)
    monkeypatch.setattr(sim, "build_package", build_then_withdraw)
    assert world.publish() == {"status": "withdrawn"} and len(world.retracted) == 1
    assert builds == ["half", "floor", "half"] and sim.permission.hidden(_manifest(floor.job)["archive"])


GIVE_WAY = {"time": (ValueError(sim.BUILD_TIMED_OUT), {"status": "build_timed_out"}),
            "size": (ValueError(sim.OVER_CAP), {"status": "over_size_cap"}),
            "shape": (sim.ShapeRefused("website_company_simulation_success_rule_missing"),
                      {"status": "refused", "reason": "website_company_simulation_success_rule_missing",
                       "detail": ""})}


@pytest.mark.parametrize("why", sorted(GIVE_WAY))
def test_a_kept_floor_its_records_no_longer_give_is_taken_back_and_the_half_is_not_rebuilt_each_pass(
        tmp_path, monkeypatch, why):
    error, status = GIVE_WAY[why]
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    builds: list[str] = []
    real = sim.build_package

    def build(source, *a, **k):
        builds.append(source["source_kind"])
        if source["source_kind"] == "half":
            raise error
        return real(source, *a, **k)
    monkeypatch.setattr(sim, "build_package", build)
    world = World(floor.job)
    assert world.publish() == status
    assert world.publish()["status"] == "published" and _manifest(floor.job)["source_kind"] == "floor"
    # The handoff's floor record changes: the published floor is no longer the one the job's records give.
    value = json.loads((floor.workspace / "handoff.json").read_text())
    value["room_floor"]["inputs_digest"] = "sha256:" + "2" * 64
    value["digest"] = canonical_digest(value, digest_field="digest")
    _write(floor.workspace / "handoff.json", value)
    assert sim.package_source(floor.job.capture, queue_root=floor.job.queue, manifest=_manifest(floor.job),
                              now=NOW)["source_kind"] == "half"
    first = _at(world, 60)
    assert first == status and builds == ["half", "floor", "half"]
    manifest = _manifest(floor.job)
    assert manifest["archive"] is None and len(world.retracted) == 1 and not _package(floor.job).exists()
    assert manifest["source_kind"] == "half" and manifest["refused_half"]["size_rule"] == sim.SIZE_RULE
    for seconds in (120, 180, 3600):
        assert _at(world, seconds)["status"] == status["status"]
    assert builds == ["half", "floor", "half"] and len(world.retracted) == 1 and len(world.delivered) == 1
    # Past the window a timed-out or shape-refused half is tried once more (the policy before this change); a size
    # refusal under this rule stands.
    _at(world, sim.permission.LIVE_STATEMENT_RECHECK_SECONDS + 61)
    assert builds == ["half", "floor", "half"] + ([] if why == "size" else ["half"])


def test_a_timed_out_half_is_tried_once_a_window_and_replaces_the_delivered_floor_when_it_builds(tmp_path,
                                                                                                    monkeypatch):
    floor = Floor(tmp_path, monkeypatch, half=True)
    floor.started(reason="spend_cap_reached")
    builds: list[str] = []
    real = sim.build_package
    slow = [True]

    def build(source, *a, **k):
        builds.append(source["source_kind"])
        if source["source_kind"] == "half" and slow[0]:
            raise ValueError(sim.BUILD_TIMED_OUT)
        return real(source, *a, **k)
    monkeypatch.setattr(sim, "build_package", build)
    world = World(floor.job)
    assert _at(world, 0) == {"status": "build_timed_out"}
    assert _at(world, 60)["status"] == "published" and _manifest(floor.job)["source_kind"] == "floor"
    assert _manifest(floor.job)["delivered"]  # the floor is with the website meanwhile
    for seconds in (120, 3600, sim.permission.LIVE_STATEMENT_RECHECK_SECONDS - 1):
        assert _at(world, seconds)["status"] == "current"
    assert builds == ["half", "floor"]  # not tried again within the window
    slow[0] = False  # the host recovered
    window = sim.permission.LIVE_STATEMENT_RECHECK_SECONDS + 1
    assert _at(world, window)["status"] == "published" and builds == ["half", "floor", "half"]
    manifest = _manifest(floor.job)
    assert manifest["source_kind"] == "half" and "refused_half" not in manifest and len(world.retracted) == 1
    assert "object.usdz" in world.delivered[-1]["files"] and "success_rule.json" in world.delivered[-1]["files"]
    assert _at(world, window + 60)["status"] == "current" and builds == ["half", "floor", "half"]
