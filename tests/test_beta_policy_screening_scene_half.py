"""A screening campaign is planned again when the scene's completed team half is compiled again.

capture-447ae4f4: every learned policy was planned ``infeasible`` from the half's first Franka station.
The stance search and the owner-approved chair move compile that half again with a new station, but the
source (intent, capture, task, revision) is the same, so the campaign id was too and screening never
planned again. The worker now supersedes a campaign that holds no reservation and no active row when the
half the source's profile names differs from the one the plan froze, once per new half.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline import team_candidate_quick10_registry as registry
from blueprint_pipeline.beta_policy_screening_scene_half import current_half, half_identity, planned_half
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_plan import candidate_rows
from blueprint_pipeline.team_scene_packet import (
    CLAIM_CEILING, EXECUTION_AUTHORITY, PENDING_CONTROLS, RECORD_FILENAME, SCHEMA, _record, read_team_scene_inputs,
)
from blueprint_pipeline.team_scene_packet_worker import request_key
from tests.test_beta_policy_screening_coordinator import Backend
from tests.test_beta_policy_screening_worker import prepared_source_queue


def compile_half(root: Path, *, revision: str, station: list[float], attempt: str) -> str:
    """A sealed team half as the packet worker writes it: its packet's bytes carry the station."""

    directory = root / "halves" / attempt
    directory.mkdir(parents=True)
    packet = directory / "native_task_arena_packet_receipt.v1.json"
    packet.write_text(json.dumps({"robot_base_position_world_m": station}))
    runtime = directory / "runtime_source.json"
    runtime.write_text("{}")
    packet_record = _record(packet)
    stance = canonical_digest({"position_world_m": station})
    value = {"schema_version": SCHEMA, "claim_ceiling": CLAIM_CEILING, "scene_revision_digest": revision,
             "configuration_digest": canonical_digest({"scene_revision_digest": revision,
                                                       "packet": packet_record["sha256"], "station": stance}),
             "plan_digest": canonical_digest({"scene_plan": station}),
             "task_success_contract": {"contract_digest": canonical_digest({"value": "contract"})},
             "task_success_contract_digest": canonical_digest({"value": "contract"}),
             "base_native_packet": packet_record, "runtime_source": _record(runtime),
             "construction_result": packet_record, "execution_authority": dict(EXECUTION_AUTHORITY),
             "control_diagnostic": dict(PENDING_CONTROLS), "robot_station": {"stance_digest": stance},
             "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path = directory / RECORD_FILENAME
    path.write_text(json.dumps(value, sort_keys=True))
    read_team_scene_inputs(path)  # a half every reader accepts
    return str(path)


def scene_half_world(tmp_path, monkeypatch):
    """The worker's prepared source, its profile naming a compiled half that planning freezes for real."""

    world = prepared_source_queue(tmp_path, monkeypatch)
    [request] = world["requests"]
    capture = tmp_path / "partition/scenes/site-job1/captures/walkthrough-job1"
    profile = registry.profile_for_capture(capture)
    revision = request["scene_revision_digest"]
    profile.update(scene_revision_digest=revision,
                   team_scene_inputs_path=compile_half(tmp_path, revision=revision, station=[0.0, -0.8, 0.0],
                                                       attempt="first"))
    stub = capabilities.build_planning_inputs

    def planning_inputs(**kwargs):
        # The planner copies the half's own fields into the scene inputs it seals for the plan.
        built = stub(**kwargs)
        half = read_team_scene_inputs(kwargs["profile"]["team_scene_inputs_path"])
        inputs_dir = Path(kwargs["inputs_dir"])
        inputs_dir.mkdir(parents=True, exist_ok=True)
        sealed = inputs_dir / (canonical_digest(half)[7:] + ".scene_inputs.json")
        sealed.write_text(json.dumps({**half, "screening_capture_root": str(capture)}, sort_keys=True))
        built["scene"]["scene_inputs"] = _record(sealed)
        return built

    monkeypatch.setattr(capabilities, "build_planning_inputs", planning_inputs)
    world.update(profile=profile, revision=revision, key=request_key(request),
                 kwargs={"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]})
    return world


def recompile(world, tmp_path, *, station, attempt):
    world["profile"]["team_scene_inputs_path"] = compile_half(tmp_path, revision=world["revision"],
                                                              station=station, attempt=attempt)


def plan_of(world, campaign_id):
    return json.loads((world["screening_queue"] / "plans" / f"{campaign_id}.json").read_text())


def finish_without_spend(world):
    """Every funded candidate is refused before allocation; released money funds the waitlist, then nothing."""

    class Refuses(Backend):
        def prepare_candidate(self, **kwargs):
            self.preparations += 1
            raise ValueError("beta_screening_not_allocated")

    for _ in range(4):
        coordinator.process_screening_queue(queue_root=world["screening_queue"], executor=Refuses(),
                                            require_publication_ack=False, max_launches=20)
    [publication] = coordinator.pending_publications(queue_root=world["screening_queue"])
    assert publication["status"] == "blocked"
    assert publication["budget"]["committed_upper_bound_usd"] == "0"
    return publication


def test_identity_is_the_halfs_bytes_not_where_they_were_compiled(tmp_path):
    revision = canonical_digest({"value": "revision"})
    first = compile_half(tmp_path, revision=revision, station=[0.0, -0.8, 0.0], attempt="a")
    copy = compile_half(tmp_path, revision=revision, station=[0.0, -0.8, 0.0], attempt="b")
    moved = compile_half(tmp_path, revision=revision, station=[0.3, -0.6, 0.0], attempt="c")
    profile = {"scene_revision_digest": revision, "team_scene_inputs_path": first}
    assert current_half(profile, scene_revision_digest=revision) is not None
    assert (current_half(profile, scene_revision_digest=revision)
            == current_half({**profile, "team_scene_inputs_path": copy}, scene_revision_digest=revision))
    assert (current_half(profile, scene_revision_digest=revision)
            != current_half({**profile, "team_scene_inputs_path": moved}, scene_revision_digest=revision))
    # Another revision's half, a frozen pair, a missing or broken record: no half is known.
    other = canonical_digest({"value": "another revision"})
    assert current_half(profile, scene_revision_digest=other) is None
    assert current_half({"pair_runtime_inputs_path": "/x", "scene_revision_digest": revision},
                        scene_revision_digest=revision) is None
    Path(moved).chmod(0o644)
    Path(moved).write_text("{")
    assert current_half({**profile, "team_scene_inputs_path": moved}, scene_revision_digest=revision) is None
    assert half_identity({"scene_revision_digest": revision}) is None
    # A plan's own sealed inputs name the half it froze; changed bytes name none.
    sealed = tmp_path / "scene_inputs.json"
    sealed.write_text(json.dumps(read_team_scene_inputs(first)))
    assert planned_half({"scene_inputs": _record(sealed)}) == current_half(profile, scene_revision_digest=revision)
    record = _record(sealed)
    sealed.write_text("{}")
    assert planned_half({"scene_inputs": record}) is None


def test_a_finished_campaign_whose_half_was_compiled_again_is_planned_again_once(tmp_path, monkeypatch):
    world = scene_half_world(tmp_path, monkeypatch)
    root = world["screening_queue"]
    [adopted] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    first = adopted["campaign_id"]
    before = finish_without_spend(world)
    # The known-good control is never planned again; every candidate is.
    rows = {row["candidate_id"]: row for row in candidate_rows(before)}
    # The stance search finds another station: the half is compiled again, completed.
    recompile(world, tmp_path, station=[0.3, -0.6, 0.0], attempt="second")
    [replanned] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    second = replanned["campaign_id"]
    assert replanned["status"] == "enqueued" and replanned["superseded_campaign_id"] == first and second != first
    assert len(world["planning_calls"]) == 2
    # The new plan froze the new half; the old one keeps the half it was planned from.
    now = current_half(world["profile"], scene_revision_digest=world["revision"])
    assert planned_half(plan_of(world, second)) == now != planned_half(plan_of(world, first))
    # No candidate spent, so every one is planned again, and the plan is sized as a re-plan.
    assert {row["candidate_id"] for row in plan_of(world, second)["candidates"]} == set(rows)
    ledger = coordinator._ledger(root)
    assert ledger["campaigns"][first]["superseded_by"] == second
    assert ledger["campaigns"][second]["supersedes"] == first
    # The finished campaign's rows are left exactly as published: none was queued to close.
    after = next(row for row in coordinator.pending_publications(queue_root=root) if row["campaign_id"] == first)
    assert after["candidates"] == before["candidates"] and after["status"] == "blocked"
    successor = next(row for row in coordinator.pending_publications(queue_root=root) if row["campaign_id"] == second)
    assert successor["status"] == "queued" and successor["source"] == before["source"]
    # Superseded once: later passes on the same half plan nothing.
    for _ in range(2):
        [replay] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
        assert replay["status"] == "already_enqueued" and replay["campaign_id"] == second
    assert len(world["planning_calls"]) == 2
    assert len(list((root / "plans").glob("*.json"))) == 2


def test_queued_rows_of_a_campaign_whose_half_changed_close_as_replanned(tmp_path, monkeypatch):
    world = scene_half_world(tmp_path, monkeypatch)
    [adopted] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    queued = {row["candidate_id"] for row in plan_of(world, adopted["campaign_id"])["candidates"]
              if row["status"] in {"selected", "waitlisted"}}
    recompile(world, tmp_path, station=[0.3, -0.6, 0.0], attempt="second")
    [replanned] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    assert replanned["superseded_campaign_id"] == adopted["campaign_id"]
    old = coordinator._ledger(world["screening_queue"])["campaigns"][adopted["campaign_id"]]
    assert {key for key, row in old["candidates"].items()
            if row["blockers"] == [coordinator.REPLANNED_BLOCKER]} == queued
    assert all(row["status"] == "blocked" for key, row in old["candidates"].items() if key in queued)


@pytest.mark.parametrize("observation", [{"status": "running"}, RuntimeError("provider response lost")])
def test_a_campaign_with_an_active_row_or_a_reservation_is_never_superseded(tmp_path, monkeypatch, observation):
    world = scene_half_world(tmp_path, monkeypatch)
    root = world["screening_queue"]
    [adopted] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    coordinator.process_screening_queue(queue_root=root, executor=Backend(observation),
                                        require_publication_ack=False)
    [publication] = coordinator.pending_publications(queue_root=root)
    assert publication["budget"]["reserved_usd"] != "0"
    recompile(world, tmp_path, station=[0.3, -0.6, 0.0], attempt="second")
    [replay] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    assert replay["status"] == "already_enqueued" and replay["campaign_id"] == adopted["campaign_id"]
    assert len(world["planning_calls"]) == 1
    assert "superseded_by" not in coordinator._ledger(root)["campaigns"][adopted["campaign_id"]]


def test_an_unchanged_or_unfinished_half_never_plans_again(tmp_path, monkeypatch):
    world = scene_half_world(tmp_path, monkeypatch)
    [adopted] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    finish_without_spend(world)
    # The same half compiled again into another directory is the same half.
    recompile(world, tmp_path, station=[0.0, -0.8, 0.0], attempt="same-bytes")
    for _ in range(2):
        [replay] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
        assert replay["status"] == "already_enqueued" and replay["campaign_id"] == adopted["campaign_id"]
    # A changed half whose request is not completed (the operator's own row) is not re-planned from.
    recompile(world, tmp_path, station=[0.3, -0.6, 0.0], attempt="second")
    status = world["packet_queue"] / "status" / f"{world['key']}.json"
    status.write_text(json.dumps({"status": "skipped"}))
    [replay] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    assert replay["status"] == "already_enqueued" and replay["campaign_id"] == adopted["campaign_id"]
    # A half of another scene revision belongs to that revision's own source.
    status.write_text(json.dumps({"status": "completed"}))
    world["profile"]["scene_revision_digest"] = canonical_digest({"value": "another revision"})
    [replay] = worker.enqueue_prepared_scenes(**world["kwargs"])["sources"]
    assert replay["status"] == "already_enqueued" and replay["campaign_id"] == adopted["campaign_id"]
    assert len(world["planning_calls"]) == 1
    assert len(list((world["screening_queue"] / "plans").glob("*.json"))) == 1
