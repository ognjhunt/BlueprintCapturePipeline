"""Withdrawal deletion of a site's compiled team scene halves and every record that points at them.

Beta journey step: site withdrawal after the site's scene compiled its robot-team half. Hermetic and
synthetic: the halves here are a few JSON records and placeholder bytes, never a capture, mesh, USD or
image. The one test that compiles a half runs the real worker on the repository's own synthetic rigid
fixture (tests/test_team_scene_packet.py); nothing reaches a provider, a bucket, the WebApp or the host.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_team_scene.py
from __future__ import annotations

import fcntl
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_native_g1 as native_g1
from blueprint_pipeline import website_capture_withdrawal_team_scene as team_scene
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_registry_reader import REGISTRY_SCHEMA
from blueprint_pipeline.team_scene_capture_root import capture_key
from tests import test_website_capture_withdrawal_deletion as base
from tests.test_website_capture_withdrawal_native_g1 import PLACEHOLDER, _namespace

world = base.world  # the deletion tests' site, runs, B2 and roots
OTHER_CAPTURE = "/x/scenes/site-req2/captures/walkthrough-req2"


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _half(root: Path, key: str, attempt: str, records: dict[str, Any]) -> Path:
    """One compiled attempt as the team worker lands it: ``<key>/<attempt>/scene``."""
    scene = root / key / attempt / "scene"
    base._write(scene / "team_scene_inputs.v1.json", {"schema_version": "blueprint.team_scene_inputs.v1", **records})
    base._write(scene / "base/collision.usda", PLACEHOLDER)
    return scene


def _registry(path: Path, derived: Path, *, profiles: list[dict[str, Any]] | None = None,
              site_rows: dict[str, Any] | None = None) -> None:
    value: dict[str, Any] = {"schema_version": REGISTRY_SCHEMA, "profiles": profiles or [], "derived_profiles": {
        "root": str(derived), "defaults": {"allowed_founder_approvers": ["founder@blueprint"],
                                           "max_hourly_rate_usd": 0.8, "open_loop_horizon": 8}}}
    if site_rows is not None:
        value["site_screening_candidates"] = site_rows
    base._write(path, _sealed(value, "registry_digest"))
    path.chmod(0o644)


@pytest.fixture
def team(world, tmp_path):
    resolution = world["resolution"]
    halves, queue, probes = (Path(resolution[name]) for name in (
        "team_scene_halves_root", "team_scene_packet_queue_root", "rigid_probe_queue_root"))
    derived = tmp_path / "inputs/team-scene-profiles"
    _registry(Path(resolution["team_registry"]), derived)
    capture = str(world["capture"].resolve())
    key = capture_key(world["capture"].resolve())
    # The half the worker files the site's capture under: an older attempt, and the current one with a G1
    # namespace beside its inputs (removed first, by the G1 section).
    older = _half(halves, key, "1111111111111111-aaaa0001", {})
    newer = _half(halves, key, "2222222222222222-aaaa0002", {"capture_id": "walkthrough-req1"})
    _namespace(newer)
    # A half under another key whose own records name the site (a supplement capture, a retired workspace).
    by_records = _half(halves, "c" * 32, "3333333333333333-aaaa0003", {
        "capture_root": "/x/scenes/site-req1/captures/walkthrough-req1"})
    # Another site's half, and another capture's half that names nothing: never touched.
    theirs = _half(halves, "d" * 32, "4444444444444444-aaaa0004", {"scene_id": "site-req2"})
    unrelated = _half(halves, "e" * 32, "5555555555555555-aaaa0005", {})
    inputs = newer / "team_scene_inputs.v1.json"
    rows = [base._write(derived / f"{key}.json", {"capture_root": capture, "scene_id": "site-req1",
                                                  "team_scene_inputs_path": str(inputs)}),
            base._write(derived / f"{key}.tmp", {"capture_root": capture}),
            # Only the half it names ties this one to the site.
            base._write(derived / f"{'c' * 32}.json", {"capture_root": "/elsewhere/captures/c",
                                                       "team_scene_inputs_path": str(by_records / "x.json")})]
    other_rows = [base._write(derived / f"{'d' * 32}.json", {"capture_root": OTHER_CAPTURE, "scene_id": "site-req2"}),
                  base._write(derived / f"{'e' * 32}.json", {"capture_root": "/elsewhere/captures/e"})]
    request = {"capture_id": "walkthrough-req1", "intent_id": world["intent_id"]}
    records = [base._write(queue / "processed" / f"{'1' * 32}.json", request),
               # A status names its request's half, not always the site: it goes with its request.
               base._write(queue / "status" / f"{'1' * 32}.json", {"status": "completed",
                                                                   "attempt_root": str(newer.parent),
                                                                   "team_scene_inputs_path": str(inputs)}),
               base._write(queue / "pending" / f"{'2' * 32}.json", request),
               # Only the half its status names ties this request to the site; both go.
               base._write(queue / "processed" / f"{'6' * 32}.json", {"capture_id": "supplement-" + "a" * 40}),
               base._write(queue / "status" / f"{'6' * 32}.json", {"status": "completed",
                                                                   "attempt_root": str(by_records.parent)}),
               base._write(queue / "holds" / f"{'3' * 32}.json", request),
               base._write(queue / "holds" / "settled" / f"{'4' * 32}.json", request),
               base._write(queue / "status" / ".team-scene-k3x9.tmp", {"capture_id": "walkthrough-req1"})]
    other_records = [base._write(queue / "processed" / f"{'9' * 32}.json", {"capture_id": "walkthrough-req2"}),
                     base._write(queue / "status" / f"{'9' * 32}.json", {"capture_id": "walkthrough-req2",
                                                                         "capture_root": OTHER_CAPTURE})]
    lock = base._write(queue / ".lock", b"")  # every worker run creates and holds it
    probe_entries = [base._write(probes / "pending" / f"{key}.json", {"capture_root": capture}),
                     base._write(probes / "done" / f"{'c' * 32}.json", {"capture_root": "/elsewhere/captures/c"})]
    job = probes / "jobs" / f"{key}-0123456789abcdef"
    base._write(job / "attempts/00/rigid_robot_free_probe_attempt.json", {"attempt_id": "development"})
    other_probe = base._write(probes / "pending" / f"{'d' * 32}.json", {"capture_root": OTHER_CAPTURE})
    return {"halves": halves, "queue": queue, "probes": probes, "derived": derived, "key": key, "capture": capture,
            "older": older, "newer": newer, "by_records": by_records, "theirs": theirs, "unrelated": unrelated,
            "rows": rows, "other_rows": other_rows, "records": records, "other_records": other_records,
            "lock": lock, "probe_entries": probe_entries, "job": job, "other_probe": other_probe}


def test_plan_then_apply_deletes_the_sites_halves_and_every_record_pointing_at_them(world, team, tmp_path):
    before = base._snapshot(tmp_path)
    section = base._plan(world, uncovered_roots=())["team_scene"]
    assert base._snapshot(tmp_path) == before  # The plan reads only.
    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs, uncovered_roots=())
    assert plan["unresolved"] == [] and plan["team_scene"] == section
    assert {row["directory"]: row["bound_by"] for row in section["halves"]} == {
        str(team["halves"] / team["key"]): "capture_key", str(team["halves"] / ("c" * 32)): "records"}
    [own] = [row for row in section["halves"] if row["bound_by"] == "capture_key"]
    # Its G1 namespace is the G1 section's; the half is inventoried without it and checked whole after it goes.
    assert own["native_g1_namespaces_removed_first"] == ["2222222222222222-aaaa0002/scene/native-g1"]
    assert own["file_count"] == 4
    assert [row[0] for row in section["derived_profile_rows"]] == sorted(map(str, team["rows"]))
    assert [row[0] for row in section["packet_queue_records"]] == sorted(map(str, team["records"]))
    assert [row[0] for row in section["probe_entries"]] == sorted(map(str, team["probe_entries"]))
    assert section["probe_jobs_remaining"] == [str(team["job"])]

    receipt = base._apply(world, plan, cloud_storage=gcs, uncovered_roots=())
    gone = [team["halves"] / team["key"], team["halves"] / ("c" * 32), *team["rows"], *team["records"],
            *team["probe_entries"]]
    assert not any(path.exists() for path in gone)
    kept = [team["theirs"], team["unrelated"], *team["other_rows"], *team["other_records"], team["other_probe"],
            team["job"], team["lock"]]
    assert all(path.exists() for path in kept)
    deleted = receipt["deleted"]["team_scene"]
    assert deleted["halves"] == [row["directory"] for row in section["halves"]]
    assert deleted["half_files_deleted"] == 4 + 2
    assert deleted["derived_profile_rows"] == sorted(map(str, team["rows"]))
    assert deleted["packet_queue_records"] == sorted(map(str, team["records"]))
    assert deleted["probe_entries"] == sorted(map(str, team["probe_entries"]))
    holders = {row["holder"]: row for row in receipt["remaining"]}
    assert "team_scene_halves" not in holders
    # The probe's jobs keep bundles built from the half and their money records: named, and not complete.
    assert holders["rigid_robot_free_probe_jobs"]["paths"] == [str(team["job"])]
    assert receipt["pipeline_controlled_deletion_complete"] is False
    shutil.rmtree(team["job"])
    plan = base._plan(world, cloud_storage=gcs, uncovered_roots=())
    assert plan["unresolved"] == [] and not team_scene.incomplete(plan["team_scene"])


@pytest.mark.parametrize("fault,reason", [
    ("half_names_another_site", "team_scene_half_names_another_site"),
    ("link_in_another_sites_half", "unsafe_entry"),
    ("linked_half", "unsafe_entry"),
    ("special_file", "unsafe_entry"),
    ("unreadable_record", "record_unreadable"),
    ("row_names_another_site", "derived_profile_names_another_site"),
    ("linked_row", "unsafe_entry"),
    ("request_names_another_site", "team_scene_packet_names_another_site"),
    ("linked_request", "unsafe_entry"),
    ("in_flight", "team_scene_packet_in_flight"),
    ("registry_writable", "team_registry_not_operator_owned"),
    ("registry_unsealed", "team_registry_invalid"),
    ("linked_probe_entry", "unsafe_entry"),
])
def test_a_half_or_record_it_cannot_judge_fails_closed_before_any_deletion(world, team, tmp_path, monkeypatch,
                                                                            fault, reason):
    registry = Path(world["resolution"]["team_registry"])
    if fault == "half_names_another_site":
        base._write(team["older"] / "stray.json", {"scene_id": "site-req9"})
    elif fault == "link_in_another_sites_half":
        os.symlink("/etc", team["theirs"] / "escape")  # another site's half: it could still hide this one's
    elif fault == "linked_half":
        os.symlink(team["theirs"].parents[1], team["halves"] / ("f" * 32))
    elif fault == "special_file":
        os.mkfifo(team["older"] / "pipe")
    elif fault == "unreadable_record":
        target, real_open = team["by_records"] / "team_scene_inputs.v1.json", Path.open

        def guarded(self, *args, **kwargs):
            if self == target:
                raise PermissionError(13, "Permission denied", str(self))
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", guarded)
    elif fault == "row_names_another_site":
        base._write(team["rows"][0], {"capture_root": team["capture"], "shared_with": OTHER_CAPTURE})
    elif fault == "linked_row":
        team["rows"][0].unlink()
        os.symlink(team["other_rows"][0], team["rows"][0])
    elif fault == "request_names_another_site":
        base._write(team["records"][0], {"capture_id": "walkthrough-req1", "also": "walkthrough-req2"})
    elif fault == "linked_request":
        team["records"][2].unlink()
        os.symlink(team["other_records"][0], team["records"][2])
    elif fault == "in_flight":
        # A claimed request: a compile is running, maybe remote, and could still land a new half.
        base._write(team["queue"] / "processing" / f"{'5' * 32}.json", {"capture_id": "walkthrough-req1"})
    elif fault == "registry_writable":
        registry.chmod(0o666)
    elif fault == "registry_unsealed":
        value = json.loads(registry.read_text())
        registry.write_text(json.dumps({**value, "profiles": [{"capture_root": "/rewritten"}]}))
    else:
        team["probe_entries"][0].unlink()
        os.symlink(team["other_probe"], team["probe_entries"][0])
    plan = base._plan(world)
    assert {"website_withdrawal_deletion_" + reason} <= {row["reason"] for row in plan["unresolved"]}, \
        plan["unresolved"]
    before = base._snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        base._apply(world, plan)
    assert base._snapshot(tmp_path) == before


@pytest.mark.parametrize("change,code", [
    ("half_gains_a_file", "team_scene_half_changed"),
    ("new_g1_namespace", "team_scene_half_changed"),
    ("row_rewritten", "team_scene_record_changed"),
    ("status_rewritten", "team_scene_record_changed"),
    ("probe_entry_gone", "team_scene_record_changed"),
])
def test_a_half_or_record_changed_after_review_is_not_deleted(world, team, tmp_path, change, code):
    plan = base._plan(world)
    # As in a deletion: the G1 namespaces inside the halves go first.
    native_g1.apply_native_g1(plan["native_g1"])
    if change == "half_gains_a_file":
        base._write(team["older"] / "late.json", {"written": "after review"})
    elif change == "new_g1_namespace":
        _namespace(team["older"], source="f" * 64)  # a G1 registration ran since the plan
    elif change == "row_rewritten":
        base._write(team["rows"][0], {"capture_root": team["capture"], "rewritten": "after review"})
    elif change == "status_rewritten":
        base._write(team["records"][1], {"status": "blocked", "rewritten": "after review"})
    else:
        team["probe_entries"][0].unlink()
    before = base._snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match=code):
        team_scene.apply_team_scene(plan["team_scene"])
    # Everything is checked before anything goes: not one record pointing at a half was removed.
    assert base._snapshot(tmp_path) == before


def test_a_change_after_review_refuses_the_whole_deletion(world, team, tmp_path):
    plan = base._plan(world)
    base._write(team["older"] / "late.json", {"written": "after review"})
    before = base._snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="plan_changed"):
        base._apply(world, plan)
    assert base._snapshot(tmp_path) == before


def test_a_worker_run_holding_the_packet_queue_refuses_the_deletion(world, team, tmp_path):
    plan = base._plan(world)
    before = base._snapshot(tmp_path)
    with team["lock"].open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(deletion.WithdrawalDeletionError, match="team_scene_packets_busy"):
            base._apply(world, plan)
    assert base._snapshot(tmp_path) == before


def test_a_queue_no_worker_has_used_is_not_written_to(world, team, tmp_path):
    team["lock"].unlink()
    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs)
    base._apply(world, plan, cloud_storage=gcs)
    assert not team["lock"].exists() and not (team["halves"] / team["key"]).exists()


def test_what_the_deletion_was_not_given_is_named_and_keeps_it_incomplete(world, team):
    gcs = base.FakeGcs()
    roots = {"team_registry": None, "team_scene_packet_queue_root": None, "rigid_probe_queue_root": None,
             "uncovered_roots": ()}
    receipt = base._apply(world, base._plan(world, cloud_storage=gcs, **roots), cloud_storage=gcs, **roots)
    holders = {row["holder"] for row in receipt["remaining"]}
    assert {"team_registry_not_read", "team_scene_packet_queue_not_scanned",
            "rigid_robot_free_probe_queue_not_scanned"} <= holders
    assert receipt["pipeline_controlled_deletion_complete"] is False
    # The halves still go; what was not scanned is untouched.
    assert not (team["halves"] / team["key"]).exists()
    assert all(path.exists() for path in [*team["rows"], *team["records"], *team["probe_entries"]])
    roots = {"team_scene_halves_root": None, "native_g1_registration_root": None}
    section = base._plan(world, **roots)["team_scene"]
    assert section["halves"] == [] and "team_scene_halves_not_scanned" in {
        row["holder"] for row in team_scene.remaining(section)}


def test_root_owned_registry_rows_naming_the_site_are_named_never_written(world, team):
    registry = Path(world["resolution"]["team_registry"])
    _registry(registry, team["derived"], profiles=[{"capture_root": team["capture"], "scene_id": "site-req1"},
                                                   {"capture_root": OTHER_CAPTURE, "scene_id": "site-req2"}],
              site_rows={"site-req1": {}, "site-req2": {}})
    before = registry.read_bytes()
    shutil.rmtree(team["job"])
    gcs = base.FakeGcs()
    plan = base._plan(world, cloud_storage=gcs, uncovered_roots=())
    assert plan["unresolved"] == []
    receipt = base._apply(world, plan, cloud_storage=gcs, uncovered_roots=())
    [row] = [row for row in receipt["remaining"] if row["holder"] == "team_registry_operator_rows"]
    assert (row["profile_indexes"], row["site_screening_candidates"]) == ([0], True)
    assert receipt["pipeline_controlled_deletion_complete"] is False and registry.read_bytes() == before


def test_a_real_compiled_half_is_found_deleted_and_never_compiled_again(tmp_path, monkeypatch):
    """The worker's own half, row, request, status and probe entry; then every reader of them, with them gone."""
    from blueprint_pipeline import rigid_robot_free_probe_runner as runner
    from blueprint_pipeline import team_scene_packet_worker as worker
    from blueprint_pipeline.beta_policy_screening_scene_half import current_half
    from blueprint_pipeline.rigid_robot_free_probe_bundle import current_probe_state
    from blueprint_pipeline.rigid_robot_free_probe_lane import ENABLE_ENV
    from blueprint_pipeline.rigid_support_pad import half_support_hold
    from blueprint_pipeline.robot_access_furniture import exchange_robot_access_decisions, half_proposal
    from blueprint_pipeline.robot_access_measurement import measurement_due
    from blueprint_pipeline.team_candidate_quick10_registry import (
        TeamCandidateRegistryError,
        offer_eligibility,
        profile_for_capture,
        scene_facts,
    )
    from blueprint_pipeline.team_scene_packet import recorded_embodiment_stations
    from tests.test_team_scene_packet_worker import CAPTURE_ID, REQUEST_ID, SCENE_ID, _world

    probes = tmp_path / "rigid-robot-free-probes"
    monkeypatch.setenv(runner.QUEUE_ENV, str(probes))
    real = _world(tmp_path, monkeypatch)
    queue, key, capture = real["queue"], real["key"], real["capture"]
    [status] = real["process"]()["results"]
    assert status["status"] == "completed"
    profile = profile_for_capture(capture)
    inputs_path = profile["team_scene_inputs_path"]
    request = (queue / "processed" / f"{key}.json").read_bytes()
    base._write(capture / "raw/video.mp4", b"synthetic-video")
    withdrawal.acknowledge_withdrawal(capture_root=capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": REQUEST_ID, "scene_id": SCENE_ID,
        "capture_id": CAPTURE_ID, "withdrawal_id": "withdrawal-development",
        "requested_at_iso": "2026-11-06T00:00:00Z"})
    site, identity = deletion._site_identity(capture)
    unresolved: list[dict[str, Any]] = []
    section = team_scene.plan_team_scene(
        identity, [], capture_ids=native_g1._site_capture_ids(site, identity),
        capture_keys=native_g1._site_capture_keys(site, identity), native_g1={"packet_namespaces": []},
        halves_root=Path(real["inputs"]["output_root"]), registry=real["operator"] / "registry.json",
        packet_queue_root=queue, probe_queue_root=probes, unresolved=unresolved)
    assert unresolved == []
    # Exactly the records the worker wrote for this capture, found by the deletion's own rules.
    assert [row["directory"] for row in section["halves"]] == [str(real["halves"])]
    assert Path(inputs_path).is_relative_to(real["halves"])
    assert [row[0] for row in section["derived_profile_rows"]] == [
        str(real["operator"] / "derived" / f"{real['halves'].name}.json")]
    assert [row[0] for row in section["packet_queue_records"]] == [
        str(queue / "processed" / f"{key}.json"), str(queue / "status" / f"{key}.json")]
    # The half's record queued the capture's robot-free probe (record_team_scene).
    assert [row[0] for row in section["probe_entries"]] == [str(probes / "pending" / f"{real['halves'].name}.json")]
    with ExitStack() as locks:
        team_scene.hold_packet_queue(locks, queue)
        team_scene.apply_team_scene(section)
    assert not real["halves"].exists() and list(probes.glob(f"*/{real['halves'].name}.json")) == []

    # The registry's readers: no row, so no profile, no offer, no probe request.
    assert profile_for_capture(capture) is None
    assert offer_eligibility(None) == (None, ["team_candidate_profile_missing"])
    assert current_probe_state(capture)["status"] == "profile_missing"
    # A reader still holding the old row, or a status naming the half, gets a typed refusal or nothing.
    with pytest.raises(TeamCandidateRegistryError, match="team_scene_inputs_unavailable"):
        scene_facts(profile)
    assert offer_eligibility(profile) == (None, ["team_scene_inputs_unavailable"])
    assert current_half(profile, scene_revision_digest=profile["scene_revision_digest"]) is None
    for reader in (half_proposal, measurement_due, half_support_hold, worker.unscorable_half,
                   worker.stale_unreached_station):
        assert reader(inputs_path) is None, reader
    assert recorded_embodiment_stations(inputs_path, plan_digest="sha256:" + "0" * 64) == {}
    assert exchange_robot_access_decisions(queue, transport=lambda *_: pytest.fail("asked the site")) == {}
    # The probe lane, switched on, defers a leftover entry of the capture on every pass, and spends nothing.
    monkeypatch.setenv(ENABLE_ENV, "1")
    leftover = base._write(probes / "pending" / f"{real['halves'].name}.json", {
        "schema_version": runner.ENTRY_SCHEMA, "capture_root": str(capture),
        "probe_request_digest": "sha256:" + "1" * 64})
    hooks = {"source_commit": lambda: "a" * 40,
             "allocator_runner": lambda *_, **__: pytest.fail("allocated for a withdrawn capture")}
    for _ in range(2):
        assert runner.run_queue(probes, hooks=hooks) == [
            {"status": "deferred", "capture_root": str(capture), "reason": "profile_missing"}]
    assert leftover.is_file() and not (probes / "jobs").exists()

    # Nothing compiles the withdrawn capture's half again: its request queued again is refused by name.
    base._write(queue / "pending" / f"{key}.json", request)
    [status] = real["process"]()["results"]
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_capture_withdrawn"])
    assert not real["halves"].exists() and profile_for_capture(capture) is None
    # A deterministic refusal this release wrote is not retried by the timer's pass.
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # Nor once the capture itself is deleted.
    shutil.rmtree(capture)
    os.replace(queue / "processed" / f"{key}.json", queue / "pending" / f"{key}.json")
    [status] = real["process"]()["results"]
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_capture_root_missing"])
    assert not real["halves"].exists()
