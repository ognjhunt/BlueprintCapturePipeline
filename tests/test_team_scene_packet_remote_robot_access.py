# Covers (for impacted-test selection):
#   src/blueprint_pipeline/team_scene_packet_remote.py
#   src/blueprint_pipeline/team_scene_packet_worker.py
#   src/blueprint_pipeline/team_scene_packet.py
"""Movable furniture on the remote team-scene lane: the worker's compile reads only what ships or what the host read.

The compile reads three things under the request's own intent (``robot_access_scope``) besides its packet
inputs: what the intent knows of its movable furniture (read on the host before compiling, as
``prepare_team_scene`` reads it), the offer to measure that furniture while it is unmeasured (the intent's
registration, preparation, scene review and source video, none of which ships), and the site owner's sealed
approvals of a proposal to move it (which ship). Both routes hand the compile the same arguments; the planner
ships the approvals and reads the offer on the host; a compile given that offer writes the host's half byte
for byte without reading the preparation; and a landed compile is never measured (or bought) again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import robot_access_furniture as access
from blueprint_pipeline import robot_access_measurement as measurement
from blueprint_pipeline import team_scene_packet as packet
from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.task_evaluation_owner_success_confirmation import DIRECTORY as OWNER_DIRECTORY
from tests.remote_team_scene_packet_support import COMMIT, TeamCollectorWorld, tree_snapshot

REVISION = {"revision_digest": "sha256:" + "1" * 64}
KNOWN = {"status": "measured", "record": {"items": []}, "sidecar_digest": "sha256:" + "2" * 64,
         "preparation_digest": "sha256:" + "3" * 64, "offer": {"never": "passed"}}


class Stop(Exception):
    """Ends a compile at its first step: only its arguments matter here."""


def test_both_routes_hand_the_compile_the_intent_scopes_and_the_hosts_movable_furniture(tmp_path: Path,
                                                                                         monkeypatch) -> None:
    scope = {"intent": {"intent_id": "intent-1"}, "directory": tmp_path / "intents" / "intent-1"}
    documents = {"support_plane": tmp_path / "attempt" / "documents" / "support_plane.json"}
    reads, compiled, planned = [], [], []
    monkeypatch.setattr(measurement, "furniture_for_scope",
                        lambda value, *, revision_digest: reads.append((value, revision_digest)) or dict(KNOWN))
    monkeypatch.setattr(packet, "fetch_team_scene_sources", lambda **_: (REVISION, documents))

    def materialize(**kwargs):
        compiled.append(kwargs)
        raise Stop()

    monkeypatch.setattr(packet, "materialize_team_scene", materialize)
    inputs = {"droid_profile_path": tmp_path / "droid.json", "droid_profile_digest": "sha256:" + "4" * 64,
              "runtime_source_receipt_path": tmp_path / "runtime.json"}
    attempt, bundle = tmp_path / "attempt", tmp_path / "bundle.zip"
    with pytest.raises(Stop):
        packet.prepare_team_scene(run_root=tmp_path / "run", capture_root=tmp_path / "capture",
                                  scenario_id=worker.SCENARIO_ID, output_root=attempt,
                                  configured_scene_bundle_path=bundle, **inputs, **worker._intent_scopes(scope))

    def plan(**kwargs):
        planned.append(kwargs["arguments"])
        return lane.HostDecision("remote_ineligible:test")

    monkeypatch.setattr(lane, "plan_remote_team_scene", plan)
    remote = SimpleNamespace(mode="cloud_run", source_commit=COMMIT, config=None, jobs_root=tmp_path / "jobs",
                             filesystem_root=tmp_path, cache_root=None, host_environment=None, clock=lambda: 0.0)
    with pytest.raises(Stop):
        worker._compile_routed({"request_digest": "sha256:" + "5" * 64, "scene_revision_digest": REVISION[
            "revision_digest"]}, status={"request_key": "key"}, run=tmp_path / "run", capture=tmp_path / "capture",
            attempt=attempt, bundle=bundle, inputs=inputs, fetcher=None, embodiments=(), models=None,
            owner_scope=scope, remote=remote)
    # The host's compile, the remote planner and a host-decided compile on the remote route see the same call.
    host, routed = compiled
    assert host == routed == planned[0]
    assert host["robot_access_scope"] == host["owner_success_confirmation_scope"] == scope
    assert host["movable_furniture"] == {key: value for key, value in KNOWN.items() if key != "offer"}
    assert reads == [(scope, REVISION["revision_digest"])] * 2
    # An unbound request compiles exactly as before: no scope, so nothing about furniture is read.
    assert worker._intent_scopes(None) == {}


def test_the_planner_ships_the_intents_approvals_and_reads_the_measurement_offer_on_the_host(tmp_path: Path,
                                                                                            monkeypatch) -> None:
    fs = tmp_path / "fs"
    base = fs / "var/lib/blueprint/test"
    directory = base / "intents" / "intent-1"
    approvals, confirmations = directory / access.CONFIRMATION_DIRECTORY, directory / OWNER_DIRECTORY
    approvals.mkdir(parents=True)
    confirmations.mkdir()
    approval = approvals / ("a" * 64 + ".json")
    approval.write_text("{}", encoding="utf-8")
    (approvals / "notes.txt").write_text("not a record", encoding="utf-8")
    confirmation = confirmations / ("b" * 64 + ".json")
    confirmation.write_text("{}", encoding="utf-8")
    (directory / "revoked.json").write_text("{}", encoding="utf-8")
    attempt = base / "halves" / "0123456789abcdef-01234567"
    document = attempt / "documents" / "support_plane.json"
    document.parent.mkdir(parents=True)
    document.write_text("{}", encoding="utf-8")
    scope = {"intent": {"intent_id": "intent-1"}, "directory": str(directory)}
    arguments = {"revision": REVISION, "droid_profile_digest": "sha256:" + "4" * 64,
                 **{name: str(base / name) for name in lane.PATH_ARGUMENTS}, "documents": {"support_plane": document},
                 "owner_success_confirmation_scope": scope, "robot_access_scope": scope,
                 "movable_furniture": {"status": "not_measured", "reason": "no_sidecar_for_the_current_preparation"}}

    # Every record either sealed-confirmation reader may read under the intent ships, its revocation included.
    files = {label: path for _, label, path in lane._plan_files(arguments, attempt)}
    assert files["robot_access_confirmation." + approval.name] == approval
    assert files["owner_success_confirmation." + confirmation.name] == confirmation
    assert files["robot_access_confirmation.revoked"] == files["owner_success_confirmation.revoked"]
    assert not any(label.endswith("notes.txt") for label in files)

    # Both scopes travel as worker paths, the host-read furniture as data, and both map back on the worker.
    converted = lane._worker_arguments(arguments, fs)
    assert converted is not None
    worker_directory = "/var/lib/blueprint/test/intents/intent-1"
    assert converted["robot_access_scope"] == converted["owner_success_confirmation_scope"] == {
        "intent": scope["intent"], "directory": worker_directory}
    assert converted["movable_furniture"] == arguments["movable_furniture"]

    # The offer is read on the host only when the compile could read it: a bound intent, furniture unmeasured.
    offers = []
    offer = {"status": "available", "binding": {"items": []}, "binding_digest": "sha256:" + "6" * 64}
    monkeypatch.setattr(measurement, "measurement_offer",
                        lambda value, *, revision_digest: offers.append((value, revision_digest)) or dict(offer))
    assert lane._robot_access_offer({**arguments, "movable_furniture": dict(KNOWN)}) is None
    assert lane._robot_access_offer({key: value for key, value in arguments.items()
                                     if key != "robot_access_scope"}) is None
    assert lane._robot_access_offer({key: value for key, value in arguments.items()
                                     if key != "movable_furniture"}) is None
    assert offers == []
    assert lane._robot_access_offer(arguments) == offer
    assert offers == [(scope, REVISION["revision_digest"])]

    roots = SimpleNamespace(local=lambda path: fs / str(path).lstrip("/"))
    local = lane._local_arguments({**converted, "robot_access_offer": offer}, roots)
    assert local["robot_access_scope"] == {"intent": scope["intent"], "directory": directory}
    assert (local["movable_furniture"], local["robot_access_offer"]) == (arguments["movable_furniture"], offer)


@pytest.mark.slow
def test_a_compile_given_the_host_read_offer_writes_the_hosts_half_without_reading_the_preparation(
        tmp_path: Path, monkeypatch, chairs_room) -> None:
    """capture-447's chairs room (``tests/test_robot_access_furniture.py``), compiled on the host and then as the
    worker compiles it, at the same output root: once with the furniture unmeasured (the half records the
    measurement as due, from the offer), and once measured (the half proposes moving one chair). The worker's
    compile cannot read the intent's preparation; its half is the host's, path for path and byte for byte."""

    from tests import test_robot_access_furniture as chairs
    from tests import test_team_scene_packet_single_mesh_room as single

    inputs = single._inputs(tmp_path, monkeypatch)
    scope = inputs["owner_success_confirmation_scope"]
    chairs.renewed_lineage(tmp_path, scope, monkeypatch)
    paid = chairs.PaidCalls([target for _, target in chairs.CHAIRS])
    monkeypatch.setattr(measurement, "lift", paid.lift)
    output = tmp_path / "team-scene"
    host_inputs = {**{key: value for key, value in inputs.items() if key != "owner_success_confirmation_scope"},
                   **worker._intent_scopes(scope)}

    def both(expected: str) -> dict:
        arguments = packet.with_movable_furniture(host_inputs, revision=inputs["revision"])
        host = packet.materialize_team_scene(**arguments, output_root=output)
        record = json.loads(Path(host["path"]).read_text(encoding="utf-8"))
        assert record["robot_access_removal"]["status"] == expected, record["robot_access_removal"]
        snapshot = tree_snapshot(output)
        offer = lane._robot_access_offer(arguments)
        assert (offer is None) is (arguments["movable_furniture"]["status"] == "measured")
        shutil.rmtree(output)
        with monkeypatch.context() as blocked:
            # As on the worker: nothing under the intent but its shipped approvals can be read.
            def unreadable(_scope):
                raise OSError("the intent's preparation does not ship")

            blocked.setattr(measurement, "current_preparation", unreadable)
            remote = packet.materialize_team_scene(
                **arguments, output_root=output, **({"robot_access_offer": offer} if offer is not None else {}))
        assert tree_snapshot(output) == snapshot
        assert {key: remote.get(key) for key in ("inputs_digest", "robot_access_removal")} == {
            key: host.get(key) for key in ("inputs_digest", "robot_access_removal")}
        return record

    due = both("measurement_due")["robot_access_removal"]
    assert measurement.run_measurement(scope, binding_digest=due["binding_digest"],
                                       revision_digest=inputs["revision"]["revision_digest"], analyze=paid.analyze,
                                       segment=paid.segment, geometry=lambda *_: {"frames": []},
                                       frames=lambda **_: [])["sidecar_digest"]
    shutil.rmtree(output)
    proposed = both("proposed")["robot_access_removal"]
    assert [row["obstacle_id"] for row in proposed["items"]] == [chairs.BLOCKING]


@pytest.fixture
def chairs_room(monkeypatch) -> None:
    from tests import test_robot_access_furniture as chairs
    from tests import test_team_scene_packet_single_mesh_room as single

    write = single.write_scanned_room
    monkeypatch.setattr(single, "write_scanned_room",
                        lambda path, **options: write(path, parts=[mesh for mesh, _ in chairs.CHAIRS], **options))


def test_a_landed_compile_is_never_measured_again(tmp_path: Path, monkeypatch) -> None:
    """A half whose furniture measurement is due is measured once, before its compile is handed off; when that
    compile lands, the hand-off's measurement record stands and nothing is measured or bought again."""

    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    half = str(tmp_path / "an-earlier-half" / packet.RECORD_FILENAME)
    binding = "sha256:" + "7" * 64
    measured = {"status": "measured", "binding_digest": binding, "sidecar_digest": "sha256:" + "8" * 64,
                "reserved_cost_usd": 0.22, "spend": "website_preparation_spend_within_the_jobs_approved_caps"}
    runs = []
    monkeypatch.setattr(worker, "_current_half", lambda profile, revision: half)
    monkeypatch.setattr(measurement, "measurement_due",
                        lambda path: {"binding_digest": binding} if path == half else None)
    monkeypatch.setattr(worker, "_measure_furniture", lambda request, digest: runs.append(digest) or dict(measured))

    [handed] = host.process()["results"]
    assert handed["status"] == "processing" and handed["robot_access_measurement"] == measured
    assert runs == [binding] and world.compiles == []
    world.drive(until=lambda: bool(lane.markers(host.jobs, "collected")) and world.terminal())
    [finished] = host.process()["remote_settled"]
    assert (finished["status"], finished["outcome"]) == ("completed", "compiled"), finished
    assert finished["robot_access_measurement"] == measured
    assert runs == [binding] and world.compiles == ["worker"]
