# Covers (for impacted-test selection):
#   src/blueprint_pipeline/team_scene_packet_collector.py
#   src/blueprint_pipeline/team_scene_packet_remote.py
#   src/blueprint_pipeline/team_scene_packet_worker.py
"""The team stage's paid unit: dispatch through the allocator, follow, land, tear down, and hand back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_scene_packet import RECORD_FILENAME, read_team_scene_inputs
from tests.remote_team_scene_packet_support import TEAM_WORST_CASE_USD, TeamCollectorWorld, tree_snapshot


def _texts(root: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sorted(root.rglob("*"))
                     if path.is_file() and path.stat().st_size < 4 * 1024 * 1024)


def _settled(run: dict) -> list[dict]:
    """What a no-spend run finished of the remote lane's requests (a landed compile or a handed-back one)."""

    return run["remote_settled"]


def _settlements(world: TeamCollectorWorld) -> list[dict]:
    root = allocator._settled_root()
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def test_a_handed_off_request_compiles_remotely_and_is_recorded_as_a_host_compile(tmp_path: Path,
                                                                                  monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    [handed] = host.process()["results"]
    assert handed["status"] == "processing" and world.compiles == []
    attempt = Path(handed["attempt_root"])

    world.drive(until=lambda: bool(lane.markers(host.jobs, "collected")) and world.terminal())
    lease = world.lease()
    assert (lease["state"], lease["attempt"]) == ("completed", 1)
    # One paid attempt, admitted at its worst case, torn down to provider-zero and settled under it.
    [consumed] = sorted((world.remote.spend / "consumed").glob("remote-cpu-*.json"))
    assert json.loads(consumed.read_text(encoding="utf-8"))["worst_case_usd"] == TEAM_WORST_CASE_USD
    [settled] = _settlements(world)
    assert settled["basis"] == "execution_runtime" and 0 < settled["settled_usd"] <= TEAM_WORST_CASE_USD
    teardown = json.loads((host.jobs / "teardowns" / f"{lease['attempt_id']}.json").read_text(encoding="utf-8"))
    assert teardown["compute_zero_proven"] and teardown["provider_zero_proven"]
    # The whole scene half landed where the host compile would have written it, and the host never compiled it.
    assert world.compiles == ["worker"] and host.state() == "processing"
    read_team_scene_inputs(attempt / "scene" / RECORD_FILENAME)
    assert lane.markers(host.jobs, "authoritative") == []

    [finished] = _settled(host.process())
    assert (finished["status"], finished["outcome"], finished["blockers"]) == ("completed", "compiled", [])
    assert finished["attempt_root"] == str(attempt) and host.state() == "processed"
    assert finished["team_scene_inputs_path"] == str(attempt / "scene" / RECORD_FILENAME)
    assert finished["remote_compile"]["attempt_id"] == lease["attempt_id"]
    assert finished["offer"]["status"] is not None and world.compiles == ["worker"]
    profile = profile_for_capture(host.world["capture"])
    assert profile["team_scene_inputs_path"] == finished["team_scene_inputs_path"]
    assert profile["scene_revision_digest"] == host.world["scene"]["revision"]["revision_digest"]
    # Every marker and envelope is consumed; no URL or credential reached host disk.
    for kind in ("authoritative", "collected", "fallback"):
        assert lane.markers(host.jobs, kind) == []
    assert not any((host.jobs / "envelopes" / "team_scene_packet").iterdir())
    written = _texts(host.jobs) + _texts(host.world["queue"])
    assert "X-Amz-" not in written and "https://" not in written


def test_infrastructure_failures_retry_once_then_the_host_compiles(tmp_path: Path, monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    world.remote.jobs.script("crash", "crash")
    host = world.host
    host.process()
    world.drive(until=lambda: bool(lane.markers(host.jobs, "fallback")) and world.terminal())
    lease = world.lease()
    assert (lease["state"], lease["attempt"]) == ("fallback_host", 2)
    assert len(list((world.remote.spend / "consumed").glob("remote-cpu-*.json"))) == 2
    assert all(row["settled_usd"] <= TEAM_WORST_CASE_USD for row in _settlements(world))
    assert world.compiles == [] and host.state() == "processing"
    [status] = _settled(host.process())
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    assert status["remote_compile"]["mode"] == "fallback_host"
    assert world.compiles == ["host"] and lane.markers(host.jobs, "fallback") == []


def test_a_refused_admission_hands_the_request_back_before_anything_is_staged(tmp_path: Path, monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch, with_authority=False)
    host = world.host
    host.process()
    summary = world.collect()
    [row] = summary["rows"].values()
    assert row["status"] == "dispatch_refused" and row["blocker"] == "remote_cpu_standing_authority_missing"
    world.remote.assert_untouched()
    [(_, fallback)] = lane.markers(host.jobs, "fallback")
    assert fallback["reason"] == "remote_cpu_standing_authority_missing"
    [status] = _settled(host.process())
    assert status["status"] == "completed" and world.compiles == ["host"]


def test_a_withdrawn_capture_is_never_dispatched_or_published(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import website_atlas_preview_store

    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    [handed] = host.process()["results"]
    monkeypatch.setattr(website_atlas_preview_store, "capture_withdrawn", lambda capture: True)
    [row] = world.collect()["rows"].values()
    assert row == {"status": "returned_to_host", "reason": "team_scene_capture_withdrawn"}
    world.remote.assert_untouched()
    [status] = _settled(host.process())
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_capture_withdrawn"])
    assert world.compiles == [] and host.calls == [] and not Path(handed["attempt_root"]).exists()


def test_a_deterministic_remote_refusal_is_the_requests_own_blocked_status(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import team_scene_packet
    from blueprint_pipeline.team_scene_packet import TeamScenePacketError

    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    [handed] = host.process()["results"]
    original = team_scene_packet.materialize_team_scene

    def refuses(**kwargs):
        if world.in_worker:
            world.compiles.append("worker")
            raise TeamScenePacketError("team_scene_owner_contract_invalid:owner_contract_required")
        return original(**kwargs)

    monkeypatch.setattr(team_scene_packet, "materialize_team_scene", refuses)
    world.drive(until=lambda: world.terminal())
    assert world.lease()["state"] == "blocked"
    [(_, collected)] = lane.markers(host.jobs, "collected")
    assert collected["result"]["blockers"] == ["team_scene_owner_contract_invalid:owner_contract_required"]
    [status] = _settled(host.process())
    assert (status["status"], status["blockers"]) == (
        "blocked", ["team_scene_owner_contract_invalid:owner_contract_required"])
    # As after a host refusal: the attempt is removed and nothing is recorded or published.
    assert not Path(handed["attempt_root"]).exists() and profile_for_capture(host.world["capture"]) is None
    assert host.calls == [] and world.compiles == ["worker"]


def test_shadow_mode_compares_byte_parity_and_lands_nothing(tmp_path: Path, monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch, mode="cloud_run_shadow")
    host = world.host
    [status] = host.process()["results"]
    assert status["status"] == "completed" and status["remote_compile"]["shadowed"] is True
    scene = Path(status["attempt_root"]) / "scene"
    before = tree_snapshot(scene)
    world.drive(until=lambda: world.terminal())
    lease = world.lease()
    assert lease["state"] == "shadow_compared"
    parity = json.loads((host.jobs / "parity" / "team_scene_packet" / f"{lease['attempt_id']}.json").read_text(
        encoding="utf-8"))
    assert (parity["parity"], parity["mismatches"]) == ("passed", [])
    assert tree_snapshot(scene) == before and world.compiles == ["host", "worker"]
    assert lane.markers(host.jobs, "shadow") == [] and not any((host.jobs / "envelopes" / "team_scene_packet").iterdir())


@pytest.mark.parametrize("mode", ["host", "cloud_run_shadow"])
def test_a_rollback_hands_an_undispatched_request_back_to_the_host(tmp_path: Path, monkeypatch, mode: str) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    host.process()
    world.mode = mode
    [row] = world.collect()["rows"].values()
    assert row == {"status": "returned_to_host", "reason": "remote_cpu_mode_rolled_back"}
    world.remote.assert_untouched()
    [status] = _settled(host.process(host.dispatch("host")))
    assert status["status"] == "completed" and world.compiles == ["host"]


def test_without_a_provider_every_undispatched_hand_off_still_goes_back(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import team_scene_packet_collector as collector

    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    host.process()
    summary = collector.drain_without_provider(world.collector(), ["remote_cpu_config_invalid:missing"])
    [row] = summary["rows"].values()
    assert row["status"] == "returned_to_host"
    assert json.loads((host.jobs / collector.SUMMARY_NAME).read_text(encoding="utf-8"))["blockers"] == [
        "remote_cpu_config_invalid:missing"]
    world.remote.assert_untouched()
    assert _settled(host.process())[0]["status"] == "completed" and world.compiles == ["host"]
