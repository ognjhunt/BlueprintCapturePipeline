# Covers (for impacted-test selection):
#   src/blueprint_pipeline/team_scene_packet_collector.py
#   src/blueprint_pipeline/team_scene_packet_remote.py
#   src/blueprint_pipeline/team_scene_packet_worker.py
#   src/blueprint_pipeline/remote_cpu_job_contract.py
"""Recovery on the remote team-scene lane: kills, lost markers, consent, stale claims and unusable inputs.

The first two tests are the PR #3331 review's reproductions (a killed host fallback that looped, a collected
marker lost after the attempt was decided), kept as regressions.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import team_scene_packet
from blueprint_pipeline import team_scene_packet_collector as collector
from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline import website_atlas_preview_store
from tests.remote_team_scene_packet_support import TeamCollectorWorld, team_authority, team_config, team_stage


class Killed(BaseException):
    """Stands in for SIGKILL (the OOM killer, a unit timeout): no Python handler runs."""


def _consumed(world: TeamCollectorWorld) -> int:
    return len(list((world.remote.spend / "consumed").glob("remote-cpu-*.json")))


def _later(monkeypatch, hours: float) -> None:
    """Move the no-spend worker's clock past an automatic retry's backoff."""

    now = worker._utc_now()
    monkeypatch.setattr(worker, "_utc_now", lambda: now + timedelta(hours=hours))


def test_a_killed_host_fallback_is_an_interrupted_claim_with_its_backoff(tmp_path: Path, monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch, with_authority=False)
    host = world.host
    host.process()
    world.collect()
    [(_, fallback)] = lane.markers(host.jobs, "fallback")
    assert fallback["reason"] == "remote_cpu_standing_authority_missing"  # not temporary: the host compiles
    calls = []

    def killed(**kwargs):
        calls.append(Path(kwargs["output_root"]))
        Path(kwargs["output_root"], "partial.bin").write_bytes(b"x")
        raise Killed()

    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", killed)
    with pytest.raises(Killed):
        host.process()
    # The hand-back was recorded and its marker gone before the compile started.
    assert lane.markers(host.jobs, "fallback") == [] and host.state() == "processing"
    assert host.status()["remote_compile"]["mode"] == "fallback_host" and len(calls) == 1
    # The next run sees an ordinary interrupted claim: no immediate recompile, the backoff, the attempt removed.
    run = host.process()
    [recovered] = run["recovered"]
    assert recovered["blockers"] == ["team_scene_packet_interrupted"] and recovered["automatic_retry"]["not_before"]
    assert recovered["remote_compile"]["mode"] == "fallback_host"
    assert len(calls) == 1 and not calls[0].exists() and host.state() == "processed"
    assert host.process()["results"] == [] and len(calls) == 1
    # Once the backoff passes the retry is planned again, and goes remote, not straight to the host.
    _later(monkeypatch, 1)
    [replanned] = host.process()["results"]
    assert replanned["status"] == "processing" and replanned["remote_compile"]["mode"] == "cloud_run"
    assert len(calls) == 1 and len(lane.markers(host.jobs, "authoritative")) == 1


def test_a_lost_collected_marker_is_delivered_again_and_the_compile_is_never_paid_twice(tmp_path: Path,
                                                                                       monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    [handed] = host.process()["results"]
    attempt = Path(handed["attempt_root"])
    original, failed = lane.write_collected, []

    def full_disk(*args, **kwargs):
        if not failed:
            failed.append(1)
            raise OSError(28, "No space left on device")
        return original(*args, **kwargs)

    monkeypatch.setattr(lane, "write_collected", full_disk)
    world.drive(until=lambda: bool(failed) and world.terminal())
    assert world.lease()["state"] == "completed" and len(lane.markers(host.jobs, "collected")) == 1
    [settled] = host.process()["remote_settled"]
    assert (settled["status"], settled["outcome"]) == ("completed", "compiled")
    assert settled["attempt_root"] == str(attempt) and (attempt / "scene").is_dir()
    assert _consumed(world) == 1 and world.compiles == ["worker"]


def test_a_temporary_hand_back_is_planned_again_after_the_backoff_then_the_host_compiles(tmp_path: Path,
                                                                                        monkeypatch) -> None:
    # A daily cap below one worst case: every dispatch is refused, by a cap that rolls over.
    world = TeamCollectorWorld(tmp_path, monkeypatch, authority=team_authority(max_daily_usd=0.5))
    host = world.host
    host.process()
    for hours in (1, 2, 4):
        world.collect()
        [(_, fallback)] = lane.markers(host.jobs, "fallback")
        assert fallback["reason"] == "remote_cpu_daily_cap_exceeded"
        [deferred] = host.process()["remote_settled"]
        assert (deferred["status"], deferred["blockers"]) == ("blocked", ["team_scene_remote_compile_deferred"])
        assert world.compiles == [] and host.state() == "processed"
        _later(monkeypatch, hours)
        [replanned] = host.process()["results"]
        assert replanned["status"] == "processing"  # handed off again, not compiled
    world.remote.assert_untouched()
    # Its automatic retries spent, the next hand-back compiles on the host.
    world.collect()
    [status] = host.process()["remote_settled"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled") and world.compiles == ["host"]


@pytest.mark.parametrize("when", ["running", "collecting"])
def test_consent_withdrawn_while_an_attempt_runs_stops_it_and_lands_nothing(tmp_path: Path, monkeypatch,
                                                                            when: str) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    [handed] = host.process()["results"]
    attempt = Path(handed["attempt_root"])
    world.collect()
    if when == "collecting":
        world.drive(until=lambda: world.lease()["state"] == "collecting")
    else:
        assert world.lease()["state"] == "dispatched"
    monkeypatch.setattr(website_atlas_preview_store, "capture_withdrawn", lambda capture: True)
    world.drive(until=lambda: world.terminal())
    lease = world.lease()
    assert (lease["state"], lease["outcome"]) == ("fallback_host", "team_scene_capture_withdrawn")
    assert not (attempt / "scene").exists() and lane.markers(host.jobs, "collected") == []
    if when == "running":
        assert any(request.url.endswith(":cancel") for request in world.remote.rest.mutations)
        assert world.compiles == []
    [status] = host.process()["remote_settled"]
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_capture_withdrawn"])
    assert host.calls == [] and not attempt.exists()


def test_a_marker_for_another_attempt_never_settles_the_request(tmp_path: Path, monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    host.process()
    path, marker = host.marker("authoritative")
    # The request is claimed again for a newer attempt (planned again after a deferral, say).
    status_path = host.world["queue"] / "status" / f"{host.world['key']}.json"
    newer = Path(marker["plan"]["attempt_root"]).with_name("0123456789abcdef-01234567")
    status_path.write_text(json.dumps({**host.status(), "attempt_root": str(newer)}), encoding="utf-8")
    [row] = world.collect()["rows"].values()
    assert row == {"status": "returned_to_host", "reason": "remote_cpu_row_not_claimed"}
    assert lane.markers(host.jobs, "fallback") == [] and not path.exists()
    world.remote.assert_untouched()
    # A stale fallback or collected marker on the worker's side is quarantined, and the request is left alone.
    lane.write_fallback(host.jobs, marker, reason="remote_cpu_attempts_exhausted", attempts=2, now=1.0)
    run = host.process()
    [quarantined] = run["remote_quarantined"]
    assert (quarantined["kind"], quarantined["reason"]) == ("fallback", "team_scene_packet_marker_stale")
    assert run["remote_settled"] == [] and lane.markers(host.jobs, "fallback") == []
    assert (host.jobs / "quarantine" / "team_scene_packet" / quarantined["quarantined_as"]).is_file()
    assert world.compiles == []


def test_an_unreadable_marker_is_quarantined_so_the_path_unit_cannot_loop(tmp_path: Path, monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    directory = host.jobs / "collected" / "team_scene_packet"
    directory.mkdir(parents=True)
    (directory / "garbage.json").write_text("{not json", encoding="utf-8")
    run = host.process()
    assert [row["reason"] for row in run["remote_quarantined"]] == ["team_scene_packet_marker_unreadable"]
    assert not any(directory.iterdir())


def test_a_planner_failure_compiles_on_the_host_and_a_failed_hand_off_leaves_no_envelope(tmp_path: Path,
                                                                                       monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    envelopes = host.jobs / "envelopes" / "team_scene_packet"

    def broken(**_):
        raise ValueError("census unavailable")

    monkeypatch.setattr(lane, "plan_remote_team_scene", broken)
    [status] = host.process()["results"]
    assert (status["status"], status["remote_compile"]["host_decision"]) == (
        "completed", "remote_ineligible:planner_failed:ValueError")
    assert world.compiles == ["host"] and not (envelopes.is_dir() and any(envelopes.iterdir()))


def test_a_hand_off_that_cannot_be_written_takes_its_envelope_and_attempt_with_it(tmp_path: Path,
                                                                                monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(lane, "write_handoff", full)
    [status] = host.process()["results"]
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_packet_failed:OSError"])
    assert worker.transient_cause(status) == "team_scene_packet_failed:OSError"
    envelopes = host.jobs / "envelopes" / "team_scene_packet"
    halves = host.world["halves"]
    assert not any(envelopes.iterdir()) and not (halves.is_dir() and any(halves.iterdir()))


def test_a_landing_root_the_unit_cannot_write_hands_back_before_anything_is_staged(tmp_path: Path,
                                                                                 monkeypatch) -> None:
    world = TeamCollectorWorld(tmp_path, monkeypatch)
    host = world.host
    host.process()
    _, marker = host.marker("authoritative")
    plan = lane.RemotePlan.from_record(marker["plan"])
    real_mkdir = Path.mkdir

    def read_only(self, *args, **kwargs):
        if self.name.startswith(".landing-probe-"):
            raise OSError(30, "Read-only file system")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", read_only)
    assert collector._landing_refusal(plan) == "remote_cpu_landing_root_unwritable"
    [row] = world.collect()["rows"].values()
    assert row == {"status": "returned_to_host", "reason": "remote_cpu_landing_root_unwritable"}
    world.remote.assert_untouched()
    monkeypatch.setattr(Path, "mkdir", real_mkdir)
    assert collector._landing_refusal(lane.RemotePlan.from_record(
        {**marker["plan"], "attempt_root": str(tmp_path / "absent")})) == "remote_cpu_landing_root_missing"


def test_landing_holds_its_own_disk_reservation(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import control_plane_disk_budget

    world = TeamCollectorWorld(tmp_path, monkeypatch)
    world.host.process()
    taken, released = [], []

    class Reservation:
        def release(self, *, outcome: str) -> None:
            released.append(outcome)

    def reserve(role, **kwargs):
        taken.append((role, kwargs["workload"], kwargs["expected_bytes"], kwargs["target_root"]))
        return Reservation()

    monkeypatch.setattr(control_plane_disk_budget, "reserve_control_plane_disk", reserve)
    ledger = world.host.local("/var/lib/blueprint/pipeline-control-plane/disk-reservations")
    original = world.collector
    world.collector = lambda **changes: original(disk_reservation_root=ledger, **changes)
    world.drive(until=lambda: world.terminal())
    assert world.lease()["state"] == "completed"
    [(role, workload, expected, target)] = taken
    assert (role, workload, released) == (collector.DISK_ROLE, "team_scene_landing", ["completed"])
    assert expected > 0 and Path(target) == Path(world.host.status()["attempt_root"])


def test_an_invalid_team_entry_disables_only_the_team_stage(tmp_path: Path) -> None:
    from blueprint_pipeline import remote_cpu_job_allocator as allocator
    from blueprint_pipeline import task_evaluation_episode_compilation_remote as episode
    from tests.remote_cpu_allocator_fakes import seal

    config = team_config()
    config["stages"]["team_scene_packet"] = team_stage(task_timeout_seconds=7200)  # past the contract maximum
    config = seal(config, "config_digest")
    assert contract.config_blockers(config, stage="episode_compilation") == []
    assert contract.config_blockers(config, stage="team_scene_packet") == ["remote_cpu_config_invalid:stages"]
    # Unscoped, as an install-time check reads it, the entry to fix is named.
    assert contract.config_blockers(config) == ["remote_cpu_config_invalid:stage:team_scene_packet",
                                                "remote_cpu_config_invalid:stages"]
    path = tmp_path / "remote-cpu-workers.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    path.chmod(0o640)
    environ = {episode.CONFIG_ENV: str(path)}
    assert episode.remote_configured(environ) is True and episode.resolve_execution_mode(environ)["effective"] == (
        "cloud_run")
    assert lane.remote_configured(environ) is False and lane.resolve_execution_mode(environ)["effective"] == "host"
    assert allocator.load_remote_cpu_config(path, stage="episode_compilation")[1] == []
    assert allocator.load_remote_cpu_config(path, stage="team_scene_packet")[1] == ["remote_cpu_config_invalid:stages"]
