"""The root runner: revalidate spooled requests, then run fixed commands only."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/spool_runner.py

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult  # noqa: E402
from operator_door import holds  # noqa: E402
from operator_door.requests import SCHEMA, enqueue, validate_request  # noqa: E402
from operator_door import spool_runner  # noqa: E402
from operator_door.spool_runner import process_spool  # noqa: E402
from tests.test_operator_door_requests import _canonical_provenance  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeRunner:
    guard_root = Path("/var/lib/blueprint-operator-door/requests/holds")

    def __init__(self, active_deploy: str = "", systemd_run_rc: int = 0,
                 enabled_state: str = "enabled", guarded: bool = True,
                 need_daemon_reload: str = "no", active_state: str = "inactive",
                 stop_rc: int = 0, initial_active_state: str = "active") -> None:
        self.calls: list[list[str]] = []
        self.active_deploy = active_deploy
        self.systemd_run_rc = systemd_run_rc
        self.enabled_state = enabled_state
        self.guarded = guarded
        self.need_daemon_reload = need_daemon_reload
        self.active_state = active_state
        self.stop_rc = stop_rc
        self.initial_active_state = initial_active_state
        self.stop_count = 0

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        self.calls.append(list(argv))
        if argv[:2] == ["systemctl", "list-units"]:
            return CommandResult(0, self.active_deploy, "")
        if argv[:2] == ["systemctl", "is-enabled"]:
            return CommandResult(0 if self.enabled_state == "enabled" else 1,
                                 self.enabled_state + "\n", "")
        if argv[:2] == ["systemctl", "is-active"]:
            state = self.initial_active_state if self.stop_count == 0 else self.active_state
            return CommandResult(0 if state == "active" else 3, state + "\n", "")
        if argv[:2] == ["systemctl", "cat"]:
            unit = argv[-1]
            guard = f"ConditionPathExists=!{self.guard_root / f'{unit}.json'}"
            return CommandResult(0, f"[Unit]\n{guard}\n" if self.guarded else "[Unit]\n", "")
        if argv[:2] == ["systemctl", "show"] and "--property=ActiveState,SubState,Result,InvocationID,ExecMainStatus,Type" in argv:
            return CommandResult(0, "ActiveState=active\nSubState=running\nInvocationID=fixture-invocation\n", "")
        if argv[:2] == ["systemctl", "show"]:
            return CommandResult(0, self.need_daemon_reload + "\n", "")
        if argv[:2] == ["systemctl", "stop"]:
            self.stop_count += 1
            return CommandResult(self.stop_rc, "", "stop failed" if self.stop_rc else "")
        if argv[0] == "systemd-run":
            return CommandResult(self.systemd_run_rc, "", "Failed to start" if self.systemd_run_rc else "")
        return CommandResult(0, "", "")


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DoorConfig:
    for state in ("pending", "processing", "completed", "results"):
        (tmp_path / "requests" / state).mkdir(parents=True)
    config = DoorConfig(state_root=str(tmp_path), install_root="/opt/blueprint/operator-door")
    monkeypatch.setattr(FakeRunner, "guard_root", Path(config.spool_root) / "holds")
    return config


def _result(config: DoorConfig, request_id: str) -> dict:
    return json.loads((Path(config.spool_root) / "results" / f"{request_id}.json").read_text(encoding="utf-8"))


def _spooled(config: DoorConfig, body: dict) -> str:
    return enqueue(config, validate_request(body), requested_by="cloud")


def test_deploy_launches_the_door_deploy_script_with_validated_parameters(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA, "wait_for_idle": False})
    runner = FakeRunner()
    assert process_spool(config, runner=runner) == 0
    launch = [call for call in runner.calls if call[0] == "systemd-run"][0]
    unit = f"blueprint-operator-door-deploy-{SHA[:12]}-{request_id[-8:]}.service"
    assert launch[:6] == ["systemd-run", f"--unit={unit}", "--collect", "--service-type=exec",
                          "--property=TimeoutStartSec=3h", "--setenv=PYTHONDONTWRITEBYTECODE=1"]
    # TimeoutStartSec does not bound an exec unit once it has started; RuntimeMaxSec does.
    assert "--property=RuntimeMaxSec=3h" in launch
    assert launch[-2:] == ["/bin/bash", "/opt/blueprint/operator-door/door-deploy.sh"]
    env = dict(part.removeprefix("--setenv=").split("=", 1) for part in launch if part.startswith("--setenv="))
    assert env["DOOR_REQUEST_ID"] == request_id and env["DOOR_COMMIT"] == SHA
    assert "DOOR_MODE" not in env and env["DOOR_WAIT_FOR_IDLE"] == "0"
    assert env["DOOR_SOURCE_CLONE"] == DoorConfig().source_clone
    assert env["DOOR_GITHUB_KEY"] == DoorConfig().github_deploy_key
    assert env["DOOR_GITHUB_KNOWN_HOSTS"] == DoorConfig().github_known_hosts
    assert env["DOOR_RESULTS_DIR"] == str(Path(config.spool_root) / "results")
    result = _result(config, request_id)
    assert result["status"] == "launched" and result["unit"] == unit
    assert (Path(config.spool_root) / "completed" / f"{request_id}.json").exists()
    assert not list((Path(config.spool_root) / "pending").glob("*.json"))


ACTIVE_DEPLOY = "blueprint-fable-deploy-2c212fe5.service loaded active running x\n"
OLDER = "1111111111111111111111111111111111111111"
NEWER = "2222222222222222222222222222222222222222"


def _deploy_launches(runner: FakeRunner) -> list[list[str]]:
    return [call for call in runner.calls if call[0] == "systemd-run"]


def _launch_env(launch: list[str]) -> dict[str, str]:
    return dict(part.removeprefix("--setenv=").split("=", 1) for part in launch if part.startswith("--setenv="))


def _parked(config: DoorConfig) -> list[str]:
    return sorted(path.stem for path in (Path(config.spool_root) / "parked").glob("*.json"))


def _parked_record(config: DoorConfig, request_id: str) -> dict:
    return json.loads((Path(config.spool_root) / "parked" / f"{request_id}.json").read_text(encoding="utf-8"))


def test_a_canonical_deploy_is_still_refused_while_another_deploy_runs(config: DoorConfig) -> None:
    """A deploy with release provenance names one exact commit, so nothing may replace it."""
    payload = _canonical_provenance()
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA, "release_provenance_json": payload,
                                   "release_provenance_sha256": hashlib.sha256(payload.encode()).hexdigest()})
    runner = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    process_spool(config, runner=runner)
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    result = _result(config, request_id)
    assert result == {**result, "status": "refused", "code": "deploy_in_progress:blueprint-fable-deploy-2c212fe5.service"}
    assert _parked(config) == []


def test_a_deploy_requested_during_a_deploy_is_parked_not_refused(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    runner = FakeRunner(active_deploy=ACTIVE_DEPLOY)

    process_spool(config, runner=runner)

    assert _deploy_launches(runner) == []
    result = _result(config, request_id)
    assert result == {**result, "status": "parked", "commit": SHA,
                      "parked_behind": "blueprint-fable-deploy-2c212fe5.service"}
    assert _parked(config) == [request_id]
    assert not list((Path(config.spool_root) / "pending").glob("*.json"))


def test_the_parked_record_holds_the_commit_and_what_ran_when_it_was_asked_for(
    config: DoorConfig, tmp_path: Path,
) -> None:
    target = tmp_path / "releases" / NEWER
    link = tmp_path / "active"
    link.symlink_to(target)
    config = DoorConfig(state_root=config.state_root, install_root=config.install_root, active_release_link=str(link))
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA, "wait_for_idle": False})

    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))

    record = _parked_record(config, request_id)
    assert record["commit"] == SHA and record["wait_for_idle"] is False
    assert record["active_release_at_park"] == NEWER
    assert (Path(config.spool_root) / "parked" / f"{request_id}.json").stat().st_mode & 0o022 == 0


def test_a_commit_rewritten_in_the_doors_own_file_never_reaches_the_root_unit(config: DoorConfig) -> None:
    """completed/ is the door account's file, so it is never read back to choose a commit."""
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    document_path = Path(config.spool_root) / "completed" / f"{request_id}.json"
    document = json.loads(document_path.read_text())
    document["request"]["commit"] = NEWER
    document["request"]["wait_for_idle"] = False
    document_path.write_text(json.dumps(document))

    runner = FakeRunner()
    process_spool(config, runner=runner)

    env = _launch_env(_deploy_launches(runner)[0])
    assert env["DOOR_COMMIT"] == SHA and env["DOOR_WAIT_FOR_IDLE"] == "1"


@pytest.mark.parametrize("deploying", [True, False])
def test_an_unreadable_unit_list_never_reads_as_nothing_deploying(config: DoorConfig, deploying: bool) -> None:
    """A timed-out systemctl returned empty output, which used to mean a free host."""
    class Unreadable(FakeRunner):
        def run(self, argv, timeout):
            if argv[:2] == ["systemctl", "list-units"]:
                self.calls.append(list(argv))
                return CommandResult(124, "", "timed out")
            return super().run(argv, timeout)

    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    if deploying:
        process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    runner = Unreadable()

    process_spool(config, runner=runner)

    assert _deploy_launches(runner) == []
    assert _parked(config) == [request_id]


def test_an_unreadable_unit_list_refuses_a_canonical_deploy(config: DoorConfig) -> None:
    class Unreadable(FakeRunner):
        def run(self, argv, timeout):
            if argv[:2] == ["systemctl", "list-units"]:
                return CommandResult(1, "", "bus error")
            return super().run(argv, timeout)

    payload = _canonical_provenance()
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA, "release_provenance_json": payload,
                                   "release_provenance_sha256": hashlib.sha256(payload.encode()).hexdigest()})

    process_spool(config, runner=Unreadable())

    assert _result(config, request_id)["code"].startswith("deploy_in_progress:unknown")


def test_the_parked_deploy_starts_once_the_running_one_ends(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA, "wait_for_idle": False})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))

    runner = FakeRunner()
    process_spool(config, runner=runner)

    launch = _deploy_launches(runner)[0]
    assert launch[-1] == "/opt/blueprint/operator-door/door-deploy.sh"
    env = _launch_env(launch)
    assert env["DOOR_REQUEST_ID"] == request_id and env["DOOR_COMMIT"] == SHA
    assert env["DOOR_WAIT_FOR_IDLE"] == "0"
    assert env["DOOR_PARKED"] == "1"
    assert env["DOOR_PARKED_DIR"] == str(Path(config.spool_root) / "parked")
    assert env["DOOR_ACTIVE_RELEASE_LINK"] == config.active_release_link
    # A boot without a network yet fails its fetch and stays parked, so nothing waits for one.
    assert not any("network-online" in argument for argument in launch)
    assert _result(config, request_id)["status"] == "launched"


def test_a_deploy_requested_directly_is_never_marked_parked(config: DoorConfig) -> None:
    """The rollback guard reads this mark. An explicit rollback must stay possible."""
    _spooled(config, {"kind": "deploy", "commit": SHA})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    env = _launch_env(_deploy_launches(runner)[0])
    assert "DOOR_PARKED" not in env and "DOOR_PARKED_DIR" not in env


def test_the_record_stays_until_the_deploy_settles_it_so_a_killed_launch_starts_again(config: DoorConfig) -> None:
    """Power loss or SIGKILL before settling leaves no outcome. The record is what retries."""
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))

    first = FakeRunner()
    process_spool(config, runner=first)
    assert _parked(config) == [request_id]
    # While that unit runs, nothing starts a second one.
    busy = FakeRunner(active_deploy=f"blueprint-operator-door-deploy-x-{request_id[-8:]}.service loaded active running x\n")
    process_spool(config, runner=busy)
    assert _deploy_launches(busy) == []
    # It died without settling: the unit is gone and the record is not.
    again = FakeRunner()
    process_spool(config, runner=again)
    assert _launch_env(_deploy_launches(again)[0])["DOOR_REQUEST_ID"] == request_id


def test_the_newest_parked_request_starts_and_the_others_keep_their_records(config: DoorConfig) -> None:
    active = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    first = _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=active)
    time.sleep(1.1)  # request ids sort by their one-second timestamp
    second = _spooled(config, {"kind": "deploy", "commit": NEWER})
    process_spool(config, runner=active)
    assert _parked(config) == [first, second]

    runner = FakeRunner()
    process_spool(config, runner=runner)

    launches = _deploy_launches(runner)
    assert len(launches) == 1
    assert _launch_env(launches[0])["DOOR_REQUEST_ID"] == second
    assert _result(config, first)["status"] == "parked"
    assert _parked(config) == [first, second]


def test_a_failed_launch_fails_nobody_and_the_next_wake_up_tries_again(config: DoorConfig) -> None:
    """Nothing about a request made systemd-run fail, so an outage must not use up the queue."""
    active = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    first = _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=active)
    time.sleep(1.1)
    second = _spooled(config, {"kind": "deploy", "commit": NEWER})
    process_spool(config, runner=active)

    process_spool(config, runner=FakeRunner(systemd_run_rc=1))

    assert _result(config, second)["status"] == "parked"
    assert _result(config, first)["status"] == "parked"
    assert _parked(config) == [first, second]

    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == second
    assert _result(config, second)["status"] == "launched"


def test_the_runner_never_needs_git_or_a_network_to_park(config: DoorConfig) -> None:
    """Its unit has PrivateNetwork=true and no capabilities; parking must not change that."""
    _spooled(config, {"kind": "deploy", "commit": OLDER})
    busy = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    process_spool(config, runner=busy)
    idle = FakeRunner()
    process_spool(config, runner=idle)
    for call in (*busy.calls, *idle.calls):
        assert call[0] in {"systemctl", "systemd-run"}, call
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-operator-door-runner.service").read_text()
    assert "PrivateNetwork=true" in unit
    assert "CapabilityBoundingSet=\n" in unit
    assert "ReadWritePaths=/var/lib/blueprint-operator-door/requests\n" in unit


def test_parking_is_bounded_and_refuses_past_the_bound(config: DoorConfig, monkeypatch) -> None:
    monkeypatch.setattr(spool_runner, "_PARKED_MAX", 1)
    first = _spooled(config, {"kind": "deploy", "commit": OLDER})
    runner = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    process_spool(config, runner=runner)
    time.sleep(1.1)
    second = _spooled(config, {"kind": "deploy", "commit": NEWER})
    process_spool(config, runner=runner)

    assert _parked(config) == [first]
    assert _result(config, second)["code"] == "deploy_in_progress:blueprint-fable-deploy-2c212fe5.service"


def _ages_in_hours(monkeypatch, hours: float) -> None:
    real = time.time
    monkeypatch.setattr(spool_runner.time, "time", lambda: real() + hours * 3600)


def test_a_request_as_old_as_the_longest_wait_can_still_start(config: DoorConfig, monkeypatch) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    _ages_in_hours(monkeypatch, 7.9)

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == request_id


def test_a_request_nobody_can_still_be_waiting_for_ends_and_the_newer_one_goes_on(config: DoorConfig, monkeypatch) -> None:
    active = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    old = _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=active)
    path = Path(config.spool_root) / "parked" / f"{old}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "parked_at_epoch": time.time() - 8.5 * 3600}))
    time.sleep(1.1)
    new = _spooled(config, {"kind": "deploy", "commit": NEWER})
    process_spool(config, runner=active)

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _result(config, old)["status"] == "refused" and _result(config, old)["code"] == "deploy_parked_expired"
    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == new
    assert _parked(config) == [new]


def test_age_is_not_judged_while_a_deploy_runs(config: DoorConfig, monkeypatch) -> None:
    """Behind a long deploy it is still queued; the door sorts it out when it can act."""
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    _ages_in_hours(monkeypatch, 9)

    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))

    assert _result(config, request_id)["status"] == "parked" and _parked(config) == [request_id]


class _StartsUnits(FakeRunner):
    """A unit it starts shows as active afterwards, as systemd's does."""

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        result = super().run(argv, timeout)
        if argv[0] == "systemd-run":
            self.active_deploy = next(item.split("=", 1)[1] for item in argv if item.startswith("--unit="))
        return result


def test_a_new_direct_deploy_is_not_kept_behind_a_queue_that_cannot_start(config: DoorConfig) -> None:
    """A stuck queue's unit comes and goes each minute. It may be this deploy that repairs what stuck it."""
    parked = _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    direct = _spooled(config, {"kind": "deploy", "commit": NEWER})

    runner = _StartsUnits()
    process_spool(config, runner=runner)

    launches = _deploy_launches(runner)
    assert len(launches) == 1
    env = _launch_env(launches[0])
    assert env["DOOR_REQUEST_ID"] == direct and "DOOR_PARKED" not in env
    assert _result(config, direct)["status"] == "launched"
    assert _parked(config) == [parked]


def test_a_canonical_deploy_is_not_refused_because_a_queue_exists(config: DoorConfig) -> None:
    parked = _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    payload = _canonical_provenance()
    canonical = _spooled(config, {"kind": "deploy", "commit": SHA, "release_provenance_json": payload,
                                  "release_provenance_sha256": hashlib.sha256(payload.encode()).hexdigest()})

    runner = _StartsUnits()
    process_spool(config, runner=runner)

    assert _result(config, canonical)["status"] == "launched"
    assert _parked(config) == [parked]


def test_a_record_whose_outcome_exists_is_gone_without_another_deploy(config: DoorConfig) -> None:
    """Killed between writing the outcome and removing the record: the client already has its answer."""
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    (Path(config.spool_root) / "results" / f"{request_id}.outcome.json").write_text('{"status": "failed"}')

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _deploy_launches(runner) == [] and _parked(config) == []
    assert _result(config, request_id)["status"] == "parked"  # no second answer written over it


def test_an_empty_or_corrupt_outcome_is_no_answer_so_the_record_still_ends_the_request(config: DoorConfig) -> None:
    """Power loss can leave one. Removing the record on it would leave nothing to end the request."""
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    (Path(config.spool_root) / "results" / f"{request_id}.outcome.json").write_text("")

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == request_id
    assert _parked(config) == [request_id]


def test_abandoned_temporary_files_in_the_queue_are_cleared(config: DoorConfig) -> None:
    parked = Path(config.spool_root) / "parked"
    parked.mkdir(exist_ok=True)
    old, fresh = parked / ".x.1.tmp", parked / ".y.2.tmp"
    old.write_text("x")
    fresh.write_text("x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))

    process_spool(config, runner=FakeRunner())

    assert not old.exists() and fresh.exists()


def test_nothing_in_the_queue_is_touched_while_a_deploy_runs(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    bad = Path(config.spool_root) / "parked" / "20260101T000000Z-deploy-0000bad0.json"
    bad.write_text("not json")

    runner = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    process_spool(config, runner=runner)

    assert bad.exists() and _deploy_launches(runner) == []
    assert _result(config, request_id)["status"] == "parked"


def test_a_deploy_killed_outright_is_reported_and_the_others_go_on(config: DoorConfig) -> None:
    active = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    first = _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=active)
    time.sleep(1.1)
    second = _spooled(config, {"kind": "deploy", "commit": NEWER})
    process_spool(config, runner=active)
    path = Path(config.spool_root) / "parked" / f"{second}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "deploying_commit": NEWER}))

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _result(config, second)["status"] == "refused"
    assert _result(config, second)["code"] == "parked_deploy_interrupted"
    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == first
    assert _parked(config) == [first]


def test_a_killed_deploy_starts_no_second_deploy_by_itself(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    path = Path(config.spool_root) / "parked" / f"{request_id}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "deploying_commit": SHA}))

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _deploy_launches(runner) == []
    assert _parked(config) == []


def test_the_deploy_drains_if_any_parked_request_asked_it_to(config: DoorConfig) -> None:
    active = FakeRunner(active_deploy=ACTIVE_DEPLOY)
    _spooled(config, {"kind": "deploy", "commit": OLDER})
    process_spool(config, runner=active)
    time.sleep(1.1)
    _spooled(config, {"kind": "deploy", "commit": NEWER, "wait_for_idle": False})
    process_spool(config, runner=active)

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _launch_env(_deploy_launches(runner)[0])["DOOR_WAIT_FOR_IDLE"] == "1"


def _one_good_request(config: DoorConfig) -> str:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    return request_id


@pytest.mark.parametrize("name", ["00000000T000000Z-deploy-0000bad0", "99999999T999999Z-deploy-0000bad0"])
@pytest.mark.parametrize("kind", ["directory", "huge_integer", "huge_float", "not_a_number", "deep", "fifo", "symlink",
                                  "oversize", "other_kind", "epoch_infinite", "epoch_nan", "epoch_huge", "epoch_bool",
                                  "epoch_missing"])
def test_one_bad_file_never_stops_the_requests_around_it(config: DoorConfig, kind: str, name: str) -> None:
    """Whichever side of a good record it sorts on, and whatever is wrong with it."""
    good = _one_good_request(config)
    parked = Path(config.spool_root) / "parked"
    if kind == "other_kind":
        name = name.replace("-deploy-", "-unit-")
    path = parked / f"{name}.json"
    base = {"schema": spool_runner._PARKED_SCHEMA, "id": name, "commit": SHA, "wait_for_idle": True,
            "parked_at_epoch": 1e12}
    if kind == "directory":
        path.mkdir()
        (path / "inside").write_text("x")
    elif kind == "huge_integer":
        path.write_text('{"schema": "x", "parked_at_epoch": ' + "9" * 5000 + "}")
    elif kind == "huge_float":
        path.write_text(json.dumps(base).replace('"wait_for_idle": true', '"wait_for_idle": 1e999'))
    elif kind == "not_a_number":
        path.write_text(json.dumps(base).replace('"wait_for_idle": true', '"wait_for_idle": NaN'))
    elif kind == "deep":
        path.write_text("[" * 60_000)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to(parked / f"{good}.json")
    elif kind == "oversize":
        path.write_text(json.dumps({**base, "padding": "x" * (2 * 1024 * 1024)}))
    elif kind == "epoch_infinite":
        path.write_text(json.dumps(base).replace("1000000000000.0", "1e999"))
    elif kind == "epoch_nan":
        path.write_text(json.dumps(base).replace("1000000000000.0", "NaN"))
    elif kind == "epoch_huge":
        path.write_text(json.dumps(base).replace("1000000000000.0", "1" + "0" * 400))
    elif kind == "epoch_bool":
        path.write_text(json.dumps({**base, "parked_at_epoch": True}))
    elif kind == "epoch_missing":
        path.write_text(json.dumps({key: value for key, value in base.items() if key != "parked_at_epoch"}))
    else:
        path.write_text(json.dumps(base))

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == good
    assert not path.exists() and not path.is_symlink()
    assert _parked(config) == [good]


def test_a_file_that_cannot_be_removed_is_kept_and_holds_nothing_up(config: DoorConfig, monkeypatch) -> None:
    good = _one_good_request(config)
    bad = Path(config.spool_root) / "parked" / "00000000T000000Z-deploy-0000bad0.json"
    bad.write_text("not json")
    unlink = Path.unlink

    def stuck(self, missing_ok=False):
        if self == bad:
            raise PermissionError(1, "stuck")
        return unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", stuck)

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _launch_env(_deploy_launches(runner)[0])["DOOR_REQUEST_ID"] == good
    assert bad.exists()


_BAD_RECORDS = {
    "not_json": "not json",
    "no_schema": json.dumps({"id": "x"}),
    "no_id": json.dumps({"schema": spool_runner._PARKED_SCHEMA}),
    "bad_commit": json.dumps({"schema": spool_runner._PARKED_SCHEMA, "id": "SELF",
                              "commit": "not-a-commit", "wait_for_idle": True}),
    "too_deep": "[" * 60_000,
}


@pytest.mark.parametrize("record", list(_BAD_RECORDS))
def test_a_bad_parked_record_is_refused_at_release(config: DoorConfig, record: str) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    (Path(config.spool_root) / "parked" / f"{request_id}.json").write_text(_BAD_RECORDS[record].replace("SELF", request_id))

    runner = FakeRunner()
    process_spool(config, runner=runner)

    assert _deploy_launches(runner) == []
    assert _result(config, request_id)["status"] == "refused"
    assert _parked(config) == []


def test_a_junk_name_in_the_parked_directory_cannot_stop_the_spool(config: DoorConfig) -> None:
    (Path(config.spool_root) / "parked").mkdir(exist_ok=True)
    (Path(config.spool_root) / "parked" / "junk.json").write_text("{}")
    request_id = _spooled(config, {"kind": "door-upgrade", "commit": SHA})

    process_spool(config, runner=FakeRunner())

    assert _parked(config) == []
    assert _result(config, request_id)["status"] == "launched"


def test_any_failure_with_parked_deploys_leaves_every_other_request_running(config: DoorConfig, monkeypatch) -> None:
    def broken(_config, _runner):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spool_runner, "_release_parked", broken)
    request_id = _spooled(config, {"kind": "door-upgrade", "commit": SHA})

    assert process_spool(config, runner=FakeRunner()) == 0
    assert _result(config, request_id)["status"] == "launched"


def test_a_deeply_nested_request_file_is_refused_not_a_crash(config: DoorConfig) -> None:
    from operator_door.requests import RequestRefused, load_request_file

    path = Path(config.spool_root) / "pending" / "deep.json"
    path.write_text("[" * 60_000)

    with pytest.raises(RequestRefused):
        load_request_file(path)


def test_the_hold_sweep_wakes_the_runner_only_while_a_deploy_is_parked(config: DoorConfig) -> None:
    """Nothing else notices a deploy ending."""
    idle = FakeRunner()
    assert spool_runner.wake_parked(config, runner=idle) == 0
    assert idle.calls == []

    _spooled(config, {"kind": "deploy", "commit": SHA})
    process_spool(config, runner=FakeRunner(active_deploy=ACTIVE_DEPLOY))
    woken = FakeRunner()
    assert spool_runner.wake_parked(config, runner=woken) == 0
    assert woken.calls == [["systemctl", "start", "--no-block", "blueprint-operator-door-runner.service"]]


def test_waking_with_no_parked_directory_yet_is_harmless(tmp_path: Path) -> None:
    """A sweep can run before the first spool pass after an upgrade made the directory."""
    config = DoorConfig(state_root=str(tmp_path), install_root="/opt/blueprint/operator-door")
    runner = FakeRunner()

    assert spool_runner.wake_parked(config, runner=runner) == 0
    assert runner.calls == []


def test_wake_parked_is_a_door_command() -> None:
    """The unit calls it by this name, so a rename must fail here and not on the host."""
    from operator_door import __main__ as door_main

    with pytest.raises(SystemExit) as stopped:
        door_main.main(["wake-parked", "--help"])
    assert stopped.value.code == 0


def test_failed_launch_is_recorded(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "door-upgrade", "commit": SHA})
    process_spool(config, runner=FakeRunner(systemd_run_rc=1))
    result = _result(config, request_id)
    assert result["status"] == "failed" and result["returncode"] == 1 and "Failed" in result["stderr_tail"]


def test_unit_actions_wait_for_systemctl_and_observe_the_objective(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "unit", "unit": "blueprint-pubsub-handoff-listener.timer",
                                   "action": "start"})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert ["systemctl", "start", "--", "blueprint-pubsub-handoff-listener.timer"] in runner.calls
    assert _result(config, request_id)["status"] == "done"


def test_notifier_repair_uses_fixed_sandbox_without_starting_target(config: DoorConfig) -> None:
    digest = "sha256:" + "a" * 64
    request_id = _spooled(config, {"kind": "unit", "unit": "blueprint-pipeline-control-plane.service",
                                  "action": "repair-notifier-binding", "expected_postcheck_sha256": digest,
                                  "expected_source_commit": SHA})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    launch = next(call for call in runner.calls if call[0] == "systemd-run")
    assert launch[-2:] == ["/bin/bash", "/opt/blueprint/operator-door/door-repair-notifier-binding.sh"]
    assert "--property=PrivateNetwork=yes" in launch
    assert "--property=CapabilityBoundingSet=" in launch
    assert "--property=RuntimeMaxSec=60s" in launch
    assert ("--property=ReadWritePaths=/etc/systemd/system/blueprint-pipeline-control-plane.service.d "
            + str(Path(config.spool_root) / "results")) in launch
    assert f"--setenv=DOOR_EXPECTED_POSTCHECK_SHA256={digest}" in launch
    assert not any(call[:2] == ["systemctl", "--no-block"] for call in runner.calls)
    assert _result(config, request_id)["status"] == "launched"


def test_notifier_repair_refuses_overlapping_deploy(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "unit", "unit": "blueprint-pipeline-control-plane.service",
                                  "action": "repair-notifier-binding", "expected_postcheck_sha256": "sha256:" + "a" * 64,
                                  "expected_source_commit": SHA})
    runner = FakeRunner(active_deploy="blueprint-operator-door-deploy-abc.service loaded active running x\n")
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "notifier_repair_deploy_in_progress"
    assert not any(call[0] == "systemd-run" for call in runner.calls)


def _hold(config: DoorConfig, owner: str = "alice") -> str:
    return _spooled(config, {"kind": "hold", "unit": "blueprint-scene-progression.timer",
                             "owner": owner, "reason": "inspect capture", "expires_in_seconds": 3600})


def test_explicit_dispatch_stop_cannot_change_owner_or_auto_expire_on_renewal(config: DoorConfig) -> None:
    unit = "blueprint-agent-run-dispatcher.timer"
    body = {"kind": "hold", "unit": unit, "owner": "founder-stop", "reason": "Founder stop",
            "expires_in_seconds": 60, "require_explicit_release": True}
    initial = _spooled(config, body)
    process_spool(config, runner=FakeRunner())
    assert _result(config, initial)["hold"]["require_explicit_release"] is True
    path = Path(config.spool_root) / "holds" / f"{unit}.json"
    # An overdue review deadline does not surrender stop ownership.
    record = {**json.loads(path.read_text()), "expires_at_epoch": 1}
    holds.write(path.parent, unit, record)
    for change, code in [({"owner": "other"}, "hold_active:founder-stop"),
                         ({"require_explicit_release": False}, "hold_explicit_release_required")]:
        request_id = _spooled(config, {**body, **change})
        runner = FakeRunner()
        process_spool(config, runner=runner)
        assert _result(config, request_id)["code"] == code
        assert runner.calls == []
        assert json.loads(path.read_text()) == record
    renewed = _spooled(config, body)
    process_spool(config, runner=FakeRunner())
    assert _result(config, renewed)["status"] == "done"
    assert json.loads(path.read_text())["require_explicit_release"] is True


def test_hold_stops_a_timer_records_owner_and_schedules_expiry(config: DoorConfig) -> None:
    request_id = _hold(config)
    runner = FakeRunner()
    process_spool(config, runner=runner)
    unit = "blueprint-scene-progression.timer"
    assert runner.calls[:3] == [
        ["systemctl", "cat", "--", unit],
        ["systemctl", "show", "--property=NeedDaemonReload", "--value", "--", unit],
        ["systemctl", "is-enabled", "--", unit],
    ]
    assert runner.calls[3:9] == [
        ["systemctl", "is-active", "--", unit],
        ["systemctl", "stop", "--", unit],
        ["systemctl", "is-active", "--", unit],
        ["systemctl", "stop", "--", unit],
        ["systemctl", "is-active", "--", unit],
        ["systemctl", "disable", "--", unit],
    ]
    launch = runner.calls[9]
    assert launch[0] == "systemd-run"
    assert f"--unit=blueprint-operator-door-hold-expiry-{request_id[-8:]}" in launch
    assert "--on-active=3600s" in launch and "--collect" in launch
    assert f"--setenv=DOOR_HOLD_UNIT={unit}" in launch
    assert f"--setenv=DOOR_HOLD_REQUEST_ID={request_id}" in launch
    assert launch[-2:] == ["/bin/bash", "/opt/blueprint/operator-door/door-hold-expire.sh"]
    record = Path(config.spool_root) / "holds" / f"{unit}.json"
    hold = json.loads(record.read_text(encoding="utf-8"))
    assert record.stat().st_mode & 0o777 == 0o644
    assert {key: hold[key] for key in ("schema", "unit", "owner", "reason", "requested_by", "request_id", "status")} == {
        "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice",
        "reason": "inspect capture", "requested_by": "cloud", "request_id": request_id, "status": "active"}
    assert hold["enabled_before"] is True
    assert hold["expires_at"] > hold["created_at"]
    assert _result(config, request_id)["status"] == "done"


def test_hold_publishes_its_boot_guard_before_stopping_or_disabling(config: DoorConfig) -> None:
    unit = "blueprint-scene-progression.timer"
    record = Path(config.spool_root) / "holds" / f"{unit}.json"

    class ObserveDisable(FakeRunner):
        stops = 0

        def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
            if argv[:2] == ["systemctl", "stop"]:
                self.stops += 1
                if self.stops == 1:
                    assert not record.exists()
                else:
                    assert json.loads(record.read_text())["status"] == "active"
            if argv[:2] == ["systemctl", "disable"]:
                assert json.loads(record.read_text())["status"] == "active"
            return super().run(argv, timeout)

    request_id = _hold(config)
    process_spool(config, runner=ObserveDisable())
    assert _result(config, request_id)["status"] == "done"
    assert json.loads(record.read_text())["status"] == "active"


def test_hold_does_not_succeed_while_the_trigger_is_still_active(config: DoorConfig) -> None:
    request_id = _hold(config)
    runner = FakeRunner(active_state="active")
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "hold_stop_incomplete"
    assert ["systemctl", "stop", "--", "blueprint-scene-progression.timer"] in runner.calls
    assert ["systemctl", "is-active", "--", "blueprint-scene-progression.timer"] in runner.calls
    assert not any(call[:2] == ["systemctl", "disable"] for call in runner.calls)
    assert not (Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json").exists()


@pytest.mark.parametrize("runner", [FakeRunner(active_state="unknown"), FakeRunner(stop_rc=1)])
def test_failed_first_stop_restores_an_unheld_trigger(config: DoorConfig, runner: FakeRunner) -> None:
    request_id = _hold(config)
    process_spool(config, runner=runner)
    result = _result(config, request_id)
    assert result["status"] == "failed" and result["rollback_returncode"] == 0
    assert ["systemctl", "enable", "--", "blueprint-scene-progression.timer"] in runner.calls
    assert ["systemctl", "--no-block", "start", "--", "blueprint-scene-progression.timer"] in runner.calls
    assert not (Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json").exists()


def test_failed_hold_does_not_start_a_previously_inactive_disabled_trigger(config: DoorConfig) -> None:
    request_id = _hold(config)
    runner = FakeRunner(enabled_state="disabled", initial_active_state="inactive", systemd_run_rc=1)
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "hold_expiry_schedule_failed"
    assert not any(call[:3] == ["systemctl", "--no-block", "start"] for call in runner.calls)
    assert not any(call[:2] == ["systemctl", "enable"] for call in runner.calls)
    assert not (Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json").exists()


def test_failed_hold_record_write_restores_a_stopped_trigger(config: DoorConfig,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    request_id = _hold(config)
    runner = FakeRunner()

    def no_space(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(holds, "write", no_space)
    process_spool(config, runner=runner)
    assert _result(config, request_id)["status"] == "unknown"
    assert ["systemctl", "enable", "--", "blueprint-scene-progression.timer"] in runner.calls
    assert ["systemctl", "--no-block", "start", "--", "blueprint-scene-progression.timer"] in runner.calls
    assert not (Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json").exists()


def test_hold_refuses_a_host_unit_without_a_durable_guard(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "hold", "unit": "blueprint-extra-job.timer",
                                   "owner": "alice", "reason": "inspect", "expires_in_seconds": 3600})
    runner = FakeRunner(guarded=False)
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "hold_unit_guard_missing"
    assert runner.calls == [["systemctl", "cat", "--", "blueprint-extra-job.timer"]]
    assert not (Path(config.spool_root) / "holds" / "blueprint-extra-job.timer.json").exists()


def test_hold_refuses_a_guard_for_a_different_state_root(config: DoorConfig) -> None:
    request_id = _hold(config)
    runner = FakeRunner()
    runner.guard_root = Path("/var/lib/blueprint-operator-door/requests/holds")
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "hold_unit_guard_missing"
    assert runner.calls == [["systemctl", "cat", "--", "blueprint-scene-progression.timer"]]


@pytest.mark.parametrize("reload_state", ["yes", "unknown"])
def test_hold_refuses_when_systemd_has_stale_loaded_unit(config: DoorConfig, reload_state: str) -> None:
    request_id = _hold(config)
    runner = FakeRunner(need_daemon_reload=reload_state)
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "hold_unit_guard_missing"
    assert [call[:2] for call in runner.calls] == [["systemctl", "cat"], ["systemctl", "show"]]


def test_hold_refuses_a_unit_whose_drop_in_resets_its_guard(config: DoorConfig) -> None:
    request_id = _hold(config)

    class ResetGuard(FakeRunner):
        def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
            result = super().run(argv, timeout)
            if argv[:2] == ["systemctl", "cat"]:
                return CommandResult(0, result.stdout + "[Unit]\nConditionPathExists=\n", "")
            return result

    runner = ResetGuard()
    process_spool(config, runner=runner)
    assert _result(config, request_id)["code"] == "hold_unit_guard_missing"
    assert runner.calls == [["systemctl", "cat", "--", "blueprint-scene-progression.timer"]]


def test_other_owner_cannot_replace_an_active_hold_but_same_owner_can_extend(config: DoorConfig) -> None:
    first = _hold(config)
    process_spool(config, runner=FakeRunner())
    other = _hold(config, "bob")
    refused_runner = FakeRunner()
    process_spool(config, runner=refused_runner)
    assert _result(config, other)["status"] == "refused"
    assert _result(config, other)["code"] == "hold_active:alice"
    assert refused_runner.calls == []
    extended = _hold(config)
    process_spool(config, runner=FakeRunner())
    record = Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json"
    assert json.loads(record.read_text())["request_id"] == extended
    assert _result(config, extended)["status"] == "done"
    assert first != extended


def test_same_owner_renewal_never_shortens_an_active_hold(config: DoorConfig) -> None:
    first = _spooled(config, {"kind": "hold", "unit": "blueprint-scene-progression.timer",
                              "owner": "alice", "reason": "inspect", "expires_in_seconds": 86400})
    process_spool(config, runner=FakeRunner())
    record = Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json"
    first_expiry = json.loads(record.read_text())["expires_at_epoch"]
    renewed = _spooled(config, {"kind": "hold", "unit": "blueprint-scene-progression.timer",
                                "owner": "alice", "reason": "still inspecting", "expires_in_seconds": 60})

    process_spool(config, runner=FakeRunner())

    saved = json.loads(record.read_text())
    assert saved["request_id"] == renewed != first
    assert saved["expires_at_epoch"] >= first_expiry


def test_release_hold_starts_the_timer_and_marks_the_record(config: DoorConfig) -> None:
    _hold(config)
    process_spool(config, runner=FakeRunner())
    release = _spooled(config, {"kind": "release-hold", "unit": "blueprint-scene-progression.timer"})
    marker = Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json"

    class ObserveStart(FakeRunner):
        def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
            if argv[:3] == ["systemctl", "--no-block", "start"]:
                assert not marker.exists(), "the active guard must be removed before start"
            return super().run(argv, timeout)

    runner = ObserveStart()
    process_spool(config, runner=runner)
    assert runner.calls == [
        ["systemctl", "enable", "--", "blueprint-scene-progression.timer"],
        ["systemctl", "--no-block", "start", "--", "blueprint-scene-progression.timer"],
    ]
    assert not marker.exists()
    archived = list((Path(config.spool_root) / "holds" / "history").glob("*.json"))
    assert len(archived) == 1
    hold = json.loads(archived[0].read_text())
    assert hold["status"] == "released" and hold["released_by"] == "cloud" and hold["released_at"]
    result = _result(config, release)
    assert result["status"] == "done"
    assert result["hold"]["released_at"] == hold["released_at"]
    assert result["hold"]["released_by"] == hold["released_by"]
    again = _spooled(config, {"kind": "release-hold", "unit": "blueprint-scene-progression.timer"})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert _result(config, again)["code"] == "hold_not_active"
    assert runner.calls == []


def test_release_preserves_a_previously_disabled_boot_policy(config: DoorConfig) -> None:
    request_id = _hold(config)
    process_spool(config, runner=FakeRunner(enabled_state="disabled"))
    record = Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json"
    assert json.loads(record.read_text())["enabled_before"] is False
    release = _spooled(config, {"kind": "release-hold", "unit": "blueprint-scene-progression.timer"})
    runner = FakeRunner(enabled_state="disabled")

    process_spool(config, runner=runner)

    assert _result(config, request_id)["status"] == "done"
    assert _result(config, release)["status"] == "done"
    assert ["systemctl", "enable", "--", "blueprint-scene-progression.timer"] not in runner.calls
    assert ["systemctl", "--no-block", "start", "--", "blueprint-scene-progression.timer"] in runner.calls


def test_failed_expiry_scheduling_restores_a_new_timer_but_preserves_an_existing_hold(config: DoorConfig) -> None:
    first = _hold(config)
    runner = FakeRunner(systemd_run_rc=1)
    process_spool(config, runner=runner)
    assert _result(config, first)["code"] == "hold_expiry_schedule_failed"
    assert [call[:3] for call in runner.calls] == [
        ["systemctl", "cat", "--"],
        ["systemctl", "show", "--property=NeedDaemonReload"],
        ["systemctl", "is-enabled", "--"],
        ["systemctl", "is-active", "--"],
        ["systemctl", "stop", "--"],
        ["systemctl", "is-active", "--"],
        ["systemctl", "stop", "--"],
        ["systemctl", "is-active", "--"],
        ["systemctl", "disable", "--"],
        ["systemd-run", runner.calls[9][1], "--on-active=3600s"],
        ["systemctl", "enable", "--"],
        ["systemctl", "--no-block", "start"],
    ]
    record = Path(config.spool_root) / "holds" / "blueprint-scene-progression.timer.json"
    assert not record.exists()
    archived = list((Path(config.spool_root) / "holds" / "history").glob("*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text())["status"] == "failed_released"

    existing = _hold(config)
    process_spool(config, runner=FakeRunner())
    renewed = _hold(config)
    runner = FakeRunner(systemd_run_rc=1)
    process_spool(config, runner=runner)
    assert _result(config, renewed)["code"] == "hold_expiry_schedule_failed"
    assert not any(call[:3] == ["systemctl", "--no-block", "start"] for call in runner.calls)
    saved = json.loads(record.read_text())
    assert saved["request_id"] == existing and saved["status"] == "active"


def test_door_upgrade_launches_the_upgrade_script(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "door-upgrade", "commit": SHA})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    launch = [call for call in runner.calls if call[0] == "systemd-run"][0]
    assert f"--unit=blueprint-operator-door-upgrade-{SHA[:12]}-{request_id[-8:]}.service" in launch
    assert "--property=TimeoutStartSec=30min" in launch and "--property=RuntimeMaxSec=30min" in launch
    assert launch[-1] == "/opt/blueprint/operator-door/door-upgrade.sh"
    env = dict(part.removeprefix("--setenv=").split("=", 1) for part in launch if part.startswith("--setenv="))
    assert env["DOOR_COMMIT"] == SHA and "DOOR_WAIT_FOR_IDLE" not in env  # only what the kind defines


def test_tampered_spool_files_are_refused_not_executed(config: DoorConfig) -> None:
    pending = Path(config.spool_root) / "pending"
    request_id = "20260923T000000Z-deploy-0000abcd"
    (pending / f"{request_id}.json").write_text(json.dumps({
        "schema": SCHEMA, "id": request_id,
        "request": {"kind": "deploy", "commit": SHA, "flags": "--arm-path-units"},
    }), encoding="utf-8")
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    assert _result(config, request_id)["code"] == "request_key_unknown:flags"


def test_mismatched_ids_are_refused(config: DoorConfig) -> None:
    pending = Path(config.spool_root) / "pending"
    request_id = "20260923T000000Z-deploy-0000abcd"
    (pending / f"{request_id}.json").write_text(json.dumps({
        "schema": SCHEMA, "id": "20260923T000000Z-deploy-ffffffff",
        "request": {"kind": "deploy", "commit": SHA},
    }), encoding="utf-8")
    process_spool(config, runner=FakeRunner())
    assert _result(config, request_id)["code"] == "spool_id_mismatch"


def test_hold_with_another_kinds_id_is_refused_before_stopping_a_timer(config: DoorConfig) -> None:
    request_id = "20260923T000000Z-unit-0000abcd"
    pending = Path(config.spool_root) / "pending"
    (pending / f"{request_id}.json").write_text(json.dumps({
        "schema": SCHEMA, "id": request_id, "requested_by": "cloud",
        "request": {"kind": "hold", "unit": "blueprint-scene-progression.timer",
                    "owner": "alice", "reason": "inspect", "expires_in_seconds": 3600},
    }), encoding="utf-8")
    runner = FakeRunner()

    process_spool(config, runner=runner)

    assert _result(config, request_id)["code"] == "spool_kind_mismatch"
    assert runner.calls == []
    assert not (Path(config.spool_root) / "holds").exists()


def test_junk_files_are_drained_so_the_path_unit_cannot_loop(config: DoorConfig, tmp_path: Path) -> None:
    pending = Path(config.spool_root) / "pending"
    (pending / "junk.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "x.json"
    target.write_text("{}", encoding="utf-8")
    os.symlink(target, pending / "20260923T000000Z-deploy-0000abce.json")
    process_spool(config, runner=FakeRunner())
    assert not list(pending.glob("*.json"))
    assert _result(config, "20260923T000000Z-deploy-0000abce")["code"] == "spool_file_unsafe"


def test_results_are_world_readable_for_the_door(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "unit", "unit": "blueprint-gpu-spend-guard.service", "action": "start"})
    process_spool(config, runner=FakeRunner())
    path = Path(config.spool_root) / "results" / f"{request_id}.json"
    assert oct(path.stat().st_mode & 0o777) == oct(0o644)


def test_old_requests_and_results_are_pruned(config: DoorConfig) -> None:
    spool = Path(config.spool_root)
    old_request = spool / "completed" / "20250101T000000Z-deploy-00000001.json"
    old_log = spool / "results" / "20250101T000000Z-deploy-00000001.log"
    fresh = spool / "results" / "20260923T000000Z-deploy-00000002.json"
    for path in (old_request, old_log, fresh):
        path.write_text("{}", encoding="utf-8")
    for path in (old_request, old_log):
        os.utime(path, (1_000_000, 1_000_000))
    process_spool(config, runner=FakeRunner())
    assert not old_request.exists() and not old_log.exists() and fresh.exists()


def test_unreadable_and_late_files_are_drained(config: DoorConfig) -> None:
    spool = Path(config.spool_root)
    first = _spooled(config, {"kind": "unit", "unit": "blueprint-gpu-spend-guard.service", "action": "start"})
    locked = spool / "pending" / "20260923T000000Z-unit-0000beef.json"
    locked.write_text("{}", encoding="utf-8")
    locked.chmod(0o000)
    runner = FakeRunner()
    arrived: list[str] = []

    def late_arrival(argv, timeout):
        if argv[:2] == ["systemctl", "start"] and not arrived:
            arrived.append(_spooled(config, {"kind": "unit", "unit": "blueprint-gpu-spend-guard.service",
                                              "action": "reset-failed"}))
        return FakeRunner.run(runner, argv, timeout)

    runner.run = late_arrival  # type: ignore[method-assign]
    try:
        process_spool(config, runner=runner)
    finally:
        for path in (spool / "completed").glob("*beef*"):
            path.chmod(0o644)
    assert not list((spool / "pending").glob("*.json"))
    assert _result(config, first)["status"] == "done"
    assert sum(call[:2] == ["systemctl", "reset-failed"] for call in runner.calls) == 1


def test_stranded_claims_are_failed_after_an_hour(config: DoorConfig) -> None:
    spool = Path(config.spool_root)
    stranded = spool / "processing" / "20260923T000000Z-deploy-0000cafe.json"
    stranded.write_text("{}", encoding="utf-8")
    os.utime(stranded, (1_000_000, 1_000_000))
    process_spool(config, runner=FakeRunner())
    assert not stranded.exists()
    assert _result(config, "20260923T000000Z-deploy-0000cafe") == {
        **_result(config, "20260923T000000Z-deploy-0000cafe"), "status": "unknown", "code": "stranded"}


def test_every_launched_unit_is_a_named_service_the_door_can_inspect(config: DoorConfig) -> None:
    """``unit_properties`` accepts only ``.service`` names, so a bare name left ``unit_state`` null."""

    from operator_door.hostinfo import UNIT_NAME

    for body in ({"kind": "deploy", "commit": SHA}, {"kind": "door-upgrade", "commit": SHA}):
        request_id = _spooled(config, body)
        process_spool(config, runner=FakeRunner())
        assert UNIT_NAME.fullmatch(_result(config, request_id)["unit"])


@pytest.mark.parametrize(("body", "extra_env"), [
    ({"kind": "retire-scene-workspace", "scene_id": "site-capture-with-a-long-name-1"}, {}),
    ({"kind": "retire-scene-workspace", "scene_id": "s", "bucket": "blueprint-8c1ca.appspot.com", "apply": True},
     {"DOOR_BUCKET": "blueprint-8c1ca.appspot.com", "DOOR_APPLY": "1"}),
])
def test_retire_scene_workspace_launches_a_bounded_sandboxed_service(config: DoorConfig, body: dict,
                                                                   extra_env: dict) -> None:
    request_id = _spooled(config, body)
    runner = FakeRunner()
    process_spool(config, runner=runner)

    label = hashlib.sha256(body["scene_id"].encode("utf-8")).hexdigest()[:12]
    unit = f"blueprint-operator-door-retire-{label}-{request_id[-8:]}.service"
    results = str(Path(config.spool_root) / "results")
    env = {"DOOR_REQUEST_ID": request_id, "DOOR_RESULTS_DIR": results, "DOOR_VENV_PYTHON": DoorConfig().venv_python,
           "DOOR_SCENE_ID": body["scene_id"], **extra_env}
    assert [call for call in runner.calls if call[0] == "systemd-run"] == [[
        "systemd-run", f"--unit={unit}", "--collect", "--service-type=exec",
        "--property=TimeoutStartSec=2h", "--setenv=PYTHONDONTWRITEBYTECODE=1", "--property=RuntimeMaxSec=2h",
        "--property=ProtectSystem=strict", "--property=PrivateTmp=yes", "--property=NoNewPrivileges=yes",
        "--property=PrivateDevices=yes", "--property=ProtectHome=yes",
        "--property=ProtectKernelTunables=yes", "--property=ProtectControlGroups=yes",
        "--property=CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_CHOWN CAP_SYS_PTRACE",
        "--property=AmbientCapabilities=CAP_DAC_OVERRIDE",
        "--property=ReadWritePaths=/var/lib/blueprint/pubsub-handoffs "
        "/var/lib/blueprint/pipeline-control-plane/disk-reservations "
        f"/var/lib/blueprint/pipeline-control-plane/storage-pins {results}",
        *(f"--setenv={key}={value}" for key, value in env.items()),
        "--", "/bin/bash", "/opt/blueprint/operator-door/door-retire-scene-workspace.sh",
    ]]
    assert _result(config, request_id) == {**_result(config, request_id), "status": "launched", "unit": unit}
    # A retirement is not a deploy: it neither waits for nor blocks one.
    assert not any(call[:2] == ["systemctl", "list-units"] for call in runner.calls)


def test_a_retirement_unit_never_names_its_scene(config: DoorConfig) -> None:
    """A scene id is caller text; in a unit name it could match `blueprint-*deploy*` and block deploys."""

    request_id = _spooled(config, {"kind": "retire-scene-workspace", "scene_id": "site-deploy-1"})
    process_spool(config, runner=FakeRunner())

    unit = _result(config, request_id)["unit"]
    assert "deploy" not in unit and "site" not in unit
    assert re.fullmatch(r"blueprint-operator-door-retire-[0-9a-f]{12}-[0-9a-f]{8}\.service", unit)


def test_restore_launches_the_active_release_in_a_bounded_sandbox(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "restore-scene-workspace", "scene_id": "scene-1",
                                   "bucket": "blueprint-8c1ca.appspot.com"})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    [call] = [row for row in runner.calls if row[0] == "systemd-run"]
    assert call[1].startswith("--unit=blueprint-operator-door-restore-")
    assert "--property=RuntimeMaxSec=2h" in call
    assert "--property=ProtectHome=yes" in call
    assert "--setenv=DOOR_BUCKET=blueprint-8c1ca.appspot.com" in call
    assert call[-1] == "/opt/blueprint/operator-door/door-restore-scene-workspace.sh"
    assert _result(config, request_id)["status"] == "launched"


def test_lane_scratch_renew_launches_only_the_configured_root(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "lane-scratch", "action": "renew", "root": "inputs",
                                   "lane": "g1", "name": "run-1", "owner": "agent-1",
                                   "expected_digest": "sha256:" + "a" * 64, "ttl_seconds": 3600})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    [call] = [row for row in runner.calls if row[0] == "systemd-run"]
    assert "--property=RuntimeMaxSec=2min" in call
    assert "--property=ProtectSystem=strict" in call
    assert "--property=ReadWritePaths=/var/lib/blueprint/task-evaluation-inputs/lanes " \
           + str(Path(config.spool_root) / "results") in call
    assert "--setenv=DOOR_SCRATCH_ROOT=/var/lib/blueprint/task-evaluation-inputs/lanes" in call
    assert "--setenv=DOOR_SCRATCH_LANE=g1" in call
    assert "--setenv=DOOR_SCRATCH_ACTION=renew" in call
    assert call[-1] == "/opt/blueprint/operator-door/door-lane-scratch.sh"
    assert _result(config, request_id)["status"] == "launched"


def test_provider_output_resume_launches_the_release_module(config: DoorConfig) -> None:
    """Review I5: the transient unit is sandboxed to the canary tree, the disk ledger and its
    result; the script runs the active release's resume as ``blueprint`` with a private umask."""
    request_id = _spooled(config, {"kind": "provider-output-resume", "run": "activation-1", "attempt": 3,
                                   "ingest": True})
    runner = FakeRunner()
    process_spool(config, runner=runner)

    [call] = [row for row in runner.calls if row[0] == "systemd-run"]
    label = hashlib.sha256(b"activation-1/3").hexdigest()[:12]
    unit = f"blueprint-operator-door-output-resume-{label}-{request_id[-8:]}.service"
    results = str(Path(config.spool_root) / "results")
    canaries = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries"
    assert call[1] == f"--unit={unit}"
    assert "--property=RuntimeMaxSec=2h" in call and "--property=TimeoutStartSec=2h" in call
    for sandbox in ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
                    "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes"):
        assert f"--property={sandbox}" in call, sandbox
    assert (f"--property=ReadWritePaths={canaries} "
            f"/var/lib/blueprint/pipeline-control-plane/disk-reservations {results}") in call
    for value in (f"DOOR_CANARY_ROOT={canaries}", "DOOR_RUN=activation-1", "DOOR_ATTEMPT=3", "DOOR_INGEST=1",
                  "DOOR_SERVICE_USER=blueprint", "DOOR_CONTROL_PLANE_REPO=/opt/blueprint/task-evaluation-control-plane"):
        assert f"--setenv={value}" in call, value
    assert call[-1] == "/opt/blueprint/operator-door/door-provider-output-resume.sh"
    assert _result(config, request_id) == {**_result(config, request_id), "status": "launched", "unit": unit}
    # Without ingest the script is told nothing more; a resume never waits for or blocks a deploy.
    plain = _spooled(config, {"kind": "provider-output-resume", "run": "activation-1", "attempt": 3})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    [call] = [row for row in runner.calls if row[0] == "systemd-run"]
    assert not any(value.startswith("--setenv=DOOR_INGEST") for value in call)
    assert not any(row[:2] == ["systemctl", "list-units"] for row in runner.calls)
    assert _result(config, plain)["status"] == "launched"


def test_canonical_provenance_is_staged_private_and_not_in_environment(config: DoorConfig) -> None:
    from tests.test_operator_door_requests import _canonical_provenance

    text = _canonical_provenance()
    digest = hashlib.sha256(text.encode()).hexdigest()
    request_id = _spooled(
        config,
        {
            "kind": "deploy",
            "commit": SHA,
            "release_provenance_json": text,
            "release_provenance_sha256": digest,
        },
    )
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert _result(config, request_id)["status"] == "launched"
    launch = [call for call in runner.calls if call[0] == "systemd-run"][0]
    env = dict(
        part.removeprefix("--setenv=").split("=", 1)
        for part in launch
        if part.startswith("--setenv=")
    )
    staged = Path(config.spool_root) / "results" / f"{request_id}.release-provenance.json"
    assert staged.read_bytes() == text.encode()
    assert staged.stat().st_mode & 0o777 == 0o600
    assert env["DOOR_RELEASE_PROVENANCE_FILE"] == str(staged)
    assert env["DOOR_RELEASE_PROVENANCE_SHA256"] == digest
    assert not any(text in part for part in launch)


def test_canonical_provenance_cannot_replace_existing_results_symlink(config: DoorConfig) -> None:
    from tests.test_operator_door_requests import _canonical_provenance

    text = _canonical_provenance()
    digest = hashlib.sha256(text.encode()).hexdigest()
    request_id = _spooled(
        config,
        {
            "kind": "deploy",
            "commit": SHA,
            "release_provenance_json": text,
            "release_provenance_sha256": digest,
        },
    )
    victim = Path(config.state_root) / "victim"
    victim.write_text("preserve")
    (Path(config.spool_root) / "results" / f"{request_id}.release-provenance.json").symlink_to(
        victim
    )
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert victim.read_text() == "preserve"
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    assert _result(config, request_id)["status"] != "launched"


_PRIOR_SELECTED = "20261009T154718Z-selected-handoff-9ef60661"


@pytest.mark.parametrize("mode", ["atlas-status", "atlas-retry", "preparation-resume", "discover-selector"])
def test_selected_atlas_runner_keeps_service_user_and_only_exact_job_write_path(config, mode):
    from tests.test_operator_door_requests import _selected_request
    body = {**_selected_request(), "mode": mode}
    if mode in {"atlas-retry", "preparation-resume"}:
        body["atlas_status_digest"] = "sha256:" + "c" * 64
    if mode == "discover-selector":
        fields = {"kind", "mode", "bucket", "scene_id", "capture_id", "marker_generation"}
        body = {key: value for key, value in body.items() if key in fields}
    request_id = _spooled(config, body)
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert _result(config, request_id)["status"] == "launched"
    launch = next(call for call in runner.calls if call[0] == "systemd-run")
    env = dict(part.removeprefix("--setenv=").split("=", 1) for part in launch if part.startswith("--setenv="))
    assert env["DOOR_SERVICE_USER"] == "blueprint"
    assert json.loads(env["DOOR_SELECTED_HANDOFF_REQUEST"]) == body
    writable = str(Path(config.spool_root) / "results")
    if mode == "atlas-retry":
        writable += (f" /var/lib/blueprint/pubsub-handoffs/{body['bucket']}/scenes/{body['scene_id']}"
                     f"/captures/{body['capture_id']}/pipeline/atlas")
    if mode == "preparation-resume":
        writable += " /var/lib/blueprint"
    assert f"--property=ReadWritePaths={writable}" in launch
    assert "--property=ReadWritePaths=/var/lib/blueprint" not in launch
    assert "--property=ProtectSystem=strict" in launch
    assert launch[-1].endswith("door-selected-handoff.sh")


def _preclaim_evidence(config, *, native_status="capture_owner_observation_unavailable_retryable", prior=_PRIOR_SELECTED):
    from tests.test_operator_door_requests import _selected_request

    selection = {**_selected_request(), "mode": "dispatch"}
    results = Path(config.spool_root) / "results"
    native_path = results / f"{prior}.selected-handoff.json"
    outcome_path = results / f"{prior}.outcome.json"
    native = {"schema_version": "selected_handoff_recovery.v1", "mode": "dispatch", "status": "blocked",
              "native_status": native_status, "native_disposition": "retryable",
              "source_membership_verified": True, "current_admission_verified": True,
              "selector_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
              "blockers": ["selected_handoff_native_admission_or_result_blocked"],
              "provider_dispatch_performed": "unknown; consult native receipts"}
    outcome = {"schema": "blueprint_operator_door_outcome.v1", "status": "failed",
               "code": "selected_handoff_blocked", "exit_code": 1, "result": str(native_path)}
    native_path.write_text(json.dumps(native))
    native_path.chmod(0o640)
    outcome_path.write_text(json.dumps(outcome))
    outcome_path.chmod(0o600)
    return {**selection, "resume_request_id": prior}, native_path, outcome_path


@pytest.mark.parametrize("launch_rc", [0, 1])
@pytest.mark.parametrize("native_status", [
    "capture_owner_observation_unavailable_retryable",
    "capture_source_membership_unavailable_retryable",
])
def test_selected_preclaim_recovery_is_linked_exclusive_and_keeps_original_evidence(config, monkeypatch, launch_rc, native_status):
    import operator_door.spool_runner as module

    body, native_path, outcome_path = _preclaim_evidence(config, native_status=native_status)
    before = (native_path.read_bytes(), outcome_path.read_bytes())
    syncs = []
    real_fsync = os.fsync
    def sync(fd):
        syncs.append(os.fstat(fd).st_mode)
        real_fsync(fd)
    monkeypatch.setattr(module.os, "fsync", sync)
    request_id = _spooled(config, body)
    runner = FakeRunner(systemd_run_rc=launch_rc)
    process_spool(config, runner=runner)
    launch = next(call for call in runner.calls if call[0] == "systemd-run")
    env = dict(part.removeprefix("--setenv=").split("=", 1) for part in launch if part.startswith("--setenv="))
    assert json.loads(env["DOOR_SELECTED_HANDOFF_REQUEST"]) == {key: value for key, value in body.items() if key != "resume_request_id"}
    marker_path = native_path.parent / f"{_PRIOR_SELECTED}.selected-handoff-recovery.json"
    marker = json.loads(marker_path.read_text())
    assert marker["request_id"] == request_id and marker["prior_request_id"] == _PRIOR_SELECTED
    assert marker["selector_sha256"] == json.loads(before[0])["selector_sha256"]
    assert marker_path.stat().st_mode & 0o777 == 0o600
    import stat
    assert any(stat.S_ISREG(mode) for mode in syncs) and any(stat.S_ISDIR(mode) for mode in syncs)
    assert (native_path.read_bytes(), outcome_path.read_bytes()) == before
    second_id = _spooled(config, body)
    runner.calls.clear()
    process_spool(config, runner=runner)
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    assert _result(config, second_id)["code"] == "selected_handoff_recovery_already_consumed"
    assert json.loads(marker_path.read_text()) == marker


@pytest.mark.parametrize("mutation", [
    "schema", "completed", "postlease", "unknown", "status_list", "status_dict", "hash", "admission", "membership", "disposition", "mode",
    "blockers", "outcome_schema", "outcome_status", "outcome_code", "outcome_exit_bool", "outcome_path",
    "missing", "symlink", "writable", "oversize", "duplicate", "nonfinite", "foreign_owner", "self",
])
def test_selected_preclaim_recovery_refuses_untrusted_or_ineligible_evidence(config, monkeypatch, mutation):
    body, native_path, outcome_path = _preclaim_evidence(config)
    native = json.loads(native_path.read_text())
    outcome = json.loads(outcome_path.read_text())
    changes = {"schema": ("schema_version", "wrong"), "completed": ("status", "native_handler_returned"),
               "postlease": ("native_status", "provider_failed_retryable"), "unknown": ("native_status", "unknown"),
               "status_list": ("native_status", []), "status_dict": ("native_status", {}),
               "hash": ("selector_sha256", "0" * 64), "admission": ("current_admission_verified", False),
               "membership": ("source_membership_verified", False), "disposition": ("native_disposition", "terminal_success"),
               "mode": ("mode", "inspect"), "blockers": ("blockers", [])}
    outcome_changes = {"outcome_schema": ("schema", "wrong"), "outcome_status": ("status", "completed"),
                       "outcome_code": ("code", "script_exited_early"), "outcome_exit_bool": ("exit_code", True),
                       "outcome_path": ("result", "/other/result")}
    if mutation in changes:
        key, value = changes[mutation]
        native[key] = value
        native_path.write_text(json.dumps(native))
    if mutation in outcome_changes:
        key, value = outcome_changes[mutation]
        outcome[key] = value
        outcome_path.write_text(json.dumps(outcome))
    if mutation == "missing":
        native_path.unlink()
    if mutation == "symlink":
        native_path.unlink()
        native_path.symlink_to(outcome_path)
    if mutation == "writable":
        native_path.chmod(0o660)
    if mutation == "oversize":
        native_path.write_text(" " * 16385)
    if mutation == "duplicate":
        native_path.write_text('{"x": 1, "x": 2}')
    if mutation == "nonfinite":
        native_path.write_text('{"x": NaN}')
    if mutation == "foreign_owner":
        import operator_door.spool_runner as module
        real_fstat = os.fstat
        from types import SimpleNamespace
        def foreign(fd):
            info = real_fstat(fd)
            return SimpleNamespace(st_uid=info.st_uid + 1, st_mode=info.st_mode, st_size=info.st_size, st_nlink=info.st_nlink)
        monkeypatch.setattr(module.os, "fstat", foreign)
    request_id = _spooled(config, body)
    if mutation == "self":
        pending = Path(config.spool_root) / "pending" / f"{request_id}.json"
        document = json.loads(pending.read_text())
        document["request"]["resume_request_id"] = request_id
        pending.write_text(json.dumps(document))
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    assert _result(config, request_id)["status"] == "refused"
    assert not (native_path.parent / f"{_PRIOR_SELECTED}.selected-handoff-recovery.json").exists()


def test_selected_staging_recovery_links_failed_successor_without_resetting_history(config):
    body, original_native, original_outcome = _preclaim_evidence(config)
    original_bytes = (original_native.read_bytes(), original_outcome.read_bytes())
    first_id = _spooled(config, body)
    process_spool(config, runner=FakeRunner())
    original_marker = original_native.parent / f"{_PRIOR_SELECTED}.selected-handoff-recovery.json"
    marker_bytes = original_marker.read_bytes()
    successor_body, failed_native, failed_outcome = _preclaim_evidence(
        config, prior=first_id, native_status="capture_source_membership_unavailable_retryable")
    failed_bytes = (failed_native.read_bytes(), failed_outcome.read_bytes())
    second_id = _spooled(config, successor_body)
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert any(call[0] == "systemd-run" for call in runner.calls)
    new_marker_path = failed_native.parent / f"{first_id}.selected-handoff-recovery.json"
    new_marker = json.loads(new_marker_path.read_bytes())
    assert new_marker["prior_request_id"] == first_id and new_marker["request_id"] == second_id
    assert original_marker.read_bytes() == marker_bytes
    assert (original_native.read_bytes(), original_outcome.read_bytes()) == original_bytes
    assert (failed_native.read_bytes(), failed_outcome.read_bytes()) == failed_bytes
    duplicate_id = _spooled(config, successor_body)
    runner.calls.clear()
    process_spool(config, runner=runner)
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    assert _result(config, duplicate_id)["code"] == "selected_handoff_recovery_already_consumed"
    assert json.loads(new_marker_path.read_bytes()) == new_marker


def test_scene_window_extension_runs_the_release_module_as_the_named_caller(config: DoorConfig) -> None:
    body = {"kind": "extend-scene-execution-window", "intent_id": "scene-" + "a" * 64,
            "intent_digest": "sha256:" + "b" * 64, "expires_at": "2026-11-07T10:30:10Z",
            "authorization_reference": "founder approval (time only) $(touch x)",
            "ack": "extend-scene-execution-window", "apply": True}
    request_id = _spooled(config, body)
    runner = FakeRunner()
    process_spool(config, runner=runner)
    [call] = [row for row in runner.calls if row[0] == "systemd-run"]
    label = hashlib.sha256(body["intent_id"].encode()).hexdigest()[:12]
    unit = f"blueprint-operator-door-scene-window-{label}-{request_id[-8:]}.service"
    assert call[1] == f"--unit={unit}" and "--property=RuntimeMaxSec=5min" in call
    for sandbox in ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "ProtectHome=yes", "UMask=0077"):
        assert f"--property={sandbox}" in call, sandbox
    results = str(Path(config.spool_root) / "results")
    assert f"--property=ReadWritePaths=/var/lib/blueprint/pipeline-control-plane {results}" in call
    hidden = next(value for value in call if value.startswith("--property=InaccessiblePaths="))
    # The sync token stays readable for the signed WebApp notice; other secrets stay hidden.
    assert "/etc/blueprint/provider-secrets" not in hidden.split("=", 2)[-1].replace("-/", "/").split()
    assert "-/etc/blueprint-operator-door" in hidden and "-/etc/blueprint/credentials" in hidden
    assert "--setenv=DOOR_REQUESTED_BY=cloud" in call and "--setenv=DOOR_SERVICE_USER=blueprint" in call
    assert f"--setenv=DOOR_SCENE_WINDOW_REQUEST={json.dumps(body, sort_keys=True)}" in call
    assert call[-1] == "/opt/blueprint/operator-door/door-extend-scene-execution-window.sh"
    assert _result(config, request_id) == {**_result(config, request_id), "status": "launched", "unit": unit}
    # Not a deploy: it neither waits for nor blocks one.
    assert not any(row[:2] == ["systemctl", "list-units"] for row in runner.calls)


def test_scene_window_caller_name_is_rechecked_before_launch(config: DoorConfig) -> None:
    body = {"kind": "extend-scene-execution-window", "intent_id": "scene-" + "a" * 64,
            "intent_digest": "sha256:" + "b" * 64, "expires_at": "2026-11-07T10:30:10Z",
            "authorization_reference": "founder approval", "ack": "extend-scene-execution-window", "apply": False}
    request_id = enqueue(config, validate_request(body), requested_by="bad caller")
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert not [row for row in runner.calls if row[0] == "systemd-run"]
    assert _result(config, request_id)["code"] == "scene_window_caller_invalid"
