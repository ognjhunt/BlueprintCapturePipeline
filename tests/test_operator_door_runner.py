"""The root runner: revalidate spooled requests, then run fixed commands only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult  # noqa: E402
from operator_door.requests import SCHEMA, enqueue, validate_request  # noqa: E402
from operator_door.runner import process_spool  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeRunner:
    def __init__(self, active_deploy: str = "", systemd_run_rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.active_deploy = active_deploy
        self.systemd_run_rc = systemd_run_rc

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        self.calls.append(list(argv))
        if argv[:2] == ["systemctl", "list-units"]:
            return CommandResult(0, self.active_deploy, "")
        if argv[0] == "systemd-run":
            return CommandResult(self.systemd_run_rc, "", "Failed to start" if self.systemd_run_rc else "")
        return CommandResult(0, "", "")


@pytest.fixture()
def config(tmp_path: Path) -> DoorConfig:
    for state in ("pending", "processing", "completed", "results"):
        (tmp_path / "requests" / state).mkdir(parents=True)
    return DoorConfig(state_root=str(tmp_path), install_root="/opt/blueprint/operator-door")


def _result(config: DoorConfig, request_id: str) -> dict:
    return json.loads((Path(config.spool_root) / "results" / f"{request_id}.json").read_text(encoding="utf-8"))


def _spooled(config: DoorConfig, body: dict) -> str:
    return enqueue(config, validate_request(body), requested_by="cloud")


def test_deploy_launches_the_door_deploy_script_with_validated_parameters(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA, "mode": "canary", "wait_for_idle": False})
    runner = FakeRunner()
    assert process_spool(config, runner=runner) == 0
    launch = [call for call in runner.calls if call[0] == "systemd-run"][0]
    unit = f"blueprint-operator-door-deploy-{SHA[:12]}-{request_id[-8:]}"
    assert launch[:6] == ["systemd-run", f"--unit={unit}", "--collect", "--service-type=exec",
                          "--property=TimeoutStartSec=3h", "--setenv=PYTHONDONTWRITEBYTECODE=1"]
    assert launch[-2:] == ["/bin/bash", "/opt/blueprint/operator-door/door-deploy.sh"]
    env = dict(part.removeprefix("--setenv=").split("=", 1) for part in launch if part.startswith("--setenv="))
    assert env["DOOR_REQUEST_ID"] == request_id and env["DOOR_COMMIT"] == SHA
    assert env["DOOR_MODE"] == "canary" and env["DOOR_WAIT_FOR_IDLE"] == "0"
    assert env["DOOR_SOURCE_CLONE"] == DoorConfig().source_clone
    assert env["DOOR_RESULTS_DIR"] == str(Path(config.spool_root) / "results")
    result = _result(config, request_id)
    assert result["status"] == "launched" and result["unit"] == unit
    assert (Path(config.spool_root) / "completed" / f"{request_id}.json").exists()
    assert not list((Path(config.spool_root) / "pending").glob("*.json"))


def test_deploy_is_refused_while_another_deploy_runs(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "deploy", "commit": SHA})
    runner = FakeRunner(active_deploy="blueprint-fable-deploy-2c212fe5.service loaded active running x\n")
    process_spool(config, runner=runner)
    assert not any(call[0] == "systemd-run" for call in runner.calls)
    result = _result(config, request_id)
    assert result == {**result, "status": "refused", "code": "deploy_in_progress:blueprint-fable-deploy-2c212fe5.service"}


def test_failed_launch_is_recorded(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "door-upgrade", "commit": SHA})
    process_spool(config, runner=FakeRunner(systemd_run_rc=1))
    result = _result(config, request_id)
    assert result["status"] == "failed" and result["returncode"] == 1 and "Failed" in result["stderr_tail"]


def test_unit_actions_use_no_block_systemctl(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "unit", "unit": "blueprint-pubsub-handoff-listener.timer",
                                   "action": "start"})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    assert ["systemctl", "--no-block", "start", "--", "blueprint-pubsub-handoff-listener.timer"] in runner.calls
    assert _result(config, request_id)["status"] == "done"


def test_stage_replay_launches_the_replay_script(config: DoorConfig) -> None:
    child = "sam31-" + "cd" * 16
    request_id = _spooled(config, {"kind": "stage-replay", "commit": SHA, "child": child})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    launch = [call for call in runner.calls if call[0] == "systemd-run"][0]
    assert f"--unit=blueprint-operator-door-replay-{request_id[-8:]}" in launch
    assert "--setenv=DOOR_CHILD=" + child in launch and "--setenv=DOOR_PARENT=" in launch
    assert launch[-1] == "/opt/blueprint/operator-door/door-replay.sh"


def test_door_upgrade_launches_the_upgrade_script(config: DoorConfig) -> None:
    request_id = _spooled(config, {"kind": "door-upgrade", "commit": SHA})
    runner = FakeRunner()
    process_spool(config, runner=runner)
    launch = [call for call in runner.calls if call[0] == "systemd-run"][0]
    assert f"--unit=blueprint-operator-door-upgrade-{SHA[:12]}-{request_id[-8:]}" in launch
    assert launch[-1] == "/opt/blueprint/operator-door/door-upgrade.sh"


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
