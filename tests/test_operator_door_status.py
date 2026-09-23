"""Host facts and the status document, with every host command faked."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult, HostInfo, HostRefused  # noqa: E402
from operator_door.status import build_status  # noqa: E402

SHOW_OUTPUT = """Id=blueprint-task-evaluation-scene-progression.service
ActiveState=inactive
SubState=dead
Result=success
ExecMainStatus=0
UnitFileState=static

Id=blueprint-pipeline-intake.service
ActiveState=active
SubState=running
Result=success
ExecMainStatus=0
UnitFileState=enabled
"""


class FakeRunner:
    def __init__(self, responses: dict[str, CommandResult]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for prefix, result in self.responses.items():
            if joined.startswith(prefix):
                return result
        return CommandResult(0, "", "")


@pytest.fixture()
def host_tree(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path.resolve()
    state = base / "pipeline-control-plane"
    locks = state / "provider-locks"
    receipts = state / "deploy-receipts"
    guard = state / "gpu_spend_guard"
    releases = base / "releases" / ("a" * 40)
    for directory in (locks, receipts, guard, releases, base / "door" / "requests" / "pending"):
        directory.mkdir(parents=True)
    (locks / "vast_paid_launch.lock").write_text('{"holder": "last"}', encoding="utf-8")
    (locks / "vast_paid_launch.slot1.lock").write_text("", encoding="utf-8")
    held = (locks / "vast_paid_launch.lock").stat()
    device = f"{os.major(held.st_dev):02x}:{os.minor(held.st_dev):02x}"
    (base / "locks").write_text(
        f"1: FLOCK  ADVISORY  WRITE {os.getpid()} {device}:{held.st_ino} 0 EOF\n"
        f"2: -> FLOCK  ADVISORY  WRITE 999999 {device}:{held.st_ino} 0 EOF\n",
        encoding="utf-8",
    )
    for index, commit in enumerate(("b" * 40, "c" * 40)):
        receipt = receipts / f"iteration_{commit[:12]}.json"
        receipt.write_text(json.dumps({"status": "deployed", "source_commit": commit}), encoding="utf-8")
        os.utime(receipt, (1_000 + index, 1_000 + index))
    (guard / "latest.json").write_text(json.dumps({"live_resource_count": 0, "spend_usd": 1.2}), encoding="utf-8")
    link = base / "active"
    os.symlink(releases, link)
    (base / "door" / "requests" / "pending" / "x.json").write_text("{}", encoding="utf-8")
    return {"base": base, "state": state, "link": link}


def _config(tree: dict[str, Path]) -> DoorConfig:
    return DoorConfig(
        read_roots=(str(tree["base"]),),
        hidden_paths=(str(tree["base"] / "hidden"),),
        control_plane_state=str(tree["state"]),
        active_release_link=str(tree["link"]),
        state_root=str(tree["base"] / "door"),
        controller_units=("blueprint-pipeline-intake.service",),
    )


def _host(tree: dict[str, Path], runner: FakeRunner, version: dict | Exception | None = None) -> HostInfo:
    def fetch(url: str) -> dict:
        if isinstance(version, Exception):
            raise version
        return version or {"source_commit": "a" * 40, "commit_proven": True, "blockers": [], "extra": "x"}

    return HostInfo(_config(tree), runner=runner, proc_locks_path=str(tree["base"] / "locks"), fetch_json=fetch)


def test_unit_properties_parses_blocks_and_never_asks_for_environment(host_tree: dict[str, Path]) -> None:
    runner = FakeRunner({"systemctl show": CommandResult(0, SHOW_OUTPUT, "")})
    units = _host(host_tree, runner).unit_properties(
        ["blueprint-task-evaluation-scene-progression.service", "blueprint-pipeline-intake.service"]
    )
    assert [unit["Id"] for unit in units] == [
        "blueprint-task-evaluation-scene-progression.service",
        "blueprint-pipeline-intake.service",
    ]
    assert units[1]["ActiveState"] == "active"
    argv = runner.calls[0]
    assert argv[:2] == ["systemctl", "show"]
    assert not any("Environment" in part for part in argv)


def test_unit_names_are_validated(host_tree: dict[str, Path]) -> None:
    host = _host(host_tree, FakeRunner({}))
    for bad in ("sshd.service", "blueprint-x.service; rm -rf /", "--all", "blueprint-x"):
        with pytest.raises(HostRefused):
            host.unit_properties([bad])


def test_list_units_parses_plain_rows(host_tree: dict[str, Path]) -> None:
    rows = (
        "blueprint-a.service loaded failed failed Blueprint A thing\n"
        "blueprint-b.timer loaded active waiting Blueprint B timer\n"
    )
    runner = FakeRunner({"systemctl list-units": CommandResult(0, rows, "")})
    units = _host(host_tree, runner).list_units("blueprint-*", states=("failed",))
    assert units[0] == {"unit": "blueprint-a.service", "load": "loaded", "active": "failed",
                        "sub": "failed", "description": "Blueprint A thing"}
    assert "--state=failed" in runner.calls[0]


def test_journal_is_bounded_and_redacted(host_tree: dict[str, Path]) -> None:
    runner = FakeRunner({"journalctl": CommandResult(0, "ok line\nGEMINI_API_KEY=abcdef12345\n", "")})
    host = _host(host_tree, runner)
    text = host.journal("blueprint-pipeline-intake.service", lines=50_000, since="-1h")
    assert "abcdef12345" not in text and "ok line" in text
    argv = runner.calls[0]
    assert argv[:3] == ["journalctl", "-u", "blueprint-pipeline-intake.service"]
    assert "-n" in argv and argv[argv.index("-n") + 1] == str(DoorConfig().max_journal_lines)
    assert "--since=-1h" in argv


@pytest.mark.parametrize("since", ["--output=json", "1h; reboot", "x" * 80])
def test_journal_since_is_validated(host_tree: dict[str, Path], since: str) -> None:
    with pytest.raises(HostRefused):
        _host(host_tree, FakeRunner({})).journal("blueprint-pipeline-intake.service", lines=10, since=since)


def test_lock_holders_come_from_proc_locks_not_file_contents(host_tree: dict[str, Path]) -> None:
    locks = _host(host_tree, FakeRunner({})).paid_launch_locks()
    by_name = {Path(item["path"]).name: item for item in locks}
    assert by_name["vast_paid_launch.lock"]["held"] is True
    assert by_name["vast_paid_launch.lock"]["holders"] == [os.getpid()]
    assert by_name["vast_paid_launch.slot1.lock"]["held"] is False


def test_status_assembles_every_section(host_tree: dict[str, Path]) -> None:
    runner = FakeRunner({
        "systemctl show": CommandResult(0, SHOW_OUTPUT, ""),
        "systemctl list-units": CommandResult(0, "blueprint-a.service loaded failed failed A\n", ""),
    })
    status = build_status(_config(host_tree), _host(host_tree, runner), caller={"name": "cloud"})
    assert status["schema"] == "blueprint_operator_door_status.v1"
    assert status["deployed"] == {"source_commit": "a" * 40, "commit_proven": True, "blockers": []}
    assert status["active_release"]["commit"] == "a" * 40
    assert [r["source_commit"] for r in status["deploys"]["recent_receipts"]] == ["c" * 40, "b" * 40]
    assert status["paid_launch_locks"][0]["held"] in (True, False)
    assert status["spend_guard"] == {"live_resource_count": 0, "spend_usd": 1.2}
    assert status["failed_units"] == ["blueprint-a.service"]
    assert status["door_requests"] == {"pending": 1, "processing": 0}
    assert status["door"]["caller"] == {"name": "cloud"}
    assert set(status["disk"]) and "loadavg" in status["load"]


def test_status_survives_a_failing_section(host_tree: dict[str, Path]) -> None:
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({}), OSError("refused")), caller={})
    assert status["deployed"] == {"error": "version_unavailable:OSError"}
    assert status["active_release"]["commit"] == "a" * 40
