"""Host facts and the status document, with every host command faked."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/hostinfo.py
#   deploy/operator-door/operator_door/status.py

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
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


CAPACITY_SUMMARY = {
    "schema_version": "control_plane_capacity_summary.v1",
    "observed_at_epoch": 1_000.0,
    "level": "warning",
    "report_digest": "sha256:" + "0" * 64,
    "alerts": [{"code": "usage_unclassified_root", "root": "/var/lib/blueprint/x", "allocated_bytes": 2 * 1024**3}],
    "mounts": [{"mount": "/var/lib/blueprint", "status": "measured", "level": "ok", "free_bytes": 1}],
    "usage": {
        "status": "complete", "observed_at_epoch": 900.0, "age_seconds": 100.0,
        "mounts": [{"mount": "/", "used_bytes": 100, "surveyed_bytes": 97, "classified_bytes": 90,
                    "attributed_fraction": 0.97}],
        "by_class": [{"storage_class": "work", "allocated_bytes": 60, "apparent_bytes": 58, "files": 3}],
        "top_roots": [{"root": "/var/lib/blueprint/pubsub-handoffs", "storage_class": "work", "allocated_bytes": 60}],
        "top_owners": [{"owner": "scene:site-capture-1", "root": "/var/lib/blueprint/pubsub-handoffs",
                        "storage_class": "work", "allocated_bytes": 60}],
        "unclassified_roots": [{"root": "/var/lib/blueprint/x", "allocated_bytes": 2 * 1024**3}],
    },
    "volume_resize": None,
    # Not a summary key: the door must never pass it through.
    "project_spend": {"spend_usd": 3.0},
}


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
    summary = state / "capacity" / "summary.json"
    summary.parent.mkdir()
    summary.write_text(json.dumps(CAPACITY_SUMMARY), encoding="utf-8")
    return {"base": base, "state": state, "link": link, "summary": summary}


def _config(tree: dict[str, Path]) -> DoorConfig:
    return DoorConfig(
        read_roots=(str(tree["base"]),),
        hidden_paths=(str(tree["base"] / "hidden"),),
        control_plane_state=str(tree["state"]),
        active_release_link=str(tree["link"]),
        state_root=str(tree["base"] / "door"),
        controller_units=("blueprint-pipeline-intake.service",),
        capacity_summary=str(tree["summary"]),
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
    assert status["holds"] == []
    assert status["break_glass"] == {"unreported": 0, "latest": None}
    assert status["door"]["caller"] == {"name": "cloud"}
    assert set(status["disk"]) and "loadavg" in status["load"]


def test_status_shows_active_and_overdue_holds_with_remaining_seconds(host_tree: dict[str, Path]) -> None:
    root = host_tree["base"] / "door" / "requests" / "holds"
    root.mkdir()
    now = int(time.time())
    for unit, expiry in (("blueprint-scene-progression.timer", now + 120),
                         ("blueprint-pubsub-handoff-listener.timer", now - 1),
                         ("blueprint-agent-run-dispatcher.timer", now - 1)):
        (root / f"{unit}.json").write_text(json.dumps({
            "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice",
            "reason": "inspect", "requested_by": "cloud", "request_id": "20260926T120000Z-hold-0000abcd",
            "created_at": "2026-09-26T12:00:00+00:00", "expires_at": "2026-09-26T13:00:00+00:00",
            "expires_at_epoch": expiry, "status": "active",
            **({"require_explicit_release": True} if unit == "blueprint-agent-run-dispatcher.timer" else {})}),
            encoding="utf-8")
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})
    assert len(status["holds"]) == 3
    hold = next(row for row in status["holds"] if row["unit"] == "blueprint-scene-progression.timer")
    assert hold["unit"] == "blueprint-scene-progression.timer" and hold["owner"] == "alice"
    assert 0 < hold["remaining_seconds"] <= 120
    overdue = next(row for row in status["holds"] if row["unit"] == "blueprint-pubsub-handoff-listener.timer")
    assert overdue["remaining_seconds"] == 0 and overdue["expired"] is True
    assert overdue["require_explicit_release"] is False
    stopped = next(row for row in status["holds"] if row["unit"] == "blueprint-agent-run-dispatcher.timer")
    assert stopped["expired"] is True and stopped["require_explicit_release"] is True


def test_status_counts_unreported_break_glass_notes(host_tree: dict[str, Path]) -> None:
    root = host_tree["state"] / "cleanup-receipts"
    root.mkdir()
    old = "20260926T120000Z-0123456789ab.json"
    new = "20260927T120000Z-abcdef012345.json"
    for name, epoch, operator in ((old, 1790424000, "alice"), (new, 1790510400, "bob")):
        (root / name).write_text(json.dumps({"created_at_epoch": epoch, "created_at": "2026-09-27T12:00:00Z",
                                             "operator": operator, "reason": "door repair"}), encoding="utf-8")
    (root / "reported.jsonl").write_text(json.dumps({"name": old}) + "\n", encoding="utf-8")
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})
    assert status["break_glass"] == {"unreported": 1,
                                     "latest": {"created_at": "2026-09-27T12:00:00Z",
                                                "operator": "bob", "reason": "door repair"}}


def test_status_survives_a_failing_section(host_tree: dict[str, Path]) -> None:
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({}), OSError("refused")), caller={})
    assert status["deployed"] == {"error": "version_unavailable:OSError"}
    assert status["active_release"]["commit"] == "a" * 40


def test_a_receipt_with_credential_shaped_content_names_the_refusal(host_tree: dict[str, Path]) -> None:
    receipt = host_tree["state"] / "deploy-receipts" / "iteration_dddddddddddd.json"
    receipt.write_text(json.dumps({"status": "deployed", "api_key": "abcdefghijklmnop12"}), encoding="utf-8")
    os.utime(receipt, (2_000, 2_000))
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})
    newest = status["deploys"]["recent_receipts"][0]
    assert newest["name"] == receipt.name and newest["error"] == "secret_content_refused"
    assert "status" not in newest


def test_status_reports_capacity_usage(host_tree: dict[str, Path]) -> None:
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={"name": "cloud"})
    assert status["capacity"]["level"] == "warning"
    assert status["capacity"]["usage"]["top_owners"][0]["owner"] == "scene:site-capture-1"
    assert "project_spend" not in status["capacity"]
    assert set(status["disk"])


def test_status_capacity_section_fails_soft(host_tree: dict[str, Path]) -> None:
    host_tree["summary"].unlink()
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})
    assert status["capacity"] == {"error": "capacity_unavailable:FileNotFoundError"}
    assert status["active_release"]["commit"] == "a" * 40


def test_a_capacity_summary_with_credential_shaped_content_is_refused(host_tree: dict[str, Path]) -> None:
    host_tree["summary"].write_text(json.dumps({**CAPACITY_SUMMARY, "level": "sk-" + "A" * 30}), encoding="utf-8")
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})
    assert status["capacity"] == {"error": "capacity_unavailable:SecretContentRefused"}


def test_notifier_binding_reports_effective_old_hook_without_command_or_secret(host_tree: dict[str, Path]) -> None:
    script = "/opt/blueprint/BlueprintCapturePipeline/deploy/systemd/blueprint-control-plane-postchecks.sh"
    command = f"{{ path=/bin/bash ; argv[]=/bin/bash {script} --token synthetic-private-value ; }}"
    output = ("FragmentPath=/etc/systemd/system/blueprint-pipeline-control-plane.service\n"
              "DropInPaths=/etc/systemd/system/blueprint-pipeline-control-plane.service.d/override.conf\n"
              f"ExecStartPost={command}\n")
    runner = FakeRunner({"systemctl show": CommandResult(0, output, "")})
    binding = _host(host_tree, runner).notifier_binding()
    assert binding["recognized_script_paths"] == [script]
    assert binding["release_selector_present"] is False
    assert binding["postcheck_command_sha256"] == "sha256:" + hashlib.sha256(command.encode()).hexdigest()
    assert binding["drop_in_paths"] == ["/etc/systemd/system/blueprint-pipeline-control-plane.service.d/override.conf"]
    assert "synthetic-private-value" not in json.dumps(binding)
    assert "argv[]" not in json.dumps(binding)
    assert runner.calls == [["systemctl", "show", "--no-pager", "-p",
                             "FragmentPath,DropInPaths,ExecStartPost", "--",
                             "blueprint-pipeline-control-plane.service"]]


def test_notifier_binding_recognizes_release_selector_as_metadata_only(host_tree: dict[str, Path]) -> None:
    command = "{ argv[]=/bin/bash -lc cd $BLUEPRINT_LIVE_CONTROL_PLANE_REPO && exec /bin/bash deploy/systemd/blueprint-control-plane-postchecks.sh ; }"
    runner = FakeRunner({"systemctl show": CommandResult(0, f"ExecStartPost={command}\n", "")})
    binding = _host(host_tree, runner).notifier_binding()
    assert binding["release_selector_present"] is True
    assert binding["postcheck_present"] is True
    assert binding["claim_ceiling"] == "effective_command_metadata_only"
    assert "source_commit" not in binding


def test_notifier_binding_failure_and_unrecognized_paths_remain_unknown(host_tree: dict[str, Path]) -> None:
    failed = _host(host_tree, FakeRunner({"systemctl show": CommandResult(1, "", "secret stderr")}))
    assert failed.notifier_binding()["status"] == "unavailable"
    output = "FragmentPath=/private/secret/path\nDropInPaths=/private/secret/dropin\nExecStartPost=\n"
    binding = _host(host_tree, FakeRunner({"systemctl show": CommandResult(0, output, "")})).notifier_binding()
    assert binding["status"] == "unknown"
    assert binding["fragment_path"] is None and binding["drop_in_paths"] == []
    assert binding["metadata_paths_complete"] is False
    assert "/private/secret" not in json.dumps(binding)
    secret = "sk-" + "A" * 40
    output = ("DropInPaths=/etc/systemd/system/blueprint-pipeline-control-plane.service.d/"
              f"override-{secret}.conf\n")
    binding = _host(host_tree, FakeRunner({"systemctl show": CommandResult(0, output, "")})).notifier_binding()
    assert binding["drop_in_paths"] == [] and binding["metadata_paths_complete"] is False
    assert secret not in json.dumps(binding)


def test_status_includes_bounded_notifier_binding(host_tree: dict[str, Path]) -> None:
    status = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})
    assert status["notifier_binding"]["unit"] == "blueprint-pipeline-control-plane.service"
    assert status["notifier_binding"]["status"] == "unknown"
