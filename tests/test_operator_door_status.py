"""Host facts and the status document, with every host command faked."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/hostinfo.py
#   deploy/operator-door/operator_door/status.py
#   src/blueprint_pipeline/control_plane_capacity_controller.py

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
    "reclaim_outlook": {"observed_at_epoch": 900.0, "reclaimable_bytes": 0, "volume_growth": "not_configured"},
    "stall": {"observed_at_epoch": 1_000.0, "thresholds_seconds": {"trigger": 3600.0},
              "checks": {"launch_activations": {"status": "observed", "processing": 0, "stuck": 0}}},
    "provider_spend": {"observed_at_epoch": 1_000.0, "lines": {"openai_24h_usd": 75.0},
                       "providers": {"openai": {"status": "observed", "month_to_date_usd": 27.9}}},
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
    assert status["door_requests"] == {"pending": 1, "processing": 0, "parked": 0}
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


def test_preparation_delivery_shows_live_and_parked_callbacks(host_tree: dict[str, Path]) -> None:
    # D-05: parked callbacks leave the due backlog and are counted on their own.
    metrics = host_tree["base"] / "handoffs" / ".website_preparation_delivery_metrics.json"
    metrics.parent.mkdir()
    metrics.write_text(json.dumps({
        "schema_version": "website_preparation_delivery_metrics.v1", "observed_at_epoch": 1_000.0,
        "due_backlog": 0, "oldest_due_age_seconds": 0.0, "overloaded": False, "live_count": 1,
        "oldest_live_age_seconds": 4_000.0, "parked_count": 7, "oldest_parked_age_seconds": 408_000.0,
        "capture_relative_path": "never/projected"}), encoding="utf-8")
    config = DoorConfig(**{**_config(host_tree).__dict__, "handoff_delivery_metrics": str(metrics)})
    status = build_status(config, _host(host_tree, FakeRunner({})), caller={})
    assert status["preparation_delivery"] == {
        "schema_version": "website_preparation_delivery_metrics.v1", "observed_at_epoch": 1_000.0,
        "due_backlog": 0, "oldest_due_age_seconds": 0.0, "overloaded": False, "live_count": 1,
        "oldest_live_age_seconds": 4_000.0, "parked_count": 7, "oldest_parked_age_seconds": 408_000.0}


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
    # The controller writes these three for the door (B-08 spend, D-01 stall, the reclaim outlook).
    assert status["capacity"]["provider_spend"]["providers"]["openai"]["month_to_date_usd"] == 27.9
    assert status["capacity"]["stall"]["checks"]["launch_activations"]["stuck"] == 0
    assert status["capacity"]["reclaim_outlook"]["volume_growth"] == "not_configured"
    assert "project_spend" not in status["capacity"]
    assert set(status["disk"])


def test_every_section_the_capacity_controller_writes_reaches_door_status() -> None:
    from blueprint_pipeline.control_plane_capacity_controller import capacity_summary
    from operator_door.status import _CAPACITY_KEYS

    report = {"level": "ok", "alerts": [], "mounts": [], "usage": {"status": "complete"},
              "volume_resize": {"status": "not_needed"}, "reclaim_outlook": {"reclaimable_bytes": 0},
              "stall": {"observed_at_epoch": 1.0, "checks": {}},
              "provider_spend": {"observed_at_epoch": 1.0, "lines": {}, "providers": {}}}
    assert set(capacity_summary(report)) <= set(_CAPACITY_KEYS)


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


def test_an_inactive_launch_reconciler_timer_is_a_required_trigger() -> None:
    """Leases and provider-zero receipts it files gate succession and completion."""
    from operator_door.config import DoorConfig as _Config
    from operator_door.status import _unit_health

    timer = "blueprint-task-evaluation-launch-reconciler.timer"
    assert timer in _Config.__dataclass_fields__["controller_units"].default
    health = _unit_health([{"Id": timer, "ActiveState": "inactive"}], [])
    assert health["current"] == [{"unit": timer, "state": "required_trigger_inactive", "invocation_id": None}]


MEMINFO = """MemTotal:        7090000 kB
MemFree:          120000 kB
MemAvailable:     900000 kB
Buffers:           10000 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Shmem:            450000 kB
HugePages_Total:       0
"""


def _memory_tree(base: Path, units: dict[str, tuple[str, str | None]]) -> dict[str, Path]:
    cgroup = base / "cgroup"
    system = cgroup / "system.slice"
    system.mkdir(parents=True)
    (system / "memory.current").write_text("4000\n", encoding="utf-8")
    (system / "memory.peak").write_text("5000\n", encoding="utf-8")
    (system / "cgroup.procs").write_text("1\n", encoding="utf-8")
    for name, (current, peak) in units.items():
        (system / name).mkdir()
        (system / name / "memory.current").write_text(current, encoding="utf-8")
        (system / name / "cgroup.procs").write_text("4242\n", encoding="utf-8")
        if peak is not None:
            (system / name / "memory.peak").write_text(peak, encoding="utf-8")
    (cgroup / "user.slice").mkdir()
    (cgroup / "user.slice" / "memory.current").write_text("300\n", encoding="utf-8")
    meminfo = base / "meminfo"
    meminfo.write_text(MEMINFO, encoding="utf-8")
    shm, run = base / "shm", base / "run"
    shm.mkdir()
    run.mkdir()
    mounts = base / "mounts"
    mounts.write_text(f"tmpfs {shm} tmpfs rw,nosuid,nodev 0 0\n"
                      f"/dev/root {run} ext4 rw 0 0\n", encoding="utf-8")
    return {"cgroup": cgroup, "meminfo": meminfo, "mounts": mounts, "shm": shm, "run": run}


def _memory_host(tree: dict[str, Path], paths: dict[str, Path]) -> HostInfo:
    return HostInfo(_config(tree), runner=FakeRunner({}), proc_meminfo_path=str(paths["meminfo"]),
                    cgroup_root=str(paths["cgroup"]), mounts_path=str(paths["mounts"]),
                    tmpfs_paths=(str(paths["shm"]), str(paths["run"]), str(tree["base"] / "tmp")))


def test_parse_meminfo_reports_fixed_fields_in_bytes_and_none_when_absent() -> None:
    from operator_door.hostinfo import parse_meminfo

    assert parse_meminfo(MEMINFO) == {"MemTotal": 7090000 * 1024, "MemAvailable": 900000 * 1024,
                                      "SwapTotal": 0, "SwapFree": 0, "Shmem": 450000 * 1024}
    partial = parse_meminfo("MemTotal: 10 kB\nMemAvailable: lots kB\nShmem: 5 MB\n")
    assert partial == {"MemTotal": 10240, "MemAvailable": None, "SwapTotal": None, "SwapFree": None,
                       "Shmem": None}


def test_parse_cgroup_bytes_accepts_only_a_byte_count() -> None:
    from operator_door.hostinfo import parse_cgroup_bytes

    assert parse_cgroup_bytes(b"123\n") == 123
    assert parse_cgroup_bytes(b"max\n") is None
    assert parse_cgroup_bytes(b"1 2\n") is None
    assert parse_cgroup_bytes(b"") is None
    assert parse_cgroup_bytes(None) is None


def test_memory_reports_top_units_by_current_with_peak_slices_and_tmpfs(host_tree: dict[str, Path]) -> None:
    units = {f"unit-{index:02d}.service": (f"{index * 1000}\n", f"{index * 2000}\n") for index in range(25)}
    units["blueprint-team-scene-packet.service"] = ("2500000000\n", None)  # no memory.peak on this kernel
    units["system-getty.slice"] = ("700\n", "800\n")
    paths = _memory_tree(host_tree["base"], units)
    memory = _memory_host(host_tree, paths).memory()

    assert memory["meminfo"]["MemTotal"] == 7090000 * 1024 and memory["meminfo"]["Shmem"] == 450000 * 1024
    assert memory["slices"] == {"system.slice": {"current_bytes": 4000, "peak_bytes": 5000},
                                "user.slice": {"current_bytes": 300, "peak_bytes": None},
                                "init.scope": None}
    system = memory["system_units"]
    assert system["count"] == 27 and system["scan_truncated"] is False
    top = system["top"]
    assert len(top) == 20
    assert top[0] == {"unit": "blueprint-team-scene-packet.service", "current_bytes": 2500000000, "peak_bytes": None}
    assert top[1] == {"unit": "unit-24.service", "current_bytes": 24000, "peak_bytes": 48000}
    assert [row["current_bytes"] for row in top] == sorted((row["current_bytes"] for row in top), reverse=True)
    # Only the tmpfs mount is sized; a non-tmpfs or unlisted path is None.
    shm = os.statvfs(paths["shm"])
    assert memory["tmpfs"][str(paths["shm"])] == {"total_bytes": shm.f_blocks * shm.f_frsize,
                                                   "used_bytes": (shm.f_blocks - shm.f_bfree) * shm.f_frsize}
    assert memory["tmpfs"][str(paths["run"])] is None
    assert memory["tmpfs"][str(host_tree["base"] / "tmp")] is None
    # Numbers and unit names only: no pids, paths or command lines.
    encoded = json.dumps({key: memory[key] for key in ("meminfo", "slices", "system_units")})
    assert "4242" not in encoded and str(host_tree["base"]) not in encoded
    assert all(set(row) == {"unit", "current_bytes", "peak_bytes"} for row in top)


def test_memory_skips_unsafe_entries_and_oversized_values(host_tree: dict[str, Path]) -> None:
    units = {"good.service": ("10\n", None),
             "huge.service": ("9" * 100 + "\n", None),  # over the read cap: dropped, never parsed
             "max.service": ("max\n", None),
             "not a unit": ("20\n", None)}
    paths = _memory_tree(host_tree["base"], units)
    system = paths["cgroup"] / "system.slice"
    target = host_tree["base"] / "elsewhere.service"
    target.mkdir()
    (target / "memory.current").write_text("99\n", encoding="utf-8")
    os.symlink(target, system / "linked.service")
    (system / "file.service").write_text("", encoding="utf-8")
    (system / "nocurrent.service").mkdir()
    memory = _memory_host(host_tree, paths).memory()
    assert memory["system_units"]["top"] == [{"unit": "good.service", "current_bytes": 10, "peak_bytes": None}]


def test_memory_caps_the_system_slice_scan(host_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from operator_door import hostinfo

    paths = _memory_tree(host_tree["base"], {f"u{index}.service": ("1\n", None) for index in range(10)})
    monkeypatch.setattr(hostinfo, "_MEMORY_SCAN_ENTRIES", 5)
    system = _memory_host(host_tree, paths).memory()["system_units"]
    assert system["scan_truncated"] is True and system["count"] <= 5


def test_memory_fails_soft_to_none_when_nothing_is_readable(host_tree: dict[str, Path]) -> None:
    missing = host_tree["base"] / "missing"
    runner = FakeRunner({"journalctl -k": CommandResult(1, "", "No journal files were found.")})
    host = HostInfo(_config(host_tree), runner=runner, proc_meminfo_path=str(missing / "meminfo"),
                    cgroup_root=str(missing / "cgroup"), mounts_path=str(missing / "mounts"))
    memory = host.memory()
    assert memory == {"meminfo": None,
                      "slices": {"system.slice": None, "user.slice": None, "init.scope": None},
                      "system_units": None,
                      "tmpfs": {"/dev/shm": None, "/run": None, "/tmp": None},
                      "oom_reports": None}
    status = build_status(_config(host_tree), host, caller={})
    assert status["memory"] == memory


def test_status_carries_the_memory_section(host_tree: dict[str, Path]) -> None:
    paths = _memory_tree(host_tree["base"], {"blueprint-pipeline-intake.service": ("1024\n", "2048\n")})
    status = build_status(_config(host_tree), _memory_host(host_tree, paths), caller={})
    assert status["schema"] == "blueprint_operator_door_status.v1"
    assert status["memory"]["system_units"]["top"] == [
        {"unit": "blueprint-pipeline-intake.service", "current_bytes": 1024, "peak_bytes": 2048}]



def _kernel(at: str, *messages: str, host: str = "bp-control-plane") -> str:
    return "".join(f"{at} {host} kernel: {message}\n" for message in messages)


# Linux 6.x (rss_anon/rss_file/rss_shmem columns), journalctl before systemd 254 ("+0000").
OOM_6X = _kernel(
    "2026-11-03T14:39:10+0000",
    "python3 invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0, oom_score_adj=0",
    "CPU: 1 PID: 412233 Comm: python3 Not tainted 6.8.0-1015-gcp #17~22.04.1-Ubuntu",
    "Hardware name: Google Google Compute Engine/Google Compute Engine, BIOS Google 08/09/2024",
    "Call Trace:",
    " <TASK>",
    " dump_stack_lvl+0x76/0xa0",
    " oom_kill_process+0x10e/0x200",
    " </TASK>",
    "Mem-Info:",
    "active_anon:1652 inactive_anon:1884113 isolated_anon:0",
) + "                                                  active_file:120 inactive_file:85 isolated_file:0\n" + _kernel(
    "2026-11-03T14:39:10+0000",
    "audit: type=1326 audit(1793716750.123:45): auid=1000 uid=1000 pid=2210 comm=\"node\" "
    "exe=\"/usr/bin/node\" cmdline=\"node /srv/paperclip/agent.js --token=abc\"",
    "2097050 pages RAM",
    "Tasks state (memory values in pages):",
    "[  pid  ]   uid  tgid total_vm      rss rss_anon rss_file rss_shmem pgtables_bytes swapents oom_score_adj name",
    "[    341]     0   341    67895     1803      320     1483         0   487424        0          -250 systemd-journal",
    "[   1022]   997  1022   402114    21000    20500      500         0  1400832        0             0 python3",
    "[   2210]  1000  2210  2950112   310000   309000     1000         0  5767168        0           200 node",
    "[   2211]  1000  2211  2950112   280000   279000     1000         0  5767168        0           200 node",
    "[   2290]  1000  2290    10850     1200     1000      200         0    73728        0             0 tmux: server",
    *(f"[  {3000 + index}]  1000  {3000 + index}   100000   {100 + index}   {100 + index}        0         0"
      f"    73728        0           200 bash" for index in range(20)),
    "[ 412233]   999 412233   987654   654321   654000      321         0  5767168        0             0 python3",
    "oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,"
    "task_memcg=/system.slice/blueprint-team-scene-packet.service,task=python3,pid=412233,uid=999",
    "Out of memory: Killed process 412233 (python3) total-vm:3950616kB, anon-rss:2616000kB, "
    "file-rss:1284kB, shmem-rss:0kB, UID:999 pgtables:5632kB oom_score_adj:0",
) + _kernel("2026-11-03T14:39:11+0000",
            "oom_reaper: reaped process 412233 (python3), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB")

# Linux 5.x (pgtables_bytes), journalctl from systemd 254 on ("+00:00"), killed in user.slice.
OOM_5X = _kernel(
    "2026-11-03T05:31:39-06:00",
    "node invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=200",
    "CPU: 0 PID: 2210 Comm: node Not tainted 5.15.0-1060-gcp #68~20.04.1-Ubuntu",
    "Tasks state (memory values in pages):",
    "[  pid  ]   uid  tgid total_vm      rss pgtables_bytes swapents oom_score_adj name",
    "[    341]     0   341    67895     1803   487424        0          -250 systemd-journal",
    "[   2210]  1000  2210  2950112   410000  5767168        0           200 node",
    "[  52001]   999 52001   900000   600000  5000000        0             0 python3",
    "oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg="
    "/user.slice/user-1000.slice/user@1000.service/app.slice/paperclip-agent@7.service,task=node,pid=2210,uid=1000",
    "Out of memory: Killed process 2210 (node) total-vm:11800448kB, anon-rss:1640000kB, file-rss:0kB, "
    "shmem-rss:0kB, UID:1000 pgtables:5632kB oom_score_adj:200",
)

# Linux 4.x cgroup v1 memcg OOM: no UID or oom_score_adj on the killed line, "Task in" for the cgroup.
OOM_4X = _kernel(
    "2026-11-03T12:01:10+0000",
    "java invoked oom-killer: gfp_mask=0x24000c0, order=0, oom_score_adj=0",
    "java cpuset=/ mems_allowed=0",
    "CPU: 0 PID: 7001 Comm: java Not tainted 4.15.0-213-generic #224-Ubuntu",
    "Task in /system.slice/blueprint-legacy.service killed as a result of limit of /system.slice/blueprint-legacy.service",
    "memory: usage 1048576kB, limit 1048576kB, failcnt 42",
    "Memory cgroup stats for /system.slice/blueprint-legacy.service: cache:0KB rss:1048576KB",
    "[ pid ]   uid  tgid total_vm      rss nr_ptes nr_pmds swapents oom_score_adj name",
    "[ 7001]   999  7001   812345   262000     900       5        0          -100 java",
    "[ 7010]   999  7010    10000      500      30       3        0             0 sh",
    "Memory cgroup out of memory: Kill process 7001 (java) score 1000 or sacrifice child",
    "Killed process 7001 (java) total-vm:3249380kB, anon-rss:1046000kB, file-rss:2000kB, shmem-rss:0kB",
)

# A table whose header was lost (rate-limited): rows are only counted.
OOM_NO_HEADER = _kernel(
    "2026-11-03T12:57:26+0000",
    "python3 invoked oom-killer: gfp_mask=0xcc0(GFP_KERNEL), order=0, oom_score_adj=0",
    "printk: 3 messages suppressed",
    "[    341]     0   341    67895     1803   487424        0          -250 systemd-journal",
    "[  60001]   999 60001   900000   640000  5000000        0             0 python3",
    "Out of memory: Killed process 60001 (python3) total-vm:3600000kB, anon-rss:2560000kB, file-rss:4kB, "
    "shmem-rss:0kB, UID:999 pgtables:5000kB oom_score_adj:0",
)


def test_parse_oom_reports_reads_a_6x_report() -> None:
    from operator_door.hostinfo import parse_oom_reports

    parsed = parse_oom_reports(OOM_6X, page_size=4096)
    assert parsed["count"] == 1
    (event,) = parsed["events"]
    assert event["at"] == "2026-11-03T14:39:10Z" and event["invoker"] == "python3"
    assert event["constraint"] == {"constraint": "CONSTRAINT_NONE", "global_oom": True, "oom_memcg": None,
                                   "task_memcg": "/system.slice/blueprint-team-scene-packet.service"}
    assert event["killed"] == {"pid": 412233, "name": "python3", "uid": 999, "total_vm_kb": 3950616,
                               "anon_rss_kb": 2616000, "file_rss_kb": 1284, "shmem_rss_kb": 0,
                               "oom_score_adj": 0}
    tasks = event["tasks"]
    assert tasks["rows"] == 26 and tasks["unparsed_rows"] == 0 and tasks["parsed"] is True
    top = tasks["top"]
    assert len(top) == 15
    assert top[0] == {"pid": 412233, "uid": 999, "name": "python3", "rss_pages": 654321,
                      "total_vm_pages": 987654, "oom_score_adj": 0, "rss_bytes": 654321 * 4096}
    assert [row["pid"] for row in top[1:5]] == [2210, 2211, 1022, 341]
    assert top[5]["name"] == "tmux: server" and top[6] == {
        "pid": 3019, "uid": 1000, "name": "bash", "rss_pages": 119, "total_vm_pages": 100000,
        "oom_score_adj": 200, "rss_bytes": 119 * 4096}
    assert [row["rss_pages"] for row in top] == sorted((row["rss_pages"] for row in top), reverse=True)
    assert tasks["total_rss_pages"] == 654321 + 310000 + 280000 + 21000 + 1803 + 1200 + sum(range(100, 120))
    assert tasks["by_uid"][0] == {"uid": 999, "tasks": 1, "rss_pages": 654321}
    assert tasks["by_uid"][1] == {"uid": 1000, "tasks": 23,
                                  "rss_pages": 310000 + 280000 + 1200 + sum(range(100, 120))}


def test_parse_oom_reports_reads_5x_and_4x_memcg_reports() -> None:
    from operator_door.hostinfo import parse_oom_reports

    parsed = parse_oom_reports(OOM_4X + OOM_5X, page_size=None)
    assert parsed["count"] == 2
    old, new = parsed["events"]
    assert old["at"] == "2026-11-03T12:01:10Z" and old["invoker"] == "java"
    assert old["constraint"] == {"constraint": None, "global_oom": None,
                                 "oom_memcg": "/system.slice/blueprint-legacy.service",
                                 "task_memcg": "/system.slice/blueprint-legacy.service"}
    # The killed line has no UID or oom_score_adj on 4.x: both come from the task table.
    assert old["killed"] == {"pid": 7001, "name": "java", "uid": 999, "total_vm_kb": 3249380,
                             "anon_rss_kb": 1046000, "file_rss_kb": 2000, "shmem_rss_kb": 0,
                             "oom_score_adj": -100}
    assert old["tasks"]["top"][0] == {"pid": 7001, "uid": 999, "name": "java", "rss_pages": 262000,
                                      "total_vm_pages": 812345, "oom_score_adj": -100, "rss_bytes": None}
    assert new["at"] == "2026-11-03T11:31:39Z" and new["invoker"] == "node"
    assert new["constraint"]["task_memcg"] == (
        "/user.slice/user-1000.slice/user@1000.service/app.slice/paperclip-agent@7.service")
    assert new["killed"]["pid"] == 2210 and new["killed"]["uid"] == 1000
    assert new["killed"]["anon_rss_kb"] == 1640000 and new["killed"]["oom_score_adj"] == 200
    assert [row["pid"] for row in new["tasks"]["top"]] == [52001, 2210, 341]


def test_parse_oom_reports_keeps_counts_and_the_killed_line_when_the_table_is_unrecognised() -> None:
    from operator_door.hostinfo import parse_oom_reports

    bad_header = OOM_5X.replace("rss pgtables_bytes", "resident pgtables_bytes")
    for text in (OOM_NO_HEADER, bad_header):
        (event,) = parse_oom_reports(text, page_size=4096)["events"]
        assert event["killed"]["pid"] in {60001, 2210} and event["killed"]["anon_rss_kb"] is not None
        rows = 2 if text is OOM_NO_HEADER else 3
        assert event["tasks"] == {"rows": rows, "unparsed_rows": rows, "parsed": False, "total_rss_pages": None,
                                  "top": None, "by_uid": None, "by_uid_truncated": None}


def test_parse_oom_reports_keeps_the_last_n_and_an_unfinished_report() -> None:
    from operator_door.hostinfo import parse_oom_reports

    unfinished = _kernel("2026-11-03T15:00:00+0000", "python3 invoked oom-killer: gfp_mask=0xcc0, order=0",
                         "Tasks state (memory values in pages):")
    text = OOM_5X + OOM_4X + OOM_NO_HEADER + OOM_6X + unfinished
    parsed = parse_oom_reports(text, page_size=4096, limit=3)
    assert parsed["count"] == 5
    assert [event["at"] for event in parsed["events"]] == [
        "2026-11-03T12:57:26Z", "2026-11-03T14:39:10Z", "2026-11-03T15:00:00Z"]
    assert parsed["events"][-1]["killed"] is None and parsed["events"][-1]["tasks"]["rows"] == 0
    assert parse_oom_reports("", page_size=4096) == {"count": 0, "events": []}
    assert parse_oom_reports("-- No entries --\n" + OOM_6X.split("Tasks state")[1], page_size=4096) == {
        "count": 0, "events": []}  # a report whose start fell outside the window is skipped


def test_parse_oom_reports_returns_no_command_lines_paths_or_host_details() -> None:
    from operator_door.hostinfo import parse_oom_reports

    odd = OOM_5X.replace(" python3\n", " python3 /srv/agents/run.py --api-key=sk-live-abcdefghijklmnop\n").replace(
        "task_memcg=/user.slice", "task_memcg=/kubepods/burstable/pod-1/../user.slice")
    encoded = json.dumps([parse_oom_reports(OOM_6X + odd, page_size=4096)])
    for leaked in ("bp-control-plane", "/usr/bin", "/srv", "cmdline", "token", "sk-live", "Hardware", "6.8.0",
                   "5.15.0", "Google", "cpuset", "gfp_mask", "kubepods", "dump_stack"):
        assert leaked not in encoded
    (_, event) = json.loads(encoded)[0]["events"]
    assert event["constraint"]["task_memcg"] is None
    assert [row["name"] for row in event["tasks"]["top"]] == [None, "node", "systemd-journal"]


class _KernelLogRunner(FakeRunner):
    def __init__(self, result: CommandResult) -> None:
        super().__init__({"journalctl -k": result})
        self.timeouts: list[float] = []

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        self.timeouts.append(timeout)
        return super().run(argv, timeout)


def test_oom_reports_read_a_bounded_kernel_journal(host_tree: dict[str, Path]) -> None:
    runner = _KernelLogRunner(CommandResult(0, OOM_5X + OOM_6X, ""))
    paths = _memory_tree(host_tree["base"], {})
    host = HostInfo(_config(host_tree), runner=runner, proc_meminfo_path=str(paths["meminfo"]),
                    cgroup_root=str(paths["cgroup"]), mounts_path=str(paths["mounts"]))
    reports = host.memory()["oom_reports"]
    assert runner.calls == [["journalctl", "-k", "--since=-24h", "-n", "20000", "-o", "short-iso",
                             "--no-pager", "-q"]]
    assert runner.timeouts == [10]
    assert reports["since"] == "-24h" and reports["page_size_bytes"] == os.sysconf("SC_PAGE_SIZE")
    assert reports["log_truncated"] is False and reports["count"] == 2
    assert [event["killed"]["pid"] for event in reports["events"]] == [2210, 412233]


def test_oom_reports_keep_the_newest_tail_of_an_oversized_log(
        host_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from operator_door import hostinfo

    monkeypatch.setattr(hostinfo, "_OOM_LOG_BYTES", len(OOM_6X) + 10)
    host = HostInfo(_config(host_tree), runner=FakeRunner({"journalctl -k": CommandResult(0, OOM_5X + OOM_6X, "")}))
    reports = host._oom_reports()
    assert reports["log_truncated"] is True and reports["count"] == 1
    assert reports["events"][0]["killed"]["pid"] == 412233

    monkeypatch.setattr(hostinfo, "_OOM_LOG_LINES", 5)
    runner = FakeRunner({"journalctl -k": CommandResult(0, OOM_NO_HEADER, "")})
    reports = HostInfo(_config(host_tree), runner=runner)._oom_reports()
    assert runner.calls[0][3:5] == ["-n", "5"] and reports["log_truncated"] is True


def test_oom_reports_fail_soft_to_none(host_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from operator_door import hostinfo

    for result in (CommandResult(1, "", "Failed to open journal"), CommandResult(124, "", "timeout"),
                   CommandResult(127, "", "FileNotFoundError")):
        assert HostInfo(_config(host_tree), runner=FakeRunner({"journalctl -k": result}))._oom_reports() is None

    def broken(text: str, *, page_size: int | None, limit: int = 4) -> dict:
        raise IndexError("parser surprise")

    monkeypatch.setattr(hostinfo, "parse_oom_reports", broken)
    paths = _memory_tree(host_tree["base"], {})
    memory = _memory_host(host_tree, paths).memory()
    assert memory["oom_reports"] is None and memory["meminfo"]["MemTotal"] == 7090000 * 1024


def test_status_carries_the_oom_reports(host_tree: dict[str, Path]) -> None:
    paths = _memory_tree(host_tree["base"], {})
    host = HostInfo(_config(host_tree), runner=FakeRunner({"journalctl -k": CommandResult(0, OOM_6X, "")}),
                    proc_meminfo_path=str(paths["meminfo"]), cgroup_root=str(paths["cgroup"]),
                    mounts_path=str(paths["mounts"]))
    status = build_status(_config(host_tree), host, caller={})
    assert status["memory"]["oom_reports"]["events"][0]["killed"]["name"] == "python3"


def test_status_says_which_paid_stage_holds_the_deploy(host_tree: dict[str, Path]) -> None:
    parked = host_tree["base"] / "door" / "requests" / "parked"
    parked.mkdir(exist_ok=True)
    live = [{"stage": "beta_screening", "state": "screening_running", "id": "camp-1/cand-1"}]
    (parked / ".activation-hold").write_text(json.dumps({
        "schema": "blueprint_operator_door_activation_hold.v1", "active_release": "a" * 40,
        "first_held_at_epoch": 1.0, "cap_at_epoch": 2.0, "recheck_after_epoch": 3.0, "live": live,
        "stages": ["beta_screening"], "requests": {}}), encoding="utf-8")
    hold = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})["deploy_activation_hold"]
    assert hold == {"active_release": "a" * 40, "first_held_at_epoch": 1.0, "cap_at_epoch": 2.0, "bound": None,
                    "recheck_after_epoch": 3.0, "live": live, "stages": ["beta_screening"],
                    "paid_stages": ["beta_screening"], "unpaid_stages": []}
    # A record an older door wrote names no stage: it was an activation, which is paid.
    (parked / ".activation-hold").write_text(json.dumps({
        "active_release": "a" * 40, "live": [{"activation_id": "act-1", "state": "pending"}]}), encoding="utf-8")
    hold = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})["deploy_activation_hold"]
    assert hold["stages"] == ["scene_configuration_activation"] and hold["live"][0]["activation_id"] == "act-1"
    assert hold["paid_stages"] == ["scene_configuration_activation"] and hold["unpaid_stages"] == []
    (parked / ".activation-hold").unlink()
    assert build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})["deploy_activation_hold"] is None


def test_status_says_whether_the_work_holding_the_deploy_is_paid(host_tree: dict[str, Path]) -> None:
    parked = host_tree["base"] / "door" / "requests" / "parked"
    parked.mkdir(exist_ok=True)
    live = [{"stage": "scene_configuration_preparation", "state": "preparation_processing", "id": "prep-1"},
            {"stage": "paid_launch_lock", "state": "paid_launch_lock_held", "id": "vast_paid_launch.lock"}]
    (parked / ".activation-hold").write_text(json.dumps({
        "schema": "blueprint_operator_door_activation_hold.v1", "active_release": "a" * 40, "live": live,
        "stages": ["paid_launch_lock", "scene_configuration_preparation"],
        "paid_stages": ["paid_launch_lock"], "unpaid_stages": ["scene_configuration_preparation"],
        "cap_at_epoch": 9.0, "bound": "paid"}), encoding="utf-8")
    hold = build_status(_config(host_tree), _host(host_tree, FakeRunner({})), caller={})["deploy_activation_hold"]
    assert hold["paid_stages"] == ["paid_launch_lock"]
    assert hold["unpaid_stages"] == ["scene_configuration_preparation"]
    assert (hold["bound"], hold["cap_at_epoch"]) == ("paid", 9.0)
