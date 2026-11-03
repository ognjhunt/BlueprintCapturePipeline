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
    host = HostInfo(_config(host_tree), runner=FakeRunner({}), proc_meminfo_path=str(missing / "meminfo"),
                    cgroup_root=str(missing / "cgroup"), mounts_path=str(missing / "mounts"))
    memory = host.memory()
    assert memory == {"meminfo": None,
                      "slices": {"system.slice": None, "user.slice": None, "init.scope": None},
                      "system_units": None,
                      "tmpfs": {"/dev/shm": None, "/run": None, "/tmp": None}}
    status = build_status(_config(host_tree), host, caller={})
    assert status["memory"] == memory


def test_status_carries_the_memory_section(host_tree: dict[str, Path]) -> None:
    paths = _memory_tree(host_tree["base"], {"blueprint-pipeline-intake.service": ("1024\n", "2048\n")})
    status = build_status(_config(host_tree), _memory_host(host_tree, paths), caller={})
    assert status["schema"] == "blueprint_operator_door_status.v1"
    assert status["memory"]["system_units"]["top"] == [
        {"unit": "blueprint-pipeline-intake.service", "current_bytes": 1024, "peak_bytes": 2048}]
