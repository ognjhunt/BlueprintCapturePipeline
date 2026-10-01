"""ADP accounting freshness precedes capacity without granting execution authority.

Native cases use disposable units on an ephemeral hosted CI runner only. They
copy dependency directives, never execute production unit commands, and run the
real local publisher/observer against synthetic receipts with no network.
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.test_task_evaluation_scene_spend import observed_monitor

REPO = Path(__file__).resolve().parents[1]
UNITS = REPO / "deploy" / "systemd"
CAPACITY = "blueprint-control-plane-capacity.service"
PUBLISHER = "blueprint-scene-project-spend-refresh.service"


def _unit(name: str) -> configparser.ConfigParser:
    result = configparser.ConfigParser(interpolation=None, strict=False)
    result.optionxform = str
    result.read(UNITS / name)
    return result


def test_capacity_orders_the_optional_publisher_without_requiring_success() -> None:
    unit = _unit(CAPACITY)["Unit"]
    assert PUBLISHER in unit["Wants"].split()
    assert PUBLISHER in unit["After"].split()
    for directive in ("Requires", "Requisite", "BindsTo"):
        assert PUBLISHER not in unit.get(directive, "").split()


def test_capacity_dependency_graph_does_not_activate_gc_or_dispatch() -> None:
    pending = [CAPACITY]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if not (UNITS / name).is_file():
            continue
        unit = _unit(name)["Unit"]
        for directive in ("Wants", "Requires", "Requisite", "BindsTo", "Upholds"):
            pending.extend(unit.get(directive, "").split())
    assert PUBLISHER in visited
    assert not any("gc" in name or "dispatcher" in name for name in visited)
    publisher = _unit(PUBLISHER)["Service"]
    assert publisher["Type"] == "oneshot"
    assert publisher.get("RemainAfterExit", "no") == "no"
    assert publisher["TimeoutStartSec"] == "360"
    assert publisher["PrivateNetwork"] == "true"
    timer = _unit("blueprint-control-plane-capacity.timer")["Timer"]
    assert timer["Unit"] == CAPACITY
    assert "OnUnitActiveSec" in timer  # native ticks accelerate this same trigger


PROBE = '''import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

role, directory = sys.argv[1:]
root = Path(directory)

def event(stage, **fields):
    row = dict(stage=stage, monotonic=time.monotonic(), pid=os.getpid(), **fields)
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\\n")

event(role + "_start")
if role == "publisher":
    mode = (root / "mode").read_text()
    if mode == "failure":
        event("publisher_failure")
        sys.exit(7)
    if mode == "timeout":
        time.sleep(30)
    if mode == "blocked":
        deadline = time.monotonic() + 8
        while not (root / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("fixture release deadline")
            time.sleep(0.02)
    from blueprint_pipeline.task_evaluation_scene_spend import refresh_configured_scene_project_spend
    publication = refresh_configured_scene_project_spend()
    event("publisher_finish", observed=publication["pointer"]["observed_at_epoch"])
elif role == "capacity":
    from blueprint_pipeline import control_plane_capacity_controller as capacity
    report = capacity.run_controller(
        mounts=[str(root)], report_root=root / "capacity",
        reservation_root=root / "reservations", volume=None, ack="", token="",
        survey=None, webhook_url="https://fixture.invalid/never-requested",
        poster=lambda *_args: None,
        disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, used=10 * 1024**3, free=90 * 1024**3),
        release_retirement_summary_path=root / "absent-retirement",
        break_glass_notes_root=root / "absent-notes",
    )
    pointer = json.loads((root / "current.json").read_text())
    event("capacity_finish", observed=pointer["observed_at_epoch"],
          alerts=[row["code"] for row in report["alerts"]])
else:
    raise RuntimeError("unexpected fixture role")
'''


def _quote(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


class NativeUnits:
    def __init__(self, root: Path, *, publisher_timeout: str = "8s") -> None:
        self.root = root
        prefix = "blueprint-spend-order-test-" + uuid.uuid4().hex
        self.publisher = prefix + "-publisher.service"
        self.capacity = prefix + "-capacity.service"
        self.timer = prefix + "-capacity.timer"
        self.held = prefix + "-dispatcher.timer"
        self.names = [self.timer, self.capacity, self.publisher, self.held]
        self.command = (["sudo", "-n"] if os.geteuid() else []) + ["systemctl"]
        (root / "probe.py").write_text(PROBE)
        (root / "mode").write_text("success")
        self.files = []
        for name, role in [(self.publisher, "publisher"), (self.capacity, "capacity")]:
            dependencies = ""
            if role == "capacity":
                declared = _unit(CAPACITY)["Unit"]
                for directive in ("Wants", "After", "Requires", "Requisite", "BindsTo"):
                    if PUBLISHER in declared.get(directive, "").split():
                        dependencies += f"{directive}={self.publisher}\n"
            text = (
                "[Unit]\n" + dependencies + "StartLimitIntervalSec=0\n"
                "[Service]\nType=oneshot\n"
                f"User={os.getuid()}\nGroup={os.getgid()}\n"
                "NoNewPrivileges=true\nPrivateNetwork=true\nProtectSystem=strict\n"
                f"ReadWritePaths={_quote(root)}\n"
                f"TimeoutStartSec={publisher_timeout if role == 'publisher' else '8s'}\n"
                "TimeoutStopSec=1s\n"
                f"Environment={_quote('PYTHONPATH=' + str(REPO / 'src'))}\n"
                f"Environment={_quote('BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG=' + str(root / 'monitor.json'))}\n"
                f"ExecStart={_quote(sys.executable)} {_quote(root / 'probe.py')} {role} {_quote(root)}\n"
            )
            self._file(name, text)
        self._file(self.timer, (
            "[Unit]\n[Timer]\nOnActiveSec=30ms\nOnUnitActiveSec=2s\n"
            f"AccuracySec=1ms\nRandomizedDelaySec=0\nUnit={self.capacity}\n"
        ))
        # An unrelated, unarmed dispatcher represents an existing owner freeze.
        # Its dependency would run capacity again, making accidental activation visible.
        self._file(self.held, (
            "[Unit]\n[Timer]\nOnActiveSec=10ms\n"
            f"AccuracySec=1ms\nUnit={self.capacity}\n"
        ))

    def _file(self, name: str, text: str) -> None:
        path = self.root / name
        path.write_text(text)
        self.files.append(str(path))

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        # Arguments below are fixed verbs and this instance's unique fixture units.
        return subprocess.run(
            [*self.command, *args], capture_output=True, text=True,
            check=check, timeout=15,
        )

    def events(self, stage: str | None = None) -> list[dict]:
        path = self.root / "events.jsonl"
        rows = [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []
        return [row for row in rows if stage is None or row["stage"] == stage]

    def wait(self, stage: str, *, count: int = 1) -> list[dict]:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            rows = self.events(stage)
            if len(rows) >= count:
                return rows
            time.sleep(0.03)
        diagnostics = self.run("show", *self.names, check=False)
        pytest.fail(f"missing {stage} count={count}: {self.events()}\n{diagnostics.stdout}")

    def properties(self, unit: str) -> dict[str, str]:
        result = self.run("show", "--property=ActiveState,UnitFileState,Result", unit)
        return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    def close(self) -> None:
        self.run("stop", *self.names, check=False)
        self.run("reset-failed", *self.names, check=False)
        self.run("disable", "--runtime", *self.names, check=False)


@pytest.fixture
def native_units(tmp_path: Path, monkeypatch):
    hosted = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("RUNNER_ENVIRONMENT") == "github-hosted"
    opted_in = os.getenv("BLUEPRINT_TEST_SYSTEMD_ORDERING") == "1"
    if not (hosted or opted_in):
        pytest.skip("native ordering requires an ephemeral systemd test runner")
    if not Path("/run/systemd/system").is_dir() or not shutil.which("systemctl"):
        pytest.fail("native ordering runner has no systemd system manager")
    if os.geteuid():
        subprocess.run(["sudo", "-n", "true"], check=True, timeout=5)
    observed_monitor(tmp_path, monkeypatch)  # synthetic, deliberately stale publication
    instances = []

    def create(*, publisher_timeout: str = "8s") -> NativeUnits:
        instance = NativeUnits(tmp_path, publisher_timeout=publisher_timeout)
        instances.append(instance)
        instance.run("link", "--runtime", *instance.files)
        return instance

    yield create
    for instance in reversed(instances):
        instance.close()


def _assert_fresh_after_publication(instance: NativeUnits, count: int = 1) -> None:
    capacity = instance.wait("capacity_finish", count=count)
    finished = instance.events("publisher_finish")
    started = instance.events("capacity_start")
    for index in range(count):
        assert finished[index]["monotonic"] < started[index]["monotonic"]
        assert capacity[index]["observed"] == finished[index]["observed"]
        assert "project_spend_observation_blocked" not in capacity[index]["alerts"]


def test_native_first_capacity_start_waits_for_a_fresh_publication(native_units) -> None:
    instance = native_units()
    frozen = instance.properties(instance.held)
    instance.run("start", instance.capacity)
    _assert_fresh_after_publication(instance)
    assert instance.properties(instance.publisher)["ActiveState"] == "inactive"
    assert instance.properties(instance.held) == frozen
    assert frozen["ActiveState"] == "inactive"


@pytest.mark.parametrize("mode, result", [("failure", "exit-code"), ("timeout", "timeout")])
def test_native_failed_or_timed_out_publisher_still_reports_stale_accounting(native_units, mode, result) -> None:
    instance = native_units(publisher_timeout="2s" if mode == "timeout" else "8s")
    (instance.root / "mode").write_text(mode)
    before = (instance.root / "current.json").read_bytes()
    instance.run("start", instance.capacity)
    [observed] = instance.wait("capacity_finish")
    assert "project_spend_observation_blocked" in observed["alerts"]
    assert instance.properties(instance.publisher)["Result"] == result
    assert (instance.root / "current.json").read_bytes() == before
    assert instance.events("publisher_finish") == []
    # A later start retries the same publisher after failure, without reset-failed.
    (instance.root / "mode").write_text("success")
    instance.run("start", instance.capacity)
    second = instance.wait("capacity_finish", count=2)[1]
    assert "project_spend_observation_blocked" not in second["alerts"]
    assert len(instance.events("publisher_start")) == 2


def test_native_concurrent_starts_join_one_in_flight_publication(native_units) -> None:
    instance = native_units()
    (instance.root / "mode").write_text("blocked")
    instance.run("start", "--no-block", instance.publisher)
    instance.wait("publisher_start")
    instance.run("start", "--no-block", instance.capacity)
    instance.run("start", "--no-block", instance.capacity)
    instance.run("start", "--no-block", instance.publisher)
    time.sleep(0.1)
    assert len(instance.events("publisher_start")) == 1
    assert instance.events("capacity_start") == []
    (instance.root / "release").touch()
    _assert_fresh_after_publication(instance)
    assert len(instance.events("publisher_start")) == 1
    assert len(instance.events("capacity_start")) == 1


def test_native_repeated_timer_ticks_refresh_again_and_preserve_a_dispatcher_freeze(native_units) -> None:
    instance = native_units()
    frozen = instance.properties(instance.held)
    instance.run("start", instance.timer)
    _assert_fresh_after_publication(instance, count=3)
    instance.run("stop", instance.timer)
    finished = instance.events("publisher_finish")
    assert finished[0]["observed"] < finished[1]["observed"] < finished[2]["observed"]
    assert len({row["pid"] for row in instance.events("publisher_start")}) >= 3
    assert instance.properties(instance.held) == frozen
    assert frozen["ActiveState"] == "inactive"
