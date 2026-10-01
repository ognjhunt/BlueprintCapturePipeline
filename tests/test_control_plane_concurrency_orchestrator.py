"""A failed child cannot be retired or transformed into a successful summary."""

import pytest
import json
import sys
import subprocess
from types import SimpleNamespace


def test_checkout_accepts_generated_metadata_but_refuses_changed_or_untracked_code(tmp_path):
    from scripts.control_plane_concurrency_orchestrator import require_clean_checkout

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "scripts/original.py"
    source.parent.mkdir()
    source.write_text("original = True\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "scripts/original.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Contract Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "source",
        ],
        check=True,
    )
    (tmp_path / "generated-ci-report.json").write_text("{}")
    require_clean_checkout(tmp_path)
    extra = source.parent / "untracked.py"
    extra.write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="harness_untracked_code"):
        require_clean_checkout(tmp_path)
    extra.unlink()
    package = tmp_path / "blueprint_pipeline"
    package.mkdir()
    shadow = package / "__init__.py"
    shadow.write_text("shadow = True\n")
    with pytest.raises(ValueError, match="harness_untracked_code"):
        require_clean_checkout(tmp_path)
    shadow.unlink()
    source.write_text("original = False\n")
    with pytest.raises(ValueError, match="harness_checkout_dirty"):
        require_clean_checkout(tmp_path)


def test_child_failure_and_timeout_are_terminal_without_retirement(tmp_path):
    from scripts.control_plane_concurrency_orchestrator import child_failure

    class Process:
        pid = 123

        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    assert (
        child_failure(
            [{"process": Process(1), "started": 0, "scene_key": "one"}],
            now=1,
            child_timeout=10,
            global_deadline=20,
        )
        == "child_failed:one:1"
    )
    assert (
        child_failure(
            [{"process": Process(None), "started": 0, "scene_key": "one"}],
            now=11,
            child_timeout=10,
            global_deadline=20,
        )
        == "child_timeout:one"
    )
    assert (
        child_failure(
            [{"process": Process(None), "started": 0, "scene_key": "one"}],
            now=21,
            child_timeout=30,
            global_deadline=20,
        )
        == "global_timeout"
    )


def test_live_acceptance_requires_kernel_confinement_before_creating_roots(tmp_path):
    from scripts.control_plane_concurrency_orchestrator import require_kernel_confinement

    with pytest.raises(ValueError, match="harness_kernel_confinement_required"):
        require_kernel_confinement(process_root=tmp_path / "unavailable-proc")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["initial_measurement", "synchronous_builder"])
def test_parent_failure_or_watchdog_seals_failed_report(tmp_path, monkeypatch, failure):
    import time
    from scripts import control_plane_concurrency_orchestrator as module
    from tests.test_task_evaluation_completed_scene_progression import _config
    from tests import test_task_evaluation_completed_scene_progression as fixture
    from tests import test_task_evaluation_scene_configuration_submission as template

    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(fixture, "SHA", source)
    monkeypatch.setattr(template, "SHA", source)
    configuration = tmp_path / "configuration"
    configuration.mkdir()
    _config(configuration, monkeypatch)
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    monkeypatch.setattr(module, "require_kernel_confinement", lambda: {"contract_fixture": True})
    if failure == "initial_measurement":

        def unreadable(root):
            raise ValueError("allocation_measurement_incomplete")

        monkeypatch.setattr(module, "allocated_tree_bytes", unreadable)
    else:
        monkeypatch.setattr(module, "_runtime_bundle", lambda *args: time.sleep(2))
    arguments = SimpleNamespace(
        report=run / "report.json",
        control_plane_root=run / "control",
        object_store_root=run / "objects",
        worker_root=run / "workers",
        release_binding=configuration / "release.json",
        runtime_source_root=tmp_path / "packet",
        expected_beta_concurrency=1,
        owner_confirmed_concurrency=False,
        maximum_retained_gib=0.25,
        maximum_delta_gib=0.25,
        child_timeout_seconds=120,
        global_timeout_seconds=0.1,
    )
    started = time.monotonic()
    assert module.run_benchmark(arguments) == 1
    assert time.monotonic() - started < 1.5
    report = json.loads(arguments.report.read_text())
    assert report["status"] == "failed" and report["owner_sized_acceptance_complete"] is False
    expected = (
        "allocation_measurement_incomplete"
        if failure == "initial_measurement"
        else "global_timeout"
    )
    assert any(expected in value for value in report["blockers"])
    if failure == "initial_measurement":
        assert report["baseline_allocated_bytes"] is report["final_allocated_bytes"] is None


@pytest.mark.slow
def test_two_joined_children_share_original_holds_and_retire_only_after_join(tmp_path, monkeypatch):
    from scripts import control_plane_concurrency_orchestrator as module
    from scripts import control_plane_concurrency_retirement as retirement
    from tests.test_native_task_arena_bundle import _runtime_source_packet
    from tests.test_task_evaluation_completed_scene_progression import _config
    from tests import test_task_evaluation_completed_scene_progression as fixture
    from tests import test_task_evaluation_scene_configuration_submission as template

    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(fixture, "SHA", source)
    monkeypatch.setattr(template, "SHA", source)
    configuration = tmp_path / "configuration"
    configuration.mkdir()
    _config(configuration, monkeypatch)
    packet = _runtime_source_packet(tmp_path)
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(module, "require_kernel_confinement", lambda: {"contract_fixture": True})
    real_runtime = module._runtime_bundle

    def fixture_runtime(*args):
        value = real_runtime(*args)
        value["source_kind"] = "contract_fixture"
        return value

    monkeypatch.setattr(module, "_runtime_bundle", fixture_runtime)
    real_retire = retirement.retire_fixture_chains
    monkeypatch.setattr(
        retirement,
        "retire_fixture_chains",
        lambda **kwargs: real_retire(**kwargs, process_root=proc),
    )
    real_popen = module.subprocess.Popen
    driver = """import sys
from pathlib import Path
from collections import namedtuple
from blueprint_pipeline import control_plane_disk_budget as budget
from blueprint_pipeline import task_evaluation_launch_preparation_worker as preparation
from blueprint_pipeline import task_evaluation_episode_compilation_worker as compilation
from scripts import control_plane_concurrency_policy as policy
original,headroom=budget.reserve_control_plane_disk,budget.disk_headroom
usage=namedtuple('Usage','total used free')(200*1024**3,50*1024**3,150*1024**3)
def measured(role,**kwargs):return original(role,**kwargs,disk_usage=lambda _:usage)
for module in (budget,preparation,compilation,policy):module.reserve_control_plane_disk=measured
budget.disk_headroom=lambda **kwargs:headroom(**kwargs,disk_usage=lambda _:usage)
from scripts.control_plane_concurrency_protocol import run_child
raise SystemExit(run_child(Path(sys.argv[-1])))
"""

    def launch(command, *args, **kwargs):
        if command[1:3] == ["-m", "scripts.control_plane_concurrency_protocol"]:
            command = [sys.executable, "-c", driver, *command[3:]]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    arguments = SimpleNamespace(
        report=run / "report.json",
        control_plane_root=run / "control",
        object_store_root=run / "objects",
        worker_root=run / "workers",
        release_binding=configuration / "release.json",
        runtime_source_root=packet.parent,
        expected_beta_concurrency=1,
        owner_confirmed_concurrency=False,
        maximum_retained_gib=0.25,
        maximum_delta_gib=0.25,
        child_timeout_seconds=120,
        global_timeout_seconds=240,
    )
    result = module.run_benchmark(arguments)
    report = json.loads(arguments.report.read_text())
    assert result == 0, report["blockers"]
    assert report["completed_scenes"] == report["measured_peak_concurrency"] == 2
    assert report["concurrency_hold_seconds"] >= 2
    assert report["kernel_confinement"] == {"contract_fixture": True}
    assert report["owner_sized_acceptance_complete"] is False
    assert report["retirement"]["residual_pins"] == report["retirement"]["residual_leases"] == 0
    assert all(scene["runtime_source_kind"] == "contract_fixture" for scene in report["scenes"])
