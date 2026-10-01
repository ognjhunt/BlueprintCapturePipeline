"""The joined runner must retire the actual compiled chain, not synthetic rows."""
from collections import namedtuple
import subprocess
import threading

import pytest


@pytest.mark.slow
def test_joined_runner_keeps_both_activation_bindings_and_retires_real_cache(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_disk_budget as budget
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as preparation
    from blueprint_pipeline import task_evaluation_episode_compilation_worker as compilation
    from scripts import control_plane_concurrency_policy as policy
    from scripts.control_plane_concurrency_load_test import create_run_roots, REQUIRED_STAGES
    from scripts.control_plane_concurrency_runner import run_scene
    from scripts.control_plane_concurrency_retirement import retire_fixture_chains
    from scripts.control_plane_concurrency_provider import fixture_publisher
    from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import build_task_evaluation_runtime_source_bundle
    from tests.test_native_task_arena_bundle import _runtime_source_packet
    from tests.test_task_evaluation_configured_controls_progression import _runtime
    usage = namedtuple("Usage", "total used free")(200*1024**3, 50*1024**3, 150*1024**3)
    real_reserve, real_headroom = budget.reserve_control_plane_disk, budget.disk_headroom
    def measured_capacity(role, **kwargs):
        # Only this local contract test supplies capacity. Live runner admission
        # reads the real filesystem and shares the original reservation ledger.
        return real_reserve(role, **kwargs, disk_usage=lambda _: usage)
    for module in (budget, preparation, compilation, policy):
        monkeypatch.setattr(module, "reserve_control_plane_disk", measured_capacity)
    monkeypatch.setattr(budget, "disk_headroom", lambda **kwargs:
        real_headroom(**kwargs, disk_usage=lambda _: usage))
    source = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    roots = create_run_roots(tmp_path/"control", tmp_path/"objects", tmp_path/"workers")
    packet = _runtime_source_packet(tmp_path)
    bundle = roots["workers"]/"runtime-source.zip"
    build_task_evaluation_runtime_source_bundle(source_root=packet.parent, output_path=bundle,
        expected_production_commit=source, runtime_identity=_runtime()["runtime"]["identity"])
    published = fixture_publisher(roots["objects"])(path=bundle, object_name="runtime-source.zip")
    scene = run_scene(scene_key="joined-1",control_root=roots["control_plane"],
        object_root=roots["objects"],worker_root=roots["workers"],source_commit=source,
        runtime_bundle={"reference":{k:published[k] for k in ("uri","digest","size_bytes")},
                        "source_kind":"contract_fixture"},release_binding=None,
        heavy_slot=threading.Semaphore(1))
    assert [row["stage"] for row in scene["stages"]] == list(REQUIRED_STAGES[:-1])
    assert scene["actual_provider_calls"] == 0
    assert scene["configuration_activation"]["preparer"]["status"] == "prepared"
    assert scene["native_activation"]["preparer"]["status"] == "prepared"
    # An empty proc directory is an external reader fixture for this local
    # contract test; live acceptance scans the actual Linux process tree.
    process_root=tmp_path/"proc";process_root.mkdir()
    retired=retire_fixture_chains(control_root=roots["control_plane"],scenes=[scene],
        source_commit=source,producer_pids=[],process_root=process_root)
    assert retired["status"] == "completed"
    assert retired["residual_pins"] == retired["residual_leases"] == 0
    assert retired["after_allocated_bytes"] < retired["before_allocated_bytes"]
    assert retired["automatic_terminal_reconciliation_proven"] is False
    assert retired["normal_six_hour_retention_proven"] is False
