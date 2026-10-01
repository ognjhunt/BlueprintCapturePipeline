"""ADP-009D: refuse misleading concurrency/storage acceptance reports."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/control_plane_concurrency_load_test.py"
SPEC = importlib.util.spec_from_file_location("concurrency_load_test", SCRIPT)
assert SPEC and SPEC.loader
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_p95_uses_nearest_rank_and_rejects_unmeasured_values():
    assert harness.p95(list(range(1, 21))) == 19
    for values in ([], [float("nan")], [-1], [True]):
        with pytest.raises(ValueError):
            harness.p95(values)


def test_allocation_sampler_observes_a_transient_peak_and_propagates_errors(tmp_path, monkeypatch):
    import time
    from scripts import control_plane_concurrency_load_test as module

    sampler = module.AllocationSampler(tmp_path, interval_seconds=0.01)
    with sampler:
        payload = tmp_path / "transient"
        payload.write_bytes(b"a" * 1024**2)
        deadline = time.monotonic() + 2
        while sampler.peak_bytes < 1024**2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sampler.peak_bytes >= 1024**2
        payload.unlink()
    assert sampler.peak_bytes > sampler.final_bytes
    assert sampler.sample_count >= 3
    calls = 0

    def unreadable(root):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("allocation_measurement_incomplete")
        return 0

    monkeypatch.setattr(module, "allocated_tree_bytes", unreadable)
    with pytest.raises(ValueError, match="allocation_measurement_incomplete"):
        with module.AllocationSampler(tmp_path, interval_seconds=0.01):
            time.sleep(0.05)


def test_allocated_measurement_counts_hardlinks_once_and_ignores_external_symlink(tmp_path):
    root = tmp_path / "host"
    root.mkdir()
    blob = root / "blob"
    blob.write_bytes(os.urandom(128 * 1024))
    before = harness.allocated_tree_bytes(root)
    os.link(blob, root / "alias")
    outside = tmp_path / "outside"
    outside.write_bytes(os.urandom(1024 * 1024))
    (root / "external").symlink_to(outside)
    assert harness.allocated_tree_bytes(root) - before < blob.stat().st_blocks * 512


def test_sampler_rescans_a_transient_incomplete_walk_but_records_it(tmp_path, monkeypatch):
    from scripts import control_plane_concurrency_load_test as module

    calls = 0

    def changing_tree(root):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("allocation_measurement_incomplete")
        return 1024

    monkeypatch.setattr(module, "allocated_tree_bytes", changing_tree)
    with module.AllocationSampler(tmp_path) as sampler:
        assert sampler.initial_bytes == 1024
    assert sampler.incomplete_scan_count == 1


def test_owned_run_roots_never_reuse_or_nest_fixture_storage(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("user-owned")
    with pytest.raises(ValueError, match="exists"):
        harness.create_run_roots(existing, tmp_path / "objects", tmp_path / "workers")
    with pytest.raises(ValueError, match="overlap"):
        harness.create_run_roots(tmp_path / "new", tmp_path / "new/objects", tmp_path / "workers")
    assert (existing / "keep").read_text() == "user-owned"
    assert not (tmp_path / "new").exists()


def completed_scene(index):
    return {
        "scene_id": f"scene-{index}",
        "stages": [
            {
                "stage": stage,
                "status": "completed",
                "wall_seconds": 0.1,
                "cpu_seconds": 0.01,
                "peak_allocated_bytes": 4096,
                "input_digest": "sha256:" + "a" * 64,
                "output_digest": "sha256:" + "b" * 64,
                "blockers": [],
            }
            for stage in harness.REQUIRED_STAGES
        ],
        "residual_leases": 0,
        "residual_pins": 0,
    }


def summary(scenes, **kwargs):
    return harness.build_summary(
        source_commit="1" * 40,
        expected_beta_concurrency=2,
        owner_confirmed=False,
        scenes=scenes,
        measured_peak_concurrency=4,
        concurrency_hold_seconds=0.1,
        baseline_allocated_bytes=4096,
        final_allocated_bytes=8192,
        maximum_retained_bytes=16384,
        maximum_delta_bytes=8192,
        external_calls=0,
        **kwargs,
    )


def test_partial_chain_and_missing_n_way_overlap_cannot_pass():
    scenes = [completed_scene(i) for i in range(4)]
    scenes[0]["stages"].pop()
    report = summary(scenes)
    assert report["status"] == "failed"
    assert "scene_chain_incomplete:scene-0" in report["blockers"]
    complete = [completed_scene(i) for i in range(4)]
    report = harness.build_summary(
        source_commit="1" * 40,
        expected_beta_concurrency=2,
        owner_confirmed=False,
        scenes=complete,
        measured_peak_concurrency=2,
        concurrency_hold_seconds=1,
        baseline_allocated_bytes=0,
        final_allocated_bytes=0,
        maximum_retained_bytes=0,
        maximum_delta_bytes=0,
        external_calls=0,
    )
    assert "target_concurrency_not_observed" in report["blockers"]


def test_final_delta_and_retained_evidence_are_independent_gates():
    scenes = [completed_scene(i) for i in range(4)]
    report = harness.build_summary(
        source_commit="1" * 40,
        expected_beta_concurrency=2,
        owner_confirmed=True,
        scenes=scenes,
        measured_peak_concurrency=4,
        concurrency_hold_seconds=0.1,
        baseline_allocated_bytes=0,
        final_allocated_bytes=4096,
        maximum_retained_bytes=8192,
        maximum_delta_bytes=1024,
        external_calls=0,
    )
    assert report["status"] == "failed"
    assert report["blockers"] == ["control_plane_disk_delta_exceeded"]


def test_provisional_success_does_not_close_owner_sized_gate():
    report = summary([completed_scene(i) for i in range(4)])
    assert report["status"] == "passed"
    assert report["owner_sized_acceptance_complete"] is False
    assert report["claim_ceiling"] == "development_only"


def test_capacity_failure_and_residual_lease_survive_summary():
    scenes = [completed_scene(i) for i in range(4)]
    scenes[0]["stages"][3].update(
        status="blocked", blockers=["control_plane_disk_budget_exceeded:launch_preparation"]
    )
    scenes[1]["residual_leases"] = 1
    report = summary(scenes)
    assert report["status"] == "failed"
    assert "control_plane_disk_budget_exceeded:launch_preparation" in report["blockers"]
    assert "residual_leases:scene-1" in report["blockers"]


def test_cli_requires_explicit_size_concurrency_and_separate_owned_roots(tmp_path):
    parser = harness.argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--expected-beta-concurrency",
            "2",
            "--maximum-retained-gib",
            "0.25",
            "--maximum-delta-gib",
            "0.25",
            "--control-plane-root",
            str(tmp_path / "host"),
            "--object-store-root",
            str(tmp_path / "objects"),
            "--worker-root",
            str(tmp_path / "workers"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert args.expected_beta_concurrency == 2
    assert args.owner_confirmed_concurrency is False
    for flag, value in (
        ("--expected-beta-concurrency", "0"),
        ("--maximum-retained-gib", "nan"),
        ("--child-timeout-seconds", "0"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--expected-beta-concurrency",
                    "2",
                    "--maximum-retained-gib",
                    "0.25",
                    "--maximum-delta-gib",
                    "0.25",
                    "--control-plane-root",
                    str(tmp_path / "host"),
                    "--object-store-root",
                    str(tmp_path / "objects"),
                    "--worker-root",
                    str(tmp_path / "workers"),
                    "--report",
                    str(tmp_path / "report.json"),
                    flag,
                    value,
                ]
            )


def test_child_environment_keeps_runtime_but_drops_inherited_provider_authority():
    env = harness.child_environment(
        {
            "PATH": "/bin",
            "PYTHONPATH": "src:.",
            "HOME": "/secret-home",
            "GOOGLE_APPLICATION_CREDENTIALS": "/secret",
            "AWS_SECRET_ACCESS_KEY": "never-copy",
            "BLUEPRINT_REMOTE_CPU_WORKERS_CONFIG": "/etc/blueprint/remote-cpu-workers.json",
            "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT": "/var/lib/blueprint/live",
            "OPENAI_API_KEY": "never-copy",
            "LD_PRELOAD": "/injected.so",
        }
    )
    assert env == {"PATH": "/bin", "PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"}


@pytest.mark.slow
def test_child_fence_blocks_network_provider_launch_and_external_writes(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    outside = tmp_path / "keep"
    outside.write_text("user-owned")
    program = """import json,pathlib,socket,subprocess,sys,os,shutil
from scripts.control_plane_concurrency_load_test import install_child_fences
root=pathlib.Path(sys.argv[1]); outside=pathlib.Path(sys.argv[2]); blocked=[]
install_child_fences([root])
for label, action in [
 ("network",lambda:socket.socket().connect(("127.0.0.1",9))),
 ("provider",lambda:subprocess.run(["curl","https://example.com"],check=True)),
 ("write",lambda:outside.write_text("wrong")),
 ("delete",lambda:outside.unlink())]:
 try:action()
 except PermissionError:blocked.append(label)
(root/"receipt.json").write_text(json.dumps(blocked))
owned=root/"temporary";owned.mkdir();(owned/"data").write_text("owned temporary")
shutil.rmtree(owned)
foreign=os.open(outside.parent,os.O_RDONLY)
try:
 try:os.unlink(outside.name,dir_fd=foreign)
 except PermissionError:blocked.append("foreign-directory-fd")
finally:os.close(foreign)
print(json.dumps(blocked))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(root), str(outside)],
        cwd=SCRIPT.parent.parent,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(SCRIPT.parent.parent / "src")
            + os.pathsep
            + str(SCRIPT.parent.parent),
        },
    )
    assert result.returncode == 0, result.stderr
    assert '"network", "provider", "write", "delete"' in result.stdout
    assert '"foreign-directory-fd"' in result.stdout
    assert outside.read_text() == "user-owned"
